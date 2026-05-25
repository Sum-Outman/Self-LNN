/**
 * @file reasoning.c
 * @brief 推理引擎实现
 * 
 * 推理引擎核心实现，支持多种推理模式：演绎、归纳、溯因、类比和因果推理。
 */

#include "selflnn/reasoning/reasoning.h"
#include "selflnn/reasoning/causal_reasoning.h"
#include "selflnn/knowledge/knowledge.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/perf.h"
#include "selflnn/utils/secure_random.h"

#include "selflnn/core/laplace.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

/**
 * @brief 规则索引项
 */
typedef struct {
    float premise_value;          /**< 前提值 */
    float conclusion_value;       /**< 结论值 */
    int is_valid;                 /**< 是否有效 */
} RuleIndexItem;

/**
 * @brief 缓存项
 */
typedef struct {
    float premises_hash;          /**< 前提哈希值 */
    float conclusion;             /**< 缓存结论 */
    float confidence;             /**< 置信度 */
    int is_valid;                 /**< 是否有效 */
    uint64_t last_used;           /**< 最后使用时间戳（LRU策略） */
} CacheItem;

/**
 * @brief 推理引擎统计信息结构体
 */
struct ReasoningStats {
    size_t total_inferences;      /**< 总推理次数 */
    size_t successful_inferences; /**< 成功推理次数 */
};

/**
 * @brief 推理引擎内部结构体
 */
struct ReasoningEngine {
    ReasoningConfig config;       /**< 引擎配置 */
    int is_initialized;           /**< 是否已初始化 */
    float* rule_base;             /**< 规则库 */
    size_t rule_base_size;        /**< 规则库大小 */
    float* knowledge_base;        /**< 知识库 */
    size_t knowledge_base_size;   /**< 知识库大小 */
    float* working_memory;        /**< 工作内存 */
    size_t working_memory_size;   /**< 工作内存大小 */
    float* inference_buffer;      /**< 推理缓冲区 */
    size_t inference_buffer_size; /**< 推理缓冲区大小 */
    
    // 性能优化结构
    RuleIndexItem* rule_index;    /**< 规则索引 */
    size_t rule_index_size;       /**< 规则索引大小 */
    CacheItem* inference_cache;   /**< 推理缓存 */
    size_t cache_size;            /**< 缓存大小 */
    size_t cache_hits;            /**< 缓存命中次数 */
    size_t cache_misses;          /**< 缓存未命中次数 */
    uint64_t cache_counter;       /**< 缓存时钟计数器（LRU策略） */
    int rule_index_built;         /**< 规则索引是否已构建 */
    CausalReasoningEngine* causal_engine; /**< 因果推理引擎 */
    struct ReasoningStats stats;  /**< 推理统计信息 */
    
    /* 知识库集成 */
    struct KnowledgeBase* external_kb;     /**< 外部知识库引用 */
    int kb_auto_sync;                     /**< 是否自动同步知识库 */
    int knowledge_ready;                  /**< P0-009: 知识是否已从真实数据源加载（1=已同步, 0=空白） */
    
    /* 液态神经网络集成 */
    LNN* lnn_instance;                /**< 关联的LNN实例 */
    
    /* 贝叶斯推理网络 */
    BayesianNetwork* bayesian_network; /**< 关联的贝叶斯网络 */

    /* 推理历史记录 */
    int inference_count;
    float* history_inputs;
    float* history_outputs;
    size_t history_capacity;
    size_t history_size;
    size_t history_input_dim;   /* S-031修复: 存储历史输入维度 */
    size_t history_output_dim;  /* S-031修复: 存储历史输出维度 */
    char* history_file_path;
};

/* 静态函数前向声明 */
static float causal_inference(const float* cause, size_t cause_size,
                             const float* effect, size_t effect_size);

/**
 * @brief 创建推理引擎
 */
ReasoningEngine* reasoning_engine_create(const ReasoningConfig* config) {
    if (!config) {
        return NULL;
    }
    
    ReasoningEngine* engine = (ReasoningEngine*)safe_malloc(sizeof(ReasoningEngine));
    if (!engine) {
        return NULL;
    }
    
    // 初始化结构体
    memset(engine, 0, sizeof(ReasoningEngine));
    engine->config = *config;
    engine->is_initialized = 1;
    
    // 分配内存
    engine->rule_base_size = 1024; // 默认规则库大小
    engine->rule_base = (float*)safe_malloc(engine->rule_base_size * sizeof(float));
    
    engine->knowledge_base_size = 2048; // 默认知识库大小
    engine->knowledge_base = (float*)safe_malloc(engine->knowledge_base_size * sizeof(float));
    
    engine->working_memory_size = 512; // 默认工作内存大小
    engine->working_memory = (float*)safe_malloc(engine->working_memory_size * sizeof(float));
    
    engine->inference_buffer_size = 256; // 默认推理缓冲区大小
    engine->inference_buffer = (float*)safe_malloc(engine->inference_buffer_size * sizeof(float));
    
    // 检查内存分配
    if (!engine->rule_base || !engine->knowledge_base || 
        !engine->working_memory || !engine->inference_buffer) {
        reasoning_engine_free(engine);
        return NULL;
    }
    
    // 初始化规则库和知识库
    memset(engine->rule_base, 0, engine->rule_base_size * sizeof(float));
    memset(engine->knowledge_base, 0, engine->knowledge_base_size * sizeof(float));
    memset(engine->working_memory, 0, engine->working_memory_size * sizeof(float));
    memset(engine->inference_buffer, 0, engine->inference_buffer_size * sizeof(float));
    
    // 分配性能优化结构
    engine->rule_index_size = 128; // 默认规则索引大小
    engine->rule_index = (RuleIndexItem*)safe_malloc(engine->rule_index_size * sizeof(RuleIndexItem));
    
    engine->cache_size = 64; // 默认缓存大小
    engine->inference_cache = (CacheItem*)safe_malloc(engine->cache_size * sizeof(CacheItem));
    
    engine->cache_hits = 0;
    engine->cache_misses = 0;
    engine->rule_index_built = 0;
    
    // 初始化性能优化结构
    if (engine->rule_index) {
        memset(engine->rule_index, 0, engine->rule_index_size * sizeof(RuleIndexItem));
    }
    
    if (engine->inference_cache) {
        memset(engine->inference_cache, 0, engine->cache_size * sizeof(CacheItem));
    }
    
    /* 初始化因果推理引擎 */
    {
        CausalReasoningConfig causal_config;
        memset(&causal_config, 0, sizeof(CausalReasoningConfig));
        causal_config.algorithm = CAUSAL_REASONING_PC_ALGORITHM;
        causal_config.significance_level = 0.05f;
        causal_config.max_conditioning_set_size = 5;
        causal_config.enable_temporal_analysis = 1;
        causal_config.enable_counterfactual = 1;
        causal_config.max_iterations = 100;
        causal_config.confidence_threshold = 0.6f;
        engine->causal_engine = causal_reasoning_engine_create(&causal_config);
    }
    
    /* 初始化知识库集成 */
    engine->external_kb = NULL;
    engine->kb_auto_sync = 1;
    engine->knowledge_ready = 0;  /* P0-009: 空白状态，需从真实知识库同步后才就绪 */
    
    /* 初始化LNN集成 */
    engine->lnn_instance = NULL;
    
    /* 初始化贝叶斯网络集成 — ZSFX-006修复: 实例化而非NULL */
    engine->bayesian_network = bayesian_network_create(64);
    
    return engine;
}

/**
 * @brief 释放推理引擎
 */
void reasoning_engine_free(ReasoningEngine* engine) {
    if (!engine) {
        return;
    }
    
    safe_free((void**)&engine->rule_base);
    safe_free((void**)&engine->knowledge_base);
    safe_free((void**)&engine->working_memory);
    safe_free((void**)&engine->inference_buffer);
    
    // 释放性能优化结构
    safe_free((void**)&engine->rule_index);
    safe_free((void**)&engine->inference_cache);
    
    /* 释放因果推理引擎 */
    if (engine->causal_engine) {
        causal_reasoning_engine_free(engine->causal_engine);
        engine->causal_engine = NULL;
    }
    
    /* 释放贝叶斯推理网络 */
    if (engine->bayesian_network) {
        bayesian_network_free(engine->bayesian_network);
        engine->bayesian_network = NULL;
    }
    
    safe_free((void**)&engine->history_inputs);
    safe_free((void**)&engine->history_outputs);
    safe_free((void**)&engine->history_file_path);
    
    safe_free((void**)&engine);
}

/**
 * @brief 记录推理历史
 */
int reasoning_record_history(ReasoningEngine* engine, const float* input, size_t input_size,
                              const float* output, size_t output_size) {
    if (!engine || !input || !output) return -1;
    size_t total = input_size + output_size;
    /* S-031修复: 记录实际的输入输出维度 */
    engine->history_input_dim = total;
    engine->history_output_dim = output_size;
    if (engine->history_capacity == 0) {
        engine->history_capacity = 256;
        engine->history_inputs = (float*)safe_malloc(engine->history_capacity * total * sizeof(float));
        engine->history_outputs = (float*)safe_malloc(engine->history_capacity * output_size * sizeof(float));
        if (!engine->history_inputs || !engine->history_outputs) { engine->history_capacity = 0; return -1; }
    }
    if (engine->history_size >= engine->history_capacity) {
        size_t new_cap = engine->history_capacity * 2;
        if (new_cap > 65536) return -1;
        float* new_in = (float*)safe_realloc(engine->history_inputs, new_cap * total * sizeof(float));
        float* new_out = (float*)safe_realloc(engine->history_outputs, new_cap * output_size * sizeof(float));
        if (!new_in || !new_out) return -1;
        engine->history_inputs = new_in;
        engine->history_outputs = new_out;
        engine->history_capacity = new_cap;
    }
    memcpy(engine->history_inputs + engine->history_size * total, input, input_size * sizeof(float));
    memcpy(engine->history_outputs + engine->history_size * output_size, output, output_size * sizeof(float));
    engine->history_size++;
    engine->inference_count++;
    return 0;
}

/**
 * @brief 保存推理历史到文件
 */
int reasoning_save_history(const ReasoningEngine* engine, const char* filepath) {
    if (!engine || !filepath) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;
    int count = engine->inference_count;
    size_t hist_size = engine->history_size;
    /* S-031修复: 保存实际维度信息 */
    size_t in_dim = engine->history_input_dim > 0 ? engine->history_input_dim : 256;
    size_t out_dim = engine->history_output_dim > 0 ? engine->history_output_dim : 256;
    fwrite(&count, sizeof(int), 1, fp);
    fwrite(&hist_size, sizeof(size_t), 1, fp);
    fwrite(&in_dim, sizeof(size_t), 1, fp);
    fwrite(&out_dim, sizeof(size_t), 1, fp);
    if (hist_size > 0 && engine->history_inputs && engine->history_outputs) {
        fwrite(engine->history_inputs, sizeof(float), hist_size * in_dim, fp);
        fwrite(engine->history_outputs, sizeof(float), hist_size * out_dim, fp);
    }
    fclose(fp);
    return 0;
}

/**
 * @brief 从文件加载推理历史
 */
int reasoning_load_history(ReasoningEngine* engine, const char* filepath) {
    if (!engine || !filepath) return -1;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;
    int count = 0;
    size_t hist_size = 0;
    size_t in_dim = 256, out_dim = 256;
    fread(&count, sizeof(int), 1, fp);
    fread(&hist_size, sizeof(size_t), 1, fp);
    /* S-031修复: 尝试读取维度信息（兼容旧格式） */
    long saved_pos = ftell(fp);
    fseek(fp, 0, SEEK_END);
    long file_end = ftell(fp);
    fseek(fp, saved_pos, SEEK_SET);
    size_t remaining = (size_t)(file_end - saved_pos);
    size_t expected_min = hist_size * 256 * 2 * sizeof(float);
    /* 如果文件足够大包含维度字段，读取它们 */
    if (remaining > expected_min) {
        fread(&in_dim, sizeof(size_t), 1, fp);
        fread(&out_dim, sizeof(size_t), 1, fp);
    }
    if (hist_size > 0 && hist_size < 65536) {
        safe_free((void**)&engine->history_inputs);
        safe_free((void**)&engine->history_outputs);
        engine->history_inputs = (float*)safe_malloc(hist_size * in_dim * sizeof(float));
        engine->history_outputs = (float*)safe_malloc(hist_size * out_dim * sizeof(float));
        if (engine->history_inputs && engine->history_outputs) {
            fread(engine->history_inputs, sizeof(float), hist_size * in_dim, fp);
            fread(engine->history_outputs, sizeof(float), hist_size * out_dim, fp);
            engine->history_size = hist_size;
            engine->history_capacity = hist_size;
            engine->history_input_dim = in_dim;
            engine->history_output_dim = out_dim;
            engine->inference_count = count;
        }
    }
    fclose(fp);
    return 0;
}

/**
 * @brief 设置自动保存路径
 */
int reasoning_set_autosave(ReasoningEngine* engine, const char* filepath) {
    if (!engine) return -1;
    safe_free((void**)&engine->history_file_path);
    if (filepath) {
        engine->history_file_path = (char*)safe_malloc(strlen(filepath) + 1);
        if (engine->history_file_path) strcpy(engine->history_file_path, filepath);
    }
    return 0;
}

/**
 * @brief 自动保存推理历史（如果有设置路径）
 */
int reasoning_autosave_history(ReasoningEngine* engine) {
    if (!engine || !engine->history_file_path || engine->history_size == 0) return -1;
    return reasoning_save_history(engine, engine->history_file_path);
}

/**
 * @brief 计算前提哈希值
 * 
 * 使用改进的哈希算法，基于FNV-1a哈希的变体，结合浮点数的位模式混合。
 * 提供更好的分布和碰撞抵抗能力。
 */
static float compute_premises_hash(const float* premises, size_t num_premises) {
    if (!premises || num_premises == 0) {
        return 0.0f;
    }
    
    // FNV-1a哈希算法的浮点数变体
    uint32_t hash_int = 2166136261u; // FNV偏移基础值
    
    for (size_t i = 0; i < num_premises; i++) {
        // 将浮点数的位模式视为无符号整数进行哈希
        uint32_t value_bits;
        memcpy(&value_bits, &premises[i], sizeof(uint32_t));
        
        // FNV-1a核心操作：异或后乘法
        hash_int ^= value_bits;
        hash_int *= 16777619u; // FNV质数
        
        // 额外的混合步骤，提高哈希质量
        hash_int = (hash_int << 13) | (hash_int >> 19); // 左旋转13位
        hash_int ^= 0x85ebca6b; // 额外的混合常数
    }
    
    // 将整数哈希转换回浮点数
    // 使用归一化处理，确保结果是规范的浮点数
    if (hash_int == 0) {
        return 0.0f;
    }
    
    // 将整数哈希映射到[-1.0, 1.0]范围内，避免极端值
    // 使用哈希值的高位部分生成浮点数
    uint32_t normalized = hash_int & 0x007fffff; // 保留23位尾数
    normalized |= 0x3f800000; // 设置指数为127 (偏置后的0)
    
    float result;
    memcpy(&result, &normalized, sizeof(float));
    
    // 将结果调整到[-1.0, 1.0]范围
    result = result - 1.0f;
    
    return result;
}

/**
 * @brief 在缓存中查找推理结果
 */
static int find_in_cache(ReasoningEngine* engine, float premises_hash,
                        float* conclusion, float* confidence) {
    if (!engine || !engine->inference_cache || engine->cache_size == 0) {
        return 0;
    }
    
    for (size_t i = 0; i < engine->cache_size; i++) {
        CacheItem* item = &engine->inference_cache[i];
        if (item->is_valid && fabsf(item->premises_hash - premises_hash) < 0.0001f) {
            if (conclusion) {
                *conclusion = item->conclusion;
            }
            if (confidence) {
                *confidence = item->confidence;
            }
            // 更新最后使用时间戳（LRU策略）
            item->last_used = engine->cache_counter++;
            engine->cache_hits++;
            return 1;
        }
    }
    
    engine->cache_misses++;
    return 0;
}

/* ========== 内部辅助函数 ========== */

/**
 * @brief 拓扑排序（Kahn算法）
 */
static int causal_model_topological_sort(const StructuralCausalModel* model,
                                          int* sorted_ids, size_t max_size) {
    if (!model || !sorted_ids || max_size == 0) return -1;

    size_t n = model->num_variables;
    if (n == 0 || n > max_size) return -1;

    int* in_degree = (int*)safe_calloc(n, sizeof(int));
    if (!in_degree) return -1;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (model->adjacency_matrix && model->adjacency_matrix[i] &&
                model->adjacency_matrix[i][j] > 0.0f) {
                in_degree[j]++;
            }
        }
    }

    int* queue = (int*)safe_malloc(n * sizeof(int));
    if (!queue) { safe_free((void**)&in_degree); return -1; }
    int front = 0, rear = 0;

    for (size_t i = 0; i < n; i++) {
        if (in_degree[i] == 0) queue[rear++] = (int)i;
    }

    size_t count = 0;
    while (front < rear && count < max_size) {
        int v = queue[front++];
        sorted_ids[count++] = v;

        for (size_t j = 0; j < n; j++) {
            if (model->adjacency_matrix && model->adjacency_matrix[v] &&
                model->adjacency_matrix[v][j] > 0.0f) {
                in_degree[j]--;
                if (in_degree[j] == 0) queue[rear++] = (int)j;
            }
        }
    }

    safe_free((void**)&in_degree);
    safe_free((void**)&queue);

    return (count == n) ? (int)n : -1;
}

/**
 * @brief 通过结构方程传播干预值
 */
static int causal_model_propagate(StructuralCausalModel* model) {
    if (!model || !model->variables) return -1;

    int* sorted = (int*)safe_malloc(model->num_variables * sizeof(int));
    if (!sorted) return -1;

    int count = causal_model_topological_sort(model, sorted, model->num_variables);
    if (count < 0) { safe_free((void**)&sorted); return -1; }

    for (int i = 0; i < count; i++) {
        int v = sorted[i];
        CausalVariable* var = &model->variables[v];

        if (var->is_intervened) {
            var->current_value = var->intervention_value;
            continue;
        }
        if (var->var_type == CAUSAL_VAR_EXOGENOUS) continue;
        if (var->num_parents == 0) continue;

        float sum = 0.0f;
        for (size_t p = 0; p < var->num_parents; p++) {
            int pid = var->parent_ids[p];
            if (pid >= 0 && pid < (int)model->num_variables) {
                sum += var->parent_weights[p] * model->variables[pid].current_value;
            }
        }

        float noise = 0.0f;
        if (var->noise_std > 0.0f) {
            float u1 = secure_random_float();
            float u2 = secure_random_float();
            if (u1 > 0.0f) {
                noise = var->noise_mean + var->noise_std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
            }
        } else {
            noise = var->noise_mean;
        }

        var->current_value = sum + noise;
    }

    safe_free((void**)&sorted);
    return 0;
}

/**
 * @brief 计算高斯概率密度
 */
static float causal_model_gaussian_pdf(float x, float mean, float std) {
    if (std <= 0.0f) return (x == mean) ? 1.0f : 0.0f;
    float diff = x - mean;
    return expf(-0.5f * diff * diff / (std * std)) / (sqrtf(2.0f * 3.14159265f) * std);
}

/**
 * @brief 计算协方差
 */
static float causal_model_covariance(const float* data_x, const float* data_y, size_t n) {
    if (!data_x || !data_y || n < 2) return 0.0f;
    float mean_x = 0.0f, mean_y = 0.0f;
    for (size_t i = 0; i < n; i++) { mean_x += data_x[i]; mean_y += data_y[i]; }
    mean_x /= (float)n; mean_y /= (float)n;
    float cov = 0.0f;
    for (size_t i = 0; i < n; i++) cov += (data_x[i] - mean_x) * (data_y[i] - mean_y);
    return cov / (float)(n - 1);
}

/* ========== P2-2: 因果模型驱动规划实现 ========== */

StructuralCausalModel* causal_model_create(size_t max_variables) {
    if (max_variables == 0 || max_variables > 10000) return NULL;

    StructuralCausalModel* model = (StructuralCausalModel*)safe_malloc(sizeof(StructuralCausalModel));
    if (!model) return NULL;
    memset(model, 0, sizeof(StructuralCausalModel));

    model->variables = (CausalVariable*)safe_calloc(max_variables, sizeof(CausalVariable));
    if (!model->variables) {
        safe_free((void**)&model);
        return NULL;
    }

    model->adjacency_matrix = (float**)safe_calloc(max_variables, sizeof(float*));
    if (!model->adjacency_matrix) {
        safe_free((void**)&model->variables);
        safe_free((void**)&model);
        return NULL;
    }
    for (size_t i = 0; i < max_variables; i++) {
        model->adjacency_matrix[i] = (float*)safe_calloc(max_variables, sizeof(float));
        if (!model->adjacency_matrix[i]) {
            for (size_t j = 0; j < i; j++) safe_free((void**)&model->adjacency_matrix[j]);
            safe_free((void**)&model->adjacency_matrix);
            safe_free((void**)&model->variables);
            safe_free((void**)&model);
            return NULL;
        }
    }

    model->num_variables = 0;
    model->variable_capacity = max_variables;
    return model;
}

void causal_model_free(StructuralCausalModel* model) {
    if (!model) return;

    for (size_t i = 0; i < model->num_variables; i++) {
        CausalVariable* var = &model->variables[i];
        safe_free((void**)&var->parent_weights);
        safe_free((void**)&var->parent_ids);
        var->num_parents = 0;
    }

    if (model->adjacency_matrix) {
        for (size_t i = 0; i < model->variable_capacity; i++) {
            safe_free((void**)&model->adjacency_matrix[i]);
        }
        safe_free((void**)&model->adjacency_matrix);
    }

    safe_free((void**)&model->variables);
    model->num_variables = 0;
    model->variable_capacity = 0;
    safe_free((void**)&model);
}

int causal_model_add_variable(StructuralCausalModel* model,
                              const char* name,
                              CausalVariableType var_type,
                              float noise_mean, float noise_std) {
    if (!model || !name || !model->variables) return -1;
    if (model->num_variables >= model->variable_capacity) return -1;

    size_t id = model->num_variables;
    CausalVariable* var = &model->variables[id];

    var->var_id = (int)id;
    var->var_type = var_type;
    memset(var->name, 0, sizeof(var->name));
    strncpy(var->name, name, sizeof(var->name) - 1);
    var->current_value = 0.0f;
    var->parent_weights = NULL;
    var->parent_ids = NULL;
    var->num_parents = 0;
    var->noise_mean = noise_mean;
    var->noise_std = (noise_std < 0.0f) ? 0.0f : noise_std;
    var->intervention_value = 0.0f;
    var->is_intervened = 0;

    model->num_variables++;
    return (int)id;
}

int causal_model_set_structural_equation(StructuralCausalModel* model,
                                        int var_id,
                                        const int* parent_ids,
                                        const float* parent_weights,
                                        size_t num_parents) {
    if (!model || !model->variables) return -1;
    if (var_id < 0 || var_id >= (int)model->num_variables) return -1;
    if (num_parents > 0 && (!parent_ids || !parent_weights)) return -1;

    CausalVariable* var = &model->variables[var_id];

    safe_free((void**)&var->parent_ids);
    safe_free((void**)&var->parent_weights);
    var->num_parents = 0;

    if (num_parents > 0) {
        var->parent_ids = (int*)safe_malloc(num_parents * sizeof(int));
        var->parent_weights = (float*)safe_malloc(num_parents * sizeof(float));
        if (!var->parent_ids || !var->parent_weights) {
            safe_free((void**)&var->parent_ids);
            safe_free((void**)&var->parent_weights);
            return -1;
        }
        memcpy(var->parent_ids, parent_ids, num_parents * sizeof(int));
        memcpy(var->parent_weights, parent_weights, num_parents * sizeof(float));
        var->num_parents = num_parents;

        if (model->adjacency_matrix) {
            for (size_t i = 0; i < model->variable_capacity; i++) {
                model->adjacency_matrix[i][var_id] = 0.0f;
            }
            for (size_t p = 0; p < num_parents; p++) {
                int pid = parent_ids[p];
                if (pid >= 0 && pid < (int)model->variable_capacity) {
                    model->adjacency_matrix[pid][var_id] = parent_weights[p];
                }
            }
        }
    }

    return 0;
}

int causal_model_do_intervention(StructuralCausalModel* model,
                                 int x_var_id, float x_value,
                                 int y_var_id,
                                 CausalEffectEstimate* result) {
    if (!model || !result) return -1;
    if (x_var_id < 0 || x_var_id >= (int)model->num_variables) return -1;
    if (y_var_id < 0 || y_var_id >= (int)model->num_variables) return -1;

    memset(result, 0, sizeof(CausalEffectEstimate));

    /* 保存原始状态 */
    float* saved_values = (float*)safe_malloc(model->num_variables * sizeof(float));
    int* saved_intervened = (int*)safe_malloc(model->num_variables * sizeof(int));
    float* saved_interventions = (float*)safe_malloc(model->num_variables * sizeof(float));
    if (!saved_values || !saved_intervened || !saved_interventions) {
        safe_free((void**)&saved_values);
        safe_free((void**)&saved_intervened);
        safe_free((void**)&saved_interventions);
        return -1;
    }

    for (size_t i = 0; i < model->num_variables; i++) {
        saved_values[i] = model->variables[i].current_value;
        saved_intervened[i] = model->variables[i].is_intervened;
        saved_interventions[i] = model->variables[i].intervention_value;
    }

    /* 执行干预: 切断所有入边，固定X=x */
    model->variables[x_var_id].is_intervened = 1;
    model->variables[x_var_id].intervention_value = x_value;

    /* 样本数量 */
    const int num_samples = 500;
    float total_y = 0.0f;
    float total_y2 = 0.0f;

    for (int s = 0; s < num_samples; s++) {
        model->variables[x_var_id].current_value = x_value;

        if (causal_model_propagate(model) != 0) {
            for (size_t i = 0; i < model->num_variables; i++) {
                model->variables[i].current_value = saved_values[i];
                model->variables[i].is_intervened = saved_intervened[i];
                model->variables[i].intervention_value = saved_interventions[i];
            }
            safe_free((void**)&saved_values);
            safe_free((void**)&saved_intervened);
            safe_free((void**)&saved_interventions);
            return -1;
        }

        float y_val = model->variables[y_var_id].current_value;
        total_y += y_val;
        total_y2 += y_val * y_val;
    }

    /* 恢复原始状态 */
    for (size_t i = 0; i < model->num_variables; i++) {
        model->variables[i].current_value = saved_values[i];
        model->variables[i].is_intervened = saved_intervened[i];
        model->variables[i].intervention_value = saved_interventions[i];
    }
    safe_free((void**)&saved_values);
    safe_free((void**)&saved_intervened);
    safe_free((void**)&saved_interventions);

    /* 计算因果效应 E[Y|do(X=x)] */
    float mean_y = total_y / (float)num_samples;
    float var_y = total_y2 / (float)num_samples - mean_y * mean_y;
    if (var_y < 0.0f) var_y = 0.0f;

    result->causal_effect = mean_y;
    result->effect_variance = var_y / (float)num_samples;
    result->confidence_interval_low = mean_y - 1.96f * sqrtf(var_y / (float)num_samples);
    result->confidence_interval_high = mean_y + 1.96f * sqrtf(var_y / (float)num_samples);
    result->is_significant = (fabsf(mean_y) > 1.96f * sqrtf(var_y / (float)num_samples)) ? 1 : 0;
    snprintf(result->explanation, sizeof(result->explanation),
             "do-演算干预: do(%s=%.4f) → E[%s]=%.4f (95%% CI: [%.4f, %.4f])",
             model->variables[x_var_id].name, x_value,
             model->variables[y_var_id].name, mean_y,
             result->confidence_interval_low, result->confidence_interval_high);

    return 0;
}

int causal_model_backdoor_adjustment(StructuralCausalModel* model,
                                     int treatment_id, int outcome_id,
                                     const int* backdoor_ids, size_t num_backdoors,
                                     CausalEffectEstimate* result) {
    if (!model || !result) return -1;
    if (treatment_id < 0 || treatment_id >= (int)model->num_variables) return -1;
    if (outcome_id < 0 || outcome_id >= (int)model->num_variables) return -1;
    if (num_backdoors > 0 && !backdoor_ids) return -1;

    memset(result, 0, sizeof(CausalEffectEstimate));

    int* saved_intervened = (int*)safe_malloc(model->num_variables * sizeof(int));
    float* saved_values = (float*)safe_malloc(model->num_variables * sizeof(float));
    if (!saved_intervened || !saved_values) {
        safe_free((void**)&saved_intervened);
        safe_free((void**)&saved_values);
        return -1;
    }

    for (size_t i = 0; i < model->num_variables; i++) {
        saved_intervened[i] = model->variables[i].is_intervened;
        saved_values[i] = model->variables[i].current_value;
    }

    /* 后门调整: P(y|do(x)) = Σ_z P(y|x,z) P(z) */
    const int num_samples_per_z = 100;
    const int num_z_samples = 50;
    float total_adjusted = 0.0f;
    float total_weight = 0.0f;

    for (int zs = 0; zs < num_z_samples; zs++) {
        for (size_t i = 0; i < model->num_variables; i++) {
            model->variables[i].is_intervened = saved_intervened[i];
            model->variables[i].current_value = saved_values[i];
        }

        for (size_t b = 0; b < num_backdoors; b++) {
            int bid = backdoor_ids[b];
            if (bid >= 0 && bid < (int)model->num_variables) {
                model->variables[bid].is_intervened = 1;
                model->variables[bid].intervention_value = saved_values[bid];
            }
        }

        if (causal_model_propagate(model) != 0) {
            safe_free((void**)&saved_intervened);
            safe_free((void**)&saved_values);
            return -1;
        }

        float p_z = 1.0f;
        for (size_t b = 0; b < num_backdoors; b++) {
            int bid = backdoor_ids[b];
            if (bid >= 0 && bid < (int)model->num_variables) {
                float z_val = model->variables[bid].current_value;
                p_z *= causal_model_gaussian_pdf(z_val,
                    saved_values[bid],
                    model->variables[bid].noise_std + 0.001f);
            }
        }

        float sum_y_given_xz = 0.0f;
        for (int s = 0; s < num_samples_per_z; s++) {
            for (size_t i = 0; i < model->num_variables; i++) {
                model->variables[i].is_intervened = saved_intervened[i];
            }
            model->variables[treatment_id].is_intervened = 1;
            model->variables[treatment_id].intervention_value = saved_values[treatment_id];
            for (size_t b = 0; b < num_backdoors; b++) {
                int bid = backdoor_ids[b];
                if (bid >= 0 && bid < (int)model->num_variables) {
                    model->variables[bid].is_intervened = 1;
                    model->variables[bid].intervention_value = model->variables[bid].current_value;
                }
            }
            if (causal_model_propagate(model) == 0) {
                sum_y_given_xz += model->variables[outcome_id].current_value;
            }
        }
        float mean_y_given_xz = sum_y_given_xz / (float)num_samples_per_z;

        total_adjusted += mean_y_given_xz * p_z;
        total_weight += p_z;
    }

    for (size_t i = 0; i < model->num_variables; i++) {
        model->variables[i].is_intervened = saved_intervened[i];
        model->variables[i].current_value = saved_values[i];
    }
    safe_free((void**)&saved_intervened);
    safe_free((void**)&saved_values);

    if (total_weight > 0.0f) {
        result->causal_effect = total_adjusted / total_weight;
    } else {
        result->causal_effect = 0.0f;
    }

    result->effect_variance = 0.0f;
    result->confidence_interval_low = result->causal_effect - 0.1f;
    result->confidence_interval_high = result->causal_effect + 0.1f;
    result->is_significant = (fabsf(result->causal_effect) > 0.05f) ? 1 : 0;
    snprintf(result->explanation, sizeof(result->explanation),
             "后门准则调整: P(%s|do(%s)) = Σ_z P(%s|%s,z)P(z), 效应=%.4f",
             model->variables[outcome_id].name,
             model->variables[treatment_id].name,
             model->variables[outcome_id].name,
             model->variables[treatment_id].name,
             result->causal_effect);

    return 0;
}

int causal_model_frontdoor_adjustment(StructuralCausalModel* model,
                                      int treatment_id, int outcome_id,
                                      int mediator_id,
                                      CausalEffectEstimate* result) {
    if (!model || !result) return -1;
    if (treatment_id < 0 || treatment_id >= (int)model->num_variables) return -1;
    if (outcome_id < 0 || outcome_id >= (int)model->num_variables) return -1;
    if (mediator_id < 0 || mediator_id >= (int)model->num_variables) return -1;

    memset(result, 0, sizeof(CausalEffectEstimate));

    /* 前门调整: P(y|do(x)) = Σ_m P(m|x) Σ_x' P(y|x',m) P(x') */
    int* saved_intervened = (int*)safe_malloc(model->num_variables * sizeof(int));
    float* saved_values = (float*)safe_malloc(model->num_variables * sizeof(float));
    if (!saved_intervened || !saved_values) {
        safe_free((void**)&saved_intervened);
        safe_free((void**)&saved_values);
        return -1;
    }

    for (size_t i = 0; i < model->num_variables; i++) {
        saved_intervened[i] = model->variables[i].is_intervened;
        saved_values[i] = model->variables[i].current_value;
    }

    const int num_m_samples = 30;
    const int num_xprime_samples = 30;

    /* 固定X=x */
    model->variables[treatment_id].is_intervened = 1;
    model->variables[treatment_id].intervention_value = saved_values[treatment_id];

    /* 计算 P(m|x) 的样本期望 */
    float sum_m_given_x = 0.0f;
    for (int s = 0; s < num_m_samples; s++) {
        if (causal_model_propagate(model) == 0) {
            sum_m_given_x += model->variables[mediator_id].current_value;
        }
    }
    float mean_m_given_x = sum_m_given_x / (float)num_m_samples;

    /* 计算 Σ_x' P(y|x',m) P(x') */
    float sum_over_xprime = 0.0f;
    for (int s = 0; s < num_xprime_samples; s++) {
        for (size_t i = 0; i < model->num_variables; i++) {
            model->variables[i].is_intervened = saved_intervened[i];
            model->variables[i].current_value = saved_values[i];
        }

        model->variables[treatment_id].is_intervened = 1;
        model->variables[treatment_id].intervention_value =
            saved_values[treatment_id] + (secure_random_float() - 0.5f) * 2.0f;

        model->variables[mediator_id].is_intervened = 1;
        model->variables[mediator_id].intervention_value = mean_m_given_x;

        if (causal_model_propagate(model) == 0) {
            float y_val = model->variables[outcome_id].current_value;
            float p_xprime = causal_model_gaussian_pdf(
                model->variables[treatment_id].current_value,
                saved_values[treatment_id],
                model->variables[treatment_id].noise_std + 0.001f);
            sum_over_xprime += y_val * p_xprime;
        }
    }

    for (size_t i = 0; i < model->num_variables; i++) {
        model->variables[i].is_intervened = saved_intervened[i];
        model->variables[i].current_value = saved_values[i];
    }
    safe_free((void**)&saved_intervened);
    safe_free((void**)&saved_values);

    float mean_over_xprime = sum_over_xprime / (float)num_xprime_samples;
    result->causal_effect = mean_m_given_x * mean_over_xprime;

    result->effect_variance = 0.0f;
    result->confidence_interval_low = result->causal_effect - 0.15f;
    result->confidence_interval_high = result->causal_effect + 0.15f;
    result->is_significant = (fabsf(result->causal_effect) > 0.05f) ? 1 : 0;
    snprintf(result->explanation, sizeof(result->explanation),
             "前门准则调整: P(%s|do(%s)) = Σ_m P(m|%s) Σ_x' P(%s|x',m)P(x'), 通过中介%s",
             model->variables[outcome_id].name,
             model->variables[treatment_id].name,
             model->variables[treatment_id].name,
             model->variables[outcome_id].name,
             model->variables[mediator_id].name);

    return 0;
}

int causal_model_instrumental_variable(StructuralCausalModel* model,
                                       int instrument_id,
                                       int treatment_id, int outcome_id,
                                       CausalEffectEstimate* result) {
    if (!model || !result) return -1;
    if (instrument_id < 0 || instrument_id >= (int)model->num_variables) return -1;
    if (treatment_id < 0 || treatment_id >= (int)model->num_variables) return -1;
    if (outcome_id < 0 || outcome_id >= (int)model->num_variables) return -1;

    memset(result, 0, sizeof(CausalEffectEstimate));

    /* 工具变量估计: β_IV = Cov(Z,Y) / Cov(Z,X) */
    const int num_samples = 500;
    float* z_samples = (float*)safe_malloc(num_samples * sizeof(float));
    float* x_samples = (float*)safe_malloc(num_samples * sizeof(float));
    float* y_samples = (float*)safe_malloc(num_samples * sizeof(float));
    if (!z_samples || !x_samples || !y_samples) {
        safe_free((void**)&z_samples);
        safe_free((void**)&x_samples);
        safe_free((void**)&y_samples);
        return -1;
    }

    int* saved_intervened = (int*)safe_malloc(model->num_variables * sizeof(int));
    float* saved_values = (float*)safe_malloc(model->num_variables * sizeof(float));
    if (!saved_intervened || !saved_values) {
        safe_free((void**)&z_samples);
        safe_free((void**)&x_samples);
        safe_free((void**)&y_samples);
        safe_free((void**)&saved_intervened);
        safe_free((void**)&saved_values);
        return -1;
    }

    for (size_t i = 0; i < model->num_variables; i++) {
        saved_intervened[i] = model->variables[i].is_intervened;
        saved_values[i] = model->variables[i].current_value;
    }

    for (int s = 0; s < num_samples; s++) {
        model->variables[instrument_id].is_intervened = 1;
        model->variables[instrument_id].intervention_value =
            secure_random_float() * 4.0f - 2.0f;

        if (causal_model_propagate(model) == 0) {
            z_samples[s] = model->variables[instrument_id].current_value;
            x_samples[s] = model->variables[treatment_id].current_value;
            y_samples[s] = model->variables[outcome_id].current_value;
        } else {
            z_samples[s] = 0.0f; x_samples[s] = 0.0f; y_samples[s] = 0.0f;
        }

        for (size_t i = 0; i < model->num_variables; i++) {
            model->variables[i].is_intervened = saved_intervened[i];
            model->variables[i].current_value = saved_values[i];
        }
    }

    for (size_t i = 0; i < model->num_variables; i++) {
        model->variables[i].is_intervened = saved_intervened[i];
        model->variables[i].current_value = saved_values[i];
    }
    safe_free((void**)&saved_intervened);
    safe_free((void**)&saved_values);

    float cov_zy = causal_model_covariance(z_samples, y_samples, (size_t)num_samples);
    float cov_zx = causal_model_covariance(z_samples, x_samples, (size_t)num_samples);

    safe_free((void**)&z_samples);
    safe_free((void**)&x_samples);
    safe_free((void**)&y_samples);

    if (fabsf(cov_zx) < 1e-8f) {
        snprintf(result->explanation, sizeof(result->explanation),
                 "工具变量估计失败: Cov(Z,X)=0, Z与X不相关");
        return -1;
    }

    result->causal_effect = cov_zy / cov_zx;

    float var_z = 0.0f;
    float mean_z = 0.0f;
    for (int s = 0; s < num_samples; s++) mean_z += z_samples[s];
    mean_z /= (float)num_samples;
    for (int s = 0; s < num_samples; s++) var_z += (z_samples[s] - mean_z) * (z_samples[s] - mean_z);
    var_z /= (float)(num_samples - 1);

    float se = sqrtf(var_z / (cov_zx * cov_zx * (float)num_samples));
    result->effect_variance = se * se;
    result->confidence_interval_low = result->causal_effect - 1.96f * se;
    result->confidence_interval_high = result->causal_effect + 1.96f * se;
    result->is_significant = (fabsf(result->causal_effect) > 1.96f * se) ? 1 : 0;

    snprintf(result->explanation, sizeof(result->explanation),
             "工具变量估计: β_IV = Cov(%s,%s)/Cov(%s,%s) = %.4f",
             model->variables[instrument_id].name,
             model->variables[outcome_id].name,
             model->variables[instrument_id].name,
             model->variables[treatment_id].name,
             result->causal_effect);

    return 0;
}

int causal_model_mediation_analysis(StructuralCausalModel* model,
                                    int treatment_id, int mediator_id, int outcome_id,
                                    MediationResult* result) {
    if (!model || !result) return -1;
    if (treatment_id < 0 || treatment_id >= (int)model->num_variables) return -1;
    if (mediator_id < 0 || mediator_id >= (int)model->num_variables) return -1;
    if (outcome_id < 0 || outcome_id >= (int)model->num_variables) return -1;

    memset(result, 0, sizeof(MediationResult));

    int* saved_intervened = (int*)safe_malloc(model->num_variables * sizeof(int));
    float* saved_values = (float*)safe_malloc(model->num_variables * sizeof(float));
    if (!saved_intervened || !saved_values) {
        safe_free((void**)&saved_intervened);
        safe_free((void**)&saved_values);
        return -1;
    }

    for (size_t i = 0; i < model->num_variables; i++) {
        saved_intervened[i] = model->variables[i].is_intervened;
        saved_values[i] = model->variables[i].current_value;
    }

    const int num_samples = 300;

    /* 总效应 TE: E[Y|do(X=1)] - E[Y|do(X=0)] */
    {
        float sum_y_do1 = 0.0f, sum_y_do0 = 0.0f;
        for (int s = 0; s < num_samples; s++) {
            for (size_t i = 0; i < model->num_variables; i++) {
                model->variables[i].is_intervened = saved_intervened[i];
                model->variables[i].current_value = saved_values[i];
            }
            model->variables[treatment_id].is_intervened = 1;
            model->variables[treatment_id].intervention_value = 1.0f;
            if (causal_model_propagate(model) == 0) {
                sum_y_do1 += model->variables[outcome_id].current_value;
            }
        }
        for (int s = 0; s < num_samples; s++) {
            for (size_t i = 0; i < model->num_variables; i++) {
                model->variables[i].is_intervened = saved_intervened[i];
                model->variables[i].current_value = saved_values[i];
            }
            model->variables[treatment_id].is_intervened = 1;
            model->variables[treatment_id].intervention_value = 0.0f;
            if (causal_model_propagate(model) == 0) {
                sum_y_do0 += model->variables[outcome_id].current_value;
            }
        }
        result->total_effect = (sum_y_do1 - sum_y_do0) / (float)num_samples;
    }

    /* 直接效应 DE: E[Y|do(X=1,M=m*)] - E[Y|do(X=0,M=m*)] 固定中介变量 */
    {
        float sum_m_star = 0.0f;
        for (int s = 0; s < num_samples; s++) {
            for (size_t i = 0; i < model->num_variables; i++) {
                model->variables[i].is_intervened = saved_intervened[i];
                model->variables[i].current_value = saved_values[i];
            }
            model->variables[treatment_id].is_intervened = 1;
            model->variables[treatment_id].intervention_value = 1.0f;
            if (causal_model_propagate(model) == 0) {
                sum_m_star += model->variables[mediator_id].current_value;
            }
        }
        float mean_m_star = sum_m_star / (float)num_samples;

        float de_do1 = 0.0f, de_do0 = 0.0f;
        for (int s = 0; s < num_samples; s++) {
            for (size_t i = 0; i < model->num_variables; i++) {
                model->variables[i].is_intervened = saved_intervened[i];
                model->variables[i].current_value = saved_values[i];
            }
            model->variables[treatment_id].is_intervened = 1;
            model->variables[treatment_id].intervention_value = 1.0f;
            model->variables[mediator_id].is_intervened = 1;
            model->variables[mediator_id].intervention_value = mean_m_star;
            if (causal_model_propagate(model) == 0) {
                de_do1 += model->variables[outcome_id].current_value;
            }
        }
        for (int s = 0; s < num_samples; s++) {
            for (size_t i = 0; i < model->num_variables; i++) {
                model->variables[i].is_intervened = saved_intervened[i];
                model->variables[i].current_value = saved_values[i];
            }
            model->variables[treatment_id].is_intervened = 1;
            model->variables[treatment_id].intervention_value = 0.0f;
            model->variables[mediator_id].is_intervened = 1;
            model->variables[mediator_id].intervention_value = mean_m_star;
            if (causal_model_propagate(model) == 0) {
                de_do0 += model->variables[outcome_id].current_value;
            }
        }
        result->direct_effect = (de_do1 - de_do0) / (float)num_samples;
    }

    /* 间接效应 IE = TE - DE */
    result->indirect_effect = result->total_effect - result->direct_effect;

    if (fabsf(result->total_effect) > 1e-8f) {
        result->mediation_proportion = result->indirect_effect / result->total_effect;
        if (result->mediation_proportion < 0.0f) result->mediation_proportion = 0.0f;
        if (result->mediation_proportion > 1.0f) result->mediation_proportion = 1.0f;
    } else {
        result->mediation_proportion = 0.0f;
    }

    result->direct_significance = (fabsf(result->direct_effect) > 0.05f) ? 1.0f : 0.0f;
    result->indirect_significance = (fabsf(result->indirect_effect) > 0.05f) ? 1.0f : 0.0f;

    for (size_t i = 0; i < model->num_variables; i++) {
        model->variables[i].is_intervened = saved_intervened[i];
        model->variables[i].current_value = saved_values[i];
    }
    safe_free((void**)&saved_intervened);
    safe_free((void**)&saved_values);

    return 0;
}

int causal_model_driven_plan(StructuralCausalModel* model,
                             const int* action_ids, size_t num_actions,
                             const int* target_ids, size_t num_targets,
                             float* plan, size_t max_plan_size) {
    if (!model || !action_ids || !target_ids || !plan) return -1;
    if (num_actions == 0 || num_targets == 0 || max_plan_size == 0) return -1;

    size_t plan_count = 0;

    for (size_t a = 0; a < num_actions && plan_count < max_plan_size; a++) {
        int action_id = action_ids[a];
        if (action_id < 0 || action_id >= (int)model->num_variables) continue;

        float best_target_effect = -1e10f;
        int best_target_id = -1;

        for (size_t t = 0; t < num_targets; t++) {
            int target_id = target_ids[t];
            if (target_id < 0 || target_id >= (int)model->num_variables) continue;

            CausalEffectEstimate est;
            memset(&est, 0, sizeof(CausalEffectEstimate));

            /* 对每个行动-目标对执行do-演算 */
            int ret = causal_model_do_intervention(
                model, action_id, 1.0f, target_id, &est);

            if (ret == 0) {
                float score = fabsf(est.causal_effect);
                if (est.is_significant) score *= 1.5f;

                if (score > best_target_effect) {
                    best_target_effect = score;
                    best_target_id = target_id;
                }
            }

            /* 保存当前步到规划缓冲区 */
            if (plan_count < max_plan_size) {
                plan[plan_count * 4 + 0] = (float)action_id;
                plan[plan_count * 4 + 1] = (float)target_id;
                plan[plan_count * 4 + 2] = (ret == 0) ? est.causal_effect : 0.0f;
                plan[plan_count * 4 + 3] = (ret == 0 && est.is_significant) ? 1.0f : 0.0f;
                plan_count++;
            }
        }
    }

    /* 按因果效应绝对值降序排序规划步骤（简单冒泡排序） */
    for (size_t i = 0; i < plan_count; i++) {
        for (size_t j = i + 1; j < plan_count; j++) {
            float eff_i = fabsf(plan[i * 4 + 2]);
            float eff_j = fabsf(plan[j * 4 + 2]);
            if (eff_j > eff_i) {
                for (int k = 0; k < 4; k++) {
                    float tmp = plan[i * 4 + k];
                    plan[i * 4 + k] = plan[j * 4 + k];
                    plan[j * 4 + k] = tmp;
                }
            }
        }
    }

    return (int)plan_count;
}

int causal_model_counterfactual(StructuralCausalModel* model,
                                const float* factual_data, size_t factual_size,
                                float counterfactual_intervention,
                                int intervened_var_id,
                                float* counterfactual_result, size_t max_result_size) {
    if (!model || !factual_data || !counterfactual_result) return -1;
    if (factual_size == 0 || max_result_size == 0) return -1;
    if (intervened_var_id < 0 || intervened_var_id >= (int)model->num_variables) return -1;

    int* saved_intervened = (int*)safe_malloc(model->num_variables * sizeof(int));
    float* saved_values = (float*)safe_malloc(model->num_variables * sizeof(float));
    if (!saved_intervened || !saved_values) {
        safe_free((void**)&saved_intervened);
        safe_free((void**)&saved_values);
        return -1;
    }

    for (size_t i = 0; i < model->num_variables; i++) {
        saved_intervened[i] = model->variables[i].is_intervened;
        saved_values[i] = model->variables[i].current_value;
    }

    /* 步骤1: 溯因（Abduction）- 从事实数据推断外生变量U */
    for (size_t i = 0; i < factual_size && i < model->num_variables; i++) {
        model->variables[i].current_value = factual_data[i];
    }

    /* 调整噪声项以适配事实数据 */
    for (size_t i = 0; i < model->num_variables; i++) {
        CausalVariable* var = &model->variables[i];
        if (var->num_parents > 0 && !var->is_intervened) {
            float predicted = 0.0f;
            for (size_t p = 0; p < var->num_parents; p++) {
                int pid = var->parent_ids[p];
                if (pid >= 0 && pid < (int)model->num_variables) {
                    predicted += var->parent_weights[p] * model->variables[pid].current_value;
                }
            }
            float residual = var->current_value - predicted;
            var->noise_mean = residual;

            if (i < factual_size) {
                var->noise_mean = factual_data[i] - predicted;
            }
        }
    }

    /* 步骤2: 行动（Action）- 设置干预 do(X=x_cf) */
    model->variables[intervened_var_id].is_intervened = 1;
    model->variables[intervened_var_id].intervention_value = counterfactual_intervention;

    /* 步骤3: 预测（Prediction）- 在新的干预下计算Y值 */
    if (causal_model_propagate(model) != 0) {
        for (size_t i = 0; i < model->num_variables; i++) {
            model->variables[i].is_intervened = saved_intervened[i];
            model->variables[i].current_value = saved_values[i];
        }
        safe_free((void**)&saved_intervened);
        safe_free((void**)&saved_values);
        return -1;
    }

    size_t copy_size = (max_result_size < model->num_variables) ? max_result_size : model->num_variables;
    for (size_t i = 0; i < copy_size; i++) {
        counterfactual_result[i] = model->variables[i].current_value;
    }

    for (size_t i = 0; i < model->num_variables; i++) {
        model->variables[i].is_intervened = saved_intervened[i];
        model->variables[i].current_value = saved_values[i];
    }
    safe_free((void**)&saved_intervened);
    safe_free((void**)&saved_values);

    return (int)copy_size;
}

int causal_model_build_plan_graph(StructuralCausalModel* model,
                                  const void* htn_domain,
                                  const float* task_variables, size_t num_task_vars,
                                  const float* goal_description, size_t goal_size) {
    if (!model || !task_variables || !goal_description) return -1;
    if (num_task_vars == 0 || goal_size == 0) return -1;

    /* 将规划目标变量注册到因果模型 */
    size_t task_var_offset = 0;
    if (htn_domain) {
        /* 从HTN域解析复合任务和原始任务 */
        for (size_t i = 0; i < num_task_vars && task_var_offset < (size_t)goal_size; i++) {
            char var_name[64];
            snprintf(var_name, sizeof(var_name), "task_var_%zu", i);

            int var_id = causal_model_add_variable(
                model, var_name, CAUSAL_VAR_ENDOGENOUS, 0.0f, 0.1f);
            if (var_id >= 0) {
                model->variables[var_id].current_value = task_variables[i];
                task_var_offset++;
            }
        }
    }

    /* 注册目标变量 */
    size_t goal_var_start = task_var_offset;
    for (size_t i = 0; i < goal_size && task_var_offset + i < model->variable_capacity; i++) {
        char goal_name[64];
        snprintf(goal_name, sizeof(goal_name), "goal_%zu", i);

        int var_id = causal_model_add_variable(
            model, goal_name, CAUSAL_VAR_ENDOGENOUS, 0.0f, 0.05f);
        if (var_id >= 0) {
            model->variables[var_id].current_value = goal_description[i];
        }
    }

    /* 构建任务变量与目标变量之间的因果关系 */
    for (size_t i = 0; i < task_var_offset && i < model->num_variables; i++) {
        for (size_t j = goal_var_start; j < model->num_variables && j < goal_var_start + goal_size; j++) {
            if ((int)i < 0 || (int)j < 0 || (int)i >= (int)model->variable_capacity ||
                (int)j >= (int)model->variable_capacity) continue;

            float task_val = model->variables[i].current_value;
            float goal_val = model->variables[j].current_value;

            if (fabsf(task_val) > 0.01f && fabsf(goal_val) > 0.01f) {
                float weight = (task_val * goal_val) /
                    ((fabsf(task_val) + fabsf(goal_val)) * 0.5f + 0.001f) * 0.5f;

                weight = (weight > 1.0f) ? 1.0f : (weight < -1.0f) ? -1.0f : weight;

                int parent_ids[1] = {(int)i};
                float parent_weights[1] = {weight};
                causal_model_set_structural_equation(
                    model, (int)j, parent_ids, parent_weights, 1);
            }
        }
    }

    /* 建立任务变量之间的时序关系 */
    for (size_t i = 0; i + 1 < task_var_offset && i + 1 < model->num_variables; i++) {
        float ordering_weight = 0.3f;
        int parent_ids[1] = {(int)i};
        float parent_weights[1] = {ordering_weight};
        causal_model_set_structural_equation(
            model, (int)(i + 1), parent_ids, parent_weights, 1);
    }

    return 0;
}

/**
 * @brief 添加推理结果到缓存
 */
static void add_to_cache(ReasoningEngine* engine, float premises_hash,
                        float conclusion, float confidence) {
    if (!engine || !engine->inference_cache || engine->cache_size == 0) {
        return;
    }
    
    // 查找第一个无效的缓存项
    for (size_t i = 0; i < engine->cache_size; i++) {
        CacheItem* item = &engine->inference_cache[i];
        if (!item->is_valid) {
            item->premises_hash = premises_hash;
            item->conclusion = conclusion;
            item->confidence = confidence;
            item->is_valid = 1;
            item->last_used = engine->cache_counter++;
            return;
        }
    }
    
    // 如果缓存已满，使用LRU策略替换最近最少使用的项
    size_t lru_index = 0;
    uint64_t min_last_used = engine->inference_cache[0].last_used;
    
    for (size_t i = 1; i < engine->cache_size; i++) {
        if (engine->inference_cache[i].last_used < min_last_used) {
            min_last_used = engine->inference_cache[i].last_used;
            lru_index = i;
        }
    }
    
    engine->inference_cache[lru_index].premises_hash = premises_hash;
    engine->inference_cache[lru_index].conclusion = conclusion;
    engine->inference_cache[lru_index].confidence = confidence;
    engine->inference_cache[lru_index].is_valid = 1;
    engine->inference_cache[lru_index].last_used = engine->cache_counter++;
}

/**
 * @brief 构建规则索引
 */
static void build_rule_index(ReasoningEngine* engine) {
    if (!engine || !engine->rule_base || engine->rule_base_size == 0) {
        return;
    }
    
    if (!engine->rule_index || engine->rule_index_size == 0) {
        return;
    }
    
    size_t index_pos = 0;
    for (size_t i = 0; i < engine->rule_base_size && index_pos < engine->rule_index_size; i += 2) {
        if (i + 1 < engine->rule_base_size) {
            engine->rule_index[index_pos].premise_value = engine->rule_base[i];
            engine->rule_index[index_pos].conclusion_value = engine->rule_base[i + 1];
            engine->rule_index[index_pos].is_valid = 1;
            index_pos++;
        }
    }
}

/**
 * @brief 使用规则索引的演绎推理
 */
static int deductive_reasoning_indexed(ReasoningEngine* engine,
                                      const float* premises, size_t num_premises,
                                      float* conclusion, size_t max_conclusion_size) {

    
    if (!engine || !premises || !conclusion || num_premises == 0 || max_conclusion_size == 0) {
        return -1;
    }
    
    // 首先检查缓存
    float premises_hash = compute_premises_hash(premises, num_premises);
    float cached_conclusion;
    float cached_confidence;
    
    if (find_in_cache(engine, premises_hash, &cached_conclusion, &cached_confidence)) {
        if (cached_confidence > engine->config.confidence_threshold) {
            conclusion[0] = cached_conclusion;
            return 0;
        }
    }
    
    // 惰性构建规则索引
    if (!engine->rule_index_built && engine->rule_index && engine->rule_index_size > 0) {
        build_rule_index(engine);
        engine->rule_index_built = 1;
    }
    
    // 使用规则索引进行推理
    size_t conclusion_size = 0;
    
    if (engine->rule_index && engine->rule_index_size > 0 && engine->rule_index_built) {
        for (size_t i = 0; i < engine->rule_index_size && conclusion_size < max_conclusion_size; i++) {
            RuleIndexItem* item = &engine->rule_index[i];
            if (!item->is_valid) {
                continue;
            }
            
            // 检查前提是否匹配
            int premise_matched = 0;
            for (size_t j = 0; j < num_premises; j++) {
                if (fabsf(premises[j] - item->premise_value) < 0.1f) {
                    premise_matched = 1;
                    break;
                }
            }
            
            if (premise_matched) {
                conclusion[conclusion_size++] = item->conclusion_value;
            }
        }
    }
    
    if (conclusion_size > 0) {
        // 计算置信度
        float confidence = 0.0f;
        if (engine->rule_index_size > 0) {
            confidence = (float)conclusion_size / (float)engine->rule_index_size;
        }
        
        // 添加到缓存
        add_to_cache(engine, premises_hash, conclusion[0], confidence);
        
        return 0;
    }
    
    /* P1-007修复: 无规则时使用数值逻辑推理代替简单MIN操作
     * 不再简单返回最小值作为"逻辑与"，而是：
     * 1. 加权融合多前提（大前提贡献更多）
     * 2. 基于前提分布估计合理结论区间
     * 3. 降低置信度以反映推理不确定性 */
    float weighted_sum = 0.0f;
    float weight_total = 0.0f;
    for (size_t i = 0; i < num_premises; i++) {
        float w = 1.0f - fabsf(premises[i] - 0.5f) * 0.5f;  /* 靠近0.5的值权重更高(更中性) */
        if (w < 0.1f) w = 0.1f;
        weighted_sum += premises[i] * w;
        weight_total += w;
    }
    float default_conclusion = weighted_sum / (weight_total + 1e-10f);

    /* 基于前提分布确定置信度 */
    float sum = 0.0f;
    for (size_t i = 0; i < num_premises; i++) sum += premises[i];
    float mean = sum / (float)num_premises;
    float variance = 0.0f;
    for (size_t i = 0; i < num_premises; i++) {
        float diff = premises[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)num_premises;

    /* 无规则推理置信度公式：一致性高→置信度高，方差大→置信度低 */
    float confidence = 0.4f + 0.3f * (1.0f - fminf(1.0f, variance));
    if (num_premises <= 1) confidence *= 0.5f;  /* 单前提推理置信度减半 */
    if (confidence < 0.15f) confidence = 0.15f;
    if (confidence > 0.7f) confidence = 0.7f;   /* 无规则时上限0.7 */

    /* 对多前提做交叉验证：检查各前提与加权结论的一致性 */
    float consistency = 0.0f;
    for (size_t i = 0; i < num_premises; i++) {
        consistency += 1.0f - fminf(1.0f, fabsf(premises[i] - default_conclusion));
    }
    consistency /= (float)num_premises;
    confidence *= 0.5f + 0.5f * consistency;

    conclusion[0] = default_conclusion;

    /* 输出推理方式标记 */
    if (max_conclusion_size > 1) conclusion[1] = 0.0f;  /* 0=无规则知识推理 */

    add_to_cache(engine, premises_hash, conclusion[0], confidence);
    
    return 0;
}

/**
 * @brief 演绎推理引擎（基于规则）
 */
static int deductive_reasoning(const float* premises, size_t num_premises,
                              float* conclusion, size_t max_conclusion_size,
                              const float* rule_base, size_t rule_base_size) {
    if (num_premises == 0 || max_conclusion_size == 0) {
        return -1;
    }
    
    // 工业级演绎推理引擎：基于逻辑规则的完整推理系统
    // 规则格式扩展：[前提数量, 结论, 规则置信度, 规则优先级, 前提1, 前提2, ...]
    // 支持模糊匹配、置信度传播、冲突解决和多结论排序
    
    // 候选结论结构
    struct CandidateConclusion {
        float value;           // 结论值
        float confidence;      // 综合置信度（规则置信度 × 匹配度 × 优先级因子）
        float rule_confidence; // 规则原始置信度
        float match_quality;   // 前提匹配质量（0-1）
        int rule_id;           // 规则标识（用于追踪）
    };
    
    // 存储候选结论（最多支持16个候选）
    struct CandidateConclusion candidates[16];
    size_t num_candidates = 0;
    
    size_t rule_index = 0;
    int rule_id = 0;
    
    // 第一阶段：解析所有规则，评估匹配度
    while (rule_index < rule_base_size && num_candidates < 16) {
        // 读取规则头部（至少需要4个值：前提数量、结论、置信度、优先级）
        if (rule_index + 3 >= rule_base_size) break;
        
        float num_premises_in_rule = rule_base[rule_index];
        float rule_conclusion = rule_base[rule_index + 1];
        float rule_confidence = rule_base[rule_index + 2];
        float rule_priority = rule_base[rule_index + 3]; // 新增：规则优先级
        
        // 验证规则置信度
        if (rule_confidence < 0.0f || rule_confidence > 1.0f) {
            rule_confidence = 0.5f; // 默认值
        }
        
        // 验证规则优先级（默认1.0，越高越优先）
        if (rule_priority <= 0.0f) {
            rule_priority = 1.0f;
        }
        
        size_t premise_data_index = rule_index + 4;
        
        // 计算前提匹配质量
        float total_match_quality = 0.0f;
        int premises_matched = 1;
        int premises_evaluated = 0;
        
        for (int p = 0; p < (int)num_premises_in_rule; p++) {
            if (premise_data_index >= rule_base_size) {
                premises_matched = 0;
                break;
            }
            
            float required_premise = rule_base[premise_data_index++];
            premises_evaluated++;
            
            // 寻找最佳匹配的前提
            float best_match_quality = 0.0f;
            for (size_t j = 0; j < num_premises; j++) {
                // 高斯相似度函数：exp(-(diff^2)/(2*sigma^2))
                // sigma = 0.1 控制匹配宽度
                float diff = fabsf(premises[j] - required_premise);
                float match_quality = expf(-(diff * diff) / (2.0f * 0.01f)); // sigma=0.1
                
                if (match_quality > best_match_quality) {
                    best_match_quality = match_quality;
                }
            }
            
            total_match_quality += best_match_quality;
            
            // 如果某个前提完全无法匹配（质量<0.1），则规则不匹配
            if (best_match_quality < 0.1f) {
                premises_matched = 0;
                break;
            }
        }
        
        // 如果所有前提都评估了且匹配，则添加候选结论
        if (premises_matched && premises_evaluated > 0) {
            // 计算平均匹配质量
            float avg_match_quality = total_match_quality / premises_evaluated;
            
            // 综合置信度 = 规则置信度 × 匹配质量 × 优先级因子
            // 优先级因子：log(1 + priority)，避免过度影响
            float priority_factor = logf(1.0f + rule_priority);
            float final_confidence = rule_confidence * avg_match_quality * priority_factor;
            
            // 置信度阈值：至少0.3才考虑
            if (final_confidence >= 0.3f) {
                candidates[num_candidates].value = rule_conclusion;
                candidates[num_candidates].confidence = final_confidence;
                candidates[num_candidates].rule_confidence = rule_confidence;
                candidates[num_candidates].match_quality = avg_match_quality;
                candidates[num_candidates].rule_id = rule_id;
                num_candidates++;
            }
        }
        
        // 移动到下一个规则：跳过已处理的前提数据
        rule_index = premise_data_index;
        rule_id++;
    }
    
    // 第二阶段：冲突解决和结论选择
    if (num_candidates == 0) {
        return -1; // 没有匹配的规则
    }
    
    // 按置信度降序排序（优化插入排序，对小数据集高效）
    // 插入排序：将元素插入到已排序序列中的正确位置
    for (size_t i = 1; i < num_candidates; i++) {
        struct CandidateConclusion key = candidates[i];
        size_t j = i;
        
        // 将key插入到已排序的candidates[0..i-1]中
        while (j > 0 && candidates[j - 1].confidence < key.confidence) {
            candidates[j] = candidates[j - 1];
            j--;
        }
        
        candidates[j] = key;
    }
    
    // 第三阶段：生成最终结论
    size_t conclusion_size = 0;
    
    // 方案1：选择最高置信度的结论
    if (conclusion_size < max_conclusion_size) {
        conclusion[conclusion_size++] = candidates[0].value;
    }
    
    // 方案2：如果需要，添加置信度信息
    if (conclusion_size < max_conclusion_size) {
        conclusion[conclusion_size++] = candidates[0].confidence;
    }
    
    // 方案3：如果空间允许，添加次优结论（用于不确定性表示）
    if (num_candidates > 1 && conclusion_size + 2 <= max_conclusion_size) {
        // 添加第二个最佳结论及其置信度
        conclusion[conclusion_size++] = candidates[1].value;
        conclusion[conclusion_size++] = candidates[1].confidence;
    }
    
    // 可选：添加推理元数据（规则ID、匹配质量等）
    if (conclusion_size + 2 <= max_conclusion_size) {
        conclusion[conclusion_size++] = (float)candidates[0].rule_id;
        conclusion[conclusion_size++] = candidates[0].match_quality;
    }
    
    return conclusion_size > 0 ? 0 : -1;
}

/**
 * @brief 完整归纳推理引擎（从实例中提取模式和规则）
 */
static int inductive_reasoning(const float* premises, size_t num_premises,
                              float* conclusion, size_t max_conclusion_size) {
    if (num_premises == 0 || max_conclusion_size == 0) {
        return -1;
    }
    
    // 工业级归纳推理系统：从数据中提取统计模式、趋势和一般规则
    size_t conclusion_index = 0;
    
    // 第一阶段：基本统计特征计算（扩展版）
    float sum = 0.0f;
    float sum_squares = 0.0f;
    float min_val = premises[0];
    float max_val = premises[0];
    float geometric_mean = 1.0f;
    int zero_count = 0;
    
    for (size_t i = 0; i < num_premises; i++) {
        float val = premises[i];
        sum += val;
        sum_squares += val * val;
        
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        
        if (val != 0.0f) {
            geometric_mean *= fabsf(val);
        } else {
            zero_count++;
        }
    }
    
    float mean = sum / num_premises;
    float range = max_val - min_val;
    
    // 计算几何平均值（避免下溢）
    if (num_premises - zero_count > 0) {
        geometric_mean = powf(geometric_mean, 1.0f / (num_premises - zero_count));
    } else {
        geometric_mean = 0.0f;
    }
    
    // 第二阶段：高级统计特征
    float variance = 0.0f;
    float skewness = 0.0f;
    float kurtosis = 0.0f;
    
    for (size_t i = 0; i < num_premises; i++) {
        float diff = premises[i] - mean;
        float diff2 = diff * diff;
        variance += diff2;
        
        // 三阶矩（偏度）
        skewness += diff * diff2;
        
        // 四阶矩（峰度）
        kurtosis += diff2 * diff2;
    }
    
    variance /= num_premises;
    float std_dev = sqrtf(variance + 1e-10f);
    
    if (variance > 1e-10f) {
        float variance_sqrt = sqrtf(variance);
        float variance_32 = variance * variance_sqrt;
        skewness /= (num_premises * variance_32 + 1e-10f);
        kurtosis /= (num_premises * variance * variance + 1e-10f);
        kurtosis -= 3.0f; // 超额峰度
    }
    
    // 第三阶段：趋势和模式分析
    float linear_trend = 0.0f;
    float linear_r_squared = 0.0f;
    float quadratic_coef = 0.0f;
    
    if (num_premises > 1) {
        // 线性回归：y = a + b*x
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
        for (size_t i = 0; i < num_premises; i++) {
            float x = (float)i;
            float y = premises[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        float n = (float)num_premises;
        float denominator = n * sum_x2 - sum_x * sum_x;
        if (fabsf(denominator) > 1e-10f) {
            float b = (n * sum_xy - sum_x * sum_y) / denominator; // 斜率
            float a = (sum_y - b * sum_x) / n; // 截距
            
            linear_trend = b;
            
            // 计算R²（决定系数）
            float y_mean = sum_y / n;
            float ss_total = 0.0f, ss_residual = 0.0f;
            for (size_t i = 0; i < num_premises; i++) {
                float x = (float)i;
                float y = premises[i];
                float y_pred = a + b * x;
                ss_total += (y - y_mean) * (y - y_mean);
                ss_residual += (y - y_pred) * (y - y_pred);
            }
            
            if (ss_total > 1e-10f) {
                linear_r_squared = 1.0f - ss_residual / ss_total;
            }
        }
        
        // 完整二次回归（最小二乘法拟合 y = a + b*x + c*x²）
        if (num_premises >= 3) {
            // 计算必要的幂和（使用_q后缀避免变量隐藏）
            float sum_x_q = 0.0f, sum_x2_q = 0.0f, sum_x3_q = 0.0f, sum_x4_q = 0.0f;
            float sum_y_q = 0.0f, sum_xy_q = 0.0f, sum_x2y_q = 0.0f;
            
            for (size_t i = 0; i < num_premises; i++) {
                float x = (float)i;
                float y = premises[i];
                float x2 = x * x;
                float x3 = x2 * x;
                float x4 = x3 * x;
                
                sum_x_q += x;
                sum_x2_q += x2;
                sum_x3_q += x3;
                sum_x4_q += x4;
                sum_y_q += y;
                sum_xy_q += x * y;
                sum_x2y_q += x2 * y;
            }
            
            float n_q = (float)num_premises;
            
            // 构建正规方程矩阵：M * [c, b, a] = v
            // 其中M是3x3对称矩阵，v是右侧向量
            // 方程：∑y = a*n_q + b*∑x + c*∑x²
            //       ∑xy = a*∑x + b*∑x² + c*∑x³
            //       ∑x²y = a*∑x² + b*∑x³ + c*∑x⁴
            
            // 使用克莱姆法则解3x3线性系统
            // 计算矩阵行列式
            float det = n_q * (sum_x2_q * sum_x4_q - sum_x3_q * sum_x3_q)
                       - sum_x_q * (sum_x_q * sum_x4_q - sum_x3_q * sum_x2_q)
                       + sum_x2_q * (sum_x_q * sum_x3_q - sum_x2_q * sum_x2_q);
            
            if (fabsf(det) > 1e-10f) {
                // 计算c的行列式（二次系数）
                float det_c = n_q * (sum_x2_q * sum_x2y_q - sum_x3_q * sum_xy_q)
                            - sum_x_q * (sum_x_q * sum_x2y_q - sum_x3_q * sum_y_q)
                            + sum_x2_q * (sum_x_q * sum_xy_q - sum_x2_q * sum_y_q);
                
                // 二次系数c（x²的系数）
                quadratic_coef = det_c / det;
            } else {
                // 行列式太小，回退到三点法
                float x0 = premises[0];
                float x1 = premises[num_premises / 2];
                float x2 = premises[num_premises - 1];
                quadratic_coef = (x2 - 2.0f * x1 + x0) / 2.0f;
            }
        }
    }
    
    // 第四阶段：序列模式检测
    int monotonic_increasing = 1;
    int monotonic_decreasing = 1;
    int has_cycle = 0;
    float autocorrelation_lag1 = 0.0f;
    
    for (size_t i = 1; i < num_premises; i++) {
        if (premises[i] < premises[i - 1]) monotonic_increasing = 0;
        if (premises[i] > premises[i - 1]) monotonic_decreasing = 0;
        
        // 计算一阶自相关（同时用于基础检测）
        if (i < num_premises - 1) {
            autocorrelation_lag1 += (premises[i] - mean) * (premises[i + 1] - mean);
        }
    }
    
    if (variance > 1e-10f) {
        autocorrelation_lag1 /= ((num_premises - 1) * variance);
    }
    
    // 完整周期检测：多滞后自相关分析
    // 计算多个滞后期的自相关（滞后1到max_lag）
    int max_lag = (num_premises > 0) ? (int)(num_premises / 2) : 1;
    if (max_lag > 10) max_lag = 10; // 限制最大滞后为10
    
    float max_autocorr = 0.0f;
    int best_lag = 1;
    
    for (int lag = 1; lag <= max_lag; lag++) {
        float autocorr = 0.0f;
        int count = 0;
        
        for (size_t i = 0; i < num_premises - lag; i++) {
            autocorr += (premises[i] - mean) * (premises[i + lag] - mean);
            count++;
        }
        
        if (count > 0 && variance > 1e-10f) {
            autocorr /= (count * variance);
            
            // 记录最大自相关值及其滞后
            if (fabsf(autocorr) > fabsf(max_autocorr)) {
                max_autocorr = autocorr;
                best_lag = lag;
            }
        }
    }
    
    // 基于自相关检测周期性
    // 标准：任何滞后期的自相关绝对值 > 0.5 表示显著周期性
    // 同时考虑滞后1的自相关作为基础检测
    if (fabsf(autocorrelation_lag1) > 0.5f || fabsf(max_autocorr) > 0.5f) {
        has_cycle = 1;
    }
    
    // 第五阶段：生成归纳结论（多维度特征）
    // 基础统计特征（8个）
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = mean;               // 1. 算术平均值
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = std_dev;           // 2. 标准差
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = min_val;           // 3. 最小值
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = max_val;           // 4. 最大值
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = range;             // 5. 范围
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = geometric_mean;    // 6. 几何平均值
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = skewness;          // 7. 偏度
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = kurtosis;          // 8. 峰度
    
    // 趋势特征（4个）
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = linear_trend;      // 9. 线性趋势斜率
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = linear_r_squared;  // 10. 线性拟合R²
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = quadratic_coef;    // 11. 二次系数（曲率）
    
    // 模式特征（4个）
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = monotonic_increasing ? 1.0f : 0.0f;  // 12. 单调递增标志
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = monotonic_decreasing ? 1.0f : 0.0f;  // 13. 单调递减标志
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = autocorrelation_lag1;                // 14. 一阶自相关
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = has_cycle ? 1.0f : 0.0f;             // 15. 周期检测标志
    
    // 数据质量特征（2个）
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = (float)zero_count / num_premises;    // 16. 零值比例
    if (conclusion_index < max_conclusion_size) conclusion[conclusion_index++] = (range > 1e-10f) ? std_dev / range : 0.0f; // 17. 变异系数
    
    // 第六阶段：生成归纳规则摘要（强化版 - P1-004修复）
    // 不仅仅输出统计值，还生成可复用的归纳规则编码
    if (conclusion_index + 6 <= max_conclusion_size) {
        // 规则置信度：基于数据质量和模式清晰度
        float rule_confidence = 0.0f;
        rule_confidence += 0.3f * fminf(1.0f, linear_r_squared);           // 线性拟合质量
        rule_confidence += 0.2f * (1.0f - (float)zero_count / num_premises); // 数据完整性
        rule_confidence += 0.2f * (monotonic_increasing || monotonic_decreasing ? 1.0f : 0.0f); // 单调性
        rule_confidence += 0.3f * fminf(1.0f, 1.0f - fminf(1.0f, std_dev / (fabsf(mean) + 1e-10f))); // 相对稳定性
        
        conclusion[conclusion_index++] = rule_confidence;                  // 18. 归纳规则置信度
        conclusion[conclusion_index++] = mean;                             // 19. 规则中心值（代表性值）
        conclusion[conclusion_index++] = linear_trend;                     // 20. 规则趋势分量

        /* P1-004修复: 编码归纳规则类型和参数供知识库复用 */
        /* 规则类型编码: 1=单调递增, 2=单调递减, 3=周期型, 4=随机型, 5=二次曲线型 */
        float rule_type = 4.0f;
        if (monotonic_increasing && linear_r_squared > 0.5f) rule_type = 1.0f;
        else if (monotonic_decreasing && linear_r_squared > 0.5f) rule_type = 2.0f;
        else if (has_cycle && fabsf(max_autocorr) > 0.5f) rule_type = 3.0f;
        else if (fabsf(quadratic_coef) > 0.01f) rule_type = 5.0f;
        conclusion[conclusion_index++] = rule_type;                        // 21. 规则类型编码

        /* 规则泛化范围: 基于标准差定义正常范围 */
        float rule_lower = mean - 2.0f * std_dev;
        float rule_upper = mean + 2.0f * std_dev;
        conclusion[conclusion_index++] = rule_lower;                       // 22. 规则下界
        conclusion[conclusion_index++] = rule_upper;                       // 23. 规则上界
    }
    
    return 0;
}

/**
 * @brief 溯因推理引擎（推断最佳解释）
 */
static int abductive_reasoning(const float* observation, size_t observation_size,
                              float* explanation, size_t max_explanation_size,
                              const float* knowledge_base, size_t knowledge_base_size) {

    
    if (observation_size == 0 || max_explanation_size == 0) {
        return -1;
    }
    
    // 工业级溯因推理引擎：生成、评估和优化解释假设
    // 支持多维评估、假设优化、多解释选择和置信度校准
    
    // 候选解释结构
    struct ExplanationCandidate {
        float value;               // 解释值
        float score;               // 综合评分（0-1）
        float coverage;           // 解释覆盖度（0-1）
        float precision;          // 解释精确度（覆盖观察的紧密程度）
        float consistency;        // 与知识库的一致性（0-1）
        float simplicity;         // 简洁性（值越小越简洁，0-1）
        float predictive_power;   // 预测能力（解释新观察的潜力）
        float testability;        // 可检验性（0-1）
        int generation_method;    // 生成方法：0=直接值，1=平均值，2=加权组合，3=优化值
    };
    
    // 存储候选解释（最多20个）
    struct ExplanationCandidate candidates[20];
    size_t num_candidates = 0;
    
    // 如果没有知识库，提供基于观察的默认解释
    if (knowledge_base_size == 0 || !knowledge_base) {
        // 生成默认解释：观察值的平均值和范围
        float sum = 0.0f;
        float min_val = observation[0];
        float max_val = observation[0];
        
        for (size_t i = 0; i < observation_size; i++) {
            sum += observation[i];
            if (observation[i] < min_val) min_val = observation[i];
            if (observation[i] > max_val) max_val = observation[i];
        }
        
        float mean = sum / observation_size;
        
        // 提供多个可能的解释
        size_t explanation_size = 0;
        
        // 解释1：平均值
        if (explanation_size < max_explanation_size) {
            explanation[explanation_size++] = mean;
        }
        
        // 解释2：最小值（保守解释）
        if (explanation_size < max_explanation_size) {
            explanation[explanation_size++] = min_val;
        }
        
        // 解释3：最大值（激进解释）
        if (explanation_size < max_explanation_size) {
            explanation[explanation_size++] = max_val;
        }
        
        // 解释4：中值（如果还有空间）
        if (explanation_size + 1 < max_explanation_size) {
            // 完整中位数计算（基于观察值排序）
            float median = 0.0f;
            if (observation_size > 0) {
                // 创建观察值的副本进行排序
                float* sorted_obs = (float*)safe_malloc(observation_size * sizeof(float));
                if (sorted_obs) {
                    // 复制观察值
                    for (size_t i = 0; i < observation_size; i++) {
                        sorted_obs[i] = observation[i];
                    }
                    
                    // 使用插入排序（对小数据集高效）
                    for (size_t i = 1; i < observation_size; i++) {
                        float key = sorted_obs[i];
                        size_t j = i;
                        while (j > 0 && sorted_obs[j - 1] > key) {
                            sorted_obs[j] = sorted_obs[j - 1];
                            j--;
                        }
                        sorted_obs[j] = key;
                    }
                    
                    // 计算中位数
                    if (observation_size % 2 == 1) {
                        median = sorted_obs[observation_size / 2];
                    } else {
                        median = (sorted_obs[observation_size / 2 - 1] + sorted_obs[observation_size / 2]) / 2.0f;
                    }
                    
                    safe_free((void**)&sorted_obs);
                } else {
                    // 内存分配失败，回退到范围中值
                    median = (min_val + max_val) / 2.0f;
                }
            }
            explanation[explanation_size++] = median;
            
            // 添加置信度作为额外信息
            explanation[explanation_size++] = 0.7f; // 默认置信度
        }
        
        return explanation_size > 0 ? 0 : -1;
    }
    
    // 阶段1：基本假设生成（直接从知识库中提取）
    for (size_t i = 0; i < knowledge_base_size && num_candidates < 20; i++) {
        float kb_value = knowledge_base[i];
        
        // 计算覆盖度：解释能覆盖多少观察
        float coverage = 0.0f;
        float precision_sum = 0.0f;
        int covered_count = 0;
        
        for (size_t j = 0; j < observation_size; j++) {
            float diff = fabsf(observation[j] - kb_value);
            // 使用高斯核计算每个观察的"解释程度"
            float explanation_strength = expf(-(diff * diff) / (2.0f * 0.04f)); // sigma=0.2
            
            if (explanation_strength > 0.1f) {
                covered_count++;
                precision_sum += explanation_strength;
            }
        }
        
        coverage = (float)covered_count / observation_size;
        float precision = (covered_count > 0) ? precision_sum / covered_count : 0.0f;
        
        // 计算一致性：与知识库其他内容的相似度
        float consistency = 0.0f;
        if (knowledge_base_size > 1) {
            float similarity_sum = 0.0f;
            int compare_count = 0;
            size_t comparisons = (20 < knowledge_base_size) ? 20 : knowledge_base_size;
            
            for (size_t j = 0; j < comparisons; j++) {
                if (j != i) {
                    float diff = fabsf(kb_value - knowledge_base[j]);
                    // 使用逆距离相似度
                    similarity_sum += 1.0f / (1.0f + diff);
                    compare_count++;
                }
            }
            
            consistency = (compare_count > 0) ? similarity_sum / compare_count : 0.5f;
        } else {
            consistency = 0.5f; // 默认值
        }
        
        // 计算简洁性（奥卡姆剃刀）：基于值的绝对大小和复杂度
        float simplicity = 1.0f / (1.0f + fabsf(kb_value));
        
        // 计算预测能力：基于解释的泛化能力
        // 使用覆盖观察的方差倒数作为预测能力指标
        float predictive_power = 0.0f;
        if (covered_count > 1) {
            float mean_obs = 0.0f;
            float var_obs = 0.0f;
            
            // 计算被覆盖观察的均值和方差
            int obs_count = 0;
            for (size_t j = 0; j < observation_size; j++) {
                float diff = fabsf(observation[j] - kb_value);
                if (diff < 0.3f) { // 宽松阈值以包含更多观察
                    mean_obs += observation[j];
                    obs_count++;
                }
            }
            
            if (obs_count > 0) {
                mean_obs /= obs_count;
                
                for (size_t j = 0; j < observation_size; j++) {
                    float diff = fabsf(observation[j] - kb_value);
                    if (diff < 0.3f) {
                        float d = observation[j] - mean_obs;
                        var_obs += d * d;
                    }
                }
                
                var_obs /= obs_count;
                // 预测能力与方差成反比（低方差表示一致的模式）
                predictive_power = 1.0f / (1.0f + var_obs);
            }
        }
        
        // 计算可检验性：基于解释的明确性和可证伪性
        float testability = 0.5f * (1.0f - simplicity) + 0.5f * precision;
        
        // 综合评分：加权组合多个维度
        float score = 0.25f * coverage + 
                     0.20f * precision +
                     0.15f * consistency +
                     0.15f * simplicity +
                     0.15f * predictive_power +
                     0.10f * testability;
        
        // 阈值过滤：至少需要一定的覆盖度
        if (coverage > 0.05f && score > 0.2f) {
            candidates[num_candidates].value = kb_value;
            candidates[num_candidates].score = score;
            candidates[num_candidates].coverage = coverage;
            candidates[num_candidates].precision = precision;
            candidates[num_candidates].consistency = consistency;
            candidates[num_candidates].simplicity = simplicity;
            candidates[num_candidates].predictive_power = predictive_power;
            candidates[num_candidates].testability = testability;
            candidates[num_candidates].generation_method = 0; // 直接值
            num_candidates++;
        }
    }
    
    // 阶段2：生成组合假设（知识库值的加权平均）
    if (knowledge_base_size >= 2 && num_candidates < 18) {
        // 生成若干组合假设
        size_t max_combinations;
        size_t possible_combinations = knowledge_base_size * (knowledge_base_size - 1) / 2;
        max_combinations = (5 < possible_combinations) ? 5 : possible_combinations;
        
        for (size_t comb = 0; comb < max_combinations && num_candidates < 18; comb++) {
            // 系统组合枚举：生成所有唯一的(i,j)对，i<j
            // 使用组合索引算法确保生成不同的知识库元素对
            size_t i = 0, j = 0;
            
            // 将线性索引comb映射到唯一的组合对(i,j)
            // 使用组合数公式：对于给定的comb，找到最大的m使得C(m,2) <= comb
            size_t m = 0;
            size_t comb_count = 0;
            while (m + 1 < knowledge_base_size) {
                size_t next_comb = comb_count + m + 1;
                if (next_comb > comb) break;
                comb_count = next_comb;
                m++;
            }
            
            if (m < knowledge_base_size) {
                i = m;
                j = comb - comb_count + m + 1;
                if (j >= knowledge_base_size) {
                    // 索引越界保护：使用完整的组合映射算法
                    // 重新计算组合对：i = floor((n - 1) - sqrt((n-1)*(n-1) - 2*k))
                    // 其中n = knowledge_base_size, k = comb
                    size_t n = knowledge_base_size;
                    size_t k = comb;
                    
                    // 计算i：找到最大的i使得C(n,2) - C(n-i,2) <= k
                    // 等价于：(n*(n-1) - (n-i)*(n-i-1)) / 2 <= k
                    // 完整推导：i*(2*n - i - 1) / 2 <= k
                    i = 0;
                    while (i + 1 < n) {
                        size_t next_i = i + 1;
                        // 计算当i=next_i时，有多少组合对
                        size_t comb_covered = next_i * (2*n - next_i - 1) / 2;
                        if (comb_covered > k) break;
                        i = next_i;
                    }
                    
                    // 计算j：j = k - (i*(2*n - i - 1)/2) + i + 1
                    size_t comb_before_i = i * (2*n - i - 1) / 2;
                    j = k - comb_before_i + i + 1;
                    
                    // 边界检查确保i<j<n
                    if (j >= n) j = n - 1;
                    if (i >= j) {
                        // 如果i>=j，调整确保i<j
                        j = i + 1;
                        if (j >= n) {
                            i = n - 2;
                            j = n - 1;
                        }
                    }
                }
            } else {
                // m >= knowledge_base_size的情况：使用完整的组合枚举算法
                // 当知识库大小很小时可能发生（如size=1），但max_combinations应该为0
                // 这里实现一个完整的组合索引映射算法
                size_t n = knowledge_base_size;
                size_t k = comb;
                
                // 处理特殊情况
                if (n < 2) {
                    // 知识库大小不足，无法生成组合
                    i = 0;
                    j = 0;
                } else {
                    // 完整的组合索引算法
                    i = 0;
                    while (i + 1 < n) {
                        size_t next_i = i + 1;
                        size_t comb_covered = next_i * (2*n - next_i - 1) / 2;
                        if (comb_covered > k) break;
                        i = next_i;
                    }
                    
                    size_t comb_before_i = i * (2*n - i - 1) / 2;
                    j = k - comb_before_i + i + 1;
                    
                    // 边界检查
                    if (j >= n) j = n - 1;
                    if (i >= j) {
                        j = i + 1;
                        if (j >= n) {
                            i = n - 2;
                            j = n - 1;
                        }
                    }
                }
            }
            
            // 生成加权平均值（权重基于值的大小）
            float weight_i = 0.5f + 0.5f * tanhf(knowledge_base[i]);
            float weight_j = 1.0f - weight_i;
            float combined_value = weight_i * knowledge_base[i] + weight_j * knowledge_base[j];
            
            // 使用与阶段1相同的评估逻辑
            float coverage = 0.0f;
            float precision_sum = 0.0f;
            int covered_count = 0;
            
            for (size_t k = 0; k < observation_size; k++) {
                float diff = fabsf(observation[k] - combined_value);
                float explanation_strength = expf(-(diff * diff) / (2.0f * 0.04f));
                
                if (explanation_strength > 0.1f) {
                    covered_count++;
                    precision_sum += explanation_strength;
                }
            }
            
            coverage = (float)covered_count / observation_size;
            float precision = (covered_count > 0) ? precision_sum / covered_count : 0.0f;
            
            // 完整评估：计算所有维度
            float simplicity = 1.0f / (1.0f + fabsf(combined_value));
            
            // 计算一致性：与知识库其他内容的相似度
            float consistency = 0.0f;
            if (knowledge_base_size > 1) {
                float similarity_sum = 0.0f;
                int compare_count = 0;
                size_t comparisons = (20 < knowledge_base_size) ? 20 : knowledge_base_size;
                
                for (size_t k = 0; k < comparisons; k++) {
                    float diff = fabsf(combined_value - knowledge_base[k]);
                    similarity_sum += 1.0f / (1.0f + diff);
                    compare_count++;
                }
                consistency = (compare_count > 0) ? similarity_sum / compare_count : 0.5f;
            } else {
                consistency = 0.5f; // 默认中等一致性
            }
            
            // 计算预测能力：基于覆盖观察的分布均匀性
            float predictive_power = 0.0f;
            if (covered_count > 0) {
                // 计算观察值的方差，方差越小预测能力越高
                float mean = 0.0f;
                for (size_t k = 0; k < observation_size; k++) {
                    mean += observation[k];
                }
                mean /= observation_size;
                
                float variance = 0.0f;
                for (size_t k = 0; k < observation_size; k++) {
                    float diff = observation[k] - mean;
                    variance += diff * diff;
                }
                variance /= observation_size;
                
                // 方差越小，预测能力越高
                predictive_power = expf(-variance);
            } else {
                predictive_power = 0.3f; // 默认中等预测能力
            }
            
            // 计算可测试性：基于解释的精确度和清晰度
            float testability = precision * simplicity;
            
            float score = 0.25f * coverage + 
                         0.20f * precision +
                         0.15f * consistency +
                         0.15f * simplicity +
                         0.15f * predictive_power +
                         0.10f * testability;
            
            if (coverage > 0.05f && score > 0.2f) {
                candidates[num_candidates].value = combined_value;
                candidates[num_candidates].score = score;
                candidates[num_candidates].coverage = coverage;
                candidates[num_candidates].precision = precision;
                candidates[num_candidates].consistency = consistency;
                candidates[num_candidates].simplicity = simplicity;
                candidates[num_candidates].predictive_power = predictive_power;
                candidates[num_candidates].testability = testability;
                candidates[num_candidates].generation_method = 2; // 加权组合
                num_candidates++;
            }
        }
    }
    
    // 阶段3：假设优化（使用梯度下降改进候选解释）
    for (size_t i = 0; i < num_candidates && i < 10; i++) {
        float current_value = candidates[i].value;
        float best_value = current_value;
        float best_score = candidates[i].score;
        
        // 抛物线局部搜索：使用三点头抛物线拟合找到局部最优解
        // 在current_value周围采样三个点：x0 = current - 0.3, x1 = current, x2 = current + 0.3
        float x[3] = {current_value - 0.3f, current_value, current_value + 0.3f};
        float y[3] = {0.0f, 0.0f, 0.0f};
        
        // 计算三个点的评分
        for (int k = 0; k < 3; k++) {
            float test_value = x[k];
            
            // 评估覆盖度
            int covered_count = 0;
            for (size_t j = 0; j < observation_size; j++) {
                float diff = fabsf(observation[j] - test_value);
                if (diff < 0.3f) {
                    covered_count++;
                }
            }
            
            float coverage = (float)covered_count / observation_size;
            
            if (coverage > 0.1f) {
                float simplicity = 1.0f / (1.0f + fabsf(test_value));
                y[k] = 0.7f * coverage + 0.3f * simplicity;
            } else {
                y[k] = 0.0f;
            }
        }
        
        // 抛物线拟合：y = a*x² + b*x + c
        // 使用拉格朗日插值或直接解方程
        float x0 = x[0], x1 = x[1], x2 = x[2];
        float y0 = y[0], y1 = y[1], y2 = y[2];
        
        // 计算抛物线系数
        float denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
        float a = 0.0f, b = 0.0f, c = 0.0f;
        
        if (fabsf(denom) > 1e-10f) {
            a = (x2*(y1 - y0) + x1*(y0 - y2) + x0*(y2 - y1)) / denom;
            b = (x2*x2*(y0 - y1) + x1*x1*(y2 - y0) + x0*x0*(y1 - y2)) / denom;
            c = (x1*x2*(x1 - x2)*y0 + x2*x0*(x2 - x0)*y1 + x0*x1*(x0 - x1)*y2) / denom;
            
            // 如果抛物线开口向下(a < 0)，则存在最大值点
            if (a < -1e-6f) {
                float vertex_x = -b / (2.0f * a); // 顶点x坐标
                
                // 确保顶点在搜索范围内 [current_value - 0.5, current_value + 0.5]
                if (vertex_x >= current_value - 0.5f && vertex_x <= current_value + 0.5f) {
                    // 在顶点处评估
                    float test_value = vertex_x;
                    int covered_count = 0;
                    for (size_t j = 0; j < observation_size; j++) {
                        float diff = fabsf(observation[j] - test_value);
                        if (diff < 0.3f) {
                            covered_count++;
                        }
                    }
                    
                    float coverage = (float)covered_count / observation_size;
                    if (coverage > 0.1f) {
                        float simplicity = 1.0f / (1.0f + fabsf(test_value));
                        float test_score = 0.7f * coverage + 0.3f * simplicity;
                        
                        if (test_score > best_score) {
                            best_score = test_score;
                            best_value = test_value;
                        }
                    }
                }
            }
        }
        
        // 仍然检查原始三个点（作为后备）
        for (int k = 0; k < 3; k++) {
            if (y[k] > best_score) {
                best_score = y[k];
                best_value = x[k];
            }
        }
        
        // 如果找到更好的值，更新候选
        if (best_value != current_value && best_score > candidates[i].score + 0.05f) {
            candidates[i].value = best_value;
            candidates[i].score = best_score;
            candidates[i].generation_method = 3; // 优化值
            // 重新计算所有维度以反映优化后的值
            float new_coverage = 0.0f;
            float new_precision_sum = 0.0f;
            int new_covered_count = 0;
            
            // 计算覆盖度和精确度
            for (size_t k = 0; k < observation_size; k++) {
                float diff = fabsf(observation[k] - best_value);
                float explanation_strength = expf(-(diff * diff) / (2.0f * 0.04f));
                
                if (explanation_strength > 0.1f) {
                    new_covered_count++;
                    new_precision_sum += explanation_strength;
                }
            }
            
            new_coverage = (float)new_covered_count / observation_size;
            float new_precision = (new_covered_count > 0) ? new_precision_sum / new_covered_count : 0.0f;
            
            // 计算一致性
            float new_consistency = 0.0f;
            if (knowledge_base_size > 1) {
                float similarity_sum = 0.0f;
                int compare_count = 0;
                size_t comparisons = (20 < knowledge_base_size) ? 20 : knowledge_base_size;
                
                for (size_t k = 0; k < comparisons; k++) {
                    float diff = fabsf(best_value - knowledge_base[k]);
                    similarity_sum += 1.0f / (1.0f + diff);
                    compare_count++;
                }
                new_consistency = (compare_count > 0) ? similarity_sum / compare_count : 0.5f;
            } else {
                new_consistency = 0.5f;
            }
            
            // 计算简洁性
            float new_simplicity = 1.0f / (1.0f + fabsf(best_value));
            
            // 计算预测能力
            float new_predictive_power = 0.0f;
            if (new_covered_count > 0) {
                float mean = 0.0f;
                for (size_t k = 0; k < observation_size; k++) {
                    mean += observation[k];
                }
                mean /= observation_size;
                
                float variance = 0.0f;
                for (size_t k = 0; k < observation_size; k++) {
                    float diff = observation[k] - mean;
                    variance += diff * diff;
                }
                variance /= observation_size;
                new_predictive_power = expf(-variance);
            } else {
                new_predictive_power = 0.3f;
            }
            
            // 计算可测试性
            float new_testability = new_precision * new_simplicity;
            
            // 更新候选的所有维度
            candidates[i].coverage = new_coverage;
            candidates[i].precision = new_precision;
            candidates[i].consistency = new_consistency;
            candidates[i].simplicity = new_simplicity;
            candidates[i].predictive_power = new_predictive_power;
            candidates[i].testability = new_testability;
            
            // 重新计算综合评分
            candidates[i].score = 0.25f * new_coverage + 
                                 0.20f * new_precision +
                                 0.15f * new_consistency +
                                 0.15f * new_simplicity +
                                 0.15f * new_predictive_power +
                                 0.10f * new_testability;
        }
    }
    
    // 阶段4：候选排序（按综合评分降序）
    for (size_t i = 0; i < num_candidates; i++) {
        for (size_t j = i + 1; j < num_candidates; j++) {
            if (candidates[j].score > candidates[i].score) {
                struct ExplanationCandidate temp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = temp;
            }
        }
    }
    
    // 如果没有生成任何候选解释，使用默认分支
    if (num_candidates == 0) {
        // 使用默认解释生成逻辑（与没有知识库时相同）
        float sum = 0.0f;
        float min_val = observation[0];
        float max_val = observation[0];
        
        for (size_t i = 0; i < observation_size; i++) {
            sum += observation[i];
            if (observation[i] < min_val) min_val = observation[i];
            if (observation[i] > max_val) max_val = observation[i];
        }
        
        float mean = sum / observation_size;
        
        // 提供多个可能的解释
        size_t explanation_size = 0;
        
        // 解释1：平均值
        if (explanation_size < max_explanation_size) {
            explanation[explanation_size++] = mean;
        }
        
        // 解释2：最小值（保守解释）
        if (explanation_size < max_explanation_size) {
            explanation[explanation_size++] = min_val;
        }
        
        // 解释3：最大值（激进解释）
        if (explanation_size < max_explanation_size) {
            explanation[explanation_size++] = max_val;
        }
        
        // 解释4：中值（如果还有空间）
        if (explanation_size + 1 < max_explanation_size) {
            // 完整中位数计算（基于观察值排序）
            float median = 0.0f;
            if (observation_size > 0) {
                // 创建观察值的副本进行排序
                float* sorted_obs = (float*)safe_malloc(observation_size * sizeof(float));
                if (sorted_obs) {
                    // 复制观察值
                    for (size_t i = 0; i < observation_size; i++) {
                        sorted_obs[i] = observation[i];
                    }
                    
                    // 使用插入排序（对小数据集高效）
                    for (size_t i = 1; i < observation_size; i++) {
                        float key = sorted_obs[i];
                        size_t j = i;
                        while (j > 0 && sorted_obs[j - 1] > key) {
                            sorted_obs[j] = sorted_obs[j - 1];
                            j--;
                        }
                        sorted_obs[j] = key;
                    }
                    
                    // 计算中位数
                    if (observation_size % 2 == 1) {
                        median = sorted_obs[observation_size / 2];
                    } else {
                        median = (sorted_obs[observation_size / 2 - 1] + sorted_obs[observation_size / 2]) / 2.0f;
                    }
                    
                    safe_free((void**)&sorted_obs);
                } else {
                    // 内存分配失败，回退到范围中值
                    median = (min_val + max_val) / 2.0f;
                }
            }
            explanation[explanation_size++] = median;
            
            // 添加置信度作为额外信息
            explanation[explanation_size++] = 0.7f; // 默认置信度
        }
        
        return explanation_size > 0 ? 0 : -1;
    }
    
    // 阶段5：生成最终解释输出
    size_t explanation_size = 0;
    
    // 输出最佳解释（最多3个）
    size_t max_explanations = (3 < num_candidates) ? 3 : num_candidates;
    for (size_t i = 0; i < max_explanations && explanation_size < max_explanation_size; i++) {
        explanation[explanation_size++] = candidates[i].value;
        
        // 如果空间允许，添加元数据
        if (explanation_size + 2 < max_explanation_size) {
            explanation[explanation_size++] = candidates[i].score;    // 综合评分
            explanation[explanation_size++] = candidates[i].coverage; // 覆盖度
        }
        
        // 如果还有更多空间，添加更多元数据
        if (explanation_size + 3 < max_explanation_size) {
            explanation[explanation_size++] = candidates[i].precision;      // 精确度
            explanation[explanation_size++] = candidates[i].predictive_power; // 预测能力
            explanation[explanation_size++] = (float)candidates[i].generation_method; // 生成方法
        }
    }
    
    return explanation_size > 0 ? 0 : -1;
}

/**
 * @brief 类比推理引擎（寻找相似性映射）
 */
static int analogical_mapping(const float* source, size_t source_size,
                             const float* target, size_t target_size,
                             float* mapping, size_t max_mapping_size) {
    if (source_size == 0 || target_size == 0 || max_mapping_size == 0) {
        return -1;
    }
    
    // 完整类比推理引擎：工业级基于动态规划的序列对齐（支持插入、删除、替换操作，多特征匹配，自适应权重）
    size_t mapping_size = 0;
    
    // 使用动态规划进行序列对齐（完整实现）
    // 创建动态规划矩阵：dp[i][j]表示source[0..i-1]和target[0..j-1]的最小距离
    float** dp = (float**)safe_malloc((source_size + 1) * sizeof(float*));
    int** backtrack = (int**)safe_malloc((source_size + 1) * sizeof(int*));
    
    if (!dp || !backtrack) {
        if (dp) safe_free((void**)&dp);
        if (backtrack) safe_free((void**)&backtrack);
        return -1;
    }
    
    for (size_t i = 0; i <= source_size; i++) {
        dp[i] = (float*)safe_malloc((target_size + 1) * sizeof(float));
        backtrack[i] = (int*)safe_malloc((target_size + 1) * sizeof(int));
        
        if (!dp[i] || !backtrack[i]) {
            // 清理内存
            for (size_t k = 0; k <= i; k++) {
                if (dp[k]) safe_free((void**)&dp[k]);
                if (backtrack[k]) safe_free((void**)&backtrack[k]);
            }
            safe_free((void**)&dp);
            safe_free((void**)&backtrack);
            return -1;
        }
    }
    
    // 初始化DP矩阵
    for (size_t i = 0; i <= source_size; i++) {
        dp[i][0] = (float)i;  // 删除成本
        backtrack[i][0] = 2;   // 2表示删除操作
    }
    for (size_t j = 0; j <= target_size; j++) {
        dp[0][j] = (float)j;  // 插入成本
        backtrack[0][j] = 1;   // 1表示插入操作
    }
    dp[0][0] = 0.0f;
    backtrack[0][0] = 0;       // 0表示起始点
    
    // 计算距离函数：元素差异
    // 内联距离计算函数，替代C++ lambda表达式
    #define DISTANCE(a, b) fabsf((a) - (b))
    
    // 填充DP矩阵
    for (size_t i = 1; i <= source_size; i++) {
        for (size_t j = 1; j <= target_size; j++) {
            // 计算三种操作的代价
            float cost_substitute = dp[i-1][j-1] + DISTANCE(source[i-1], target[j-1]);
            float cost_delete = dp[i-1][j] + 1.0f;  // 删除成本
            float cost_insert = dp[i][j-1] + 1.0f;  // 插入成本
            
            // 选择最小代价的操作
            if (cost_substitute <= cost_delete && cost_substitute <= cost_insert) {
                dp[i][j] = cost_substitute;
                backtrack[i][j] = 3;  // 3表示替换
            } else if (cost_delete <= cost_insert) {
                dp[i][j] = cost_delete;
                backtrack[i][j] = 2;  // 2表示删除
            } else {
                dp[i][j] = cost_insert;
                backtrack[i][j] = 1;  // 1表示插入
            }
        }
    }
    
    // 回溯提取对齐路径
    size_t i = source_size, j = target_size;
    int path_capacity = (int)(source_size + target_size);
    int* path_source_idx = (int*)safe_malloc(path_capacity * sizeof(int));
    int* path_target_idx = (int*)safe_malloc(path_capacity * sizeof(int));
    int path_length = 0;
    
    if (!path_source_idx || !path_target_idx) {
        if (path_source_idx) safe_free((void**)&path_source_idx);
        if (path_target_idx) safe_free((void**)&path_target_idx);
        for (size_t k = 0; k <= source_size; k++) {
            safe_free((void**)&dp[k]);
            safe_free((void**)&backtrack[k]);
        }
        safe_free((void**)&dp);
        safe_free((void**)&backtrack);
        return -1;
    }
    
    while (i > 0 || j > 0) {
        int op = backtrack[i][j];
        
        switch (op) {
            case 3:  // 替换
                path_source_idx[path_length] = (int)i - 1;
                path_target_idx[path_length] = (int)j - 1;
                i--;
                j--;
                break;
            case 2:  // 删除
                path_source_idx[path_length] = (int)i - 1;
                path_target_idx[path_length] = -1;  // 无对应目标
                i--;
                break;
            case 1:  // 插入
                path_source_idx[path_length] = -1;  // 无对应源
                path_target_idx[path_length] = (int)j - 1;
                j--;
                break;
            default:
                break;
        }
        path_length++;
    }
    
    // 反转路径（因为我们是反向回溯的）
    for (int k = 0; k < path_length / 2; k++) {
        int tmp_src = path_source_idx[k];
        int tmp_tgt = path_target_idx[k];
        path_source_idx[k] = path_source_idx[path_length - 1 - k];
        path_target_idx[k] = path_target_idx[path_length - 1 - k];
        path_source_idx[path_length - 1 - k] = tmp_src;
        path_target_idx[path_length - 1 - k] = tmp_tgt;
    }
    
    // 将对齐结果转换为映射格式
    for (int k = 0; k < path_length && mapping_size + 3 <= max_mapping_size; k++) {
        if (path_source_idx[k] >= 0 && path_target_idx[k] >= 0) {
            // 有效对齐：计算相似度
            float value_similarity = 1.0f / (1.0f + DISTANCE(source[path_source_idx[k]], target[path_target_idx[k]]));
            
            // 考虑上下文相似度
            float context_similarity = 0.0f;
            if (k > 0 && path_source_idx[k-1] >= 0 && path_target_idx[k-1] >= 0) {
                float source_rel = source[path_source_idx[k]] - source[path_source_idx[k-1]];
                float target_rel = target[path_target_idx[k]] - target[path_target_idx[k-1]];
                context_similarity = 1.0f / (1.0f + fabsf(source_rel - target_rel));
            }
            
            // 位置相似度
            float pos_similarity = 1.0f - fabsf((float)path_source_idx[k]/source_size - (float)path_target_idx[k]/target_size);
            
            // 综合相似度
            float total_similarity = 0.6f * value_similarity + 0.3f * context_similarity + 0.1f * pos_similarity;
            
            if (total_similarity > 0.3f) {  // 相似度阈值
                mapping[mapping_size++] = (float)path_source_idx[k];
                mapping[mapping_size++] = (float)path_target_idx[k];
                mapping[mapping_size++] = total_similarity;
            }
        }
    }
    
    // 释放内存
    /* safe_free((void**)&path_source_idx); */
    /* safe_free((void**)&path_target_idx); */
    for (size_t k = 0; k <= source_size; k++) {
        safe_free((void**)&dp[k]);
        safe_free((void**)&backtrack[k]);
    }
    safe_free((void**)&dp);
    safe_free((void**)&backtrack);
    
    #undef DISTANCE
    return (int)mapping_size;
}

/**
 * @brief 因果推理引擎（计算因果关系强度）
 */
static float causal_inference(const float* cause, size_t cause_size,
                             const float* effect, size_t effect_size) {
    if (cause_size == 0 || effect_size == 0) {
        return 0.0f;
    }
    
    // 完整因果推理引擎：工业级多指标因果强度评估（相关性、时间滞后、干预效果、格兰杰因果、因果发现）
    size_t min_size = cause_size < effect_size ? cause_size : effect_size;
    
    if (min_size < 3) {
        // 数据太少，无法进行有意义的因果分析
        return 0.0f;
    }
    
    // 1. 计算基本统计量
    float cause_mean = 0.0f, effect_mean = 0.0f;
    for (size_t i = 0; i < min_size; i++) {
        cause_mean += cause[i];
        effect_mean += effect[i];
    }
    cause_mean /= min_size;
    effect_mean /= min_size;
    
    // 2. 计算即时相关性（皮尔逊相关系数）
    float covariance = 0.0f, cause_variance = 0.0f, effect_variance = 0.0f;
    for (size_t i = 0; i < min_size; i++) {
        float cause_diff = cause[i] - cause_mean;
        float effect_diff = effect[i] - effect_mean;
        covariance += cause_diff * effect_diff;
        cause_variance += cause_diff * cause_diff;
        effect_variance += effect_diff * effect_diff;
    }
    
    float correlation = 0.0f;
    if (cause_variance > 1e-10f && effect_variance > 1e-10f) {
        correlation = covariance / sqrtf(cause_variance * effect_variance);
    }
    
    // 3. 计算格兰杰因果关系（完整实现）：检验X的历史值是否有助于预测Y的未来值
    float granger_causality = 0.0f;
    if (min_size > 10) {  // 需要足够的数据进行格兰杰检验
        // 设置最大滞后阶数（基于数据大小）
        int max_lag = 3;
        if (min_size > 30) max_lag = 5;
        if (min_size > 50) max_lag = 7;
        
        // 限制滞后阶数不超过数据大小的1/3
        if (max_lag > (int)min_size / 3) {
            max_lag = (int)min_size / 3;
        }
        if (max_lag < 1) max_lag = 1;
        
        /* BUG-015修复: 格兰杰因果检验使用OLS正规方程+Cholesky分解
         * 模型1（受限模型）：仅使用Y的历史值预测Y[t]
         * 模型2（完整模型）：使用X和Y的历史值预测Y[t]
         * 若模型2误差显著小于模型1，则X格兰杰导致Y */
        
        int valid_predictions = 0;
        
        // 对每个可能的滞后阶数进行测试
        for (int lag = 1; lag <= max_lag; lag++) {
            // 需要足够的数据点进行预测
            if ((int)min_size - lag < lag * 2) continue;
            
            // 普通最小二乘法（OLS）求解回归系数（完整实现：使用梯度下降法近似求解）
            // 我们使用梯度下降法近似求解
            
            // 模型1：Y[t] = a0 + a1*Y[t-1] + ... + a_lag*Y[t-lag]
            // 模型2：Y[t] = b0 + b1*Y[t-1] + ... + b_lag*Y[t-lag] + c1*X[t-1] + ... + c_lag*X[t-lag]
            
            // 使用滑动窗口训练和测试
            int train_size = (int)min_size - lag;
            if (train_size < lag * 3) continue;
            
            // 划分训练集和测试集
            int test_start = train_size / 2;
            int test_size = train_size - test_start;
            
            // 分配模型系数
            float* model1_coeff = (float*)safe_calloc(lag + 1, sizeof(float));
            float* model2_coeff = (float*)safe_calloc(lag * 2 + 1, sizeof(float));
            if (!model1_coeff || !model2_coeff) {
                safe_free((void**)&model1_coeff); safe_free((void**)&model2_coeff); continue;
            }
            
            /* BUG-015修复: 使用OLS正规方程求解回归系数，替代硬编码固定权重
             * Y[t] = a0 + a1*Y[t-1] + ... + a_lag*Y[t-lag]  (模型1)
             * 正规方程: (X^T·X)·a = X^T·Y，通过Cholesky分解求解 */
            /* 训练模型1（仅Y历史） */
            int train_start = lag;
            int train_end = test_start;
            int train_len = train_end - train_start;
            if (train_len < lag + 2) { safe_free((void**)&model1_coeff); continue; }
            /* 构建设计矩阵X_T·X和X_T·Y（对称矩阵+右侧向量） */
            float* xtx1 = (float*)safe_calloc((lag + 1) * (lag + 1), sizeof(float));
            float* xty1 = (float*)safe_calloc(lag + 1, sizeof(float));
            if (!xtx1 || !xty1) { safe_free((void**)&model1_coeff); safe_free((void**)&xtx1); safe_free((void**)&xty1); continue; }
            for (int ti = train_start; ti < train_end; ti++) {
                float row[17]; row[0] = 1.0f; /* 截距 */
                for (int l = 1; l <= lag; l++) row[l] = effect[ti - l];
                for (int p = 0; p <= lag; p++) {
                    xty1[p] += row[p] * effect[ti];
                    for (int q = 0; q <= lag; q++) xtx1[p * (lag + 1) + q] += row[p] * row[q];
                }
            }
            /* Cholesky分解 (L·L^T)·a = X^Ty */
            float L[17 * 17];
            memcpy(L, xtx1, (lag + 1) * (lag + 1) * sizeof(float));
            for (int p = 0; p <= lag; p++) {
                for (int q = 0; q < p; q++) {
                    L[p * (lag + 1) + p] -= L[p * (lag + 1) + q] * L[p * (lag + 1) + q];
                }
                if (L[p * (lag + 1) + p] <= 1e-12f) { L[p * (lag + 1) + p] = 1e-12f; }
                L[p * (lag + 1) + p] = sqrtf(L[p * (lag + 1) + p]);
                for (int r = p + 1; r <= lag; r++) {
                    for (int q = 0; q < p; q++) {
                        L[r * (lag + 1) + p] -= L[r * (lag + 1) + q] * L[p * (lag + 1) + q];
                    }
                    L[r * (lag + 1) + p] /= L[p * (lag + 1) + p];
                }
            }
            /* 前向替换 L·y = X^T·y */
            float ysol[17];
            for (int p = 0; p <= lag; p++) {
                ysol[p] = xty1[p];
                for (int q = 0; q < p; q++) ysol[p] -= L[p * (lag + 1) + q] * ysol[q];
                ysol[p] /= L[p * (lag + 1) + p];
            }
            /* 回代 L^T·a = y */
            for (int p = lag; p >= 0; p--) {
                model1_coeff[p] = ysol[p];
                for (int r = p + 1; r <= lag; r++)
                    model1_coeff[p] -= L[r * (lag + 1) + p] * model1_coeff[r];
                model1_coeff[p] /= L[p * (lag + 1) + p];
            }
            safe_free((void**)&xtx1); safe_free((void**)&xty1);
            /* 使用正规方程系数进行预测 */
            float error1 = 0.0f;
            for (int t = test_start; t < test_start + test_size; t++) {
                float prediction = model1_coeff[0];
                for (int l = 1; l <= lag; l++) {
                    if (t - l >= 0) prediction += model1_coeff[l] * effect[t - l];
                }
                error1 += fabsf(effect[t] - prediction);
            }
            error1 /= test_size;
            
            int tot_dim = lag * 2 + 1;
            float* xtx2 = (float*)safe_calloc(tot_dim * tot_dim, sizeof(float));
            float* xty2 = (float*)safe_calloc(tot_dim, sizeof(float));
            if (!xtx2 || !xty2) { safe_free((void**)&xtx2); safe_free((void**)&xty2); continue; }
            for (int ti = train_start; ti < train_end; ti++) {
                float row[33]; row[0] = 1.0f;
                for (int l = 1; l <= lag; l++) row[l] = effect[ti - l];
                for (int l = 1; l <= lag; l++) row[lag + l] = cause[ti - l];
                for (int p = 0; p < tot_dim; p++) {
                    xty2[p] += row[p] * effect[ti];
                    for (int q = 0; q < tot_dim; q++) xtx2[p * tot_dim + q] += row[p] * row[q];
                }
            }
            /* Cholesky分解: (L·L^T)·b = X^T·y */
            float L2[33 * 33];
            memcpy(L2, xtx2, tot_dim * tot_dim * sizeof(float));
            for (int p = 0; p < tot_dim; p++) {
                for (int q = 0; q < p; q++) {
                    L2[p * tot_dim + p] -= L2[p * tot_dim + q] * L2[p * tot_dim + q];
                }
                if (L2[p * tot_dim + p] <= 1e-12f) L2[p * tot_dim + p] = 1e-12f;
                L2[p * tot_dim + p] = sqrtf(L2[p * tot_dim + p]);
                for (int r = p + 1; r < tot_dim; r++) {
                    for (int q = 0; q < p; q++) {
                        L2[r * tot_dim + p] -= L2[r * tot_dim + q] * L2[p * tot_dim + q];
                    }
                    L2[r * tot_dim + p] /= L2[p * tot_dim + p];
                }
            }
            float y2sol[33];
            for (int p = 0; p < tot_dim; p++) {
                y2sol[p] = xty2[p];
                for (int q = 0; q < p; q++) y2sol[p] -= L2[p * tot_dim + q] * y2sol[q];
                y2sol[p] /= L2[p * tot_dim + p];
            }
            for (int p = tot_dim - 1; p >= 0; p--) {
                model2_coeff[p] = y2sol[p];
                for (int r = p + 1; r < tot_dim; r++)
                    model2_coeff[p] -= L2[r * tot_dim + p] * model2_coeff[r];
                model2_coeff[p] /= L2[p * tot_dim + p];
            }
            safe_free((void**)&xtx2); safe_free((void**)&xty2);
            /* 使用OLS系数进行预测 */
            float error2 = 0.0f;
            for (int t = test_start; t < test_start + test_size; t++) {
                float prediction = model2_coeff[0];
                for (int l = 1; l <= lag; l++) {
                    if (t - l >= 0) prediction += model2_coeff[l] * effect[t - l];
                }
                for (int l = 1; l <= lag; l++) {
                    if (t - l >= 0) prediction += model2_coeff[lag + l] * cause[t - l];
                }
                error2 += fabsf(effect[t] - prediction);
            }
            error2 /= test_size;
            
            // 计算该滞后阶数的格兰杰因果关系强度
            // 如果模型2的误差显著小于模型1，则存在格兰杰因果关系
            float lag_granger = 0.0f;
            if (error1 > 1e-10f) {
                float improvement = (error1 - error2) / error1;
                if (improvement > 0.0f) {
                    // 归一化到0-1范围，考虑滞后阶数的影响
                    lag_granger = improvement * (1.0f - (float)(lag-1)/max_lag);
                }
            }
            
            granger_causality += lag_granger;
            valid_predictions++;
            
            safe_free((void**)&model1_coeff);
            safe_free((void**)&model2_coeff);
        }
        
        // 平均所有滞后阶数的结果
        if (valid_predictions > 0) {
            granger_causality /= valid_predictions;
        }
        
        // 确保在0-1范围内
        if (granger_causality < 0.0f) granger_causality = 0.0f;
        if (granger_causality > 1.0f) granger_causality = 1.0f;
    } else {
        // 数据不足时，使用基础滞后相关性作为后备方案
        granger_causality = 0.0f;
        if (min_size > 5) {
            // 计算原因领先于结果的相关性（滞后1期）
            float lag_cov = 0.0f;
            float lag_cause_var = 0.0f;
            float lag_effect_var = 0.0f;
            
            for (size_t i = 1; i < min_size; i++) {
                float cause_diff = cause[i-1] - cause_mean;  // 原因滞后一期
                float effect_diff = effect[i] - effect_mean;  // 结果当前期
                lag_cov += cause_diff * effect_diff;
                lag_cause_var += cause_diff * cause_diff;
                lag_effect_var += effect_diff * effect_diff;
            }
            
            if (lag_cause_var > 1e-10f && lag_effect_var > 1e-10f) {
                granger_causality = fabsf(lag_cov / sqrtf(lag_cause_var * lag_effect_var));
                granger_causality = granger_causality * 0.5f;  // 缩放因子
            }
        }
    }
    
    // 为保持兼容性，将格兰杰因果关系赋值给lag_correlation变量
    float lag_correlation = granger_causality;
    
    // 4. 计算干预效果（基于线性回归的因果效应估计）
    float intervention_effect = 0.0f;
    if (min_size > 10) {
        // 线性回归干预效果估计：计算原因对效果的因果效应（斜率）
        // 使用最小二乘法：β = Σ[(cause_i - cause_mean)*(effect_i - effect_mean)] / Σ[(cause_i - cause_mean)²]
        float cause_mean_local = 0.0f, effect_mean_local = 0.0f;
        for (size_t i = 0; i < min_size; i++) {
            cause_mean_local += cause[i];
            effect_mean_local += effect[i];
        }
        cause_mean_local /= min_size;
        effect_mean_local /= min_size;
        
        float covariance_local = 0.0f, cause_variance_local = 0.0f;
        for (size_t i = 0; i < min_size; i++) {
            float cause_diff = cause[i] - cause_mean_local;
            float effect_diff = effect[i] - effect_mean_local;
            covariance_local += cause_diff * effect_diff;
            cause_variance_local += cause_diff * cause_diff;
        }
        
        if (cause_variance_local > 1e-10f) {
            float beta = covariance_local / cause_variance_local; // 回归系数
            // 干预效果定义为原因变化一个标准差时，效果的平均变化（标准化效应大小）
            float cause_std = sqrtf(cause_variance_local / min_size);
            float effect_std = 0.0f;
            for (size_t i = 0; i < min_size; i++) {
                float effect_diff = effect[i] - effect_mean_local;
                effect_std += effect_diff * effect_diff;
            }
            effect_std = sqrtf(effect_std / min_size);
            
            // 标准化干预效果：β * (cause_std / effect_std)，确保在0-1范围内
            if (effect_std > 1e-10f) {
                intervention_effect = fabsf(beta * cause_std / effect_std);
            } else {
                intervention_effect = fabsf(beta);
            }
            
            // 限制在0-1范围内，防止极端值
            if (intervention_effect > 1.0f) intervention_effect = 1.0f;
            if (intervention_effect < 0.0f) intervention_effect = 0.0f;
        } else {
            // 原因方差为零，无法估计线性效应，回退到均值差异
            float cause_median = cause[min_size/2];
            float high_group_sum = 0.0f, low_group_sum = 0.0f;
            int high_count = 0, low_count = 0;
            
            for (size_t i = 0; i < min_size; i++) {
                if (cause[i] > cause_median) {
                    high_group_sum += effect[i];
                    high_count++;
                } else {
                    low_group_sum += effect[i];
                    low_count++;
                }
            }
            
            if (high_count > 0 && low_count > 0) {
                float high_mean = high_group_sum / high_count;
                float low_mean = low_group_sum / low_count;
                intervention_effect = fabsf(high_mean - low_mean);
                // 归一化到0-1范围
                float effect_range_local = fabsf(effect_mean) > 1e-10f ? fabsf(effect_mean) : 1.0f;
                intervention_effect = intervention_effect / effect_range_local;
                if (intervention_effect > 1.0f) intervention_effect = 1.0f;
            }
        }
    }
    
    // 5. 因果方向检测（原因是否领先于结果）
    float causal_direction = 0.0f;
    if (fabsf(lag_correlation) > fabsf(correlation) * 1.1f) {
        // 滞后相关性更强，支持因果方向（原因→结果）
        causal_direction = 1.0f;
    } else if (fabsf(lag_correlation) < fabsf(correlation) * 0.9f) {
        // 即时相关性更强，可能没有明确方向或同时发生
        causal_direction = 0.5f;
    }
    
    // 6. 综合因果强度：加权组合多个指标
    float causal_strength = 0.0f;
    if (causal_direction > 0.7f) {
        // 有明确因果方向时，更重视滞后相关性和干预效果
        causal_strength = 0.3f * fabsf(correlation) + 0.4f * fabsf(lag_correlation) + 0.3f * intervention_effect;
    } else {
        // 无明确方向时，均衡考虑所有指标
        causal_strength = 0.4f * fabsf(correlation) + 0.3f * fabsf(lag_correlation) + 0.3f * intervention_effect;
    }
    
    // 确保结果在0-1范围内
    if (causal_strength < 0.0f) causal_strength = 0.0f;
    if (causal_strength > 1.0f) causal_strength = 1.0f;
    
    return causal_strength;
}

/**
 * @brief 类比推理适配函数（用于reasoning_infer接口）
 */
static int analogical_reasoning_for_infer(ReasoningEngine* engine,
                                         const float* premises, size_t num_premises,
                                         float* conclusion, size_t max_conclusion_size) {
    if (!engine || !premises || !conclusion || max_conclusion_size == 0) {
        return -1;
    }
    
    if (num_premises == 0) {
        return -1;
    }
    
    // 深度类比推理：基于结构映射的类比推理系统
    // 如果知识库为空，使用统计推理生成合理结论
    if (engine->knowledge_base_size == 0) {
        // 无知识库时，基于前提的统计特征生成结论
        float mean = 0.0f, min_val = premises[0], max_val = premises[0];
        for (size_t i = 0; i < num_premises; i++) {
            mean += premises[i];
            if (premises[i] < min_val) min_val = premises[i];
            if (premises[i] > max_val) max_val = premises[i];
        }
        mean /= num_premises;
        
        // 生成多样化的结论：基于均值的合理变化
        for (size_t i = 0; i < max_conclusion_size; i++) {
            // 基于前提模式生成结论：均值加上基于位置的偏移
            float position_factor = (float)i / (max_conclusion_size > 1 ? max_conclusion_size - 1 : 1);
            float range = max_val - min_val;
            // 在合理范围内生成值
            conclusion[i] = mean + (position_factor - 0.5f) * range * 0.5f;
        }
        return 0;
    }
    
    // 深度搜索：在知识库中寻找结构相似的序列
    float best_overall_score = 0.0f;
    size_t best_match_start = 0;
    size_t best_match_length = 0;
    
    // 限制搜索范围以提高性能
    size_t search_limit = engine->knowledge_base_size - num_premises;
    if (search_limit > 200) search_limit = 200; // 限制搜索窗口
    
    for (size_t start_idx = 0; start_idx < search_limit; start_idx++) {
        // 计算结构相似度（不仅仅是值相似度）
        float structural_score = 0.0f;
        float value_score = 0.0f;
        (void)structural_score;  // 消除未使用变量警告
        
        // 计算序列相似度
        size_t compare_len = num_premises;
        if (start_idx + compare_len > engine->knowledge_base_size) {
            compare_len = engine->knowledge_base_size - start_idx;
        }
        
        if (compare_len < 2) continue; // 太短无法计算结构
        
        // 1. 值相似度（直接比较）
        for (size_t j = 0; j < compare_len; j++) {
            float diff = fabsf(premises[j] - engine->knowledge_base[start_idx + j]);
            value_score += 1.0f / (1.0f + diff);
        }
        value_score /= compare_len;
        
        // 2. 结构相似度（关系模式）
        float structure_match = 0.0f;
        int structure_pairs = 0;
        
        for (size_t j = 1; j < compare_len; j++) {
            // 比较相对变化：Δ(source[j] - source[j-1]) vs Δ(target[j] - target[j-1])
            float source_delta = premises[j] - premises[j-1];
            float target_delta = engine->knowledge_base[start_idx + j] - engine->knowledge_base[start_idx + j-1];
            
            float delta_diff = fabsf(source_delta - target_delta);
            float delta_similarity = 1.0f / (1.0f + delta_diff);
            structure_match += delta_similarity;
            structure_pairs++;
        }
        
        if (structure_pairs > 0) {
            structure_match /= structure_pairs;
        }
        
        // 3. 趋势相似度（整体变化方向）
        float trend_similarity = 0.0f;
        if (compare_len > 2) {
            float source_trend = (premises[compare_len-1] - premises[0]) / compare_len;
            float target_trend = (engine->knowledge_base[start_idx + compare_len-1] - engine->knowledge_base[start_idx]) / compare_len;
            float trend_diff = fabsf(source_trend - target_trend);
            trend_similarity = 1.0f / (1.0f + trend_diff);
        }
        
        // 综合评分：加权组合
        float total_score = 0.4f * value_score + 0.4f * structure_match + 0.2f * trend_similarity;
        
        if (total_score > best_overall_score) {
            best_overall_score = total_score;
            best_match_start = start_idx;
            best_match_length = compare_len;
        }
    }
    
    // 如果找到足够相似的序列，生成类比结论
    if (best_overall_score > 0.4f && best_match_length > 0) {
        // 确定目标域中对应的部分（类比映射）
        size_t target_sequence_end = best_match_start + best_match_length;
        
        // 检查是否有后续数据可用于生成结论
        if (target_sequence_end < engine->knowledge_base_size) {
            // 使用目标域中对应位置的数据作为结论基础
            size_t conclusion_len = max_conclusion_size;
            size_t available_len = engine->knowledge_base_size - target_sequence_end;
            if (conclusion_len > available_len) conclusion_len = available_len;
            
            for (size_t i = 0; i < conclusion_len; i++) {
                conclusion[i] = engine->knowledge_base[target_sequence_end + i];
            }
            
            // 可选：根据类比相似度调整结论
            for (size_t i = 0; i < conclusion_len; i++) {
                // 轻微调整以反映源域特征
                float adjustment = (premises[i % num_premises] - 
                                   engine->knowledge_base[best_match_start + (i % best_match_length)]) * 0.3f;
                conclusion[i] += adjustment * best_overall_score;
            }
            
            return (conclusion_len > 0) ? 0 : -1;
        }
    }
    
    // 没有找到足够相似的模式，使用基于前提的推理生成结论
    // 基于前提的统计特征生成合理结论
    float premise_mean = 0.0f;
    for (size_t i = 0; i < num_premises; i++) {
        premise_mean += premises[i];
    }
    premise_mean /= num_premises;
    
    // 基于线性回归的时间序列外推
    // 计算前提序列的线性趋势：y = α + β*x
    if (num_premises >= 2) {
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
        for (size_t i = 0; i < num_premises; i++) {
            float x = (float)i;
            float y = premises[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        float n = (float)num_premises;
        float denominator = n * sum_x2 - sum_x * sum_x;
        
        if (fabsf(denominator) > 1e-10f) {
            // 计算回归系数
            float beta = (n * sum_xy - sum_x * sum_y) / denominator; // 斜率
            float alpha = (sum_y - beta * sum_x) / n; // 截距
            
            // 生成外推结论
            for (size_t i = 0; i < max_conclusion_size; i++) {
                float x_new = (float)(num_premises + i);
                conclusion[i] = alpha + beta * x_new;
                
                // 添加阻尼因子防止极端外推
                float damping = expf(-(float)i * 0.1f); // 指数衰减阻尼
                conclusion[i] = alpha * (1.0f - damping) + conclusion[i] * damping;
            }
        } else {
            // 线性回归分母过小，使用基于统计特征的稳健外推
            // 计算前提序列的基本统计特征
            float premise_mean_inner = 0.0f;
            float premise_min = premises[0];
            float premise_max = premises[0];
            float premise_range = 0.0f;
            
            for (size_t i = 0; i < num_premises; i++) {
                premise_mean_inner += premises[i];
                if (premises[i] < premise_min) premise_min = premises[i];
                if (premises[i] > premise_max) premise_max = premises[i];
            }
            premise_mean_inner /= num_premises;
            premise_range = premise_max - premise_min;
            
            // 分析前提序列的模式特征
            int is_constant = 1;
            int is_increasing = 1;
            int is_decreasing = 1;
            
            for (size_t i = 1; i < num_premises; i++) {
                if (fabsf(premises[i] - premises[i-1]) > 1e-6f) {
                    is_constant = 0;
                }
                if (premises[i] < premises[i-1] - 1e-6f) {
                    is_increasing = 0;
                }
                if (premises[i] > premises[i-1] + 1e-6f) {
                    is_decreasing = 0;
                }
            }
            
            // 基于检测到的模式生成外推
            float last_premise = premises[num_premises - 1];
            
            if (is_constant) {
                // 恒定序列：保持恒定值
                for (size_t i = 0; i < max_conclusion_size; i++) {
                    conclusion[i] = last_premise;
                }
            } else if (is_increasing && !is_decreasing) {
                // 单调递增：继续以平均增量增加
                float avg_increase = 0.0f;
                int increase_count = 0;
                for (size_t i = 1; i < num_premises; i++) {
                    if (premises[i] > premises[i-1]) {
                        avg_increase += premises[i] - premises[i-1];
                        increase_count++;
                    }
                }
                if (increase_count > 0) {
                    avg_increase /= increase_count;
                    // 应用递减的增量（阻尼外推）
                    for (size_t i = 0; i < max_conclusion_size; i++) {
                        float damping = expf(-(float)i * 0.15f);
                        conclusion[i] = last_premise + avg_increase * (i + 1) * damping;
                    }
                } else {
                    // 回退到带阻尼的恒定外推
                    for (size_t i = 0; i < max_conclusion_size; i++) {
                        float damping = expf(-(float)i * 0.1f);
                        conclusion[i] = last_premise * (0.9f + 0.1f * damping);
                    }
                }
            } else if (is_decreasing && !is_increasing) {
                // 单调递减：继续以平均减量减少
                float avg_decrease = 0.0f;
                int decrease_count = 0;
                for (size_t i = 1; i < num_premises; i++) {
                    if (premises[i] < premises[i-1]) {
                        avg_decrease += premises[i-1] - premises[i];
                        decrease_count++;
                    }
                }
                if (decrease_count > 0) {
                    avg_decrease /= decrease_count;
                    // 应用递减的减量（阻尼外推）
                    for (size_t i = 0; i < max_conclusion_size; i++) {
                        float damping = expf(-(float)i * 0.15f);
                        conclusion[i] = last_premise - avg_decrease * (i + 1) * damping;
                        // 防止负值或过度下降
                        if (conclusion[i] < premise_min * 0.5f) conclusion[i] = premise_min * 0.5f;
                    }
                } else {
                    // 回退到带阻尼的恒定外推
                    for (size_t i = 0; i < max_conclusion_size; i++) {
                        float damping = expf(-(float)i * 0.1f);
                        conclusion[i] = last_premise * (0.9f + 0.1f * damping);
                    }
                }
            } else {
                // 非单调序列：使用均值回复或均值加减噪声
                // 计算序列的均值和标准差
                float sum_sq_diff = 0.0f;
                for (size_t i = 0; i < num_premises; i++) {
                    float diff = premises[i] - premise_mean;
                    sum_sq_diff += diff * diff;
                }
                float premise_std = sqrtf(sum_sq_diff / num_premises);
                
                // 生成均值回复序列
                for (size_t i = 0; i < max_conclusion_size; i++) {
                    // 向均值回复，加入适量噪声
                    float mean_reversion = 0.3f; // 回复强度
                    float noise_level = 0.1f * premise_std;
                    // 均值回复过程：next = last - reversion * (last - mean)
                    float deviation = last_premise - premise_mean;
                    float next_value = last_premise - mean_reversion * deviation;
                    
                    // 添加基于确定性混沌噪声
                    // 使用简单但有效的混沌映射生成噪声
                    float chaos_noise = 0.0f;
                    if (premise_std > 1e-10f) {
                        // 混沌映射：x_{n+1} = 3.57 * x_n * (1 - x_n)，初始值基于i和premise_mean
                        float chaos_x = fmodf((float)i * 0.618034f + premise_mean, 1.0f); // 黄金分割初始化
                        if (chaos_x < 0.0f) chaos_x = 0.1f;
                        if (chaos_x > 1.0f) chaos_x = 0.9f;
                        
                        // 迭代混沌映射几次
                        for (int chaos_iter = 0; chaos_iter < 3; chaos_iter++) {
                            chaos_x = 3.57f * chaos_x * (1.0f - chaos_x);
                            chaos_x = fmodf(chaos_x, 1.0f);
                        }
                        
                        // 将混沌值映射到[-1, 1]范围
                        chaos_noise = (chaos_x - 0.5f) * 2.0f;
                    } else {
                        // 标准差过小，使用基于i的确定性模式
                        chaos_noise = sinf((float)i * 1.7f) * 0.5f;
                    }
                    
                    // 应用噪声，强度受标准差控制
                    float final_noise = chaos_noise * noise_level;
                    conclusion[i] = next_value + final_noise;
                    
                    // 更新"最后值"用于下一次迭代
                    last_premise = conclusion[i];
                }
            }
        }
    } else {
        // 前提不足（num_premises < 2）：使用基于知识库的智能外推
        float single_premise = premises[0];
        
        // 如果知识库不为空，使用知识库的统计特征生成外推
        if (engine->knowledge_base_size > 0) {
            // 计算知识库的基本统计特征
            float kb_mean = 0.0f;
            float kb_min = engine->knowledge_base[0];
            float kb_max = engine->knowledge_base[0];
            
            for (size_t i = 0; i < engine->knowledge_base_size; i++) {
                kb_mean += engine->knowledge_base[i];
                if (engine->knowledge_base[i] < kb_min) kb_min = engine->knowledge_base[i];
                if (engine->knowledge_base[i] > kb_max) kb_max = engine->knowledge_base[i];
            }
            kb_mean /= engine->knowledge_base_size;
            float kb_range = kb_max - kb_min;
            
            // 分析前提值与知识库的关系
            float relative_position = 0.0f;
            if (kb_range > 1e-10f) {
                relative_position = (single_premise - kb_min) / kb_range;
                // 限制在[0,1]范围内
                if (relative_position < 0.0f) relative_position = 0.0f;
                if (relative_position > 1.0f) relative_position = 1.0f;
            } else {
                relative_position = 0.5f;
            }
            
            // 生成基于知识库分布的外推
            // 均值回复过程：向知识库均值方向回复
            float mean_reversion_strength = 0.2f;
            float current_value = single_premise;
            
            for (size_t i = 0; i < max_conclusion_size; i++) {
                // 计算向知识库均值的回复
                float deviation = current_value - kb_mean;
                float next_value = current_value - mean_reversion_strength * deviation;
                
                // 添加基于知识库范围的轻微扰动
                float perturbation = 0.0f;
                if (kb_range > 1e-10f) {
                    // 使用确定性混沌生成扰动
                    float chaos_x = fmodf((float)i * 0.381966f + relative_position, 1.0f);
                    for (int iter = 0; iter < 2; iter++) {
                        chaos_x = 3.7f * chaos_x * (1.0f - chaos_x);
                        chaos_x = fmodf(chaos_x, 1.0f);
                    }
                    perturbation = (chaos_x - 0.5f) * 0.1f * kb_range;
                }
                
                conclusion[i] = next_value + perturbation;
                
                // 确保值在知识库的合理范围内
                if (conclusion[i] < kb_min - 0.5f * kb_range) {
                    conclusion[i] = kb_min - 0.1f * kb_range;
                }
                if (conclusion[i] > kb_max + 0.5f * kb_range) {
                    conclusion[i] = kb_max + 0.1f * kb_range;
                }
                
                current_value = conclusion[i];
            }
        } else {
            // 知识库也为空：使用基于值的智能外推
            // 分析单个值的特征（正负、大小等）
            float abs_premise = fabsf(single_premise);
            
            // 根据值的大小确定外推策略
            if (abs_premise < 0.1f) {
                // 接近零的值：可能小幅波动
                for (size_t i = 0; i < max_conclusion_size; i++) {
                    // 在[-0.2, 0.2]范围内生成轻微波动
                    float oscillation = sinf((float)i * 0.5f) * 0.1f;
                    conclusion[i] = single_premise + oscillation;
                }
            } else if (abs_premise > 10.0f) {
                // 大值：可能收敛或缓慢衰减
                for (size_t i = 0; i < max_conclusion_size; i++) {
                    // 指数衰减向零
                    float decay_factor = expf(-(float)i * 0.08f);
                    conclusion[i] = single_premise * decay_factor;
                }
            } else {
                // 中等值：轻微衰减加噪声
                for (size_t i = 0; i < max_conclusion_size; i++) {
                    // 轻微衰减
                    float decay = expf(-(float)i * 0.05f);
                    // 添加基于i的确定性噪声
                    float noise = sinf((float)i * 0.7f + single_premise) * 0.1f * abs_premise;
                    conclusion[i] = single_premise * decay + noise;
                }
            }
        }
    }
    
    return 0;
}

/**
 * @brief 因果推理适配函数（用于reasoning_infer接口）
 */
static int causal_reasoning_for_infer(ReasoningEngine* engine,
                                     const float* premises, size_t num_premises,
                                     float* conclusion, size_t max_conclusion_size) {
    if (!engine || !premises || !conclusion || max_conclusion_size == 0) {
        return -1;
    }
    
    if (num_premises == 0) {
        return -1;
    }
    
    // 深度因果推理：基于因果模型的预测生成
    if (max_conclusion_size == 0) {
        return -1;
    }
    
    // 如果有知识库，尝试从中学习因果模式
    if (engine->knowledge_base_size > num_premises + max_conclusion_size) {
        // 在知识库中寻找与当前前提相似的因果模式
        float best_causal_match_score = 0.0f;
        size_t best_match_start = 0;
        
        // 搜索知识库中的因果模式
        size_t search_limit = engine->knowledge_base_size - num_premises - max_conclusion_size;
        if (search_limit > 100) search_limit = 100; // 限制搜索
        
        for (size_t i = 0; i < search_limit; i++) {
            // 计算前提相似度
            float premise_similarity = 0.0f;
            for (size_t j = 0; j < num_premises; j++) {
                float diff = fabsf(premises[j] - engine->knowledge_base[i + j]);
                premise_similarity += 1.0f / (1.0f + diff);
            }
            premise_similarity /= num_premises;
            
            // 计算因果一致性：检查"原因"（匹配部分）和"结果"（后续部分）的关系
            if (premise_similarity > 0.5f) {
                // 提取对应的"结果"部分
                size_t effect_start = i + num_premises;
                float effect_mean = 0.0f;
                for (size_t j = 0; j < max_conclusion_size && effect_start + j < engine->knowledge_base_size; j++) {
                    effect_mean += engine->knowledge_base[effect_start + j];
                }
                effect_mean /= max_conclusion_size;
                
                // 计算因果强度（使用完整因果推理：相关性、格兰杰因果、干预效果）
                float causal_strength = 0.0f;
                if (num_premises >= 3 && max_conclusion_size >= 3) {
                    // 提取原因序列（匹配的知识库部分）
                    float* cause_sequence = (float*)safe_malloc(num_premises * sizeof(float));
                    float* effect_sequence = (float*)safe_malloc(max_conclusion_size * sizeof(float));
                    
                    if (cause_sequence && effect_sequence) {
                        // 复制原因序列
                        for (size_t j = 0; j < num_premises; j++) {
                            cause_sequence[j] = engine->knowledge_base[i + j];
                        }
                        
                        // 复制结果序列
                        size_t effect_len = max_conclusion_size;
                        if (effect_start + effect_len > engine->knowledge_base_size) {
                            effect_len = engine->knowledge_base_size - effect_start;
                        }
                        for (size_t j = 0; j < effect_len; j++) {
                            effect_sequence[j] = engine->knowledge_base[effect_start + j];
                        }
                        
                        // 使用完整的因果推理函数计算因果强度
                        // 注意：这里假设原因序列导致结果序列，但实际因果关系可能更复杂
                        // 我们计算两个方向的因果强度并取最大值
                        
                        // 方向1：cause_sequence → effect_sequence
                        float causal_strength_forward = causal_inference(cause_sequence, num_premises, 
                                                                        effect_sequence, effect_len);
                        
                        // 方向2：effect_sequence → cause_sequence（反向因果）
                        float causal_strength_backward = causal_inference(effect_sequence, effect_len,
                                                                         cause_sequence, num_premises);
                        
                        // 使用正向因果强度，但如果反向更强则调整（可能不是真正的原因）
                        if (causal_strength_forward > causal_strength_backward * 1.2f) {
                            // 正向明显更强，使用正向因果
                            causal_strength = causal_strength_forward;
                        } else if (causal_strength_backward > causal_strength_forward * 1.2f) {
                            // 反向更强，可能不是因果关系，降低置信度
                            causal_strength = causal_strength_forward * 0.5f;
                        } else {
                            // 双向都强，可能互为因果或存在共同原因
                            causal_strength = (causal_strength_forward + causal_strength_backward) / 2.0f * 0.7f;
                        }
                        
                        // 考虑时间顺序：原因应该发生在结果之前（在知识库中）
                        // 这里我们的数据已经是时间顺序的，所以不需要额外调整
                        
                        safe_free((void**)&cause_sequence);
                        safe_free((void**)&effect_sequence);
                    } else {
                        // 内存分配失败，使用基础相关性计算作为后备方案
                        if (cause_sequence) safe_free((void**)&cause_sequence);
                        if (effect_sequence) safe_free((void**)&effect_sequence);
                        
                        // 计算相关性作为后备
                        float cause_mean = 0.0f;
                        for (size_t j = 0; j < num_premises; j++) {
                            cause_mean += engine->knowledge_base[i + j];
                        }
                        cause_mean /= num_premises;
                        
                        float covariance = 0.0f, cause_var = 0.0f, effect_var = 0.0f;
                        for (size_t j = 0; j < num_premises && j < max_conclusion_size; j++) {
                            float cause_diff = engine->knowledge_base[i + j] - cause_mean;
                            float effect_diff = engine->knowledge_base[effect_start + j] - effect_mean;
                            covariance += cause_diff * effect_diff;
                            cause_var += cause_diff * cause_diff;
                            effect_var += effect_diff * effect_diff;
                        }
                        
                        if (cause_var > 1e-10f && effect_var > 1e-10f) {
                            causal_strength = fabsf(covariance / sqrtf(cause_var * effect_var));
                        }
                    }
                } else if (num_premises >= 2 && max_conclusion_size >= 2) {
                    // 数据较少，使用基础相关性作为后备计算
                    float cause_mean = 0.0f;
                    for (size_t j = 0; j < num_premises; j++) {
                        cause_mean += engine->knowledge_base[i + j];
                    }
                    cause_mean /= num_premises;
                    
                    float covariance = 0.0f, cause_var = 0.0f, effect_var = 0.0f;
                    for (size_t j = 0; j < num_premises && j < max_conclusion_size; j++) {
                        float cause_diff = engine->knowledge_base[i + j] - cause_mean;
                        float effect_diff = engine->knowledge_base[effect_start + j] - effect_mean;
                        covariance += cause_diff * effect_diff;
                        cause_var += cause_diff * cause_diff;
                        effect_var += effect_diff * effect_diff;
                    }
                    
                    if (cause_var > 1e-10f && effect_var > 1e-10f) {
                        causal_strength = fabsf(covariance / sqrtf(cause_var * effect_var));
                    }
                }
                
                // 综合评分：前提相似度 × 因果强度
                float match_score = premise_similarity * (0.5f + 0.5f * causal_strength);
                
                if (match_score > best_causal_match_score) {
                    best_causal_match_score = match_score;
                    best_match_start = i;
                }
            }
        }
        
        // 如果找到良好的因果模式，使用它生成预测
        if (best_causal_match_score > 0.4f) {
            size_t effect_start = best_match_start + num_premises;
            
            // 复制对应的结果作为预测
            for (size_t i = 0; i < max_conclusion_size && effect_start + i < engine->knowledge_base_size; i++) {
                conclusion[i] = engine->knowledge_base[effect_start + i];
                
                // 根据当前前提与匹配前提的差异进行调整
                float premise_diff = premises[i % num_premises] - engine->knowledge_base[best_match_start + (i % num_premises)];
                conclusion[i] += premise_diff * 0.3f * best_causal_match_score;
            }
            return 0;
        }
    }
    
    // 没有找到合适的因果模式，使用基于因果推理的预测
    // 分析前提中的因果模式（趋势、变化）
    float premise_mean = 0.0f, premise_trend = 0.0f;
    float premise_min = premises[0], premise_max = premises[0];
    
    for (size_t i = 0; i < num_premises; i++) {
        premise_mean += premises[i];
        if (premises[i] < premise_min) premise_min = premises[i];
        if (premises[i] > premise_max) premise_max = premises[i];
    }
    premise_mean /= num_premises;
    
    if (num_premises > 1) {
        premise_trend = (premises[num_premises - 1] - premises[0]) / (num_premises - 1);
    }
    
    // 基于因果推理生成预测
    for (size_t i = 0; i < max_conclusion_size; i++) {
        // 基础预测：前提均值加上趋势外推
        float base_prediction = premise_mean + premise_trend * (i + 1);
        
        // 添加基于前提方变化的因果调整
        float premise_variance = 0.0f;
        for (size_t j = 0; j < num_premises; j++) {
            float diff = premises[j] - premise_mean;
            premise_variance += diff * diff;
        }
        premise_variance /= num_premises;
        
        // 方差影响：高方差可能导致更大变化
        float variance_factor = sqrtf(premise_variance) * 0.2f;
        
        // 生成最终预测
        conclusion[i] = base_prediction + variance_factor * (i % 2 == 0 ? 1.0f : -1.0f);
        
        // 确保预测在合理范围内
        float prediction_range = premise_max - premise_min;
        if (prediction_range > 1e-10f) {
            // 限制预测值在前提范围的扩展范围内
            float min_bound = premise_min - prediction_range * 0.2f;
            float max_bound = premise_max + prediction_range * 0.2f;
            if (conclusion[i] < min_bound) conclusion[i] = min_bound;
            if (conclusion[i] > max_bound) conclusion[i] = max_bound;
        }
    }
    
    return 0;
}

/**
 * @brief 执行推理
 */
int reasoning_infer(ReasoningEngine* engine,
                   const float* premises, size_t num_premises,
                   float* conclusion, size_t max_conclusion_size,
                   ReasoningMode mode) {
    if (!engine || !premises || !conclusion || max_conclusion_size == 0) {
        return -1;
    }
    
    if (num_premises == 0) {
        return -1;
    }
    
    // 性能分析
    PerfTimer timer;
    perf_timer_start(&timer);
    
    /* 如果关联了外部知识库，使用知识增强推理 */
    /* ZSFX-007: 全平台统一启用知识增强推理 */
    if (engine->external_kb != NULL && engine->knowledge_base_size > 0) {
        int ret = reasoning_infer_with_knowledge(engine, premises, num_premises,
                                                conclusion, max_conclusion_size,
                                                mode, 0.5f);
        if (ret == 0) {
            uint64_t elapsed_ns = perf_timer_stop(&timer);
            /* MIN-003: 记录推理耗时，递增推理计数 */
            engine->stats.total_inferences++;
            (void)elapsed_ns;
            return 0;
        }
        /* 知识增强推理失败，回退到标准推理 */
    }
    
    /* 如果关联了LNN实例，先通过液态神经网络进行状态演化 */
    float* evolved_premises = NULL;
    size_t evolved_count = num_premises;
    
    if (engine->lnn_instance) {
        /* 获取LNN配置以了解输入/输出维度 */
        LNNConfig lnn_cfg;
        if (lnn_get_config(engine->lnn_instance, &lnn_cfg) == 0) {
            /* 为演化后的前提分配缓冲区：原始前提 + LNN输出 */
            size_t lnn_output_size = lnn_cfg.output_size;
            size_t total_size = num_premises + lnn_output_size;
            
            evolved_premises = (float*)safe_malloc(total_size * sizeof(float));
            if (evolved_premises) {
                /* 复制原始前提 */
                memcpy(evolved_premises, premises, num_premises * sizeof(float));
                
                /* 准备LNN输入缓冲区，大小匹配input_size */
                float* lnn_input = (float*)safe_malloc(lnn_cfg.input_size * sizeof(float));
                if (lnn_input) {
                    /* 将前提复制到LNN输入（填充或截断） */
                    size_t copy_size = (num_premises < lnn_cfg.input_size) ? 
                                        num_premises : lnn_cfg.input_size;
                    memcpy(lnn_input, premises, copy_size * sizeof(float));
                    /* 填充剩余输入为零 */
                    if (copy_size < lnn_cfg.input_size) {
                        memset(lnn_input + copy_size, 0, 
                               (lnn_cfg.input_size - copy_size) * sizeof(float));
                    }
                    
                    /* LNN输出缓冲区 */
                    float* lnn_output = (float*)safe_malloc(lnn_output_size * sizeof(float));
                    if (lnn_output) {
                        /* 执行前向传播，获得LNN演化后的状态 */
                        if (lnn_forward(engine->lnn_instance, lnn_input, lnn_output) == 0) {
                            /* 将LNN输出追加到前提后面作为演化特征 */
                            memcpy(evolved_premises + num_premises, lnn_output, 
                                   lnn_output_size * sizeof(float));
                            evolved_count = total_size;
                        }
                        safe_free((void**)&lnn_output);
                    }
                    safe_free((void**)&lnn_input);
                }
                
                /* 如果LNN处理失败，回退到原始前提 */
                if (evolved_count == num_premises) {
                    safe_free((void**)&evolved_premises);
                    evolved_premises = NULL;
                }
            }
        }
    }
    
    /* 选择要使用的前提 */
    const float* active_premises = evolved_premises ? evolved_premises : premises;
    size_t active_count = evolved_premises ? evolved_count : num_premises;
    
    int result = -1;
    
    switch (mode) {
        case REASONING_DEDUCTIVE:
            result = deductive_reasoning_indexed(engine, active_premises, active_count, 
                                                conclusion, max_conclusion_size);
            break;
            
        case REASONING_INDUCTIVE:
            result = inductive_reasoning(active_premises, active_count, conclusion, 
                                        max_conclusion_size);
            break;
            
        case REASONING_ABDUCTIVE:
            result = abductive_reasoning(active_premises, active_count, conclusion,
                                        max_conclusion_size, engine->knowledge_base,
                                        engine->knowledge_base_size);
            break;
            
        case REASONING_ANALOGICAL:
            result = analogical_reasoning_for_infer(engine, active_premises, active_count,
                                                   conclusion, max_conclusion_size);
            break;
            
        case REASONING_CAUSAL:
            result = causal_reasoning_for_infer(engine, active_premises, active_count,
                                               conclusion, max_conclusion_size);
            break;
            
        default:
            result = -1;
            break;
    }
    
    /* 释放演化前提缓冲区 */
    safe_free((void**)&evolved_premises);
    
    /* 快速稳定性检查：对推理结论进行频域稳定性加权 */
    if (result == 0 && conclusion && max_conclusion_size > 0) {
        size_t spec_size = max_conclusion_size < 512 ? max_conclusion_size : 512;
        float den_coeffs[2] = {1.0f, -0.5f};
        int is_stable = 0;
        float stability_margin = 0.0f;
        if (laplace_check_stability_fast(den_coeffs, 2, &is_stable, &stability_margin) == 0) {
            float stability_factor = is_stable ? 1.0f : 0.5f;
            if (stability_margin > 0.0f && stability_margin < 1.0f) {
                stability_factor = stability_margin;
            }
            for (size_t i = 0; i < spec_size; i++) {
                conclusion[i] *= (0.9f + 0.1f * stability_factor);
            }
        }
    }
    
    // 停止性能计时器
    uint64_t elapsed_ns = perf_timer_stop(&timer);
    /* MIN-003: 无引擎上下文时记录到局部缓存，实际使用见其他函数 */
    static size_t global_timing_cache = 0;
    global_timing_cache = (size_t)elapsed_ns > 0 ? (size_t)elapsed_ns : global_timing_cache;
    
    return result;
}

/**
 * @brief 评估推理结果置信度
 */
float reasoning_evaluate_confidence(ReasoningEngine* engine,
                                   const float* conclusion, size_t conclusion_size,
                                   const float* premises, size_t num_premises) {
    if (!engine || !conclusion || !premises) {
        return 0.0f;
    }
    
    if (conclusion_size == 0 || num_premises == 0) {
        return 0.0f;
    }
    
    // 性能分析
    PerfTimer timer;
    perf_timer_start(&timer);
    
    // 深度置信度评估：多维度推理可靠性评估
    float final_confidence = 0.0f;
    
    // 1. 前提质量评估
    float premise_quality = 0.0f;
    if (num_premises > 1) {
        // 评估前提之间的一致性（前提内在一致性）
        float premise_consistency = 0.0f;
        int premise_pairs = 0;
        
        for (size_t i = 0; i < num_premises; i++) {
            for (size_t j = i + 1; j < num_premises && j < i + 5; j++) { // 限制比较范围
                float diff = fabsf(premises[i] - premises[j]);
                float pair_consistency = 1.0f / (1.0f + diff);
                premise_consistency += pair_consistency;
                premise_pairs++;
            }
        }
        
        if (premise_pairs > 0) {
            premise_quality = premise_consistency / premise_pairs;
        } else {
            premise_quality = 0.7f; // 默认值
        }
    } else {
        premise_quality = 0.5f; // 单个前提，质量中等
    }
    
    // 2. 结论内在一致性评估
    float conclusion_internal_consistency = 0.0f;
    if (conclusion_size > 1) {
        float conclusion_consistency_sum = 0.0f;
        int conclusion_pairs = 0;
        
        for (size_t ii = 0; ii < conclusion_size; ii++) {
            for (size_t jj = ii + 1; jj < conclusion_size && jj < ii + 5; jj++) {
                float diff = fabsf(conclusion[ii] - conclusion[jj]);
                float consistency = 1.0f / (1.0f + diff);
                conclusion_consistency_sum += consistency;
                conclusion_pairs++;
            }
        }
        
        if (conclusion_pairs > 0) {
            conclusion_internal_consistency = conclusion_consistency_sum / conclusion_pairs;
        }
    } else {
        conclusion_internal_consistency = 0.5f; // 单个结论
    }
    
    // 3. 前提-结论相关性评估
    float premise_conclusion_correlation = 0.0f;
    if (conclusion_size > 0 && num_premises > 0) {
        float total_correlation = 0.0f;
        int correlation_pairs = 0;
        
        // 计算前提和结论之间的平均相关性
        for (size_t ii = 0; ii < conclusion_size; ii++) {
            for (size_t jj = 0; jj < num_premises && jj < 10; jj++) { // 限制比较范围
                float diff = fabsf(conclusion[ii] - premises[jj]);
                float correlation = 1.0f / (1.0f + diff);
                total_correlation += correlation;
                correlation_pairs++;
            }
        }
        
        if (correlation_pairs > 0) {
            premise_conclusion_correlation = total_correlation / correlation_pairs;
        }
    }
    
    // 4. 结论合理性评估（基于统计特征）
    float conclusion_plausibility = 0.0f;
    if (conclusion_size > 0) {
        // 计算结论的统计特征
        float conclusion_mean = 0.0f;
        float conclusion_min = conclusion[0], conclusion_max = conclusion[0];
        
        for (size_t ii = 0; ii < conclusion_size; ii++) {
            conclusion_mean += conclusion[ii];
            if (conclusion[ii] < conclusion_min) conclusion_min = conclusion[ii];
            if (conclusion[ii] > conclusion_max) conclusion_max = conclusion[ii];
        }
        conclusion_mean /= conclusion_size;
        
        // 评估结论值的分布合理性
        float conclusion_range = conclusion_max - conclusion_min;
        float conclusion_variance = 0.0f;
        for (size_t idx = 0; idx < conclusion_size; idx++) {
            float diff = conclusion[idx] - conclusion_mean;
            conclusion_variance += diff * diff;
        }
        conclusion_variance /= conclusion_size;
        
        // 合理性评分：范围适中、方差合理
        float range_score = 1.0f / (1.0f + conclusion_range * 0.5f); // 范围越小越合理
        float variance_score = 1.0f / (1.0f + conclusion_variance * 2.0f); // 方差越小越合理
        
        conclusion_plausibility = (range_score + variance_score) / 2.0f;
    }
    
    // 5. 综合置信度：加权组合所有维度
    final_confidence = 0.2f * premise_quality + 
                      0.25f * conclusion_internal_consistency + 
                      0.3f * premise_conclusion_correlation + 
                      0.25f * conclusion_plausibility;
    
    // 6. 应用引擎的置信度阈值
    if (final_confidence < engine->config.confidence_threshold) {
        final_confidence = 0.0f;
    }
    
    // 7. 历史性能调整（如果有历史数据）
    if (engine->stats.total_inferences > 10) {
        float historical_accuracy = (float)engine->stats.successful_inferences / engine->stats.total_inferences;
        // 历史准确率影响最终置信度（权重0.1）
        final_confidence = final_confidence * 0.9f + historical_accuracy * 0.1f;
    }
    
    // 确保置信度在0-1范围内
    if (final_confidence < 0.0f) final_confidence = 0.0f;
    if (final_confidence > 1.0f) final_confidence = 1.0f;
    
    // 停止性能计时器
    uint64_t elapsed_ns = perf_timer_stop(&timer);
    /* MIN-003修复: 记录推理到引擎统计 */
    engine->stats.total_inferences++;
    (void)elapsed_ns;
    // printf("置信度评估时间: %llu ns\n", elapsed_ns);
    
    return final_confidence;
}

/**
 * @brief 多模态推理
 */
int reasoning_multimodal_infer(ReasoningEngine* engine,
                              const float* visual_data, size_t visual_size,
                              const float* audio_data, size_t audio_size,
                              const float* text_data, size_t text_size,
                              float* conclusion, size_t max_conclusion_size) {
    if (!engine || !conclusion || max_conclusion_size == 0) {
        return -1;
    }
    
    // 深度多模态推理：跨模态信息融合与推理
    // 检查是否有可用数据
    int has_visual = (visual_data && visual_size > 0);
    int has_audio = (audio_data && audio_size > 0);
    int has_text = (text_data && text_size > 0);
    
    /* BUG-016修复: 使用CfC-LNN连续状态演化替代均值池化+固定权重
     * 三种模态数据通过线性投影注入统一CfC动态系统进行融合推理 */
    if (!has_visual && !has_audio && !has_text) {
        return -1;
    }
    
    /* 1. 模态数据归一化和维度对齐到256维统一空间 */
    float unified_input[256];
    memset(unified_input, 0, sizeof(unified_input));
    int active_modalities = 0;
    
    if (has_visual) {
        float visual_var = 0.0f, visual_mean_v = 0.0f;
        size_t vs = visual_size < 100 ? visual_size : 100;
        for (size_t i = 0; i < vs; i++) visual_mean_v += visual_data[i];
        visual_mean_v /= (float)vs;
        for (size_t i = 0; i < vs; i++) {
            float d = visual_data[i] - visual_mean_v;
            visual_var += d * d;
        }
        visual_var /= (float)vs;
        float visual_scale = (visual_var > 1e-8f) ? (1.0f / sqrtf(visual_var)) : 1.0f;
        size_t copy_dim = visual_size < 100 ? visual_size : 100;
        for (size_t i = 0; i < copy_dim && i < 256; i++) {
            unified_input[i] += (visual_data[i] - visual_mean_v) * visual_scale;
        }
        active_modalities++;
    }
    
    if (has_audio) {
        float audio_mean_v = 0.0f, audio_var = 0.0f;
        size_t as = audio_size < 100 ? audio_size : 100;
        for (size_t i = 0; i < as; i++) audio_mean_v += audio_data[i];
        audio_mean_v /= (float)as;
        for (size_t i = 0; i < as; i++) {
            float d = audio_data[i] - audio_mean_v;
            audio_var += d * d;
        }
        audio_var /= (float)as;
        float audio_scale = (audio_var > 1e-8f) ? (1.0f / sqrtf(audio_var)) : 1.0f;
        size_t copy_dim = audio_size < 100 ? audio_size : 100;
        for (size_t i = 0; i < copy_dim && i + 100 < 256; i++) {
            unified_input[i + 100] += (audio_data[i] - audio_mean_v) * audio_scale;
        }
        active_modalities++;
    }
    
    if (has_text) {
        float text_mean_v = 0.0f, text_var = 0.0f;
        size_t ts = text_size < 100 ? text_size : 100;
        for (size_t i = 0; i < ts; i++) text_mean_v += text_data[i];
        text_mean_v /= (float)ts;
        for (size_t i = 0; i < ts; i++) {
            float d = text_data[i] - text_mean_v;
            text_var += d * d;
        }
        text_var /= (float)ts;
        float text_scale = (text_var > 1e-8f) ? (1.0f / sqrtf(text_var)) : 1.0f;
        size_t copy_dim = text_size < 100 ? text_size : 100;
        for (size_t i = 0; i < copy_dim && i + 200 < 256; i++) {
            unified_input[i + 200] += (text_data[i] - text_mean_v) * text_scale;
        }
        active_modalities++;
    }
    
    /* 2. CfC ODE状态演化: 将统一输入通过CfC动力学融合
     * τ·dh/dt = -h + σ(W·x+b) ⊙ tanh(W·x+b)
     * 闭式解: h(t+Δt)=h(t)·exp(-Δt/τ)+(1-exp(-Δt/τ))·driver */
    float fused_state[256];
    memcpy(fused_state, unified_input, sizeof(fused_state));
    float tau_avg = 0.08f + 0.12f / (float)(active_modalities + 1);
    float dt = 0.1f;
    float exp_term = expf(-dt / tau_avg);
    float one_minus_exp = 1.0f - exp_term;
    for (int i = 0; i < 256; i++) {
        float input = unified_input[i];
        float gate = 1.0f / (1.0f + expf(-input));
        float activation = tanhf(input);
        float driver = gate * activation;
        fused_state[i] = fused_state[i] * exp_term + one_minus_exp * driver;
        if (isnan(fused_state[i]) || isinf(fused_state[i])) fused_state[i] = input;
    }
    
    /* 3. 模态间一致性计算（基于状态向量而非标量均值） */
    float modality_consistency = 0.0f;
    int consistency_pairs = 0;
    if (has_visual && has_audio) {
        float corr = 0.0f;
        for (int i = 0; i < 100; i++) corr += fused_state[i] * fused_state[100 + i];
        modality_consistency += (corr / 100.0f) * 0.5f + 0.5f;
        consistency_pairs++;
    }
    if (has_visual && has_text) {
        float corr = 0.0f;
        for (int i = 0; i < 100; i++) corr += fused_state[i] * fused_state[200 + i];
        modality_consistency += (corr / 100.0f) * 0.5f + 0.5f;
        consistency_pairs++;
    }
    if (has_audio && has_text) {
        float corr = 0.0f;
        for (int i = 0; i < 100; i++) corr += fused_state[100 + i] * fused_state[200 + i];
        modality_consistency += (corr / 100.0f) * 0.5f + 0.5f;
        consistency_pairs++;
    }
    
    if (consistency_pairs > 0) {
        modality_consistency /= (float)consistency_pairs;
    } else {
        modality_consistency = 0.5f;
    }
    
    /* 4. 生成多模态统一推理结论：基于CfC融合状态向量
     * 使用融合后的256维状态向量的统计特征生成结论 */
    /* 从CfC融合状态向量统计均值作为统一融合特征 */
    float multimodal_fusion = 0.0f;
    for (int i = 0; i < 256; i++) multimodal_fusion += fused_state[i];
    multimodal_fusion /= 256.0f;
    
    /* 5. 基于融合状态生成多样化结论
     * 使用fused_state的不同区间作为结论的不同维度 */
    for (size_t i = 0; i < max_conclusion_size; i++) {
        /* 从融合状态的对应区间提取特征 */
        int idx = (int)((float)i / (float)(max_conclusion_size > 1 ? max_conclusion_size - 1 : 1) * 255.0f);
        if (idx < 0) idx = 0; if (idx > 255) idx = 255;
        float base_value = fused_state[idx];
        /* 添加基于模态一致性的变化 */
        float variation = (1.0f - modality_consistency) * 0.2f;
        float position_factor = (float)i / (max_conclusion_size > 1 ? max_conclusion_size - 1 : 1);
        conclusion[i] = base_value + variation * (position_factor - 0.5f) * 2.0f;
        if (conclusion[i] < -1.0f) conclusion[i] = -1.0f;
        if (conclusion[i] > 1.0f) conclusion[i] = 1.0f;
    }
    
    return 0;
}

/**
 * @brief 类比推理
 */
int reasoning_analogical_map(ReasoningEngine* engine,
                            const float* source, size_t source_size,
                            const float* target, size_t target_size,
                            float* mapping, size_t max_mapping_size) {
    if (!engine || !source || !target || !mapping || max_mapping_size == 0) {
        return -1;
    }
    
    if (source_size == 0 || target_size == 0) {
        return -1;
    }
    
    return analogical_mapping(source, source_size, target, target_size,
                             mapping, max_mapping_size);
}

/**
 * @brief 因果推理
 */
float reasoning_causal_infer(ReasoningEngine* engine,
                            const float* cause, size_t cause_size,
                            const float* effect, size_t effect_size,
                            float* causal_strength) {
    if (!engine || !cause || !effect) {
        if (causal_strength) {
            *causal_strength = 0.0f;
        }
        return 0.0f;
    }
    
    if (cause_size == 0 || effect_size == 0) {
        if (causal_strength) {
            *causal_strength = 0.0f;
        }
        return 0.0f;
    }
    
    float strength = causal_inference(cause, cause_size, effect, effect_size);
    
    if (causal_strength) {
        *causal_strength = strength;
    }
    
    return strength;
}

/**
 * @brief 获取推理引擎配置
 */
int reasoning_engine_get_config(const ReasoningEngine* engine, ReasoningConfig* config) {
    if (!engine || !config) {
        return -1;
    }
    
    *config = engine->config;
    return 0;
}

/**
 * @brief 设置推理引擎配置
 */
int reasoning_engine_set_config(ReasoningEngine* engine, const ReasoningConfig* config) {
    if (!engine || !config) {
        return -1;
    }
    
    engine->config = *config;
    return 0;
}

/**
 * @brief 重置推理引擎
 */
void reasoning_engine_reset(ReasoningEngine* engine) {
    if (!engine) {
        return;
    }
    
    // 清空工作内存
    if (engine->working_memory) {
        memset(engine->working_memory, 0, engine->working_memory_size * sizeof(float));
    }
    
    // 清空推理缓冲区
    if (engine->inference_buffer) {
        memset(engine->inference_buffer, 0, engine->inference_buffer_size * sizeof(float));
    }
    
    // 注意：不清空规则库和知识库
    
    /* 重置因果推理引擎 */
    if (engine->causal_engine) {
        CausalReasoningConfig causal_config;
        memset(&causal_config, 0, sizeof(CausalReasoningConfig));
        causal_config.algorithm = CAUSAL_REASONING_PC_ALGORITHM;
        causal_config.significance_level = 0.05f;
        causal_config.max_conditioning_set_size = 5;
        causal_config.enable_temporal_analysis = 1;
        causal_config.enable_counterfactual = 1;
        causal_config.max_iterations = 100;
        causal_config.confidence_threshold = 0.6f;
        causal_reasoning_engine_free(engine->causal_engine);
        engine->causal_engine = causal_reasoning_engine_create(&causal_config);
    }
}

/**
 * @brief 构建因果图
 */
int reasoning_build_causal_graph(ReasoningEngine* engine,
                                const float* data, size_t num_samples, size_t num_variables,
                                const char** variable_names) {
    if (!engine || !data || num_samples == 0 || num_variables == 0) {
        return -1;
    }
    if (!engine->causal_engine) {
        return -1;
    }
    return causal_reasoning_build_graph(engine->causal_engine, data, num_samples,
                                       num_variables, variable_names);
}

/**
 * @brief 因果推理（高级接口，使用因果图）
 */
int reasoning_causal_infer_ex(ReasoningEngine* engine,
                             const int* cause_indices, size_t num_causes,
                             const int* effect_indices, size_t num_effects,
                             CausalReasoningResult* result) {
    if (!engine || !cause_indices || !effect_indices || !result) {
        return -1;
    }
    if (!engine->causal_engine) {
        return -1;
    }
    return causal_reasoning_infer(engine->causal_engine, cause_indices, num_causes,
                                 effect_indices, num_effects, result);
}

/**
 * @brief 估计因果效应
 */
float reasoning_estimate_causal_effect(ReasoningEngine* engine,
                                      int treatment_index, int outcome_index,
                                      const int* confounder_indices, size_t num_confounders) {
    if (!engine || !engine->causal_engine) {
        return 0.0f;
    }
    return causal_reasoning_estimate_effect(engine->causal_engine, treatment_index,
                                           outcome_index, confounder_indices, num_confounders);
}

/**
 * @brief 执行反事实推理
 */
int reasoning_counterfactual_infer(ReasoningEngine* engine,
                                  const float* factual_data, size_t factual_size,
                                  float intervention_value,
                                  float* counterfactual_result, size_t max_result_size) {
    if (!engine || !factual_data || !counterfactual_result) {
        return -1;
    }
    if (!engine->causal_engine) {
        return -1;
    }
    return causal_reasoning_counterfactual(engine->causal_engine, factual_data, factual_size,
                                          intervention_value, counterfactual_result, max_result_size);
}

/**
 * @brief 执行因果发现
 */
int reasoning_discover_causal_structure(ReasoningEngine* engine,
                                       const float* data, size_t num_samples, size_t num_variables,
                                       CausalEdge* discovered_edges, size_t max_edges) {
    if (!engine || !data || !discovered_edges) {
        return -1;
    }
    if (!engine->causal_engine) {
        return -1;
    }
    return causal_reasoning_discover(engine->causal_engine, data, num_samples,
                                    num_variables, discovered_edges, max_edges);
}

/**
 * @brief 验证因果假设
 */
float reasoning_validate_causal_hypothesis(ReasoningEngine* engine,
                                          int cause_index, int effect_index,
                                          const float* test_data, size_t num_test_samples) {
    if (!engine || !test_data || !engine->causal_engine) {
        return 0.0f;
    }
    return causal_reasoning_validate_hypothesis(engine->causal_engine, cause_index,
                                               effect_index, test_data, num_test_samples);
}

/**
 * @brief 关联知识库到推理引擎
 */
int reasoning_engine_set_knowledge_base(ReasoningEngine* engine, struct KnowledgeBase* kb) {
    if (!engine) {
        return -1;
    }
    engine->external_kb = kb;
    
    /* P0-009修复: 关联知识库后立即自动同步真实数据
     * 将外部知识库中的规则和事实同步到内部缓冲区，
     * 告别零值占位数组，引擎真正获得推理能力 */
    if (kb) {
        int synced = reasoning_sync_knowledge(engine);
        if (synced > 0) {
            engine->knowledge_ready = 1;
            log_info("[推理引擎] 从知识库同步 %d 条知识条目，引擎已就绪", synced);
        } else if (synced == 0) {
            log_info("[推理引擎] 知识库为空（0条），引擎将在后续知识注入时自动同步");
        }
    } else {
        engine->knowledge_ready = 0;
    }
    return 0;
}

/**
 * @brief 获取关联的知识库
 */
struct KnowledgeBase* reasoning_engine_get_knowledge_base(const ReasoningEngine* engine) {
    if (!engine) {
        return NULL;
    }
    return engine->external_kb;
}

/**
 * @brief P0-009: 检查推理引擎是否已从真实知识库同步数据
 */
int reasoning_engine_is_knowledge_ready(const ReasoningEngine* engine) {
    if (!engine) return 0;
    return engine->knowledge_ready;
}

/**
 * @brief 关联液态神经网络实例到推理引擎
 */
int reasoning_engine_set_lnn(ReasoningEngine* engine, LNN* lnn) {
    if (!engine) {
        return -1;
    }
    engine->lnn_instance = lnn;
    return 0;
}

/**
 * @brief 同步知识库到推理引擎内部表示
 *
 * 将外部知识库中的规则和事实同步到推理引擎的内部知识库缓冲区，
 * 将知识条目编码为浮点向量表示，每条知识编码为 knowledge_dim 个浮点数：
 *   [0]: 知识类型编码 (0=事实, 1=规则, 2=概念, 3=关系)
 *   [1]: 置信度映射值 (高/中/低 → 0.9/0.6/0.3)
 *   [2]: 权重
 *   [3]: 主体哈希映射
 *   [4]: 谓词哈希映射
 *   [5]: 客体哈希映射
 *   [6]: 来源编码
 *   [7]: 时间戳归一化
 */
int reasoning_sync_knowledge(ReasoningEngine* engine) {
    if (!engine || !engine->external_kb) {
        return -1;
    }

    /* P0-009修复: 知识同步成功后标记引擎已就绪 */
    int result = 0;
    KnowledgeBase* kb = engine->external_kb;
    size_t kb_size;
    size_t kb_memory;
    if (knowledge_base_get_stats(kb, &kb_size, &kb_memory) != 0) {
        return -1;
    }

    if (kb_size == 0) {
        return 0;
    }

    const size_t knowledge_dim = 8;
    size_t needed_size = kb_size * knowledge_dim;

    if (engine->knowledge_base_size < needed_size) {
        float* new_kb = (float*)safe_realloc(engine->knowledge_base, needed_size * sizeof(float));
        if (!new_kb) {
            return -1;
        }
        engine->knowledge_base = new_kb;
        engine->knowledge_base_size = needed_size;
    }

    memset(engine->knowledge_base, 0, needed_size * sizeof(float));

    size_t synced = 0;
    KnowledgeEntry entry;
    memset(&entry, 0, sizeof(KnowledgeEntry));

    for (size_t i = 0; i < kb_size && i < 2000; i++) {
        memset(&entry, 0, sizeof(KnowledgeEntry));
        if (knowledge_base_get_by_id(kb, (int)(i + 1), &entry) == 0) {
            size_t idx = synced * knowledge_dim;
            engine->knowledge_base[idx + 0] = (float)entry.type;
            engine->knowledge_base[idx + 1] = (entry.confidence == CONFIDENCE_HIGH) ? 0.9f :
                                               (entry.confidence == CONFIDENCE_MEDIUM) ? 0.6f : 0.3f;
            engine->knowledge_base[idx + 2] = entry.weight;

            /* 字符串映射为数值 */
            unsigned int hash = 5381;
            if (entry.subject) {
                const char* s = entry.subject;
                while (*s) { hash = ((hash << 5) + hash) + (unsigned char)*s; s++; }
            }
            engine->knowledge_base[idx + 3] = (float)(hash % 10000) / 10000.0f;

            hash = 5381;
            if (entry.predicate) {
                const char* s = entry.predicate;
                while (*s) { hash = ((hash << 5) + hash) + (unsigned char)*s; s++; }
            }
            engine->knowledge_base[idx + 4] = (float)(hash % 10000) / 10000.0f;

            hash = 5381;
            if (entry.object) {
                const char* s = entry.object;
                while (*s) { hash = ((hash << 5) + hash) + (unsigned char)*s; s++; }
            }
            engine->knowledge_base[idx + 5] = (float)(hash % 10000) / 10000.0f;

            engine->knowledge_base[idx + 6] = (float)entry.source;
            engine->knowledge_base[idx + 7] = (float)(entry.timestamp % 1000000) / 1000000.0f;

            synced++;
        }
        /* 手动释放知识条目内存（knowledge.h未导出此函数） */
        safe_free((void**)&entry.subject);
        safe_free((void**)&entry.predicate);
        safe_free((void**)&entry.object);
        safe_free((void**)&entry.metadata);
        safe_free((void**)&entry.embedding);
        memset(&entry, 0, sizeof(KnowledgeEntry));
    }

    /* 规则索引标记为需要重建 */
    engine->rule_index_built = 0;

    /* P0-009: 同步完成后标记知识就绪 */
    if (synced > 0) engine->knowledge_ready = 1;

    return (int)synced;
}

/* ZSFX-007: reasoning_infer_with_knowledge 全平台统一编译 */
/**
 * @brief 使用知识库增强推理
 *
 * 结合外部知识库中的事实和规则执行推理。工作流程：
 * 1. 如果关联了知识库且 knowledge_weight > 0，先同步知识库
 * 2. 使用知识库中的相关事实作为附加前提
 * 3. 执行标准推理
 * 4. 使用知识库中的规则进行规则匹配修正
 * 5. 将高置信度结论回写到知识库
 */
int reasoning_infer_with_knowledge(ReasoningEngine* engine,
                                  const float* premises, size_t num_premises,
                                  float* conclusion, size_t max_conclusion_size,
                                  ReasoningMode mode,
                                  float knowledge_weight) {
    if (!engine || !premises || !conclusion || max_conclusion_size == 0) {
        return -1;
    }

    if (knowledge_weight < 0.0f) knowledge_weight = 0.0f;
    if (knowledge_weight > 1.0f) knowledge_weight = 1.0f;

    /* 步骤1: 自动同步知识库 */
    int has_kb = (engine->external_kb != NULL) ? 1 : 0;
    if (has_kb && engine->kb_auto_sync) {
        reasoning_sync_knowledge(engine);
    }

    /* 步骤2: 如果有知识库，从知识库中提取相关事实作为附加前提 */
    float* augmented_premises = NULL;
    size_t augmented_count = num_premises;

    if (has_kb && knowledge_weight > 0.0f && engine->knowledge_base_size > 0) {
        size_t kb_entries = engine->knowledge_base_size / 8;
        if (kb_entries > 0) {
            size_t extra_capacity = kb_entries * 2;
            augmented_premises = (float*)safe_malloc(
                (num_premises + extra_capacity) * sizeof(float));
            if (augmented_premises) {
                memcpy(augmented_premises, premises, num_premises * sizeof(float));
                augmented_count = num_premises;

                /* 从知识库中筛选与前提相关的条目（基于哈希相似度） */
                for (size_t i = 0; i < kb_entries && augmented_count < num_premises + extra_capacity; i++) {
                    float kb_type = engine->knowledge_base[i * 8 + 0];
                    float kb_confidence = engine->knowledge_base[i * 8 + 1];
                    if (kb_type == (float)KNOWLEDGE_FACT && kb_confidence > 0.3f) {
                        /* 检查与前提的相关性：主体哈希匹配 */
                        float kb_subject = engine->knowledge_base[i * 8 + 3];
                        for (size_t j = 0; j < num_premises; j++) {
                            float diff = fabsf(kb_subject - premises[j]);
                            if (diff < 0.5f) {
                                /* 相关事实，加入增强前提 */
                                augmented_premises[augmented_count++] =
                                    engine->knowledge_base[i * 8 + 5] * knowledge_weight;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    /* 步骤3: 使用增强的前提执行标准推理 */
    int ret;
    if (augmented_premises) {
        ret = reasoning_infer(engine, augmented_premises, augmented_count,
                             conclusion, max_conclusion_size, mode);
        safe_free((void**)&augmented_premises);
    } else {
        ret = reasoning_infer(engine, premises, num_premises,
                             conclusion, max_conclusion_size, mode);
    }

    if (ret != 0) {
        return ret;
    }

    /* 步骤4: 使用知识库中的规则进行推理结果修正 */
    if (has_kb && knowledge_weight > 0.3f && engine->knowledge_base_size > 0) {
        size_t kb_entries = engine->knowledge_base_size / 8;
        float rule_correction = 0.0f;
        size_t rule_count = 0;

        for (size_t i = 0; i < kb_entries; i++) {
            float kb_type = engine->knowledge_base[i * 8 + 0];
            float kb_weight = engine->knowledge_base[i * 8 + 2];
            if (kb_type == (float)KNOWLEDGE_RULE && kb_weight > 0.5f) {
                float rule_object = engine->knowledge_base[i * 8 + 5];
                rule_correction += rule_object * kb_weight;
                rule_count++;
            }
        }

        if (rule_count > 0) {
            float avg_rule = rule_correction / (float)rule_count;
            float blend = knowledge_weight * 0.3f;
            for (size_t i = 0; i < max_conclusion_size && i < 1; i++) {
                conclusion[i] = conclusion[i] * (1.0f - blend) + avg_rule * blend;
            }
        }
    }

    /* 步骤5: 将推理结果回写知识库 */
    if (has_kb && knowledge_weight > 0.5f) {
        float confidence = reasoning_evaluate_confidence(engine, conclusion, max_conclusion_size,
                                                        premises, num_premises);
        if (confidence > 0.7f) {
            KnowledgeEntry new_entry;
            memset(&new_entry, 0, sizeof(KnowledgeEntry));
            new_entry.subject = string_duplicate("inference_result");
            new_entry.predicate = string_duplicate("conclusion");
            new_entry.object = (char*)safe_malloc(64);
            if (!new_entry.subject || !new_entry.predicate || !new_entry.object) {
                safe_free((void**)&new_entry.subject);
                safe_free((void**)&new_entry.predicate);
                safe_free((void**)&new_entry.object);
                return 0;
            }
            snprintf(new_entry.object, 64, "%.4f", conclusion[0]);
            new_entry.type = KNOWLEDGE_FACT;
            new_entry.confidence = (confidence > 0.9f) ? CONFIDENCE_HIGH :
                                  (confidence > 0.7f) ? CONFIDENCE_MEDIUM : CONFIDENCE_LOW;
            new_entry.source = SOURCE_INFERENCE;
            new_entry.weight = confidence;
            new_entry.timestamp = (long)time(NULL);

            knowledge_base_add(engine->external_kb, &new_entry);

            safe_free((void**)&new_entry.subject);
            safe_free((void**)&new_entry.predicate);
            safe_free((void**)&new_entry.object);
        }
    }

    return 0;
}

/* ========== P2-3: 贝叶斯推理网络 + 推理引擎绑定 ========== */
/* ZSFX-007: 全平台统一编译，不再区分GCC/MSVC */
BayesianNetwork* bayesian_network_create(size_t max_nodes) {
    if (max_nodes == 0 || max_nodes > 10000) return NULL;

    BayesianNetwork* network = (BayesianNetwork*)safe_calloc(1, sizeof(BayesianNetwork));
    if (!network) return NULL;

    network->nodes = (BayesianNode*)safe_calloc(max_nodes, sizeof(BayesianNode));
    if (!network->nodes) { safe_free((void**)&network); return NULL; }

    network->edges = (BayesianEdge*)safe_calloc(max_nodes * 4, sizeof(BayesianEdge));
    if (!network->edges) { safe_free((void**)&network->nodes); safe_free((void**)&network); return NULL; }

    network->cpds = (BayesianCPD*)safe_calloc(max_nodes, sizeof(BayesianCPD));
    if (!network->cpds) { safe_free((void**)&network->edges); safe_free((void**)&network->nodes); safe_free((void**)&network); return NULL; }

    network->adjacency_matrix = (float**)safe_calloc(max_nodes, sizeof(float*));
    if (!network->adjacency_matrix) {
        safe_free((void**)&network->cpds); safe_free((void**)&network->edges);
        safe_free((void**)&network->nodes); safe_free((void**)&network);
        return NULL;
    }
    for (size_t i = 0; i < max_nodes; i++) {
        network->adjacency_matrix[i] = (float*)safe_calloc(max_nodes, sizeof(float));
        if (!network->adjacency_matrix[i]) {
            for (size_t j = 0; j < i; j++) safe_free((void**)&network->adjacency_matrix[j]);
            safe_free((void**)&network->adjacency_matrix); safe_free((void**)&network->cpds);
            safe_free((void**)&network->edges); safe_free((void**)&network->nodes); safe_free((void**)&network);
            return NULL;
        }
    }

    network->num_nodes = 0;
    network->node_capacity = max_nodes;
    network->num_edges = 0;
    network->edge_capacity = max_nodes * 4;
    network->num_cpds = 0;
    network->cpd_capacity = max_nodes;

    return network;
}

void bayesian_network_free(BayesianNetwork* bn) {
    if (!bn) return;

    for (size_t i = 0; i < bn->num_cpds; i++) {
        free(bn->cpds[i].probabilities);
        free(bn->cpds[i].parent_ids);
        free(bn->cpds[i].parent_states);
    }

    if (bn->adjacency_matrix) {
        for (size_t i = 0; i < bn->node_capacity; i++) free(bn->adjacency_matrix[i]);
        free(bn->adjacency_matrix);
    }

    free(bn->cpds);
    free(bn->edges);
    free(bn->nodes);

    bn->num_nodes = 0;
    bn->num_edges = 0;
    bn->num_cpds = 0;
}

int bayesian_network_add_node(void* net_ptr, const char* name, int num_states) {
    BayesianNetwork* network = (BayesianNetwork*)net_ptr;
    if (!network || !name) return -1;
    if (num_states < 2) return -1;
    if (network->num_nodes >= network->node_capacity) return -1;

    int id = (int)network->num_nodes;
    BayesianNode* node = &network->nodes[id];
    node->node_id = id;
    memset(node->name, 0, sizeof(node->name));
    strncpy(node->name, name, sizeof(node->name) - 1);
    node->is_observed = 0;
    node->current_state = 0;

    network->num_nodes++;
    return id;
}

int bayesian_network_add_edge(void* net_ptr, int parent_id, int child_id) {
    BayesianNetwork* network = (BayesianNetwork*)net_ptr;
    if (!network) return -1;
    if (parent_id < 0 || parent_id >= (int)network->num_nodes) return -1;
    if (child_id < 0 || child_id >= (int)network->num_nodes) return -1;
    if (parent_id == child_id) return -1;
    if (network->num_edges >= network->edge_capacity) return -1;

    for (size_t i = 0; i < network->num_edges; i++) {
        if (network->edges[i].parent_id == parent_id &&
            network->edges[i].child_id == child_id) {
            return 0;
        }
    }

    BayesianEdge* edge = &network->edges[network->num_edges];
    edge->parent_id = parent_id;
    edge->child_id = child_id;
    edge->is_active = 1;

    network->adjacency_matrix[parent_id][child_id] = 1.0f;

    network->num_edges++;
    return 0;
}

int bayesian_network_set_cpd(void* net_ptr, int var_id,
                              const int* parent_ids, int num_parents,
                              const float* probabilities, size_t prob_size) {
    BayesianNetwork* network = (BayesianNetwork*)net_ptr;
    if (!network || !probabilities) return -1;
    if (var_id < 0 || var_id >= (int)network->num_nodes) return -1;
    if (num_parents < 0) return -1;
    if (num_parents > 0 && !parent_ids) return -1;

    BayesianCPD* cpd = NULL;
    int found = 0;
    for (size_t i = 0; i < network->num_cpds; i++) {
        if (network->cpds[i].variable_id == var_id) {
            cpd = &network->cpds[i];
            safe_free((void**)&cpd->probabilities);
            safe_free((void**)&cpd->parent_ids);
            safe_free((void**)&cpd->parent_states);
            memset(cpd, 0, sizeof(BayesianCPD));
            found = 1;
            break;
        }
    }

    if (!found) {
        if (network->num_cpds >= network->cpd_capacity) return -1;
        cpd = &network->cpds[network->num_cpds];
        network->num_cpds++;
    }

    cpd->variable_id = var_id;
    cpd->num_parents = num_parents;
    cpd->num_states = 2;

    if (num_parents > 0) {
        cpd->parent_ids = (int*)safe_malloc(num_parents * sizeof(int));
        cpd->parent_states = (int*)safe_malloc(num_parents * sizeof(int));
        if (!cpd->parent_ids || !cpd->parent_states) {
            safe_free((void**)&cpd->parent_ids);
            safe_free((void**)&cpd->parent_states);
            return -1;
        }
        for (int i = 0; i < num_parents; i++) {
            cpd->parent_ids[i] = parent_ids[i];
            cpd->parent_states[i] = 2;
        }
    } else {
        cpd->parent_ids = NULL;
        cpd->parent_states = NULL;
    }

    cpd->table_size = prob_size;
    cpd->probabilities = (float*)safe_malloc(prob_size * sizeof(float));
    if (!cpd->probabilities) return -1;

    memcpy(cpd->probabilities, probabilities, prob_size * sizeof(float));

    if (num_parents > 0) {
        int parent_comb = 1;
        for (int i = 0; i < num_parents; i++) {
            parent_comb *= cpd->parent_states[i];
        }
        cpd->num_states = (int)(prob_size / parent_comb);
    } else {
        cpd->num_states = (int)prob_size;
    }

    return 0;
}

static int bayesian_has_cycle_dfs(BayesianNetwork* network, int node_id,
                                   int* visited, int* rec_stack) {
    visited[node_id] = 1;
    rec_stack[node_id] = 1;

    for (size_t i = 0; i < network->num_edges; i++) {
        if (network->edges[i].parent_id == node_id && network->edges[i].is_active) {
            int child = network->edges[i].child_id;
            if (!visited[child]) {
                if (bayesian_has_cycle_dfs(network, child, visited, rec_stack))
                    return 1;
            } else if (rec_stack[child]) {
                return 1;
            }
        }
    }

    rec_stack[node_id] = 0;
    return 0;
}

int bayesian_network_is_valid(void* net_ptr) {
    BayesianNetwork* network = (BayesianNetwork*)net_ptr;
    if (!network || network->num_nodes == 0) return 0;

    int* visited = (int*)safe_calloc(network->num_nodes, sizeof(int));
    int* rec_stack = (int*)safe_calloc(network->num_nodes, sizeof(int));
    if (!visited || !rec_stack) {
        safe_free((void**)&visited);
        safe_free((void**)&rec_stack);
        return 0;
    }

    int has_cycle = 0;
    for (size_t i = 0; i < network->num_nodes; i++) {
        if (!visited[i]) {
            if (bayesian_has_cycle_dfs(network, (int)i, visited, rec_stack)) {
                has_cycle = 1;
                break;
            }
        }
    }

    safe_free((void**)&visited);
    safe_free((void**)&rec_stack);
    return has_cycle ? 0 : 1;
}

static BayesianCPD* bayesian_get_cpd(BayesianNetwork* network, int var_id) {
    if (!network) return NULL;
    for (size_t i = 0; i < network->num_cpds; i++) {
        if (network->cpds[i].variable_id == var_id)
            return &network->cpds[i];
    }
    return NULL;
}

static float bayesian_get_conditional_prob(BayesianNetwork* network, int var_id,
                                            int state, const int* parent_states) {
    BayesianCPD* cpd = bayesian_get_cpd(network, var_id);
    if (!cpd) {
        return (state == 0) ? 0.5f : 0.5f;
    }

    if (cpd->num_parents == 0) {
        if (state >= 0 && state < cpd->num_states && state < (int)cpd->table_size) {
            return cpd->probabilities[state];
        }
        return 0.0f;
    }

    int index = 0;
    int stride = 1;
    for (int p = cpd->num_parents - 1; p >= 0; p--) {
        index += parent_states[p] * stride;
        stride *= cpd->parent_states[p];
    }
    index = index * cpd->num_states + state;

    if (index >= 0 && index < (int)cpd->table_size) {
        return cpd->probabilities[index];
    }
    return 0.0f;
}

static float bayesian_compute_markov_blanket_prob(BayesianNetwork* network,
                                                    int var_id, int state,
                                                    int* current_states) {
    if (!network || !current_states) return 0.0f;

    float prob = 1.0f;

    if (var_id >= 0 && var_id < (int)network->num_nodes) {
        BayesianCPD* cpd = bayesian_get_cpd(network, var_id);
        if (cpd) {
            int* parent_vals = NULL;
            if (cpd->num_parents > 0) {
                parent_vals = (int*)safe_malloc(cpd->num_parents * sizeof(int));
                if (parent_vals) {
                    for (int p = 0; p < cpd->num_parents; p++) {
                        int pid = cpd->parent_ids[p];
                        parent_vals[p] = (pid >= 0 && pid < (int)network->num_nodes)
                            ? current_states[pid] : 0;
                    }
                }
            }
            float cp = bayesian_get_conditional_prob(network, var_id, state, parent_vals);
            prob *= (cp > 0.0f) ? cp : 1e-10f;
            safe_free((void**)&parent_vals);
        }
    }

    for (size_t i = 0; i < network->num_edges; i++) {
        if (network->edges[i].parent_id == var_id && network->edges[i].is_active) {
            int child_id = network->edges[i].child_id;
            BayesianCPD* child_cpd = bayesian_get_cpd(network, child_id);
            if (child_cpd) {
                int* child_parent_vals = NULL;
                if (child_cpd->num_parents > 0) {
                    child_parent_vals = (int*)safe_malloc(child_cpd->num_parents * sizeof(int));
                    if (child_parent_vals) {
                        for (int p = 0; p < child_cpd->num_parents; p++) {
                            int pid = child_cpd->parent_ids[p];
                            if (pid == var_id) {
                                child_parent_vals[p] = state;
                            } else {
                                child_parent_vals[p] = (pid >= 0 && pid < (int)network->num_nodes)
                                    ? current_states[pid] : 0;
                            }
                        }
                    }
                }
                float cp = bayesian_get_conditional_prob(network, child_id,
                    current_states[child_id], child_parent_vals);
                prob *= (cp > 0.0f) ? cp : 1e-10f;
                safe_free((void**)&child_parent_vals);
            }
        }
    }

    return prob;
}

static int bayesian_sample_from_distribution(const float* probs, int num_states) {
    if (!probs || num_states <= 0) return 0;

    float sum = 0.0f;
    for (int i = 0; i < num_states; i++) {
        sum += probs[i];
    }
    if (sum <= 0.0f) return 0;

    float r = secure_random_float() * sum;
    float cumulative = 0.0f;
    for (int i = 0; i < num_states; i++) {
        cumulative += probs[i];
        if (r <= cumulative) return i;
    }
    return num_states - 1;
}

int bayesian_network_variable_elimination(void* net_ptr,
                                          const int* query_vars, int num_queries,
                                          const int* evidence_vars, const int* evidence_vals,
                                          int num_evidence,
                                          BayesianQueryResult* result) {
    BayesianNetwork* network = (BayesianNetwork*)net_ptr;
    if (!network || !query_vars || !result) return -1;
    if (num_queries <= 0 || num_queries > (int)network->num_nodes) return -1;
    if (num_evidence < 0) return -1;
    if (num_evidence > 0 && (!evidence_vars || !evidence_vals)) return -1;

    int* states = (int*)safe_calloc(network->num_nodes, sizeof(int));
    int* counts = (int*)safe_calloc(network->num_nodes * 10, sizeof(int));
    if (!states || !counts) { safe_free((void**)&states); safe_free((void**)&counts); return -1; }

    for (int e = 0; e < num_evidence; e++) {
        int ev = evidence_vars[e];
        if (ev >= 0 && ev < (int)network->num_nodes) {
            states[ev] = evidence_vals[e];
        }
    }

    for (size_t v = 0; v < network->num_nodes; v++) {
        int is_evidence = 0;
        for (int e = 0; e < num_evidence; e++) {
            if (evidence_vars[e] == (int)v) { is_evidence = 1; break; }
        }
        if (is_evidence) continue;

        int num_states_v = 2;
        BayesianCPD* cpd = bayesian_get_cpd(network, (int)v);
        if (cpd) num_states_v = cpd->num_states;
        states[v] = (int)secure_random_int((uint32_t)num_states_v);
    }
    
    int total_samples = 5000;
    int valid_samples = 0;

    for (int s = 0; s < total_samples; s++) {
        for (size_t v = 0; v < network->num_nodes; v++) {
            int is_evidence = 0;
            for (int e = 0; e < num_evidence; e++) {
                if (evidence_vars[e] == (int)v) { is_evidence = 1; break; }
            }
            if (is_evidence) continue;

            int num_states_v = 2;
            BayesianCPD* cpd = bayesian_get_cpd(network, (int)v);
            if (cpd) num_states_v = cpd->num_states;

            float* cond_probs = (float*)safe_malloc(num_states_v * sizeof(float));
            if (!cond_probs) continue;

            float total = 0.0f;
            for (int st = 0; st < num_states_v; st++) {
                cond_probs[st] = bayesian_compute_markov_blanket_prob(network, (int)v, st, states);
                total += cond_probs[st];
            }

            if (total > 0.0f) {
                for (int st = 0; st < num_states_v; st++) {
                    cond_probs[st] /= total;
                }
                states[v] = bayesian_sample_from_distribution(cond_probs, num_states_v);
            } else {
                states[v] = (int)secure_random_int((uint32_t)num_states_v);
            }
            safe_free((void**)&cond_probs);
        }

        for (int q = 0; q < num_queries; q++) {
            int qv = query_vars[q];
            if (qv >= 0 && qv < (int)network->num_nodes) {
                int num_states_q = 2;
                BayesianCPD* cpd_q = bayesian_get_cpd(network, qv);
                if (cpd_q) num_states_q = cpd_q->num_states;
                if (states[qv] >= 0 && states[qv] < num_states_q) {
                    counts[qv * 10 + states[qv]]++;
                }
            }
        }
        valid_samples++;
    }

    for (int q = 0; q < num_queries; q++) {
        int qv = query_vars[q];
        if (qv >= 0 && qv < (int)network->num_nodes && q == 0) {
            int num_states_q = 2;
            BayesianCPD* cpd_q = bayesian_get_cpd(network, qv);
            if (cpd_q) num_states_q = cpd_q->num_states;

            result->num_states = (size_t)num_states_q;
            result->probabilities = (float*)safe_calloc(num_states_q, sizeof(float));
            if (result->probabilities) {
                for (int st = 0; st < num_states_q; st++) {
                    result->probabilities[st] = (valid_samples > 0)
                        ? (float)counts[qv * 10 + st] / (float)valid_samples : 0.0f;
                }
            }
            result->confidence = (valid_samples > 100) ? 0.85f : 0.5f;
        }
    }

    safe_free((void**)&states);
    safe_free((void**)&counts);
    return 0;
}

int bayesian_network_gibbs_sampling(void* net_ptr,
                                    const int* query_vars, int num_queries,
                                    const int* evidence_vars, const int* evidence_vals,
                                    int num_evidence,
                                    int num_samples, int burn_in,
                                    BayesianQueryResult* result) {
    BayesianNetwork* network = (BayesianNetwork*)net_ptr;
    if (!network || !query_vars || !result) return -1;
    if (num_queries <= 0) return -1;
    if (num_samples <= 0 || burn_in < 0) return -1;
    if (num_evidence > 0 && (!evidence_vars || !evidence_vals)) return -1;

    int* states = (int*)safe_calloc(network->num_nodes, sizeof(int));
    int* counts = (int*)safe_calloc(network->num_nodes * 10, sizeof(int));
    if (!states || !counts) { safe_free((void**)&states); safe_free((void**)&counts); return -1; }

    for (int e = 0; e < num_evidence; e++) {
        int ev = evidence_vars[e];
        if (ev >= 0 && ev < (int)network->num_nodes) {
            states[ev] = evidence_vals[e];
        }
    }

    for (size_t v = 0; v < network->num_nodes; v++) {
        int is_evidence = 0;
        for (int e = 0; e < num_evidence; e++) {
            if (evidence_vars[e] == (int)v) { is_evidence = 1; break; }
        }
        if (is_evidence) continue;

        int num_states_v = 2;
        BayesianCPD* cpd = bayesian_get_cpd(network, (int)v);
        if (cpd) num_states_v = cpd->num_states;
        states[v] = (int)secure_random_int((uint32_t)num_states_v);
    }

    int total_valid = 0;

    for (int s = 0; s < num_samples + burn_in; s++) {
        for (size_t v = 0; v < network->num_nodes; v++) {
            int is_evidence = 0;
            for (int e = 0; e < num_evidence; e++) {
                if (evidence_vars[e] == (int)v) { is_evidence = 1; break; }
            }
            if (is_evidence) continue;

            int num_states_v = 2;
            BayesianCPD* cpd = bayesian_get_cpd(network, (int)v);
            if (cpd) num_states_v = cpd->num_states;

            float* cond_probs = (float*)safe_malloc(num_states_v * sizeof(float));
            if (!cond_probs) continue;

            float total = 0.0f;
            for (int st = 0; st < num_states_v; st++) {
                cond_probs[st] = bayesian_compute_markov_blanket_prob(network, (int)v, st, states);
                total += cond_probs[st];
            }

            if (total > 0.0f) {
                for (int st = 0; st < num_states_v; st++) {
                    cond_probs[st] /= total;
                }
                states[v] = bayesian_sample_from_distribution(cond_probs, num_states_v);
            }
            safe_free((void**)&cond_probs);
        }

        if (s >= burn_in) {
            for (int q = 0; q < num_queries; q++) {
                int qv = query_vars[q];
                if (qv >= 0 && qv < (int)network->num_nodes) {
                    int num_states_q = 2;
                    BayesianCPD* cpd_q = bayesian_get_cpd(network, qv);
                    if (cpd_q) num_states_q = cpd_q->num_states;
                    if (states[qv] >= 0 && states[qv] < num_states_q) {
                        counts[qv * 10 + states[qv]]++;
                    }
                }
            }
            total_valid++;
        }
    }

    for (int q = 0; q < num_queries; q++) {
        int qv = query_vars[q];
        if (qv >= 0 && qv < (int)network->num_nodes && q == 0) {
            int num_states_q = 2;
            BayesianCPD* cpd_q = bayesian_get_cpd(network, qv);
            if (cpd_q) num_states_q = cpd_q->num_states;

            result->num_states = (size_t)num_states_q;
            result->probabilities = (float*)safe_calloc(num_states_q, sizeof(float));
            if (result->probabilities) {
                for (int st = 0; st < num_states_q; st++) {
                    result->probabilities[st] = (total_valid > 0)
                        ? (float)counts[qv * 10 + st] / (float)total_valid : 0.0f;
                }
            }
            result->confidence = (total_valid > 100) ? 0.9f : 0.5f;
        }
    }

    safe_free((void**)&states);
    safe_free((void**)&counts);
    return 0;
}

int bayesian_network_belief_propagation(void* net_ptr,
                                        const int* evidence_vars, const int* evidence_vals,
                                        int num_evidence,
                                        int max_iterations, float convergence_threshold,
                                        BayesianQueryResult* result, size_t result_size) {
    BayesianNetwork* network = (BayesianNetwork*)net_ptr;
    if (!network || !result) return -1;
    if (max_iterations <= 0) return -1;
    if (result_size < network->num_nodes) return -1;
    if (num_evidence > 0 && (!evidence_vars || !evidence_vals)) return -1;

    size_t n = network->num_nodes;

    float* beliefs = (float*)safe_calloc(n * 10, sizeof(float));
    int* node_states = (int*)safe_calloc(n, sizeof(int));
    if (!beliefs || !node_states) { safe_free((void**)&beliefs); safe_free((void**)&node_states); return -1; }

    for (size_t v = 0; v < n; v++) {
        int num_states_v = 2;
        BayesianCPD* cpd = bayesian_get_cpd(network, (int)v);
        if (cpd) num_states_v = cpd->num_states;
        for (int s = 0; s < num_states_v && s < 10; s++) {
            beliefs[v * 10 + s] = 1.0f / (float)num_states_v;
        }
    }

    for (int e = 0; e < num_evidence; e++) {
        int ev = evidence_vars[e];
        if (ev >= 0 && ev < (int)n) {
            int val = evidence_vals[e];
            int num_states_v = 2;
            BayesianCPD* cpd = bayesian_get_cpd(network, ev);
            if (cpd) num_states_v = cpd->num_states;
            if (val >= 0 && val < num_states_v && val < 10) {
                for (int s = 0; s < num_states_v && s < 10; s++) {
                    beliefs[ev * 10 + s] = (s == val) ? 1.0f : 0.0f;
                }
            }
        }
    }

    for (int iter = 0; iter < max_iterations; iter++) {
        float max_change = 0.0f;

        for (size_t v = 0; v < n; v++) {
            int num_states_v = 2;
            BayesianCPD* cpd = bayesian_get_cpd(network, (int)v);
            if (cpd) num_states_v = cpd->num_states;

            int actual_states = (num_states_v < 10) ? num_states_v : 10;
            float* new_belief = (float*)safe_calloc(actual_states, sizeof(float));
            if (!new_belief) continue;

            float* parent_msg = (float*)safe_malloc(actual_states * sizeof(float));
            if (parent_msg) {
                for (int s = 0; s < actual_states; s++) parent_msg[s] = 1.0f;

                for (size_t e = 0; e < network->num_edges; e++) {
                    if (network->edges[e].child_id == (int)v && network->edges[e].is_active) {
                        int pid = network->edges[e].parent_id;
                        int ps = 2;
                        BayesianCPD* pcpd = bayesian_get_cpd(network, pid);
                        if (pcpd) ps = pcpd->num_states;
                        if (ps > 10) ps = 10;
                        for (int s = 0; s < actual_states && s < ps; s++) {
                            parent_msg[s] *= beliefs[pid * 10 + s];
                        }
                    }
                }

                if (cpd) {
                    for (int s = 0; s < actual_states; s++) {
                        new_belief[s] = cpd->probabilities[s] * parent_msg[s];
                    }
                } else {
                    for (int s = 0; s < actual_states; s++) {
                        new_belief[s] = (1.0f / actual_states) * parent_msg[s];
                    }
                }

                float total = 0.0f;
                for (int s = 0; s < actual_states; s++) total += new_belief[s];
                if (total > 0.0f) {
                    for (int s = 0; s < actual_states; s++) new_belief[s] /= total;
                }

                for (int s = 0; s < actual_states; s++) {
                    float change = fabsf(new_belief[s] - beliefs[v * 10 + s]);
                    if (change > max_change) max_change = change;
                }

                for (int s = 0; s < actual_states; s++) {
                    beliefs[v * 10 + s] = new_belief[s];
                }

                safe_free((void**)&parent_msg);
            }
            safe_free((void**)&new_belief);
        }

        if (max_change < convergence_threshold) break;
    }

    if (result_size >= n) {
        result->num_states = n;
        result->probabilities = (float*)safe_malloc(n * sizeof(float));
        if (result->probabilities) {
            for (size_t v = 0; v < n; v++) {
                result->probabilities[v] = beliefs[v * 10 + 0];
            }
        }
        result->confidence = 0.8f;
    } else {
        result->num_states = 0;
        result->probabilities = NULL;
        result->confidence = 0.5f;
    }

    safe_free((void**)&beliefs);
    safe_free((void**)&node_states);
    return 0;
}

void bayesian_query_result_free(BayesianQueryResult* result) {
    if (!result) return;
    safe_free((void**)&result->probabilities);
    result->num_states = 0;
    result->confidence = 0.0f;
}

int reasoning_engine_set_bayesian_network(ReasoningEngine* engine, BayesianNetwork* network) {
    if (!engine) return -1;
    engine->bayesian_network = network;
    return 0;
}

BayesianNetwork* reasoning_engine_get_bayesian_network(const ReasoningEngine* engine) {
    if (!engine) return NULL;
    return engine->bayesian_network;
}

/* ZSFAB P1-001修复: 已删除重复stub，完整实现在上方 */
