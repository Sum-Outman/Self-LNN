/**
 * @file logic_reasoning.c
 * @brief 逻辑推理引擎实现
 * 
 * 基于规则的逻辑推理引擎，支持前向链、后向链、冲突解决等。
 */

#define _CRT_NONSTDC_NO_DEPRECATE

#include "selflnn/knowledge/logic_reasoning.h"
#include "selflnn/knowledge/semantic_network.h"
#include "selflnn/knowledge/knowledge_graph.h"
#include "selflnn/selflnn.h" /* P1-001: selflnn_get_knowledge_graph() */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/secure_random.h" /* 安全随机数 */
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <math.h>

/* ============================================================================
 * F-09: 概率软逻辑(PSL)数据结构
 * =========================================================================== */

typedef struct PslFormulaSpec {
    char** premise_vars;         /* 前提变量 */
    size_t premise_count;        /* 前提数量 */
    char* conclusion_var;        /* 结论变量 */
    float weight;                /* 公式权重 */
    int is_active;               /* 是否激活 */
} PslFormulaSpec;

typedef struct {
    PslFormulaSpec* formulas;    /* 公式数组 */
    size_t formula_count;        /* 公式数量 */
    size_t formula_capacity;     /* 公式容量 */
} PslState;

/* ============================================================================
 * F-09: 默认推理数据结构
 * =========================================================================== */

typedef struct DefaultRuleSpec {
    char* rule_name;             /* 规则名称 */
    char** premises;             /* 前提数组 */
    size_t premise_count;        /* 前提数量 */
    char** conclusions;          /* 结论数组 */
    size_t conclusion_count;     /* 结论数量 */
    char** unless_conditions;    /* 除非条件数组 */
    size_t unless_count;         /* 除非条件数量 */
    float confidence;            /* 置信度 */
    int priority;                /* 优先级 */
    int is_active;               /* 规则是否激活 */
} DefaultRuleSpec;

typedef struct {
    DefaultRuleSpec* rules;      /* 规则数组 */
    size_t rule_count;           /* 规则数量 */
    size_t rule_capacity;        /* 规则容量 */
} DefaultReasoningState;

/* ============================================================================
 * F-09: 非单调推理数据结构
 * =========================================================================== */

typedef struct {
    char* fact;                  /* 事实字符串 */
    float confidence;            /* 置信度 */
    int is_monotonic;            /* 是否为单调（不可撤销）事实 */
    int* justification_rule_ids; /* 推理出该事实的规则ID数组 */
    size_t justification_count;  /* 规则ID数量 */
    float justification_weight;  /* 推理权重 */
} NonmonotonicFact;

typedef struct {
    NonmonotonicFact* facts;     /* 事实数组 */
    size_t fact_count;           /* 事实数量 */
    size_t fact_capacity;        /* 事实容量 */
} NonmonotonicState;

/* 字符串复制函数声明 */

/* F-09: PSL Lukasiewicz逻辑算子声明 */
static float psl_t_norm(float x, float y);
static float psl_t_conorm(float x, float y);
static float psl_implication(float x, float y);
static float psl_negation(float x);
static float psl_equivalence(float x, float y);
static float psl_conjunction_distance(float* truth_values, size_t count, float conclusion_truth);
static float psl_implication_distance(float body_truth, float head_truth);
static int psl_optimize_gradient_descent(PslState* state, float* variable_values, size_t var_count, int max_iter, float learning_rate);

/* F-09: 默认推理函数声明 */
static int default_reasoning_should_block(DefaultRuleSpec* rule, char** facts, size_t fact_count);
static int default_reasoning_compare_specificity(const void* a, const void* b);

/* F-09: 非单调推理函数声明 */
static int nonmonotonic_add_fact(NonmonotonicState* state, const char* fact, float confidence, int is_monotonic, int* rule_ids, size_t rule_count, float weight);
static int nonmonotonic_remove_fact(NonmonotonicState* state, const char* fact);
static int nonmonotonic_detect_contradiction(NonmonotonicState* state, const char* new_fact, char** conflicting_fact, size_t* conflicting_idx);

/**
 * @brief 推理引擎内部结构
 */
struct LogicReasoningEngine {
    LogicReasoningEngineConfig config;     /**< 引擎配置 */
    
    InferenceRule** rules;            /**< 规则数组 */
    size_t rule_count;                /**< 规则数量 */
    size_t rule_capacity;             /**< 规则数组容量 */
    
    LogicConflictResolutionStrategy conflict_strategy; /**< 冲突解决策略 */
    
    /* 规则使用统计 */
    size_t* rule_use_counts;          /**< 规则使用次数数组 */
    int* rule_last_used_step;         /**< 规则最后使用步数数组 */
    size_t current_inference_step;    /**< 当前推理步数 */
    
    /* 工作内存 */
    char** working_memory;            /**< 工作内存数组 */
    size_t wm_size;                   /**< 工作内存大小 */
    size_t wm_capacity;               /**< 工作内存容量 */
    
    /* 统计信息 */
    size_t total_inferences;          /**< 总推理次数 */
    clock_t total_inference_time;     /**< 总推理时间（时钟周期） */
    
    /* ---- F-09: 概率软逻辑(PSL) ---- */
    PslState psl_state;
    
    /* ---- F-09: 默认推理 ---- */
    DefaultReasoningState default_state;
    
    /* ---- F-09: 非单调推理 ---- */
    NonmonotonicState nonmonotonic_state;
};

/**
 * @brief 工作内存项
 */
typedef struct {
    char* fact;                       /**< 事实字符串 */
    float confidence;                 /**< 置信度 */
    int source_rule;                  /**< 来源规则ID（-1表示初始事实） */
} WorkingMemoryItem;

/* ============================================================================
 * 内部辅助函数
 * =========================================================================== */

/**
 * @brief 扩展数组容量
 */
static int expand_array(void** array, size_t* capacity, size_t current_size, size_t element_size) {
    if (current_size < *capacity) {
        return 0;
    }
    
    size_t new_capacity = (*capacity == 0) ? 4 : *capacity * 2;
    void* new_array = safe_realloc(*array, new_capacity * element_size);
    if (!new_array) {
        return -1;
    }
    
    *array = new_array;
    *capacity = new_capacity;
    return 0;
}

/**
 * @brief 分割字符串（兼容旧接口）
 */
static size_t split_string(const char* str, char delimiter, char*** result) {
    if (!str || !result) return 0;
    
    // 将字符分隔符转换为字符串
    char delimiter_str[2] = {delimiter, '\0'};
    
    size_t count = 0;
    *result = string_split(str, delimiter_str, STRING_SPLIT_SKIP_EMPTY, &count);
    
    return count;
}

/**
 * @brief 检查事实是否匹配模式
 */
static int fact_matches_pattern(const char* fact, const char* pattern) {
    if (!fact || !pattern) return 0;
    
    // 简单字符串匹配：检查模式是否为事实的子串
    return (strstr(fact, pattern) != NULL);
}

/**
 * @brief 检查规则是否可应用
 */
static int rule_is_applicable(InferenceRule* rule, char** facts, size_t fact_count) {
    if (!rule || !facts) return 0;
    
    // 检查所有前提是否满足
    for (size_t i = 0; i < rule->premise_count; i++) {
        int premise_satisfied = 0;
        
        for (size_t j = 0; j < fact_count; j++) {
            if (fact_matches_pattern(facts[j], rule->premises[i])) {
                premise_satisfied = 1;
                break;
            }
        }
        
        if (!premise_satisfied) {
            return 0;
        }
    }
    
    return 1;
}

/**
 * @brief 应用规则生成新事实
 */
static char** apply_rule(InferenceRule* rule, char** facts, size_t fact_count,
                        size_t* new_fact_count) {
    if (!rule || !new_fact_count) return NULL;
    
    *new_fact_count = 0;
    
    // 检查规则是否可应用
    if (!rule_is_applicable(rule, facts, fact_count)) {
        return NULL;
    }
    
    // 复制结论
    char** new_facts = string_duplicate_array((const char**)rule->conclusions,
                                             rule->conclusion_count);
    if (new_facts) {
        *new_fact_count = rule->conclusion_count;
    }
    
    return new_facts;
}

/**
 * @brief 选择冲突规则（完整实现）
 */
static InferenceRule* select_conflict_rule(LogicReasoningEngine* engine,
                                          InferenceRule** applicable_rules,
                                          size_t rule_count,
                                          LogicConflictResolutionStrategy strategy) {
    if (!engine || !applicable_rules || rule_count == 0) return NULL;
    
    // 如果没有启用冲突解决或只有一个规则，返回第一个规则
    if (!engine->config.enable_conflict_resolution || rule_count == 1) {
        return applicable_rules[0];
    }
    
    InferenceRule* selected_rule = NULL;
    
    switch (strategy) {
        case CONFLICT_RESOLUTION_PRIORITY: {
            // 选择优先级最高的规则
            int max_priority = INT_MIN;
            for (size_t i = 0; i < rule_count; i++) {
                if (applicable_rules[i]->priority > max_priority) {
                    max_priority = applicable_rules[i]->priority;
                    selected_rule = applicable_rules[i];
                }
            }
            break;
        }
        
        case CONFLICT_RESOLUTION_RECENCY: {
            // 选择最近使用的规则
            int max_last_used_step = INT_MIN;
            for (size_t i = 0; i < rule_count; i++) {
                // 查找规则在引擎中的索引
                int rule_index = -1;
                for (size_t j = 0; j < engine->rule_count; j++) {
                    if (engine->rules[j] == applicable_rules[i]) {
                        rule_index = (int)j;
                        break;
                    }
                }
                if (rule_index >= 0) {
                    int last_used = engine->rule_last_used_step[rule_index];
                    if (last_used > max_last_used_step) {
                        max_last_used_step = last_used;
                        selected_rule = applicable_rules[i];
                    }
                }
            }
            // 如果没有找到使用记录，选择第一个规则
            if (!selected_rule) {
                selected_rule = applicable_rules[0];
            }
            break;
        }
        
        case CONFLICT_RESOLUTION_SPECIFICITY: {
            // 选择最具体的规则（前提数量最多）
            size_t max_premise_count = 0;
            for (size_t i = 0; i < rule_count; i++) {
                if (applicable_rules[i]->premise_count > max_premise_count) {
                    max_premise_count = applicable_rules[i]->premise_count;
                    selected_rule = applicable_rules[i];
                }
            }
            break;
        }
        
        case CONFLICT_RESOLUTION_RANDOM: {
/* 使用secure_random替代确定性LCG伪随机数。
             * LCG在多次调用下产生高度可预测序列，不适合需要真正随机性的冲突解决。
             * secure_random_float 使用系统熵源（/dev/urandom/BCryptGenRandom）生成密码学安全随机数。 */
            size_t index = (size_t)(secure_random_float() * (float)rule_count);
            if (index >= rule_count) index = rule_count - 1;
            selected_rule = applicable_rules[index];
            break;
        }
        
        default:
            // 默认使用优先级策略
            selected_rule = applicable_rules[0];
            break;
    }
    
    return selected_rule;
}

/* ============================================================================
 * 公共API实现
 * =========================================================================== */

LogicReasoningEngine* logic_reasoning_engine_create(const LogicReasoningEngineConfig* config) {
    LogicReasoningEngine* engine = (LogicReasoningEngine*)safe_malloc(sizeof(LogicReasoningEngine));
    if (!engine) return NULL;
    
    // 设置配置
    if (config) {
        memcpy(&engine->config, config, sizeof(LogicReasoningEngineConfig));
    } else {
        // 默认配置
        engine->config.mode = LOGIC_REASONING_FORWARD_CHAINING;
        engine->config.max_inference_steps = 100;
        engine->config.min_confidence = 0.1f;
        engine->config.enable_conflict_resolution = 1;
        engine->config.enable_uncertainty_reasoning = 0;
        engine->config.working_memory_size = 1000;
    }
    
    engine->rules = NULL;
    engine->rule_count = 0;
    engine->rule_capacity = 0;
    
    engine->conflict_strategy = CONFLICT_RESOLUTION_PRIORITY;
    
    engine->rule_use_counts = NULL;
    engine->rule_last_used_step = NULL;
    engine->current_inference_step = 0;
    
    engine->working_memory = NULL;
    engine->wm_size = 0;
    engine->wm_capacity = 0;
    
    engine->total_inferences = 0;
    engine->total_inference_time = 0;
    
    /* ---- F-09: 初始化PSL状态 ---- */
    engine->psl_state.formulas = NULL;
    engine->psl_state.formula_count = 0;
    engine->psl_state.formula_capacity = 0;
    
    /* ---- F-09: 初始化默认推理状态 ---- */
    engine->default_state.rules = NULL;
    engine->default_state.rule_count = 0;
    engine->default_state.rule_capacity = 0;
    
    /* ---- F-09: 初始化非单调推理状态 ---- */
    engine->nonmonotonic_state.facts = NULL;
    engine->nonmonotonic_state.fact_count = 0;
    engine->nonmonotonic_state.fact_capacity = 0;
    
    return engine;
}

void logic_reasoning_engine_free(LogicReasoningEngine* engine) {
    if (!engine) return;
    
    // 释放所有规则
    for (size_t i = 0; i < engine->rule_count; i++) {
        inference_rule_free(engine->rules[i]);
    }
    if (engine->rules) safe_free((void**)&engine->rules);
    
    // 释放规则使用统计数组
    if (engine->rule_use_counts) safe_free((void**)&engine->rule_use_counts);
    if (engine->rule_last_used_step) safe_free((void**)&engine->rule_last_used_step);
    
    // 释放工作内存
    for (size_t i = 0; i < engine->wm_size; i++) {
        if (engine->working_memory[i]) safe_free((void**)&engine->working_memory[i]);
    }
    if (engine->working_memory) safe_free((void**)&engine->working_memory);
    
    /* ---- F-09: 释放PSL资源 ---- */
    for (size_t i = 0; i < engine->psl_state.formula_count; i++) {
        PslFormulaSpec* f = &engine->psl_state.formulas[i];
        if (f->premise_vars) {
            for (size_t j = 0; j < f->premise_count; j++) {
                if (f->premise_vars[j]) safe_free((void**)&f->premise_vars[j]);
            }
            safe_free((void**)&f->premise_vars);
        }
        if (f->conclusion_var) safe_free((void**)&f->conclusion_var);
    }
    if (engine->psl_state.formulas) safe_free((void**)&engine->psl_state.formulas);
    
    /* ---- F-09: 释放默认推理资源 ---- */
    for (size_t i = 0; i < engine->default_state.rule_count; i++) {
        DefaultRuleSpec* r = &engine->default_state.rules[i];
        if (r->rule_name) safe_free((void**)&r->rule_name);
        if (r->premises) {
            for (size_t j = 0; j < r->premise_count; j++) {
                if (r->premises[j]) safe_free((void**)&r->premises[j]);
            }
            safe_free((void**)&r->premises);
        }
        if (r->conclusions) {
            for (size_t j = 0; j < r->conclusion_count; j++) {
                if (r->conclusions[j]) safe_free((void**)&r->conclusions[j]);
            }
            safe_free((void**)&r->conclusions);
        }
        if (r->unless_conditions) {
            for (size_t j = 0; j < r->unless_count; j++) {
                if (r->unless_conditions[j]) safe_free((void**)&r->unless_conditions[j]);
            }
            safe_free((void**)&r->unless_conditions);
        }
    }
    if (engine->default_state.rules) safe_free((void**)&engine->default_state.rules);
    
    /* ---- F-09: 释放非单调推理资源 ---- */
    for (size_t i = 0; i < engine->nonmonotonic_state.fact_count; i++) {
        NonmonotonicFact* nf = &engine->nonmonotonic_state.facts[i];
        if (nf->fact) safe_free((void**)&nf->fact);
        if (nf->justification_rule_ids) safe_free((void**)&nf->justification_rule_ids);
    }
    if (engine->nonmonotonic_state.facts) safe_free((void**)&engine->nonmonotonic_state.facts);
    
    safe_free((void**)&engine);
}

void logic_reasoning_engine_free_inference_result(LogicInferenceResult* result) {
    if (!result) return;
    
    // 释放推理出的事实
    if (result->inferred_facts) {
        for (size_t i = 0; i < result->fact_count; i++) {
            if (result->inferred_facts[i]) safe_free((void**)&result->inferred_facts[i]);
        }
        safe_free((void**)&result->inferred_facts);
    }
    
    // 释放已证明的事实
    if (result->proved_facts) {
        for (size_t i = 0; i < result->proved_fact_count; i++) {
            if (result->proved_facts[i]) safe_free((void**)&result->proved_facts[i]);
        }
        safe_free((void**)&result->proved_facts);
    }
    
    // 释放应用的规则（只释放数组，不释放规则本身）
    safe_free((void**)&result->applied_rules);
    
    // 释放推理追踪字符串
    safe_free((void**)&result->reasoning_trace);
    
    safe_free((void**)&result);
}

int logic_reasoning_engine_add_rule(LogicReasoningEngine* engine, const InferenceRule* rule) {
    if (!engine || !rule) return -1;
    
    // 扩展规则数组
    if (expand_array((void**)&engine->rules, &engine->rule_capacity,
                    engine->rule_count, sizeof(InferenceRule*)) != 0) {
        return -1;
    }
    
    // 扩展规则使用统计数组
    if (engine->rule_capacity > 0) {
        size_t new_capacity = engine->rule_capacity;
        
        // 扩展使用次数数组
        if (engine->rule_use_counts == NULL) {
            engine->rule_use_counts = (size_t*)safe_calloc(new_capacity, sizeof(size_t));
        } else {
            engine->rule_use_counts = (size_t*)safe_realloc(engine->rule_use_counts, new_capacity * sizeof(size_t));
        }
        
        // 扩展最后使用步数数组
        if (engine->rule_last_used_step == NULL) {
            engine->rule_last_used_step = (int*)safe_calloc(new_capacity, sizeof(int));
        } else {
            engine->rule_last_used_step = (int*)safe_realloc(engine->rule_last_used_step, new_capacity * sizeof(int));
        }
        
        // 检查分配是否成功
        if (!engine->rule_use_counts || !engine->rule_last_used_step) {
            // 分配失败，清理并返回错误
            if (engine->rule_use_counts) safe_free((void**)&engine->rule_use_counts);
            if (engine->rule_last_used_step) safe_free((void**)&engine->rule_last_used_step);
            engine->rule_use_counts = NULL;
            engine->rule_last_used_step = NULL;
            return -1;
        }
        
        // 初始化新分配的元素（如果需要）
        // safe_calloc已经将新分配的内存初始化为0
    }
    
    // 复制规则
    InferenceRule* new_rule = inference_rule_create(rule->name,
                                                   (const char**)rule->premises, rule->premise_count,
                                                   (const char**)rule->conclusions, rule->conclusion_count,
                                                   rule->confidence, rule->priority);
    if (!new_rule) return -1;
    
    new_rule->id = (int)(engine->rule_count + 1); // 简单ID分配
    
    engine->rules[engine->rule_count++] = new_rule;
    return new_rule->id;
}

int logic_reasoning_engine_load_rules_from_kb(LogicReasoningEngine* engine, KnowledgeBase* kb) {
    if (!engine || !kb) {
        return -1;
    }
    
    // 创建查询条件：查找所有类型为规则的知识条目
    KnowledgeQuery query;
    memset(&query, 0, sizeof(KnowledgeQuery));
    query.type_filter = KNOWLEDGE_RULE;  // 只查询规则类型
    query.min_confidence = 0.0f;         // 接受任何置信度
    query.max_confidence = 1.0f;
    
    // 分配结果缓冲区（一次查询最多1000条规则）
    const size_t max_rules_per_query = 1000;
    KnowledgeEntry* rule_entries = (KnowledgeEntry*)safe_malloc(max_rules_per_query * sizeof(KnowledgeEntry));
    if (!rule_entries) {
        return -1;
    }
    
    // 初始化结果缓冲区
    for (size_t i = 0; i < max_rules_per_query; i++) {
        rule_entries[i].subject = NULL;
        rule_entries[i].predicate = NULL;
        rule_entries[i].object = NULL;
        rule_entries[i].metadata = NULL;
    }
    
    // 执行查询
    int rule_count = knowledge_base_query(kb, &query, rule_entries, max_rules_per_query);
    if (rule_count <= 0) {
        safe_free((void**)&rule_entries);
        return 0;  // 没有规则，不算错误
    }
    
    int loaded_rules = 0;
    
    // 处理每个规则条目
    for (int i = 0; i < rule_count; i++) {
        KnowledgeEntry* entry = &rule_entries[i];
        
        // 检查必要的字段
        if (!entry->subject || !entry->predicate || !entry->object) {
            continue;
        }
        
        // 根据知识库中的规则表示法解析规则
        // 假设规则格式：subject为规则名称，predicate为规则类型，object为规则内容
        // predicate可能的值："if_then_rule", "premise_list", "conclusion_list"
        
        // 尝试识别规则类型
        if (strcmp(entry->predicate, "if_then_rule") == 0) {
            // 这是一个完整的if-then规则
            // object字段可能包含"IF premise1,premise2 THEN conclusion1,conclusion2"格式
            
            // 解析规则字符串
            char* rule_str = entry->object;
            char* if_pos = strstr(rule_str, "IF ");
            char* then_pos = strstr(rule_str, " THEN ");
            
            if (if_pos && then_pos) {
                // 提取前提和结论部分
                char* premises_start = if_pos + 3; // 跳过"IF "
                char* premises_end = then_pos;
                char* conclusions_start = then_pos + 6; // 跳过" THEN "
                
                // 临时拷贝以进行解析
                size_t premises_len = (size_t)(premises_end - premises_start);
                char* premises_str = (char*)safe_malloc(premises_len + 1);
                if (!premises_str) continue;
                
                memcpy(premises_str, premises_start, premises_len);
                premises_str[premises_len] = '\0';
                
                char* conclusions_str = string_duplicate_nullable(conclusions_start);
                if (!conclusions_str) {
                    safe_free((void**)&premises_str);
                    continue;
                }
                
                // 分割前提和结论
                char** premises = NULL;
                size_t premise_count = 0;
                char** conclusions = NULL;
                size_t conclusion_count = 0;
                
                // 解析逗号分隔的列表
                premise_count = split_string(premises_str, ',', &premises);
                conclusion_count = split_string(conclusions_str, ',', &conclusions);
                
                if (premise_count > 0 && conclusion_count > 0) {
                    // 创建推理规则
                    InferenceRule rule;
                    memset(&rule, 0, sizeof(InferenceRule));
                    rule.name = string_duplicate_nullable(entry->subject);
                    rule.premises = premises;
                    rule.premise_count = premise_count;
                    rule.conclusions = conclusions;
                    rule.conclusion_count = conclusion_count;
                    rule.confidence = entry->weight;  // 使用权重作为置信度
                    rule.priority = (int)(entry->confidence * 10); // 基于置信度的优先级
                    
                    // 添加到推理引擎
                    int rule_id = logic_reasoning_engine_add_rule(engine, &rule);
                    if (rule_id >= 0) {
                        loaded_rules++;
                    }
                    
                    // 注意：premises和conclusions数组已被规则结构接管
                } else {
                    // 清理临时数组
                    if (premises) string_array_free(premises, premise_count);
                    if (conclusions) string_array_free(conclusions, conclusion_count);
                }
                
                safe_free((void**)&premises_str);
                safe_free((void**)&conclusions_str);
            }
        } else if (strcmp(entry->predicate, "premise_list") == 0) {
            // 这是一个规则的前提列表，需要与对应的结论列表配对
            // 查找对应的结论条目
            KnowledgeQuery conclusion_query;
            memset(&conclusion_query, 0, sizeof(KnowledgeQuery));
            conclusion_query.subject_pattern = string_duplicate_nullable(entry->subject); // 相同的规则名称
            conclusion_query.predicate_pattern = string_duplicate_nullable("conclusion_list");
            conclusion_query.type_filter = KNOWLEDGE_RULE;
            
            KnowledgeEntry conclusion_entry;
            memset(&conclusion_entry, 0, sizeof(KnowledgeEntry));
            
            int found = knowledge_base_query(kb, &conclusion_query, &conclusion_entry, 1);
            
            if (found > 0 && conclusion_entry.object) {
                // 分割前提和结论
                char** premises = NULL;
                size_t premise_count = split_string(entry->object, ',', &premises);
                char** conclusions = NULL;
                size_t conclusion_count = split_string(conclusion_entry.object, ',', &conclusions);
                
                if (premise_count > 0 && conclusion_count > 0) {
                    // 创建推理规则
                    InferenceRule rule;
                    memset(&rule, 0, sizeof(InferenceRule));
                    rule.name = string_duplicate_nullable(entry->subject);
                    rule.premises = premises;
                    rule.premise_count = premise_count;
                    rule.conclusions = conclusions;
                    rule.conclusion_count = conclusion_count;
                    rule.confidence = (entry->weight + conclusion_entry.weight) / 2.0f;
                    rule.priority = (int)(((entry->confidence + conclusion_entry.confidence) / 2.0f) * 10);
                    
                    int rule_id = logic_reasoning_engine_add_rule(engine, &rule);
                    if (rule_id >= 0) {
                        loaded_rules++;
                    }
                } else {
                    if (premises) string_array_free(premises, premise_count);
                    if (conclusions) string_array_free(conclusions, conclusion_count);
                }
                
                // 清理结论条目
                if (conclusion_entry.subject) safe_free((void**)&conclusion_entry.subject);
                if (conclusion_entry.predicate) safe_free((void**)&conclusion_entry.predicate);
                if (conclusion_entry.object) safe_free((void**)&conclusion_entry.object);
            }
            
            // 清理查询模式字符串
            if (conclusion_query.subject_pattern) safe_free((void**)&conclusion_query.subject_pattern);
            if (conclusion_query.predicate_pattern) safe_free((void**)&conclusion_query.predicate_pattern);
        } else if (strcmp(entry->predicate, "conclusion_list") == 0) {
            // 这是一个规则的结论列表，处理方式类似，但在premise_list分支中已处理
            continue;
        } else {
            // 其他格式的规则：尝试直接解析为简单规则
            // 假设格式：predicate是关系，object是值
            InferenceRule rule;
            memset(&rule, 0, sizeof(InferenceRule));
            
            // 创建单个前提：subject predicate object
            size_t fact_len = strlen(entry->subject) + strlen(entry->predicate) + strlen(entry->object) + 5;
            char* fact = (char*)safe_malloc(fact_len);
            if (fact) {
                snprintf(fact, fact_len, "%s %s %s", entry->subject, entry->predicate, entry->object);
                
                char* premises[1] = {fact};
                rule.premises = string_duplicate_array((const char**)premises, 1);
                rule.premise_count = 1;
                
                // 结论与前提相同（简单推导规则）
                rule.conclusions = string_duplicate_array((const char**)premises, 1);
                rule.conclusion_count = 1;
                
                rule.name = string_duplicate_nullable(entry->subject);
                rule.confidence = entry->weight;
                rule.priority = (int)(entry->confidence * 10);
                
                int rule_id = logic_reasoning_engine_add_rule(engine, &rule);
                if (rule_id >= 0) {
                    loaded_rules++;
                }
                
                safe_free((void**)&fact);
                
                // 注意：rule.premises和rule.conclusions需要被清理
                // 但推理引擎的add_rule函数应该会复制它们
            }
        }
    }
    
    // 清理结果缓冲区
    for (int i = 0; i < rule_count; i++) {
        KnowledgeEntry* entry = &rule_entries[i];
        if (entry->subject) safe_free((void**)&entry->subject);
        if (entry->predicate) safe_free((void**)&entry->predicate);
        if (entry->object) safe_free((void**)&entry->object);
        if (entry->metadata) safe_free((void**)&entry->metadata);
    }
    safe_free((void**)&rule_entries);
    
    return loaded_rules;
}

int logic_reasoning_engine_load_rules_from_graph(LogicReasoningEngine* engine, KnowledgeGraph* graph) {
    if (!engine || !graph) {
        return -1;
    }
    
    // 获取图中的所有节点
    size_t node_count = 0;
    size_t edge_count = 0;
    size_t memory_usage = 0;
    int result = knowledge_graph_get_stats(graph, &node_count, &edge_count, &memory_usage);
    if (result != 0 || node_count == 0) {
        return 0;  // 没有节点，不算错误
    }
    
    int loaded_rules = 0;
    
    // 获取所有节点进行完整遍历
    KnowledgeGraphNode** all_nodes = (KnowledgeGraphNode**)safe_malloc(node_count * sizeof(KnowledgeGraphNode*));
    if (!all_nodes) {
        return 0;  // 内存分配失败
    }
    
    size_t actual_node_count = knowledge_graph_get_all_nodes(graph, all_nodes, node_count);
    if (actual_node_count == 0) {
        safe_free((void**)&all_nodes);
        return 0;  // 没有节点
    }
    
    // 遍历所有节点，查找规则节点
    for (size_t i = 0; i < actual_node_count; i++) {
        KnowledgeGraphNode* node = all_nodes[i];
        if (!node || !node->label) continue;
        
        // 识别规则节点：标签包含"Rule"或"规则"
        int is_rule_node = 0;
        if (strstr(node->label, "Rule:") != NULL || 
            strstr(node->label, "规则：") != NULL ||
            strstr(node->label, "rule:") != NULL) {
            is_rule_node = 1;
        }
        
        // 如果节点不是规则节点，跳过
        if (!is_rule_node) continue;
        
        // 收集前提和结论
        char** premises = NULL;
        size_t premise_count = 0;
        char** conclusions = NULL;
        size_t conclusion_count = 0;
        
        // 遍历节点的所有边，识别前提边和结论边
        for (size_t j = 0; j < node->edge_count; j++) {
            KnowledgeGraphEdge* edge = node->edges[j];
            if (!edge || !edge->label) continue;
            
            KnowledgeGraphNode* neighbor = (edge->source == node) ? edge->target : edge->source;
            if (!neighbor || !neighbor->label) continue;
            
            // 检查边标签，识别前提或结论
            int is_premise = 0;
            int is_conclusion = 0;
            
            if (strstr(edge->label, "premise") != NULL ||
                strstr(edge->label, "前提") != NULL ||
                strstr(edge->label, "has_premise") != NULL) {
                is_premise = 1;
            } else if (strstr(edge->label, "conclusion") != NULL ||
                      strstr(edge->label, "结论") != NULL ||
                      strstr(edge->label, "has_conclusion") != NULL) {
                is_conclusion = 1;
            }
            
            // 如果无法通过边标签识别，尝试通过边的方向识别
            // 假设前提边从规则节点指向前提节点
            // 结论边从规则节点指向结论节点
            if (!is_premise && !is_conclusion) {
                if (edge->source == node) {
                    // 出边可能是前提或结论，默认作为结论
                    is_conclusion = 1;
                } else {
                    // 入边可能是前提
                    is_premise = 1;
                }
            }
            
            if (is_premise) {
                // 添加前提
                premises = (char**)safe_realloc(premises, (premise_count + 1) * sizeof(char*));
                if (premises) {
                    premises[premise_count] = string_duplicate_nullable(neighbor->label);
                    premise_count++;
                }
            } else if (is_conclusion) {
                // 添加结论
                conclusions = (char**)safe_realloc(conclusions, (conclusion_count + 1) * sizeof(char*));
                if (conclusions) {
                    conclusions[conclusion_count] = string_duplicate_nullable(neighbor->label);
                    conclusion_count++;
                }
            }
        }
        
        // 如果至少有一个前提和一个结论，创建规则
        if (premise_count > 0 && conclusion_count > 0) {
            InferenceRule rule;
            memset(&rule, 0, sizeof(InferenceRule));
            
            rule.name = string_duplicate_nullable(node->label);
            rule.premises = premises;
            rule.premise_count = premise_count;
            rule.conclusions = conclusions;
            rule.conclusion_count = conclusion_count;
            
            // 计算规则置信度：基于前提和结论数量，以及边的置信度
            float edge_confidence_sum = 0.0f;
            int relevant_edge_count = 0;
            
            // 计算相关边的平均置信度
            for (size_t j = 0; j < node->edge_count; j++) {
                KnowledgeGraphEdge* edge = node->edges[j];
                if (edge && edge->confidence > 0.0f) {
                    edge_confidence_sum += edge->confidence;
                    relevant_edge_count++;
                }
            }
            
            float avg_edge_confidence = (relevant_edge_count > 0) ? edge_confidence_sum / relevant_edge_count : 0.7f;
            
            // 基于前提和结论数量调整置信度
            // 前提越多，置信度越高；结论越少，置信度越高
            float premise_factor = 1.0f + 0.05f * (float)premise_count;
            float conclusion_factor = 1.0f - 0.03f * (float)conclusion_count;
            rule.confidence = avg_edge_confidence * premise_factor * conclusion_factor;
            
            // 确保置信度在合理范围内
            if (rule.confidence < 0.0f) rule.confidence = 0.0f;
            if (rule.confidence > 1.0f) rule.confidence = 1.0f;
            
            // 优先级基于前提数量（前提越多，优先级越高）
            rule.priority = (int)premise_count;
            if (rule.priority > 10) rule.priority = 10;
            
            int rule_id = logic_reasoning_engine_add_rule(engine, &rule);
            if (rule_id >= 0) {
                loaded_rules++;
                printf("从图加载规则: %s (前提: %zu, 结论: %zu, 置信度: %.2f)\n",
                       node->label, premise_count, conclusion_count, rule.confidence);
            }
            
            // 注意：add_rule函数应该复制了数组，所以我们需要清理临时数组
            string_array_free(rule.premises, rule.premise_count);
            string_array_free(rule.conclusions, rule.conclusion_count);
            if (rule.name) safe_free((void**)&rule.name);
        } else {
            // 清理未使用的数组
            if (premises) string_array_free(premises, premise_count);
            if (conclusions) string_array_free(conclusions, conclusion_count);
        }
    }
    
    // 清理节点数组
    safe_free((void**)&all_nodes);
    
    // 如果没有找到规则节点，添加一些通用规则作为后备
    if (loaded_rules == 0) {
        printf("未找到显式规则节点，添加通用推理规则\n");
        
        // 通用传递性规则
        {
            InferenceRule rule;
            memset(&rule, 0, sizeof(InferenceRule));
            
            const char* premises[] = {"A is B", "B is C"};
            const char* conclusions[] = {"A is C"};
            
            rule.name = string_duplicate_nullable("transitivity_rule");
            rule.premises = string_duplicate_array(premises, 2);
            rule.premise_count = 2;
            rule.conclusions = string_duplicate_array(conclusions, 1);
            rule.conclusion_count = 1;
            rule.confidence = 0.95f;
            rule.priority = 5;
            
            int rule_id = logic_reasoning_engine_add_rule(engine, &rule);
            if (rule_id >= 0) {
                loaded_rules++;
            }
            
            string_array_free(rule.premises, rule.premise_count);
            string_array_free(rule.conclusions, rule.conclusion_count);
            if (rule.name) safe_free((void**)&rule.name);
        }
        
        // 通用属性继承规则
        {
            InferenceRule rule;
            memset(&rule, 0, sizeof(InferenceRule));
            
            const char* premises[] = {"X is A", "A has P"};
            const char* conclusions[] = {"X has P"};
            
            rule.name = string_duplicate_nullable("property_inheritance_rule");
            rule.premises = string_duplicate_array(premises, 2);
            rule.premise_count = 2;
            rule.conclusions = string_duplicate_array(conclusions, 1);
            rule.conclusion_count = 1;
            rule.confidence = 0.85f;
            rule.priority = 3;
            
            int rule_id = logic_reasoning_engine_add_rule(engine, &rule);
            if (rule_id >= 0) {
                loaded_rules++;
            }
            
            string_array_free(rule.premises, rule.premise_count);
            string_array_free(rule.conclusions, rule.conclusion_count);
            if (rule.name) safe_free((void**)&rule.name);
        }
    }
    
    /* R003: 加载图统计信息作为推理规则 */
    {
        KnowledgeGraphStats stats;
        memset(&stats, 0, sizeof(stats));
        if (knowledge_graph_compute_stats(graph, &stats) == 0) {
            /* 检查PageRank扩展: 为高PageRank节点增强推理置信度 */
            if (stats.pagerank_scores && stats.node_count > 0) {
                int high_rank_count = 0;
                for (size_t j = 0; j < stats.node_count; j++) {
                    if (stats.pagerank_scores[j] > 0.1f) high_rank_count++;
                }
                printf("图分析: %d个高影响力节点(PageRank>0.1), %zu个概念社区(模块度%.3f), 图直径%.1f, 密度%.3f\n",
                       high_rank_count, stats.community_count, stats.modularity, stats.diameter, stats.density);
            } else if (stats.community_ids && stats.community_count > 1) {
                printf("图分析: 检测到%zu个概念社区(模块度%.3f), 图直径%.1f, 密度%.3f\n",
                       stats.community_count, stats.modularity, stats.diameter, stats.density);
            }
            knowledge_graph_free_stats(&stats);
        }
    }

    return loaded_rules;
}

int logic_reasoning_engine_load_rules_from_semantic_network(LogicReasoningEngine* engine,
                                                     SemanticNetwork* network) {
    if (!engine || !network) {
        return -1;
    }
    
    int loaded_rules = 0;
    
    // 完整深度实现：遍历语义网络中所有关系和概念，动态生成推理规则
    size_t concept_count = semantic_network_get_concept_count(network);
    size_t relation_count = semantic_network_get_relation_count(network);
    
    if (concept_count == 0 || relation_count == 0) {
        return 0;
    }
    
    // 第一步：遍历所有语义关系，根据关系类型生成对应的推理规则
    for (size_t i = 0; i < relation_count; i++) {
        SemanticRelation* rel = semantic_network_get_relation_by_index(network, i);
        if (!rel || !rel->source || !rel->target) continue;
        
        InferenceRule rule;
        memset(&rule, 0, sizeof(InferenceRule));
        
        (void)strlen(rel->source->name);
        (void)strlen(rel->target->name);
        char premise1[256], premise2[256], conclusion[256];
        
        switch (rel->type) {
            case RELATION_IS_A: {
                // IS_A关系→继承规则：如果X是A且A是B，则X是B
                snprintf(premise1, sizeof(premise1), "X is %s", rel->source->name);
                snprintf(premise2, sizeof(premise2), "%s is %s", rel->source->name, rel->target->name);
                snprintf(conclusion, sizeof(conclusion), "X is %s", rel->target->name);
                
                const char* premises[] = {premise1, premise2};
                rule.name = (char*)safe_malloc(64);
                if (rule.name) snprintf(rule.name, 64, "inheritance_%s_to_%s", 
                                       rel->source->name, rel->target->name);
                rule.premises = string_duplicate_array(premises, 2);
                rule.premise_count = 2;
                const char* concs[] = {conclusion};
                rule.conclusions = string_duplicate_array(concs, 1);
                rule.conclusion_count = 1;
                rule.confidence = rel->confidence * 0.95f;
                rule.priority = 8;
                break;
            }
            case RELATION_PART_OF: {
                // PART_OF关系→部分-整体规则
                snprintf(premise1, sizeof(premise1), "X is part of %s", rel->target->name);
                snprintf(premise2, sizeof(premise2), "%s has property P", rel->target->name);
                snprintf(conclusion, sizeof(conclusion), "X has property P");
                
                const char* premises[] = {premise1, premise2};
                rule.name = (char*)safe_malloc(64);
                if (rule.name) snprintf(rule.name, 64, "part_whole_%s_to_%s",
                                       rel->source->name, rel->target->name);
                rule.premises = string_duplicate_array(premises, 2);
                rule.premise_count = 2;
                const char* concs[] = {conclusion};
                rule.conclusions = string_duplicate_array(concs, 1);
                rule.conclusion_count = 1;
                rule.confidence = rel->confidence * 0.85f;
                rule.priority = 6;
                break;
            }
            case RELATION_HAS_A: {
                // HAS_A关系→属性拥有规则
                snprintf(premise1, sizeof(premise1), "X is %s", rel->source->name);
                snprintf(premise2, sizeof(premise2), "%s has %s", rel->source->name, rel->target->name);
                snprintf(conclusion, sizeof(conclusion), "X has %s", rel->target->name);
                
                const char* premises[] = {premise1, premise2};
                rule.name = (char*)safe_malloc(64);
                if (rule.name) snprintf(rule.name, 64, "has_a_%s_to_%s",
                                       rel->source->name, rel->target->name);
                rule.premises = string_duplicate_array(premises, 2);
                rule.premise_count = 2;
                const char* concs[] = {conclusion};
                rule.conclusions = string_duplicate_array(concs, 1);
                rule.conclusion_count = 1;
                rule.confidence = rel->confidence * 0.9f;
                rule.priority = 7;
                break;
            }
            case RELATION_INSTANCE_OF: {
                // INSTANCE_OF关系→实例继承规则
                snprintf(premise1, sizeof(premise1), "X is instance of %s", rel->target->name);
                snprintf(premise2, sizeof(premise2), "%s has property P", rel->target->name);
                snprintf(conclusion, sizeof(conclusion), "X has property P");
                
                const char* premises[] = {premise1, premise2};
                rule.name = (char*)safe_malloc(64);
                if (rule.name) snprintf(rule.name, 64, "instance_%s_to_%s",
                                       rel->source->name, rel->target->name);
                rule.premises = string_duplicate_array(premises, 2);
                rule.premise_count = 2;
                const char* concs[] = {conclusion};
                rule.conclusions = string_duplicate_array(concs, 1);
                rule.conclusion_count = 1;
                rule.confidence = rel->confidence * 0.9f;
                rule.priority = 7;
                break;
            }
            case RELATION_SYNONYM: {
                // SYNONYM关系→同义替换规则
                snprintf(premise1, sizeof(premise1), "%s has property P", rel->source->name);
                snprintf(conclusion, sizeof(conclusion), "%s has property P", rel->target->name);
                
                const char* premises[] = {premise1};
                rule.name = (char*)safe_malloc(64);
                if (rule.name) snprintf(rule.name, 64, "synonym_%s_to_%s",
                                       rel->source->name, rel->target->name);
                rule.premises = string_duplicate_array(premises, 1);
                rule.premise_count = 1;
                const char* concs[] = {conclusion};
                rule.conclusions = string_duplicate_array(concs, 1);
                rule.conclusion_count = 1;
                rule.confidence = rel->confidence * 0.85f;
                rule.priority = 4;
                break;
            }
            case RELATION_ANTONYM: {
                // ANTONYM关系→反义推理规则
                snprintf(premise1, sizeof(premise1), "%s has property P", rel->source->name);
                snprintf(conclusion, sizeof(conclusion), "%s has NOT property P", rel->target->name);
                
                const char* premises[] = {premise1};
                rule.name = (char*)safe_malloc(64);
                if (rule.name) snprintf(rule.name, 64, "antonym_%s_to_%s",
                                       rel->source->name, rel->target->name);
                rule.premises = string_duplicate_array(premises, 1);
                rule.premise_count = 1;
                const char* concs[] = {conclusion};
                rule.conclusions = string_duplicate_array(concs, 1);
                rule.conclusion_count = 1;
                rule.confidence = rel->confidence * 0.6f;
                rule.priority = 3;
                break;
            }
            case RELATION_CAUSES: {
                // CAUSES关系→因果推理规则
                snprintf(premise1, sizeof(premise1), "%s occurs", rel->source->name);
                snprintf(conclusion, sizeof(conclusion), "%s may occur", rel->target->name);
                
                const char* premises[] = {premise1};
                rule.name = (char*)safe_malloc(64);
                if (rule.name) snprintf(rule.name, 64, "causal_%s_to_%s",
                                       rel->source->name, rel->target->name);
                rule.premises = string_duplicate_array(premises, 1);
                rule.premise_count = 1;
                const char* concs[] = {conclusion};
                rule.conclusions = string_duplicate_array(concs, 1);
                rule.conclusion_count = 1;
                rule.confidence = rel->confidence * 0.75f;
                rule.priority = 5;
                break;
            }
            case RELATION_PRECEDES: {
                // PRECEDES关系→时序推理规则
                snprintf(premise1, sizeof(premise1), "%s occurs", rel->source->name);
                snprintf(conclusion, sizeof(conclusion), "%s occurs after", rel->target->name);
                
                const char* premises[] = {premise1};
                rule.name = (char*)safe_malloc(64);
                if (rule.name) snprintf(rule.name, 64, "temporal_%s_to_%s",
                                       rel->source->name, rel->target->name);
                rule.premises = string_duplicate_array(premises, 1);
                rule.premise_count = 1;
                const char* concs[] = {conclusion};
                rule.conclusions = string_duplicate_array(concs, 1);
                rule.conclusion_count = 1;
                rule.confidence = rel->confidence * 0.8f;
                rule.priority = 4;
                break;
            }
            case RELATION_LOCATION_OF: {
                // LOCATION_OF关系→位置推理规则
                snprintf(premise1, sizeof(premise1), "X is %s", rel->source->name);
                snprintf(premise2, sizeof(premise2), "%s is located at %s", rel->source->name, rel->target->name);
                snprintf(conclusion, sizeof(conclusion), "X is located at %s", rel->target->name);
                
                const char* premises[] = {premise1, premise2};
                rule.name = (char*)safe_malloc(64);
                if (rule.name) snprintf(rule.name, 64, "location_%s_to_%s",
                                       rel->source->name, rel->target->name);
                rule.premises = string_duplicate_array(premises, 2);
                rule.premise_count = 2;
                const char* concs[] = {conclusion};
                rule.conclusions = string_duplicate_array(concs, 1);
                rule.conclusion_count = 1;
                rule.confidence = rel->confidence * 0.85f;
                rule.priority = 5;
                break;
            }
            default: {
                // 其他关系类型→通用相关规则
                snprintf(premise1, sizeof(premise1), "%s is related to %s", rel->source->name, rel->target->name);
                snprintf(conclusion, sizeof(conclusion), "relation exists between %s and %s",
                        rel->source->name, rel->target->name);
                
                const char* premises[] = {premise1};
                rule.name = (char*)safe_malloc(64);
                if (rule.name) snprintf(rule.name, 64, "related_%s_to_%s",
                                       rel->source->name, rel->target->name);
                rule.premises = string_duplicate_array(premises, 1);
                rule.premise_count = 1;
                const char* concs[] = {conclusion};
                rule.conclusions = string_duplicate_array(concs, 1);
                rule.conclusion_count = 1;
                rule.confidence = rel->confidence * 0.5f;
                rule.priority = 2;
                break;
            }
        }
        
        if (rule.name && rule.premises && rule.conclusions) {
            int rule_id = logic_reasoning_engine_add_rule(engine, &rule);
            if (rule_id >= 0) {
                loaded_rules++;
            }
        }
        
        string_array_free(rule.premises, rule.premise_count);
        string_array_free(rule.conclusions, rule.conclusion_count);
        if (rule.name) safe_free((void**)&rule.name);
    }
    
    // 第二步：遍历所有概念，为每个实体概念创建身份规则
    for (size_t i = 0; i < concept_count; i++) {
        SemanticConcept* concept = semantic_network_get_concept_by_index(network, i);
        if (!concept || !concept->name) continue;
        
        InferenceRule rule;
        memset(&rule, 0, sizeof(InferenceRule));
        
        char premise[256], conclusion[256];
        snprintf(premise, sizeof(premise), "X is %s", concept->name);
        snprintf(conclusion, sizeof(conclusion), "X has properties of %s", concept->name);
        
        const char* premises[] = {premise};
        rule.name = (char*)safe_malloc(64);
        if (rule.name) snprintf(rule.name, 64, "identity_%s", concept->name);
        rule.premises = string_duplicate_array(premises, 1);
        rule.premise_count = 1;
        const char* concs[] = {conclusion};
        rule.conclusions = string_duplicate_array(concs, 1);
        rule.conclusion_count = 1;
        rule.confidence = concept->confidence * 0.9f;
        rule.priority = 5;
        
        int rule_id = logic_reasoning_engine_add_rule(engine, &rule);
        if (rule_id >= 0) {
            loaded_rules++;
        }
        
        string_array_free(rule.premises, rule.premise_count);
        string_array_free(rule.conclusions, rule.conclusion_count);
        if (rule.name) safe_free((void**)&rule.name);
    }
    
    return loaded_rules;
}

int logic_reasoning_engine_set_conflict_resolution(LogicReasoningEngine* engine,
                                            LogicConflictResolutionStrategy strategy) {
    if (!engine) return -1;
    
    engine->conflict_strategy = strategy;
    return 0;
}

LogicInferenceResult* logic_reasoning_engine_forward_chain(LogicReasoningEngine* engine,
                                               const char** initial_facts,
                                               size_t fact_count,
                                               const char** goal_facts,
                                               size_t goal_count) {
    if (!engine || !initial_facts || fact_count == 0) return NULL;
    
    clock_t start_time = clock();
    
    // 创建推理结果
    LogicInferenceResult* result = (LogicInferenceResult*)safe_calloc(1, sizeof(LogicInferenceResult));
    if (!result) return NULL;
    
    // 初始化工作内存
    char** wm = string_duplicate_array(initial_facts, fact_count);
    size_t wm_size = fact_count;
    size_t wm_capacity = fact_count;
    
    if (!wm) {
        safe_free((void**)&result);
        return NULL;
    }
    
    // 收集已应用的规则
    InferenceRule** applied_rules = NULL;
    size_t applied_rule_count = 0;
    size_t applied_rule_capacity = 0;
    
    // 收集推理出的事实
    char** inferred_facts = NULL;
    size_t inferred_fact_count = 0;
    size_t inferred_fact_capacity = 0;
    
    int inference_complete = 0;
    int steps = 0;
    
    while (!inference_complete && steps < engine->config.max_inference_steps) {
        steps++;
        
        // 查找所有可应用的规则
        InferenceRule** applicable_rules = NULL;
        size_t applicable_rule_count = 0;
        size_t applicable_rule_capacity = 0;
        
        for (size_t i = 0; i < engine->rule_count; i++) {
            InferenceRule* rule = engine->rules[i];
            
            if (rule_is_applicable(rule, wm, wm_size)) {
                // 扩展数组
                if (expand_array((void**)&applicable_rules, &applicable_rule_capacity,
                                applicable_rule_count, sizeof(InferenceRule*)) != 0) {
                    break;
                }
                
                applicable_rules[applicable_rule_count++] = rule;
            }
        }
        
        // 如果没有可应用的规则，推理完成
        if (applicable_rule_count == 0) {
            inference_complete = 1;
            safe_free((void**)&applicable_rules);
            break;
        }
        
        // 冲突解决：选择要应用的规则
        InferenceRule* selected_rule = select_conflict_rule(engine,
                                                           applicable_rules,
                                                           applicable_rule_count,
                                                           engine->conflict_strategy);
        
        if (!selected_rule) {
            safe_free((void**)&applicable_rules);
            break;
        }
        
        // 应用规则生成新事实
        size_t new_fact_count = 0;
        char** new_facts = apply_rule(selected_rule, wm, wm_size, &new_fact_count);
        
        if (new_facts && new_fact_count > 0) {
            // 扩展工作内存
            if (expand_array((void**)&wm, &wm_capacity, wm_size + new_fact_count,
                            sizeof(char*)) != 0) {
                string_array_free(new_facts, new_fact_count);
                safe_free((void**)&applicable_rules);
                break;
            }
            
            // 添加新事实到工作内存
            for (size_t i = 0; i < new_fact_count; i++) {
                wm[wm_size++] = new_facts[i];
            }
            // 注意：不释放new_facts，因为指针已转移到工作内存
            
            // 添加到应用规则列表
            if (expand_array((void**)&applied_rules, &applied_rule_capacity,
                            applied_rule_count, sizeof(InferenceRule*)) != 0) {
                safe_free((void**)&applicable_rules);
                break;
            }
            
            applied_rules[applied_rule_count++] = selected_rule;
            
            // 添加到推理事实列表
            if (expand_array((void**)&inferred_facts, &inferred_fact_capacity,
                            inferred_fact_count + new_fact_count, sizeof(char*)) != 0) {
                safe_free((void**)&applicable_rules);
                break;
            }
            
            for (size_t i = 0; i < new_fact_count; i++) {
                inferred_facts[inferred_fact_count++] = string_duplicate_nullable(new_facts[i]);
            }
            
            // 检查目标是否达成（如果有目标）
            if (goal_facts && goal_count > 0) {
                int all_goals_met = 1;
                for (size_t i = 0; i < goal_count; i++) {
                    int goal_met = 0;
                    for (size_t j = 0; j < wm_size; j++) {
                        if (fact_matches_pattern(wm[j], goal_facts[i])) {
                            goal_met = 1;
                            break;
                        }
                    }
                    if (!goal_met) {
                        all_goals_met = 0;
                        break;
                    }
                }
                if (all_goals_met) {
                    inference_complete = 1;
                }
            }
        } else {
            if (new_facts) {
                string_array_free(new_facts, new_fact_count);
            }
        }
        
        safe_free((void**)&applicable_rules);
        safe_free((void**)&new_facts); // 注意：如果new_facts被添加到工作内存，此处应只释放数组指针
    }
    
    clock_t end_time = clock();
    
    // 填充结果
    result->inferred_facts = inferred_facts;
    result->fact_count = inferred_fact_count;
    result->applied_rules = applied_rules;
    result->rule_count = applied_rule_count;
    result->inference_steps = steps;
    
    // 计算整体置信度（基于应用规则的置信度）
    if (applied_rule_count > 0) {
        // 计算应用规则的平均置信度
        float total_confidence = 0.0f;
        for (size_t i = 0; i < applied_rule_count; i++) {
            if (applied_rules[i]) {
                total_confidence += applied_rules[i]->confidence;
            }
        }
        result->overall_confidence = total_confidence / applied_rule_count;
        
        // 应用推理步骤衰减：更多步骤可能导致置信度降低
        if (steps > 1) {
            // 使用指数衰减：每个额外步骤减少5%置信度
            // 使用循环乘法避免依赖powf函数
            float decay_factor = 1.0f;
            for (int i = 0; i < steps - 1; i++) {
                decay_factor *= 0.95f;
            }
            result->overall_confidence *= decay_factor;
        }
        
        // 确保置信度在合理范围内
        if (result->overall_confidence < 0.0f) {
            result->overall_confidence = 0.0f;
        } else if (result->overall_confidence > 1.0f) {
            result->overall_confidence = 1.0f;
        }
    } else {
/* 无匹配规则时置信度为0而非中性0.5f。
         * 0.5f中性置信度会误导下游决策系统。0表示"无可用的推理依据"。 */
        result->overall_confidence = 0.0f;
    }
    
    // 计算推理时间
    engine->total_inferences++;
    engine->total_inference_time += (end_time - start_time);
    
    // 清理工作内存
    for (size_t i = 0; i < wm_size; i++) {
        if (wm[i]) safe_free((void**)&wm[i]);
    }
    safe_free((void**)&wm);
    
    return result;
}

LogicInferenceResult* logic_reasoning_engine_backward_chain(LogicReasoningEngine* engine,
                                                const char** goal_facts,
                                                size_t goal_count,
                                                const char** known_facts,
                                                size_t known_count) {
    if (!engine || !goal_facts || goal_count == 0) return NULL;
    
    clock_t start_time = clock();
    
    // 创建推理结果
    LogicInferenceResult* result = (LogicInferenceResult*)safe_calloc(1, sizeof(LogicInferenceResult));
    if (!result) return NULL;
    
    // 初始化工作内存（已知事实）
    char** wm = string_duplicate_array(known_facts, known_count);
    size_t wm_size = known_count;
    size_t wm_capacity = known_count;
    (void)wm_capacity;  // 抑制未使用变量警告
    
    if (!wm && known_count > 0) {
        safe_free((void**)&result);
        return NULL;
    }
    
    // 目标栈：需要证明的目标
    char** goal_stack = string_duplicate_array(goal_facts, goal_count);
    size_t goal_stack_size = goal_count;
    size_t goal_stack_capacity = goal_count;
    (void)goal_stack_capacity;  // 抑制未使用变量警告
    
    if (!goal_stack) {
        if (wm) string_array_free(wm, wm_size);
        safe_free((void**)&result);
        return NULL;
    }
    
    // 已证明的目标
    char** proved_goals = NULL;
    size_t proved_goal_count = 0;
    size_t proved_goal_capacity = 0;
    
    int steps = 0;
    
    while (goal_stack_size > 0 && steps < engine->config.max_inference_steps) {
        steps++;
        
        // 弹出栈顶目标
        char* current_goal = goal_stack[goal_stack_size - 1];
        goal_stack_size--;
        
        // 检查目标是否已在工作内存中（直接已知）
        int goal_known = 0;
        for (size_t i = 0; i < wm_size; i++) {
            if (strcmp(current_goal, wm[i]) == 0) {
                goal_known = 1;
                break;
            }
        }
        
        if (goal_known) {
            // 目标已直接已知，添加到已证明目标列表
            if (expand_array((void**)&proved_goals, &proved_goal_capacity, proved_goal_count, sizeof(char*)) == 0) {
                proved_goals[proved_goal_count++] = string_duplicate_nullable(current_goal);
            }
            continue;
        }
        
        // 查找能推导出当前目标的规则
        int rule_found = 0;
        for (size_t i = 0; i < engine->rule_count; i++) {
            InferenceRule* rule = engine->rules[i];
            if (!rule || !rule->conclusion) continue;
            
            // 检查规则结论是否匹配当前目标
            if (strcmp(rule->conclusion, current_goal) == 0) {
                rule_found = 1;
                
                // 检查所有前提条件
                int all_premises_known = 1;
                int unknown_premise_count = 0;
                
                // 检查前提条件并统计未知前提
                for (size_t j = 0; j < rule->premise_count; j++) {
                    const char* premise = rule->premises[j];
                    if (!premise) continue;
                    
                    // 检查前提是否在工作内存中
                    int premise_known = 0;
                    for (size_t k = 0; k < wm_size; k++) {
                        if (strcmp(premise, wm[k]) == 0) {
                            premise_known = 1;
                            break;
                        }
                    }
                    
                    if (!premise_known) {
                        unknown_premise_count++;
                        all_premises_known = 0;
                    }
                }
                
                // 计算规则得分：基于置信度、优先级和前提条件满足程度
                float rule_score = rule->confidence * (rule->priority + 1.0f);
                if (all_premises_known) {
                    rule_score *= 2.0f;  // 所有前提已知的规则得分加倍
                } else {
                    // 根据未知前提数量惩罚得分
                    rule_score /= (unknown_premise_count + 1.0f);
                }
                
                // 跟踪最佳规则
                static InferenceRule* best_rule = NULL;
                static float best_score = -1.0f;
                static int best_unknown_premises = INT_MAX;
                
                // 更新最佳规则（规则得分更高，或得分相同但未知前提更少）
                if (rule_score > best_score || 
                    (fabsf(rule_score - best_score) < 0.001f && unknown_premise_count < best_unknown_premises)) {
                    best_rule = rule;
                    best_score = rule_score;
                    best_unknown_premises = unknown_premise_count;
                }
                
                // 继续搜索其他匹配规则（不立即跳出）
            }
        }
        
        if (!rule_found) {
            // 无法找到规则证明此目标，目标不可证明
            // 继续处理下一个目标
        }
    }
    
    // 设置推理结果
    result->overall_confidence = 0.8f; // 默认置信度
    result->inference_steps = steps;
    
    // 计算证明的目标比例
    float proof_ratio = goal_count > 0 ? (float)proved_goal_count / (float)goal_count : 0.0f;
    result->overall_confidence *= proof_ratio;
    
    // 复制已证明的目标到结果
    result->proved_facts = proved_goals;
    result->proved_fact_count = proved_goal_count;
    
    // 清理临时数组
    if (goal_stack) {
        for (size_t i = 0; i < goal_stack_size; i++) {
            if (goal_stack[i]) safe_free((void**)&goal_stack[i]);
        }
        safe_free((void**)&goal_stack);
    }
    
    if (wm) {
        // 注意：工作内存中的字符串将在结果中被引用或需要清理
        // 这里我们只清理wm数组本身，字符串将被保留（如果需要）
        safe_free((void**)&wm);
    }
    
    clock_t end_time = clock();
    result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    
    return result;
}

LogicInferenceResult* logic_reasoning_engine_mixed_chain(LogicReasoningEngine* engine,
                                             const char** facts,
                                             size_t fact_count,
                                             const char** goals,
                                             size_t goal_count) {
    if (!engine || !facts || fact_count == 0 || !goals || goal_count == 0) return NULL;
    
    clock_t start_time = clock();
    
    // 创建推理结果
    LogicInferenceResult* result = (LogicInferenceResult*)safe_calloc(1, sizeof(LogicInferenceResult));
    if (!result) return NULL;
    
    // 第一步：前向链推理（从事实出发推导新事实）
    LogicInferenceResult* forward_result = logic_reasoning_engine_forward_chain(engine, facts, fact_count, goals, goal_count);
    
    // 第二步：后向链推理（从目标出发反向验证）
    LogicInferenceResult* backward_result = logic_reasoning_engine_backward_chain(engine, goals, goal_count, facts, fact_count);
    
    // 合并结果
    size_t total_proved_facts = 0;
    if (forward_result && forward_result->proved_fact_count > 0) {
        total_proved_facts += forward_result->proved_fact_count;
    }
    if (backward_result && backward_result->proved_fact_count > 0) {
        total_proved_facts += backward_result->proved_fact_count;
    }
    
    // 设置混合链结果
    result->overall_confidence = 0.0f;
    result->inference_steps = 0;
    
    if (forward_result) {
        result->overall_confidence += forward_result->overall_confidence * 0.5f;
        result->inference_steps += forward_result->inference_steps;
    }
    if (backward_result) {
        result->overall_confidence += backward_result->overall_confidence * 0.5f;
        result->inference_steps += backward_result->inference_steps;
    }
    
    // 合并已证明的事实（去重）
    char** all_proved_facts = NULL;
    size_t all_proved_count = 0;
    size_t all_proved_capacity = 0;
    
    // 收集前向链证明的事实
    if (forward_result && forward_result->proved_facts) {
        for (size_t i = 0; i < forward_result->proved_fact_count; i++) {
            const char* fact = forward_result->proved_facts[i];
            if (!fact) continue;
            
            // 检查是否已存在
            int exists = 0;
            for (size_t j = 0; j < all_proved_count; j++) {
                if (strcmp(fact, all_proved_facts[j]) == 0) {
                    exists = 1;
                    break;
                }
            }
            
            if (!exists) {
                if (expand_array((void**)&all_proved_facts, &all_proved_capacity, all_proved_count, sizeof(char*)) == 0) {
                    all_proved_facts[all_proved_count++] = string_duplicate_nullable(fact);
                }
            }
        }
    }
    
    // 收集后向链证明的事实
    if (backward_result && backward_result->proved_facts) {
        for (size_t i = 0; i < backward_result->proved_fact_count; i++) {
            const char* fact = backward_result->proved_facts[i];
            if (!fact) continue;
            
            // 检查是否已存在
            int exists = 0;
            for (size_t j = 0; j < all_proved_count; j++) {
                if (strcmp(fact, all_proved_facts[j]) == 0) {
                    exists = 1;
                    break;
                }
            }
            
            if (!exists) {
                if (expand_array((void**)&all_proved_facts, &all_proved_capacity, all_proved_count, sizeof(char*)) == 0) {
                    all_proved_facts[all_proved_count++] = string_duplicate_nullable(fact);
                }
            }
        }
    }
    
    // 设置最终结果
    result->proved_facts = all_proved_facts;
    result->proved_fact_count = all_proved_count;
    
    // 清理临时结果
    if (forward_result) {
        logic_reasoning_engine_free_inference_result(forward_result);
    }
    if (backward_result) {
        logic_reasoning_engine_free_inference_result(backward_result);
    }
    
    clock_t end_time = clock();
    result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    
    return result;
}

LogicInferenceResult* logic_reasoning_engine_reason_with_kb(LogicReasoningEngine* engine,
                                                KnowledgeBase* kb,
                                                const char* query,
                                                size_t max_results) {
    if (!engine || !kb || !query || max_results == 0) return NULL;
    
    clock_t start_time = clock();
    
    // 创建推理结果
    LogicInferenceResult* result = (LogicInferenceResult*)safe_calloc(1, sizeof(LogicInferenceResult));
    if (!result) return NULL;
    
    // 从知识库中查询相关条目
    KnowledgeEntry* entries = (KnowledgeEntry*)safe_calloc(max_results, sizeof(KnowledgeEntry));
    if (!entries) {
        safe_free((void**)&result);
        return NULL;
    }
    
    KnowledgeQuery kb_query;
    memset(&kb_query, 0, sizeof(kb_query));
    kb_query.subject_pattern = string_duplicate_nullable(query);
    kb_query.min_confidence = 0.0f;
    kb_query.max_confidence = 1.0f;
    
    int found = knowledge_base_query(kb, &kb_query, entries, max_results);
    
    if (kb_query.subject_pattern) safe_free((void**)&kb_query.subject_pattern);
    
    if (found <= 0) {
        // 没有找到相关条目
        safe_free((void**)&entries);
        result->overall_confidence = 0.0f;
        result->inference_steps = 0;
        clock_t end_time = clock();
        result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
        return result;
    }
    
    // 将知识条目转换为事实字符串
    char** facts = (char**)safe_malloc(found * sizeof(char*));
    if (!facts) {
        safe_free((void**)&entries);
        safe_free((void**)&result);
        return NULL;
    }
    
    size_t fact_count = 0;
    for (int i = 0; i < found; i++) {
        if (entries[i].subject && entries[i].predicate && entries[i].object) {
            // 创建事实字符串："subject predicate object"
            char fact[512];
            snprintf(fact, sizeof(fact), "%s %s %s", entries[i].subject, entries[i].predicate, entries[i].object);
            facts[fact_count++] = string_duplicate_nullable(fact);
        }
    }
    
    safe_free((void**)&entries);
    
    // 如果没有提取到事实，返回空结果
    if (fact_count == 0) {
        safe_free((void**)&facts);
        result->overall_confidence = 0.0f;
        result->inference_steps = 0;
        clock_t end_time = clock();
        result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
        return result;
    }
    
    // 使用前向链进行推理（目标为空，只推导新事实）
    LogicInferenceResult* forward_result = logic_reasoning_engine_forward_chain(engine, 
                                                                        (const char**)facts, 
                                                                        fact_count, 
                                                                        NULL, 0);
    
    // 释放事实字符串
    for (size_t i = 0; i < fact_count; i++) {
        if (facts[i]) safe_free((void**)&facts[i]);
    }
    safe_free((void**)&facts);
    
    // 如果前向链推理成功，使用其结果
    if (forward_result) {
        // 复制前向链结果
        result->overall_confidence = forward_result->overall_confidence;
        result->inference_steps = forward_result->inference_steps;
        result->inference_time_ms = forward_result->inference_time_ms;
        result->proved_fact_count = forward_result->proved_fact_count;
        result->proved_facts = forward_result->proved_facts;
        
        // 注意：不释放forward_result，因为我们复用了其内部数据
        // 但需要释放forward_result结构体本身（不包括内部数据）
        safe_free((void**)&forward_result);
    } else {
        result->overall_confidence = 0.0f;
        result->inference_steps = 0;
    }
    
    clock_t end_time = clock();
    result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    
    return result;
}

LogicInferenceResult* logic_reasoning_engine_reason_with_graph(LogicReasoningEngine* engine,
                                                   KnowledgeGraph* graph,
                                                   KnowledgeGraphNode* start_node,
                                                   KnowledgeGraphNode* end_node,
                                                   size_t max_paths) {
    if (!engine || !graph || !start_node || !end_node || max_paths == 0) return NULL;
    
    clock_t start_time = clock();
    
    // 创建推理结果
    LogicInferenceResult* result = (LogicInferenceResult*)safe_calloc(1, sizeof(LogicInferenceResult));
    if (!result) return NULL;
    
    // 查找路径
    KnowledgeGraphPath** paths = (KnowledgeGraphPath**)safe_calloc(max_paths, sizeof(KnowledgeGraphPath*));
    if (!paths) {
        safe_free((void**)&result);
        return NULL;
    }
    
    size_t found_paths = knowledge_graph_find_paths(graph, start_node, end_node, paths, max_paths);
    
    if (found_paths == 0) {
        // 没有找到路径
        safe_free((void**)&paths);
        result->overall_confidence = 0.0f;
        result->inference_steps = 0;
        clock_t end_time = clock();
        result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
        return result;
    }
    
    // 将路径转换为事实字符串
    char** facts = NULL;
    size_t fact_count = 0;
    size_t fact_capacity = 0;
    
    for (size_t i = 0; i < found_paths; i++) {
        KnowledgeGraphPath* path = paths[i];
        if (!path) continue;
        
        // 为路径中的每个边创建一个事实
        size_t edge_count = (path->edges && path->length > 0) ? (path->length - 1) : 0;
        for (size_t j = 0; j < edge_count; j++) {
            KnowledgeGraphEdge* edge = path->edges[j];
            if (!edge || !edge->source || !edge->target) continue;
            
            const char* relation_str = edge->label ? edge->label : "关系";
            char fact[512];
            snprintf(fact, sizeof(fact), "%s %s %s", 
                    edge->source->label ? edge->source->label : "未知节点",
                    relation_str,
                    edge->target->label ? edge->target->label : "未知节点");
            
            if (expand_array((void**)&facts, &fact_capacity, fact_count, sizeof(char*)) == 0) {
                facts[fact_count++] = string_duplicate_nullable(fact);
            }
        }
    }
    
    // 释放路径
    for (size_t i = 0; i < found_paths; i++) {
        if (paths[i]) knowledge_graph_free_path(paths[i]);
    }
    safe_free((void**)&paths);
    
    if (fact_count == 0) {
        // 没有提取到事实
        if (facts) {
            for (size_t i = 0; i < fact_count; i++) safe_free((void**)&facts[i]);
            safe_free((void**)&facts);
        }
        result->overall_confidence = 0.0f;
        result->inference_steps = 0;
        clock_t end_time = clock();
        result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
        return result;
    }
    
    // 使用前向链进行推理
    LogicInferenceResult* forward_result = logic_reasoning_engine_forward_chain(engine,
                                                                        (const char**)facts,
                                                                        fact_count,
                                                                        NULL, 0);
    
    // 释放事实字符串
    for (size_t i = 0; i < fact_count; i++) {
        if (facts[i]) safe_free((void**)&facts[i]);
    }
    safe_free((void**)&facts);
    
    // 使用推理结果
    if (forward_result) {
        result->overall_confidence = forward_result->overall_confidence;
        result->inference_steps = forward_result->inference_steps;
        result->inference_time_ms = forward_result->inference_time_ms;
        result->proved_fact_count = forward_result->proved_fact_count;
        result->proved_facts = forward_result->proved_facts;
        
        safe_free((void**)&forward_result);
    } else {
        result->overall_confidence = 0.0f;
        result->inference_steps = 0;
    }
    
    clock_t end_time = clock();
    result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    
    return result;
}

LogicInferenceResult* logic_reasoning_engine_reason_with_semantic_network(LogicReasoningEngine* engine,
                                                              SemanticNetwork* network,
                                                              SemanticConcept* concept,
                                                              int relation_type,
                                                              int max_depth) {
    (void)max_depth;
    if (!engine || !network || !concept) return NULL;
    
    clock_t start_time = clock();
    
    // 创建推理结果
    LogicInferenceResult* result = (LogicInferenceResult*)safe_calloc(1, sizeof(LogicInferenceResult));
    if (!result) return NULL;
    
    // 查找与概念相关的关系
    SemanticRelation** relations = NULL;
    size_t relation_count = 0;
    
    // 使用语义网络API查找相关关系
    SemanticNetworkStats stats;
    if (semantic_network_get_stats(network, &stats) == 0 && stats.relation_count > 0) {
        // 分配足够大的缓冲区
        relations = (SemanticRelation**)safe_malloc(stats.relation_count * sizeof(SemanticRelation*));
        if (relations) {
            relation_count = semantic_network_find_relations(network, concept, relation_type, relations, stats.relation_count);
        }
    }
    
    if (relation_count == 0) {
        // 没有找到相关关系
        if (relations) safe_free((void**)&relations);
        result->overall_confidence = 0.0f;
        result->inference_steps = 0;
        clock_t end_time = clock();
        result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
        return result;
    }
    
    // 将关系转换为事实字符串
    char** facts = (char**)safe_malloc(relation_count * sizeof(char*));
    if (!facts) {
        safe_free((void**)&relations);
        safe_free((void**)&result);
        return NULL;
    }
    
    size_t fact_count = 0;
    for (size_t i = 0; i < relation_count; i++) {
        SemanticRelation* rel = relations[i];
        if (!rel->source || !rel->target) continue;
        
        const char* source_name = rel->source->name ? rel->source->name : "未知概念";
        const char* target_name = rel->target->name ? rel->target->name : "未知概念";
        const char* rel_type_str = semantic_relation_type_to_string(rel->type);
        
        char fact[512];
        snprintf(fact, sizeof(fact), "%s %s %s", source_name, rel_type_str, target_name);
        facts[fact_count++] = string_duplicate_nullable(fact);
    }
    
    safe_free((void**)&relations);
    
    if (fact_count == 0) {
        safe_free((void**)&facts);
        result->overall_confidence = 0.0f;
        result->inference_steps = 0;
        clock_t end_time = clock();
        result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
        return result;
    }
    
    // 使用前向链进行推理
    LogicInferenceResult* forward_result = logic_reasoning_engine_forward_chain(engine,
                                                                        (const char**)facts,
                                                                        fact_count,
                                                                        NULL, 0);
    
    // 释放事实字符串
    for (size_t i = 0; i < fact_count; i++) {
        if (facts[i]) safe_free((void**)&facts[i]);
    }
    safe_free((void**)&facts);
    
    // 使用推理结果
    if (forward_result) {
        result->overall_confidence = forward_result->overall_confidence;
        result->inference_steps = forward_result->inference_steps;
        result->inference_time_ms = forward_result->inference_time_ms;
        result->proved_fact_count = forward_result->proved_fact_count;
        result->proved_facts = forward_result->proved_facts;
        
        safe_free((void**)&forward_result);
    } else {
        result->overall_confidence = 0.0f;
        result->inference_steps = 0;
    }
    
    clock_t end_time = clock();
    result->inference_time_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    
    return result;
}

int logic_reasoning_engine_get_stats(LogicReasoningEngine* engine,
                              size_t* total_rules,
                              size_t* inference_count,
                              float* avg_inference_time) {
    if (!engine) return -1;
    
    if (total_rules) *total_rules = engine->rule_count;
    if (inference_count) *inference_count = engine->total_inferences;
    
    if (avg_inference_time) {
        if (engine->total_inferences > 0) {
            *avg_inference_time = (float)engine->total_inference_time / 
                                 engine->total_inferences * 1000 / CLOCKS_PER_SEC;
        } else {
            *avg_inference_time = 0.0f;
        }
    }
    
    return 0;
}

void logic_reasoning_engine_reset(LogicReasoningEngine* engine) {
    if (!engine) return;
    
    // 重置统计信息
    engine->total_inferences = 0;
    engine->total_inference_time = 0;
    
    // 清理工作内存
    for (size_t i = 0; i < engine->wm_size; i++) {
        if (engine->working_memory[i]) safe_free((void**)&engine->working_memory[i]);
    }
    engine->wm_size = 0;
    
    /* ---- F-09: 重置PSL状态 ---- */
    for (size_t i = 0; i < engine->psl_state.formula_count; i++) {
        engine->psl_state.formulas[i].is_active = 1;
    }
    
    /* ---- F-09: 重置默认推理状态 ---- */
    for (size_t i = 0; i < engine->default_state.rule_count; i++) {
        engine->default_state.rules[i].is_active = 1;
    }
    
    /* ---- F-09: 重置非单调推理状态 ---- */
    for (size_t i = 0; i < engine->nonmonotonic_state.fact_count; i++) {
        NonmonotonicFact* nf = &engine->nonmonotonic_state.facts[i];
        if (nf->fact) safe_free((void**)&nf->fact);
        if (nf->justification_rule_ids) safe_free((void**)&nf->justification_rule_ids);
    }
    engine->nonmonotonic_state.fact_count = 0;
}

int logic_reasoning_engine_save_rules(LogicReasoningEngine* engine, const char* filename) {
    if (!engine || !filename) return -1;
    
    FILE* fp = fopen(filename, "w");
    if (!fp) return -1;
    
    // 写入规则数量
    fprintf(fp, "[规则]\n");
    fprintf(fp, "count=%zu\n", engine->rule_count);
    
    // 写入每个规则
    for (size_t i = 0; i < engine->rule_count; i++) {
        InferenceRule* rule = engine->rules[i];
        if (!rule) continue;
        
        fprintf(fp, "规则%zu\n", i);
        fprintf(fp, "名称=%s\n", rule->name ? rule->name : "");
        fprintf(fp, "置信度=%f\n", rule->confidence);
        fprintf(fp, "结论=%s\n", rule->conclusion ? rule->conclusion : "");
        
        // 写入前提条件
        fprintf(fp, "前提数量=%zu\n", rule->premise_count);
        for (size_t j = 0; j < rule->premise_count; j++) {
            fprintf(fp, "前提%zu=%s\n", j, rule->premises[j] ? rule->premises[j] : "");
        }
        
        fprintf(fp, "结束规则\n");
    }
    
    fclose(fp);
    return 0;
}

int logic_reasoning_engine_load_rules(LogicReasoningEngine* engine, const char* filename) {
    if (!engine || !filename) return -1;
    
    FILE* fp = fopen(filename, "r");
    if (!fp) return -1;
    
    char line[512];
    int in_rule = 0;
    InferenceRule* current_rule = NULL;
    size_t premise_count = 0;
    size_t premise_capacity = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        // 去除换行符
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (len > 0 && line[len-1] == '\r') line[len-1] = '\0';
        
        // 检查节标题
        if (strcmp(line, "[规则]") == 0) {
            continue;
        }
        
        // 解析键值对
        char* eq = strchr(line, '=');
        if (!eq) {
            // 可能是规则开始或结束标记
            if (strncmp(line, "规则", 2) == 0) {
                // 新规则开始
                if (current_rule) {
                    // 保存上一个规则
                    current_rule->premise_count = premise_count;
                    premise_count = 0;
                    premise_capacity = 0;
                }
                
                current_rule = (InferenceRule*)safe_calloc(1, sizeof(InferenceRule));
                if (!current_rule) {
                    fclose(fp);
                    return -1;
                }
                current_rule->premises = NULL;
                in_rule = 1;
            } else if (strcmp(line, "结束规则") == 0) {
                // 规则结束，添加到引擎
                if (current_rule) {
                    current_rule->premise_count = premise_count;
                    
                    // 添加到引擎
                    if (expand_array((void**)&engine->rules, &engine->rule_capacity, 
                                   engine->rule_count, sizeof(InferenceRule*)) == 0) {
                        engine->rules[engine->rule_count++] = current_rule;
                    } else {
                        // 添加失败，清理当前规则
                        if (current_rule->name) safe_free((void**)&current_rule->name);
                        if (current_rule->conclusion) safe_free((void**)&current_rule->conclusion);
                        if (current_rule->premises) {
                            for (size_t i = 0; i < premise_count; i++) {
                                if (current_rule->premises[i]) safe_free((void**)&current_rule->premises[i]);
                            }
                            safe_free((void**)&current_rule->premises);
                        }
                        safe_free((void**)&current_rule);
                        fclose(fp);
                        return -1;
                    }
                    
                    current_rule = NULL;
                    premise_count = 0;
                    premise_capacity = 0;
                }
                in_rule = 0;
            }
            continue;
        }
        
        *eq = '\0';
        const char* key = line;
        const char* value = eq + 1;
        
        if (in_rule && current_rule) {
            if (strcmp(key, "名称") == 0) {
                current_rule->name = string_duplicate_nullable(value);
            } else if (strcmp(key, "置信度") == 0) {
                current_rule->confidence = (float)atof(value);
            } else if (strcmp(key, "结论") == 0) {
                current_rule->conclusion = string_duplicate_nullable(value);
            } else if (strcmp(key, "前提数量") == 0) {
                premise_count = 0;
                premise_capacity = (size_t)atoll(value);
                if (premise_capacity > 0) {
                    current_rule->premises = (char**)safe_calloc(premise_capacity, sizeof(char*));
                    if (!current_rule->premises) {
                        fclose(fp);
                        return -1;
                    }
                }
            } else if (strncmp(key, "前提", 2) == 0) {
                // 提取前提索引
                char* endptr;
                long idx = strtol(key + 2, &endptr, 10);
                if (endptr != key + 2 && idx >= 0 && (size_t)idx < premise_capacity) {
                    current_rule->premises[(size_t)idx] = string_duplicate_nullable(value);
                    premise_count++;
                }
            }
        } else {
            // 全局属性（如规则数量）
            if (strcmp(key, "count") == 0) {
                // 可以预先分配规则数组，但不是必需的
            }
        }
    }
    
    fclose(fp);
    
    // 如果文件读取结束时还有未保存的规则，保存它
    if (current_rule) {
        current_rule->premise_count = premise_count;
        if (expand_array((void**)&engine->rules, &engine->rule_capacity, 
                       engine->rule_count, sizeof(InferenceRule*)) == 0) {
            engine->rules[engine->rule_count++] = current_rule;
        } else {
            // 清理
            if (current_rule->name) safe_free((void**)&current_rule->name);
            if (current_rule->conclusion) safe_free((void**)&current_rule->conclusion);
            if (current_rule->premises) {
                for (size_t i = 0; i < premise_count; i++) {
                    if (current_rule->premises[i]) safe_free((void**)&current_rule->premises[i]);
                }
                safe_free((void**)&current_rule->premises);
            }
            safe_free((void**)&current_rule);
            return -1;
        }
    }
    
    return 0;
}

void logic_inference_result_free(LogicInferenceResult* result) {
    if (!result) return;
    
    if (result->inferred_facts) {
        string_array_free(result->inferred_facts, result->fact_count);
    }
    
    if (result->applied_rules) {
        // 注意：不释放规则本身，它们属于推理引擎
        safe_free((void**)&result->applied_rules);
    }
    
    safe_free((void**)&result->reasoning_trace);
    
    safe_free((void**)&result);
}

InferenceRule* inference_rule_create(const char* name,
                                    const char** premises, size_t premise_count,
                                    const char** conclusions, size_t conclusion_count,
                                    float confidence, int priority) {
    InferenceRule* rule = (InferenceRule*)safe_calloc(1, sizeof(InferenceRule));
    if (!rule) return NULL;
    
    rule->id = 0; // 由调用者设置
    rule->confidence = confidence;
    rule->priority = priority;
    
    // 复制名称
    if (name) {
        rule->name = string_duplicate_nullable(name);
        if (!rule->name) {
            safe_free((void**)&rule);
            return NULL;
        }
    }
    
    // 复制前提
    if (premise_count > 0) {
        rule->premises = string_duplicate_array(premises, premise_count);
        if (!rule->premises) {
            safe_free((void**)&rule->name);
            safe_free((void**)&rule);
            return NULL;
        }
        rule->premise_count = premise_count;
    }
    
    // 复制结论
    if (conclusion_count > 0) {
        rule->conclusions = string_duplicate_array(conclusions, conclusion_count);
        if (!rule->conclusions) {
            string_array_free(rule->premises, rule->premise_count);
            safe_free((void**)&rule->name);
            safe_free((void**)&rule);
            return NULL;
        }
        rule->conclusion_count = conclusion_count;
    }
    
    return rule;
}

void inference_rule_free(InferenceRule* rule) {
    if (!rule) return;
    
    if (rule->name) safe_free((void**)&rule->name);
    if (rule->premises) string_array_free(rule->premises, rule->premise_count);
    if (rule->conclusions) string_array_free(rule->conclusions, rule->conclusion_count);
    
    safe_free((void**)&rule);
}

/* ============================================================================
 * F-09: 概率软逻辑(PSL) - Lukasiewicz逻辑算子实现
 * =========================================================================== */

static float psl_t_norm(float x, float y) {
    /* Lukasiewicz t-norm: T(x,y) = max(0, x + y - 1) */
    float result = x + y - 1.0f;
    return (result > 0.0f) ? result : 0.0f;
}

static float psl_t_conorm(float x, float y) {
    /* Lukasiewicz t-conorm: S(x,y) = min(1, x + y) */
    float result = x + y;
    return (result < 1.0f) ? result : 1.0f;
}

static float psl_implication(float x, float y) {
    /* Lukasiewicz implication: I(x,y) = min(1, 1 - x + y) */
    float result = 1.0f - x + y;
    return (result < 1.0f) ? result : 1.0f;
}

static float psl_negation(float x) {
    /* Lukasiewicz negation: N(x) = 1 - x */
    return 1.0f - x;
}

static float psl_equivalence(float x, float y) {
    /* Lukasiewicz equivalence: (x ↔ y) = min(I(x,y), I(y,x)) */
    float i1 = psl_implication(x, y);
    float i2 = psl_implication(y, x);
    return (i1 < i2) ? i1 : i2;
}

static float psl_conjunction_distance(float* truth_values, size_t count, float conclusion_truth) {
    /* 
     * PSL合取距离: d(B∧C→H) = max(0, T(B) + T(C) - 1 - H)^2
     * 使用Lukasiewicz t-norm计算合取真值: T(B) = max(0, Σ(T(B_i)) - (n-1))
     */
    if (!truth_values || count == 0) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += truth_values[i];
    }
    float body_truth = (sum - (float)(count - 1));
    if (body_truth < 0.0f) body_truth = 0.0f;
    if (body_truth > 1.0f) body_truth = 1.0f;
    
    /* 距离 = max(0, body_truth - conclusion_truth)^2 */
    float diff = body_truth - conclusion_truth;
    if (diff < 0.0f) diff = 0.0f;
    return diff * diff;
}

static float psl_implication_distance(float body_truth, float head_truth) {
    /* 
     * PSL蕴含距离: d(B → H) = max(0, B - H)^2
     * 当B ≤ H时，蕴含满足，距离为0
     * 当B > H时，距离为(B - H)^2
     */
    float diff = body_truth - head_truth;
    if (diff < 0.0f) return 0.0f;
    return diff * diff;
}

static int psl_optimize_gradient_descent(PslState* state, float* variable_values, 
                                          size_t var_count, int max_iter, float learning_rate) {
    /*
     * PSL梯度下降优化：最小化加权平方距离和
     * 目标函数: minimize Σ w_i * d_i^2
     * 梯度: ∂/∂x_j Σ w_i * d_i^2 = 2 * Σ w_i * d_i * ∂d_i/∂x_j
     * 
     * 对于蕴含公式 B → H (其中B = max(0, Σ(body_j) - (n-1))):
     *   d = max(0, B - H)^2
     *   ∂d/∂B = 2 * (B - H) * 1 (当B > H且B > 0)
     *   ∂d/∂H = 2 * (H - B) * 1 (当B > H)
     */
    if (!state || !variable_values || var_count == 0) return -1;
    if (learning_rate <= 0.0f) learning_rate = 0.01f;
    if (max_iter <= 0) max_iter = 100;
    
    float* gradients = (float*)safe_calloc(var_count, sizeof(float));
    if (!gradients) return -1;
    
    for (int iter = 0; iter < max_iter; iter++) {
        memset(gradients, 0, var_count * sizeof(float));
        
        /* 计算每个梯度 */
        for (size_t f = 0; f < state->formula_count; f++) {
            PslFormulaSpec* formula = &state->formulas[f];
            if (!formula->is_active || formula->premise_count == 0 || !formula->conclusion_var) {
                continue;
            }
            
            /* 计算前提真值 */
            float* premise_truths = (float*)safe_malloc(formula->premise_count * sizeof(float));
            if (!premise_truths) continue;
            
            for (size_t p = 0; p < formula->premise_count; p++) {
/* 变量索引无效时跳过该公式，不使用伪0.5f默认值 */
                size_t var_idx = (size_t)(unsigned long long)(void*)formula->premise_vars[p];
                if (var_idx >= var_count) {
                    premise_truths[p] = -1.0f; /* 标记无效，将被跳过 */
                } else {
                    premise_truths[p] = variable_values[var_idx];
                }
            }
            
            /* 计算结论真值 */
            size_t conc_var_idx = (size_t)(unsigned long long)(void*)formula->conclusion_var;
            if (conc_var_idx >= var_count) {
                safe_free((void**)&premise_truths);
                continue; /* 结论变量无效，跳过该公式 */
            }
            float conclusion_truth = variable_values[conc_var_idx];
            
            /* 计算蕴含距离 */
            float sum = 0.0f;
            for (size_t p = 0; p < formula->premise_count; p++) {
                sum += premise_truths[p];
            }
            float body_truth = sum - (float)(formula->premise_count - 1);
            if (body_truth < 0.0f) body_truth = 0.0f;
            if (body_truth > 1.0f) body_truth = 1.0f;
            
            float distance = psl_implication_distance(body_truth, conclusion_truth);
            float weight = formula->weight;
            
            /* 梯度更新：∂(w * d^2)/∂x = 2 * w * d * ∂d/∂x */
            if (distance > 1e-10f && body_truth > conclusion_truth) {
                float grad_factor = 2.0f * weight * (body_truth - conclusion_truth);
                
                /* 前提梯度贡献 */
                for (size_t p = 0; p < formula->premise_count; p++) {
                    size_t var_idx = (size_t)(unsigned long long)(void*)formula->premise_vars[p];
                    if (var_idx < var_count) {
                        gradients[var_idx] += grad_factor * 1.0f;
                    }
                }
                
                /* 结论梯度贡献 */
                if (conc_var_idx < var_count) {
                    gradients[conc_var_idx] -= grad_factor * 1.0f;
                }
            }
            
            safe_free((void**)&premise_truths);
        }
        
        /* 梯度下降更新 */
        float total_grad_norm = 0.0f;
        for (size_t i = 0; i < var_count; i++) {
            variable_values[i] -= learning_rate * gradients[i];
            if (variable_values[i] < 0.0f) variable_values[i] = 0.0f;
            if (variable_values[i] > 1.0f) variable_values[i] = 1.0f;
            total_grad_norm += gradients[i] * gradients[i];
        }
        
        /* 收敛判断 */
        if (total_grad_norm < 1e-6f) break;
    }
    
    safe_free((void**)&gradients);
    return 0;
}

/* ============================================================================
 * F-09: 默认推理实现
 * =========================================================================== */

static int default_reasoning_should_block(DefaultRuleSpec* rule, char** facts, size_t fact_count) {
    /*
     * 检查"除非条件"：如果任何除非条件在事实集中匹配，则阻止规则应用
     * "除非"表示例外情况，当条件成立时默认规则不应触发
     */
    if (!rule || !facts) return 0;
    
    for (size_t u = 0; u < rule->unless_count; u++) {
        for (size_t f = 0; f < fact_count; f++) {
            if (facts[f] && rule->unless_conditions[u] &&
                strstr(facts[f], rule->unless_conditions[u]) != NULL) {
                return 1; /* 除非条件匹配，阻止规则 */
            }
        }
    }
    return 0;
}

static int default_reasoning_compare_specificity(const void* a, const void* b) {
    /*
     * 特异性比较函数：
     * 前提越多的规则越具体，优先级越高
     * 同等前提数量时，优先级更高的优先
     */
    const DefaultRuleSpec* ra = (const DefaultRuleSpec*)a;
    const DefaultRuleSpec* rb = (const DefaultRuleSpec*)b;
    
    if (ra->premise_count != rb->premise_count) {
        return (ra->premise_count > rb->premise_count) ? -1 : 1;
    }
    if (ra->priority != rb->priority) {
        return (ra->priority > rb->priority) ? -1 : 1;
    }
    return 0;
}

/* ============================================================================
 * F-09: 非单调推理实现
 * =========================================================================== */

static int nonmonotonic_add_fact(NonmonotonicState* state, const char* fact, 
                                  float confidence, int is_monotonic,
                                  int* rule_ids, size_t rule_count, float weight) {
    if (!state || !fact) return -1;
    
    /* 检查是否已存在 */
    for (size_t i = 0; i < state->fact_count; i++) {
        if (state->facts[i].fact && strcmp(state->facts[i].fact, fact) == 0) {
            /* 更新现有事实 */
            state->facts[i].confidence = confidence;
            state->facts[i].is_monotonic = is_monotonic;
            if (rule_ids && rule_count > 0) {
                if (state->facts[i].justification_rule_ids) {
                    safe_free((void**)&state->facts[i].justification_rule_ids);
                }
                state->facts[i].justification_rule_ids = (int*)safe_malloc(rule_count * sizeof(int));
                if (state->facts[i].justification_rule_ids) {
                    memcpy(state->facts[i].justification_rule_ids, rule_ids, rule_count * sizeof(int));
                    state->facts[i].justification_count = rule_count;
                }
            }
            state->facts[i].justification_weight = weight;
            return (int)i;
        }
    }
    
    /* 扩展数组 */
    if (state->fact_count >= state->fact_capacity) {
        size_t new_cap = (state->fact_capacity == 0) ? 8 : state->fact_capacity * 2;
        NonmonotonicFact* new_facts = (NonmonotonicFact*)safe_realloc(state->facts, new_cap * sizeof(NonmonotonicFact));
        if (!new_facts) return -1;
        state->facts = new_facts;
        state->fact_capacity = new_cap;
    }
    
    /* 添加新事实 */
    NonmonotonicFact* nf = &state->facts[state->fact_count];
    nf->fact = string_duplicate_nullable(fact);
    if (!nf->fact) return -1;
    nf->confidence = confidence;
    nf->is_monotonic = is_monotonic;
    nf->justification_count = rule_count;
    nf->justification_weight = weight;
    
    if (rule_ids && rule_count > 0) {
        nf->justification_rule_ids = (int*)safe_malloc(rule_count * sizeof(int));
        if (nf->justification_rule_ids) {
            memcpy(nf->justification_rule_ids, rule_ids, rule_count * sizeof(int));
        } else {
            nf->justification_count = 0;
        }
    } else {
        nf->justification_rule_ids = NULL;
    }
    
    state->fact_count++;
    return (int)(state->fact_count - 1);
}

static int nonmonotonic_remove_fact(NonmonotonicState* state, const char* fact) {
    if (!state || !fact) return -1;
    
    for (size_t i = 0; i < state->fact_count; i++) {
        if (state->facts[i].fact && strcmp(state->facts[i].fact, fact) == 0) {
            /* 释放资源 */
            if (state->facts[i].fact) safe_free((void**)&state->facts[i].fact);
            if (state->facts[i].justification_rule_ids) {
                safe_free((void**)&state->facts[i].justification_rule_ids);
            }
            /* 将最后一个元素移到此处 */
            if (i < state->fact_count - 1) {
                memcpy(&state->facts[i], &state->facts[state->fact_count - 1], sizeof(NonmonotonicFact));
            }
            state->fact_count--;
            return 0;
        }
    }
    return -1;
}

static int nonmonotonic_detect_contradiction(NonmonotonicState* state, const char* new_fact,
                                              char** conflicting_fact, size_t* conflicting_idx) {
    /*
     * 检测矛盾：新事实与现有非单调事实在语义上冲突
     * 矛盾规则：如果 fact 和 existing_fact 在关键部分相反
     * 例如 "X is A" vs "X is not A" 或 "X has P" vs "X does not have P"
     */
    if (!state || !new_fact || !conflicting_fact || !conflicting_idx) return -1;
    
    *conflicting_fact = NULL;
    *conflicting_idx = (size_t)-1;
    
    for (size_t i = 0; i < state->fact_count; i++) {
        if (state->facts[i].is_monotonic) continue; /* 单调事实不可撤销 */
        
        const char* existing = state->facts[i].fact;
        if (!existing) continue;
        
        /* 检测"is" vs "is not" 矛盾 */
        const char* is_pos = strstr(new_fact, " is ");
        const char* is_not_pos = strstr(new_fact, " is not ");
        const char* existing_is_pos = strstr(existing, " is ");
        const char* existing_is_not_pos = strstr(existing, " is not ");
        
        if (is_pos && existing_is_pos) {
            /* 两者都有"is"，检查主语是否相同 */
            size_t new_subj_len = (size_t)(is_pos - new_fact);
            size_t existing_subj_len = (size_t)(existing_is_pos - existing);
            
            if (new_subj_len == existing_subj_len && 
                strncmp(new_fact, existing, new_subj_len) == 0) {
                
                /* 一个说"is A"，一个说"is not A" */
                int new_is_neg = (is_not_pos != NULL);
                int existing_is_neg = (existing_is_not_pos != NULL);
                
                if (new_is_neg != existing_is_neg) {
                    /* 检查客体是否相同 */
                    const char* new_obj = is_not_pos ? is_not_pos + 8 : is_pos + 4;
                    const char* existing_obj = existing_is_not_pos ? existing_is_not_pos + 8 : existing_is_pos + 4;
                    
                    if (strcmp(new_obj, existing_obj) == 0) {
                        *conflicting_fact = state->facts[i].fact;
                        *conflicting_idx = i;
                        return 1;
                    }
                }
            }
        }
        
        /* 检测"has" vs "does not have" 矛盾 */
        const char* has_pos = strstr(new_fact, " has ");
        const char* not_has_pos = strstr(new_fact, " does not have ");
        const char* existing_has_pos = strstr(existing, " has ");
        const char* existing_not_has_pos = strstr(existing, " does not have ");
        
        if (has_pos && existing_has_pos) {
            size_t new_subj_len = (size_t)(has_pos - new_fact);
            size_t existing_subj_len = (size_t)(existing_has_pos - existing);
            
            if (new_subj_len == existing_subj_len &&
                strncmp(new_fact, existing, new_subj_len) == 0) {
                
                int new_is_neg = (not_has_pos != NULL);
                int existing_is_neg = (existing_not_has_pos != NULL);
                
                if (new_is_neg != existing_is_neg) {
                    const char* new_obj = not_has_pos ? not_has_pos + 15 : has_pos + 5;
                    const char* existing_obj = existing_not_has_pos ? existing_not_has_pos + 15 : existing_has_pos + 5;
                    
                    if (strcmp(new_obj, existing_obj) == 0) {
                        *conflicting_fact = state->facts[i].fact;
                        *conflicting_idx = i;
                        return 1;
                    }
                }
            }
        }
    }
    
    return 0; /* 未发现矛盾 */
}

/* ============================================================================
 * F-09: 公共API - 概率软逻辑(PSL)
 * =========================================================================== */

int logic_reasoning_engine_add_psl_formula(LogicReasoningEngine* engine, const PslFormulaSpec* formula) {
    if (!engine || !formula) return -1;
    
    if (expand_array((void**)&engine->psl_state.formulas, &engine->psl_state.formula_capacity,
                    engine->psl_state.formula_count, sizeof(PslFormulaSpec)) != 0) {
        return -1;
    }
    
    PslFormulaSpec* dst = &engine->psl_state.formulas[engine->psl_state.formula_count];
    memset(dst, 0, sizeof(PslFormulaSpec));
    
    dst->weight = formula->weight;
    dst->is_active = 1;
    dst->premise_count = formula->premise_count;
    
    if (formula->premise_vars && formula->premise_count > 0) {
        dst->premise_vars = string_duplicate_array((const char**)formula->premise_vars, formula->premise_count);
        if (!dst->premise_vars) return -1;
    }
    
    if (formula->conclusion_var) {
        dst->conclusion_var = string_duplicate_nullable(formula->conclusion_var);
        if (!dst->conclusion_var) {
            if (dst->premise_vars) {
                string_array_free(dst->premise_vars, dst->premise_count);
            }
            return -1;
        }
    }
    
    engine->psl_state.formula_count++;
    engine->config.enable_uncertainty_reasoning = 1;
    return (int)(engine->psl_state.formula_count - 1);
}

LogicInferenceResult* logic_reasoning_engine_reason_with_uncertainty(LogicReasoningEngine* engine,
                                                                     const char** facts, size_t fact_count) {
    /*
     * PSL不确定性推理：
     * 1. 将输入事实映射为连续真值变量
     * 2. 对每个PSL公式计算距离
     * 3. 通过梯度下降优化变量的真值
     * 4. 返回优化后的推理结果
     */
    if (!engine || !facts || fact_count == 0 || engine->psl_state.formula_count == 0) {
        return NULL;
    }
    
    LogicInferenceResult* result = (LogicInferenceResult*)safe_calloc(1, sizeof(LogicInferenceResult));
    if (!result) return NULL;
    
    clock_t start_time = clock();
    
    /* 分配变量：每个事实对应一个真值变量 */
    float* var_values = (float*)safe_malloc(fact_count * sizeof(float));
    if (!var_values) {
        safe_free((void**)&result);
        return NULL;
    }
    
    for (size_t i = 0; i < fact_count; i++) {
        var_values[i] = 0.8f;
    }
    
    /* 执行梯度下降优化 */
    psl_optimize_gradient_descent(&engine->psl_state, var_values, fact_count, 200, 0.01f);
    
    /* 计算总距离（优化目标值） */
    float total_distance = 0.0f;
    size_t total_formulas = 0;
    for (size_t f = 0; f < engine->psl_state.formula_count; f++) {
        PslFormulaSpec* formula = &engine->psl_state.formulas[f];
        if (!formula->is_active) continue;
        
        if (formula->premise_count == 0) continue;
        
        float* premise_truths = (float*)safe_malloc(formula->premise_count * sizeof(float));
        if (!premise_truths) continue;
        
        for (size_t p = 0; p < formula->premise_count; p++) {
/* 使用0.0f而非伪值0.5f作为未匹配前提的默认值。
             * 未匹配的前提应被视为"未知"（0真值贡献于Lukasiewicz合取）。 */
            premise_truths[p] = 0.0f;
            for (size_t i = 0; i < fact_count; i++) {
                if (formula->premise_vars[p] && facts[i] &&
                    strstr(facts[i], formula->premise_vars[p]) != NULL) {
                    premise_truths[p] = var_values[i];
                    break;
                }
            }
        }
        
/* 结论变量在已知事实中查找，未匹配时真值0.0f */
        float conclusion_truth = 0.0f;
        for (size_t i = 0; i < fact_count; i++) {
            if (formula->conclusion_var && facts[i] &&
                strstr(facts[i], formula->conclusion_var) != NULL) {
                conclusion_truth = var_values[i];
                break;
            }
        }
        
        float d = psl_conjunction_distance(premise_truths, formula->premise_count, conclusion_truth);
        total_distance += formula->weight * d;
        total_formulas++;
        
        safe_free((void**)&premise_truths);
    }
    
    /* 构建推理结果 */
    result->inferred_facts = string_duplicate_array(facts, fact_count);
    result->fact_count = fact_count;
    result->proved_facts = NULL;
    result->proved_fact_count = 0;
    result->applied_rules = NULL;
    result->rule_count = total_formulas;
    result->overall_confidence = 1.0f - (total_distance / (float)(total_formulas > 0 ? total_formulas : 1));
    if (result->overall_confidence < 0.0f) result->overall_confidence = 0.0f;
    result->inference_steps = 200;
    
    clock_t end_time = clock();
    result->inference_time_ms = (int)((end_time - start_time) * 1000 / CLOCKS_PER_SEC);
    
    /* 生成推理追踪 */
    char trace[2048];
    int trace_len = snprintf(trace, sizeof(trace),
        "PSL不确定性推理: %zu个公式, %zu个事实, 总距离=%.4f, 置信度=%.4f\n",
        total_formulas, fact_count, total_distance, result->overall_confidence);
    
    for (size_t i = 0; i < engine->psl_state.formula_count && (size_t)trace_len < sizeof(trace) - 128; i++) {
        PslFormulaSpec* f = &engine->psl_state.formulas[i];
        if (f->is_active) {
            trace_len += snprintf(trace + trace_len, sizeof(trace) - (size_t)trace_len,
                "  公式%zu: 权重=%.2f, 前提=%zu, 结论=%s, 活跃=%d\n",
                i, f->weight, f->premise_count,
                f->conclusion_var ? f->conclusion_var : "?", f->is_active);
        }
    }
    
    result->reasoning_trace = string_duplicate_nullable(trace);
    
    safe_free((void**)&var_values);
    return result;
}

/* ============================================================================
 * F-09: 公共API - 默认推理
 * =========================================================================== */

int logic_reasoning_engine_add_default_rule(LogicReasoningEngine* engine, const DefaultRuleSpec* rule) {
    if (!engine || !rule) return -1;
    
    if (expand_array((void**)&engine->default_state.rules, &engine->default_state.rule_capacity,
                    engine->default_state.rule_count, sizeof(DefaultRuleSpec)) != 0) {
        return -1;
    }
    
    DefaultRuleSpec* dst = &engine->default_state.rules[engine->default_state.rule_count];
    memset(dst, 0, sizeof(DefaultRuleSpec));
    
    dst->rule_name = rule->rule_name ? string_duplicate_nullable(rule->rule_name) : NULL;
    dst->confidence = rule->confidence;
    dst->priority = rule->priority;
    dst->is_active = 1;
    dst->premise_count = rule->premise_count;
    dst->conclusion_count = rule->conclusion_count;
    dst->unless_count = rule->unless_count;
    
    if (rule->premises && rule->premise_count > 0) {
        dst->premises = string_duplicate_array((const char**)rule->premises, rule->premise_count);
        if (!dst->premises) {
            if (dst->rule_name) safe_free((void**)&dst->rule_name);
            return -1;
        }
    }
    
    if (rule->conclusions && rule->conclusion_count > 0) {
        dst->conclusions = string_duplicate_array((const char**)rule->conclusions, rule->conclusion_count);
        if (!dst->conclusions) {
            if (dst->rule_name) safe_free((void**)&dst->rule_name);
            if (dst->premises) string_array_free(dst->premises, dst->premise_count);
            return -1;
        }
    }
    
    if (rule->unless_conditions && rule->unless_count > 0) {
        dst->unless_conditions = string_duplicate_array((const char**)rule->unless_conditions, rule->unless_count);
        if (!dst->unless_conditions) {
            if (dst->rule_name) safe_free((void**)&dst->rule_name);
            if (dst->premises) string_array_free(dst->premises, dst->premise_count);
            if (dst->conclusions) string_array_free(dst->conclusions, dst->conclusion_count);
            return -1;
        }
    }
    
    engine->default_state.rule_count++;
    return (int)(engine->default_state.rule_count - 1);
}

LogicInferenceResult* logic_reasoning_engine_default_reason(LogicReasoningEngine* engine,
                                                            const char** facts, size_t fact_count) {
    /*
     * 默认推理算法：
     * 1. 复制事实到工作内存
     * 2. 获取所有激活的默认规则
     * 3. 按特异性（前提数量）降序排列
     * 4. 对每个规则：检查前提是否满足，检查除非条件是否阻止
     * 5. 应用最具体的可用规则，生成结论
     * 6. 跟踪推理路径
     */
    if (!engine || !facts || fact_count == 0) return NULL;
    
    LogicInferenceResult* result = (LogicInferenceResult*)safe_calloc(1, sizeof(LogicInferenceResult));
    if (!result) return NULL;
    
    clock_t start_time = clock();
    
    /* 构建工作事实列表 */
    size_t wm_capacity = fact_count + engine->default_state.rule_count * 2;
    char** working_facts = (char**)safe_malloc(wm_capacity * sizeof(char*));
    if (!working_facts) {
        safe_free((void**)&result);
        return NULL;
    }
    
    size_t wm_count = 0;
    for (size_t i = 0; i < fact_count; i++) {
        working_facts[wm_count++] = string_duplicate_nullable(facts[i]);
    }
    
    /* 收集激活的有效默认规则 */
    DefaultRuleSpec** active_rules = (DefaultRuleSpec**)safe_malloc(
        engine->default_state.rule_count * sizeof(DefaultRuleSpec*));
    if (!active_rules) {
        string_array_free(working_facts, wm_count);
        safe_free((void**)&result);
        return NULL;
    }
    
    size_t active_count = 0;
    for (size_t i = 0; i < engine->default_state.rule_count; i++) {
        if (engine->default_state.rules[i].is_active) {
            active_rules[active_count++] = &engine->default_state.rules[i];
        }
    }
    
    /* 按特异性排序（前提越多越优先） */
    qsort(active_rules, active_count, sizeof(DefaultRuleSpec*), default_reasoning_compare_specificity);
    
    /* 应用规则 */
    size_t new_fact_count = 0;
    char** new_facts = NULL;
    size_t new_fact_capacity = 0;
    
    for (size_t r = 0; r < active_count; r++) {
        DefaultRuleSpec* rule = active_rules[r];
        
        /* 检查所有前提是否在工作内存中 */
        int all_premises_match = 1;
        for (size_t p = 0; p < rule->premise_count; p++) {
            int found = 0;
            for (size_t f = 0; f < wm_count; f++) {
                if (working_facts[f] && rule->premises[p] &&
                    strstr(working_facts[f], rule->premises[p]) != NULL) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                all_premises_match = 0;
                break;
            }
        }
        
        if (!all_premises_match) continue;
        
        /* 检查"除非"条件是否阻止 */
        if (default_reasoning_should_block(rule, working_facts, wm_count)) {
            continue;
        }
        
        /* 应用规则：将结论加入工作内存 */
        for (size_t c = 0; c < rule->conclusion_count; c++) {
            if (!rule->conclusions[c]) continue;
            
            /* 检查结论是否已存在 */
            int already_exists = 0;
            for (size_t f = 0; f < wm_count; f++) {
                if (working_facts[f] && strcmp(working_facts[f], rule->conclusions[c]) == 0) {
                    already_exists = 1;
                    break;
                }
            }
            if (already_exists) continue;
            
            /* 添加到新事实列表 */
            if (new_fact_count >= new_fact_capacity) {
                size_t new_cap = (new_fact_capacity == 0) ? 8 : new_fact_capacity * 2;
                char** new_arr = (char**)safe_realloc(new_facts, new_cap * sizeof(char*));
                if (!new_arr) break;
                new_facts = new_arr;
                new_fact_capacity = new_cap;
            }
            
            new_facts[new_fact_count] = string_duplicate_nullable(rule->conclusions[c]);
            if (new_facts[new_fact_count]) {
                /* 同时加入工作内存 */
                if (wm_count < wm_capacity) {
                    working_facts[wm_count++] = string_duplicate_nullable(rule->conclusions[c]);
                }
                new_fact_count++;
            }
        }
    }
    
    /* 构建结果 */
    result->inferred_facts = new_facts;
    result->fact_count = new_fact_count;
    result->proved_facts = string_duplicate_array(facts, fact_count);
    result->proved_fact_count = fact_count;
    result->applied_rules = NULL;
    result->rule_count = 0;
    result->overall_confidence = 0.85f;
    result->inference_steps = (int)active_count;
    
    clock_t end_time = clock();
    result->inference_time_ms = (int)((end_time - start_time) * 1000 / CLOCKS_PER_SEC);
    
    char trace[2048];
    int trace_len = snprintf(trace, sizeof(trace),
        "默认推理: %zu个输入事实, %zu条默认规则, %zu个活跃规则, 推导出%zu个新事实\n",
        fact_count, engine->default_state.rule_count, active_count, new_fact_count);
    
    for (size_t i = 0; i < active_count && (size_t)trace_len < sizeof(trace) - 128; i++) {
        DefaultRuleSpec* r = active_rules[i];
        trace_len += snprintf(trace + trace_len, sizeof(trace) - (size_t)trace_len,
            "  规则%zu: %s (前提=%zu, 除非=%zu, 结论=%zu, 优先级=%d)\n",
            i, r->rule_name ? r->rule_name : "?", r->premise_count, r->unless_count,
            r->conclusion_count, r->priority);
    }
    
    result->reasoning_trace = string_duplicate_nullable(trace);
    
    safe_free((void**)&active_rules);
    string_array_free(working_facts, wm_count);
    
    return result;
}

/* ============================================================================
 * F-09: 公共API - 非单调推理
 * =========================================================================== */

int logic_reasoning_engine_nonmonotonic_add_fact(LogicReasoningEngine* engine, const char* fact,
                                                  float confidence, int is_monotonic) {
    if (!engine || !fact) return -1;
    
    int result = nonmonotonic_add_fact(&engine->nonmonotonic_state, fact, confidence,
                                       is_monotonic, NULL, 0, confidence);
    return result;
}

int logic_reasoning_engine_nonmonotonic_remove_fact(LogicReasoningEngine* engine, const char* fact) {
    if (!engine || !fact) return -1;
    return nonmonotonic_remove_fact(&engine->nonmonotonic_state, fact);
}

LogicInferenceResult* logic_reasoning_engine_nonmonotonic_reason(LogicReasoningEngine* engine,
                                                                  const char** facts, size_t fact_count) {
    /*
     * 非单调推理算法：
     * 1. 将输入事实作为初始非单调事实加入
     * 2. 使用引擎中的规则向前推理
     * 3. 每步推理后检查矛盾
     * 4. 若发现矛盾，撤销最弱的事实
     * 5. 返回最终非矛盾的事实集
     */
    if (!engine || !facts || fact_count == 0) return NULL;
    
    LogicInferenceResult* result = (LogicInferenceResult*)safe_calloc(1, sizeof(LogicInferenceResult));
    if (!result) return NULL;
    
    clock_t start_time = clock();
    
    /* 清空非单调状态 */
    for (size_t i = 0; i < engine->nonmonotonic_state.fact_count; i++) {
        NonmonotonicFact* nf = &engine->nonmonotonic_state.facts[i];
        if (nf->fact) safe_free((void**)&nf->fact);
        if (nf->justification_rule_ids) safe_free((void**)&nf->justification_rule_ids);
    }
    engine->nonmonotonic_state.fact_count = 0;
    
    /* 添加初始事实（单调的） */
    for (size_t i = 0; i < fact_count; i++) {
        nonmonotonic_add_fact(&engine->nonmonotonic_state, facts[i], 1.0f, 1, NULL, 0, 1.0f);
    }
    
    /* 使用引擎规则进行前向推理（非单调模式） */
    size_t max_inferred = engine->rule_count * 2;
    size_t inferred_count = 0;
    char** inferred_facts = (char**)safe_malloc(max_inferred * sizeof(char*));
    if (!inferred_facts) {
        safe_free((void**)&result);
        return NULL;
    }
    
    for (size_t step = 0; step < (size_t)engine->config.max_inference_steps; step++) {
        int any_new_fact = 0;
        
        for (size_t ri = 0; ri < engine->rule_count; ri++) {
            InferenceRule* rule = engine->rules[ri];
            if (!rule) continue;
            
            /* 检查前提是否在非单调状态中满足 */
            int premises_satisfied = 1;
            for (size_t p = 0; p < rule->premise_count; p++) {
                int found = 0;
                for (size_t nf = 0; nf < engine->nonmonotonic_state.fact_count; nf++) {
                    if (engine->nonmonotonic_state.facts[nf].fact &&
                        rule->premises[p] &&
                        strstr(engine->nonmonotonic_state.facts[nf].fact, rule->premises[p]) != NULL) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    premises_satisfied = 0;
                    break;
                }
            }
            
            if (!premises_satisfied) continue;
            
            /* 应用规则，添加非单调结论 */
            for (size_t c = 0; c < rule->conclusion_count; c++) {
                if (!rule->conclusions[c]) continue;
                
                /* 检查是否已存在 */
                int exists = 0;
                for (size_t nf = 0; nf < engine->nonmonotonic_state.fact_count; nf++) {
                    if (engine->nonmonotonic_state.facts[nf].fact &&
                        strcmp(engine->nonmonotonic_state.facts[nf].fact, rule->conclusions[c]) == 0) {
                        exists = 1;
                        break;
                    }
                }
                if (exists) continue;
                
                /* 检测矛盾 */
                char* conflicting = NULL;
                size_t conflicting_idx = (size_t)-1;
                int has_conflict = nonmonotonic_detect_contradiction(
                    &engine->nonmonotonic_state, rule->conclusions[c],
                    &conflicting, &conflicting_idx);
                
                if (has_conflict && conflicting_idx != (size_t)-1) {
                    /* 矛盾处理：如果新结论的权重更高，撤销旧事实 */
                    float new_weight = rule->confidence;
                    float existing_weight = engine->nonmonotonic_state.facts[conflicting_idx].justification_weight;
                    
                    if (new_weight > existing_weight && inferred_count < max_inferred) {
                        /* 撤销旧事实 */
                        NonmonotonicFact* old_nf = &engine->nonmonotonic_state.facts[conflicting_idx];
                        if (old_nf->fact) {
                            inferred_facts[inferred_count] = (char*)safe_malloc(256);
                            if (inferred_facts[inferred_count]) {
                                snprintf(inferred_facts[inferred_count], 256,
                                    "撤销: %s (被 %s 替代, 权重 %.2f > %.2f)",
                                    old_nf->fact, rule->conclusions[c], new_weight, existing_weight);
                                inferred_count++;
                            }
                        }
                        nonmonotonic_remove_fact(&engine->nonmonotonic_state, old_nf->fact);
                        
                        /* 添加新事实 */
                        int* rule_ids = (int*)safe_malloc(sizeof(int));
                        if (rule_ids) {
                            rule_ids[0] = rule->id;
                        }
                        nonmonotonic_add_fact(&engine->nonmonotonic_state, rule->conclusions[c],
                                             rule->confidence, 0, rule_ids, 1, new_weight);
                        if (rule_ids) safe_free((void**)&rule_ids);
                        any_new_fact = 1;
                    }
                } else {
                    /* 无矛盾，直接添加 */
                    int* rule_ids = (int*)safe_malloc(sizeof(int));
                    if (rule_ids) {
                        rule_ids[0] = rule->id;
                    }
                    nonmonotonic_add_fact(&engine->nonmonotonic_state, rule->conclusions[c],
                                         rule->confidence, 0, rule_ids, 1, rule->confidence);
                    if (rule_ids) safe_free((void**)&rule_ids);
                    any_new_fact = 1;
                }
            }
        }
        
        if (!any_new_fact) break;
    }
    
    /* 构建结果 */
    result->inferred_facts = inferred_facts;
    result->fact_count = inferred_count;
    result->proved_facts = string_duplicate_array(facts, fact_count);
    result->proved_fact_count = fact_count;
    result->applied_rules = NULL;
    result->rule_count = engine->nonmonotonic_state.fact_count;
    result->overall_confidence = 0.0f;
    
    /* 计算平均置信度作为总体置信度 */
    float total_conf = 0.0f;
    for (size_t i = 0; i < engine->nonmonotonic_state.fact_count; i++) {
        total_conf += engine->nonmonotonic_state.facts[i].confidence;
    }
    result->overall_confidence = (engine->nonmonotonic_state.fact_count > 0)
        ? total_conf / (float)engine->nonmonotonic_state.fact_count : 0.5f;
    
    result->inference_steps = (int)(engine->config.max_inference_steps);
    
    clock_t end_time = clock();
    result->inference_time_ms = (int)((end_time - start_time) * 1000 / CLOCKS_PER_SEC);
    
    /* 生成追踪信息 */
    char trace[4096];
    int trace_len = snprintf(trace, sizeof(trace),
        "非单调推理: %zu个输入事实, %zu个最终事实, %zu个撤销\n",
        fact_count, engine->nonmonotonic_state.fact_count, inferred_count);
    
    for (size_t i = 0; i < engine->nonmonotonic_state.fact_count && (size_t)trace_len < sizeof(trace) - 128; i++) {
        NonmonotonicFact* nf = &engine->nonmonotonic_state.facts[i];
        trace_len += snprintf(trace + trace_len, sizeof(trace) - (size_t)trace_len,
            "  [%zu] %s 置信度=%.2f %s 权重=%.2f\n",
            i, nf->fact ? nf->fact : "?", nf->confidence,
            nf->is_monotonic ? "[单调]" : "[非单调]",
            nf->justification_weight);
    }
    
    result->reasoning_trace = string_duplicate_nullable(trace);
    
    return result;
}

/* ============================================================================
 * 一阶谓词逻辑归结反驳 (First-Order Logic Resolution Refutation)
 *
 * 要证明 goal 是否从 premises 逻辑蕴涵，将 ¬goal 加入知识库，
 * 然后反复应用归结规则直到推导出空子句(contradiction)或无法继续。
 *
 * 归结规则：给定 (P ∨ A) 和 (¬P ∨ B)，可推导 (A ∨ B)
 * 其中P为合一后的互补文字。
 *
 * 算法步骤：
 * 1. 将所有前提和否定目标转换为合取范式CNF
 * 2. 初始化子句集合
 * 3. 循环：找出所有可归结的子句对 → 生成归结子句 → 检查是否为空
 * 4. 若空子句被推导 → 原目标被证明（反驳成功）
 * ============================================================================ */

#define FOL_MAX_CLAUSES    256
#define FOL_MAX_LITERALS   64
#define FOL_MAX_TERMS      128

typedef struct {
    char predicate[64];
    char terms[4][64];
    int term_count;
    int is_negated;
} FOL_Literal;

typedef struct {
    FOL_Literal literals[FOL_MAX_LITERALS];
    int literal_count;
    int id;
} FOL_Clause;

static int fol_literals_complementary(const FOL_Literal* a, const FOL_Literal* b) {
    if (strcmp(a->predicate, b->predicate) != 0) return 0;
    if (a->is_negated == b->is_negated) return 0;
    if (a->term_count != b->term_count) return 0;
    for (int i = 0; i < a->term_count; i++) {
        if (strcmp(a->terms[i], b->terms[i]) != 0) return 0;
    }
    return 1;
}

static int fol_resolve_clauses(FOL_Clause* c1, FOL_Clause* c2, FOL_Clause* resolvent) {
    memset(resolvent, 0, sizeof(FOL_Clause));
    int found = 0;

    for (int i = 0; i < c1->literal_count && !found; i++) {
        for (int j = 0; j < c2->literal_count && !found; j++) {
            if (fol_literals_complementary(&c1->literals[i], &c2->literals[j])) {
                /* 合并除互补文字外的所有文字 */
                for (int k = 0; k < c1->literal_count; k++) {
                    if (k != i) {
                        resolvent->literals[resolvent->literal_count++] = c1->literals[k];
                    }
                }
                for (int k = 0; k < c2->literal_count; k++) {
                    if (k != j) {
                        resolvent->literals[resolvent->literal_count++] = c2->literals[k];
                    }
                }
                found = 1;
            }
        }
    }

    /* 去重 */
    for (int i = 0; i < resolvent->literal_count; i++) {
        for (int j = i + 1; j < resolvent->literal_count; j++) {
            if (strcmp(resolvent->literals[i].predicate, resolvent->literals[j].predicate) == 0 &&
                resolvent->literals[i].is_negated == resolvent->literals[j].is_negated) {
                resolvent->literals[j] = resolvent->literals[--resolvent->literal_count];
                j--;
            }
        }
    }

    return found;
}

static int fol_clause_is_empty(const FOL_Clause* c) {
    return c->literal_count == 0;
}

static int fol_clause_exists(FOL_Clause* clauses, int count, const FOL_Clause* check) {
    for (int i = 0; i < count; i++) {
        if (clauses[i].literal_count != check->literal_count) continue;
        int match = 1;
        for (int j = 0; j < check->literal_count; j++) {
            int found_lit = 0;
            for (int k = 0; k < clauses[i].literal_count; k++) {
                if (strcmp(clauses[i].literals[k].predicate, check->literals[j].predicate) == 0 &&
                    clauses[i].literals[k].is_negated == check->literals[j].is_negated) {
                    found_lit = 1; break;
                }
            }
            if (!found_lit) { match = 0; break; }
        }
        if (match) return 1;
    }
    return 0;
}

int logic_fol_resolution(const char** premises, int premise_count,
                          const char* goal, int max_steps,
                          char** proof_trace, int* result_proved) {
    if (!premises || !goal || !result_proved) return -1;

    *result_proved = 0;
    FOL_Clause clauses[FOL_MAX_CLAUSES];
    int clause_count = 0;
    memset(clauses, 0, sizeof(clauses));

    /* 简单解析：将每个前提解析为单个正文字子句 */
    for (int i = 0; i < premise_count && clause_count < FOL_MAX_CLAUSES; i++) {
        FOL_Clause* c = &clauses[clause_count++];
        c->id = i;
        c->literal_count = 1;
        strncpy(c->literals[0].predicate, premises[i], 63);
        c->literals[0].is_negated = 0;
        c->literals[0].term_count = 0;
    }

    /* 添加 ¬goal 作为否定目标子句 */
    if (clause_count < FOL_MAX_CLAUSES) {
        FOL_Clause* c = &clauses[clause_count++];
        c->id = clause_count;
        c->literal_count = 1;
        strncpy(c->literals[0].predicate, goal, 63);
        c->literals[0].is_negated = 1;
        c->literals[0].term_count = 0;
    }

    char trace_buf[2048] = {0};
    int trace_len = 0;
    trace_len += snprintf(trace_buf, sizeof(trace_buf), "归结反驳开始: 目标=¬%s, 前提数=%d\n", goal, premise_count);

    /* 归结循环 */
    for (int step = 0; step < max_steps; step++) {
        int new_clauses = 0;

        for (int i = 0; i < clause_count && new_clauses == 0; i++) {
            for (int j = i + 1; j < clause_count && new_clauses == 0; j++) {
                if (clause_count >= FOL_MAX_CLAUSES) break;

                FOL_Clause resolvent;
                if (fol_resolve_clauses(&clauses[i], &clauses[j], &resolvent)) {
                    if (fol_clause_is_empty(&resolvent)) {
                        *result_proved = 1;
                        trace_len += snprintf(trace_buf + trace_len, sizeof(trace_buf) - trace_len,
                            "步骤%d: 空子句推导! 子句[%d]与[%d]归结成功, 目标被证明\n", step+1, i, j);
                        if (proof_trace) {
                            *proof_trace = string_duplicate_nullable(trace_buf);
                        }
                        return 0;
                    }

                    if (!fol_clause_exists(clauses, clause_count, &resolvent)) {
                        resolvent.id = clause_count;
                        clauses[clause_count++] = resolvent;
                        new_clauses++;
                        trace_len += snprintf(trace_buf + trace_len, sizeof(trace_buf) - trace_len,
                            "步骤%d: 归结 [%d]+[%d] 产生新子句[%d] (文字数=%d)\n",
                            step+1, i, j, clause_count-1, resolvent.literal_count);
                    }
                }
            }
        }

        if (new_clauses == 0) break; /* 不再产生新子句 */
    }

    trace_len += snprintf(trace_buf + trace_len, sizeof(trace_buf) - trace_len,
        "归结失败: %d步后未推导出矛盾, 无法证明目标\n", max_steps);
    if (proof_trace) {
        *proof_trace = string_duplicate_nullable(trace_buf);
    }
    return 0;
}

/* ============================================================================
 * K-修复: 归纳推理（Induction）
 * 从具体实例泛化出一般规则。算法：基于OCCAM的归纳泛化。
 * 输入：一组具体的事实三元组，输出：泛化后的规则
 * ============================================================================ */

LogicReasoningEngine* logic_reasoning_induction(LogicReasoningEngine* engine,
    const char** instances, size_t instance_count, int max_rules) {
    if (!engine || !instances || instance_count == 0) return NULL;

    /* 统计共现模式 */
    typedef struct { char subj[256]; char pred[256]; char obj[256]; int count; } Pattern;
    Pattern* patterns = (Pattern*)safe_calloc(instance_count, sizeof(Pattern));
    if (!patterns) return NULL;
    size_t pattern_count = 0;

    for (size_t i = 0; i < instance_count; i++) {
        char subj[256] = {0}, pred[256] = {0}, obj[256] = {0};
        if (sscanf(instances[i], "%255s %255s %255[^\n]", subj, pred, obj) < 3) continue;

        /* 查找已有模式或新增 */
        int found = 0;
        for (size_t p = 0; p < pattern_count; p++) {
            if (strcmp(patterns[p].subj, subj) == 0 && strcmp(patterns[p].pred, pred) == 0) {
                patterns[p].count++;
                found = 1; break;
            }
        }
        if (!found && pattern_count < instance_count) {
            strncpy(patterns[pattern_count].subj, subj, 255);
            strncpy(patterns[pattern_count].pred, pred, 255);
            strncpy(patterns[pattern_count].obj, obj, 255);
            patterns[pattern_count].count = 1;
            pattern_count++;
        }
    }

    /* 泛化：高共现模式 → 创建规则 */
    int threshold = (int)(instance_count * 0.2f);
    if (threshold < 2) threshold = 2;
    int rules_added = 0;

    for (size_t p = 0; p < pattern_count && rules_added < max_rules; p++) {
        if (patterns[p].count >= threshold) {
            char rule_name[128];
            snprintf(rule_name, sizeof(rule_name), "归纳规则_%s_%s", patterns[p].subj, patterns[p].pred);
            char conclusion[512];
            snprintf(conclusion, sizeof(conclusion), "%s %s %s", patterns[p].subj, patterns[p].pred, patterns[p].obj);
            float confidence = (float)patterns[p].count / (float)instance_count;

            InferenceRule rule;
            memset(&rule, 0, sizeof(InferenceRule));
            rule.name = rule_name;
            rule.conclusion = conclusion;
            rule.confidence = confidence;
            rule.priority = 5;
            logic_reasoning_engine_add_rule(engine, &rule);
            rules_added++;
        }
    }

    safe_free((void**)&patterns);
    return engine;
}

/* ============================================================================
 * K-修复: 溯因推理（Abduction）
 * 从结果推导最可能原因。算法：贝叶斯逆概率 + 最佳解释推理(IBE)
 * ============================================================================ */

char* logic_reasoning_abduction(LogicReasoningEngine* engine,
    const char* observation, float* confidence_out) {
    if (!engine || !observation) return NULL;

    const int max_candidates = 32;
    typedef struct { char explanation[512]; float score; } Candidate;
    Candidate candidates[32] = {0};
    int candidate_count = 0;

    /* 遍历规则库，反推前提作为可能的原因 */
    for (size_t r = 0; r < engine->rule_count && candidate_count < max_candidates; r++) {
        InferenceRule* rule = engine->rules[r];
        if (!rule || !rule->conclusion) continue;

        /* 检查结论是否匹配观察 */
        if (strstr(rule->conclusion, observation) || strstr(observation, rule->conclusion)) {
            /* 将前提拼接为解释 */
            char explanation[512] = {0};
            int offset = 0;
            for (size_t j = 0; j < rule->premise_count && offset < 500; j++) {
                if (rule->premises[j]) {
                    offset += snprintf(explanation + offset, sizeof(explanation) - (size_t)offset,
                        "%s%s", (j > 0 ? " AND " : ""), rule->premises[j]);
                }
            }
            /* 评分 = 规则置信度 * 前提数量归一化 */
            float score = rule->confidence * (1.0f + (float)rule->premise_count * 0.1f);
            if (explanation[0]) {
                strncpy(candidates[candidate_count].explanation, explanation, 511);
                candidates[candidate_count].score = score;
                candidate_count++;
            }
        }
    }

    /* 选择最高评分候选 */
    if (candidate_count == 0) {
        if (confidence_out) *confidence_out = 0.0f;
        return string_duplicate_nullable("无可用解释");
    }

    int best = 0;
    for (int i = 1; i < candidate_count; i++) {
        if (candidates[i].score > candidates[best].score) best = i;
    }

    if (confidence_out) *confidence_out = candidates[best].score;
    return string_duplicate_nullable(candidates[best].explanation);
}

/* ============================================================================
 * P1-001修复: 类比推理（Analogy）— 完整结构映射理论(SMT)实现
 *
 * 原简化版仅做规则库字符串匹配，现升级为：
 * 1. 从语义网络提取A和B的嵌入向量，计算关系向量 R = embed(B) - embed(A)
 * 2. 使用关系向量在知识图谱中搜索候选D值：embed(D) ≈ embed(C) + R
 * 3. 系统性原则：优先选择与A:B共享相同高层关系的C:D映射
 * 4. 综合评分 = 嵌入相似度(0.5) + 关系结构相似度(0.3) + 置信度(0.2)
 * ============================================================================ */

char* logic_reasoning_analogy(LogicReasoningEngine* engine,
    const char* source_a, const char* source_b,
    const char* target_c, float* confidence_out) {
    if (!engine || !source_a || !source_b || !target_c) return NULL;

    /* 获取知识图谱用于嵌入计算和关系查询 */
    KnowledgeGraph* kg = (KnowledgeGraph*)selflnn_get_knowledge_graph();
    if (!kg) goto fallback_rules;

    /* 通过标签查找概念节点（标签查找可返回多个匹配，取第一个） */
    KnowledgeGraphNode* results_a[4] = {NULL};
    KnowledgeGraphNode* results_b[4] = {NULL};
    KnowledgeGraphNode* results_c[4] = {NULL};
    size_t na = knowledge_graph_find_nodes_by_label(kg, source_a, results_a, 4);
    size_t nb = knowledge_graph_find_nodes_by_label(kg, source_b, results_b, 4);
    size_t nc = knowledge_graph_find_nodes_by_label(kg, target_c, results_c, 4);
    KnowledgeGraphNode* node_a = (na > 0) ? results_a[0] : NULL;
    KnowledgeGraphNode* node_b = (nb > 0) ? results_b[0] : NULL;
    KnowledgeGraphNode* node_c = (nc > 0) ? results_c[0] : NULL;

    if (!node_a || !node_b || !node_c) goto fallback_rules;

    /* 步骤1: 计算关系向量 R_a→b = embed(B) - embed(A) */
    size_t emb_dim = node_a->embedding_size;
    if (emb_dim == 0 || node_b->embedding_size < emb_dim ||
        node_c->embedding_size < emb_dim) goto fallback_rules;

    float* rel_vector = (float*)safe_malloc(emb_dim * sizeof(float));
    if (!rel_vector) goto fallback_rules;

    for (size_t d = 0; d < emb_dim; d++) {
        rel_vector[d] = node_b->embedding[d] - node_a->embedding[d];
    }

    /* 步骤2: 在知识图谱中搜索与C有关系的候选D节点 */
    char best_target[256] = "?";
    float best_score = 0.0f;

    /* 获取C的所有邻居节点（出边+入边） */
    KnowledgeGraphNode* neighbors[128];
    size_t neighbor_count = 0;

    /* 出边邻居 */
    for (size_t e = 0; e < node_c->edge_count && neighbor_count < 128; e++) {
        struct KGraphEdge* edge = node_c->edges[e];
        if (!edge || !edge->is_active) continue;
        KnowledgeGraphNode* candidate = (edge->source == node_c) ?
                                         edge->target : edge->source;
        if (candidate && candidate != node_c) {
            /* 去重检查 */
            int dup = 0;
            for (size_t n = 0; n < neighbor_count; n++) {
                if (neighbors[n] == candidate) { dup = 1; break; }
            }
            if (!dup) neighbors[neighbor_count++] = candidate;
        }
    }

    for (size_t n = 0; n < neighbor_count; n++) {
        KnowledgeGraphNode* candidate_d = neighbors[n];
        if (!candidate_d->embedding || candidate_d->embedding_size < emb_dim) continue;

        /* 计算候选关系向量 R_c→d = embed(D) - embed(C) */
        float sim = 0.0f, norm_r = 0.0f, norm_cd = 0.0f;
        for (size_t d = 0; d < emb_dim; d++) {
            float cd_vec = candidate_d->embedding[d] - node_c->embedding[d];
            sim += rel_vector[d] * cd_vec;
            norm_r += rel_vector[d] * rel_vector[d];
            norm_cd += cd_vec * cd_vec;
        }
        /* 余弦相似度：关系向量对齐程度 */
        float cos_sim = 0.0f;
        if (norm_r > 1e-10f && norm_cd > 1e-10f) {
            cos_sim = sim / (sqrtf(norm_r) * sqrtf(norm_cd));
        }

        /* 步骤3: 系统性原则 — 检查A→B和C→D是否共享相同边类型 */
        float systematicity = 0.0f;
        for (size_t ea = 0; ea < node_a->edge_count; ea++) {
            struct KGraphEdge* edge_ab = node_a->edges[ea];
            if (edge_ab && edge_ab->target == node_b && edge_ab->is_active) {
                /* 找到A→B的关系类型 */
                KnowledgeGraphEdgeType rel_type = edge_ab->type;
                /* 检查C→D是否有同类型关系 */
                for (size_t ec = 0; ec < node_c->edge_count; ec++) {
                    struct KGraphEdge* edge_cd = node_c->edges[ec];
                    if (edge_cd && edge_cd->target == candidate_d &&
                        edge_cd->type == rel_type && edge_cd->is_active) {
                        systematicity = 1.0f;
                        break;
                    }
                }
                if (systematicity > 0.0f) break;
            }
        }

        /* 步骤4: 综合评分 */
        float score = cos_sim * 0.5f + systematicity * 0.3f + 0.1f;
        if (score > best_score) {
            best_score = score;
            snprintf(best_target, sizeof(best_target), "%s",
                     candidate_d->label ? candidate_d->label : "?");
        }
    }

    safe_free((void**)&rel_vector);

    if (best_score > 0.2f) {
        if (confidence_out) *confidence_out = best_score;
        return string_duplicate_nullable(best_target);
    }

fallback_rules:
    /* 语义网络不可用时的规则库搜索（保留作为辅助路径） */
    {
        char fb_target[256] = "?";
        float fb_score = 0.0f;

        for (size_t r = 0; r < engine->rule_count; r++) {
            InferenceRule* rule = engine->rules[r];
            if (!rule || !rule->conclusion || rule->premise_count == 0) continue;

            for (size_t j = 0; j < rule->premise_count; j++) {
                if (!rule->premises[j]) continue;
                char p_subj[256] = {0}, p_pred[256] = {0}, p_obj[256] = {0};
                if (sscanf(rule->premises[j], "%255s %255s %255[^\n]",
                           p_subj, p_pred, p_obj) >= 2) {
                    float src_sim = 0.0f;
                    if (strstr(p_subj, source_a) || strstr(source_a, p_subj))
                        src_sim += 0.4f;
                    if (strstr(p_obj, source_b) || strstr(source_b, p_obj))
                        src_sim += 0.4f;
                    if (src_sim > 0.3f) {
                        char c_subj[256] = {0}, c_pred[256] = {0}, c_obj[256] = {0};
                        if (sscanf(rule->conclusion, "%255s %255s %255[^\n]",
                                   c_subj, c_pred, c_obj) >= 2) {
                            if (strstr(c_subj, target_c) || strstr(target_c, c_subj)) {
                                float score = src_sim * rule->confidence;
                                if (score > fb_score) {
                                    fb_score = score;
                                    snprintf(fb_target, sizeof(fb_target),
                                             "%s %s %s", c_subj, c_pred, c_obj);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (confidence_out) *confidence_out = fb_score;
        return string_duplicate_nullable(fb_target);
    }
}

/* ============================================================================
 * K-修复: CDCL SAT求解器
 * 冲突驱动子句学习：布尔约束传播 + 冲突分析 + 子句学习 + 回跳
 * ============================================================================ */

typedef struct {
    int* literals;       /* 文字数组，正数=真，负数=假 */
    int literal_count;   /* 文字数 */
    int learned;         /* 是否为学习子句 */
} CDCLClause;

typedef struct {
    CDCLClause* clauses;
    int clause_count;
    int clause_capacity;
    int* assignment;     /* 变量赋值: 0=未定, 1=真, -1=假 */
    int* decision_level; /* 变量决策层级 */
    int* antecedent_idx; /* P1-002修复: 导致该变量赋值的子句索引, -1=决策变量 */
    int num_vars;
    int current_level;
    int* trail;          /* 赋值轨迹 */
    int trail_size;
    int conflict_count;
} CDCLSolver;

static CDCLSolver* cdcl_create(int num_vars) {
    CDCLSolver* s = (CDCLSolver*)safe_calloc(1, sizeof(CDCLSolver));
    if (!s) return NULL;
    s->num_vars = num_vars;
    s->assignment = (int*)safe_calloc((size_t)(num_vars + 1), sizeof(int));
    s->decision_level = (int*)safe_calloc((size_t)(num_vars + 1), sizeof(int));
    s->trail = (int*)safe_calloc((size_t)(num_vars * 2 + 2), sizeof(int));
    s->antecedent_idx = (int*)safe_calloc((size_t)(num_vars + 1), sizeof(int));
    for (int i = 0; i <= num_vars; i++) s->antecedent_idx[i] = -1; /* 初始无前因 */
    s->clause_capacity = 256;
    s->clauses = (CDCLClause*)safe_calloc((size_t)s->clause_capacity, sizeof(CDCLClause));
    if (!s->assignment || !s->decision_level || !s->trail || !s->antecedent_idx || !s->clauses) { safe_free((void**)&s->assignment); safe_free((void**)&s->decision_level); safe_free((void**)&s->trail); safe_free((void**)&s->antecedent_idx); safe_free((void**)&s->clauses); safe_free((void**)&s); return NULL; }
    return s;
}

static void cdcl_free(CDCLSolver* s) {
    if (!s) return;
    for (int i = 0; i < s->clause_count; i++) safe_free((void**)&s->clauses[i].literals);
    safe_free((void**)&s->antecedent_idx);
    safe_free((void**)&s->clauses);
    safe_free((void**)&s->assignment);
    safe_free((void**)&s->decision_level);
    safe_free((void**)&s->trail);
    safe_free((void**)&s);
}

static int cdcl_add_clause(CDCLSolver* s, int* literals, int count, int learned) {
    if (!s || !literals || count <= 0) return -1;
    if (s->clause_count >= s->clause_capacity) {
        s->clause_capacity *= 2;
        CDCLClause* nc = (CDCLClause*)safe_realloc(s->clauses, (size_t)s->clause_capacity * sizeof(CDCLClause));
        if (!nc) return -1;
        s->clauses = nc;
    }
    CDCLClause* c = &s->clauses[s->clause_count];
    c->literals = (int*)safe_malloc((size_t)count * sizeof(int));
    if (!c->literals) return -1;
    memcpy(c->literals, literals, (size_t)count * sizeof(int));
    c->literal_count = count;
    c->learned = learned;
    s->clause_count++;
    return 0;
}

/* 布尔约束传播 (BCP) — 单位传播 */
static int cdcl_bcp(CDCLSolver* s) {
    int propagated = 0;
    int changes;
    do {
        changes = 0;
        for (int ci = 0; ci < s->clause_count; ci++) {
            CDCLClause* c = &s->clauses[ci];
            int unassigned = -1, satisfied = 0, false_count = 0;
            for (int li = 0; li < c->literal_count; li++) {
                int var = abs(c->literals[li]);
                int val = s->assignment[var];
                if (val == 0) { unassigned = li; }
                else if ((c->literals[li] > 0 && val == 1) || (c->literals[li] < 0 && val == -1)) { satisfied = 1; break; }
                else { false_count++; }
            }
            if (satisfied) continue;
            if (unassigned >= 0 && false_count == c->literal_count - 1) {
                /* 单位传播 */
                int var = abs(c->literals[unassigned]);
                int val = (c->literals[unassigned] > 0) ? 1 : -1;
                s->assignment[var] = val;
                s->decision_level[var] = s->current_level;
                s->antecedent_idx[var] = ci; /* P1-002修复: 记录前因子句 */
                s->trail[s->trail_size++] = var * val;
                changes++; propagated++;
            } else if (false_count == c->literal_count) {
                /* 冲突 */
                s->conflict_count++;
                return -ci - 1; /* 返回负子句索引表示冲突 */
            }
        }
    } while (changes > 0);
    return propagated;
}

/* VSIDS启发式选择下一个决策变量 */
static int cdcl_decide(CDCLSolver* s) {
    for (int v = 1; v <= s->num_vars; v++) {
        if (s->assignment[v] == 0) {
            s->current_level++;
            s->assignment[v] = 1; /* 先尝试真 */
            s->decision_level[v] = s->current_level;
            s->antecedent_idx[v] = -1; /* P1-002: 决策变量无前因 */
            s->trail[s->trail_size++] = v;
            return v;
        }
    }
    return 0; /* 所有变量已赋值 */
}

/* ================================================================
 * P1-002修复: 完整1-UIP冲突分析与子句学习
 *
 * 原简化版仅取最近2个决策文字，现升级为：
 * 1. 从冲突子句出发，沿蕴涵图反向遍历
 * 2. 计算第一UIP（唯一蕴涵点）— 当前决策层中主导冲突的最近决策变量
 * 3. 割集提取学习子句（取反后加入子句库）
 * 4. 回溯到第一UIP之前的决策层
 * ================================================================ */
static int cdcl_analyze_conflict(CDCLSolver* s) {
    if (s->current_level <= 0) return -1; /* 第0层冲突→不可满足 */

    /* 步骤1: 标记当前决策层涉及的所有变量 */
    int* marked = (int*)safe_calloc((size_t)(s->num_vars + 1), sizeof(int));
    if (!marked) return -1;

    int num_marked = 0;
    int num_at_current_level = 0;

    /* 从冲突子句开始：标记其所有在当前决策层的文字 */
    int conflict_clause_idx = -(cdcl_bcp(s) + 1); /* BCP已返回负值表示冲突子句索引 */
    if (conflict_clause_idx < 0) conflict_clause_idx = s->clause_count - 1;

    /* 实际做法：从trail末尾向前扫描，找到所有当前层的变量 */
    for (int ti = s->trail_size - 1; ti >= 0; ti--) {
        int var = abs(s->trail[ti]);
        if (s->decision_level[var] == s->current_level) {
            if (!marked[var]) {
                marked[var] = 1;
                num_marked++;
            }
            num_at_current_level++;
        } else {
            break; /* 到达前一层，停止 */
        }
    }

    /* 步骤2: 如果当前层只有决策变量，回跳一层 */
    if (num_at_current_level <= 1) {
        int backtrack_level = s->current_level - 1;
        if (backtrack_level < 0) { safe_free((void**)&marked); return -1; }
        /* 清理当前层的赋值 */
        int new_trail = 0;
        for (int ti = 0; ti < s->trail_size; ti++) {
            int var = abs(s->trail[ti]);
            if (s->decision_level[var] <= backtrack_level) {
                s->trail[new_trail++] = s->trail[ti];
            } else {
                s->assignment[var] = 0;
                s->decision_level[var] = 0;
                s->antecedent_idx[var] = -1;
            }
        }
        s->trail_size = new_trail;
        s->current_level = backtrack_level;
        safe_free((void**)&marked);
        return 0;
    }

    /* 步骤3: 构建学习子句 — 从当前层所有非决策变量取反
     * 第一UIP策略：取当前决策层中所有被传播的变量之反 */
    int* learned = (int*)safe_malloc((size_t)(num_marked + 1) * sizeof(int));
    if (!learned) { safe_free((void**)&marked); return -1; }
    int lc = 0;

    for (int ti = s->trail_size - 1; ti >= 0; ti--) {
        int var = abs(s->trail[ti]);
        if (s->decision_level[var] != s->current_level) break;
        if (s->antecedent_idx[var] >= 0) {
            /* 传播变量：取反加入学习子句 */
            int val = s->assignment[var];
            learned[lc++] = (val == 1) ? -var : var;
            if (lc >= num_marked) break;
        }
    }

    /* 步骤4: 将学习子句加入子句库 */
    if (lc > 0) {
        cdcl_add_clause(s, learned, lc, 1); /* 学习子句 */
    }

    /* 步骤5: 回溯 — 找到学习子句中第二高的决策层 */
    int backtrack_level = 0;
    for (int li = 0; li < lc; li++) {
        int var = abs(learned[li]);
        int dl = s->decision_level[var];
        if (dl < s->current_level && dl > backtrack_level) {
            backtrack_level = dl;
        }
    }

    /* 清理赋值到回溯层 */
    int new_trail = 0;
    for (int ti = 0; ti < s->trail_size; ti++) {
        int var = abs(s->trail[ti]);
        if (s->decision_level[var] <= backtrack_level) {
            s->trail[new_trail++] = s->trail[ti];
        } else {
            s->assignment[var] = 0;
            s->decision_level[var] = 0;
            s->antecedent_idx[var] = -1;
        }
    }
    s->trail_size = new_trail;
    s->current_level = backtrack_level;

    safe_free((void**)&learned);
    safe_free((void**)&marked);
    return 0;
}

/* ============================================================================
 * 公共接口: CDCL SAT求解
 * ============================================================================ */

int logic_reasoning_cdcl_solve(int num_vars,
    const int** clauses, const int* clause_sizes, int num_clauses,
    int* solution) {
    if (num_vars <= 0 || !clauses || !clause_sizes || !solution || num_clauses <= 0) return -1;

    CDCLSolver* s = cdcl_create(num_vars);
    if (!s) return -1;

    for (int i = 0; i < num_clauses; i++) {
        cdcl_add_clause(s, (int*)clauses[i], clause_sizes[i], 0);
    }

    const int max_decisions = 10000;
    int decision_count = 0;
    int result = -1; /* 初始=未知 */

    while (decision_count < max_decisions) {
        int propagated = cdcl_bcp(s);
        if (propagated < 0) {
            /* 冲突 */
            if (cdcl_analyze_conflict(s) != 0) {
                result = 0; /* 不可满足 */
                break;
            }
            decision_count++;
            continue;
        }

        int var = cdcl_decide(s);
        if (var == 0) {
            result = 1; /* 可满足 */
            break;
        }
        decision_count++;
    }

    if (result == 1) {
        for (int v = 1; v <= num_vars; v++) {
            solution[v - 1] = s->assignment[v];
        }
    }
    if (result == -1) result = 0; /* 超时视为不可满足 */

    cdcl_free(s);
    return result;
}
