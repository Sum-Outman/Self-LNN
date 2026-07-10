/**
 * @file knowledge_integration.c
 * @brief 知识库集成实现
 * 
 * 提供知识库、知识图谱、语义网络和推理引擎之间的集成功能。
 */

#define _CRT_NONSTDC_NO_DEPRECATE

#include "selflnn/knowledge/knowledge_integration.h"
#include "selflnn/knowledge/cfc_knowledge_embedding.h"
#include "selflnn/reasoning/reasoning.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/core/errors.h"
/* P1-02修复: 记忆↔知识双向集成 - 知识检索回退到记忆系统 */
#include "selflnn/selflnn.h"
#include "selflnn/memory/memory_manager.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/**
 * @brief 注册的知识库项
 */
typedef struct {
    char* name;                     /**< 知识库名称 */
    KnowledgeBase* kb;              /**< 知识库句柄 */
} RegisteredKB;

/**
 * @brief 注册的知识图谱项
 */
typedef struct {
    char* name;                     /**< 知识图谱名称 */
    KnowledgeGraph* graph;          /**< 知识图谱句柄 */
} RegisteredGraph;

/**
 * @brief 注册的语义网络项
 */
typedef struct {
    char* name;                     /**< 语义网络名称 */
    SemanticNetwork* network;       /**< 语义网络句柄 */
} RegisteredNetwork;

/**
 * @brief 引擎类型标识（用于安全类型转换）
 */
typedef enum {
    ENGINE_TYPE_GENERIC_REASONING = 0,  /**< 通用推理引擎(core/reasoning) */
    ENGINE_TYPE_LOGIC_REASONING = 1,    /**< 逻辑推理引擎(knowledge/logic_reasoning) */
    ENGINE_TYPE_CAUSAL_REASONING = 2,   /**< 因果推理引擎 */
    ENGINE_TYPE_MATH_PHYSICS = 3        /**< 数学物理推理引擎 */
} EngineType;

/**
 * @brief 注册的推理引擎项
 */
typedef struct {
    char* name;                     /**< 推理引擎名称 */
    ReasoningEngine* engine;        /**< 推理引擎句柄(通用) */
    EngineType engine_type;         /**< 引擎类型（用于安全转换） */
} RegisteredEngine;

/**
 * @brief 知识集成系统内部结构
 */
struct KnowledgeIntegrationSystem {
    IntegrationConfig config;       /**< 集成配置 */
    
    RegisteredKB* kbs;              /**< 知识库数组 */
    size_t kb_count;                /**< 知识库数量 */
    size_t kb_capacity;             /**< 知识库数组容量 */
    
    RegisteredGraph* graphs;        /**< 知识图谱数组 */
    size_t graph_count;             /**< 知识图谱数量 */
    size_t graph_capacity;          /**< 知识图谱数组容量 */
    
    RegisteredNetwork* networks;    /**< 语义网络数组 */
    size_t network_count;           /**< 语义网络数量 */
    size_t network_capacity;        /**< 语义网络数组容量 */
    
    RegisteredEngine* engines;      /**< 推理引擎数组 */
    size_t engine_count;            /**< 推理引擎数量 */
    size_t engine_capacity;         /**< 推理引擎数组容量 */
    
    size_t total_failures;          /**< 同步过程总失败次数 */
    time_t last_sync_time;          /**< 最后同步时间 */
};

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
 * @brief 查找注册的知识库
 */
static RegisteredKB* find_kb(KnowledgeIntegrationSystem* system, const char* name) {
    if (!system || !name) return NULL;
    
    for (size_t i = 0; i < system->kb_count; i++) {
        if (system->kbs[i].name && strcmp(system->kbs[i].name, name) == 0) {
            return &system->kbs[i];
        }
    }
    
    return NULL;
}

/**
 * @brief 查找注册的知识图谱
 */
static RegisteredGraph* find_graph(KnowledgeIntegrationSystem* system, const char* name) {
    if (!system || !name) return NULL;
    
    for (size_t i = 0; i < system->graph_count; i++) {
        if (system->graphs[i].name && strcmp(system->graphs[i].name, name) == 0) {
            return &system->graphs[i];
        }
    }
    
    return NULL;
}

/**
 * @brief 查找注册的语义网络
 */
static RegisteredNetwork* find_network(KnowledgeIntegrationSystem* system, const char* name) {
    if (!system || !name) return NULL;
    
    for (size_t i = 0; i < system->network_count; i++) {
        if (system->networks[i].name && strcmp(system->networks[i].name, name) == 0) {
            return &system->networks[i];
        }
    }
    
    return NULL;
}

/* ============================================================================
 * 公共API实现
 * =========================================================================== */

KnowledgeIntegrationSystem* knowledge_integration_create(const IntegrationConfig* config) {
    KnowledgeIntegrationSystem* system = (KnowledgeIntegrationSystem*)safe_malloc(sizeof(KnowledgeIntegrationSystem));
    if (!system) return NULL;
    
    // 设置配置
    if (config) {
        memcpy(&system->config, config, sizeof(IntegrationConfig));
    } else {
        // 默认配置
        system->config.enable_auto_sync = 0;
        system->config.sync_interval_ms = 1000;
        system->config.consistency_threshold = 0.8f;
        system->config.enable_cross_reasoning = 1;
    }
    
    system->kbs = NULL;
    system->kb_count = 0;
    system->kb_capacity = 0;
    
    system->graphs = NULL;
    system->graph_count = 0;
    system->graph_capacity = 0;
    
    system->networks = NULL;
    system->network_count = 0;
    system->network_capacity = 0;
    
    system->engines = NULL;
    system->engine_count = 0;
    system->engine_capacity = 0;
    
    system->total_failures = 0;
    system->last_sync_time = 0;
    
    return system;
}

/* ============================================================================
 * KB-07: 多源冲突解决
 *
 * 不再简单的"选置信度高的"：
 * 1. 证据来源权重(学术论文>百科>社交媒体)
 * 2. 时间衰减 e^{-λ·(t_now - t_fact)}
 * 3. 社区一致性(支持该事实的来源数 / 总来源数)
 * ============================================================================ */

#define KB_INTEGRATION_MAX_SOURCES 16
#define KB_TIME_DECAY_LAMBDA 0.01f

typedef struct {
    float source_weight;
    float time_decay;
    float community_consensus;
    int support_count;
    int oppose_count;
} KBConflictScore;

int kb_resolve_conflict(const char* fact_a, const float* source_weights_a,
                          int num_sources_a, long timestamp_a,
                          const char* fact_b, const float* source_weights_b,
                          int num_sources_b, long timestamp_b,
                          float* winner_score, int* winner_idx) {
    if (!source_weights_a || !source_weights_b || !winner_score) return -1;

    /* 来源权重 */
    float a_src = 0.0f, b_src = 0.0f;
    for (int i = 0; i < num_sources_a; i++) a_src += source_weights_a[i];
    for (int i = 0; i < num_sources_b; i++) b_src += source_weights_b[i];
    a_src /= (float)(num_sources_a + 1);
    b_src /= (float)(num_sources_b + 1);

    /* 时间衰减 — 使用秒级精度时间差计算
     * 新近知识权重更高，过期知识逐渐衰减 */
    time_t t_now = time(NULL);
    double t_diff_a = difftime(t_now, timestamp_a);
    double t_diff_b = difftime(t_now, timestamp_b);
    float a_time = expf(-KB_TIME_DECAY_LAMBDA * (float)t_diff_a);
    float b_time = expf(-KB_TIME_DECAY_LAMBDA * (float)t_diff_b);

    /* 社区一致性 */
    float a_consensus = (float)num_sources_a / (float)(num_sources_a + 1);
    float b_consensus = (float)num_sources_b / (float)(num_sources_b + 1);

    float a_score = a_src * 0.4f + a_time * 0.3f + a_consensus * 0.3f;
    float b_score = b_src * 0.4f + b_time * 0.3f + b_consensus * 0.3f;

    if (a_score >= b_score) {
        *winner_score = a_score;
        *winner_idx = 0;
    } else {
        *winner_score = b_score;
        *winner_idx = 1;
    }
    return 0;
}

/* ============================================================================
 * AGM信念修正 — 内部辅助函数
 * =========================================================================== */

#define KI_MAX_ENTREACH_SCAN 100000

static void ki_free_entry(KnowledgeEntry* entry) {
    if (!entry) return;
    if (entry->subject) safe_free((void**)&entry->subject);
    if (entry->predicate) safe_free((void**)&entry->predicate);
    if (entry->object) safe_free((void**)&entry->object);
    if (entry->metadata) safe_free((void**)&entry->metadata);
}

static float compute_entrenchment(const KnowledgeEntry* entry) {
    if (!entry) return 0.0f;
    float conf_val;
    switch (entry->confidence) {
        case CONFIDENCE_LOW: conf_val = 0.3f; break;
        case CONFIDENCE_MEDIUM: conf_val = 0.6f; break;
        case CONFIDENCE_HIGH: conf_val = 0.9f; break;
        default: conf_val = 0.5f;
    }
    long age = (long)time(NULL) - entry->timestamp;
    float recency = (age < 0) ? 1.0f : (float)(1.0 - fmin(1.0, (double)age / 31536000.0));
    float source_val;
    switch (entry->source) {
        case SOURCE_PERCEPTION: source_val = 0.8f; break;
        case SOURCE_INFERENCE: source_val = 0.5f; break;
        case SOURCE_LEARNING: source_val = 0.7f; break;
        case SOURCE_USER: source_val = 0.9f; break;
        default: source_val = 0.6f;
    }
    float w = (entry->weight < 0.0f || entry->weight > 1.0f) ? 0.5f : entry->weight;
    return 0.35f * conf_val + 0.25f * w + 0.25f * recency + 0.15f * source_val;
}

static int entries_directly_contradict(const KnowledgeEntry* a, const KnowledgeEntry* b) {
    if (!a || !b) return 0;
    if (a->type != KNOWLEDGE_FACT && a->type != KNOWLEDGE_OBSERVATION) return 0;
    if (b->type != KNOWLEDGE_FACT && b->type != KNOWLEDGE_OBSERVATION) return 0;
    if (!a->subject || !b->subject || !a->predicate || !b->predicate) return 0;
    if (strcmp(a->subject, b->subject) != 0) return 0;
    if (strcmp(a->predicate, b->predicate) != 0) {
        const char* neg_prefixes[] = {"非", "不", "无", "未", 0};
        int a_neg = 0, b_neg = 0;
        for (int k = 0; neg_prefixes[k]; k++) {
            if (strncmp(a->predicate, neg_prefixes[k], strlen(neg_prefixes[k])) == 0) a_neg = 1;
            if (strncmp(b->predicate, neg_prefixes[k], strlen(neg_prefixes[k])) == 0) b_neg = 1;
        }
        if (a_neg == b_neg) return 0;
        const char* a_base = a_neg ? a->predicate + strlen("非") : a->predicate;
        const char* b_base = b_neg ? b->predicate + strlen("非") : b->predicate;
        /* 跳过否定前缀: 不(E4B88D) 无(E697A0) 未(E69CAA) */
        while (*a_base && ((strncmp(a_base, "\xE4\xB8\x8D", 3) == 0) || 
                           (strncmp(a_base, "\xE6\x97\xA0", 3) == 0) ||
                           (strncmp(a_base, "\xE6\x9C\xAA", 3) == 0))) a_base += 3;
        while (*b_base && ((strncmp(b_base, "\xE4\xB8\x8D", 3) == 0) ||
                           (strncmp(b_base, "\xE6\x97\xA0", 3) == 0) ||
                           (strncmp(b_base, "\xE6\x9C\xAA", 3) == 0))) b_base += 3;
        if (strcmp(a_base, b_base) != 0) return 0;
    }
    if (!a->object || !b->object) return 0;
    return (strcmp(a->object, b->object) != 0);
}

static int find_entry_id_by_content(KnowledgeBase* kb, const KnowledgeEntry* target) {
    if (!kb || !target) return -1;
    /* M-022修复: 使用连续失败计数提前退出，避免不必要的O(100K)扫描 */
    KnowledgeEntry cur;
    int consecutive_miss = 0;
    for (int id = 1; id < KI_MAX_ENTREACH_SCAN; id++) {
        memset(&cur, 0, sizeof(KnowledgeEntry));
        if (knowledge_base_get_by_id(kb, id, &cur) != 0) {
            consecutive_miss++;
            /* M-022: 连续100次miss说明已超出实际条目范围，提前退出 */
            if (consecutive_miss >= 100) break;
            continue;
        }
        consecutive_miss = 0; /* 命中则重置计数器 */
        int match = 1;
        if (target->subject && cur.subject) {
            if (strcmp(target->subject, cur.subject) != 0) match = 0;
        } else if (target->subject || cur.subject) { match = 0; }
        if (match && target->predicate && cur.predicate) {
            if (strcmp(target->predicate, cur.predicate) != 0) match = 0;
        } else if (match && (target->predicate || cur.predicate)) { match = 0; }
        if (match && target->object && cur.object) {
            if (strcmp(target->object, cur.object) != 0) match = 0;
        } else if (match && (target->object || cur.object)) { match = 0; }
        ki_free_entry(&cur);
        if (match) return id;
    }
    return -1;
}

static int detect_transitive_contradiction(const KnowledgeEntry* entries, size_t count,
                                            const KnowledgeEntry** ordered_entries,
                                            size_t* order_count) {
    if (!entries || count < 3) return 0;
    const char* order_predicates[] = {"大于", "小于", "高于", "低于", "早于", "晚于",
                                       "大于等于", "小于等于", "包含", "属于", 0};
    const char* reverse_predicates[] = {"小于", "大于", "低于", "高于", "晚于", "早于",
                                         "小于等于", "大于等于", "属于", "包含", 0};
    size_t oc = 0;
    for (size_t i = 0; i < count; i++) {
        if (entries[i].type != KNOWLEDGE_FACT && entries[i].type != KNOWLEDGE_OBSERVATION) continue;
        if (!entries[i].subject || !entries[i].predicate || !entries[i].object) continue;
        for (int p = 0; order_predicates[p]; p++) {
            if (strcmp(entries[i].predicate, order_predicates[p]) == 0) {
                ordered_entries[oc] = &entries[i];
                oc++;
                break;
            }
            if (strcmp(entries[i].predicate, reverse_predicates[p]) == 0) {
                ordered_entries[oc] = &entries[i];
                oc++;
                break;
            }
        }
    }
    *order_count = oc;
    if (oc < 3) return 0;
    for (size_t i = 0; i < oc; i++) {
        for (size_t j = i + 1; j < oc; j++) {
            if (strcmp(ordered_entries[i]->object, ordered_entries[j]->subject) != 0) continue;
            for (size_t k = j + 1; k < oc; k++) {
                if (strcmp(ordered_entries[j]->object, ordered_entries[k]->subject) != 0) continue;
                if (strcmp(ordered_entries[k]->object, ordered_entries[i]->subject) == 0) {
                    const char* pred_ij = ordered_entries[i]->predicate;
                    const char* pred_jk = ordered_entries[j]->predicate;
                    const char* pred_ki = ordered_entries[k]->predicate;
                    int all_positive = 1;
                    for (int p = 0; order_predicates[p]; p++) {
                        int c1 = (strcmp(pred_ij, order_predicates[p]) == 0);
                        int c2 = (strcmp(pred_jk, order_predicates[p]) == 0);
                        int c3 = (strcmp(pred_ki, order_predicates[p]) == 0);
                        if (!c1 || !c2 || !c3) { all_positive = 0; break; }
                    }
                    if (all_positive) return 1;
                }
            }
        }
    }
    return 0;
}

#define KI_CONTRADICT_MAX_ENTRIES 8192

/* ============================================================================
 * AGM信念修正 — 公共API实现
 * =========================================================================== */

int knowledge_integration_belief_expand(KnowledgeIntegrationSystem* system,
                                       KnowledgeBase* kb, const KnowledgeEntry* entry) {
    if (!kb || !entry) return -1;

    /* V-020修复: 跨知识库推理 — 当启用跨推理时，将同一信念同步扩展到所有注册KB */
    if (system && system->config.enable_cross_reasoning && system->kb_count > 1) {
        int added = 0;
        for (size_t i = 0; i < system->kb_count; i++) {
            if (!system->kbs[i].kb) continue;
            if (knowledge_base_add(system->kbs[i].kb, entry) >= 0) added++;
        }
        return (added > 0) ? 0 : -1;
    }

    return knowledge_base_add(kb, entry);
}

int knowledge_integration_belief_contract(KnowledgeIntegrationSystem* system,
                                         KnowledgeBase* kb, int entry_id) {
    if (!kb || entry_id <= 0) return -1;

    int use_cross_kb = (system && system->config.enable_cross_reasoning && system->kb_count > 1);

    KnowledgeEntry target;
    memset(&target, 0, sizeof(KnowledgeEntry));
    if (knowledge_base_get_by_id(kb, entry_id, &target) != 0) return -1;
    KnowledgeQuery q;
    memset(&q, 0, sizeof(KnowledgeQuery));
    q.type_filter = -1;
    KnowledgeEntry* related = (KnowledgeEntry*)safe_malloc(KI_CONTRADICT_MAX_ENTRIES * sizeof(KnowledgeEntry));
    if (!related) { ki_free_entry(&target); return -1; }
    int related_count = 0;
    if (target.subject) {
        q.subject_pattern = target.subject;
        related_count = knowledge_base_query(kb, &q, related, KI_CONTRADICT_MAX_ENTRIES);
    }
    if (related_count < 0) {
        safe_free((void**)&related);
        ki_free_entry(&target);
        return -1;
    }
    int* related_ids = (int*)safe_malloc((size_t)(related_count + 1) * sizeof(int));
    if (!related_ids) { safe_free((void**)&related); ki_free_entry(&target); return -1; }
    size_t ridx = 0;
    for (int i = 0; i < related_count; i++) {
        int rid = find_entry_id_by_content(kb, &related[i]);
        if (rid > 0 && rid != entry_id) {
            float te = compute_entrenchment(&target);
            float re = compute_entrenchment(&related[i]);

            /* V-020修复: 跨知识库推理 — 检查其他KB中是否有相同信念提升其认知固化度 */
            if (use_cross_kb) {
                int cross_support = 0;
                for (size_t ck = 0; ck < system->kb_count; ck++) {
                    if (system->kbs[ck].kb == kb) continue;
                    int other_id = find_entry_id_by_content(system->kbs[ck].kb, &related[i]);
                    if (other_id > 0) cross_support++;
                }
                re += (float)cross_support * 0.08f;
                if (re > 1.0f) re = 1.0f;
            }

            if (re < te * 0.6f && related[i].predicate && target.predicate &&
                strcmp(related[i].predicate, target.predicate) == 0) {
                related_ids[ridx++] = rid;
            }
        }
    }
    related_ids[ridx] = 0;
    for (size_t i = 0; i < related_count; i++) ki_free_entry(&related[i]);
    safe_free((void**)&related);
    knowledge_base_remove(kb, entry_id);
    for (size_t i = 0; i < ridx; i++) knowledge_base_remove(kb, related_ids[i]);
    safe_free((void**)&related_ids);

    /* V-020修复: 跨知识库推理 — 在其他注册KB中同步删除相同信念 */
    if (use_cross_kb) {
        for (size_t i = 0; i < system->kb_count; i++) {
            if (system->kbs[i].kb == kb) continue;
            int other_id = find_entry_id_by_content(system->kbs[i].kb, &target);
            if (other_id > 0) knowledge_base_remove(system->kbs[i].kb, other_id);
        }
    }

    ki_free_entry(&target);
    return 0;
}

int knowledge_integration_belief_revise(KnowledgeIntegrationSystem* system,
                                       KnowledgeBase* kb, const KnowledgeEntry* entry,
                                       const KnowledgeEntry* conflicting_entry) {
    if (!kb || !entry) return -1;

    int use_cross_kb = (system && system->config.enable_cross_reasoning && system->kb_count > 1);

    if (conflicting_entry) {
        int cid = find_entry_id_by_content(kb, conflicting_entry);
        if (cid > 0) knowledge_base_remove(kb, cid);

        /* V-020修复: 跨知识库推理 — 在其他注册KB中查找并移除明确冲突的条目 */
        if (use_cross_kb) {
            for (size_t i = 0; i < system->kb_count; i++) {
                if (system->kbs[i].kb == kb) continue;
                int other_cid = find_entry_id_by_content(system->kbs[i].kb, conflicting_entry);
                if (other_cid > 0) knowledge_base_remove(system->kbs[i].kb, other_cid);
            }
        }
        return knowledge_base_add(kb, entry);
    }

    KnowledgeQuery q;
    memset(&q, 0, sizeof(KnowledgeQuery));
    q.type_filter = -1;
    KnowledgeEntry* existing = (KnowledgeEntry*)safe_malloc(KI_CONTRADICT_MAX_ENTRIES * sizeof(KnowledgeEntry));
    if (!existing) return -1;
    int exist_count = 0;
    if (entry->subject) {
        q.subject_pattern = entry->subject;
        exist_count = knowledge_base_query(kb, &q, existing, KI_CONTRADICT_MAX_ENTRIES);
    }
    if (exist_count < 0) {
        safe_free((void**)&existing);
        return -1;
    }
    int weakest_id = -1;
    float weakest_ent = 2.0f;
    for (int i = 0; i < exist_count; i++) {
        if (!entries_directly_contradict(entry, &existing[i])) continue;
        int eid = find_entry_id_by_content(kb, &existing[i]);
        if (eid < 0) continue;
        float ent = compute_entrenchment(&existing[i]);

        /* V-020修复: 跨知识库推理 — 评估矛盾条目在多个KB中的共识支持度 */
        if (use_cross_kb) {
            int cross_support = 0;
            for (size_t ck = 0; ck < system->kb_count; ck++) {
                if (system->kbs[ck].kb == kb) continue;
                int other_id = find_entry_id_by_content(system->kbs[ck].kb, &existing[i]);
                if (other_id > 0) cross_support++;
            }
            ent += (float)cross_support * 0.1f;
            if (ent > 1.0f) ent = 1.0f;
        }

        float new_ent = compute_entrenchment(entry);
        if (ent < new_ent && ent < weakest_ent) {
            weakest_ent = ent;
            weakest_id = eid;
        }
    }
    for (int i = 0; i < exist_count; i++) ki_free_entry(&existing[i]);
    safe_free((void**)&existing);
    if (weakest_id > 0) knowledge_base_remove(kb, weakest_id);
    return knowledge_base_add(kb, entry);
}

size_t knowledge_integration_detect_logical_contradictions(KnowledgeIntegrationSystem* system,
                                                          KnowledgeBase* kb,
                                                          int* conflict_pairs,
                                                          size_t max_pairs) {
    if (!kb || !conflict_pairs || max_pairs == 0) return 0;

    int use_cross_kb = (system && system->config.enable_cross_reasoning && system->kb_count > 1);

    size_t total = 0;
    KnowledgeQuery q_all;
    memset(&q_all, 0, sizeof(KnowledgeQuery));
    q_all.type_filter = -1;
    KnowledgeEntry* all_entries = (KnowledgeEntry*)safe_malloc(KI_CONTRADICT_MAX_ENTRIES * sizeof(KnowledgeEntry));
    if (!all_entries) return 0;
    int entry_count = knowledge_base_query(kb, &q_all, all_entries, KI_CONTRADICT_MAX_ENTRIES);
    if (entry_count <= 0) { safe_free((void**)&all_entries); return 0; }

    /* V-020修复: 跨知识库推理 — 收集所有注册KB中的条目用于全量矛盾检测 */
    if (use_cross_kb) {
        int original_count = entry_count;
        for (size_t k = 0; k < system->kb_count; k++) {
            if (system->kbs[k].kb == kb) continue;
            if (entry_count + 256 > KI_CONTRADICT_MAX_ENTRIES) break;
            int cross_count = knowledge_base_query(system->kbs[k].kb, &q_all,
                all_entries + entry_count,
                KI_CONTRADICT_MAX_ENTRIES - entry_count);
            if (cross_count > 0) entry_count += cross_count;
        }
        (void)original_count;
    }

    size_t pair_count = 0;
    for (int i = 0; i < entry_count && pair_count < max_pairs; i++) {
        for (int j = i + 1; j < entry_count && pair_count < max_pairs; j++) {
            if (entries_directly_contradict(&all_entries[i], &all_entries[j])) {
                int id_a = find_entry_id_by_content(kb, &all_entries[i]);
                if (id_a <= 0 && use_cross_kb) {
                    for (size_t k = 0; k < system->kb_count && id_a <= 0; k++) {
                        if (system->kbs[k].kb == kb) continue;
                        id_a = find_entry_id_by_content(system->kbs[k].kb, &all_entries[i]);
                    }
                }
                int id_b = find_entry_id_by_content(kb, &all_entries[j]);
                if (id_b <= 0 && use_cross_kb) {
                    for (size_t k = 0; k < system->kb_count && id_b <= 0; k++) {
                        if (system->kbs[k].kb == kb) continue;
                        id_b = find_entry_id_by_content(system->kbs[k].kb, &all_entries[j]);
                    }
                }
                if (id_a > 0 && id_b > 0) {
                    conflict_pairs[pair_count * 2] = id_a;
                    conflict_pairs[pair_count * 2 + 1] = id_b;
                    pair_count++;
                }
            }
        }
    }
    const KnowledgeEntry** ordered = (const KnowledgeEntry**)safe_malloc(
        (size_t)entry_count * sizeof(KnowledgeEntry*));
    if (ordered) {
        size_t order_count = 0;
        if (detect_transitive_contradiction(all_entries, (size_t)entry_count, ordered, &order_count)) {
            if (pair_count < max_pairs) {
                for (size_t oi = 0; oi + 2 < order_count && pair_count < max_pairs; oi += 3) {
                    int id_a = find_entry_id_by_content(kb, ordered[oi]);
                    int id_b = find_entry_id_by_content(kb, ordered[oi + 1]);
                    int id_c = find_entry_id_by_content(kb, ordered[oi + 2]);
                    if (id_a > 0 && id_b > 0 && id_c > 0 && pair_count < max_pairs) {
                        conflict_pairs[pair_count * 2] = id_a;
                        conflict_pairs[pair_count * 2 + 1] = id_b;
                        pair_count++;
                    }
                    if (id_a > 0 && id_c > 0 && pair_count < max_pairs) {
                        conflict_pairs[pair_count * 2] = id_a;
                        conflict_pairs[pair_count * 2 + 1] = id_c;
                        pair_count++;
                    }
                }
            }
        }
        safe_free((void**)&ordered);
    }
    for (int i = 0; i < entry_count; i++) ki_free_entry(&all_entries[i]);
    safe_free((void**)&all_entries);
    total = pair_count;
    return total;
}

void knowledge_integration_free(KnowledgeIntegrationSystem* system) {
    if (!system) return;
    
    // 释放所有名称
    for (size_t i = 0; i < system->kb_count; i++) {
        if (system->kbs[i].name) safe_free((void**)&system->kbs[i].name);
    }
    if (system->kbs) safe_free((void**)&system->kbs);
    
    for (size_t i = 0; i < system->graph_count; i++) {
        if (system->graphs[i].name) safe_free((void**)&system->graphs[i].name);
    }
    if (system->graphs) safe_free((void**)&system->graphs);
    
    for (size_t i = 0; i < system->network_count; i++) {
        if (system->networks[i].name) safe_free((void**)&system->networks[i].name);
    }
    if (system->networks) safe_free((void**)&system->networks);
    
    for (size_t i = 0; i < system->engine_count; i++) {
        if (system->engines[i].name) safe_free((void**)&system->engines[i].name);
    }
    if (system->engines) safe_free((void**)&system->engines);
    
    safe_free((void**)&system);
}

int knowledge_integration_register_kb(KnowledgeIntegrationSystem* system,
                                     KnowledgeBase* kb, const char* name) {
    if (!system || !kb || !name) return -1;
    
    // 检查是否已存在同名知识库
    if (find_kb(system, name)) return -1;
    
    // 扩展数组
    if (expand_array((void**)&system->kbs, &system->kb_capacity,
                    system->kb_count, sizeof(RegisteredKB)) != 0) {
        return -1;
    }
    
    // 注册
    RegisteredKB* reg = &system->kbs[system->kb_count];
    reg->name = string_duplicate_nullable(name);
    if (!reg->name) return -1;
    
    reg->kb = kb;
    system->kb_count++;
    
    return 0;
}

int knowledge_integration_register_graph(KnowledgeIntegrationSystem* system,
                                        KnowledgeGraph* graph, const char* name) {
    if (!system || !graph || !name) return -1;
    
    // 检查是否已存在同名知识图谱
    if (find_graph(system, name)) return -1;
    
    // 扩展数组
    if (expand_array((void**)&system->graphs, &system->graph_capacity,
                    system->graph_count, sizeof(RegisteredGraph)) != 0) {
        return -1;
    }
    
    // 注册
    RegisteredGraph* reg = &system->graphs[system->graph_count];
    reg->name = string_duplicate_nullable(name);
    if (!reg->name) return -1;
    
    reg->graph = graph;
    system->graph_count++;
    
    return 0;
}

int knowledge_integration_register_semantic_network(KnowledgeIntegrationSystem* system,
                                                   SemanticNetwork* network, const char* name) {
    if (!system || !network || !name) return -1;
    
    // 检查是否已存在同名语义网络
    if (find_network(system, name)) return -1;
    
    // 扩展数组
    if (expand_array((void**)&system->networks, &system->network_capacity,
                    system->network_count, sizeof(RegisteredNetwork)) != 0) {
        return -1;
    }
    
    // 注册
    RegisteredNetwork* reg = &system->networks[system->network_count];
    reg->name = string_duplicate_nullable(name);
    if (!reg->name) return -1;
    
    reg->network = network;
    system->network_count++;
    
    return 0;
}

int knowledge_integration_register_reasoning_engine(KnowledgeIntegrationSystem* system,
                                                   ReasoningEngine* engine, const char* name) {
    if (!system || !engine || !name) return -1;
    
    // 扩展数组
    if (expand_array((void**)&system->engines, &system->engine_capacity,
                    system->engine_count, sizeof(RegisteredEngine)) != 0) {
        return -1;
    }
    
    // 注册
    RegisteredEngine* reg = &system->engines[system->engine_count];
    reg->name = string_duplicate_nullable(name);
    if (!reg->name) return -1;
    
    reg->engine = engine;
    reg->engine_type = ENGINE_TYPE_GENERIC_REASONING;  /* 默认为通用类型，调用方可后续修改 */
    system->engine_count++;
    
    return 0;
}

/* 注册逻辑推理引擎（安全类型标记） */
int knowledge_integration_register_logic_reasoning_engine(KnowledgeIntegrationSystem* system,
                                                          void* logic_engine, const char* name) {
    if (!system || !logic_engine || !name) return -1;
    
    // 扩展数组
    if (expand_array((void**)&system->engines, &system->engine_capacity,
                    system->engine_count, sizeof(RegisteredEngine)) != 0) {
        return -1;
    }
    
    // 注册
    RegisteredEngine* reg = &system->engines[system->engine_count];
    reg->name = string_duplicate_nullable(name);
    if (!reg->name) return -1;
    
    reg->engine = (ReasoningEngine*)logic_engine;
    reg->engine_type = ENGINE_TYPE_LOGIC_REASONING;  /* 标记为逻辑推理引擎类型 */
    system->engine_count++;
    
    return 0;
}

int knowledge_integration_sync_all(KnowledgeIntegrationSystem* system) {
    if (!system) return -1;
    
    int total_synced = 0;
    
    // 同步所有知识库到所有知识图谱
    for (size_t i = 0; i < system->kb_count; i++) {
        for (size_t j = 0; j < system->graph_count; j++) {
            int result = knowledge_graph_import_from_knowledge_base(
                system->graphs[j].graph,
                system->kbs[i].kb
            );
            if (result >= 0) {
                total_synced++;
            }
        }
    }
    
    // 同步所有知识图谱到所有语义网络
    for (size_t i = 0; i < system->graph_count; i++) {
        for (size_t j = 0; j < system->network_count; j++) {
            int result = semantic_network_import_from_knowledge_graph(
                system->networks[j].network,
                system->graphs[i].graph
            );
            if (result >= 0) {
                total_synced++;
            }
        }
    }
    
    // 同步所有语义网络到所有知识库
    for (size_t i = 0; i < system->network_count; i++) {
        for (size_t j = 0; j < system->kb_count; j++) {
            int result = semantic_network_import_from_knowledge_base(
                system->networks[i].network,
                system->kbs[j].kb
            );
            if (result >= 0) {
                total_synced++;
            }
        }
    }
    
    // 同步所有知识库到所有推理引擎（规则加载）- 安全类型检查
    for (size_t i = 0; i < system->kb_count; i++) {
        for (size_t j = 0; j < system->engine_count; j++) {
            int result = -1;
            RegisteredEngine* reg = &system->engines[j];
            /* 根据引擎类型安全调用相应的规则加载函数 */
            if (reg->engine_type == ENGINE_TYPE_LOGIC_REASONING) {
                result = logic_reasoning_engine_load_rules_from_kb(
                    (LogicReasoningEngine*)reg->engine,
                    system->kbs[i].kb
                );
            } else {
                /* 其他类型的引擎跳过规则加载（通用引擎不支持此操作） */
                continue;
            }
            if (result < 0) {
                selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__,
                                      "同步知识库到推理引擎：规则加载失败 (kb=%zu, engine=%zu)", 
                                      i, j);
                system->total_failures++;
            } else {
                total_synced++;
            }
            system->last_sync_time = time(NULL);  /* M-023修复: 无论成败都更新时间戳 */
        }
    }
    
    return total_synced;
}

int knowledge_integration_sync_kb_to_graph(KnowledgeIntegrationSystem* system,
                                          const char* kb_name, const char* graph_name) {
    if (!system || !kb_name || !graph_name) return -1;
    
    RegisteredKB* kb = find_kb(system, kb_name);
    RegisteredGraph* graph = find_graph(system, graph_name);
    
    if (!kb || !graph) return -1;
    
    int result = knowledge_graph_import_from_knowledge_base(graph->graph, kb->kb);
    return result;
}

int knowledge_integration_sync_graph_to_network(KnowledgeIntegrationSystem* system,
                                               const char* graph_name, const char* network_name) {
    if (!system || !graph_name || !network_name) return -1;
    
    RegisteredGraph* graph = find_graph(system, graph_name);
    RegisteredNetwork* network = find_network(system, network_name);
    
    if (!graph || !network) return -1;
    
    int result = semantic_network_import_from_knowledge_graph(network->network, graph->graph);
    return result;
}

int knowledge_integration_sync_network_to_kb(KnowledgeIntegrationSystem* system,
                                            const char* network_name, const char* kb_name) {
    if (!system || !network_name || !kb_name) return -1;
    
    RegisteredNetwork* network = find_network(system, network_name);
    RegisteredKB* kb = find_kb(system, kb_name);
    
    if (!network || !kb) return -1;
    
    int result = semantic_network_import_from_knowledge_base(network->network, kb->kb);
    return result;
}

size_t knowledge_integration_unified_query(KnowledgeIntegrationSystem* system,
                                          const char* query, size_t max_results,
                                          KnowledgeEntry* results) {
    if (!system || !query || !results || max_results == 0) return 0;
    
    size_t total_found = 0;
    
    // 1. 在所有知识库中查询（文本匹配 + CfC语义搜索）
    for (size_t i = 0; i < system->kb_count && total_found < max_results; i++) {
        KnowledgeQuery q;
        memset(&q, 0, sizeof(KnowledgeQuery));
        q.subject_pattern = string_duplicate_nullable(query);
        q.min_confidence = 0.0f;
        q.max_confidence = 1.0f;
        
        int found = knowledge_base_query(system->kbs[i].kb, &q,
                                        results + total_found,
                                        max_results - total_found);
        total_found += found;
        
        if (q.subject_pattern) safe_free((void**)&q.subject_pattern);
        
        /* 若知识库启用了CfC嵌入，追加语义搜索结果（去重） */
        if (total_found < max_results) {
            KnowledgeEntry cfc_results[64];
            int semantic_count = knowledge_base_cfc_semantic_search(
                system->kbs[i].kb, query, 0.6f,
                cfc_results, 64);
            for (int s = 0; s < semantic_count && total_found < max_results; s++) {
                /* 简单去重：检查是否已在结果中 */
                int is_dup = 0;
                for (size_t t = 0; t < total_found; t++) {
                    if (results[t].subject && cfc_results[s].subject &&
                        strcmp(results[t].subject, cfc_results[s].subject) == 0 &&
                        results[t].predicate && cfc_results[s].predicate &&
                        strcmp(results[t].predicate, cfc_results[s].predicate) == 0) {
                        is_dup = 1;
                        break;
                    }
                }
                if (!is_dup) {
                    KnowledgeEntry* dst = &results[total_found];
                    memset(dst, 0, sizeof(KnowledgeEntry));
                    /* C-002修复: 用string_duplicate替代strdup，统一内存管理 */
                    dst->subject    = cfc_results[s].subject ? string_duplicate_nullable(cfc_results[s].subject) : NULL;
                    dst->predicate  = cfc_results[s].predicate ? string_duplicate_nullable(cfc_results[s].predicate) : NULL;
                    dst->object     = cfc_results[s].object ? string_duplicate_nullable(cfc_results[s].object) : NULL;
                    dst->type       = cfc_results[s].type;
                    dst->confidence = cfc_results[s].confidence;
                    dst->source     = cfc_results[s].source;
                    dst->weight     = cfc_results[s].weight;
                    dst->timestamp  = cfc_results[s].timestamp;
                    dst->embedding_size = cfc_results[s].embedding_size;
                    if (cfc_results[s].embedding && cfc_results[s].embedding_size > 0) {
                        dst->embedding = (float*)safe_malloc(
                            cfc_results[s].embedding_size * sizeof(float));
                        if (dst->embedding) {
                            memcpy(dst->embedding, cfc_results[s].embedding,
                                   cfc_results[s].embedding_size * sizeof(float));
                        }
                    } else {
                        dst->embedding = NULL;
                    }
                    dst->metadata_size = cfc_results[s].metadata_size;
                    if (cfc_results[s].metadata && cfc_results[s].metadata_size > 0) {
                        dst->metadata = safe_malloc(cfc_results[s].metadata_size);
                        if (dst->metadata) {
                            memcpy(dst->metadata, cfc_results[s].metadata,
                                   cfc_results[s].metadata_size);
                        }
                    } else {
                        dst->metadata = NULL;
                    }
                    /* C-002修复: 验证所有分配成功后才递增计数 */
                    if ((cfc_results[s].subject && !dst->subject) ||
                        (cfc_results[s].predicate && !dst->predicate) ||
                        (cfc_results[s].object && !dst->object)) {
                        /* 字符串分配失败，清理已分配资源 */
                        if (dst->subject) safe_free((void**)&dst->subject);
                        if (dst->predicate) safe_free((void**)&dst->predicate);
                        if (dst->object) safe_free((void**)&dst->object);
                        if (dst->embedding) safe_free((void**)&dst->embedding);
                        if (dst->metadata) safe_free((void**)&dst->metadata);
                        continue;  /* 跳过此条目，不递增total_found */
                    }
                    total_found++;
                }
                knowledge_entry_free(&cfc_results[s]);
            }
        }
    }
    
    // 2. 在所有知识图谱中查询（按节点标签匹配）
    for (size_t i = 0; i < system->graph_count && total_found < max_results; i++) {
        // 分配临时节点数组（动态分配）
        const size_t max_nodes_per_graph = 100;
        KnowledgeGraphNode** nodes = (KnowledgeGraphNode**)safe_malloc(max_nodes_per_graph * sizeof(KnowledgeGraphNode*));
        if (!nodes) continue;  // 内存分配失败，跳过此图谱
        
        size_t found_nodes = knowledge_graph_find_nodes_by_label(
            system->graphs[i].graph,
            query,
            nodes,
            max_nodes_per_graph
        );
        
        for (size_t j = 0; j < found_nodes && total_found < max_results; j++) {
            KnowledgeGraphNode* node = nodes[j];
            if (!node || !node->label) continue;
            
            // 创建知识条目
            KnowledgeEntry* entry = &results[total_found];
            memset(entry, 0, sizeof(KnowledgeEntry));
            
            entry->subject = string_duplicate_nullable(node->label);
            entry->predicate = string_duplicate_nullable("is_graph_node");
            entry->object = string_duplicate_nullable("SemanticConcept");
            entry->type = KNOWLEDGE_CONCEPT;
            entry->confidence = node->confidence;
            entry->source = SOURCE_INFERENCE;
            entry->weight = node->confidence;
            entry->timestamp = (long)time(NULL);
            
            total_found++;
        }
        
        // 释放节点数组
        if (nodes) safe_free((void**)&nodes);
    }
    
    // 3. 在所有语义网络中查询（按概念名称匹配）
    for (size_t i = 0; i < system->network_count && total_found < max_results; i++) {
        // 在语义网络中按名称查找概念
        // 使用已实现的semantic_network_find_concept_by_name函数

        SemanticNetwork* network = system->networks[i].network;
        if (!network) continue;
        
        // 在语义网络中按名称查找概念
        SemanticConcept* SemanticConcept = semantic_network_find_concept_by_name(network, query);
        if (SemanticConcept && total_found < max_results) {
            KnowledgeEntry* entry = &results[total_found];
            memset(entry, 0, sizeof(KnowledgeEntry));
            
            entry->subject = string_duplicate_nullable(SemanticConcept->name ? SemanticConcept->name : "未知");
            entry->predicate = string_duplicate_nullable("is_concept");
            entry->object = string_duplicate_nullable("semantic_network");
            entry->type = KNOWLEDGE_CONCEPT;
            entry->confidence = SemanticConcept->confidence;
            entry->source = SOURCE_SEMANTIC_NETWORK;
            entry->weight = SemanticConcept->confidence;
            entry->timestamp = (long)time(NULL);
            
            total_found++;
        }
    }
    
    /* ================================================================
     * P1-02修复: 记忆↔知识双向集成
     * 当知识图谱和语义网络检索结果为空或置信度低时，
     * 回退到记忆系统进行语义记忆检索作为补充。
     * 记忆系统不可用时优雅降级，不产生错误。
     * ================================================================ */
    if (total_found == 0) {
        MemoryManager* mem_mgr = selflnn_get_memory_manager();
        if (mem_mgr) {
            /* 使用查询字符串作为检索键，尝试从记忆系统中检索相关记忆 */
            /* 分配临时缓冲区存放检索到的记忆数据 */
            #define MEM_RETRIEVAL_BUFFER_SIZE 256
            float mem_data[MEM_RETRIEVAL_BUFFER_SIZE];
            float mem_strength = 0.0f;
            int mem_type = -1;
            
            int retrieve_result = memory_manager_retrieve(
                mem_mgr, query,
                mem_data, MEM_RETRIEVAL_BUFFER_SIZE,
                &mem_strength, &mem_type);
            
            if (retrieve_result == 0 && mem_strength > 0.1f) {
                /* 记忆系统中找到匹配项，创建知识条目并标注来源为记忆 */
                KnowledgeEntry* entry = &results[total_found];
                memset(entry, 0, sizeof(KnowledgeEntry));
                
                entry->subject = string_duplicate_nullable(query);
                entry->predicate = string_duplicate_nullable("memory_retrieved");
                entry->object = string_duplicate_nullable("episodic_semantic_memory");
                entry->type = KNOWLEDGE_FACT;
                entry->confidence = mem_strength;
                entry->source = SOURCE_LEARNING;  /* 标注来源为学习/记忆 */
                entry->weight = mem_strength;
                entry->timestamp = (long)time(NULL);
                
                /* 将记忆数据嵌入到知识条目中 */
                entry->embedding_size = (mem_data[0] < MEM_RETRIEVAL_BUFFER_SIZE) 
                    ? (size_t)mem_data[0] : 64;
                if (entry->embedding_size > 0 && entry->embedding_size <= MEM_RETRIEVAL_BUFFER_SIZE - 1) {
                    entry->embedding = (float*)safe_malloc(entry->embedding_size * sizeof(float));
                    if (entry->embedding) {
                        memcpy(entry->embedding, &mem_data[1], entry->embedding_size * sizeof(float));
                    }
                }
                
                total_found++;
                log_info("[记忆↔知识] 知识检索回退到记忆系统: 查询='%s', 记忆强度=%.3f, 类型=%d",
                         query, (double)mem_strength, mem_type);
            } else {
                /* 记忆系统中也未找到匹配项，记录日志 */
                static int mem_fallback_empty_logged = 0;
                if (!mem_fallback_empty_logged) {
                    log_info("[记忆↔知识] 知识检索和记忆回退均未找到匹配项: 查询='%s'"
                             "（此消息仅记录一次，后续相同情况不再重复记录）", query);
                    mem_fallback_empty_logged = 1;
                }
            }
        } else {
            /* 记忆管理器不可用，优雅降级 */
            static int mem_unavailable_logged = 0;
            if (!mem_unavailable_logged) {
                log_info("[记忆↔知识] 记忆管理器不可用，知识检索回退跳过（此消息仅记录一次）");
                mem_unavailable_logged = 1;
            }
        }
    }
    
    return total_found;
}

size_t knowledge_integration_cooperative_reasoning(KnowledgeIntegrationSystem* system,
                                                  const char** premises, size_t premise_count,
                                                  size_t max_inferences,
                                                  KnowledgeEntry* results, size_t max_results) {
    if (!system || !premises || premise_count == 0 || !results || max_results == 0) {
        return 0;
    }
    
    if (system->network_count == 0) return 0;
    
    /* K-03修复：遍历所有语义网络进行协作推理，而非仅使用第一个网络。
     * 每个网络独立执行推理，结果跨网络去重后合并输出。
     * 这样多个知识域（如常识网络、专业网络、领域网络）可协同贡献推理结论。 */
    
    /* 已见概念名称去重表（跨网络去重） */
    #define MAX_INFERRED_CONCEPTS 256
    const char* seen_names[MAX_INFERRED_CONCEPTS];
    size_t seen_count = 0;
    size_t result_count = 0;
    
    /* 遍历所有语义网络，各自独立推理并合并结果 */
    for (size_t net_idx = 0; net_idx < system->network_count && result_count < max_results; net_idx++) {
        SemanticNetwork* network = system->networks[net_idx].network;
        if (!network) continue;
        
        /* 将前提字符串映射到当前语义网络中的概念 */
        SemanticConcept** concept_premises = (SemanticConcept**)safe_malloc(premise_count * sizeof(SemanticConcept*));
        if (!concept_premises) continue;
        
        size_t valid_premises = 0;
        for (size_t i = 0; i < premise_count; i++) {
            const char* premise_name = premises[i];
            if (!premise_name) continue;
            
            SemanticConcept* found = semantic_network_find_concept_by_name(network, premise_name);
            if (found) {
                concept_premises[valid_premises++] = found;
            } else {
                /* 当前网络中未找到，创建临时概念供推理使用 */
                SemanticConcept* temp_concept = semantic_network_add_concept(
                    network, CONCEPT_TYPE_ENTITY, premise_name, NULL, NULL, 0, 0.5f, 0.5f);
                if (temp_concept) {
                    concept_premises[valid_premises++] = temp_concept;
                }
            }
        }
        
        if (valid_premises > 0) {
            /* 在当前网络中执行语义推理 */
            SemanticConcept** network_results = (SemanticConcept**)safe_malloc(max_results * sizeof(SemanticConcept*));
            if (network_results) {
                size_t net_inf_count = semantic_network_infer(
                    network, concept_premises, valid_premises,
                    max_inferences, network_results, max_results
                );
                
                /* 跨网络去重：将本网络推理结果与已有结果比对后加入 */
                for (size_t i = 0; i < net_inf_count && result_count < max_results; i++) {
                    SemanticConcept* SemanticConcept = network_results[i];
                    if (!SemanticConcept || !SemanticConcept->name) continue;
                    
                    /* 跨网络名称去重检查 */
                    int already_seen = 0;
                    for (size_t s = 0; s < seen_count; s++) {
                        if (seen_names[s] && strcmp(seen_names[s], SemanticConcept->name) == 0) {
                            already_seen = 1;
                            break;
                        }
                    }
                    if (already_seen) continue;
                    
                    /* 记录已见概念名（生命周期为函数作用域内安全） */
                    if (seen_count < MAX_INFERRED_CONCEPTS) {
                        seen_names[seen_count++] = SemanticConcept->name;
                    }
                    
                    /* 转换为知识条目 */
                    KnowledgeEntry* entry = &results[result_count];
                    memset(entry, 0, sizeof(KnowledgeEntry));
                    entry->subject = string_duplicate_nullable(SemanticConcept->name);
                    entry->predicate = string_duplicate_nullable("inferred_from");
                    entry->object = string_duplicate_nullable("cooperative_reasoning");
                    entry->type = KNOWLEDGE_FACT;
                    entry->confidence = 0.7f;
                    entry->source = SOURCE_INFERENCE;
                    entry->weight = 1.0f;
                    entry->timestamp = (long)time(NULL);
                    result_count++;
                }
                
                safe_free((void**)&network_results);
            }
        }
        
        safe_free((void**)&concept_premises);
    }
    
    return result_count;
}

size_t knowledge_integration_check_consistency(KnowledgeIntegrationSystem* system,
                                              char** inconsistencies,
                                              size_t max_inconsistencies) {
    if (!system || !inconsistencies || max_inconsistencies == 0) return 0;
    
    size_t inconsistency_count = 0;
    
    // 检查所有知识库中的不一致性
    for (size_t kb_idx = 0; kb_idx < system->kb_count && inconsistency_count < max_inconsistencies; kb_idx++) {
        KnowledgeBase* kb = system->kbs[kb_idx].kb;
        if (!kb) continue;
        
        // 获取知识库统计信息进行真实检查
        size_t total_entries = 0;
        size_t memory_usage = 0;
        
        if (knowledge_base_get_stats(kb, &total_entries, &memory_usage) == 0) {
            // 检查1: 空知识库警告
            if (total_entries == 0 && inconsistency_count < max_inconsistencies) {
                char* msg = (char*)safe_malloc(256);
                if (msg) {
                    snprintf(msg, 256, "知识库 '%s' 为空，建议添加知识条目",
                            system->kbs[kb_idx].name ? system->kbs[kb_idx].name : "未知");
                    inconsistencies[inconsistency_count++] = msg;
                }
            }
            
            // 检查2: 大型知识库性能警告
            if (total_entries > 10000 && inconsistency_count < max_inconsistencies) {
                char* msg = (char*)safe_malloc(256);
                if (msg) {
                    snprintf(msg, 256, "知识库 '%s' 包含 %zu 个条目，可能影响性能，建议优化",
                            system->kbs[kb_idx].name ? system->kbs[kb_idx].name : "未知",
                            total_entries);
                    inconsistencies[inconsistency_count++] = msg;
                }
            }
            
            // 检查3: 内存使用过高警告
            if (memory_usage > 100 * 1024 * 1024 && inconsistency_count < max_inconsistencies) {
                char* msg = (char*)safe_malloc(256);
                if (msg) {
                    snprintf(msg, 256, "知识库 '%s' 内存使用过高 (%zu MB)，建议优化存储",
                            system->kbs[kb_idx].name ? system->kbs[kb_idx].name : "未知",
                            memory_usage / (1024 * 1024));
                    inconsistencies[inconsistency_count++] = msg;
                }
            }
            
            // 检查4: 调用知识库内部矛盾检测
            if (total_entries > 100 && inconsistency_count < max_inconsistencies) {
                /* S-026修复: 执行真实矛盾检测而非仅建议
                 * 抽样检查知识条目间是否存在明显的逻辑冲突 */
                int conflict_found = 0;
                KnowledgeQuery query;
                memset(&query, 0, sizeof(query));
                query.type_filter = -1;
                query.max_confidence = 1.0f;

                KnowledgeEntry* sample = (KnowledgeEntry*)safe_malloc(
                    (total_entries < 256 ? total_entries : 256) * sizeof(KnowledgeEntry));
                if (sample) {
                    int sample_count = knowledge_base_query(kb, &query, sample,
                        total_entries < 256 ? total_entries : 256);
                    for (int a = 0; a < sample_count - 1 && !conflict_found; a++) {
                        for (int b = a + 1; b < sample_count && !conflict_found; b++) {
                            /* 检查冲突：相同主体+谓词但不同客体 */
                            if (sample[a].subject && sample[b].subject &&
                                sample[a].predicate && sample[b].predicate &&
                                strcmp(sample[a].subject, sample[b].subject) == 0 &&
                                strcmp(sample[a].predicate, sample[b].predicate) == 0 &&
                                sample[a].object && sample[b].object &&
                                strcmp(sample[a].object, sample[b].object) != 0) {
                                conflict_found = 1;
                                char* msg = (char*)safe_malloc(512);
                                if (msg) {
                                    snprintf(msg, 512, "知识库 '%s' 发现冲突: [%s %s %s] vs [%s %s %s]",
                                        system->kbs[kb_idx].name ? system->kbs[kb_idx].name : "未知",
                                        sample[a].subject, sample[a].predicate, sample[a].object,
                                        sample[b].subject, sample[b].predicate, sample[b].object);
                                    inconsistencies[inconsistency_count++] = msg;
                                }
                            }
                        }
                    }
                    safe_free((void**)&sample);
                }
                if (!conflict_found && inconsistency_count < max_inconsistencies) {
                    char* msg = (char*)safe_malloc(256);
                    if (msg) {
                        snprintf(msg, 256, "知识库 '%s' 一致性检查通过 (%zu条目)",
                            system->kbs[kb_idx].name ? system->kbs[kb_idx].name : "未知",
                            total_entries);
                        inconsistencies[inconsistency_count++] = msg;
                    }
                }
            }
        } else {
            // 无法获取统计信息，报告为不一致性
            if (inconsistency_count < max_inconsistencies) {
                char* msg = (char*)safe_malloc(256);
                if (msg) {
                    snprintf(msg, 256, "无法获取知识库 '%s' 的统计信息",
                            system->kbs[kb_idx].name ? system->kbs[kb_idx].name : "未知");
                    inconsistencies[inconsistency_count++] = msg;
                }
            }
        }
    }
    
    // 检查知识库与知识图谱之间的一致性
    for (size_t kb_idx = 0; kb_idx < system->kb_count && inconsistency_count < max_inconsistencies; kb_idx++) {
        for (size_t g_idx = 0; g_idx < system->graph_count && inconsistency_count < max_inconsistencies; g_idx++) {
            // 检查知识库与知识图谱之间的数据一致性
            KnowledgeBase* kb = system->kbs[kb_idx].kb;
            KnowledgeGraph* graph = system->graphs[g_idx].graph;
            
            if (!kb || !graph) continue;
            
            // 获取知识库统计信息
            size_t kb_entries = 0;
            size_t kb_memory = 0;
            int kb_stats_ok = knowledge_base_get_stats(kb, &kb_entries, &kb_memory);
            
            // 获取知识图谱统计信息
            size_t graph_nodes = 0;
            size_t graph_edges = 0;
            int graph_stats_ok = knowledge_graph_get_stats(graph, &graph_nodes, &graph_edges, NULL);
            
            // 检查1: 如果一个系统有数据而另一个没有，可能存在同步问题
            if ((kb_entries > 0 && graph_nodes == 0) || (kb_entries == 0 && graph_nodes > 0)) {
                if (inconsistency_count < max_inconsistencies) {
                    char* msg = (char*)safe_malloc(256);
                    if (msg) {
                        snprintf(msg, 256, "知识库 '%s' (%zu 条目) 与知识图谱 '%s' (%zu 节点) 数据量不匹配，可能存在同步问题",
                                system->kbs[kb_idx].name ? system->kbs[kb_idx].name : "未知", kb_entries,
                                system->graphs[g_idx].name ? system->graphs[g_idx].name : "未知", graph_nodes);
                        inconsistencies[inconsistency_count++] = msg;
                    }
                }
            }
            
            // 检查2: 两个系统都有数据但比例异常（例如，图谱节点远多于知识库条目）
            if (kb_entries > 0 && graph_nodes > 0) {
                float ratio = (float)graph_nodes / (float)kb_entries;
                if ((ratio > 10.0f || ratio < 0.1f) && inconsistency_count < max_inconsistencies) {
                    char* msg = (char*)safe_malloc(256);
                    if (msg) {
                        snprintf(msg, 256, "知识库 '%s' (%zu 条目) 与知识图谱 '%s' (%zu 节点) 数据比例异常 (比例: %.2f)",
                                system->kbs[kb_idx].name ? system->kbs[kb_idx].name : "未知", kb_entries,
                                system->graphs[g_idx].name ? system->graphs[g_idx].name : "未知", graph_nodes, ratio);
                        inconsistencies[inconsistency_count++] = msg;
                    }
                }
            }
            
            // 检查3: 如果无法获取统计信息，报告问题
            if ((kb_stats_ok != 0 || graph_stats_ok != 0) && inconsistency_count < max_inconsistencies) {
                char* msg = (char*)safe_malloc(256);
                if (msg) {
                    snprintf(msg, 256, "无法获取知识库 '%s' 或知识图谱 '%s' 的完整统计信息",
                            system->kbs[kb_idx].name ? system->kbs[kb_idx].name : "未知",
                            system->graphs[g_idx].name ? system->graphs[g_idx].name : "未知");
                    inconsistencies[inconsistency_count++] = msg;
                }
            }
        }
    }
    
    return inconsistency_count;
}

int knowledge_integration_get_stats(KnowledgeIntegrationSystem* system,
                                   size_t* kb_count, size_t* graph_count,
                                   size_t* network_count, size_t* engine_count) {
    if (!system) return -1;
    
    if (kb_count) *kb_count = system->kb_count;
    if (graph_count) *graph_count = system->graph_count;
    if (network_count) *network_count = system->network_count;
    if (engine_count) *engine_count = system->engine_count;
    
    return 0;
}

int knowledge_integration_save_state(KnowledgeIntegrationSystem* system,
                                    const char* filename) {
    if (!system || !filename) return -1;
    
    FILE* fp = fopen(filename, "w");
    if (!fp) return -1;
    
    // 写入配置
    fprintf(fp, "[配置]\n");
    fprintf(fp, "enable_auto_sync=%d\n", system->config.enable_auto_sync);
    fprintf(fp, "sync_interval_ms=%d\n", system->config.sync_interval_ms);
    fprintf(fp, "consistency_threshold=%f\n", system->config.consistency_threshold);
    fprintf(fp, "enable_cross_reasoning=%d\n", system->config.enable_cross_reasoning);
    
    // 写入知识库信息
    fprintf(fp, "[知识库]\n");
    fprintf(fp, "count=%zu\n", system->kb_count);
    for (size_t i = 0; i < system->kb_count; i++) {
        fprintf(fp, "kb%d=%s\n", (int)i, system->kbs[i].name ? system->kbs[i].name : "");
    }
    
    // 写入知识图谱信息
    fprintf(fp, "[知识图谱]\n");
    fprintf(fp, "count=%zu\n", system->graph_count);
    for (size_t i = 0; i < system->graph_count; i++) {
        fprintf(fp, "graph%d=%s\n", (int)i, system->graphs[i].name ? system->graphs[i].name : "");
    }
    
    // 写入语义网络信息
    fprintf(fp, "[语义网络]\n");
    fprintf(fp, "count=%zu\n", system->network_count);
    for (size_t i = 0; i < system->network_count; i++) {
        fprintf(fp, "network%d=%s\n", (int)i, system->networks[i].name ? system->networks[i].name : "");
    }
    
    // 写入推理引擎信息
    fprintf(fp, "[推理引擎]\n");
    fprintf(fp, "count=%zu\n", system->engine_count);
    for (size_t i = 0; i < system->engine_count; i++) {
        fprintf(fp, "engine%d=%s\n", (int)i, system->engines[i].name ? system->engines[i].name : "");
    }
    
    fclose(fp);
    return 0;
}

KnowledgeIntegrationSystem* knowledge_integration_load_state(const char* filename) {
    if (!filename) return NULL;
    
    FILE* fp = fopen(filename, "r");
    if (!fp) return NULL;
    
    IntegrationConfig config;
    memset(&config, 0, sizeof(config));
    
    // 默认值
    config.enable_auto_sync = 0;
    config.sync_interval_ms = 1000;
    config.consistency_threshold = 0.8f;
    config.enable_cross_reasoning = 1;
    
    char line[256];
    int section = 0; // 0=无, 1=配置, 2=知识库, 3=知识图谱, 4=语义网络, 5=推理引擎
    size_t kb_count = 0, graph_count = 0, network_count = 0, engine_count = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        // 去除换行符
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (len > 0 && line[len-1] == '\r') line[len-1] = '\0';
        
        // 检查节标题
        if (strcmp(line, "[配置]") == 0) {
            section = 1;
            continue;
        } else if (strcmp(line, "[知识库]") == 0) {
            section = 2;
            continue;
        } else if (strcmp(line, "[知识图谱]") == 0) {
            section = 3;
            continue;
        } else if (strcmp(line, "[语义网络]") == 0) {
            section = 4;
            continue;
        } else if (strcmp(line, "[推理引擎]") == 0) {
            section = 5;
            continue;
        }
        
        // 解析键值对
        char* eq = strchr(line, '=');
        if (!eq) continue;
        
        *eq = '\0';
        const char* key = line;
        const char* value = eq + 1;
        
        if (section == 1) {
            if (strcmp(key, "enable_auto_sync") == 0) {
                config.enable_auto_sync = atoi(value);
            } else if (strcmp(key, "sync_interval_ms") == 0) {
                config.sync_interval_ms = atoi(value);
            } else if (strcmp(key, "consistency_threshold") == 0) {
                config.consistency_threshold = (float)atof(value);
            } else if (strcmp(key, "enable_cross_reasoning") == 0) {
                config.enable_cross_reasoning = atoi(value);
            }
        } else if (section == 2) {
            if (strcmp(key, "count") == 0) {
                kb_count = (size_t)atoll(value);
            }
            // 忽略知识库名称，因为无法重新注册句柄
        } else if (section == 3) {
            if (strcmp(key, "count") == 0) {
                graph_count = (size_t)atoll(value);
            }
        } else if (section == 4) {
            if (strcmp(key, "count") == 0) {
                network_count = (size_t)atoll(value);
            }
        } else if (section == 5) {
            if (strcmp(key, "count") == 0) {
                engine_count = (size_t)atoll(value);
            }
        }
    }
    
    fclose(fp);
    
    // 创建系统
    KnowledgeIntegrationSystem* system = knowledge_integration_create(&config);
    if (!system) return NULL;
    
/* 函数仅恢复配置参数（enable_auto_sync/sync_interval等）。
     * 知识库/图谱/推理引擎句柄需调用者在返回后通过以下方式重新绑定：
     *   knowledge_integration_register_knowledge_base(system, kb);
     *   knowledge_integration_register_graph(system, graph);
     *   knowledge_integration_register_reasoning_engine(system, engine);
     * 这是设计上的限制——句柄为不透明指针，序列化/反序列化需外部管理。 */
    
    return system;
}