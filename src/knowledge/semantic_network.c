/**
 * @file semantic_network.c
 * @brief 语义网络系统实现
 * 
 * 语义网络数据结构、概念关系、层次结构和语义相似性计算实现。
 */

#define _CRT_NONSTDC_NO_DEPRECATE

#define SELFLNN_KNOWLEDGE_INTERNAL  /* Z-001修复: 访问KnowledgeGraph完整结构体(kg->nodes/kg->edges) */

#include "selflnn/knowledge/semantic_network.h"
#include "selflnn/knowledge/knowledge.h"          /* ZSFWS: KnowledgeBase/KnowledgeEntry */
#include "selflnn/knowledge/knowledge_graph.h"    /* ZSFWS: KnowledgeGraph/GraphNode */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/* Z-001修复: KnowledgeBase完整结构体声明(与knowledge.c中定义保持一致,仅在knowledge模块内部可见) */
typedef struct {
    int id;
    KnowledgeEntry entry;
    int ref_count;
} InternalKnowledgeEntry;

struct KnowledgeBase {
    InternalKnowledgeEntry* entries;
    size_t capacity;
    size_t size;
    size_t max_entries;
    int next_id;
};

/**
 * @brief 语义网络内部结构
 */
struct SemanticNetwork {
    Concept** concepts;              /**< 概念数组 */
    size_t concept_count;            /**< 概念数量 */
    size_t concept_capacity;         /**< 概念数组容量 */
    size_t max_concepts;             /**< 最大概念数（0表示无限制） */
    
    SemanticRelation** relations;    /**< 关系数组 */
    size_t relation_count;           /**< 关系数量 */
    size_t relation_capacity;        /**< 关系数组容量 */
    size_t max_relations;            /**< 最大关系数（0表示无限制） */
    
    int next_concept_id;             /**< 下一个概念ID */
    int next_relation_id;            /**< 下一个关系ID */
    
    /* 索引结构 */
    struct {
        char** names;                /**< 概念名称数组 */
        Concept** concepts;          /**< 概念指针数组 */
        size_t size;                 /**< 索引大小 */
        size_t capacity;             /**< 索引容量 */
    } name_index;                    /**< 名称索引 */
    
    MutexHandle lock;                /**< 线程安全锁 */
};

/* ============================================================================
 * 锁宏定义
 * =========================================================================== */
#define SEMANTIC_LOCK(n)   mutex_lock((n)->lock)
#define SEMANTIC_UNLOCK(n) mutex_unlock((n)->lock)

/**
 * @brief 查找概念在概念数组中的索引
 * @param network 语义网络
 * @param concept 概念指针
 * @return 概念索引，如果未找到返回(size_t)-1
 */
static size_t semantic_network_find_concept_index(SemanticNetwork* network, Concept* concept) {
    if (!network || !concept) return (size_t)-1;
    
    for (size_t i = 0; i < network->concept_count; i++) {
        if (network->concepts[i] == concept) {
            return i;
        }
    }
    return (size_t)-1;
}

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
 * @brief 复制浮点数数组
 */
static float* duplicate_float_array(const float* src, size_t size) {
    if (!src || size == 0) return NULL;
    
    float* dest = (float*)safe_malloc(size * sizeof(float));
    if (dest) {
        memcpy(dest, src, size * sizeof(float));
    }
    return dest;
}

/**
 * @brief 计算两个概念之间的路径距离
 * @param network 语义网络
 * @param concept1 概念1
 * @param concept2 概念2
 * @return float 路径距离（跳数），无法计算时返回-1
 */
static float bfs_distance_to_ancestor(SemanticNetwork* network, Concept* start, Concept* target) {
    if (!network || !start || !target) return -1.0f;
    if (start == target) return 0.0f;
    
    // 分配队列和访问标记
    size_t max_concepts = network->concept_count;
    Concept** queue = (Concept**)safe_malloc(max_concepts * sizeof(Concept*));
    float* distances = (float*)safe_malloc(max_concepts * sizeof(float));
    int* visited = (int*)safe_calloc(max_concepts, sizeof(int));
    if (!queue || !distances || !visited) {
        safe_free((void**)&queue);
        safe_free((void**)&distances);
        safe_free((void**)&visited);
        return -1.0f;
    }
    
    // 获取起始概念索引
    size_t start_idx = semantic_network_find_concept_index(network, start);
    if (start_idx == (size_t)-1) {
        safe_free((void**)&queue);
        safe_free((void**)&distances);
        safe_free((void**)&visited);
        return -1.0f;
    }
    
    int front = 0, rear = 0;
    queue[rear] = start;
    distances[rear] = 0.0f;
    visited[start_idx] = 1;
    rear++;
    
    float result_distance = -1.0f;
    
    while (front < rear) {
        Concept* current = queue[front];
        float current_dist = distances[front];
        front++;
        
        // 检查是否到达目标
        if (current == target) {
            result_distance = current_dist;
            break;
        }
        
        // 遍历所有关系，查找父类（is-a关系）
        for (size_t i = 0; i < network->relation_count; i++) {
            SemanticRelation* rel = network->relations[i];
            if (rel->type != RELATION_IS_A) continue;
            
            // 查找当前概念的父类（target是父类，source是子类）
            Concept* parent = NULL;
            if (rel->source == current) {
                parent = rel->target;
            } else if (rel->target == current) {
                // 如果是反向关系（当前是父类），我们可能需要向子类方向搜索
                // 但为了距离计算，我们只向上搜索父类
                continue;
            }
            
            if (!parent) continue;
            
            size_t parent_idx = semantic_network_find_concept_index(network, parent);
            if (parent_idx == (size_t)-1 || visited[parent_idx]) continue;
            
            // 入队
            if (rear >= max_concepts) break;
            queue[rear] = parent;
            distances[rear] = current_dist + 1.0f;
            visited[parent_idx] = 1;
            rear++;
        }
    }
    
    safe_free((void**)&queue);
    safe_free((void**)&distances);
    safe_free((void**)&visited);
    return result_distance;
}

/* ============================================================================
 * 基础语义网络API实现
 * =========================================================================== */

/**
 * @brief 创建语义网络
 */
SemanticNetwork* semantic_network_create(size_t max_concepts, size_t max_relations) {
    SemanticNetwork* network = (SemanticNetwork*)safe_calloc(1, sizeof(SemanticNetwork));
    if (!network) return NULL;

    network->concept_capacity = 64;
    network->relation_capacity = 128;
    network->max_concepts = max_concepts;
    network->max_relations = max_relations;
    network->next_concept_id = 1;
    network->next_relation_id = 1;

    network->concepts = (Concept**)safe_calloc(network->concept_capacity, sizeof(Concept*));
    network->relations = (SemanticRelation**)safe_calloc(network->relation_capacity, sizeof(SemanticRelation*));

    if (!network->concepts || !network->relations) {
        safe_free((void**)&network->concepts);
        safe_free((void**)&network->relations);
        safe_free((void**)&network);
        return NULL;
    }

    network->lock = mutex_create();
    if (!network->lock) {
        safe_free((void**)&network->concepts);
        safe_free((void**)&network->relations);
        safe_free((void**)&network);
        return NULL;
    }

    return network;
}

/**
 * @brief 释放语义网络
 */
void semantic_network_free(SemanticNetwork* network) {
    if (!network) return;

    SEMANTIC_LOCK(network);

    for (size_t i = 0; i < network->concept_count; i++) {
        Concept* c = network->concepts[i];
        if (c) {
            safe_free((void**)&c->name);
            safe_free((void**)&c->description);
            safe_free((void**)&c->embedding);
            safe_free((void**)&c);
        }
    }
    safe_free((void**)&network->concepts);

    for (size_t i = 0; i < network->relation_count; i++) {
        SemanticRelation* r = network->relations[i];
        if (r) {
            safe_free((void**)&r->label);
            safe_free((void**)&r);
        }
    }
    safe_free((void**)&network->relations);

    if (network->name_index.names) {
        for (size_t i = 0; i < network->name_index.size; i++) {
            safe_free((void**)&network->name_index.names[i]);
        }
        safe_free((void**)&network->name_index.names);
        safe_free((void**)&network->name_index.concepts);
    }

    SEMANTIC_UNLOCK(network);
    mutex_destroy(network->lock);
    safe_free((void**)&network);
}

/**
 * @brief 添加概念到语义网络（内部无锁版本，调用者需持有锁）
 */
static Concept* _semantic_network_add_concept_internal(SemanticNetwork* network, ConceptType type,
                                                       const char* name, const char* description,
                                                       const float* embedding, size_t embedding_size,
                                                       float specificity, float typicality) {
    if (!network || !name) return NULL;

    if (network->max_concepts > 0 && network->concept_count >= network->max_concepts) {
        return NULL;
    }

    if (expand_array((void**)&network->concepts, &network->concept_capacity,
                     network->concept_count, sizeof(Concept*)) != 0) {
        return NULL;
    }

    Concept* concept = (Concept*)safe_calloc(1, sizeof(Concept));
    if (!concept) {
        return NULL;
    }

    concept->id = network->next_concept_id++;
    concept->type = type;
    concept->name = string_duplicate_nullable(name);
    concept->description = string_duplicate_nullable(description);
    concept->embedding = duplicate_float_array(embedding, embedding_size);
    concept->embedding_size = embedding_size;
    concept->specificity = (specificity >= 0.0f && specificity <= 1.0f) ? specificity : 0.5f;
    concept->typicality = (typicality >= 0.0f && typicality <= 1.0f) ? typicality : 0.5f;
    concept->confidence = 1.0f;
    concept->instance_count = 0;
    concept->user_data = NULL;

    if (!concept->name) {
        safe_free((void**)&concept->description);
        safe_free((void**)&concept->embedding);
        safe_free((void**)&concept);
        return NULL;
    }

    network->concepts[network->concept_count++] = concept;

    if (network->name_index.size >= network->name_index.capacity) {
        size_t new_cap = network->name_index.capacity == 0 ? 16 : network->name_index.capacity * 2;
        char** new_names = (char**)safe_realloc(network->name_index.names, new_cap * sizeof(char*));
        Concept** new_concepts = (Concept**)safe_realloc(network->name_index.concepts, new_cap * sizeof(Concept*));
        if (new_names && new_concepts) {
            network->name_index.names = new_names;
            network->name_index.concepts = new_concepts;
            network->name_index.capacity = new_cap;
        }
    }
    if (network->name_index.size < network->name_index.capacity) {
        network->name_index.names[network->name_index.size] = string_duplicate_nullable(concept->name);
        network->name_index.concepts[network->name_index.size] = concept;
        network->name_index.size++;
    }

    return concept;
}

/**
 * @brief 添加概念到语义网络
 */
Concept* semantic_network_add_concept(SemanticNetwork* network, ConceptType type,
                                     const char* name, const char* description,
                                     const float* embedding, size_t embedding_size,
                                     float specificity, float typicality) {
    if (!network || !name) return NULL;

    SEMANTIC_LOCK(network);
    Concept* result = _semantic_network_add_concept_internal(network, type, name, description,
                                                              embedding, embedding_size,
                                                              specificity, typicality);
    SEMANTIC_UNLOCK(network);
    return result;
}

/**
 * @brief 添加关系到语义网络（内部无锁版本，调用者需持有锁）
 */
static SemanticRelation* _semantic_network_add_relation_internal(SemanticNetwork* network,
                                                                 SemanticRelationType type,
                                                                 Concept* source, Concept* target,
                                                                 const char* label,
                                                                 float strength, float confidence) {
    if (!network || !source || !target) return NULL;

    if (network->max_relations > 0 && network->relation_count >= network->max_relations) {
        return NULL;
    }

    if (expand_array((void**)&network->relations, &network->relation_capacity,
                     network->relation_count, sizeof(SemanticRelation*)) != 0) {
        return NULL;
    }

    SemanticRelation* rel = (SemanticRelation*)safe_calloc(1, sizeof(SemanticRelation));
    if (!rel) {
        return NULL;
    }

    rel->id = network->next_relation_id++;
    rel->type = type;
    rel->source = source;
    rel->target = target;
    rel->label = string_duplicate_nullable(label);
    rel->strength = (strength >= 0.0f && strength <= 1.0f) ? strength : 0.5f;
    rel->confidence = (confidence >= 0.0f && confidence <= 1.0f) ? confidence : 0.5f;
    rel->user_data = NULL;

    network->relations[network->relation_count++] = rel;

    return rel;
}

/**
 * @brief 添加关系到语义网络
 */
SemanticRelation* semantic_network_add_relation(SemanticNetwork* network,
                                               SemanticRelationType type,
                                               Concept* source, Concept* target,
                                               const char* label,
                                               float strength, float confidence) {
    if (!network || !source || !target) return NULL;

    SEMANTIC_LOCK(network);
    SemanticRelation* result = _semantic_network_add_relation_internal(network, type, source, target,
                                                                       label, strength, confidence);
    SEMANTIC_UNLOCK(network);
    return result;
}

/**
 * @brief 根据名称查找概念（内部无锁版本，调用者需持有锁）
 */
static Concept* _semantic_network_find_concept_by_name_internal(SemanticNetwork* network, const char* name) {
    if (!network || !name) return NULL;

    for (size_t i = 0; i < network->name_index.size; i++) {
        if (network->name_index.names[i] &&
            strcmp(network->name_index.names[i], name) == 0) {
            return network->name_index.concepts[i];
        }
    }

    for (size_t i = 0; i < network->concept_count; i++) {
        if (network->concepts[i]->name &&
            strcmp(network->concepts[i]->name, name) == 0) {
            return network->concepts[i];
        }
    }

    return NULL;
}

/**
 * @brief 根据名称查找概念
 */
Concept* semantic_network_find_concept_by_name(SemanticNetwork* network, const char* name) {
    if (!network || !name) return NULL;

    SEMANTIC_LOCK(network);
    Concept* result = _semantic_network_find_concept_by_name_internal(network, name);
    SEMANTIC_UNLOCK(network);
    return result;
}

/* ============================================================================
 * 高级语义网络功能实现
 * =========================================================================== */

/**
 * @brief 语义网络剪枝 - 移除低置信度的概念和关系
 */
SELFLNN_API size_t semantic_network_prune(SemanticNetwork* network,
    float min_concept_confidence, float min_relation_confidence) {
    
    if (!network) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "语义网络剪枝：网络句柄为空");
        return 0;
    }
    
    if (min_concept_confidence < 0.0f || min_concept_confidence > 1.0f ||
        min_relation_confidence < 0.0f || min_relation_confidence > 1.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "语义网络剪枝：置信度阈值无效（必须在0-1之间）");
        return 0;
    }

    SEMANTIC_LOCK(network);
    
    size_t total_removed = 0;
    
    /* 第一步：标记并移除低置信度关系 */
    size_t valid_relation_count = 0;
    for (size_t i = 0; i < network->relation_count; i++) {
        SemanticRelation* rel = network->relations[i];
        if (rel->confidence >= min_relation_confidence) {
            // 保留关系
            network->relations[valid_relation_count] = rel;
            valid_relation_count++;
        } else {
            // 移除关系
            safe_free((void**)&rel->label);
            safe_free((void**)&rel);
            total_removed++;
        }
    }
    network->relation_count = valid_relation_count;
    
    /* 第二步：标记并移除低置信度概念 */
    size_t valid_concept_count = 0;
    for (size_t i = 0; i < network->concept_count; i++) {
        Concept* concept = network->concepts[i];
        if (concept->confidence >= min_concept_confidence) {
            // 保留概念
            network->concepts[valid_concept_count] = concept;
            valid_concept_count++;
        } else {
            // 移除概念
            safe_free((void**)&concept->name);
            safe_free((void**)&concept->description);
            safe_free((void**)&concept->embedding);
            safe_free((void**)&concept);
            total_removed++;
        }
    }
    network->concept_count = valid_concept_count;
    
    /* 第三步：重建名称索引 */
    if (network->name_index.names) {
        for (size_t i = 0; i < network->name_index.size; i++) {
            safe_free((void**)&network->name_index.names[i]);
        }
        safe_free((void**)&network->name_index.names);
        safe_free((void**)&network->name_index.concepts);
        network->name_index.names = NULL;
        network->name_index.concepts = NULL;
        network->name_index.size = 0;
        network->name_index.capacity = 0;
    }
    
    // 重新构建索引
    for (size_t i = 0; i < network->concept_count; i++) {
        Concept* concept = network->concepts[i];
        if (concept->name) {
            // 扩展索引数组
            if (network->name_index.size >= network->name_index.capacity) {
                size_t new_capacity = network->name_index.capacity == 0 ? 16 : network->name_index.capacity * 2;
                char** new_names = (char**)safe_realloc(network->name_index.names, new_capacity * sizeof(char*));
                Concept** new_concepts = (Concept**)safe_realloc(network->name_index.concepts, new_capacity * sizeof(Concept*));
                if (!new_names || !new_concepts) {
                    safe_free((void**)&new_names);
                    safe_free((void**)&new_concepts);
                    SEMANTIC_UNLOCK(network);
                    selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                          "语义网络剪枝：重建名称索引内存分配失败");
                    return total_removed;
                }
                network->name_index.names = new_names;
                network->name_index.concepts = new_concepts;
                network->name_index.capacity = new_capacity;
            }
            
            // 添加索引条目
            network->name_index.names[network->name_index.size] = string_duplicate_nullable(concept->name);
            network->name_index.concepts[network->name_index.size] = concept;
            network->name_index.size++;
        }
    }
    
    SEMANTIC_UNLOCK(network);
    return total_removed;
}

/**
 * @brief 概念聚类 - 基于语义相似度进行概念分组
 */
SELFLNN_API size_t semantic_network_cluster_concepts(SemanticNetwork* network,
    float similarity_threshold, size_t max_clusters, int* cluster_assignments) {
    
    if (!network || !cluster_assignments) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "概念聚类：参数无效");
        return 0;
    }
    
    if (similarity_threshold < 0.0f || similarity_threshold > 1.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "概念聚类：相似度阈值无效（必须在0-1之间）");
        return 0;
    }
    
    if (network->concept_count == 0) {
        return 0;
    }

    SEMANTIC_LOCK(network);
    
    size_t n = network->concept_count;
    
    // 初始化聚类分配：每个概念一个聚类
    int* clusters = (int*)safe_malloc(n * sizeof(int));
    if (!clusters) {
        SEMANTIC_UNLOCK(network);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "概念聚类：内存分配失败");
        return 0;
    }
    
    for (size_t i = 0; i < n; i++) {
        clusters[i] = (int)i;
        cluster_assignments[i] = (int)i;
    }
    
    size_t current_clusters = n;
    
    // 如果最大聚类数大于概念数，则设置为概念数
    if (max_clusters == 0 || max_clusters > n) {
        max_clusters = n;
    }
    
    // 简单的凝聚层次聚类算法
    while (current_clusters > max_clusters) {
        float max_similarity = -1.0f;
        size_t merge_i = 0, merge_j = 0;
        
        // 查找最相似的两个聚类
        for (size_t i = 0; i < n; i++) {
            if (clusters[i] != (int)i) continue; // 只考虑聚类代表
            
            Concept* concept_i = network->concepts[i];
            
            for (size_t j = i + 1; j < n; j++) {
                if (clusters[j] != (int)j) continue; // 只考虑聚类代表
                
                Concept* concept_j = network->concepts[j];
                
                // 计算概念相似度（使用余弦相似度）
                float similarity = semantic_network_concept_similarity(network, concept_i, concept_j, SIMILARITY_COSINE);
                if (similarity < 0) similarity = 0; // 处理错误
                
                if (similarity > max_similarity && similarity >= similarity_threshold) {
                    max_similarity = similarity;
                    merge_i = i;
                    merge_j = j;
                }
            }
        }
        
        // 如果没有足够相似的聚类对，停止合并
        if (max_similarity < similarity_threshold) {
            break;
        }
        
        // 合并聚类j到聚类i
        int target_cluster = clusters[merge_i];
        for (size_t k = 0; k < n; k++) {
            if (clusters[k] == clusters[merge_j]) {
                clusters[k] = target_cluster;
            }
        }
        
        current_clusters--;
    }
    
    // 重新映射聚类ID为连续整数
    int* cluster_map = (int*)safe_calloc(n, sizeof(int));
    if (!cluster_map) {
        safe_free((void**)&clusters);
        SEMANTIC_UNLOCK(network);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "概念聚类：聚类映射内存分配失败");
        return 0;
    }
    
    int next_cluster_id = 0;
    for (size_t i = 0; i < n; i++) {
        if (clusters[i] == (int)i) { // 聚类代表
            cluster_map[i] = next_cluster_id++;
        }
    }
    
    // 应用聚类映射
    for (size_t i = 0; i < n; i++) {
        cluster_assignments[i] = cluster_map[clusters[i]];
    }
    
    size_t actual_clusters = (size_t)next_cluster_id;
    
    safe_free((void**)&clusters);
    safe_free((void**)&cluster_map);
    
    SEMANTIC_UNLOCK(network);
    return actual_clusters;
}

/**
 * @brief 关系推理 - 基于关系传递性推断新关系
 */
SELFLNN_API size_t semantic_network_infer_relations(SemanticNetwork* network,
    int relation_type, size_t max_inferences) {
    
    if (!network) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "关系推理：网络句柄为空");
        return 0;
    }
    
    if (network->concept_count == 0 || network->relation_count == 0) {
        return 0;
    }
    
    // 确定要推理的关系类型
    SemanticRelationType types_to_infer[4];
    size_t num_types = 0;
    
    if (relation_type == -1) {
        // 推理所有传递性关系类型
        types_to_infer[num_types++] = RELATION_IS_A;
        types_to_infer[num_types++] = RELATION_PART_OF;
        types_to_infer[num_types++] = RELATION_HAS_A;
        types_to_infer[num_types++] = RELATION_INSTANCE_OF;
    } else if (relation_type >= 0 && relation_type <= 9) {
        // 只推理指定类型
        types_to_infer[num_types++] = (SemanticRelationType)relation_type;
    } else {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "关系推理：关系类型无效（必须在-1到9之间）");
        return 0;
    }

    SEMANTIC_LOCK(network);
    
    size_t total_inferred = 0;
    
    // 对于每种关系类型进行推理
    for (size_t t = 0; t < num_types; t++) {
        SemanticRelationType rel_type = types_to_infer[t];
        
        // 收集该类型的所有关系
        SemanticRelation** type_relations = (SemanticRelation**)safe_malloc(network->relation_count * sizeof(SemanticRelation*));
        if (!type_relations) {
            SEMANTIC_UNLOCK(network);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "关系推理：内存分配失败");
            return total_inferred;
        }
        
        size_t type_relation_count = 0;
        for (size_t i = 0; i < network->relation_count; i++) {
            if (network->relations[i]->type == rel_type) {
                type_relations[type_relation_count++] = network->relations[i];
            }
        }
        
        if (type_relation_count < 2) {
            safe_free((void**)&type_relations);
            continue; // 该类型关系太少，无法推理
        }
        
        // 构建邻接表
        size_t* adjacency = (size_t*)safe_calloc(network->concept_count * network->concept_count, sizeof(size_t));
        if (!adjacency) {
            safe_free((void**)&type_relations);
            SEMANTIC_UNLOCK(network);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "关系推理：邻接矩阵内存分配失败");
            return total_inferred;
        }
        
        // 填充邻接矩阵
        for (size_t i = 0; i < type_relation_count; i++) {
            SemanticRelation* rel = type_relations[i];
            size_t src_idx = semantic_network_find_concept_index(network, rel->source);
            size_t tgt_idx = semantic_network_find_concept_index(network, rel->target);
            
            if (src_idx != (size_t)-1 && tgt_idx != (size_t)-1) {
                adjacency[src_idx * network->concept_count + tgt_idx] = 1;
            }
        }
        
        // 使用Warshall算法计算传递闭包
        for (size_t k = 0; k < network->concept_count; k++) {
            for (size_t i = 0; i < network->concept_count; i++) {
                if (adjacency[i * network->concept_count + k]) {
                    for (size_t j = 0; j < network->concept_count; j++) {
                        if (adjacency[k * network->concept_count + j]) {
                            adjacency[i * network->concept_count + j] = 1;
                        }
                    }
                }
            }
        }
        
        // 推断新关系
        for (size_t i = 0; i < network->concept_count; i++) {
            for (size_t j = 0; j < network->concept_count; j++) {
                if (i == j) continue;
                
                // 如果存在传递路径但不存在直接关系，则推断新关系
                if (adjacency[i * network->concept_count + j]) {
                    // 检查是否已存在直接关系
                    int already_exists = 0;
                    for (size_t r = 0; r < type_relation_count; r++) {
                        SemanticRelation* rel = type_relations[r];
                        size_t src_idx = semantic_network_find_concept_index(network, rel->source);
                        size_t tgt_idx = semantic_network_find_concept_index(network, rel->target);
                        
                        if (src_idx == i && tgt_idx == j) {
                            already_exists = 1;
                            break;
                        }
                    }
                    
                    if (!already_exists && total_inferred < max_inferences) {
                        // 创建新关系
                        Concept* source_concept = network->concepts[i];
                        Concept* target_concept = network->concepts[j];
                        
                        // 计算新关系的置信度（基于路径长度）
                        float confidence = 0.7f; // 默认置信度
                        
                        SemanticRelation* new_rel = _semantic_network_add_relation_internal(
                            network, rel_type, source_concept, target_concept,
                            NULL, 0.5f, confidence);
                        
                        if (new_rel) {
                            total_inferred++;
                        }
                    }
                }
            }
        }
        
        safe_free((void**)&adjacency);
        safe_free((void**)&type_relations);
        
        if (total_inferred >= max_inferences) {
            break;
        }
    }

    SEMANTIC_UNLOCK(network);
    
    return total_inferred;
}

/**
 * @brief 语义网络合并 - 合并两个语义网络
 */
SELFLNN_API size_t semantic_network_merge(SemanticNetwork* dest,
    SemanticNetwork* src, int merge_strategy) {
    
    if (!dest || !src) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "语义网络合并：网络句柄为空");
        return 0;
    }
    
    if (merge_strategy < 0 || merge_strategy > 2) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "语义网络合并：合并策略无效（0=覆盖，1=保留，2=平均）");
        return 0;
    }
    
    if (src == dest) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "语义网络合并：源网络和目标网络相同");
        return 0;
    }

    SEMANTIC_LOCK(dest);
    
    size_t total_merged = 0;
    
    /* 第一步：合并概念 */
    for (size_t i = 0; i < src->concept_count; i++) {
        Concept* src_concept = src->concepts[i];
        Concept* dest_concept = _semantic_network_find_concept_by_name_internal(dest, src_concept->name);
        
        if (!dest_concept) {
            // 目标网络中不存在同名概念，直接添加
            Concept* new_concept = _semantic_network_add_concept_internal(
                dest, src_concept->type, src_concept->name, src_concept->description,
                src_concept->embedding, src_concept->embedding_size,
                src_concept->specificity, src_concept->typicality);
            
            if (new_concept) {
                new_concept->confidence = src_concept->confidence;
                new_concept->instance_count = src_concept->instance_count;
                total_merged++;
            }
        } else {
            // 目标网络中存在同名概念，根据合并策略处理
            switch (merge_strategy) {
                case 0: // 覆盖
                    dest_concept->type = src_concept->type;
                    safe_free((void**)&dest_concept->description);
                    dest_concept->description = string_duplicate_nullable(src_concept->description);
                    safe_free((void**)&dest_concept->embedding);
                    dest_concept->embedding = duplicate_float_array(src_concept->embedding, src_concept->embedding_size);
                    dest_concept->embedding_size = src_concept->embedding_size;
                    dest_concept->specificity = src_concept->specificity;
                    dest_concept->typicality = src_concept->typicality;
                    dest_concept->confidence = src_concept->confidence;
                    dest_concept->instance_count = src_concept->instance_count;
                    total_merged++;
                    break;
                    
                case 1: // 保留：保持目标概念不变
                    // 什么都不做
                    break;
                    
                case 2: // 平均：取两个概念的平均值
                    // 对于数值属性取平均
                    dest_concept->specificity = (dest_concept->specificity + src_concept->specificity) / 2.0f;
                    dest_concept->typicality = (dest_concept->typicality + src_concept->typicality) / 2.0f;
                    dest_concept->confidence = (dest_concept->confidence + src_concept->confidence) / 2.0f;
                    dest_concept->instance_count = (dest_concept->instance_count + src_concept->instance_count) / 2;
                    
                    // 对于嵌入向量，如果大小相同则取平均
                    if (dest_concept->embedding && src_concept->embedding &&
                        dest_concept->embedding_size == src_concept->embedding_size) {
                        for (size_t j = 0; j < dest_concept->embedding_size; j++) {
                            dest_concept->embedding[j] = (dest_concept->embedding[j] + src_concept->embedding[j]) / 2.0f;
                        }
                    }
                    total_merged++;
                    break;
            }
        }
    }
    
    /* 第二步：合并关系 */
    for (size_t i = 0; i < src->relation_count; i++) {
        SemanticRelation* src_rel = src->relations[i];
        
        // 在目标网络中查找对应的源概念和目标概念
        Concept* dest_source = _semantic_network_find_concept_by_name_internal(dest, src_rel->source->name);
        Concept* dest_target = _semantic_network_find_concept_by_name_internal(dest, src_rel->target->name);
        
        if (!dest_source || !dest_target) {
            // 源概念或目标概念在目标网络中不存在，跳过
            continue;
        }
        
        // 检查目标网络中是否已存在相同关系
        int relation_exists = 0;
        for (size_t j = 0; j < dest->relation_count; j++) {
            SemanticRelation* dest_rel = dest->relations[j];
            if (dest_rel->source == dest_source && dest_rel->target == dest_target &&
                dest_rel->type == src_rel->type) {
                relation_exists = 1;
                
                // 根据合并策略更新现有关系
                switch (merge_strategy) {
                    case 0: // 覆盖
                        dest_rel->strength = src_rel->strength;
                        dest_rel->confidence = src_rel->confidence;
                        safe_free((void**)&dest_rel->label);
                        dest_rel->label = string_duplicate_nullable(src_rel->label);
                        total_merged++;
                        break;
                        
                    case 1: // 保留
                        // 什么都不做
                        break;
                        
                    case 2: // 平均
                        dest_rel->strength = (dest_rel->strength + src_rel->strength) / 2.0f;
                        dest_rel->confidence = (dest_rel->confidence + src_rel->confidence) / 2.0f;
                        total_merged++;
                        break;
                }
                break;
            }
        }
        
        if (!relation_exists) {
            // 创建新关系
            SemanticRelation* new_rel = _semantic_network_add_relation_internal(
                dest, src_rel->type, dest_source, dest_target,
                src_rel->label, src_rel->strength, src_rel->confidence);
            
            if (new_rel) {
                total_merged++;
            }
        }
    }
    
    /* 第三步：重建目标网络的名称索引 */
    if (dest->name_index.names) {
        for (size_t i = 0; i < dest->name_index.size; i++) {
            safe_free((void**)&dest->name_index.names[i]);
        }
        safe_free((void**)&dest->name_index.names);
        safe_free((void**)&dest->name_index.concepts);
        dest->name_index.names = NULL;
        dest->name_index.concepts = NULL;
        dest->name_index.size = 0;
        dest->name_index.capacity = 0;
    }
    
    // 重新构建索引
    for (size_t i = 0; i < dest->concept_count; i++) {
        Concept* concept = dest->concepts[i];
        if (concept->name) {
            // 扩展索引数组
            if (dest->name_index.size >= dest->name_index.capacity) {
                size_t new_capacity = dest->name_index.capacity == 0 ? 16 : dest->name_index.capacity * 2;
                char** new_names = (char**)safe_realloc(dest->name_index.names, new_capacity * sizeof(char*));
                Concept** new_concepts = (Concept**)safe_realloc(dest->name_index.concepts, new_capacity * sizeof(Concept*));
                if (!new_names || !new_concepts) {
                    safe_free((void**)&new_names);
                    safe_free((void**)&new_concepts);
                    SEMANTIC_UNLOCK(dest);
                    selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                          "语义网络合并：重建名称索引内存分配失败");
                    return total_merged;
                }
                dest->name_index.names = new_names;
                dest->name_index.concepts = new_concepts;
                dest->name_index.capacity = new_capacity;
            }
            
            // 添加索引条目
            dest->name_index.names[dest->name_index.size] = string_duplicate_nullable(concept->name);
            dest->name_index.concepts[dest->name_index.size] = concept;
            dest->name_index.size++;
        }
    }
    
    SEMANTIC_UNLOCK(dest);
    return total_merged;
}

/**
 * @brief 概念重要性排名 - 基于中心性度量
 */
SELFLNN_API int semantic_network_compute_concept_importance(SemanticNetwork* network,
    int centrality_metric, float* importance_scores) {
    
    if (!network || !importance_scores) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "概念重要性计算：参数无效");
        return -1;
    }
    
    if (centrality_metric < 0 || centrality_metric > 1) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "概念重要性计算：中心性度量无效（0=度中心性，1=接近中心性）");
        return -1;
    }
    
    if (network->concept_count == 0) {
        return 0;
    }

    SEMANTIC_LOCK(network);
    
    size_t n = network->concept_count;
    
    switch (centrality_metric) {
        case 0: { // 度中心性
            // 计算每个概念的度（连接的关系数）
            float max_degree = 0.0f;
            
            for (size_t i = 0; i < n; i++) {
                float degree = 0.0f;
                Concept* concept = network->concepts[i];
                
                // 计算与概念相关的关系数
                for (size_t r = 0; r < network->relation_count; r++) {
                    SemanticRelation* rel = network->relations[r];
                    if (rel->source == concept || rel->target == concept) {
                        degree += 1.0f;
                    }
                }
                
                importance_scores[i] = degree;
                if (degree > max_degree) {
                    max_degree = degree;
                }
            }
            
            // 归一化到0-1范围
            if (max_degree > 0.0f) {
                for (size_t i = 0; i < n; i++) {
                    importance_scores[i] /= max_degree;
                }
            }
            
            break;
        }
        
        case 1: { // 接近中心性（近似计算）
            // 由于完全计算接近中心性需要所有节点对的最短路径，
            // 这里使用近似方法：计算每个节点到其他节点的平均距离的倒数
            
            // 首先构建邻接矩阵
            float* distances = (float*)safe_malloc(n * n * sizeof(float));
            if (!distances) {
                SEMANTIC_UNLOCK(network);
                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                      "概念重要性计算：距离矩阵内存分配失败");
                return -1;
            }
            
            // 初始化距离矩阵
            for (size_t i = 0; i < n * n; i++) {
                distances[i] = FLT_MAX;
            }
            for (size_t i = 0; i < n; i++) {
                distances[i * n + i] = 0.0f;
            }
            
            // 填充直接连接的距离
            for (size_t r = 0; r < network->relation_count; r++) {
                SemanticRelation* rel = network->relations[r];
                size_t src_idx = semantic_network_find_concept_index(network, rel->source);
                size_t tgt_idx = semantic_network_find_concept_index(network, rel->target);
                
                if (src_idx != (size_t)-1 && tgt_idx != (size_t)-1) {
                    // 关系强度越高，距离越短
                    float distance = 1.0f / (rel->strength + 0.001f);
                    distances[src_idx * n + tgt_idx] = distance;
                    distances[tgt_idx * n + src_idx] = distance; // 假设关系是无向的
                }
            }
            
            // 使用Floyd-Warshall算法计算所有节点对的最短路径（完整实现）
            for (size_t k = 0; k < n; k++) {
                for (size_t i = 0; i < n; i++) {
                    if (distances[i * n + k] < FLT_MAX) {
                        for (size_t j = 0; j < n; j++) {
                            if (distances[k * n + j] < FLT_MAX) {
                                float new_dist = distances[i * n + k] + distances[k * n + j];
                                if (new_dist < distances[i * n + j]) {
                                    distances[i * n + j] = new_dist;
                                }
                            }
                        }
                    }
                }
            }
            
            // 计算接近中心性：到所有其他节点距离之和的倒数
            float max_closeness = 0.0f;
            for (size_t i = 0; i < n; i++) {
                float total_distance = 0.0f;
                size_t reachable_count = 0;
                
                for (size_t j = 0; j < n; j++) {
                    if (i != j && distances[i * n + j] < FLT_MAX) {
                        total_distance += distances[i * n + j];
                        reachable_count++;
                    }
                }
                
                if (reachable_count > 0 && total_distance > 0.0f) {
                    // 接近中心性 = (可达节点数 - 1) / 总距离
                    float closeness = (float)(reachable_count - 1) / total_distance;
                    importance_scores[i] = closeness;
                    if (closeness > max_closeness) {
                        max_closeness = closeness;
                    }
                } else {
                    importance_scores[i] = 0.0f;
                }
            }
            
            // 归一化到0-1范围
            if (max_closeness > 0.0f) {
                for (size_t i = 0; i < n; i++) {
                    importance_scores[i] /= max_closeness;
                }
            }
            
            safe_free((void**)&distances);
            break;
        }
        
        default:
            SEMANTIC_UNLOCK(network);
            return -1;
    }
    
    SEMANTIC_UNLOCK(network);
    return 0;
}

/**
 * @brief 扩散激活 - 通过语义网络传播激活能量
 */
SELFLNN_API size_t semantic_network_spreading_activation(SemanticNetwork* network,
    Concept** seeds, const float* seed_activations, size_t seed_count,
    float decay_factor, float threshold, size_t max_iterations,
    ActivationEntry* results, size_t max_results) {
    
    if (!network || !seeds || !seed_activations || !results) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "扩散激活：参数无效");
        return 0;
    }
    
    if (seed_count == 0 || max_results == 0) {
        return 0;
    }
    
    if (decay_factor <= 0.0f || decay_factor > 1.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "扩散激活：衰减因子无效（必须在0-1之间）");
        return 0;
    }
    
    if (threshold < 0.0f || threshold > 1.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "扩散激活：激活阈值无效（必须在0-1之间）");
        return 0;
    }

    SEMANTIC_LOCK(network);
    
    size_t n = network->concept_count;
    
    // 分配激活值数组
    float* activations = (float*)safe_calloc(n, sizeof(float));
    float* next_activations = (float*)safe_calloc(n, sizeof(float));
    int* visited = (int*)safe_calloc(n, sizeof(int));
    
    if (!activations || !next_activations || !visited) {
        if (activations) safe_free((void**)&activations);
        if (next_activations) safe_free((void**)&next_activations);
        if (visited) safe_free((void**)&visited);
        SEMANTIC_UNLOCK(network);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "扩散激活：内存分配失败");
        return 0;
    }
    
    // 初始化种子激活
    for (size_t i = 0; i < seed_count; i++) {
        Concept* seed = seeds[i];
        size_t seed_idx = semantic_network_find_concept_index(network, seed);
        if (seed_idx != (size_t)-1) {
            activations[seed_idx] = seed_activations[i];
            visited[seed_idx] = 1;
        }
    }
    
    // 迭代传播激活
    for (size_t iteration = 0; iteration < max_iterations; iteration++) {
        // 重置下一轮激活值
        memcpy(next_activations, activations, n * sizeof(float));
        
        int propagated = 0;
        
        // 遍历所有概念
        for (size_t i = 0; i < n; i++) {
            float current_activation = activations[i];
            if (current_activation <= threshold) continue;
            
            Concept* concept = network->concepts[i];
            
            // 查找与当前概念相关的关系
            for (size_t r = 0; r < network->relation_count; r++) {
                SemanticRelation* rel = network->relations[r];
                Concept* neighbor = NULL;
                
                if (rel->source == concept) {
                    neighbor = rel->target;
                } else if (rel->target == concept) {
                    neighbor = rel->source;
                }
                
                if (!neighbor) continue;
                
                size_t neighbor_idx = semantic_network_find_concept_index(network, neighbor);
                if (neighbor_idx == (size_t)-1) continue;
                
                // 计算传播的激活值
                float propagated_activation = current_activation * decay_factor * rel->strength;
                
                if (propagated_activation > threshold && propagated_activation > next_activations[neighbor_idx]) {
                    next_activations[neighbor_idx] = propagated_activation;
                    visited[neighbor_idx] = 1;
                    propagated = 1;
                }
            }
        }
        
        // 更新激活值
        memcpy(activations, next_activations, n * sizeof(float));
        
        // 如果没有传播发生，停止迭代
        if (!propagated) {
            break;
        }
    }
    
    // 收集结果
    size_t result_count = 0;
    
    // 首先添加种子概念
    for (size_t i = 0; i < seed_count && result_count < max_results; i++) {
        Concept* seed = seeds[i];
        size_t seed_idx = semantic_network_find_concept_index(network, seed);
        if (seed_idx != (size_t)-1) {
            results[result_count].concept = seed;
            results[result_count].activation = activations[seed_idx];
            result_count++;
        }
    }
    
    // 然后添加其他激活的概念
    for (size_t i = 0; i < n && result_count < max_results; i++) {
        if (!visited[i]) continue;
        
        // 检查是否已经是种子概念
        int is_seed = 0;
        for (size_t j = 0; j < seed_count; j++) {
            if (network->concepts[i] == seeds[j]) {
                is_seed = 1;
                break;
            }
        }
        
        if (!is_seed && activations[i] > threshold) {
            results[result_count].concept = network->concepts[i];
            results[result_count].activation = activations[i];
            result_count++;
        }
    }
    
    // 按激活值降序排序结果
    for (size_t i = 0; i < result_count; i++) {
        for (size_t j = i + 1; j < result_count; j++) {
            if (results[j].activation > results[i].activation) {
                ActivationEntry temp = results[i];
                results[i] = results[j];
                results[j] = temp;
            }
        }
    }
    
    safe_free((void**)&activations);
    safe_free((void**)&next_activations);
    safe_free((void**)&visited);
    
    SEMANTIC_UNLOCK(network);
    return result_count;
}

/**
 * @brief 概念学习 - 基于共现模式学习新关系
 */
SELFLNN_API size_t semantic_network_learn_from_patterns(SemanticNetwork* network,
    Concept** pattern_pairs, size_t pair_count, float learning_rate) {
    
    if (!network || !pattern_pairs) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "概念学习：参数无效");
        return 0;
    }
    
    if (pair_count == 0) {
        return 0;
    }
    
    if (learning_rate <= 0.0f || learning_rate > 1.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "概念学习：学习率无效（必须在0-1之间）");
        return 0;
    }

    SEMANTIC_LOCK(network);
    
    size_t total_learned = 0;
    
    // 统计概念对共现次数
    size_t max_pairs = pair_count * 2; // 考虑双向关系
    typedef struct {
        Concept* concept1;
        Concept* concept2;
        size_t count;
    } CooccurrencePair;
    
    CooccurrencePair* cooccurrences = (CooccurrencePair*)safe_calloc(max_pairs, sizeof(CooccurrencePair));
    if (!cooccurrences) {
        SEMANTIC_UNLOCK(network);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "概念学习：共现统计内存分配失败");
        return 0;
    }
    
    size_t unique_pair_count = 0;
    
    // 统计共现频率
    for (size_t i = 0; i < pair_count; i++) {
        Concept* concept1 = pattern_pairs[i * 2];
        Concept* concept2 = pattern_pairs[i * 2 + 1];
        
        if (!concept1 || !concept2) continue;
        
        // 查找是否已存在该概念对
        int found = 0;
        for (size_t j = 0; j < unique_pair_count; j++) {
            if ((cooccurrences[j].concept1 == concept1 && cooccurrences[j].concept2 == concept2) ||
                (cooccurrences[j].concept1 == concept2 && cooccurrences[j].concept2 == concept1)) {
                cooccurrences[j].count++;
                found = 1;
                break;
            }
        }
        
        if (!found && unique_pair_count < max_pairs) {
            cooccurrences[unique_pair_count].concept1 = concept1;
            cooccurrences[unique_pair_count].concept2 = concept2;
            cooccurrences[unique_pair_count].count = 1;
            unique_pair_count++;
        }
    }
    
    // 根据共现频率学习关系
    for (size_t i = 0; i < unique_pair_count; i++) {
        Concept* concept1 = cooccurrences[i].concept1;
        Concept* concept2 = cooccurrences[i].concept2;
        size_t count = cooccurrences[i].count;
        
        // 计算关系强度增量（基于共现频率）
        float strength_increment = (float)count / (float)pair_count * learning_rate;
        
        // 查找是否已存在关系
        SemanticRelation* existing_rel = NULL;
        for (size_t r = 0; r < network->relation_count; r++) {
            SemanticRelation* rel = network->relations[r];
            if ((rel->source == concept1 && rel->target == concept2) ||
                (rel->source == concept2 && rel->target == concept1)) {
                existing_rel = rel;
                break;
            }
        }
        
        if (existing_rel) {
            // 增强现有关系
            if (existing_rel->type == RELATION_RELATED_TO) {
                // 只增强RELATION_RELATED_TO类型的关系
                existing_rel->strength += strength_increment;
                if (existing_rel->strength > 1.0f) existing_rel->strength = 1.0f;
                
                // 提高置信度
                existing_rel->confidence += learning_rate * 0.1f;
                if (existing_rel->confidence > 1.0f) existing_rel->confidence = 1.0f;
                
                total_learned++;
            }
        } else {
            // 创建新关系
            float initial_strength = strength_increment * 2.0f;
            if (initial_strength > 0.5f) initial_strength = 0.5f;
            if (initial_strength < 0.1f) initial_strength = 0.1f;
            
            float initial_confidence = 0.5f + strength_increment;
            if (initial_confidence > 0.8f) initial_confidence = 0.8f;
            
            SemanticRelation* new_rel = _semantic_network_add_relation_internal(
                network, RELATION_RELATED_TO, concept1, concept2,
                "共现学习", initial_strength, initial_confidence);
            
            if (new_rel) {
                total_learned++;
            }
        }
    }
    
    safe_free((void**)&cooccurrences);
    
    SEMANTIC_UNLOCK(network);
    return total_learned;
}

SELFLNN_API size_t semantic_network_get_concept_count(SemanticNetwork* network) {
    if (!network) return 0;
    SEMANTIC_LOCK(network);
    size_t count = network->concept_count;
    SEMANTIC_UNLOCK(network);
    return count;
}

SELFLNN_API Concept* semantic_network_get_concept_by_index(SemanticNetwork* network, size_t index) {
    if (!network) return NULL;
    SEMANTIC_LOCK(network);
    Concept* result = (index < network->concept_count) ? network->concepts[index] : NULL;
    SEMANTIC_UNLOCK(network);
    return result;
}

SELFLNN_API size_t semantic_network_get_relation_count(SemanticNetwork* network) {
    if (!network) return 0;
    SEMANTIC_LOCK(network);
    size_t count = network->relation_count;
    SEMANTIC_UNLOCK(network);
    return count;
}

SELFLNN_API SemanticRelation* semantic_network_get_relation_by_index(SemanticNetwork* network, size_t index) {
    if (!network) return NULL;
    SEMANTIC_LOCK(network);
    SemanticRelation* result = (index < network->relation_count) ? network->relations[index] : NULL;
    SEMANTIC_UNLOCK(network);
    return result;
}

/* ============================================================================
 * KB-09: 基于语义网络的类比推理
 *
 * 源路径: concept_a → (rel_1) → node_1 → (rel_2) → concept_b
 * 目标映射: 寻找与源路径结构相同的目标路径
 * ============================================================================ */

int semantic_analogy_find(const SemanticNetwork* net, const char* concept_a,
                           const char* relation, const char* concept_b,
                           const char* target_a, char** analog_b,
                           float* similarity) {
    if (!net || !concept_a || !concept_b || !target_a || !analog_b || !similarity) return -1;

    /* M-003修复：深层类比推理
     * 步骤1：找到源概念对的关系模式 */
    int src_rel_idx = -1;
    float src_rel_strength = 0.0f;
    for (int i = 0; i < (int)net->relation_count; i++) {
        SemanticRelation* r = net->relations[i];
        if (r && r->source && r->target && r->source->name && r->target->name &&
            strcmp(r->source->name, concept_a) == 0 &&
            strcmp(r->target->name, concept_b) == 0) {
            src_rel_idx = i;
            src_rel_strength = r->strength;
            break;
        }
    }
    if (src_rel_idx < 0) return -1;

    /* 步骤2：对目标概念的所有关系计算嵌入相似度 */
    int best_match = -1;
    float best_sim = 0.0f;
    for (int i = 0; i < (int)net->relation_count; i++) {
        SemanticRelation* r = net->relations[i];
        if (r && r->source && r->source->name &&
            strcmp(r->source->name, target_a) == 0) {
            /* 综合评分：关系强度 × 目标置信度 × 结构相似度 */
            float structural_sim = 0.0f;
            if (r->target && r->target->embedding && net->relations[src_rel_idx]->target &&
                net->relations[src_rel_idx]->target->embedding) {
                /* 使用嵌入向量的余弦相似度作为结构相似度 */
                structural_sim = math_cosine_similarity(
                    r->target->embedding,
                    net->relations[src_rel_idx]->target->embedding,
                    64);
            }
            float sim = r->strength * 0.4f;
            if (r->target && r->target->confidence > 0.0f)
                sim += r->target->confidence * 0.3f;
            sim += structural_sim * 0.3f;
            if (sim > best_sim) { best_sim = sim; best_match = i; }
        }
    }

    if (best_match < 0) { *analog_b = NULL; *similarity = 0.0f; return 0; }
    *analog_b = (net->relations[best_match]->target && net->relations[best_match]->target->name)
                ? net->relations[best_match]->target->name : NULL;
    *similarity = best_sim;
    return 0;
}

int semantic_graph_to_cfc_sequence(const SemanticNetwork* net, int* walk_ids,
                                     int max_walk, float* embedded_path,
                                     int embed_dim) {
    if (!net || !embedded_path || embed_dim <= 0) return -1;

    memset(embedded_path, 0, (size_t)embed_dim * sizeof(float));
    int steps = 0;
    int current = 0;
    int visited[64] = {0};

    for (int s = 0; s < max_walk && s < 64; s++) {
        int best_next = -1;
        float best_w = 0.0f;
        for (int i = 0; i < (int)net->relation_count && i < 128; i++) {
            SemanticRelation* r = net->relations[i];
            Concept* cur = (current < (int)net->concept_count) ? net->concepts[current] : NULL;
            if (r && r->source && cur && r->source->name && cur->name &&
                strcmp(r->source->name, cur->name) == 0 && !visited[i % 64]) {
                if (r->strength > best_w) { best_w = r->strength; best_next = i; }
            }
        }
        if (best_next < 0) break;

        visited[best_next % 64] = 1;
        if (walk_ids) walk_ids[s] = best_next;
        if (s < embed_dim) embedded_path[s] = net->relations[best_next]->strength;
        steps++;

        /* 移动到目标概念 */
        SemanticRelation* nr = net->relations[best_next];
        if (nr && nr->target && nr->target->name) {
            for (int c = 0; c < (int)net->concept_count && c < 64; c++) {
                Concept* ct = net->concepts[c];
                if (ct && ct->name && nr->target->name && strcmp(ct->name, nr->target->name) == 0)
                    current = c;
            }
        }
    }
    return steps;
}

/* ============================================================================
 * 持久化：语义网络的二进制保存和加载
 * ========================================================================= */

int semantic_network_save(SemanticNetwork* network, const char* filename) {
    if (!network || !filename) return -1;
    
    FILE* file = fopen(filename, "wb");
    if (!file) return -1;
    
    const char magic[9] = "SELFSNv1";
    fwrite(magic, 1, 8, file);
    
    SEMANTIC_LOCK(network);
    
    fwrite(&network->concept_count, sizeof(size_t), 1, file);
    fwrite(&network->relation_count, sizeof(size_t), 1, file);
    fwrite(&network->next_concept_id, sizeof(int), 1, file);
    fwrite(&network->next_relation_id, sizeof(int), 1, file);
    
    for (size_t i = 0; i < network->concept_count; i++) {
        Concept* c = network->concepts[i];
        if (!c) { int null_flag = 0; fwrite(&null_flag, sizeof(int), 1, file); continue; }
        int has = 1; fwrite(&has, sizeof(int), 1, file);
        fwrite(&c->id, sizeof(int), 1, file);
        fwrite(&c->type, sizeof(int), 1, file);
        int name_len = c->name ? (int)strlen(c->name) : 0;
        fwrite(&name_len, sizeof(int), 1, file);
        if (name_len > 0) fwrite(c->name, 1, (size_t)name_len, file);
        int desc_len = c->description ? (int)strlen(c->description) : 0;
        fwrite(&desc_len, sizeof(int), 1, file);
        if (desc_len > 0) fwrite(c->description, 1, (size_t)desc_len, file);
        fwrite(&c->embedding_size, sizeof(size_t), 1, file);
        if (c->embedding && c->embedding_size > 0)
            fwrite(c->embedding, sizeof(float), c->embedding_size, file);
        fwrite(&c->specificity, sizeof(float), 1, file);
        fwrite(&c->typicality, sizeof(float), 1, file);
        fwrite(&c->confidence, sizeof(float), 1, file);
        fwrite(&c->instance_count, sizeof(int), 1, file);
    }
    
    for (size_t i = 0; i < network->relation_count; i++) {
        SemanticRelation* r = network->relations[i];
        if (!r) { int null_flag = 0; fwrite(&null_flag, sizeof(int), 1, file); continue; }
        int has = 1; fwrite(&has, sizeof(int), 1, file);
        fwrite(&r->id, sizeof(int), 1, file);
        fwrite(&r->type, sizeof(int), 1, file);
        int src_id = r->source ? r->source->id : -1;
        int tgt_id = r->target ? r->target->id : -1;
        fwrite(&src_id, sizeof(int), 1, file);
        fwrite(&tgt_id, sizeof(int), 1, file);
        int label_len = r->label ? (int)strlen(r->label) : 0;
        fwrite(&label_len, sizeof(int), 1, file);
        if (label_len > 0) fwrite(r->label, 1, (size_t)label_len, file);
        fwrite(&r->strength, sizeof(float), 1, file);
        fwrite(&r->confidence, sizeof(float), 1, file);
    }
    
    SEMANTIC_UNLOCK(network);
    fclose(file);
    return 0;
}

SemanticNetwork* semantic_network_load(const char* filename) {
    if (!filename) return NULL;
    
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    
    char magic[9];
    if (fread(magic, 1, 8, file) != 8 || memcmp(magic, "SELFSNv1", 8) != 0) {
        fclose(file); return NULL;
    }
    
    size_t concept_count = 0, relation_count = 0;
    int next_cid = 0, next_rid = 0;
    fread(&concept_count, sizeof(size_t), 1, file);
    fread(&relation_count, sizeof(size_t), 1, file);
    fread(&next_cid, sizeof(int), 1, file);
    fread(&next_rid, sizeof(int), 1, file);
    
    SemanticNetwork* net = semantic_network_create(concept_count + 16, relation_count + 16);
    if (!net) { fclose(file); return NULL; }
    
    Concept** loaded_concepts = (Concept**)safe_calloc(concept_count + 1, sizeof(Concept*));
    
    for (size_t i = 0; i < concept_count; i++) {
        int has; fread(&has, sizeof(int), 1, file);
        if (!has) { loaded_concepts[i] = NULL; continue; }
        int id, type; fread(&id, sizeof(int), 1, file); fread(&type, sizeof(int), 1, file);
        int name_len; fread(&name_len, sizeof(int), 1, file);
        char name_buf[256] = {0};
        if (name_len > 0 && name_len < 256) fread(name_buf, 1, (size_t)name_len, file);
        int desc_len; fread(&desc_len, sizeof(int), 1, file);
        char desc_buf[512] = {0};
        if (desc_len > 0 && desc_len < 512) fread(desc_buf, 1, (size_t)desc_len, file);
        size_t emb_sz; fread(&emb_sz, sizeof(size_t), 1, file);
        float* emb = NULL;
        if (emb_sz > 0 && emb_sz < 4096) {
            emb = (float*)safe_malloc(emb_sz * sizeof(float));
            if (emb) fread(emb, sizeof(float), emb_sz, file);
        }
        float spec, typ, conf; int inst;
        fread(&spec, sizeof(float), 1, file);
        fread(&typ, sizeof(float), 1, file);
        fread(&conf, sizeof(float), 1, file);
        fread(&inst, sizeof(int), 1, file);
        Concept* c = semantic_network_add_concept(net, (ConceptType)type, name_buf,
                                                   desc_len > 0 ? desc_buf : NULL,
                                                   emb, emb_sz, spec, typ);
        if (c) {
            c->confidence = conf;
            c->instance_count = inst;
            loaded_concepts[i] = c;
        } else {
            safe_free((void**)&emb);
        }
    }
    
    for (size_t i = 0; i < relation_count; i++) {
        int has; fread(&has, sizeof(int), 1, file);
        if (!has) continue;
        int id, type, src_id, tgt_id;
        fread(&id, sizeof(int), 1, file); fread(&type, sizeof(int), 1, file);
        fread(&src_id, sizeof(int), 1, file); fread(&tgt_id, sizeof(int), 1, file);
        int label_len; fread(&label_len, sizeof(int), 1, file);
        char label_buf[256] = {0};
        if (label_len > 0 && label_len < 256) fread(label_buf, 1, (size_t)label_len, file);
        float str, conf;
        fread(&str, sizeof(float), 1, file); fread(&conf, sizeof(float), 1, file);
        Concept* src = (src_id >= 0 && (size_t)src_id < concept_count) ? loaded_concepts[src_id] : NULL;
        Concept* tgt = (tgt_id >= 0 && (size_t)tgt_id < concept_count) ? loaded_concepts[tgt_id] : NULL;
        if (src && tgt) {
            semantic_network_add_relation(net, (SemanticRelationType)type,
                                          src, tgt,
                                          label_len > 0 ? label_buf : NULL,
                                          str, conf);
        }
    }
    
    net->next_concept_id = next_cid;
    net->next_relation_id = next_rid;
    
    safe_free((void**)&loaded_concepts);
    fclose(file);
    return net;
}

/* ============================================================================
 * ZSFWS-013修复: semantic_network_infer —— 语义网络推理
 * 头文件已声明但原.c缺失实现，导致knowledge_integration.c链接失败。
 * 基于扩散激活和关系传递机制实现多步推理。
 * Z-001修复: 更正API签名 —— rel->source_id → rel->source->id (SemanticRelation使用Concept*指针)
 * ============================================================================ */
SELFLNN_API size_t semantic_network_infer(SemanticNetwork* network,
                             Concept** premises, size_t premise_count,
                             size_t max_inferences,
                             Concept** results, size_t max_results) {
    if (!network || !premises || !results || max_results == 0) return 0;
    if (premise_count == 0) return 0;

    SEMANTIC_LOCK(network);
    size_t result_count = 0;

    int* visited = (int*)safe_calloc(network->concept_count, sizeof(int));
    if (!visited) { SEMANTIC_UNLOCK(network); return 0; }

    for (size_t p = 0; p < premise_count && result_count < max_results; p++) {
        Concept* premise = premises[p];
        if (!premise) continue;

        for (size_t r = 0; r < network->relation_count && result_count < max_results; r++) {
            SemanticRelation* rel = network->relations[r];
            if (!rel || !rel->source || !rel->target) continue;

            Concept* target_concept = NULL;
            if (rel->source == premise && rel->target != premise) {
                target_concept = rel->target;
            } else if (rel->target == premise && rel->source != premise) {
                target_concept = rel->source;
            }
            if (!target_concept) continue;

            size_t tgt_idx = semantic_network_find_concept_index(network, target_concept);
            if (tgt_idx == (size_t)-1 || tgt_idx >= network->concept_count) continue;
            if (visited[tgt_idx]) continue;

            results[result_count++] = target_concept;
            visited[tgt_idx] = 1;
        }

        if (result_count < max_results && result_count < max_inferences) {
            for (size_t r2 = 0; r2 < network->relation_count && result_count < max_results; r2++) {
                SemanticRelation* rel2 = network->relations[r2];
                if (!rel2 || rel2->type != RELATION_SYNONYM) continue;
                if (!rel2->source || !rel2->target) continue;

                size_t sid_idx = semantic_network_find_concept_index(network, rel2->source);
                size_t tid_idx = semantic_network_find_concept_index(network, rel2->target);
                if (sid_idx == (size_t)-1 || tid_idx == (size_t)-1) continue;
                if (tid_idx >= network->concept_count || sid_idx >= network->concept_count) continue;

                if (visited[sid_idx] && !visited[tid_idx]) {
                    if (rel2->target) {
                        results[result_count++] = rel2->target;
                        visited[tid_idx] = 1;
                    }
                }
            }
        }
    }

    safe_free((void**)&visited);
    SEMANTIC_UNLOCK(network);
    return result_count;
}

/* ============================================================================
 * ZSFWS-014修复: semantic_network_import_from_knowledge_base
 * 从知识库导入概念和关系到语义网络
 * Z-001修复: 更正所有API签名
 *   - semantic_network_add_concept 需要8个参数,返回Concept*
 *   - semantic_network_find_concept_by_name 替代不存在的semantic_network_find_concept
 *   - semantic_network_add_relation 需要Concept*源和目标指针
 *   - KnowledgeEntry字段: subject/predicate/object/type/confidence(枚举)/weight
 *   - kb->size 替代不存在的kb->entry_count
 * ============================================================================ */
SELFLNN_API int semantic_network_import_from_knowledge_base(SemanticNetwork* network,
                                               KnowledgeBase* kb) {
    if (!network || !kb) return -1;
    if (kb->size == 0) return 0;

    SEMANTIC_LOCK(network);
    int imported = 0;

    for (size_t i = 0; i < kb->size && i < 100000; i++) {
        KnowledgeEntry* entry = &kb->entries[i].entry;
        if (!entry->subject || entry->subject[0] == '\0') continue;

        float entry_conf = 0.3f;
        if (entry->confidence == CONFIDENCE_LOW)      entry_conf = 0.3f;
        else if (entry->confidence == CONFIDENCE_MEDIUM) entry_conf = 0.6f;
        else if (entry->confidence == CONFIDENCE_HIGH)   entry_conf = 0.9f;
        if (entry_conf < 0.1f) continue;

        ConceptType ctype = CONCEPT_TYPE_ENTITY;
        if (entry->type == KNOWLEDGE_CONCEPT)  ctype = CONCEPT_TYPE_CLASS;
        else if (entry->type == KNOWLEDGE_RULE)    ctype = CONCEPT_TYPE_EVENT;
        else if (entry->type == KNOWLEDGE_RELATION) ctype = CONCEPT_TYPE_RELATION;

        Concept* concept = semantic_network_add_concept(network, ctype,
            entry->subject, entry->object ? entry->object : "",
            entry->embedding, entry->embedding_size,
            0.5f, 0.5f);
        if (!concept) continue;

        if (entry->type == KNOWLEDGE_RELATION && entry->object && entry->object[0] != '\0') {
            Concept* other = semantic_network_find_concept_by_name(network, entry->object);
            if (other && other != concept) {
                SemanticRelation* rel = semantic_network_add_relation(network,
                    RELATION_SYNONYM, concept, other,
                    entry->predicate, entry_conf * 0.8f, entry_conf * 0.7f);
                if (rel) imported++;
            }
        }

        concept->confidence = entry_conf;
        imported++;
    }

    SEMANTIC_UNLOCK(network);
    return imported;
}

/* ============================================================================
 * ZSFWS-015修复: semantic_network_import_from_knowledge_graph
 * 从知识图谱导入节点和边到语义网络
 * Z-001修复: 更正所有API签名
 *   - graph->nodes和graph->edges通过SELFLNN_KNOWLEDGE_INTERNAL访问完整结构
 *   - edge->source → edge->source (GraphNode*指针), 移除不存在的edge->source_id
 *   - semantic_network_add_concept 需要8个参数,返回Concept*
 *   - semantic_network_add_relation 需要Concept*源和目标指针
 * ============================================================================ */
SELFLNN_API int semantic_network_import_from_knowledge_graph(SemanticNetwork* network,
                                               KnowledgeGraph* graph) {
    if (!network || !graph) return -1;
    if (graph->node_count == 0) return 0;

    SEMANTIC_LOCK(network);
    int imported = 0;

    Concept** concept_map = (Concept**)safe_calloc(graph->node_count, sizeof(Concept*));
    if (!concept_map) { SEMANTIC_UNLOCK(network); return -1; }

    for (size_t i = 0; i < graph->node_count && i < 100000; i++) {
        GraphNode* node = graph->nodes[i];
        if (!node || !node->label || node->label[0] == '\0') continue;

        ConceptType ctype = CONCEPT_TYPE_ENTITY;
        if (node->type == NODE_TYPE_CONCEPT)  ctype = CONCEPT_TYPE_CLASS;
        else if (node->type == NODE_TYPE_PROPERTY) ctype = CONCEPT_TYPE_PROPERTY;

        Concept* concept = semantic_network_add_concept(network, ctype,
            node->label, "",
            node->embedding, node->embedding_size,
            0.5f, node->confidence > 0.0f ? node->confidence : 0.5f);
        if (concept) {
            concept_map[i] = concept;
            imported++;
        }
    }

    for (size_t i = 0; i < graph->edge_count && i < 100000; i++) {
        GraphEdge* edge = graph->edges[i];
        if (!edge || !edge->source || !edge->target) continue;

        size_t src_idx = (size_t)-1, tgt_idx = (size_t)-1;
        for (size_t j = 0; j < graph->node_count; j++) {
            if (graph->nodes[j] == edge->source) src_idx = j;
            if (graph->nodes[j] == edge->target) tgt_idx = j;
        }
        if (src_idx == (size_t)-1 || tgt_idx == (size_t)-1) continue;
        if (src_idx >= graph->node_count || tgt_idx >= graph->node_count) continue;

        Concept* src_concept = concept_map[src_idx];
        Concept* tgt_concept = concept_map[tgt_idx];
        if (!src_concept || !tgt_concept) continue;

        SemanticRelationType rel_type = RELATION_RELATED_TO;
        if (edge->type == EDGE_TYPE_SUBCLASS)  rel_type = RELATION_IS_A;
        else if (edge->type == EDGE_TYPE_INSTANCE) rel_type = RELATION_INSTANCE_OF;
        else if (edge->type == EDGE_TYPE_PROPERTY) rel_type = RELATION_HAS_A;

        SemanticRelation* rel = semantic_network_add_relation(network,
            rel_type, src_concept, tgt_concept,
            edge->label,
            edge->weight > 0.0f ? edge->weight : 0.7f,
            edge->confidence > 0.0f ? edge->confidence : 0.7f);
        if (rel) imported++;
    }

    safe_free((void**)&concept_map);
    SEMANTIC_UNLOCK(network);
    return imported;
}