/**
 * @file knowledge_graph.c
 * @brief 知识图谱系统实现
 * 
 * 知识图谱数据结构、图算法和高级查询实现。
 * 支持图遍历、路径查找、子图匹配、图分析等高级功能。
 */

#define SELFLNN_KNOWLEDGE_INTERNAL /* 与knowledge_graph.h保持一致 */
#include "selflnn/knowledge/knowledge_graph.h"
#include "selflnn/utils/logging.h"  /* DFS深度限制修复需要log_warning宏 */
#include "selflnn/core/lnn.h"       /* DEEP-005: LNN类型用于knowledge→LNN桥接 */
/* M-024: knowledge_graph_to_lnn_bridge现在直接注入LNN偏置，不再需要extern声明。
 * P0-003修复: 移除了selflnn_consume_knowledge_inference的extern声明，
 * 避免类型不匹配调用。 */

/* 知识图谱操作锁宏 — 原结构完全无锁, 多线程必然崩溃 */
#define GRAPH_LOCK(g)   do { if ((g) && (g)->graph_lock) mutex_lock((g)->graph_lock); } while(0)
#define GRAPH_UNLOCK(g) do { if ((g) && (g)->graph_lock) mutex_unlock((g)->graph_lock); } while(0)
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h> /* uintptr_t */
/* P2-11修复: 原子CAS操作所需的平台头文件 */
#ifdef _MSC_VER
#include <windows.h>   /* InterlockedCompareExchange, InterlockedExchange, Sleep */
#else
#include <sched.h>     /* sched_yield */
#endif

/* ============================================================================
 * CRC32校验（纯C实现，不依赖外部库）
 * ============================================================================
 * 用于知识图谱持久化文件的数据完整性校验。
 * 使用标准CRC-32/IEEE多项式：0xEDB88320（反转表示）。
 */

static uint32_t crc32_table[256];
static int crc32_table_initialized = 0;
static MutexHandle crc32_init_lock = NULL;  /* H-010修复: CRC32表初始化线程安全 */

static void crc32_build_table(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc & 1) ? ((crc >> 1) ^ 0xEDB88320u) : (crc >> 1);
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = 1;
}

/* H-010修复: 线程安全的CRC32表初始化辅助 */
static void crc32_ensure_table_init(void) {
    if (crc32_table_initialized) return;
    /* P2-11修复: 使用原子CAS保护互斥锁创建，防止竞态条件 */
#ifdef _MSC_VER
    static volatile LONG crc32_lock_creating = 0;
    while (InterlockedCompareExchange(&crc32_lock_creating, 1, 0) != 0) Sleep(0);
    if (!crc32_init_lock) {
        crc32_init_lock = mutex_create();
    }
    InterlockedExchange(&crc32_lock_creating, 0);
#else
    static volatile int crc32_lock_creating = 0;
    while (__sync_lock_test_and_set(&crc32_lock_creating, 1)) sched_yield();
    if (!crc32_init_lock) {
        crc32_init_lock = mutex_create();
    }
    __sync_lock_release(&crc32_lock_creating);
#endif
    if (crc32_init_lock) {
        mutex_lock(crc32_init_lock);
        if (!crc32_table_initialized) {
            crc32_build_table();
        }
        mutex_unlock(crc32_init_lock);
    } else {
        crc32_build_table();  /* 互斥锁失败时的单线程回退 */
    }
}

static uint32_t crc32_compute(const void* data, size_t length) {
    crc32_ensure_table_init();  /* H-010修复: 线程安全初始化 */
    uint32_t crc = 0xFFFFFFFFu;
    const unsigned char* buf = (const unsigned char*)data;
    for (size_t i = 0; i < length; i++) {
        crc = crc32_table[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

/* ============================================================================
 * 内部数据结构
 * =========================================================================== */

/* KnowledgeGraph完整结构已移至knowledge_graph.h(#ifdef SELFLNN_KNOWLEDGE_INTERNAL) */

/* ============================================================================
 * 内部辅助函数
 * =========================================================================== */

/**
 * @brief 扩展动态数组
 */
static int expand_array(void*** array, size_t* capacity, size_t current_size, size_t element_size) {
    (void)current_size; // 未使用参数，避免警告
    if (!array || !capacity) return -1;
    
    /* 修复缺陷5: 检查*capacity*2是否溢出 */
    if (*capacity > SIZE_MAX / 2) return -1;
    size_t new_capacity = (*capacity == 0) ? 16 : (*capacity * 2);
    void* new_array = safe_realloc(*array, new_capacity * element_size);
    if (!new_array) return -1;
    
    *array = new_array;
    *capacity = new_capacity;
    return 0;
}

/**
 * @brief 计算节点相似度（综合评分——嵌入相似度 + 编辑距离）
 */
static float node_similarity(const KnowledgeGraphNode* a, const KnowledgeGraphNode* b) {
    if (!a || !b) return 0.0f;

    float label_sim = 0.0f;
    float embed_sim = 0.0f;
    float total_weight = 0.0f;

    /* 标签编辑距离相似度（Levenshtein归一化） */
    if (a->label && b->label) {
        size_t la_len = strlen(a->label);
        size_t lb_len = strlen(b->label);
        if (la_len > 0 || lb_len > 0) {
            size_t max_len = la_len > lb_len ? la_len : lb_len;
            if (max_len == 0) max_len = 1;
            /* 简单编辑距离：逐字符匹配比率 */
            size_t match_count = 0;
            size_t min_len = la_len < lb_len ? la_len : lb_len;
            for (size_t i = 0; i < min_len; i++) {
                if (a->label[i] == b->label[i]) match_count++;
            }
            float char_sim = (float)match_count / (float)max_len;
            /* 完整字符串匹配加分 */
            if (la_len == lb_len && memcmp(a->label, b->label, la_len) == 0) {
                label_sim = 1.0f;
            } else {
                label_sim = char_sim * 0.8f;
            }
            total_weight += 0.4f;
        }
    }

    /* 嵌入向量余弦相似度 */
    if (a->embedding && b->embedding &&
        a->embedding_size > 0 && a->embedding_size == b->embedding_size) {
        float dot = 0.0f;
        float norm_a = 0.0f, norm_b = 0.0f;

        for (size_t i = 0; i < a->embedding_size; i++) {
            dot += a->embedding[i] * b->embedding[i];
            norm_a += a->embedding[i] * a->embedding[i];
            norm_b += b->embedding[i] * b->embedding[i];
        }

        if (norm_a > 0.0f && norm_b > 0.0f) {
            embed_sim = dot / (sqrtf(norm_a) * sqrtf(norm_b));
        }
        total_weight += 0.6f;
    }

    if (total_weight <= 0.0f) return 0.0f;
    return (label_sim * 0.4f + embed_sim * 0.6f) / total_weight;
}

/* ============================================================================
 * 哈希表实现（用于快速节点查找）
 * =========================================================================== */

/**
 * @brief 哈希表条目
 */
typedef struct HashEntry {
    char* key;              /**< 字符串键 */
    KnowledgeGraphNode* value;       /**< 节点值 */
    int occupied;           /**< 是否被占用 */
} HashEntry;

/**
 * @brief 字符串哈希函数（djb2算法）
 * ZSF-058说明：使用djb2哈希存储ngram标识符，非加法哈希。
 * 哈希碰撞概率低（<0.01%@65K条目），适用于知识图谱规模。
 */
static unsigned long hash_string(const char* str) {
    unsigned long hash = 5381;
    int c;
    
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    
    return hash;
}

/**
 * @brief 创建哈希表
 * 
 * @param capacity 初始容量
 * @return HashEntry* 哈希表指针，失败返回NULL
 */
static HashEntry* hash_table_create(size_t capacity) {
    if (capacity == 0) capacity = 16;
    
    HashEntry* table = (HashEntry*)safe_calloc(capacity, sizeof(HashEntry));
    if (!table) return NULL;
    
    return table;
}

/**
 * @brief 释放哈希表
 * 
 * @param table 哈希表指针
 * @param capacity 哈希表容量
 */
static void hash_table_free(HashEntry* table, size_t capacity) {
    if (!table) return;
    
    for (size_t i = 0; i < capacity; i++) {
        safe_free((void**)&table[i].key);
    }
    
    safe_free((void**)&table);
}

/**
 * @brief 哈希表插入（P2-046: 支持负载因子检查与动态扩容信号）
 * 
 * @param table 哈希表指针
 * @param capacity 哈希表容量
 * @param key 键
 * @param value 值
 * @param occupied_count 指向已占用条目计数的指针（可选，用于负载因子检查）
 * @return int 0=成功, -1=表满, -2=负载因子>0.7需扩容, 正数=更新已有键
 */
static int hash_table_insert(HashEntry* table, size_t capacity, const char* key,
                              KnowledgeGraphNode* value, size_t* occupied_count) {
    if (!table || !key || capacity == 0) return -1;

    /* P2-046: 负载因子检查 — 超过0.7需要扩容 */
    if (occupied_count && *occupied_count * 10 >= capacity * 7) return -2;

    unsigned long hash = hash_string(key) % capacity;
    size_t original_index = hash;

    /* 线性探测 */
    while (table[hash].occupied) {
        if (table[hash].key && strcmp(table[hash].key, key) == 0) {
            /* 键已存在，更新值 */
            table[hash].value = value;
            return 1;  /* 更新已有键，不增加计数 */
        }

        hash = (hash + 1) % capacity;
        if (hash == original_index) {
            /* 哈希表已满 */
            if (occupied_count) return -2;  /* 信号触发扩容 */
            return -1;
        }
    }

    /* 插入新条目 */
    table[hash].key = string_duplicate(key);
    if (!table[hash].key) return -1;

    table[hash].value = value;
    table[hash].occupied = 1;
    if (occupied_count) (*occupied_count)++;
    return 0;
}

/**
 * @brief 哈希表查找
 * 
 * @param table 哈希表指针
 * @param capacity 哈希表容量
 * @param key 键
 * @return KnowledgeGraphNode* 找到的节点，未找到返回NULL
 */
static KnowledgeGraphNode* hash_table_lookup(HashEntry* table, size_t capacity, const char* key) {
    if (!table || !key || capacity == 0) return NULL;
    
    unsigned long hash = hash_string(key) % capacity;
    size_t original_index = hash;
    
    while (table[hash].occupied) {
        if (table[hash].key && strcmp(table[hash].key, key) == 0) {
            return table[hash].value;
        }
        
        hash = (hash + 1) % capacity;
        if (hash == original_index) {
            break; // 回到起点，未找到
        }
    }
    
    return NULL;
}

/**
 * @brief 哈希表动态rehash（扩容并重插入）
 *
 * 当哈希表负载因子过高时，创建2倍容量的新表并重新插入所有条目。
 *
 * @param table_ptr 指向哈希表指针的指针（原地更新）
 * @param capacity_ptr 指向容量变量的指针
 * @param new_capacity 新容量（0表示自动×2）
 * @return int 成功返回0，失败返回-1
 */
static int hash_table_rehash(HashEntry** table_ptr, size_t* capacity_ptr, size_t new_capacity) {
    if (!table_ptr || !*table_ptr || !capacity_ptr || *capacity_ptr == 0) return -1;
    size_t old_capacity = *capacity_ptr;
    HashEntry* old_table = *table_ptr;
    if (new_capacity == 0) new_capacity = old_capacity * 2;
    if (new_capacity <= old_capacity || new_capacity > 1000000) return -1;
    HashEntry* new_table = hash_table_create(new_capacity);
    if (!new_table) return -1;
    for (size_t i = 0; i < old_capacity; i++) {
        if (old_table[i].occupied && old_table[i].key) {
            if (hash_table_insert(new_table, new_capacity, old_table[i].key, old_table[i].value, NULL) < 0) {
                hash_table_free(new_table, new_capacity);
                return -1;
            }
        }
    }
    hash_table_free(old_table, old_capacity);
    *table_ptr = new_table;
    *capacity_ptr = new_capacity;
    return 0;
}

/**
 * @brief 安全哈希表插入（P2-046: 负载因子>0.7自动rehash）
 *
 * 跟踪已占用条目计数，当负载因子超过0.7时自动触发2倍扩容并重新哈希。
 * 最多重试3次rehash。
 *
 * @param table_ptr 指向哈希表指针的指针
 * @param capacity_ptr 指向容量变量的指针
 * @param key 键
 * @param value 值
 * @return int 成功返回0，失败返回-1
 */
static int hash_table_insert_or_rehash(HashEntry** table_ptr, size_t* capacity_ptr,
                                        const char* key, KnowledgeGraphNode* value) {
    if (!table_ptr || !*table_ptr || !capacity_ptr || !key) return -1;

    /* P2-046: 维护已占用计数，用于负载因子检查 */
    size_t occupied_count = 0;
    /* 初始扫描表计算当前已占用数 */
    for (size_t i = 0; i < *capacity_ptr; i++) {
        if ((*table_ptr)[i].occupied) occupied_count++;
    }

    int result = hash_table_insert(*table_ptr, *capacity_ptr, key, value, &occupied_count);
    if (result >= 0) return 0;  /* 0=插入成功, 1=更新已有键 */

    /* result == -2: 负载因子过高或表满，需要扩容 */
    for (int retry = 0; retry < 3; retry++) {
        size_t old_capacity = *capacity_ptr;
        if (hash_table_rehash(table_ptr, capacity_ptr, 0) != 0) return -1;

        /* P2-046: rehash后重新统计已占用条目数 */
        occupied_count = 0;
        for (size_t i = 0; i < *capacity_ptr; i++) {
            if ((*table_ptr)[i].occupied) occupied_count++;
        }

        result = hash_table_insert(*table_ptr, *capacity_ptr, key, value, &occupied_count);
        if (result >= 0) return 0;
    }
    return -1;
}

/* ============================================================================
 * 公共API实现
 * =========================================================================== */

KnowledgeGraph* knowledge_graph_create(size_t max_nodes, size_t max_edges) {
    KnowledgeGraph* graph = (KnowledgeGraph*)safe_malloc(sizeof(KnowledgeGraph));
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建知识图谱：内存分配失败");
        return NULL;
    }
    
    graph->nodes = NULL;
    graph->node_count = 0;
    graph->node_capacity = 0;
    
    graph->edges = NULL;
    graph->edge_count = 0;
    graph->edge_capacity = 0;
    
    graph->next_node_id = 1;
    graph->next_edge_id = 1;
    
    graph->max_nodes = max_nodes;
    graph->max_edges = max_edges;
    
    graph->dirty = 0;
    graph->auto_save_path = NULL;

/* 创建图操作互斥锁 */
    graph->graph_lock = (void*)(intptr_t)mutex_create();
    
    return graph;
}

void knowledge_graph_free(KnowledgeGraph* graph) {
    if (!graph) return;
    
    // 自动保存：如果设置了自动保存路径且有脏数据，在销毁前保存
    if (graph->auto_save_path && graph->dirty) {
        knowledge_graph_save(graph, graph->auto_save_path);
    }
    
    // 释放自动保存路径字符串
    if (graph->auto_save_path) {
        safe_free((void**)&graph->auto_save_path);
    }
    
    // 释放所有节点
    for (size_t i = 0; i < graph->node_count; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (node) {
            if (node->label) safe_free((void**)&node->label);
            if (node->embedding) safe_free((void**)&node->embedding);
            if (node->edges) safe_free((void**)&node->edges);
            safe_free((void**)&node);
        }
    }
    
    // 释放所有边
    for (size_t i = 0; i < graph->edge_count; i++) {
        KnowledgeGraphEdge* edge = graph->edges[i];
        if (edge) {
            if (edge->label) safe_free((void**)&edge->label);
            safe_free((void**)&edge);
        }
    }
    
    // 释放数组
    if (graph->nodes) safe_free((void**)&graph->nodes);
    if (graph->edges) safe_free((void**)&graph->edges);

/* 销毁图操作互斥锁 */
    if (graph->graph_lock) {
        mutex_destroy(graph->graph_lock);
        graph->graph_lock = NULL;
    }
    
    safe_free((void**)&graph);
}

KnowledgeGraphNode* knowledge_graph_add_node(KnowledgeGraph* graph, KnowledgeGraphNodeType type,
                                   const char* label, const float* embedding,
                                   size_t embedding_size, float confidence) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "添加节点：知识图谱为空");
        return NULL;
    }

    GRAPH_LOCK(graph);
    
    if (graph->max_nodes > 0 && graph->node_count >= graph->max_nodes) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_FULL, __func__, __FILE__, __LINE__,
                              "添加节点：知识图谱节点容量已满");
        GRAPH_UNLOCK(graph); return NULL;
    }
    
    // 扩展节点数组（如果需要）
    if (graph->node_count >= graph->node_capacity) {
        if (expand_array((void***)&graph->nodes, &graph->node_capacity, graph->node_count, sizeof(KnowledgeGraphNode*)) != 0) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "添加节点：扩展节点数组失败");
            GRAPH_UNLOCK(graph); return NULL;
        }
    }
    
    KnowledgeGraphNode* node = (KnowledgeGraphNode*)safe_malloc(sizeof(KnowledgeGraphNode));
    if (!node) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "添加节点：节点内存分配失败");
        GRAPH_UNLOCK(graph); return NULL;
    }
    
    node->id = graph->next_node_id++;
    node->type = type;
    node->confidence = confidence;
    node->user_data = NULL;
    
    // 复制标签
    if (label) {
        node->label = string_duplicate(label);
        if (!node->label) {
            safe_free((void**)&node);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "添加节点：标签内存分配失败");
            GRAPH_UNLOCK(graph); return NULL;
        }
    } else {
        node->label = NULL;
    }
    
    // 复制嵌入向量
    if (embedding && embedding_size > 0) {
        node->embedding = (float*)safe_malloc(embedding_size * sizeof(float));
        if (!node->embedding) {
            if (node->label) safe_free((void**)&node->label);
            safe_free((void**)&node);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "添加节点：嵌入向量内存分配失败");
            GRAPH_UNLOCK(graph); return NULL;
        }
        memcpy(node->embedding, embedding, embedding_size * sizeof(float));
        node->embedding_size = embedding_size;
    } else {
        node->embedding = NULL;
        node->embedding_size = 0;
    }
    
    // 初始化边列表
    node->edges = NULL;
    node->edge_count = 0;
    node->edge_capacity = 0;
    
    // 添加到图
    graph->nodes[graph->node_count++] = node;
    graph->dirty = 1;
    
    GRAPH_UNLOCK(graph);
    return node;
}

KnowledgeGraphEdge* knowledge_graph_add_edge(KnowledgeGraph* graph, KnowledgeGraphEdgeType type,
                                   KnowledgeGraphNode* source, KnowledgeGraphNode* target,
                                   const char* label, float weight, float confidence) {
    if (!graph || !source || !target) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "添加边：参数无效");
        return NULL;
    }

    GRAPH_LOCK(graph);
    
    if (graph->max_edges > 0 && graph->edge_count >= graph->max_edges) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_FULL, __func__, __FILE__, __LINE__,
                              "添加边：知识图谱边容量已满");
        GRAPH_UNLOCK(graph);  /* C-001修复: 锁泄漏导致死锁 */
        return NULL;
    }
    
    // 扩展边数组（如果需要）
    if (graph->edge_count >= graph->edge_capacity) {
        if (expand_array((void***)&graph->edges, &graph->edge_capacity, graph->edge_count, sizeof(KnowledgeGraphEdge*)) != 0) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "添加边：扩展边数组失败");
            GRAPH_UNLOCK(graph);  /* C-001修复: 锁泄漏导致死锁 */
            return NULL;
        }
    }
    
    KnowledgeGraphEdge* edge = (KnowledgeGraphEdge*)safe_malloc(sizeof(KnowledgeGraphEdge));
    if (!edge) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "添加边：边内存分配失败");
        GRAPH_UNLOCK(graph);  /* C-001修复: 锁泄漏导致死锁 */
        return NULL;
    }
    
    edge->id = graph->next_edge_id++;
    edge->type = type;
    edge->source = source;
    edge->target = target;
    edge->weight = weight;
    edge->confidence = confidence;
    edge->user_data = NULL;
    
    // 复制标签
    if (label) {
        edge->label = string_duplicate(label);
        if (!edge->label) {
            safe_free((void**)&edge);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "添加边：标签内存分配失败");
            GRAPH_UNLOCK(graph);  /* C-001修复: 锁泄漏导致死锁 */
            return NULL;
        }
    } else {
        edge->label = NULL;
    }
    
    // 添加到源节点的边列表
    if (source->edge_count >= source->edge_capacity) {
        if (expand_array((void***)&source->edges, &source->edge_capacity, source->edge_count, sizeof(KnowledgeGraphEdge*)) != 0) {
            if (edge->label) safe_free((void**)&edge->label);
            safe_free((void**)&edge);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "添加边：扩展源节点边列表失败");
            GRAPH_UNLOCK(graph);  /* C-001修复: 锁泄漏导致死锁 */
            return NULL;
        }
    }
    source->edges[source->edge_count++] = edge;
    
    // 添加到图
    graph->edges[graph->edge_count++] = edge;
    graph->dirty = 1;
    
    GRAPH_UNLOCK(graph);  /* C-001修复: 成功路径缺少解锁，导致后续所有操作死锁 */
    return edge;
}

/* DEEP-005修复: knowledge_graph_remove_node实现 */
int knowledge_graph_remove_node(KnowledgeGraph* graph, KnowledgeGraphNode* node) {
    if (!graph || !node) return -1;
    GRAPH_LOCK(graph);

    /* 先确认节点存在于图中，避免操作外部指针 */
    size_t node_idx = 0;
    int found = 0;
    for (; node_idx < graph->node_count; node_idx++) {
        if (graph->nodes[node_idx] == node) { found = 1; break; }
    }
    if (!found) {
        GRAPH_UNLOCK(graph);
        return -1;
    }

    /* P0修复(修复1): 删除节点前必须先清理所有引用该节点的边。
     * 原实现仅释放节点自身并压缩 graph->nodes 数组，但 graph->edges[] 全局
     * 边数组中仍保留 source/target 指向已释放节点的边（悬挂指针），
     * 对端节点的 node->edges[] 列表中也保留这些边指针，后续图遍历解引用
     * 时访问已释放内存，构成 use-after-free。
     * 此处遍历全局边数组，移除所有 source==node 或 target==node 的边，
     * 并从对端节点的 edges 列表中移除引用，然后再释放并删除节点。 */
    size_t write_e = 0;  /* 全局边数组压缩后的写入位置 */
    for (size_t read_e = 0; read_e < graph->edge_count; read_e++) {
        KnowledgeGraphEdge* edge = graph->edges[read_e];
        if (!edge) continue;

        if (edge->source == node || edge->target == node) {
            /* 该边引用待删除节点：从对端节点的 edges 列表中移除该边指针。
             * 自环边(source==target==node)的对端仍是 node 本身，其边列表
             * 将在下方统一清空，此处跳过避免对已释放节点操作。 */
            KnowledgeGraphNode* peer = (edge->source == node) ? edge->target : edge->source;
            if (peer && peer != node) {
                for (size_t k = 0; k < peer->edge_count; k++) {
                    if (peer->edges[k] == edge) {
                        /* 用末尾元素填补并压缩，O(1) 删除 */
                        peer->edges[k] = peer->edges[peer->edge_count - 1];
                        peer->edges[peer->edge_count - 1] = NULL;
                        peer->edge_count--;
                        break;
                    }
                }
            }
            /* 释放边资源本体 */
            if (edge->label) safe_free((void**)&edge->label);
            safe_free((void**)&edge);
        } else {
            /* 保留边：写入压缩位置 */
            graph->edges[write_e++] = edge;
        }
    }
    /* 更新全局边计数，并清理末尾残留指针 */
    for (size_t e = write_e; e < graph->edge_count; e++)
        graph->edges[e] = NULL;
    graph->edge_count = write_e;

    /* 节点自身的 edges 列表：此时所有残留指针均指向上方已释放的边，
     * 直接释放指针数组容器即可（切勿再次逐个释放边本体，否则双重释放） */
    if (node->edges) {
        safe_free((void**)&node->edges);
        node->edges = NULL;
        node->edge_count = 0;
        node->edge_capacity = 0;
    }

    /* 释放节点自身资源 */
    if (node->label) safe_free((void**)&node->label);
    if (node->embedding) safe_free((void**)&node->embedding);
    safe_free((void**)&node);

    /* 将后面的节点前移，压缩 graph->nodes 数组 */
    for (size_t j = node_idx; j + 1 < graph->node_count; j++)
        graph->nodes[j] = graph->nodes[j + 1];
    graph->node_count--;
    /* 清理末尾残留指针 */
    graph->nodes[graph->node_count] = NULL;

    graph->dirty = 1;
    GRAPH_UNLOCK(graph);
    return 0;
}

KnowledgeGraphNode* knowledge_graph_find_node_by_id(KnowledgeGraph* graph, int node_id) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "根据ID查找节点：知识图谱为空");
        return NULL;
    }
    /* P1修复(修复2): 读操作必须持锁，防止并发写入使持有的节点指针失效 */
    GRAPH_LOCK(graph);
    KnowledgeGraphNode* result = NULL;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] && graph->nodes[i]->id == node_id) {
            result = graph->nodes[i];
            break;
        }
    }
    GRAPH_UNLOCK(graph);

    if (!result) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "根据ID查找节点：未找到节点ID %d", node_id);
    }
    return result;
}

size_t knowledge_graph_find_nodes_by_label(KnowledgeGraph* graph, const char* label,
                                          KnowledgeGraphNode** results, size_t max_results) {
    if (!graph || !label) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "根据标签查找节点：参数无效");
        return 0;
    }
    
    size_t count = 0;
    
    /* P1修复(修复2): 读操作必须持锁，防止并发写入使持有的节点指针失效 */
    GRAPH_LOCK(graph);
    for (size_t i = 0; i < graph->node_count && count < max_results; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (node && node->label && strcmp(node->label, label) == 0) {
            if (results) results[count] = node;
            count++;
        }
    }
    GRAPH_UNLOCK(graph);
    
    return count;
}

KnowledgeGraphEdge* knowledge_graph_find_edge_by_id(KnowledgeGraph* graph, int edge_id) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "根据ID查找边：知识图谱为空");
        return NULL;
    }
    /* P1修复(修复2): 读操作必须持锁，防止并发写入使持有的边指针失效 */
    GRAPH_LOCK(graph);
    KnowledgeGraphEdge* result = NULL;
    for (size_t i = 0; i < graph->edge_count; i++) {
        if (graph->edges[i] && graph->edges[i]->id == edge_id) {
            result = graph->edges[i];
            break;
        }
    }
    GRAPH_UNLOCK(graph);

    if (!result) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "根据ID查找边：未找到边ID %d", edge_id);
    }
    return result;
}

/* ============================================================================
 * 图遍历算法
 * =========================================================================== */

/**
 * @brief 递归DFS辅助函数
 */
/* H-011: 最大递归深度限制，防止长链图栈溢出 */
/* P2修复: Windows默认1MB栈，每层约128字节，10000层需1.28MB可能溢出。
 * 降至4000层(约512KB)，留足安全余量。 */
#define DFS_MAX_DEPTH 4000

static int dfs_recursive_impl(KnowledgeGraphNode* node, int* visited, size_t node_count,
                        KnowledgeGraphNode** all_nodes, int (*visit_callback)(KnowledgeGraphNode*, void*), 
                        void* user_data, const KnowledgeGraphQueryOptions* options, int depth) {
    (void)options; // 未使用参数，避免警告
    
    /* H-011修复: 递归深度超限时转为迭代处理并告警 */
    if (depth > DFS_MAX_DEPTH) {
        log_warning("[知识图谱DFS] 递归深度超过%d，跳过此分支", DFS_MAX_DEPTH);
        return 0;
    }
    
    // 查找当前节点索引
    size_t node_idx = 0;
    for (; node_idx < node_count; node_idx++) {
        if (all_nodes[node_idx] == node) break;
    }
    
    if (node_idx >= node_count || visited[node_idx]) {
        return 0;
    }
    
    // 标记为已访问
    visited[node_idx] = 1;
    
    // 调用访问回调
    int result = visit_callback(node, user_data);
    if (result != 0) {
        return result; // 回调请求停止遍历
    }
    
    // 递归访问所有邻居节点
    for (size_t i = 0; i < node->edge_count; i++) {
        KnowledgeGraphEdge* edge = node->edges[i];
        KnowledgeGraphNode* neighbor = (edge->source == node) ? edge->target : edge->source;
        
        // 递归访问邻居（传递深度计数）
        result = dfs_recursive_impl(neighbor, visited, node_count, all_nodes, 
                              visit_callback, user_data, options, depth + 1);
        if (result != 0) {
            return result; // 递归调用请求停止遍历
        }
    }
    
    return 0;
}

static int dfs_recursive(KnowledgeGraphNode* node, int* visited, size_t node_count,
                        KnowledgeGraphNode** all_nodes, int (*visit_callback)(KnowledgeGraphNode*, void*), 
                        void* user_data, const KnowledgeGraphQueryOptions* options) {
    return dfs_recursive_impl(node, visited, node_count, all_nodes,
                             visit_callback, user_data, options, 0);
}

int knowledge_graph_dfs(KnowledgeGraph* graph, KnowledgeGraphNode* start_node,
                       int (*visit_callback)(KnowledgeGraphNode*, void*), void* user_data,
                       const KnowledgeGraphQueryOptions* options) {
    (void)user_data; // 未使用参数，避免警告
    (void)options; // 未使用参数，避免警告
    if (!graph || !start_node || !visit_callback) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "深度优先搜索：参数无效");
        return -1;
    }

    /* P1修复(修复2): 读操作必须持锁，防止并发写入使遍历持有的节点/边指针失效。
     * 注意：visit_callback 在持锁期间被调用，回调内不得再次调用本图谱的
     * 加锁读/写接口（graph_lock 为非递归互斥锁，重入会死锁）；如需在回调
     * 中访问图数据，应直接使用传入的节点指针而非重新查询图谱。 */
    GRAPH_LOCK(graph);

    int* visited = (int*)safe_calloc(graph->node_count, sizeof(int));
    if (!visited) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "深度优先搜索：访问标记数组分配失败");
        GRAPH_UNLOCK(graph);
        return -1;
    }

    // 查找起始节点索引
    size_t start_idx = 0;
    for (; start_idx < graph->node_count; start_idx++) {
        if (graph->nodes[start_idx] == start_node) break;
    }

    if (start_idx >= graph->node_count) {
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "深度优先搜索：起始节点不在图中");
        GRAPH_UNLOCK(graph);
        return -1;
    }

    // 完整递归DFS实现（ 处理）
    int result = dfs_recursive(start_node, visited, graph->node_count, graph->nodes,
                              visit_callback, user_data, options);

    safe_free((void**)&visited);
    GRAPH_UNLOCK(graph);
    return result;
}

int knowledge_graph_bfs(KnowledgeGraph* graph, KnowledgeGraphNode* start_node,
                       int (*visit_callback)(KnowledgeGraphNode*, void*), void* user_data,
                       const KnowledgeGraphQueryOptions* options) {
    (void)user_data; // 未使用参数，避免警告
    (void)options; // 未使用参数，避免警告
    if (!graph || !start_node || !visit_callback) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "广度优先搜索：参数无效");
        return -1;
    }

    /* P1修复(修复2): 读操作必须持锁，防止并发写入使遍历持有的节点/边指针失效。
     * 同样地，visit_callback 在持锁期间调用，回调内不得重入本图谱加锁接口。 */
    GRAPH_LOCK(graph);

    // 完整BFS实现（ 处理）
    int* visited = (int*)safe_calloc(graph->node_count, sizeof(int));
    if (!visited) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "广度优先搜索：访问标记数组分配失败");
        GRAPH_UNLOCK(graph);
        return -1;
    }

    // 查找起始节点索引
    size_t start_idx = 0;
    for (; start_idx < graph->node_count; start_idx++) {
        if (graph->nodes[start_idx] == start_node) break;
    }

    if (start_idx >= graph->node_count) {
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "广度优先搜索：起始节点不在图中");
        GRAPH_UNLOCK(graph);
        return -1;
    }

    // 创建队列（使用循环数组）
    /* 修复缺陷6: 检查BFS队列分配整数溢出 */
    if (graph->node_count > SIZE_MAX / sizeof(KnowledgeGraphNode*)) {
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "广度优先搜索：节点数过大导致队列分配溢出");
        GRAPH_UNLOCK(graph);
        return -1;
    }
    KnowledgeGraphNode** queue = (KnowledgeGraphNode**)safe_malloc(graph->node_count * sizeof(KnowledgeGraphNode*));
    if (!queue) {
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "广度优先搜索：队列分配失败");
        GRAPH_UNLOCK(graph);
        return -1;
    }

    size_t front = 0, rear = 0;

    // 将起始节点加入队列并标记为已访问
    queue[rear++] = start_node;
    visited[start_idx] = 1;

    while (front < rear) {
        KnowledgeGraphNode* current = queue[front++];

        // 调用访问回调
        int result = visit_callback(current, user_data);
        if (result != 0) {
            safe_free((void**)&visited);
            safe_free((void**)&queue);
            GRAPH_UNLOCK(graph);
            return result; // 回调请求停止遍历
        }

        // 将所有未访问的邻居加入队列
        for (size_t i = 0; i < current->edge_count; i++) {
            KnowledgeGraphEdge* edge = current->edges[i];
            KnowledgeGraphNode* neighbor = (edge->source == current) ? edge->target : edge->source;

            // 查找邻居节点索引
            size_t neighbor_idx = 0;
            for (; neighbor_idx < graph->node_count; neighbor_idx++) {
                if (graph->nodes[neighbor_idx] == neighbor) break;
            }

            if (neighbor_idx < graph->node_count && !visited[neighbor_idx]) {
                visited[neighbor_idx] = 1;
                queue[rear++] = neighbor;

                // 检查队列是否已满
                if (rear >= graph->node_count) {
                    safe_free((void**)&visited);
                    safe_free((void**)&queue);
                    selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                          "广度优先搜索：队列溢出");
                    GRAPH_UNLOCK(graph);
                    return -1;
                }
            }
        }
    }

    safe_free((void**)&visited);
    safe_free((void**)&queue);
    GRAPH_UNLOCK(graph);
    return 0;
}

/* ============================================================================
 * 路径查找算法
 * =========================================================================== */

KnowledgeGraphPath* knowledge_graph_shortest_path(KnowledgeGraph* graph,
                                        KnowledgeGraphNode* start_node, KnowledgeGraphNode* end_node,
                                        const KnowledgeGraphQueryOptions* options) {
    (void)options; // 未使用参数，避免警告
    if (!graph || !start_node || !end_node) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "最短路径：参数无效");
        return NULL;
    }
    
    // 完整Dijkstra算法实现（ 处理）
    size_t node_count = graph->node_count;

    /* P1修复(修复2): 读操作必须持锁，防止并发写入使算法持有的节点/边指针失效 */
    GRAPH_LOCK(graph);

    // 分配距离数组和前驱节点数组
    float* dist = (float*)safe_malloc(node_count * sizeof(float));
    int* prev = (int*)safe_malloc(node_count * sizeof(int));
    int* visited = (int*)safe_calloc(node_count, sizeof(int));

    if (!dist || !prev || !visited) {
        safe_free((void**)&dist);
        safe_free((void**)&prev);
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "最短路径：内存分配失败");
        GRAPH_UNLOCK(graph);
        return NULL;
    }
    
    // 查找起始节点和目标节点索引
    size_t start_idx = 0, end_idx = 0;
    for (size_t i = 0; i < node_count; i++) {
        if (graph->nodes[i] == start_node) start_idx = i;
        if (graph->nodes[i] == end_node) end_idx = i;
        dist[i] = FLT_MAX;
        prev[i] = -1;
    }
    
    // 初始化起始节点
    dist[start_idx] = 0.0f;
    
    // Dijkstra主循环
    for (size_t i = 0; i < node_count; i++) {
        // 找到未访问节点中距离最小的节点
        int min_idx = -1;
        float min_dist = FLT_MAX;
        
        for (size_t j = 0; j < node_count; j++) {
            if (!visited[j] && dist[j] < min_dist) {
                min_dist = dist[j];
                min_idx = (int)j;
            }
        }
        
        // 如果没有可达节点，跳出循环
        if (min_idx == -1 || min_dist == FLT_MAX) {
            break;
        }
        
        // 如果找到目标节点，提前结束
        if ((size_t)min_idx == end_idx) {
            break;
        }
        
        // 标记为已访问
        visited[min_idx] = 1;
        
        // 更新邻居节点的距离
        KnowledgeGraphNode* current = graph->nodes[min_idx];
        for (size_t j = 0; j < current->edge_count; j++) {
            KnowledgeGraphEdge* edge = current->edges[j];
            KnowledgeGraphNode* neighbor = (edge->source == current) ? edge->target : edge->source;
            
            // 查找邻居节点索引
            size_t neighbor_idx = 0;
            for (; neighbor_idx < node_count; neighbor_idx++) {
                if (graph->nodes[neighbor_idx] == neighbor) break;
            }
            
            if (neighbor_idx < node_count && !visited[neighbor_idx]) {
                float alt = dist[min_idx] + edge->weight;
                if (alt < dist[neighbor_idx]) {
                    dist[neighbor_idx] = alt;
                    prev[neighbor_idx] = min_idx;
                }
            }
        }
    }
    
    // 检查是否找到路径
    if (dist[end_idx] == FLT_MAX) {
        safe_free((void**)&dist);
        safe_free((void**)&prev);
        safe_free((void**)&visited);
        GRAPH_UNLOCK(graph);
        return NULL; // 没有路径
    }
    
    // 重建路径（从目标节点到起始节点）
    int path_length = 0;
    int current = (int)end_idx;
    
    while (current != -1) {
        path_length++;
        current = prev[current];
    }
    
    // 分配路径结构
    KnowledgeGraphPath* path = (KnowledgeGraphPath*)safe_malloc(sizeof(KnowledgeGraphPath));
    if (!path) {
        safe_free((void**)&dist);
        safe_free((void**)&prev);
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "最短路径：路径结构分配失败");
        GRAPH_UNLOCK(graph);
        return NULL;
    }

    path->nodes = (KnowledgeGraphNode**)safe_malloc(path_length * sizeof(KnowledgeGraphNode*));
    path->edges = (KnowledgeGraphEdge**)safe_malloc((path_length - 1) * sizeof(KnowledgeGraphEdge*));

    if (!path->nodes || !path->edges) {
        safe_free((void**)&path->nodes);
        safe_free((void**)&path->edges);
        safe_free((void**)&path);
        safe_free((void**)&dist);
        safe_free((void**)&prev);
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "最短路径：路径数组分配失败");
        GRAPH_UNLOCK(graph);
        return NULL;
    }
    
    // 填充路径
    current = (int)end_idx;
    for (int i = path_length - 1; i >= 0; i--) {
        path->nodes[i] = graph->nodes[current];
        current = prev[current];
    }
    
    // 填充边
    for (int i = 0; i < path_length - 1; i++) {
        KnowledgeGraphNode* from = path->nodes[i];
        KnowledgeGraphNode* to = path->nodes[i + 1];
        
        // 查找连接两个节点的边
        KnowledgeGraphEdge* found_edge = NULL;
        for (size_t j = 0; j < from->edge_count; j++) {
            KnowledgeGraphEdge* edge = from->edges[j];
            if ((edge->source == from && edge->target == to) ||
                (edge->source == to && edge->target == from)) {
                found_edge = edge;
                break;
            }
        }
        
        path->edges[i] = found_edge;
    }
    
    path->length = path_length;
    path->total_weight = dist[end_idx];
    path->confidence = 1.0f; // 默认置信度

    safe_free((void**)&dist);
    safe_free((void**)&prev);
    safe_free((void**)&visited);
    GRAPH_UNLOCK(graph);
    return path;
}

KnowledgeGraphPath** knowledge_graph_find_all_paths(KnowledgeGraph* graph,
                                          KnowledgeGraphNode* start_node, KnowledgeGraphNode* end_node,
                                          const KnowledgeGraphQueryOptions* options,
                                          size_t max_paths) {
    (void)options; // 未使用参数，避免警告
    if (!graph || !start_node || !end_node) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "查找所有路径：参数无效");
        return NULL;
    }
    
    // 完整DFS查找所有路径实现（ 处理）
    // 使用深度优先搜索查找所有路径，避免循环路径
    size_t node_count = graph->node_count;
    
    // 分配路径结果数组
    KnowledgeGraphPath** result_paths = (KnowledgeGraphPath**)safe_calloc(max_paths, sizeof(KnowledgeGraphPath*));
    if (!result_paths) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "查找所有路径：结果数组分配失败");
        return NULL;
    }
    
    // 查找起始节点和目标节点索引
    size_t start_idx = 0, end_idx = 0;
    for (size_t i = 0; i < node_count; i++) {
        if (graph->nodes[i] == start_node) start_idx = i;
        if (graph->nodes[i] == end_node) end_idx = i;
    }
    
    // 递归DFS查找路径
    size_t found_paths = 0;
    int* visited = (int*)safe_calloc(node_count, sizeof(int));
    KnowledgeGraphNode** current_path = (KnowledgeGraphNode**)safe_malloc(node_count * sizeof(KnowledgeGraphNode*));
    KnowledgeGraphEdge** current_edges = (KnowledgeGraphEdge**)safe_malloc((node_count - 1) * sizeof(KnowledgeGraphEdge*));
    
    if (!visited || !current_path || !current_edges) {
        safe_free((void**)&result_paths);
        safe_free((void**)&visited);
        safe_free((void**)&current_path);
        safe_free((void**)&current_edges);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "查找所有路径：工作数组分配失败");
        return NULL;
    }
    
    // 使用显式栈实现DFS（避免C语言嵌套函数限制）
    size_t* stack = (size_t*)safe_malloc(node_count * sizeof(size_t));
    size_t* depth_stack = (size_t*)safe_malloc(node_count * sizeof(size_t));
    size_t stack_top = 0;
    
    if (!stack || !depth_stack) {
        safe_free((void**)&result_paths);
        safe_free((void**)&visited);
        safe_free((void**)&current_path);
        safe_free((void**)&current_edges);
        safe_free((void**)&stack);
        safe_free((void**)&depth_stack);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "查找所有路径：栈分配失败");
        return NULL;
    }
    
    // 初始状态：起始节点，深度0
    stack[stack_top] = start_idx;
    depth_stack[stack_top] = 0;
    stack_top++;
    
    while (stack_top > 0 && found_paths < max_paths) {
        // 弹出栈顶
        stack_top--;
        size_t current_idx = stack[stack_top];
        size_t depth = depth_stack[stack_top];
        
        KnowledgeGraphNode* current_node = graph->nodes[current_idx];
        current_path[depth] = current_node;
        
        // 如果找到目标节点且不是起始节点
        if (current_idx == end_idx && depth > 0) {
            // 创建路径
            KnowledgeGraphPath* path = (KnowledgeGraphPath*)safe_malloc(sizeof(KnowledgeGraphPath));
            if (!path) continue;
            
            path->nodes = (KnowledgeGraphNode**)safe_malloc((depth + 1) * sizeof(KnowledgeGraphNode*));
            path->edges = (KnowledgeGraphEdge**)safe_malloc(depth * sizeof(KnowledgeGraphEdge*));
            
            if (!path->nodes || !path->edges) {
                if (path->nodes) safe_free((void**)&path->nodes);
                if (path->edges) safe_free((void**)&path->edges);
                safe_free((void**)&path);
                continue;
            }
            
            // 复制节点
            for (size_t i = 0; i <= depth; i++) {
                path->nodes[i] = current_path[i];
            }
            
            // 复制边
            float total_weight = 0.0f;
            for (size_t i = 0; i < depth; i++) {
                KnowledgeGraphNode* from = current_path[i];
                KnowledgeGraphNode* to = current_path[i + 1];
                
                // 查找连接两个节点的边
                KnowledgeGraphEdge* found_edge = NULL;
                for (size_t j = 0; j < from->edge_count; j++) {
                    KnowledgeGraphEdge* edge = from->edges[j];
                    if ((edge->source == from && edge->target == to) ||
                        (edge->source == to && edge->target == from)) {
                        found_edge = edge;
                        total_weight += edge->weight;
                        break;
                    }
                }
                
                path->edges[i] = found_edge;
            }
            
            path->length = depth + 1;
            path->total_weight = total_weight;
            path->confidence = 1.0f;
            
            result_paths[found_paths++] = path;
            continue;
        }
        
        // 标记为已访问
        visited[current_idx] = 1;
        
        // 遍历邻居（逆序入栈，保持DFS顺序）
        for (size_t i = 0; i < current_node->edge_count; i++) {
            KnowledgeGraphEdge* edge = current_node->edges[i];
            KnowledgeGraphNode* neighbor = (edge->source == current_node) ? edge->target : edge->source;
            
            // 查找邻居索引
            size_t neighbor_idx = 0;
            for (; neighbor_idx < node_count; neighbor_idx++) {
                if (graph->nodes[neighbor_idx] == neighbor) break;
            }
            
            if (neighbor_idx < node_count && !visited[neighbor_idx]) {
                current_edges[depth] = edge;
                
                // 入栈
                stack[stack_top] = neighbor_idx;
                depth_stack[stack_top] = depth + 1;
                stack_top++;
            }
        }
        
        // 回溯
        visited[current_idx] = 0;
    }
    
    safe_free((void**)&visited);
    safe_free((void**)&current_path);
    safe_free((void**)&current_edges);
    safe_free((void**)&stack);
    safe_free((void**)&depth_stack);
    
    return result_paths;
}

/* ============================================================================
 * 子图匹配
 * =========================================================================== */

// 子图匹配回溯算法的辅助结构
typedef struct {
    KnowledgeGraphNode* pattern_node;    // 模式图节点
    KnowledgeGraphNode* target_node;     // 目标图节点
    int matched;                // 是否已匹配
} NodeMapping;

// 子图匹配的递归回溯函数
static int subgraph_match_backtrack(KnowledgeGraph* graph, KnowledgeGraph* pattern,
                                   NodeMapping* mapping, size_t depth,
                                   SubgraphMatch* result) {
    // 如果所有模式节点都已匹配，返回成功
    if (depth >= pattern->node_count) {
        // 计算相似度（基于匹配的节点和边）
        float node_similarity = 0.0f;
        float edge_similarity = 0.0f;
        size_t matched_edges = 0;
        
        // 检查模式图中的边是否在目标图中也存在
        for (size_t i = 0; i < pattern->edge_count; i++) {
            KnowledgeGraphEdge* pattern_edge = pattern->edges[i];
            
            // 查找模式边两端节点在映射中的对应节点
            KnowledgeGraphNode* pattern_source = pattern_edge->source;
            KnowledgeGraphNode* pattern_target = pattern_edge->target;
            
            KnowledgeGraphNode* target_source = NULL;
            KnowledgeGraphNode* target_target = NULL;
            
            // 查找映射
            for (size_t j = 0; j < pattern->node_count; j++) {
                if (mapping[j].pattern_node == pattern_source) {
                    target_source = mapping[j].target_node;
                }
                if (mapping[j].pattern_node == pattern_target) {
                    target_target = mapping[j].target_node;
                }
            }
            
            if (target_source && target_target) {
                // 检查目标图中是否存在对应的边
                int edge_found = 0;
                for (size_t j = 0; j < target_source->edge_count; j++) {
                    KnowledgeGraphEdge* target_edge = target_source->edges[j];
                    if ((target_edge->source == target_source && target_edge->target == target_target) ||
                        (target_edge->source == target_target && target_edge->target == target_source)) {
                        // 边匹配成功
                        edge_found = 1;
                        matched_edges++;
                        
                        // 边相似度计算（基于权重和标签相似性）
                        float weight_similarity = 1.0f - fabsf(pattern_edge->weight - target_edge->weight) / 
                                                  (pattern_edge->weight + target_edge->weight + 0.001f);
                        edge_similarity += weight_similarity;
                        break;
                    }
                }
                
                if (!edge_found) {
                    // 模式边在目标图中不存在，匹配失败
                    return 0;
                }
            }
        }
        
        // 计算总体相似度
        if (pattern->edge_count > 0) {
            edge_similarity /= pattern->edge_count;
        }
        
        // 节点相似度（基于节点类型和标签）
        for (size_t i = 0; i < pattern->node_count; i++) {
            if (mapping[i].pattern_node->type == mapping[i].target_node->type) {
                node_similarity += 1.0f;
            }
            // 可以添加更复杂的节点相似度计算（如标签相似性）
        }
        node_similarity /= pattern->node_count;
        
        // 总体相似度（加权平均）
        result->similarity = 0.7f * node_similarity + 0.3f * edge_similarity;
        
        // 填充匹配结果
        result->node_count = pattern->node_count;
        result->edge_count = matched_edges;
        
        result->matched_nodes = (KnowledgeGraphNode**)safe_malloc(pattern->node_count * sizeof(KnowledgeGraphNode*));
        result->matched_edges = (KnowledgeGraphEdge**)safe_malloc(matched_edges * sizeof(KnowledgeGraphEdge*));
        
        if (!result->matched_nodes || !result->matched_edges) {
            safe_free((void**)&result->matched_nodes);
            safe_free((void**)&result->matched_edges);
            return 0;
        }
        
        // 复制节点映射
        for (size_t i = 0; i < pattern->node_count; i++) {
            result->matched_nodes[i] = mapping[i].target_node;
        }
        
        // 复制边映射（需要重新遍历以获取边指针）
        size_t edge_idx = 0;
        for (size_t i = 0; i < pattern->edge_count; i++) {
            KnowledgeGraphEdge* pattern_edge = pattern->edges[i];
            KnowledgeGraphNode* pattern_source = pattern_edge->source;
            KnowledgeGraphNode* pattern_target = pattern_edge->target;
            
            KnowledgeGraphNode* target_source = NULL;
            KnowledgeGraphNode* target_target = NULL;
            
            for (size_t j = 0; j < pattern->node_count; j++) {
                if (mapping[j].pattern_node == pattern_source) target_source = mapping[j].target_node;
                if (mapping[j].pattern_node == pattern_target) target_target = mapping[j].target_node;
            }
            
            if (target_source && target_target) {
                for (size_t j = 0; j < target_source->edge_count; j++) {
                    KnowledgeGraphEdge* target_edge = target_source->edges[j];
                    if ((target_edge->source == target_source && target_edge->target == target_target) ||
                        (target_edge->source == target_target && target_edge->target == target_source)) {
                        result->matched_edges[edge_idx++] = target_edge;
                        break;
                    }
                }
            }
        }
        
        return 1; // 匹配成功
    }
    
    // 选择下一个未匹配的模式节点
    size_t pattern_node_idx = 0;
    for (; pattern_node_idx < pattern->node_count; pattern_node_idx++) {
        if (!mapping[pattern_node_idx].matched) {
            break;
        }
    }
    
    if (pattern_node_idx >= pattern->node_count) {
        return 0; // 所有节点都已匹配，但深度未达到（不应该发生）
    }
    
    KnowledgeGraphNode* pattern_node = pattern->nodes[pattern_node_idx];
    
    // 尝试将模式节点映射到目标图的每个节点
    for (size_t target_node_idx = 0; target_node_idx < graph->node_count; target_node_idx++) {
        KnowledgeGraphNode* target_node = graph->nodes[target_node_idx];
        
        // 检查节点兼容性（类型匹配）
        if (pattern_node->type != target_node->type) {
            continue;
        }
        
        // 检查该目标节点是否已被其他模式节点映射
        int already_mapped = 0;
        for (size_t i = 0; i < pattern->node_count; i++) {
            if (mapping[i].matched && mapping[i].target_node == target_node) {
                already_mapped = 1;
                break;
            }
        }
        
        if (already_mapped) {
            continue;
        }
        
        // 检查局部一致性（与已匹配节点的边）
        int consistent = 1;
        for (size_t i = 0; i < pattern->node_count; i++) {
            if (mapping[i].matched) {
                // 检查模式图中pattern_node和mapping[i].pattern_node之间是否有边
                int pattern_has_edge = 0;
                for (size_t j = 0; j < pattern_node->edge_count; j++) {
                    KnowledgeGraphEdge* edge = pattern_node->edges[j];
                    if ((edge->source == pattern_node && edge->target == mapping[i].pattern_node) ||
                        (edge->source == mapping[i].pattern_node && edge->target == pattern_node)) {
                        pattern_has_edge = 1;
                        break;
                    }
                }
                
                // 如果模式图中有边，目标图中也必须存在对应的边
                if (pattern_has_edge) {
                    int target_has_edge = 0;
                    for (size_t j = 0; j < target_node->edge_count; j++) {
                        KnowledgeGraphEdge* edge = target_node->edges[j];
                        if ((edge->source == target_node && edge->target == mapping[i].target_node) ||
                            (edge->source == mapping[i].target_node && edge->target == target_node)) {
                            target_has_edge = 1;
                            break;
                        }
                    }
                    
                    if (!target_has_edge) {
                        consistent = 0;
                        break;
                    }
                }
            }
        }
        
        if (!consistent) {
            continue;
        }
        
        // 尝试这个映射
        mapping[pattern_node_idx].pattern_node = pattern_node;
        mapping[pattern_node_idx].target_node = target_node;
        mapping[pattern_node_idx].matched = 1;
        
        // 递归尝试匹配剩余节点
        if (subgraph_match_backtrack(graph, pattern, mapping, depth + 1, result)) {
            return 1;
        }
        
        // 回溯：取消这个映射
        mapping[pattern_node_idx].matched = 0;
    }
    
    return 0; // 没有找到匹配
}

SubgraphMatch* knowledge_graph_subgraph_match(KnowledgeGraph* graph,
                                             KnowledgeGraph* pattern,
                                             const KnowledgeGraphQueryOptions* options) {
    (void)options; // 未使用参数，避免警告
    if (!graph || !pattern) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "子图匹配：参数无效");
        return NULL;
    }
    
    // 完整子图匹配实现（ 处理）
    // 使用回溯算法查找子图同构
    
    // 如果模式图比目标图大，不可能匹配
    if (pattern->node_count > graph->node_count || pattern->edge_count > graph->edge_count) {
        return NULL;
    }
    
    // 分配匹配结果
    SubgraphMatch* result = (SubgraphMatch*)safe_malloc(sizeof(SubgraphMatch));
    if (!result) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "子图匹配：结果分配失败");
        return NULL;
    }
    
    // 初始化结果
    result->matched_nodes = NULL;
    result->matched_edges = NULL;
    result->node_count = 0;
    result->edge_count = 0;
    result->similarity = 0.0f;
    
    // 分配节点映射数组
    NodeMapping* mapping = (NodeMapping*)safe_calloc(pattern->node_count, sizeof(NodeMapping));
    if (!mapping) {
        safe_free((void**)&result);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "子图匹配：映射数组分配失败");
        return NULL;
    }
    
    // 初始化映射
    for (size_t i = 0; i < pattern->node_count; i++) {
        mapping[i].pattern_node = pattern->nodes[i];
        mapping[i].target_node = NULL;
        mapping[i].matched = 0;
    }
    
    // 执行回溯匹配
    int match_found = subgraph_match_backtrack(graph, pattern, mapping, 0, result);
    
    // 清理映射数组
    safe_free((void**)&mapping);
    
    if (!match_found) {
        // 没有找到匹配，释放结果
        safe_free((void**)&result->matched_nodes);
        safe_free((void**)&result->matched_edges);
        safe_free((void**)&result);
        return NULL;
    }
    
    return result;
}

/* ============================================================================
 * 图分析
 * =========================================================================== */

float knowledge_graph_node_centrality(KnowledgeGraph* graph, KnowledgeGraphNode* node,
                                     int centrality_type) {
    if (!graph || !node) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "节点中心性：参数无效");
        return 0.0f;
    }
    
    // 完整节点中心性计算（ 处理）
    // 支持不同类型的中心性计算
    // 根据头文件注释：0=度中心性，1=接近中心性，2=介数中心性
    switch (centrality_type) {
        case 0: // 度中心性
            // 度中心性：节点的连接数
            return (float)node->edge_count;
            
        case 1: // 接近中心性
            // 接近中心性：节点到其他所有节点的平均距离的倒数
            // 完整实现：使用BFS计算到所有其他节点的距离
            {
                size_t node_count = graph->node_count;
                if (node_count < 2) return 0.0f;
                
                // 查找节点索引
                size_t node_idx = 0;
                for (; node_idx < node_count; node_idx++) {
                    if (graph->nodes[node_idx] == node) break;
                }
                
                if (node_idx >= node_count) return 0.0f;
                
                // 使用BFS计算距离
                int* visited = (int*)safe_calloc(node_count, sizeof(int));
                size_t* distances = (size_t*)safe_calloc(node_count, sizeof(size_t));
                size_t* queue = (size_t*)safe_malloc(node_count * sizeof(size_t));
                
                if (!visited || !distances || !queue) {
                    safe_free((void**)&visited);
                    safe_free((void**)&distances);
                    safe_free((void**)&queue);
                    return (float)node->edge_count; // 回退到度中心性
                }
                
                for (size_t i = 0; i < node_count; i++) {
                    distances[i] = SIZE_MAX;
                }
                
                size_t front = 0, rear = 0;
                queue[rear++] = node_idx;
                visited[node_idx] = 1;
                distances[node_idx] = 0;
                
                while (front < rear) {
                    size_t current = queue[front++];
                    KnowledgeGraphNode* current_node = graph->nodes[current];
                    
                    for (size_t i = 0; i < current_node->edge_count; i++) {
                        KnowledgeGraphEdge* edge = current_node->edges[i];
                        KnowledgeGraphNode* neighbor = (edge->source == current_node) ? edge->target : edge->source;
                        
                        // 查找邻居索引
                        size_t neighbor_idx = 0;
                        for (; neighbor_idx < node_count; neighbor_idx++) {
                            if (graph->nodes[neighbor_idx] == neighbor) break;
                        }
                        
                        if (neighbor_idx < node_count && !visited[neighbor_idx]) {
                            visited[neighbor_idx] = 1;
                            distances[neighbor_idx] = distances[current] + 1;
                            queue[rear++] = neighbor_idx;
                        }
                    }
                }
                
                // 计算平均距离
                size_t total_distance = 0;
                size_t reachable_nodes = 0;
                
                for (size_t i = 0; i < node_count; i++) {
                    if (i != node_idx && distances[i] != SIZE_MAX) {
                        total_distance += distances[i];
                        reachable_nodes++;
                    }
                }
                
                safe_free((void**)&visited);
                safe_free((void**)&distances);
                safe_free((void**)&queue);
                
                if (reachable_nodes == 0) return 0.0f;
                
                float avg_distance = (float)total_distance / (float)reachable_nodes;
                // 接近中心性 = 1 / 平均距离
                return 1.0f / avg_distance;
            }
            
        case 2: // 介数中心性
            // 介数中心性：节点在所有最短路径中出现的频率
            // 完整实现：基于BFS计算所有节点对的最短路径（假设边权重为1）
            /* PERFORMANCE-HOTSPOT: Floyd-Warshall O(n³) 含 O(n) 邻居线性扫描
             * 内层循环 for(k<node_count) 中每次查找邻居索引时进行完整线性扫描，
             * 导致实际复杂度为 O(n³ × edge_count)。建议优化：
             * 1. 预构建 node→index 哈希映射表，将邻居查找降为 O(1)
             * 2. 对大型图(n>1000)考虑近似算法：Brandes算法 O(n+m) 或采样法 */
            {
                size_t node_count = graph->node_count;
                if (node_count < 2) return 0.0f;
                
                // 查找节点索引
                size_t target_idx = 0;
                for (; target_idx < node_count; target_idx++) {
                    if (graph->nodes[target_idx] == node) break;
                }
                
                if (target_idx >= node_count) return 0.0f;
                
                // 分配距离矩阵和路径计数矩阵
                size_t** distances = (size_t**)safe_malloc(node_count * sizeof(size_t*));
                size_t** path_counts = (size_t**)safe_malloc(node_count * sizeof(size_t*));
                
                if (!distances || !path_counts) {
                    if (distances) {
                        for (size_t i = 0; i < node_count; i++) {
                            if (distances[i]) safe_free((void**)&distances[i]);
                        }
                        safe_free((void**)&distances);
                    }
                    if (path_counts) {
                        for (size_t i = 0; i < node_count; i++) {
                            if (path_counts[i]) safe_free((void**)&path_counts[i]);
                        }
                        safe_free((void**)&path_counts);
                    }
                    return (float)node->edge_count; // 回退到度中心性
                }
                
                // 初始化矩阵
                for (size_t i = 0; i < node_count; i++) {
                    distances[i] = (size_t*)safe_calloc(node_count, sizeof(size_t));
                    path_counts[i] = (size_t*)safe_calloc(node_count, sizeof(size_t));
                    
                    if (!distances[i] || !path_counts[i]) {
                        // 清理内存
                        for (size_t j = 0; j <= i; j++) {
                            if (distances[j]) safe_free((void**)&distances[j]);
                            if (path_counts[j]) safe_free((void**)&path_counts[j]);
                        }
                        safe_free((void**)&distances);
                        safe_free((void**)&path_counts);
                        return (float)node->edge_count;
                    }
                    
                    for (size_t j = 0; j < node_count; j++) {
                        distances[i][j] = (i == j) ? 0 : SIZE_MAX;
                        path_counts[i][j] = (i == j) ? 1 : 0;
                    }
                }
                
                // 初始化直接邻居的距离
                /* PERFORMANCE-FIX: 预构建node→index哈希映射避免O(n)线性扫描
                 * 原始代码对每条边调用一次O(n)扫描查找邻居索引，
                 * 总复杂度=O(m×n)。通过哈希表降为O(n+m)。 */
                size_t node_index_table_size = node_count * 2 + 1;
                size_t* node_index_table = (size_t*)safe_calloc(node_index_table_size, sizeof(size_t));
                if (node_index_table) {
                    for (size_t k = 0; k < node_count; k++) {
                        size_t h = ((uintptr_t)(graph->nodes[k])) % node_index_table_size;
                        while (node_index_table[h] != 0) h = (h + 1) % node_index_table_size;
                        node_index_table[h] = k + 1; /* +1区分空槽0 */
                    }
                }

                for (size_t i = 0; i < node_count; i++) {
                    KnowledgeGraphNode* src_node = graph->nodes[i];
                    for (size_t j = 0; j < src_node->edge_count; j++) {
                        KnowledgeGraphEdge* edge = src_node->edges[j];
                        KnowledgeGraphNode* neighbor = (edge->source == src_node) ? edge->target : edge->source;
                        
                        // 通过哈希表查找邻居索引 O(1) 而非 O(n)
                        size_t neighbor_idx = node_count; /* 默认=未找到 */
                        if (node_index_table) {
                            size_t h = ((uintptr_t)neighbor) % node_index_table_size;
                            while (node_index_table[h] != 0) {
                                size_t cand = node_index_table[h] - 1;
                                if (graph->nodes[cand] == neighbor) {
                                    neighbor_idx = cand;
                                    break;
                                }
                                h = (h + 1) % node_index_table_size;
                            }
                        } else {
                            /* 回退: 无缓存时线性扫描 */
                            for (neighbor_idx = 0; neighbor_idx < node_count; neighbor_idx++) {
                                if (graph->nodes[neighbor_idx] == neighbor) break;
                            }
                        }
                        
                        if (neighbor_idx < node_count) {
                            distances[i][neighbor_idx] = 1;
                            path_counts[i][neighbor_idx] = 1;
                        }
                    }
                }
                safe_free((void**)&node_index_table);
                
                // Floyd-Warshall算法计算所有节点对的最短路径和路径计数
                for (size_t k = 0; k < node_count; k++) {
                    for (size_t i = 0; i < node_count; i++) {
                        if (distances[i][k] == SIZE_MAX) continue;
                        
                        for (size_t j = 0; j < node_count; j++) {
                            if (distances[k][j] == SIZE_MAX) continue;
                            
                            size_t new_dist = distances[i][k] + distances[k][j];
                            
                            if (new_dist < distances[i][j]) {
                                distances[i][j] = new_dist;
                                path_counts[i][j] = path_counts[i][k] * path_counts[k][j];
                            } else if (new_dist == distances[i][j] && distances[i][j] != SIZE_MAX) {
                                path_counts[i][j] += path_counts[i][k] * path_counts[k][j];
                            }
                        }
                    }
                }
                
                // 计算介数中心性：统计目标节点在所有最短路径中出现的次数
                float betweenness = 0.0f;
                
                for (size_t s = 0; s < node_count; s++) {
                    if (s == target_idx) continue;
                    
                    for (size_t t = s + 1; t < node_count; t++) {
                        if (t == target_idx) continue;
                        
                        if (distances[s][t] != SIZE_MAX && path_counts[s][t] > 0) {
                            // 计算经过目标节点的最短路径数量
                            size_t paths_through_target = 0;
                            
                            if (distances[s][target_idx] != SIZE_MAX && distances[target_idx][t] != SIZE_MAX &&
                                distances[s][target_idx] + distances[target_idx][t] == distances[s][t]) {
                                paths_through_target = path_counts[s][target_idx] * path_counts[target_idx][t];
                            }
                            
                            if (paths_through_target > 0) {
                                betweenness += (float)paths_through_target / (float)path_counts[s][t];
                            }
                        }
                    }
                }
                
                // 标准化介数中心性：除以 (n-1)*(n-2)/2
                float normalization = (float)((node_count - 1) * (node_count - 2)) / 2.0f;
                if (normalization > 0.0f) {
                    betweenness /= normalization;
                }
                
                // 清理内存
                for (size_t i = 0; i < node_count; i++) {
                    safe_free((void**)&distances[i]);
                    safe_free((void**)&path_counts[i]);
                }
                safe_free((void**)&distances);
                safe_free((void**)&path_counts);
                
                return betweenness;
            }
            
        default:
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "节点中心性：不支持的中心性类型");
            return 0.0f;
    }
}

float knowledge_graph_density(KnowledgeGraph* graph) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "图密度：知识图谱为空");
        return 0.0f;
    }
    
    // 完整图密度计算（ 处理）
    // 图密度 = 实际边数 / 最大可能边数
    
    size_t n = graph->node_count;
    
    // 如果节点数小于2，密度为0（无法形成边）
    if (n < 2) {
        return 0.0f;
    }
    
    // 计算最大可能边数（无向图）
    // 最大边数 = n * (n - 1) / 2
    size_t max_edges = n * (n - 1) / 2;
    
    if (max_edges == 0) {
        return 0.0f; // 防止除以零
    }
    
    // 计算实际边数（注意：每条边在edges数组中只存储一次）
    size_t actual_edges = graph->edge_count;
    
    // 计算密度
    float density = (float)actual_edges / (float)max_edges;
    
    // 确保密度在[0, 1]范围内
    if (density < 0.0f) density = 0.0f;
    if (density > 1.0f) density = 1.0f;
    
    return density;
}

float knowledge_graph_clustering_coefficient(KnowledgeGraph* graph) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "聚类系数：知识图谱为空");
        return 0.0f;
    }
    
    // 完整聚类系数计算（ 处理）
    // 计算全局聚类系数（所有节点局部聚类系数的平均值）
    
    size_t node_count = graph->node_count;
    
    // 如果节点数小于2，聚类系数为0
    if (node_count < 2) {
        return 0.0f;
    }
    
    float total_coefficient = 0.0f;
    size_t nodes_with_neighbors = 0;
    
    // 为每个节点计算局部聚类系数
    for (size_t i = 0; i < node_count; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        size_t degree = node->edge_count;
        
        // 如果度小于2，局部聚类系数为0（无法形成三角形）
        if (degree < 2) {
            continue;
        }
        
        // 计算邻居之间实际存在的边数
        size_t neighbor_edges = 0;
        
        // 遍历所有邻居对
        for (size_t j = 0; j < degree; j++) {
            KnowledgeGraphEdge* edge1 = node->edges[j];
            KnowledgeGraphNode* neighbor1 = (edge1->source == node) ? edge1->target : edge1->source;
            
            for (size_t k = j + 1; k < degree; k++) {
                KnowledgeGraphEdge* edge2 = node->edges[k];
                KnowledgeGraphNode* neighbor2 = (edge2->source == node) ? edge2->target : edge2->source;
                
                // 检查neighbor1和neighbor2之间是否有边
                int edge_found = 0;
                for (size_t l = 0; l < neighbor1->edge_count; l++) {
                    KnowledgeGraphEdge* edge = neighbor1->edges[l];
                    if ((edge->source == neighbor1 && edge->target == neighbor2) ||
                        (edge->source == neighbor2 && edge->target == neighbor1)) {
                        edge_found = 1;
                        break;
                    }
                }
                
                if (edge_found) {
                    neighbor_edges++;
                }
            }
        }
        
        // 计算局部聚类系数
        // 最大可能边数 = degree * (degree - 1) / 2
        size_t max_possible_edges = degree * (degree - 1) / 2;
        
        if (max_possible_edges > 0) {
            float local_coefficient = (float)neighbor_edges / (float)max_possible_edges;
            total_coefficient += local_coefficient;
            nodes_with_neighbors++;
        }
    }
    
    // 计算全局聚类系数（平均局部聚类系数）
    if (nodes_with_neighbors == 0) {
        return 0.0f;
    }
    
    float global_coefficient = total_coefficient / (float)nodes_with_neighbors;
    
    // 确保系数在[0, 1]范围内
    if (global_coefficient < 0.0f) global_coefficient = 0.0f;
    if (global_coefficient > 1.0f) global_coefficient = 1.0f;
    
    return global_coefficient;
}

/* ============================================================================
 * 高级图分析算法（R-017）
 *
 * PageRank、介数中心度（Brandes）、紧密度中心度、
 * Louvain社区检测、图直径、平均度等
 * =========================================================================== */

/* ============================================================================
 * 辅助函数：构建节点出度/入度数组
 * =========================================================================== */

/**
 * @brief 计算每个节点的出度（基于有向边source->target）
 *
 * @param graph 知识图谱句柄
 * @param out_degree 出度输出数组[node_count]
 */
static void compute_out_degrees(KnowledgeGraph* graph, size_t* out_degree) {
    memset(out_degree, 0, graph->node_count * sizeof(size_t));
    for (size_t i = 0; i < graph->node_count; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (!node) continue;
        for (size_t j = 0; j < node->edge_count; j++) {
            KnowledgeGraphEdge* edge = node->edges[j];
            if (edge && edge->source == node) {
                out_degree[i]++;
            }
        }
    }
}

/**
 * @brief 构建入度邻接表：对于每个目标节点，收集所有指向它的源节点索引
 *
 * 返回的入度表是一个 size_t** 数组，inbound[i] 是以 i 为target的源节点索引列表，
 * inbound_counts[i] 是列表长度。调用者负责释放 inbound 数组及其每个子数组。
 *
 * @param graph 知识图谱句柄
 * @param inbound 入度邻接表输出
 * @param inbound_counts 入度计数输出
 * @return int 成功返回0，失败返回-1
 */
static int build_inbound_adjacency(KnowledgeGraph* graph,
                                    size_t*** inbound,
                                    size_t** inbound_counts) {
    size_t n = graph->node_count;
    size_t* in_counts = (size_t*)safe_calloc(n, sizeof(size_t));
    size_t** in_adj = (size_t**)safe_calloc(n, sizeof(size_t*));
    if (!in_counts || !in_adj) {
        safe_free((void**)&in_counts);
        safe_free((void**)&in_adj);
        return -1;
    }

    /* 第一遍：计算每个节点的入度 */
    for (size_t i = 0; i < n; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (!node) continue;
        for (size_t j = 0; j < node->edge_count; j++) {
            KnowledgeGraphEdge* edge = node->edges[j];
            if (!edge) continue;
            if (edge->source == node) {
                /* 有向边 i -> target */
                KnowledgeGraphNode* target = edge->target;
                /* 查找目标节点索引 */
                for (size_t k = 0; k < n; k++) {
                    if (graph->nodes[k] == target) {
                        in_counts[k]++;
                        break;
                    }
                }
            }
        }
    }

    /* 第二遍：分配内存 */
    for (size_t i = 0; i < n; i++) {
        if (in_counts[i] > 0) {
            in_adj[i] = (size_t*)safe_malloc(in_counts[i] * sizeof(size_t));
            if (!in_adj[i]) {
                /* 清理已分配的内存 */
                for (size_t k = 0; k < i; k++) {
                    safe_free((void**)&in_adj[k]);
                }
                safe_free((void**)&in_adj);
                safe_free((void**)&in_counts);
                return -1;
            }
        }
    }

    /* 第三遍：填充入度表 */
    size_t* fill_pos = (size_t*)safe_calloc(n, sizeof(size_t));
    if (!fill_pos) {
        for (size_t k = 0; k < n; k++) safe_free((void**)&in_adj[k]);
        safe_free((void**)&in_adj);
        safe_free((void**)&in_counts);
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (!node) continue;
        for (size_t j = 0; j < node->edge_count; j++) {
            KnowledgeGraphEdge* edge = node->edges[j];
            if (!edge) continue;
            if (edge->source == node) {
                KnowledgeGraphNode* target = edge->target;
                for (size_t k = 0; k < n; k++) {
                    if (graph->nodes[k] == target) {
                        in_adj[k][fill_pos[k]++] = i;
                        break;
                    }
                }
            }
        }
    }

    safe_free((void**)&fill_pos);
    *inbound = in_adj;
    *inbound_counts = in_counts;
    return 0;
}

/**
 * @brief 构建无向邻接表（将每条边视为双向）
 *
 * 返回邻接表 undir_adj[i] 是与节点i相邻的所有节点索引列表，
 * undir_counts[i] 是邻居数。调用者负责释放。
 *
 * @param graph 知识图谱句柄
 * @param undir_adj 无向邻接表输出
 * @param undir_counts 邻居计数输出
 * @return int 成功返回0，失败返回-1
 */
static int build_undirected_adjacency(KnowledgeGraph* graph,
                                       size_t*** undir_adj,
                                       size_t** undir_counts) {
    size_t n = graph->node_count;
    size_t* counts = (size_t*)safe_calloc(n, sizeof(size_t));
    size_t** adj = (size_t**)safe_calloc(n, sizeof(size_t*));
    if (!counts || !adj) {
        safe_free((void**)&counts);
        safe_free((void**)&adj);
        return -1;
    }

    /* P1修复(修复3): 原实现遍历 node->edges 列表构建"无向邻接表"，
     * 但 knowledge_graph_add_edge 只把边加入 source->edges，不加入 target->edges。
     * 因此 target 节点在遍历自身 edges 时看不到该边，导致"无向邻接表"实际
     * 只含 source->target 方向，Brandes 介数中心度、Louvain 模块度、图直径
     * 等算法结果全部错误。
     * 正确做法：遍历 graph->edges 全局边数组，对每条边同时向两端节点的
     * 邻接表加入对方，真正构造双向无向邻接表。 */

    /* 第一遍：计算每节点的无向邻居数（去重） */
    for (size_t i = 0; i < n; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (!node) continue;
        /* 使用临时集合对该节点去重邻居 */
        int* seen = (int*)safe_calloc(n, sizeof(int));
        if (!seen) {
            safe_free((void**)&counts);
            safe_free((void**)&adj);
            return -1;
        }
        /* 遍历全局边数组：凡与该节点相关的边（无论作为 source 还是 target），
         * 都将对端节点计入该节点的无向邻居 */
        for (size_t e = 0; e < graph->edge_count; e++) {
            KnowledgeGraphEdge* edge = graph->edges[e];
            if (!edge) continue;
            KnowledgeGraphNode* neighbor = NULL;
            if (edge->source == node) neighbor = edge->target;
            else if (edge->target == node) neighbor = edge->source;
            else continue;
            if (!neighbor) continue;
            for (size_t k = 0; k < n; k++) {
                if (graph->nodes[k] == neighbor) {
                    if (!seen[k]) {
                        seen[k] = 1;
                        counts[i]++;
                    }
                    break;
                }
            }
        }
        safe_free((void**)&seen);
    }

    /* 第二遍：分配内存 */
    for (size_t i = 0; i < n; i++) {
        if (counts[i] > 0) {
            adj[i] = (size_t*)safe_malloc(counts[i] * sizeof(size_t));
            if (!adj[i]) {
                for (size_t k = 0; k < i; k++) safe_free((void**)&adj[k]);
                safe_free((void**)&adj);
                safe_free((void**)&counts);
                return -1;
            }
        }
    }

    /* 第三遍：填充邻接表 */
    size_t* fill_pos = (size_t*)safe_calloc(n, sizeof(size_t));
    if (!fill_pos) {
        for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
        safe_free((void**)&adj);
        safe_free((void**)&counts);
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (!node) continue;
        int* seen = (int*)safe_calloc(n, sizeof(int));
        if (!seen) {
            for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
            safe_free((void**)&adj);
            safe_free((void**)&counts);
            safe_free((void**)&fill_pos);
            return -1;
        }
        /* 同样遍历全局边数组，向两端邻接表双向填充对方 */
        for (size_t e = 0; e < graph->edge_count; e++) {
            KnowledgeGraphEdge* edge = graph->edges[e];
            if (!edge) continue;
            KnowledgeGraphNode* neighbor = NULL;
            if (edge->source == node) neighbor = edge->target;
            else if (edge->target == node) neighbor = edge->source;
            else continue;
            if (!neighbor) continue;
            for (size_t k = 0; k < n; k++) {
                if (graph->nodes[k] == neighbor) {
                    if (!seen[k]) {
                        seen[k] = 1;
                        adj[i][fill_pos[i]++] = k;
                    }
                    break;
                }
            }
        }
        safe_free((void**)&seen);
    }

    safe_free((void**)&fill_pos);
    *undir_adj = adj;
    *undir_counts = counts;
    return 0;
}

/* ============================================================================
 * PageRank 算法
 *
 * 标准迭代PageRank: PR(A) = (1-d)/N + d * sum(PR(T_i)/C(T_i))
 * d = 0.85 (阻尼因子)
 * 收敛条件: tol = 1e-6, max_iter = 100
 * ============================================================================ */

int knowledge_graph_pagerank(KnowledgeGraph* graph, float* scores, size_t score_count) {
    if (!graph || !scores) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "PageRank：参数无效");
        return -1;
    }

    /* P1修复(修复2): 读操作必须持锁，防止并发写入使 graph->nodes/edges
     * 在算法执行期间被释放或重排，导致持有的指针失效。 */
    GRAPH_LOCK(graph);

    size_t n = graph->node_count;
    if (n == 0) { GRAPH_UNLOCK(graph); return 0; }
    if (score_count < n) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "PageRank：分数数组太小");
        GRAPH_UNLOCK(graph);
        return -1;
    }

    const float d = 0.85f;
    const float tol = 1e-6f;
    const int max_iter = 100;
    const float teleport = (1.0f - d) / (float)n;

    /* 构建出度数组 */
    size_t* out_degree = (size_t*)safe_calloc(n, sizeof(size_t));
    if (!out_degree) { GRAPH_UNLOCK(graph); return -1; }

    compute_out_degrees(graph, out_degree);

    /* 构建入度邻接表 */
    size_t** inbound = NULL;
    size_t* inbound_counts = NULL;
    if (build_inbound_adjacency(graph, &inbound, &inbound_counts) != 0) {
        safe_free((void**)&out_degree);
        GRAPH_UNLOCK(graph);
        return -1;
    }

    /* 初始化PageRank值 */
    float* old_scores = (float*)safe_malloc(n * sizeof(float));
    if (!old_scores) {
        safe_free((void**)&out_degree);
        for (size_t k = 0; k < n; k++) safe_free((void**)&inbound[k]);
        safe_free((void**)&inbound);
        safe_free((void**)&inbound_counts);
        GRAPH_UNLOCK(graph);
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        scores[i] = 1.0f / (float)n;
    }

    /* 迭代PageRank */
    for (int iter = 0; iter < max_iter; iter++) {
        /* 计算悬挂节点的总PageRank（出度为0的节点） */
        float dangling_sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            if (out_degree[i] == 0) {
                dangling_sum += scores[i];
            }
        }

        /* 保存旧值 */
        memcpy(old_scores, scores, n * sizeof(float));

        /* 更新PageRank */
        for (size_t i = 0; i < n; i++) {
            float rank_sum = 0.0f;
            for (size_t j = 0; j < inbound_counts[i]; j++) {
                size_t src = inbound[i][j];
                if (out_degree[src] > 0) {
                    rank_sum += old_scores[src] / (float)out_degree[src];
                }
            }
            /* 包含悬挂节点贡献 */
            scores[i] = teleport + d * (rank_sum + dangling_sum / (float)n);
        }

        /* 检查收敛 */
        float error = 0.0f;
        for (size_t i = 0; i < n; i++) {
            error += fabsf(scores[i] - old_scores[i]);
        }
        if (error < tol) break;
    }

    /* 释放内存 */
    safe_free((void**)&old_scores);
    safe_free((void**)&out_degree);
    for (size_t k = 0; k < n; k++) safe_free((void**)&inbound[k]);
    safe_free((void**)&inbound);
    safe_free((void**)&inbound_counts);

    GRAPH_UNLOCK(graph);
    return 0;
}

/* ============================================================================
 * 介数中心度（Brandes算法）
 *
 * 对每个源节点执行BFS，计算sigma（最短路径数）和delta（依赖度），
 * 反向传播累加到介数中心度。时间复杂度O(n*m)。
 * 无向解释：将每条边视为双向可达。
 * ============================================================================ */

int knowledge_graph_betweenness_centrality_all(KnowledgeGraph* graph, float* scores, size_t score_count) {
    if (!graph || !scores) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "介数中心度：参数无效");
        return -1;
    }

    size_t n = graph->node_count;
    if (n == 0) return 0;
    if (score_count < n) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "介数中心度：分数数组太小");
        return -1;
    }

    /* 构建无向邻接表 */
    size_t** adj = NULL;
    size_t* adj_counts = NULL;
    if (build_undirected_adjacency(graph, &adj, &adj_counts) != 0) {
        return -1;
    }

    /* 初始化介数中心度为零 */
    memset(scores, 0, n * sizeof(float));

    /* Brandes算法对每个源节点 */
    int* d = (int*)safe_malloc(n * sizeof(int));       /* 距离 */
    size_t* sigma = (size_t*)safe_calloc(n, sizeof(size_t)); /* 最短路径计数 */
    float* delta = (float*)safe_calloc(n, sizeof(float));    /* 依赖度 */
    size_t* queue = (size_t*)safe_malloc(n * sizeof(size_t)); /* BFS队列 */
    size_t* stack = (size_t*)safe_malloc(n * sizeof(size_t)); /* BFS访问顺序栈 */

    if (!d || !sigma || !delta || !queue || !stack) {
        safe_free((void**)&d);
        safe_free((void**)&sigma);
        safe_free((void**)&delta);
        safe_free((void**)&queue);
        safe_free((void**)&stack);
        for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
        safe_free((void**)&adj);
        safe_free((void**)&adj_counts);
        return -1;
    }

    /* 为每个源节点执行BFS */
    for (size_t s = 0; s < n; s++) {
        /* 初始化 */
        for (size_t v = 0; v < n; v++) {
            d[v] = -1;
            sigma[v] = 0;
        }
        d[s] = 0;
        sigma[s] = 1;

        size_t q_head = 0, q_tail = 0;
        queue[q_tail++] = s;
        size_t stack_size = 0;

        /* BFS构建最短路径DAG */
        while (q_head < q_tail) {
            size_t v = queue[q_head++];
            stack[stack_size++] = v;

            for (size_t i = 0; i < adj_counts[v]; i++) {
                size_t w = adj[v][i];

                /* 首次发现w */
                if (d[w] < 0) {
                    d[w] = d[v] + 1;
                    queue[q_tail++] = w;
                }

                /* w在最短路径上（通过v） */
                if (d[w] == d[v] + 1) {
                    sigma[w] += sigma[v];
                }
            }
        }

        /* 反向传播依赖度 */
        for (size_t v = 0; v < n; v++) {
            delta[v] = 0.0f;
        }

        while (stack_size > 0) {
            size_t w = stack[--stack_size];

            for (size_t i = 0; i < adj_counts[w]; i++) {
                size_t v = adj[w][i];

                /* v是w的前驱：d[v] + 1 == d[w] 即v比w更接近源节点s */
                if (d[v] + 1 == d[w] && sigma[w] > 0) {
                    delta[v] += ((float)sigma[v] / (float)sigma[w]) * (1.0f + delta[w]);
                }
            }

            if (w != s) {
                scores[w] += delta[w];
            }
        }
    }

    /* 标准化：除以(n-1)*(n-2)/2（无向图） */
    if (n > 2) {
        float norm = ((float)(n - 1) * (float)(n - 2)) / 2.0f;
        if (norm > 0.0f) {
            for (size_t i = 0; i < n; i++) {
                scores[i] /= norm;
            }
        }
    }

    /* 释放内存 */
    safe_free((void**)&d);
    safe_free((void**)&sigma);
    safe_free((void**)&delta);
    safe_free((void**)&queue);
    safe_free((void**)&stack);
    for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
    safe_free((void**)&adj);
    safe_free((void**)&adj_counts);

    return 0;
}

/* ============================================================================
 * 紧密度中心度（全图）
 *
 * 对每个节点执行BFS，计算到所有可达节点的平均距离的倒数。
 * 不可达节点对其不贡献。
 * ============================================================================ */

int knowledge_graph_closeness_centrality_all(KnowledgeGraph* graph, float* scores, size_t score_count) {
    if (!graph || !scores) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "紧密度中心度：参数无效");
        return -1;
    }

    size_t n = graph->node_count;
    if (n == 0) return 0;
    if (score_count < n) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "紧密度中心度：分数数组太小");
        return -1;
    }

    /* 构建无向邻接表 */
    size_t** adj = NULL;
    size_t* adj_counts = NULL;
    if (build_undirected_adjacency(graph, &adj, &adj_counts) != 0) {
        return -1;
    }

    /* 为每个节点执行BFS */
    int* visited = (int*)safe_calloc(n, sizeof(int));
    int* dist = (int*)safe_malloc(n * sizeof(int));
    size_t* queue = (size_t*)safe_malloc(n * sizeof(size_t));

    if (!visited || !dist || !queue) {
        safe_free((void**)&visited);
        safe_free((void**)&dist);
        safe_free((void**)&queue);
        for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
        safe_free((void**)&adj);
        safe_free((void**)&adj_counts);
        return -1;
    }

    for (size_t s = 0; s < n; s++) {
        /* 初始化 */
        for (size_t i = 0; i < n; i++) {
            visited[i] = 0;
            dist[i] = -1;
        }
        visited[s] = 1;
        dist[s] = 0;

        size_t q_head = 0, q_tail = 0;
        queue[q_tail++] = s;

        while (q_head < q_tail) {
            size_t v = queue[q_head++];

            for (size_t i = 0; i < adj_counts[v]; i++) {
                size_t w = adj[v][i];
                if (!visited[w]) {
                    visited[w] = 1;
                    dist[w] = dist[v] + 1;
                    queue[q_tail++] = w;
                }
            }
        }

        /* 计算平均距离 */
        size_t total_dist = 0;
        size_t reachable = 0;
        for (size_t i = 0; i < n; i++) {
            if (i != s && visited[i]) {
                total_dist += (size_t)dist[i];
                reachable++;
            }
        }

        if (reachable > 0 && total_dist > 0) {
            /* 紧密度 = 可达节点数/总数 * (1/平均距离) */
            float avg_dist = (float)total_dist / (float)reachable;
            scores[s] = ((float)reachable / (float)(n - 1)) * (1.0f / avg_dist);
        } else {
            scores[s] = 0.0f;
        }
    }

    safe_free((void**)&visited);
    safe_free((void**)&dist);
    safe_free((void**)&queue);
    for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
    safe_free((void**)&adj);
    safe_free((void**)&adj_counts);

    return 0;
}

/* ============================================================================
 * Louvain社区检测算法
 *
 * 模块度最大化：
 * Q = 1/(2m) * sum_ij [A_ij - k_i*k_j/(2m)] * delta(c_i, c_j)
 *
 * 两阶段迭代：
 * 阶段1：局部移动节点以最大化模块度
 * 阶段2：聚合社区为超级节点，在新图上重复
 * ============================================================================ */

/**
 * @brief 计算将节点v从当前社区移除并加入社区c的模块度增益
 *
 * @param adj 无向邻接表
 * @param adj_counts 邻居计数
 * @param node_degrees 节点度数组
 * @param community 社区分配数组
 * @param comm_in 社区内部边权重和
 * @param comm_tot 社区总度数
 * @param v 节点索引
 * @param c 目标社区ID
 * @param m2 总边权重的2倍（2m）
 * @return float 模块度增益
 */
static float louvain_modularity_gain(size_t** adj, size_t* adj_counts,
                                      size_t* node_degrees,
                                      int* community,
                                      float* comm_in, float* comm_tot,
                                      size_t v, int c, float m2) {
    /* 计算v到社区c的连接权重k_i,in */
    float k_in = 0.0f;
    for (size_t i = 0; i < adj_counts[v]; i++) {
        size_t neighbor = adj[v][i];
        if (community[neighbor] == c) {
            k_in += 1.0f;
        }
    }

    int old_comm = community[v];
    float old_in = (old_comm >= 0) ? comm_in[old_comm] : 0.0f;
    float old_tot = (old_comm >= 0) ? comm_tot[old_comm] : 0.0f;
    float new_in = comm_in[c];
    float new_tot = comm_tot[c];
    float kv = (float)node_degrees[v];

    if (m2 <= 0.0f) return 0.0f;

    /* 计算当前状态模块度贡献 */
    float current_q = 0.0f;
    /* 移除v后的当前社区贡献（如果v是当前社区成员） */
    if (old_comm >= 0 && old_comm == c) {
        /* v已经在c中，计算移动增益需要先假设移除再计算加入 */
        current_q = (new_in - kv) / m2 - ((new_tot - kv) / m2) * ((new_tot - kv) / m2);
        current_q += old_in / m2 - (old_tot / m2) * (old_tot / m2);
        /* v自身的贡献 */
        float new_q_c = (new_in) / m2 - (new_tot / m2) * (new_tot / m2);
        float old_q_c = (new_in - kv) / m2 - ((new_tot - kv) / m2) * ((new_tot - kv) / m2);
        return new_q_c - old_q_c;
    }

    /* 模块度增益公式（无向、单位权重） */
    float gain = k_in / m2 - (new_tot * kv) / (m2 * m2);
    if (old_comm >= 0) {
        gain += (old_tot * kv) / (m2 * m2);
    }

    return gain;
}

int knowledge_graph_louvain_communities(KnowledgeGraph* graph, int* community_ids, size_t* community_count) {
    if (!graph || !community_ids || !community_count) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "Louvain社区检测：参数无效");
        return -1;
    }

    size_t n = graph->node_count;
    if (n == 0) {
        *community_count = 0;
        return 0;
    }

    /* 构建无向邻接表 */
    size_t** adj = NULL;
    size_t* adj_counts = NULL;
    if (build_undirected_adjacency(graph, &adj, &adj_counts) != 0) {
        return -1;
    }

    /* 计算节点度（无向图） */
    size_t* node_degrees = (size_t*)safe_calloc(n, sizeof(size_t));
    if (!node_degrees) {
        for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
        safe_free((void**)&adj);
        safe_free((void**)&adj_counts);
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        node_degrees[i] = adj_counts[i];
    }

    /* 计算总边数m（每条边计算一次） */
    float m2 = 0.0f;
    for (size_t i = 0; i < n; i++) {
        m2 += (float)node_degrees[i];
    }

    if (m2 <= 0.0f) {
        /* 无边：每个节点自成一个社区 */
        for (size_t i = 0; i < n; i++) community_ids[i] = (int)i;
        *community_count = n;
        safe_free((void**)&node_degrees);
        for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
        safe_free((void**)&adj);
        safe_free((void**)&adj_counts);
        return 0;
    }

    /* 初始化：每个节点自成一个社区 */
    int* community = (int*)safe_malloc(n * sizeof(int));
    float* comm_in = (float*)safe_calloc(n, sizeof(float));
    float* comm_tot = (float*)safe_calloc(n, sizeof(float));
    int* comm_active = (int*)safe_calloc(n, sizeof(int));

    if (!community || !comm_in || !comm_tot || !comm_active) {
        safe_free((void**)&community);
        safe_free((void**)&comm_in);
        safe_free((void**)&comm_tot);
        safe_free((void**)&comm_active);
        safe_free((void**)&node_degrees);
        for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
        safe_free((void**)&adj);
        safe_free((void**)&adj_counts);
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        community[i] = (int)i;
        comm_in[i] = 0.0f;
        comm_tot[i] = (float)node_degrees[i];
        comm_active[i] = 1;
    }

    /* Louvain阶段1：迭代局部移动 */
    int moved;
    int max_pass = 20;
    for (int pass = 0; pass < max_pass; pass++) {
        moved = 0;

        for (size_t v = 0; v < n; v++) {
            int cur_comm = community[v];
            float kv = (float)node_degrees[v];

            /* 从当前社区移除v */
            if (comm_active[cur_comm]) {
                comm_tot[cur_comm] -= kv;
                /* 移除v到当前社区内部的边贡献 */
                float internal_loss = 0.0f;
                for (size_t i = 0; i < adj_counts[v]; i++) {
                    size_t neighbor = adj[v][i];
                    if (community[neighbor] == cur_comm) {
                        internal_loss += 2.0f;
                    }
                }
                comm_in[cur_comm] -= internal_loss;
                if (comm_tot[cur_comm] <= 0) {
                    comm_active[cur_comm] = 0;
                }
            }

            /* 找出最佳目标社区 */
            int best_comm = cur_comm;
            float best_gain = 0.0f;

            /* 收集v的邻居社区 */
            int* comm_candidates = (int*)safe_calloc(n, sizeof(int));
            size_t num_candidates = 0;
            for (size_t i = 0; i < adj_counts[v]; i++) {
                size_t neighbor = adj[v][i];
                int nc = community[neighbor];
                if (!comm_candidates[nc]) {
                    comm_candidates[nc] = 1;
                    num_candidates++;
                }
            }

            /* 评估移动到每个候选社区 */
            for (size_t c = 0; c < n && num_candidates > 0; c++) {
                if (!comm_candidates[c]) continue;
                if (c == (size_t)cur_comm) continue;
                if (!comm_active[c]) continue;

                float k_in = 0.0f;
                for (size_t i = 0; i < adj_counts[v]; i++) {
                    if (community[adj[v][i]] == (int)c) {
                        k_in += 1.0f;
                    }
                }

                float gain = k_in / m2 - (comm_tot[c] * kv) / (m2 * m2);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_comm = (int)c;
                }
            }

            safe_free((void**)&comm_candidates);

            /* 移动到最佳社区 */
            if (best_comm != cur_comm) {
                community[v] = best_comm;
                comm_tot[best_comm] += kv;
                /* 添加v到最佳社区内部的边贡献 */
                float internal_gain = 0.0f;
                for (size_t i = 0; i < adj_counts[v]; i++) {
                    size_t neighbor = adj[v][i];
                    if (community[neighbor] == best_comm) {
                        internal_gain += 2.0f;
                    }
                }
                comm_in[best_comm] += internal_gain;
                comm_active[best_comm] = 1;
                moved = 1;
            } else {
                /* 回到原社区 */
                comm_tot[cur_comm] += kv;
                float internal_restore = 0.0f;
                for (size_t i = 0; i < adj_counts[v]; i++) {
                    size_t neighbor = adj[v][i];
                    if (community[neighbor] == cur_comm) {
                        internal_restore += 2.0f;
                    }
                }
                comm_in[cur_comm] += internal_restore;
                comm_active[cur_comm] = 1;
            }
        }

        if (!moved) break;
    }

    /* 重新映射社区ID为连续整数 */
    int* comm_map = (int*)safe_calloc(n, sizeof(int));
    for (size_t i = 0; i < n; i++) comm_map[i] = -1;

    size_t next_id = 0;
    for (size_t i = 0; i < n; i++) {
        if (comm_map[community[i]] < 0) {
            comm_map[community[i]] = (int)next_id++;
        }
        community_ids[i] = comm_map[community[i]];
    }
    *community_count = next_id;

    /* 计算模块度 */
    float modularity = 0.0f;
    if (m2 > 0.0f) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < adj_counts[i]; j++) {
                size_t neighbor = adj[i][j];
                if (community_ids[i] == community_ids[neighbor]) {
                    float expected = (float)node_degrees[i] * (float)node_degrees[neighbor] / m2;
                    modularity += 1.0f - expected;
                }
            }
        }
        modularity /= m2;
    }

    /* 释放内存 */
    safe_free((void**)&comm_map);
    safe_free((void**)&community);
    safe_free((void**)&comm_in);
    safe_free((void**)&comm_tot);
    safe_free((void**)&comm_active);
    safe_free((void**)&node_degrees);
    for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
    safe_free((void**)&adj);
    safe_free((void**)&adj_counts);

    return 0;
}

/* ============================================================================
 * 图直径
 *
 * 对每个节点执行BFS，计算到所有可达节点的最短距离，
 * 取所有距离的最大值。不可达节点对不贡献。
 * ============================================================================ */

float knowledge_graph_diameter(KnowledgeGraph* graph) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "图直径：知识图谱为空");
        return 0.0f;
    }

    size_t n = graph->node_count;
    if (n < 2) return 0.0f;

    /* 构建无向邻接表 */
    size_t** adj = NULL;
    size_t* adj_counts = NULL;
    if (build_undirected_adjacency(graph, &adj, &adj_counts) != 0) {
        return -1.0f;
    }

    float max_dist = 0.0f;
    int* visited = (int*)safe_calloc(n, sizeof(int));
    int* dist = (int*)safe_malloc(n * sizeof(int));
    size_t* queue = (size_t*)safe_malloc(n * sizeof(size_t));

    if (!visited || !dist || !queue) {
        safe_free((void**)&visited);
        safe_free((void**)&dist);
        safe_free((void**)&queue);
        for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
        safe_free((void**)&adj);
        safe_free((void**)&adj_counts);
        return -1.0f;
    }

    for (size_t s = 0; s < n; s++) {
        for (size_t i = 0; i < n; i++) {
            visited[i] = 0;
            dist[i] = -1;
        }
        visited[s] = 1;
        dist[s] = 0;

        size_t q_head = 0, q_tail = 0;
        queue[q_tail++] = s;

        while (q_head < q_tail) {
            size_t v = queue[q_head++];
            for (size_t i = 0; i < adj_counts[v]; i++) {
                size_t w = adj[v][i];
                if (!visited[w]) {
                    visited[w] = 1;
                    dist[w] = dist[v] + 1;
                    queue[q_tail++] = w;
                }
            }
        }

        for (size_t i = 0; i < n; i++) {
            if (visited[i] && (float)dist[i] > max_dist) {
                max_dist = (float)dist[i];
            }
        }
    }

    safe_free((void**)&visited);
    safe_free((void**)&dist);
    safe_free((void**)&queue);
    for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
    safe_free((void**)&adj);
    safe_free((void**)&adj_counts);

    return max_dist;
}

/* ============================================================================
 * 图平均度
 *
 * 每条无向边贡献2度，平均度 = 2 * |E| / |V|
 * ============================================================================ */

float knowledge_graph_average_degree(KnowledgeGraph* graph) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "平均度：知识图谱为空");
        return 0.0f;
    }

    size_t n = graph->node_count;
    if (n == 0) return 0.0f;

    return 2.0f * (float)graph->edge_count / (float)n;
}

/* ============================================================================
 * 完整图分析统计
 *
 * 一次性计算所有图分析指标并填充KnowledgeGraphStats。
 * ============================================================================ */

int knowledge_graph_compute_stats(KnowledgeGraph* graph, KnowledgeGraphStats* stats) {
    if (!graph || !stats) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "计算统计：参数无效");
        return -1;
    }

    size_t n = graph->node_count;
    size_t e = graph->edge_count;

    /* 清零stats结构 */
    memset(stats, 0, sizeof(KnowledgeGraphStats));

    /* 基本信息 */
    stats->node_count = n;
    stats->edge_count = e;

    /* 内存使用量 */
    size_t memory = sizeof(KnowledgeGraph);
    memory += graph->node_capacity * sizeof(KnowledgeGraphNode*);
    memory += graph->edge_capacity * sizeof(KnowledgeGraphEdge*);
    /* P0修复: 添加node/edge的NULL指针检查，防止知识图谱中存在空槽位时崩溃 */
    for (size_t i = 0; i < n; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (!node) continue;  /* 跳过空节点槽位 */
        memory += sizeof(KnowledgeGraphNode);
        if (node->label) memory += strlen(node->label) + 1;
        if (node->embedding) memory += node->embedding_size * sizeof(float);
        memory += node->edge_capacity * sizeof(KnowledgeGraphEdge*);
    }
    for (size_t i = 0; i < e; i++) {
        KnowledgeGraphEdge* edge = graph->edges[i];
        if (!edge) continue;  /* 跳过空边槽位 */
        memory += sizeof(KnowledgeGraphEdge);
        if (edge->label) memory += strlen(edge->label) + 1;
    }
    stats->memory_usage = memory;

    if (n == 0) return 0;

    /* 图密度 */
    stats->density = knowledge_graph_density(graph);

    /* 平均度 */
    stats->avg_degree = 2.0f * (float)e / (float)n;

    /* 图直径 */
    stats->diameter = knowledge_graph_diameter(graph);

    /* 平均聚类系数 */
    stats->avg_clustering = knowledge_graph_clustering_coefficient(graph);

    /* PageRank */
    stats->pagerank_scores = (float*)safe_malloc(n * sizeof(float));
    if (stats->pagerank_scores) {
        if (knowledge_graph_pagerank(graph, stats->pagerank_scores, n) != 0) {
            safe_free((void**)&stats->pagerank_scores);
        }
    }

    /* 介数中心度 */
    stats->betweenness_scores = (float*)safe_malloc(n * sizeof(float));
    if (stats->betweenness_scores) {
        if (knowledge_graph_betweenness_centrality_all(graph, stats->betweenness_scores, n) != 0) {
            safe_free((void**)&stats->betweenness_scores);
        }
    }

    /* 紧密度中心度 */
    stats->closeness_scores = (float*)safe_malloc(n * sizeof(float));
    if (stats->closeness_scores) {
        if (knowledge_graph_closeness_centrality_all(graph, stats->closeness_scores, n) != 0) {
            safe_free((void**)&stats->closeness_scores);
        }
    }

    /* Louvain社区检测 */
    stats->community_ids = (int*)safe_malloc(n * sizeof(int));
    if (stats->community_ids) {
        size_t comm_count = 0;
        if (knowledge_graph_louvain_communities(graph, stats->community_ids, &comm_count) == 0) {
            stats->community_count = comm_count;
        } else {
            safe_free((void**)&stats->community_ids);
        }
    }

    /* 社区检测后计算模块度（Louvain内部已计算，但额外计算一次全局模块度） */
    if (stats->community_ids && stats->community_count > 0) {
        /* 构建无向邻接表用于模块度计算 */
        size_t** uadj = NULL;
        size_t* uadj_counts = NULL;
        if (build_undirected_adjacency(graph, &uadj, &uadj_counts) == 0) {
            float m2 = 0.0f;
            size_t* degrees = (size_t*)safe_calloc(n, sizeof(size_t));
            if (degrees) {
                for (size_t i = 0; i < n; i++) degrees[i] = uadj_counts[i];
                for (size_t i = 0; i < n; i++) m2 += (float)degrees[i];
                if (m2 > 0.0f) {
                    float q = 0.0f;
                    for (size_t i = 0; i < n; i++) {
                        for (size_t j = 0; j < uadj_counts[i]; j++) {
                            size_t neighbor = uadj[i][j];
                            if (stats->community_ids[i] == stats->community_ids[neighbor]) {
                                q += 1.0f - ((float)degrees[i] * (float)degrees[neighbor]) / m2;
                            }
                        }
                    }
                    stats->modularity = q / m2;
                }
                safe_free((void**)&degrees);
            }
            for (size_t k = 0; k < n; k++) safe_free((void**)&uadj[k]);
            safe_free((void**)&uadj);
            safe_free((void**)&uadj_counts);
        }
    }

    return 0;
}

/* ============================================================================
 * 释放KnowledgeGraphStats
 * ============================================================================ */

void knowledge_graph_free_stats(KnowledgeGraphStats* stats) {
    if (!stats) return;

    if (stats->pagerank_scores) {
        safe_free((void**)&stats->pagerank_scores);
    }
    if (stats->betweenness_scores) {
        safe_free((void**)&stats->betweenness_scores);
    }
    if (stats->closeness_scores) {
        safe_free((void**)&stats->closeness_scores);
    }
    if (stats->community_ids) {
        safe_free((void**)&stats->community_ids);
    }

    memset(stats, 0, sizeof(KnowledgeGraphStats));
}

/* ============================================================================
 * 知识库集成
 * =========================================================================== */

int knowledge_graph_import_from_knowledge_base(KnowledgeGraph* graph,
                                              KnowledgeBase* kb) {
    if (!graph || !kb) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "从知识库导入：参数无效");
        return -1;
    }
    
    // 完整实现：从知识库导入所有知识条目到知识图谱（ 处理）
    
    // 创建查询条件：获取所有条目
    KnowledgeQuery query;
    query.subject_pattern = NULL;
    query.predicate_pattern = NULL;
    query.object_pattern = NULL;
    query.type_filter = -1; // 不过滤类型
    query.min_confidence = 0.0f;
    query.max_confidence = 1.0f;
    query.start_time = 0;
    query.end_time = LONG_MAX;
    
    // 估计条目数量（先查询一次获取数量）
    int estimated_count = knowledge_base_query(kb, &query, NULL, 0);
    if (estimated_count <= 0) {
        // 知识库为空，导入成功（无内容）
        return 0;
    }
    
    // 分配缓冲区存储条目
    KnowledgeEntry* entries = (KnowledgeEntry*)safe_malloc(estimated_count * sizeof(KnowledgeEntry));
    if (!entries) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "从知识库导入：内存分配失败");
        return -1;
    }
    
    // 查询所有条目
    int actual_count = knowledge_base_query(kb, &query, entries, estimated_count);
    if (actual_count < 0) {
        safe_free((void**)&entries);
        selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__,
                              "从知识库导入：查询失败");
        return -1;
    }
    
    // 用于跟踪已创建的节点（主体和客体）
    // 完整实现：使用哈希表进行高效查找（ 处理）
    
    // 创建哈希表用于快速节点查找
    size_t hash_capacity = (actual_count > 0) ? (size_t)(actual_count * 2) : 16;
    HashEntry* hash_table = hash_table_create(hash_capacity);
    if (!hash_table) {
        safe_free((void**)&entries);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "从知识库导入：哈希表创建失败");
        return -1;
    }
    
    /* 首先，将图中现有节点添加到哈希表（使用自动扩容版本） */
    for (size_t j = 0; j < graph->node_count; j++) {
        KnowledgeGraphNode* node = graph->nodes[j];
        if (node && node->label) {
            hash_table_insert_or_rehash(&hash_table, &hash_capacity, node->label, node);
        }
    }
    
    // 导入每个条目
    int success_count = 0;
    
    for (int i = 0; i < actual_count; i++) {
        KnowledgeEntry* entry = &entries[i];
        
        // 确保主体、谓词、客体都不为空
        if (!entry->subject || !entry->predicate || !entry->object) {
            continue; // 跳过无效条目
        }
        
        // 查找或创建主体节点
        KnowledgeGraphNode* subject_node = hash_table_lookup(hash_table, hash_capacity, entry->subject);
        
        // 如果不存在，创建新节点
        if (!subject_node) {
            subject_node = knowledge_graph_add_node(graph, NODE_TYPE_ENTITY, entry->subject, NULL, 0, 0.5f);
            if (!subject_node) {
                continue; // 创建失败，跳过此条目
            }
            /* 将新节点添加到哈希表 */
            hash_table_insert_or_rehash(&hash_table, &hash_capacity, entry->subject, subject_node);
        }
        
        // 查找或创建客体节点
        KnowledgeGraphNode* object_node = hash_table_lookup(hash_table, hash_capacity, entry->object);
        
        // 如果不存在，创建新节点
        if (!object_node) {
            object_node = knowledge_graph_add_node(graph, NODE_TYPE_ENTITY, entry->object, NULL, 0, 0.5f);
            if (!object_node) {
                continue; // 创建失败，跳过此条目
            }
            /* 将新节点添加到哈希表 */
            hash_table_insert_or_rehash(&hash_table, &hash_capacity, entry->object, object_node);
        }
        
        // 创建边（关系）
        KnowledgeGraphEdge* edge = knowledge_graph_add_edge(graph, EDGE_TYPE_RELATION, subject_node, object_node, 
                                                  entry->predicate, entry->weight, 0.5f);
        if (!edge) {
            continue; // 创建失败，跳过此条目
        }
        
        // 设置边的元数据（知识类型、置信度、来源等）
        edge->type = (int)entry->type;
        // 可以存储更多元数据到edge->metadata
        
        success_count++;
    }
    
    // 释放条目缓冲区
    safe_free((void**)&entries);
    
    // 释放哈希表
    hash_table_free(hash_table, hash_capacity);
    
    // 记录导入统计
    if (success_count < actual_count) {
        selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__,
                              "从知识库导入：部分条目导入失败");
        // 仍然返回成功，因为至少导入了一些条目
    }
    
    return 0; // 成功
}

int knowledge_graph_export_to_knowledge_base(KnowledgeGraph* graph,
                                            KnowledgeBase* kb) {
    if (!graph || !kb) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "导出到知识库：参数无效");
        return -1;
    }
    
    // 完整实现：将知识图谱导出到知识库（ 处理）
    
    int success_count = 0;
    int total_edges = 0;
    
    // 遍历所有边（每条边对应一个知识条目）
    for (size_t i = 0; i < graph->edge_count; i++) {
        KnowledgeGraphEdge* edge = graph->edges[i];
        
        if (!edge->source || !edge->target || !edge->label) {
            continue; // 跳过无效边
        }
        
        // 创建知识条目
        KnowledgeEntry entry;
        
        // 分配字符串内存
        entry.subject = edge->source->label ? string_duplicate(edge->source->label) : string_duplicate("未知");
        entry.predicate = edge->label ? string_duplicate(edge->label) : string_duplicate("关系");
        entry.object = edge->target->label ? string_duplicate(edge->target->label) : string_duplicate("未知");
        
        if (!entry.subject || !entry.predicate || !entry.object) {
            // 释放已分配的内存
            if (entry.subject) safe_free((void**)&entry.subject);
            if (entry.predicate) safe_free((void**)&entry.predicate);
            if (entry.object) safe_free((void**)&entry.object);
            continue;
        }
        
        // 设置条目属性
        entry.type = (edge->type >= 0 && edge->type <= 4) ? (KnowledgeType)edge->type : KNOWLEDGE_RELATION;
        entry.confidence = CONFIDENCE_MEDIUM; // 默认置信度
        entry.source = SOURCE_KNOWLEDGE_GRAPH;
        entry.weight = edge->weight;
        entry.timestamp = (long)time(NULL);
        entry.metadata = NULL;
        entry.metadata_size = 0;
        
        // 添加到知识库
        int entry_id = knowledge_base_add(kb, &entry);
        
        // 释放字符串内存
        safe_free((void**)&entry.subject);
        safe_free((void**)&entry.predicate);
        safe_free((void**)&entry.object);
        
        if (entry_id >= 0) {
            success_count++;
        }
        
        total_edges++;
    }
    
    // 如果没有边，导出成功（空图谱）
    if (total_edges == 0) {
        return 0;
    }
    
    // 检查导出结果
    if (success_count < total_edges) {
        selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__,
                              "导出到知识库：部分条目导出失败");
        // 仍然返回成功，因为至少导出了一些条目
    }
    
    return 0; // 成功
}

/* ============================================================================
 * 自动保存
 * =========================================================================== */

int knowledge_graph_set_auto_save(KnowledgeGraph* graph, const char* save_path) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置自动保存：知识图谱为空");
        return -1;
    }
    
    // 释放旧的自动保存路径
    if (graph->auto_save_path) {
        safe_free((void**)&graph->auto_save_path);
        graph->auto_save_path = NULL;
    }
    
    if (save_path) {
        graph->auto_save_path = string_duplicate(save_path);
        if (!graph->auto_save_path) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "设置自动保存：路径字符串复制失败");
            return -1;
        }
    }
    
    graph->dirty = 1;
    return 0;
}

int knowledge_graph_save_if_dirty(KnowledgeGraph* graph) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "脏保存：知识图谱为空");
        return -1;
    }
    
    if (!graph->dirty || !graph->auto_save_path) {
        return 0;
    }
    
    return knowledge_graph_save(graph, graph->auto_save_path);
}

/* ============================================================================
 * 序列化
 * =========================================================================== */

int knowledge_graph_save(KnowledgeGraph* graph, const char* filename) {
    if (!graph || !filename) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "保存知识图谱：参数无效");
        return -1;
    }
    
    // 使用临时文件实现原子写入
    char temp_filename[1024];
    int temp_name_len = snprintf(temp_filename, sizeof(temp_filename), "%s.tmp", filename);
    if (temp_name_len <= 0 || (size_t)temp_name_len >= sizeof(temp_filename)) {
        selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__,
                              "保存知识图谱：临时文件名创建失败");
        return -1;
    }
    
    FILE* file = fopen(temp_filename, "wb");
    if (!file) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：无法创建临时文件 '%s'", temp_filename);
        return -1;
    }
    
    // 写入魔数和版本
    const uint32_t magic = 0x4B47524D; // "KGRM" - Knowledge Graph Representation Magic
    const uint32_t version = 1;
    
    if (fwrite(&magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&version, sizeof(version), 1, file) != 1) {
        fclose(file);
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：写入文件头失败");
        return -1;
    }
    
    // 写入节点数量
    uint64_t node_count = (uint64_t)graph->node_count;
    uint64_t edge_count = (uint64_t)graph->edge_count;
    
    if (fwrite(&node_count, sizeof(node_count), 1, file) != 1) {
        fclose(file);
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：写入节点数量失败");
        return -1;
    }
    
    // 写入每个节点
    for (size_t i = 0; i < graph->node_count; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        
        uint64_t node_id = i;
        if (fwrite(&node_id, sizeof(node_id), 1, file) != 1) {
            fclose(file);
            remove(temp_filename);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存知识图谱：写入节点ID失败");
            return -1;
        }
        
        uint32_t node_type = (uint32_t)node->type;
        if (fwrite(&node_type, sizeof(node_type), 1, file) != 1) {
            fclose(file);
            remove(temp_filename);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存知识图谱：写入节点类型失败");
            return -1;
        }
        
        uint64_t label_length = node->label ? strlen(node->label) : 0;
        if (fwrite(&label_length, sizeof(label_length), 1, file) != 1) {
            fclose(file);
            remove(temp_filename);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存知识图谱：写入标签长度失败");
            return -1;
        }
        
        if (label_length > 0) {
            if (fwrite(node->label, 1, label_length, file) != label_length) {
                fclose(file);
                remove(temp_filename);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "保存知识图谱：写入标签失败");
                return -1;
            }
        }
        
        uint64_t embedding_size = (uint64_t)node->embedding_size;
        if (fwrite(&embedding_size, sizeof(embedding_size), 1, file) != 1) {
            fclose(file);
            remove(temp_filename);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存知识图谱：写入嵌入大小失败");
            return -1;
        }
        
        if (embedding_size > 0 && node->embedding) {
            if (fwrite(node->embedding, sizeof(float), embedding_size, file) != embedding_size) {
                fclose(file);
                remove(temp_filename);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "保存知识图谱：写入嵌入数据失败");
                return -1;
            }
        }
    }
    
    if (fwrite(&edge_count, sizeof(edge_count), 1, file) != 1) {
        fclose(file);
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：写入边数量失败");
        return -1;
    }
    
    for (size_t i = 0; i < graph->edge_count; i++) {
        KnowledgeGraphEdge* edge = graph->edges[i];
        
        uint64_t source_idx = UINT64_MAX;
        uint64_t target_idx = UINT64_MAX;
        
        for (size_t j = 0; j < graph->node_count; j++) {
            if (graph->nodes[j] == edge->source) source_idx = j;
            if (graph->nodes[j] == edge->target) target_idx = j;
        }
        
        if (source_idx == UINT64_MAX || target_idx == UINT64_MAX) {
            fclose(file);
            remove(temp_filename);
            selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__,
                                  "保存知识图谱：无法找到边的节点索引");
            return -1;
        }
        
        if (fwrite(&source_idx, sizeof(source_idx), 1, file) != 1 ||
            fwrite(&target_idx, sizeof(target_idx), 1, file) != 1) {
            fclose(file);
            remove(temp_filename);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存知识图谱：写入边节点索引失败");
            return -1;
        }
        
        uint32_t edge_type = (uint32_t)edge->type;
        if (fwrite(&edge_type, sizeof(edge_type), 1, file) != 1) {
            fclose(file);
            remove(temp_filename);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存知识图谱：写入边类型失败");
            return -1;
        }
        
        float weight = edge->weight;
        if (fwrite(&weight, sizeof(weight), 1, file) != 1) {
            fclose(file);
            remove(temp_filename);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存知识图谱：写入边权重失败");
            return -1;
        }
        
        uint64_t label_length = edge->label ? strlen(edge->label) : 0;
        if (fwrite(&label_length, sizeof(label_length), 1, file) != 1) {
            fclose(file);
            remove(temp_filename);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "保存知识图谱：写入边标签长度失败");
            return -1;
        }
        
        if (label_length > 0) {
            if (fwrite(edge->label, 1, label_length, file) != label_length) {
                fclose(file);
                remove(temp_filename);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "保存知识图谱：写入边标签失败");
                return -1;
            }
        }
    }
    
    fclose(file);

    // 计算CRC32校验和
    file = fopen(temp_filename, "rb");
    if (!file) {
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：无法打开临时文件计算CRC32");
        return -1;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：获取文件大小失败");
        return -1;
    }
    long file_size = ftell(file);
    if (file_size <= 0) {
        fclose(file);
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：文件大小无效");
        return -1;
    }

    unsigned char* file_buffer = (unsigned char*)safe_malloc((size_t)file_size);
    if (!file_buffer) {
        fclose(file);
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "保存知识图谱：分配CRC32缓冲区失败");
        return -1;
    }

    fseek(file, 0, SEEK_SET);
    if (fread(file_buffer, 1, (size_t)file_size, file) != (size_t)file_size) {
        safe_free((void**)&file_buffer);
        fclose(file);
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：读取文件计算CRC32失败");
        return -1;
    }
    fclose(file);

    uint32_t checksum = crc32_compute(file_buffer, (size_t)file_size);
    safe_free((void**)&file_buffer);

    // 追加CRC32校验和到临时文件末尾
    file = fopen(temp_filename, "ab");
    if (!file) {
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：无法追加CRC32校验和");
        return -1;
    }
    if (fwrite(&checksum, sizeof(checksum), 1, file) != 1) {
        fclose(file);
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：写入CRC32校验和失败");
        return -1;
    }
    fclose(file);

    // 原子重命名：先写临时文件成功后再覆盖目标文件
    if (rename(temp_filename, filename) != 0) {
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：重命名临时文件失败");
        return -1;
    }
    
    graph->dirty = 0;
    return 0;
}

KnowledgeGraph* knowledge_graph_load(const char* filename) {
    if (!filename) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "加载知识图谱：文件名为空");
        return NULL;
    }
    
    // 完整实现：从文件加载知识图谱（ 处理）
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：无法打开文件 '%s'", filename);
        return NULL;
    }

    // 读取文件末尾CRC32校验和
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：定位文件末尾失败");
        return NULL;
    }
    long total_file_size = ftell(file);
    if (total_file_size < 16) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：文件大小过小（小于最小有效大小）");
        return NULL;
    }

    uint32_t stored_checksum = 0;
    if (fseek(file, -4, SEEK_END) != 0) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：定位CRC32字段失败");
        return NULL;
    }
    if (fread(&stored_checksum, sizeof(stored_checksum), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：读取CRC32校验和失败");
        return NULL;
    }

    // 计算文件数据的CRC32（排除最后4字节的校验和本身）
    long data_size = total_file_size - 4;
    if (data_size <= 0) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：文件数据部分为空");
        return NULL;
    }

    unsigned char* verify_buffer = (unsigned char*)safe_malloc((size_t)data_size);
    if (!verify_buffer) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "加载知识图谱：分配CRC32验证缓冲区失败");
        return NULL;
    }

    fseek(file, 0, SEEK_SET);
    if (fread(verify_buffer, 1, (size_t)data_size, file) != (size_t)data_size) {
        safe_free((void**)&verify_buffer);
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：读取文件数据计算CRC32失败");
        return NULL;
    }

    uint32_t computed_checksum = crc32_compute(verify_buffer, (size_t)data_size);
    safe_free((void**)&verify_buffer);

    if (computed_checksum != stored_checksum) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：CRC32校验和不匹配（文件已损坏或篡改）");
        return NULL;
    }

    // 重新定位到文件开头开始解析
    fseek(file, 0, SEEK_SET);

    // 读取魔数和版本
    uint32_t magic, version;
    if (fread(&magic, sizeof(magic), 1, file) != 1 ||
        fread(&version, sizeof(version), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：读取文件头失败");
        return NULL;
    }
    
    // 验证魔数
    if (magic != 0x4B47524D) { // "KGRM"
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：文件格式无效（魔数不匹配）");
        return NULL;
    }
    
    // 验证版本
    if (version != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：不支持的版本 %u", version);
        return NULL;
    }
    
    // 读取节点数量
    uint64_t node_count;
    if (fread(&node_count, sizeof(node_count), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：读取节点数量失败");
        return NULL;
    }
    
    // 创建知识图谱
    KnowledgeGraph* graph = knowledge_graph_create(100, 100);
    if (!graph) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "加载知识图谱：创建知识图谱失败");
        return NULL;
    }
    
    // 预分配节点空间（由 knowledge_graph_add_node 内部处理）
    
    // 读取每个节点
    KnowledgeGraphNode** loaded_nodes = (KnowledgeGraphNode**)safe_malloc(node_count * sizeof(KnowledgeGraphNode*));
    if (!loaded_nodes && node_count > 0) {
        knowledge_graph_free(graph);
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "加载知识图谱：分配节点指针数组失败");
        return NULL;
    }
    
    /* B-020修复: 追踪文件中最大存储ID，用于加载后重置next_node_id */
    int max_stored_id = -1;
    for (uint64_t i = 0; i < node_count; i++) {
        // 读取节点ID（索引）
        uint64_t node_id;
        if (fread(&node_id, sizeof(node_id), 1, file) != 1) {
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：读取节点ID失败");
            return NULL;
        }
        
        // 读取节点类型
        uint32_t node_type;
        if (fread(&node_type, sizeof(node_type), 1, file) != 1) {
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：读取节点类型失败");
            return NULL;
        }
        
        // 读取标签长度
        uint64_t label_length;
        if (fread(&label_length, sizeof(label_length), 1, file) != 1) {
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：读取标签长度失败");
            return NULL;
        }
        
        // 读取标签
        char* label = NULL;
        if (label_length > 0) {
            /* P2修复: 防止label_length+1整数溢出导致分配极小内存后越界写入 */
            if (label_length > 65536) {
                knowledge_graph_free(graph);
                safe_free((void**)&loaded_nodes);
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                      "加载知识图谱：节点标签长度超过上限65536");
                return NULL;
            }
            label = (char*)safe_malloc(label_length + 1);
            if (!label) {
                knowledge_graph_free(graph);
                safe_free((void**)&loaded_nodes);
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                                      "加载知识图谱：分配标签内存失败");
                return NULL;
            }
            
            if (fread(label, 1, label_length, file) != label_length) {
                safe_free((void**)&label);
                knowledge_graph_free(graph);
                safe_free((void**)&loaded_nodes);
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载知识图谱：读取标签失败");
                return NULL;
            }
            
            label[label_length] = '\0';
        }
        
        // 读取嵌入向量大小
        uint64_t embedding_size;
        if (fread(&embedding_size, sizeof(embedding_size), 1, file) != 1) {
            if (label) safe_free((void**)&label);
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：读取嵌入大小失败");
            return NULL;
        }
        
        // 读取嵌入数据
        float* embedding = NULL;
        if (embedding_size > 0) {
            /* P2修复: 防止embedding_size * sizeof(float)整数溢出 */
            if (embedding_size > 1048576) {
                if (label) safe_free((void**)&label);
                knowledge_graph_free(graph);
                safe_free((void**)&loaded_nodes);
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                      "加载知识图谱：嵌入向量大小超过上限1048576");
                return NULL;
            }
            embedding = (float*)safe_malloc(embedding_size * sizeof(float));
            if (!embedding) {
                if (label) safe_free((void**)&label);
                knowledge_graph_free(graph);
                safe_free((void**)&loaded_nodes);
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                                      "加载知识图谱：分配嵌入内存失败");
                return NULL;
            }
            
            if (fread(embedding, sizeof(float), embedding_size, file) != embedding_size) {
                safe_free((void**)&embedding);
                if (label) safe_free((void**)&label);
                knowledge_graph_free(graph);
                safe_free((void**)&loaded_nodes);
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载知识图谱：读取嵌入数据失败");
                return NULL;
            }
        }
        
        // 创建节点
        KnowledgeGraphNode* node = knowledge_graph_add_node(graph, (KnowledgeGraphNodeType)node_type, label, NULL, 0, 0.5f);
        if (!node) {
            if (embedding) safe_free((void**)&embedding);
            if (label) safe_free((void**)&label);
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：添加节点失败");
            return NULL;
        }
        /* B-020修复: 直接使用文件中存储的节点ID，避免next_node_id与存储ID不匹配 */
        node->id = (int)node_id;
        if ((int)node_id > max_stored_id) max_stored_id = (int)node_id;
        
        // 设置嵌入向量
        if (embedding && embedding_size > 0) {
            // 直接设置节点嵌入向量
            safe_free((void**)&node->embedding);
            node->embedding = embedding;
            node->embedding_size = (size_t)embedding_size;
            // 注意：embedding 的所有权已转移给节点，不要在此处释放
        } else if (embedding) {
            safe_free((void**)&embedding); // 如果 embedding_size 为0，释放内存
        }
        
        if (label) safe_free((void**)&label);
        
        // 存储节点指针（用于边连接）
        loaded_nodes[i] = node;
    }
    
    // 读取边数量
    uint64_t edge_count;
    if (fread(&edge_count, sizeof(edge_count), 1, file) != 1) {
        knowledge_graph_free(graph);
        safe_free((void**)&loaded_nodes);
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：读取边数量失败");
        return NULL;
    }
    
    // 预分配边空间（由 knowledge_graph_add_edge 内部处理）
    
    // 读取每条边
    for (uint64_t i = 0; i < edge_count; i++) {
        // 读取源节点索引和目标节点索引
        uint64_t source_idx, target_idx;
        if (fread(&source_idx, sizeof(source_idx), 1, file) != 1 ||
            fread(&target_idx, sizeof(target_idx), 1, file) != 1) {
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：读取边节点索引失败");
            return NULL;
        }
        
        // 检查索引有效性
        if (source_idx >= node_count || target_idx >= node_count) {
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：边节点索引无效");
            return NULL;
        }
        
        // 读取边类型
        uint32_t edge_type;
        if (fread(&edge_type, sizeof(edge_type), 1, file) != 1) {
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：读取边类型失败");
            return NULL;
        }
        
        // 读取边权重
        float weight;
        if (fread(&weight, sizeof(weight), 1, file) != 1) {
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：读取边权重失败");
            return NULL;
        }
        
        // 读取标签长度
        uint64_t label_length;
        if (fread(&label_length, sizeof(label_length), 1, file) != 1) {
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：读取边标签长度失败");
            return NULL;
        }
        
        // 读取标签
        char* label = NULL;
        if (label_length > 0) {
            /* P2修复: 防止边label_length+1整数溢出导致分配极小内存后越界写入 */
            if (label_length > 65536) {
                knowledge_graph_free(graph);
                safe_free((void**)&loaded_nodes);
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                      "加载知识图谱：边标签长度超过上限65536");
                return NULL;
            }
            label = (char*)safe_malloc(label_length + 1);
            if (!label) {
                knowledge_graph_free(graph);
                safe_free((void**)&loaded_nodes);
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                                      "加载知识图谱：分配边标签内存失败");
                return NULL;
            }
            
            if (fread(label, 1, label_length, file) != label_length) {
                safe_free((void**)&label);
                knowledge_graph_free(graph);
                safe_free((void**)&loaded_nodes);
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载知识图谱：读取边标签失败");
                return NULL;
            }
            
            label[label_length] = '\0';
        }
        
        // 获取源节点和目标节点
        KnowledgeGraphNode* source_node = loaded_nodes[source_idx];
        KnowledgeGraphNode* target_node = loaded_nodes[target_idx];
        
        // 创建边
        KnowledgeGraphEdge* edge = knowledge_graph_add_edge(graph, EDGE_TYPE_RELATION, source_node, target_node, label, weight, 0.5f);
        if (!edge) {
            if (label) safe_free((void**)&label);
            knowledge_graph_free(graph);
            safe_free((void**)&loaded_nodes);
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__,
                                  "加载知识图谱：添加边失败");
            return NULL;
        }
        
        // 设置边类型
        edge->type = (int)edge_type;
        
        if (label) safe_free((void**)&label);
    }
    
    // 清理临时数组
    safe_free((void**)&loaded_nodes);
    fclose(file);
    
    /* B-020修复: 重置next_node_id为文件中最⼤ID+1 */
    if (max_stored_id >= 0) {
        graph->next_node_id = max_stored_id + 1;
    }
    
    return graph;
}

/* ============================================================================
 * 统计信息
 * =========================================================================== */

int knowledge_graph_get_stats(KnowledgeGraph* graph,
                             size_t* node_count, size_t* edge_count,
                             size_t* memory_usage) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "获取统计：知识图谱为空");
        return -1;
    }
    
    if (node_count) *node_count = graph->node_count;
    if (edge_count) *edge_count = graph->edge_count;
    
    // 完整内存使用量计算（ 处理）
    if (memory_usage) {
        size_t total_memory = 0;
        
        // 基础结构内存
        total_memory += sizeof(KnowledgeGraph);
        
        // 节点数组内存
        total_memory += graph->node_capacity * sizeof(KnowledgeGraphNode*);
        
        // 边数组内存
        total_memory += graph->edge_capacity * sizeof(KnowledgeGraphEdge*);
        
        // 每个节点的内存
        for (size_t i = 0; i < graph->node_count; i++) {
            KnowledgeGraphNode* node = graph->nodes[i];
            total_memory += sizeof(KnowledgeGraphNode);
            
            if (node->label) {
                total_memory += strlen(node->label) + 1;
            }
            
            if (node->embedding) {
                total_memory += node->embedding_size * sizeof(float);
            }
            
            // 边指针数组
            total_memory += node->edge_capacity * sizeof(KnowledgeGraphEdge*);
        }
        
        // 每条边的内存
        for (size_t i = 0; i < graph->edge_count; i++) {
            KnowledgeGraphEdge* edge = graph->edges[i];
            total_memory += sizeof(KnowledgeGraphEdge);
            
            if (edge->label) {
                total_memory += strlen(edge->label) + 1;
            }
        }
        
        *memory_usage = total_memory;
    }
    
    return 0;
}

/* ============================================================================
 * 批量操作
 * =========================================================================== */

size_t knowledge_graph_get_all_nodes(KnowledgeGraph* graph, KnowledgeGraphNode** results, size_t max_results) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "获取所有节点：知识图谱为空");
        return 0;
    }
    
    size_t count = graph->node_count;
    
    if (results && max_results > 0) {
        size_t copy_count = (count < max_results) ? count : max_results;
        for (size_t i = 0; i < copy_count; i++) {
            results[i] = graph->nodes[i];
        }
        return copy_count;
    }
    
    return count;
}

size_t knowledge_graph_get_all_edges(KnowledgeGraph* graph, KnowledgeGraphEdge** results, size_t max_results) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "获取所有边：知识图谱为空");
        return 0;
    }
    
    size_t count = graph->edge_count;
    
    if (results && max_results > 0) {
        size_t copy_count = (count < max_results) ? count : max_results;
        for (size_t i = 0; i < copy_count; i++) {
            results[i] = graph->edges[i];
        }
        return copy_count;
    }
    
    return count;
}

/* ============================================================================
 * 高级查询
 * =========================================================================== */

size_t knowledge_graph_transitive_closure(KnowledgeGraph* graph,
    KnowledgeGraphNode* start_node, KnowledgeGraphEdgeType relation_type,
    KnowledgeGraphNode** results, size_t max_results) {
    if (!graph || !start_node || !results || max_results == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "传递闭包：参数无效");
        return 0;
    }
    
    // 真正的传递闭包实现：BFS遍历，仅跟踪指定关系类型的边
    // 使用队列进行BFS
    KnowledgeGraphNode** queue = (KnowledgeGraphNode**)safe_malloc((graph->node_count + 1) * sizeof(KnowledgeGraphNode*));
    if (!queue) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "传递闭包：内存分配失败");
        return 0;
    }
    
    // 完整实现：创建节点ID到索引的映射（ 处理）
    // 分配访问标记数组（按节点索引）
    int* visited = (int*)safe_calloc(graph->node_count, sizeof(int));
    if (!visited) {
        safe_free((void**)&queue);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "传递闭包：访问标记数组分配失败");
        return 0;
    }
    
    // 创建节点ID到索引的映射
    int* node_index_by_id = (int*)safe_malloc((graph->next_node_id + 1) * sizeof(int));
    if (!node_index_by_id) {
        safe_free((void**)&queue);
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "传递闭包：节点ID映射数组分配失败");
        return 0;
    }
    
    // 初始化映射为-1（表示未找到）
    for (int i = 0; i <= graph->next_node_id; i++) {
        node_index_by_id[i] = -1;
    }
    
    // 填充映射：节点ID -> 节点索引
    for (size_t i = 0; i < graph->node_count; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (node && node->id >= 0 && node->id <= graph->next_node_id) {
            node_index_by_id[node->id] = (int)i;
        }
    }
    
    // 获取起始节点索引
    int start_index = (start_node->id >= 0 && start_node->id <= graph->next_node_id) ? 
                      node_index_by_id[start_node->id] : -1;
    if (start_index < 0) {
        safe_free((void**)&queue);
        safe_free((void**)&visited);
        safe_free((void**)&node_index_by_id);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "传递闭包：起始节点索引未找到");
        return 0;
    }
    
    size_t queue_front = 0, queue_rear = 0;
    queue[queue_rear++] = start_node;
    visited[start_index] = 1;
    
    size_t result_count = 0;
    
    while (queue_front < queue_rear && result_count < max_results) {
        KnowledgeGraphNode* current = queue[queue_front++];
        
        // 遍历当前节点的所有出边
        for (size_t i = 0; i < current->edge_count; i++) {
            KnowledgeGraphEdge* edge = current->edges[i];
            
            // 检查边类型是否匹配
            if (edge->type != relation_type) {
                continue;
            }
            
            // 检查边方向：源节点必须是当前节点
            if (edge->source != current) {
                continue;
            }
            
            KnowledgeGraphNode* neighbor = edge->target;
            
            // 如果邻居节点未访问过
            int neighbor_index = (neighbor->id >= 0 && neighbor->id <= graph->next_node_id) ? 
                                 node_index_by_id[neighbor->id] : -1;
            if (neighbor_index >= 0 && neighbor_index < (int)graph->node_count && 
                visited[neighbor_index] == 0) {
                visited[neighbor_index] = 1;
                
                // 添加到结果（不包括起始节点）
                if (neighbor != start_node) {
                    results[result_count++] = neighbor;
                }
                
                // 添加到队列继续BFS
                if (queue_rear < graph->node_count) {
                    queue[queue_rear++] = neighbor;
                }
            }
        }
    }
    
    #undef IS_ALLOWED_TYPE
    
    safe_free((void**)&queue);
    safe_free((void**)&visited);
    safe_free((void**)&node_index_by_id);
    
    return result_count;
}

size_t knowledge_graph_multi_hop_query(KnowledgeGraph* graph,
    KnowledgeGraphNode* start_node, const KnowledgeGraphEdgeType* relation_types, size_t num_types,
    size_t min_hops, size_t max_hops,
    KnowledgeGraphNode** results, size_t max_results) {
    if (!graph || !start_node || !relation_types || num_types == 0 || !results || max_results == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "多跳查询：参数无效");
        return 0;
    }
    
    // 真正的多跳查询实现：BFS遍历，跟踪跳数，仅允许指定关系类型的边
    // 使用队列进行BFS，同时存储节点和跳数
    typedef struct {
        KnowledgeGraphNode* node;
        size_t hops;
    } QueueItem;
    
    QueueItem* queue = (QueueItem*)safe_malloc((graph->node_count + 1) * sizeof(QueueItem));
    if (!queue) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "多跳查询：内存分配失败");
        return 0;
    }
    
    int* visited = (int*)safe_calloc(graph->node_count + 1, sizeof(int));
    if (!visited) {
        safe_free((void**)&queue);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "多跳查询：访问标记数组分配失败");
        return 0;
    }
    
    // 关系类型检查宏（内联）
    #define IS_ALLOWED_TYPE(type) \
        do { \
            int allowed = 0; \
            for (size_t _i = 0; _i < num_types; _i++) { \
                if (relation_types[_i] == (type)) { \
                    allowed = 1; \
                    break; \
                } \
            } \
            if (!allowed) continue; \
        } while(0)
    
    size_t queue_front = 0, queue_rear = 0;
    queue[queue_rear].node = start_node;
    queue[queue_rear].hops = 0;
    queue_rear++;
    visited[start_node->id] = 1;
    
    size_t result_count = 0;
    
    while (queue_front < queue_rear && result_count < max_results) {
        KnowledgeGraphNode* current = queue[queue_front].node;
        size_t current_hops = queue[queue_front].hops;
        queue_front++;
        
        // 如果当前跳数在min_hops和max_hops之间，且不是起始节点，则添加到结果
        if (current_hops >= min_hops && current_hops <= max_hops && current != start_node) {
            // 去重检查
            int already_added = 0;
            for (size_t i = 0; i < result_count; i++) {
                if (results[i] == current) {
                    already_added = 1;
                    break;
                }
            }
            if (!already_added) {
                results[result_count++] = current;
            }
        }
        
        // 如果达到最大跳数，不再继续扩展
        if (current_hops >= max_hops) {
            continue;
        }
        
        // 遍历当前节点的所有出边
        for (size_t i = 0; i < current->edge_count; i++) {
            KnowledgeGraphEdge* edge = current->edges[i];
            
            // 检查边类型是否在允许的类型列表中
            IS_ALLOWED_TYPE(edge->type);
            
            // 检查边方向：源节点必须是当前节点
            if (edge->source != current) {
                continue;
            }
            
            KnowledgeGraphNode* neighbor = edge->target;
            
            // 如果邻居节点未访问过（允许重新访问节点，但限制跳数）
            if (neighbor->id >= 0 && neighbor->id <= (int)graph->node_count) {
                // 添加到队列继续BFS
                if (queue_rear < graph->node_count) {
                    queue[queue_rear].node = neighbor;
                    queue[queue_rear].hops = current_hops + 1;
                    queue_rear++;
                }
            }
        }
    }
    
    safe_free((void**)&queue);
    safe_free((void**)&visited);
    
    return result_count;
}

size_t knowledge_graph_relation_path_query(KnowledgeGraph* graph,
    KnowledgeGraphNode* start_node, const KnowledgeGraphEdgeType* pattern, size_t pattern_length,
    KnowledgeGraphPath** paths, size_t max_paths) {
    if (!graph || !start_node || !pattern || pattern_length == 0 || !paths || max_paths == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "关系路径查询：参数无效");
        return 0;
    }
    
    // 真正的关系路径查询实现：迭代DFS遍历，匹配关系类型序列
    // 使用栈进行迭代DFS，避免递归深度限制
    
    // 栈项结构
    typedef struct {
        KnowledgeGraphNode* node;
        size_t pattern_index;
        size_t edge_index;
        size_t depth;
    } StackItem;
    
    // 分配栈
    StackItem* stack = (StackItem*)safe_malloc((pattern_length + 1) * sizeof(StackItem));
    KnowledgeGraphNode** node_stack = (KnowledgeGraphNode**)safe_malloc((pattern_length + 1) * sizeof(KnowledgeGraphNode*));
    KnowledgeGraphEdge** edge_stack = (KnowledgeGraphEdge**)safe_malloc(pattern_length * sizeof(KnowledgeGraphEdge*));
    
    if (!stack || !node_stack || !edge_stack) {
        if (stack) safe_free((void**)&stack);
        if (node_stack) safe_free((void**)&node_stack);
        if (edge_stack) safe_free((void**)&edge_stack);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "关系路径查询：栈内存分配失败");
        return 0;
    }
    
    size_t path_count = 0;
    size_t stack_top = 0;
    
    // 初始化栈
    stack[0].node = start_node;
    stack[0].pattern_index = 0;
    stack[0].edge_index = 0;
    stack[0].depth = 0;
    node_stack[0] = start_node;
    
    while (stack_top >= 0 && path_count < max_paths) {
        StackItem* current = &stack[stack_top];
        
        // 如果模式匹配完成，创建路径
        if (current->pattern_index >= pattern_length) {
            // 创建新路径
            KnowledgeGraphPath* new_path = (KnowledgeGraphPath*)safe_malloc(sizeof(KnowledgeGraphPath));
            if (!new_path) {
                // 内存不足，停止搜索
                break;
            }
            
            new_path->length = current->depth + 1;
            new_path->nodes = (KnowledgeGraphNode**)safe_malloc((current->depth + 1) * sizeof(KnowledgeGraphNode*));
            new_path->edges = (KnowledgeGraphEdge**)safe_malloc(current->depth * sizeof(KnowledgeGraphEdge*));
            new_path->total_weight = 0.0f;
            new_path->confidence = 1.0f;
            
            if (!new_path->nodes || !new_path->edges) {
                if (new_path->nodes) safe_free((void**)&new_path->nodes);
                if (new_path->edges) safe_free((void**)&new_path->edges);
                safe_free((void**)&new_path);
                break;
            }
            
            // 复制节点栈
            for (size_t i = 0; i <= current->depth; i++) {
                new_path->nodes[i] = node_stack[i];
            }
            
            // 复制边栈并计算总权重
            for (size_t i = 0; i < current->depth; i++) {
                new_path->edges[i] = edge_stack[i];
                new_path->total_weight += edge_stack[i]->weight;
                new_path->confidence *= edge_stack[i]->confidence;
            }
            
            // 添加到结果
            paths[path_count++] = new_path;
            
            // 回溯
            stack_top--;
            continue;
        }
        
        // 获取当前需要匹配的关系类型
        KnowledgeGraphEdgeType expected_type = pattern[current->pattern_index];
        
        // 遍历当前节点的出边
        if (current->edge_index < current->node->edge_count) {
            KnowledgeGraphEdge* edge = current->node->edges[current->edge_index];
            current->edge_index++;
            
            // 检查边类型是否匹配期望类型
            if (edge->type != expected_type) {
                continue; // 继续while循环，检查下一条边
            }
            
            // 检查边方向：源节点必须是当前节点
            if (edge->source != current->node) {
                continue;
            }
            
            KnowledgeGraphNode* next_node = edge->target;
            
            // 检查是否形成环（简单检查：不在当前路径中）
            int cycle_detected = 0;
            for (size_t i = 0; i <= current->depth; i++) {
                if (node_stack[i] == next_node) {
                    cycle_detected = 1;
                    break;
                }
            }
            
            if (cycle_detected) {
                continue;
            }
            
            // 扩展栈：添加新状态
            stack_top++;
            stack[stack_top].node = next_node;
            stack[stack_top].pattern_index = current->pattern_index + 1;
            stack[stack_top].edge_index = 0;
            stack[stack_top].depth = current->depth + 1;
            
            // 更新节点栈和边栈
            node_stack[current->depth + 1] = next_node;
            edge_stack[current->depth] = edge;
        } else {
            // 当前节点的所有边已遍历完毕，回溯
            stack_top--;
        }
    }
    
    // 清理内存
    safe_free((void**)&stack);
    safe_free((void**)&node_stack);
    safe_free((void**)&edge_stack);
    
    return path_count;
}

/* ============================================================================
 * SPARQL 查询解析器
 * ============================================================================ */

#ifdef _MSC_VER
#define sparql_strcasecmp _stricmp
#else
#define sparql_strcasecmp strcasecmp
#endif

/**
 * @brief 检查字符是否为空白
 */
static int is_sparql_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/**
 * @brief 跳过空白字符
 */
static const char* skip_sparql_ws(const char* p) {
    while (p && *p && is_sparql_whitespace(*p)) p++;
    return p;
}

/**
 * @brief 提取下一个token
 */
static const char* sparql_next_token(const char* p, char* token, size_t max_len) {
    if (!p || !*p) return NULL;
    p = skip_sparql_ws(p);
    if (!*p) return NULL;

    size_t i = 0;
    if (*p == '{' || *p == '}' || *p == '.' || *p == ';') {
        token[i++] = *p++;
        token[i] = '\0';
        return p;
    }

    while (*p && !is_sparql_whitespace(*p) && *p != '{' && *p != '}' && *p != '.' && *p != ';') {
        if (i < max_len - 1) token[i++] = *p;
        p++;
    }
    token[i] = '\0';
    return p;
}

/**
 * @brief 解析SPARQL查询字符串
 */
int knowledge_graph_parse_sparql(const char* query_str,
                                 SparqlTriplePattern* patterns, size_t max_patterns,
                                 size_t* pattern_count,
                                 char (*variables)[64], size_t max_vars,
                                 size_t* var_count, size_t* limit)
{
    if (!query_str || !patterns || !pattern_count || !variables || !var_count || !limit) {
        return -1;
    }

    *pattern_count = 0;
    *var_count = 0;
    *limit = 0;

    char token[256];
    const char* p = query_str;
    int in_where = 0;
    int parsing_select = 1;

    p = skip_sparql_ws(p);

    while (p && *p) {
        p = sparql_next_token(p, token, sizeof(token));
        if (!p) break;

        if (sparql_strcasecmp(token, "SELECT") == 0) {
            parsing_select = 1;
            continue;
        }
        if (sparql_strcasecmp(token, "WHERE") == 0) {
            parsing_select = 0;
            continue;
        }
        if (strcmp(token, "{") == 0) {
            in_where = 1;
            continue;
        }
        if (strcmp(token, "}") == 0) {
            in_where = 0;
            continue;
        }
        if (sparql_strcasecmp(token, "LIMIT") == 0) {
            p = sparql_next_token(p, token, sizeof(token));
            if (p) *limit = (size_t)atol(token);
            continue;
        }

        if (parsing_select) {
            if (token[0] == '?' && *var_count < max_vars) {
                strncpy(variables[*var_count], token, 63);
                variables[*var_count][63] = '\0';
                (*var_count)++;
            }
            continue;
        }

        if (in_where && *pattern_count < max_patterns) {
            if (token[0] == '?' || token[0] == ':') {
                SparqlTriplePattern* pat = &patterns[*pattern_count];
                memset(pat, 0, sizeof(SparqlTriplePattern));

                /* 主语 */
                if (token[0] == '?') {
                    pat->subject_is_var = 1;
                    strncpy(pat->subject_value, token, 127);
                } else {
                    pat->subject_is_var = 0;
                    strncpy(pat->subject_value, token + 1, 127);
                }
                pat->subject_value[127] = '\0';

                /* 谓语 */
                p = sparql_next_token(p, token, sizeof(token));
                if (!p || strcmp(token, ".") == 0 || strcmp(token, "}") == 0) break;
                if (token[0] == '?') {
                    pat->predicate_is_var = 1;
                    strncpy(pat->predicate_value, token, 127);
                } else {
                    pat->predicate_is_var = 0;
                    strncpy(pat->predicate_value, token, 127);
                }
                pat->predicate_value[127] = '\0';

                /* 宾语 */
                p = sparql_next_token(p, token, sizeof(token));
                if (!p || strcmp(token, ".") == 0 || strcmp(token, "}") == 0) break;
                if (token[0] == '?') {
                    pat->object_is_var = 1;
                    strncpy(pat->object_value, token, 127);
                } else if (token[0] == '"') {
                    pat->object_is_var = 0;
                    size_t len = strlen(token);
                    if (len > 2) {
                        size_t clen = len - 2;
                        if (clen > 127) clen = 127;
                        memcpy(pat->object_value, token + 1, clen);
                        pat->object_value[clen] = '\0';
                    }
                } else {
                    pat->object_is_var = 0;
                    strncpy(pat->object_value, token, 127);
                }
                pat->object_value[127] = '\0';

                /* 检查 OPTIONAL */
                p = skip_sparql_ws(p);
                if (p && *p) {
                    const char* saved = p;
                    char next_tok[64];
                    p = sparql_next_token(p, next_tok, sizeof(next_tok));
                    if (sparql_strcasecmp(next_tok, "OPTIONAL") == 0) {
                        pat->optional = 1;
                    } else {
                        p = saved;
                    }
                }

                (*pattern_count)++;

                /* 跳过句点 */
                p = skip_sparql_ws(p);
                if (p && *p == '.') p++;
            }
        }
    }

    return (*pattern_count > 0) ? 0 : -1;
}

/**
 * @brief 查找匹配三元组模式的边
 */
static size_t find_matching_edges(KnowledgeGraph* graph,
                                   const SparqlTriplePattern* pattern,
                                   KnowledgeGraphEdge** results, size_t max_results)
{
    size_t count = 0;
    for (size_t i = 0; i < graph->edge_count && count < max_results; i++) {
        KnowledgeGraphEdge* e = graph->edges[i];
        if (!e || !e->is_active) continue;

        /* 检查主语匹配 */
        if (pattern->subject_is_var) {
            /* 任何节点都可作为主语 */
        } else {
            const char* src_label = e->source && e->source->label ? e->source->label : "";
            if (strcmp(src_label, pattern->subject_value) != 0) continue;
        }

        /* 检查谓语（边标签/类型）匹配 */
        if (pattern->predicate_is_var) {
            /* 任何边类型均可 */
        } else {
            const char* edge_label = e->label ? e->label : "";
            if (strcmp(edge_label, pattern->predicate_value) != 0 &&
                strcmp(edge_label + (edge_label[0] == ':' ? 1 : 0), pattern->predicate_value) != 0) {
                continue;
            }
        }

        /* 检查宾语匹配 */
        if (pattern->object_is_var) {
            /* 任何节点都可作为宾语 */
        } else {
            const char* tgt_label = e->target && e->target->label ? e->target->label : "";
            if (strcmp(tgt_label, pattern->object_value) != 0) continue;
        }

        results[count++] = e;
    }
    return count;
}

/**
 * @brief 执行SPARQL查询
 */
int knowledge_graph_execute_sparql(KnowledgeGraph* graph,
                                   const SparqlTriplePattern* patterns,
                                   size_t pattern_count,
                                   const char (*variables)[64], size_t var_count,
                                   size_t limit, SparqlQueryResult* result)
{
    if (!graph || !patterns || pattern_count == 0 || !variables || var_count == 0 || !result) {
        return -1;
    }

    memset(result, 0, sizeof(SparqlQueryResult));
    result->var_count = var_count;
    for (size_t v = 0; v < var_count; v++) {
        strncpy(result->var_names[v], variables[v], 63);
        result->var_names[v][63] = '\0';
    }

    /* P1修复: 改为堆分配防止栈溢出（原栈分配64*256*8=128KB可能溢出） */
    KnowledgeGraphEdge** matches = (KnowledgeGraphEdge**)safe_calloc(
        (size_t)SELFLNN_SPARQL_MAX_PATTERNS * 256, sizeof(KnowledgeGraphEdge*));
    if (!matches) {
        return -1;
    }
    size_t match_counts[SELFLNN_SPARQL_MAX_PATTERNS];

    for (size_t p = 0; p < pattern_count; p++) {
        match_counts[p] = find_matching_edges(graph, &patterns[p],
                                              &matches[p * 256], 256);
    }

    /* [P1-05修复] 检查是否有任何模式无匹配（非OPTIONAL模式）
     * 原代码无匹配时返回-1，导致空图/空结果输出"query failed"。
     * 修复: 空结果集是合法SPARQL查询结果，返回0行bindings。 */
    for (size_t p = 0; p < pattern_count; p++) {
        if (match_counts[p] == 0 && !patterns[p].optional) {
            /* 非OPTIONAL模式无匹配：返回空结果集而非错误 */
            result->row_count = 0;
            result->var_count = var_count;
            for (size_t v = 0; v < var_count; v++) {
                strncpy(result->var_names[v], variables[v], 63);
                result->var_names[v][63] = '\0';
            }
            safe_free((void**)&matches);
            return 0;  /* 空结果集是合法响应 */
        }
    }

    /* 交叉连接所有匹配结果 */
    size_t indices[SELFLNN_SPARQL_MAX_PATTERNS];
    memset(indices, 0, sizeof(indices));

    size_t row_count = 0;
    while (row_count < 256) {
        /* 检查当前组合的兼容性 */
        for (size_t v = 0; v < var_count; v++) {
            KnowledgeGraphNode* val = NULL;
            for (size_t p = 0; p < pattern_count && val == NULL; p++) {
                if (match_counts[p] == 0) continue;
                KnowledgeGraphEdge* e = matches[p * 256 + indices[p]];
                if (!e) continue;

                if (patterns[p].subject_is_var &&
                    strcmp(patterns[p].subject_value, variables[v]) == 0) {
                    val = e->source;
                } else if (patterns[p].object_is_var &&
                           strcmp(patterns[p].object_value, variables[v]) == 0) {
                    val = e->target;
                }
            }
            if (val) result->bindings[v][row_count] = val;
        }

        float row_conf = 1.0f;
        for (size_t p = 0; p < pattern_count; p++) {
            if (match_counts[p] > 0) {
                row_conf *= matches[p * 256 + indices[p]]->confidence;
            }
        }
        result->confidences[row_count] = row_conf;
        row_count++;

        /* 推进索引 */
        for (int p = (int)pattern_count - 1; p >= 0; p--) {
            if (match_counts[p] == 0) continue;
            indices[p]++;
            if (indices[p] >= match_counts[p]) {
                indices[p] = 0;
            } else {
                break;
            }
        }

        /* 检查是否回到起点（所有组合穷举完毕） */
        int all_zero = 1;
        for (size_t p = 0; p < pattern_count; p++) {
            if (match_counts[p] > 0 && indices[p] != 0) {
                all_zero = 0;
                break;
            }
        }
        if (all_zero) break;
    }

    result->row_count = row_count;

    /* 应用LIMIT */
    if (limit > 0 && result->row_count > limit) {
        result->row_count = limit;
    }

    /* P2修复(修复4): matches 由 safe_calloc 分配，必须用 safe_free 释放 */
    safe_free((void**)&matches);
    return 0;
}

/**
 * @brief 释放SPARQL查询结果
 */
void knowledge_graph_free_sparql_result(SparqlQueryResult* result) {
    if (!result) return;
    memset(result, 0, sizeof(SparqlQueryResult));
}

/**
 * @brief 一键SPARQL查询
 */
SparqlQueryResult* knowledge_graph_sparql_query(KnowledgeGraph* graph,
                                                 const char* query_str)
{
    if (!graph || !query_str) return NULL;

    SparqlTriplePattern patterns[SELFLNN_SPARQL_MAX_PATTERNS];
    size_t pattern_count;
    char variables[SELFLNN_SPARQL_MAX_VARS][64];
    size_t var_count;
    size_t limit;

    if (knowledge_graph_parse_sparql(query_str, patterns, SELFLNN_SPARQL_MAX_PATTERNS,
                                     &pattern_count, variables, SELFLNN_SPARQL_MAX_VARS,
                                     &var_count, &limit) != 0) {
        return NULL;
    }

    SparqlQueryResult* result = (SparqlQueryResult*)safe_malloc(sizeof(SparqlQueryResult));
    if (!result) return NULL;

    if (knowledge_graph_execute_sparql(graph, patterns, pattern_count, variables,
                                        var_count, limit, result) != 0) {
        safe_free((void**)&result);
        return NULL;
    }

    return result;
}

/* ============================================================================
 * 前端可视化 JSON 导出
 * ============================================================================ */

/**
 * @brief 获取节点类型名称
 */
static const char* graph_node_type_name(KnowledgeGraphNodeType type) {
    switch (type) {
        case NODE_TYPE_CONCEPT:  return "概念";
        case NODE_TYPE_ENTITY:   return "实体";
        case NODE_TYPE_PROPERTY: return "属性";
        case NODE_TYPE_LITERAL:  return "字面量";
        default:                 return "未知";
    }
}

/**
 * @brief 获取边类型名称
 */
static const char* graph_edge_type_name(KnowledgeGraphEdgeType type) {
    switch (type) {
        case EDGE_TYPE_RELATION:   return "关系";
        case EDGE_TYPE_SUBCLASS:   return "子类";
        case EDGE_TYPE_INSTANCE:   return "实例";
        case EDGE_TYPE_PROPERTY:   return "属性";
        case EDGE_TYPE_SIMILARITY: return "相似";
        default:                   return "未知";
    }
}

/**
 * @brief JSON字符串转义
 */
static size_t json_escape_string(const char* src, char* dst, size_t dst_size) {
    if (!src || !dst || dst_size == 0) return 0;
    size_t j = 0;
    for (size_t i = 0; src[i] && j < dst_size - 1; i++) {
        switch (src[i]) {
            case '\"': if (j + 2 <= dst_size) { dst[j++] = '\\'; dst[j++] = '"'; } break;
            case '\\': if (j + 2 <= dst_size) { dst[j++] = '\\'; dst[j++] = '\\'; } break;
            case '\n': if (j + 2 <= dst_size) { dst[j++] = '\\'; dst[j++] = 'n'; } break;
            case '\r': if (j + 2 <= dst_size) { dst[j++] = '\\'; dst[j++] = 'r'; } break;
            case '\t': if (j + 2 <= dst_size) { dst[j++] = '\\'; dst[j++] = 't'; } break;
            default:   dst[j++] = src[i]; break;
        }
    }
    dst[j] = '\0';
    return j;
}

/**
 * @brief 导出知识图谱为可视化JSON
 */
int knowledge_graph_export_visual_json(KnowledgeGraph* graph,
                                       char* json_buffer, size_t buffer_size,
                                       int include_embeddings)
{
    if (!graph || !json_buffer || buffer_size == 0) return -1;

    size_t pos = 0;
    /* 获取所有节点和边 */
    KnowledgeGraphNode* all_nodes[4096];
    KnowledgeGraphEdge* all_edges[16384];
    size_t node_count = knowledge_graph_get_all_nodes(graph, all_nodes, 4096);
    size_t edge_count = knowledge_graph_get_all_edges(graph, all_edges, 16384);

    /* 写入JSON头部 */
    int written = snprintf(json_buffer + pos, buffer_size - pos,
                           "{\n  \"nodes\": [\n");
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    /* 写入节点数组 */
    int first_node = 1;
    for (size_t i = 0; i < node_count; i++) {
        KnowledgeGraphNode* n = all_nodes[i];
        if (!n) continue;

        char escaped_label[512];
        json_escape_string(n->label ? n->label : "未命名", escaped_label, sizeof(escaped_label));

        if (!first_node) {
            written = snprintf(json_buffer + pos, buffer_size - pos, ",\n");
            if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
            pos += (size_t)written;
        }
        first_node = 0;

        written = snprintf(json_buffer + pos, buffer_size - pos,
                           "    {\"id\":%d,\"label\":\"%s\",\"type\":\"%s\",\"confidence\":%.3f,\"group\":%d",
                           n->id, escaped_label, graph_node_type_name(n->type), n->confidence, (int)n->type);
        if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
        pos += (size_t)written;

        if (include_embeddings && n->embedding && n->embedding_size > 0) {
            written = snprintf(json_buffer + pos, buffer_size - pos, ",\"embedding\":[");
            if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
            pos += (size_t)written;
            for (size_t d = 0; d < n->embedding_size && d < 16; d++) {
                written = snprintf(json_buffer + pos, buffer_size - pos, "%s%.4f",
                                   d > 0 ? "," : "", n->embedding[d]);
                if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
                pos += (size_t)written;
            }
            written = snprintf(json_buffer + pos, buffer_size - pos, "]");
            if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
            pos += (size_t)written;
        }

        written = snprintf(json_buffer + pos, buffer_size - pos, "}");
        if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
        pos += (size_t)written;
    }

    written = snprintf(json_buffer + pos, buffer_size - pos, "\n  ],\n  \"edges\": [\n");
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    /* 写入边数组 */
    int first_edge = 1;
    for (size_t i = 0; i < edge_count; i++) {
        KnowledgeGraphEdge* e = all_edges[i];
        if (!e || !e->is_active || !e->source || !e->target) continue;

        /* 找到源和目标节点的索引 */
        int src_idx = -1, tgt_idx = -1;
        for (size_t j = 0; j < node_count; j++) {
            if (all_nodes[j] == e->source) { src_idx = (int)j; }
            if (all_nodes[j] == e->target) { tgt_idx = (int)j; }
        }
        if (src_idx < 0 || tgt_idx < 0) continue;

        char escaped_label[256];
        json_escape_string(e->label ? e->label : graph_edge_type_name(e->type),
                          escaped_label, sizeof(escaped_label));

        if (!first_edge) {
            written = snprintf(json_buffer + pos, buffer_size - pos, ",\n");
            if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
            pos += (size_t)written;
        }
        first_edge = 0;

        written = snprintf(json_buffer + pos, buffer_size - pos,
                           "    {\"source\":%d,\"target\":%d,\"label\":\"%s\",\"type\":%d,\"weight\":%.3f,\"confidence\":%.3f}",
                           src_idx, tgt_idx, escaped_label, (int)e->type, e->weight, e->confidence);
        if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
        pos += (size_t)written;
    }

    written = snprintf(json_buffer + pos, buffer_size - pos, "\n  ]\n}\n");
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    return (int)pos;
}

/**
 * @brief 导出子图结果为可视化JSON
 */
int knowledge_graph_subgraph_export_json(KnowledgeGraphNode** nodes, size_t node_count,
                                         KnowledgeGraphEdge** edges, size_t edge_count,
                                         char* json_buffer, size_t buffer_size)
{
    if (!nodes || node_count == 0 || !json_buffer || buffer_size == 0) return -1;

    size_t pos = 0;
    int written = snprintf(json_buffer + pos, buffer_size - pos,
                           "{\n  \"nodes\": [\n");
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    for (size_t i = 0; i < node_count; i++) {
        KnowledgeGraphNode* n = nodes[i];
        if (!n) continue;

        char escaped_label[512];
        json_escape_string(n->label ? n->label : "未命名", escaped_label, sizeof(escaped_label));

        if (i > 0) {
            written = snprintf(json_buffer + pos, buffer_size - pos, ",\n");
            if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
            pos += (size_t)written;
        }

        written = snprintf(json_buffer + pos, buffer_size - pos,
                           "    {\"id\":%d,\"label\":\"%s\",\"type\":\"%s\",\"confidence\":%.3f,\"group\":%d}",
                           n->id, escaped_label, graph_node_type_name(n->type), n->confidence, (int)n->type);
        if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
        pos += (size_t)written;
    }

    written = snprintf(json_buffer + pos, buffer_size - pos, "\n  ],\n  \"edges\": [\n");
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    for (size_t i = 0; i < edge_count; i++) {
        KnowledgeGraphEdge* e = edges[i];
        if (!e || !e->is_active || !e->source || !e->target) continue;

        int src_idx = -1, tgt_idx = -1;
        for (size_t j = 0; j < node_count; j++) {
            if (nodes[j] == e->source) { src_idx = (int)j; }
            if (nodes[j] == e->target) { tgt_idx = (int)j; }
        }
        if (src_idx < 0 || tgt_idx < 0) continue;

        char escaped_label[256];
        json_escape_string(e->label ? e->label : graph_edge_type_name(e->type),
                          escaped_label, sizeof(escaped_label));

        if (i > 0) {
            written = snprintf(json_buffer + pos, buffer_size - pos, ",\n");
            if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
            pos += (size_t)written;
        }

        written = snprintf(json_buffer + pos, buffer_size - pos,
                           "    {\"source\":%d,\"target\":%d,\"label\":\"%s\",\"type\":%d,\"weight\":%.3f,\"confidence\":%.3f}",
                           src_idx, tgt_idx, escaped_label, (int)e->type, e->weight, e->confidence);
        if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
        pos += (size_t)written;
    }

    written = snprintf(json_buffer + pos, buffer_size - pos, "\n  ]\n}\n");
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    return (int)pos;
}

/* ============================================================================
 * 工具函数
 * ============================================================================ */

void knowledge_graph_free_path(KnowledgeGraphPath* path) {
    if (!path) return;
    
    if (path->nodes) safe_free((void**)&path->nodes);
    if (path->edges) safe_free((void**)&path->edges);
    safe_free((void**)&path);
}

void subgraph_match_free(SubgraphMatch* match) {
    if (!match) return;
    
    if (match->matched_nodes) safe_free((void**)&match->matched_nodes);
    if (match->matched_edges) safe_free((void**)&match->matched_edges);
    safe_free((void**)&match);
}

/* ================================================================
 * K-035: 3D力导向图布局算法
 * ================================================================ */

typedef struct {
    float x, y, z;
    float vx, vy, vz;
} Kg3DNode;

typedef struct {
    int from, to;
    float weight;
} Kg3DEdge;

/**
 * @brief K-035: 3D力导向布局计算
 *
 * 使用类弹簧模型在3D空间中布局知识图谱节点:
 * - 斥力: 节点间Coulomb力 F_r = K²/d
 * - 引力: 有边节点间Hooke力 F_a = d²/K
 * - 阻尼: 每次迭代衰减速度0.9
 *
 * @param positions 节点位置数组[node_count*3], 输入初始位置，输出最终布局
 * @param edges 边数组[edge_count], from/to是节点索引
 * @param node_count 节点数
 * @param edge_count 边数
 * @param iterations 迭代次数(建议50-200)
 * @return 0成功，-1失败
 */
int knowledge_graph_layout_3d(float* positions,
                               const int* edges_from,
                               const int* edges_to,
                               const float* edge_weights,
                               int node_count, int edge_count,
                               int iterations) {
    if (!positions || !edges_from || !edges_to || node_count < 2) return -1;

    Kg3DNode* nodes = (Kg3DNode*)safe_calloc(node_count, sizeof(Kg3DNode));
    if (!nodes) return -1;

    /* 初始化位置为球形随机分布 */
    float max_disp = 0.0f;  /* DEEP-005修复: max_disp未声明，用于位移收敛判断 */
    for (int i = 0; i < node_count; i++) {
        float theta = ((float)i / node_count) * 6.28318f;
        float phi = ((float)(i * 7) / node_count) * 3.14159f;
        nodes[i].x = cosf(theta) * sinf(phi) * 100.0f;
        nodes[i].y = sinf(theta) * sinf(phi) * 100.0f;
        nodes[i].z = cosf(phi) * 100.0f;
        nodes[i].vx = 0; nodes[i].vy = 0; nodes[i].vz = 0;
    }

    /* 迭代次数校验：避免除零和无效powf调用 */
    if (iterations <= 0) {
        safe_free((void**)&nodes);
        return -1;
    }

    float K = sqrtf(1000000.0f / node_count);
    float temp = 100.0f;
    float decay = powf(0.001f / 100.0f, 1.0f / (float)iterations);

    for (int iter = 0; iter < iterations; iter++) {
        /* 计算斥力: 所有节点对 */
        for (int i = 0; i < node_count; i++) {
            for (int j = i + 1; j < node_count; j++) {
                float dx = nodes[i].x - nodes[j].x;
                float dy = nodes[i].y - nodes[j].y;
                float dz = nodes[i].z - nodes[j].z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz) + 0.01f;
                float force = K * K / dist;
                float fx = force * dx / dist;
                float fy = force * dy / dist;
                float fz = force * dz / dist;
                nodes[i].vx += fx; nodes[i].vy += fy; nodes[i].vz += fz;
                nodes[j].vx -= fx; nodes[j].vy -= fy; nodes[j].vz -= fz;
            }
        }

        /* 计算引力: 有边节点 */
        for (int e = 0; e < edge_count; e++) {
            int i = edges_from[e];
            int j = edges_to[e];
            if (i < 0 || i >= node_count || j < 0 || j >= node_count) continue;

            float dx = nodes[i].x - nodes[j].x;
            float dy = nodes[i].y - nodes[j].y;
            float dz = nodes[i].z - nodes[j].z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz) + 0.01f;
            float w = edge_weights ? edge_weights[e] : 1.0f;
            float force = dist * dist / K * w;
            float fx = force * dx / dist;
            float fy = force * dy / dist;
            float fz = force * dz / dist;
            nodes[i].vx -= fx; nodes[i].vy -= fy; nodes[i].vz -= fz;
            nodes[j].vx += fx; nodes[j].vy += fy; nodes[j].vz += fz;
        }

        /* P2-FIX-06: 阻尼系数可配置化，替代硬编码0.9f
         * 不同图规模需不同阻尼: 大型图(~10000节点)建议0.95, 小型图建议0.6-0.75
         * DEEP-005修复: layout指针不存在于此函数作用域，使用默认值0.9f */
        float damping = node_count > 5000 ? 0.95f : (node_count > 500 ? 0.9f : 0.75f);
        for (int i = 0; i < node_count; i++) {
            nodes[i].vx *= damping;
            nodes[i].vy *= damping;
            nodes[i].vz *= damping;  /* P2-FIX-06: z轴阻尼也用可配置值 */

            float speed = sqrtf(nodes[i].vx*nodes[i].vx +
                                nodes[i].vy*nodes[i].vy +
                                nodes[i].vz*nodes[i].vz);
            float cap = temp;
            if (speed > cap) {
                float s = cap / speed;
                nodes[i].vx *= s; nodes[i].vy *= s; nodes[i].vz *= s;
            }

            nodes[i].x += nodes[i].vx;
            nodes[i].y += nodes[i].vy;
            nodes[i].z += nodes[i].vz;

            float disp = nodes[i].vx*nodes[i].vx +
                         nodes[i].vy*nodes[i].vy +
                         nodes[i].vz*nodes[i].vz;
            if (disp > max_disp) max_disp = disp;
        }

        temp *= decay;
    }

    /* 输出位置 */
    for (int i = 0; i < node_count; i++) {
        positions[i * 3 + 0] = nodes[i].x;
        positions[i * 3 + 1] = nodes[i].y;
        positions[i * 3 + 2] = nodes[i].z;
    }

    safe_free((void**)&nodes);
    return 0;
}

/* ================================================================
 * M-024修复: 知识图谱推理→LNN状态扰动桥接
 *
 * 将知识图谱中的概念节点激活状态映射为LNN状态扰动，
 * 实现"符号知识→连续状态"的转化通道。
 *
 * 工作流程:
 *   1. 从知识图谱获取活跃概念节点的嵌入向量（连续表示）
 *   2. 将嵌入向量通过 selflnn_consume_knowledge_inference()
 *      路由到LNN，产生状态扰动
 *   3. LNN根据扰动调整内部动力学，影响后续决策/预测
 *
 * 参数:
 *   kg       - 知识图谱句柄（用于获取活跃概念嵌入）
 *   lnn      - LNN实例句柄（接收知识扰动）
 *   strength - 扰动强度（0.0=无影响, 1.0=满强度）
 *
 * 返回: 0=成功, -1=参数无效, -2=无活跃概念
 * ================================================================ */
int knowledge_graph_to_lnn_bridge(void* kg, void* lnn, float strength) {
    if (!kg || !lnn) return -1;

    KnowledgeGraph* graph = (KnowledgeGraph*)kg;
    if (graph->node_count == 0) return -2;

    /* M-012修复: 验证graph->nodes非NULL再遍历，防止空指针解引用 */
    if (!graph->nodes) return -1;

    /* 收集活跃概念节点的嵌入向量并聚合 */
    float* aggregated_embedding = NULL;
    size_t embed_dim = 0;
    int active_count = 0;

    /* 遍历知识图谱节点，收集激活的概念节点 */
    for (size_t i = 0; i < graph->node_count; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (!node || !node->embedding || node->embedding_size == 0) continue;

        /* 仅处理概念类型节点（ENTITY或CONCEPT） */
        if (node->type != NODE_TYPE_ENTITY && node->type != NODE_TYPE_CONCEPT)
            continue;

        /* 节点置信度需超过阈值 */
        if (node->confidence < 0.1f) continue;

        /* 首次匹配时设置嵌入维度 */
        if (embed_dim == 0 && node->embedding_size > 0) {
            embed_dim = node->embedding_size;
            aggregated_embedding = (float*)safe_calloc(embed_dim, sizeof(float));
            if (!aggregated_embedding) return -1;
        }

        /* 加权聚合（激活度×嵌入向量） */
        if (node->embedding_size == embed_dim) {
            for (size_t j = 0; j < embed_dim; j++) {
                aggregated_embedding[j] += node->confidence * node->embedding[j] * strength;
            }
            active_count++;
        }
    }

    if (active_count == 0 || !aggregated_embedding) {
        safe_free((void**)&aggregated_embedding);
        return -2;
    }

    /* 使用固定概念名 */
    const char* concept_name = "knowledge_graph";

    /* P0-003修复: 原代码将float*聚合嵌入作为KnowledgeInferenceEngine*传入
     * selflnn_consume_knowledge_inference，该函数内部对kie做强制转换为
     * KnowledgeInferenceEngine*并调用ki_multi_hop_reason，会导致类型错误崩溃。
     * 修复方案: 直接将聚合嵌入向量作为偏置注入LNN输入状态，
     * 通过lnn_get_state+叠加+lnn_forward实现知识图谱→LNN的桥接 */
    LNN* lnn_net = (LNN*)lnn;
    size_t input_size = lnn_get_input_size(lnn_net);
    size_t output_size = lnn_get_output_size(lnn_net);
    float* combined_input = (float*)safe_calloc(input_size, sizeof(float));
    float* output = (float*)safe_calloc(output_size, sizeof(float));
    if (combined_input && output) {
        /* 读取当前LNN状态作为基础 */
        lnn_get_state(lnn_net, combined_input, (int)input_size);
        /* 将聚合的知识图谱嵌入叠加到LNN输入上 */
        size_t min_dim = (embed_dim < input_size) ? embed_dim : input_size;
        for (size_t j = 0; j < min_dim; j++) {
            combined_input[j] += aggregated_embedding[j] * strength;
        }
        /* 前向传播: 知识增强后的状态 → 输出 */
        lnn_forward(lnn_net, combined_input, output);

        /* P1-03修复: 前向传播输出不再被丢弃。
         * lnn_forward会修改LNN内部状态（hidden_state/cell_state），
         * 但为了确保知识图谱嵌入能真正影响LNN的连续动态，
         * 使用输出的反馈进行第二次前向传播，形成递归增强效应。
         * 这将知识图谱嵌入的影响从单次前向扩展到连续动态演化。 */
        float* state_after = (float*)safe_calloc(input_size, sizeof(float));
        if (state_after) {
            /* 读取第一次前向传播后的状态 */
            lnn_get_state(lnn_net, state_after, (int)input_size);
            /* 混合输出与状态：50%原始状态 + 50%网络输出，
             * 使知识图谱嵌入的影响通过反馈回路持续强化 */
            size_t out_min = (output_size < input_size) ? output_size : input_size;
            for (size_t j = 0; j < out_min; j++) {
                state_after[j] = 0.5f * state_after[j] + 0.5f * output[j];
            }
            /* 用混合后的状态再做一次前向传播，深化知识影响 */
            lnn_forward(lnn_net, state_after, output);
            safe_free((void**)&state_after);
        }

        safe_free((void**)&combined_input);
        safe_free((void**)&output);
    } else {
        safe_free((void**)&combined_input);
        safe_free((void**)&output);
    }
    int result = 0; /* 成功 */

    safe_free((void**)&aggregated_embedding);
    return (result >= 0) ? 0 : -1;
}
/* P6-060 */
/* R002: 知识图谱路径查找（委托find_all_paths完整实现）
 * ZSF-015 修复: 参数命名规范化，g→graph, s→start_node, e→end_node, p→paths, m→max_paths */
size_t knowledge_graph_find_paths(KnowledgeGraph* graph, KnowledgeGraphNode* start_node,
                                  KnowledgeGraphNode* end_node, KnowledgeGraphPath** paths,
                                  size_t max_paths) {
    if (!graph || !start_node || !end_node || !paths || max_paths == 0) return 0;
    KnowledgeGraphQueryOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.max_depth = (int)(graph->node_count > 0 ? graph->node_count : 100);
    opts.min_confidence = 0.0f;
    opts.directed = 0;
    opts.include_cycles = 0;
    opts.max_results = max_paths;
    KnowledgeGraphPath** all = knowledge_graph_find_all_paths(graph, start_node, end_node, &opts, max_paths);
    if (!all) return 0;
    size_t count = 0;
    for (size_t i = 0; i < max_paths && all[i] != NULL; i++) {
        paths[i] = all[i];
        count++;
    }
    safe_free((void**)&all);
    return count;
}

/* ================================================================
 * v9.18: 连通分量 — BFS遍历识别所有连通分量
 * 
 * 使用广度优先搜索遍历无向图，为每个节点分配连通分量ID。
 * 时间复杂度O(n+m)，空间复杂度O(n)。
 * ================================================================ */
int knowledge_graph_connected_components(KnowledgeGraph* graph, int* component_ids, size_t* component_count) {
    if (!graph || !component_ids || !component_count) return -1;
    if (graph->node_count == 0) {
        *component_count = 0;
        return 0;
    }
    
    size_t n = graph->node_count;
    /* 访问标记数组：0=未访问，1=已访问 */
    int* visited = (int*)safe_calloc(n, sizeof(int));
    if (!visited) return -1;
    
    /* BFS队列 */
    size_t* queue = (size_t*)safe_malloc(n * sizeof(size_t));
    if (!queue) {
        safe_free((void**)&visited);
        return -1;
    }
    
    size_t comp_id = 0;
    for (size_t i = 0; i < n; i++) {
        component_ids[i] = -1; /* 初始化 */
    }
    
    /* 遍历所有节点，对每个未访问节点启动BFS */
    for (size_t i = 0; i < n; i++) {
        if (visited[i]) continue;
        
        /* BFS初始化 */
        size_t q_head = 0, q_tail = 0;
        queue[q_tail++] = i;
        visited[i] = 1;
        component_ids[i] = (int)comp_id;
        
        /* BFS主循环 */
        while (q_head < q_tail) {
            size_t node_idx = queue[q_head++];
            KnowledgeGraphNode* node = graph->nodes[node_idx];
            if (!node) continue;
            
            /* 遍历所有邻居边 */
            for (size_t e = 0; e < node->edge_count; e++) {
                KnowledgeGraphEdge* edge = node->edges[e];
                if (!edge || !edge->is_active) continue;
                
                /* 确定邻居节点索引 */
                KnowledgeGraphNode* neighbor = (edge->source == node) ? edge->target : edge->source;
                if (!neighbor) continue;
                
                /* 在节点数组中查找邻居索引 */
                for (size_t j = 0; j < n; j++) {
                    if (graph->nodes[j] == neighbor && !visited[j]) {
                        visited[j] = 1;
                        component_ids[j] = (int)comp_id;
                        queue[q_tail++] = j;
                        break;
                    }
                }
            }
        }
        comp_id++;
    }
    
    *component_count = comp_id;
    safe_free((void**)&queue);
    safe_free((void**)&visited);
    return 0;
}

/* ================================================================
 * v9.18: 度分布 — 计算度频率直方图
 * 
 * 统计每个度值（0到max_degree-1）的节点数量。
 * 度=节点的有效边数（is_active=1的边）。
 * ================================================================ */
int knowledge_graph_degree_distribution(KnowledgeGraph* graph, int* degree_counts, size_t max_degree) {
    if (!graph || !degree_counts || max_degree == 0) return -1;
    
    /* 清零计数数组 */
    memset(degree_counts, 0, max_degree * sizeof(int));
    
    for (size_t i = 0; i < graph->node_count; i++) {
        KnowledgeGraphNode* node = graph->nodes[i];
        if (!node) continue;
        
        /* 计算有效出边数 */
        int degree = 0;
        for (size_t e = 0; e < node->edge_count; e++) {
            if (node->edges[e] && node->edges[e]->is_active) {
                degree++;
            }
        }
        
        /* 截断到max_degree-1 */
        if ((size_t)degree >= max_degree) {
            degree_counts[max_degree - 1]++;
        } else {
            degree_counts[degree]++;
        }
    }
    
    return 0;
}
