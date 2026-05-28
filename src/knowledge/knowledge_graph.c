/**
 * @file knowledge_graph.c
 * @brief 知识图谱系统实现
 * 
 * 知识图谱数据结构、图算法和高级查询实现。
 * 支持图遍历、路径查找、子图匹配、图分析等高级功能。
 */

#define SELFLNN_KNOWLEDGE_INTERNAL  /* ZSFZS-F034: 与knowledge_graph.h保持一致 */
#include "selflnn/knowledge/knowledge_graph.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <time.h>

/* ============================================================================
 * CRC32校验（纯C实现，不依赖外部库）
 * ============================================================================
 * 用于知识图谱持久化文件的数据完整性校验。
 * 使用标准CRC-32/IEEE多项式：0xEDB88320（反转表示）。
 */

static uint32_t crc32_table[256];
static int crc32_table_initialized = 0;

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

static uint32_t crc32_compute(const void* data, size_t length) {
    if (!crc32_table_initialized) crc32_build_table();
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

/* ZSFZS-F034: KnowledgeGraph完整结构已移至knowledge_graph.h(#ifdef SELFLNN_KNOWLEDGE_INTERNAL) */

/* ============================================================================
 * 内部辅助函数
 * =========================================================================== */

/**
 * @brief 扩展动态数组
 */
static int expand_array(void*** array, size_t* capacity, size_t current_size, size_t element_size) {
    (void)current_size; // 未使用参数，避免警告
    if (!array || !capacity) return -1;
    
    size_t new_capacity = (*capacity == 0) ? 16 : (*capacity * 2);
    void* new_array = safe_realloc(*array, new_capacity * element_size);
    if (!new_array) return -1;
    
    *array = new_array;
    *capacity = new_capacity;
    return 0;
}

/**
 * @brief 计算节点相似度
 */
static float node_similarity(const GraphNode* a, const GraphNode* b) {
    if (!a || !b) return 0.0f;
    
    // 简单实现：基于标签匹配
    if (a->label && b->label && strcmp(a->label, b->label) == 0) {
        return 1.0f;
    }
    
    // 嵌入向量相似度（如果可用）
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
            return dot / (sqrtf(norm_a) * sqrtf(norm_b));
        }
    }
    
    return 0.0f;
}

/* ============================================================================
 * 哈希表实现（用于快速节点查找）
 * =========================================================================== */

/**
 * @brief 哈希表条目
 */
typedef struct HashEntry {
    char* key;              /**< 字符串键 */
    GraphNode* value;       /**< 节点值 */
    int occupied;           /**< 是否被占用 */
} HashEntry;

/**
 * @brief 字符串哈希函数（djb2算法）
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
                              GraphNode* value, size_t* occupied_count) {
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
 * @return GraphNode* 找到的节点，未找到返回NULL
 */
static GraphNode* hash_table_lookup(HashEntry* table, size_t capacity, const char* key) {
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
                                        const char* key, GraphNode* value) {
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
        GraphNode* node = graph->nodes[i];
        if (node) {
            if (node->label) safe_free((void**)&node->label);
            if (node->embedding) safe_free((void**)&node->embedding);
            if (node->edges) safe_free((void**)&node->edges);
            safe_free((void**)&node);
        }
    }
    
    // 释放所有边
    for (size_t i = 0; i < graph->edge_count; i++) {
        GraphEdge* edge = graph->edges[i];
        if (edge) {
            if (edge->label) safe_free((void**)&edge->label);
            safe_free((void**)&edge);
        }
    }
    
    // 释放数组
    if (graph->nodes) safe_free((void**)&graph->nodes);
    if (graph->edges) safe_free((void**)&graph->edges);
    
    safe_free((void**)&graph);
}

GraphNode* knowledge_graph_add_node(KnowledgeGraph* graph, GraphNodeType type,
                                   const char* label, const float* embedding,
                                   size_t embedding_size, float confidence) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "添加节点：知识图谱为空");
        return NULL;
    }
    
    if (graph->max_nodes > 0 && graph->node_count >= graph->max_nodes) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_FULL, __func__, __FILE__, __LINE__,
                              "添加节点：知识图谱节点容量已满");
        return NULL;
    }
    
    // 扩展节点数组（如果需要）
    if (graph->node_count >= graph->node_capacity) {
        if (expand_array((void***)&graph->nodes, &graph->node_capacity, graph->node_count, sizeof(GraphNode*)) != 0) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "添加节点：扩展节点数组失败");
            return NULL;
        }
    }
    
    GraphNode* node = (GraphNode*)safe_malloc(sizeof(GraphNode));
    if (!node) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "添加节点：节点内存分配失败");
        return NULL;
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
            return NULL;
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
            return NULL;
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
    
    return node;
}

GraphEdge* knowledge_graph_add_edge(KnowledgeGraph* graph, GraphEdgeType type,
                                   GraphNode* source, GraphNode* target,
                                   const char* label, float weight, float confidence) {
    if (!graph || !source || !target) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "添加边：参数无效");
        return NULL;
    }
    
    if (graph->max_edges > 0 && graph->edge_count >= graph->max_edges) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_FULL, __func__, __FILE__, __LINE__,
                              "添加边：知识图谱边容量已满");
        return NULL;
    }
    
    // 扩展边数组（如果需要）
    if (graph->edge_count >= graph->edge_capacity) {
        if (expand_array((void***)&graph->edges, &graph->edge_capacity, graph->edge_count, sizeof(GraphEdge*)) != 0) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "添加边：扩展边数组失败");
            return NULL;
        }
    }
    
    GraphEdge* edge = (GraphEdge*)safe_malloc(sizeof(GraphEdge));
    if (!edge) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "添加边：边内存分配失败");
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
            return NULL;
        }
    } else {
        edge->label = NULL;
    }
    
    // 添加到源节点的边列表
    if (source->edge_count >= source->edge_capacity) {
        if (expand_array((void***)&source->edges, &source->edge_capacity, source->edge_count, sizeof(GraphEdge*)) != 0) {
            if (edge->label) safe_free((void**)&edge->label);
            safe_free((void**)&edge);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "添加边：扩展源节点边列表失败");
            return NULL;
        }
    }
    source->edges[source->edge_count++] = edge;
    
    // 添加到图
    graph->edges[graph->edge_count++] = edge;
    graph->dirty = 1;
    
    return edge;
}

GraphNode* knowledge_graph_find_node_by_id(KnowledgeGraph* graph, int node_id) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "根据ID查找节点：知识图谱为空");
        return NULL;
    }
    
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] && graph->nodes[i]->id == node_id) {
            return graph->nodes[i];
        }
    }
    
    selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                          "根据ID查找节点：未找到节点ID %d", node_id);
    return NULL;
}

size_t knowledge_graph_find_nodes_by_label(KnowledgeGraph* graph, const char* label,
                                          GraphNode** results, size_t max_results) {
    if (!graph || !label) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "根据标签查找节点：参数无效");
        return 0;
    }
    
    size_t count = 0;
    
    for (size_t i = 0; i < graph->node_count && count < max_results; i++) {
        GraphNode* node = graph->nodes[i];
        if (node && node->label && strcmp(node->label, label) == 0) {
            if (results) results[count] = node;
            count++;
        }
    }
    
    return count;
}

GraphEdge* knowledge_graph_find_edge_by_id(KnowledgeGraph* graph, int edge_id) {
    if (!graph) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "根据ID查找边：知识图谱为空");
        return NULL;
    }
    
    for (size_t i = 0; i < graph->edge_count; i++) {
        if (graph->edges[i] && graph->edges[i]->id == edge_id) {
            return graph->edges[i];
        }
    }
    
    selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                          "根据ID查找边：未找到边ID %d", edge_id);
    return NULL;
}

/* ============================================================================
 * 图遍历算法
 * =========================================================================== */

/**
 * @brief 递归DFS辅助函数
 */
static int dfs_recursive(GraphNode* node, int* visited, size_t node_count,
                        GraphNode** all_nodes, int (*visit_callback)(GraphNode*, void*), 
                        void* user_data, const GraphQueryOptions* options) {
    (void)options; // 未使用参数，避免警告
    
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
        GraphEdge* edge = node->edges[i];
        GraphNode* neighbor = (edge->source == node) ? edge->target : edge->source;
        
        // 递归访问邻居
        result = dfs_recursive(neighbor, visited, node_count, all_nodes, 
                              visit_callback, user_data, options);
        if (result != 0) {
            return result; // 递归调用请求停止遍历
        }
    }
    
    return 0;
}

int knowledge_graph_dfs(KnowledgeGraph* graph, GraphNode* start_node,
                       int (*visit_callback)(GraphNode*, void*), void* user_data,
                       const GraphQueryOptions* options) {
    (void)user_data; // 未使用参数，避免警告
    (void)options; // 未使用参数，避免警告
    if (!graph || !start_node || !visit_callback) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "深度优先搜索：参数无效");
        return -1;
    }
    
    int* visited = (int*)safe_calloc(graph->node_count, sizeof(int));
    if (!visited) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "深度优先搜索：访问标记数组分配失败");
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
        return -1;
    }
    
    // 完整递归DFS实现（ 处理）
    int result = dfs_recursive(start_node, visited, graph->node_count, graph->nodes,
                              visit_callback, user_data, options);
    
    safe_free((void**)&visited);
    return result;
}

int knowledge_graph_bfs(KnowledgeGraph* graph, GraphNode* start_node,
                       int (*visit_callback)(GraphNode*, void*), void* user_data,
                       const GraphQueryOptions* options) {
    (void)user_data; // 未使用参数，避免警告
    (void)options; // 未使用参数，避免警告
    if (!graph || !start_node || !visit_callback) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "广度优先搜索：参数无效");
        return -1;
    }
    
    // 完整BFS实现（ 处理）
    int* visited = (int*)safe_calloc(graph->node_count, sizeof(int));
    if (!visited) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "广度优先搜索：访问标记数组分配失败");
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
        return -1;
    }
    
    // 创建队列（使用循环数组）
    GraphNode** queue = (GraphNode**)safe_malloc(graph->node_count * sizeof(GraphNode*));
    if (!queue) {
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "广度优先搜索：队列分配失败");
        return -1;
    }
    
    size_t front = 0, rear = 0;
    
    // 将起始节点加入队列并标记为已访问
    queue[rear++] = start_node;
    visited[start_idx] = 1;
    
    while (front < rear) {
        GraphNode* current = queue[front++];
        
        // 调用访问回调
        int result = visit_callback(current, user_data);
        if (result != 0) {
            safe_free((void**)&visited);
            safe_free((void**)&queue);
            return result; // 回调请求停止遍历
        }
        
        // 将所有未访问的邻居加入队列
        for (size_t i = 0; i < current->edge_count; i++) {
            GraphEdge* edge = current->edges[i];
            GraphNode* neighbor = (edge->source == current) ? edge->target : edge->source;
            
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
                    return -1;
                }
            }
        }
    }
    
    safe_free((void**)&visited);
    safe_free((void**)&queue);
    return 0;
}

/* ============================================================================
 * 路径查找算法
 * =========================================================================== */

GraphPath* knowledge_graph_shortest_path(KnowledgeGraph* graph,
                                        GraphNode* start_node, GraphNode* end_node,
                                        const GraphQueryOptions* options) {
    (void)options; // 未使用参数，避免警告
    if (!graph || !start_node || !end_node) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "最短路径：参数无效");
        return NULL;
    }
    
    // 完整Dijkstra算法实现（ 处理）
    size_t node_count = graph->node_count;
    
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
        GraphNode* current = graph->nodes[min_idx];
        for (size_t j = 0; j < current->edge_count; j++) {
            GraphEdge* edge = current->edges[j];
            GraphNode* neighbor = (edge->source == current) ? edge->target : edge->source;
            
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
    GraphPath* path = (GraphPath*)safe_malloc(sizeof(GraphPath));
    if (!path) {
        safe_free((void**)&dist);
        safe_free((void**)&prev);
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "最短路径：路径结构分配失败");
        return NULL;
    }
    
    path->nodes = (GraphNode**)safe_malloc(path_length * sizeof(GraphNode*));
    path->edges = (GraphEdge**)safe_malloc((path_length - 1) * sizeof(GraphEdge*));
    
    if (!path->nodes || !path->edges) {
        safe_free((void**)&path->nodes);
        safe_free((void**)&path->edges);
        safe_free((void**)&path);
        safe_free((void**)&dist);
        safe_free((void**)&prev);
        safe_free((void**)&visited);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "最短路径：路径数组分配失败");
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
        GraphNode* from = path->nodes[i];
        GraphNode* to = path->nodes[i + 1];
        
        // 查找连接两个节点的边
        GraphEdge* found_edge = NULL;
        for (size_t j = 0; j < from->edge_count; j++) {
            GraphEdge* edge = from->edges[j];
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
    return path;
}

GraphPath** knowledge_graph_find_all_paths(KnowledgeGraph* graph,
                                          GraphNode* start_node, GraphNode* end_node,
                                          const GraphQueryOptions* options,
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
    GraphPath** result_paths = (GraphPath**)safe_calloc(max_paths, sizeof(GraphPath*));
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
    GraphNode** current_path = (GraphNode**)safe_malloc(node_count * sizeof(GraphNode*));
    GraphEdge** current_edges = (GraphEdge**)safe_malloc((node_count - 1) * sizeof(GraphEdge*));
    
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
        
        GraphNode* current_node = graph->nodes[current_idx];
        current_path[depth] = current_node;
        
        // 如果找到目标节点且不是起始节点
        if (current_idx == end_idx && depth > 0) {
            // 创建路径
            GraphPath* path = (GraphPath*)safe_malloc(sizeof(GraphPath));
            if (!path) continue;
            
            path->nodes = (GraphNode**)safe_malloc((depth + 1) * sizeof(GraphNode*));
            path->edges = (GraphEdge**)safe_malloc(depth * sizeof(GraphEdge*));
            
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
                GraphNode* from = current_path[i];
                GraphNode* to = current_path[i + 1];
                
                // 查找连接两个节点的边
                GraphEdge* found_edge = NULL;
                for (size_t j = 0; j < from->edge_count; j++) {
                    GraphEdge* edge = from->edges[j];
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
            GraphEdge* edge = current_node->edges[i];
            GraphNode* neighbor = (edge->source == current_node) ? edge->target : edge->source;
            
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
    GraphNode* pattern_node;    // 模式图节点
    GraphNode* target_node;     // 目标图节点
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
            GraphEdge* pattern_edge = pattern->edges[i];
            
            // 查找模式边两端节点在映射中的对应节点
            GraphNode* pattern_source = pattern_edge->source;
            GraphNode* pattern_target = pattern_edge->target;
            
            GraphNode* target_source = NULL;
            GraphNode* target_target = NULL;
            
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
                    GraphEdge* target_edge = target_source->edges[j];
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
        
        result->matched_nodes = (GraphNode**)safe_malloc(pattern->node_count * sizeof(GraphNode*));
        result->matched_edges = (GraphEdge**)safe_malloc(matched_edges * sizeof(GraphEdge*));
        
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
            GraphEdge* pattern_edge = pattern->edges[i];
            GraphNode* pattern_source = pattern_edge->source;
            GraphNode* pattern_target = pattern_edge->target;
            
            GraphNode* target_source = NULL;
            GraphNode* target_target = NULL;
            
            for (size_t j = 0; j < pattern->node_count; j++) {
                if (mapping[j].pattern_node == pattern_source) target_source = mapping[j].target_node;
                if (mapping[j].pattern_node == pattern_target) target_target = mapping[j].target_node;
            }
            
            if (target_source && target_target) {
                for (size_t j = 0; j < target_source->edge_count; j++) {
                    GraphEdge* target_edge = target_source->edges[j];
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
    
    GraphNode* pattern_node = pattern->nodes[pattern_node_idx];
    
    // 尝试将模式节点映射到目标图的每个节点
    for (size_t target_node_idx = 0; target_node_idx < graph->node_count; target_node_idx++) {
        GraphNode* target_node = graph->nodes[target_node_idx];
        
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
                    GraphEdge* edge = pattern_node->edges[j];
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
                        GraphEdge* edge = target_node->edges[j];
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
                                             const GraphQueryOptions* options) {
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

float knowledge_graph_node_centrality(KnowledgeGraph* graph, GraphNode* node,
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
                    GraphNode* current_node = graph->nodes[current];
                    
                    for (size_t i = 0; i < current_node->edge_count; i++) {
                        GraphEdge* edge = current_node->edges[i];
                        GraphNode* neighbor = (edge->source == current_node) ? edge->target : edge->source;
                        
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
                for (size_t i = 0; i < node_count; i++) {
                    GraphNode* src_node = graph->nodes[i];
                    for (size_t j = 0; j < src_node->edge_count; j++) {
                        GraphEdge* edge = src_node->edges[j];
                        GraphNode* neighbor = (edge->source == src_node) ? edge->target : edge->source;
                        
                        // 查找邻居索引
                        size_t neighbor_idx = 0;
                        for (; neighbor_idx < node_count; neighbor_idx++) {
                            if (graph->nodes[neighbor_idx] == neighbor) break;
                        }
                        
                        if (neighbor_idx < node_count) {
                            distances[i][neighbor_idx] = 1;
                            path_counts[i][neighbor_idx] = 1;
                        }
                    }
                }
                
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
        GraphNode* node = graph->nodes[i];
        size_t degree = node->edge_count;
        
        // 如果度小于2，局部聚类系数为0（无法形成三角形）
        if (degree < 2) {
            continue;
        }
        
        // 计算邻居之间实际存在的边数
        size_t neighbor_edges = 0;
        
        // 遍历所有邻居对
        for (size_t j = 0; j < degree; j++) {
            GraphEdge* edge1 = node->edges[j];
            GraphNode* neighbor1 = (edge1->source == node) ? edge1->target : edge1->source;
            
            for (size_t k = j + 1; k < degree; k++) {
                GraphEdge* edge2 = node->edges[k];
                GraphNode* neighbor2 = (edge2->source == node) ? edge2->target : edge2->source;
                
                // 检查neighbor1和neighbor2之间是否有边
                int edge_found = 0;
                for (size_t l = 0; l < neighbor1->edge_count; l++) {
                    GraphEdge* edge = neighbor1->edges[l];
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
        GraphNode* node = graph->nodes[i];
        if (!node) continue;
        for (size_t j = 0; j < node->edge_count; j++) {
            GraphEdge* edge = node->edges[j];
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
        GraphNode* node = graph->nodes[i];
        if (!node) continue;
        for (size_t j = 0; j < node->edge_count; j++) {
            GraphEdge* edge = node->edges[j];
            if (!edge) continue;
            if (edge->source == node) {
                /* 有向边 i -> target */
                GraphNode* target = edge->target;
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
        GraphNode* node = graph->nodes[i];
        if (!node) continue;
        for (size_t j = 0; j < node->edge_count; j++) {
            GraphEdge* edge = node->edges[j];
            if (!edge) continue;
            if (edge->source == node) {
                GraphNode* target = edge->target;
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

    /* 第一遍：计算邻居数（去重） */
    for (size_t i = 0; i < n; i++) {
        GraphNode* node = graph->nodes[i];
        if (!node) continue;
        /* 使用临时集合去重邻居 */
        int* seen = (int*)safe_calloc(n, sizeof(int));
        if (!seen) {
            safe_free((void**)&counts);
            safe_free((void**)&adj);
            return -1;
        }
        for (size_t j = 0; j < node->edge_count; j++) {
            GraphEdge* edge = node->edges[j];
            if (!edge) continue;
            GraphNode* neighbor = (edge->source == node) ? edge->target : edge->source;
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
        GraphNode* node = graph->nodes[i];
        if (!node) continue;
        int* seen = (int*)safe_calloc(n, sizeof(int));
        if (!seen) {
            for (size_t k = 0; k < n; k++) safe_free((void**)&adj[k]);
            safe_free((void**)&adj);
            safe_free((void**)&counts);
            safe_free((void**)&fill_pos);
            return -1;
        }
        for (size_t j = 0; j < node->edge_count; j++) {
            GraphEdge* edge = node->edges[j];
            if (!edge) continue;
            GraphNode* neighbor = (edge->source == node) ? edge->target : edge->source;
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

    size_t n = graph->node_count;
    if (n == 0) return 0;
    if (score_count < n) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "PageRank：分数数组太小");
        return -1;
    }

    const float d = 0.85f;
    const float tol = 1e-6f;
    const int max_iter = 100;
    const float teleport = (1.0f - d) / (float)n;

    /* 构建出度数组 */
    size_t* out_degree = (size_t*)safe_calloc(n, sizeof(size_t));
    if (!out_degree) return -1;

    compute_out_degrees(graph, out_degree);

    /* 构建入度邻接表 */
    size_t** inbound = NULL;
    size_t* inbound_counts = NULL;
    if (build_inbound_adjacency(graph, &inbound, &inbound_counts) != 0) {
        safe_free((void**)&out_degree);
        return -1;
    }

    /* 初始化PageRank值 */
    float* old_scores = (float*)safe_malloc(n * sizeof(float));
    if (!old_scores) {
        safe_free((void**)&out_degree);
        for (size_t k = 0; k < n; k++) safe_free((void**)&inbound[k]);
        safe_free((void**)&inbound);
        safe_free((void**)&inbound_counts);
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
    memory += graph->node_capacity * sizeof(GraphNode*);
    memory += graph->edge_capacity * sizeof(GraphEdge*);
    for (size_t i = 0; i < n; i++) {
        GraphNode* node = graph->nodes[i];
        memory += sizeof(GraphNode);
        if (node->label) memory += strlen(node->label) + 1;
        if (node->embedding) memory += node->embedding_size * sizeof(float);
        memory += node->edge_capacity * sizeof(GraphEdge*);
    }
    for (size_t i = 0; i < e; i++) {
        GraphEdge* edge = graph->edges[i];
        memory += sizeof(GraphEdge);
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
        GraphNode* node = graph->nodes[j];
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
        GraphNode* subject_node = hash_table_lookup(hash_table, hash_capacity, entry->subject);
        
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
        GraphNode* object_node = hash_table_lookup(hash_table, hash_capacity, entry->object);
        
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
        GraphEdge* edge = knowledge_graph_add_edge(graph, EDGE_TYPE_RELATION, subject_node, object_node, 
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
        GraphEdge* edge = graph->edges[i];
        
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
        GraphNode* node = graph->nodes[i];
        
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
        GraphEdge* edge = graph->edges[i];
        
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

    unsigned char* file_buffer = (unsigned char*)malloc((size_t)file_size);
    if (!file_buffer) {
        fclose(file);
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "保存知识图谱：分配CRC32缓冲区失败");
        return -1;
    }

    fseek(file, 0, SEEK_SET);
    if (fread(file_buffer, 1, (size_t)file_size, file) != (size_t)file_size) {
        free(file_buffer);
        fclose(file);
        remove(temp_filename);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存知识图谱：读取文件计算CRC32失败");
        return -1;
    }
    fclose(file);

    uint32_t checksum = crc32_compute(file_buffer, (size_t)file_size);
    free(file_buffer);

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

    unsigned char* verify_buffer = (unsigned char*)malloc((size_t)data_size);
    if (!verify_buffer) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "加载知识图谱：分配CRC32验证缓冲区失败");
        return NULL;
    }

    fseek(file, 0, SEEK_SET);
    if (fread(verify_buffer, 1, (size_t)data_size, file) != (size_t)data_size) {
        free(verify_buffer);
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载知识图谱：读取文件数据计算CRC32失败");
        return NULL;
    }

    uint32_t computed_checksum = crc32_compute(verify_buffer, (size_t)data_size);
    free(verify_buffer);

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
    GraphNode** loaded_nodes = (GraphNode**)safe_malloc(node_count * sizeof(GraphNode*));
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
        GraphNode* node = knowledge_graph_add_node(graph, (GraphNodeType)node_type, label, NULL, 0, 0.5f);
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
        GraphNode* source_node = loaded_nodes[source_idx];
        GraphNode* target_node = loaded_nodes[target_idx];
        
        // 创建边
        GraphEdge* edge = knowledge_graph_add_edge(graph, EDGE_TYPE_RELATION, source_node, target_node, label, weight, 0.5f);
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
        total_memory += graph->node_capacity * sizeof(GraphNode*);
        
        // 边数组内存
        total_memory += graph->edge_capacity * sizeof(GraphEdge*);
        
        // 每个节点的内存
        for (size_t i = 0; i < graph->node_count; i++) {
            GraphNode* node = graph->nodes[i];
            total_memory += sizeof(GraphNode);
            
            if (node->label) {
                total_memory += strlen(node->label) + 1;
            }
            
            if (node->embedding) {
                total_memory += node->embedding_size * sizeof(float);
            }
            
            // 边指针数组
            total_memory += node->edge_capacity * sizeof(GraphEdge*);
        }
        
        // 每条边的内存
        for (size_t i = 0; i < graph->edge_count; i++) {
            GraphEdge* edge = graph->edges[i];
            total_memory += sizeof(GraphEdge);
            
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

size_t knowledge_graph_get_all_nodes(KnowledgeGraph* graph, GraphNode** results, size_t max_results) {
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

size_t knowledge_graph_get_all_edges(KnowledgeGraph* graph, GraphEdge** results, size_t max_results) {
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
    GraphNode* start_node, GraphEdgeType relation_type,
    GraphNode** results, size_t max_results) {
    if (!graph || !start_node || !results || max_results == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "传递闭包：参数无效");
        return 0;
    }
    
    // 真正的传递闭包实现：BFS遍历，仅跟踪指定关系类型的边
    // 使用队列进行BFS
    GraphNode** queue = (GraphNode**)safe_malloc((graph->node_count + 1) * sizeof(GraphNode*));
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
        GraphNode* node = graph->nodes[i];
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
        GraphNode* current = queue[queue_front++];
        
        // 遍历当前节点的所有出边
        for (size_t i = 0; i < current->edge_count; i++) {
            GraphEdge* edge = current->edges[i];
            
            // 检查边类型是否匹配
            if (edge->type != relation_type) {
                continue;
            }
            
            // 检查边方向：源节点必须是当前节点
            if (edge->source != current) {
                continue;
            }
            
            GraphNode* neighbor = edge->target;
            
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
    GraphNode* start_node, const GraphEdgeType* relation_types, size_t num_types,
    size_t min_hops, size_t max_hops,
    GraphNode** results, size_t max_results) {
    if (!graph || !start_node || !relation_types || num_types == 0 || !results || max_results == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "多跳查询：参数无效");
        return 0;
    }
    
    // 真正的多跳查询实现：BFS遍历，跟踪跳数，仅允许指定关系类型的边
    // 使用队列进行BFS，同时存储节点和跳数
    typedef struct {
        GraphNode* node;
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
        GraphNode* current = queue[queue_front].node;
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
            GraphEdge* edge = current->edges[i];
            
            // 检查边类型是否在允许的类型列表中
            IS_ALLOWED_TYPE(edge->type);
            
            // 检查边方向：源节点必须是当前节点
            if (edge->source != current) {
                continue;
            }
            
            GraphNode* neighbor = edge->target;
            
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
    GraphNode* start_node, const GraphEdgeType* pattern, size_t pattern_length,
    GraphPath** paths, size_t max_paths) {
    if (!graph || !start_node || !pattern || pattern_length == 0 || !paths || max_paths == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "关系路径查询：参数无效");
        return 0;
    }
    
    // 真正的关系路径查询实现：迭代DFS遍历，匹配关系类型序列
    // 使用栈进行迭代DFS，避免递归深度限制
    
    // 栈项结构
    typedef struct {
        GraphNode* node;
        size_t pattern_index;
        size_t edge_index;
        size_t depth;
    } StackItem;
    
    // 分配栈
    StackItem* stack = (StackItem*)safe_malloc((pattern_length + 1) * sizeof(StackItem));
    GraphNode** node_stack = (GraphNode**)safe_malloc((pattern_length + 1) * sizeof(GraphNode*));
    GraphEdge** edge_stack = (GraphEdge**)safe_malloc(pattern_length * sizeof(GraphEdge*));
    
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
            GraphPath* new_path = (GraphPath*)safe_malloc(sizeof(GraphPath));
            if (!new_path) {
                // 内存不足，停止搜索
                break;
            }
            
            new_path->length = current->depth + 1;
            new_path->nodes = (GraphNode**)safe_malloc((current->depth + 1) * sizeof(GraphNode*));
            new_path->edges = (GraphEdge**)safe_malloc(current->depth * sizeof(GraphEdge*));
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
        GraphEdgeType expected_type = pattern[current->pattern_index];
        
        // 遍历当前节点的出边
        if (current->edge_index < current->node->edge_count) {
            GraphEdge* edge = current->node->edges[current->edge_index];
            current->edge_index++;
            
            // 检查边类型是否匹配期望类型
            if (edge->type != expected_type) {
                continue; // 继续while循环，检查下一条边
            }
            
            // 检查边方向：源节点必须是当前节点
            if (edge->source != current->node) {
                continue;
            }
            
            GraphNode* next_node = edge->target;
            
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
                                   GraphEdge** results, size_t max_results)
{
    size_t count = 0;
    for (size_t i = 0; i < graph->edge_count && count < max_results; i++) {
        GraphEdge* e = graph->edges[i];
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

    /* 为每个模式查找匹配边 */
    GraphEdge* matches[SELFLNN_SPARQL_MAX_PATTERNS][256];
    size_t match_counts[SELFLNN_SPARQL_MAX_PATTERNS];

    for (size_t p = 0; p < pattern_count; p++) {
        match_counts[p] = find_matching_edges(graph, &patterns[p],
                                              matches[p], 256);
    }

    /* 检查是否有任何模式无匹配（非OPTIONAL模式） */
    for (size_t p = 0; p < pattern_count; p++) {
        if (match_counts[p] == 0 && !patterns[p].optional) {
            return -1;
        }
    }

    /* 交叉连接所有匹配结果 */
    size_t indices[SELFLNN_SPARQL_MAX_PATTERNS];
    memset(indices, 0, sizeof(indices));

    size_t row_count = 0;
    while (row_count < 256) {
        /* 检查当前组合的兼容性 */
        for (size_t v = 0; v < var_count; v++) {
            GraphNode* val = NULL;
            for (size_t p = 0; p < pattern_count && val == NULL; p++) {
                if (match_counts[p] == 0) continue;
                GraphEdge* e = matches[p][indices[p]];
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
                row_conf *= matches[p][indices[p]]->confidence;
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
static const char* graph_node_type_name(GraphNodeType type) {
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
static const char* graph_edge_type_name(GraphEdgeType type) {
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
    GraphNode* all_nodes[4096];
    GraphEdge* all_edges[16384];
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
        GraphNode* n = all_nodes[i];
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
        GraphEdge* e = all_edges[i];
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
int knowledge_graph_subgraph_export_json(GraphNode** nodes, size_t node_count,
                                         GraphEdge** edges, size_t edge_count,
                                         char* json_buffer, size_t buffer_size)
{
    if (!nodes || node_count == 0 || !json_buffer || buffer_size == 0) return -1;

    size_t pos = 0;
    int written = snprintf(json_buffer + pos, buffer_size - pos,
                           "{\n  \"nodes\": [\n");
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    for (size_t i = 0; i < node_count; i++) {
        GraphNode* n = nodes[i];
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
        GraphEdge* e = edges[i];
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

void knowledge_graph_free_path(GraphPath* path) {
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
    for (int i = 0; i < node_count; i++) {
        float theta = ((float)i / node_count) * 6.28318f;
        float phi = ((float)(i * 7) / node_count) * 3.14159f;
        nodes[i].x = cosf(theta) * sinf(phi) * 100.0f;
        nodes[i].y = sinf(theta) * sinf(phi) * 100.0f;
        nodes[i].z = cosf(phi) * 100.0f;
        nodes[i].vx = 0; nodes[i].vy = 0; nodes[i].vz = 0;
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

        /* 阻尼 + 温度衰减 + 位置更新 */
        float max_disp = 0;
        for (int i = 0; i < node_count; i++) {
            nodes[i].vx *= 0.9f;
            nodes[i].vy *= 0.9f;
            nodes[i].vz *= 0.9f;

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