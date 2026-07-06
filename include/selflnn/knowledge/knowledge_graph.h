/**
 * @file knowledge_graph.h
 * @brief 知识图谱系统接口
 * 
 * 知识图谱数据结构、图算法和高级查询接口。
 * 支持图遍历、路径查找、子图匹配、图分析等高级功能。
 */

#ifndef SELFLNN_KNOWLEDGE_GRAPH_H
#define SELFLNN_KNOWLEDGE_GRAPH_H

#include "selflnn/knowledge/knowledge.h"
#include "selflnn/core/common.h"
#include "selflnn/utils/platform.h"  /* DEEP-005: MutexHandle依赖 */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 图节点类型
 */
typedef enum {
    NODE_TYPE_CONCEPT = 0,      /**< 概念节点 */
    NODE_TYPE_ENTITY = 1,       /**< 实体节点 */
    NODE_TYPE_PROPERTY = 2,     /**< 属性节点 */
    NODE_TYPE_LITERAL = 3       /**< 字面量节点 */
} KnowledgeGraphNodeType;

/**
 * @brief 图边类型
 */
typedef enum {
    EDGE_TYPE_RELATION = 0,     /**< 关系边 */
    EDGE_TYPE_SUBCLASS = 1,     /**< 子类边 */
    EDGE_TYPE_INSTANCE = 2,     /**< 实例边 */
    EDGE_TYPE_PROPERTY = 3,     /**< 属性边 */
    EDGE_TYPE_SIMILARITY = 4    /**< 相似性边 */
} KnowledgeGraphEdgeType;

/**
 * @brief 图节点结构
 */
typedef struct KGraphNode {
    int id;                     /* 节点ID */
    KnowledgeGraphNodeType type;         /* 节点类型 */
    char* label;                /* 节点标签 */
    float* embedding;           /* 节点嵌入向量（可选） */
    size_t embedding_size;      /* 嵌入向量大小 */
    float confidence;           /* 节点置信度 */
    void* user_data;            /* 用户数据指针 */
    
    struct KGraphEdge** edges;  /* 边列表 */
    size_t edge_count;          /* 边数量 */
    size_t edge_capacity;       /* 边列表容量 */
} KnowledgeGraphNode;

/**
 * @brief 图边结构
 */
typedef struct KGraphEdge {
    int id;                     /* 边ID */
    KnowledgeGraphEdgeType type;         /* 边类型 */
    KnowledgeGraphNode* source; /* 源节点 */
    KnowledgeGraphNode* target; /* 目标节点 */
    char* label;                /* 边标签 */
    float weight;               /* 边权重 */
    float confidence;           /* 边置信度 */
    int is_active;              /* 边是否激活 */
    void* user_data;            /* 用户数据指针 */
} KnowledgeGraphEdge;

/**
 * @brief 知识图谱句柄
 */
typedef struct KnowledgeGraph KnowledgeGraph;

#ifdef SELFLNN_KNOWLEDGE_INTERNAL
/* 知识图谱完整结构（仅内部模块可见，与CfCNetwork/CfCCell模式一致） */
struct KnowledgeGraph {
    struct KGraphNode** nodes;
    size_t node_count;
    size_t node_capacity;
    struct KGraphEdge** edges;
    size_t edge_count;
    size_t edge_capacity;
    int next_node_id;
    int next_edge_id;
    size_t max_nodes;
    size_t max_edges;
    int dirty;
    char* auto_save_path;
/* 图操作互斥锁 — 原结构完全无锁, 多线程必然崩溃 */
    void* graph_lock;           /* MutexHandle */
};
#endif

/**
 * @brief 知识图谱完整统计结构
 *
 * 包含图分析算法的所有计算结果：
 * PageRank、介数中心度、紧密度中心度、Louvain社区检测等。
 * 调用 knowledge_graph_free_stats() 释放内部分配的内存。
 */
typedef struct {
    size_t node_count;              /**< 节点数 */
    size_t edge_count;              /**< 边数 */
    size_t memory_usage;            /**< 内存使用量（字节） */
    float density;                  /**< 图密度 [0,1] */
    float avg_degree;               /**< 平均度 */
    float diameter;                 /**< 图直径（最长最短路径），不可达时返回INFINITY */
    float avg_clustering;           /**< 平均聚类系数 */
    float* pagerank_scores;         /**< PageRank分数数组[节点数]，索引对应graph->nodes[] */
    float* betweenness_scores;      /**< 介数中心度数组[节点数] */
    float* closeness_scores;        /**< 紧密度中心度数组[节点数] */
    int* community_ids;             /**< Louvain社区ID数组[节点数] */
    size_t community_count;         /**< 检测到的社区数量 */
    float modularity;               /**< 模块度 */
} KnowledgeGraphStats;

/**
 * @brief 路径结构
 */
typedef struct {
    KnowledgeGraphNode** nodes;  /**< 路径节点数组 */
    KnowledgeGraphEdge** edges;  /**< 路径边数组 */
    size_t length;              /**< 路径长度（节点数） */
    float total_weight;         /**< 路径总权重 */
    float confidence;           /**< 路径置信度 */
} KnowledgeGraphPath;

/**
 * @brief 释放图路径
 * 
 * @param path 路径指针
 */
void knowledge_graph_free_path(KnowledgeGraphPath* path);

/**
 * @brief 子图匹配结果
 */
typedef struct {
    KnowledgeGraphNode** matched_nodes;  /**< 匹配的节点数组 */
    KnowledgeGraphEdge** matched_edges;  /**< 匹配的边数组 */
    size_t node_count;          /**< 匹配节点数 */
    size_t edge_count;          /**< 匹配边数 */
    float similarity;           /**< 相似度得分 */
} SubgraphMatch;

/**
 * @brief 图查询选项
 */
typedef struct {
    int max_depth;              /**< 最大搜索深度 */
    float min_confidence;       /**< 最小置信度 */
    float max_weight;           /**< 最大权重限制 */
    int directed;               /**< 是否定向搜索（1=有向，0=无向） */
    int include_cycles;         /**< 是否允许环 */
    size_t max_results;         /**< 最大结果数 */
} KnowledgeGraphQueryOptions;

/**
 * @brief 创建知识图谱
 * 
 * @param max_nodes 最大节点数（0表示无限制）
 * @param max_edges 最大边数（0表示无限制）
 * @return KnowledgeGraph* 知识图谱句柄，失败返回NULL
 */
SELFLNN_API KnowledgeGraph* knowledge_graph_create(size_t max_nodes, size_t max_edges);

/**
 * @brief 释放知识图谱
 * 
 * @param graph 知识图谱句柄
 */
SELFLNN_API void knowledge_graph_free(KnowledgeGraph* graph);

/**
 * @brief 添加节点到知识图谱
 * 
 * @param graph 知识图谱句柄
 * @param type 节点类型
 * @param label 节点标签
 * @param embedding 节点嵌入向量（可选）
 * @param embedding_size 嵌入向量大小
 * @param confidence 节点置信度
 * @return KnowledgeGraphNode* 创建的节点指针，失败返回NULL
 */
KnowledgeGraphNode* knowledge_graph_add_node(KnowledgeGraph* graph, KnowledgeGraphNodeType type,
                                   const char* label, const float* embedding,
                                   size_t embedding_size, float confidence);

/**
 * @brief 移除知识图谱节点
 * @param graph 知识图谱句柄
 * @param node 待移除的节点
 * @return 0成功，-1失败
 */
int knowledge_graph_remove_node(KnowledgeGraph* graph, KnowledgeGraphNode* node);

/**
 * @brief 添加边到知识图谱
 * 
 * @param graph 知识图谱句柄
 * @param type 边类型
 * @param source 源节点
 * @param target 目标节点
 * @param label 边标签
 * @param weight 边权重
 * @param confidence 边置信度
 * @return KnowledgeGraphEdge* 创建的边指针，失败返回NULL
 */
KnowledgeGraphEdge* knowledge_graph_add_edge(KnowledgeGraph* graph, KnowledgeGraphEdgeType type,
                                   KnowledgeGraphNode* source, KnowledgeGraphNode* target,
                                   const char* label, float weight, float confidence);

/**
 * @brief 根据ID查找节点
 * 
 * @param graph 知识图谱句柄
 * @param node_id 节点ID
 * @return KnowledgeGraphNode* 节点指针，未找到返回NULL
 */
KnowledgeGraphNode* knowledge_graph_find_node_by_id(KnowledgeGraph* graph, int node_id);

/**
 * @brief 根据标签查找节点
 * 
 * @param graph 知识图谱句柄
 * @param label 节点标签
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回匹配的节点数
 */
size_t knowledge_graph_find_nodes_by_label(KnowledgeGraph* graph, const char* label,
                                          KnowledgeGraphNode** results, size_t max_results);

/**
 * @brief 根据ID查找边
 * 
 * @param graph 知识图谱句柄
 * @param edge_id 边ID
 * @return KnowledgeGraphEdge* 边指针，未找到返回NULL
 */
KnowledgeGraphEdge* knowledge_graph_find_edge_by_id(KnowledgeGraph* graph, int edge_id);

/**
 * @brief 查找两个节点之间的所有路径
 * 
 * @param graph 知识图谱句柄
 * @param start_node 起始节点
 * @param end_node 目标节点（如果为NULL，则查找从起始节点出发的所有路径）
 * @param paths 路径输出缓冲区（数组）
 * @param max_paths 最大路径数
 * @return size_t 返回找到的路径数
 */
size_t knowledge_graph_find_paths(KnowledgeGraph* graph, KnowledgeGraphNode* start_node,
                                 KnowledgeGraphNode* end_node, KnowledgeGraphPath** paths,
                                 size_t max_paths);

/**
 * @brief 深度优先遍历
 * 
 * @param graph 知识图谱句柄
 * @param start_node 起始节点
 * @param callback 节点回调函数
 * @param user_data 用户数据
 * @param options 遍历选项
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_dfs(KnowledgeGraph* graph, KnowledgeGraphNode* start_node,
                       int (*callback)(KnowledgeGraphNode*, void*), void* user_data,
                       const KnowledgeGraphQueryOptions* options);

/**
 * @brief 广度优先遍历
 * 
 * @param graph 知识图谱句柄
 * @param start_node 起始节点
 * @param callback 节点回调函数
 * @param user_data 用户数据
 * @param options 遍历选项
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_bfs(KnowledgeGraph* graph, KnowledgeGraphNode* start_node,
                       int (*callback)(KnowledgeGraphNode*, void*), void* user_data,
                       const KnowledgeGraphQueryOptions* options);

/**
 * @brief 查找最短路径（Dijkstra算法）
 * 
 * @param graph 知识图谱句柄
 * @param start_node 起始节点
 * @param end_node 目标节点
 * @param options 查询选项
 * @return GraphPath* 最短路径，失败返回NULL
 */
KnowledgeGraphPath* knowledge_graph_shortest_path(KnowledgeGraph* graph,
                                        KnowledgeGraphNode* start_node, KnowledgeGraphNode* end_node,
                                        const KnowledgeGraphQueryOptions* options);

/**
 * @brief 查找所有路径
 * 
 * @param graph 知识图谱句柄
 * @param start_node 起始节点
 * @param end_node 目标节点
 * @param options 查询选项
 * @param max_paths 最大路径数
 * @return GraphPath** 路径数组，调用者负责释放，失败返回NULL
 */
KnowledgeGraphPath** knowledge_graph_find_all_paths(KnowledgeGraph* graph,
                                          KnowledgeGraphNode* start_node, KnowledgeGraphNode* end_node,
                                          const KnowledgeGraphQueryOptions* options,
                                          size_t max_paths);

/**
 * @brief 子图匹配
 * 
 * @param graph 目标图
 * @param pattern 模式子图
 * @param options 匹配选项
 * @return SubgraphMatch* 匹配结果，调用者负责释放，失败返回NULL
 */
SubgraphMatch* knowledge_graph_subgraph_match(KnowledgeGraph* graph,
                                             KnowledgeGraph* pattern,
                                             const KnowledgeGraphQueryOptions* options);

/**
 * @brief 计算节点中心性
 * 
 * @param graph 知识图谱句柄
 * @param node 节点
 * @param centrality_type 中心性类型（0=度中心性，1=接近中心性，2=介数中心性）
 * @return float 中心性值
 */
float knowledge_graph_node_centrality(KnowledgeGraph* graph, KnowledgeGraphNode* node,
                                     int centrality_type);

/* ================================================================
 * 高级图分析算法（R-017）
 * ================================================================ */

/**
 * @brief 计算PageRank分数（全图）
 *
 * 使用标准迭代PageRank算法，damping=0.85, max_iter=100, tol=1e-6。
 * 基于有向边计算，考虑边权重。
 *
 * @param graph 知识图谱句柄
 * @param scores 分数输出数组（大小必须 >= graph->node_count）
 * @param score_count 分数数组大小
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_pagerank(KnowledgeGraph* graph, float* scores, size_t score_count);

/**
 * @brief 计算全图介数中心度（Brandes算法）
 *
 * 使用Brandes算法计算所有节点的介数中心度。
 * 无向解释（每条边双向），时间复杂度O(n*m)。
 * 结果已标准化（除以(n-1)*(n-2)/2）。
 *
 * @param graph 知识图谱句柄
 * @param scores 分数输出数组（大小必须 >= graph->node_count）
 * @param score_count 分数数组大小
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_betweenness_centrality_all(KnowledgeGraph* graph, float* scores, size_t score_count);

/**
 * @brief 计算全图紧密度中心度
 *
 * 对每个节点执行BFS，计算到所有可达节点的平均距离的倒数。
 *
 * @param graph 知识图谱句柄
 * @param scores 分数输出数组（大小必须 >= graph->node_count）
 * @param score_count 分数数组大小
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_closeness_centrality_all(KnowledgeGraph* graph, float* scores, size_t score_count);

/**
 * @brief Louvain社区检测算法
 *
 * 使用模块度最大化的Louvain算法进行社区检测。
 * 将图视为无向图处理。
 *
 * @param graph 知识图谱句柄
 * @param community_ids 社区ID输出数组（大小必须 >= graph->node_count）
 * @param community_count 检测到的社区数量输出
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_louvain_communities(KnowledgeGraph* graph, int* community_ids, size_t* community_count);

/**
 * @brief 计算图直径
 *
 * 图直径 = 所有可达节点对之间最短路径的最大值。
 * 对每个节点执行BFS，取所有找到的最短距离的最大值。
 * 不可达时返回INFINITY。
 *
 * @param graph 知识图谱句柄
 * @return float 图直径
 */
float knowledge_graph_diameter(KnowledgeGraph* graph);

/**
 * @brief 计算图平均度
 *
 * 平均度 = 总边数 * 2 / 节点数（无向解释）。
 *
 * @param graph 知识图谱句柄
 * @return float 平均度
 */
float knowledge_graph_average_degree(KnowledgeGraph* graph);

/**
 * @brief 计算完整图分析统计
 *
 * 一次性计算所有图分析指标（PageRank、介数中心度、紧密度中心度、
 * 社区检测、图密度、直径、平均度等），填充KnowledgeGraphStats结构。
 * 完成后调用 knowledge_graph_free_stats() 释放内部分配的内存。
 *
 * @param graph 知识图谱句柄
 * @param stats 统计结果输出
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_compute_stats(KnowledgeGraph* graph, KnowledgeGraphStats* stats);

/**
 * @brief 释放KnowledgeGraphStats中动态分配的内存
 *
 * @param stats 统计结果
 */
void knowledge_graph_free_stats(KnowledgeGraphStats* stats);

/**
 * @brief 计算图密度
 * 
 * @param graph 知识图谱句柄
 * @return float 图密度
 */
float knowledge_graph_density(KnowledgeGraph* graph);

/**
 * @brief 计算聚类系数
 * 
 * @param graph 知识图谱句柄
 * @return float 平均聚类系数
 */
float knowledge_graph_clustering_coefficient(KnowledgeGraph* graph);

/**
 * @brief 从知识库导入知识到图谱
 * 
 * @param graph 知识图谱句柄
 * @param kb 知识库句柄
 * @return int 成功返回导入的条目数，失败返回-1
 */
int knowledge_graph_import_from_knowledge_base(KnowledgeGraph* graph,
                                              KnowledgeBase* kb);

/**
 * @brief 从图谱导出知识到知识库
 * 
 * @param graph 知识图谱句柄
 * @param kb 知识库句柄
 * @return int 成功返回导出的条目数，失败返回-1
 */
int knowledge_graph_export_to_knowledge_base(KnowledgeGraph* graph,
                                            KnowledgeBase* kb);

/**
 * @brief 获取图谱统计信息
 * 
 * @param graph 知识图谱句柄
 * @param node_count 节点数输出
 * @param edge_count 边数输出
 * @param memory_usage 内存使用量输出
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_get_stats(KnowledgeGraph* graph,
                             size_t* node_count, size_t* edge_count,
                             size_t* memory_usage);

/**
 * @brief 设置自动保存路径
 * 
 * 设置后，在知识图谱销毁时如果有修改，会自动保存到指定路径。
 * 使用临时文件+原子重名机制保证写入安全性。
 * 
 * @param graph 知识图谱句柄
 * @param save_path 自动保存路径（为NULL时取消自动保存）
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_set_auto_save(KnowledgeGraph* graph, const char* save_path);

/**
 * @brief 仅在脏数据时保存
 * 
 * 如果自上次保存后有修改且设置了自动保存路径，则执行一次保存。
 * 
 * @param graph 知识图谱句柄
 * @return int 成功返回0，无修改或无路径返回0，失败返回-1
 */
int knowledge_graph_save_if_dirty(KnowledgeGraph* graph);

/**
 * @brief 保存知识图谱到文件
 * 
 * 使用原子写入机制：先写入临时文件，成功后重命名覆盖目标文件。
 * 即使写入过程中程序崩溃，原始文件也不会损坏。
 * 
 * @param graph 知识图谱句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_save(KnowledgeGraph* graph, const char* filename);

/**
 * @brief 从文件加载知识图谱
 * 
 * @param filename 文件名
 * @return KnowledgeGraph* 知识图谱句柄，失败返回NULL
 */
KnowledgeGraph* knowledge_graph_load(const char* filename);

/**
 * @brief 释放路径内存
 * 
 * @param path 路径指针
 */
void graph_path_free(KnowledgeGraphPath* path);

/**
 * @brief 释放子图匹配结果内存
 * 
 * @param match 匹配结果指针
 */
void subgraph_match_free(SubgraphMatch* match);

/**
 * @brief 获取所有节点
 * 
 * @param graph 知识图谱句柄
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回节点数量，如果results不为NULL，则填充节点指针
 */
size_t knowledge_graph_get_all_nodes(KnowledgeGraph* graph, KnowledgeGraphNode** results, size_t max_results);

/**
 * @brief 获取所有边
 * 
 * @param graph 知识图谱句柄
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回边数量，如果results不为NULL，则填充边指针
 */
size_t knowledge_graph_get_all_edges(KnowledgeGraph* graph, KnowledgeGraphEdge** results, size_t max_results);

/**
 * @brief 传递闭包推理 - 查找通过指定关系类型链可达的所有节点
 * 
 * 例如，如果A->B（is_a）且B->C（is_a），则A可通过is_a链到达C。
 * 使用BFS遍历，仅跟踪指定关系类型的边。
 * 
 * @param graph 知识图谱句柄
 * @param start_node 起始节点
 * @param relation_type 关系类型（边类型）
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回可达节点数（不包括起始节点）
 */
SELFLNN_API size_t knowledge_graph_transitive_closure(KnowledgeGraph* graph,
    KnowledgeGraphNode* start_node, KnowledgeGraphEdgeType relation_type,
    KnowledgeGraphNode** results, size_t max_results);

/**
 * @brief 多跳关系查询 - 查找经过N跳指定关系类型可达的节点
 * 
 * 从起始节点出发，经过min_hops到max_hops步（每步的边类型必须在relation_types中），
 * 查找所有可达的目标节点。
 * 
 * @param graph 知识图谱句柄
 * @param start_node 起始节点
 * @param relation_types 允许的关系类型数组
 * @param num_types 关系类型数量
 * @param min_hops 最小跳数
 * @param max_hops 最大跳数
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回匹配节点数
 */
SELFLNN_API size_t knowledge_graph_multi_hop_query(KnowledgeGraph* graph,
    KnowledgeGraphNode* start_node, const KnowledgeGraphEdgeType* relation_types, size_t num_types,
    size_t min_hops, size_t max_hops,
    KnowledgeGraphNode** results, size_t max_results);

/**
 * @brief 关系路径模式查询 - 查找匹配指定关系类型序列的路径
 * 
 * 从起始节点出发，按照pattern指定的关系类型序列（pattern[0], pattern[1], ...），
 * 查找所有匹配该序列的路径。
 * 
 * @param graph 知识图谱句柄
 * @param start_node 起始节点
 * @param pattern 关系类型模式数组
 * @param pattern_length 模式长度（路径跳数）
 * @param paths 路径输出缓冲区
 * @param max_paths 最大路径数
 * @return size_t 返回匹配路径数
 */
SELFLNN_API size_t knowledge_graph_relation_path_query(KnowledgeGraph* graph,
    KnowledgeGraphNode* start_node, const KnowledgeGraphEdgeType* pattern, size_t pattern_length,
    KnowledgeGraphPath** paths, size_t max_paths);

/* ============================================================================
 * SPARQL 查询解析器
 *
 * 支持 SPARQL 子集语法解析和查询执行。
 * 支持的语法：SELECT, WHERE, FILTER, OPTIONAL, LIMIT, ORDER BY
 * 支持的表达式：rdf:type, ?变量, 字面量, 数值比较
 * ============================================================================ */

#define SELFLNN_SPARQL_MAX_VARS 32
#define SELFLNN_SPARQL_MAX_PATTERNS 64
#define SELFLNN_SPARQL_MAX_FILTERS 16
#define SELFLNN_SPARQL_QUERY_MAX_LEN 4096

/**
 * @brief SPARQL 三元组模式
 */
typedef struct {
    int subject_is_var;           /**< 主语是否为变量（0=常量，1=变量） */
    char subject_value[128];      /**< 主语值（变量名或节点标签） */
    int predicate_is_var;         /**< 谓语是否为变量 */
    char predicate_value[128];    /**< 谓语值 */
    int object_is_var;            /**< 宾语是否为变量 */
    char object_value[128];       /**< 宾语值 */
    int optional;                 /**< 是否 OPTIONAL */
} SparqlTriplePattern;

/**
 * @brief SPARQL 查询结果
 */
typedef struct {
    char var_names[SELFLNN_SPARQL_MAX_VARS][64];  /**< 变量名列表 */
    size_t var_count;                              /**< 变量数量 */
    KnowledgeGraphNode* bindings[SELFLNN_SPARQL_MAX_VARS][256]; /**< 绑定结果 [var][row] */
    size_t row_count;                              /**< 结果行数 */
    float confidences[256];                        /**< 各结果置信度 */
} SparqlQueryResult;

/**
 * @brief 解析 SPARQL 查询字符串
 *
 * 解析 SELECT ?x WHERE { ... } 格式的查询，构建内部查询结构。
 *
 * @param query_str SPARQL 查询字符串
 * @param patterns [out] 解析出的三元组模式数组
 * @param max_patterns 最大模式数
 * @param pattern_count [out] 实际模式数
 * @param variables [out] 变量名数组
 * @param max_vars 最大变量数
 * @param var_count [out] 实际变量数
 * @param limit [out] LIMIT 值（0表示无限制）
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_parse_sparql(const char* query_str,
                                 SparqlTriplePattern* patterns, size_t max_patterns,
                                 size_t* pattern_count,
                                 char (*variables)[64], size_t max_vars,
                                 size_t* var_count, size_t* limit);

/**
 * @brief 执行 SPARQL 查询
 *
 * 在知识图谱上执行已解析的 SPARQL 查询。
 *
 * @param graph 知识图谱句柄
 * @param patterns 三元组模式数组
 * @param pattern_count 模式数量
 * @param variables 变量名数组
 * @param var_count 变量数量
 * @param limit 最大结果数（0表示无限制）
 * @param result [out] 查询结果
 * @return int 成功返回0，失败返回-1
 */
int knowledge_graph_execute_sparql(KnowledgeGraph* graph,
                                   const SparqlTriplePattern* patterns,
                                   size_t pattern_count,
                                   const char (*variables)[64], size_t var_count,
                                   size_t limit, SparqlQueryResult* result);

/**
 * @brief 释放 SPARQL 查询结果
 *
 * @param result 查询结果句柄
 */
void knowledge_graph_free_sparql_result(SparqlQueryResult* result);

/**
 * @brief 一键 SPARQL 查询（解析 + 执行）
 *
 * @param graph 知识图谱句柄
 * @param query_str SPARQL 查询字符串
 * @return SparqlQueryResult* 查询结果，失败返回NULL
 */
SparqlQueryResult* knowledge_graph_sparql_query(KnowledgeGraph* graph,
                                                 const char* query_str);

/* ============================================================================
 * 前端可视化 JSON 导出
 *
 * 将知识图谱导出为标准 JSON 格式，用于 D3.js / ECharts 等前端可视化库。
 * 输出格式：{ "nodes": [...], "edges": [...] }
 * ============================================================================ */

/**
 * @brief 导出知识图谱为可视化 JSON
 *
 * 将知识图谱的节点和边转换为前端可视化友好格式。
 * 节点包含：id, label, type, confidence, group
 * 边包含：source, target, label, type, weight
 *
 * @param graph 知识图谱句柄
 * @param json_buffer [out] JSON 字符串缓冲区
 * @param buffer_size 缓冲区大小
 * @param include_embeddings 是否包含嵌入向量（0=不包含，1=包含）
 * @return int 成功返回写入字符数，失败返回-1
 */
int knowledge_graph_export_visual_json(KnowledgeGraph* graph,
                                       char* json_buffer, size_t buffer_size,
                                       int include_embeddings);

/**
 * @brief 导出子图结果为可视化 JSON
 *
 * @param nodes 子图节点数组
 * @param node_count 节点数量
 * @param edges 子图边数组
 * @param edge_count 边数量
 * @param json_buffer [out] JSON 字符串缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回写入字符数，失败返回-1
 */
int knowledge_graph_subgraph_export_json(KnowledgeGraphNode** nodes, size_t node_count,
                                         KnowledgeGraphEdge** edges, size_t edge_count,
                                         char* json_buffer, size_t buffer_size);

/* ================================================================
 * K-035: 3D力导向图布局
 * ================================================================ */

int knowledge_graph_layout_3d(float* positions,
                               const int* edges_from,
                               const int* edges_to,
                               const float* edge_weights,
                               int node_count, int edge_count,
                               int iterations);

/* ================================================================
 * M-024/R002: 知识图谱→LNN桥接 —— 将图谱概念嵌入注入液态神经网络
 * ================================================================ */

/**
 * @brief 将知识图谱的概念嵌入聚合后注入LNN状态
 * 遍历所有CONCEPT/ENTITY节点，按confidence加权聚合嵌入向量，
 * 通过selflnn_consume_knowledge_inference注入LNN。
 * @param kg 知识图谱句柄
 * @param lnn LNN实例句柄
 * @param strength 扰动强度 (0.0~1.0, 推荐0.1~0.3)
 * @return 0=成功, -1=参数无效, -2=无活跃概念
 */
SELFLNN_API int knowledge_graph_to_lnn_bridge(void* kg, void* lnn, float strength);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_KNOWLEDGE_GRAPH_H