/**
 * @file graph_query.h
 * @brief 知识图谱查询引擎：SPARQL风格查询 + 图模式匹配 + 子图匹配
 *
 * 基于graph_storage三大存储引擎的统一查询层：
 * 1. SPARQL风格查询（RDFTripleStore）
 * 2. 图模式匹配（AdjacencyList）
 * 3. 子图匹配（PropertyGraph + 任意引擎）
 */

#ifndef SELFLNN_GRAPH_QUERY_H
#define SELFLNN_GRAPH_QUERY_H

#include "selflnn/knowledge/graph_storage.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============ 通用查询模式定义 ============ */

/** @brief 查询变量绑定状态 */
typedef enum {
    BOUND_CONSTANT = 0,    /**< 绑定为常量值 */
    BOUND_VARIABLE = 1     /**< 绑定为变量（需要查询时求解） */
} QueryBindingType;

/** @brief 三元组模式（SPARQL风格） */
typedef struct {
    int subject_is_var;
    char subject_value[256];
    int predicate_is_var;
    char predicate_value[256];
    int object_is_var;
    char object_value[256];
    int optional;
    float min_confidence;
} QueryTriplePattern;

/** @brief 属性约束 */
typedef struct {
    char property_key[128];
    PropertyType prop_type;
    PropertyValue prop_value;
    int is_var;
    char var_name[64];
    int op;                 /* 0=EQ, 1=NE, 2=GT, 3=GE, 4=LT, 5=LE */
} QueryPropertyConstraint;

/** @brief 图模式：节点+边+属性约束 */
typedef struct {
    char name[64];                     /* 模式名称 */
    int node_count;                    /* 模式中节点数 */
    char node_labels[32][128];         /* 各节点标签（空=任意） */
    int edge_count;                    /* 模式中边数 */
    int edge_sources[64];              /* 各边源节点索引 */
    int edge_targets[64];              /* 各边目标节点索引 */
    char edge_labels[64][128];         /* 各边标签 */
    int edge_directed[64];             /* 各边是否有向 */
    QueryPropertyConstraint node_constraints[32][8]; /* 节点属性约束 */
    size_t node_constraint_counts[32];
    QueryPropertyConstraint edge_constraints[64][8]; /* 边属性约束 */
    size_t edge_constraint_counts[64];
} QueryGraphPattern;

/* ============ 查询结果定义 ============ */

#define QUERY_MAX_VARS 32
#define QUERY_MAX_BINDINGS 512

/** @brief 单个变量绑定（节点或值） */
typedef struct {
    int bound_type;          /* 0=未绑定, 1=节点ID, 2=属性值, 3=RDF节点ID */
    int node_id;             /* AdjacencyList/PropertyGraph节点ID */
    int rdf_node_id;         /* RDF节点ID */
    PropertyValue value;     /* 字面量值 */
    float confidence;
} QueryBinding;

/** @brief 查询结果行 */
typedef struct {
    QueryBinding variables[QUERY_MAX_VARS];
    float confidence;
} QueryResultRow;

/** @brief 查询结果集 */
typedef struct {
    char var_names[QUERY_MAX_VARS][64];
    size_t var_count;
    QueryResultRow rows[QUERY_MAX_BINDINGS];
    size_t row_count;
    size_t capacity;
    int sorted;
} QueryResultSet;

/** @brief 子图匹配结果 */
typedef struct {
    int* matched_node_ids;     /* 匹配到的节点ID数组 */
    size_t node_count;
    int source_engine_type;    /* 0=AdjacencyList, 1=PropertyGraph */
    float similarity;
    char** matched_labels;     /* 各节点对应模式标签 */
} SubgraphMatchResult;

/** @brief 子图匹配结果集 */
typedef struct {
    SubgraphMatchResult* matches;
    size_t match_count;
    size_t capacity;
} SubgraphMatchSet;

/* ============ 查询选项 ============ */

/** @brief 查询引擎选项 */
typedef struct {
    size_t max_results;
    float min_confidence;
    int use_index;
    int enable_optimization;
    int timeout_ms;
    int sort_by_confidence;
} QueryOptions;

/* ============ SPARQL风格查询（RDFTripleStore） ============ */

/**
 * @brief 解析SPARQL查询字符串为内部模式
 * @param query_str SPARQL查询字符串
 * @param patterns 输出模式数组
 * @param max_patterns 最大模式数
 * @param pattern_count 输出实际模式数
 * @param variables 输出变量名数组
 * @param max_vars 最大变量数
 * @param var_count 输出实际变量数
 * @param limit 输出LIMIT值
 * @return int 成功0，失败-1
 */
int graph_query_parse_sparql(const char* query_str,
                             QueryTriplePattern* patterns, size_t max_patterns,
                             size_t* pattern_count,
                             char (*variables)[64], size_t max_vars,
                             size_t* var_count, size_t* limit);

/**
 * @brief 在RDF三元组存储上执行SPARQL查询
 * @param store RDF三元组存储
 * @param patterns 三元组模式
 * @param pattern_count 模式数
 * @param variables 变量名
 * @param var_count 变量数
 * @param limit 最大结果数(0=无限制)
 * @param options 查询选项(NULL使用默认)
 * @return QueryResultSet* 结果集，失败NULL
 */
QueryResultSet* graph_query_execute_sparql(RDFTripleStore* store,
                                           const QueryTriplePattern* patterns,
                                           size_t pattern_count,
                                           const char (*variables)[64],
                                           size_t var_count, size_t limit,
                                           const QueryOptions* options);

/**
 * @brief 一键SPARQL查询
 * @param store RDF三元组存储
 * @param query_str SPARQL查询字符串
 * @return QueryResultSet* 结果集，失败NULL
 */
QueryResultSet* graph_query_sparql(RDFTripleStore* store, const char* query_str);

/* ============ 图模式匹配（AdjacencyList） ============ */

/**
 * @brief 在邻接表上执行图模式匹配
 * @param al 邻接表
 * @param pattern 查询图模式
 * @param start_node_id 起始节点ID（-1自动选择）
 * @param options 查询选项
 * @return SubgraphMatchSet* 匹配结果集
 */
SubgraphMatchSet* graph_query_match_pattern(AdjacencyList* al,
                                            const QueryGraphPattern* pattern,
                                            int start_node_id,
                                            const QueryOptions* options);

/**
 * @brief 在邻接表上查找指定标签的星形模式
 * @param al 邻接表
 * @param center_label 中心节点标签
 * @param neighbor_labels 邻居节点标签（NULL表示任意）
 * @param neighbor_count 邻居标签数
 * @param max_distance 最大距离
 * @param options 查询选项
 * @return SubgraphMatchSet* 匹配集
 */
SubgraphMatchSet* graph_query_find_star_pattern(AdjacencyList* al,
                                                const char* center_label,
                                                const char** neighbor_labels,
                                                size_t neighbor_count,
                                                int max_distance,
                                                const QueryOptions* options);

/**
 * @brief 在邻接表上查找路径模式
 * @param al 邻接表
 * @param start_label 起始标签
 * @param path_edge_labels 路径边标签序列
 * @param path_length 路径长度
 * @param min_confidence 最小置信度
 * @return SubgraphMatchSet* 匹配集
 */
SubgraphMatchSet* graph_query_find_path_pattern(AdjacencyList* al,
                                                const char* start_label,
                                                const char** path_edge_labels,
                                                size_t path_length,
                                                float min_confidence);

/* ============ 子图匹配（PropertyGraph） ============ */

/**
 * @brief 在属性图上执行VF2风格子图匹配
 * @param pg 属性图
 * @param pattern 查询图模式
 * @param options 查询选项
 * @return SubgraphMatchSet* 匹配集
 */
SubgraphMatchSet* graph_query_subgraph_isomorphism(PropertyGraph* pg,
                                                   const QueryGraphPattern* pattern,
                                                   const QueryOptions* options);

/**
 * @brief 在属性图上查找与示例节点相似的子图
 * @param pg 属性图
 * @param example_node_id 示例节点ID
 * @param max_distance 扩展距离
 * @param similarity_threshold 相似度阈值
 * @param options 查询选项
 * @return SubgraphMatchSet* 匹配集
 */
SubgraphMatchSet* graph_query_find_similar_subgraph(PropertyGraph* pg,
                                                    int example_node_id,
                                                    int max_distance,
                                                    float similarity_threshold,
                                                    const QueryOptions* options);

/**
 * @brief 在属性图上执行属性约束查询
 * @param pg 属性图
 * @param constraints 节点属性约束数组
 * @param constraint_count 约束数
 * @param edge_label 边标签（NULL=任意）
 * @param max_hops 最大跳数
 * @param options 查询选项
 * @return QueryResultSet* 匹配节点结果集
 */
QueryResultSet* graph_query_by_property(PropertyGraph* pg,
                                        const QueryPropertyConstraint* constraints,
                                        size_t constraint_count,
                                        const char* edge_label,
                                        int max_hops,
                                        const QueryOptions* options);

/* ============ 结果集操作 ============ */

/**
 * @brief 创建空结果集
 * @param var_count 变量数
 * @param var_names 变量名数组
 * @return QueryResultSet* 结果集
 */
QueryResultSet* query_result_set_create(size_t var_count,
                                        const char (*var_names)[64]);

/**
 * @brief 结果集添加行
 * @param rs 结果集
 * @param row 行数据
 * @return int 成功0
 */
int query_result_set_add_row(QueryResultSet* rs, const QueryResultRow* row);

/**
 * @brief 结果集合并（UNION）
 * @param rs1 结果集1
 * @param rs2 结果集2
 * @return QueryResultSet* 合并结果
 */
QueryResultSet* query_result_set_union(const QueryResultSet* rs1,
                                       const QueryResultSet* rs2);

/**
 * @brief 结果集连接（JOIN）
 * @param rs1 结果集1
 * @param rs2 结果集2
 * @return QueryResultSet* 连接结果
 */
QueryResultSet* query_result_set_join(const QueryResultSet* rs1,
                                      const QueryResultSet* rs2);

/**
 * @brief 结果集投影
 * @param rs 原结果集
 * @param var_names 要保留的变量
 * @param var_count 变量数
 * @return QueryResultSet* 投影结果
 */
QueryResultSet* query_result_set_project(const QueryResultSet* rs,
                                         const char (*var_names)[64],
                                         size_t var_count);

/**
 * @brief 结果集排序
 * @param rs 结果集
 * @param var_name 排序依据变量
 * @param ascending 是否升序
 */
void query_result_set_sort(QueryResultSet* rs, const char* var_name,
                           int ascending);

/**
 * @brief 结果集限制
 * @param rs 结果集
 * @param limit 最大行数
 */
void query_result_set_limit(QueryResultSet* rs, size_t limit);

/**
 * @brief 结果集过滤
 * @param rs 结果集
 * @param var_name 变量名
 * @param op 操作符(0=EQ,1=NE,2=GT,3=GE,4=LT,5=LE)
 * @param value 比较值
 */
void query_result_set_filter(QueryResultSet* rs, const char* var_name,
                             int op, const PropertyValue* value);

/**
 * @brief 释放结果集
 * @param rs 结果集
 */
void query_result_set_free(QueryResultSet* rs);

/**
 * @brief 释放子图匹配集
 * @param set 匹配集
 */
void subgraph_match_set_free(SubgraphMatchSet* set);

/**
 * @brief 创建默认查询选项
 * @return QueryOptions 默认选项
 */
QueryOptions query_options_default(void);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GRAPH_QUERY_H */
