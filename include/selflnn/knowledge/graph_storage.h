/**
 * @file graph_storage.h
 * @brief 知识图谱存储引擎：邻接表 + 属性图 + RDF三元组
 *
 * 提供三种互补的图存储模型：
 * 1. 邻接表存储（AdjacencyList）：基于邻接表的图结构，高效支持邻居查询和遍历
 * 2. 属性图存储（PropertyGraph）：节点和边带键值对属性，支持属性索引和约束
 * 3. RDF三元组存储（RDFTripleStore）：SPO三元组模型，支持SPO/OSP/POS多索引查询
 */

#ifndef SELFLNN_GRAPH_STORAGE_H
#define SELFLNN_GRAPH_STORAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============ 属性值类型 ============ */

/** @brief 属性值数据类型 */
typedef enum {
    PROP_TYPE_INT = 0,       /**< 整数 */
    PROP_TYPE_FLOAT = 1,     /**< 浮点数 */
    PROP_TYPE_STRING = 2,    /**< 字符串 */
    PROP_TYPE_BOOL = 3,      /**< 布尔值 */
    PROP_TYPE_NONE = 4       /**< 空值 */
} PropertyType;

/** @brief 属性值联合体 */
typedef struct {
    PropertyType type;
    union {
        int64_t int_val;
        double float_val;
        int bool_val;
    } data;
    char* str_val;           /**< 字符串值（堆分配） */
} PropertyValue;

/** @brief 键值对属性 */
typedef struct {
    char* key;               /**< 属性键 */
    PropertyValue value;     /**< 属性值 */
} Property;

/** @brief 属性列表 */
typedef struct {
    Property* items;         /**< 属性数组 */
    size_t count;            /**< 属性数量 */
    size_t capacity;         /**< 属性容量 */
} PropertyList;

/* ============ 邻接表存储引擎 ============ */

/** @brief 邻接表节点 */
typedef struct ALNode {
    int id;                  /**< 节点唯一ID */
    char* label;             /**< 节点标签 */
    int* out_neighbors;      /**< 出边邻居节点ID数组 */
    float* out_weights;      /**< 出边权重数组 */
    int* out_edge_ids;       /**< 出边ID数组 */
    size_t out_degree;       /**< 出度 */
    size_t out_capacity;     /**< 出边容量 */
    int* in_neighbors;       /**< 入边邻居节点ID数组 */
    float* in_weights;       /**< 入边权重数组 */
    int* in_edge_ids;        /**< 入边ID数组 */
    size_t in_degree;        /**< 入度 */
    size_t in_capacity;      /**< 入边容量 */
    void* user_data;         /**< 用户数据 */
} ALNode;

/** @brief 邻接表边 */
typedef struct ALEdge {
    int id;                  /**< 边唯一ID */
    int source_id;           /**< 源节点ID */
    int target_id;           /**< 目标节点ID */
    char* label;             /**< 边标签 */
    float weight;            /**< 边权重 */
    int directed;            /**< 是否有向（0=无向，1=有向） */
} ALEdge;

/** @brief 邻接表图句柄 */
typedef struct AdjacencyList AdjacencyList;

/**
 * @brief 创建邻接表图
 * @param initial_node_capacity 初始节点容量
 * @return AdjacencyList* 句柄，失败返回NULL
 */
AdjacencyList* adjacency_list_create(size_t initial_node_capacity);

/**
 * @brief 释放邻接表图
 * @param al 句柄
 */
void adjacency_list_free(AdjacencyList* al);

/**
 * @brief 添加节点
 * @param al 句柄
 * @param label 节点标签（可为NULL）
 * @return int 节点ID，失败返回-1
 */
int adjacency_list_add_node(AdjacencyList* al, const char* label);

/**
 * @brief 添加有向边
 * @param al 句柄
 * @param source_id 源节点ID
 * @param target_id 目标节点ID
 * @param label 边标签（可为NULL）
 * @param weight 边权重
 * @return int 边ID，失败返回-1
 */
int adjacency_list_add_edge(AdjacencyList* al, int source_id, int target_id,
                            const char* label, float weight);

/**
 * @brief 添加无向边
 * @param al 句柄
 * @param node_a 节点A
 * @param node_b 节点B
 * @param label 边标签
 * @param weight 边权重
 * @return int 边ID，失败返回-1
 */
int adjacency_list_add_undirected_edge(AdjacencyList* al, int node_a, int node_b,
                                       const char* label, float weight);

/**
 * @brief 获取节点的出边邻居
 * @param al 句柄
 * @param node_id 节点ID
 * @param neighbors 输出缓冲区（节点ID数组）
 * @param weights 输出缓冲区（权重数组，可为NULL）
 * @param max_count 缓冲区大小
 * @return int 实际邻居数，失败返回-1
 */
int adjacency_list_get_out_neighbors(AdjacencyList* al, int node_id,
                                     int* neighbors, float* weights,
                                     size_t max_count);

/**
 * @brief 获取节点的入边邻居
 * @param al 句柄
 * @param node_id 节点ID
 * @param neighbors 输出缓冲区
 * @param weights 输出缓冲区（可为NULL）
 * @param max_count 缓冲区大小
 * @return int 实际邻居数，失败返回-1
 */
int adjacency_list_get_in_neighbors(AdjacencyList* al, int node_id,
                                    int* neighbors, float* weights,
                                    size_t max_count);

/**
 * @brief 获取节点度数
 * @param al 句柄
 * @param node_id 节点ID
 * @param out_degree 出度（输出参数）
 * @param in_degree 入度（输出参数）
 * @return int 成功返回0，失败返回-1
 */
int adjacency_list_get_degree(AdjacencyList* al, int node_id,
                              size_t* out_degree, size_t* in_degree);

/**
 * @brief 获取节点数
 * @param al 句柄
 * @return size_t 节点数
 */
size_t adjacency_list_node_count(AdjacencyList* al);

/**
 * @brief 获取边数
 * @param al 句柄
 * @return size_t 边数
 */
size_t adjacency_list_edge_count(AdjacencyList* al);

/**
 * @brief 查找节点ID通过标签
 * @param al 句柄
 * @param label 标签
 * @param results 结果缓冲区
 * @param max_results 最大结果数
 * @return int 匹配数
 */
int adjacency_list_find_by_label(AdjacencyList* al, const char* label,
                                 int* results, size_t max_results);

/**
 * @brief 深度优先遍历
 * @param al 句柄
 * @param start_id 起始节点ID
 * @param callback 回调(node_id, user_data)
 * @param user_data 用户数据
 * @return int 成功返回0
 */
int adjacency_list_dfs(AdjacencyList* al, int start_id,
                       int (*callback)(int, void*), void* user_data);

/**
 * @brief 广度优先遍历
 * @param al 句柄
 * @param start_id 起始节点ID
 * @param callback 回调(node_id, user_data)
 * @param user_data 用户数据
 * @return int 成功返回0
 */
int adjacency_list_bfs(AdjacencyList* al, int start_id,
                       int (*callback)(int, void*), void* user_data);

/**
 * @brief 检查边是否存在
 * @param al 句柄
 * @param source_id 源节点
 * @param target_id 目标节点
 * @return int 存在返回1，不存在返回0，失败返回-1
 */
int adjacency_list_has_edge(AdjacencyList* al, int source_id, int target_id);

/**
 * @brief 通过边ID获取邻接表边指针
 * @param al 句柄
 * @param edge_id 边ID
 * @return const ALEdge* 边指针，不存在返回NULL
 */
const ALEdge* adjacency_list_get_edge_by_id(AdjacencyList* al, int edge_id);

/* ============ 属性图存储引擎 ============ */

/** @brief 属性图节点 */
typedef struct PGNode {
    int id;                  /**< 节点ID */
    char* label;             /**< 节点标签 */
    PropertyList properties; /**< 节点属性列表 */
} PGNode;

/** @brief 属性图边 */
typedef struct PGEdge {
    int id;                  /**< 边ID */
    int source_id;           /**< 源节点ID */
    int target_id;           /**< 目标节点ID */
    char* label;             /**< 边标签 */
    float weight;            /**< 边权重 */
    PropertyList properties; /**< 边属性列表 */
    int directed;            /**< 是否有向 */
} PGEdge;

/** @brief 属性图句柄 */
typedef struct PropertyGraph PropertyGraph;

/**
 * @brief 创建属性图
 * @param initial_capacity 初始节点容量
 * @return PropertyGraph* 句柄，失败返回NULL
 */
PropertyGraph* property_graph_create(size_t initial_capacity);

/**
 * @brief 释放属性图
 * @param pg 句柄
 */
void property_graph_free(PropertyGraph* pg);

/**
 * @brief 添加节点
 * @param pg 句柄
 * @param label 节点标签
 * @return int 节点ID，失败返回-1
 */
int property_graph_add_node(PropertyGraph* pg, const char* label);

/**
 * @brief 添加边
 * @param pg 句柄
 * @param source_id 源节点ID
 * @param target_id 目标节点ID
 * @param label 边标签
 * @param weight 边权重
 * @param directed 是否有向
 * @return int 边ID，失败返回-1
 */
int property_graph_add_edge(PropertyGraph* pg, int source_id, int target_id,
                            const char* label, float weight, int directed);

/**
 * @brief 设置节点属性（int）
 * @param pg 句柄
 * @param node_id 节点ID
 * @param key 属性键
 * @param value 属性值
 * @return int 成功返回0，失败返回-1
 */
int property_graph_set_node_property_int(PropertyGraph* pg, int node_id,
                                         const char* key, int64_t value);

/**
 * @brief 设置节点属性（float）
 * @param pg 句柄
 * @param node_id 节点ID
 * @param key 属性键
 * @param value 属性值
 * @return int 成功返回0，失败返回-1
 */
int property_graph_set_node_property_float(PropertyGraph* pg, int node_id,
                                           const char* key, double value);

/**
 * @brief 设置节点属性（string）
 * @param pg 句柄
 * @param node_id 节点ID
 * @param key 属性键
 * @param value 属性值
 * @return int 成功返回0，失败返回-1
 */
int property_graph_set_node_property_string(PropertyGraph* pg, int node_id,
                                            const char* key, const char* value);

/**
 * @brief 获取节点属性值
 * @param pg 句柄
 * @param node_id 节点ID
 * @param key 属性键
 * @param value 属性值输出
 * @return int 成功返回0，未找到返回1，失败返回-1
 */
int property_graph_get_node_property(PropertyGraph* pg, int node_id,
                                     const char* key, PropertyValue* value);

/**
 * @brief 设置边属性（int）
 * @param pg 句柄
 * @param edge_id 边ID
 * @param key 属性键
 * @param value 属性值
 * @return int 成功返回0
 */
int property_graph_set_edge_property_int(PropertyGraph* pg, int edge_id,
                                         const char* key, int64_t value);

/**
 * @brief 设置边属性（float）
 * @param pg 句柄
 * @param edge_id 边ID
 * @param key 属性键
 * @param value 属性值
 * @return int 成功返回0
 */
int property_graph_set_edge_property_float(PropertyGraph* pg, int edge_id,
                                           const char* key, double value);

/**
 * @brief 设置边属性（string）
 * @param pg 句柄
 * @param edge_id 边ID
 * @param key 属性键
 * @param value 属性值
 * @return int 成功返回0
 */
int property_graph_set_edge_property_string(PropertyGraph* pg, int edge_id,
                                            const char* key, const char* value);

/**
 * @brief 获取边属性值
 * @param pg 句柄
 * @param edge_id 边ID
 * @param key 属性键
 * @param value 输出
 * @return int 成功返回0，未找到返回1
 */
int property_graph_get_edge_property(PropertyGraph* pg, int edge_id,
                                     const char* key, PropertyValue* value);

/**
 * @brief 按属性键值查询节点
 * @param pg 句柄
 * @param key 属性键
 * @param value 属性值
 * @param results 结果缓冲区
 * @param max_results 最大结果数
 * @return int 匹配数
 */
int property_graph_find_nodes_by_property(PropertyGraph* pg, const char* key,
                                          const PropertyValue* value,
                                          int* results, size_t max_results);

/**
 * @brief 按标签查询节点
 * @param pg 句柄
 * @param label 标签
 * @param results 结果缓冲区
 * @param max_results 最大结果数
 * @return int 匹配数
 */
int property_graph_find_nodes_by_label(PropertyGraph* pg, const char* label,
                                       int* results, size_t max_results);

/**
 * @brief 获取属性图统计
 * @param pg 句柄
 * @param node_count 节点数输出
 * @param edge_count 边数输出
 * @return int 成功返回0
 */
int property_graph_get_stats(PropertyGraph* pg, size_t* node_count, size_t* edge_count);

/**
 * @brief 保存属性图到文件
 * @param pg 句柄
 * @param filename 文件名
 * @return int 成功返回0
 */
int property_graph_save(PropertyGraph* pg, const char* filename);

/**
 * @brief 从文件加载属性图
 * @param filename 文件名
 * @return PropertyGraph* 句柄，失败返回NULL
 */
PropertyGraph* property_graph_load(const char* filename);

/**
 * @brief 通过节点ID获取属性图节点指针
 * @param pg 句柄
 * @param node_id 节点ID
 * @return const PGNode* 节点指针，不存在返回NULL
 */
const PGNode* property_graph_get_node(PropertyGraph* pg, int node_id);

/**
 * @brief 检查两个属性图节点间是否存在边
 * @param pg 句柄
 * @param src_node_id 源节点ID
 * @param tgt_node_id 目标节点ID
 * @return int 存在边返回1，不存在返回0，失败返回-1
 */
int property_graph_has_edge(PropertyGraph* pg, int src_node_id, int tgt_node_id);

/**
 * @brief 获取两个节点间的边（第一条匹配）
 * @param pg 句柄
 * @param src_node_id 源节点ID
 * @param tgt_node_id 目标节点ID
 * @return const PGEdge* 边指针，不存在返回NULL
 */
const PGEdge* property_graph_get_edge_between(PropertyGraph* pg,
                                               int src_node_id,
                                               int tgt_node_id);

/* ============ 邻接表增强访问器 ============ */

/**
 * @brief 通过节点ID获取邻接表节点指针
 * @param al 句柄
 * @param node_id 节点ID
 * @return const ALNode* 节点指针，不存在返回NULL
 */
const ALNode* adjacency_list_get_node(AdjacencyList* al, int node_id);

/**
 * @brief 获取邻接表分配的最大节点ID范围
 * @param al 句柄
 * @return int 最大节点ID（可用于 0 到 max-1 的迭代）
 */
int adjacency_list_get_node_capacity(AdjacencyList* al);

/**
 * @brief 获取属性图分配的最大节点ID范围
 * @param pg 句柄
 * @return int 最大节点ID（可用于 0 到 max-1 的迭代）
 */
int property_graph_get_node_capacity(PropertyGraph* pg);

/**
 * @brief 获取属性图中指定节点的边度数
 * 
 * 计算与指定节点相连的边数量（包括出边和入边）。
 * 此函数用于VF2子图同构的度数剪枝优化。
 *
 * @param pg 属性图句柄
 * @param node_id 节点ID
 * @return int 边度数，节点不存在返回-1
 */
int property_graph_node_degree(PropertyGraph* pg, int node_id);

/* ============ RDF三元组存储引擎 ============ */

/** @brief RDF节点类型 */
typedef enum {
    RDF_NODE_IRI = 0,        /**< IRI资源 */
    RDF_NODE_BLANK = 1,      /**< 空白节点 */
    RDF_NODE_LITERAL = 2     /**< 字面量 */
} RDFNodeType;

/** @brief RDF节点 */
typedef struct RDFNode {
    int id;                  /**< 节点内部ID */
    RDFNodeType node_type;   /**< 节点类型 */
    char* value;             /**< 节点值（IRI字符串/字面量/空白ID） */
    char* datatype_iri;      /**< 字面量数据类型IRI（字面量时使用） */
    char* lang_tag;          /**< 语言标签（字面量时使用） */
} RDFNode;

/** @brief RDF三元组 */
typedef struct RDFTriple {
    int id;                  /**< 三元组ID */
    int subject_id;          /**< 主语ID */
    int predicate_id;        /**< 谓语ID */
    int object_id;           /**< 宾语ID */
    float confidence;        /**< 置信度 */
} RDFTriple;

/** @brief 命名空间前缀映射 */
typedef struct RDFNamespace {
    char* prefix;            /**< 前缀 */
    char* iri;               /**< IRI */
} RDFNamespace;

/** @brief RDF三元组存储句柄 */
typedef struct RDFTripleStore RDFTripleStore;

/**
 * @brief 创建RDF三元组存储
 * @return RDFTripleStore* 句柄，失败返回NULL
 */
RDFTripleStore* rdf_triple_store_create(void);

/**
 * @brief 释放RDF三元组存储
 * @param store 句柄
 */
void rdf_triple_store_free(RDFTripleStore* store);

/**
 * @brief 注册命名空间前缀
 * @param store 句柄
 * @param prefix 前缀
 * @param iri IRI
 * @return int 成功返回0
 */
int rdf_triple_store_add_prefix(RDFTripleStore* store, const char* prefix, const char* iri);

/**
 * @brief 查找前缀对应的IRI
 * @param store 句柄
 * @param prefix 前缀
 * @return const char* IRI，未找到返回NULL
 */
const char* rdf_triple_store_get_prefix_iri(RDFTripleStore* store, const char* prefix);

/**
 * @brief 创建或获取RDF节点
 * @param store 句柄
 * @param node_type 节点类型
 * @param value 节点值
 * @param datatype_iri 数据类型IRI（可为NULL）
 * @param lang_tag 语言标签（可为NULL）
 * @return int 节点ID，失败返回-1
 */
int rdf_triple_store_get_node(RDFTripleStore* store, RDFNodeType node_type,
                              const char* value, const char* datatype_iri,
                              const char* lang_tag);

/**
 * @brief 添加三元组
 * @param store 句柄
 * @param subject_id 主语节点ID
 * @param predicate_id 谓语节点ID
 * @param object_id 宾语节点ID
 * @param confidence 置信度
 * @return int 三元组ID，失败返回-1
 */
int rdf_triple_store_add_triple(RDFTripleStore* store, int subject_id,
                                int predicate_id, int object_id, float confidence);

/**
 * @brief 根据主语查询三元组
 * @param store 句柄
 * @param subject_id 主语ID（-1表示任意）
 * @param predicate_id 谓语ID（-1表示任意）
 * @param object_id 宾语ID（-1表示任意）
 * @param results 结果缓冲区
 * @param max_results 最大结果数
 * @return int 匹配数
 */
int rdf_triple_store_query(RDFTripleStore* store, int subject_id,
                           int predicate_id, int object_id,
                           RDFTriple* results, size_t max_results);

/**
 * @brief 直接添加三元组（通过字符串值）
 * @param store 句柄
 * @param subject 主语值
 * @param predicate 谓语值
 * @param object_val 宾语值
 * @param confidence 置信度
 * @return int 三元组ID，失败返回-1
 */
int rdf_triple_store_add_triple_by_values(RDFTripleStore* store,
                                          const char* subject, const char* predicate,
                                          const char* object_val, float confidence);

/**
 * @brief 根据值查询三元组
 * @param store 句柄
 * @param subject 主语值（NULL表示任意）
 * @param predicate 谓语值（NULL表示任意）
 * @param object_val 宾语值（NULL表示任意）
 * @param results 结果缓冲区
 * @param max_results 最大结果数
 * @return int 匹配数
 */
int rdf_triple_store_query_by_values(RDFTripleStore* store,
                                     const char* subject, const char* predicate,
                                     const char* object_val,
                                     RDFTriple* results, size_t max_results);

/**
 * @brief 获取RDF节点对象
 * @param store 句柄
 * @param node_id 节点ID
 * @return const RDFNode* 节点指针，失败返回NULL
 */
const RDFNode* rdf_triple_store_get_node_by_id(RDFTripleStore* store, int node_id);

/**
 * @brief 获取三元组数
 * @param store 句柄
 * @return size_t 三元组数
 */
size_t rdf_triple_store_count(RDFTripleStore* store);

/**
 * @brief 获取节点数
 * @param store 句柄
 * @return size_t 节点数
 */
size_t rdf_triple_store_node_count(RDFTripleStore* store);

/**
 * @brief 导出到N-Triples格式
 * @param store 句柄
 * @param filename 文件名
 * @return int 成功返回0
 */
int rdf_triple_store_export_ntriples(RDFTripleStore* store, const char* filename);

/**
 * @brief 导出到Turtle格式
 * @param store 句柄
 * @param filename 文件名
 * @return int 成功返回0
 */
int rdf_triple_store_export_turtle(RDFTripleStore* store, const char* filename);

/**
 * @brief 从N-Triples文件导入
 * @param store 句柄
 * @param filename 文件名
 * @return int 成功返回导入的三元组数，失败返回-1
 */
int rdf_triple_store_import_ntriples(RDFTripleStore* store, const char* filename);

/**
 * @brief 保存RDF存储到二进制文件
 * @param store 句柄
 * @param filename 文件名
 * @return int 成功返回0
 */
int rdf_triple_store_save(RDFTripleStore* store, const char* filename);

/**
 * @brief 从二进制文件加载RDF存储
 * @param filename 文件名
 * @return RDFTripleStore* 句柄，失败返回NULL
 */
RDFTripleStore* rdf_triple_store_load(const char* filename);

/** @brief 释放属性值 */
void property_value_free(PropertyValue* value);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GRAPH_STORAGE_H */
