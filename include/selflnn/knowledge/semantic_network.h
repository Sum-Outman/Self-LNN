/**
 * @file semantic_network.h
 * @brief 语义网络系统接口
 * 
 * 语义网络数据结构、概念关系、层次结构和语义相似性计算。
 * 支持概念分类、关系推理、语义距离计算等高级功能。
 */

#ifndef SELFLNN_SEMANTIC_NETWORK_H
#define SELFLNN_SEMANTIC_NETWORK_H

#include "selflnn/knowledge/knowledge.h"
#include "selflnn/knowledge/knowledge_graph.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 概念类型
 */
typedef enum {
    CONCEPT_TYPE_ENTITY = 0,        /**< 实体概念（具体事物） */
    CONCEPT_TYPE_CLASS = 1,         /**< 类概念（抽象类别） */
    CONCEPT_TYPE_PROPERTY = 2,      /**< 属性概念 */
    CONCEPT_TYPE_EVENT = 3,         /**< 事件概念 */
    CONCEPT_TYPE_RELATION = 4       /**< 关系概念 */
} ConceptType;

/**
 * @brief 语义关系类型
 */
typedef enum {
    RELATION_IS_A = 0,              /**< "是"关系（继承） */
    RELATION_PART_OF = 1,           /**< "部分"关系（组成） */
    RELATION_HAS_A = 2,             /**< "有"关系（拥有） */
    RELATION_INSTANCE_OF = 3,       /**< "实例"关系 */
    RELATION_SYNONYM = 4,           /**< 同义关系 */
    RELATION_ANTONYM = 5,           /**< 反义关系 */
    RELATION_RELATED_TO = 6,        /**< 相关关系 */
    RELATION_CAUSES = 7,            /**< 因果关系 */
    RELATION_PRECEDES = 8,          /**< 时间先后关系 */
    RELATION_LOCATION_OF = 9        /**< 位置关系 */
} SemanticRelationType;

/**
 * @brief 将语义关系类型转换为字符串
 * 
 * @param type 语义关系类型
 * @return const char* 关系类型字符串表示
 */
const char* semantic_relation_type_to_string(SemanticRelationType type);

/**
 * @brief 概念结构
 */
typedef struct {
    int id;                         /**< 概念ID */
    ConceptType type;               /**< 概念类型 */
    char* name;                     /**< 概念名称 */
    char* description;              /**< 概念描述 */
    float* embedding;               /**< 语义嵌入向量 */
    size_t embedding_size;          /**< 嵌入向量大小 */
    float specificity;              /**< 概念特异性（0-1） */
    float typicality;               /**< 概念典型性（0-1） */
    float confidence;               /**< 概念置信度（0-1） */
    int instance_count;             /**< 实例数量 */
    void* user_data;                /**< 用户数据指针 */
} Concept;

/**
 * @brief 语义关系结构
 */
typedef struct {
    int id;                         /**< 关系ID */
    SemanticRelationType type;      /**< 关系类型 */
    Concept* source;                /**< 源概念 */
    Concept* target;                /**< 目标概念 */
    char* label;                    /**< 关系标签 */
    float strength;                 /**< 关系强度（0-1） */
    float confidence;               /**< 关系置信度（0-1） */
    void* user_data;                /**< 用户数据指针 */
} SemanticRelation;

/**
 * @brief 语义网络句柄
 */
typedef struct SemanticNetwork SemanticNetwork;

/**
 * @brief 语义相似性度量方法
 */
typedef enum {
    SIMILARITY_COSINE = 0,          /**< 余弦相似度 */
    SIMILARITY_EUCLIDEAN = 1,       /**< 欧几里得距离 */
    SIMILARITY_JACCARD = 2,         /**< Jaccard相似度 */
    SIMILARITY_PATH = 3,            /**< 路径相似度 */
    SIMILARITY_WUP = 4              /**< Wu-Palmer相似度 */
} SimilarityMetric;

/**
 * @brief 层次结构查询选项
 */
typedef struct {
    int max_depth;                  /**< 最大搜索深度 */
    float min_strength;             /**< 最小关系强度 */
    int include_instances;          /**< 是否包含实例 */
    int include_subclasses;         /**< 是否包含子类 */
    size_t max_results;             /**< 最大结果数 */
} HierarchyQueryOptions;

/**
 * @brief 语义网络统计信息
 */
typedef struct {
    size_t concept_count;           /**< 概念数量 */
    size_t relation_count;          /**< 关系数量 */
    float avg_hierarchy_depth;      /**< 平均层次深度 */
    float network_density;          /**< 网络密度 */
    float avg_semantic_similarity;  /**< 平均语义相似度 */
} SemanticNetworkStats;

/**
 * @brief 创建语义网络
 * 
 * @param max_concepts 最大概念数（0表示无限制）
 * @param max_relations 最大关系数（0表示无限制）
 * @return SemanticNetwork* 语义网络句柄，失败返回NULL
 */
SemanticNetwork* semantic_network_create(size_t max_concepts, size_t max_relations);

/**
 * @brief 释放语义网络
 * 
 * @param network 语义网络句柄
 */
void semantic_network_free(SemanticNetwork* network);

/**
 * @brief 添加概念到语义网络
 * 
 * @param network 语义网络句柄
 * @param type 概念类型
 * @param name 概念名称
 * @param description 概念描述
 * @param embedding 语义嵌入向量（可选）
 * @param embedding_size 嵌入向量大小
 * @param specificity 概念特异性
 * @param typicality 概念典型性
 * @return Concept* 创建的概念指针，失败返回NULL
 */
Concept* semantic_network_add_concept(SemanticNetwork* network, ConceptType type,
                                     const char* name, const char* description,
                                     const float* embedding, size_t embedding_size,
                                     float specificity, float typicality);

/**
 * @brief 添加关系到语义网络
 * 
 * @param network 语义网络句柄
 * @param type 关系类型
 * @param source 源概念
 * @param target 目标概念
 * @param label 关系标签（可选）
 * @param strength 关系强度
 * @param confidence 关系置信度
 * @return SemanticRelation* 创建的关系指针，失败返回NULL
 */
SemanticRelation* semantic_network_add_relation(SemanticNetwork* network,
                                               SemanticRelationType type,
                                               Concept* source, Concept* target,
                                               const char* label,
                                               float strength, float confidence);

/**
 * @brief 根据名称查找概念
 * 
 * @param network 语义网络句柄
 * @param name 概念名称
 * @return Concept* 概念指针，未找到返回NULL
 */
Concept* semantic_network_find_concept_by_name(SemanticNetwork* network, const char* name);

/**
 * @brief 根据ID查找概念
 * 
 * @param network 语义网络句柄
 * @param concept_id 概念ID
 * @return Concept* 概念指针，未找到返回NULL
 */
Concept* semantic_network_find_concept_by_id(SemanticNetwork* network, int concept_id);

/**
 * @brief 查找概念的所有直接关系
 * 
 * @param network 语义网络句柄
 * @param concept 概念
 * @param relation_type 关系类型过滤（-1表示不过滤）
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回匹配的关系数
 */
size_t semantic_network_find_relations(SemanticNetwork* network, Concept* concept,
                                      int relation_type,
                                      SemanticRelation** results, size_t max_results);

/**
 * @brief 计算两个概念的语义相似度
 * 
 * @param network 语义网络句柄
 * @param concept1 概念1
 * @param concept2 概念2
 * @param metric 相似度度量方法
 * @return float 相似度得分（0-1），失败返回-1
 */
float semantic_network_concept_similarity(SemanticNetwork* network,
                                         Concept* concept1, Concept* concept2,
                                         SimilarityMetric metric);

/**
 * @brief 查找概念的所有父类（is-a关系）
 * 
 * @param network 语义网络句柄
 * @param concept 概念
 * @param options 查询选项
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回匹配的父类数
 */
size_t semantic_network_find_parents(SemanticNetwork* network, Concept* concept,
                                    const HierarchyQueryOptions* options,
                                    Concept** results, size_t max_results);

/**
 * @brief 查找概念的所有子类（is-a关系）
 * 
 * @param network 语义网络句柄
 * @param concept 概念
 * @param options 查询选项
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回匹配的子类数
 */
size_t semantic_network_find_children(SemanticNetwork* network, Concept* concept,
                                     const HierarchyQueryOptions* options,
                                     Concept** results, size_t max_results);

/**
 * @brief 查找概念的所有实例（instance-of关系）
 * 
 * @param network 语义网络句柄
 * @param concept 概念
 * @param options 查询选项
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回匹配的实例数
 */
size_t semantic_network_find_instances(SemanticNetwork* network, Concept* concept,
                                      const HierarchyQueryOptions* options,
                                      Concept** results, size_t max_results);

/**
 * @brief 查找概念的最近公共祖先（LCA）
 * 
 * @param network 语义网络句柄
 * @param concept1 概念1
 * @param concept2 概念2
 * @return Concept* 最近公共祖先概念，失败返回NULL
 */
Concept* semantic_network_find_lowest_common_ancestor(SemanticNetwork* network,
                                                     Concept* concept1,
                                                     Concept* concept2);

/**
 * @brief 计算概念的层次深度
 * 
 * @param network 语义网络句柄
 * @param concept 概念
 * @return int 层次深度，失败返回-1
 */
int semantic_network_concept_depth(SemanticNetwork* network, Concept* concept);

/**
 * @brief 计算概念的信息内容（基于层次结构）
 * 
 * @param network 语义网络句柄
 * @param concept 概念
 * @return float 信息内容值
 */
float semantic_network_concept_information_content(SemanticNetwork* network,
                                                  Concept* concept);

/**
 * @brief 执行语义推理
 * 
 * @param network 语义网络句柄
 * @param premises 前提概念数组
 * @param premise_count 前提数量
 * @param max_inferences 最大推理数量
 * @param results 推理结果输出缓冲区
 * @param max_results 最大结果数
 * @return size_t 返回推理出的概念数
 */
size_t semantic_network_infer(SemanticNetwork* network,
                             Concept** premises, size_t premise_count,
                             size_t max_inferences,
                             Concept** results, size_t max_results);

/**
 * @brief 获取语义网络统计信息
 * 
 * @param network 语义网络句柄
 * @param stats 统计信息输出
 * @return int 成功返回0，失败返回-1
 */
int semantic_network_get_stats(SemanticNetwork* network, SemanticNetworkStats* stats);

/**
 * @brief 从知识库导入语义网络
 * 
 * @param network 语义网络句柄
 * @param kb 知识库句柄
 * @return int 成功返回导入的条目数，失败返回-1
 */
int semantic_network_import_from_knowledge_base(SemanticNetwork* network,
                                               KnowledgeBase* kb);

/**
 * @brief 从知识图谱导入语义网络
 * 
 * @param network 语义网络句柄
 * @param graph 知识图谱句柄
 * @return int 成功返回导入的条目数，失败返回-1
 */
int semantic_network_import_from_knowledge_graph(SemanticNetwork* network,
                                                KnowledgeGraph* graph);

/**
 * @brief 保存语义网络到文件
 * 
 * @param network 语义网络句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int semantic_network_save(SemanticNetwork* network, const char* filename);

/**
 * @brief 从文件加载语义网络
 * 
 * @param filename 文件名
 * @return SemanticNetwork* 语义网络句柄，失败返回NULL
 */
SemanticNetwork* semantic_network_load(const char* filename);

/**
 * @brief 语义网络可视化生成（文本格式）
 * 
 * @param network 语义网络句柄
 * @param root_concept 根概念（NULL表示从所有根开始）
 * @param max_depth 最大显示深度
 * @return char* 层次结构字符串，调用者负责释放
 */
char* semantic_network_visualize_hierarchy(SemanticNetwork* network,
                                          Concept* root_concept, int max_depth);

/**
 * @brief 扩散激活结果条目
 */
typedef struct {
    Concept* concept;               /**< 被激活的概念 */
    float activation;               /**< 激活值 */
} ActivationEntry;

/**
 * @brief 扩散激活 - 通过语义网络传播激活能量
 * 
 * 模拟认知科学中的扩散激活机制：当一个概念被激活时，
 * 激活能量沿着关系传播到相关概念，能量逐级衰减。
 * 
 * @param network 语义网络句柄
 * @param seeds 种子概念数组（初始激活源）
 * @param seed_activations 种子激活值数组（与种子概念一一对应）
 * @param seed_count 种子数量
 * @param decay_factor 衰减因子（0-1，越小衰减越快）
 * @param threshold 激活阈值（低于此值的概念被忽略）
 * @param max_iterations 最大传播迭代次数
 * @param max_results 最大结果数
 * @param results 激活结果输出缓冲区
 * @return size_t 返回激活的概念数（包括种子）
 */
SELFLNN_API size_t semantic_network_spreading_activation(SemanticNetwork* network,
    Concept** seeds, const float* seed_activations, size_t seed_count,
    float decay_factor, float threshold, size_t max_iterations,
    ActivationEntry* results, size_t max_results);

/**
 * @brief 语义网络剪枝 - 移除低置信度的概念和关系
 * 
 * 移除置信度低于指定阈值的概念和关系，优化网络结构。
 * 
 * @param network 语义网络句柄
 * @param min_concept_confidence 最小概念置信度阈值
 * @param min_relation_confidence 最小关系置信度阈值
 * @return size_t 返回移除的概念和关系总数
 */
SELFLNN_API size_t semantic_network_prune(SemanticNetwork* network,
    float min_concept_confidence, float min_relation_confidence);

/**
 * @brief 概念聚类 - 基于语义相似度进行概念分组
 * 
 * 使用层次聚类算法将相似概念分组，返回聚类结果。
 * 
 * @param network 语义网络句柄
 * @param similarity_threshold 相似度阈值（0-1）
 * @param max_clusters 最大聚类数
 * @param cluster_assignments 聚类分配输出数组（大小等于概念数）
 * @return size_t 返回实际聚类数
 */
SELFLNN_API size_t semantic_network_cluster_concepts(SemanticNetwork* network,
    float similarity_threshold, size_t max_clusters, int* cluster_assignments);

/**
 * @brief 关系推理 - 基于关系传递性推断新关系
 * 
 * 例如：如果A是B的父类，B是C的父类，则推断A是C的父类。
 * 支持多种关系类型的传递性推理。
 * 
 * @param network 语义网络句柄
 * @param relation_type 关系类型（-1表示所有类型）
 * @param max_inferences 最大推理数量
 * @return size_t 返回推断出的新关系数
 */
SELFLNN_API size_t semantic_network_infer_relations(SemanticNetwork* network,
    int relation_type, size_t max_inferences);

/**
 * @brief 语义网络合并 - 合并两个语义网络
 * 
 * 将源网络的概念和关系合并到目标网络中，处理重复概念。
 * 
 * @param dest 目标语义网络句柄
 * @param src 源语义网络句柄
 * @param merge_strategy 合并策略（0=覆盖，1=保留，2=平均）
 * @return size_t 返回合并的概念和关系总数
 */
SELFLNN_API size_t semantic_network_merge(SemanticNetwork* dest,
    SemanticNetwork* src, int merge_strategy);

/**
 * @brief 概念重要性排名 - 基于中心性度量
 * 
 * 计算概念的重要性分数（基于度中心性、接近中心性等）。
 * 
 * @param network 语义网络句柄
 * @param centrality_metric 中心性度量方法（0=度中心性，1=接近中心性）
 * @param importance_scores 重要性分数输出数组（大小等于概念数）
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int semantic_network_compute_concept_importance(SemanticNetwork* network,
    int centrality_metric, float* importance_scores);

/**
 * @brief 概念学习 - 基于共现模式学习新关系
 * 
 * 使用Hebbian学习机制：如果两个概念经常同时出现，它们之间的关系会增强。
 * 对于不存在的概念对，创建新的RELATION_RELATED_TO关系。
 * 
 * @param network 语义网络句柄
 * @param pattern_pairs 共现概念对数组（每对两个概念指针）
 * @param pair_count 共现对数量
 * @param learning_rate 学习率（0-1）
 * @return size_t 返回创建或增强的关系数
 */
SELFLNN_API size_t semantic_network_learn_from_patterns(SemanticNetwork* network,
    Concept** pattern_pairs, size_t pair_count, float learning_rate);

/**
 * @brief 获取语义网络中的概念数量
 *
 * @param network 语义网络句柄
 * @return size_t 概念数量
 */
SELFLNN_API size_t semantic_network_get_concept_count(SemanticNetwork* network);

/**
 * @brief 按索引获取语义网络中的概念
 *
 * @param network 语义网络句柄
 * @param index 概念索引（0到concept_count-1）
 * @return Concept* 概念指针，索引无效返回NULL
 */
SELFLNN_API Concept* semantic_network_get_concept_by_index(SemanticNetwork* network, size_t index);

/**
 * @brief 获取语义网络中的关系数量
 *
 * @param network 语义网络句柄
 * @return size_t 关系数量
 */
SELFLNN_API size_t semantic_network_get_relation_count(SemanticNetwork* network);

/**
 * @brief 按索引获取语义网络中的关系
 *
 * @param network 语义网络句柄
 * @param index 关系索引（0到relation_count-1）
 * @return SemanticRelation* 关系指针，索引无效返回NULL
 */
SELFLNN_API SemanticRelation* semantic_network_get_relation_by_index(SemanticNetwork* network, size_t index);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_SEMANTIC_NETWORK_H