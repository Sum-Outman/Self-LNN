/**
 * @file graph_reasoning.h
 * @brief 知识图谱推理引擎（CfC嵌入 + 液态推理）API（A03.1.3）
 *
 * 本引擎将CfC（Closed-form Continuous-time）液态神经网络嵌入到知识图谱
 * 推理的全流程中，提供三种推理模式：
 * 1. CfC ODE连续空间嵌入 — 实体/关系作为ODE状态向量，关系作为连续变换
 * 2. CfC复数域旋转嵌入 — 四元数旋转建模对称/反对称/组合关系
 * 3. CfC图液态推理 — CfC ODE在知识图谱上信息传播，替代GNN
 *
 * 集成 graph_storage 引擎（PropertyGraph / AdjacencyList / RDFTripleStore）
 * 和 cfc_knowledge_embedding 引擎，提供统一推理接口。
 *
 * 100% 纯C实现，无第三方库依赖。
 */

#ifndef SELFLNN_GRAPH_REASONING_H
#define SELFLNN_GRAPH_REASONING_H

#include <stddef.h>
#include "selflnn/knowledge/graph_storage.h"
#include "selflnn/knowledge/cfc_knowledge_embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============ 常量定义 ============ */

#define GR_MAX_PATHS            64
#define GR_MAX_PATH_LENGTH      32
#define GR_MAX_RULES            256
#define GR_MAX_CONSISTENCY_ISSUES 1024
#define GR_MAX_CANDIDATES       4096
#define GR_MAX_ENTITIES         262144
#define GR_MAX_RELATIONS        65536
#define GR_MAX_CHAIN_LENGTH     64      /**< 推理链最大长度 */
#define GR_MAX_CHAINS           128     /**< 最大推理链数 */

/* ============ 推理配置和结果类型 ============ */

/** @brief 推理模式 */
typedef enum {
    GR_REASON_CONTINUOUS = 0,      /**< CfC ODE连续空间嵌入推理 */
    GR_REASON_QUATERNION,          /**< CfC复数域旋转嵌入推理 */
    GR_REASON_LIQUID_GRAPH,        /**< CfC图液态推理（基于cfc_cell） */
    GR_REASON_HYBRID               /**< 混合推理（连续+四元数+液态综合） */
} GraphReasonMode;

/** @brief 路径排序模式 */
typedef enum {
    GR_RANK_TOTAL_SCORE = 0,       /**< 按路径总评分排序（越低越好） */
    GR_RANK_SHORTEST,              /**< 按路径长度排序（越短越好） */
    GR_RANK_HIGHEST_CONFIDENCE,    /**< 按路径最高置信度排序 */
    GR_RANK_LIQUID_ENERGY          /**< 按液态能量排序（越低越稳定） */
} GraphReasonPathRankMode;

/** @brief 推理引擎配置 */
typedef struct {
    GraphReasonMode reason_mode;   /**< 推理模式 */
    int embedding_dim;             /**< 嵌入维度（默认128） */
    float learning_rate;           /**< 学习率（默认0.001） */
    float margin;                  /**< 评分边界（默认1.0） */
    int num_negative_samples;      /**< 负采样数（默认10） */
    int batch_size;                /**< 批大小（默认128） */
    int max_epochs;                /**< 最大训练轮数（默认200） */
    float cfc_tau;                 /**< CfC时间常数（默认2.0） */
    float cfc_dt;                  /**< CfC时间步长（默认0.1） */
    int cfc_steps;                 /**< CfC ODE积分步数（默认10） */
    int graph_prop_steps;          /**< 图传播步数（默认3） */
    float graph_prop_dropout;      /**< 图传播dropout（默认0.1） */
    int top_k_predictions;         /**< 链路预测返回top-K（默认10） */
    float consistency_threshold;   /**< 一致性检查阈值（默认0.5） */
    float rule_min_confidence;     /**< 规则挖掘最小置信度（默认0.3） */
    float rule_min_support;        /**< 规则挖掘最小支持度（默认2） */
    int enable_self_diagnosis;     /**< 启用自我诊断（默认1） */
    int enable_auto_repair;        /**< 启用自动修复（默认0） */
} GraphReasonConfig;

/** @brief 推理三元组 */
typedef struct {
    int head_id;                   /**< 头实体ID */
    int rel_id;                    /**< 关系ID */
    int tail_id;                   /**< 尾实体ID */
    float score;                   /**< 推理评分（越小越合理） */
    float confidence;              /**< 置信度[0,1] */
} ReasonTriple;

/** @brief 链路预测结果 */
typedef struct {
    int entity_id;                 /**< 候选实体ID */
    float score;                   /**< 评分 */
    float confidence;              /**< 置信度 */
} LinkPrediction;

/** @brief 推理路径节点 */
typedef struct {
    int entity_id;                 /**< 实体ID */
    int relation_id;               /**< 关系ID（-1表示起点） */
    float weight;                  /**< 边权重（起点为1.0） */
    float state_energy;            /**< CfC状态能量 */
} PathNode;

/** @brief 推理路径 */
typedef struct {
    PathNode nodes[GR_MAX_PATH_LENGTH]; /**< 路径节点序列 */
    int length;                    /**< 路径长度 */
    float total_score;             /**< 路径总评分 */
    float liquid_energy;           /**< 液态能量（图传播累积） */
} ReasonPath;

/** @brief 路径查询结果集 */
typedef struct {
    ReasonPath paths[GR_MAX_PATHS]; /**< 路径数组 */
    int count;                     /**< 路径数 */
} PathResultSet;

/** @brief 挖掘的规则 */
typedef struct {
    int body_relations[GR_MAX_PATH_LENGTH]; /**< 规则体关系序列 */
    int body_length;                /**< 体长度 */
    int head_relation;              /**< 规则头关系 */
    float confidence;               /**< 置信度 */
    float support;                  /**< 支持度 */
    float head_coverage;            /**< 头部覆盖度 */
} MinedRule;

/** @brief 规则挖掘结果 */
typedef struct {
    MinedRule rules[GR_MAX_RULES]; /**< 规则数组 */
    int count;                     /**< 规则数 */
} RuleSet;

/** @brief 推理链节点（多跳推理过程中经过的中间实体） */
typedef struct {
    int entity_id;                 /**< 实体ID */
    int relation_id;               /**< 到达该实体的关系ID（起点为-1） */
    float arrival_score;           /**< 到达该实体的评分 */
    float liquid_energy;           /**< 到达时的液态能量 */
    int hop_depth;                 /**< 跳数深度 */
} ReasoningChainNode;

/** @brief 推理链（完整的多跳推理路径记录） */
typedef struct {
    ReasoningChainNode nodes[GR_MAX_CHAIN_LENGTH]; /**< 链节点序列 */
    int length;                    /**< 链长度 */
    float total_score;             /**< 链总评分 */
    float avg_energy;              /**< 链平均液态能量 */
} ReasoningChain;

/** @brief 推理链集合 */
typedef struct {
    ReasoningChain chains[GR_MAX_CHAINS]; /**< 链数组 */
    int count;                     /**< 链数 */
} ReasoningChainSet;

/** @brief 一致性问题类型 */
typedef enum {
    GR_ISSUE_CYCLE = 0,            /**< 循环矛盾 */
    GR_ISSUE_CONFLICT,             /**< 冲突三元组 */
    GR_ISSUE_UNUSUAL_ENERGY,       /**< 异常状态能量 */
    GR_ISSUE_LOW_CONFIDENCE,       /**< 低置信度三元组 */
    GR_ISSUE_MISSING_RELATION,     /**< 缺失关系 */
    GR_ISSUE_ISOLATED_NODE,        /**< 孤立节点 */
    GR_ISSUE_INCONSISTENT_TYPE     /**< 类型不一致 */
} ConsistencyIssueType;

/** @brief 一致性问题 */
typedef struct {
    ConsistencyIssueType issue_type; /**< 问题类型 */
    int entity_id;                 /**< 相关实体ID */
    int relation_id;               /**< 相关关系ID（-1表示无） */
    int related_entity_id;         /**< 相关实体ID */
    char description[256];         /**< 问题描述 */
    float severity;                /**< 严重程度[0,1] */
    float repair_score;            /**< 修复评分 */
} ConsistencyIssue;

/** @brief 一致性检查结果 */
typedef struct {
    ConsistencyIssue issues[GR_MAX_CONSISTENCY_ISSUES]; /**< 问题数组 */
    int count;                     /**< 问题数 */
    float overall_health;          /**< 整体健康度[0,1] */
} ConsistencyReport;

/* ============ 推理引擎句柄（不透明指针） ============ */

typedef struct GraphReasoner GraphReasoner;

/* ============ 推理引擎生命周期 ============ */

/**
 * @brief 创建推理引擎
 *
 * @param config 推理引擎配置（NULL时使用默认值）
 * @param property_graph 属性图引擎句柄（可为NULL，系统会内部创建）
 * @param adjacency_list 邻接表引擎句柄（可为NULL，系统会内部创建）
 * @param rdf_store RDF存储引擎句柄（可为NULL）
 * @return GraphReasoner* 引擎句柄，失败返回NULL
 */
GraphReasoner* graph_reasoner_create(const GraphReasonConfig* config,
                                     PropertyGraph* property_graph,
                                     AdjacencyList* adjacency_list,
                                     RDFTripleStore* rdf_store);

/**
 * @brief 销毁推理引擎
 *
 * @param reasoner 引擎句柄
 */
void graph_reasoner_destroy(GraphReasoner* reasoner);

/**
 * @brief 获取引擎配置
 *
 * @param reasoner 引擎句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int graph_reasoner_get_config(const GraphReasoner* reasoner, GraphReasonConfig* config);

/**
 * @brief 设置引擎配置（训练前调用）
 *
 * @param reasoner 引擎句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int graph_reasoner_set_config(GraphReasoner* reasoner, const GraphReasonConfig* config);

/* ============ 嵌入训练接口 ============ */

/**
 * @brief 从图存储引擎加载数据进行训练
 *
 * 自动从PropertyGraph/AdjacencyList/RDFTripleStore提取实体、关系和三元组，
 * 构建CfC嵌入训练集，执行嵌入训练。
 *
 * @param reasoner 引擎句柄
 * @param epochs 训练轮数（0表示使用配置中的max_epochs）
 * @return int 成功返回0，失败返回-1
 */
int graph_reasoner_train(GraphReasoner* reasoner, int epochs);

/**
 * @brief 从外部三元组数据训练
 *
 * @param reasoner 引擎句柄
 * @param triples 三元组数组
 * @param triple_count 三元组数
 * @param entity_names 实体名字典（name->id映射）
 * @param entity_count 实体数
 * @param relation_names 关系名字典
 * @param relation_count 关系数
 * @param epochs 训练轮数
 * @return int 成功返回0，失败返回-1
 */
int graph_reasoner_train_from_triples(GraphReasoner* reasoner,
                                       const ReasonTriple* triples, int triple_count,
                                       const char** entity_names, int entity_count,
                                       const char** relation_names, int relation_count,
                                       int epochs);

/**
 * @brief 获取当前的训练损失历史
 *
 * @param reasoner 引擎句柄
 * @param losses 损失值输出缓冲区
 * @param max_count 最大返回数
 * @return int 实际返回的损失值数量
 */
int graph_reasoner_get_loss_history(const GraphReasoner* reasoner,
                                     float* losses, int max_count);

/* ============ 链路预测接口 ============ */

/**
 * @brief 预测尾实体（给定头和关系）
 *
 * @param reasoner 引擎句柄
 * @param head_id 头实体ID
 * @param rel_id 关系ID
 * @param candidates 候选实体数组（NULL表示所有实体）
 * @param num_candidates 候选数（0表示所有实体）
 * @param results 预测结果输出缓冲区
 * @param max_results 最大返回数
 * @return int 实际返回的预测数，失败返回-1
 */
int graph_reasoner_predict_tail(GraphReasoner* reasoner,
                                 int head_id, int rel_id,
                                 const int* candidates, int num_candidates,
                                 LinkPrediction* results, int max_results);

/**
 * @brief 预测头实体（给定关系和尾）
 *
 * @param reasoner 引擎句柄
 * @param rel_id 关系ID
 * @param tail_id 尾实体ID
 * @param candidates 候选实体数组（NULL表示所有实体）
 * @param num_candidates 候选数
 * @param results 预测结果输出缓冲区
 * @param max_results 最大返回数
 * @return int 实际返回的预测数
 */
int graph_reasoner_predict_head(GraphReasoner* reasoner,
                                 int rel_id, int tail_id,
                                 const int* candidates, int num_candidates,
                                 LinkPrediction* results, int max_results);

/**
 * @brief 预测关系（给定头和尾）
 *
 * @param reasoner 引擎句柄
 * @param head_id 头实体ID
 * @param tail_id 尾实体ID
 * @param candidates 候选关系数组（NULL表示所有关系）
 * @param num_candidates 候选数
 * @param results 预测结果输出缓冲区
 * @param max_results 最大返回数
 * @return int 实际返回的预测数
 */
int graph_reasoner_predict_relation(GraphReasoner* reasoner,
                                     int head_id, int tail_id,
                                     const int* candidates, int num_candidates,
                                     LinkPrediction* results, int max_results);

/* ============ 三元组分类接口 ============ */

/**
 * @brief 分类三元组的有效性
 *
 * @param reasoner 引擎句柄
 * @param head_id 头实体ID
 * @param rel_id 关系ID
 * @param tail_id 尾实体ID
 * @param confidence 输出置信度[0,1]
 * @return int 有效返回1，无效返回0，失败返回-1
 */
int graph_reasoner_classify_triple(GraphReasoner* reasoner,
                                    int head_id, int rel_id, int tail_id,
                                    float* confidence);

/**
 * @brief 批量分类三元组
 *
 * @param reasoner 引擎句柄
 * @param triples 三元组数组
 * @param count 三元组数
 * @param results 结果数组（1=有效，0=无效）
 * @param confidences 置信度数组输出
 * @return int 成功返回0
 */
int graph_reasoner_classify_triples_batch(GraphReasoner* reasoner,
                                           const ReasonTriple* triples, int count,
                                           int* results, float* confidences);

/* ============ 路径推理接口 ============ */

/**
 * @brief 寻找两个实体间的推理路径
 *
 * 使用CfC液态能量指导路径搜索，找到最合理的推理路径。
 *
 * @param reasoner 引擎句柄
 * @param source_id 源实体ID
 * @param target_id 目标实体ID
 * @param max_depth 最大搜索深度
 * @param results 路径结果输出
 * @return int 实际找到的路径数，失败返回-1
 */
int graph_reasoner_find_paths(GraphReasoner* reasoner,
                               int source_id, int target_id,
                               int max_depth, PathResultSet* results);

/**
 * @brief 对推理路径进行排序
 *
 * 基于多特征（路径长度、液态能量、三元组评分、置信度）对路径集排序。
 *
 * @param reasoner 引擎句柄
 * @param path_set 路径结果集（输入输出，排序后覆盖原顺序）
 * @param mode 排序模式（参见GraphReasonPathRankMode）
 * @return int 排序后的路径数，失败返回-1
 */
int graph_reasoner_rank_paths(GraphReasoner* reasoner,
                               PathResultSet* path_set,
                               int mode);

/**
 * @brief 多跳推理（基于液态传播）
 *
 * 使用CfC液态传播进行多跳推理，返回可能相关的实体。
 *
 * @param reasoner 引擎句柄
 * @param seed_entity 起始实体
 * @param hops 跳数
 * @param results 结果实体数组
 * @param scores 结果评分数组
 * @param max_results 最大结果数
 * @return int 实际返回的结果数
 */
int graph_reasoner_multi_hop_reason(GraphReasoner* reasoner,
                                     int seed_entity, int hops,
                                     int* results, float* scores,
                                     int max_results);

/**
 * @brief 增强多跳推理（带推理链记录）
 *
 * 基于CfC液态能量引导的BFS传播，同时记录完整推理链（中间实体+关系+评分）。
 *
 * @param reasoner 引擎句柄
 * @param seed_entity 起始实体
 * @param hops 最大跳数
 * @param results 结果实体数组（按评分排序）
 * @param scores 结果评分数组
 * @param max_results 最大结果数
 * @param chains 推理链集合输出（记录每条推理路径，可为NULL）
 * @return int 实际返回的结果数，失败返回-1
 */
int graph_reasoner_multi_hop_enhanced(GraphReasoner* reasoner,
                                       int seed_entity, int hops,
                                       int* results, float* scores,
                                       int max_results,
                                       ReasoningChainSet* chains);

/* ============ LNN图推理接口 ============ */

/**
 * @brief LNN图嵌入推理（基于液态动力学）
 *
 * 使用CfC液态神经网络（LNN）动力学在整个知识图谱上进行信息传播，
 * 计算每个实体的稳定状态嵌入，用于推理隐含关系。
 *
 * @param reasoner 引擎句柄
 * @param seed_entity 起始推理实体ID（-1表示全图推理）
 * @param propagation_steps 液态传播步数
 * @param results 结果实体ID数组
 * @param scores 结果推理评分数组
 * @param max_results 最大结果数
 * @return int 推理到的实体数，失败返回-1
 */
int graph_reasoner_lnn_infer(GraphReasoner* reasoner,
                              int seed_entity,
                              int propagation_steps,
                              int* results, float* scores,
                              int max_results);

/* ============ 规则挖掘接口 ============ */

/**
 * @brief 挖掘知识图谱中的推理规则
 *
 * @param reasoner 引擎句柄
 * @param max_body_length 规则体最大长度
 * @param min_confidence 最小置信度
 * @param min_support 最小支持度
 * @param rules 规则结果输出
 * @return int 挖掘到的规则数
 */
int graph_reasoner_mine_rules(GraphReasoner* reasoner,
                               int max_body_length,
                               float min_confidence, float min_support,
                               RuleSet* rules);

/**
 * @brief 应用规则进行推理
 *
 * @param reasoner 引擎句柄
 * @param rule 要应用的规则
 * @param head_entity 推理起点实体
 * @param results 推理结果实体输出
 * @param scores 结果评分输出
 * @param max_results 最大结果数
 * @return int 推理到的实体数
 */
int graph_reasoner_apply_rule(GraphReasoner* reasoner,
                               const MinedRule* rule, int head_entity,
                               int* results, float* scores, int max_results);

/* ============ 一致性检查接口 ============ */

/**
 * @brief 执行知识图谱一致性检查
 *
 * 检测循环矛盾、冲突三元组、异常能量状态、孤立节点等。
 *
 * @param reasoner 引擎句柄
 * @param report 检查报告输出
 * @return int 发现问题数，失败返回-1
 */
int graph_reasoner_check_consistency(GraphReasoner* reasoner,
                                      ConsistencyReport* report);

/**
 * @brief 修复一致性问题
 *
 * @param reasoner 引擎句柄
 * @param issue 要修复的问题
 * @return int 修复成功返回0，无法修复返回1，失败返回-1
 */
int graph_reasoner_repair_issue(GraphReasoner* reasoner,
                                 const ConsistencyIssue* issue);

/* ============ 嵌入操作接口 ============ */

/**
 * @brief 获取实体嵌入向量
 *
 * @param reasoner 引擎句柄
 * @param entity_id 实体ID
 * @param embedding 输出缓冲区
 * @param max_dim 缓冲区最大维度
 * @return int 实际返回的维度数，失败返回-1
 */
int graph_reasoner_get_entity_embedding(const GraphReasoner* reasoner,
                                         int entity_id,
                                         float* embedding, int max_dim);

/**
 * @brief 获取关系嵌入向量
 *
 * @param reasoner 引擎句柄
 * @param relation_id 关系ID
 * @param embedding 输出缓冲区
 * @param max_dim 缓冲区最大维度
 * @return int 实际返回的维度数，失败返回-1
 */
int graph_reasoner_get_relation_embedding(const GraphReasoner* reasoner,
                                           int relation_id,
                                           float* embedding, int max_dim);

/**
 * @brief 比较两个实体的语义相似度
 *
 * @param reasoner 引擎句柄
 * @param entity_a 实体A
 * @param entity_b 实体B
 * @return float 余弦相似度[-1,1]，失败返回-2
 */
float graph_reasoner_entity_similarity(const GraphReasoner* reasoner,
                                        int entity_a, int entity_b);

/**
 * @brief 获取实体或关系的CfC液态能量
 *
 * 液态能量反映实体/关系在连续动态系统中的稳定性。
 * 低能量表示高度稳定，高能量表示不稳定或异常。
 *
 * @param reasoner 引擎句柄
 * @param entity_id 实体ID（-1表示获取关系能量）
 * @param relation_id 关系ID（仅entity_id=-1时有效）
 * @return float 液态能量值，失败返回-1
 */
float graph_reasoner_get_liquid_energy(const GraphReasoner* reasoner,
                                        int entity_id, int relation_id);

/* ============ 查询集成接口 ============ */

/**
 * @brief 执行推理增强的图查询
 *
 * 将推理结果注入图查询结果，增强查询的召回率。
 *
 * @param reasoner 引擎句柄
 * @param query_result 图查询结果集（支持查询结果中的推理增强）
 * @param max_results 最大结果数
 * @return int 增强后的结果数
 */
int graph_reasoner_enhance_query(GraphReasoner* reasoner,
                                  void* query_result, int max_results);

/**
 * @brief 获取实体名称
 *
 * @param reasoner 引擎句柄
 * @param entity_id 实体ID
 * @return const char* 实体名称，失败返回NULL
 */
const char* graph_reasoner_get_entity_name(const GraphReasoner* reasoner,
                                            int entity_id);

/**
 * @brief 获取关系名称
 *
 * @param reasoner 引擎句柄
 * @param relation_id 关系ID
 * @return const char* 关系名称，失败返回NULL
 */
const char* graph_reasoner_get_relation_name(const GraphReasoner* reasoner,
                                              int relation_id);

/**
 * @brief 获取实体数
 *
 * @param reasoner 引擎句柄
 * @return int 实体数
 */
int graph_reasoner_entity_count(const GraphReasoner* reasoner);

/**
 * @brief 获取关系数
 *
 * @param reasoner 引擎句柄
 * @return int 关系数
 */
int graph_reasoner_relation_count(const GraphReasoner* reasoner);

/* ============ 保存/加载接口 ============ */

/**
 * @brief 保存推理引擎状态
 *
 * @param reasoner 引擎句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int graph_reasoner_save(const GraphReasoner* reasoner, const char* filepath);

/**
 * @brief 加载推理引擎状态
 *
 * @param filepath 文件路径
 * @param property_graph 属性图引擎（可为NULL）
 * @param adjacency_list 邻接表引擎（可为NULL）
 * @param rdf_store RDF存储引擎（可为NULL）
 * @return GraphReasoner* 引擎句柄，失败返回NULL
 */
GraphReasoner* graph_reasoner_load(const char* filepath,
                                    PropertyGraph* property_graph,
                                    AdjacencyList* adjacency_list,
                                    RDFTripleStore* rdf_store);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GRAPH_REASONING_H */
