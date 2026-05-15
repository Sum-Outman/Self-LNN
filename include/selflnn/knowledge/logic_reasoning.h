/**
 * @file logic_reasoning.h
 * @brief 逻辑推理引擎接口
 * 
 * 基于规则的逻辑推理引擎，支持前向链、后向链、冲突解决、不确定性推理等。
 * 提供与知识库、知识图谱和语义网络的集成。
 */

#ifndef SELFLNN_LOGIC_REASONING_H
#define SELFLNN_LOGIC_REASONING_H

#include "selflnn/knowledge/knowledge.h"
#include "selflnn/knowledge/knowledge_graph.h"
#include "selflnn/knowledge/semantic_network.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 推理模式
 */
typedef enum {
    REASONING_FORWARD_CHAINING = 0,   /**< 前向链推理 */
    REASONING_BACKWARD_CHAINING = 1,  /**< 后向链推理 */
    REASONING_MIXED_CHAINING = 2,     /**< 混合链推理 */
    REASONING_MONOTONIC = 3,          /**< 单调推理 */
    REASONING_NONMONOTONIC = 4        /**< 非单调推理 */
} ReasoningMode;

/**
 * @brief 规则结构
 */
typedef struct {
    int id;                           /**< 规则ID */
    char* name;                       /**< 规则名称 */
    char** premises;                  /**< 前提条件数组 */
    size_t premise_count;             /**< 前提条件数量 */
    char** conclusions;               /**< 结论数组 */
    size_t conclusion_count;          /**< 结论数量 */
    char* conclusion;                 /**< 主要结论（快捷方式，通常是conclusions[0]） */
    float confidence;                 /**< 规则置信度（0-1） */
    int priority;                     /**< 规则优先级 */
    void* user_data;                  /**< 用户数据指针 */
} InferenceRule;

/**
 * @brief 默认推理规则结构体（前向声明，完整定义在.c中）
 */
typedef struct DefaultRuleSpec DefaultRuleSpec;
typedef struct PslFormulaSpec PslFormulaSpec;

/**
 * @brief 推理引擎配置
 */
typedef struct {
    ReasoningMode mode;               /**< 推理模式 */
    int max_inference_steps;          /**< 最大推理步数 */
    float min_confidence;             /**< 最小置信度阈值 */
    int enable_conflict_resolution;   /**< 是否启用冲突解决 */
    int enable_uncertainty_reasoning; /**< 是否启用不确定性推理 */
    size_t working_memory_size;       /**< 工作内存大小 */
} ReasoningEngineConfig;

/**
 * @brief 推理引擎句柄
 */
typedef struct ReasoningEngine ReasoningEngine;

/**
 * @brief 逻辑推理结果
 */
typedef struct {
    char** inferred_facts;            /**< 推理出的事实数组 */
    size_t fact_count;                /**< 事实数量 */
    char** proved_facts;              /**< 已证明的事实数组 */
    size_t proved_fact_count;         /**< 已证明的事实数量 */
    InferenceRule** applied_rules;    /**< 应用的规则数组 */
    size_t rule_count;                /**< 规则数量 */
    float overall_confidence;         /**< 总体置信度 */
    int inference_steps;              /**< 推理步数 */
    int inference_time_ms;            /**< 推理时间（毫秒） */
    char* reasoning_trace;            /**< 推理追踪（可选） */
} LogicInferenceResult;

/**
 * @brief 冲突解决策略
 */
typedef enum {
    CONFLICT_RESOLUTION_PRIORITY = 0, /**< 优先级策略 */
    CONFLICT_RESOLUTION_RECENCY = 1,  /**< 最近使用策略 */
    CONFLICT_RESOLUTION_SPECIFICITY = 2, /**< 特异性策略 */
    CONFLICT_RESOLUTION_RANDOM = 3    /**< 随机策略 */
} ConflictResolutionStrategy;

/**
 * @brief 创建逻辑推理引擎
 * 
 * @param config 引擎配置
 * @return ReasoningEngine* 逻辑推理引擎句柄，失败返回NULL
 */
ReasoningEngine* logic_reasoning_engine_create(const ReasoningEngineConfig* config);

/**
 * @brief 释放逻辑推理引擎
 * 
 * @param engine 逻辑推理引擎句柄
 */
void logic_reasoning_engine_free(ReasoningEngine* engine);

/**
 * @brief 释放逻辑推理结果
 * 
 * @param result 逻辑推理结果指针
 */
void logic_reasoning_engine_free_inference_result(LogicInferenceResult* result);

/**
 * @brief 添加规则到逻辑推理引擎
 * 
 * @param engine 逻辑推理引擎句柄
 * @param rule 规则
 * @return int 成功返回规则ID，失败返回-1
 */
int logic_reasoning_engine_add_rule(ReasoningEngine* engine, const InferenceRule* rule);

/**
 * @brief 从知识库加载规则到逻辑推理引擎
 * 
 * @param engine 逻辑推理引擎句柄
 * @param kb 知识库句柄
 * @return int 成功返回加载的规则数，失败返回-1
 */
int logic_reasoning_engine_load_rules_from_kb(ReasoningEngine* engine, KnowledgeBase* kb);

/**
 * @brief 从知识图谱加载规则到逻辑推理引擎
 * 
 * @param engine 逻辑推理引擎句柄
 * @param graph 知识图谱句柄
 * @return int 成功返回加载的规则数，失败返回-1
 */
int logic_reasoning_engine_load_rules_from_graph(ReasoningEngine* engine, KnowledgeGraph* graph);

/**
 * @brief 从语义网络加载规则到逻辑推理引擎
 * 
 * @param engine 逻辑推理引擎句柄
 * @param network 语义网络句柄
 * @return int 成功返回加载的规则数，失败返回-1
 */
int logic_reasoning_engine_load_rules_from_semantic_network(ReasoningEngine* engine,
                                                     SemanticNetwork* network);

/**
 * @brief 设置逻辑推理引擎冲突解决策略
 * 
 * @param engine 逻辑推理引擎句柄
 * @param strategy 冲突解决策略
 * @return int 成功返回0，失败返回-1
 */
int logic_reasoning_engine_set_conflict_resolution(ReasoningEngine* engine,
                                            ConflictResolutionStrategy strategy);

/**
 * @brief 执行逻辑推理引擎前向链推理
 * 
 * @param engine 逻辑推理引擎句柄
 * @param initial_facts 初始事实数组
 * @param fact_count 初始事实数量
 * @param goal_facts 目标事实数组（可选，用于指导推理）
 * @param goal_count 目标事实数量
 * @return InferenceResult* 推理结果，调用者负责释放，失败返回NULL
 */
LogicInferenceResult* logic_reasoning_engine_forward_chain(ReasoningEngine* engine,
                                               const char** initial_facts,
                                               size_t fact_count,
                                               const char** goal_facts,
                                               size_t goal_count);

/**
 * @brief 执行逻辑推理引擎后向链推理
 * 
 * @param engine 逻辑推理引擎句柄
 * @param goal_facts 目标事实数组
 * @param goal_count 目标事实数量
 * @param known_facts 已知事实数组（可选）
 * @param known_count 已知事实数量
 * @return InferenceResult* 推理结果，调用者负责释放，失败返回NULL
 */
LogicInferenceResult* logic_reasoning_engine_backward_chain(ReasoningEngine* engine,
                                                const char** goal_facts,
                                                size_t goal_count,
                                                const char** known_facts,
                                                size_t known_count);

/**
 * @brief 执行逻辑推理引擎混合链推理
 * 
 * @param engine 逻辑推理引擎句柄
 * @param facts 事实数组
 * @param fact_count 事实数量
 * @param goals 目标数组
 * @param goal_count 目标数量
 * @return InferenceResult* 推理结果，调用者负责释放，失败返回NULL
 */
LogicInferenceResult* logic_reasoning_engine_mixed_chain(ReasoningEngine* engine,
                                             const char** facts,
                                             size_t fact_count,
                                             const char** goals,
                                             size_t goal_count);

/**
 * @brief 基于知识库执行逻辑推理引擎推理
 * 
 * @param engine 逻辑推理引擎句柄
 * @param kb 知识库句柄
 * @param query 查询模式
 * @param max_results 最大结果数
 * @return InferenceResult* 推理结果，调用者负责释放，失败返回NULL
 */
LogicInferenceResult* logic_reasoning_engine_reason_with_kb(ReasoningEngine* engine,
                                                KnowledgeBase* kb,
                                                const char* query,
                                                size_t max_results);

/**
 * @brief 基于知识图谱执行逻辑推理引擎推理
 * 
 * @param engine 逻辑推理引擎句柄
 * @param graph 知识图谱句柄
 * @param start_node 起始节点
 * @param end_node 目标节点（可选）
 * @param max_paths 最大路径数
 * @return InferenceResult* 推理结果，调用者负责释放，失败返回NULL
 */
LogicInferenceResult* logic_reasoning_engine_reason_with_graph(ReasoningEngine* engine,
                                                   KnowledgeGraph* graph,
                                                   GraphNode* start_node,
                                                   GraphNode* end_node,
                                                   size_t max_paths);

/**
 * @brief 基于语义网络执行逻辑推理引擎推理
 * 
 * @param engine 逻辑推理引擎句柄
 * @param network 语义网络句柄
 * @param concept 起始概念
 * @param relation_type 关系类型
 * @param max_depth 最大深度
 * @return InferenceResult* 推理结果，调用者负责释放，失败返回NULL
 */
LogicInferenceResult* logic_reasoning_engine_reason_with_semantic_network(ReasoningEngine* engine,
                                                              SemanticNetwork* network,
                                                              Concept* concept,
                                                              int relation_type,
                                                              int max_depth);

/**
 * @brief 获取逻辑推理引擎统计信息
 * 
 * @param engine 逻辑推理引擎句柄
 * @param total_rules 总规则数输出
 * @param inference_count 总推理次数输出
 * @param avg_inference_time 平均推理时间输出（毫秒）
 * @return int 成功返回0，失败返回-1
 */
int logic_reasoning_engine_get_stats(ReasoningEngine* engine,
                              size_t* total_rules,
                              size_t* inference_count,
                              float* avg_inference_time);

/**
 * @brief 重置逻辑推理引擎
 * 
 * @param engine 逻辑推理引擎句柄
 */
void logic_reasoning_engine_reset(ReasoningEngine* engine);

/**
 * @brief 保存逻辑推理引擎规则到文件
 * 
 * @param engine 逻辑推理引擎句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int logic_reasoning_engine_save_rules(ReasoningEngine* engine, const char* filename);

/**
 * @brief 从文件加载逻辑推理引擎规则
 * 
 * @param engine 逻辑推理引擎句柄
 * @param filename 文件名
 * @return int 成功返回加载的规则数，失败返回-1
 */
int logic_reasoning_engine_load_rules(ReasoningEngine* engine, const char* filename);

/**
 * @brief 释放逻辑推理结果内存
 * 
 * @param result 逻辑推理结果
 */
void logic_inference_result_free(LogicInferenceResult* result);

/**
 * @brief 创建规则
 * 
 * @param name 规则名称
 * @param premises 前提条件数组
 * @param premise_count 前提条件数量
 * @param conclusions 结论数组
 * @param conclusion_count 结论数量
 * @param confidence 规则置信度
 * @param priority 规则优先级
 * @return InferenceRule* 规则指针，调用者负责释放，失败返回NULL
 */
InferenceRule* inference_rule_create(const char* name,
                                    const char** premises, size_t premise_count,
                                    const char** conclusions, size_t conclusion_count,
                                    float confidence, int priority);

/**
 * @brief 释放规则内存
 * 
 * @param rule 规则指针
 */
void inference_rule_free(InferenceRule* rule);

/* ============================================================================
 * F-09: 概率软逻辑(PSL) API
 * =========================================================================== */

/**
 * @brief 添加PSL概率软逻辑公式
 * 
 * PSL使用Lukasiewicz逻辑在[0,1]连续真值空间中进行推理。
 * 每个公式 w: premise_1 ∧ ... ∧ premise_n → conclusion 有权重w。
 * 通过梯度下降最小化加权平方距离和。
 * 
 * @param engine 推理引擎句柄
 * @param formula PSL公式结构体指针
 * @return int 成功返回公式ID，失败返回-1
 */
int logic_reasoning_engine_add_psl_formula(ReasoningEngine* engine, const PslFormulaSpec* formula);

/**
 * @brief 执行PSL不确定性推理
 * 
 * 将输入事实映射为连续真值变量，通过梯度下降优化真值，
 * 返回满足所有加权公式的最优真值分配。
 * 
 * @param engine 推理引擎句柄
 * @param facts 事实数组
 * @param fact_count 事实数量
 * @return LogicInferenceResult* 推理结果，调用者负责释放
 */
LogicInferenceResult* logic_reasoning_engine_reason_with_uncertainty(ReasoningEngine* engine,
                                                                     const char** facts, size_t fact_count);

/* ============================================================================
 * F-09: 默认推理 API
 * =========================================================================== */

/**
 * @brief 添加默认推理规则
 * 
 * 默认规则格式: 前提 → 结论, 除非 例外条件
 * 当"除非"条件在事实集中匹配时，规则被阻止。
 * 更具体的规则（前提更多）优先级更高。
 * 
 * @param engine 推理引擎句柄
 * @param rule 默认规则结构体指针
 * @return int 成功返回规则ID，失败返回-1
 */
int logic_reasoning_engine_add_default_rule(ReasoningEngine* engine, const DefaultRuleSpec* rule);

/**
 * @brief 执行默认推理
 * 
 * 按特异性降序排列规则，检查前提和"除非"条件，
 * 应用第一个满足条件的规则生成结论。
 * 
 * @param engine 推理引擎句柄
 * @param facts 事实数组
 * @param fact_count 事实数量
 * @return LogicInferenceResult* 推理结果，调用者负责释放
 */
LogicInferenceResult* logic_reasoning_engine_default_reason(ReasoningEngine* engine,
                                                            const char** facts, size_t fact_count);

/* ============================================================================
 * F-09: 非单调推理 API
 * =========================================================================== */

/**
 * @brief 添加非单调事实
 * 
 * 非单调事实可以被后续推理撤销，单调事实不可撤销。
 * 
 * @param engine 推理引擎句柄
 * @param fact 事实字符串
 * @param confidence 置信度(0-1)
 * @param is_monotonic 是否为单调事实（1=不可撤销，0=可撤销）
 * @return int 成功返回事实ID，失败返回-1
 */
int logic_reasoning_engine_nonmonotonic_add_fact(ReasoningEngine* engine, const char* fact,
                                                  float confidence, int is_monotonic);

/**
 * @brief 移除非单调事实
 * 
 * @param engine 推理引擎句柄
 * @param fact 事实字符串
 * @return int 成功返回0，失败返回-1
 */
int logic_reasoning_engine_nonmonotonic_remove_fact(ReasoningEngine* engine, const char* fact);

/**
 * @brief 执行非单调推理
 * 
 * 1. 添加初始事实到非单调状态
 * 2. 使用引擎规则前向推理
 * 3. 每步检测矛盾（如 "X is A" vs "X is not A"）
 * 4. 发现矛盾时撤销权重较低的事实
 * 5. 返回最终非矛盾事实集
 * 
 * @param engine 推理引擎句柄
 * @param facts 初始事实数组
 * @param fact_count 事实数量
 * @return LogicInferenceResult* 推理结果，调用者负责释放
 */
LogicInferenceResult* logic_reasoning_engine_nonmonotonic_reason(ReasoningEngine* engine,
                                                                  const char** facts, size_t fact_count);

/* K-修复: 归纳推理——从具体实例泛化出一般规则 */
ReasoningEngine* logic_reasoning_induction(ReasoningEngine* engine,
    const char** instances, size_t instance_count, int max_rules);

/* K-修复: 溯因推理——从结果推导最可能原因 */
char* logic_reasoning_abduction(ReasoningEngine* engine,
    const char* observation, float* confidence_out);

/* K-修复: 类比推理——A:B :: C:D 结构映射 */
char* logic_reasoning_analogy(ReasoningEngine* engine,
    const char* source_a, const char* source_b,
    const char* target_c, float* confidence_out);

/* K-修复: CDCL SAT求解器（冲突驱动子句学习） */
int logic_reasoning_cdcl_solve(int num_vars,
    const int** clauses, const int* clause_sizes, int num_clauses,
    int* solution);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_LOGIC_REASONING_H