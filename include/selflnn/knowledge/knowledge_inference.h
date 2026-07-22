/**
 * @file knowledge_inference.h
 * @brief 知识推理增强系统接口（传递/规则/类比/时序/反事实/图推理/归纳/贝叶斯/冲突消解）
 */

#ifndef SELFLNN_KNOWLEDGE_INFERENCE_H
#define SELFLNN_KNOWLEDGE_INFERENCE_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KI_MAX_PREMISES 16
#define KI_MAX_CHAIN 32
#define KI_MAX_ANALOGS 8
#define KI_MAX_PATH_LENGTH 64
#define KI_MAX_PATHS 32
#define KI_MAX_PATTERN_ELEMENTS 16
#define KI_MAX_PATTERNS 32
#define KI_MAX_CONCEPTS 64
#define KI_MAX_EXAMPLES 32
#define KI_MAX_CONFLICTS 64
#define KI_MAX_COMPONENTS 128
#define KI_MAX_GRAPH_NODES 256
#define KI_MAX_RELATED_FACTS 512

typedef struct {
    char* subject;
    char* predicate;
    char* object;
    float confidence;
    int source_id;
} KIFact;

typedef struct {
    int rule_id;
    char condition[256];
    char conclusion[256];
    float confidence;
    float priority;
    int usage_count;
} KIRule;

typedef struct {
    int chain_id;
    KIFact steps[KI_MAX_CHAIN];
    int step_count;
    float total_confidence;
    int is_valid;
} KIInferenceChain;

typedef struct {
    int analog_id;
    char source_domain[64];
    char target_domain[64];
    float similarity;
    KIFact source_facts[KI_MAX_PREMISES];
    int source_count;
    KIFact mapped_facts[KI_MAX_PREMISES];
    int mapped_count;
} KIAnalogy;

typedef struct {
    time_t from_time;
    time_t to_time;
    KIFact timeline[KI_MAX_CHAIN];
    int event_count;
    float temporal_consistency;
} KITimeline;

typedef struct {
    int counterfactual_id;
    char hypothesis[256];
    KIFact assumed_facts[KI_MAX_PREMISES];
    int assumed_count;
    KIFact consequences[KI_MAX_CHAIN];
    int consequence_count;
    float plausibility;
} KICounterfactual;

typedef struct {
    KIFact path_facts[KI_MAX_PATH_LENGTH];
    int path_length;
    float total_confidence;
    float path_score;
} KIPath;

typedef struct {
    int concept_id;
    char concept_name[256];
    KIFact examples[KI_MAX_EXAMPLES];
    int example_count;
    float abstraction_level;
    int parent_concept_id;
    int sub_concept_ids[16];
    int sub_concept_count;
} KIConcept;

typedef struct {
    char pattern_description[256];
    KIFact pattern_elements[KI_MAX_PATTERN_ELEMENTS];
    int element_count;
    int occurrence_count;
    float pattern_confidence;
} KIPattern;

typedef struct {
    char knowledge_component[256];
    float mastery_probability;
    float guess_probability;
    float slip_probability;
    float transition_learn;
    float transition_forget;
    int observation_count;
    int correct_count;
    time_t last_observed;
} KIBayesianComponent;

typedef struct {
    KIBayesianComponent components[KI_MAX_COMPONENTS];
    int component_count;
    float overall_mastery;
} KIBayesianState;

typedef struct {
    int conflict_id;
    int entry_id_a;
    int entry_id_b;
    KIFact fact_a;
    KIFact fact_b;
    char description[256];
    float severity;
} KIConflict;

typedef enum {
    KI_RESOLVE_CONFIDENCE = 0,
    KI_RESOLVE_TIME = 1,
    KI_RESOLVE_SOURCE = 2,
    KI_RESOLVE_MERGE = 3,
    KI_RESOLVE_PRIORITY_SPECIFICITY = 4
} KIResolutionStrategy;

typedef struct KnowledgeInferenceEngine KnowledgeInferenceEngine;

KnowledgeInferenceEngine* ki_engine_create(void);
void ki_engine_free(KnowledgeInferenceEngine* kie);

int ki_set_knowledge_base(KnowledgeInferenceEngine* kie, void* kb);

int ki_transitive_infer(KnowledgeInferenceEngine* kie, const KIFact* facts, int count, KIInferenceChain* chain);
int ki_chain_validate(const KIInferenceChain* chain, float* confidence);

int ki_add_rule(KnowledgeInferenceEngine* kie, const KIRule* rule);
int ki_forward_chain(KnowledgeInferenceEngine* kie, const KIFact* facts, int count, KIFact* inferred, int* inf_count);
int ki_backward_chain(KnowledgeInferenceEngine* kie, const KIFact* goal, KIFact* needed, int* need_count);

/* M-004修复: 可配置容量上限API */
int ki_set_max_rules(KnowledgeInferenceEngine* kie, int max_rules);
int ki_get_max_rules(KnowledgeInferenceEngine* kie);
int ki_set_max_iterations(KnowledgeInferenceEngine* kie, int max_iter);
int ki_get_max_iterations(KnowledgeInferenceEngine* kie);

int ki_find_analogies(KnowledgeInferenceEngine* kie, const char* source_domain, const char* target_domain, KIAnalogy* out, int max_count);
int ki_map_analogy(KnowledgeInferenceEngine* kie, KIAnalogy* analogy);

/**
 * @brief H-002修复: 完整类比推理流程（结构映射理论SMT）
 *
 * 整合ki_find_analogies和ki_map_analogy，提供一步到位的类比推理。
 * 基于结构映射理论，实现关系结构提取→结构对齐→系统性优先→候选推理。
 *
 * @param kie 推理引擎句柄
 * @param source_domain 源域名称（如"太阳系"）
 * @param target_domain 目标域名称（如"原子"）
 * @param results 类比结果输出数组
 * @param result_count 输出有效类比数量
 * @return int 成功返回0，失败返回-1
 */
int ki_analogical_reasoning(KnowledgeInferenceEngine* kie,
                             const char* source_domain, const char* target_domain,
                             KIAnalogy* results, int* result_count);

/**
 * @brief H-002修复: 溯因推理（最佳解释推理IBE）
 *
 * 从观察事实推导最可能的解释假设。
 * 实现观察收集→假设生成→假设评分（覆盖率+简洁性+一致性）→最佳假设选择。
 *
 * @param kie 推理引擎句柄
 * @param observations 观察事实数组
 * @param observation_count 观察事实数量
 * @param hypotheses 输出假设数组（调用者分配）
 * @param hypothesis_count 输出假设数量
 * @return int 成功返回0，失败返回-1
 */
int ki_abductive_reasoning(KnowledgeInferenceEngine* kie,
                            const KIFact* observations, int observation_count,
                            KIFact* hypotheses, int* hypothesis_count);

/**
 * @brief H-002深度增强: LNN语义嵌入类比推理
 *
 * 利用LNN的连续动态系统计算实体和关系的语义嵌入向量，
 * 用余弦相似度替代字符串编辑距离，实现更精准的类比映射。
 * 如果LNN不可用(NULL)，自动回退到字符串相似度。
 *
 * @param kie 推理引擎句柄
 * @param lnn LNN实例指针（NULL=回退到字符串相似度）
 * @param source_domain 源域
 * @param target_domain 目标域
 * @param results 类比结果
 * @param result_count 输出类比数量
 * @return int 成功返回0
 */
int ki_lnn_analogical_reasoning(KnowledgeInferenceEngine* kie, void* lnn,
                                 const char* source_domain, const char* target_domain,
                                 KIAnalogy* results, int* result_count);

/**
 * @brief H-002深度增强: 多跳因果链溯因推理
 *
 * 构建因果图后在图中搜索最佳解释路径。
 * 实现BFS多跳搜索 + 路径评分（覆盖率+简洁性+因果强度）。
 * 相对于单跳溯因，能发现更复杂的因果链解释。
 *
 * @param kie 推理引擎句柄
 * @param observations 观察事实
 * @param observation_count 观察数量
 * @param hypotheses 输出假设
 * @param hypothesis_count 输出假设数量
 * @param max_hops 最大搜索跳数（默认3，最大8）
 * @return int 成功返回0
 */
int ki_multi_hop_abductive_reasoning(KnowledgeInferenceEngine* kie,
                                      const KIFact* observations, int observation_count,
                                      KIFact* hypotheses, int* hypothesis_count,
                                      int max_hops);

/**
 * @brief H-002深度增强: 跨域类比知识迁移
 *
 * 在发现源域到目标域的类比后，自动将源域的知识迁移到目标域，
 * 生成新的知识三元组并注入知识库。
 * 实现"如果A域有X→Y关系，且A≈B，则B域可能有X'→Y'关系"的推理。
 *
 * @param kie 推理引擎句柄
 * @param source_domain 源域
 * @param target_domain 目标域
 * @param min_similarity 最小相似度阈值
 * @param transferred_facts 输出迁移的知识
 * @param transfer_count 输出迁移数量
 * @return int 成功返回0
 */
int ki_cross_domain_analogy_transfer(KnowledgeInferenceEngine* kie,
                                      const char* source_domain, const char* target_domain,
                                      float min_similarity,
                                      KIFact* transferred_facts, int* transfer_count);

/**
 * @brief H-002深度增强: 验证溯因假设并反馈到LNN
 *
 * 对每个假设检查是否与现有知识矛盾，计算验证得分，
 * 并将验证结果反馈到LNN状态扰动，完成"观察→假设→验证→学习"闭环。
 *
 * @param kie 推理引擎
 * @param lnn LNN实例（NULL=跳过LNN反馈）
 * @param hypotheses 假设数组
 * @param hypothesis_count 假设数量
 * @param verified_scores 输出验证得分数组
 * @return int 成功返回0
 */
int ki_verify_hypotheses(KnowledgeInferenceEngine* kie, void* lnn,
                          const KIFact* hypotheses, int hypothesis_count,
                          float* verified_scores);

int ki_build_timeline(KnowledgeInferenceEngine* kie, const KIFact* events, int count, KITimeline* timeline);
int ki_query_timeline(const KITimeline* timeline, time_t at_time, KIFact* facts, int max_count);
int ki_detect_temporal_patterns(KnowledgeInferenceEngine* kie, const KITimeline* timeline, char* pattern_desc, size_t max_len);

int ki_counterfactual_reason(KnowledgeInferenceEngine* kie, const char* hypothesis, const KIFact* facts, int count, KICounterfactual* cf);
int ki_evaluate_counterfactual(const KICounterfactual* cf, float* plausibility);

int ki_probabilistic_infer(KnowledgeInferenceEngine* kie, const KIFact* facts, const float* probs, int count, KIFact* result, float* result_prob);

int ki_graph_dfs(KnowledgeInferenceEngine* kie, const char* start_concept, int max_depth, KIFact* path, int* path_count);
int ki_graph_bfs(KnowledgeInferenceEngine* kie, const char* start_concept, int max_depth, KIFact* related, int* related_count);
int ki_path_ranking(KnowledgeInferenceEngine* kie, const char* from_concept, const char* to_concept, KIPath* paths, int* path_count, int max_paths);
int ki_subgraph_match(KnowledgeInferenceEngine* kie, const KIFact* pattern, int pattern_count, KIFact* matches, int* match_count);

int ki_inductive_generalize(KnowledgeInferenceEngine* kie, const KIFact* examples, int count, KIFact* generalized, int* gen_count);
int ki_discover_patterns(KnowledgeInferenceEngine* kie, const KIFact* facts, int count, KIPattern* patterns, int* pattern_count, int max_patterns);
int ki_abstract_concepts(KnowledgeInferenceEngine* kie, const KIConcept* concepts, int count, KIConcept* abstracted, int* abs_count);

int ki_bayesian_init(KnowledgeInferenceEngine* kie, const char* component_name, float initial_mastery);
int ki_bayesian_update(KnowledgeInferenceEngine* kie, const char* component_name, int is_correct, int observation_weight);
int ki_bayesian_predict(KnowledgeInferenceEngine* kie, const char* component_name, float* predicted_performance);
int ki_bayesian_get_state(KnowledgeInferenceEngine* kie, KIBayesianState* state);

int ki_detect_conflicts(KnowledgeInferenceEngine* kie, const KIFact* new_facts, int new_count, KIConflict* conflicts, int* conflict_count);
int ki_resolve_conflicts(KnowledgeInferenceEngine* kie, KIConflict* conflicts, int count, int strategy, KIFact* resolved, int* resolved_count);

/**
 * @brief 深度多跳推理链查询
 * 
 * 从起始实体在知识图谱中执行BFS多跳遍历，
 * 每跳级联传递置信度(衰减因子0.9)，记录完整证据链。
 * 支持最大跳数控制、循环检测、置信度阈值过滤。
 *
 * @param kie 推理引擎句柄
 * @param start_entity 起始实体名称
 * @param target_pattern 目标匹配模式(NULL=推导所有可达实体)
 * @param max_hops 最大跳数(1-8)
 * @param inferred_facts 输出推理结果数组
 * @param max_facts 最大结果数
 * @param fact_count 输出实际结果数
 * @return int 成功返回0
 */
int ki_multi_hop_reason(KnowledgeInferenceEngine* kie,
                        const char* start_entity,
                        const char* target_pattern,
                        int max_hops,
                        KIFact* inferred_facts,
                        int max_facts,
                        int* fact_count);

/* K-修复: 贝叶斯网络变量消除法推理 */
int ki_bayesian_variable_elimination(
    int* query_vars, int query_count,
    int* evidence_vars, int* evidence_values, int evidence_count,
    int* all_vars, int* cardinalities, int var_count,
    float** cpt_list, int* cpt_var_indices, int* cpt_var_counts, int cpt_count,
    float* result_prob);

/**
 * @brief M-018修复: 知识推理→LNN连续状态桥接
 *
 * 将知识推理引擎的推理结果汇总映射为LNN状态扰动，
 * 建立"符号化知识推理→连续动态LNN→自主决策"的完整数据闭环。
 *
 * 内部调用ki_multi_hop_reason和ki_forward_chain生成推理结果，
 * 然后通过selflnn_consume_knowledge_inference将置信度加权
 * 的推理事实映射为LNN输入偏置扰动。
 *
 * @param kie        知识推理引擎句柄
 * @param lnn        LNN实例指针(void*避免头文件循环依赖)
 * @param concepts   核心概念名称数组(NULL=使用所有已注册概念)
 * @param conc_count 概念数量(0=自动探测引擎内注册概念)
 * @param strength   扰动强度(0.0-1.0)
 * @return 成功注入LNN的概念数，失败返回-1
 */
int ki_bridge_inference_to_lnn(KnowledgeInferenceEngine* kie, void* lnn,
                                const char** concepts, int conc_count,
                                float strength);

/* ============================================================================
 * H-002深度增强: 案例推理(Case-Based Reasoning) API
 * ============================================================================ */

/**
 * @brief 案例推理(CBR)完整4R流程
 *
 * 新推理范式：基于历史案例的问题解决引擎。
 * 实现Retrieve→Reuse→Revise→Retain完整循环，
 * 使用分层索引（谓词分类+FNV-1a哈希+语义相似度）加速检索。
 *
 * @param kie 推理引擎句柄
 * @param problem 问题事实数组
 * @param problem_count 问题事实数量
 * @param solutions 输出解决方案
 * @param sol_count 输出解决方案数量
 * @param best_similarity 输出最佳匹配相似度(可为NULL)
 * @return int 成功返回0，无匹配案例返回0(非错误)，失败返回-1
 */
int ki_case_based_reasoning(KnowledgeInferenceEngine* kie,
                             const KIFact* problem, int problem_count,
                             KIFact* solutions, int* sol_count,
                             float* best_similarity);

/**
 * @brief 向案例库添加案例
 *
 * @param problem 问题描述
 * @param solution 解决方案描述
 * @param success_rating 成功率(0-1)
 * @param problem_facts 问题事实(可为NULL)
 * @param fact_count 事实数量
 * @return int 成功返回case_id，失败返回-1
 */
int ki_cbr_add_case(const char* problem, const char* solution,
                     float success_rating,
                     const KIFact* problem_facts, int fact_count);

/**
 * @brief 获取案例库统计信息
 *
 * @param total_cases 输出总案例数
 * @param avg_success_rate 输出平均成功率
 * @param indexed_cases 输出已索引案例数
 * @return int 成功返回0
 */
int ki_cbr_get_stats(int* total_cases, float* avg_success_rate,
                      int* indexed_cases);

/* ============================================================================
 * H-002深度增强: 缺省推理(Default Reasoning) API
 * ============================================================================ */

/**
 * @brief 添加缺省推理规则
 *
 * 缺省逻辑形式：α : Mβ₁,...,Mβₙ / γ
 * 读作："如果α成立，且β₁...βₙ一致（无矛盾），则推断γ"
 *
 * @param rule_name 规则名称
 * @param prerequisites 前提条件α
 * @param prereq_count 前提数量
 * @param justifications 缺省假设β
 * @param just_count 假设数量
 * @param conclusion 结论γ
 * @param priority 优先级(0-1)
 * @return int 成功返回rule_id，失败返回-1
 */
int ki_default_add_rule(const char* rule_name,
                         const KIFact* prerequisites, int prereq_count,
                         const KIFact* justifications, int just_count,
                         const KIFact* conclusion, float priority);

/**
 * @brief 执行缺省推理
 *
 * 遍历所有缺省规则，对满足前提条件且假设一致的规则输出结论。
 * 支持多规则优先级排序和回溯修正。
 *
 * @param kie 推理引擎句柄
 * @param known_facts 已知事实（当前上下文）
 * @param known_count 已知事实数量
 * @param inferred 输出推理结果
 * @param inf_count 输出推理结果数量
 * @return int 成功返回0
 */
int ki_default_reasoning(KnowledgeInferenceEngine* kie,
                          const KIFact* known_facts, int known_count,
                          KIFact* inferred, int* inf_count);

/**
 * @brief 回溯修正—撤销与新事实矛盾的缺省推理
 *
 * 非单调推理的核心机制：当新事实出现时，
 * 检查并撤销不再一致的缺省推理结论。
 *
 * @param kie 推理引擎句柄
 * @param new_facts 新出现的事实
 * @param new_count 新事实数量
 * @param retracted 输出被撤销的结论
 * @param ret_count 输出被撤销数量
 * @return int 成功返回0
 */
int ki_default_retract(KnowledgeInferenceEngine* kie,
                        const KIFact* new_facts, int new_count,
                        KIFact* retracted, int* ret_count);

/**
 * @brief 获取缺省推理引擎统计
 */
int ki_default_get_stats(int* rule_count, int* applied_count,
                          float* avg_confidence);

/**
 * @brief 预置通用缺省规则
 *
 * 初始化一组常用缺省推理规则，涵盖：
 * 常识推理、分类推理、时序推理、因果推理等场景。
 * 包含鸟类飞行、朋友帮助、下雨地面湿等5条预置规则。
 *
 * @return int 成功返回0
 */
int ki_default_init_preset_rules(void);

void factor_free(void* f);

#ifdef __cplusplus
}
#endif
#endif
