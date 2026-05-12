/**
 * @file knowledge_inference.h
 * @brief 知识推理增强系统接口（传递/规则/类比/时序/反事实/图推理/归纳/贝叶斯/冲突消解）
 */

#ifndef SELFLNN_KNOWLEDGE_INFERENCE_H
#define SELFLNN_KNOWLEDGE_INFERENCE_H

#include <stddef.h>

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
    KI_RESOLVE_MERGE = 3
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

int ki_find_analogies(KnowledgeInferenceEngine* kie, const char* source_domain, const char* target_domain, KIAnalogy* out, int max_count);
int ki_map_analogy(KnowledgeInferenceEngine* kie, KIAnalogy* analogy);

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

#ifdef __cplusplus
}
#endif
#endif
