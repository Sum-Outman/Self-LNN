#ifndef SELFLNN_DEEP_THOUGHT_CHAIN_H
#define SELFLNN_DEEP_THOUGHT_CHAIN_H

#include "selflnn/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DTC_MAX_CHAIN_LEN 256
#define DTC_MAX_BRANCHES 16
#define DTC_EMBED_DIM 512

typedef enum {
    DTC_STEP_OBSERVE = 0,
    DTC_STEP_ANALYZE = 1,
    DTC_STEP_HYPOTHESIZE = 2,
    DTC_STEP_REASON = 3,
    DTC_STEP_EVALUATE = 4,
    DTC_STEP_SYNTHESIZE = 5,
    DTC_STEP_CONCLUDE = 6,
    DTC_STEP_ACT = 7
} DTCStepType;

typedef struct {
    DTCStepType step_type;
    float thought_embedding[DTC_EMBED_DIM];
    float confidence;
    float uncertainty;
    float branching_factor;
    char* thought_text;
    size_t text_len;
    int has_branches;
    size_t parent_index;
} DTCThoughtNode;

typedef struct {
    DTCThoughtNode nodes[DTC_MAX_CHAIN_LEN];
    size_t num_nodes;
    float chain_coherence;
    float reasoning_depth;
    float solution_confidence;
    float* final_output;
    size_t output_dim;
} DTCChainResult;

typedef struct {
    float temperature;
    float exploration_rate;
    size_t max_depth;
    size_t max_branches;
    int enable_backtracking;
    int enable_parallel_branches;
    float confidence_threshold;
    float branching_threshold;
    int beam_width;
    int enable_self_consistency;
} DTCConfig;

#define DTC_CONFIG_DEFAULT { \
    0.3f, 0.2f, 64, 4, 1, 1, 0.3f, 0.5f, 3, 1 \
}

typedef struct DTCSystem DTCSystem;

DTCSystem* dtc_system_create(DTCConfig config);

void dtc_system_destroy(DTCSystem* system);

int dtc_reason_chain(DTCSystem* system,
                      const float* input_data,
                      size_t input_dim,
                      const char* query,
                      DTCChainResult* result_out);

int dtc_expand_branch(DTCSystem* system,
                       DTCChainResult* chain,
                       size_t node_index,
                       DTCStepType branch_type);

int dtc_beam_search(DTCSystem* system,
                     const float* input_data,
                     size_t input_dim,
                     const char* query,
                     int beam_width,
                     DTCChainResult* result_out);

int dtc_self_consistency_rerank(DTCSystem* system,
                                 DTCChainResult* chain,
                                 DTCChainResult* reranked_out);

int dtc_merge_thoughts(DTCSystem* system,
                        DTCChainResult* chain,
                        size_t* merge_pairs,
                        size_t num_pairs,
                        DTCChainResult* merged_out);

int dtc_evaluate_chain(DTCSystem* system,
                        DTCChainResult* chain,
                        float* coherence_out,
                        float* depth_out,
                        float* confidence_out);

int dtc_prune_chain(DTCSystem* system,
                     DTCChainResult* chain,
                     float min_confidence);

int dtc_backtrack(DTCSystem* system,
                   DTCChainResult* chain,
                   size_t to_index);

int dtc_get_best_path(DTCSystem* system,
                       DTCChainResult* chain,
                       size_t* path_indices,
                       size_t* path_len);

int dtc_chain_to_text(DTCSystem* system,
                       DTCChainResult* chain,
                       char* text_out,
                       size_t max_len);

int dtc_chain_to_plan(DTCSystem* system,
                       DTCChainResult* chain,
                       float* plan_out,
                       size_t* plan_len,
                       size_t max_plan_steps);

void dtc_chain_free(DTCChainResult* chain);

#ifdef __cplusplus
}
#endif

#endif
