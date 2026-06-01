#ifndef SELFLNN_DEEP_REFLECTION_H
#define SELFLNN_DEEP_REFLECTION_H

#include "selflnn/core/tensor.h"
#include "selflnn/cognition/metacognition.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DR_MAX_CHAIN_LEN 128
#define DR_MAX_INSIGHTS 64
#define DR_MAX_CONTRADICTIONS 32
#define DR_EMBED_DIM 512

typedef enum {
    DR_LAYER_DESCRIPTIVE = 0,
    DR_LAYER_ANALYTICAL = 1,
    DR_LAYER_CRITICAL = 2,
    DR_LAYER_CAUSAL = 3,
    DR_LAYER_EPISTEMIC = 4,
    DR_LAYER_META = 5,
    DR_LAYER_TRANSFORMATIVE = 6
} DRReflectionLayer;

typedef struct {
    DRReflectionLayer layer;
    float content_embedding[DR_EMBED_DIM];
    float depth_score;
    float novelty_score;
    float coherence_score;
    char* reflection_text;
    size_t text_len;
    float contradiction_flag;
    int needs_deeper;
} DRLayerResult;

typedef struct {
    DRLayerResult layers[DR_MAX_CHAIN_LEN];
    size_t num_layers;
    float overall_depth;
    float transformative_potential;
    float self_consistency;
    char* synthesis;
    size_t synthesis_len;
    float insight_scores[DR_MAX_INSIGHTS];
    size_t num_insights;
} DRReflectionChain;

typedef struct {
    float depth_threshold;
    float novelty_threshold;
    int max_iterations;
    int enable_causal_analysis;
    int enable_contradiction_detection;
    int enable_synthesis;
    float learning_rate;
    size_t embedding_dim;
} DRConfig;

#define DR_CONFIG_DEFAULT { \
    0.3f, 0.2f, 7, 1, 1, 1, 0.01f, DR_EMBED_DIM \
}

typedef enum {
    DR_RESOLVE_UNRESOLVED = 0,
    DR_RESOLVE_MERGED = 1,
    DR_RESOLVE_OVERRIDE_A = 2,
    DR_RESOLVE_OVERRIDE_B = 3,
    DR_RESOLVE_ABSTRACTED = 4
} DRConflictResolution;

typedef struct {
    size_t layer_a;
    size_t layer_b;
    float conflict_score;
    float bayesian_score;
    float nn_score;
    float similarity;
    char description[256];
    DRConflictResolution resolution;
} DRConflictInfo;

typedef struct DeepReflectionEngine DeepReflectionEngine;

DeepReflectionEngine* dr_engine_create(DRConfig config);

void dr_engine_destroy(DeepReflectionEngine* engine);

int dr_reflect(DeepReflectionEngine* engine,
                const char* topic,
                const float* context_data,
                size_t context_size,
                DRReflectionChain* chain_out);

int dr_reflect_multi_passage(DeepReflectionEngine* engine,
                              const char* topic,
                              const float** passages, size_t num_passages,
                              const size_t* passage_sizes,
                              DRReflectionChain* chain_out);

int dr_epistemic_assessment(DeepReflectionEngine* engine,
                             DRReflectionChain* chain,
                             float* knowledge_certainty_out,
                             float* knowledge_coverage_out,
                             float* knowledge_boundary_out);

int dr_detect_knowledge_conflicts(DeepReflectionEngine* engine,
                                    DRReflectionChain* chain,
                                    DRConflictInfo* conflicts_out,
                                    size_t* num_conflicts);

int dr_resolve_conflicts(DeepReflectionEngine* engine,
                          DRConflictInfo* conflicts,
                          size_t num_conflicts,
                          DRReflectionChain* chain_out);

int dr_generate_hypotheses(DeepReflectionEngine* engine,
                            DRReflectionChain* chain,
                            float* hypotheses_out,
                            size_t* num_hypotheses,
                            size_t max_hypotheses);

int dr_root_cause_analysis(DeepReflectionEngine* engine,
                            DRReflectionChain* chain,
                            size_t* root_layer_idx,
                            float* root_confidence);

int dr_get_insights(DeepReflectionEngine* engine,
                     DRReflectionChain* chain,
                     char** insights_out,
                     size_t* num_insights);

int dr_detect_contradictions(DeepReflectionEngine* engine,
                              DRReflectionChain* chain,
                              char** contradictions_out,
                              size_t* num_contradictions);

int dr_generate_synthesis(DeepReflectionEngine* engine,
                           DRReflectionChain* chain,
                           char* synthesis_out,
                           size_t max_len);

int dr_chain_to_plan(DeepReflectionEngine* engine,
                      DRReflectionChain* chain,
                      float* plan_actions,
                      size_t* num_actions);

int dr_integrate_with_metacognition(DeepReflectionEngine* engine,
                                     MetacognitionSystem* meta_system,
                                     DRReflectionChain* chain);

void dr_chain_free(DRReflectionChain* chain);

#ifdef __cplusplus
}
#endif

#endif
