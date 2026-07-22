/**
 * @file knowledge_bridge.h
 * @brief knowledge 与 cognition 之间的类型桥接头文件
 *
 * M-005(K-CYCLE-002)修复: 打破 cognition -> knowledge -> cognition 循环依赖。
 * 此头文件只包含前向声明和函数指针类型，不产生任何链接依赖。
 * knowledge 模块通过此桥接头文件间接使用 cognition 的抽象能力，
 * cognition 在运行时通过注册函数注入具体实现。
 */
#ifndef SELFLNN_KNOWLEDGE_BRIDGE_H
#define SELFLNN_KNOWLEDGE_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== 前向声明（不包含cognition头文件） ========== */
typedef struct AbstractionSystem    AbstractionSystem;
typedef struct Concept              Concept;
typedef struct AnalogyMapping       AnalogyMapping;
typedef struct AbstractionState     AbstractionState;

/* 抽象配置结构体（与cognition/abstraction.h中定义保持一致） */
typedef struct {
    int    primary_abstraction_type;
    int    max_abstraction_levels;
    float  abstraction_threshold;
    float  similarity_threshold;
    int    concept_representation;
    int    enable_concept_formation;
    int    enable_analogical_reasoning;
    int    enable_pattern_induction;
    int    enable_metaphor_processing;
    float  compression_ratio;
    float  generalization_strength;
    int    enable_multimodal_abstraction;
    int    enable_dynamic_abstraction;
    int    enable_hierarchical_abstraction;
    float  learning_rate;
    float  forgetting_factor;
    int    min_instances_for_concept;
    int    max_accumulated_instances;
} BridgeAbstractionConfig;

/* 抽象层次枚举 */
typedef enum {
    BRIDGE_ABSTRACTION_CONCRETE = 0,
    BRIDGE_ABSTRACTION_FEATURE  = 1,
    BRIDGE_ABSTRACTION_CONCEPT  = 2,
    BRIDGE_ABSTRACTION_SYMBOLIC = 3,
    BRIDGE_ABSTRACTION_THEORY   = 4,
    BRIDGE_ABSTRACTION_METAPHOR = 5
} BridgeAbstractionLevel;

/* ========== 抽象能力函数指针表（运行时注入） ========== */
typedef struct {
    AbstractionSystem* (*create)      (const BridgeAbstractionConfig* config);
    void               (*destroy)     (AbstractionSystem* system);
    int                (*process)     (AbstractionSystem* system,
                                       const float* input, size_t input_size,
                                       int target_level,
                                       float* output, size_t max_output_size);
    int                (*learn_concept)(AbstractionSystem* system,
                                       const float** instances, size_t count,
                                       size_t instance_size, const char* name,
                                       Concept* concept);
    int                (*analogical_reasoning)(AbstractionSystem* system,
                                       const float* src, size_t src_sz,
                                       const float* tgt, size_t tgt_sz,
                                       AnalogyMapping* mapping, size_t max_mappings);
    int                (*pattern_induction)(AbstractionSystem* system,
                                       const float** patterns, size_t count,
                                       size_t pattern_size,
                                       float* induced, size_t max_size);
    int                (*get_state)   (AbstractionSystem* system,
                                       AbstractionState* state);
} KnowledgeAbstractionOps;

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_KNOWLEDGE_BRIDGE_H */