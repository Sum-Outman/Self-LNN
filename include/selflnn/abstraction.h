/**
 * @file abstraction.h
 * @brief 抽象能力系统接口
 * 
 * 抽象能力（Abstraction）系统实现，支持：
 * 1. 概念学习与形成（Concept Learning & Formation）
 * 2. 符号抽象与表示（Symbolic Abstraction & Representation）
 * 3. 类比推理与迁移（Analogical Reasoning & Transfer）
 * 4. 模式识别与归纳（Pattern Recognition & Induction）
 * 5. 范畴化与分类（Categorization & Classification）
 * 6. 隐喻理解与生成（Metaphor Understanding & Generation）
 * 
 *  ，提供完整的抽象能力算法。
 */

#ifndef SELFLNN_ABSTRACTION_H
#define SELFLNN_ABSTRACTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 抽象能力类型
 */
typedef enum {
    ABSTRACTION_CONCEPT_LEARNING = 0,      /**< 概念学习 */
    ABSTRACTION_SYMBOLIC_REPRESENTATION = 1, /**< 符号表示 */
    ABSTRACTION_ANALOGICAL_REASONING = 2,  /**< 类比推理 */
    ABSTRACTION_PATTERN_RECOGNITION = 3,   /**< 模式识别 */
    ABSTRACTION_CATEGORIZATION = 4,        /**< 范畴化 */
    ABSTRACTION_METAPHOR_PROCESSING = 5,   /**< 隐喻处理 */
    ABSTRACTION_INDUCTION = 6,             /**< 归纳推理 */
    ABSTRACTION_COMPRESSION = 7,           /**< 抽象压缩 */
    ABSTRACTION_GENERALIZATION = 8,        /**< 泛化能力 */
    ABSTRACTION_MULTI_LEVEL = 9            /**< 多级抽象 */
} AbstractionType;

/**
 * @brief 抽象层次
 */
typedef enum {
    ABSTRACTION_LEVEL_CONCRETE = 0,        /**< 具体层：原始数据 */
    ABSTRACTION_LEVEL_FEATURE = 1,         /**< 特征层：特征提取 */
    ABSTRACTION_LEVEL_CONCEPT = 2,         /**< 概念层：概念形成 */
    ABSTRACTION_LEVEL_SYMBOLIC = 3,        /**< 符号层：符号表示 */
    ABSTRACTION_LEVEL_THEORY = 4,          /**< 理论层：理论构建 */
    ABSTRACTION_LEVEL_METAPHOR = 5         /**< 隐喻层：隐喻理解 */
} AbstractionLevel;

/**
 * @brief 概念表示形式
 */
typedef enum {
    CONCEPT_REPRESENTATION_VECTOR = 0,     /**< 向量表示 */
    CONCEPT_REPRESENTATION_SYMBOL = 1,     /**< 符号表示 */
    CONCEPT_REPRESENTATION_GRAPH = 2,      /**< 图表示 */
    CONCEPT_REPRESENTATION_PROTOTYPE = 3,  /**< 原型表示 */
    CONCEPT_REPRESENTATION_EXEMPLAR = 4,   /**< 样例表示 */
    CONCEPT_REPRESENTATION_HYBRID = 5      /**< 混合表示 */
} ConceptRepresentation;

/**
 * @brief 抽象能力配置
 */
typedef struct {
    AbstractionType primary_type;          /**< 主要抽象类型 */
    int max_abstraction_levels;            /**< 最大抽象层次数 */
    float abstraction_threshold;           /**< 抽象阈值 */
    float similarity_threshold;            /**< 相似度阈值 */
    ConceptRepresentation concept_representation; /**< 概念表示形式 */
    int enable_concept_formation;          /**< 是否启用概念形成 */
    int enable_analogical_reasoning;       /**< 是否启用类比推理 */
    int enable_pattern_induction;          /**< 是否启用模式归纳 */
    int enable_metaphor_processing;        /**< 是否启用隐喻处理 */
    float compression_ratio;               /**< 压缩比率 */
    float generalization_strength;         /**< 泛化强度 */
    int enable_multimodal_abstraction;     /**< 是否启用多模态抽象 */
    int enable_dynamic_abstraction;        /**< 是否启用动态抽象 */
    int enable_hierarchical_abstraction;   /**< 是否启用分层抽象 */
    float learning_rate;                   /**< 学习率 */
    float forgetting_factor;               /**< 遗忘因子 */
    int min_instances_for_concept;         /**< 形成概念的最小实例数（积累足够实例后才形成概念） */
    int max_accumulated_instances;         /**< 最大积累实例数（概念学习缓冲区容量） */
} AbstractionConfig;

/**
 * @brief 概念结构
 */
typedef struct {
    char* name;                           /**< 概念名称 */
    float* representation;                /**< 概念表示 */
    size_t representation_size;           /**< 表示大小 */
    AbstractionLevel level;               /**< 抽象层次 */
    float prototype_strength;             /**< 原型强度 */
    float typicality;                     /**< 典型性 */
    float centrality;                     /**< 中心性 */
    float* features;                      /**< 特征向量 */
    size_t features_size;                 /**< 特征大小 */
    int instance_count;                   /**< 实例数量 */
    float* boundaries;                    /**< 概念边界 */
    size_t boundaries_size;               /**< 边界大小 */
    int is_abstract;                      /**< 是否为抽象概念 */
    int is_learned;                       /**< 是否已学习 */
} Concept;

/**
 * @brief 类比映射
 */
typedef struct {
    Concept* source_concept;              /**< 源概念 */
    Concept* target_concept;              /**< 目标概念 */
    float* mapping_weights;               /**< 映射权重 */
    size_t mapping_size;                  /**< 映射大小 */
    float similarity_score;               /**< 相似度得分 */
    float structural_score;               /**< 结构得分 */
    float pragmatic_score;                /**< 语用得分 */
    int is_valid;                         /**< 是否有效 */
    char* relation_description;           /**< 关系描述 */
} AnalogyMapping;

/**
 * @brief 抽象能力状态
 */
typedef struct {
    Concept** concepts;                   /**< 概念数组 */
    size_t concepts_count;                /**< 概念数量 */
    size_t concepts_capacity;             /**< 概念容量 */
    AbstractionLevel current_level;       /**< 当前抽象层次 */
    float abstraction_quality;            /**< 抽象质量 */
    float compression_efficiency;         /**< 压缩效率 */
    float generalization_ability;         /**< 泛化能力 */
    int total_abstractions;               /**< 总抽象次数 */
    int successful_abstractions;          /**< 成功抽象次数 */
    int failed_abstractions;              /**< 失败抽象次数 */
    int is_initialized;                   /**< 是否已初始化 */
} AbstractionState;

/**
 * @brief 抽象能力系统句柄
 */
typedef struct AbstractionSystem AbstractionSystem;

/**
 * @brief 创建抽象能力系统
 * 
 * @param config 抽象能力配置
 * @return AbstractionSystem* 抽象能力系统句柄，失败返回NULL
 */
AbstractionSystem* abstraction_system_create(const AbstractionConfig* config);

/**
 * @brief 释放抽象能力系统
 * 
 * @param system 抽象能力系统句柄
 */
void abstraction_system_free(AbstractionSystem* system);

/**
 * @brief 执行抽象处理
 * 
 * @param system 抽象能力系统句柄
 * @param input 输入数据
 * @param input_size 输入数据大小
 * @param target_level 目标抽象层次
 * @param abstracted_output 抽象输出缓冲区
 * @param max_output_size 输出最大大小
 * @return int 成功返回抽象输出大小，失败返回-1
 */
int abstraction_process(AbstractionSystem* system,
                       const float* input, size_t input_size,
                       AbstractionLevel target_level,
                       float* abstracted_output, size_t max_output_size);

/**
 * @brief 概念学习与形成
 * 
 * @param system 抽象能力系统句柄
 * @param instances 实例数据
 * @param instances_count 实例数量
 * @param instance_size 实例大小
 * @param concept_name 概念名称
 * @param learned_concept 学习到的概念输出
 * @return int 成功返回0，失败返回-1
 */
int abstraction_learn_concept(AbstractionSystem* system,
                             const float** instances, size_t instances_count,
                             size_t instance_size, const char* concept_name,
                             Concept* learned_concept);

/**
 * @brief 概念检索与匹配
 * 
 * @param system 抽象能力系统句柄
 * @param instance 实例数据
 * @param instance_size 实例大小
 * @param matched_concept 匹配的概念输出
 * @param similarity_score 相似度得分输出
 * @return int 成功返回0，失败返回-1
 */
int abstraction_match_concept(AbstractionSystem* system,
                             const float* instance, size_t instance_size,
                             Concept* matched_concept, float* similarity_score);

/**
 * @brief 类比推理
 * 
 * @param system 抽象能力系统句柄
 * @param source_domain 源领域数据
 * @param source_size 源领域大小
 * @param target_domain 目标领域数据
 * @param target_size 目标领域大小
 * @param analogy_mapping 类比映射输出
 * @param max_mappings 最大映射数
 * @return int 成功返回映射数量，失败返回-1
 */
int abstraction_analogical_reasoning(AbstractionSystem* system,
                                    const float* source_domain, size_t source_size,
                                    const float* target_domain, size_t target_size,
                                    AnalogyMapping* analogy_mapping, size_t max_mappings);

/**
 * @brief 释放类比映射结果中的动态分配内存
 * 
 * 释放在类比推理过程中分配的 mapping_weights 和 relation_description，
 * 注意：source_concept 和 target_concept 指向概念库中的概念，由
 * abstraction_system_free 统一清理。
 * 
 * @param mappings 类比映射数组
 * @param count 映射数量
 */
void abstraction_mapping_free_all(AnalogyMapping* mappings, size_t count);

/**
 * @brief 模式识别与归纳
 * 
 * @param system 抽象能力系统句柄
 * @param patterns 模式数据
 * @param patterns_count 模式数量
 * @param pattern_size 模式大小
 * @param induced_pattern 归纳出的模式输出
 * @param max_pattern_size 模式最大大小
 * @return int 成功返回归纳模式大小，失败返回-1
 */
int abstraction_pattern_induction(AbstractionSystem* system,
                                 const float** patterns, size_t patterns_count,
                                 size_t pattern_size,
                                 float* induced_pattern, size_t max_pattern_size);

/**
 * @brief 抽象层次转换
 * 
 * @param system 抽象能力系统句柄
 * @param input 输入数据
 * @param input_size 输入数据大小
 * @param source_level 源抽象层次
 * @param target_level 目标抽象层次
 * @param converted_output 转换输出缓冲区
 * @param max_output_size 输出最大大小
 * @return int 成功返回转换输出大小，失败返回-1
 */
int abstraction_level_conversion(AbstractionSystem* system,
                                const float* input, size_t input_size,
                                AbstractionLevel source_level,
                                AbstractionLevel target_level,
                                float* converted_output, size_t max_output_size);

/**
 * @brief 概念网络构建
 * 
 * @param system 抽象能力系统句柄
 * @param concepts 概念数组
 * @param concepts_count 概念数量
 * @param relation_strength 关系强度矩阵输出
 * @param max_relations 最大关系数
 * @return int 成功返回关系数量，失败返回-1
 */
int abstraction_build_concept_network(AbstractionSystem* system,
                                     Concept** concepts, size_t concepts_count,
                                     float* relation_strength, size_t max_relations);

/**
 * @brief 隐喻理解与生成
 * 
 * @param system 抽象能力系统句柄
 * @param source_concept 源概念
 * @param target_domain 目标领域
 * @param target_size 目标领域大小
 * @param metaphor_interpretation 隐喻解释输出
 * @param max_interpretation_size 解释最大大小
 * @return int 成功返回解释大小，失败返回-1
 */
int abstraction_metaphor_processing(AbstractionSystem* system,
                                   const Concept* source_concept,
                                   const float* target_domain, size_t target_size,
                                   char* metaphor_interpretation, size_t max_interpretation_size);

/**
 * @brief 抽象质量评估
 * 
 * @param system 抽象能力系统句柄
 * @param abstracted_data 抽象数据
 * @param abstracted_size 抽象数据大小
 * @param original_data 原始数据
 * @param original_size 原始数据大小
 * @param quality_metrics 质量指标输出
 * @param max_metrics 最大指标数
 * @return int 成功返回指标数量，失败返回-1
 */
int abstraction_evaluate_quality(AbstractionSystem* system,
                                const float* abstracted_data, size_t abstracted_size,
                                const float* original_data, size_t original_size,
                                float* quality_metrics, size_t max_metrics);

/**
 * @brief 获取抽象能力状态
 * 
 * @param system 抽象能力系统句柄
 * @param state 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int abstraction_get_state(AbstractionSystem* system, AbstractionState* state);

/**
 * @brief 重置抽象能力系统
 * 
 * @param system 抽象能力系统句柄
 * @return int 成功返回0，失败返回-1
 */
int abstraction_reset(AbstractionSystem* system);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_ABSTRACTION_H */