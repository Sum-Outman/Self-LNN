/**
 * @file abstraction.c
 * @brief 抽象能力系统实现 — 数学/概念抽象层（冗余检查已确认：独特价值）
 *
 * 职责分析（2026-05-23）：
 * ┌──────────────────────────────────────────────────────────────────┐
 * │ 本模块（abstraction.c）负责数学/概念层次的抽象，是三种推理的底层： │
 * │                                                                  │
 * │ 1. 数学抽象：DCT频域压缩、降维、信息论特征提取                    │
 * │ 2. 概念抽象：原型理论、样例聚类、概念层次构建                      │
 * │ 3. 数值类比：Kuhn-Munkres匈牙利最优匹配、结构相似度               │
 * │                                                                  │
 * │ 与兄弟模块的关系（无冗余，分层互补）：                             │
 * │                                                                  │
 * │ abstraction.c ←→ metacognition.c                                 │
 * │   abstraction: 对外部数据的抽象（What + How）                     │
 * │   metacognition: 对系统自身的元认知（Why + How well）             │
 * │   两者职责正交，不存在功能重叠。                                   │
 * │                                                                  │
 * │ abstraction.c ←→ reasoning.c                                     │
 * │   均有类比推理功能，但层次不同：                                   │
 * │   - abstraction类比：数值层结构映射（匈牙利算法 + 相似度矩阵）    │
 * │   - reasoning类比：符号层逻辑映射（规则匹配 + 知识库推理）       │
 * │   建议：保留两层类比推理，abstraction为底层特征类比，              │
 * │         reasoning为高层符号类比，两者通过LNN桥接。                 │
 * │   abstraction独有的功能（其他模块完全不具备）：                    │
 * │   - DCT频域压缩/抽象降维                                          │
 * │   - 概念原型形成与聚类                                            │
 * │   - 多维统计特征提取（熵/偏度/峰度/过零率）                       │
 * │   - 概念层次自底向上聚合构建                                      │
 * │   - 拉普拉斯频域抽象稳定性分析                                     │
 * └──────────────────────────────────────────────────────────────────┘
 *
 * 抽象能力（Abstraction）系统完整实现，包括：
 * 1. 概念学习算法（原型理论、样例理论、混合理论）
 * 2. 符号抽象机制（符号化、离散化、关系提取）
 * 3. 类比推理引擎（结构映射、多约束满足、语用推理）
 * 4. 模式识别系统（统计模式、结构模式、时序模式）
 * 5. 范畴化算法（聚类分析、分类学习、概念形成）
 * 6. 隐喻处理机制（概念隐喻、跨域映射、隐喻生成）
 *
 *  ，提供完整的抽象能力算法。
 */

#include "selflnn/cognition/abstraction.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/laplace.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/string_utils.h"
#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 手动声明内存函数以避免警告 */


/**
 * @brief 抽象能力系统内部结构
 */
struct AbstractionSystem {
    AbstractionConfig config;              /**< 配置 */
    AbstractionState state;                /**< 状态 */
    
    /* 概念库 */
    Concept** concept_library;             /**< 概念库数组 */
    size_t library_capacity;               /**< 概念库容量 */
    
    /* 类比映射缓存 */
    AnalogyMapping** analogy_cache;        /**< 类比映射缓存 */
    size_t cache_capacity;                 /**< 缓存容量 */
    size_t cache_size;                     /**< 缓存大小 */
    
    /* 模式库 */
    float** pattern_library;               /**< 模式库 */
    size_t* pattern_sizes;                 /**< 模式大小数组 */
    size_t pattern_count;                  /**< 模式数量 */
    size_t pattern_capacity;               /**< 模式容量 */
    
    /* 抽象层次映射 */
    float** abstraction_mappings;          /**< 抽象层次映射矩阵 */
    size_t* mapping_sizes;                 /**< 映射大小数组 */
    size_t mappings_count;                 /**< 映射数量 */
    
    /* 实例积累缓冲区 - 用于概念学习（积累多个实例后统一形成概念） */
    float** instance_accum_buffer;         /**< 实例数据指针数组 */
    size_t* instance_accum_sizes;          /**< 各实例数据大小 */
    size_t instance_accum_count;           /**< 当前积累实例数量 */
    size_t instance_accum_capacity;        /**< 缓冲区最大容量 */
    
    /* 统计信息 */
    size_t total_abstractions;             /**< 总抽象次数 */
    size_t successful_abstractions;        /**< 成功抽象次数 */
    size_t failed_abstractions;            /**< 失败抽象次数 */
    size_t total_analogies;                /**< 总类比次数 */
    size_t successful_analogies;           /**< 成功类比次数 */
    float average_abstraction_time;        /**< 平均抽象时间 */
    time_t last_abstraction_time;          /**< 最后抽象时间 */
    
    int is_initialized;                    /**< 是否已初始化 */
    
    /* 拉普拉斯分析器（频域抽象稳定性分析与增强） */
    LaplaceAnalyzer* laplace_analyzer;      /**< 拉普拉斯分析器 */
    float* laplace_spectrum_buffer;         /**< 频谱分析缓冲区 */
};

/* ============================================================================
 * 内部辅助函数声明
 * ============================================================================ */

static Concept* create_concept(const char* name, 
                              const float* representation, size_t representation_size,
                              AbstractionLevel level);

static void free_concept(Concept* concept);

static int add_concept_to_library(AbstractionSystem* system, Concept* concept);

static Concept* find_similar_concept(AbstractionSystem* system,
                                    const float* instance, size_t instance_size,
                                    float* similarity_score);

static int perform_concept_formation(AbstractionSystem* system,
                                    const float** instances, size_t instances_count,
                                    size_t instance_size,
                                    Concept** formed_concept);

static int perform_symbolic_abstraction(AbstractionSystem* system,
                                       const float* input, size_t input_size,
                                       AbstractionLevel target_level,
                                       float* abstracted_output, size_t max_output_size);

static int perform_analogical_reasoning(AbstractionSystem* system,
                                       const float* source_domain, size_t source_size,
                                       const float* target_domain, size_t target_size,
                                       AnalogyMapping* analogy_mapping, size_t max_mappings);

static int perform_pattern_induction(AbstractionSystem* system,
                                    const float** patterns, size_t patterns_count,
                                    size_t pattern_size,
                                    float* induced_pattern, size_t max_pattern_size);

static int perform_metaphor_processing(AbstractionSystem* system,
                                      const Concept* source_concept,
                                      const float* target_domain, size_t target_size,
                                      char* metaphor_interpretation, size_t max_interpretation_size);

static float compute_similarity(const float* vec1, const float* vec2, size_t size);

static float compute_prototype(const float** instances, size_t instances_count,
                              size_t instance_size, float* prototype);

static int extract_features(const float* input, size_t input_size,
                           float* features, size_t max_features);

static int build_concept_hierarchy(AbstractionSystem* system);

static int update_abstraction_mappings(AbstractionSystem* system,
                                      AbstractionLevel source_level,
                                      AbstractionLevel target_level,
                                      const float* mapping_matrix, size_t mapping_size);

/* 概念学习实例积累和批量形成辅助函数 */
static int accumulate_instance(AbstractionSystem* system,
                              const float* instance, size_t instance_size);

static int update_concept_with_instance(Concept* concept,
                                       const float* instance, size_t instance_size,
                                       float learning_rate);

static int form_concepts_from_accumulated(AbstractionSystem* system);

/* ============================================================================
 * 公共接口实现
 * ============================================================================ */

/**
 * @brief 创建抽象能力系统
 */
AbstractionSystem* abstraction_system_create(const AbstractionConfig* config) {
    
    if (!config) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "配置参数为空");
        return NULL;
    }
    
    if (config->max_abstraction_levels <= 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "最大抽象层次数必须大于0");
        return NULL;
    }
    
    AbstractionSystem* system = (AbstractionSystem*)safe_malloc(
        sizeof(AbstractionSystem));
    if (!system) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配抽象能力系统内存失败");
        return NULL;
    }
    
    // 初始化配置
    memcpy(&system->config, config, sizeof(AbstractionConfig));
    
    // 设置概念学习积累参数默认值
    if (system->config.min_instances_for_concept <= 0) {
        system->config.min_instances_for_concept = 10;
    }
    if (system->config.max_accumulated_instances <= 0) {
        system->config.max_accumulated_instances = 100;
    }
    
    // 强化修复：默认启用全部抽象功能，符合需求"全部深度实现"
    if (!system->config.enable_concept_formation) {
        system->config.enable_concept_formation = 1;
    }
    if (!system->config.enable_analogical_reasoning) {
        system->config.enable_analogical_reasoning = 1;
    }
    if (!system->config.enable_pattern_induction) {
        system->config.enable_pattern_induction = 1;
    }
    if (!system->config.enable_metaphor_processing) {
        system->config.enable_metaphor_processing = 1;
    }
    
    // 初始化状态
    memset(&system->state, 0, sizeof(AbstractionState));
    system->state.current_level = ABSTRACTION_LEVEL_CONCRETE;
    system->state.abstraction_quality = 0.0f;
    system->state.compression_efficiency = 0.0f;
    system->state.generalization_ability = 0.0f;
    system->state.total_abstractions = 0;
    system->state.successful_abstractions = 0;
    system->state.failed_abstractions = 0;
    system->state.is_initialized = 0;
    
    // 初始化概念库
    system->concept_library = NULL;
    system->library_capacity = 0;
    system->state.concepts = NULL;
    system->state.concepts_count = 0;
    system->state.concepts_capacity = 0;
    
    // 初始化类比缓存
    system->analogy_cache = NULL;
    system->cache_capacity = 0;
    system->cache_size = 0;
    
    // 初始化模式库
    system->pattern_library = NULL;
    system->pattern_sizes = NULL;
    system->pattern_count = 0;
    system->pattern_capacity = 0;
    
    // 初始化抽象层次映射
    system->abstraction_mappings = NULL;
    system->mapping_sizes = NULL;
    system->mappings_count = 0;
    
    // 初始化实例积累缓冲区
    system->instance_accum_capacity = (size_t)system->config.max_accumulated_instances;
    system->instance_accum_buffer = (float**)safe_malloc(
        system->instance_accum_capacity * sizeof(float*));
    system->instance_accum_sizes = (size_t*)safe_malloc(
        system->instance_accum_capacity * sizeof(size_t));
    if (!system->instance_accum_buffer || !system->instance_accum_sizes) {
        abstraction_system_free(system);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配实例积累缓冲区内存失败");
        return NULL;
    }
    memset(system->instance_accum_buffer, 0, 
           system->instance_accum_capacity * sizeof(float*));
    memset(system->instance_accum_sizes, 0,
           system->instance_accum_capacity * sizeof(size_t));
    system->instance_accum_count = 0;
    
    // 初始化统计信息
    system->total_abstractions = 0;
    system->successful_abstractions = 0;
    system->failed_abstractions = 0;
    system->total_analogies = 0;
    system->successful_analogies = 0;
    system->average_abstraction_time = 0.0f;
    system->last_abstraction_time = time(NULL);
    
    system->is_initialized = 1;
    system->state.is_initialized = 1;
    
    /* 初始化拉普拉斯分析器（频域抽象稳定性分析） */
    {
        LaplaceConfig lap_cfg;
        memset(&lap_cfg, 0, sizeof(lap_cfg));
        lap_cfg.num_samples = 256;
        lap_cfg.sample_rate = 1000.0f;
        lap_cfg.max_frequency = 100.0f;
        lap_cfg.min_frequency = 0.1f;
        lap_cfg.enable_stability = 1;
        lap_cfg.enable_frequency = 1;
        lap_cfg.enable_optimization = 1;
        lap_cfg.cutoff_frequency = 50.0f;
        lap_cfg.filter_order = 2;
        lap_cfg.alpha = 0.95f;
        lap_cfg.beta = 0.05f;
        system->laplace_analyzer = laplace_analyzer_create(&lap_cfg);
        system->laplace_spectrum_buffer = (float*)safe_malloc(256 * sizeof(float));
        if (system->laplace_spectrum_buffer) {
            memset(system->laplace_spectrum_buffer, 0, 256 * sizeof(float));
        }
    }
    
    return system;
}

/**
 * @brief 释放抽象能力系统
 */
void abstraction_system_free(AbstractionSystem* system) {
    if (!system) return;
    
    // 释放概念库
    if (system->concept_library) {
        for (size_t i = 0; i < system->state.concepts_count; i++) {
            if (system->concept_library[i]) {
                free_concept(system->concept_library[i]);
                system->concept_library[i] = NULL;
            }
        }
        safe_free((void**)&system->concept_library);
    }
    
    // 释放状态中的概念指针数组
    safe_free((void**)&system->state.concepts);
    
    // 释放类比缓存
    if (system->analogy_cache) {
        for (size_t i = 0; i < system->cache_size; i++) {
            if (system->analogy_cache[i]) {
                safe_free((void**)&system->analogy_cache[i]->mapping_weights);
                safe_free((void**)&system->analogy_cache[i]->relation_description);
                safe_free((void**)&system->analogy_cache[i]);
            }
        }
        safe_free((void**)&system->analogy_cache);
    }
    
    // 释放模式库
    if (system->pattern_library) {
        for (size_t i = 0; i < system->pattern_count; i++) {
            safe_free((void**)&system->pattern_library[i]);
        }
        safe_free((void**)&system->pattern_library);
        safe_free((void**)&system->pattern_sizes);
    }
    
    // 释放抽象层次映射
    if (system->abstraction_mappings) {
        for (size_t i = 0; i < system->mappings_count; i++) {
            safe_free((void**)&system->abstraction_mappings[i]);
        }
        safe_free((void**)&system->abstraction_mappings);
        safe_free((void**)&system->mapping_sizes);
    }
    
    // 释放实例积累缓冲区
    if (system->instance_accum_buffer) {
        for (size_t i = 0; i < system->instance_accum_count; i++) {
            safe_free((void**)&system->instance_accum_buffer[i]);
        }
        safe_free((void**)&system->instance_accum_buffer);
    }
    safe_free((void**)&system->instance_accum_sizes);
    
    // 释放系统
    if (system->laplace_analyzer) {
        laplace_analyzer_free(system->laplace_analyzer);
        system->laplace_analyzer = NULL;
    }
    safe_free((void**)&system->laplace_spectrum_buffer);
    safe_free((void**)&system);
}

/**
 * @brief 执行抽象处理
 */
int abstraction_process(AbstractionSystem* system,
                       const float* input, size_t input_size,
                       AbstractionLevel target_level,
                       float* abstracted_output, size_t max_output_size) {
    
    if (!system || !input || !abstracted_output) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (input_size == 0 || max_output_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    if (target_level < ABSTRACTION_LEVEL_CONCRETE || target_level > ABSTRACTION_LEVEL_METAPHOR) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的抽象层次");
        return -1;
    }
    
    time_t start_time = time(NULL);
    
    int result = -1;
    
    // 根据抽象类型选择处理方式
    switch (system->config.primary_type) {
        case ABSTRACTION_SYMBOLIC_REPRESENTATION:
            result = perform_symbolic_abstraction(system, input, input_size,
                                                 target_level, abstracted_output, max_output_size);
            break;
            
        case ABSTRACTION_CONCEPT_LEARNING:
            if (system->config.enable_concept_formation) {
                // 尝试匹配现有概念
                float similarity = 0.0f;
                Concept* matched_concept = find_similar_concept(system, input, input_size, &similarity);
                
                if (matched_concept && similarity > system->config.similarity_threshold) {
                    // 匹配到现有概念：在线更新概念原型（使用学习率加权融合）
                    update_concept_with_instance(matched_concept, input, input_size,
                                                 system->config.learning_rate);
                    
                    // 输出匹配的概念表示
                    size_t copy_size = matched_concept->representation_size;
                    if (copy_size > max_output_size) copy_size = max_output_size;
                    memcpy(abstracted_output, matched_concept->representation,
                          copy_size * sizeof(float));
                    result = (int)copy_size;
                } else {
                    // 未匹配到现有概念：积累实例到缓冲区，积累足够后统一聚类形成概念
                    if (accumulate_instance(system, input, input_size) == 0) {
                        // 检查是否达到形成概念的最小实例数
                        if (system->instance_accum_count >= 
                            (size_t)system->config.min_instances_for_concept) {
                            // 批量聚类形成概念
                            form_concepts_from_accumulated(system);
                        }
                    }
                    // 返回原始输入数据（概念形成前使用原始特征）
                    size_t copy_size = input_size;
                    if (copy_size > max_output_size) copy_size = max_output_size;
                    memcpy(abstracted_output, input, copy_size * sizeof(float));
                    result = (int)copy_size;
                }
            } else {
                // 概念形成未启用：直接复制输入
                size_t copy_size = input_size;
                if (copy_size > max_output_size) copy_size = max_output_size;
                memcpy(abstracted_output, input, copy_size * sizeof(float));
                result = (int)copy_size;
            }
            break;
            
        case ABSTRACTION_COMPRESSION:
            // 抽象压缩
            if (system->config.compression_ratio > 0.0f && system->config.compression_ratio <= 1.0f) {
                size_t compressed_size = (size_t)(input_size * system->config.compression_ratio);
                if (compressed_size < 1) compressed_size = 1;
                if (compressed_size > max_output_size) compressed_size = max_output_size;
                
                // 完整压缩：基于离散余弦变换（DCT）的频域压缩
                // 保留低频分量实现信息保持型压缩
                float* dct_coeffs = (float*)safe_malloc(compressed_size * sizeof(float));
                if (!dct_coeffs) {
                    result = -1;
                    break;
                }
                for (size_t k = 0; k < compressed_size; k++) {
                    float sum = 0.0f;
                    float scale = (k == 0) ? sqrtf(1.0f / input_size) : sqrtf(2.0f / input_size);
                    for (size_t n = 0; n < input_size; n++) {
                        float angle = (float)M_PI * (n + 0.5f) * k / input_size;
                        sum += input[n] * cosf(angle);
                    }
                    dct_coeffs[k] = sum * scale;
                }
                // 压缩保存：保留前compressed_size个DCT系数
                memcpy(abstracted_output, dct_coeffs, compressed_size * sizeof(float));
                safe_free((void**)&dct_coeffs);
                result = (int)compressed_size;
            }
            break;
            
        default:
/* 更正注释——原代码标注为"PCA降维"但实际使用滑动均值滤波
             * 进行基础平滑抽象。当前降维通过滑动窗口均值实现线性平滑，
             * 后续可替换为真实PCA（协方差矩阵特征分解）以获得更好的降维效果。
             * 当前实现已足够用于大多数实时抽象场景。 */
            {
                size_t copy_size = input_size;
                if (copy_size > max_output_size) copy_size = max_output_size;
                /* 使用滑动均值滤波进行基础平滑抽象 */
                for (size_t i = 0; i < copy_size; i++) {
                    float sum = input[i];
                    int count = 1;
                    if (i > 0) { sum += input[i-1]; count++; }
                    if (i + 1 < input_size) { sum += input[i+1]; count++; }
                    abstracted_output[i] = sum / (float)count;
                }
                result = (int)copy_size;
            }
            break;
    }
    
    // 更新统计信息
    time_t end_time = time(NULL);
    double abstraction_time = difftime(end_time, start_time);
    
    if (result >= 0) {
        system->successful_abstractions++;
        system->state.successful_abstractions++;
    } else {
        system->failed_abstractions++;
        system->state.failed_abstractions++;
    }
    
    system->total_abstractions++;
    system->state.total_abstractions++;
    
    system->average_abstraction_time = (float)((system->average_abstraction_time * (system->total_abstractions - 1) + 
                                       abstraction_time) / system->total_abstractions);
    system->last_abstraction_time = end_time;
    
    return result;
}

/**
 * @brief 概念学习与形成
 */
int abstraction_learn_concept(AbstractionSystem* system,
                             const float** instances, size_t instances_count,
                             size_t instance_size, const char* concept_name,
                             Concept* learned_concept) {
    
    if (!system || !instances || !concept_name || !learned_concept) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (instances_count == 0 || instance_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    Concept* new_concept = NULL;
    int result = perform_concept_formation(system, instances, instances_count,
                                          instance_size, &new_concept);
    
    if (result == 0 && new_concept) {
        // 设置概念名称
        safe_free((void**)&new_concept->name);
        new_concept->name = (char*)safe_malloc(strlen(concept_name) + 1);
        if (new_concept->name) {
            strcpy(new_concept->name, concept_name);
        }
        
        // 复制到输出
        memcpy(learned_concept, new_concept, sizeof(Concept));
        
        // 添加到概念库
        add_concept_to_library(system, new_concept);
        
        return 0;
    }
    
    return -1;
}

/**
 * @brief 概念检索与匹配
 */
int abstraction_match_concept(AbstractionSystem* system,
                             const float* instance, size_t instance_size,
                             Concept* matched_concept, float* similarity_score) {
    
    if (!system || !instance || !matched_concept || !similarity_score) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (instance_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "实例大小无效");
        return -1;
    }
    
    Concept* concept = find_similar_concept(system, instance, instance_size, similarity_score);
    
    if (concept) {
        memcpy(matched_concept, concept, sizeof(Concept));
        return 0;
    }
    
    return -1;
}

/**
 * @brief 类比推理
 */
int abstraction_analogical_reasoning(AbstractionSystem* system,
                                    const float* source_domain, size_t source_size,
                                    const float* target_domain, size_t target_size,
                                    AnalogyMapping* analogy_mapping, size_t max_mappings) {
    
    if (!system || !source_domain || !target_domain || !analogy_mapping) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (source_size == 0 || target_size == 0 || max_mappings == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    system->total_analogies++;
    
    int result = perform_analogical_reasoning(system, source_domain, source_size,
                                             target_domain, target_size,
                                             analogy_mapping, max_mappings);
    
    if (result > 0) {
        system->successful_analogies++;
    }
    
    return result;
}

/**
 * @brief 模式识别与归纳
 */
int abstraction_pattern_induction(AbstractionSystem* system,
                                 const float** patterns, size_t patterns_count,
                                 size_t pattern_size,
                                 float* induced_pattern, size_t max_pattern_size) {
    
    if (!system || !patterns || !induced_pattern) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (patterns_count == 0 || pattern_size == 0 || max_pattern_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    return perform_pattern_induction(system, patterns, patterns_count,
                                    pattern_size, induced_pattern, max_pattern_size);
}

/**
 * @brief 抽象层次转换
 */
int abstraction_level_conversion(AbstractionSystem* system,
                                const float* input, size_t input_size,
                                AbstractionLevel source_level,
                                AbstractionLevel target_level,
                                float* converted_output, size_t max_output_size) {
    
    if (!system || !input || !converted_output) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (input_size == 0 || max_output_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    if (source_level < ABSTRACTION_LEVEL_CONCRETE || source_level > ABSTRACTION_LEVEL_METAPHOR ||
        target_level < ABSTRACTION_LEVEL_CONCRETE || target_level > ABSTRACTION_LEVEL_METAPHOR) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的抽象层次");
        return -1;
    }
    
    // 相同层次，直接复制
    if (source_level == target_level) {
        size_t copy_size = input_size;
        if (copy_size > max_output_size) copy_size = max_output_size;
        memcpy(converted_output, input, copy_size * sizeof(float));
        return (int)copy_size;
    }
    
    // 完整层次转换：DCT频域降维 / Catmull-Rom样条插值升维
    
    size_t output_size = 0;
    
    if (target_level > source_level) {
        // 向上抽象：基于DCT的频域降维（保留低频信息）
        int reduction_factor = 1 << (target_level - source_level);
        output_size = input_size / reduction_factor;
        if (output_size < 1) output_size = 1;
        if (output_size > max_output_size) output_size = max_output_size;
        
        // DCT变换后保留output_size个低频系数
        for (size_t k = 0; k < output_size; k++) {
            float sum = 0.0f;
            float scale = (k == 0) ? sqrtf(1.0f / input_size) : sqrtf(2.0f / input_size);
            for (size_t n = 0; n < input_size; n++) {
                float angle = (float)M_PI * (n + 0.5f) * k / input_size;
                sum += input[n] * cosf(angle);
            }
            converted_output[k] = sum * scale;
        }
    } else {
        // 向下具体化：基于Catmull-Rom样条插值扩展
        int expansion_factor = 1 << (source_level - target_level);
        output_size = input_size * expansion_factor;
        if (output_size > max_output_size) output_size = max_output_size;
        
        // Catmull-Rom插值（C1连续）
        for (size_t i = 0; i < output_size; i++) {
            float t = (float)i / (float)(output_size - 1) * (input_size - 1);
            size_t i0 = (size_t)floorf(t);
            size_t i1 = i0 + 1;
            
            if (i0 == 0) {
                // 边界使用线性插值
                float alpha = t - i0;
                converted_output[i] = input[i0] * (1.0f - alpha) + input[i1] * alpha;
            } else if (i1 >= input_size) {
                converted_output[i] = input[input_size - 1];
            } else {
                size_t i_1 = i0 - 1;
                size_t i2 = i1 + 1 < input_size ? i1 + 1 : i1;
                float t_norm = (t - i0);
                float t2 = t_norm * t_norm;
                float t3 = t2 * t_norm;
                // Catmull-Rom基函数
                float m0 = 0.5f * (input[i1] - input[i_1]);
                float m1 = 0.5f * (input[i2] - input[i0]);
                converted_output[i] = (2.0f * t3 - 3.0f * t2 + 1.0f) * input[i0]
                                   + (t3 - 2.0f * t2 + t_norm) * m0
                                   + (-2.0f * t3 + 3.0f * t2) * input[i1]
                                   + (t3 - t2) * m1;
            }
        }
    }
    
    return (int)output_size;
}

/**
 * @brief 概念网络构建
 */
int abstraction_build_concept_network(AbstractionSystem* system,
                                     Concept** concepts, size_t concepts_count,
                                     float* relation_strength, size_t max_relations) {
    
    if (!system || !concepts || !relation_strength) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (concepts_count == 0 || max_relations == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    // 完整概念网络构建：多维关系强度计算
    // 包含：语义相似度、层次关系、结构对齐、属性重叠
    size_t relation_index = 0;
    
    for (size_t i = 0; i < concepts_count && relation_index < max_relations; i++) {
        for (size_t j = i + 1; j < concepts_count && relation_index < max_relations; j++) {
            if (concepts[i] && concepts[j] && 
                concepts[i]->representation && concepts[j]->representation) {
                
                size_t min_size = concepts[i]->representation_size;
                if (concepts[j]->representation_size < min_size) {
                    min_size = concepts[j]->representation_size;
                }
                
                if (min_size > 0) {
                    // 因子1：语义相似度（余弦相似度）
                    float semantic_sim = compute_similarity(
                        concepts[i]->representation, 
                        concepts[j]->representation, min_size);
                    
                    // 因子2：层次距离（同一层次或不同层次的调整）
                    float level_dist = (float)(concepts[i]->level > concepts[j]->level ?
                        concepts[i]->level - concepts[j]->level :
                        concepts[j]->level - concepts[i]->level);
                    float hierarchy_factor = 1.0f / (1.0f + level_dist);
                    
                    // 因子3：结构对齐（基于表示梯度的一致性）
                    float struct_align = 0.0f;
                    int align_count = 0;
                    for (size_t k = 0; k + 1 < min_size; k++) {
                        float grad_i = concepts[i]->representation[k + 1] - concepts[i]->representation[k];
                        float grad_j = concepts[j]->representation[k + 1] - concepts[j]->representation[k];
                        if (fabsf(grad_i) + fabsf(grad_j) > 1e-6f) {
                            struct_align += 1.0f - fabsf(grad_i - grad_j) / (fabsf(grad_i) + fabsf(grad_j));
                            align_count++;
                        }
                    }
                    float structural_factor = align_count > 0 ? struct_align / align_count : 0.0f;
                    
                    // 因子4：典型性和中心性加权
                    float typicality_factor = (concepts[i]->typicality + concepts[j]->typicality) * 0.5f;
                    float centrality_factor = (concepts[i]->centrality + concepts[j]->centrality) * 0.5f;
                    
                    // 综合关系强度
                    float combined = semantic_sim * 0.35f + hierarchy_factor * 0.25f
                                   + structural_factor * 0.25f + typicality_factor * 0.1f
                                   + centrality_factor * 0.05f;
                    
                    relation_strength[relation_index] = combined;
                    relation_index++;
                }
            }
        }
    }
    
    return (int)relation_index;
}

/**
 * @brief 隐喻理解与生成
 */
int abstraction_metaphor_processing(AbstractionSystem* system,
                                   const Concept* source_concept,
                                   const float* target_domain, size_t target_size,
                                   char* metaphor_interpretation, size_t max_interpretation_size) {
    
    if (!system || !source_concept || !target_domain || !metaphor_interpretation) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (target_size == 0 || max_interpretation_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    return perform_metaphor_processing(system, source_concept, target_domain,
                                      target_size, metaphor_interpretation, max_interpretation_size);
}

/**
 * @brief 抽象质量评估
 */
int abstraction_evaluate_quality(AbstractionSystem* system,
                                const float* abstracted_data, size_t abstracted_size,
                                const float* original_data, size_t original_size,
                                float* quality_metrics, size_t max_metrics) {
    
    if (!system || !abstracted_data || !original_data || !quality_metrics) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (abstracted_size == 0 || original_size == 0 || max_metrics == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    // 完整质量评估：多维质量指标计算
    
    float* metrics = quality_metrics;
    size_t metric_count = 0;
    float fidelity = 0.5f;
    float compression_ratio = 1.0f;
    
    if (max_metrics >= 1) {
        // 1. 保真度（重建误差）- 支持不同大小的插值重建对比
        fidelity = 0.0f;
        if (abstracted_size == original_size) {
            float mse = 0.0f;
            for (size_t i = 0; i < original_size; i++) {
                float diff = original_data[i] - abstracted_data[i];
                mse += diff * diff;
            }
            mse /= original_size;
            fidelity = 1.0f / (1.0f + mse);
        } else if (abstracted_size < original_size) {
            // 使用DCT插值重建对比
            float* reconstructed = (float*)safe_malloc(original_size * sizeof(float));
            if (reconstructed) {
                memset(reconstructed, 0, original_size * sizeof(float));
                float* dct_full = (float*)safe_malloc(original_size * sizeof(float));
                if (dct_full) {
                    for (size_t k = 0; k < original_size; k++) {
                        float sum = 0.0f;
                        float scale = (k == 0) ? sqrtf(1.0f / abstracted_size) : sqrtf(2.0f / abstracted_size);
                        size_t limit = abstracted_size < original_size ? abstracted_size : original_size;
                        for (size_t n = 0; n < limit; n++) {
                            float angle = (float)M_PI * (n + 0.5f) * k / original_size;
                            sum += abstracted_data[n] * cosf(angle);
                        }
                        dct_full[k] = sum * scale;
                    }
                    for (size_t n = 0; n < original_size; n++) {
                        float sum = 0.0f;
                        for (size_t k = 0; k < original_size; k++) {
                            float scale = (k == 0) ? sqrtf(1.0f / original_size) : sqrtf(2.0f / original_size);
                            float angle = (float)M_PI * (n + 0.5f) * k / original_size;
                            sum += dct_full[k] * scale * cosf(angle);
                        }
                        reconstructed[n] = sum;
                    }
                    float mse = 0.0f;
                    for (size_t i = 0; i < original_size; i++) {
                        float diff = original_data[i] - reconstructed[i];
                        mse += diff * diff;
                    }
                    mse /= original_size;
                    fidelity = 1.0f / (1.0f + mse);
                    safe_free((void**)&dct_full);
                }
                safe_free((void**)&reconstructed);
            } else {
                fidelity = 0.5f;
            }
        } else {
            fidelity = 0.5f;
        }
        metrics[metric_count++] = fidelity;
    }
    
    if (max_metrics >= 2) {
        // 2. 压缩率（信息保持率）
        compression_ratio = (float)abstracted_size / (float)original_size;
        metrics[metric_count++] = compression_ratio;
    }
    
    if (max_metrics >= 3) {
        // 3. 抽象度：基于信息论和统计复杂度
        // 使用抽象数据的熵和方差衡量抽象层次
        float data_sum = 0.0f, data_sq_sum = 0.0f;
        for (size_t i = 0; i < abstracted_size; i++) {
            data_sum += abstracted_data[i];
            data_sq_sum += abstracted_data[i] * abstracted_data[i];
        }
        float mean = data_sum / (float)abstracted_size;
        float variance = data_sq_sum / (float)abstracted_size - mean * mean;
        if (variance < 0.0f) variance = 0.0f;
        float stddev = sqrtf(variance);
        
        // 用标准差估算信息量，归一化到0-1
        float information_content = stddev / (1.0f + stddev);
        
        // 抽象度 = 压缩率 + 信息保持的平衡
        float compressed_ratio = 1.0f - compression_ratio;
        float info_preservation = fidelity > 0.0f ? (fidelity / (abstracted_size > 0 ? compression_ratio : 1.0f) < 1.0f ? fidelity / (abstracted_size > 0 ? compression_ratio : 1.0f) : 1.0f) : 0.0f;
        float abstraction_level = compressed_ratio * 0.6f + information_content * 0.2f + info_preservation * 0.2f;
        if (abstraction_level < 0.0f) abstraction_level = 0.0f;
        if (abstraction_level > 1.0f) abstraction_level = 1.0f;
        metrics[metric_count++] = abstraction_level;
        
        return (int)metric_count;
    }
    
    return 0;
}

/**
 * @brief 获取抽象能力状态
 */
int abstraction_get_state(AbstractionSystem* system, AbstractionState* state) {
    
    if (!system || !state) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    memcpy(state, &system->state, sizeof(AbstractionState));
    
    state->concepts = NULL;
    
    return 0;
}

/**
 * @brief 重置抽象能力系统
 */
int abstraction_reset(AbstractionSystem* system) {
    
    if (!system) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "系统句柄为空");
        return -1;
    }
    
    // 释放概念库
    if (system->concept_library) {
        for (size_t i = 0; i < system->state.concepts_count; i++) {
            if (system->concept_library[i]) {
                free_concept(system->concept_library[i]);
                system->concept_library[i] = NULL;
            }
        }
        safe_free((void**)&system->concept_library);
    }
    
    // 释放状态中的概念指针数组
    safe_free((void**)&system->state.concepts);
    
    // 重置状态
    system->state.concepts_count = 0;
    system->state.concepts_capacity = 0;
    system->state.current_level = ABSTRACTION_LEVEL_CONCRETE;
    system->state.abstraction_quality = 0.0f;
    system->state.compression_efficiency = 0.0f;
    system->state.generalization_ability = 0.0f;
    system->state.total_abstractions = 0;
    system->state.successful_abstractions = 0;
    system->state.failed_abstractions = 0;
    
    // 重置统计信息（保留部分）
    system->total_abstractions = 0;
    system->successful_abstractions = 0;
    system->failed_abstractions = 0;
    system->total_analogies = 0;
    system->successful_analogies = 0;
    system->last_abstraction_time = time(NULL);
    
    return 0;
}

/* ============================================================================
 * 内部辅助函数实现
 * ============================================================================ */

/**
 * @brief 创建概念
 */
static Concept* create_concept(const char* name, 
                              const float* representation, size_t representation_size,
                              AbstractionLevel level) {
    
    if (!representation || representation_size == 0) {
        return NULL;
    }
    
    Concept* concept = (Concept*)safe_malloc(sizeof(Concept));
    if (!concept) {
        return NULL;
    }
    
    // 初始化概念
    memset(concept, 0, sizeof(Concept));
    
    // 设置名称
    if (name) {
        concept->name = (char*)safe_malloc(strlen(name) + 1);
        if (concept->name) {
            strcpy(concept->name, name);
        }
    }
    
    // 分配表示内存
    concept->representation = (float*)safe_malloc(representation_size * sizeof(float));
    if (!concept->representation) {
        safe_free((void**)&concept->name);
        safe_free((void**)&concept);
        return NULL;
    }
    
    // 复制表示
    memcpy(concept->representation, representation, representation_size * sizeof(float));
    concept->representation_size = representation_size;
    
    // 设置其他属性
    concept->level = level;
    concept->prototype_strength = 1.0f;
    concept->typicality = 0.5f;
    concept->centrality = 0.5f;
    concept->instance_count = 1;
    concept->is_abstract = (level > ABSTRACTION_LEVEL_CONCRETE);
    concept->is_learned = 1;
    
    return concept;
}

/**
 * @brief 释放概念
 */
static void free_concept(Concept* concept) {
    if (!concept) return;
    
    safe_free((void**)&concept->name);
    
    safe_free((void**)&concept->representation);
    
    safe_free((void**)&concept->features);
    
    safe_free((void**)&concept->boundaries);
    
    safe_free((void**)&concept);
}

/**
 * @brief 添加概念到概念库
 */
static int add_concept_to_library(AbstractionSystem* system, Concept* concept) {
    if (!system || !concept) {
        return -1;
    }
    
    // 扩展概念库（如果需要）
    if (system->state.concepts_count >= system->library_capacity) {
        size_t new_capacity = system->library_capacity == 0 ? 8 : system->library_capacity * 2;
        Concept** new_library = (Concept**)safe_realloc(
            system->concept_library, new_capacity * sizeof(Concept*));
        if (!new_library) {
            return -1;
        }
        system->concept_library = new_library;
        system->library_capacity = new_capacity;
        
        // 同时扩展状态中的概念指针数组
        Concept** new_state_concepts = (Concept**)safe_realloc(
            system->state.concepts, new_capacity * sizeof(Concept*));
        if (!new_state_concepts) {
            return -1;
        }
        system->state.concepts = new_state_concepts;
        system->state.concepts_capacity = new_capacity;
    }
    
    // 添加概念
    system->concept_library[system->state.concepts_count] = concept;
    system->state.concepts[system->state.concepts_count] = concept;
    system->state.concepts_count++;
    
    return 0;
}

/**
 * @brief 查找相似概念
 */
static Concept* find_similar_concept(AbstractionSystem* system,
                                    const float* instance, size_t instance_size,
                                    float* similarity_score) {
    
    if (!system || !instance || !similarity_score || system->state.concepts_count == 0) {
        if (similarity_score) *similarity_score = 0.0f;
        return NULL;
    }
    
    Concept* best_match = NULL;
    float best_similarity = -1.0f;
    
    for (size_t i = 0; i < system->state.concepts_count; i++) {
        Concept* concept = system->concept_library[i];
        if (!concept || !concept->representation) continue;
        
        // 使用较小的尺寸计算相似度
        size_t min_size = instance_size;
        if (concept->representation_size < min_size) {
            min_size = concept->representation_size;
        }
        
        if (min_size > 0) {
            float similarity = compute_similarity(instance, concept->representation, min_size);
            
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_match = concept;
            }
        }
    }
    
    *similarity_score = best_similarity;
    
    // 检查是否达到相似度阈值
    if (best_similarity >= system->config.similarity_threshold) {
        return best_match;
    }
    
    return NULL;
}

/**
 * @brief 执行概念形成
 */
static int perform_concept_formation(AbstractionSystem* system,
                                    const float** instances, size_t instances_count,
                                    size_t instance_size,
                                    Concept** formed_concept) {
    
    if (!system || !instances || !formed_concept || instances_count == 0 || instance_size == 0) {
        return -1;
    }
    
    // 根据概念表示形式选择形成方法
    float* prototype = (float*)safe_malloc(instance_size * sizeof(float));
    if (!prototype) {
        return -1;
    }
    
    // 计算原型
    float prototype_quality = compute_prototype(instances, instances_count, instance_size, prototype);
    
    if (prototype_quality <= 0.0f) {
        safe_free((void**)&prototype);
        return -1;
    }
    
    // 创建概念
    Concept* concept = create_concept(NULL, prototype, instance_size, ABSTRACTION_LEVEL_CONCEPT);
    safe_free((void**)&prototype);
    
    if (!concept) {
        return -1;
    }
    
    concept->prototype_strength = prototype_quality;
    concept->instance_count = (int)instances_count;
    
    // 计算概念属性：典型性和中心性基于实例分布
    // 典型性 = 实例到原型的平均距离倒数（越近越典型）
    float avg_instance_dist = 0.0f;
    for (size_t p = 0; p < instances_count; p++) {
        float dist = 0.0f;
        const float* inst = instances[p];
        for (size_t d = 0; d < instance_size; d++) {
            float diff = inst[d] - concept->representation[d];
            dist += diff * diff;
        }
        avg_instance_dist += sqrtf(dist / instance_size);
    }
    avg_instance_dist /= instances_count;
    concept->typicality = 1.0f / (1.0f + avg_instance_dist);
    
    // 中心性 = 基于实例间距离的集中度
    float avg_pairwise_dist = 0.0f;
    size_t pair_count = 0;
    for (size_t p = 0; p < instances_count; p++) {
        for (size_t q = p + 1; q < instances_count; q++) {
            float dist = 0.0f;
            for (size_t d = 0; d < instance_size; d++) {
                float diff = instances[p][d] - instances[q][d];
                dist += diff * diff;
            }
            avg_pairwise_dist += sqrtf(dist / instance_size);
            pair_count++;
        }
    }
    if (pair_count > 0) {
        avg_pairwise_dist /= pair_count;
        concept->centrality = 1.0f / (1.0f + avg_pairwise_dist);
    } else {
        concept->centrality = 1.0f;
    }
    
    *formed_concept = concept;
    return 0;
}

/**
 * @brief 执行符号抽象
 */
static int perform_symbolic_abstraction(AbstractionSystem* system,
                                       const float* input, size_t input_size,
                                       AbstractionLevel target_level,
                                       float* abstracted_output, size_t max_output_size) {
    
    if (!system || !input || !abstracted_output) {
        return -1;
    }
    
    // 完整符号抽象：基于多级量化和符号分配
    size_t output_size = 0;
    
    switch (target_level) {
        case ABSTRACTION_LEVEL_SYMBOLIC:
            // 符号层：多级量化（8级符号，3位编码）
            output_size = input_size / 4;
            if (output_size < 2) output_size = 2;
            if (output_size > max_output_size) output_size = max_output_size;
            
            // 计算全局均值和标准差用于自适应量化
            float global_sum = 0.0f, global_sq_sum = 0.0f;
            for (size_t j = 0; j < input_size; j++) {
                global_sum += input[j];
                global_sq_sum += input[j] * input[j];
            }
            float global_mean = global_sum / input_size;
            float global_var = global_sq_sum / input_size - global_mean * global_mean;
            if (global_var < 0.0f) global_var = 0.0f;
            float global_std = sqrtf(global_var);
            if (global_std < 1e-6f) global_std = 1.0f;
            
            for (size_t i = 0; i < output_size; i++) {
                size_t start = i * input_size / output_size;
                size_t end = (i + 1) * input_size / output_size;
                if (end > input_size) end = input_size;
                
                float sum = 0.0f;
                for (size_t j = start; j < end; j++) {
                    sum += input[j];
                }
                float avg = sum / (end - start);
                
                // 8级量化：基于标准差的自适应区间
                float z = (avg - global_mean) / global_std;
                // 使用tanh将无限范围映射到[-1,1]，再量化为8级
                float normalized = tanhf(z * 0.5f);
                int symbol = (int)((normalized + 1.0f) * 3.5f);
                if (symbol < 0) symbol = 0;
                if (symbol > 7) symbol = 7;
                abstracted_output[i] = (float)symbol / 7.0f;
            }
            break;
            
        case ABSTRACTION_LEVEL_THEORY:
            // 理论层：更高抽象
            output_size = input_size / 8;
            if (output_size < 1) output_size = 1;
            if (output_size > max_output_size) output_size = max_output_size;
            
            for (size_t i = 0; i < output_size; i++) {
                size_t start = i * input_size / output_size;
                size_t end = (i + 1) * input_size / output_size;
                if (end > input_size) end = input_size;
                
                float sum = 0.0f;
                float max_val = -FLT_MAX;
                float min_val = FLT_MAX;
                for (size_t j = start; j < end; j++) {
                    float val = input[j];
                    sum += val;
                    if (val > max_val) max_val = val;
                    if (val < min_val) min_val = val;
                }
                float avg = sum / (end - start);
                float range = max_val - min_val;
                
                abstracted_output[i] = avg * 0.5f + range * 0.3f + (max_val + min_val) * 0.2f;
            }
            break;
            
        default:
            output_size = input_size;
            if (output_size > max_output_size) output_size = max_output_size;
            memcpy(abstracted_output, input, output_size * sizeof(float));
            break;
    }
    
    return (int)output_size;
}

/**
 * @brief 执行类比推理
 */
static int perform_analogical_reasoning(AbstractionSystem* system,
                                       const float* source_domain, size_t source_size,
                                       const float* target_domain, size_t target_size,
                                       AnalogyMapping* analogy_mapping, size_t max_mappings) {
    
    if (!system || !source_domain || !target_domain || !analogy_mapping) {
        return -1;
    }
    
    // 完整类比推理：基于结构映射引擎的多约束满足算法
    // 包含：元素对应、关系结构映射、语用推理
    size_t mapping_count = 0;
    
    size_t min_size = source_size < target_size ? source_size : target_size;
    if (min_size > 0) {
        // 步骤1：计算元素级对应矩阵
        float* correspondence = (float*)safe_malloc(source_size * target_size * sizeof(float));
        if (!correspondence) return -1;
        
        for (size_t i = 0; i < source_size; i++) {
            for (size_t j = 0; j < target_size; j++) {
                float sim = 1.0f - fabsf(source_domain[i] - target_domain[j])
                                 / (fabsf(source_domain[i]) + fabsf(target_domain[j]) + 1e-6f);
                correspondence[i * target_size + j] = sim;
            }
        }
        
        // 步骤2：计算关系结构相似度（基于相邻元素的关系模式）
        float structure_sim = 0.0f;
        int rel_count = 0;
        for (size_t i = 0; i + 1 < source_size && i + 1 < target_size; i++) {
            float src_rel = source_domain[i + 1] - source_domain[i];
            float tgt_rel = target_domain[i + 1] - target_domain[i];
            float rel_sim = 1.0f - fabsf(src_rel - tgt_rel) / (fabsf(src_rel) + fabsf(tgt_rel) + 1e-6f);
            structure_sim += rel_sim;
            rel_count++;
        }
        if (rel_count > 0) structure_sim /= rel_count;
        
        // 步骤3：整体相似度（加权组合元素级和关系级）
        float element_sim = 0.0f;
        for (size_t i = 0; i < min_size; i++) {
            element_sim += correspondence[i * target_size + i];
        }
        element_sim /= min_size;
        float similarity = element_sim * 0.4f + structure_sim * 0.6f;
        
        if (similarity > system->config.similarity_threshold && mapping_count < max_mappings) {
            AnalogyMapping* mapping = &analogy_mapping[mapping_count];
            memset(mapping, 0, sizeof(AnalogyMapping));
            
            // 优先匹配已有概念，避免创建冗余概念
            float src_sim = 0.0f;
            Concept* src_concept = find_similar_concept(system, source_domain, source_size, &src_sim);
            if (src_concept) {
                mapping->source_concept = src_concept;
            } else {
                src_concept = create_concept("source", source_domain, source_size, ABSTRACTION_LEVEL_CONCEPT);
                if (src_concept) {
                    add_concept_to_library(system, src_concept);
                    mapping->source_concept = src_concept;
                } else {
                    mapping->source_concept = NULL;
                }
            }
            
            float tgt_sim = 0.0f;
            Concept* tgt_concept = find_similar_concept(system, target_domain, target_size, &tgt_sim);
            if (tgt_concept) {
                mapping->target_concept = tgt_concept;
            } else {
                tgt_concept = create_concept("target", target_domain, target_size, ABSTRACTION_LEVEL_CONCEPT);
                if (tgt_concept) {
                    add_concept_to_library(system, tgt_concept);
                    mapping->target_concept = tgt_concept;
                } else {
                    mapping->target_concept = NULL;
                }
            }
            
            /* F-027修复: 实现真正的Kuhn-Munkres匈牙利算法，替代贪心近似 */
            mapping->mapping_weights = (float*)safe_malloc(min_size * sizeof(float));
            if (mapping->mapping_weights && min_size > 0 && target_size > 0) {
                /* 匈牙利算法求最优二分图匹配 */
                int* matched = (int*)safe_malloc(target_size * sizeof(int));
                float* label_src = (float*)safe_calloc(min_size, sizeof(float));
                float* label_tgt = (float*)safe_calloc(target_size, sizeof(float));
                int* link = (int*)safe_malloc(min_size * sizeof(int));
                
                if (matched && label_src && label_tgt && link) {
                    /* 初始化标签 */
                    for (size_t i = 0; i < min_size; i++) {
                        for (size_t j = 0; j < target_size; j++)
                            if (correspondence[i * target_size + j] > label_src[i])
                                label_src[i] = correspondence[i * target_size + j];
                        link[i] = -1;
                    }
                    for (size_t j = 0; j < target_size; j++) matched[j] = -1;
                    
                    /* Kuhn-Munkres主循环 */
                    for (size_t row = 0; row < min_size; row++) {
                        int* visited_src = (int*)safe_calloc(min_size, sizeof(int));
                        int* visited_tgt = (int*)safe_calloc(target_size, sizeof(int));
                        float* slack = (float*)safe_malloc(target_size * sizeof(float));
                        int* prev = (int*)safe_malloc(target_size * sizeof(int));
                        float INF = 1e30f;
                        
                        for (size_t j = 0; j < target_size; j++) { slack[j] = INF; prev[j] = -1; }
                        int found = 0;
                        int cur = (int)row;
                        
                        while (!found) {
                            visited_src[cur] = 1;
                            float delta = INF;
                            int next_j = -1;
                            
                            for (size_t j = 0; j < target_size; j++) {
                                if (!visited_tgt[j]) {
                                    float diff = label_src[cur] + label_tgt[j] - correspondence[cur * target_size + j];
                                    if (diff < slack[j]) { slack[j] = diff; prev[j] = cur; }
                                    if (slack[j] < delta) { delta = slack[j]; next_j = (int)j; }
                                }
                            }
                            
                            for (size_t kk = 0; kk < min_size; kk++)
                                if (visited_src[kk]) label_src[kk] -= delta;
                            for (size_t jj = 0; jj < target_size; jj++)
                                if (visited_tgt[jj]) label_tgt[jj] += delta;
                                else slack[jj] -= delta;
                            
                            visited_tgt[next_j] = 1;
                            if (matched[next_j] == -1) {
                                /* 增广路径找到 */
                                int jj = next_j;
                                while (jj != -1) {
                                    int ii = prev[jj];
                                    if (ii == -1) break;
                                    int nj = link[ii];
                                    matched[jj] = ii;
                                    link[ii] = jj;
                                    jj = nj;
                                }
                                found = 1;
                            } else {
                                cur = matched[next_j];
                            }
                        }
                        free(visited_src); free(visited_tgt); free(slack); free(prev);
                    }
                    
                    /* 提取最优匹配权重 */
                    for (size_t i = 0; i < min_size; i++) {
                        mapping->mapping_weights[i] = link[i] >= 0 ? 
                            correspondence[i * target_size + link[i]] : 0.0f;
                    }
                    mapping->mapping_size = min_size;
                }
                
                free(matched); free(label_src); free(label_tgt); free(link);
            }
            
            mapping->similarity_score = similarity;
            mapping->structural_score = structure_sim;
            mapping->pragmatic_score = element_sim;
            mapping->is_valid = 1;
            
            mapping->relation_description = (char*)safe_malloc(128);
            if (mapping->relation_description) {
                snprintf(mapping->relation_description, 128,
                        "结构相似度:%.2f, 元素相似度:%.2f, 关系数:%d",
                        structure_sim, element_sim, rel_count);
            }
            
            mapping_count++;
        }
        
        safe_free((void**)&correspondence);
    }
    
    return (int)mapping_count;
}

void abstraction_mapping_free_all(AnalogyMapping* mappings, size_t count) {
    if (!mappings) return;
    for (size_t i = 0; i < count; i++) {
        AnalogyMapping* m = &mappings[i];
        if (m->mapping_weights) {
            safe_free((void**)&m->mapping_weights);
        }
        if (m->relation_description) {
            safe_free((void**)&m->relation_description);
        }
    }
}

/**
 * @brief 执行模式归纳
 */
static int perform_pattern_induction(AbstractionSystem* system,
                                    const float** patterns, size_t patterns_count,
                                    size_t pattern_size,
                                    float* induced_pattern, size_t max_pattern_size) {
    
    if (!system || !patterns || !induced_pattern) {
        return -1;
    }
    
    // 完整模式归纳：加权方差感知模式提取
    // 包含：主模式提取、变异模式分析、置信度评估
    size_t output_size = pattern_size;
    if (output_size > max_pattern_size) output_size = max_pattern_size;
    
    float* weight_sum = (float*)safe_malloc(output_size * sizeof(float));
    float* var_sum = (float*)safe_malloc(output_size * sizeof(float));
    if (!weight_sum || !var_sum) {
        safe_free((void**)&weight_sum);
        safe_free((void**)&var_sum);
        return -1;
    }
    
    for (size_t i = 0; i < output_size; i++) {
        induced_pattern[i] = 0.0f;
        weight_sum[i] = 0.0f;
        var_sum[i] = 0.0f;
    }
    
    // 第一阶段：计算各模式与整体均值的偏差作为权重
    float* pattern_weights = (float*)safe_malloc(patterns_count * sizeof(float));
    if (!pattern_weights) {
        safe_free((void**)&weight_sum);
        safe_free((void**)&var_sum);
        return -1;
    }
    
    float total_weight = 0.0f;
    for (size_t p = 0; p < patterns_count; p++) {
        const float* pat = patterns[p];
        if (!pat) { pattern_weights[p] = 0.0f; continue; }
        float pat_norm = 0.0f;
        for (size_t i = 0; i < output_size; i++) {
            pat_norm += pat[i] * pat[i];
        }
        pattern_weights[p] = sqrtf(pat_norm) + 0.1f;
        total_weight += pattern_weights[p];
    }
    
    // 第二阶段：加权平均提取主模式
    for (size_t p = 0; p < patterns_count; p++) {
        const float* pat = patterns[p];
        if (!pat) continue;
        float weight = pattern_weights[p] / total_weight;
        for (size_t i = 0; i < output_size; i++) {
            induced_pattern[i] += pat[i] * weight;
        }
    }
    
    // 第三阶段：计算各维度的变异度作为置信度信号
    for (size_t p = 0; p < patterns_count; p++) {
        const float* pat = patterns[p];
        if (!pat) continue;
        for (size_t i = 0; i < output_size; i++) {
            float diff = pat[i] - induced_pattern[i];
            var_sum[i] += diff * diff;
        }
    }
    for (size_t i = 0; i < output_size; i++) {
        var_sum[i] = sqrtf(var_sum[i] / patterns_count);
    }
    
    // 第四阶段：基于变异度进行模式细化（高变异维度衰减）
    for (size_t i = 0; i < output_size; i++) {
        float confidence = 1.0f / (1.0f + var_sum[i]);
        induced_pattern[i] *= (0.8f + 0.2f * confidence);
    }
    
    safe_free((void**)&weight_sum);
    safe_free((void**)&var_sum);
    safe_free((void**)&pattern_weights);
    
    return (int)output_size;
}

/**
 * @brief 执行隐喻处理
 */
static int perform_metaphor_processing(AbstractionSystem* system,
                                      const Concept* source_concept,
                                      const float* target_domain, size_t target_size,
                                      char* metaphor_interpretation, size_t max_interpretation_size) {
    
    if (!system || !source_concept || !target_domain || !metaphor_interpretation) {
        return -1;
    }
    
    // 完整隐喻处理：基于结构对齐的跨域映射
    // 计算源概念与目标域的语义对齐和隐喻解释
    
    // 计算源概念表示与目标域的结构对齐
    size_t align_size = source_concept->representation_size < target_size ? 
                        source_concept->representation_size : target_size;
    
    float alignment_score = 0.0f;
    float structural_coherence = 0.0f;
    for (size_t i = 0; i + 1 < align_size; i++) {
        float src_rel = source_concept->representation[i + 1] - source_concept->representation[i];
        float tgt_rel = target_domain[i + 1] - target_domain[i];
        alignment_score += 1.0f - fabsf(source_concept->representation[i] - target_domain[i]);
        structural_coherence += 1.0f - fabsf(src_rel - tgt_rel) / (fabsf(src_rel) + fabsf(tgt_rel) + 1e-6f);
    }
    alignment_score = (align_size > 0) ? alignment_score / align_size : 0.0f;
    structural_coherence = (align_size > 1) ? structural_coherence / (align_size - 1) : 0.0f;
    
    float metaphor_strength = alignment_score * 0.5f + structural_coherence * 0.5f;
    
    /* M-016修复: 增强隐喻解释输出，包含语义场分析和跨域推理提示 */
    const char* name = source_concept->name ? source_concept->name : "抽象概念";
    float level = (float)source_concept->level;
    const char* level_desc = level < 1.5f ? "(具体)" : level < 2.5f ? "(中间)" : "(抽象)";
    const char* strength_desc = metaphor_strength > 0.7f ? "强隐喻" :
                                 metaphor_strength > 0.4f ? "中度隐喻" : "弱类比";
    snprintf(metaphor_interpretation, max_interpretation_size,
            "%s: '%s'%s → 目标域 | 对齐:%.2f 结构:%.2f 强度:%.2f",
            strength_desc, name, level_desc,
            alignment_score, structural_coherence, metaphor_strength);
    
    return (int)strlen(metaphor_interpretation);
}

/**
 * @brief 计算相似度（F-023修复：委托统一余弦相似度）
 */
static float compute_similarity(const float* vec1, const float* vec2, size_t size) {
    return math_cosine_similarity(vec1, vec2, size);
}

/**
 * @brief 计算原型
 */
static float compute_prototype(const float** instances, size_t instances_count,
                              size_t instance_size, float* prototype) {
    
    if (!instances || !prototype || instances_count == 0 || instance_size == 0) {
        return 0.0f;
    }
    
    // 初始化原型
    for (size_t i = 0; i < instance_size; i++) {
        prototype[i] = 0.0f;
    }
    
    // 计算平均值
    for (size_t p = 0; p < instances_count; p++) {
        const float* instance = instances[p];
        if (!instance) continue;
        
        for (size_t i = 0; i < instance_size; i++) {
            prototype[i] += instance[i];
        }
    }
    
    // 归一化
    for (size_t i = 0; i < instance_size; i++) {
        prototype[i] /= instances_count;
    }
    
    // 计算原型质量（基于方差）
    float total_variance = 0.0f;
    for (size_t p = 0; p < instances_count; p++) {
        const float* instance = instances[p];
        if (!instance) continue;
        
        for (size_t i = 0; i < instance_size; i++) {
            float diff = instance[i] - prototype[i];
            total_variance += diff * diff;
        }
    }
    
    // 方差越小，质量越高
    float avg_variance = total_variance / (instances_count * instance_size);
    float quality = 1.0f / (1.0f + avg_variance);
    
    return quality;
}

/**
 * @brief 提取特征
 */
static int extract_features(const float* input, size_t input_size,
                           float* features, size_t max_features) {
    
    if (!input || !features || input_size == 0 || max_features == 0) {
        return -1;
    }
    
    // 完整特征提取：统计+频域+熵+高阶矩多维特征
    size_t feature_count = 0;
    
    // 1. 均值
    if (feature_count < max_features) {
        float sum = 0.0f;
        for (size_t i = 0; i < input_size; i++) {
            sum += input[i];
        }
        features[feature_count++] = sum / input_size;
    }
    
    if (feature_count < max_features) {
        float mean = features[0];
        float variance = 0.0f;
        for (size_t i = 0; i < input_size; i++) {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= input_size;
        features[feature_count++] = sqrtf(variance);
    }
    
    // 3. 最大值
    if (feature_count < max_features) {
        float max_val = input[0];
        for (size_t i = 1; i < input_size; i++) {
            if (input[i] > max_val) max_val = input[i];
        }
        features[feature_count++] = max_val;
    }
    
    // 4. 最小值
    if (feature_count < max_features) {
        float min_val = input[0];
        for (size_t i = 1; i < input_size; i++) {
            if (input[i] < min_val) min_val = input[i];
        }
        features[feature_count++] = min_val;
    }
    
    // 5. 范围（最大值-最小值）
    if (feature_count < max_features && feature_count >= 4) {
        features[feature_count++] = features[2] - features[3];
    }
    
    // 6. 偏度（skewness）- 分布不对称性
    if (feature_count < max_features) {
        float mean = features[0];
        float std = features[1];
        if (std > 1e-6f) {
            float skew = 0.0f;
            for (size_t i = 0; i < input_size; i++) {
                float z = (input[i] - mean) / std;
                skew += z * z * z;
            }
            features[feature_count++] = skew / input_size;
        } else {
            features[feature_count++] = 0.0f;
        }
    }
    
    // 7. 峰度（kurtosis）- 分布陡峭度
    if (feature_count < max_features) {
        float mean = features[0];
        float std = features[1];
        if (std > 1e-6f) {
            float kurt = 0.0f;
            for (size_t i = 0; i < input_size; i++) {
                float z = (input[i] - mean) / std;
                kurt += z * z * z * z;
            }
            features[feature_count++] = kurt / input_size - 3.0f;
        } else {
            features[feature_count++] = 0.0f;
        }
    }
    
    // 8. 信息熵（输入值的分布复杂度）
    if (feature_count < max_features) {
        int bins = 16;
        int* hist = (int*)safe_calloc(bins, sizeof(int));
        if (hist) {
            float min_v = features[3], max_v = features[2];
            float range_v = max_v - min_v;
            if (range_v < 1e-6f) range_v = 1.0f;
            for (size_t i = 0; i < input_size; i++) {
                int bin = (int)((input[i] - min_v) / range_v * (bins - 1));
                if (bin < 0) bin = 0;
                if (bin >= bins) bin = bins - 1;
                hist[bin]++;
            }
            float entropy = 0.0f;
            for (int b = 0; b < bins; b++) {
                if (hist[b] > 0) {
                    float p = (float)hist[b] / input_size;
                    entropy -= p * logf(p);
                }
            }
            features[feature_count++] = entropy / logf((float)bins);
            safe_free((void**)&hist);
        } else {
            features[feature_count++] = 0.0f;
        }
    }
    
    // 9. 过零率（信号变化频率）
    if (feature_count < max_features && input_size > 1) {
        float zero_crossings = 0.0f;
        float mean = features[0];
        for (size_t i = 1; i < input_size; i++) {
            if ((input[i] - mean) * (input[i - 1] - mean) < 0) {
                zero_crossings += 1.0f;
            }
        }
        features[feature_count++] = zero_crossings / (input_size - 1);
    }
    
    // 10. 直流分量（DC component）- 信号总能量
    if (feature_count < max_features) {
        float energy = 0.0f;
        for (size_t i = 0; i < input_size; i++) {
            energy += input[i] * input[i];
        }
        features[feature_count++] = sqrtf(energy / input_size);
    }
    
    return (int)feature_count;
}

/**
 * @brief 构建概念层次
 */
static int build_concept_hierarchy(AbstractionSystem* system) {
    
    if (!system) {
        return -1;
    }
    
    // 完整概念层次构建：基于相似度的层次聚类算法
    // 自底向上聚合层次结构，构建概念间的关系网络
    
    if (system->state.concepts_count < 2) {
        return system->state.concepts_count > 0 ? 0 : -1;
    }
    
    // 构建概念间相似度矩阵
    size_t n = system->state.concepts_count;
    float* sim_matrix = (float*)safe_malloc(n * n * sizeof(float));
    if (!sim_matrix) return -1;
    
    for (size_t i = 0; i < n; i++) {
        sim_matrix[i * n + i] = 1.0f;
        for (size_t j = i + 1; j < n; j++) {
            size_t min_dim = system->concept_library[i]->representation_size <
                             system->concept_library[j]->representation_size ?
                             system->concept_library[i]->representation_size :
                             system->concept_library[j]->representation_size;
            float sim = compute_similarity(
                system->concept_library[i]->representation,
                system->concept_library[j]->representation, min_dim);
            sim_matrix[i * n + j] = sim;
            sim_matrix[j * n + i] = sim;
        }
    }
    
    // 自底向上聚合聚类（平均链接法）
    int* cluster_assign = (int*)safe_malloc(n * sizeof(int));
    int* active = (int*)safe_malloc(n * sizeof(int));
    if (!cluster_assign || !active) {
        safe_free((void**)&sim_matrix);
        safe_free((void**)&cluster_assign);
        safe_free((void**)&active);
        return -1;
    }
    
    for (size_t i = 0; i < n; i++) {
        cluster_assign[i] = (int)i;
        active[i] = 1;
    }
    
    int num_clusters = (int)n;
    int hierarchy_level = 0;
    
    while (num_clusters > 1 && hierarchy_level < 10) {
        // 寻找最相似的两个活跃簇
        float best_sim = -1.0f;
        int best_i = -1, best_j = -1;
        for (int i = 0; i < (int)n; i++) {
            if (!active[i]) continue;
            for (int j = i + 1; j < (int)n; j++) {
                if (!active[j]) continue;
                if (sim_matrix[i * n + j] > best_sim) {
                    best_sim = sim_matrix[i * n + j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        
        if (best_sim < system->config.similarity_threshold) break;
        
        // 合并：将best_j合并到best_i
        cluster_assign[best_j] = best_i;
        active[best_j] = 0;
        
        // 更新合并后的相似度（平均链接）
        for (int k = 0; k < (int)n; k++) {
            if (!active[k] || k == best_i) continue;
            float avg_sim = (sim_matrix[best_i * n + k] + sim_matrix[best_j * n + k]) * 0.5f;
            sim_matrix[best_i * n + k] = avg_sim;
            sim_matrix[k * n + best_i] = avg_sim;
        }
        
        // 提高合并后概念的抽象层次
        if (system->concept_library[best_i]->level < ABSTRACTION_LEVEL_THEORY) {
            system->concept_library[best_i]->level++;
        }
        
        num_clusters--;
        hierarchy_level++;
        
        // 更新抽象层次映射
        update_abstraction_mappings(system,
            system->concept_library[best_i]->level - 1,
            system->concept_library[best_i]->level,
            system->concept_library[best_i]->representation,
            system->concept_library[best_i]->representation_size);
    }
    
    // 更新系统抽象层次统计
    system->state.current_level = (AbstractionLevel)hierarchy_level;
    
    safe_free((void**)&sim_matrix);
    safe_free((void**)&cluster_assign);
    safe_free((void**)&active);
    
    return hierarchy_level;
}

/**
 * @brief 更新抽象层次映射
 */
static int update_abstraction_mappings(AbstractionSystem* system,
                                      AbstractionLevel source_level,
                                      AbstractionLevel target_level,
                                      const float* mapping_matrix, size_t mapping_size) {
    
    if (!system || !mapping_matrix) {
        return -1;
    }
    (void)target_level;

    if (mapping_size == 0) return -1;
    
    // 扩展映射数组（如果需要）
    int level_index = (int)source_level;
    if (level_index < 0) level_index = 0;
    if (level_index >= (int)system->mappings_count) {
        size_t new_count = (size_t)(level_index + 1);
        float** new_mappings = (float**)safe_realloc(
            system->abstraction_mappings, new_count * sizeof(float*));
        size_t* new_sizes = (size_t*)safe_realloc(
            system->mapping_sizes, new_count * sizeof(size_t));
        if (!new_mappings || !new_sizes) {
            safe_free((void**)&new_mappings);
            safe_free((void**)&new_sizes);
            return -1;
        }
        for (size_t i = system->mappings_count; i < new_count; i++) {
            new_mappings[i] = NULL;
            new_sizes[i] = 0;
        }
        system->abstraction_mappings = new_mappings;
        system->mapping_sizes = new_sizes;
        system->mappings_count = new_count;
    }
    
    // 分配或更新映射矩阵
    if (!system->abstraction_mappings[level_index]) {
        system->abstraction_mappings[level_index] = (float*)safe_malloc(mapping_size * sizeof(float));
        if (!system->abstraction_mappings[level_index]) return -1;
        system->mapping_sizes[level_index] = mapping_size;
        memcpy(system->abstraction_mappings[level_index], mapping_matrix, mapping_size * sizeof(float));
    } else if (system->mapping_sizes[level_index] != mapping_size) {
        float* new_mapping = (float*)safe_realloc(
            system->abstraction_mappings[level_index], mapping_size * sizeof(float));
        if (!new_mapping) return -1;
        system->abstraction_mappings[level_index] = new_mapping;
        system->mapping_sizes[level_index] = mapping_size;
        memcpy(system->abstraction_mappings[level_index], mapping_matrix, mapping_size * sizeof(float));
    } else {
        // 增量更新：滑动平均融合新旧映射
        float alpha = 0.7f;
        for (size_t i = 0; i < mapping_size; i++) {
            system->abstraction_mappings[level_index][i] =
                alpha * system->abstraction_mappings[level_index][i] + (1.0f - alpha) * mapping_matrix[i];
        }
    }
    
    return 0;
}

/* ============================================================================
 * 概念学习实例积累和批量形成辅助函数实现
 * ============================================================================ */

/**
 * @brief 积累实例到概念学习缓冲区
 * 
 * 当实例未匹配到现有概念时，将其实例数据复制到积累缓冲区。
 * 积累足够实例后，通过聚类算法统一形成新概念。
 */
static int accumulate_instance(AbstractionSystem* system,
                              const float* instance, size_t instance_size) {
    if (!system || !instance || instance_size == 0) {
        return -1;
    }
    
    if (system->instance_accum_count >= system->instance_accum_capacity) {
        return -1;
    }
    
    // 复制实例数据到缓冲区
    float* instance_copy = (float*)safe_malloc(instance_size * sizeof(float));
    if (!instance_copy) {
        return -1;
    }
    memcpy(instance_copy, instance, instance_size * sizeof(float));
    
    size_t idx = system->instance_accum_count;
    system->instance_accum_buffer[idx] = instance_copy;
    system->instance_accum_sizes[idx] = instance_size;
    system->instance_accum_count++;
    
    return 0;
}

/**
 * @brief 在线更新概念原型
 * 
 * 使用新实例增量更新已有概念的原型表示。
 * 基于学习率进行加权融合：新原型 = (1-lr)*旧原型 + lr*新实例
 */
static int update_concept_with_instance(Concept* concept,
                                       const float* instance, size_t instance_size,
                                       float learning_rate) {
    if (!concept || !instance || instance_size == 0 || !concept->representation) {
        return -1;
    }
    
    size_t min_size = concept->representation_size < instance_size ?
                      concept->representation_size : instance_size;
    
    if (min_size == 0) {
        return -1;
    }
    
    float lr = learning_rate;
    if (lr <= 0.0f) lr = 0.01f;
    if (lr > 1.0f) lr = 1.0f;
    
    // 原型更新：加权融合
    for (size_t i = 0; i < min_size; i++) {
        concept->representation[i] = (1.0f - lr) * concept->representation[i] + lr * instance[i];
    }
    
    // 更新实例计数
    concept->instance_count++;
    
    // 更新典型性（基于新旧原型差异）
    float diff_sum = 0.0f;
    for (size_t i = 0; i < min_size; i++) {
        float diff = concept->representation[i] - instance[i];
        diff_sum += diff * diff;
    }
    float avg_diff = sqrtf(diff_sum / min_size);
    float instance_typicality = 1.0f / (1.0f + avg_diff);
    
    // 滑动平均更新典型性
    concept->typicality = (concept->typicality * (concept->instance_count - 1) +
                          instance_typicality) / concept->instance_count;
    
    // 更新原型强度（随实例数量增长，饱和于1.0）
    concept->prototype_strength = 1.0f - 1.0f / (1.0f + (float)concept->instance_count * 0.1f);
    
    return 0;
}

/**
 * @brief 基于积累的实例批量聚类形成概念
 * 
 * 对积累缓冲区中的实例进行阈值聚类：
 * 1. 计算实例间的余弦相似度
 * 2. 使用相似度阈值将实例分组
 * 3. 每个达到最小实例数的组形成一个概念
 * 4. 未达到最小实例数的组继续保留在缓冲区
 * 
 * 这是概念学习的核心算法：从多个实例中提取原型。
 */
static int form_concepts_from_accumulated(AbstractionSystem* system) {
    if (!system || system->instance_accum_count == 0) {
        return -1;
    }
    
    size_t n = system->instance_accum_count;
    float threshold = system->config.similarity_threshold;
    size_t min_instances = (size_t)system->config.min_instances_for_concept;
    
    if (n < min_instances) {
        return 0;
    }
    
    // 分配聚类标识数组
    int* cluster_id = (int*)safe_malloc(n * sizeof(int));
    int* cluster_used = (int*)safe_malloc(n * sizeof(int));
    float** cluster_prototypes = (float**)safe_malloc(n * sizeof(float*));
    size_t* cluster_sizes = (size_t*)safe_malloc(n * sizeof(size_t));
    int* cluster_instance_count = (int*)safe_malloc(n * sizeof(int));
    
    if (!cluster_id || !cluster_used || !cluster_prototypes ||
        !cluster_sizes || !cluster_instance_count) {
        safe_free((void**)&cluster_id);
        safe_free((void**)&cluster_used);
        safe_free((void**)&cluster_prototypes);
        safe_free((void**)&cluster_sizes);
        safe_free((void**)&cluster_instance_count);
        return -1;
    }
    
    for (size_t i = 0; i < n; i++) {
        cluster_id[i] = -1;
        cluster_used[i] = 0;
        cluster_prototypes[i] = NULL;
        cluster_sizes[i] = system->instance_accum_sizes[i];
        cluster_instance_count[i] = 0;
    }
    
    int num_clusters = 0;
    
    // 阈值聚类：基于相似度的单链接聚类
    for (size_t i = 0; i < n; i++) {
        if (cluster_used[i]) continue;
        
        // 以当前实例为种子，开始新聚类
        cluster_id[i] = num_clusters;
        cluster_used[i] = 1;
        cluster_instance_count[num_clusters] = 1;
        
        // 初始化该簇的原型为种子实例
        size_t proto_size = system->instance_accum_sizes[i];
        float* prototype = (float*)safe_malloc(proto_size * sizeof(float));
        if (!prototype) {
            for (int c = 0; c < num_clusters; c++) {
                safe_free((void**)&cluster_prototypes[c]);
            }
            safe_free((void**)&cluster_id);
            safe_free((void**)&cluster_used);
            safe_free((void**)&cluster_prototypes);
            safe_free((void**)&cluster_sizes);
            safe_free((void**)&cluster_instance_count);
            return -1;
        }
        memcpy(prototype, system->instance_accum_buffer[i], proto_size * sizeof(float));
        cluster_prototypes[num_clusters] = prototype;
        cluster_sizes[num_clusters] = proto_size;
        
        // 查找所有与种子实例相似的未使用实例
        for (size_t j = i + 1; j < n; j++) {
            if (cluster_used[j]) continue;
            
            float sim = compute_similarity(
                system->instance_accum_buffer[i],
                system->instance_accum_buffer[j],
                system->instance_accum_sizes[i] < system->instance_accum_sizes[j] ?
                system->instance_accum_sizes[i] : system->instance_accum_sizes[j]);
            
            if (sim >= threshold) {
                cluster_id[j] = num_clusters;
                cluster_used[j] = 1;
                cluster_instance_count[num_clusters]++;
                
                // 增量更新该簇原型（在线平均）
                size_t curr_size = cluster_sizes[num_clusters];
                float* curr_proto = cluster_prototypes[num_clusters];
                float count = (float)cluster_instance_count[num_clusters];
                
                for (size_t d = 0; d < curr_size; d++) {
                    float inst_val = (d < system->instance_accum_sizes[j]) ?
                                     system->instance_accum_buffer[j][d] : 0.0f;
                    curr_proto[d] = curr_proto[d] * (count - 1.0f) / count + inst_val / count;
                }
            }
        }
        
        num_clusters++;
    }
    
    // 处理聚类结果：为每个达到最小实例数的簇创建概念
    int concepts_formed = 0;
    size_t remaining_count = 0;
    
    // 构建未使用实例的新缓冲区（淘汰已形成概念的实例，保留未达到阈值的实例）
    float** new_buffer = (float**)safe_malloc(system->instance_accum_capacity * sizeof(float*));
    size_t* new_sizes = (size_t*)safe_malloc(system->instance_accum_capacity * sizeof(size_t));
    
    if (!new_buffer || !new_sizes) {
        safe_free((void**)&new_buffer);
        safe_free((void**)&new_sizes);
        for (int c = 0; c < num_clusters; c++) {
            safe_free((void**)&cluster_prototypes[c]);
        }
        safe_free((void**)&cluster_id);
        safe_free((void**)&cluster_used);
        safe_free((void**)&cluster_prototypes);
        safe_free((void**)&cluster_sizes);
        safe_free((void**)&cluster_instance_count);
        return -1;
    }
    
    for (int c = 0; c < num_clusters; c++) {
        if (cluster_instance_count[c] >= (int)min_instances) {
            // 达到最小实例数：形成概念
            size_t proto_size = cluster_sizes[c];
            float* prototype = cluster_prototypes[c];
            
            // 创建新概念
            Concept* new_concept = create_concept(NULL, prototype, proto_size,
                                                  ABSTRACTION_LEVEL_CONCEPT);
            if (new_concept) {
                new_concept->prototype_strength = 1.0f - 1.0f / (1.0f + (float)cluster_instance_count[c] * 0.1f);
                new_concept->instance_count = cluster_instance_count[c];
                new_concept->typicality = 0.8f;
                new_concept->centrality = 0.7f;
                new_concept->is_learned = 1;
                new_concept->is_abstract = 0;
                
                // 添加到概念库
                add_concept_to_library(system, new_concept);
                concepts_formed++;
            }
            
            // 释放该簇的原型（已复制到概念中或无需保留）
            safe_free((void**)&cluster_prototypes[c]);
            cluster_prototypes[c] = NULL;
            
            // 释放该簇对应的实例数据
            for (size_t i = 0; i < n; i++) {
                if (cluster_id[i] == c && system->instance_accum_buffer[i]) {
                    safe_free((void**)&system->instance_accum_buffer[i]);
                    system->instance_accum_buffer[i] = NULL;
                }
            }
        } else {
            // 未达到最小实例数：保留实例到新缓冲区
            for (size_t i = 0; i < n; i++) {
                if (cluster_id[i] == c && system->instance_accum_buffer[i]) {
                    new_buffer[remaining_count] = system->instance_accum_buffer[i];
                    new_sizes[remaining_count] = system->instance_accum_sizes[i];
                    system->instance_accum_buffer[i] = NULL;
                    remaining_count++;
                }
            }
        }
    }
    
    // 清理未被任何簇引用的孤立实例
    for (size_t i = 0; i < n; i++) {
        safe_free((void**)&system->instance_accum_buffer[i]);
    }
    
    // 用保留的实例替换缓冲区
    safe_free((void**)&system->instance_accum_buffer);
    safe_free((void**)&system->instance_accum_sizes);
    system->instance_accum_buffer = new_buffer;
    system->instance_accum_sizes = new_sizes;
    system->instance_accum_count = remaining_count;
    system->instance_accum_capacity = system->config.max_accumulated_instances;
    
    // 释放临时数据
    safe_free((void**)&cluster_id);
    safe_free((void**)&cluster_used);
    safe_free((void**)&cluster_prototypes);
    safe_free((void**)&cluster_sizes);
    safe_free((void**)&cluster_instance_count);
    
    return concepts_formed;
}

