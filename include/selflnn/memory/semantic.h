/**
 * @file semantic.h
 * @brief 语义记忆系统接口
 * 
 * 语义记忆管理，支持概念、知识和关系网络。
 */

#ifndef SELFLNN_SEMANTIC_H
#define SELFLNN_SEMANTIC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 语义记忆配置
 */
typedef struct {
    size_t capacity;           /**< 最大概念容量 */
    float association_strength; /**< 关联强度 */
    float generalization_level; /**< 泛化级别 */
    int enable_hierarchy;      /**< 是否启用层次结构 */
} SemanticMemoryConfig;

/**
 * @brief 语义记忆句柄
 */
typedef struct SemanticMemory SemanticMemory;

/**
 * @brief 创建语义记忆实例
 * 
 * @param config 语义记忆配置
 * @return SemanticMemory* 语义记忆句柄，失败返回NULL
 */
SemanticMemory* semantic_memory_create(const SemanticMemoryConfig* config);

/**
 * @brief 释放语义记忆实例
 * 
 * @param memory 语义记忆句柄
 */
void semantic_memory_free(SemanticMemory* memory);

/**
 * @brief 存储语义概念
 * 
 * @param memory 语义记忆句柄
 * @param concept_id 概念ID
 * @param data 概念数据
 * @param data_size 数据大小
 * @param strength 初始强度
 * @return int 成功返回0，失败返回-1
 */
int semantic_memory_store_concept(SemanticMemory* memory, const char* concept_id,
                                 const float* data, size_t data_size, float strength);

/**
 * @brief 检索语义概念
 * 
 * @param memory 语义记忆句柄
 * @param concept_id 概念ID
 * @param data 数据输出缓冲区
 * @param data_size 缓冲区大小
 * @param strength 概念强度输出缓冲区
 * @return int 成功返回0，未找到返回-1
 */
int semantic_memory_retrieve_concept(SemanticMemory* memory, const char* concept_id,
                                    float* data, size_t data_size, float* strength);

/**
 * @brief 建立概念关联
 * 
 * @param memory 语义记忆句柄
 * @param concept_id1 第一个概念ID
 * @param concept_id2 第二个概念ID
 * @param relation_type 关系类型
 * @param strength 关联强度
 * @return int 成功返回0，失败返回-1
 */
int semantic_memory_establish_relation(SemanticMemory* memory, const char* concept_id1,
                                      const char* concept_id2, const char* relation_type,
                                      float strength);

/**
 * @brief 检索相关概念
 * 
 * @param memory 语义记忆句柄
 * @param concept_id 概念ID
 * @param relation_type 关系类型（NULL表示所有类型）
 * @param max_results 最大结果数
 * @param results 结果概念ID输出缓冲区
 * @param strengths 关联强度输出缓冲区
 * @return int 成功返回相关概念数，失败返回-1
 */
int semantic_memory_retrieve_related(SemanticMemory* memory, const char* concept_id,
                                    const char* relation_type, size_t max_results,
                                    char** results, float* strengths);

/**
 * @brief 泛化概念
 * 
 * @param memory 语义记忆句柄
 * @param concept_id 概念ID
 * @param generalization_level 泛化级别
 * @param generalized_concept 泛化后概念输出缓冲区
 * @param data_size 缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int semantic_memory_generalize(SemanticMemory* memory, const char* concept_id,
                              float generalization_level, float* generalized_concept,
                              size_t data_size);

/**
 * @brief 具体化概念
 * 
 * @param memory 语义记忆句柄
 * @param concept_id 概念ID
 * @param specialization_level 具体化级别
 * @param specialized_concept 具体化后概念输出缓冲区
 * @param data_size 缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int semantic_memory_specialize(SemanticMemory* memory, const char* concept_id,
                              float specialization_level, float* specialized_concept,
                              size_t data_size);

/**
 * @brief 获取语义记忆配置
 * 
 * @param memory 语义记忆句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int semantic_memory_get_config(const SemanticMemory* memory, SemanticMemoryConfig* config);

/**
 * @brief 设置语义记忆配置
 * 
 * @param memory 语义记忆句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int semantic_memory_set_config(SemanticMemory* memory, const SemanticMemoryConfig* config);

/**
 * @brief 获取语义记忆统计信息
 * 
 * @param memory 语义记忆句柄
 * @param total_concepts 总概念数输出缓冲区
 * @param avg_relations 平均关联数输出缓冲区
 * @param network_density 网络密度输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int semantic_memory_get_stats(const SemanticMemory* memory, size_t* total_concepts,
                             float* avg_relations, float* network_density);

/**
 * @brief 重置语义记忆
 * 
 * @param memory 语义记忆句柄
 */
void semantic_memory_reset(SemanticMemory* memory);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_SEMANTIC_H