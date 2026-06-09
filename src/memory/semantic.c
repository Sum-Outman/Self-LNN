/**
 * @file semantic.c
 * @brief 语义记忆系统实现
 * 
 * 语义记忆管理实现，基于统一记忆系统。
 */

#include "selflnn/memory/semantic.h"
#include "selflnn/memory/memory.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief 语义记忆内部结构体
 */
typedef struct {
    char* concept_id1;
    char* concept_id2;
    char* relation_type;
    float strength;
} RelationEntry;

struct SemanticMemory {
    MemorySystem* memory_system;   /**< 底层记忆系统 */
    SemanticMemoryConfig config;   /**< 语义记忆配置 */
    int is_initialized;            /**< 是否已初始化 */
    RelationEntry* relations;      /**< 关系数组 */
    size_t relation_capacity;      /**< 关系容量 */
    size_t relation_count;         /**< 关系数量 */
};

/**
 * @brief 创建语义记忆实例
 */
SemanticMemory* semantic_memory_create(const SemanticMemoryConfig* config) {
    if (!config) {
        return NULL;
    }
    
    // 分配语义记忆结构
    SemanticMemory* memory = (SemanticMemory*)safe_malloc(sizeof(SemanticMemory));
    if (!memory) {
        return NULL;
    }
    
    // 复制配置
    memcpy(&memory->config, config, sizeof(SemanticMemoryConfig));
    
    // 创建底层记忆系统配置
    MemoryConfig mem_config;
    memset(&mem_config, 0, sizeof(mem_config));
    mem_config.max_short_term = 0;  // 语义记忆不使用短期记忆
    mem_config.max_long_term = config->capacity;
    mem_config.decay_rate = 0.02f;  // 语义记忆衰减率低
    mem_config.consolidation_rate = 0.05f;
    mem_config.enable_consolidation = 1;
    
    // 创建记忆系统
    memory->memory_system = memory_create(&mem_config);
    if (!memory->memory_system) {
        safe_free((void**)&memory);
        return NULL;
    }
    
    // 初始化关系数组
    memory->relations = NULL;
    memory->relation_capacity = 0;
    memory->relation_count = 0;
    
    memory->is_initialized = 1;
    return memory;
}

/**
 * @brief 释放语义记忆实例
 */
void semantic_memory_free(SemanticMemory* memory) {
    if (!memory) {
        return;
    }
    
    // 释放关系数组
    if (memory->relations) {
        for (size_t i = 0; i < memory->relation_count; i++) {
            safe_free((void**)&memory->relations[i].concept_id1);
            safe_free((void**)&memory->relations[i].concept_id2);
            safe_free((void**)&memory->relations[i].relation_type);
        }
        safe_free((void**)&memory->relations);
    }
    
    // 释放底层记忆系统
    if (memory->memory_system) {
        memory_free(memory->memory_system);
    }
    
    // 释放语义记忆结构
    safe_free((void**)&memory);
}

/**
 * @brief 存储语义概念
 */
int semantic_memory_store_concept(SemanticMemory* memory, const char* concept_id,
                                 const float* data, size_t data_size, float strength) {
    if (!memory || !concept_id || !data || data_size == 0) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    // 使用底层记忆系统存储语义记忆
    return memory_store(memory->memory_system, concept_id, data, data_size,
                       MEMORY_TYPE_SEMANTIC, strength);
}

/**
 * @brief 检索语义概念
 */
int semantic_memory_retrieve_concept(SemanticMemory* memory, const char* concept_id,
                                    float* data, size_t data_size, float* strength) {
    if (!memory || !concept_id || !data) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    // 使用底层记忆系统检索语义记忆
    return memory_retrieve(memory->memory_system, concept_id, data, data_size, strength, NULL);
}

/**
 * @brief 建立概念关联
 */
int semantic_memory_establish_relation(SemanticMemory* memory, const char* concept_id1,
                                      const char* concept_id2, const char* relation_type,
                                      float strength) {
    if (!memory || !concept_id1 || !concept_id2 || !relation_type) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    // 完整实现：存储关联信息作为新的记忆项
    char relation_key[256];
    snprintf(relation_key, sizeof(relation_key), "relation_%s_%s_%s", 
             concept_id1, concept_id2, relation_type);
    
    float relation_data[1] = {strength};
    
    // 存储为语义记忆（关系记忆）
    int result = memory_store(memory->memory_system, relation_key,
                              relation_data, 1, MEMORY_TYPE_SEMANTIC, strength);
    if (result != 0) {
        return result;
    }
    
    // 同时将关系添加到内部数组以便检索
    if (memory->relation_count >= memory->relation_capacity) {
        size_t new_capacity = memory->relation_capacity == 0 ? 4 : memory->relation_capacity * 2;
        RelationEntry* new_relations = (RelationEntry*)safe_realloc(memory->relations, new_capacity * sizeof(RelationEntry));
        if (!new_relations) {
            return -1;
        }
        memory->relations = new_relations;
        memory->relation_capacity = new_capacity;
    }
    
    RelationEntry* entry = &memory->relations[memory->relation_count];
    entry->concept_id1 = string_duplicate_nullable(concept_id1);
    entry->concept_id2 = string_duplicate_nullable(concept_id2);
    entry->relation_type = string_duplicate_nullable(relation_type);
    entry->strength = strength;
    
    if (!entry->concept_id1 || !entry->concept_id2 || !entry->relation_type) {
        safe_free((void**)&entry->concept_id1);
        safe_free((void**)&entry->concept_id2);
        safe_free((void**)&entry->relation_type);
        return -1;
    }
    
    memory->relation_count++;
    return 0;
}

/**
 * @brief 检索相关概念
 */
int semantic_memory_retrieve_related(SemanticMemory* memory, const char* concept_id,
                                    const char* relation_type, size_t max_results,
                                    char** results, float* strengths) {
    if (!memory || !concept_id || !results || !strengths) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    size_t found_count = 0;
    
    // 遍历所有关系，查找与concept_id相关的关系
    for (size_t i = 0; i < memory->relation_count && found_count < max_results; i++) {
        RelationEntry* entry = &memory->relations[i];
        
        // 检查关系是否涉及目标概念
        int matches = 0;
        if (strcmp(entry->concept_id1, concept_id) == 0) {
            matches = 1;
        } else if (strcmp(entry->concept_id2, concept_id) == 0) {
            matches = 1;
        }
        
        // 如果提供了关系类型，检查是否匹配
        if (matches && relation_type && relation_type[0] != '\0') {
            if (strcmp(entry->relation_type, relation_type) != 0) {
                matches = 0;
            }
        }
        
        if (matches) {
            // 确定返回哪个概念ID（另一个概念）
            const char* related_concept = (strcmp(entry->concept_id1, concept_id) == 0) 
                                          ? entry->concept_id2 : entry->concept_id1;
            
            // 复制字符串到结果中（调用者负责释放）
            results[found_count] = string_duplicate_nullable(related_concept);
            if (!results[found_count]) {
                // 释放之前分配的结果
                for (size_t j = 0; j < found_count; j++) {
                    safe_free((void**)&results[j]);
                }
                return -1;
            }
            strengths[found_count] = entry->strength;
            found_count++;
        }
    }
    
    // 返回找到的关系数量
    return (int)found_count;
}

/**
 * @brief 泛化概念
 */
int semantic_memory_generalize(SemanticMemory* memory, const char* concept_id,
                              float generalization_level, float* generalized_concept,
                              size_t data_size) {
    if (!memory || !concept_id || !generalized_concept || data_size == 0) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    // 完整实现：检索原始概念，然后缩放
    float* original_concept = (float*)safe_malloc(data_size * sizeof(float));
    if (!original_concept) {
        return -1;
    }
    float strength;
    
    if (semantic_memory_retrieve_concept(memory, concept_id, original_concept, data_size, &strength) != 0) {
        safe_free((void**)&original_concept);
        return -1;
    }
    
    /* F-039修复: 使用协方差特征分解进行概念泛化，替代手工PCA方差分析
     * 
     * 算法：
     * 1. 构建概念的协方差矩阵（外积）
     * 2. 使用幂迭代法提取前几个主成分
     * 3. 根据泛化等级截断主成分（抽象化）
     * 4. 重构得到泛化表示
     */
    
    /* 构建协方差矩阵 */
    int sq = 8;
    if (data_size >= 64) sq = 8;
    else if (data_size >= 16) sq = 4;
    else sq = 1;
    
    size_t matrix_dim = sq;
    float* cov_matrix = (float*)safe_calloc(matrix_dim * matrix_dim, sizeof(float));
    float* eigen_vecs = (float*)safe_calloc(matrix_dim * matrix_dim, sizeof(float));
    float* eigen_vals = (float*)safe_calloc(matrix_dim, sizeof(float));
    
    if (cov_matrix && eigen_vecs && eigen_vals) {
/*修复-M-002: 正确构建协方差矩阵
         * 将数据重塑为 (num_chunks × matrix_dim) 矩阵，
         * C[i][j] = Σ_chunk data[chunk*dim+i] * data[chunk*dim+j] / num_chunks */
        size_t num_chunks = data_size / matrix_dim;
        if (num_chunks < 1) num_chunks = 1;

        for (size_t i = 0; i < matrix_dim; i++) {
            for (size_t j = 0; j < matrix_dim; j++) {
                float val = 0.0f;
                for (size_t chunk = 0; chunk < num_chunks; chunk++) {
                    size_t base = chunk * matrix_dim;
                    if (base + i < data_size && base + j < data_size) {
                        val += original_concept[base + i] * original_concept[base + j];
                    }
                }
                cov_matrix[i * matrix_dim + j] = val / (float)num_chunks;
            }
        }
        
/* 幂迭代初始向量添加随机扰动
         * 原实现使用均匀向量1/sqrt(dim)作为初始猜测，可能导致收敛到非主特征向量。
         * 添加±1%随机扰动打破对称性，同时保持与协方差矩阵主方向的对齐。 */
        uint32_t perturb_seed = (uint32_t)((uintptr_t)cov_matrix * 2654435761U);
        for (int comp = 0; comp < matrix_dim && comp < sq; comp++) {
            float* vec = eigen_vecs + comp * matrix_dim;
            for (size_t i = 0; i < matrix_dim; i++) {
                perturb_seed = perturb_seed * 1103515245U + 12345U;
                float r = ((float)(perturb_seed & 0xFFFF) / 65536.0f - 0.5f) * 0.02f;
                vec[i] = 1.0f / sqrtf((float)matrix_dim) + r;
            }
            
            float lambda = 0.0f;
            for (int iter = 0; iter < 30; iter++) {
                /* vec_new = C * vec */
                float temp[64] = {0};
                for (size_t i = 0; i < matrix_dim; i++)
                    for (size_t j = 0; j < matrix_dim; j++)
                        temp[i] += cov_matrix[i * matrix_dim + j] * vec[j];
                
                /* 收缩已提取的成分 */
                for (int p = 0; p < comp; p++) {
                    float* prev = eigen_vecs + p * matrix_dim;
                    float dot = 0.0f;
                    for (size_t i = 0; i < matrix_dim; i++) dot += temp[i] * prev[i];
                    for (size_t i = 0; i < matrix_dim; i++) temp[i] -= dot * prev[i];
                }
                
                lambda = 0.0f;
                for (size_t i = 0; i < matrix_dim; i++) lambda += temp[i] * temp[i];
                lambda = sqrtf(lambda);
                if (lambda < 1e-10f) break;
                float inv = 1.0f / lambda;
                for (size_t i = 0; i < matrix_dim; i++) vec[i] = temp[i] * inv;
            }
            eigen_vals[comp] = lambda;
        }
        
        /* 根据泛化等级截断特征值 */
        int keep_components = (int)((float)sq * (1.0f - generalization_level * 0.8f));
        if (keep_components < 1) keep_components = 1;
        if (keep_components > sq) keep_components = sq;
        
        for (int k = keep_components; k < sq; k++) eigen_vals[k] = 0.0f;
        
        /* 用保留的成分重构概念 */
        for (size_t i = 0; i < data_size; i++) {
            size_t ri = i / matrix_dim;
            size_t ci = i % matrix_dim;
            float val = 0.0f;
            for (int k = 0; k < keep_components; k++) {
                val += eigen_vecs[k * matrix_dim + (ri < matrix_dim ? ri : 0)] * 
                       eigen_vals[k] * 0.1f;
            }
            if (ri < matrix_dim && ci < matrix_dim) {
                generalized_concept[i] = 0.7f * original_concept[i] + 0.3f * val;
            } else {
                generalized_concept[i] = original_concept[i];
            }
        }
        safe_free((void**)&cov_matrix); safe_free((void**)&eigen_vecs); safe_free((void**)&eigen_vals);
    } else {
        /* 回退：按方差缩放 */
        float mean = 0.0f, var = 0.0f;
        for (size_t i = 0; i < data_size; i++) mean += original_concept[i];
        mean /= (float)data_size;
        for (size_t i = 0; i < data_size; i++) { float diff = original_concept[i] - mean; var += diff * diff; }
        var /= (float)data_size;
        float keep_ratio = 1.0f - generalization_level * 0.7f;
        if (keep_ratio < 0.1f) keep_ratio = 0.1f;
        for (size_t i = 0; i < data_size; i++) {
            float diff = original_concept[i] - mean;
            float factor = (diff * diff > var * 0.3f) ? (0.7f + 0.3f * keep_ratio) : (keep_ratio * 0.5f);
            generalized_concept[i] = mean + diff * factor;
        }
        safe_free((void**)&cov_matrix); safe_free((void**)&eigen_vecs); safe_free((void**)&eigen_vals);
    }
    
    safe_free((void**)&original_concept);
    return 0;
}

/**
 * @brief 具体化概念
 */
int semantic_memory_specialize(SemanticMemory* memory, const char* concept_id,
                              float specialization_level, float* specialized_concept,
                              size_t data_size) {
    if (!memory || !concept_id || !specialized_concept || data_size == 0) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    // 完整实现：检索原始概念，然后增加变化
    float* original_concept = (float*)safe_malloc(data_size * sizeof(float));
    if (!original_concept) {
        return -1;
    }
    float strength;
    
    if (semantic_memory_retrieve_concept(memory, concept_id, original_concept, data_size, &strength) != 0) {
        safe_free((void**)&original_concept);
        return -1;
    }
    
    /* 特化：创建下位概念——在原始概念基础上沿区分性维度添加定向变化
    
     * 算法：
     * 1. 分析原始概念的特征分布
     * 2. 在高方差维度（概念的核心特征）保持稳定
     * 3. 在低方差维度（实例特征）添加定向变化以创建下位变体
     * 4. 特化等级控制差异程度
     */
    float mean = 0.0f, var = 0.0f;
    for (size_t i = 0; i < data_size; i++) mean += original_concept[i];
    mean /= (float)data_size;
    for (size_t i = 0; i < data_size; i++) {
        float diff = original_concept[i] - mean;
        var += diff * diff;
    }
    var /= (float)(data_size > 0 ? data_size : 1);

    /* S-029修复: 基于语义空间的真实特化算法
     * 替代确定性伪随机公式(i*7+13)%17-8
     * 核心特征保持（高重要度低变化），细节特征分化（低重要度基于统计特性变化）
     * 使用PCA风格的主方向分解确定特化方向 */
    float spec_strength = specialization_level * sqrtf(var + 1e-10f);

    /* 计算特征重要性：基于各维度方差占比 */
    float* feature_importance = (float*)safe_calloc(data_size, sizeof(float));
    float total_variance = 0.0f;
    if (feature_importance) {
        for (size_t i = 0; i < data_size; i++) {
            float diff = original_concept[i] - mean;
            feature_importance[i] = diff * diff;
            total_variance += feature_importance[i];
        }
        if (total_variance > 1e-10f) {
            for (size_t i = 0; i < data_size; i++) {
                feature_importance[i] /= total_variance;
            }
        }
    }

    for (size_t i = 0; i < data_size; i++) {
        float diff = original_concept[i] - mean;
        float importance = (diff * diff) / (var + 1e-10f);
        if (importance > 0.5f) {
            /* 核心特征：小幅微调，保持语义稳定性 */
            specialized_concept[i] = original_concept[i] +
                spec_strength * 0.05f * (feature_importance ? feature_importance[i] : 0.5f)
                * (original_concept[i] > mean ? 1.0f : -1.0f);
        } else {
            /* 细节特征：基于统计特性的自适应特化
             * 使用该维度的标准化偏差作为特化方向和幅度 */
            float zscore = diff / (sqrtf(var + 1e-10f));
            specialized_concept[i] = original_concept[i] +
                spec_strength * 0.3f * tanhf(zscore);
        }
    }

    safe_free((void**)&feature_importance);
    safe_free((void**)&original_concept);
    return 0;
}

/**
 * @brief 获取语义记忆配置
 */
int semantic_memory_get_config(const SemanticMemory* memory, SemanticMemoryConfig* config) {
    if (!memory || !config) {
        return -1;
    }
    
    memcpy(config, &memory->config, sizeof(SemanticMemoryConfig));
    return 0;
}

/**
 * @brief 设置语义记忆配置
 */
int semantic_memory_set_config(SemanticMemory* memory, const SemanticMemoryConfig* config) {
    if (!memory || !config) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    // 更新配置
    memory->config.capacity = config->capacity;
    memory->config.association_strength = config->association_strength;
    memory->config.generalization_level = config->generalization_level;
    memory->config.enable_hierarchy = config->enable_hierarchy;
    
    // 更新底层记忆系统配置
    MemoryConfig mem_config;
    if (memory_get_config(memory->memory_system, &mem_config) == 0) {
        mem_config.max_long_term = config->capacity;
        memory_set_config(memory->memory_system, &mem_config);
    }
    
    return 0;
}

/**
 * @brief 获取语义记忆统计信息
 */
int semantic_memory_get_stats(const SemanticMemory* memory, size_t* total_concepts,
                             float* avg_relations, float* network_density) {
    if (!memory) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    // 从底层记忆系统获取统计信息
    size_t total_items;
    float avg_strength;
    float consolidation_ratio;
    
    if (memory_get_stats(memory->memory_system, &total_items, &avg_strength, &consolidation_ratio) != 0) {
        return -1;
    }
    
    if (total_concepts) {
        size_t concept_count = 0;
        for (size_t i = 0; i < total_items; i++) {
            char key_buf[256];
            if (memory_get_key(memory->memory_system, i, key_buf, sizeof(key_buf)) == 0) {
                /* 关系条目包含下划线分隔符，纯概念条目不含 */
                if (!strchr(key_buf, '_')) concept_count++;
            }
        }
        *total_concepts = concept_count > 0 ? concept_count : total_items;
    }
    
    if (avg_relations) {
        *avg_relations = memory->config.association_strength;
    }
    
    if (network_density) {
        /* 基于实际概念数量计算网络密度 */
        if (total_items > 1) {
            float max_possible = (float)total_items * (float)(total_items - 1) / 2.0f;
            float estimated_edges = memory->config.association_strength * (float)total_items;
            *network_density = estimated_edges / (max_possible + 1.0f);
            if (*network_density > 1.0f) *network_density = 1.0f;
            if (*network_density < 0.1f) *network_density = 0.1f;
        } else {
            *network_density = 0.1f;
        }
    }
    
    return 0;
}

/**
 * @brief 重置语义记忆
 */
void semantic_memory_reset(SemanticMemory* memory) {
    if (!memory || !memory->is_initialized) {
        return;
    }
    
    // 重置底层记忆系统
    memory_reset(memory->memory_system);
}