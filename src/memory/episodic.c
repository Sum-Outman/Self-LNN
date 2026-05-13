/**
 * @file episodic.c
 * @brief 情景记忆系统实现
 * 
 * 情景记忆管理实现，基于统一记忆系统。
 */

#include "selflnn/memory/episodic.h"
#include "selflnn/memory/memory.h"
#include "selflnn/utils/memory_utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* F-029: 文件作用域排序结构体（供qsort比较函数使用） */
typedef struct {
    char* related_id;
    float assoc_strength;
} MatchResult;

typedef struct {
    char* event_id;
    float timestamp;
} TimeMatch;

/**
 * @brief 情景记忆内部元数据结构（事件索引）
 */
typedef struct {
    char* event_id;             /**< 事件ID */
    float timestamp;            /**< 事件时间戳 */
    float strength;             /**< 事件强度 */
    int has_associations;       /**< 是否有关联 */
} EventIndex;

/**
 * @brief 关联索引结构
 */
typedef struct {
    char* event_id1;            /**< 事件ID1 */
    char* event_id2;            /**< 事件ID2 */
    float association_strength; /**< 关联强度 */
} AssociationIndex;

/**
 * @brief 情景记忆内部结构体
 */
struct EpisodicMemory {
    MemorySystem* memory_system;     /**< 底层记忆系统 */
    EpisodicMemoryConfig config;     /**< 情景记忆配置 */
    int is_initialized;              /**< 是否已初始化 */
    float current_time;              /**< 当前时间 */
    
    /* 事件索引（用于时间范围检索和相关事件检索） */
    EventIndex* event_index;         /**< 事件索引数组 */
    size_t event_count;              /**< 事件索引计数 */
    size_t event_capacity;           /**< 事件索引容量 */
    
    /* 关联索引（用于事件关联检索） */
    AssociationIndex* assoc_index;   /**< 关联索引数组 */
    size_t assoc_count;              /**< 关联索引计数 */
    size_t assoc_capacity;           /**< 关联索引容量 */
};

/**
 * @brief 创建情景记忆实例
 */
EpisodicMemory* episodic_memory_create(const EpisodicMemoryConfig* config) {
    if (!config) {
        return NULL;
    }
    
    // 分配情景记忆结构
    EpisodicMemory* memory = (EpisodicMemory*)safe_malloc(sizeof(EpisodicMemory));
    if (!memory) {
        return NULL;
    }
    
    // 复制配置
    memcpy(&memory->config, config, sizeof(EpisodicMemoryConfig));
    
    // 创建底层记忆系统配置
    MemoryConfig mem_config;
    mem_config.max_short_term = 0;  // 情景记忆不使用短期记忆
    mem_config.max_long_term = config->capacity;
    mem_config.decay_rate = 0.05f;  // 情景记忆衰减率中等
    mem_config.consolidation_rate = 0.1f;
    mem_config.enable_consolidation = 1;
    
    // 创建记忆系统
    memory->memory_system = memory_create(&mem_config);
    if (!memory->memory_system) {
        safe_free((void**)&memory);
        return NULL;
    }
    
    memory->is_initialized = 1;
    memory->current_time = 0.0f;
    
    // 初始化事件索引
    memory->event_capacity = config->capacity > 0 ? config->capacity : 100;
    memory->event_index = (EventIndex*)safe_malloc(
        memory->event_capacity * sizeof(EventIndex));
    if (!memory->event_index) {
        memory_free(memory->memory_system);
        safe_free((void**)&memory);
        return NULL;
    }
    memset(memory->event_index, 0, memory->event_capacity * sizeof(EventIndex));
    memory->event_count = 0;
    
    // 初始化关联索引
    memory->assoc_capacity = config->capacity * 2;
    memory->assoc_index = (AssociationIndex*)safe_malloc(
        memory->assoc_capacity * sizeof(AssociationIndex));
    if (!memory->assoc_index) {
        safe_free((void**)&memory->event_index);
        memory_free(memory->memory_system);
        safe_free((void**)&memory);
        return NULL;
    }
    memset(memory->assoc_index, 0, memory->assoc_capacity * sizeof(AssociationIndex));
    memory->assoc_count = 0;
    
    return memory;
}

/**
 * @brief 释放情景记忆实例
 */
void episodic_memory_free(EpisodicMemory* memory) {
    if (!memory) {
        return;
    }
    
    // 释放底层记忆系统
    if (memory->memory_system) {
        memory_free(memory->memory_system);
    }
    
    // 释放事件索引
    if (memory->event_index) {
        for (size_t i = 0; i < memory->event_count; i++) {
            safe_free((void**)&memory->event_index[i].event_id);
        }
        safe_free((void**)&memory->event_index);
    }
    
    // 释放关联索引
    if (memory->assoc_index) {
        for (size_t i = 0; i < memory->assoc_count; i++) {
            safe_free((void**)&memory->assoc_index[i].event_id1);
            safe_free((void**)&memory->assoc_index[i].event_id2);
        }
        safe_free((void**)&memory->assoc_index);
    }
    
    // 释放情景记忆结构
    safe_free((void**)&memory);
}

/**
 * @brief 存储情景记忆事件
 */
int episodic_memory_store_event(EpisodicMemory* memory, const char* event_id,
                               const float* data, size_t data_size,
                               float timestamp, float strength) {
    if (!memory || !event_id || !data || data_size == 0) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    // 如果时间戳为0，使用当前时间
    float actual_timestamp = (timestamp > 0) ? timestamp : memory->current_time;
    
    // 使用底层记忆系统存储情景记忆
    int result = memory_store(memory->memory_system, event_id, data, data_size,
                             MEMORY_TYPE_EPISODIC, strength);
    
    if (result == 0) {
        // 添加到事件索引
        if (memory->event_count >= memory->event_capacity) {
            // 扩展事件索引容量
            size_t new_capacity = memory->event_capacity * 2;
            EventIndex* new_index = (EventIndex*)safe_realloc(
                memory->event_index, new_capacity * sizeof(EventIndex));
            if (new_index) {
                memset(&new_index[memory->event_capacity], 0,
                       (new_capacity - memory->event_capacity) * sizeof(EventIndex));
                memory->event_index = new_index;
                memory->event_capacity = new_capacity;
            }
        }
        
        if (memory->event_count < memory->event_capacity) {
            memory->event_index[memory->event_count].event_id = 
                (char*)safe_malloc(strlen(event_id) + 1);
            if (memory->event_index[memory->event_count].event_id) {
                strcpy(memory->event_index[memory->event_count].event_id, event_id);
                memory->event_index[memory->event_count].timestamp = actual_timestamp;
                memory->event_index[memory->event_count].strength = strength;
                memory->event_index[memory->event_count].has_associations = 0;
                memory->event_count++;
            }
        }
        
        // 更新时间
        memory->current_time = fmaxf(memory->current_time, actual_timestamp) + 1.0f;
    }
    
    return result;
}

/**
 * @brief 检索情景记忆事件
 */
int episodic_memory_retrieve_event(EpisodicMemory* memory, const char* event_id,
                                  float* data, size_t data_size,
                                  float* timestamp, float* strength) {
    if (!memory || !event_id || !data) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    // 使用底层记忆系统检索情景记忆
    int result = memory_retrieve(memory->memory_system, event_id, data, data_size, strength, NULL);

    /* M-028修复: 返回事件索引中的真实存储时间戳而非current_time */
    if (result == 0 && timestamp) {
        int found = 0;
        for (size_t i = 0; i < memory->event_count; i++) {
            if (memory->event_index[i].event_id &&
                strcmp(memory->event_index[i].event_id, event_id) == 0) {
                *timestamp = memory->event_index[i].timestamp;
                found = 1;
                break;
            }
        }
        if (!found) *timestamp = memory->current_time;
    }
    
    return result;
}

/**
 * @brief 关联两个事件
 */
int episodic_memory_associate_events(EpisodicMemory* memory, const char* event_id1,
                                    const char* event_id2, float association_strength) {
    if (!memory || !event_id1 || !event_id2) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    if (!memory->config.enable_chaining) {
        return 0;  // 事件链未启用
    }
    
    // 完整实现：存储关联信息作为新的记忆项
    char association_key[256];
    snprintf(association_key, sizeof(association_key), "episodic_assoc_%s_%s", event_id1, event_id2);
    
    float association_data[1] = {association_strength};
    
    // 存储为情景记忆（关联记忆）
    int result = memory_store(memory->memory_system, association_key,
                       association_data, 1, MEMORY_TYPE_EPISODIC, association_strength);
    
    if (result == 0) {
        // 添加到关联索引
        if (memory->assoc_count >= memory->assoc_capacity) {
            size_t new_capacity = memory->assoc_capacity * 2;
            AssociationIndex* new_index = (AssociationIndex*)safe_realloc(
                memory->assoc_index, new_capacity * sizeof(AssociationIndex));
            if (new_index) {
                memset(&new_index[memory->assoc_capacity], 0,
                       (new_capacity - memory->assoc_capacity) * sizeof(AssociationIndex));
                memory->assoc_index = new_index;
                memory->assoc_capacity = new_capacity;
            }
        }
        
        if (memory->assoc_count < memory->assoc_capacity) {
            memory->assoc_index[memory->assoc_count].event_id1 =
                (char*)safe_malloc(strlen(event_id1) + 1);
            memory->assoc_index[memory->assoc_count].event_id2 =
                (char*)safe_malloc(strlen(event_id2) + 1);
            if (memory->assoc_index[memory->assoc_count].event_id1 &&
                memory->assoc_index[memory->assoc_count].event_id2) {
                strcpy(memory->assoc_index[memory->assoc_count].event_id1, event_id1);
                strcpy(memory->assoc_index[memory->assoc_count].event_id2, event_id2);
                memory->assoc_index[memory->assoc_count].association_strength = association_strength;
                memory->assoc_count++;
            } else {
                safe_free((void**)&memory->assoc_index[memory->assoc_count].event_id1);
                safe_free((void**)&memory->assoc_index[memory->assoc_count].event_id2);
            }
        }
        
        // 标记两个事件的关联状态
        for (size_t i = 0; i < memory->event_count; i++) {
            if (strcmp(memory->event_index[i].event_id, event_id1) == 0 ||
                strcmp(memory->event_index[i].event_id, event_id2) == 0) {
                memory->event_index[i].has_associations = 1;
            }
        }
    }
    
    return result;
}

/**
 * @brief 检索相关事件
 * 
 * 通过关联索引查找与指定事件相关联的其他事件。
 * 搜索同时考虑 event_id1->event_id2 和 event_id2->event_id1 双向关联。
 * 结果按关联强度降序排列（F-029修复：使用qsort替代冒泡排序）。
 */

/* F-029: qsort比较函数 — 按关联强度降序 */
static int cmp_match_by_strength_desc(const void* a, const void* b) {
    float sa = ((const MatchResult*)a)->assoc_strength;
    float sb = ((const MatchResult*)b)->assoc_strength;
    if (sa > sb) return -1;
    if (sa < sb) return 1;
    return 0;
}

/* F-029: qsort比较函数 — 按时间戳升序 */
static int cmp_timematch_by_time_asc(const void* a, const void* b) {
    float ta = ((const TimeMatch*)a)->timestamp;
    float tb = ((const TimeMatch*)b)->timestamp;
    if (ta < tb) return -1;
    if (ta > tb) return 1;
    return 0;
}
int episodic_memory_retrieve_related(EpisodicMemory* memory, const char* event_id,
                                    size_t max_results, char** results, float* strengths) {
    if (!memory || !event_id || !results || !strengths) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    if (max_results == 0 || memory->assoc_count == 0) {
        return 0;
    }
    
    MatchResult* matches = (MatchResult*)safe_malloc(
        (memory->assoc_count * 2) * sizeof(MatchResult));
    if (!matches) {
        return -1;
    }
    
    size_t match_count = 0;
    
    // 搜索关联索引（双向搜索）
    for (size_t i = 0; i < memory->assoc_count; i++) {
        if (strcmp(memory->assoc_index[i].event_id1, event_id) == 0) {
            // event_id -> event_id2
            matches[match_count].related_id = memory->assoc_index[i].event_id2;
            matches[match_count].assoc_strength = memory->assoc_index[i].association_strength;
            match_count++;
        } else if (strcmp(memory->assoc_index[i].event_id2, event_id) == 0) {
            // event_id2 -> event_id1（反向关联）
            matches[match_count].related_id = memory->assoc_index[i].event_id1;
            matches[match_count].assoc_strength = memory->assoc_index[i].association_strength;
            match_count++;
        }
    }
    
    if (match_count == 0) {
        safe_free((void**)&matches);
        return 0;
    }
    
    // 按关联强度降序排序（F-029: qsort O(nlogn)替代冒泡排序 O(n²)）
    qsort(matches, match_count, sizeof(MatchResult), cmp_match_by_strength_desc);
    
    // 返回结果（不超过max_results）
    size_t return_count = match_count < max_results ? match_count : max_results;
    for (size_t i = 0; i < return_count; i++) {
        // 分配并复制结果ID
        results[i] = (char*)safe_malloc(strlen(matches[i].related_id) + 1);
        if (results[i]) {
            strcpy(results[i], matches[i].related_id);
        }
        strengths[i] = matches[i].assoc_strength;
    }
    
    safe_free((void**)&matches);
    return (int)return_count;
}

/**
 * @brief 按时间范围检索事件
 * 
 * 通过事件索引按时间范围检索情景记忆事件。
 * 结果按时间戳升序排列（从早到晚）。
 */
int episodic_memory_retrieve_by_time(EpisodicMemory* memory, float start_time,
                                    float end_time, size_t max_results,
                                    char** results, float* timestamps) {
    if (!memory || !results || !timestamps) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    if (max_results == 0 || memory->event_count == 0) {
        return 0;
    }
    
    TimeMatch* matches = (TimeMatch*)safe_malloc(
        memory->event_count * sizeof(TimeMatch));
    if (!matches) {
        return -1;
    }
    
    size_t match_count = 0;
    
    // 搜索事件索引
    for (size_t i = 0; i < memory->event_count; i++) {
        if (memory->event_index[i].event_id &&
            memory->event_index[i].timestamp >= start_time &&
            memory->event_index[i].timestamp <= end_time) {
            matches[match_count].event_id = memory->event_index[i].event_id;
            matches[match_count].timestamp = memory->event_index[i].timestamp;
            match_count++;
        }
    }
    
    if (match_count == 0) {
        safe_free((void**)&matches);
        return 0;
    }
    
    // 按时间戳升序排序（F-029: qsort O(nlogn)替代冒泡排序 O(n²)）
    qsort(matches, match_count, sizeof(TimeMatch), cmp_timematch_by_time_asc);
    
    // 返回结果（不超过max_results）
    size_t return_count = match_count < max_results ? match_count : max_results;
    for (size_t i = 0; i < return_count; i++) {
        results[i] = (char*)safe_malloc(strlen(matches[i].event_id) + 1);
        if (results[i]) {
            strcpy(results[i], matches[i].event_id);
        }
        timestamps[i] = matches[i].timestamp;
    }
    
    safe_free((void**)&matches);
    return (int)return_count;
}

/**
 * @brief 获取情景记忆配置
 */
int episodic_memory_get_config(const EpisodicMemory* memory, EpisodicMemoryConfig* config) {
    if (!memory || !config) {
        return -1;
    }
    
    memcpy(config, &memory->config, sizeof(EpisodicMemoryConfig));
    return 0;
}

/**
 * @brief 设置情景记忆配置
 */
int episodic_memory_set_config(EpisodicMemory* memory, const EpisodicMemoryConfig* config) {
    if (!memory || !config) {
        return -1;
    }
    
    if (!memory->is_initialized) {
        return -1;
    }
    
    // 更新配置
    memory->config.capacity = config->capacity;
    memory->config.temporal_resolution = config->temporal_resolution;
    memory->config.association_strength = config->association_strength;
    memory->config.enable_chaining = config->enable_chaining;
    
    // 更新底层记忆系统配置
    MemoryConfig mem_config;
    if (memory_get_config(memory->memory_system, &mem_config) == 0) {
        mem_config.max_long_term = config->capacity;
        memory_set_config(memory->memory_system, &mem_config);
    }
    
    return 0;
}

/**
 * @brief 获取情景记忆统计信息
 */
int episodic_memory_get_stats(const EpisodicMemory* memory, size_t* total_events,
                             float* avg_association, float* temporal_density) {
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
    
    if (total_events) {
        *total_events = total_items;
    }
    
    if (avg_association) {
        *avg_association = memory->config.association_strength;
    }
    
    if (temporal_density) {
        // 计算时间密度：总事件数 / 当前时间
        *temporal_density = memory->current_time > 0 ? 
                           (float)total_items / memory->current_time : 0.0f;
    }
    
    return 0;
}

/**
 * @brief 重置情景记忆
 */
void episodic_memory_reset(EpisodicMemory* memory) {
    if (!memory || !memory->is_initialized) {
        return;
    }
    
    // 重置底层记忆系统
    memory_reset(memory->memory_system);
    
    // 重置事件索引
    for (size_t i = 0; i < memory->event_count; i++) {
        safe_free((void**)&memory->event_index[i].event_id);
    }
    memory->event_count = 0;
    
    // 重置关联索引
    for (size_t i = 0; i < memory->assoc_count; i++) {
        safe_free((void**)&memory->assoc_index[i].event_id1);
        safe_free((void**)&memory->assoc_index[i].event_id2);
    }
    memory->assoc_count = 0;
    
    // 重置时间
    memory->current_time = 0.0f;
}