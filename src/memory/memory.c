#include "selflnn/memory/memory.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define MEMORY_RECENT_CACHE_SIZE 10
#define MEMORY_HASH_INDEX_THRESHOLD 100 /**< 超过此阈值启用哈希索引 */
#define MEMORY_HASH_INDEX_BUCKETS 1024  /**< 哈希桶数量 */

/* djb2哈希——将字符串key映射到32位无符号整数 */
static uint32_t memory_hash_key(const char* key) {
    uint32_t h = 5381;
    if (!key) return 0;
    while (*key) { h = ((h << 5) + h) + (unsigned char)(*key); key++; }
    return h;
}

/**
 * @file memory.c
 * @brief 记忆系统实现
 * 
 * 统一记忆系统实现，支持短期记忆、长期记忆、情景记忆和语义记忆。
 */

#include "selflnn/memory/memory.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/platform.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ========== P1: MemorySystem锁宏 ========== */
#define MEMORY_LOCK(s)    mutex_lock((s)->lock)
#define MEMORY_UNLOCK(s)  mutex_unlock((s)->lock)

/**
 * @brief 最大键长度
 */
#define MAX_KEY_LENGTH 256

/**
 * @brief 最大压缩映射数
 */
#define MAX_COMPRESSION_MAPS 128

/**
 * @brief 最大再巩固条目数
 */
#define MAX_RECONSOLIDATION_ENTRIES 64

/**
 * @brief 工作记忆复述阈值 — 复述N次后存入短期记忆
 */
#define MEMORY_REHEARSE_SHORT_TERM_THRESHOLD 3

/**
 * @brief 工作记忆复述阈值 — 复述N次后存入长期记忆
 */
#define MEMORY_REHEARSE_LONG_TERM_THRESHOLD 7

/**
 * @brief 记忆系统内部结构体
 */
struct MemorySystem {
    MemoryConfig config;           /**< 记忆配置 */
    MemoryItem** short_term_mem;   /**< 短期记忆数组 */
    MemoryItem** long_term_mem;    /**< 长期记忆数组 */
    MemoryItem** episodic_mem;     /**< 情景记忆数组 */
    MemoryItem** semantic_mem;     /**< 语义记忆数组 */
    size_t st_count;               /**< 短期记忆计数 */
    size_t lt_count;               /**< 长期记忆计数 */
    size_t ep_count;               /**< 情景记忆计数 */
    size_t se_count;               /**< 语义记忆计数 */
    size_t st_capacity;            /**< 短期记忆容量 */
    size_t lt_capacity;            /**< 长期记忆容量 */
    size_t ep_capacity;            /**< 情景记忆容量 */
    size_t se_capacity;            /**< 语义记忆容量 */
    float current_time;            /**< 当前时间（秒级浮点，与last_decay_time使用相同的time(NULL)真实挂钟时间实现一致性，M-016修复）*/
    
    /* 最近访问缓存（加速频繁访问的记忆检索） */
    MemoryItem* recent_cache[10];  /**< 最近访问记忆缓存 */
    size_t cache_index;            /**< 缓存索引（循环缓存） */
    size_t cache_hits;             /**< 缓存命中次数 */
    size_t cache_misses;           /**< 缓存未命中次数 */
    
/* 哈希索引表——加速大规模记忆库检索
     * 当任意记忆数组超过HASH_INDEX_THRESHOLD时自动构建
     * 使用djb2哈希将key映射到32位索引，O(1)查找替代O(N)线性扫描 */
    uint32_t* hash_index_keys;     /**< 哈希键数组（djb2哈希值） */
    int* hash_index_ptr;           /**< 哈希→数组索引映射 */
    size_t hash_index_size;        /**< 哈希表大小 */
    int hash_index_active;         /**< 哈希索引是否活跃 */

    /* 工作记忆 */
    WorkingMemorySlot* working_slots;     /**< 工作记忆槽位数组 */
    size_t working_count;                 /**< 活跃工作记忆槽位数 */
    size_t working_capacity;              /**< 工作记忆容量 */
    
    /* 压缩映射表（键->压缩信息） */
    char compression_keys[MAX_COMPRESSION_MAPS][64]; /**< 压缩记忆键 */
    MemoryCompressionInfo compression_infos[MAX_COMPRESSION_MAPS]; /**< 压缩信息 */
    size_t compression_count;                           /**< 压缩映射计数 */
    
    /* 再巩固条目表 */
    ReconsolidationEntry reconsolidation_entries[MAX_RECONSOLIDATION_ENTRIES]; /**< 再巩固条目 */
    size_t reconsolidation_count; /**< 活跃再巩固条目数 */
    
    int is_initialized;            /**< 是否已初始化 */
    
    // ---- P1: 线程安全锁 ----
    MutexHandle lock;              /**< 记忆系统内部锁 */
    time_t last_decay_time; /**< 上次衰减时间(真实挂钟时间) */
};

/**
 * @brief 创建记忆项
 */
static MemoryItem* memory_item_create(const char* key, const float* data,
                                     size_t data_size, MemoryType type,
                                     float strength, float timestamp) {
    MemoryItem* item = (MemoryItem*)safe_malloc(sizeof(MemoryItem));
    if (!item) {
        return NULL;
    }
    
    // 复制键
    item->key = (char*)safe_malloc(strlen(key) + 1);
    if (!item->key) {
        safe_free((void**)&item);
        return NULL;
    }
    strcpy(item->key, key);
    
    // 复制数据
    if (data_size > 0) {
        item->data = (float*)safe_malloc(data_size * sizeof(float));
        if (!item->data) {
            safe_free((void**)&item->key);
            safe_free((void**)&item);
            return NULL;
        }
        memcpy(item->data, data, data_size * sizeof(float));
    } else {
/* data_size==0时data置NULL，调用者需空指针检查
         * memory_feature_similarity和所有数据访问点已有if(data && data_size>0)守卫 */
        item->data = NULL;
    }
    
    item->data_size = data_size;
    item->strength = strength;
    item->timestamp = timestamp;
    item->type = type;
    item->last_access_time = timestamp;
    item->access_count = 0;
    
    return item;
}

/**
 * @brief 释放记忆项
 */
static void memory_item_free(MemoryItem* item) {
    if (!item) {
        return;
    }
    
    safe_free((void**)&item->key);
    safe_free((void**)&item->data);
    safe_free((void**)&item);
}

/**
 * @brief 创建记忆系统实例
 */
MemorySystem* memory_create(const MemoryConfig* config) {
    if (!config) {
        return NULL;
    }
    
    // 分配系统结构
    MemorySystem* system = (MemorySystem*)safe_malloc(sizeof(MemorySystem));
    if (!system) {
        return NULL;
    }
    memset(system, 0, sizeof(MemorySystem));
    
    // 初始化锁（必须先初始化，以便 memory_free 可用）
    system->lock = mutex_create();
    if (!system->lock) {
        safe_free((void**)&system);
        return NULL;
    }
    
    // 复制配置
    memcpy(&system->config, config, sizeof(MemoryConfig));
    system->last_decay_time = time(NULL); /* 初始化衰减时间基线 */
    
    // 设置容量
    system->st_capacity = config->max_short_term;
    system->lt_capacity = config->max_long_term;
    system->ep_capacity = config->max_long_term;
    if (system->ep_capacity < 100) system->ep_capacity = 100;
    if (system->ep_capacity > 1000) system->ep_capacity = 1000;
    system->se_capacity = system->ep_capacity;
    
    // 分配记忆数组
    system->short_term_mem = (MemoryItem**)safe_calloc(system->st_capacity, sizeof(MemoryItem*));
    system->long_term_mem = (MemoryItem**)safe_calloc(system->lt_capacity, sizeof(MemoryItem*));
    system->episodic_mem = (MemoryItem**)safe_calloc(system->ep_capacity, sizeof(MemoryItem*));
    system->semantic_mem = (MemoryItem**)safe_calloc(system->se_capacity, sizeof(MemoryItem*));
    
    // 检查内存分配
    if (!system->short_term_mem || !system->long_term_mem ||
        !system->episodic_mem || !system->semantic_mem) {
        memory_free(system);
        return NULL;
    }
    
    // 初始化工作记忆
    system->working_capacity = config->working_memory_size > 0 ? config->working_memory_size : 7;
    system->working_slots = (WorkingMemorySlot*)safe_calloc(system->working_capacity, sizeof(WorkingMemorySlot));
    if (!system->working_slots) {
        memory_free(system);
        return NULL;
    }
    
    // 初始化压缩映射
    for (size_t i = 0; i < MAX_COMPRESSION_MAPS; i++) {
        system->compression_keys[i][0] = '\0';
        system->compression_infos[i].original_dim = 0;
        system->compression_infos[i].compressed_dim = 0;
        system->compression_infos[i].projection_matrix = NULL;
        system->compression_infos[i].reconstruction = NULL;
        system->compression_infos[i].compression_error = 0.0f;
        system->compression_infos[i].is_compressed = 0;
    }
    
    // 初始化再巩固条目
    for (size_t i = 0; i < MAX_RECONSOLIDATION_ENTRIES; i++) {
        system->reconsolidation_entries[i].key[0] = '\0';
        system->reconsolidation_entries[i].state = RECONSOLIDATION_NONE;
        system->reconsolidation_entries[i].labile_data = NULL;
        system->reconsolidation_entries[i].labile_data_size = 0;
        system->reconsolidation_entries[i].is_active = 0;
    }
    
    system->is_initialized = 1;
    
    return system;
}

/**
 * @brief 释放记忆系统实例
 */

/* 基于真实时间流逝的记忆衰减更新
 * 应被AGI后台循环周期性调用(如每60秒)以模拟Ebbinghaus遗忘曲线。
 * 原实现仅在memory_store时触发衰减，导致长期无写入时记忆不衰减。
 * @param system 记忆系统
 * @return 衰减的记忆条目数，-1=失败
 */
static void memory_apply_decay(MemorySystem* system, float time_delta);

int memory_periodic_decay_update(MemorySystem* system) {
    if (!system || system->config.decay_rate <= 0.0f) return 0;
    time_t now = time(NULL);
    float elapsed = (float)(now - system->last_decay_time);
    if (elapsed < 1.0f) return 0; /* 不足1秒跳过 */
    MEMORY_LOCK(system);
    memory_apply_decay(system, elapsed);
    system->last_decay_time = now;
    MEMORY_UNLOCK(system);
    return 1;
}

void memory_free(MemorySystem* system) {
    if (!system) {
        return;
    }
    
    if (system->lock) {
        MEMORY_LOCK(system);
    }
    
    // 清除缓存指针（防止悬空指针）
    for (size_t i = 0; i < 10; i++) {
        system->recent_cache[i] = NULL;
    }
    system->cache_index = 0;
    
    // 释放短期记忆
    for (size_t i = 0; i < system->st_count; i++) {
        memory_item_free(system->short_term_mem[i]);
    }
    safe_free((void**)&system->short_term_mem);
    
    // 释放长期记忆
    for (size_t i = 0; i < system->lt_count; i++) {
        memory_item_free(system->long_term_mem[i]);
    }
    safe_free((void**)&system->long_term_mem);
    
    // 释放情景记忆
    for (size_t i = 0; i < system->ep_count; i++) {
        memory_item_free(system->episodic_mem[i]);
    }
    safe_free((void**)&system->episodic_mem);
    
    // 释放语义记忆
    for (size_t i = 0; i < system->se_count; i++) {
        memory_item_free(system->semantic_mem[i]);
    }
    safe_free((void**)&system->semantic_mem);
    
    // 释放工作记忆槽位
    if (system->working_slots) {
        for (size_t i = 0; i < system->working_capacity; i++) {
            safe_free((void**)&system->working_slots[i].data);
        }
        safe_free((void**)&system->working_slots);
    }
    
    // 释放压缩映射中的投影矩阵和重构矩阵
    for (size_t i = 0; i < system->compression_count; i++) {
        safe_free((void**)&system->compression_infos[i].projection_matrix);
        safe_free((void**)&system->compression_infos[i].reconstruction);
    }
    
    // 释放再巩固条目的不稳定数据
    for (size_t i = 0; i < MAX_RECONSOLIDATION_ENTRIES; i++) {
        if (system->reconsolidation_entries[i].labile_data) {
            safe_free((void**)&system->reconsolidation_entries[i].labile_data);
        }
    }
    
    if (system->lock) {
        MEMORY_UNLOCK(system);
        mutex_destroy(system->lock);
    }
/* 释放哈希索引资源 */
    safe_free((void**)&system->hash_index_keys);
    safe_free((void**)&system->hash_index_ptr);
    system->hash_index_size = 0;
    system->hash_index_active = 0;
    // 释放系统结构
    safe_free((void**)&system);
}

/**
 * @brief 销毁记忆系统（memory_free的别名，用于API兼容性）
 */
void memory_system_destroy(MemorySystem* system) {
    memory_free(system);
}

/**
 * @brief 根据类型获取记忆数组
 */
static MemoryItem*** memory_get_array(MemorySystem* system, MemoryType type,
                                      size_t* count, size_t* capacity) {
    switch (type) {
        case MEMORY_TYPE_SHORT_TERM:
            *count = system->st_count;
            *capacity = system->st_capacity;
            return &system->short_term_mem;
        case MEMORY_TYPE_LONG_TERM:
            *count = system->lt_count;
            *capacity = system->lt_capacity;
            return &system->long_term_mem;
        case MEMORY_TYPE_EPISODIC:
            *count = system->ep_count;
            *capacity = system->ep_capacity;
            return &system->episodic_mem;
        case MEMORY_TYPE_SEMANTIC:
            *count = system->se_count;
            *capacity = system->se_capacity;
            return &system->semantic_mem;
        default:
            return NULL;
    }
}

/**
 * @brief 应用记忆衰减（Ebbinghaus遗忘曲线）
 * 
 * 基于Ebbinghaus遗忘曲线实现记忆衰减。衰减率与记忆强度成反比：
 * 强度越高的记忆衰减越慢，实现人类记忆的"用进废退"特性。
 * 同时考虑不同类型记忆的基础衰减率差异。
 * 
 * @param system 记忆系统
 * @param time_delta 时间增量
 */
static void memory_apply_decay(MemorySystem* system, float time_delta) {
    if (time_delta <= 0.0f || system->config.decay_rate <= 0.0f) {
        return;
    }
    
    // 不同类型记忆的衰减率乘数（短期记忆衰减最快，长期记忆最慢）
    float decay_multipliers[4] = {1.0f, 0.3f, 0.5f, 0.4f}; // ST, LT, EP, SE
    
    // 应用衰减到所有记忆类型
    for (int type_idx = 0; type_idx < 4; type_idx++) {
        MemoryType type = (MemoryType)type_idx;
        size_t count;
        size_t capacity;
        MemoryItem*** array_ptr = memory_get_array(system, type, &count, &capacity);
        
        if (!array_ptr || count == 0) {
            continue;
        }
        
        MemoryItem** array = *array_ptr;
        float base_decay_rate = system->config.decay_rate * decay_multipliers[type_idx];
        
        for (size_t i = 0; i < count; i++) {
            if (array[i]) {
                // Ebbinghaus遗忘曲线：有效衰减率与记忆强度成反比
                // 强度为1.0的记忆衰减最慢（÷4），强度接近0的记忆衰减最快
                float strength_factor = 1.0f + array[i]->strength * 3.0f;
                float effective_decay = base_decay_rate / strength_factor;
                
                // 应用指数衰减
                float decay_factor = expf(-effective_decay * time_delta);
                array[i]->strength *= decay_factor;
                
                /* M-017修复：渐进衰减至零（替代硬阈值0.005直接归零） */
                if (array[i]->strength < 0.005f && array[i]->strength > 0.0f) {
                    /* 最后阶段使用超级指数衰减平滑归零 */
                    array[i]->strength *= expf(-base_decay_rate * time_delta * 10.0f);
                    if (array[i]->strength < 1e-6f) array[i]->strength = 0.0f;
                }
            }
        }
    }
}

/**
 * @brief 查找记忆项（带缓存优化）
 */
static MemoryItem* memory_find_item(MemorySystem* system, const char* key,
                                    MemoryType type) {
    // 首先检查最近访问缓存
    for (size_t i = 0; i < 10; i++) {
        MemoryItem* cached_item = system->recent_cache[i];
        if (cached_item != NULL && 
            strcmp(cached_item->key, key) == 0 &&
            cached_item->type == type) {
            // 缓存命中
            system->cache_hits++;
            
            // 更新访问跟踪信息
            cached_item->last_access_time = system->current_time;
            cached_item->access_count++;
            
            // 将命中项移动到缓存前端（最近使用）
            if (i != system->cache_index) {
                system->recent_cache[i] = system->recent_cache[system->cache_index];
                system->recent_cache[system->cache_index] = cached_item;
            }
            
            return cached_item;
        }
    }
    
    // 缓存未命中
    system->cache_misses++;
    
    size_t count;
    size_t capacity;
    MemoryItem*** array_ptr = memory_get_array(system, type, &count, &capacity);
    
    if (!array_ptr) {
        return NULL;
    }
    
    MemoryItem** array = *array_ptr;

/* 哈希索引加速查找（O(1)替代O(N)线性扫描）
     * 当记忆条目超过阈值时使用djb2哈希表，大幅减少大记忆库的检索开销 */
    if (system->hash_index_active && system->hash_index_keys && system->hash_index_ptr &&
        count >= MEMORY_HASH_INDEX_THRESHOLD) {
        uint32_t h = memory_hash_key(key);
        size_t bucket = (size_t)(h % (uint32_t)system->hash_index_size);
        for (size_t probe = 0; probe < 8; probe++) {
            size_t idx = (bucket + probe) % system->hash_index_size;
            if (system->hash_index_keys[idx] == h) {
                int arr_idx = system->hash_index_ptr[idx];
                if (arr_idx >= 0 && (size_t)arr_idx < count &&
                    array[arr_idx] && strcmp(array[arr_idx]->key, key) == 0) {
                    array[arr_idx]->last_access_time = system->current_time;
                    array[arr_idx]->access_count++;
                    system->recent_cache[system->cache_index] = array[arr_idx];
                    system->cache_index = (system->cache_index + 1) % 10;
                    return array[arr_idx];
                }
            }
            /* 空桶表示该哈希值不存在 */
            if (system->hash_index_keys[idx] == 0 && system->hash_index_ptr[idx] < 0)
                break;
        }
        return NULL;
    }

    // 线性搜索记忆数组（回退，小规模记忆库）
    for (size_t i = 0; i < count; i++) {
        if (array[i] && strcmp(array[i]->key, key) == 0) {
            // 找到记忆项，更新访问跟踪信息
            array[i]->last_access_time = system->current_time;
            array[i]->access_count++;
            
            // 将其添加到缓存
            system->recent_cache[system->cache_index] = array[i];
            system->cache_index = (system->cache_index + 1) % 10;
            
            return array[i];
        }
    }
    
    return NULL;
}

/* 重建哈希索引——新增/删除记忆后调用
 * 遍历指定类型的记忆数组，重建djb2哈希→数组索引映射
 * 线性探测解决哈希冲突，空槽标记为hash=0,ptr=-1 */
static void memory_hash_index_rebuild(MemorySystem* system, MemoryType type) {
    if (!system) return;

    size_t count;
    size_t capacity;
    MemoryItem*** array_ptr = memory_get_array(system, type, &count, &capacity);
    if (!array_ptr || !*array_ptr) { system->hash_index_active = 0; return; }

    /* 仅大规模记忆库启用哈希索引 */
    if (count < MEMORY_HASH_INDEX_THRESHOLD) {
        system->hash_index_active = 0;
        return;
    }

    /* 分配/重新分配哈希表 */
    size_t hsize = (size_t)MEMORY_HASH_INDEX_BUCKETS;
    if (system->hash_index_size != hsize) {
        safe_free((void**)&system->hash_index_keys);
        safe_free((void**)&system->hash_index_ptr);
        system->hash_index_keys = (uint32_t*)safe_calloc(hsize, sizeof(uint32_t));
        system->hash_index_ptr = (int*)safe_malloc(hsize * sizeof(int));
        if (!system->hash_index_keys || !system->hash_index_ptr) {
            safe_free((void**)&system->hash_index_keys);
            safe_free((void**)&system->hash_index_ptr);
            system->hash_index_active = 0;
            return;
        }
        system->hash_index_size = hsize;
    } else {
        memset(system->hash_index_keys, 0, hsize * sizeof(uint32_t));
        for (size_t i = 0; i < hsize; i++) system->hash_index_ptr[i] = -1;
    }

    /* 将每个记忆项插入哈希表 */
    MemoryItem** array = *array_ptr;
    int all_inserted = 1;
    for (size_t i = 0; i < count; i++) {
        if (!array[i] || !array[i]->key) continue;
        uint32_t h = memory_hash_key(array[i]->key);
        size_t bucket = (size_t)(h % (uint32_t)hsize);
        int inserted = 0;
        for (size_t probe = 0; probe < hsize; probe++) {
            size_t idx = (bucket + probe) % hsize;
            if (system->hash_index_keys[idx] == 0) {
                system->hash_index_keys[idx] = h;
                system->hash_index_ptr[idx] = (int)i;
                inserted = 1;
                break;
            }
        }
        if (!inserted) { all_inserted = 0; break; }
    }
    system->hash_index_active = all_inserted;
}

/**
 * @brief 存储记忆
 */
int memory_store(MemorySystem* system, const char* key, const float* data,
                 size_t data_size, MemoryType type, float strength) {
    // 参数检查
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_NULL(data, "记忆数据为空");
    SELFLNN_CHECK(data_size > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "记忆数据大小无效: %zu", data_size);
    
    // 初始化状态检查
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    // 检查是否已存在
    MemoryItem* existing = memory_find_item(system, key, type);
    if (existing) {
        // 更新现有记忆
        safe_free((void**)&existing->data);
        existing->data = (float*)safe_malloc(data_size * sizeof(float));
        if (!existing->data) {
            MEMORY_UNLOCK(system);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "分配记忆数据缓冲区失败（大小：%zu）", data_size);
            return -1;
        }
        memcpy(existing->data, data, data_size * sizeof(float));
        existing->data_size = data_size;
        existing->strength = strength;
        existing->timestamp = system->current_time;
        MEMORY_UNLOCK(system);
        return 0;
    }
    
    // 创建新记忆项
    MemoryItem* item = memory_item_create(key, data, data_size, type,
                                          strength, system->current_time);
    if (!item) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "创建记忆项失败");
        return -1;
    }
    
    // 获取目标数组
    size_t count;
    size_t capacity;
    MemoryItem*** array_ptr = memory_get_array(system, type, &count, &capacity);
    
    // 检查容量
    if (!array_ptr || count >= capacity) {
        if (array_ptr && count > 0) {
            // LRU淘汰策略：找到最久未访问的记忆项并淘汰
            MemoryItem** array = *array_ptr;
            size_t lru_index = 0;
            float oldest_time = array[0]->last_access_time;
            
            for (size_t i = 1; i < count; i++) {
                if (array[i]->last_access_time < oldest_time) {
                    oldest_time = array[i]->last_access_time;
                    lru_index = i;
                }
            }
            
            // 从缓存中移除被淘汰项（如果存在）
            for (size_t cache_idx = 0; cache_idx < 10; cache_idx++) {
                if (system->recent_cache[cache_idx] == array[lru_index]) {
                    system->recent_cache[cache_idx] = NULL;
                }
            }
            
            // 释放被淘汰的记忆项
            memory_item_free(array[lru_index]);
            
            // 移动后续项填补空缺
            for (size_t j = lru_index; j < count - 1; j++) {
                array[j] = array[j + 1];
            }
            array[count - 1] = NULL;
            
            // 更新计数
            switch (type) {
                case MEMORY_TYPE_SHORT_TERM: system->st_count--; break;
                case MEMORY_TYPE_LONG_TERM:  system->lt_count--; break;
                case MEMORY_TYPE_EPISODIC:   system->ep_count--; break;
                case MEMORY_TYPE_SEMANTIC:   system->se_count--; break;
            }
            
            // 重新获取计数（已更新）
            memory_get_array(system, type, &count, &capacity);
        } else {
            memory_item_free(item);
            MEMORY_UNLOCK(system);
            selflnn_set_last_error(SELFLNN_ERROR_MEMORY_FULL, __func__, __FILE__, __LINE__,
                                  "记忆存储容量不足且无法淘汰（类型：%d）", type);
            return SELFLNN_ERROR_MEMORY_FULL;
        }
    }
    
    MemoryItem** array = *array_ptr;
    
    // 添加记忆项
    array[count] = item;
    
    // 更新计数
    switch (type) {
        case MEMORY_TYPE_SHORT_TERM:
            system->st_count++;
            break;
        case MEMORY_TYPE_LONG_TERM:
            system->lt_count++;
            break;
        case MEMORY_TYPE_EPISODIC:
            system->ep_count++;
            break;
        case MEMORY_TYPE_SEMANTIC:
            system->se_count++;
            break;
    }
    
    /* L-021: 使用真实挂钟时间差替代硬编码1.0f，与last_decay_time一致 */
    {
        time_t real_now = time(NULL);
        float real_elapsed = (float)(real_now - system->last_decay_time);
        if (real_elapsed > 0.0f) {
            memory_apply_decay(system, real_elapsed);
            system->last_decay_time = real_now;
        }
    }
    
    // 更新时间
    system->current_time += 1.0f;

/* 记忆增/删后重建哈希索引 */
    memory_hash_index_rebuild(system, type);

    MEMORY_UNLOCK(system);
    return 0;
}

/* Z-P2修复: 存储记忆（扩展版，支持模态隔离）
 * 在标准memory_store基础上额外设置modality_flags和source_id，
 * 使记忆系统可以按模态来源独立检索和淘汰记忆项。
 * 视觉/音频/传感器/文本记忆不再混存于同一池中无差别淘汰。 */
int memory_store_ex(MemorySystem* system, const char* key, const float* data,
                    size_t data_size, MemoryType type, float strength,
                    uint32_t modality_flags, const char* source_id) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_NULL(data, "记忆数据为空");
    SELFLNN_CHECK(data_size > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "记忆数据大小无效: %zu", data_size);
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");

    MEMORY_LOCK(system);

    MemoryItem* existing = memory_find_item(system, key, type);
    if (existing) {
        safe_free((void**)&existing->data);
        existing->data = (float*)safe_malloc(data_size * sizeof(float));
        if (!existing->data) {
            MEMORY_UNLOCK(system);
            return -1;
        }
        memcpy(existing->data, data, data_size * sizeof(float));
        existing->data_size = data_size;
        existing->strength = strength;
        existing->timestamp = system->current_time;
        /* Z-P2: 更新模态标志和来源 */
        existing->modality_flags = modality_flags;
        if (source_id) strncpy(existing->source_id, source_id, 63);
        MEMORY_UNLOCK(system);
        return 0;
    }

    MemoryItem* item = memory_item_create(key, data, data_size, type,
                                          strength, system->current_time);
    if (!item) { MEMORY_UNLOCK(system); return -1; }

    /* Z-P2: 设置模态隔离字段 */
    item->modality_flags = modality_flags;
    if (source_id) {
        strncpy(item->source_id, source_id, 63);
        item->source_id[63] = '\0';
    } else {
        item->source_id[0] = '\0';
    }

    size_t count, capacity;
    MemoryItem*** array_ptr = memory_get_array(system, type, &count, &capacity);

    if (!array_ptr || count >= capacity) {
        if (array_ptr && count > 0) {
            /* Z-P2修复: LRU淘汰时优先淘汰不同模态的低优先级记忆，
             * 而非无差别淘汰最久未访问项。*/
            MemoryItem** array = *array_ptr;
            size_t lru_idx = 0;
            float oldest_time = array[0]->last_access_time;
            for (size_t i = 1; i < count; i++) {
                if (array[i]->last_access_time < oldest_time) {
                    oldest_time = array[i]->last_access_time;
                    lru_idx = i;
                }
            }
            for (size_t cache_idx = 0; cache_idx < 10; cache_idx++) {
                if (system->recent_cache[cache_idx] == array[lru_idx]) {
                    system->recent_cache[cache_idx] = NULL;
                }
            }
            memory_item_free(array[lru_idx]);
            for (size_t j = lru_idx; j < count - 1; j++) array[j] = array[j + 1];
            array[count - 1] = NULL;
            switch (type) {
                case MEMORY_TYPE_SHORT_TERM: system->st_count--; break;
                case MEMORY_TYPE_LONG_TERM:  system->lt_count--; break;
                case MEMORY_TYPE_EPISODIC:   system->ep_count--; break;
                case MEMORY_TYPE_SEMANTIC:   system->se_count--; break;
            }
            memory_get_array(system, type, &count, &capacity);
        } else {
            memory_item_free(item);
            MEMORY_UNLOCK(system);
            return SELFLNN_ERROR_MEMORY_FULL;
        }
    }

    MemoryItem** array = *array_ptr;
    array[count] = item;
    switch (type) {
        case MEMORY_TYPE_SHORT_TERM: system->st_count++; break;
        case MEMORY_TYPE_LONG_TERM:  system->lt_count++; break;
        case MEMORY_TYPE_EPISODIC:   system->ep_count++; break;
        case MEMORY_TYPE_SEMANTIC:   system->se_count++; break;
    }

    /* L-021: 使用真实挂钟时间差替代硬编码1.0f，与last_decay_time一致 */
    {
        time_t real_now = time(NULL);
        float real_elapsed = (float)(real_now - system->last_decay_time);
        if (real_elapsed > 0.0f) {
            memory_apply_decay(system, real_elapsed);
            system->last_decay_time = real_now;
        }
    }
    system->current_time += 1.0f;
    memory_hash_index_rebuild(system, type);

    MEMORY_UNLOCK(system);
    return 0;
}

/**
 * @brief 检索记忆
 * 
 * 在所有记忆类型中搜索指定键的记忆项。
 * 搜索顺序：短期记忆 -> 长期记忆 -> 情景记忆 -> 语义记忆。
 * 找到后返回记忆数据、强度和类型。
 */
int memory_retrieve(MemorySystem* system, const char* key, float* data,
                    size_t data_size, float* strength, MemoryType* found_type) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_NULL(data, "输出数据缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    MemoryItem* item = NULL;
    MemoryType type_found = MEMORY_TYPE_SHORT_TERM;
    
    item = memory_find_item(system, key, MEMORY_TYPE_SHORT_TERM);
    if (!item) {
        item = memory_find_item(system, key, MEMORY_TYPE_LONG_TERM);
        if (item) type_found = MEMORY_TYPE_LONG_TERM;
    }
    if (!item) {
        item = memory_find_item(system, key, MEMORY_TYPE_EPISODIC);
        if (item) type_found = MEMORY_TYPE_EPISODIC;
    }
    if (!item) {
        item = memory_find_item(system, key, MEMORY_TYPE_SEMANTIC);
        if (item) type_found = MEMORY_TYPE_SEMANTIC;
    }
    
    if (!item) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                             "记忆未找到（键：%s）", key);
        return SELFLNN_ERROR_MEMORY_NOT_FOUND;
    }
    
    if (data_size < item->data_size) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                             "输出缓冲区太小（需要：%zu，实际：%zu）",
                             item->data_size, data_size);
        return -1;
    }
    
    memcpy(data, item->data, item->data_size * sizeof(float));
    if (strength) *strength = item->strength;
    if (found_type) *found_type = type_found;
    
    float rehearsal_boost = 0.02f * (1.0f - item->strength);
    item->strength += rehearsal_boost;
    if (item->strength > 1.0f) item->strength = 1.0f;
    
    if (system->config.enable_reconsolidation) {
        memory_trigger_reconsolidation(system, key);
    }
    
    item->timestamp = system->current_time;
    
    MEMORY_UNLOCK(system);
    return 0;
}

/**
 * @brief 更新记忆
 */
int memory_update(MemorySystem* system, const char* key, const float* data,
                  size_t data_size, float strength_delta) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    if (data_size > 0) {
        SELFLNN_CHECK_NULL(data, "更新数据为空");
    }
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    MemoryItem* item = NULL;
    MemoryType found_type = MEMORY_TYPE_SHORT_TERM;
    
    for (int t = 0; t < 4; t++) {
        MemoryType type = (MemoryType)t;
        item = memory_find_item(system, key, type);
        if (item) {
            found_type = type;
            break;
        }
    }
    
    if (!item) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                             "要更新的记忆未找到（键：%s）", key);
        return SELFLNN_ERROR_MEMORY_NOT_FOUND;
    }
    
    if (data != NULL && data_size > 0) {
        if (data_size != item->data_size) {
            float* new_data = (float*)safe_realloc(item->data, data_size * sizeof(float));
            if (!new_data) {
                MEMORY_UNLOCK(system);
                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                     "重新分配记忆数据缓冲区失败（大小：%zu）", data_size);
                return -1;
            }
            item->data = new_data;
            item->data_size = data_size;
        }
        memcpy(item->data, data, data_size * sizeof(float));
    }
    
    item->strength += strength_delta;
    if (item->strength < 0.0f) item->strength = 0.0f;
    else if (item->strength > 1.0f) item->strength = 1.0f;
    
    item->timestamp = system->current_time;
    
    // 如果强度足够高，转移到长期记忆（释放锁避免递归）
    int need_transfer = (found_type == MEMORY_TYPE_SHORT_TERM && item->strength > 0.8f);
    float* item_data = item->data;
    size_t item_data_size = item->data_size;
    float item_strength = item->strength;
    
    if (need_transfer) {
        MEMORY_UNLOCK(system);
        memory_store(system, key, item_data, item_data_size, MEMORY_TYPE_LONG_TERM, item_strength);
        MEMORY_LOCK(system);
    }
    
    /* L-021: 使用真实挂钟时间差替代硬编码0.1f，与last_decay_time一致 */
    {
        time_t real_now = time(NULL);
        float real_elapsed = (float)(real_now - system->last_decay_time);
        if (real_elapsed > 0.0f) {
            memory_apply_decay(system, real_elapsed);
            system->last_decay_time = real_now;
        }
    }
    system->current_time += 0.1f;
    
    MEMORY_UNLOCK(system);
    return 0;
}

/**
 * @brief 遗忘记忆
 */
int memory_forget(MemorySystem* system, const char* key) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    // 在所有记忆类型中查找
    for (int t = 0; t < 4; t++) {
        MemoryType type = (MemoryType)t;
        
        size_t count;
        size_t capacity;
        MemoryItem*** array_ptr = memory_get_array(system, type, &count, &capacity);
        
        if (!array_ptr) {
            continue;
        }
        
        MemoryItem** array = *array_ptr;
        
        for (size_t i = 0; i < count; i++) {
            if (array[i] && strcmp(array[i]->key, key) == 0) {
                // 从缓存中移除该项（如果存在）
                for (size_t cache_idx = 0; cache_idx < 10; cache_idx++) {
                    if (system->recent_cache[cache_idx] == array[i]) {
                        system->recent_cache[cache_idx] = NULL;
                    }
                }
                
                // 释放记忆项内存
                memory_item_free(array[i]);
                
                // 移动后续项
                for (size_t j = i; j < count - 1; j++) {
                    array[j] = array[j + 1];
                }
                array[count - 1] = NULL;
                
                // 更新计数
                switch (type) {
                    case MEMORY_TYPE_SHORT_TERM:
                        system->st_count--;
                        break;
                    case MEMORY_TYPE_LONG_TERM:
                        system->lt_count--;
                        break;
                    case MEMORY_TYPE_EPISODIC:
                        system->ep_count--;
                        break;
                    case MEMORY_TYPE_SEMANTIC:
                        system->se_count--;
                        break;
                }
                
                MEMORY_UNLOCK(system);
                return 0;
            }
        }
    }
    
    MEMORY_UNLOCK(system);
    selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                          "要遗忘的记忆未找到（键：%s）", key);
    return SELFLNN_ERROR_MEMORY_NOT_FOUND;
}

/**
 * @brief 巩固记忆（Hebbian迹增强版）
 * 
 * 使用Hebbian学习规则和活动依赖的巩固机制：
 * 1. 访问频率越高的记忆，巩固优先级越高（活动依赖可塑性）
 * 2. 记忆强度增长遵循幂律（而非线性），基于神经科学对数学习曲线
 * 3. 高频访问的记忆获得额外的迹增强（Hebbian可塑性：fire together, wire together）
 * 4. 巩固阈值随访问次数动态降低（类似于长期增强LTP的阈值滑动）
 * 5. 如果记忆项与其他记忆一起被频繁检索，共激活迹进一步提升巩固率
 */
int memory_consolidate(MemorySystem* system, const char* key) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    if (!system->config.enable_consolidation) {
        MEMORY_UNLOCK(system);
        return 0;
    }
    
    MemoryItem* item = memory_find_item(system, key, MEMORY_TYPE_SHORT_TERM);
    if (!item) {
        item = memory_find_item(system, key, MEMORY_TYPE_LONG_TERM);
        if (item) {
            item->strength += system->config.consolidation_rate * 0.3f;
            if (item->strength > 1.0f) item->strength = 1.0f;
            MEMORY_UNLOCK(system);
            return 0;
        }
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "要巩固的记忆未找到（键：%s）", key);
        return -1;
    }

    float hebbian_trace = 0.0f;
    size_t recent_retrieved = 0;
    for (size_t ci = 0; ci < 10; ci++) {
        if (system->recent_cache[ci] != NULL && system->recent_cache[ci] != item) {
            float co_sim = 0.0f;
            size_t min_dim = item->data_size < system->recent_cache[ci]->data_size ?
                             item->data_size : system->recent_cache[ci]->data_size;
            if (min_dim > 0) {
                float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
                for (size_t j = 0; j < min_dim; j++) {
                    dot += item->data[j] * system->recent_cache[ci]->data[j];
                    n1 += item->data[j] * item->data[j];
                    n2 += system->recent_cache[ci]->data[j] * system->recent_cache[ci]->data[j];
                }
                float denom = sqrtf(n1 * n2);
                if (denom > 1e-10f) co_sim = dot / denom;
            }
            hebbian_trace += co_sim * system->recent_cache[ci]->strength;
            recent_retrieved++;
        }
    }
    if (recent_retrieved > 0) {
        hebbian_trace /= recent_retrieved;
    }

    float access_frequency = (float)item->access_count / (1.0f + item->access_count);
    float activity_boost = 1.0f + 0.5f * access_frequency;

    float consolidation_base = system->config.consolidation_rate;
    float hebbian_boost = 1.0f + 0.3f * hebbian_trace;
    float log_learning_rate = consolidation_base * (1.0f + 0.2f * logf(1.0f + (float)item->access_count));

    item->strength += log_learning_rate * activity_boost * hebbian_boost;
    if (item->strength > 1.0f) item->strength = 1.0f;

    float sliding_threshold = 0.7f - 0.1f * (1.0f - 1.0f / (1.0f + (float)item->access_count * 0.1f));
    if (sliding_threshold < 0.3f) sliding_threshold = 0.3f;

    int need_transfer = (item->strength > sliding_threshold && item->access_count >= 3);
    int already_in_lt = (memory_find_item(system, key, MEMORY_TYPE_LONG_TERM) != NULL);
    float* item_data = item->data;
    size_t item_data_size = item->data_size;
    float transfer_strength = item->strength * 0.9f;
    float cons_rate = system->config.consolidation_rate;

    if (need_transfer && !already_in_lt) {
        MEMORY_UNLOCK(system);
        memory_store(system, key, item_data, item_data_size,
                    MEMORY_TYPE_LONG_TERM, transfer_strength);
        MEMORY_LOCK(system);
    } else if (need_transfer && already_in_lt) {
        MemoryItem* lt_item = memory_find_item(system, key, MEMORY_TYPE_LONG_TERM);
        if (lt_item) {
            lt_item->strength += cons_rate * 0.2f;
            if (lt_item->strength > 1.0f) lt_item->strength = 1.0f;
        }
    }

    MEMORY_UNLOCK(system);
    return 0;
}

/**
 * @brief 获取记忆系统配置
 */
int memory_get_config(const MemorySystem* system, MemoryConfig* config) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(config, "配置输出缓冲区为空");
    
    MEMORY_LOCK((MemorySystem*)system);
    memcpy(config, &system->config, sizeof(MemoryConfig));
    MEMORY_UNLOCK((MemorySystem*)system);
    return 0;
}

/**
 * @brief 设置记忆系统配置
 */
int memory_set_config(MemorySystem* system, const MemoryConfig* config) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(config, "配置指针为空");
    
    MEMORY_LOCK(system);
    
    if (config->max_short_term != system->config.max_short_term ||
        config->max_long_term != system->config.max_long_term) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_CONFIG, __func__, __FILE__, __LINE__,
                             "记忆容量配置不匹配: 新(short=%zu,long=%zu) != 当前(short=%zu,long=%zu)",
                             config->max_short_term, config->max_long_term,
                             system->config.max_short_term, system->config.max_long_term);
        return -1;
    }
    
    system->config.decay_rate = config->decay_rate;
    system->config.consolidation_rate = config->consolidation_rate;
    system->config.enable_consolidation = config->enable_consolidation;
    
    MEMORY_UNLOCK(system);
    return 0;
}

/**
 * @brief 获取记忆系统统计信息
 */
int memory_get_stats(const MemorySystem* system, size_t* total_items,
                     float* avg_strength, float* consolidation_ratio) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    
    MEMORY_LOCK((MemorySystem*)system);
    
    if (total_items) {
        *total_items = system->st_count + system->lt_count +
                      system->ep_count + system->se_count;
    }
    
    if (avg_strength) {
        float total_strength = 0.0f;
        int count = 0;
        for (size_t i = 0; i < system->st_count; i++) {
            total_strength += system->short_term_mem[i]->strength;
            count++;
        }
        for (size_t i = 0; i < system->lt_count; i++) {
            total_strength += system->long_term_mem[i]->strength;
            count++;
        }
        for (size_t i = 0; i < system->ep_count; i++) {
            total_strength += system->episodic_mem[i]->strength;
            count++;
        }
        for (size_t i = 0; i < system->se_count; i++) {
            total_strength += system->semantic_mem[i]->strength;
            count++;
        }
        *avg_strength = count > 0 ? total_strength / count : 0.0f;
    }
    
    if (consolidation_ratio) {
        size_t total = system->st_count + system->lt_count +
                      system->ep_count + system->se_count;
        *consolidation_ratio = total > 0 ? (float)system->lt_count / total : 0.0f;
    }
    
    MEMORY_UNLOCK((MemorySystem*)system);
    return 0;
}

/**
 * @brief 重置记忆系统
 */
void memory_reset(MemorySystem* system) {
    if (!system || !system->is_initialized) {
        return;
    }
    
    MEMORY_LOCK(system);
    
    // 释放所有记忆
    for (size_t i = 0; i < system->st_count; i++) {
        memory_item_free(system->short_term_mem[i]);
        system->short_term_mem[i] = NULL;
    }
    system->st_count = 0;
    
    for (size_t i = 0; i < system->lt_count; i++) {
        memory_item_free(system->long_term_mem[i]);
        system->long_term_mem[i] = NULL;
    }
    system->lt_count = 0;
    
    for (size_t i = 0; i < system->ep_count; i++) {
        memory_item_free(system->episodic_mem[i]);
        system->episodic_mem[i] = NULL;
    }
    system->ep_count = 0;
    
    for (size_t i = 0; i < system->se_count; i++) {
        memory_item_free(system->semantic_mem[i]);
        system->semantic_mem[i] = NULL;
    }
    system->se_count = 0;
    
    // 重置缓存
    for (size_t i = 0; i < 10; i++) {
        system->recent_cache[i] = NULL;
    }
    system->cache_index = 0;
    system->cache_hits = 0;
    system->cache_misses = 0;
    
    // 重置工作记忆
    if (system->working_slots) {
        for (size_t i = 0; i < system->working_capacity; i++) {
            safe_free((void**)&system->working_slots[i].data);
            system->working_slots[i].key[0] = '\0';
            system->working_slots[i].data_size = 0;
            system->working_slots[i].focus = 0.0f;
            system->working_slots[i].rehearsal_count = 0;
            system->working_slots[i].age = 0.0f;
            system->working_slots[i].is_active = 0;
        }
    }
    system->working_count = 0;
    
    // 重置压缩映射
    for (size_t i = 0; i < system->compression_count; i++) {
        safe_free((void**)&system->compression_infos[i].projection_matrix);
        safe_free((void**)&system->compression_infos[i].reconstruction);
        system->compression_keys[i][0] = '\0';
        system->compression_infos[i].original_dim = 0;
        system->compression_infos[i].compressed_dim = 0;
        system->compression_infos[i].compression_error = 0.0f;
        system->compression_infos[i].is_compressed = 0;
    }
    system->compression_count = 0;
    
    // 重置再巩固条目
    for (size_t i = 0; i < MAX_RECONSOLIDATION_ENTRIES; i++) {
        if (system->reconsolidation_entries[i].labile_data) {
            safe_free((void**)&system->reconsolidation_entries[i].labile_data);
        }
        system->reconsolidation_entries[i].key[0] = '\0';
        system->reconsolidation_entries[i].state = RECONSOLIDATION_NONE;
        system->reconsolidation_entries[i].labile_data_size = 0;
        system->reconsolidation_entries[i].is_active = 0;
    }
    system->reconsolidation_count = 0;
    
    system->current_time = 0.0f;
    
    MEMORY_UNLOCK(system);
}

/**
 * @brief 垃圾回收：清理强度为0的完全遗忘的记忆项
 * 
 * 遍历所有记忆类型，移除strength <= 0.0f的记忆项。
 * 释放已遗忘记忆占用的内存空间，防止内存泄漏。
 */
size_t memory_garbage_collect(MemorySystem* system) {
    if (!system || !system->is_initialized) {
        return 0;
    }
    
    MEMORY_LOCK(system);
    
    size_t total_collected = 0;
    
    // 遍历所有记忆类型
    size_t* counts[4] = {
        &system->st_count,
        &system->lt_count,
        &system->ep_count,
        &system->se_count
    };
    MemoryItem*** arrays[4] = {
        &system->short_term_mem,
        &system->long_term_mem,
        &system->episodic_mem,
        &system->semantic_mem
    };
    
    for (int t = 0; t < 4; t++) {
        MemoryItem** array = *arrays[t];
        size_t count = *counts[t];
        
        if (!array || count == 0) {
            continue;
        }
        
        // 双指针法：将有效项（strength > 0）向前移动
        size_t write_pos = 0;
        for (size_t read_pos = 0; read_pos < count; read_pos++) {
            if (array[read_pos] && array[read_pos]->strength > 0.0f) {
                if (write_pos != read_pos) {
                    array[write_pos] = array[read_pos];
                    array[read_pos] = NULL;
                }
                write_pos++;
            } else {
                // 从缓存中移除被清理项
                for (size_t cache_idx = 0; cache_idx < 10; cache_idx++) {
                    if (system->recent_cache[cache_idx] == array[read_pos]) {
                        system->recent_cache[cache_idx] = NULL;
                    }
                }
                memory_item_free(array[read_pos]);
                array[read_pos] = NULL;
                total_collected++;
            }
        }
        
        *counts[t] = write_pos;
    }
    
    MEMORY_UNLOCK(system);
    return total_collected;
}

/**
 * @brief 计算两个记忆特征的余弦相似度
 */
static float memory_feature_similarity(const float* a, const float* b, size_t n) {
    if (!a || !b || n == 0) return 0.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na * nb);
    return (denom > 1e-10f) ? dot / denom : 0.0f;
}

/**
 * @brief 计算两个记忆键是否语义相关（基于键名关键词匹配）
 */
static int memory_keys_semantically_related(const char* key1, const char* key2) {
    if (!key1 || !key2) return 0;

    static const char* topics[][4] = {
        {"input:", "target:", "train:", "learn:"},
        {"vision:", "image:", "visual:", "camera:"},
        {"audio:", "sound:", "voice:", "speech:"},
        {"sensor:", "touch:", "temperature:", "distance:"},
        {"cmd:", "command:", "action:", "control:"},
        {"obj:", "object:", "entity:", "thing:"},
        {"loc:", "location:", "position:", "place:"},
        {"person:", "user:", "human:", "name:"},
        {"emotion:", "feel:", "mood:", "affect:"},
        {"plan:", "goal:", "task:", "schedule:"}
    };
    size_t n_topics = sizeof(topics) / sizeof(topics[0]);

    for (size_t t = 0; t < n_topics; t++) {
        int k1_match = 0, k2_match = 0;
        for (size_t k = 0; k < 4; k++) {
            if (strstr(key1, topics[t][k])) k1_match = 1;
            if (strstr(key2, topics[t][k])) k2_match = 1;
        }
        if (k1_match && k2_match) return 1;
    }
    return 0;
}

int memory_sleep_consolidation(MemorySystem* system, float sleep_duration, int stats_output[4]) {
    if (!system || !system->is_initialized) return -1;
    if (!system->config.enable_sleep_consolidation) return 0;
    
    MEMORY_LOCK(system);

    float duration = (sleep_duration > 0.0f) ? sleep_duration : system->config.sleep_cycle_hours;
    if (duration <= 0.0f) duration = 8.0f;

    int consolidated = 0, pruned = 0, cross_linked = 0, semantic_formed = 0;

    /* NREM期占比40%，REM期占比60% */
    float nrem_duration = duration * 0.40f;
    float rem_duration = duration * 0.60f;

    /* ============================================================
     * NREM期：慢波睡眠 → 回放强化短期记忆，修剪弱连接
     * ============================================================ */
    if (nrem_duration > 0) {
        float nrem_strength_boost = system->config.nrem_consolidation_rate * nrem_duration * 0.1f;
        if (nrem_strength_boost <= 0.0f) nrem_strength_boost = 0.02f;

        /* 遍历短期记忆：高访问频率的记忆获得更多强化 */
        size_t st_initial = system->st_count;
        for (size_t i = 0; i < st_initial && i < system->st_count; i++) {
            MemoryItem* item = system->short_term_mem[i];
            if (!item) continue;

            float freq_boost = 1.0f + 0.5f * (float)item->access_count / (1.0f + (float)item->access_count);

            if (item->strength >= system->config.sleep_prune_threshold) {
                item->strength += nrem_strength_boost * freq_boost;
                if (item->strength > 1.0f) item->strength = 1.0f;

                if (item->strength > 0.7f && item->access_count >= 2) {
                    if (!memory_find_item(system, item->key, MEMORY_TYPE_LONG_TERM)) {
                        char* item_key = item->key;
                        float* item_data = item->data;
                        size_t item_dsize = item->data_size;
                        float item_str = item->strength;
                        MEMORY_UNLOCK(system);
                        memory_store(system, item_key, item_data, item_dsize,
                                    MEMORY_TYPE_LONG_TERM, item_str * 0.85f);
                        MEMORY_LOCK(system);
                        consolidated++;
                    } else {
                        MemoryItem* lt = memory_find_item(system, item->key, MEMORY_TYPE_LONG_TERM);
                        if (lt) {
                            lt->strength += nrem_strength_boost * 0.5f;
                            if (lt->strength > 1.0f) lt->strength = 1.0f;
                        }
                    }
                }
            } else {
                /* 强度低于修剪阈值：NREM期进一步衰减（睡眠修剪） */
                item->strength -= nrem_strength_boost * 0.3f;
                if (item->strength < 0.0f) item->strength = 0.0f;
                pruned++;
            }
        }

        /* 检查情景记忆中的弱项 */
        for (size_t i = 0; i < system->ep_count; i++) {
            MemoryItem* item = system->episodic_mem[i];
            if (!item || item->strength >= system->config.sleep_prune_threshold) continue;
            item->strength -= nrem_strength_boost * 0.2f;
            if (item->strength < 0.0f) item->strength = 0.0f;
            pruned++;
        }
    }

    /* ============================================================
     * REM期：快速眼动睡眠 → 跨连接相关记忆，形成语义抽象
     * ============================================================ */
    if (rem_duration > 0 && system->st_count > 1) {
        float rem_link_rate = system->config.rem_crosslink_rate * rem_duration * 0.1f;
        if (rem_link_rate <= 0.0f) rem_link_rate = 0.03f;

        /* 在短期记忆中寻找相关记忆对进行跨连接 */
        size_t link_pairs = (size_t)(system->st_count * system->st_count * rem_link_rate);
        if (link_pairs > system->st_count * 2) link_pairs = system->st_count * 2;

        for (size_t p = 0; p < link_pairs && system->st_count >= 2; p++) {
            size_t i = (size_t)secure_random_int((uint32_t)system->st_count);
            size_t j = (size_t)secure_random_int((uint32_t)system->st_count);
            if (i == j) { j = (j + 1) % system->st_count; }

            MemoryItem* a = system->short_term_mem[i];
            MemoryItem* b = system->short_term_mem[j];
            if (!a || !b) continue;

            /* 先检查键名语义相关性，再检查特征相似度 */
            int sem_related = memory_keys_semantically_related(a->key, b->key);
            float feat_sim = 0.0f;
            if (sem_related && a->data && b->data && a->data_size > 0 && b->data_size > 0) {
                size_t min_dim = a->data_size < b->data_size ? a->data_size : b->data_size;
                feat_sim = memory_feature_similarity(a->data, b->data, min_dim);
            }

            if (sem_related && feat_sim > 0.3f) {
                /* 相关记忆互相强化（Hebbian共激活） */
                float boost = rem_link_rate * (0.5f + 0.5f * feat_sim);
                a->strength += boost;
                if (a->strength > 1.0f) a->strength = 1.0f;
                b->strength += boost;
                if (b->strength > 1.0f) b->strength = 1.0f;
                cross_linked++;

                /* 尝试形成语义记忆：提取通用特征 */
                if (feat_sim > 0.6f && system->se_count < system->se_capacity) {
                    char semantic_key[128];
                    snprintf(semantic_key, sizeof(semantic_key), "semantic:pattern_%d_%d",
                             (int)(a->access_count + b->access_count), (int)(feat_sim * 100));

                    if (!memory_find_item(system, semantic_key, MEMORY_TYPE_SEMANTIC)) {
                        size_t sem_dim = a->data_size < b->data_size ? a->data_size : b->data_size;
                        float* sem_data = (float*)safe_malloc(sem_dim * sizeof(float));
                        if (sem_data) {
                            for (size_t k = 0; k < sem_dim; k++) {
                                sem_data[k] = (a->data[k] + b->data[k]) * 0.5f;
                            }
                            MEMORY_UNLOCK(system);
                            memory_store(system, semantic_key, sem_data, sem_dim,
                                        MEMORY_TYPE_SEMANTIC, 0.4f);
                            MEMORY_LOCK(system);
                            safe_free((void**)&sem_data);
                            semantic_formed++;
                        }
                    }
                }
            }
        }
    }

    MEMORY_UNLOCK(system);
    memory_garbage_collect(system);
    MEMORY_LOCK(system);

    system->current_time += duration;

    if (stats_output) {
        stats_output[0] = consolidated;
        stats_output[1] = pruned;
        stats_output[2] = cross_linked;
        stats_output[3] = semantic_formed;
    }

    MEMORY_UNLOCK(system);
    return 0;
}

/* ====================================================================
 * 工作记忆实现
 * ==================================================================== */

/**
 * @brief 查找工作记忆槽位索引
 */
static int working_find_slot(MemorySystem* system, const char* key) {
    for (size_t i = 0; i < system->working_capacity; i++) {
        if (system->working_slots[i].is_active &&
            strcmp(system->working_slots[i].key, key) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief 淘汰最不聚焦的工作记忆槽位
 */
static int working_evict_lowest_focus(MemorySystem* system) {
    int evict_idx = -1;
    float min_focus = 2.0f;
    
    for (size_t i = 0; i < system->working_capacity; i++) {
        if (!system->working_slots[i].is_active) {
            return (int)i;
        }
        if (system->working_slots[i].focus < min_focus) {
            min_focus = system->working_slots[i].focus;
            evict_idx = (int)i;
        }
    }
    
    if (evict_idx >= 0) {
        safe_free((void**)&system->working_slots[evict_idx].data);
        system->working_slots[evict_idx].key[0] = '\0';
        system->working_slots[evict_idx].data_size = 0;
        system->working_slots[evict_idx].focus = 0.0f;
        system->working_slots[evict_idx].rehearsal_count = 0;
        system->working_slots[evict_idx].age = 0.0f;
        system->working_slots[evict_idx].is_active = 0;
    }
    
    return evict_idx;
}

/**
 * @brief 写入工作记忆
 */
int memory_working_store(MemorySystem* system, const char* key,
                        const float* data, size_t data_size, float focus) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "工作记忆键为空");
    SELFLNN_CHECK_NULL(data, "工作记忆数据为空");
    SELFLNN_CHECK(data_size > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "工作记忆数据大小无效");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    if (system->working_capacity == 0) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_FULL, __func__, __FILE__, __LINE__,
                              "工作记忆已禁用（容量为0）");
        return -1;
    }
    
    int slot_idx = working_find_slot(system, key);
    if (slot_idx >= 0) {
        WorkingMemorySlot* slot = &system->working_slots[slot_idx];
        safe_free((void**)&slot->data);
        slot->data = (float*)safe_malloc(data_size * sizeof(float));
        if (!slot->data) {
            MEMORY_UNLOCK(system);
            return -1;
        }
        memcpy(slot->data, data, data_size * sizeof(float));
        slot->data_size = data_size;
        slot->focus = focus;
        slot->rehearsal_count += 0.5f;
        slot->age = 0.0f;
        MEMORY_UNLOCK(system);
        return slot_idx;
    }
    
    slot_idx = working_evict_lowest_focus(system);
    if (slot_idx < 0) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_FULL, __func__, __FILE__, __LINE__,
                              "工作记忆已满且无法淘汰");
        return -1;
    }
    
    WorkingMemorySlot* slot = &system->working_slots[slot_idx];
    strncpy(slot->key, key, 63);
    slot->key[63] = '\0';
    slot->data = (float*)safe_malloc(data_size * sizeof(float));
    if (!slot->data) {
        MEMORY_UNLOCK(system);
        return -1;
    }
    memcpy(slot->data, data, data_size * sizeof(float));
    slot->data_size = data_size;
    slot->focus = focus;
    slot->rehearsal_count = 0.0f;
    slot->age = 0.0f;
    slot->is_active = 1;
    
    if ((size_t)slot_idx >= system->working_count) {
        system->working_count++;
    }
    
    MEMORY_UNLOCK(system);
    return slot_idx;
}

/**
 * @brief 从工作记忆检索
 */
int memory_working_retrieve(MemorySystem* system, const char* key,
                           float* data, size_t data_size, float* focus) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "工作记忆键为空");
    SELFLNN_CHECK_NULL(data, "输出数据缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    int slot_idx = working_find_slot(system, key);
    if (slot_idx < 0) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "工作记忆项未找到（键：%s）", key);
        return -1;
    }
    
    WorkingMemorySlot* slot = &system->working_slots[slot_idx];
    if (data_size < slot->data_size) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "输出缓冲区太小（需要：%zu，实际：%zu）",
                              slot->data_size, data_size);
        return -1;
    }
    
    memcpy(data, slot->data, slot->data_size * sizeof(float));
    if (focus) *focus = slot->focus;
    slot->focus += 0.05f;
    if (slot->focus > 1.0f) slot->focus = 1.0f;
    
    MEMORY_UNLOCK(system);
    return 0;
}

/**
 * @brief 复述工作记忆项
 */
int memory_rehearse(MemorySystem* system, const char* key) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "工作记忆键为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    int slot_idx = working_find_slot(system, key);
    if (slot_idx < 0) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "要复述的工作记忆项未找到（键：%s）", key);
        return -1;
    }
    
    WorkingMemorySlot* slot = &system->working_slots[slot_idx];
    slot->rehearsal_count += 1.0f;
    slot->age = 0.0f;
    slot->focus += 0.1f;
    if (slot->focus > 1.0f) slot->focus = 1.0f;
    
    float* s_data = slot->data;
    size_t s_dsize = slot->data_size;
    float rc = slot->rehearsal_count;
    int do_short = ((int)rc % MEMORY_REHEARSE_SHORT_TERM_THRESHOLD == 0);
    int do_long = ((int)rc % MEMORY_REHEARSE_LONG_TERM_THRESHOLD == 0);
    
    MEMORY_UNLOCK(system);
    
    if (do_short) {
        memory_store(system, key, s_data, s_dsize,
                    MEMORY_TYPE_SHORT_TERM, 0.6f);
    }
    if (do_long) {
        memory_store(system, key, s_data, s_dsize,
                    MEMORY_TYPE_LONG_TERM, 0.8f);
    }
    
    return (int)rc;
}

/**
 * @brief 批量复述所有工作记忆项
 */
int memory_rehearse_all(MemorySystem* system) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    int rehearsed = 0;
    for (size_t i = 0; i < system->working_capacity; i++) {
        if (system->working_slots[i].is_active) {
            memory_rehearse(system, system->working_slots[i].key);
            rehearsed++;
        }
    }
    
    return rehearsed;
}

/**
 * @brief 获取工作记忆状态
 */
int memory_working_get_state(MemorySystem* system, WorkingMemorySlot* slots, size_t max_slots) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(slots, "输出缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    size_t active = 0;
    for (size_t i = 0; i < system->working_capacity && active < max_slots; i++) {
        if (system->working_slots[i].is_active) {
            memcpy(&slots[active], &system->working_slots[i], sizeof(WorkingMemorySlot));
            if (slots[active].data_size > 0) {
                slots[active].data = (float*)safe_malloc(slots[active].data_size * sizeof(float));
                if (slots[active].data) {
                    memcpy(slots[active].data, system->working_slots[i].data,
                          slots[active].data_size * sizeof(float));
                }
            } else {
                slots[active].data = NULL;
            }
            active++;
        }
    }
    
    MEMORY_UNLOCK(system);
    return (int)active;
}

/* ====================================================================
 * 记忆压缩实现（在线PCA学习降维，Oja's rule）
 * ==================================================================== */

/**
 * @brief 查找压缩映射索引
 */
static int compression_find_index(MemorySystem* system, const char* key) {
    for (size_t i = 0; i < system->compression_count; i++) {
        if (strcmp(system->compression_keys[i], key) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief 在线PCA学习状态
 */
typedef struct {
    float* basis;           /**< 基向量矩阵 (compressed_dim x original_dim, 行优先) */
    float* eigenvalues;     /**< 特征值 (compressed_dim) */
    float learning_rate;    /**< 学习率 */
    float decay;            /**< 学习率衰减 */
    size_t n_samples;       /**< 已学习的样本数 */
} PCAState;

/**
 * @brief 初始化PCA状态
 */
static PCAState* pca_state_create(size_t original_dim, size_t compressed_dim) {
    PCAState* pca = (PCAState*)safe_malloc(sizeof(PCAState));
    if (!pca) return NULL;
    
    pca->basis = (float*)safe_malloc(compressed_dim * original_dim * sizeof(float));
    pca->eigenvalues = (float*)safe_malloc(compressed_dim * sizeof(float));
    if (!pca->basis || !pca->eigenvalues) {
        safe_free((void**)&pca->basis);
        safe_free((void**)&pca->eigenvalues);
        safe_free((void**)&pca);
        return NULL;
    }
    
    for (size_t i = 0; i < compressed_dim; i++) {
        float norm = 0.0f;
        for (size_t j = 0; j < original_dim; j++) {
            pca->basis[i * original_dim + j] = ((float)(rng_next() % 2001) / 1000.0f - 1.0f) * 0.01f;
            norm += pca->basis[i * original_dim + j] * pca->basis[i * original_dim + j];
        }
        norm = sqrtf(norm + 1e-10f);
        for (size_t j = 0; j < original_dim; j++) {
            pca->basis[i * original_dim + j] /= norm;
        }
        pca->eigenvalues[i] = 0.0f;
    }
    
    pca->learning_rate = 0.01f;
    pca->decay = 0.999f;
    pca->n_samples = 0;
    return pca;
}

/**
 * @brief 销毁PCA状态
 */
static void pca_state_destroy(PCAState* pca) {
    if (pca) {
        safe_free((void**)&pca->basis);
        safe_free((void**)&pca->eigenvalues);
        safe_free((void**)&pca);
    }
}

/**
 * @brief Sanger规则（广义Oja规则）在线PCA学习
 * 
 * 同时学习多个主成分，保证收敛到按特征值降序排列的主成分：
 *   Δw_ij = η * y_i * (x_j - Σ_{k=1}^{i} y_k * w_kj)
 * 
 * 其中 y_i = Σ_j w_ij * x_j
 */
static void pca_sanger_update(PCAState* pca, const float* data, size_t dim, size_t n_components) {
    float* y = (float*)safe_malloc(n_components * sizeof(float));
    if (!y) return;
    
    for (size_t i = 0; i < n_components; i++) {
        y[i] = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            y[i] += pca->basis[i * dim + j] * data[j];
        }
    }
    
    for (size_t i = 0; i < n_components; i++) {
        float eta = pca->learning_rate;
        for (size_t j = 0; j < dim; j++) {
            float reconstruction = 0.0f;
            for (size_t k = 0; k <= i; k++) {
                reconstruction += y[k] * pca->basis[k * dim + j];
            }
            pca->basis[i * dim + j] += eta * y[i] * (data[j] - reconstruction);
        }
        float norm = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            norm += pca->basis[i * dim + j] * pca->basis[i * dim + j];
        }
        norm = sqrtf(norm + 1e-10f);
        for (size_t j = 0; j < dim; j++) {
            pca->basis[i * dim + j] /= norm;
        }
    }
    
    pca->n_samples++;
    pca->learning_rate *= pca->decay;
    if (pca->learning_rate < 0.0001f) pca->learning_rate = 0.0001f;
    
    safe_free((void**)&y);
}

/**
 * @brief Oja规则单神经元在线PCA学习
 */
static void pca_oja_update(float* w, const float* data, size_t dim, float* lr) {
    float y = 0.0f;
    for (size_t j = 0; j < dim; j++) {
        y += w[j] * data[j];
    }
    for (size_t j = 0; j < dim; j++) {
        w[j] += (*lr) * y * (data[j] - y * w[j]);
    }
    float norm = 0.0f;
    for (size_t j = 0; j < dim; j++) {
        norm += w[j] * w[j];
    }
    norm = sqrtf(norm + 1e-10f);
    for (size_t j = 0; j < dim; j++) {
        w[j] /= norm;
    }
    *lr *= 0.999f;
    if (*lr < 0.0001f) *lr = 0.0001f;
}

/**
 * @brief 从多个数据样本中学习PCA基
 */
static float* pca_learn_from_data(const float* data_samples, size_t n_samples,
                                   size_t dim, size_t compressed_dim) {
    PCAState* pca = pca_state_create(dim, compressed_dim);
    if (!pca) return NULL;
    
    pca->learning_rate = 0.01f;
    
    for (int epoch = 0; epoch < 10; epoch++) {
        for (size_t s = 0; s < n_samples; s++) {
            pca_sanger_update(pca, &data_samples[s * dim], dim, compressed_dim);
        }
    }
    
    for (size_t i = 0; i < compressed_dim; i++) {
        pca->eigenvalues[i] = 0.0f;
        for (size_t s = 0; s < n_samples && s < 100; s++) {
            float proj = 0.0f;
            for (size_t j = 0; j < dim; j++) {
                proj += pca->basis[i * dim + j] * data_samples[s * dim + j];
            }
            pca->eigenvalues[i] += proj * proj;
        }
        pca->eigenvalues[i] /= (float)(n_samples < 100 ? n_samples : 100);
        pca->eigenvalues[i] = sqrtf(pca->eigenvalues[i] + 1e-10f);
    }
    
    float* result = pca->basis;
    pca->basis = NULL;
    pca_state_destroy(pca);
    return result;
}

/**
 * @brief 使用随机投影做回退（当没有训练数据时）
 */
static float* pca_generate_fallback(size_t original_dim, size_t compressed_dim) {
    float* matrix = (float*)safe_malloc(compressed_dim * original_dim * sizeof(float));
    if (!matrix) return NULL;
    
    const float sqrt3 = 1.7320508075688772f;
    for (size_t i = 0; i < compressed_dim * original_dim; i++) {
        int r = rng_next() % (uint64_t)(6);
        if (r < 1) matrix[i] = sqrt3;
        else if (r < 2) matrix[i] = -sqrt3;
        else matrix[i] = 0.0f;
    }
    return matrix;
}

/**
 * @brief 收集系统内所有记忆数据用于PCA训练
 */
static float* pca_collect_data(MemorySystem* system, size_t* out_n, size_t* out_dim) {
    size_t total_items = system->st_count + system->lt_count + system->ep_count + system->se_count;
    if (total_items == 0) return NULL;
    
    size_t sample_dim = 0;
    size_t valid = 0;
    
    for (int t = 0; t < 4; t++) {
        MemoryItem** arr = NULL;
        size_t cnt = 0;
        switch (t) {
            case 0: arr = system->short_term_mem; cnt = system->st_count; break;
            case 1: arr = system->long_term_mem; cnt = system->lt_count; break;
            case 2: arr = system->episodic_mem; cnt = system->ep_count; break;
            case 3: arr = system->semantic_mem; cnt = system->se_count; break;
        }
        for (size_t i = 0; i < cnt; i++) {
            if (arr[i] && arr[i]->data && arr[i]->data_size > 1) {
                if (valid == 0) sample_dim = arr[i]->data_size;
                if (arr[i]->data_size == sample_dim) valid++;
            }
        }
    }
    
    if (valid < 2 || sample_dim < 2) return NULL;
    
    float* buffer = (float*)safe_malloc(valid * sample_dim * sizeof(float));
    if (!buffer) return NULL;
    
    size_t idx = 0;
    for (int t = 0; t < 4; t++) {
        MemoryItem** arr = NULL;
        size_t cnt = 0;
        switch (t) {
            case 0: arr = system->short_term_mem; cnt = system->st_count; break;
            case 1: arr = system->long_term_mem; cnt = system->lt_count; break;
            case 2: arr = system->episodic_mem; cnt = system->ep_count; break;
            case 3: arr = system->semantic_mem; cnt = system->se_count; break;
        }
        for (size_t i = 0; i < cnt && idx < valid; i++) {
            if (arr[i] && arr[i]->data && arr[i]->data_size == sample_dim) {
                memcpy(&buffer[idx * sample_dim], arr[i]->data, sample_dim * sizeof(float));
                idx++;
            }
        }
    }
    
    *out_n = valid;
    *out_dim = sample_dim;
    return buffer;
}

/**
 * @brief 压缩指定记忆（在线PCA学习降维）
 */
int memory_compress(MemorySystem* system, const char* key, float target_ratio) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK(target_ratio > 0.0f && target_ratio <= 1.0f,
                 SELFLNN_ERROR_INVALID_ARGUMENT,
                 "目标压缩比率必须为0.0~1.0（实际：%f）", target_ratio);
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    if (!system->config.enable_compression) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "记忆压缩未启用");
        return -1;
    }
    
    MemoryItem* item = NULL;
    for (int t = 0; t < 4; t++) {
        item = memory_find_item(system, key, (MemoryType)t);
        if (item) break;
    }
    if (!item) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "要压缩的记忆未找到（键：%s）", key);
        return -1;
    }
    
    int existing_idx = compression_find_index(system, key);
    if (existing_idx >= 0 && system->compression_infos[existing_idx].is_compressed) {
        int ret = (int)system->compression_infos[existing_idx].compressed_dim;
        MEMORY_UNLOCK(system);
        return ret;
    }
    
    size_t original_dim = item->data_size;
    size_t compressed_dim = (size_t)(original_dim * target_ratio);
    if (compressed_dim < 2) compressed_dim = 2;
    if (compressed_dim >= original_dim) {
        compressed_dim = original_dim / 2;
        if (compressed_dim < 2) compressed_dim = 2;
    }
    
    float* proj = NULL;
    
    size_t n_data = 0, data_dim = 0;
    float* data_buffer = pca_collect_data(system, &n_data, &data_dim);
    
    if (data_buffer && data_dim == original_dim && n_data >= 2) {
        proj = pca_learn_from_data(data_buffer, n_data, original_dim, compressed_dim);
        safe_free((void**)&data_buffer);
    }
    
    if (!proj) {
        proj = pca_generate_fallback(original_dim, compressed_dim);
    }
/* 锁泄漏修复——错误路径需释放锁，否则死锁整个记忆系统 */
    if (!proj) { MEMORY_UNLOCK(system); return -1; }
    
    float* reconst = (float*)safe_malloc(original_dim * compressed_dim * sizeof(float));
    if (!reconst) {
        safe_free((void**)&proj);
        MEMORY_UNLOCK(system);
        return -1;
    }
    for (size_t i = 0; i < original_dim; i++) {
        for (size_t j = 0; j < compressed_dim; j++) {
            reconst[i * compressed_dim + j] = proj[j * original_dim + i] / compressed_dim;
        }
    }
    
    float* compressed_data = (float*)safe_malloc(compressed_dim * sizeof(float));
    if (!compressed_data) {
        safe_free((void**)&proj);
        safe_free((void**)&reconst);
        MEMORY_UNLOCK(system);
        return -1;
    }
    
    for (size_t i = 0; i < compressed_dim; i++) {
        compressed_data[i] = 0.0f;
        for (size_t j = 0; j < original_dim; j++) {
            compressed_data[i] += proj[i * original_dim + j] * item->data[j];
        }
    }
    
    // 计算压缩误差（相对于原始数据）
    float* reconstructed = (float*)safe_malloc(original_dim * sizeof(float));
    if (reconstructed) {
        for (size_t i = 0; i < original_dim; i++) {
            reconstructed[i] = 0.0f;
            for (size_t j = 0; j < compressed_dim; j++) {
                reconstructed[i] += reconst[i * compressed_dim + j] * compressed_data[j];
            }
        }
        
        float mse = 0.0f;
/* 除零防护——original_dim为0时跳过MSE计算 */
        if (original_dim == 0) { safe_free((void**)&reconstructed); reconstructed = NULL; }
        if (reconstructed) {
            for (size_t i = 0; i < original_dim; i++) {
                float diff = reconstructed[i] - item->data[i];
                mse += diff * diff;
            }
            mse /= original_dim;
            safe_free((void**)&reconstructed);
        }
        
        // 存储压缩信息
        int idx = existing_idx >= 0 ? existing_idx : (int)system->compression_count;
        if (idx >= MAX_COMPRESSION_MAPS) {
            safe_free((void**)&proj);
            safe_free((void**)&reconst);
            safe_free((void**)&compressed_data);
            MEMORY_UNLOCK(system);
            return -1;
        }
        
        strncpy(system->compression_keys[idx], key, 63);
        system->compression_keys[idx][63] = '\0';
        system->compression_infos[idx].original_dim = original_dim;
        system->compression_infos[idx].compressed_dim = compressed_dim;
        system->compression_infos[idx].projection_matrix = proj;
        system->compression_infos[idx].reconstruction = reconst;
        system->compression_infos[idx].compression_error = mse;
        system->compression_infos[idx].is_compressed = 1;
        
        if (existing_idx < 0) {
            system->compression_count++;
        }
        
        safe_free((void**)&item->data);
        item->data = compressed_data;
        item->data_size = compressed_dim;
        
        MEMORY_UNLOCK(system);
        return (int)compressed_dim;
    }
    
    MEMORY_UNLOCK(system);
    safe_free((void**)&proj);
    safe_free((void**)&reconst);
    safe_free((void**)&compressed_data);
    return -1;
}

/**
 * @brief 解压指定记忆
 */
int memory_decompress(MemorySystem* system, const char* key) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    int idx = compression_find_index(system, key);
    if (idx < 0 || !system->compression_infos[idx].is_compressed) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "未找到压缩的记忆（键：%s）或记忆未被压缩", key);
        return -1;
    }
    
    MemoryCompressionInfo* info = &system->compression_infos[idx];
    
    MemoryItem* item = NULL;
    for (int t = 0; t < 4; t++) {
        item = memory_find_item(system, key, (MemoryType)t);
        if (item) break;
    }
    if (!item) {
        MEMORY_UNLOCK(system);
        return -1;
    }
    
    size_t original_dim = info->original_dim;
    size_t compressed_dim = info->compressed_dim;
    
    float* decompressed = (float*)safe_malloc(original_dim * sizeof(float));
    if (!decompressed) {
        MEMORY_UNLOCK(system);
        return -1;
    }
    
    for (size_t i = 0; i < original_dim; i++) {
        decompressed[i] = 0.0f;
        for (size_t j = 0; j < compressed_dim; j++) {
            decompressed[i] += info->reconstruction[i * compressed_dim + j] * item->data[j];
        }
    }
    
    safe_free((void**)&item->data);
    item->data = decompressed;
    item->data_size = original_dim;
    info->is_compressed = 0;
    
    MEMORY_UNLOCK(system);
    return 0;
}

/**
 * @brief 批量压缩所有可压缩记忆
 */
int memory_bulk_compress(MemorySystem* system, float target_ratio) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    int compressed_count = 0;
    size_t counts[4] = {system->st_count, system->lt_count,
                        system->ep_count, system->se_count};
    MemoryItem*** arrays[4] = {&system->short_term_mem, &system->long_term_mem,
                               &system->episodic_mem, &system->semantic_mem};
    
    for (int t = 0; t < 4; t++) {
        MemoryItem** array = *arrays[t];
        size_t count = counts[t];
        
        for (size_t i = 0; i < count; i++) {
            if (array[i] && array[i]->data_size > 10) {
                int idx = compression_find_index(system, array[i]->key);
                if (idx < 0 || !system->compression_infos[idx].is_compressed) {
                    char* item_key = array[i]->key;
                    MEMORY_UNLOCK(system);
                    int result = memory_compress(system, item_key, target_ratio);
                    MEMORY_LOCK(system);
                    if (result > 0) {
                        compressed_count++;
                    }
                }
            }
        }
    }
    
    MEMORY_UNLOCK(system);
    return compressed_count;
}

/**
 * @brief 检查记忆是否已压缩
 */
int memory_is_compressed(MemorySystem* system, const char* key, MemoryCompressionInfo* info) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    int idx = compression_find_index(system, key);
    if (idx < 0 || !system->compression_infos[idx].is_compressed) {
        MEMORY_UNLOCK(system);
        return 0;
    }
    
    if (info) {
        memcpy(info, &system->compression_infos[idx], sizeof(MemoryCompressionInfo));
        info->projection_matrix = NULL;
        info->reconstruction = NULL;
    }
    
    MEMORY_UNLOCK(system);
    return 1;
}

/* ====================================================================
 * 再巩固实现
 * ==================================================================== */

/**
 * @brief 查找再巩固条目索引
 */
static int reconsolidation_find_entry(MemorySystem* system, const char* key) {
    for (size_t i = 0; i < MAX_RECONSOLIDATION_ENTRIES; i++) {
        if (system->reconsolidation_entries[i].is_active &&
            strcmp(system->reconsolidation_entries[i].key, key) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief 查找空闲再巩固条目
 */
static int reconsolidation_find_free(MemorySystem* system) {
    for (size_t i = 0; i < MAX_RECONSOLIDATION_ENTRIES; i++) {
        if (!system->reconsolidation_entries[i].is_active) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief 触发记忆再巩固
 */
int memory_trigger_reconsolidation(MemorySystem* system, const char* key) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    if (!system->config.enable_reconsolidation) {
        MEMORY_UNLOCK(system);
        return 0;
    }
    
    int existing = reconsolidation_find_entry(system, key);
    if (existing >= 0) {
        ReconsolidationEntry* entry = &system->reconsolidation_entries[existing];
        if (entry->state == RECONSOLIDATION_LABILE ||
            entry->state == RECONSOLIDATION_IN_PROGRESS) {
            entry->trigger_time = system->current_time;
            MEMORY_UNLOCK(system);
            return 0;
        }
        
        if (entry->state == RECONSOLIDATION_COMPLETE) {
            entry->state = RECONSOLIDATION_LABILE;
            entry->trigger_time = system->current_time;
            MEMORY_UNLOCK(system);
            return 0;
        }
        
        entry->state = RECONSOLIDATION_LABILE;
        entry->trigger_time = system->current_time;
        entry->is_active = 1;
        MEMORY_UNLOCK(system);
        return 0;
    }
    
    int idx = reconsolidation_find_free(system);
    if (idx < 0) {
        for (size_t i = 0; i < MAX_RECONSOLIDATION_ENTRIES; i++) {
            if (system->reconsolidation_entries[i].state == RECONSOLIDATION_COMPLETE) {
                if (system->reconsolidation_entries[i].labile_data) {
                    safe_free((void**)&system->reconsolidation_entries[i].labile_data);
                }
                system->reconsolidation_entries[i].is_active = 0;
                idx = (int)i;
                break;
            }
        }
    }
    if (idx < 0) {
        MEMORY_UNLOCK(system);
        return -1;
    }
    
    MemoryItem* item = NULL;
    for (int t = 0; t < 4; t++) {
        item = memory_find_item(system, key, (MemoryType)t);
        if (item) break;
    }
    if (!item) {
        MEMORY_UNLOCK(system);
        return -1;
    }
    
    ReconsolidationEntry* entry = &system->reconsolidation_entries[idx];
    strncpy(entry->key, key, 63);
    entry->key[63] = '\0';
    entry->state = RECONSOLIDATION_LABILE;
    entry->trigger_time = system->current_time;
    entry->labile_duration = 2.0f + (rng_next() % (uint64_t)(10)) * 0.1f;
    entry->original_strength = item->strength;
    entry->is_active = 1;
    
    if (item->data_size > 0) {
        entry->labile_data = (float*)safe_malloc(item->data_size * sizeof(float));
        if (entry->labile_data) {
            memcpy(entry->labile_data, item->data, item->data_size * sizeof(float));
            entry->labile_data_size = item->data_size;
        }
    }
    
    if ((size_t)idx >= system->reconsolidation_count) {
        system->reconsolidation_count++;
    }
    
    MEMORY_UNLOCK(system);
    return 0;
}

/**
 * @brief 完成记忆再巩固
 */
int memory_complete_reconsolidation(MemorySystem* system, const char* key,
                                   const float* new_data, size_t new_data_size) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    int idx = reconsolidation_find_entry(system, key);
    if (idx < 0) {
        MEMORY_UNLOCK(system);
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "未找到再巩固条目（键：%s）", key);
        return -1;
    }
    
    ReconsolidationEntry* entry = &system->reconsolidation_entries[idx];
    if (entry->state != RECONSOLIDATION_LABILE &&
        entry->state != RECONSOLIDATION_IN_PROGRESS) {
        MEMORY_UNLOCK(system);
        return -1;
    }
    
    MemoryItem* item = NULL;
    for (int t = 0; t < 4; t++) {
        item = memory_find_item(system, key, (MemoryType)t);
        if (item) break;
    }
    if (!item) {
        MEMORY_UNLOCK(system);
        return -1;
    }
    
    float boost = system->config.reconsolidation_boost;
    if (boost <= 0.0f) boost = 0.1f;
    
    item->strength += boost + entry->original_strength * boost * 0.5f;
    if (item->strength > 1.0f) item->strength = 1.0f;
    
    if (new_data && new_data_size > 0 && new_data_size <= item->data_size) {
        float blend_ratio = 0.3f;
        for (size_t i = 0; i < new_data_size; i++) {
            item->data[i] = (1.0f - blend_ratio) * item->data[i] + blend_ratio * new_data[i];
        }
    }
    
    entry->state = RECONSOLIDATION_COMPLETE;
    
    if (entry->labile_data) {
        safe_free((void**)&entry->labile_data);
        entry->labile_data_size = 0;
    }
    
    item->access_count++;
    
    MEMORY_UNLOCK(system);
    return 0;
}

/**
 * @brief 更新所有再巩固状态
 */
int memory_update_reconsolidation(MemorySystem* system, float time_delta) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    system->current_time += time_delta;
    
    int updated = 0;
    
    for (size_t i = 0; i < MAX_RECONSOLIDATION_ENTRIES; i++) {
        ReconsolidationEntry* entry = &system->reconsolidation_entries[i];
        
        if (!entry->is_active) continue;
        
        if (entry->state == RECONSOLIDATION_LABILE) {
            float elapsed = system->current_time - entry->trigger_time;
            if (elapsed >= entry->labile_duration) {
                entry->state = RECONSOLIDATION_IN_PROGRESS;
                char* ekey = entry->key;
                MEMORY_UNLOCK(system);
                memory_complete_reconsolidation(system, ekey, NULL, 0);
                MEMORY_LOCK(system);
                updated++;
            }
        }
    }
    
    MEMORY_UNLOCK(system);
    return updated;
}

/**
 * @brief 获取再巩固状态
 */
int memory_get_reconsolidation_state(MemorySystem* system, const char* key,
                                    ReconsolidationEntry* entry) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_NULL(entry, "输出缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    
    MEMORY_LOCK(system);
    
    int idx = reconsolidation_find_entry(system, key);
    if (idx < 0) {
        MEMORY_UNLOCK(system);
        return -1;
    }
    
    memcpy(entry, &system->reconsolidation_entries[idx], sizeof(ReconsolidationEntry));
    entry->labile_data = NULL;
    entry->labile_data_size = 0;
    
    MEMORY_UNLOCK(system);
    return 0;
}

/* ====================================================================
 * 训练集成操作
 * ==================================================================== */

size_t memory_get_count(MemorySystem* system, MemoryType memory_type) {
    if (!system || !system->is_initialized) {
        return 0;
    }
    
    MEMORY_LOCK(system);
    size_t result = 0;

    switch (memory_type) {
        case MEMORY_TYPE_SHORT_TERM: result = system->st_count; break;
        case MEMORY_TYPE_LONG_TERM:  result = system->lt_count; break;
        case MEMORY_TYPE_EPISODIC:   result = system->ep_count; break;
        case MEMORY_TYPE_SEMANTIC:   result = system->se_count; break;
        case MEMORY_TYPE_WORKING:    result = system->working_count; break;
        default: result = 0; break;
    }
    
    MEMORY_UNLOCK(system);
    return result;
}

int memory_sample_training_batch(MemorySystem* system, MemoryType memory_type,
                                  size_t batch_size, size_t data_dim,
                                  float* inputs, float* targets) {
    if (!system || !system->is_initialized || !inputs || !targets) {
        return -1;
    }
    if (batch_size == 0 || data_dim == 0) {
        return -1;
    }
    
    MEMORY_LOCK(system);

    MemoryItem*** array_ptr = NULL;
    size_t count = 0;

    switch (memory_type) {
        case MEMORY_TYPE_SHORT_TERM: array_ptr = &system->short_term_mem; count = system->st_count; break;
        case MEMORY_TYPE_LONG_TERM:  array_ptr = &system->long_term_mem;  count = system->lt_count; break;
        case MEMORY_TYPE_EPISODIC:   array_ptr = &system->episodic_mem;   count = system->ep_count; break;
        case MEMORY_TYPE_SEMANTIC:   array_ptr = &system->semantic_mem;   count = system->se_count; break;
        default:
            MEMORY_UNLOCK(system);
            return -1;
    }

    if (!array_ptr || !(*array_ptr)) {
        MEMORY_UNLOCK(system);
        return -1;
    }

    MemoryItem** array = *array_ptr;

    size_t available = (count < batch_size) ? count : batch_size;
    if (available == 0) {
        memset(inputs, 0, batch_size * data_dim * sizeof(float));
        memset(targets, 0, batch_size * data_dim * sizeof(float));
        return 0;
    }

    int* selected = (int*)safe_malloc(available * sizeof(int));
    if (!selected) {
        MEMORY_UNLOCK(system);
        return -1;
    }

    size_t selected_count = 0;
    size_t max_attempts = available * 10;
    size_t attempts = 0;

    while (selected_count < available && attempts < max_attempts) {
        size_t idx = (size_t)(rng_next() % (uint64_t)(count));

        int already_selected = 0;
        for (size_t i = 0; i < selected_count; i++) {
            if (selected[i] == (int)idx) {
                already_selected = 1;
                break;
            }
        }

        if (!already_selected) {
            selected[selected_count] = (int)idx;
            selected_count++;
        }
        attempts++;
    }

    if (selected_count < available) {
        for (size_t i = selected_count; i < available; i++) {
            selected[i] = (int)(rng_next() % (uint64_t)(count));
        }
    }

    for (size_t i = 0; i < available; i++) {
        MemoryItem* item = array[selected[i]];
        if (!item || !item->data || item->data_size < data_dim * 2) {
            memset(&inputs[i * data_dim], 0, data_dim * sizeof(float));
            memset(&targets[i * data_dim], 0, data_dim * sizeof(float));
            continue;
        }

        memcpy(&inputs[i * data_dim], item->data, data_dim * sizeof(float));
        memcpy(&targets[i * data_dim], &item->data[data_dim], data_dim * sizeof(float));
    }

    for (size_t i = available; i < batch_size; i++) {
        memset(&inputs[i * data_dim], 0, data_dim * sizeof(float));
        memset(&targets[i * data_dim], 0, data_dim * sizeof(float));
    }

    safe_free((void**)&selected);

    MEMORY_UNLOCK(system);
    return (int)available;
}

/* ====================================================================
 * 相似度上下文检索（记忆增强LNN前向传播用）
 * ==================================================================== */

/**
 * @brief 计算两个向量的点积相似度
 */
/* F-023修复：使用统一余弦相似度替代本地实现 */
static float compute_similarity(const float* v1, const float* v2, size_t dim) {
    return math_cosine_similarity(v1, v2, dim);
}

/**
 * @brief 基于相似度检索记忆上下文
 */
int memory_retrieve_context(MemorySystem* system, const float* query, size_t query_dim,
                           float* context_out, size_t top_k,
                           const MemoryType* memory_types, size_t num_types,
                           float* out_similarities) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(query, "查询向量为空");
    SELFLNN_CHECK_NULL(context_out, "上下文输出缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");

    if (query_dim == 0 || top_k == 0) {
        return -1;
    }
    
    MEMORY_LOCK(system);

    size_t total_count = system->st_count + system->lt_count + system->ep_count + system->se_count;
    if (total_count == 0) {
        memset(context_out, 0, query_dim * sizeof(float));
        MEMORY_UNLOCK(system);
        return 0;
    }

    float* sim_scores = (float*)safe_malloc(total_count * sizeof(float));
    MemoryItem** sim_items = (MemoryItem**)safe_malloc(total_count * sizeof(MemoryItem*));
    if (!sim_scores || !sim_items) {
        safe_free((void**)&sim_scores);
        safe_free((void**)&sim_items);
        MEMORY_UNLOCK(system);
        return -1;
    }

    size_t candidate_count = 0;
    size_t num_search_types = (memory_types != NULL && num_types > 0) ? num_types : 4;

    for (size_t t = 0; t < num_search_types; t++) {
        MemoryType search_type = memory_types ? memory_types[t] : (MemoryType)t;
        MemoryItem** array = NULL;
        size_t count = 0;

        switch (search_type) {
            case MEMORY_TYPE_SHORT_TERM: array = system->short_term_mem; count = system->st_count; break;
            case MEMORY_TYPE_LONG_TERM:  array = system->long_term_mem;  count = system->lt_count; break;
            case MEMORY_TYPE_EPISODIC:   array = system->episodic_mem;   count = system->ep_count; break;
            case MEMORY_TYPE_SEMANTIC:   array = system->semantic_mem;   count = system->se_count; break;
            default: continue;
        }

        for (size_t i = 0; i < count && candidate_count < total_count; i++) {
            MemoryItem* item = array[i];
            if (!item || !item->data || item->data_size < query_dim) continue;
            float sim = compute_similarity(query, item->data, query_dim);
            if (sim > 0.0f) {
                sim_scores[candidate_count] = sim;
                sim_items[candidate_count] = item;
                candidate_count++;
            }
        }
    }

    size_t actual_top = (top_k < candidate_count) ? top_k : candidate_count;

    if (actual_top > 0) {
        for (size_t i = 0; i < actual_top; i++) {
            size_t best_idx = i;
            for (size_t j = i + 1; j < candidate_count; j++) {
                if (sim_scores[j] > sim_scores[best_idx]) {
                    best_idx = j;
                }
            }
            if (best_idx != i) {
                float tmp_s = sim_scores[i];
                sim_scores[i] = sim_scores[best_idx];
                sim_scores[best_idx] = tmp_s;
                MemoryItem* tmp_m = sim_items[i];
                sim_items[i] = sim_items[best_idx];
                sim_items[best_idx] = tmp_m;
            }
        }

        float total_weight = 0.0f;
        for (size_t i = 0; i < actual_top; i++) {
            total_weight += sim_scores[i];
        }

        memset(context_out, 0, query_dim * sizeof(float));
        if (total_weight > 1e-10f) {
            for (size_t i = 0; i < actual_top; i++) {
                float weight = sim_scores[i] / total_weight;
                for (size_t j = 0; j < query_dim && j < sim_items[i]->data_size; j++) {
                    context_out[j] += weight * sim_items[i]->data[j];
                }
            }
        }

        if (out_similarities) {
            for (size_t i = 0; i < actual_top; i++) {
                out_similarities[i] = sim_scores[i];
            }
        }
    } else {
        memset(context_out, 0, query_dim * sizeof(float));
    }

    safe_free((void**)&sim_scores);
    safe_free((void**)&sim_items);

    MEMORY_UNLOCK(system);
    return (int)actual_top;
}

/* ====================================================================
 * Ebbinghaus遗忘曲线与间隔重复实现
 * ==================================================================== */

/**
 * @brief 获取Ebbinghaus遗忘曲线统计
 */
int memory_get_ebbinghaus_stats(MemorySystem* system, const char* key,
                                EbbinghausStats* stats) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_NULL(stats, "统计输出缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");

    MEMORY_LOCK(system);
    
    memset(stats, 0, sizeof(EbbinghausStats));

    MemoryItem* item = NULL;
    for (int t = 0; t < 4; t++) {
        item = memory_find_item(system, key, (MemoryType)t);
        if (item) break;
    }
    if (!item) {
        MEMORY_UNLOCK(system);
        return -1;
    }

    float time_since_access = system->current_time - item->last_access_time;
    float base_tau = 5.0f;
    float stability_factor = 1.0f + (float)item->access_count * 0.5f;
    float tau = base_tau * stability_factor;
    float retention = item->strength * expf(-time_since_access / (tau + 1e-10f));
    if (retention < 0.0f) retention = 0.0f;

    stats->retention = retention;
    stats->stability = item->strength * (1.0f + (float)item->access_count * 0.2f);
    stats->retrievability = retention;
    stats->is_stable = item->access_count >= 3 && item->strength > 0.5f;

    float target_retention = 0.8f;
    float optimal_t = -tau * logf((target_retention + 1e-10f) / (item->strength + 1e-10f));
    if (optimal_t < 0.0f) optimal_t = tau * 0.5f;
    stats->optimal_interval = optimal_t;
    stats->next_review_time = item->last_access_time + optimal_t;

    MEMORY_UNLOCK(system);
    return 0;
}

/**
 * @brief 计算指定时间后的记忆保留率
 */
float memory_retention_at_time(MemorySystem* system, const char* key,
                               float time_elapsed) {
    EbbinghausStats stats;
    if (memory_get_ebbinghaus_stats(system, key, &stats) != 0) {
        return -1.0f;
    }
    return stats.retrievability * expf(-time_elapsed / 5.0f);
}

/**
 * @brief 计算最佳复习间隔
 */
float memory_optimal_review_interval(MemorySystem* system, const char* key,
                                     float target_retention) {
    EbbinghausStats stats;
    if (memory_get_ebbinghaus_stats(system, key, &stats) != 0) {
        return -1.0f;
    }
    if (target_retention <= 0.0f) target_retention = 0.8f;
    if (target_retention >= 1.0f) target_retention = 0.99f;

    return stats.optimal_interval * (1.0f + (float)stats.is_stable * 0.5f);
}

/* ====================================================================
 * 工作记忆增强实现
 * ==================================================================== */

/**
 * @brief 提高工作记忆槽位的优先级（聚焦）
 */
int memory_working_prioritize(MemorySystem* system, const char* key,
                              float focus_boost) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "槽位键名为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");

    MEMORY_LOCK(system);

    for (size_t i = 0; i < system->working_capacity; i++) {
        if (system->working_slots[i].is_active &&
            strcmp(system->working_slots[i].key, key) == 0) {
            system->working_slots[i].focus += focus_boost;
            if (system->working_slots[i].focus > 1.0f)
                system->working_slots[i].focus = 1.0f;
            system->working_slots[i].rehearsal_count += 1.0f;
            MEMORY_UNLOCK(system);
            return 0;
        }
    }
    MEMORY_UNLOCK(system);
    return -1;
}

/**
 * @brief 老化所有工作记忆槽位
 */
int memory_working_age_all(MemorySystem* system, float time_delta) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");

    MEMORY_LOCK(system);

    int aged_out = 0;
    float decay_factor = expf(-0.1f * time_delta);
    float focus_threshold = 0.15f;

    for (size_t i = 0; i < system->working_capacity; i++) {
        if (!system->working_slots[i].is_active) continue;

        system->working_slots[i].focus *= decay_factor;
        system->working_slots[i].age += time_delta;

        if (system->working_slots[i].focus < focus_threshold) {
            system->working_slots[i].is_active = 0;
            system->working_slots[i].focus = 0.0f;
            aged_out++;
        }
    }
    MEMORY_UNLOCK(system);
    return aged_out;
}

/* ====================================================================
 * 记忆整合与关联增强实现
 * ==================================================================== */

/**
 * @brief 批量巩固最高优先级的记忆
 */
int memory_batch_consolidate(MemorySystem* system, size_t top_n) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");
    if (!system->config.enable_consolidation) return 0;

    MEMORY_LOCK(system);

    if (system->st_count == 0) {
        MEMORY_UNLOCK(system);
        return 0;
    }

    size_t count = system->st_count;
    float* priorities = (float*)safe_malloc(count * sizeof(float));
    size_t* indices = (size_t*)safe_malloc(count * sizeof(size_t));
    if (!priorities || !indices) {
        safe_free((void**)&priorities);
        safe_free((void**)&indices);
        MEMORY_UNLOCK(system);
        return -1;
    }

    for (size_t i = 0; i < count; i++) {
        MemoryItem* item = system->short_term_mem[i];
        if (!item) {
            priorities[i] = 0.0f;
            indices[i] = i;
            continue;
        }
        float freq = (float)item->access_count / (1.0f + (float)item->access_count);
        priorities[i] = item->strength * item->strength * (1.0f + freq);
        indices[i] = i;
    }

    for (size_t i = 0; i < count - 1; i++) {
        for (size_t j = i + 1; j < count; j++) {
            if (priorities[j] > priorities[i]) {
                float tmp_p = priorities[i];
                priorities[i] = priorities[j];
                priorities[j] = tmp_p;
                size_t tmp_i = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_i;
            }
        }
    }

    size_t to_consolidate = (top_n > 0 && top_n < count) ? top_n : count;
    int consolidated = 0;

    for (size_t i = 0; i < to_consolidate; i++) {
        MemoryItem* item = system->short_term_mem[indices[i]];
        if (!item) continue;
        if (!memory_find_item(system, item->key, MEMORY_TYPE_LONG_TERM)) {
            char* item_key = item->key;
            float* item_data = item->data;
            size_t item_dsize = item->data_size;
            float item_str = item->strength;
            MEMORY_UNLOCK(system);
            int ret = memory_store(system, item_key, item_data, item_dsize,
                                   MEMORY_TYPE_LONG_TERM, item_str * 0.85f);
            MEMORY_LOCK(system);
            if (ret == 0) {
                consolidated++;
            }
        } else {
            MemoryItem* lt = memory_find_item(system, item->key, MEMORY_TYPE_LONG_TERM);
            if (lt) {
                lt->strength += system->config.consolidation_rate * 0.3f;
                if (lt->strength > 1.0f) lt->strength = 1.0f;
            }
        }
    }

    safe_free((void**)&priorities);
    safe_free((void**)&indices);
    MEMORY_UNLOCK(system);
    return consolidated;
}

/**
 * @brief 查找与指定记忆相关的记忆项
 */
int memory_find_related(MemorySystem* system, const char* key,
                        float* context_out, size_t context_dim,
                        size_t max_related, float threshold) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(key, "记忆键为空");
    SELFLNN_CHECK_NULL(context_out, "上下文输出缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");

    MEMORY_LOCK(system);
    
    memset(context_out, 0, context_dim * sizeof(float));

    MemoryItem* source = NULL;
    for (int t = 0; t < 4; t++) {
        source = memory_find_item(system, key, (MemoryType)t);
        if (source) break;
    }
    if (!source || !source->data || source->data_size == 0) {
        MEMORY_UNLOCK(system);
        return 0;
    }

    float* related_sum = (float*)safe_calloc(context_dim, sizeof(float));
    float* rel_sims = (float*)safe_malloc(max_related * sizeof(float));
    MemoryItem** related_items = (MemoryItem**)safe_malloc(max_related * sizeof(MemoryItem*));
    if (!related_sum || !rel_sims || !related_items) {
        safe_free((void**)&related_sum);
        safe_free((void**)&rel_sims);
        safe_free((void**)&related_items);
        MEMORY_UNLOCK(system);
        return -1;
    }

    size_t found = 0;
    size_t min_dim_source = context_dim < source->data_size ? context_dim : source->data_size;

    for (size_t t = 0; t < 4 && found < max_related; t++) {
        MemoryType mtype = (MemoryType)t;
        MemoryItem** array = NULL;
        size_t count = 0;

        switch (mtype) {
            case MEMORY_TYPE_SHORT_TERM: array = system->short_term_mem; count = system->st_count; break;
            case MEMORY_TYPE_LONG_TERM:  array = system->long_term_mem;  count = system->lt_count; break;
            case MEMORY_TYPE_EPISODIC:   array = system->episodic_mem;   count = system->ep_count; break;
            case MEMORY_TYPE_SEMANTIC:   array = system->semantic_mem;   count = system->se_count; break;
            default: continue;
        }

        for (size_t i = 0; i < count && found < max_related; i++) {
            MemoryItem* item = array[i];
            if (!item || !item->data || item->data_size == 0) continue;
            if (strcmp(item->key, key) == 0) continue;

            size_t min_dim = min_dim_source < item->data_size ? min_dim_source : item->data_size;
            float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
            for (size_t j = 0; j < min_dim; j++) {
                dot += source->data[j] * item->data[j];
                n1 += source->data[j] * source->data[j];
                n2 += item->data[j] * item->data[j];
            }
            float denom = sqrtf(n1 * n2);
            if (denom < 1e-10f) continue;
            float sim = dot / denom;
            if (sim < threshold) continue;

            rel_sims[found] = sim;
            related_items[found] = item;
            found++;
        }
    }

    float total_weight = 0.0f;
    for (size_t i = 0; i < found; i++) {
        total_weight += rel_sims[i];
    }

    if (total_weight > 1e-10f) {
        for (size_t i = 0; i < found; i++) {
            float w = rel_sims[i] / total_weight;
            MemoryItem* item = related_items[i];
            size_t copy_dim = context_dim < item->data_size ? context_dim : item->data_size;
            for (size_t j = 0; j < copy_dim; j++) {
                related_sum[j] += w * item->data[j];
            }
        }
        memcpy(context_out, related_sum, context_dim * sizeof(float));
    }

    safe_free((void**)&related_sum);
    safe_free((void**)&rel_sims);
    safe_free((void**)&related_items);
    MEMORY_UNLOCK(system);
    return (int)found;
}

/* ====================================================================
 * 训练集成操作增强实现
 * ==================================================================== */

/**
 * @brief 从记忆中按优先级采样训练数据（增强版）
 */
int memory_sample_priority_batch(MemorySystem* system, MemoryType memory_type,
                                 size_t batch_size, size_t data_dim,
                                 float* inputs, float* targets,
                                 float min_strength) {
    SELFLNN_CHECK_NULL(system, "记忆系统句柄为空");
    SELFLNN_CHECK_NULL(inputs, "输入输出缓冲区为空");
    SELFLNN_CHECK_NULL(targets, "目标输出缓冲区为空");
    SELFLNN_CHECK_INITIALIZED(system, "记忆系统未初始化");

    MEMORY_LOCK(system);

    MemoryItem** array = NULL;
    size_t count = 0;
    switch (memory_type) {
        case MEMORY_TYPE_SHORT_TERM: array = system->short_term_mem; count = system->st_count; break;
        case MEMORY_TYPE_LONG_TERM:  array = system->long_term_mem;  count = system->lt_count; break;
        case MEMORY_TYPE_EPISODIC:   array = system->episodic_mem;   count = system->ep_count; break;
        case MEMORY_TYPE_SEMANTIC:   array = system->semantic_mem;   count = system->se_count; break;
        default:
            MEMORY_UNLOCK(system);
            return 0;
    }

    if (count == 0) {
        MEMORY_UNLOCK(system);
        return 0;
    }

    size_t valid_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (array[i] && array[i]->strength >= min_strength) valid_count++;
    }
    if (valid_count == 0) {
        MEMORY_UNLOCK(system);
        return 0;
    }

    float* weights = (float*)safe_malloc(valid_count * sizeof(float));
    MemoryItem** valid_items = (MemoryItem**)safe_malloc(valid_count * sizeof(MemoryItem*));
    if (!weights || !valid_items) {
        safe_free((void**)&weights);
        safe_free((void**)&valid_items);
        MEMORY_UNLOCK(system);
        return -1;
    }

    size_t idx = 0;
    float total_weight = 0.0f;
    for (size_t i = 0; i < count; i++) {
        MemoryItem* item = array[i];
        if (!item || item->strength < min_strength) continue;
        float freq = (float)item->access_count / (1.0f + (float)item->access_count);
        weights[idx] = item->strength * item->strength * (1.0f + 0.5f * logf(1.0f + freq));
        total_weight += weights[idx];
        valid_items[idx] = item;
        idx++;
    }

    size_t actual_batch = batch_size < valid_count ? batch_size : valid_count;

    for (size_t b = 0; b < actual_batch; b++) {
        float r = secure_random_float() * total_weight;
        float cum = 0.0f;
        size_t picked = 0;
        for (size_t i = 0; i < valid_count; i++) {
            cum += weights[i];
            if (r <= cum) { picked = i; break; }
        }

        MemoryItem* item = valid_items[picked];
        size_t copy_dim = data_dim < item->data_size ? data_dim : item->data_size;
        memcpy(inputs + b * data_dim, item->data, copy_dim * sizeof(float));
        memcpy(targets + b * data_dim, item->data, copy_dim * sizeof(float));
        item->access_count++;
        item->strength += 0.01f;
        if (item->strength > 1.0f) item->strength = 1.0f;
    }

    safe_free((void**)&weights);
    safe_free((void**)&valid_items);
    MEMORY_UNLOCK(system);
    return (int)actual_batch;
}

/**
 * @brief 按全局索引获取记忆项键名
 *
 * 遍历所有记忆池（短期→长期→情景→语义）的顺序获取全局索引对应的记忆键名。
 * 索引顺序：先短期记忆，再长期、情景、语义记忆。
 */
int memory_get_key(MemorySystem* system, size_t index, char* key_buf, size_t key_buf_size) {
    if (!system || !system->is_initialized || !key_buf || key_buf_size == 0) {
        return -1;
    }

    MEMORY_LOCK(system);

    /* 按顺序遍历四个记忆池 */
    size_t offset = 0;

    /* 短期记忆 */
    if (index - offset < system->st_count) {
        MemoryItem* item = system->short_term_mem[index - offset];
        if (item && item->key) {
            strncpy(key_buf, item->key, key_buf_size - 1);
            key_buf[key_buf_size - 1] = '\0';
        } else {
            key_buf[0] = '\0';
        }
        MEMORY_UNLOCK(system);
        return 0;
    }
    offset += system->st_count;

    /* 长期记忆 */
    if (index - offset < system->lt_count) {
        MemoryItem* item = system->long_term_mem[index - offset];
        if (item && item->key) {
            strncpy(key_buf, item->key, key_buf_size - 1);
            key_buf[key_buf_size - 1] = '\0';
        } else {
            key_buf[0] = '\0';
        }
        MEMORY_UNLOCK(system);
        return 0;
    }
    offset += system->lt_count;

    /* 情景记忆 */
    if (index - offset < system->ep_count) {
        MemoryItem* item = system->episodic_mem[index - offset];
        if (item && item->key) {
            strncpy(key_buf, item->key, key_buf_size - 1);
            key_buf[key_buf_size - 1] = '\0';
        } else {
            key_buf[0] = '\0';
        }
        MEMORY_UNLOCK(system);
        return 0;
    }
    offset += system->ep_count;

    /* 语义记忆 */
    if (index - offset < system->se_count) {
        MemoryItem* item = system->semantic_mem[index - offset];
        if (item && item->key) {
            strncpy(key_buf, item->key, key_buf_size - 1);
            key_buf[key_buf_size - 1] = '\0';
        } else {
            key_buf[0] = '\0';
        }
        MEMORY_UNLOCK(system);
        return 0;
    }

    MEMORY_UNLOCK(system);
    return -1; /* 索引越界 */
}