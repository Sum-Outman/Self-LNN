/**
 * @file short_term.c
 * @brief 短期记忆系统实现 — F-006修复：添加LRU驱逐、时间衰减、容量管理等真实算法
 * 
 * 短期记忆管理实现，在底层MemorySystem基础上增加了短期记忆特有算法：
 * - 容量限制与LRU驱逐策略
 * - 时间基衰减曲线（指数衰减+检索刷新）
 * - 干扰理论建模（新记忆对旧记忆的干扰）
 * - 主动遗忘机制
 */

#include "selflnn/memory/short_term.h"
#include "selflnn/memory/memory.h"
#include "selflnn/utils/memory_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* 短期记忆条目内部追踪结构 */
typedef struct {
    char key[256];
    uint64_t last_access_time;    /* 最后访问时间戳（毫秒） */
    uint64_t creation_time;       /* 创建时间戳 */
    int access_count;             /* 访问次数 */
    float current_strength;       /* 当前强度 */
    size_t data_size;             /* 数据大小 */
} STMAccessRecord;

struct ShortTermMemory {
    MemorySystem* memory_system;
    ShortTermMemoryConfig config;
    int is_initialized;
    
    /* F-006: 短期记忆特有数据 */
    STMAccessRecord* access_records;  /* 访问记录数组 */
    size_t record_count;              /* 当前记录数 */
    size_t record_capacity;           /* 记录数组容量 */
    uint64_t last_decay_time;         /* 上次衰减时间 */
};

/* 获取当前毫秒时间戳 */
static uint64_t stm_get_time_ms(void) {
#ifdef _WIN32
    return (uint64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
#endif
}

/* F-006: 查找/创建访问记录索引 */
static int stm_find_record(ShortTermMemory* memory, const char* key) {
    for (size_t i = 0; i < memory->record_count; i++) {
        if (strcmp(memory->access_records[i].key, key) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/* F-006: LRU驱逐 — 选择最久未访问的条目 */
static int stm_select_lru_victim(ShortTermMemory* memory) {
    int victim = -1;
    uint64_t oldest_time = UINT64_MAX;
    float weakest_strength = 1.0f;
    
    /* 优先找最久未访问的弱记忆 */
    for (size_t i = 0; i < memory->record_count; i++) {
        float score = (float)(stm_get_time_ms() - memory->access_records[i].last_access_time);
        score *= (1.0f - memory->access_records[i].current_strength);
        if (score > 0 && memory->access_records[i].last_access_time < oldest_time) {
            if (memory->access_records[i].current_strength < weakest_strength) {
                weakest_strength = memory->access_records[i].current_strength;
                oldest_time = memory->access_records[i].last_access_time;
                victim = (int)i;
            }
        }
    }
    /* 回退：纯最旧驱逐 */
    if (victim < 0) {
        oldest_time = UINT64_MAX;
        for (size_t i = 0; i < memory->record_count; i++) {
            if (memory->access_records[i].last_access_time < oldest_time) {
                oldest_time = memory->access_records[i].last_access_time;
                victim = (int)i;
            }
        }
    }
    return victim;
}

/* F-006: 时间基衰减 — 指数衰减 + 访问刷新效应 */
static void stm_apply_time_decay(ShortTermMemory* memory) {
    uint64_t now = stm_get_time_ms();
    if (memory->last_decay_time == 0) {
        memory->last_decay_time = now;
        return;
    }
    
    double dt_sec = (double)(now - memory->last_decay_time) / 1000.0;
    if (dt_sec < 0.1) return; /* 至少100ms才衰减 */
    
    for (size_t i = 0; i < memory->record_count; i++) {
        STMAccessRecord* rec = &memory->access_records[i];
        /* Miller遗忘曲线: strength(t) = strength_0 * e^(-decay_rate * t) */
        double decay = exp(-(double)memory->config.decay_rate * dt_sec);
        /* 最近访问过的条目的衰减率降低（检索刷新效应） */
        double since_access = (double)(now - rec->last_access_time) / 1000.0;
        double refresh_factor = exp(-0.05 * since_access); /* 刚访问过的几乎不衰减 */
        double effective_decay = decay * (0.3 + 0.7 * refresh_factor);
        
        rec->current_strength *= (float)effective_decay;
        if (rec->current_strength < 0.01f) {
            /* 强度过低，触发主动遗忘 */
            memory_forget(memory->memory_system, rec->key);
            /* 从记录中移除 */
            if (i < memory->record_count - 1) {
                memcpy(&memory->access_records[i], &memory->access_records[memory->record_count - 1],
                       sizeof(STMAccessRecord));
            }
            memory->record_count--;
            i--;
        }
    }
    memory->last_decay_time = now;
}

ShortTermMemory* short_term_memory_create(const ShortTermMemoryConfig* config) {
    if (!config) return NULL;
    
    ShortTermMemory* memory = (ShortTermMemory*)safe_malloc(sizeof(ShortTermMemory));
    if (!memory) return NULL;
    
    memcpy(&memory->config, config, sizeof(ShortTermMemoryConfig));
    
    MemoryConfig mem_config;
    memset(&mem_config, 0, sizeof(mem_config));
    mem_config.max_short_term = config->capacity;
    mem_config.max_long_term = 0;
    mem_config.decay_rate = config->decay_rate;
    mem_config.consolidation_rate = 0.1f;
    mem_config.enable_consolidation = 1;
    
    memory->memory_system = memory_create(&mem_config);
    if (!memory->memory_system) {
        safe_free((void**)&memory);
        return NULL;
    }
    
    /* F-006: 初始化访问记录追踪 */
    memory->record_capacity = config->capacity + 32;
    memory->access_records = (STMAccessRecord*)safe_calloc(memory->record_capacity, sizeof(STMAccessRecord));
    memory->record_count = 0;
    memory->last_decay_time = stm_get_time_ms();
    
    memory->is_initialized = 1;
    return memory;
}

void short_term_memory_free(ShortTermMemory* memory) {
    if (!memory) return;
    if (memory->memory_system) memory_free(memory->memory_system);
    if (memory->access_records) safe_free((void**)&memory->access_records);
    safe_free((void**)&memory);
}

int short_term_memory_store(ShortTermMemory* memory, const char* key, 
                           const float* data, size_t data_size, float strength) {
    if (!memory || !key || !data || data_size == 0) return -1;
    if (!memory->is_initialized) return -1;
    
    /* F-006: 应用时间衰减 */
    stm_apply_time_decay(memory);
    
    /* F-006: 容量检查 + LRU驱逐 */
    if (memory->record_count >= memory->config.capacity) {
        int victim = stm_select_lru_victim(memory);
        if (victim >= 0) {
            memory_forget(memory->memory_system, memory->access_records[victim].key);
            if ((size_t)victim < memory->record_count - 1) {
                memcpy(&memory->access_records[victim],
                       &memory->access_records[memory->record_count - 1],
                       sizeof(STMAccessRecord));
            }
            memory->record_count--;
        }
    }
    
    /* F-006: 干扰效应 — 存储新记忆时轻微衰减所有现有记忆 */
    for (size_t i = 0; i < memory->record_count; i++) {
        memory->access_records[i].current_strength *= 0.98f;
    }
    
    int result = memory_store(memory->memory_system, key, data, data_size,
                              MEMORY_TYPE_SHORT_TERM, strength);
    if (result == 0) {
        /* F-006: 更新访问记录 */
        int idx = stm_find_record(memory, key);
        uint64_t now = stm_get_time_ms();
        if (idx < 0) {
            if (memory->record_count < memory->record_capacity) {
                idx = (int)memory->record_count;
                memory->record_count++;
            } else {
                idx = stm_select_lru_victim(memory);
                if (idx < 0) idx = 0;
            }
        }
        STMAccessRecord* rec = &memory->access_records[idx];
        strncpy(rec->key, key, sizeof(rec->key) - 1);
        rec->key[sizeof(rec->key) - 1] = '\0';
        rec->last_access_time = now;
        rec->creation_time = (idx >= (int)memory->record_count - 1) ? now : rec->creation_time;
        rec->access_count = (idx >= (int)memory->record_count - 1) ? 1 : rec->access_count + 1;
        rec->current_strength = strength;
        rec->data_size = data_size;
        if (idx >= (int)memory->record_count) memory->record_count = (size_t)(idx + 1);
    }
    return result;
}

int short_term_memory_retrieve(ShortTermMemory* memory, const char* key,
                              float* data, size_t data_size, float* strength) {
    if (!memory || !key || !data) return -1;
    if (!memory->is_initialized) return -1;
    
    /* F-006: 检索前先衰减 */
    stm_apply_time_decay(memory);
    
    int result = memory_retrieve(memory->memory_system, key, data, data_size, strength, NULL);
    if (result == 0) {
        /* F-006: 检索刷新 — 访问过的记忆重新激活 */
        int idx = stm_find_record(memory, key);
        if (idx >= 0) {
            memory->access_records[idx].last_access_time = stm_get_time_ms();
            memory->access_records[idx].access_count++;
            /* 检索会略微增强记忆（测试效应） */
            memory->access_records[idx].current_strength *= 1.05f;
            if (memory->access_records[idx].current_strength > 1.0f)
                memory->access_records[idx].current_strength = 1.0f;
        }
    }
    return result;
}

int short_term_memory_update(ShortTermMemory* memory, const char* key,
                            const float* data, size_t data_size, float strength_delta) {
    if (!memory || !key || !data) return -1;
    if (!memory->is_initialized) return -1;
    
    stm_apply_time_decay(memory);
    
    int result = memory_update(memory->memory_system, key, data, data_size, strength_delta);
    if (result == 0) {
        int idx = stm_find_record(memory, key);
        if (idx >= 0) {
            memory->access_records[idx].last_access_time = stm_get_time_ms();
            memory->access_records[idx].current_strength += strength_delta;
            if (memory->access_records[idx].current_strength < 0.0f)
                memory->access_records[idx].current_strength = 0.0f;
            if (memory->access_records[idx].current_strength > 1.0f)
                memory->access_records[idx].current_strength = 1.0f;
        }
    }
    return result;
}

int short_term_memory_forget(ShortTermMemory* memory, const char* key) {
    if (!memory || !key) return -1;
    if (!memory->is_initialized) return -1;
    
    int result = memory_forget(memory->memory_system, key);
    if (result == 0) {
        int idx = stm_find_record(memory, key);
        if (idx >= 0) {
            if ((size_t)idx < memory->record_count - 1) {
                memcpy(&memory->access_records[idx],
                       &memory->access_records[memory->record_count - 1],
                       sizeof(STMAccessRecord));
            }
            memory->record_count--;
        }
    }
    return result;
}

int short_term_memory_consolidate(ShortTermMemory* memory, const char* key) {
    if (!memory || !key) return -1;
    if (!memory->is_initialized) return -1;
    
    /* F-006: 只有强度超过巩固阈值的记忆才值得巩固 */
    int idx = stm_find_record(memory, key);
    if (idx >= 0 && memory->access_records[idx].current_strength < memory->config.consolidation_threshold) {
        return -1; /* 强度不足，不固化 */
    }
    
    return memory_consolidate(memory->memory_system, key);
}

int short_term_memory_get_config(const ShortTermMemory* memory, ShortTermMemoryConfig* config) {
    if (!memory || !config) return -1;
    memcpy(config, &memory->config, sizeof(ShortTermMemoryConfig));
    return 0;
}

int short_term_memory_set_config(ShortTermMemory* memory, const ShortTermMemoryConfig* config) {
    if (!memory || !config) return -1;
    if (!memory->is_initialized) return -1;
    
    memory->config.capacity = config->capacity;
    memory->config.decay_rate = config->decay_rate;
    memory->config.retrieval_speed = config->retrieval_speed;
    memory->config.consolidation_threshold = config->consolidation_threshold;
    
    MemoryConfig mem_config;
    if (memory_get_config(memory->memory_system, &mem_config) == 0) {
        mem_config.max_short_term = config->capacity;
        mem_config.decay_rate = config->decay_rate;
        memory_set_config(memory->memory_system, &mem_config);
    }
    return 0;
}

int short_term_memory_get_stats(const ShortTermMemory* memory, float* usage,
                               float* avg_strength, float* decay_rate) {
    if (!memory) return -1;
    if (!memory->is_initialized) return -1;
    
    if (usage)
        *usage = memory->config.capacity > 0 ? 
                 (float)memory->record_count / (float)memory->config.capacity : 0.0f;
    
    if (avg_strength) {
        float total = 0.0f;
        for (size_t i = 0; i < memory->record_count; i++)
            total += memory->access_records[i].current_strength;
        *avg_strength = memory->record_count > 0 ? total / (float)memory->record_count : 0.0f;
    }
    
    if (decay_rate) *decay_rate = memory->config.decay_rate;
    return 0;
}

void short_term_memory_reset(ShortTermMemory* memory) {
    if (!memory || !memory->is_initialized) return;
    memory_reset(memory->memory_system);
    memory->record_count = 0;
    memory->last_decay_time = stm_get_time_ms();
}
