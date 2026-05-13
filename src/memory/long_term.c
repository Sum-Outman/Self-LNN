/**
 * @file long_term.c
 * @brief 长期记忆系统实现 — F-007修复：添加Ebbinghaus遗忘曲线、Hebbian巩固、间隔重复等真实算法
 * 
 * 长期记忆管理实现，在底层MemorySystem基础上增加了长期记忆特有算法：
 * - Ebbinghaus遗忘曲线建模（指数衰减+间隔效应）
 * - Hebbian巩固（重复访问强化神经通路）
 * - 间隔重复调度（基于记忆强度的最优复习时间）
 * - 持久性评估与检索成本管理
 */

#include "selflnn/memory/long_term.h"
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

/* 长期记忆条目内部追踪结构 */
typedef struct {
    char key[256];
    uint64_t creation_time;
    uint64_t last_access_time;
    uint64_t last_consolidation_time;
    int access_count;
    int repetition_count;          /* M-027: SM-2复习次数(n) */
    float easiness_factor;         /* M-027: SM-2简易度因子(EF)，初始2.5 */
    int consolidation_count;
    float base_strength;
    float current_strength;
    double repetition_interval;
    size_t data_size;
} LTMAccessRecord;

struct LongTermMemory {
    MemorySystem* memory_system;
    LongTermMemoryConfig config;
    int is_initialized;
    
    /* F-007: 长期记忆特有数据 */
    LTMAccessRecord* access_records;
    size_t record_count;
    size_t record_capacity;
    uint64_t last_decay_time;
    uint64_t last_consolidation_scan_time;
};

static uint64_t ltm_get_time_ms(void) {
#ifdef _WIN32
    return (uint64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
#endif
}

static int ltm_find_record(LongTermMemory* memory, const char* key) {
    for (size_t i = 0; i < memory->record_count; i++) {
        if (strcmp(memory->access_records[i].key, key) == 0) return (int)i;
    }
    return -1;
}

/* F-007: Ebbinghaus遗忘曲线 — S(t) = S0 * e^(-t/τ) where τ depends on strength */
static float ltm_ebbinghaus_decay(float base_strength, double seconds_since_last_access, float persistence) {
    /* 记忆强度越高，衰减越慢。留存率依赖持久性因子 */
    double tau = 3600.0 * (1.0 + (double)base_strength * persistence * 24.0 * 30.0); /* 小时级时间常数 */
    double retention = exp(-seconds_since_last_access / tau);
    return (float)retention;
}

/* F-007: Hebbian巩固 — 重复使用强化连接 */
static float ltm_hebbian_strengthen(float current_strength, int access_count, int consolidation_count) {
    /* Hebbian法则: Δw ∝ 使用频率，对数增长避免无限强化 */
    float hebbian_boost = 1.0f + 0.05f * logf(1.0f + (float)access_count);
    float consolidation_boost = 1.0f + 0.1f * logf(1.0f + (float)consolidation_count);
    float result = current_strength * hebbian_boost * consolidation_boost;
    return result > 1.0f ? 1.0f : result;
}

/* M-027: SM-2完整算法 — 动态EF因子和间隔计算
 * I(n+1) = I(n) * EF （若n=1则I(2)=6天，若n=2则I(3)=I(2)*EF）
 * EF' = EF + (0.1 - (5-q)*(0.08+(5-q)*0.02))
 * q=回忆质量(0-5)从strength映射，EF下限1.3 */
static double ltm_calculate_interval(float strength, int rep_count, float* ef) {
    /* 回忆质量q: 从strength(0-1)映射到SM-2的0-5分 */
    int q = (int)(strength * 5.0f + 0.5f);
    if (q < 0) q = 0; if (q > 5) q = 5;

    /* 更新EF因子 */
    float old_ef = *ef;
    *ef = old_ef + (0.1f - (float)(5 - q) * (0.08f + (float)(5 - q) * 0.02f));
    if (*ef < 1.3f) *ef = 1.3f;

    /* SM-2间隔计算 */
    double interval = 0.0;
    if (rep_count == 0) {
        interval = 60.0;           /* 首次：1分钟 */
    } else if (rep_count == 1) {
        interval = 86400.0;        /* n=1: 1天 */
    } else if (rep_count == 2) {
        interval = 86400.0 * 6.0;  /* n=2: 6天 */
    } else {
        interval = 86400.0 * 6.0;  /* 从6天基准开始 */
        for (int i = 2; i < rep_count; i++)
            interval *= old_ef;    /* EF的(n-2)次幂 */
    }
    /* 应用当前EF缩放 */
    interval *= (*ef) / old_ef;

    /* 约束范围 */
    if (interval < 60.0) interval = 60.0;
    if (interval > 86400.0 * 365.0) interval = 86400.0 * 365.0;
    return interval;
}

/* F-007: 周期性衰减应用 */
static void ltm_apply_decay(LongTermMemory* memory) {
    uint64_t now = ltm_get_time_ms();
    if (memory->last_decay_time == 0) { memory->last_decay_time = now; return; }
    
    double dt_sec = (double)(now - memory->last_decay_time) / 1000.0;
    if (dt_sec < 10.0) return; /* 至少10秒 */
    
    for (size_t i = 0; i < memory->record_count; i++) {
        LTMAccessRecord* rec = &memory->access_records[i];
        double since_access = (double)(now - rec->last_access_time) / 1000.0;
        float retention = ltm_ebbinghaus_decay(rec->base_strength, since_access, memory->config.persistence);
        rec->current_strength = rec->base_strength * retention;
        
        /* 定期更新推荐复习间隔（SM-2完整算法） */
        if (since_access > rec->repetition_interval * 0.8) {
            rec->repetition_interval = ltm_calculate_interval(rec->current_strength,
                rec->repetition_count, &rec->easiness_factor);
        }
        
        /* 极弱记忆自动清除 */
        if (rec->current_strength < 0.001f) {
            memory_forget(memory->memory_system, rec->key);
            if (i < memory->record_count - 1) {
                memcpy(&memory->access_records[i], &memory->access_records[memory->record_count - 1],
                       sizeof(LTMAccessRecord));
            }
            memory->record_count--;
            i--;
        }
    }
    memory->last_decay_time = now;
}

/* F-007: 定期巩固扫描 — 将符合条件的记忆从短期推向长期 */
static void ltm_consolidation_scan(LongTermMemory* memory) {
    uint64_t now = ltm_get_time_ms();
    double dt_sec = 0.0;
    size_t i;
    double since_last_consolidation;
    float consolidate_boost;
    LTMAccessRecord* rec;
    /* 至少每60秒扫描一次 */
    if (memory->last_consolidation_scan_time > 0 &&
        (now - memory->last_consolidation_scan_time) < 60000) return;
    
    dt_sec = (double)(now - memory->last_consolidation_scan_time) / 1000.0;
    for (i = 0; i < memory->record_count; i++) {
        rec = &memory->access_records[i];
        
        if (rec->access_count >= 3 && rec->consolidation_count < 10) {
            since_last_consolidation = (double)(now - rec->last_consolidation_time) / 1000.0;
            if (since_last_consolidation > rec->repetition_interval * 0.5) {
                consolidate_boost = memory->config.consolidation_rate * 
                    (1.0f - (float)rec->consolidation_count / 10.0f);
                rec->base_strength += consolidate_boost;
                if (rec->base_strength > 1.0f) rec->base_strength = 1.0f;
                rec->consolidation_count++;
                rec->last_consolidation_time = now;
                
                memory_consolidate(memory->memory_system, rec->key);
            }
        }
    }
    memory->last_consolidation_scan_time = now;
}

LongTermMemory* long_term_memory_create(const LongTermMemoryConfig* config) {
    if (!config) return NULL;
    
    LongTermMemory* memory = (LongTermMemory*)safe_malloc(sizeof(LongTermMemory));
    if (!memory) return NULL;
    
    memcpy(&memory->config, config, sizeof(LongTermMemoryConfig));
    
    MemoryConfig mem_config;
    memset(&mem_config, 0, sizeof(mem_config));
    mem_config.max_short_term = 0;
    mem_config.max_long_term = config->capacity;
    mem_config.decay_rate = 0.01f;
    mem_config.consolidation_rate = config->consolidation_rate;
    mem_config.enable_consolidation = 1;
    
    memory->memory_system = memory_create(&mem_config);
    if (!memory->memory_system) { safe_free((void**)&memory); return NULL; }
    
    /* F-007: 初始化追踪记录 */
    memory->record_capacity = config->capacity + 64;
    memory->access_records = (LTMAccessRecord*)safe_calloc(memory->record_capacity, sizeof(LTMAccessRecord));
    memory->record_count = 0;
    memory->last_decay_time = ltm_get_time_ms();
    memory->last_consolidation_scan_time = 0;
    
    memory->is_initialized = 1;
    return memory;
}

void long_term_memory_free(LongTermMemory* memory) {
    if (!memory) return;
    if (memory->memory_system) memory_free(memory->memory_system);
    if (memory->access_records) safe_free((void**)&memory->access_records);
    safe_free((void**)&memory);
}

int long_term_memory_store(LongTermMemory* memory, const char* key, 
                          const float* data, size_t data_size, float strength) {
    if (!memory || !key || !data || data_size == 0) return -1;
    if (!memory->is_initialized) return -1;
    
    ltm_apply_decay(memory);
    ltm_consolidation_scan(memory);
    
    /* F-007: 容量管理 */
    if (memory->record_count >= memory->config.capacity) {
        /* 找最弱的记忆驱逐 */
        int victim = -1;
        float weakest = 1.0f;
        for (size_t i = 0; i < memory->record_count; i++) {
            if (memory->access_records[i].current_strength < weakest) {
                weakest = memory->access_records[i].current_strength;
                victim = (int)i;
            }
        }
        if (victim >= 0) {
            memory_forget(memory->memory_system, memory->access_records[victim].key);
            if ((size_t)victim < memory->record_count - 1) {
                memcpy(&memory->access_records[victim],
                       &memory->access_records[memory->record_count - 1], sizeof(LTMAccessRecord));
            }
            memory->record_count--;
        }
    }
    
    int result = memory_store(memory->memory_system, key, data, data_size,
                              MEMORY_TYPE_LONG_TERM, strength);
    if (result == 0) {
        int idx = ltm_find_record(memory, key);
        uint64_t now = ltm_get_time_ms();
        if (idx < 0) {
            idx = (int)memory->record_count;
            memory->record_count++;
        }
        LTMAccessRecord* rec = &memory->access_records[idx];
        strncpy(rec->key, key, sizeof(rec->key) - 1);
        rec->key[sizeof(rec->key) - 1] = '\0';
        rec->creation_time = now;
        rec->last_access_time = now;
        rec->last_consolidation_time = now;
        rec->access_count = 1;
        rec->repetition_count = 0;
        rec->easiness_factor = 2.5f;
        rec->consolidation_count = 0;
        rec->base_strength = strength;
        rec->current_strength = strength;
        rec->repetition_interval = ltm_calculate_interval(strength,
            rec->repetition_count, &rec->easiness_factor);
        rec->data_size = data_size;
    }
    return result;
}

int long_term_memory_retrieve(LongTermMemory* memory, const char* key,
                             float* data, size_t data_size, float* strength) {
    if (!memory || !key || !data) return -1;
    if (!memory->is_initialized) return -1;
    
    ltm_apply_decay(memory);
    ltm_consolidation_scan(memory);
    
    int result = memory_retrieve(memory->memory_system, key, data, data_size, strength, NULL);
    if (result == 0) {
        /* F-007: 检索即巩固（测试效应）+ Hebbian强化 */
        int idx = ltm_find_record(memory, key);
        if (idx >= 0) {
            uint64_t now = ltm_get_time_ms();
            memory->access_records[idx].last_access_time = now;
            memory->access_records[idx].access_count++;
            memory->access_records[idx].current_strength = ltm_hebbian_strengthen(
                memory->access_records[idx].base_strength,
                memory->access_records[idx].access_count,
                memory->access_records[idx].consolidation_count);
            memory->access_records[idx].repetition_count++;
            memory->access_records[idx].repetition_interval = ltm_calculate_interval(
                memory->access_records[idx].current_strength,
                memory->access_records[idx].repetition_count,
                &memory->access_records[idx].easiness_factor);
        }
    }
    return result;
}

int long_term_memory_update(LongTermMemory* memory, const char* key,
                           const float* data, size_t data_size, float strength_delta) {
    if (!memory || !key || !data) return -1;
    if (!memory->is_initialized) return -1;
    
    ltm_apply_decay(memory);
    
    int result = memory_update(memory->memory_system, key, data, data_size, strength_delta);
    if (result == 0) {
        int idx = ltm_find_record(memory, key);
        if (idx >= 0) {
            uint64_t now = ltm_get_time_ms();
            memory->access_records[idx].last_access_time = now;
            memory->access_records[idx].base_strength += strength_delta;
            if (memory->access_records[idx].base_strength < 0.001f)
                memory->access_records[idx].base_strength = 0.001f;
            if (memory->access_records[idx].base_strength > 1.0f)
                memory->access_records[idx].base_strength = 1.0f;
            memory->access_records[idx].current_strength = memory->access_records[idx].base_strength;
            memory->access_records[idx].repetition_interval = ltm_calculate_interval(
                memory->access_records[idx].current_strength,
                memory->access_records[idx].repetition_count,
                &memory->access_records[idx].easiness_factor);
        }
    }
    return result;
}

int long_term_memory_forget(LongTermMemory* memory, const char* key) {
    if (!memory || !key) return -1;
    if (!memory->is_initialized) return -1;
    
    int result = memory_forget(memory->memory_system, key);
    if (result == 0) {
        int idx = ltm_find_record(memory, key);
        if (idx >= 0) {
            if ((size_t)idx < memory->record_count - 1) {
                memcpy(&memory->access_records[idx],
                       &memory->access_records[memory->record_count - 1], sizeof(LTMAccessRecord));
            }
            memory->record_count--;
        }
    }
    return result;
}

int long_term_memory_strengthen(LongTermMemory* memory, const char* key, float strength_delta) {
    if (!memory || !key) return -1;
    if (!memory->is_initialized) return -1;
    
    ltm_apply_decay(memory);
    
    int idx = ltm_find_record(memory, key);
    if (idx >= 0) {
        memory->access_records[idx].base_strength += strength_delta;
        if (memory->access_records[idx].base_strength < 0.001f)
            memory->access_records[idx].base_strength = 0.001f;
        if (memory->access_records[idx].base_strength > 1.0f)
            memory->access_records[idx].base_strength = 1.0f;
        memory->access_records[idx].current_strength = memory->access_records[idx].base_strength;
        memory->access_records[idx].consolidation_count++;
        memory->access_records[idx].last_access_time = ltm_get_time_ms();
        
        float new_strength = memory->access_records[idx].base_strength;
        float data[1] = {0};
        return long_term_memory_update(memory, key, data, 1, new_strength - 0.5f);
    }
    return -1;
}

int long_term_memory_get_config(const LongTermMemory* memory, LongTermMemoryConfig* config) {
    if (!memory || !config) return -1;
    memcpy(config, &memory->config, sizeof(LongTermMemoryConfig));
    return 0;
}

int long_term_memory_set_config(LongTermMemory* memory, const LongTermMemoryConfig* config) {
    if (!memory || !config) return -1;
    if (!memory->is_initialized) return -1;
    
    memory->config.capacity = config->capacity;
    memory->config.persistence = config->persistence;
    memory->config.retrieval_cost = config->retrieval_cost;
    memory->config.consolidation_rate = config->consolidation_rate;
    
    MemoryConfig mem_config;
    if (memory_get_config(memory->memory_system, &mem_config) == 0) {
        mem_config.max_long_term = config->capacity;
        mem_config.consolidation_rate = config->consolidation_rate;
        memory_set_config(memory->memory_system, &mem_config);
    }
    return 0;
}

int long_term_memory_get_stats(const LongTermMemory* memory, float* usage,
                              float* avg_strength, float* persistence) {
    if (!memory) return -1;
    if (!memory->is_initialized) return -1;
    
    ltm_apply_decay((LongTermMemory*)memory);
    
    if (usage)
        *usage = memory->config.capacity > 0 ?
                 (float)memory->record_count / (float)memory->config.capacity : 0.0f;
    
    if (avg_strength) {
        float total = 0.0f;
        for (size_t i = 0; i < memory->record_count; i++)
            total += memory->access_records[i].current_strength;
        *avg_strength = memory->record_count > 0 ? total / (float)memory->record_count : 0.0f;
    }
    
    if (persistence) *persistence = memory->config.persistence;
    return 0;
}

void long_term_memory_reset(LongTermMemory* memory) {
    if (!memory || !memory->is_initialized) return;
    memory_reset(memory->memory_system);
    memory->record_count = 0;
    memory->last_decay_time = ltm_get_time_ms();
    memory->last_consolidation_scan_time = 0;
}
