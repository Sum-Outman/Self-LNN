/**
 * @file memory.h
 * @brief 记忆系统接口
 * 
 * 统一记忆系统接口，支持短期记忆、长期记忆、情景记忆、语义记忆，
 * 以及工作记忆、记忆压缩和再巩固机制。
 */

#ifndef SELFLNN_MEMORY_H
#define SELFLNN_MEMORY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 记忆类型枚举
 */
typedef enum {
    MEMORY_TYPE_SHORT_TERM,    /**< 短期记忆 */
    MEMORY_TYPE_LONG_TERM,     /**< 长期记忆 */
    MEMORY_TYPE_EPISODIC,      /**< 情景记忆 */
    MEMORY_TYPE_SEMANTIC,      /**< 语义记忆 */
    MEMORY_TYPE_WORKING        /**< 工作记忆 */
} MemoryType;

/**
 * @brief 记忆项结构
 */
typedef struct {
    char* key;                 /**< 记忆键 */
    float* data;               /**< 记忆数据 */
    size_t data_size;          /**< 数据大小 */
    float strength;            /**< 记忆强度（0.0~1.0） */
    float timestamp;           /**< 创建/更新时间戳 */
    MemoryType type;           /**< 记忆类型 */
    float last_access_time;    /**< 最后访问时间（用于LRU淘汰策略） */
    size_t access_count;       /**< 访问次数（用于频率统计和热度评估） */
} MemoryItem;

/**
 * @brief 记忆系统配置
 */
typedef struct {
    size_t max_short_term;     /**< 短期记忆最大容量 */
    size_t max_long_term;      /**< 长期记忆最大容量 */
    float decay_rate;          /**< 衰减率 */
    float consolidation_rate;  /**< 巩固率 */
    int enable_consolidation;  /**< 是否启用记忆巩固 */

    size_t working_memory_size;    /**< 工作记忆容量（0表示禁用） */
    int enable_compression;        /**< 是否启用记忆压缩 */
    float compression_ratio;       /**< 压缩比率（0.0~1.0，1.0=无压缩） */
    int enable_reconsolidation;    /**< 是否启用再巩固机制 */
    float reconsolidation_boost;   /**< 再巩固强度提升值 */

    /* 睡眠巩固（记忆的睡眠模拟算法） */
    int enable_sleep_consolidation;      /**< 是否启用睡眠巩固 */
    float sleep_cycle_hours;             /**< 睡眠周期时长（模拟小时） */
    float nrem_consolidation_rate;       /**< NREM期巩固速率 */
    float rem_crosslink_rate;            /**< REM期跨连接速率 */
    float sleep_prune_threshold;         /**< 睡眠修剪阈值（低于此强度的记忆在NREM期被衰减） */
} MemoryConfig;

/**
 * @brief 记忆系统句柄
 */
typedef struct MemorySystem MemorySystem;

/**
 * @brief 工作记忆槽位
 */
typedef struct {
    char key[64];              /**< 槽位键名 */
    float* data;               /**< 槽位数据 */
    size_t data_size;          /**< 数据大小 */
    float focus;               /**< 注意力聚焦度（0.0~1.0） */
    float rehearsal_count;     /**< 复述次数 */
    float age;                 /**< 槽位存在时间 */
    int is_active;             /**< 是否激活 */
} WorkingMemorySlot;

/**
 * @brief 记忆压缩参数
 */
typedef struct {
    size_t original_dim;       /**< 原始维度 */
    size_t compressed_dim;     /**< 压缩后维度 */
    float* projection_matrix;  /**< 投影矩阵（压缩用） */
    float* reconstruction;     /**< 重构矩阵（解压用） */
    float compression_error;   /**< 压缩误差 */
    int is_compressed;         /**< 是否已压缩 */
} MemoryCompressionInfo;

/**
 * @brief 再巩固状态
 */
typedef enum {
    RECONSOLIDATION_NONE,       /**< 无需再巩固 */
    RECONSOLIDATION_LABILE,     /**< 不稳定状态（再巩固窗口期） */
    RECONSOLIDATION_IN_PROGRESS,/**< 再巩固进行中 */
    RECONSOLIDATION_COMPLETE   /**< 再巩固完成 */
} ReconsolidationState;

/**
 * @brief 再巩固条目
 */
typedef struct {
    char key[64];              /**< 记忆键 */
    ReconsolidationState state;/**< 再巩固状态 */
    float trigger_time;        /**< 触发时间 */
    float labile_duration;     /**< 不稳定窗口持续时间 */
    float original_strength;   /**< 原始强度 */
    float* labile_data;        /**< 不稳定状态数据副本 */
    size_t labile_data_size;   /**< 不稳定数据大小 */
    int is_active;             /**< 是否活跃 */
} ReconsolidationEntry;

/* ====================================================================
 * 基础记忆操作
 * ==================================================================== */

/**
 * @brief 创建记忆系统实例
 * 
 * @param config 记忆配置
 * @return MemorySystem* 记忆系统句柄，失败返回NULL
 */
MemorySystem* memory_create(const MemoryConfig* config);

/**
 * @brief 释放记忆系统实例
 * 
 * @param system 记忆系统句柄
 */
void memory_free(MemorySystem* system);

/**
 * @brief 存储记忆
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @param data 记忆数据
 * @param data_size 数据大小
 * @param type 记忆类型
 * @param strength 初始强度
 * @return int 成功返回0，失败返回-1
 */
int memory_store(MemorySystem* system, const char* key, const float* data,
                 size_t data_size, MemoryType type, float strength);

/**
 * @brief 检索记忆
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @param data 数据输出缓冲区
 * @param data_size 缓冲区大小
 * @param strength 记忆强度输出缓冲区
 * @param found_type 找到的记忆类型输出缓冲区（可为NULL）
 * @return int 成功返回0，未找到返回-1
 */
int memory_retrieve(MemorySystem* system, const char* key, float* data,
                    size_t data_size, float* strength, MemoryType* found_type);

/**
 * @brief 更新记忆
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @param data 新数据
 * @param data_size 数据大小
 * @param strength_delta 强度变化量
 * @return int 成功返回0，失败返回-1
 */
int memory_update(MemorySystem* system, const char* key, const float* data,
                  size_t data_size, float strength_delta);

/**
 * @brief 遗忘记忆
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @return int 成功返回0，失败返回-1
 */
int memory_forget(MemorySystem* system, const char* key);

/**
 * @brief 巩固记忆
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @return int 成功返回0，失败返回-1
 */
int memory_consolidate(MemorySystem* system, const char* key);

/* ====================================================================
 * 工作记忆操作
 * ==================================================================== */

/**
 * @brief 写入工作记忆
 * 
 * 将数据写入工作记忆槽位。如果槽位已满，根据注意力聚焦度淘汰。
 * 
 * @param system 记忆系统句柄
 * @param key 槽位键名
 * @param data 数据
 * @param data_size 数据大小
 * @param focus 注意力聚焦度（0.0~1.0）
 * @return int 成功返回槽位索引，失败返回-1
 */
int memory_working_store(MemorySystem* system, const char* key,
                        const float* data, size_t data_size, float focus);

/**
 * @brief 从工作记忆检索
 * 
 * @param system 记忆系统句柄
 * @param key 槽位键名
 * @param data 数据输出缓冲区
 * @param data_size 缓冲区大小
 * @param focus 注意力聚焦度输出
 * @return int 成功返回0，失败返回-1
 */
int memory_working_retrieve(MemorySystem* system, const char* key,
                           float* data, size_t data_size, float* focus);

/**
 * @brief 复述工作记忆项
 * 
 * 对工作记忆中的指定项进行复述，增强其保持。
 * 
 * @param system 记忆系统句柄
 * @param key 槽位键名
 * @return int 成功返回复述次数，失败返回-1
 */
int memory_rehearse(MemorySystem* system, const char* key);

/**
 * @brief 批量复述所有工作记忆项
 * 
 * @param system 记忆系统句柄
 * @return int 成功返回复述的槽位数，失败返回-1
 */
int memory_rehearse_all(MemorySystem* system);

/**
 * @brief 获取工作记忆状态
 * 
 * @param system 记忆系统句柄
 * @param slots 槽位数组输出缓冲区
 * @param max_slots 最大槽位数
 * @return int 成功返回活跃槽位数，失败返回-1
 */
int memory_working_get_state(MemorySystem* system, WorkingMemorySlot* slots, size_t max_slots);

/* ====================================================================
 * 记忆压缩操作
 * ==================================================================== */

/**
 * @brief 压缩指定记忆
 * 
 * 使用PCA风格的降维压缩算法，将高维记忆数据压缩为低维表示。
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @param target_ratio 目标压缩比率（0.0~1.0）
 * @return int 成功返回压缩后维度，失败返回-1
 */
int memory_compress(MemorySystem* system, const char* key, float target_ratio);

/**
 * @brief 解压指定记忆
 * 
 * 将压缩后的记忆数据恢复到近似原始维度。
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @return int 成功返回0，失败返回-1
 */
int memory_decompress(MemorySystem* system, const char* key);

/**
 * @brief 批量压缩所有可压缩记忆
 * 
 * @param system 记忆系统句柄
 * @param target_ratio 目标压缩比率
 * @return int 成功返回压缩的记忆数，失败返回-1
 */
int memory_bulk_compress(MemorySystem* system, float target_ratio);

/**
 * @brief 检查记忆是否已压缩
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @param info 压缩信息输出缓冲区
 * @return int 已压缩返回1，未压缩返回0，失败返回-1
 */
int memory_is_compressed(MemorySystem* system, const char* key, MemoryCompressionInfo* info);

/* ====================================================================
 * 再巩固操作
 * ==================================================================== */

/**
 * @brief 触发记忆再巩固
 * 
 * 使指定记忆进入不稳定状态，开启再巩固窗口期。
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @return int 成功返回0，失败返回-1
 */
int memory_trigger_reconsolidation(MemorySystem* system, const char* key);

/**
 * @brief 完成记忆再巩固
 * 
 * 在再巩固窗口期内完成记忆的再巩固，增强记忆强度。
 * 如果提供了新数据，会将其整合到再巩固的记忆中。
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @param new_data 新数据（可为NULL）
 * @param new_data_size 新数据大小
 * @return int 成功返回0，失败返回-1
 */
int memory_complete_reconsolidation(MemorySystem* system, const char* key,
                                   const float* new_data, size_t new_data_size);

/**
 * @brief 更新所有再巩固状态
 * 
 * 根据时间推移更新所有再巩固条目的状态。
 * 
 * @param system 记忆系统句柄
 * @param time_delta 时间增量
 * @return int 成功返回更新的条目数，失败返回-1
 */
int memory_update_reconsolidation(MemorySystem* system, float time_delta);

/**
 * @brief 获取再巩固状态
 * 
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @param entry 再巩固信息输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int memory_get_reconsolidation_state(MemorySystem* system, const char* key,
                                    ReconsolidationEntry* entry);

/* ====================================================================
 * 配置与统计操作
 * ==================================================================== */

/**
 * @brief 获取记忆系统配置
 * 
 * @param system 记忆系统句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int memory_get_config(const MemorySystem* system, MemoryConfig* config);

/**
 * @brief 设置记忆系统配置
 * 
 * @param system 记忆系统句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int memory_set_config(MemorySystem* system, const MemoryConfig* config);

/**
 * @brief 获取记忆系统统计信息
 * 
 * @param system 记忆系统句柄
 * @param total_items 总记忆项数输出缓冲区
 * @param avg_strength 平均强度输出缓冲区
 * @param consolidation_ratio 巩固率输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int memory_get_stats(const MemorySystem* system, size_t* total_items,
                     float* avg_strength, float* consolidation_ratio);

/**
 * @brief 销毁记忆系统（memory_free的别名，用于API兼容性）
 * 
 * @param system 记忆系统句柄
 */
void memory_system_destroy(MemorySystem* system);

/**
 * @brief 重置记忆系统
 * 
 * @param system 记忆系统句柄
 */
void memory_reset(MemorySystem* system);

/**
 * @brief 垃圾回收：清理强度为0的完全遗忘的记忆项
 * 
 * @param system 记忆系统句柄
 * @return size_t 回收的记忆项数量
 */
size_t memory_garbage_collect(MemorySystem* system);

/**
 * @brief 执行记忆的睡眠巩固（睡眠模拟算法）
 *
 * 模拟NREM-REM睡眠周期对记忆进行巩固：
 * - NREM期（前40%）：回放强化近期短期记忆，修剪弱连接
 * - REM期（后60%）：关联交叉连接相关记忆，形成语义抽象
 * - 高频共激活的记忆会建立强关联
 * - 强度低于修剪阈值的微弱记忆被衰减
 *
 * @param system 记忆系统句柄
 * @param sleep_duration 模拟睡眠时长（小时，0使用配置的默认值）
 * @param stats_output 统计输出缓冲区（[巩固数, 修剪数, 关联数, 语义数]，可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int memory_sleep_consolidation(MemorySystem* system, float sleep_duration, int stats_output[4]);

/* ====================================================================
 * 训练集成操作
 * ==================================================================== */

/**
 * @brief 从记忆中采样训练数据
 * 
 * 从指定类型的记忆中随机采样一批数据，用于LNN训练。
 * 记忆键中前缀为"input:"的作为输入数据，"target:"的作为目标数据。
 * 
 * @param system 记忆系统句柄
 * @param memory_type 记忆类型（MEMORY_TYPE_SHORT_TERM等）
 * @param batch_size 采样批次大小
 * @param data_dim 每条数据的特征维度
 * @param inputs 输出缓冲区（batch_size * data_dim个float）
 * @param targets 输出缓冲区（batch_size * data_dim个float）
 * @return int 成功返回实际采样的样本数，失败返回-1
 */
int memory_sample_training_batch(MemorySystem* system, MemoryType memory_type,
                                 size_t batch_size, size_t data_dim,
                                 float* inputs, float* targets);

/**
 * @brief 获取记忆类型中存储的条目数
 * 
 * @param system 记忆系统句柄
 * @param memory_type 记忆类型
 * @return size_t 条目数
 */
size_t memory_get_count(MemorySystem* system, MemoryType memory_type);

/**
 * @brief 按全局索引获取记忆项键名
 *
 * 遍历所有记忆池（短期→长期→情景→语义），按全局顺序获取指定索引的记忆键名。
 * 用于统计和枚举所有记忆条目。
 *
 * @param system 记忆系统句柄
 * @param index 全局索引（0至全部记忆项总数-1）
 * @param key_buf 键名输出缓冲区
 * @param key_buf_size 缓冲区大小
 * @return int 成功返回0，索引越界返回-1
 */
int memory_get_key(MemorySystem* system, size_t index, char* key_buf, size_t key_buf_size);

/**
 * @brief 基于相似度检索记忆上下文
 * 
 * 使用输入查询向量与所有记忆项计算点积相似度，
 * 返回 top_k 个最相似记忆项的加权聚合上下文向量。
 * 用于注入LNN前向传播，实现记忆增强的推理。
 * 
 * @param system 记忆系统句柄
 * @param query 查询向量（通常为LNN输入）
 * @param query_dim 查询向量维度
 * @param context_out 上下文输出缓冲区（必须为 query_dim 大小）
 * @param top_k 返回最相似的前K项
 * @param memory_types 要搜索的记忆类型数组（NULL表示搜索所有类型）
 * @param num_types 记忆类型数组大小
 * @param out_similarities 输出每个选中记忆的相似度数组（可选，大小为 top_k）
 * @return int 成功返回实际聚合的记忆数，失败返回-1
 */
int memory_retrieve_context(MemorySystem* system, const float* query, size_t query_dim,
                           float* context_out, size_t top_k,
                           const MemoryType* memory_types, size_t num_types,
                           float* out_similarities);

/* ====================================================================
 * Ebbinghaus遗忘曲线与间隔重复
 * ==================================================================== */

/**
 * @brief Ebbinghaus遗忘曲线统计
 */
typedef struct {
    float retention;              /**< 当前保留率（0.0~1.0） */
    float optimal_interval;       /**< 最佳复习间隔（时间单位） */
    float next_review_time;       /**< 下次复习时间 */
    float stability;              /**< 记忆稳定性（强度*访问次数） */
    float retrievability;         /**< 可检索性（0.0~1.0） */
    int is_stable;                /**< 是否已稳定（间隔重复>3次） */
} EbbinghausStats;

/**
 * @brief 获取Ebbinghaus遗忘曲线统计
 *
 * 基于记忆项的强度、访问次数和当前时间计算遗忘曲线参数。
 * 使用公式：R = S * exp(-t/τ)，其中：
 *   R = 保留率，S = 记忆强度，t = 经过时间，τ = 衰减时间常数
 * 衰减时间常数τ = base_τ * (1 + stability_factor * access_count)
 *
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @param stats 统计输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int memory_get_ebbinghaus_stats(MemorySystem* system, const char* key,
                                EbbinghausStats* stats);

/**
 * @brief 计算指定时间后的记忆保留率
 *
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @param time_elapsed 经过时间
 * @return float 保留率（0.0~1.0），失败返回-1.0f
 */
float memory_retention_at_time(MemorySystem* system, const char* key,
                               float time_elapsed);

/**
 * @brief 计算最佳复习间隔
 *
 * 基于目标保留率计算下次最佳复习时间。
 * 使用间隔重复的扩展效应：每次成功复习后，间隔指数增长。
 *
 * @param system 记忆系统句柄
 * @param key 记忆键
 * @param target_retention 目标保留率（0.0~1.0，推荐0.8~0.9）
 * @return float 最佳复习间隔（时间单位），失败返回-1.0f
 */
float memory_optimal_review_interval(MemorySystem* system, const char* key,
                                     float target_retention);

/* ====================================================================
 * 工作记忆增强
 * ==================================================================== */

/**
 * @brief 提高工作记忆槽位的优先级（聚焦）
 *
 * 将指定键的工作记忆槽位的注意力聚焦度提升。
 * 高聚焦度的槽位不会被低聚焦度槽位淘汰。
 *
 * @param system 记忆系统句柄
 * @param key 槽位键名
 * @param focus_boost 聚焦提升值（0.0~1.0）
 * @return int 成功返回0，失败返回-1
 */
int memory_working_prioritize(MemorySystem* system, const char* key,
                              float focus_boost);

/**
 * @brief 老化所有工作记忆槽位
 *
 * 对所有工作记忆槽位应用时间老化，降低其聚焦度和年龄。
 * 聚焦度低于阈值的槽位变为非激活状态。
 *
 * @param system 记忆系统句柄
 * @param time_delta 时间增量
 * @return int 成功返回变为非激活的槽位数，失败返回-1
 */
int memory_working_age_all(MemorySystem* system, float time_delta);

/* ====================================================================
 * 记忆整合与关联增强
 * ==================================================================== */

/**
 * @brief 批量巩固最高优先级的记忆
 *
 * 按（强度 × 访问频率）排序，巩固top_n个短期记忆到长期记忆。
 * 使用加权策略：强度高且访问频繁的记忆优先巩固。
 *
 * @param system 记忆系统句柄
 * @param top_n 要巩固的记忆数量（0表示巩固所有可达阈值的记忆）
 * @return int 成功返回实际巩固的记忆数，失败返回-1
 */
int memory_batch_consolidate(MemorySystem* system, size_t top_n);

/**
 * @brief 查找与指定记忆相关的记忆项
 *
 * 基于余弦相似度搜索与指定记忆相关的所有记忆项。
 * 返回加权聚合的上下文向量。
 *
 * @param system 记忆系统句柄
 * @param key 源记忆键
 * @param context_out 上下文输出缓冲区
 * @param context_dim 上下文维度
 * @param max_related 最大相关项数
 * @param threshold 相似度阈值（0.0~1.0）
 * @return int 成功返回找到的相关记忆数，失败返回-1
 */
int memory_find_related(MemorySystem* system, const char* key,
                        float* context_out, size_t context_dim,
                        size_t max_related, float threshold);

/* ====================================================================
 * 训练集成操作（增强）
 * ==================================================================== */

/**
 * @brief 从记忆中按优先级采样训练数据（增强版）
 *
 * 优先采样强度高、访问频率高的记忆项，并兼顾多样性。
 * 采样概率 = strength^2 * (1 + 0.5 * log(access_count + 1))
 *
 * @param system 记忆系统句柄
 * @param memory_type 记忆类型
 * @param batch_size 采样批次大小
 * @param data_dim 数据维度
 * @param inputs 输入输出缓冲区
 * @param targets 目标输出缓冲区
 * @param min_strength 最低强度过滤
 * @return int 成功返回实际采样的样本数，失败返回-1
 */
int memory_sample_priority_batch(MemorySystem* system, MemoryType memory_type,
                                 size_t batch_size, size_t data_dim,
                                 float* inputs, float* targets,
                                 float min_strength);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_MEMORY_H
