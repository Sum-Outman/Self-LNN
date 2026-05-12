/**
 * @file perf.h
 * @brief 性能分析工具
 * 
 * 提供性能计时、统计和关键路径分析功能。
 */

#ifndef SELFLNN_PERF_H
#define SELFLNN_PERF_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 基础性能计时器
 * ============================================================================ */

/**
 * @brief 性能计时器结构
 */
typedef struct {
    uint64_t start_time;  /**< 开始时间 */
    uint64_t end_time;    /**< 结束时间 */
    uint64_t total_time;  /**< 总时间（纳秒） */
    int is_running;       /**< 是否正在运行 */
} PerfTimer;

/**
 * @brief 性能统计结构
 */
typedef struct {
    uint64_t min_time;    /**< 最小时间（纳秒） */
    uint64_t max_time;    /**< 最大时间（纳秒） */
    uint64_t total_time;  /**< 总时间（纳秒） */
    uint64_t count;       /**< 调用次数 */
    double avg_time;      /**< 平均时间（纳秒） */
} PerfStats;

/**
 * @brief 获取当前时间戳（纳秒）
 * 
 * @return uint64_t 当前时间戳（纳秒）
 */
uint64_t perf_timestamp_ns(void);

/**
 * @brief 初始化性能计时器
 * 
 * @param timer 计时器指针
 */
void perf_timer_init(PerfTimer* timer);

/**
 * @brief 开始计时
 * 
 * @param timer 计时器指针
 */
void perf_timer_start(PerfTimer* timer);

/**
 * @brief 停止计时
 * 
 * @param timer 计时器指针
 * @return uint64_t 经过的时间（纳秒）
 */
uint64_t perf_timer_stop(PerfTimer* timer);

/**
 * @brief 获取计时器经过的时间（不停止计时器）
 * 
 * @param timer 计时器指针
 * @return uint64_t 经过的时间（纳秒）
 */
uint64_t perf_timer_elapsed(const PerfTimer* timer);

/**
 * @brief 重置计时器
 * 
 * @param timer 计时器指针
 */
void perf_timer_reset(PerfTimer* timer);

/**
 * @brief 初始化性能统计
 * 
 * @param stats 统计指针
 */
void perf_stats_init(PerfStats* stats);

/**
 * @brief 更新性能统计
 * 
 * @param stats 统计指针
 * @param elapsed_time 经过的时间（纳秒）
 */
void perf_stats_update(PerfStats* stats, uint64_t elapsed_time);

/**
 * @brief 重置性能统计
 * 
 * @param stats 统计指针
 */
void perf_stats_reset(PerfStats* stats);

/**
 * @brief 获取性能统计字符串
 * 
 * @param stats 统计指针
 * @param buffer 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int perf_stats_to_string(const PerfStats* stats, char* buffer, size_t buffer_size);

/* ============================================================================
 * 关键路径性能分析器（Critical Path Profiler）
 *
 * 分层嵌套的性能分析器，用于追踪系统关键执行路径上的各阶段耗时。
 * 支持父子层级的 section 嵌套，自动构建调用树并生成分析报告。
 *
 * 使用方法：
 *   PerfProfiler profiler;
 *   perf_profiler_init(&profiler, "训练流水线");
 *
 *   perf_profiler_begin(&profiler, "数据加载");
 *   // ... 数据加载代码 ...
 *   perf_profiler_end(&profiler);
 *
 *   perf_profiler_begin(&profiler, "前向传播");
 *   perf_profiler_begin(&profiler, "卷积层");
 *   // ... 卷积 ...
 *   perf_profiler_end(&profiler);
 *   perf_profiler_begin(&profiler, "全连接层");
 *   // ... 全连接 ...
 *   perf_profiler_end(&profiler);
 *   perf_profiler_end(&profiler);
 *
 *   char report[4096];
 *   perf_profiler_dump(&profiler, report, sizeof(report));
 *   printf("%s", report);
 * ============================================================================ */

#define PERF_PROFILER_MAX_SECTIONS  128
#define PERF_SECTION_NAME_MAX       64
#define PERF_PROFILER_MAX_DEPTH     32

/**
 * @brief 关键路径段统计
 */
typedef struct {
    char name[PERF_SECTION_NAME_MAX];    /**< 段名称 */
    uint64_t total_time_ns;              /**< 总耗时（纳秒） */
    uint64_t min_time_ns;                /**< 最小耗时（纳秒） */
    uint64_t max_time_ns;                /**< 最大耗时（纳秒） */
    uint64_t call_count;                 /**< 调用次数 */
    uint64_t self_time_ns;               /**< 自身耗时（不含子段，纳秒） */
    int parent_index;                    /**< 父段索引，-1表示根段 */
    int child_indices[PERF_PROFILER_MAX_DEPTH]; /**< 子段索引列表 */
    int child_count;                     /**< 子段数量 */
} PerfSection;

/**
 * @brief 关键路径性能分析器
 */
typedef struct {
    PerfSection sections[PERF_PROFILER_MAX_SECTIONS]; /**< 段数组 */
    int section_count;                  /**< 当前段数量 */
    int depth_stack[PERF_PROFILER_MAX_DEPTH]; /**< 调用栈（段索引） */
    int stack_depth;                    /**< 当前栈深度 */
    uint64_t start_time_ns;             /**< 分析器启动时间 */
    uint64_t end_time_ns;               /**< 分析器结束时间 */
    int is_running;                     /**< 是否正在运行 */
    char name[PERF_SECTION_NAME_MAX];   /**< 分析器名称 */
    int enabled;                        /**< 是否启用（可动态开关） */
} PerfProfiler;

/**
 * @brief 初始化关键路径性能分析器
 *
 * @param profiler 分析器指针
 * @param name 分析器名称（用于报告标识）
 */
void perf_profiler_init(PerfProfiler* profiler, const char* name);

/**
 * @brief 开始一个关键路径段
 *
 * 在关键路径上标记一个阶段的开始。支持嵌套调用，必须与 perf_profiler_end 配对。
 *
 * @param profiler 分析器指针
 * @param section_name 段名称
 * @return int 成功返回0，失败返回-1（段数量超限或嵌套过深）
 */
int perf_profiler_begin(PerfProfiler* profiler, const char* section_name);

/**
 * @brief 结束当前关键路径段
 *
 * 结束最近 perf_profiler_begin 开始的段，记录耗时。
 *
 * @param profiler 分析器指针
 * @return int 成功返回0，失败返回-1（栈为空或不匹配）
 */
int perf_profiler_end(PerfProfiler* profiler);

/**
 * @brief 获取当前段的累计耗时（纳秒）
 *
 * @param profiler 分析器指针
 * @return uint64_t 当前段已耗时（纳秒），无活跃段返回0
 */
uint64_t perf_profiler_current_elapsed(const PerfProfiler* profiler);

/**
 * @brief 重置分析器
 *
 * 清除所有记录的段数据，但保留分析器名称和配置。
 *
 * @param profiler 分析器指针
 */
void perf_profiler_reset(PerfProfiler* profiler);

/**
 * @brief 启用/禁用分析器
 *
 * 禁用时，perf_profiler_begin/end 不记录数据，零开销。
 *
 * @param profiler 分析器指针
 * @param enabled 1=启用，0=禁用
 */
void perf_profiler_set_enabled(PerfProfiler* profiler, int enabled);

/**
 * @brief 生成关键路径分析报告
 *
 * 生成格式化的分析报告，包含：
 * - 总执行时间
 * - 每个段的调用次数、总耗时、平均耗时、最小/最大耗时
 * - 父子层级关系（缩进显示）
 * - 自身耗时（不含子段）
 * - 占总时间的百分比
 *
 * @param profiler 分析器指针
 * @param buffer 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int perf_profiler_dump(const PerfProfiler* profiler, char* buffer, size_t buffer_size);

/**
 * @brief 查找指定名称的段统计
 *
 * @param profiler 分析器指针
 * @param section_name 段名称
 * @return PerfSection* 段指针，未找到返回NULL
 */
const PerfSection* perf_profiler_find_section(const PerfProfiler* profiler,
                                              const char* section_name);

/**
 * @brief 获取分析器总耗时（纳秒）
 *
 * @param profiler 分析器指针
 * @return uint64_t 总耗时（纳秒）
 */
uint64_t perf_profiler_total_time(const PerfProfiler* profiler);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_PERF_H */
