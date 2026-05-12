/**
 * @file perf.c
 * @brief 性能分析工具实现
 * 
 * 提供性能计时、统计和关键路径分析功能实现。
 */

#include "selflnn/utils/perf.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#include <sys/time.h>
#endif

/* ============================================================================
 * 时间戳
 * ============================================================================ */

/**
 * @brief 获取当前时间戳（纳秒）
 */
uint64_t perf_timestamp_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);

    return (uint64_t)((counter.QuadPart * 1000000000ULL) / frequency.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/* ============================================================================
 * 基础性能计时器
 * ============================================================================ */

/**
 * @brief 初始化性能计时器
 */
void perf_timer_init(PerfTimer* timer) {
    if (!timer) {
        return;
    }

    memset(timer, 0, sizeof(PerfTimer));
}

/**
 * @brief 开始计时
 */
void perf_timer_start(PerfTimer* timer) {
    if (!timer) {
        return;
    }

    timer->start_time = perf_timestamp_ns();
    timer->is_running = 1;
}

/**
 * @brief 停止计时
 */
uint64_t perf_timer_stop(PerfTimer* timer) {
    if (!timer || !timer->is_running) {
        return 0;
    }

    timer->end_time = perf_timestamp_ns();
    timer->total_time = timer->end_time - timer->start_time;
    timer->is_running = 0;

    return timer->total_time;
}

/**
 * @brief 获取计时器经过的时间（不停止计时器）
 */
uint64_t perf_timer_elapsed(const PerfTimer* timer) {
    if (!timer || !timer->is_running) {
        return 0;
    }

    uint64_t current_time = perf_timestamp_ns();
    return current_time - timer->start_time;
}

/**
 * @brief 重置计时器
 */
void perf_timer_reset(PerfTimer* timer) {
    if (!timer) {
        return;
    }

    memset(timer, 0, sizeof(PerfTimer));
}

/**
 * @brief 初始化性能统计
 */
void perf_stats_init(PerfStats* stats) {
    if (!stats) {
        return;
    }

    memset(stats, 0, sizeof(PerfStats));
    stats->min_time = UINT64_MAX;
}

/**
 * @brief 更新性能统计
 */
void perf_stats_update(PerfStats* stats, uint64_t elapsed_time) {
    if (!stats) {
        return;
    }

    if (elapsed_time < stats->min_time) {
        stats->min_time = elapsed_time;
    }

    if (elapsed_time > stats->max_time) {
        stats->max_time = elapsed_time;
    }

    stats->total_time += elapsed_time;
    stats->count++;

    if (stats->count > 0) {
        stats->avg_time = (double)stats->total_time / (double)stats->count;
    }
}

/**
 * @brief 重置性能统计
 */
void perf_stats_reset(PerfStats* stats) {
    if (!stats) {
        return;
    }

    memset(stats, 0, sizeof(PerfStats));
    stats->min_time = UINT64_MAX;
}

/**
 * @brief 获取性能统计字符串
 */
int perf_stats_to_string(const PerfStats* stats, char* buffer, size_t buffer_size) {
    if (!stats || !buffer || buffer_size == 0) {
        return -1;
    }

    const char* unit = "ns";
    double min_time = (double)stats->min_time;
    double max_time = (double)stats->max_time;
    double avg_time = stats->avg_time;
    double total_time = (double)stats->total_time;

    if (total_time > 1000000000.0) {
        min_time /= 1000000000.0;
        max_time /= 1000000000.0;
        avg_time /= 1000000000.0;
        total_time /= 1000000000.0;
        unit = "s";
    } else if (total_time > 1000000.0) {
        min_time /= 1000000.0;
        max_time /= 1000000.0;
        avg_time /= 1000000.0;
        total_time /= 1000000.0;
        unit = "ms";
    } else if (total_time > 1000.0) {
        min_time /= 1000.0;
        max_time /= 1000.0;
        avg_time /= 1000.0;
        total_time /= 1000.0;
        unit = "\xce\xbcs";
    }

    int written = snprintf(buffer, buffer_size,
        "调用次数: %llu, 总时间: %.2f%s, 平均: %.2f%s, 最小: %.2f%s, 最大: %.2f%s",
        (unsigned long long)stats->count,
        total_time, unit,
        avg_time, unit,
        min_time, unit,
        max_time, unit);

    if (written < 0 || (size_t)written >= buffer_size) {
        return -1;
    }

    return 0;
}

/* ============================================================================
 * 关键路径性能分析器
 * ============================================================================ */

/**
 * @brief 查找或创建段
 *
 * 在父段下查找指定名称的子段，如果不存在则创建。
 *
 * @param sections 段数组
 * @param section_count 当前段数量（输入输出参数）
 * @param parent_index 父段索引
 * @param name 段名称
 * @return int 段索引，失败返回-1
 */
static int perf_profiler_find_or_create_section(PerfSection* sections,
                                                int* section_count,
                                                int parent_index,
                                                const char* name) {
    if (!sections || !section_count || !name) return -1;

    for (int i = 0; i < *section_count; i++) {
        if (sections[i].parent_index == parent_index &&
            strncmp(sections[i].name, name, PERF_SECTION_NAME_MAX - 1) == 0) {
            return i;
        }
    }

    if (*section_count >= PERF_PROFILER_MAX_SECTIONS) return -1;

    int idx = (*section_count)++;
    memset(&sections[idx], 0, sizeof(PerfSection));
    strncpy(sections[idx].name, name, PERF_SECTION_NAME_MAX - 1);
    sections[idx].name[PERF_SECTION_NAME_MAX - 1] = '\0';
    sections[idx].min_time_ns = UINT64_MAX;
    sections[idx].max_time_ns = 0;
    sections[idx].parent_index = parent_index;

    if (parent_index >= 0 && parent_index < idx) {
        PerfSection* parent = &sections[parent_index];
        if (parent->child_count < PERF_PROFILER_MAX_DEPTH) {
            parent->child_indices[parent->child_count++] = idx;
        }
    }

    return idx;
}

/**
 * @brief 初始化关键路径性能分析器
 */
void perf_profiler_init(PerfProfiler* profiler, const char* name) {
    if (!profiler) return;

    memset(profiler, 0, sizeof(PerfProfiler));
    profiler->enabled = 1;
    profiler->stack_depth = 0;
    profiler->section_count = 0;
    profiler->is_running = 0;

    if (name) {
        strncpy(profiler->name, name, PERF_SECTION_NAME_MAX - 1);
        profiler->name[PERF_SECTION_NAME_MAX - 1] = '\0';
    } else {
        strncpy(profiler->name, "未命名分析器", PERF_SECTION_NAME_MAX - 1);
        profiler->name[PERF_SECTION_NAME_MAX - 1] = '\0';
    }
}

/**
 * @brief 开始一个关键路径段
 */
int perf_profiler_begin(PerfProfiler* profiler, const char* section_name) {
    if (!profiler || !section_name || !profiler->enabled) return -1;

    int parent_index = -1;
    if (profiler->stack_depth > 0) {
        parent_index = profiler->depth_stack[profiler->stack_depth - 1];
    }

    int section_idx = perf_profiler_find_or_create_section(
        profiler->sections, &profiler->section_count,
        parent_index, section_name);

    if (section_idx < 0) return -1;

    if (profiler->stack_depth >= PERF_PROFILER_MAX_DEPTH) return -1;
    profiler->depth_stack[profiler->stack_depth++] = section_idx;

    if (!profiler->is_running) {
        profiler->start_time_ns = perf_timestamp_ns();
        profiler->is_running = 1;
    }

    return 0;
}

/**
 * @brief 结束当前关键路径段
 */
int perf_profiler_end(PerfProfiler* profiler) {
    if (!profiler || !profiler->enabled) return -1;
    if (profiler->stack_depth <= 0) return -1;

    uint64_t now = perf_timestamp_ns();
    int section_idx = profiler->depth_stack[profiler->stack_depth - 1];
    profiler->stack_depth--;

    if (section_idx < 0 || section_idx >= profiler->section_count) return -1;

    PerfSection* section = &profiler->sections[section_idx];
    uint64_t elapsed = now - profiler->start_time_ns;

    if (profiler->stack_depth == 0) {
        profiler->end_time_ns = now;
        profiler->is_running = 0;
    }

    uint64_t cumulative_child_time = 0;
    for (int i = 0; i < section->child_count; i++) {
        int child_idx = section->child_indices[i];
        if (child_idx >= 0 && child_idx < profiler->section_count) {
            cumulative_child_time += profiler->sections[child_idx].total_time_ns;
        }
    }

    section->total_time_ns += elapsed;
    if (elapsed < section->min_time_ns) section->min_time_ns = elapsed;
    if (elapsed > section->max_time_ns) section->max_time_ns = elapsed;
    section->call_count++;

    uint64_t self_time = elapsed;
    if (self_time > cumulative_child_time) {
        self_time -= cumulative_child_time;
    } else {
        self_time = 0;
    }
    section->self_time_ns += self_time;

    return 0;
}

/**
 * @brief 获取当前段的累计耗时（纳秒）
 */
uint64_t perf_profiler_current_elapsed(const PerfProfiler* profiler) {
    if (!profiler || !profiler->enabled || profiler->stack_depth <= 0) return 0;

    int section_idx = profiler->depth_stack[profiler->stack_depth - 1];
    if (section_idx < 0 || section_idx >= profiler->section_count) return 0;

    uint64_t now = perf_timestamp_ns();
    return now - profiler->start_time_ns;
}

/**
 * @brief 重置分析器
 */
void perf_profiler_reset(PerfProfiler* profiler) {
    if (!profiler) return;

    char saved_name[PERF_SECTION_NAME_MAX];
    strncpy(saved_name, profiler->name, PERF_SECTION_NAME_MAX - 1);
    saved_name[PERF_SECTION_NAME_MAX - 1] = '\0';
    int saved_enabled = profiler->enabled;

    memset(profiler, 0, sizeof(PerfProfiler));
    profiler->enabled = saved_enabled;
    profiler->stack_depth = 0;
    profiler->section_count = 0;
    profiler->is_running = 0;
    strncpy(profiler->name, saved_name, PERF_SECTION_NAME_MAX - 1);
    profiler->name[PERF_SECTION_NAME_MAX - 1] = '\0';
}

/**
 * @brief 启用/禁用分析器
 */
void perf_profiler_set_enabled(PerfProfiler* profiler, int enabled) {
    if (!profiler) return;
    profiler->enabled = enabled ? 1 : 0;
}

/**
 * @brief 格式化时间
 *
 * 根据时间大小自动选择单位，并格式化为字符串。
 */
static const char* perf_format_time_ns(uint64_t ns, char* buf, size_t buf_size) {
    if (!buf || buf_size == 0) return "";

    if (ns >= 1000000000ULL) {
        snprintf(buf, buf_size, "%.2f s", (double)ns / 1000000000.0);
    } else if (ns >= 1000000ULL) {
        snprintf(buf, buf_size, "%.2f ms", (double)ns / 1000000.0);
    } else if (ns >= 1000ULL) {
        snprintf(buf, buf_size, "%.2f \xc2\xb5s", (double)ns / 1000.0);
    } else {
        snprintf(buf, buf_size, "%llu ns", (unsigned long long)ns);
    }
    return buf;
}

/**
 * @brief 递归输出段报告
 *
 * @param sections 段数组
 * @param section_idx 当前段索引
 * @param depth 当前缩进深度
 * @param total_time 总时间
 * @param buffer 输出缓冲区（输入输出）
 * @param pos 当前位置（输入输出）
 * @param buffer_size 缓冲区大小
 */
static void perf_profiler_dump_section(const PerfSection* sections,
                                       int section_idx,
                                       int depth,
                                       uint64_t total_time,
                                       char* buffer,
                                       size_t* pos,
                                       size_t buffer_size) {
    if (!sections || section_idx < 0 || !buffer || !pos) return;

    const PerfSection* sec = &sections[section_idx];

    char time_str[32];
    char avg_str[32];
    char min_str[32];
    char max_str[32];
    char self_str[32];

    perf_format_time_ns(sec->total_time_ns, time_str, sizeof(time_str));
    perf_format_time_ns(
        sec->call_count > 0 ? sec->total_time_ns / sec->call_count : 0,
        avg_str, sizeof(avg_str));
    perf_format_time_ns(sec->min_time_ns, min_str, sizeof(min_str));
    perf_format_time_ns(sec->max_time_ns, max_str, sizeof(max_str));
    perf_format_time_ns(sec->self_time_ns, self_str, sizeof(self_str));

    double pct = total_time > 0
        ? (double)sec->total_time_ns / (double)total_time * 100.0
        : 0.0;

    int indent = depth * 2;
    if (indent > 40) indent = 40;

    int written = snprintf(buffer + *pos, buffer_size > *pos ? buffer_size - *pos : 0,
        "%*s- %s\n"
        "%*s  调用次数: %llu  |  总时间: %s  |  平均: %s\n"
        "%*s  最小: %s  |  最大: %s  |  自身: %s  |  占比: %.1f%%\n",
        indent, "", sec->name,
        indent, "",
        (unsigned long long)sec->call_count,
        time_str, avg_str,
        indent, "",
        min_str, max_str, self_str, pct);

    if (written > 0) {
        *pos += (size_t)written;
        if (*pos >= buffer_size) *pos = buffer_size - 1;
    }

    for (int i = 0; i < sec->child_count; i++) {
        int child_idx = sec->child_indices[i];
        if (child_idx >= 0) {
            perf_profiler_dump_section(sections, child_idx, depth + 1,
                                       total_time, buffer, pos, buffer_size);
        }
    }
}

/**
 * @brief 生成关键路径分析报告
 */
int perf_profiler_dump(const PerfProfiler* profiler, char* buffer, size_t buffer_size) {
    if (!profiler || !buffer || buffer_size == 0) return -1;

    buffer[0] = '\0';
    size_t pos = 0;

    uint64_t total = perf_profiler_total_time(profiler);

    char total_str[32];
    perf_format_time_ns(total, total_str, sizeof(total_str));

    int hdr_written = snprintf(buffer + pos, buffer_size - pos,
        "============================================================\n"
        "  关键路径分析报告: %s\n"
        "  总执行时间: %s  |  段数量: %d  |  已启用: %s\n"
        "============================================================\n",
        profiler->name,
        total_str,
        profiler->section_count,
        profiler->enabled ? "是" : "否");

    if (hdr_written > 0) {
        pos += (size_t)hdr_written;
        if (pos >= buffer_size) pos = buffer_size - 1;
    }

    for (int i = 0; i < profiler->section_count; i++) {
        if (profiler->sections[i].parent_index == -1) {
            perf_profiler_dump_section(profiler->sections, i, 0,
                                       total, buffer, &pos, buffer_size);
        }
    }

    if (pos < buffer_size) {
        int footer_written = snprintf(buffer + pos, buffer_size - pos,
            "============================================================\n");
        if (footer_written > 0) {
            pos += (size_t)footer_written;
            if (pos >= buffer_size) pos = buffer_size - 1;
        }
    }

    buffer[pos] = '\0';
    return 0;
}

/**
 * @brief 查找指定名称的段统计
 */
const PerfSection* perf_profiler_find_section(const PerfProfiler* profiler,
                                              const char* section_name) {
    if (!profiler || !section_name) return NULL;

    for (int i = 0; i < profiler->section_count; i++) {
        if (strncmp(profiler->sections[i].name, section_name,
                    PERF_SECTION_NAME_MAX - 1) == 0) {
            return &profiler->sections[i];
        }
    }

    return NULL;
}

/**
 * @brief 获取分析器总耗时（纳秒）
 */
uint64_t perf_profiler_total_time(const PerfProfiler* profiler) {
    if (!profiler || profiler->section_count == 0) return 0;

    uint64_t max_end = profiler->end_time_ns;
    if (profiler->is_running) {
        max_end = perf_timestamp_ns();
    }
    if (max_end > profiler->start_time_ns) {
        return max_end - profiler->start_time_ns;
    }

    uint64_t total = 0;
    for (int i = 0; i < profiler->section_count; i++) {
        if (profiler->sections[i].parent_index == -1) {
            total += profiler->sections[i].total_time_ns;
        }
    }

    return total;
}
