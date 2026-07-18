/**
 * SELF-LNN 日志系统
 * 
 * 提供统一的日志记录功能，支持不同日志级别和输出目标。
 */

#ifndef SELFLNN_UTILS_LOGGING_H
#define SELFLNN_UTILS_LOGGING_H

#include "selflnn/core/common.h"
#include <stdio.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

// 日志级别枚举（在common.h中定义）

// 日志输出目标
typedef enum {
    LOG_TARGET_STDOUT = 0,   // 标准输出
    LOG_TARGET_STDERR = 1,   // 标准错误
    LOG_TARGET_FILE = 2,     // 文件
    LOG_TARGET_SYSLOG = 3,   // 系统日志
    LOG_TARGET_CUSTOM = 4    // 自定义回调
} LogTarget;

// 日志配置
typedef struct {
    LogLevel min_level;      // 最小日志级别
    LogTarget target;        // 输出目标
    const char* file_path;   // 文件路径（如果目标为文件）
    int enable_timestamp;    // 是否启用时间戳
    int enable_color;        // 是否启用颜色（仅控制台）
    int enable_thread_id;    // 是否显示线程ID
} LogConfig;

// 模块级日志过滤位掩码
#define LOG_MODULE_CORE        (1u << 0)
#define LOG_MODULE_LNN         (1u << 1)
#define LOG_MODULE_GPU         (1u << 2)
#define LOG_MODULE_MULTIMODAL  (1u << 3)
#define LOG_MODULE_KNOWLEDGE   (1u << 4)
#define LOG_MODULE_REASONING   (1u << 5)
#define LOG_MODULE_MEMORY      (1u << 6)
#define LOG_MODULE_LEARNING    (1u << 7)
#define LOG_MODULE_TRAINING    (1u << 8)
#define LOG_MODULE_ROBOT       (1u << 9)
#define LOG_MODULE_SAFETY      (1u << 10)
#define LOG_MODULE_EVOLUTION   (1u << 11)
#define LOG_MODULE_BACKEND     (1u << 12)
#define LOG_MODULE_COGNITION   (1u << 13)
#define LOG_MODULE_AGI         (1u << 14)
#define LOG_MODULE_ALL         (0xFFFFFFFFu)

// 日志回调函数类型
typedef void (*LogCallbackFunc)(LogLevel level, const char* message, void* user_data);

// 日志条目
typedef struct {
    LogLevel level;
    const char* message;
    const char* file;
    int line;
    const char* function;
    double timestamp;
} LogEntry;

// 初始化日志系统
SELFLNN_API int logging_init(const LogConfig* config);

// 关闭日志系统
SELFLNN_API void logging_shutdown(void);

// 设置日志级别
SELFLNN_API void logging_set_level(LogLevel level);

// 获取当前日志级别
SELFLNN_API LogLevel logging_get_level(void);

// 通过字符串设置日志级别（"DEBUG"/"INFO"/"WARNING"/"ERROR"/"FATAL"）
SELFLNN_API int logging_set_level_str(const char* level_str);

// 设置日志回调函数
SELFLNN_API void logging_set_callback(LogCallbackFunc callback, void* user_data);

// 设置模块日志过滤器（位掩码：0=禁用全部日志, LOG_MODULE_ALL=启用全部）
SELFLNN_API void logging_set_module_filter(unsigned int module_mask);

// 获取当前模块日志过滤器掩码
SELFLNN_API unsigned int logging_get_module_filter(void);

// 按模块输出的日志宏
#define log_module_debug(module, ...)    logging_log_module(LOG_LEVEL_DEBUG, module, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define log_module_info(module, ...)     logging_log_module(LOG_LEVEL_INFO, module, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define log_module_warning(module, ...)  logging_log_module(LOG_LEVEL_WARNING, module, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define log_module_error(module, ...)    logging_log_module(LOG_LEVEL_ERROR, module, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define log_module_fatal(module, ...)    logging_log_module(LOG_LEVEL_FATAL, module, __FILE__, __LINE__, __func__, __VA_ARGS__)

SELFLNN_API void logging_log_module(LogLevel level, unsigned int module, const char* file, int line, const char* function, const char* format, ...) SELFLNN_FORMAT(6, 7);

// 主日志函数
SELFLNN_API void logging_log(LogLevel level, const char* file, int line, const char* function, const char* format, ...) SELFLNN_FORMAT(5, 6);

// 简化日志宏
#define log_debug(...)    logging_log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define log_info(...)     logging_log(LOG_LEVEL_INFO, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define log_warning(...)  logging_log(LOG_LEVEL_WARNING, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define log_warn(...)     log_warning(__VA_ARGS__)  /* 别名：跨平台兼容 */
#define log_error(...)    logging_log(LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define log_fatal(...)    logging_log(LOG_LEVEL_FATAL, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_INFO(...)     log_info(__VA_ARGS__)   /* 大写别名：跨模块兼容 */

// 条件日志宏
#define log_if(cond, level, ...) \
    do { \
        if (cond) { \
            logging_log(level, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)

#define log_if_debug(cond, ...)    log_if(cond, LOG_LEVEL_DEBUG, __VA_ARGS__)
#define log_if_info(cond, ...)     log_if(cond, LOG_LEVEL_INFO, __VA_ARGS__)
#define log_if_warning(cond, ...)  log_if(cond, LOG_LEVEL_WARNING, __VA_ARGS__)
#define log_if_error(cond, ...)    log_if(cond, LOG_LEVEL_ERROR, __VA_ARGS__)

// 断言日志
#define log_assert(expr, ...) \
    do { \
        if (!(expr)) { \
            logging_log(LOG_LEVEL_FATAL, __FILE__, __LINE__, __func__, "断言失败: " __VA_ARGS__); \
        } \
    } while (0)

// 性能日志（用于性能关键代码）
// 修复: 将 start 变量提升到调用方作用域，使 log_perf_end 能获取到时间戳。
// 用法: log_perf_begin(myperf); ... log_perf_end("myperf", myperf_start);
#define log_perf_begin(name) double name##_start = logging_get_timestamp()

#define log_perf_end(name, start) \
    do { \
        double __perf_end = logging_get_timestamp(); \
        double __perf_duration = __perf_end - (start); \
        logging_log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__, \
                   "性能: %s 耗时 %.6f 秒", name, __perf_duration); \
    } while (0)

// 结构化日志（JSON格式）
SELFLNN_API void logging_log_json(LogLevel level, const char* file, int line, 
                                  const char* function, const char* message, ...);

#define log_json(level, message, ...) \
    logging_log_json(level, __FILE__, __LINE__, __func__, message, ##__VA_ARGS__)

// 获取当前时间戳（秒，高精度）
SELFLNN_API double logging_get_timestamp(void);

// 获取日志级别名称
SELFLNN_API const char* logging_get_level_name(LogLevel level);

// 获取日志级别颜色（ANSI转义序列）
SELFLNN_API const char* logging_get_level_color(LogLevel level);

// 设置线程名（用于日志输出）
SELFLNN_API void logging_set_thread_name(const char* name);

// 获取当前线程名
SELFLNN_API const char* logging_get_thread_name(void);

// 日志缓冲区管理
typedef struct LogBuffer LogBuffer;

SELFLNN_API LogBuffer* logging_buffer_create(size_t capacity);
SELFLNN_API void logging_buffer_destroy(LogBuffer* buffer);
SELFLNN_API void logging_buffer_append(LogBuffer* buffer, const LogEntry* entry);
SELFLNN_API size_t logging_buffer_get_count(const LogBuffer* buffer);
SELFLNN_API const LogEntry* logging_buffer_get_entry(const LogBuffer* buffer, size_t index);
SELFLNN_API void logging_buffer_clear(LogBuffer* buffer);

// 日志轮转（针对文件目标）
SELFLNN_API int logging_rotate_file(const char* new_path);

// 日志轮转自动配置
SELFLNN_API void logging_set_rotation(size_t max_bytes, int max_files);

// 默认日志配置
SELFLNN_API const LogConfig* logging_get_default_config(void);

// 全局日志函数（简化接口）
SELFLNN_API void selflnn_log(LogLevel level, const char* format, ...) SELFLNN_FORMAT(2, 3);

/* ================================================================
 * K-029: 日志系统增强 —— 运行时级别调整 + 文件重开 + 模块过滤
 * ================================================================ */

SELFLNN_API void logging_set_level_at_runtime(LogLevel level);

SELFLNN_API int logging_reopen_file(void);

SELFLNN_API size_t logging_get_file_size(void);

SELFLNN_API void logging_set_module_mask(unsigned int module_mask);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_UTILS_LOGGING_H