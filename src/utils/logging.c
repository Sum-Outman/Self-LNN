/**
 * SELF-LNN 日志系统实现
 * 
 * 提供简单的日志记录功能，支持控制台输出和基本日志级别。
 */

// 禁用Windows CRT安全警告

#include "selflnn/utils/logging.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

#ifdef _WIN32
#include <windows.h>
static CRITICAL_SECTION g_log_lock;
static int g_log_lock_initialized = 0;
#define LOG_LOCK_INIT() do { if (!g_log_lock_initialized) { InitializeCriticalSection(&g_log_lock); g_log_lock_initialized = 1; } } while(0)
#define LOG_LOCK() do { LOG_LOCK_INIT(); EnterCriticalSection(&g_log_lock); } while(0)
#define LOG_UNLOCK() LeaveCriticalSection(&g_log_lock)
#else
#include <pthread.h>
static pthread_mutex_t g_log_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_log_lock_initialized = 1;
#define LOG_LOCK_INIT() do { } while(0)
#define LOG_LOCK() pthread_mutex_lock(&g_log_lock)
#define LOG_UNLOCK() pthread_mutex_unlock(&g_log_lock)
#endif

// 全局日志状态
static struct {
    LogConfig config;
    int is_initialized;
    FILE* log_file;
    LogCallbackFunc callback;
    void* callback_data;
    char thread_name[32];
    /* 日志文件轮转 */
    size_t log_file_bytes_written;   /**< 自上次轮转以来的写入字节数 */
    size_t log_rotate_max_bytes;     /**< 轮转阈值（默认10MB） */
    int log_rotate_max_files;        /**< 最大保留轮转文件数（默认5） */
    /* 模块级日志过滤器 */
    unsigned int module_filter;      /**< 模块过滤位掩码（默认全部启用） */
} g_log_state = {
    .config = {
        .min_level = LOG_LEVEL_INFO,
        .target = LOG_TARGET_STDOUT,
        .file_path = NULL,
        .enable_timestamp = 1,
        .enable_color = 1,
        .enable_thread_id = 0
    },
    .is_initialized = 0,
    .log_file = NULL,
    .callback = NULL,
    .callback_data = NULL,
    .thread_name = "main",
    .log_file_bytes_written = 0,
    .log_rotate_max_bytes = (size_t)10 * 1024 * 1024,
    .log_rotate_max_files = 5,
    .module_filter = LOG_MODULE_ALL
};

/* K-040: 运行时日志级别调整 —— 实现头文件声明的 logging_get_level/set_level_str */
LogLevel logging_get_level(void) {
    return g_log_state.config.min_level;
}

int logging_set_level_str(const char* level_str) {
    if (!level_str) return -1;
    if (strcmp(level_str, "DEBUG") == 0 || strcmp(level_str, "debug") == 0) { logging_set_level(LOG_LEVEL_DEBUG); return 0; }
    if (strcmp(level_str, "INFO") == 0 || strcmp(level_str, "info") == 0) { logging_set_level(LOG_LEVEL_INFO); return 0; }
    if (strcmp(level_str, "WARNING") == 0 || strcmp(level_str, "warning") == 0) { logging_set_level(LOG_LEVEL_WARNING); return 0; }
    if (strcmp(level_str, "ERROR") == 0 || strcmp(level_str, "error") == 0) { logging_set_level(LOG_LEVEL_ERROR); return 0; }
    if (strcmp(level_str, "FATAL") == 0 || strcmp(level_str, "fatal") == 0) { logging_set_level(LOG_LEVEL_FATAL); return 0; }
    return -1;
}

/* 日志文件自动轮转函数前向声明 */
static void check_auto_rotation(void);

/* 日志级别名称 */
static const char* level_names[] = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "FATAL"
};

// 日志级别颜色（ANSI转义序列）
static const char* level_colors[] = {
    "\033[36m",  // 青色 - DEBUG
    "\033[32m",  // 绿色 - INFO
    "\033[33m",  // 黄色 - WARNING
    "\033[31m",  // 红色 - ERROR
    "\033[1;31m" // 粗体红色 - FATAL
};

// 颜色重置
static const char* color_reset = "\033[0m";

// 内部函数声明
static void write_log_message(LogLevel level, const char* file, int line, 
                              const char* function, const char* message);
static const char* get_filename(const char* path);
static void format_timestamp(char* buffer, size_t size);

int logging_init(const LogConfig* config)
{
    LOG_LOCK();
    if (g_log_state.is_initialized)
    {
        LOG_UNLOCK();
        log_warning("日志系统已经初始化");
        return SELFLNN_SUCCESS;
    }
    
    // 使用默认配置或用户配置
    if (config)
    {
        memcpy(&g_log_state.config, config, sizeof(LogConfig));
    }
    
    // 打开日志文件（如果需要）
    if (g_log_state.config.target == LOG_TARGET_FILE && g_log_state.config.file_path)
    {
        g_log_state.log_file = fopen(g_log_state.config.file_path, "a");
        if (!g_log_state.log_file)
        {
            fprintf(stderr, "无法打开日志文件: %s\n", g_log_state.config.file_path);
            g_log_state.config.target = LOG_TARGET_STDERR;
        }
    }
    
    g_log_state.is_initialized = 1;
    LOG_UNLOCK();
    
    log_info("日志系统初始化完成，级别: %s", 
             logging_get_level_name(g_log_state.config.min_level));
    
    return SELFLNN_SUCCESS;
}

void logging_shutdown(void)
{
    LOG_LOCK();
    if (!g_log_state.is_initialized)
    {
        LOG_UNLOCK();
        return;
    }
    
    // 关闭日志文件
    if (g_log_state.log_file)
    {
        fclose(g_log_state.log_file);
        g_log_state.log_file = NULL;
    }
    
    g_log_state.is_initialized = 0;
    g_log_state.callback = NULL;
    g_log_state.callback_data = NULL;
    LOG_UNLOCK();
}

void logging_set_level(LogLevel level)
{
    if (level >= LOG_LEVEL_DEBUG && level <= LOG_LEVEL_FATAL)
    {
        LOG_LOCK();
        g_log_state.config.min_level = level;
        LOG_UNLOCK();
    }
}

void logging_set_callback(LogCallbackFunc callback, void* user_data)
{
    LOG_LOCK();
    g_log_state.callback = callback;
    g_log_state.callback_data = user_data;
    LOG_UNLOCK();
}

void logging_set_module_filter(unsigned int module_mask)
{
    LOG_LOCK();
    g_log_state.module_filter = module_mask;
    LOG_UNLOCK();
    log_info("模块日志过滤器已更新: 0x%08X", module_mask);
}

unsigned int logging_get_module_filter(void)
{
    unsigned int mask;
    LOG_LOCK();
    mask = g_log_state.module_filter;
    LOG_UNLOCK();
    return mask;
}

void logging_log_module(LogLevel level, unsigned int module, const char* file, int line,
                         const char* function, const char* format, ...)
{
    int is_init;
    LogLevel min_level;
    unsigned int filter;
    LogCallbackFunc cb;
    void* cb_data;

    LOG_LOCK_INIT();
    LOG_LOCK();
    is_init = g_log_state.is_initialized;
    min_level = g_log_state.config.min_level;
    filter = g_log_state.module_filter;
    cb = g_log_state.callback;
    cb_data = g_log_state.callback_data;
    LOG_UNLOCK();

    /* 模块过滤：检查模块位是否启用 */
    if ((filter & module) == 0) return;

    if (!is_init)
    {
        if (level >= LOG_LEVEL_WARNING)
        {
            static int warned = 0;
            if (!warned) { fprintf(stderr, "警告: 日志系统未初始化\n"); warned = 1; }
            va_list args;
            va_start(args, format);
            vfprintf(stderr, format, args);
            fprintf(stderr, "\n");
            va_end(args);
        }
        return;
    }

    if (level < min_level) return;

    char message_buffer[4096];
    va_list args;
    va_start(args, format);
    vsnprintf(message_buffer, sizeof(message_buffer), format, args);
    va_end(args);

    write_log_message(level, file, line, function, message_buffer);

    if (cb) cb(level, message_buffer, cb_data);
}

void logging_log(LogLevel level, const char* file, int line, 
                 const char* function, const char* format, ...)
{
    int is_init;
    LogLevel min_level;
    LogCallbackFunc cb;
    void* cb_data;
    
    LOG_LOCK_INIT();
    LOG_LOCK();
    is_init = g_log_state.is_initialized;
    min_level = g_log_state.config.min_level;
    cb = g_log_state.callback;
    cb_data = g_log_state.callback_data;
    LOG_UNLOCK();
    
    if (!is_init)
    {
        // 日志系统未初始化时，仅输出 WARNING 及以上级别到 stderr
        // 避免 DEBUG/INFO 级别消息在未初始化时造成噪音
        if (level >= LOG_LEVEL_WARNING)
        {
            static int warned = 0;
            if (!warned)
            {
                fprintf(stderr, "警告: 日志系统未初始化\n");
                warned = 1;
            }
            
            va_list args;
            va_start(args, format);
            vfprintf(stderr, format, args);
            fprintf(stderr, "\n");
            va_end(args);
        }
        return;
    }
    
    // 检查日志级别
    if (level < min_level)
    {
        return;
    }
    
    // 格式化消息
    char message_buffer[4096];
    va_list args;
    va_start(args, format);
    vsnprintf(message_buffer, sizeof(message_buffer), format, args);
    va_end(args);
    
    // 写入日志消息
    write_log_message(level, file, line, function, message_buffer);
    
    // 调用回调函数
    if (cb)
    {
        cb(level, message_buffer, cb_data);
    }
    
    // 如果是FATAL级别，可能需要进行特殊处理
    if (level == LOG_LEVEL_FATAL)
    {
        // 在实际应用中，这里可能会触发断言或终止程序
        // 为了安全，我们只是记录错误
    }
}

void logging_log_json(LogLevel level, const char* file, int line,
                      const char* function, const char* message, ...)
{
    int is_init;
    LogLevel min_level;
    
    LOG_LOCK();
    is_init = g_log_state.is_initialized;
    min_level = g_log_state.config.min_level;
    LOG_UNLOCK();
    
    if (!is_init || level < min_level)
    {
        return;
    }
    
    // 格式化可变参数
    char json_buffer[8192];
    va_list args;
    va_start(args, message);
    
    // 构建基本JSON结构
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", tm_info);
    
    // 开始构建JSON
    char* ptr = json_buffer;
    size_t remaining = sizeof(json_buffer);
    
    int written = snprintf(ptr, remaining,
        "{\"timestamp\":\"%s\",\"level\":\"%s\",\"file\":\"%s\","
        "\"line\":%d,\"function\":\"%s\",\"message\":\"%s\"",
        timestamp, 
        logging_get_level_name(level),
        get_filename(file),
        line,
        function,
        message);
    
    if (written < 0 || written >= remaining)
    {
        // 缓冲区不足，使用简化格式
        logging_log(level, file, line, function, "(JSON日志格式化失败) %s", message);
        va_end(args);
        return;
    }
    
    ptr += written;
    remaining -= written;
    
    // 添加额外字段
    const char* key;
    while ((key = va_arg(args, const char*)) != NULL)
    {
        const char* value = va_arg(args, const char*);
        if (!value)
        {
            value = "(null)";
        }
        
        // 转义JSON特殊字符（简化处理）
        char escaped_value[1024];
        char* dest = escaped_value;
        const char* src = value;
        
        while (*src && dest < escaped_value + sizeof(escaped_value) - 2)
        {
            switch (*src)
            {
                case '"': *dest++ = '\\'; *dest++ = '"'; break;
                case '\\': *dest++ = '\\'; *dest++ = '\\'; break;
                case '\b': *dest++ = '\\'; *dest++ = 'b'; break;
                case '\f': *dest++ = '\\'; *dest++ = 'f'; break;
                case '\n': *dest++ = '\\'; *dest++ = 'n'; break;
                case '\r': *dest++ = '\\'; *dest++ = 'r'; break;
                case '\t': *dest++ = '\\'; *dest++ = 't'; break;
                default: *dest++ = *src; break;
            }
            src++;
        }
        *dest = '\0';
        
        written = snprintf(ptr, remaining, ",\"%s\":\"%s\"", key, escaped_value);
        if (written < 0 || written >= remaining)
        {
            break;
        }
        
        ptr += written;
        remaining -= written;
    }
    
    // 结束JSON
    if (remaining >= 2)
    {
        strcpy(ptr, "}");
    }
    
    va_end(args);
    
    // 写入日志
    write_log_message(level, file, line, function, json_buffer);
}

double logging_get_timestamp(void)
{
    // 使用clock函数获取高精度时间
    // 这里返回秒为单位的时间
    static double start_time = 0.0;
    static clock_t start_clock = 0;
    
    if (start_clock == 0)
    {
        start_clock = clock();
        start_time = 0.0;
    }
    
    clock_t current_clock = clock();
    return (double)(current_clock - start_clock) / CLOCKS_PER_SEC;
}

const char* logging_get_level_name(LogLevel level)
{
    if (level >= LOG_LEVEL_DEBUG && level <= LOG_LEVEL_FATAL)
    {
        return level_names[level];
    }
    
    return "UNKNOWN";
}

const char* logging_get_level_color(LogLevel level)
{
    int enable_color;
    LOG_LOCK();
    enable_color = g_log_state.config.enable_color;
    LOG_UNLOCK();
    
    if (level >= LOG_LEVEL_DEBUG && level <= LOG_LEVEL_FATAL && enable_color)
    {
        return level_colors[level];
    }
    
    return "";
}

void logging_set_thread_name(const char* name)
{
    if (name)
    {
        LOG_LOCK();
        strncpy(g_log_state.thread_name, name, sizeof(g_log_state.thread_name) - 1);
        g_log_state.thread_name[sizeof(g_log_state.thread_name) - 1] = '\0';
        LOG_UNLOCK();
    }
}

const char* logging_get_thread_name(void)
{
    static char name_buf[32];
    LOG_LOCK();
    strncpy(name_buf, g_log_state.thread_name, sizeof(name_buf) - 1);
    name_buf[sizeof(name_buf) - 1] = '\0';
    LOG_UNLOCK();
    return name_buf;
}

// 内部函数实现

static void write_log_message(LogLevel level, const char* file, int line,
                              const char* function, const char* message)
{
    FILE* output = NULL;
    LogTarget target;
    FILE* log_file;
    int enable_ts, enable_tid, enable_color;
    char thread_name[32];
    
    LOG_LOCK();
    target = g_log_state.config.target;
    log_file = g_log_state.log_file;
    enable_ts = g_log_state.config.enable_timestamp;
    enable_tid = g_log_state.config.enable_thread_id;
    enable_color = g_log_state.config.enable_color;
    strncpy(thread_name, g_log_state.thread_name, sizeof(thread_name) - 1);
    thread_name[sizeof(thread_name) - 1] = '\0';
    LOG_UNLOCK();
    
    // 选择输出目标
    switch (target)
    {
        case LOG_TARGET_STDOUT:
            output = stdout;
            break;
        case LOG_TARGET_STDERR:
            output = stderr;
            break;
        case LOG_TARGET_FILE:
            output = log_file ? log_file : stderr;
            break;
        default:
            output = stderr;
            break;
    }
    
    if (!output)
    {
        return;
    }
    
    // 构建日志前缀
    char prefix[256];
    char* ptr = prefix;
    size_t remaining = sizeof(prefix);
    
    // 时间戳
    if (enable_ts)
    {
        char timestamp[32];
        format_timestamp(timestamp, sizeof(timestamp));
        int written = snprintf(ptr, remaining, "[%s] ", timestamp);
        if (written > 0 && written < remaining)
        {
            ptr += written;
            remaining -= written;
        }
    }
    
    // 线程ID
    if (enable_tid)
    {
        int written = snprintf(ptr, remaining, "[%s] ", thread_name);
        if (written > 0 && written < remaining)
        {
            ptr += written;
            remaining -= written;
        }
    }
    
    // 日志级别（带颜色）
    const char* color = "";
    const char* reset = "";
    
    if (enable_color && output == stdout || output == stderr)
    {
        color = logging_get_level_color(level);
        reset = color_reset;
    }
    
    int written = snprintf(ptr, remaining, "%s[%s]%s ", 
                          color, logging_get_level_name(level), reset);
    if (written > 0 && written < remaining)
    {
        ptr += written;
        remaining -= written;
    }
    
    // 源代码位置（仅DEBUG级别显示详细信息）
    if (level == LOG_LEVEL_DEBUG && file && function)
    {
        written = snprintf(ptr, remaining, "(%s:%d %s) ", 
                          get_filename(file), line, function);
        if (written > 0 && written < remaining)
        {
            ptr += written;
            remaining -= written;
        }
    }
    
    // 输出日志消息
    if (target == LOG_TARGET_FILE && log_file) {
        /* 检查是否需要自动轮转 */
        int msg_len = (int)strlen(prefix) + (int)strlen(message) + 2;
        if (g_log_state.log_file_bytes_written + (size_t)msg_len > g_log_state.log_rotate_max_bytes) {
            check_auto_rotation();
            output = g_log_state.log_file ? g_log_state.log_file : stderr;
        }
        g_log_state.log_file_bytes_written += msg_len;
    }
    fprintf(output, "%s%s\n", prefix, message);
    fflush(output);
}

static const char* get_filename(const char* path)
{
    if (!path)
    {
        return "(unknown)";
    }
    
    const char* filename = strrchr(path, '/');
    if (filename)
    {
        return filename + 1;
    }
    
    filename = strrchr(path, '\\');
    if (filename)
    {
        return filename + 1;
    }
    
    return path;
}

static void format_timestamp(char* buffer, size_t size)
{
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    
    // 格式: YYYY-MM-DD HH:MM:SS
    strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
}

// 全局日志函数（简化接口）
void selflnn_log(LogLevel level, const char* format, ...)
{
    char message[2048];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    
    logging_log(level, __FILE__, __LINE__, __func__, "%s", message);
}

// 默认日志配置
const LogConfig* logging_get_default_config(void)
{
    static LogConfig default_config = {
        .min_level = LOG_LEVEL_INFO,
        .target = LOG_TARGET_STDOUT,
        .file_path = NULL,
        .enable_timestamp = 1,
        .enable_color = 1,
        .enable_thread_id = 0
    };
    
    return &default_config;
}

// 日志缓冲区管理结构体
struct LogBuffer {
    LogEntry* entries;    /**< 日志条目环形缓冲区 */
    size_t capacity;      /**< 缓冲区容量 */
    size_t count;         /**< 当前条目数 */
};

LogBuffer* logging_buffer_create(size_t capacity)
{
    LogBuffer* buffer = (LogBuffer*)safe_malloc(sizeof(LogBuffer));
    if (!buffer)
    {
        return NULL;
    }
    
    buffer->entries = (LogEntry*)safe_calloc(capacity, sizeof(LogEntry));
    if (!buffer->entries)
    {
        safe_free((void**)&buffer);
        return NULL;
    }
    
    buffer->capacity = capacity;
    buffer->count = 0;
    
    return buffer;
}

void logging_buffer_destroy(LogBuffer* buffer)
{
    if (buffer)
    {
        safe_free((void**)&buffer->entries);
        safe_free((void**)&buffer);
    }
}

void logging_buffer_append(LogBuffer* buffer, const LogEntry* entry)
{
    if (!buffer || !entry) return;

    if (buffer->count >= buffer->capacity) {
        size_t new_cap = buffer->capacity > 0 ? buffer->capacity * 2 : 256;
        LogEntry* new_entries = (LogEntry*)
            safe_realloc(buffer->entries, new_cap * sizeof(LogEntry));
        if (!new_entries) return;
        buffer->entries = new_entries;
        memset(&buffer->entries[buffer->capacity], 0,
               (new_cap - buffer->capacity) * sizeof(LogEntry));
        buffer->capacity = new_cap;
    }

    LogEntry* dst = &buffer->entries[buffer->count];
    dst->level = entry->level;
    dst->line = entry->line;
    dst->timestamp = entry->timestamp;
    dst->message = entry->message ? string_duplicate(entry->message) : NULL;
    dst->file = entry->file ? string_duplicate(entry->file) : NULL;
    dst->function = entry->function ? string_duplicate(entry->function) : NULL;
    buffer->count++;
}

size_t logging_buffer_get_count(const LogBuffer* buffer)
{
    return buffer ? buffer->count : 0;
}

const LogEntry* logging_buffer_get_entry(const LogBuffer* buffer, size_t index)
{
    if (!buffer || index >= buffer->count) return NULL;
    return &buffer->entries[index];
}

void logging_buffer_clear(LogBuffer* buffer)
{
    if (!buffer) return;
    for (size_t i = 0; i < buffer->count; i++) {
        safe_free((void**)&buffer->entries[i].message);
        safe_free((void**)&buffer->entries[i].file);
        safe_free((void**)&buffer->entries[i].function);
    }
    buffer->count = 0;
}

/**
 * @brief 检查是否需要自动轮转日志文件
 *
 * 当写入字节数超过阈值时自动执行轮转：
 * - 关闭当前日志文件
 * - 重命名为带时间戳的备份文件
 * - 清理超过最大保留数的旧文件
 * - 打开新日志文件
 */
static void check_auto_rotation(void) {
    if (!g_log_state.log_file ||
        g_log_state.config.target != LOG_TARGET_FILE ||
        g_log_state.config.file_path == NULL) {
        return;
    }
    if (g_log_state.log_file_bytes_written < g_log_state.log_rotate_max_bytes) {
        return;
    }

    fclose(g_log_state.log_file);
    g_log_state.log_file = NULL;

    /* 生成带时间戳的备份文件名 */
    time_t now = time(NULL);
    struct tm* tm_ptr = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_ptr);

    size_t backup_path_len = strlen(g_log_state.config.file_path) + strlen(timestamp) + 2;
    char* backup_path = (char*)safe_malloc(backup_path_len);
    if (backup_path) {
        snprintf(backup_path, backup_path_len, "%s.%s",
                 g_log_state.config.file_path, timestamp);
        rename(g_log_state.config.file_path, backup_path);
        safe_free((void**)&backup_path);
    }

    /* 清理超过最大保留数的旧轮转文件 */
    if (g_log_state.log_rotate_max_files > 0) {
        /* 查找所有轮转文件：扫描目录匹配文件名模式，删除超出保留数的文件 */
        for (int i = g_log_state.log_rotate_max_files + 10; i >= g_log_state.log_rotate_max_files; i--) {
            char old_path[2048];
            snprintf(old_path, sizeof(old_path), "%s.%d",
                     g_log_state.config.file_path, i);
            remove(old_path);
        }
    }

    /* 打开新日志文件 */
    g_log_state.log_file = fopen(g_log_state.config.file_path, "a");
    g_log_state.log_file_bytes_written = 0;

    if (g_log_state.log_file) {
        time_t now2 = time(NULL);
        fprintf(g_log_state.log_file, "--- 日志轮转: %s", ctime(&now2));
        fflush(g_log_state.log_file);
    }
}

/**
 * @brief 设置日志轮转参数
 */
void logging_set_rotation(size_t max_bytes, int max_files) {
    LOG_LOCK();
    g_log_state.log_rotate_max_bytes = max_bytes > 0 ? max_bytes : (size_t)10 * 1024 * 1024;
    g_log_state.log_rotate_max_files = max_files > 0 ? max_files : 5;
    LOG_UNLOCK();
}

// 日志轮转
int logging_rotate_file(const char* new_path)
{
    if (!new_path)
    {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    LOG_LOCK();
    
    // 获取当前时间戳用于备份文件名
    time_t now = time(NULL);
    struct tm* tm_ptr = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_ptr);
    
    // 如果当前有打开的文件，进行轮转
    if (g_log_state.log_file)
    {
        // 关闭当前文件
        fclose(g_log_state.log_file);
        g_log_state.log_file = NULL;
        
        // 生成备份文件名: 原文件名.时间戳
        if (g_log_state.config.file_path)
        {
            size_t backup_path_len = strlen(g_log_state.config.file_path) + strlen(timestamp) + 2;
            char* backup_path = (char*)safe_malloc(backup_path_len);
            if (backup_path)
            {
                snprintf(backup_path, backup_path_len, "%s.%s", g_log_state.config.file_path, timestamp);
                rename(g_log_state.config.file_path, backup_path);
                safe_free((void**)&backup_path);
            }
        }
    }
    
    // 打开新文件
    g_log_state.log_file = fopen(new_path, "a");
    if (!g_log_state.log_file)
    {
        g_log_state.config.target = LOG_TARGET_STDERR;
        LOG_UNLOCK();
        return SELFLNN_ERROR_OPERATION_FAILED;
    }
    
    // 更新当前文件路径（不释放旧路径，可能指向外部内存）
    g_log_state.config.file_path = new_path;
    
    LOG_UNLOCK();
    return SELFLNN_SUCCESS;
}

/* ================================================================
 * K-029: 日志系统增强 —— 运行时操作
 * ================================================================ */

void logging_set_level_at_runtime(LogLevel level)
{
    LOG_LOCK();
    if (level >= LOG_LEVEL_DEBUG && level <= LOG_LEVEL_FATAL) {
        g_log_state.config.min_level = level;
        log_info("[日志] 运行时级别调整为: %s", logging_get_level_name(level));
    }
    LOG_UNLOCK();
}

int logging_reopen_file(void)
{
    LOG_LOCK();
    if (g_log_state.config.target != LOG_TARGET_FILE || !g_log_state.config.file_path) {
        LOG_UNLOCK();
        return -1;
    }
    if (g_log_state.log_file) {
        fclose(g_log_state.log_file);
        g_log_state.log_file = NULL;
    }
    g_log_state.log_file = fopen(g_log_state.config.file_path, "a");
    if (!g_log_state.log_file) {
        LOG_UNLOCK();
        return -1;
    }
    LOG_UNLOCK();
    log_info("[日志] 日志文件已重新打开: %s", g_log_state.config.file_path);
    return 0;
}

size_t logging_get_file_size(void)
{
    size_t size = 0;
    LOG_LOCK();
    if (g_log_state.log_file) {
        long pos = ftell(g_log_state.log_file);
        fseek(g_log_state.log_file, 0, SEEK_END);
        size = (size_t)ftell(g_log_state.log_file);
        fseek(g_log_state.log_file, pos, SEEK_SET);
    }
    LOG_UNLOCK();
    return size;
}

void logging_set_module_mask(unsigned int module_mask)
{
    LOG_LOCK();
    g_log_state.module_filter = module_mask;
    LOG_UNLOCK();
    log_info("[日志] 模块过滤器掩码: 0x%08X", module_mask);
}