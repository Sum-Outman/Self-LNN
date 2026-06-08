/**
 * @file errors.c
 * @brief SELF-LNN统一错误处理系统实现
 * 
 * 实现统一的错误代码处理、错误信息管理和错误传播机制。
 */

#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

/**
 * @brief 线程局部错误上下文
 * 
 * 每个线程有自己的错误上下文，避免多线程竞争。
 */
#ifdef _WIN32
static __declspec(thread) SelfLNNErrorContext s_last_error = {0};
#else
static __thread SelfLNNErrorContext s_last_error = {0};
#endif

/**
 * @brief 程序退出时自动清理最后错误消息
 */
static void errors_auto_cleanup(void) {
    selflnn_clear_last_error();
}

/**
 * @brief 初始化错误系统的内部构造辅助
 * 
 * 使用全局静态变量的初始化来注册 atexit 清理。
 * 只要有任何错误被设置，退出时就能自动释放消息内存。
 */
static int errors_auto_cleanup_init(void) {
    static int registered = 0;
    if (!registered) {
        atexit(errors_auto_cleanup);
        registered = 1;
    }
    return 0;
}

/**
 * @brief 错误代码到字符串的映射
 */
static const struct {
    SelfLNNErrorCode code;
    const char* description;
} s_error_descriptions[] = {
    // 成功代码
    {SELFLNN_SUCCESS, "操作成功"},
    
    // 通用错误
    {SELFLNN_ERROR_GENERIC, "通用错误"},
    {SELFLNN_ERROR_INVALID_ARGUMENT, "无效参数"},
    {SELFLNN_ERROR_NULL_POINTER, "空指针"},
    {SELFLNN_ERROR_OUT_OF_MEMORY, "内存不足"},
    {SELFLNN_ERROR_NOT_INITIALIZED, "未初始化"},
    {SELFLNN_ERROR_ALREADY_INITIALIZED, "已初始化"},
    {SELFLNN_ERROR_MODULE_LINK_FAILURE, "模块链接失败（必要模块缺失）"},
    {SELFLNN_ERROR_IO_ERROR, "I/O错误"},
    {SELFLNN_ERROR_FORMAT_ERROR, "格式错误"},
    {SELFLNN_ERROR_TIMEOUT, "超时"},
    {SELFLNN_ERROR_INVALID_STATE, "无效状态错误"},
    {SELFLNN_ERROR_COMPUTATION, "计算错误"},
    {SELFLNN_ERROR_ALGORITHM_FAILURE, "算法失败"},
    {SELFLNN_ERROR_INITIALIZATION_FAILED, "初始化失败"},
    {SELFLNN_ERROR_NOT_FOUND, "未找到"},
    {SELFLNN_ERROR_OUT_OF_BOUNDS, "越界"},
    {SELFLNN_ERROR_ALREADY_EXISTS, "已存在"},
    
    // 网络相关错误
    {SELFLNN_ERROR_NETWORK_CONFIG, "网络配置错误"},
    {SELFLNN_ERROR_NETWORK_NOT_CREATED, "网络未创建"},
    {SELFLNN_ERROR_NETWORK_ALREADY_CREATED, "网络已创建"},
    {SELFLNN_ERROR_LAYER_MISMATCH, "网络层不匹配"},
    {SELFLNN_ERROR_WEIGHT_SIZE, "权重大小错误"},
    {SELFLNN_ERROR_ACTIVATION_SIZE, "激活大小错误"},
    {SELFLNN_ERROR_GRADIENT_SIZE, "梯度大小错误"},
    
    // CfC单元错误
    {SELFLNN_ERROR_CFC_CELL_CONFIG, "CfC单元配置错误"},
    {SELFLNN_ERROR_CFC_CELL_NOT_CREATED, "CfC单元未创建"},
    {SELFLNN_ERROR_CFC_TIME_CONSTANT, "时间常数错误"},
    {SELFLNN_ERROR_CFC_NOISE_STD, "噪声标准差错误"},
    
    // 记忆系统错误
    {SELFLNN_ERROR_MEMORY_ALLOCATION, "记忆分配错误"},
    {SELFLNN_ERROR_MEMORY_FULL, "记忆已满"},
    {SELFLNN_ERROR_MEMORY_EMPTY, "记忆为空"},
    {SELFLNN_ERROR_MEMORY_CORRUPTION, "记忆损坏"},
    
    // 多模态处理错误
    {SELFLNN_ERROR_MODALITY_NOT_SUPPORTED, "模态不支持"},
    {SELFLNN_ERROR_FEATURE_EXTRACTION, "特征提取错误"},
    {SELFLNN_ERROR_FEATURE_FUSION, "特征融合错误"},
    
    // 训练相关错误
    {SELFLNN_ERROR_TRAINING_NOT_ENABLED, "训练未启用"},
    {SELFLNN_ERROR_TRAINING_CONVERGENCE, "训练不收敛"},
    {SELFLNN_ERROR_LEARNING_RATE, "学习率错误"},
    {SELFLNN_ERROR_GRADIENT_EXPLOSION, "梯度爆炸"},
    {SELFLNN_ERROR_GRADIENT_VANISHING, "梯度消失"},
    {SELFLNN_ERROR_PROJECTION_LOCKED, "投影矩阵已锁定（不参与训练,前向监控损失仍返回）"},
    
    // 并发处理错误
    {SELFLNN_ERROR_THREAD_CREATION, "线程创建错误"},
    {SELFLNN_ERROR_THREAD_JOIN, "线程加入错误"},
    {SELFLNN_ERROR_MUTEX_INIT, "互斥锁初始化错误"},
    {SELFLNN_ERROR_MUTEX_LOCK, "互斥锁锁定错误"},
    {SELFLNN_ERROR_MUTEX_UNLOCK, "互斥锁解锁错误"},
    
    // GPU相关错误
    {SELFLNN_ERROR_GPU_NOT_AVAILABLE, "GPU不可用"},
    {SELFLNN_ERROR_GPU_MEMORY, "GPU内存错误"},
    {SELFLNN_ERROR_GPU_KERNEL, "GPU内核错误"},
    
    // 自我认知错误
    // 设备相关错误
    {SELFLNN_ERROR_DEVICE_OFFLINE, "设备离线"},
    {SELFLNN_ERROR_DEVICE_NOT_FOUND, "设备未找到"},
    {SELFLNN_ERROR_DEVICE_BUSY, "设备忙"},
    {SELFLNN_ERROR_DEVICE_PERMISSION, "设备权限不足"},
    {SELFLNN_ERROR_DEVICE_PROTOCOL, "设备协议错误"},
    {SELFLNN_ERROR_DEVICE_TIMEOUT, "设备操作超时"},

    // 安全相关错误
    {SELFLNN_ERROR_SAFETY_VIOLATION, "安全违规"},
    {SELFLNN_ERROR_SAFETY_PHYSICAL, "物理安全违规"},
    {SELFLNN_ERROR_SAFETY_PRIVACY, "隐私安全违规"},
    {SELFLNN_ERROR_SAFETY_BOUNDARY, "安全边界越界"},
    {SELFLNN_ERROR_SAFETY_EMERGENCY_STOP, "紧急停止已触发"},
    {SELFLNN_ERROR_SAFETY_CONTENT_FILTER, "内容过滤拦截"},

    // 多模态相关错误
    {SELFLNN_ERROR_MULTIMODAL_NO_DATA, "多模态无数据输入"},
    {SELFLNN_ERROR_MULTIMODAL_DECODE, "多模态解码失败"},
    {SELFLNN_ERROR_MULTIMODAL_UNSUPPORTED, "不支持的模态类型"},
    {SELFLNN_ERROR_MULTIMODAL_DIM_MISMATCH, "模态维度不匹配"},

    {SELFLNN_ERROR_SELF_COGNITION, "自我认知错误"},
    {SELFLNN_ERROR_PLANNING_FAILED, "规划失败"},
    {SELFLNN_ERROR_REASONING_FAILED, "推理失败"},
    
    // 警告代码
    {SELFLNN_WARNING_LOW_CONFIDENCE, "低置信度警告"},
    {SELFLNN_WARNING_HIGH_UNCERTAINTY, "高不确定性警告"},
    {SELFLNN_WARNING_SLOW_CONVERGENCE, "收敛缓慢警告"},
    {SELFLNN_WARNING_MEMORY_LIMIT, "内存限制警告"},
    
    // 信息代码
    {SELFLNN_INFO_CONVERGED, "已收敛信息"},
    {SELFLNN_INFO_OPTIMAL, "最优解信息"},
    {SELFLNN_INFO_STABLE, "稳定状态信息"},
    
    // 结束标记
    {0, NULL}
};

const char* selflnn_error_to_string(SelfLNNErrorCode error_code) {
    for (size_t i = 0; s_error_descriptions[i].description != NULL; i++) {
        if (s_error_descriptions[i].code == error_code) {
            return s_error_descriptions[i].description;
        }
    }
    
    // 未知错误代码
    static char unknown_buffer[64];
    snprintf(unknown_buffer, sizeof(unknown_buffer), "未知错误代码: %d", error_code);
    return unknown_buffer;
}

int selflnn_error_context_to_string(const SelfLNNErrorContext* context,
                                   char* buffer, size_t buffer_size) {
    if (!context || !buffer || buffer_size == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    const char* error_desc = selflnn_error_to_string(context->code);
    
    int written = snprintf(buffer, buffer_size,
                          "错误代码: %d\n"
                          "错误描述: %s\n"
                          "位置: %s (%s:%d)\n"
                          "消息: %s\n",
                          context->code,
                          error_desc,
                          context->function ? context->function : "(未知函数)",
                          context->file ? context->file : "(未知文件)",
                          context->line,
                          context->message ? context->message : "(无消息)");
    
    if (written < 0 || (size_t)written >= buffer_size) {
        return SELFLNN_ERROR_FORMAT_ERROR;
    }
    
    return SELFLNN_SUCCESS;
}

void selflnn_set_last_error(SelfLNNErrorCode error_code,
                           const char* function, const char* file, int line,
                           const char* format, ...) {
    // 确保退出时清理
    errors_auto_cleanup_init();
    
    // 清除之前的错误消息
    if (s_last_error.message) {
        safe_free((void**)&s_last_error.message);
    }
    
    // 设置错误上下文基本信息
    s_last_error.code = error_code;
    s_last_error.function = function;
    s_last_error.file = file;
    s_last_error.line = line;
    s_last_error.user_data = NULL;
    s_last_error.user_data_size = 0;
    
    // 格式化错误消息
    if (format && format[0] != '\0') {
        char message_buffer[1024];
        va_list args;
        va_start(args, format);
        int written = vsnprintf(message_buffer, sizeof(message_buffer), format, args);
        va_end(args);
        
        if (written > 0 && written < (int)sizeof(message_buffer)) {
            s_last_error.message = string_duplicate(message_buffer);
        } else {
            s_last_error.message = string_duplicate("(无法格式化错误消息)");
        }
    } else {
        s_last_error.message = NULL;
    }
}

const SelfLNNErrorContext* selflnn_get_last_error_context(void) {
    return &s_last_error;
}

void selflnn_clear_last_error(void) {
    if (s_last_error.message) {
        safe_free((void**)&s_last_error.message);
    }
    
    memset(&s_last_error, 0, sizeof(SelfLNNErrorContext));
}

int selflnn_is_success(SelfLNNErrorCode error_code) {
    return error_code == SELFLNN_SUCCESS;
}

int selflnn_is_error(SelfLNNErrorCode error_code) {
    return error_code < 0;
}

int selflnn_is_warning(SelfLNNErrorCode error_code) {
    return error_code >= 1000 && error_code < 2000;
}

/* H-002 / I-001: 精简委托包装函数，仅为向后兼容保留。
 * 内部直接委托到 selflnn_is_warning，无额外逻辑。
 * I-001修复后消除了重复实现，统一为一个判断入口。 */
int selflnn_is_warning_code(SelfLNNErrorCode error_code) {
    return selflnn_is_warning(error_code);
}

// 信息代码检查
int selflnn_is_info(SelfLNNErrorCode error_code) {
    return error_code >= 2000;
}

/* ============================================================================
 * 错误恢复机制实现
 * =========================================================================== */

/**
 * @brief 线程局部错误恢复上下文栈（支持嵌套TRY）
 */
#ifdef _WIN32
static __declspec(thread) SelfLNNRecoveryContext* s_recovery_stack[16];
static __declspec(thread) int s_recovery_depth = -1;
#else
static __thread SelfLNNRecoveryContext* s_recovery_stack[16];
static __thread int s_recovery_depth = -1;
#endif

int selflnn_recovery_try(SelfLNNRecoveryContext* context) {
    if (!context) return SELFLNN_ERROR_INVALID_ARGUMENT;

    context->has_error = 0;
    context->caught_error = SELFLNN_SUCCESS;

    if (s_recovery_depth < 15) {
        s_recovery_stack[++s_recovery_depth] = context;
    }

    int result = setjmp(context->env);

    if (result == 0) {
        return 0;
    }

    context->has_error = 1;
    context->caught_error = (SelfLNNErrorCode)result;
    return result;
}

void selflnn_recovery_throw(SelfLNNRecoveryContext* context, SelfLNNErrorCode error_code) {
    if (!context) return;

    context->has_error = 1;
    context->caught_error = error_code;

    longjmp(context->env, (int)error_code);
}

int selflnn_has_recovery_error(void) {
    if (s_recovery_depth >= 0 && s_recovery_stack[s_recovery_depth]) {
        return s_recovery_stack[s_recovery_depth]->has_error;
    }
    return 0;
}

SelfLNNErrorCode selflnn_get_recovery_error(void) {
    if (s_recovery_depth >= 0 && s_recovery_stack[s_recovery_depth]) {
        return s_recovery_stack[s_recovery_depth]->caught_error;
    }
    return SELFLNN_SUCCESS;
}

void selflnn_clear_recovery(void) {
    if (s_recovery_depth >= 0) {
        s_recovery_stack[s_recovery_depth] = NULL;
        s_recovery_depth--;
    }
}

/* ============================================================================
 * 错误恢复策略引擎：重试/隔离（禁止任何降级处理）
 * ============================================================================ */

#define MAX_RETRY_COUNT 5
#define BASE_RETRY_DELAY_MS 100
#define MAX_RETRY_DELAY_MS 5000

typedef struct {
    SelfLNNErrorCode error_type;
    int retry_count;
    int retry_delay_ms;
    int isolated;
    time_t first_failure_time;
    time_t last_failure_time;
} RecoveryStrategy;

static RecoveryStrategy g_recovery_strategies[16];
static int g_recovery_strategy_count = 0;

int selflnn_recovery_retry(SelfLNNErrorCode error_type, int (*retry_func)(void*), void* user_data) {
    int delay_ms = BASE_RETRY_DELAY_MS;
    int last_result = -1;

    for (int attempt = 0; attempt < MAX_RETRY_COUNT; attempt++) {
        if (attempt > 0) {
#ifdef _WIN32
            Sleep((DWORD)delay_ms);
#else
            usleep((useconds_t)(delay_ms * 1000));
#endif
            delay_ms *= 2;
            if (delay_ms > MAX_RETRY_DELAY_MS) delay_ms = MAX_RETRY_DELAY_MS;
        }

        last_result = retry_func(user_data);
        if (last_result == 0) {
            selflnn_clear_last_error();
            return 0;
        }
    }

    return last_result;
}

int selflnn_recovery_isolate(SelfLNNErrorCode error_type) {
    int idx = -1;
    for (int i = 0; i < g_recovery_strategy_count; i++) {
        if (g_recovery_strategies[i].error_type == error_type) {
            idx = i;
            break;
        }
    }
    if (idx < 0 && g_recovery_strategy_count < 16) {
        idx = g_recovery_strategy_count++;
    }
    if (idx >= 0) {
        g_recovery_strategies[idx].error_type = error_type;
        g_recovery_strategies[idx].isolated = 1;
        g_recovery_strategies[idx].last_failure_time = time(NULL);
        if (g_recovery_strategies[idx].first_failure_time == 0) {
            g_recovery_strategies[idx].first_failure_time = time(NULL);
        }
    }
    return 0;
}

int selflnn_is_component_isolated(SelfLNNErrorCode error_type) {
    for (int i = 0; i < g_recovery_strategy_count; i++) {
        if (g_recovery_strategies[i].error_type == error_type &&
            g_recovery_strategies[i].isolated) {
            return 1;
        }
    }
    return 0;
}

int selflnn_recovery_reset(SelfLNNErrorCode error_type) {
    for (int i = 0; i < g_recovery_strategy_count; i++) {
        if (g_recovery_strategies[i].error_type == error_type) {
            memset(&g_recovery_strategies[i], 0, sizeof(RecoveryStrategy));
            return 0;
        }
    }
    return -1;
}