/**
 * @file errors.h
 * @brief SELF-LNN统一错误处理系统
 * 
 * K-045: 提供统一的错误代码定义、错误信息处理和错误传播机制。
 * 符合全液态神经网络AGI系统100%纯C实现要求。
 * 宏使用C11标准（__VA_ARGS__需C99+/C11支持），编译器最低要求C11。
 */

#ifndef SELFLNN_ERRORS_H
#define SELFLNN_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <setjmp.h>
#include "platform_types.h"

/**
 * @brief 通用错误代码
 * 
 * 使用负数表示错误，0表示成功，正数表示警告或信息。
 */
typedef enum {
    // 成功代码
    SELFLNN_SUCCESS = 0,                    /**< 操作成功 */
    
    // 通用错误代码 (1-99)
    SELFLNN_ERROR_GENERIC = -1,             /**< 通用错误 */
    SELFLNN_ERROR_INVALID_ARGUMENT = -2,    /**< 无效参数 */
    SELFLNN_ERROR_INVALID_DIMENSION = -2,   /**< 无效维度（无效参数别名） */
    SELFLNN_ERROR_NULL_POINTER = -3,        /**< 空指针 */
    SELFLNN_ERROR_OUT_OF_MEMORY = -4,       /**< 内存不足 */
    SELFLNN_ERROR_NOT_INITIALIZED = -5,     /**< 未初始化 */
    SELFLNN_ERROR_ALREADY_INITIALIZED = -6, /**< 已初始化 */
    /* -7 位已移除：原 SELFLNN_ERROR_NOT_IMPLEMENTED，违反"禁止任何降级处理"原则 */
    SELFLNN_ERROR_MODULE_LINK_FAILURE = -7,  /**< 模块链接失败（编译时模块缺失） */
    SELFLNN_ERROR_IO_ERROR = -8,            /**< I/O错误 */
    SELFLNN_ERROR_FORMAT_ERROR = -9,        /**< 格式错误 */
    SELFLNN_ERROR_TIMEOUT = -10,            /**< 超时 */
    SELFLNN_ERROR_INVALID_STATE = -11,      /**< 无效状态错误 */
    SELFLNN_ERROR_COMPUTATION = -12,        /**< 计算错误 */
    SELFLNN_ERROR_ALGORITHM_FAILURE = -13,  /**< 算法失败 */
    SELFLNN_ERROR_INITIALIZATION_FAILED = -14, /**< 初始化失败 */
    SELFLNN_ERROR_NOT_FOUND = -15,          /**< 未找到 */
    SELFLNN_ERROR_OUT_OF_BOUNDS = -16,      /**< 越界 */
    SELFLNN_ERROR_ALREADY_EXISTS = -17,     /**< 已存在 */
    
    // 网络相关错误 (100-199)
    SELFLNN_ERROR_NETWORK_CONFIG = -100,    /**< 网络配置错误 */
    SELFLNN_ERROR_NETWORK_NOT_CREATED = -101, /**< 网络未创建 */
    SELFLNN_ERROR_NETWORK_ALREADY_CREATED = -102, /**< 网络已创建 */
    SELFLNN_ERROR_LAYER_MISMATCH = -103,    /**< 网络层不匹配 */
    SELFLNN_ERROR_WEIGHT_SIZE = -104,       /**< 权重大小错误 */
    SELFLNN_ERROR_ACTIVATION_SIZE = -105,   /**< 激活大小错误 */
    SELFLNN_ERROR_GRADIENT_SIZE = -106,     /**< 梯度大小错误 */
    SELFLNN_ERROR_NETWORK_FORWARD = -107,   /**< 网络前向传播错误 */
    SELFLNN_ERROR_NETWORK_BACKWARD = -108,  /**< 网络反向传播错误 */
    SELFLNN_ERROR_NETWORK_CREATION = -110,  /**< 网络创建失败 */
    
    // CfC单元错误 (200-299)
    SELFLNN_ERROR_CFC_CELL_CONFIG = -200,   /**< CfC单元配置错误 */
    SELFLNN_ERROR_CFC_CELL_NOT_CREATED = -201, /**< CfC单元未创建 */
    SELFLNN_ERROR_CFC_TIME_CONSTANT = -202, /**< 时间常数错误 */
    SELFLNN_ERROR_CFC_NOISE_STD = -203,     /**< 噪声标准差错误 */
    
    // 记忆系统错误 (300-399)
    SELFLNN_ERROR_MEMORY_ALLOCATION = -300, /**< 记忆分配错误 */
    SELFLNN_ERROR_MEMORY_FULL = -301,       /**< 记忆已满 */
    SELFLNN_ERROR_MEMORY_EMPTY = -302,      /**< 记忆为空 */
    SELFLNN_ERROR_MEMORY_CORRUPTION = -303, /**< 记忆损坏 */
    SELFLNN_ERROR_MEMORY_NOT_FOUND = -304,  /**< 记忆未找到 */
    SELFLNN_ERROR_MEMORY_CONFIG = -305,     /**< 记忆配置错误 */
    
    // 多模态处理错误 (400-499)
    SELFLNN_ERROR_MODALITY_NOT_SUPPORTED = -400, /**< 模态不支持 */
    SELFLNN_ERROR_FEATURE_EXTRACTION = -401, /**< 特征提取错误 */
    SELFLNN_ERROR_FEATURE_FUSION = -402,     /**< 特征融合错误 */
    
    // 训练相关错误 (500-599)
    SELFLNN_ERROR_TRAINING_NOT_ENABLED = -500, /**< 训练未启用 */
    SELFLNN_ERROR_TRAINING_CONVERGENCE = -501, /**< 训练不收敛 */
    SELFLNN_ERROR_LEARNING_RATE = -502,     /**< 学习率错误 */
    SELFLNN_ERROR_GRADIENT_EXPLOSION = -503, /**< 梯度爆炸 */
    SELFLNN_ERROR_GRADIENT_VANISHING = -504, /**< 梯度消失 */
    SELFLNN_ERROR_NO_DATA = -505,             /**< 无可用数据 */
    SELFLNN_ERROR_PROJECTION_LOCKED = -506,   /**< ZSFX-DEEP-R8: 投影矩阵已锁定不参与训练（前向损失仍可用） */
    
    // 并发处理错误 (600-699)
    SELFLNN_ERROR_THREAD_CREATION = -600,   /**< 线程创建错误 */
    SELFLNN_ERROR_THREAD_JOIN = -601,       /**< 线程加入错误 */
    SELFLNN_ERROR_MUTEX_INIT = -602,        /**< 互斥锁初始化错误 */
    SELFLNN_ERROR_MUTEX_LOCK = -603,        /**< 互斥锁锁定错误 */
    SELFLNN_ERROR_MUTEX_UNLOCK = -604,      /**< 互斥锁解锁错误 */
    
    // GPU相关错误 (700-799)
    SELFLNN_ERROR_GPU_NOT_AVAILABLE = -700,      /**< GPU不可用 */
    SELFLNN_ERROR_GPU_MEMORY = -701,             /**< GPU内存错误 */
    SELFLNN_ERROR_GPU_KERNEL = -702,             /**< GPU内核错误 */
    SELFLNN_ERROR_GPU_COMPUTE_FAILED = -703,     /**< GPU计算失败 */
    SELFLNN_ERROR_GPU_NOT_INITIALIZED = -704,    /**< GPU未初始化 */
    SELFLNN_ERROR_GPU_MEMORY_NOT_ALLOCATED = -705, /**< GPU内存未分配 */
    SELFLNN_ERROR_GPU_MEMORY_COPY_FAILED = -706, /**< GPU内存拷贝失败 */
    SELFLNN_ERROR_GPU_KERNEL_FAILED = -707,      /**< GPU内核执行失败 */
    
    // AI模型相关错误 (750-779)
    SELFLNN_ERROR_MODEL_NOT_FOUND = -750,   /**< 模型未找到 */
    SELFLNN_ERROR_TRAINING_FAILED = -751,   /**< 训练失败 */
    SELFLNN_ERROR_INFERENCE_FAILED = -752,  /**< 推理失败 */
    
    // 设备相关错误 (780-799)
    SELFLNN_ERROR_DEVICE_OFFLINE = -780,    /**< 设备离线 */
    SELFLNN_ERROR_DEVICE_NOT_FOUND = -781,  /**< 设备未找到 */
    SELFLNN_ERROR_DEVICE_BUSY = -782,       /**< 设备忙 */
    SELFLNN_ERROR_DEVICE_PERMISSION = -783, /**< 设备权限不足 */
    SELFLNN_ERROR_DEVICE_PROTOCOL = -784,   /**< 设备协议错误 */
    SELFLNN_ERROR_DEVICE_TIMEOUT = -785,    /**< 设备操作超时 */
    
    // 安全相关错误 (800-829)
    SELFLNN_ERROR_SAFETY_VIOLATION = -800,     /**< 安全违规 */
    SELFLNN_ERROR_SAFETY_PHYSICAL = -801,      /**< 物理安全违规 */
    SELFLNN_ERROR_SAFETY_PRIVACY = -802,       /**< 隐私安全违规 */
    SELFLNN_ERROR_SAFETY_BOUNDARY = -803,      /**< 安全边界越界 */
    SELFLNN_ERROR_SAFETY_EMERGENCY_STOP = -804,/**< 紧急停止已触发 */
    SELFLNN_ERROR_SAFETY_CONTENT_FILTER = -805,/**< 内容过滤拦截 */
    
    // 多模态相关错误 (830-859)
    SELFLNN_ERROR_MULTIMODAL_NO_DATA = -830,   /**< 多模态无数据输入 */
    SELFLNN_ERROR_MULTIMODAL_DECODE = -831,    /**< 多模态解码失败 */
    SELFLNN_ERROR_MULTIMODAL_UNSUPPORTED = -832,/**< 不支持的模态类型 */
    SELFLNN_ERROR_MULTIMODAL_DIM_MISMATCH = -833,/**< 模态维度不匹配 */
    
    // 自我认知错误 (860-899)
    SELFLNN_ERROR_SELF_COGNITION = -860,    /**< 自我认知错误 */
    SELFLNN_ERROR_PLANNING_FAILED = -861,   /**< 规划失败 */
    SELFLNN_ERROR_REASONING_FAILED = -862,  /**< 推理失败 */
    
    // 硬件接口错误 (900-949)
    SELFLNN_ERROR_HARDWARE_FAILURE = -900,  /**< 硬件失败错误 */
    SELFLNN_ERROR_PARTIAL_FAILURE = -901,   /**< 部分失败错误（部分组件失败但操作仍可继续） */
    
    // 调度器错误 (950-999)
    SELFLNN_ERROR_SCHEDULER_MODULE_NOT_FOUND = -950, /**< 调度器模块未找到 */
    SELFLNN_ERROR_SCHEDULER_TASK_FAILED = -951,      /**< 调度器任务失败 */
    SELFLNN_ERROR_SCHEDULER_DEADLOCK = -952,         /**< 调度器死锁 */
    SELFLNN_ERROR_SCHEDULER_QUEUE_FULL = -953,       /**< 调度器队列已满 */
    
    // 警告代码 (1000+)
    SELFLNN_WARNING_LOW_CONFIDENCE = 1000,  /**< 低置信度警告 */
    SELFLNN_WARNING_HIGH_UNCERTAINTY = 1001, /**< 高不确定性警告 */
    SELFLNN_WARNING_SLOW_CONVERGENCE = 1002, /**< 收敛缓慢警告 */
    SELFLNN_WARNING_MEMORY_LIMIT = 1003,    /**< 内存限制警告 */
    
    // 信息代码 (2000+)
    SELFLNN_INFO_CONVERGED = 2000,          /**< 已收敛信息 */
    SELFLNN_INFO_OPTIMAL = 2001,            /**< 最优解信息 */
    SELFLNN_INFO_STABLE = 2002              /**< 稳定状态信息 */
} SelfLNNErrorCode;

/**
 * @brief 错误上下文结构体
 * 
 * 包含错误的完整上下文信息，便于调试和错误恢复。
 */
typedef struct {
    SelfLNNErrorCode code;          /**< 错误代码 */
    const char* function;           /**< 函数名 */
    const char* file;               /**< 文件名 */
    int line;                       /**< 行号 */
    const char* message;            /**< 错误消息 */
    void* user_data;                /**< 用户数据（可选） */
    size_t user_data_size;          /**< 用户数据大小 */
} SelfLNNErrorContext;

/**
 * @brief 获取错误代码的字符串描述
 * 
 * @param error_code 错误代码
 * @return const char* 错误描述字符串
 */
SELFLNN_API const char* selflnn_error_to_string(SelfLNNErrorCode error_code);

/**
 * @brief 获取错误上下文的完整描述
 * 
 * @param context 错误上下文
 * @param buffer 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回0，失败返回错误代码
 */
SELFLNN_API int selflnn_error_context_to_string(const SelfLNNErrorContext* context,
                                   char* buffer, size_t buffer_size);

/**
 * @brief 设置当前线程的最后一个错误
 * 
 * @param error_code 错误代码
 * @param function 函数名
 * @param file 文件名
 * @param line 行号
 * @param format 错误消息格式字符串
 * @param ... 格式参数
 */
SELFLNN_API void selflnn_set_last_error(SelfLNNErrorCode error_code,
                           const char* function, const char* file, int line,
                           const char* format, ...);

/**
 * @brief 获取当前线程的最后一个错误
 * 
 * @return const SelfLNNErrorContext* 错误上下文指针
 */
SELFLNN_API const SelfLNNErrorContext* selflnn_get_last_error_context(void);

/**
 * @brief 清除当前线程的最后一个错误
 */
SELFLNN_API void selflnn_clear_last_error(void);

/**
 * @brief 检查错误代码是否表示成功
 * 
 * @param error_code 错误代码
 * @return int 成功返回1，失败返回0
 */
SELFLNN_API int selflnn_is_success(SelfLNNErrorCode error_code);

/**
 * @brief 检查错误代码是否表示错误（负数）
 * 
 * @param error_code 错误代码
 * @return int 是错误返回1，否则返回0
 */
SELFLNN_API int selflnn_is_error(SelfLNNErrorCode error_code);

/**
 * @brief 检查错误代码是否表示警告（正数）
 * 
 * @param error_code 错误代码
 * @return int 是警告返回1，否则返回0
 */
SELFLNN_API int selflnn_is_warning(SelfLNNErrorCode error_code);

/**
 * @brief 错误检查宏
 * 
 * 错误检查代码：如果条件为假，则设置错误并返回错误代码。
 */
#define SELFLNN_CHECK(condition, error_code, ...) \
    do { \
        if (!(condition)) { \
            selflnn_set_last_error((error_code), __func__, __FILE__, __LINE__, __VA_ARGS__); \
            return (error_code); \
        } \
    } while(0)

/**
 * @brief 空指针检查宏
 */
#define SELFLNN_CHECK_NULL(ptr, ...) \
    SELFLNN_CHECK((ptr) != NULL, SELFLNN_ERROR_NULL_POINTER, __VA_ARGS__)

/**
 * @brief 内存分配检查宏
 */
#define SELFLNN_CHECK_MEMORY(ptr, ...) \
    SELFLNN_CHECK((ptr) != NULL, SELFLNN_ERROR_OUT_OF_MEMORY, __VA_ARGS__)

/**
 * @brief 初始化状态检查宏
 */
#define SELFLNN_CHECK_INITIALIZED(obj, ...) \
    SELFLNN_CHECK((obj) != NULL && (obj)->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED, __VA_ARGS__)

/**
 * @brief 通用失败宏 - 记录错误上下文并返回-1
 *
 * 替代直接使用 return -1; 的模式，在返回之前记录错误上下文信息。
 * 调用 selflnn_get_last_error_context() 获取详细错误信息。
 */
#define SELFLNN_CHECK_FAIL(...) \
    do { \
        selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        return -1; \
    } while(0)

/**
 * @brief 错误恢复上下文
 * 
 * 使用setjmp/longjmp实现非局部错误跳转。
 * 每个函数在进入时需要保存上下文，出错时通过longjmp跳回。
 */
typedef struct {
    jmp_buf env;                        /**< setjmp/longjmp环境 */
    SelfLNNErrorCode caught_error;      /**< 捕获的错误代码 */
    int has_error;                      /**< 是否有错误 */
} SelfLNNRecoveryContext;

/**
 * @brief 创建错误恢复上下文
 * 
 * @param context 恢复上下文指针
 * @return int 如果直接返回则为0，如果从错误恢复则为错误代码
 */
SELFLNN_API int selflnn_recovery_try(SelfLNNRecoveryContext* context);

/**
 * @brief 抛出错误（通过longjmp跳转到最近的恢复点）
 * 
 * @param context 恢复上下文指针
 * @param error_code 错误代码
 */
SELFLNN_API void selflnn_recovery_throw(SelfLNNRecoveryContext* context, SelfLNNErrorCode error_code);

/**
 * @brief 检查当前是否有错误需要恢复
 * 
 * @return int 有错误返回1，否则返回0
 */
SELFLNN_API int selflnn_has_recovery_error(void);

/**
 * @brief 获取当前待恢复的错误代码
 * 
 * @return SelfLNNErrorCode 错误代码，无错误返回SELFLNN_SUCCESS
 */
SELFLNN_API SelfLNNErrorCode selflnn_get_recovery_error(void);

/**
 * @brief 清除恢复状态
 */
SELFLNN_API void selflnn_clear_recovery(void);

/**
 * @brief 错误恢复重试策略（指数退避）
 * 
 * @param error_type 错误类型
 * @param retry_func 重试函数指针
 * @param user_data 用户数据
 * @return int 成功返回0，全部重试失败返回-1
 */
SELFLNN_API int selflnn_recovery_retry(SelfLNNErrorCode error_type, 
    int (*retry_func)(void*), void* user_data);

/**
 * @brief 错误恢复隔离策略（标记组件为隔离状态）
 * 
 * @param error_type 错误类型
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int selflnn_recovery_isolate(SelfLNNErrorCode error_type);

/**
 * @brief 检查组件是否已被隔离
 * 
 * @param error_type 错误类型
 * @return int 已隔离返回1，否则返回0
 */
SELFLNN_API int selflnn_is_component_isolated(SelfLNNErrorCode error_type);

/**
 * @brief 重置指定错误类型的恢复状态
 * 
 * @param error_type 错误类型
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int selflnn_recovery_reset(SelfLNNErrorCode error_type);

/**
 * @brief 带自动清理的TRY块
 * 
 * 使用方式：
 * SELFLNN_TRY(context) {
 *     // 可能出错的代码
 *     SELFLNN_THROW(context, SELFLNN_ERROR_GENERIC);
 * } SELFLNN_CATCH(context) {
 *     // 错误处理代码
 *     const SelfLNNErrorContext* err = selflnn_get_last_error_context();
 * } SELFLNN_END_TRY
 */
#define SELFLNN_TRY(context) \
    do { \
        SelfLNNRecoveryContext* _try_ctx = (context); \
        int _try_result = selflnn_recovery_try(_try_ctx); \
        if (_try_result == 0) {

#define SELFLNN_CATCH(context) \
            _try_ctx->caught_error = SELFLNN_SUCCESS; \
        } \
        if (_try_result != 0) { \
            (void)_try_result; \
            _try_ctx->caught_error = (SelfLNNErrorCode)_try_result;

#define SELFLNN_END_TRY \
        } \
        selflnn_clear_recovery(); \
    } while(0)

/**
 * @brief 在TRY块中抛出错误
 */
#define SELFLNN_THROW(context, error_code) \
    do { \
        selflnn_set_last_error((error_code), __func__, __FILE__, __LINE__, "错误恢复抛出"); \
        selflnn_recovery_throw((context), (error_code)); \
    } while(0)

/**
 * @brief 带错误传播的检查宏（在TRY块中使用）
 * 
 * 如果条件不满足，设置错误并通过longjmp跳转到CATCH块。
 */
#define SELFLNN_TRY_CHECK(context, condition, error_code, ...) \
    do { \
        if (!(condition)) { \
            selflnn_set_last_error((error_code), __func__, __FILE__, __LINE__, __VA_ARGS__); \
            selflnn_recovery_throw((context), (error_code)); \
        } \
    } while(0)

/**
 * @brief TRY块中的空指针检查
 */
#define SELFLNN_TRY_CHECK_NULL(context, ptr, ...) \
    SELFLNN_TRY_CHECK((context), (ptr) != NULL, SELFLNN_ERROR_NULL_POINTER, __VA_ARGS__)

/**
 * @brief TRY块中的内存分配检查
 */
#define SELFLNN_TRY_CHECK_MEMORY(context, ptr, ...) \
    SELFLNN_TRY_CHECK((context), (ptr) != NULL, SELFLNN_ERROR_OUT_OF_MEMORY, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_ERRORS_H