/**
 * @file thread_pool.h
 * @brief 线程池接口
 * 
 * 高性能线程池实现，支持任务队列和工作线程管理。
 */

#ifndef SELFLNN_THREAD_POOL_H
#define SELFLNN_THREAD_POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 任务函数指针类型
 */
typedef void (*TaskFunction)(void* arg);

/**
 * @brief 线程池任务结构
 */
typedef struct {
    TaskFunction function;   /**< 任务函数 */
    void* argument;          /**< 任务参数 */
    int priority;            /**< 任务优先级 */
} ThreadPoolTask;

/**
 * @brief 线程池配置
 */
typedef struct {
    size_t num_threads;      /**< 线程数 */
    size_t max_tasks;        /**< 最大任务数 */
    int dynamic_scaling;     /**< 是否动态伸缩 */
    int enable_priority;     /**< 是否启用优先级 */
    int enable_work_stealing; /**< 是否启用工作窃取 */
    size_t max_tasks_per_thread; /**< 每个线程队列最大任务数 */
    size_t work_stealing_threshold; /**< 工作窃取阈值 */
    int task_timeout_ms;     /**< 任务超时时间（毫秒） */
    int idle_thread_timeout_ms; /**< 空闲线程超时时间（毫秒） */
} ThreadPoolConfig;

/**
 * @brief 线程池句柄
 */
typedef struct ThreadPool ThreadPool;

/**
 * @brief 创建线程池
 * 
 * @param config 线程池配置
 * @return ThreadPool* 线程池句柄，失败返回NULL
 */
ThreadPool* thread_pool_create(const ThreadPoolConfig* config);

/**
 * @brief 释放线程池
 * 
 * @param pool 线程池句柄
 */
void thread_pool_free(ThreadPool* pool);

/**
 * @brief 提交任务到线程池
 * 
 * @param pool 线程池句柄
 * @param function 任务函数
 * @param argument 任务参数
 * @param priority 任务优先级
 * @return int 成功返回0，失败返回-1
 */
int thread_pool_submit(ThreadPool* pool, TaskFunction function, 
                      void* argument, int priority);

/**
 * @brief 批量提交任务
 * 
 * @param pool 线程池句柄
 * @param tasks 任务数组
 * @param num_tasks 任务数量
 * @return int 成功提交的任务数，失败返回-1
 */
int thread_pool_submit_batch(ThreadPool* pool, const ThreadPoolTask* tasks,
                            size_t num_tasks);

/**
 * @brief 等待所有任务完成
 * 
 * @param pool 线程池句柄
 * @param timeout_ms 超时时间（毫秒），0表示无限等待
 * @return int 成功返回0，超时返回1，失败返回-1
 */
int thread_pool_wait_all(ThreadPool* pool, unsigned int timeout_ms);

/**
 * @brief 获取线程池统计信息
 * 
 * @param pool 线程池句柄
 * @param active_threads 活跃线程数输出缓冲区
 * @param pending_tasks 等待任务数输出缓冲区
 * @param completed_tasks 已完成任务数输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int thread_pool_get_stats(const ThreadPool* pool,
                         size_t* active_threads,
                         size_t* pending_tasks,
                         size_t* completed_tasks);

/**
 * @brief 调整线程池大小
 * 
 * @param pool 线程池句柄
 * @param new_num_threads 新线程数
 * @return int 成功返回0，失败返回-1
 */
int thread_pool_resize(ThreadPool* pool, size_t new_num_threads);

/**
 * @brief 暂停线程池
 * 
 * @param pool 线程池句柄
 * @return int 成功返回0，失败返回-1
 */
int thread_pool_pause(ThreadPool* pool);

/**
 * @brief 恢复线程池
 * 
 * @param pool 线程池句柄
 * @return int 成功返回0，失败返回-1
 */
int thread_pool_resume(ThreadPool* pool);

/**
 * @brief 获取线程池配置
 * 
 * @param pool 线程池句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int thread_pool_get_config(const ThreadPool* pool, ThreadPoolConfig* config);

/**
 * @brief 设置线程池配置
 * 
 * @param pool 线程池句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int thread_pool_set_config(ThreadPool* pool, const ThreadPoolConfig* config);

/**
 * @brief 重置线程池统计信息
 * 
 * @param pool 线程池句柄
 */
void thread_pool_reset_stats(ThreadPool* pool);

// ==================== 无锁队列集成接口 ====================

/**
 * @brief 通过无锁队列提交高优先级任务
 * 
 * @param pool 线程池句柄
 * @param function 任务函数
 * @param argument 任务参数
 * @return int 成功返回0，失败返回-1
 */
int thread_pool_submit_lf(ThreadPool* pool, TaskFunction function, void* argument);

/**
 * @brief 从无锁队列中取出任务
 * 
 * @param pool 线程池句柄
 * @param task 任务输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int thread_pool_receive_lf(ThreadPool* pool, ThreadPoolTask* task);

/**
 * @brief 获取无锁任务队列大小
 * 
 * @param pool 线程池句柄
 * @return size_t 队列大小，失败返回0
 */
size_t thread_pool_lf_queue_size(const ThreadPool* pool);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_THREAD_POOL_H