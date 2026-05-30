/**
 * @file task_scheduler.h
 * @brief K-038: 抢占式优先级任务调度器接口
 */

#ifndef SELFLNN_TASK_SCHEDULER_H
#define SELFLNN_TASK_SCHEDULER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TaskScheduler TaskScheduler;

typedef int (*TaskSchedulerFunc)(void* context);

TaskScheduler* task_scheduler_create(void);

void task_scheduler_free(TaskScheduler* s);

int task_scheduler_submit(TaskScheduler* s, const char* name,
                           TaskSchedulerFunc func, void* context,
                           int priority, int timeout_seconds,
                           const int* depends_on, int dep_count);

int task_scheduler_tick(TaskScheduler* s);

void task_scheduler_get_stats(const TaskScheduler* s,
                               int* total, int* pending,
                               int* running, int* completed,
                               int* preempted, int* failed);

/**
 * @brief 注册线程池用于异步任务调度
 * 注册后任务通过线程池异步执行，不阻塞调度器主循环。
 * @param s 调度器句柄
 * @param pool 线程池指针
 * @param submit_fn 线程池提交函数(void* pool, void(*fn)(void*), void* arg)
 * @return int 成功返回0
 */
int task_scheduler_register_thread_pool(TaskScheduler* s,
                                         void* pool,
                                         int (*submit_fn)(void* pool, void (*fn)(void*), void* arg));

/**
 * @brief 暂停所有任务调度
 * @param s 调度器句柄
 */
void task_scheduler_pause(TaskScheduler* s);

/**
 * @brief 取消所有待处理任务
 * @param s 调度器句柄
 */
void task_scheduler_cancel_all(TaskScheduler* s);

#ifdef __cplusplus
}
#endif

#endif
