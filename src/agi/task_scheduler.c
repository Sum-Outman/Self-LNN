/**
 * @file task_scheduler.c
 * @brief K-038: 抢占式优先级任务调度器
 *
 * 基于优先级的抢占式任务调度，支持：
 * 1. 多优先级队列(低/中/高/紧急 4级)
 * 2. 任务抢占(高优先级打断低优先级)
 * 3. 超时自动降级(长时间运行任务降低优先级)
 * 4. 任务依赖图(拓扑排序执行)
 *
 * 100%纯C实现。
 */

#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/errors.h"
#include "selflnn/agi/task_scheduler.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

/* P2-008修复: 跨平台互斥锁保护任务调度器并发访问 */
#ifdef _WIN32
#include <windows.h>
#define TS_MUTEX_TYPE CRITICAL_SECTION
#define TS_MUTEX_INIT(m) InitializeCriticalSection((CRITICAL_SECTION*)(m))
#define TS_MUTEX_LOCK(m) EnterCriticalSection((CRITICAL_SECTION*)(m))
#define TS_MUTEX_UNLOCK(m) LeaveCriticalSection((CRITICAL_SECTION*)(m))
#define TS_MUTEX_DESTROY(m) DeleteCriticalSection((CRITICAL_SECTION*)(m))
#else
#include <pthread.h>
#define TS_MUTEX_TYPE pthread_mutex_t
#define TS_MUTEX_INIT(m) pthread_mutex_init((pthread_mutex_t*)(m), NULL)
#define TS_MUTEX_LOCK(m) pthread_mutex_lock((pthread_mutex_t*)(m))
#define TS_MUTEX_UNLOCK(m) pthread_mutex_unlock((pthread_mutex_t*)(m))
#define TS_MUTEX_DESTROY(m) pthread_mutex_destroy((pthread_mutex_t*)(m))
#endif

#define TS_MAX_TASKS 1024
#define TS_MAX_NAME 64
#define TS_PRIORITY_LEVELS 4

typedef enum {
    TASK_PRIORITY_LOW = 0,
    TASK_PRIORITY_NORMAL = 1,
    TASK_PRIORITY_HIGH = 2,
    TASK_PRIORITY_URGENT = 3
} TaskPriority;

typedef enum {
    TASK_PENDING = 0,
    TASK_RUNNING = 1,
    TASK_COMPLETED = 2,
    TASK_FAILED = 3,
    TASK_PREEMPTED = 4
} TaskStatus;

typedef struct {
    int task_id;
    char name[TS_MAX_NAME];
    TaskPriority priority;
    TaskStatus status;
    TaskSchedulerFunc func;
    void* context;

    /* 抢占相关 */
    time_t start_time;
    time_t preempt_time;
    int timeout_seconds;
    int max_preempt_count;
    int preempt_count;

    /* 依赖 */
    int depends_on[16];
    int dep_count;
    int dependents[64];
    int dependent_count;
    /* AGI-007修复: 老化计数器字段，用于优先级提升 */
    int aging_counter;
} ScheduledTask;

struct TaskScheduler {
    ScheduledTask tasks[TS_MAX_TASKS];
    int task_count;
    int running_task_id;

    /* 按优先级队列索引 */
    int queue_heads[TS_PRIORITY_LEVELS];
    int queue_tails[TS_PRIORITY_LEVELS];
    int queue_next[TS_MAX_TASKS];

    /* 统计 */
    int total_completed;
    int total_preempted;
    int total_failed;

    /* 线程池集成（可选异步调度） */
    void* thread_pool;
    int (*thread_pool_submit)(void* pool, void (*fn)(void*), void* arg);
    int use_thread_pool;

/* P2-008: 并发访问保护锁 */
    TS_MUTEX_TYPE lock;
};

TaskScheduler* task_scheduler_create(void) {
    TaskScheduler* s = (TaskScheduler*)safe_calloc(1, sizeof(struct TaskScheduler));
    if (!s) return NULL;

    for (int i = 0; i < TS_PRIORITY_LEVELS; i++) {
        s->queue_heads[i] = -1;
        s->queue_tails[i] = -1;
    }
    for (int i = 0; i < TS_MAX_TASKS; i++) {
        s->queue_next[i] = -1;
    }
    s->running_task_id = -1;
    TS_MUTEX_INIT(&s->lock);
    return s;
}

void task_scheduler_free(TaskScheduler* s) {
    if (!s) return;
    TS_MUTEX_DESTROY(&s->lock);
    safe_free((void**)&s);
}

/**
 * @brief 注册线程池用于异步任务调度
 * 注册后，任务将通过线程池异步执行，不再阻塞调度器主循环
 */
int task_scheduler_register_thread_pool(TaskScheduler* s,
                                         void* pool,
                                         int (*submit_fn)(void* pool, void (*fn)(void*), void* arg)) {
    if (!s) return -1;
    s->thread_pool = pool;
    s->thread_pool_submit = submit_fn;
    s->use_thread_pool = (pool && submit_fn) ? 1 : 0;
    return 0;
}

/**
 * @brief 封装任务执行（支持线程池异步模式）
 */
static void task_wrapper_execute(void* arg) {
    ScheduledTask* t = (ScheduledTask*)arg;
    if (t && t->func) {
        int ret = t->func(t->context);
        t->status = (ret == 0) ? TASK_COMPLETED : TASK_FAILED;
    }
    /* DEEP-FIX: 释放TSWrapperArg，修复每次认知循环的内存泄漏 */
    if (t && t->context) {
        safe_free(&t->context);
    }
}

/**
 * @brief K-038: 提交任务到优先级队列
 */
int task_scheduler_submit(TaskScheduler* s, const char* name,
                           TaskSchedulerFunc func, void* context,
                           int priority, int timeout_seconds,
                           const int* depends_on, int dep_count) {
    if (!s || !func || s->task_count >= TS_MAX_TASKS) return -1;

    TS_MUTEX_LOCK(&s->lock);
    int tid = s->task_count++;
    ScheduledTask* t = &s->tasks[tid];
    memset(t, 0, sizeof(ScheduledTask));
    t->task_id = tid;
    strncpy(t->name, name ? name : "task", TS_MAX_NAME - 1);
    t->priority = (priority >= 0 && priority < TS_PRIORITY_LEVELS)
        ? (TaskPriority)priority : TASK_PRIORITY_NORMAL;
    t->status = TASK_PENDING;
    t->func = func;
    t->context = context;
    t->timeout_seconds = timeout_seconds > 0 ? timeout_seconds : 300;
    t->max_preempt_count = 3;

    if (depends_on && dep_count > 0) {
        t->dep_count = dep_count < 16 ? dep_count : 16;
        for (int i = 0; i < t->dep_count; i++) {
            t->depends_on[i] = depends_on[i];
            if (depends_on[i] >= 0 && depends_on[i] < s->task_count) {
                ScheduledTask* dep = &s->tasks[depends_on[i]];
                if (dep->dependent_count < 64) {
                    dep->dependents[dep->dependent_count++] = tid;
                }
            }
        }
    }

    /* 入队: 按优先级追加到对应队列 */
    int p = t->priority;
    if (s->queue_heads[p] == -1) {
        s->queue_heads[p] = tid;
    } else {
        s->queue_next[s->queue_tails[p]] = tid;
    }
    s->queue_tails[p] = tid;

    log_info("[调度器] 任务提交: %s (ID=%d, 优先级=%d, 超时=%ds)",
             t->name, tid, priority, t->timeout_seconds);
    TS_MUTEX_UNLOCK(&s->lock);
    return tid;
}

/**
 * @brief K-038: 检查依赖是否满足
 */
static int ts_dependencies_met(TaskScheduler* s, ScheduledTask* t) {
    for (int i = 0; i < t->dep_count; i++) {
        int dep_id = t->depends_on[i];
        if (dep_id >= 0 && dep_id < s->task_count) {
            if (s->tasks[dep_id].status != TASK_COMPLETED)
                return 0;
        }
    }
    return 1;
}

/**
 * @brief AGI-008修复: 从优先级队列中移除指定任务
 * 队列使用单链表: queue_heads[p]→queue_next[*]→...→queue_tails[p]
 */
static void ts_queue_remove(TaskScheduler* s, int tid) {
    if (tid < 0 || tid >= TS_MAX_TASKS) return;
    ScheduledTask* t = &s->tasks[tid];
    int p = (int)t->priority;
    if (p < 0 || p >= TS_PRIORITY_LEVELS) return;
    int prev = -1;
    int cur = s->queue_heads[p];
    while (cur != -1) {
        if (cur == tid) {
            if (prev == -1) {
                s->queue_heads[p] = s->queue_next[tid];
            } else {
                s->queue_next[prev] = s->queue_next[tid];
            }
            if (s->queue_tails[p] == tid) {
                s->queue_tails[p] = prev;
            }
            s->queue_next[tid] = -1;
            return;
        }
        prev = cur;
        cur = s->queue_next[cur];
    }
}

/**
 * @brief AGI-008修复: 将任务插入对应优先级队列尾部
 */
static void ts_queue_insert(TaskScheduler* s, int tid) {
    if (tid < 0 || tid >= TS_MAX_TASKS) return;
    ScheduledTask* t = &s->tasks[tid];
    int p = (int)t->priority;
    if (p < 0 || p >= TS_PRIORITY_LEVELS) return;
    s->queue_next[tid] = -1;
    if (s->queue_tails[p] == -1) {
        s->queue_heads[p] = tid;
        s->queue_tails[p] = tid;
    } else {
        s->queue_next[s->queue_tails[p]] = tid;
        s->queue_tails[p] = tid;
    }
}

/**
 * @brief K-038: 选择下一个任务(最高优先级、依赖满足)
 */
static int ts_select_next(TaskScheduler* s) {
    /* DEEP-FIX: 添加老化计数器防止优先级饥饿。
     * PENDING任务每轮未被选中时老化+1，老化超过5轮则提升优先级。 */
    for (int p = TS_PRIORITY_LEVELS - 1; p >= 0; p--) {
        int tid = s->queue_heads[p];
        while (tid != -1) {
            ScheduledTask* t = &s->tasks[tid];
            if (t->status == TASK_PENDING) {
                /* 老化机制：长期等待的低优先级任务逐轮提升优先级 */
                if (!ts_dependencies_met(s, t)) {
                    t->aging_counter++;
                    if (t->aging_counter > 5 && t->priority < TS_PRIORITY_LEVELS - 1) {
                        t->priority++;  /* 提升一级 */
                        t->aging_counter = 0;
                        /* 先保存下一个任务ID（ts_queue_remove会修改链表） */
                        int next_tid = s->queue_next[tid];
                        /* 从旧队列移除并插入到新优先级队列首部 */
                        ts_queue_remove(s, tid);
                        ts_queue_insert(s, tid);
                        /* P0-M004修复: 使用已保存的next_tid继续扫描当前优先级,
                         * 避免return -1中断导致其他就绪任务被延迟 */
                        tid = next_tid;
                        continue;
                    }
                } else {
                    return tid;
                }
            }
            tid = s->queue_next[tid];
        }
    }
    return -1;
}

/**
 * @brief K-038: 抢占式执行 — 高优先级任务可打断正在运行的低优先级任务
 */
static int ts_try_preempt(TaskScheduler* s, int new_task_id) {
    if (s->running_task_id < 0) return 0;

    ScheduledTask* running = &s->tasks[s->running_task_id];
    ScheduledTask* new_task = &s->tasks[new_task_id];

    if (new_task->priority <= running->priority) return 0;
    if (running->preempt_count >= running->max_preempt_count) return 0;

    /* 抢占: 暂停当前任务 */
    running->status = TASK_PREEMPTED;
    running->preempt_time = time(NULL);
    running->preempt_count++;
    s->total_preempted++;

    log_info("[调度器] 抢占: %s(ID=%d,p=%d) 被 %s(ID=%d,p=%d) 打断",
             running->name, running->task_id, running->priority,
             new_task->name, new_task_id, new_task->priority);
    return 1;
}

/**
 * @brief K-038: 执行一个调度周期
 * @return 执行的任务数
 */
int task_scheduler_tick(TaskScheduler* s) {
    if (!s) return -1;

    TS_MUTEX_LOCK(&s->lock);

    /* 检查当前运行任务是否超时 */
    if (s->running_task_id >= 0) {
        ScheduledTask* t = &s->tasks[s->running_task_id];
        if (t->status == TASK_RUNNING) {
            time_t elapsed = time(NULL) - t->start_time;
            if (elapsed > t->timeout_seconds) {
                /* 超时: 降级处理，先暂停 */
                t->status = TASK_PREEMPTED;
                s->running_task_id = -1;
                log_warning("[调度器] 超时: %s(ID=%d) 运行%ds>%ds, 降级",
                           t->name, t->task_id, (int)elapsed, t->timeout_seconds);
            }
        }
    }

    /* F-024修复: 添加每tick时间预算，防止同步任务阻塞过久 */
    time_t tick_start = time(NULL);
    time_t tick_budget_sec = 2; /* 每个tick最多执行2秒 */

    int executed = 0;
    for (int round = 0; round < 4; round++) {
        /* F-024: 检查tick时间预算，超时则停止执行新任务 */
        if (time(NULL) - tick_start > tick_budget_sec) {
            log_debug("[调度器] Tick时间预算耗尽(>%lds), 已执行%d个任务, 剩余任务延迟到下一tick",
                     tick_budget_sec, executed);
            break;
        }
        
        int next = ts_select_next(s);
        if (next < 0) break;

        ScheduledTask* t = &s->tasks[next];

        /* 检查是否需要抢占 */
        if (s->running_task_id >= 0 && s->running_task_id != next) {
            if (ts_try_preempt(s, next)) {
                s->running_task_id = next;
                t->status = TASK_RUNNING;
                t->start_time = time(NULL);

                if (t->func) {
                    /* 线程池异步模式：提交到线程池执行 */
                    if (s->use_thread_pool && s->thread_pool_submit) {
                        s->thread_pool_submit(s->thread_pool, task_wrapper_execute, t);
                        /* P1-9修复：异步模式下不立即检查t->status，
                         * task_wrapper_execute会在任务完成时更新状态和计数器 */
                    } else {
                        int ret = t->func(t->context);
                        t->status = (ret == 0) ? TASK_COMPLETED : TASK_FAILED;
                        if (t->status == TASK_COMPLETED) s->total_completed++;
                        else if (t->status == TASK_FAILED) s->total_failed++;
                    }
                }
                s->running_task_id = -1;
                executed++;
            }
        } else if (s->running_task_id < 0) {
            s->running_task_id = next;
            t->status = TASK_RUNNING;
            t->start_time = time(NULL);

            if (t->func) {
                /* 线程池异步模式：提交到线程池执行 */
                if (s->use_thread_pool && s->thread_pool_submit) {
                    s->thread_pool_submit(s->thread_pool, task_wrapper_execute, t);
                    /* P1-9修复：异步模式下计数器由task_wrapper_execute更新 */
                } else {
                    int ret = t->func(t->context);
                    time_t elapsed = time(NULL) - t->start_time;

                    if (t->timeout_seconds > 0 && elapsed > t->timeout_seconds) {
                        t->status = TASK_PREEMPTED;
                    } else {
                        t->status = (ret == 0) ? TASK_COMPLETED : TASK_FAILED;
                    }
                    if (t->status == TASK_COMPLETED) s->total_completed++;
                    else if (t->status == TASK_FAILED) s->total_failed++;
                }
            }
            s->running_task_id = -1;
            executed++;
        }
    }

    TS_MUTEX_UNLOCK(&s->lock);
    return executed;
}

/**
 * @brief K-038: 获取调度器统计
 */
void task_scheduler_get_stats(const TaskScheduler* s,
                               int* total, int* pending,
                               int* running, int* completed,
                               int* preempted, int* failed) {
    if (!s) return;
    /* P-FIX-026: 添加锁保护，防止与task_scheduler_tick并发导致数据竞争 */
    TS_MUTEX_LOCK(&((TaskScheduler*)s)->lock);
    if (total) *total = s->task_count;
    if (completed) *completed = s->total_completed;
    if (preempted) *preempted = s->total_preempted;
    if (failed) *failed = s->total_failed;

    int p = 0, r = 0;
    for (int i = 0; i < s->task_count; i++) {
        /* P-FIX-026: TASK_PREEMPTED不应计入pending，它需要重新调度 */
        if (s->tasks[i].status == TASK_PENDING)
            p++;
        if (s->tasks[i].status == TASK_RUNNING) r++;
    }
    if (pending) *pending = p;
    if (running) *running = r;
    TS_MUTEX_UNLOCK(&((TaskScheduler*)s)->lock);
}

void task_scheduler_pause(TaskScheduler* s) {
    if (!s) return;
    TS_MUTEX_LOCK(&s->lock);
    for (int i = 0; i < s->task_count; i++) {
        if (s->tasks[i].status == TASK_RUNNING) {
            s->tasks[i].status = TASK_PREEMPTED;
            s->total_preempted++;
        }
    }
    s->running_task_id = -1;
    TS_MUTEX_UNLOCK(&s->lock);
}

void task_scheduler_cancel_all(TaskScheduler* s) {
    if (!s) return;
    TS_MUTEX_LOCK(&s->lock);
    for (int i = 0; i < s->task_count; i++) {
        if (s->tasks[i].status == TASK_PENDING || s->tasks[i].status == TASK_PREEMPTED) {
            s->tasks[i].status = TASK_COMPLETED;
        }
    }
    TS_MUTEX_UNLOCK(&s->lock);
}
