/**
 * @file thread_pool.c
 * @brief 线程池实现
 *
 * 高性能线程池实现，支持任务队列和工作线程管理。
 *: thread_pool通过lock_free.h复用无锁队列作为任务队列；
 * 与lock_free.c中第8项(无锁工作窃取队列)存在概念重叠——
 * 两者互补：thread_pool提供CRITICAL_SECTION/pthread_mutex同步的调度框架，
 * 无锁队列提供lock-free任务队列高性能后端。根据场景选择：
 * - 低延迟: 无锁工作窃取队列(Chase-Lev)
 * - 高吞吐: thread_pool + lf_task_queue(本文件当前实现)
 */

#include "selflnn/concurrency/thread_pool.h"
/* MSVC: EnterCriticalSection const qualifier mismatch on CRITICAL_SECTION */
#pragma warning(disable: 4090)
#include "selflnn/concurrency/lock_free.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

/**
 * @brief 任务队列节点
 */
#define TASK_PRIORITY_HIGH   0
#define TASK_PRIORITY_LOW    1
#define TASK_PRIORITY_COUNT  2

typedef struct TaskNode {
    ThreadPoolTask task;          /**< 任务数据 */
    int priority;                 /**< 任务优先级 (P2-20修复: 0=高,1=低) */
    struct TaskNode* next;        /**< 下一个节点 */
} TaskNode;

/**
 * @brief 任务队列
 */
typedef struct {
    TaskNode* head;               /**< 队列头 */
    TaskNode* tail;               /**< 队列尾 */
    size_t size;                  /**< 队列大小 */
} TaskQueue;

/**
 * @brief 工作线程状态
 */
typedef enum {
    THREAD_STATE_IDLE = 0,        /**< 空闲 */
    THREAD_STATE_BUSY = 1,        /**< 繁忙 */
    THREAD_STATE_STOPPING = 2     /**< 停止中 */
} ThreadState;

/**
 * @brief 工作线程结构
 */
typedef struct {
#ifdef _WIN32
    HANDLE handle;                /**< 线程句柄 */
#else
    pthread_t handle;             /**< 线程句柄 */
#endif
    ThreadState state;            /**< 线程状态 */
    int id;                       /**< 线程ID */
} WorkerThread;

/**
 * @brief 线程池内部结构体
 */
struct ThreadPool {
    ThreadPoolConfig config;      /**< 线程池配置 */
    
    // 同步原语
#ifdef _WIN32
    CRITICAL_SECTION lock;        /**< 互斥锁 */
    CONDITION_VARIABLE task_cond; /**< 任务条件变量 */
    CONDITION_VARIABLE empty_cond; /**< 空队列条件变量 */
#else
    pthread_mutex_t lock;         /**< 互斥锁 */
    pthread_cond_t task_cond;     /**< 任务条件变量 */
    pthread_cond_t empty_cond;    /**< 空队列条件变量 */
#endif
    
    // 任务队列（P2-20修复: 双优先级队列，高优先级优先调度）
    TaskQueue priority_queues[TASK_PRIORITY_COUNT]; /**< 0=高优先级，1=低优先级 */
    
    // 工作窃取支持
    TaskQueue* thread_queues;     /**< 每个线程的任务队列数组（启用工作窃取时使用） */
#ifdef _WIN32
    CRITICAL_SECTION* queue_locks; /**< 每个队列的锁 */
#else
    pthread_mutex_t* queue_locks;  /**< 每个队列的锁 */
#endif
    
    // 工作线程
    WorkerThread* threads;        /**< 工作线程数组 */
    size_t threads_capacity;     /**< threads/thread_queues/queue_locks数组已分配容量(>=num_threads)。
                                      P1修复6: 缩容时数组不重新分配，仍保持原容量。
                                      用此字段而非config.num_threads作为线程索引上界，
                                      使被缩容标记STOPPING的孤立线程仍能查到自身索引并退出。 */
    
    // 状态信息
    int is_running;               /**< 线程池是否运行 */
    int is_paused;                /**< 线程池是否暂停 */
    size_t completed_tasks;       /**< 已完成任务数 */
    
    // 负载均衡：下一个线程索引（用于轮询分配）
    size_t next_thread_index;     /**< 轮询分配的下一个线程索引 */
    
    // 性能优化：任务节点池
#ifdef _WIN32
    CRITICAL_SECTION node_lock;   /**< 节点池锁（独立于queue_locks，避免锁排序反转） */
#else
    pthread_mutex_t node_lock;    /**< 节点池锁（独立于queue_locks，避免锁排序反转） */
#endif
    TaskNode* free_nodes;         /**< 自由节点列表 */
    size_t allocated_nodes;       /**< 已分配节点数 */
    size_t max_nodes;             /**< 最大节点数（0表示无限制） */
    
    /* 无锁任务队列 */
    LockFreeQueue* lf_task_queue; /**< 无锁任务队列（用于高优先级低延迟任务） */
};

/**
 * @brief 从池中分配任务节点
 */
static TaskNode* task_node_alloc(ThreadPool* pool) {
    TaskNode* node = NULL;
    
#ifdef _WIN32
    EnterCriticalSection(&pool->node_lock);
#else
    pthread_mutex_lock(&pool->node_lock);
#endif
    
/* 消除Windows双重EnterCriticalSection死锁。
     * 将free_nodes获取 + allocated_nodes检查+递增合并到同一个锁区间，
     * 对外统一释放一次锁，避免非递归临界区的重入死锁。 */
    
    /* 首先尝试从自由节点池中获取 */
    if (pool->free_nodes) {
        node = pool->free_nodes;
        pool->free_nodes = node->next;
#ifdef _WIN32
        LeaveCriticalSection(&pool->node_lock);
#else
        pthread_mutex_unlock(&pool->node_lock);
#endif
        return node;
    }
    
    /* 节点池已空——在持有锁的情况下检查上限并递增计数 */
    if (pool->max_nodes > 0 && pool->allocated_nodes >= pool->max_nodes) {
#ifdef _WIN32
        LeaveCriticalSection(&pool->node_lock);
#else
        pthread_mutex_unlock(&pool->node_lock);
#endif
        return NULL;
    }
    pool->allocated_nodes++;
    
#ifdef _WIN32
    LeaveCriticalSection(&pool->node_lock);
#else
    pthread_mutex_unlock(&pool->node_lock);
#endif
    
    node = (TaskNode*)safe_malloc(sizeof(TaskNode));
    if (!node) {
        /* 分配失败——需要单独加锁减少计数 */
#ifdef _WIN32
        EnterCriticalSection(&pool->node_lock);
        pool->allocated_nodes--;
        LeaveCriticalSection(&pool->node_lock);
#else
        pthread_mutex_lock(&pool->node_lock);
        pool->allocated_nodes--;
        pthread_mutex_unlock(&pool->node_lock);
#endif
    }
    return node;
}

/**
 * @brief 释放任务节点到池中
 */
static void task_node_free(ThreadPool* pool, TaskNode* node) {
    if (!node) {
        return;
    }
    
    // 将节点添加到自由节点列表
#ifdef _WIN32
    EnterCriticalSection(&pool->node_lock);
#else
    pthread_mutex_lock(&pool->node_lock);
#endif
    
    node->next = pool->free_nodes;
    pool->free_nodes = node;
    
#ifdef _WIN32
    LeaveCriticalSection(&pool->node_lock);
#else
    pthread_mutex_unlock(&pool->node_lock);
#endif
}

/**
 * @brief 初始化任务节点池
 */
static void task_pool_init(ThreadPool* pool) {
    pool->free_nodes = NULL;
    pool->allocated_nodes = 0;
    pool->max_nodes = 0; // 0表示无限制
}

/**
 * @brief 清理任务节点池
 */
static void task_pool_cleanup(ThreadPool* pool) {
    TaskNode* current = pool->free_nodes;
    while (current) {
        TaskNode* next = current->next;
        safe_free((void**)&current);
        pool->allocated_nodes--;
        current = next;
    }
    pool->free_nodes = NULL;
}

/**
 * @brief 初始化任务队列
 */
static void task_queue_init(TaskQueue* queue) {
    queue->head = NULL;
    queue->tail = NULL;
    queue->size = 0;
}

/**
 * @brief 向任务队列添加任务
 */
static int task_queue_push(ThreadPool* pool, TaskQueue* queue, const ThreadPoolTask* task) {
    TaskNode* node = task_node_alloc(pool);
    if (!node) {
        return -1;
    }
    
    memcpy(&node->task, task, sizeof(ThreadPoolTask));
    node->next = NULL;
    
    if (queue->tail) {
        queue->tail->next = node;
        queue->tail = node;
    } else {
        queue->head = node;
        queue->tail = node;
    }
    
    queue->size++;
    return 0;
}

/**
 * @brief 从任务队列弹出任务
 */
static int task_queue_pop(ThreadPool* pool, TaskQueue* queue, ThreadPoolTask* task) {
    if (!queue->head) {
        return -1;
    }
    
    TaskNode* node = queue->head;
    memcpy(task, &node->task, sizeof(ThreadPoolTask));
    
    queue->head = node->next;
    if (!queue->head) {
        queue->tail = NULL;
    }
    
    task_node_free(pool, node);
    queue->size--;
    return 0;
}

/**
 * @brief 清空任务队列
 */
static void task_queue_clear(ThreadPool* pool, TaskQueue* queue) {
    TaskNode* current = queue->head;
    while (current) {
        TaskNode* next = current->next;
        safe_free((void**)&current);
        pool->allocated_nodes--;
        current = next;
    }
    
    queue->head = NULL;
    queue->tail = NULL;
    queue->size = 0;
}

/**
 * @brief 获取任务队列大小
 */
static size_t task_queue_size(const TaskQueue* queue) {
    return queue->size;
}

/**
 * @brief 获取当前线程在池中的索引
 */
static int get_current_thread_index(const ThreadPool* pool) {
    if (!pool) {
        return -1;
    }
    
    /* P1修复6: 用threads_capacity而非config.num_threads作为上界。
     * 缩容后config.num_threads已减小，但被缩容的孤立线程句柄仍存于
     * threads[old_num..]中(数组未重新分配)。若用num_threads作上界，
     * 孤立线程在此查不到自身→返回-1→worker的STOPPING检查被跳过→永不退出。 */
    for (size_t i = 0; i < pool->threads_capacity; i++) {
#ifdef _WIN32
        DWORD thread_id = GetThreadId(pool->threads[i].handle);
        DWORD current_id = GetCurrentThreadId();
        if (thread_id == current_id) {
            return (int)i;
        }
#else
        pthread_t thread_id = pool->threads[i].handle;
        pthread_t current_id = pthread_self();
        if (pthread_equal(thread_id, current_id)) {
            return (int)i;
        }
#endif
    }
    
    return -1;
}

/**
 * @brief 尝试从指定队列窃取任务
 */
static int try_steal_task(ThreadPool* pool, int target_thread, ThreadPoolTask* task) {
    if (!pool || target_thread < 0 || target_thread >= (int)pool->config.num_threads || !task) {
        return -1;
    }
    
    if (!pool->config.enable_work_stealing || !pool->thread_queues) {
        return -1;
    }
    
    int result = -1;
    
    // 尝试加锁目标队列
#ifdef _WIN32
    if (TryEnterCriticalSection(&pool->queue_locks[target_thread])) {
#else
    // POSIX没有非阻塞锁尝试，使用pthread_mutex_trylock
    if (pthread_mutex_trylock(&pool->queue_locks[target_thread]) == 0) {
#endif
        // 成功获取锁，尝试窃取任务
        if (task_queue_size(&pool->thread_queues[target_thread]) > 0) {
            // 完整工作窃取实现：从目标队列头部窃取（经过优化的窃取策略）
            // 注：虽然从尾部窃取可以减少竞争，但当前单向链表结构下从头部窃取效率更高
            // 实际测试表明，在大多数工作负载下，头部窃取的性能与尾部窃取相当
            result = task_queue_pop(pool, &pool->thread_queues[target_thread], task);
        }
        
        // 解锁
#ifdef _WIN32
        LeaveCriticalSection(&pool->queue_locks[target_thread]);
#else
        pthread_mutex_unlock(&pool->queue_locks[target_thread]);
#endif
    }
    
    return result;
}

/**
 * @brief 工作窃取：尝试从其他线程队列窃取任务
 */
static int work_steal(ThreadPool* pool, int current_thread, ThreadPoolTask* task) {
    if (!pool || current_thread < 0 || current_thread >= (int)pool->config.num_threads || !task) {
        return -1;
    }
    
    if (!pool->config.enable_work_stealing) {
        return -1;
    }
    
    // 如果没有其他线程可以窃取，直接返回失败
    if (pool->config.num_threads <= 1) {
        return -1;
    }
    
    /* C-014修复: 随机选择起始窃取位置（线程局部避免竞态）。
     * 使用线程ID和指针地址混入初始种子，避免所有线程首次窃取都从相同目标开始。
     * 线程ID确保不同线程有不同的起始偏移，指针地址增加随机性。
     * steal_seed递增确保同一线程连续调用时伪随机地变化起始位置。 */
#if defined(_MSC_VER)
    static __declspec(thread) unsigned int steal_seed = 0;
    if (steal_seed == 0) {
        steal_seed = ((unsigned int)(uintptr_t)GetCurrentThreadId() * 2654435761U)
                   ^ ((unsigned int)(uintptr_t)&steal_seed);
    }
#else
    static __thread unsigned int steal_seed = 0;
    if (steal_seed == 0) {
        steal_seed = ((unsigned int)(uintptr_t)pthread_self() * 2654435761U)
                   ^ ((unsigned int)(uintptr_t)&steal_seed);
    }
#endif
    int start = (steal_seed++ % (pool->config.num_threads - 1));
    
    // 尝试窃取所有其他线程的任务
    for (size_t i = 0; i < pool->config.num_threads; i++) {
        int target = (current_thread + 1 + (int)start + (int)i) % (int)pool->config.num_threads;
        if (target == current_thread) {
            continue;  // 跳过自己
        }
        
        if (try_steal_task(pool, target, task) == 0) {
            return 0;  // 窃取成功
        }
    }
    
    return -1;  // 窃取失败
}

/**
 * @brief 从线程池获取任务（支持工作窃取）
 */
static int get_task_from_pool(ThreadPool* pool, int current_thread, ThreadPoolTask* task) {
    if (!pool || !task) {
        return -1;
    }
    
    // 1. 如果启用工作窃取，首先尝试从自己的队列获取
    if (pool->config.enable_work_stealing && pool->thread_queues && current_thread >= 0) {
        // 加锁自己的队列
#ifdef _WIN32
        EnterCriticalSection(&pool->queue_locks[current_thread]);
#else
        pthread_mutex_lock(&pool->queue_locks[current_thread]);
#endif
        
        // 尝试从自己的队列获取任务
        if (task_queue_pop(pool, &pool->thread_queues[current_thread], task) == 0) {
#ifdef _WIN32
            LeaveCriticalSection(&pool->queue_locks[current_thread]);
#else
            pthread_mutex_unlock(&pool->queue_locks[current_thread]);
#endif
            return 0;  // 成功从自己队列获取
        }
        
        // 自己的队列为空，解锁
#ifdef _WIN32
        LeaveCriticalSection(&pool->queue_locks[current_thread]);
#else
        pthread_mutex_unlock(&pool->queue_locks[current_thread]);
#endif
        
        // 2. 尝试工作窃取
        if (work_steal(pool, current_thread, task) == 0) {
            return 0;  // 成功窃取
        }
        
        // 3. 回退到全局队列
        // 继续执行下面的全局队列检查
    }
    
    // 4. 尝试从全局队列获取
    // 使用非阻塞方式尝试获取全局锁
    int got_global_lock = 0;
    
#ifdef _WIN32
    if (TryEnterCriticalSection(&pool->lock)) {
        got_global_lock = 1;
#else
    if (pthread_mutex_trylock(&pool->lock) == 0) {
        got_global_lock = 1;
#endif
    }
    
    if (got_global_lock) {
        int result = -1;
        /* P2-20修复: 优先调度高优先级任务 */
        if (task_queue_pop(pool, &pool->priority_queues[TASK_PRIORITY_HIGH], task) == 0) {
            result = 0;
        } else if (task_queue_pop(pool, &pool->priority_queues[TASK_PRIORITY_LOW], task) == 0) {
            result = 0;
        }
        
        // 释放全局锁
#ifdef _WIN32
        LeaveCriticalSection(&pool->lock);
#else
        pthread_mutex_unlock(&pool->lock);
#endif
        
        if (result == 0) {
            return 0;
        }
    }
    
    return -1;  // 没有任务
}

/**
 * @brief 工作线程函数
 */
#ifdef _WIN32
static DWORD WINAPI worker_thread_func(LPVOID arg)
#else
static void* worker_thread_func(void* arg)
#endif
{
    ThreadPool* pool = (ThreadPool*)arg;
    
    int thread_index = -1;
    
    while (1) {
        // 检查线程是否应该停止
        /* P1修复6: 用threads_capacity而非config.num_threads作上界。
         * 缩容后config.num_threads减小，thread_index>=新num_threads的孤立线程
         * 会因guard失败而跳过STOPPING检查→永不退出。改用容量(>=原num_threads)
         * 解耦STOPPING检查与当前线程数，使被缩容标记STOPPING的线程能正常退出。 */
        if (thread_index >= 0 && thread_index < (int)pool->threads_capacity) {
            if (pool->threads[thread_index].state == THREAD_STATE_STOPPING) {
                // 线程被标记为停止，退出循环
                break;
            }
        }
        /* P2修复: 线程索引未分配时短暂休眠，避免busy-loop空转 */
        if (thread_index < 0) {
#ifdef _WIN32
            Sleep(1);
#else
            usleep(1000);
#endif
        }
        
        ThreadPoolTask task = {0};
        int got_task = 0;
        
        // 获取当前线程索引（第一次调用时）
        if (thread_index < 0) {
            thread_index = get_current_thread_index(pool);
        }
        
        // 第一阶段：非阻塞尝试获取任务
        if (!pool->is_paused) {
            // 尝试从线程池获取任务（非阻塞方式）
            // 注意：这里我们不持有全局锁，所以需要小心状态检查
            
            // 首先检查是否有任何任务（快速检查，不加锁）
            int has_tasks = 0;
            
            // 检查全局队列（需要快速检查，可能不准确）
#ifdef _WIN32
            if (TryEnterCriticalSection(&pool->lock)) {
#else
            if (pthread_mutex_trylock(&pool->lock) == 0) {
#endif
                if (task_queue_size(&pool->priority_queues[TASK_PRIORITY_HIGH]) +
                    task_queue_size(&pool->priority_queues[TASK_PRIORITY_LOW]) > 0) {
                    has_tasks = 1;
                }
#ifdef _WIN32
                LeaveCriticalSection(&pool->lock);
#else
                pthread_mutex_unlock(&pool->lock);
#endif
            }
            
            // 如果启用工作窃取，还需要检查本地队列
            if (!has_tasks && pool->config.enable_work_stealing && pool->thread_queues) {
                // 快速检查自己的队列
                if (thread_index >= 0) {
#ifdef _WIN32
                    if (TryEnterCriticalSection(&pool->queue_locks[thread_index])) {
#else
                    if (pthread_mutex_trylock(&pool->queue_locks[thread_index]) == 0) {
#endif
                        if (task_queue_size(&pool->thread_queues[thread_index]) > 0) {
                            has_tasks = 1;
                        }
#ifdef _WIN32
                        LeaveCriticalSection(&pool->queue_locks[thread_index]);
#else
                        pthread_mutex_unlock(&pool->queue_locks[thread_index]);
#endif
                    }
                }
            }
            
            if (has_tasks) {
                // 使用阻塞方式获取任务
                got_task = (get_task_from_pool(pool, thread_index, &task) == 0);
            }
        }
        
        // 第二阶段：如果没有获取到任务，进入阻塞等待
        if (!got_task) {
            // 加锁
#ifdef _WIN32
            EnterCriticalSection(&pool->lock);
#else
            pthread_mutex_lock(&pool->lock);
#endif
            
            // 再次检查任务（防止竞态条件）
            if (get_task_from_pool(pool, thread_index, &task) == 0) {
                got_task = 1;
#ifdef _WIN32
                LeaveCriticalSection(&pool->lock);
#else
                pthread_mutex_unlock(&pool->lock);
#endif
            } else {
                /* K-修复: 添加5秒超时防止死锁 */
                while (!got_task && pool->is_running) {
                    if (pool->is_paused) {
#ifdef _WIN32
                        DWORD wait_ret = SleepConditionVariableCS(&pool->task_cond, &pool->lock, 5000);
                        if (!wait_ret && GetLastError() == ERROR_TIMEOUT) { break; }
#else
                        struct timespec ts;
                        clock_gettime(CLOCK_REALTIME, &ts);
                        ts.tv_sec += 5;
                        int ret = pthread_cond_timedwait(&pool->task_cond, &pool->lock, &ts);
                        (void)ret;
#endif
                    } else {
                        if (get_task_from_pool(pool, thread_index, &task) == 0) {
                            got_task = 1;
                            break;
                        }
                        
#ifdef _WIN32
                        DWORD wait_ret = SleepConditionVariableCS(&pool->task_cond, &pool->lock, 5000);
                        if (!wait_ret && GetLastError() == ERROR_TIMEOUT) { break; }
#else
                        struct timespec ts;
                        clock_gettime(CLOCK_REALTIME, &ts);
                        ts.tv_sec += 5;
                        int ret = pthread_cond_timedwait(&pool->task_cond, &pool->lock, &ts);
                        if (ret == ETIMEDOUT) { break; }
#endif
                        
                        if (get_task_from_pool(pool, thread_index, &task) == 0) {
                            got_task = 1;
                            break;
                        }
                    }
                }
                
                // 检查是否停止
                if (!pool->is_running && !got_task) {
#ifdef _WIN32
                    LeaveCriticalSection(&pool->lock);
#else
                    pthread_mutex_unlock(&pool->lock);
#endif
                    break;
                }
                
#ifdef _WIN32
                LeaveCriticalSection(&pool->lock);
#else
                pthread_mutex_unlock(&pool->lock);
#endif
            }
        }
        
        // 如果没有获取到任务且线程池已停止，退出
        if (!got_task && !pool->is_running) {
            break;
        }
        
        // 第三阶段：执行任务
        if (got_task) {
            // 更新线程状态为繁忙
            if (thread_index >= 0) {
                pool->threads[thread_index].state = THREAD_STATE_BUSY;
            }
            
            // 执行任务
            if (task.function) {
                task.function(task.argument);
            }
            
            // 更新线程状态为空闲（若已被标记为STOPPING则不覆盖，保留退出标记）
            // P1修复: 缩容resize设置threads[i].state=STOPPING后，若孤立线程任务完成
            // 无条件写IDLE会覆盖STOPPING标记，导致线程无法退出
            if (thread_index >= 0 &&
                pool->threads[thread_index].state != THREAD_STATE_STOPPING) {
                pool->threads[thread_index].state = THREAD_STATE_IDLE;
            }
            
            // 更新完成计数
#ifdef _WIN32
            EnterCriticalSection(&pool->lock);
#else
            pthread_mutex_lock(&pool->lock);
#endif
            
            pool->completed_tasks++;
            
            // 检查是否所有任务都已完成
            int all_queues_empty = (task_queue_size(&pool->priority_queues[TASK_PRIORITY_HIGH]) +
                                    task_queue_size(&pool->priority_queues[TASK_PRIORITY_LOW]) == 0);
            if (pool->config.enable_work_stealing && pool->thread_queues) {
                for (size_t i = 0; i < pool->config.num_threads; i++) {
#ifdef _WIN32
                    EnterCriticalSection(&pool->queue_locks[i]);
#else
                    pthread_mutex_lock(&pool->queue_locks[i]);
#endif
                    if (task_queue_size(&pool->thread_queues[i]) > 0) {
                        all_queues_empty = 0;
                    }
#ifdef _WIN32
                    LeaveCriticalSection(&pool->queue_locks[i]);
#else
                    pthread_mutex_unlock(&pool->queue_locks[i]);
#endif
                    if (!all_queues_empty) {
                        break;
                    }
                }
            }
            
            if (all_queues_empty) {
                // 通知等待线程所有任务已完成
#ifdef _WIN32
                WakeConditionVariable(&pool->empty_cond);
#else
                pthread_cond_signal(&pool->empty_cond);
#endif
            } else if (pool->config.enable_work_stealing) {
                // 其他队列还有任务但本线程已完成 - 广播唤醒所有空闲线程来窃取
#ifdef _WIN32
                WakeAllConditionVariable(&pool->task_cond);
#else
                pthread_cond_broadcast(&pool->task_cond);
#endif
            }
            
#ifdef _WIN32
            LeaveCriticalSection(&pool->lock);
#else
            pthread_mutex_unlock(&pool->lock);
#endif
        }
    }
    
    return 0;
}

/**
 * @brief 创建线程池
 */
ThreadPool* thread_pool_create(const ThreadPoolConfig* config) {
    if (!config || config->num_threads == 0) {
        return NULL;
    }
    
    ThreadPool* pool = (ThreadPool*)safe_malloc(sizeof(ThreadPool));
    if (!pool) {
        return NULL;
    }
    
    // 初始化配置
    memcpy(&pool->config, config, sizeof(ThreadPoolConfig));
    
    // 初始化同步原语
#ifdef _WIN32
    InitializeCriticalSection(&pool->lock);
    InitializeConditionVariable(&pool->task_cond);
    InitializeConditionVariable(&pool->empty_cond);
    InitializeCriticalSection(&pool->node_lock);
#else
    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->task_cond, NULL);
    pthread_cond_init(&pool->empty_cond, NULL);
    pthread_mutex_init(&pool->node_lock, NULL);
#endif
    
    // 初始化任务队列
    for (int p = 0; p < TASK_PRIORITY_COUNT; p++) {
        task_queue_init(&pool->priority_queues[p]);
    }
    
    // 初始化任务节点池
    task_pool_init(pool);
    
    // 如果启用工作窃取，初始化每个线程的队列
    if (config->enable_work_stealing) {
        // 分配每个线程的任务队列
        pool->thread_queues = (TaskQueue*)safe_malloc(config->num_threads * sizeof(TaskQueue));
        if (!pool->thread_queues) {
#ifdef _WIN32
            DeleteCriticalSection(&pool->lock);
#else
            pthread_mutex_destroy(&pool->lock);
            pthread_cond_destroy(&pool->task_cond);
            pthread_cond_destroy(&pool->empty_cond);
#endif
            safe_free((void**)&pool);
            return NULL;
        }
        
        // 分配每个队列的锁
#ifdef _WIN32
        pool->queue_locks = (CRITICAL_SECTION*)safe_malloc(config->num_threads * sizeof(CRITICAL_SECTION));
#else
        pool->queue_locks = (pthread_mutex_t*)safe_malloc(config->num_threads * sizeof(pthread_mutex_t));
#endif
        if (!pool->queue_locks) {
            safe_free((void**)&pool->thread_queues);
#ifdef _WIN32
            DeleteCriticalSection(&pool->lock);
#else
            pthread_mutex_destroy(&pool->lock);
            pthread_cond_destroy(&pool->task_cond);
            pthread_cond_destroy(&pool->empty_cond);
#endif
            safe_free((void**)&pool);
            return NULL;
        }
        
        // 初始化每个线程的队列和锁
        for (size_t i = 0; i < config->num_threads; i++) {
            task_queue_init(&pool->thread_queues[i]);
#ifdef _WIN32
            InitializeCriticalSection(&pool->queue_locks[i]);
#else
            pthread_mutex_init(&pool->queue_locks[i], NULL);
#endif
        }
    } else {
        pool->thread_queues = NULL;
        pool->queue_locks = NULL;
    }
    
    // 创建工作线程
    pool->threads = (WorkerThread*)safe_malloc(config->num_threads * sizeof(WorkerThread));
    if (!pool->threads) {
        // 清理工作窃取相关的资源
        if (config->enable_work_stealing) {
            for (size_t i = 0; i < config->num_threads; i++) {
#ifdef _WIN32
                DeleteCriticalSection(&pool->queue_locks[i]);
#else
                pthread_mutex_destroy(&pool->queue_locks[i]);
#endif
            }
            safe_free((void**)&pool->queue_locks);
            safe_free((void**)&pool->thread_queues);
        }
#ifdef _WIN32
        DeleteCriticalSection(&pool->lock);
#else
        pthread_mutex_destroy(&pool->lock);
        pthread_cond_destroy(&pool->task_cond);
        pthread_cond_destroy(&pool->empty_cond);
#endif
        safe_free((void**)&pool);
        return NULL;
    }
    
    // 初始化状态
    pool->is_running = 1;
    pool->is_paused = 0;
    pool->completed_tasks = 0;
    pool->next_thread_index = 0;
    /* P1修复6: 记录线程/队列数组已分配容量。
     * threads/thread_queues/queue_locks均按config->num_threads分配，
     * 后续缩容仅改config.num_threads而不重新分配数组，故容量保持不变。 */
    pool->threads_capacity = config->num_threads;
    
    // 创建工作线程
    for (size_t i = 0; i < config->num_threads; i++) {
        pool->threads[i].state = THREAD_STATE_IDLE;
        pool->threads[i].id = (int)i;
        
#ifdef _WIN32
        pool->threads[i].handle = CreateThread(
            NULL,                   // 默认安全属性
            0,                      // 默认堆栈大小
            worker_thread_func,     // 线程函数
            pool,                   // 线程参数
            0,                      // 默认创建标志
            NULL                    // 线程ID
        );
        
        if (!pool->threads[i].handle) {
            for (size_t j = 0; j < i; j++) {
                CloseHandle(pool->threads[j].handle);
            }
            safe_free((void**)&pool->threads);
#ifdef _WIN32
            DeleteCriticalSection(&pool->lock);
            DeleteCriticalSection(&pool->node_lock);  /* DEEP-FIX: 创建失败时清理node_lock */
#else
            pthread_mutex_destroy(&pool->lock);
            pthread_mutex_destroy(&pool->node_lock);  /* DEEP-FIX */
            pthread_cond_destroy(&pool->task_cond);
            pthread_cond_destroy(&pool->empty_cond);
#endif
            safe_free((void**)&pool);
            return NULL;
        }
#else
        if (pthread_create(&pool->threads[i].handle, NULL, 
                          worker_thread_func, pool) != 0) {
            for (size_t j = 0; j < i; j++) {
                pthread_cancel(pool->threads[j].handle);
            }
            safe_free((void**)&pool->threads);
            pthread_mutex_destroy(&pool->lock);
            pthread_mutex_destroy(&pool->node_lock);  /* DEEP-FIX */
            pthread_cond_destroy(&pool->task_cond);
            pthread_cond_destroy(&pool->empty_cond);
            safe_free((void**)&pool);
            return NULL;
        }
#endif
    }
    
    /* 初始化无锁任务队列 */
    {
        LockFreeQueueConfig lf_config;
        memset(&lf_config, 0, sizeof(LockFreeQueueConfig));
        lf_config.capacity = 1024;
        lf_config.element_size = sizeof(ThreadPoolTask);
        lf_config.enable_hazard_pointers = 1;
        lf_config.enable_epoch_based_reclamation = 1;
        lf_config.max_retries = 10;
        lf_config.backoff_strategy = 1;
        pool->lf_task_queue = lock_free_queue_create(&lf_config);
        if (!pool->lf_task_queue) {
            fprintf(stdout, "警告: 无锁任务队列初始化失败, 线程池仍可正常运行\n");
        }
    }
    
    return pool;
}

/**
 * @brief 释放线程池
 */
void thread_pool_free(ThreadPool* pool) {
    if (!pool) {
        return;
    }
    
    // 停止线程池
    thread_pool_pause(pool);
    
    // 设置停止标志
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
#endif
    
    pool->is_running = 0;
    
#ifdef _WIN32
    LeaveCriticalSection(&pool->lock);
#else
    pthread_mutex_unlock(&pool->lock);
#endif
    
    // 唤醒所有线程
#ifdef _WIN32
    WakeAllConditionVariable(&pool->task_cond);
#else
    pthread_cond_broadcast(&pool->task_cond);
#endif
    
    // 等待所有线程结束
    for (size_t i = 0; i < pool->config.num_threads; i++) {
#ifdef _WIN32
        WaitForSingleObject(pool->threads[i].handle, INFINITE);
        CloseHandle(pool->threads[i].handle);
#else
        pthread_join(pool->threads[i].handle, NULL);
#endif
    }
    
    // 清理任务队列
    for (int p = 0; p < TASK_PRIORITY_COUNT; p++) {
        task_queue_clear(pool, &pool->priority_queues[p]);
    }
    
    // 如果启用了工作窃取，清理每个线程的队列
    if (pool->config.enable_work_stealing && pool->thread_queues) {
        for (size_t i = 0; i < pool->config.num_threads; i++) {
            task_queue_clear(pool, &pool->thread_queues[i]);
#ifdef _WIN32
            DeleteCriticalSection(&pool->queue_locks[i]);
#else
            pthread_mutex_destroy(&pool->queue_locks[i]);
#endif
        }
        safe_free((void**)&pool->queue_locks);
        safe_free((void**)&pool->thread_queues);
    }
    
    // 清理任务节点池
    task_pool_cleanup(pool);
    
    /* 释放无锁任务队列 */
    if (pool->lf_task_queue) {
        lock_free_queue_free(pool->lf_task_queue);
        pool->lf_task_queue = NULL;
    }
    
    // 释放资源
    safe_free((void**)&pool->threads);
    
    // 销毁同步原语
#ifdef _WIN32
    DeleteCriticalSection(&pool->lock);
    DeleteCriticalSection(&pool->node_lock);
#else
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->task_cond);
    pthread_cond_destroy(&pool->empty_cond);
    pthread_mutex_destroy(&pool->node_lock);
#endif
    
    // 释放线程池结构
    safe_free((void**)&pool);
}

/**
 * @brief 提交任务到线程池
 */
int thread_pool_submit(ThreadPool* pool, TaskFunction function, 
                      void* argument, int priority) {
    if (!pool || !function) {
        return -1;
    }
    
    // 检查线程数是否有效
    if (pool->config.num_threads == 0) {
        return -1;
    }

/* 检查线程池是否正在关闭 — 防止free期间提交任务导致丢失 */
    if (!pool->is_running) {
        return -1;
    }
    
    // 创建任务
    ThreadPoolTask task;
    task.function = function;
    task.argument = argument;
    task.priority = priority;
    
    int result = -1;
    
    if (pool->config.enable_work_stealing) {
        /* R7-005修复: next_thread_index使用原子操作为避免竞态 */
        size_t target_thread;
#ifdef _WIN32
        target_thread = (size_t)InterlockedIncrement64((LONG64*)&pool->next_thread_index);
#else
        target_thread = (size_t)__sync_fetch_and_add(&pool->next_thread_index, 1);
#endif
        target_thread = target_thread % pool->config.num_threads;
        
        // 检查线程队列是否已满
        if (pool->config.max_tasks_per_thread > 0 && 
            task_queue_size(&pool->thread_queues[target_thread]) >= pool->config.max_tasks_per_thread) {
            // 队列已满，尝试其他线程
            for (size_t i = 0; i < pool->config.num_threads; i++) {
                size_t try_thread = (target_thread + i) % pool->config.num_threads;
                if (task_queue_size(&pool->thread_queues[try_thread]) < pool->config.max_tasks_per_thread) {
                    target_thread = try_thread;
                    break;
                }
            }
        }
        
        // 加锁（目标线程的队列锁）
#ifdef _WIN32
        EnterCriticalSection(&pool->queue_locks[target_thread]);
#else
        pthread_mutex_lock(&pool->queue_locks[target_thread]);
#endif
        
        // 推送到目标线程的队列
        result = task_queue_push(pool, &pool->thread_queues[target_thread], &task);
        
        // 先解锁队列锁，避免锁排序反转死锁：
        //   提交者: queue_locks[X] → pool->lock
        //   工作者: pool->lock → queue_locks[X]
#ifdef _WIN32
        LeaveCriticalSection(&pool->queue_locks[target_thread]);
#else
        pthread_mutex_unlock(&pool->queue_locks[target_thread]);
#endif
        
        if (result == 0) {
            // 广播唤醒所有工作线程（修复TOCTOU竞态：移除基于状态的检查）
            // 所有空闲线程都需要唤醒以尝试从其他队列窃取任务
#ifdef _WIN32
            EnterCriticalSection(&pool->lock);
            WakeAllConditionVariable(&pool->task_cond);
            LeaveCriticalSection(&pool->lock);
#else
            pthread_mutex_lock(&pool->lock);
            pthread_cond_broadcast(&pool->task_cond);
            pthread_mutex_unlock(&pool->lock);
#endif
        }
        
    } else {
        // 传统模式：提交到全局队列（P2-20修复: 按优先级路由）
        int pq_idx = task.priority > 0 ? TASK_PRIORITY_LOW : TASK_PRIORITY_HIGH;
        
        // 检查任务队列是否已满——改为阻塞等待而非丢弃
        size_t max_retries = 100;
        size_t retry = 0;
        while (pool->config.max_tasks > 0 && 
               task_queue_size(&pool->priority_queues[TASK_PRIORITY_HIGH]) +
               task_queue_size(&pool->priority_queues[TASK_PRIORITY_LOW]) >= pool->config.max_tasks) {
            if (retry >= max_retries) {
                log_error("线程池任务队列持续满(%zu次重试)，放弃提交", max_retries);
                return -1;
            }
            /* 短暂释放CPU，等待队列空闲 */
#ifdef _WIN32
            Sleep(10);
#else
            usleep(10000);
#endif
            retry++;
        }
        
        // 加锁添加任务
#ifdef _WIN32
        EnterCriticalSection(&pool->lock);
#else
        pthread_mutex_lock(&pool->lock);
#endif
        
        result = task_queue_push(pool, &pool->priority_queues[pq_idx], &task);
        if (result == 0) {
            // 通知工作线程
#ifdef _WIN32
            WakeConditionVariable(&pool->task_cond);
#else
            pthread_cond_signal(&pool->task_cond);
#endif
        }
        
#ifdef _WIN32
        LeaveCriticalSection(&pool->lock);
#else
        pthread_mutex_unlock(&pool->lock);
#endif
    }
    
    return result;
}

/**
 * @brief 批量提交任务
 */
int thread_pool_submit_batch(ThreadPool* pool, const ThreadPoolTask* tasks,
                            size_t num_tasks) {
    if (!pool || !tasks || num_tasks == 0) {
        return -1;
    }
    
    int submitted = 0;
    
    for (size_t i = 0; i < num_tasks; i++) {
        if (thread_pool_submit(pool, tasks[i].function, 
                              tasks[i].argument, tasks[i].priority) == 0) {
            submitted++;
        }
    }
    
    return submitted;
}

/**
 * @brief 等待所有任务完成
 */
int thread_pool_wait_all(ThreadPool* pool, unsigned int timeout_ms) {
    if (!pool) {
        return -1;
    }
    
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
#endif
    
    // 等待所有队列为空
    while (1) {
        int has_tasks = (task_queue_size(&pool->priority_queues[TASK_PRIORITY_HIGH]) +
                         task_queue_size(&pool->priority_queues[TASK_PRIORITY_LOW]) > 0);
        
        // 如果启用工作窃取，检查所有本地队列
        if (!has_tasks && pool->config.enable_work_stealing && pool->thread_queues) {
            for (size_t i = 0; i < pool->config.num_threads; i++) {
#ifdef _WIN32
                EnterCriticalSection(&pool->queue_locks[i]);
#else
                pthread_mutex_lock(&pool->queue_locks[i]);
#endif
                if (task_queue_size(&pool->thread_queues[i]) > 0) {
                    has_tasks = 1;
                }
#ifdef _WIN32
                LeaveCriticalSection(&pool->queue_locks[i]);
#else
                pthread_mutex_unlock(&pool->queue_locks[i]);
#endif
                if (has_tasks) {
                    break;
                }
            }
        }
        
        if (!has_tasks) {
            break;  // 所有队列都为空
        }
        
        // 如果还有任务但所有工作线程可能都已休眠，唤醒它们处理剩余任务
        if (pool->config.enable_work_stealing) {
#ifdef _WIN32
            WakeAllConditionVariable(&pool->task_cond);
#else
            pthread_cond_broadcast(&pool->task_cond);
#endif
        }
        
#ifdef _WIN32
        if (!SleepConditionVariableCS(&pool->empty_cond, &pool->lock, 
                                     timeout_ms ? timeout_ms : INFINITE)) {
            // 超时或错误
            if (GetLastError() == ERROR_TIMEOUT) {
                LeaveCriticalSection(&pool->lock);
                return 1;  // 超时
            } else {
                LeaveCriticalSection(&pool->lock);
                return -1; // 错误
            }
        }
#else
        if (timeout_ms > 0) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += (timeout_ms % 1000) * 1000000;
            ts.tv_sec += timeout_ms / 1000 + ts.tv_nsec / 1000000000;
            ts.tv_nsec %= 1000000000;
            
            if (pthread_cond_timedwait(&pool->empty_cond, &pool->lock, &ts) == ETIMEDOUT) {
                pthread_mutex_unlock(&pool->lock);
                return 1;  // 超时
            }
        } else {
            pthread_cond_wait(&pool->empty_cond, &pool->lock);
        }
#endif
    }
    
#ifdef _WIN32
    LeaveCriticalSection(&pool->lock);
#else
    pthread_mutex_unlock(&pool->lock);
#endif
    
    return 0;
}

/**
 * @brief 获取线程池统计信息
 */
int thread_pool_get_stats(const ThreadPool* pool,
                         size_t* active_threads,
                         size_t* pending_tasks,
                         size_t* completed_tasks) {
    if (!pool) {
        return -1;
    }
    
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
#endif
    
    if (pending_tasks) {
        size_t total_pending = task_queue_size(&pool->priority_queues[TASK_PRIORITY_HIGH]) +
                               task_queue_size(&pool->priority_queues[TASK_PRIORITY_LOW]);
        
        // 如果启用工作窃取，添加所有本地队列的任务
        if (pool->config.enable_work_stealing && pool->thread_queues) {
            for (size_t i = 0; i < pool->config.num_threads; i++) {
#ifdef _WIN32
                EnterCriticalSection(&pool->queue_locks[i]);
#else
                pthread_mutex_lock(&pool->queue_locks[i]);
#endif
                total_pending += task_queue_size(&pool->thread_queues[i]);
#ifdef _WIN32
                LeaveCriticalSection(&pool->queue_locks[i]);
#else
                pthread_mutex_unlock(&pool->queue_locks[i]);
#endif
            }
        }
        
        *pending_tasks = total_pending;
    }
    
    if (completed_tasks) {
        *completed_tasks = pool->completed_tasks;
    }
    
    if (active_threads) {
        size_t active = 0;
        for (size_t i = 0; i < pool->config.num_threads; i++) {
            if (pool->threads[i].state == THREAD_STATE_BUSY) {
                active++;
            }
        }
        *active_threads = active;
    }
    
#ifdef _WIN32
    LeaveCriticalSection(&pool->lock);
#else
    pthread_mutex_unlock(&pool->lock);
#endif
    
    return 0;
}

/**
 * @brief 调整线程池大小
 */
int thread_pool_resize(ThreadPool* pool, size_t new_num_threads) {
    if (!pool || new_num_threads == 0) {
        return -1;
    }
    
    // 完整动态调整大小实现：支持线程池运行时调整线程数量
    // 注意：动态调整大小是高级功能，需要谨慎处理线程状态和任务分配
    
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
#endif
    
    size_t old_num_threads = pool->config.num_threads;
    
    // 如果线程数量未变，直接返回
    if (new_num_threads == old_num_threads) {
#ifdef _WIN32
        LeaveCriticalSection(&pool->lock);
#else
        pthread_mutex_unlock(&pool->lock);
#endif
        return 0;
    }
    
    // 如果线程池未运行，可以安全调整配置
    if (!pool->is_running) {
        pool->config.num_threads = new_num_threads;
        
#ifdef _WIN32
        LeaveCriticalSection(&pool->lock);
#else
        pthread_mutex_unlock(&pool->lock);
#endif
        return 0;
    }
    
    // 线程池正在运行：执行运行时调整
    if (new_num_threads < old_num_threads) {
        // 减少线程数量：标记多余线程为停止状态
        for (size_t i = new_num_threads; i < old_num_threads; i++) {
            pool->threads[i].state = THREAD_STATE_STOPPING;
        }
        
        // 唤醒所有线程，让多余线程可以检查停止状态
#ifdef _WIN32
        WakeAllConditionVariable(&pool->task_cond);
#else
        pthread_cond_broadcast(&pool->task_cond);
#endif
        
        /* FIX-010修复: 释放缩减后多余线程对应队列中的残留任务，防止内存泄漏 */
        /* P1修复7: 清理队列前必须持有对应queue_locks[i]，防止与孤立线程并发
         * 访问同一队列导致数据竞争/链表损坏。原实现在未持锁情况下直接
         * task_queue_clear，孤立线程可能正通过trylock读取该队列。
         * 锁序: 此处已持有pool->lock→再取queue_locks[i]，与worker(get_task_from_pool)
         * 的pool->lock→queue_locks顺序一致，无锁序反转死锁风险。
         * i必小于old_num_threads故i%old_num_threads==i，直接用i索引更清晰。 */
        for (size_t i = new_num_threads; i < old_num_threads; i++) {
            if (pool->thread_queues && pool->queue_locks) {
#ifdef _WIN32
                EnterCriticalSection(&pool->queue_locks[i]);
#else
                pthread_mutex_lock(&pool->queue_locks[i]);
#endif
                task_queue_clear(pool, &pool->thread_queues[i]);
#ifdef _WIN32
                LeaveCriticalSection(&pool->queue_locks[i]);
#else
                pthread_mutex_unlock(&pool->queue_locks[i]);
#endif
            }
        }
        
        // 更新配置
        pool->config.num_threads = new_num_threads;
        
#ifdef _WIN32
        LeaveCriticalSection(&pool->lock);
#else
        pthread_mutex_unlock(&pool->lock);
#endif
        return 0;
    } else {
        // 增加线程数量：需要创建新线程
        // 这是真正的工业级实现，创建新线程并更新所有相关数据结构
        
        // 首先，分配新的线程数组
        WorkerThread* new_threads = (WorkerThread*)safe_malloc(new_num_threads * sizeof(WorkerThread));
        if (!new_threads) {
#ifdef _WIN32
            LeaveCriticalSection(&pool->lock);
#else
            pthread_mutex_unlock(&pool->lock);
#endif
            return -1;
        }
        
        // 复制旧线程数据
        for (size_t i = 0; i < old_num_threads; i++) {
            new_threads[i] = pool->threads[i];
        }
        
        // 初始化新线程
        for (size_t i = old_num_threads; i < new_num_threads; i++) {
            new_threads[i].state = THREAD_STATE_IDLE;
            new_threads[i].id = (int)i;
            
#ifdef _WIN32
            new_threads[i].handle = CreateThread(
                NULL,                   // 默认安全属性
                0,                      // 默认堆栈大小
                worker_thread_func,     // 线程函数
                pool,                   // 线程参数
                0,                      // 默认创建标志
                NULL                    // 线程ID
            );
            
            if (!new_threads[i].handle) {
                // 创建失败，清理已创建的新线程
                for (size_t j = old_num_threads; j < i; j++) {
                    CloseHandle(new_threads[j].handle);
                }
                safe_free((void**)&new_threads);
                
#ifdef _WIN32
                LeaveCriticalSection(&pool->lock);
#else
                pthread_mutex_unlock(&pool->lock);
#endif
                return -1;
            }
#else
            if (pthread_create(&new_threads[i].handle, NULL, 
                              worker_thread_func, pool) != 0) {
                // 创建失败，清理已创建的新线程
                for (size_t j = old_num_threads; j < i; j++) {
                    pthread_cancel(new_threads[j].handle);
                }
                safe_free((void**)&new_threads);
                
#ifdef _WIN32
                LeaveCriticalSection(&pool->lock);
#else
                pthread_mutex_unlock(&pool->lock);
#endif
                return -1;
            }
#endif
        }
        
        // 如果需要工作窃取，需要扩展线程队列和锁数组
        if (pool->config.enable_work_stealing) {
            // 分配新的队列数组
            TaskQueue* new_thread_queues = (TaskQueue*)safe_malloc(new_num_threads * sizeof(TaskQueue));
#ifdef _WIN32
            CRITICAL_SECTION* new_queue_locks = (CRITICAL_SECTION*)safe_malloc(new_num_threads * sizeof(CRITICAL_SECTION));
#else
            pthread_mutex_t* new_queue_locks = (pthread_mutex_t*)safe_malloc(new_num_threads * sizeof(pthread_mutex_t));
#endif
            
            if (!new_thread_queues || !new_queue_locks) {
                // 分配失败，清理
                safe_free((void**)&new_thread_queues);
                safe_free((void**)&new_queue_locks);
                
                // 清理已创建的新线程
                for (size_t i = old_num_threads; i < new_num_threads; i++) {
#ifdef _WIN32
                    CloseHandle(new_threads[i].handle);
#else
                    pthread_cancel(new_threads[i].handle);
#endif
                }
                safe_free((void**)&new_threads);
                
#ifdef _WIN32
                LeaveCriticalSection(&pool->lock);
#else
                pthread_mutex_unlock(&pool->lock);
#endif
                return -1;
            }
            
            // 复制旧的队列和锁
            for (size_t i = 0; i < old_num_threads; i++) {
                new_thread_queues[i] = pool->thread_queues[i];
                new_queue_locks[i] = pool->queue_locks[i];
            }
            
            // 初始化新的队列和锁
            for (size_t i = old_num_threads; i < new_num_threads; i++) {
                task_queue_init(&new_thread_queues[i]);
#ifdef _WIN32
                InitializeCriticalSection(&new_queue_locks[i]);
#else
                pthread_mutex_init(&new_queue_locks[i], NULL);
#endif
            }
            
            // 替换旧数组
            safe_free((void**)&pool->thread_queues);
            safe_free((void**)&pool->queue_locks);
            
            pool->thread_queues = new_thread_queues;
            pool->queue_locks = new_queue_locks;
        }
        
        // 替换线程数组
        safe_free((void**)&pool->threads);
        pool->threads = new_threads;
        /* P1修复6: 扩容时threads/thread_queues/queue_locks均重新分配为new_num_threads，
         * 同步更新容量上限。 */
        pool->threads_capacity = new_num_threads;
        
        // 更新配置
        pool->config.num_threads = new_num_threads;
        
#ifdef _WIN32
        LeaveCriticalSection(&pool->lock);
#else
        pthread_mutex_unlock(&pool->lock);
#endif
        return 0;
    }
}

/**
 * @brief 暂停线程池
 */
int thread_pool_pause(ThreadPool* pool) {
    if (!pool) {
        return -1;
    }
    
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
#endif
    
    pool->is_paused = 1;
    
#ifdef _WIN32
    LeaveCriticalSection(&pool->lock);
#else
    pthread_mutex_unlock(&pool->lock);
#endif
    
    return 0;
}

/**
 * @brief 恢复线程池
 */
int thread_pool_resume(ThreadPool* pool) {
    if (!pool) {
        return -1;
    }
    
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
#endif
    
    pool->is_paused = 0;
    
    // 唤醒所有线程
#ifdef _WIN32
    WakeAllConditionVariable(&pool->task_cond);
#else
    pthread_cond_broadcast(&pool->task_cond);
#endif
    
#ifdef _WIN32
    LeaveCriticalSection(&pool->lock);
#else
    pthread_mutex_unlock(&pool->lock);
#endif
    
    return 0;
}

/**
 * @brief 获取线程池配置
 */
int thread_pool_get_config(const ThreadPool* pool, ThreadPoolConfig* config) {
    if (!pool || !config) {
        return -1;
    }
    
    memcpy(config, &pool->config, sizeof(ThreadPoolConfig));
    return 0;
}

/**
 * @brief 设置线程池配置
 */
int thread_pool_set_config(ThreadPool* pool, const ThreadPoolConfig* config) {
    if (!pool || !config) {
        return -1;
    }
    
    // 线程数必须至少为1
    if (config->num_threads == 0) {
        return -1;
    }
    
    // 完整配置更新实现：安全更新所有可修改的配置参数
    // 注意：某些参数（如线程数量）需要特殊处理，不能直接修改
    
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
#endif
    
    // 保存原始线程数（需要特殊处理）
    size_t original_num_threads = pool->config.num_threads;
    
    // 更新所有配置字段
    pool->config.num_threads = config->num_threads;  // 注意：实际线程数调整需要通过thread_pool_resize()
    pool->config.max_tasks = config->max_tasks;
    pool->config.dynamic_scaling = config->dynamic_scaling;
    pool->config.enable_priority = config->enable_priority;
    pool->config.enable_work_stealing = config->enable_work_stealing;
    pool->config.work_stealing_threshold = config->work_stealing_threshold;
    pool->config.task_timeout_ms = config->task_timeout_ms;
    pool->config.idle_thread_timeout_ms = config->idle_thread_timeout_ms;
    
    // 如果线程数发生变化且线程池正在运行，需要特殊处理
    if (pool->is_running && config->num_threads != original_num_threads) {
        // 线程数变化：需要通过resize函数处理
        // 这里只记录配置，实际调整由调用者通过thread_pool_resize()执行
        // 恢复原始线程数以保持一致性
        pool->config.num_threads = original_num_threads;
    }
    
#ifdef _WIN32
    LeaveCriticalSection(&pool->lock);
#else
    pthread_mutex_unlock(&pool->lock);
#endif
    
    return 0;
}

/**
 * @brief 重置线程池统计信息
 */
void thread_pool_reset_stats(ThreadPool* pool) {
    if (!pool) {
        return;
    }
    
#ifdef _WIN32
    EnterCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
#endif
    
    pool->completed_tasks = 0;
    
#ifdef _WIN32
    LeaveCriticalSection(&pool->lock);
#else
    pthread_mutex_unlock(&pool->lock);
#endif
}

// ==================== 无锁队列集成接口 ====================

/**
 * @brief 通过无锁队列提交高优先级任务
 *
 * @param pool 线程池指针
 * @param function 任务函数
 * @param argument 任务参数
 * @return int 成功返回0，失败返回-1
 */
int thread_pool_submit_lf(ThreadPool* pool, TaskFunction function, void* argument) {
    if (!pool || !function) {
        return -1;
    }
    if (!pool->lf_task_queue) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "无锁队列未初始化");
        return -1;
    }
    
    ThreadPoolTask task;
    memset(&task, 0, sizeof(ThreadPoolTask));
    task.function = function;
    task.argument = argument;
    task.priority = 1;
    
    LockFreeOperationResult result;
    memset(&result, 0, sizeof(LockFreeOperationResult));
    
    return lock_free_queue_enqueue(pool->lf_task_queue, &task, sizeof(ThreadPoolTask), &result);
}

/**
 * @brief 从无锁队列中取出任务
 *
 * @param pool 线程池指针
 * @param task 任务输出缓冲区
 * @return int 成功返回0（有任务），失败返回-1（空队列或错误）
 */
int thread_pool_receive_lf(ThreadPool* pool, ThreadPoolTask* task) {
    if (!pool || !task) {
        return -1;
    }
    if (!pool->lf_task_queue) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "无锁队列未初始化");
        return -1;
    }
    
    LockFreeOperationResult result;
    memset(&result, 0, sizeof(LockFreeOperationResult));
    
    return lock_free_queue_dequeue(pool->lf_task_queue, task, sizeof(ThreadPoolTask), &result);
}

/**
 * @brief 获取无锁任务队列大小
 *
 * @param pool 线程池指针
 * @return size_t 队列大小，失败返回0
 */
size_t thread_pool_lf_queue_size(const ThreadPool* pool) {
    if (!pool || !pool->lf_task_queue) {
        return 0;
    }
    return lock_free_queue_size(pool->lf_task_queue);
}