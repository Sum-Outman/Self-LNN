/**
 * @file lock_free.h
 * @brief 无锁数据结构接口
 * 
 * 高性能无锁数据结构完整实现，包括：
 * 1. 无锁队列（Lock-free Queue）
 * 2. 无锁栈（Lock-free Stack）
 * 3. 无锁哈希表（Lock-free Hash Table）
 * 4. 无锁内存池（Lock-free Memory Pool）
 * 5. 无锁环形缓冲区（Lock-free Ring Buffer）
 * 6. 无锁优先队列（Lock-free Priority Queue）
 * 7. 无锁跳表（Lock-free Skip List）
 * 8. 无锁工作窃取队列（Lock-free Work Stealing Queue）
 * 
 *  ，提供完整的无锁算法实现。
 */

#ifndef SELFLNN_LOCK_FREE_H
#define SELFLNN_LOCK_FREE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 无锁数据结构算法类型
 */
typedef enum {
    LOCK_FREE_QUEUE = 0,           /**< 无锁队列 */
    LOCK_FREE_STACK = 1,           /**< 无锁栈 */
    LOCK_FREE_HASH_TABLE = 2,      /**< 无锁哈希表 */
    LOCK_FREE_MEMORY_POOL = 3,     /**< 无锁内存池 */
    LOCK_FREE_RING_BUFFER = 4,     /**< 无锁环形缓冲区 */
    LOCK_FREE_PRIORITY_QUEUE = 5,  /**< 无锁优先队列 */
    LOCK_FREE_SKIP_LIST = 6,       /**< 无锁跳表 */
    LOCK_FREE_WORK_STEALING = 7    /**< 无锁工作窃取队列 */
} LockFreeAlgorithm;

/**
 * @brief 无锁队列配置
 */
typedef struct {
    size_t capacity;               /**< 队列容量 */
    size_t element_size;           /**< 元素大小 */
    int enable_hazard_pointers;    /**< 是否启用危险指针 */
    int enable_epoch_based_reclamation; /**< 是否启用基于时代的回收 */
    int max_retries;               /**< 最大重试次数 */
    int backoff_strategy;          /**< 退避策略 */
} LockFreeQueueConfig;

/**
 * @brief 无锁栈配置
 */
typedef struct {
    size_t capacity;               /**< 栈容量 */
    size_t element_size;           /**< 元素大小 */
    int enable_elimination;        /**< 是否启用消除技术 */
    int elimination_array_size;    /**< 消除数组大小 */
    int max_retries;               /**< 最大重试次数 */
    int backoff_strategy;          /**< 退避策略 */
} LockFreeStackConfig;

/**
 * @brief 无锁哈希表配置
 */
typedef struct {
    size_t capacity;               /**< 哈希表容量 */
    size_t key_size;               /**< 键大小 */
    size_t value_size;             /**< 值大小 */
    int hash_function_type;        /**< 哈希函数类型 */
    int enable_resizing;           /**< 是否启用动态调整大小 */
    float load_factor_threshold;   /**< 负载因子阈值 */
    int max_probe_length;          /**< 最大探测长度 */
} LockFreeHashTableConfig;

/**
 * @brief 无锁内存池配置
 */
typedef struct {
    size_t block_size;             /**< 内存块大小 */
    size_t num_blocks;             /**< 内存块数量 */
    int enable_slab_allocator;     /**< 是否启用slab分配器 */
    int slab_size;                 /**< slab大小 */
    int enable_batch_allocation;   /**< 是否启用批量分配 */
    size_t batch_size;             /**< 批量大小 */
} LockFreeMemoryPoolConfig;

/**
 * @brief 无锁数据结构操作结果
 */
typedef struct {
    int success;                   /**< 操作是否成功 */
    int retries;                   /**< 重试次数 */
    size_t operation_time_ns;      /**< 操作时间（纳秒） */
    int backoff_count;             /**< 退避计数 */
    char error_message[128];       /**< 错误信息 */
} LockFreeOperationResult;

/**
 * @brief 无锁队列句柄
 */
typedef struct LockFreeQueue LockFreeQueue;

/**
 * @brief 无锁栈句柄
 */
typedef struct LockFreeStack LockFreeStack;

/**
 * @brief 无锁哈希表句柄
 */
typedef struct LockFreeHashTable LockFreeHashTable;

/**
 * @brief 无锁内存池句柄
 */
typedef struct LockFreeMemoryPool LockFreeMemoryPool;

/**
 * @brief 无锁环形缓冲区句柄
 */
typedef struct LockFreeRingBuffer LockFreeRingBuffer;

/**
 * @brief 无锁优先队列句柄
 */
typedef struct LockFreePriorityQueue LockFreePriorityQueue;

/**
 * @brief 无锁跳表句柄
 */
typedef struct LockFreeSkipList LockFreeSkipList;

/**
 * @brief 无锁工作窃取队列句柄
 */
typedef struct LockFreeWorkStealingQueue LockFreeWorkStealingQueue;

/**
 * @brief 创建无锁工作窃取队列
 * 
 * @param capacity 队列容量
 * @return LockFreeWorkStealingQueue* 队列句柄，失败返回NULL
 */
LockFreeWorkStealingQueue* lock_free_work_stealing_create(size_t capacity);

/**
 * @brief 从工作窃取队列窃取一个任务
 * 
 * @param queue 工作窃取队列句柄
 * @return void* 任务指针，队列为空返回NULL
 */
void* lock_free_work_stealing_steal(LockFreeWorkStealingQueue* queue);

/**
 * @brief 销毁工作窃取队列
 * 
 * @param queue 工作窃取队列句柄
 */
void lock_free_work_stealing_free(LockFreeWorkStealingQueue* queue);

/* ========== 无锁队列接口 ========== */

/**
 * @brief 创建无锁队列
 * 
 * @param config 队列配置
 * @return LockFreeQueue* 无锁队列句柄，失败返回NULL
 */
LockFreeQueue* lock_free_queue_create(const LockFreeQueueConfig* config);

/**
 * @brief 释放无锁队列
 * 
 * @param queue 无锁队列句柄
 */
void lock_free_queue_free(LockFreeQueue* queue);

/**
 * @brief 入队操作
 * 
 * @param queue 无锁队列句柄
 * @param element 元素指针
 * @param element_size 元素大小
 * @param result 操作结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int lock_free_queue_enqueue(LockFreeQueue* queue, const void* element, 
                           size_t element_size, LockFreeOperationResult* result);

/**
 * @brief 出队操作
 * 
 * @param queue 无锁队列句柄
 * @param element 元素输出缓冲区
 * @param element_size 元素大小
 * @param result 操作结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int lock_free_queue_dequeue(LockFreeQueue* queue, void* element, 
                           size_t element_size, LockFreeOperationResult* result);

/**
 * @brief 获取队列大小
 * 
 * @param queue 无锁队列句柄
 * @return size_t 队列当前大小
 */
size_t lock_free_queue_size(const LockFreeQueue* queue);

/**
 * @brief 检查队列是否为空
 * 
 * @param queue 无锁队列句柄
 * @return int 为空返回1，否则返回0
 */
int lock_free_queue_is_empty(const LockFreeQueue* queue);

/**
 * @brief 检查队列是否已满
 * 
 * @param queue 无锁队列句柄
 * @return int 已满返回1，否则返回0
 */
int lock_free_queue_is_full(const LockFreeQueue* queue);

/* ========== 无锁栈接口 ========== */

/**
 * @brief 创建无锁栈
 * 
 * @param config 栈配置
 * @return LockFreeStack* 无锁栈句柄，失败返回NULL
 */
LockFreeStack* lock_free_stack_create(const LockFreeStackConfig* config);

/**
 * @brief 释放无锁栈
 * 
 * @param stack 无锁栈句柄
 */
void lock_free_stack_free(LockFreeStack* stack);

/**
 * @brief 压栈操作
 * 
 * @param stack 无锁栈句柄
 * @param element 元素指针
 * @param element_size 元素大小
 * @param result 操作结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int lock_free_stack_push(LockFreeStack* stack, const void* element, 
                        size_t element_size, LockFreeOperationResult* result);

/**
 * @brief 弹栈操作
 * 
 * @param stack 无锁栈句柄
 * @param element 元素输出缓冲区
 * @param element_size 元素大小
 * @param result 操作结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int lock_free_stack_pop(LockFreeStack* stack, void* element, 
                       size_t element_size, LockFreeOperationResult* result);

/**
 * @brief 获取栈大小
 * 
 * @param stack 无锁栈句柄
 * @return size_t 栈当前大小
 */
size_t lock_free_stack_size(const LockFreeStack* stack);

/**
 * @brief 检查栈是否为空
 * 
 * @param stack 无锁栈句柄
 * @return int 为空返回1，否则返回0
 */
int lock_free_stack_is_empty(const LockFreeStack* stack);

/* ========== 无锁哈希表接口 ========== */

/**
 * @brief 创建无锁哈希表
 * 
 * @param config 哈希表配置
 * @return LockFreeHashTable* 无锁哈希表句柄，失败返回NULL
 */
LockFreeHashTable* lock_free_hash_table_create(const LockFreeHashTableConfig* config);

/**
 * @brief 释放无锁哈希表
 * 
 * @param table 无锁哈希表句柄
 */
void lock_free_hash_table_free(LockFreeHashTable* table);

/**
 * @brief 插入键值对
 * 
 * @param table 无锁哈希表句柄
 * @param key 键指针
 * @param key_size 键大小
 * @param value 值指针
 * @param value_size 值大小
 * @param result 操作结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int lock_free_hash_table_insert(LockFreeHashTable* table, const void* key, size_t key_size,
                               const void* value, size_t value_size, LockFreeOperationResult* result);

/**
 * @brief 查找键值对
 * 
 * @param table 无锁哈希表句柄
 * @param key 键指针
 * @param key_size 键大小
 * @param value 值输出缓冲区
 * @param value_size 值大小
 * @param result 操作结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int lock_free_hash_table_lookup(LockFreeHashTable* table, const void* key, size_t key_size,
                               void* value, size_t value_size, LockFreeOperationResult* result);

/**
 * @brief 删除键值对
 * 
 * @param table 无锁哈希表句柄
 * @param key 键指针
 * @param key_size 键大小
 * @param result 操作结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int lock_free_hash_table_delete(LockFreeHashTable* table, const void* key, size_t key_size,
                               LockFreeOperationResult* result);

/**
 * @brief 获取哈希表大小
 * 
 * @param table 无锁哈希表句柄
 * @return size_t 哈希表当前大小
 */
size_t lock_free_hash_table_size(const LockFreeHashTable* table);

/* ========== 无锁内存池接口 ========== */

/**
 * @brief 创建无锁内存池
 * 
 * @param config 内存池配置
 * @return LockFreeMemoryPool* 无锁内存池句柄，失败返回NULL
 */
LockFreeMemoryPool* lock_free_memory_pool_create(const LockFreeMemoryPoolConfig* config);

/**
 * @brief 释放无锁内存池
 * 
 * @param pool 无锁内存池句柄
 */
void lock_free_memory_pool_free(LockFreeMemoryPool* pool);

/**
 * @brief 分配内存块
 * 
 * @param pool 无锁内存池句柄
 * @param result 操作结果输出缓冲区
 * @return void* 分配的内存块指针，失败返回NULL
 */
void* lock_free_memory_pool_allocate(LockFreeMemoryPool* pool, LockFreeOperationResult* result);

/**
 * @brief 释放内存块
 * 
 * @param pool 无锁内存池句柄
 * @param block 内存块指针
 * @param result 操作结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int lock_free_memory_pool_deallocate(LockFreeMemoryPool* pool, void* block, 
                                    LockFreeOperationResult* result);

/**
 * @brief 获取可用内存块数量
 * 
 * @param pool 无锁内存池句柄
 * @return size_t 可用内存块数量
 */
size_t lock_free_memory_pool_available_blocks(const LockFreeMemoryPool* pool);

/**
 * @brief 获取总内存块数量
 * 
 * @param pool 无锁内存池句柄
 * @return size_t 总内存块数量
 */
size_t lock_free_memory_pool_total_blocks(const LockFreeMemoryPool* pool);

/* ========== 无锁环形缓冲区接口 ========== */

/**
 * @brief 无锁环形缓冲区配置
 */
typedef struct {
    size_t capacity;               /**< 缓冲区容量（必须是2的幂） */
    size_t element_size;           /**< 元素大小 */
    int is_multi_producer;         /**< 是否支持多生产者 */
    int is_multi_consumer;         /**< 是否支持多消费者 */
    int enable_batch_operation;    /**< 是否启用批量操作 */
    int spin_count;                /**< 自旋等待次数 */
} LockFreeRingBufferConfig;

/**
 * @brief 创建无锁环形缓冲区
 * @param config 配置
 * @return LockFreeRingBuffer* 成功返回句柄，失败返回NULL
 */
LockFreeRingBuffer* lock_free_ring_buffer_create(const LockFreeRingBufferConfig* config);

/**
 * @brief 销毁无锁环形缓冲区
 * @param rb 句柄
 */
void lock_free_ring_buffer_free(LockFreeRingBuffer* rb);

/**
 * @brief 写入一个元素（生产者调用）
 * @param rb 句柄
 * @param element 元素指针
 * @return int 成功返回0，满返回1，失败返回-1
 */
int lock_free_ring_buffer_write(LockFreeRingBuffer* rb, const void* element);

/**
 * @brief 读取一个元素（消费者调用）
 * @param rb 句柄
 * @param element 输出缓冲区
 * @return int 成功返回0，空返回1，失败返回-1
 */
int lock_free_ring_buffer_read(LockFreeRingBuffer* rb, void* element);

/**
 * @brief 尝试写入（非阻塞）
 * @param rb 句柄
 * @param element 元素指针
 * @return int 成功返回0，满返回1，失败返回-1
 */
int lock_free_ring_buffer_try_write(LockFreeRingBuffer* rb, const void* element);

/**
 * @brief 尝试读取（非阻塞）
 * @param rb 句柄
 * @param element 输出缓冲区
 * @return int 成功返回0，空返回1，失败返回-1
 */
int lock_free_ring_buffer_try_read(LockFreeRingBuffer* rb, void* element);

/**
 * @brief 批量写入
 * @param rb 句柄
 * @param elements 元素数组
 * @param count 数量
 * @return size_t 成功写入数量
 */
size_t lock_free_ring_buffer_write_batch(LockFreeRingBuffer* rb, const void* elements, size_t count);

/**
 * @brief 批量读取
 * @param rb 句柄
 * @param elements 输出缓冲区
 * @param max_count 最大读取数量
 * @return size_t 实际读取数量
 */
size_t lock_free_ring_buffer_read_batch(LockFreeRingBuffer* rb, void* elements, size_t max_count);

/**
 * @brief 获取缓冲区使用量
 * @param rb 句柄
 * @return size_t 缓冲区当前元素数
 */
size_t lock_free_ring_buffer_size(const LockFreeRingBuffer* rb);

/**
 * @brief 获取缓冲区容量
 * @param rb 句柄
 * @return size_t 缓冲区容量
 */
size_t lock_free_ring_buffer_capacity(const LockFreeRingBuffer* rb);

/**
 * @brief 检查缓冲区是否为空
 * @param rb 句柄
 * @return int 空返回1，非空返回0
 */
int lock_free_ring_buffer_is_empty(const LockFreeRingBuffer* rb);

/**
 * @brief 检查缓冲区是否已满
 * @param rb 句柄
 * @return int 满返回1，未满返回0
 */
int lock_free_ring_buffer_is_full(const LockFreeRingBuffer* rb);

/**
 * @brief 清空缓冲区
 * @param rb 句柄
 */
void lock_free_ring_buffer_clear(LockFreeRingBuffer* rb);

/* ========== 无锁优先队列接口 ========== */

/**
 * @brief 无锁优先队列配置
 */
typedef struct {
    size_t capacity;               /**< 队列容量 */
    size_t element_size;           /**< 元素大小 */
    int is_min_heap;               /**< 1为最小堆(升序)，0为最大堆(降序) */
    int enable_batch_operation;    /**< 是否启用批量操作 */
    int max_retries;               /**< 最大CAS重试次数 */
} LockFreePriorityQueueConfig;

/**
 * @brief 创建无锁优先队列
 * @param config 配置
 * @return LockFreePriorityQueue* 成功返回句柄，失败返回NULL
 */
LockFreePriorityQueue* lock_free_priority_queue_create(const LockFreePriorityQueueConfig* config);

/**
 * @brief 销毁无锁优先队列
 * @param pq 句柄
 */
void lock_free_priority_queue_free(LockFreePriorityQueue* pq);

/**
 * @brief 插入元素
 * @param pq 句柄
 * @param element 元素指针
 * @param priority 优先级（值越小优先级越高，最小堆模式）
 * @return int 成功返回0，失败返回-1
 */
int lock_free_priority_queue_push(LockFreePriorityQueue* pq, const void* element, int priority);

/**
 * @brief 取出最高优先级元素
 * @param pq 句柄
 * @param element 输出缓冲区
 * @return int 成功返回0，空返回1，失败返回-1
 */
int lock_free_priority_queue_pop(LockFreePriorityQueue* pq, void* element);

/**
 * @brief 查看最高优先级元素（不移除）
 * @param pq 句柄
 * @param element 输出缓冲区
 * @return int 成功返回0，空返回1，失败返回-1
 */
int lock_free_priority_queue_peek(const LockFreePriorityQueue* pq, void* element);

/**
 * @brief 获取队列大小
 * @param pq 句柄
 * @return size_t 队列大小
 */
size_t lock_free_priority_queue_size(const LockFreePriorityQueue* pq);

/**
 * @brief 检查队列是否为空
 * @param pq 句柄
 * @return int 空返回1，非空返回0
 */
int lock_free_priority_queue_is_empty(const LockFreePriorityQueue* pq);

/* ========== 无锁跳表接口 ========== */

/**
 * @brief 无锁跳表配置
 */
typedef struct {
    size_t max_level;              /**< 最大层数（建议12-32） */
    size_t element_size;           /**< 元素大小 */
    int probability;               /**< 层数概率分母（4=1/4概率升层） */
    int enable_dynamic_level;      /**< 是否动态调整层数 */
    int max_retries;               /**< 最大CAS重试次数 */
} LockFreeSkipListConfig;

/**
 * @brief 无锁跳表句柄
 */

/**
 * @brief 创建无锁跳表
 * @param config 配置
 * @return LockFreeSkipList* 成功返回句柄，失败返回NULL
 */
LockFreeSkipList* lock_free_skip_list_create(const LockFreeSkipListConfig* config);

/**
 * @brief 销毁无锁跳表
 * @param sl 句柄
 */
void lock_free_skip_list_free(LockFreeSkipList* sl);

/**
 * @brief 插入键值对
 * @param sl 句柄
 * @param key 键
 * @param key_size 键大小
 * @param value 值指针
 * @return int 成功返回0，已存在返回1，失败返回-1
 */
int lock_free_skip_list_insert(LockFreeSkipList* sl, const void* key, size_t key_size, void* value);

/**
 * @brief 查找键对应的值
 * @param sl 句柄
 * @param key 键
 * @param key_size 键大小
 * @return void* 值指针，未找到返回NULL
 */
void* lock_free_skip_list_find(const LockFreeSkipList* sl, const void* key, size_t key_size);

/**
 * @brief 删除键值对
 * @param sl 句柄
 * @param key 键
 * @param key_size 键大小
 * @return int 成功返回0，未找到返回1，失败返回-1
 */
int lock_free_skip_list_erase(LockFreeSkipList* sl, const void* key, size_t key_size);

/**
 * @brief 获取跳表大小
 * @param sl 句柄
 * @return size_t 元素数量
 */
size_t lock_free_skip_list_size(const LockFreeSkipList* sl);

/**
 * @brief 检查跳表是否为空
 * @param sl 句柄
 * @return int 空返回1，非空返回0
 */
int lock_free_skip_list_is_empty(const LockFreeSkipList* sl);

/**
 * @brief 清空跳表
 * @param sl 句柄
 */
void lock_free_skip_list_clear(LockFreeSkipList* sl);

/* ========== 通用无锁操作 ========== */

/**
 * @brief 初始化无锁操作结果
 * 
 * @param result 操作结果结构
 */
void lock_free_operation_result_init(LockFreeOperationResult* result);

/**
 * @brief 获取无锁算法类型名称
 * 
 * @param algorithm 算法类型
 * @return const char* 算法类型名称
 */
const char* lock_free_algorithm_name(LockFreeAlgorithm algorithm);

/**
 * @brief 设置内存屏障（Memory Barrier）
 */
void lock_free_memory_barrier(void);

/**
 * @brief 比较并交换（Compare-and-Swap）操作
 * 
 * @param ptr 指针地址
 * @param expected 期望值
 * @param desired 期望设置的值
 * @return int 成功返回1，失败返回0
 */
int lock_free_compare_and_swap(void* ptr, void* expected, void* desired);

/**
 * @brief 获取并增加（Fetch-and-Add）操作
 * 
 * @param ptr 指针地址
 * @param value 增加的值
 * @return int64_t 增加前的值
 */
int64_t lock_free_fetch_and_add(void* ptr, int64_t value);

/**
 * @brief 获取并或（Fetch-and-Or）操作
 * 
 * @param ptr 指针地址
 * @param value 或的值
 * @return int64_t 操作前的值
 */
int64_t lock_free_fetch_and_or(void* ptr, int64_t value);

/* ============================
 * 增强T6：高并发线程池+工作窃取队列API
 * ============================ */

/**
 * @brief 无锁线程池配置结构
 */
typedef struct {
    size_t min_threads;                /**< 最小线程数 */
    size_t max_threads;                /**< 最大线程数 */
    size_t idle_timeout_ms;            /**< 空闲线程超时（毫秒） */
    size_t queue_capacity;             /**< 任务队列容量 */
    int enable_work_stealing;          /**< 启用工作窃取标志 */
    int enable_dynamic_resize;         /**< 启用动态调整线程数 */
    int thread_priority;               /**< 线程优先级（0=默认） */
    size_t stack_size;                 /**< 线程栈大小（0=默认） */
} LockFreeThreadPoolConfig;

/**
 * @brief 默认无锁线程池配置
 * 
 * 使用系统CPU核心数作为最大线程数，启用工作窃取和动态调整。
 */
#define LOCK_FREE_THREAD_POOL_CONFIG_DEFAULT { \
    .min_threads = 2, \
    .max_threads = 0, /* 将在创建时设为 CPU核心数*2 */ \
    .idle_timeout_ms = 60000, \
    .queue_capacity = 4096, \
    .enable_work_stealing = 1, \
    .enable_dynamic_resize = 1, \
    .thread_priority = 0, \
    .stack_size = 0 \
}

/**
 * @brief 任务优先级枚举
 */
typedef enum {
    TASK_PRIORITY_LOW = 0,
    TASK_PRIORITY_NORMAL = 1,
    TASK_PRIORITY_HIGH = 2,
    TASK_PRIORITY_CRITICAL = 3
} TaskPriority;

/**
 * @brief 任务函数类型
 */
typedef void (*ThreadPoolTaskFunc)(void* arg);

/**
 * @brief 线程池（不透明类型）
 */
typedef struct LockFreeThreadPool LockFreeThreadPool;

/**
 * @brief 唤醒回调类型
 * @param pool 线程池指针
 * @param user_data 用户数据
 */
typedef void (*ThreadPoolWakeFunc)(struct LockFreeThreadPool* pool, void* user_data);

/**
 * @brief 创建线程池
 * 
 * @param config 线程池配置（若为NULL则使用默认配置）
 * @return LockFreeThreadPool* 成功返回线程池句柄，失败返回NULL
 */
LockFreeThreadPool* lock_free_thread_pool_create(const LockFreeThreadPoolConfig* config);

/**
 * @brief 销毁线程池
 * 
 * 等待所有进行中的任务完成，然后销毁所有资源。
 * 
 * @param pool 线程池句柄
 */
void lock_free_thread_pool_free(LockFreeThreadPool* pool);

/**
 * @brief 提交任务到线程池
 * 
 * @param pool 线程池句柄
 * @param func 任务函数
 * @param arg 任务参数
 * @param priority 任务优先级
 * @return int64_t 成功返回任务ID，失败返回-1
 */
int64_t lock_free_thread_pool_submit(LockFreeThreadPool* pool,
                                       ThreadPoolTaskFunc func,
                                       void* arg,
                                       TaskPriority priority);

/**
 * @brief 等待指定任务完成
 * 
 * @param pool 线程池句柄
 * @param task_id 任务ID
 * @return int 成功返回0，失败返回-1
 */
int lock_free_thread_pool_wait(LockFreeThreadPool* pool, int64_t task_id);

/**
 * @brief 等待所有任务完成
 * 
 * @param pool 线程池句柄
 * @return int 成功返回0，失败返回-1
 */
int lock_free_thread_pool_wait_all(LockFreeThreadPool* pool);

/**
 * @brief 获取线程池当前统计信息
 * 
 * @param pool 线程池句柄
 * @param active_threads 输出活跃线程数
 * @param idle_threads 输出空闲线程数
 * @param pending_tasks 输出待处理任务数
 * @param completed_tasks 输出已完成任务数
 * @return int 成功返回0，失败返回-1
 */
int lock_free_thread_pool_get_stats(LockFreeThreadPool* pool,
                                      size_t* active_threads,
                                      size_t* idle_threads,
                                      size_t* pending_tasks,
                                      size_t* completed_tasks);

/**
 * @brief 动态调整线程池大小
 * 
 * @param pool 线程池句柄
 * @param min_threads 新的最小线程数
 * @param max_threads 新的最大线程数
 * @return int 成功返回0，失败返回-1
 */
int lock_free_thread_pool_resize(LockFreeThreadPool* pool,
                                   size_t min_threads,
                                   size_t max_threads);

/**
 * @brief 设置线程池唤醒回调
 * 
 * 当有新任务提交且有线程处于空闲等待状态时调用。
 * 
 * @param pool 线程池句柄
 * @param wake_func 唤醒函数（可设为NULL取消）
 * @param user_data 用户数据
 * @return int 成功返回0，失败返回-1
 */
int lock_free_thread_pool_set_wake_callback(LockFreeThreadPool* pool,
                                              ThreadPoolWakeFunc wake_func,
                                              void* user_data);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LOCK_FREE_H */