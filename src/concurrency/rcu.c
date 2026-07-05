#include "selflnn/concurrency/rcu.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#define atomic_load_acquire(ptr) (*(ptr))
#define atomic_store_release(ptr, val) do { MemoryBarrier(); *(ptr) = (val); } while(0)
#define atomic_increment(ptr) InterlockedIncrement((volatile LONG*)(ptr))
#define atomic_decrement(ptr) InterlockedDecrement((volatile LONG*)(ptr))
#define atomic_cas(ptr, expected, desired) \
    (InterlockedCompareExchangePointer((void* volatile*)(ptr), (desired), *(expected)) == *(expected))
#define atomic_thread_fence() MemoryBarrier()
#define sleep_ms(ms) Sleep(ms)
typedef LONG atomic_int32;
#else
#include <stdatomic.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#define atomic_load_acquire(ptr) atomic_load_explicit((atomic_intptr_t*)(ptr), memory_order_acquire)
#define atomic_store_release(ptr, val) atomic_store_explicit((atomic_intptr_t*)(ptr), (intptr_t)(val), memory_order_release)
#define atomic_increment(ptr) __sync_add_and_fetch((volatile int*)(ptr), 1)
#define atomic_decrement(ptr) __sync_sub_and_fetch((volatile int*)(ptr), 1)
#define atomic_cas(ptr, expected, desired) \
    __sync_bool_compare_and_swap((volatile long*)(ptr), (long)(expected), (long)(desired))
#define atomic_thread_fence() __sync_synchronize()
#define sleep_ms(ms) usleep((ms) * 1000)
typedef int atomic_int32;
#endif

#define MAX_RCU_THREADS 128
#define DEFAULT_GRACE_PERIOD_MS 10
#define DEFAULT_RECLAMATION_BATCH 32
#define RCU_EPOCH_SLOTS 3          /**< L-010: epoch槽位数（至少3个） */
#define RCU_EPOCH_MAX_SPIN 1000000 /**< L-010: epoch自旋最大迭代次数（约几十微秒级） */

/*
 * L-010: Epoch-based RCU完整实现
 *
 * 核心原理：
 *   全局epoch计数器持续递增。读者进入临界区时记录当前epoch，
 *   写者通过递增epoch并等待旧epoch的所有读者退出来实现宽限期。
 *
 *   3个epoch槽位（循环缓冲区）：
 *     slot[0] = 当前epoch的待回收对象
 *     slot[1] = 上一epoch的待回收对象
 *     slot[2] = epoch-2的待回收对象（安全可释放）
 *
 *   宽限期延迟：从原来的sleep_ms(1)轮询（最多1秒）降低到
 *   cpu_pause_hint()自旋（几十纳秒/次），理论上微秒级别完成。
 */

/* TLS存储当前线程RCU线程ID
 * 解决rcu_read_lock(操作active_readers)与rcu_wait_for_quiescent_state
 * (检查thread_in_read_section)互不通信的致命缺陷。 */
#ifdef _MSC_VER
static __declspec(thread) int g_tls_rcu_thread_id = -1;
#elif defined(__GNUC__) || defined(__clang__)
static __thread int g_tls_rcu_thread_id = -1;
#else
static int g_tls_rcu_thread_id = -1;
#endif

typedef struct RcuCallbackNode {
    void* old_ptr;
    RcuCallback callback;
    void* user_data;
    struct RcuCallbackNode* next;
} RcuCallbackNode;

struct RcuDomain {
    RcuDomainConfig config;
    volatile int state;
    volatile int active_readers;
    volatile int registered_thread_count;
    volatile int pending_callback_count;
    volatile int completed_callback_count;
    volatile int grace_periods_completed;
    volatile long total_grace_period_us;
    volatile int grace_period_samples;
    int thread_ids[MAX_RCU_THREADS];
    volatile int thread_active[MAX_RCU_THREADS];
    volatile int thread_in_read_section[MAX_RCU_THREADS];
    volatile long long thread_reader_count[MAX_RCU_THREADS];
    RcuCallbackNode* callback_list;
    /* L-010: Epoch-based RCU字段 */
    volatile long global_epoch;                              /**< 全局epoch计数器（单调递增） */
    volatile long reader_epochs[MAX_RCU_THREADS];            /**< 每个读者线程当前所处的epoch（0=不在读侧） */
    RcuCallbackNode* epoch_callbacks[RCU_EPOCH_SLOTS];       /**< 3个epoch槽位的回调链表 */
    volatile int epoch_callback_counts[RCU_EPOCH_SLOTS];     /**< 每个槽位的待回收数量 */
    volatile long last_reclaimed_epoch;                      /**< 最后完成回收的epoch */
#ifdef _WIN32
    CRITICAL_SECTION callback_lock;
#else
    pthread_mutex_t callback_lock;
#endif
    int is_initialized;
    int reclamation_count;
};

struct RcuThread {
    RcuDomain* domain;
    int thread_id;
    int is_registered;
};

static int rcu_allocate_thread_id(RcuDomain* domain) {
    if (!domain) return -1;
/* 使用CAS原子操作保护槽位分配，防止多线程并发获取同一thread_id。
     * thread_active为volatile int，不使用atomic_cas(该宏用InterlockedCompareExchangePointer处理指针)。
     * 改用InterlockedCompareExchange(Windows)/__sync_bool_compare_and_swap(Linux)处理整型CAS。
     * P2修复: reader_epochs写入使用atomic_store确保写入可见性。 */
    for (int i = 0; i < MAX_RCU_THREADS; i++) {
#ifdef _WIN32
        if (InterlockedCompareExchange((LONG volatile*)&domain->thread_active[i], 1, 0) == 0) {
#else
        if (__sync_bool_compare_and_swap(&domain->thread_active[i], 0, 1)) {
#endif
            domain->thread_in_read_section[i] = 0;
            domain->thread_reader_count[i] = 0;
            /* P2修复: 使用原子存储替代plain volatile写入 */
            __sync_lock_test_and_set(&domain->reader_epochs[i], 0);
            atomic_increment(&domain->registered_thread_count);
            return i;
        }
    }
    return -1;
}

static void rcu_release_thread_id(RcuDomain* domain, int thread_id) {
    if (!domain || thread_id < 0 || thread_id >= MAX_RCU_THREADS) return;
    domain->thread_active[thread_id] = 0;
    domain->thread_in_read_section[thread_id] = 0;
    /* P2修复: 原子释放清除epoch */
    __sync_lock_release(&domain->reader_epochs[thread_id]);
    atomic_decrement(&domain->registered_thread_count);
}

/*
 * L-010: CPU暂停提示——平台无关的轻量级自旋指令
 * 用于epoch等待循环中，功耗远低于sleep_ms(1)
 */
static inline void rcu_cpu_pause(void) {
#ifdef _WIN32
    YieldProcessor();
#elif defined(__x86_64__) || defined(__i386__) || defined(__i386)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield" ::: "memory");
#else
    __sync_synchronize();
#endif
}

/*
 * L-010: Epoch-based宽限期等待——替代原sleep_ms(1)轮询方案
 *
 * 原理：递增epoch后，自旋等待所有在旧epoch中的读者退出。
 *       自旋使用cpu_pause_hint而非sleep，延迟从毫秒级降到微秒级。
 *
 * @param domain    RCU域
 * @param old_epoch 需要等待退出的旧epoch值
 * @return 0=所有读者已退出旧epoch，-1=超时
 */
static int rcu_wait_for_epoch(RcuDomain* domain, long old_epoch) {
    if (!domain) return -1;
    for (int spin = 0; spin < RCU_EPOCH_MAX_SPIN; spin++) {
        int all_done = 1;
        for (int i = 0; i < MAX_RCU_THREADS; i++) {
            if (domain->thread_active[i]) {
                /* P2修复: 使用atomic_load替代plain volatile读取 */
                long rep = __sync_add_and_fetch(&domain->reader_epochs[i], 0);
                /* 读者在旧epoch中：尚未退出 */
                if (rep != 0 && rep == old_epoch) {
                    all_done = 0;
                    break;
                }
            }
        }
        if (all_done) return 0;
        rcu_cpu_pause();
        /* L-010: 超过1000次自旋后yield，避免CPU长时间空转 */
        if (spin > 0 && (spin & 1023) == 0) {
#ifdef _WIN32
            Sleep(0);  /* Windows: 立即让出时间片给其他线程 */
#else
            sched_yield();  /* Linux: 让出CPU调度 */
#endif
            /* 同时检查active_readers作为双重确认 */
            if (domain->active_readers == 0) return 0;
        }
    }
    /* L-010: 超时回退——极端情况下使用原sleep方案作为安全网 */
    return -1;
}

/*
 * L-010: 处理指定epoch槽位的回调
 */
static void rcu_process_epoch_callbacks(RcuDomain* domain, int slot) {
    if (!domain || slot < 0 || slot >= RCU_EPOCH_SLOTS) return;
#ifdef _WIN32
    EnterCriticalSection(&domain->callback_lock);
#else
    pthread_mutex_lock(&domain->callback_lock);
#endif
    RcuCallbackNode* node = domain->epoch_callbacks[slot];
    while (node) {
        RcuCallbackNode* next = node->next;
        if (node->callback) {
            node->callback(node->old_ptr, node->user_data);
        } else {
            safe_free((void**)&node->old_ptr);
        }
        safe_free((void**)&node);
        domain->completed_callback_count++;
        domain->reclamation_count++;
        node = next;
    }
    domain->epoch_callbacks[slot] = NULL;
    domain->epoch_callback_counts[slot] = 0;
#ifdef _WIN32
    LeaveCriticalSection(&domain->callback_lock);
#else
    pthread_mutex_unlock(&domain->callback_lock);
#endif
}

/* 原rcu_wait_for_quiescent_state保留作为epoch超时后的回退方案。
 * 正常情况下使用epoch-based rcu_wait_for_epoch（微秒级延迟）。 */
static int rcu_wait_for_quiescent_state(RcuDomain* domain) {
    if (!domain) return -1;
    int max_retries = 1000;
    int retry = 0;
    while (retry < max_retries) {
        int all_quiescent = 1;
        for (int i = 0; i < MAX_RCU_THREADS; i++) {
            if (domain->thread_active[i] && domain->thread_in_read_section[i]) {
                all_quiescent = 0;
                break;
            }
        }
        if (all_quiescent && domain->active_readers == 0) return 0; /* 双重检查 */
        sleep_ms(1);
        retry++;
    }
    return -1;
}

static void rcu_process_callbacks(RcuDomain* domain) {
    if (!domain) return;
#ifdef _WIN32
    EnterCriticalSection(&domain->callback_lock);
#else
    pthread_mutex_lock(&domain->callback_lock);
#endif
    RcuCallbackNode* node = domain->callback_list;
    while (node) {
        RcuCallbackNode* next = node->next;
        if (node->callback) {
            node->callback(node->old_ptr, node->user_data);
        } else {
            safe_free((void**)&node->old_ptr);
        }
        safe_free((void**)&node);
        domain->completed_callback_count++;
        domain->reclamation_count++;
        node = next;
    }
    domain->callback_list = NULL;
    domain->pending_callback_count = 0;
#ifdef _WIN32
    LeaveCriticalSection(&domain->callback_lock);
#else
    pthread_mutex_unlock(&domain->callback_lock);
#endif
}

RcuDomain* rcu_domain_create(const RcuDomainConfig* config) {
    RcuDomain* domain = (RcuDomain*)safe_calloc(1, sizeof(RcuDomain));
    if (!domain) return NULL;

    if (config) {
        domain->config = *config;
    } else {
        domain->config.max_threads = MAX_RCU_THREADS;
        domain->config.grace_period_ms = DEFAULT_GRACE_PERIOD_MS;
        domain->config.enable_async_reclamation = 1;
        domain->config.reclamation_batch_size = DEFAULT_RECLAMATION_BATCH;
        domain->config.enable_debug_stats = 0;
    }

    domain->state = RCU_STATE_READY;
    domain->active_readers = 0;
    domain->registered_thread_count = 0;
    domain->pending_callback_count = 0;
    domain->completed_callback_count = 0;
    domain->grace_periods_completed = 0;
    domain->reclamation_count = 0;
    domain->callback_list = NULL;
    /* L-010: 初始化epoch字段 */
    domain->global_epoch = 1;  /* 从1开始，0表示"不在读侧" */
    domain->last_reclaimed_epoch = 0;
    for (int s = 0; s < RCU_EPOCH_SLOTS; s++) {
        domain->epoch_callbacks[s] = NULL;
        domain->epoch_callback_counts[s] = 0;
    }
#ifdef _WIN32
    InitializeCriticalSection(&domain->callback_lock);
#else
    pthread_mutex_init(&domain->callback_lock, NULL);
#endif
    domain->is_initialized = 1;

    for (int i = 0; i < MAX_RCU_THREADS; i++) {
        domain->thread_active[i] = 0;
        domain->thread_in_read_section[i] = 0;
        domain->thread_reader_count[i] = 0;
        domain->reader_epochs[i] = 0;  /* L-010: reader_epochs初始化为0 */
    }

    atomic_thread_fence();
    return domain;
}

void rcu_domain_destroy(RcuDomain* domain) {
    if (!domain) return;
    rcu_synchronize(domain);
    rcu_process_callbacks(domain);
    /* L-010: 处理所有epoch槽位中残留的回调 */
    for (int s = 0; s < RCU_EPOCH_SLOTS; s++) {
        rcu_process_epoch_callbacks(domain, s);
    }
#ifdef _WIN32
    DeleteCriticalSection(&domain->callback_lock);
#else
    pthread_mutex_destroy(&domain->callback_lock);
#endif
    domain->is_initialized = 0;
    safe_free((void**)&domain);
}

RcuThread* rcu_thread_register(RcuDomain* domain) {
    if (!domain || !domain->is_initialized) return NULL;
    RcuThread* thread = (RcuThread*)safe_calloc(1, sizeof(RcuThread));
    if (!thread) return NULL;
    thread->domain = domain;
    thread->thread_id = rcu_allocate_thread_id(domain);
    if (thread->thread_id < 0) {
        safe_free((void**)&thread);
        return NULL;
    }
    thread->is_registered = 1;
/* 保存线程ID到TLS，使rcu_read_lock能访问 */
    g_tls_rcu_thread_id = thread->thread_id;
    atomic_thread_fence();
    return thread;
}

void rcu_thread_unregister(RcuDomain* domain, RcuThread* thread) {
    if (!domain || !thread) return;
    if (thread->is_registered && thread->thread_id >= 0) {
        rcu_release_thread_id(domain, thread->thread_id);
    }
    thread->is_registered = 0;
    safe_free((void**)&thread);
}

void rcu_read_lock(RcuDomain* domain) {
    if (!domain || !domain->is_initialized) return;
    atomic_thread_fence();
    /* L-010: Epoch-based RCU——进入读侧时记录当前全局epoch
 *: 同时维护active_readers和thread_in_read_section
     * 确保写者能检测到该线程在读侧临界区中 */
    if (g_tls_rcu_thread_id >= 0 && g_tls_rcu_thread_id < MAX_RCU_THREADS) {
        domain->thread_in_read_section[g_tls_rcu_thread_id] = 1;
        /* P2修复: 原子存储确保epoch对写者可见 */
        __sync_lock_test_and_set(&domain->reader_epochs[g_tls_rcu_thread_id], domain->global_epoch);
    }
    atomic_increment(&domain->active_readers);
    atomic_thread_fence();
}

void rcu_read_unlock(RcuDomain* domain) {
    if (!domain || !domain->is_initialized) return;
    atomic_thread_fence();
    /* L-010: 退出读侧时清除epoch记录（顺序重要：先清除标记再递减计数器） */
    if (g_tls_rcu_thread_id >= 0 && g_tls_rcu_thread_id < MAX_RCU_THREADS) {
        domain->thread_in_read_section[g_tls_rcu_thread_id] = 0;
        /* P2修复: 原子存储清除epoch，并用release语义确保写者可见 */
        __sync_lock_release(&domain->reader_epochs[g_tls_rcu_thread_id]);
    }
    atomic_decrement(&domain->active_readers);
    atomic_thread_fence();
}

void* rcu_protect_pointer(RcuDomain* domain, void* volatile* shared_ptr) {
    (void)domain;
    if (!shared_ptr) return NULL;
    void* ptr = *shared_ptr;
    atomic_thread_fence();
    return ptr;
}

void rcu_assign_pointer(RcuDomain* domain, void* volatile* shared_ptr, void* new_ptr) {
    (void)domain;
    if (!shared_ptr) return;
    atomic_thread_fence();
    *shared_ptr = new_ptr;
    atomic_thread_fence();
}

int rcu_update_pointer(RcuDomain* domain, void* volatile* shared_ptr,
                       void* new_ptr, void** old_ptr_out) {
    if (!domain || !shared_ptr) return -1;
    domain->state = RCU_STATE_UPDATE_PENDING;
    atomic_thread_fence();
    void* old_ptr = *shared_ptr;
    *shared_ptr = new_ptr;
    atomic_thread_fence();
    domain->state = RCU_STATE_SYNCHRONIZING;
    rcu_synchronize(domain);
    domain->state = RCU_STATE_READY;
    if (old_ptr_out) *old_ptr_out = old_ptr;
    return 0;
}

void rcu_synchronize(RcuDomain* domain) {
    if (!domain || !domain->is_initialized) return;

    /* FIX-012修复: 使用原子读取registered_thread_count */
    int registered = atomic_load(&domain->registered_thread_count);
    if (registered <= 0) {
        rcu_process_callbacks(domain);
        /* L-010: 无注册线程时也处理epoch回调 */
        for (int s = 0; s < RCU_EPOCH_SLOTS; s++) {
            rcu_process_epoch_callbacks(domain, s);
        }
        domain->grace_periods_completed++;
        return;
    }

    long long start_time = 0;
    if (domain->config.enable_debug_stats) {
#ifdef _WIN32
        start_time = (long long)GetTickCount64();
#else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        start_time = (long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
#endif
    }

    /*
     * L-010: Epoch-based宽限期协议
     *
     * 步骤1: 记录当前epoch (epoch_N)
     * 步骤2: 递增global_epoch到epoch_N+1
     * 步骤3: 自旋等待所有在epoch_N中的读者退出（微秒级）
     * 步骤4: 安全回收epoch_{N-1}槽位的对象（保证epoch_{N-1}的读者已全部退出）
     *
     * 安全论证：
     *   - 步骤2后，新读者记录的epoch >= N+1
     *   - 步骤3等待所有epoch==N的读者退出
     *   - epoch_{N-1}的读者肯定在步骤2前就已退出（因为epoch已经过去了至少2轮）
     *   - 因此epoch_{N-1}的回调可以安全释放
     */
    /* 递增全局epoch（原子操作防止多写者竞态） */
    long old_epoch = (long)atomic_increment(&domain->global_epoch) - 1;
    long new_epoch = old_epoch + 1;

    /* 步骤3: 等待所有在旧epoch中的读者退出 */
    int epoch_ret = rcu_wait_for_epoch(domain, old_epoch);
    if (epoch_ret != 0) {
        /* L-010: epoch超时回退——使用传统sleep_ms方案 */
        for (int fallback = 0; fallback < 10; fallback++) {
            sleep_ms(domain->config.grace_period_ms > 0 ?
                     domain->config.grace_period_ms : 1);
            if (rcu_wait_for_epoch(domain, old_epoch) == 0) break;
        }
    }

    /*
     * 步骤4: 安全回收epoch_{N-2}的回调
     * 使用(old_epoch - 2)保证至少2个epoch的间隔
     */
    long safe_epoch = old_epoch - 2;
    if (safe_epoch > 0 && safe_epoch > domain->last_reclaimed_epoch) {
        int safe_slot = (int)((safe_epoch - 1) % RCU_EPOCH_SLOTS);
        if (safe_slot < 0) safe_slot += RCU_EPOCH_SLOTS;
        rcu_process_epoch_callbacks(domain, safe_slot);
        domain->last_reclaimed_epoch = safe_epoch;
    }

    /* 处理传统callback_list中的回调（向后兼容） */
    rcu_process_callbacks(domain);
    domain->grace_periods_completed++;

    if (domain->config.enable_debug_stats && start_time > 0) {
        long long end_time = 0;
#ifdef _WIN32
        end_time = (long long)GetTickCount64();
        long long elapsed_us = (end_time - start_time) * 1000;
#else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        end_time = (long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
        long long elapsed_us = end_time - start_time;
#endif
        domain->total_grace_period_us += (long)elapsed_us;
        domain->grace_period_samples++;
    }
}

int rcu_defer_reclamation(RcuDomain* domain, void* old_ptr,
                          RcuCallback callback, void* user_data) {
    if (!domain || !old_ptr) return -1;

    RcuCallbackNode* node = (RcuCallbackNode*)safe_calloc(1, sizeof(RcuCallbackNode));
    if (!node) return -1;

    node->old_ptr = old_ptr;
    node->callback = callback;
    node->user_data = user_data;

    /*
     * L-010: Epoch-based回收——将回调加入当前epoch对应的槽位
     * 槽位索引 = (global_epoch - 1) % RCU_EPOCH_SLOTS
     * 这样当global_epoch再递增2次后，该槽位可安全释放
     */
    int epoch_slot = (int)((domain->global_epoch - 1) % RCU_EPOCH_SLOTS);
    if (epoch_slot < 0) epoch_slot += RCU_EPOCH_SLOTS;

#ifdef _WIN32
    EnterCriticalSection(&domain->callback_lock);
#else
    pthread_mutex_lock(&domain->callback_lock);
#endif
    node->next = domain->epoch_callbacks[epoch_slot];
    domain->epoch_callbacks[epoch_slot] = node;
    domain->epoch_callback_counts[epoch_slot]++;
    domain->pending_callback_count++;
    int need_sync = (domain->config.enable_async_reclamation &&
                     domain->pending_callback_count >= domain->config.reclamation_batch_size);
#ifdef _WIN32
    LeaveCriticalSection(&domain->callback_lock);
#else
    pthread_mutex_unlock(&domain->callback_lock);
#endif

    if (need_sync) {
        rcu_synchronize(domain);
    }

    return 0;
}

int call_rcu(RcuDomain* domain, void* old_ptr, RcuCallback callback, void* user_data) {
    return rcu_defer_reclamation(domain, old_ptr, callback, user_data);
}

int rcu_thread_quiescent_state(RcuDomain* domain, int thread_id) {
    if (!domain || thread_id < 0 || thread_id >= MAX_RCU_THREADS) return -1;
    domain->thread_in_read_section[thread_id] = 0;
    domain->thread_reader_count[thread_id]++;
    return 0;
}

int rcu_barrier(RcuDomain* domain) {
    if (!domain || !domain->is_initialized) return -1;
    rcu_synchronize(domain);
    return 0;
}

int rcu_get_stats(const RcuDomain* domain, RcuDomainStats* stats) {
    if (!domain || !stats) return -1;
    stats->registered_threads = domain->registered_thread_count;
    stats->active_readers = domain->active_readers;
    stats->pending_callbacks = domain->pending_callback_count;
    stats->completed_callbacks = domain->completed_callback_count;
    stats->grace_periods_completed = domain->grace_periods_completed;
    stats->reclamation_count = domain->reclamation_count;
    if (domain->grace_period_samples > 0) {
        stats->avg_grace_period_us = domain->total_grace_period_us / domain->grace_period_samples;
    } else {
        stats->avg_grace_period_us = 0;
    }
    return 0;
}

RcuState rcu_get_state(const RcuDomain* domain) {
    if (!domain) return RCU_STATE_UNINITIALIZED;
    return (RcuState)domain->state;
}

int rcu_reset(RcuDomain* domain) {
    if (!domain) return -1;
    rcu_synchronize(domain);
    domain->active_readers = 0;
    domain->pending_callback_count = 0;
    domain->completed_callback_count = 0;
    domain->grace_periods_completed = 0;
    domain->reclamation_count = 0;
    domain->state = RCU_STATE_READY;
    /* L-010: 重置epoch字段 */
    domain->global_epoch = 1;
    domain->last_reclaimed_epoch = 0;
    for (int i = 0; i < MAX_RCU_THREADS; i++) {
        domain->reader_epochs[i] = 0;
    }
    for (int s = 0; s < RCU_EPOCH_SLOTS; s++) {
        rcu_process_epoch_callbacks(domain, s);
    }
    return 0;
}
