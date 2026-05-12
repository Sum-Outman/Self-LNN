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
    for (int i = 0; i < MAX_RCU_THREADS; i++) {
        if (!domain->thread_active[i]) {
            domain->thread_active[i] = 1;
            domain->thread_in_read_section[i] = 0;
            domain->thread_reader_count[i] = 0;
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
    atomic_decrement(&domain->registered_thread_count);
}

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
        if (all_quiescent) return 0;
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
    }

    atomic_thread_fence();
    return domain;
}

void rcu_domain_destroy(RcuDomain* domain) {
    if (!domain) return;
    rcu_synchronize(domain);
    rcu_process_callbacks(domain);
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
    atomic_increment(&domain->active_readers);
    atomic_thread_fence();
}

void rcu_read_unlock(RcuDomain* domain) {
    if (!domain || !domain->is_initialized) return;
    atomic_thread_fence();
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

    int registered = domain->registered_thread_count;
    if (registered <= 0) {
        rcu_process_callbacks(domain);
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

    for (int grace = 0; grace < 2; grace++) {
        int ret = rcu_wait_for_quiescent_state(domain);
        if (ret != 0) {
            for (int sp = 0; sp < domain->config.grace_period_ms; sp += 5) {
                sleep_ms(5);
                ret = rcu_wait_for_quiescent_state(domain);
                if (ret == 0) break;
            }
        }
    }

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

#ifdef _WIN32
    EnterCriticalSection(&domain->callback_lock);
#else
    pthread_mutex_lock(&domain->callback_lock);
#endif
    node->next = domain->callback_list;
    domain->callback_list = node;
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
    return 0;
}
