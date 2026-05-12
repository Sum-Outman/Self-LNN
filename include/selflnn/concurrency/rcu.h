#ifndef SELFLNN_RCU_H
#define SELFLNN_RCU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RCU_SYNC_GRACEFUL = 0,
    RCU_SYNC_QUIESCENT = 1,
    RCU_SYNC_DEFFERED = 2
} RcuSyncType;

typedef enum {
    RCU_STATE_UNINITIALIZED = 0,
    RCU_STATE_READY = 1,
    RCU_STATE_UPDATE_PENDING = 2,
    RCU_STATE_SYNCHRONIZING = 3
} RcuState;

typedef struct RcuDomain RcuDomain;
typedef struct RcuThread RcuThread;
typedef struct RcuNode RcuNode;

typedef void (*RcuCallback)(void* old_ptr, void* user_data);

typedef struct {
    int max_threads;
    int grace_period_ms;
    int enable_async_reclamation;
    int reclamation_batch_size;
    int enable_debug_stats;
} RcuDomainConfig;

typedef struct {
    int registered_threads;
    int active_readers;
    int pending_callbacks;
    int completed_callbacks;
    int grace_periods_completed;
    int reclamation_count;
    long avg_grace_period_us;
} RcuDomainStats;

RcuDomain* rcu_domain_create(const RcuDomainConfig* config);
void rcu_domain_destroy(RcuDomain* domain);

RcuThread* rcu_thread_register(RcuDomain* domain);
void rcu_thread_unregister(RcuDomain* domain, RcuThread* thread);

void rcu_read_lock(RcuDomain* domain);
void rcu_read_unlock(RcuDomain* domain);

void* rcu_protect_pointer(RcuDomain* domain, void* volatile* shared_ptr);
void rcu_assign_pointer(RcuDomain* domain, void* volatile* shared_ptr, void* new_ptr);

int rcu_update_pointer(RcuDomain* domain, void* volatile* shared_ptr,
                       void* new_ptr, void** old_ptr_out);

void rcu_synchronize(RcuDomain* domain);

int rcu_defer_reclamation(RcuDomain* domain, void* old_ptr,
                          RcuCallback callback, void* user_data);

int call_rcu(RcuDomain* domain, void* old_ptr, RcuCallback callback, void* user_data);

int rcu_thread_quiescent_state(RcuDomain* domain, int thread_id);

int rcu_barrier(RcuDomain* domain);

int rcu_get_stats(const RcuDomain* domain, RcuDomainStats* stats);
RcuState rcu_get_state(const RcuDomain* domain);
int rcu_reset(RcuDomain* domain);

#define RCU_DEREFERENCE(ptr) (__extension__({ \
    __typeof__(ptr) _ptr = (ptr); \
    _ptr; \
}))

#define rcu_assign_pointer_volatile(ptr, val) \
    do { \
        __typeof__(ptr) _val = (val); \
        rcu_assign_pointer(NULL, (void* volatile*)(ptr), (void*)_val); \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif
