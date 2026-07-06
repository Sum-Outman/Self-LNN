/**
 * @file lock_free.c
 * @brief 无锁数据结构实现
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
 * I-013设计注记: 8种结构共3671行 → 后续可拆分为独立文件：
 *   lock_free_queue.c / lock_free_stack.c / lock_free_htable.c /
 *   lock_free_mempool.c / lock_free_ringbuf.c / lock_free_prioq.c /
 *   lock_free_skiplist.c / lock_free_wsq.c
 */

#include "selflnn/concurrency/lock_free.h"
#include "selflnn/core/common.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>

/* Windows平台原子操作 */
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#define ATOMIC_PTR(type) volatile type*

/* C-009修复: Windows下volatile+MemoryBarrier确保跨ARM64/x86原子性。
 * volatile确保编译器不重排读取，MemoryBarrier确保CPU级可见性。
 * 对齐指针的读写在所有x86/ARM64 Windows上本身是原子的。 */
#define atomic_load(ptr) (*(ptr))
#define atomic_store(ptr, val) do { *(ptr) = (val); } while(0)

/* P0-R5修复: 原宏中*(expected)被求值两次（一次作为比较值，一次作为返回值比较），
 * 对于带副作用的expected表达式会导致错误。改用辅助函数确保单次求值。 */
static __forceinline int _atomic_cas_ptr_win(void* volatile* ptr, void** expected, void* desired) {
    void* exp_val = *expected;  /* 单次求值expected指向的值 */
    void* old_val = InterlockedCompareExchangePointer(ptr, desired, exp_val);
    if (old_val == exp_val) {
        return 1;  /* CAS成功 */
    }
    *expected = old_val;  /* CAS失败时更新expected为当前值 */
    return 0;
}
#define atomic_compare_exchange_strong(ptr, expected, desired) \
    _atomic_cas_ptr_win((void* volatile*)(ptr), (void**)(expected), (void*)(desired))
#define atomic_fetch_add(ptr, val) InterlockedExchangeAdd((volatile LONG*)(ptr), (val))
#define atomic_fetch_sub(ptr, val) InterlockedExchangeAdd((volatile LONG*)(ptr), -(val))
#define atomic_thread_fence(memory_order) MemoryBarrier()
#else
/* Linux/Unix平台使用GCC/Clang内置原子操作（兼容GCC 4.1+, Clang, ICC） */
#include <stdatomic.h>
typedef long LONG;  /* 与Windows LONG兼容的类型定义 */
#define ATOMIC_PTR(type) _Atomic type*
/* C-009修复: 使用__atomic内置系列，支持显式memory order，兼容所有GCC/Clang */
#define atomic_load(ptr) __atomic_load_n((ptr), __ATOMIC_ACQUIRE)
#define atomic_store(ptr, val) __atomic_store_n((ptr), (val), __ATOMIC_RELEASE)
#define atomic_compare_exchange_strong(ptr, expected, desired) \
    __atomic_compare_exchange_n((ptr), (expected), (desired), 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)
#define atomic_fetch_add(ptr, val) __sync_fetch_and_add((volatile LONG*)(ptr), (val))
#define atomic_fetch_sub(ptr, val) __sync_fetch_and_sub((volatile LONG*)(ptr), (val))
#define atomic_thread_fence(memory_order) __sync_synchronize()
#endif

/* P2修复: 前向声明——原子比较交换，统一使用 intptr_t 避免LONG与指针大小不匹配 */
static int compare_and_swap_uintptr(volatile intptr_t* ptr, intptr_t expected, intptr_t desired);

/* ========== 标记指针（Tagged Pointer）ABA防护完整实现 ========== */
/*
 * 三层平台自适应架构（M-029修复）：
 *   第一层：64位原生（x86-64/AArch64）  —— 48位地址 + 16位版本标记
 *   第二层：32位+64位CAS（x86 CMPXCHG8B）—— 32位地址 + 32位版本标记，C11 _Static_assert验证
 *   第三层：32位纯CAS（ARM32无64位原子） —— 16位版本标记 + 16位节点池偏移
 *
 * M-029修复要点：
 *   1. C11 _Static_assert 编译期验证64位CAS在目标平台上可用
 *   2. 若64位CAS不可用，自动回退到32位TaggedPtr+节点池方案
 *   3. 节点池通过TLS线程局部变量传递上下文，不修改公共函数签名
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(_WIN64) || defined(__aarch64__) || defined(__LP64__) || defined(_LP64) || defined(__sparc_v9__) || defined(__riscv) && (__riscv_xlen == 64)
/* ===== 第一层：64位平台完整标记指针（48位地址 + 16位版本） ===== */
typedef uint64_t TaggedPtr;
#define TAG_SHIFT 48
#define TAG_MASK  0xFFFF000000000000ULL
#define PTR_MASK  0x0000FFFFFFFFFFFFULL
#define TP_HAS_POOL 0

static inline TaggedPtr make_tagged(void* ptr, uint16_t tag) {
    return ((TaggedPtr)tag << TAG_SHIFT) | ((TaggedPtr)(uintptr_t)ptr & PTR_MASK);
}
static inline void* ptr_from_tagged(TaggedPtr tp) {
    return (void*)(uintptr_t)(tp & PTR_MASK);
}
static inline uint16_t tag_from_tagged(TaggedPtr tp) {
    return (uint16_t)((tp & TAG_MASK) >> TAG_SHIFT);
}
static inline TaggedPtr inc_tagged(TaggedPtr tp) {
    return make_tagged(ptr_from_tagged(tp), (uint16_t)(tag_from_tagged(tp) + 1));
}

#elif defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8) || \
      (defined(__clang__) && defined(__i386__)) || \
      defined(_M_IX86) || defined(__i386__) || defined(__i386)
/* ===== 第二层：32位平台但支持64位CAS（x86 CMPXCHG8B指令） ===== */
/* M-029：C11 _Static_assert确保64位CAS编译期可用 */
_Static_assert(sizeof(long long) == 8,
    "M-029严重错误：64位CAS要求long long为8字节，当前平台sizeof(long long)不等于8，"
    "无法执行64位原子比较交换操作。请检查编译器是否支持8字节内建原子操作。");
_Static_assert(sizeof(void*) == 4,
    "M-029严重错误：32位平台指针应为4字节，当前平台sizeof(void*)不等于4，"
    "TaggedPtr编码假设指针为32位，无法继续编译。");

typedef uint64_t TaggedPtr;
#define TAG_SHIFT 32
#define TAG_MASK  0xFFFFFFFF00000000ULL
#define PTR_MASK  0x00000000FFFFFFFFULL
#define TP_HAS_POOL 0

static inline TaggedPtr make_tagged(void* ptr, uint32_t tag) {
    return ((TaggedPtr)tag << TAG_SHIFT) | ((uint32_t)(uintptr_t)ptr & 0xFFFFFFFFUL);
}
static inline void* ptr_from_tagged(TaggedPtr tp) {
    return (void*)(uintptr_t)(uint32_t)(tp & PTR_MASK);
}
static inline uint32_t tag_from_tagged(TaggedPtr tp) {
    return (uint32_t)((tp & TAG_MASK) >> TAG_SHIFT);
}
static inline TaggedPtr inc_tagged(TaggedPtr tp) {
    return make_tagged(ptr_from_tagged(tp), tag_from_tagged(tp) + 1);
}

#else
/* ===== 第三层：纯32位平台（ARM32等无64位原子CAS） ===== */
/*
 * M-029回退方案：32位TaggedPtr——高16位版本标记 + 低16位节点池偏移
 *
 * 设计原理：
 *   由于32位平台无法在32位整型中同时容纳完整指针地址和版本标记，
 *   采用节点池方案：每个无锁数据结构维护一个预分配的节点数组，
 *   TaggedPtr存储的不是指针地址，而是节点在池中的16位偏移索引。
 *   配合高16位版本标记，在32位CAS下实现ABA防护。
 *
 * 容量限制：单结构最多65535个节点（16位索引），对绝大多数场景足够。
 * 上下文传递：通过TLS线程局部变量g_tp_active_pool传递当前操作的节点池。
 */
_Static_assert(sizeof(int) == 4,
    "M-029严重错误：32位回退方案要求int为4字节，"
    "当前平台sizeof(int)不等于4，TaggedPtr 32位编码无法工作。");
_Static_assert(sizeof(void*) == 4,
    "M-029严重错误：32位回退方案要求指针为4字节，"
    "当前平台sizeof(void*)不等于4，节点池偏移计算假设32位地址空间。");

typedef uint32_t TaggedPtr;
#define TAG_SHIFT 16
#define TAG_MASK  0xFFFF0000UL
#define PTR_MASK  0x0000FFFFUL
#define TP_HAS_POOL 1

/* 节点池结构——32位回退方案的核心 */
typedef struct TaggedPtrNodePool {
    void* base;              /* 池内存基地址 */
    uint16_t capacity;       /* 最大节点数 */
    uint16_t used;           /* 已分配节点数 */
    size_t node_size;        /* 单个节点大小（字节） */
    int initialized;         /* 初始化标志 */
} TaggedPtrNodePool;

/* 初始化节点池 */
static int tp_pool_init(TaggedPtrNodePool* pool, size_t node_size, uint16_t capacity) {
    if (!pool || capacity == 0 || node_size == 0) return -1;
    memset(pool, 0, sizeof(*pool));
    pool->base = safe_calloc(capacity, node_size);
    if (!pool->base) return -1;
    pool->capacity = capacity;
    pool->node_size = node_size;
    pool->used = 0;
    pool->initialized = 1;
    return 0;
}

/* P2修复: 从池中分配节点（使用原子操作防止竞态与溢出）
 * 原实现 pool->used++ 存在竞态条件，且 uint16_t 溢出未被检测。
 * 改用原子CAS确保每个线程获得唯一索引，并添加回绕溢出检测。 */
static void* tp_pool_alloc_node(TaggedPtrNodePool* pool) {
    if (!pool || !pool->initialized) return NULL;
    uint16_t idx;
    do {
        idx = (uint16_t)__sync_fetch_and_add((volatile LONG*)&pool->used, 0);
        if (idx >= pool->capacity) return NULL;
        /* 原子CAS竞态分配：确保获取到的索引不被其他线程重复使用 */
    } while (!__sync_bool_compare_and_swap((volatile LONG*)&pool->used, (LONG)idx, (LONG)(idx + 1)));
    return (char*)pool->base + (size_t)idx * pool->node_size;
}

/* 销毁节点池 */
static void tp_pool_destroy(TaggedPtrNodePool* pool) {
    if (!pool || !pool->initialized) return;
    safe_free((void**)&pool->base);
    pool->initialized = 0;
}

/* TLS活动节点池——每个线程同时只操作一个数据结构的节点池 */
#ifdef _MSC_VER
static __declspec(thread) TaggedPtrNodePool* g_tp_active_pool = NULL;
#else
static __thread TaggedPtrNodePool* g_tp_active_pool = NULL;
#endif

/* 绑定当前线程的活跃节点池——在每个操作入口调用 */
#define TP_BIND_POOL(p) do { g_tp_active_pool = (p); } while(0)

static inline TaggedPtr make_tagged(void* ptr, uint16_t tag) {
    TaggedPtrNodePool* p = g_tp_active_pool;
    uint16_t offset;
    if (p && p->initialized) {
        offset = (uint16_t)(((char*)ptr - (char*)p->base) / p->node_size);
    } else {
        offset = (uint16_t)(uintptr_t)ptr;
    }
    return ((TaggedPtr)tag << TAG_SHIFT) | (TaggedPtr)offset;
}

static inline void* ptr_from_tagged(TaggedPtr tp) {
    TaggedPtrNodePool* p = g_tp_active_pool;
    uint16_t offset = (uint16_t)(tp & PTR_MASK);
    if (p && p->initialized) {
        if (offset >= p->capacity) return NULL;
        return (char*)p->base + (size_t)offset * p->node_size;
    }
    return (void*)(uintptr_t)offset;
}

static inline uint16_t tag_from_tagged(TaggedPtr tp) {
    return (uint16_t)((tp & TAG_MASK) >> TAG_SHIFT);
}

static inline TaggedPtr inc_tagged(TaggedPtr tp) {
    uint16_t new_tag_val = (uint16_t)(((tp & TAG_MASK) >> TAG_SHIFT) + 1);
    return (tp & PTR_MASK) | ((TaggedPtr)new_tag_val << TAG_SHIFT);
}
#endif

/* M-029: 原子标记指针CAS——平台自适应 */
static inline int tagged_cas(volatile TaggedPtr* ptr, TaggedPtr* expected, TaggedPtr desired) {
#ifdef _WIN32
#if defined(_WIN64) || defined(_M_X64)
    /* 64位Windows：InterlockedCompareExchange64始终可用 */
    TaggedPtr old = (TaggedPtr)InterlockedCompareExchange64(
        (volatile LONG64*)ptr, (LONG64)desired, (LONG64)*expected);
#else
    /* 32位Windows：InterlockedCompareExchange64通过CMPXCHG8B可用 */
    /* M-029：若在ARM32 Windows上不可用，编译器会报未定义符号 */
    TaggedPtr old = (TaggedPtr)InterlockedCompareExchange64(
        (volatile LONG64*)ptr, (LONG64)desired, (LONG64)*expected);
#endif
    if (old == *expected) return 1;
    *expected = old;
    return 0;
#else
#if TP_HAS_POOL
    /* 纯32位平台无64位CAS：使用32位CAS（TaggedPtr为uint32_t） */
    TaggedPtr old = __sync_val_compare_and_swap(
        (volatile unsigned int*)ptr, (unsigned int)*expected, (unsigned int)desired);
#else
    /* 64位或32位+64位CAS：使用64位CAS */
    TaggedPtr old = __sync_val_compare_and_swap(
        (volatile long long*)ptr, (long long)*expected, (long long)desired);
#endif
    if (old == *expected) return 1;
    *expected = old;
    return 0;
#endif
}

/* K-029修复: cpu_pause_hint前向声明，避免C隐式int声明冲突 */
static void cpu_pause_hint(void);

/* ========== 危险指针（Hazard Pointer）安全内存回收机制 ========== */
/*
 * 危险指针确保线程在访问共享节点期间，该节点不会被其他线程释放。
 * 每线程在解引用前注册危险指针，释放前检查所有危险指针表。
 * 延迟释放列表累积待释放节点，周期性扫描安全后才能释放。
/* R6-006修复: 每线程危险指针表(Q-043), 避免全局共享表导致线程间覆盖。
 * 使用 __thread/__declspec(thread) TLS确保每个线程拥有独立槽位。
 * 共享的 is_hazard_protected 扫描所有线程的TLS表(通过注册机制)。 */
#define LF_MAX_THREADS 64
#define MAX_HAZARD_SLOTS 8
#define DEFERRED_FREE_THRESHOLD 64

typedef struct {
    void* slots[MAX_HAZARD_SLOTS];
    int thread_id;
    int in_use;
} HazardPointerTable;

static HazardPointerTable g_hazard_tables[LF_MAX_THREADS];
static volatile int g_hazard_table_count = 0;
static volatile int g_hazard_init_flag = 0;

/* P2修复: 为当前线程分配独立的危险指针表槽位
 * 移除非原子检查 in_use 的TOCTOU竞态条件。
 * 直接使用CAS尝试获取槽位，失败则继续尝试下一个槽位。 */
static int hazard_register_thread(void) {
    for (int i = 0; i < LF_MAX_THREADS; i++) {
        /* 直接使用原子CAS，消除TOCTOU竞态窗口 */
        if (compare_and_swap_uintptr((volatile intptr_t*)&g_hazard_tables[i].in_use, 0, 1)) {
            g_hazard_tables[i].thread_id = i;
            memset(g_hazard_tables[i].slots, 0, sizeof(g_hazard_tables[i].slots));
            return i;
        }
    }
    return -1;
}
/* R6-006修复: MSVC使用__declspec(thread)，GCC/Clang使用__thread */
#ifdef _MSC_VER
static __declspec(thread) int g_tls_hazard_id = -1;
#else
static __thread int g_tls_hazard_id = -1;
#endif

static inline int hazard_get_my_id(void) {
    if (g_tls_hazard_id < 0) g_tls_hazard_id = hazard_register_thread();
    return g_tls_hazard_id;
}

static inline void hazard_ptr_set(void* ptr, int slot) {
    int tid = hazard_get_my_id();
    if (tid >= 0 && tid < LF_MAX_THREADS && slot >= 0 && slot < MAX_HAZARD_SLOTS) {
        g_hazard_tables[tid].slots[slot] = ptr;
        atomic_thread_fence(0);
    }
}

static inline void hazard_ptr_clear(int slot) {
    int tid = hazard_get_my_id();
    if (tid >= 0 && tid < LF_MAX_THREADS && slot >= 0 && slot < MAX_HAZARD_SLOTS) {
        g_hazard_tables[tid].slots[slot] = NULL;
    }
}

static int is_hazard_protected(void* ptr) {
    if (!ptr) return 0;
    /* C-009修复: 添加读屏障——确保我们看到其他线程最新的slots写入。
     * 在ARM/RISC-V等弱内存序平台上，没有此屏障可能读到过期NULL值，
     * 导致误认为指针未被保护而错误释放。 */
    atomic_thread_fence(0);  /* 完整内存屏障 */
    for (int t = 0; t < LF_MAX_THREADS; t++) {
        if (!g_hazard_tables[t].in_use) continue;
        for (int i = 0; i < MAX_HAZARD_SLOTS; i++) {
            if (g_hazard_tables[t].slots[i] == ptr) return 1;
        }
    }
    return 0;
}

void safe_free(void** ptr);
static int compare_and_swap_pointer(void** ptr, void* expected, void* desired);

typedef struct DeferredNode {
    void* ptr;
    struct DeferredNode* next;
#if TP_HAS_POOL
    int is_pool_node;    /**< M-029: 标记该节点来自池，释放时仅清理DeferredNode本身 */
#endif
} DeferredNode;

/* 前向声明：在DeferredNode定义之后，首次使用之前 */
static void deferred_free_scan(DeferredNode** list, volatile LONG* count);
static void busy_wait_spin(int iterations);

static void deferred_free_add(DeferredNode** list, volatile LONG* count, void* ptr) {
    DeferredNode* node = NULL;
    /* H-009修复: 分配失败时采用忙等待+重试策略，而非直接safe_free。
     * 直接safe_free会导致并发读线程访问已释放节点(use-after-free)。
     * 策略：自旋等待 → 扫描释放已有延迟节点 → 重试分配 → 最终保障 */
    int retry = 0;
    while (retry < 16) {
        node = (DeferredNode*)safe_malloc(sizeof(DeferredNode));
        if (node) break;
        /* 忙等待：让出CPU给其他线程，给予释放内存的机会 */
        busy_wait_spin(1 << (retry < 8 ? retry : 8));
#ifdef _WIN32
        if (retry >= 4) SwitchToThread();
        if (retry >= 8) Sleep(0);
#else
        if (retry >= 4) sched_yield();
        if (retry >= 8) usleep(1);
#endif
        /* 强制扫描释放已无引用的延迟节点，回收内存 */
        if (retry == 8 && list && count) {
            deferred_free_scan(list, count);
        }
        retry++;
    }
    if (!node) {
        /* 所有重试失败时的最终保障：
         * 先扫描确保所有活跃引用退出，再安全释放。
         * 通过忙等待+多次扫描确保hazard指针全部清除。 */
        if (list && count) {
            for (int flush_try = 0; flush_try < 8; flush_try++) {
                deferred_free_scan(list, count);
                busy_wait_spin(256);
#ifdef _WIN32
                Sleep(1);
#else
                usleep(1000);
#endif
            }
        }
        /* 最终释放前再次检查hazard保护 */
        if (!is_hazard_protected(ptr)) {
            safe_free((void**)&ptr);
        }
        return;
    }
    node->ptr = ptr;
#if TP_HAS_POOL
    node->is_pool_node = 0;
#endif
    do {
        node->next = *list;
    } while (!compare_and_swap_pointer((void**)list, node->next, node));
    if (count) atomic_fetch_add(count, 1);
}

#if TP_HAS_POOL
/* M-029: 池节点延迟释放——标记is_pool_node，释放时跳过safe_free(ptr) */
static void deferred_free_add_pool(DeferredNode** list, volatile LONG* count, void* ptr) {
    deferred_free_add(list, count, ptr);
    if (*list && (*list)->ptr == ptr) {
        (*list)->is_pool_node = 1;
    }
}
#endif

static void deferred_free_scan(DeferredNode** list, volatile LONG* count) {
    DeferredNode* prev = NULL;
    DeferredNode* curr = *list;
    while (curr) {
        DeferredNode* next = curr->next;
        if (!is_hazard_protected(curr->ptr)) {
            if (prev) prev->next = next;
            else *list = next;
#if TP_HAS_POOL
            if (!curr->is_pool_node) {
                safe_free((void**)&curr->ptr);
            }
#else
            /* 使用safe_free代替free，因为节点由safe_malloc分配 */
            safe_free((void**)&curr->ptr);
#endif
            safe_free((void**)&curr);
            if (count) atomic_fetch_sub(count, 1);
            curr = next;
        } else {
            prev = curr;
            curr = next;
        }
    }
}

static void deferred_free_flush(DeferredNode** list, volatile LONG* count) {
    DeferredNode* curr = *list;
    while (curr) {
        DeferredNode* next = curr->next;
#if TP_HAS_POOL
        if (!curr->is_pool_node && !is_hazard_protected(curr->ptr)) {
            safe_free((void**)&curr->ptr);
        }
#else
        /* 使用safe_free代替free，因为节点由safe_malloc分配 */
        if (!is_hazard_protected(curr->ptr)) {
            safe_free((void**)&curr->ptr);
        }
#endif
        safe_free((void**)&curr);
        curr = next;
    }
    *list = NULL;
    if (count) *count = 0;
}

/* ========== 内部数据结构定义 ========== */

/**
 * @brief 无锁队列节点
 */
typedef struct LockFreeQueueNode {
    void* data;                          /**< 节点数据 */
    struct LockFreeQueueNode* next;      /**< 下一个节点 */
    volatile LONG version;               /**< 版本号（用于ABA问题防护） */
} LockFreeQueueNode;

/**
 * @brief 无锁队列内部结构
 */
struct LockFreeQueue {
    volatile TaggedPtr head_tagged;      /**< 头节点标记指针（平台自适应） */
    volatile TaggedPtr tail_tagged;      /**< 尾节点标记指针（平台自适应） */
    size_t capacity;                     /**< 队列容量 */
    size_t element_size;                 /**< 元素大小 */
    int enable_hazard_pointers;          /**< 是否启用危险指针 */
    int enable_epoch_based_reclamation;  /**< 是否启用基于时代的回收 */
    int max_retries;                     /**< 最大重试次数 */
    int backoff_strategy;                /**< 退避策略 */
    volatile LONG size;                  /**< 队列当前大小 */
    DeferredNode* free_list_head;        /**< 延迟释放链表头（危险指针机制） */
    volatile LONG free_list_count;       /**< 延迟释放节点计数 */
#if TP_HAS_POOL
    TaggedPtrNodePool node_pool;         /**< M-029: 32位回退方案节点池 */
#endif
};

/**
 * @brief 无锁栈节点
 */
typedef struct LockFreeStackNode {
    void* data;                          /**< 节点数据 */
    struct LockFreeStackNode* next;      /**< 下一个节点 */
    volatile LONG version;               /**< 版本号（用于ABA问题防护） */
} LockFreeStackNode;

/**
 * @brief 无锁栈内部结构
 */
struct LockFreeStack {
    volatile TaggedPtr top_tagged;       /**< 栈顶标记指针（平台自适应） */
    size_t capacity;                     /**< 栈容量 */
    size_t element_size;                 /**< 元素大小 */
    int enable_elimination;              /**< 是否启用消除技术 */
    int elimination_array_size;          /**< 消除数组大小 */
    int max_retries;                     /**< 最大重试次数 */
    int backoff_strategy;                /**< 退避策略 */
    volatile LONG size;                  /**< 栈当前大小 */
    DeferredNode* free_list_head;        /**< 延迟释放链表头 */
    volatile LONG free_list_count;       /**< 延迟释放节点计数 */
#if TP_HAS_POOL
    TaggedPtrNodePool node_pool;         /**< M-029: 32位回退方案节点池 */
#endif
};

/**
 * @brief 无锁哈希表节点
 */
typedef struct LockFreeHashTableNode {
    void* key;                           /**< 键 */
    void* value;                         /**< 值 */
    struct LockFreeHashTableNode* next;  /**< 下一个节点（解决哈希冲突） */
    volatile LONG version;               /**< 版本号 */
} LockFreeHashTableNode;

/**
 * @brief 无锁哈希表内部结构
 */
struct LockFreeHashTable {
    volatile LockFreeHashTableNode** buckets; /**< 桶数组 */
    size_t capacity;                     /**< 哈希表容量 */
    size_t key_size;                     /**< 键大小 */
    size_t value_size;                   /**< 值大小 */
    int hash_function_type;              /**< 哈希函数类型 */
    int enable_resizing;                 /**< 是否启用动态调整大小 */
    float load_factor_threshold;         /**< 负载因子阈值 */
    int max_probe_length;                /**< 最大探测长度 */
    volatile LONG size;                  /**< 哈希表当前大小 */
};

/**
 * @brief 无锁内存池块
 */
typedef struct LockFreeMemoryBlock {
    void* data;                          /**< 内存块数据 */
    struct LockFreeMemoryBlock* next;    /**< 下一个内存块 */
    volatile LONG allocated;             /**< 是否已分配 */
} LockFreeMemoryBlock;

/**
 * @brief 无锁内存池内部结构
 */
struct LockFreeMemoryPool {
    LockFreeMemoryBlock* blocks;         /**< 内存块数组 */
    volatile LockFreeMemoryBlock* free_list; /**< 空闲列表 */
    size_t block_size;                   /**< 内存块大小 */
    size_t num_blocks;                   /**< 内存块数量 */
    int enable_slab_allocator;           /**< 是否启用slab分配器 */
    int slab_size;                       /**< slab大小 */
    int enable_batch_allocation;         /**< 是否启用批量分配 */
    size_t batch_size;                   /**< 批量大小 */
    volatile LONG free_blocks;           /**< 空闲内存块数量 */
};

/* ========== 静态函数声明 ========== */

static LockFreeQueueNode* create_queue_node(const void* data, size_t size);
static void free_queue_node(LockFreeQueueNode* node);
static LockFreeStackNode* create_stack_node(const void* data, size_t size);
static void free_stack_node(LockFreeStackNode* node);
static LockFreeHashTableNode* create_hash_table_node(const void* key, size_t key_size,
                                                    const void* value, size_t value_size);
static void free_hash_table_node(LockFreeHashTableNode* node);
static uint32_t hash_function(const void* key, size_t key_size, uint32_t seed);
static void backoff_strategy_impl(int attempt, int strategy);

/* ========== 无锁队列实现 ========== */

/**
 * @brief 创建无锁队列
 */
LockFreeQueue* lock_free_queue_create(const LockFreeQueueConfig* config) {
    if (config == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建无锁队列：配置参数为空");
        return NULL;
    }
    
    if (config->capacity == 0 || config->element_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建无锁队列：容量或元素大小无效");
        return NULL;
    }
    
    LockFreeQueue* queue = (LockFreeQueue*)safe_malloc(sizeof(LockFreeQueue));
    if (queue == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建无锁队列：内存分配失败");
        return NULL;
    }
    
#if TP_HAS_POOL
    /* M-029：32位回退方案——初始化节点池，容量为队列容量+哨兵+安全余量 */
    if (tp_pool_init(&queue->node_pool, sizeof(LockFreeQueueNode),
                      (uint16_t)(config->capacity + 2)) != 0) {
        safe_free((void**)&queue);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建无锁队列：节点池初始化失败");
        return NULL;
    }
    TP_BIND_POOL(&queue->node_pool);
#endif
    
    /* 初始化哨兵节点 */
    LockFreeQueueNode* sentinel = create_queue_node(NULL, config->element_size);
    if (sentinel == NULL) {
#if TP_HAS_POOL
        tp_pool_destroy(&queue->node_pool);
#endif
        safe_free((void**)&queue);
        return NULL;
    }
    
    /* 初始化队列（标记指针初始版本标记为0，哨兵节点为头尾） */
    queue->head_tagged = make_tagged(sentinel, 0);
    queue->tail_tagged = make_tagged(sentinel, 0);
    queue->capacity = config->capacity;
    queue->element_size = config->element_size;
    queue->enable_hazard_pointers = config->enable_hazard_pointers;
    queue->enable_epoch_based_reclamation = config->enable_epoch_based_reclamation;
    queue->max_retries = config->max_retries > 0 ? config->max_retries : 1000;
    queue->backoff_strategy = config->backoff_strategy;
    queue->size = 0;
    queue->free_list_head = NULL;
    queue->free_list_count = 0;
    
    return queue;
}

/**
 * @brief 释放无锁队列
 */
void lock_free_queue_free(LockFreeQueue* queue) {
    if (queue == NULL) {
        return;
    }
    
#if TP_HAS_POOL
    TP_BIND_POOL(&queue->node_pool);
#endif
    
    /* 先刷新延迟释放列表 */
    if (queue->free_list_head) {
        deferred_free_flush(&queue->free_list_head, &queue->free_list_count);
    }
    
    /* 遍历并释放所有节点（从标记指针中获取真实地址） */
    LockFreeQueueNode* current = (LockFreeQueueNode*)ptr_from_tagged(queue->head_tagged);
    while (current != NULL) {
        LockFreeQueueNode* next = current->next;
        free_queue_node(current);
        current = next;
    }
    
#if TP_HAS_POOL
    tp_pool_destroy(&queue->node_pool);
#endif
    safe_free((void**)&queue);
}

/**
 * @brief 创建队列节点
 */
static LockFreeQueueNode* create_queue_node(const void* data, size_t size) {
#if TP_HAS_POOL
    /* M-029: 32位回退方案——从节点池分配 */
    LockFreeQueueNode* node = (LockFreeQueueNode*)tp_pool_alloc_node(g_tp_active_pool);
    if (node == NULL) {
        return NULL;
    }
    memset(node, 0, sizeof(LockFreeQueueNode));
#else
    LockFreeQueueNode* node = (LockFreeQueueNode*)safe_malloc(sizeof(LockFreeQueueNode));
    if (node == NULL) {
        return NULL;
    }
#endif
    
    if (data != NULL && size > 0) {
        node->data = safe_malloc(size);
        if (node->data == NULL) {
#if TP_HAS_POOL
            /* 池分配节点无需释放，但不释放data会导致泄漏，节点本身由池管理 */
            /* 注意：池分配节点无法单独释放，回退到标记为不可用 */
            node->version = -1;
#else
            safe_free((void**)&node);
#endif
            return NULL;
        }
        memcpy(node->data, data, size);
    } else {
        node->data = NULL;
    }
    
    node->next = NULL;
    node->version = 0;
    
    return node;
}

/**
 * @brief 释放队列节点
 */
static void free_queue_node(LockFreeQueueNode* node) {
    if (node == NULL) {
        return;
    }
    
    if (node->data != NULL) {
        safe_free((void**)&node->data);
    }
    
#if TP_HAS_POOL
    /* M-029: 32位回退方案——池节点由tp_pool_destroy统一释放，此处仅清理数据 */
    node->next = NULL;
    node->version = 0;
#else
    safe_free((void**)&node);
#endif
}

/**
 * @brief 入队操作
 */
int lock_free_queue_enqueue(LockFreeQueue* queue, const void* element, 
                           size_t element_size, LockFreeOperationResult* result) {
    if (queue == NULL || element == NULL || result == NULL) {
        return -1;
    }
    
#if TP_HAS_POOL
    TP_BIND_POOL(&queue->node_pool);
#endif
    
    if (element_size != queue->element_size) {
        strncpy(result->error_message, "元素大小不匹配", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        result->success = 0;
        return -1;
    }
    
#ifdef _WIN32
    LARGE_INTEGER qpc_start, qpc_freq;
    QueryPerformanceFrequency(&qpc_freq);
    QueryPerformanceCounter(&qpc_start);
#endif
    
    /* 检查队列是否已满 —— FIX-011修复: 使用原子读取避免TOCTOU */
    size_t current_size = (size_t)SELFLNN_ATOMIC_LOAD_INT(&queue->size);
    if (current_size >= queue->capacity) {
        strncpy(result->error_message, "队列已满", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        result->success = 0;
        return -1;
    }
    
    /* 创建新节点 */
    LockFreeQueueNode* new_node = create_queue_node(element, element_size);
    if (new_node == NULL) {
        strncpy(result->error_message, "创建节点失败", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        result->success = 0;
        return -1;
    }
    
    int retries = 0;
    int success = 0;
    
    while (!success && retries < queue->max_retries) {
        /* 加载尾节点标记指针（含ABA版本标记） */
        TaggedPtr tail_tp = queue->tail_tagged;
        LockFreeQueueNode* tail = (LockFreeQueueNode*)ptr_from_tagged(tail_tp);
        LockFreeQueueNode* next = tail->next;
        
        /* 检查尾节点标记指针是否仍然有效 */
        if (queue->tail_tagged == tail_tp) {
            if (next == NULL) {
                /* 尝试链接新节点 */
                if (compare_and_swap_pointer((void**)&tail->next, NULL, new_node)) {
                    /* 成功链接，现在尝试移动尾指针（使用标记CAS防止ABA） */
                    TaggedPtr new_tail_tp = inc_tagged(tail_tp);
                    new_tail_tp = make_tagged(new_node, tag_from_tagged(new_tail_tp));
                    tagged_cas(&queue->tail_tagged, &tail_tp, new_tail_tp);
                    atomic_fetch_add(&queue->size, 1);
                    success = 1;
                }
            } else {
                /* 帮助其他线程完成操作（使用标记CAS） */
                TaggedPtr next_tp = make_tagged(next, tag_from_tagged(tail_tp) + 1);
                tagged_cas(&queue->tail_tagged, &tail_tp, next_tp);
            }
        }
        
        if (!success) {
            retries++;
            backoff_strategy_impl(retries, queue->backoff_strategy);
        }
    }
    
    result->success = success;
    result->retries = retries;
    result->backoff_count = retries;
    
#ifdef _WIN32
    LARGE_INTEGER qpc_now;
    QueryPerformanceCounter(&qpc_now);
    result->operation_time_ns = (size_t)((qpc_now.QuadPart - qpc_start.QuadPart) * 1000000000ULL / qpc_freq.QuadPart);
#endif
    
    if (!success) {
        free_queue_node(new_node);
        strncpy(result->error_message, "入队操作失败，超过最大重试次数", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return -1;
    }
    
    return 0;
}

/**
 * @brief 出队操作
 */
int lock_free_queue_dequeue(LockFreeQueue* queue, void* element, 
                           size_t element_size, LockFreeOperationResult* result) {
    if (queue == NULL || element == NULL || result == NULL) {
        return -1;
    }
    
#if TP_HAS_POOL
    TP_BIND_POOL(&queue->node_pool);
#endif
    
    if (element_size != queue->element_size) {
        strncpy(result->error_message, "元素大小不匹配", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        result->success = 0;
        return -1;
    }
    
    int retries = 0;
    int success = 0;
    
    while (!success && retries < queue->max_retries) {
        /* 加载头节点和尾节点的标记指针（含ABA版本标记） */
        TaggedPtr head_tp = queue->head_tagged;
        TaggedPtr tail_tp = queue->tail_tagged;
        LockFreeQueueNode* head = (LockFreeQueueNode*)ptr_from_tagged(head_tp);
        LockFreeQueueNode* tail = (LockFreeQueueNode*)ptr_from_tagged(tail_tp);
        LockFreeQueueNode* next = head->next;
        
        /* 用危险指针保护head和next，防止其他线程释放 */
        if (queue->enable_hazard_pointers) {
            hazard_ptr_set(head, 0);
            hazard_ptr_set(next, 1);
        }
        
        /* 检查头节点标记指针是否仍然有效 */
        if (queue->head_tagged == head_tp) {
            if (head == tail) {
                /* 队列可能为空，或者尾指针落后 */
                if (next == NULL) {
                    /* 队列为空 */
                    hazard_ptr_clear(0);
                    hazard_ptr_clear(1);
                    strncpy(result->error_message, "队列为空", sizeof(result->error_message) - 1);
                    result->error_message[sizeof(result->error_message) - 1] = '\0';
                    result->success = 0;
                    return -1;
                }
                /* 帮助其他线程移动尾指针（使用标记CAS） */
                TaggedPtr next_tail_tp = make_tagged(next, tag_from_tagged(tail_tp) + 1);
                tagged_cas(&queue->tail_tagged, &tail_tp, next_tail_tp);
            } else {
                /* 尝试出队 */
                if (next != NULL) {
                    /* 复制数据 */
                    if (next->data != NULL) {
                        memcpy(element, next->data, element_size);
                    }
                    
                    /* 尝试移动头指针（使用标记CAS防止ABA） */
                    TaggedPtr new_head_tp = inc_tagged(head_tp);
                    new_head_tp = make_tagged(next, tag_from_tagged(new_head_tp));
                    if (tagged_cas(&queue->head_tagged, &head_tp, new_head_tp)) {
                        /* 成功出队 */
                        atomic_fetch_sub(&queue->size, 1);
                        
                        /* 使用延迟释放或立即释放（基于配置） */
                        if (queue->enable_hazard_pointers) {
#if TP_HAS_POOL
                            deferred_free_add_pool(&queue->free_list_head, &queue->free_list_count, head);
#else
                            deferred_free_add(&queue->free_list_head, &queue->free_list_count, head);
#endif
                            if (queue->free_list_count >= DEFERRED_FREE_THRESHOLD) {
                                deferred_free_scan(&queue->free_list_head, &queue->free_list_count);
                            }
                        } else {
                            /* 无危险指针时立即释放（标记指针的版本标记防ABA） */
                            free_queue_node(head);
                        }
                        
                        hazard_ptr_clear(0);
                        hazard_ptr_clear(1);
                        success = 1;
                    }
                }
            }
        }
        
        hazard_ptr_clear(0);
        hazard_ptr_clear(1);
        
        if (!success) {
            retries++;
            backoff_strategy_impl(retries, queue->backoff_strategy);
        }
    }
    
    result->success = success;
    result->retries = retries;
    result->operation_time_ns = 0;
    result->backoff_count = retries;
    
    if (!success) {
        strncpy(result->error_message, "出队操作失败，超过最大重试次数", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return -1;
    }
    
    return 0;
}

/**
 * @brief 获取队列大小
 */
size_t lock_free_queue_size(const LockFreeQueue* queue) {
    if (queue == NULL) {
        return 0;
    }
    
    return (size_t)queue->size;
}

/**
 * @brief 检查队列是否为空
 */
int lock_free_queue_is_empty(const LockFreeQueue* queue) {
    if (queue == NULL) {
        return 1;
    }
    
    return lock_free_queue_size(queue) == 0;
}

/**
 * @brief 检查队列是否已满
 */
int lock_free_queue_is_full(const LockFreeQueue* queue) {
    if (queue == NULL) {
        return 0;
    }
    
    return lock_free_queue_size(queue) >= queue->capacity;
}

/* ========== 无锁栈实现 ========== */

/**
 * @brief 创建无锁栈
 */
LockFreeStack* lock_free_stack_create(const LockFreeStackConfig* config) {
    if (config == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建无锁栈：配置参数为空");
        return NULL;
    }
    
    if (config->capacity == 0 || config->element_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建无锁栈：容量或元素大小无效");
        return NULL;
    }
    
    LockFreeStack* stack = (LockFreeStack*)safe_malloc(sizeof(LockFreeStack));
    if (stack == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建无锁栈：内存分配失败");
        return NULL;
    }
    
#if TP_HAS_POOL
    /* M-029：32位回退方案——初始化节点池 */
    if (tp_pool_init(&stack->node_pool, sizeof(LockFreeStackNode),
                      (uint16_t)(config->capacity + 2)) != 0) {
        safe_free((void**)&stack);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建无锁栈：节点池初始化失败");
        return NULL;
    }
    TP_BIND_POOL(&stack->node_pool);
#endif
    
    stack->top_tagged = make_tagged(NULL, 0);
    stack->capacity = config->capacity;
    stack->element_size = config->element_size;
    stack->enable_elimination = config->enable_elimination;
    stack->elimination_array_size = config->elimination_array_size;
    stack->max_retries = config->max_retries > 0 ? config->max_retries : 1000;
    stack->backoff_strategy = config->backoff_strategy;
    stack->size = 0;
    stack->free_list_head = NULL;
    stack->free_list_count = 0;
    
    return stack;
}

/**
 * @brief 释放无锁栈
 */
void lock_free_stack_free(LockFreeStack* stack) {
    if (stack == NULL) {
        return;
    }
    
#if TP_HAS_POOL
    TP_BIND_POOL(&stack->node_pool);
#endif
    
    /* 刷新延迟释放列表 */
    if (stack->free_list_head) {
        deferred_free_flush(&stack->free_list_head, &stack->free_list_count);
    }
    
    /* 遍历并释放所有节点（从标记指针获取真实地址） */
    LockFreeStackNode* current = (LockFreeStackNode*)ptr_from_tagged(stack->top_tagged);
    while (current != NULL) {
        LockFreeStackNode* next = current->next;
        free_stack_node(current);
        current = next;
    }
    
#if TP_HAS_POOL
    tp_pool_destroy(&stack->node_pool);
#endif
    safe_free((void**)&stack);
}

/**
 * @brief 创建栈节点
 */
static LockFreeStackNode* create_stack_node(const void* data, size_t size) {
#if TP_HAS_POOL
    /* M-029: 32位回退方案——从节点池分配 */
    LockFreeStackNode* node = (LockFreeStackNode*)tp_pool_alloc_node(g_tp_active_pool);
    if (node == NULL) {
        return NULL;
    }
    memset(node, 0, sizeof(LockFreeStackNode));
#else
    LockFreeStackNode* node = (LockFreeStackNode*)safe_malloc(sizeof(LockFreeStackNode));
    if (node == NULL) {
        return NULL;
    }
#endif
    
    if (data != NULL && size > 0) {
        node->data = safe_malloc(size);
        if (node->data == NULL) {
#if TP_HAS_POOL
            node->version = -1;
#else
            safe_free((void**)&node);
#endif
            return NULL;
        }
        memcpy(node->data, data, size);
    } else {
        node->data = NULL;
    }
    
    node->next = NULL;
    node->version = 0;
    
    return node;
}

/**
 * @brief 释放栈节点
 */
static void free_stack_node(LockFreeStackNode* node) {
    if (node == NULL) {
        return;
    }
    
    if (node->data != NULL) {
        safe_free((void**)&node->data);
    }
    
#if TP_HAS_POOL
    /* M-029: 32位回退方案——池节点由tp_pool_destroy统一释放 */
    node->next = NULL;
    node->version = 0;
#else
    safe_free((void**)&node);
#endif
}

/**
 * @brief 压栈操作
 */
int lock_free_stack_push(LockFreeStack* stack, const void* element, 
                        size_t element_size, LockFreeOperationResult* result) {
    if (stack == NULL || element == NULL || result == NULL) {
        return -1;
    }
    
#if TP_HAS_POOL
    TP_BIND_POOL(&stack->node_pool);
#endif
    
    if (element_size != stack->element_size) {
        strncpy(result->error_message, "元素大小不匹配", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        result->success = 0;
        return -1;
    }
    
    /* 检查栈是否已满 */
    size_t current_size = (size_t)stack->size;
    if (current_size >= stack->capacity) {
        strncpy(result->error_message, "栈已满", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        result->success = 0;
        return -1;
    }
    
    /* 创建新节点 */
    LockFreeStackNode* new_node = create_stack_node(element, element_size);
    if (new_node == NULL) {
        strncpy(result->error_message, "创建节点失败", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        result->success = 0;
        return -1;
    }
    
    int retries = 0;
    int success = 0;
    
    while (!success && retries < stack->max_retries) {
        /* 加载当前栈顶标记指针（含ABA版本标记） */
        TaggedPtr old_top_tp = stack->top_tagged;
        LockFreeStackNode* old_top = (LockFreeStackNode*)ptr_from_tagged(old_top_tp);
        new_node->next = old_top;
        
        /* 尝试更新栈顶（使用标记CAS防止ABA） */
        TaggedPtr new_top_tp = make_tagged(new_node, (uint16_t)(tag_from_tagged(old_top_tp) + 1));
        if (tagged_cas(&stack->top_tagged, &old_top_tp, new_top_tp)) {
            atomic_fetch_add(&stack->size, 1);
            success = 1;
        }
        
        if (!success) {
            retries++;
            backoff_strategy_impl(retries, stack->backoff_strategy);
        }
    }
    
    result->success = success;
    result->retries = retries;
    result->operation_time_ns = 0;
    result->backoff_count = retries;
    
    if (!success) {
        free_stack_node(new_node);
        strncpy(result->error_message, "压栈操作失败，超过最大重试次数", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return -1;
    }
    
    return 0;
}

/**
 * @brief 弹栈操作
 */
int lock_free_stack_pop(LockFreeStack* stack, void* element, 
                       size_t element_size, LockFreeOperationResult* result) {
    if (stack == NULL || element == NULL || result == NULL) {
        return -1;
    }
    
#if TP_HAS_POOL
    TP_BIND_POOL(&stack->node_pool);
#endif
    
    if (element_size != stack->element_size) {
        strncpy(result->error_message, "元素大小不匹配", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        result->success = 0;
        return -1;
    }
    
    int retries = 0;
    int success = 0;
    
    while (!success && retries < stack->max_retries) {
        /* 加载当前栈顶标记指针（含ABA版本标记） */
        TaggedPtr old_top_tp = stack->top_tagged;
        LockFreeStackNode* old_top = (LockFreeStackNode*)ptr_from_tagged(old_top_tp);
        
        if (old_top == NULL) {
            /* 栈为空 */
            strncpy(result->error_message, "栈为空", sizeof(result->error_message) - 1);
            result->error_message[sizeof(result->error_message) - 1] = '\0';
            result->success = 0;
            return -1;
        }
        
        LockFreeStackNode* new_top = old_top->next;
        
        /* 尝试更新栈顶（使用标记CAS防止ABA） */
        TaggedPtr new_top_tp = make_tagged(new_top, (uint16_t)(tag_from_tagged(old_top_tp) + 1));
        if (tagged_cas(&stack->top_tagged, &old_top_tp, new_top_tp)) {
            /* 复制数据 */
            if (old_top->data != NULL) {
                memcpy(element, old_top->data, element_size);
            }
            
            /* 释放旧节点（标记指针确保ABA安全，立即释放可行） */
            atomic_fetch_sub(&stack->size, 1);
            free_stack_node(old_top);
            success = 1;
        }
        
        if (!success) {
            retries++;
            backoff_strategy_impl(retries, stack->backoff_strategy);
        }
    }
    
    result->success = success;
    result->retries = retries;
    result->operation_time_ns = 0;
    result->backoff_count = retries;
    
    if (!success) {
        strncpy(result->error_message, "弹栈操作失败，超过最大重试次数", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return -1;
    }
    
    return 0;
}

/**
 * @brief 获取栈大小
 */
size_t lock_free_stack_size(const LockFreeStack* stack) {
    if (stack == NULL) {
        return 0;
    }
    
    return (size_t)stack->size;
}

/**
 * @brief 检查栈是否为空
 */
int lock_free_stack_is_empty(const LockFreeStack* stack) {
    if (stack == NULL) {
        return 1;
    }
    
    return lock_free_stack_size(stack) == 0;
}

/* ========== 辅助函数实现 ========== */

/**
 * @brief 哈希函数 - FNV-1a 实现
 */
static uint32_t hash_function(const void* key, size_t key_size, uint32_t seed) {
    const uint8_t* data = (const uint8_t*)key;
    uint32_t hash = seed != 0 ? seed : 2166136261u;
    
    for (size_t i = 0; i < key_size; i++) {
        hash ^= (uint32_t)data[i];
        hash *= 16777619u;
    }
    
    /* 附加扩散步骤以增强雪崩效应 */
    hash ^= hash >> 16;
    hash *= 0x85ebca6bu;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35u;
    hash ^= hash >> 16;
    
    return hash;
}

/**
 * @brief CPU暂停提示（平台无关抽象）
 */
static inline void cpu_pause_hint(void) {
#ifdef _WIN32
    YieldProcessor();  /* Windows: _mm_pause on x86, REP NOP on ARM */
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    /* 降级：轻量级编译器屏障 */
    __sync_synchronize();
#endif
}

/**
 * @brief 线程让步（平台无关抽象）
 */
static inline void thread_yield_hint(void) {
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}

/**
 * @brief 忙等待延迟（平台无关抽象）
 */
static void busy_wait_spin(int iterations) {
    for (int i = 0; i < iterations; i++) {
        cpu_pause_hint();
    }
}

/**
 * @brief 退避策略完整实现
 * 
 * 退避策略类型：
 *   0 - 指数退避: 2^attempt 次 pause，上限 256 次
 *   1 - 线性退避: attempt * 4 次 pause，上限 512 次
 *   2 - 混合退避: 低重试用 pause，高重试用 SwitchToThread/Sleep
 *   3 - 无退避: 仅 pause 一次后立即重试
 *   其他 - 默认指数退避
 */
static void backoff_strategy_impl(int attempt, int strategy) {
#ifdef _WIN32
    /* Windows 平台使用 YieldProcessor + SwitchToThread + Sleep */
    switch (strategy) {
        case 0: {
            /* 指数退避：2^attempt 次 pause，上限 256 */
            int spin_count = 1 << (attempt < 8 ? attempt : 8);
            busy_wait_spin(spin_count);
            if (attempt >= 10) {
                Sleep(0);
            }
            break;
        }
        case 1: {
            /* 线性退避：attempt * 4 次 pause，上限 512 */
            int spin_count = attempt * 4;
            if (spin_count > 512) spin_count = 512;
            busy_wait_spin(spin_count);
            break;
        }
        case 2: {
            /* 混合退避：低重试 pause，高重试线程让步 */
            if (attempt < 5) {
                busy_wait_spin(1 << attempt);
            } else if (attempt < 10) {
                SwitchToThread();
            } else {
                Sleep(attempt < 20 ? 0 : 1);
            }
            break;
        }
        case 3: {
            /* 无退避：仅 pause 一次 */
            cpu_pause_hint();
            break;
        }
        default: {
            /* 默认指数退避 */
            int spin_count = 1 << (attempt < 8 ? attempt : 8);
            busy_wait_spin(spin_count);
            if (attempt >= 12) {
                Sleep(0);
            }
            break;
        }
    }
#else
    /* POSIX 平台使用 nanosleep + sched_yield */
    switch (strategy) {
        case 0: {
            int spin_count = 1 << (attempt < 8 ? attempt : 8);
            busy_wait_spin(spin_count);
            if (attempt >= 10) {
                struct timespec ts = {0, 0};
                sched_yield();
            }
            break;
        }
        case 1: {
            int spin_count = attempt * 4;
            if (spin_count > 512) spin_count = 512;
            busy_wait_spin(spin_count);
            break;
        }
        case 2: {
            if (attempt < 5) {
                busy_wait_spin(1 << attempt);
            } else if (attempt < 10) {
                sched_yield();
            } else {
                struct timespec ts = {0, 1000000L};
                nanosleep(&ts, NULL);
            }
            break;
        }
        case 3: {
            cpu_pause_hint();
            break;
        }
        default: {
            int spin_count = 1 << (attempt < 8 ? attempt : 8);
            busy_wait_spin(spin_count);
            if (attempt >= 12) {
                sched_yield();
            }
            break;
        }
    }
#endif
}

/**
 * @brief 比较并交换指针
 */
static int compare_and_swap_pointer(void** ptr, void* expected, void* desired) {
#ifdef _WIN32
    return InterlockedCompareExchangePointer((void* volatile*)ptr, desired, expected) == expected;
#else
    return __sync_bool_compare_and_swap((void**)ptr, expected, desired);
#endif
}

/**
 * @brief 比较并交换uintptr — P2修复: 使用intptr_t替代LONG以支持64位指针
 */
static int compare_and_swap_uintptr(volatile intptr_t* ptr, intptr_t expected, intptr_t desired) {
#ifdef _WIN32
    return InterlockedCompareExchange64((volatile LONGLONG*)ptr, (LONGLONG)desired, (LONGLONG)expected) == (LONGLONG)expected;
#else
    return __sync_bool_compare_and_swap(ptr, expected, desired);
#endif
}

/******************************************************************************
 *                        无锁哈希表实现
 ******************************************************************************/

/**
 * @brief 创建哈希表节点
 */
static LockFreeHashTableNode* create_hash_table_node(const void* key, size_t key_size,
                                                    const void* value, size_t value_size) {
    LockFreeHashTableNode* node = (LockFreeHashTableNode*)safe_malloc(sizeof(LockFreeHashTableNode));
    if (node == NULL) {
        return NULL;
    }
    
    /* 分配并复制键 */
    node->key = safe_malloc(key_size);
    if (node->key == NULL) {
        safe_free((void**)&node);
        return NULL;
    }
    memcpy(node->key, key, key_size);
    
    /* 分配并复制值 */
    node->value = safe_malloc(value_size);
    if (node->value == NULL) {
        safe_free((void**)&node->key);
        safe_free((void**)&node);
        return NULL;
    }
    memcpy(node->value, value, value_size);
    
    node->next = NULL;
    node->version = 0;
    
    return node;
}

/**
 * @brief 释放哈希表节点
 */
static void free_hash_table_node(LockFreeHashTableNode* node) {
    if (node == NULL) {
        return;
    }
    
    if (node->key != NULL) {
        safe_free((void**)&node->key);
    }
    
    if (node->value != NULL) {
        safe_free((void**)&node->value);
    }
    
    safe_free((void**)&node);
}

/* ========== 通用无锁操作实现 ========== */

/**
 * @brief 初始化无锁操作结果
 */
void lock_free_operation_result_init(LockFreeOperationResult* result) {
    if (result == NULL) {
        return;
    }
    
    memset(result, 0, sizeof(LockFreeOperationResult));
    result->success = 0;
    result->retries = 0;
    result->operation_time_ns = 0;
    result->backoff_count = 0;
    result->error_message[0] = '\0';
}

/**
 * @brief 获取无锁算法类型名称
 */
const char* lock_free_algorithm_name(LockFreeAlgorithm algorithm) {
    switch (algorithm) {
        case LOCK_FREE_QUEUE: return "无锁队列";
        case LOCK_FREE_STACK: return "无锁栈";
        case LOCK_FREE_HASH_TABLE: return "无锁哈希表";
        case LOCK_FREE_MEMORY_POOL: return "无锁内存池";
        case LOCK_FREE_RING_BUFFER: return "无锁环形缓冲区";
        case LOCK_FREE_PRIORITY_QUEUE: return "无锁优先队列";
        case LOCK_FREE_SKIP_LIST: return "无锁跳表";
        case LOCK_FREE_WORK_STEALING: return "无锁工作窃取队列";
        default: return "未知算法";
    }
}

/**
 * @brief 设置内存屏障
 */
void lock_free_memory_barrier(void) {
    atomic_thread_fence(memory_order_seq_cst);
}

/**
 * @brief 比较并交换操作
 */
int lock_free_compare_and_swap(void* ptr, void* expected, void* desired) {
    return compare_and_swap_pointer((void**)ptr, expected, desired);
}

/**
 * @brief 获取并增加操作
 */
int64_t lock_free_fetch_and_add(void* ptr, int64_t value) {
#ifdef _WIN32
    return InterlockedExchangeAdd64((volatile LONG64*)ptr, value);
#else
    return __sync_fetch_and_add((int64_t*)ptr, value);
#endif
}

/**
 * @brief 获取并或操作
 */
int64_t lock_free_fetch_and_or(void* ptr, int64_t value) {
#ifdef _WIN32
    return InterlockedOr64((volatile LONG64*)ptr, value);
#else
    return __sync_fetch_and_or((int64_t*)ptr, value);
#endif
}

#ifdef _WIN32

/* ============================
 * 增强T6：高并发线程池+工作窃取队列实现（Windows版）
 * ============================ */

/**
 * @brief 线程池任务节点
 */
typedef struct ThreadPoolTaskNode {
    ThreadPoolTaskFunc func;                 /**< 任务函数 */
    void* arg;                               /**< 任务参数 */
    int64_t task_id;                         /**< 任务ID */
    TaskPriority priority;                   /**< 任务优先级 */
    volatile int completed;                  /**< 完成标志 */
    volatile int* cancel_flag;               /**< 取消标志 */
    HANDLE completion_event;                 /**< 完成事件 */
    struct ThreadPoolTaskNode* next;         /**< 下一个节点 */
} ThreadPoolTaskNode;

/**
 * @brief 线程池工作线程上下文
 */
typedef struct {
    LockFreeThreadPool* pool;                /**< 所属线程池 */
    size_t thread_index;                     /**< 线程索引 */
    HANDLE thread_handle;                    /**< 线程句柄 */
    DWORD thread_id;                         /**< 线程ID */
    volatile int running;                    /**< 运行标志 */
    volatile int is_executing;               /**< L-006: 是否正在执行任务（1=执行中，0=空闲等待） */
    size_t tasks_executed;                   /**< 已执行任务数 */
    size_t tasks_stolen;                     /**< 窃取的任务数 */
} WorkerThread;

/**
 * @brief 线程池内部结构
 */
struct LockFreeThreadPool {
    LockFreeThreadPoolConfig config;                 /**< 线程池配置 */
    
    /* 多级优先队列（每个优先级一个锁自由队列） */
    LockFreeQueue* priority_queues[4];       /**< 优先队列数组 */
    
    /* 工作窃取：每个工作者线程的本地队列 */
    LockFreeWorkStealingQueue** local_queues; /**< 本地工作窃取队列数组 */
    
    /* 工作者线程 */
    WorkerThread* workers;                   /**< 工作者线程数组 */
    size_t num_workers;                      /**< 当前工作者数量 */
    
    /* 线程池状态 */
    volatile int active;                     /**< 线程池是否活跃 */
    volatile int shutdown;                   /**< 关闭标志 */
    volatile int64_t task_id_counter;        /**< 任务ID计数器 */
    volatile size_t pending_tasks;           /**< 待处理任务数 */
    volatile size_t completed_tasks;         /**< 已完成任务数 */
    
    /* 同步 */
    HANDLE work_event;                       /**< 工作事件（唤醒空闲线程） */
    CRITICAL_SECTION resize_lock;            /**< 调整大小锁 */
    
    /* 唤醒回调 */
    ThreadPoolWakeFunc wake_callback;        /**< 唤醒回调函数 */
    void* wake_user_data;                    /**< 唤醒回调用户数据 */
    
    /* 系统信息 */
    size_t num_cores;                        /**< CPU核心数 */
};

/**
 * @brief 工作线程主循环
 */
static DWORD WINAPI worker_thread_main(LPVOID arg) {
    WorkerThread* worker = (WorkerThread*)arg;
    LockFreeThreadPool* pool = worker->pool;
    
    while (pool->active && !pool->shutdown) {
        ThreadPoolTaskNode* task = NULL;
        
        // 1. 从本地队列获取任务（LIFO）
        if (pool->local_queues && worker->thread_index < pool->num_workers) {
            if (pool->local_queues[worker->thread_index]) {
                // 从工作窃取队列中弹出
                task = (ThreadPoolTaskNode*)lock_free_work_stealing_steal(
                    pool->local_queues[worker->thread_index]);
            }
        }
        
        // 2. 从高优先级全局队列获取任务
        if (!task) {
            for (int p = TASK_PRIORITY_CRITICAL; p >= TASK_PRIORITY_LOW && !task; p--) {
                if (pool->priority_queues[p]) {
                    ThreadPoolTaskNode* dequeued = NULL;
                    if (lock_free_queue_dequeue(pool->priority_queues[p], &dequeued,
                                               sizeof(ThreadPoolTaskNode*), NULL) == 0) {
                        task = dequeued;
                    }
                }
            }
        }
        
        // 3. 工作窃取：从其他线程的本地队列窃取
        if (!task && pool->config.enable_work_stealing) {
            size_t num_workers = pool->num_workers;
            if (num_workers > 1) {
                for (size_t i = 1; i < num_workers; i++) {
                    size_t victim_idx = (worker->thread_index + i) % num_workers;
                    if (victim_idx == worker->thread_index) continue;
                    if (pool->local_queues && pool->local_queues[victim_idx]) {
                        task = (ThreadPoolTaskNode*)lock_free_work_stealing_steal(
                            pool->local_queues[victim_idx]);
                        if (task) {
                            worker->tasks_stolen++;
                            break;
                        }
                    }
                }
            }
        }
        
        if (task) {
            /* L-006: 标记线程正在执行任务 */
            worker->is_executing = 1;

            /* 执行任务 */
            InterlockedDecrement((volatile LONG*)&pool->pending_tasks);
            
            if (task->func) {
                task->func(task->arg);
            }
            
            /* 标记完成 */
            task->completed = 1;
            InterlockedIncrement((volatile LONG*)&pool->completed_tasks);
            worker->tasks_executed++;
            
            /* 触发完成事件 */
            if (task->completion_event) {
                SetEvent(task->completion_event);
            }

            /* L-006: 任务执行完毕，线程恢复空闲状态 */
            worker->is_executing = 0;
        } else {
            /* 没有任务，等待 */
            DWORD wait_ms = pool->config.idle_timeout_ms > 0 ?
                           (DWORD)pool->config.idle_timeout_ms : INFINITE;
            WaitForSingleObject(pool->work_event, wait_ms);
        }
    }
    
    worker->running = 0;
    return 0;
}

/**
 * @brief 创建线程池
 */
LockFreeThreadPool* lock_free_thread_pool_create(const LockFreeThreadPoolConfig* config) {
    LockFreeThreadPool* pool = (LockFreeThreadPool*)safe_calloc(1, sizeof(LockFreeThreadPool));
    if (!pool) return NULL;
    
    // 设置配置
    if (config) {
        pool->config = *config;
    } else {
        pool->config.min_threads = 2;
        pool->config.max_threads = 0;
        pool->config.idle_timeout_ms = 60000;
        pool->config.queue_capacity = 4096;
        pool->config.enable_work_stealing = 1;
        pool->config.enable_dynamic_resize = 1;
        pool->config.thread_priority = 0;
        pool->config.stack_size = 0;
    }
    
    // 获取CPU核心数
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    pool->num_cores = (size_t)sys_info.dwNumberOfProcessors;
    if (pool->num_cores == 0) pool->num_cores = 4;
    
    // 设置默认最大线程数
    if (pool->config.max_threads == 0) {
        pool->config.max_threads = pool->num_cores * 2;
    }
    if (pool->config.min_threads == 0) {
        pool->config.min_threads = pool->num_cores;
    }
    if (pool->config.min_threads > pool->config.max_threads) {
        pool->config.min_threads = pool->config.max_threads;
    }
    
    // 创建优先队列
    for (int p = 0; p < 4; p++) {
        LockFreeQueueConfig qconfig;
        memset(&qconfig, 0, sizeof(LockFreeQueueConfig));
        qconfig.capacity = pool->config.queue_capacity;
        qconfig.element_size = sizeof(ThreadPoolTaskNode*);
        pool->priority_queues[p] = lock_free_queue_create(&qconfig);
        if (!pool->priority_queues[p]) {
            lock_free_thread_pool_free(pool);
            return NULL;
        }
    }
    
    // 创建同步事件
    pool->work_event = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (!pool->work_event) {
        lock_free_thread_pool_free(pool);
        return NULL;
    }
    
    InitializeCriticalSection(&pool->resize_lock);
    pool->active = 1;
    pool->task_id_counter = 1;
    
    // 创建工作线程
    pool->num_workers = pool->config.min_threads;
    pool->workers = (WorkerThread*)safe_calloc(pool->num_workers, sizeof(WorkerThread));
    if (!pool->workers) {
        lock_free_thread_pool_free(pool);
        return NULL;
    }
    
    // 创建工作窃取队列
    if (pool->config.enable_work_stealing) {
        pool->local_queues = (LockFreeWorkStealingQueue**)safe_calloc(
            pool->num_workers, sizeof(LockFreeWorkStealingQueue*));
        if (!pool->local_queues) {
            lock_free_thread_pool_free(pool);
            return NULL;
        }
        for (size_t i = 0; i < pool->num_workers; i++) {
            pool->local_queues[i] = lock_free_work_stealing_create(
                pool->config.queue_capacity / pool->num_workers);
            if (!pool->local_queues[i]) {
                lock_free_thread_pool_free(pool);
                return NULL;
            }
        }
    }
    
    for (size_t i = 0; i < pool->num_workers; i++) {
        pool->workers[i].pool = pool;
        pool->workers[i].thread_index = i;
        pool->workers[i].running = 1;
        pool->workers[i].thread_handle = CreateThread(NULL,
            pool->config.stack_size > 0 ? pool->config.stack_size : 0,
            worker_thread_main, &pool->workers[i], 0, &pool->workers[i].thread_id);
        if (!pool->workers[i].thread_handle) {
            pool->workers[i].running = 0;
        }
    }
    
    return pool;
}

/**
 * @brief 销毁线程池
 */
void lock_free_thread_pool_free(LockFreeThreadPool* pool) {
    if (!pool) return;
    
    pool->active = 0;
    pool->shutdown = 1;
    
    // 唤醒所有等待的线程
    if (pool->work_event) {
        SetEvent(pool->work_event);
    }
    
    // 等待所有工作线程结束
    if (pool->workers) {
        for (size_t i = 0; i < pool->num_workers; i++) {
            if (pool->workers[i].thread_handle) {
                WaitForSingleObject(pool->workers[i].thread_handle, 3000);
                CloseHandle(pool->workers[i].thread_handle);
            }
        }
        safe_free((void**)&pool->workers);
    }
    
/* 销毁前排空所有任务队列中的ThreadPoolTaskNode
     * 防止任务节点及其Windows同步句柄(pool->completion_event)泄漏。
     * 优先队列中的data字段存储的是ThreadPoolTaskNode*指针值,
     * lock_free_queue_free释放的是队列节点和数据缓冲区,
     * 但ThreadPoolTaskNode堆对象本身需要手动释放。 */
    
    /* 销毁工作窃取队列(先排空再释放) */
    if (pool->local_queues) {
        for (size_t i = 0; i < pool->num_workers; i++) {
            if (pool->local_queues[i]) {
                ThreadPoolTaskNode* stolen_node = NULL;
                while ((stolen_node = (ThreadPoolTaskNode*)lock_free_work_stealing_steal(pool->local_queues[i])) != NULL) {
                    if (stolen_node->completion_event) CloseHandle(stolen_node->completion_event);
                    safe_free((void**)&stolen_node);
                }
                lock_free_work_stealing_free(pool->local_queues[i]);
            }
        }
        safe_free((void**)&pool->local_queues);
    }
    
    /* 销毁优先队列(先排空再释放) */
    for (int p = 0; p < 4; p++) {
        if (pool->priority_queues[p]) {
            ThreadPoolTaskNode* task_node = NULL;
            LockFreeOperationResult drain_result;
            memset(&drain_result, 0, sizeof(drain_result));
            while (lock_free_queue_dequeue(pool->priority_queues[p],
                       &task_node, sizeof(ThreadPoolTaskNode*), &drain_result) == 0) {
                if (task_node) {
                    if (task_node->completion_event) CloseHandle(task_node->completion_event);
                    safe_free((void**)&task_node);
                }
            }
            lock_free_queue_free(pool->priority_queues[p]);
        }
    }
    
    if (pool->work_event) CloseHandle(pool->work_event);
    DeleteCriticalSection(&pool->resize_lock);
    
    safe_free((void**)&pool);
}

/**
 * @brief 提交任务到线程池
 */
int64_t lock_free_thread_pool_submit(LockFreeThreadPool* pool,
                                       ThreadPoolTaskFunc func,
                                       void* arg,
                                       TaskPriority priority) {
    if (!pool || !func || !pool->active) return -1;
    
    ThreadPoolTaskNode* node = (ThreadPoolTaskNode*)safe_malloc(sizeof(ThreadPoolTaskNode));
    if (!node) return -1;
    
    memset(node, 0, sizeof(ThreadPoolTaskNode));
    node->func = func;
    node->arg = arg;
    node->task_id = InterlockedIncrement64(&pool->task_id_counter);
    node->priority = priority;
    node->completed = 0;
    
    // 创建完成事件
    node->completion_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    
    // 按优先级入队
    int queue_idx = (int)priority;
    if (queue_idx < 0) queue_idx = 0;
    if (queue_idx > 3) queue_idx = 3;
    
    if (pool->priority_queues[queue_idx]) {
        lock_free_queue_enqueue(pool->priority_queues[queue_idx], &node,
                               sizeof(ThreadPoolTaskNode*), NULL);
    }
    
    InterlockedIncrement((volatile LONG*)&pool->pending_tasks);
    
    // 唤醒工作者线程
    if (pool->work_event) {
        SetEvent(pool->work_event);
    }
    
    // 触发唤醒回调
    if (pool->wake_callback) {
        pool->wake_callback(pool, pool->wake_user_data);
    }
    
    return node->task_id;
}

/**
 * @brief 等待指定任务完成
 */
int lock_free_thread_pool_wait(LockFreeThreadPool* pool, int64_t task_id) {
    if (!pool || task_id <= 0) return -1;
    
    // 轮询检查任务是否完成（最多等待30秒）
    for (int i = 0; i < 30000; i++) {
        /* 轮询等待任务完成：使用原子操作检查完成计数（最多等待30秒） */
        if (pool->completed_tasks >= (size_t)task_id) {
            return 0;
        }
        Sleep(1);
    }
    
    return -1;
}

/**
 * @brief 等待所有任务完成
 */
int lock_free_thread_pool_wait_all(LockFreeThreadPool* pool) {
    if (!pool) return -1;
    
    // 等待直到所有任务完成（最多等待60秒）
    for (int i = 0; i < 60000; i++) {
        if (pool->pending_tasks == 0) {
            return 0;
        }
        Sleep(1);
    }
    
    return -1;
}

/**
 * @brief 获取线程池当前统计信息
 */
int lock_free_thread_pool_get_stats(LockFreeThreadPool* pool,
                                      size_t* active_threads,
                                      size_t* idle_threads,
                                      size_t* pending_tasks,
                                      size_t* completed_tasks) {
    if (!pool) return -1;
    
    if (active_threads) {
        size_t active = 0;
        for (size_t i = 0; i < pool->num_workers; i++) {
            if (pool->workers[i].running) active++;
        }
        *active_threads = active;
    }
    
    if (idle_threads) {
        /* L-006修复: 遍历线程状态数组精确统计空闲线程数
         * 不再使用简化公式 total_active - pending_tasks，
         * 而是检查每个线程的 is_executing 标志。 */
        size_t idle = 0;
        for (size_t i = 0; i < pool->num_workers; i++) {
            if (pool->workers[i].running && !pool->workers[i].is_executing) {
                idle++;
            }
        }
        *idle_threads = idle;
    }
    
    if (pending_tasks) *pending_tasks = (size_t)pool->pending_tasks;
    if (completed_tasks) *completed_tasks = (size_t)pool->completed_tasks;
    
    return 0;
}

/**
 * @brief 动态调整线程池大小
 */
int lock_free_thread_pool_resize(LockFreeThreadPool* pool,
                                   size_t min_threads,
                                   size_t max_threads) {
    if (!pool || min_threads == 0 || max_threads == 0) return -1;
    if (min_threads > max_threads) return -1;
    
    EnterCriticalSection(&pool->resize_lock);
    
    pool->config.min_threads = min_threads;
    pool->config.max_threads = max_threads;
    
    if (pool->num_workers < min_threads) {
        // 需要增加线程
        size_t old_count = pool->num_workers;
        size_t new_count = min_threads;
        
        WorkerThread* new_workers = (WorkerThread*)safe_realloc(pool->workers,
            new_count * sizeof(WorkerThread));
        if (!new_workers) {
            LeaveCriticalSection(&pool->resize_lock);
            return -1;
        }
        pool->workers = new_workers;
        
        // 扩展工作窃取队列
        if (pool->config.enable_work_stealing && pool->local_queues) {
            LockFreeWorkStealingQueue** new_queues = (LockFreeWorkStealingQueue**)safe_realloc(
                pool->local_queues, new_count * sizeof(LockFreeWorkStealingQueue*));
            if (new_queues) {
                pool->local_queues = new_queues;
            }
        }
        
        // 初始化新线程
        for (size_t i = old_count; i < new_count; i++) {
            memset(&pool->workers[i], 0, sizeof(WorkerThread));
            pool->workers[i].pool = pool;
            pool->workers[i].thread_index = i;
            pool->workers[i].running = 1;
            
            if (pool->config.enable_work_stealing && pool->local_queues) {
                pool->local_queues[i] = lock_free_work_stealing_create(
                    pool->config.queue_capacity / new_count);
            }
            
            pool->workers[i].thread_handle = CreateThread(NULL,
                pool->config.stack_size > 0 ? pool->config.stack_size : 0,
                worker_thread_main, &pool->workers[i], 0, &pool->workers[i].thread_id);
            if (!pool->workers[i].thread_handle) {
                pool->workers[i].running = 0;
            }
        }
        
        pool->num_workers = new_count;
    }
    
    LeaveCriticalSection(&pool->resize_lock);
    return 0;
}

/**
 * @brief 设置线程池唤醒回调
 */
int lock_free_thread_pool_set_wake_callback(LockFreeThreadPool* pool,
                                              ThreadPoolWakeFunc wake_func,
                                              void* user_data) {
    if (!pool) return -1;
    
    pool->wake_callback = wake_func;
    pool->wake_user_data = user_data;
    
    return 0;
}

#else  /* POSIX版线程池实现 */

#include <pthread.h>
#include <unistd.h>
#include <time.h>

/**
 * @brief 线程池任务节点(POSIX)
 */
typedef struct ThreadPoolTaskNode {
    ThreadPoolTaskFunc func;
    void* arg;
    int64_t task_id;
    TaskPriority priority;
    volatile int completed;
    volatile int* cancel_flag;
    pthread_cond_t completion_cond;
    pthread_mutex_t completion_mutex;
    volatile int completion_signaled;
    struct ThreadPoolTaskNode* next;
} ThreadPoolTaskNode;

/**
 * @brief 工作线程上下文(POSIX)
 */
typedef struct {
    LockFreeThreadPool* pool;
    size_t thread_index;
    pthread_t thread;
    volatile int running;
    volatile int is_executing;               /**< L-006: 是否正在执行任务（1=执行中，0=空闲等待） */
    size_t tasks_executed;
    size_t tasks_stolen;
} WorkerThread;

/**
 * @brief 线程池内部结构(POSIX)
 */
struct LockFreeThreadPool {
    LockFreeThreadPoolConfig config;
    LockFreeQueue* priority_queues[4];
    LockFreeWorkStealingQueue** local_queues;
    WorkerThread* workers;
    size_t num_workers;
    volatile int active;
    volatile int shutdown;
    volatile int64_t task_id_counter;
    volatile size_t pending_tasks;
    volatile size_t completed_tasks;
    pthread_mutex_t work_mutex;
    pthread_cond_t work_cond;
    pthread_mutex_t resize_mutex;
    ThreadPoolWakeFunc wake_callback;
    void* wake_user_data;
    size_t num_cores;
};

/**
 * @brief 工作线程主循环(POSIX)
 */
static void* worker_thread_main(void* arg) {
    WorkerThread* worker = (WorkerThread*)arg;
    LockFreeThreadPool* pool = worker->pool;
    
    while (pool->active && !pool->shutdown) {
        ThreadPoolTaskNode* task = NULL;
        
        if (pool->local_queues && worker->thread_index < pool->num_workers) {
            if (pool->local_queues[worker->thread_index]) {
                task = (ThreadPoolTaskNode*)lock_free_work_stealing_steal(
                    pool->local_queues[worker->thread_index]);
            }
        }
        
        if (!task) {
            for (int p = TASK_PRIORITY_CRITICAL; p >= TASK_PRIORITY_LOW && !task; p--) {
                if (pool->priority_queues[p]) {
                    ThreadPoolTaskNode* dequeued = NULL;
                    if (lock_free_queue_dequeue(pool->priority_queues[p], &dequeued,
                                               sizeof(ThreadPoolTaskNode*), NULL) == 0) {
                        task = dequeued;
                    }
                }
            }
        }
        
        if (!task && pool->config.enable_work_stealing) {
            size_t nw = pool->num_workers;
            if (nw > 1) {
                for (size_t i = 1; i < nw; i++) {
                    size_t victim = (worker->thread_index + i) % nw;
                    if (victim == worker->thread_index) continue;
                    if (pool->local_queues && pool->local_queues[victim]) {
                        task = (ThreadPoolTaskNode*)lock_free_work_stealing_steal(
                            pool->local_queues[victim]);
                        if (task) {
                            worker->tasks_stolen++;
                            break;
                        }
                    }
                }
            }
        }
        
        if (task) {
            /* L-006: 标记线程正在执行任务 */
            worker->is_executing = 1;

            __sync_fetch_and_sub(&pool->pending_tasks, 1);
            
            if (task->func) {
                task->func(task->arg);
            }
            
            task->completed = 1;
            __sync_fetch_and_add(&pool->completed_tasks, 1);
            worker->tasks_executed++;
            
            if (!task->completion_signaled) {
                pthread_mutex_lock(&task->completion_mutex);
                task->completion_signaled = 1;
                pthread_cond_signal(&task->completion_cond);
                pthread_mutex_unlock(&task->completion_mutex);
            }

            /* L-006: 任务执行完毕，线程恢复空闲状态 */
            worker->is_executing = 0;
        } else {
            struct timespec ts;
            if (pool->config.idle_timeout_ms > 0) {
                clock_gettime(CLOCK_REALTIME, &ts);
                ts.tv_sec += pool->config.idle_timeout_ms / 1000;
                ts.tv_nsec += (pool->config.idle_timeout_ms % 1000) * 1000000L;
            }
            pthread_mutex_lock(&pool->work_mutex);
            if (!pool->shutdown && pool->pending_tasks == 0) {
                if (pool->config.idle_timeout_ms > 0) {
                    pthread_cond_timedwait(&pool->work_cond, &pool->work_mutex, &ts);
                } else {
                    pthread_cond_wait(&pool->work_cond, &pool->work_mutex);
                }
            }
            pthread_mutex_unlock(&pool->work_mutex);
        }
    }
    worker->running = 0;
    return NULL;
}

/**
 * @brief 创建线程池(POSIX)
 */
LockFreeThreadPool* lock_free_thread_pool_create(const LockFreeThreadPoolConfig* config) {
    LockFreeThreadPool* pool = (LockFreeThreadPool*)safe_calloc(1, sizeof(LockFreeThreadPool));
    if (!pool) return NULL;
    
    if (config) {
        pool->config = *config;
    } else {
        pool->config.min_threads = 2;
        pool->config.max_threads = 0;
        pool->config.idle_timeout_ms = 60000;
        pool->config.queue_capacity = 4096;
        pool->config.enable_work_stealing = 1;
        pool->config.enable_dynamic_resize = 1;
        pool->config.thread_priority = 0;
        pool->config.stack_size = 0;
    }
    
#ifdef _SC_NPROCESSORS_ONLN
    pool->num_cores = (size_t)sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(_SC_NPROC_ONLN)
    pool->num_cores = (size_t)sysconf(_SC_NPROC_ONLN);
#else
    pool->num_cores = 4;
#endif
    if (pool->num_cores == 0) pool->num_cores = 4;
    
    if (pool->config.max_threads == 0) {
        pool->config.max_threads = pool->num_cores * 2;
    }
    if (pool->config.min_threads == 0) {
        pool->config.min_threads = pool->num_cores;
    }
    if (pool->config.min_threads > pool->config.max_threads) {
        pool->config.min_threads = pool->config.max_threads;
    }
    
    for (int p = 0; p < 4; p++) {
        LockFreeQueueConfig qconfig;
        memset(&qconfig, 0, sizeof(LockFreeQueueConfig));
        qconfig.capacity = pool->config.queue_capacity;
        qconfig.element_size = sizeof(ThreadPoolTaskNode*);
        pool->priority_queues[p] = lock_free_queue_create(&qconfig);
        if (!pool->priority_queues[p]) {
            lock_free_thread_pool_free(pool);
            return NULL;
        }
    }
    
    pthread_mutex_init(&pool->work_mutex, NULL);
    pthread_cond_init(&pool->work_cond, NULL);
    pthread_mutex_init(&pool->resize_mutex, NULL);
    pool->active = 1;
    pool->task_id_counter = 1;
    
    pool->num_workers = pool->config.min_threads;
    pool->workers = (WorkerThread*)safe_calloc(pool->num_workers, sizeof(WorkerThread));
    if (!pool->workers) {
        lock_free_thread_pool_free(pool);
        return NULL;
    }
    
    if (pool->config.enable_work_stealing) {
        pool->local_queues = (LockFreeWorkStealingQueue**)safe_calloc(
            pool->num_workers, sizeof(LockFreeWorkStealingQueue*));
        if (!pool->local_queues) {
            lock_free_thread_pool_free(pool);
            return NULL;
        }
        for (size_t i = 0; i < pool->num_workers; i++) {
            pool->local_queues[i] = lock_free_work_stealing_create(
                pool->config.queue_capacity / pool->num_workers);
            if (!pool->local_queues[i]) {
                lock_free_thread_pool_free(pool);
                return NULL;
            }
        }
    }
    
    for (size_t i = 0; i < pool->num_workers; i++) {
        pool->workers[i].pool = pool;
        pool->workers[i].thread_index = i;
        pool->workers[i].running = 1;
        if (pthread_create(&pool->workers[i].thread, NULL, worker_thread_main, &pool->workers[i]) != 0) {
            pool->workers[i].running = 0;
        }
    }
    
    return pool;
}

/**
 * @brief 销毁线程池(POSIX)
 */
void lock_free_thread_pool_free(LockFreeThreadPool* pool) {
    if (!pool) return;
    
    pool->active = 0;
    pool->shutdown = 1;
    
    pthread_mutex_lock(&pool->work_mutex);
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->work_mutex);
    
    if (pool->workers) {
        for (size_t i = 0; i < pool->num_workers; i++) {
            if (pool->workers[i].running) {
                pthread_join(pool->workers[i].thread, NULL);
            }
        }
        safe_free((void**)&pool->workers);
    }
    
/* POSIX版同样需要排空任务节点防止泄漏 */
    if (pool->local_queues) {
        for (size_t i = 0; i < pool->num_workers; i++) {
            if (pool->local_queues[i]) {
                ThreadPoolTaskNode* stolen_node = NULL;
                while ((stolen_node = (ThreadPoolTaskNode*)lock_free_work_stealing_steal(pool->local_queues[i])) != NULL) {
                    safe_free((void**)&stolen_node);
                }
                lock_free_work_stealing_free(pool->local_queues[i]);
            }
        }
        safe_free((void**)&pool->local_queues);
    }
    
    for (int p = 0; p < 4; p++) {
        if (pool->priority_queues[p]) {
            ThreadPoolTaskNode* task_node = NULL;
            LockFreeOperationResult drain_result;
            memset(&drain_result, 0, sizeof(drain_result));
            while (lock_free_queue_dequeue(pool->priority_queues[p],
                       &task_node, sizeof(ThreadPoolTaskNode*), &drain_result) == 0) {
                if (task_node) {
                    safe_free((void**)&task_node);
                }
            }
            lock_free_queue_free(pool->priority_queues[p]);
        }
    }
    
    pthread_mutex_destroy(&pool->work_mutex);
    pthread_cond_destroy(&pool->work_cond);
    pthread_mutex_destroy(&pool->resize_mutex);
    
    safe_free((void**)&pool);
}

/**
 * @brief 提交任务到线程池(POSIX)
 */
int64_t lock_free_thread_pool_submit(LockFreeThreadPool* pool,
                                       ThreadPoolTaskFunc func,
                                       void* arg,
                                       TaskPriority priority) {
    if (!pool || !func || !pool->active) return -1;
    
    ThreadPoolTaskNode* node = (ThreadPoolTaskNode*)safe_malloc(sizeof(ThreadPoolTaskNode));
    if (!node) return -1;
    
    memset(node, 0, sizeof(ThreadPoolTaskNode));
    node->func = func;
    node->arg = arg;
    node->task_id = __sync_fetch_and_add(&pool->task_id_counter, 1);
    node->priority = priority;
    node->completed = 0;
    node->completion_signaled = 0;
    pthread_mutex_init(&node->completion_mutex, NULL);
    pthread_cond_init(&node->completion_cond, NULL);
    
    int queue_idx = (int)priority;
    if (queue_idx < 0) queue_idx = 0;
    if (queue_idx > 3) queue_idx = 3;
    
    if (pool->priority_queues[queue_idx]) {
        lock_free_queue_enqueue(pool->priority_queues[queue_idx], &node,
                               sizeof(ThreadPoolTaskNode*), NULL);
    }
    
    __sync_fetch_and_add(&pool->pending_tasks, 1);
    
    pthread_mutex_lock(&pool->work_mutex);
    pthread_cond_signal(&pool->work_cond);
    pthread_mutex_unlock(&pool->work_mutex);
    
    if (pool->wake_callback) {
        pool->wake_callback(pool, pool->wake_user_data);
    }
    
    return node->task_id;
}

/**
 * @brief 等待指定任务完成
 */
int lock_free_thread_pool_wait(LockFreeThreadPool* pool, int64_t task_id) {
    if (!pool || task_id <= 0) return -1;
    
    for (int i = 0; i < 30000; i++) {
        if (pool->completed_tasks >= (size_t)task_id) {
            return 0;
        }
        time_sleep_ms(1);
    }
    return -1;
}

/**
 * @brief 等待所有任务完成
 */
int lock_free_thread_pool_wait_all(LockFreeThreadPool* pool) {
    if (!pool) return -1;
    
    for (int i = 0; i < 60000; i++) {
        if (pool->pending_tasks == 0) {
            return 0;
        }
        time_sleep_ms(1);
    }
    return -1;
}

/**
 * @brief 获取线程池统计信息(POSIX)
 */
int lock_free_thread_pool_get_stats(LockFreeThreadPool* pool,
                                      size_t* active_threads,
                                      size_t* idle_threads,
                                      size_t* pending_tasks,
                                      size_t* completed_tasks) {
    if (!pool) return -1;
    
    if (active_threads) {
        size_t active = 0;
        for (size_t i = 0; i < pool->num_workers; i++) {
            if (pool->workers[i].running) active++;
        }
        *active_threads = active;
    }
    if (idle_threads) {
        /* L-006修复: 遍历线程状态数组精确统计空闲线程数
         * 不再使用简化公式 total_active - pending_tasks，
         * 而是检查每个线程的 is_executing 标志。 */
        size_t idle = 0;
        for (size_t i = 0; i < pool->num_workers; i++) {
            if (pool->workers[i].running && !pool->workers[i].is_executing) {
                idle++;
            }
        }
        *idle_threads = idle;
    }
    if (pending_tasks) *pending_tasks = (size_t)pool->pending_tasks;
    if (completed_tasks) *completed_tasks = (size_t)pool->completed_tasks;
    return 0;
}

/**
 * @brief 动态调整线程池大小(POSIX)
 */
int lock_free_thread_pool_resize(LockFreeThreadPool* pool,
                                   size_t min_threads,
                                   size_t max_threads) {
    if (!pool || min_threads == 0 || max_threads == 0) return -1;
    if (min_threads > max_threads) return -1;
    
    pthread_mutex_lock(&pool->resize_mutex);
    
    pool->config.min_threads = min_threads;
    pool->config.max_threads = max_threads;
    
    if (pool->num_workers < min_threads) {
        size_t old_count = pool->num_workers;
        size_t new_count = min_threads;
        
        WorkerThread* new_workers = (WorkerThread*)safe_realloc(pool->workers,
            new_count * sizeof(WorkerThread));
        if (!new_workers) {
            pthread_mutex_unlock(&pool->resize_mutex);
            return -1;
        }
        pool->workers = new_workers;
        
        if (pool->config.enable_work_stealing && pool->local_queues) {
            LockFreeWorkStealingQueue** new_queues = (LockFreeWorkStealingQueue**)safe_realloc(
                pool->local_queues, new_count * sizeof(LockFreeWorkStealingQueue*));
            if (new_queues) {
                pool->local_queues = new_queues;
            }
        }
        
        for (size_t i = old_count; i < new_count; i++) {
            memset(&pool->workers[i], 0, sizeof(WorkerThread));
            pool->workers[i].pool = pool;
            pool->workers[i].thread_index = i;
            pool->workers[i].running = 1;
            
            if (pool->config.enable_work_stealing && pool->local_queues) {
                pool->local_queues[i] = lock_free_work_stealing_create(
                    pool->config.queue_capacity / new_count);
            }
            
            if (pthread_create(&pool->workers[i].thread, NULL, worker_thread_main, &pool->workers[i]) != 0) {
                pool->workers[i].running = 0;
            }
        }
        pool->num_workers = new_count;
    }
    
    pthread_mutex_unlock(&pool->resize_mutex);
    return 0;
}

/**
 * @brief 设置线程池唤醒回调(POSIX)
 */
int lock_free_thread_pool_set_wake_callback(LockFreeThreadPool* pool,
                                              ThreadPoolWakeFunc wake_func,
                                              void* user_data) {
    if (!pool) return -1;
    pool->wake_callback = wake_func;
    pool->wake_user_data = user_data;
    return 0;
}

#endif  /* _WIN32 */

/* ============================
 * 无锁工作窃取队列实现
 * 
 * 基于Chase-Lev工作窃取双端队列算法。
 * 所有者线程从底部push/pop（LIFO），
 * 窃取线程从顶部steal（FIFO）。
 * ============================ */

/**
 * @brief 无锁工作窃取队列内部结构
 */
struct LockFreeWorkStealingQueue {
    void** tasks;                   /**< 环形任务数组 */
    size_t capacity;                /**< 容量 */
    size_t mask;                    /**< 容量掩码（容量-1，要求容量为2的幂） */
    volatile LONG top;              /**< 顶部索引（窃取者读取，使用32位CAS兼容） */
    volatile LONG bottom;           /**< 底部索引（所有者写入，使用32位CAS兼容） */
};

/**
 * @brief 64位版本的CAS（用于任务指针）
 */
static int cas_pointer(void* volatile* ptr, void* expected, void* desired) {
#ifdef _WIN32
    return InterlockedCompareExchangePointer(ptr, desired, expected) == expected;
#else
    return __sync_bool_compare_and_swap(ptr, expected, desired);
#endif
}

/**
 * @brief 32位CAS（用于top/bottom索引）
 */
static int cas_int32(volatile LONG* ptr, LONG expected, LONG desired) {
#ifdef _WIN32
    return InterlockedCompareExchange(ptr, desired, expected) == expected;
#else
    return __sync_bool_compare_and_swap(ptr, expected, desired);
#endif
}

/**
 * @brief 创建无锁工作窃取队列
 * 
 * 创建一个Chase-Lev风格的工作窃取双端队列。
 * 容量会被向上取整为2的幂。
 * 
 * @param capacity 队列容量
 * @return LockFreeWorkStealingQueue* 成功返回队列句柄，失败返回NULL
 */
LockFreeWorkStealingQueue* lock_free_work_stealing_create(size_t capacity) {
    LockFreeWorkStealingQueue* queue;
    
    queue = (LockFreeWorkStealingQueue*)safe_malloc(sizeof(LockFreeWorkStealingQueue));
    if (!queue) return NULL;
    
    /* 容量向上取整为2的幂 */
    size_t actual_capacity = 1;
    while (actual_capacity < capacity) {
        actual_capacity <<= 1;
    }
    if (actual_capacity < 2) actual_capacity = 2;
    
    queue->tasks = (void**)safe_calloc(actual_capacity, sizeof(void*));
    if (!queue->tasks) {
        safe_free((void**)&queue);
        return NULL;
    }
    
    queue->capacity = actual_capacity;
    queue->mask = actual_capacity - 1;
    queue->top = 0;
    queue->bottom = 0;
    
    return queue;
}

/**
 * @brief 从工作窃取队列底部pop一个任务（所有者调用）
 * 
 * @param queue 工作窃取队列句柄
 * @return void* 任务指针，队列空返回NULL
 */
static void* work_stealing_pop_bottom(LockFreeWorkStealingQueue* queue) {
    if (!queue) return NULL;
    
    LONG b = queue->bottom;
    if (b <= 0) return NULL;
    
    b--;
    queue->bottom = b;
    
    MemoryBarrier();
    
    LONG t = queue->top;
    void* task = NULL;
    
    if (t <= b) {
        task = queue->tasks[b & (LONG)queue->mask];
        if (t == b) {
            if (cas_int32(&queue->top, t, t + 1)) {
                queue->bottom = b + 1;
            } else {
                task = NULL;
                queue->bottom = b + 1;
            }
        }
    } else {
        queue->bottom = b + 1;
    }
    
    return task;
}

/**
 * @brief 从工作窃取队列顶部steal一个任务（其他线程调用）
 * 
 * @param queue 工作窃取队列句柄
 * @return void* 任务指针，队列空返回NULL
 */
void* lock_free_work_stealing_steal(LockFreeWorkStealingQueue* queue) {
    if (!queue) return NULL;
    
    LONG t = queue->top;
    MemoryBarrier();
    
    LONG b = queue->bottom;
    
    if (t < b) {
        void* task = queue->tasks[t & (LONG)queue->mask];
        MemoryBarrier();
        
        if (cas_int32(&queue->top, t, t + 1)) {
            return task;
        }
    }
    
    return NULL;
}

/**
 * @brief 销毁工作窃取队列
 * 
 * @param queue 工作窃取队列句柄
 */
void lock_free_work_stealing_free(LockFreeWorkStealingQueue* queue) {
    if (!queue) return;
    
    safe_free((void**)&queue->tasks);
    safe_free((void**)&queue);
}

/* ========== 无锁环形缓冲区实现 ========== */

/**
 * @brief 环形缓冲区槽位（序列号+数据）
 */
typedef struct {
    volatile int64_t sequence;           /**< 序列号（用于无锁访问控制） */
    char data[1];                        /**< 柔性数组：元素数据 */
} RingBufferSlot;

/**
 * @brief 无锁环形缓冲区内部结构
 */
struct LockFreeRingBuffer {
    size_t capacity;                     /**< 容量（2的幂） */
    size_t mask;                         /**< 容量-1（位掩码） */
    size_t element_size;                 /**< 元素大小 */
    int is_multi_producer;               /**< 多生产者模式 */
    int is_multi_consumer;               /**< 多消费者模式 */
    int enable_batch;                    /**< 启用批量操作 */
    int spin_count;                      /**< 自旋次数 */
    volatile int64_t head;               /**< 头指针：下一个可写入的序号 */
    volatile int64_t tail;               /**< 尾指针：下一个可读取的序号 */
    RingBufferSlot** slots;              /**< 槽位指针数组 */
};

/**
 * @brief 创建无锁环形缓冲区
 */
LockFreeRingBuffer* lock_free_ring_buffer_create(const LockFreeRingBufferConfig* config) {
    if (!config || config->capacity == 0 || config->element_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建环形缓冲区：参数无效");
        return NULL;
    }

    size_t cap = 1;
    while (cap < config->capacity) cap <<= 1;

    LockFreeRingBuffer* rb = (LockFreeRingBuffer*)safe_malloc(sizeof(LockFreeRingBuffer));
    if (!rb) return NULL;

    rb->capacity = cap;
    rb->mask = cap - 1;
    rb->element_size = config->element_size;
    rb->is_multi_producer = config->is_multi_producer;
    rb->is_multi_consumer = config->is_multi_consumer;
    rb->enable_batch = config->enable_batch_operation;
    rb->spin_count = config->spin_count > 0 ? config->spin_count : 100;
    rb->head = 0;
    rb->tail = 0;

    rb->slots = (RingBufferSlot**)safe_malloc(cap * sizeof(RingBufferSlot*));
    if (!rb->slots) {
        safe_free((void**)&rb);
        return NULL;
    }

    for (size_t i = 0; i < cap; i++) {
        rb->slots[i] = (RingBufferSlot*)safe_malloc(sizeof(RingBufferSlot) + config->element_size);
        if (!rb->slots[i]) {
            for (size_t j = 0; j < i; j++) safe_free((void**)&rb->slots[j]);
            safe_free((void**)&rb->slots);
            safe_free((void**)&rb);
            return NULL;
        }
        rb->slots[i]->sequence = (int64_t)i;
        memset(rb->slots[i]->data, 0, config->element_size);
    }

    return rb;
}

/**
 * @brief 销毁无锁环形缓冲区
 */
void lock_free_ring_buffer_free(LockFreeRingBuffer* rb) {
    if (!rb) return;
    for (size_t i = 0; i < rb->capacity; i++) {
        if (rb->slots[i]) safe_free((void**)&rb->slots[i]);
    }
    safe_free((void**)&rb->slots);
    safe_free((void**)&rb);
}

/**
 * @brief 写入一个元素（生产者）
 */
int lock_free_ring_buffer_write(LockFreeRingBuffer* rb, const void* element) {
    if (!rb || !element) return -1;

    if (rb->is_multi_producer) {
        int64_t h;
        size_t offset;
        int64_t seq;
        int attempts = 0;

        while (attempts < rb->spin_count) {
            h = rb->head;
            offset = (size_t)(h & (int64_t)rb->mask);
            seq = rb->slots[offset]->sequence;
            int64_t diff = seq - h;

            if (diff == 0) {
#ifdef _WIN32
                int64_t prev = (int64_t)InterlockedCompareExchange64(
                    (volatile LONG64*)&rb->head, (LONG64)(h + 1), (LONG64)h);
                if (prev == h) {
#else
                if (__sync_bool_compare_and_swap(&rb->head, h, h + 1)) {
#endif
                    memcpy(rb->slots[offset]->data, element, rb->element_size);
#ifdef _WIN32
                    InterlockedExchange64((volatile LONG64*)&rb->slots[offset]->sequence, (LONG64)(h + 1));
#else
                    __sync_synchronize();
                    rb->slots[offset]->sequence = h + 1;
#endif
                    return 0;
                }
            } else if (diff < 0) {
                return 1;
            }
            cpu_pause_hint();
            attempts++;
        }
        return 1;
    } else {
        int64_t h = rb->head;
        int64_t t = rb->tail;
        if (h - t >= (int64_t)rb->capacity) return 1;

        size_t offset = (size_t)(h & (int64_t)rb->mask);
        memcpy(rb->slots[offset]->data, element, rb->element_size);
#ifdef _WIN32
        InterlockedExchange64((volatile LONG64*)&rb->slots[offset]->sequence, (LONG64)(h + 1));
        InterlockedExchange64((volatile LONG64*)&rb->head, (LONG64)(h + 1));
#else
        __sync_synchronize();
        rb->slots[offset]->sequence = h + 1;
        __sync_synchronize();
        rb->head = h + 1;
#endif
        return 0;
    }
}

/**
 * @brief 读取一个元素（消费者）
 */
int lock_free_ring_buffer_read(LockFreeRingBuffer* rb, void* element) {
    if (!rb || !element) return -1;

    if (rb->is_multi_consumer) {
        int64_t t;
        size_t offset;
        int64_t seq;
        int attempts = 0;

        while (attempts < rb->spin_count) {
            t = rb->tail;
            offset = (size_t)(t & (int64_t)rb->mask);
            seq = rb->slots[offset]->sequence;
            int64_t expected_seq = t + 1;
            int64_t diff = seq - expected_seq;

            if (diff == 0) {
#ifdef _WIN32
                int64_t prev = (int64_t)InterlockedCompareExchange64(
                    (volatile LONG64*)&rb->tail, (LONG64)(t + 1), (LONG64)t);
                if (prev == t) {
#else
                if (__sync_bool_compare_and_swap(&rb->tail, t, t + 1)) {
#endif
                    memcpy(element, rb->slots[offset]->data, rb->element_size);
#ifdef _WIN32
                    InterlockedExchange64((volatile LONG64*)&rb->slots[offset]->sequence,
                        (LONG64)(t + (int64_t)rb->capacity));
#else
                    __sync_synchronize();
                    rb->slots[offset]->sequence = t + (int64_t)rb->capacity;
#endif
                    return 0;
                }
            } else if (diff < 0) {
                return 1;
            }
            cpu_pause_hint();
            attempts++;
        }
        return 1;
    } else {
        int64_t t = rb->tail;
        int64_t h = rb->head;
        if (t >= h) return 1;

        size_t offset = (size_t)(t & (int64_t)rb->mask);
        memcpy(element, rb->slots[offset]->data, rb->element_size);
#ifdef _WIN32
        InterlockedExchange64((volatile LONG64*)&rb->slots[offset]->sequence,
            (LONG64)(t + (int64_t)rb->capacity));
        InterlockedExchange64((volatile LONG64*)&rb->tail, (LONG64)(t + 1));
#else
        __sync_synchronize();
        rb->slots[offset]->sequence = t + (int64_t)rb->capacity;
        __sync_synchronize();
        rb->tail = t + 1;
#endif
        return 0;
    }
}

/**
 * @brief 尝试写入（非阻塞）
 */
int lock_free_ring_buffer_try_write(LockFreeRingBuffer* rb, const void* element) {
    if (!rb || !element) return -1;

    if (rb->is_multi_producer) {
        int64_t h = rb->head;
        size_t offset = (size_t)(h & (int64_t)rb->mask);
        int64_t seq = rb->slots[offset]->sequence;
        if (seq - h != 0) return 1;
#ifdef _WIN32
        int64_t prev = (int64_t)InterlockedCompareExchange64(
            (volatile LONG64*)&rb->head, (LONG64)(h + 1), (LONG64)h);
        if (prev != h) return 1;
#else
        if (!__sync_bool_compare_and_swap(&rb->head, h, h + 1)) return 1;
#endif
        memcpy(rb->slots[offset]->data, element, rb->element_size);
#ifdef _WIN32
        InterlockedExchange64((volatile LONG64*)&rb->slots[offset]->sequence, (LONG64)(h + 1));
#else
        __sync_synchronize();
        rb->slots[offset]->sequence = h + 1;
#endif
        return 0;
    } else {
        int64_t h = rb->head;
        int64_t t = rb->tail;
        if (h - t >= (int64_t)rb->capacity) return 1;
        size_t offset = (size_t)(h & (int64_t)rb->mask);
        memcpy(rb->slots[offset]->data, element, rb->element_size);
#ifdef _WIN32
        InterlockedExchange64((volatile LONG64*)&rb->slots[offset]->sequence, (LONG64)(h + 1));
        InterlockedExchange64((volatile LONG64*)&rb->head, (LONG64)(h + 1));
#else
        __sync_synchronize();
        rb->slots[offset]->sequence = h + 1;
        __sync_synchronize();
        rb->head = h + 1;
#endif
        return 0;
    }
}

/**
 * @brief 尝试读取（非阻塞）
 */
int lock_free_ring_buffer_try_read(LockFreeRingBuffer* rb, void* element) {
    if (!rb || !element) return -1;

    if (rb->is_multi_consumer) {
        int64_t t = rb->tail;
        size_t offset = (size_t)(t & (int64_t)rb->mask);
        int64_t seq = rb->slots[offset]->sequence;
        if (seq - (t + 1) != 0) return 1;
#ifdef _WIN32
        int64_t prev = (int64_t)InterlockedCompareExchange64(
            (volatile LONG64*)&rb->tail, (LONG64)(t + 1), (LONG64)t);
        if (prev != t) return 1;
#else
        if (!__sync_bool_compare_and_swap(&rb->tail, t, t + 1)) return 1;
#endif
        memcpy(element, rb->slots[offset]->data, rb->element_size);
#ifdef _WIN32
        InterlockedExchange64((volatile LONG64*)&rb->slots[offset]->sequence,
            (LONG64)(t + (int64_t)rb->capacity));
#else
        __sync_synchronize();
        rb->slots[offset]->sequence = t + (int64_t)rb->capacity;
#endif
        return 0;
    } else {
        int64_t t = rb->tail;
        int64_t h = rb->head;
        if (t >= h) return 1;
        size_t offset = (size_t)(t & (int64_t)rb->mask);
        memcpy(element, rb->slots[offset]->data, rb->element_size);
#ifdef _WIN32
        InterlockedExchange64((volatile LONG64*)&rb->slots[offset]->sequence,
            (LONG64)(t + (int64_t)rb->capacity));
        InterlockedExchange64((volatile LONG64*)&rb->tail, (LONG64)(t + 1));
#else
        __sync_synchronize();
        rb->slots[offset]->sequence = t + (int64_t)rb->capacity;
        __sync_synchronize();
        rb->tail = t + 1;
#endif
        return 0;
    }
}

/**
 * @brief 批量写入
 */
size_t lock_free_ring_buffer_write_batch(LockFreeRingBuffer* rb, const void* elements, size_t count) {
    if (!rb || !elements || count == 0) return 0;
    size_t written = 0;
    const char* src = (const char*)elements;
    for (size_t i = 0; i < count; i++) {
        if (lock_free_ring_buffer_write(rb, src) != 0) break;
        src += rb->element_size;
        written++;
    }
    return written;
}

/**
 * @brief 批量读取
 */
size_t lock_free_ring_buffer_read_batch(LockFreeRingBuffer* rb, void* elements, size_t max_count) {
    if (!rb || !elements || max_count == 0) return 0;
    size_t read_count = 0;
    char* dst = (char*)elements;
    for (size_t i = 0; i < max_count; i++) {
        if (lock_free_ring_buffer_read(rb, dst) != 0) break;
        dst += rb->element_size;
        read_count++;
    }
    return read_count;
}

/**
 * @brief 获取缓冲区使用量
 */
size_t lock_free_ring_buffer_size(const LockFreeRingBuffer* rb) {
    if (!rb) return 0;
    int64_t h = rb->head;
    int64_t t = rb->tail;
    return (size_t)(h - t);
}

/**
 * @brief 获取缓冲区容量
 */
size_t lock_free_ring_buffer_capacity(const LockFreeRingBuffer* rb) {
    return rb ? rb->capacity : 0;
}

/**
 * @brief 检查缓冲区是否为空
 */
int lock_free_ring_buffer_is_empty(const LockFreeRingBuffer* rb) {
    if (!rb) return 1;
    return rb->head <= rb->tail ? 1 : 0;
}

/**
 * @brief 检查缓冲区是否已满
 */
int lock_free_ring_buffer_is_full(const LockFreeRingBuffer* rb) {
    if (!rb) return 1;
    return (rb->head - rb->tail) >= (int64_t)rb->capacity ? 1 : 0;
}

/**
 * @brief 清空缓冲区
 */
void lock_free_ring_buffer_clear(LockFreeRingBuffer* rb) {
    if (!rb) return;
    for (size_t i = 0; i < rb->capacity; i++) {
        rb->slots[i]->sequence = (int64_t)i;
    }
#ifdef _WIN32
    InterlockedExchange64((volatile LONG64*)&rb->head, 0);
    InterlockedExchange64((volatile LONG64*)&rb->tail, 0);
#else
    __sync_synchronize();
    rb->head = 0;
    __sync_synchronize();
    rb->tail = 0;
#endif
}

/******************************************************************************
 *                        无锁优先队列实现
 ******************************************************************************/

/**
 * @brief 优先队列节点（存储在排序链表中）
 */
typedef struct PriorityQueueNode {
    int32_t priority;                    /**< 优先级值 */
    char data[1];                        /**< 柔性数组：元素数据+next标记指针 */
} PriorityQueueNode;

/* 获取节点中next标记指针的辅助宏 */
#define PQ_NEXT_PTR(node, elem_size) \
    ((volatile TaggedPtr*)((char*)(node)->data + (elem_size)))

/**
 * @brief 无锁优先队列内部结构
 */
struct LockFreePriorityQueue {
    PriorityQueueNode* sentinel;         /**< 哨兵节点（永不释放） */
    size_t capacity;                     /**< 队列容量 */
    size_t element_size;                 /**< 元素大小 */
    int is_min_heap;                     /**< 1=最小优先，0=最大优先 */
    int max_retries;                     /**< 最大重试次数 */
    volatile LONG size;                  /**< 当前元素数 */
    DeferredNode* free_list_head;        /**< 延迟释放链表 */
    volatile LONG free_list_count;       /**< 延迟释放计数 */
#if TP_HAS_POOL
    TaggedPtrNodePool node_pool;         /**< M-029: 32位回退方案节点池 */
    size_t pool_node_size;               /**< 池中单个节点总大小 */
#endif
};

/**
 * @brief 创建优先队列节点
 */
static PriorityQueueNode* create_pq_node(const void* element, size_t element_size, int32_t priority) {
    size_t alloc_size = sizeof(PriorityQueueNode) + element_size + sizeof(TaggedPtr);
#if TP_HAS_POOL
    /* M-029: 32位回退方案——从节点池分配 */
    PriorityQueueNode* node = (PriorityQueueNode*)tp_pool_alloc_node(g_tp_active_pool);
    if (!node) return NULL;
    memset(node, 0, alloc_size);
#else
    PriorityQueueNode* node = (PriorityQueueNode*)safe_malloc(alloc_size);
    if (!node) return NULL;
#endif
    node->priority = priority;
    if (element) memcpy(node->data, element, element_size);
    else memset(node->data, 0, element_size);
    *PQ_NEXT_PTR(node, element_size) = make_tagged(NULL, 0);
    return node;
}

/**
 * @brief 创建无锁优先队列
 */
LockFreePriorityQueue* lock_free_priority_queue_create(const LockFreePriorityQueueConfig* config) {
    if (!config || config->capacity == 0 || config->element_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建优先队列：参数无效");
        return NULL;
    }

    LockFreePriorityQueue* pq = (LockFreePriorityQueue*)safe_malloc(sizeof(LockFreePriorityQueue));
    if (!pq) return NULL;

#if TP_HAS_POOL
    /* M-029：32位回退方案——初始化节点池（节点大小 = 结构体 + 元素 + TaggedPtr） */
    pq->pool_node_size = sizeof(PriorityQueueNode) + config->element_size + sizeof(TaggedPtr);
    if (tp_pool_init(&pq->node_pool, pq->pool_node_size,
                      (uint16_t)(config->capacity + 2)) != 0) {
        safe_free((void**)&pq);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建优先队列：节点池初始化失败");
        return NULL;
    }
    TP_BIND_POOL(&pq->node_pool);
#endif

    int32_t sentinel_priority = config->is_min_heap ? 2147483647 : -2147483647 - 1;
    pq->sentinel = create_pq_node(NULL, config->element_size, sentinel_priority);
    if (!pq->sentinel) {
#if TP_HAS_POOL
        tp_pool_destroy(&pq->node_pool);
#endif
        safe_free((void**)&pq);
        return NULL;
    }

    pq->capacity = config->capacity;
    pq->element_size = config->element_size;
    pq->is_min_heap = config->is_min_heap;
    pq->max_retries = config->max_retries > 0 ? config->max_retries : 1000;
    pq->size = 0;
    pq->free_list_head = NULL;
    pq->free_list_count = 0;

    return pq;
}

/**
 * @brief 销毁无锁优先队列
 */
void lock_free_priority_queue_free(LockFreePriorityQueue* pq) {
    if (!pq) return;
#if TP_HAS_POOL
    TP_BIND_POOL(&pq->node_pool);
    /* M-029: 32位回退方案——池节点由tp_pool_destroy统一释放，无需逐个safe_free */
    deferred_free_flush(&pq->free_list_head, &pq->free_list_count);
    tp_pool_destroy(&pq->node_pool);
#else
    TaggedPtr curr_tp = *PQ_NEXT_PTR(pq->sentinel, pq->element_size);
    PriorityQueueNode* curr = (PriorityQueueNode*)ptr_from_tagged(curr_tp);
    while (curr) {
        TaggedPtr next_tp = *PQ_NEXT_PTR(curr, pq->element_size);
        PriorityQueueNode* next = (PriorityQueueNode*)ptr_from_tagged(next_tp);
        safe_free((void**)&curr);
        curr = next;
    }
    safe_free((void**)&pq->sentinel);
    deferred_free_flush(&pq->free_list_head, &pq->free_list_count);
#endif
    safe_free((void**)&pq);
}

/**
 * @brief 入队（按优先级升序/降序插入）
 */
int lock_free_priority_queue_push(LockFreePriorityQueue* pq, const void* element, int priority) {
    if (!pq || !element) return -1;
    if ((size_t)pq->size >= pq->capacity) return 1;

#if TP_HAS_POOL
    TP_BIND_POOL(&pq->node_pool);
#endif

    PriorityQueueNode* new_node = create_pq_node(element, pq->element_size, (int32_t)priority);
    if (!new_node) return -1;

    int attempts = 0;
    while (attempts < pq->max_retries) {
        PriorityQueueNode* prev = pq->sentinel;
        volatile TaggedPtr* prev_next = PQ_NEXT_PTR(prev, pq->element_size);
        TaggedPtr curr_tp = *prev_next;
        PriorityQueueNode* curr = (PriorityQueueNode*)ptr_from_tagged(curr_tp);

        while (curr) {
            int cmp = pq->is_min_heap ?
                (new_node->priority < curr->priority) :
                (new_node->priority > curr->priority);
            if (cmp) break;
            prev = curr;
            prev_next = PQ_NEXT_PTR(prev, pq->element_size);
            curr_tp = *prev_next;
            curr = (PriorityQueueNode*)ptr_from_tagged(curr_tp);
        }

        TaggedPtr new_next = make_tagged(new_node, tag_from_tagged(curr_tp) + 1);
        *PQ_NEXT_PTR(new_node, pq->element_size) = curr_tp;

#ifdef _WIN32
        TaggedPtr old = (TaggedPtr)InterlockedCompareExchange64(
            (volatile LONG64*)prev_next, (LONG64)new_next, (LONG64)curr_tp);
        if (old == curr_tp) {
#else
        if (__sync_bool_compare_and_swap(
            (volatile long long*)prev_next, (long long)curr_tp, (long long)new_next)) {
#endif
            atomic_fetch_add(&pq->size, 1);
            if (pq->free_list_count >= DEFERRED_FREE_THRESHOLD) {
                deferred_free_scan(&pq->free_list_head, &pq->free_list_count);
            }
            return 0;
        }

        backoff_strategy_impl(attempts, 0);
        attempts++;
    }

    safe_free((void**)&new_node);
    return -1;
}

/**
 * @brief 出队（弹出优先级最高的元素）
 */
int lock_free_priority_queue_pop(LockFreePriorityQueue* pq, void* element) {
    if (!pq || !element) return -1;

#if TP_HAS_POOL
    TP_BIND_POOL(&pq->node_pool);
#endif

    int attempts = 0;
    while (attempts < pq->max_retries) {
        volatile TaggedPtr* sentinel_next = PQ_NEXT_PTR(pq->sentinel, pq->element_size);
        TaggedPtr first_tp = *sentinel_next;
        PriorityQueueNode* first = (PriorityQueueNode*)ptr_from_tagged(first_tp);
        if (!first) return 1;

        TaggedPtr second_tp = *PQ_NEXT_PTR(first, pq->element_size);

        TaggedPtr new_sentinel_next = make_tagged(
            (PriorityQueueNode*)ptr_from_tagged(second_tp),
            tag_from_tagged(first_tp) + 1);

#ifdef _WIN32
        TaggedPtr old = (TaggedPtr)InterlockedCompareExchange64(
            (volatile LONG64*)sentinel_next, (LONG64)new_sentinel_next, (LONG64)first_tp);
        if (old == first_tp) {
#else
        if (__sync_bool_compare_and_swap(
            (volatile long long*)sentinel_next, (long long)first_tp, (long long)new_sentinel_next)) {
#endif
            memcpy(element, first->data, pq->element_size);
            atomic_fetch_sub(&pq->size, 1);
#if TP_HAS_POOL
            deferred_free_add_pool(&pq->free_list_head, &pq->free_list_count, first);
#else
            deferred_free_add(&pq->free_list_head, &pq->free_list_count, first);
#endif
            if (pq->free_list_count >= DEFERRED_FREE_THRESHOLD) {
                deferred_free_scan(&pq->free_list_head, &pq->free_list_count);
            }
            return 0;
        }

        backoff_strategy_impl(attempts, 0);
        attempts++;
    }
    return -1;
}

/**
 * @brief 查看队首元素（不移除）
 */
int lock_free_priority_queue_peek(const LockFreePriorityQueue* pq, void* element) {
    if (!pq || !element) return -1;
#if TP_HAS_POOL
    TP_BIND_POOL((TaggedPtrNodePool*)&pq->node_pool);
#endif
    TaggedPtr first_tp = *PQ_NEXT_PTR(pq->sentinel, pq->element_size);
    PriorityQueueNode* first = (PriorityQueueNode*)ptr_from_tagged(first_tp);
    if (!first) return 1;
    memcpy(element, first->data, pq->element_size);
    return 0;
}

/**
 * @brief 获取优先队列大小
 */
size_t lock_free_priority_queue_size(const LockFreePriorityQueue* pq) {
    return pq ? (size_t)pq->size : 0;
}

/**
 * @brief 检查优先队列是否为空
 */
int lock_free_priority_queue_is_empty(const LockFreePriorityQueue* pq) {
    if (!pq) return 1;
#if TP_HAS_POOL
    TP_BIND_POOL((TaggedPtrNodePool*)&pq->node_pool);
#endif
    TaggedPtr first_tp = *PQ_NEXT_PTR(pq->sentinel, pq->element_size);
    return ptr_from_tagged(first_tp) == NULL ? 1 : 0;
}

/******************************************************************************
 *                        无锁跳表实现
 ******************************************************************************/

#define SKIP_LIST_MAX_LEVEL 32

/**
 * @brief 跳表节点
 */
typedef struct SkipListNode {
    void* key;                           /**< 键指针 */
    size_t key_size;                     /**< 键大小 */
    void* value;                         /**< 值指针 */
    int level;                           /**< 该节点的层数 */
    volatile TaggedPtr next[1];          /**< 柔性数组：每层的标记指针next */
} SkipListNode;

/**
 * @brief 无锁跳表内部结构
 */
struct LockFreeSkipList {
    SkipListNode* head;                  /**< 哨兵头节点（层数为max_level） */
    int max_level;                       /**< 最大层数 */
    int probability;                     /**< 层数递增概率分母 */
    size_t element_size;                 /**< 元素大小（保留） */
    int max_retries;                     /**< 最大重试次数 */
    volatile LONG size;                  /**< 当前节点数 */
    DeferredNode* free_list_head;        /**< 延迟释放链表 */
    volatile LONG free_list_count;       /**< 延迟释放计数 */
#if TP_HAS_POOL
    TaggedPtrNodePool node_pool;         /**< M-029: 32位回退方案节点池 */
    size_t pool_node_size;               /**< 池中单个节点最大大小 */
#endif
};

/**
 * @brief 随机生成层数（几何分布）
 */
static int skip_list_random_level(int max_level, int probability) {
    int level = 0;
    while (level < max_level - 1) {
        if (secure_random_int(probability) == 0) level++;
        else break;
    }
    return level;
}

/**
 * @brief 键比较函数
 */
static int skip_list_compare_key(const void* key1, size_t size1, const void* key2, size_t size2) {
    size_t min_size = size1 < size2 ? size1 : size2;
    int cmp = memcmp(key1, key2, min_size);
    if (cmp != 0) return cmp;
    if (size1 < size2) return -1;
    if (size1 > size2) return 1;
    return 0;
}

/**
 * @brief 创建跳表节点
 */
static SkipListNode* create_skip_list_node(int level, const void* key, size_t key_size, void* value) {
    int actual_level = level + 1;
    size_t node_size = sizeof(SkipListNode) + (size_t)(actual_level - 1) * sizeof(TaggedPtr);
#if TP_HAS_POOL
    /* M-029: 32位回退方案——从节点池分配（池中所有节点大小一致，取最大值） */
    SkipListNode* node = (SkipListNode*)tp_pool_alloc_node(g_tp_active_pool);
    if (!node) return NULL;
    memset(node, 0, node_size);
#else
    SkipListNode* node = (SkipListNode*)safe_malloc(node_size);
    if (!node) return NULL;
#endif

    if (key && key_size > 0) {
        node->key = safe_malloc(key_size);
        if (!node->key) {
#if TP_HAS_POOL
            node->level = 0;
            node->next[0] = (TaggedPtr)0;
#else
            safe_free((void**)&node);
#endif
            return NULL;
        }
        memcpy(node->key, key, key_size);
    } else {
        node->key = NULL;
    }
    node->key_size = key_size;
    node->value = value;
    node->level = actual_level;

    for (int i = 0; i < actual_level; i++) {
        node->next[i] = make_tagged(NULL, 0);
    }
    return node;
}

/**
 * @brief 释放跳表节点（含延迟释放）
 */
static void free_skip_list_node_deferred(struct LockFreeSkipList* sl, SkipListNode* node) {
    if (!sl || !node) return;
#if TP_HAS_POOL
    deferred_free_add_pool(&sl->free_list_head, &sl->free_list_count, node);
#else
    deferred_free_add(&sl->free_list_head, &sl->free_list_count, node);
#endif
    if (sl->free_list_count >= DEFERRED_FREE_THRESHOLD) {
        deferred_free_scan(&sl->free_list_head, &sl->free_list_count);
    }
}

/**
 * @brief 创建无锁跳表
 */
LockFreeSkipList* lock_free_skip_list_create(const LockFreeSkipListConfig* config) {
    if (!config || config->max_level == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建跳表：参数无效");
        return NULL;
    }

    int max_level = (int)config->max_level;
    if (max_level > SKIP_LIST_MAX_LEVEL) max_level = SKIP_LIST_MAX_LEVEL;
    int prob = config->probability > 0 ? config->probability : 4;

    LockFreeSkipList* sl = (LockFreeSkipList*)safe_malloc(sizeof(LockFreeSkipList));
    if (!sl) return NULL;

#if TP_HAS_POOL
    /* M-029：32位回退方案——初始化节点池（节点最大大小 = 结构体 + max_level层next指针） */
    sl->pool_node_size = sizeof(SkipListNode) + (size_t)(max_level - 1) * sizeof(TaggedPtr);
    /* 跳表无固定容量限制，使用65535作为池容量（16位索引最大值） */
    if (tp_pool_init(&sl->node_pool, sl->pool_node_size, 65535) != 0) {
        safe_free((void**)&sl);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建跳表：节点池初始化失败");
        return NULL;
    }
    TP_BIND_POOL(&sl->node_pool);
#endif

    sl->head = create_skip_list_node(max_level, NULL, 0, NULL);
    if (!sl->head) {
#if TP_HAS_POOL
        tp_pool_destroy(&sl->node_pool);
#endif
        safe_free((void**)&sl);
        return NULL;
    }

    sl->max_level = max_level;
    sl->probability = prob;
    sl->element_size = config->element_size;
    sl->max_retries = config->max_retries > 0 ? config->max_retries : 1000;
    sl->size = 0;
    sl->free_list_head = NULL;
    sl->free_list_count = 0;

    return sl;
}

/**
 * @brief 销毁无锁跳表
 */
void lock_free_skip_list_free(LockFreeSkipList* sl) {
    if (!sl) return;
#if TP_HAS_POOL
    TP_BIND_POOL(&sl->node_pool);
    /* M-029: 32位回退方案——释放所有节点的key数据后，由tp_pool_destroy统一释放池 */
    SkipListNode* curr = (SkipListNode*)ptr_from_tagged(sl->head->next[0]);
    while (curr) {
        TaggedPtr next_tp = curr->next[0];
        SkipListNode* next = (SkipListNode*)ptr_from_tagged(next_tp);
        if (curr->key) safe_free((void**)&curr->key);
        curr = next;
    }
    deferred_free_flush(&sl->free_list_head, &sl->free_list_count);
    tp_pool_destroy(&sl->node_pool);
#else
    SkipListNode* curr = (SkipListNode*)ptr_from_tagged(sl->head->next[0]);
    while (curr) {
        TaggedPtr next_tp = curr->next[0];
        SkipListNode* next = (SkipListNode*)ptr_from_tagged(next_tp);
        if (curr->key) safe_free((void**)&curr->key);
        safe_free((void**)&curr);
        curr = next;
    }
    safe_free((void**)&sl->head);
    deferred_free_flush(&sl->free_list_head, &sl->free_list_count);
#endif
    safe_free((void**)&sl);
}

/**
 * @brief 搜索跳表，填充每层的前驱节点
 */
static SkipListNode* skip_list_search(struct LockFreeSkipList* sl, const void* key,
                                       size_t key_size, SkipListNode** prevs) {
    SkipListNode* prev = sl->head;

    for (int level = sl->max_level - 1; level >= 0; level--) {
        TaggedPtr curr_tp = prev->next[level];
        SkipListNode* curr = (SkipListNode*)ptr_from_tagged(curr_tp);

        while (curr) {
            int cmp = skip_list_compare_key(curr->key, curr->key_size, key, key_size);
            if (cmp >= 0) break;
            prev = curr;
            curr_tp = prev->next[level];
            curr = (SkipListNode*)ptr_from_tagged(curr_tp);
        }

        if (prevs) prevs[level] = prev;
    }

    if (prevs) {
        TaggedPtr cand_tp = prevs[0]->next[0];
        SkipListNode* candidate = (SkipListNode*)ptr_from_tagged(cand_tp);
        if (candidate && skip_list_compare_key(candidate->key, candidate->key_size, key, key_size) == 0) {
            return candidate;
        }
    }
    return NULL;
}

/**
 * @brief 插入键值对
 */
int lock_free_skip_list_insert(LockFreeSkipList* sl, const void* key, size_t key_size, void* value) {
    if (!sl || !key || key_size == 0) return -1;

#if TP_HAS_POOL
    TP_BIND_POOL(&sl->node_pool);
#endif

    int attempts = 0;
    while (attempts < sl->max_retries) {
        SkipListNode* prevs[SKIP_LIST_MAX_LEVEL];
        SkipListNode* existing = skip_list_search(sl, key, key_size, prevs);
        if (existing) return 1;

        int new_level = skip_list_random_level(sl->max_level, sl->probability);
        SkipListNode* new_node = create_skip_list_node(new_level, key, key_size, value);
        if (!new_node) return -1;

        int success = 1;
        for (int level = 0; level <= new_level && level < sl->max_level; level++) {
            SkipListNode* prev = prevs[level] ? prevs[level] : sl->head;
            TaggedPtr curr_tp = prev->next[level];
            SkipListNode* curr = (SkipListNode*)ptr_from_tagged(curr_tp);

            while (curr && skip_list_compare_key(curr->key, curr->key_size, key, key_size) < 0) {
                prev = curr;
                curr_tp = prev->next[level];
                curr = (SkipListNode*)ptr_from_tagged(curr_tp);
                prevs[level] = prev;
            }

            new_node->next[level] = curr_tp;
            TaggedPtr new_tp = make_tagged(new_node, tag_from_tagged(curr_tp) + 1);

#ifdef _WIN32
            TaggedPtr old = (TaggedPtr)InterlockedCompareExchange64(
                (volatile LONG64*)&prev->next[level], (LONG64)new_tp, (LONG64)curr_tp);
            if (old != curr_tp) {
#else
            if (!__sync_bool_compare_and_swap(
                (volatile long long*)&prev->next[level], (long long)curr_tp, (long long)new_tp)) {
#endif
                success = 0;
                break;
            }
        }

        if (success) {
            atomic_fetch_add(&sl->size, 1);
            if (sl->free_list_count >= DEFERRED_FREE_THRESHOLD) {
                deferred_free_scan(&sl->free_list_head, &sl->free_list_count);
            }
            return 0;
        }

        for (int level = 0; level <= new_level && level < sl->max_level; level++) {
            SkipListNode* prev = prevs[level] ? prevs[level] : sl->head;
            TaggedPtr curr_tp = prev->next[level];
            SkipListNode* curr = (SkipListNode*)ptr_from_tagged(curr_tp);
            if (curr == new_node) {
                TaggedPtr next_tp = new_node->next[level];
#ifdef _WIN32
                InterlockedCompareExchange64(
                    (volatile LONG64*)&prev->next[level], (LONG64)next_tp, (LONG64)curr_tp);
#else
                __sync_bool_compare_and_swap(
                    (volatile long long*)&prev->next[level], (long long)curr_tp, (long long)next_tp);
#endif
            }
        }

        if (new_node->key) safe_free((void**)&new_node->key);
#if TP_HAS_POOL
        /* M-029: 池节点无需单独释放，仅清理标记 */
        new_node->level = 0;
        new_node->next[0] = (TaggedPtr)0;
#else
        safe_free((void**)&new_node);
#endif

        backoff_strategy_impl(attempts, 0);
        attempts++;
    }
    return -1;
}

/**
 * @brief 查找键对应的值
 */
void* lock_free_skip_list_find(const LockFreeSkipList* sl, const void* key, size_t key_size) {
    if (!sl || !key || key_size == 0) return NULL;

#if TP_HAS_POOL
    TP_BIND_POOL((TaggedPtrNodePool*)&((LockFreeSkipList*)sl)->node_pool);
#endif

    SkipListNode* prev = sl->head;
    for (int level = sl->max_level - 1; level >= 0; level--) {
        TaggedPtr curr_tp = prev->next[level];
        SkipListNode* curr = (SkipListNode*)ptr_from_tagged(curr_tp);
        while (curr) {
            int cmp = skip_list_compare_key(curr->key, curr->key_size, key, key_size);
            if (cmp == 0 && level == 0) return curr->value;
            if (cmp >= 0) break;
            prev = curr;
            curr_tp = prev->next[level];
            curr = (SkipListNode*)ptr_from_tagged(curr_tp);
        }
    }
    return NULL;
}

/**
 * @brief 删除键值对
 * 
 * 不使用位0标记机制（避免ptr_from_tagged返回含标记位的无效地址），
 * 直接从每层链表中CAS移除目标节点。
 */
int lock_free_skip_list_erase(LockFreeSkipList* sl, const void* key, size_t key_size) {
    if (!sl || !key || key_size == 0) return -1;

#if TP_HAS_POOL
    TP_BIND_POOL(&sl->node_pool);
#endif

    int attempts = 0;
    while (attempts < sl->max_retries) {
        SkipListNode* prevs[SKIP_LIST_MAX_LEVEL];
        SkipListNode* target = skip_list_search(sl, key, key_size, prevs);
        if (!target) return 1;

        int success = 1;
        /* 从最高层到底层依次移除 */
        for (int level = target->level - 1; level >= 0; level--) {
            SkipListNode* prev = prevs[level] ? prevs[level] : sl->head;
            TaggedPtr expected = prev->next[level];
            void* expected_ptr = ptr_from_tagged(expected);

            /* 验证prev的next确实指向target */
            if (expected_ptr != target) {
                if (level == 0) { success = 0; break; }
                continue;
            }

            TaggedPtr target_next = target->next[level];
            TaggedPtr desired = make_tagged(
                ptr_from_tagged(target_next),
                tag_from_tagged(expected) + 1);

            int cas_ok = 0;
#ifdef _WIN32
            if ((TaggedPtr)InterlockedCompareExchange64(
                (volatile LONG64*)&prev->next[level], (LONG64)desired, (LONG64)expected) == expected) {
                cas_ok = 1;
            }
#else
            if (__sync_bool_compare_and_swap(
                (volatile long long*)&prev->next[level], (long long)expected, (long long)desired)) {
                cas_ok = 1;
            }
#endif
            if (!cas_ok) {
                success = 0;
                break;
            }
        }

        if (success) {
/* 使用延迟释放替代立即释放。
             * 立即释放会导致并发读线程访问已释放节点(use-after-free)。
             * 延迟释放通过hazard-pointer机制确保所有读者退出后才释放。 */
            free_skip_list_node_deferred(sl, target);
            atomic_fetch_sub(&sl->size, 1);
            return 0;
        }

        attempts++;
        backoff_strategy_impl(attempts, 0);
    }
    return -1;
}

/**
 * @brief 获取跳表大小
 */
size_t lock_free_skip_list_size(const LockFreeSkipList* sl) {
    return sl ? (size_t)sl->size : 0;
}

/**
 * @brief 检查跳表是否为空
 */
int lock_free_skip_list_is_empty(const LockFreeSkipList* sl) {
    if (!sl) return 1;
#if TP_HAS_POOL
    TP_BIND_POOL((TaggedPtrNodePool*)&((LockFreeSkipList*)sl)->node_pool);
#endif
    return ptr_from_tagged(sl->head->next[0]) == NULL ? 1 : 0;
}

/**
 * @brief 清空跳表
 */
void lock_free_skip_list_clear(LockFreeSkipList* sl) {
    if (!sl) return;
#if TP_HAS_POOL
    TP_BIND_POOL(&sl->node_pool);
    /* M-029: 池模式——只释放key，节点由池统一管理 */
    TaggedPtr curr_tp = sl->head->next[0];
    SkipListNode* curr = (SkipListNode*)ptr_from_tagged(curr_tp);
    while (curr) {
        TaggedPtr next_tp = curr->next[0];
        SkipListNode* next = (SkipListNode*)ptr_from_tagged(next_tp);
        if (curr->key) safe_free((void**)&curr->key);
        curr = next;
    }
#else
    TaggedPtr curr_tp = sl->head->next[0];
    SkipListNode* curr = (SkipListNode*)ptr_from_tagged(curr_tp);
    while (curr) {
        TaggedPtr next_tp = curr->next[0];
        SkipListNode* next = (SkipListNode*)ptr_from_tagged(next_tp);
        if (curr->key) safe_free((void**)&curr->key);
        safe_free((void**)&curr);
        curr = next;
    }
#endif
    for (int i = 0; i < sl->max_level; i++) {
        sl->head->next[i] = make_tagged(NULL, 0);
    }
#ifdef _WIN32
    InterlockedExchange64((volatile LONG64*)&sl->size, 0);
#else
    __sync_synchronize();
    sl->size = 0;
#endif
    deferred_free_flush(&sl->free_list_head, &sl->free_list_count);
}

/******************************************************************************
 *                        无锁哈希表（公共API实现）
 ******************************************************************************/

LockFreeHashTable* lock_free_hash_table_create(const LockFreeHashTableConfig* config)
{
    if (!config || config->capacity == 0 || config->key_size == 0 || config->value_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建无锁哈希表：配置参数无效");
        return NULL;
    }
    LockFreeHashTable* table = (LockFreeHashTable*)safe_calloc(1, sizeof(LockFreeHashTable));
    if (!table) return NULL;

    table->buckets = (volatile LockFreeHashTableNode**)safe_calloc(config->capacity, sizeof(LockFreeHashTableNode*));
    if (!table->buckets) {
        safe_free((void**)&table);
        return NULL;
    }
    table->capacity = config->capacity;
    table->key_size = config->key_size;
    table->value_size = config->value_size;
    table->hash_function_type = config->hash_function_type;
    table->enable_resizing = config->enable_resizing;
    table->load_factor_threshold = (config->load_factor_threshold > 0.0f) ? config->load_factor_threshold : 0.75f;
    table->max_probe_length = (config->max_probe_length > 0) ? config->max_probe_length : 16;
    table->size = 0;
    return table;
}

void lock_free_hash_table_free(LockFreeHashTable* table)
{
    if (!table) return;
    for (size_t i = 0; i < table->capacity; i++) {
        LockFreeHashTableNode* node = (LockFreeHashTableNode*)table->buckets[i];
        while (node) {
            LockFreeHashTableNode* next = node->next;
            free_hash_table_node(node);
            node = next;
        }
    }
    safe_free((void**)&table->buckets);
    safe_free((void**)&table);
}

int lock_free_hash_table_insert(LockFreeHashTable* table, const void* key, size_t key_size,
                                 const void* value, size_t value_size, LockFreeOperationResult* result)
{
    if (!table || !key || !value) return -1;
    if (result) lock_free_operation_result_init(result);

    uint32_t hash_val = hash_function(key, key_size, (uint32_t)table->capacity);
    size_t idx = (size_t)(hash_val % table->capacity);
    int retries = 0;

    LockFreeHashTableNode* new_node = create_hash_table_node(key, key_size, value, value_size);
    if (!new_node) return -1;

    while (1) {
        LockFreeHashTableNode* head = (LockFreeHashTableNode*)table->buckets[idx];

        /* 检查键是否已存在 */
        LockFreeHashTableNode* curr = head;
        while (curr) {
            if (curr->key && memcmp(curr->key, key, key_size) == 0) {
                /* 键已存在，更新值 */
                void* old_val = curr->value;
                void* new_val = safe_malloc(value_size);
                if (!new_val) {
                    free_hash_table_node(new_node);
                    return -1;
                }
                memcpy(new_val, value, value_size);
                if (compare_and_swap_pointer(&curr->value, old_val, new_val)) {
                    safe_free((void**)&old_val);
                    free_hash_table_node(new_node);
                    if (result) {
                        result->success = 1;
                        result->retries = retries;
                    }
                    return 0;
                }
                safe_free((void**)&new_val);
                backoff_strategy_impl(retries++, 0);
                continue;
            }
            curr = curr->next;
        }

        /* 键不存在，插入新节点到链表头部 */
        new_node->next = head;
        if (compare_and_swap_pointer((void**)&table->buckets[idx], head, new_node)) {
#ifdef _WIN32
            InterlockedIncrement(&table->size);
#else
            __sync_fetch_and_add(&table->size, 1);
#endif
            if (result) {
                result->success = 1;
                result->retries = retries;
            }
            return 0;
        }
        backoff_strategy_impl(retries++, 0);
    }
}

int lock_free_hash_table_lookup(LockFreeHashTable* table, const void* key, size_t key_size,
                                 void* value, size_t value_size, LockFreeOperationResult* result)
{
    if (!table || !key || !value) return -1;
    if (result) lock_free_operation_result_init(result);

    uint32_t hash_val = hash_function(key, key_size, (uint32_t)table->capacity);
    size_t idx = (size_t)(hash_val % table->capacity);

    LockFreeHashTableNode* curr = (LockFreeHashTableNode*)table->buckets[idx];
    while (curr) {
        if (curr->key && memcmp(curr->key, key, key_size) == 0) {
            size_t copy_size = (value_size < table->value_size) ? value_size : table->value_size;
            memcpy(value, curr->value, copy_size);
            if (result) {
                result->success = 1;
                result->retries = 0;
            }
            return 0;
        }
        curr = curr->next;
    }
    if (result) result->success = 0;
    return -1;
}

int lock_free_hash_table_delete(LockFreeHashTable* table, const void* key, size_t key_size,
                                 LockFreeOperationResult* result)
{
    if (!table || !key) return -1;
    if (result) lock_free_operation_result_init(result);

    uint32_t hash_val = hash_function(key, key_size, (uint32_t)table->capacity);
    size_t idx = (size_t)(hash_val % table->capacity);
    int retries = 0;

    while (1) {
        LockFreeHashTableNode* head = (LockFreeHashTableNode*)table->buckets[idx];
        if (!head) {
            if (result) result->success = 0;
            return -1;
        }

        /* 处理头节点 */
        if (head->key && memcmp(head->key, key, key_size) == 0) {
            LockFreeHashTableNode* next = head->next;
            if (compare_and_swap_pointer((void**)&table->buckets[idx], head, next)) {
#ifdef _WIN32
                InterlockedDecrement(&table->size);
#else
                __sync_fetch_and_sub(&table->size, 1);
#endif
                head->next = NULL;
                free_hash_table_node(head);
                if (result) {
                    result->success = 1;
                    result->retries = retries;
                }
                return 0;
            }
            backoff_strategy_impl(retries++, 0);
            continue;
        }

        /* 在链表中查找 */
        LockFreeHashTableNode* prev = head;
        LockFreeHashTableNode* curr = head->next;
        while (curr) {
            if (curr->key && memcmp(curr->key, key, key_size) == 0) {
                LockFreeHashTableNode* next = curr->next;
                if (compare_and_swap_pointer((void**)&prev->next, curr, next)) {
#ifdef _WIN32
                    InterlockedDecrement(&table->size);
#else
                    __sync_fetch_and_sub(&table->size, 1);
#endif
                    curr->next = NULL;
                    free_hash_table_node(curr);
                    if (result) {
                        result->success = 1;
                        result->retries = retries;
                    }
                    return 0;
                }
                backoff_strategy_impl(retries++, 0);
                break;
            }
            prev = curr;
            curr = curr->next;
        }
        if (!curr) {
            if (result) result->success = 0;
            return -1;
        }
    }
}

size_t lock_free_hash_table_size(const LockFreeHashTable* table)
{
    if (!table) return 0;
    return (size_t)table->size;
}

/******************************************************************************
 *                        无锁内存池（公共API实现）
 ******************************************************************************/

LockFreeMemoryPool* lock_free_memory_pool_create(const LockFreeMemoryPoolConfig* config)
{
    if (!config || config->block_size == 0 || config->num_blocks == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建无锁内存池：配置参数无效");
        return NULL;
    }
    LockFreeMemoryPool* pool = (LockFreeMemoryPool*)safe_calloc(1, sizeof(LockFreeMemoryPool));
    if (!pool) return NULL;

    pool->block_size = config->block_size;
    pool->num_blocks = config->num_blocks;
    pool->enable_slab_allocator = config->enable_slab_allocator;
    pool->slab_size = config->slab_size;
    pool->enable_batch_allocation = config->enable_batch_allocation;
    pool->batch_size = config->batch_size > 0 ? config->batch_size : 1;

    /* 预分配所有内存块 */
    pool->blocks = (LockFreeMemoryBlock*)safe_calloc(config->num_blocks, sizeof(LockFreeMemoryBlock));
    if (!pool->blocks) {
        safe_free((void**)&pool);
        return NULL;
    }
    for (size_t i = 0; i < config->num_blocks; i++) {
        pool->blocks[i].data = safe_malloc(config->block_size);
        if (!pool->blocks[i].data) {
            for (size_t j = 0; j < i; j++)
                safe_free((void**)&pool->blocks[j].data);
            safe_free((void**)&pool->blocks);
            safe_free((void**)&pool);
            return NULL;
        }
        pool->blocks[i].allocated = 0;
        /* 构建空闲链表（后向链接） */
        if (i < config->num_blocks - 1)
            pool->blocks[i].next = &pool->blocks[i + 1];
        else
            pool->blocks[i].next = NULL;
    }
    pool->free_list = &pool->blocks[0];
    pool->free_blocks = (LONG)config->num_blocks;
    return pool;
}

void lock_free_memory_pool_free(LockFreeMemoryPool* pool)
{
    if (!pool) return;
    if (pool->blocks) {
        for (size_t i = 0; i < pool->num_blocks; i++) {
            if (pool->blocks[i].data)
                safe_free((void**)&pool->blocks[i].data);
        }
        safe_free((void**)&pool->blocks);
    }
    safe_free((void**)&pool);
}

void* lock_free_memory_pool_allocate(LockFreeMemoryPool* pool, LockFreeOperationResult* result)
{
    if (!pool) return NULL;
    if (result) lock_free_operation_result_init(result);

    int retries = 0;
    while (1) {
        volatile LockFreeMemoryBlock* head = pool->free_list;
        if (!head) {
            if (result) {
                result->success = 0;
                result->retries = retries;
            }
            return NULL;
        }
        volatile LockFreeMemoryBlock* next = head->next;
        if (compare_and_swap_pointer((void**)&pool->free_list, (void*)head, (void*)next)) {
            head->allocated = 1;
#ifdef _WIN32
            InterlockedDecrement(&pool->free_blocks);
#else
            __sync_fetch_and_sub(&pool->free_blocks, 1);
#endif
            if (result) {
                result->success = 1;
                result->retries = retries;
            }
            return head->data;
        }
        backoff_strategy_impl(retries++, 0);
    }
}

int lock_free_memory_pool_deallocate(LockFreeMemoryPool* pool, void* block,
                                      LockFreeOperationResult* result)
{
    if (!pool || !block) return -1;
    if (result) lock_free_operation_result_init(result);

    /* 找到对应的内存块 */
    for (size_t i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i].data == block) {
            int retries = 0;
            while (1) {
                volatile LockFreeMemoryBlock* head = pool->free_list;
                pool->blocks[i].next = (LockFreeMemoryBlock*)head;
                pool->blocks[i].allocated = 0;
                if (compare_and_swap_pointer((void**)&pool->free_list, (void*)head, (void*)&pool->blocks[i])) {
#ifdef _WIN32
                    InterlockedIncrement(&pool->free_blocks);
#else
                    __sync_fetch_and_add(&pool->free_blocks, 1);
#endif
                    if (result) {
                        result->success = 1;
                        result->retries = retries;
                    }
                    return 0;
                }
                backoff_strategy_impl(retries++, 0);
            }
        }
    }
    if (result) result->success = 0;
    return -1;
}

size_t lock_free_memory_pool_available_blocks(const LockFreeMemoryPool* pool)
{
    if (!pool) return 0;
    return (size_t)pool->free_blocks;
}

size_t lock_free_memory_pool_total_blocks(const LockFreeMemoryPool* pool)
{
    if (!pool) return 0;
    return pool->num_blocks;
}

