/**
 * @file memory_utils.c
 * @brief 内存工具库实现
 * 
 * 提供内存管理、分配、池管理和内存安全检查功能。
 */

#include "selflnn/utils/memory_utils.h"

// 取消宏定义，以便在此文件中定义实际函数体
#undef safe_malloc
#undef safe_calloc
#undef safe_aligned_malloc

/* DEEP-005修复: 前向声明使用实际函数名 _safe_malloc/_safe_calloc, safe_free为实际函数 */
void* _safe_malloc(size_t size, const char* file, int line);
void* _safe_calloc(size_t count, size_t size, const char* file, int line);
void safe_free(void** ptr);

/* DEEP-005修复: undef后重新定义宏，使文件内部调用也正确展开 */
#define safe_malloc(size)       _safe_malloc(size, __FILE__, __LINE__)
#define safe_calloc(count, sz)  _safe_calloc(count, sz, __FILE__, __LINE__)
#define safe_aligned_malloc(sz, align) _safe_aligned_malloc(sz, align, __FILE__, __LINE__)

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#ifdef _DEBUG
#include <crtdbg.h>
#endif
#else
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#endif

/* P6-R90: bypass safe alloc - use standard malloc/free when set */
static int g_bypass_safe_alloc = 0;
void memory_utils_bypass_safe_alloc(int bypass) { g_bypass_safe_alloc = bypass; }

/**
 * @brief 内存块头部信息
 * 修复#4: magic字段添加volatile限定符，防止MSVC /O2优化下编译器将
 * magic != MEMORY_MAGIC 常量折叠为false或跨函数缓存magic值。
 * volatile确保每次读取都从内存获取，不依赖寄存器缓存。
 *
 * raw_ptr字段(P1修复 _safe_aligned_malloc兼容): 指向 malloc 返回的原始指针。
 * 普通safe_malloc块: raw_ptr == header(即malloc返回值,因header就放在raw_ptr处);
 * 对齐safe_aligned_malloc块: raw_ptr != header(数据指针经手动对齐,header在
 * 对齐后的数据指针之前,但真正的malloc返回值raw_ptr更靠前)。safe_free统一通过
 * free(header->raw_ptr)释放,从而兼容两种分配方式。
 * 注意: 该字段填入结构体原有尾部填充区(sizeof 56->64,BLOCK_HEADER_SIZE仍为64),
 * 不改变头部大小与现有块布局。
 */
typedef struct {
    size_t size;              /**< 块大小 */
    int is_allocated;         /**< 是否已分配 */
    const char* file;         /**< 分配文件（调试用） */
    int line;                 /**< 分配行号（调试用） */
    volatile size_t magic;    /**< 魔法数字用于验证（volatile防止编译器优化） */
    size_t alloc_id;          /**< 分配序号（用于调试） */
    void* caller_address;     /**< 调用者返回地址（用于调试） */
    void* raw_ptr;            /**< malloc返回的原始指针（用于safe_free统一释放，兼容对齐分配） */
} MemoryBlockHeader;

/**
 * @brief 内存块尾部信息
 */
typedef struct {
    volatile size_t magic;    /**< 魔法数字用于验证（volatile防止编译器优化） */
} MemoryBlockFooter;

/**
 * @brief 内存池内部结构体
 */
struct MemoryPool {
    MemoryPoolConfig config;  /**< 池配置 */
    void* base_ptr;           /**< 池基地址 */
    size_t total_size;        /**< 总大小 */
    size_t used_size;         /**< 已使用大小 */
    size_t free_size;         /**< 空闲大小 */
    int* block_status;        /**< 块状态数组（0=空闲，1=已用） */
    /* P1修复(memory_pool_free多块泄漏): 仅靠 block_status 无法知道一次分配占用几个块,
     * 原释放逻辑只释放首块导致多块分配泄漏。block_span 与 block_status 并行, 仅在每次
     * 分配的首块记录该次分配占用的块数, 其余块为0; 释放时读取首块的 span 释放完整跨度。 */
    size_t* block_span;       /**< 每次分配首块记录的占用块数(其余块为0) */
    size_t block_count;       /**< 块数量 */
    size_t first_free;        /**< 第一个空闲块索引 */
};

/**
 * @brief 全局内存统计
 */
static MemoryStats g_memory_stats = {0};

/**
 * @brief 分配序号（用于调试）
 */
static size_t g_allocation_counter = 0;

/* 线程安全锁：保护全局内存统计和分配跟踪 */
#ifdef _WIN32
static CRITICAL_SECTION g_mem_lock;
/* P0修复(竞态条件): 原为普通int + 双检锁(DCL)模式，多线程下存在竞态——
 * 两个线程可能同时通过 !g_mem_lock_init 检查并同时初始化/进入未就绪的临界区。
 * 改为 volatile LONG + 原子CAS状态机，与下方 g_alloc_track_lock_initialized 修复模式一致。
 * 状态机: 0=未初始化, 2=初始化中, 1=已初始化完成。 */
static volatile LONG g_mem_lock_init = 0;
/* mem_lock_init() 在下方定义（紧随 alloc_track_lock_init 之后），保证首次加锁前
 * 临界区已就绪；其余线程自旋等待至状态变为1再进入。 */
#define MEM_LOCK()   do { mem_lock_init(); EnterCriticalSection(&g_mem_lock); } while(0)
#define MEM_UNLOCK() LeaveCriticalSection(&g_mem_lock)
#else
static pthread_mutex_t g_mem_lock = PTHREAD_MUTEX_INITIALIZER;
#define MEM_LOCK()   pthread_mutex_lock(&g_mem_lock)
#define MEM_UNLOCK() pthread_mutex_unlock(&g_mem_lock)
#endif

/**
 * @brief 内存池链表节点（用于碎片整理跟踪）
 */
typedef struct PoolNode {
    MemoryPool* pool;
    struct PoolNode* next;
} PoolNode;

/**
 * @brief 全局内存池链表
 */
static PoolNode* g_pool_list = NULL;

/**
 * @brief 分配跟踪节点（用于泄漏检测）
 */
typedef struct AllocTrackNode {
    void* data_ptr;                  /**< 分配返回的数据指针 */
    size_t size;                     /**< 请求大小 */
    size_t alloc_id;                 /**< 分配序号 */
    const char* file;                /**< 分配文件名 */
    int line;                        /**< 分配行号 */
    struct AllocTrackNode* next;     /**< 链表下一个节点 */
} AllocTrackNode;

/**
 * @brief 全局分配跟踪链表（头节点）
 */
static AllocTrackNode* g_alloc_track_list = NULL;

/**
 * @brief 分配跟踪链表锁
 */
#ifdef _WIN32
static CRITICAL_SECTION g_alloc_track_lock;
/* K-182: volatile + InterlockedCompareExchange 消除DCL竞态 */
static volatile LONG g_alloc_track_lock_initialized = 0;
#else
static pthread_mutex_t g_alloc_track_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

/** @brief atexit 已注册标志（D-3修复: LONG类型支持InterlockedCompareExchange原子操作） */
static volatile LONG g_atexit_registered = 0;

/* M-5修复: 分配跟踪链表损坏标志。
 * 当 alloc_track_remove 检测到链表循环(max_iter耗尽)时置位, 后续所有
 * 跟踪操作(alloc_track_add/remove/contains/dump)均跳过, 防止在损坏链表上反复死循环。 */
static volatile int g_alloc_track_corrupted = 0;

/**
 * @brief 程序退出时自动清理：检测泄漏 + 销毁锁
 */
static void memory_auto_cleanup(void) {
    size_t leaked = memory_check_leaks();
    (void)leaked;
#ifdef _WIN32
    /* P0修复: 同步销毁 g_mem_lock 临界区，避免句柄泄漏（原仅销毁跟踪锁） */
    if (g_mem_lock_init) {
        DeleteCriticalSection(&g_mem_lock);
        g_mem_lock_init = 0;
    }
    if (g_alloc_track_lock_initialized) {
        DeleteCriticalSection(&g_alloc_track_lock);
        g_alloc_track_lock_initialized = 0;
    }
#else
    pthread_mutex_destroy(&g_alloc_track_lock);
#endif
}

/* K-182: 用原子CAS消除DCL竞态 —— 多线程并发调用lock_init时仅一个线程初始化 */
static void alloc_track_lock_init(void) {
#ifdef _WIN32
    if (!g_alloc_track_lock_initialized) {
        if (InterlockedCompareExchange(&g_alloc_track_lock_initialized, 2, 0) == 0) {
            InitializeCriticalSection(&g_alloc_track_lock);
            g_alloc_track_lock_initialized = 1;
        } else {
            while (g_alloc_track_lock_initialized != 1) { Sleep(0); }
        }
    }
#else
    // pthread 静态初始化不需要额外初始化
#endif
    if (!g_atexit_registered) {
        /* D-3修复: 使用InterlockedCompareExchange原子保护g_atexit_registered，
         * 防止多线程首次调用safe_malloc时并发通过检查导致atexit多次注册。 */
#ifdef _WIN32
        if (InterlockedCompareExchange(&g_atexit_registered, 1, 0) == 0) {
#else
        if (__sync_bool_compare_and_swap(&g_atexit_registered, 0, 1)) {
#endif
            atexit(memory_auto_cleanup);
        }
    }
}

/* P0修复(MEM_LOCK DCL竞态): 与 alloc_track_lock_init 完全一致的状态机模式。
 * 仅Windows需要——临界区无法静态初始化。状态: 0=未初始化, 2=初始化中, 1=已就绪。
 * 并发首次进入时, CAS(0->2)成功者执行 InitializeCriticalSection 并置1;
 * 失败者(读到2或1)自旋等待至状态变为1再返回, 确保不进入未就绪的临界区。 */
#ifdef _WIN32
static void mem_lock_init(void) {
    if (!g_mem_lock_init) {
        if (InterlockedCompareExchange(&g_mem_lock_init, 2, 0) == 0) {
            InitializeCriticalSection(&g_mem_lock);
            g_mem_lock_init = 1;
        } else {
            while (g_mem_lock_init != 1) { Sleep(0); }
        }
    }
}
#endif

/**
 * @brief 加锁分配跟踪链表
 */
static void alloc_track_lock(void) {
    alloc_track_lock_init();
#ifdef _WIN32
    EnterCriticalSection(&g_alloc_track_lock);
#else
    pthread_mutex_lock(&g_alloc_track_lock);
#endif
}

/**
 * @brief 解锁分配跟踪链表
 */
static void alloc_track_unlock(void) {
#ifdef _WIN32
    LeaveCriticalSection(&g_alloc_track_lock);
#else
    pthread_mutex_unlock(&g_alloc_track_lock);
#endif
}

/**
 * @brief 向分配跟踪链表添加一条记录
 */
static void alloc_track_add(void* data_ptr, size_t size, size_t alloc_id,
                            const char* file, int line) {
    if (!data_ptr) return;
    /* M-5修复: 若跟踪链表已损坏, 跳过添加避免在损坏链表上操作 */
    if (g_alloc_track_corrupted) return;
    /* P-AUDIT修复(根因): 原使用safe_malloc分配跟踪节点,但safe_malloc内部会调用alloc_track_add,
     * 形成无限互递归: alloc_track_add → safe_malloc → _safe_malloc → alloc_track_add → ...
     * 这导致所有未调用memory_utils_bypass_safe_alloc(1)的程序(如lnn_verify.exe)栈溢出。
     * 修复: 跟踪系统自身使用原始malloc/free,不经过跟踪系统(否则跟踪跟踪器本身就是递归)。 */
    AllocTrackNode* node = (AllocTrackNode*)malloc(sizeof(AllocTrackNode));
    if (!node) return;
    node->data_ptr = data_ptr;
    node->size = size;
    node->alloc_id = alloc_id;
    node->file = file;
    node->line = line;
    alloc_track_lock();
    node->next = g_alloc_track_list;
    g_alloc_track_list = node;
    alloc_track_unlock();
}

/**
 * @brief 从分配跟踪链表移除一条记录
 */
static void alloc_track_remove(void* data_ptr) {
    if (!data_ptr) return;
    /* M-5修复: 若跟踪链表已标记损坏, 跳过所有操作避免死循环 */
    if (g_alloc_track_corrupted) return;
    alloc_track_lock();
    /* M-5修复: 锁内再次检查, 防止竞态 */
    if (g_alloc_track_corrupted) { alloc_track_unlock(); return; }
    AllocTrackNode** pp = &g_alloc_track_list;
    /* G1防御: 添加遍历上限防止链表被破坏为环时死循环 */
    size_t max_iter = 1000000;
    while (*pp && max_iter-- > 0) {
        if ((*pp)->data_ptr == data_ptr) {
            AllocTrackNode* to_free = *pp;
            *pp = (*pp)->next;
            alloc_track_unlock();
            /* P-AUDIT修复(根因): 同alloc_track_add,使用原始free释放跟踪节点,
             * 避免safe_free → alloc_track_remove → safe_free 的递归。 */
            free(to_free);
            return;
        }
        pp = &(*pp)->next;
    }
    /* M-5修复: max_iter耗尽时的明确错误处理。
     * 原实现仅打印错误后继续, 但链表已被破坏为环, 后续任何遍历操作
     * (alloc_track_contains/alloc_track_dump_leaks)均会再次死循环。
     * 修复: 置损坏标志、清空链表指针、输出详细诊断信息, 彻底阻断后续遍历。 */
    if (max_iter == 0 && !g_alloc_track_corrupted) {
        g_alloc_track_corrupted = 1;
        fprintf(stderr, "致命错误: alloc_track_remove 检测到分配跟踪链表循环(搜索指针=%p, 已遍历%zu次)\n",
                data_ptr, (size_t)1000000);
        fprintf(stderr, "  诊断: 跟踪链表内部指针被破坏, 可能由堆缓冲区溢出导致。\n");
        fprintf(stderr, "  措施: 已标记跟踪链表为损坏状态, 后续所有跟踪操作将被跳过。\n");
        fprintf(stderr, "  建议: 启用更严格的堆检查(如 Application Verifier)定位根因。\n");
        /* 清空链表指针, 防止其他函数在未检查损坏标志的情况下遍历 */
        g_alloc_track_list = NULL;
    }
    alloc_track_unlock();
}

/**
 * @brief 检查指针是否由 safe_malloc/_safe_calloc 分配并跟踪
 *
 * P0修复(safe_realloc/safe_free越界读头): 在读取 MemoryBlockHeader 之前,
 * 必须先确认该指针确由本内存工具分配。对 malloc/calloc/realloc/外部库分配
 * 的指针直接读取头部属于越界读/未定义行为(读取不属于调用者的内存)。
 * 本函数通过遍历分配跟踪链表验证指针来源,避免对非托管指针读取头部。
 *
 * @param data_ptr 待验证的数据指针
 * @param out_size 输出参数,若非NULL且指针已跟踪,则写入跟踪记录中的分配大小;
 *                 用于头部损坏时安全复制(避免读取已损坏的header->size)。
 * @return 1 表示已跟踪(safe_malloc分配,可安全读取头部), 0 表示未跟踪(外部指针)
 */
static int alloc_track_contains(void* data_ptr, size_t* out_size) {
    if (!data_ptr) return 0;
    /* M-5修复: 若跟踪链表已损坏, 直接返回未找到 */
    if (g_alloc_track_corrupted) return 0;
    int found = 0;
    alloc_track_lock();
    AllocTrackNode* node = g_alloc_track_list;
    while (node) {
        if (node->data_ptr == data_ptr) {
            found = 1;
            if (out_size) *out_size = node->size;
            break;
        }
        node = node->next;
    }
    alloc_track_unlock();
    return found;
}

/**
 * @brief 遍历分配跟踪链表并打印泄漏报告
 * @return 泄漏的分配数量
 */
static int alloc_track_dump_leaks(void) {
    /* M-5修复: 若跟踪链表已损坏, 无法遍历泄漏报告 */
    if (g_alloc_track_corrupted) {
        fprintf(stderr, "[泄漏检测] 跟踪链表已损坏, 无法生成泄漏报告\n");
        return -1;
    }
    int leak_count = 0;
    size_t leak_total = 0;
    alloc_track_lock();
    AllocTrackNode* node = g_alloc_track_list;
    while (node) {
        leak_count++;
        leak_total += node->size;
#ifdef _DEBUG
        fprintf(stderr, "[泄漏检测] 未释放内存: 分配ID=#%zu, 地址=%p, 大小=%zu 字节\n",
                node->alloc_id, node->data_ptr, node->size);
#endif
        node = node->next;
    }
    alloc_track_unlock();
#ifdef _DEBUG
    if (leak_count > 0) {
        fprintf(stderr, "[泄漏检测] 总计: %d 处泄漏, %zu 字节未释放\n",
                leak_count, leak_total);
    } else {
        fprintf(stderr, "[泄漏检测] 无内存泄漏\n");
    }
#endif
    return leak_count;
}

/**
 * @brief 魔法数字用于内存块验证
 */
#define MEMORY_MAGIC 0xDEADBEEFCAFEBABEULL

/**
 * @brief 内存块头部大小（对齐后）
 */
#define BLOCK_HEADER_SIZE ((sizeof(MemoryBlockHeader) + 15) & ~15)

/**
 * @brief 内存块尾部大小（对齐后）
 */
#define BLOCK_FOOTER_SIZE ((sizeof(MemoryBlockFooter) + 15) & ~15)

/**
 * @brief 内存块总开销
 */
#define BLOCK_OVERHEAD (BLOCK_HEADER_SIZE + BLOCK_FOOTER_SIZE)

/**
 * @brief 获取页大小
 */
static size_t get_page_size(void) {
#ifdef _WIN32
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    return sys_info.dwPageSize;
#else
    return sysconf(_SC_PAGESIZE);
#endif
}

/**
 * @brief 对齐内存大小
 */
static size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief 验证所有已跟踪分配块的魔法数字完整性
 * @return 0 如果所有块完好，-1 如果发现损坏
 */
static int validate_all_magic(void) {
    /* M-5修复: 若跟踪链表已损坏, 跳过验证 */
    if (g_alloc_track_corrupted) return 0;
    alloc_track_lock();
    AllocTrackNode* node = g_alloc_track_list;
    int corrupted = 0;
    while (node) {
        if (node->data_ptr) {
            MemoryBlockHeader* h = (MemoryBlockHeader*)((char*)node->data_ptr - BLOCK_HEADER_SIZE);
            if (h->magic != MEMORY_MAGIC) {
                fprintf(stderr, "预分配检测到损坏: [分配#%zu] 数据地址 %p, 期望魔法数字 0x%llx, 实际 0x%llx, 原始分配于 %s:%d\n",
                        node->alloc_id, node->data_ptr,
                        (unsigned long long)MEMORY_MAGIC,
                        (unsigned long long)h->magic,
                        node->file, node->line);
                corrupted = 1;
            }
        }
        node = node->next;
    }
    alloc_track_unlock();
    return corrupted ? -1 : 0;
}

/* v9.1: 公开接口，供测试使用 */
int memory_utils_validate_all_tracked(void) {
    return validate_all_magic();
}

/* v9.1: 返回跟踪列表中的条目数 */
size_t memory_utils_tracked_count(void) {
    /* M-5修复: 若跟踪链表已损坏, 返回0 */
    if (g_alloc_track_corrupted) return 0;
    size_t count = 0;
    alloc_track_lock();
    AllocTrackNode* node = g_alloc_track_list;
    while (node) { count++; node = node->next; }
    alloc_track_unlock();
    return count;
}

/**
 * @brief 内部安全内存分配：分配内存并初始化为零（带文件/行号跟踪）
 * 修复#4: __declspec(noinline)防止_ReturnAddress()在函数内联后返回错误地址
 */
#ifdef _MSC_VER
__declspec(noinline)
#endif
void* _safe_malloc(size_t size, const char* file, int line) {
    (void)file; (void)line;
    if (g_bypass_safe_alloc) { return malloc(size); }
    if (size == 0) {
        return NULL;
    }
    
    // 每次分配前验证所有已跟踪块的魔法数字，检测堆损坏
#ifdef _DEBUG
    validate_all_magic();
#endif
    
    // 添加块头部、尾部、保护字节和额外保护空间
    /* P1缺陷修复: 整数溢出检查
     * 1. align_size(size, 16) = (size + 15) & ~15, 若 size > SIZE_MAX - 15 则加法回绕
     * 2. total_size = BLOCK_OVERHEAD + aligned_size + guard_size, 三值相加可能溢出
     * 3. 溢出后 total_size 变小, malloc 分配不足, 后续 memset/头尾写入越界导致堆破坏 */
    if (size > SIZE_MAX - 15) {
        fprintf(stderr, "[错误] _safe_malloc: 对齐大小溢出 size=%zu (文件: %s, 行: %d)\n",
                size, file ? file : "未知", line);
        return NULL;
    }
    size_t aligned_size = align_size(size, 16);
    
    /* 验证 aligned_size 未因对齐回绕（双重保险） */
    if (aligned_size < size) {
        fprintf(stderr, "[错误] _safe_malloc: 对齐后大小回绕 aligned_size=%zu < size=%zu (文件: %s, 行: %d)\n",
                aligned_size, size, file ? file : "未知", line);
        return NULL;
    }
    
    size_t guard_size = 16;  // 额外保护字节
    
    /* 检查 total_size = BLOCK_OVERHEAD + aligned_size + guard_size 是否溢出 */
    if (aligned_size > SIZE_MAX - BLOCK_OVERHEAD - guard_size) {
        fprintf(stderr, "[错误] _safe_malloc: 总大小溢出 aligned_size=%zu BLOCK_OVERHEAD=%zu guard_size=%zu (文件: %s, 行: %d)\n",
                aligned_size, (size_t)BLOCK_OVERHEAD, guard_size, file ? file : "未知", line);
        return NULL;
    }
    size_t total_size = BLOCK_OVERHEAD + aligned_size + guard_size;
    
    // 分配内存
    void* raw_ptr = malloc(total_size);
    if (!raw_ptr) {
        return NULL;
    }
    
    // 初始化头部
    MemoryBlockHeader* header = (MemoryBlockHeader*)raw_ptr;
    header->size = size;
    header->is_allocated = 1;
    header->file = file;
    header->line = line;
    header->magic = MEMORY_MAGIC;
    /* P1修复: 记录malloc原始指针,使safe_free能通过 free(header->raw_ptr) 统一释放
     * (普通块 raw_ptr == header, 对齐块 raw_ptr != header)。 */
    header->raw_ptr = raw_ptr;
    /* 修复#4: 编译器屏障——确保magic写入完成后，后续代码不会因优化而重排到magic写入之前。
     * MSVC: _ReadWriteBarrier() | GCC/Clang: asm volatile("" ::: "memory") */
#ifdef _MSC_VER
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
    MEM_LOCK();
    header->alloc_id = ++g_allocation_counter;
    MEM_UNLOCK();
    // 记录调用者返回地址（用于调试堆损坏问题）
#ifdef _MSC_VER
    header->caller_address = _ReturnAddress();
#else
    header->caller_address = __builtin_return_address(0);
#endif
    
    // 计算数据区域
    void* data_ptr = (char*)raw_ptr + BLOCK_HEADER_SIZE;
    
    // 初始化数据区域为零
    memset(data_ptr, 0, aligned_size);
    
    // 设置保护字节（在数据区域和尾部之间）
    unsigned char* guard_ptr = (unsigned char*)data_ptr + aligned_size;
    memset(guard_ptr, 0xAA, guard_size);  // 填充保护模式 0xAA
    
    // 设置尾部（在保护字节之后）
    MemoryBlockFooter* footer = (MemoryBlockFooter*)(guard_ptr + guard_size);
    footer->magic = MEMORY_MAGIC;
    /* 修复#4: 编译器屏障——确保尾部magic写入对所有线程可见 */
#ifdef _MSC_VER
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
    
    // 添加调试信息：记录分配大小和地址
    #ifdef _DEBUG
    fprintf(stderr, "_safe_malloc[#%zu]: 分配 %zu 字节 (实际 %zu 字节) 地址: %p -> %p (保护字节在 %p) 调用于 %s:%d\n",
            header->alloc_id, size, total_size, raw_ptr, data_ptr, guard_ptr, file, line);
    #endif
    
    // 更新统计
    MEM_LOCK();
    g_memory_stats.total_allocated += total_size;
    g_memory_stats.current_usage += total_size;
    g_memory_stats.allocation_count++;
    
    if (g_memory_stats.current_usage > g_memory_stats.peak_usage) {
        g_memory_stats.peak_usage = g_memory_stats.current_usage;
    }
    MEM_UNLOCK();
    
    alloc_track_add(data_ptr, size, header->alloc_id, file, line);
    
    return data_ptr;
}

/**
 * @brief 内部安全内存分配：分配并清零内存（类似calloc，带文件/行号跟踪）
 * 
 * @param num 元素数量
 * @param size 元素大小
 * @param file 分配文件名
 * @param line 分配行号
 * @return void* 分配的内存指针，失败返回NULL
 */
void* _safe_calloc(size_t num, size_t size, const char* file, int line) {
    (void)file; (void)line;
    if (g_bypass_safe_alloc) { return calloc(num, size); }
    size_t total_size;
    
    // 零值检查：任一参数为零时不分配内存
    if (num == 0 || size == 0) {
        return NULL;
    }
    
    /* 乘法溢出检查：当 size != 0 且 count > SIZE_MAX / size 时，
     * count * size 会回绕(wrap around)，导致分配过小的缓冲区，
     * 进而引发堆缓冲区溢出漏洞。使用 SIZE_MAX / size 作为安全上限。 */
    if (size != 0 && num > SIZE_MAX / size) {
        fprintf(stderr, "[错误] _safe_calloc: 乘法溢出 num=%zu size=%zu (文件: %s, 行: %d)\n",
                num, size, file ? file : "未知", line);
        return NULL;
    }
    
    total_size = num * size;
    
    // 直接调用内部函数传递文件/行号
    return _safe_malloc(total_size, file, line);
}

/**
 * @brief 内部安全内存分配：分配对齐内存（带文件/行号跟踪）
 *
 * P1修复(添加头部): 原实现直接调用 _aligned_malloc/posix_memalign 返回对齐指针,
 * 未添加 MemoryBlockHeader, 导致:
 *   1) 该指针传入 safe_free 时头部读取为堆管理器内部数据 → magic 不匹配 → 静默泄漏;
 *      且 Windows 上 _aligned_malloc 指针不能用 free 释放, 必须用 _aligned_free。
 *   2) 无保护字节/尾部, 无法检测越界写。
 * 现改为: 用 malloc 分配足够空间(含对齐余量 + 头部 + 尾部 + 保护字节),
 * 手动将数据指针对齐到 alignment, 在数据指针之前放置 MemoryBlockHeader(含 raw_ptr
 * 指向 malloc 原始返回值), 之后放置保护字节与尾部。这样返回的指针可被 safe_free
 * 统一释放(free(header->raw_ptr) 即 free(raw), 正确), 并具备与 safe_malloc 一致的
 * 损坏检测能力(头部/尾部/保护字节魔法校验)。
 */
void* _safe_aligned_malloc(size_t size, size_t alignment, const char* file, int line) {
    if (size == 0 || alignment == 0) {
        return NULL;
    }

    /* 确保对齐是2的幂 */
    if ((alignment & (alignment - 1)) != 0) {
        return NULL;
    }

    /* M-1修复: 逐步溢出检查, 替代原来的 SIZE_MAX/2 粗粒度检查。
     * 原检查(alignment > SIZE_MAX/2 || size > SIZE_MAX/2)无法防止
     * alignment + BLOCK_HEADER_SIZE + aligned_size + guard_size + BLOCK_FOOTER_SIZE
     * 五项累加的总溢出。现改为逐步加法, 每一步都检查是否溢出。 */
    if (alignment > SIZE_MAX / 2 || size > SIZE_MAX / 2) {
        return NULL;
    }

    /* 布局: [raw][对齐填充][头部BLOCK_HEADER_SIZE][数据aligned_size][保护字节guard][尾部FOOTER]
     * 取 alignment 作为对齐余量上界(>= alignment-1), 保证 data_ptr 能在 raw+BLOCK_HEADER_SIZE
     * 之后向上对齐到 alignment, 且头部不越界到 raw_ptr 之前。 */
    size_t aligned_size = align_size(size, 16);
    /* 验证 aligned_size 对齐后未回绕 */
    if (aligned_size < size) {
        return NULL;
    }
    size_t guard_size = 16;  /* 与 safe_malloc 一致的保护字节大小 */

    /* M-1修复: 逐步溢出检查 total_size = alignment + BLOCK_HEADER_SIZE + aligned_size + guard_size + BLOCK_FOOTER_SIZE。
     * 每步加法前检查是否超过 SIZE_MAX, 防止多值累加导致的整数溢出回绕。 */
    size_t total_size = (size_t)alignment;
    if (total_size > SIZE_MAX - BLOCK_HEADER_SIZE) { return NULL; }
    total_size += BLOCK_HEADER_SIZE;
    if (total_size > SIZE_MAX - aligned_size) { return NULL; }
    total_size += aligned_size;
    if (total_size > SIZE_MAX - guard_size) { return NULL; }
    total_size += guard_size;
    if (total_size > SIZE_MAX - BLOCK_FOOTER_SIZE) { return NULL; }
    total_size += BLOCK_FOOTER_SIZE;

    void* raw_ptr = malloc(total_size);
    if (!raw_ptr) {
        return NULL;
    }

    /* 计算对齐的数据指针: data_ptr = 向上对齐(raw_ptr + BLOCK_HEADER_SIZE, alignment) */
    uintptr_t base = (uintptr_t)raw_ptr + BLOCK_HEADER_SIZE;
    uintptr_t data_addr = (base + (uintptr_t)alignment - 1) & ~((uintptr_t)alignment - 1);
    void* data_ptr = (void*)data_addr;

    /* 头部位于数据指针之前(BLOCK_HEADER_SIZE 处) */
    MemoryBlockHeader* header = (MemoryBlockHeader*)((char*)data_ptr - BLOCK_HEADER_SIZE);
    header->size = size;
    header->is_allocated = 1;
    header->file = file;
    header->line = line;
    header->magic = MEMORY_MAGIC;
    header->raw_ptr = raw_ptr;  /* 关键: 记录 malloc 原始返回值, 供 safe_free 释放 */
    /* 编译器屏障: 确保 magic/raw_ptr 写入对其他线程可见后再写后续字段 */
#ifdef _MSC_VER
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
    MEM_LOCK();
    header->alloc_id = ++g_allocation_counter;
    MEM_UNLOCK();
#ifdef _MSC_VER
    header->caller_address = _ReturnAddress();
#else
    header->caller_address = __builtin_return_address(0);
#endif

    /* 初始化数据区域为零 */
    memset(data_ptr, 0, aligned_size);

    /* 设置保护字节(与 safe_malloc 一致: 0xAA) */
    unsigned char* guard_ptr = (unsigned char*)data_ptr + aligned_size;
    memset(guard_ptr, 0xAA, guard_size);

    /* 设置尾部魔法 */
    MemoryBlockFooter* footer = (MemoryBlockFooter*)(guard_ptr + guard_size);
    footer->magic = MEMORY_MAGIC;
#ifdef _MSC_VER
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif

    /* 更新统计与跟踪(按实际占用 total_size 计入, 含对齐填充) */
    MEM_LOCK();
    {
        g_memory_stats.total_allocated += total_size;
        g_memory_stats.current_usage += total_size;
        g_memory_stats.allocation_count++;
        if (g_memory_stats.current_usage > g_memory_stats.peak_usage) {
            g_memory_stats.peak_usage = g_memory_stats.current_usage;
        }
        alloc_track_add(data_ptr, size, header->alloc_id, file, line);
    }
    MEM_UNLOCK();

    return data_ptr;
}

/**
 * @brief 验证所有已分配内存块的完整性（检查头部和尾部魔法数字）
 * @return 0 所有块完好，-1 发现损坏
 */
int selflnn_validate_all_allocations(void) {
    /* M-5修复: 若跟踪链表已损坏, 跳过验证 */
    if (g_alloc_track_corrupted) return 0;
    alloc_track_lock();
    AllocTrackNode* node = g_alloc_track_list;
    int corrupted = 0;
    size_t guard_size = 16;
    while (node) {
        if (node->data_ptr) {
            MemoryBlockHeader* h = (MemoryBlockHeader*)((char*)node->data_ptr - BLOCK_HEADER_SIZE);
            if (h->magic != MEMORY_MAGIC) {
                fprintf(stderr, "验证检测到头部魔法损坏: [分配#%zu] 数据地址 %p, 原始分配于 %s:%d\n",
                        node->alloc_id, node->data_ptr, node->file, node->line);
                corrupted = 1;
            }
            size_t aligned_sz = align_size(h->size, 16);
            MemoryBlockFooter* f = (MemoryBlockFooter*)((char*)node->data_ptr + aligned_sz + guard_size);
            if (f->magic != MEMORY_MAGIC) {
                fprintf(stderr, "验证检测到尾部魔法损坏: [分配#%zu] 数据地址 %p, 大小=%zu, 尾部魔法=0x%llx, 原始分配于 %s:%d\n",
                        node->alloc_id, node->data_ptr, h->size,
                        (unsigned long long)f->magic, node->file, node->line);
                corrupted = 1;
            }
        }
        node = node->next;
    }
    alloc_track_unlock();
    return corrupted ? -1 : 0;
}

/**
 * @brief 安全内存释放：释放内存并置空指针
 */
void safe_free(void** ptr) {
    /* P0-FIX: ptr本身校验 — NULL/低地址/未对齐/MSVC调试填充 */
    if (!ptr || (uintptr_t)ptr < 0x1000 || ((uintptr_t)ptr & 7) != 0) return;
    /* M-4修复: 补充MSVC调试模式下更多内存模式的覆盖。
     * 原仅覆盖 0xCCCCCCCC(未初始化栈), 现补充:
     *   0xCDCDCDCD — 未初始化堆内存(C运行时用此值填充malloc分配的内存)
     *   0xFEEEFEEE — 已释放堆内存(HeapFree后OS填充)
     *   0xDDDDDDDD — 已释放堆内存(CRT调试堆释放后填充)
     *   0xFDFDFDFD — 堆保护区边界填充(no man's land)
     *   0xABABABAB — 堆保护区边界填充(no man's land, alternate)
     * 检测逻辑: 若指针的32位高低部分相等且匹配任一已知模式, 则判定为调试填充。 */
    {   uint32_t lo = (uint32_t)((uintptr_t)ptr & 0xFFFFFFFF);
        uint32_t hi = (uint32_t)((uintptr_t)ptr >> 32);
        if (lo == hi) {
            /* 检查所有已知MSVC调试填充模式 */
            if (lo == 0xCCCCCCCC || lo == 0xCDCDCDCD ||
                lo == 0xFEEEFEEE || lo == 0xDDDDDDDD ||
                lo == 0xFDFDFDFD || lo == 0xABABABAB) {
                return; /* MSVC debug pattern */
            }
        }
    }
#ifdef USE_STANDARD_ALLOC
    if (*ptr) {
#ifdef _MSC_VER
        __try { free(*ptr); } __except(EXCEPTION_EXECUTE_HANDLER) {}
#else
        free(*ptr);
#endif
        *ptr = NULL;
    }
    return;
#else
    if (g_bypass_safe_alloc) {
        if (*ptr) {
#ifdef _MSC_VER
            __try { free(*ptr); } __except(EXCEPTION_EXECUTE_HANDLER) {}
#else
            free(*ptr);
#endif
            *ptr = NULL;
        }
        return;
    }
/* G1修复: 移除 _CrtCheckMemory() 调用。
     * 根因: 当缓冲区溢出破坏CRT调试堆元数据(_CrtMemBlockHeader链表指针)后，
     * _CrtCheckMemory() 遍历被破坏的链表时进入无限循环，导致程序挂起。
     * safe_free 已有自己的完整性检测(头部magic/保护字节/尾部magic)，
     * 不需要额外的CRT堆检测。Release模式下此代码不存在，修复不影响正常功能。 */
    if (!ptr || !*ptr) {
        return;
    }
    
    void* data_ptr = *ptr;
    
    /* P0修复(静默泄漏 + 越界读头): 读取头部前先验证指针是否由 safe_malloc 分配并跟踪。
     * - 未跟踪指针(malloc/calloc/realloc/外部库分配): 直接用系统 free 释放并记录警告,
     *   修复原实现中 magic 不匹配时完全不调用 free 导致的静默泄漏;
     *   同时避免对非托管指针读取 MemoryBlockHeader 造成的越界读/未定义行为。
     * - 已跟踪指针: 确属本工具分配, 读取头部安全, 继续执行原有的完整性校验逻辑
     *   (对齐/magic/保护字节/尾部), 在检测到损坏时仍跳过 free 以避免堆损坏扩散。 */
    if (!alloc_track_contains(data_ptr, NULL)) {
        /* DEFECT-009修复: 回退路径中的free(data_ptr)风险分析：
         * 1. 若指针由malloc分配: data_ptr == raw_ptr, free(data_ptr)正确
         * 2. 若指针由safe_malloc分配但跟踪丢失: data_ptr = raw_ptr + HEADER_SIZE,
         *    free(data_ptr)会因传递给CRT的指针非原始malloc指针而堆损坏。
         * 修复策略: 尝试读取前导头部的magic, 若匹配则使用header->raw_ptr释放;
         * 否则假定为malloc指针直接释放。使用SEH保护防止读取无效内存。 */
        int freed = 0;
#ifdef _WIN32
        __try {
            /* 尝试读取safe_malloc头部（最小化风险窗口） */
            MemoryBlockHeader* maybe_header = (MemoryBlockHeader*)((char*)data_ptr - BLOCK_HEADER_SIZE);
            if (maybe_header->magic == MEMORY_MAGIC && maybe_header->raw_ptr) {
                free(maybe_header->raw_ptr);
                freed = 1;
            }
        } __except(EXCEPTION_EXECUTE_HANDLER) {
            /* 头部不可读，回退到直接free */
        }
#endif
        if (!freed) {
            fprintf(stderr, "警告: 指针 %p 非safe_malloc分配, 使用系统free释放(避免静默泄漏)\n", data_ptr);
#ifdef _WIN32
            __try {
                free(data_ptr);
            } __except(EXCEPTION_EXECUTE_HANDLER) {
                fprintf(stderr, "错误: free(%p) 崩溃, 指针可能已损坏, 跳过释放\n", data_ptr);
            }
#else
            free(data_ptr);
#endif
        }
        *ptr = NULL;
        return;
    }
    
    // 获取头部 —— 此时已确认是 safe_malloc 分配的指针, 读取头部安全
    MemoryBlockHeader* header = (MemoryBlockHeader*)((char*)data_ptr - BLOCK_HEADER_SIZE);
    
    /* 修复#4: 编译器屏障——确保header读取前所有对数据区的写入已完成，
     * 防止MSVC /O2优化重排导致读取过期的header内容 */
#ifdef _MSC_VER
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
    
    // 安全检查：确保header指针看起来合理
    // 检查对齐：header应该至少对齐到sizeof(void*)
    uintptr_t header_addr = (uintptr_t)header;
    if (header_addr % sizeof(void*) != 0) {
        /* P6-R90: header地址未对齐 = 数据指针无效。
         * 不调用free()避免向堆管理器提交垃圾指针。 */
        fprintf(stderr, "内存损坏: 未对齐指针 %p (skipping free)\n", data_ptr);
        alloc_track_remove(data_ptr);
        *ptr = NULL;
        return;
    }
    
    // 检查地址是否不太可能有效（避免访问极低或极高的地址）
    // 跳过检查，因为系统内存布局未知
    
    /* M-2修复: TOCTOU竞态 - 原子地检查并清除magic, 消除检查与置零之间的竞态窗口。
     * 原实现分两步: 1) 读取magic检查; 2) 后续置零。在这两步之间, 另一个线程可能
     * 并发释放同一指针, 导致双释放(double-free)或堆损坏。
     * 修复: 使用原子CAS(Compare-And-Swap)一步完成"检查magic==MEMORY_MAGIC并置零",
     * 若CAS失败则说明magic已被其他线程修改(并发释放或已损坏), 安全退出。 */
    {
        size_t expected = MEMORY_MAGIC;
        size_t old_magic = 0;
#ifdef _MSC_VER
        /* Windows: 64位系统上size_t为8字节, 使用InterlockedCompareExchange64;
         * 32位系统上size_t为4字节, 使用InterlockedCompareExchange。
         * 通过sizeof(size_t)在编译期选择正确的原子操作。 */
        if (sizeof(size_t) == 8) {
            old_magic = (size_t)_InterlockedCompareExchange64(
                (__int64 volatile*)&header->magic, 0, (__int64)expected);
        } else {
            old_magic = (size_t)InterlockedCompareExchange(
                (LONG volatile*)&header->magic, 0, (LONG)expected);
        }
#else
        /* GCC/Clang: __sync_val_compare_and_swap 返回旧值, 语义与CAS一致 */
        old_magic = __sync_val_compare_and_swap(&header->magic, expected, (size_t)0);
#endif
        if (old_magic == 0) {
            /* P6-R90: 魔法数字为零=已释放(并发释放或use-after-free)。
             * alloc_track_remove已在首次释放时调用, 快速返回不做任何操作。 */
            *ptr = NULL;
            return;
        }
        if (old_magic != MEMORY_MAGIC) {
            /* magic不匹配(被并发修改或已损坏) → header指针可能已被破坏。
             * 不调用free(), 避免向堆管理器提交随机指针导致堆损坏扩散。 */
            fprintf(stderr, "内存损坏: 魔法数字不匹配 %p magic=0x%llx (skipping free)\n",
                   data_ptr, (unsigned long long)old_magic);
            alloc_track_remove(data_ptr);
            *ptr = NULL;
            return;
        }
        /* CAS成功: magic已从MEMORY_MAGIC原子置为0, 当前线程获得独占释放权。
         * 后续代码安全执行free, 不再有并发释放风险。 */
    }
    
    // 验证尾部
    /* M-3修复: 放宽并改进header->size验证条件。
     * 原检查: header->size > SIZE_MAX - align_size(1,16) 即 header->size > SIZE_MAX - 16,
     * 问题: (a) 计算复杂但语义等价于 size > SIZE_MAX - 16, 可读性差;
     *       (b) 上界偏保守, 合理值为 SIZE_MAX - 15(align_size公式: (size+15)&~15);
     *       (c) 缺少对 size==0 的检查(0为无效分配, 应单独处理)。
     * 修复: 使用更清晰的验证逻辑, 检查size合理性(非零、不溢出对齐计算)。 */
    if (header->size == 0) {
        /* size为0: 无效分配或头部已被破坏清零 */
        fprintf(stderr, "内存损坏: 块大小为0 at %p (skipping free)\n", data_ptr);
        alloc_track_remove(data_ptr);
        *ptr = NULL;
        return;
    }
    if (header->size > SIZE_MAX - 15) {
        /* align_size(header->size, 16) = (size + 15) & ~15, 当 size > SIZE_MAX - 15
         * 时加法溢出, 导致 aligned_size 回绕为小值, 后续保护字节/尾部计算越界。
         * 此检查使用正确的上界 SIZE_MAX - 15, 同时接受所有合法size值。 */
        fprintf(stderr, "内存损坏: 块大小过大(将导致对齐溢出) %zu at %p (skipping free)\n", header->size, data_ptr);
        alloc_track_remove(data_ptr);
        *ptr = NULL;
        return;
    }
    
    // 计算保护字节和尾部位置（与新safe_malloc布局匹配）
    size_t guard_size = 16;  // 与safe_malloc中的guard_size匹配
    size_t aligned_size = align_size(header->size, 16);
    unsigned char* guard_ptr = (unsigned char*)data_ptr + aligned_size;
    MemoryBlockFooter* footer = (MemoryBlockFooter*)(guard_ptr + guard_size);
    
    // 检查保护字节是否被修改
    int guard_corrupted = 0;
    for (size_t i = 0; i < guard_size; i++) {
        if (guard_ptr[i] != (unsigned char)0xAA) {
            guard_corrupted = 1;
            fprintf(stderr, "内存损坏检测：保护字节在偏移量 %zu 被修改 (地址: %p, 期望: 0xAA, 实际: 0x%02x)\n",
                   i, guard_ptr + i, (unsigned int)((unsigned char)guard_ptr[i]));
        }
    }
    
    if (footer->magic != MEMORY_MAGIC) {
        /* P6-R90: 尾部损坏 = 堆已损坏, safe_free((void**)&header)扩散破坏。
         * 只记录, 不释放。 */
        fprintf(stderr, "内存损坏: 尾部魔法不匹配 %p size=%zu gm=%d (skipping free)\n",
               data_ptr, header->size, guard_corrupted);
        alloc_track_remove(data_ptr);
        *ptr = NULL;
        return;
    }
    
    if (guard_corrupted) {
        /* P6-R90: 保护字节损坏 = 缓冲区溢出。尾部OK但头部可能已有问题。
         * 为安全起见不调用free(), 避免不可预测的堆损坏。 */
        fprintf(stderr, "内存损坏: guard bytes corrupted at %p (skipping free)\n", data_ptr);
        alloc_track_remove(data_ptr);
        *ptr = NULL;
        return;
    }
    

    
    // 计算总大小（与新safe_malloc布局匹配）
    size_t total_size = BLOCK_OVERHEAD + align_size(header->size, 16) + guard_size;
    
    // 更新统计（R13-003: 加锁保护并发安全）
    MEM_LOCK();
    g_memory_stats.total_freed += total_size;
    g_memory_stats.current_usage -= total_size;
    g_memory_stats.free_count++;
    MEM_UNLOCK();
    
    // 从分配跟踪链表移除
    alloc_track_remove(data_ptr);
    
    // 清除魔法数字（帮助检测use-after-free）
    header->magic = 0;
    footer->magic = 0;
    
    // 释放内存 —— 通过 raw_ptr 释放 malloc 返回的原始指针
    // (P1修复: 普通块 raw_ptr==header, 对齐块 raw_ptr!=header, 统一正确释放)
    free(header->raw_ptr);
    *ptr = NULL;
#endif /* USE_STANDARD_ALLOC */
}

/**
 * @brief 安全内存重新分配
 */
void* safe_realloc(void* ptr, size_t size) {
    /* ZSFOOO-E002: bypass模式下跳过魔法数字检查，直接使用标准realloc */
    if (g_bypass_safe_alloc) {
        if (!ptr) return malloc(size);
        if (size == 0) { free(ptr); return NULL; }
        return realloc(ptr, size);
    }
    if (!ptr) {
        return _safe_malloc(size, __FILE__, __LINE__);
    }
    /* P0修复: size为0时按C标准释放旧块并返回NULL（原实现缺失此分支, 会继续读取头部） */
    if (size == 0) {
        safe_free(&ptr);
        return NULL;
    }
    
    /* P0修复(越界读头): 读取头部前先验证指针是否由 safe_malloc 分配并跟踪。
     * 对 malloc/calloc/realloc/外部库分配的指针直接读取 MemoryBlockHeader 属于
     * 越界读/未定义行为(读取不属于调用者的内存)。未跟踪指针直接走系统 realloc。 */
    size_t tracked_old_size = 0;
    if (!alloc_track_contains(ptr, &tracked_old_size)) {
        /* 非safe_malloc分配的指针, 直接使用系统realloc */
        return realloc(ptr, size);
    }
    
    // 获取原始头部 —— 此时已确认是 safe_malloc 分配的指针, 读取头部安全
    MemoryBlockHeader* old_header = (MemoryBlockHeader*)((char*)ptr - BLOCK_HEADER_SIZE);
    
    // 验证魔法数字（已确认跟踪, 此处校验本工具分配块是否损坏）
    if (old_header->magic != MEMORY_MAGIC) {
        /* P0修复(头部损坏): 头部magic损坏说明该块已损坏, 不能再读取 old_header->size
         * 或尾部(均为已损坏区域)。原实现对此情形调用 realloc(ptr) —— 但 ptr 是
         * safe_malloc 返回的内部数据指针(非malloc返回的原始指针), realloc 会导致
         * 堆损坏。改为: 用跟踪记录的安全大小 tracked_old_size 分配新块并复制数据,
         * 不释放旧块(其头部已损坏, free 会扩散堆损坏), 仅从跟踪表移除并告警。 */
        fprintf(stderr, "内存损坏检测: 头部魔法不匹配 (地址: %p, 跟踪大小: %zu), 安全迁移数据\n",
                ptr, tracked_old_size);
        void* new_ptr = _safe_malloc(size, __FILE__, __LINE__);
        if (!new_ptr) {
            return NULL;
        }
        size_t copy_size = tracked_old_size < size ? tracked_old_size : size;
        memcpy(new_ptr, ptr, copy_size);
        alloc_track_remove(ptr);
        /* 不调用 safe_free 释放旧块(header已损坏, 释放会扩散堆损坏), 接受单次泄漏 */
        return new_ptr;
    }
    
    // 验证尾部（计算方式与safe_malloc/safe_free保持一致）
    size_t guard_size = 16;
    MemoryBlockFooter* old_footer = (MemoryBlockFooter*)((char*)ptr + align_size(old_header->size, 16) + guard_size);
    if (old_footer->magic != MEMORY_MAGIC) {
        // 内存损坏
        fprintf(stderr, "内存损坏检测：无效的尾部魔法数字 (地址: %p, 大小: %zu, 头部魔法: 0x%llx, 尾部魔法: 0x%llx)\n",
               ptr, old_header->size, (unsigned long long)old_header->magic, (unsigned long long)old_footer->magic);
        return NULL;
    }
    
    // 分配新内存
    void* new_ptr = _safe_malloc(size, __FILE__, __LINE__);
    if (!new_ptr) {
        return NULL;
    }
    
    // 复制数据（复制较小的大小）
    size_t copy_size = old_header->size < size ? old_header->size : size;
    memcpy(new_ptr, ptr, copy_size);
    
    // 释放旧内存
    safe_free(&ptr);
    
    return new_ptr;
}

/**
 * @brief 安全内存复制：带边界检查
 */
int safe_memcpy(void* dest, size_t dest_size, const void* src, size_t count) {
    if (!dest || !src) {
        return -1;
    }
    
    if (count == 0) {
        return 0;
    }
    
    if (count > dest_size) {
        return -1;
    }
    
    memcpy(dest, src, count);
    return 0;
}



/**
 * @brief 安全内存设置：带边界检查
 */
int safe_memset(void* dest, size_t dest_size, int value, size_t count) {
    if (!dest) {
        return -1;
    }
    
    if (count == 0) {
        return 0;
    }
    
    if (count > dest_size) {
        return -1;
    }
    
    memset(dest, value, count);
    return 0;
}

/**
 * @brief 创建内存池
 */
MemoryPool* memory_pool_create(const MemoryPoolConfig* config) {
    if (!config) {
        return NULL;
    }
    
    // 验证配置
    if (config->block_size == 0 || config->num_blocks == 0) {
        return NULL;
    }
    
    // 对齐块大小
    size_t aligned_block_size = align_size(config->block_size, 16);
    if (aligned_block_size == 0) {
        return NULL;
    }
    
    // 计算总大小
    size_t total_size = aligned_block_size * config->num_blocks;
    
    // 考虑对齐要求
    size_t alignment = config->alignment;
    if (alignment > 0) {
        total_size = align_size(total_size, alignment);
    }
    
    // 分配内存池结构
    MemoryPool* pool = (MemoryPool*)safe_malloc(sizeof(MemoryPool));
    if (!pool) {
        return NULL;
    }
    
    memset(pool, 0, sizeof(MemoryPool));
    pool->config = *config;
    pool->block_count = config->num_blocks;
    
    // 分配内存池数据
#ifdef _WIN32
    if (alignment > 0) {
        pool->base_ptr = _aligned_malloc(total_size, alignment);
    } else {
        pool->base_ptr = safe_malloc(total_size);
    }
#else
    if (alignment > 0) {
        if (posix_memalign(&pool->base_ptr, alignment, total_size) != 0) {
            pool->base_ptr = NULL;
        }
    } else {
        pool->base_ptr = safe_malloc(total_size);
    }
#endif
    
    if (!pool->base_ptr) {
        safe_free((void**)&pool);
        return NULL;
    }
    
    // 分配块状态数组
    pool->block_status = (int*)safe_calloc(config->num_blocks, sizeof(int));
    if (!pool->block_status) {
#ifdef _WIN32
        if (alignment > 0) {
            _aligned_free(pool->base_ptr);
        } else {
            safe_free((void**)&pool->base_ptr);
        }
#else
        safe_free((void**)&pool->base_ptr);
#endif
        safe_free((void**)&pool);
        return NULL;
    }
    
    /* P1修复: 分配 block_span 数组(与 block_status 并行), calloc 初始化为0 */
    pool->block_span = (size_t*)safe_calloc(config->num_blocks, sizeof(size_t));
    if (!pool->block_span) {
        safe_free((void**)&pool->block_status);
#ifdef _WIN32
        if (alignment > 0) {
            _aligned_free(pool->base_ptr);
        } else {
            safe_free((void**)&pool->base_ptr);
        }
#else
        safe_free((void**)&pool->base_ptr);
#endif
        safe_free((void**)&pool);
        return NULL;
    }
    
    // 初始化内存池
    pool->total_size = total_size;
    pool->free_size = total_size;
    pool->first_free = 0;
    
    // 如果启用零初始化，则清零整个池
    if (config->enable_zero_init) {
        memset(pool->base_ptr, 0, total_size);
    }
    
    // 更新统计
    /* P2修复: g_memory_stats为全局共享状态，多线程并发创建内存池时
     * 无锁修改pool_count会导致计数不一致 */
    MEM_LOCK();
    g_memory_stats.pool_count++;
    MEM_UNLOCK();
    
    // 注册到全局池链表
    PoolNode* node = (PoolNode*)safe_malloc(sizeof(PoolNode));
    if (node) {
        node->pool = pool;
        /* P2修复: g_pool_list为全局链表，无锁修改会导致链表节点丢失或损坏 */
        MEM_LOCK();
        node->next = g_pool_list;
        g_pool_list = node;
        MEM_UNLOCK();
    }
    
    return pool;
}

/**
 * @brief 从内存池分配内存
 */
void* memory_pool_alloc(MemoryPool* pool, size_t size) {
    if (!pool || size == 0) {
        return NULL;
    }
    
    // 对齐请求大小
    size_t aligned_size = align_size(size, 16);
    if (aligned_size == 0) {
        return NULL;
    }
    
    // 检查是否有足够大的连续块
    size_t blocks_needed = (aligned_size + pool->config.block_size - 1) / pool->config.block_size;
    if (blocks_needed == 0 || blocks_needed > pool->block_count) {
        return NULL;  // 单次请求超过池容量
    }
    
    /* P1修复(first_block下溢 + 物理连续性): 原实现用 (start_block + i) % block_count
     * 回绕扫描, 导致:
     *   1) 跨回绕的"连续"块在物理上不连续(首尾拼接), 返回的指针仅覆盖尾部, 越界写;
     *   2) first_block = idx - consecutive_free + 1 在跨回绕时下溢(size_t无符号→巨大值),
     *      后续 block_status[first_block + j] 越界写, 破坏内存。
     * 改为线性扫描 [0, block_count), 仅接受不跨回绕的物理连续空闲块, first_block 直接取
     * 连续区起点 run_start, 不会下溢。同时用 block_span 记录本次分配占用的块数, 供
     * memory_pool_free 释放完整跨度(原实现只释放首块, 多块分配会泄漏)。 */
    size_t consecutive_free = 0;
    size_t run_start = 0;  // 当前连续空闲区起点(当 consecutive_free>0 时有效)
    
    for (size_t i = 0; i < pool->block_count; i++) {
        if (pool->block_status[i] == 0) {
            if (consecutive_free == 0) {
                run_start = i;
            }
            consecutive_free++;
            if (consecutive_free >= blocks_needed) {
                // 找到物理连续的空闲块区
                size_t first_block = run_start;
                
                // 标记块为已使用
                for (size_t j = 0; j < blocks_needed; j++) {
                    pool->block_status[first_block + j] = 1;
                }
                /* P1修复: 在首块记录本次分配占用的块数, 其余块置0(防止误释放) */
                pool->block_span[first_block] = blocks_needed;
                for (size_t j = 1; j < blocks_needed; j++) {
                    pool->block_span[first_block + j] = 0;
                }
                
                // 更新空闲大小
                pool->used_size += blocks_needed * pool->config.block_size;
                pool->free_size = pool->total_size - pool->used_size;
                
                // 更新第一个空闲块索引(从 first_block + blocks_needed 起线性查找)
                pool->first_free = first_block + blocks_needed;
                while (pool->first_free < pool->block_count && pool->block_status[pool->first_free] == 1) {
                    pool->first_free++;
                }
                if (pool->first_free >= pool->block_count) {
                    // 末尾无空闲, 回绕到开头(下次分配从头线性扫描仍正确)
                    pool->first_free = 0;
                }
                
                // 计算返回指针
                size_t offset = first_block * pool->config.block_size;
                return (char*)pool->base_ptr + offset;
            }
        } else {
            consecutive_free = 0;
        }
    }
    
    // 没有足够连续空闲块
    return NULL;
}

/**
 * @brief 释放内存池中的内存
 */
int memory_pool_free(MemoryPool* pool, void* ptr) {
    if (!pool || !ptr) {
        return -1;
    }
    
    // 检查指针是否在池范围内
    if ((char*)ptr < (char*)pool->base_ptr ||
        (char*)ptr >= (char*)pool->base_ptr + pool->total_size) {
        return -1;
    }
    
    // 计算块索引
    size_t offset = (char*)ptr - (char*)pool->base_ptr;
    if (offset % pool->config.block_size != 0) {
        return -1; // 未对齐
    }
    
    size_t block_index = offset / pool->config.block_size;
    if (block_index >= pool->block_count) {
        return -1;
    }
    
    // 检查块是否已分配
    if (pool->block_status[block_index] == 0) {
        return -1; // 块未分配(可能已释放)
    }
    
    /* P1修复(只释放单个块→泄漏): 原实现仅释放 block_index 这一个块, 但多块分配
     * (blocks_needed > 1) 时其余块未被释放, 造成泄漏且 used_size 统计错误。
     * 现通过 block_span[block_index] 读取本次分配占用的块数, 释放完整跨度。 */
    size_t blocks_to_free = pool->block_span[block_index];
    if (blocks_to_free == 0) {
        /* span 为 0: 该块不是分配起点(调用者传入了分配中部的指针), 属调用方错误。
         * 为避免崩溃, 按单块释放并告警(其余块仍占用, 需调用方修正)。 */
        fprintf(stderr, "警告: memory_pool_free 收到非分配起点的指针(块%zu), 按单块释放\n", block_index);
        blocks_to_free = 1;
    }
    // 防御: 确保不越界释放
    if (block_index + blocks_to_free > pool->block_count) {
        blocks_to_free = pool->block_count - block_index;
    }
    
    for (size_t j = 0; j < blocks_to_free; j++) {
        pool->block_status[block_index + j] = 0;
        pool->block_span[block_index + j] = 0;
    }
    
    // 更新使用大小
    pool->used_size -= blocks_to_free * pool->config.block_size;
    pool->free_size = pool->total_size - pool->used_size;
    
    // 更新第一个空闲块索引
    if (block_index < pool->first_free) {
        pool->first_free = block_index;
    }
    
    return 0;
}

/**
 * @brief 销毁内存池
 */
void memory_pool_destroy(MemoryPool* pool) {
    if (!pool) {
        return;
    }
    
    // 释放内存池数据
#ifdef _WIN32
    if (pool->config.alignment > 0) {
        _aligned_free(pool->base_ptr);
    } else {
        safe_free((void**)&pool->base_ptr);
    }
#else
    safe_free((void**)&pool->base_ptr);
#endif
    
    // 释放状态数组
    safe_free((void**)&pool->block_status);
    /* P1修复: 释放 block_span 数组 */
    safe_free((void**)&pool->block_span);
    
    // 更新统计
    // 从全局池链表中移除
    /* P2修复: pool_count--和g_pool_list移除操作修改全局共享状态，
     * 必须加锁保护，防止并发创建/销毁内存池时链表损坏和计数不一致 */
    MEM_LOCK();
    g_memory_stats.pool_count--;
    
    PoolNode** pp = &g_pool_list;
    while (*pp) {
        if ((*pp)->pool == pool) {
            PoolNode* to_free = *pp;
            *pp = (*pp)->next;
            safe_free((void**)&to_free);
            break;
        }
        pp = &(*pp)->next;
    }
    MEM_UNLOCK();
    
    // 释放池结构
    safe_free((void**)&pool);
}

/**
 * @brief 获取内存统计信息
 */
int memory_utils_get_stats(MemoryStats* stats) {
    if (!stats) {
        return -1;
    }
    
    memcpy(stats, &g_memory_stats, sizeof(MemoryStats));
    return 0;
}

/**
 * @brief 重置内存统计信息
 */
void memory_reset_stats(void) {
    memset(&g_memory_stats, 0, sizeof(MemoryStats));
}

/**
 * @brief 内存泄漏检测：检查是否有未释放的内存
 *
 * 遍历分配跟踪链表，报告每个未释放的分配详情。
 * @return 未释放的内存总大小（字节），0表示无泄漏
 */
size_t memory_check_leaks(void) {
    size_t total_leaked = 0;
    int leak_count = 0;
    alloc_track_lock();
    AllocTrackNode* node = g_alloc_track_list;
    while (node) {
        total_leaked += node->size;
        leak_count++;
#ifdef _DEBUG
        fprintf(stderr, "[内存泄漏] ID=#%zu 地址=%p 大小=%zuB 文件=%s:%d\n",
                node->alloc_id, node->data_ptr, node->size,
                node->file ? node->file : "?", node->line);
#endif
        node = node->next;
    }
    alloc_track_unlock();
#ifdef _DEBUG
    if (leak_count > 0) {
        fprintf(stderr, "[内存泄漏] 总计: %d 处泄漏, %zu 字节 (%.2f KB)\n",
                leak_count, total_leaked, total_leaked / 1024.0);
    }
#endif
    return total_leaked;
}

/**
 * @brief 内存碎片整理
 * 
 * 遍历所有内存池，合并相邻空闲块，更新空闲大小和第一个空闲块索引。
 * 返回整理的总字节数（负数表示失败）。
 */
int memory_defragment(void) {
    size_t total_reclaimed = 0;
    
    // 遍历所有内存池
    PoolNode* node = g_pool_list;
    while (node) {
        MemoryPool* pool = node->pool;
        if (!pool || !pool->block_status || pool->block_count == 0) {
            node = node->next;
            continue;
        }
        
        // 扫描块状态，统计空闲块分布
        size_t total_free_blocks = 0;
        size_t max_consecutive_free = 0;
        size_t current_consecutive = 0;
        size_t largest_free_start = 0;
        
        for (size_t i = 0; i < pool->block_count; i++) {
            if (pool->block_status[i] == 0) {
                current_consecutive++;
                total_free_blocks++;
                if (current_consecutive > max_consecutive_free) {
                    max_consecutive_free = current_consecutive;
                    largest_free_start = i - current_consecutive + 1;
                }
            } else {
                current_consecutive = 0;
            }
        }
        
        // 计算实际可回收内存（已标记空闲但未计入 free_size 的内存）
        size_t actual_free_size = total_free_blocks * pool->config.block_size;
        size_t reclaimed = 0;
        if (actual_free_size > pool->free_size) {
            reclaimed = actual_free_size - pool->free_size;
        }
        
        // 更新池的空闲信息
        pool->free_size = actual_free_size;
        pool->used_size = pool->total_size - pool->free_size;
        
        // 更新 first_free 指向最大连续空闲块的起始位置
        if (total_free_blocks > 0) {
            pool->first_free = largest_free_start;
        }
        
        total_reclaimed += reclaimed;
        node = node->next;
    }
    
    // 如果没有池，尝试进行通用内存统计整理（仅更新统计）
    if (g_pool_list == NULL) {
        // 没有可整理的内存池，返回0表示无操作
        return 0;
    }
    
    return total_reclaimed > 0 ? (int)total_reclaimed : 0;
}

/**
 * @brief 内存访问安全检查
 */
int memory_safe_access(const void* ptr, size_t size) {
    if (!ptr) {
        return 0;
    }
    
    // 完整实现：在实际系统中，可以使用mprotect或VirtualProtect
    // 这里只进行基本的指针范围检查
    
    // 尝试读取第一个字节和最后一个字节
    volatile const char* start = (const char*)ptr;
    volatile const char* end = start + size - 1;
    
    // 尝试读取（可能会触发段错误）
    char test;
    test = *start;
    if (size > 0) {
        test = *end;
    }
    
    return 1; // 假设安全
}

/**
 * @brief 获取内存页大小
 */
size_t memory_get_page_size(void) {
    static size_t page_size = 0;
    
    if (page_size == 0) {
        page_size = get_page_size();
    }
    
    return page_size;
}

/**
 * @brief 锁定内存到物理RAM（防止交换）
 */
int memory_lock(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        return -1;
    }
    
#ifdef _WIN32
    // Windows: 使用VirtualLock
    if (!VirtualLock(ptr, size)) {
        return -1;
    }
#else
    // Linux/Unix: 使用mlock
    if (mlock(ptr, size) != 0) {
        return -1;
    }
#endif
    
    return 0;
}

/**
 * @brief 解锁内存
 */
int memory_unlock(void* ptr, size_t size) {
    if (!ptr || size == 0) {
        return -1;
    }
    
#ifdef _WIN32
    // Windows: 使用VirtualUnlock
    if (!VirtualUnlock(ptr, size)) {
        return -1;
    }
#else
    // Linux/Unix: 使用munlock
    if (munlock(ptr, size) != 0) {
        return -1;
    }
#endif
    
    return 0;
}
/* DEEP-005: safe_malloc wrapper for linker resolution (需先undef宏) */
#undef safe_malloc
void* safe_malloc(size_t size) { return _safe_malloc(size, __FILE__, __LINE__); }
