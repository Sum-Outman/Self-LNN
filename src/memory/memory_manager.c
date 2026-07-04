/**
 * @file memory_manager.c
 * @brief 记忆管理器实现
 *
 * 记忆系统总管理器实现，协调各种记忆类型。
 * 包含完整CPU端伙伴分配器（Buddy Allocator）用于减少内存碎片。
 */

#include "selflnn/memory/memory_manager.h"
#include "selflnn/memory/memory.h"
#include "selflnn/memory/semantic.h" /* P2-005: 语义记忆集成 */
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/logging.h"       /* M-017: 溢出块跟踪日志 */
#include "selflnn/core/laplace.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

/* ====================================================================
 * 伙伴分配器（Buddy Allocator）完整实现
 * 用于CPU端内存管理，减少碎片化。
 *
 * 原理：
 *   - 使用2的幂次方大小分级管理内存
 *   - 分配时向上取整到2的幂次方后切分（split）
 *   - 释放时将伙伴块合并（merge）
 *   - 伙伴地址 = base + ((addr - base) ^ block_size)
 * ==================================================================== */

/* 伙伴分配器常量 */
#define BUDDY_MIN_SHIFT         6                               /* 最小块：2^6 = 64字节 */
#define BUDDY_MIN_SIZE          ((size_t)1 << BUDDY_MIN_SHIFT)  /* 64 */
#define BUDDY_MAX_SHIFT         26                              /* 最大块：2^26 = 64MB */
#define BUDDY_MAX_SIZE          ((size_t)1 << BUDDY_MAX_SHIFT)  /* 64MB */
#define BUDDY_DEFAULT_POOL_SIZE (((size_t)64) * 1024 * 1024)    /* 默认64MB */
#define BUDDY_MAX_DESCRIPTORS   65536                           /* 最大块描述符数 */
#define BUDDY_MAGIC_FREE        0xBDD0F1EEu                     /* 空闲块魔数 */
#define BUDDY_MAGIC_USED        0xBDD0A5EDu                     /* 使用中块魔数 */
#define BUDDY_MAGIC_PAGE        0xBDD0A5E0u                     /* 页级空闲块魔数 */

#define BUDDY_MAX_LEVELS        (BUDDY_MAX_SHIFT - BUDDY_MIN_SHIFT + 1) /* 21级 */

/* 默认页面合并阈值 = 4KB */
#define BUDDY_DEFAULT_PAGE_THRESHOLD   ((size_t)4096)
/* 页面合并启用标志 */
#define BUDDY_PAGE_MERGE_ENABLED       1

/* 溢出回退最大块数 */
#define BUDDY_MAX_OVERFLOW_BLOCKS      1024

/**
 * @brief 溢出回退块记录
 * 当伙伴分配器无空闲块时，使用malloc分配并通过此结构跟踪
 */
typedef struct {
    void* address;
    size_t size;
} OverflowBlock;

/**
 * @brief 伙伴块描述符
 */
typedef struct BuddyBlock BuddyBlock;
struct BuddyBlock {
    void* address;
    size_t size;
    int level;
    int is_free;
    unsigned int magic;
    BuddyBlock* next;
    BuddyBlock* prev;
};

/**
 * @brief CPU端伙伴分配器
 */
typedef struct {
    BuddyBlock* free_lists[BUDDY_MAX_LEVELS];
    BuddyBlock* used_list;
    BuddyBlock* page_free_list;
    int levels;
    size_t min_size;
    size_t max_size;
    size_t pool_size;
    void* pool_base;

    BuddyBlock descriptor_pool[BUDDY_MAX_DESCRIPTORS];
    int next_descriptor;
    int descriptor_capacity;
    /* P1-10修复：描述符空闲链表，回收释放的描述符实现复用 */
    int descriptor_free_list[BUDDY_MAX_DESCRIPTORS];
    int descriptor_free_count;

    size_t current_used;
    size_t peak_used;
    size_t largest_free_block;
    int total_free_blocks;
    int total_used_blocks;
    int free_per_level[BUDDY_MAX_LEVELS];
    int defragment_count;

    size_t page_size;
    size_t page_threshold;
    int page_merge_enabled;
    size_t avg_allocation_size;
    size_t total_allocations;
    int adaptive_min_shift;
    int page_merge_count;

    /* 溢出回退跟踪（当伙伴分配器无空闲块时使用malloc） */
    OverflowBlock overflow_blocks[BUDDY_MAX_OVERFLOW_BLOCKS];
    int overflow_count;
    size_t overflow_used;

    MutexHandle lock;
    int is_initialized;
} CpuBuddyAllocator;

/* 获取2的幂次方 */
static size_t buddy_next_pow2(size_t x)
{
    if (x == 0) return BUDDY_MIN_SIZE;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
#if defined(__x86_64__) || defined(_M_X64) || defined(_WIN64)
    x |= x >> 32;
#endif
    return x + 1;
}

/* 计算大小对应的级别（向上取整） */
static int buddy_get_level(size_t size)
{
    size_t s = buddy_next_pow2(size);
    int level = 0;
    size_t lsize = BUDDY_MIN_SIZE;
    while (lsize < s && level < BUDDY_MAX_LEVELS - 1) {
        lsize <<= 1;
        level++;
    }
    return level;
}

/* 获取指定级别的块大小 */
static size_t buddy_level_size(int level)
{
    if (level < 0) return 0;
    if (level >= BUDDY_MAX_LEVELS) level = BUDDY_MAX_LEVELS - 1;
    return BUDDY_MIN_SIZE << level;
}

/* 从描述符池分配一个描述符（P1-10修复：优先从回收链复用） */
static BuddyBlock* buddy_alloc_descriptor(CpuBuddyAllocator* allocator)
{
    /* 优先从空闲链表回收 */
    if (allocator->descriptor_free_count > 0) {
        allocator->descriptor_free_count--;
        int idx = allocator->descriptor_free_list[allocator->descriptor_free_count];
        BuddyBlock* block = &allocator->descriptor_pool[idx];
        memset(block, 0, sizeof(BuddyBlock));
        return block;
    }
    int idx = allocator->next_descriptor;
    if (idx >= allocator->descriptor_capacity) {
        return NULL;
    }
    allocator->next_descriptor = idx + 1;
    BuddyBlock* block = &allocator->descriptor_pool[idx];
    memset(block, 0, sizeof(BuddyBlock));
    return block;
}

/* 释放描述符回池（P1-10修复：加入空闲链表供后续复用） */
static void buddy_free_descriptor(CpuBuddyAllocator* allocator, BuddyBlock* block)
{
    if (!block) return;
    if (block >= allocator->descriptor_pool &&
        block < allocator->descriptor_pool + allocator->descriptor_capacity) {
        /* 计算描述符索引并加入空闲链表 */
        int idx = (int)(block - allocator->descriptor_pool);
        if (allocator->descriptor_free_count < BUDDY_MAX_DESCRIPTORS) {
            allocator->descriptor_free_list[allocator->descriptor_free_count] = idx;
            allocator->descriptor_free_count++;
        }
        memset(block, 0, sizeof(BuddyBlock));
    }
}

/* 在链表中移除块 */
static void buddy_remove_from_list(BuddyBlock** list, BuddyBlock* block)
{
    if (!list || !block) return;
    if (block->prev) {
        block->prev->next = block->next;
    } else {
        *list = block->next;
    }
    if (block->next) {
        block->next->prev = block->prev;
    }
    block->next = NULL;
    block->prev = NULL;
}

/* 在链表头部插入块 */
static void buddy_insert_head(BuddyBlock** list, BuddyBlock* block)
{
    if (!list || !block) return;
    block->next = *list;
    block->prev = NULL;
    if (*list) {
        (*list)->prev = block;
    }
    *list = block;
}

/**
 * @brief 计算伙伴地址
 *  伙伴地址 = pool_base + ((addr - pool_base) ^ block_size)
 */
static void* buddy_find_buddy_address(void* pool_base, void* addr, size_t block_size)
{
    size_t offset = (size_t)((uint8_t*)addr - (uint8_t*)pool_base);
    size_t buddy_offset = offset ^ block_size;
    return (uint8_t*)pool_base + buddy_offset;
}

/* 在指定级别的空闲链表中按地址查找块 */
static BuddyBlock* buddy_find_in_free_list(BuddyBlock* list, void* address)
{
    while (list) {
        if (list->address == address) return list;
        list = list->next;
    }
    return NULL;
}

/**
 * @brief 初始化伙伴分配器
 */
static int buddy_init(CpuBuddyAllocator* allocator, size_t pool_size)
{
    if (!allocator) return -1;

    memset(allocator, 0, sizeof(CpuBuddyAllocator));

    allocator->min_size = BUDDY_MIN_SIZE;
    allocator->max_size = BUDDY_MAX_SIZE;
    allocator->levels = BUDDY_MAX_LEVELS;
    allocator->descriptor_capacity = BUDDY_MAX_DESCRIPTORS;
    allocator->next_descriptor = 0;
    allocator->largest_free_block = 0;
    allocator->total_free_blocks = 0;
    allocator->total_used_blocks = 0;
    allocator->defragment_count = 0;
    for (int i = 0; i < BUDDY_MAX_LEVELS; i++) {
        allocator->free_lists[i] = NULL;
        allocator->free_per_level[i] = 0;
    }
    allocator->used_list = NULL;
    allocator->page_free_list = NULL;
    allocator->current_used = 0;
    allocator->peak_used = 0;

    allocator->page_size = 4096;
    allocator->page_threshold = BUDDY_DEFAULT_PAGE_THRESHOLD;
    allocator->page_merge_enabled = BUDDY_PAGE_MERGE_ENABLED;
    allocator->avg_allocation_size = BUDDY_MIN_SIZE;
    allocator->total_allocations = 0;
    allocator->adaptive_min_shift = BUDDY_MIN_SHIFT;
    allocator->page_merge_count = 0;

    memset(allocator->overflow_blocks, 0, sizeof(allocator->overflow_blocks));
    allocator->overflow_count = 0;
    allocator->overflow_used = 0;

    size_t actual_pool_size = pool_size;
    if (actual_pool_size < BUDDY_MIN_SIZE * 2) {
        actual_pool_size = BUDDY_DEFAULT_POOL_SIZE;
    }
    actual_pool_size = buddy_next_pow2(actual_pool_size);
    if (actual_pool_size > BUDDY_MAX_SIZE) {
        actual_pool_size = BUDDY_MAX_SIZE;
    }
    allocator->pool_size = actual_pool_size;

    int max_level = buddy_get_level(actual_pool_size);
    if (max_level >= BUDDY_MAX_LEVELS) max_level = BUDDY_MAX_LEVELS - 1;
    allocator->max_size = buddy_level_size(max_level);

    allocator->pool_base = memory_map(actual_pool_size);
    if (!allocator->pool_base) {
        return -1;
    }
    memset(allocator->pool_base, 0, actual_pool_size);

    BuddyBlock* initial = buddy_alloc_descriptor(allocator);
    if (!initial) {
        memory_unmap(allocator->pool_base, actual_pool_size);
        allocator->pool_base = NULL;
        return -1;
    }

    initial->address = allocator->pool_base;
    initial->size = actual_pool_size;
    initial->level = max_level;
    initial->is_free = 1;
    initial->magic = BUDDY_MAGIC_FREE;

    buddy_insert_head(&allocator->free_lists[max_level], initial);
    allocator->total_free_blocks = 1;
    allocator->free_per_level[max_level] = 1;
    allocator->largest_free_block = actual_pool_size;

    allocator->lock = mutex_create();
    if (!allocator->lock) {
        memory_unmap(allocator->pool_base, actual_pool_size);
        allocator->pool_base = NULL;
        buddy_free_descriptor(allocator, initial);
        return -1;
    }

    allocator->is_initialized = 1;
    return 0;
}

/**
 * @brief 清理伙伴分配器
 */
static void buddy_cleanup(CpuBuddyAllocator* allocator)
{
    if (!allocator) return;

    /* 释放所有溢出回退块 */
    for (int i = 0; i < allocator->overflow_count; i++) {
        if (allocator->overflow_blocks[i].address) {
            free(allocator->overflow_blocks[i].address);
            allocator->overflow_blocks[i].address = NULL;
        }
    }
    allocator->overflow_count = 0;
    allocator->overflow_used = 0;

    if (allocator->lock) {
        mutex_destroy(allocator->lock);
        allocator->lock = NULL;
    }

    if (allocator->pool_base) {
        memory_unmap(allocator->pool_base, allocator->pool_size);
        allocator->pool_base = NULL;
    }

    for (int i = 0; i < BUDDY_MAX_LEVELS; i++) {
        allocator->free_lists[i] = NULL;
        allocator->free_per_level[i] = 0;
    }
    allocator->used_list = NULL;
    allocator->page_free_list = NULL;
    allocator->total_free_blocks = 0;
    allocator->total_used_blocks = 0;
    allocator->current_used = 0;
    allocator->peak_used = 0;
    allocator->largest_free_block = 0;
    allocator->next_descriptor = 0;
    allocator->page_merge_count = 0;
    allocator->is_initialized = 0;
}

/**
 * @brief 从伙伴分配器分配内存
 *
 * 算法：
 *   1. 向上取整到2的幂次方，确定所需级别 L
 *   2. 在 free_lists[L] 查找空闲块
 *   3. 若无，逐级向上查找 free_lists[L+1], free_lists[L+2], ...
 *   4. 找到后分裂到所需级别
 *   5. 标记为使用中返回
 */
static void* buddy_alloc(CpuBuddyAllocator* allocator, size_t size)
{
    if (!allocator || !allocator->is_initialized || size == 0) return NULL;

    /* 持锁前预计算所需级别和大小，避免持锁期间调用safe_malloc */
    int required_level = buddy_get_level(size);
    if (required_level < 0) required_level = 0;
    size_t required_level_size = buddy_level_size(required_level);

    mutex_lock(allocator->lock);

    if (size >= allocator->page_threshold && allocator->page_merge_enabled) {
        BuddyBlock* page_block = allocator->page_free_list;
        while (page_block) {
            if (page_block->size >= size && page_block->magic == BUDDY_MAGIC_PAGE) {
                buddy_remove_from_list(&allocator->page_free_list, page_block);
                allocator->total_free_blocks--;
                allocator->page_merge_count--;
                page_block->is_free = 0;
                page_block->magic = BUDDY_MAGIC_USED;
                buddy_insert_head(&allocator->used_list, page_block);
                allocator->total_used_blocks++;
                size_t alloc_sz = page_block->size;
                allocator->current_used += alloc_sz;
                if (allocator->current_used > allocator->peak_used) {
                    allocator->peak_used = allocator->current_used;
                }
                allocator->avg_allocation_size = (allocator->avg_allocation_size * allocator->total_allocations + size) / (allocator->total_allocations + 1);
                allocator->total_allocations++;
                void* result = page_block->address;
                mutex_unlock(allocator->lock);
                return result;
            }
            page_block = page_block->next;
        }
    }

    if (required_level_size > allocator->max_size) {
        /* 请求大小超过伙伴分配器最大池，释放锁后使用safe_malloc回退 */
        mutex_unlock(allocator->lock);
        void* overflow_ptr = safe_malloc(size);
        if (!overflow_ptr) {
            return NULL;
        }
        /* 重新获取锁以更新跟踪数据 */
        mutex_lock(allocator->lock);
        int idx = allocator->overflow_count;
        if (idx < BUDDY_MAX_OVERFLOW_BLOCKS) {
            allocator->overflow_blocks[idx].address = overflow_ptr;
            allocator->overflow_blocks[idx].size = size;
            allocator->overflow_count++;
            allocator->overflow_used += size;
        } else {
            log_warning("伙伴分配器溢出块数量已达到上限(%d)，无法跟踪此块(%zu字节)。"
                        "请增大BUDDY_POOL_SIZE或BUDDY_MAX_OVERFLOW_BLOCKS。",
                        BUDDY_MAX_OVERFLOW_BLOCKS, size);
        }
        allocator->total_allocations++;
        mutex_unlock(allocator->lock);
        return overflow_ptr;
    }

    int current_level = required_level;
    BuddyBlock* block = NULL;

    while (current_level < allocator->levels) {
        block = allocator->free_lists[current_level];
        if (block) break;
        current_level++;
    }

    if (!block || current_level >= allocator->levels) {
        /* 伙伴分配器无空闲块，释放锁后使用safe_malloc回退分配 */
        mutex_unlock(allocator->lock);
        void* overflow_ptr = safe_malloc(size);
        if (!overflow_ptr) {
            return NULL;
        }
        /* 重新获取锁以更新跟踪数据 */
        mutex_lock(allocator->lock);
        int idx = allocator->overflow_count;
        if (idx < BUDDY_MAX_OVERFLOW_BLOCKS) {
            allocator->overflow_blocks[idx].address = overflow_ptr;
            allocator->overflow_blocks[idx].size = size;
            allocator->overflow_count++;
            allocator->overflow_used += size;
        } else {
            log_warning("伙伴分配器溢出块数量已达到上限(%d)，无法跟踪此块(%zu字节)。"
                        "请增大BUDDY_POOL_SIZE或BUDDY_MAX_OVERFLOW_BLOCKS。",
                        BUDDY_MAX_OVERFLOW_BLOCKS, size);
        }
        allocator->total_allocations++;
        mutex_unlock(allocator->lock);
        return overflow_ptr;
    }

    buddy_remove_from_list(&allocator->free_lists[current_level], block);
    allocator->free_per_level[current_level]--;
    allocator->total_free_blocks--;

    if (current_level > required_level) {
        size_t block_size = buddy_level_size(current_level);
        while (current_level > required_level) {
            current_level--;
            block_size >>= 1;

            BuddyBlock* buddy = buddy_alloc_descriptor(allocator);
            if (!buddy) {
                /* 描述符耗尽，释放锁后使用safe_malloc回退 */
                mutex_unlock(allocator->lock);
                void* overflow_ptr = safe_malloc(size);
                if (!overflow_ptr) {
                    return NULL;
                }
                /* 重新获取锁以更新跟踪数据 */
                mutex_lock(allocator->lock);
                int oidx = allocator->overflow_count;
                if (oidx < BUDDY_MAX_OVERFLOW_BLOCKS) {
                    allocator->overflow_blocks[oidx].address = overflow_ptr;
                    allocator->overflow_blocks[oidx].size = size;
                    allocator->overflow_count++;
                    allocator->overflow_used += size;
                } else {
                    log_warning("伙伴分配器溢出块数量已达到上限(%d)，无法跟踪此块(%zu字节)。"
                                "请增大BUDDY_POOL_SIZE或BUDDY_MAX_OVERFLOW_BLOCKS。",
                                BUDDY_MAX_OVERFLOW_BLOCKS, size);
                }
                allocator->total_allocations++;
                mutex_unlock(allocator->lock);
                return overflow_ptr;
            }

            buddy->address = (uint8_t*)block->address + block_size;
            buddy->size = block_size;
            buddy->level = current_level;
            buddy->is_free = 1;
            buddy->magic = BUDDY_MAGIC_FREE;

            buddy_insert_head(&allocator->free_lists[current_level], buddy);
            allocator->free_per_level[current_level]++;
            allocator->total_free_blocks++;
        }
    }

    block->is_free = 0;
    block->magic = BUDDY_MAGIC_USED;

    buddy_insert_head(&allocator->used_list, block);
    allocator->total_used_blocks++;

    size_t allocated_size = required_level_size;
    allocator->current_used += allocated_size;
    if (allocator->current_used > allocator->peak_used) {
        allocator->peak_used = allocator->current_used;
    }

    allocator->avg_allocation_size = (allocator->avg_allocation_size * allocator->total_allocations + size) / (allocator->total_allocations + 1);
    allocator->total_allocations++;

    if (allocator->total_allocations > 100) {
        int new_min_shift = BUDDY_MIN_SHIFT;
        size_t avg = allocator->avg_allocation_size;
        if (avg > 8192) new_min_shift = 13;
        else if (avg > 2048) new_min_shift = 11;
        else if (avg > 512) new_min_shift = 9;
        if (new_min_shift != allocator->adaptive_min_shift) {
            allocator->adaptive_min_shift = new_min_shift;
        }
    }

    if (allocator->largest_free_block > 0 && allocated_size >= allocator->largest_free_block) {
        allocator->largest_free_block = 0;
        for (int i = allocator->levels - 1; i >= 0; i--) {
            if (allocator->free_lists[i]) {
                allocator->largest_free_block = buddy_level_size(i);
                break;
            }
        }
    }

    void* result = block->address;
    mutex_unlock(allocator->lock);
    return result;
}

/**
 * @brief 释放伙伴分配器内存
 *
 * 算法：
 *   1. 在 used_list 中找到对应块
 *   2. 标记为空闲
 *   3. 循环检查伙伴是否空闲，若是则合并到上一级
 *   4. 将最终块插入到对应级别的空闲链表
 */
static int buddy_free(CpuBuddyAllocator* allocator, void* ptr)
{
    if (!allocator || !allocator->is_initialized || !ptr) return -1;

    mutex_lock(allocator->lock);

    /* 检查是否属于溢出回退块 */
    for (int i = 0; i < allocator->overflow_count; i++) {
        if (allocator->overflow_blocks[i].address == ptr) {
            /* 从跟踪数组中移除，用最后一个元素填补 */
            free(ptr);
            if (allocator->overflow_used >= allocator->overflow_blocks[i].size) {
                allocator->overflow_used -= allocator->overflow_blocks[i].size;
            }
            allocator->overflow_blocks[i] = allocator->overflow_blocks[allocator->overflow_count - 1];
            allocator->overflow_count--;
            mutex_unlock(allocator->lock);
            return 0;
        }
    }

    /* M-017修复: 指针不在溢出跟踪列表中，但可能是不在跟踪列表的溢出块
     * （溢出计数已达上限时分配但仍通过free释放）。
     * 检查指针是否在伙伴池范围之外，若是则直接free。 */
    if (allocator->pool_base && allocator->pool_size > 0) {
        uint8_t* pool_start = (uint8_t*)allocator->pool_base;
        uint8_t* pool_end = pool_start + allocator->pool_size;
        if ((uint8_t*)ptr < pool_start || (uint8_t*)ptr >= pool_end) {
            free(ptr);
            mutex_unlock(allocator->lock);
            return 0;
        }
    }

    BuddyBlock* block = allocator->used_list;
    while (block) {
        if (block->address == ptr && !block->is_free && block->magic == BUDDY_MAGIC_USED) {
            break;
        }
        block = block->next;
    }

    if (!block) {
        mutex_unlock(allocator->lock);
        return -1;
    }

    buddy_remove_from_list(&allocator->used_list, block);
    allocator->total_used_blocks--;

    size_t free_size = buddy_level_size(block->level);
    if (free_size <= allocator->current_used) {
        allocator->current_used -= free_size;
    } else {
        allocator->current_used = 0;
    }

    block->is_free = 1;
    block->magic = BUDDY_MAGIC_FREE;

    int current_level = block->level;
    BuddyBlock* current_block = block;

    while (current_level < allocator->levels - 1) {
        size_t buddy_block_size = buddy_level_size(current_level);
        void* buddy_addr = buddy_find_buddy_address(
            allocator->pool_base, current_block->address, buddy_block_size);

        if ((uint8_t*)buddy_addr < (uint8_t*)allocator->pool_base ||
            (uint8_t*)buddy_addr >= (uint8_t*)allocator->pool_base + allocator->pool_size) {
            break;
        }

        BuddyBlock* buddy = buddy_find_in_free_list(
            allocator->free_lists[current_level], buddy_addr);

        if (!buddy || buddy->magic != BUDDY_MAGIC_FREE || !buddy->is_free) {
            break;
        }

        buddy_remove_from_list(&allocator->free_lists[current_level], buddy);
        allocator->free_per_level[current_level]--;
        allocator->total_free_blocks--;

        void* lower_addr = current_block->address;
        if (buddy_addr < current_block->address) {
            lower_addr = buddy_addr;
        }

        current_block->address = lower_addr;
        current_block->size = buddy_block_size * 2;
        current_level++;
        current_block->level = current_level;

        buddy_free_descriptor(allocator, buddy);
    }

    current_block->level = current_level;
    current_block->size = buddy_level_size(current_level);

    if (allocator->page_merge_enabled && current_block->size >= allocator->page_threshold) {
        current_block->magic = BUDDY_MAGIC_PAGE;
        buddy_insert_head(&allocator->page_free_list, current_block);
        allocator->page_merge_count++;
    } else {
        current_block->magic = BUDDY_MAGIC_FREE;
        buddy_insert_head(&allocator->free_lists[current_level], current_block);
        allocator->free_per_level[current_level]++;
    }
    allocator->total_free_blocks++;

    size_t new_largest = buddy_level_size(current_level);
    if (new_largest > allocator->largest_free_block) {
        allocator->largest_free_block = new_largest;
    }

    mutex_unlock(allocator->lock);
    return 0;
}

/**
 * @brief 碎片整理：扫描所有空闲块并尽可能合并
 */
static int buddy_defragment(CpuBuddyAllocator* allocator)
{
    if (!allocator || !allocator->is_initialized) return -1;

    mutex_lock(allocator->lock);

    int total_merged = 0;
    int merged = 1;

    while (merged) {
        merged = 0;
        for (int level = 0; level < allocator->levels - 1; level++) {
            size_t block_size = buddy_level_size(level);
            BuddyBlock* current = allocator->free_lists[level];

            while (current) {
                BuddyBlock* next_current = current->next;
                void* buddy_addr = buddy_find_buddy_address(
                    allocator->pool_base, current->address, block_size);

                if ((uint8_t*)buddy_addr >= (uint8_t*)allocator->pool_base &&
                    (uint8_t*)buddy_addr < (uint8_t*)allocator->pool_base + allocator->pool_size) {

                    BuddyBlock* buddy = buddy_find_in_free_list(
                        allocator->free_lists[level], buddy_addr);

                    if (buddy && buddy != current &&
                        buddy->magic == BUDDY_MAGIC_FREE && buddy->is_free) {

                        buddy_remove_from_list(&allocator->free_lists[level], buddy);
                        allocator->free_per_level[level]--;
                        allocator->total_free_blocks--;

                        void* lower_addr = current->address;
                        if (buddy_addr < current->address) {
                            lower_addr = buddy_addr;
                        }

                        current->address = lower_addr;
                        current->size = block_size * 2;
                        current->level = level + 1;

                        buddy_remove_from_list(&allocator->free_lists[level], current);
                        allocator->free_per_level[level]--;

                        buddy_insert_head(&allocator->free_lists[level + 1], current);
                        allocator->free_per_level[level + 1]++;

                        buddy_free_descriptor(allocator, buddy);

                        merged = 1;
                        total_merged++;
                        break;
                    }
                }
                current = next_current;
            }
            if (merged) break;
        }
    }

    if (allocator->page_merge_enabled && allocator->page_free_list) {
        BuddyBlock* current = allocator->page_free_list;
        while (current) {
            BuddyBlock* next = current->next;
            size_t block_size = current->size;
            int block_level = buddy_get_level(block_size);
            if (block_level >= 0 && block_level < allocator->levels) {
                buddy_remove_from_list(&allocator->page_free_list, current);
                allocator->page_merge_count--;
                current->magic = BUDDY_MAGIC_FREE;
                current->is_free = 1;
                current->level = block_level;
                current->size = buddy_level_size(block_level);
                buddy_insert_head(&allocator->free_lists[block_level], current);
                allocator->free_per_level[block_level]++;
            }
            current = next;
        }

        for (int level = 0; level < allocator->levels - 1; level++) {
            int scan_again = 1;
            while (scan_again) {
                scan_again = 0;
                BuddyBlock* cur = allocator->free_lists[level];
                while (cur) {
                    BuddyBlock* next_inner = cur->next;
                    void* buddy_addr = buddy_find_buddy_address(allocator->pool_base, cur->address, buddy_level_size(level));
                    if (buddy_addr) {
                        BuddyBlock* buddy = buddy_find_in_free_list(allocator->free_lists[level], buddy_addr);
                        if (buddy) {
                            void* lower_addr = (uint8_t*)cur->address < (uint8_t*)buddy->address ? cur->address : buddy->address;
                            buddy_remove_from_list(&allocator->free_lists[level], cur);
                            buddy_remove_from_list(&allocator->free_lists[level], buddy);
                            allocator->free_per_level[level] -= 2;
                            cur->address = lower_addr;
                            cur->size = buddy_level_size(level) * 2;
                            cur->level = level + 1;
                            buddy_insert_head(&allocator->free_lists[level + 1], cur);
                            allocator->free_per_level[level + 1]++;
                            buddy_free_descriptor(allocator, buddy);
                            total_merged++;
                            scan_again = 1;
                            break;
                        }
                    }
                    cur = next_inner;
                }
            }
        }
    }

    allocator->largest_free_block = 0;
    for (int i = allocator->levels - 1; i >= 0; i--) {
        if (allocator->free_lists[i]) {
            allocator->largest_free_block = buddy_level_size(i);
            break;
        }
    }

    allocator->defragment_count++;
    mutex_unlock(allocator->lock);
    return total_merged;
}

/**
 * @brief 获取伙伴分配器统计信息
 */
static void buddy_get_stats(CpuBuddyAllocator* allocator, BuddyAllocatorStats* stats)
{
    if (!allocator || !stats) return;

    mutex_lock(allocator->lock);

    stats->pool_size = allocator->pool_size;
    stats->used_size = allocator->current_used;
    stats->free_size = allocator->pool_size > allocator->current_used ?
                       allocator->pool_size - allocator->current_used : 0;
    stats->largest_free_block = allocator->largest_free_block;
    stats->total_blocks = allocator->total_free_blocks + allocator->total_used_blocks;
    stats->free_blocks = allocator->total_free_blocks;
    stats->levels = allocator->levels;
    for (int i = 0; i < BUDDY_MAX_LEVELS && i < 21; i++) {
        stats->free_per_level[i] = allocator->free_per_level[i];
    }
    stats->defragment_count = allocator->defragment_count;
    stats->page_merge_count = allocator->page_merge_count;
    stats->page_threshold = allocator->page_threshold;
    stats->avg_allocation_size = allocator->avg_allocation_size;
    stats->adaptive_min_shift = allocator->adaptive_min_shift;
    stats->overflow_blocks = allocator->overflow_count;
    stats->overflow_used = allocator->overflow_used;

    /* P2修复: largest_free_block为0时除零保护 */
    if (stats->free_size > 0 && stats->free_blocks > 1 && allocator->largest_free_block > 0) {
        float ideal_free = (float)stats->free_size / (float)allocator->largest_free_block;
        stats->fragmentation_ratio = 1.0f - (ideal_free / (float)stats->free_blocks);
        if (stats->fragmentation_ratio < 0.0f) stats->fragmentation_ratio = 0.0f;
        if (stats->fragmentation_ratio > 1.0f) stats->fragmentation_ratio = 1.0f;
    } else {
        stats->fragmentation_ratio = 0.0f;
    }

    mutex_unlock(allocator->lock);
}

/* ====================================================================
 * 记忆管理器实现
 * ==================================================================== */

/**
 * @brief 记忆管理器内部结构体
 */
struct MemoryManager {
    MemorySystem* memory_system;
    MemoryManagerConfig config;
    int is_initialized;

    /* P2-005修复: 语义记忆实例 — 独立管理概念/关系/泛化/特化等语义知识 */
    SemanticMemory* semantic_memory;

    CpuBuddyAllocator buddy_allocator;
    int buddy_enabled;
};

/**
 * @brief 根据优先级选择记忆类型
 */
static MemoryType select_memory_type(int priority)
{
    if (priority >= 3) {
        return MEMORY_TYPE_SHORT_TERM;
    } else if (priority >= 1) {
        return MEMORY_TYPE_LONG_TERM;
    } else if (priority == 0) {
        return MEMORY_TYPE_EPISODIC;
    } else {
        return MEMORY_TYPE_SEMANTIC;
    }
}

/**
 * @brief 创建记忆管理器实例
 */
MemoryManager* memory_manager_create(const MemoryManagerConfig* config)
{
    if (!config) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "创建记忆管理器：配置为空");
        return NULL;
    }

    MemoryManager* manager = (MemoryManager*)safe_malloc(sizeof(MemoryManager));
    if (!manager) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建记忆管理器：内存分配失败");
        return NULL;
    }

    memset(manager, 0, sizeof(MemoryManager));
    memcpy(&manager->config, config, sizeof(MemoryManagerConfig));

    MemoryConfig mem_config;
    memset(&mem_config, 0, sizeof(MemoryConfig));
    mem_config.max_short_term = config->short_term_capacity;
    mem_config.max_long_term = config->long_term_capacity;
    mem_config.decay_rate = 0.1f;
    mem_config.consolidation_rate = config->consolidation_rate;
    mem_config.enable_consolidation = 1;

    manager->memory_system = memory_create(&mem_config);
    if (!manager->memory_system) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "创建记忆管理器：底层记忆系统创建失败");
        safe_free((void**)&manager);
        return NULL;
    }

    manager->buddy_enabled = config->enable_buddy_allocator;
    if (manager->buddy_enabled) {
        size_t pool_sz = config->buddy_pool_size > 0 ?
                         config->buddy_pool_size : BUDDY_DEFAULT_POOL_SIZE;
        if (buddy_init(&manager->buddy_allocator, pool_sz) != 0) {
            manager->buddy_enabled = 0;
        }
    }

    /* P2-005修复: 创建语义记忆实例 */
    {
        SemanticMemoryConfig sm_cfg;
        memset(&sm_cfg, 0, sizeof(sm_cfg));
        sm_cfg.capacity = config->semantic_capacity > 0 ?
                          config->semantic_capacity : 10000;
        sm_cfg.association_strength = 0.5f;
        sm_cfg.generalization_level = 0.3f;
        sm_cfg.enable_hierarchy = 1;
        manager->semantic_memory = semantic_memory_create(&sm_cfg);
        if (!manager->semantic_memory) {
            log_warning("[记忆管理器] 语义记忆创建失败，将跳过语义存储功能");
        }
    }

    manager->is_initialized = 1;
    return manager;
}

/**
 * @brief 释放记忆管理器实例
 */
void memory_manager_free(MemoryManager* manager)
{
    if (!manager) return;

    if (manager->buddy_enabled) {
        buddy_cleanup(&manager->buddy_allocator);
    }

    if (manager->memory_system) {
        memory_free(manager->memory_system);
    }

    /* P2-005修复: 释放语义记忆实例 */
    if (manager->semantic_memory) {
        semantic_memory_free(manager->semantic_memory);
        manager->semantic_memory = NULL;
    }

    safe_free((void**)&manager);
}

/**
 * @brief 存储记忆（自动选择类型）
 */
int memory_manager_store(MemoryManager* manager, const char* key,
                        const float* data, size_t data_size,
                        int priority, float strength)
{
    SELFLNN_CHECK_NULL(manager, "存储记忆：管理器为空");
    SELFLNN_CHECK_NULL(key, "存储记忆：键为空");
    SELFLNN_CHECK_NULL(data, "存储记忆：数据为空");
    SELFLNN_CHECK(data_size > 0, SELFLNN_ERROR_INVALID_ARGUMENT, "存储记忆：数据大小为零");
    SELFLNN_CHECK(manager->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED, "存储记忆：管理器未初始化");

    MemoryType type = select_memory_type(priority);
    int result = memory_store(manager->memory_system, key, data, data_size, type, strength);
    if (result != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "存储记忆 '%s' 失败", key);
        return result;
    }

    /* 快速稳定性检查：对存储的记忆数据进行频域稳定性加权 */
    if (manager->config.enable_laplace_stability_check && data && data_size > 0) {
        size_t spec_size = data_size < 512 ? data_size : 512;
        float den_coeffs[2] = {1.0f, -0.5f};
        int is_stable = 0;
        float stability_margin = 0.0f;
        if (laplace_check_stability_fast(den_coeffs, 2, &is_stable, &stability_margin) == 0) {
            float stability_factor = is_stable ? 1.0f : 0.7f;
            if (stability_margin > 0.0f && stability_margin < 1.0f) {
                stability_factor = 0.7f + 0.3f * stability_margin;
            }
            float* enhanced_data = (float*)safe_malloc(data_size * sizeof(float));
            if (enhanced_data) {
                memcpy(enhanced_data, data, data_size * sizeof(float));
                for (size_t i = 0; i < spec_size; i++) {
                    enhanced_data[i] *= (0.9f + 0.1f * stability_factor);
                }
                memory_store(manager->memory_system, key, enhanced_data,
                            data_size, type, strength * stability_factor);
                safe_free((void**)&enhanced_data);
            }
        }
    }

    return result;
}

/* R6-⑤修复: 存储记忆（扩展版，支持模态来源隔离）
 * 调用memory_store_ex设置modality_flags和source_id，
 * 使记忆系统可以按模态来源独立检索和淘汰。 */
int memory_manager_store_modal(MemoryManager* manager, const char* key,
                        const float* data, size_t data_size,
                        int priority, float strength,
                        uint32_t modality_flags, const char* source_id)
{
    SELFLNN_CHECK_NULL(manager, "存储记忆：管理器为空");
    SELFLNN_CHECK_NULL(key, "存储记忆：键为空");
    SELFLNN_CHECK_NULL(data, "存储记忆：数据为空");
    SELFLNN_CHECK(data_size > 0, SELFLNN_ERROR_INVALID_ARGUMENT, "存储记忆：数据大小为零");
    SELFLNN_CHECK(manager->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED, "存储记忆：管理器未初始化");

    MemoryType type = select_memory_type(priority);
    int result = memory_store_ex(manager->memory_system, key, data, data_size,
                                  type, strength, modality_flags, source_id);
    if (result != 0) return result;

    if (manager->config.enable_laplace_stability_check && data && data_size > 0) {
        size_t spec_size = data_size < 512 ? data_size : 512;
        float den_coeffs[2] = {1.0f, -0.5f};
        int is_stable = 0;
        float stability_margin = 0.0f;
        if (laplace_check_stability_fast(den_coeffs, 2, &is_stable, &stability_margin) == 0) {
            float stability_factor = is_stable ? 1.0f : 0.7f;
            if (stability_margin > 0.0f && stability_margin < 1.0f) {
                stability_factor = 0.7f + 0.3f * stability_margin;
            }
            float* enhanced_data = (float*)safe_malloc(data_size * sizeof(float));
            if (enhanced_data) {
                memcpy(enhanced_data, data, data_size * sizeof(float));
                for (size_t i = 0; i < spec_size; i++)
                    enhanced_data[i] *= (0.9f + 0.1f * stability_factor);
                memory_store_ex(manager->memory_system, key, enhanced_data,
                               data_size, type, strength * stability_factor,
                               modality_flags, source_id);
                safe_free((void**)&enhanced_data);
            }
        }
    }

    return result;
}

/**
 * @brief 检索记忆（自动搜索所有类型）
 */
int memory_manager_retrieve(MemoryManager* manager, const char* key,
                           float* data, size_t data_size,
                           float* strength, int* memory_type)
{
    SELFLNN_CHECK_NULL(manager, "检索记忆：管理器为空");
    SELFLNN_CHECK_NULL(key, "检索记忆：键为空");
    SELFLNN_CHECK_NULL(data, "检索记忆：数据缓冲区为空");
    SELFLNN_CHECK(manager->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED, "检索记忆：管理器未初始化");

    MemoryType found_type = MEMORY_TYPE_SHORT_TERM;
    int result = memory_retrieve(manager->memory_system, key, data, data_size, strength, &found_type);

    if (result == 0 && memory_type) {
        *memory_type = (int)found_type;
    } else if (result != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "检索记忆 '%s' 未找到", key);
    }
    return result;
}

/**
 * @brief 更新记忆
 */
int memory_manager_update(MemoryManager* manager, const char* key,
                         const float* data, size_t data_size, float strength_delta)
{
    SELFLNN_CHECK_NULL(manager, "更新记忆：管理器为空");
    SELFLNN_CHECK_NULL(key, "更新记忆：键为空");
    SELFLNN_CHECK_NULL(data, "更新记忆：数据为空");
    SELFLNN_CHECK(manager->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED, "更新记忆：管理器未初始化");

    int result = memory_update(manager->memory_system, key, data, data_size, strength_delta);
    if (result != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "更新记忆 '%s' 失败", key);
    }
    return result;
}

int memory_manager_forget(MemoryManager* manager, const char* key)
{
    SELFLNN_CHECK_NULL(manager, "遗忘记忆：管理器为空");
    SELFLNN_CHECK_NULL(key, "遗忘记忆：键为空");
    SELFLNN_CHECK(manager->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED, "遗忘记忆：管理器未初始化");

    int result = memory_forget(manager->memory_system, key);
    if (result != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "遗忘记忆 '%s' 失败", key);
    }
    return result;
}

int memory_manager_consolidate(MemoryManager* manager, const char* key)
{
    SELFLNN_CHECK_NULL(manager, "巩固记忆：管理器为空");
    SELFLNN_CHECK_NULL(key, "巩固记忆：键为空");
    SELFLNN_CHECK(manager->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED, "巩固记忆：管理器未初始化");

    int result = memory_consolidate(manager->memory_system, key);
    if (result != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "巩固记忆 '%s' 失败", key);
        return result;
    }

    /* 快速稳定性检查：巩固后评估记忆稳定性 */
    if (manager->config.enable_laplace_stability_check) {
        float den_coeffs[2] = {1.0f, -0.5f};
        int is_stable = 0;
        if (laplace_check_stability_fast(den_coeffs, 2, &is_stable, NULL) == 0) {
            if (!is_stable) {
                memory_consolidate(manager->memory_system, key);
            }
        }
    }

    return result;
}

int memory_manager_integrate(MemoryManager* manager, const char* key1,
                            const char* key2, float association_strength)
{
    SELFLNN_CHECK_NULL(manager, "整合记忆：管理器为空");
    SELFLNN_CHECK_NULL(key1, "整合记忆：键1为空");
    SELFLNN_CHECK_NULL(key2, "整合记忆：键2为空");
    SELFLNN_CHECK(manager->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED, "整合记忆：管理器未初始化");

    if (!manager->config.enable_integration) {
        return 0;
    }

    char association_key[256];
    snprintf(association_key, sizeof(association_key), "assoc_%s_%s", key1, key2);

    float association_data[1] = { association_strength };

    int result = memory_store(manager->memory_system, association_key,
                             association_data, 1, MEMORY_TYPE_SEMANTIC, association_strength);
    if (result != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MEMORY_ALLOCATION, __func__, __FILE__, __LINE__,
                              "整合记忆 '%s' 和 '%s' 失败", key1, key2);
    }
    return result;
}

/**
 * @brief 获取记忆管理器配置
 */
int memory_manager_get_config(const MemoryManager* manager, MemoryManagerConfig* config)
{
    SELFLNN_CHECK_NULL(manager, "获取配置：管理器为空");
    SELFLNN_CHECK_NULL(config, "获取配置：配置输出为空");

    memcpy(config, &manager->config, sizeof(MemoryManagerConfig));
    return SELFLNN_SUCCESS;
}

int memory_manager_set_config(MemoryManager* manager, const MemoryManagerConfig* config)
{
    SELFLNN_CHECK_NULL(manager, "设置配置：管理器为空");
    SELFLNN_CHECK_NULL(config, "设置配置：配置为空");
    SELFLNN_CHECK(manager->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED, "设置配置：管理器未初始化");

    manager->config.short_term_capacity = config->short_term_capacity;
    manager->config.long_term_capacity = config->long_term_capacity;
    manager->config.episodic_capacity = config->episodic_capacity;
    manager->config.semantic_capacity = config->semantic_capacity;
    manager->config.consolidation_rate = config->consolidation_rate;
    manager->config.enable_integration = config->enable_integration;
    manager->config.buddy_pool_size = config->buddy_pool_size;
    manager->config.enable_buddy_allocator = config->enable_buddy_allocator;
    manager->config.enable_laplace_stability_check = config->enable_laplace_stability_check;

    if (config->enable_buddy_allocator && !manager->buddy_enabled) {
        size_t pool_sz = config->buddy_pool_size > 0 ?
                         config->buddy_pool_size : BUDDY_DEFAULT_POOL_SIZE;
        if (buddy_init(&manager->buddy_allocator, pool_sz) == 0) {
            manager->buddy_enabled = 1;
        }
    } else if (!config->enable_buddy_allocator && manager->buddy_enabled) {
        buddy_cleanup(&manager->buddy_allocator);
        manager->buddy_enabled = 0;
    }

    MemoryConfig mem_config;
    if (memory_get_config(manager->memory_system, &mem_config) == 0) {
        mem_config.max_short_term = config->short_term_capacity;
        mem_config.max_long_term = config->long_term_capacity;
        mem_config.consolidation_rate = config->consolidation_rate;
        memory_set_config(manager->memory_system, &mem_config);
    }

    return SELFLNN_SUCCESS;
}

int memory_manager_get_stats(const MemoryManager* manager, size_t* total_memories,
                            float* consolidation_ratio, float* integration_level)
{
    SELFLNN_CHECK_NULL(manager, "获取统计：管理器为空");
    SELFLNN_CHECK(manager->is_initialized, SELFLNN_ERROR_NOT_INITIALIZED, "获取统计：管理器未初始化");

    size_t total_items;
    float avg_strength;
    float consolidation_val;

    if (memory_get_stats(manager->memory_system, &total_items, &avg_strength, &consolidation_val) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__,
                              "获取统计：底层记忆系统查询失败");
        return -1;
    }

    if (total_memories) *total_memories = total_items;
    if (consolidation_ratio) *consolidation_ratio = consolidation_val;

    if (integration_level) {
        float base_integration = 0.3f;
        if (total_items > 0) {
            float memory_factor = 1.0f - expf(-(float)total_items / 100.0f);
            float consolidation_factor = consolidation_val;
            *integration_level = base_integration + memory_factor * 0.4f + consolidation_factor * 0.3f;
            if (*integration_level < 0.0f) *integration_level = 0.0f;
            if (*integration_level > 1.0f) *integration_level = 1.0f;
        } else {
            *integration_level = base_integration;
        }
    }

    return SELFLNN_SUCCESS;
}

/**
 * @brief 重置记忆管理器
 */
void memory_manager_reset(MemoryManager* manager)
{
    if (!manager || !manager->is_initialized) return;

    memory_reset(manager->memory_system);

    if (manager->buddy_enabled) {
        buddy_cleanup(&manager->buddy_allocator);
        size_t pool_sz = manager->config.buddy_pool_size > 0 ?
                         manager->config.buddy_pool_size : BUDDY_DEFAULT_POOL_SIZE;
        buddy_init(&manager->buddy_allocator, pool_sz);
    }
}

MemorySystem* memory_manager_get_system(MemoryManager* manager)
{
    if (!manager || !manager->is_initialized) return NULL;
    return manager->memory_system;
}

/* P2-005修复: 语义记忆访问器
 * 供知识库学习、多模态教学等模块获取语义记忆进行概念存储/检索/泛化 */
void* memory_manager_get_semantic(MemoryManager* manager)
{
    if (!manager || !manager->is_initialized) return NULL;
    return manager->semantic_memory;
}

/**
 * @brief 从伙伴分配器分配内存
 */
void* memory_manager_pool_alloc(MemoryManager* manager, size_t size)
{
    if (!manager || !manager->is_initialized || !manager->buddy_enabled) return NULL;
    return buddy_alloc(&manager->buddy_allocator, size);
}

/**
 * @brief 释放伙伴分配器内存
 */
int memory_manager_pool_free(MemoryManager* manager, void* ptr)
{
    if (!manager || !manager->is_initialized || !manager->buddy_enabled || !ptr) return -1;
    return buddy_free(&manager->buddy_allocator, ptr);
}

/**
 * @brief 执行伙伴分配器碎片整理
 */
int memory_manager_pool_defragment(MemoryManager* manager)
{
    if (!manager || !manager->is_initialized || !manager->buddy_enabled) return -1;
    return buddy_defragment(&manager->buddy_allocator);
}

/**
 * @brief 获取伙伴分配器统计信息
 */
int memory_manager_get_buddy_stats(const MemoryManager* manager, BuddyAllocatorStats* stats)
{
    if (!manager || !stats) return -1;
    if (!manager->buddy_enabled) {
        memset(stats, 0, sizeof(BuddyAllocatorStats));
        return 0;
    }
    buddy_get_stats((CpuBuddyAllocator*)&manager->buddy_allocator, stats);
    return 0;
}
