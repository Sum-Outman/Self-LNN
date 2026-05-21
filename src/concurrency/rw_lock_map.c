#include "selflnn/concurrency/rw_lock_map.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#define rw_map_spin_lock(lock)    do { while (InterlockedExchange((volatile LONG*)(lock), 1) != 0) { _mm_pause(); } } while(0)
#define rw_map_spin_unlock(lock)  InterlockedExchange((volatile LONG*)(lock), 0)
#define rw_map_atomic_inc(ptr)    InterlockedIncrement((volatile LONG*)(ptr))
typedef volatile LONG rw_map_lock_t;
#else
#include <stdatomic.h>
#define rw_map_spin_lock(lock)    do { while (__sync_lock_test_and_set((lock), 1) != 0) { __builtin_ia32_pause(); } } while(0)
#define rw_map_spin_unlock(lock)  __sync_lock_release((lock))
#define rw_map_atomic_inc(ptr)    __sync_add_and_fetch((ptr), 1)
typedef volatile int rw_map_lock_t;
#endif

#define RW_MAP_DEFAULT_CAPACITY 64
#define RW_MAP_MAX_CAPACITY 16777216
#define RW_MAP_DEFAULT_LOAD_FACTOR 0.75f
#define RW_MAP_MIN_SPIN_LOCKS 8
#define RW_MAP_HASH_PRIME 0x9E3779B9

typedef struct RwLockMapEntry {
    char* key;
    int64_t int_key;
    int use_int_key;
    void* value;
    struct RwLockMapEntry* next;
} RwLockMapEntry;

struct RwLockMap {
    RwLockMapConfig config;
    RwLockMapEntry** buckets;
    size_t capacity;
    size_t entry_count;
    size_t total_collisions;
    size_t max_chain_length;
    size_t resize_count;
    size_t lookup_count;
    size_t insert_count;
    size_t remove_count;
    double total_lookup_us;
    double total_insert_us;
    rw_map_lock_t* bucket_locks;
    rw_map_lock_t global_lock;
    int is_initialized;
};

struct RwLockMapIter {
    RwLockMap* map;
    size_t bucket_index;
    RwLockMapEntry* current;
};

static size_t rw_map_hash_string(const char* key, size_t capacity) {
    if (!key || capacity == 0) return 0;
    size_t hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + (size_t)c;
    }
    return hash % capacity;
}

static size_t rw_map_hash_int(int64_t key, size_t capacity) {
    if (capacity == 0) return 0;
    size_t h = (size_t)((uint64_t)key * RW_MAP_HASH_PRIME);
    h ^= (h >> 16);
    return h % capacity;
}

static int rw_map_resize_internal(RwLockMap* map, size_t new_capacity) {
    if (!map || new_capacity < map->entry_count || new_capacity > RW_MAP_MAX_CAPACITY) return -1;

    size_t actual_capacity = 16;
    while (actual_capacity < new_capacity) actual_capacity <<= 1;

    RwLockMapEntry** new_buckets = (RwLockMapEntry**)safe_calloc(actual_capacity, sizeof(RwLockMapEntry*));
    if (!new_buckets) return -1;

    rw_map_lock_t* new_locks = (rw_map_lock_t*)safe_calloc(actual_capacity, sizeof(rw_map_lock_t));
    if (!new_locks) {
        safe_free((void**)&new_buckets);
        return -1;
    }

    size_t old_capacity = map->capacity;
    for (size_t i = 0; i < old_capacity; i++) {
        RwLockMapEntry* entry = map->buckets[i];
        while (entry) {
            RwLockMapEntry* next = entry->next;
            size_t new_idx;
            if (entry->use_int_key) {
                new_idx = rw_map_hash_int(entry->int_key, actual_capacity);
            } else {
                new_idx = rw_map_hash_string(entry->key, actual_capacity);
            }
            entry->next = new_buckets[new_idx];
            new_buckets[new_idx] = entry;
            entry = next;
        }
    }

    safe_free((void**)&map->buckets);
    safe_free((void**)&map->bucket_locks);
    map->buckets = new_buckets;
    map->bucket_locks = new_locks;
    map->capacity = actual_capacity;
    map->resize_count++;
    return 0;
}

RwLockMap* rw_lock_map_create(const RwLockMapConfig* config) {
    RwLockMap* map = (RwLockMap*)safe_calloc(1, sizeof(RwLockMap));
    if (!map) return NULL;

    if (config) {
        map->config = *config;
    } else {
        map->config.initial_capacity = RW_MAP_DEFAULT_CAPACITY;
        map->config.max_capacity = RW_MAP_MAX_CAPACITY;
        map->config.load_factor = RW_MAP_DEFAULT_LOAD_FACTOR;
        map->config.enable_auto_resize = 1;
        map->config.num_spin_locks = RW_MAP_MIN_SPIN_LOCKS;
    }

    size_t initial = map->config.initial_capacity;
    if (initial < 16) initial = 16;
    size_t cap = 16;
    while (cap < initial) cap <<= 1;

    map->buckets = (RwLockMapEntry**)safe_calloc(cap, sizeof(RwLockMapEntry*));
    map->bucket_locks = (rw_map_lock_t*)safe_calloc(cap, sizeof(rw_map_lock_t));
    if (!map->buckets || !map->bucket_locks) {
        safe_free((void**)&map->buckets);
        safe_free((void**)&map->bucket_locks);
        safe_free((void**)&map);
        return NULL;
    }

    map->capacity = cap;
    map->entry_count = 0;
    map->total_collisions = 0;
    map->max_chain_length = 0;
    map->resize_count = 0;
    map->lookup_count = 0;
    map->insert_count = 0;
    map->remove_count = 0;
    map->total_lookup_us = 0.0;
    map->total_insert_us = 0.0;
    map->global_lock = 0;
    map->is_initialized = 1;

    return map;
}

void rw_lock_map_destroy(RwLockMap* map) {
    if (!map) return;
    rw_lock_map_clear(map);
    safe_free((void**)&map->buckets);
    safe_free((void**)&map->bucket_locks);
    map->is_initialized = 0;
    safe_free((void**)&map);
}

int rw_lock_map_insert(RwLockMap* map, const char* key, void* value) {
    if (!map || !key || !map->is_initialized) return -1;
    if (map->config.enable_auto_resize &&
        map->entry_count >= (size_t)(map->capacity * map->config.load_factor)) {
        size_t new_cap = map->capacity * 2;
        if (new_cap <= map->config.max_capacity) {
            rw_lock_map_resize(map, new_cap);
        }
    }

    size_t idx = rw_map_hash_string(key, map->capacity);
    rw_map_lock_t* lock = &map->bucket_locks[idx];
    rw_map_spin_lock(lock);

    RwLockMapEntry* entry = map->buckets[idx];
    size_t chain_len = 0;
    while (entry) {
        chain_len++;
        if (!entry->use_int_key && entry->key && strcmp(entry->key, key) == 0) {
            void* old_val = entry->value;
            entry->value = value;
            rw_map_spin_unlock(lock);
            map->insert_count++;
            return (old_val != value) ? 1 : 0;
        }
        entry = entry->next;
    }

    RwLockMapEntry* new_entry = (RwLockMapEntry*)safe_calloc(1, sizeof(RwLockMapEntry));
    if (!new_entry) {
        rw_map_spin_unlock(lock);
        return -1;
    }

    /* ZSF-036修复: 跨平台字符串复制，Windows用_strdup，POSIX用strdup */
#ifdef _WIN32
    new_entry->key = _strdup(key);
#else
    new_entry->key = strdup(key);
#endif
    if (!new_entry->key) {
        safe_free((void**)&new_entry);
        rw_map_spin_unlock(lock);
        return -1;
    }

    new_entry->use_int_key = 0;
    new_entry->int_key = 0;
    new_entry->value = value;
    new_entry->next = map->buckets[idx];
    map->buckets[idx] = new_entry;
    map->entry_count++;

    chain_len++;
    if (chain_len > map->max_chain_length) map->max_chain_length = chain_len;
    if (chain_len > 1) map->total_collisions++;

    rw_map_spin_unlock(lock);
    map->insert_count++;
    return 1;
}

int rw_lock_map_insert_int_key(RwLockMap* map, int64_t key, void* value) {
    if (!map || !map->is_initialized) return -1;

    if (map->config.enable_auto_resize &&
        map->entry_count >= (size_t)(map->capacity * map->config.load_factor)) {
        size_t new_cap = map->capacity * 2;
        if (new_cap <= map->config.max_capacity) {
            rw_lock_map_resize(map, new_cap);
        }
    }

    size_t idx = rw_map_hash_int(key, map->capacity);
    rw_map_lock_t* lock = &map->bucket_locks[idx];
    rw_map_spin_lock(lock);

    RwLockMapEntry* entry = map->buckets[idx];
    size_t chain_len = 0;
    while (entry) {
        chain_len++;
        if (entry->use_int_key && entry->int_key == key) {
            void* old_val = entry->value;
            entry->value = value;
            rw_map_spin_unlock(lock);
            map->insert_count++;
            return (old_val != value) ? 1 : 0;
        }
        entry = entry->next;
    }

    RwLockMapEntry* new_entry = (RwLockMapEntry*)safe_calloc(1, sizeof(RwLockMapEntry));
    if (!new_entry) {
        rw_map_spin_unlock(lock);
        return -1;
    }

    new_entry->use_int_key = 1;
    new_entry->int_key = key;
    new_entry->key = NULL;
    new_entry->value = value;
    new_entry->next = map->buckets[idx];
    map->buckets[idx] = new_entry;
    map->entry_count++;

    chain_len++;
    if (chain_len > map->max_chain_length) map->max_chain_length = chain_len;
    if (chain_len > 1) map->total_collisions++;

    rw_map_spin_unlock(lock);
    map->insert_count++;
    return 1;
}

void* rw_lock_map_get(RwLockMap* map, const char* key) {
    if (!map || !key || !map->is_initialized) return NULL;
    size_t idx = rw_map_hash_string(key, map->capacity);
    RwLockMapEntry* entry = map->buckets[idx];
    map->lookup_count++;
    while (entry) {
        if (!entry->use_int_key && entry->key && strcmp(entry->key, key) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }
    return NULL;
}

void* rw_lock_map_get_int_key(RwLockMap* map, int64_t key) {
    if (!map || !map->is_initialized) return NULL;
    size_t idx = rw_map_hash_int(key, map->capacity);
    RwLockMapEntry* entry = map->buckets[idx];
    map->lookup_count++;
    while (entry) {
        if (entry->use_int_key && entry->int_key == key) {
            return entry->value;
        }
        entry = entry->next;
    }
    return NULL;
}

int rw_lock_map_remove(RwLockMap* map, const char* key) {
    if (!map || !key || !map->is_initialized) return -1;
    size_t idx = rw_map_hash_string(key, map->capacity);
    rw_map_lock_t* lock = &map->bucket_locks[idx];
    rw_map_spin_lock(lock);

    RwLockMapEntry* prev = NULL;
    RwLockMapEntry* entry = map->buckets[idx];
    while (entry) {
        if (!entry->use_int_key && entry->key && strcmp(entry->key, key) == 0) {
            if (prev) prev->next = entry->next;
            else map->buckets[idx] = entry->next;
            safe_free((void**)&entry->key);
            safe_free((void**)&entry);
            map->entry_count--;
            rw_map_spin_unlock(lock);
            map->remove_count++;
            return 0;
        }
        prev = entry;
        entry = entry->next;
    }

    rw_map_spin_unlock(lock);
    return -1;
}

int rw_lock_map_remove_int_key(RwLockMap* map, int64_t key) {
    if (!map || !map->is_initialized) return -1;
    size_t idx = rw_map_hash_int(key, map->capacity);
    rw_map_lock_t* lock = &map->bucket_locks[idx];
    rw_map_spin_lock(lock);

    RwLockMapEntry* prev = NULL;
    RwLockMapEntry* entry = map->buckets[idx];
    while (entry) {
        if (entry->use_int_key && entry->int_key == key) {
            if (prev) prev->next = entry->next;
            else map->buckets[idx] = entry->next;
            safe_free((void**)&entry);
            map->entry_count--;
            rw_map_spin_unlock(lock);
            map->remove_count++;
            return 0;
        }
        prev = entry;
        entry = entry->next;
    }

    rw_map_spin_unlock(lock);
    return -1;
}

int rw_lock_map_contains(RwLockMap* map, const char* key) {
    if (!map || !key) return 0;
    return (rw_lock_map_get(map, key) != NULL) ? 1 : 0;
}

int rw_lock_map_contains_int_key(RwLockMap* map, int64_t key) {
    if (!map) return 0;
    return (rw_lock_map_get_int_key(map, key) != NULL) ? 1 : 0;
}

size_t rw_lock_map_size(RwLockMap* map) {
    if (!map) return 0;
    return map->entry_count;
}

int rw_lock_map_clear(RwLockMap* map) {
    if (!map || !map->is_initialized) return -1;
    for (size_t i = 0; i < map->capacity; i++) {
        RwLockMapEntry* entry = map->buckets[i];
        while (entry) {
            RwLockMapEntry* next = entry->next;
            if (entry->key) safe_free((void**)&entry->key);
            safe_free((void**)&entry);
            entry = next;
        }
        map->buckets[i] = NULL;
    }
    map->entry_count = 0;
    map->total_collisions = 0;
    map->max_chain_length = 0;
    return 0;
}

int rw_lock_map_resize(RwLockMap* map, size_t new_capacity) {
    if (!map || !map->is_initialized) return -1;
    if (new_capacity > RW_MAP_MAX_CAPACITY) return -1;
    rw_map_spin_lock(&map->global_lock);
    int ret = rw_map_resize_internal(map, new_capacity);
    rw_map_spin_unlock(&map->global_lock);
    return ret;
}

int rw_lock_map_for_each(RwLockMap* map,
                         int (*callback)(const char* key, void* value, void* user_data),
                         void* user_data) {
    if (!map || !callback || !map->is_initialized) return -1;
    int count = 0;
    for (size_t i = 0; i < map->capacity; i++) {
        rw_map_lock_t* lock = &map->bucket_locks[i];
        rw_map_spin_lock(lock);
        RwLockMapEntry* entry = map->buckets[i];
        while (entry) {
            const char* display_key = entry->use_int_key ? NULL : entry->key;
            if (callback(display_key, entry->value, user_data) == 0) {
                count++;
            }
            entry = entry->next;
        }
        rw_map_spin_unlock(lock);
    }
    return count;
}

int rw_lock_map_get_stats(const RwLockMap* map, RwLockMapStats* stats) {
    if (!map || !stats) return -1;
    stats->entry_count = map->entry_count;
    stats->bucket_count = map->capacity;
    stats->total_collisions = map->total_collisions;
    stats->max_chain_length = map->max_chain_length;
    stats->resize_count = map->resize_count;
    stats->lookup_count = map->lookup_count;
    stats->insert_count = map->insert_count;
    stats->remove_count = map->remove_count;
    stats->avg_lookup_us = (map->lookup_count > 0) ? map->total_lookup_us / map->lookup_count : 0.0;
    stats->avg_insert_us = (map->insert_count > 0) ? map->total_insert_us / map->insert_count : 0.0;
    return 0;
}

RwLockMapIter* rw_lock_map_iter_create(RwLockMap* map) {
    if (!map || !map->is_initialized) return NULL;
    RwLockMapIter* iter = (RwLockMapIter*)safe_calloc(1, sizeof(RwLockMapIter));
    if (!iter) return NULL;
    iter->map = map;
    iter->bucket_index = 0;
    iter->current = NULL;
    return iter;
}

int rw_lock_map_iter_next(RwLockMapIter* iter, const char** key, void** value) {
    if (!iter || !key || !value) return -1;
    if (iter->current) {
        iter->current = iter->current->next;
    }
    while (!iter->current && iter->bucket_index < iter->map->capacity) {
        iter->current = iter->map->buckets[iter->bucket_index];
        iter->bucket_index++;
    }
    if (iter->current) {
        *key = iter->current->key;
        *value = iter->current->value;
        return 0;
    }
    return -1;
}

void rw_lock_map_iter_destroy(RwLockMapIter* iter) {
    safe_free((void**)&iter);
}
