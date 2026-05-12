#ifndef SELFLNN_RW_LOCK_MAP_H
#define SELFLNN_RW_LOCK_MAP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RwLockMap RwLockMap;
typedef struct RwLockMapIter RwLockMapIter;

typedef struct {
    size_t initial_capacity;
    size_t max_capacity;
    float load_factor;
    int enable_auto_resize;
    int num_spin_locks;
} RwLockMapConfig;

typedef struct {
    size_t entry_count;
    size_t bucket_count;
    size_t total_collisions;
    size_t max_chain_length;
    size_t resize_count;
    size_t lookup_count;
    size_t insert_count;
    size_t remove_count;
    double avg_lookup_us;
    double avg_insert_us;
} RwLockMapStats;

RwLockMap* rw_lock_map_create(const RwLockMapConfig* config);
void rw_lock_map_destroy(RwLockMap* map);

int rw_lock_map_insert(RwLockMap* map, const char* key, void* value);
int rw_lock_map_insert_int_key(RwLockMap* map, int64_t key, void* value);

void* rw_lock_map_get(RwLockMap* map, const char* key);
void* rw_lock_map_get_int_key(RwLockMap* map, int64_t key);

int rw_lock_map_remove(RwLockMap* map, const char* key);
int rw_lock_map_remove_int_key(RwLockMap* map, int64_t key);

int rw_lock_map_contains(RwLockMap* map, const char* key);
int rw_lock_map_contains_int_key(RwLockMap* map, int64_t key);

size_t rw_lock_map_size(RwLockMap* map);
int rw_lock_map_clear(RwLockMap* map);
int rw_lock_map_resize(RwLockMap* map, size_t new_capacity);

int rw_lock_map_for_each(RwLockMap* map,
                         int (*callback)(const char* key, void* value, void* user_data),
                         void* user_data);

int rw_lock_map_get_stats(const RwLockMap* map, RwLockMapStats* stats);

RwLockMapIter* rw_lock_map_iter_create(RwLockMap* map);
int rw_lock_map_iter_next(RwLockMapIter* iter, const char** key, void** value);
void rw_lock_map_iter_destroy(RwLockMapIter* iter);

#ifdef __cplusplus
}
#endif

#endif
