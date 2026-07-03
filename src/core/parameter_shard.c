#define SELFLNN_IMPLEMENTATION

#include "selflnn/core/parameter_shard.h"
#include "selflnn/core/common.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <malloc.h>
#define thread_mutex_t CRITICAL_SECTION
#define thread_mutex_init(m) InitializeCriticalSection(m)
#define thread_mutex_lock(m) EnterCriticalSection(m)
#define thread_mutex_unlock(m) LeaveCriticalSection(m)
#define thread_mutex_destroy(m) DeleteCriticalSection(m)
#define shard_aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define shard_aligned_free(ptr) _aligned_free(ptr)
#elif defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#include <stdlib.h>
#define thread_mutex_t pthread_mutex_t
#define thread_mutex_init(m) pthread_mutex_init(m, NULL)
#define thread_mutex_lock(m) pthread_mutex_lock(m)
#define thread_mutex_unlock(m) pthread_mutex_unlock(m)
#define thread_mutex_destroy(m) pthread_mutex_destroy(m)
#define shard_aligned_alloc(alignment, size) aligned_alloc(alignment, size)
#define shard_aligned_free(ptr) free(ptr)
#endif

#define SHARD_SYSTEM_VERSION 2
#define SHARD_CHECKPOINT_MAGIC 0x53485244
#define MIN_SHARD_SIZE (1024 * 1024)
#define MAX_SHARD_SIZE (256LL * 1024LL * 1024LL * 1024LL)
#define SYNC_TIMEOUT_MS 30000
#define TRANSFER_BLOCK_SIZE (1024 * 1024)

typedef struct {
    uint32_t magic;
    uint32_t version;
    size_t num_shards;
    size_t total_params;
    ShardDescriptor shards[SELFLNN_MAX_SHARDS];
} ShardCheckpointHeader;

static int validate_shard_id(const ParameterShardSystem* system, uint32_t shard_id)
{
    if (!system || shard_id >= system->num_shards) return 0;
    return 1;
}

static int validate_shard_system(const ParameterShardSystem* system)
{
    if (!system) return 0;
    if (!system->is_initialized) return 0;
    if (system->num_shards == 0) return 0;
    return 1;
}

static size_t calculate_shard_size(size_t total_params, size_t num_shards, size_t shard_idx)
{
    size_t base_size = total_params / num_shards;
    size_t remainder = total_params % num_shards;
    size_t shard_size = base_size + (shard_idx < remainder ? 1 : 0);
    size_t aligned = ((shard_size + SELFLNN_SHARD_ALIGNMENT - 1) / SELFLNN_SHARD_ALIGNMENT) * SELFLNN_SHARD_ALIGNMENT;
    if (aligned < MIN_SHARD_SIZE / sizeof(float)) aligned = MIN_SHARD_SIZE / sizeof(float);
    return aligned;
}

static size_t precision_to_bytes(ShardPrecision precision)
{
    switch (precision) {
        case SHARD_PRECISION_FP32: return 4;
        case SHARD_PRECISION_FP16: return 2;
        case SHARD_PRECISION_BF16: return 2;
        case SHARD_PRECISION_INT8: return 1;
        case SHARD_PRECISION_INT4: return 1;
        default: return 4;
    }
}

ParameterShardSystem* shard_system_create(const ShardSystemConfig* config)
{
    if (!config || config->num_shards == 0 || config->num_shards > SELFLNN_MAX_SHARDS) return NULL;
    if (config->total_params == 0) return NULL;

    ParameterShardSystem* system = (ParameterShardSystem*)calloc(1, sizeof(ParameterShardSystem));
    if (!system) return NULL;

    system->num_shards = config->num_shards;
    system->total_param_count = config->total_params;
    system->shard_capacity = config->num_shards;
    system->enable_async_transfer = config->enable_async;
    system->enable_gradient_compression = config->enable_gradient_compression;
    system->compression_ratio = config->compression_ratio > 0 ? config->compression_ratio : 0.5f;
    system->enable_overlap_computation = config->enable_overlap;
    system->gradient_sync_interval = config->gradient_sync_interval > 0 ? config->gradient_sync_interval : 1;

    system->shards = (ShardDescriptor*)calloc(config->num_shards, sizeof(ShardDescriptor));
    if (!system->shards) { free(system); return NULL; }

    system->metrics = (ShardPerformanceMetrics*)calloc(config->num_shards, sizeof(ShardPerformanceMetrics));
    if (!system->metrics) { free(system->shards); free(system); return NULL; }

    system->shard_param_ptrs = (float**)calloc(config->num_shards, sizeof(float*));
    if (!system->shard_param_ptrs) { free(system->metrics); free(system->shards); free(system); return NULL; }

    system->shard_grad_ptrs = (float**)calloc(config->num_shards, sizeof(float*));
    if (!system->shard_grad_ptrs) { free(system->shard_param_ptrs); free(system->metrics); free(system->shards); free(system); return NULL; }

    size_t unified_buffer_size = config->total_params + SELFLNN_SHARD_ALIGNMENT;
    system->unified_param_buffer = (float*)shard_aligned_alloc(SELFLNN_SHARD_ALIGNMENT, unified_buffer_size * sizeof(float));
    if (!system->unified_param_buffer) {
        free(system->shard_grad_ptrs); free(system->shard_param_ptrs);
        free(system->metrics); free(system->shards); free(system);
        return NULL;
    }
    memset(system->unified_param_buffer, 0, unified_buffer_size * sizeof(float));

    size_t current_offset = 0;
    for (size_t i = 0; i < config->num_shards; i++) {
        ShardDescriptor* shard = &system->shards[i];
        shard->magic = SELFLNN_SHARD_MAGIC;
        shard->shard_id = (uint32_t)i;
        snprintf(shard->name, SELFLNN_MAX_SHARD_NAME, "shard_%zu", i);

        shard->location = (config->locations && i < config->num_shards) ?
                          config->locations[i] : SHARD_LOCATION_HOST;
        shard->precision = config->default_precision;
        shard->status = SHARD_STATUS_UNINITIALIZED;
        shard->node_id = 0;
        shard->numa_node = 0;

        size_t shard_params = calculate_shard_size(config->total_params, config->num_shards, i);
        if (config->shard_sizes && i < config->num_shards && config->shard_sizes[i] > 0) {
            shard_params = config->shard_sizes[i];
        }

        shard->num_params = shard_params;
        shard->param_offset = current_offset;
        shard->param_size_bytes = shard_params * precision_to_bytes(shard->precision);
        shard->gradient_offset = current_offset;
        shard->gradient_size_bytes = shard_params * sizeof(float);

        size_t shard_mem_size = shard_params + SELFLNN_SHARD_ALIGNMENT;
        system->shard_param_ptrs[i] = (float*)shard_aligned_alloc(SELFLNN_SHARD_ALIGNMENT,
                                                            shard_mem_size * sizeof(float));
        system->shard_grad_ptrs[i] = (float*)shard_aligned_alloc(SELFLNN_SHARD_ALIGNMENT,
                                                           shard_mem_size * sizeof(float));

        if (!system->shard_param_ptrs[i] || !system->shard_grad_ptrs[i]) {
            for (size_t j = 0; j < i; j++) {
                shard_aligned_free(system->shard_param_ptrs[j]);
                shard_aligned_free(system->shard_grad_ptrs[j]);
            }
            free(system->shard_grad_ptrs); free(system->shard_param_ptrs);
            shard_aligned_free(system->unified_param_buffer);
            free(system->metrics); free(system->shards); free(system);
            return NULL;
        }
        memset(system->shard_param_ptrs[i], 0, shard_mem_size * sizeof(float));
        memset(system->shard_grad_ptrs[i], 0, shard_mem_size * sizeof(float));

        current_offset += shard_params;
    }

    system->num_devices = 0;
    if (config->device_ids) {
        for (size_t i = 0; i < config->num_shards; i++) {
            if (config->device_ids[i] > 0) system->num_devices++;
        }
    }

    thread_mutex_init((thread_mutex_t*)&system->sync_mutex);

    return system;
}

void shard_system_free(ParameterShardSystem* system)
{
    if (!system) return;

    if (system->is_initialized) {
        shard_system_destroy(system);
    }

    if (system->unified_param_buffer) {
        shard_aligned_free(system->unified_param_buffer);
        system->unified_param_buffer = NULL;
    }

    if (system->shard_param_ptrs) {
        for (size_t i = 0; i < system->shard_capacity; i++) {
            if (system->shard_param_ptrs[i]) {
                shard_aligned_free(system->shard_param_ptrs[i]);
            }
        }
        free(system->shard_param_ptrs);
        system->shard_param_ptrs = NULL;
    }

    if (system->shard_grad_ptrs) {
        for (size_t i = 0; i < system->shard_capacity; i++) {
            if (system->shard_grad_ptrs[i]) {
                shard_aligned_free(system->shard_grad_ptrs[i]);
            }
        }
        free(system->shard_grad_ptrs);
        system->shard_grad_ptrs = NULL;
    }

    if (system->shards) {
        free(system->shards);
        system->shards = NULL;
    }

    if (system->metrics) {
        free(system->metrics);
        system->metrics = NULL;
    }

    thread_mutex_destroy((thread_mutex_t*)&system->sync_mutex);

    free(system);
}

int shard_system_initialize(ParameterShardSystem* system)
{
    if (!system) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (system->is_initialized) return SELFLNN_SUCCESS;
    if (system->num_shards == 0) return SELFLNN_ERROR_INVALID_STATE;

    for (size_t i = 0; i < system->num_shards; i++) {
        system->shards[i].status = SHARD_STATUS_ACTIVE;
        system->shards[i].magic = SELFLNN_SHARD_MAGIC;
        system->metrics[i].read_latency_us = 0.0;
        system->metrics[i].write_latency_us = 0.0;
        system->metrics[i].compute_latency_us = 0.0;
        system->metrics[i].bandwidth_gbps = 0.0;
        system->metrics[i].total_bytes_transferred = 0;
        system->metrics[i].total_ops = 0;
        system->metrics[i].error_count = 0;
    }

    system->num_active_shards = (uint32_t)system->num_shards;
    system->global_param_offset = 0;

    thread_mutex_lock((thread_mutex_t*)&system->sync_mutex);
    system->is_initialized = 1;
    thread_mutex_unlock((thread_mutex_t*)&system->sync_mutex);

    return SELFLNN_SUCCESS;
}

int shard_system_destroy(ParameterShardSystem* system)
{
    if (!system) return SELFLNN_ERROR_INVALID_ARGUMENT;

    thread_mutex_lock((thread_mutex_t*)&system->sync_mutex);

    system->is_initialized = 0;
    system->num_active_shards = 0;

    for (size_t i = 0; i < system->num_shards; i++) {
        system->shards[i].status = SHARD_STATUS_UNINITIALIZED;
    }

    thread_mutex_unlock((thread_mutex_t*)&system->sync_mutex);

    return SELFLNN_SUCCESS;
}

int shard_system_add_shard(ParameterShardSystem* system, const ShardDescriptor* descriptor)
{
    if (!system || !descriptor) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (system->num_shards >= SELFLNN_MAX_SHARDS) return SELFLNN_ERROR_OPERATION_FAILED;

    thread_mutex_lock((thread_mutex_t*)&system->sync_mutex);

    size_t idx = system->num_shards;

    if (idx >= system->shard_capacity) {
        size_t new_capacity = system->shard_capacity == 0 ? 8 : system->shard_capacity * 2;
        if (new_capacity > SELFLNN_MAX_SHARDS) new_capacity = SELFLNN_MAX_SHARDS;

        ShardDescriptor* new_shards = (ShardDescriptor*)realloc(system->shards,
            new_capacity * sizeof(ShardDescriptor));
        ShardPerformanceMetrics* new_metrics = NULL;
        float** new_param_ptrs = NULL;
        float** new_grad_ptrs = NULL;

        /* 两步法：先全部realloc到临时变量，任一失败释放所有已成功的 */
        if (new_shards) {
            new_metrics = (ShardPerformanceMetrics*)realloc(system->metrics,
                new_capacity * sizeof(ShardPerformanceMetrics));
            if (new_metrics) {
                new_param_ptrs = (float**)realloc(system->shard_param_ptrs,
                    new_capacity * sizeof(float*));
                if (new_param_ptrs) {
                    new_grad_ptrs = (float**)realloc(system->shard_grad_ptrs,
                        new_capacity * sizeof(float*));
                }
            }
        }

        if (!new_shards || !new_metrics || !new_param_ptrs || !new_grad_ptrs) {
            /* 释放所有已成功的临时缓冲区 */
            if (new_shards) free(new_shards);
            if (new_metrics) free(new_metrics);
            if (new_param_ptrs) free(new_param_ptrs);
            if (new_grad_ptrs) free(new_grad_ptrs);
            thread_mutex_unlock((thread_mutex_t*)&system->sync_mutex);
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }

        memset(new_param_ptrs + system->shard_capacity, 0,
            (new_capacity - system->shard_capacity) * sizeof(float*));
        memset(new_grad_ptrs + system->shard_capacity, 0,
            (new_capacity - system->shard_capacity) * sizeof(float*));

        system->shards = new_shards;
        system->metrics = new_metrics;
        system->shard_param_ptrs = new_param_ptrs;
        system->shard_grad_ptrs = new_grad_ptrs;
        system->shard_capacity = new_capacity;
    }

    memcpy(&system->shards[idx], descriptor, sizeof(ShardDescriptor));
    system->shards[idx].shard_id = (uint32_t)idx;
    system->shards[idx].magic = SELFLNN_SHARD_MAGIC;

    size_t shard_mem_size = descriptor->num_params + SELFLNN_SHARD_ALIGNMENT;
    system->shard_param_ptrs[idx] = (float*)shard_aligned_alloc(SELFLNN_SHARD_ALIGNMENT,
        shard_mem_size * sizeof(float));
    system->shard_grad_ptrs[idx] = (float*)shard_aligned_alloc(SELFLNN_SHARD_ALIGNMENT,
        shard_mem_size * sizeof(float));

    if (!system->shard_param_ptrs[idx] || !system->shard_grad_ptrs[idx]) {
        shard_aligned_free(system->shard_param_ptrs[idx]);
        shard_aligned_free(system->shard_grad_ptrs[idx]);
        system->shard_param_ptrs[idx] = NULL;
        system->shard_grad_ptrs[idx] = NULL;
        thread_mutex_unlock((thread_mutex_t*)&system->sync_mutex);
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }

    memset(system->shard_param_ptrs[idx], 0, shard_mem_size * sizeof(float));
    memset(system->shard_grad_ptrs[idx], 0, shard_mem_size * sizeof(float));

    system->num_shards++;
    system->total_param_count += descriptor->num_params;
    system->num_active_shards++;

    thread_mutex_unlock((thread_mutex_t*)&system->sync_mutex);

    return SELFLNN_SUCCESS;
}

int shard_system_remove_shard(ParameterShardSystem* system, uint32_t shard_id)
{
    if (!system || !validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;

    thread_mutex_lock((thread_mutex_t*)&system->sync_mutex);

    if (system->shard_param_ptrs[shard_id]) {
        shard_aligned_free(system->shard_param_ptrs[shard_id]);
        system->shard_param_ptrs[shard_id] = NULL;
    }
    if (system->shard_grad_ptrs[shard_id]) {
        shard_aligned_free(system->shard_grad_ptrs[shard_id]);
        system->shard_grad_ptrs[shard_id] = NULL;
    }

    system->total_param_count -= system->shards[shard_id].num_params;

    for (size_t i = shard_id; i < system->num_shards - 1; i++) {
        memcpy(&system->shards[i], &system->shards[i + 1], sizeof(ShardDescriptor));
        system->shards[i].shard_id = (uint32_t)i;
        system->shard_param_ptrs[i] = system->shard_param_ptrs[i + 1];
        system->shard_grad_ptrs[i] = system->shard_grad_ptrs[i + 1];
        memcpy(&system->metrics[i], &system->metrics[i + 1], sizeof(ShardPerformanceMetrics));
    }

    system->shard_param_ptrs[system->num_shards - 1] = NULL;
    system->shard_grad_ptrs[system->num_shards - 1] = NULL;
    system->num_shards--;
    system->num_active_shards--;

    thread_mutex_unlock((thread_mutex_t*)&system->sync_mutex);

    return SELFLNN_SUCCESS;
}

int shard_system_get_shard(const ParameterShardSystem* system, uint32_t shard_id, ShardDescriptor* descriptor)
{
    if (!system || !descriptor || !validate_shard_id(system, shard_id))
        return SELFLNN_ERROR_INVALID_ARGUMENT;

    memcpy(descriptor, &system->shards[shard_id], sizeof(ShardDescriptor));
    return SELFLNN_SUCCESS;
}

int shard_system_distribute_parameters(ParameterShardSystem* system, const float* params, size_t num_params)
{
    if (!system || !params) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!system->is_initialized) return SELFLNN_ERROR_NOT_INITIALIZED;

    size_t copy_size = num_params < system->total_param_count ? num_params : system->total_param_count;
    memcpy(system->unified_param_buffer, params, copy_size * sizeof(float));

    size_t param_offset = 0;
    for (size_t i = 0; i < system->num_shards; i++) {
        size_t shard_size = system->shards[i].num_params;
        size_t copy_to_shard = shard_size;
        if (param_offset + copy_to_shard > copy_size) {
            copy_to_shard = copy_size > param_offset ? copy_size - param_offset : 0;
        }
        if (copy_to_shard > 0) {
            memcpy(system->shard_param_ptrs[i],
                   &params[param_offset],
                   copy_to_shard * sizeof(float));
        }
        param_offset += shard_size;
    }

    return SELFLNN_SUCCESS;
}

int shard_system_collect_parameters(const ParameterShardSystem* system, float* params, size_t num_params)
{
    if (!system || !params) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!system->is_initialized) return SELFLNN_ERROR_NOT_INITIALIZED;

    size_t total_collected = 0;

    for (size_t i = 0; i < system->num_shards; i++) {
        size_t shard_size = system->shards[i].num_params;
        size_t copy_size = shard_size;
        if (total_collected + copy_size > num_params) {
            copy_size = num_params > total_collected ? num_params - total_collected : 0;
        }
        if (copy_size > 0) {
            memcpy(&params[total_collected],
                   system->shard_param_ptrs[i],
                   copy_size * sizeof(float));
        }
        total_collected += shard_size;
    }

    return SELFLNN_SUCCESS;
}

int shard_system_synchronize_gradients(ParameterShardSystem* system)
{
    if (!system) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!system->is_initialized) return SELFLNN_ERROR_NOT_INITIALIZED;

    if (system->num_shards <= 1) return SELFLNN_SUCCESS;

    size_t max_shard_size = 0;
    for (size_t i = 0; i < system->num_shards; i++) {
        if (system->shards[i].num_params > max_shard_size) {
            max_shard_size = system->shards[i].num_params;
        }
    }

    float* reduce_buffer = (float*)malloc((size_t)max_shard_size * sizeof(float));
    if (!reduce_buffer) return SELFLNN_ERROR_OUT_OF_MEMORY;

    for (size_t i = 0; i < system->num_shards; i++) {
        memcpy(reduce_buffer, system->shard_grad_ptrs[i], system->shards[i].num_params * sizeof(float));

        thread_mutex_lock((thread_mutex_t*)&system->sync_mutex);

        for (size_t j = 0; j < system->num_shards; j++) {
            if (i == j) continue;
            size_t min_size = system->shards[i].num_params < system->shards[j].num_params ?
                              system->shards[i].num_params : system->shards[j].num_params;
            for (size_t k = 0; k < min_size; k++) {
                system->shard_grad_ptrs[j][k] += reduce_buffer[k];
            }
        }

        thread_mutex_unlock((thread_mutex_t*)&system->sync_mutex);
    }

    float inv_num_shards = 1.0f / (float)system->num_shards;
    for (size_t i = 0; i < system->num_shards; i++) {
        size_t shard_size = system->shards[i].num_params;
        for (size_t j = 0; j < shard_size; j++) {
            system->shard_grad_ptrs[i][j] *= inv_num_shards;
        }
    }

    system->metrics[0].total_ops += system->total_param_count * system->num_shards;

    free(reduce_buffer);
    return SELFLNN_SUCCESS;
}

int shard_system_allreduce_gradients(ParameterShardSystem* system, float* buffer, size_t size)
{
    if (!system || !buffer) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!system->is_initialized) return SELFLNN_ERROR_NOT_INITIALIZED;

    float* allreduce_buffer = (float*)malloc((size_t)size * sizeof(float));
    if (!allreduce_buffer) return SELFLNN_ERROR_OUT_OF_MEMORY;

    memset(allreduce_buffer, 0, size * sizeof(float));

    for (size_t i = 0; i < system->num_shards; i++) {
        size_t shard_size = system->shards[i].num_params;
        size_t min_size = shard_size < size ? shard_size : size;
        for (size_t j = 0; j < min_size; j++) {
            allreduce_buffer[j] += system->shard_grad_ptrs[i][j];
        }
    }

    float inv_num_shards = 1.0f / (float)system->num_shards;
    for (size_t i = 0; i < size; i++) {
        allreduce_buffer[i] *= inv_num_shards;
    }

    memcpy(buffer, allreduce_buffer, size * sizeof(float));

    for (size_t i = 0; i < system->num_shards; i++) {
        size_t shard_size = system->shards[i].num_params;
        size_t min_size = shard_size < size ? shard_size : size;
        memcpy(system->shard_grad_ptrs[i], allreduce_buffer, min_size * sizeof(float));
    }

    free(allreduce_buffer);
    system->metrics[0].total_ops += size * system->num_shards;

    return SELFLNN_SUCCESS;
}

int shard_system_scatter_parameters(ParameterShardSystem* system, const float* src, size_t src_size)
{
    if (!system || !src) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!system->is_initialized) return SELFLNN_ERROR_NOT_INITIALIZED;

    size_t offset = 0;
    for (size_t i = 0; i < system->num_shards; i++) {
        size_t shard_size = system->shards[i].num_params;
        size_t copy_size = shard_size;
        if (offset + copy_size > src_size) {
            copy_size = src_size > offset ? src_size - offset : 0;
        }
        if (copy_size > 0) {
            memcpy(system->shard_param_ptrs[i], &src[offset], copy_size * sizeof(float));
        }
        offset += shard_size;
    }

    system->metrics[0].total_bytes_transferred += offset * sizeof(float);
    return SELFLNN_SUCCESS;
}

/* ============================================================================
 * Top-K稀疏梯度压缩：减少通信量，保留最重要的梯度信号
 * ============================================================================ */

typedef struct {
    size_t index;
    float value;
} GradIndexPair;

static int grad_pair_cmp_desc(const void* a, const void* b) {
    float va = ((const GradIndexPair*)a)->value;
    float vb = ((const GradIndexPair*)b)->value;
    float abs_a = va < 0 ? -va : va;
    float abs_b = vb < 0 ? -vb : vb;
    if (abs_a > abs_b) return -1;
    if (abs_a < abs_b) return 1;
    return 0;
}

int shard_system_compress_topk(ParameterShardSystem* system,
                                const float* gradient, size_t gradient_size,
                                size_t top_k, float* compressed_buffer,
                                size_t* compressed_size) {
    if (!system || !gradient || !compressed_buffer || !compressed_size) return -1;
    if (top_k == 0 || top_k > gradient_size) return -1;

    GradIndexPair* pairs = (GradIndexPair*)malloc((size_t)gradient_size * sizeof(GradIndexPair));
    if (!pairs) return -1;

    for (size_t i = 0; i < gradient_size; i++) {
        pairs[i].index = i;
        pairs[i].value = gradient[i];
    }

    qsort(pairs, gradient_size, sizeof(GradIndexPair), grad_pair_cmp_desc);

    /* 存储Top-K索引和值对 */
    size_t out_idx = 0;
    for (size_t i = 0; i < top_k && i < gradient_size; i++) {
        compressed_buffer[out_idx++] = (float)pairs[i].index;
        compressed_buffer[out_idx++] = pairs[i].value;
    }

    *compressed_size = out_idx;
    free(pairs);

    /* 更新压缩统计 */
    if (system->metrics && system->num_shards > 0) {
        system->metrics[0].total_bytes_transferred += *compressed_size * sizeof(float);
        system->metrics[0].total_ops++;
    }

    return 0;
}

int shard_system_decompress_topk(const float* compressed_buffer, size_t compressed_size,
                                   float* gradient_out, size_t gradient_size,
                                   size_t top_k) {
    if (!compressed_buffer || !gradient_out) return -1;

    memset(gradient_out, 0, gradient_size * sizeof(float));

    size_t num_pairs = compressed_size / 2;
    if (num_pairs > top_k) num_pairs = top_k;

    for (size_t i = 0; i < num_pairs; i++) {
        size_t index = (size_t)compressed_buffer[i * 2];
        float value = compressed_buffer[i * 2 + 1];
        if (index < gradient_size) {
            gradient_out[index] = value;
        }
    }

    return 0;
}

int shard_system_sync_compressed(ParameterShardSystem* system,
                                   const float* local_gradient, size_t gradient_size,
                                   size_t top_k_ratio_percent) {
    if (!system || !local_gradient) return -1;
    if (top_k_ratio_percent == 0 || top_k_ratio_percent > 100) return -1;

    size_t top_k = gradient_size * top_k_ratio_percent / 100;
    if (top_k < 1) top_k = 1;
    if (top_k > gradient_size) top_k = gradient_size;

    float* compressed = (float*)malloc((size_t)top_k * 2 * sizeof(float));
    if (!compressed) return -1;

    size_t compressed_size = 0;
    int ret = shard_system_compress_topk(system, local_gradient, gradient_size,
                                          top_k, compressed, &compressed_size);
    if (ret != 0) {
        free(compressed);
        return ret;
    }

    /* AllReduce压缩后的梯度 */
    float* global_buffer = (float*)calloc(gradient_size, sizeof(float));
    if (!global_buffer) {
        free(compressed);
        return -1;
    }

    shard_system_decompress_topk(compressed, compressed_size,
                                   global_buffer, gradient_size, top_k);

    float* all_reduced = (float*)malloc((size_t)gradient_size * sizeof(float));
    if (!all_reduced) {
        free(compressed);
        free(global_buffer);
        return -1;
    }

    shard_system_allreduce_gradients(system, global_buffer, gradient_size);

    for (size_t i = 0; i < gradient_size; i++) {
        all_reduced[i] = global_buffer[i] * (float)system->num_shards;
    }

    memcpy((void*)local_gradient, all_reduced, gradient_size * sizeof(float));

    free(all_reduced);
    free(global_buffer);
    free(compressed);

    return 0;
}

int shard_system_gather_parameters(const ParameterShardSystem* system, float* dst, size_t dst_size)
{
    if (!system || !dst) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!system->is_initialized) return SELFLNN_ERROR_NOT_INITIALIZED;

    size_t offset = 0;
    for (size_t i = 0; i < system->num_shards; i++) {
        size_t shard_size = system->shards[i].num_params;
        size_t copy_size = shard_size;
        if (offset + copy_size > dst_size) {
            copy_size = dst_size > offset ? dst_size - offset : 0;
        }
        if (copy_size > 0) {
            memcpy(&dst[offset], system->shard_param_ptrs[i], copy_size * sizeof(float));
        }
        offset += shard_size;
    }

    system->metrics[0].total_bytes_transferred += offset * sizeof(float);
    return SELFLNN_SUCCESS;
}

int shard_system_forward_shard(ParameterShardSystem* system, uint32_t shard_id,
                              const float* input, float* output, size_t batch_size)
{
    if (!system || !input || !output) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!system->is_initialized) return SELFLNN_ERROR_NOT_INITIALIZED;

    size_t shard_params = system->shards[shard_id].num_params;
    float* shard_params_ptr = system->shard_param_ptrs[shard_id];

    for (size_t b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        for (size_t j = 0; j < shard_params; j++) {
            sum += input[b * shard_params + j] * shard_params_ptr[j];
        }
        output[b] = 1.0f / (1.0f + expf(-sum));
    }

    system->metrics[shard_id].total_ops += batch_size * shard_params;
    return SELFLNN_SUCCESS;
}

int shard_system_backward_shard(ParameterShardSystem* system, uint32_t shard_id,
                               const float* grad_output, float* grad_input, size_t batch_size)
{
    if (!system || !grad_output) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!system->is_initialized) return SELFLNN_ERROR_NOT_INITIALIZED;

    size_t shard_params = system->shards[shard_id].num_params;
    float* shard_params_ptr = system->shard_param_ptrs[shard_id];
    float* shard_grads_ptr = system->shard_grad_ptrs[shard_id];

    for (size_t j = 0; j < shard_params; j++) {
        float grad_sum = 0.0f;
        for (size_t b = 0; b < batch_size; b++) {
            grad_sum += grad_output[b] * shard_params_ptr[j];
        }
        if (grad_input) {
            size_t idx = 0;
            for (size_t b = 0; b < batch_size; b++) {
                grad_input[b] += shard_params_ptr[j] * grad_output[b];
                idx++;
            }
        }
        shard_grads_ptr[j] += grad_sum / (float)batch_size;
    }

    system->metrics[shard_id].total_ops += shard_params * batch_size;
    return SELFLNN_SUCCESS;
}

int shard_system_get_shard_params(const ParameterShardSystem* system, uint32_t shard_id,
                                 float** params, size_t* num_params)
{
    if (!system || !params || !num_params) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;

    *params = system->shard_param_ptrs[shard_id];
    *num_params = system->shards[shard_id].num_params;
    return SELFLNN_SUCCESS;
}

int shard_system_set_shard_params(ParameterShardSystem* system, uint32_t shard_id,
                                 const float* params, size_t num_params)
{
    if (!system || !params) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;

    size_t copy_size = num_params < system->shards[shard_id].num_params ?
                       num_params : system->shards[shard_id].num_params;
    memcpy(system->shard_param_ptrs[shard_id], params, copy_size * sizeof(float));
    return SELFLNN_SUCCESS;
}

int shard_system_get_shard_gradients(const ParameterShardSystem* system, uint32_t shard_id,
                                    float** gradients, size_t* num_gradients)
{
    if (!system || !gradients || !num_gradients) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;

    *gradients = system->shard_grad_ptrs[shard_id];
    *num_gradients = system->shards[shard_id].num_params;
    return SELFLNN_SUCCESS;
}

int shard_system_zero_gradients(ParameterShardSystem* system)
{
    if (!system) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!system->is_initialized) return SELFLNN_ERROR_NOT_INITIALIZED;

    for (size_t i = 0; i < system->num_shards; i++) {
        memset(system->shard_grad_ptrs[i], 0, system->shards[i].num_params * sizeof(float));
    }

    return SELFLNN_SUCCESS;
}

int shard_system_get_metrics(const ParameterShardSystem* system, uint32_t shard_id,
                            ShardPerformanceMetrics* metrics)
{
    if (!system || !metrics) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;

    memcpy(metrics, &system->metrics[shard_id], sizeof(ShardPerformanceMetrics));
    return SELFLNN_SUCCESS;
}

int shard_system_rebalance(ParameterShardSystem* system)
{
    if (!system) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!system->is_initialized) return SELFLNN_ERROR_NOT_INITIALIZED;

    size_t total_params = system->total_param_count;
    size_t new_base = total_params / system->num_shards;
    size_t remainder = total_params % system->num_shards;

    float* all_params = (float*)malloc((size_t)total_params * sizeof(float));
    if (!all_params) return SELFLNN_ERROR_OUT_OF_MEMORY;

    if (shard_system_gather_parameters(system, all_params, total_params) != SELFLNN_SUCCESS) {
        free(all_params);
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    size_t offset = 0;
    for (size_t i = 0; i < system->num_shards; i++) {
        size_t new_size = new_base + (i < remainder ? 1 : 0);
        system->shards[i].num_params = new_size;

        float* new_param_buf = (float*)shard_aligned_alloc(SELFLNN_SHARD_ALIGNMENT,
                                                     (new_size + SELFLNN_SHARD_ALIGNMENT) * sizeof(float));
        float* new_grad_buf = (float*)shard_aligned_alloc(SELFLNN_SHARD_ALIGNMENT,
                                                    (new_size + SELFLNN_SHARD_ALIGNMENT) * sizeof(float));
        if (!new_param_buf || !new_grad_buf) {
            free(all_params);
            shard_aligned_free(new_param_buf);
            shard_aligned_free(new_grad_buf);
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }

        memcpy(new_param_buf, &all_params[offset], new_size * sizeof(float));
        memset(new_grad_buf, 0, new_size * sizeof(float));

        shard_aligned_free(system->shard_param_ptrs[i]);
        shard_aligned_free(system->shard_grad_ptrs[i]);
        system->shard_param_ptrs[i] = new_param_buf;
        system->shard_grad_ptrs[i] = new_grad_buf;

        offset += new_size;
    }

    free(all_params);
    return SELFLNN_SUCCESS;
}

int shard_system_checkpoint(const ParameterShardSystem* system, const char* filepath)
{
    if (!system || !filepath) return SELFLNN_ERROR_INVALID_ARGUMENT;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return SELFLNN_ERROR_OPERATION_FAILED;

    ShardCheckpointHeader header;
    header.magic = SHARD_CHECKPOINT_MAGIC;
    header.version = SHARD_SYSTEM_VERSION;
    header.num_shards = system->num_shards;
    header.total_params = system->total_param_count;
    memcpy(header.shards, system->shards, system->num_shards * sizeof(ShardDescriptor));

    if (fwrite(&header, sizeof(ShardCheckpointHeader), 1, fp) != 1) {
        fclose(fp);
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    for (size_t i = 0; i < system->num_shards; i++) {
        size_t shard_size = system->shards[i].num_params;
        if (fwrite(system->shard_param_ptrs[i], sizeof(float), shard_size, fp) != shard_size) {
            fclose(fp);
            return SELFLNN_ERROR_OPERATION_FAILED;
        }
    }

    fclose(fp);
    return SELFLNN_SUCCESS;
}

int shard_system_restore(ParameterShardSystem* system, const char* filepath)
{
    if (!system || !filepath) return SELFLNN_ERROR_INVALID_ARGUMENT;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return SELFLNN_ERROR_OPERATION_FAILED;

    ShardCheckpointHeader header;
    if (fread(&header, sizeof(ShardCheckpointHeader), 1, fp) != 1) {
        fclose(fp);
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    if (header.magic != SHARD_CHECKPOINT_MAGIC) {
        fclose(fp);
        return SELFLNN_ERROR_OPERATION_FAILED;
    }

    if (header.num_shards != system->num_shards) {
        fclose(fp);
        return SELFLNN_ERROR_INVALID_STATE;
    }

    for (size_t i = 0; i < system->num_shards; i++) {
        size_t shard_size = system->shards[i].num_params;
        if (fread(system->shard_param_ptrs[i], sizeof(float), shard_size, fp) != shard_size) {
            fclose(fp);
            return SELFLNN_ERROR_OPERATION_FAILED;
        }
    }

    system->total_param_count = header.total_params;
    memcpy(system->shards, header.shards, system->num_shards * sizeof(ShardDescriptor));

    fclose(fp);
    return SELFLNN_SUCCESS;
}

size_t shard_system_get_total_params(const ParameterShardSystem* system)
{
    if (!system) return 0;
    return system->total_param_count;
}

size_t shard_system_get_active_shards(const ParameterShardSystem* system)
{
    if (!system) return 0;
    return system->num_active_shards;
}

int shard_system_is_available(const ParameterShardSystem* system)
{
    if (!system) return 0;
    return system->is_initialized && system->num_active_shards > 0;
}

int shard_system_offload_shard(ParameterShardSystem* system, uint32_t shard_id, ShardLocation target_location)
{
    if (!system || !validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;

    thread_mutex_lock((thread_mutex_t*)&system->sync_mutex);

    system->shards[shard_id].status = SHARD_STATUS_OFFLOADED;
    system->shards[shard_id].location = target_location;
    system->num_active_shards--;

    thread_mutex_unlock((thread_mutex_t*)&system->sync_mutex);

    return SELFLNN_SUCCESS;
}

int shard_system_load_shard(ParameterShardSystem* system, uint32_t shard_id)
{
    if (!system || !validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;

    thread_mutex_lock((thread_mutex_t*)&system->sync_mutex);

    system->shards[shard_id].status = SHARD_STATUS_ACTIVE;
    system->num_active_shards++;

    thread_mutex_unlock((thread_mutex_t*)&system->sync_mutex);

    return SELFLNN_SUCCESS;
}

int shard_system_prefetch_shard(ParameterShardSystem* system, uint32_t shard_id)
{
    if (!system || !validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;

    system->shards[shard_id].status = SHARD_STATUS_SYNCING;
    system->shards[shard_id].status = SHARD_STATUS_ACTIVE;

    return SELFLNN_SUCCESS;
}

int shard_system_pipeline_transfer(ParameterShardSystem* system, uint32_t src_shard, uint32_t dst_shard,
                                  size_t offset, size_t size)
{
    if (!system) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!validate_shard_id(system, src_shard) || !validate_shard_id(system, dst_shard))
        return SELFLNN_ERROR_INVALID_ARGUMENT;

    size_t src_size = system->shards[src_shard].num_params;
    size_t dst_size = system->shards[dst_shard].num_params;

    if (offset >= src_size || offset >= dst_size) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (offset + size > src_size || offset + size > dst_size) {
        size = (src_size - offset) < (dst_size - offset) ? (src_size - offset) : (dst_size - offset);
    }

    size_t remaining = size;
    size_t current_offset = offset;
    while (remaining > 0) {
        size_t block_size = remaining < TRANSFER_BLOCK_SIZE ? remaining : TRANSFER_BLOCK_SIZE;
        memcpy(&system->shard_param_ptrs[dst_shard][current_offset],
               &system->shard_param_ptrs[src_shard][current_offset],
               block_size * sizeof(float));
        current_offset += block_size;
        remaining -= block_size;
    }

    system->metrics[src_shard].total_bytes_transferred += size * sizeof(float);
    system->metrics[dst_shard].total_bytes_transferred += size * sizeof(float);

    return SELFLNN_SUCCESS;
}

int shard_system_register_sync_callback(ParameterShardSystem* system,
                                       ShardSyncCallback callback, void* user_data)
{
    if (!system || !callback) return SELFLNN_ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < PS_MAX_SYNC_CALLBACKS; i++) {
        if (!system->sync_callbacks[i].active) {
            system->sync_callbacks[i].callback = callback;
            system->sync_callbacks[i].user_data = user_data;
            system->sync_callbacks[i].active = 1;
            return SELFLNN_SUCCESS;
        }
    }
    return SELFLNN_ERROR_OUT_OF_MEMORY;
}

int shard_system_register_transfer_func(ParameterShardSystem* system, ShardTransferFunc func)
{
    if (!system || !func) return SELFLNN_ERROR_INVALID_ARGUMENT;
    system->transfer_func = func;
    return SELFLNN_SUCCESS;
}

/* S-009修复: 基于TCP Socket的分片间梯度同步
 * 替代纯进程内共享指针方式，支持跨进程/跨机器的参数分片同步
 * 每个分片可在不同进程的网络节点上，通过TCP发送/接收梯度
 *
 * 协议格式: [shard_id:4][param_count:4][gradient_data:param_count*4]
 */
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

int shard_system_sync_over_socket(ParameterShardSystem* system,
    const char* peer_host, unsigned short peer_port, uint32_t shard_id)
{
    if (!system || !peer_host) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!validate_shard_id(system, shard_id)) return SELFLNN_ERROR_INVALID_ARGUMENT;

    size_t param_count = system->shards[shard_id].num_params;
    if (param_count == 0) return SELFLNN_SUCCESS;

    float* grads = system->shard_grad_ptrs[shard_id];
    if (!grads) return SELFLNN_ERROR_INVALID_ARGUMENT;

    /* 创建TCP客户端socket */
    int sock = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return SELFLNN_ERROR_GENERIC;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(peer_port);
    addr.sin_addr.s_addr = inet_addr(peer_host);

    /* 连接并发送/接收梯度 */
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        size_t payload_size = param_count * sizeof(float);
        uint32_t header[2] = { shard_id, (uint32_t)param_count };

        /* 发送本地梯度 */
        if (send(sock, (const char*)header, sizeof(header), 0) == sizeof(header)) {
            if (send(sock, (const char*)grads, (int)payload_size, 0) == (int)payload_size) {
                /* 接收远程梯度并求平均 */
                float* remote = (float*)malloc((size_t)param_count * sizeof(float));
                if (remote) {
                    size_t received = 0;
                    char* rbuf = (char*)remote;
                    while (received < payload_size) {
                        int n = recv(sock, rbuf + received,
                            (int)(payload_size - received), 0);
                        if (n <= 0) break;
                        received += (size_t)n;
                    }
                    if (received >= payload_size) {
                        for (size_t i = 0; i < param_count; i++)
                            grads[i] = (grads[i] + remote[i]) * 0.5f;
                    }
                    free(remote);
                }
            }
        }
    }

#ifdef _WIN32
    closesocket((SOCKET)sock);
#else
    close(sock);
#endif
    return SELFLNN_SUCCESS;
}
