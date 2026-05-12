#ifndef SELFLNN_PARAMETER_SHARD_H
#define SELFLNN_PARAMETER_SHARD_H

#include <stddef.h>
#include <stdint.h>
#include "selflnn/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SELFLNN_MAX_SHARDS 1024
#define SELFLNN_MAX_SHARD_NAME 64
#define SELFLNN_SHARD_ALIGNMENT 256
#define SELFLNN_SHARD_MAGIC 0x53485244
#define PS_MAX_SYNC_CALLBACKS 16

typedef enum {
    SHARD_STATUS_UNINITIALIZED = 0,
    SHARD_STATUS_ACTIVE = 1,
    SHARD_STATUS_SYNCING = 2,
    SHARD_STATUS_ERROR = 3,
    SHARD_STATUS_OFFLOADED = 4
} ShardStatus;

typedef enum {
    SHARD_LOCATION_HOST = 0,
    SHARD_LOCATION_DEVICE_CUDA = 1,
    SHARD_LOCATION_DEVICE_ROCM = 2,
    SHARD_LOCATION_DEVICE_OPENCL = 3,
    SHARD_LOCATION_DEVICE_VULKAN = 4,
    SHARD_LOCATION_DEVICE_METAL = 5,
    SHARD_LOCATION_DEVICE_ASCEND = 6,
    SHARD_LOCATION_DEVICE_CAMBRICON = 7,
    SHARD_LOCATION_DEVICE_TPU = 8,
    SHARD_LOCATION_NETWORK = 9,
    SHARD_LOCATION_DISK = 10
} ShardLocation;

typedef enum {
    SHARD_PRECISION_FP32 = 0,
    SHARD_PRECISION_FP16 = 1,
    SHARD_PRECISION_BF16 = 2,
    SHARD_PRECISION_INT8 = 3,
    SHARD_PRECISION_INT4 = 4
} ShardPrecision;

typedef struct {
    uint32_t magic;
    uint32_t shard_id;
    char name[SELFLNN_MAX_SHARD_NAME];
    ShardLocation location;
    ShardPrecision precision;
    ShardStatus status;
    size_t num_params;
    size_t param_offset;
    size_t param_size_bytes;
    size_t gradient_offset;
    size_t gradient_size_bytes;
    uint64_t device_id;
    uint32_t node_id;
    uint32_t numa_node;
    float memory_usage_gb;
    float compute_power_tflops;
    double bandwidth_gbps;
} ShardDescriptor;

typedef struct {
    double read_latency_us;
    double write_latency_us;
    double compute_latency_us;
    double bandwidth_gbps;
    uint64_t total_bytes_transferred;
    uint64_t total_ops;
    uint64_t error_count;
    uint64_t last_error_time;
} ShardPerformanceMetrics;

/* 前向声明回调类型，供ParameterShardSystem结构体使用 */
typedef void (*ShardSyncCallback)(uint32_t shard_id,
                                 ShardStatus old_status,
                                 ShardStatus new_status,
                                 void* user_data);

typedef int (*ShardTransferFunc)(const void* src,
                                 void* dst,
                                 size_t size,
                                 uint32_t src_device,
                                 uint32_t dst_device);

typedef struct {
    ShardSyncCallback callback;
    void* user_data;
    int active;
} ShardSyncCallbackEntry;

typedef struct {
    size_t num_shards;
    size_t num_devices;
    size_t total_param_count;
    float* unified_param_buffer;
    float** shard_param_ptrs;
    float** shard_grad_ptrs;
    ShardDescriptor* shards;
    ShardPerformanceMetrics* metrics;
    size_t shard_capacity;
    size_t global_param_offset;
    int enable_async_transfer;
    int enable_gradient_compression;
    float compression_ratio;
    int enable_overlap_computation;
    size_t gradient_sync_interval;
    size_t num_active_shards;
    volatile int is_initialized;
    void* sync_mutex;
    void* transfer_streams;
    ShardSyncCallbackEntry sync_callbacks[PS_MAX_SYNC_CALLBACKS];
    ShardTransferFunc transfer_func;
} ParameterShardSystem;

typedef struct {
    size_t num_shards;
    size_t total_params;
    ShardPrecision default_precision;
    int enable_async;
    int enable_gradient_compression;
    float compression_ratio;
    int enable_overlap;
    size_t gradient_sync_interval;
    uint64_t* device_ids;
    ShardLocation* locations;
    size_t* shard_sizes;
} ShardSystemConfig;

ParameterShardSystem* shard_system_create(const ShardSystemConfig* config);
void shard_system_free(ParameterShardSystem* system);

int shard_system_initialize(ParameterShardSystem* system);
int shard_system_destroy(ParameterShardSystem* system);

int shard_system_add_shard(ParameterShardSystem* system,
                          const ShardDescriptor* descriptor);

int shard_system_remove_shard(ParameterShardSystem* system, uint32_t shard_id);

int shard_system_get_shard(const ParameterShardSystem* system,
                          uint32_t shard_id,
                          ShardDescriptor* descriptor);

int shard_system_distribute_parameters(ParameterShardSystem* system,
                                      const float* params,
                                      size_t num_params);

int shard_system_collect_parameters(const ParameterShardSystem* system,
                                   float* params,
                                   size_t num_params);

int shard_system_synchronize_gradients(ParameterShardSystem* system);

int shard_system_allreduce_gradients(ParameterShardSystem* system,
                                    float* buffer,
                                    size_t size);

int shard_system_scatter_parameters(ParameterShardSystem* system,
                                   const float* src,
                                   size_t src_size);

int shard_system_gather_parameters(const ParameterShardSystem* system,
                                  float* dst,
                                  size_t dst_size);

int shard_system_forward_shard(ParameterShardSystem* system,
                              uint32_t shard_id,
                              const float* input,
                              float* output,
                              size_t batch_size);

int shard_system_backward_shard(ParameterShardSystem* system,
                               uint32_t shard_id,
                               const float* grad_output,
                               float* grad_input,
                               size_t batch_size);

int shard_system_get_shard_params(const ParameterShardSystem* system,
                                 uint32_t shard_id,
                                 float** params,
                                 size_t* num_params);

int shard_system_set_shard_params(ParameterShardSystem* system,
                                 uint32_t shard_id,
                                 const float* params,
                                 size_t num_params);

int shard_system_get_shard_gradients(const ParameterShardSystem* system,
                                    uint32_t shard_id,
                                    float** gradients,
                                    size_t* num_gradients);

int shard_system_zero_gradients(ParameterShardSystem* system);

int shard_system_get_metrics(const ParameterShardSystem* system,
                            uint32_t shard_id,
                            ShardPerformanceMetrics* metrics);

int shard_system_rebalance(ParameterShardSystem* system);

int shard_system_checkpoint(const ParameterShardSystem* system,
                           const char* filepath);

int shard_system_restore(ParameterShardSystem* system,
                        const char* filepath);

size_t shard_system_get_total_params(const ParameterShardSystem* system);

size_t shard_system_get_active_shards(const ParameterShardSystem* system);

int shard_system_is_available(const ParameterShardSystem* system);

int shard_system_offload_shard(ParameterShardSystem* system,
                              uint32_t shard_id,
                              ShardLocation target_location);

int shard_system_load_shard(ParameterShardSystem* system,
                           uint32_t shard_id);

int shard_system_prefetch_shard(ParameterShardSystem* system,
                               uint32_t shard_id);

int shard_system_pipeline_transfer(ParameterShardSystem* system,
                                  uint32_t src_shard,
                                  uint32_t dst_shard,
                                  size_t offset,
                                  size_t size);

int shard_system_register_sync_callback(ParameterShardSystem* system,
                                       ShardSyncCallback callback,
                                       void* user_data);

int shard_system_register_transfer_func(ParameterShardSystem* system,
                                       ShardTransferFunc func);

#ifdef __cplusplus
}
#endif

#endif
