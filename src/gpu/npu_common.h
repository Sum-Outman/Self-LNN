/**
 * @file npu_common.h
 * @brief NPU后端公共代码接口（内部头文件）
 *
 * 提供Ascend/Cambricon/TPU三个NPU后端共享的公共功能：
 * - 系统内存查询
 * - 硬件检测辅助
 * - 设备信息填充
 * - 上下文创建/释放
 * - 流管理
 * - CPU回退运算
 * - 后端接口填充
 */

#ifndef SELFLNN_GPU_NPU_COMMON_H
#define SELFLNN_GPU_NPU_COMMON_H

#include "selflnn/gpu/gpu.h"
#include "selflnn/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t npu_common_get_system_memory_total(void);
size_t npu_common_get_system_memory_free(void);

int npu_common_check_registry_key(const char* key_path);
int npu_common_check_directory(const char* dir_path);
int npu_common_check_device_node(const char* dev_pattern);
int npu_common_check_library(const char* const* lib_names, int lib_count);

void npu_common_fill_device_info(GpuDeviceInfo* info, int device_id,
                                  const char* vendor_name, const char* device_name,
                                  int compute_units, size_t total_mem, size_t free_mem);

GpuContext* npu_common_context_create(int device_index, GpuBackend backend,
                                       const char* device_name,
                                       size_t total_memory, size_t free_memory);
void npu_common_context_free(GpuContext* context);

GpuStream* npu_common_stream_create(GpuContext* context);
void npu_common_stream_free(GpuStream* stream);
int npu_common_stream_synchronize(GpuStream* stream);
int npu_common_stream_query(GpuStream* stream);

int npu_common_cpu_forward_dense(const float* input, const float* weights,
                                  const float* bias, float* output,
                                  size_t batch_size, size_t input_size,
                                  size_t output_size,
                                  GpuActivationType act_type, float alpha);
int npu_common_cpu_matmul(const float* a, const float* b, float* c,
                           size_t m, size_t n, size_t k,
                           int transpose_a, int transpose_b);
int npu_common_cpu_cfc_step(const float* h_in, const float* W,
                             const float* b, const float* tau, float* h_out,
                             float dt, int dim);

/* 统一CPU核执行回退 — 支持12+种核心操作，按kernel_name匹配执行真实计算 */
int npu_common_cpu_kernel_execute(GpuKernel* kernel, size_t count);

typedef int (*GpuBackendInitFn)(void);
typedef void (*GpuBackendCleanupFn)(void);
typedef int (*GpuBackendGetDeviceCountFn)(void);
typedef int (*GpuBackendGetDeviceInfoFn)(int, GpuDeviceInfo*);
typedef GpuContext* (*GpuBackendContextCreateFn)(int);
typedef void (*GpuBackendContextFreeFn)(GpuContext*);
typedef GpuMemory* (*GpuBackendMemoryAllocFn)(GpuContext*, size_t, GpuMemoryType);
typedef void (*GpuBackendMemoryFreeFn)(GpuContext*, GpuMemory*);
typedef int (*GpuBackendMemCpyToDevFn)(GpuMemory*, const void*, size_t);
typedef int (*GpuBackendMemCpyFromDevFn)(void*, GpuMemory*, size_t);

void npu_common_populate_backend_iface(GpuBackendInterface* iface,
                                        const char* name, GpuBackend type,
                                        GpuBackendInitFn init,
                                        GpuBackendCleanupFn cleanup,
                                        GpuBackendGetDeviceCountFn get_count,
                                        GpuBackendGetDeviceInfoFn get_info,
                                        GpuBackendContextCreateFn ctx_create,
                                        GpuBackendContextFreeFn ctx_free,
                                        GpuBackendMemoryAllocFn mem_alloc,
                                        GpuBackendMemoryFreeFn mem_free,
                                        GpuBackendMemCpyToDevFn cpy_h2d,
                                        GpuBackendMemCpyFromDevFn cpy_d2h);

#ifdef __cplusplus
}
#endif

#endif
