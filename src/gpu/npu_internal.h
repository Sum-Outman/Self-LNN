#ifndef SELFLNN_NPU_INTERNAL_H
#define SELFLNN_NPU_INTERNAL_H

#include "selflnn/gpu/gpu.h"
#include "selflnn/core/common.h"
#include "selflnn/utils/logging.h"
#include <stddef.h>

/* NPU后端日志宏（兼容Ascend/Cambricon/TPU SDK风格） */
#define LOG_WARN(...)  log_warning(__VA_ARGS__)
#define LOG_INFO(...)  log_info(__VA_ARGS__)
#define LOG_ERROR(...) log_error(__VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_MAX_INPUTS 16
#define NPU_MAX_OUTPUTS 16
#define NPU_MAX_DIMS 8
#define NPU_MODEL_NAME_MAX 256

/**
 * @brief NPU后端接口结构体
 *
 * 每个NPU后端（Ascend/Cambricon/TPU）需要实现此接口。
 */
typedef struct {
    const char* name;
    GpuBackend backend_type;
    int (*npu_init)(GpuContext* context);
    void (*npu_cleanup)(GpuContext* context);
    int (*npu_get_device_count)(GpuContext* context);
    const char* (*npu_get_backend_name)(GpuContext* context);
    NpuModel* (*npu_load_model)(GpuContext* context, const char* model_path,
                                const NpuInferenceConfig* config);
    void (*npu_unload_model)(NpuModel* model);
    int (*npu_infer)(NpuModel* model, const float** inputs, float** outputs,
                     int batch_size);
    int (*npu_infer_async)(NpuModel* model, const float** inputs,
                           float** outputs, int batch_size);
    int (*npu_infer_wait)(NpuModel* model, int timeout_ms);
    int (*npu_get_model_info)(NpuModel* model, NpuModelInfo* info);
} NpuBackendInterface;

/**
 * @brief NPU模型内部结构体
 */
struct NpuModel {
    GpuContext* context;
    NpuBackendInterface* backend;
    char model_path[512];
    char model_name[NPU_MODEL_NAME_MAX];
    size_t model_size_bytes;
    int input_count;
    int output_count;
    size_t input_sizes[NPU_MAX_INPUTS];
    size_t output_sizes[NPU_MAX_OUTPUTS];
    int input_dims[NPU_MAX_INPUTS][NPU_MAX_DIMS];
    int output_dims[NPU_MAX_OUTPUTS][NPU_MAX_DIMS];
    int input_dim_counts[NPU_MAX_INPUTS];
    int output_dim_counts[NPU_MAX_OUTPUTS];
    int batch_size;
    int enable_fp16;
    int enable_profiling;
    int timeout_ms;
    int is_loaded;
    float estimated_inference_time_ms;
    void* backend_data;
    int async_in_flight;
};

/**
 * @brief NPU后端接口声明函数
 */
const NpuBackendInterface* ascend_get_npu_interface(void);
const NpuBackendInterface* cambricon_get_npu_interface(void);
const NpuBackendInterface* tpu_get_npu_interface(void);

/**
 * @brief 根据GpuContext获取对应的NPU后端接口
 */
NpuBackendInterface* npu_get_backend_for_context(GpuContext* context);

#ifdef __cplusplus
}
#endif

#endif
