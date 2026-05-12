#include "selflnn/gpu/gpu.h"
#include "selflnn/core/common.h"
#include "selflnn/utils/logging.h"
#include "gpu_internal.h"
#include "npu_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static NpuBackendInterface* g_active_npu_backend = NULL;

NpuBackendInterface* npu_get_backend_for_context(GpuContext* context) {
    if (!context) return NULL;
    GpuBackend be_type = context->backend;
    switch (be_type) {
        case GPU_BACKEND_ASCEND:
            return (NpuBackendInterface*)ascend_get_npu_interface();
        case GPU_BACKEND_CAMBRICON:
            return (NpuBackendInterface*)cambricon_get_npu_interface();
        case GPU_BACKEND_TPU:
            return (NpuBackendInterface*)tpu_get_npu_interface();
        default:
            LOG_ERROR("当前GPU后端不支持NPU推理: %d", (int)be_type);
            return NULL;
    }
}

int gpu_npu_init(GpuContext* context) {
    if (!context) {
        LOG_ERROR("gpu_npu_init: 上下文为空");
        return -1;
    }
    NpuBackendInterface* iface = npu_get_backend_for_context(context);
    if (!iface) {
        LOG_ERROR("gpu_npu_init: 无法获取NPU后端接口");
        return -1;
    }
    g_active_npu_backend = iface;
    int ret = iface->npu_init(context);
    if (ret == 0) {
        LOG_INFO("NPU后端初始化成功: %s", iface->name);
    } else {
        LOG_ERROR("NPU后端初始化失败: %s", iface->name);
    }
    return ret;
}

void gpu_npu_cleanup(GpuContext* context) {
    if (!context) return;
    NpuBackendInterface* iface = npu_get_backend_for_context(context);
    if (iface) {
        iface->npu_cleanup(context);
    }
    g_active_npu_backend = NULL;
}

NpuModel* gpu_npu_load_model(GpuContext* context, const char* model_path,
                             const NpuInferenceConfig* config) {
    if (!context || !model_path) {
        LOG_ERROR("gpu_npu_load_model: 参数无效");
        return NULL;
    }
    NpuBackendInterface* iface = npu_get_backend_for_context(context);
    if (!iface) {
        LOG_ERROR("gpu_npu_load_model: 无法获取NPU后端接口");
        return NULL;
    }
    return iface->npu_load_model(context, model_path, config);
}

void gpu_npu_unload_model(NpuModel* model) {
    if (!model || !model->backend) return;
    model->backend->npu_unload_model(model);
}

int gpu_npu_infer(NpuModel* model, const float** inputs, float** outputs,
                  int batch_size) {
    if (!model || !model->backend || !inputs || !outputs || batch_size <= 0) {
        LOG_ERROR("gpu_npu_infer: 参数无效");
        return -1;
    }
    return model->backend->npu_infer(model, inputs, outputs, batch_size);
}

int gpu_npu_infer_async(NpuModel* model, const float** inputs, float** outputs,
                        int batch_size) {
    if (!model || !model->backend || !inputs || !outputs || batch_size <= 0) {
        LOG_ERROR("gpu_npu_infer_async: 参数无效");
        return -1;
    }
    if (model->async_in_flight) {
        LOG_ERROR("gpu_npu_infer_async: 上一次异步推理尚未完成");
        return -1;
    }
    int ret = model->backend->npu_infer_async(model, inputs, outputs, batch_size);
    if (ret == 0) {
        model->async_in_flight = 1;
    }
    return ret;
}

int gpu_npu_infer_wait(NpuModel* model, int timeout_ms) {
    if (!model || !model->backend) {
        LOG_ERROR("gpu_npu_infer_wait: 参数无效");
        return -1;
    }
    int ret = model->backend->npu_infer_wait(model, timeout_ms);
    if (ret == 0) {
        model->async_in_flight = 0;
    }
    return ret;
}

int gpu_npu_get_model_info(NpuModel* model, NpuModelInfo* info) {
    if (!model || !model->backend || !info) {
        LOG_ERROR("gpu_npu_get_model_info: 参数无效");
        return -1;
    }
    return model->backend->npu_get_model_info(model, info);
}

int gpu_npu_get_device_count(GpuContext* context) {
    if (!context) return -1;
    NpuBackendInterface* iface = npu_get_backend_for_context(context);
    if (!iface) return -1;
    return iface->npu_get_device_count(context);
}

const char* gpu_npu_get_backend_name(GpuContext* context) {
    if (!context) return NULL;
    NpuBackendInterface* iface = npu_get_backend_for_context(context);
    if (!iface) return NULL;
    return iface->npu_get_backend_name(context);
}
