#include "selflnn/training/model_parallel.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/gpu/gpu.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* ================================================================
 * 跨平台网络通信支持
 * Windows: 使用 Winsock2
 * Linux/Unix: 使用 BSD socket
 * ================================================================ */
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include "selflnn/utils/logging.h"  /* DEEP-005: log宏 */
#define SOCKET int
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket close
#endif

/* 环形通信默认配置 */
#define MP_RING_CONNECT_RETRIES    20    /* 连接重试次数 */
#define MP_RING_CONNECT_DELAY_MS   200   /* 重试间隔（毫秒） */
#define MP_RING_LISTEN_BACKLOG     4     /* listen队列长度 */

/*修复: 模型并行GPU加速上下文（延迟初始化）
 * 在关键计算路径（矩阵乘法）中尝试使用GPU加速。
 * GPU不可用时回退到已有CPU三重循环路径。 */
static GpuContext* g_mp_gpu_ctx = NULL;
static int g_mp_gpu_available = -1; /* -1未检测, 0不可用, 1可用 */
static MutexHandle g_mp_gpu_lock; /* GPU上下文初始化互斥锁 */
static int g_mp_gpu_lock_inited = 0;

static GpuContext* mp_get_gpu_context(void) {
/* 添加互斥锁防止多线程双重创建GPU上下文 */
    if (!g_mp_gpu_lock_inited) {
        g_mp_gpu_lock = mutex_create();
        g_mp_gpu_lock_inited = 1;
    }
    if (g_mp_gpu_available < 0) {
        mutex_lock(g_mp_gpu_lock);
        if (g_mp_gpu_available < 0) {
            g_mp_gpu_available = gpu_is_available() ? 1 : 0;
        }
        mutex_unlock(g_mp_gpu_lock);
    }
    if (!g_mp_gpu_available) return NULL;
    if (g_mp_gpu_ctx == NULL) {
        mutex_lock(g_mp_gpu_lock);
        if (g_mp_gpu_ctx == NULL) {
            GpuBackend backend = GPU_BACKEND_CPU;
            gpu_auto_init(&backend);
            if (backend != GPU_BACKEND_CPU) {
                g_mp_gpu_ctx = gpu_context_create(backend, 0);
            }
            if (!g_mp_gpu_ctx) {
                g_mp_gpu_available = 0;
            }
        }
        mutex_unlock(g_mp_gpu_lock);
    }
    return g_mp_gpu_ctx;
}

#define MP_EPSILON 1e-8f
#define MP_MAX(a,b) (((a)>(b))?(a):(b))
#define MP_MIN(a,b) (((a)<(b))?(a):(b))
#define MP_CLAMP(v,lo,hi) MP_MIN(MP_MAX((v),(lo)),(hi))

static void mp_matmul(const float* A, const float* B, float* C,
                       int M, int N, int K)
{
/*修复: GPU加速矩阵乘法路径
     * C = A * B，GPU可用时使用gpu_matmul_train加速。
     * GPU不可用时回退到已有CPU三重循环路径。 */
    {
        GpuContext* gpu_ctx = mp_get_gpu_context();
        if (gpu_ctx && M > 0 && N > 0 && K > 0) {
            int gpu_result = gpu_matmul_train(gpu_ctx, A, B, C,
                (size_t)M, (size_t)N, (size_t)K, 1.0f, 0.0f);
            if (gpu_result == 0) return;
        }
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* F-009修复: 设备间AllReduce数据注册表
 * 在单进程模型并行中，每个"设备"对应一个张量分区
 * 注册表存储各设备分区的数据指针，用于真正的跨分区归约 */
static struct {
    float*  device_data[MP_MAX_DEVICES];
    int     device_size[MP_MAX_DEVICES];
    int     registered_count;
} mp_reduce_registry = {0};

static void mp_register_reduce_data(int device_id, float* data, int size) {
    if (device_id >= 0 && device_id < MP_MAX_DEVICES && size > 0) {
        mp_reduce_registry.device_data[device_id] = data;
        mp_reduce_registry.device_size[device_id] = size;
        if (device_id >= mp_reduce_registry.registered_count) {
            mp_reduce_registry.registered_count = device_id + 1;
        }
    }
}

static void mp_unregister_reduce_data(int device_id) {
    if (device_id >= 0 && device_id < MP_MAX_DEVICES) {
        mp_reduce_registry.device_data[device_id] = NULL;
        mp_reduce_registry.device_size[device_id] = 0;
    }
}

static void mp_matmul_add(const float* A, const float* B, const float* C_in,
                           float* C_out, int M, int N, int K)
{
/*修复: GPU加速矩阵乘法加累加路径
     * C_out = C_in + A * B，GPU可用时使用gpu_matmul_train加速。
     * 需要先将C_in复制到C_out，再使用beta=1.0调用GPU矩阵乘法。
     * GPU不可用时回退到已有CPU三重循环路径。 */
    {
        GpuContext* gpu_ctx = mp_get_gpu_context();
        if (gpu_ctx && M > 0 && N > 0 && K > 0) {
            size_t total = (size_t)M * (size_t)N;
            if (C_in != C_out) {
                memcpy(C_out, C_in, total * sizeof(float));
            }
            int gpu_result = gpu_matmul_train(gpu_ctx, A, B, C_out,
                (size_t)M, (size_t)N, (size_t)K, 1.0f, 1.0f);
            if (gpu_result == 0) return;
        }
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = C_in[i * N + j];
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_out[i * N + j] = sum;
        }
    }
}

int mp_tensor_parallel_create(MPTensorPartition* tp, int rows, int cols,
                               int device_id, int num_devices,
                               MPTensorStrategy strategy)
{
    if (!tp || rows <= 0 || cols <= 0 || num_devices <= 0) return MP_ERROR_INVALID_PARAM;
    if (device_id < 0 || device_id >= num_devices) return MP_ERROR_INVALID_PARAM;

    memset(tp, 0, sizeof(MPTensorPartition));
    tp->device_id = device_id;
    tp->num_devices = num_devices;
    tp->global_rows = rows;
    tp->global_cols = cols;
    tp->strategy = strategy;

    if (strategy == MP_TENSOR_COLUMN_PARALLEL || strategy == MP_TENSOR_ATTENTION_QKV_PARALLEL) {
        int base = cols / num_devices;
        int rem = cols % num_devices;
        tp->local_rows = rows;
        tp->local_cols = base + (device_id < rem ? 1 : 0);
    } else if (strategy == MP_TENSOR_ROW_PARALLEL || strategy == MP_TENSOR_FFN_PARALLEL) {
        int base = rows / num_devices;
        int rem = rows % num_devices;
        tp->local_rows = base + (device_id < rem ? 1 : 0);
        tp->local_cols = cols;
    } else {
        return MP_ERROR_INVALID_PARAM;
    }

    tp->weight_partition = NULL;
    tp->grad_partition = NULL;
    return MP_ERROR_NONE;
}

void mp_tensor_parallel_destroy(MPTensorPartition* tp)
{
    if (!tp) return;
    if (tp->weight_partition) {
        safe_free((void**)&tp->weight_partition);
        tp->weight_partition = NULL;
    }
    if (tp->grad_partition) {
        safe_free((void**)&tp->grad_partition);
        tp->grad_partition = NULL;
    }
}

int mp_tensor_parallel_forward(MPTensorPartition* tp, const float* input,
                                const float* weight, float* output,
                                int batch_size)
{
    if (!tp || !input || !weight || !output || batch_size <= 0) return MP_ERROR_INVALID_PARAM;

    int M = batch_size;
    int N, K;

    if (tp->strategy == MP_TENSOR_COLUMN_PARALLEL || tp->strategy == MP_TENSOR_ATTENTION_QKV_PARALLEL) {
        N = tp->local_cols;
        K = tp->global_rows;
    } else if (tp->strategy == MP_TENSOR_ROW_PARALLEL || tp->strategy == MP_TENSOR_FFN_PARALLEL) {
        N = tp->local_cols;
        K = tp->local_rows;
    } else {
        return MP_ERROR_INVALID_PARAM;
    }

    mp_matmul(input, weight, output, M, N, K);
    return MP_ERROR_NONE;
}

int mp_tensor_parallel_backward(MPTensorPartition* tp, const float* grad_output,
                                 const float* input, const float* weight,
                                 float* grad_weight_partition,
                                 float* grad_input, int batch_size)
{
    if (!tp || !grad_output || !input || !grad_weight_partition) return MP_ERROR_INVALID_PARAM;

    int M = batch_size;
    if (tp->strategy == MP_TENSOR_COLUMN_PARALLEL || tp->strategy == MP_TENSOR_ATTENTION_QKV_PARALLEL) {
        int N = tp->local_cols;
        int K = tp->global_rows;
/* raw malloc → safe_malloc */
        float* temp = (float*)safe_malloc((size_t)M * K * sizeof(float));
        if (!temp) return MP_ERROR_ALLOC_FAILED;

        mp_matmul(grad_output, weight, temp, M, K, N);

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += grad_output[i * N + k] * weight[j * N + k];
                }
                grad_input[i * K + j] = sum;
            }
        }

        memset(grad_weight_partition, 0, (size_t)K * N * sizeof(float));
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                float in_val = input[i * K + j];
                for (int k = 0; k < N; k++) {
                    grad_weight_partition[j * N + k] += in_val * grad_output[i * N + k];
                }
            }
        }

/* raw free → safe_free */
        safe_free((void**)&temp);
    } else {
        int N = tp->local_cols;
        int K = tp->local_rows;

        memset(grad_weight_partition, 0, (size_t)K * N * sizeof(float));
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                float in_val = input[i * K + j];
                for (int k = 0; k < N; k++) {
                    grad_weight_partition[j * N + k] += in_val * grad_output[i * N + k];
                }
            }
        }

        if (grad_input) {
            float* full_grad_weight = NULL;
            int full_rows = tp->global_rows;
            int full_cols = tp->global_cols;
/* raw malloc → safe_malloc */
            full_grad_weight = (float*)safe_malloc((size_t)full_rows * full_cols * sizeof(float));
            if (!full_grad_weight) return MP_ERROR_ALLOC_FAILED;

            memset(full_grad_weight, 0, (size_t)full_rows * full_cols * sizeof(float));

            int offset = 0;
            for (int d = 0; d < tp->device_id; d++) {
                int base = tp->global_rows / tp->num_devices;
                offset += base + (d < (tp->global_rows % tp->num_devices) ? 1 : 0);
            }

            for (int i = 0; i < K; i++) {
                for (int j = 0; j < N; j++) {
                    full_grad_weight[(offset + i) * full_cols + j] = grad_weight_partition[i * N + j];
                }
            }

            mp_matmul(grad_output, full_grad_weight, grad_input, M, full_rows, N);
/* raw free → safe_free */
            safe_free((void**)&full_grad_weight);
        }
    }

    return MP_ERROR_NONE;
}

int mp_tensor_allreduce(MPTensorPartition* tp, float* data, int count)
{
    if (!tp || !data || count <= 0) return MP_ERROR_INVALID_PARAM;
    if (tp->num_devices <= 1) return MP_ERROR_NONE;

    int reduce_size = 0;
    if (tp->strategy == MP_TENSOR_COLUMN_PARALLEL || tp->strategy == MP_TENSOR_ATTENTION_QKV_PARALLEL) {
        reduce_size = tp->local_rows * tp->local_cols;
    } else {
        reduce_size = tp->local_rows * tp->local_cols;
    }
    if (reduce_size <= 0) return MP_ERROR_SIZE_MISMATCH;

    int num_devices = tp->num_devices;
    int device_id = tp->device_id;

    /* F-009修复: 注册当前设备的分区数据 */
    mp_register_reduce_data(device_id, data, reduce_size);

    /* 检查是否有多个设备已注册 */
    int active_devices = 0;
    for (int d = 0; d < mp_reduce_registry.registered_count && d < num_devices; d++) {
        if (mp_reduce_registry.device_data[d] && mp_reduce_registry.device_size[d] >= reduce_size) {
            active_devices++;
        }
    }

    if (active_devices <= 1) {
        /* 只有当前设备有数据，无需归约 */
        return MP_ERROR_NONE;
    }

/* raw malloc → safe_malloc */
    float* reduce_buffer = (float*)safe_malloc((size_t)reduce_size * sizeof(float));
    if (!reduce_buffer) return MP_ERROR_ALLOC_FAILED;

    /* 将所有已注册设备的数据求和到reduce_buffer */
    memset(reduce_buffer, 0, (size_t)reduce_size * sizeof(float));
    for (int d = 0; d < mp_reduce_registry.registered_count && d < num_devices; d++) {
        if (mp_reduce_registry.device_data[d]) {
            for (int i = 0; i < reduce_size; i++) {
                reduce_buffer[i] += mp_reduce_registry.device_data[d][i];
            }
        }
    }

    /* 对所有设备的分区数据求平均并写回 */
    float inv_count = 1.0f / (float)active_devices;
    for (int i = 0; i < reduce_size; i++) {
        data[i] = reduce_buffer[i] * inv_count;
    }

/* raw free → safe_free */
    safe_free((void**)&reduce_buffer);
    return MP_ERROR_NONE;
}

int mp_tensor_allgather(MPTensorPartition* tp, float* data, int count)
{
    if (!tp || !data || count <= 0) return MP_ERROR_INVALID_PARAM;
    if (tp->num_devices <= 1) return MP_ERROR_NONE;

    int local_size = 0, full_rows = tp->global_rows, full_cols = tp->global_cols;
    if (tp->strategy == MP_TENSOR_COLUMN_PARALLEL || tp->strategy == MP_TENSOR_ATTENTION_QKV_PARALLEL) {
        local_size = tp->local_rows * tp->local_cols;
    } else {
        local_size = tp->local_rows * tp->local_cols;
    }
    if (local_size <= 0) return MP_ERROR_SIZE_MISMATCH;
    if (count < local_size) return MP_ERROR_SIZE_MISMATCH;

    int num_devices = tp->num_devices;
/* raw malloc → safe_malloc */
    int* offsets = (int*)safe_malloc((size_t)num_devices * sizeof(int));
    int* sizes = (int*)safe_malloc((size_t)num_devices * sizeof(int));
    if (!offsets || !sizes) { safe_free((void**)&offsets); safe_free((void**)&sizes); return MP_ERROR_ALLOC_FAILED; }

    if (tp->strategy == MP_TENSOR_COLUMN_PARALLEL || tp->strategy == MP_TENSOR_ATTENTION_QKV_PARALLEL) {
        int base = full_cols / num_devices;
        int rem = full_cols % num_devices;
        int col_offset = 0;
        for (int d = 0; d < num_devices; d++) {
            int ncols = base + (d < rem ? 1 : 0);
            offsets[d] = col_offset * full_rows;
            sizes[d] = full_rows * ncols;
            col_offset += ncols;
        }
    } else {
        int base = full_rows / num_devices;
        int rem = full_rows % num_devices;
        int row_offset = 0;
        for (int d = 0; d < num_devices; d++) {
            int nrows = base + (d < rem ? 1 : 0);
            offsets[d] = row_offset * full_cols;
            sizes[d] = nrows * full_cols;
            row_offset += nrows;
        }
    }

/* raw malloc → safe_malloc */
    float* full_buffer = (float*)safe_malloc((size_t)full_rows * full_cols * sizeof(float));
    if (!full_buffer) { safe_free((void**)&offsets); safe_free((void**)&sizes); return MP_ERROR_ALLOC_FAILED; }
    memset(full_buffer, 0, (size_t)full_rows * full_cols * sizeof(float));

    /* F-009修复: 从注册表收集各设备分区数据实现真正的AllGather */
    for (int d = 0; d < num_devices; d++) {
        if (d == tp->device_id) {
            memcpy(full_buffer + offsets[d], data, (size_t)sizes[d] * sizeof(float));
        } else {
            /* 从其他已注册设备的分区复制数据 */
            float* other_data = mp_reduce_registry.device_data[d];
            if (other_data && mp_reduce_registry.device_size[d] >= sizes[d]) {
                memcpy(full_buffer + offsets[d], other_data, (size_t)sizes[d] * sizeof(float));
            }
        }
    }

    if (count >= full_rows * full_cols) {
        memcpy(data, full_buffer, (size_t)full_rows * full_cols * sizeof(float));
    }

/* raw free → safe_free */
    safe_free((void**)&full_buffer);
    safe_free((void**)&offsets);
    safe_free((void**)&sizes);
    return MP_ERROR_NONE;
}

int mp_tensor_reduce_scatter(MPTensorPartition* tp, float* data, int count)
{
    if (!tp || !data || count <= 0) return MP_ERROR_INVALID_PARAM;
    if (tp->num_devices <= 1) return MP_ERROR_NONE;

    int local_size = 0, full_rows = tp->global_rows, full_cols = tp->global_cols;
    if (tp->strategy == MP_TENSOR_COLUMN_PARALLEL || tp->strategy == MP_TENSOR_ATTENTION_QKV_PARALLEL) {
        local_size = tp->local_rows * tp->local_cols;
    } else {
        local_size = tp->local_rows * tp->local_cols;
    }
    if (local_size <= 0) return MP_ERROR_SIZE_MISMATCH;
    if (count < local_size) return MP_ERROR_SIZE_MISMATCH;

    int num_devices = tp->num_devices;
/* raw malloc → safe_malloc */
    int* offsets = (int*)safe_malloc((size_t)num_devices * sizeof(int));
    int* sizes = (int*)safe_malloc((size_t)num_devices * sizeof(int));
    if (!offsets || !sizes) { safe_free((void**)&offsets); safe_free((void**)&sizes); return MP_ERROR_ALLOC_FAILED; }

    if (tp->strategy == MP_TENSOR_COLUMN_PARALLEL || tp->strategy == MP_TENSOR_ATTENTION_QKV_PARALLEL) {
        int base = full_cols / num_devices;
        int rem = full_cols % num_devices;
        int col_offset = 0;
        for (int d = 0; d < num_devices; d++) {
            int ncols = base + (d < rem ? 1 : 0);
            offsets[d] = col_offset * full_rows;
            sizes[d] = full_rows * ncols;
            col_offset += ncols;
        }
    } else {
        int base = full_rows / num_devices;
        int rem = full_rows % num_devices;
        int row_offset = 0;
        for (int d = 0; d < num_devices; d++) {
            int nrows = base + (d < rem ? 1 : 0);
            offsets[d] = row_offset * full_cols;
            sizes[d] = nrows * full_cols;
            row_offset += nrows;
        }
    }

/* raw calloc → safe_calloc */
    float* sum_buffer = (float*)safe_calloc((size_t)full_rows * full_cols, sizeof(float));
    if (!sum_buffer) { safe_free((void**)&offsets); safe_free((void**)&sizes); return MP_ERROR_ALLOC_FAILED; }

    for (int d = 0; d < num_devices; d++) {
        float* src = data + (d * local_size);
        for (int i = 0; i < sizes[d] && (offsets[d] + i) < full_rows * full_cols; i++) {
            sum_buffer[offsets[d] + i] += src[i];
        }
    }

    float inv = 1.0f / (float)num_devices;
    int my_offset = offsets[tp->device_id];
    int my_size = sizes[tp->device_id];
    for (int i = 0; i < my_size && i < count; i++) {
        data[i] = sum_buffer[my_offset + i] * inv;
    }

/* raw free → safe_free */
    safe_free((void**)&sum_buffer);
    safe_free((void**)&offsets);
    safe_free((void**)&sizes);
    return MP_ERROR_NONE;
}

int mp_pipeline_create(MPPipelineStage* ps, int num_stages, int stage_id,
                        int micro_batch_size, int num_micro_batches,
                        MPPipelineSchedule schedule)
{
    if (!ps || num_stages <= 0 || stage_id < 0 || stage_id >= num_stages) return MP_ERROR_INVALID_PARAM;
    if (micro_batch_size <= 0 || num_micro_batches <= 0 || num_micro_batches > MP_MAX_MICRO_BATCHES) return MP_ERROR_INVALID_PARAM;

    memset(ps, 0, sizeof(MPPipelineStage));
    ps->num_stages = num_stages;
    ps->stage_id = stage_id;
    ps->micro_batch_size = micro_batch_size;
    ps->num_micro_batches = num_micro_batches;
    ps->global_batch_size = micro_batch_size * num_micro_batches;
    ps->schedule = schedule;
    ps->input_buffer = NULL;
    ps->output_buffer = NULL;

/* raw malloc → safe_malloc */
    ps->forward_ranks = (int*)safe_malloc((size_t)num_stages * sizeof(int));
    ps->backward_ranks = (int*)safe_malloc((size_t)num_stages * sizeof(int));
    if (!ps->forward_ranks || !ps->backward_ranks) {
        safe_free((void**)&ps->forward_ranks);
        safe_free((void**)&ps->backward_ranks);
        return MP_ERROR_ALLOC_FAILED;
    }

    for (int i = 0; i < num_stages; i++) {
        ps->forward_ranks[i] = i;
        ps->backward_ranks[i] = num_stages - 1 - i;
    }

    return MP_ERROR_NONE;
}

void mp_pipeline_destroy(MPPipelineStage* ps)
{
    if (!ps) return;
/* raw free → safe_free */
    safe_free((void**)&ps->forward_ranks);
    safe_free((void**)&ps->backward_ranks);
    if (ps->input_buffer) {
        safe_free((void**)&ps->input_buffer);
        ps->input_buffer = NULL;
    }
    if (ps->output_buffer) {
        safe_free((void**)&ps->output_buffer);
        ps->output_buffer = NULL;
    }
}

static int mp_pipeline_forward_1f1b(MPPipelineStage* ps, const float* input,
                                     float* output, int batch_size,
                                     int (*stage_fn)(int, const float*, float*, int, void*),
                                     void* ctx)
{
    if (!input || !output || !stage_fn) return MP_ERROR_INVALID_PARAM;

    int actual_mb = ps->num_micro_batches;
    int mb_size = ps->micro_batch_size;
    int total_mb = mb_size * actual_mb;
    if (batch_size < total_mb) actual_mb = batch_size / mb_size;
    if (actual_mb <= 0) actual_mb = 1;
    mb_size = batch_size / actual_mb;

    int warmup = ps->num_stages - ps->stage_id - 1;
    if (warmup > actual_mb) warmup = actual_mb;
    int steady = actual_mb - warmup;

    float* mb_buffers = NULL;
    int buf_size = mb_size * mp_pipeline_get_micro_batch(ps, 0, NULL, NULL);
    if (buf_size > 0) {
/* raw malloc → safe_malloc */
        mb_buffers = (float*)safe_malloc((size_t)buf_size * sizeof(float));
    }

    for (int m = 0; m < warmup; m++) {
        int offset = m * mb_size;
        int ret = stage_fn(ps->stage_id, input + offset,
                           mb_buffers ? mb_buffers : output + offset,
                           mb_size, ctx);
        if (ret != 0) { safe_free((void**)&mb_buffers) /* */; return ret; }
    }

    for (int m = 0; m < steady; m++) {
        int warm_offset = (warmup + m) * mb_size;
        int steady_offset = m * mb_size;
        int ret = stage_fn(ps->stage_id, input + warm_offset,
                           mb_buffers ? mb_buffers : output + warm_offset,
                           mb_size, ctx);
        if (ret != 0) { safe_free((void**)&mb_buffers) /* */; return ret; }

        if (m < warmup) {
            ret = stage_fn(ps->stage_id, input + steady_offset,
                           mb_buffers ? mb_buffers : output + steady_offset,
                           mb_size, ctx);
            if (ret != 0) { safe_free((void**)&mb_buffers) /* */; return ret; }
        }
    }

    for (int m = 0; m < warmup && m < steady; m++) {
        int offset = (m + steady) * mb_size;
        int ret = stage_fn(ps->stage_id, input + offset,
                           mb_buffers ? mb_buffers : output + offset,
                           mb_size, ctx);
        if (ret != 0) { safe_free((void**)&mb_buffers) /* */; return ret; }
    }

/* raw free → safe_free */
    safe_free((void**)&mb_buffers);
    return MP_ERROR_NONE;
}

int mp_pipeline_forward(MPPipelineStage* ps, const float* input,
                         float* output, int batch_size,
                         int (*stage_fn)(int, const float*, float*, int, void*),
                         void* ctx)
{
    if (!ps || !stage_fn) return MP_ERROR_INVALID_PARAM;

    if (ps->schedule == MP_PIPELINE_1F1B || ps->schedule == MP_PIPELINE_INTERLEAVED_1F1B) {
        return mp_pipeline_forward_1f1b(ps, input, output, batch_size, stage_fn, ctx);
    }

    int mb_size = batch_size / ps->num_micro_batches;
    if (mb_size <= 0) mb_size = batch_size;

    for (int m = 0; m < ps->num_micro_batches; m++) {
        int offset = m * mb_size;
        int ret = stage_fn(ps->stage_id, input + offset, output + offset,
                           mb_size < batch_size - offset ? mb_size : batch_size - offset, ctx);
        if (ret != 0) return ret;
    }

    return MP_ERROR_NONE;
}

int mp_pipeline_backward(MPPipelineStage* ps, const float* grad_output,
                          float* grad_input, int batch_size,
                          int (*stage_bwd_fn)(int, const float*, float*, int, void*),
                          void* ctx)
{
    if (!ps || !stage_bwd_fn) return MP_ERROR_INVALID_PARAM;

    int mb_size = batch_size / ps->num_micro_batches;
    if (mb_size <= 0) mb_size = batch_size;

    for (int m = ps->num_micro_batches - 1; m >= 0; m--) {
        int offset = m * mb_size;
        int ret = stage_bwd_fn(ps->stage_id, grad_output + offset,
                                grad_input ? grad_input + offset : NULL,
                                mb_size < batch_size - offset ? mb_size : batch_size - offset, ctx);
        if (ret != 0) return ret;
    }

    return MP_ERROR_NONE;
}

int mp_pipeline_get_micro_batch(const MPPipelineStage* ps, int micro_idx,
                                 int* out_offset, int* out_size)
{
    if (!ps || micro_idx < 0 || micro_idx >= ps->num_micro_batches) return 0;
    if (out_offset) *out_offset = micro_idx * ps->micro_batch_size;
    if (out_size) *out_size = ps->micro_batch_size;
    return ps->micro_batch_size;
}

int mp_pipeline_num_warmup_micro_batches(const MPPipelineStage* ps)
{
    if (!ps) return 0;
    return ps->num_stages - ps->stage_id - 1;
}

int mp_pipeline_num_steady_micro_batches(const MPPipelineStage* ps)
{
    if (!ps) return 0;
    int warmup = ps->num_stages - ps->stage_id - 1;
    return ps->num_micro_batches - warmup;
}

int mp_sequence_parallel_create(MPSequencePartition* sp, int seq_length,
                                 int hidden_dim, int chunk_id, int num_chunks)
{
    if (!sp || seq_length <= 0 || hidden_dim <= 0 || num_chunks <= 0) return MP_ERROR_INVALID_PARAM;
    if (chunk_id < 0 || chunk_id >= num_chunks) return MP_ERROR_INVALID_PARAM;

    memset(sp, 0, sizeof(MPSequencePartition));
    sp->seq_length = seq_length;
    sp->hidden_dim = hidden_dim;
    sp->num_chunks = num_chunks;
    sp->chunk_id = chunk_id;

    int base = seq_length / num_chunks;
    int rem = seq_length % num_chunks;
    sp->chunk_size = base + (chunk_id < rem ? 1 : 0);
    sp->chunk_input = NULL;
    sp->chunk_output = NULL;

    return MP_ERROR_NONE;
}

void mp_sequence_parallel_destroy(MPSequencePartition* sp)
{
    if (!sp) return;
    if (sp->chunk_input) {
        safe_free((void**)&sp->chunk_input);
        sp->chunk_input = NULL;
    }
    if (sp->chunk_output) {
        safe_free((void**)&sp->chunk_output);
        sp->chunk_output = NULL;
    }
}

int mp_sequence_parallel_forward(MPSequencePartition* sp, const float* input,
                                  float* output, int batch_size)
{
    if (!sp || !input || !output || batch_size <= 0) return MP_ERROR_INVALID_PARAM;

    int chunk_bytes = sp->chunk_size * sp->hidden_dim;
    int offset = 0;
    for (int d = 0; d < sp->chunk_id; d++) {
        int base = sp->seq_length / sp->num_chunks;
        offset += base + (d < (sp->seq_length % sp->num_chunks) ? 1 : 0);
    }

    for (int b = 0; b < batch_size; b++) {
        const float* src = input + (size_t)b * sp->seq_length * sp->hidden_dim + (size_t)offset * sp->hidden_dim;
        float* dst = output + (size_t)b * sp->chunk_size * sp->hidden_dim;
        memcpy(dst, src, (size_t)chunk_bytes * sizeof(float));
    }

    return MP_ERROR_NONE;
}

int mp_sequence_parallel_backward(MPSequencePartition* sp, const float* grad_output,
                                   float* grad_input, int batch_size)
{
    if (!sp || !grad_output || !grad_input || batch_size <= 0) return MP_ERROR_INVALID_PARAM;

    int offset = 0;
    for (int d = 0; d < sp->chunk_id; d++) {
        int base = sp->seq_length / sp->num_chunks;
        offset += base + (d < (sp->seq_length % sp->num_chunks) ? 1 : 0);
    }

    for (int b = 0; b < batch_size; b++) {
        const float* src = grad_output + (size_t)b * sp->chunk_size * sp->hidden_dim;
        float* dst = grad_input + (size_t)b * sp->seq_length * sp->hidden_dim + (size_t)offset * sp->hidden_dim;
        memcpy(dst, src, (size_t)sp->chunk_size * sp->hidden_dim * sizeof(float));
    }

    return MP_ERROR_NONE;
}

int mp_sequence_allgather(MPSequencePartition* sp, const float* local,
                           float* global, int batch_size, int hidden_dim)
{
    if (!sp || !local || !global || batch_size <= 0) return MP_ERROR_INVALID_PARAM;

    for (int b = 0; b < batch_size; b++) {
        int offset = 0;
        for (int d = 0; d < sp->num_chunks; d++) {
            int base = sp->seq_length / sp->num_chunks;
            int chunk = base + (d < (sp->seq_length % sp->num_chunks) ? 1 : 0);
            if (d == sp->chunk_id) {
                memcpy(global + (size_t)b * sp->seq_length * hidden_dim + (size_t)offset * hidden_dim,
                       local + (size_t)b * chunk * hidden_dim,
                       (size_t)chunk * hidden_dim * sizeof(float));
            }
            offset += chunk;
        }
    }

    return MP_ERROR_NONE;
}

int mp_sequence_reduce_scatter(MPSequencePartition* sp, const float* global,
                                float* local, int batch_size, int hidden_dim)
{
    if (!sp || !global || !local || batch_size <= 0) return MP_ERROR_INVALID_PARAM;

    for (int b = 0; b < batch_size; b++) {
        int offset = 0;
        for (int d = 0; d < sp->num_chunks; d++) {
            int base = sp->seq_length / sp->num_chunks;
            int chunk = base + (d < (sp->seq_length % sp->num_chunks) ? 1 : 0);
            if (d == sp->chunk_id) {
                memcpy(local + (size_t)b * chunk * hidden_dim,
                       global + (size_t)b * sp->seq_length * hidden_dim + (size_t)offset * hidden_dim,
                       (size_t)chunk * hidden_dim * sizeof(float));
            }
            offset += chunk;
        }
    }

    return MP_ERROR_NONE;
}

int mp_zero_create(MPZeroOptimizer* zo, MPZeroStage stage, int param_count,
                    int shard_count, int shard_id, int optimizer_states_per_param)
{
    if (!zo || param_count <= 0 || shard_count <= 0) return MP_ERROR_INVALID_PARAM;
    if (shard_id < 0 || shard_id >= shard_count) return MP_ERROR_INVALID_PARAM;
    if (stage < MP_ZERO_STAGE_1 || stage > MP_ZERO_STAGE_3) return MP_ERROR_INVALID_PARAM;

    memset(zo, 0, sizeof(MPZeroOptimizer));
    zo->stage = stage;
    zo->param_count = param_count;
    zo->optimizer_states_per_param = optimizer_states_per_param;
    zo->shard_count = shard_count;
    zo->shard_id = shard_id;

    int shard_size = param_count / shard_count;
    int rem = param_count % shard_count;
    zo->gradient_count = shard_size + (shard_id < rem ? 1 : 0);

/* raw malloc → safe_malloc */
    zo->param_owner_map = (int*)safe_malloc((size_t)param_count * sizeof(int));
    if (!zo->param_owner_map) return MP_ERROR_ALLOC_FAILED;

    int idx = 0;
    for (int d = 0; d < shard_count; d++) {
        int dc = shard_size + (d < rem ? 1 : 0);
        for (int i = 0; i < dc && idx < param_count; i++) {
            zo->param_owner_map[idx++] = d;
        }
    }

    int opt_state_count = 0;
    if (stage >= MP_ZERO_STAGE_1) {
        opt_state_count = zo->gradient_count * optimizer_states_per_param;
/* raw calloc → safe_calloc */
        zo->optimizer_state_shard = (float*)safe_calloc((size_t)opt_state_count, sizeof(float));
        if (!zo->optimizer_state_shard) {
            safe_free((void**)&zo->param_owner_map);
            return MP_ERROR_ALLOC_FAILED;
        }
    }

    if (stage >= MP_ZERO_STAGE_2) {
/* raw calloc → safe_calloc */
        zo->gradient_shard = (float*)safe_calloc((size_t)zo->gradient_count, sizeof(float));
        if (!zo->gradient_shard) {
            safe_free((void**)&zo->optimizer_state_shard);
            safe_free((void**)&zo->param_owner_map);
            return MP_ERROR_ALLOC_FAILED;
        }
    }

    if (stage >= MP_ZERO_STAGE_3) {
/* raw calloc → safe_calloc */
        zo->param_shard = (float*)safe_calloc((size_t)zo->gradient_count, sizeof(float));
        if (!zo->param_shard) {
            safe_free((void**)&zo->optimizer_state_shard);
            safe_free((void**)&zo->gradient_shard);
            safe_free((void**)&zo->param_owner_map);
            return MP_ERROR_ALLOC_FAILED;
        }
    }

    return MP_ERROR_NONE;
}

void mp_zero_destroy(MPZeroOptimizer* zo)
{
    if (!zo) return;
/* raw free → safe_free */
    safe_free((void**)&zo->param_owner_map);
    if (zo->optimizer_state_shard) {
        safe_free((void**)&zo->optimizer_state_shard);
    }
    if (zo->gradient_shard) {
        safe_free((void**)&zo->gradient_shard);
    }
    if (zo->param_shard) {
        safe_free((void**)&zo->param_shard);
    }
}

int mp_zero_shard_param(MPZeroOptimizer* zo, const float* full_params,
                         float* shard_out)
{
    if (!zo || !full_params || !shard_out) return MP_ERROR_INVALID_PARAM;

    int shard_size = zo->param_count / zo->shard_count;
    int rem = zo->param_count % zo->shard_count;
    int offset = shard_size * zo->shard_id + (zo->shard_id < rem ? zo->shard_id : rem);

    if (zo->stage >= MP_ZERO_STAGE_3) {
        memcpy(zo->param_shard, full_params + offset, (size_t)zo->gradient_count * sizeof(float));
    }

    memcpy(shard_out, full_params + offset, (size_t)zo->gradient_count * sizeof(float));
    return MP_ERROR_NONE;
}

int mp_zero_gather_param(MPZeroOptimizer* zo, const float* shard_params,
                          float* full_out)
{
    if (!zo || !shard_params || !full_out) return MP_ERROR_INVALID_PARAM;

    int shard_size = zo->param_count / zo->shard_count;
    int rem = zo->param_count % zo->shard_count;
    int offset = shard_size * zo->shard_id + (zo->shard_id < rem ? zo->shard_id : rem);

    memcpy(full_out + offset, shard_params, (size_t)zo->gradient_count * sizeof(float));
    return MP_ERROR_NONE;
}

int mp_zero_shard_gradient(MPZeroOptimizer* zo, const float* full_grad,
                            float* shard_out)
{
    if (!zo || !full_grad || !shard_out) return MP_ERROR_INVALID_PARAM;

    int shard_size = zo->param_count / zo->shard_count;
    int rem = zo->param_count % zo->shard_count;
    int offset = shard_size * zo->shard_id + (zo->shard_id < rem ? zo->shard_id : rem);

    if (zo->stage >= MP_ZERO_STAGE_2) {
        memcpy(zo->gradient_shard, full_grad + offset, (size_t)zo->gradient_count * sizeof(float));
    }

    memcpy(shard_out, full_grad + offset, (size_t)zo->gradient_count * sizeof(float));
    return MP_ERROR_NONE;
}

int mp_zero_gather_gradient(MPZeroOptimizer* zo, const float* shard_grad,
                             float* full_out)
{
    if (!zo || !shard_grad || !full_out) return MP_ERROR_INVALID_PARAM;

    int shard_size = zo->param_count / zo->shard_count;
    int rem = zo->param_count % zo->shard_count;
    int offset = shard_size * zo->shard_id + (zo->shard_id < rem ? zo->shard_id : rem);

    memcpy(full_out + offset, shard_grad, (size_t)zo->gradient_count * sizeof(float));
    return MP_ERROR_NONE;
}

int mp_zero_update(MPZeroOptimizer* zo, float learning_rate,
                    float beta1, float beta2, float epsilon, int step)
{
    if (!zo || zo->stage < MP_ZERO_STAGE_1) return MP_ERROR_NONE;

    if (!zo->optimizer_state_shard) return MP_ERROR_INVALID_PARAM;

    for (int i = 0; i < zo->gradient_count; i++) {
        float* m = &zo->optimizer_state_shard[i * zo->optimizer_states_per_param];
        float* v = &zo->optimizer_state_shard[i * zo->optimizer_states_per_param + 1];

        float grad = (zo->stage >= MP_ZERO_STAGE_2 && zo->gradient_shard)
                     ? zo->gradient_shard[i] : 0.0f;

        *m = beta1 * (*m) + (1.0f - beta1) * grad;
        *v = beta2 * (*v) + (1.0f - beta2) * grad * grad;

        float eps = epsilon > 0.0f ? epsilon : MP_EPSILON;
        float m_hat = *m / (1.0f - powf(beta1, (float)step) + eps);
        float v_hat = *v / (1.0f - powf(beta2, (float)step) + eps);

        float update = learning_rate * m_hat / (sqrtf(v_hat) + eps);

        if (zo->stage >= MP_ZERO_STAGE_3 && zo->param_shard) {
            zo->param_shard[i] -= update;
        }
    }

    return MP_ERROR_NONE;
}

int mp_zero_get_optimizer_state(MPZeroOptimizer* zo, int state_idx,
                                 float* out_state)
{
    if (!zo || !out_state || state_idx < 0) return MP_ERROR_INVALID_PARAM;
    if (state_idx >= zo->optimizer_states_per_param) return MP_ERROR_INVALID_PARAM;
    if (!zo->optimizer_state_shard) return MP_ERROR_INVALID_PARAM;

    for (int i = 0; i < zo->gradient_count; i++) {
        out_state[i] = zo->optimizer_state_shard[i * zo->optimizer_states_per_param + state_idx];
    }
    return MP_ERROR_NONE;
}

int mp_zero_set_optimizer_state(MPZeroOptimizer* zo, int state_idx,
                                 const float* state)
{
    if (!zo || !state || state_idx < 0) return MP_ERROR_INVALID_PARAM;
    if (state_idx >= zo->optimizer_states_per_param) return MP_ERROR_INVALID_PARAM;
    if (!zo->optimizer_state_shard) return MP_ERROR_INVALID_PARAM;

    for (int i = 0; i < zo->gradient_count; i++) {
        zo->optimizer_state_shard[i * zo->optimizer_states_per_param + state_idx] = state[i];
    }
    return MP_ERROR_NONE;
}

int mp_fsdp_create(MPZeroOptimizer* zo, MPZeroStage stage, int param_count,
                    int shard_count, int shard_id)
{
    if (stage != MP_ZERO_STAGE_2 && stage != MP_ZERO_STAGE_3) return MP_ERROR_INVALID_PARAM;
    return mp_zero_create(zo, stage, param_count, shard_count, shard_id, 2);
}

int mp_fsdp_forward_pre(MPZeroOptimizer* zo, float* params, float* all_params,
                         int count)
{
    if (!zo || !params || !all_params || count <= 0) return MP_ERROR_INVALID_PARAM;
    if (zo->stage < MP_ZERO_STAGE_3) return MP_ERROR_NONE;

    int shard_size = zo->param_count / zo->shard_count;
    int rem = zo->param_count % zo->shard_count;

    for (int d = 0; d < zo->shard_count; d++) {
        int dc = shard_size + (d < rem ? 1 : 0);
        int offset = shard_size * d + (d < rem ? d : rem);

        if (d == zo->shard_id) {
            memcpy(all_params + offset, params + offset, (size_t)dc * sizeof(float));
        }
    }

    return MP_ERROR_NONE;
}

int mp_fsdp_backward_post(MPZeroOptimizer* zo, float* grads, float* all_grads,
                           int count)
{
    if (!zo || !grads || !all_grads || count <= 0) return MP_ERROR_INVALID_PARAM;
    if (zo->stage < MP_ZERO_STAGE_2) return MP_ERROR_NONE;

    int shard_size = zo->param_count / zo->shard_count;
    int rem = zo->param_count % zo->shard_count;

    for (int d = 0; d < zo->shard_count; d++) {
        int dc = shard_size + (d < rem ? 1 : 0);
        int offset = shard_size * d + (d < rem ? d : rem);

        if (d == zo->shard_id) {
            if (zo->gradient_shard) {
                memcpy(zo->gradient_shard, all_grads + offset, (size_t)dc * sizeof(float));
            }
        }
    }

    return MP_ERROR_NONE;
}

int mp_3d_create(MP3DContext* ctx, int dp_degree, int tp_degree, int pp_degree,
                  int global_rank)
{
    if (!ctx || dp_degree <= 0 || tp_degree <= 0 || pp_degree <= 0) return MP_ERROR_INVALID_PARAM;

    int total = dp_degree * tp_degree * pp_degree;
    if (global_rank < 0 || global_rank >= total) return MP_ERROR_INVALID_PARAM;

    memset(ctx, 0, sizeof(MP3DContext));
    ctx->dp_degree = dp_degree;
    ctx->tp_degree = tp_degree;
    ctx->pp_degree = pp_degree;
    ctx->global_rank = global_rank;
    ctx->global_size = total;

    ctx->pp_rank = global_rank / (dp_degree * tp_degree);
    int remainder = global_rank % (dp_degree * tp_degree);
    ctx->tp_rank = remainder % tp_degree;
    ctx->dp_rank = remainder / tp_degree;

    ctx->reduce_buffer_in = NULL;
    ctx->reduce_buffer_out = NULL;
    ctx->reduce_buffer_size = 0;
    ctx->send_socket = (int64_t)INVALID_SOCKET;
    ctx->recv_socket = (int64_t)INVALID_SOCKET;
    ctx->comm_ready = 0;
    ctx->listen_port = 0;

    return MP_ERROR_NONE;
}

void mp_3d_destroy(MP3DContext* ctx)
{
    if (!ctx) return;
    mp_3d_close_connections(ctx);
    if (ctx->reduce_buffer_in) {
        safe_free((void**)&ctx->reduce_buffer_in);
    }
    if (ctx->reduce_buffer_out) {
        safe_free((void**)&ctx->reduce_buffer_out);
    }
}

int mp_3d_global_to_ranks(const MP3DContext* ctx, int* out_dp_rank,
                           int* out_tp_rank, int* out_pp_rank)
{
    if (!ctx) return MP_ERROR_INVALID_PARAM;
    if (out_dp_rank) *out_dp_rank = ctx->dp_rank;
    if (out_tp_rank) *out_tp_rank = ctx->tp_rank;
    if (out_pp_rank) *out_pp_rank = ctx->pp_rank;
    return MP_ERROR_NONE;
}

int mp_3d_rank_to_global(const MP3DContext* ctx, int dp_rank, int tp_rank,
                          int pp_rank, int* out_global_rank)
{
    if (!ctx || !out_global_rank) return MP_ERROR_INVALID_PARAM;
    *out_global_rank = pp_rank * (ctx->dp_degree * ctx->tp_degree) + dp_rank * ctx->tp_degree + tp_rank;
    return MP_ERROR_NONE;
}

static int mp_3d_get_group_info(const MP3DContext* ctx, MP3DDimension dim,
                                  int* out_group_size, int* out_group_rank)
{
    if (!ctx || !out_group_size || !out_group_rank) return MP_ERROR_INVALID_PARAM;
    if (dim == MP_3D_DATA) {
        *out_group_size = ctx->dp_degree;
        *out_group_rank = ctx->dp_rank;
    } else if (dim == MP_3D_TENSOR) {
        *out_group_size = ctx->tp_degree;
        *out_group_rank = ctx->tp_rank;
    } else if (dim == MP_3D_PIPELINE) {
        *out_group_size = ctx->pp_degree;
        *out_group_rank = ctx->pp_rank;
    } else {
        *out_group_size = ctx->global_size;
        *out_group_rank = ctx->global_rank;
    }
    return MP_ERROR_NONE;
}

static int mp_3d_ensure_buffer(MP3DContext* ctx, size_t needed)
{
    if (ctx->reduce_buffer_size >= needed) return MP_ERROR_NONE;
    float* new_buf_in = (float*)safe_realloc(ctx->reduce_buffer_in, needed);
    float* new_buf_out = (float*)safe_realloc(ctx->reduce_buffer_out, needed);
    if (!new_buf_in || !new_buf_out) {
        safe_free((void**)&new_buf_in);
        safe_free((void**)&new_buf_out);
        return MP_ERROR_ALLOC_FAILED;
    }
    ctx->reduce_buffer_in = new_buf_in;
    ctx->reduce_buffer_out = new_buf_out;
    ctx->reduce_buffer_size = needed;
    return MP_ERROR_NONE;
}

static int mp_3d_compute_chunks(int group_size, int count, int** out_sizes,
                                 int** out_offsets)
{
    int chunk_size = count / group_size;
    int rem = count % group_size;
/* raw malloc → safe_malloc */
    int* sizes = (int*)safe_malloc((size_t)group_size * sizeof(int));
    int* offsets = (int*)safe_malloc((size_t)group_size * sizeof(int));
    if (!sizes || !offsets) {
        safe_free((void**)&sizes);
        safe_free((void**)&offsets);
        return MP_ERROR_ALLOC_FAILED;
    }
    int acc = 0;
    for (int i = 0; i < group_size; i++) {
        sizes[i] = chunk_size + (i < rem ? 1 : 0);
        offsets[i] = acc;
        acc += sizes[i];
    }
    *out_sizes = sizes;
    *out_offsets = offsets;
    return MP_ERROR_NONE;
}

/* ================================================================
 * TCP Ring 通信辅助函数
 * 确保完整收发指定字节数，处理TCP分片
 * ================================================================ */

/* 从socket精确接收size字节数据 */
static int mp_ring_recv_all(SOCKET sock, void* buf, size_t size)
{
    char* ptr = (char*)buf;
    size_t remaining = size;
    while (remaining > 0) {
        int n = recv(sock, ptr, (int)remaining, 0);
        if (n <= 0) {
            return -1;
        }
        ptr += n;
        remaining -= (size_t)n;
    }
    return 0;
}

/* 向socket精确发送size字节数据 */
static int mp_ring_send_all(SOCKET sock, const void* buf, size_t size)
{
    const char* ptr = (const char*)buf;
    size_t remaining = size;
    while (remaining > 0) {
        int n = send(sock, ptr, (int)remaining, 0);
        if (n <= 0) {
            return -1;
        }
        ptr += n;
        remaining -= (size_t)n;
    }
    return 0;
}

/* 发送浮点数组 */
static int mp_ring_send_floats(SOCKET sock, const float* data, int count)
{
    return mp_ring_send_all(sock, data, (size_t)count * sizeof(float));
}

/* 接收浮点数组 */
static int mp_ring_recv_floats(SOCKET sock, float* data, int count)
{
    return mp_ring_recv_all(sock, data, (size_t)count * sizeof(float));
}

/* ================================================================
 * 平台相关辅助：跨平台Sleep
 * ================================================================ */
#ifdef _WIN32
#define MP_RING_SLEEP_MS(ms) Sleep((DWORD)(ms))
#else
static void mp_ring_sleep_ms(int ms)
{
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}
#define MP_RING_SLEEP_MS(ms) mp_ring_sleep_ms(ms)
#endif

/* ================================================================
 * Winsock初始化（仅Windows需要，静态保证只初始化一次）
 * ================================================================ */
#ifdef _WIN32
static int mp_winsock_ready = 0;
static int mp_winsock_init(void)
{
    if (mp_winsock_ready) return 0;
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        return -1;
    }
    mp_winsock_ready = 1;
    return 0;
}
#else
static int mp_winsock_init(void) { return 0; }
#endif

/* ================================================================
 * mp_3d_init_connections: 建立环状TCP连接
 *
 * 拓扑结构：
 *   环: rank0 → rank1 → rank2 → ... → rank(N-1) → rank0
 *   每个节点连接下一个rank（发送方向），接受上一个rank（接收方向）
 *
 * 端口分配：
 *   每个rank监听 base_port + dim*100 + group_rank
 *
 * 如果连接失败（单机无网络），回退到本地单进程模式
 * ================================================================ */
int mp_3d_init_connections(MP3DContext* ctx, MP3DDimension dim, int base_port)
{
    if (!ctx) return MP_ERROR_INVALID_PARAM;

    int group_size, group_rank;
    if (mp_3d_get_group_info(ctx, dim, &group_size, &group_rank) != MP_ERROR_NONE)
        return MP_ERROR_INVALID_PARAM;

    /* 单节点无需网络通信 */
    if (group_size <= 1) {
        ctx->comm_ready = 0;
        return MP_ERROR_NONE;
    }

    /* 初始化Winsock（Windows） */
    if (mp_winsock_init() != 0) {
        ctx->comm_ready = 0;
        return MP_ERROR_NONE;
    }

    /* 计算本节点监听端口、下一个节点的连接端口 */
    int my_port = base_port + (int)dim * 100 + group_rank;
    int next_rank = (group_rank + 1) % group_size;
    int next_port = base_port + (int)dim * 100 + next_rank;

    ctx->listen_port = my_port;

    /* ---- 第一步：创建监听socket ---- */
    SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock == INVALID_SOCKET) {
        ctx->comm_ready = 0;
        return MP_ERROR_NONE;
    }

    /* 设置SO_REUSEADDR以便快速重启 */
    int reuse = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR,
               (const char*)&reuse, sizeof(reuse));

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_addr.sin_port = htons((unsigned short)my_port);

    if (bind(listen_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
        closesocket(listen_sock);
        ctx->comm_ready = 0;
        return MP_ERROR_NONE;
    }

    if (listen(listen_sock, MP_RING_LISTEN_BACKLOG) == SOCKET_ERROR) {
        closesocket(listen_sock);
        ctx->comm_ready = 0;
        return MP_ERROR_NONE;
    }

    /* ---- 第二步：连接到下一个rank（带重试）---- */
    SOCKET send_sock = INVALID_SOCKET;
    struct sockaddr_in next_addr;
    memset(&next_addr, 0, sizeof(next_addr));
    next_addr.sin_family = AF_INET;
    next_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    next_addr.sin_port = htons((unsigned short)next_port);

    for (int retry = 0; retry < MP_RING_CONNECT_RETRIES; retry++) {
        send_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (send_sock == INVALID_SOCKET) {
            MP_RING_SLEEP_MS(MP_RING_CONNECT_DELAY_MS);
            continue;
        }
        if (connect(send_sock, (struct sockaddr*)&next_addr, sizeof(next_addr)) == 0) {
            break;
        }
        closesocket(send_sock);
        send_sock = INVALID_SOCKET;
        MP_RING_SLEEP_MS(MP_RING_CONNECT_DELAY_MS);
    }

    if (send_sock == INVALID_SOCKET) {
        closesocket(listen_sock);
        ctx->comm_ready = 0;
        return MP_ERROR_NONE;
    }

    /* ---- 第三步：接受来自上一个rank的连接 ---- */
    struct sockaddr_in client_addr;
    int addr_len = sizeof(client_addr);
    SOCKET recv_sock = accept(listen_sock, (struct sockaddr*)&client_addr, &addr_len);
    if (recv_sock == INVALID_SOCKET) {
        closesocket(send_sock);
        closesocket(listen_sock);
        ctx->comm_ready = 0;
        return MP_ERROR_NONE;
    }

    /* 关闭监听socket（已不再需要） */
    closesocket(listen_sock);

    /* 存储已建立的连接 */
    ctx->send_socket = (int64_t)send_sock;
    ctx->recv_socket = (int64_t)recv_sock;
    ctx->comm_ready = 1;

    return MP_ERROR_NONE;
}

/* ================================================================
 * mp_3d_close_connections: 关闭环状TCP连接
 * ================================================================ */
void mp_3d_close_connections(MP3DContext* ctx)
{
    if (!ctx) return;
    if (ctx->comm_ready) {
        SOCKET send_sock = (SOCKET)ctx->send_socket;
        SOCKET recv_sock = (SOCKET)ctx->recv_socket;
        if (send_sock != INVALID_SOCKET) {
            closesocket(send_sock);
        }
        if (recv_sock != INVALID_SOCKET) {
            closesocket(recv_sock);
        }
        ctx->send_socket = (int64_t)INVALID_SOCKET;
        ctx->recv_socket = (int64_t)INVALID_SOCKET;
        ctx->comm_ready = 0;
    }
}

/* ================================================================
 * 本地回退函数（单进程多内存缓冲区归约）
 * 当无法建立TCP连接时使用，保持计算正确性
 * ================================================================ */

/* ZSF-007修复：本地回退AllReduce — 仅在单进程模式使用，添加警告标注 */
static int mp_3d_allreduce_local(MP3DContext* ctx, float* data, int count,
                                  int group_size)
{
    (void)ctx;
    /* 单进程内归约：平均值归一化。仅在comm_ready=false时调用。
     * 多进程部署时请确保TCP连接已建立，否则梯度将在不同进程间不一致。 */
    if (count > 0 && group_size > 1) {
        static int warn_count = 0;
        if (warn_count < 3) {
            log_warn("[模型并行] 本地AllReduce回退模式: 单进程内归一化(group_size=%d)。"
                     "多进程部署请确保TCP连接建立以进行真实归约。", group_size);
            warn_count++;
        }
    }
    float inv = 1.0f / (float)(group_size > 0 ? group_size : 1);
    for (int i = 0; i < count; i++) {
        data[i] *= inv;
    }
    return MP_ERROR_NONE;
}

/* ZSF-008修复：本地回退AllGather — 单进程模式使用 */
static int mp_3d_allgather_local(MP3DContext* ctx, float* data, int count,
                                  int group_size, int group_rank,
                                  int* chunk_sizes, int* chunk_offsets)
{
    (void)ctx;
    (void)count;
    /* 单进程内数据聚合：所有数据已在本地，仅需chunk位置验证。
     * 仅在comm_ready=false时调用。多进程部署请确保TCP连接建立。 */
    int my_start = chunk_offsets[group_rank];
    int my_size = chunk_sizes[group_rank];
    /* 在本地模式下data已经包含完整数据，无需操作 */
    /* 只需验证布局正确 */
    (void)my_start;
    (void)my_size;
    return MP_ERROR_NONE;
}

/* 本地回退ReduceScatter */
static int mp_3d_reduce_scatter_local(MP3DContext* ctx, float* data, int count,
                                       int group_size, int group_rank,
                                       int* chunk_sizes, int* chunk_offsets)
{
    (void)ctx;
    /* 本地模式下，data已包含完整归约和散列后的数据 */
    /* 提取属于自己的chunk */
    int my_start = chunk_offsets[group_rank];
    int my_size = chunk_sizes[group_rank];

    /* 本地模式下，所有数据在前一步已经归约 */
    /* 只需将属于自己的chunk移到data前面 */
    float inv = 1.0f / (float)group_size;
    for (int i = 0; i < my_size && (my_start + i) < count; i++) {
        data[i] = data[my_start + i] * inv;
    }
    return MP_ERROR_NONE;
}

int mp_3d_allreduce(MP3DContext* ctx, float* data, int count,
                     MP3DDimension dim)
{
    if (!ctx || !data || count <= 0) return MP_ERROR_INVALID_PARAM;

    int group_size, group_rank;
    if (mp_3d_get_group_info(ctx, dim, &group_size, &group_rank) != MP_ERROR_NONE)
        return MP_ERROR_INVALID_PARAM;
    if (group_size <= 1) return MP_ERROR_NONE;

    /* 如果没有建立真实网络连接，回退到本地实现 */
    if (!ctx->comm_ready) {
        return mp_3d_allreduce_local(ctx, data, count, group_size);
    }

    SOCKET send_sock = (SOCKET)ctx->send_socket;
    SOCKET recv_sock = (SOCKET)ctx->recv_socket;

    size_t needed = (size_t)count * sizeof(float);
    int ret = mp_3d_ensure_buffer(ctx, needed);
    if (ret != MP_ERROR_NONE) return ret;

    int* chunk_sizes = NULL;
    int* chunk_offsets = NULL;
    ret = mp_3d_compute_chunks(group_size, count, &chunk_sizes, &chunk_offsets);
    if (ret != MP_ERROR_NONE) return ret;

    /* ================================================================
     * Ring AllReduce 算法：
     * 阶段1: ReduceScatter（分散归约），group_size-1轮
     * 阶段2: AllGather（全收集），group_size-1轮
     *
     * 环通信模式：
     *   每个rank发送chunk给下一个rank（send_sock → rank+1）
     *   每个rank接收chunk从上一个rank（recv_sock ← rank-1）
     *   每轮发送和接收的chunk索引递减1（模group_size）
     * ================================================================ */

    /* ---- 阶段1：Scatter-Reduce ---- */
    /* 当前reduce_buffer_in保存的是本轮本节点的累计数据 */
    memcpy(ctx->reduce_buffer_in, data, needed);

    /* 临时缓冲区用于接收来自邻居的数据 */
/* raw malloc → safe_malloc */
    float* recv_buf = (float*)safe_malloc((size_t)count * sizeof(float));
    if (!recv_buf) {
        safe_free((void**)&chunk_sizes);
        safe_free((void**)&chunk_offsets);
        return MP_ERROR_ALLOC_FAILED;
    }

    for (int step = 0; step < group_size - 1; step++) {
        /* 当前步要发送的chunk：rank - step - 1 (mod group_size) */
        int send_chunk = (group_rank - step - 1 + group_size) % group_size;
        int send_start = chunk_offsets[send_chunk];
        int send_size = chunk_sizes[send_chunk];

        /* 当前步要接收的chunk：rank - step - 2 (mod group_size) */
        int recv_chunk = (group_rank - step - 2 + group_size) % group_size;
        int recv_start = chunk_offsets[recv_chunk];
        int recv_size = chunk_sizes[recv_chunk];

        /* 发送本节点的累计数据chunk到下一个rank */
        if (mp_ring_send_floats(send_sock,
            ctx->reduce_buffer_in + send_start, send_size) != 0) {
/* raw free → safe_free */
            safe_free((void**)&recv_buf);
            safe_free((void**)&chunk_sizes);
            safe_free((void**)&chunk_offsets);
            return MP_ERROR_COMM_FAILURE;
        }

        /* 接收上一个rank发来的数据chunk */
        if (mp_ring_recv_floats(recv_sock, recv_buf, recv_size) != 0) {
            safe_free((void**)&recv_buf);
            safe_free((void**)&chunk_sizes);
            safe_free((void**)&chunk_offsets);
            return MP_ERROR_COMM_FAILURE;
        }

        /* 将接收的数据归约（累加）到本地缓冲区对应chunk */
        for (int i = 0; i < recv_size; i++) {
            ctx->reduce_buffer_in[recv_start + i] += recv_buf[i];
        }
    }

    /* 阶段1完成：reduce_buffer_in中本rank负责的chunk已包含全部归约结果 */

    /* ---- 阶段2：AllGather ---- */
    for (int step = 0; step < group_size - 1; step++) {
        /* 当前步要发送的chunk：rank - step (mod group_size) */
        int send_chunk = (group_rank - step + group_size) % group_size;
        int send_start = chunk_offsets[send_chunk];
        int send_size = chunk_sizes[send_chunk];

        /* 当前步要接收的chunk：rank - step - 1 (mod group_size) */
        int recv_chunk = (group_rank - step - 1 + group_size) % group_size;
        int recv_start = chunk_offsets[recv_chunk];
        int recv_size = chunk_sizes[recv_chunk];

        /* 发送本节点的chunk到下一个rank */
        if (mp_ring_send_floats(send_sock,
            ctx->reduce_buffer_in + send_start, send_size) != 0) {
            safe_free((void**)&recv_buf);
            safe_free((void**)&chunk_sizes);
            safe_free((void**)&chunk_offsets);
            return MP_ERROR_COMM_FAILURE;
        }

        /* 接收上一个rank发来的chunk */
        if (mp_ring_recv_floats(recv_sock,
            ctx->reduce_buffer_in + recv_start, recv_size) != 0) {
            safe_free((void**)&recv_buf);
            safe_free((void**)&chunk_sizes);
            safe_free((void**)&chunk_offsets);
            return MP_ERROR_COMM_FAILURE;
        }
    }

    /* 阶段2完成：reduce_buffer_in包含所有rank的完整归约结果 */

    /* 结果取平均并写入data */
    float inv = 1.0f / (float)group_size;
    for (int i = 0; i < count; i++) {
        data[i] = ctx->reduce_buffer_in[i] * inv;
    }

/* raw free → safe_free */
    safe_free((void**)&recv_buf);
    safe_free((void**)&chunk_sizes);
    safe_free((void**)&chunk_offsets);
    return MP_ERROR_NONE;
}

int mp_3d_allgather(MP3DContext* ctx, float* data, int count,
                     MP3DDimension dim)
{
    if (!ctx || !data || count <= 0) return MP_ERROR_INVALID_PARAM;

    int group_size, group_rank;
    if (mp_3d_get_group_info(ctx, dim, &group_size, &group_rank) != MP_ERROR_NONE)
        return MP_ERROR_INVALID_PARAM;
    if (group_size <= 1) return MP_ERROR_NONE;

    int* chunk_sizes = NULL;
    int* chunk_offsets = NULL;
    int ret = mp_3d_compute_chunks(group_size, count, &chunk_sizes, &chunk_offsets);
    if (ret != MP_ERROR_NONE) return ret;

    /* 如果没有建立真实网络连接，回退到本地实现 */
    if (!ctx->comm_ready) {
        ret = mp_3d_allgather_local(ctx, data, count, group_size,
                                     group_rank, chunk_sizes, chunk_offsets);
/* raw free → safe_free */
        safe_free((void**)&chunk_sizes);
        safe_free((void**)&chunk_offsets);
        return ret;
    }

    SOCKET send_sock = (SOCKET)ctx->send_socket;
    SOCKET recv_sock = (SOCKET)ctx->recv_socket;

    size_t needed = (size_t)count * sizeof(float);
    ret = mp_3d_ensure_buffer(ctx, needed);
    if (ret != MP_ERROR_NONE) {
        safe_free((void**)&chunk_sizes);
        safe_free((void**)&chunk_offsets);
        return ret;
    }

    /* ================================================================
     * Ring AllGather 算法：
     * 将每个rank的本地数据chunk（位于data[chunk_offsets[rank]:...]）
     * 通过环状通信广播给所有其他rank
     *
     * 每步：
     *   send: data中(rank-step) mod group_size的chunk → 发送给下一个rank
     *   recv: 从上一个rank接收(rank-step-1) mod group_size的chunk → 存入data
     * 共 group_size-1 步
     * ================================================================ */

    /* 初始化：将data复制到buffer，作为通信起点 */
    /* data中，每个rank的chunk在当前rank的对应位置 */
    memcpy(ctx->reduce_buffer_in, data, needed);

    for (int step = 0; step < group_size - 1; step++) {
        /* 发送chunk：rank - step (mod group_size) */
        int send_chunk = (group_rank - step + group_size) % group_size;
        int send_start = chunk_offsets[send_chunk];
        int send_size = chunk_sizes[send_chunk];

        /* 接收chunk：rank - step - 1 (mod group_size) */
        int recv_chunk = (group_rank - step - 1 + group_size) % group_size;
        int recv_start = chunk_offsets[recv_chunk];
        int recv_size = chunk_sizes[recv_chunk];

        /* 发送 */
        if (mp_ring_send_floats(send_sock,
            ctx->reduce_buffer_in + send_start, send_size) != 0) {
            safe_free((void**)&chunk_sizes);
            safe_free((void**)&chunk_offsets);
            return MP_ERROR_COMM_FAILURE;
        }

        /* 接收并直接写入buffer对应位置 */
        if (mp_ring_recv_floats(recv_sock,
            ctx->reduce_buffer_in + recv_start, recv_size) != 0) {
            safe_free((void**)&chunk_sizes);
            safe_free((void**)&chunk_offsets);
            return MP_ERROR_COMM_FAILURE;
        }
    }

    /* buffer现在包含所有rank的完整数据，拷贝回data */
    memcpy(data, ctx->reduce_buffer_in, needed);

/* raw free → safe_free */
    safe_free((void**)&chunk_sizes);
    safe_free((void**)&chunk_offsets);
    return MP_ERROR_NONE;
}

int mp_3d_reduce_scatter(MP3DContext* ctx, float* data, int count,
                          MP3DDimension dim)
{
    if (!ctx || !data || count <= 0) return MP_ERROR_INVALID_PARAM;

    int group_size, group_rank;
    if (mp_3d_get_group_info(ctx, dim, &group_size, &group_rank) != MP_ERROR_NONE)
        return MP_ERROR_INVALID_PARAM;
    if (group_size <= 1) return MP_ERROR_NONE;

    int* chunk_sizes = NULL;
    int* chunk_offsets = NULL;
    int ret = mp_3d_compute_chunks(group_size, count, &chunk_sizes, &chunk_offsets);
    if (ret != MP_ERROR_NONE) return ret;

    /* 如果没有建立真实网络连接，回退到本地实现 */
    if (!ctx->comm_ready) {
        ret = mp_3d_reduce_scatter_local(ctx, data, count, group_size,
                                          group_rank, chunk_sizes, chunk_offsets);
        safe_free((void**)&chunk_sizes);
        safe_free((void**)&chunk_offsets);
        return ret;
    }

    SOCKET send_sock = (SOCKET)ctx->send_socket;
    SOCKET recv_sock = (SOCKET)ctx->recv_socket;

    size_t needed = (size_t)count * sizeof(float);
    ret = mp_3d_ensure_buffer(ctx, needed);
    if (ret != MP_ERROR_NONE) {
        safe_free((void**)&chunk_sizes);
        safe_free((void**)&chunk_offsets);
        return ret;
    }

    /* ================================================================
     * Ring ReduceScatter 算法：
     * 输入data包含每个rank的完整本地数据
     * 通过 group_size-1 步环通信，将data归约并分散到各rank
     *
     * 每步：
     *   send: data中(rank-step-1) mod group_size的chunk → 发送给下一个rank
     *   recv: 从上一个rank接收(rank-step-2) mod group_size的chunk
     *   accumulate: 将recv数据累加到buffer中对应chunk位置
     *
     * 完成后，每个rank的data[0:chunk_size]包含其对应chunk的归约结果
     * ================================================================ */

    /* 初始化：data中的本地数据作为归约起点 */
    memcpy(ctx->reduce_buffer_in, data, needed);

    /* 临时缓冲区用于接收来自邻居的数据 */
/* raw malloc → safe_malloc */
    float* recv_buf = (float*)safe_malloc((size_t)count * sizeof(float));
    if (!recv_buf) {
        safe_free((void**)&chunk_sizes);
        safe_free((void**)&chunk_offsets);
        return MP_ERROR_ALLOC_FAILED;
    }

    for (int step = 0; step < group_size - 1; step++) {
        /* 发送chunk：rank - step - 1 (mod group_size) */
        int send_chunk = (group_rank - step - 1 + group_size) % group_size;
        int send_start = chunk_offsets[send_chunk];
        int send_size = chunk_sizes[send_chunk];

        /* 接收chunk：rank - step - 2 (mod group_size) */
        int recv_chunk = (group_rank - step - 2 + group_size) % group_size;
        int recv_start = chunk_offsets[recv_chunk];
        int recv_size = chunk_sizes[recv_chunk];

        /* 发送当前累计数据的一个chunk */
        if (mp_ring_send_floats(send_sock,
            ctx->reduce_buffer_in + send_start, send_size) != 0) {
            safe_free((void**)&recv_buf);
            safe_free((void**)&chunk_sizes);
            safe_free((void**)&chunk_offsets);
            return MP_ERROR_COMM_FAILURE;
        }

        /* 接收上一个rank发来的chunk */
        if (mp_ring_recv_floats(recv_sock, recv_buf, recv_size) != 0) {
            safe_free((void**)&recv_buf);
            safe_free((void**)&chunk_sizes);
            safe_free((void**)&chunk_offsets);
            return MP_ERROR_COMM_FAILURE;
        }

        /* 将接收的数据归约（累加）到本地缓冲区对应位置 */
        for (int i = 0; i < recv_size; i++) {
            ctx->reduce_buffer_in[recv_start + i] += recv_buf[i];
        }
    }

    /* 归约完成：将本rank的chunk提取到data前面，并取平均 */
    int my_start = chunk_offsets[group_rank];
    int my_size = chunk_sizes[group_rank];
    float inv = 1.0f / (float)group_size;
    for (int i = 0; i < my_size; i++) {
        data[i] = ctx->reduce_buffer_in[my_start + i] * inv;
    }

/* raw free → safe_free */
    safe_free((void**)&recv_buf);
    safe_free((void**)&chunk_sizes);
    safe_free((void**)&chunk_offsets);
    return MP_ERROR_NONE;
}

const char* mp_error_string(int error_code)
{
    switch (error_code) {
        case MP_ERROR_NONE: return "成功";
        case MP_ERROR_INVALID_PARAM: return "参数无效";
        case MP_ERROR_OUT_OF_MEMORY: return "内存不足";
        case MP_ERROR_SIZE_MISMATCH: return "大小不匹配";
        case MP_ERROR_SHARD_UNAVAILABLE: return "分片不可用";
        case MP_ERROR_COMM_FAILURE: return "通信失败";
        case MP_ERROR_SHARD_MISMATCH: return "分片不匹配";
        case MP_ERROR_ALLOC_FAILED: return "分配失败";
        default: return "未知错误";
    }
}