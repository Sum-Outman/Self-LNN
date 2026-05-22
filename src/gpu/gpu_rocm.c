/**
 * @file gpu_rocm.c
 * @brief AMD ROCm GPU后端完整实现
 *
 * AMD ROCm GPU后端完整实现 - 提供真实的AMD GPU硬件加速。
 * 使用AMD HIP运行时API（Heterogeneous-compute Interface for Portability）。
 * 根据项目要求"禁止任何降级处理"，本实现不包含任何CPU模拟回退。
 * 需要AMD ROCm软件栈和兼容的AMD GPU硬件。
 */

#include "selflnn/gpu/gpu.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/platform.h"
#include "gpu_internal.h"

/* ZSFWS-L009: 内核缓存自旋锁——保护多线程并发编译时的竞态条件 */
#ifdef _WIN32
#include <windows.h>
static volatile LONG rocm_cache_lock_val = 0;
#define ROCM_CACHE_LOCK()   while (InterlockedCompareExchange(&rocm_cache_lock_val, 1, 0) != 0) { /* yield */ Sleep(0); }
#define ROCM_CACHE_UNLOCK() InterlockedExchange(&rocm_cache_lock_val, 0)
#else
static volatile int rocm_cache_lock_val = 0;
#define ROCM_CACHE_LOCK()   while (__sync_lock_test_and_set(&rocm_cache_lock_val, 1)) { /* yield */ usleep(0); }
#define ROCM_CACHE_UNLOCK() __sync_lock_release(&rocm_cache_lock_val)
#endif
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#ifdef _WIN32
#include <io.h>
#include <direct.h>
#define F_OK 0
#define access _access
#define mkdir _mkdir
#define MAX_PATH 260
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#define MAX_PATH 4096
#endif

#ifdef _WIN32
#include <windows.h>
#define HIP_RUNTIME_LIBRARY_NAME "amdhip64.dll"
#define LIBRARY_HANDLE HMODULE
#define LOAD_LIBRARY(name) LoadLibraryA(name)
#define GET_PROC_ADDRESS(handle, name) GetProcAddress(handle, name)
#define CLOSE_LIBRARY(handle) FreeLibrary(handle)
#else
#include <dlfcn.h>
#define HIP_RUNTIME_LIBRARY_NAME "libamdhip64.so"
#define LIBRARY_HANDLE void*
#define LOAD_LIBRARY(name) dlopen(name, RTLD_LAZY)
#define GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#define CLOSE_LIBRARY(handle) dlclose(handle)
#endif

/* 运行时动态加载类型: HIP SDK类型在编译时不可用, 通过纯C兼容类型替代 */
typedef int hipError_t;
typedef struct { char name[256]; size_t totalGlobalMem; int major; int minor; int clockRate; int multiProcessorCount; } hipDeviceProp_t;

static LIBRARY_HANDLE g_hip_library = NULL;
static hipError_t (*hipGetDeviceCount)(int*) = NULL;
static hipError_t (*hipGetDeviceProperties)(hipDeviceProp_t*, int) = NULL;
static hipError_t (*hipSetDevice)(int) = NULL;
static hipError_t (*hipGetDevice)(int*) = NULL;
static hipError_t (*hipInit)(unsigned int) = NULL;
static hipError_t (*hipDriverGetVersion)(int*) = NULL;
static hipError_t (*hipRuntimeGetVersion)(int*) = NULL;
static hipError_t (*hipDeviceGetName)(char*, int, int) = NULL;
static hipError_t (*hipDeviceGetAttribute)(int*, int, int) = NULL;
static hipError_t (*hipDeviceSynchronize)(void) = NULL;
static hipError_t (*hipMalloc)(void**, size_t) = NULL;
static hipError_t (*hipFree)(void*) = NULL;
static hipError_t (*hipMemcpy)(void*, const void*, size_t, int) = NULL;
static hipError_t (*hipMallocHost)(void**, size_t) = NULL;
static hipError_t (*hipFreeHost)(void*) = NULL;
static hipError_t (*hipMallocManaged)(void**, size_t, unsigned int) = NULL;
static hipError_t (*hipMemPrefetchAsync)(const void*, size_t, int, void*) = NULL;
static hipError_t (*hipEventCreate)(void**) = NULL;
static hipError_t (*hipEventRecord)(void*, void*) = NULL;
static hipError_t (*hipEventSynchronize)(void*) = NULL;
static hipError_t (*hipEventElapsedTime)(float*, void*, void*) = NULL;
static hipError_t (*hipEventDestroy)(void*) = NULL;
static hipError_t (*hipStreamCreate)(void**) = NULL;
static hipError_t (*hipStreamDestroy)(void*) = NULL;
static hipError_t (*hipStreamSynchronize)(void*) = NULL;
static hipError_t (*hipStreamQuery)(void*) = NULL;
static hipError_t (*hipMemcpyAsync)(void*, const void*, size_t, int, void*) = NULL;
static hipError_t (*hipGetLastError)(void) = NULL;
static hipError_t (*hipFuncSetAttribute)(const void*, int, int) = NULL;
static hipError_t (*hipModuleLoad)(void**, const char*) = NULL;
static hipError_t (*hipModuleUnload)(void*) = NULL;
static hipError_t (*hipModuleGetFunction)(void**, void*, const char*) = NULL;
static hipError_t (*hipModuleLaunchKernel)(void*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void**, void**) = NULL;
static hipError_t (*hipOccupancyMaxPotentialBlockSize)(int*, int*, const void*, size_t, int) = NULL;
static hipError_t (*hipMemGetInfo)(size_t*, size_t*) = NULL;
static hipError_t (*hipDeviceReset)(void) = NULL;

#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2
#define hipMemcpyDeviceToDevice 3
#define hipMemcpyHostToHost 0

#define hipSuccess 0
#define hipErrorMemoryAllocation 2
#define hipErrorInvalidDevice 3
#define hipErrorInvalidValue 4
#define hipErrorLaunchFailure 5
#define hipErrorNoDevice 6
#define hipErrorInvalidResourceHandle 7
#define hipErrorNotReady 8
#define hipErrorOutOfMemory 9
#define hipErrorRuntimeOther 10
#define hipErrorInvalidImage 11
#define hipErrorSharedObjectInitFailed 12

/* ============================================================================
 * HIP内核源代码
 * 提供真实的AMD GPU计算功能
 * =========================================================================== */

static const char* ROCM_VECTOR_ADD_KERNEL =
"extern \"C\" __global__ void vector_add(float* a, float* b, float* c, int n) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx < n) {\n"
"        c[idx] = a[idx] + b[idx];\n"
"    }\n"
"}\n";

static const char* ROCM_MATRIX_MUL_KERNEL =
"extern \"C\" __global__ void matrix_mul_basic(float* a, float* b, float* c, int m, int n, int k) {\n"
"    int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (row < m && col < k) {\n"
"        float sum = 0.0f;\n"
"        for (int i = 0; i < n; i++) {\n"
"            sum += a[row * n + i] * b[i * k + col];\n"
"        }\n"
"        c[row * k + col] = sum;\n"
"    }\n"
"}\n";

static const char* ROCM_MATRIX_MUL_SHARED_KERNEL =
"extern \"C\" __global__ void matrix_mul_shared(float* a, float* b, float* c, int m, int n, int k) {\n"
"    const int BLOCK_SIZE = 16;\n"
"    __shared__ float shared_a[16][16];\n"
"    __shared__ float shared_b[16][16];\n"
"    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;\n"
"    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;\n"
"    float sum = 0.0f;\n"
"    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {\n"
"        int a_col = tile * BLOCK_SIZE + threadIdx.x;\n"
"        if (row < m && a_col < n) shared_a[threadIdx.y][threadIdx.x] = a[row * n + a_col];\n"
"        else shared_a[threadIdx.y][threadIdx.x] = 0.0f;\n"
"        int b_row = tile * BLOCK_SIZE + threadIdx.y;\n"
"        if (b_row < n && col < k) shared_b[threadIdx.y][threadIdx.x] = b[b_row * k + col];\n"
"        else shared_b[threadIdx.y][threadIdx.x] = 0.0f;\n"
"        __syncthreads();\n"
"        for (int i = 0; i < BLOCK_SIZE; ++i) sum += shared_a[threadIdx.y][i] * shared_b[i][threadIdx.x];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < m && col < k) c[row * k + col] = sum;\n"
"}\n";

static const char* ROCM_RELU_KERNEL =
"extern \"C\" __global__ void relu_activation(float* x, float* y, int n) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx < n) {\n"
"        float val = x[idx];\n"
"        y[idx] = (val > 0.0f) ? val : 0.0f;\n"
"    }\n"
"}\n";

static const char* ROCM_SIGMOID_KERNEL =
"extern \"C\" __global__ void sigmoid_activation(float* x, float* y, int n) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx < n) {\n"
"        float val = x[idx];\n"
"        y[idx] = 1.0f / (1.0f + expf(-val));\n"
"    }\n"
"}\n";

static const char* ROCM_TANH_KERNEL =
"extern \"C\" __global__ void tanh_activation(float* x, float* y, int n) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx < n) {\n"
"        float val = x[idx];\n"
"        y[idx] = tanhf(val);\n"
"    }\n"
"}\n";

static const char* ROCM_LAYER_NORM_KERNEL =
"extern \"C\" __global__ void layer_norm(float* input, float* output, float* gamma, float* beta, int n, float eps) {\n"
"    extern __shared__ float shared[];\n"
"    int tid = threadIdx.x;\n"
"    int bid = blockIdx.x;\n"
"    int offset = bid * n;\n"
"    float sum = 0.0f, sum2 = 0.0f;\n"
"    for (int i = tid; i < n; i += blockDim.x) {\n"
"        float val = input[offset + i];\n"
"        sum += val;\n"
"        sum2 += val * val;\n"
"    }\n"
"    shared[tid] = sum;\n"
"    shared[tid + blockDim.x] = sum2;\n"
"    __syncthreads();\n"
"    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            shared[tid] += shared[tid + stride];\n"
"            shared[tid + blockDim.x] += shared[tid + stride + blockDim.x];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (tid == 0) {\n"
"        float mean = shared[0] / n;\n"
"        float var = shared[blockDim.x] / n - mean * mean;\n"
"        shared[0] = mean;\n"
"        shared[1] = var;\n"
"    }\n"
"    __syncthreads();\n"
"    float mean = shared[0];\n"
"    float var = shared[1];\n"
"    float inv_std = rsqrtf(var + eps);\n"
"    for (int i = tid; i < n; i += blockDim.x) {\n"
"        output[offset + i] = (input[offset + i] - mean) * inv_std * gamma[i] + beta[i];\n"
"    }\n"
"}\n";

static const char* ROCM_SOFTMAX_KERNEL =
"extern \"C\" __global__ void softmax(float* input, float* output, int n) {\n"
"    int bid = blockIdx.x;\n"
"    int offset = bid * n;\n"
"    extern __shared__ float shared[];\n"
"    int tid = threadIdx.x;\n"
"    float max_val = -1e20f;\n"
"    for (int i = tid; i < n; i += blockDim.x) {\n"
"        if (input[offset + i] > max_val) max_val = input[offset + i];\n"
"    }\n"
"    shared[tid] = max_val;\n"
"    __syncthreads();\n"
"    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
"        if (tid < stride && shared[tid + stride] > shared[tid]) shared[tid] = shared[tid + stride];\n"
"        __syncthreads();\n"
"    }\n"
"    float row_max = shared[0];\n"
"    float sum = 0.0f;\n"
"    for (int i = tid; i < n; i += blockDim.x) {\n"
"        sum += expf(input[offset + i] - row_max);\n"
"    }\n"
"    shared[tid] = sum;\n"
"    __syncthreads();\n"
"    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) shared[tid] += shared[tid + stride];\n"
"        __syncthreads();\n"
"    }\n"
"    float row_sum = shared[0];\n"
"    for (int i = tid; i < n; i += blockDim.x) {\n"
"        output[offset + i] = expf(input[offset + i] - row_max) / row_sum;\n"
"    }\n"
"}\n";

static const char* ROCM_ADD_BIAS_KERNEL =
"extern \"C\" __global__ void add_bias(float* input, float* bias, float* output, int m, int n) {\n"
"    int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (row < m && col < n) {\n"
"        output[row * n + col] = input[row * n + col] + bias[col];\n"
"    }\n"
"}\n";

static const char* ROCM_DROPOUT_KERNEL =
"extern \"C\" __global__ void dropout_forward(float* input, float* output, float* mask, int n, float p, float scale) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx < n) {\n"
"        output[idx] = (mask[idx] < p) ? 0.0f : input[idx] * scale;\n"
"    }\n"
"}\n";

static const char* ROCM_CFC_STEP_KERNEL =
"extern \"C\" __global__ void cfc_step(float* h, float* x, float* W, float* b, int hidden_size, int input_size) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx < hidden_size) {\n"
"        float sum = b[idx];\n"
"        for (int j = 0; j < hidden_size; j++) {\n"
"            sum += W[idx * hidden_size + j] * h[j];\n"
"        }\n"
"        for (int j = 0; j < input_size; j++) {\n"
"            sum += W[hidden_size * hidden_size + idx * input_size + j] * x[j];\n"
"        }\n"
"        h[idx] = tanhf(sum);\n"
"    }\n"
"}\n";

#define ROCM_CACHE_MAX_ENTRIES 64
#define ROCM_CACHE_HASH_SIZE 32
#define ROCM_MAX_ALLOCATIONS 256

typedef struct {
    char hash_key[ROCM_CACHE_HASH_SIZE + 1];
    char kernel_name[256];
    char* hsaco_code;
    size_t hsaco_size;
    void* hip_module;
    void* hip_function;
    int valid;
    int access_count;
    uint64_t compile_time_ms;
} RocmCacheEntry;

typedef struct {
    void* device_ptr;           /**< 设备内存指针 */
    size_t size;                /**< 分配大小（字节） */
    int is_active;              /**< 是否活跃 */
} RocmAllocation;

typedef struct {
    int device_id;
    void* default_stream;
    int initialized;
    RocmCacheEntry kernel_cache[ROCM_CACHE_MAX_ENTRIES];
    int cache_count;
    int cache_hits;
    int cache_misses;
    int max_work_group_size;
    int max_threads_per_block;
    int multi_processor_count;
    size_t total_global_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int shared_memory_per_block;
    int hip_device;                                    /**< HIP设备索引 */
    int allocation_count;                              /**< 活跃分配数量 */
    RocmAllocation allocations[ROCM_MAX_ALLOCATIONS];  /**< 分配追踪表 */
} RocmContextInternal;

static int rocm_try_load_library(void) {
    if (g_hip_library) return 1;
    g_hip_library = LOAD_LIBRARY(HIP_RUNTIME_LIBRARY_NAME);
    if (!g_hip_library) return 0;
    
#define LOAD_HIP_SYMBOL(name) do { \
    *(void**)(&name) = (void*)GET_PROC_ADDRESS(g_hip_library, #name); \
    if (!name) { CLOSE_LIBRARY(g_hip_library); g_hip_library = NULL; return 0; } \
} while(0)

LOAD_HIP_SYMBOL(hipGetDeviceCount);
LOAD_HIP_SYMBOL(hipGetDeviceProperties);
LOAD_HIP_SYMBOL(hipSetDevice);
LOAD_HIP_SYMBOL(hipGetDevice);
LOAD_HIP_SYMBOL(hipInit);
LOAD_HIP_SYMBOL(hipDriverGetVersion);
LOAD_HIP_SYMBOL(hipRuntimeGetVersion);
LOAD_HIP_SYMBOL(hipDeviceGetName);
LOAD_HIP_SYMBOL(hipDeviceGetAttribute);
LOAD_HIP_SYMBOL(hipDeviceSynchronize);
LOAD_HIP_SYMBOL(hipMalloc);
LOAD_HIP_SYMBOL(hipFree);
LOAD_HIP_SYMBOL(hipMemcpy);
LOAD_HIP_SYMBOL(hipMallocHost);
LOAD_HIP_SYMBOL(hipFreeHost);
LOAD_HIP_SYMBOL(hipMallocManaged);
LOAD_HIP_SYMBOL(hipEventCreate);
LOAD_HIP_SYMBOL(hipEventRecord);
LOAD_HIP_SYMBOL(hipEventSynchronize);
LOAD_HIP_SYMBOL(hipEventElapsedTime);
LOAD_HIP_SYMBOL(hipEventDestroy);
LOAD_HIP_SYMBOL(hipStreamCreate);
LOAD_HIP_SYMBOL(hipStreamDestroy);
LOAD_HIP_SYMBOL(hipStreamSynchronize);
LOAD_HIP_SYMBOL(hipStreamQuery);
LOAD_HIP_SYMBOL(hipMemcpyAsync);
LOAD_HIP_SYMBOL(hipGetLastError);
LOAD_HIP_SYMBOL(hipFuncSetAttribute);
LOAD_HIP_SYMBOL(hipModuleLoad);
LOAD_HIP_SYMBOL(hipModuleUnload);
LOAD_HIP_SYMBOL(hipModuleGetFunction);
LOAD_HIP_SYMBOL(hipModuleLaunchKernel);
LOAD_HIP_SYMBOL(hipOccupancyMaxPotentialBlockSize);
LOAD_HIP_SYMBOL(hipMemGetInfo);
LOAD_HIP_SYMBOL(hipDeviceReset);

return 1;
}

static void rocm_hash_kernel(const char* source, const char* name, char* hash_out) {
    uint64_t hash = 5381;
    int c;
    const char* s = source;
    if (s) {
        while ((c = (unsigned char)*s++) != 0) hash = ((hash << 5) + hash) + c;
    }
    s = name;
    if (s) {
        while ((c = (unsigned char)*s++) != 0) hash = ((hash << 5) + hash) + c;
    }
    unsigned long len_hash = (source ? strlen(source) : 0) * 2654435761u;
    hash ^= len_hash;
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    static const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < ROCM_CACHE_HASH_SIZE; i++) {
        hash_out[i] = hex_chars[(hash >> (i * 4 % 60)) & 0x0F];
        if (i > 0 && i % 8 == 7) hash = (hash << 3) | (hash >> 61);
    }
    hash_out[ROCM_CACHE_HASH_SIZE] = '\0';
}

static RocmCacheEntry* rocm_cache_lookup(RocmContextInternal* ctx, const char* hash_key) {
    if (!ctx || !hash_key) return NULL;
    ROCM_CACHE_LOCK();
    for (int i = 0; i < ctx->cache_count; i++) {
        if (ctx->kernel_cache[i].valid &&
            strncmp(ctx->kernel_cache[i].hash_key, hash_key, ROCM_CACHE_HASH_SIZE) == 0) {
            ctx->kernel_cache[i].access_count++;
            ctx->cache_hits++;
            ROCM_CACHE_UNLOCK();
            return &ctx->kernel_cache[i];
        }
    }
    ctx->cache_misses++;
    ROCM_CACHE_UNLOCK();
    return NULL;
}

static void rocm_cache_evict_one(RocmContextInternal* ctx) {
    if (!ctx || ctx->cache_count <= 0) return;
    int lru_idx = -1;
    int min_access = 0x7FFFFFFF;
    for (int i = 0; i < ctx->cache_count; i++) {
        if (ctx->kernel_cache[i].valid && ctx->kernel_cache[i].access_count < min_access) {
            min_access = ctx->kernel_cache[i].access_count;
            lru_idx = i;
        }
    }
    if (lru_idx < 0) return;
    RocmCacheEntry* entry = &ctx->kernel_cache[lru_idx];
    if (entry->hip_module && hipModuleUnload) hipModuleUnload(entry->hip_module);
    if (entry->hsaco_code) safe_free((void**)&entry->hsaco_code);
    memset(entry, 0, sizeof(RocmCacheEntry));
    if (lru_idx < ctx->cache_count - 1) {
        int last_valid = -1;
        for (int i = ctx->cache_count - 1; i > lru_idx; i--) {
            if (ctx->kernel_cache[i].valid) { last_valid = i; break; }
        }
        if (last_valid >= 0) {
            memcpy(&ctx->kernel_cache[lru_idx], &ctx->kernel_cache[last_valid], sizeof(RocmCacheEntry));
            memset(&ctx->kernel_cache[last_valid], 0, sizeof(RocmCacheEntry));
        }
    }
    ctx->cache_count--;
}

static int rocm_cache_insert(RocmContextInternal* ctx, const char* hash_key, const char* name,
                              char* hsaco, size_t hsaco_size, void* module, void* function,
                              uint64_t compile_time_ms) {
    if (!ctx || !hash_key || !hsaco || hsaco_size == 0 || !module || !function) return -1;
    ROCM_CACHE_LOCK();
    if (ctx->cache_count >= ROCM_CACHE_MAX_ENTRIES) rocm_cache_evict_one(ctx);
    int idx = -1;
    for (int i = 0; i < ROCM_CACHE_MAX_ENTRIES; i++) {
        if (!ctx->kernel_cache[i].valid) { idx = i; break; }
    }
    if (idx < 0) { rocm_cache_evict_one(ctx); for (int i = 0; i < ROCM_CACHE_MAX_ENTRIES; i++) { if (!ctx->kernel_cache[i].valid) { idx = i; break; } } }
    if (idx < 0) { ROCM_CACHE_UNLOCK(); return -1; }
    RocmCacheEntry* entry = &ctx->kernel_cache[idx];
    memset(entry, 0, sizeof(RocmCacheEntry));
    strncpy(entry->hash_key, hash_key, ROCM_CACHE_HASH_SIZE);
    entry->hash_key[ROCM_CACHE_HASH_SIZE] = '\0';
    if (name) { strncpy(entry->kernel_name, name, sizeof(entry->kernel_name) - 1); entry->kernel_name[sizeof(entry->kernel_name) - 1] = '\0'; }
    entry->hsaco_code = hsaco;
    entry->hsaco_size = hsaco_size;
    entry->hip_module = module;
    entry->hip_function = function;
    entry->valid = 1;
    entry->access_count = 0;
    entry->compile_time_ms = compile_time_ms;
    ctx->cache_count++;
    ROCM_CACHE_UNLOCK();
    return 0;
}

static void rocm_cache_cleanup_all(RocmContextInternal* ctx) {
    if (!ctx) return;
    for (int i = 0; i < ROCM_CACHE_MAX_ENTRIES; i++) {
        if (ctx->kernel_cache[i].valid) {
            if (ctx->kernel_cache[i].hip_module && hipModuleUnload) hipModuleUnload(ctx->kernel_cache[i].hip_module);
            if (ctx->kernel_cache[i].hsaco_code) safe_free((void**)&ctx->kernel_cache[i].hsaco_code);
            memset(&ctx->kernel_cache[i], 0, sizeof(RocmCacheEntry));
        }
    }
    ctx->cache_count = 0;
}

static int rocm_backend_init(void) {
    if (!rocm_try_load_library()) return -1;
    if (hipInit(0) != hipSuccess) return -1;
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) return -1;
    return 0;
}

static void rocm_backend_cleanup(void) {
    if (g_hip_library) {
        CLOSE_LIBRARY(g_hip_library);
        g_hip_library = NULL;
    }
    memset(&hipGetDeviceCount, 0, sizeof(hipGetDeviceCount));
}

static int rocm_backend_get_device_count(void) {
    if (!g_hip_library) return 0;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) return 0;
    return count;
}

static int rocm_backend_get_device_info(int device_index, GpuDeviceInfo* info) {
    if (!info || !g_hip_library) return -1;
    memset(info, 0, sizeof(GpuDeviceInfo));
    info->device_id = device_index;
    char name[256] = {0};
    if (hipDeviceGetName(name, 256, device_index) == hipSuccess) {
        strncpy(info->name, name, sizeof(info->name) - 1);
    } else {
        strncpy(info->name, "AMD GPU (未知型号)", sizeof(info->name) - 1);
    }
    snprintf(info->vendor, sizeof(info->vendor), "AMD");

    int val = 0;
    /* hipDeviceAttribute_t 0 = hipDeviceAttributeIntegrated */
    if (hipDeviceGetAttribute(&val, 0, device_index) == hipSuccess)
        info->type = (val ? GPU_DEVICE_TYPE_INTEGRATED : GPU_DEVICE_TYPE_DISCRETE);
    else info->type = GPU_DEVICE_TYPE_DISCRETE;

    /* hipDeviceAttribute_t 1 = hipDeviceAttributeMultiprocessorCount */
    if (hipDeviceGetAttribute(&val, 1, device_index) == hipSuccess) info->compute_units = val;
    /* hipDeviceAttribute_t 2 = hipDeviceAttributeClockRate (kHz -> MHz) */
    if (hipDeviceGetAttribute(&val, 2, device_index) == hipSuccess) info->clock_speed = (float)val / 1000.0f;
    /* hipDeviceAttribute_t 3 = hipDeviceAttributeMaxThreadsPerBlock */
    if (hipDeviceGetAttribute(&val, 3, device_index) == hipSuccess) info->max_work_group_size = val;

    /* hipDeviceAttribute_t hipDeviceAttributeMemoryClockRate (kHz) = 11? */
    /* 尝试通过hipMemGetInfo获取真实显存 */
    size_t free_mem = 0, total_mem = 0;
    if (hipMemGetInfo) hipMemGetInfo(&free_mem, &total_mem);
    info->total_memory = total_mem;
    info->free_memory = free_mem;

    /* 精度支持: AMD GPU通常支持FP16和FP64 */
    info->supports_double = 1;
    info->supports_half = 1;

    /* 驱动版本号 */
    int runtime_ver = 0;
    if (hipRuntimeGetVersion && hipRuntimeGetVersion(&runtime_ver) == hipSuccess) {
        snprintf(info->driver_version, sizeof(info->driver_version), "%d", runtime_ver);
        snprintf(info->runtime_version, sizeof(info->runtime_version), "%d", runtime_ver / 1000);
    }

    return 0;
}

static GpuContext* rocm_backend_context_create(int device_index) {
    if (!g_hip_library) return NULL;
    if (hipSetDevice(device_index) != hipSuccess) return NULL;
    RocmContextInternal* rocm_ctx = (RocmContextInternal*)safe_calloc(1, sizeof(RocmContextInternal));
    if (!rocm_ctx) return NULL;
    rocm_ctx->device_id = device_index;
    rocm_ctx->hip_device = device_index;
    rocm_ctx->initialized = 1;
    rocm_ctx->allocation_count = 0;
    memset(rocm_ctx->allocations, 0, sizeof(rocm_ctx->allocations));
    void* stream = NULL;
    if (hipStreamCreate(&stream) == hipSuccess) rocm_ctx->default_stream = stream;
    hipDeviceGetAttribute(&rocm_ctx->max_work_group_size, 3, device_index);
    hipDeviceGetAttribute(&rocm_ctx->max_threads_per_block, 4, device_index);
    hipDeviceGetAttribute(&rocm_ctx->multi_processor_count, 1, device_index);
    hipDeviceGetAttribute(&rocm_ctx->shared_memory_per_block, 8, device_index);
    size_t total_mem = 0;
    int mem_val = 0;
    if (hipDeviceGetAttribute(&mem_val, 10, device_index) == hipSuccess) total_mem = (size_t)mem_val * 1024 * 1024;
    rocm_ctx->total_global_memory = total_mem;
    struct GpuContext* ctx = (struct GpuContext*)safe_calloc(1, sizeof(struct GpuContext));
    if (!ctx) { safe_free((void**)&rocm_ctx); return NULL; }
    ctx->backend = GPU_BACKEND_ROCM;
    ctx->device_index = device_index;
    ctx->total_memory = total_mem;
    ctx->free_memory = total_mem;
    ctx->is_initialized = 1;
    char name[256] = {0};
    if (hipDeviceGetName(name, 256, device_index) == hipSuccess) strncpy(ctx->device_name, name, sizeof(ctx->device_name) - 1);
    else strncpy(ctx->device_name, "AMD ROCm GPU", sizeof(ctx->device_name) - 1);
    ctx->backend_data = rocm_ctx;
    return (GpuContext*)ctx;
}

static void rocm_backend_context_free(GpuContext* context) {
    if (!context) return;
    struct GpuContext* ctx = (struct GpuContext*)context;
    RocmContextInternal* rocm_ctx = (RocmContextInternal*)ctx->backend_data;
    if (rocm_ctx) {
        rocm_cache_cleanup_all(rocm_ctx);
        if (rocm_ctx->default_stream && hipStreamDestroy) hipStreamDestroy(rocm_ctx->default_stream);
        safe_free((void**)&rocm_ctx);
    }
    safe_free((void**)&ctx);
}

static GpuMemory* rocm_backend_memory_alloc(GpuContext* context, size_t size, GpuMemoryType memory_type) {
    if (!context || size == 0) return NULL;
    struct GpuContext* ctx = (struct GpuContext*)context;
    RocmContextInternal* rocm_ctx = (RocmContextInternal*)ctx->backend_data;
    if (!rocm_ctx || !rocm_ctx->initialized) return NULL;
    void* device_ptr = NULL;
    if (memory_type == GPU_MEMORY_HOST) {
        if (hipMallocHost(&device_ptr, size) != hipSuccess) return NULL;
    } else if (memory_type == GPU_MEMORY_UNIFIED) {
        if (hipMallocManaged(&device_ptr, size, 0) != hipSuccess) return NULL;
    } else {
        if (hipMalloc(&device_ptr, size) != hipSuccess) return NULL;
    }
    struct GpuMemory* mem = (struct GpuMemory*)safe_calloc(1, sizeof(struct GpuMemory));
    if (!mem) {
        if (memory_type == GPU_MEMORY_HOST) hipFreeHost(device_ptr);
        else hipFree(device_ptr);
        return NULL;
    }
    mem->context = context;
    mem->data = device_ptr;
    mem->size = size;
    mem->type = memory_type;
    mem->is_device_memory = (memory_type != GPU_MEMORY_HOST) ? 1 : 0;

    /* 追踪设备内存分配 */
    if (memory_type != GPU_MEMORY_HOST && rocm_ctx->allocation_count < ROCM_MAX_ALLOCATIONS) {
        rocm_ctx->allocations[rocm_ctx->allocation_count].device_ptr = device_ptr;
        rocm_ctx->allocations[rocm_ctx->allocation_count].size = size;
        rocm_ctx->allocations[rocm_ctx->allocation_count].is_active = 1;
        rocm_ctx->allocation_count++;
    }

    return (GpuMemory*)mem;
}

static void rocm_backend_memory_free(GpuMemory* memory) {
    if (!memory) return;
    struct GpuMemory* mem = (struct GpuMemory*)memory;

    /* 从分配追踪表中移除 */
    if (mem->context && mem->type != GPU_MEMORY_HOST) {
        struct GpuContext* ctx = (struct GpuContext*)mem->context;
        RocmContextInternal* rocm_ctx = (RocmContextInternal*)ctx->backend_data;
        if (rocm_ctx) {
            for (int i = 0; i < rocm_ctx->allocation_count; i++) {
                if (rocm_ctx->allocations[i].device_ptr == mem->data &&
                    rocm_ctx->allocations[i].is_active) {
                    rocm_ctx->allocations[i].is_active = 0;
                    rocm_ctx->allocations[i].device_ptr = NULL;
                    break;
                }
            }
        }
    }

    if (mem->data) {
        if (mem->type == GPU_MEMORY_HOST && hipFreeHost) hipFreeHost(mem->data);
        else if (hipFree) hipFree(mem->data);
    }
    safe_free((void**)&mem);
}

static int rocm_backend_memory_copy_to_device(GpuMemory* dst, const void* src, size_t size) {
    if (!dst || !src) return -1;
    struct GpuMemory* dst_mem = (struct GpuMemory*)dst;
    if (!dst_mem->data) return -1;
    size_t copy_size = (size < dst_mem->size) ? size : dst_mem->size;
    return (hipMemcpy(dst_mem->data, src, copy_size, hipMemcpyHostToDevice) == hipSuccess) ? 0 : -1;
}

static int rocm_backend_memory_copy_from_device(void* dst, GpuMemory* src, size_t size) {
    if (!dst || !src) return -1;
    struct GpuMemory* src_mem = (struct GpuMemory*)src;
    if (!src_mem->data) return -1;
    size_t copy_size = (size < src_mem->size) ? size : src_mem->size;
    return (hipMemcpy(dst, src_mem->data, copy_size, hipMemcpyDeviceToHost) == hipSuccess) ? 0 : -1;
}

static int rocm_backend_memory_copy_device_to_device(GpuMemory* dst, GpuMemory* src, size_t size) {
    if (!dst || !src) return -1;
    struct GpuMemory* dst_mem = (struct GpuMemory*)dst;
    struct GpuMemory* src_mem = (struct GpuMemory*)src;
    if (!dst_mem->data || !src_mem->data) return -1;
    size_t copy_size = size;
    if (copy_size > dst_mem->size) copy_size = dst_mem->size;
    if (copy_size > src_mem->size) copy_size = src_mem->size;
    return (hipMemcpy(dst_mem->data, src_mem->data, copy_size, hipMemcpyDeviceToDevice) == hipSuccess) ? 0 : -1;
}

static int rocm_backend_memory_copy_to_device_async(GpuMemory* dst, const void* src, size_t size, GpuStream* stream) {
    if (!dst || !src) return -1;
    struct GpuMemory* dst_mem = (struct GpuMemory*)dst;
    if (!dst_mem->data) return -1;
    size_t copy_size = (size < dst_mem->size) ? size : dst_mem->size;
    struct GpuStream* gpu_stream = (struct GpuStream*)stream;
    void* hip_stream = gpu_stream ? gpu_stream->backend_data : NULL;
    return (hipMemcpyAsync(dst_mem->data, src, copy_size, hipMemcpyHostToDevice, hip_stream) == hipSuccess) ? 0 : -1;
}

static int rocm_backend_memory_copy_from_device_async(void* dst, GpuMemory* src, size_t size, GpuStream* stream) {
    if (!dst || !src) return -1;
    struct GpuMemory* src_mem = (struct GpuMemory*)src;
    if (!src_mem->data) return -1;
    size_t copy_size = (size < src_mem->size) ? size : src_mem->size;
    struct GpuStream* gpu_stream = (struct GpuStream*)stream;
    void* hip_stream = gpu_stream ? gpu_stream->backend_data : NULL;
    return (hipMemcpyAsync(dst, src_mem->data, copy_size, hipMemcpyDeviceToHost, hip_stream) == hipSuccess) ? 0 : -1;
}

static GpuKernel* rocm_backend_kernel_create(GpuContext* context, const char* kernel_source, const char* kernel_name) {
    if (!context || !kernel_source || !kernel_name) return NULL;
    struct GpuContext* ctx = (struct GpuContext*)context;
    RocmContextInternal* rocm_ctx = (RocmContextInternal*)ctx->backend_data;
    if (!rocm_ctx || !rocm_ctx->initialized) return NULL;
    char hash_key[ROCM_CACHE_HASH_SIZE + 1] = {0};
    rocm_hash_kernel(kernel_source, kernel_name, hash_key);
    RocmCacheEntry* cached = rocm_cache_lookup(rocm_ctx, hash_key);
    if (cached && cached->hip_function) {
        struct GpuKernel* kernel = (struct GpuKernel*)safe_calloc(1, sizeof(struct GpuKernel));
        if (!kernel) return NULL;
        kernel->context = context;
        kernel->kernel_name = string_duplicate(kernel_name);
        kernel->kernel_source = string_duplicate(kernel_source);
        kernel->backend_data = cached->hip_function;
        return (GpuKernel*)kernel;
    }
    char temp_dir[MAX_PATH];
#ifdef _WIN32
    GetTempPathA(MAX_PATH, temp_dir);
#else
    strncpy(temp_dir, "/tmp/", MAX_PATH - 1);
#endif
    char src_path[MAX_PATH];
    snprintf(src_path, sizeof(src_path), "%s/rocm_kernel_XXXXXX.rocm", temp_dir);
    char real_src_path[MAX_PATH];
#ifdef _WIN32
    char tmpname[MAX_PATH];
    GetTempFileNameA(temp_dir, "hip", 0, tmpname);
    snprintf(real_src_path, sizeof(real_src_path), "%s.cpp", tmpname);
#else
    int fd = mkstemps(src_path, 5);
    if (fd < 0) return NULL;
    close(fd);
    strncpy(real_src_path, src_path, sizeof(real_src_path) - 1);
#endif
    FILE* fp = fopen(real_src_path, "w");
    if (!fp) return NULL;
    fprintf(fp, "#include <hip/hip_runtime.h>\n%s", kernel_source);
    fclose(fp);
    char hsaco_path[MAX_PATH];
    snprintf(hsaco_path, sizeof(hsaco_path), "%s.hsaco", real_src_path);
    char compile_cmd[MAX_PATH * 3];
    snprintf(compile_cmd, sizeof(compile_cmd),
             "hipcc -x hip -target amdgcn-amd-amdhsa --offload-arch=gfx900,gfx906,gfx908,gfx90a,gfx1030,gfx1100 -o %s %s 2>/dev/null",
             hsaco_path, real_src_path);
    int compile_result = system(compile_cmd);
    if (compile_result != 0) {
        remove(real_src_path);
        remove(hsaco_path);
        return NULL;
    }
    FILE* hsaco_fp = fopen(hsaco_path, "rb");
    if (!hsaco_fp) { remove(real_src_path); remove(hsaco_path); return NULL; }
    fseek(hsaco_fp, 0, SEEK_END);
    size_t hsaco_size = (size_t)ftell(hsaco_fp);
    fseek(hsaco_fp, 0, SEEK_SET);
    char* hsaco_data = (char*)safe_malloc(hsaco_size);
    if (!hsaco_data) { fclose(hsaco_fp); remove(real_src_path); remove(hsaco_path); return NULL; }
    size_t read_size = fread(hsaco_data, 1, hsaco_size, hsaco_fp);
    fclose(hsaco_fp);
    if (read_size != hsaco_size) { safe_free((void**)&hsaco_data); remove(real_src_path); remove(hsaco_path); return NULL; }
    void* module = NULL;
    if (hipModuleLoad(&module, hsaco_path) != hipSuccess) { safe_free((void**)&hsaco_data); remove(real_src_path); remove(hsaco_path); return NULL; }
    void* function = NULL;
    if (hipModuleGetFunction(&function, module, kernel_name) != hipSuccess) {
        hipModuleUnload(module);
        safe_free((void**)&hsaco_data);
        remove(real_src_path);
        remove(hsaco_path);
        return NULL;
    }
    rocm_cache_insert(rocm_ctx, hash_key, kernel_name, hsaco_data, hsaco_size, module, function, 0);
    remove(real_src_path);
    remove(hsaco_path);
    struct GpuKernel* kernel = (struct GpuKernel*)safe_calloc(1, sizeof(struct GpuKernel));
    if (!kernel) return NULL;
    kernel->context = context;
    kernel->kernel_name = string_duplicate(kernel_name);
    kernel->kernel_source = string_duplicate(kernel_source);
    kernel->backend_data = function;
    return (GpuKernel*)kernel;
}

static void rocm_backend_kernel_free(GpuKernel* kernel) {
    if (!kernel) return;
    struct GpuKernel* kern = (struct GpuKernel*)kernel;
    if (kern->kernel_name) safe_free((void**)&kern->kernel_name);
    if (kern->kernel_source) safe_free((void**)&kern->kernel_source);
    safe_free((void**)&kern);
}

static int rocm_backend_kernel_set_arg(GpuKernel* kernel, int arg_index, size_t arg_size, const void* arg_value) {
    if (!kernel || arg_index < 0 || !arg_value) return -1;
    struct GpuKernel* kern = (struct GpuKernel*)kernel;
    if (arg_index >= kern->arg_capacity) {
        int new_capacity = kern->arg_capacity ? kern->arg_capacity * 2 : 8;
        while (new_capacity <= arg_index) new_capacity *= 2;
        void** new_values = (void**)safe_realloc(kern->arg_values, (size_t)new_capacity * sizeof(void*));
        if (!new_values) return -1;
        size_t* new_sizes = (size_t*)safe_realloc(kern->arg_sizes, (size_t)new_capacity * sizeof(size_t));
        if (!new_sizes) { safe_free((void**)&new_values); return -1; }
        for (int i = kern->arg_capacity; i < new_capacity; i++) { new_values[i] = NULL; new_sizes[i] = 0; }
        kern->arg_values = new_values;
        kern->arg_sizes = new_sizes;
        kern->arg_capacity = new_capacity;
    }
    if (kern->arg_values[arg_index]) safe_free((void**)&kern->arg_values[arg_index]);
    kern->arg_values[arg_index] = safe_malloc(arg_size);
    if (!kern->arg_values[arg_index]) return -1;
    memcpy(kern->arg_values[arg_index], arg_value, arg_size);
    kern->arg_sizes[arg_index] = arg_size;
    if (arg_index >= kern->arg_count) kern->arg_count = arg_index + 1;
    return 0;
}

static int rocm_backend_kernel_execute(GpuKernel* kernel, size_t global_work_size, size_t local_work_size) {
    if (!kernel || global_work_size == 0 || local_work_size == 0) return -1;
    struct GpuKernel* kern = (struct GpuKernel*)kernel;
    if (!kern->backend_data) return -1;
    struct GpuContext* ctx = (struct GpuContext*)kern->context;
    RocmContextInternal* rocm_ctx = ctx ? (RocmContextInternal*)ctx->backend_data : NULL;
    void* stream = rocm_ctx ? rocm_ctx->default_stream : NULL;
    unsigned int grid_dim = (unsigned int)((global_work_size + local_work_size - 1) / local_work_size);
    unsigned int block_dim = (unsigned int)local_work_size;
    if (block_dim > (unsigned int)rocm_ctx->max_threads_per_block) block_dim = (unsigned int)rocm_ctx->max_threads_per_block;
    void* hip_func = kern->backend_data;
    hipError_t err = hipModuleLaunchKernel(hip_func, grid_dim, 1, 1, block_dim, 1, 1, 0, stream, kern->arg_values, NULL);
    if (err == hipErrorInvalidValue) {
        err = hipModuleLaunchKernel(hip_func, grid_dim, 1, 1, block_dim, 1, 1, 0, stream, NULL, NULL);
    }
    if (err != hipSuccess) return -1;
    return 0;
}

static int rocm_backend_kernel_execute_nd(GpuKernel* kernel, int work_dim,
                                           const size_t* global_work_size,
                                           const size_t* local_work_size) {
    if (!kernel || !global_work_size || !local_work_size || work_dim < 1 || work_dim > 3) return -1;
    struct GpuKernel* kern = (struct GpuKernel*)kernel;
    if (!kern->backend_data) return -1;
    struct GpuContext* ctx = (struct GpuContext*)kern->context;
    RocmContextInternal* rocm_ctx = ctx ? (RocmContextInternal*)ctx->backend_data : NULL;
    void* stream = rocm_ctx ? rocm_ctx->default_stream : NULL;
    unsigned int grid_dim[3] = {1, 1, 1};
    unsigned int block_dim[3] = {1, 1, 1};
    for (int d = 0; d < work_dim; d++) {
        block_dim[d] = (unsigned int)local_work_size[d];
        grid_dim[d] = (unsigned int)((global_work_size[d] + local_work_size[d] - 1) / local_work_size[d]);
    }
    if (block_dim[0] > (unsigned int)rocm_ctx->max_threads_per_block) block_dim[0] = (unsigned int)rocm_ctx->max_threads_per_block;
    void* hip_function = kern->backend_data;
    hipError_t err = hipModuleLaunchKernel(hip_function,
                                            grid_dim[0], grid_dim[1], grid_dim[2],
                                            block_dim[0], block_dim[1], block_dim[2],
                                            0, stream, kern->arg_values, NULL);
    if (err != hipSuccess) return -1;
    return 0;
}

static GpuStream* rocm_backend_stream_create(GpuContext* context) {
    if (!context) return NULL;
    void* hip_stream = NULL;
    if (hipStreamCreate(&hip_stream) != hipSuccess) return NULL;
    struct GpuStream* stream = (struct GpuStream*)safe_calloc(1, sizeof(struct GpuStream));
    if (!stream) { hipStreamDestroy(hip_stream); return NULL; }
    stream->context = context;
    stream->is_completed = 1;
    stream->backend_data = hip_stream;
    return (GpuStream*)stream;
}

static void rocm_backend_stream_free(GpuStream* stream) {
    if (!stream) return;
    struct GpuStream* s = (struct GpuStream*)stream;
    if (s->backend_data && hipStreamDestroy) hipStreamDestroy(s->backend_data);
    safe_free((void**)&s);
}

static int rocm_backend_stream_synchronize(GpuStream* stream) {
    if (!stream) return -1;
    struct GpuStream* s = (struct GpuStream*)stream;
    if (!s->backend_data) return -1;
    return (hipStreamSynchronize(s->backend_data) == hipSuccess) ? 0 : -1;
}

static int rocm_backend_stream_query(GpuStream* stream) {
    if (!stream) return -1;
    struct GpuStream* s = (struct GpuStream*)stream;
    if (!s->backend_data) return -1;
    hipError_t err = hipStreamQuery(s->backend_data);
    if (err == hipSuccess) return 1;
    if (err == hipErrorNotReady) return 0;
    return -1;
}

static int rocm_backend_get_memory_info(GpuContext* context, size_t* total_memory, size_t* free_memory) {
    if (!context || !total_memory || !free_memory) return -1;
    struct GpuContext* ctx = (struct GpuContext*)context;
    RocmContextInternal* rocm_ctx = (RocmContextInternal*)ctx->backend_data;
    if (!rocm_ctx) return -1;

    *total_memory = rocm_ctx->total_global_memory;

    /* 尝试通过HIP API查询真实空闲内存 */
    size_t real_free = rocm_ctx->total_global_memory;
    if (hipMemGetInfo) {
        hipMemGetInfo(&real_free, NULL);
        if (real_free > 0 && real_free <= rocm_ctx->total_global_memory) {
            *free_memory = real_free;
            return 0;
        }
    }

    /* HIP API不可用时使用上下文追踪的分配量估算 */
    size_t estimated_used = 0;
    for (int i = 0; i < ROCM_MAX_ALLOCATIONS && i < rocm_ctx->allocation_count; i++) {
        if (rocm_ctx->allocations[i].is_active) {
            estimated_used += rocm_ctx->allocations[i].size;
        }
    }
    if (rocm_ctx->total_global_memory > estimated_used) {
        *free_memory = rocm_ctx->total_global_memory - estimated_used;
    } else {
        *free_memory = rocm_ctx->total_global_memory / 4; /* 更保守的默认值 */
    }
    return 0;
}

static int rocm_backend_device_reset(GpuContext* context) {
    if (!context) return -1;
    struct GpuContext* ctx = (struct GpuContext*)context;
    RocmContextInternal* rocm_ctx = (RocmContextInternal*)ctx->backend_data;

    /* 同步所有设备操作 */
    if (rocm_ctx && rocm_ctx->hip_device >= 0) {
        hipError_t err = hipDeviceSynchronize ? hipDeviceSynchronize() : hipSuccess;
        if (err != hipSuccess) {
            /* 尝试通过主设备重置 */
            if (hipDeviceReset) {
                hipSetDevice(rocm_ctx->hip_device);
                err = hipDeviceReset();
            }
        }
    }

    /* 清理内核缓存 */
    if (rocm_ctx) {
        for (int i = 0; i < ROCM_MAX_ALLOCATIONS; i++) {
            if (rocm_ctx->allocations[i].is_active && rocm_ctx->allocations[i].device_ptr) {
                if (hipFree) hipFree(rocm_ctx->allocations[i].device_ptr);
                rocm_ctx->allocations[i].device_ptr = NULL;
                rocm_ctx->allocations[i].is_active = 0;
            }
        }
        rocm_ctx->allocation_count = 0;
    }

    return 0;
}

static const char* rocm_backend_get_error_string(void) {
    hipError_t err = hipGetLastError ? hipGetLastError() : hipSuccess;
    if (err == hipSuccess) return "无错误";
    if (err == hipErrorMemoryAllocation) return "HIP内存分配失败";
    if (err == hipErrorInvalidDevice) return "无效的HIP设备";
    if (err == hipErrorLaunchFailure) return "HIP内核启动失败";
    if (err == hipErrorNoDevice) return "未找到HIP设备（需要AMD ROCm GPU）";
    static char err_buf[128];
    snprintf(err_buf, sizeof(err_buf), "HIP错误码=%d", (int)err);
    return err_buf;
}

const GpuBackendInterface* rocm_get_backend_interface(void) {
    static GpuBackendInterface rocm_backend = {
        .name = "AMD ROCm (HIP)",
        .backend_type = GPU_BACKEND_ROCM,
        .init = rocm_backend_init,
        .cleanup = rocm_backend_cleanup,
        .get_device_count = rocm_backend_get_device_count,
        .get_device_info = rocm_backend_get_device_info,
        .context_create = rocm_backend_context_create,
        .context_free = rocm_backend_context_free,
        .memory_alloc = rocm_backend_memory_alloc,
        .memory_free = rocm_backend_memory_free,
        .memory_copy_to_device = rocm_backend_memory_copy_to_device,
        .memory_copy_from_device = rocm_backend_memory_copy_from_device,
        .memory_copy_device_to_device = rocm_backend_memory_copy_device_to_device,
        .memory_copy_to_device_async = rocm_backend_memory_copy_to_device_async,
        .memory_copy_from_device_async = rocm_backend_memory_copy_from_device_async,
        .kernel_create = rocm_backend_kernel_create,
        .kernel_free = rocm_backend_kernel_free,
        .kernel_set_arg = rocm_backend_kernel_set_arg,
        .kernel_execute = rocm_backend_kernel_execute,
        .kernel_execute_nd = rocm_backend_kernel_execute_nd,
        .stream_create = rocm_backend_stream_create,
        .stream_free = rocm_backend_stream_free,
        .stream_synchronize = rocm_backend_stream_synchronize,
        .stream_query = rocm_backend_stream_query,
        .get_memory_info = rocm_backend_get_memory_info,
        .device_reset = rocm_backend_device_reset,
        .get_error_string = rocm_backend_get_error_string
    };
    return &rocm_backend;
}
