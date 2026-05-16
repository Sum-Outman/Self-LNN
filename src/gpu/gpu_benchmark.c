/**
 * @file gpu_benchmark.c
 * @brief GPU 基准测试和内存压力测试实现
 */

#include "selflnn/gpu/gpu_benchmark.h"
#include "selflnn/gpu/gpu.h"
#include "selflnn/utils/perf.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/gpu/gpu_memory_pool.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/secure_random.h"
#include "gpu_internal.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define GPU_BENCHMARK_VERSION "1.0.0"

#define GPU_BENCHMARK_MAX_BLOCKS 1024
#define GPU_BENCHMARK_MAX_BACKENDS 16
#define GPU_BENCHMARK_REPORT_SIZE (64 * 1024)

/* 多后端内核源码模板（根据后端类型选择语法） */
static const char* benchmark_matmul_kernel(GpuBackend backend) {
    switch (backend) {
        case GPU_BACKEND_CUDA:
            return "extern \"C\" __global__ void matmul(const float* a, const float* b, float* c, int n) {"
                   "  int row=blockIdx.y*blockDim.y+threadIdx.y; int col=blockIdx.x*blockDim.x+threadIdx.x;"
                   "  if(row>=n||col>=n) return; float sum=0;"
                   "  for(int k=0;k<n;k++) sum+=a[row*n+k]*b[k*n+col];"
                   "  c[row*n+col]=sum; }";
        case GPU_BACKEND_VULKAN:
            return "#version 450\nlayout(local_size_x=16,local_size_y=16) in;\n"
                   "layout(binding=0) buffer A{float a[];}; layout(binding=1) buffer B{float b[];};"
                   " layout(binding=2) buffer C{float c[];}; layout(push_constant) uniform Params{int n;};"
                   " void main(){uint r=gl_GlobalInvocationID.y;uint c_idx=gl_GlobalInvocationID.x;"
                   " if(r>=n||c_idx>=n)return;float s=0;"
                   " for(uint k=0;k<n;k++) s+=a[r*n+k]*b[k*n+c_idx];c[r*n+c_idx]=s;}";
        case GPU_BACKEND_METAL:
            return "#include <metal_stdlib>\nusing namespace metal;\n"
                   "kernel void matmul(device const float* a, device const float* b, device float* c,"
                   " constant int& n, uint2 gid[[thread_position_in_grid]]){"
                   " uint r=gid.y;uint col=gid.x; if(r>=n||col>=n)return;"
                   " float s=0;for(uint k=0;k<n;k++)s+=a[r*n+k]*b[k*n+col];c[r*n+col]=s;}";
        case GPU_BACKEND_OPENCL:
        case GPU_BACKEND_ROCM:
        default:
            return "__kernel void matmul(__global const float* a, __global const float* b, __global float* c, int n) {"
                   " int row=get_global_id(0); int col=get_global_id(1);"
                   " if(row>=n||col>=n) return; float sum=0;"
                   " for(int k=0;k<n;k++) sum+=a[row*n+k]*b[k*n+col];"
                   " c[row*n+col]=sum; }";
    }
}

static const char* benchmark_conv_kernel(GpuBackend backend) {
    switch (backend) {
        case GPU_BACKEND_CUDA:
            return "extern \"C\" __global__ void conv2d(const float* in, const float* w, float* out, int C, int H, int W, int K, int ksize) {"
                   " int oc=blockIdx.x*blockDim.x+threadIdx.x;int oh=blockIdx.y*blockDim.y+threadIdx.y;int ow=blockIdx.z*blockDim.z+threadIdx.z;"
                   " if(oc>=K||oh>=H||ow>=W)return;float s=0;"
                   " for(int ic=0;ic<C;ic++)for(int kh=0;kh<ksize;kh++)for(int kw=0;kw<ksize;kw++){"
                   " int ih=oh+kh-ksize/2;int iw=ow+kw-ksize/2;"
                   " if(ih>=0&&ih<H&&iw>=0&&iw<W)s+=in[(ic*H+ih)*W+iw]*w[((oc*C+ic)*ksize+kh)*ksize+kw];}out[(oc*H+oh)*W+ow]=s;}";
        case GPU_BACKEND_OPENCL:
        case GPU_BACKEND_ROCM:
        default:
            return "__kernel void conv2d(__global const float* in, __global const float* w,"
                   " __global float* out, int C, int H, int W, int K, int ksize) {"
                   " int oc=get_global_id(0);int oh=get_global_id(1);int ow=get_global_id(2);"
                   " if(oc>=K||oh>=H||ow>=W)return;float s=0;"
                   " for(int ic=0;ic<C;ic++)for(int kh=0;kh<ksize;kh++)for(int kw=0;kw<ksize;kw++){"
                   " int ih=oh+kh-ksize/2;int iw=ow+kw-ksize/2;"
                   " if(ih>=0&&ih<H&&iw>=0&&iw<W)s+=in[(ic*H+ih)*W+iw]*w[((oc*C+ic)*ksize+kh)*ksize+kw];}out[(oc*H+oh)*W+ow]=s;}";
    }
}
#define GPU_BENCHMARK_JSON_SIZE (32 * 1024)

static double gpu_benchmark_get_time_s(void)
{
    uint64_t ns = perf_timestamp_ns();
    return (double)ns / 1.0e9;
}

static double gpu_benchmark_elapsed_s(double start_time)
{
    return gpu_benchmark_get_time_s() - start_time;
}

GpuBenchmarkConfig gpu_benchmark_config_default(void)
{
    GpuBenchmarkConfig cfg;
    cfg.enable_memory_pressure = 1;
    cfg.enable_compute_benchmark = 1;
    cfg.enable_bandwidth_benchmark = 1;
    cfg.enable_latency_benchmark = 1;
    cfg.memory_pressure_step_ratio = 0.05;
    cfg.memory_pressure_max_ratio = 0.95;
    cfg.memory_pressure_min_blocks = 4;
    cfg.memory_pressure_iterations = 3;
    cfg.compute_matrix_sizes[0] = 512;
    cfg.compute_matrix_sizes[1] = 1024;
    cfg.compute_matrix_sizes[2] = 2048;
    cfg.compute_matrix_sizes[3] = 4096;
    cfg.compute_iterations = 10;
    cfg.compute_warmup = 3;
    cfg.bandwidth_min_bytes = 1024 * 1024;
    cfg.bandwidth_max_bytes = 1024 * 1024 * 1024;
    cfg.bandwidth_iterations = 10;
    cfg.latency_iterations = 100;
    cfg.latency_transfer_bytes = 64;
    cfg.verbose = 0;
    cfg.timeout_seconds = 300.0;
    return cfg;
}

double gpu_estimate_theoretical_gflops(const GpuDeviceInfo* info, int precision_bits)
{
    if (!info || info->compute_units <= 0 || info->clock_speed <= 0.0f)
        return 0.0;

    double ops_per_cycle_per_unit = 0.0;
    int vector_width = 1;

    if (info->type == GPU_DEVICE_TYPE_CPU) {
        unsigned int simd = info->simd_flags;
        if (simd & CPU_SIMD_AVX512F) {
            vector_width = (precision_bits == 64) ? 8 : 16;
        } else if (simd & CPU_SIMD_AVX2) {
            vector_width = (precision_bits == 64) ? 4 : 8;
        } else if (simd & CPU_SIMD_AVX) {
            vector_width = (precision_bits == 64) ? 4 : 8;
        } else if (simd & CPU_SIMD_SSE2) {
            vector_width = (precision_bits == 64) ? 2 : 4;
        } else if (simd & CPU_SIMD_NEON) {
            vector_width = (precision_bits == 64) ? 2 : 4;
        }
        ops_per_cycle_per_unit = (precision_bits == 32) ? 8.0 : 4.0;
    } else {
        if (precision_bits == 16) {
            ops_per_cycle_per_unit = 64.0;
        } else if (precision_bits == 32) {
            ops_per_cycle_per_unit = 32.0;
        } else if (precision_bits == 64) {
            ops_per_cycle_per_unit = (info->supports_double) ? 8.0 : 1.0;
        } else {
            ops_per_cycle_per_unit = 32.0;
        }
        vector_width = 1;
    }

    double clock_ghz = (double)info->clock_speed / 1000.0;
    double gflops = (double)info->compute_units * clock_ghz * ops_per_cycle_per_unit * (double)vector_width;

    if (gflops < 0.1) gflops = 0.1;
    return gflops;
}

double gpu_estimate_theoretical_bandwidth(const GpuDeviceInfo* info)
{
    if (!info) return 0.0;

    if (info->type == GPU_DEVICE_TYPE_CPU) {
        if (info->l3_cache > 0) {
            return (double)info->clock_speed * 16.0 * (double)info->logical_cores / 1000.0;
        }
        if (info->l2_cache > 0) {
            return (double)info->clock_speed * 8.0 * (double)info->logical_cores / 1000.0;
        }
        return (double)info->clock_speed * 4.0 * (double)info->logical_cores / 1000.0;
    }

    /* 真实带宽测量：通过GPU内存分配+memcpy+计时获取实际带宽 */
    size_t test_size = (info->total_memory > 0) ? info->total_memory / 256 : (256ULL * 1024 * 1024);
    if (test_size < 4 * 1024 * 1024) test_size = 4 * 1024 * 1024;
    if (test_size > 256 * 1024 * 1024) test_size = 256 * 1024 * 1024;
    size_t count = test_size / sizeof(float);
    float* host_src = (float*)safe_calloc(count, sizeof(float));
    float* host_dst = (float*)safe_calloc(count, sizeof(float));
    if (!host_src || !host_dst) {
        safe_free((void**)&host_src);
        safe_free((void**)&host_dst);
        goto fallback_estimate;
    }
    for (size_t i = 0; i < count; i++) host_src[i] = (float)(i & 0xFF) / 255.0f;

    /* 尝试使用CPU后端进行真实带宽测量（始终可用） */
    GpuContext* ctx = gpu_context_create(GPU_BACKEND_CPU, 0);
    if (!ctx) {
        safe_free((void**)&host_src);
        safe_free((void**)&host_dst);
        goto fallback_estimate;
    }

    GpuMemory* dev_buf = gpu_memory_alloc(ctx, test_size, GPU_MEMORY_DEVICE);
    if (!dev_buf) {
        gpu_context_free(ctx);
        safe_free((void**)&host_src);
        safe_free((void**)&host_dst);
        goto fallback_estimate;
    }

    /* 预热：执行3次避免冷启动影响 */
    for (int w = 0; w < 3; w++) {
        gpu_memory_copy_to_device(dev_buf, host_src, test_size);
        gpu_memory_copy_from_device(host_dst, dev_buf, test_size);
    }

    /* 实测：执行5次取中位数 */
    const int trials = 5;
    double times[5];
    for (int t = 0; t < trials; t++) {
        uint64_t t0 = perf_timestamp_ns();
        gpu_memory_copy_to_device(dev_buf, host_src, test_size);
        gpu_memory_copy_from_device(host_dst, dev_buf, test_size);
        uint64_t t1 = perf_timestamp_ns();
        times[t] = (double)(t1 - t0) / 1e9;
    }
    /* 排序取中位数 */
    for (int i = 0; i < trials - 1; i++)
        for (int j = i + 1; j < trials; j++)
            if (times[i] > times[j]) { double tmp = times[i]; times[i] = times[j]; times[j] = tmp; }
    double elapsed = times[2];

    double bandwidth_gbs = (elapsed > 0.0) ? (2.0 * (double)test_size / elapsed) / 1e9 : 0.0;

    gpu_memory_free(dev_buf);
    gpu_context_free(ctx);
    safe_free((void**)&host_src);
    safe_free((void**)&host_dst);

    if (bandwidth_gbs > 0.0) return bandwidth_gbs;

fallback_estimate:
    /* 实测不可用时，基于总显存大小的保守估算（不使用名称字符串匹配） */
    size_t memory_mb = info->total_memory / (1024 * 1024);
    if (memory_mb >= 24576)        return 600.0;
    else if (memory_mb >= 16384)  return 400.0;
    else if (memory_mb >= 8192)   return 200.0;
    else if (memory_mb >= 4096)   return 80.0;
    else if (memory_mb >= 2048)   return 40.0;
    else if (memory_mb >= 1024)   return 20.0;
    else                          return 10.0;
}

const char* gpu_benchmark_version(void)
{
    return GPU_BENCHMARK_VERSION;
}

int gpu_run_memory_pressure_test(GpuContext* context,
                                 const GpuBenchmarkConfig* config,
                                 GpuMemoryPressureResult* result)
{
    if (!context || !result) return -1;

    GpuBenchmarkConfig cfg = config ? *config : gpu_benchmark_config_default();
    memset(result, 0, sizeof(GpuMemoryPressureResult));

    size_t total_memory = 0, free_memory = 0;
    if (gpu_get_memory_info(context, &total_memory, &free_memory) != 0) {
        strncpy(result->error_message, "获取内存信息失败", sizeof(result->error_message) - 1);
        return -1;
    }
    result->total_memory = total_memory;
    if (total_memory == 0) {
        strncpy(result->error_message, "设备总内存为0", sizeof(result->error_message) - 1);
        return -1;
    }

    double start_time = gpu_benchmark_get_time_s();

    size_t step_size = (size_t)((double)total_memory * cfg.memory_pressure_step_ratio);
    if (step_size < 1024) step_size = 1024;
    size_t max_allocation = (size_t)((double)total_memory * cfg.memory_pressure_max_ratio);

    size_t peak_allocated = 0;
    size_t max_single = 0;
    int success_count = 0;
    int fail_count = 0;
    double total_alloc_time_ms = 0.0;

    GpuMemory* blocks[GPU_BENCHMARK_MAX_BLOCKS];
    size_t block_sizes[GPU_BENCHMARK_MAX_BLOCKS];
    int block_count = 0;

    int iteration;
    for (iteration = 0; iteration < cfg.memory_pressure_iterations; iteration++) {
        block_count = 0;
        size_t current_allocated = 0;

        size_t alloc_size = step_size;

        while (current_allocated < max_allocation && block_count < GPU_BENCHMARK_MAX_BLOCKS) {
            size_t remaining = max_allocation - current_allocated;
            size_t this_alloc = (alloc_size < remaining) ? alloc_size : remaining;

            if (this_alloc < 64) this_alloc = 64;

            PerfTimer timer;
            perf_timer_init(&timer);
            perf_timer_start(&timer);

            GpuMemory* mem = gpu_memory_alloc(context, this_alloc, GPU_MEMORY_DEVICE);

            uint64_t alloc_time_ns = perf_timer_stop(&timer);
            double alloc_time_ms = (double)alloc_time_ns / 1.0e6;
            total_alloc_time_ms += alloc_time_ms;

            if (mem) {
                blocks[block_count] = mem;
                block_sizes[block_count] = this_alloc;
                block_count++;
                current_allocated += this_alloc;
                success_count++;

                if (this_alloc > max_single) {
                    max_single = this_alloc;
                }

                unsigned char* test_pattern = (unsigned char*)safe_malloc(this_alloc > 4096 ? 4096 : this_alloc);
                if (test_pattern) {
                    size_t test_size = this_alloc > 4096 ? 4096 : this_alloc;
                    memset(test_pattern, 0x5A, test_size);
                    gpu_memory_copy_to_device(mem, test_pattern, test_size);
                    safe_free((void**)&test_pattern);
                }

                if (gpu_benchmark_elapsed_s(start_time) > cfg.timeout_seconds) {
                    result->overflow_detected = 1;
                    break;
                }

                alloc_size = (size_t)((double)alloc_size * 1.2);
                if (alloc_size < step_size) alloc_size = step_size;
            } else {
                fail_count++;
                if (alloc_size > 4096) {
                    alloc_size = alloc_size / 2;
                    if (alloc_size < 64) alloc_size = 64;
                } else {
                    break;
                }
                if (fail_count > 64) break;
            }
        }

        if (current_allocated > peak_allocated) {
            peak_allocated = current_allocated;
        }

        int i;
        for (i = 0; i < block_count; i++) {
            if (blocks[i]) {
                gpu_memory_free(blocks[i]);
                blocks[i] = NULL;
            }
        }
        block_count = 0;

        if (gpu_benchmark_elapsed_s(start_time) > cfg.timeout_seconds) {
            result->overflow_detected = 1;
            break;
        }
    }

    result->peak_allocated = peak_allocated;
    result->max_single_allocation = max_single;
    result->allocation_count = success_count;
    result->total_test_time_ms = gpu_benchmark_elapsed_s(start_time) * 1000.0;

    int total_attempts = success_count + fail_count;
    if (total_attempts > 0) {
        result->allocation_success_rate = (double)success_count / (double)total_attempts;
    } else {
        result->allocation_success_rate = 0.0;
    }

    if (total_memory > 0) {
        result->peak_utilization_ratio = (double)peak_allocated / (double)total_memory;
    }

    if (success_count > 0) {
        result->average_allocation_time_ms = total_alloc_time_ms / (double)success_count;
    }

    return 0;
}

static int benchmark_matrix_multiply(GpuContext* context, int matrix_size,
                                     int iterations, int warmup,
                                     double* out_gflops, double* out_time_ms)
{
    size_t n = (size_t)matrix_size;
    size_t mat_bytes = n * n * sizeof(float);

    float* h_a = (float*)safe_malloc(mat_bytes);
    float* h_b = (float*)safe_malloc(mat_bytes);
    float* h_c = (float*)safe_malloc(mat_bytes);
    if (!h_a || !h_b || !h_c) {
        safe_free((void**)&h_a);
        safe_free((void**)&h_b);
        safe_free((void**)&h_c);
        return -1;
    }

    size_t i, j;
    for (i = 0; i < n * n; i++) {
        h_a[i] = secure_random_float() * 2.0f - 1.0f;
        h_b[i] = secure_random_float() * 2.0f - 1.0f;
    }

    GpuMemory* d_a = gpu_memory_alloc(context, mat_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_b = gpu_memory_alloc(context, mat_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_c = gpu_memory_alloc(context, mat_bytes, GPU_MEMORY_DEVICE);

    if (!d_a || !d_b || !d_c) {
        if (d_a) gpu_memory_free(d_a);
        if (d_b) gpu_memory_free(d_b);
        if (d_c) gpu_memory_free(d_c);
        safe_free((void**)&h_a);
        safe_free((void**)&h_b);
        safe_free((void**)&h_c);
        return -1;
    }

    gpu_memory_copy_to_device(d_a, h_a, mat_bytes);
    gpu_memory_copy_to_device(d_b, h_b, mat_bytes);

    const char* matmul_kernel = benchmark_matmul_kernel(context->backend);

    GpuKernel* kernel = gpu_kernel_create(context, matmul_kernel, "matmul");
    if (!kernel) {
        gpu_memory_free(d_a);
        gpu_memory_free(d_b);
        gpu_memory_free(d_c);
        safe_free((void**)&h_a);
        safe_free((void**)&h_b);
        safe_free((void**)&h_c);
        return -1;
    }

    gpu_kernel_set_arg(kernel, 0, sizeof(GpuMemory*), &d_a);
    gpu_kernel_set_arg(kernel, 1, sizeof(GpuMemory*), &d_b);
    gpu_kernel_set_arg(kernel, 2, sizeof(GpuMemory*), &d_c);
    gpu_kernel_set_arg(kernel, 3, sizeof(int), &matrix_size);

    size_t global_size[2] = {n, n};
    size_t local_size[2] = {16, 16};

    int w;
    for (w = 0; w < warmup; w++) {
        gpu_kernel_execute_nd(kernel, 2, global_size, local_size);
    }

    PerfTimer timer;
    perf_timer_init(&timer);
    perf_timer_start(&timer);

    int iter;
    for (iter = 0; iter < iterations; iter++) {
        gpu_kernel_execute_nd(kernel, 2, global_size, local_size);
    }

    uint64_t total_ns = perf_timer_stop(&timer);
    double total_ms = (double)total_ns / 1.0e6;

    gpu_memory_copy_from_device(h_c, d_c, mat_bytes);

    double ops = 2.0 * (double)n * (double)n * (double)n;
    *out_gflops = (ops * (double)iterations) / (total_ns / 1.0e9) / 1.0e9;
    *out_time_ms = total_ms / (double)iterations;

    gpu_kernel_free(kernel);
    gpu_memory_free(d_a);
    gpu_memory_free(d_b);
    gpu_memory_free(d_c);
    safe_free((void**)&h_a);
    safe_free((void**)&h_b);
    safe_free((void**)&h_c);

    return 0;
}

static int benchmark_convolution(GpuContext* context, int input_size,
                                 int iterations, int warmup,
                                 double* out_gflops, double* out_time_ms)
{
    int in_channels = 64;
    int out_channels = 64;
    int kernel_size = 3;
    int height = input_size;
    int width = input_size;

    size_t input_bytes = (size_t)in_channels * (size_t)height * (size_t)width * sizeof(float);
    size_t weight_bytes = (size_t)out_channels * (size_t)in_channels * (size_t)kernel_size * (size_t)kernel_size * sizeof(float);
    size_t output_bytes = (size_t)out_channels * (size_t)height * (size_t)width * sizeof(float);

    float* h_input = (float*)safe_malloc(input_bytes);
    float* h_weight = (float*)safe_malloc(weight_bytes);
    if (!h_input || !h_weight) {
        safe_free((void**)&h_input);
        safe_free((void**)&h_weight);
        return -1;
    }

    size_t i;
    for (i = 0; i < input_bytes / sizeof(float); i++) {
        h_input[i] = secure_random_float() * 2.0f - 1.0f;
    }
    for (i = 0; i < weight_bytes / sizeof(float); i++) {
        h_weight[i] = secure_random_float() * 0.1f;
    }

    GpuMemory* d_input = gpu_memory_alloc(context, input_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_weight = gpu_memory_alloc(context, weight_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_output = gpu_memory_alloc(context, output_bytes, GPU_MEMORY_DEVICE);

    if (!d_input || !d_weight || !d_output) {
        if (d_input) gpu_memory_free(d_input);
        if (d_weight) gpu_memory_free(d_weight);
        if (d_output) gpu_memory_free(d_output);
        safe_free((void**)&h_input);
        safe_free((void**)&h_weight);
        return -1;
    }

    gpu_memory_copy_to_device(d_input, h_input, input_bytes);
    gpu_memory_copy_to_device(d_weight, h_weight, weight_bytes);

    const char* conv_kernel = benchmark_conv_kernel(context->backend);

    GpuKernel* kernel = gpu_kernel_create(context, conv_kernel, "conv2d");
    if (!kernel) {
        gpu_memory_free(d_input);
        gpu_memory_free(d_weight);
        gpu_memory_free(d_output);
        safe_free((void**)&h_input);
        safe_free((void**)&h_weight);
        return -1;
    }

    gpu_kernel_set_arg(kernel, 0, sizeof(GpuMemory*), &d_input);
    gpu_kernel_set_arg(kernel, 1, sizeof(GpuMemory*), &d_weight);
    gpu_kernel_set_arg(kernel, 2, sizeof(GpuMemory*), &d_output);
    gpu_kernel_set_arg(kernel, 3, sizeof(int), &in_channels);
    gpu_kernel_set_arg(kernel, 4, sizeof(int), &height);
    gpu_kernel_set_arg(kernel, 5, sizeof(int), &width);
    gpu_kernel_set_arg(kernel, 6, sizeof(int), &out_channels);
    gpu_kernel_set_arg(kernel, 7, sizeof(int), &kernel_size);

    size_t global_size[3] = {(size_t)out_channels, (size_t)height, (size_t)width};
    size_t local_size[3] = {4, 8, 8};

    int w;
    for (w = 0; w < warmup; w++) {
        gpu_kernel_execute_nd(kernel, 3, global_size, local_size);
    }

    PerfTimer timer;
    perf_timer_init(&timer);
    perf_timer_start(&timer);

    int iter;
    for (iter = 0; iter < iterations; iter++) {
        gpu_kernel_execute_nd(kernel, 3, global_size, local_size);
    }

    uint64_t total_ns = perf_timer_stop(&timer);
    double total_ms = (double)total_ns / 1.0e6;

    double ops = 2.0 * (double)out_channels * (double)in_channels * (double)kernel_size * (double)kernel_size
                 * (double)height * (double)width;
    *out_gflops = (ops * (double)iterations) / (total_ns / 1.0e9) / 1.0e9;
    *out_time_ms = total_ms / (double)iterations;

    gpu_kernel_free(kernel);
    gpu_memory_free(d_input);
    gpu_memory_free(d_weight);
    gpu_memory_free(d_output);
    safe_free((void**)&h_input);
    safe_free((void**)&h_weight);

    return 0;
}

static int benchmark_elementwise(GpuContext* context, size_t num_elements,
                                 int iterations, int warmup,
                                 double* out_gflops, double* out_time_ms)
{
    size_t bytes = num_elements * sizeof(float);

    float* h_a = (float*)safe_malloc(bytes);
    float* h_b = (float*)safe_malloc(bytes);
    float* h_c = (float*)safe_malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        safe_free((void**)&h_a);
        safe_free((void**)&h_b);
        safe_free((void**)&h_c);
        return -1;
    }

    size_t i;
    for (i = 0; i < num_elements; i++) {
        h_a[i] = secure_random_float();
        h_b[i] = secure_random_float();
    }

    GpuMemory* d_a = gpu_memory_alloc(context, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_b = gpu_memory_alloc(context, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_c = gpu_memory_alloc(context, bytes, GPU_MEMORY_DEVICE);

    if (!d_a || !d_b || !d_c) {
        if (d_a) gpu_memory_free(d_a);
        if (d_b) gpu_memory_free(d_b);
        if (d_c) gpu_memory_free(d_c);
        safe_free((void**)&h_a);
        safe_free((void**)&h_b);
        safe_free((void**)&h_c);
        return -1;
    }

    gpu_memory_copy_to_device(d_a, h_a, bytes);
    gpu_memory_copy_to_device(d_b, h_b, bytes);

    const char* ew_kernel =
        "__kernel void elementwise(__global const float* a, __global const float* b, __global float* c, int n) {"
        "    int i = get_global_id(0);"
        "    if (i >= n) return;"
        "    float x = a[i];"
        "    float y = b[i];"
        "    c[i] = x * y + sin(x) * cos(y) + exp(-x * x) * tanh(y);"
        "}";

    GpuKernel* kernel = gpu_kernel_create(context, ew_kernel, "elementwise");
    if (!kernel) {
        gpu_memory_free(d_a);
        gpu_memory_free(d_b);
        gpu_memory_free(d_c);
        safe_free((void**)&h_a);
        safe_free((void**)&h_b);
        safe_free((void**)&h_c);
        return -1;
    }

    size_t global_size = num_elements;
    size_t local_size = 256;

    gpu_kernel_set_arg(kernel, 0, sizeof(GpuMemory*), &d_a);
    gpu_kernel_set_arg(kernel, 1, sizeof(GpuMemory*), &d_b);
    gpu_kernel_set_arg(kernel, 2, sizeof(GpuMemory*), &d_c);
    gpu_kernel_set_arg(kernel, 3, sizeof(int), &num_elements);

    int w;
    for (w = 0; w < warmup; w++) {
        gpu_kernel_execute(kernel, global_size, local_size);
    }

    PerfTimer timer;
    perf_timer_init(&timer);
    perf_timer_start(&timer);

    int iter;
    for (iter = 0; iter < iterations; iter++) {
        gpu_kernel_execute(kernel, global_size, local_size);
    }

    uint64_t total_ns = perf_timer_stop(&timer);
    double total_ms = (double)total_ns / 1.0e6;

    double ops_per_element = 10.0;
    double total_ops = ops_per_element * (double)num_elements * (double)iterations;
    *out_gflops = total_ops / ((double)total_ns / 1.0e9) / 1.0e9;
    *out_time_ms = total_ms / (double)iterations;

    gpu_kernel_free(kernel);
    gpu_memory_free(d_a);
    gpu_memory_free(d_b);
    gpu_memory_free(d_c);
    safe_free((void**)&h_a);
    safe_free((void**)&h_b);
    safe_free((void**)&h_c);

    return 0;
}

int gpu_run_compute_benchmark(GpuContext* context,
                              const GpuBenchmarkConfig* config,
                              GpuComputeBenchmarkResult* result)
{
    if (!context || !result) return -1;

    GpuBenchmarkConfig cfg = config ? *config : gpu_benchmark_config_default();
    memset(result, 0, sizeof(GpuComputeBenchmarkResult));

    result->precision_bits = 32;
    strncpy(result->description, "单精度浮点计算性能", sizeof(result->description) - 1);

    double best_gflops = 0.0;
    double total_gflops = 0.0;
    int valid_count = 0;
    double total_time_ms = 0.0;

    int s;
    for (s = 0; s < 4; s++) {
        int matrix_size = cfg.compute_matrix_sizes[s];
        if (matrix_size <= 0) continue;

        double gflops = 0.0, time_ms = 0.0;
        if (benchmark_matrix_multiply(context, matrix_size, cfg.compute_iterations, cfg.compute_warmup, &gflops, &time_ms) == 0) {
            total_gflops += gflops;
            total_time_ms += time_ms;
            valid_count++;
            if (gflops > best_gflops) best_gflops = gflops;

            if (gflops > result->matrix_mul_gflops) {
                result->matrix_mul_gflops = gflops;
                result->matrix_mul_time_ms = time_ms;
            }
        }
    }

    {
        double gflops = 0.0, time_ms = 0.0;
        if (benchmark_convolution(context, 128, cfg.compute_iterations, cfg.compute_warmup, &gflops, &time_ms) == 0) {
            result->convolution_gflops = gflops;
            result->convolution_time_ms = time_ms;
            total_gflops += gflops;
            total_time_ms += time_ms;
            valid_count++;
            if (gflops > best_gflops) best_gflops = gflops;
        }
    }

    {
        double gflops = 0.0, time_ms = 0.0;
        if (benchmark_elementwise(context, 1024 * 1024 * 16, cfg.compute_iterations, cfg.compute_warmup, &gflops, &time_ms) == 0) {
            result->elementwise_gflops = gflops;
            result->elementwise_time_ms = time_ms;
            total_gflops += gflops;
            total_time_ms += time_ms;
            valid_count++;
            if (gflops > best_gflops) best_gflops = gflops;
        }
    }

    if (valid_count > 0) {
        result->average_gflops = total_gflops / (double)valid_count;
    }

    result->memory_bound_gflops = result->elementwise_gflops;

    return 0;
}

int gpu_run_bandwidth_benchmark(GpuContext* context,
                                const GpuBenchmarkConfig* config,
                                GpuBandwidthBenchmarkResult* result)
{
    if (!context || !result) return -1;

    GpuBenchmarkConfig cfg = config ? *config : gpu_benchmark_config_default();
    memset(result, 0, sizeof(GpuBandwidthBenchmarkResult));

    result->iteration_count = cfg.bandwidth_iterations;
    strncpy(result->description, "GPU内存带宽基准测试", sizeof(result->description) - 1);

    size_t sizes[] = {
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    double best_h2d_bw = 0.0, best_d2h_bw = 0.0, best_d2d_bw = 0.0;
    double total_h2d_time = 0.0, total_d2h_time = 0.0, total_d2d_time = 0.0;
    int h2d_count = 0, d2h_count = 0, d2d_count = 0;

    int si;
    for (si = 0; si < num_sizes; si++) {
        size_t size = sizes[si];
        if (size > cfg.bandwidth_max_bytes) break;
        if (size < cfg.bandwidth_min_bytes) continue;

        unsigned char* h_buf = (unsigned char*)safe_malloc(size);
        if (!h_buf) continue;

        memset(h_buf, 0xAB, size);

        GpuMemory* d_buf = gpu_memory_alloc(context, size, GPU_MEMORY_DEVICE);
        if (!d_buf) {
            safe_free((void**)&h_buf);
            continue;
        }

        int iter;
        for (iter = 0; iter < cfg.bandwidth_iterations; iter++) {
            PerfTimer timer;
            perf_timer_init(&timer);
            perf_timer_start(&timer);
            gpu_memory_copy_to_device(d_buf, h_buf, size);
            uint64_t h2d_ns = perf_timer_stop(&timer);

            double h2d_bw = (double)size / ((double)h2d_ns / 1.0e9) / (1024.0 * 1024.0 * 1024.0);
            if (h2d_bw > best_h2d_bw) {
                best_h2d_bw = h2d_bw;
                result->host_to_device_bandwidth_gbps = best_h2d_bw;
                result->transfer_size = size;
            }
            total_h2d_time += (double)h2d_ns / 1.0e6;
            h2d_count++;

            memset(h_buf, 0, size);

            perf_timer_init(&timer);
            perf_timer_start(&timer);
            gpu_memory_copy_from_device(h_buf, d_buf, size);
            uint64_t d2h_ns = perf_timer_stop(&timer);

            double d2h_bw = (double)size / ((double)d2h_ns / 1.0e9) / (1024.0 * 1024.0 * 1024.0);
            if (d2h_bw > best_d2h_bw) {
                best_d2h_bw = d2h_bw;
                result->device_to_host_bandwidth_gbps = best_d2h_bw;
            }
            total_d2h_time += (double)d2h_ns / 1.0e6;
            d2h_count++;
        }

        {
            GpuMemory* d_buf2 = gpu_memory_alloc(context, size, GPU_MEMORY_DEVICE);
            if (d_buf2) {
                int iter2;
                for (iter2 = 0; iter2 < cfg.bandwidth_iterations; iter2++) {
                    gpu_memory_copy_to_device(d_buf, h_buf, size);

                    PerfTimer timer;
                    perf_timer_init(&timer);
                    perf_timer_start(&timer);
                    gpu_memory_copy_device_to_device(d_buf2, d_buf, size);
                    uint64_t d2d_ns = perf_timer_stop(&timer);

                    double d2d_bw = (double)size / ((double)d2d_ns / 1.0e9) / (1024.0 * 1024.0 * 1024.0);
                    if (d2d_bw > best_d2d_bw) {
                        best_d2d_bw = d2d_bw;
                        result->device_to_device_bandwidth_gbps = best_d2d_bw;
                    }
                    total_d2d_time += (double)d2d_ns / 1.0e6;
                    d2d_count++;
                }
                gpu_memory_free(d_buf2);
            }
        }

        gpu_memory_free(d_buf);
        safe_free((void**)&h_buf);
    }

    if (h2d_count > 0) {
        result->host_to_device_time_ms = total_h2d_time / (double)h2d_count;
    }
    if (d2h_count > 0) {
        result->device_to_host_time_ms = total_d2h_time / (double)d2h_count;
    }
    if (d2d_count > 0) {
        result->device_to_device_time_ms = total_d2d_time / (double)d2d_count;
    }

    if (result->host_to_device_bandwidth_gbps < 0.001)
        result->host_to_device_bandwidth_gbps = 0.0;
    if (result->device_to_host_bandwidth_gbps < 0.001)
        result->device_to_host_bandwidth_gbps = 0.0;
    if (result->device_to_device_bandwidth_gbps < 0.001)
        result->device_to_device_bandwidth_gbps = 0.0;

    return 0;
}

static int benchmark_kernel_latency(GpuContext* context, int iterations, double* out_latency_us)
{
    const char* null_kernel =
        "__kernel void null_kernel() {"
        "    return;"
        "}";

    GpuKernel* kernel = gpu_kernel_create(context, null_kernel, "null_kernel");
    if (!kernel) return -1;

    size_t global_size = 1;
    size_t local_size = 1;

    int i;
    for (i = 0; i < 10; i++) {
        gpu_kernel_execute(kernel, global_size, local_size);
    }

    PerfTimer timer;
    perf_timer_init(&timer);
    perf_timer_start(&timer);

    for (i = 0; i < iterations; i++) {
        gpu_kernel_execute(kernel, global_size, local_size);
    }

    uint64_t total_ns = perf_timer_stop(&timer);

    *out_latency_us = (double)total_ns / (double)iterations / 1000.0;

    gpu_kernel_free(kernel);
    return 0;
}

static int benchmark_memory_latency(GpuContext* context, size_t transfer_bytes,
                                    int iterations, double* out_latency_us)
{
    if (transfer_bytes < 4) transfer_bytes = 4;

    unsigned char* h_buf = (unsigned char*)safe_malloc(transfer_bytes);
    if (!h_buf) return -1;

    GpuMemory* d_buf = gpu_memory_alloc(context, transfer_bytes, GPU_MEMORY_DEVICE);
    if (!d_buf) {
        safe_free((void**)&h_buf);
        return -1;
    }

    memset(h_buf, 0xAB, transfer_bytes);

    int i;
    for (i = 0; i < 10; i++) {
        gpu_memory_copy_to_device(d_buf, h_buf, transfer_bytes);
    }

    PerfTimer timer;
    perf_timer_init(&timer);
    perf_timer_start(&timer);

    for (i = 0; i < iterations; i++) {
        gpu_memory_copy_to_device(d_buf, h_buf, transfer_bytes);
    }

    uint64_t total_ns = perf_timer_stop(&timer);

    *out_latency_us = (double)total_ns / (double)iterations / 1000.0;

    gpu_memory_free(d_buf);
    safe_free((void**)&h_buf);
    return 0;
}

int gpu_run_full_benchmark(GpuBackend backend, int device_index,
                           const GpuBenchmarkConfig* config,
                           GpuBenchmarkResult* result)
{
    if (!result) return -1;

    GpuBenchmarkConfig cfg = config ? *config : gpu_benchmark_config_default();
    memset(result, 0, sizeof(GpuBenchmarkResult));

    result->backend = backend;
    result->device_index = device_index;
    result->overall_score = 0.0;
    result->memory_score = 0.0;
    result->compute_score = 0.0;
    result->bandwidth_score = 0.0;

    const char* name = gpu_backend_name(backend);
    strncpy(result->backend_name, name ? name : "未知", sizeof(result->backend_name) - 1);

    GpuDeviceInfo dev_info;
    memset(&dev_info, 0, sizeof(dev_info));
    if (gpu_get_device_info(backend, device_index, &dev_info) != 0) {
        dev_info.total_memory = 0;
        dev_info.compute_units = 0;
        dev_info.clock_speed = 0.0f;
    }
    result->device_info = dev_info;

    double total_start = gpu_benchmark_get_time_s();

    GpuContext* context = gpu_context_create(backend, device_index);
    if (!context) {
        snprintf(result->summary, sizeof(result->summary),
                 "创建GPU上下文失败: 后端=%s 设备=%d",
                 result->backend_name, device_index);
        return -1;
    }

    if (cfg.enable_memory_pressure) {
        if (gpu_run_memory_pressure_test(context, &cfg, &result->memory_pressure) != 0) {
            if (cfg.verbose) {
                fprintf(stderr, "内存压力测试失败: %s\n", result->memory_pressure.error_message);
            }
        }
    }

    if (cfg.enable_compute_benchmark) {
        if (gpu_run_compute_benchmark(context, &cfg, &result->compute_result) != 0) {
            if (cfg.verbose) {
                fprintf(stderr, "计算性能测试失败\n");
            }
        }
    }

    if (cfg.enable_bandwidth_benchmark) {
        if (gpu_run_bandwidth_benchmark(context, &cfg, &result->bandwidth_result) != 0) {
            if (cfg.verbose) {
                fprintf(stderr, "带宽测试失败\n");
            }
        }
    }

    if (cfg.enable_latency_benchmark) {
        if (benchmark_kernel_latency(context, cfg.latency_iterations, &result->kernel_launch_latency_us) != 0) {
            result->kernel_launch_latency_us = -1.0;
        }
        if (benchmark_memory_latency(context, cfg.latency_transfer_bytes, cfg.latency_iterations, &result->memory_copy_latency_us) != 0) {
            result->memory_copy_latency_us = -1.0;
        }
    }

    gpu_context_free(context);

    result->total_test_time_s = gpu_benchmark_elapsed_s(total_start);

    {
        int hours = (int)(result->total_test_time_s / 3600.0);
        int minutes = (int)((result->total_test_time_s - hours * 3600) / 60.0);
        int seconds = (int)(result->total_test_time_s - hours * 3600 - minutes * 60);
        if (hours > 0) {
            snprintf(result->test_duration_str, sizeof(result->test_duration_str),
                     "%d小时%d分%d秒", hours, minutes, seconds);
        } else if (minutes > 0) {
            snprintf(result->test_duration_str, sizeof(result->test_duration_str),
                     "%d分%d秒", minutes, seconds);
        } else {
            snprintf(result->test_duration_str, sizeof(result->test_duration_str),
                     "%d秒", seconds);
        }
    }

    {
        double score = 0.0;
        int components = 0;

        if (result->memory_pressure.peak_allocated > 0 && result->memory_pressure.total_memory > 0) {
            double util = result->memory_pressure.peak_utilization_ratio;
            double p95 = (result->memory_pressure.total_memory > 0)
                         ? (double)result->memory_pressure.max_single_allocation / (double)result->memory_pressure.total_memory * 100.0
                         : 0.0;
            double mem_score = util * 100.0 * 5.0 + p95 * 3.0;
            if (mem_score > 1000.0) mem_score = 1000.0;
            result->memory_score = mem_score;
            score += mem_score;
            components++;
        }

        if (result->compute_result.average_gflops > 0) {
            double theory = gpu_estimate_theoretical_gflops(&dev_info, 32);
            double utilization = (theory > 0) ? (result->compute_result.average_gflops / theory) : 0.0;
            if (utilization > 1.0) utilization = 1.0;

            double raw_score = result->compute_result.average_gflops;
            double log_score = (raw_score > 1.0) ? log10(raw_score) * 200.0 : 0.0;
            if (log_score > 800.0) log_score = 800.0;

            double util_bonus = utilization * 200.0;
            result->compute_score = log_score + util_bonus;
            if (result->compute_score > 1000.0) result->compute_score = 1000.0;
            score += result->compute_score;
            components++;
        }

        if (result->bandwidth_result.host_to_device_bandwidth_gbps > 0 ||
            result->bandwidth_result.device_to_host_bandwidth_gbps > 0) {
            double best_bw = result->bandwidth_result.host_to_device_bandwidth_gbps;
            if (result->bandwidth_result.device_to_host_bandwidth_gbps > best_bw) {
                best_bw = result->bandwidth_result.device_to_host_bandwidth_gbps;
            }
            double theory_bw = gpu_estimate_theoretical_bandwidth(&dev_info);
            double bw_util = (theory_bw > 0) ? (best_bw / theory_bw) : 0.0;
            if (bw_util > 1.0) bw_util = 1.0;

            double bw_score = best_bw * 10.0 + bw_util * 300.0;
            if (bw_score > 1000.0) bw_score = 1000.0;
            result->bandwidth_score = bw_score;
            score += bw_score;
            components++;
        }

        if (components > 0) {
            result->overall_score = score / (double)components;
            if (result->overall_score > 1000.0) result->overall_score = 1000.0;
        }
    }

    {
        char tmp[1024];
        int pos = 0;

        pos += snprintf(tmp + pos, sizeof(tmp) - (size_t)pos,
                        "设备: %s | 后端: %s | 总内存: %.1f MB",
                        dev_info.name[0] ? dev_info.name : "未知",
                        result->backend_name,
                        (double)dev_info.total_memory / (1024.0 * 1024.0));

        if (result->memory_pressure.peak_allocated > 0) {
            pos += snprintf(tmp + pos, sizeof(tmp) - (size_t)pos,
                            " | 峰值分配: %.1f MB (%.1f%%)",
                            (double)result->memory_pressure.peak_allocated / (1024.0 * 1024.0),
                            result->memory_pressure.peak_utilization_ratio * 100.0);
        }

        if (result->compute_result.average_gflops > 0) {
            pos += snprintf(tmp + pos, sizeof(tmp) - (size_t)pos,
                            " | 计算: %.1f GFLOPS", result->compute_result.average_gflops);
        }

        if (result->bandwidth_result.host_to_device_bandwidth_gbps > 0) {
            pos += snprintf(tmp + pos, sizeof(tmp) - (size_t)pos,
                            " | 带宽(H2D): %.2f GB/s",
                            result->bandwidth_result.host_to_device_bandwidth_gbps);
        }

        pos += snprintf(tmp + pos, sizeof(tmp) - (size_t)pos,
                        " | 综合评分: %.0f/1000", result->overall_score);

        strncpy(result->summary, tmp, sizeof(result->summary) - 1);
        result->summary[sizeof(result->summary) - 1] = '\0';
    }

    return 0;
}

int gpu_benchmark_all_backends(const GpuBenchmarkConfig* config,
                               GpuBenchmarkResult* results, int max_results)
{
    if (!results || max_results <= 0) return -1;

    GpuBenchmarkConfig cfg = config ? *config : gpu_benchmark_config_default();

    GpuBackendAvailability infos[GPU_BENCHMARK_MAX_BACKENDS];
    unsigned int available = gpu_get_available_backends(infos, GPU_BENCHMARK_MAX_BACKENDS);

    if (available == 0) {
        if (gpu_probe_backend(GPU_BACKEND_CPU, NULL) == 1) {
            available |= GPU_BACKEND_FLAG_CPU;
        }
    }

    GpuBackend backends[] = {
        GPU_BACKEND_CUDA,
        GPU_BACKEND_ROCM,
        GPU_BACKEND_INTEL,
        GPU_BACKEND_VULKAN,
        GPU_BACKEND_OPENCL,
        GPU_BACKEND_METAL,
        GPU_BACKEND_ASCEND,
        GPU_BACKEND_CAMBRICON,
        GPU_BACKEND_TPU,
        GPU_BACKEND_CPU
    };
    int num_backends = sizeof(backends) / sizeof(backends[0]);

    unsigned int flag_map[] = {
        GPU_BACKEND_FLAG_CUDA,
        GPU_BACKEND_FLAG_ROCM,
        GPU_BACKEND_FLAG_INTEL,
        GPU_BACKEND_FLAG_VULKAN,
        GPU_BACKEND_FLAG_OPENCL,
        GPU_BACKEND_FLAG_METAL,
        GPU_BACKEND_FLAG_ASCEND,
        GPU_BACKEND_FLAG_CAMBRICON,
        GPU_BACKEND_FLAG_TPU,
        GPU_BACKEND_FLAG_CPU
    };

    int tested_count = 0;
    int bi;
    for (bi = 0; bi < num_backends && tested_count < max_results; bi++) {
        if (!(available & flag_map[bi])) continue;

        if (gpu_init(backends[bi]) == 0) {
            int device_count = gpu_get_device_count(backends[bi]);
            int di;
            for (di = 0; di < device_count && tested_count < max_results; di++) {
                if (gpu_run_full_benchmark(backends[bi], di, &cfg, &results[tested_count]) == 0) {
                    tested_count++;
                }
            }
        }
    }

    return tested_count;
}

int gpu_benchmark_format_report(const GpuBenchmarkResult* result,
                                char* buffer, size_t buffer_size)
{
    if (!result || !buffer || buffer_size < 256) return -1;

    char line[512];
    int pos = 0;

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "================================================================================\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "                    GPU 基准测试报告\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "================================================================================\n\n");

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "【设备信息】\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  设备名称:     %s\n", result->device_info.name[0] ? result->device_info.name : "未知");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  后端类型:     %s\n", result->backend_name);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  设备类型:     %s\n",
        result->device_info.type == GPU_DEVICE_TYPE_DISCRETE ? "独立GPU" :
        result->device_info.type == GPU_DEVICE_TYPE_INTEGRATED ? "集成GPU" :
        result->device_info.type == GPU_DEVICE_TYPE_CPU ? "CPU" : "未知");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  计算单元数:   %d\n", result->device_info.compute_units);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  时钟频率:     %.0f MHz\n", result->device_info.clock_speed);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  总显存:       %.1f MB\n", (double)result->device_info.total_memory / (1024.0 * 1024.0));
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  供应商:       %s\n", result->device_info.vendor[0] ? result->device_info.vendor : "未知");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  架构:         %s\n\n", result->device_info.architecture[0] ? result->device_info.architecture : "未知");

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "【内存压力测试】\n");
    if (result->memory_pressure.peak_allocated > 0) {
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  峰值分配量:       %.1f MB (利用率 %.1f%%)\n",
            (double)result->memory_pressure.peak_allocated / (1024.0 * 1024.0),
            result->memory_pressure.peak_utilization_ratio * 100.0);
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  最大单次分配:     %.1f MB\n",
            (double)result->memory_pressure.max_single_allocation / (1024.0 * 1024.0));
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  分配成功率:       %.1f%%\n",
            result->memory_pressure.allocation_success_rate * 100.0);
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  平均分配耗时:     %.2f ms\n",
            result->memory_pressure.average_allocation_time_ms);
    } else {
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  (测试未执行或失败)\n");
    }

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "\n【计算性能测试】\n");
    if (result->compute_result.average_gflops > 0) {
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  矩阵乘法:         %.1f GFLOPS (耗时 %.2f ms)\n",
            result->compute_result.matrix_mul_gflops,
            result->compute_result.matrix_mul_time_ms);
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  卷积:             %.1f GFLOPS (耗时 %.2f ms)\n",
            result->compute_result.convolution_gflops,
            result->compute_result.convolution_time_ms);
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  逐元素操作:       %.1f GFLOPS (耗时 %.2f ms)\n",
            result->compute_result.elementwise_gflops,
            result->compute_result.elementwise_time_ms);
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  平均 GFLOPS:      %.1f GFLOPS\n",
            result->compute_result.average_gflops);
        {
            double theory = gpu_estimate_theoretical_gflops(&result->device_info, 32);
            if (theory > 0) {
                pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
                    "  理论峰值:         %.1f GFLOPS\n", theory);
                pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
                    "  利用率:           %.1f%%\n",
                    (result->compute_result.average_gflops / theory) * 100.0);
            }
        }
    } else {
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  (测试未执行或失败)\n");
    }

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "\n【带宽测试】\n");
    if (result->bandwidth_result.host_to_device_bandwidth_gbps > 0 ||
        result->bandwidth_result.device_to_host_bandwidth_gbps > 0) {
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  主机→设备:        %.2f GB/s (平均 %.2f ms)\n",
            result->bandwidth_result.host_to_device_bandwidth_gbps,
            result->bandwidth_result.host_to_device_time_ms);
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  设备→主机:        %.2f GB/s (平均 %.2f ms)\n",
            result->bandwidth_result.device_to_host_bandwidth_gbps,
            result->bandwidth_result.device_to_host_time_ms);
        if (result->bandwidth_result.device_to_device_bandwidth_gbps > 0) {
            pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
                "  设备→设备:        %.2f GB/s (平均 %.2f ms)\n",
                result->bandwidth_result.device_to_device_bandwidth_gbps,
                result->bandwidth_result.device_to_device_time_ms);
        }
        {
            double theory = gpu_estimate_theoretical_bandwidth(&result->device_info);
            if (theory > 0) {
                double best_bw = result->bandwidth_result.host_to_device_bandwidth_gbps;
                if (result->bandwidth_result.device_to_host_bandwidth_gbps > best_bw)
                    best_bw = result->bandwidth_result.device_to_host_bandwidth_gbps;
                pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
                    "  理论带宽:         %.1f GB/s\n", theory);
                pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
                    "  带宽利用率:       %.1f%%\n",
                    (best_bw / theory) * 100.0);
            }
        }
    } else {
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  (测试未执行或失败)\n");
    }

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "\n【延迟测试】\n");
    if (result->kernel_launch_latency_us >= 0) {
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  内核启动延迟:     %.2f us\n", result->kernel_launch_latency_us);
    } else {
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  (测试未执行)\n");
    }
    if (result->memory_copy_latency_us >= 0) {
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  小数据拷贝延迟:   %.2f us\n", result->memory_copy_latency_us);
    } else {
        pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
            "  (测试未执行)\n");
    }

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "\n================================================================================\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "【综合评分】\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  内存性能评分:     %.0f / 1000\n", result->memory_score);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  计算性能评分:     %.0f / 1000\n", result->compute_score);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  带宽性能评分:     %.0f / 1000\n", result->bandwidth_score);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  综合评分:         %.0f / 1000\n", result->overall_score);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "================================================================================\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "测试耗时: %s\n", result->test_duration_str);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "引擎版本: %s\n", GPU_BENCHMARK_VERSION);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "================================================================================\n");

    return pos;
}

int gpu_benchmark_format_json(const GpuBenchmarkResult* result,
                              char* buffer, size_t buffer_size)
{
    if (!result || !buffer || buffer_size < 128) return -1;

    int pos = 0;

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos, "{\n");

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"backend\": \"%s\",\n", result->backend_name);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"device_index\": %d,\n", result->device_index);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"device_name\": \"%s\",\n", result->device_info.name);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"device_type\": %d,\n", (int)result->device_info.type);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"compute_units\": %d,\n", result->device_info.compute_units);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"clock_speed_mhz\": %.0f,\n", result->device_info.clock_speed);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"total_memory_mb\": %.1f,\n",
        (double)result->device_info.total_memory / (1024.0 * 1024.0));

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"memory_pressure\": {\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"peak_allocated_mb\": %.1f,\n",
        (double)result->memory_pressure.peak_allocated / (1024.0 * 1024.0));
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"peak_utilization_pct\": %.1f,\n",
        result->memory_pressure.peak_utilization_ratio * 100.0);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"max_single_allocation_mb\": %.1f\n",
        (double)result->memory_pressure.max_single_allocation / (1024.0 * 1024.0));
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos, "  },\n");

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"compute_benchmark\": {\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"matrix_mul_gflops\": %.1f,\n", result->compute_result.matrix_mul_gflops);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"convolution_gflops\": %.1f,\n", result->compute_result.convolution_gflops);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"elementwise_gflops\": %.1f,\n", result->compute_result.elementwise_gflops);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"average_gflops\": %.1f\n", result->compute_result.average_gflops);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos, "  },\n");

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"bandwidth_benchmark\": {\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"host_to_device_gbps\": %.2f,\n",
        result->bandwidth_result.host_to_device_bandwidth_gbps);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"device_to_host_gbps\": %.2f,\n",
        result->bandwidth_result.device_to_host_bandwidth_gbps);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"device_to_device_gbps\": %.2f\n",
        result->bandwidth_result.device_to_device_bandwidth_gbps);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos, "  },\n");

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"latency\": {\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"kernel_launch_us\": %.2f,\n", result->kernel_launch_latency_us);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"memory_copy_us\": %.2f\n", result->memory_copy_latency_us);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos, "  },\n");

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"scores\": {\n");
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"memory_score\": %.0f,\n", result->memory_score);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"compute_score\": %.0f,\n", result->compute_score);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"bandwidth_score\": %.0f,\n", result->bandwidth_score);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "    \"overall_score\": %.0f\n", result->overall_score);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos, "  },\n");

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"test_duration_s\": %.2f,\n", result->total_test_time_s);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"version\": \"%s\",\n", GPU_BENCHMARK_VERSION);
    pos += snprintf(buffer + pos, buffer_size - (size_t)pos,
        "  \"summary\": \"%s\"\n", result->summary);

    pos += snprintf(buffer + pos, buffer_size - (size_t)pos, "}\n");

    return pos;
}
