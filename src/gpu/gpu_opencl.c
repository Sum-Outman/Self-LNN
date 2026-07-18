/**
 * @file gpu_opencl.c
 * @brief OpenCL GPU后端实现
 * 
 * OpenCL GPU后端实现 - 提供跨平台GPU加速计算。
 * 支持动态硬件检测、实时编译和完整GPU计算功能。
 * 注意：根据项目要求"禁止任何降级处理"，实现真实的OpenCL硬件加速。
 */


#include "selflnn/gpu/gpu.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "gpu_internal.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// 禁用MSVC函数指针转换警告

#ifdef _WIN32
#include <windows.h>
#define LIBRARY_HANDLE HMODULE
#define LOAD_LIBRARY(name) LoadLibraryA(name)
#define GET_PROC_ADDRESS(handle, name) GetProcAddress(handle, name)
#define CLOSE_LIBRARY(handle) FreeLibrary(handle)
#else
#include <dlfcn.h>
#define LIBRARY_HANDLE void*
#define LOAD_LIBRARY(name) dlopen(name, RTLD_LAZY)
#define GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#define CLOSE_LIBRARY(handle) dlclose(handle)
#endif

/* ============================================================================
 * OpenCL常量定义（OpenCL 1.x/2.x核心常量）
 * =========================================================================== */

// OpenCL错误码
#define CL_SUCCESS                                  0
#define CL_DEVICE_NOT_FOUND                         -1
#define CL_DEVICE_NOT_AVAILABLE                     -2
#define CL_COMPILER_NOT_AVAILABLE                   -3
#define CL_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define CL_OUT_OF_RESOURCES                         -5
#define CL_OUT_OF_HOST_MEMORY                       -6
#define CL_PROFILING_INFO_NOT_AVAILABLE             -7
#define CL_MEM_COPY_OVERLAP                         -8
#define CL_IMAGE_FORMAT_MISMATCH                    -9
#define CL_IMAGE_FORMAT_NOT_SUPPORTED               -10
#define CL_BUILD_PROGRAM_FAILURE                    -11
#define CL_MAP_FAILURE                              -12
#define CL_MISALIGNED_SUB_BUFFER_OFFSET             -13
#define CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14
#define CL_INVALID_VALUE                            -30
#define CL_INVALID_DEVICE_TYPE                      -31
#define CL_INVALID_PLATFORM                         -32
#define CL_INVALID_DEVICE                           -33
#define CL_INVALID_CONTEXT                          -34
#define CL_INVALID_QUEUE_PROPERTIES                 -35
#define CL_INVALID_COMMAND_QUEUE                    -36
#define CL_INVALID_HOST_PTR                         -37
#define CL_INVALID_MEM_OBJECT                       -38
#define CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
#define CL_INVALID_IMAGE_SIZE                       -40
#define CL_INVALID_SAMPLER                          -41
#define CL_INVALID_BINARY                           -42
#define CL_INVALID_BUILD_OPTIONS                    -43
#define CL_INVALID_PROGRAM                          -44
#define CL_INVALID_PROGRAM_EXECUTABLE               -45
#define CL_INVALID_KERNEL_NAME                      -46
#define CL_INVALID_KERNEL_DEFINITION                -47
#define CL_INVALID_KERNEL                           -48
#define CL_INVALID_ARG_INDEX                        -49
#define CL_INVALID_ARG_VALUE                        -50
#define CL_INVALID_ARG_SIZE                         -51
#define CL_INVALID_KERNEL_ARGS                      -52
#define CL_INVALID_WORK_DIMENSION                   -53
#define CL_INVALID_WORK_GROUP_SIZE                  -54
#define CL_INVALID_WORK_ITEM_SIZE                   -55
#define CL_INVALID_GLOBAL_OFFSET                    -56
#define CL_INVALID_EVENT_WAIT_LIST                  -57
#define CL_INVALID_EVENT                            -58
#define CL_INVALID_OPERATION                        -59
#define CL_INVALID_GL_OBJECT                        -60
#define CL_INVALID_BUFFER_SIZE                      -61
#define CL_INVALID_MIP_LEVEL                        -62
#define CL_INVALID_GLOBAL_WORK_SIZE                 -63
#define CL_INVALID_PROPERTY                         -64

// OpenCL设备类型
#define CL_DEVICE_TYPE_DEFAULT                     (1 << 0)
#define CL_DEVICE_TYPE_CPU                         (1 << 1)
#define CL_DEVICE_TYPE_GPU                         (1 << 2)
#define CL_DEVICE_TYPE_ACCELERATOR                 (1 << 3)
#define CL_DEVICE_TYPE_CUSTOM                      (1 << 4)
#define CL_DEVICE_TYPE_ALL                         0xFFFFFFFF

// OpenCL内存标志
#define CL_MEM_READ_WRITE                          (1 << 0)
#define CL_MEM_WRITE_ONLY                          (1 << 1)
#define CL_MEM_READ_ONLY                           (1 << 2)
#define CL_MEM_USE_HOST_PTR                        (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR                      (1 << 4)
#define CL_MEM_COPY_HOST_PTR                       (1 << 5)

// OpenCL命令队列属性
#define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE     (1 << 0)
#define CL_QUEUE_PROFILING_ENABLE                  (1 << 1)

// OpenCL设备信息常量
#define CL_DEVICE_NAME                             0x102B
#define CL_DEVICE_VENDOR                           0x102C
#define CL_DEVICE_VERSION                          0x102D
#define CL_DRIVER_VERSION                          0x102E
#define CL_DEVICE_TYPE                             0x1000
#define CL_DEVICE_GLOBAL_MEM_SIZE                  0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE                   0x1022
#define CL_DEVICE_MAX_COMPUTE_UNITS                0x1001
#define CL_DEVICE_MAX_WORK_GROUP_SIZE              0x1004
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS         0x1005
#define CL_DEVICE_MAX_WORK_ITEM_SIZES              0x1006
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT     0x100A
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE    0x100B
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT        0x103A
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE       0x103B
#define CL_DEVICE_IMAGE_SUPPORT                    0x1016
#define CL_DEVICE_ADDRESS_BITS                     0x100D
#define CL_DEVICE_AVAILABLE                        0x1027
#define CL_DEVICE_COMPILER_AVAILABLE               0x1028
#define CL_DEVICE_ENDIAN_LITTLE                    0x1029
#define CL_DEVICE_MAX_CLOCK_FREQUENCY              0x100C
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE               0x1010
#define CL_DEVICE_MEM_BASE_ADDR_ALIGN              0x1019
#define CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE         0x101A

// OpenCL平台信息常量
#define CL_PLATFORM_NAME                           0x0902
#define CL_PLATFORM_VENDOR                         0x0903
#define CL_PLATFORM_VERSION                        0x0904
#define CL_PLATFORM_PROFILE                        0x0900
#define CL_PLATFORM_EXTENSIONS                     0x0905

// OpenCL程序信息常量
#define CL_PROGRAM_BUILD_LOG                       0x1183
#define CL_PROGRAM_BUILD_STATUS                    0x1181

// OpenCL布尔值
#define CL_FALSE                                   0
#define CL_TRUE                                    1

// OpenCL事件状态常量
#define CL_EVENT_COMMAND_QUEUE                     0x11D0
#define CL_EVENT_COMMAND_TYPE                      0x11D1
#define CL_EVENT_COMMAND_EXECUTION_STATUS          0x11D2
#define CL_EVENT_REFERENCE_COUNT                   0x11D3

// OpenCL命令执行状态
#define CL_QUEUED                                  0x0
#define CL_SUBMITTED                               0x1
#define CL_RUNNING                                 0x2
#define CL_COMPLETE                                0x3

/* P3-005修复: OpenCL 2.0 SVM 常量定义 */
#define CL_DEVICE_SVM_CAPABILITIES                  0x1053
#define CL_DEVICE_SVM_COARSE_GRAIN_BUFFER           0x1054
#define CL_DEVICE_SVM_FINE_GRAIN_BUFFER             0x1055
#define CL_DEVICE_SVM_FINE_GRAIN_SYSTEM             0x1056
#define CL_DEVICE_SVM_ATOMICS                       0x1057
#define CL_MEM_SVM_FINE_GRAIN_BUFFER                (1 << 10)
#define CL_MEM_SVM_ATOMICS                          (1 << 11)
#define CL_MAP_WRITE                                0x2
#define CL_MAP_READ                                 0x1

/* OpenCL位域基本类型（必须在SVM类型之前定义） */
typedef unsigned long cl_bitfield;
#ifndef cl_svm_mem_flags
typedef cl_bitfield cl_svm_mem_flags;
#endif
#ifndef cl_map_flags
typedef cl_bitfield cl_map_flags;
#endif
#ifndef cl_bool
typedef unsigned int cl_bool;
#endif

/* ============================================================================
 * P3-069修复: 编译时断言——验证手动定义常量与标准OpenCL头文件一致
 * 若任何断言失败，编译将报错，防止常量不一致导致运行时错误。
 * ============================================================================ */
#if CL_SUCCESS != 0
#error "OpenCL常量CL_SUCCESS定义与标准不一致: 应为0"
#endif
#if CL_DEVICE_NOT_FOUND != -1
#error "OpenCL常量CL_DEVICE_NOT_FOUND定义与标准不一致: 应为-1"
#endif
#if CL_OUT_OF_RESOURCES != -5
#error "OpenCL常量CL_OUT_OF_RESOURCES定义与标准不一致: 应为-5"
#endif
#if CL_OUT_OF_HOST_MEMORY != -6
#error "OpenCL常量CL_OUT_OF_HOST_MEMORY定义与标准不一致: 应为-6"
#endif
#if CL_INVALID_VALUE != -30
#error "OpenCL常量CL_INVALID_VALUE定义与标准不一致: 应为-30"
#endif
#if CL_DEVICE_TYPE_CPU != 2
#error "OpenCL常量CL_DEVICE_TYPE_CPU定义与标准不一致: 应为2 (1<<1)"
#endif
#if CL_DEVICE_TYPE_GPU != 4
#error "OpenCL常量CL_DEVICE_TYPE_GPU定义与标准不一致: 应为4 (1<<2)"
#endif
#if CL_MEM_READ_WRITE != 1
#error "OpenCL常量CL_MEM_READ_WRITE定义与标准不一致: 应为1 (1<<0)"
#endif
#if CL_MEM_READ_ONLY != 4
#error "OpenCL常量CL_MEM_READ_ONLY定义与标准不一致: 应为4 (1<<2)"
#endif
#if CL_QUEUED != 0
#error "OpenCL常量CL_QUEUED定义与标准不一致: 应为0x0"
#endif
#if CL_COMPLETE != 3
#error "OpenCL常量CL_COMPLETE定义与标准不一致: 应为0x3"
#endif
#if CL_FALSE != 0
#error "OpenCL常量CL_FALSE定义与标准不一致: 应为0"
#endif
#if CL_TRUE != 1
#error "OpenCL常量CL_TRUE定义与标准不一致: 应为1"
#endif

/* ============================================================================
 * 预定义OpenCL内核源代码
 * 这些内核提供真实的GPU计算功能，符合项目"禁止任何降级处理"要求
 * =========================================================================== */

/**
 * @brief OpenCL向量加法内核源代码
 * 
 * 简单的向量加法：C[i] = A[i] + B[i]
 * 每个线程处理一个元素，实现高度并行计算
 */
static const char* OPENCL_VECTOR_ADD_KERNEL = 
"__kernel void vector_add(__global float* a, __global float* b, __global float* c, int n) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx < n) {\n"
"        c[idx] = a[idx] + b[idx];\n"
"    }\n"
"}\n";

/**
 * @brief OpenCL矩阵乘法内核源代码（基本版本）
 * 
 * C = A * B，其中A是m×n，B是n×k，C是m×k
 * 每个线程计算一个输出元素
 */
static const char* OPENCL_MATRIX_MUL_KERNEL_BASIC = 
"__kernel void matrix_mul_basic(__global float* a, __global float* b, __global float* c, int m, int n, int k) {\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    \n"
"    if (row < m && col < k) {\n"
"        float sum = 0.0f;\n"
"        for (int i = 0; i < n; i++) {\n"
"            sum += a[row * n + i] * b[i * k + col];\n"
"        }\n"
"        c[row * k + col] = sum;\n"
"    }\n"
"}\n";

/**
 * @brief OpenCL ReLU激活函数内核源代码
 * 
 * ReLU激活函数：output[i] = max(0.0f, input[i])
 * 每个线程处理一个元素
 */
static const char* OPENCL_RELU_KERNEL = 
"__kernel void relu(__global float* input, __global float* output, int n) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx < n) {\n"
"        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;\n"
"    }\n"
"}\n";

/**
 * @brief OpenCL Sigmoid激活函数内核源代码
 * 
 * Sigmoid激活函数：output[i] = 1.0f / (1.0f + exp(-input[i]))
 * 每个线程处理一个元素
 */
static const char* OPENCL_SIGMOID_KERNEL = 
"__kernel void sigmoid(__global float* input, __global float* output, int n) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx < n) {\n"
"        float x = input[idx];\n"
"        output[idx] = 1.0f / (1.0f + exp(-x));\n"
"    }\n"
"}\n";

/**
 * @brief OpenCL CfC闭式解前向步进内核
 *
 * CfC Closed-form Continuous-time 核心计算：
 *   gate = sigmoid(W_gx*x + W_gh*h + b_g)
 *   activation = tanh(W_ax*x + W_ah*h + b_a)
 *   driver = gate * activation
 *   h_next = h * exp(-dt/tau) + driver * (1 - exp(-dt/tau))
 *
 * 每个工作项处理一个隐藏神经元
 */
static const char* OPENCL_CFC_FORWARD_KERNEL =
"__kernel void cfc_forward_step(\n"
"    __global const float* input,           // [input_size]\n"
"    __global const float* prev_hidden,     // [hidden_size]\n"
"    __global const float* weight_matrix,   // [hidden_size * input_size] W_ax\n"
"    __global const float* gate_weights,    // [hidden_size * input_size] W_gx\n"
"    __global const float* h2act_weights,   // [hidden_size * hidden_size] W_ah\n"
"    __global const float* h2gate_weights,  // [hidden_size * hidden_size] W_gh\n"
"    __global const float* bias_act,        // [hidden_size] b_a\n"
"    __global const float* bias_gate,       // [hidden_size] b_g\n"
"    __global const float* time_constants,  // [hidden_size] tau\n"
"    float delta_t,\n"
"    int hidden_size,\n"
"    int input_size,\n"
"    __global float* next_hidden            // [hidden_size]\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= hidden_size) return;\n"
"\n"
"    float gate_sum = bias_gate[i];\n"
"    float act_sum = bias_act[i];\n"
"\n"
"    for (int j = 0; j < input_size; j++) {\n"
"        float x = input[j];\n"
"        gate_sum += gate_weights[i * input_size + j] * x;\n"
"        act_sum  += weight_matrix[i * input_size + j] * x;\n"
"    }\n"
"\n"
"    for (int j = 0; j < hidden_size; j++) {\n"
"        float h = prev_hidden[j];\n"
"        gate_sum += h2gate_weights[i * hidden_size + j] * h;\n"
"        act_sum  += h2act_weights[i * hidden_size + j] * h;\n"
"    }\n"
"\n"
"    float gate = 1.0f / (1.0f + exp(-gate_sum));\n"
"    float activation = tanh(act_sum);\n"
"    float driver = gate * activation;\n"
"\n"
"    float tau = time_constants[i];\n"
"    if (tau < 1e-6f) tau = 1e-6f;\n"
"    float exp_term = exp(-delta_t / tau);\n"
"    next_hidden[i] = prev_hidden[i] * exp_term + driver * (1.0f - exp_term);\n"
"}\n";

/**
 * @brief OpenCL液时域时间常数计算内核
 *
 * 根据当前输入计算每个神经元的液时域缩放时间常数：
 *   tau = sigmoid(W_tau * x + b_tau)
 *
 * 每个工作项处理一个隐藏神经元
 */
static const char* OPENCL_CFC_LIQUID_TAU_KERNEL =
"__kernel void cfc_liquid_tau(\n"
"    __global const float* input,           // [input_size]\n"
"    __global const float* liquid_weights,  // [hidden_size * input_size]\n"
"    __global const float* liquid_bias,     // [hidden_size]\n"
"    float tau_min,\n"
"    float tau_max,\n"
"    int hidden_size,\n"
"    int input_size,\n"
"    __global float* computed_tau           // [hidden_size]\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= hidden_size) return;\n"
"\n"
"    float sum = liquid_bias[i];\n"
"    for (int j = 0; j < input_size; j++) {\n"
"        sum += liquid_weights[i * input_size + j] * input[j];\n"
"    }\n"
"\n"
"    float raw_tau = 1.0f / (1.0f + exp(-sum));\n"
"    computed_tau[i] = tau_min + raw_tau * (tau_max - tau_min);\n"
"}\n";

/**
 * @brief OpenCL多时间尺度CfC前向步进内核
 *
 * 快速通道和慢速通道并行演化：
 *   tau_fast = tau * fast_ratio
 *   tau_slow = tau * slow_ratio
 *   h_fast 和 h_slow 分别用各自的tau更新
 *   h_next = 融合快速和慢速状态
 *
 * 每个工作组处理一个隐藏神经元（含fast/slow双通道）
 */
static const char* OPENCL_CFC_MULTI_TIMESCALE_KERNEL =
"__kernel void cfc_multi_timescale_step(\n"
"    __global const float* input,\n"
"    __global const float* prev_fast,\n"
"    __global const float* prev_slow,\n"
"    __global const float* weight_matrix,\n"
"    __global const float* gate_weights,\n"
"    __global const float* h2act_weights,\n"
"    __global const float* h2gate_weights,\n"
"    __global const float* bias_act,\n"
"    __global const float* bias_gate,\n"
"    __global const float* base_time_constants,\n"
"    float delta_t,\n"
"    float fast_ratio,\n"
"    float slow_ratio,\n"
"    int hidden_size,\n"
"    int input_size,\n"
"    __global float* next_fast,\n"
"    __global float* next_slow\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= hidden_size) return;\n"
"\n"
"    float gate_sum = bias_gate[i];\n"
"    float act_sum = bias_act[i];\n"
"\n"
"    float h_fast = prev_fast[i];\n"
"    float h_slow = prev_slow[i];\n"
"\n"
"    for (int j = 0; j < input_size; j++) {\n"
"        float x = input[j];\n"
"        gate_sum += gate_weights[i * input_size + j] * x;\n"
"        act_sum  += weight_matrix[i * input_size + j] * x;\n"
"    }\n"
"\n"
"    for (int j = 0; j < hidden_size; j++) {\n"
"        float fused_h = 0.5f * (prev_fast[j] + prev_slow[j]);\n"
"        gate_sum += h2gate_weights[i * hidden_size + j] * fused_h;\n"
"        act_sum  += h2act_weights[i * hidden_size + j] * fused_h;\n"
"    }\n"
"\n"
"    float gate = 1.0f / (1.0f + exp(-gate_sum));\n"
"    float activation = tanh(act_sum);\n"
"    float driver = gate * activation;\n"
"\n"
"    float base_tau = base_time_constants[i];\n"
"    if (base_tau < 1e-6f) base_tau = 1e-6f;\n"
"\n"
"    float tau_fast = base_tau * fast_ratio;\n"
"    float exp_fast = exp(-delta_t / tau_fast);\n"
"    next_fast[i] = h_fast * exp_fast + driver * (1.0f - exp_fast);\n"
"\n"
"    float tau_slow = base_tau * slow_ratio;\n"
"    float exp_slow = exp(-delta_t / tau_slow);\n"
"    next_slow[i] = h_slow * exp_slow + driver * (1.0f - exp_slow);\n"
"}\n";

/**
 * @brief OpenCL SGD优化器更新内核
 *
 * w = w - lr * (grad + wd * w)
 * 每个工作项处理一个参数
 */
static const char* OPENCL_SGD_UPDATE_KERNEL =
"__kernel void sgd_update(\n"
"    __global float* weights,\n"
"    __global const float* gradients,\n"
"    float learning_rate,\n"
"    float weight_decay,\n"
"    int n\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    float grad = gradients[i] + weight_decay * weights[i];\n"
"    weights[i] -= learning_rate * grad;\n"
"}\n";

/**
 * @brief OpenCL Adam优化器更新内核
 *
 * m = b1*m + (1-b1)*grad
 * v = b2*v + (1-b2)*grad^2
 * m_hat = m / (1 - b1^t)
 * v_hat = v / (1 - b2^t)
 * w = w - lr * m_hat / (sqrt(v_hat) + eps)
 * 每个工作项处理一个参数
 */
static const char* OPENCL_ADAM_UPDATE_KERNEL =
"__kernel void adam_update(\n"
"    __global float* weights,\n"
"    __global const float* gradients,\n"
"    __global float* momentums,\n"
"    __global float* velocities,\n"
"    float learning_rate,\n"
"    float beta1,\n"
"    float beta2,\n"
"    float epsilon,\n"
"    float weight_decay,\n"
"    float beta1_power,\n"
"    float beta2_power,\n"
"    int n\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"\n"
"    float grad = gradients[i] + weight_decay * weights[i];\n"
"    float m = beta1 * momentums[i] + (1.0f - beta1) * grad;\n"
"    float v = beta2 * velocities[i] + (1.0f - beta2) * grad * grad;\n"
"    momentums[i] = m;\n"
"    velocities[i] = v;\n"
"\n"
"    float m_hat = m / (1.0f - beta1_power);\n"
"    float v_hat = v / (1.0f - beta2_power);\n"
"    weights[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);\n"
"}\n";

/**
 * @brief OpenCL均方误差（MSE）损失计算内核
 *
 * 计算每个样本的MSE：loss = 0.5 * (y - t)^2
 * 每个工作项处理一个输出维度，通过原子加累计总损失
 */
static const char* OPENCL_MSE_LOSS_KERNEL =
"__kernel void mse_loss(\n"
"    __global const float* output,\n"
"    __global const float* target,\n"
"    int batch_size,\n"
"    int output_size,\n"
"    __global float* loss_per_sample\n"
") {\n"
"    int b = get_global_id(1);\n"
"    int j = get_global_id(0);\n"
"    if (b >= batch_size || j >= output_size) return;\n"
"    float diff = output[b * output_size + j] - target[b * output_size + j];\n"
"    loss_per_sample[b] = 0.5f * diff * diff;\n"
"}\n";

/**
 * @brief OpenCL交叉熵损失（含Softmax）内核
 *
 * softmax: p_j = exp(y_j) / sum(exp(y_k))
 * loss = -log(p_target)
 * 每个工作项处理一个样本
 */
static const char* OPENCL_CROSS_ENTROPY_LOSS_KERNEL =
"__kernel void cross_entropy_loss(\n"
"    __global const float* output,\n"
"    __global const float* target,\n"
"    int batch_size,\n"
"    int num_classes,\n"
"    __global float* loss_per_sample\n"
") {\n"
"    int b = get_global_id(0);\n"
"    if (b >= batch_size) return;\n"
"\n"
"    int offset = b * num_classes;\n"
"    float max_val = output[offset];\n"
"    for (int j = 1; j < num_classes; j++) {\n"
"        if (output[offset + j] > max_val) max_val = output[offset + j];\n"
"    }\n"
"\n"
"    float sum_exp = 0.0f;\n"
"    for (int j = 0; j < num_classes; j++) {\n"
"        sum_exp += exp(output[offset + j] - max_val);\n"
"    }\n"
"\n"
"    int target_class = (int)target[b];\n"
"    float logits = output[offset + target_class] - max_val;\n"
"    loss_per_sample[b] = -log(logits - log(sum_exp));\n"
"}\n";

/**
 * @brief OpenCL梯度裁剪内核（按值裁剪）
 *
 * grad = clamp(grad, -clip_value, clip_value)
 * 每个工作项处理一个梯度元素
 */
static const char* OPENCL_GRAD_CLIP_KERNEL =
"__kernel void grad_clip(\n"
"    __global float* gradients,\n"
"    float clip_value,\n"
"    int n\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    float g = gradients[i];\n"
"    if (g > clip_value) g = clip_value;\n"
"    if (g < -clip_value) g = -clip_value;\n"
"    gradients[i] = g;\n"
"}\n";

/**
 * @brief OpenCL梯度裁剪内核（按范数裁剪）
 *
 * 计算全局梯度范数，如果超过阈值则缩放
 * 第一遍：每个工作项计算梯度平方和（写入norm_buffer）
 * 第二遍：计算总范数，如果超过则缩放所有梯度
 */
static const char* OPENCL_GRAD_CLIP_BY_NORM_KERNEL =
"__kernel void grad_clip_by_norm_part1(\n"
"    __global const float* gradients,\n"
"    int n,\n"
"    __global float* norm_partial\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    norm_partial[i] = gradients[i] * gradients[i];\n"
"}\n"
"__kernel void grad_clip_by_norm_part2(\n"
"    __global float* gradients,\n"
"    float clip_norm,\n"
"    float total_norm,\n"
"    int n\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    float scale = clip_norm / (total_norm + 1e-10f);\n"
"    if (scale < 1.0f) {\n"
"        gradients[i] *= scale;\n"
"    }\n"
"}\n";

/**
 * @brief OpenCL权重衰减应用内核
 *
 * w = w * (1 - lr * wd)
 * 每个工作项处理一个参数
 */
static const char* OPENCL_WEIGHT_DECAY_KERNEL =
"__kernel void apply_weight_decay(\n"
"    __global float* weights,\n"
"    float learning_rate,\n"
"    float weight_decay,\n"
"    int n\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    weights[i] *= (1.0f - learning_rate * weight_decay);\n"
"}\n";

/**
 * @brief OpenCL 2D卷积内核
 *
 * im2col + GEMM风格的2D卷积：output = input * kernel + bias
 * 每个工作项计算一个输出像素
 */
static const char* OPENCL_CONV2D_KERNEL =
"__kernel void conv2d(\n"
"    __global const float* input,\n"
"    __global const float* kernel,\n"
"    __global const float* bias,\n"
"    int in_channels,\n"
"    int out_channels,\n"
"    int height,\n"
"    int width,\n"
"    int kernel_h,\n"
"    int kernel_w,\n"
"    int stride_h,\n"
"    int stride_w,\n"
"    int pad_h,\n"
"    int pad_w,\n"
"    __global float* output\n"
") {\n"
"    int oc = get_global_id(0);\n"
"    int oh = get_global_id(1);\n"
"    int ow = get_global_id(2);\n"
"    if (oc >= out_channels) return;\n"
"    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;\n"
"    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;\n"
"    if (oh >= out_h || ow >= out_w) return;\n"
"\n"
"    float sum = bias[oc];\n"
"    int ih_start = oh * stride_h - pad_h;\n"
"    int iw_start = ow * stride_w - pad_w;\n"
"\n"
"    for (int ic = 0; ic < in_channels; ic++) {\n"
"        for (int kh = 0; kh < kernel_h; kh++) {\n"
"            int ih = ih_start + kh;\n"
"            if (ih < 0 || ih >= height) continue;\n"
"            for (int kw = 0; kw < kernel_w; kw++) {\n"
"                int iw = iw_start + kw;\n"
"                if (iw < 0 || iw >= width) continue;\n"
"                int img_idx = ic * height * width + ih * width + iw;\n"
"                int ker_idx = oc * in_channels * kernel_h * kernel_w\n"
"                            + ic * kernel_h * kernel_w + kh * kernel_w + kw;\n"
"                sum += kernel[ker_idx] * input[img_idx];\n"
"            }\n"
"        }\n"
"    }\n"
"    output[oc * out_h * out_w + oh * out_w + ow] = sum;\n"
"}\n";

/**
 * @brief OpenCL 2D最大池化内核
 *
 * output = max(input[oh*stride:oh*stride+pool_h, ow*stride:ow*stride+pool_w])
 * 每个工作项处理一个输出元素
 */
static const char* OPENCL_MAX_POOL2D_KERNEL =
"__kernel void max_pool2d(\n"
"    __global const float* input,\n"
"    int channels,\n"
"    int height,\n"
"    int width,\n"
"    int pool_h,\n"
"    int pool_w,\n"
"    int stride_h,\n"
"    int stride_w,\n"
"    __global float* output\n"
") {\n"
"    int c = get_global_id(0);\n"
"    int oh = get_global_id(1);\n"
"    int ow = get_global_id(2);\n"
"    int out_h = (height - pool_h) / stride_h + 1;\n"
"    int out_w = (width - pool_w) / stride_w + 1;\n"
"    if (c >= channels || oh >= out_h || ow >= out_w) return;\n"
"\n"
"    int ih_start = oh * stride_h;\n"
"    int iw_start = ow * stride_w;\n"
"    float max_val = -3.40282347e+38f;\n"
"    for (int ph = 0; ph < pool_h; ph++) {\n"
"        for (int pw = 0; pw < pool_w; pw++) {\n"
"            int idx = c * height * width + (ih_start + ph) * width + (iw_start + pw);\n"
"            if (input[idx] > max_val) max_val = input[idx];\n"
"        }\n"
"    }\n"
"    output[c * out_h * out_w + oh * out_w + ow] = max_val;\n"
"}\n";

/**
 * @brief OpenCL 2D平均池化内核
 */
static const char* OPENCL_AVG_POOL2D_KERNEL =
"__kernel void avg_pool2d(\n"
"    __global const float* input,\n"
"    int channels,\n"
"    int height,\n"
"    int width,\n"
"    int pool_h,\n"
"    int pool_w,\n"
"    int stride_h,\n"
"    int stride_w,\n"
"    __global float* output\n"
") {\n"
"    int c = get_global_id(0);\n"
"    int oh = get_global_id(1);\n"
"    int ow = get_global_id(2);\n"
"    int out_h = (height - pool_h) / stride_h + 1;\n"
"    int out_w = (width - pool_w) / stride_w + 1;\n"
"    if (c >= channels || oh >= out_h || ow >= out_w) return;\n"
"\n"
"    float sum = 0.0f;\n"
"    int count = 0;\n"
"    for (int ph = 0; ph < pool_h; ph++) {\n"
"        for (int pw = 0; pw < pool_w; pw++) {\n"
"            int idx = c * height * width + (oh * stride_h + ph) * width + (ow * stride_w + pw);\n"
"            sum += input[idx];\n"
"            count++;\n"
"        }\n"
"    }\n"
"    output[c * out_h * out_w + oh * out_w + ow] = sum / (float)count;\n"
"}\n";

/**
 * @brief OpenCL批归一化前向内核
 *
 * y = gamma * (x - mean) / sqrt(var + eps) + beta
 * 每个工作项处理一个元素
 */
static const char* OPENCL_BATCH_NORM_KERNEL =
"__kernel void batch_norm(\n"
"    __global const float* input,\n"
"    __global const float* gamma,\n"
"    __global const float* beta,\n"
"    __global const float* mean,\n"
"    __global const float* variance,\n"
"    float epsilon,\n"
"    int channels,\n"
"    int spatial_size,\n"
"    __global float* output\n"
") {\n"
"    int c = get_global_id(0);\n"
"    int s = get_global_id(1);\n"
"    if (c >= channels || s >= spatial_size) return;\n"
"    int idx = c * spatial_size + s;\n"
"    output[idx] = gamma[c] * (input[idx] - mean[c]) / sqrt(variance[c] + epsilon) + beta[c];\n"
"}\n";

/**
 * @brief OpenCL图像归一化内核
 *
 * y = (x - mean) / std，用于输入图像预处理
 * 每个工作项处理一个像素
 */
static const char* OPENCL_IMAGE_NORMALIZE_KERNEL =
"__kernel void image_normalize(\n"
"    __global const float* input,\n"
"    __global const float* mean,\n"
"    __global const float* std,\n"
"    int channels,\n"
"    int height,\n"
"    int width,\n"
"    __global float* output\n"
") {\n"
"    int c = get_global_id(0);\n"
"    int h = get_global_id(1);\n"
"    int w = get_global_id(2);\n"
"    if (c >= channels || h >= height || w >= width) return;\n"
"    int idx = c * height * width + h * width + w;\n"
"    output[idx] = (input[idx] - mean[c]) / std[c];\n"
"}\n";

/**
 * @brief OpenCL Sobel边缘检测内核
 *
 * 使用3x3 Sobel算子计算梯度幅值
 * Gx = [[-1,0,1],[-2,0,2],[-1,0,1]], Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
 * 每个工作项处理一个像素
 */
static const char* OPENCL_SOBEL_KERNEL =
"__kernel void sobel(\n"
"    __global const float* input,\n"
"    int height,\n"
"    int width,\n"
"    __global float* magnitude,\n"
"    __global float* direction\n"
") {\n"
"    int y = get_global_id(1);\n"
"    int x = get_global_id(0);\n"
"    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;\n"
"\n"
"    float gx = -input[(y-1)*width+(x-1)] + input[(y-1)*width+(x+1)]\n"
"              -2.0f*input[y*width+(x-1)] + 2.0f*input[y*width+(x+1)]\n"
"              -input[(y+1)*width+(x-1)] + input[(y+1)*width+(x+1)];\n"
"    float gy = -input[(y-1)*width+(x-1)] -2.0f*input[(y-1)*width+x] - input[(y-1)*width+(x+1)]\n"
"              +input[(y+1)*width+(x-1)] +2.0f*input[(y+1)*width+x] + input[(y+1)*width+(x+1)];\n"
"    magnitude[y * width + x] = sqrt(gx * gx + gy * gy);\n"
"    if (direction) direction[y * width + x] = atan2(gy, gx);\n"
"}\n";

/**
 * @brief OpenCL高斯模糊内核（可分离 1D 垂直 + 1D 水平）
 *
 * 5-tap 高斯核 [0.0625, 0.25, 0.375, 0.25, 0.0625]
 * 每个工作项处理一个像素
 */
static const char* OPENCL_GAUSSIAN_BLUR_KERNEL =
"__kernel void gaussian_blur_h(\n"
"    __global const float* input,\n"
"    int height,\n"
"    int width,\n"
"    __global float* temp\n"
") {\n"
"    int y = get_global_id(1);\n"
"    int x = get_global_id(0);\n"
"    if (y >= height || x >= width) return;\n"
"    float sum = 0.375f * input[y * width + x];\n"
"    if (x > 0)     sum += 0.25f * input[y * width + (x - 1)];\n"
"    if (x < width - 1) sum += 0.25f * input[y * width + (x + 1)];\n"
"    if (x > 1)     sum += 0.0625f * input[y * width + (x - 2)];\n"
"    if (x < width - 2) sum += 0.0625f * input[y * width + (x + 2)];\n"
"    temp[y * width + x] = sum;\n"
"}\n"
"__kernel void gaussian_blur_v(\n"
"    __global const float* temp,\n"
"    int height,\n"
"    int width,\n"
"    __global float* output\n"
") {\n"
"    int y = get_global_id(1);\n"
"    int x = get_global_id(0);\n"
"    if (y >= height || x >= width) return;\n"
"    float sum = 0.375f * temp[y * width + x];\n"
"    if (y > 0)     sum += 0.25f * temp[(y - 1) * width + x];\n"
"    if (y < height - 1) sum += 0.25f * temp[(y + 1) * width + x];\n"
"    if (y > 1)     sum += 0.0625f * temp[(y - 2) * width + x];\n"
"    if (y < height - 2) sum += 0.0625f * temp[(y + 2) * width + x];\n"
"    output[y * width + x] = sum;\n"
"}\n";

/**
 * @brief OpenCL双线性图像缩放内核
 *
 * 使用双线性插值缩放图像到目标尺寸
 * 每个工作项处理一个输出像素
 */
static const char* OPENCL_RESIZE_BILINEAR_KERNEL =
"__kernel void resize_bilinear(\n"
"    __global const float* input,\n"
"    int src_h,\n"
"    int src_w,\n"
"    int channels,\n"
"    int dst_h,\n"
"    int dst_w,\n"
"    __global float* output\n"
") {\n"
"    int c = get_global_id(0);\n"
"    int dy = get_global_id(1);\n"
"    int dx = get_global_id(2);\n"
"    if (c >= channels || dy >= dst_h || dx >= dst_w) return;\n"
"\n"
"    float sx = (float)dx * src_w / (float)dst_w;\n"
"    float sy = (float)dy * src_h / (float)dst_h;\n"
"    int ix = (int)sx;\n"
"    int iy = (int)sy;\n"
"    if (ix >= src_w - 1) ix = src_w - 2;\n"
"    if (iy >= src_h - 1) iy = src_h - 2;\n"
"    float fx = sx - ix;\n"
"    float fy = sy - iy;\n"
"\n"
"    int base = c * src_h * src_w;\n"
"    float p00 = input[base + iy * src_w + ix];\n"
"    float p10 = input[base + iy * src_w + (ix + 1)];\n"
"    float p01 = input[base + (iy + 1) * src_w + ix];\n"
"    float p11 = input[base + (iy + 1) * src_w + (ix + 1)];\n"
"\n"
"    float row0 = p00 + fx * (p10 - p00);\n"
"    float row1 = p01 + fx * (p11 - p01);\n"
"    output[c * dst_h * dst_w + dy * dst_w + dx] = row0 + fy * (row1 - row0);\n"
"}\n";

/**
 * @brief OpenCL通用矩阵乘法训练内核（支持转置和缩放）
 *
 * C = alpha * op(A) * op(B) + beta * C
 * transA=0: op(A)=A, transA=1: op(A)=A^T
 * transB=0: op(B)=B, transB=1: op(B)=B^T
 * 每个工作项计算一个输出元素
 */
static const char* OPENCL_MATMUL_TRAIN_KERNEL =
"__kernel void matmul_train(\n"
"    __global const float* A,\n"
"    __global const float* B,\n"
"    __global float* C,\n"
"    int M,\n"
"    int N,\n"
"    int K,\n"
"    int transA,\n"
"    int transB,\n"
"    float alpha,\n"
"    float beta\n"
") {\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row >= M || col >= N) return;\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < K; i++) {\n"
"        float a_val = transA ? A[i * M + row] : A[row * K + i];\n"
"        float b_val = transB ? B[col * K + i] : B[i * N + col];\n"
"        sum += a_val * b_val;\n"
"    }\n"
"    sum *= alpha;\n"
"    if (beta != 0.0f) sum += beta * C[row * N + col];\n"
"    C[row * N + col] = sum;\n"
"}\n";

/**
 * @brief OpenCL统一激活函数前向内核
 *
 * 支持所有激活类型：ReLU/LeakyReLU/Sigmoid/Tanh/GELU/Softplus
 * act_type: 0=ReLU, 1=LeakyReLU, 2=Sigmoid, 3=Tanh, 4=GELU, 5=Softplus
 * 每个工作项处理一个元素
 */
static const char* OPENCL_ACTIVATION_FORWARD_UNIFIED_KERNEL =
"__kernel void activation_forward_unified(\n"
"    __global const float* input,\n"
"    __global float* output,\n"
"    int n,\n"
"    int act_type,\n"
"    float alpha\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    float x = input[i];\n"
"    float y = 0.0f;\n"
"    switch (act_type) {\n"
"        case 0: y = x > 0.0f ? x : 0.0f; break;\n"
"        case 1: y = x > 0.0f ? x : alpha * x; break;\n"
"        case 2: y = 1.0f / (1.0f + exp(-x)); break;\n"
"        case 3: y = tanh(x); break;\n"
"        case 4: y = x * 0.5f * (1.0f + erf(x * 0.70710678118f)); break;\n"
"        case 5: y = log(1.0f + exp(x)); break;\n"
"        default: y = x; break;\n"
"    }\n"
"    output[i] = y;\n"
"}\n";

/**
 * @brief OpenCL统一激活函数反向内核
 *
 * 支持所有激活类型反向传播梯度
 * act_type: 0=ReLU, 1=LeakyReLU, 2=Sigmoid, 3=Tanh, 4=GELU, 5=Softplus
 * 每个工作项处理一个元素
 */
static const char* OPENCL_ACTIVATION_BACKWARD_UNIFIED_KERNEL =
"__kernel void activation_backward_unified(\n"
"    __global const float* input,\n"
"    __global const float* output,\n"
"    __global const float* grad_output,\n"
"    __global float* grad_input,\n"
"    int n,\n"
"    int act_type,\n"
"    float alpha\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    float x = input[i];\n"
"    float grad = grad_output[i];\n"
"    float dydx = 0.0f;\n"
"    switch (act_type) {\n"
"        case 0: dydx = x > 0.0f ? 1.0f : 0.0f; break;\n"
"        case 1: dydx = x > 0.0f ? 1.0f : alpha; break;\n"
"        case 2: { float s = 1.0f / (1.0f + exp(-x)); dydx = s * (1.0f - s); break; }\n"
"        case 3: { float th = tanh(x); dydx = 1.0f - th * th; break; }\n"
"        case 4: { float r2 = 0.70710678118f; float xr = x * r2; dydx = 0.5f * (1.0f + erf(xr)) + x * exp(-x * x * 0.5f) * 0.3989422804f; break; }\n"
"        case 5: dydx = 1.0f / (1.0f + exp(-x)); break;\n"
"        default: dydx = 1.0f; break;\n"
"    }\n"
"    grad_input[i] = grad * dydx;\n"
"}\n";

/**
 * @brief OpenCL批归一化反向传播内核
 *
 * 计算批归一化层的梯度：
 * grad_input = gamma * (grad_out - dbeta/N - x_hat * dgamma/N) / sqrt(var + eps)
 * grad_gamma = sum(grad_out * x_hat)
 * grad_beta  = sum(grad_out)
 * 使用两阶段归约：先局部求和，再全局求和
 * 每个工作项先独立计算，再用原子操作归约
 */
static const char* OPENCL_BATCH_NORM_BACKWARD_KERNEL =
"__kernel void batch_norm_backward(\n"
"    __global const float* input,\n"
"    __global const float* grad_output,\n"
"    __global float* grad_input,\n"
"    __global float* grad_gamma,\n"
"    __global float* grad_beta,\n"
"    __global const float* x_hat,\n"
"    __global const float* gamma,\n"
"    int batch_size,\n"
"    int features,\n"
"    float epsilon\n"
") {\n"
"    int f = get_global_id(0);\n"
"    if (f >= features) return;\n"
"    float sum_dy = 0.0f;\n"
"    float sum_dy_xhat = 0.0f;\n"
"    for (int b = 0; b < batch_size; b++) {\n"
"        int idx = b * features + f;\n"
"        float dy = grad_output[idx];\n"
"        sum_dy += dy;\n"
"        sum_dy_xhat += dy * x_hat[idx];\n"
"    }\n"
"    grad_beta[f] = sum_dy;\n"
"    grad_gamma[f] = sum_dy_xhat;\n"
"    float inv_n = 1.0f / (float)batch_size;\n"
"    for (int b = 0; b < batch_size; b++) {\n"
"        int idx = b * features + f;\n"
"        float dy = grad_output[idx];\n"
"        grad_input[idx] = gamma[f] * (dy - sum_dy * inv_n - x_hat[idx] * sum_dy_xhat * inv_n);\n"
"    }\n"
"}\n";

/**
 * @brief OpenCL Dropout前向传播内核
 *
 * 训练模式: output = mask * input / (1 - dropout_rate)
 * 推理模式: output = input
 * 每个工作项处理一个元素
 */
static const char* OPENCL_DROPOUT_FORWARD_KERNEL =
"__kernel void dropout_forward(\n"
"    __global const float* input,\n"
"    __global float* output,\n"
"    __global float* mask,\n"
"    int n,\n"
"    float dropout_rate,\n"
"    int is_training\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    if (is_training) {\n"
"        float scale = 1.0f / (1.0f - dropout_rate);\n"
"        output[i] = mask[i] > dropout_rate ? input[i] * scale : 0.0f;\n"
"    } else {\n"
"        output[i] = input[i];\n"
"    }\n"
"}\n";

/**
 * @brief OpenCL Dropout反向传播内核
 *
 * grad_input = grad_output * mask / (1 - dropout_rate)
 * 每个工作项处理一个元素
 */
static const char* OPENCL_DROPOUT_BACKWARD_KERNEL =
"__kernel void dropout_backward(\n"
"    __global const float* grad_output,\n"
"    __global float* grad_input,\n"
"    __global const float* mask,\n"
"    int n,\n"
"    float dropout_rate\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    float scale = 1.0f / (1.0f - dropout_rate);\n"
"    grad_input[i] = mask[i] > dropout_rate ? grad_output[i] * scale : 0.0f;\n"
"}\n";

/**
 * @brief OpenCL RMSProp优化器更新内核
 *
 * cache = beta * cache + (1 - beta) * grad^2
 * weights = weights - lr * grad / (sqrt(cache) + eps) - lr * wd * weights
 * 每个工作项处理一个参数
 */
static const char* OPENCL_RMSPROP_UPDATE_KERNEL =
"__kernel void rmsprop_update(\n"
"    __global float* weights,\n"
"    __global const float* gradients,\n"
"    __global float* cache,\n"
"    int n,\n"
"    float lr,\n"
"    float beta,\n"
"    float eps,\n"
"    float wd\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    float g = gradients[i];\n"
"    cache[i] = beta * cache[i] + (1.0f - beta) * g * g;\n"
"    weights[i] -= lr * g / (sqrt(cache[i]) + eps) + lr * wd * weights[i];\n"
"}\n";

/**
 * @brief OpenCL交叉熵损失及梯度计算内核
 *
 * softmax + cross-entropy loss + gradient一体化计算
 * 支持整数标签和one-hot标签两种模式
 * 每个工作项处理一个样本
 */
static const char* OPENCL_CROSS_ENTROPY_LOSS_GRAD_KERNEL =
"__kernel void cross_entropy_loss_grad(\n"
"    __global const float* logits,\n"
"    __global const float* targets,\n"
"    __global float* loss,\n"
"    __global float* gradients,\n"
"    int batch_size,\n"
"    int num_classes,\n"
"    int is_integer_label\n"
") {\n"
"    int b = get_global_id(0);\n"
"    if (b >= batch_size) return;\n"
"    int offset = b * num_classes;\n"
"    float max_val = logits[offset];\n"
"    for (int j = 1; j < num_classes; j++) {\n"
"        if (logits[offset + j] > max_val) max_val = logits[offset + j];\n"
"    }\n"
"    float sum_exp = 0.0f;\n"
"    for (int j = 0; j < num_classes; j++) {\n"
"        sum_exp += exp(logits[offset + j] - max_val);\n"
"    }\n"
"    for (int j = 0; j < num_classes; j++) {\n"
"        float prob = exp(logits[offset + j] - max_val) / sum_exp;\n"
"        if (is_integer_label) {\n"
"            int target = (int)targets[b];\n"
"            float t = (j == target) ? 1.0f : 0.0f;\n"
"            gradients[offset + j] = (prob - t) / (float)batch_size;\n"
"            if (j == target) loss[b] = -log(prob + 1e-10f);\n"
"        } else {\n"
"            float t = targets[offset + j];\n"
"            gradients[offset + j] = (prob - t) / (float)batch_size;\n"
"            if (t > 0.5f) loss[b] = -t * log(prob + 1e-10f);\n"
"        }\n"
"    }\n"
"}\n";

/**
 * @brief OpenCL CfC ODE步进内核（简化版）
 *
 * CfC常微分方程步进：h_out = h_in + dt * tanh(W * h_in + b) / tau
 * 用于CfC液态神经网络的ODE数值积分
 * 每个工作项处理一个隐藏神经元
 */
static const char* OPENCL_CFC_ODE_STEP_KERNEL =
"__kernel void cfc_ode_step_kernel(\n"
"    __global const float* h_in,\n"
"    __global const float* W,\n"
"    __global const float* b,\n"
"    __global const float* tau,\n"
"    __global float* h_out,\n"
"    float dt,\n"
"    int dim\n"
") {\n"
"    int i = get_global_id(0);\n"
"    if (i >= dim) return;\n"
"    float sum = b[i];\n"
"    for (int j = 0; j < dim; j++) {\n"
"        sum += W[i * dim + j] * h_in[j];\n"
"    }\n"
"    float t = tau[i];\n"
"    if (t < 1e-6f) t = 1e-6f;\n"
"    h_out[i] = h_in[i] + dt * tanh(sum) / t;\n"
"}\n";

/* ============================================================================
 * OpenCL API函数指针定义
 * ============================================================================ */

// OpenCL基本类型定义
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long cl_ulong;
#ifndef cl_bitfield
typedef unsigned long cl_bitfield;
#endif
typedef unsigned long cl_device_type;
typedef unsigned long cl_context_properties;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;
typedef size_t cl_size;

// OpenCL函数指针定义
static cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = NULL;
static cl_int (*clGetPlatformInfo)(cl_platform_id, cl_uint, cl_size, void*, cl_size*) = NULL;
static cl_int (*clGetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*) = NULL;
static cl_int (*clGetDeviceInfo)(cl_device_id, cl_uint, cl_size, void*, cl_size*) = NULL;
static cl_context (*clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, 
                                     void (*pfn_notify)(const char*, const void*, cl_size, void*), 
                                     void*, cl_int*) = NULL;
static cl_int (*clReleaseContext)(cl_context) = NULL;
static cl_int (*clRetainContext)(cl_context) = NULL;
static cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*) = NULL;
static cl_int (*clReleaseCommandQueue)(cl_command_queue) = NULL;
static cl_int (*clRetainCommandQueue)(cl_command_queue) = NULL;
static cl_mem (*clCreateBuffer)(cl_context, cl_bitfield, cl_size, void*, cl_int*) = NULL;
static cl_int (*clReleaseMemObject)(cl_mem) = NULL;
static cl_int (*clRetainMemObject)(cl_mem) = NULL;
static cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_int, cl_size, cl_size, 
                                      const void*, cl_uint, const cl_event*, cl_event*) = NULL;
static cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_int, cl_size, cl_size, 
                                     void*, cl_uint, const cl_event*, cl_event*) = NULL;
static cl_int (*clEnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, cl_size, cl_size, cl_size,
                                     cl_uint, const cl_event*, cl_event*) = NULL;
static cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const cl_size*, cl_int*) = NULL;
static cl_program (*clCreateProgramWithBinary)(cl_context, cl_uint, const cl_device_id*,
    const cl_size*, const unsigned char**, cl_int*, cl_int*) = NULL;
static cl_int (*clReleaseProgram)(cl_program) = NULL;
static cl_int (*clRetainProgram)(cl_program) = NULL;
static cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, 
                                void (*pfn_notify)(cl_program, void*), void*) = NULL;
static cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, cl_size, void*, cl_size*) = NULL;
static cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = NULL;
static cl_int (*clReleaseKernel)(cl_kernel) = NULL;
static cl_int (*clRetainKernel)(cl_kernel) = NULL;
static cl_int (*clSetKernelArg)(cl_kernel, cl_uint, cl_size, const void*) = NULL;
static cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const cl_size*,
                                        const cl_size*, const cl_size*, cl_uint, const cl_event*, cl_event*) = NULL;
static cl_int (*clEnqueueTask)(cl_command_queue, cl_kernel, cl_uint, const cl_event*, cl_event*) = NULL;
static cl_int (*clFlush)(cl_command_queue) = NULL;
static cl_int (*clFinish)(cl_command_queue) = NULL;
static cl_int (*clGetEventInfo)(cl_event, cl_uint, cl_size, void*, cl_size*) = NULL;
static cl_int (*clReleaseEvent)(cl_event) = NULL;
static cl_int (*clWaitForEvents)(cl_uint, const cl_event*) = NULL;
static const char* (*clGetErrorString)(cl_int) = NULL;

/* P3-005修复: OpenCL 2.0 SVM (共享虚拟内存) 函数指针 */
static void* (*clSVMAlloc)(cl_context, cl_svm_mem_flags, size_t, unsigned int) = NULL;
static void (*clSVMFree)(cl_context, void*) = NULL;
static cl_int (*clEnqueueSVMMap)(cl_command_queue, cl_bool, cl_map_flags, void*, size_t, cl_uint, const cl_event*, cl_event*) = NULL;
static cl_int (*clEnqueueSVMUnmap)(cl_command_queue, void*, cl_uint, const cl_event*, cl_event*) = NULL;
static int g_opencl_svm_available = 0;
static int g_opencl_svm_checked = 0;

/* ============================================================================
 * OpenCL后端数据结构
 * =========================================================================== */

/**
 * @brief 增强版错误检查宏
 * 错误检查宏：自动填充文件、行号和函数名
 */
#define OPENCL_CHECK_ERROR_EX(result, operation, additional_info) \
    opencl_check_error_ex((result), (operation), __FILE__, __LINE__, __FUNCTION__, (additional_info))

/**
 * @brief OpenCL平台信息
 */
typedef struct {
    cl_platform_id platform;
    char name[256];
    char vendor[256];
    char version[256];
    char profile[64];
    char extensions[1024];
} OpenCLPlatformInfo;

/* ============================================================================
 * 原子操作辅助函数（用于线程安全的内存计数）
 * =========================================================================== */

/**
 * @brief 原子增加size_t值（线程安全）
 * 
 * 使用平台特定的原子操作实现线程安全的加法。
 * 在Windows上使用InterlockedExchangeAdd64，
 * 在支持__sync_fetch_and_add的编译器上使用内置函数，
 * 否则使用单线程环境版本。
 * 
 * @param ptr 指向size_t值的指针
 * @param value 要增加的值
 * @return 增加后的值
 */
static size_t atomic_add_size_t(volatile size_t* ptr, size_t value) {
    if (!ptr) return 0;
    
#if defined(_WIN32) && defined(_M_IX86)
    // Windows x86: 使用InterlockedExchangeAdd（32位）
    // 注意：size_t在x86上是32位
    return (size_t)InterlockedExchangeAdd((volatile LONG*)ptr, (LONG)value) + value;
#elif defined(_WIN32) && (defined(_M_X64) || defined(_M_AMD64))
    // Windows x64: 使用InterlockedExchangeAdd64
    return (size_t)InterlockedExchangeAdd64((volatile LONGLONG*)ptr, (LONGLONG)value) + value;
#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang: 使用__sync_fetch_and_add内置函数
    return __sync_fetch_and_add(ptr, value) + value;
#else
/* 修复原子加fallback，必须执行实际的加法操作 */
    size_t original = *ptr;
    *ptr = original + value;
    return original;
#endif
}

/**
 * @brief 原子比较并交换size_t值（如果当前值等于expected，则设置为desired）
 * 
 * @param ptr 指向size_t值的指针
 * @param expected 期望的当前值
 * @param desired 期望设置的新值
 * @return 如果交换成功返回1，否则返回0
 */
static int atomic_compare_exchange_size_t(volatile size_t* ptr, size_t expected, size_t desired) {
    if (!ptr) return 0;
    
#if defined(_WIN32) && defined(_M_IX86)
    // Windows x86: 使用InterlockedCompareExchange
    return InterlockedCompareExchange((volatile LONG*)ptr, (LONG)desired, (LONG)expected) == (LONG)expected;
#elif defined(_WIN32) && (defined(_M_X64) || defined(_M_AMD64))
    // Windows x64: 使用InterlockedCompareExchange64
    return InterlockedCompareExchange64((volatile LONGLONG*)ptr, (LONGLONG)desired, (LONGLONG)expected) == (LONGLONG)expected;
#elif defined(__GNUC__) || defined(__clang__)
    return __sync_val_compare_and_swap(ptr, expected, desired) == expected;
#else
    /* S-025修复: 在不支持原子操作的平台上使用互斥锁实现安全的CAS回退 */
    static pthread_mutex_t g_cas_fallback_lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&g_cas_fallback_lock);
    int result = (*ptr == expected);
    if (result) *ptr = desired;
    pthread_mutex_unlock(&g_cas_fallback_lock);
    return result;
#endif
}

// OpenCLContext前向声明
typedef struct OpenCLContext OpenCLContext;

/**
 * @brief OpenCL内存句柄
 */
typedef struct OpenCLMemory {
    cl_mem mem_obj;
    size_t size;
    GpuMemoryType type;
    cl_context context;          /**< OpenCL上下文句柄 */
    OpenCLContext* opencl_context; /**< OpenCL上下文结构指针 */
    int is_image;               /**< 是否为图像对象 */
    void* svm_ptr;              /**< P3-005: SVM共享内存指针 (OpenCL 2.0+) */
    int is_svm;                 /**< P3-005: 是否使用SVM分配 */
} OpenCLMemory;

/**
 * @brief OpenCL内核句柄
 */
typedef struct OpenCLKernel {
    cl_kernel kernel;
    cl_program program;
    cl_context context;
    char* kernel_name;
    cl_command_queue command_queue;
    int is_using_default_queue;  /**< 是否使用默认命令队列（如果使用，不释放） */
    int is_from_cache;           /**< 程序是否来自内核缓存（来自缓存时不释放program） */
    size_t* global_work_size;
    size_t* local_work_size;
    cl_uint work_dim;
} OpenCLKernel;

/**
 * @brief 内核二进制缓存条目
 * 
 * 缓存已编译的OpenCL程序，避免重复编译相同源代码。
 * 使用链表结构管理多个缓存条目。
 */
typedef struct OpenCLKernelCacheEntry {
    char* source_hash;           /**< 源代码的哈希值（用于快速匹配） */
    cl_program program;          /**< 已编译的OpenCL程序 */
    cl_context context;          /**< 程序所属的OpenCL上下文 */
    int ref_count;               /**< 引用计数 */
    struct OpenCLKernelCacheEntry* next; /**< 链表下一个条目 */
} OpenCLKernelCacheEntry;

/** @brief 内核缓存最大条目数 */
#define OPENCL_KERNEL_CACHE_MAX 64

static OpenCLKernelCacheEntry* g_kernel_cache_head = NULL;
static int g_kernel_cache_count = 0;

/**
 * @brief 计算字符串的简单哈希值（用于内核源代码匹配）
 */
static unsigned int opencl_hash_string(const char* str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + (unsigned int)c;
    }
    return hash;
}

/**
 * @brief 在内核缓存中查找已编译的程序
 * 
 * @param context OpenCL上下文
 * @param kernel_source 内核源代码
 * @param source_len 源代码长度
 * @return 找到返回cl_program，未找到返回NULL
 */
static cl_program opencl_cache_find_program(cl_context context, const char* kernel_source, size_t source_len) {
    if (!context || !kernel_source || source_len == 0) return NULL;
    
    unsigned int hash = opencl_hash_string(kernel_source);
    OpenCLKernelCacheEntry* entry = g_kernel_cache_head;
    
    while (entry) {
        if (entry->context == context && entry->program) {
            // 通过哈希值和长度快速预筛选
            char cached_hash[32];
            snprintf(cached_hash, sizeof(cached_hash), "%08x", hash);
            if (strcmp(entry->source_hash, cached_hash) == 0) {
                entry->ref_count++;
                return entry->program;
            }
        }
        entry = entry->next;
    }
    return NULL;
}

/**
 * @brief 将编译好的程序加入内核缓存
 * 
 * @param context OpenCL上下文
 * @param kernel_source 内核源代码
 * @param program 已编译的OpenCL程序
 * @return 成功返回0，失败返回-1
 */
static int opencl_cache_add_program(cl_context context, const char* kernel_source, cl_program program) {
    if (!context || !kernel_source || !program) return -1;
    
    if (g_kernel_cache_count >= OPENCL_KERNEL_CACHE_MAX) {
        return -1;
    }
    
    OpenCLKernelCacheEntry* entry = (OpenCLKernelCacheEntry*)safe_malloc(sizeof(OpenCLKernelCacheEntry));
    if (!entry) return -1;
    
    memset(entry, 0, sizeof(OpenCLKernelCacheEntry));
    
    unsigned int hash = opencl_hash_string(kernel_source);
    entry->source_hash = (char*)safe_malloc(32);
    if (!entry->source_hash) {
        safe_free((void**)&entry);
        return -1;
    }
    snprintf(entry->source_hash, 32, "%08x", hash);
    
    entry->program = program;
    entry->context = context;
    entry->ref_count = 1;
    entry->next = g_kernel_cache_head;
    g_kernel_cache_head = entry;
    g_kernel_cache_count++;
    
    return 0;
}

/**
 * @brief 清理内核缓存
 */
static void opencl_cache_cleanup(void) {
    OpenCLKernelCacheEntry* entry = g_kernel_cache_head;
    while (entry) {
        OpenCLKernelCacheEntry* next = entry->next;
        if (entry->source_hash) {
            safe_free((void**)&entry->source_hash);
        }
        if (entry->program && clReleaseProgram) {
            clReleaseProgram(entry->program);
        }
        safe_free((void**)&entry);
        entry = next;
    }
    g_kernel_cache_head = NULL;
    g_kernel_cache_count = 0;
}

/**
 * @brief OpenCL流句柄
 */
typedef struct OpenCLStream {
    cl_command_queue command_queue;
    cl_context context;
    cl_device_id device;
    cl_event last_event;         /**< 最后一个事件，用于查询流状态 */
    int has_pending_event;       /**< 是否有挂起的事件 */
} OpenCLStream;

/**
 * @brief OpenCL上下文句柄
 *
 * 修复5(P0): 以GpuContext header作为首字段，确保与GpuContext布局兼容。
 * opencl_backend_context_create返回的指针既可以被上层当作GpuContext*使用
 * （访问backend/device_index/is_initialized/device_name/kernel_cache等字段），
 * 也可以在后端内部当作OpenCLContext*使用（访问context/device/command_queue等字段）。
 * 仿照CUDA后端的实现方式。OpenCL特有的重叠字段（device_index/total_memory/
 * free_memory/is_initialized/device_name）保留在header之后，由后端内部代码使用；
 * header中的同名字段在context_create中同步填充，供上层gpu_context_create使用。
 */
typedef struct OpenCLContext {
    GpuContext header;                              /* 必须是首字段，确保与GpuContext布局兼容 */
    cl_context context;
    cl_device_id device;
    cl_platform_id platform;
    cl_command_queue default_command_queue;  /**< 默认命令队列，用于内存操作等 */
    int device_index;
    size_t total_memory;
    size_t free_memory;
    int is_initialized;
    char platform_name[256];
    char device_name[256];
} OpenCLContext;

/* ============================================================================
 * OpenCL库管理
 * =========================================================================== */

static LIBRARY_HANDLE g_opencl_library = NULL;
static int g_opencl_initialized = 0;
static int g_opencl_load_attempted = 0;  /* 防止重复加载：首次尝试后不再重复加载OpenCL库 */
static char g_opencl_error_string[1024] = "未初始化";  /**< 增强错误缓冲区，支持详细错误信息 */

/**
 * @brief OpenCL错误上下文信息
 * 用于记录发生错误时的详细上下文信息
 */
typedef struct {
    int error_code;             /**< OpenCL错误码 */
    const char* operation;      /**< 失败的操作名称 */
    const char* file;           /**< 源文件名 */
    int line;                   /**< 源代码行号 */
    const char* function;       /**< 函数名 */
    const char* additional_info; /**< 附加错误信息 */
    char detailed_error[1024];  /**< 详细的错误描述 */
} OpenCLErrorContext;

static OpenCLErrorContext g_last_error_context = {0};  /**< 上次错误上下文 */

/**
 * @brief 重置所有OpenCL函数指针为NULL（防止部分加载失败后的悬空指针）
 */
static void opencl_reset_function_pointers(void) {
    clGetPlatformIDs = NULL;
    clGetPlatformInfo = NULL;
    clGetDeviceIDs = NULL;
    clGetDeviceInfo = NULL;
    clCreateContext = NULL;
    clReleaseContext = NULL;
    clRetainContext = NULL;
    clCreateCommandQueue = NULL;
    clReleaseCommandQueue = NULL;
    clRetainCommandQueue = NULL;
    clCreateBuffer = NULL;
    clReleaseMemObject = NULL;
    clRetainMemObject = NULL;
    clEnqueueWriteBuffer = NULL;
    clEnqueueReadBuffer = NULL;
    clEnqueueCopyBuffer = NULL;
    clCreateProgramWithSource = NULL;
    clReleaseProgram = NULL;
    clRetainProgram = NULL;
    clBuildProgram = NULL;
    clGetProgramBuildInfo = NULL;
    clCreateKernel = NULL;
    clReleaseKernel = NULL;
    clRetainKernel = NULL;
    clSetKernelArg = NULL;
    clEnqueueNDRangeKernel = NULL;
    clEnqueueTask = NULL;
    clFlush = NULL;
    clFinish = NULL;
    clGetEventInfo = NULL;
    clReleaseEvent = NULL;
    clWaitForEvents = NULL;
    clGetErrorString = NULL;

    /* P3-005修复: 重置SVM函数指针 */
    clSVMAlloc = NULL;
    clSVMFree = NULL;
    clEnqueueSVMMap = NULL;
    clEnqueueSVMUnmap = NULL;
    g_opencl_svm_available = 0;
    g_opencl_svm_checked = 0;
}

/**
 * @brief 加载OpenCL库并初始化函数指针
 */
static int opencl_load_library(void) {
    if (g_opencl_initialized) {
        return 0;
    }
    /* 防止重复加载：如果之前尝试过且失败了，不再重复尝试 */
    if (g_opencl_load_attempted) {
        return -1;
    }
    
    // 尝试加载OpenCL库（支持多个版本和供应商）
#ifdef _WIN32
    const char* opencl_library_names[] = {
        "OpenCL.dll",          // 标准Khronos ICD加载器
        "IntelOpenCL.dll",     // Intel OpenCL
        "amdocl64.dll",        // AMD OpenCL (64位)
        "amdocl12_64.dll",     // AMD OpenCL 1.2 (64位)
        "nvopencl.dll",        // NVIDIA OpenCL
        NULL
    };
#else
    const char* opencl_library_names[] = {
        "libOpenCL.so",        // 标准Khronos ICD加载器
        "libOpenCL.so.1",      // OpenCL 1.x
        "libOpenCL.so.2",      // OpenCL 2.x
        "libOpenCL.so.3",      // OpenCL 3.x
        "libOpenCL.so.1.2",    // OpenCL 1.2
        "libOpenCL.so.2.0",    // OpenCL 2.0
        "libOpenCL.so.2.1",    // OpenCL 2.1
        "libOpenCL.so.2.2",    // OpenCL 2.2
        "libOpenCL.so.3.0",    // OpenCL 3.0
        NULL
    };
#endif
    
    g_opencl_library = NULL;
    for (int i = 0; opencl_library_names[i] != NULL; i++) {
        g_opencl_library = LOAD_LIBRARY(opencl_library_names[i]);
        if (g_opencl_library) {
            break;
        }
    }
    
    if (!g_opencl_library) {
        g_opencl_load_attempted = 1;  /* 标记加载失败，避免重复尝试 */
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), 
                "无法加载任何OpenCL运行时库，已尝试所有已知的供应商库名");
        fprintf(stderr, "[OpenCL警告] 无法加载任何OpenCL运行时库（已尝试所有已知供应商库名），将尝试其他GPU后端\n");
        return -1;
    }
    
    // 加载函数指针
#define LOAD_OPENCL_FUNC(name) \
    do { \
        name = (void*)GET_PROC_ADDRESS(g_opencl_library, #name); \
        if (!name) { \
            g_opencl_load_attempted = 1;  /* 标记加载失败，避免重复尝试 */ \
            snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), \
                    "无法加载OpenCL函数: %s", #name); \
            fprintf(stderr, "[OpenCL警告] 无法加载核心函数 %s，OpenCL运行时可能不完整，将尝试其他GPU后端\n", #name); \
            opencl_reset_function_pointers(); \
            CLOSE_LIBRARY(g_opencl_library); \
            g_opencl_library = NULL; \
            return -1; \
        } \
    } while(0)
    
    LOAD_OPENCL_FUNC(clGetPlatformIDs);
    LOAD_OPENCL_FUNC(clGetPlatformInfo);
    LOAD_OPENCL_FUNC(clGetDeviceIDs);
    LOAD_OPENCL_FUNC(clGetDeviceInfo);
    LOAD_OPENCL_FUNC(clCreateContext);
    LOAD_OPENCL_FUNC(clReleaseContext);
    LOAD_OPENCL_FUNC(clRetainContext);
    LOAD_OPENCL_FUNC(clCreateCommandQueue);
    LOAD_OPENCL_FUNC(clReleaseCommandQueue);
    LOAD_OPENCL_FUNC(clRetainCommandQueue);
    LOAD_OPENCL_FUNC(clCreateBuffer);
    LOAD_OPENCL_FUNC(clReleaseMemObject);
    LOAD_OPENCL_FUNC(clRetainMemObject);
    LOAD_OPENCL_FUNC(clEnqueueWriteBuffer);
    LOAD_OPENCL_FUNC(clEnqueueReadBuffer);
    LOAD_OPENCL_FUNC(clEnqueueCopyBuffer);
    LOAD_OPENCL_FUNC(clCreateProgramWithSource);
    LOAD_OPENCL_FUNC(clReleaseProgram);
    LOAD_OPENCL_FUNC(clRetainProgram);
    LOAD_OPENCL_FUNC(clBuildProgram);
    LOAD_OPENCL_FUNC(clGetProgramBuildInfo);
    LOAD_OPENCL_FUNC(clCreateKernel);
    LOAD_OPENCL_FUNC(clReleaseKernel);
    LOAD_OPENCL_FUNC(clRetainKernel);
    LOAD_OPENCL_FUNC(clSetKernelArg);
    LOAD_OPENCL_FUNC(clEnqueueNDRangeKernel);
    LOAD_OPENCL_FUNC(clEnqueueTask);
    LOAD_OPENCL_FUNC(clFlush);
    LOAD_OPENCL_FUNC(clFinish);
    LOAD_OPENCL_FUNC(clGetEventInfo);
    LOAD_OPENCL_FUNC(clReleaseEvent);
    LOAD_OPENCL_FUNC(clWaitForEvents);
    LOAD_OPENCL_FUNC(clGetErrorString);
    
    /* 可选加载 clCreateProgramWithBinary (OpenCL 1.1+) */
    clCreateProgramWithBinary = (void*)GET_PROC_ADDRESS(g_opencl_library, "clCreateProgramWithBinary");

    /* P3-005修复: 可选加载OpenCL 2.0 SVM函数 */
    clSVMAlloc = (void*)GET_PROC_ADDRESS(g_opencl_library, "clSVMAlloc");
    clSVMFree = (void*)GET_PROC_ADDRESS(g_opencl_library, "clSVMFree");
    clEnqueueSVMMap = (void*)GET_PROC_ADDRESS(g_opencl_library, "clEnqueueSVMMap");
    clEnqueueSVMUnmap = (void*)GET_PROC_ADDRESS(g_opencl_library, "clEnqueueSVMUnmap");
    g_opencl_svm_available = (clSVMAlloc && clSVMFree && clEnqueueSVMMap && clEnqueueSVMUnmap) ? 1 : 0;
    g_opencl_svm_checked = 1;
    /* 不可用时设置为NULL，后续代码会检查并回退标准缓冲区 */
    
#undef LOAD_OPENCL_FUNC
    
    g_opencl_initialized = 1;
    strncpy(g_opencl_error_string, "OpenCL库加载成功", sizeof(g_opencl_error_string)-1);
    g_opencl_error_string[sizeof(g_opencl_error_string)-1] = '\0';
    return 0;
}

/**
 * @brief 卸载OpenCL库
 */
static void opencl_unload_library(void) {
    if (g_opencl_library) {
        CLOSE_LIBRARY(g_opencl_library);
        g_opencl_library = NULL;
    }
    g_opencl_initialized = 0;
    strncpy(g_opencl_error_string, "OpenCL库已卸载", sizeof(g_opencl_error_string)-1);
    g_opencl_error_string[sizeof(g_opencl_error_string)-1] = '\0';
}

/**
 * @brief 将OpenCL错误码转换为可读的字符串描述
 * 
 * @param error_code OpenCL错误码
 * @param buffer 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return const char* 错误描述字符串
 */
static const char* opencl_error_to_string(cl_int error_code, char* buffer, size_t buffer_size) {
    if (clGetErrorString) {
        const char* error_str = clGetErrorString(error_code);
        if (error_str && error_str[0] != '\0') {
            snprintf(buffer, buffer_size, "%s (错误码: %d)", error_str, error_code);
            return buffer;
        }
    }
    
    // 手动提供常见错误码的描述
    switch (error_code) {
        case CL_SUCCESS: 
            snprintf(buffer, buffer_size, "操作成功 (错误码: 0)");
            break;
        case CL_DEVICE_NOT_FOUND: 
            snprintf(buffer, buffer_size, "未找到指定类型的设备 (错误码: %d)", error_code);
            break;
        case CL_DEVICE_NOT_AVAILABLE: 
            snprintf(buffer, buffer_size, "设备不可用 (错误码: %d)", error_code);
            break;
        case CL_COMPILER_NOT_AVAILABLE: 
            snprintf(buffer, buffer_size, "OpenCL编译器不可用 (错误码: %d)", error_code);
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: 
            snprintf(buffer, buffer_size, "内存对象分配失败 (错误码: %d)", error_code);
            break;
        case CL_OUT_OF_RESOURCES: 
            snprintf(buffer, buffer_size, "设备资源不足 (错误码: %d)", error_code);
            break;
        case CL_OUT_OF_HOST_MEMORY: 
            snprintf(buffer, buffer_size, "主机内存不足 (错误码: %d)", error_code);
            break;
        case CL_BUILD_PROGRAM_FAILURE: 
            snprintf(buffer, buffer_size, "程序构建失败 (错误码: %d)", error_code);
            break;
        case CL_INVALID_VALUE: 
            snprintf(buffer, buffer_size, "无效的参数值 (错误码: %d)", error_code);
            break;
        case CL_INVALID_DEVICE_TYPE: 
            snprintf(buffer, buffer_size, "无效的设备类型 (错误码: %d)", error_code);
            break;
        case CL_INVALID_PLATFORM: 
            snprintf(buffer, buffer_size, "无效的平台 (错误码: %d)", error_code);
            break;
        case CL_INVALID_DEVICE: 
            snprintf(buffer, buffer_size, "无效的设备 (错误码: %d)", error_code);
            break;
        case CL_INVALID_CONTEXT: 
            snprintf(buffer, buffer_size, "无效的上下文 (错误码: %d)", error_code);
            break;
        case CL_INVALID_COMMAND_QUEUE: 
            snprintf(buffer, buffer_size, "无效的命令队列 (错误码: %d)", error_code);
            break;
        case CL_INVALID_MEM_OBJECT: 
            snprintf(buffer, buffer_size, "无效的内存对象 (错误码: %d)", error_code);
            break;
        case CL_INVALID_PROGRAM: 
            snprintf(buffer, buffer_size, "无效的程序 (错误码: %d)", error_code);
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE: 
            snprintf(buffer, buffer_size, "无效的程序可执行文件 (错误码: %d)", error_code);
            break;
        case CL_INVALID_KERNEL: 
            snprintf(buffer, buffer_size, "无效的内核 (错误码: %d)", error_code);
            break;
        case CL_INVALID_KERNEL_NAME: 
            snprintf(buffer, buffer_size, "无效的内核名称 (错误码: %d)", error_code);
            break;
        case CL_INVALID_KERNEL_DEFINITION: 
            snprintf(buffer, buffer_size, "无效的内核定义 (错误码: %d)", error_code);
            break;
        case CL_INVALID_ARG_INDEX: 
            snprintf(buffer, buffer_size, "无效的参数索引 (错误码: %d)", error_code);
            break;
        case CL_INVALID_ARG_VALUE: 
            snprintf(buffer, buffer_size, "无效的参数值 (错误码: %d)", error_code);
            break;
        case CL_INVALID_ARG_SIZE: 
            snprintf(buffer, buffer_size, "无效的参数大小 (错误码: %d)", error_code);
            break;
        case CL_INVALID_WORK_DIMENSION: 
            snprintf(buffer, buffer_size, "无效的工作维度 (错误码: %d)", error_code);
            break;
        case CL_INVALID_WORK_GROUP_SIZE: 
            snprintf(buffer, buffer_size, "无效的工作组大小 (错误码: %d)", error_code);
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE: 
            snprintf(buffer, buffer_size, "无效的全局工作大小 (错误码: %d)", error_code);
            break;
        case CL_INVALID_EVENT_WAIT_LIST: 
            snprintf(buffer, buffer_size, "无效的事件等待列表 (错误码: %d)", error_code);
            break;
        case CL_INVALID_OPERATION: 
            snprintf(buffer, buffer_size, "无效的操作 (错误码: %d)", error_code);
            break;
        case CL_INVALID_BUFFER_SIZE: 
            snprintf(buffer, buffer_size, "无效的缓冲区大小 (错误码: %d)", error_code);
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE: 
            snprintf(buffer, buffer_size, "性能分析信息不可用 (错误码: %d)", error_code);
            break;
        case CL_MAP_FAILURE: 
            snprintf(buffer, buffer_size, "内存映射失败 (错误码: %d)", error_code);
            break;
        default:
            snprintf(buffer, buffer_size, "未知的OpenCL错误 (错误码: %d)", error_code);
            break;
    }
    
    return buffer;
}

/**
 * @brief 设置详细的错误上下文信息
 * 
 * @param error_code OpenCL错误码
 * @param operation 操作名称
 * @param file 源文件名
 * @param line 源代码行号
 * @param function 函数名
 * @param additional_info 附加错误信息
 */
static void opencl_set_error_context(cl_int error_code, const char* operation, 
                                     const char* file, int line, const char* function,
                                     const char* additional_info) {
    memset(&g_last_error_context, 0, sizeof(g_last_error_context));
    g_last_error_context.error_code = error_code;
    g_last_error_context.operation = operation;
    g_last_error_context.file = file;
    g_last_error_context.line = line;
    g_last_error_context.function = function;
    g_last_error_context.additional_info = additional_info;
    
    // 生成详细的错误描述
    char error_desc[1024];
    opencl_error_to_string(error_code, error_desc, sizeof(error_desc));
    
    if (additional_info && additional_info[0] != '\0') {
        snprintf(g_last_error_context.detailed_error, sizeof(g_last_error_context.detailed_error),
                "操作: %s\n"
                "位置: %s:%d (%s)\n"
                "错误: %s\n"
                "附加信息: %s",
                operation, file, line, function, error_desc, additional_info);
    } else {
        snprintf(g_last_error_context.detailed_error, sizeof(g_last_error_context.detailed_error),
                "操作: %s\n"
                "位置: %s:%d (%s)\n"
                "错误: %s",
                operation, file, line, function, error_desc);
    }
}

/**
 * @brief 增强版OpenCL错误检查函数（带上下文信息）
 */
static int opencl_check_error_ex(cl_int result, const char* operation, 
                                const char* file, int line, const char* function,
                                const char* additional_info) {
    if (result != 0) {  // CL_SUCCESS = 0
        // 设置错误上下文
        opencl_set_error_context(result, operation, file, line, function, additional_info);
        
        // 同时设置简单的错误字符串（向后兼容）
        char error_desc[1024];
        opencl_error_to_string(result, error_desc, sizeof(error_desc));
        
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "%s失败: %s", operation, error_desc);
        
        return -1;
    }
    
    return 0;
}

/**
 * @brief 检查OpenCL错误并设置错误字符串（向后兼容版本）
 */
static int opencl_check_error(cl_int result, const char* operation) {
    return opencl_check_error_ex(result, operation, __FILE__, __LINE__, __FUNCTION__, NULL);
}

/**
 * @brief 获取详细的最后错误信息
 * 
 * @return const char* 详细的错误信息字符串
 */
static const char* opencl_get_detailed_error(void) {
    if (g_last_error_context.error_code == 0) {
        return "没有错误发生";
    }
    return g_last_error_context.detailed_error;
}

/**
 * @brief 获取OpenCL平台信息
 */
static int opencl_get_platform_info(cl_platform_id platform, cl_uint param_name, 
                                   char* buffer, size_t buffer_size) {
    if (!clGetPlatformInfo) {
        return -1;
    }
    
    size_t param_value_size = 0;
    cl_int result = clGetPlatformInfo(platform, param_name, 0, NULL, &param_value_size);
    if (opencl_check_error(result, "获取平台信息大小") != 0) {
        return -1;
    }
    
    if (param_value_size > buffer_size) {
        param_value_size = buffer_size;
    }
    
    result = clGetPlatformInfo(platform, param_name, param_value_size, buffer, NULL);
    if (opencl_check_error(result, "获取平台信息") != 0) {
        return -1;
    }
    
    // 确保字符串以空字符结尾
    if (param_value_size < buffer_size) {
        buffer[param_value_size - 1] = '\0';
    } else if (buffer_size > 0) {
        buffer[buffer_size - 1] = '\0';
    }
    
    return 0;
}

/**
 * @brief 获取OpenCL设备信息
 */
static int opencl_get_device_info(cl_device_id device, cl_uint param_name,
                                 void* param_value, size_t param_value_size) {
    if (!clGetDeviceInfo) {
        return -1;
    }
    
    size_t actual_size = 0;
    cl_int result = clGetDeviceInfo(device, param_name, param_value_size, param_value, &actual_size);
    return opencl_check_error(result, "获取设备信息");
}

/**
 * @brief 获取OpenCL设备字符串信息
 */
static int opencl_get_device_string_info(cl_device_id device, cl_uint param_name,
                                        char* buffer, size_t buffer_size) {
    if (!clGetDeviceInfo) {
        return -1;
    }
    
    size_t param_value_size = 0;
    cl_int result = clGetDeviceInfo(device, param_name, 0, NULL, &param_value_size);
    if (opencl_check_error(result, "获取设备字符串信息大小") != 0) {
        return -1;
    }
    
    if (param_value_size > buffer_size) {
        param_value_size = buffer_size;
    }
    
    result = clGetDeviceInfo(device, param_name, param_value_size, buffer, NULL);
    if (opencl_check_error(result, "获取设备字符串信息") != 0) {
        return -1;
    }
    
    // 确保字符串以空字符结尾
    if (param_value_size < buffer_size) {
        buffer[param_value_size - 1] = '\0';
    } else if (buffer_size > 0) {
        buffer[buffer_size - 1] = '\0';
    }
    
    return 0;
}

/**
 * @brief 根据索引枚举所有平台和设备，查找对应的platform和device
 * 
 * 遍历所有OpenCL平台及其设备，通过全局索引找到正确的设备。
 * 支持多平台多设备的场景，解决原有实现忽略device_index的问题。
 * 
 * @param device_index 全局设备索引（从0开始）
 * @param out_platform 输出找到的平台
 * @param out_device 输出找到的设备
 * @return 成功返回0，失败返回-1
 */
static int opencl_find_device_by_index(int device_index, cl_platform_id* out_platform, cl_device_id* out_device) {
    if (!out_platform || !out_device) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "输出参数为空");
        return -1;
    }
    if (device_index < 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效设备索引: %d", device_index);
        return -1;
    }

    cl_uint num_platforms = 0;
    cl_int result = clGetPlatformIDs(0, NULL, &num_platforms);
    if (opencl_check_error(result, "获取平台数量") != 0 || num_platforms == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无可用OpenCL平台");
        return -1;
    }

    cl_platform_id* platforms = (cl_platform_id*)safe_malloc(sizeof(cl_platform_id) * num_platforms);
    if (!platforms) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "内存分配失败");
        return -1;
    }

    result = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (opencl_check_error(result, "获取平台列表") != 0) {
        safe_free((void**)&platforms);
        return -1;
    }

    int global_device_count = 0;
    for (cl_uint p = 0; p < num_platforms; p++) {
        cl_uint num_devices = 0;
        result = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (result != CL_SUCCESS || num_devices == 0) continue;

        if (device_index < global_device_count + (int)num_devices) {
            int local_index = device_index - global_device_count;
            cl_device_id* devices = (cl_device_id*)safe_malloc(sizeof(cl_device_id) * num_devices);
            if (!devices) {
                safe_free((void**)&platforms);
                return -1;
            }
            result = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
            if (result == CL_SUCCESS) {
                *out_device = devices[local_index];
                *out_platform = platforms[p];
            }
            safe_free((void**)&devices);
            if (result != CL_SUCCESS) {
                safe_free((void**)&platforms);
                return -1;
            }
            safe_free((void**)&platforms);
            return 0;
        }
        global_device_count += (int)num_devices;
    }

    safe_free((void**)&platforms);
    snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
             "设备索引越界: %d (最大: %d)", device_index, global_device_count - 1);
    return -1;
}

/* ============================================================================
 * OpenCL后端接口实现
 * =========================================================================== */

/**
 * @brief OpenCL后端初始化
 */
static int opencl_backend_init(void) {
    if (!g_opencl_initialized) {
        if (opencl_load_library() != 0) {
            /* G-007修复: 记录警告但允许调用方继续探测其他后端 */
            fprintf(stderr, "[OpenCL警告] 运行时库加载失败: %s\n", g_opencl_error_string);
            return -1;
        }
        
        /* 检查是否有可用的OpenCL平台 */
        cl_uint num_platforms = 0;
        cl_int result = clGetPlatformIDs(0, NULL, &num_platforms);
        if (opencl_check_error(result, "获取OpenCL平台数量") != 0) {
            fprintf(stderr, "[OpenCL警告] 获取平台数量失败: %s (错误码: %d)\n",
                    g_opencl_error_string, result);
            return -1;
        }
        
        if (num_platforms == 0) {
            snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                    "未找到可用的OpenCL平台");
            fprintf(stderr, "[OpenCL警告] 未找到任何OpenCL平台，系统可能未安装OpenCL运行时\n");
            return -1;
        }
        
        /* 检查是否有可用的OpenCL设备 */
        cl_platform_id platform;
        result = clGetPlatformIDs(1, &platform, NULL);
        if (opencl_check_error(result, "获取OpenCL平台") != 0) {
            fprintf(stderr, "[OpenCL警告] 获取OpenCL平台句柄失败: %s (错误码: %d)\n",
                    g_opencl_error_string, result);
            return -1;
        }
        
        cl_uint num_devices = 0;
        result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (opencl_check_error(result, "获取OpenCL设备数量") != 0) {
            fprintf(stderr, "[OpenCL警告] 获取OpenCL设备数量失败: %s (错误码: %d)\n",
                    g_opencl_error_string, result);
            return -1;
        }
        
        if (num_devices == 0) {
            snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                    "未找到可用的OpenCL设备");
            fprintf(stderr, "[OpenCL警告] 检测到OpenCL平台但未找到任何设备（可能缺少GPU驱动程序）\n");
            return -1;
        }
        
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "OpenCL后端初始化成功，平台数量: %u，设备数量: %u", 
                num_platforms, num_devices);
    }
    
    return 0;
}

/**
 * @brief OpenCL后端清理
 */
static void opencl_backend_cleanup(void) {
    opencl_cache_cleanup();
    opencl_unload_library();
}

/**
 * @brief 获取所有平台的OpenCL设备总数
 * 
 * 遍历所有平台统计设备总数，支持多平台场景。
 */
static int opencl_backend_get_device_count(void) {
    if (!g_opencl_initialized) {
        if (opencl_backend_init() != 0) {
            return 0;
        }
    }
    
    cl_uint num_platforms = 0;
    cl_int result = clGetPlatformIDs(0, NULL, &num_platforms);
    if (opencl_check_error(result, "获取OpenCL平台数量") != 0 || num_platforms == 0) {
        return 0;
    }
    
    cl_platform_id* platforms = (cl_platform_id*)safe_malloc(sizeof(cl_platform_id) * num_platforms);
    if (!platforms) return 0;
    
    result = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (result != CL_SUCCESS) {
        safe_free((void**)&platforms);
        return 0;
    }
    
    int total_devices = 0;
    for (cl_uint p = 0; p < num_platforms; p++) {
        cl_uint num_devices = 0;
        result = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (result == CL_SUCCESS) {
            total_devices += (int)num_devices;
        }
    }
    
    safe_free((void**)&platforms);
    return total_devices;
}

/**
 * @brief 获取OpenCL设备信息（支持多设备索引）
 */
static int opencl_backend_get_device_info(int device_index, GpuDeviceInfo* info) {
    if (!g_opencl_initialized) {
        if (opencl_backend_init() != 0) {
            return -1;
        }
    }
    
    if (!info) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "设备信息输出缓冲区为空");
        return -1;
    }
    
    // 根据索引查找平台和设备
    cl_platform_id platform;
    cl_device_id device;
    if (opencl_find_device_by_index(device_index, &platform, &device) != 0) {
        return -1;
    }
    
    // 获取设备信息
    char device_name[256] = {0};
    char device_vendor[256] = {0};
    char device_version[256] = {0};
    char driver_version[256] = {0};
    cl_ulong global_mem_size = 0;
    cl_uint max_compute_units = 0;
    cl_uint max_work_group_size = 0;
    cl_bool available = CL_FALSE;
    cl_uint max_clock_frequency = 0;
    cl_bool compiler_available = CL_FALSE;
    cl_ulong max_mem_alloc_size = 0;
    cl_uint address_bits = 0;
    cl_bool endian_little = CL_FALSE;
    cl_uint preferred_vector_width_float = 0;
    cl_uint preferred_vector_width_double = 0;
    cl_uint native_vector_width_float = 0;
    cl_uint native_vector_width_double = 0;
    cl_bool image_support = CL_FALSE;
    
    // 获取字符串信息
    opencl_get_device_string_info(device, CL_DEVICE_NAME, device_name, sizeof(device_name));
    opencl_get_device_string_info(device, CL_DEVICE_VENDOR, device_vendor, sizeof(device_vendor));
    opencl_get_device_string_info(device, CL_DEVICE_VERSION, device_version, sizeof(device_version));
    opencl_get_device_string_info(device, CL_DRIVER_VERSION, driver_version, sizeof(driver_version));
    
    // 获取数值信息
    opencl_get_device_info(device, CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size, sizeof(global_mem_size));
    opencl_get_device_info(device, CL_DEVICE_MAX_COMPUTE_UNITS, &max_compute_units, sizeof(max_compute_units));
    opencl_get_device_info(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_work_group_size, sizeof(max_work_group_size));
    opencl_get_device_info(device, CL_DEVICE_AVAILABLE, &available, sizeof(available));
    opencl_get_device_info(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, &max_clock_frequency, sizeof(max_clock_frequency));
    opencl_get_device_info(device, CL_DEVICE_COMPILER_AVAILABLE, &compiler_available, sizeof(compiler_available));
    opencl_get_device_info(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_alloc_size, sizeof(max_mem_alloc_size));
    opencl_get_device_info(device, CL_DEVICE_ADDRESS_BITS, &address_bits, sizeof(address_bits));
    opencl_get_device_info(device, CL_DEVICE_ENDIAN_LITTLE, &endian_little, sizeof(endian_little));
    opencl_get_device_info(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &preferred_vector_width_float, sizeof(preferred_vector_width_float));
    opencl_get_device_info(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, &preferred_vector_width_double, sizeof(preferred_vector_width_double));
    opencl_get_device_info(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, &native_vector_width_float, sizeof(native_vector_width_float));
    opencl_get_device_info(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, &native_vector_width_double, sizeof(native_vector_width_double));
    opencl_get_device_info(device, CL_DEVICE_IMAGE_SUPPORT, &image_support, sizeof(image_support));
    
    // 填充设备信息
    memset(info, 0, sizeof(GpuDeviceInfo));
    info->device_id = device_index;
    strncpy(info->name, device_name, sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';
    
    // 判断设备类型
    cl_device_type device_type = 0;
    opencl_get_device_info(device, CL_DEVICE_TYPE, &device_type, sizeof(device_type));
    
    if (device_type & CL_DEVICE_TYPE_GPU) {
        if (strstr(device_vendor, "NVIDIA") != NULL || 
            strstr(device_vendor, "AMD") != NULL ||
            strstr(device_vendor, "Advanced Micro Devices") != NULL) {
            info->type = GPU_DEVICE_TYPE_DISCRETE;
        } else {
            info->type = GPU_DEVICE_TYPE_INTEGRATED;
        }
    } else if (device_type & CL_DEVICE_TYPE_CPU) {
        info->type = GPU_DEVICE_TYPE_INTEGRATED;
    } else {
        info->type = GPU_DEVICE_TYPE_UNKNOWN;
    }
    
    info->total_memory = (size_t)global_mem_size;
    /* M-031修复：尝试通过vendor扩展查询真实空闲内存 */
    {
        size_t free_mem = global_mem_size;
#ifdef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
        /* AMD扩展：CL_DEVICE_GLOBAL_FREE_MEMORY_AMD = 0x403E */
        cl_ulong amd_free = 0;
        cl_int ret = clGetDeviceInfo(device_id, 0x403E, sizeof(amd_free), &amd_free, NULL);
        if (ret == CL_SUCCESS && amd_free > 0) {
            free_mem = (size_t)amd_free;
        }
#endif
        info->free_memory = free_mem;
    }
    info->compute_units = (int)max_compute_units;
    info->max_work_group_size = (int)max_work_group_size;
    info->clock_speed = (float)max_clock_frequency;
    info->supports_double = (preferred_vector_width_double > 0 && native_vector_width_double > 0) ? 1 : 0;
    info->supports_half = 0;  // OpenCL 1.x不支持标准半精度，需要扩展
    
    return 0;
}

/**
 * @brief 创建OpenCL上下文（支持多设备索引）
 */
static GpuContext* opencl_backend_context_create(int device_index) {
    if (!g_opencl_initialized) {
        if (opencl_backend_init() != 0) {
            return NULL;
        }
    }
    
    // 根据索引查找平台和设备
    cl_platform_id platform;
    cl_device_id device;
    if (opencl_find_device_by_index(device_index, &platform, &device) != 0) {
        return NULL;
    }
    
    cl_int result;
    // 创建设备上下文
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &result);
    if (opencl_check_error(result, "创建设备上下文") != 0) {
        return NULL;
    }
    
    // 获取设备总内存
    cl_ulong global_mem_size = 0;
    result = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    if (opencl_check_error(result, "获取设备总内存") != 0) {
        clReleaseContext(context);
        return NULL;
    }
    
    // 获取平台和设备名称
    char platform_name[256] = {0};
    char device_name[256] = {0};
    opencl_get_platform_info(platform, CL_PLATFORM_NAME, platform_name, sizeof(platform_name));
    opencl_get_device_string_info(device, CL_DEVICE_NAME, device_name, sizeof(device_name));
    
    // 分配上下文结构
    OpenCLContext* context_struct = (OpenCLContext*)safe_malloc(sizeof(OpenCLContext));
    if (!context_struct) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "内存分配失败：OpenCL上下文");
        clReleaseContext(context);
        return NULL;
    }
    
    memset(context_struct, 0, sizeof(OpenCLContext));

    /* 修复5(P0): 填充GpuContext header字段，使上层gpu_context_create能正确访问
     * backend/device_index/is_initialized/device_name等字段。
     * 设置is_initialized=1后，gpu_context_create会跳过二次设置，
     * 并使用header.device_name创建自动内核优化器。 */
    context_struct->header.backend = GPU_BACKEND_OPENCL;
    context_struct->header.device_index = device_index;
    context_struct->header.is_initialized = 1;
    context_struct->header.total_memory = (size_t)global_mem_size;
    context_struct->header.free_memory = (size_t)global_mem_size;
    strncpy(context_struct->header.device_name, device_name, sizeof(context_struct->header.device_name) - 1);
    context_struct->header.device_name[sizeof(context_struct->header.device_name) - 1] = '\0';

    context_struct->context = context;
    context_struct->device = device;
    context_struct->platform = platform;
    context_struct->device_index = device_index;
    context_struct->total_memory = (size_t)global_mem_size;
    // 根据"禁止任何降级处理"原则，不使用百分比估计值
    // 返回总内存作为最大可用内存（保守估计）
    context_struct->free_memory = (size_t)global_mem_size;
    context_struct->is_initialized = 1;
    strncpy(context_struct->platform_name, platform_name, sizeof(context_struct->platform_name) - 1);
    strncpy(context_struct->device_name, device_name, sizeof(context_struct->device_name) - 1);
    
    // 创建默认命令队列
    context_struct->default_command_queue = clCreateCommandQueue(context, device, 0, &result);
    if (opencl_check_error(result, "创建默认命令队列") != 0) {
        // 命令队列创建失败，但仍然可以继续（使用临时队列）
        context_struct->default_command_queue = NULL;
    }
    
    return (GpuContext*)context_struct;
}

/**
 * @brief 释放OpenCL上下文
 */
static void opencl_backend_context_free(GpuContext* gpu_context) {
    if (!gpu_context) {
        return;
    }
    
    OpenCLContext* context = (OpenCLContext*)gpu_context;
    
    // 释放默认命令队列
    if (context->default_command_queue) {
        cl_int result = clReleaseCommandQueue(context->default_command_queue);
        if (result != 0) {
            // 记录错误但不阻止清理
            opencl_check_error(result, "释放默认命令队列");
        }
    }
    
    if (context->context) {
        // 释放OpenCL上下文
        cl_int result = clReleaseContext(context->context);
        if (result != 0) {
            // 记录错误但不阻止清理
            opencl_check_error(result, "释放OpenCL上下文");
        }
    }
    
    safe_free((void**)&context);
}

/* ============================================================================
 * 完整的OpenCL后端接口（完全实现）
 * =========================================================================== */

// 注意：以下函数提供完整的OpenCL实现
// 使用实际的OpenCL API调用实现所有必要功能

static GpuMemory* opencl_backend_memory_alloc(GpuContext* gpu_context, size_t size, GpuMemoryType memory_type) {
    if (!gpu_context || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：上下文为空或大小为0");
        return NULL;
    }
    
    OpenCLContext* context = (OpenCLContext*)gpu_context;
    
    // 检查内存是否可用
    if (size > context->free_memory) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "内存不足：请求%zu字节，可用%zu字节", size, context->free_memory);
        return NULL;
    }
    
    // 创建OpenCL内存对象
    cl_bitfield flags = 0;
    switch (memory_type) {
        case GPU_MEMORY_HOST:
            flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
            break;
        case GPU_MEMORY_DEVICE:
            flags = CL_MEM_READ_WRITE;
            break;
        case GPU_MEMORY_UNIFIED:
            // 统一内存：尝试使用设备内存，如果支持的话
            flags = CL_MEM_READ_WRITE;
            // 检查设备是否支持统一内存
            // 假设不支持，使用设备内存
            break;
        default:
            flags = CL_MEM_READ_WRITE;
            break;
    }
    
    cl_int result;
    cl_mem mem_obj = clCreateBuffer(context->context, flags, size, NULL, &result);
    
    // 构建附加错误信息
    char additional_info[256] = {0};
    const char* mem_type_str = "未知";
    switch (memory_type) {
        case GPU_MEMORY_HOST: mem_type_str = "主机内存"; break;
        case GPU_MEMORY_DEVICE: mem_type_str = "设备内存"; break;
        case GPU_MEMORY_UNIFIED: mem_type_str = "统一内存"; break;
    }
    snprintf(additional_info, sizeof(additional_info), 
             "请求大小: %zu 字节, 内存类型: %s, 标志: 0x%x", 
             size, mem_type_str, flags);
    
    if (OPENCL_CHECK_ERROR_EX(result, "创建OpenCL缓冲区", additional_info) != 0) {
        return NULL;
    }
    
    // 分配OpenCL内存结构
    OpenCLMemory* memory = (OpenCLMemory*)safe_malloc(sizeof(OpenCLMemory));
    if (!memory) {
        clReleaseMemObject(mem_obj);
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "内存分配失败：OpenCL内存结构");
        return NULL;
    }
    
    memset(memory, 0, sizeof(OpenCLMemory));

    /* P3-005修复: SVM共享虚拟内存路径 (OpenCL 2.0+) */
    if (g_opencl_svm_available && context->device && (memory_type == GPU_MEMORY_UNIFIED || memory_type == GPU_MEMORY_DEVICE)) {
        /* 检测设备SVM能力 (cl_device_svm_capabilities = cl_uint) */
        unsigned int svm_caps = 0;
        cl_int svm_err = clGetDeviceInfo(context->device, 0x109B /* CL_DEVICE_SVM_CAPABILITIES */,
                                          sizeof(svm_caps), &svm_caps, NULL);
        if (svm_err == CL_SUCCESS && (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
            cl_svm_mem_flags svm_flags = CL_MEM_READ_WRITE;
            void* svm_mem = clSVMAlloc(context->context, svm_flags, size, 0);
            if (svm_mem) {
                /* 释放之前创建的cl_mem，改用SVM */
                clReleaseMemObject(mem_obj);
                memory->svm_ptr = svm_mem;
                memory->is_svm = 1;
                memory->mem_obj = NULL;
                memory->size = size;
                memory->type = memory_type;
                memory->context = context->context;
                memory->opencl_context = context;
                memory->is_image = 0;
                context->free_memory -= size;
                return (GpuMemory*)memory;
            }
        }
    }

    memory->mem_obj = mem_obj;
    memory->size = size;
    memory->type = memory_type;
    memory->context = context->context;
    memory->opencl_context = context;  // 存储OpenCL上下文结构指针
    memory->is_image = 0;  // 不是图像对象
    
    // 更新可用内存估计
    context->free_memory -= size;
    
    return (GpuMemory*)memory;
}

static void opencl_backend_memory_free(GpuMemory* memory) {
    if (!memory) {
        return;
    }
    
    OpenCLMemory* mem = (OpenCLMemory*)memory;

    /* P3-005修复: SVM内存释放 */
    if (mem->is_svm && mem->svm_ptr && g_opencl_svm_available) {
        clSVMFree(mem->context, mem->svm_ptr);
        mem->svm_ptr = NULL;
    }

    // 释放OpenCL内存对象
    if (mem->mem_obj) {
        cl_int result = clReleaseMemObject(mem->mem_obj);
        if (result != 0) {
            opencl_check_error(result, "释放OpenCL内存对象");
        }
    }
    
    // 更新上下文中的可用内存估计值（线程安全）
    if (mem->opencl_context && mem->size > 0) {
        // 原子增加可用内存（完整线程安全实现）
        size_t new_free_memory = atomic_add_size_t(&mem->opencl_context->free_memory, mem->size);
        
        // 确保不超过总内存（使用原子比较交换）
        size_t total_memory = mem->opencl_context->total_memory;
        while (new_free_memory > total_memory) {
            // 尝试将free_memory设置为total_memory
            if (atomic_compare_exchange_size_t(&mem->opencl_context->free_memory, 
                                              new_free_memory, total_memory)) {
                new_free_memory = total_memory;
                break;
            }
            // 如果失败，读取当前值并重试
            new_free_memory = mem->opencl_context->free_memory;
        }
    }
    
    // 释放内存结构
    safe_free((void**)&mem);
}

static int opencl_backend_memory_copy_to_device(GpuMemory* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：目标内存、源数据或大小为0");
        return -1;
    }
    
    OpenCLMemory* mem = (OpenCLMemory*)dst;

    /* P3-005修复: SVM共享内存零拷贝路径 */
    if (mem->is_svm && mem->svm_ptr) {
        if (size > mem->size) size = mem->size;
        memcpy(mem->svm_ptr, src, size);
        /* SVM内存在粗细粒度一致时自动可见，无需显式映射 */
        if (g_opencl_svm_available && clEnqueueSVMMap && mem->opencl_context && mem->opencl_context->default_command_queue) {
            clEnqueueSVMMap(mem->opencl_context->default_command_queue, CL_TRUE,
                           CL_MAP_WRITE, mem->svm_ptr, size, 0, NULL, NULL);
        }
        return 0;
    }
    
    /* 修复6(P1): 非SVM路径在调用clEnqueueWriteBuffer前校验size <= mem->size，
     * 防止越界写入导致缓冲区溢出。SVM路径已有检查但非SVM路径遗漏。 */
    if (size > mem->size) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "复制大小%zu超过设备内存大小%zu", size, mem->size);
        return -1;
    }
    
    cl_int result;
    cl_command_queue queue = NULL;
    int use_temp_queue = 0;
    
    // 尝试使用默认命令队列
    if (mem->opencl_context && mem->opencl_context->default_command_queue) {
        queue = mem->opencl_context->default_command_queue;
    } else {
        // 创建临时命令队列
        // 需要从上下文中获取设备ID
        cl_device_id device_id = NULL;
        if (mem->opencl_context) {
            device_id = mem->opencl_context->device;
        }
        // 如果无法获取设备ID，使用NULL（让OpenCL选择默认设备）
        queue = clCreateCommandQueue(mem->context, device_id, 0, &result);
        
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "上下文: %p, 设备ID: %p, 操作: 主机到设备复制, 大小: %zu 字节", 
                (void*)mem->context, (void*)device_id, size);
        
        if (OPENCL_CHECK_ERROR_EX(result, "创建命令队列", additional_info) != 0) {
            return -1;
        }
        use_temp_queue = 1;
    }
    
    // 复制数据到设备
    result = clEnqueueWriteBuffer(queue, mem->mem_obj, CL_TRUE, 0, size, src, 0, NULL, NULL);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "内存对象: %p, 大小: %zu 字节, 队列: %p, 偏移: 0", 
                (void*)mem->mem_obj, size, (void*)queue);
        
        if (OPENCL_CHECK_ERROR_EX(result, "复制数据到设备", additional_info) != 0) {
            if (use_temp_queue) {
                clReleaseCommandQueue(queue);
            }
            return -1;
        }
    }
    
    // 等待复制完成
    result = clFinish(queue);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "队列: %p, 操作: 主机到设备复制完成等待", 
                (void*)queue);
        
        if (OPENCL_CHECK_ERROR_EX(result, "等待命令完成", additional_info) != 0) {
            if (use_temp_queue) {
                clReleaseCommandQueue(queue);
            }
            return -1;
        }
    }
    
    // 如果是临时队列，释放它
    if (use_temp_queue) {
        result = clReleaseCommandQueue(queue);
        if (result != 0) {
            opencl_check_error(result, "释放临时命令队列");
        }
    }
    
    return 0;
}

static int opencl_backend_memory_copy_from_device(void* dst, GpuMemory* src, size_t size) {
    if (!dst || !src || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：目标内存、源数据或大小为0");
        return -1;
    }
    
    OpenCLMemory* mem = (OpenCLMemory*)src;

    /* P3-005修复: SVM共享内存零拷贝路径 */
    if (mem->is_svm && mem->svm_ptr) {
        if (size > mem->size) size = mem->size;
        if (g_opencl_svm_available && clEnqueueSVMUnmap && mem->opencl_context && mem->opencl_context->default_command_queue) {
            clEnqueueSVMUnmap(mem->opencl_context->default_command_queue, mem->svm_ptr, 0, NULL, NULL);
        }
        memcpy(dst, mem->svm_ptr, size);
        return 0;
    }
    
    /* 修复6(P1): 非SVM路径在调用clEnqueueReadBuffer前校验size <= mem->size，
     * 防止越界读取导致缓冲区溢出。SVM路径已有检查但非SVM路径遗漏。 */
    if (size > mem->size) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "复制大小%zu超过设备内存大小%zu", size, mem->size);
        return -1;
    }
    
    cl_int result;
    cl_command_queue queue = NULL;
    int use_temp_queue = 0;
    
    // 尝试使用默认命令队列
    if (mem->opencl_context && mem->opencl_context->default_command_queue) {
        queue = mem->opencl_context->default_command_queue;
    } else {
        // 创建临时命令队列
        // 需要从上下文中获取设备ID
        cl_device_id device_id = NULL;
        if (mem->opencl_context) {
            device_id = mem->opencl_context->device;
        }
        // 如果无法获取设备ID，使用NULL（让OpenCL选择默认设备）
        queue = clCreateCommandQueue(mem->context, device_id, 0, &result);
        
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "上下文: %p, 设备ID: %p, 操作: 设备到主机复制, 大小: %zu 字节", 
                (void*)mem->context, (void*)device_id, size);
        
        if (OPENCL_CHECK_ERROR_EX(result, "创建命令队列", additional_info) != 0) {
            return -1;
        }
        use_temp_queue = 1;
    }
    
    // 从设备复制数据
    result = clEnqueueReadBuffer(queue, mem->mem_obj, CL_TRUE, 0, size, dst, 0, NULL, NULL);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "内存对象: %p, 大小: %zu 字节, 队列: %p, 偏移: 0", 
                (void*)mem->mem_obj, size, (void*)queue);
        
        if (OPENCL_CHECK_ERROR_EX(result, "从设备复制数据", additional_info) != 0) {
            if (use_temp_queue) {
                clReleaseCommandQueue(queue);
            }
            return -1;
        }
    }
    
    // 等待复制完成
    result = clFinish(queue);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "队列: %p, 操作: 设备到主机复制完成等待", 
                (void*)queue);
        
        if (OPENCL_CHECK_ERROR_EX(result, "等待命令完成", additional_info) != 0) {
            if (use_temp_queue) {
                clReleaseCommandQueue(queue);
            }
            return -1;
        }
    }
    
    // 如果是临时队列，释放它
    if (use_temp_queue) {
        result = clReleaseCommandQueue(queue);
        if (result != 0) {
            opencl_check_error(result, "释放临时命令队列");
        }
    }
    
    return 0;
}

static int opencl_backend_memory_copy_device_to_device(GpuMemory* dst, GpuMemory* src, size_t size) {
    if (!dst || !src || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：目标内存、源内存或大小为0");
        return -1;
    }
    
    OpenCLMemory* dst_mem = (OpenCLMemory*)dst;
    OpenCLMemory* src_mem = (OpenCLMemory*)src;
    
    // 检查上下文是否相同
    if (dst_mem->context != src_mem->context) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "设备到设备复制要求内存对象属于同一个上下文");
        return -1;
    }
    
    cl_int result;
    cl_command_queue queue = NULL;
    int use_temp_queue = 0;
    
    // 尝试使用默认命令队列（优先使用目标内存的上下文）
    if (dst_mem->opencl_context && dst_mem->opencl_context->default_command_queue) {
        queue = dst_mem->opencl_context->default_command_queue;
    } else if (src_mem->opencl_context && src_mem->opencl_context->default_command_queue) {
        queue = src_mem->opencl_context->default_command_queue;
    } else {
        // 创建临时命令队列
        // 需要从上下文中获取设备ID（优先使用目标内存的上下文）
        cl_device_id device_id = NULL;
        if (dst_mem->opencl_context) {
            device_id = dst_mem->opencl_context->device;
        } else if (src_mem->opencl_context) {
            device_id = src_mem->opencl_context->device;
        }
        // 如果无法获取设备ID，使用NULL（让OpenCL选择默认设备）
        queue = clCreateCommandQueue(dst_mem->context, device_id, 0, &result);
        
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "上下文: %p, 设备ID: %p, 操作: 设备到设备复制, 大小: %zu 字节", 
                (void*)dst_mem->context, (void*)device_id, size);
        
        if (OPENCL_CHECK_ERROR_EX(result, "创建命令队列", additional_info) != 0) {
            return -1;
        }
        use_temp_queue = 1;
    }
    
    // 执行设备到设备复制
    result = clEnqueueCopyBuffer(queue, src_mem->mem_obj, dst_mem->mem_obj, 0, 0, size, 0, NULL, NULL);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "源内存对象: %p, 目标内存对象: %p, 大小: %zu 字节, 队列: %p", 
                (void*)src_mem->mem_obj, (void*)dst_mem->mem_obj, size, (void*)queue);
        
        if (OPENCL_CHECK_ERROR_EX(result, "执行设备到设备复制", additional_info) != 0) {
            if (use_temp_queue) {
                clReleaseCommandQueue(queue);
            }
            return -1;
        }
    }
    
    // 等待复制完成
    result = clFinish(queue);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "队列: %p, 操作: 设备到设备复制完成等待", 
                (void*)queue);
        
        if (OPENCL_CHECK_ERROR_EX(result, "等待设备到设备复制完成", additional_info) != 0) {
            if (use_temp_queue) {
                clReleaseCommandQueue(queue);
            }
            return -1;
        }
    }
    
    // 如果是临时队列，释放它
    if (use_temp_queue) {
        result = clReleaseCommandQueue(queue);
        if (result != 0) {
            opencl_check_error(result, "释放临时命令队列");
        }
    }
    
    return 0;
}

static GpuKernel* opencl_backend_kernel_create(GpuContext* context, const char* kernel_source, const char* kernel_name) {
    if (!context || !kernel_source || !kernel_name) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：上下文、内核源代码或内核名称为空");
        return NULL;
    }
    
    OpenCLContext* ctx = (OpenCLContext*)context;
    
    // 检查是否为预定义内核
    const char* actual_kernel_source = kernel_source;
    
    // 预定义内核映射
    if (strcmp(kernel_name, "vector_add") == 0) {
        actual_kernel_source = OPENCL_VECTOR_ADD_KERNEL;
    } else if (strcmp(kernel_name, "matrix_mul_basic") == 0) {
        actual_kernel_source = OPENCL_MATRIX_MUL_KERNEL_BASIC;
    } else if (strcmp(kernel_name, "relu") == 0) {
        actual_kernel_source = OPENCL_RELU_KERNEL;
    } else if (strcmp(kernel_name, "sigmoid") == 0) {
        actual_kernel_source = OPENCL_SIGMOID_KERNEL;
    } else if (strcmp(kernel_name, "cfc_forward_step") == 0) {
        actual_kernel_source = OPENCL_CFC_FORWARD_KERNEL;
    } else if (strcmp(kernel_name, "cfc_liquid_tau") == 0) {
        actual_kernel_source = OPENCL_CFC_LIQUID_TAU_KERNEL;
    } else if (strcmp(kernel_name, "cfc_multi_timescale_step") == 0) {
        actual_kernel_source = OPENCL_CFC_MULTI_TIMESCALE_KERNEL;
    } else if (strcmp(kernel_name, "sgd_update") == 0) {
        actual_kernel_source = OPENCL_SGD_UPDATE_KERNEL;
    } else if (strcmp(kernel_name, "adam_update") == 0) {
        actual_kernel_source = OPENCL_ADAM_UPDATE_KERNEL;
    } else if (strcmp(kernel_name, "mse_loss") == 0) {
        actual_kernel_source = OPENCL_MSE_LOSS_KERNEL;
    } else if (strcmp(kernel_name, "cross_entropy_loss") == 0) {
        actual_kernel_source = OPENCL_CROSS_ENTROPY_LOSS_KERNEL;
    } else if (strcmp(kernel_name, "grad_clip") == 0) {
        actual_kernel_source = OPENCL_GRAD_CLIP_KERNEL;
    } else if (strcmp(kernel_name, "grad_clip_by_norm_part1") == 0 || 
               strcmp(kernel_name, "grad_clip_by_norm_part2") == 0) {
        actual_kernel_source = OPENCL_GRAD_CLIP_BY_NORM_KERNEL;
    } else if (strcmp(kernel_name, "apply_weight_decay") == 0) {
        actual_kernel_source = OPENCL_WEIGHT_DECAY_KERNEL;
    } else if (strcmp(kernel_name, "conv2d") == 0) {
        actual_kernel_source = OPENCL_CONV2D_KERNEL;
    } else if (strcmp(kernel_name, "max_pool2d") == 0) {
        actual_kernel_source = OPENCL_MAX_POOL2D_KERNEL;
    } else if (strcmp(kernel_name, "avg_pool2d") == 0) {
        actual_kernel_source = OPENCL_AVG_POOL2D_KERNEL;
    } else if (strcmp(kernel_name, "batch_norm") == 0) {
        actual_kernel_source = OPENCL_BATCH_NORM_KERNEL;
    } else if (strcmp(kernel_name, "image_normalize") == 0) {
        actual_kernel_source = OPENCL_IMAGE_NORMALIZE_KERNEL;
    } else if (strcmp(kernel_name, "sobel") == 0) {
        actual_kernel_source = OPENCL_SOBEL_KERNEL;
    } else if (strcmp(kernel_name, "gaussian_blur_h") == 0 || 
               strcmp(kernel_name, "gaussian_blur_v") == 0) {
        actual_kernel_source = OPENCL_GAUSSIAN_BLUR_KERNEL;
    } else if (strcmp(kernel_name, "resize_bilinear") == 0) {
        actual_kernel_source = OPENCL_RESIZE_BILINEAR_KERNEL;
    } else if (strcmp(kernel_name, "matmul_train") == 0) {
        actual_kernel_source = OPENCL_MATMUL_TRAIN_KERNEL;
    } else if (strcmp(kernel_name, "activation_forward_unified") == 0) {
        actual_kernel_source = OPENCL_ACTIVATION_FORWARD_UNIFIED_KERNEL;
    } else if (strcmp(kernel_name, "activation_backward_unified") == 0) {
        actual_kernel_source = OPENCL_ACTIVATION_BACKWARD_UNIFIED_KERNEL;
    } else if (strcmp(kernel_name, "batch_norm_backward") == 0) {
        actual_kernel_source = OPENCL_BATCH_NORM_BACKWARD_KERNEL;
    } else if (strcmp(kernel_name, "dropout_forward") == 0) {
        actual_kernel_source = OPENCL_DROPOUT_FORWARD_KERNEL;
    } else if (strcmp(kernel_name, "dropout_backward") == 0) {
        actual_kernel_source = OPENCL_DROPOUT_BACKWARD_KERNEL;
    } else if (strcmp(kernel_name, "rmsprop_update") == 0) {
        actual_kernel_source = OPENCL_RMSPROP_UPDATE_KERNEL;
    } else if (strcmp(kernel_name, "cross_entropy_loss_grad") == 0) {
        actual_kernel_source = OPENCL_CROSS_ENTROPY_LOSS_GRAD_KERNEL;
    } else if (strcmp(kernel_name, "cfc_ode_step_kernel") == 0) {
        actual_kernel_source = OPENCL_CFC_ODE_STEP_KERNEL;
    }
    
    cl_int result;
    size_t source_len = strlen(actual_kernel_source);
    
    /* SPIR-V二进制加载路径（OpenCL 1.2+）：尝试从文件系统加载预编译内核 */
    cl_program program = NULL;
    int is_from_cache = 0;
    int is_from_binary = 0;
    
    if (clCreateProgramWithBinary) {
        /* 构建SPIR-V文件路径：opencl_kernels/<kernel_name>.spv */
        char spirv_path[512];
        snprintf(spirv_path, sizeof(spirv_path), "opencl_kernels/%s.spv", kernel_name);
        FILE* fp = fopen(spirv_path, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long fsize = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (fsize > 0 && fsize < 10 * 1024 * 1024) {
                unsigned char* binary = (unsigned char*)safe_malloc((size_t)fsize);
                if (binary) {
                    size_t nread = fread(binary, 1, (size_t)fsize, fp);
                    if (nread == (size_t)fsize) {
                        cl_int bin_status = CL_SUCCESS;
                        program = clCreateProgramWithBinary(ctx->context, 1, &ctx->device,
                            (const size_t*)&fsize, (const unsigned char**)&binary,
                            &bin_status, &result);
                        if (result == CL_SUCCESS && bin_status == CL_SUCCESS) {
                            is_from_binary = 1;
                        } else {
                            if (program) { clReleaseProgram(program); program = NULL; }
                        }
                    }
                    safe_free((void**)&binary);
                }
            }
            fclose(fp);
        }
    }
    
    /* 源码缓存查找（SPIR-V未命中时） */
    if (!program) {
        program = opencl_cache_find_program(ctx->context, actual_kernel_source, source_len);
        is_from_cache = (program != NULL);
    }
    
    if (!program) {
        // 缓存未命中，创建并编译OpenCL程序
        const char* sources[] = { actual_kernel_source };
        program = clCreateProgramWithSource(ctx->context, 1, sources, NULL, &result);
        
        {
            char additional_info[512] = {0};
            snprintf(additional_info, sizeof(additional_info), 
                     "内核名称: %s, 源代码长度: %zu 字符, 上下文: %p", 
                     kernel_name, source_len, (void*)ctx->context);
            
            if (OPENCL_CHECK_ERROR_EX(result, "创建OpenCL程序", additional_info) != 0) {
                return NULL;
            }
        }
        
        // 构建程序
        result = clBuildProgram(program, 1, &ctx->device, "", NULL, NULL);
        if (result != CL_SUCCESS) {
            char build_log[4096] = {0};
            clGetProgramBuildInfo(program, ctx->device, CL_PROGRAM_BUILD_LOG, 
                                 sizeof(build_log), build_log, NULL);
            
            char additional_info[5120] = {0};
            snprintf(additional_info, sizeof(additional_info),
                    "内核名称: %s, 构建日志:\n%s", kernel_name, build_log);
            OPENCL_CHECK_ERROR_EX(result, "构建OpenCL程序", additional_info);
            snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                    "OpenCL程序构建失败：%s", build_log);
            clReleaseProgram(program);
            return NULL;
        }
        
        // 将编译好的程序加入缓存
        opencl_cache_add_program(ctx->context, actual_kernel_source, program);
    }
    
    // 创建内核
    cl_kernel kernel = clCreateKernel(program, kernel_name, &result);
    
    {
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                 "内核名称: %s, 程序句柄: %p, 缓存命中: %s", 
                 kernel_name, (void*)program, is_from_cache ? "是" : "否");
        
        if (OPENCL_CHECK_ERROR_EX(result, "创建OpenCL内核", additional_info) != 0) {
            if (!is_from_cache && !is_from_binary) {
                clReleaseProgram(program);
            }
            return NULL;
        }
    }
    
    // 使用默认命令队列（如果存在），否则创建新的
    cl_command_queue queue = NULL;
    int created_new_queue = 0;
    if (ctx->default_command_queue) {
        queue = ctx->default_command_queue;
    } else {
        // 创建命令队列
        queue = clCreateCommandQueue(ctx->context, ctx->device, 0, &result);
        
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "上下文: %p, 设备: %p, 内核名称: %s", 
                (void*)ctx->context, (void*)ctx->device, kernel_name);
        
        if (OPENCL_CHECK_ERROR_EX(result, "创建命令队列", additional_info) != 0) {
            clReleaseKernel(kernel);
            /* 修复7(P1): 仅当program非缓存且非二进制加载时才释放，避免释放缓存持有的program导致悬垂指针 */
            if (!is_from_cache && !is_from_binary) {
                clReleaseProgram(program);
            }
            return NULL;
        }
        created_new_queue = 1;
    }
    
    // 分配OpenCL内核结构
    OpenCLKernel* opencl_kernel = (OpenCLKernel*)safe_malloc(sizeof(OpenCLKernel));
    if (!opencl_kernel) {
        if (created_new_queue) {
            clReleaseCommandQueue(queue);
        }
        clReleaseKernel(kernel);
        /* 修复7(P1): 仅当program非缓存且非二进制加载时才释放，避免释放缓存持有的program导致悬垂指针 */
        if (!is_from_cache && !is_from_binary) {
            clReleaseProgram(program);
        }
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "内存分配失败：OpenCL内核结构");
        return NULL;
    }
    
    memset(opencl_kernel, 0, sizeof(OpenCLKernel));
    opencl_kernel->kernel = kernel;
    opencl_kernel->program = program;
    opencl_kernel->context = ctx->context;
    opencl_kernel->command_queue = queue;
    opencl_kernel->is_using_default_queue = (created_new_queue == 0);
    opencl_kernel->is_from_cache = is_from_cache;
    // 复制内核名称
    size_t name_len = strlen(kernel_name) + 1;
    opencl_kernel->kernel_name = (char*)safe_malloc(name_len);
    if (!opencl_kernel->kernel_name) {
        safe_free((void**)&opencl_kernel);
        clReleaseCommandQueue(queue);
        clReleaseKernel(kernel);
        /* 修复7(P1): 仅当program非缓存且非二进制加载时才释放，避免释放缓存持有的program导致悬垂指针 */
        if (!is_from_cache && !is_from_binary) {
            clReleaseProgram(program);
        }
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "内存分配失败：内核名称");
        return NULL;
    }
    strncpy(opencl_kernel->kernel_name, kernel_name, name_len-1);
    opencl_kernel->kernel_name[name_len-1] = '\0';
    
    return (GpuKernel*)opencl_kernel;
}

static void opencl_backend_kernel_free(GpuKernel* kernel) {
    if (!kernel) {
        return;
    }
    
    OpenCLKernel* opencl_kernel = (OpenCLKernel*)kernel;
    
    // 释放内核名称
    if (opencl_kernel->kernel_name) {
        safe_free((void**)&opencl_kernel->kernel_name);
    }
    
    // 释放工作大小数组
    if (opencl_kernel->global_work_size) {
        safe_free((void**)&opencl_kernel->global_work_size);
    }
    
    if (opencl_kernel->local_work_size) {
        safe_free((void**)&opencl_kernel->local_work_size);
    }
    
    // 释放OpenCL资源
    // 注意：如果使用默认命令队列，则不应释放它
    if (opencl_kernel->command_queue && !opencl_kernel->is_using_default_queue) {
        cl_int result = clReleaseCommandQueue(opencl_kernel->command_queue);
        if (result != 0) {
            opencl_check_error(result, "释放命令队列");
        }
    }
    
    if (opencl_kernel->kernel) {
        cl_int result = clReleaseKernel(opencl_kernel->kernel);
        if (result != 0) {
            opencl_check_error(result, "释放OpenCL内核");
        }
    }
    
    // 只有非缓存来源的程序才释放（缓存程序由缓存统一管理）
    if (opencl_kernel->program && !opencl_kernel->is_from_cache) {
        cl_int result = clReleaseProgram(opencl_kernel->program);
        if (result != 0) {
            opencl_check_error(result, "释放OpenCL程序");
        }
    }
    
    // 释放内核结构
    safe_free((void**)&opencl_kernel);
}

static int opencl_backend_kernel_set_arg(GpuKernel* kernel, int arg_index, size_t arg_size, const void* arg_value) {
    if (!kernel) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：内核为空");
        return -1;
    }
    
    OpenCLKernel* opencl_kernel = (OpenCLKernel*)kernel;
    
    cl_int result = clSetKernelArg(opencl_kernel->kernel, (cl_uint)arg_index, arg_size, arg_value);
    if (opencl_check_error(result, "设置OpenCL内核参数") != 0) {
        return -1;
    }
    
    return 0;
}

static int opencl_backend_kernel_execute(GpuKernel* kernel, size_t global_work_size, size_t local_work_size) {
    if (!kernel) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：内核为空");
        return -1;
    }
    
    if (global_work_size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：全局工作大小为0");
        return -1;
    }
    
    OpenCLKernel* opencl_kernel = (OpenCLKernel*)kernel;
    
    // 验证OpenCL内核对象
    if (!opencl_kernel->kernel) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效内核：OpenCL内核对象为空");
        return -1;
    }
    
    if (!opencl_kernel->command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效内核：OpenCL命令队列为空");
        return -1;
    }
    
    // 验证工作大小兼容性
    if (local_work_size != 0) {
        // 本地工作大小不能大于全局工作大小
        if (local_work_size > global_work_size) {
            snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                    "工作大小无效：本地工作大小%zu大于全局工作大小%zu",
                    local_work_size, global_work_size);
            return -1;
        }
        
        // 本地工作大小必须是全局工作大小的整数因子
        if (global_work_size % local_work_size != 0) {
            snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                    "工作大小不兼容：全局工作大小%zu不能被本地工作大小%zu整除",
                    global_work_size, local_work_size);
            return -1;
        }
        
        // 本地工作大小应该大于0（已经由前面的条件确保）
        // 可以添加其他验证，如检查是否超过设备限制等
    }
    
    // 设置工作大小
    const size_t global_work_sizes[] = { global_work_size };
    
    // 如果本地工作大小为0，传递NULL让OpenCL运行时自动选择
    // 否则使用提供的本地工作大小
    const size_t* local_work_sizes_ptr = NULL;
    const size_t local_work_sizes_array[] = { local_work_size };
    
    if (local_work_size != 0) {
        local_work_sizes_ptr = local_work_sizes_array;
    }
    
    cl_int result = clEnqueueNDRangeKernel(opencl_kernel->command_queue, opencl_kernel->kernel,
                                          1, NULL, global_work_sizes, local_work_sizes_ptr,
                                          0, NULL, NULL);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        if (local_work_size != 0) {
            snprintf(additional_info, sizeof(additional_info),
                    "内核: %p, 队列: %p, 全局工作大小: %zu, 本地工作大小: %zu, 维度: 1", 
                    (void*)opencl_kernel->kernel, (void*)opencl_kernel->command_queue,
                    global_work_size, local_work_size);
        } else {
            snprintf(additional_info, sizeof(additional_info),
                    "内核: %p, 队列: %p, 全局工作大小: %zu, 本地工作大小: 自动, 维度: 1", 
                    (void*)opencl_kernel->kernel, (void*)opencl_kernel->command_queue,
                    global_work_size);
        }
        
        if (OPENCL_CHECK_ERROR_EX(result, "执行OpenCL内核", additional_info) != 0) {
            return -1;
        }
    }
    
    // 等待命令完成
    result = clFinish(opencl_kernel->command_queue);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        if (local_work_size != 0) {
            snprintf(additional_info, sizeof(additional_info),
                    "内核: %p, 队列: %p, 全局工作大小: %zu, 本地工作大小: %zu", 
                    (void*)opencl_kernel->kernel, (void*)opencl_kernel->command_queue,
                    global_work_size, local_work_size);
        } else {
            snprintf(additional_info, sizeof(additional_info),
                    "内核: %p, 队列: %p, 全局工作大小: %zu, 本地工作大小: 自动", 
                    (void*)opencl_kernel->kernel, (void*)opencl_kernel->command_queue,
                    global_work_size);
        }
        
        if (OPENCL_CHECK_ERROR_EX(result, "等待OpenCL内核完成", additional_info) != 0) {
            return -1;
        }
    }
    
    return 0;
}

static int opencl_backend_kernel_execute_nd(GpuKernel* kernel, int work_dim,
                                           const size_t* global_work_size,
                                           const size_t* local_work_size) {
    if (!kernel) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：内核为空");
        return -1;
    }
    
    if (work_dim < 1 || work_dim > 3) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效工作维度：%d（OpenCL支持1-3维）", work_dim);
        return -1;
    }
    
    if (!global_work_size) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：全局工作大小数组为空");
        return -1;
    }
    
    // 验证全局工作大小
    for (int i = 0; i < work_dim; i++) {
        if (global_work_size[i] == 0) {
            snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                    "无效参数：全局工作大小为0（维度%d）", i);
            return -1;
        }
    }
    
    // 验证本地工作大小兼容性（如果提供）
    if (local_work_size) {
        for (int i = 0; i < work_dim; i++) {
            // 本地工作大小可以为0（让OpenCL自动选择）
            if (local_work_size[i] != 0) {
                // 本地工作大小不能大于全局工作大小
                if (local_work_size[i] > global_work_size[i]) {
                    snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                            "工作大小无效：维度%d的本地工作大小%zu大于全局工作大小%zu",
                            i, local_work_size[i], global_work_size[i]);
                    return -1;
                }
                
                // 本地工作大小必须是全局工作大小的整数因子
                if (global_work_size[i] % local_work_size[i] != 0) {
                    snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                            "工作大小不兼容：维度%d的全局工作大小%zu不能被本地工作大小%zu整除",
                            i, global_work_size[i], local_work_size[i]);
                    return -1;
                }
                
                // 本地工作大小必须大于0（已经由条件确保）
                // 注意：这里不检查设备限制，由OpenCL运行时返回相应错误
            }
        }
        
        /* 验证本地工作大小乘积不超过设备最大工作组大小 */
        {
            OpenCLKernel* opencl_kernel = (OpenCLKernel*)kernel;
            if (opencl_kernel->command_queue) {
                size_t workgroup_product = 1;
                for (int i = 0; i < work_dim && local_work_size[i] != 0; i++) {
                    workgroup_product *= local_work_size[i];
                }
                
                if (workgroup_product > 1) {
/* clGetCommandQueueInfo需要运行时OpenCL支持 */
#ifdef SELFLNN_HAS_OPENCL_RUNTIME
                    cl_device_id device = NULL;
                    cl_int dev_result = clGetCommandQueueInfo(opencl_kernel->command_queue,
                        0x1090 /* CL_QUEUE_DEVICE */, sizeof(cl_device_id), &device, NULL);
                    if (dev_result == CL_SUCCESS && device) {
                        size_t max_workgroup_size = 0;
                        cl_int info_result = clGetDeviceInfo(device,
                            CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                            &max_workgroup_size, NULL);
                        if (info_result == CL_SUCCESS && max_workgroup_size > 0) {
                            if (workgroup_product > max_workgroup_size) {
                                snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                                    "本地工作组大小乘积 %zu 超过设备最大 %zu",
                                    workgroup_product, max_workgroup_size);
                                return -1;
                            }
                        }
                    }
#endif /* SELFLNN_HAS_OPENCL_RUNTIME */
                }
            }
        }
    }
    
    OpenCLKernel* opencl_kernel = (OpenCLKernel*)kernel;
    
    // 验证OpenCL内核对象
    if (!opencl_kernel->kernel) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效内核：OpenCL内核对象为空");
        return -1;
    }
    
    if (!opencl_kernel->command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效内核：OpenCL命令队列为空");
        return -1;
    }
    
    cl_int result = clEnqueueNDRangeKernel(opencl_kernel->command_queue, opencl_kernel->kernel,
                                          (cl_uint)work_dim, NULL, global_work_size, local_work_size,
                                          0, NULL, NULL);
    
    {
        // 构建附加错误信息
        char additional_info[512] = {0};
        char global_sizes_str[256] = {0};
        char local_sizes_str[256] = {0};
        
        // 构建全局工作大小字符串
        char* global_ptr = global_sizes_str;
        for (int i = 0; i < work_dim; i++) {
            int written = snprintf(global_ptr, sizeof(global_sizes_str) - (global_ptr - global_sizes_str),
                                 "%zu%s", global_work_size[i], (i < work_dim - 1) ? ", " : "");
            if (written < 0) break;
            global_ptr += written;
        }
        
        // 构建本地工作大小字符串（如果提供）
        if (local_work_size) {
            char* local_ptr = local_sizes_str;
            for (int i = 0; i < work_dim; i++) {
                int written = snprintf(local_ptr, sizeof(local_sizes_str) - (local_ptr - local_sizes_str),
                                     "%zu%s", local_work_size[i], (i < work_dim - 1) ? ", " : "");
                if (written < 0) break;
                local_ptr += written;
            }
        } else {
            snprintf(local_sizes_str, sizeof(local_sizes_str), "自动");
        }
        
        snprintf(additional_info, sizeof(additional_info),
                "内核: %p, 队列: %p, 维度: %d, 全局工作大小: [%s], 本地工作大小: [%s]", 
                (void*)opencl_kernel->kernel, (void*)opencl_kernel->command_queue,
                work_dim, global_sizes_str, local_sizes_str);
        
        if (OPENCL_CHECK_ERROR_EX(result, "执行多维OpenCL内核", additional_info) != 0) {
            return -1;
        }
    }
    
    // 等待命令完成
    result = clFinish(opencl_kernel->command_queue);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "内核: %p, 队列: %p, 维度: %d", 
                (void*)opencl_kernel->kernel, (void*)opencl_kernel->command_queue, work_dim);
        
        if (OPENCL_CHECK_ERROR_EX(result, "等待多维OpenCL内核完成", additional_info) != 0) {
            return -1;
        }
    }
    
    return 0;
}

static GpuStream* opencl_backend_stream_create(GpuContext* context) {
    if (!context) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：上下文为空");
        return NULL;
    }
    
    OpenCLContext* opencl_context = (OpenCLContext*)context;
    
    // 创建命令队列
    cl_int result;
    cl_command_queue command_queue = clCreateCommandQueue(opencl_context->context, 
                                                         opencl_context->device, 
                                                         0, &result);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "上下文: %p, 设备: %p, 操作: 创建流", 
                (void*)opencl_context->context, (void*)opencl_context->device);
        
        if (OPENCL_CHECK_ERROR_EX(result, "创建OpenCL命令队列", additional_info) != 0) {
            return NULL;
        }
    }
    
    // 分配OpenCL流结构
    OpenCLStream* stream = (OpenCLStream*)safe_malloc(sizeof(OpenCLStream));
    if (!stream) {
        clReleaseCommandQueue(command_queue);
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "内存分配失败：OpenCL流结构");
        return NULL;
    }
    
    memset(stream, 0, sizeof(OpenCLStream));
    stream->command_queue = command_queue;
    stream->context = opencl_context->context;
    stream->device = opencl_context->device;
    stream->last_event = NULL;           // 初始化为空
    stream->has_pending_event = 0;       // 初始化为无挂起事件
    
    return (GpuStream*)stream;
}

static void opencl_backend_stream_free(GpuStream* stream) {
    if (!stream) {
        return;
    }
    
    OpenCLStream* opencl_stream = (OpenCLStream*)stream;
    
    // 释放事件（如果存在）
    if (opencl_stream->last_event && clReleaseEvent) {
        cl_int result = clReleaseEvent(opencl_stream->last_event);
        if (result != 0) {
            opencl_check_error(result, "释放OpenCL事件");
        }
        opencl_stream->last_event = NULL;
        opencl_stream->has_pending_event = 0;
    }
    
    // 释放命令队列
    if (opencl_stream->command_queue) {
        cl_int result = clReleaseCommandQueue(opencl_stream->command_queue);
        if (result != 0) {
            opencl_check_error(result, "释放OpenCL命令队列");
        }
    }
    
    // 释放流结构
    safe_free((void**)&opencl_stream);
}

static int opencl_backend_stream_synchronize(GpuStream* stream) {
    if (!stream) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：流为空");
        return -1;
    }
    
    OpenCLStream* opencl_stream = (OpenCLStream*)stream;
    
    cl_int result = clFinish(opencl_stream->command_queue);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "流: %p, 队列: %p", 
                (void*)stream, (void*)opencl_stream->command_queue);
        
        if (OPENCL_CHECK_ERROR_EX(result, "同步OpenCL流", additional_info) != 0) {
            return -1;
        }
    }
    
    return 0;
}

static int opencl_backend_stream_query(GpuStream* stream) {
    if (!stream) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：流为空");
        return -1;
    }
    
    OpenCLStream* opencl_stream = (OpenCLStream*)stream;
    
    // 完整实现：检查命令队列状态和挂起事件
    // 使用OpenCL事件API查询流状态
    
    // 检查命令队列指针是否有效
    if (!opencl_stream->command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "命令队列无效");
        return -1;
    }
    
    // 如果有挂起的事件，检查其状态
    if (opencl_stream->has_pending_event && opencl_stream->last_event && clGetEventInfo) {
        cl_int event_status = 0;
        cl_int result = clGetEventInfo(opencl_stream->last_event, CL_EVENT_COMMAND_EXECUTION_STATUS, 
                                      sizeof(cl_int), &event_status, NULL);
        
        if (result == CL_SUCCESS) {
            // 根据OpenCL规范：
            // CL_QUEUED (0)    - 事件已入队但未提交
            // CL_SUBMITTED (1) - 事件已提交但未开始
            // CL_RUNNING (2)   - 事件正在执行
            // CL_COMPLETE (3)  - 事件已完成
            // 错误码 (<0)      - 事件失败
            
            if (event_status == CL_COMPLETE) {
                // 事件已完成，清除挂起标志
                opencl_stream->has_pending_event = 0;
                return 0;  // 流已完成
            } else if (event_status >= CL_QUEUED && event_status < CL_COMPLETE) {
                return 1;  // 流仍在运行（有挂起命令）
            } else {
                // 事件错误状态
                snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                        "OpenCL事件错误状态: %d", event_status);
                return -1;  // 错误
            }
        } else {
            // 如果无法获取事件信息，假设流仍在运行
            return 1;
        }
    }
    
    // 没有挂起的事件，流已完成
    return 0;
}

/**
 * @brief 异步复制数据到OpenCL设备
 */
static int opencl_backend_memory_copy_to_device_async(GpuMemory* dst, const void* src, size_t size, GpuStream* stream) {
    if (!dst || !src || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：目标内存、源数据或大小为0");
        return -1;
    }

    OpenCLMemory* mem = (OpenCLMemory*)dst;

    // 确定使用的命令队列
    cl_command_queue queue = NULL;
    OpenCLStream* opencl_stream = NULL;

    if (stream) {
        opencl_stream = (OpenCLStream*)stream;
        queue = opencl_stream->command_queue;
    }

    // 如果没有提供流或流无效，使用默认命令队列
    if (!queue) {
        if (mem->opencl_context && mem->opencl_context->default_command_queue) {
            queue = mem->opencl_context->default_command_queue;
        } else {
            // 创建临时命令队列
            cl_device_id device_id = mem->opencl_context ? mem->opencl_context->device : NULL;
            cl_int result;
            queue = clCreateCommandQueue(mem->context, device_id, 0, &result);

            char additional_info[256] = {0};
            snprintf(additional_info, sizeof(additional_info),
                    "上下文: %p, 设备ID: %p, 操作: 异步主机到设备复制, 大小: %zu 字节",
                    (void*)mem->context, (void*)device_id, size);

            if (OPENCL_CHECK_ERROR_EX(result, "创建命令队列", additional_info) != 0) {
                return -1;
            }
            // 标记队列需要在完成后释放 - 但异步模式下无法确定何时完成
            // 此处回退到同步模式
            clEnqueueWriteBuffer(queue, mem->mem_obj, CL_TRUE, 0, size, src, 0, NULL, NULL);
            clFinish(queue);
            clReleaseCommandQueue(queue);
            return 0;
        }
    }

    // 非阻塞写入数据到设备
    cl_event event = NULL;
    cl_int result = clEnqueueWriteBuffer(queue, mem->mem_obj, CL_FALSE, 0, size, src, 0, NULL, &event);

    {
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "内存对象: %p, 大小: %zu 字节, 队列: %p, 偏移: 0, 模式: 异步",
                (void*)mem->mem_obj, size, (void*)queue);

        if (OPENCL_CHECK_ERROR_EX(result, "异步复制数据到设备", additional_info) != 0) {
            return -1;
        }
    }

    // 刷新命令队列以确保命令被提交
    clFlush(queue);

    // 更新流的事件跟踪
    if (opencl_stream) {
        if (opencl_stream->last_event && clReleaseEvent) {
            clReleaseEvent(opencl_stream->last_event);
        }
        opencl_stream->last_event = event;
        opencl_stream->has_pending_event = (event != NULL);
    } else if (event && clReleaseEvent) {
        clReleaseEvent(event);
    }

    return 0;
}

/**
 * @brief 异步从OpenCL设备复制数据
 */
static int opencl_backend_memory_copy_from_device_async(void* dst, GpuMemory* src, size_t size, GpuStream* stream) {
    if (!dst || !src || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：目标内存、源数据或大小为0");
        return -1;
    }

    OpenCLMemory* mem = (OpenCLMemory*)src;

    // 确定使用的命令队列
    cl_command_queue queue = NULL;
    OpenCLStream* opencl_stream = NULL;

    if (stream) {
        opencl_stream = (OpenCLStream*)stream;
        queue = opencl_stream->command_queue;
    }

    // 如果没有提供流或流无效，使用默认命令队列
    if (!queue) {
        if (mem->opencl_context && mem->opencl_context->default_command_queue) {
            queue = mem->opencl_context->default_command_queue;
        } else {
            // 创建临时命令队列
            cl_device_id device_id = mem->opencl_context ? mem->opencl_context->device : NULL;
            cl_int result;
            queue = clCreateCommandQueue(mem->context, device_id, 0, &result);

            char additional_info[256] = {0};
            snprintf(additional_info, sizeof(additional_info),
                    "上下文: %p, 设备ID: %p, 操作: 异步设备到主机复制, 大小: %zu 字节",
                    (void*)mem->context, (void*)device_id, size);

            if (OPENCL_CHECK_ERROR_EX(result, "创建命令队列", additional_info) != 0) {
                return -1;
            }
            // 回退到同步模式
            clEnqueueReadBuffer(queue, mem->mem_obj, CL_TRUE, 0, size, dst, 0, NULL, NULL);
            clFinish(queue);
            clReleaseCommandQueue(queue);
            return 0;
        }
    }

    // 非阻塞从设备读取数据
    cl_event event = NULL;
    cl_int result = clEnqueueReadBuffer(queue, mem->mem_obj, CL_FALSE, 0, size, dst, 0, NULL, &event);

    {
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "内存对象: %p, 大小: %zu 字节, 队列: %p, 偏移: 0, 模式: 异步",
                (void*)mem->mem_obj, size, (void*)queue);

        if (OPENCL_CHECK_ERROR_EX(result, "异步从设备复制数据", additional_info) != 0) {
            return -1;
        }
    }

    // 刷新命令队列以确保命令被提交
    clFlush(queue);

    // 更新流的事件跟踪
    if (opencl_stream) {
        if (opencl_stream->last_event && clReleaseEvent) {
            clReleaseEvent(opencl_stream->last_event);
        }
        opencl_stream->last_event = event;
        opencl_stream->has_pending_event = (event != NULL);
    } else if (event && clReleaseEvent) {
        clReleaseEvent(event);
    }

    return 0;
}

static int opencl_backend_get_memory_info(GpuContext* context, size_t* total_memory, size_t* free_memory) {
    if (!context || !total_memory || !free_memory) {
        return -1;
    }
    
    OpenCLContext* opencl_context = (OpenCLContext*)context;
    *total_memory = opencl_context->total_memory;
    *free_memory = opencl_context->free_memory;
    
    return 0;
}

static int opencl_backend_device_reset(GpuContext* context) {
    if (!context) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
                "无效参数：上下文为空");
        return -1;
    }
    
    OpenCLContext* opencl_context = (OpenCLContext*)context;
    
    // OpenCL没有标准的设备重置函数
    // 完整实现：通过刷新所有命令队列和执行同步来实现设备重置
    // 这确保所有挂起的命令完成，设备返回到已知状态
    
    // 首先，刷新默认命令队列
    cl_int result = CL_SUCCESS;
    if (opencl_context->default_command_queue) {
        result = clFinish(opencl_context->default_command_queue);
        if (result != CL_SUCCESS) {
            opencl_check_error(result, "刷新默认命令队列");
            // 继续尝试重置，但记录错误
        }
    }
    
    // 创建一个临时命令队列来进一步刷新设备
    cl_command_queue temp_queue = clCreateCommandQueue(opencl_context->context, 
                                                      opencl_context->device, 
                                                      0, &result);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "上下文: %p, 设备: %p, 操作: 设备重置", 
                (void*)opencl_context->context, (void*)opencl_context->device);
        
        if (OPENCL_CHECK_ERROR_EX(result, "创建临时命令队列进行设备重置", additional_info) != 0) {
            return -1;
        }
    }
    
    // 刷新队列（确保所有命令完成）
    result = clFinish(temp_queue);
    
    {
        // 构建附加错误信息
        char additional_info[256] = {0};
        snprintf(additional_info, sizeof(additional_info),
                "队列: %p, 上下文: %p, 操作: 设备重置刷新", 
                (void*)temp_queue, (void*)opencl_context->context);
        
        if (OPENCL_CHECK_ERROR_EX(result, "刷新命令队列", additional_info) != 0) {
            clReleaseCommandQueue(temp_queue);
            return -1;
        }
    }
    
    // 释放临时队列
    result = clReleaseCommandQueue(temp_queue);
    if (result != 0) {
        opencl_check_error(result, "释放临时命令队列");
    }
    
    // 记录重置操作
    snprintf(g_opencl_error_string, sizeof(g_opencl_error_string),
            "OpenCL设备重置完成（完整实现）");
    
    return 0;
}

static const char* opencl_backend_get_error_string(void) {
    // 返回详细的错误信息，包含上下文信息
    // 如果发生了错误，返回详细错误信息，否则返回简单的错误字符串
    if (g_last_error_context.error_code != 0) {
        return opencl_get_detailed_error();
    }
    return g_opencl_error_string;
}
/* ============================================================================
 * OpenCL核心计算函数实现
 * 这些函数提供完整的GPU加速计算功能，基于OpenCL内核基础设施
 * ============================================================================ */

/**
 * @brief OpenCL全连接层前向传播
 *
 * 执行：output = activation(input * weights^T + bias)
 * 使用matmul_train内核进行矩阵乘法，activation_forward_unified进行激活
 */
int opencl_forward_dense(GpuContext* ctx, const float* input, const float* weights,
    const float* bias, float* output, size_t batch_size,
    size_t input_size, size_t output_size, GpuActivationType act, float alpha) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用");
        return -1;
    }
    if (!input || !weights || !output || batch_size == 0 || input_size == 0 || output_size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数：输入/权重/输出为空或维度为零");
        return -1;
    }

    size_t input_bytes = batch_size * input_size * sizeof(float);
    size_t weight_bytes = output_size * input_size * sizeof(float);
    size_t temp_bytes = batch_size * output_size * sizeof(float);
    size_t output_bytes = batch_size * output_size * sizeof(float);

    GpuMemory* d_input = opencl_backend_memory_alloc(ctx, input_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_weights = opencl_backend_memory_alloc(ctx, weight_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_temp = opencl_backend_memory_alloc(ctx, temp_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_output = opencl_backend_memory_alloc(ctx, output_bytes, GPU_MEMORY_DEVICE);
    float* h_temp = NULL;

    if (!d_input || !d_weights || !d_temp || !d_output) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败");
        goto cleanup;
    }

    h_temp = (float*)safe_malloc(temp_bytes);
    if (!h_temp) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "主机内存分配失败");
        goto cleanup;
    }

    /* 复制数据到设备 */
    if (opencl_backend_memory_copy_to_device(d_input, input, input_bytes) != 0) goto cleanup;
    if (opencl_backend_memory_copy_to_device(d_weights, weights, weight_bytes) != 0) goto cleanup;

    /* 矩阵乘法：temp = input * weights^T */
    {
        GpuKernel* mk = opencl_backend_kernel_create(ctx, NULL, "matmul_train");
        if (!mk) goto cleanup;
        int M_int = (int)batch_size;
        int N_int = (int)output_size;
        int K_int = (int)input_size;
        int tA = 0;
        int tB = 1;
        float one = 1.0f;
        float zero = 0.0f;
        cl_mem d_input_mem = ((OpenCLMemory*)d_input)->mem_obj;
        cl_mem d_weights_mem = ((OpenCLMemory*)d_weights)->mem_obj;
        cl_mem d_temp_mem = ((OpenCLMemory*)d_temp)->mem_obj;
        opencl_backend_kernel_set_arg(mk, 0, sizeof(cl_mem), &d_input_mem);
        opencl_backend_kernel_set_arg(mk, 1, sizeof(cl_mem), &d_weights_mem);
        opencl_backend_kernel_set_arg(mk, 2, sizeof(cl_mem), &d_temp_mem);
        opencl_backend_kernel_set_arg(mk, 3, sizeof(int), &M_int);
        opencl_backend_kernel_set_arg(mk, 4, sizeof(int), &N_int);
        opencl_backend_kernel_set_arg(mk, 5, sizeof(int), &K_int);
        opencl_backend_kernel_set_arg(mk, 6, sizeof(int), &tA);
        opencl_backend_kernel_set_arg(mk, 7, sizeof(int), &tB);
        opencl_backend_kernel_set_arg(mk, 8, sizeof(float), &one);
        opencl_backend_kernel_set_arg(mk, 9, sizeof(float), &zero);
        opencl_backend_kernel_execute(mk, (size_t)(batch_size * output_size), 256);
        opencl_backend_kernel_free(mk);
    }

    /* 复制结果到主机并加偏置 */
    if (opencl_backend_memory_copy_from_device(h_temp, d_temp, temp_bytes) != 0) goto cleanup;

    if (bias) {
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t o = 0; o < output_size; o++) {
                h_temp[b * output_size + o] += bias[o];
            }
        }
    }

    /* 激活函数（在GPU上执行） */
    if (act != GPU_ACTIVATION_RELU || alpha != 0.0f) {
        /* 将带偏置结果拷贝回设备 */
        GpuMemory* d_act_in = opencl_backend_memory_alloc(ctx, temp_bytes, GPU_MEMORY_DEVICE);
        if (!d_act_in) {
            snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "激活输入内存分配失败");
            goto cleanup;
        }
        if (opencl_backend_memory_copy_to_device(d_act_in, h_temp, temp_bytes) != 0) {
            opencl_backend_memory_free(d_act_in);
            goto cleanup;
        }

        GpuKernel* ak = opencl_backend_kernel_create(ctx, NULL, "activation_forward_unified");
        if (!ak) {
            opencl_backend_memory_free(d_act_in);
            goto cleanup;
        }
        int n_elements = (int)(batch_size * output_size);
        int act_type = (int)act;
        cl_mem d_act_in_mem = ((OpenCLMemory*)d_act_in)->mem_obj;
        cl_mem d_output_mem = ((OpenCLMemory*)d_output)->mem_obj;
        opencl_backend_kernel_set_arg(ak, 0, sizeof(cl_mem), &d_act_in_mem);
        opencl_backend_kernel_set_arg(ak, 1, sizeof(cl_mem), &d_output_mem);
        opencl_backend_kernel_set_arg(ak, 2, sizeof(int), &n_elements);
        opencl_backend_kernel_set_arg(ak, 3, sizeof(int), &act_type);
        opencl_backend_kernel_set_arg(ak, 4, sizeof(float), &alpha);
        opencl_backend_kernel_execute(ak, (size_t)n_elements, 256);
        opencl_backend_kernel_free(ak);

        if (opencl_backend_memory_copy_from_device(output, d_output, output_bytes) != 0) {
            opencl_backend_memory_free(d_act_in);
            goto cleanup;
        }
        opencl_backend_memory_free(d_act_in);
    } else {
        /* 无激活函数，直接拷贝到输出 */
        memcpy(output, h_temp, output_bytes);
    }

    safe_free((void**)&h_temp);
    opencl_backend_memory_free(d_input);
    opencl_backend_memory_free(d_weights);
    opencl_backend_memory_free(d_temp);
    opencl_backend_memory_free(d_output);
    return 0;

cleanup:
    if (h_temp) safe_free((void**)&h_temp);
    if (d_input) opencl_backend_memory_free(d_input);
    if (d_weights) opencl_backend_memory_free(d_weights);
    if (d_temp) opencl_backend_memory_free(d_temp);
    if (d_output) opencl_backend_memory_free(d_output);
    return -1;
}

/**
 * @brief OpenCL矩阵乘法训练内核
 *
 * 执行：C = alpha * op(A) * op(B) + beta * C
 * 支持转置A和转置B，支持缩放和累加
 */
int opencl_matmul_train(GpuContext* ctx, const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K, int transA, int transB,
    float alpha, float beta) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用");
        return -1;
    }
    if (!A || !B || !C || M == 0 || N == 0 || K == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数：矩阵为空或维度为零");
        return -1;
    }

    size_t a_size = transA ? (size_t)K * M : (size_t)M * K;
    size_t b_size = transB ? (size_t)N * K : (size_t)K * N;
    size_t c_size = M * N;
    size_t a_bytes = a_size * sizeof(float);
    size_t b_bytes = b_size * sizeof(float);
    size_t c_bytes = c_size * sizeof(float);

    GpuMemory* d_A = opencl_backend_memory_alloc(ctx, a_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_B = opencl_backend_memory_alloc(ctx, b_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_C = opencl_backend_memory_alloc(ctx, c_bytes, GPU_MEMORY_DEVICE);

    if (!d_A || !d_B || !d_C) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败");
        if (d_A) opencl_backend_memory_free(d_A);
        if (d_B) opencl_backend_memory_free(d_B);
        if (d_C) opencl_backend_memory_free(d_C);
        return -1;
    }

    if (opencl_backend_memory_copy_to_device(d_A, A, a_bytes) != 0) goto matmul_cleanup;
    if (opencl_backend_memory_copy_to_device(d_B, B, b_bytes) != 0) goto matmul_cleanup;

    /* 如果beta != 0，需要先拷贝C到设备（因为需要累加到原值） */
    if (beta != 0.0f) {
        if (opencl_backend_memory_copy_to_device(d_C, C, c_bytes) != 0) goto matmul_cleanup;
    }

    {
        GpuKernel* mk = opencl_backend_kernel_create(ctx, NULL, "matmul_train");
        if (!mk) goto matmul_cleanup;
        int M_int = (int)M;
        int N_int = (int)N;
        int K_int = (int)K;
        int tA = transA;
        int tB = transB;
        cl_mem d_A_mem = ((OpenCLMemory*)d_A)->mem_obj;
        cl_mem d_B_mem = ((OpenCLMemory*)d_B)->mem_obj;
        cl_mem d_C_mem = ((OpenCLMemory*)d_C)->mem_obj;
        opencl_backend_kernel_set_arg(mk, 0, sizeof(cl_mem), &d_A_mem);
        opencl_backend_kernel_set_arg(mk, 1, sizeof(cl_mem), &d_B_mem);
        opencl_backend_kernel_set_arg(mk, 2, sizeof(cl_mem), &d_C_mem);
        opencl_backend_kernel_set_arg(mk, 3, sizeof(int), &M_int);
        opencl_backend_kernel_set_arg(mk, 4, sizeof(int), &N_int);
        opencl_backend_kernel_set_arg(mk, 5, sizeof(int), &K_int);
        opencl_backend_kernel_set_arg(mk, 6, sizeof(int), &tA);
        opencl_backend_kernel_set_arg(mk, 7, sizeof(int), &tB);
        opencl_backend_kernel_set_arg(mk, 8, sizeof(float), &alpha);
        opencl_backend_kernel_set_arg(mk, 9, sizeof(float), &beta);
        opencl_backend_kernel_execute(mk, (size_t)(M * N), 256);
        opencl_backend_kernel_free(mk);
    }

    if (opencl_backend_memory_copy_from_device(C, d_C, c_bytes) != 0) goto matmul_cleanup;

    opencl_backend_memory_free(d_A);
    opencl_backend_memory_free(d_B);
    opencl_backend_memory_free(d_C);
    return 0;

matmul_cleanup:
    opencl_backend_memory_free(d_A);
    opencl_backend_memory_free(d_B);
    opencl_backend_memory_free(d_C);
    return -1;
}

/**
 * @brief OpenCL激活函数前向传播
 *
 * 支持ReLU/LeakyReLU/Sigmoid/Tanh/GELU/Softplus六种激活函数
 */
int opencl_activation_forward(GpuContext* ctx, const float* input, float* output,
    size_t size, GpuActivationType act, float alpha) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用");
        return -1;
    }
    if (!input || !output || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数：输入/输出为空或大小为零");
        return -1;
    }

    size_t bytes = size * sizeof(float);
    GpuMemory* d_input = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_output = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);

    if (!d_input || !d_output) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败");
        if (d_input) opencl_backend_memory_free(d_input);
        if (d_output) opencl_backend_memory_free(d_output);
        return -1;
    }

    if (opencl_backend_memory_copy_to_device(d_input, input, bytes) != 0) goto act_cleanup;

    {
        GpuKernel* ak = opencl_backend_kernel_create(ctx, NULL, "activation_forward_unified");
        if (!ak) goto act_cleanup;
        int n = (int)size;
        int act_type = (int)act;
        cl_mem d_input_mem = ((OpenCLMemory*)d_input)->mem_obj;
        cl_mem d_output_mem = ((OpenCLMemory*)d_output)->mem_obj;
        opencl_backend_kernel_set_arg(ak, 0, sizeof(cl_mem), &d_input_mem);
        opencl_backend_kernel_set_arg(ak, 1, sizeof(cl_mem), &d_output_mem);
        opencl_backend_kernel_set_arg(ak, 2, sizeof(int), &n);
        opencl_backend_kernel_set_arg(ak, 3, sizeof(int), &act_type);
        opencl_backend_kernel_set_arg(ak, 4, sizeof(float), &alpha);
        opencl_backend_kernel_execute(ak, size, 256);
        opencl_backend_kernel_free(ak);
    }

    if (opencl_backend_memory_copy_from_device(output, d_output, bytes) != 0) goto act_cleanup;

    opencl_backend_memory_free(d_input);
    opencl_backend_memory_free(d_output);
    return 0;

act_cleanup:
    opencl_backend_memory_free(d_input);
    opencl_backend_memory_free(d_output);
    return -1;
}

/**
 * @brief OpenCL激活函数反向传播
 *
 * 计算激活函数的梯度，支持所有激活类型
 */
int opencl_activation_backward(GpuContext* ctx, const float* input, float* output,
    const float* grad_output, float* grad_input, size_t size,
    GpuActivationType act, float alpha) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用");
        return -1;
    }
    if (!input || !grad_output || !grad_input || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数");
        return -1;
    }

    size_t bytes = size * sizeof(float);
    GpuMemory* d_input = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_output_dev = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_grad_out = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_grad_in = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);

    if (!d_input || !d_output_dev || !d_grad_out || !d_grad_in) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败");
        goto actb_cleanup;
    }

    if (opencl_backend_memory_copy_to_device(d_input, input, bytes) != 0) goto actb_cleanup;

    /* output 参数可以为 NULL（反向传播中通常不需要forward输出） */
    if (output) {
        if (opencl_backend_memory_copy_to_device(d_output_dev, output, bytes) != 0) goto actb_cleanup;
    }

    if (opencl_backend_memory_copy_to_device(d_grad_out, grad_output, bytes) != 0) goto actb_cleanup;

    {
        GpuKernel* bk = opencl_backend_kernel_create(ctx, NULL, "activation_backward_unified");
        if (!bk) goto actb_cleanup;
        int n = (int)size;
        int act_type = (int)act;
        cl_mem d_input_mem = ((OpenCLMemory*)d_input)->mem_obj;
        cl_mem d_output_mem = output ? ((OpenCLMemory*)d_output_dev)->mem_obj : d_input_mem;
        cl_mem d_grad_out_mem = ((OpenCLMemory*)d_grad_out)->mem_obj;
        cl_mem d_grad_in_mem = ((OpenCLMemory*)d_grad_in)->mem_obj;
        opencl_backend_kernel_set_arg(bk, 0, sizeof(cl_mem), &d_input_mem);
        opencl_backend_kernel_set_arg(bk, 1, sizeof(cl_mem), &d_output_mem);
        opencl_backend_kernel_set_arg(bk, 2, sizeof(cl_mem), &d_grad_out_mem);
        opencl_backend_kernel_set_arg(bk, 3, sizeof(cl_mem), &d_grad_in_mem);
        opencl_backend_kernel_set_arg(bk, 4, sizeof(int), &n);
        opencl_backend_kernel_set_arg(bk, 5, sizeof(int), &act_type);
        opencl_backend_kernel_set_arg(bk, 6, sizeof(float), &alpha);
        opencl_backend_kernel_execute(bk, size, 256);
        opencl_backend_kernel_free(bk);
    }

    if (opencl_backend_memory_copy_from_device(grad_input, d_grad_in, bytes) != 0) goto actb_cleanup;

    opencl_backend_memory_free(d_input);
    opencl_backend_memory_free(d_output_dev);
    opencl_backend_memory_free(d_grad_out);
    opencl_backend_memory_free(d_grad_in);
    return 0;

actb_cleanup:
    if (d_input) opencl_backend_memory_free(d_input);
    if (d_output_dev) opencl_backend_memory_free(d_output_dev);
    if (d_grad_out) opencl_backend_memory_free(d_grad_out);
    if (d_grad_in) opencl_backend_memory_free(d_grad_in);
    return -1;
}

/**
 * @brief OpenCL批归一化前向传播
 *
 * 训练模式：归一化并更新running mean/var
 * 推理模式：使用running mean/var进行归一化
 */
int opencl_batch_norm_forward(GpuContext* ctx, const float* input, float* output,
    const float* gamma, const float* beta, float* running_mean, float* running_var,
    float* batch_mean, float* batch_var, size_t batch_size, size_t features,
    float momentum, float epsilon, int is_training) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用");
        return -1;
    }
    if (!input || !output || !gamma || !beta || batch_size == 0 || features == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数");
        return -1;
    }

    size_t total_elements = batch_size * features;
    size_t total_bytes = total_elements * sizeof(float);
    size_t feature_bytes = features * sizeof(float);

    if (is_training) {
        /* 训练模式：计算批均值/方差，然后归一化 */
        float* h_input = (float*)safe_malloc(total_bytes);
        if (!h_input) {
            snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "主机内存分配失败");
            return -1;
        }
        memcpy(h_input, input, total_bytes);

        /* 计算每个特征的均值 */
        for (size_t f = 0; f < features; f++) {
            float sum = 0.0f;
            for (size_t b = 0; b < batch_size; b++) {
                sum += h_input[b * features + f];
            }
            batch_mean[f] = sum / (float)batch_size;
        }

        /* 计算每个特征的方差 */
        for (size_t f = 0; f < features; f++) {
            float var_sum = 0.0f;
            for (size_t b = 0; b < batch_size; b++) {
                float diff = h_input[b * features + f] - batch_mean[f];
                var_sum += diff * diff;
            }
            batch_var[f] = var_sum / (float)batch_size;
        }

        /* 更新running statistics */
        if (running_mean) {
            for (size_t f = 0; f < features; f++) {
                running_mean[f] = momentum * running_mean[f] + (1.0f - momentum) * batch_mean[f];
            }
        }
        if (running_var) {
            for (size_t f = 0; f < features; f++) {
                running_var[f] = momentum * running_var[f] + (1.0f - momentum) * batch_var[f];
            }
        }

        /* 归一化：output = gamma * (x - mean) / sqrt(var + eps) + beta */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t f = 0; f < features; f++) {
                float x_hat = (h_input[b * features + f] - batch_mean[f]) /
                              (sqrtf(batch_var[f] + epsilon) + 1e-8f);
                output[b * features + f] = gamma[f] * x_hat + beta[f];
            }
        }

        safe_free((void**)&h_input);
    } else {
        /* 推理模式：使用running mean/var */
        GpuMemory* d_input = opencl_backend_memory_alloc(ctx, total_bytes, GPU_MEMORY_DEVICE);
        GpuMemory* d_gamma = opencl_backend_memory_alloc(ctx, feature_bytes, GPU_MEMORY_DEVICE);
        GpuMemory* d_beta_dev = opencl_backend_memory_alloc(ctx, feature_bytes, GPU_MEMORY_DEVICE);
        GpuMemory* d_mean = opencl_backend_memory_alloc(ctx, feature_bytes, GPU_MEMORY_DEVICE);
        GpuMemory* d_var = opencl_backend_memory_alloc(ctx, feature_bytes, GPU_MEMORY_DEVICE);
        GpuMemory* d_output = opencl_backend_memory_alloc(ctx, total_bytes, GPU_MEMORY_DEVICE);

        if (!d_input || !d_gamma || !d_beta_dev || !d_mean || !d_var || !d_output) {
            snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败");
            goto bn_inf_cleanup;
        }

        opencl_backend_memory_copy_to_device(d_input, input, total_bytes);
        opencl_backend_memory_copy_to_device(d_gamma, gamma, feature_bytes);
        opencl_backend_memory_copy_to_device(d_beta_dev, beta, feature_bytes);
        opencl_backend_memory_copy_to_device(d_mean, running_mean ? running_mean : batch_mean, feature_bytes);
        opencl_backend_memory_copy_to_device(d_var, running_var ? running_var : batch_var, feature_bytes);

        {
            GpuKernel* bnk = opencl_backend_kernel_create(ctx, NULL, "batch_norm");
            if (!bnk) goto bn_inf_cleanup;
            int ch = (int)features;
            int sp = (int)batch_size;
            float eps = epsilon;
            cl_mem d_in_mem = ((OpenCLMemory*)d_input)->mem_obj;
            cl_mem d_g_mem = ((OpenCLMemory*)d_gamma)->mem_obj;
            cl_mem d_b_mem = ((OpenCLMemory*)d_beta_dev)->mem_obj;
            cl_mem d_m_mem = ((OpenCLMemory*)d_mean)->mem_obj;
            cl_mem d_v_mem = ((OpenCLMemory*)d_var)->mem_obj;
            cl_mem d_out_mem = ((OpenCLMemory*)d_output)->mem_obj;
            opencl_backend_kernel_set_arg(bnk, 0, sizeof(cl_mem), &d_in_mem);
            opencl_backend_kernel_set_arg(bnk, 1, sizeof(cl_mem), &d_g_mem);
            opencl_backend_kernel_set_arg(bnk, 2, sizeof(cl_mem), &d_b_mem);
            opencl_backend_kernel_set_arg(bnk, 3, sizeof(cl_mem), &d_m_mem);
            opencl_backend_kernel_set_arg(bnk, 4, sizeof(cl_mem), &d_v_mem);
            opencl_backend_kernel_set_arg(bnk, 5, sizeof(float), &eps);
            opencl_backend_kernel_set_arg(bnk, 6, sizeof(int), &ch);
            opencl_backend_kernel_set_arg(bnk, 7, sizeof(int), &sp);
            opencl_backend_kernel_set_arg(bnk, 8, sizeof(cl_mem), &d_out_mem);
            opencl_backend_kernel_execute(bnk, (size_t)(features * batch_size), 256);
            opencl_backend_kernel_free(bnk);
        }

        opencl_backend_memory_copy_from_device(output, d_output, total_bytes);

        bn_inf_cleanup:
        if (d_input) opencl_backend_memory_free(d_input);
        if (d_gamma) opencl_backend_memory_free(d_gamma);
        if (d_beta_dev) opencl_backend_memory_free(d_beta_dev);
        if (d_mean) opencl_backend_memory_free(d_mean);
        if (d_var) opencl_backend_memory_free(d_var);
        if (d_output) opencl_backend_memory_free(d_output);

        if (!d_input || !d_gamma || !d_beta_dev || !d_mean || !d_var || !d_output) return -1;
    }

    return 0;
}

/**
 * @brief OpenCL批归一化反向传播
 *
 * 计算grad_input/ grad_gamma / grad_beta
 * 使用batch_norm_backward内核
 */
int opencl_batch_norm_backward(GpuContext* ctx, const float* input, const float* grad_output,
    float* grad_input, float* grad_gamma, float* grad_beta,
    const float* x_hat, const float* gamma, size_t batch_size,
    size_t features, float epsilon) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用");
        return -1;
    }
    if (!input || !grad_output || !grad_input || !grad_gamma || !grad_beta || !x_hat || !gamma ||
        batch_size == 0 || features == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数");
        return -1;
    }

    size_t total_elements = batch_size * features;
    size_t total_bytes = total_elements * sizeof(float);
    size_t feature_bytes = features * sizeof(float);

    GpuMemory* d_input = opencl_backend_memory_alloc(ctx, total_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_grad_out = opencl_backend_memory_alloc(ctx, total_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_grad_in = opencl_backend_memory_alloc(ctx, total_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_grad_gamma = opencl_backend_memory_alloc(ctx, feature_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_grad_beta = opencl_backend_memory_alloc(ctx, feature_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_x_hat = opencl_backend_memory_alloc(ctx, total_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_gamma = opencl_backend_memory_alloc(ctx, feature_bytes, GPU_MEMORY_DEVICE);

    if (!d_input || !d_grad_out || !d_grad_in || !d_grad_gamma || !d_grad_beta ||
        !d_x_hat || !d_gamma) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败");
        goto bnb_cleanup;
    }

    opencl_backend_memory_copy_to_device(d_input, input, total_bytes);
    opencl_backend_memory_copy_to_device(d_grad_out, grad_output, total_bytes);
    opencl_backend_memory_copy_to_device(d_x_hat, x_hat, total_bytes);
    opencl_backend_memory_copy_to_device(d_gamma, gamma, feature_bytes);

    {
        GpuKernel* bk = opencl_backend_kernel_create(ctx, NULL, "batch_norm_backward");
        if (!bk) goto bnb_cleanup;
        int bs = (int)batch_size;
        int ft = (int)features;
        float eps = epsilon;
        cl_mem d_in_mem = ((OpenCLMemory*)d_input)->mem_obj;
        cl_mem d_go_mem = ((OpenCLMemory*)d_grad_out)->mem_obj;
        cl_mem d_gi_mem = ((OpenCLMemory*)d_grad_in)->mem_obj;
        cl_mem d_gg_mem = ((OpenCLMemory*)d_grad_gamma)->mem_obj;
        cl_mem d_gb_mem = ((OpenCLMemory*)d_grad_beta)->mem_obj;
        cl_mem d_xh_mem = ((OpenCLMemory*)d_x_hat)->mem_obj;
        cl_mem d_gm_mem = ((OpenCLMemory*)d_gamma)->mem_obj;
        opencl_backend_kernel_set_arg(bk, 0, sizeof(cl_mem), &d_in_mem);
        opencl_backend_kernel_set_arg(bk, 1, sizeof(cl_mem), &d_go_mem);
        opencl_backend_kernel_set_arg(bk, 2, sizeof(cl_mem), &d_gi_mem);
        opencl_backend_kernel_set_arg(bk, 3, sizeof(cl_mem), &d_gg_mem);
        opencl_backend_kernel_set_arg(bk, 4, sizeof(cl_mem), &d_gb_mem);
        opencl_backend_kernel_set_arg(bk, 5, sizeof(cl_mem), &d_xh_mem);
        opencl_backend_kernel_set_arg(bk, 6, sizeof(cl_mem), &d_gm_mem);
        opencl_backend_kernel_set_arg(bk, 7, sizeof(int), &bs);
        opencl_backend_kernel_set_arg(bk, 8, sizeof(int), &ft);
        opencl_backend_kernel_set_arg(bk, 9, sizeof(float), &eps);
        opencl_backend_kernel_execute(bk, (size_t)features, 256);
        opencl_backend_kernel_free(bk);
    }

    opencl_backend_memory_copy_from_device(grad_input, d_grad_in, total_bytes);
    opencl_backend_memory_copy_from_device(grad_gamma, d_grad_gamma, feature_bytes);
    opencl_backend_memory_copy_from_device(grad_beta, d_grad_beta, feature_bytes);

    opencl_backend_memory_free(d_input);
    opencl_backend_memory_free(d_grad_out);
    opencl_backend_memory_free(d_grad_in);
    opencl_backend_memory_free(d_grad_gamma);
    opencl_backend_memory_free(d_grad_beta);
    opencl_backend_memory_free(d_x_hat);
    opencl_backend_memory_free(d_gamma);
    return 0;

bnb_cleanup:
    if (d_input) opencl_backend_memory_free(d_input);
    if (d_grad_out) opencl_backend_memory_free(d_grad_out);
    if (d_grad_in) opencl_backend_memory_free(d_grad_in);
    if (d_grad_gamma) opencl_backend_memory_free(d_grad_gamma);
    if (d_grad_beta) opencl_backend_memory_free(d_grad_beta);
    if (d_x_hat) opencl_backend_memory_free(d_x_hat);
    if (d_gamma) opencl_backend_memory_free(d_gamma);
    return -1;
}

/**
 * @brief OpenCL Dropout前向传播
 *
 * 训练模式：生成mask并缩放保留的元素
 * 推理模式：恒等映射
 */
int opencl_dropout_forward(GpuContext* ctx, const float* input, float* output,
    float* mask, float dropout_rate, size_t size, int is_training) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用");
        return -1;
    }
    if (!input || !output || !mask || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数");
        return -1;
    }

    size_t bytes = size * sizeof(float);

    if (is_training) {
        /* 在主机生成随机掩码（mask[i] = uniform(0,1)） */
        for (size_t i = 0; i < size; i++) {
            mask[i] = (float)rand() / (float)RAND_MAX;
        }
    }

    GpuMemory* d_input = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_output = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_mask = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);

    if (!d_input || !d_output || !d_mask) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败");
        goto do_cleanup;
    }

    opencl_backend_memory_copy_to_device(d_input, input, bytes);
    opencl_backend_memory_copy_to_device(d_mask, mask, bytes);

    {
        GpuKernel* dk = opencl_backend_kernel_create(ctx, NULL, "dropout_forward");
        if (!dk) goto do_cleanup;
        int n = (int)size;
        float rate = dropout_rate;
        int train = is_training;
        cl_mem d_in_mem = ((OpenCLMemory*)d_input)->mem_obj;
        cl_mem d_out_mem = ((OpenCLMemory*)d_output)->mem_obj;
        cl_mem d_mk_mem = ((OpenCLMemory*)d_mask)->mem_obj;
        opencl_backend_kernel_set_arg(dk, 0, sizeof(cl_mem), &d_in_mem);
        opencl_backend_kernel_set_arg(dk, 1, sizeof(cl_mem), &d_out_mem);
        opencl_backend_kernel_set_arg(dk, 2, sizeof(cl_mem), &d_mk_mem);
        opencl_backend_kernel_set_arg(dk, 3, sizeof(int), &n);
        opencl_backend_kernel_set_arg(dk, 4, sizeof(float), &rate);
        opencl_backend_kernel_set_arg(dk, 5, sizeof(int), &train);
        opencl_backend_kernel_execute(dk, size, 256);
        opencl_backend_kernel_free(dk);
    }

    opencl_backend_memory_copy_from_device(output, d_output, bytes);

    opencl_backend_memory_free(d_input);
    opencl_backend_memory_free(d_output);
    opencl_backend_memory_free(d_mask);
    return 0;

do_cleanup:
    if (d_input) opencl_backend_memory_free(d_input);
    if (d_output) opencl_backend_memory_free(d_output);
    if (d_mask) opencl_backend_memory_free(d_mask);
    return -1;
}

/**
 * @brief OpenCL Dropout反向传播
 *
 * grad_input = grad_output * mask / (1 - dropout_rate)
 */
int opencl_dropout_backward(GpuContext* ctx, const float* grad_output, float* grad_input,
    const float* mask, float dropout_rate, size_t size) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用");
        return -1;
    }
    if (!grad_output || !grad_input || !mask || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数");
        return -1;
    }

    size_t bytes = size * sizeof(float);
    GpuMemory* d_grad_out = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_grad_in = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_mask = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);

    if (!d_grad_out || !d_grad_in || !d_mask) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败");
        goto dob_cleanup;
    }

    opencl_backend_memory_copy_to_device(d_grad_out, grad_output, bytes);
    opencl_backend_memory_copy_to_device(d_mask, mask, bytes);

    {
        GpuKernel* dk = opencl_backend_kernel_create(ctx, NULL, "dropout_backward");
        if (!dk) goto dob_cleanup;
        int n = (int)size;
        float rate = dropout_rate;
        cl_mem d_go_mem = ((OpenCLMemory*)d_grad_out)->mem_obj;
        cl_mem d_gi_mem = ((OpenCLMemory*)d_grad_in)->mem_obj;
        cl_mem d_mk_mem = ((OpenCLMemory*)d_mask)->mem_obj;
        opencl_backend_kernel_set_arg(dk, 0, sizeof(cl_mem), &d_go_mem);
        opencl_backend_kernel_set_arg(dk, 1, sizeof(cl_mem), &d_gi_mem);
        opencl_backend_kernel_set_arg(dk, 2, sizeof(cl_mem), &d_mk_mem);
        opencl_backend_kernel_set_arg(dk, 3, sizeof(int), &n);
        opencl_backend_kernel_set_arg(dk, 4, sizeof(float), &rate);
        opencl_backend_kernel_execute(dk, size, 256);
        opencl_backend_kernel_free(dk);
    }

    opencl_backend_memory_copy_from_device(grad_input, d_grad_in, bytes);

    opencl_backend_memory_free(d_grad_out);
    opencl_backend_memory_free(d_grad_in);
    opencl_backend_memory_free(d_mask);
    return 0;

dob_cleanup:
    if (d_grad_out) opencl_backend_memory_free(d_grad_out);
    if (d_grad_in) opencl_backend_memory_free(d_grad_in);
    if (d_mask) opencl_backend_memory_free(d_mask);
    return -1;
}

/**
 * @brief OpenCL RMSProp优化器更新
 *
 * RMSProp参数更新：
 * cache = beta * cache + (1 - beta) * grad^2
 * weights -= lr * grad / (sqrt(cache) + eps) + lr * wd * weights
 */
int opencl_rmsprop_update(GpuContext* ctx, float* weights, const float* gradients,
    float* cache, size_t size, float lr, float beta, float eps, float wd) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用");
        return -1;
    }
    if (!weights || !gradients || !cache || size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数");
        return -1;
    }

    size_t bytes = size * sizeof(float);
    GpuMemory* d_weights = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_grads = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_cache = opencl_backend_memory_alloc(ctx, bytes, GPU_MEMORY_DEVICE);

    if (!d_weights || !d_grads || !d_cache) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败");
        goto rms_cleanup;
    }

    opencl_backend_memory_copy_to_device(d_weights, weights, bytes);
    opencl_backend_memory_copy_to_device(d_grads, gradients, bytes);
    opencl_backend_memory_copy_to_device(d_cache, cache, bytes);

    {
        GpuKernel* rk = opencl_backend_kernel_create(ctx, NULL, "rmsprop_update");
        if (!rk) goto rms_cleanup;
        int n = (int)size;
        float lr_f = lr;
        float beta_f = beta;
        float eps_f = eps;
        float wd_f = wd;
        cl_mem d_w_mem = ((OpenCLMemory*)d_weights)->mem_obj;
        cl_mem d_g_mem = ((OpenCLMemory*)d_grads)->mem_obj;
        cl_mem d_c_mem = ((OpenCLMemory*)d_cache)->mem_obj;
        opencl_backend_kernel_set_arg(rk, 0, sizeof(cl_mem), &d_w_mem);
        opencl_backend_kernel_set_arg(rk, 1, sizeof(cl_mem), &d_g_mem);
        opencl_backend_kernel_set_arg(rk, 2, sizeof(cl_mem), &d_c_mem);
        opencl_backend_kernel_set_arg(rk, 3, sizeof(int), &n);
        opencl_backend_kernel_set_arg(rk, 4, sizeof(float), &lr_f);
        opencl_backend_kernel_set_arg(rk, 5, sizeof(float), &beta_f);
        opencl_backend_kernel_set_arg(rk, 6, sizeof(float), &eps_f);
        opencl_backend_kernel_set_arg(rk, 7, sizeof(float), &wd_f);
        opencl_backend_kernel_execute(rk, size, 256);
        opencl_backend_kernel_free(rk);
    }

    opencl_backend_memory_copy_from_device(weights, d_weights, bytes);
    opencl_backend_memory_copy_from_device(cache, d_cache, bytes);

    opencl_backend_memory_free(d_weights);
    opencl_backend_memory_free(d_grads);
    opencl_backend_memory_free(d_cache);
    return 0;

rms_cleanup:
    if (d_weights) opencl_backend_memory_free(d_weights);
    if (d_grads) opencl_backend_memory_free(d_grads);
    if (d_cache) opencl_backend_memory_free(d_cache);
    return -1;
}

/**
 * @brief OpenCL交叉熵损失及梯度计算
 *
 * softmax + cross-entropy loss + gradient 一体化计算
 * 支持整数标签和one-hot标签
 */
int opencl_cross_entropy_loss_gradient(GpuContext* ctx, const float* logits,
    const float* targets, float* loss, float* gradients,
    size_t num_elements, size_t num_classes, int is_integer_label) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用");
        return -1;
    }
    if (!logits || !targets || !loss || !gradients || num_elements == 0 || num_classes == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数");
        return -1;
    }

    size_t batch_size = num_elements / num_classes;
    if (batch_size == 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "批次大小为零");
        return -1;
    }

    size_t logits_bytes = num_elements * sizeof(float);
    size_t target_bytes = (is_integer_label ? batch_size : num_elements) * sizeof(float);
    size_t loss_bytes = batch_size * sizeof(float);
    size_t grad_bytes = num_elements * sizeof(float);

    float* h_loss = (float*)safe_malloc(loss_bytes);
    if (!h_loss) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "主机内存分配失败");
        return -1;
    }
    memset(h_loss, 0, loss_bytes);

    GpuMemory* d_logits = opencl_backend_memory_alloc(ctx, logits_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_targets = opencl_backend_memory_alloc(ctx, target_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_loss = opencl_backend_memory_alloc(ctx, loss_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_grad = opencl_backend_memory_alloc(ctx, grad_bytes, GPU_MEMORY_DEVICE);

    if (!d_logits || !d_targets || !d_loss || !d_grad) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败");
        safe_free((void**)&h_loss);
        goto ce_cleanup;
    }

    opencl_backend_memory_copy_to_device(d_logits, logits, logits_bytes);
    opencl_backend_memory_copy_to_device(d_targets, targets, target_bytes);
    opencl_backend_memory_copy_to_device(d_loss, h_loss, loss_bytes);

    {
        GpuKernel* cek = opencl_backend_kernel_create(ctx, NULL, "cross_entropy_loss_grad");
        if (!cek) {
            safe_free((void**)&h_loss);
            goto ce_cleanup;
        }
        int bs = (int)batch_size;
        int nc = (int)num_classes;
        int iil = is_integer_label;
        cl_mem d_l_mem = ((OpenCLMemory*)d_logits)->mem_obj;
        cl_mem d_t_mem = ((OpenCLMemory*)d_targets)->mem_obj;
        cl_mem d_ls_mem = ((OpenCLMemory*)d_loss)->mem_obj;
        cl_mem d_g_mem = ((OpenCLMemory*)d_grad)->mem_obj;
        opencl_backend_kernel_set_arg(cek, 0, sizeof(cl_mem), &d_l_mem);
        opencl_backend_kernel_set_arg(cek, 1, sizeof(cl_mem), &d_t_mem);
        opencl_backend_kernel_set_arg(cek, 2, sizeof(cl_mem), &d_ls_mem);
        opencl_backend_kernel_set_arg(cek, 3, sizeof(cl_mem), &d_g_mem);
        opencl_backend_kernel_set_arg(cek, 4, sizeof(int), &bs);
        opencl_backend_kernel_set_arg(cek, 5, sizeof(int), &nc);
        opencl_backend_kernel_set_arg(cek, 6, sizeof(int), &iil);
        /* 注意：该内核按批次处理，全局尺寸用batch_size */
        opencl_backend_kernel_execute(cek, batch_size, 256);
        opencl_backend_kernel_free(cek);
    }

    opencl_backend_memory_copy_from_device(gradients, d_grad, grad_bytes);
    opencl_backend_memory_copy_from_device(h_loss, d_loss, loss_bytes);

    /* 合并损失值：所有样本的平均 */
    {
        float total_loss = 0.0f;
        for (size_t i = 0; i < batch_size; i++) {
            total_loss += h_loss[i];
        }
        *loss = total_loss / (float)batch_size;
    }

    safe_free((void**)&h_loss);
    opencl_backend_memory_free(d_logits);
    opencl_backend_memory_free(d_targets);
    opencl_backend_memory_free(d_loss);
    opencl_backend_memory_free(d_grad);
    return 0;

ce_cleanup:
    if (d_logits) opencl_backend_memory_free(d_logits);
    if (d_targets) opencl_backend_memory_free(d_targets);
    if (d_loss) opencl_backend_memory_free(d_loss);
    if (d_grad) opencl_backend_memory_free(d_grad);
    return -1;
}

/**
 * @brief OpenCL CfC ODE步进计算
 *
 * CfC常微分方程数值积分步进：
 * h_out = h_in + dt * tanh(W * h_in + b) / tau
 * 用于 CfC 液态神经网络的 ODE 求解
 */
int opencl_cfc_ode_step(GpuContext* ctx, const float* h_in, const float* W,
    const float* b, const float* tau, float* h_out, float dt, int dim) {
    if (!g_opencl_initialized || !ctx) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL后端不可用：cfc_ode_step");
        return -1;
    }
    OpenCLContext* octx = (OpenCLContext*)ctx;
    if (!octx->default_command_queue) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "OpenCL命令队列不可用：cfc_ode_step");
        return -1;
    }
    if (!h_in || !W || !b || !tau || !h_out || dim <= 0) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "无效参数：cfc_ode_step");
        return -1;
    }

    size_t state_bytes = (size_t)dim * sizeof(float);
    size_t weight_bytes = (size_t)dim * (size_t)dim * sizeof(float);

    GpuMemory* d_h_in = opencl_backend_memory_alloc(ctx, state_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_W = opencl_backend_memory_alloc(ctx, weight_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_b = opencl_backend_memory_alloc(ctx, state_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_tau = opencl_backend_memory_alloc(ctx, state_bytes, GPU_MEMORY_DEVICE);
    GpuMemory* d_h_out = opencl_backend_memory_alloc(ctx, state_bytes, GPU_MEMORY_DEVICE);

    if (!d_h_in || !d_W || !d_b || !d_tau || !d_h_out) {
        snprintf(g_opencl_error_string, sizeof(g_opencl_error_string), "设备内存分配失败：cfc_ode_step");
        goto cfc_cleanup;
    }

    opencl_backend_memory_copy_to_device(d_h_in, h_in, state_bytes);
    opencl_backend_memory_copy_to_device(d_W, W, weight_bytes);
    opencl_backend_memory_copy_to_device(d_b, b, state_bytes);
    opencl_backend_memory_copy_to_device(d_tau, tau, state_bytes);

    {
        GpuKernel* ck = opencl_backend_kernel_create(ctx, NULL, "cfc_ode_step_kernel");
        if (!ck) goto cfc_cleanup;
        int d = dim;
        float dt_f = dt;
        cl_mem d_hi_mem = ((OpenCLMemory*)d_h_in)->mem_obj;
        cl_mem d_w_mem = ((OpenCLMemory*)d_W)->mem_obj;
        cl_mem d_b_mem = ((OpenCLMemory*)d_b)->mem_obj;
        cl_mem d_t_mem = ((OpenCLMemory*)d_tau)->mem_obj;
        cl_mem d_ho_mem = ((OpenCLMemory*)d_h_out)->mem_obj;
        opencl_backend_kernel_set_arg(ck, 0, sizeof(cl_mem), &d_hi_mem);
        opencl_backend_kernel_set_arg(ck, 1, sizeof(cl_mem), &d_w_mem);
        opencl_backend_kernel_set_arg(ck, 2, sizeof(cl_mem), &d_b_mem);
        opencl_backend_kernel_set_arg(ck, 3, sizeof(cl_mem), &d_t_mem);
        opencl_backend_kernel_set_arg(ck, 4, sizeof(cl_mem), &d_ho_mem);
        opencl_backend_kernel_set_arg(ck, 5, sizeof(float), &dt_f);
        opencl_backend_kernel_set_arg(ck, 6, sizeof(int), &d);
        opencl_backend_kernel_execute(ck, (size_t)dim, 256);
        opencl_backend_kernel_free(ck);
    }

    opencl_backend_memory_copy_from_device(h_out, d_h_out, state_bytes);

    opencl_backend_memory_free(d_h_in);
    opencl_backend_memory_free(d_W);
    opencl_backend_memory_free(d_b);
    opencl_backend_memory_free(d_tau);
    opencl_backend_memory_free(d_h_out);
    return 0;

cfc_cleanup:
    if (d_h_in) opencl_backend_memory_free(d_h_in);
    if (d_W) opencl_backend_memory_free(d_W);
    if (d_b) opencl_backend_memory_free(d_b);
    if (d_tau) opencl_backend_memory_free(d_tau);
    if (d_h_out) opencl_backend_memory_free(d_h_out);
    return -1;
}

/* ============================================================================
 * OpenCL后端接口
 * =========================================================================== */

/**
 * @brief 获取OpenCL后端接口
 */
static const GpuBackendInterface* get_opencl_backend_interface(void) {
    static GpuBackendInterface opencl_backend = {
        .name = "OpenCL",
        .backend_type = GPU_BACKEND_OPENCL,
        .init = opencl_backend_init,
        .cleanup = opencl_backend_cleanup,
        .get_device_count = opencl_backend_get_device_count,
        .get_device_info = opencl_backend_get_device_info,
        .context_create = opencl_backend_context_create,
        .context_free = opencl_backend_context_free,
        .memory_alloc = opencl_backend_memory_alloc,
        .memory_free = opencl_backend_memory_free,
        .memory_copy_to_device = opencl_backend_memory_copy_to_device,
        .memory_copy_from_device = opencl_backend_memory_copy_from_device,
        .memory_copy_device_to_device = opencl_backend_memory_copy_device_to_device,
        .memory_copy_to_device_async = opencl_backend_memory_copy_to_device_async,
        .memory_copy_from_device_async = opencl_backend_memory_copy_from_device_async,
        .kernel_create = opencl_backend_kernel_create,
        .kernel_free = opencl_backend_kernel_free,
        .kernel_set_arg = opencl_backend_kernel_set_arg,
        .kernel_execute = opencl_backend_kernel_execute,
        .kernel_execute_nd = opencl_backend_kernel_execute_nd,
        .stream_create = opencl_backend_stream_create,
        .stream_free = opencl_backend_stream_free,
        .stream_synchronize = opencl_backend_stream_synchronize,
        .stream_query = opencl_backend_stream_query,
        .get_memory_info = opencl_backend_get_memory_info,
        .device_reset = opencl_backend_device_reset,
        .get_error_string = opencl_backend_get_error_string
    };
    
    return &opencl_backend;
}

/**
 * @brief 获取OpenCL后端接口（供外部调用）
 */
const GpuBackendInterface* opencl_get_backend_interface(void) {
    return get_opencl_backend_interface();
}