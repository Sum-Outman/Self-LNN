# SELF-LNN 嵌入式平台部署指南 / Embedded Platform Deployment Guide

本文档详细介绍如何将 SELF-LNN 全液态神经网络系统部署到各种嵌入式平台，包括 FPGA SoC、ARM Linux 开发板和树莓派。
This document details how to deploy the SELF-LNN full liquid neural network system to various embedded platforms, including FPGA SoCs, ARM Linux development boards, and Raspberry Pi.

> **重要声明 / Important Notice**：SELF-LNN 完整 AGI 系统（含记忆、规划、认知、多模态、多机器人协调）运行时需要 **70~110MB 常驻内存**，仅可在具备 Linux 操作系统的嵌入式平台（如树莓派、Jetson、FPGA SoC 硬核处理器）上运行。裁剪后的核心 CfC 推理引擎可在高端 MCU（STM32H7、ESP32）上有限运行。
> The full SELF-LNN AGI system (with memory, planning, cognition, multimodal, multi-robot coordination) requires **70~110MB resident memory** and can only run on embedded platforms with a Linux OS (e.g., Raspberry Pi, Jetson, FPGA SoC hard-core processors). The trimmed core CfC inference engine can run on high-end MCUs (STM32H7, ESP32) with limited functionality.

## 目录 / Table of Contents
1. [概述 / Overview](#概述--overview)
2. [FPGA 平台部署 / FPGA Platform Deployment](#fpga-平台部署--fpga-platform-deployment)
3. [ARM 平台部署 / ARM Platform Deployment](#arm-平台部署--arm-platform-deployment)
4. [轻量推理引擎部署（ESP32/STM32）/ Lightweight Inference Engine Deployment](#轻量推理引擎部署esp32stm32--lightweight-inference-engine-deploymentesp32stm32)
5. [树莓派部署 / Raspberry Pi Deployment](#树莓派部署--raspberry-pi-deployment)
6. [性能优化 / Performance Optimization](#性能优化--performance-optimization)
7. [常见问题 / FAQ](#常见问题--faq)

---

## 概述 / Overview

SELF-LNN 采用纯 C 语言实现，无任何外部依赖，这使得它适合部署到具备足够内存的嵌入式平台。部署时需要考虑以下因素：
SELF-LNN is implemented in pure C with no external dependencies, making it suitable for deployment to embedded platforms with sufficient memory. The following factors need to be considered during deployment:

- **硬件资源限制 / Hardware Resource Constraints**：处理器性能、**内存容量（最关键因素）**、存储容量 / Processor performance, **memory capacity (most critical factor)**, storage capacity
- **实时性要求 / Real-time Requirements**：系统响应时间、中断处理 / System response time, interrupt handling
- **功耗限制 / Power Constraints**：嵌入式设备的能源消耗 / Energy consumption of embedded devices
- **外设接口 / Peripheral Interfaces**：传感器、执行器的连接方式 / Connection methods for sensors and actuators

### 各平台可行性总览 / Platform Feasibility Overview

| 平台 / Platform | 内存 / Memory | 可行性 / Feasibility | 说明 / Description |
|------|------|--------|------|
| x86_64桌面/服务器 (Desktop/Server) | ≥8GB | ✅ 完整系统 (Full) | 全功能运行，含训练 / Full functionality including training |
| 树莓派 3B+/4B/5 (Raspberry Pi) | ≥4GB | ✅ 完整系统 (Full) | 推理+有限训练 / Inference + limited training |
| NVIDIA Jetson | ≥4GB | ✅ 完整系统 (Full) | GPU加速推理 / GPU-accelerated inference |
| Zynq-7000 / UltraScale+ | 512MB~1GB DDR | ✅ 可运行 (Runnable) | ARM硬核运行C代码+FPGA加速 / ARM core runs C code + FPGA acceleration |
| Cyclone V / Arria 10 SoC | 256MB~1GB DDR | ✅ 可运行 (Runnable) | ARM硬核+HPS |
| STM32H7 (Cortex-M7) | 1MB RAM | ⚠️ 仅推理引擎 (Inference only) | 裁剪后运行CfC核心推理 / Trimmed CfC core inference |
| ESP32 | 520KB RAM | ⚠️ 有限推理 (Limited) | 极小网络推理（≤64隐藏单元）/ Minimal network (≤64 hidden units) |

---

## FPGA 平台部署 / FPGA Platform Deployment

### 适用 FPGA 平台 / Supported FPGA Platforms
- Xilinx (AMD)：Zynq-7000 系列、Zynq UltraScale+（含 ARM 硬核处理器 / with ARM hard-core processor）
- Intel (Altera)：Cyclone V、Arria 10 系列（含 ARM 硬核 HPS / with ARM hard-core HPS）

> **注意 / Note**：仅支持含 CPU 硬核的 FPGA SoC 方案（ARM硬核运行 C 代码 + FPGA 逻辑加速矩阵运算）。纯 FPGA 逻辑（无 CPU 硬核）无法直接运行 SELF-LNN 的 C 代码。
> Only FPGA SoCs with CPU hard cores are supported (ARM runs C code + FPGA accelerates matrix operations). Pure FPGA logic (without CPU hard core) cannot directly run SELF-LNN's C code.

### 部署方式 / Deployment Methods

#### 方式一：基于软核 CPU (MicroBlaze / Nios II) / Method 1: Soft-core CPU Based

**1. 准备开发环境 / Prepare Development Environment**

```bash
# Xilinx Vivado
# 需要安装 Vivado Design Suite 2021.1 或更高版本

# Intel Quartus
# 需要安装 Quartus Prime 18.0 或更高版本
```

**2. 创建嵌入式硬件平台 / Create Embedded Hardware Platform**

```tcl
# Vivado Tcl 示例 (Xilinx Zynq)
create_project -force my_lnn_project ./my_lnn_project -part xc7z020clg484-1

# 添加 Zynq 处理系统
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" } [get_bd_cells processing_system7_0]

# 添加 GPIO (用于传感器/执行器)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_gpio_0/S_AXI} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0} } [get_bd_intf_pins axi_gpio_0/S_AXI]

# 生成比特流
validate_bd_design
save_bd_design
make_wrapper -files [get_files ./my_lnn_project/my_lnn_project.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ./my_lnn_project/my_lnn_project.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v
update_compile_order -fileset sources_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
```

**3. 交叉编译 SELF-LNN / Cross-compile SELF-LNN**

```bash
# 创建嵌入式平台 CMake 工具链文件
cat > toolchain-zynq.cmake << 'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-xilinx-linux-gnueabi-gcc)
set(CMAKE_CXX_COMPILER arm-xilinx-linux-gnueabi-g++)

set(CMAKE_FIND_ROOT_PATH /opt/Xilinx/SDK/2021.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF

# 配置和编译
mkdir build-fpga && cd build-fpga
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-zynq.cmake \
      -DBUILD_TESTS=OFF \
      -DBUILD_EXAMPLES=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..
cmake --build . --parallel
```

**4. 定制平台适配层 / Custom Platform Adaptation Layer**

```c
// src/utils/platform_fpga.c
#include "selflnn/utils/platform.h"
#include <xil_printf.h>
#include <xil_io.h>
#include <sleep.h>

// 平台类型检测
PlatformType platform_get_type(void) {
    // 对于 FPGA，我们可以返回特定类型或 UNKNOWN
    return PLATFORM_UNKNOWN;
}

// 简化的延时函数
void time_sleep_ms(unsigned int milliseconds) {
    usleep(milliseconds * 1000);
}

// 简化的内存管理 (使用片上 BRAM)
void* memory_map(size_t size) {
    // 在 FPGA 上，我们可以分配内存或直接映射 BRAM 地址
    return malloc(size);
}
```

#### 方式二：HLS 高层次综合实现 / Method 2: HLS High-Level Synthesis

对于需要更高性能的液态神经网络，可以使用 Vivado HLS 将核心 CfC 细胞单元转换为硬件加速器。
For higher performance requirements, Vivado HLS can be used to convert the core CfC cell unit into a hardware accelerator.

```c
// cfc_cell_hls.cpp - HLS 实现：真正的 CfC 封闭形式解
#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_math.h"

typedef ap_fixed<16, 4> fixed16_t;
typedef ap_fixed<32, 8> fixed32_t;

// CfC HLS 加速器核心
// 实现 CfC 封闭形式解:
//   tau = sigmoid(W_tau * x + b_tau)
//   driver = sigmoid(W_drv * x + b_drv) * tanh(W_h * x + b_h)
//   h_new = h * exp(-dt/tau) + (1 - exp(-dt/tau)) * driver
void cfc_cell_hls(
    hls::stream<fixed16_t> &input,
    hls::stream<fixed16_t> &state_in,
    hls::stream<fixed16_t> &state_out,
    fixed16_t weights_in[64],     // W_tau: 输入→时间常数权重
    fixed16_t weights_drv[64],    // W_drv: 输入→驱动门权重
    fixed16_t weights_hid[64],    // W_h:  输入→隐藏权重
    fixed16_t biases[3],          // [b_tau, b_drv, b_h]
    fixed16_t dt                  // 时间步长
) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=state_in
#pragma HLS INTERFACE axis port=state_out
#pragma HLS INTERFACE bram port=weights_in
#pragma HLS INTERFACE bram port=weights_drv
#pragma HLS INTERFACE bram port=weights_hid
#pragma HLS INTERFACE bram port=biases
#pragma HLS PIPELINE II=1

    fixed16_t x = input.read();
    fixed16_t h = state_in.read();
    
    // 计算时间常数 tau = sigmoid(W_tau * x + b_tau)
    fixed32_t sum_tau = biases[0];
    for (int i = 0; i < 64; i++) {
        sum_tau += weights_in[i] * x;
    }
    fixed16_t tau = 1.0f / (1.0f + hls::exp(-sum_tau));
    
    // 计算驱动门 sigmoid(W_drv * x + b_drv)
    fixed32_t sum_drv = biases[1];
    for (int i = 0; i < 64; i++) {
        sum_drv += weights_drv[i] * x;
    }
    fixed16_t gate_drv = 1.0f / (1.0f + hls::exp(-sum_drv));
    
    // 计算隐藏激活 tanh(W_h * x + b_h)
    fixed32_t sum_hid = biases[2];
    for (int i = 0; i < 64; i++) {
        sum_hid += weights_hid[i] * x;
    }
    fixed16_t act_hid = hls::tanh(sum_hid);
    
    // 驱动函数: driver = sigmoid(W_drv * x + b_drv) * tanh(W_h * x + b_h)
    fixed16_t driver = gate_drv * act_hid;
    
    // CfC 封闭形式解: h_new = h * exp(-dt/tau) + (1 - exp(-dt/tau)) * driver
    fixed16_t decay = hls::exp(-dt / tau);
    fixed16_t h_new = h * decay + (1.0f - decay) * driver;
    
    state_out.write(h_new);
}
```

### FPGA 部署步骤总结 / FPGA Deployment Summary

1. **硬件平台设计 / Hardware Platform Design**：使用 Vivado/Quartus 创建包含处理器和外设的硬件 / Create hardware with processor and peripherals using Vivado/Quartus
2. **软件交叉编译 / Software Cross-compilation**：使用交叉编译器编译 SELF-LNN / Compile SELF-LNN with cross-compiler
3. **平台适配 / Platform Adaptation**：实现定制的平台抽象层 / Implement custom platform abstraction layer
4. **系统集成 / System Integration**：将软件和硬件整合 / Integrate software and hardware
5. **测试验证 / Testing and Verification**：在 FPGA 上验证系统功能 / Verify system functionality on FPGA

---

## ARM 平台部署 / ARM Platform Deployment

### 适用 ARM 平台 / Supported ARM Platforms
- 树莓派 (见树莓派章节 / see Raspberry Pi section)
- BeagleBone
- NVIDIA Jetson

### 通用 ARM 部署步骤 / Generic ARM Deployment Steps

**1. 准备交叉编译环境 / Prepare Cross-compilation Environment**

```bash
# Ubuntu/Debian 安装 ARM 交叉编译器
sudo apt update
sudo apt install gcc-arm-linux-gnueabi g++-arm-linux-gnueabi

# 或者使用 ARM GCC 工具链
wget https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-eabi.tar.xz
tar -xf arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-eabi.tar.xz
export PATH=$PATH:/path/to/arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-eabi/bin
```

**2. 创建 ARM 平台 CMake 配置 / Create ARM CMake Configuration**

```cmake
# toolchain-arm.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# 交叉编译器设置
set(CMAKE_C_COMPILER arm-linux-gnueabi-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabi-g++)

# 系统根目录 (可选)
# set(CMAKE_SYSROOT /path/to/sysroot)

# 搜索路径模式
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# 编译标志优化
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=armv7-a -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard -ffast-math")
```

**3. 编译项目 / Build the Project**

```bash
mkdir build-arm && cd build-arm
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-arm.cmake \
      -DBUILD_TESTS=ON \
      -DBUILD_EXAMPLES=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..
cmake --build . --parallel
```

**4. 创建轻量级配置 / Create Lightweight Configuration**

```c
// include/selflnn/config.h - 嵌入式配置
#ifndef SELFLNN_CONFIG_H
#define SELFLNN_CONFIG_H

// 内存配置
#define LNN_MAX_INPUT_SIZE 256
#define LNN_MAX_HIDDEN_SIZE 128
#define LNN_MAX_OUTPUT_SIZE 64
#define LNN_MAX_LAYERS 4

// 内存池大小 (根据可用 RAM 调整)
// 注意：此配置仅适用于 ARM Linux 平台（如树莓派、BeagleBone），
// 不可用于 STM32 等裸机 MCU。STM32 需使用下面 ESP32/STM32 章节的 CfC 轻量引擎方案。
// 64KB 内存池仅够最基本的推理分配，完整 AGI 系统至少需要 4MB 以上。
#define LNN_MEMORY_POOL_SIZE (64 * 1024)  // 64KB

// 禁用不需要的功能以节省内存
#define LNN_DISABLE_MULTIAGENT
#define LNN_DISABLE_KNOWLEDGE_GRAPH
#define LNN_DISABLE_DEEP_VISION

#endif // SELFLNN_CONFIG_H
```

**5. ARM 平台特定优化 / ARM Platform-specific Optimization**

```c
// src/utils/math_utils_arm.c - NEON 优化
#include <arm_neon.h>

// NEON 优化的矩阵乘法
void matrix_multiply_neon(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j += 4) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int p = 0; p < k; p++) {
                float32x4_t a = vdupq_n_f32(A[i * k + p]);
                float32x4_t b = vld1q_f32(&B[p * n + j]);
                sum = vfmaq_f32(sum, a, b);
            }
            vst1q_f32(&C[i * n + j], sum);
        }
    }
}
```

---

> **STM32 部署说明 / STM32 Deployment Note**：STM32H7（Cortex-M7, 1MB RAM）及更高端型号可运行 CfC 轻量推理引擎，具体部署方式见下方 [轻量推理引擎部署（ESP32/STM32）](#轻量推理引擎部署esp32stm32--lightweight-inference-engine-deploymentesp32stm32) 章节。
> STM32H7 (Cortex-M7, 1MB RAM) and higher-end models can run the CfC lightweight inference engine. See the [Lightweight Inference Engine](#轻量推理引擎部署esp32stm32--lightweight-inference-engine-deploymentesp32stm32) section for details.

## 轻量推理引擎部署（ESP32/STM32） / Lightweight Inference Engine Deployment (ESP32/STM32)

### 适用平台与可行性 / Supported Platforms and Feasibility

| 平台 / Platform | Flash | RAM | 可行性 / Feasibility | 说明 / Description |
|------|-------|-----|--------|------|
| **ESP32** (ESP-WROOM-32) | 4MB | 520KB SRAM | ⚠️ 有限推理 (Limited) | 可运行 CfC 核心推理引擎（≤64隐藏单元，无记忆/规划模块）/ CfC core inference engine (≤64 hidden units, no memory/planning) |
| **STM32H7** (Cortex-M7) | 2MB | 1MB RAM | ⚠️ 有限推理 (Limited) | 可运行裁剪版 CfC + 轻量知识库 / Trimmed CfC + lightweight knowledge base |

### ESP32 部署指南（推荐轻量平台） / ESP32 Deployment Guide (Recommended Lightweight Platform)

ESP32 拥有 520KB SRAM 和双核 Xtensa LX6 处理器（240MHz），可运行裁剪版 CfC 推理引擎。
ESP32 has 520KB SRAM and a dual-core Xtensa LX6 processor (240MHz), capable of running a trimmed CfC inference engine.

**1. 项目结构 / Project Structure**

```
selflnn-esp32/
├── components/
│   └── selflnn/
│       ├── include/
│       │   └── cfc_minimal.h
│       └── cfc_minimal.c
├── main/
│   ├── CMakeLists.txt
│   └── main.c
└── CMakeLists.txt
```

**2. 真正的 CfC 轻量推理引擎实现 / CfC Lightweight Inference Engine Implementation**

```c
// components/selflnn/include/cfc_minimal.h
#ifndef CFC_MINIMAL_H
#define CFC_MINIMAL_H

#include <stdint.h>

// CfC 轻量引擎配置
#define CFC_MAX_INPUT_SIZE   32
#define CFC_MAX_HIDDEN_SIZE  64
#define CFC_MAX_OUTPUT_SIZE  16

// CfC 轻量引擎
typedef struct {
    // 权重矩阵 [隐藏维度][输入维度]
    float W_tau[CFC_MAX_HIDDEN_SIZE * CFC_MAX_INPUT_SIZE];
    float W_drv[CFC_MAX_HIDDEN_SIZE * CFC_MAX_INPUT_SIZE];
    float W_hid[CFC_MAX_HIDDEN_SIZE * CFC_MAX_INPUT_SIZE];
    
    // 偏置
    float b_tau[CFC_MAX_HIDDEN_SIZE];
    float b_drv[CFC_MAX_HIDDEN_SIZE];
    float b_hid[CFC_MAX_HIDDEN_SIZE];
    
    // 输出权重 [输出维度][隐藏维度]
    float W_out[CFC_MAX_OUTPUT_SIZE * CFC_MAX_HIDDEN_SIZE];
    float b_out[CFC_MAX_OUTPUT_SIZE];
    
    // 隐藏状态
    float hidden_state[CFC_MAX_HIDDEN_SIZE];
    
    // 实际维度（可在初始化时设置较小值以节省内存）
    int input_size;
    int hidden_size;
    int output_size;
} CfCMinimal;

void cfc_minimal_init(CfCMinimal* net, int in, int hid, int out);
void cfc_minimal_forward(CfCMinimal* net, const float* input, float* output, float dt);

#endif
```

```c
// components/selflnn/cfc_minimal.c
#include "cfc_minimal.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static float sigmoidf(float x) {
    if (x >= 8.0f) return 1.0f;
    if (x <= -8.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

void cfc_minimal_init(CfCMinimal* net, int in, int hid, int out) {
    net->input_size = in;
    net->hidden_size = hid;
    net->output_size = out;
    
    // 权重随机初始化（Xavier 初始化）
    float scale = sqrtf(2.0f / (in + hid));
    for (int i = 0; i < hid * in; i++) {
        net->W_tau[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        net->W_drv[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        net->W_hid[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
    for (int i = 0; i < hid; i++) {
        net->b_tau[i] = 0.0f;
        net->b_drv[i] = 0.0f;
        net->b_hid[i] = 0.0f;
    }
    scale = sqrtf(2.0f / hid);
    for (int i = 0; i < out * hid; i++) {
        net->W_out[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
    for (int i = 0; i < out; i++) {
        net->b_out[i] = 0.0f;
    }
    
    memset(net->hidden_state, 0, hid * sizeof(float));
}

void cfc_minimal_forward(CfCMinimal* net, const float* input, float* output, float dt) {
    int in = net->input_size;
    int hid = net->hidden_size;
    int out = net->output_size;
    
    float new_hidden[CFC_MAX_HIDDEN_SIZE];
    
    // 逐神经元计算 CfC 封闭形式解
    for (int i = 0; i < hid; i++) {
        // 计算 tau = sigmoid(W_tau * x + b_tau)
        float sum_tau = net->b_tau[i];
        for (int j = 0; j < in; j++) {
            sum_tau += net->W_tau[i * in + j] * input[j];
        }
        float tau = sigmoidf(sum_tau);
        // tau 约束在 [0.01, 0.99] 防止数值不稳定
        if (tau < 0.01f) tau = 0.01f;
        if (tau > 0.99f) tau = 0.99f;
        
        // 计算驱动门 sigmoid(W_drv * x + b_drv)
        float sum_drv = net->b_drv[i];
        for (int j = 0; j < in; j++) {
            sum_drv += net->W_drv[i * in + j] * input[j];
        }
        float gate_drv = sigmoidf(sum_drv);
        
        // 计算隐藏激活 tanh(W_hid * x + b_hid)
        float sum_hid = net->b_hid[i];
        for (int j = 0; j < in; j++) {
            sum_hid += net->W_hid[i * in + j] * input[j];
        }
        float act_hid = tanhf(sum_hid);
        
        // driver = sigmoid(W_drv * x) * tanh(W_hid * x)
        float driver = gate_drv * act_hid;
        
        // CfC 封闭形式解: h_new = h * exp(-dt/tau) + (1 - exp(-dt/tau)) * driver
        float decay = expf(-dt / tau);
        new_hidden[i] = net->hidden_state[i] * decay + (1.0f - decay) * driver;
    }
    
    // 更新隐藏状态
    memcpy(net->hidden_state, new_hidden, hid * sizeof(float));
    
    // 计算输出
    for (int i = 0; i < out; i++) {
        float sum = net->b_out[i];
        for (int j = 0; j < hid; j++) {
            sum += net->W_out[i * hid + j] * net->hidden_state[j];
        }
        output[i] = sum;
    }
}
```

**3. ESP32 主程序示例 / ESP32 Main Program Example**

```c
// main/main.c
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "cfc_minimal.h"

static const char* TAG = "SELF-LNN";
static CfCMinimal lnn;
static float sensor_buf[CFC_MAX_INPUT_SIZE];
static float control_buf[CFC_MAX_OUTPUT_SIZE];

// 传感器读取任务
void sensor_task(void* arg) {
    while (1) {
        // 读取模拟传感器（示例：ADC 读取）
        for (int i = 0; i < lnn.input_size; i++) {
            // sensor_buf[i] = adc_read(i);  // 实际 ADC 读取
            sensor_buf[i] = 0.0f;
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// CfC 推理任务
void inference_task(void* arg) {
    while (1) {
        cfc_minimal_forward(&lnn, sensor_buf, control_buf, 0.01f);
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

// 控制输出任务
void control_task(void* arg) {
    while (1) {
        for (int i = 0; i < lnn.output_size; i++) {
            int pwm = (int)((control_buf[i] + 1.0f) * 127.5f);
            if (pwm < 0) pwm = 0;
            if (pwm > 255) pwm = 255;
            // ledc_write(i, pwm);  // 实际 PWM 输出
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "SELF-LNN CfC 轻量推理引擎 (ESP32)");
    
    // 初始化 CfC 网络: 输入=16, 隐藏=32, 输出=8
    // ESP32 内存限制：32隐藏单元约需 32×16×3×4 + 32×3×4 + 8×32×4 ≈ 7.5KB 权重
    cfc_minimal_init(&lnn, 16, 32, 8);
    ESP_LOGI(TAG, "CfC 初始化完成: %d→%d→%d", lnn.input_size, lnn.hidden_size, lnn.output_size);
    
    // 创建 FreeRTOS 任务
    xTaskCreate(sensor_task, "sensor", 4096, NULL, 1, NULL);
    xTaskCreate(inference_task, "inference", 8192, NULL, 2, NULL);
    xTaskCreate(control_task, "control", 4096, NULL, 1, NULL);
}
```

**4. CMake 构建配置 / CMake Build Configuration**

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(selflnn_esp32)
```

### ESP32 内存估算表 / ESP32 Memory Estimation Table

| 配置 (输入→隐藏→输出) / Config (Input→Hidden→Output) | 权重大小 / Weights | 运行时缓冲区 / Runtime Buffer | 总计 / Total | 可行性 / Feasibility |
|----------------------|---------|------------|------|--------|
| 8→16→4 | ~2.5KB | ~4KB | ~6.5KB | ✅ 绰绰有余 (Ample) |
| 16→32→8 | ~7.5KB | ~8KB | ~15.5KB | ✅ 可行 (Feasible) |
| 32→64→16 | ~33KB | ~16KB | ~49KB | ⚠️ 紧张但可行 (Tight but feasible) |
| 64→128→32 | ~132KB | ~32KB | ~164KB | ❌ 超出ESP32内存 (Exceeds ESP32 memory) |

> **ESP32 部署限制 / ESP32 Deployment Limitations**：仅支持 CfC 核心推理，不支持记忆管理器、知识库、规划引擎、自我认知、多机器人协调等高级功能。这些功能需在 PC/服务器端运行。
> Only CfC core inference is supported. High-level features (memory manager, knowledge base, planning engine, self-cognition, multi-robot coordination) are NOT supported and must run on PC/server.

### STM32H7 部署 / STM32H7 Deployment

STM32H7（Cortex-M7 @400MHz, 1MB RAM）比 ESP32 有更充裕的内存，可运行稍大规模的 CfC 推理引擎。
STM32H7 (Cortex-M7 @400MHz, 1MB RAM) has more abundant memory than ESP32 and can run larger CfC inference engines.

**使用 STM32CubeIDE：/ Using STM32CubeIDE:**

1. 创建新的 STM32H7 项目 / Create a new STM32H7 project
2. 将上述 `cfc_minimal.h` 和 `cfc_minimal.c` 添加到项目 / Add `cfc_minimal.h` and `cfc_minimal.c` to the project
3. 配置内存管理（使用静态内存分配）/ Configure memory management (use static memory allocation)
4. 适配平台抽象层 / Adapt the platform abstraction layer

```c
// platform_stm32.c - STM32 平台适配
#include "cfc_minimal.h"
#include "stm32h7xx_hal.h"

static CfCMinimal lnn;
static float input_buf[32];
static float output_buf[16];

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc) {
    // ADC 转换完成回调 — 触发推理
    cfc_minimal_forward(&lnn, input_buf, output_buf, 0.01f);
    
    // 映射输出到 PWM
    for (int i = 0; i < lnn.output_size; i++) {
        uint32_t duty = (uint32_t)((output_buf[i] + 1.0f) * 500.0f);
        __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1 + i, duty);
    }
}

void main(void) {
    HAL_Init();
    SystemClock_Config();
    
    // 初始化 CfC 网络: 16→64→8
    // STM32H7 1MB RAM：64隐藏单元配置约需 64×16×3×4 + 64×3×4 + 8×64×4 ≈ 14KB 权重
    cfc_minimal_init(&lnn, 16, 64, 8);
    
    // 启动 ADC 和 TIM（略）
    while (1) {
        // 主循环可处理通信等低优先级任务
    }
}
```

---

## 树莓派部署 / Raspberry Pi Deployment

树莓派是最流行的 ARM 嵌入式平台之一，具有丰富的外设支持和强大的社区支持。
Raspberry Pi is one of the most popular ARM embedded platforms, with rich peripheral support and a strong community.

### 1. 系统要求 / System Requirements

- 树莓派 3B+ 或更高版本 (推荐 4B/5) / Raspberry Pi 3B+ or higher (recommended: 4B/5)
- Raspberry Pi OS (64位 / 64-bit)
- 至少 4GB 内存 / At least 4GB RAM
- 至少 16GB 存储 / At least 16GB storage

### 2. 直接编译部署 / Direct Compilation and Deployment

树莓派足够强大，可以直接在上面编译代码。
Raspberry Pi is powerful enough to compile code directly on the device.

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装依赖
sudo apt install -y build-essential cmake git

# 项目已在本地，无需克隆
# Project is already local, no cloning needed
cd ~/self-Z

# 创建构建目录
mkdir build && cd build

# 配置 CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=ON \
      -DBUILD_EXAMPLES=ON \
      ..

# 编译 (树莓派 4 有 4 核)
cmake --build . --parallel

# 运行测试
ctest
```

### 3. 交叉编译 (可选，更快) / Cross-compilation (Optional, Faster)

在功能更强大的主机上交叉编译，然后传输到树莓派。
Cross-compile on a more powerful host machine, then transfer to the Raspberry Pi.

```bash
# 安装树莓派交叉编译器
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# 创建交叉编译工具链文件
cat > toolchain-rpi.cmake << 'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "-march=armv8-a+crc -mtune=cortex-a72")
EOF

# 交叉编译
mkdir build-rpi && cd build-rpi
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-rpi.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      ..
cmake --build . --parallel

# 传输到树莓派
scp -r build-rpi/examples pi@raspberrypi.local:~/selflnn/
```

### 4. 配置 GPIO 和外设 / Configure GPIO and Peripherals

使用 WiringPi 或 pigpio 库访问树莓派的 GPIO 和外设接口。
Use WiringPi or pigpio libraries to access Raspberry Pi GPIO and peripheral interfaces.

```c
// examples/raspberrypi_example.c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include "selflnn/core/lnn.h"
#include "selflnn/robot/robot.h"

// GPIO 控制
#define GPIO_EXPORT "/sys/class/gpio/export"
#define GPIO_UNEXPORT "/sys/class/gpio/unexport"
#define GPIO_DIR "/sys/class/gpio/gpio%d/direction"
#define GPIO_VAL "/sys/class/gpio/gpio%d/value"

void gpio_export(int pin) {
    int fd = open(GPIO_EXPORT, O_WRONLY);
    if (fd < 0) return;
    char buf[4];
    sprintf(buf, "%d", pin);
    write(fd, buf, strlen(buf));
    close(fd);
}

void gpio_set_direction(int pin, int out) {
    char path[64];
    sprintf(path, GPIO_DIR, pin);
    int fd = open(path, O_WRONLY);
    if (fd < 0) return;
    write(fd, out ? "out" : "in", out ? 3 : 2);
    close(fd);
}

void gpio_set_value(int pin, int value) {
    char path[64];
    sprintf(path, GPIO_VAL, pin);
    int fd = open(path, O_WRONLY);
    if (fd < 0) return;
    write(fd, value ? "1" : "0", 1);
    close(fd);
}

int main() {
    printf("SELF-LNN 树莓派示例\n");
    
    // 初始化 GPIO
    gpio_export(17);
    gpio_set_direction(17, 1);  // 输出
    gpio_export(27);
    gpio_set_direction(27, 1);  // 输出
    
    // 创建神经网络
    LNNConfig config = {
        .input_size = 16,
        .hidden_size = 64,
        .output_size = 8,
        .learning_rate = 0.01f
    };
    LNN* network = lnn_create(&config);
    
    // 主循环
    float input[16] = {0};
    float output[8] = {0};
    
    while (1) {
        // 读取传感器 (示例)
        for (int i = 0; i < 16; i++) {
            // 实际应用中这里会读取真实传感器
            input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        
        // 神经网络推理
        lnn_forward(network, input, output);
        
        // 输出到执行器
        for (int i = 0; i < 2; i++) {
            int value = output[i] > 0.0f ? 1 : 0;
            gpio_set_value(17 + i, value);
        }
        
        usleep(100000);  // 100ms 延时
    }
    
    // 清理
    lnn_free(network);
    
    return 0;
}
```

### 5. 使用 WiringPi 或 pigpio 库 / Using WiringPi or pigpio Library

虽然 SELF-LNN 本身不依赖外部库，但在实际应用中我们可以使用库来简化 GPIO 操作。
Although SELF-LNN itself does not depend on external libraries, in practical applications we can use libraries to simplify GPIO operations.

```bash
# 安装 pigpio
sudo apt install pigpio

# 启动 pigpio 守护进程
sudo pigpiod
```

```c
// 使用 pigpio 的示例
#include <pigpio.h>
#include "selflnn/core/lnn.h"

int main() {
    if (gpioInitialise() < 0) {
        fprintf(stderr, "pigpio initialization failed\n");
        return 1;
    }
    
    // 设置引脚
    gpioSetMode(17, PI_OUTPUT);
    gpioSetMode(27, PI_OUTPUT);
    gpioSetMode(22, PI_INPUT);
    
    // 创建网络...
    
    // 清理
    gpioTerminate();
    return 0;
}
```

### 6. 树莓派部署总结 / Raspberry Pi Deployment Summary

1. 系统准备：安装 Raspberry Pi OS 64位，确保有足够内存和存储 / System preparation: Install Raspberry Pi OS 64-bit, ensure sufficient memory and storage
2. 代码获取：克隆仓库 / Code acquisition: Clone the repository
3. 编译：可直接编译或交叉编译 / Build: Direct compilation or cross-compilation
4. 外设配置：配置 GPIO、I2C、SPI 等外设接口 / Peripheral configuration: Configure GPIO, I2C, SPI, etc.
5. 部署运行：传输可执行文件并运行 / Deployment: Transfer the executable and run

---

## 性能优化 / Performance Optimization

### 1. 内存优化 / Memory Optimization

- 使用静态内存分配替代动态分配 / Use static memory allocation instead of dynamic allocation
- 调整神经网络大小以匹配目标平台 / Adjust neural network size to match target platform
- 使用 `LNN_MEMORY_POOL_SIZE` 控制内存池大小 / Control memory pool size with `LNN_MEMORY_POOL_SIZE`
- 禁用不需要的功能模块 / Disable unnecessary feature modules

### 2. 计算优化 / Computation Optimization

- 使用定点数计算代替浮点数（FPGA 上尤其重要）/ Use fixed-point arithmetic instead of floating-point (especially important on FPGA)
- 使用硬件加速器加速矩阵运算 / Use hardware accelerators for matrix operations
- 优化内存访问模式，利用缓存 / Optimize memory access patterns, leverage cache

### 3. 功耗优化 / Power Optimization

- 动态频率调整 / Dynamic frequency scaling
- 睡眠模式管理 / Sleep mode management
- 外设电源管理 / Peripheral power management
- 批量推理减少唤醒次数 / Batch inference to reduce wake-up frequency

### 4. 实时性优化 / Real-time Optimization

- 使用 RTOS 实时调度 / Use RTOS real-time scheduling
- 中断优先级管理 / Interrupt priority management
- 时间关键路径优化 / Time-critical path optimization

---

## 常见问题 / Frequently Asked Questions (FAQ)

### Q: SELF-LNN 在嵌入式设备上需要多少内存？/ How much memory does SELF-LNN need on embedded devices?
A: **取决于功能等级，差距极大 / Varies greatly by functionality level:**

| 等级 / Level | Flash | RAM | 说明 / Description |
|------|-------|-----|------|
| ✅ 完整 AGI 系统 (Full AGI) | ≥500MB | **≥70~110MB** | 含记忆/规划/认知/多模态/多机器人，需要 Linux 系统支持 / With memory/planning/cognition/multimodal/multi-robot, requires Linux |
| ⚠️ CfC 推理引擎 (ESP32) | ~200KB | ~50KB | 仅 CfC 核心推理（≤64隐藏单元），无高级功能 / CfC core inference only (≤64 hidden units), no advanced features |
| ⚠️ CfC 推理引擎 (STM32H7) | ~300KB | ~128KB | CfC 核心推理 + 轻量知识库 / CfC core inference + lightweight knowledge base |

### Q: 实时性能如何？/ What is the real-time performance?
A: 在 400MHz Cortex-M7 (STM32H7) 上，CfC 网络（64 隐藏单元）单步推理约 3~8ms（含 sigmoid/exp/tanh 浮点运算）。在 1.5GHz Cortex-A72 (树莓派 4) 上约 0.1~0.5ms。FPGA 硬件加速理论上可达微秒级。
On 400MHz Cortex-M7 (STM32H7), a CfC network (64 hidden units) takes approximately 3~8ms per step (including sigmoid/exp/tanh floating-point operations). On 1.5GHz Cortex-A72 (Raspberry Pi 4), it takes about 0.1~0.5ms. FPGA hardware acceleration can theoretically reach microsecond level.

### Q: FPGA 加速能达到什么效果？/ What acceleration can FPGA achieve?
A: 将 CfC 核心计算单元部署到 FPGA 逻辑中，主要加速矩阵-向量乘法和激活函数计算。理论上，高速 FPGA（如 Zynq UltraScale+）可将推理延迟降低到微秒级别，同时将 CPU 释放出来处理其他任务。
By deploying the CfC core computation unit into FPGA logic, matrix-vector multiplication and activation function computation are accelerated. Theoretically, high-speed FPGAs (e.g., Zynq UltraScale+) can reduce inference latency to microseconds while freeing the CPU for other tasks.

### Q: 如何在不同平台间迁移？/ How to migrate between different platforms?
A: SELF-LNN 的平台抽象层（Platform Abstraction Layer）提供了统一的接口。只需实现 `platform.h` 中声明的函数即可完成平台迁移。
SELF-LNN's Platform Abstraction Layer provides a unified interface. Simply implement the functions declared in `platform.h` to complete platform migration.

## 联系与支持 / Contact and Support

如有问题或建议，请通过以下方式联系我们：
For questions or suggestions, please contact us through:

- **GitHub Issues**：提交问题报告 / Submit issue reports
- **文档反馈**：在文档中反馈 / Provide documentation feedback
- **技术讨论**：参与技术交流 / Join technical discussions

