#ifndef SELFLNN_CORE_QUATERNION_LNN_KERNELS_H
#define SELFLNN_CORE_QUATERNION_LNN_KERNELS_H

/**
 * @file quaternion_lnn_kernels.h
 * @brief 四元数LNN的GPU(OpenCL)内核程序源代码
 *
 * 将原本嵌入在quaternion_lnn.c中的~580行GPU内核字符串提取到独立文件，
 * 便于独立调试、版本管理和多文件共享。
 * 这些内核用于OpenCL GPU后端的四元数前向传播、反向传播和参数更新。
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * 内核1: 四元数前向传播（原始版本，单线程直接循环）
 * 执行Hamilton四元数乘法：hidden = activation(weights * input + biases)
 * ================================================================ */
static const char* QLNN_KERNEL_FORWARD_PROPAGATION =
    "__kernel void quaternion_forward_propagation(__global float* input, __global float* output,\n"
    "                                             __global float* weights, __global float* biases,\n"
    "                                             __global float* hidden_state, uint input_quats,\n"
    "                                             uint hidden_quats, uint output_quats) {\n"
    "    uint hidden_idx = get_global_id(0);\n"
    "    if (hidden_idx >= hidden_quats) return;\n"
    "    float acc_w = 0.0f, acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f;\n"
    "    for (uint input_idx = 0; input_idx < input_quats; input_idx++) {\n"
    "        uint input_base = input_idx * 4;\n"
    "        uint weight_base = (hidden_idx * input_quats + input_idx) * 4;\n"
    "        float in_w = input[input_base + 0];\n"
    "        float in_x = input[input_base + 1];\n"
    "        float in_y = input[input_base + 2];\n"
    "        float in_z = input[input_base + 3];\n"
    "        float w_w = weights[weight_base + 0];\n"
    "        float w_x = weights[weight_base + 1];\n"
    "        float w_y = weights[weight_base + 2];\n"
    "        float w_z = weights[weight_base + 3];\n"
    "        acc_w += w_w * in_w - w_x * in_x - w_y * in_y - w_z * in_z;\n"
    "        acc_x += w_w * in_x + w_x * in_w + w_y * in_z - w_z * in_y;\n"
    "        acc_y += w_w * in_y - w_x * in_z + w_y * in_w + w_z * in_x;\n"
    "        acc_z += w_w * in_z + w_x * in_y - w_y * in_x + w_z * in_w;\n"
    "    }\n"
    "    uint bias_base = hidden_idx * 4;\n"
    "    acc_w += biases[bias_base + 0]; acc_x += biases[bias_base + 1];\n"
    "    acc_y += biases[bias_base + 2]; acc_z += biases[bias_base + 3];\n"
    "    float w = acc_w, x = acc_x, y = acc_y, z = acc_z;\n"
    "    float v_norm = sqrt(x*x + y*y + z*z);\n"
    "    float hidden_w, hidden_x, hidden_y, hidden_z;\n"
    "    if (v_norm < 1e-12f) {\n"
    "        float tanh_w = tanh(w);\n"
    "        hidden_w = tanh_w; hidden_x = 0.0f; hidden_y = 0.0f; hidden_z = 0.0f;\n"
    "    } else {\n"
    "        float tanh_w = tanh(w); float tanh_v = tanh(v_norm);\n"
    "        float vx_n = x / v_norm; float vy_n = y / v_norm; float vz_n = z / v_norm;\n"
    "        float denom = 1.0f + tanh_w * tanh_v;\n"
    "        if (denom < 1e-12f) denom = 1e-12f;\n"
    "        hidden_w = (tanh_w + tanh_v * tanh_w) / denom;\n"
    "        hidden_x = tanh_v * vx_n / denom;\n"
    "        hidden_y = tanh_v * vy_n / denom;\n"
    "        hidden_z = tanh_v * vz_n / denom;\n"
    "    }\n"
    "    uint out_base = hidden_idx * 4;\n"
    "    hidden_state[out_base + 0] = hidden_w; hidden_state[out_base + 1] = hidden_x;\n"
    "    hidden_state[out_base + 2] = hidden_y; hidden_state[out_base + 3] = hidden_z;\n"
    "    if (hidden_idx < output_quats) {\n"
    "        output[out_base + 0] = hidden_w; output[out_base + 1] = hidden_x;\n"
    "        output[out_base + 2] = hidden_y; output[out_base + 3] = hidden_z;\n"
    "    }\n"
    "}\n";

/* ================================================================
 * 内核1b: 四元数前向传播（__local内存分块优化版）
 * 使用工作组共享内存缓存输入数据分块，减少全局内存带宽消耗
 * 工作组大小 = TILE_SIZE，每个线程处理1个hidden_quat
 * ================================================================ */
static const char* QLNN_KERNEL_FORWARD_PROPAGATION_TILED =
    "#define TILE_SIZE 32\n"
    "__kernel void quaternion_forward_tiled(__global float* input, __global float* output,\n"
    "    __global float* weights, __global float* biases,\n"
    "    __global float* hidden_state, uint input_quats,\n"
    "    uint hidden_quats, uint output_quats) {\n"
    "    uint hidden_idx = get_global_id(0);\n"
    "    uint lid = get_local_id(0);\n"
    "    uint group_id = get_group_id(0);\n"
    "    \n"
    "    __local float input_tile[TILE_SIZE * 4];\n"
    "    \n"
    "    float acc_w = 0.0f, acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f;\n"
    "    uint num_tiles = (input_quats + TILE_SIZE - 1) / TILE_SIZE;\n"
    "    \n"
    "    for (uint tile = 0; tile < num_tiles; tile++) {\n"
    "        uint tile_start = tile * TILE_SIZE;\n"
    "        uint tile_end = min(tile_start + TILE_SIZE, input_quats);\n"
    "        uint tile_count = tile_end - tile_start;\n"
    "        \n"
    "        /* 协作加载输入分块到__local内存 */\n"
    "        for (uint i = lid; i < tile_count; i++) {\n"
    "            uint global_offset = (tile_start + i) * 4;\n"
    "            uint local_offset = i * 4;\n"
    "            input_tile[local_offset + 0] = input[global_offset + 0];\n"
    "            input_tile[local_offset + 1] = input[global_offset + 1];\n"
    "            input_tile[local_offset + 2] = input[global_offset + 2];\n"
    "            input_tile[local_offset + 3] = input[global_offset + 3];\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        \n"
    "        /* 使用__local内存计算部分Hamilton乘积 */\n"
    "        if (hidden_idx < hidden_quats) {\n"
    "            for (uint i = 0; i < tile_count; i++) {\n"
    "                uint weight_base = (hidden_idx * input_quats + tile_start + i) * 4;\n"
    "                uint local_base = i * 4;\n"
    "                float in_w = input_tile[local_base + 0];\n"
    "                float in_x = input_tile[local_base + 1];\n"
    "                float in_y = input_tile[local_base + 2];\n"
    "                float in_z = input_tile[local_base + 3];\n"
    "                float w_w = weights[weight_base + 0];\n"
    "                float w_x = weights[weight_base + 1];\n"
    "                float w_y = weights[weight_base + 2];\n"
    "                float w_z = weights[weight_base + 3];\n"
    "                acc_w += w_w * in_w - w_x * in_x - w_y * in_y - w_z * in_z;\n"
    "                acc_x += w_w * in_x + w_x * in_w + w_y * in_z - w_z * in_y;\n"
    "                acc_y += w_w * in_y - w_x * in_z + w_y * in_w + w_z * in_x;\n"
    "                acc_z += w_w * in_z + w_x * in_y - w_y * in_x + w_z * in_w;\n"
    "            }\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    \n"
    "    if (hidden_idx < hidden_quats) {\n"
    "        uint bias_base = hidden_idx * 4;\n"
    "        acc_w += biases[bias_base + 0]; acc_x += biases[bias_base + 1];\n"
    "        acc_y += biases[bias_base + 2]; acc_z += biases[bias_base + 3];\n"
    "        float w = acc_w, x = acc_x, y = acc_y, z = acc_z;\n"
    "        float v_norm = sqrt(x*x + y*y + z*z);\n"
    "        float hw, hx, hy, hz;\n"
    "        if (v_norm < 1e-12f) {\n"
    "            float tw = tanh(w);\n"
    "            hw = tw; hx = 0.0f; hy = 0.0f; hz = 0.0f;\n"
    "        } else {\n"
    "            float tw = tanh(w); float tv = tanh(v_norm);\n"
    "            float denom = 1.0f + tw * tv;\n"
    "            if (denom < 1e-12f) denom = 1e-12f;\n"
    "            hw = (tw + tv * tw) / denom;\n"
    "            float vni = 1.0f / (v_norm * denom);\n"
    "            hx = tv * x * vni; hy = tv * y * vni; hz = tv * z * vni;\n"
    "        }\n"
    "        uint out_base = hidden_idx * 4;\n"
    "        hidden_state[out_base+0]=hw; hidden_state[out_base+1]=hx;\n"
    "        hidden_state[out_base+2]=hy; hidden_state[out_base+3]=hz;\n"
    "        if (hidden_idx < output_quats) {\n"
    "            output[out_base+0]=hw; output[out_base+1]=hx;\n"
    "            output[out_base+2]=hy; output[out_base+3]=hz;\n"
    "        }\n"
    "    }\n"
    "}\n";

/* ================================================================
 * 内核2: 输出梯度计算
 * dL/dO = 2*(O - T) / output_quats (MSE损失梯度)
 * ================================================================ */
static const char* QLNN_KERNEL_OUTPUT_GRADIENT =
    "__kernel void quaternion_output_gradient(__global float* output, __global float* target,\n"
    "                                          __global float* output_grad, uint output_quats) {\n"
    "    uint idx = get_global_id(0);\n"
    "    if (idx >= output_quats) return;\n"
    "    uint base = idx * 4;\n"
    "    float diff_w = output[base + 0] - target[base + 0];\n"
    "    float diff_x = output[base + 1] - target[base + 1];\n"
    "    float diff_y = output[base + 2] - target[base + 2];\n"
    "    float diff_z = output[base + 3] - target[base + 3];\n"
    "    float scale = 2.0f / (float)output_quats;\n"
    "    output_grad[base + 0] = diff_w * scale;\n"
    "    output_grad[base + 1] = diff_x * scale;\n"
    "    output_grad[base + 2] = diff_y * scale;\n"
    "    output_grad[base + 3] = diff_z * scale;\n"
    "}\n";

/* ================================================================
 * 内核3: 隐藏层梯度反向传播
 * grad_hidden = W_out^T * output_grad ⊙ tanh'(hidden)
 * ================================================================ */
static const char* QLNN_KERNEL_HIDDEN_GRADIENT =
    "__kernel void quaternion_hidden_gradient(__global float* hidden_state,\n"
    "    __global float* output_grad, __global float* weights,\n"
    "    __global float* hidden_grad, uint hidden_quats, uint output_quats) {\n"
    "    uint h_idx = get_global_id(0);\n"
    "    if (h_idx >= hidden_quats) return;\n"
    "    uint h_base = h_idx * 4;\n"
    "    float grad_w = 0.0f, grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;\n"
    "    for (uint o_idx = 0; o_idx < output_quats; o_idx++) {\n"
    "        uint o_base = o_idx * 4;\n"
    "        uint wgt_base = (o_idx * hidden_quats + h_idx) * 4;\n"
    "        float og_w = output_grad[o_base + 0];\n"
    "        float og_x = output_grad[o_base + 1];\n"
    "        float og_y = output_grad[o_base + 2];\n"
    "        float og_z = output_grad[o_base + 3];\n"
    "        float w_w = weights[wgt_base + 0];\n"
    "        float w_x = weights[wgt_base + 1];\n"
    "        float w_y = weights[wgt_base + 2];\n"
    "        float w_z = weights[wgt_base + 3];\n"
    "        grad_w += w_w*og_w - w_x*og_x - w_y*og_y - w_z*og_z;\n"
    "        grad_x += w_w*og_x + w_x*og_w + w_y*og_z - w_z*og_y;\n"
    "        grad_y += w_w*og_y - w_x*og_z + w_y*og_w + w_z*og_x;\n"
    "        grad_z += w_w*og_z + w_x*og_y - w_y*og_x + w_z*og_w;\n"
    "    }\n"
    "    float hw=hidden_state[h_base+0],hx=hidden_state[h_base+1];\n"
    "    float hy=hidden_state[h_base+2],hz=hidden_state[h_base+3];\n"
    "    float deriv = 1.0f - (hw*hw+hx*hx+hy*hy+hz*hz);\n"
    "    if (deriv < 0.0f) deriv = 0.0f;\n"
    "    hidden_grad[h_base+0] = grad_w * deriv;\n"
    "    hidden_grad[h_base+1] = grad_x * deriv;\n"
    "    hidden_grad[h_base+2] = grad_y * deriv;\n"
    "    hidden_grad[h_base+3] = grad_z * deriv;\n"
    "}\n";

/* ================================================================
 * 内核4: 权重梯度计算
 * dL/dW = hidden_grad ⊗ input (四元数外积)
 * ================================================================ */
static const char* QLNN_KERNEL_WEIGHT_GRADIENT =
    "__kernel void quaternion_weight_gradient(__global float* output_grad,\n"
    "    __global float* hidden_state, __global float* input_data,\n"
    "    __global float* weight_grad, uint output_quats, uint hidden_quats,\n"
    "    uint input_quats) {\n"
    "    uint gid = get_global_id(0);\n"
    "    uint total = output_quats * hidden_quats + hidden_quats * input_quats;\n"
    "    if (gid >= total) return;\n"
    "    uint base = gid * 4;\n"
    "    float gw=0.0f,gx=0.0f,gy=0.0f,gz=0.0f;\n"
    "    if (gid < output_quats * hidden_quats) {\n"
    "        uint o_idx = gid / hidden_quats;\n"
    "        uint h_idx = gid % hidden_quats;\n"
    "        uint og_base = o_idx * 4;\n"
    "        uint hs_base = h_idx * 4;\n"
    "        float og_w=output_grad[og_base+0],og_x=output_grad[og_base+1];\n"
    "        float og_y=output_grad[og_base+2],og_z=output_grad[og_base+3];\n"
    "        float hs_w=hidden_state[hs_base+0],hs_x=hidden_state[hs_base+1];\n"
    "        float hs_y=hidden_state[hs_base+2],hs_z=hidden_state[hs_base+3];\n"
    "        gw=og_w*hs_w-og_x*hs_x-og_y*hs_y-og_z*hs_z;\n"
    "        gx=og_w*hs_x+og_x*hs_w+og_y*hs_z-og_z*hs_y;\n"
    "        gy=og_w*hs_y-og_x*hs_z+og_y*hs_w+og_z*hs_x;\n"
    "        gz=og_w*hs_z+og_x*hs_y-og_y*hs_x+og_z*hs_w;\n"
    "    } else {\n"
    "        uint idx = gid - output_quats * hidden_quats;\n"
    "        uint h_idx = idx / input_quats;\n"
    "        uint in_idx = idx % input_quats;\n"
    "        uint hg_base = h_idx * 4, in_base = in_idx * 4;\n"
    "        float hg_w=output_grad[hg_base+0],hg_x=output_grad[hg_base+1];\n"
    "        float hg_y=output_grad[hg_base+2],hg_z=output_grad[hg_base+3];\n"
    "        float in_w=input_data[in_base+0],in_x=input_data[in_base+1];\n"
    "        float in_y=input_data[in_base+2],in_z=input_data[in_base+3];\n"
    "        gw=hg_w*in_w-hg_x*in_x-hg_y*in_y-hg_z*in_z;\n"
    "        gx=hg_w*in_x+hg_x*in_w+hg_y*in_z-hg_z*in_y;\n"
    "        gy=hg_w*in_y-hg_x*in_z+hg_y*in_w+hg_z*in_x;\n"
    "        gz=hg_w*in_z+hg_x*in_y-hg_y*in_x+hg_z*in_w;\n"
    "    }\n"
    "    weight_grad[base+0]=gw; weight_grad[base+1]=gx;\n"
    "    weight_grad[base+2]=gy; weight_grad[base+3]=gz;\n"
    "}\n";

/* ================================================================
 * 内核5: 参数更新（SGD with 梯度裁剪）
 * ================================================================ */
static const char* QLNN_KERNEL_PARAMETER_UPDATE =
    "__kernel void quaternion_parameter_update(__global float* weights,\n"
    "    __global float* biases, __global float* weight_grad, __global float* bias_grad,\n"
    "    uint weight_count, uint bias_count, float learning_rate) {\n"
    "    uint idx = get_global_id(0);\n"
    "    uint total = weight_count + bias_count;\n"
    "    if (idx >= total) return;\n"
    "    float lr = learning_rate;\n"
    "    if (idx < weight_count) {\n"
    "        uint base = idx * 4;\n"
    "        float gw=weight_grad[base+0],gx=weight_grad[base+1];\n"
    "        float gy=weight_grad[base+2],gz=weight_grad[base+3];\n"
    "        float gn=sqrt(gw*gw+gx*gx+gy*gy+gz*gz);\n"
    "        if(gn>5.0f){float s=5.0f/gn;gw*=s;gx*=s;gy*=s;gz*=s;}\n"
    "        weights[base+0]-=lr*gw; weights[base+1]-=lr*gx;\n"
    "        weights[base+2]-=lr*gy; weights[base+3]-=lr*gz;\n"
    "    } else {\n"
    "        uint base=(idx-weight_count)*4;\n"
    "        float gw=bias_grad[base+0],gx=bias_grad[base+1];\n"
    "        float gy=bias_grad[base+2],gz=bias_grad[base+3];\n"
    "        float gn=sqrt(gw*gw+gx*gx+gy*gy+gz*gz);\n"
    "        if(gn>0.5f){float s=0.5f/gn;gw*=s;gx*=s;gy*=s;gz*=s;}\n"
    "        biases[base+0]-=lr*0.1f*gw; biases[base+1]-=lr*0.1f*gx;\n"
    "        biases[base+2]-=lr*0.1f*gy; biases[base+3]-=lr*0.1f*gz;\n"
    "    }\n"
    "}\n";

/* ================================================================
 * 内核6: 偏置梯度计算
 * ================================================================ */
static const char* QLNN_KERNEL_BIAS_GRADIENT =
    "__kernel void quaternion_bias_gradient(__global float* hidden_grad,\n"
    "    __global float* output_grad, __global float* bias_grad,\n"
    "    uint hidden_quats, uint output_quats) {\n"
    "    uint idx = get_global_id(0);\n"
    "    uint total = hidden_quats + output_quats;\n"
    "    if (idx >= total) return;\n"
    "    uint base = idx * 4;\n"
    "    if (idx < hidden_quats) {\n"
    "        bias_grad[base+0]=hidden_grad[base+0];\n"
    "        bias_grad[base+1]=hidden_grad[base+1];\n"
    "        bias_grad[base+2]=hidden_grad[base+2];\n"
    "        bias_grad[base+3]=hidden_grad[base+3];\n"
    "    } else {\n"
    "        uint o_idx=idx-hidden_quats; uint ob=o_idx*4;\n"
    "        bias_grad[base+0]=output_grad[ob+0];\n"
    "        bias_grad[base+1]=output_grad[ob+1];\n"
    "        bias_grad[base+2]=output_grad[ob+2];\n"
    "        bias_grad[base+3]=output_grad[ob+3];\n"
    "    }\n"
    "}\n";

/* ================================================================
 * 内核7: 权重更新（含梯度裁剪）
 * ================================================================ */
static const char* QLNN_KERNEL_WEIGHT_UPDATE =
    "__kernel void quaternion_weight_update(__global float* weights,\n"
    "    __global float* biases, __global float* weight_grad,\n"
    "    __global float* bias_grad, uint weight_count, uint bias_count,\n"
    "    float learning_rate) {\n"
    "    uint idx = get_global_id(0);\n"
    "    uint total = weight_count + bias_count;\n"
    "    if (idx >= total) return;\n"
    "    float lr = learning_rate;\n"
    "    if (idx < weight_count) {\n"
    "        uint base = idx * 4;\n"
    "        float gw=weight_grad[base+0],gx=weight_grad[base+1];\n"
    "        float gy=weight_grad[base+2],gz=weight_grad[base+3];\n"
    "        float gn=sqrt(gw*gw+gx*gx+gy*gy+gz*gz);\n"
    "        if(gn>5.0f){float s=5.0f/gn;gw*=s;gx*=s;gy*=s;gz*=s;}\n"
    "        weights[base+0]-=lr*gw; weights[base+1]-=lr*gx;\n"
    "        weights[base+2]-=lr*gy; weights[base+3]-=lr*gz;\n"
    "    } else {\n"
    "        uint base=(idx-weight_count)*4;\n"
    "        float gw=bias_grad[base+0],gx=bias_grad[base+1];\n"
    "        float gy=bias_grad[base+2],gz=bias_grad[base+3];\n"
    "        float gn=sqrt(gw*gw+gx*gx+gy*gy+gz*gz);\n"
    "        if(gn>0.5f){float s=0.5f/gn;gw*=s;gx*=s;gy*=s;gz*=s;}\n"
    "        biases[base+0]-=lr*0.1f*gw; biases[base+1]-=lr*0.1f*gx;\n"
    "        biases[base+2]-=lr*0.1f*gy; biases[base+3]-=lr*0.1f*gz;\n"
    "    }\n"
    "}\n";

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_CORE_QUATERNION_LNN_KERNELS_H */
