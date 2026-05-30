/**
 * @file laplace_operator.c
 * @brief Laplace神经算子实现
 *
 * 实现基于傅里叶神经算子(FNO)框架的谱域算子学习算法。
 * 包含FFT、谱卷积、升维/降维投影、Adam优化等完整实现。
 * 100%纯C实现，无外部依赖。
 */

#include "selflnn/reasoning/laplace_operator.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========== 内部工具函数 ========== */

/**
 * @brief GELU激活函数: x * 0.5 * (1 + erf(x / sqrt(2)))
 */
static double gelu_activation(double x) {
    return 0.5 * x * (1.0 + erf(x / M_SQRT2));
}

/**
 * @brief GELU导数
 */
static double gelu_derivative(double x) {
    double cdf = 0.5 * (1.0 + erf(x / M_SQRT2));
    double pdf = exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
    return cdf + x * pdf;
}

/**
 * @brief ReLU激活函数
 */
static double relu_activation(double x) {
    return x > 0.0 ? x : 0.0;
}

/**
 * @brief ReLU导数
 */
static double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

/**
 * @brief SiLU激活函数: x * sigmoid(x)
 */
static double silu_activation(double x) {
    double sig = 1.0 / (1.0 + exp(-x));
    return x * sig;
}

/**
 * @brief SiLU导数
 */
static double silu_derivative(double x) {
    double sig = 1.0 / (1.0 + exp(-x));
    return sig + x * sig * (1.0 - sig);
}

/**
 * @brief 应用激活函数
 */
static double apply_activation(double x, LNOActivationType act) {
    switch (act) {
    case LNO_ACTIVATION_GELU: return gelu_activation(x);
    case LNO_ACTIVATION_RELU: return relu_activation(x);
    case LNO_ACTIVATION_SILU: return silu_activation(x);
    case LNO_ACTIVATION_TANH: return tanh(x);
    default: return x;
    }
}

/**
 * @brief 激活函数导数
 */
static double activation_derivative(double x, LNOActivationType act) {
    switch (act) {
    case LNO_ACTIVATION_GELU: return gelu_derivative(x);
    case LNO_ACTIVATION_RELU: return relu_derivative(x);
    case LNO_ACTIVATION_SILU: return silu_derivative(x);
    case LNO_ACTIVATION_TANH: return 1.0 - tanh(x) * tanh(x);
    default: return 1.0;
    }
}

/**
 * @brief 判断整数是否为2的幂
 */
static int is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief 计算大于等于n的最小2的幂
 */
static size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/**
 * @brief 位反转置换
 */
static void bit_reverse(LaplaceComplex* data, size_t n) {
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < j) {
            LaplaceComplex tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
        size_t mask = n >> 1;
        while (j & mask) {
            j ^= mask;
            mask >>= 1;
        }
        j ^= mask;
    }
}

/* ========== FFT实现(Cooley-Tukey radix-2) ========== */

/**
 * @brief 一维复数FFT(原地)
 *
 * @param data 复数数组 [n]，输入时为空域信号，输出时为频域信号
 * @param n FFT点数，必须为2的幂
 * @param inverse 是否为逆变换(1=逆变换, 0=正变换)
 */
static void fft_1d(LaplaceComplex* data, size_t n, int inverse) {
    if (!data || n == 0 || !is_power_of_two(n)) return;

    bit_reverse(data, n);

    double sign = inverse ? 1.0 : -1.0;

    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = sign * 2.0 * M_PI / (double)len;
        LaplaceComplex wlen = {cos(angle), sin(angle)};

        for (size_t i = 0; i < n; i += len) {
            LaplaceComplex w = {1.0, 0.0};
            size_t half = len >> 1;
            for (size_t j = 0; j < half; j++) {
                LaplaceComplex u = data[i + j];
                LaplaceComplex v = {
                    w.real * data[i + j + half].real - w.imag * data[i + j + half].imag,
                    w.real * data[i + j + half].imag + w.imag * data[i + j + half].real
                };
                data[i + j].real = u.real + v.real;
                data[i + j].imag = u.imag + v.imag;
                data[i + j + half].real = u.real - v.real;
                data[i + j + half].imag = u.imag - v.imag;

                double w_new_real = w.real * wlen.real - w.imag * wlen.imag;
                double w_new_imag = w.real * wlen.imag + w.imag * wlen.real;
                w.real = w_new_real;
                w.imag = w_new_imag;
            }
        }
    }

    if (inverse) {
        double inv_n = 1.0 / (double)n;
        for (size_t i = 0; i < n; i++) {
            data[i].real *= inv_n;
            data[i].imag *= inv_n;
        }
    }
}

/**
 * @brief 实值FFT(rfft): 将实值信号变换到频域，只保留正频率
 *
 * @param input 实值输入 [n]
 * @param n 输入长度
 * @param output 频域输出(复数) [n/2+1]
 */
static void rfft_1d(const double* input, size_t n, LaplaceComplex* output) {
    size_t n2 = next_power_of_two(n);
    LaplaceComplex* buf = (LaplaceComplex*)safe_malloc(n2 * sizeof(LaplaceComplex));
    if (!buf) return;

    for (size_t i = 0; i < n; i++) {
        buf[i].real = input[i];
        buf[i].imag = 0.0;
    }
    for (size_t i = n; i < n2; i++) {
        buf[i].real = 0.0;
        buf[i].imag = 0.0;
    }

    fft_1d(buf, n2, 0);

    size_t out_size = n / 2 + 1;
    for (size_t i = 0; i < out_size; i++) {
        output[i] = buf[i];
    }

    safe_free((void**)&buf);
}

/**
 * @brief 逆实值FFT(irfft): 从频域恢复实值信号
 *
 * @param input 频域输入(复数) [n/2+1]
 * @param n 输出长度
 * @param output 实值输出 [n]
 */
static void irfft_1d(const LaplaceComplex* input, size_t n, double* output) {
    size_t n2 = next_power_of_two(n);
    LaplaceComplex* buf = (LaplaceComplex*)safe_malloc(n2 * sizeof(LaplaceComplex));
    if (!buf) return;

    size_t half = n / 2 + 1;
    for (size_t i = 0; i < half; i++) {
        buf[i] = input[i];
    }
    for (size_t i = 0; i < half - 2; i++) {
        size_t sym = n2 - 1 - i;
        buf[sym].real = input[half - 2 - i].real;
        buf[sym].imag = -input[half - 2 - i].imag;
    }
    if (n2 >= 1) buf[n2 - 1] = input[0];
    if (n2 >= 2) buf[n2 / 2] = input[half - 1];

    fft_1d(buf, n2, 1);

    for (size_t i = 0; i < n; i++) {
        output[i] = buf[i].real;
    }

    safe_free((void**)&buf);
}

/**
 * @brief 2D FFT(原地)
 */
static void fft_2d(LaplaceComplex* data, size_t n1, size_t n2, int inverse) {
    /* 每行做FFT */
    for (size_t i = 0; i < n1; i++) {
        fft_1d(&data[i * n2], n2, inverse);
    }
    /* 每列做FFT */
    LaplaceComplex* col = (LaplaceComplex*)safe_malloc(n1 * sizeof(LaplaceComplex));
    if (!col) return;
    for (size_t j = 0; j < n2; j++) {
        for (size_t i = 0; i < n1; i++) {
            col[i] = data[i * n2 + j];
        }
        fft_1d(col, n1, inverse);
        for (size_t i = 0; i < n1; i++) {
            data[i * n2 + j] = col[i];
        }
    }
    safe_free((void**)&col);
}

/* ========== 谱卷积层 ========== */

/**
 * @brief 初始化1D谱卷积层
 */
static int spectral_conv1d_init(LaplaceSpectralConv1D* layer,
                                size_t in_ch, size_t out_ch,
                                size_t modes_in, size_t modes_out) {
    if (!layer) return -1;
    layer->in_channels = in_ch;
    layer->out_channels = out_ch;
    layer->modes_in = modes_in;
    layer->modes_out = modes_out;

    size_t weight_count = out_ch * in_ch * modes_out;
    layer->weights = (LaplaceComplex*)safe_malloc(weight_count * sizeof(LaplaceComplex));
    if (!layer->weights) return -1;

    /* Xavier初始化 */
    double scale = sqrt(2.0 / (double)(in_ch * modes_in + out_ch * modes_out));
    for (size_t i = 0; i < weight_count; i++) {
        double u1 = (double)secure_random_float();
        double u2 = (double)secure_random_float();
        double r = scale * sqrt(-2.0 * log(u1 + 1e-10));
        double theta = 2.0 * M_PI * u2;
        layer->weights[i].real = r * cos(theta);
        layer->weights[i].imag = r * sin(theta);
    }

    layer->bias = (double*)safe_malloc(out_ch * sizeof(double));
    if (!layer->bias) { safe_free((void**)&layer->weights); return -1; }
    memset(layer->bias, 0, out_ch * sizeof(double));

    return 0;
}

/**
 * @brief 1D谱卷积前向: 在傅里叶域应用可学习核
 *
 * 对每个输入通道做rfft，截断到modes_out个模态，
 * 与可学习复数权重相乘，求和所有输入通道，加偏置，irfft回空域
 */
static int spectral_conv1d_forward(const LaplaceSpectralConv1D* layer,
                                   const double* input,
                                   size_t batch, size_t grid_size,
                                   size_t in_ch, double* output) {
    if (!layer || !input || !output) return -1;

    size_t half_modes = layer->modes_out;
    LaplaceComplex* freq_buffer = (LaplaceComplex*)safe_malloc(
        (half_modes + 1) * sizeof(LaplaceComplex));
    LaplaceComplex* temp_spectral = (LaplaceComplex*)safe_malloc(
        in_ch * (half_modes + 1) * sizeof(LaplaceComplex));
    if (!freq_buffer || !temp_spectral) {
        safe_free((void**)&freq_buffer);
        safe_free((void**)&temp_spectral);
        return -1;
    }

    for (size_t b = 0; b < batch; b++) {
        /* 对每个输入通道计算rfft */
        for (size_t c = 0; c < in_ch; c++) {
            size_t in_idx = b * in_ch * grid_size + c * grid_size;
            rfft_1d(&input[in_idx], grid_size, freq_buffer);
            for (size_t m = 0; m < half_modes; m++) {
                temp_spectral[c * (half_modes + 1) + m] = freq_buffer[m];
            }
        }

        /* 应用可学习权重: 对每个输出通道 */
        for (size_t oc = 0; oc < layer->out_channels; oc++) {
            for (size_t m = 0; m < half_modes; m++) {
                double sum_real = 0.0, sum_imag = 0.0;
                for (size_t ic = 0; ic < in_ch; ic++) {
                    size_t w_idx = oc * in_ch * layer->modes_out + ic * layer->modes_out + m;
                    size_t f_idx = ic * (half_modes + 1) + m;
                    double w_r = layer->weights[w_idx].real;
                    double w_i = layer->weights[w_idx].imag;
                    double f_r = temp_spectral[f_idx].real;
                    double f_i = temp_spectral[f_idx].imag;
                    sum_real += w_r * f_r - w_i * f_i;
                    sum_imag += w_r * f_i + w_i * f_r;
                }
                freq_buffer[m].real = sum_real;
                freq_buffer[m].imag = sum_imag;
            }

            irfft_1d(freq_buffer, grid_size, &output[b * layer->out_channels * grid_size + oc * grid_size]);
            for (size_t i = 0; i < grid_size; i++) {
                output[b * layer->out_channels * grid_size + oc * grid_size + i] += layer->bias[oc];
            }
        }
    }

    safe_free((void**)&freq_buffer);
    safe_free((void**)&temp_spectral);
    return 0;
}

/**
 * @brief 初始化2D谱卷积层
 */
static int spectral_conv2d_init(LaplaceSpectralConv2D* layer,
                                size_t in_ch, size_t out_ch,
                                size_t m1_in, size_t m2_in,
                                size_t m1_out, size_t m2_out) {
    if (!layer) return -1;
    layer->in_channels = in_ch;
    layer->out_channels = out_ch;
    layer->modes1_in = m1_in;
    layer->modes2_in = m2_in;
    layer->modes1_out = m1_out;
    layer->modes2_out = m2_out;

    size_t weight_count = out_ch * in_ch * m1_out * m2_out;
    layer->weights = (LaplaceComplex*)safe_malloc(weight_count * sizeof(LaplaceComplex));
    if (!layer->weights) return -1;

    double scale = sqrt(2.0 / (double)(in_ch * m1_in * m2_in + out_ch * m1_out * m2_out));
    for (size_t i = 0; i < weight_count; i++) {
        double u1 = (double)secure_random_float();
        double u2 = (double)secure_random_float();
        double r = scale * sqrt(-2.0 * log(u1 + 1e-10));
        double theta = 2.0 * M_PI * u2;
        layer->weights[i].real = r * cos(theta);
        layer->weights[i].imag = r * sin(theta);
    }

    layer->bias = (double*)safe_malloc(out_ch * sizeof(double));
    if (!layer->bias) { safe_free((void**)&layer->weights); return -1; }
    memset(layer->bias, 0, out_ch * sizeof(double));

    return 0;
}

/**
 * @brief 2D谱卷积前向
 */
static int spectral_conv2d_forward(const LaplaceSpectralConv2D* layer,
                                   const double* input,
                                   size_t batch, size_t grid_size,
                                   size_t in_ch, double* output) {
    if (!layer || !input || !output) return -1;

    LaplaceComplex* freq_2d = (LaplaceComplex*)safe_malloc(
        grid_size * grid_size * sizeof(LaplaceComplex));
    LaplaceComplex* temp_spectral = (LaplaceComplex*)safe_malloc(
        in_ch * grid_size * grid_size * sizeof(LaplaceComplex));
    if (!freq_2d || !temp_spectral) {
        safe_free((void**)&freq_2d);
        safe_free((void**)&temp_spectral);
        return -1;
    }

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < in_ch; c++) {
            size_t in_off = b * in_ch * grid_size * grid_size + c * grid_size * grid_size;
            for (size_t i = 0; i < grid_size * grid_size; i++) {
                freq_2d[i].real = input[in_off + i];
                freq_2d[i].imag = 0.0;
            }
            fft_2d(freq_2d, grid_size, grid_size, 0);
            for (size_t i = 0; i < grid_size * grid_size; i++) {
                temp_spectral[c * grid_size * grid_size + i] = freq_2d[i];
            }
        }

        size_t m1 = layer->modes1_out;
        size_t m2 = layer->modes2_out;

        for (size_t oc = 0; oc < layer->out_channels; oc++) {
            LaplaceComplex* out_spec = (LaplaceComplex*)safe_malloc(
                grid_size * grid_size * sizeof(LaplaceComplex));
            if (!out_spec) { safe_free((void**)&freq_2d); safe_free((void**)&temp_spectral); return -1; }
            memset(out_spec, 0, grid_size * grid_size * sizeof(LaplaceComplex));

            for (size_t ic = 0; ic < in_ch; ic++) {
                for (size_t j1 = 0; j1 < m1; j1++) {
                    for (size_t j2 = 0; j2 < m2; j2++) {
                        size_t w_idx = oc * in_ch * m1 * m2 + ic * m1 * m2 + j1 * m2 + j2;
                        size_t f_idx = ic * grid_size * grid_size + j1 * grid_size + j2;
                        out_spec[j1 * grid_size + j2].real +=
                            layer->weights[w_idx].real * temp_spectral[f_idx].real -
                            layer->weights[w_idx].imag * temp_spectral[f_idx].imag;
                        out_spec[j1 * grid_size + j2].imag +=
                            layer->weights[w_idx].real * temp_spectral[f_idx].imag +
                            layer->weights[w_idx].imag * temp_spectral[f_idx].real;
                    }
                }
            }

            for (size_t j = 0; j < grid_size * grid_size; j++) {
                out_spec[j].imag = 0.0;
            }

            fft_2d(out_spec, grid_size, grid_size, 1);

            size_t out_off = b * layer->out_channels * grid_size * grid_size + oc * grid_size * grid_size;
            for (size_t i = 0; i < grid_size * grid_size; i++) {
                output[out_off + i] = out_spec[i].real + layer->bias[oc];
            }

            safe_free((void**)&out_spec);
        }
    }

    safe_free((void**)&freq_2d);
    safe_free((void**)&temp_spectral);
    return 0;
}

/* ========== 线性层(升维/降维) ========== */

/**
 * @brief 线性变换前向: y = W^T * x + b (沿通道维度)
 *
 * @param input [batch][in_ch][spatial] 
 * @param output [batch][out_ch][spatial]
 * @param weight [in_ch][out_ch]
 * @param bias [out_ch]
 */
static void linear_layer_forward(const double* input, double* output,
                                 const double* weight, const double* bias,
                                 size_t batch, size_t in_ch, size_t out_ch,
                                 size_t spatial_dim) {
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < spatial_dim; s++) {
            for (size_t oc = 0; oc < out_ch; oc++) {
                double sum = bias ? bias[oc] : 0.0;
                for (size_t ic = 0; ic < in_ch; ic++) {
                    sum += input[b * in_ch * spatial_dim + ic * spatial_dim + s] *
                           weight[ic * out_ch + oc];
                }
                output[b * out_ch * spatial_dim + oc * spatial_dim + s] = sum;
            }
        }
    }
}

/* ========== 批量归一化 ========== */

/**
 * @brief 批量归一化前向 (推理模式)
 */
static void batch_norm_forward(const double* input, double* output,
                               double gamma, double beta,
                               double running_mean, double running_var,
                               size_t batch, size_t spatial_dim) {
    double epsilon = 1e-5;
    double std = sqrt(running_var + epsilon);
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < spatial_dim; s++) {
            size_t idx = b * spatial_dim + s;
            output[idx] = gamma * (input[idx] - running_mean) / std + beta;
        }
    }
}

/* ========== 解析梯度反向传播辅助函数 ========== */

/**
 * @brief 线性层反向传播
 *
 * 计算 dL/dW、dL/db 和 dL/dx
 * W布局: weight[ic * out_ch + oc]，即 [in_ch][out_ch]
 *
 * @param x 输入 [batch × in_ch × spatial]
 * @param dy 输出梯度 [batch × out_ch × spatial]
 * @param batch 批量大小
 * @param in_ch 输入通道数
 * @param out_ch 输出通道数
 * @param spatial 空间维度大小
 * @param dW 权重梯度输出 [in_ch × out_ch]，可为NULL
 * @param db 偏置梯度输出 [out_ch]，可为NULL
 * @param dx 输入梯度输出 [batch × in_ch × spatial]，可为NULL
 */
static void linear_layer_backward(const double* x, const double* dy,
                                  const double* weight,
                                  size_t batch, size_t in_ch, size_t out_ch,
                                  size_t spatial,
                                  double* dW, double* db, double* dx) {
    if (dW) {
        memset(dW, 0, in_ch * out_ch * sizeof(double));
        for (size_t b = 0; b < batch; b++) {
            for (size_t s = 0; s < spatial; s++) {
                for (size_t oc = 0; oc < out_ch; oc++) {
                    double dy_val = dy[b * out_ch * spatial + oc * spatial + s];
                    for (size_t ic = 0; ic < in_ch; ic++) {
                        double x_val = x[b * in_ch * spatial + ic * spatial + s];
                        dW[ic * out_ch + oc] += x_val * dy_val;
                    }
                }
            }
        }
    }
    if (db) {
        memset(db, 0, out_ch * sizeof(double));
        for (size_t b = 0; b < batch; b++) {
            for (size_t s = 0; s < spatial; s++) {
                for (size_t oc = 0; oc < out_ch; oc++) {
                    db[oc] += dy[b * out_ch * spatial + oc * spatial + s];
                }
            }
        }
    }
    if (dx && weight) {
        memset(dx, 0, batch * in_ch * spatial * sizeof(double));
        for (size_t b = 0; b < batch; b++) {
            for (size_t s = 0; s < spatial; s++) {
                for (size_t ic = 0; ic < in_ch; ic++) {
                    double sum = 0.0;
                    for (size_t oc = 0; oc < out_ch; oc++) {
                        sum += dy[b * out_ch * spatial + oc * spatial + s] *
                               weight[ic * out_ch + oc];
                    }
                    dx[b * in_ch * spatial + ic * spatial + s] = sum;
                }
            }
        }
    }
}

/**
 * @brief 1D谱卷积层反向传播
 *
 * 计算谱权重/偏置梯度及输入梯度
 *
 * @param layer 谱卷积层
 * @param x 输入 [batch × in_ch × grid]
 * @param dy 输出梯度 [batch × out_ch × grid]
 * @param batch 批量大小
 * @param grid_size 网格大小
 * @param in_ch 输入通道数
 * @param dW 权重梯度输出 [out_ch × in_ch × modes_out × 2 (实+虚)]，可为NULL
 * @param db 偏置梯度输出 [out_ch]，可为NULL
 * @param dx 输入梯度输出 [batch × in_ch × grid]，可为NULL
 */
static void spectral_conv1d_backward(const LaplaceSpectralConv1D* layer,
                                     const double* x, const double* dy,
                                     size_t batch, size_t grid_size,
                                     size_t in_ch,
                                     double* dW, double* db, double* dx) {
    size_t half_modes = grid_size / 2;
    size_t modes_out = layer->modes_out;
    size_t out_ch = layer->out_channels;

    /* 分配频域工作缓冲区 */
    LaplaceComplex* freq_x = (LaplaceComplex*)safe_malloc(
        (half_modes + 1) * sizeof(LaplaceComplex));
    LaplaceComplex* freq_dy = (LaplaceComplex*)safe_malloc(
        (half_modes + 1) * sizeof(LaplaceComplex));
    LaplaceComplex* temp_x = (LaplaceComplex*)safe_malloc(
        in_ch * (half_modes + 1) * sizeof(LaplaceComplex));
    LaplaceComplex* temp_dy = (LaplaceComplex*)safe_malloc(
        out_ch * (half_modes + 1) * sizeof(LaplaceComplex));
    if (!freq_x || !freq_dy || !temp_x || !temp_dy) {
        safe_free((void**)&freq_x); safe_free((void**)&freq_dy);
        safe_free((void**)&temp_x); safe_free((void**)&temp_dy);
        return;
    }

    if (dW) memset(dW, 0, out_ch * in_ch * modes_out * 2 * sizeof(double));
    if (db) {
        memset(db, 0, out_ch * sizeof(double));
        for (size_t b = 0; b < batch; b++)
            for (size_t oc = 0; oc < out_ch; oc++)
                for (size_t i = 0; i < grid_size; i++)
                    db[oc] += dy[b * out_ch * grid_size + oc * grid_size + i];
    }

    if (dx) memset(dx, 0, batch * in_ch * grid_size * sizeof(double));

    for (size_t b = 0; b < batch; b++) {
        /* 对输入做rfft */
        for (size_t ic = 0; ic < in_ch; ic++) {
            size_t in_off = b * in_ch * grid_size + ic * grid_size;
            rfft_1d(&x[in_off], grid_size, freq_x);
            for (size_t m = 0; m <= half_modes; m++)
                temp_x[ic * (half_modes + 1) + m] = freq_x[m];
        }

        /* 对dy做rfft */
        for (size_t oc = 0; oc < out_ch; oc++) {
            size_t dy_off = b * out_ch * grid_size + oc * grid_size;
            rfft_1d(&dy[dy_off], grid_size, freq_dy);
            for (size_t m = 0; m <= half_modes; m++)
                temp_dy[oc * (half_modes + 1) + m] = freq_dy[m];
        }

        /* 计算权重梯度 dL/dW = conj(rfft(x)) * rfft(dL/dy) (复数乘法，累加batch) */
        if (dW) {
            for (size_t oc = 0; oc < out_ch; oc++) {
                for (size_t ic = 0; ic < in_ch; ic++) {
                    for (size_t m = 0; m < modes_out; m++) {
                        double x_r = temp_x[ic * (half_modes + 1) + m].real;
                        double x_i = temp_x[ic * (half_modes + 1) + m].imag;
                        double dy_r = temp_dy[oc * (half_modes + 1) + m].real;
                        double dy_i = temp_dy[oc * (half_modes + 1) + m].imag;
                        size_t w_idx = (oc * in_ch * modes_out + ic * modes_out + m) * 2;
                        dW[w_idx]     += x_r * dy_r + x_i * dy_i;
                        dW[w_idx + 1] += x_r * dy_i - x_i * dy_r;
                    }
                }
            }
        }

        /* 计算输入梯度 dL/dx = irfft(sum_oc conj(W) * rfft(dL/dy)) */
        if (dx) {
            LaplaceComplex* dx_hat = freq_x;
            for (size_t ic = 0; ic < in_ch; ic++) {
                for (size_t m = 0; m <= half_modes; m++) {
                    double sum_r = 0.0, sum_i = 0.0;
                    for (size_t oc = 0; oc < out_ch; oc++) {
                        double w_r = layer->weights[oc * in_ch * modes_out + ic * modes_out + (m < modes_out ? m : 0)].real;
                        double w_i = layer->weights[oc * in_ch * modes_out + ic * modes_out + (m < modes_out ? m : 0)].imag;
                        double dy_r = temp_dy[oc * (half_modes + 1) + m].real;
                        double dy_i = temp_dy[oc * (half_modes + 1) + m].imag;
                        sum_r += w_r * dy_r + w_i * dy_i;
                        sum_i += w_r * dy_i - w_i * dy_r;
                    }
                    dx_hat[m].real = sum_r;
                    dx_hat[m].imag = sum_i;
                }
                size_t in_off = b * in_ch * grid_size + ic * grid_size;
                irfft_1d(dx_hat, grid_size, &dx[in_off]);
            }
        }
    }

    safe_free((void**)&freq_x); safe_free((void**)&freq_dy);
    safe_free((void**)&temp_x); safe_free((void**)&temp_dy);
}

/**
 * @brief 2D谱卷积层反向传播
 */
static void spectral_conv2d_backward(const LaplaceSpectralConv2D* layer,
                                     const double* x, const double* dy,
                                     size_t batch, size_t grid_size,
                                     size_t in_ch,
                                     double* dW, double* db, double* dx) {
    size_t spatial = grid_size * grid_size;
    size_t modes1 = layer->modes1_out;
    size_t modes2 = layer->modes2_out;
    size_t out_ch = layer->out_channels;

    LaplaceComplex* freq_2d = (LaplaceComplex*)safe_malloc(
        spatial * sizeof(LaplaceComplex));
    LaplaceComplex* temp_x = (LaplaceComplex*)safe_malloc(
        in_ch * spatial * sizeof(LaplaceComplex));
    LaplaceComplex* temp_dy = (LaplaceComplex*)safe_malloc(
        out_ch * spatial * sizeof(LaplaceComplex));
    if (!freq_2d || !temp_x || !temp_dy) {
        safe_free((void**)&freq_2d); safe_free((void**)&temp_x); safe_free((void**)&temp_dy);
        return;
    }

    if (dW) memset(dW, 0, out_ch * in_ch * modes1 * modes2 * 2 * sizeof(double));
    if (db) {
        memset(db, 0, out_ch * sizeof(double));
        for (size_t b = 0; b < batch; b++)
            for (size_t oc = 0; oc < out_ch; oc++)
                for (size_t i = 0; i < spatial; i++)
                    db[oc] += dy[b * out_ch * spatial + oc * spatial + i];
    }
    if (dx) memset(dx, 0, batch * in_ch * spatial * sizeof(double));

    for (size_t b = 0; b < batch; b++) {
        for (size_t ic = 0; ic < in_ch; ic++) {
            size_t in_off = b * in_ch * spatial + ic * spatial;
            for (size_t i = 0; i < spatial; i++) {
                freq_2d[i].real = x[in_off + i];
                freq_2d[i].imag = 0.0;
            }
            fft_2d(freq_2d, grid_size, grid_size, 0);
            for (size_t i = 0; i < spatial; i++)
                temp_x[ic * spatial + i] = freq_2d[i];
        }

        for (size_t oc = 0; oc < out_ch; oc++) {
            size_t dy_off = b * out_ch * spatial + oc * spatial;
            for (size_t i = 0; i < spatial; i++) {
                freq_2d[i].real = dy[dy_off + i];
                freq_2d[i].imag = 0.0;
            }
            fft_2d(freq_2d, grid_size, grid_size, 0);
            for (size_t i = 0; i < spatial; i++)
                temp_dy[oc * spatial + i] = freq_2d[i];
        }

        if (dW) {
            for (size_t oc = 0; oc < out_ch; oc++) {
                for (size_t ic = 0; ic < in_ch; ic++) {
                    for (size_t m1 = 0; m1 < modes1; m1++) {
                        for (size_t m2 = 0; m2 < modes2; m2++) {
                            size_t freq_idx = m1 * grid_size + m2;
                            double x_r = temp_x[ic * spatial + freq_idx].real;
                            double x_i = temp_x[ic * spatial + freq_idx].imag;
                            double dy_r = temp_dy[oc * spatial + freq_idx].real;
                            double dy_i = temp_dy[oc * spatial + freq_idx].imag;
                            size_t w_idx = (oc * in_ch * modes1 * modes2 +
                                           ic * modes1 * modes2 + m1 * modes2 + m2) * 2;
                            dW[w_idx]     += x_r * dy_r + x_i * dy_i;
                            dW[w_idx + 1] += x_r * dy_i - x_i * dy_r;
                        }
                    }
                }
            }
        }

        if (dx) {
            for (size_t ic = 0; ic < in_ch; ic++) {
                for (size_t i = 0; i < spatial; i++) {
                    double sum_r = 0.0, sum_i = 0.0;
                    size_t m1 = i / grid_size;
                    size_t m2 = i % grid_size;
                    for (size_t oc = 0; oc < out_ch; oc++) {
                        double w_r = (m1 < modes1 && m2 < modes2) ?
                            layer->weights[oc * in_ch * modes1 * modes2 + ic * modes1 * modes2 + m1 * modes2 + m2].real : 0.0;
                        double w_i = (m1 < modes1 && m2 < modes2) ?
                            layer->weights[oc * in_ch * modes1 * modes2 + ic * modes1 * modes2 + m1 * modes2 + m2].imag : 0.0;
                        double dy_r = temp_dy[oc * spatial + i].real;
                        double dy_i = temp_dy[oc * spatial + i].imag;
                        sum_r += w_r * dy_r + w_i * dy_i;
                        sum_i += w_r * dy_i - w_i * dy_r;
                    }
                    freq_2d[i].real = sum_r;
                    freq_2d[i].imag = sum_i;
                }
                fft_2d(freq_2d, grid_size, grid_size, 1);
                size_t in_off = b * in_ch * spatial + ic * spatial;
                double inv_n = 1.0 / (double)spatial;
                for (size_t i = 0; i < spatial; i++) {
                    dx[in_off + i] = freq_2d[i].real * inv_n;
                }
            }
        }
    }

    safe_free((void**)&freq_2d); safe_free((void**)&temp_x); safe_free((void**)&temp_dy);
}

/* ========== 算子创建与销毁 ========== */

LaplaceNeuralOperator* laplace_operator_create_1d(const LaplaceOperatorConfig* config) {
    if (!config || config->num_layers == 0 || config->num_layers > 6) return NULL;

    LaplaceNeuralOperator* op = (LaplaceNeuralOperator*)safe_malloc(sizeof(LaplaceNeuralOperator));
    if (!op) return NULL;
    memset(op, 0, sizeof(LaplaceNeuralOperator));

    op->config = *config;
    op->input_dim = 1;

    size_t in_ch = config->in_channels;
    size_t hidden = config->hidden_channels;
    size_t out_ch = config->out_channels;

    /* 升维层 */
    op->lifting_weight = (double*)safe_malloc(in_ch * hidden * sizeof(double));
    op->lifting_bias = (double*)safe_malloc(hidden * sizeof(double));
    if (!op->lifting_weight || !op->lifting_bias) {
        laplace_operator_destroy(op);
        return NULL;
    }
    double l_scale = sqrt(2.0 / (double)(in_ch + hidden));
    for (size_t i = 0; i < in_ch * hidden; i++) {
        op->lifting_weight[i] = l_scale * (secure_random_float() * 2.0f - 1.0f);
    }
    memset(op->lifting_bias, 0, hidden * sizeof(double));

    /* 谱卷积层 */
    op->spectral_layers_1d = (LaplaceSpectralConv1D*)safe_malloc(
        config->num_layers * sizeof(LaplaceSpectralConv1D));
    if (!op->spectral_layers_1d) {
        laplace_operator_destroy(op);
        return NULL;
    }
    memset(op->spectral_layers_1d, 0, config->num_layers * sizeof(LaplaceSpectralConv1D));

    for (size_t i = 0; i < config->num_layers; i++) {
        if (spectral_conv1d_init(&op->spectral_layers_1d[i],
                                 hidden, hidden,
                                 config->modes_1d, config->modes_1d) != 0) {
            laplace_operator_destroy(op);
            return NULL;
        }
    }

    /* 降维层 */
    op->projection_weight = (double*)safe_malloc(hidden * out_ch * sizeof(double));
    op->projection_bias = (double*)safe_malloc(out_ch * sizeof(double));
    if (!op->projection_weight || !op->projection_bias) {
        laplace_operator_destroy(op);
        return NULL;
    }
    double p_scale = sqrt(2.0 / (double)(hidden + out_ch));
    for (size_t i = 0; i < hidden * out_ch; i++) {
        op->projection_weight[i] = p_scale * (secure_random_float() * 2.0f - 1.0f);
    }
    memset(op->projection_bias, 0, out_ch * sizeof(double));

    /* 批量归一化参数 */
    if (config->use_batch_norm) {
        op->bn_gamma = (double*)safe_malloc(config->num_layers * hidden * sizeof(double));
        op->bn_beta = (double*)safe_malloc(config->num_layers * hidden * sizeof(double));
        op->bn_running_mean = (double*)safe_malloc(config->num_layers * hidden * sizeof(double));
        op->bn_running_var = (double*)safe_malloc(config->num_layers * hidden * sizeof(double));
        op->bn_d_gamma = (double*)safe_calloc(config->num_layers * hidden, sizeof(double));
        op->bn_d_beta = (double*)safe_calloc(config->num_layers * hidden, sizeof(double));
        if (!op->bn_gamma || !op->bn_beta || !op->bn_running_mean || !op->bn_running_var ||
            !op->bn_d_gamma || !op->bn_d_beta) {
            laplace_operator_destroy(op);
            return NULL;
        }
        for (size_t i = 0; i < config->num_layers * hidden; i++) {
            op->bn_gamma[i] = 1.0;
            op->bn_beta[i] = 0.0;
            op->bn_running_mean[i] = 0.0;
            op->bn_running_var[i] = 1.0;
        }
    }

    /* 计算总参数量用于Adam优化器 */
    size_t total_params = in_ch * hidden + hidden + /* lifting */
                          config->num_layers * (hidden * hidden * config->modes_1d * 2 + hidden) + /* spectral */
                          hidden * out_ch + out_ch; /* projection */
    if (config->use_batch_norm) {
        total_params += config->num_layers * hidden * 2;
    }

    op->adam_m = (double*)safe_malloc(total_params * sizeof(double));
    op->adam_v = (double*)safe_malloc(total_params * sizeof(double));
    if (!op->adam_m || !op->adam_v) {
        laplace_operator_destroy(op);
        return NULL;
    }
    memset(op->adam_m, 0, total_params * sizeof(double));
    memset(op->adam_v, 0, total_params * sizeof(double));
    op->adam_step = 0;

    return op;
}

LaplaceNeuralOperator* laplace_operator_create_2d(const LaplaceOperatorConfig* config) {
    if (!config || config->num_layers == 0 || config->num_layers > 6) return NULL;

    LaplaceNeuralOperator* op = (LaplaceNeuralOperator*)safe_malloc(sizeof(LaplaceNeuralOperator));
    if (!op) return NULL;
    memset(op, 0, sizeof(LaplaceNeuralOperator));

    op->config = *config;
    op->input_dim = 2;

    size_t in_ch = config->in_channels;
    size_t hidden = config->hidden_channels;
    size_t out_ch = config->out_channels;

    /* 升维层 */
    op->lifting_weight = (double*)safe_malloc(in_ch * hidden * sizeof(double));
    op->lifting_bias = (double*)safe_malloc(hidden * sizeof(double));
    if (!op->lifting_weight || !op->lifting_bias) {
        laplace_operator_destroy(op);
        return NULL;
    }
    double l_scale = sqrt(2.0 / (double)(in_ch + hidden));
    for (size_t i = 0; i < in_ch * hidden; i++) {
        op->lifting_weight[i] = l_scale * (secure_random_float() * 2.0f - 1.0f);
    }
    memset(op->lifting_bias, 0, hidden * sizeof(double));

    /* 2D谱卷积层 */
    op->spectral_layers_2d = (LaplaceSpectralConv2D*)safe_malloc(
        config->num_layers * sizeof(LaplaceSpectralConv2D));
    if (!op->spectral_layers_2d) {
        laplace_operator_destroy(op);
        return NULL;
    }
    memset(op->spectral_layers_2d, 0, config->num_layers * sizeof(LaplaceSpectralConv2D));

    for (size_t i = 0; i < config->num_layers; i++) {
        if (spectral_conv2d_init(&op->spectral_layers_2d[i],
                                 hidden, hidden,
                                 config->modes1_2d, config->modes2_2d,
                                 config->modes1_2d, config->modes2_2d) != 0) {
            laplace_operator_destroy(op);
            return NULL;
        }
    }

    /* 降维层 */
    op->projection_weight = (double*)safe_malloc(hidden * out_ch * sizeof(double));
    op->projection_bias = (double*)safe_malloc(out_ch * sizeof(double));
    if (!op->projection_weight || !op->projection_bias) {
        laplace_operator_destroy(op);
        return NULL;
    }
    double p_scale = sqrt(2.0 / (double)(hidden + out_ch));
    for (size_t i = 0; i < hidden * out_ch; i++) {
        op->projection_weight[i] = p_scale * (secure_random_float() * 2.0f - 1.0f);
    }
    memset(op->projection_bias, 0, out_ch * sizeof(double));

    if (config->use_batch_norm) {
        op->bn_gamma = (double*)safe_malloc(config->num_layers * hidden * sizeof(double));
        op->bn_beta = (double*)safe_malloc(config->num_layers * hidden * sizeof(double));
        op->bn_running_mean = (double*)safe_malloc(config->num_layers * hidden * sizeof(double));
        op->bn_running_var = (double*)safe_malloc(config->num_layers * hidden * sizeof(double));
        op->bn_d_gamma = (double*)safe_calloc(config->num_layers * hidden, sizeof(double));
        op->bn_d_beta = (double*)safe_calloc(config->num_layers * hidden, sizeof(double));
        if (!op->bn_gamma || !op->bn_beta || !op->bn_running_mean || !op->bn_running_var ||
            !op->bn_d_gamma || !op->bn_d_beta) {
            laplace_operator_destroy(op);
            return NULL;
        }
        for (size_t i = 0; i < config->num_layers * hidden; i++) {
            op->bn_gamma[i] = 1.0;
            op->bn_beta[i] = 0.0;
            op->bn_running_mean[i] = 0.0;
            op->bn_running_var[i] = 1.0;
        }
    }

    size_t total_params = in_ch * hidden + hidden +
                          config->num_layers * (hidden * hidden * config->modes1_2d * config->modes2_2d * 2 + hidden) +
                          hidden * out_ch + out_ch;
    if (config->use_batch_norm) {
        total_params += config->num_layers * hidden * 2;
    }

    op->adam_m = (double*)safe_malloc(total_params * sizeof(double));
    op->adam_v = (double*)safe_malloc(total_params * sizeof(double));
    if (!op->adam_m || !op->adam_v) {
        laplace_operator_destroy(op);
        return NULL;
    }
    memset(op->adam_m, 0, total_params * sizeof(double));
    memset(op->adam_v, 0, total_params * sizeof(double));
    op->adam_step = 0;

    return op;
}

void laplace_operator_destroy(LaplaceNeuralOperator* op) {
    if (!op) return;
    safe_free((void**)&op->lifting_weight);
    safe_free((void**)&op->lifting_bias);
    safe_free((void**)&op->projection_weight);
    safe_free((void**)&op->projection_bias);
    safe_free((void**)&op->bn_gamma);
    safe_free((void**)&op->bn_beta);
    safe_free((void**)&op->bn_running_mean);
    safe_free((void**)&op->bn_running_var);
    safe_free((void**)&op->bn_d_gamma);
    safe_free((void**)&op->bn_d_beta);
    safe_free((void**)&op->adam_m);
    safe_free((void**)&op->adam_v);

    if (op->spectral_layers_1d) {
        for (size_t i = 0; i < op->config.num_layers; i++) {
            safe_free((void**)&op->spectral_layers_1d[i].weights);
            safe_free((void**)&op->spectral_layers_1d[i].bias);
        }
        safe_free((void**)&op->spectral_layers_1d);
    }

    if (op->spectral_layers_2d) {
        for (size_t i = 0; i < op->config.num_layers; i++) {
            safe_free((void**)&op->spectral_layers_2d[i].weights);
            safe_free((void**)&op->spectral_layers_2d[i].bias);
        }
        safe_free((void**)&op->spectral_layers_2d);
    }

    safe_free((void**)&op);
}

/* ========== 前向传播 ========== */

int laplace_operator_forward_1d(const LaplaceNeuralOperator* op,
                                const double* input,
                                size_t batch, size_t grid_size,
                                double* output) {
    if (!op || !input || !output || batch == 0 || grid_size == 0) return -1;
    if (!is_power_of_two(grid_size)) return -1;

    size_t in_ch = op->config.in_channels;
    size_t hidden = op->config.hidden_channels;
    size_t out_ch = op->config.out_channels;
    size_t num_layers = op->config.num_layers;

    /* 分配工作缓冲区 */
    double* h = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
    double* h_next = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
    if (!h || !h_next) {
        safe_free((void**)&h);
        safe_free((void**)&h_next);
        return -1;
    }

    /* 升维: in_ch -> hidden */
    linear_layer_forward(input, h,
                         op->lifting_weight, op->lifting_bias,
                         batch, in_ch, hidden, grid_size);

    /* 逐层谱卷积 + 激活 + 残差连接 */
    for (size_t l = 0; l < num_layers; l++) {
        /* 谱卷积 */
        spectral_conv1d_forward(&op->spectral_layers_1d[l],
                                h, batch, grid_size, hidden, h_next);

        /* 批量归一化 + 激活 + 残差 */
        if (op->config.use_batch_norm) {
            double* bn_out = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
            if (!bn_out) { safe_free((void**)&h); safe_free((void**)&h_next); return -1; }
            for (size_t c = 0; c < hidden; c++) {
                batch_norm_forward(&h_next[c * grid_size], &bn_out[c * grid_size],
                                   op->bn_gamma[l * hidden + c],
                                   op->bn_beta[l * hidden + c],
                                   op->bn_running_mean[l * hidden + c],
                                   op->bn_running_var[l * hidden + c],
                                   batch, grid_size);
            }
            for (size_t i = 0; i < batch * hidden * grid_size; i++) {
                h_next[i] = bn_out[i];
            }
            safe_free((void**)&bn_out);
        }

        /* 激活函数 */
        for (size_t i = 0; i < batch * hidden * grid_size; i++) {
            h_next[i] = apply_activation(h_next[i], op->config.activation);
        }

        /* 残差连接: h = h + h_next */
        for (size_t i = 0; i < batch * hidden * grid_size; i++) {
            h[i] = h[i] + h_next[i];
        }
    }

    /* 降维: hidden -> out_ch */
    linear_layer_forward(h, output,
                         op->projection_weight, op->projection_bias,
                         batch, hidden, out_ch, grid_size);

    safe_free((void**)&h);
    safe_free((void**)&h_next);
    return 0;
}

int laplace_operator_forward_2d(const LaplaceNeuralOperator* op,
                                const double* input,
                                size_t batch, size_t grid_size,
                                double* output) {
    if (!op || !input || !output || batch == 0 || grid_size == 0) return -1;
    if (!is_power_of_two(grid_size)) return -1;

    size_t in_ch = op->config.in_channels;
    size_t hidden = op->config.hidden_channels;
    size_t out_ch = op->config.out_channels;
    size_t num_layers = op->config.num_layers;
    size_t spatial = grid_size * grid_size;

    double* h = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
    double* h_next = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
    if (!h || !h_next) {
        safe_free((void**)&h);
        safe_free((void**)&h_next);
        return -1;
    }

    linear_layer_forward(input, h,
                         op->lifting_weight, op->lifting_bias,
                         batch, in_ch, hidden, spatial);

    for (size_t l = 0; l < num_layers; l++) {
        spectral_conv2d_forward(&op->spectral_layers_2d[l],
                                h, batch, grid_size, hidden, h_next);

        if (op->config.use_batch_norm) {
            double* bn_out = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
            if (!bn_out) { safe_free((void**)&h); safe_free((void**)&h_next); return -1; }
            for (size_t c = 0; c < hidden; c++) {
                batch_norm_forward(&h_next[c * spatial], &bn_out[c * spatial],
                                   op->bn_gamma[l * hidden + c],
                                   op->bn_beta[l * hidden + c],
                                   op->bn_running_mean[l * hidden + c],
                                   op->bn_running_var[l * hidden + c],
                                   batch, spatial);
            }
            for (size_t i = 0; i < batch * hidden * spatial; i++) {
                h_next[i] = bn_out[i];
            }
            safe_free((void**)&bn_out);
        }

        for (size_t i = 0; i < batch * hidden * spatial; i++) {
            h_next[i] = apply_activation(h_next[i], op->config.activation);
        }

        for (size_t i = 0; i < batch * hidden * spatial; i++) {
            h[i] = h[i] + h_next[i];
        }
    }

    linear_layer_forward(h, output,
                         op->projection_weight, op->projection_bias,
                         batch, hidden, out_ch, spatial);

    safe_free((void**)&h);
    safe_free((void**)&h_next);
    return 0;
}

/* ========== 损失计算 ========== */

/**
 * @brief 计算均方误差损失和梯度
 *
 * @param pred 预测值 [n]
 * @param target 目标值 [n]
 * @param n 元素数
 * @param grad 输出梯度 [n]，可为NULL
 * @return double 损失值
 */
static double mse_loss(const double* pred, const double* target,
                       size_t n, double* grad) {
    double loss = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = pred[i] - target[i];
        loss += diff * diff;
        if (grad) {
            grad[i] = 2.0 * diff / (double)n;
        }
    }
    return loss / (double)n;
}

/* ========== Adam优化器步进 ========== */

/**
 * @brief Adam参数更新
 *
 * @param param 参数指针
 * @param grad 梯度指针
 * @param m 一阶动量
 * @param v 二阶动量
 * @param step 步数
 * @param lr 学习率
 * @param n 参数数量
 */
static void adam_update(double* param, const double* grad,
                        double* m, double* v,
                        size_t step, double lr, size_t n) {
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    double m_corr = 1.0 / (1.0 - pow(beta1, (double)step));
    double v_corr = 1.0 / (1.0 - pow(beta2, (double)step));

    for (size_t i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];

        double m_hat = m[i] * m_corr;
        double v_hat = v[i] * v_corr;

        param[i] -= lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

/* ========== 训练步骤 ========== */

double laplace_operator_train_step_1d(LaplaceNeuralOperator* op,
                                      const double* input,
                                      const double* target,
                                      size_t batch, size_t grid_size) {
    if (!op || !input || !target) return -1.0;

    size_t in_ch = op->config.in_channels;
    size_t hidden = op->config.hidden_channels;
    size_t out_ch = op->config.out_channels;
    size_t num_layers = op->config.num_layers;
    size_t n_elem = batch * out_ch * grid_size;

    /* 分配中间保存缓冲区 */
    double* pred = (double*)safe_malloc(n_elem * sizeof(double));
    double* loss_grad = (double*)safe_malloc(n_elem * sizeof(double));
    double* h = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
    double* h_next = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
    double** layer_inputs = NULL;
    double** spec_outputs = NULL;
    if (!pred || !loss_grad || !h || !h_next) {
        safe_free((void**)&pred); safe_free((void**)&loss_grad);
        safe_free((void**)&h); safe_free((void**)&h_next);
        return -1.0;
    }
    layer_inputs = (double**)safe_malloc(num_layers * sizeof(double*));
    spec_outputs = (double**)safe_malloc(num_layers * sizeof(double*));
    if (!layer_inputs || !spec_outputs) {
        safe_free((void**)&pred); safe_free((void**)&loss_grad);
        safe_free((void**)&h); safe_free((void**)&h_next);
        safe_free((void**)&layer_inputs); safe_free((void**)&spec_outputs);
        return -1.0;
    }
    memset(layer_inputs, 0, num_layers * sizeof(double*));
    memset(spec_outputs, 0, num_layers * sizeof(double*));
    {
        int alloc_ok = 1;
        for (size_t l = 0; l < num_layers; l++) {
            layer_inputs[l] = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
            spec_outputs[l] = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
            if (!layer_inputs[l] || !spec_outputs[l]) { alloc_ok = 0; break; }
        }
        if (!alloc_ok) {
            for (size_t l = 0; l < num_layers; l++) {
                safe_free((void**)&layer_inputs[l]); safe_free((void**)&spec_outputs[l]);
            }
            safe_free((void**)&pred); safe_free((void**)&loss_grad);
            safe_free((void**)&h); safe_free((void**)&h_next);
            safe_free((void**)&layer_inputs); safe_free((void**)&spec_outputs);
            return -1.0;
        }
    }

    /* ========== 前向传播，保存所有中间状态 ========== */

    /* 升维: in_ch -> hidden */
    linear_layer_forward(input, h,
                         op->lifting_weight, op->lifting_bias,
                         batch, in_ch, hidden, grid_size);

    /* 逐层谱卷积 + 激活 + 残差连接 */
    for (size_t l = 0; l < num_layers; l++) {
        memcpy(layer_inputs[l], h, batch * hidden * grid_size * sizeof(double));

        spectral_conv1d_forward(&op->spectral_layers_1d[l],
                                h, batch, grid_size, hidden, h_next);

        if (op->config.use_batch_norm) {
            double* bn_out = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
            if (!bn_out) { /* fallback: 跳过BN */ } else {
                for (size_t c = 0; c < hidden; c++) {
                    batch_norm_forward(&h_next[c * grid_size], &bn_out[c * grid_size],
                                       op->bn_gamma[l * hidden + c],
                                       op->bn_beta[l * hidden + c],
                                       op->bn_running_mean[l * hidden + c],
                                       op->bn_running_var[l * hidden + c],
                                       batch, grid_size);
                }
                memcpy(h_next, bn_out, batch * hidden * grid_size * sizeof(double));
                safe_free((void**)&bn_out);
            }
        }

        memcpy(spec_outputs[l], h_next, batch * hidden * grid_size * sizeof(double));

        for (size_t i = 0; i < batch * hidden * grid_size; i++) {
            h_next[i] = apply_activation(h_next[i], op->config.activation);
        }

        for (size_t i = 0; i < batch * hidden * grid_size; i++) {
            h[i] = h[i] + h_next[i];
        }
    }

    /* 保存激活前h用于投影反向传播 */
    double* proj_input = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
    if (!proj_input) {
        for (size_t l = 0; l < num_layers; l++) {
            safe_free((void**)&layer_inputs[l]); safe_free((void**)&spec_outputs[l]);
        }
        safe_free((void**)&pred); safe_free((void**)&loss_grad);
        safe_free((void**)&h); safe_free((void**)&h_next);
        safe_free((void**)&layer_inputs); safe_free((void**)&spec_outputs);
        return -1.0;
    }
    memcpy(proj_input, h, batch * hidden * grid_size * sizeof(double));

    /* 降维: hidden -> out_ch */
    linear_layer_forward(proj_input, pred,
                         op->projection_weight, op->projection_bias,
                         batch, hidden, out_ch, grid_size);

    /* ========== 计算损失 ========== */
    double loss = mse_loss(pred, target, n_elem, loss_grad);

    /* ========== 解析梯度反向传播 ========== */
    double lr = op->config.learning_rate;
    size_t param_idx = 0;

    /* --- 投影层反向传播 --- */
    size_t proj_size = hidden * out_ch;
    double* proj_dW = (double*)safe_malloc(proj_size * sizeof(double));
    double* proj_db = (double*)safe_malloc(out_ch * sizeof(double));
    double* dL_dh = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
    if (proj_dW && proj_db && dL_dh) {
        linear_layer_backward(proj_input, loss_grad, op->projection_weight,
                              batch, hidden, out_ch, grid_size,
                              proj_dW, proj_db, dL_dh);

        adam_update(op->projection_weight, proj_dW,
                    &op->adam_m[param_idx], &op->adam_v[param_idx],
                    op->adam_step, lr, proj_size);
        param_idx += proj_size;
        adam_update(op->projection_bias, proj_db,
                    &op->adam_m[param_idx], &op->adam_v[param_idx],
                    op->adam_step, lr, out_ch);
        param_idx += out_ch;
    } else {
        param_idx += proj_size + out_ch;
    }
    safe_free((void**)&proj_dW);
    safe_free((void**)&proj_db);
    safe_free((void**)&proj_input);

    /* --- 谱层反向传播（反向遍历） --- */
    double* dL_dh_next = dL_dh;
    for (int l = (int)num_layers - 1; l >= 0; l--) {
        LaplaceSpectralConv1D* layer = &op->spectral_layers_1d[l];
        size_t w_complex = layer->out_channels * layer->in_channels * layer->modes_out * 2;
        size_t b_size = layer->out_channels;

        /* 梯度通过激活函数反向 (dL/dspec = dL/dh_next * activation'(spec_output)) */
        for (size_t i = 0; i < batch * hidden * grid_size; i++) {
            dL_dh_next[i] *= activation_derivative(spec_outputs[l][i], op->config.activation);
        }

        /* BN梯度反向传播：计算dL/dgamma和dL/dbeta，dL_dh_next变为BN输入梯度 */
        if (op->config.use_batch_norm) {
            for (size_t c = 0; c < hidden; c++) {
                double gamma = op->bn_gamma[l * hidden + c];
                double beta_val = op->bn_beta[l * hidden + c];
                double var = op->bn_running_var[l * hidden + c];
                double std_inv = 1.0 / sqrt(var + 1e-5);
                double dgamma = 0.0;
                double dbeta = 0.0;
                for (size_t b = 0; b < batch; b++) {
                    for (size_t s = 0; s < grid_size; s++) {
                        size_t idx = b * hidden * grid_size + c * grid_size + s;
                        double dy = dL_dh_next[idx];
                        double y = spec_outputs[l][idx];
                        double x_hat = (y - beta_val) / gamma;
                        dgamma += dy * x_hat;
                        dbeta += dy;
                        dL_dh_next[idx] = dy * gamma * std_inv;
                    }
                }
                op->bn_d_gamma[l * hidden + c] = dgamma;
                op->bn_d_beta[l * hidden + c] = dbeta;
            }
        }

        /* 谱卷积反向传播 (计算dW, db, 和上一层dL/dh) */
        double* lay_dW = (double*)safe_malloc(w_complex * sizeof(double));
        double* lay_db = (double*)safe_malloc(b_size * sizeof(double));
        double* dL_dh_prev = (double*)safe_malloc(batch * hidden * grid_size * sizeof(double));
        if (lay_dW && lay_db && dL_dh_prev) {
            spectral_conv1d_backward(layer, layer_inputs[l], dL_dh_next,
                                     batch, grid_size, hidden,
                                     lay_dW, lay_db, dL_dh_prev);

            adam_update((double*)layer->weights, lay_dW,
                        &op->adam_m[param_idx], &op->adam_v[param_idx],
                        op->adam_step, lr, w_complex);
            param_idx += w_complex;
            adam_update(layer->bias, lay_db,
                        &op->adam_m[param_idx], &op->adam_v[param_idx],
                        op->adam_step, lr, b_size);
            param_idx += b_size;

            /* 残差梯度：dL/dh_prev = dL/dh_next(来自残差直通路径) + dL_from_spectral */
            for (size_t i = 0; i < batch * hidden * grid_size; i++) {
                dL_dh_next[i] = dL_dh_prev[i];
            }
        } else {
            param_idx += w_complex + b_size;
        }
        safe_free((void**)&lay_dW);
        safe_free((void**)&lay_db);
        safe_free((void**)&dL_dh_prev);
    }

    /* --- 升维层反向传播 --- */
    size_t lifting_size = in_ch * hidden;
    double* lift_dW = (double*)safe_malloc(lifting_size * sizeof(double));
    double* lift_db = (double*)safe_malloc(hidden * sizeof(double));
    if (lift_dW && lift_db) {
        linear_layer_backward(input, dL_dh_next, op->lifting_weight,
                              batch, in_ch, hidden, grid_size,
                              lift_dW, lift_db, NULL);
        adam_update(op->lifting_weight, lift_dW,
                    &op->adam_m[param_idx], &op->adam_v[param_idx],
                    op->adam_step, lr, lifting_size);
        param_idx += lifting_size;
        adam_update(op->lifting_bias, lift_db,
                    &op->adam_m[param_idx], &op->adam_v[param_idx],
                    op->adam_step, lr, hidden);
        param_idx += hidden;
    } else {
        param_idx += lifting_size + hidden;
    }
    safe_free((void**)&lift_dW);
    safe_free((void**)&lift_db);

    safe_free((void**)&dL_dh);

    /* BN参数（使用真实梯度更新） */
    if (op->config.use_batch_norm) {
        for (size_t i = 0; i < num_layers * hidden; i++) {
            op->bn_gamma[i] -= lr * op->bn_d_gamma[i];
            op->bn_beta[i] -= lr * op->bn_d_beta[i];
        }
    }

    op->adam_step++;

    /* 清理 */
    safe_free((void**)&pred);
    safe_free((void**)&loss_grad);
    safe_free((void**)&h);
    safe_free((void**)&h_next);
    for (size_t l = 0; l < num_layers; l++) {
        safe_free((void**)&layer_inputs[l]);
        safe_free((void**)&spec_outputs[l]);
    }
    safe_free((void**)&layer_inputs);
    safe_free((void**)&spec_outputs);

    return loss;
}

double laplace_operator_train_step_2d(LaplaceNeuralOperator* op,
                                      const double* input,
                                      const double* target,
                                      size_t batch, size_t grid_size) {
    if (!op || !input || !target) return -1.0;

    size_t in_ch = op->config.in_channels;
    size_t hidden = op->config.hidden_channels;
    size_t out_ch = op->config.out_channels;
    size_t num_layers = op->config.num_layers;
    size_t spatial = grid_size * grid_size;
    double lr = op->config.learning_rate;

    /* ==================== 前向传播（保存中间结果用于反向传播） ==================== */
    double* h = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
    double* h_next = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
    double* pred = (double*)safe_malloc(batch * out_ch * spatial * sizeof(double));
    if (!h || !h_next || !pred) {
        safe_free((void**)&h); safe_free((void**)&h_next); safe_free((void**)&pred);
        return -1.0;
    }

    /* 升维层: input -> h */
    linear_layer_forward(input, h,
                         op->lifting_weight, op->lifting_bias,
                         batch, in_ch, hidden, spatial);

    /* 保存各谱卷积层的输入(残差前)和输出(激活前) */
    double** layer_inputs = (double**)safe_malloc(num_layers * sizeof(double*));
    double** spec_outputs = (double**)safe_malloc(num_layers * sizeof(double*));
    if (!layer_inputs || !spec_outputs) {
        safe_free((void**)&h); safe_free((void**)&h_next); safe_free((void**)&pred);
        safe_free((void**)&layer_inputs); safe_free((void**)&spec_outputs);
        return -1.0;
    }
    for (size_t l = 0; l < num_layers; l++) {
        layer_inputs[l] = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
        spec_outputs[l] = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
        if (!layer_inputs[l] || !spec_outputs[l]) {
            for (size_t k = 0; k <= l; k++) {
                safe_free((void**)&layer_inputs[k]);
                safe_free((void**)&spec_outputs[k]);
            }
            safe_free((void**)&layer_inputs); safe_free((void**)&spec_outputs);
            safe_free((void**)&h); safe_free((void**)&h_next); safe_free((void**)&pred);
            return -1.0;
        }
        memcpy(layer_inputs[l], h, batch * hidden * spatial * sizeof(double));

        spectral_conv2d_forward(&op->spectral_layers_2d[l],
                                h, batch, grid_size, hidden, h_next);

        memcpy(spec_outputs[l], h_next, batch * hidden * spatial * sizeof(double));

        if (op->config.use_batch_norm) {
            double* bn_out = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
            if (!bn_out) {
                for (size_t k = 0; k < num_layers; k++) {
                    safe_free((void**)&layer_inputs[k]);
                    safe_free((void**)&spec_outputs[k]);
                }
                safe_free((void**)&layer_inputs); safe_free((void**)&spec_outputs);
                safe_free((void**)&h); safe_free((void**)&h_next); safe_free((void**)&pred);
                return -1.0;
            }
            for (size_t c = 0; c < hidden; c++) {
                batch_norm_forward(&h_next[c * spatial], &bn_out[c * spatial],
                                   op->bn_gamma[l * hidden + c],
                                   op->bn_beta[l * hidden + c],
                                   op->bn_running_mean[l * hidden + c],
                                   op->bn_running_var[l * hidden + c],
                                   batch, spatial);
            }
            for (size_t i = 0; i < batch * hidden * spatial; i++) {
                h_next[i] = bn_out[i];
            }
            safe_free((void**)&bn_out);
        }

        for (size_t i = 0; i < batch * hidden * spatial; i++) {
            h_next[i] = apply_activation(h_next[i], op->config.activation);
        }

        for (size_t i = 0; i < batch * hidden * spatial; i++) {
            h[i] = h[i] + h_next[i];
        }
    }

    /* 保存投影层输入 */
    double* proj_input = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
    if (!proj_input) {
        for (size_t k = 0; k < num_layers; k++) {
            safe_free((void**)&layer_inputs[k]);
            safe_free((void**)&spec_outputs[k]);
        }
        safe_free((void**)&layer_inputs); safe_free((void**)&spec_outputs);
        safe_free((void**)&h); safe_free((void**)&h_next); safe_free((void**)&pred);
        return -1.0;
    }
    memcpy(proj_input, h, batch * hidden * spatial * sizeof(double));

    linear_layer_forward(h, pred,
                         op->projection_weight, op->projection_bias,
                         batch, hidden, out_ch, spatial);

    /* ==================== 损失计算 ==================== */
    size_t n_elem = batch * out_ch * spatial;
    double* loss_grad = (double*)safe_malloc(n_elem * sizeof(double));
    if (!loss_grad) {
        safe_free((void**)&proj_input);
        for (size_t k = 0; k < num_layers; k++) {
            safe_free((void**)&layer_inputs[k]);
            safe_free((void**)&spec_outputs[k]);
        }
        safe_free((void**)&layer_inputs); safe_free((void**)&spec_outputs);
        safe_free((void**)&h); safe_free((void**)&h_next); safe_free((void**)&pred);
        return -1.0;
    }
    double loss = mse_loss(pred, target, n_elem, loss_grad);

    /* ==================== 反向传播（解析梯度） ==================== */
    double* dL_dh = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
    if (!dL_dh) {
        safe_free((void**)&loss_grad); safe_free((void**)&proj_input);
        for (size_t k = 0; k < num_layers; k++) {
            safe_free((void**)&layer_inputs[k]);
            safe_free((void**)&spec_outputs[k]);
        }
        safe_free((void**)&layer_inputs); safe_free((void**)&spec_outputs);
        safe_free((void**)&h); safe_free((void**)&h_next); safe_free((void**)&pred);
        return -1.0;
    }

    size_t param_idx = 0;

    /* --- 投影层反向传播 --- */
    size_t proj_size = hidden * out_ch;
    double* proj_dW = (double*)safe_malloc(proj_size * sizeof(double));
    double* proj_db = (double*)safe_malloc(out_ch * sizeof(double));
    if (proj_dW && proj_db) {
        linear_layer_backward(proj_input, loss_grad, op->projection_weight,
                              batch, hidden, out_ch, spatial,
                              proj_dW, proj_db, dL_dh);
        adam_update(op->projection_weight, proj_dW,
                    &op->adam_m[param_idx], &op->adam_v[param_idx],
                    op->adam_step, lr, proj_size);
        param_idx += proj_size;
        adam_update(op->projection_bias, proj_db,
                    &op->adam_m[param_idx], &op->adam_v[param_idx],
                    op->adam_step, lr, out_ch);
        param_idx += out_ch;
    } else {
        param_idx += proj_size + out_ch;
    }
    safe_free((void**)&proj_dW);
    safe_free((void**)&proj_db);

    /* --- 各谱卷积层反向传播（逆序） --- */
    for (int l = (int)num_layers - 1; l >= 0; l--) {
        LaplaceSpectralConv2D* layer = &op->spectral_layers_2d[l];
        size_t w_complex = layer->out_channels * layer->in_channels *
                           layer->modes1_out * layer->modes2_out * 2;
        size_t b_size = layer->out_channels;

        for (size_t i = 0; i < batch * hidden * spatial; i++) {
            dL_dh[i] *= activation_derivative(spec_outputs[l][i], op->config.activation);
        }

        /* BN梯度反向传播 */
        if (op->config.use_batch_norm) {
            for (size_t c = 0; c < hidden; c++) {
                double gamma = op->bn_gamma[l * hidden + c];
                double beta_val = op->bn_beta[l * hidden + c];
                double var = op->bn_running_var[l * hidden + c];
                double std_inv = 1.0 / sqrt(var + 1e-5);
                double dgamma = 0.0;
                double dbeta = 0.0;
                for (size_t b = 0; b < batch; b++) {
                    for (size_t s = 0; s < spatial; s++) {
                        size_t idx = b * hidden * spatial + c * spatial + s;
                        double dy = dL_dh[idx];
                        double y = spec_outputs[l][idx];
                        double x_hat = (y - beta_val) / gamma;
                        dgamma += dy * x_hat;
                        dbeta += dy;
                        dL_dh[idx] = dy * gamma * std_inv;
                    }
                }
                op->bn_d_gamma[l * hidden + c] = dgamma;
                op->bn_d_beta[l * hidden + c] = dbeta;
            }
        }

        double* lay_dW = (double*)safe_malloc(w_complex * sizeof(double));
        double* lay_db = (double*)safe_malloc(b_size * sizeof(double));
        double* dL_dh_prev = (double*)safe_malloc(batch * hidden * spatial * sizeof(double));
        if (lay_dW && lay_db && dL_dh_prev) {
            spectral_conv2d_backward(layer, layer_inputs[l], dL_dh,
                                     batch, grid_size, hidden,
                                     lay_dW, lay_db, dL_dh_prev);

            adam_update((double*)layer->weights, lay_dW,
                        &op->adam_m[param_idx], &op->adam_v[param_idx],
                        op->adam_step, lr, w_complex);
            param_idx += w_complex;
            adam_update(layer->bias, lay_db,
                        &op->adam_m[param_idx], &op->adam_v[param_idx],
                        op->adam_step, lr, b_size);
            param_idx += b_size;

            for (size_t i = 0; i < batch * hidden * spatial; i++) {
                dL_dh[i] = dL_dh_prev[i];
            }
        } else {
            param_idx += w_complex + b_size;
        }
        safe_free((void**)&lay_dW);
        safe_free((void**)&lay_db);
        safe_free((void**)&dL_dh_prev);
    }

    /* --- 升维层反向传播 --- */
    size_t lifting_size = in_ch * hidden;
    double* lift_dW = (double*)safe_malloc(lifting_size * sizeof(double));
    double* lift_db = (double*)safe_malloc(hidden * sizeof(double));
    if (lift_dW && lift_db) {
        linear_layer_backward(input, dL_dh, op->lifting_weight,
                              batch, in_ch, hidden, spatial,
                              lift_dW, lift_db, NULL);
        adam_update(op->lifting_weight, lift_dW,
                    &op->adam_m[param_idx], &op->adam_v[param_idx],
                    op->adam_step, lr, lifting_size);
        param_idx += lifting_size;
        adam_update(op->lifting_bias, lift_db,
                    &op->adam_m[param_idx], &op->adam_v[param_idx],
                    op->adam_step, lr, hidden);
        param_idx += hidden;
    } else {
        param_idx += lifting_size + hidden;
    }
    safe_free((void**)&lift_dW);
    safe_free((void**)&lift_db);
    safe_free((void**)&dL_dh);

    if (op->config.use_batch_norm) {
        for (size_t i = 0; i < num_layers * hidden; i++) {
            op->bn_gamma[i] -= lr * op->bn_d_gamma[i];
            op->bn_beta[i] -= lr * op->bn_d_beta[i];
        }
    }

    op->adam_step++;

    safe_free((void**)&pred);
    safe_free((void**)&loss_grad);
    safe_free((void**)&h);
    safe_free((void**)&h_next);
    safe_free((void**)&proj_input);
    for (size_t l = 0; l < num_layers; l++) {
        safe_free((void**)&layer_inputs[l]);
        safe_free((void**)&spec_outputs[l]);
    }
    safe_free((void**)&layer_inputs);
    safe_free((void**)&spec_outputs);

    return loss;
}

/* ========== 辅助接口 ========== */

void laplace_operator_set_learning_rate(LaplaceNeuralOperator* op, double lr) {
    if (op) op->config.learning_rate = lr;
}

void laplace_operator_reset_optimizer(LaplaceNeuralOperator* op) {
    if (!op) return;
    size_t in_ch = op->config.in_channels;
    size_t hidden = op->config.hidden_channels;
    size_t out_ch = op->config.out_channels;
    size_t total_params = in_ch * hidden + hidden +
                          op->config.num_layers * (hidden * hidden * op->config.modes_1d * 2 + hidden) +
                          hidden * out_ch + out_ch;
    if (op->input_dim == 2) {
        total_params = in_ch * hidden + hidden +
                       op->config.num_layers * (hidden * hidden * op->config.modes1_2d * op->config.modes2_2d * 2 + hidden) +
                       hidden * out_ch + out_ch;
    }
    if (op->config.use_batch_norm) {
        total_params += op->config.num_layers * hidden * 2;
    }

    memset(op->adam_m, 0, total_params * sizeof(double));
    memset(op->adam_v, 0, total_params * sizeof(double));
    op->adam_step = 0;
}

/* ========== 参数序列化 ========== */

int laplace_operator_save(const LaplaceNeuralOperator* op, const char* filepath) {
    if (!op || !filepath) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    /* 写入配置头 */
    fwrite(&op->config, sizeof(LaplaceOperatorConfig), 1, fp);
    fwrite(&op->input_dim, sizeof(size_t), 1, fp);

    size_t in_ch = op->config.in_channels;
    size_t hidden = op->config.hidden_channels;
    size_t out_ch = op->config.out_channels;

    /* 写入升维层 */
    fwrite(op->lifting_weight, sizeof(double), in_ch * hidden, fp);
    fwrite(op->lifting_bias, sizeof(double), hidden, fp);

    /* 写入谱卷积层 */
    if (op->input_dim == 1) {
        for (size_t i = 0; i < op->config.num_layers; i++) {
            LaplaceSpectralConv1D* l = &op->spectral_layers_1d[i];
            size_t w_count = l->out_channels * l->in_channels * l->modes_out;
            fwrite(l->weights, sizeof(LaplaceComplex), w_count, fp);
            fwrite(l->bias, sizeof(double), l->out_channels, fp);
        }
    } else {
        for (size_t i = 0; i < op->config.num_layers; i++) {
            LaplaceSpectralConv2D* l = &op->spectral_layers_2d[i];
            size_t w_count = l->out_channels * l->in_channels *
                             l->modes1_out * l->modes2_out;
            fwrite(l->weights, sizeof(LaplaceComplex), w_count, fp);
            fwrite(l->bias, sizeof(double), l->out_channels, fp);
        }
    }

    /* 写入降维层 */
    fwrite(op->projection_weight, sizeof(double), hidden * out_ch, fp);
    fwrite(op->projection_bias, sizeof(double), out_ch, fp);

    /* 写入BN参数 */
    if (op->config.use_batch_norm) {
        size_t bn_size = op->config.num_layers * hidden;
        fwrite(op->bn_gamma, sizeof(double), bn_size, fp);
        fwrite(op->bn_beta, sizeof(double), bn_size, fp);
        fwrite(op->bn_running_mean, sizeof(double), bn_size, fp);
        fwrite(op->bn_running_var, sizeof(double), bn_size, fp);
    }

    fclose(fp);
    return 0;
}

int laplace_operator_load(LaplaceNeuralOperator* op, const char* filepath) {
    if (!op || !filepath) return -1;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    LaplaceOperatorConfig saved_config;
    size_t saved_dim;

    if (fread(&saved_config, sizeof(LaplaceOperatorConfig), 1, fp) != 1 ||
        fread(&saved_dim, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    if (saved_config.in_channels != op->config.in_channels ||
        saved_config.out_channels != op->config.out_channels ||
        saved_config.hidden_channels != op->config.hidden_channels ||
        saved_config.num_layers != op->config.num_layers) {
        fclose(fp);
        return -1;
    }

    size_t in_ch = op->config.in_channels;
    size_t hidden = op->config.hidden_channels;
    size_t out_ch = op->config.out_channels;

    fread(op->lifting_weight, sizeof(double), in_ch * hidden, fp);
    fread(op->lifting_bias, sizeof(double), hidden, fp);

    if (saved_dim == 1) {
        for (size_t i = 0; i < op->config.num_layers; i++) {
            LaplaceSpectralConv1D* l = &op->spectral_layers_1d[i];
            size_t w_count = l->out_channels * l->in_channels * l->modes_out;
            fread(l->weights, sizeof(LaplaceComplex), w_count, fp);
            fread(l->bias, sizeof(double), l->out_channels, fp);
        }
    } else {
        for (size_t i = 0; i < op->config.num_layers; i++) {
            LaplaceSpectralConv2D* l = &op->spectral_layers_2d[i];
            size_t w_count = l->out_channels * l->in_channels *
                             l->modes1_out * l->modes2_out;
            fread(l->weights, sizeof(LaplaceComplex), w_count, fp);
            fread(l->bias, sizeof(double), l->out_channels, fp);
        }
    }

    fread(op->projection_weight, sizeof(double), hidden * out_ch, fp);
    fread(op->projection_bias, sizeof(double), out_ch, fp);

    if (op->config.use_batch_norm) {
        size_t bn_size = op->config.num_layers * hidden;
        fread(op->bn_gamma, sizeof(double), bn_size, fp);
        fread(op->bn_beta, sizeof(double), bn_size, fp);
        fread(op->bn_running_mean, sizeof(double), bn_size, fp);
        fread(op->bn_running_var, sizeof(double), bn_size, fp);
    }

    fclose(fp);
    op->input_dim = saved_dim;
    return 0;
}
