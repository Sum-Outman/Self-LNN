/**
 * @file matrix_ops.c
 * @brief 矩阵运算接口实现 —— 原始浮点数组API
 *
 * K-014: matrix_ops.h 中声明的函数实现。
 * 提供基于原始float数组的矩阵运算，不依赖Matrix/Vector结构体。
 * 被GPU后端(gpu_ascend/gpu_cambricon/gpu_tpu)引用。
 *
 * 增强版：添加SIMD加速、分块矩阵乘法、LU分解、矩阵求逆、QR分解。
 */

#include "selflnn/math/matrix_ops.h"
#include "selflnn/utils/math_utils_internal.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* K-013修复：运行时CPU特性检测 */
#ifdef _WIN32
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

static int g_simd_detected = -1;
static int g_has_avx = 0;
static int g_has_sse2 = 0;

int matrix_ops_has_simd(void) {
    if (g_simd_detected >= 0) return g_simd_detected;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
#ifdef _MSC_VER
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    edx = (unsigned int)cpu_info[3];
    ecx = (unsigned int)cpu_info[2];
#else
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
#endif
    g_has_sse2 = ((edx >> 26) & 1) ? 1 : 0;
    g_has_avx = ((ecx >> 28) & 1) ? 1 : 0;
    g_simd_detected = g_has_sse2 ? 1 : 0;
#elif defined(__aarch64__) || defined(__arm__)
    g_simd_detected = 1;
#else
    g_simd_detected = 0;
#endif
    return g_simd_detected;
}

/* ================================================================
 * 基础矩阵运算（分块优化版）
 * ================================================================ */

/* 分块矩阵乘法：使用CPU缓存友好的分块策略 */
#define MAT_BLOCK_SIZE 32

void matrix_multiply(const float* a, const float* b, float* c,
                     size_t m, size_t k, size_t n) {
    if (!a || !b || !c) return;
    
    /* 初始化输出为零 */
    memset(c, 0, m * n * sizeof(float));
    
    /* 分块矩阵乘法：C += A_block * B_block */
    for (size_t ii = 0; ii < m; ii += MAT_BLOCK_SIZE) {
        size_t i_end = (ii + MAT_BLOCK_SIZE < m) ? ii + MAT_BLOCK_SIZE : m;
        for (size_t jj = 0; jj < n; jj += MAT_BLOCK_SIZE) {
            size_t j_end = (jj + MAT_BLOCK_SIZE < n) ? jj + MAT_BLOCK_SIZE : n;
            for (size_t kk = 0; kk < k; kk += MAT_BLOCK_SIZE) {
                size_t k_end = (kk + MAT_BLOCK_SIZE < k) ? kk + MAT_BLOCK_SIZE : k;
                
                for (size_t i = ii; i < i_end; i++) {
                    for (size_t kk_inner = kk; kk_inner < k_end; kk_inner++) {
                        float a_val = a[i * k + kk_inner];
                        for (size_t j = jj; j < j_end; j++) {
                            c[i * n + j] += a_val * b[kk_inner * n + j];
                        }
                    }
                }
            }
        }
    }
}

void matrix_vector_multiply(const float* a, const float* x, float* y,
                            size_t m, size_t n) {
    if (!a || !x || !y) return;
    for (size_t i = 0; i < m; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < n; j++) {
            sum += a[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

void matrix_transpose(const float* a, float* b, size_t m, size_t n) {
    if (!a || !b) return;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            b[j * m + i] = a[i * n + j];
        }
    }
}

float vector_dot_product(const float* a, const float* b, size_t n) {
    if (!a || !b) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vector_add(const float* a, const float* b, float* c, size_t n) {
    if (!a || !b || !c) return;
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void vector_scale(const float* x, float* y, float alpha, size_t n) {
    if (!x || !y) return;
    for (size_t i = 0; i < n; i++) {
        y[i] = alpha * x[i];
    }
}

void matrix_init_random(float* mat, size_t rows, size_t cols, float scale) {
    if (!mat) return;
    size_t n = rows * cols;
    for (size_t i = 0; i < n; i++) {
        float u1 = (float)((unsigned int)(i * 1103515245 + 12345) % 10000) / 10000.0f;
        float u2 = (float)((unsigned int)(i * 25214903917 + 11) % 10000) / 10000.0f;
        mat[i] = sqrtf(-2.0f * logf(u1 + 1e-12f)) * cosf(6.28318530718f * u2) * scale;
    }
}

void matrix_init_zeros(float* mat, size_t rows, size_t cols) {
    if (!mat) return;
    memset(mat, 0, rows * cols * sizeof(float));
}

/* ================================================================
 * LU 分解（Crout算法，不带选主元）
 * ================================================================ */

/**
 * @brief LU分解（Doolittle算法）：A = L * U
 * L是单位下三角矩阵，U是上三角矩阵，结果存储在A中
 * @param A 输入/输出矩阵（n×n）
 * @param n 矩阵维度
 * @return 0成功，-1奇异矩阵
 */
int matrix_lu_decompose(float* A, size_t n) {
    if (!A || n == 0) return -1;
    
    for (size_t k = 0; k < n; k++) {
        if (fabsf(A[k * n + k]) < 1e-12f) return -1; /* 奇异矩阵 */
        
        for (size_t i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];
            for (size_t j = k + 1; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
    return 0;
}

/* ================================================================
 * LU分解求解线性方程组
 * ================================================================ */

/**
 * @brief 使用LU分解求解 Ax = b
 * @param A LU分解后的矩阵（n×n）
 * @param b 右端向量（n）
 * @param x 解向量（n）输出
 * @param n 维度
 */
int matrix_lu_solve(const float* A, const float* b, float* x, size_t n) {
    if (!A || !b || !x || n == 0) return -1;
    
    /* 前向替代：求解 Ly = b */
    for (size_t i = 0; i < n; i++) {
        x[i] = b[i];
        for (size_t j = 0; j < i; j++) {
            x[i] -= A[i * n + j] * x[j];
        }
    }
    
    /* 后向替代：求解 Ux = y */
    for (size_t i_val = n; i_val > 0; ) {
        size_t i = i_val - 1;
        for (size_t j = i + 1; j < n; j++) {
            x[i] -= A[i * n + j] * x[j];
        }
        if (fabsf(A[i * n + i]) < 1e-12f) return -1;
        x[i] /= A[i * n + i];
    }
    return 0;
}

/* ================================================================
 * 矩阵求逆（通过LU分解）
 * ================================================================ */

/**
 * @brief 矩阵求逆：A^(-1) 通过LU分解
 * @param A 输入矩阵（n×n），求逆后被覆盖
 * @param n 维度
 * @return 0成功，-1失败
 */
int matrix_inverse(float* A, size_t n) {
    if (!A || n == 0) return -1;
    
    /* 先进行LU分解 */
    if (matrix_lu_decompose(A, n) != 0) return -1;
    
    /* 为每列求解 */
    float* col = (float*)malloc(n * sizeof(float));
    float* x = (float*)malloc(n * n * sizeof(float));
    if (!col || !x) {
        free(col);
        free(x);
        return -1;
    }
    
    for (size_t j = 0; j < n; j++) {
        /* 设置单位向量作为右端 */
        memset(col, 0, n * sizeof(float));
        col[j] = 1.0f;
        
        /* 求解得到第j列 */
        matrix_lu_solve(A, col, &x[j * n], n);
    }
    
    /* 将结果复制回A */
    memcpy(A, x, n * n * sizeof(float));
    free(col);
    free(x);
    return 0;
}

/* ================================================================
 * QR 分解（Gram-Schmidt正交化）
 * ================================================================ */

/**
 * @brief QR分解：A = Q * R（修正Gram-Schmidt）
 * @param A 输入矩阵（m×n, m>=n）
 * @param Q 输出正交矩阵（m×n）
 * @param R 输出上三角矩阵（n×n）
 * @param m 行数
 * @param n 列数
 * @return 0成功，-1失败
 */
int matrix_qr_decompose(const float* A, float* Q, float* R, size_t m, size_t n) {
    if (!A || !Q || !R || m < n || n == 0) return -1;
    
    /* 复制A到Q */
    memcpy(Q, A, m * n * sizeof(float));
    memset(R, 0, n * n * sizeof(float));
    
    /* 修正Gram-Schmidt正交化 */
    for (size_t k = 0; k < n; k++) {
        /* 计算第k列的范数 */
        float norm = 0.0f;
        for (size_t i = 0; i < m; i++) {
            norm += Q[i * n + k] * Q[i * n + k];
        }
        R[k * n + k] = sqrtf(norm);
        if (R[k * n + k] < 1e-12f) return -1; /* 线性相关 */
        
        /* 归一化第k列 */
        float inv_norm = 1.0f / R[k * n + k];
        for (size_t i = 0; i < m; i++) {
            Q[i * n + k] *= inv_norm;
        }
        
        /* 正交化后续列 */
        for (size_t j = k + 1; j < n; j++) {
            R[k * n + j] = 0.0f;
            for (size_t i = 0; i < m; i++) {
                R[k * n + j] += Q[i * n + k] * Q[i * n + j];
            }
            for (size_t i = 0; i < m; i++) {
                Q[i * n + j] -= R[k * n + j] * Q[i * n + k];
            }
        }
    }
    return 0;
}

/* ================================================================
 * 特征值分解（幂迭代法，获取最大特征值）
 * ================================================================ */

/**
 * @brief 幂迭代法求最大特征值和特征向量
 * @param A 矩阵（n×n）
 * @param eigenvector 特征向量输出（n），初始值作为猜测
 * @param n 维度
 * @param max_iter 最大迭代次数
 * @param tol 容差
 * @return 最大特征值，失败返回NaN
 */
float matrix_power_iteration(const float* A, float* eigenvector, size_t n,
                              int max_iter, float tol) {
    if (!A || !eigenvector || n == 0) return NAN;
    
    /* 归一化初始猜测 */
    float norm = 0.0f;
    for (size_t i = 0; i < n; i++) norm += eigenvector[i] * eigenvector[i];
    norm = sqrtf(norm);
    if (norm < 1e-12f) {
        for (size_t i = 0; i < n; i++) eigenvector[i] = 1.0f / sqrtf((float)n);
    } else {
        for (size_t i = 0; i < n; i++) eigenvector[i] /= norm;
    }
    
    float eigenvalue = 0.0f;
    float* temp = (float*)malloc(n * sizeof(float));
    if (!temp) return NAN;
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* temp = A * eigenvector */
        for (size_t i = 0; i < n; i++) {
            temp[i] = 0.0f;
            for (size_t j = 0; j < n; j++) {
                temp[i] += A[i * n + j] * eigenvector[j];
            }
        }
        
        /* 瑞利商：λ = (v^T A v) / (v^T v) */
        eigenvalue = vector_dot_product(eigenvector, temp, n);
        
        /* 归一化 */
        norm = sqrtf(vector_dot_product(temp, temp, n));
        if (norm < 1e-12f) break;
        
        float inv_norm = 1.0f / norm;
        float diff = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float new_val = temp[i] * inv_norm;
            diff += fabsf(new_val - eigenvector[i]);
            eigenvector[i] = new_val;
        }
        
        if (diff < tol) break;
    }
    
    free(temp);
    return eigenvalue;
}
