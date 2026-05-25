#ifndef SELFLNN_MATH_MATRIX_OPS_H
#define SELFLNN_MATH_MATRIX_OPS_H

#include "selflnn/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 矩阵运算接口 */

/* 矩阵乘法: C[m×n] = A[m×k] * B[k×n] */
void matrix_multiply(const float* a, const float* b, float* c,
                     size_t m, size_t k, size_t n);

/* 矩阵-向量乘法: y[m] = A[m×n] * x[n] */
void matrix_vector_multiply(const float* a, const float* x, float* y,
                            size_t m, size_t n);

/* 矩阵转置: B[n×m] = A[m×n]^T */
void matrix_transpose(const float* a, float* b, size_t m, size_t n);

/* 向量点积 */
float vector_dot_product(const float* a, const float* b, size_t n);

/* 向量加法: c[i] = a[i] + b[i] */
void vector_add(const float* a, const float* b, float* c, size_t n);

/* 向量缩放: y[i] = alpha * x[i] */
void vector_scale(const float* x, float* y, float alpha, size_t n);

/* 矩阵初始化 */
void matrix_init_random(float* mat, size_t rows, size_t cols, float scale);
void matrix_init_zeros(float* mat, size_t rows, size_t cols);

/* SIMD加速标记: 返回1表示支持SIMD加速 */
int matrix_ops_has_simd(void);

/* ZSFWS-S013修复: 补充在matrix_ops.c中实现但未在头文件中声明的5个核心函数 */

/* LU分解: 将A分解为LU，返回0成功 */
int matrix_lu_decompose(float* A, int* pivot, size_t n);

/* LU求解: 使用LU分解结果求解Ax=b，返回0成功/-1失败 */
int matrix_lu_solve(const float* A, const float* b, float* x, size_t n);

/* 矩阵求逆: A^(-1) = invA */
int matrix_inverse(const float* A, float* invA, size_t n);

/* QR分解: A = Q*R (使用Householder变换) */
int matrix_qr_decompose(const float* A, float* Q, float* R, size_t m, size_t n);

/* 幂迭代法: 计算矩阵A的最大特征值和特征向量 */
float matrix_power_iteration(const float* A, float* eigenvector, size_t n,
                              int max_iter, float tol);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MATH_MATRIX_OPS_H */
