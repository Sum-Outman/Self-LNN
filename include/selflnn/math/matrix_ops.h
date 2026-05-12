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

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MATH_MATRIX_OPS_H */
