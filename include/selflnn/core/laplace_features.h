#ifndef SELFLNN_LAPLACE_FEATURES_H
#define SELFLNN_LAPLACE_FEATURES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== A01.3.1 多尺度拉普拉斯金字塔 ========== */

typedef struct {
    int num_levels;
    float** pyramid;        /* [levels] 每层数据指针 */
    int* widths;            /* 每层宽度 */
    int* heights;           /* 每层高度 */
    int channels;           /* 通道数 */
} LaplacianPyramid;

int laplace_build_gaussian_pyramid(const float* input, int width, int height, int channels, int num_levels, LaplacianPyramid* pyramid);
int laplace_build_laplacian_pyramid(const float* input, int width, int height, int channels, int num_levels, LaplacianPyramid* pyramid);
int laplace_reconstruct_from_pyramid(const LaplacianPyramid* pyramid, float* output, int width, int height);
void laplace_pyramid_free(LaplacianPyramid* pyramid);

/* ========== A01.3.1 拉普拉斯图卷积 ========== */

typedef struct {
    float* laplacian_matrix; /* L = D - A, n x n 矩阵 */
    float* eigenvalues;      /* n 个特征值 */
    float* eigenvectors;     /* n x n 特征向量 */
    int n;                   /* 矩阵规模 */
    int use_normalized;      /* 是否使用归一化拉普拉斯 */
} GraphLaplacian;

int laplace_graph_laplacian_build(const float* adjacency, int n, int use_normalized, GraphLaplacian* gl);
int laplace_graph_laplacian_decompose(GraphLaplacian* gl);
int laplace_graph_conv_chebyshev(const GraphLaplacian* gl, const float* features, int num_features, int filter_size, const float* coeffs, float* output);
int laplace_graph_conv_spectral(const GraphLaplacian* gl, const float* features, int num_features, const float* spectral_filter, float* output);
void laplace_graph_laplacian_free(GraphLaplacian* gl);

/* ========== A01.3.1 拉普拉斯特征映射 ========== */

typedef struct {
    float* embedding;         /* n x embedding_dim 嵌入坐标 */
    float* eigenvalues;       /* embedding_dim 个特征值 */
    int num_points;
    int embedding_dim;
    int data_dim;             /* 原始数据维度 */
    float sigma;              /* 热核宽度 */
    float* distance_matrix;   /* n x n 距离矩阵(用于近邻图构建) */
    float* training_data; /* n x data_dim 原始训练数据(用于新样本投影) */
} LaplacianEigenmap;

int laplace_eigenmap_compute(const float* data, int num_points, int data_dim, int embedding_dim, float sigma, LaplacianEigenmap* map);
int laplace_eigenmap_transform(const LaplacianEigenmap* map, const float* new_point, float* embedding);
void laplace_eigenmap_free(LaplacianEigenmap* map);

/* ========== A01.3.2 拉普拉斯稀疏编码 ========== */

typedef struct {
    float* dictionary;       /* data_dim x dict_size */
    int dict_size;
    int data_dim;
    float lambda;            /* L1稀疏权重 */
    float laplacian_reg;     /* 拉普拉斯正则化权重 */
    float* gram_matrix;      /* dict_size x dict_size */
    float* laplacian_matrix; /* dict_size x dict_size 原子拉普拉斯 */
    int use_laplacian;
} LaplacianSparseCoder;

LaplacianSparseCoder* laplace_sparse_coder_create(int data_dim, int dict_size, float lambda, float laplacian_reg);
int laplace_sparse_encode(const LaplacianSparseCoder* coder, const float* data_point, float* code, int max_iter, float tol);
int laplace_sparse_encode_batch(const LaplacianSparseCoder* coder, const float* data, int num_samples, float* codes, int max_iter, float tol);
int laplace_sparse_dict_update(LaplacianSparseCoder* coder, const float* data, int num_samples, const float* codes, float learning_rate);
void laplace_sparse_coder_free(LaplacianSparseCoder* coder);

/* ========== A01.3.2 拉普拉斯字典学习 ========== */

typedef struct {
    LaplacianSparseCoder* coder;
    float sparsity_target;   /* 目标稀疏度 */
    int max_atoms;           /* 最大原子数 */
    float* atom_norms;       /* 原子L2范数 */
} LaplacianDictLearner;

LaplacianDictLearner* laplace_dict_learner_create(int data_dim, int dict_size, float sparsity_target);
int laplace_dict_learn_batch(LaplacianDictLearner* learner, const float* data, int num_samples, int num_iterations, float learning_rate);
int laplace_dict_encode(const LaplacianDictLearner* learner, const float* data_point, float* code, int max_iter, float tol);
int laplace_dict_get_atom(const LaplacianDictLearner* learner, int atom_idx, float* atom);
void laplace_dict_learner_free(LaplacianDictLearner* learner);

/* ========== A01.3.2 拉普拉斯流形学习 ========== */

typedef struct {
    int method;              /* 0:LLE, 1:Laplacian Eigenmaps, 2:t-SNE类 */
    float* embedding;        /* n x embedding_dim */
    int num_points;
    int embedding_dim;
    float* stress;           /* 应力/重构误差 */
    float* local_errors;     /* 局部邻域误差 */
} LaplaceManifold;

int laplace_lle(const float* data, int num_points, int data_dim, int embedding_dim, int n_neighbors, float reg, float* embedding);
int laplace_manifold_learn(const float* data, int num_points, int data_dim, int embedding_dim, int method, int n_neighbors, LaplaceManifold* result);
void laplace_manifold_free(LaplaceManifold* result);

/* ========== S-005: 拉普拉斯特征与LNN训练管道桥接 ========== */

/* S-005修复: 拉普拉斯特征与LNN训练管道桥接函数
 * 将拉普拉斯特征映射(Laplacian Eigenmap)的降维结果作为
 * LNN训练的辅助特征,提升训练收敛速度和泛化能力。
 * 当无训练数据时返回0表示跳过此增强。
 * @param lnn_instance LNN实例指针（当前版本未使用，预留扩展接口）
 * @param data 输入数据矩阵 [rows × cols]
 * @param rows 数据行数（样本数）
 * @param cols 数据列数（原始特征维度）
 * @param target_dim 目标降维维度
 * @param enhanced_features 输出增强特征矩阵 [rows × target_dim]
 * @return 0=成功(包括跳过), 负值=失败 */
int laplace_features_train_bridge(void* lnn_instance, const float* data,
                                   int rows, int cols, int target_dim,
                                   float* enhanced_features);

#ifdef __cplusplus
}
#endif

#endif
