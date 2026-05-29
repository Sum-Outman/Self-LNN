/**
 * @file math_utils.h
 * @brief 数学工具库
 * 
 * 提供数学函数、激活函数、损失函数、优化算法和数值计算工具。
 */

#ifndef SELFLNN_MATH_UTILS_H
#define SELFLNN_MATH_UTILS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 激活函数类型
 */
typedef enum {
    ACTIVATION_SIGMOID = 0,    /**< Sigmoid激活函数 */
    ACTIVATION_TANH = 1,       /**< Tanh激活函数 */
    ACTIVATION_RELU = 2,       /**< ReLU激活函数 */
    ACTIVATION_LEAKY_RELU = 3, /**< Leaky ReLU激活函数 */
    ACTIVATION_ELU = 4,        /**< ELU激活函数 */
    ACTIVATION_SOFTPLUS = 5,   /**< Softplus激活函数 */
    ACTIVATION_SOFTMAX = 6,    /**< Softmax激活函数 */
    ACTIVATION_LINEAR = 7      /**< 线性激活函数 */
} ActivationType;

/**
 * @brief 损失函数类型（引用 loss.h 主定义，此处仅为向后兼容）
 */
#include "selflnn/core/loss.h"

/**
 * @brief 优化器类型（引用 optimizer.h 主定义）
 */
#include "selflnn/core/optimizer.h"

/**
 * @brief 矩阵结构体
 */
typedef struct {
    float* data;               /**< 矩阵数据 */
    size_t rows;               /**< 行数 */
    size_t cols;               /**< 列数 */
    size_t stride;             /**< 行步长（用于子矩阵） */
    size_t capacity;           /**< 数据容量（元素数量） */
    int is_view;               /**< 是否是数据视图 */
} Matrix;

/**
 * @brief 向量结构体
 */
typedef struct {
    float* data;               /**< 向量数据 */
    size_t size;               /**< 向量大小 */
    size_t capacity;           /**< 数据容量（元素数量） */
    int is_view;               /**< 是否是数据视图 */
} Vector;

/**
 * @brief 四元数结构体
 * 
 * 四元数表示形式：q = w + xi + yj + zk
 * 存储顺序：[w, x, y, z]，其中w是实部，x,y,z是虚部
 */
#ifndef SELFLNN_QUATERNION_DEFINED
#define SELFLNN_QUATERNION_DEFINED
typedef struct {
    float w;                   /**< 实部 */
    float x;                   /**< 虚部i分量 */
    float y;                   /**< 虚部j分量 */
    float z;                   /**< 虚部k分量 */
} Quaternion;
#endif

/**
 * @brief 四元数数组结构体
 */
typedef struct {
    Quaternion* data;          /**< 四元数数组数据 */
    size_t size;               /**< 四元数数量 */
    size_t capacity;           /**< 数据容量（四元数数量） */
} QuaternionArray;

/**
 * @brief 双四元数（用于6-DOF空间变换）
 * 
 * 双四元数 = 实部四元数（旋转）+ 对偶部四元数（平移）
 * 形式：q = q_r + ε * q_d，其中 ε^2 = 0
 * 实部q_r表示旋转，对偶部q_d = (1/2) * t * q_r 表示平移t
 */
typedef struct {
    Quaternion real;           /**< 实部四元数（旋转） */
    Quaternion dual;           /**< 对偶部四元数（平移相关） */
} DualQuaternion;

/**
 * @brief 随机数生成器配置
 */
typedef struct {
    uint64_t seed;             /**< 随机种子 */
    int use_clock_seed;        /**< 是否使用时钟种子 */
    int distribution_type;     /**< 分布类型：0=均匀，1=正态，2=伯努利 */
} RNGConfig;

/**
 * @brief 激活函数
 */

float activation_sigmoid(float x);
float activation_tanh(float x);
float activation_relu(float x);
float activation_leaky_relu(float x, float alpha);
float activation_elu(float x, float alpha);
float activation_softplus(float x);
float activation_linear(float x);

/**
 * @brief 激活函数导数
 */

float activation_tanh_derivative(float x);
float activation_relu_derivative(float x);
float activation_leaky_relu_derivative(float x, float alpha);
float activation_elu_derivative(float x, float alpha);
float activation_softplus_derivative(float x);
float activation_linear_derivative(float x);

/**
 * @brief 批量激活函数
 */

void activation_sigmoid_array(float* x, size_t n);
void activation_tanh_array(float* x, size_t n);
void activation_relu_array(float* x, size_t n);
void activation_leaky_relu_array(float* x, size_t n, float alpha);
void activation_elu_array(float* x, size_t n, float alpha);
void activation_softplus_array(float* x, size_t n);
void activation_linear_array(float* x, size_t n);

/**
 * @brief Softmax函数
 */

void activation_softmax(float* x, size_t n);
void activation_softmax_derivative(float* x, float* grad, size_t n);

/**
 * @brief 损失函数
 */

float loss_mean_squared_error(const float* predictions, const float* targets, size_t n);
float loss_mean_absolute_error(const float* predictions, const float* targets, size_t n);
float loss_cross_entropy(const float* predictions, const float* targets, size_t n);
float loss_huber(const float* predictions, const float* targets, size_t n, float delta);
float loss_log_cosh(const float* predictions, const float* targets, size_t n);

/**
 * @brief InfoNCE对比学习损失
 *
 * 正样本对的嵌入应该接近，负样本对应该远离。
 * L = -log( exp(sim(q, k+)/τ) / Σ_i exp(sim(q, ki)/τ) )
 *
 * @param query 查询嵌入 [batch × dim]
 * @param key_positive 正样本键 [batch × dim]
 * @param key_negatives 负样本键 [neg × dim]
 * @param dim 嵌入维度
 * @param batch 批次大小
 * @param neg_count 负样本数量
 * @param temperature 温度参数（默认0.07）
 * @return float 平均损失值
 */
float loss_infonce(const float* query, const float* key_positive,
                   const float* key_negatives, int dim, int batch, int neg_count,
                   float temperature);

/**
 * @brief 损失函数梯度
 */

void loss_mean_squared_error_gradient(const float* predictions, const float* targets, 
                                     float* gradient, size_t n);
void loss_mean_absolute_error_gradient(const float* predictions, const float* targets,
                                      float* gradient, size_t n);
void loss_cross_entropy_gradient(const float* predictions, const float* targets,
                                float* gradient, size_t n);
void loss_huber_gradient(const float* predictions, const float* targets,
                        float* gradient, size_t n, float delta);
void loss_log_cosh_gradient(const float* predictions, const float* targets,
                           float* gradient, size_t n);

/**
 * @brief 矩阵操作
 */

Matrix* matrix_create(size_t rows, size_t cols);
Matrix* matrix_create_from_array(const float* data, size_t rows, size_t cols);
Matrix* matrix_view(Matrix* src, size_t row_start, size_t col_start,
                   size_t rows, size_t cols);
void matrix_free(Matrix* mat);
void matrix_copy(Matrix* dest, const Matrix* src);
void matrix_set(Matrix* mat, float value);
void matrix_set_identity(Matrix* mat);
void matrix_set_random(Matrix* mat, float min, float max);
void matrix_add(Matrix* dest, const Matrix* a, const Matrix* b);
void matrix_subtract(Matrix* dest, const Matrix* a, const Matrix* b);
void matrix_multiply(Matrix* dest, const Matrix* a, const Matrix* b);
void matrix_multiply_elementwise(Matrix* dest, const Matrix* a, const Matrix* b);
void matrix_scale(Matrix* mat, float scalar);
void matrix_transpose(Matrix* dest, const Matrix* src);
float matrix_dot(const Matrix* a, const Matrix* b);
float matrix_frobenius_norm(const Matrix* mat);
float matrix_max(const Matrix* mat);
float matrix_min(const Matrix* mat);
float matrix_mean(const Matrix* mat);
float matrix_std(const Matrix* mat);

/**
 * @brief 向量操作
 */

Vector* vector_create(size_t size);
Vector* vector_create_from_array(const float* data, size_t size);
Vector* vector_view(Vector* src, size_t start, size_t size);
void vector_free(Vector* vec);
void vector_copy(Vector* dest, const Vector* src);
void vector_set(Vector* vec, float value);
void vector_set_random(Vector* vec, float min, float max);
void vector_add(Vector* dest, const Vector* a, const Vector* b);
void vector_subtract(Vector* dest, const Vector* a, const Vector* b);
void vector_multiply_elementwise(Vector* dest, const Vector* a, const Vector* b);
void vector_scale(Vector* vec, float scalar);
float vector_dot(const Vector* a, const Vector* b);
float vector_norm(const Vector* vec);
float vector_max(const Vector* vec);
float vector_min(const Vector* vec);
float vector_mean(const Vector* vec);
float vector_std(const Vector* vec);

/**
 * @brief 矩阵向量操作
 */

void matrix_vector_multiply(Vector* dest, const Matrix* mat, const Vector* vec);
void vector_matrix_multiply(Vector* dest, const Vector* vec, const Matrix* mat);

/**
 * @brief 原始数组矩阵向量乘法：y = A * x
 * 
 * @param y 输出向量（大小m）
 * @param A 输入矩阵（m×n，行优先存储）
 * @param x 输入向量（大小n）
 * @param m 矩阵行数
 * @param n 矩阵列数
 */
void matrix_vector_multiply_raw(float* y, const float* A, const float* x, size_t m, size_t n);

/**
 * @brief 随机数生成
 */

void rng_init(const RNGConfig* config);
void rng_seed(uint64_t seed);
uint64_t rng_next(void);
float rng_uniform(float min, float max);
float rng_normal(float mean, float stddev);
float rng_bernoulli(float p);
void rng_shuffle(int* array, size_t n);
void rng_shuffle_float(float* array, size_t n);
void rng_shuffle_size_t(size_t* array, size_t n);

/**
 * @brief 数值稳定函数
 */

float log_sum_exp(const float* x, size_t n);
float softmax_stable(float* x, size_t n, int idx);
float sigmoid_stable(float x);

/**
 * @brief 数学常数
 */

#define MATH_PI 3.14159265358979323846f
#define MATH_E 2.71828182845904523536f
#define MATH_SQRT2 1.41421356237309504880f
#define MATH_SQRT1_2 0.70710678118654752440f
#define MATH_INF_FLOAT (1.0f / 0.0f)
#define MATH_NAN_FLOAT (0.0f / 0.0f)

/**
 * @brief 实数FFT（完整工业级Cooley-Tukey算法实现）
 * 
 * 支持实数输入的高性能FFT计算，包含输入验证和数值稳定性优化。
 * 算法复杂度：O(n log n)
 * 
 * @param input 实数输入数组，长度为n
 * @param n 输入长度（必须是2的幂）
 * @param real_out 输出实部数组，长度为n
 * @param imag_out 输出虚部数组，长度为n
 */
void fft_real(const float* input, int n, float* real_out, float* imag_out);

/**
 * @brief 实用数学函数
 */

float math_clamp(float x, float min_val, float max_val);
float math_saturate(float x);
int math_is_finite(float x);
int math_is_nan(float x);
int math_is_inf(float x);
float math_lerp(float a, float b, float t);
float math_smoothstep(float edge0, float edge1, float x);
float math_sigmoid_approx(float x);
float math_fast_exp(float x);
float math_fast_log(float x);
float math_fast_sqrt(float x);
float math_fast_inv_sqrt(float x);

/**
 * @brief 统计函数
 */

float math_mean(const float* data, size_t n);
float math_variance(const float* data, size_t n);
float math_stddev(const float* data, size_t n);
float math_skewness(const float* data, size_t n);
float math_kurtosis(const float* data, size_t n);
float math_covariance(const float* x, const float* y, size_t n);
float math_correlation(const float* x, const float* y, size_t n);
void math_normalize(float* data, size_t n);
void math_standardize(float* data, size_t n);

/**
 * @brief 统一余弦相似度（F-023修复：消除5处重复实现）
 *
 * 计算两个向量的余弦相似度：cos(θ) = (a·b) / (||a||·||b||)
 *
 * @param a 向量a
 * @param b 向量b
 * @param dim 向量维度
 * @return float 余弦相似度 [-1, 1]，零向量返回0
 */
float math_cosine_similarity(const float* a, const float* b, size_t dim);

/**
 * @brief 四元数基本运算
 */

Quaternion quaternion_create(float w, float x, float y, float z);
Quaternion quaternion_from_axis_angle(float angle, float axis_x, float axis_y, float axis_z);
Quaternion quaternion_from_euler(float roll, float pitch, float yaw);
void quaternion_to_euler(const Quaternion* q, float* roll, float* pitch, float* yaw);
Quaternion quaternion_identity(void);
Quaternion quaternion_conjugate(const Quaternion* q);
Quaternion quaternion_inverse(const Quaternion* q);
float quaternion_norm(const Quaternion* q);
Quaternion quaternion_normalize(const Quaternion* q);
Quaternion quaternion_add(const Quaternion* a, const Quaternion* b);
Quaternion quaternion_subtract(const Quaternion* a, const Quaternion* b);
Quaternion quaternion_multiply(const Quaternion* a, const Quaternion* b);
Quaternion quaternion_scale(const Quaternion* q, float scalar);
float quaternion_dot(const Quaternion* a, const Quaternion* b);
Quaternion quaternion_slerp(const Quaternion* a, const Quaternion* b, float t);

/**
 * @brief 四元数向量运算
 */

void quaternion_rotate_vector(const Quaternion* q, const float* v, float* result);
Quaternion quaternion_between_vectors(const float* v1, const float* v2);

/**
 * @brief 双四元数运算（6-DOF空间变换）
 */

DualQuaternion dual_quaternion_identity(void);
DualQuaternion dual_quaternion_from_rotation_translation(const Quaternion* rotation, const float* translation);
void dual_quaternion_to_rotation_translation(const DualQuaternion* dq, Quaternion* rotation, float* translation);
DualQuaternion dual_quaternion_multiply(const DualQuaternion* a, const DualQuaternion* b);
DualQuaternion dual_quaternion_conjugate(const DualQuaternion* dq);
void dual_quaternion_transform_point(const DualQuaternion* dq, const float* point, float* result);
DualQuaternion dual_quaternion_normalize(const DualQuaternion* dq);
float dual_quaternion_norm(const DualQuaternion* dq);
void dual_quaternion_to_matrix(const DualQuaternion* dq, float* matrix_4x4);
DualQuaternion dual_quaternion_from_matrix(const float* matrix_4x4);
DualQuaternion dual_quaternion_sclerp(const DualQuaternion* a, const DualQuaternion* b, float t);

/**
 * @brief 获取四元数数组元素
 */

QuaternionArray* quaternion_array_create(size_t size);
void quaternion_array_free(QuaternionArray* array);
void quaternion_array_set(QuaternionArray* array, size_t index, const Quaternion* q);
void quaternion_array_get(const QuaternionArray* array, size_t index, Quaternion* q);
void quaternion_array_normalize(QuaternionArray* array);

/**
 * @brief 四元数权重初始化方法类型
 */
typedef enum {
    QUAT_INIT_UNIFORM = 0,       /**< 均匀分布初始化 */
    QUAT_INIT_XAVIER = 1,        /**< Xavier/Glorot初始化（适用于tanh/sigmoid） */
    QUAT_INIT_HE = 2,            /**< He初始化（适用于ReLU） */
    QUAT_INIT_ORTHOGONAL = 3,    /**< 正交初始化（保持梯度流） */
    QUAT_INIT_UNIT_NORM = 4      /**< 单位范数初始化（适用于旋转不变性） */
} QuatInitMethod;

/**
 * @brief 四元数权重初始化参数
 */
typedef struct {
    QuatInitMethod method;       /**< 初始化方法 */
    float scale;                 /**< 缩放因子（默认1.0） */
    unsigned int seed;           /**< 随机种子（0=使用全局种子） */
} QuatWeightInitConfig;

/**
 * @brief 创建随机四元数（均匀分布）
 *
 * @param min 最小值
 * @param max 最大值
 * @return Quaternion 随机四元数
 */
Quaternion quaternion_random_uniform(float min, float max);

/**
 * @brief 四元数Xavier初始化
 *
 * 根据输入/输出维度缩放初始化范围：
 *   scale = sqrt(6 / (input_quats + output_quats))
 *   q ~ U(-scale, scale)
 *
 * @param input_quats 输入四元数数量
 * @param output_quats 输出四元数数量
 * @return Quaternion 初始化的四元数权重
 */
Quaternion quaternion_xavier_init(size_t input_quats, size_t output_quats);

/**
 * @brief 四元数He初始化
 *
 * 适用于ReLU激活:
 *   scale = sqrt(12 / input_quats)
 *   q ~ U(-scale, scale)
 *
 * @param input_quats 输入四元数数量
 * @return Quaternion 初始化的四元数权重
 */
Quaternion quaternion_he_init(size_t input_quats);

/**
 * @brief 四元数正交初始化
 *
 * 生成单位四元数（范数为1），表示SO(3)上的随机旋转。
 * 适用于旋转不变性任务。
 *
 * @return Quaternion 单位范数四元数
 */
Quaternion quaternion_orthogonal_init(void);

/**
 * @brief 批量初始化四元数数组
 *
 * @param weights 四元数权重数组
 * @param count 四元数数量
 * @param input_quats 输入四元数数量（用于Xavier/He）
 * @param output_quats 输出四元数数量（用于Xavier）
 * @param config 初始化配置
 */
void quaternion_init_weights(Quaternion* weights, size_t count,
                             size_t input_quats, size_t output_quats,
                             const QuatWeightInitConfig* config);
void quaternion_array_mean(const QuaternionArray* array, Quaternion* mean);

/**
 * @brief 四元数神经网络激活函数
 */

Quaternion quaternion_activation_tanh(const Quaternion* q);
Quaternion quaternion_activation_relu(const Quaternion* q);
Quaternion quaternion_activation_sigmoid(const Quaternion* q);
void quaternion_activation_tanh_array(QuaternionArray* array);
void quaternion_activation_relu_array(QuaternionArray* array);
void quaternion_activation_sigmoid_array(QuaternionArray* array);

/**
 * @brief 四元数矩阵运算
 */

void quaternion_matrix_multiply(const Quaternion* A, size_t m, size_t n,
                               const Quaternion* B, size_t n2, size_t p,
                               Quaternion* C);

/**
 * @brief 插值函数
 */

float math_linear_interpolate(const float* x, const float* y, size_t n, float xq);
void math_cubic_spline(const float* x, const float* y, size_t n,
                      float* xq, float* yq, size_t nq);

/* ================================================================
 * 统一向量/四元数基础运算（内联函数）
 * 消除 sensor_simulation.c / gpu_cpu.c / robot 模块之间的重复定义
 * ================================================================ */

/** 3D向量加法：out = a + b */
static inline void v3_add(const float a[3], const float b[3], float out[3]) {
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
}
/** 3D向量减法：out = a - b */
static inline void v3_sub(const float a[3], const float b[3], float out[3]) {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
}
/** 3D向量缩放：out = v * s */
static inline void v3_scale(const float v[3], float s, float out[3]) {
    out[0] = v[0] * s; out[1] = v[1] * s; out[2] = v[2] * s;
}
/** 3D向量点积 */
#ifndef SELFLNN_VEC3_OPS_DEFINED_DOT
#define SELFLNN_VEC3_OPS_DEFINED_DOT
static inline float vec3_dot(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
#endif
/** 3D向量叉积：out = a × b */
#ifndef SELFLNN_VEC3_OPS_DEFINED_CROSS
#define SELFLNN_VEC3_OPS_DEFINED_CROSS
static inline void vec3_cross(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}
#endif
/** 3D向量长度 */
static inline float vec3_len(const float v[3]) {
    return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
/** 3D向量长度平方 */
static inline float vec3_lensq(const float v[3]) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}
/** 3D向量归一化 */
#ifndef SELFLNN_VEC3_OPS_DEFINED_NORMALIZE
#define SELFLNN_VEC3_OPS_DEFINED_NORMALIZE
static inline void vec3_normalize(const float v[3], float out[3]) {
    float len = vec3_len(v);
    if (len < 1e-8f) { out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f; return; }
    float inv = 1.0f / len;
    out[0] = v[0] * inv; out[1] = v[1] * inv; out[2] = v[2] * inv;
}
#endif

/** 四元数乘法：out = a * b */
static inline void quat_multiply(const float a[4], const float b[4], float out[4]) {
    out[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
    out[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0];
    out[2] = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3];
    out[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
}
/** 四元数共轭 */
static inline void quat_conjugate(const float q[4], float out[4]) {
    out[0] = -q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = q[3];
}
/** 四元数旋转：out = q * v * q^(-1) */
static inline void quat_rotate(const float q[4], const float v[3], float out[3]) {
    float qv[4] = { v[0], v[1], v[2], 0.0f };
    float qconj[4]; quat_conjugate(q, qconj);
    float tmp[4]; quat_multiply(q, qv, tmp);
    quat_multiply(tmp, qconj, tmp);
    out[0] = tmp[0]; out[1] = tmp[1]; out[2] = tmp[2];
}
/** 四元数归一化 */
static inline void quat_normalize(float q[4]) {
    float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n < 1e-15f) { q[0] = 0.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 1.0f; return; }
    float inv = 1.0f / n;
    q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
}
/** 四元数从轴角构造 (axis=[x,y,z], angle弧度) */
static inline void quat_from_axis_angle(const float axis[3], float angle, float out[4]) {
    float half = angle * 0.5f;
    float s = sinf(half);
    out[0] = axis[0] * s;
    out[1] = axis[1] * s;
    out[2] = axis[2] * s;
    out[3] = cosf(half);
}
/** 四元数转3x3旋转矩阵 (列主序, mat[9]) */
static inline void quat_to_matrix(const float q[4], float mat[9]) {
    float x = q[0], y = q[1], z = q[2], w = q[3];
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    mat[0] = 1.0f - 2.0f * (yy + zz);
    mat[1] = 2.0f * (xy + wz);
    mat[2] = 2.0f * (xz - wy);
    mat[3] = 2.0f * (xy - wz);
    mat[4] = 1.0f - 2.0f * (xx + zz);
    mat[5] = 2.0f * (yz + wx);
    mat[6] = 2.0f * (xz + wy);
    mat[7] = 2.0f * (yz - wx);
    mat[8] = 1.0f - 2.0f * (xx + yy);
}

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MATH_UTILS_H */