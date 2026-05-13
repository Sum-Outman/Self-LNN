/**
 * @file math_utils.c
 * @brief 数学工具库实现
 * 
 * 提供数学函数、激活函数、损失函数、优化算法和数值计算工具。
 */

#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef _WIN32
#include <sys/time.h>
#endif

/**
 * @brief 随机数生成器状态
 */
typedef struct {
    uint64_t state[2];         /**< 状态数组 */
    int initialized;           /**< 是否已初始化 */
} RNGState;

/**
 * @brief 全局RNG状态
 */
static RNGState g_rng_state = {{0}, 0};

#ifdef _WIN32
static CRITICAL_SECTION g_rng_lock;
static int g_rng_lock_init = 0;
#define RNG_LOCK() do { if (!g_rng_lock_init) { InitializeCriticalSection(&g_rng_lock); g_rng_lock_init = 1; } EnterCriticalSection(&g_rng_lock); } while(0)
#define RNG_UNLOCK() LeaveCriticalSection(&g_rng_lock)
#else
#include <pthread.h>
static pthread_mutex_t g_rng_lock = PTHREAD_MUTEX_INITIALIZER;
#define RNG_LOCK() pthread_mutex_lock(&g_rng_lock)
#define RNG_UNLOCK() pthread_mutex_unlock(&g_rng_lock)
#endif

/**
 * @brief 矩阵结构体实现
 */
struct Matrix {
    float* data;               /**< 矩阵数据 */
    size_t rows;               /**< 行数 */
    size_t cols;               /**< 列数 */
    size_t stride;             /**< 行步长（用于子矩阵） */
    int is_view;               /**< 是否是数据视图 */
    size_t capacity;           /**< 容量（仅用于非视图） */
};

/**
 * @brief 向量结构体实现
 */
struct Vector {
    float* data;               /**< 向量数据 */
    size_t size;               /**< 向量大小 */
    int is_view;               /**< 是否是数据视图 */
    size_t capacity;           /**< 容量（仅用于非视图） */
};

/**
 * @brief 内部工具函数
 */

static inline float fast_exp(float x) {
    // 使用标准expf函数，确保精确计算。
    return expf(x);
}

static inline float fast_log(float x) {
    // 使用标准logf函数，确保精确计算。
    return logf(x);
}

static inline float fast_sqrt(float x) {
    // 使用标准sqrtf函数，确保精确计算。
    return sqrtf(x);
}

/**
 * @brief 随机数生成：Xorshift128+算法
 */
static uint64_t xorshift128plus(uint64_t* state) {
    uint64_t x = state[0];
    uint64_t const y = state[1];
    state[0] = y;
    x ^= x << 23;
    state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return state[1] + y;
}

/**
 * @brief 激活函数实现
 */

float activation_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float activation_tanh(float x) {
    return tanhf(x);
}

float activation_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float activation_leaky_relu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

float activation_elu(float x, float alpha) {
    return x > 0.0f ? x : alpha * (expf(x) - 1.0f);
}

float activation_softplus(float x) {
    return logf(1.0f + expf(x));
}

float activation_linear(float x) {
    return x;
}

/**
 * @brief 激活函数导数实现
 */

float activation_tanh_derivative(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

float activation_relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

float activation_leaky_relu_derivative(float x, float alpha) {
    return x > 0.0f ? 1.0f : alpha;
}

float activation_elu_derivative(float x, float alpha) {
    return x > 0.0f ? 1.0f : alpha * expf(x);
}

float activation_softplus_derivative(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float activation_linear_derivative(float x) {
    (void)x;  // 参数未使用，线性导数总是1
    return 1.0f;
}

/**
 * @brief 批量激活函数实现
 */

void activation_sigmoid_array(float* x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        x[i] = activation_sigmoid(x[i]);
    }
}

void activation_tanh_array(float* x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        x[i] = tanhf(x[i]);
    }
}

void activation_relu_array(float* x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void activation_leaky_relu_array(float* x, size_t n, float alpha) {
    for (size_t i = 0; i < n; i++) {
        x[i] = x[i] > 0.0f ? x[i] : alpha * x[i];
    }
}

void activation_elu_array(float* x, size_t n, float alpha) {
    for (size_t i = 0; i < n; i++) {
        x[i] = x[i] > 0.0f ? x[i] : alpha * (expf(x[i]) - 1.0f);
    }
}

void activation_softplus_array(float* x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        x[i] = logf(1.0f + expf(x[i]));
    }
}

void activation_linear_array(float* x, size_t n) {
    // 线性激活不改变数据
    (void)x;
    (void)n;
}

/**
 * @brief Softmax函数实现
 */

void activation_softmax(float* x, size_t n) {
    if (n == 0) return;
    
    // 找到最大值以提高数值稳定性
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // 计算指数和
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // 归一化
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < n; i++) {
            x[i] *= inv_sum;
        }
    }
}

void activation_softmax_derivative(float* x, float* grad, size_t n) {
    // 计算softmax梯度
    for (size_t i = 0; i < n; i++) {
        float gi = 0.0f;
        for (size_t j = 0; j < n; j++) {
            float sj = x[j];
            if (i == j) {
                gi += grad[j] * sj * (1.0f - sj);
            } else {
                gi -= grad[j] * sj * x[i];
            }
        }
        x[i] = gi;
    }
}

/**
 * @brief 损失函数实现
 */

float loss_mean_squared_error(const float* predictions, const float* targets, size_t n) {
    if (n == 0) return 0.0f;
    
    float loss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    
    return loss / n;
}

float loss_mean_absolute_error(const float* predictions, const float* targets, size_t n) {
    if (n == 0) return 0.0f;
    
    float loss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        loss += fabsf(predictions[i] - targets[i]);
    }
    
    return loss / n;
}

float loss_cross_entropy(const float* predictions, const float* targets, size_t n) {
    if (n == 0) return 0.0f;
    
    float loss = 0.0f;
    float epsilon = 1e-8f;
    
    for (size_t i = 0; i < n; i++) {
        float p = predictions[i];
        float t = targets[i];
        
        // 限制范围以避免log(0)
        if (p < epsilon) p = epsilon;
        if (p > 1.0f - epsilon) p = 1.0f - epsilon;
        
        loss += t * logf(p) + (1.0f - t) * logf(1.0f - p);
    }
    
    return -loss / n;
}

float loss_huber(const float* predictions, const float* targets, size_t n, float delta) {
    if (n == 0) return 0.0f;
    
    float loss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(predictions[i] - targets[i]);
        if (diff <= delta) {
            loss += 0.5f * diff * diff;
        } else {
            loss += delta * (diff - 0.5f * delta);
        }
    }
    
    return loss / n;
}

float loss_log_cosh(const float* predictions, const float* targets, size_t n) {
    if (n == 0) return 0.0f;
    
    float loss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = predictions[i] - targets[i];
        loss += logf(coshf(diff));
    }
    
    return loss / n;
}

/**
 * @brief InfoNCE对比学习损失
 * 
 * L = -log( exp(sim(q,k+)/τ) / (exp(sim(q,k+)/τ) + Σ exp(sim(q,kn)/τ)) )
 * sim(a,b) = a·b / (|a|·|b|)  余弦相似度
 */
float loss_infonce(const float* query, const float* key_positive,
                   const float* key_negatives, int dim, int batch, int neg_count,
                   float temperature) {
    if (!query || !key_positive || !key_negatives || dim <= 0 || batch <= 0) return 0.0f;
    if (temperature <= 0.0f) temperature = 0.07f;
    float total_loss = 0.0f;
    for (int b = 0; b < batch; b++) {
        const float* q = query + b * dim;
        const float* kp = key_positive + b * dim;
        /* 正样本余弦相似度 */
        float dot_pos = 0.0f, nq = 0.0f, nkp = 0.0f;
        for (int d = 0; d < dim; d++) {
            dot_pos += q[d] * kp[d];
            nq += q[d] * q[d];
            nkp += kp[d] * kp[d];
        }
        float sim_pos = dot_pos / (sqrtf(nq * nkp) + 1e-8f);
        /* 分子 */
        float pos_exp = expf(sim_pos / temperature);
        float denominator = pos_exp;
        /* 负样本 */
        for (int n = 0; n < neg_count; n++) {
            const float* kn = key_negatives + n * dim;
            float dot_neg = 0.0f, nkn = 0.0f;
            for (int d = 0; d < dim; d++) {
                dot_neg += q[d] * kn[d];
                nkn += kn[d] * kn[d];
            }
            float sim_neg = dot_neg / (sqrtf(nq * nkn) + 1e-8f);
            denominator += expf(sim_neg / temperature);
        }
        total_loss += -logf((pos_exp / denominator) + 1e-8f);
    }
    return total_loss / (float)batch;
}

/**
 * @brief 损失函数梯度实现
 */

void loss_mean_squared_error_gradient(const float* predictions, const float* targets,
                                     float* gradient, size_t n) {
    if (n == 0) return;
    
    float scale = 2.0f / n;
    for (size_t i = 0; i < n; i++) {
        gradient[i] = scale * (predictions[i] - targets[i]);
    }
}

void loss_mean_absolute_error_gradient(const float* predictions, const float* targets,
                                      float* gradient, size_t n) {
    if (n == 0) return;
    
    float scale = 1.0f / n;
    for (size_t i = 0; i < n; i++) {
        float diff = predictions[i] - targets[i];
        gradient[i] = scale * (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f));
    }
}

void loss_cross_entropy_gradient(const float* predictions, const float* targets,
                                float* gradient, size_t n) {
    if (n == 0) return;
    
    float epsilon = 1e-8f;
    float scale = -1.0f / n;
    
    for (size_t i = 0; i < n; i++) {
        float p = predictions[i];
        float t = targets[i];
        
        // 限制范围以避免除零
        if (p < epsilon) p = epsilon;
        if (p > 1.0f - epsilon) p = 1.0f - epsilon;
        
        gradient[i] = scale * ((t / p) - ((1.0f - t) / (1.0f - p)));
    }
}

void loss_huber_gradient(const float* predictions, const float* targets,
                        float* gradient, size_t n, float delta) {
    if (n == 0) return;
    
    float scale = 1.0f / n;
    for (size_t i = 0; i < n; i++) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        
        if (abs_diff <= delta) {
            gradient[i] = scale * diff;
        } else {
            gradient[i] = scale * delta * (diff > 0.0f ? 1.0f : -1.0f);
        }
    }
}

void loss_log_cosh_gradient(const float* predictions, const float* targets,
                           float* gradient, size_t n) {
    if (n == 0) return;
    
    float scale = 1.0f / n;
    for (size_t i = 0; i < n; i++) {
        float diff = predictions[i] - targets[i];
        gradient[i] = scale * tanhf(diff);
    }
}

/**
 * @brief 矩阵操作实现
 */

Matrix* matrix_create(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return NULL;
    }
    
    Matrix* mat = (Matrix*)safe_malloc(sizeof(Matrix));
    if (!mat) {
        return NULL;
    }
    
    size_t size = rows * cols;
    mat->data = (float*)safe_malloc(size * sizeof(float));
    if (!mat->data) {
        safe_free((void**)&mat);
        return NULL;
    }
    
    mat->rows = rows;
    mat->cols = cols;
    mat->stride = cols;
    mat->is_view = 0;
    mat->capacity = size;
    
    // 初始化为零
    memset(mat->data, 0, size * sizeof(float));
    
    return mat;
}

Matrix* matrix_create_from_array(const float* data, size_t rows, size_t cols) {
    Matrix* mat = matrix_create(rows, cols);
    if (!mat) {
        return NULL;
    }
    
    memcpy(mat->data, data, rows * cols * sizeof(float));
    return mat;
}

Matrix* matrix_view(Matrix* src, size_t row_start, size_t col_start,
                   size_t rows, size_t cols) {
    if (!src || row_start + rows > src->rows || col_start + cols > src->cols) {
        return NULL;
    }
    
    Matrix* view = (Matrix*)safe_malloc(sizeof(Matrix));
    if (!view) {
        return NULL;
    }
    
    size_t offset = row_start * src->stride + col_start;
    view->data = src->data + offset;
    view->rows = rows;
    view->cols = cols;
    view->stride = src->stride;
    view->is_view = 1;
    view->capacity = 0;
    
    return view;
}

void matrix_free(Matrix* mat) {
    if (!mat) {
        return;
    }
    
    if (!mat->is_view && mat->data) {
        safe_free((void**)&mat->data);
    }
    
    safe_free((void**)&mat);
}

void matrix_copy(Matrix* dest, const Matrix* src) {
    if (!dest || !src || dest->rows != src->rows || dest->cols != src->cols) {
        return;
    }
    
    for (size_t i = 0; i < src->rows; i++) {
        const float* src_row = src->data + i * src->stride;
        float* dest_row = dest->data + i * dest->stride;
        
        for (size_t j = 0; j < src->cols; j++) {
            dest_row[j] = src_row[j];
        }
    }
}

void matrix_set(Matrix* mat, float value) {
    if (!mat) return;
    
    for (size_t i = 0; i < mat->rows; i++) {
        float* row = mat->data + i * mat->stride;
        for (size_t j = 0; j < mat->cols; j++) {
            row[j] = value;
        }
    }
}

void matrix_set_identity(Matrix* mat) {
    if (!mat || mat->rows != mat->cols) return;
    
    matrix_set(mat, 0.0f);
    for (size_t i = 0; i < mat->rows; i++) {
        mat->data[i * mat->stride + i] = 1.0f;
    }
}

void matrix_set_random(Matrix* mat, float min, float max) {
    if (!mat) return;
    
    for (size_t i = 0; i < mat->rows; i++) {
        float* row = mat->data + i * mat->stride;
        for (size_t j = 0; j < mat->cols; j++) {
            row[j] = min + (max - min) * (rng_uniform(0.0f, 1.0f));
        }
    }
}

void matrix_add(Matrix* dest, const Matrix* a, const Matrix* b) {
    if (!dest || !a || !b || dest->rows != a->rows || dest->cols != a->cols ||
        a->rows != b->rows || a->cols != b->cols) {
        return;
    }
    
    for (size_t i = 0; i < a->rows; i++) {
        const float* a_row = a->data + i * a->stride;
        const float* b_row = b->data + i * b->stride;
        float* dest_row = dest->data + i * dest->stride;
        
        for (size_t j = 0; j < a->cols; j++) {
            dest_row[j] = a_row[j] + b_row[j];
        }
    }
}

void matrix_subtract(Matrix* dest, const Matrix* a, const Matrix* b) {
    if (!dest || !a || !b || dest->rows != a->rows || dest->cols != a->cols ||
        a->rows != b->rows || a->cols != b->cols) {
        return;
    }
    
    for (size_t i = 0; i < a->rows; i++) {
        const float* a_row = a->data + i * a->stride;
        const float* b_row = b->data + i * b->stride;
        float* dest_row = dest->data + i * dest->stride;
        
        for (size_t j = 0; j < a->cols; j++) {
            dest_row[j] = a_row[j] - b_row[j];
        }
    }
}

void matrix_multiply(Matrix* dest, const Matrix* a, const Matrix* b) {
    if (!dest || !a || !b || dest->rows != a->rows || dest->cols != b->cols ||
        a->cols != b->rows) {
        return;
    }
    
    // 简单矩阵乘法
    for (size_t i = 0; i < dest->rows; i++) {
        float* dest_row = dest->data + i * dest->stride;
        const float* a_row = a->data + i * a->stride;
        
        for (size_t j = 0; j < dest->cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a->cols; k++) {
                sum += a_row[k] * b->data[k * b->stride + j];
            }
            dest_row[j] = sum;
        }
    }
}

void matrix_multiply_elementwise(Matrix* dest, const Matrix* a, const Matrix* b) {
    if (!dest || !a || !b || dest->rows != a->rows || dest->cols != a->cols ||
        a->rows != b->rows || a->cols != b->cols) {
        return;
    }
    
    for (size_t i = 0; i < a->rows; i++) {
        const float* a_row = a->data + i * a->stride;
        const float* b_row = b->data + i * b->stride;
        float* dest_row = dest->data + i * dest->stride;
        
        for (size_t j = 0; j < a->cols; j++) {
            dest_row[j] = a_row[j] * b_row[j];
        }
    }
}

void matrix_scale(Matrix* mat, float scalar) {
    if (!mat) return;
    
    for (size_t i = 0; i < mat->rows; i++) {
        float* row = mat->data + i * mat->stride;
        for (size_t j = 0; j < mat->cols; j++) {
            row[j] *= scalar;
        }
    }
}

void matrix_transpose(Matrix* dest, const Matrix* src) {
    if (!dest || !src || dest->rows != src->cols || dest->cols != src->rows) {
        return;
    }
    
    for (size_t i = 0; i < src->rows; i++) {
        const float* src_row = src->data + i * src->stride;
        for (size_t j = 0; j < src->cols; j++) {
            dest->data[j * dest->stride + i] = src_row[j];
        }
    }
}

float matrix_dot(const Matrix* a, const Matrix* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < a->rows; i++) {
        const float* a_row = a->data + i * a->stride;
        const float* b_row = b->data + i * b->stride;
        
        for (size_t j = 0; j < a->cols; j++) {
            sum += a_row[j] * b_row[j];
        }
    }
    
    return sum;
}

float matrix_frobenius_norm(const Matrix* mat) {
    if (!mat) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < mat->rows; i++) {
        const float* row = mat->data + i * mat->stride;
        for (size_t j = 0; j < mat->cols; j++) {
            sum += row[j] * row[j];
        }
    }
    
    return sqrtf(sum);
}

float matrix_max(const Matrix* mat) {
    if (!mat || mat->rows == 0 || mat->cols == 0) return 0.0f;
    
    float max_val = mat->data[0];
    for (size_t i = 0; i < mat->rows; i++) {
        const float* row = mat->data + i * mat->stride;
        for (size_t j = 0; j < mat->cols; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
    }
    
    return max_val;
}

float matrix_min(const Matrix* mat) {
    if (!mat || mat->rows == 0 || mat->cols == 0) return 0.0f;
    
    float min_val = mat->data[0];
    for (size_t i = 0; i < mat->rows; i++) {
        const float* row = mat->data + i * mat->stride;
        for (size_t j = 0; j < mat->cols; j++) {
            if (row[j] < min_val) min_val = row[j];
        }
    }
    
    return min_val;
}

float matrix_mean(const Matrix* mat) {
    if (!mat || mat->rows == 0 || mat->cols == 0) return 0.0f;
    
    float sum = 0.0f;
    size_t count = mat->rows * mat->cols;
    
    for (size_t i = 0; i < mat->rows; i++) {
        const float* row = mat->data + i * mat->stride;
        for (size_t j = 0; j < mat->cols; j++) {
            sum += row[j];
        }
    }
    
    return sum / count;
}

float matrix_std(const Matrix* mat) {
    if (!mat || mat->rows == 0 || mat->cols == 0) return 0.0f;
    
    float mean = matrix_mean(mat);
    float sum = 0.0f;
    size_t count = mat->rows * mat->cols;
    
    for (size_t i = 0; i < mat->rows; i++) {
        const float* row = mat->data + i * mat->stride;
        for (size_t j = 0; j < mat->cols; j++) {
            float diff = row[j] - mean;
            sum += diff * diff;
        }
    }
    
    return sqrtf(sum / count);
}

/**
 * @brief 向量操作实现
 */

Vector* vector_create(size_t size) {
    if (size == 0) {
        return NULL;
    }
    
    Vector* vec = (Vector*)safe_malloc(sizeof(Vector));
    if (!vec) {
        return NULL;
    }
    
    vec->data = (float*)safe_malloc(size * sizeof(float));
    if (!vec->data) {
        safe_free((void**)&vec);
        return NULL;
    }
    
    vec->size = size;
    vec->is_view = 0;
    vec->capacity = size;
    
    // 初始化为零
    memset(vec->data, 0, size * sizeof(float));
    
    return vec;
}

Vector* vector_create_from_array(const float* data, size_t size) {
    Vector* vec = vector_create(size);
    if (!vec) {
        return NULL;
    }
    
    memcpy(vec->data, data, size * sizeof(float));
    return vec;
}

Vector* vector_view(Vector* src, size_t start, size_t size) {
    if (!src || start + size > src->size) {
        return NULL;
    }
    
    Vector* view = (Vector*)safe_malloc(sizeof(Vector));
    if (!view) {
        return NULL;
    }
    
    view->data = src->data + start;
    view->size = size;
    view->is_view = 1;
    view->capacity = 0;
    
    return view;
}

void vector_free(Vector* vec) {
    if (!vec) {
        return;
    }
    
    if (!vec->is_view && vec->data) {
        safe_free((void**)&vec->data);
    }
    
    safe_free((void**)&vec);
}

void vector_copy(Vector* dest, const Vector* src) {
    if (!dest || !src || dest->size != src->size) {
        return;
    }
    
    memcpy(dest->data, src->data, src->size * sizeof(float));
}

void vector_set(Vector* vec, float value) {
    if (!vec) return;
    
    for (size_t i = 0; i < vec->size; i++) {
        vec->data[i] = value;
    }
}

void vector_set_random(Vector* vec, float min, float max) {
    if (!vec) return;
    
    for (size_t i = 0; i < vec->size; i++) {
        vec->data[i] = min + (max - min) * (rng_uniform(0.0f, 1.0f));
    }
}

void vector_add(Vector* dest, const Vector* a, const Vector* b) {
    if (!dest || !a || !b || dest->size != a->size || a->size != b->size) {
        return;
    }
    
    for (size_t i = 0; i < a->size; i++) {
        dest->data[i] = a->data[i] + b->data[i];
    }
}

void vector_subtract(Vector* dest, const Vector* a, const Vector* b) {
    if (!dest || !a || !b || dest->size != a->size || a->size != b->size) {
        return;
    }
    
    for (size_t i = 0; i < a->size; i++) {
        dest->data[i] = a->data[i] - b->data[i];
    }
}

void vector_multiply_elementwise(Vector* dest, const Vector* a, const Vector* b) {
    if (!dest || !a || !b || dest->size != a->size || a->size != b->size) {
        return;
    }
    
    for (size_t i = 0; i < a->size; i++) {
        dest->data[i] = a->data[i] * b->data[i];
    }
}

void vector_scale(Vector* vec, float scalar) {
    if (!vec) return;
    
    for (size_t i = 0; i < vec->size; i++) {
        vec->data[i] *= scalar;
    }
}

float vector_dot(const Vector* a, const Vector* b) {
    if (!a || !b || a->size != b->size) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < a->size; i++) {
        sum += a->data[i] * b->data[i];
    }
    
    return sum;
}

float vector_norm(const Vector* vec) {
    if (!vec) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < vec->size; i++) {
        sum += vec->data[i] * vec->data[i];
    }
    
    return sqrtf(sum);
}

float vector_max(const Vector* vec) {
    if (!vec || vec->size == 0) return 0.0f;
    
    float max_val = vec->data[0];
    for (size_t i = 1; i < vec->size; i++) {
        if (vec->data[i] > max_val) max_val = vec->data[i];
    }
    
    return max_val;
}

float vector_min(const Vector* vec) {
    if (!vec || vec->size == 0) return 0.0f;
    
    float min_val = vec->data[0];
    for (size_t i = 1; i < vec->size; i++) {
        if (vec->data[i] < min_val) min_val = vec->data[i];
    }
    
    return min_val;
}

float vector_mean(const Vector* vec) {
    if (!vec || vec->size == 0) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < vec->size; i++) {
        sum += vec->data[i];
    }
    
    return sum / vec->size;
}

float vector_std(const Vector* vec) {
    if (!vec || vec->size == 0) return 0.0f;
    
    float mean = vector_mean(vec);
    float sum = 0.0f;
    
    for (size_t i = 0; i < vec->size; i++) {
        float diff = vec->data[i] - mean;
        sum += diff * diff;
    }
    
    return sqrtf(sum / vec->size);
}

/**
 * @brief 矩阵向量操作实现
 */

void matrix_vector_multiply(Vector* dest, const Matrix* mat, const Vector* vec) {
    if (!dest || !mat || !vec || dest->size != mat->rows || mat->cols != vec->size) {
        return;
    }
    
    for (size_t i = 0; i < mat->rows; i++) {
        float sum = 0.0f;
        const float* row = mat->data + i * mat->stride;
        
        for (size_t j = 0; j < mat->cols; j++) {
            sum += row[j] * vec->data[j];
        }
        
        dest->data[i] = sum;
    }
}

void vector_matrix_multiply(Vector* dest, const Vector* vec, const Matrix* mat) {
    if (!dest || !vec || !mat || dest->size != mat->cols || vec->size != mat->rows) {
        return;
    }
    
    for (size_t j = 0; j < mat->cols; j++) {
        float sum = 0.0f;
        
        for (size_t i = 0; i < mat->rows; i++) {
            sum += vec->data[i] * mat->data[i * mat->stride + j];
        }
        
        dest->data[j] = sum;
    }
}

/**
 * @brief 原始数组矩阵向量乘法：y = A * x
 */
void matrix_vector_multiply_raw(float* y, const float* A, const float* x, size_t m, size_t n) {
    if (!y || !A || !x || m == 0 || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < m; i++) {
        float sum = 0.0f;
        const float* row = A + i * n;
        
        for (size_t j = 0; j < n; j++) {
            sum += row[j] * x[j];
        }
        
        y[i] = sum;
    }
}

/**
 * @brief 随机数生成实现
 */

void rng_init(const RNGConfig* config) {
    RNG_LOCK();
    if (!config) {
        g_rng_state.state[0] = time(NULL);
        g_rng_state.state[1] = g_rng_state.state[0] * 6364136223846793005ULL;
        g_rng_state.initialized = 1;
        RNG_UNLOCK();
        return;
    }
    
    if (config->use_clock_seed) {
#ifdef _WIN32
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        g_rng_state.state[0] = counter.QuadPart;
#else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        g_rng_state.state[0] = tv.tv_sec * 1000000ULL + tv.tv_usec;
#endif
    } else {
        g_rng_state.state[0] = config->seed;
    }
    
    g_rng_state.state[1] = g_rng_state.state[0] * 6364136223846793005ULL;
    g_rng_state.initialized = 1;
    RNG_UNLOCK();
}

void rng_seed(uint64_t seed) {
    RNG_LOCK();
    g_rng_state.state[0] = seed;
    g_rng_state.state[1] = seed * 6364136223846793005ULL;
    g_rng_state.initialized = 1;
    RNG_UNLOCK();
}

uint64_t rng_next(void) {
    RNG_LOCK();
    if (!g_rng_state.initialized) {
        g_rng_state.state[0] = (uint64_t)time(NULL);
        g_rng_state.state[1] = g_rng_state.state[0] * 6364136223846793005ULL;
        g_rng_state.initialized = 1;
    }
    uint64_t r = xorshift128plus(g_rng_state.state);
    RNG_UNLOCK();
    return r;
}

float rng_uniform(float min, float max) {
    uint64_t r = rng_next();
    float t = (float)(r >> 11) * (1.0f / (1ULL << 53));
    return min + t * (max - min);
}

float rng_normal(float mean, float stddev) {
    // Box-Muller变换
    float u1 = rng_uniform(0.0f, 1.0f);
    float u2 = rng_uniform(0.0f, 1.0f);
    
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * MATH_PI * u2);
    return mean + stddev * z0;
}

float rng_bernoulli(float p) {
    return rng_uniform(0.0f, 1.0f) < p ? 1.0f : 0.0f;
}

void rng_shuffle(int* array, size_t n) {
    if (!array || n <= 1) return;
    
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rng_next() % (i + 1);
        
        // 交换
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void rng_shuffle_float(float* array, size_t n) {
    if (!array || n <= 1) return;
    
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rng_next() % (i + 1);
        
        // 交换
        float temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void rng_shuffle_size_t(size_t* array, size_t n) {
    if (!array || n <= 1) return;
    
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rng_next() % (i + 1);
        
        // 交换
        size_t temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

/**
 * @brief 数值稳定函数实现
 */

float log_sum_exp(const float* x, size_t n) {
    if (n == 0) return 0.0f;
    
    // 找到最大值
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // 计算log-sum-exp
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += expf(x[i] - max_val);
    }
    
    return max_val + logf(sum);
}

float softmax_stable(float* x, size_t n, int idx) {
    if (n == 0 || idx < 0 || idx >= (int)n) return 0.0f;
    
    // 计算softmax并返回指定索引的值
    activation_softmax(x, n);
    return x[idx];
}

float sigmoid_stable(float x) {
    // 稳定sigmoid，避免溢出
    if (x >= 0) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

/**
 * @brief 实用数学函数实现
 */

float math_clamp(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

float math_saturate(float x) {
    return math_clamp(x, 0.0f, 1.0f);
}

int math_is_finite(float x) {
    // 完整实现
    return !math_is_nan(x) && !math_is_inf(x);
}

int math_is_nan(float x) {
    return x != x;
}

int math_is_inf(float x) {
    return !math_is_nan(x) && (x > 1e38f || x < -1e38f);
}

float math_lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float math_smoothstep(float edge0, float edge1, float x) {
    x = math_clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

float math_sigmoid_approx(float x) {
    // sigmoid函数精确计算：sigmoid(x) = 0.5 + 0.5*tanh(x/2)（数学等价形式）
    return 0.5f + 0.5f * tanhf(0.5f * x);
}

float math_fast_exp(float x) {
    return fast_exp(x);
}

float math_fast_log(float x) {
    return fast_log(x);
}

float math_fast_sqrt(float x) {
    return fast_sqrt(x);
}

float math_fast_inv_sqrt(float x) {
    // 精确反平方根计算：1.0f / sqrtf(x)。
    if (x <= 0.0f) return 0.0f; // 防止除零和负数
    return 1.0f / sqrtf(x);
}

/**
 * @brief 统计函数实现
 */

float math_mean(const float* data, size_t n) {
    if (n == 0) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    
    return sum / n;
}

float math_variance(const float* data, size_t n) {
    if (n <= 1) return 0.0f;
    
    float mean = math_mean(data, n);
    float sum = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float diff = data[i] - mean;
        sum += diff * diff;
    }
    
    return sum / (n - 1);
}

float math_stddev(const float* data, size_t n) {
    return sqrtf(math_variance(data, n));
}

float math_skewness(const float* data, size_t n) {
    if (n < 3) return 0.0f;
    
    float mean = math_mean(data, n);
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float diff = data[i] - mean;
        sum2 += diff * diff;
        sum3 += diff * diff * diff;
    }
    
    float variance = sum2 / n;
    if (variance == 0.0f) return 0.0f;
    
    return (sum3 / n) / powf(variance, 1.5f);
}

float math_kurtosis(const float* data, size_t n) {
    if (n < 4) return 0.0f;
    
    float mean = math_mean(data, n);
    float sum2 = 0.0f;
    float sum4 = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float diff = data[i] - mean;
        float diff2 = diff * diff;
        sum2 += diff2;
        sum4 += diff2 * diff2;
    }
    
    float variance = sum2 / n;
    if (variance == 0.0f) return 0.0f;
    
    return (sum4 / n) / (variance * variance) - 3.0f;
}

float math_covariance(const float* x, const float* y, size_t n) {
    if (n <= 1) return 0.0f;
    
    float mean_x = math_mean(x, n);
    float mean_y = math_mean(y, n);
    float sum = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        sum += (x[i] - mean_x) * (y[i] - mean_y);
    }
    
    return sum / (n - 1);
}

float math_correlation(const float* x, const float* y, size_t n) {
    if (n <= 1) return 0.0f;
    
    float cov = math_covariance(x, y, n);
    float std_x = math_stddev(x, n);
    float std_y = math_stddev(y, n);
    
    if (std_x == 0.0f || std_y == 0.0f) return 0.0f;
    
    return cov / (std_x * std_y);
}

void math_normalize(float* data, size_t n) {
    if (n == 0) return;
    
    // 找到最小值和最大值
    float min_val = data[0];
    float max_val = data[0];
    
    for (size_t i = 1; i < n; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    float range = max_val - min_val;
    if (range == 0.0f) {
        // 所有值相同，归一化到0.5
        for (size_t i = 0; i < n; i++) {
            data[i] = 0.5f;
        }
        return;
    }
    
    // 归一化到[0, 1]
    for (size_t i = 0; i < n; i++) {
        data[i] = (data[i] - min_val) / range;
    }
}

void math_standardize(float* data, size_t n) {
    if (n <= 1) return;
    
    float mean = math_mean(data, n);
    float std = math_stddev(data, n);
    
    if (std == 0.0f) {
        // 所有值相同，标准化到0
        for (size_t i = 0; i < n; i++) {
            data[i] = 0.0f;
        }
        return;
    }
    
    // 标准化
    for (size_t i = 0; i < n; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

/* F-023修复：统一余弦相似度 */
float math_cosine_similarity(const float* a, const float* b, size_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

/**
 * @brief 插值函数实现
 */

float math_linear_interpolate(const float* x, const float* y, size_t n, float xq) {
    if (n == 0) return 0.0f;
    if (n == 1) return y[0];
    
    // 找到xq所在的区间
    if (xq <= x[0]) return y[0];
    if (xq >= x[n-1]) return y[n-1];
    
    for (size_t i = 0; i < n-1; i++) {
        if (xq >= x[i] && xq <= x[i+1]) {
            float t = (xq - x[i]) / (x[i+1] - x[i]);
            return y[i] + t * (y[i+1] - y[i]);
        }
    }
    
    return y[n-1];
}

void math_cubic_spline(const float* x, const float* y, size_t n,
                      float* xq, float* yq, size_t nq) {
    // 完整实现：自然边界条件的三次样条插值
    // 算法：求解三对角线性系统，计算二阶导数
    
    if (n < 2 || !x || !y || !xq || !yq) {
        return;
    }
    
    // 分配工作内存
    float* h = (float*)safe_malloc((n-1) * sizeof(float));  // 区间宽度
    float* alpha = (float*)safe_malloc((n-1) * sizeof(float));  // 中间变量
    float* l = (float*)safe_malloc(n * sizeof(float));  // 下三角对角线
    float* mu = (float*)safe_malloc((n-1) * sizeof(float));  // 上三角对角线
    float* z = (float*)safe_malloc(n * sizeof(float));  // 临时变量
    float* c = (float*)safe_malloc(n * sizeof(float));  // 二阶导数
    
    if (!h || !alpha || !l || !mu || !z || !c) {
        safe_free((void**)&h);
        safe_free((void**)&alpha);
        safe_free((void**)&l);
        safe_free((void**)&mu);
        safe_free((void**)&z);
        safe_free((void**)&c);
        // 内存不足，回退到线性插值
        for (size_t i = 0; i < nq; i++) {
            yq[i] = math_linear_interpolate(x, y, n, xq[i]);
        }
        return;
    }
    
    // 步骤1: 计算区间宽度 h[i] = x[i+1] - x[i]
    for (size_t i = 0; i < n-1; i++) {
        h[i] = x[i+1] - x[i];
    }
    
    // 步骤2: 为自然样条设置边界条件
    // 自然样条：S''(x0) = S''(xn-1) = 0
    for (size_t i = 1; i < n-1; i++) {
        alpha[i] = (3.0f/h[i]) * (y[i+1] - y[i]) - (3.0f/h[i-1]) * (y[i] - y[i-1]);
    }
    
    // 步骤3: 设置三对角线性系统
    l[0] = 1.0f;
    mu[0] = 0.0f;
    z[0] = 0.0f;
    c[0] = 0.0f;  // 自然边界条件
    
    for (size_t i = 1; i < n-1; i++) {
        l[i] = 2.0f * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i];
    }
    
    l[n-1] = 1.0f;
    z[n-1] = 0.0f;
    c[n-1] = 0.0f;  // 自然边界条件
    
    // 步骤4: 回代计算二阶导数 c[i]
    for (int i = (int)n-2; i >= 0; i--) {
        c[i] = z[i] - mu[i] * c[i+1];
    }
    
    // 步骤5: 计算系数 b[i] 和 d[i]
    float* b = (float*)safe_malloc((n-1) * sizeof(float));
    float* d = (float*)safe_malloc((n-1) * sizeof(float));
    
    if (b && d) {
        for (size_t i = 0; i < n-1; i++) {
            b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2.0f * c[i]) / 3.0f;
            d[i] = (c[i+1] - c[i]) / (3.0f * h[i]);
        }
        
        // 步骤6: 对每个查询点进行插值
        for (size_t qi = 0; qi < nq; qi++) {
            float x_val = xq[qi];
            
            // 找到x_val所在的区间
            size_t i = 0;
            while (i < n-1 && x_val > x[i+1]) {
                i++;
            }
            
            if (i >= n-1) {
                i = n-2;  // 使用最后一个区间
            }
            
            float dx = x_val - x[i];
            yq[qi] = y[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
        }
        
        safe_free((void**)&b);
        safe_free((void**)&d);
    } else {
        // 内存不足，回退到线性插值
        for (size_t i = 0; i < nq; i++) {
            yq[i] = math_linear_interpolate(x, y, n, xq[i]);
        }
    }
    
    // 释放工作内存
    safe_free((void**)&h);
    safe_free((void**)&alpha);
    safe_free((void**)&l);
    safe_free((void**)&mu);
    safe_free((void**)&z);
    safe_free((void**)&c);
}

/* ============================================================================
 * 四元数数学库实现
 * =========================================================================== */

/**
 * @brief 创建四元数
 */
Quaternion quaternion_create(float w, float x, float y, float z) {
    Quaternion q = {w, x, y, z};
    return q;
}

/**
 * @brief 从轴角创建四元数
 */
Quaternion quaternion_from_axis_angle(float angle, float axis_x, float axis_y, float axis_z) {
    float half_angle = angle * 0.5f;
    float sin_half = sinf(half_angle);
    float norm = sqrtf(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
    
    if (norm > 1e-6f) {
        axis_x /= norm;
        axis_y /= norm;
        axis_z /= norm;
    }
    
    Quaternion q = {
        cosf(half_angle),
        axis_x * sin_half,
        axis_y * sin_half,
        axis_z * sin_half
    };
    return q;
}

/**
 * @brief 从欧拉角创建四元数（滚转、俯仰、偏航）
 */
Quaternion quaternion_from_euler(float roll, float pitch, float yaw) {
    float cy = cosf(yaw * 0.5f);
    float sy = sinf(yaw * 0.5f);
    float cp = cosf(pitch * 0.5f);
    float sp = sinf(pitch * 0.5f);
    float cr = cosf(roll * 0.5f);
    float sr = sinf(roll * 0.5f);
    
    Quaternion q = {
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    };
    return q;
}

/**
 * @brief 四元数转换为欧拉角
 */
void quaternion_to_euler(const Quaternion* q, float* roll, float* pitch, float* yaw) {
    // 使用标准转换公式
    float w = q->w, x = q->x, y = q->y, z = q->z;
    
    // 滚转 (x轴旋转)
    float sinr_cosp = 2.0f * (w * x + y * z);
    float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
    *roll = atan2f(sinr_cosp, cosr_cosp);
    
    // 俯仰 (y轴旋转)
    float sinp = 2.0f * (w * y - z * x);
    if (fabsf(sinp) >= 1.0f) {
        *pitch = copysignf(1.5707963267948966f, sinp);
    } else {
        *pitch = asinf(sinp);
    }
    
    // 偏航 (z轴旋转)
    float siny_cosp = 2.0f * (w * z + x * y);
    float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
    *yaw = atan2f(siny_cosp, cosy_cosp);
}

/**
 * @brief 单位四元数
 */
Quaternion quaternion_identity(void) {
    return quaternion_create(1.0f, 0.0f, 0.0f, 0.0f);
}

/**
 * @brief 四元数共轭
 */
Quaternion quaternion_conjugate(const Quaternion* q) {
    return quaternion_create(q->w, -q->x, -q->y, -q->z);
}

/**
 * @brief 四元数逆
 */
Quaternion quaternion_inverse(const Quaternion* q) {
    float norm_sq = q->w*q->w + q->x*q->x + q->y*q->y + q->z*q->z;
    if (norm_sq > 1e-12f) {
        Quaternion conj = quaternion_conjugate(q);
        return quaternion_scale(&conj, 1.0f / norm_sq);
    }
    return quaternion_identity();
}

/**
 * @brief 四元数范数
 */
float quaternion_norm(const Quaternion* q) {
    return sqrtf(q->w*q->w + q->x*q->x + q->y*q->y + q->z*q->z);
}

/**
 * @brief 归一化四元数
 */
Quaternion quaternion_normalize(const Quaternion* q) {
    float norm = quaternion_norm(q);
    if (norm > 1e-12f) {
        return quaternion_create(q->w/norm, q->x/norm, q->y/norm, q->z/norm);
    }
    return quaternion_identity();
}

/**
 * @brief 四元数加法
 */
Quaternion quaternion_add(const Quaternion* a, const Quaternion* b) {
    return quaternion_create(a->w + b->w, a->x + b->x, a->y + b->y, a->z + b->z);
}

/**
 * @brief 四元数减法
 */
Quaternion quaternion_subtract(const Quaternion* a, const Quaternion* b) {
    return quaternion_create(a->w - b->w, a->x - b->x, a->y - b->y, a->z - b->z);
}

/**
 * @brief 四元数乘法
 */
Quaternion quaternion_multiply(const Quaternion* a, const Quaternion* b) {
    Quaternion q = {
        a->w*b->w - a->x*b->x - a->y*b->y - a->z*b->z,
        a->w*b->x + a->x*b->w + a->y*b->z - a->z*b->y,
        a->w*b->y - a->x*b->z + a->y*b->w + a->z*b->x,
        a->w*b->z + a->x*b->y - a->y*b->x + a->z*b->w
    };
    return q;
}

/**
 * @brief 四元数标量乘法
 */
Quaternion quaternion_scale(const Quaternion* q, float scalar) {
    return quaternion_create(q->w * scalar, q->x * scalar, q->y * scalar, q->z * scalar);
}

/**
 * @brief 四元数点积
 */
float quaternion_dot(const Quaternion* a, const Quaternion* b) {
    return a->w*b->w + a->x*b->x + a->y*b->y + a->z*b->z;
}

/**
 * @brief 球面线性插值 (SLERP)
 */
Quaternion quaternion_slerp(const Quaternion* a, const Quaternion* b, float t) {
    float dot = quaternion_dot(a, b);
    
    // 确保最短路径
    Quaternion b_adj = *b;
    if (dot < 0.0f) {
        b_adj = quaternion_scale(&b_adj, -1.0f);
        dot = -dot;
    }
    
    const float DOT_THRESHOLD = 0.9995f;
    if (dot > DOT_THRESHOLD) {
        // 线性插值，当角度很小时
        Quaternion diff = quaternion_subtract(&b_adj, a);
        Quaternion scaled = quaternion_scale(&diff, t);
        Quaternion result = quaternion_add(a, &scaled);
        return quaternion_normalize(&result);
    }
    
    float theta_0 = acosf(dot);
    float theta = theta_0 * t;
    float sin_theta = sinf(theta);
    float sin_theta_0 = sinf(theta_0);
    
    float s0 = cosf(theta) - dot * sin_theta / sin_theta_0;
    float s1 = sin_theta / sin_theta_0;
    
    Quaternion q0 = quaternion_scale(a, s0);
    Quaternion q1 = quaternion_scale(&b_adj, s1);
    return quaternion_add(&q0, &q1);
}

/**
 * @brief 使用四元数旋转向量
 */
void quaternion_rotate_vector(const Quaternion* q, const float* v, float* result) {
    Quaternion v_quat = quaternion_create(0.0f, v[0], v[1], v[2]);
    Quaternion q_inv = quaternion_inverse(q);
    Quaternion rotated = quaternion_multiply(q, &v_quat);
    rotated = quaternion_multiply(&rotated, &q_inv);
    
    result[0] = rotated.x;
    result[1] = rotated.y;
    result[2] = rotated.z;
}

/**
 * @brief 计算两个向量之间的四元数
 */
Quaternion quaternion_between_vectors(const float* v1, const float* v2) {
    float dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    float norm_v1 = sqrtf(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
    float norm_v2 = sqrtf(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);
    
    if (norm_v1 < 1e-6f || norm_v2 < 1e-6f) {
        return quaternion_identity();
    }
    
    float w = norm_v1 * norm_v2 + dot;
    float cross[3] = {
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    };
    
    if (w < 1e-6f * norm_v1 * norm_v2) {
        // 向量方向相反
        float axis[3] = {1.0f, 0.0f, 0.0f};
        float temp = fabsf(v1[0]);
        if (fabsf(v1[1]) > temp) {
            axis[0] = 0.0f; axis[1] = 1.0f; axis[2] = 0.0f;
            temp = fabsf(v1[1]);
        }
        if (fabsf(v1[2]) > temp) {
            axis[0] = 0.0f; axis[1] = 0.0f; axis[2] = 1.0f;
        }
        
        float orthogonal[3] = {
            axis[1]*v1[2] - axis[2]*v1[1],
            axis[2]*v1[0] - axis[0]*v1[2],
            axis[0]*v1[1] - axis[1]*v1[0]
        };
        float norm_ortho = sqrtf(orthogonal[0]*orthogonal[0] + orthogonal[1]*orthogonal[1] + orthogonal[2]*orthogonal[2]);
        if (norm_ortho > 1e-6f) {
            orthogonal[0] /= norm_ortho;
            orthogonal[1] /= norm_ortho;
            orthogonal[2] /= norm_ortho;
            return quaternion_from_axis_angle(3.14159265358979323846f, orthogonal[0], orthogonal[1], orthogonal[2]);
        }
        return quaternion_from_axis_angle(3.14159265358979323846f, 1.0f, 0.0f, 0.0f);
    }
    
    Quaternion q = {w, cross[0], cross[1], cross[2]};
    return quaternion_normalize(&q);
}

/**
 * @brief 创建四元数数组
 */
QuaternionArray* quaternion_array_create(size_t size) {
    QuaternionArray* array = (QuaternionArray*)safe_malloc(sizeof(QuaternionArray));
    if (!array) return NULL;
    
    array->size = size;
    array->capacity = size;
    array->data = (Quaternion*)safe_calloc(size, sizeof(Quaternion));
    
    if (!array->data) {
        safe_free((void**)&array);
        return NULL;
    }
    
    return array;
}

/**
 * @brief 释放四元数数组
 */
void quaternion_array_free(QuaternionArray* array) {
    if (!array) return;
    
    if (array->data) {
        safe_free((void**)&array->data);
    }
    safe_free((void**)&array);
}

/**
 * @brief 设置四元数数组元素
 */
void quaternion_array_set(QuaternionArray* array, size_t index, const Quaternion* q) {
    if (!array || !array->data || index >= array->size) return;
    
    array->data[index] = *q;
}

/**
 * @brief 获取四元数数组元素
 */
void quaternion_array_get(const QuaternionArray* array, size_t index, Quaternion* q) {
    if (!array || !array->data || index >= array->size || !q) return;
    
    *q = array->data[index];
}

/**
 * @brief 归一化四元数数组
 */
void quaternion_array_normalize(QuaternionArray* array) {
    if (!array || !array->data) return;
    
    for (size_t i = 0; i < array->size; i++) {
        array->data[i] = quaternion_normalize(&array->data[i]);
    }
}

/**
 * @brief 计算四元数数组平均值
 */
void quaternion_array_mean(const QuaternionArray* array, Quaternion* mean) {
    if (!array || !array->data || array->size == 0 || !mean) return;
    
    // 使用迭代方法计算四元数平均值
    if (array->size == 1) {
        *mean = array->data[0];
        return;
    }
    
    // 初始猜测：第一个四元数
    *mean = quaternion_normalize(&array->data[0]);
    
    const int max_iterations = 10;
    const float tolerance = 1e-6f;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        Quaternion sum = quaternion_create(0.0f, 0.0f, 0.0f, 0.0f);
        
        for (size_t i = 0; i < array->size; i++) {
            // 计算与当前平均值的差值
            Quaternion conj = quaternion_conjugate(mean);
            Quaternion delta = quaternion_multiply(&array->data[i], &conj);
            sum = quaternion_add(&sum, &delta);
        }
        
        // 平均差值
        sum = quaternion_scale(&sum, 1.0f / array->size);
        
        // 更新平均值
        Quaternion normalized_sum = quaternion_normalize(&sum);
        *mean = quaternion_multiply(&normalized_sum, mean);
        *mean = quaternion_normalize(mean);
        
        // 检查收敛
        if (quaternion_norm(&sum) < tolerance) {
            break;
        }
    }
}

/**
 * @brief 四元数Tanh激活函数
 */
Quaternion quaternion_activation_tanh(const Quaternion* q) {
    // 将四元数视为向量，对每个分量应用tanh
    return quaternion_create(
        tanhf(q->w),
        tanhf(q->x),
        tanhf(q->y),
        tanhf(q->z)
    );
}

/**
 * @brief 四元数ReLU激活函数
 */
Quaternion quaternion_activation_relu(const Quaternion* q) {
    // 对每个分量应用ReLU
    return quaternion_create(
        fmaxf(0.0f, q->w),
        fmaxf(0.0f, q->x),
        fmaxf(0.0f, q->y),
        fmaxf(0.0f, q->z)
    );
}

/**
 * @brief 四元数Sigmoid激活函数
 */
Quaternion quaternion_activation_sigmoid(const Quaternion* q) {
    // 对每个分量应用sigmoid
    float sigmoid_w = 1.0f / (1.0f + expf(-q->w));
    float sigmoid_x = 1.0f / (1.0f + expf(-q->x));
    float sigmoid_y = 1.0f / (1.0f + expf(-q->y));
    float sigmoid_z = 1.0f / (1.0f + expf(-q->z));
    
    return quaternion_create(sigmoid_w, sigmoid_x, sigmoid_y, sigmoid_z);
}

/**
 * @brief 四元数数组Tanh激活函数
 */
void quaternion_activation_tanh_array(QuaternionArray* array) {
    if (!array || !array->data) return;
    
    for (size_t i = 0; i < array->size; i++) {
        array->data[i] = quaternion_activation_tanh(&array->data[i]);
    }
}

/**
 * @brief 四元数数组ReLU激活函数
 */
void quaternion_activation_relu_array(QuaternionArray* array) {
    if (!array || !array->data) return;
    
    for (size_t i = 0; i < array->size; i++) {
        array->data[i] = quaternion_activation_relu(&array->data[i]);
    }
}

/**
 * @brief 四元数数组Sigmoid激活函数
 */
void quaternion_activation_sigmoid_array(QuaternionArray* array) {
    if (!array || !array->data) return;
    
    for (size_t i = 0; i < array->size; i++) {
        array->data[i] = quaternion_activation_sigmoid(&array->data[i]);
    }
}

/**
 * @brief 四元数矩阵乘法
 */
void quaternion_matrix_multiply(const Quaternion* A, size_t m, size_t n,
                               const Quaternion* B, size_t n2, size_t p,
                               Quaternion* C) {
    // 验证维度
    if (n != n2) return;
    
    // 初始化结果矩阵为零
    for (size_t i = 0; i < m * p; i++) {
        C[i] = quaternion_create(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    // 执行矩阵乘法
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            Quaternion sum = quaternion_create(0.0f, 0.0f, 0.0f, 0.0f);
            for (size_t k = 0; k < n; k++) {
                Quaternion a = A[i * n + k];
                Quaternion b = B[k * p + j];
                Quaternion prod = quaternion_multiply(&a, &b);
                sum = quaternion_add(&sum, &prod);
            }
            C[i * p + j] = sum;
        }
    }
}

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
void fft_real(const float* input, int n, float* real_out, float* imag_out) {
    if (!input || !real_out || !imag_out || n <= 0) {
        if (real_out && imag_out) {
            for (int i = 0; i < n; i++) {
                real_out[i] = 0.0f;
                imag_out[i] = 0.0f;
            }
        }
        return;
    }

    memcpy(real_out, input, n * sizeof(float));
    memset(imag_out, 0, n * sizeof(float));

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;

        if (i < j) {
            float temp_real = real_out[i];
            real_out[i] = real_out[j];
            real_out[j] = temp_real;

            float temp_imag = imag_out[i];
            imag_out[i] = imag_out[j];
            imag_out[j] = temp_imag;
        }
    }

    for (int length = 2; length <= n; length <<= 1) {
        int half_length = length >> 1;

        float angle_step = -2.0f * (float)M_PI / length;

        for (int i = 0; i < n; i += length) {
            float w_real = 1.0f;
            float w_imag = 0.0f;

            float w_real_step = cosf(angle_step);
            float w_imag_step = sinf(angle_step);

            for (int j = 0; j < half_length; j++) {
                int even_index = i + j;
                int odd_index = even_index + half_length;

                float odd_real_rotated = w_real * real_out[odd_index] - w_imag * imag_out[odd_index];
                float odd_imag_rotated = w_real * imag_out[odd_index] + w_imag * real_out[odd_index];

                float even_real_temp = real_out[even_index];
                float even_imag_temp = imag_out[even_index];

                real_out[even_index] = even_real_temp + odd_real_rotated;
                imag_out[even_index] = even_imag_temp + odd_imag_rotated;

                real_out[odd_index] = even_real_temp - odd_real_rotated;
                imag_out[odd_index] = even_imag_temp - odd_imag_rotated;

                float w_real_new = w_real * w_real_step - w_imag * w_imag_step;
            float w_imag_new = w_real * w_imag_step + w_imag * w_real_step;
            w_real = w_real_new;
            w_imag = w_imag_new;
        }
    }
}
}

/* ============================================================================
 * 双四元数运算实现（6-DOF空间变换）
 * ========================================================================= */

DualQuaternion dual_quaternion_identity(void) {
    DualQuaternion dq;
    dq.real = quaternion_identity();
    dq.dual = quaternion_create(0.0f, 0.0f, 0.0f, 0.0f);
    return dq;
}

DualQuaternion dual_quaternion_from_rotation_translation(const Quaternion* rotation, const float* translation) {
    DualQuaternion dq;
    dq.real = quaternion_normalize(rotation);

    Quaternion t_quat = quaternion_create(0.0f, translation[0], translation[1], translation[2]);
    Quaternion half_times_q = quaternion_multiply(&t_quat, &dq.real);
    dq.dual = quaternion_scale(&half_times_q, 0.5f);

    return dq;
}

void dual_quaternion_to_rotation_translation(const DualQuaternion* dq, Quaternion* rotation, float* translation) {
    if (!rotation || !translation) return;

    *rotation = quaternion_normalize(&dq->real);

    Quaternion t_times_q = quaternion_scale(&dq->dual, 2.0f);
    Quaternion q_conj = quaternion_conjugate(&dq->real);
    Quaternion translation_quat = quaternion_multiply(&t_times_q, &q_conj);

    translation[0] = translation_quat.x;
    translation[1] = translation_quat.y;
    translation[2] = translation_quat.z;
}

DualQuaternion dual_quaternion_multiply(const DualQuaternion* a, const DualQuaternion* b) {
    DualQuaternion result;
    result.real = quaternion_multiply(&a->real, &b->real);
    Quaternion a_real_b_dual = quaternion_multiply(&a->real, &b->dual);
    Quaternion a_dual_b_real = quaternion_multiply(&a->dual, &b->real);
    result.dual = quaternion_add(&a_real_b_dual, &a_dual_b_real);
    return result;
}

DualQuaternion dual_quaternion_conjugate(const DualQuaternion* dq) {
    DualQuaternion result;
    result.real = quaternion_conjugate(&dq->real);
    Quaternion dual_conj = quaternion_conjugate(&dq->dual);
    result.dual = quaternion_scale(&dual_conj, -1.0f);
    return result;
}

void dual_quaternion_transform_point(const DualQuaternion* dq, const float* point, float* result) {
    if (!dq || !point || !result) return;

    Quaternion p_quat = quaternion_create(0.0f, point[0], point[1], point[2]);
    DualQuaternion p_dq;
    p_dq.real = quaternion_create(1.0f, 0.0f, 0.0f, 0.0f);
    p_dq.dual = p_quat;

    DualQuaternion dq_conj = dual_quaternion_conjugate(dq);
    DualQuaternion temp = dual_quaternion_multiply(dq, &p_dq);
    DualQuaternion transformed = dual_quaternion_multiply(&temp, &dq_conj);

    result[0] = transformed.dual.x;
    result[1] = transformed.dual.y;
    result[2] = transformed.dual.z;
}

float dual_quaternion_norm(const DualQuaternion* dq) {
    float real_norm = quaternion_norm(&dq->real);
    if (real_norm < 1e-12f) return 0.0f;

    float dot_rd = dq->real.w * dq->dual.w + dq->real.x * dq->dual.x
                 + dq->real.y * dq->dual.y + dq->real.z * dq->dual.z;
    return real_norm + dot_rd / real_norm;
}

DualQuaternion dual_quaternion_normalize(const DualQuaternion* dq) {
    DualQuaternion result;
    float real_norm = quaternion_norm(&dq->real);

    if (real_norm < 1e-12f) {
        result.real = quaternion_identity();
        result.dual = quaternion_create(0.0f, 0.0f, 0.0f, 0.0f);
        return result;
    }

    result.real = quaternion_normalize(&dq->real);
    float inv_real_norm = 1.0f / real_norm;
    Quaternion scaled_dual = quaternion_scale(&dq->dual, inv_real_norm);
    float dot_rd = dq->real.w * dq->dual.w + dq->real.x * dq->dual.x
                 + dq->real.y * dq->dual.y + dq->real.z * dq->dual.z;
    float correction = dot_rd / (real_norm * real_norm);
    Quaternion correction_q = quaternion_scale(&result.real, -correction);
    result.dual = quaternion_add(&scaled_dual, &correction_q);

    return result;
}

void dual_quaternion_to_matrix(const DualQuaternion* dq, float* matrix_4x4) {
    if (!dq || !matrix_4x4) return;

    for (int i = 0; i < 16; i++) matrix_4x4[i] = 0.0f;

    Quaternion q = dq->real;
    float w = q.w, x = q.x, y = q.y, z = q.z;
    float trans[3];
    dual_quaternion_to_rotation_translation(dq, NULL, trans);
    float tx = trans[0], ty = trans[1], tz = trans[2];

    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    matrix_4x4[0]  = 1.0f - 2.0f * (yy + zz);
    matrix_4x4[1]  = 2.0f * (xy + wz);
    matrix_4x4[2]  = 2.0f * (xz - wy);
    matrix_4x4[3]  = 0.0f;
    matrix_4x4[4]  = 2.0f * (xy - wz);
    matrix_4x4[5]  = 1.0f - 2.0f * (xx + zz);
    matrix_4x4[6]  = 2.0f * (yz + wx);
    matrix_4x4[7]  = 0.0f;
    matrix_4x4[8]  = 2.0f * (xz + wy);
    matrix_4x4[9]  = 2.0f * (yz - wx);
    matrix_4x4[10] = 1.0f - 2.0f * (xx + yy);
    matrix_4x4[11] = 0.0f;
    matrix_4x4[12] = tx;
    matrix_4x4[13] = ty;
    matrix_4x4[14] = tz;
    matrix_4x4[15] = 1.0f;
}

DualQuaternion dual_quaternion_from_matrix(const float* matrix_4x4) {
    DualQuaternion dq;
    if (!matrix_4x4) return dual_quaternion_identity();

    float m00 = matrix_4x4[0], m01 = matrix_4x4[1], m02 = matrix_4x4[2];
    float m10 = matrix_4x4[4], m11 = matrix_4x4[5], m12 = matrix_4x4[6];
    float m20 = matrix_4x4[8], m21 = matrix_4x4[9], m22 = matrix_4x4[10];
    float tx = matrix_4x4[12], ty = matrix_4x4[13], tz = matrix_4x4[14];

    float trace = m00 + m11 + m22;
    if (trace > 0.0f) {
        float s = 0.5f / sqrtf(trace + 1.0f);
        dq.real.w = 0.25f / s;
        dq.real.x = (m21 - m12) * s;
        dq.real.y = (m02 - m20) * s;
        dq.real.z = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
        float s = 2.0f * sqrtf(1.0f + m00 - m11 - m22);
        dq.real.w = (m21 - m12) / s;
        dq.real.x = 0.25f * s;
        dq.real.y = (m01 + m10) / s;
        dq.real.z = (m02 + m20) / s;
    } else if (m11 > m22) {
        float s = 2.0f * sqrtf(1.0f + m11 - m00 - m22);
        dq.real.w = (m02 - m20) / s;
        dq.real.x = (m01 + m10) / s;
        dq.real.y = 0.25f * s;
        dq.real.z = (m12 + m21) / s;
    } else {
        float s = 2.0f * sqrtf(1.0f + m22 - m00 - m11);
        dq.real.w = (m10 - m01) / s;
        dq.real.x = (m02 + m20) / s;
        dq.real.y = (m12 + m21) / s;
        dq.real.z = 0.25f * s;
    }
    dq.real = quaternion_normalize(&dq.real);

    Quaternion t_quat = quaternion_create(0.0f, tx * 0.5f, ty * 0.5f, tz * 0.5f);
    dq.dual = quaternion_multiply(&t_quat, &dq.real);

    return dq;
}

DualQuaternion dual_quaternion_sclerp(const DualQuaternion* a, const DualQuaternion* b, float t) {
    if (t <= 0.0f) return *a;
    if (t >= 1.0f) return *b;

    float real_dot = quaternion_dot(&a->real, &b->real);
    float dual_dot = a->real.w * b->dual.w + a->real.x * b->dual.x
                   + a->real.y * b->dual.y + a->real.z * b->dual.z
                   + a->dual.w * b->real.w + a->dual.x * b->real.x
                   + a->dual.y * b->real.y + a->dual.z * b->real.z;

    DualQuaternion b_adj = *b;
    if (real_dot < 0.0f) {
        b_adj.real = quaternion_scale(&b_adj.real, -1.0f);
        b_adj.dual = quaternion_scale(&b_adj.dual, -1.0f);
        real_dot = -real_dot;
    }

    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, real_dot)));
    float sin_theta = sinf(theta);

    DualQuaternion result;
    if (sin_theta > 1e-6f) {
        float s0 = sinf((1.0f - t) * theta) / sin_theta;
        float s1 = sinf(t * theta) / sin_theta;

        float ds0 = -(1.0f - t) * cosf((1.0f - t) * theta) * theta / sin_theta
                  - s0 * cosf(theta) / sin_theta;
        float ds1 = t * cosf(t * theta) * theta / sin_theta
                  - s1 * cosf(theta) / sin_theta;
        ds0 *= dual_dot / (real_dot * real_dot - 1.0f);
        ds1 *= dual_dot / (real_dot * real_dot - 1.0f);

        Quaternion a_real_s0 = quaternion_scale(&a->real, s0);
        Quaternion b_real_s1 = quaternion_scale(&b_adj.real, s1);
        result.real = quaternion_add(&a_real_s0, &b_real_s1);

        Quaternion a_dual_s0 = quaternion_scale(&a->dual, s0);
        Quaternion b_dual_s1 = quaternion_scale(&b_adj.dual, s1);
        Quaternion dual_sum = quaternion_add(&a_dual_s0, &b_dual_s1);

        Quaternion a_real_ds0 = quaternion_scale(&a->real, ds0);
        Quaternion b_real_ds1 = quaternion_scale(&b_adj.real, ds1);
        Quaternion adj = quaternion_add(&a_real_ds0, &b_real_ds1);
        result.dual = quaternion_add(&dual_sum, &adj);
    } else {
        float s0 = 1.0f - t;
        float s1 = t;
        Quaternion a_real_s0 = quaternion_scale(&a->real, s0);
        Quaternion b_real_s1 = quaternion_scale(&b_adj.real, s1);
        result.real = quaternion_add(&a_real_s0, &b_real_s1);

        Quaternion a_dual_s0 = quaternion_scale(&a->dual, s0);
        Quaternion b_dual_s1 = quaternion_scale(&b_adj.dual, s1);
        result.dual = quaternion_add(&a_dual_s0, &b_dual_s1);
    }
    result.real = quaternion_normalize(&result.real);

    return result;
}

/**
 * @brief 四元数均匀分布随机初始化
 */
Quaternion quaternion_random_uniform(float min, float max) {
    Quaternion q;
    float range = max - min;
    q.w = min + rng_uniform(0.0f, 1.0f) * range;
    q.x = min + rng_uniform(0.0f, 1.0f) * range;
    q.y = min + rng_uniform(0.0f, 1.0f) * range;
    q.z = min + rng_uniform(0.0f, 1.0f) * range;
    return q;
}

/**
 * @brief 四元数Xavier初始化
 */
Quaternion quaternion_xavier_init(size_t input_quats, size_t output_quats) {
    float limit = sqrtf(6.0f / (float)(input_quats + output_quats));
    return quaternion_random_uniform(-limit, limit);
}

/**
 * @brief 四元数He初始化
 */
Quaternion quaternion_he_init(size_t input_quats) {
    float limit = sqrtf(12.0f / (float)input_quats);
    return quaternion_random_uniform(-limit, limit);
}

/**
 * @brief 四元数正交初始化（单位范数）
 */
Quaternion quaternion_orthogonal_init(void) {
    float u1 = rng_uniform(0.0f, 1.0f);
    float u2 = rng_uniform(0.0f, 1.0f);
    float u3 = rng_uniform(0.0f, 1.0f);
    float theta1 = 2.0f * 3.14159265358979323846f * u1;
    float theta2 = 2.0f * 3.14159265358979323846f * u2;
    float c1 = cosf(theta1), s1 = sinf(theta1);
    return quaternion_create(c1 * sqrtf(1.0f - u3),
                             s1 * sqrtf(1.0f - u3),
                             cosf(theta2) * sqrtf(u3),
                             sinf(theta2) * sqrtf(u3));
}

/**
 * @brief 批量初始化四元数权重数组
 */
void quaternion_init_weights(Quaternion* weights, size_t count,
                             size_t input_quats, size_t output_quats,
                             const QuatWeightInitConfig* config) {
    if (!weights || count == 0 || !config) return;

    float scale = (config->scale > 0.0f) ? config->scale : 1.0f;
    if (config->seed != 0) {
        rng_seed((uint64_t)config->seed);
    }

    for (size_t i = 0; i < count; i++) {
        switch (config->method) {
            case QUAT_INIT_UNIFORM:
                weights[i] = quaternion_random_uniform(-0.1f * scale, 0.1f * scale);
                break;
            case QUAT_INIT_XAVIER:
                weights[i] = quaternion_xavier_init(input_quats, output_quats);
                weights[i] = quaternion_scale(&weights[i], scale);
                break;
            case QUAT_INIT_HE:
                weights[i] = quaternion_he_init(input_quats);
                weights[i] = quaternion_scale(&weights[i], scale);
                break;
            case QUAT_INIT_ORTHOGONAL:
                weights[i] = quaternion_orthogonal_init();
                weights[i] = quaternion_scale(&weights[i], scale);
                break;
            case QUAT_INIT_UNIT_NORM:
                weights[i] = quaternion_orthogonal_init();
                weights[i] = quaternion_scale(&weights[i], scale);
                break;
            default:
                weights[i] = quaternion_random_uniform(-0.1f, 0.1f);
                break;
        }
    }
}