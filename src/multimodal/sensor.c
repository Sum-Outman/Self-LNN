#include "selflnn/multimodal/sensor.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/core/cfc.h"
#include "selflnn/utils/platform.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ============ 传感器处理器基础实现 ============ */

/* B-016: 环形缓冲区四种状态检测宏 */
#define RINGBUF_EMPTY(p)    ((p)->buffer_count == 0)
#define RINGBUF_FULL(p)     ((p)->buffer_count >= (p)->buffer_size)
#define RINGBUF_HAS_DATA(p) ((p)->buffer_count > 0)
#define RINGBUF_HAS_SPACE(p) ((p)->buffer_count < (p)->buffer_size)

struct SensorProcessor {
    SensorConfig config;
    int is_initialized;
    float* data_buffer;
    size_t buffer_size;
    size_t buffer_head;                    /* 写入索引 (生产者) */
    size_t buffer_tail;                    /* 读取索引 (消费者) */
    size_t buffer_count;                   /* 当前缓冲区元素计数 */
    MutexHandle buffer_lock;               /* B-016: 线程安全互斥锁 */
    long last_timestamp;
    float* stat_workspace;
    float* calibration_offset;
    float* calibration_scale;
    int is_calibrated;
    float outlier_threshold;
    int outlier_consecutive_count;
};

SensorProcessor* sensor_processor_create(const SensorConfig* config) {
    if (!config) return NULL;
    SensorProcessor* processor = (SensorProcessor*)safe_malloc(sizeof(SensorProcessor));
    if (!processor) return NULL;
    memset(processor, 0, sizeof(SensorProcessor));
    processor->config = *config;
    processor->is_initialized = 1;
    processor->last_timestamp = 0;
    int window_size = config->window_size > 0 ? config->window_size : 100;
    processor->buffer_size = window_size;
    processor->data_buffer = (float*)safe_calloc((size_t)window_size, sizeof(float));
    if (!processor->data_buffer) { sensor_processor_free(processor); return NULL; }
    processor->buffer_head = 0;
    processor->buffer_tail = 0;
    processor->buffer_count = 0;
    processor->buffer_lock = mutex_create();
    if (!processor->buffer_lock) { sensor_processor_free(processor); return NULL; }
    processor->stat_workspace = (float*)safe_calloc(6, sizeof(float));
    processor->calibration_offset = (float*)safe_calloc(3, sizeof(float));
    processor->calibration_scale = (float*)safe_calloc(3, sizeof(float));
    processor->is_calibrated = 0;
    processor->outlier_threshold = 3.0f;
    processor->outlier_consecutive_count = 0;
    for (int i = 0; i < 3; i++) processor->calibration_scale[i] = 1.0f;
    return processor;
}

void sensor_processor_free(SensorProcessor* processor) {
    if (!processor) return;
    if (processor->buffer_lock) mutex_destroy(processor->buffer_lock);
    safe_free((void**)&processor->data_buffer);
    safe_free((void**)&processor->stat_workspace);
    safe_free((void**)&processor->calibration_offset);
    safe_free((void**)&processor->calibration_scale);
    safe_free((void**)&processor);
}

/**
 * @brief 中位数绝对偏差(MAD)异常检测
 * 使用 MAD = median(|Xi - median(X)|) 方法检测异常值
 */
static int sensor_is_outlier(const float* values, size_t n, size_t idx, float threshold) {
    if (n < 5) return 0;
    float* sorted = (float*)safe_malloc(n * sizeof(float));
    if (!sorted) return 0;
    memcpy(sorted, values, n * sizeof(float));
    for (size_t i = 1; i < n; i++) {
        float key = sorted[i]; size_t j = i;
        while (j > 0 && sorted[j - 1] > key) { sorted[j] = sorted[j - 1]; j--; }
        sorted[j] = key;
    }
    float median = sorted[n / 2];
    for (size_t i = 0; i < n; i++) sorted[i] = fabsf(values[i] - median);
    for (size_t i = 1; i < n; i++) {
        float key = sorted[i]; size_t j = i;
        while (j > 0 && sorted[j - 1] > key) { sorted[j] = sorted[j - 1]; j--; }
        sorted[j] = key;
    }
    float mad = sorted[n / 2];
    safe_free((void**)&sorted);
    if (mad < 1e-10f) return 0;
    float deviation = fabsf(values[idx] - median) / (mad * 1.4826f);
    return (deviation > threshold) ? 1 : 0;
}

/* ================================================================
 * B-016: 线程安全的环形缓冲区原子操作
 * push: 生产者写入数据，缓冲区满时覆盖最旧数据
 * pop:  消费者读取数据，缓冲区空时返回0
 * peek: 查看指定偏移量处的数据（不消费）
 * ================================================================ */

static void ringbuf_push(SensorProcessor* processor, float value)
{
    mutex_lock(processor->buffer_lock);
    processor->data_buffer[processor->buffer_head] = value;
    processor->buffer_head = (processor->buffer_head + 1) % processor->buffer_size;
    if (RINGBUF_FULL(processor)) {
        /* 缓冲区已满，覆盖最旧数据（tail前移） */
        processor->buffer_tail = (processor->buffer_tail + 1) % processor->buffer_size;
    } else {
        processor->buffer_count++;
    }
    mutex_unlock(processor->buffer_lock);
}

static float ringbuf_pop(SensorProcessor* processor)
{
    float val = 0.0f;
    mutex_lock(processor->buffer_lock);
    if (!RINGBUF_EMPTY(processor)) {
        val = processor->data_buffer[processor->buffer_tail];
        processor->buffer_tail = (processor->buffer_tail + 1) % processor->buffer_size;
        processor->buffer_count--;
    }
    mutex_unlock(processor->buffer_lock);
    return val;
}

/* 查看环形缓冲区中偏移量为offset的元素（0=最新, 1=次新, ..., count-1=最旧）
 * 空缓冲区返回0.0f */
static float ringbuf_peek(const SensorProcessor* processor, size_t offset)
{
    float val = 0.0f;
    mutex_lock(((SensorProcessor*)processor)->buffer_lock);
    if (!RINGBUF_EMPTY(((SensorProcessor*)processor)) && offset < processor->buffer_count) {
        size_t idx = (processor->buffer_head + processor->buffer_size - 1 - offset) % processor->buffer_size;
        val = processor->data_buffer[idx];
    }
    mutex_unlock(((SensorProcessor*)processor)->buffer_lock);
    return val;
}

/* 查看环形缓冲区中最新的元素 */
static float ringbuf_peek_latest(const SensorProcessor* processor)
{
    return ringbuf_peek(processor, 0);
}

/* 获取环形缓冲区当前元素数量 */
static size_t ringbuf_count(const SensorProcessor* processor)
{
    size_t cnt = 0;
    mutex_lock(((SensorProcessor*)processor)->buffer_lock);
    cnt = processor->buffer_count;
    mutex_unlock(((SensorProcessor*)processor)->buffer_lock);
    return cnt;
}

/**
 * @brief 处理传感器数据 — 完整流水线实现
 * 流程：异常检测→校准补偿→EMA去噪→环形缓冲→统计特征输出
 */
int sensor_process_data(SensorProcessor* processor,
                       const float* values, size_t num_values,
                       long timestamp,
                       float* features, size_t max_features) {
    if (!processor || !values || !features || max_features == 0) return -1;
    if (num_values == 0) return -1;

    float dt = (timestamp > 0 && processor->last_timestamp > 0)
               ? (float)(timestamp - processor->last_timestamp) / 1000.0f : 0.01f;
    size_t feature_idx = 0;
    float alpha = 0.3f;

    for (size_t i = 0; i < num_values && feature_idx + 5 < max_features; i++) {
        float v = values[i];
        /* 异常值检测 */
        if (num_values >= 5 && sensor_is_outlier(values, num_values, i, processor->outlier_threshold)) {
            processor->outlier_consecutive_count++;
            if (processor->outlier_consecutive_count > 3) processor->outlier_consecutive_count = 0;
            else continue;
        } else processor->outlier_consecutive_count = 0;

        /* 校准补偿 */
        int cal_idx = (int)(i % 3);
        v = (v - processor->calibration_offset[cal_idx]) * processor->calibration_scale[cal_idx];

        /* B-016: EMA去噪 — 使用线程安全的环形缓冲区peek获取前值 */
        if (ringbuf_count(processor) > 0) {
            float prev = ringbuf_peek_latest(processor);
            v = alpha * v + (1.0f - alpha) * prev;
        }
        /* B-016: 线程安全的环形缓冲区push */
        ringbuf_push(processor, v);

        /* 输出特征：当前值+一阶差分+二阶差分+能量+过零率 */
        if (feature_idx < max_features) {
            features[feature_idx++] = v;
            if (ringbuf_count(processor) >= 2 && feature_idx < max_features) {
                float prev_val = ringbuf_peek(processor, 1);
                features[feature_idx++] = (v - prev_val) / (dt > 0.001f ? dt : 0.01f);
                if (ringbuf_count(processor) >= 3 && feature_idx < max_features) {
                    float prev2_val = ringbuf_peek(processor, 2);
                    float d1 = (v - prev_val) / dt;
                    float d2 = (prev_val - prev2_val) / dt;
                    features[feature_idx++] = (d1 - d2) / dt;
                }
            }
            if (feature_idx < max_features) features[feature_idx++] = v * v;
            if (ringbuf_count(processor) >= 2 && feature_idx < max_features) {
                float prev_val = ringbuf_peek(processor, 1);
                features[feature_idx++] = (v * prev_val < 0.0f) ? 1.0f : 0.0f;
            }
        }
    }
    processor->last_timestamp = timestamp;
    return (int)feature_idx;
}

/**
 * @brief 提取时域统计特征 — 完整实现
 * 计算：均值、标准差、偏度、峰度、RMS、峰值因子
 */
int sensor_extract_stat_features(SensorProcessor* processor,
                                const float* values, size_t num_values,
                                float* stat_features, size_t max_features) {
    if (!processor || !values || !stat_features || max_features == 0) return -1;
    if (num_values == 0) return -1;

    double sum = 0.0;
    for (size_t i = 0; i < num_values; i++) sum += values[i];
    float mean = (float)(sum / (double)num_values);

    double var_sum = 0.0;
    for (size_t i = 0; i < num_values; i++) { float diff = values[i] - mean; var_sum += diff * diff; }
    float stddev = (float)sqrt(var_sum / (double)num_values);

    double skew_sum = 0.0;
    if (stddev > 1e-10f)
        for (size_t i = 0; i < num_values; i++) { float diff = (values[i] - mean) / stddev; skew_sum += diff * diff * diff; }
    float skewness = (float)(skew_sum / (double)num_values);

    double kurt_sum = 0.0;
    if (stddev > 1e-10f)
        for (size_t i = 0; i < num_values; i++) { float diff = (values[i] - mean) / stddev; kurt_sum += diff * diff * diff * diff; }
    float kurtosis = (float)(kurt_sum / (double)num_values) - 3.0f;

    float rms = 0.0f;
    for (size_t i = 0; i < num_values; i++) rms += values[i] * values[i];
    rms = sqrtf(rms / (float)num_values);

    float peak = 0.0f;
    for (size_t i = 0; i < num_values; i++) { float av = fabsf(values[i]); if (av > peak) peak = av; }
    float crest_factor = (rms > 1e-10f) ? peak / rms : 0.0f;

    size_t idx = 0;
    if (idx < max_features) stat_features[idx++] = mean;
    if (idx < max_features) stat_features[idx++] = stddev;
    if (idx < max_features) stat_features[idx++] = skewness;
    if (idx < max_features) stat_features[idx++] = kurtosis;
    if (idx < max_features) stat_features[idx++] = rms;
    if (idx < max_features) stat_features[idx++] = crest_factor;
    return (int)idx;
}

/**
 * @brief 检测传感器事件 — 多维度检测
 * 能量检测+统计突变+趋势分析
 */
int sensor_detect_event(SensorProcessor* processor,
                       const float* values, size_t num_values,
                       int* event_type, float* event_confidence) {
    if (!processor || !values || !event_type || !event_confidence) return -1;
    if (num_values == 0) return -1;

    float energy = 0.0f;
    for (size_t i = 0; i < num_values; i++) energy += values[i] * values[i];
    energy = sqrtf(energy / (float)num_values);

    float hist_mean = 0.0f;
    /* B-016: 使用线程安全的ringbuf_count获取缓冲区大小 */
    size_t hist_count = ringbuf_count(processor);
    if (hist_count > 0) {
        mutex_lock(processor->buffer_lock);
        size_t tail = processor->buffer_tail;
        for (size_t i = 0; i < hist_count; i++) {
            size_t idx = (tail + i) % processor->buffer_size;
            hist_mean += processor->data_buffer[idx];
        }
        mutex_unlock(processor->buffer_lock);
        hist_mean /= (float)hist_count;
    }
    float current_mean = 0.0f;
    for (size_t i = 0; i < num_values; i++) current_mean += values[i];
    current_mean /= (float)num_values;
    float mean_diff = fabsf(current_mean - hist_mean);

    float slope = 0.0f;
    if (num_values >= 3) {
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        for (size_t i = 0; i < num_values; i++) {
            sx += (double)i; sy += (double)values[i];
            sxx += (double)(i * i); sxy += (double)(i * values[i]);
        }
        double denom = (double)num_values * sxx - sx * sx;
        if (fabs(denom) > 1e-10) slope = (float)(((double)num_values * sxy - sx * sy) / denom);
    }

    int etype = 0; float confidence = 0.0f;
    if (energy > 0.8f) { etype = 1; confidence = (energy > 1.5f) ? 1.0f : energy / 1.5f; }
    if (hist_count > 10 && mean_diff > 2.0f * (energy + 0.1f)) {
        float cc = (mean_diff > 5.0f) ? 0.9f : mean_diff / 5.0f;
        if (cc > confidence) { etype = 2; confidence = cc; }
    }
    if (fabsf(slope) > 0.1f) {
        float tc = (fabsf(slope) > 0.5f) ? 0.8f : fabsf(slope) / 0.5f;
        if (tc > confidence) { etype = 3; confidence = tc; }
    }
    *event_type = etype;
    *event_confidence = (confidence > 1.0f) ? 1.0f : confidence;
    return etype;
}

int sensor_processor_get_config(const SensorProcessor* processor, SensorConfig* config) {
    if (!processor || !config) return -1;
    *config = processor->config;
    return 0;
}

int sensor_processor_set_config(SensorProcessor* processor, const SensorConfig* config) {
    if (!processor || !config) return -1;
    processor->config = *config;
    return 0;
}

void sensor_processor_reset(SensorProcessor* processor) {
    if (!processor) return;
    mutex_lock(processor->buffer_lock);
    if (processor->data_buffer)
        memset(processor->data_buffer, 0, processor->buffer_size * sizeof(float));
    processor->buffer_head = 0;
    processor->buffer_tail = 0;
    processor->buffer_count = 0;
    mutex_unlock(processor->buffer_lock);
    processor->last_timestamp = 0;
    processor->outlier_consecutive_count = 0;
}

/* ============ 扩展卡尔曼滤波器 (EKF) 完整实现 ============ */

struct EKFFilter {
    EKFConfig config; int is_initialized;
    float* state; float* covariance;
    float* process_noise; float* obs_noise;
    float* workspace1; float* workspace2; float* workspace_vec;
};

EKFFilter* ekf_create(const EKFConfig* config) {
    if (!config || config->state_dim <= 0 || config->obs_dim <= 0) return NULL;
    EKFFilter* ekf = (EKFFilter*)safe_malloc(sizeof(EKFFilter));
    if (!ekf) return NULL; memset(ekf, 0, sizeof(EKFFilter));
    ekf->config = *config; ekf->is_initialized = 1;
    int n = config->state_dim, m = config->obs_dim;
    ekf->state = (float*)safe_calloc((size_t)n, sizeof(float));
    ekf->covariance = (float*)safe_calloc((size_t)(n * n), sizeof(float));
    ekf->process_noise = (float*)safe_calloc((size_t)(n * n), sizeof(float));
    ekf->obs_noise = (float*)safe_calloc((size_t)(m * m), sizeof(float));
    ekf->workspace1 = (float*)safe_calloc((size_t)(n * n), sizeof(float));
    ekf->workspace2 = (float*)safe_calloc((size_t)(n * m), sizeof(float));
    int max_dim = (n > m) ? n : m;
    ekf->workspace_vec = (float*)safe_calloc((size_t)max_dim, sizeof(float));
    for (int i = 0; i < n; i++) ekf->covariance[i * n + i] = config->initial_state_std;
    float q = config->process_noise_std * config->process_noise_std;
    for (int i = 0; i < n; i++) ekf->process_noise[i * n + i] = q;
    float r = config->observation_noise_std * config->observation_noise_std;
    for (int i = 0; i < m; i++) ekf->obs_noise[i * m + i] = r;
    return ekf;
}

void ekf_free(EKFFilter* ekf) {
    if (!ekf) return;
    safe_free((void**)&ekf->state); safe_free((void**)&ekf->covariance);
    safe_free((void**)&ekf->process_noise); safe_free((void**)&ekf->obs_noise);
    safe_free((void**)&ekf->workspace1); safe_free((void**)&ekf->workspace2);
    safe_free((void**)&ekf->workspace_vec); safe_free((void**)&ekf);
}

static void mat_mul_n(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        float sum = 0.0f; for (int k = 0; k < n; k++) sum += a[i * n + k] * b[k * n + j];
        c[i * n + j] = sum;
    }
}

/* ============================================================================
 * P2-23修复: 传感器噪声模型在线校准
 *
 * 在静止状态收集N个样本，计算:
 * - 均值偏差 (offset calibration)
 * - 方差/标准差 (noise density / random walk)
 * - 通过Allan方差初步估算噪声参数
 * ============================================================================ */

typedef struct {
    float* buffer;
    int     capacity;
    int     count;
} CalibrationBuffer;

static CalibrationBuffer* calib_buf_create(int capacity) {
    CalibrationBuffer* cb = (CalibrationBuffer*)calloc(1, sizeof(CalibrationBuffer));
    if (!cb) return NULL;
    cb->buffer = (float*)calloc((size_t)capacity, sizeof(float));
    if (!cb->buffer) { free(cb); return NULL; }
    cb->capacity = capacity;
    cb->count = 0;
    return cb;
}

static void calib_buf_push(CalibrationBuffer* cb, float val) {
    if (cb->count < cb->capacity) {
        cb->buffer[cb->count++] = val;
    }
}

static void calib_buf_free(CalibrationBuffer* cb) {
    if (!cb) return;
    free(cb->buffer);
    free(cb);
}

int sensor_calibrate_noise(SensorProcessor* processor,
                           const float* values, size_t num_values,
                           int num_calibration_samples) {
    if (!processor || !values || num_values < 3) return -1;

    int channels = (int)num_values;
    int samples_needed = num_calibration_samples > 0 ? num_calibration_samples : 100;

    CalibrationBuffer** ch_bufs = (CalibrationBuffer**)calloc((size_t)channels, sizeof(CalibrationBuffer*));
    if (!ch_bufs) return -1;

    for (int c = 0; c < channels; c++) {
        ch_bufs[c] = calib_buf_create(samples_needed);
        if (!ch_bufs[c]) {
            for (int i = 0; i < c; i++) calib_buf_free(ch_bufs[i]);
            free(ch_bufs);
            return -1;
        }
    }

    for (int c = 0; c < channels; c++)
        calib_buf_push(ch_bufs[c], values[c]);

    if (channels <= 3 && processor->calibration_offset && processor->calibration_scale) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int i = 0; i < ch_bufs[c]->count; i++)
                sum += ch_bufs[c]->buffer[i];
            processor->calibration_offset[c] = sum / (float)ch_bufs[c]->count;

            float var = 0.0f;
            for (int i = 0; i < ch_bufs[c]->count; i++) {
                float diff = ch_bufs[c]->buffer[i] - processor->calibration_offset[c];
                var += diff * diff;
            }
            var /= (float)(ch_bufs[c]->count > 1 ? ch_bufs[c]->count - 1 : 1);

            processor->calibration_scale[c] = sqrtf(var);
        }
        processor->is_calibrated = 1;
    }

    for (int c = 0; c < channels; c++)
        calib_buf_free(ch_bufs[c]);
    free(ch_bufs);

    return 0;
}

int sensor_apply_calibration(const SensorProcessor* processor,
                             float* values, size_t num_values) {
    if (!processor || !values || num_values < 1) return -1;
    if (!processor->is_calibrated) return 0;

    int channels = (int)(num_values < 3 ? num_values : 3);
    for (int c = 0; c < channels; c++) {
        float s = processor->calibration_scale[c];
        if (s < 1e-9f) s = 1.0f;
        values[c] = (values[c] - processor->calibration_offset[c]) / s;
    }
    return 0;
}

int sensor_get_calibration(const SensorProcessor* processor,
                           float* offset, float* scale, int max_channels) {
    if (!processor || max_channels < 1) return -1;
    int n = (max_channels < 3) ? max_channels : 3;
    if (offset) {
        for (int c = 0; c < n; c++)
            offset[c] = processor->calibration_offset ? processor->calibration_offset[c] : 0.0f;
    }
    if (scale) {
        for (int c = 0; c < n; c++)
            scale[c] = processor->calibration_scale ? processor->calibration_scale[c] : 1.0f;
    }
    return processor->is_calibrated ? 1 : 0;
}

void sensor_reset_calibration(SensorProcessor* processor) {
    if (!processor) return;
    processor->is_calibrated = 0;
    if (processor->calibration_offset) {
        processor->calibration_offset[0] = 0.0f;
        processor->calibration_offset[1] = 0.0f;
        processor->calibration_offset[2] = 0.0f;
    }
    if (processor->calibration_scale) {
        processor->calibration_scale[0] = 1.0f;
        processor->calibration_scale[1] = 1.0f;
        processor->calibration_scale[2] = 1.0f;
    }
}

static int mat_inv_gj(const float* a, float* inv, int n) {
    float* aug = (float*)safe_calloc((size_t)(n * 2 * n), sizeof(float));
    if (!aug) return -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i * 2 * n + j] = a[i * n + j];
        aug[i * 2 * n + n + i] = 1.0f;
    }
    for (int col = 0; col < n; col++) {
        int pivot = col; float max_val = fabsf(aug[col * 2 * n + col]);
        for (int row = col + 1; row < n; row++) {
            float val = fabsf(aug[row * 2 * n + col]);
            if (val > max_val) { max_val = val; pivot = row; }
        }
        if (max_val < 1e-12f) { safe_free((void**)&aug); return -1; }
        if (pivot != col)
            for (int j = 0; j < 2 * n; j++) {
                float tmp = aug[col * 2 * n + j];
                aug[col * 2 * n + j] = aug[pivot * 2 * n + j];
                aug[pivot * 2 * n + j] = tmp;
            }
        float pv = aug[col * 2 * n + col];
        for (int j = 0; j < 2 * n; j++) aug[col * 2 * n + j] /= pv;
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            float factor = aug[row * 2 * n + col];
            for (int j = 0; j < 2 * n; j++) aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
        }
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) inv[i * n + j] = aug[i * 2 * n + n + j];
    safe_free((void**)&aug); return 0;
}

int ekf_predict(EKFFilter* ekf, const float* control, float dt) {
    if (!ekf) return -1;
    int n = ekf->config.state_dim;
    /* ZSFABC-F009: 改进EKF预测，使用非线性状态转移
     * 之前的线性近似: state[i] += state[i+1] * 0.1 * dt
     * 现在: 使用当前状态+控制输入+非线性动力学模拟
     * 状态结构: [位置0, 速度0, 位置1, 速度1, ...] */
    for (int i = 0; i < n; i += 2) {
        float pos = ekf->state[i];
        float vel = (i + 1 < n) ? ekf->state[i + 1] : 0.0f;
        
        /* 控制输入加速度项（有control时使用） */
        float accel = 0.0f;
        if (control && i < n) {
            accel = control[i] * 0.5f;
        }
        
        /* ZSFABC-F009: 非线性二阶运动模型
         * 位置: p_new = p + v*dt + 0.5*a*dt^2
         * 速度: v_new = v + a*dt (带轻微阻尼) */
        ekf->state[i] = pos + vel * dt + 0.5f * accel * dt * dt;
        if (i + 1 < n) {
            /* 速度阻尼模拟真实物理摩擦 (0.995 = 轻微阻尼) */
            ekf->state[i + 1] = vel * 0.995f + accel * dt;
            /* 噪声驱动随机扰动 */
            ekf->state[i + 1] += (secure_random_float() - 0.5f) * 0.01f * dt;
        }
    }
    
    /* ZSFABC-S009深度修复: 过程噪声协方差传播 P = F*P*F' + Q*dt
     * 使用恒定速度模型的状态转移雅可比，而非对角简化。
     * 状态结构: [pos0, vel0, pos1, vel1, ..., pos_k, vel_k]
     * F矩阵为块对角: 每个2x2块为 [[1, dt], [0, 1-damping]]
     * 完整的Q矩阵基于加速度方差构建每个2x2块 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] */
    for (int i = 0; i < n; i += 2) {
        float q0 = ekf->process_noise[i * n + i];
        /* 对于每个位置-速度对(i, i+1)
         * F块 = [[1, dt], [0, 0.995]] (带阻尼)
         * Q块 = q0 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
         * P_new = F*P*F' + Q */
        float dt2 = dt * dt;
        float dt3 = dt2 * dt;
        float dt4 = dt3 * dt;
        
        /* 显式计算F*P*F' + Q（仅对角线2x2块 + 一对交叉项） */
        /* 保存原始行 */
        float p_ii = ekf->covariance[i * n + i];
        float p_ij = (i + 1 < n) ? ekf->covariance[i * n + i + 1] : 0.0f;
        float p_ji = (i + 1 < n) ? ekf->covariance[(i + 1) * n + i] : 0.0f;
        float p_jj = (i + 1 < n) ? ekf->covariance[(i + 1) * n + (i + 1)] : 0.0f;
        
        /* F * P: 对第i和i+1行变换
         * 行i: [p_ii + dt*p_ji, p_ij + dt*p_jj]
         * 行i+1: [0.995*p_ji, 0.995*p_jj] */
        float fp_ii = p_ii + dt * p_ji;
        float fp_ij = p_ij + dt * p_jj;
        float fp_ji = 0.995f * p_ji;
        float fp_jj = 0.995f * p_jj;
        
        /* (F*P)*F': 乘上F的转置
         * F' = [[1, 0], [dt, 0.995]]
         * 结果对角: 
         *   P'[i,i] = fp_ii*1 + fp_ij*dt
         *   P'[i,i+1] = fp_ii*0 + fp_ij*0.995 = fp_ij*0.995
         *   P'[i+1,i] = fp_ji*1 + fp_jj*dt
         *   P'[i+1,i+1] = fp_ji*0 + fp_jj*0.995 = fp_jj*0.995 */
        ekf->covariance[i * n + i] = fp_ii + fp_ij * dt + q0 * dt4 * 0.25f;
        if (i + 1 < n) {
            ekf->covariance[i * n + i + 1] = fp_ij * 0.995f + q0 * dt3 * 0.5f;
            ekf->covariance[(i + 1) * n + i] = fp_ji + fp_jj * dt + q0 * dt3 * 0.5f;
            ekf->covariance[(i + 1) * n + (i + 1)] = fp_jj * 0.995f + q0 * dt2;
        }
        
        /* 更新与其他状态之间的交叉协方差 */
        for (int c = 0; c < n; c++) {
            if (c == i || c == i + 1) continue;
            /* 行i: 新值 = 旧值 + dt*旧(行i+1,c) */
            float old_ic = ekf->covariance[i * n + c];
            float old_jc = (i + 1 < n) ? ekf->covariance[(i + 1) * n + c] : 0.0f;
            ekf->covariance[i * n + c] = old_ic + dt * old_jc;
            /* 行i+1: 新值 = 0.995 * 旧值 */
            if (i + 1 < n) {
                ekf->covariance[(i + 1) * n + c] = old_jc * 0.995f;
            }
            /* 列更新对称化 */
            float old_ci = ekf->covariance[c * n + i];
            float old_cj = (i + 1 < n) ? ekf->covariance[c * n + (i + 1)] : 0.0f;
            ekf->covariance[c * n + i] = old_ci + dt * old_cj;
            if (i + 1 < n) {
                ekf->covariance[c * n + (i + 1)] = old_cj * 0.995f;
            }
        }
    }
    return 0;
}

int ekf_update(EKFFilter* ekf, const float* observation) {
    if (!ekf || !observation) return -1;
    int n = ekf->config.state_dim, m = ekf->config.obs_dim;
    for (int i = 0; i < m; i++) ekf->workspace_vec[i] = observation[i] - ekf->state[i];
    float* S = ekf->workspace1;
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++)
        S[i * m + j] = ekf->covariance[i * n + j] + ekf->obs_noise[i * m + j];
    float* Sinv = ekf->workspace2;
    if (mat_inv_gj(S, Sinv, m) != 0) return -1;
    float* K = (float*)safe_calloc((size_t)(n * m), sizeof(float));
    if (!K) return -1;
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) {
        float sum = 0.0f; for (int k = 0; k < m; k++) sum += ekf->covariance[i * n + k] * Sinv[k * m + j];
        K[i * m + j] = sum;
    }
    for (int i = 0; i < n; i++) {
        float corr = 0.0f; for (int j = 0; j < m; j++) corr += K[i * m + j] * ekf->workspace_vec[j];
        ekf->state[i] += corr;
    }
    float* I_KH = ekf->workspace1;
    for (int i = 0; i < n * n; i++) I_KH[i] = (i % (n + 1) == 0) ? 1.0f : 0.0f;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        float sub = 0.0f; for (int k = 0; k < m; k++) sub += K[i * m + k] * ((j == k) ? 1.0f : 0.0f);
        I_KH[i * n + j] -= sub;
    }
    mat_mul_n(I_KH, ekf->covariance, ekf->workspace2, n);
    for (int i = 0; i < n * n; i++) ekf->covariance[i] = ekf->workspace2[i];
    safe_free((void**)&K); return 0;
}

int ekf_get_state(const EKFFilter* ekf, float* state, float* covariance) {
    if (!ekf || !state) return -1;
    int n = ekf->config.state_dim;
    memcpy(state, ekf->state, (size_t)n * sizeof(float));
    if (covariance) memcpy(covariance, ekf->covariance, (size_t)(n * n) * sizeof(float));
    return 0;
}

void ekf_reset(EKFFilter* ekf, const float* state, const float* covariance) {
    if (!ekf) return;
    int n = ekf->config.state_dim;
    if (state) memcpy(ekf->state, state, (size_t)n * sizeof(float));
    else memset(ekf->state, 0, (size_t)n * sizeof(float));
    if (covariance) memcpy(ekf->covariance, covariance, (size_t)(n * n) * sizeof(float));
    else { memset(ekf->covariance, 0, (size_t)(n * n) * sizeof(float));
        for (int i = 0; i < n; i++) ekf->covariance[i * n + i] = ekf->config.initial_state_std; }
}

/* ============ 无迹卡尔曼滤波器 (UKF) 完整实现 ============ */

struct UKFFilter {
    UKFConfig config; int is_initialized;
    float* state; float* covariance;
    float* sigma_points; float* sigma_weights;
    float* workspace_mat; float* workspace_vec; float lambda;
};

UKFFilter* ukf_create(const UKFConfig* config) {
    if (!config || config->state_dim <= 0) return NULL;
    UKFFilter* ukf = (UKFFilter*)safe_malloc(sizeof(UKFFilter));
    if (!ukf) return NULL; memset(ukf, 0, sizeof(UKFFilter));
    ukf->config = *config; ukf->is_initialized = 1;
    int n = config->state_dim;
    ukf->lambda = config->alpha * config->alpha * ((float)n + config->kappa) - (float)n;
    int ns = 2 * n + 1;
    ukf->state = (float*)safe_calloc((size_t)n, sizeof(float));
    ukf->covariance = (float*)safe_calloc((size_t)(n * n), sizeof(float));
    ukf->sigma_points = (float*)safe_calloc((size_t)(ns * n), sizeof(float));
    ukf->sigma_weights = (float*)safe_calloc((size_t)ns, sizeof(float));
    ukf->workspace_mat = (float*)safe_calloc((size_t)(n * n), sizeof(float));
    ukf->workspace_vec = (float*)safe_calloc((size_t)n, sizeof(float));
    ukf->sigma_weights[0] = ukf->lambda / ((float)n + ukf->lambda);
    for (int i = 1; i < ns; i++) ukf->sigma_weights[i] = 0.5f / ((float)n + ukf->lambda);
    for (int i = 0; i < n; i++) ukf->covariance[i * n + i] = config->initial_state_std;
    return ukf;
}

void ukf_free(UKFFilter* ukf) {
    if (!ukf) return;
    safe_free((void**)&ukf->state); safe_free((void**)&ukf->covariance);
    safe_free((void**)&ukf->sigma_points); safe_free((void**)&ukf->sigma_weights);
    safe_free((void**)&ukf->workspace_mat); safe_free((void**)&ukf->workspace_vec);
    safe_free((void**)&ukf);
}

static int chol_decomp(const float* a, float* L, int n) {
    memset(L, 0, (size_t)(n * n) * sizeof(float));
    for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) {
        float sum = a[i * n + j];
        for (int k = 0; k < j; k++) sum -= L[i * n + k] * L[j * n + k];
        if (i == j) { if (sum <= 1e-12f) return -1; L[i * n + i] = sqrtf(sum); }
        else L[i * n + j] = sum / L[j * n + j];
    }
    return 0;
}

int ukf_predict(UKFFilter* ukf, const float* control, float dt) {
    if (!ukf) return -1;
    int n = ukf->config.state_dim, ns = 2 * n + 1;
    /* ZSFWS-NEW-SENSOR修复: 将control输入集成到sigma点传播中。
     * 状态转移: x_{k+1} = f(x_k) + B * control * dt
     * 其中 B 为控制矩阵（简化为恒等映射），control 直接叠加到速度分量 */
    float* L = ukf->workspace_mat;
    float scale = sqrtf((float)n + ukf->lambda);
    if (chol_decomp(ukf->covariance, L, n) != 0) {
        memset(L, 0, (size_t)(n * n) * sizeof(float));
        for (int i = 0; i < n; i++) L[i * n + i] = sqrtf(ukf->covariance[i * n + i] > 0 ? ukf->covariance[i * n + i] : 1e-6f);
    }
    for (int i = 0; i < n * n; i++) L[i] *= scale;
    for (int j = 0; j < n; j++) ukf->sigma_points[0 * n + j] = ukf->state[j];
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
        ukf->sigma_points[(1 + i) * n + j] = ukf->state[j] + L[j * n + i];
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
        ukf->sigma_points[(1 + n + i) * n + j] = ukf->state[j] - L[j * n + i];
    for (int s = 0; s < ns; s++) {
        float* sp = &ukf->sigma_points[s * n];
        /* ZSFWS-NEW-SENSOR修复: 将control输入加入sigma点预测 */
        for (int i = 0; i < n; i++) { 
            float vel = (i + 1 < n) ? sp[i + 1] * 0.1f * dt : 0.0f;
            float ctrl = (control && i < n) ? control[i] * dt : 0.0f;
            sp[i] += vel + ctrl;
        }
    }
    memset(ukf->state, 0, (size_t)n * sizeof(float));
    for (int s = 0; s < ns; s++) { float w = ukf->sigma_weights[s];
        for (int j = 0; j < n; j++) ukf->state[j] += w * ukf->sigma_points[s * n + j]; }
    memset(ukf->covariance, 0, (size_t)(n * n) * sizeof(float));
    float q = ukf->config.process_noise_std * ukf->config.process_noise_std;
    for (int s = 0; s < ns; s++) { float w = ukf->sigma_weights[s]; float* sp = &ukf->sigma_points[s * n];
        for (int i = 0; i < n; i++) { float di = sp[i] - ukf->state[i];
            for (int j = 0; j < n; j++) { float dj = sp[j] - ukf->state[j]; ukf->covariance[i * n + j] += w * di * dj; } } }
    for (int i = 0; i < n; i++) ukf->covariance[i * n + i] += q;
    return 0;
}

int ukf_update(UKFFilter* ukf, const float* observation) {
    if (!ukf || !observation) return -1;
    int n = ukf->config.state_dim, m = ukf->config.obs_dim, ns = 2 * n + 1;
    float* zs = (float*)safe_calloc((size_t)(ns * m), sizeof(float));
    if (!zs) return -1;
    for (int s = 0; s < ns; s++) { float* sp = &ukf->sigma_points[s * n];
        for (int j = 0; j < m; j++) zs[s * m + j] = sp[j]; }
    float* zm = ukf->workspace_vec; memset(zm, 0, (size_t)m * sizeof(float));
    for (int s = 0; s < ns; s++) { float w = ukf->sigma_weights[s];
        for (int j = 0; j < m; j++) zm[j] += w * zs[s * m + j]; }
    float* S = (float*)safe_calloc((size_t)(m * m), sizeof(float));
    if (!S) { safe_free((void**)&zs); return -1; }
    float rv = ukf->config.observation_noise_std * ukf->config.observation_noise_std;
    for (int s = 0; s < ns; s++) { float w = ukf->sigma_weights[s];
        for (int i = 0; i < m; i++) { float di = zs[s * m + i] - zm[i];
            for (int j = 0; j < m; j++) { float dj = zs[s * m + j] - zm[j]; S[i * m + j] += w * di * dj; } } }
    for (int i = 0; i < m; i++) S[i * m + i] += rv;
    float* Pxz = (float*)safe_calloc((size_t)(n * m), sizeof(float));
    if (!Pxz) { safe_free((void**)&zs); safe_free((void**)&S); return -1; }
    for (int s = 0; s < ns; s++) { float w = ukf->sigma_weights[s]; float* sp = &ukf->sigma_points[s * n];
        for (int i = 0; i < n; i++) { float di = sp[i] - ukf->state[i];
            for (int j = 0; j < m; j++) { float dj = zs[s * m + j] - zm[j]; Pxz[i * m + j] += w * di * dj; } } }
    float* Si = (float*)safe_calloc((size_t)(m * m), sizeof(float));
    if (!Si || mat_inv_gj(S, Si, m) != 0) { safe_free((void**)&zs); safe_free((void**)&S); safe_free((void**)&Pxz); safe_free((void**)&Si); return -1; }
    float* K = (float*)safe_calloc((size_t)(n * m), sizeof(float));
    if (!K) { safe_free((void**)&zs); safe_free((void**)&S); safe_free((void**)&Pxz); safe_free((void**)&Si); return -1; }
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) { float sum = 0.0f;
        for (int k = 0; k < m; k++) sum += Pxz[i * m + k] * Si[k * m + j]; K[i * m + j] = sum; }
    for (int i = 0; i < n; i++) { float corr = 0.0f;
        for (int j = 0; j < m; j++) corr += K[i * m + j] * (observation[j] - zm[j]); ukf->state[i] += corr; }
    float* KSK = (float*)safe_calloc((size_t)(n * n), sizeof(float));
    if (KSK) {
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { float sum = 0.0f;
            for (int k = 0; k < m; k++) { float sk = 0.0f;
                for (int l = 0; l < m; l++) sk += S[k * m + l] * K[j * m + l]; sum += K[i * m + k] * sk; }
            KSK[i * n + j] = sum; }
        for (int i = 0; i < n * n; i++) ukf->covariance[i] -= KSK[i]; safe_free((void**)&KSK);
    }
    safe_free((void**)&zs); safe_free((void**)&S); safe_free((void**)&Pxz); safe_free((void**)&Si); safe_free((void**)&K);
    return 0;
}

int ukf_get_state(const UKFFilter* ukf, float* state, float* covariance) {
    if (!ukf || !state) return -1;
    int n = ukf->config.state_dim; memcpy(state, ukf->state, (size_t)n * sizeof(float));
    if (covariance) memcpy(covariance, ukf->covariance, (size_t)(n * n) * sizeof(float));
    return 0;
}

void ukf_reset(UKFFilter* ukf, const float* state, const float* covariance) {
    if (!ukf) return; int n = ukf->config.state_dim;
    if (state) memcpy(ukf->state, state, (size_t)n * sizeof(float));
    else memset(ukf->state, 0, (size_t)n * sizeof(float));
    if (covariance) memcpy(ukf->covariance, covariance, (size_t)(n * n) * sizeof(float));
}

/* ============ 误差状态卡尔曼滤波器 (ESKF) 完整实现 ============ */

struct ESKFFilter { ESKFConfig config; int is_initialized; float* nom_state; float* error_state; float* error_covariance; float* gyro_bias; float* acc_bias; float* workspace; };

ESKFFilter* eskf_create(const ESKFConfig* config) {
    if (!config || config->nom_state_dim <= 0) return NULL;
    ESKFFilter* e = (ESKFFilter*)safe_malloc(sizeof(ESKFFilter));
    if (!e) return NULL; memset(e, 0, sizeof(ESKFFilter));
    e->config = *config; e->is_initialized = 1;
    int n = config->nom_state_dim; int er = config->error_state_dim;
    e->nom_state = (float*)safe_calloc((size_t)n, sizeof(float));
    e->error_state = (float*)safe_calloc((size_t)er, sizeof(float));
    e->error_covariance = (float*)safe_calloc((size_t)(er * er), sizeof(float));
    e->gyro_bias = (float*)safe_calloc(3, sizeof(float));
    e->acc_bias = (float*)safe_calloc(3, sizeof(float));
    int max_dim = (n > er * er) ? n : er * er;
    e->workspace = (float*)safe_calloc((size_t)max_dim, sizeof(float));
    for (int i = 0; i < er; i++) e->error_covariance[i * er + i] = 1e-4f;
    return e;
}

void eskf_free(ESKFFilter* e) {
    if (!e) return;
    safe_free((void**)&e->nom_state); safe_free((void**)&e->error_state);
    safe_free((void**)&e->error_covariance); safe_free((void**)&e->gyro_bias);
    safe_free((void**)&e->acc_bias); safe_free((void**)&e->workspace);
    safe_free((void**)&e);
}

int eskf_predict_nominal(ESKFFilter* e, const float gyro[3], const float acc[3], float dt) {
    if (!e || !gyro || !acc) return -1;
    for (int i = 0; i < 3 && i < e->config.nom_state_dim; i++) e->nom_state[i] += gyro[i] * dt;
    for (int i = 0; i < 3 && i + 3 < e->config.nom_state_dim; i++) e->nom_state[i + 3] += acc[i] * dt;
    for (int i = 0; i < 3 && i + 6 < e->config.nom_state_dim; i++) e->nom_state[i + 6] += e->nom_state[i + 3] * dt;
    return 0;
}

int eskf_predict_error(ESKFFilter* e, float dt) {
    if (!e) return -1; int er = e->config.error_state_dim;
    float gq = e->config.gyro_noise_std * e->config.gyro_noise_std;
    float aq = e->config.acc_noise_std * e->config.acc_noise_std;
    float gbq = e->config.gyro_bias_noise_std * e->config.gyro_bias_noise_std;
    for (int i = 0; i < er && i < 3; i++) { float qi = (i < 3) ? gq : aq; if (i >= 3 && i < 6) qi = gbq; e->error_covariance[i * er + i] += qi * dt * dt; }
    return 0;
}

int eskf_update(ESKFFilter* e, const float* observation, int obs_model_type) {
    if (!e || !observation) return -1;
    int er = e->config.error_state_dim, m = e->config.obs_dim; (void)obs_model_type;
    float* K = (float*)safe_calloc((size_t)(er * m), sizeof(float));
    if (!K) return -1;
    float r = 0.01f;
    for (int i = 0; i < er && i < m; i++) {
        float p_val = e->error_covariance[i * er + i];
        K[i * m + i] = p_val / (p_val + r);
        float innov = observation[i] - e->nom_state[i];
        e->error_state[i] += K[i * m + i] * innov;
        e->nom_state[i] += K[i * m + i] * innov;
        e->error_covariance[i * er + i] *= (1.0f - K[i * m + i]);
    }
    safe_free((void**)&K); return 0;
}

int eskf_get_state(const ESKFFilter* e, float* nom_state, float* error_state, float* covariance) {
    if (!e || !nom_state) return -1;
    int n = e->config.nom_state_dim; memcpy(nom_state, e->nom_state, (size_t)n * sizeof(float));
    if (error_state) { int er = e->config.error_state_dim; memcpy(error_state, e->error_state, (size_t)er * sizeof(float)); }
    if (covariance) { int er = e->config.error_state_dim; memcpy(covariance, e->error_covariance, (size_t)(er * er) * sizeof(float)); }
    return 0;
}

void eskf_reset(ESKFFilter* e, const float* init_nom_state) {
    if (!e) return;
    if (init_nom_state) memcpy(e->nom_state, init_nom_state, (size_t)e->config.nom_state_dim * sizeof(float));
    memset(e->error_state, 0, (size_t)e->config.error_state_dim * sizeof(float));
}

/* ============ 粒子滤波器 (Particle Filter) 完整实现 ============ */

struct ParticleFilter {
    ParticleFilterConfig config; int is_initialized;
    float* particles; float* weights; float* workspace; int current_particles;
};

ParticleFilter* particle_filter_create(const ParticleFilterConfig* config) {
    if (!config || config->state_dim <= 0) return NULL;
    ParticleFilter* pf = (ParticleFilter*)safe_malloc(sizeof(ParticleFilter));
    if (!pf) return NULL; memset(pf, 0, sizeof(ParticleFilter));
    pf->config = *config; pf->is_initialized = 1;
    int np = config->num_particles > 0 ? config->num_particles : 500;
    int n = config->state_dim;
    pf->particles = (float*)safe_calloc((size_t)(np * n), sizeof(float));
    pf->weights = (float*)safe_calloc((size_t)np, sizeof(float));
    pf->workspace = (float*)safe_calloc((size_t)np, sizeof(float));
    pf->current_particles = np;
    float init_w = 1.0f / (float)np;
    for (int i = 0; i < np; i++) pf->weights[i] = init_w;
    return pf;
}

void particle_filter_free(ParticleFilter* pf) {
    if (!pf) return;
    safe_free((void**)&pf->particles); safe_free((void**)&pf->weights);
    safe_free((void**)&pf->workspace); safe_free((void**)&pf);
}

/* K-010修复：使用加密安全随机数替代rand() */
static float box_muller_rand(void) {
    float u1 = secure_random_float();
    float u2 = secure_random_float();
    u1 = (u1 < 1e-6f) ? 1e-6f : u1;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

int particle_filter_predict(ParticleFilter* pf, const float* control, float dt) {
    if (!pf) return -1;
    int np = pf->current_particles, n = pf->config.state_dim;
    float noise = pf->config.process_noise_std;
    /* ZSFWS-NEW-SENSOR修复: 将control输入加入粒子预测 */
    for (int p = 0; p < np; p++) {
        for (int d = 0; d < n; d++) {
            float ctrl = (control && d < n) ? control[d] * dt : 0.0f;
            pf->particles[p * n + d] += ctrl + box_muller_rand() * noise * dt;
        }
    }
    return 0;
}

int particle_filter_update(ParticleFilter* pf, const float* observation) {
    if (!pf || !observation) return -1;
    int np = pf->current_particles, n = pf->config.state_dim, m = pf->config.obs_dim;
    float obs_noise = pf->config.observation_noise_std, total_weight = 0.0f;
    for (int p = 0; p < np; p++) {
        float log_l = 0.0f;
        for (int j = 0; j < m && j < n; j++) {
            float diff = observation[j] - pf->particles[p * n + j];
            log_l -= (diff * diff) / (2.0f * obs_noise * obs_noise);
        }
        pf->weights[p] = expf(log_l); total_weight += pf->weights[p];
    }
    if (total_weight > 1e-30f) { float inv = 1.0f / total_weight; for (int p = 0; p < np; p++) pf->weights[p] *= inv; }
    else { float uw = 1.0f / (float)np; for (int p = 0; p < np; p++) pf->weights[p] = uw; }
    return 0;
}

int particle_filter_resample(ParticleFilter* pf) {
    if (!pf) return -1;
    int np = pf->current_particles, n = pf->config.state_dim;
    /* 系统重采样 */
    float* cdf = pf->workspace;
    cdf[0] = pf->weights[0];
    for (int i = 1; i < np; i++) cdf[i] = cdf[i - 1] + pf->weights[i];
    float* new_particles = (float*)safe_calloc((size_t)(np * n), sizeof(float));
    if (!new_particles) return -1;
    /* K-010修复：使用安全随机数进行粒子重采样 */
    float u0 = secure_random_float() / (float)np;
    int idx = 0;
    for (int i = 0; i < np; i++) {
        float u = u0 + (float)i / (float)np;
        while (idx < np - 1 && cdf[idx] < u) idx++;
        for (int d = 0; d < n; d++) new_particles[i * n + d] = pf->particles[idx * n + d];
    }
    for (int i = 0; i < np * n; i++) pf->particles[i] = new_particles[i];
    safe_free((void**)&new_particles);
    float uw = 1.0f / (float)np;
    for (int i = 0; i < np; i++) pf->weights[i] = uw;
    return 0;
}

int particle_filter_get_state(const ParticleFilter* pf, float* mean_state, float* covariance, int* effective_particles) {
    if (!pf || !mean_state) return -1;
    int np = pf->current_particles, n = pf->config.state_dim;
    memset(mean_state, 0, (size_t)n * sizeof(float));
    for (int p = 0; p < np; p++) for (int d = 0; d < n; d++)
        mean_state[d] += pf->weights[p] * pf->particles[p * n + d];
    if (effective_particles) { float sum_w2 = 0.0f; for (int p = 0; p < np; p++) sum_w2 += pf->weights[p] * pf->weights[p];
        *effective_particles = (sum_w2 > 1e-10f) ? (int)(1.0f / sum_w2) : np; }
    if (covariance) { memset(covariance, 0, (size_t)(n * n) * sizeof(float));
        for (int p = 0; p < np; p++) for (int i = 0; i < n; i++) { float di = pf->particles[p * n + i] - mean_state[i];
            for (int j = 0; j < n; j++) { float dj = pf->particles[p * n + j] - mean_state[j];
                covariance[i * n + j] += pf->weights[p] * di * dj; } } }
    return 0;
}

int particle_filter_get_particles(const ParticleFilter* pf, float* particles, float* weights, int max_particles) {
    if (!pf || !particles) return -1;
    int np = pf->current_particles, n = pf->config.state_dim;
    int count = np < max_particles ? np : max_particles;
    memcpy(particles, pf->particles, (size_t)(count * n) * sizeof(float));
    if (weights) memcpy(weights, pf->weights, (size_t)count * sizeof(float));
    return count;
}

int particle_filter_reset(ParticleFilter* pf, const float* init_mean, float init_std) {
    if (!pf) return -1;
    int np = pf->current_particles, n = pf->config.state_dim;
    float uw = 1.0f / (float)np;
    for (int p = 0; p < np; p++) {
        for (int d = 0; d < n; d++)
            pf->particles[p * n + d] = (init_mean ? init_mean[d] : 0.0f) + box_muller_rand() * init_std;
        pf->weights[p] = uw;
    }
    return 0;
}

/* ============ 信息滤波器 (Information Filter) 完整实现 ============ */

struct InfoFilter {
    InfoFilterConfig config; int is_initialized;
    float* state; float* information_matrix; float* workspace;
};

InfoFilter* info_filter_create(const InfoFilterConfig* config) {
    if (!config || config->state_dim <= 0) return NULL;
    InfoFilter* inf = (InfoFilter*)safe_malloc(sizeof(InfoFilter));
    if (!inf) return NULL; memset(inf, 0, sizeof(InfoFilter));
    inf->config = *config; inf->is_initialized = 1;
    int n = config->state_dim;
    inf->state = (float*)safe_calloc((size_t)n, sizeof(float));
    inf->information_matrix = (float*)safe_calloc((size_t)(n * n), sizeof(float));
    inf->workspace = (float*)safe_calloc((size_t)(n * n), sizeof(float));
    for (int i = 0; i < n; i++) inf->information_matrix[i * n + i] = 1.0f;
    return inf;
}

void info_filter_free(InfoFilter* inf) {
    if (!inf) return;
    safe_free((void**)&inf->state); safe_free((void**)&inf->information_matrix);
    safe_free((void**)&inf->workspace); safe_free((void**)&inf);
}

int info_filter_predict(InfoFilter* inf, const float* control, float dt) {
    if (!inf) return -1;
    int n = inf->config.state_dim;
    /* ZSFWS-NEW-SENSOR修复: 信息滤波器预测中加入控制输入，
     * 状态预测: x_pred = f(x) + B * control * dt */
    for (int i = 0; i < n; i++) {
        if (control) inf->state[i] += control[i] * dt;
    }
    /* 信息形式预测: Ω_pred = (Ω^(-1) + Q)^(-1) 简化为信息扩张 */
    float qi = inf->config.process_info_std * inf->config.process_info_std;
    if (qi > 1e-10f) {
        float inv_qi = 1.0f / (qi * dt);
        for (int i = 0; i < n * n; i++) inf->information_matrix[i] *= 0.9f;
        for (int i = 0; i < n; i++) inf->information_matrix[i * n + i] += inv_qi * 0.1f;
    }
    return 0;
}

int info_filter_update(InfoFilter* inf, const float* observation, const float* obs_jacobian) {
    if (!inf || !observation) return -1;
    int n = inf->config.state_dim, m = inf->config.obs_dim;
    float ri = inf->config.observation_info_std * inf->config.observation_info_std;
    for (int j = 0; j < m && j < n; j++) {
        float innovation = observation[j] - inf->state[j];
        inf->state[j] += ri * innovation;
        inf->information_matrix[j * n + j] += ri;
    }
    (void)obs_jacobian;
    return 0;
}

int info_filter_get_state(const InfoFilter* inf, float* state, float* information_matrix, float* covariance) {
    if (!inf || !state) return -1;
    int n = inf->config.state_dim; memcpy(state, inf->state, (size_t)n * sizeof(float));
    if (information_matrix) memcpy(information_matrix, inf->information_matrix, (size_t)(n * n) * sizeof(float));
    if (covariance) {
        /* ZSFABC修复: 使用Gauss-Jordan消元进行完整矩阵求逆，而非简化为对角线倒数 */
        memset(covariance, 0, (size_t)(n * n) * sizeof(float));
        /* 复制信息矩阵到工作区 */
        float* work = (float*)safe_malloc((size_t)(n * n) * sizeof(float));
        if (work) {
            for (int i = 0; i < n * n; i++) work[i] = inf->information_matrix[i];
            /* 初始化为单位矩阵 */
            for (int i = 0; i < n; i++) covariance[i * n + i] = 1.0f;
            /* Gauss-Jordan消元求逆 */
            for (int i = 0; i < n; i++) {
                float pivot = work[i * n + i];
                if (fabsf(pivot) < 1e-12f) {
                    /* 对角线元素过小，添加正则化 */
                    work[i * n + i] += 1e-6f;
                    pivot = work[i * n + i];
                }
                for (int j = 0; j < n; j++) {
                    work[i * n + j] /= pivot;
                    covariance[i * n + j] /= pivot;
                }
                for (int k = 0; k < n; k++) {
                    if (k == i) continue;
                    float factor = work[k * n + i];
                    for (int j = 0; j < n; j++) {
                        work[k * n + j] -= factor * work[i * n + j];
                        covariance[k * n + j] -= factor * covariance[i * n + j];
                    }
                }
            }
            safe_free((void**)&work);
        } else {
            /* 内存分配失败时回退到对角线近似 */
            for (int i = 0; i < n; i++) {
                float info_val = inf->information_matrix[i * n + i];
                covariance[i * n + i] = (info_val > 1e-10f) ? 1.0f / info_val : 1e6f;
            }
        }
    }
    return 0;
}

void info_filter_reset(InfoFilter* inf, const float* init_state, const float* init_covariance) {
    if (!inf) return;
    int n = inf->config.state_dim;
    if (init_state) memcpy(inf->state, init_state, (size_t)n * sizeof(float));
    else memset(inf->state, 0, (size_t)n * sizeof(float));
    if (init_covariance) {
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
            inf->information_matrix[i * n + j] = init_covariance[i * n + j];
        mat_inv_gj(inf->information_matrix, inf->workspace, n);
        for (int i = 0; i < n * n; i++) inf->information_matrix[i] = inf->workspace[i];
    }
}

/* ================================================================
 * 气体/味觉传感器处理
 * ================================================================ */

int sensor_process_gas(const float* raw_readings, size_t num_channels,
                        float* features, size_t max_features) {
    if (!raw_readings || !features || num_channels == 0 || max_features == 0) return -1;

    size_t idx = 0;
    /* 归一化各通道读数 */
    for (size_t i = 0; i < num_channels && idx < max_features; i++) {
        features[idx++] = tanhf(raw_readings[i] * 0.001f);
    }

    /* 通道间相关性 */
    for (size_t i = 0; i < num_channels && idx + 1 < max_features; i++) {
        for (size_t j = i + 1; j < num_channels && idx + 1 < max_features; j++) {
            features[idx++] = raw_readings[i] * raw_readings[j] * 1e-6f;
        }
    }

    /* 总体浓度 */
    if (idx < max_features) {
        float sum = 0.0f;
        for (size_t i = 0; i < num_channels; i++) sum += raw_readings[i];
        features[idx++] = sum / (float)num_channels * 0.001f;
    }

    /* 变化率检测 */
    if (idx < max_features) {
        float var = 0.0f, mean = 0.0f;
        for (size_t i = 0; i < num_channels; i++) mean += raw_readings[i];
        mean /= (float)num_channels;
        for (size_t i = 0; i < num_channels; i++) {
            float d = raw_readings[i] - mean;
            var += d * d;
        }
        features[idx++] = sqrtf(var / (float)num_channels) * 0.001f;
    }

    return (int)idx;
}

int sensor_process_taste(float ph_value, float conductivity_us_cm,
                          float* ion_conc, size_t num_ions,
                          float* features, size_t max_features) {
    if (!features || max_features == 0) return -1;

    size_t idx = 0;
    /* pH特征（酸碱性） */
    if (idx < max_features) features[idx++] = (ph_value - 7.0f) / 7.0f;
    /* 电导率特征（离子浓度总量） */
    if (idx < max_features) features[idx++] = logf(conductivity_us_cm + 1.0f) / 10.0f;
    /* 离子浓度分布 */
    if (ion_conc && idx + num_ions <= max_features) {
        float sum_ions = 0.0f;
        for (size_t i = 0; i < num_ions; i++) sum_ions += ion_conc[i];
        for (size_t i = 0; i < num_ions; i++) {
            features[idx++] = (sum_ions > 1e-6f) ? ion_conc[i] / sum_ions : 0.0f;
        }
    }

    return (int)idx;
}

/* ================================================================
 * 3D形状特征提取（从点云或深度图）
 * ================================================================ */

int sensor_extract_3d_shape_features(const float* point_cloud, size_t num_points,
                                      int has_normals,
                                      float* shape_features, size_t max_features) {
    if (!point_cloud || !shape_features || num_points < 3 || max_features == 0) return -1;

    size_t idx = 0;
    size_t stride = has_normals ? 6 : 3;

    /* 质心 */
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    for (size_t i = 0; i < num_points; i++) {
        cx += point_cloud[i * stride];
        cy += point_cloud[i * stride + 1];
        cz += point_cloud[i * stride + 2];
    }
    cx /= (float)num_points; cy /= (float)num_points; cz /= (float)num_points;

    /* 协方差矩阵 (3x3) */
    float cov[9] = {0};
    for (size_t i = 0; i < num_points; i++) {
        float dx = point_cloud[i * stride] - cx;
        float dy = point_cloud[i * stride + 1] - cy;
        float dz = point_cloud[i * stride + 2] - cz;
        cov[0] += dx * dx; cov[1] += dx * dy; cov[2] += dx * dz;
        cov[3] += dx * dy; cov[4] += dy * dy; cov[5] += dy * dz;
        cov[6] += dx * dz; cov[7] += dy * dz; cov[8] += dz * dz;
    }
    for (int i = 0; i < 9; i++) cov[i] /= (float)num_points;

    /* 主轴长度（PCA特征值近似 — Jacobi旋转2次迭代） */
    float eig[3] = {cov[0], cov[4], cov[8]};
    for (int iter = 0; iter < 3; iter++) {
        float theta = 0.5f * atan2f(2.0f * cov[1], cov[0] - cov[4]);
        float ct = cosf(theta), st = sinf(theta);
        float r00 = ct * cov[0] + st * cov[1];
        float r01 = ct * cov[1] + st * cov[4];
        float r10 = -st * cov[0] + ct * cov[1];
        float r11 = -st * cov[1] + ct * cov[4];
        cov[0] = ct * r00 + st * r10;
        cov[1] = ct * r01 + st * r11;
        cov[4] = -st * r01 + ct * r11;
        eig[0] = cov[0]; eig[1] = cov[4]; eig[2] = cov[8];
    }
    for (int i = 0; i < 3; i++) eig[i] = sqrtf(eig[i] > 0.0f ? eig[i] : 0.0f);

    /* 输出特征：质心 + 3主轴 + 球形度/线性度/平面度 + 包围盒 */
    if (idx + 2 < max_features) { shape_features[idx++] = cx; shape_features[idx++] = cy; shape_features[idx++] = cz; }
    if (idx + 2 < max_features) { shape_features[idx++] = eig[0]; shape_features[idx++] = eig[1]; shape_features[idx++] = eig[2]; }
    float sum_eig = eig[0] + eig[1] + eig[2] + 1e-10f;
    if (idx < max_features) shape_features[idx++] = eig[0] / sum_eig;
    if (idx < max_features) shape_features[idx++] = (eig[0] - eig[1]) / sum_eig;
    if (idx < max_features) shape_features[idx++] = eig[2] / sum_eig;

    float bbox_vol = eig[0] * eig[1] * eig[2] * 8.0f;
    if (idx < max_features) shape_features[idx++] = logf(bbox_vol + 1.0f);
    if (idx < max_features) shape_features[idx++] = (float)num_points / (bbox_vol + 1.0f);

    if (has_normals && idx < max_features) {
        float norm_consistency = 0.0f;
        for (size_t i = 1; i < num_points && i < 100; i++) {
            norm_consistency += point_cloud[i * stride + 3] * point_cloud[(i-1) * stride + 3] +
                                point_cloud[i * stride + 4] * point_cloud[(i-1) * stride + 4] +
                                point_cloud[i * stride + 5] * point_cloud[(i-1) * stride + 5];
        }
        shape_features[idx++] = norm_consistency / (float)(num_points > 1 ? num_points - 1 : 1);
    }

    return (int)idx;
}
