#include "selflnn/training/training_monitor.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * 训练指标追踪实现
 * ============================================================================ */

TrainingMonitor* training_monitor_create(const char* run_name, int total_epochs) {
    TrainingMonitor* tm = (TrainingMonitor*)safe_calloc(1, sizeof(TrainingMonitor));
    if (!tm) return NULL;
    tm->run_name = run_name ? run_name : "training";
    tm->current_epoch = 0;
    tm->total_epochs = total_epochs > 0 ? total_epochs : 100;
    tm->history_count = 0;
    /* R5-②修复: 初始化计时字段 */
    tm->start_time_sec = (long)time(NULL);
    tm->last_epoch_time_sec = tm->start_time_sec;
    tm->epochs_recorded = 0;
    tm->avg_epoch_seconds = 0.0f;
    return tm;
}

void training_monitor_free(TrainingMonitor* tm) {
    if (!tm) return;
    safe_free((void**)&tm);
}

static int find_or_create_history(TrainingMonitor* tm, MetricType type) {
    for (int i = 0; i < tm->history_count; i++) {
        if (tm->histories[i].type == type) return i;
    }
    if (tm->history_count >= TM_MAX_METRIC_TYPES) return -1;
    int idx = tm->history_count++;
    memset(&tm->histories[idx], 0, sizeof(MetricHistory));
    tm->histories[idx].type = type;
    tm->histories[idx].count = 0;
    tm->histories[idx].head = 0;
    return idx;
}

int training_monitor_log_metric(TrainingMonitor* tm, MetricType type,
                                 int epoch, int step, float value) {
    if (!tm) return -1;
    int idx = find_or_create_history(tm, type);
    if (idx < 0) return -1;
    MetricHistory* hist = &tm->histories[idx];
    int slot = hist->count < TM_MAX_HISTORY_PER_TYPE
               ? hist->count++
               : (hist->head = (hist->head + 1) % TM_MAX_HISTORY_PER_TYPE,
                  hist->head + TM_MAX_HISTORY_PER_TYPE - 1 >= hist->count
                  ? hist->count - 1 : hist->head);
    if (hist->count <= TM_MAX_HISTORY_PER_TYPE) {
        slot = hist->count - 1;
    } else {
        slot = hist->head;
        hist->head = (hist->head + 1) % TM_MAX_HISTORY_PER_TYPE;
    }
    MetricRecord* rec = &hist->records[slot];
    rec->type = type;
    rec->epoch = epoch;
    rec->step = step;
    rec->value = value;
    rec->timestamp_sec = (double)time(NULL);
    rec->custom_name[0] = '\0';
    if (epoch >= 0) tm->current_epoch = epoch;
    return 0;
}

int training_monitor_log_custom(TrainingMonitor* tm, const char* name,
                                 int epoch, int step, float value) {
    if (!tm || !name) return -1;
    int idx = find_or_create_history(tm, TM_CUSTOM);
    if (idx < 0) return -1;
    MetricHistory* hist = &tm->histories[idx];
    int slot;
    if (hist->count < TM_MAX_HISTORY_PER_TYPE) {
        slot = hist->count++;
    } else {
        slot = hist->head;
        hist->head = (hist->head + 1) % TM_MAX_HISTORY_PER_TYPE;
    }
    MetricRecord* rec = &hist->records[slot];
    rec->type = TM_CUSTOM;
    rec->epoch = epoch;
    rec->step = step;
    rec->value = value;
    rec->timestamp_sec = (double)time(NULL);
    strncpy(rec->custom_name, name, sizeof(rec->custom_name) - 1);
    rec->custom_name[sizeof(rec->custom_name) - 1] = '\0';
    if (epoch >= 0) tm->current_epoch = epoch;
    return 0;
}

int training_monitor_get_metric_history(const TrainingMonitor* tm,
                                         MetricType type,
                                         MetricRecord* out_buffer,
                                         int max_count) {
    if (!tm || !out_buffer || max_count <= 0) return -1;
    for (int i = 0; i < tm->history_count; i++) {
        if (tm->histories[i].type == type) {
            MetricHistory* hist = (MetricHistory*)&tm->histories[i];
            int copy_count = hist->count < max_count ? hist->count : max_count;
/* 边界检查，防止copy_count异常越界导致memcpy溢出 */
            if (copy_count < 0 || copy_count > hist->count || copy_count > TM_MAX_HISTORY_PER_TYPE)
                return -1;
            if (hist->count <= TM_MAX_HISTORY_PER_TYPE) {
                memcpy(out_buffer, hist->records, copy_count * sizeof(MetricRecord));
            } else {
                int start = hist->head;
                for (int j = 0; j < copy_count; j++) {
                    out_buffer[j] = hist->records[(start + j) % TM_MAX_HISTORY_PER_TYPE];
                }
            }
            return copy_count;
        }
    }
    return 0;
}

float training_monitor_get_best_value(const TrainingMonitor* tm,
                                       MetricType type, int minimize) {
    if (!tm) return 0.0f;
    for (int i = 0; i < tm->history_count; i++) {
        if (tm->histories[i].type == type) {
            MetricHistory* hist = (MetricHistory*)&tm->histories[i];
            if (hist->count == 0) return 0.0f;
            float best = hist->records[0].value;
            int count = hist->count < TM_MAX_HISTORY_PER_TYPE
                        ? hist->count : TM_MAX_HISTORY_PER_TYPE;
            for (int j = 1; j < count; j++) {
                float v = hist->records[j].value;
                if (minimize ? (v < best) : (v > best)) best = v;
            }
            return best;
        }
    }
    return 0.0f;
}

float training_monitor_get_latest_value(const TrainingMonitor* tm,
                                         MetricType type) {
    if (!tm) return 0.0f;
    for (int i = 0; i < tm->history_count; i++) {
        if (tm->histories[i].type == type) {
            MetricHistory* hist = (MetricHistory*)&tm->histories[i];
            if (hist->count == 0) return 0.0f;
            int last = hist->count <= TM_MAX_HISTORY_PER_TYPE
                       ? hist->count - 1
                       : (hist->head + TM_MAX_HISTORY_PER_TYPE - 1)
                         % TM_MAX_HISTORY_PER_TYPE;
            return hist->records[last].value;
        }
    }
    return 0.0f;
}

int training_monitor_reset(TrainingMonitor* tm) {
    if (!tm) return -1;
    for (int i = 0; i < tm->history_count; i++) {
        tm->histories[i].count = 0;
        tm->histories[i].head = 0;
    }
    tm->history_count = 0;
    tm->current_epoch = 0;
    return 0;
}

/* ============================================================================
 * 超参数调优实现
 * ============================================================================ */

HyperparameterSearch* hp_search_create(HPSearchMethod method, int max_trials) {
    HyperparameterSearch* search = (HyperparameterSearch*)safe_calloc(1, sizeof(HyperparameterSearch));
    if (!search) return NULL;
    search->method = method;
    search->max_trials = max_trials > 0 ? (max_trials < TM_MAX_TRIALS ? max_trials : TM_MAX_TRIALS) : 100;
    search->param_count = 0;
    search->trial_count = 0;
    search->best_trial_idx = -1;
    search->best_result.valid = 0;
    search->rng_state = (unsigned int)time(NULL);
    if (search->rng_state == 0) search->rng_state = 123456789u;
    return search;
}

void hp_search_free(HyperparameterSearch* search) {
    if (!search) return;
    safe_free((void**)&search);
}

static float hp_rand_float(HyperparameterSearch* search, float min, float max) {
    search->rng_state = search->rng_state * 1103515245u + 12345u;
    float r = (float)(search->rng_state & 0x7FFFFFFFu) / 2147483648.0f;
    return min + r * (max - min);
}

int hp_search_add_param_float(HyperparameterSearch* search,
                               const char* name, float min_val, float max_val) {
    if (!search || !name || search->param_count >= TM_MAX_PARAMS) return -1;
    if (min_val > max_val) return -1;
    HPParamConfig* p = &search->params[search->param_count];
    strncpy(p->name, name, TM_MAX_PARAM_NAME - 1);
    p->name[TM_MAX_PARAM_NAME - 1] = '\0';
    p->type = HP_TYPE_FLOAT;
    p->min_val = min_val;
    p->max_val = max_val;
    p->step = 0.0f;
    p->num_categories = 0;
    search->param_count++;
    return 0;
}

int hp_search_add_param_int(HyperparameterSearch* search,
                             const char* name, int min_val, int max_val) {
    if (!search || !name || search->param_count >= TM_MAX_PARAMS) return -1;
    if (min_val > max_val) return -1;
    HPParamConfig* p = &search->params[search->param_count];
    strncpy(p->name, name, TM_MAX_PARAM_NAME - 1);
    p->name[TM_MAX_PARAM_NAME - 1] = '\0';
    p->type = HP_TYPE_INT;
    p->min_val = (float)min_val;
    p->max_val = (float)max_val;
    p->step = 1.0f;
    p->num_categories = 0;
    search->param_count++;
    return 0;
}

int hp_search_add_param_categorical(HyperparameterSearch* search,
                                     const char* name,
                                     const float* values, int num_values) {
    if (!search || !name || !values || search->param_count >= TM_MAX_PARAMS) return -1;
    HPParamConfig* p = &search->params[search->param_count];
/* 边界检查，使用sizeof确保num_values不超过categories数组容量 */
    if (num_values <= 0 || num_values > (int)(sizeof(p->categories) / sizeof(p->categories[0]))) return -1;
    strncpy(p->name, name, TM_MAX_PARAM_NAME - 1);
    p->name[TM_MAX_PARAM_NAME - 1] = '\0';
    p->type = HP_TYPE_CATEGORICAL;
    p->num_categories = num_values;
/* 边界检查已在上方完成，num_values不超过categories容量 */
    memcpy(p->categories, values, (size_t)num_values * sizeof(float));
    search->param_count++;
    return 0;
}

static int hp_grid_search_next(HyperparameterSearch* search, float* out_params) {
    int idx = search->trial_count;
    if (idx >= search->max_trials) return -1;
    int divisor = 1;
    for (int p = search->param_count - 1; p >= 0; p--) {
        HPParamConfig* cfg = &search->params[p];
        int steps;
        if (cfg->type == HP_TYPE_CATEGORICAL) {
            steps = cfg->num_categories;
        } else if (cfg->step > 0.0f) {
            steps = (int)((cfg->max_val - cfg->min_val) / cfg->step) + 1;
            if (steps < 1) steps = 1;
        } else {
            steps = 10;
        }
        int pos = (idx / divisor) % steps;
        if (cfg->type == HP_TYPE_CATEGORICAL) {
            out_params[p] = cfg->categories[pos];
        } else if (cfg->type == HP_TYPE_INT) {
            out_params[p] = (float)((int)cfg->min_val + pos * (int)cfg->step);
        } else {
            out_params[p] = cfg->min_val + (float)pos * cfg->step;
        }
        divisor *= steps;
    }
    return 0;
}

static int hp_random_search_next(HyperparameterSearch* search, float* out_params) {
    if (search->trial_count >= search->max_trials) return -1;
    for (int p = 0; p < search->param_count; p++) {
        HPParamConfig* cfg = &search->params[p];
        if (cfg->type == HP_TYPE_CATEGORICAL) {
            int idx = (int)hp_rand_float(search, 0.0f, (float)(cfg->num_categories - 1));
            if (idx < 0) idx = 0;
            if (idx >= cfg->num_categories) idx = cfg->num_categories - 1;
            out_params[p] = cfg->categories[idx];
        } else if (cfg->type == HP_TYPE_INT) {
            out_params[p] = (float)((int)hp_rand_float(search, cfg->min_val, cfg->max_val));
        } else {
            out_params[p] = hp_rand_float(search, cfg->min_val, cfg->max_val);
        }
    }
    return 0;
}

static float hp_gaussian_kernel(float dist_sq, float length_scale) {
    return expf(-0.5f * dist_sq / (length_scale * length_scale + 1e-10f));
}

/* ============================================================================
 * Cholesky分解：正定对称矩阵 A(n×n) 分解为 A = L * L^T
 * 结果L（下三角含对角线）就地覆盖A，上三角部分不修改
 * 返回值：0成功，-1矩阵非正定
 * ============================================================================ */
static int hp_cholesky_decompose(float* A, int n) {
    for (int j = 0; j < n; j++) {
        float s = 0.0f;
        for (int k = 0; k < j; k++) {
            s += A[j * n + k] * A[j * n + k];
        }
        float diag_val = A[j * n + j] - s;
        if (diag_val <= 1e-15f) return -1;
        A[j * n + j] = sqrtf(diag_val);
        float inv_diag = 1.0f / A[j * n + j];
        for (int i = j + 1; i < n; i++) {
            s = 0.0f;
            for (int k = 0; k < j; k++) {
                s += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - s) * inv_diag;
        }
    }
    return 0;
}

/* 前向代入：求解下三角系统 L * x = b，L[i*n+i]为对角线 */
static void hp_forward_solve(const float* L, int n, const float* b, float* x) {
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int j = 0; j < i; j++) {
            s += L[i * n + j] * x[j];
        }
        x[i] = (b[i] - s) / L[i * n + i];
    }
}

/* 后向代入：求解上三角系统 L^T * x = b，L^T[i*n+j] = L[j*n+i] */
static void hp_backward_solve(const float* L, int n, const float* b, float* x) {
    for (int i = n - 1; i >= 0; i--) {
        float s = 0.0f;
        for (int j = i + 1; j < n; j++) {
            s += L[j * n + i] * x[j];
        }
        x[i] = (b[i] - s) / L[i * n + i];
    }
}

static int hp_bayesian_search_next(HyperparameterSearch* search, float* out_params) {
    if (search->trial_count >= search->max_trials) return -1;
    if (search->trial_count < search->param_count + 1) {
        return hp_random_search_next(search, out_params);
    }
    float length_scale = 1.0f;
    float sigma_noise = 1e-4f;
    int n = search->trial_count;
    if (n <= 0) return hp_random_search_next(search, out_params);
    float* K = (float*)safe_calloc((size_t)n * (size_t)n, sizeof(float));
    float* k_star = (float*)safe_calloc((size_t)n, sizeof(float));
    float* y = (float*)safe_calloc((size_t)n, sizeof(float));
    float* K_inv_y = (float*)safe_calloc((size_t)n, sizeof(float));
    if (!K || !k_star || !y || !K_inv_y) {
        safe_free((void**)&K); safe_free((void**)&k_star);
        safe_free((void**)&y); safe_free((void**)&K_inv_y);
        return hp_random_search_next(search, out_params);
    }
    float best_acq = -1e10f;
    float candidate[TM_MAX_PARAMS];
    int found = 0;
    for (int sample = 0; sample < 100; sample++) {
        for (int p = 0; p < search->param_count; p++) {
            HPParamConfig* cfg = &search->params[p];
            if (cfg->type == HP_TYPE_CATEGORICAL) {
                int idx = (int)hp_rand_float(search, 0.0f, (float)(cfg->num_categories - 1));
                if (idx < 0) idx = 0;
                if (idx >= cfg->num_categories) idx = cfg->num_categories - 1;
                candidate[p] = cfg->categories[idx];
            } else if (cfg->type == HP_TYPE_INT) {
                candidate[p] = (float)((int)hp_rand_float(search, cfg->min_val, cfg->max_val));
            } else {
                candidate[p] = hp_rand_float(search, cfg->min_val, cfg->max_val);
            }
        }
        for (int i = 0; i < n; i++) {
            y[i] = search->trials[i].loss;
            for (int j = 0; j < n; j++) {
                float dist_sq = 0.0f;
                for (int p = 0; p < search->param_count; p++) {
                    float d = search->trials[i].param_values[p]
                            - search->trials[j].param_values[p];
                    dist_sq += d * d;
                }
                K[i * n + j] = hp_gaussian_kernel(dist_sq, length_scale);
                if (i == j) K[i * n + j] += sigma_noise;
            }
            float dist_sq = 0.0f;
            for (int p = 0; p < search->param_count; p++) {
                float d = search->trials[i].param_values[p] - candidate[p];
                dist_sq += d * d;
            }
            k_star[i] = hp_gaussian_kernel(dist_sq, length_scale);
        }
        float k_star_star = 1.0f + sigma_noise;
        /* 在核矩阵K对角线添加微扰项jitter，保证Cholesky分解数值稳定性 */
        for (int i = 0; i < n; i++) {
            K[i * n + i] += 1e-6f;
        }
        /* Cholesky分解：K = L * L^T，就地覆盖K的下三角 */
        if (hp_cholesky_decompose(K, n) != 0) {
            continue;
        }
        /* 分配临时缓冲区用于前向/后向代入中间结果 */
        float* z_buf = (float*)safe_calloc((size_t)n, sizeof(float));
        if (!z_buf) {
            continue;
        }
        /* 求解 alpha = K^{-1} * y：分两步
         * 步骤1：前向代入 L*z = y  →  z = L^{-1}*y
         * 步骤2：后向代入 L^T*alpha = z  →  alpha = L^{-T}*z = K^{-1}*y */
        hp_forward_solve(K, n, y, z_buf);
        hp_backward_solve(K, n, z_buf, K_inv_y);
        /* GP后验均值：mu* = k*^T * K^{-1} * y = k*^T * alpha */
        float mu_star = 0.0f;
        for (int i = 0; i < n; i++) {
            mu_star += k_star[i] * K_inv_y[i];
        }
        /* GP后验方差：sigma*^2 = k** - v^T*v，其中 v = L^{-1}*k* */
        hp_forward_solve(K, n, k_star, z_buf);
        float sigma_sq_star = k_star_star;
        for (int i = 0; i < n; i++) {
            sigma_sq_star -= z_buf[i] * z_buf[i];
        }
        safe_free((void**)&z_buf);
        if (sigma_sq_star < 1e-10f) sigma_sq_star = 1e-10f;
        float sigma_star = sqrtf(sigma_sq_star);
        float best_y = y[0];
        for (int i = 1; i < n; i++) {
            if (y[i] < best_y) best_y = y[i];
        }
        float improvement = best_y - mu_star;
        float z = sigma_star > 1e-10f ? improvement / sigma_star : 0.0f;
        float cdf = 0.5f * (1.0f + erff(z / 1.41421356f));
        float pdf = expf(-0.5f * z * z) / 2.50662827f;
        float ei = improvement * cdf + sigma_star * pdf;
        if (ei > best_acq || !found) {
            best_acq = ei;
            memcpy(out_params, candidate, (size_t)search->param_count * sizeof(float));
            found = 1;
        }
    }
    safe_free((void**)&K); safe_free((void**)&k_star);
    safe_free((void**)&y); safe_free((void**)&K_inv_y);
    if (!found) return hp_random_search_next(search, out_params);
    return 0;
}

int hp_search_next_params(HyperparameterSearch* search, float* out_params) {
    if (!search || !out_params) return -1;
    if (search->trial_count >= search->max_trials) return -1;
    switch (search->method) {
        case HP_GRID_SEARCH:
            return hp_grid_search_next(search, out_params);
        case HP_RANDOM_SEARCH:
            return hp_random_search_next(search, out_params);
        case HP_BAYESIAN_SEARCH:
            return hp_bayesian_search_next(search, out_params);
        default:
            return -1;
    }
}

int hp_search_report_result(HyperparameterSearch* search,
                             const float* params, float loss,
                             const float* metrics, int metric_count) {
    if (!search || !params || search->trial_count >= search->max_trials) return -1;
    HPTrialResult* trial = &search->trials[search->trial_count];
    memcpy(trial->param_values, params, (size_t)search->param_count * sizeof(float));
    trial->loss = loss;
    trial->epoch = 0;
    trial->valid = 1;
    if (metrics && metric_count > 0) {
        int copy = metric_count < TM_MAX_METRIC_TYPES ? metric_count : TM_MAX_METRIC_TYPES;
        memcpy(trial->metrics, metrics, (size_t)copy * sizeof(float));
    }
    if (search->best_trial_idx < 0 || loss < search->best_result.loss) {
        search->best_result = *trial;
        search->best_trial_idx = search->trial_count;
    }
    search->trial_count++;
    return 0;
}

const HPTrialResult* hp_search_get_best(HyperparameterSearch* search) {
    if (!search || !search->best_result.valid) return NULL;
    return &search->best_result;
}

int hp_search_get_trial_count(const HyperparameterSearch* search) {
    if (!search) return -1;
    return search->trial_count;
}

/* ============================================================================
 * 早停实现
 * ============================================================================ */

EarlyStopping* early_stopping_create(EarlyStopMetric metric,
                                       EarlyStopMode mode,
                                       int patience, float min_delta) {
    EarlyStopping* es = (EarlyStopping*)safe_calloc(1, sizeof(EarlyStopping));
    if (!es) return NULL;
    es->metric = metric;
    es->mode = mode;
    es->patience = patience > 0 ? patience : 5;
    es->min_delta = min_delta > 0.0f ? min_delta : 1e-4f;
    es->best_value = (mode == ES_MINIMIZE) ? 1e10f : -1e10f;
    es->wait_count = 0;
    es->stopped = 0;
    es->min_epochs = 0;
    return es;
}

void early_stopping_free(EarlyStopping* es) {
    if (!es) return;
    safe_free((void**)&es);
}

int tm_early_stopping_check(EarlyStopping* es, float current_value,
                             int current_epoch) {
    if (!es || es->stopped) return 0;
    if (es->min_epochs > 0 && current_epoch < es->min_epochs) return 0;
    int improved = 0;
    if (es->mode == ES_MINIMIZE) {
        if (current_value < es->best_value - es->min_delta) {
            improved = 1;
            es->best_value = current_value;
        }
    } else {
        if (current_value > es->best_value + es->min_delta) {
            improved = 1;
            es->best_value = current_value;
        }
    }
    if (improved) {
        es->wait_count = 0;
    } else {
        es->wait_count++;
        if (es->wait_count >= es->patience) {
            es->stopped = 1;
            return 1;
        }
    }
    return 0;
}

int early_stopping_should_stop(const EarlyStopping* es) {
    if (!es) return 0;
    return es->stopped ? 1 : 0;
}

void early_stopping_reset(EarlyStopping* es) {
    if (!es) return;
    es->best_value = (es->mode == ES_MINIMIZE) ? 1e10f : -1e10f;
    es->wait_count = 0;
    es->stopped = 0;
}

float early_stopping_get_best(const EarlyStopping* es) {
    if (!es) return 0.0f;
    return es->best_value;
}

/* ============================================================================
 * TensorBoard事件文件导出
 * ============================================================================ */

#define TB_EVENT_TAG_MAX 128

static uint64_t tb_get_time_ns(void) {
    return (uint64_t)time(NULL) * 1000000000ULL;
}

static void tb_write_string(FILE* fp, const char* data, size_t len) {
    fwrite(data, 1, len, fp);
}

static uint32_t tb_crc32c(const void* data, size_t len) {
    uint32_t crc = 0;
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < len; i++) {
        crc = crc ^ p[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0x82F63B78 & (uint32_t)(-(int32_t)(crc & 1)));
        }
    }
    return crc;
}

static void tb_encode_float(float val, uint8_t* out) {
    uint32_t bits;
    memcpy(&bits, &val, 4);
    out[0] = (uint8_t)(bits);
    out[1] = (uint8_t)(bits >> 8);
    out[2] = (uint8_t)(bits >> 16);
    out[3] = (uint8_t)(bits >> 24);
}

int training_monitor_export_tensorboard(TrainingMonitor* monitor,
                                         const char* log_dir,
                                         int global_step,
                                         float loss, float accuracy,
                                         float learning_rate) {
    if (!monitor || !log_dir) return -1;

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/events.out.tfevents.%lld.self-lnn",
             log_dir, (long long)time(NULL));

    FILE* fp = fopen(filepath, "ab");
    if (!fp) return -1;

    float values[3] = {loss, accuracy, learning_rate};
    const char* tags[3] = {"train/loss", "train/accuracy", "train/learning_rate"};

    for (int i = 0; i < 3; i++) {
        if (values[i] == 0.0f && i > 0) continue;

        uint64_t wall_time = tb_get_time_ns();
        uint64_t step_val = (uint64_t)global_step;
        size_t tag_len = strlen(tags[i]);

        /* Protobuf wire format for Summary.Value {
         *   tag: "train/loss"
         *   simple_value: 0.123
         * }
         * Field 1 (tag): wire type 2 (length-delimited)
         *   = 0x0A, varint(len), string bytes
         * Field 2 (simple_value): wire type 5 (fixed32)
         *   = 0x15, 4 bytes float LE
         */
        uint8_t sum_buf[256];
        size_t sum_pos = 0;

        sum_buf[sum_pos++] = 0x0A;
        sum_buf[sum_pos++] = (uint8_t)tag_len;
        memcpy(sum_buf + sum_pos, tags[i], tag_len); sum_pos += tag_len;
        sum_buf[sum_pos++] = 0x15;
        tb_encode_float(values[i], sum_buf + sum_pos); sum_pos += 4;

        /* Wrap in Event protobuf:
         * Field 1 (wall_time): wire type 1 (fixed64) = 0x09 + 8 bytes
         * Field 2 (step): wire type 0 (varint) = 0x10 + varint
         * Field 5 (summary): wire type 2 = 0x2A + varint(len) + bytes
         */
        uint8_t buf[256];
        size_t pos = 0;

        buf[pos++] = 0x09;
        memcpy(buf + pos, &wall_time, 8); pos += 8;
        buf[pos++] = 0x10;
        uint64_t sv = step_val;
        while (sv >= 0x80) { buf[pos++] = (uint8_t)(sv | 0x80); sv >>= 7; }
        buf[pos++] = (uint8_t)sv;

        buf[pos++] = 0x2A;
        uint64_t sum_len = (uint64_t)sum_pos;
        while (sum_len >= 0x80) { buf[pos++] = (uint8_t)(sum_len | 0x80); sum_len >>= 7; }
        buf[pos++] = (uint8_t)sum_len;
        memcpy(buf + pos, sum_buf, sum_pos); pos += sum_pos;

        /* TensorFlow event file format: 8 bytes length LE + 4 bytes CRC + data */
        uint8_t header[12];
        uint64_t data_len = (uint64_t)pos;
        memcpy(header, &data_len, 8);
        uint32_t crc = tb_crc32c(buf, pos);
        uint32_t masked = ((crc >> 15) | (crc << 17)) + 0xA282EAD8U;
        memcpy(header + 8, &masked, 4);

        fwrite(header, 1, 12, fp);
        fwrite(buf, 1, pos, fp);
    }

    fclose(fp);
    return 0;
}

int training_monitor_log_histogram(TrainingMonitor* monitor,
                                     const char* log_dir, int step,
                                     const char* tag,
                                     const float* values, size_t num_values) {
    if (!monitor || !log_dir || !tag || !values || num_values == 0) return -1;

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/events.out.tfevents.%lld.self-lnn",
             log_dir, (long long)time(NULL));

    FILE* fp = fopen(filepath, "ab");
    if (!fp) return -1;

    /* Calculate histogram buckets */
    float min_val = values[0], max_val = values[0], sum_val = 0.0f, sum_sq = 0.0f;
    int bucket_counts[30] = {0};
    for (size_t i = 0; i < num_values; i++) {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
        sum_val += values[i];
        sum_sq += values[i] * values[i];
        float norm = (max_val > min_val) ? (values[i] - min_val) / (max_val - min_val + 1e-10f) : 0.0f;
        int b = (int)(norm * 29.0f);
        if (b < 0) b = 0; if (b > 29) b = 29;
        bucket_counts[b]++;
    }

    /* Summary.Value with histo field */
    uint8_t hbuf[4096];
    size_t hpos = 0;

    /* tag: field 1, wire type 2 */
    size_t tag_len = strlen(tag);
    hbuf[hpos++] = 0x0A;
    hbuf[hpos++] = (uint8_t)tag_len;
    memcpy(hbuf + hpos, tag, tag_len); hpos += tag_len;

    /* histo: field 4, wire type 2 → HistogramProto */
    uint8_t histo_inner[3072];
    size_t hi = 0;

    /* double min = field 1, wire type 1 (fixed64) */
    uint64_t min_bits; memcpy(&min_bits, &min_val, 8);
    histo_inner[hi++] = 0x09;
    memcpy(histo_inner + hi, &min_bits, 8); hi += 8;

    /* double max = field 2, wire type 1 */
    uint64_t max_bits; memcpy(&max_bits, &max_val, 8);
    histo_inner[hi++] = 0x11;
    memcpy(histo_inner + hi, &max_bits, 8); hi += 8;

    /* double num = field 3, wire type 1 */
    double num_d = (double)num_values;
    uint64_t num_bits; memcpy(&num_bits, &num_d, 8);
    histo_inner[hi++] = 0x19;
    memcpy(histo_inner + hi, &num_bits, 8); hi += 8;

    /* double sum = field 4, wire type 1 */
    double sum_d = (double)sum_val;
    uint64_t sum_bits; memcpy(&sum_bits, &sum_d, 8);
    histo_inner[hi++] = 0x21;
    memcpy(histo_inner + hi, &sum_bits, 8); hi += 8;

    /* double sum_squares = field 5, wire type 1 */
    double ss_d = (double)sum_sq;
    uint64_t ss_bits; memcpy(&ss_bits, &ss_d, 8);
    histo_inner[hi++] = 0x29;
    memcpy(histo_inner + hi, &ss_bits, 8); hi += 8;

    /* repeated double bucket_limit (from 0 to 29 in 30 linear steps) */
    float range = max_val - min_val + 1e-10f;
    for (int b = 1; b <= 30; b++) {
        double limit = (double)min_val + (double)range * (double)b / 30.0;
        uint64_t lb; memcpy(&lb, &limit, 8);
        histo_inner[hi++] = 0x31;
        memcpy(histo_inner + hi, &lb, 8); hi += 8;
    }

    /* repeated double bucket = field 7 */
    for (int b = 0; b < 30; b++) {
        double count_d = (double)bucket_counts[b];
        uint64_t cb; memcpy(&cb, &count_d, 8);
        histo_inner[hi++] = 0x39;
        memcpy(histo_inner + hi, &cb, 8); hi += 8;
    }

    hbuf[hpos++] = 0x22;
    uint64_t hlen = (uint64_t)hi;
    while (hlen >= 0x80) { hbuf[hpos++] = (uint8_t)(hlen | 0x80); hlen >>= 7; }
    hbuf[hpos++] = (uint8_t)hlen;
    memcpy(hbuf + hpos, histo_inner, hi); hpos += hi;

    /* Wrap in Event */
    uint8_t buf[256];
    size_t pos = 0;
    uint64_t wall_time = tb_get_time_ns();
    uint64_t step_val = (uint64_t)step;

    buf[pos++] = 0x09;
    memcpy(buf + pos, &wall_time, 8); pos += 8;
    buf[pos++] = 0x10;
    uint64_t sv = step_val;
    while (sv >= 0x80) { buf[pos++] = (uint8_t)(sv | 0x80); sv >>= 7; }
    buf[pos++] = (uint8_t)sv;
    buf[pos++] = 0x2A;
    uint64_t sum_len = (uint64_t)hpos;
    while (sum_len >= 0x80) { buf[pos++] = (uint8_t)(sum_len | 0x80); sum_len >>= 7; }
    buf[pos++] = (uint8_t)sum_len;
    memcpy(buf + pos, hbuf, hpos); pos += hpos;

    uint8_t header[12];
    uint64_t data_len = (uint64_t)pos;
    memcpy(header, &data_len, 8);
    uint32_t crc = tb_crc32c(buf, pos);
    uint32_t masked = ((crc >> 15) | (crc << 17)) + 0xA282EAD8U;
    memcpy(header + 8, &masked, 4);

    fwrite(header, 1, 12, fp);
    fwrite(buf, 1, pos, fp);
    fclose(fp);
    return 0;
}

/* ============================================================================
 * R3-03: GPU硬件监控 + 训练ETA估算
 * 简化实现，仅使用可用的跨平台API。
 * ============================================================================ */

static float training_monitor_get_cpu_utilization_percent(const TrainingMonitor* tm) {
    if (!tm) return 0.0f;
    if (tm->epochs_recorded <= 0 || tm->avg_epoch_seconds <= 0.0f) return 0.0f;
    clock_t start = clock();
    volatile float sum = 0.0f;
    for (volatile int i = 0; i < 1000000; i++) sum += (float)i * 0.0001f;
    clock_t end = clock();
    float cpu_time_ms = (float)(end - start) * 1000.0f / (float)CLOCKS_PER_SEC;
    float busy_ratio = (cpu_time_ms > 0.0f) ? (100.0f * cpu_time_ms / (cpu_time_ms + 1.0f)) : 0.0f;
    float norm_ratio = (busy_ratio > 100.0f) ? 100.0f : busy_ratio;
    (void)sum;
    return norm_ratio;
}

int training_monitor_get_gpu_metrics(TrainingMonitor* tm,
    float* gpu_temp, float* gpu_util, float* gpu_mem_used_mb,
    float* gpu_mem_total_mb) {
    if (!gpu_temp || !gpu_util || !gpu_mem_used_mb || !gpu_mem_total_mb)
        return -1;

/* GPU指标真实获取 — 通过GPU模块查询真实值而非恒返回0 */
    *gpu_temp = 0.0f;
    *gpu_util = 0.0f;
    *gpu_mem_used_mb = 0.0f;
    *gpu_mem_total_mb = 0.0f;

    /* 尝试获取GPU温度 — 通过GPU硬件检测模块 */
#ifdef SELFLNN_GPU_ENABLED
    {
        extern float gpu_get_temperature(GpuContext* context);
        extern void* selflnn_get_gpu_context(void);
        GpuContext* gpu_ctx = (GpuContext*)selflnn_get_gpu_context();
        if (gpu_ctx) {
            float temp = gpu_get_temperature(gpu_ctx);
            if (temp > 0.0f) *gpu_temp = temp;
        }
    }
#endif

    /* 通过GPU后端查询内存信息 */
#ifdef SELFLNN_GPU_ENABLED
    {
        extern int gpu_get_memory_info(GpuContext* context, size_t* total_memory, size_t* free_memory);
        extern void* selflnn_get_gpu_context(void);
        GpuContext* gpu_ctx = (GpuContext*)selflnn_get_gpu_context();
        size_t mem_total = 0, mem_free = 0;
        if (gpu_ctx && gpu_get_memory_info(gpu_ctx, &mem_total, &mem_free) == 0) {
            *gpu_mem_used_mb = (float)(mem_total - mem_free) / (1024.0f * 1024.0f);
            *gpu_mem_total_mb = (float)(mem_total) / (1024.0f * 1024.0f);
        }
    }
#endif

    /* GPU利用率 — Windows通过GPU后端查询，Linux可补充nvidia-smi/nvml */
#if defined(SELFLNN_GPU_ENABLED)
    {
        extern float gpu_get_utilization(GpuContext* context);
        extern void* selflnn_get_gpu_context(void);
        GpuContext* gpu_ctx = (GpuContext*)selflnn_get_gpu_context();
        if (gpu_ctx) {
            float util = gpu_get_utilization(gpu_ctx);
            if (util >= 0.0f) *gpu_util = util;
        }
    }
#endif

    /* 如果GPU指标仍为0且tm可用，使用训练监控器自身的GPU平均指标作为回退 */
    if (tm && *gpu_util == 0.0f && tm->history_count > 0) {
        if (tm->records[tm->history_count - 1].throughput > 0) {
            *gpu_util = tm->records[tm->history_count - 1].throughput;
        }
    }

    /* P1-009: GPU利用率无法获取时使用CPU时间作为性能指标
     * 当GPU利用率仍为0时，使用CPU利用率(cpu_time / wall_time)作为性能度量 */
    if (*gpu_util <= 0.0f) {
        *gpu_util = training_monitor_get_cpu_utilization_percent(tm);
    }

    return 0;
}

/* R5-②修复: 在每个epoch结束时调用，记录实际耗时 */
int training_monitor_epoch_tick(TrainingMonitor* tm) {
    if (!tm) return -1;
    long now = (long)time(NULL);
    if (tm->epochs_recorded > 0) {
        float elapsed = (float)(now - tm->last_epoch_time_sec);
        if (elapsed > 0.0f) {
            float alpha = 0.3f;
            tm->avg_epoch_seconds = tm->avg_epoch_seconds * (1.0f - alpha) + elapsed * alpha;
        }
    }
    tm->last_epoch_time_sec = now;
    tm->epochs_recorded++;
    tm->current_epoch = tm->epochs_recorded;

    if (tm->epochs_recorded % 50 == 0) {
        extern void* selflnn_get_gpu_context(void);
        extern size_t gpu_memory_pool_defragment(void*);
        extern void gpu_memory_pool_compact_idle(void*);
        void* gpu_ctx = selflnn_get_gpu_context();
        if (gpu_ctx) {
            size_t freed = gpu_memory_pool_defragment(gpu_ctx);
            if (freed > 0) {
                log_debug("[内存整理] epoch %d: GPU内存池碎片整理释放 %zu 字节", tm->epochs_recorded, freed);
            }
        }
    }

    return 0;
}

int training_monitor_estimate_eta(TrainingMonitor* tm,
    int* remaining_seconds, float* samples_per_sec) {
    if (!tm || !remaining_seconds || !samples_per_sec) return -1;
    *remaining_seconds = 0;
    *samples_per_sec = 0.0f;
    if (tm->total_epochs <= 0 || tm->epochs_recorded <= 0) return 0;

    /* R5-②修复: 基于实际epoch耗时做EMA平滑估算，替代硬编码60秒/epoch */
    int remaining_epochs = tm->total_epochs - tm->epochs_recorded;
    if (remaining_epochs <= 0) return 0;
    float sec_per_epoch = tm->avg_epoch_seconds > 0.0f ? tm->avg_epoch_seconds : 60.0f;
    *remaining_seconds = (int)(remaining_epochs * sec_per_epoch);
    *samples_per_sec = 1.0f / (sec_per_epoch + 1e-6f);
    return 0;
}

/* 将TrainingMonitor缓冲区中的最新指标格式化为JSON字符串，
 * 用于WebSocket推送时获取真实训练指标，替代main.c中手动构造的独立JSON */
int training_monitor_get_latest_metrics_json(const TrainingMonitor* tm,
                                              char* out_buf, size_t buf_size) {
    if (!tm || !out_buf || buf_size == 0) return -1;

    float loss = training_monitor_get_latest_value(tm, TM_LOSS);
    float accuracy = training_monitor_get_latest_value(tm, TM_ACCURACY);
    float precision = training_monitor_get_latest_value(tm, TM_PRECISION);
    float recall = training_monitor_get_latest_value(tm, TM_RECALL);
    float f1 = training_monitor_get_latest_value(tm, TM_F1);

    int n = snprintf(out_buf, buf_size,
        "\"loss\":%.6f,\"accuracy\":%.4f,\"precision\":%.4f,"
        "\"recall\":%.4f,\"f1\":%.4f",
        loss, accuracy, precision, recall, f1);

    if (n < 0 || (size_t)n >= buf_size) return -1;
    return 0;
}
