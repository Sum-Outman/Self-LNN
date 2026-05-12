#ifndef SELFLNN_TRAINING_MONITOR_H
#define SELFLNN_TRAINING_MONITOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 训练指标追踪 (A08.3.2)
 * ============================================================================ */

#define TM_MAX_METRIC_TYPES 16
#define TM_MAX_HISTORY_PER_TYPE 1024
#define TM_MAX_PARAM_NAME 64
#define TM_MAX_PARAMS 16
#define TM_MAX_TRIALS 1000

typedef enum {
    TM_LOSS = 0,
    TM_ACCURACY,
    TM_PRECISION,
    TM_RECALL,
    TM_F1,
    TM_PERPLEXITY,
    TM_CUSTOM
} MetricType;

typedef struct {
    MetricType type;
    int epoch;
    int step;
    float value;
    double timestamp_sec;
    char custom_name[64];
} MetricRecord;

typedef struct {
    MetricRecord records[TM_MAX_HISTORY_PER_TYPE];
    int count;
    int head;
    MetricType type;
} MetricHistory;

typedef struct {
    MetricHistory histories[TM_MAX_METRIC_TYPES];
    int history_count;
    const char* run_name;
    int current_epoch;
    int total_epochs;
} TrainingMonitor;

TrainingMonitor* training_monitor_create(const char* run_name, int total_epochs);
void training_monitor_free(TrainingMonitor* tm);

int training_monitor_log_metric(TrainingMonitor* tm, MetricType type,
                                 int epoch, int step, float value);
int training_monitor_log_custom(TrainingMonitor* tm, const char* name,
                                 int epoch, int step, float value);
int training_monitor_get_metric_history(const TrainingMonitor* tm,
                                         MetricType type,
                                         MetricRecord* out_buffer,
                                         int max_count);
float training_monitor_get_best_value(const TrainingMonitor* tm,
                                       MetricType type, int minimize);
float training_monitor_get_latest_value(const TrainingMonitor* tm,
                                         MetricType type);
int training_monitor_reset(TrainingMonitor* tm);

/* ============================================================================
 * 超参数调优
 * ============================================================================ */

typedef enum {
    HP_TYPE_FLOAT = 0,
    HP_TYPE_INT,
    HP_TYPE_CATEGORICAL
} HPParamType;

typedef struct {
    char name[TM_MAX_PARAM_NAME];
    HPParamType type;
    float min_val;
    float max_val;
    float step;
    int num_categories;
    float categories[16];
} HPParamConfig;

typedef struct {
    float param_values[TM_MAX_PARAMS];
    float loss;
    float metrics[TM_MAX_METRIC_TYPES];
    int epoch;
    int valid;
} HPTrialResult;

typedef enum {
    HP_GRID_SEARCH = 0,
    HP_RANDOM_SEARCH,
    HP_BAYESIAN_SEARCH
} HPSearchMethod;

typedef struct {
    HPParamConfig params[TM_MAX_PARAMS];
    int param_count;
    HPSearchMethod method;
    HPTrialResult trials[TM_MAX_TRIALS];
    int trial_count;
    int max_trials;
    HPTrialResult best_result;
    int best_trial_idx;
    unsigned int rng_state;
} HyperparameterSearch;

HyperparameterSearch* hp_search_create(HPSearchMethod method, int max_trials);
void hp_search_free(HyperparameterSearch* search);

int hp_search_add_param_float(HyperparameterSearch* search,
                               const char* name, float min_val, float max_val);
int hp_search_add_param_int(HyperparameterSearch* search,
                             const char* name, int min_val, int max_val);
int hp_search_add_param_categorical(HyperparameterSearch* search,
                                     const char* name,
                                     const float* values, int num_values);
int hp_search_next_params(HyperparameterSearch* search, float* out_params);
int hp_search_report_result(HyperparameterSearch* search,
                             const float* params, float loss,
                             const float* metrics, int metric_count);
const HPTrialResult* hp_search_get_best(HyperparameterSearch* search);
int hp_search_get_trial_count(const HyperparameterSearch* search);

/* ============================================================================
 * 早停
 * ============================================================================ */

typedef enum {
    ES_VAL_LOSS = 0,
    ES_VAL_ACCURACY,
    ES_VAL_F1
} EarlyStopMetric;

typedef enum {
    ES_MINIMIZE = 0,
    ES_MAXIMIZE
} EarlyStopMode;

typedef struct {
    EarlyStopMetric metric;
    EarlyStopMode mode;
    int patience;
    float min_delta;
    float best_value;
    int wait_count;
    int stopped;
    int min_epochs;
} EarlyStopping;

EarlyStopping* early_stopping_create(EarlyStopMetric metric,
                                       EarlyStopMode mode,
                                       int patience, float min_delta);
void early_stopping_free(EarlyStopping* es);
int tm_early_stopping_check(EarlyStopping* es, float current_value,
                            int current_epoch);
int early_stopping_should_stop(const EarlyStopping* es);
void early_stopping_reset(EarlyStopping* es);
float early_stopping_get_best(const EarlyStopping* es);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_TRAINING_MONITOR_H */
