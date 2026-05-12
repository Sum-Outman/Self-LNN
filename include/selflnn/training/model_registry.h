/**
 * @file model_registry.h
 * @brief 模型版本注册管理增强系统接口
 */

#ifndef SELFLNN_MODEL_REGISTRY_H
#define SELFLNN_MODEL_REGISTRY_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MR_MAX_MODELS 128
#define MR_MAX_VERSIONS 256
#define MR_MAX_NAME 128
#define MR_MAX_PATH 512

typedef struct {
    int version_id;
    int model_id;
    char name[MR_MAX_NAME];
    char file_path[MR_MAX_PATH];
    float train_loss;
    float val_loss;
    float accuracy;
    float f1_score;
    size_t parameter_count;
    size_t model_size_bytes;
    time_t created_at;
    time_t deployed_at;
    int is_deployed;
    int is_stable;
    char training_config[MR_MAX_PATH];
    int parent_version;
    char notes[256];
} ModelVersion;

typedef struct {
    int model_id;
    char name[MR_MAX_NAME];
    char description[256];
    char task_type[64];
    ModelVersion* versions;
    int version_count;
    int version_capacity;
    int current_version;
    int stable_version;
    int ab_test_active;
    int ab_version_a;
    int ab_version_b;
    float ab_split_ratio;
    int* ab_results_a;
    int* ab_results_b;
    int ab_result_count;
} ModelEntry;

typedef struct {
    size_t total_models;
    size_t total_versions;
    size_t total_storage;
    int deployed_count;
    int active_ab_tests;
} ModelRegistryStats;

typedef struct ModelRegistry ModelRegistry;

ModelRegistry* model_registry_create(const char* storage_dir);
void model_registry_free(ModelRegistry* mr);

/* 注册与版本 */
int mr_register_model(ModelRegistry* mr, const char* name, const char* description, const char* task_type, int* model_id);
int mr_add_version(ModelRegistry* mr, int model_id, const char* file_path, const ModelVersion* ver, int* version_id);
int mr_get_version(const ModelRegistry* mr, int model_id, int version_id, ModelVersion* out);
int mr_get_latest_version(const ModelRegistry* mr, int model_id, ModelVersion* out);

/* 比较与选择 */
int mr_compare_versions(const ModelRegistry* mr, int model_id, int v1, int v2, float* loss_diff, float* acc_diff);
int mr_set_stable_version(ModelRegistry* mr, int model_id, int version_id);
int mr_rollback(ModelRegistry* mr, int model_id, int version_id);

/* A/B测试 */
int mr_start_ab_test(ModelRegistry* mr, int model_id, int version_a, int version_b, float split_ratio);
int mr_record_ab_result(ModelRegistry* mr, int model_id, int version_used, int success);
int mr_get_ab_stats(const ModelRegistry* mr, int model_id, float* rate_a, float* rate_b, int* winner);
int mr_stop_ab_test(ModelRegistry* mr, int model_id, int deploy_winner);

/* 血缘追踪 */
int mr_trace_lineage(const ModelRegistry* mr, int model_id, int version_id, int* lineage, int max_count);

/* 统计 */
int mr_get_stats(const ModelRegistry* mr, ModelRegistryStats* stats);
int mr_list_models(const ModelRegistry* mr, int* model_ids, int max_count);

#ifdef __cplusplus
}
#endif
#endif
