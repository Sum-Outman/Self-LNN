/**
 * @file model_registry.c
 * @brief 模型版本注册管理系统完整实现
 */
#include "selflnn/training/model_registry.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(p,m) _mkdir(p)
#endif

struct ModelRegistry {
    ModelEntry models[MR_MAX_MODELS];
    int model_count;
    char storage_dir[512];
    int next_model_id;
    int next_version_id;
};

ModelRegistry* model_registry_create(const char* storage_dir) {
    ModelRegistry* mr = (ModelRegistry*)safe_calloc(1, sizeof(ModelRegistry));
    if (!mr) return NULL;
    if (storage_dir) { snprintf(mr->storage_dir, sizeof(mr->storage_dir), "%s", storage_dir); mkdir(storage_dir, 0755); }
    else { snprintf(mr->storage_dir, sizeof(mr->storage_dir), "./model_registry"); mkdir("./model_registry", 0755); }
    mr->next_model_id = 1;
    mr->next_version_id = 1;
    return mr;
}

void model_registry_free(ModelRegistry* mr) {
    if (!mr) return;
    for (int i = 0; i < mr->model_count; i++) { safe_free((void**)&mr->models[i].versions); }
    safe_free((void**)&mr);
}

int mr_register_model(ModelRegistry* mr, const char* name, const char* description, const char* task_type, int* model_id) {
    if (!mr || !name || !model_id || mr->model_count >= MR_MAX_MODELS) return -1;
    ModelEntry* e = &mr->models[mr->model_count];
    memset(e, 0, sizeof(ModelEntry));
    e->model_id = mr->next_model_id++;
    snprintf(e->name, MR_MAX_NAME, "%s", name);
    if (description) snprintf(e->description, sizeof(e->description), "%s", description);
    if (task_type) snprintf(e->task_type, sizeof(e->task_type), "%s", task_type);
    e->current_version = -1;
    e->stable_version = -1;
    *model_id = e->model_id;
    mr->model_count++;
    return 0;
}

static ModelEntry* find_model(ModelRegistry* mr, int model_id) {
    for (int i = 0; i < mr->model_count; i++) if (mr->models[i].model_id == model_id) return &mr->models[i];
    return NULL;
}

int mr_add_version(ModelRegistry* mr, int model_id, const char* file_path, const ModelVersion* ver, int* version_id) {
    ModelEntry* e = find_model(mr, model_id);
    if (!e || !ver || !version_id) return -1;
    if (!e->versions) { e->version_capacity = 32; e->versions = (ModelVersion*)safe_calloc(e->version_capacity, sizeof(ModelVersion)); }
    else if (e->version_count >= e->version_capacity) {
        size_t nc = e->version_capacity * 2;
        void* nv = safe_realloc(e->versions, nc * sizeof(ModelVersion));
        if (!nv) return -1;
        e->versions = (ModelVersion*)nv;
        memset(e->versions + e->version_capacity, 0, e->version_capacity * sizeof(ModelVersion));
        e->version_capacity = (int)nc;
    }
    int vid = mr->next_version_id++;
    ModelVersion* mv = &e->versions[e->version_count];
    memcpy(mv, ver, sizeof(ModelVersion));
    mv->version_id = vid;
    mv->model_id = model_id;
    mv->created_at = time(NULL);
    if (file_path) snprintf(mv->file_path, sizeof(mv->file_path), "%s", file_path);
    e->version_count++;
    e->current_version = vid;
    *version_id = vid;
    return 0;
}

int mr_get_version(const ModelRegistry* mr, int model_id, int version_id, ModelVersion* out) {
    ModelEntry* e = find_model((ModelRegistry*)mr, model_id);
    if (!e || !out) return -1;
    for (int i = 0; i < e->version_count; i++) {
        if (e->versions[i].version_id == version_id) { memcpy(out, &e->versions[i], sizeof(ModelVersion)); return 0; }
    }
    return -1;
}

int mr_get_latest_version(const ModelRegistry* mr, int model_id, ModelVersion* out) {
    ModelEntry* e = find_model((ModelRegistry*)mr, model_id);
    if (!e || !out || e->current_version < 0) return -1;
    return mr_get_version(mr, model_id, e->current_version, out);
}

int mr_compare_versions(const ModelRegistry* mr, int model_id, int v1, int v2, float* loss_diff, float* acc_diff) {
    ModelVersion ver1, ver2;
    if (mr_get_version(mr, model_id, v1, &ver1) != 0 || mr_get_version(mr, model_id, v2, &ver2) != 0) return -1;
    if (loss_diff) *loss_diff = ver1.val_loss - ver2.val_loss;
    if (acc_diff) *acc_diff = ver1.accuracy - ver2.accuracy;
    return 0;
}

int mr_set_stable_version(ModelRegistry* mr, int model_id, int version_id) {
    ModelEntry* e = find_model(mr, model_id);
    if (!e) return -1;
    e->stable_version = version_id;
    ModelVersion* ver = NULL;
    for (int i = 0; i < e->version_count; i++) {
        if (e->versions[i].version_id == version_id) { ver = &e->versions[i]; break; }
    }
    if (ver) { ver->is_stable = 1; ver->is_deployed = 1; ver->deployed_at = time(NULL); }
    return 0;
}

int mr_rollback(ModelRegistry* mr, int model_id, int version_id) {
    ModelEntry* e = find_model(mr, model_id);
    if (!e) return -1;
    e->current_version = version_id;
    return 0;
}

int mr_start_ab_test(ModelRegistry* mr, int model_id, int version_a, int version_b, float split_ratio) {
    ModelEntry* e = find_model(mr, model_id);
    if (!e) return -1;
    e->ab_test_active = 1;
    e->ab_version_a = version_a;
    e->ab_version_b = version_b;
    e->ab_split_ratio = split_ratio;
    safe_free((void**)&e->ab_results_a);
    safe_free((void**)&e->ab_results_b);
    e->ab_result_count = 0;
    return 0;
}

int mr_record_ab_result(ModelRegistry* mr, int model_id, int version_used, int success) {
    ModelEntry* e = find_model(mr, model_id);
    if (!e || !e->ab_test_active) return -1;
    int new_count = e->ab_result_count + 1;
    int* na = (int*)safe_realloc(e->ab_results_a, new_count * sizeof(int));
    int* nb = (int*)safe_realloc(e->ab_results_b, new_count * sizeof(int));
    if (!na || !nb) { safe_free((void**)&na); safe_free((void**)&nb); return -1; }
    e->ab_results_a = na; e->ab_results_b = nb;
    e->ab_results_a[e->ab_result_count] = (version_used == e->ab_version_a) ? success : 0;
    e->ab_results_b[e->ab_result_count] = (version_used == e->ab_version_b) ? success : 0;
    e->ab_result_count = new_count;
    return 0;
}

int mr_get_ab_stats(const ModelRegistry* mr, int model_id, float* rate_a, float* rate_b, int* winner) {
    ModelEntry* e = find_model((ModelRegistry*)mr, model_id);
    if (!e || !rate_a || !rate_b || !winner) return -1;
    int sa = 0, sb = 0, ta = 0, tb = 0;
    for (int i = 0; i < e->ab_result_count; i++) {
        if (e->ab_results_a[i] > 0) { sa++; ta++; }
        else if (e->ab_results_a[i] == 0) ta++;
        if (e->ab_results_b[i] > 0) { sb++; tb++; }
        else if (e->ab_results_b[i] == 0) tb++;
    }
    *rate_a = ta > 0 ? (float)sa / (float)ta : 0.0f;
    *rate_b = tb > 0 ? (float)sb / (float)tb : 0.0f;
    *winner = (*rate_a > *rate_b) ? e->ab_version_a : e->ab_version_b;
    return 0;
}

int mr_stop_ab_test(ModelRegistry* mr, int model_id, int deploy_winner) {
    ModelEntry* e = find_model(mr, model_id);
    if (!e) return -1;
    if (deploy_winner) {
        float ra, rb; int w;
        if (mr_get_ab_stats(mr, model_id, &ra, &rb, &w) == 0) mr_set_stable_version(mr, model_id, w);
    }
    e->ab_test_active = 0;
    return 0;
}

int mr_trace_lineage(const ModelRegistry* mr, int model_id, int version_id, int* lineage, int max_count) {
    ModelEntry* e = find_model((ModelRegistry*)mr, model_id);
    if (!e || !lineage) return 0;
    int count = 0;
    ModelVersion* cur = NULL;
    for (int i = 0; i < e->version_count; i++) if (e->versions[i].version_id == version_id) { cur = &e->versions[i]; break; }
    while (cur && count < max_count) {
        lineage[count++] = cur->version_id;
        int parent = cur->parent_version;
        cur = NULL;
        if (parent > 0)
            for (int i = 0; i < e->version_count; i++)
                if (e->versions[i].version_id == parent) { cur = &e->versions[i]; break; }
    }
    return count;
}

int mr_get_stats(const ModelRegistry* mr, ModelRegistryStats* stats) {
    if (!mr || !stats) return -1;
    memset(stats, 0, sizeof(ModelRegistryStats));
    for (int i = 0; i < mr->model_count; i++) {
        stats->total_models++;
        stats->total_versions += mr->models[i].version_count;
        if (mr->models[i].stable_version > 0) stats->deployed_count++;
        if (mr->models[i].ab_test_active) stats->active_ab_tests++;
    }
    return 0;
}

int mr_list_models(const ModelRegistry* mr, int* model_ids, int max_count) {
    if (!mr || !model_ids) return 0;
    int count = mr->model_count < max_count ? mr->model_count : max_count;
    for (int i = 0; i < count; i++) model_ids[i] = mr->models[i].model_id;
    return count;
}
