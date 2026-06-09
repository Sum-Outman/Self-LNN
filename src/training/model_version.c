/**
 * @file model_version.c
 * @brief 模型版本管理模块实现
 *
 * 提供完整的模型版本管理功能：
 * - 版本快照创建和恢复
 * - 版本差异比较（逐参数）
 * - 自动清理和版本裁剪
 * - JSON导出
 * - 锁定保护
 */

#include "selflnn/training/model_version.h"
#include "selflnn/training/training.h"
#include "selflnn/utils/perf.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/errors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
/* 跨平台目录创建宏 */
#ifdef _WIN32
#include <direct.h>
#define mkdir_p(path) _mkdir(path)
#else
#include <sys/stat.h>
#define mkdir_p(path) mkdir(path, 0755)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ====================================================================
 * 内部常量
 * ==================================================================== */

/** 默认版本存储目录 */
#define DEFAULT_VERSIONS_DIR "model_versions"

/** 初始版本数组容量 */
#define INITIAL_VERSIONS_CAPACITY 16

/** 版本数组容量增长因子 */
#define CAPACITY_GROWTH_FACTOR 2

/** 快照文件名后缀 */
#define SNAPSHOT_FILE_SUFFIX ".snapshot.bin"

/** 网络文件后缀 */
#define NETWORK_FILE_SUFFIX ".network.bin"

/** 版本信息文件后缀 */
#define INFO_FILE_SUFFIX ".version.info"

/** 魔术字符串 */
#define VERSION_MAGIC "SELF-LNN-VER"

/** 版本文件格式版本 */
#define VERSION_FILE_FORMAT_VERSION 1

/** 差异计算阈值 */
#define DIFF_THRESHOLD 1e-6f

/* ====================================================================
 * 内部结构体
 * ==================================================================== */

/**
 * @brief 版本快照文件头部
 */
typedef struct {
    char magic[16];                    /**< 魔术字符串 */
    uint32_t format_version;           /**< 格式版本 */
    uint64_t header_size;              /**< 头部大小 */
    ModelVersionID version_id;         /**< 版本ID */
    uint64_t timestamp;                /**< 创建时间戳 */
    float loss;                        /**< 损失值 */
    float accuracy;                    /**< 准确率 */
    size_t epoch;                      /**< 训练轮数 */
    size_t parameter_count;            /**< 参数数量 */
    size_t config_size;                /**< 训练配置大小 */
    char tag[64];                      /**< 版本标签 */
} VersionSnapshotHeader;

/**
 * @brief 版本信息文件内容
 */
typedef struct {
    ModelVersionID id;
    char tag[64];
    char description[256];
    float loss;
    float accuracy;
    size_t epoch;
    uint64_t timestamp;
    size_t parameter_count;
    int is_locked;
} VersionInfoFile;

/* ====================================================================
 * 内部辅助函数
 * ==================================================================== */

/**
 * @brief 确保目录存在
 */
static int ensure_directory_exists(const char* dir_path) {
    struct stat st = {0};
    if (stat(dir_path, &st) == 0) {
        return 0;
    }
    return mkdir_p(dir_path);
}

/**
 * @brief 生成快照文件路径
 */
static void build_snapshot_path(const char* versions_dir, ModelVersionID version_id,
                                 char* out_path, size_t out_size) {
    snprintf(out_path, out_size, "%s/%llu%s",
             versions_dir, (unsigned long long)version_id, SNAPSHOT_FILE_SUFFIX);
}

/**
 * @brief 生成网络文件路径
 */
static void build_network_path(const char* versions_dir, ModelVersionID version_id,
                                char* out_path, size_t out_size) {
    snprintf(out_path, out_size, "%s/%llu%s",
             versions_dir, (unsigned long long)version_id, NETWORK_FILE_SUFFIX);
}

/**
 * @brief 生成版本信息文件路径
 */
static void build_info_path(const char* versions_dir, ModelVersionID version_id,
                             char* out_path, size_t out_size) {
    snprintf(out_path, out_size, "%s/%llu%s",
             versions_dir, (unsigned long long)version_id, INFO_FILE_SUFFIX);
}

/**
 * @brief 扩展版本数组容量
 */
static int expand_versions_capacity(ModelVersionManager* mgr) {
    size_t new_capacity = mgr->capacity == 0 ?
                          INITIAL_VERSIONS_CAPACITY :
                          mgr->capacity * CAPACITY_GROWTH_FACTOR;

    ModelVersionEntry* new_versions = (ModelVersionEntry*)
        realloc(mgr->versions, new_capacity * sizeof(ModelVersionEntry));
    if (!new_versions) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "扩展版本数组失败");
        return -1;
    }

    mgr->versions = new_versions;
    mgr->capacity = new_capacity;
    return 0;
}

/**
 * @brief 查找版本索引
 */
static int find_version_index(const ModelVersionManager* mgr, ModelVersionID version_id) {
    for (size_t i = 0; i < mgr->num_versions; i++) {
        if (mgr->versions[i].id == version_id) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief 保存版本信息到文件
 */
static int save_version_info(const ModelVersionManager* mgr, const ModelVersionEntry* entry) {
    char info_path[512];
    build_info_path(mgr->versions_dir, entry->id, info_path, sizeof(info_path));

    FILE* file = fopen(info_path, "wb");
    if (!file) {
        return -1;
    }

    VersionInfoFile info;
    memset(&info, 0, sizeof(VersionInfoFile));
    info.id = entry->id;
    memcpy(info.tag, entry->tag, sizeof(info.tag));
    memcpy(info.description, entry->description, sizeof(info.description));
    info.loss = entry->loss;
    info.accuracy = entry->accuracy;
    info.epoch = entry->epoch;
    info.timestamp = entry->timestamp;
    info.parameter_count = entry->parameter_count;
    info.is_locked = entry->is_locked;

    int ret = 0;
    if (fwrite(&info, sizeof(VersionInfoFile), 1, file) != 1) {
        ret = -1;
    }

    fclose(file);
    return ret;
}

/**
 * @brief 加载版本信息从文件
 */
static int load_version_info(const char* info_path, VersionInfoFile* info) {
    FILE* file = fopen(info_path, "rb");
    if (!file) {
        return -1;
    }

    int ret = 0;
    if (fread(info, sizeof(VersionInfoFile), 1, file) != 1) {
        ret = -1;
    }

    fclose(file);
    return ret;
}

/**
 * @brief 从训练器获取所有网络参数到一个连续缓冲区
 * 
 * 内部函数：提取网络的所有可训练参数到连续缓冲区。
 * 返回通过 malloc 分配的缓冲区，调用者负责释放。
 */
static float* extract_all_parameters(Trainer* trainer, size_t* param_count) {
    if (!trainer || !trainer_get_network(trainer) || !param_count) {
        return NULL;
    }

    LNN* network = trainer_get_network(trainer);
    size_t count = lnn_get_parameter_count(network);
    if (count == 0) {
        *param_count = 0;
        return NULL;
    }

    float* buffer = (float*)safe_malloc(count * sizeof(float));
    if (!buffer) {
        *param_count = 0;
        return NULL;
    }

    float* params = lnn_get_parameters(network);
    if (!params) {
        safe_free((void**)&buffer);
        *param_count = 0;
        return NULL;
    }

    memcpy(buffer, params, count * sizeof(float));
    *param_count = count;
    return buffer;
}

/**
 * @brief 将参数缓冲区写入网络
 * 
 * 内部函数：将连续参数缓冲区写回网络的可训练参数。
 */
static int restore_all_parameters(Trainer* trainer, const float* buffer, size_t param_count) {
    if (!trainer || !trainer_get_network(trainer) || !buffer || param_count == 0) {
        return -1;
    }

    LNN* network = trainer_get_network(trainer);
    size_t current_count = lnn_get_parameter_count(network);
    if (current_count != param_count) {
        selflnn_set_last_error(SELFLNN_ERROR_NETWORK_CONFIG, __func__, __FILE__, __LINE__,
                              "参数数量不匹配：当前=%zu，快照=%zu",
                              current_count, param_count);
        return -1;
    }

    float* params = lnn_get_parameters(network);
    if (!params) {
        return -1;
    }

    memcpy(params, buffer, param_count * sizeof(float));
    return 0;
}

/* ====================================================================
 * 公开API实现
 * ==================================================================== */

ModelVersionManager* model_version_manager_create(const char* versions_dir, size_t max_versions) {
    ModelVersionManager* mgr = (ModelVersionManager*)
        safe_malloc(sizeof(ModelVersionManager));
    if (!mgr) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建版本管理器：内存分配失败");
        return NULL;
    }

    memset(mgr, 0, sizeof(ModelVersionManager));

    const char* dir = versions_dir ? versions_dir : DEFAULT_VERSIONS_DIR;
    size_t dir_len = strlen(dir);
    size_t copy_len = dir_len < sizeof(mgr->versions_dir) - 1 ?
                      dir_len : sizeof(mgr->versions_dir) - 1;
    memcpy(mgr->versions_dir, dir, copy_len);
    mgr->versions_dir[copy_len] = '\0';

    mgr->max_versions = max_versions;
    mgr->next_id = 1;
    mgr->active_version_id = 0;
    mgr->auto_prune_enabled = 0;
    mgr->auto_snapshot_enabled = 0;
    mgr->auto_snapshot_interval = 10;
    mgr->prune_keep_best_ratio = 0.3f;

    mgr->versions = NULL;
    mgr->num_versions = 0;
    mgr->capacity = 0;

    if (ensure_directory_exists(mgr->versions_dir) != 0) {
        char dir_buf[512];
        strncpy(dir_buf, mgr->versions_dir, sizeof(dir_buf)-1);
        dir_buf[sizeof(dir_buf)-1] = '\0';
        safe_free((void**)&mgr);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "创建版本存储目录失败：%s", dir_buf);
        return NULL;
    }

    return mgr;
}

void model_version_manager_free(ModelVersionManager* mgr) {
    if (!mgr) {
        return;
    }

    if (mgr->versions) {
        safe_free((void**)&mgr->versions);
    }
    safe_free((void**)&mgr);
}

ModelVersionID model_version_snapshot(ModelVersionManager* mgr, Trainer* trainer,
                                       const char* tag, const char* description) {
    if (!mgr || !trainer) {
        return 0;
    }

    if (mgr->max_versions > 0 && mgr->num_versions >= mgr->max_versions) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "版本数量已达上限（%zu），请先清理旧版本", mgr->max_versions);
        return 0;
    }

    ModelVersionID new_id = mgr->next_id++;

    char snapshot_path[512];
    char network_path[512];
    build_snapshot_path(mgr->versions_dir, new_id, snapshot_path, sizeof(snapshot_path));
    build_network_path(mgr->versions_dir, new_id, network_path, sizeof(network_path));

    /* 1. 提取所有网络参数 */
    size_t param_count = 0;
    float* param_buffer = extract_all_parameters(trainer, &param_count);
    if (!param_buffer && param_count > 0) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建版本快照：提取参数失败");
        return 0;
    }

    /* 2. 保存网络权重到文件 */
    if (param_count > 0 && param_buffer) {
        FILE* net_file = fopen(network_path, "wb");
        if (!net_file) {
            safe_free((void**)&param_buffer);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "创建网络权重文件失败：%s", network_path);
            return 0;
        }

        size_t written = fwrite(param_buffer, sizeof(float), param_count, net_file);
        fclose(net_file);

        if (written != param_count) {
            safe_free((void**)&param_buffer);
            remove(network_path);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "写入网络权重失败：期望%zu，实际%zu",
                                  param_count, written);
            return 0;
        }
    }

    /* 3. 保存快照文件（包含训练状态元数据） */
    FILE* snap_file = fopen(snapshot_path, "wb");
    if (!snap_file) {
        if (param_buffer) safe_free((void**)&param_buffer);
        if (param_count > 0) remove(network_path);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "创建快照文件失败：%s", snapshot_path);
        return 0;
    }

    VersionSnapshotHeader header;
    memset(&header, 0, sizeof(VersionSnapshotHeader));
    memcpy(header.magic, VERSION_MAGIC, strlen(VERSION_MAGIC));
    header.format_version = VERSION_FILE_FORMAT_VERSION;
    header.header_size = sizeof(VersionSnapshotHeader);
    header.version_id = new_id;
    header.timestamp = perf_timestamp_ns() / 1000000;
    {
        TrainingState* st = trainer_get_state(trainer);
        if (st) {
            header.loss = st->best_loss;
            header.accuracy = st->best_accuracy;
            header.epoch = st->current_epoch;
        } else {
            header.loss = 0.0f;
            header.accuracy = 0.0f;
            header.epoch = 0;
        }
    }
    header.parameter_count = param_count;
    header.config_size = sizeof(TrainingConfig);

    if (tag) {
        size_t tag_len = strlen(tag);
        size_t copy_len = tag_len < sizeof(header.tag) - 1 ?
                          tag_len : sizeof(header.tag) - 1;
        memcpy(header.tag, tag, copy_len);
        header.tag[copy_len] = '\0';
    }

    int write_ok = 1;
    if (fwrite(&header, sizeof(VersionSnapshotHeader), 1, snap_file) != 1) {
        write_ok = 0;
    }

    TrainingConfig* cfg = trainer_get_config(trainer);
    if (!cfg || fwrite(cfg, sizeof(TrainingConfig), 1, snap_file) != 1) {
        write_ok = 0;
    }

    fclose(snap_file);

    if (!write_ok) {
        if (param_buffer) safe_free((void**)&param_buffer);
        remove(snapshot_path);
        if (param_count > 0) remove(network_path);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "写入快照元数据失败");
        return 0;
    }

    /* 4. 添加版本条目到管理器 */
    if (mgr->num_versions >= mgr->capacity) {
        if (expand_versions_capacity(mgr) != 0) {
            if (param_buffer) safe_free((void**)&param_buffer);
            remove(snapshot_path);
            if (param_count > 0) remove(network_path);
            return 0;
        }
    }

    ModelVersionEntry* entry = &mgr->versions[mgr->num_versions];
    memset(entry, 0, sizeof(ModelVersionEntry));
    entry->id = new_id;
    memcpy(entry->snapshot_path, snapshot_path, strlen(snapshot_path) + 1);
    if (param_count > 0) {
        memcpy(entry->network_path, network_path, strlen(network_path) + 1);
    }

    if (tag) {
        size_t tag_len = strlen(tag);
        size_t copy_len = tag_len < sizeof(entry->tag) - 1 ?
                          tag_len : sizeof(entry->tag) - 1;
        memcpy(entry->tag, tag, copy_len);
        entry->tag[copy_len] = '\0';
    }

    if (description) {
        size_t desc_len = strlen(description);
        size_t copy_len = desc_len < sizeof(entry->description) - 1 ?
                          desc_len : sizeof(entry->description) - 1;
        memcpy(entry->description, description, copy_len);
        entry->description[copy_len] = '\0';
    }

    entry->loss = header.loss;
    entry->accuracy = header.accuracy;
    entry->epoch = header.epoch;
    entry->timestamp = header.timestamp;
    entry->parameter_count = param_count;
    entry->is_active = 1;
    entry->is_locked = 0;

    mgr->active_version_id = new_id;
    mgr->num_versions++;

    /* 5. 保存版本信息文件 */
    save_version_info(mgr, entry);

    /* 6. 清理当前其他版本的活跃标志 */
    for (size_t i = 0; i < mgr->num_versions - 1; i++) {
        mgr->versions[i].is_active = 0;
    }

    if (param_buffer) {
        safe_free((void**)&param_buffer);
    }

    /* 7. 自动清理 */
    if (mgr->auto_prune_enabled && mgr->max_versions > 0 &&
        mgr->num_versions > mgr->max_versions) {
        size_t keep_count = (size_t)(mgr->max_versions * (1.0f - mgr->prune_keep_best_ratio));
        if (keep_count < 1) keep_count = 1;
        model_version_prune(mgr, keep_count, 1);
    }

    return new_id;
}

int model_version_rollback(ModelVersionManager* mgr, ModelVersionID version_id,
                            Trainer* trainer) {
    if (!mgr || !trainer) {
        return -1;
    }

    int idx = find_version_index(mgr, version_id);
    if (idx < 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MODEL_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "回滚失败：版本%llu不存在",
                              (unsigned long long)version_id);
        return -1;
    }

    ModelVersionEntry* entry = &mgr->versions[idx];

    /* 1. 检查是否有网络权重文件 */
    if (strlen(entry->network_path) == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MODEL_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "回滚失败：版本%llu没有网络权重数据",
                              (unsigned long long)version_id);
        return -1;
    }

    /* 2. 检查网络文件是否存在 */
    FILE* net_file = fopen(entry->network_path, "rb");
    if (!net_file) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "回滚失败：无法打开网络权重文件：%s",
                              entry->network_path);
        return -1;
    }

    /* 3. 读取网络权重 */
    size_t param_count = entry->parameter_count;
    if (param_count == 0) {
        fclose(net_file);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "回滚失败：版本%llu的参数数量为0",
                              (unsigned long long)version_id);
        return -1;
    }

    LNN* rollback_net = trainer_get_network(trainer);
    if (!rollback_net) {
        fclose(net_file);
        selflnn_set_last_error(SELFLNN_ERROR_NETWORK_CONFIG, __func__, __FILE__, __LINE__,
                              "回滚失败：无法获取网络句柄");
        return -1;
    }
    size_t current_param_count = lnn_get_parameter_count(rollback_net);
    if (current_param_count != param_count) {
        fclose(net_file);
        selflnn_set_last_error(SELFLNN_ERROR_NETWORK_CONFIG, __func__, __FILE__, __LINE__,
                              "回滚失败：参数数量不匹配（当前=%zu，快照=%zu）",
                              current_param_count, param_count);
        return -1;
    }

    float* param_buffer = (float*)safe_malloc(param_count * sizeof(float));
    if (!param_buffer) {
        fclose(net_file);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "回滚失败：分配参数缓冲区失败");
        return -1;
    }

    size_t read_count = fread(param_buffer, sizeof(float), param_count, net_file);
    fclose(net_file);

    if (read_count != param_count) {
        safe_free((void**)&param_buffer);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "回滚失败：读取网络权重不完整（期望%zu，实际%zu）",
                              param_count, read_count);
        return -1;
    }

    /* 4. 恢复网络权重 */
    if (restore_all_parameters(trainer, param_buffer, param_count) != 0) {
        safe_free((void**)&param_buffer);
        return -1;
    }

    safe_free((void**)&param_buffer);

    /* 5. 恢复训练状态 */
    {
        TrainingState rollback_state;
        TrainingState* cur_st = trainer_get_state(trainer);
        if (cur_st) {
            rollback_state = *cur_st;
        } else {
            memset(&rollback_state, 0, sizeof(TrainingState));
        }
        rollback_state.best_loss = entry->loss;
        rollback_state.best_accuracy = entry->accuracy;
        rollback_state.current_epoch = entry->epoch;
        trainer_set_state(trainer, &rollback_state);
    }

    /* 6. 更新活跃状态 */
    for (size_t i = 0; i < mgr->num_versions; i++) {
        mgr->versions[i].is_active = 0;
    }
    entry->is_active = 1;
    mgr->active_version_id = version_id;

    return 0;
}

int model_version_diff(ModelVersionManager* mgr, ModelVersionID v1_id, ModelVersionID v2_id,
                        float* mean_diff, float* max_diff, float* std_diff,
                        size_t* num_different_params) {
    if (!mgr || !mean_diff || !max_diff || !std_diff || !num_different_params) {
        return -1;
    }

    int idx1 = find_version_index(mgr, v1_id);
    int idx2 = find_version_index(mgr, v2_id);

    if (idx1 < 0 || idx2 < 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MODEL_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "差异比较失败：版本不存在");
        return -1;
    }

    ModelVersionEntry* e1 = &mgr->versions[idx1];
    ModelVersionEntry* e2 = &mgr->versions[idx2];

    if (e1->parameter_count != e2->parameter_count) {
        selflnn_set_last_error(SELFLNN_ERROR_NETWORK_CONFIG, __func__, __FILE__, __LINE__,
                              "差异比较失败：参数数量不同（%zu vs %zu）",
                              e1->parameter_count, e2->parameter_count);
        return -1;
    }

    if (e1->parameter_count == 0) {
        *mean_diff = 0.0f;
        *max_diff = 0.0f;
        *std_diff = 0.0f;
        *num_different_params = 0;
        return 0;
    }

    /* 加载两个版本的参数 */
    FILE* f1 = fopen(e1->network_path, "rb");
    FILE* f2 = fopen(e2->network_path, "rb");

    if (!f1 || !f2) {
        if (f1) fclose(f1);
        if (f2) fclose(f2);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "差异比较失败：无法打开权重文件");
        return -1;
    }

    size_t count = e1->parameter_count;
    float* p1 = (float*)safe_malloc(count * sizeof(float));
    float* p2 = (float*)safe_malloc(count * sizeof(float));

    if (!p1 || !p2) {
        safe_free((void**)&p1);
        safe_free((void**)&p2);
        fclose(f1);
        fclose(f2);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "差异比较失败：内存不足");
        return -1;
    }

    size_t r1 = fread(p1, sizeof(float), count, f1);
    size_t r2 = fread(p2, sizeof(float), count, f2);
    fclose(f1);
    fclose(f2);

    if (r1 != count || r2 != count) {
        safe_free((void**)&p1);
        safe_free((void**)&p2);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "差异比较失败：读取权重不完整");
        return -1;
    }

    /* 计算差异统计 */
    double sum_diff = 0.0;
    double sum_sq_diff = 0.0;
    float max_diff_val = 0.0f;
    size_t diff_count = 0;

    for (size_t i = 0; i < count; i++) {
        float diff = (float)fabs((double)p1[i] - (double)p2[i]);
        sum_diff += diff;
        sum_sq_diff += (double)diff * (double)diff;

        if (diff > max_diff_val) {
            max_diff_val = diff;
        }

        if (diff > DIFF_THRESHOLD) {
            diff_count++;
        }
    }

    double avg = sum_diff / (double)count;
    double variance = (sum_sq_diff / (double)count) - (avg * avg);
    if (variance < 0.0) variance = 0.0;
    double std = sqrt(variance);

    *mean_diff = (float)avg;
    *max_diff = max_diff_val;
    *std_diff = (float)std;
    *num_different_params = diff_count;

    safe_free((void**)&p1);
    safe_free((void**)&p2);

    return 0;
}

int model_version_list(const ModelVersionManager* mgr, ModelVersionEntry** entries,
                        size_t* count) {
    if (!mgr || !entries || !count) {
        return -1;
    }

    *entries = mgr->versions;
    *count = mgr->num_versions;
    return 0;
}

int model_version_delete(ModelVersionManager* mgr, ModelVersionID version_id) {
    if (!mgr) {
        return -1;
    }

    int idx = find_version_index(mgr, version_id);
    if (idx < 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MODEL_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "删除失败：版本%llu不存在",
                              (unsigned long long)version_id);
        return -1;
    }

    if (mgr->versions[idx].is_locked) {
        selflnn_set_last_error(SELFLNN_ERROR_GENERIC, __func__, __FILE__, __LINE__,
                              "删除失败：版本%llu已锁定",
                              (unsigned long long)version_id);
        return -1;
    }

    /* 删除快照文件 */
    if (strlen(mgr->versions[idx].snapshot_path) > 0) {
        remove(mgr->versions[idx].snapshot_path);
    }

    /* 删除网络权重文件 */
    if (strlen(mgr->versions[idx].network_path) > 0) {
        remove(mgr->versions[idx].network_path);
    }

    /* 删除版本信息文件 */
    {
        char info_path[512];
        build_info_path(mgr->versions_dir, version_id, info_path, sizeof(info_path));
        remove(info_path);
    }

    /* 从数组中移除 */
    for (size_t i = (size_t)idx; i < mgr->num_versions - 1; i++) {
        mgr->versions[i] = mgr->versions[i + 1];
    }
    mgr->num_versions--;

    return 0;
}

int model_version_tag(ModelVersionManager* mgr, ModelVersionID version_id, const char* tag) {
    if (!mgr || !tag) {
        return -1;
    }

    int idx = find_version_index(mgr, version_id);
    if (idx < 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MODEL_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "设置标签失败：版本%llu不存在",
                              (unsigned long long)version_id);
        return -1;
    }

    size_t tag_len = strlen(tag);
    size_t copy_len = tag_len < sizeof(mgr->versions[idx].tag) - 1 ?
                      tag_len : sizeof(mgr->versions[idx].tag) - 1;
    memcpy(mgr->versions[idx].tag, tag, copy_len);
    mgr->versions[idx].tag[copy_len] = '\0';

    save_version_info(mgr, &mgr->versions[idx]);

    return 0;
}

int model_version_lock(ModelVersionManager* mgr, ModelVersionID version_id, int lock) {
    if (!mgr) {
        return -1;
    }

    int idx = find_version_index(mgr, version_id);
    if (idx < 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MODEL_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "锁定/解锁失败：版本%llu不存在",
                              (unsigned long long)version_id);
        return -1;
    }

    mgr->versions[idx].is_locked = lock ? 1 : 0;
    save_version_info(mgr, &mgr->versions[idx]);

    return 0;
}

ModelVersionID model_version_get_best(const ModelVersionManager* mgr, int by_accuracy) {
    if (!mgr || mgr->num_versions == 0) {
        return 0;
    }

    ModelVersionID best_id = mgr->versions[0].id;
    float best_value = by_accuracy ?
                       mgr->versions[0].accuracy :
                       mgr->versions[0].loss;

    for (size_t i = 1; i < mgr->num_versions; i++) {
        float value = by_accuracy ?
                      mgr->versions[i].accuracy :
                      mgr->versions[i].loss;

        int is_better = by_accuracy ? (value > best_value) : (value < best_value);

        if (is_better) {
            best_value = value;
            best_id = mgr->versions[i].id;
        }
    }

    return best_id;
}

size_t model_version_prune(ModelVersionManager* mgr, size_t keep_count, int keep_best) {
    if (!mgr || mgr->num_versions <= keep_count) {
        return 0;
    }

    size_t to_delete = mgr->num_versions - keep_count;
    size_t deleted = 0;

    ModelVersionID best_id = 0;
    if (keep_best) {
        best_id = model_version_get_best(mgr, 1);
    }

    size_t i = 0;
    while (i < mgr->num_versions && deleted < to_delete) {
        ModelVersionEntry* entry = &mgr->versions[i];

        if (entry->is_locked) {
            i++;
            continue;
        }

        if (keep_best && entry->id == best_id) {
            i++;
            continue;
        }

        if (model_version_delete(mgr, entry->id) == 0) {
            deleted++;
        } else {
            i++;
        }
    }

    return deleted;
}

int model_version_export_json(const ModelVersionManager* mgr, char* buffer, size_t buffer_size) {
    if (!mgr || !buffer || buffer_size == 0) {
        return -1;
    }

    size_t pos = 0;
    int ret;

    ret = snprintf(buffer + pos, buffer_size - pos,
                   "{\n"
                   "  \"manager\": {\n"
                   "    \"num_versions\": %zu,\n"
                   "    \"max_versions\": %zu,\n"
                   "    \"active_version_id\": %llu,\n"
                   "    \"versions_dir\": \"%s\",\n"
                   "    \"auto_snapshot\": %s,\n"
                   "    \"auto_prune\": %s\n"
                   "  },\n"
                   "  \"versions\": [\n",
                   mgr->num_versions,
                   mgr->max_versions,
                   (unsigned long long)mgr->active_version_id,
                   mgr->versions_dir,
                   mgr->auto_snapshot_enabled ? "true" : "false",
                   mgr->auto_prune_enabled ? "true" : "false");
    if (ret < 0 || (size_t)ret >= buffer_size - pos) {
        return -1;
    }
    pos += (size_t)ret;

    for (size_t i = 0; i < mgr->num_versions; i++) {
        const ModelVersionEntry* e = &mgr->versions[i];

        ret = snprintf(buffer + pos, buffer_size - pos,
                       "    {\n"
                       "      \"id\": %llu,\n"
                       "      \"tag\": \"%s\",\n"
                       "      \"description\": \"%s\",\n"
                       "      \"loss\": %.6f,\n"
                       "      \"accuracy\": %.6f,\n"
                       "      \"epoch\": %zu,\n"
                       "      \"timestamp\": %llu,\n"
                       "      \"parameter_count\": %zu,\n"
                       "      \"is_active\": %s,\n"
                       "      \"is_locked\": %s\n"
                       "    }%s\n",
                       (unsigned long long)e->id,
                       e->tag,
                       e->description,
                       e->loss,
                       e->accuracy,
                       e->epoch,
                       (unsigned long long)e->timestamp,
                       e->parameter_count,
                       e->is_active ? "true" : "false",
                       e->is_locked ? "true" : "false",
                       (i < mgr->num_versions - 1) ? "," : "");
        if (ret < 0 || (size_t)ret >= buffer_size - pos) {
            return -1;
        }
        pos += (size_t)ret;
    }

    ret = snprintf(buffer + pos, buffer_size - pos,
                   "  ]\n"
                   "}\n");
    if (ret < 0 || (size_t)ret >= buffer_size - pos) {
        return -1;
    }

    return 0;
}

int model_version_set_auto_snapshot(ModelVersionManager* mgr, int enabled,
                                     size_t interval_epochs) {
    if (!mgr) {
        return -1;
    }

    mgr->auto_snapshot_enabled = enabled ? 1 : 0;
    mgr->auto_snapshot_interval = interval_epochs > 0 ? interval_epochs : 1;

    return 0;
}

int model_version_set_description(ModelVersionManager* mgr, ModelVersionID version_id,
                                   const char* description) {
    if (!mgr || !description) {
        return -1;
    }

    int idx = find_version_index(mgr, version_id);
    if (idx < 0) {
        selflnn_set_last_error(SELFLNN_ERROR_MODEL_NOT_FOUND, __func__, __FILE__, __LINE__,
                              "设置描述失败：版本%llu不存在",
                              (unsigned long long)version_id);
        return -1;
    }

    size_t desc_len = strlen(description);
    size_t copy_len = desc_len < sizeof(mgr->versions[idx].description) - 1 ?
                      desc_len : sizeof(mgr->versions[idx].description) - 1;
    memcpy(mgr->versions[idx].description, description, copy_len);
    mgr->versions[idx].description[copy_len] = '\0';

    save_version_info(mgr, &mgr->versions[idx]);

    return 0;
}

size_t model_version_count(const ModelVersionManager* mgr) {
    if (!mgr) {
        return 0;
    }
    return mgr->num_versions;
}

size_t model_version_clear_all(ModelVersionManager* mgr, int force) {
    if (!mgr) {
        return 0;
    }

    size_t deleted = 0;
    size_t i = 0;

    while (i < mgr->num_versions) {
        ModelVersionEntry* entry = &mgr->versions[i];

        if (!force && entry->is_locked) {
            i++;
            continue;
        }

        if (model_version_delete(mgr, entry->id) == 0) {
            deleted++;
        } else {
            i++;
        }
    }

    return deleted;
}
