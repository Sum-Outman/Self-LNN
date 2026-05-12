/**
 * @file model_version.h
 * @brief 模型版本管理模块
 *
 * 提供模型版本快照、回滚、差异比较、自动清理等功能，
 * 支持训练过程中的自动版本管理和手动标记版本。
 */

#ifndef SELFLNN_MODEL_VERSION_H
#define SELFLNN_MODEL_VERSION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 前向声明 - Trainer 在 training.h 中完整定义 */
typedef struct Trainer Trainer;

/**
 * @brief 模型版本ID类型
 */
typedef uint64_t ModelVersionID;

/**
 * @brief 模型版本条目
 */
typedef struct {
    ModelVersionID id;                    /**< 版本唯一ID */
    char tag[64];                         /**< 版本标签（如"最佳模型"、"实验V2"） */
    char description[256];                /**< 版本描述 */
    char snapshot_path[512];              /**< 快照文件路径 */
    char network_path[512];               /**< 网络权重文件路径（.network.bin） */
    float loss;                           /**< 保存时的损失值 */
    float accuracy;                       /**< 保存时的准确率 */
    size_t epoch;                         /**< 保存时的训练轮数 */
    uint64_t timestamp;                   /**< 创建时间戳（毫秒） */
    size_t parameter_count;               /**< 参数数量 */
    int is_active;                        /**< 是否为当前活跃版本 */
    int is_locked;                        /**< 是否锁定（防止删除） */
} ModelVersionEntry;

/**
 * @brief 模型版本管理器
 */
typedef struct ModelVersionManager {
    ModelVersionEntry* versions;          /**< 版本数组 */
    size_t num_versions;                  /**< 当前版本数量 */
    size_t capacity;                      /**< 版本数组容量 */
    size_t max_versions;                  /**< 最大版本数（0=不限制） */
    ModelVersionID next_id;               /**< 下一个版本ID */
    ModelVersionID active_version_id;     /**< 当前活跃版本ID */
    char versions_dir[512];               /**< 版本存储目录 */
    int auto_prune_enabled;               /**< 是否启用自动清理 */
    int auto_snapshot_enabled;            /**< 是否启用自动快照 */
    size_t auto_snapshot_interval;        /**< 自动快照间隔（epoch数） */
    float prune_keep_best_ratio;          /**< 清理时保留最佳版本的比例（0.0~1.0） */
} ModelVersionManager;

/**
 * @brief 创建模型版本管理器
 *
 * @param versions_dir 版本存储目录（为NULL时使用默认目录"model_versions"）
 * @param max_versions 最大保留版本数（0=不限制）
 * @return ModelVersionManager* 成功返回管理器指针，失败返回NULL
 */
ModelVersionManager* model_version_manager_create(const char* versions_dir, size_t max_versions);

/**
 * @brief 销毁模型版本管理器
 *
 * 释放所有版本条目内存，但不删除已保存的快照文件。
 *
 * @param mgr 版本管理器指针
 */
void model_version_manager_free(ModelVersionManager* mgr);

/**
 * @brief 创建版本快照
 *
 * 保存当前训练器的完整状态（网络权重+训练状态）为一个新版本。
 *
 * @param mgr 版本管理器指针
 * @param trainer 训练器指针
 * @param tag 版本标签（可为NULL）
 * @param description 版本描述（可为NULL）
 * @return ModelVersionID 成功返回版本ID，失败返回0
 */
ModelVersionID model_version_snapshot(ModelVersionManager* mgr, Trainer* trainer,
                                       const char* tag, const char* description);

/**
 * @brief 回滚到指定版本
 *
 * 将训练器的网络权重和训练状态恢复到指定版本。
 *
 * @param mgr 版本管理器指针
 * @param version_id 目标版本ID
 * @param trainer 训练器指针
 * @return int 成功返回0，失败返回-1
 */
int model_version_rollback(ModelVersionManager* mgr, ModelVersionID version_id, Trainer* trainer);

/**
 * @brief 计算两个版本之间的参数差异统计
 *
 * 加载两个版本的网络权重，计算逐参数差异的均值、最大值和标准差。
 *
 * @param mgr 版本管理器指针
 * @param v1_id 版本1的ID
 * @param v2_id 版本2的ID
 * @param mean_diff 输出平均差异
 * @param max_diff 输出最大差异
 * @param std_diff 输出差异标准差
 * @param num_different_params 输出差异大于1e-6的参数数量
 * @return int 成功返回0，失败返回-1
 */
int model_version_diff(ModelVersionManager* mgr, ModelVersionID v1_id, ModelVersionID v2_id,
                        float* mean_diff, float* max_diff, float* std_diff,
                        size_t* num_different_params);

/**
 * @brief 列出所有版本
 *
 * 返回版本条目数组的指针，调用者通过 count 获取数量。
 * 返回的指针指向内部数据，调用者不应释放。
 *
 * @param mgr 版本管理器指针
 * @param entries 输出版本条目数组指针
 * @param count 输出版本数量
 * @return int 成功返回0，失败返回-1
 */
int model_version_list(const ModelVersionManager* mgr, ModelVersionEntry** entries, size_t* count);

/**
 * @brief 删除指定版本
 *
 * 删除版本条目及其关联的快照文件。
 * 已锁定的版本无法删除。
 *
 * @param mgr 版本管理器指针
 * @param version_id 要删除的版本ID
 * @return int 成功返回0，失败返回-1
 */
int model_version_delete(ModelVersionManager* mgr, ModelVersionID version_id);

/**
 * @brief 设置版本标签
 *
 * @param mgr 版本管理器指针
 * @param version_id 版本ID
 * @param tag 新标签（最长63个字符）
 * @return int 成功返回0，失败返回-1
 */
int model_version_tag(ModelVersionManager* mgr, ModelVersionID version_id, const char* tag);

/**
 * @brief 锁定/解锁版本
 *
 * 锁定后的版本无法被删除或覆盖。
 *
 * @param mgr 版本管理器指针
 * @param version_id 版本ID
 * @param lock 1=锁定，0=解锁
 * @return int 成功返回0，失败返回-1
 */
int model_version_lock(ModelVersionManager* mgr, ModelVersionID version_id, int lock);

/**
 * @brief 获取最佳版本
 *
 * 根据性能指标（损失或准确率）返回最佳版本ID。
 *
 * @param mgr 版本管理器指针
 * @param by_accuracy 1=按准确率排序，0=按损失排序
 * @return ModelVersionID 成功返回版本ID，没有版本时返回0
 */
ModelVersionID model_version_get_best(const ModelVersionManager* mgr, int by_accuracy);

/**
 * @brief 裁剪旧版本
 *
 * 保留指定数量的最新版本（和锁定版本），删除其余版本及其快照文件。
 *
 * @param mgr 版本管理器指针
 * @param keep_count 保留的版本数
 * @param keep_best 是否额外保留最佳版本（即使不在最新范围内）
 * @return size_t 实际删除的版本数
 */
size_t model_version_prune(ModelVersionManager* mgr, size_t keep_count, int keep_best);

/**
 * @brief 导出版本信息为JSON格式字符串
 *
 * @param mgr 版本管理器指针
 * @param buffer 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int model_version_export_json(const ModelVersionManager* mgr, char* buffer, size_t buffer_size);

/**
 * @brief 设置自动快照
 *
 * @param mgr 版本管理器指针
 * @param enabled 是否启用自动快照
 * @param interval_epochs 自动快照间隔（epoch数，0表示每轮都快照）
 * @return int 成功返回0，失败返回-1
 */
int model_version_set_auto_snapshot(ModelVersionManager* mgr, int enabled, size_t interval_epochs);

/**
 * @brief 设置版本描述
 *
 * @param mgr 版本管理器指针
 * @param version_id 版本ID
 * @param description 版本描述（最长255个字符）
 * @return int 成功返回0，失败返回-1
 */
int model_version_set_description(ModelVersionManager* mgr, ModelVersionID version_id,
                                  const char* description);

/**
 * @brief 获取版本数量
 *
 * @param mgr 版本管理器指针
 * @return size_t 版本数量
 */
size_t model_version_count(const ModelVersionManager* mgr);

/**
 * @brief 清除所有版本
 *
 * 删除所有版本条目及其快照文件（锁定版本除外）。
 *
 * @param mgr 版本管理器指针
 * @param force 强制删除（包括锁定版本，1=强制）
 * @return size_t 实际删除的版本数
 */
size_t model_version_clear_all(ModelVersionManager* mgr, int force);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MODEL_VERSION_H */
