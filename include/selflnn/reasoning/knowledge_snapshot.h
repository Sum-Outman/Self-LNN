/**
 * @file knowledge_snapshot.h
 * @brief 知识版本控制系统接口（快照与差异管理）
 *
 * 为知识库提供完整的版本控制功能，支持：
 * - 版本快照与回滚
 * - 分支管理（创建、切换、合并）
 * - 差异对比（Diff）
 * - 冲突检测与合并
 * - 版本历史追溯
 * - 标签管理
 */

#ifndef SELFLNN_KNOWLEDGE_SNAPSHOT_H
#define SELFLNN_KNOWLEDGE_SNAPSHOT_H

#include <stddef.h>
#include "selflnn/knowledge/knowledge.h"

#ifdef __cplusplus
extern "C" {
#endif

/** 最大提交消息长度 */
#define KV_MAX_MESSAGE_LENGTH 512
/** 最大分支名长度 */
#define KV_MAX_BRANCH_NAME 128
/** 最大标签名长度 */
#define KV_MAX_TAG_NAME 64
/** 最大分支数 */
#define KV_MAX_BRANCHES 32

/**
 * @brief 版本提交信息
 */
typedef struct {
    int version_id;                          /**< 版本ID(递增) */
    char message[KV_MAX_MESSAGE_LENGTH];     /**< 提交消息 */
    char author[64];                         /**< 提交者 */
    long timestamp;                          /**< 提交时间戳 */
    int parent_id;                           /**< 父版本ID(-1表示根版本) */
    int merge_parent_id;                     /**< 合并父版本ID(-1表示非合并提交) */
    char branch_name[KV_MAX_BRANCH_NAME];    /**< 分支名 */
    char tag[KV_MAX_TAG_NAME];               /**< 标签名(空表示无标签) */

    size_t entry_count;                      /**< 快照中的条目数 */
    size_t data_size;                        /**< 快照数据大小(字节) */
    void* snapshot_data;                     /**< 快照数据（F-005修复：存储实际快照内容，不丢弃） */
} KnowledgeVersion;

/**
 * @brief 分支信息
 */
typedef struct {
    char name[KV_MAX_BRANCH_NAME];           /**< 分支名 */
    int head_version_id;                     /**< 分支头版本ID */
    int base_version_id;                     /**< 分支基版本ID(从哪个版本分出) */
    long created_at;                         /**< 创建时间 */
    int is_active;                           /**< 是否为当前活跃分支 */
} KnowledgeBranch;

/**
 * @brief 变更类型
 */
typedef enum {
    KV_CHANGE_ADD = 0,       /**< 添加条目 */
    KV_CHANGE_MODIFY = 1,    /**< 修改条目 */
    KV_CHANGE_DELETE = 2,    /**< 删除条目 */
    KV_CHANGE_UNCHANGED = 3  /**< 未变更 */
} KnowledgeChangeType;

/**
 * @brief 单条目变更
 */
typedef struct {
    KnowledgeChangeType change_type;  /**< 变更类型 */
    int entry_id;                     /**< 条目ID */
    KnowledgeEntry old_entry;         /**< 旧条目(ADD时为NULL) */
    KnowledgeEntry new_entry;         /**< 新条目(DELETE时为NULL) */
    char field_changed[128];          /**< 变更的字段描述(MODIFY时) */
} KnowledgeChangeEntry;

/**
 * @brief 版本差异
 */
typedef struct {
    KnowledgeChangeEntry* changes;    /**< 变更列表 */
    size_t num_changes;               /**< 变更数量 */
    size_t capacity;                  /**< 变更容量 */
    int added_count;                  /**< 添加数 */
    int modified_count;               /**< 修改数 */
    int deleted_count;                /**< 删除数 */
} KnowledgeVersionDiff;

/**
 * @brief 合并冲突
 */
typedef struct {
    int entry_id;                      /**< 冲突条目ID */
    KnowledgeEntry base_entry;         /**< 基准版本条目 */
    KnowledgeEntry our_entry;          /**< 当前分支条目 */
    KnowledgeEntry their_entry;        /**< 合并目标分支条目 */
    char description[256];             /**< 冲突描述 */
} KnowledgeMergeConflict;

/**
 * @brief 合并结果
 */
typedef struct {
    int success;                       /**< 是否成功(1=成功, 0=有冲突) */
    int new_version_id;                /**< 合并产生的新版本ID */
    KnowledgeMergeConflict* conflicts; /**< 冲突列表(有冲突时) */
    size_t num_conflicts;              /**< 冲突数量 */
    size_t conflict_capacity;          /**< 冲突容量 */
    char summary[512];                 /**< 合并摘要 */
} KnowledgeMergeResult;

/**
 * @brief 知识版本控制器
 */
typedef struct {
    KnowledgeBase* kb;                  /**< 关联的知识库 */
    KnowledgeVersion* versions;         /**< 版本数组 */
    size_t num_versions;                /**< 版本数量 */
    size_t version_capacity;            /**< 版本容量 */
    int next_version_id;               /**< 下一个版本ID */
    int current_version_id;            /**< 当前版本ID */

    KnowledgeBranch branches[KV_MAX_BRANCHES]; /**< 分支数组 */
    size_t num_branches;               /**< 分支数量 */
    char current_branch[KV_MAX_BRANCH_NAME];   /**< 当前分支名 */

    int auto_snapshot;                 /**< 是否自动创建快照 */
} KnowledgeVersionController;

/**
 * @brief 创建知识版本控制器
 *
 * @param kb 关联的知识库指针
 * @return KnowledgeVersionController* 成功返回指针，失败返回NULL
 */
KnowledgeVersionController* knowledge_version_create(KnowledgeBase* kb);

/**
 * @brief 销毁知识版本控制器
 *
 * @param kvc 控制器指针
 */
void knowledge_version_destroy(KnowledgeVersionController* kvc);

/**
 * @brief 创建初始版本(从知识库当前状态)
 *
 * @param kvc 控制器指针
 * @param message 提交消息
 * @param author 提交者
 * @return int 成功返回版本ID，失败返回-1
 */
int knowledge_version_commit(KnowledgeVersionController* kvc,
                             const char* message, const char* author);

/**
 * @brief 回滚到指定版本
 *
 * @param kvc 控制器指针
 * @param version_id 目标版本ID
 * @return int 成功返回0，失败返回-1
 */
int knowledge_version_rollback(KnowledgeVersionController* kvc, int version_id);

/**
 * @brief 创建分支
 *
 * @param kvc 控制器指针
 * @param branch_name 分支名
 * @param from_version_id 从哪个版本创建(-1表示从当前版本)
 * @return int 成功返回0，失败返回-1
 */
int knowledge_version_create_branch(KnowledgeVersionController* kvc,
                                    const char* branch_name,
                                    int from_version_id);

/**
 * @brief 切换到指定分支
 *
 * @param kvc 控制器指针
 * @param branch_name 分支名
 * @return int 成功返回0，失败返回-1
 */
int knowledge_version_switch_branch(KnowledgeVersionController* kvc,
                                    const char* branch_name);

/**
 * @brief 计算两个版本的差异
 *
 * @param kvc 控制器指针
 * @param version_id_a 版本A
 * @param version_id_b 版本B
 * @return KnowledgeVersionDiff* 差异结果，失败返回NULL(需调用knowledge_version_diff_free释放)
 */
KnowledgeVersionDiff* knowledge_version_diff(KnowledgeVersionController* kvc,
                                             int version_id_a, int version_id_b);

/**
 * @brief 释放差异结果
 *
 * @param diff 差异指针
 */
void knowledge_version_diff_free(KnowledgeVersionDiff* diff);

/**
 * @brief 合并分支
 *
 * @param kvc 控制器指针
 * @param source_branch 源分支名
 * @param target_branch 目标分支名(NULL表示当前分支)
 * @return KnowledgeMergeResult* 合并结果(需调用knowledge_version_merge_result_free释放)
 */
KnowledgeMergeResult* knowledge_version_merge(KnowledgeVersionController* kvc,
                                              const char* source_branch,
                                              const char* target_branch);

/**
 * @brief 释放合并结果
 *
 * @param result 合并结果指针
 */
void knowledge_version_merge_result_free(KnowledgeMergeResult* result);

/**
 * @brief 添加标签到指定版本
 *
 * @param kvc 控制器指针
 * @param version_id 版本ID
 * @param tag 标签名
 * @return int 成功返回0，失败返回-1
 */
int knowledge_version_add_tag(KnowledgeVersionController* kvc,
                              int version_id, const char* tag);

/**
 * @brief 获取版本历史
 *
 * @param kvc 控制器指针
 * @param branch_name 分支名(NULL表示所有分支)
 * @param versions 输出版本数组
 * @param max_versions 最大版本数
 * @return int 返回实际版本数，失败返回-1
 */
int knowledge_version_get_history(KnowledgeVersionController* kvc,
                                  const char* branch_name,
                                  KnowledgeVersion* versions,
                                  size_t max_versions);

/**
 * @brief 获取当前版本信息
 *
 * @param kvc 控制器指针
 * @return const KnowledgeVersion* 版本信息指针，失败返回NULL
 */
const KnowledgeVersion* knowledge_version_get_current(const KnowledgeVersionController* kvc);

/**
 * @brief 获取分支列表
 *
 * @param kvc 控制器指针
 * @param branches 输出分支数组
 * @param max_branches 最大分支数
 * @return int 返回实际分支数，失败返回-1
 */
int knowledge_version_get_branches(const KnowledgeVersionController* kvc,
                                   KnowledgeBranch* branches,
                                   size_t max_branches);

/**
 * @brief 删除分支
 *
 * @param kvc 控制器指针
 * @param branch_name 分支名
 * @return int 成功返回0，失败返回-1
 */
int knowledge_version_delete_branch(KnowledgeVersionController* kvc,
                                    const char* branch_name);

/**
 * @brief 启用/禁用自动快照
 *
 * @param kvc 控制器指针
 * @param enable 1=启用，0=禁用
 */
void knowledge_version_set_auto_snapshot(KnowledgeVersionController* kvc, int enable);

/**
 * @brief 保存版本控制状态到文件
 *
 * @param kvc 控制器指针
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int knowledge_version_save(const KnowledgeVersionController* kvc, const char* filepath);

/**
 * @brief 从文件加载版本控制状态
 *
 * @param filepath 文件路径
 * @param kb 关联的知识库
 * @return KnowledgeVersionController* 成功返回指针，失败返回NULL
 */
KnowledgeVersionController* knowledge_version_load(const char* filepath,
                                                    KnowledgeBase* kb);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_KNOWLEDGE_SNAPSHOT_H */
