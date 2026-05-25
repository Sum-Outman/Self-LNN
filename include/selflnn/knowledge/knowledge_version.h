/**
 * @file knowledge_version.h
 * @brief 知识版本管理系统接口
 */

#ifndef SELFLNN_KNOWLEDGE_VERSION_H
#define SELFLNN_KNOWLEDGE_VERSION_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

/* ZSFBUILD: SnapshotEntryRecord完整定义 (从knowledge_version.c移入，header函数声明需要完整类型) */
typedef struct {
    int entry_id;
    char subject[256];
    char predicate[256];
    char object[256];
    float confidence;
    int type;
    int source;
    long timestamp;
} SnapshotEntryRecord;

#ifdef __cplusplus
extern "C" {
#endif

#define KV_MAX_SNAPSHOTS 100
#define KV_MAX_DIFFS 50
#define KV_MAX_BRANCH_NAME 64
#define KV_MAX_MESSAGE 256
#define KV_MAX_ENTRIES_PER_SNAPSHOT 5000
#define KV_MAX_DIFF_DETAILS 200
#define KV_MAX_HISTORY 500

/* 差异条目类型 */
typedef enum {
    KV_DIFF_ADD = 0,
    KV_DIFF_REMOVE = 1,
    KV_DIFF_MODIFY = 2
} KVDiffType;

/* 详细差异条目 */
typedef struct {
    KVDiffType change_type;
    int entry_id;
    char subject[256];
    char predicate[256];
    char object_before[256];
    char object_after[256];
    float confidence_before;
    float confidence_after;
} KVDiffEntry;

/* 版本快照 */
typedef struct {
    int snapshot_id;
    char message[KV_MAX_MESSAGE];
    time_t created_at;
    size_t entry_count;
    size_t total_size;
    char branch[KV_MAX_BRANCH_NAME];
    int parent_id;
    int is_checkpoint;
} KnowledgeSnapshot;

/* 版本差异 */
typedef struct {
    int from_snapshot;
    int to_snapshot;
    size_t added_count;
    size_t removed_count;
    size_t modified_count;
    time_t diff_time;
    KVDiffEntry details[KV_MAX_DIFF_DETAILS];
    int detail_count;
} KnowledgeDiff;

/* 知识条目历史 */
typedef struct {
    int entry_id;
    char subject[256];
    char predicate[256];
    int snapshot_ids[KV_MAX_HISTORY];
    int history_count;
    int last_snapshot_id;
    time_t first_seen;
    time_t last_modified;
    int modification_count;
} KnowledgeEntryHistory;

/* 知识版本管理器句柄 */
typedef struct KnowledgeVersionManager KnowledgeVersionManager;

KnowledgeVersionManager* knowledge_version_create(const char* storage_dir);
void knowledge_version_free(KnowledgeVersionManager* kvm);

/* 快照管理 */
int kv_create_snapshot(KnowledgeVersionManager* kvm, const char* message);
int kv_list_snapshots(const KnowledgeVersionManager* kvm, KnowledgeSnapshot* out, int max_count);
int kv_get_snapshot(const KnowledgeVersionManager* kvm, int snapshot_id, KnowledgeSnapshot* out);
int kv_restore_snapshot(KnowledgeVersionManager* kvm, int snapshot_id, void* knowledge_base);

/* 详细差异分析 */
int kv_diff_snapshots(const KnowledgeVersionManager* kvm, int from_id, int to_id, KnowledgeDiff* diff);

/* 分支管理 */
int kv_create_branch(KnowledgeVersionManager* kvm, const char* branch_name, int from_snapshot);
int kv_switch_branch(KnowledgeVersionManager* kvm, const char* branch_name);
int kv_merge_branch(KnowledgeVersionManager* kvm, const char* source_branch, const char* target_branch);
int kv_list_branches(const KnowledgeVersionManager* kvm, char* branches, int max_count);

/* 知识溯源 */
int kv_trace_entry(const KnowledgeVersionManager* kvm, int entry_id, char* history, size_t max_len);

/* 自动快照 */
int kv_set_auto_snapshot(KnowledgeVersionManager* kvm, int interval_minutes, int max_snapshots);

/* ===== P4.4 新增增强功能 ===== */

/* 获取快照中保存的条目总数 */
int kv_get_snapshot_entry_count(const KnowledgeVersionManager* kvm, int snapshot_id);

/* 跨分支快照比较 */
int kv_compare_branch_snapshots(const KnowledgeVersionManager* kvm,
                                const char* branch_a, int snapshot_a_id,
                                const char* branch_b, int snapshot_b_id,
                                KnowledgeDiff* diff);

/* 获取知识条目的完整版本历史 */
int kv_get_entry_history(const KnowledgeVersionManager* kvm, int entry_id,
                         KnowledgeEntryHistory* history);

/* 快照差异导出为文本报告 */
int kv_diff_export_report(const KnowledgeVersionManager* kvm, int from_id, int to_id,
                          char* report, size_t max_len);

/* 清理过期快照 */
int kv_cleanup_old_snapshots(KnowledgeVersionManager* kvm, int keep_count);

/* ============================================================================
 * ZSFWS-031: 语义相似度合并策略
 *
 * 对两个快照中的实体进行语义级合并，根据字符三元组Jaccard相似度决定策略：
 *   - 相似度 >= 0.80: 自动合并（取置信度高者）
 *   - 相似度 0.50 ~ 0.80: 标记为待审核（写入日志）
 *   - 相似度 < 0.50: 视为不同实体，均保留
 *
 * 与 kv_diff_snapshots 的关系：
 *   kv_diff_snapshots: 精确匹配版本追溯（(S,P)键完全相等）
 *   kv_semantic_merge_entities: 语义模糊匹配合并（近似字符串合并）
 *
 * @param entries_a      第一个条目数组
 * @param count_a        第一个数组大小
 * @param entries_b      第二个条目数组
 * @param count_b        第二个数组大小
 * @param merged_entries 合并结果输出（调用者预分配，容量 >= count_a + count_b）
 * @param max_merged     合并结果最大容量
 * @param merged_count   输出：实际合并条目数
 * @param log_buffer     冲突日志输出缓冲区（可NULL）
 * @param log_max        日志缓冲区大小
 * @return 0成功, -1失败
 * ============================================================================ */
int kv_semantic_merge_entities(
    const SnapshotEntryRecord* entries_a, int count_a,
    const SnapshotEntryRecord* entries_b, int count_b,
    SnapshotEntryRecord* merged_entries, int max_merged,
    int* merged_count,
    char* log_buffer, size_t log_max);

#ifdef __cplusplus
}
#endif
#endif
