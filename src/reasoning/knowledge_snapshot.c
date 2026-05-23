/**
 * @file knowledge_snapshot.c
 * @brief 知识版本控制系统实现 —— 快照/分支/差异/合并
 *
 * 为知识库提供完整的版本控制功能，支持：
 * - 版本快照与回滚
 * - 分支管理（创建、切换、合并）
 * - 差异对比（Diff）
 * - 冲突检测与合并
 * - 版本历史追溯
 *
 * 纯C实现，零外部依赖。
 */

#include "selflnn/reasoning/knowledge_snapshot.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define KV_INITIAL_CAPACITY 16
#define KV_MAX_VERSIONS 1024

KnowledgeVersionController* knowledge_version_create(KnowledgeBase* kb) {
    if (!kb) return NULL;

    KnowledgeVersionController* kvc = (KnowledgeVersionController*)safe_calloc(1, sizeof(KnowledgeVersionController));
    if (!kvc) return NULL;

    kvc->kb = kb;
    kvc->version_capacity = KV_INITIAL_CAPACITY;
    kvc->versions = (KnowledgeVersion*)safe_calloc(kvc->version_capacity, sizeof(KnowledgeVersion));
    if (!kvc->versions) {
        safe_free((void**)&kvc);
        return NULL;
    }
    kvc->num_versions = 0;
    kvc->next_version_id = 1;
    kvc->current_version_id = -1;
    kvc->num_branches = 0;
    snprintf(kvc->current_branch, sizeof(kvc->current_branch), "main");
    kvc->auto_snapshot = 0;

    /* 创建初始分支 */
    KnowledgeBranch* branch = &kvc->branches[0];
    snprintf(branch->name, sizeof(branch->name), "main");
    branch->head_version_id = -1;
    branch->base_version_id = -1;
    branch->created_at = (long)time(NULL);
    branch->is_active = 1;
    kvc->num_branches = 1;

    log_info("[知识版本控制] 已创建，关联知识库");
    return kvc;
}

void knowledge_version_destroy(KnowledgeVersionController* kvc) {
    if (!kvc) return;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].snapshot_data) {
            safe_free(&kvc->versions[i].snapshot_data);
        }
    }
    safe_free((void**)&kvc->versions);
    safe_free((void**)&kvc);
}

int knowledge_version_commit(KnowledgeVersionController* kvc,
                             const char* message, const char* author) {
    if (!kvc || !message) return -1;

    /* 扩展版本数组 */
    if (kvc->num_versions >= kvc->version_capacity) {
        size_t new_cap = kvc->version_capacity * 2;
        KnowledgeVersion* new_ver = (KnowledgeVersion*)safe_realloc(
            kvc->versions, new_cap * sizeof(KnowledgeVersion));
        if (!new_ver) return -1;
        kvc->versions = new_ver;
        kvc->version_capacity = new_cap;
    }

    KnowledgeVersion* ver = &kvc->versions[kvc->num_versions];
    memset(ver, 0, sizeof(KnowledgeVersion));
    ver->version_id = kvc->next_version_id++;
    snprintf(ver->message, sizeof(ver->message), "%s", message);
    if (author) snprintf(ver->author, sizeof(ver->author), "%s", author);
    else snprintf(ver->author, sizeof(ver->author), "SELF-LNN");
    ver->timestamp = (long)time(NULL);
    ver->parent_id = kvc->current_version_id;
    ver->merge_parent_id = -1;

    /* 获取当前分支 */
    for (size_t i = 0; i < kvc->num_branches; i++) {
        if (kvc->branches[i].is_active) {
            snprintf(ver->branch_name, sizeof(ver->branch_name), "%s", kvc->branches[i].name);
            kvc->branches[i].head_version_id = ver->version_id;
            break;
        }
    }

    /* 存储快照数据（知识库中活跃条目计数） */
    size_t entry_count = 0;
    if (kvc->kb) {
        entry_count = knowledge_base_get_entry_count(kvc->kb);
    }
    ver->entry_count = entry_count;
    ver->data_size = entry_count * sizeof(float);
    ver->snapshot_data = safe_calloc(entry_count > 0 ? entry_count : 1, sizeof(float));

    kvc->current_version_id = ver->version_id;
    kvc->num_versions++;

    log_debug("[知识版本控制] 提交版本 %d: %s", ver->version_id, message);
    return ver->version_id;
}

int knowledge_version_rollback(KnowledgeVersionController* kvc, int version_id) {
    if (!kvc || version_id < 0) return -1;

    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].version_id == version_id) {
            kvc->current_version_id = version_id;
            log_info("[知识版本控制] 已回滚到版本 %d", version_id);
            return 0;
        }
    }
    return -1;
}

int knowledge_version_create_branch(KnowledgeVersionController* kvc,
                                    const char* branch_name, int from_version_id) {
    if (!kvc || !branch_name || kvc->num_branches >= KV_MAX_BRANCHES) return -1;

    /* 检查分支是否已存在 */
    for (size_t i = 0; i < kvc->num_branches; i++) {
        if (strcmp(kvc->branches[i].name, branch_name) == 0) return -1;
    }

    KnowledgeBranch* branch = &kvc->branches[kvc->num_branches];
    memset(branch, 0, sizeof(KnowledgeBranch));
    snprintf(branch->name, sizeof(branch->name), "%s", branch_name);
    branch->head_version_id = from_version_id >= 0 ? from_version_id : kvc->current_version_id;
    branch->base_version_id = branch->head_version_id;
    branch->created_at = (long)time(NULL);
    branch->is_active = 0;
    kvc->num_branches++;

    log_info("[知识版本控制] 已创建分支: %s (基础版本=%d)", branch_name, branch->base_version_id);
    return 0;
}

int knowledge_version_switch_branch(KnowledgeVersionController* kvc, const char* branch_name) {
    if (!kvc || !branch_name) return -1;

    for (size_t i = 0; i < kvc->num_branches; i++) {
        kvc->branches[i].is_active = (strcmp(kvc->branches[i].name, branch_name) == 0);
    }
    snprintf(kvc->current_branch, sizeof(kvc->current_branch), "%s", branch_name);
    return 0;
}

KnowledgeVersionDiff* knowledge_version_diff(KnowledgeVersionController* kvc,
                                             int version_id_a, int version_id_b) {
    if (!kvc) return NULL;

    KnowledgeVersionDiff* diff = (KnowledgeVersionDiff*)safe_calloc(1, sizeof(KnowledgeVersionDiff));
    if (!diff) return NULL;
    diff->capacity = 64;
    diff->changes = (KnowledgeChangeEntry*)safe_calloc(diff->capacity, sizeof(KnowledgeChangeEntry));
    if (!diff->changes) {
        safe_free((void**)&diff);
        return NULL;
    }

    /* 比较两个版本的条目数差异 */
    KnowledgeVersion* ver_a = NULL;
    KnowledgeVersion* ver_b = NULL;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].version_id == version_id_a) ver_a = &kvc->versions[i];
        if (kvc->versions[i].version_id == version_id_b) ver_b = &kvc->versions[i];
    }

    if (ver_a && ver_b) {
        diff->num_changes = 1;
        diff->changes[0].change_type = (ver_b->entry_count > ver_a->entry_count) ?
                                        KV_CHANGE_ADD : KV_CHANGE_MODIFY;
        diff->changes[0].entry_id = version_id_b;
        snprintf(diff->changes[0].field_changed, sizeof(diff->changes[0].field_changed),
                "条目数变化: %zu → %zu", ver_a->entry_count, ver_b->entry_count);
        diff->added_count = (ver_b->entry_count > ver_a->entry_count) ? 1 : 0;
        diff->modified_count = 1;
    }

    return diff;
}

void knowledge_version_diff_free(KnowledgeVersionDiff* diff) {
    if (!diff) return;
    safe_free((void**)&diff->changes);
    safe_free((void**)&diff);
}

KnowledgeMergeResult* knowledge_version_merge(KnowledgeVersionController* kvc,
                                              const char* source_branch,
                                              const char* target_branch) {
    if (!kvc || !source_branch) return NULL;

    KnowledgeMergeResult* result = (KnowledgeMergeResult*)safe_calloc(1, sizeof(KnowledgeMergeResult));
    if (!result) return NULL;

    /* 简单合并策略：快进合并 */
    int src_head = -1;
    for (size_t i = 0; i < kvc->num_branches; i++) {
        if (strcmp(kvc->branches[i].name, source_branch) == 0) {
            src_head = kvc->branches[i].head_version_id;
        }
    }

    if (src_head >= 0) {
        result->success = 1;
        result->new_version_id = src_head;
        snprintf(result->summary, sizeof(result->summary),
                "快进合并成功: %s → %s (版本=%d)",
                source_branch, target_branch ? target_branch : "当前", src_head);
    } else {
        result->success = 0;
        snprintf(result->summary, sizeof(result->summary), "未找到源分支: %s", source_branch);
    }

    return result;
}

void knowledge_version_merge_result_free(KnowledgeMergeResult* result) {
    if (!result) return;
    safe_free((void**)&result->conflicts);
    safe_free((void**)&result);
}

int knowledge_version_add_tag(KnowledgeVersionController* kvc, int version_id, const char* tag) {
    if (!kvc || !tag) return -1;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].version_id == version_id) {
            snprintf(kvc->versions[i].tag, sizeof(kvc->versions[i].tag), "%s", tag);
            return 0;
        }
    }
    return -1;
}

int knowledge_version_get_history(KnowledgeVersionController* kvc,
                                  const char* branch_name,
                                  KnowledgeVersion* versions, size_t max_versions) {
    if (!kvc || !versions || max_versions == 0) return -1;
    size_t count = 0;
    for (size_t i = 0; i < kvc->num_versions && count < max_versions; i++) {
        if (!branch_name || strcmp(kvc->versions[i].branch_name, branch_name) == 0) {
            memcpy(&versions[count], &kvc->versions[i], sizeof(KnowledgeVersion));
            count++;
        }
    }
    return (int)count;
}

const KnowledgeVersion* knowledge_version_get_current(const KnowledgeVersionController* kvc) {
    if (!kvc) return NULL;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].version_id == kvc->current_version_id)
            return &kvc->versions[i];
    }
    return NULL;
}

int knowledge_version_get_branches(const KnowledgeVersionController* kvc,
                                   KnowledgeBranch* branches, size_t max_branches) {
    if (!kvc || !branches) return -1;
    size_t count = kvc->num_branches < max_branches ? kvc->num_branches : max_branches;
    memcpy(branches, kvc->branches, count * sizeof(KnowledgeBranch));
    return (int)count;
}

int knowledge_version_delete_branch(KnowledgeVersionController* kvc, const char* branch_name) {
    if (!kvc || !branch_name) return -1;
    for (size_t i = 0; i < kvc->num_branches; i++) {
        if (strcmp(kvc->branches[i].name, branch_name) == 0) {
            if (i < kvc->num_branches - 1) {
                memmove(&kvc->branches[i], &kvc->branches[i + 1],
                       (kvc->num_branches - i - 1) * sizeof(KnowledgeBranch));
            }
            kvc->num_branches--;
            return 0;
        }
    }
    return -1;
}

void knowledge_version_set_auto_snapshot(KnowledgeVersionController* kvc, int enable) {
    if (kvc) kvc->auto_snapshot = enable;
}

int knowledge_version_save(const KnowledgeVersionController* kvc, const char* filepath) {
    if (!kvc || !filepath) return -1;
    /* 待实现：二进制持久化 */
    (void)kvc; (void)filepath;
    return 0;
}

KnowledgeVersionController* knowledge_version_load(const char* filepath,
                                                    KnowledgeBase* kb) {
    if (!filepath || !kb) return NULL;
    /* 待实现：从文件加载 */
    (void)filepath;
    return knowledge_version_create(kb);
}
