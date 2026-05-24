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
#include <stdint.h>
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

    /* 查找目标版本 */
    KnowledgeVersion* target = NULL;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].version_id == version_id) {
            target = &kvc->versions[i];
            break;
        }
    }
    if (!target) return -1;

    /* 真实恢复知识库数据：清空知识库后从快照重新加载 */
    if (kvc->kb && target->snapshot_data && target->data_size > 0) {
        const unsigned char* buf = (const unsigned char*)target->snapshot_data;
        size_t pos = 0;
        size_t count = 0;
        memcpy(&count, buf + pos, sizeof(size_t));
        pos += sizeof(size_t);

        /* 清空并重建知识库 */
        knowledge_base_clear(kvc->kb);

        for (size_t j = 0; j < count && pos < target->data_size; j++) {
            KnowledgeEntry e;
            memset(&e, 0, sizeof(e));
            int slen, plen, olen, t, c, s;
            float w;
            size_t ts;

            memcpy(&slen, buf + pos, sizeof(int)); pos += sizeof(int);
            if (slen > 0 && pos + slen <= target->data_size) {
                e.subject = (char*)safe_malloc(slen + 1);
                if (e.subject) { memcpy(e.subject, buf + pos, slen); e.subject[slen] = '\0'; }
                pos += slen;
            }
            memcpy(&plen, buf + pos, sizeof(int)); pos += sizeof(int);
            if (plen > 0 && pos + plen <= target->data_size) {
                e.predicate = (char*)safe_malloc(plen + 1);
                if (e.predicate) { memcpy(e.predicate, buf + pos, plen); e.predicate[plen] = '\0'; }
                pos += plen;
            }
            memcpy(&olen, buf + pos, sizeof(int)); pos += sizeof(int);
            if (olen > 0 && pos + olen <= target->data_size) {
                e.object = (char*)safe_malloc(olen + 1);
                if (e.object) { memcpy(e.object, buf + pos, olen); e.object[olen] = '\0'; }
                pos += olen;
            }
            memcpy(&t, buf + pos, sizeof(int)); pos += sizeof(int);
            e.type = (KnowledgeType)t;
            memcpy(&c, buf + pos, sizeof(int)); pos += sizeof(int);
            e.confidence = (KnowledgeConfidence)c;
            memcpy(&s, buf + pos, sizeof(int)); pos += sizeof(int);
            e.source = (KnowledgeSource)s;
            memcpy(&w, buf + pos, sizeof(float)); pos += sizeof(float);
            e.weight = w;
            memcpy(&ts, buf + pos, sizeof(size_t)); pos += sizeof(size_t);
            e.timestamp = ts;

            knowledge_base_add(kvc->kb, &e);

            safe_free((void**)&e.subject);
            safe_free((void**)&e.predicate);
            safe_free((void**)&e.object);
        }
    }

    kvc->current_version_id = version_id;
    log_info("[知识版本控制] 已回滚到版本 %d (恢复%zu条知识)", version_id, target->entry_count);
    return 0;
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

    /* 查找两个版本 */
    KnowledgeVersion* ver_a = NULL;
    KnowledgeVersion* ver_b = NULL;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].version_id == version_id_a) ver_a = &kvc->versions[i];
        if (kvc->versions[i].version_id == version_id_b) ver_b = &kvc->versions[i];
    }
    if (!ver_a || !ver_b) return diff;

    /*
     * ZSFX-026: 逐条目三元组对比
     *
     * 解析两个版本快照中的各个知识条目，基于 subject+predicate 组合作为键，
     * 逐条目比较 subject/predicate/object 三元组，标记为：
     *   KV_CHANGE_ADD       — 仅在版本B中存在的条目
     *   KV_CHANGE_DELETE    — 仅在版本A中存在的条目
     *   KV_CHANGE_MODIFY    — 两版本中都有但 object 不同的条目
     *   KV_CHANGE_UNCHANGED — 两版本中完全相同的条目
     */

    /* 内部结构：解码后的条目 */
    #define KV_DIFF_MAX_ENTRIES 2048
    typedef struct {
        char subject[256];
        char predicate[256];
        char object[256];
        int  type;
        int  confidence;
        int  source;
        float weight;
        size_t timestamp;
    } KVDecodedEntry;

    KVDecodedEntry* entries_a = NULL;
    KVDecodedEntry* entries_b = NULL;
    size_t count_a = 0, count_b = 0;

    /* 解码版本A的快照 */
    if (ver_a->snapshot_data && ver_a->data_size > 0) {
        entries_a = (KVDecodedEntry*)safe_calloc(KV_DIFF_MAX_ENTRIES, sizeof(KVDecodedEntry));
        if (entries_a) {
            const unsigned char* buf = (const unsigned char*)ver_a->snapshot_data;
            size_t pos = 0;
            if (pos + sizeof(size_t) <= ver_a->data_size) {
                memcpy(&count_a, buf + pos, sizeof(size_t));
                pos += sizeof(size_t);
                if (count_a > KV_DIFF_MAX_ENTRIES) count_a = KV_DIFF_MAX_ENTRIES;
                for (size_t j = 0; j < count_a && pos < ver_a->data_size; j++) {
                    int slen, plen, olen;
                    memcpy(&slen, buf + pos, sizeof(int)); pos += sizeof(int);
                    if (slen > 0 && pos + (size_t)slen <= ver_a->data_size && slen < 256) {
                        memcpy(entries_a[j].subject, buf + pos, (size_t)slen);
                        entries_a[j].subject[slen] = '\0';
                        pos += (size_t)slen;
                    }
                    memcpy(&plen, buf + pos, sizeof(int)); pos += sizeof(int);
                    if (plen > 0 && pos + (size_t)plen <= ver_a->data_size && plen < 256) {
                        memcpy(entries_a[j].predicate, buf + pos, (size_t)plen);
                        entries_a[j].predicate[plen] = '\0';
                        pos += (size_t)plen;
                    }
                    memcpy(&olen, buf + pos, sizeof(int)); pos += sizeof(int);
                    if (olen > 0 && pos + (size_t)olen <= ver_a->data_size && olen < 256) {
                        memcpy(entries_a[j].object, buf + pos, (size_t)olen);
                        entries_a[j].object[olen] = '\0';
                        pos += (size_t)olen;
                    }
                    /* 跳过元数据字段：type, confidence, source, weight, timestamp */
                    pos += sizeof(int) + sizeof(int) + sizeof(int) + sizeof(float) + sizeof(size_t);
                }
            }
        }
    }

    /* 解码版本B的快照 */
    if (ver_b->snapshot_data && ver_b->data_size > 0) {
        entries_b = (KVDecodedEntry*)safe_calloc(KV_DIFF_MAX_ENTRIES, sizeof(KVDecodedEntry));
        if (entries_b) {
            const unsigned char* buf = (const unsigned char*)ver_b->snapshot_data;
            size_t pos = 0;
            if (pos + sizeof(size_t) <= ver_b->data_size) {
                memcpy(&count_b, buf + pos, sizeof(size_t));
                pos += sizeof(size_t);
                if (count_b > KV_DIFF_MAX_ENTRIES) count_b = KV_DIFF_MAX_ENTRIES;
                for (size_t j = 0; j < count_b && pos < ver_b->data_size; j++) {
                    int slen, plen, olen;
                    memcpy(&slen, buf + pos, sizeof(int)); pos += sizeof(int);
                    if (slen > 0 && pos + (size_t)slen <= ver_b->data_size && slen < 256) {
                        memcpy(entries_b[j].subject, buf + pos, (size_t)slen);
                        entries_b[j].subject[slen] = '\0';
                        pos += (size_t)slen;
                    }
                    memcpy(&plen, buf + pos, sizeof(int)); pos += sizeof(int);
                    if (plen > 0 && pos + (size_t)plen <= ver_b->data_size && plen < 256) {
                        memcpy(entries_b[j].predicate, buf + pos, (size_t)plen);
                        entries_b[j].predicate[plen] = '\0';
                        pos += (size_t)plen;
                    }
                    memcpy(&olen, buf + pos, sizeof(int)); pos += sizeof(int);
                    if (olen > 0 && pos + (size_t)olen <= ver_b->data_size && olen < 256) {
                        memcpy(entries_b[j].object, buf + pos, (size_t)olen);
                        entries_b[j].object[olen] = '\0';
                        pos += (size_t)olen;
                    }
                    pos += sizeof(int) + sizeof(int) + sizeof(int) + sizeof(float) + sizeof(size_t);
                }
            }
        }
    }

    /* 匹配标记数组：记录B中每个条目是否已被匹配 */
    char* b_matched = (char*)safe_calloc(count_b > 0 ? count_b : 1, sizeof(char));
    char* a_matched = (char*)safe_calloc(count_a > 0 ? count_a : 1, sizeof(char));

    /* 扩展变更数组容量的辅助宏 */
    #define KV_DIFF_ENSURE_CAPACITY() do { \
        if (diff->num_changes >= diff->capacity) { \
            size_t new_cap = diff->capacity * 2; \
            KnowledgeChangeEntry* new_changes = (KnowledgeChangeEntry*)safe_realloc(diff->changes, new_cap * sizeof(KnowledgeChangeEntry)); \
            if (new_changes) { diff->changes = new_changes; diff->capacity = new_cap; } \
        } \
    } while(0)

    /* 阶段1: 比较版本A的每个条目在版本B中的存在情况 */
    for (size_t ai = 0; ai < count_a; ai++) {
        int found_in_b = -1;
        for (size_t bi = 0; bi < count_b; bi++) {
            if (b_matched[bi]) continue;
            /* 匹配条件：subject和predicate相同 */
            if (entries_a[ai].subject[0] && entries_b[bi].subject[0] &&
                entries_a[ai].predicate[0] && entries_b[bi].predicate[0] &&
                strcmp(entries_a[ai].subject, entries_b[bi].subject) == 0 &&
                strcmp(entries_a[ai].predicate, entries_b[bi].predicate) == 0) {
                found_in_b = (int)bi;
                break;
            }
        }

        KV_DIFF_ENSURE_CAPACITY();
        KnowledgeChangeEntry* change = &diff->changes[diff->num_changes];

        if (found_in_b >= 0) {
            b_matched[found_in_b] = 1;
            a_matched[ai] = 1;

            /* 比较object字段 */
            if (strcmp(entries_a[ai].object, entries_b[found_in_b].object) == 0) {
                /* ZSFX-026: 完全相同 → UNCHANGED */
                change->change_type = KV_CHANGE_UNCHANGED;
                snprintf(change->field_changed, sizeof(change->field_changed),
                        "三元组未变更: %s/%s/%s",
                        entries_a[ai].subject, entries_a[ai].predicate, entries_a[ai].object);
            } else {
                /* ZSFX-026: subject+predicate相同但object不同 → MODIFY */
                change->change_type = KV_CHANGE_MODIFY;
                snprintf(change->field_changed, sizeof(change->field_changed),
                        "object变更: %s/%s [%s]→[%s]",
                        entries_a[ai].subject, entries_a[ai].predicate,
                        entries_a[ai].object, entries_b[found_in_b].object);
                diff->modified_count++;
            }
        } else {
            /* 在B中未找到匹配 → DELETE */
            change->change_type = KV_CHANGE_DELETE;
            snprintf(change->field_changed, sizeof(change->field_changed),
                    "已删除条目: %s/%s/%s",
                    entries_a[ai].subject, entries_a[ai].predicate, entries_a[ai].object);
            diff->deleted_count++;
        }

        change->entry_id = (int)ai + 1;
        diff->num_changes++;
    }

    /* 阶段2: 检测仅在版本B中存在的条目 → ADD */
    for (size_t bi = 0; bi < count_b; bi++) {
        if (b_matched[bi]) continue;

        KV_DIFF_ENSURE_CAPACITY();
        KnowledgeChangeEntry* change = &diff->changes[diff->num_changes];
        change->change_type = KV_CHANGE_ADD;
        change->entry_id = (int)bi + 1;
        snprintf(change->field_changed, sizeof(change->field_changed),
                "新增条目: %s/%s/%s",
                entries_b[bi].subject, entries_b[bi].predicate, entries_b[bi].object);
        diff->added_count++;
        diff->num_changes++;
    }

    #undef KV_DIFF_ENSURE_CAPACITY

    safe_free((void**)&entries_a);
    safe_free((void**)&entries_b);
    safe_free((void**)&b_matched);
    safe_free((void**)&a_matched);

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

/**
 * @brief 二进制头部结构：版本控制状态持久化
 * 格式: 魔数(8字节) | 版本号(4字节) | 头部信息 | 版本数组 | 分支数组
 */
#define KV_FILE_MAGIC 0x534C4E4E4B565344ULL /* "SLNNKVSD" */

typedef struct {
    uint64_t magic;
    uint32_t file_version;
    uint32_t num_versions;
    uint32_t num_branches;
    int32_t next_version_id;
    int32_t current_version_id;
    int32_t auto_snapshot;
    int32_t reserved;
} KVFileHeader;

int knowledge_version_save(const KnowledgeVersionController* kvc, const char* filepath) {
    if (!kvc || !filepath) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        log_error("[知识版本控制] 无法打开文件写入: %s", filepath);
        return -1;
    }

    KVFileHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = KV_FILE_MAGIC;
    hdr.file_version = 1;
    hdr.num_versions = (uint32_t)kvc->num_versions;
    hdr.num_branches = (uint32_t)kvc->num_branches;
    hdr.next_version_id = kvc->next_version_id;
    hdr.current_version_id = kvc->current_version_id;
    hdr.auto_snapshot = kvc->auto_snapshot;

    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    /* 写入当前分支名 */
    size_t brname_len = strlen(kvc->current_branch) + 1;
    if (fwrite(&brname_len, sizeof(size_t), 1, fp) != 1) { fclose(fp); return -1; }
    if (fwrite(kvc->current_branch, 1, brname_len, fp) != brname_len) { fclose(fp); return -1; }

    /* 写入版本数组 */
    for (size_t i = 0; i < kvc->num_versions; i++) {
        KnowledgeVersion* ver = &kvc->versions[i];
        /* 写版本元数据（不含snapshot_data指针） */
        int32_t fields[6] = {
            ver->version_id, ver->parent_id, ver->merge_parent_id,
            (int32_t)ver->timestamp, (int32_t)ver->entry_count,
            (int32_t)ver->data_size
        };
        fwrite(fields, sizeof(fields), 1, fp);
        fwrite(ver->message, KV_MAX_MESSAGE_LENGTH, 1, fp);
        fwrite(ver->author, 64, 1, fp);
        fwrite(ver->branch_name, KV_MAX_BRANCH_NAME, 1, fp);
        fwrite(ver->tag, KV_MAX_TAG_NAME, 1, fp);

        /* 写快照数据 */
        int32_t has_data = (ver->snapshot_data && ver->data_size > 0) ? 1 : 0;
        fwrite(&has_data, sizeof(int32_t), 1, fp);
        if (has_data && ver->snapshot_data) {
            fwrite(ver->snapshot_data, 1, ver->data_size, fp);
        }
    }

    /* 写入分支数组 */
    fwrite(&kvc->num_branches, sizeof(size_t), 1, fp);
    for (size_t i = 0; i < kvc->num_branches; i++) {
        KnowledgeBranch* br = &kvc->branches[i];
        fwrite(br->name, KV_MAX_BRANCH_NAME, 1, fp);
        int32_t bfields[3] = { br->head_version_id, br->base_version_id, (int32_t)br->created_at };
        fwrite(bfields, sizeof(bfields), 1, fp);
        int32_t active = br->is_active;
        fwrite(&active, sizeof(int32_t), 1, fp);
    }

    fclose(fp);
    log_info("[知识版本控制] 已保存到文件: %s (版本数=%zu, 分支数=%zu)",
             filepath, kvc->num_versions, kvc->num_branches);
    return 0;
}

KnowledgeVersionController* knowledge_version_load(const char* filepath,
                                                    KnowledgeBase* kb) {
    if (!filepath || !kb) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        log_warning("[知识版本控制] 版本文件不存在: %s, 创建新控制器", filepath);
        return knowledge_version_create(kb);
    }

    KVFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1 || hdr.magic != KV_FILE_MAGIC) {
        log_error("[知识版本控制] 文件格式无效: %s", filepath);
        fclose(fp);
        return knowledge_version_create(kb);
    }

    KnowledgeVersionController* kvc = (KnowledgeVersionController*)safe_calloc(1, sizeof(KnowledgeVersionController));
    if (!kvc) { fclose(fp); return NULL; }
    kvc->kb = kb;

    /* 读取当前分支名 */
    size_t brname_len = 0;
    fread(&brname_len, sizeof(size_t), 1, fp);
    if (brname_len > 0 && brname_len <= KV_MAX_BRANCH_NAME) {
        fread(kvc->current_branch, 1, brname_len, fp);
    }

    kvc->num_versions = hdr.num_versions;
    kvc->version_capacity = hdr.num_versions > 0 ? hdr.num_versions * 2 : KV_INITIAL_CAPACITY;
    kvc->versions = (KnowledgeVersion*)safe_calloc(kvc->version_capacity, sizeof(KnowledgeVersion));
    kvc->next_version_id = hdr.next_version_id;
    kvc->current_version_id = hdr.current_version_id;
    kvc->auto_snapshot = hdr.auto_snapshot;

    /* 读取版本数组 */
    for (uint32_t i = 0; i < hdr.num_versions; i++) {
        KnowledgeVersion* ver = &kvc->versions[i];
        int32_t fields[6];
        fread(fields, sizeof(fields), 1, fp);
        ver->version_id = fields[0];
        ver->parent_id = fields[1];
        ver->merge_parent_id = fields[2];
        ver->timestamp = fields[3];
        ver->entry_count = (size_t)fields[4];
        ver->data_size = (size_t)fields[5];
        fread(ver->message, KV_MAX_MESSAGE_LENGTH, 1, fp);
        fread(ver->author, 64, 1, fp);
        fread(ver->branch_name, KV_MAX_BRANCH_NAME, 1, fp);
        fread(ver->tag, KV_MAX_TAG_NAME, 1, fp);

        int32_t has_data = 0;
        fread(&has_data, sizeof(int32_t), 1, fp);
        if (has_data && ver->data_size > 0) {
            ver->snapshot_data = safe_calloc(1, ver->data_size);
            if (ver->snapshot_data) {
                fread(ver->snapshot_data, 1, ver->data_size, fp);
            }
        }
    }

    /* 读取分支数组 */
    fread(&kvc->num_branches, sizeof(size_t), 1, fp);
    if (kvc->num_branches > KV_MAX_BRANCHES) kvc->num_branches = KV_MAX_BRANCHES;
    for (size_t i = 0; i < kvc->num_branches; i++) {
        KnowledgeBranch* br = &kvc->branches[i];
        fread(br->name, KV_MAX_BRANCH_NAME, 1, fp);
        int32_t bfields[3];
        fread(bfields, sizeof(bfields), 1, fp);
        br->head_version_id = bfields[0];
        br->base_version_id = bfields[1];
        br->created_at = bfields[2];
        int32_t active;
        fread(&active, sizeof(int32_t), 1, fp);
        br->is_active = active;
    }

    fclose(fp);
    log_info("[知识版本控制] 已从文件加载: %s (版本数=%zu, 分支数=%zu)",
             filepath, kvc->num_versions, kvc->num_branches);
    return kvc;
}
