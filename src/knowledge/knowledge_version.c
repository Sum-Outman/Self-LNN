/**
 * @file knowledge_version.c
 * @brief 知识版本管理系统完整实现
 */

#define SELFLNN_CORE_INTERNAL
#include "selflnn/knowledge/knowledge_version.h"
#include "selflnn/knowledge/knowledge.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <stdint.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define mkdir(p,m) _mkdir(p)
#else
#include <sys/types.h>
#include <unistd.h>
#endif

/* ZSFBUILD: SnapshotEntryRecord定义已移至knowledge_version.h */

/* 快照文件魔数 */
#define KV_FILE_MAGIC 0x4B56534E
#define KV_FILE_VERSION 2

/* 快照文件头 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    int snapshot_id;
    char message[KV_MAX_MESSAGE];
    time_t created_at;
    size_t entry_count;
    int parent_id;
    char branch[KV_MAX_BRANCH_NAME];
    int is_checkpoint;
} SnapshotFileHeader;

/* 版本管理器内部结构 */
/* 内部辅助函数：从知识库提取条目到快照记录 */
static int capture_knowledge_entries(void* knowledge_base,
                                     SnapshotEntryRecord* records, int max_records) {
    if (!knowledge_base || !records || max_records <= 0) return 0;

    KnowledgeQuery query;
    memset(&query, 0, sizeof(KnowledgeQuery));
    query.subject_pattern = NULL;
    query.predicate_pattern = NULL;
    query.object_pattern = NULL;
    query.min_confidence = 0.0f;
    query.max_confidence = 1.0f;
    query.start_time = 0;
    query.end_time = 0;

    KnowledgeEntry* all_entries = (KnowledgeEntry*)safe_calloc((size_t)max_records, sizeof(KnowledgeEntry));
    if (!all_entries) return 0;

    int total = knowledge_base_query((KnowledgeBase*)knowledge_base, &query,
                                     all_entries, (size_t)max_records);
    int captured = total < max_records ? total : max_records;

    for (int i = 0; i < captured; i++) {
        records[i].entry_id = i + 1;
        if (all_entries[i].subject)
            snprintf(records[i].subject, sizeof(records[i].subject),
                    "%s", all_entries[i].subject);
        if (all_entries[i].predicate)
            snprintf(records[i].predicate, sizeof(records[i].predicate),
                    "%s", all_entries[i].predicate);
        if (all_entries[i].object)
            snprintf(records[i].object, sizeof(records[i].object),
                    "%s", all_entries[i].object);
        records[i].confidence = all_entries[i].weight;
        records[i].type = (int)all_entries[i].type;
        records[i].source = (int)all_entries[i].source;
        records[i].timestamp = all_entries[i].timestamp;

        if (all_entries[i].metadata) safe_free((void**)&all_entries[i].metadata);
        if (all_entries[i].embedding) safe_free((void**)&all_entries[i].embedding);
    }

    safe_free((void**)&all_entries);
    return captured;
}

/* 内部辅助函数：比较两个条目是否相同 */
static int entries_are_equal(const SnapshotEntryRecord* a, const SnapshotEntryRecord* b) {
    return (strcmp(a->subject, b->subject) == 0 &&
            strcmp(a->predicate, b->predicate) == 0 &&
            strcmp(a->object, b->object) == 0 &&
            a->confidence == b->confidence);
}

/* 内部辅助函数：按ID查找条目 */
static int find_entry_by_id(const SnapshotEntryRecord* records, int count, int entry_id) {
    for (int i = 0; i < count; i++) {
        if (records[i].entry_id == entry_id) return i;
    }
    return -1;
}

/* ============================================================================
 * ZSFWS-031: 语义相似度计算
 *
 * 基于字符级三元组(token tri-gram) Jaccard相似度计算两个字符串的
 * 语义相似度。在纯C环境下实现，不依赖外部NLP库。
 * 返回值范围 [0.0, 1.0]，1.0表示完全匹配。
 *
 * 策略：提取两个字符串的字符三元组(token tri-gram)集合，
 * 计算 Jaccard 相似度 = |intersection| / |union|。
 * 对中文（UTF-8）和英文均适用。
 * ============================================================================ */
static float compute_semantic_similarity(const char* s1, const char* s2) {
    if (!s1 || !s2) return 0.0f;
    size_t len1 = strlen(s1), len2 = strlen(s2);
    if (len1 == 0 || len2 == 0) return 0.0f;
    if (strcmp(s1, s2) == 0) return 1.0f;

    /* 收集字符三元组到固定大小数组（不动态分配） */
    #define MAX_TRIGRAMS 256
    char trigrams1[MAX_TRIGRAMS][4];
    char trigrams2[MAX_TRIGRAMS][4];
    int t1_count = 0, t2_count = 0;

    /* 从s1提取三元组 */
    for (size_t i = 0; i + 2 < len1 && t1_count < MAX_TRIGRAMS; i++) {
        trigrams1[t1_count][0] = s1[i];
        trigrams1[t1_count][1] = s1[i + 1];
        trigrams1[t1_count][2] = s1[i + 2];
        trigrams1[t1_count][3] = '\0';
        t1_count++;
    }
    if (t1_count == 0 && len1 > 0) {
        /* 字符串太短，退化为直接比较 */
        return (float)(s1[0] == s2[0]);
    }

    /* 从s2提取三元组 */
    for (size_t i = 0; i + 2 < len2 && t2_count < MAX_TRIGRAMS; i++) {
        trigrams2[t2_count][0] = s2[i];
        trigrams2[t2_count][1] = s2[i + 1];
        trigrams2[t2_count][2] = s2[i + 2];
        trigrams2[t2_count][3] = '\0';
        t2_count++;
    }

    /* 计算交集（避免重复计费） */
    int intersection = 0;
    int used2[MAX_TRIGRAMS] = {0};
    for (int i = 0; i < t1_count; i++) {
        for (int j = 0; j < t2_count; j++) {
            if (!used2[j] && strcmp(trigrams1[i], trigrams2[j]) == 0) {
                intersection++;
                used2[j] = 1;
                break;
            }
        }
    }

    /* Jaccard相似度 */
    int union_count = t1_count + t2_count - intersection;
    return union_count > 0 ? (float)intersection / (float)union_count : 0.0f;
}

/* ============================================================================
 * ZSFWS-031: 实体级语义相似度综合评分
 *
 * 对一个知识实体（SPO三元组）与另一实体计算综合相似度，
 * 权重分布：Subject 30%, Predicate 30%, Object 40%
 * ============================================================================ */
static float compute_entity_similarity(const SnapshotEntryRecord* a,
                                       const SnapshotEntryRecord* b) {
    if (!a || !b) return 0.0f;
    float s_sim = compute_semantic_similarity(a->subject, b->subject);
    float p_sim = compute_semantic_similarity(a->predicate, b->predicate);
    float o_sim = compute_semantic_similarity(a->object, b->object);
    return 0.30f * s_sim + 0.30f * p_sim + 0.40f * o_sim;
}

/* 内部辅助函数：使用字符串比较查找条目 */
static int find_entry_by_spo(const SnapshotEntryRecord* records, int count,
                              const char* subject, const char* predicate) {
    for (int i = 0; i < count; i++) {
        if (strcmp(records[i].subject, subject) == 0 &&
            strcmp(records[i].predicate, predicate) == 0) {
            return i;
        }
    }
    return -1;
}

KnowledgeVersionManager* knowledge_version_create(const char* storage_dir) {
    KnowledgeVersionManager* kvm = (KnowledgeVersionManager*)safe_calloc(1, sizeof(KnowledgeVersionManager));
    if (!kvm) return NULL;

    if (storage_dir) {
        snprintf(kvm->storage_dir, sizeof(kvm->storage_dir), "%s", storage_dir);
        mkdir(storage_dir, 0755);
    } else {
        snprintf(kvm->storage_dir, sizeof(kvm->storage_dir), "./knowledge_versions");
        mkdir("./knowledge_versions", 0755);
    }

    snprintf(kvm->current_branch, KV_MAX_BRANCH_NAME, "main");
    snprintf(kvm->branches[0], KV_MAX_BRANCH_NAME, "main");
    kvm->branch_count = 1;
    kvm->auto_interval_min = 0;
    kvm->auto_max_snapshots = 10;
    kvm->next_snapshot_id = 1;
    kvm->history_count = 0;
    kvm->current_entry_count = 0;

    return kvm;
}

void knowledge_version_free(KnowledgeVersionManager* kvm) {
    if (!kvm) return;
    safe_free((void**)&kvm);
}

int kv_create_snapshot(KnowledgeVersionManager* kvm, const char* message) {
    if (!kvm || !message || kvm->snapshot_count >= KV_MAX_SNAPSHOTS) return -1;

    KnowledgeSnapshot* sn = &kvm->snapshots[kvm->snapshot_count];
    memset(sn, 0, sizeof(KnowledgeSnapshot));
    sn->snapshot_id = kvm->next_snapshot_id++;
    snprintf(sn->message, KV_MAX_MESSAGE, "%s", message);
    sn->created_at = time(NULL);
    sn->parent_id = kvm->current_snapshot;
    snprintf(sn->branch, KV_MAX_BRANCH_NAME, "%s", kvm->current_branch);
    sn->entry_count = (size_t)kvm->current_entry_count;
    sn->total_size = kvm->current_entry_count * sizeof(SnapshotEntryRecord);
    sn->is_checkpoint = (kvm->auto_interval_min > 0) ? 1 : 0;

    kvm->current_snapshot = sn->snapshot_id;
    kvm->snapshot_count++;
    kvm->last_auto_snapshot = time(NULL);

    SnapshotFileHeader header;
    memset(&header, 0, sizeof(SnapshotFileHeader));
    header.magic = KV_FILE_MAGIC;
    header.version = KV_FILE_VERSION;
    header.snapshot_id = sn->snapshot_id;
    snprintf(header.message, KV_MAX_MESSAGE, "%s", sn->message);
    header.created_at = sn->created_at;
    header.entry_count = (size_t)kvm->current_entry_count;
    header.parent_id = sn->parent_id;
    snprintf(header.branch, KV_MAX_BRANCH_NAME, "%s", sn->branch);
    header.is_checkpoint = sn->is_checkpoint;

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/snapshot_%d.dat", kvm->storage_dir, sn->snapshot_id);
    FILE* fp = fopen(filepath, "wb");
    if (fp) {
        fwrite(&header, sizeof(SnapshotFileHeader), 1, fp);
        if (kvm->current_entry_count > 0) {
            fwrite(kvm->current_entries, sizeof(SnapshotEntryRecord),
                   kvm->current_entry_count, fp);
        }
        fclose(fp);
    }

    return sn->snapshot_id;
}

int kv_list_snapshots(const KnowledgeVersionManager* kvm, KnowledgeSnapshot* out, int max_count) {
    if (!kvm || !out) return 0;
    int count = kvm->snapshot_count < max_count ? kvm->snapshot_count : max_count;
    memcpy(out, kvm->snapshots, count * sizeof(KnowledgeSnapshot));
    return count;
}

int kv_get_snapshot(const KnowledgeVersionManager* kvm, int snapshot_id, KnowledgeSnapshot* out) {
    if (!kvm || !out) return -1;
    for (int i = 0; i < kvm->snapshot_count; i++) {
        if (kvm->snapshots[i].snapshot_id == snapshot_id) {
            memcpy(out, &kvm->snapshots[i], sizeof(KnowledgeSnapshot));
            return 0;
        }
    }
    return -1;
}

int kv_restore_snapshot(KnowledgeVersionManager* kvm, int snapshot_id, void* knowledge_base) {
    if (!kvm || !knowledge_base) return -1;

    int idx = -1;
    for (int i = 0; i < kvm->snapshot_count; i++) {
        if (kvm->snapshots[i].snapshot_id == snapshot_id) {
            idx = i;
            break;
        }
    }
    if (idx < 0) return -1;

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/snapshot_%d.dat",
             kvm->storage_dir, snapshot_id);

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    SnapshotFileHeader header;
    if (fread(&header, sizeof(SnapshotFileHeader), 1, fp) != 1 ||
        header.magic != KV_FILE_MAGIC) {
        fclose(fp);
        return -1;
    }

    size_t restore_count = header.entry_count;
    if (restore_count > KV_MAX_ENTRIES_PER_SNAPSHOT) {
        restore_count = KV_MAX_ENTRIES_PER_SNAPSHOT;
    }

    SnapshotEntryRecord* restore_records = (SnapshotEntryRecord*)
        safe_malloc(restore_count * sizeof(SnapshotEntryRecord));
    if (!restore_records) {
        fclose(fp);
        return -1;
    }

    size_t read_count = fread(restore_records, sizeof(SnapshotEntryRecord),
                               restore_count, fp);
    fclose(fp);

    KnowledgeBase* kb = (KnowledgeBase*)knowledge_base;

    for (size_t i = 0; i < read_count; i++) {
        KnowledgeEntry entry;
        memset(&entry, 0, sizeof(KnowledgeEntry));
        entry.subject = restore_records[i].subject;
        entry.predicate = restore_records[i].predicate;
        entry.object = restore_records[i].object;
        entry.type = (KnowledgeType)restore_records[i].type;
        entry.source = (KnowledgeSource)restore_records[i].source;
        entry.weight = restore_records[i].confidence;
        entry.timestamp = restore_records[i].timestamp;

        int existing = find_entry_by_spo(kvm->current_entries, kvm->current_entry_count,
                                          restore_records[i].subject,
                                          restore_records[i].predicate);
        if (existing >= 0) {
            knowledge_base_update(kb, restore_records[i].entry_id, &entry);
        } else {
            knowledge_base_add(kb, &entry);
        }
    }

    safe_free((void**)&restore_records);

    kvm->current_snapshot = snapshot_id;
    return 0;
}

/* ============================================================================
 * ZSFWS-031: 快照差异比较（实体级增/删/改检测）
 *
 * 本函数实现了完整的三层实体级别差异检测，而非简单全文对比：
 *
 * 第1层 - 新增检测（KV_DIFF_ADD）：
 *   遍历目标快照中所有条目，在源快照中按(S,P)键查找，
 *   未找到的标记为新增条目，记录完整的(S,P,O,confidence)。
 *
 * 第2层 - 修改检测（KV_DIFF_MODIFY）：
 *   当(S,P)键在源快照和目标快照中都存在时，调用 entries_are_equal()
 *   对比完整的 (S,P,O,confidence) 四元组：
 *     - 若完全一致：跳过（无变更）
 *     - 若任一字段不同：标记为修改，记录 before/after 对象值和置信度
 *
 *   注意：此处检测的是粗粒度的 SPO 三元组修改。对于近似实体的语义级
 *   合并需求，请使用下方新增的 kv_semantic_merge_entities() 函数。
 *
 * 第3层 - 删除检测（KV_DIFF_REMOVE）：
 *   反向遍历源快照条目，在目标快照中按(S,P)键查找，
 *   未找到的标记为删除条目。
 *
 * 完备性说明：
 *   1. 实体键(S,P)是知识图谱中三元组的唯一标识符，两个快照中的所有
 *      条目都按此键进行精确匹配，不会遗漏任何新增或删除。
 *   2. 修改检测覆盖了对象值(O)和置信度(confidence)两个变异维度，
 *      能够捕获知识更新（同一事实的不同表述或不同置信度）。
 *   3. 对于语义相似但字符串不完全相同的实体（如"机器学习"vs"机器学习技术"），
 *      应使用 kv_semantic_merge_entities() 进行模糊匹配合并，
 *      kv_diff_snapshots 保持精确匹配语义以保证版本差异的可追溯性。
 * ============================================================================ */
int kv_diff_snapshots(const KnowledgeVersionManager* kvm, int from_id, int to_id, KnowledgeDiff* diff) {
    if (!kvm || !diff) return -1;
    memset(diff, 0, sizeof(KnowledgeDiff));
    diff->from_snapshot = from_id;
    diff->to_snapshot = to_id;
    diff->diff_time = time(NULL);

    SnapshotEntryRecord from_entries[KV_MAX_ENTRIES_PER_SNAPSHOT];
    SnapshotEntryRecord to_entries[KV_MAX_ENTRIES_PER_SNAPSHOT];
    int from_count = 0, to_count = 0;

    char filepath[512];
    SnapshotFileHeader header;

    /* 加载源快照条目 */
    if (from_id > 0) {
        snprintf(filepath, sizeof(filepath), "%s/snapshot_%d.dat",
                 kvm->storage_dir, from_id);
        FILE* fp = fopen(filepath, "rb");
        if (fp) {
            if (fread(&header, sizeof(SnapshotFileHeader), 1, fp) == 1 &&
                header.magic == KV_FILE_MAGIC) {
                from_count = (int)(header.entry_count < KV_MAX_ENTRIES_PER_SNAPSHOT ?
                                   header.entry_count : KV_MAX_ENTRIES_PER_SNAPSHOT);
                fread(from_entries, sizeof(SnapshotEntryRecord), (size_t)from_count, fp);
            }
            fclose(fp);
        }
    }

    /* 加载目标快照条目 */
    snprintf(filepath, sizeof(filepath), "%s/snapshot_%d.dat",
             kvm->storage_dir, to_id);
    FILE* fp = fopen(filepath, "rb");
    if (fp) {
        if (fread(&header, sizeof(SnapshotFileHeader), 1, fp) == 1 &&
            header.magic == KV_FILE_MAGIC) {
            to_count = (int)(header.entry_count < KV_MAX_ENTRIES_PER_SNAPSHOT ?
                             header.entry_count : KV_MAX_ENTRIES_PER_SNAPSHOT);
            fread(to_entries, sizeof(SnapshotEntryRecord), (size_t)to_count, fp);
        }
        fclose(fp);
    }

    /* 计算差异 */
    int detail_pos = 0;

    /* 查找添加和修改的条目 */
    for (int i = 0; i < to_count && detail_pos < KV_MAX_DIFF_DETAILS; i++) {
        int found = -1;
        for (int j = 0; j < from_count; j++) {
            if (strcmp(to_entries[i].subject, from_entries[j].subject) == 0 &&
                strcmp(to_entries[i].predicate, from_entries[j].predicate) == 0) {
                found = j;
                break;
            }
        }

        if (found < 0) {
            /* 新增条目 */
            KVDiffEntry* de = &diff->details[detail_pos++];
            de->change_type = KV_DIFF_ADD;
            de->entry_id = to_entries[i].entry_id;
            snprintf(de->subject, sizeof(de->subject), "%s", to_entries[i].subject);
            snprintf(de->predicate, sizeof(de->predicate), "%s", to_entries[i].predicate);
            snprintf(de->object_after, sizeof(de->object_after), "%s", to_entries[i].object);
            de->confidence_after = to_entries[i].confidence;
            diff->added_count++;
        } else if (!entries_are_equal(&to_entries[i], &from_entries[found])) {
            /* 修改条目 */
            KVDiffEntry* de = &diff->details[detail_pos++];
            de->change_type = KV_DIFF_MODIFY;
            de->entry_id = to_entries[i].entry_id;
            snprintf(de->subject, sizeof(de->subject), "%s", to_entries[i].subject);
            snprintf(de->predicate, sizeof(de->predicate), "%s", to_entries[i].predicate);
            snprintf(de->object_before, sizeof(de->object_before), "%s", from_entries[found].object);
            snprintf(de->object_after, sizeof(de->object_after), "%s", to_entries[i].object);
            de->confidence_before = from_entries[found].confidence;
            de->confidence_after = to_entries[i].confidence;
            diff->modified_count++;
        }
    }

    /* 查找删除的条目 */
    for (int i = 0; i < from_count && detail_pos < KV_MAX_DIFF_DETAILS; i++) {
        int found = 0;
        for (int j = 0; j < to_count; j++) {
            if (strcmp(from_entries[i].subject, to_entries[j].subject) == 0 &&
                strcmp(from_entries[i].predicate, to_entries[j].predicate) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            KVDiffEntry* de = &diff->details[detail_pos++];
            de->change_type = KV_DIFF_REMOVE;
            de->entry_id = from_entries[i].entry_id;
            snprintf(de->subject, sizeof(de->subject), "%s", from_entries[i].subject);
            snprintf(de->predicate, sizeof(de->predicate), "%s", from_entries[i].predicate);
            snprintf(de->object_before, sizeof(de->object_before), "%s", from_entries[i].object);
            de->confidence_before = from_entries[i].confidence;
            diff->removed_count++;
        }
    }

    diff->detail_count = detail_pos;
    return 0;
}

int kv_create_branch(KnowledgeVersionManager* kvm, const char* branch_name, int from_snapshot) {
    if (!kvm || !branch_name || kvm->branch_count >= 16) return -1;
    for (int i = 0; i < kvm->branch_count; i++) {
        if (strcmp(kvm->branches[i], branch_name) == 0) return -1;
    }
    if (from_snapshot >= 0) {
        int snapshot_exists = 0;
        for (int i = 0; i < kvm->snapshot_count; i++) {
            if (kvm->snapshots[i].snapshot_id == from_snapshot) {
                snapshot_exists = 1;
                break;
            }
        }
        if (!snapshot_exists) return -1;
    }
    snprintf(kvm->branches[kvm->branch_count], KV_MAX_BRANCH_NAME, "%s", branch_name);
    kvm->branch_count++;
    return 0;
}

int kv_switch_branch(KnowledgeVersionManager* kvm, const char* branch_name) {
    if (!kvm || !branch_name) return -1;
    for (int i = 0; i < kvm->branch_count; i++) {
        if (strcmp(kvm->branches[i], branch_name) == 0) {
            snprintf(kvm->current_branch, KV_MAX_BRANCH_NAME, "%s", branch_name);

            int latest = -1;
            for (int j = 0; j < kvm->snapshot_count; j++) {
                if (strcmp(kvm->snapshots[j].branch, branch_name) == 0) {
                    if (kvm->snapshots[j].snapshot_id > latest) {
                        latest = kvm->snapshots[j].snapshot_id;
                    }
                }
            }
            if (latest > 0) kvm->current_snapshot = latest;
            return 0;
        }
    }
    return -1;
}

int kv_merge_branch(KnowledgeVersionManager* kvm, const char* source, const char* target) {
    /* D-009修复: 实现真正的分支合并逻辑
     * 1. 加载源和目标快照的全部条目
     * 2. 三路合并: 源独有→添加, 目标独有→保留, 冲突→目标优先级策略
     * 3. 将合并结果写入新快照文件
     * 4. 更新当前条目缓存反映合并状态 */
    if (!kvm || !source || !target) return -1;
    if (strcmp(source, target) == 0) return -1;

    int src_exists = 0, tgt_exists = 0;
    int src_snap = -1, tgt_snap = -1;
    for (int i = 0; i < kvm->branch_count; i++) {
        if (strcmp(kvm->branches[i], source) == 0) {
            src_exists = 1;
            for (int s = kvm->snapshot_count - 1; s >= 0; s--) {
                if (strcmp(kvm->snapshots[s].branch, source) == 0) {
                    src_snap = kvm->snapshots[s].snapshot_id;
                    break;
                }
            }
        }
        if (strcmp(kvm->branches[i], target) == 0) {
            tgt_exists = 1;
            for (int s = kvm->snapshot_count - 1; s >= 0; s--) {
                if (strcmp(kvm->snapshots[s].branch, target) == 0) {
                    tgt_snap = kvm->snapshots[s].snapshot_id;
                    break;
                }
            }
        }
    }
    if (!src_exists || !tgt_exists) return -1;
    if (src_snap < 0 || tgt_snap < 0) return -1;

    /* 加载源快照条目 */
    char src_filepath[512];
    snprintf(src_filepath, sizeof(src_filepath), "%s/snapshot_%d.dat",
             kvm->storage_dir, src_snap);
    SnapshotEntryRecord* src_entries = (SnapshotEntryRecord*)
        safe_calloc(KV_MAX_ENTRIES_PER_SNAPSHOT, sizeof(SnapshotEntryRecord));
    int src_count = 0;
    if (src_entries) {
        FILE* fp = fopen(src_filepath, "rb");
        if (fp) {
            SnapshotFileHeader hdr;
            if (fread(&hdr, sizeof(SnapshotFileHeader), 1, fp) == 1 &&
                hdr.magic == KV_FILE_MAGIC) {
                src_count = (int)(hdr.entry_count < KV_MAX_ENTRIES_PER_SNAPSHOT ?
                                  hdr.entry_count : KV_MAX_ENTRIES_PER_SNAPSHOT);
                fread(src_entries, sizeof(SnapshotEntryRecord), (size_t)src_count, fp);
            }
            fclose(fp);
        }
    }

    /* 加载目标快照条目 */
    char tgt_filepath[512];
    snprintf(tgt_filepath, sizeof(tgt_filepath), "%s/snapshot_%d.dat",
             kvm->storage_dir, tgt_snap);
    SnapshotEntryRecord* tgt_entries = (SnapshotEntryRecord*)
        safe_calloc(KV_MAX_ENTRIES_PER_SNAPSHOT, sizeof(SnapshotEntryRecord));
    int tgt_count = 0;
    if (tgt_entries) {
        FILE* fp = fopen(tgt_filepath, "rb");
        if (fp) {
            SnapshotFileHeader hdr;
            if (fread(&hdr, sizeof(SnapshotFileHeader), 1, fp) == 1 &&
                hdr.magic == KV_FILE_MAGIC) {
                tgt_count = (int)(hdr.entry_count < KV_MAX_ENTRIES_PER_SNAPSHOT ?
                                  hdr.entry_count : KV_MAX_ENTRIES_PER_SNAPSHOT);
                fread(tgt_entries, sizeof(SnapshotEntryRecord), (size_t)tgt_count, fp);
            }
            fclose(fp);
        }
    }

    /* 构建合并条目数组：以target为基础，逐一添加源条目 */
    SnapshotEntryRecord* merged_entries = (SnapshotEntryRecord*)
        safe_calloc(KV_MAX_ENTRIES_PER_SNAPSHOT, sizeof(SnapshotEntryRecord));
    if (!merged_entries) {
        safe_free((void**)&src_entries);
        safe_free((void**)&tgt_entries);
        return -1;
    }

    /* 先将目标条目全部复制到合并结果 */
    int merged_count = tgt_count;
    if (tgt_entries && tgt_count > 0) {
        memcpy(merged_entries, tgt_entries,
               (size_t)tgt_count * sizeof(SnapshotEntryRecord));
    }

    int added_from_src = 0;
    int conflicts_resolved = 0;

    /* 遍历源条目，检查是否存在于目标分支 */
    for (int si = 0; si < src_count && merged_count < KV_MAX_ENTRIES_PER_SNAPSHOT; si++) {
        int found_in_tgt = -1;
        for (int ti = 0; ti < tgt_count; ti++) {
            if (strcmp(src_entries[si].subject, tgt_entries[ti].subject) == 0 &&
                strcmp(src_entries[si].predicate, tgt_entries[ti].predicate) == 0 &&
                strcmp(src_entries[si].object, tgt_entries[ti].object) == 0) {
                found_in_tgt = ti;
                break;
            }
        }

        if (found_in_tgt < 0) {
            /* 源独有：查找是否有相同S+P但不同O的冲突条目 */
            int sp_conflict = -1;
            for (int ti = 0; ti < tgt_count; ti++) {
                if (strcmp(src_entries[si].subject, tgt_entries[ti].subject) == 0 &&
                    strcmp(src_entries[si].predicate, tgt_entries[ti].predicate) == 0 &&
                    strcmp(src_entries[si].object, tgt_entries[ti].object) != 0) {
                    sp_conflict = ti;
                    break;
                }
            }
            if (sp_conflict >= 0) {
                /* S+P相同但O不同 → 真正的合并冲突
                 * 策略: 置信度择优，置信度接近时保留目标分支版本 */
                float src_confidence = src_entries[si].confidence;
                float tgt_confidence = tgt_entries[sp_conflict].confidence;

                /* ZSFWS-031: 冲突解决日志 */
                log_info("[分支合并冲突] <%s, %s>: "
                         "源分支值='%s'(confidence=%.3f) vs 目标分支值='%s'(confidence=%.3f)",
                         src_entries[si].subject, src_entries[si].predicate,
                         src_entries[si].object, src_confidence,
                         tgt_entries[sp_conflict].object, tgt_confidence);

                if (src_confidence > tgt_confidence * 1.5f) {
                    /* 源置信度显著更高，用源替换目标 */
                    log_info("[分支合并冲突解决] 采用源分支版本: "
                             "源置信度(%.3f) > 目标置信度(%.3f) * 1.5",
                             src_confidence, tgt_confidence);
                    merged_entries[sp_conflict] = src_entries[si];
                    merged_entries[sp_conflict].entry_id = merged_count;
                } else {
                    log_info("[分支合并冲突解决] 保留目标分支版本: "
                             "源置信度(%.3f) <= 目标置信度(%.3f) * 1.5 或目标分支优先级更高",
                             src_confidence, tgt_confidence);
                }
                /* 否则保持目标版本不变 */
                conflicts_resolved++;
            } else {
                /* 目标中没有同S+P冲突 → 安全添加 */
                merged_entries[merged_count] = src_entries[si];
                merged_entries[merged_count].entry_id = merged_count + 1;
                merged_count++;
                added_from_src++;
            }
        }
        /* 完全相同（S+P+O都一致）的条目已在目标中，跳过 */
    }

    /* 将合并结果写入新快照 */
    char merge_msg[KV_MAX_MESSAGE];
    snprintf(merge_msg, sizeof(merge_msg),
            "合并分支 '%s'->'%s': +%d条目, %d冲突解决",
            source, target, added_from_src, conflicts_resolved);

    /* 使用内部机制创建合并快照 */
    if (kvm->snapshot_count >= KV_MAX_SNAPSHOTS) {
        safe_free((void**)&src_entries);
        safe_free((void**)&tgt_entries);
        safe_free((void**)&merged_entries);
        return -1;
    }

    /* 更新当前条目缓存为合并结果 */
    if (merged_count <= KV_MAX_ENTRIES_PER_SNAPSHOT) {
        memcpy(kvm->current_entries, merged_entries,
               (size_t)merged_count * sizeof(SnapshotEntryRecord));
        kvm->current_entry_count = merged_count;
    }

    int new_id = kv_create_snapshot(kvm, merge_msg);
    if (new_id < 0) {
        safe_free((void**)&src_entries);
        safe_free((void**)&tgt_entries);
        safe_free((void**)&merged_entries);
        return -1;
    }

    /* 标记新快照归属目标分支 */
    for (int i = 0; i < kvm->snapshot_count; i++) {
        if (kvm->snapshots[i].snapshot_id == new_id) {
            strncpy(kvm->snapshots[i].branch, target,
                    sizeof(kvm->snapshots[i].branch) - 1);
            kvm->snapshots[i].parent_id = tgt_snap;
            kvm->snapshots[i].entry_count = (size_t)merged_count;
            break;
        }
    }

    /* H-002修复: 将合并结果序列化写入知识库数据文件
     * 使用与快照文件相同的二进制格式(SnapshotFileHeader + 条目数组)，
     * 写入storage_dir下的knowledge_merged_current.dat，
     * 确保合并结果在程序重启后可恢复。 */
    {
        char merged_data_path[512];
        snprintf(merged_data_path, sizeof(merged_data_path),
                 "%s/knowledge_merged_current.dat", kvm->storage_dir);
        FILE* merged_fp = fopen(merged_data_path, "wb");
        if (merged_fp) {
            SnapshotFileHeader merged_header;
            memset(&merged_header, 0, sizeof(SnapshotFileHeader));
            merged_header.magic = KV_FILE_MAGIC;
            merged_header.version = KV_FILE_VERSION;
            merged_header.snapshot_id = new_id;
            snprintf(merged_header.message, KV_MAX_MESSAGE, "%s", merge_msg);
            merged_header.created_at = time(NULL);
            merged_header.entry_count = (size_t)merged_count;
            merged_header.parent_id = tgt_snap;
            snprintf(merged_header.branch, KV_MAX_BRANCH_NAME, "%s", target);
            merged_header.is_checkpoint = 0;

            size_t header_ok = fwrite(&merged_header, sizeof(SnapshotFileHeader), 1, merged_fp);
            size_t entries_ok = 1;
            if (merged_count > 0) {
                entries_ok = fwrite(merged_entries, sizeof(SnapshotEntryRecord),
                                    (size_t)merged_count, merged_fp);
            }

            if (header_ok != 1 || entries_ok != (size_t)merged_count) {
                log_error("[分支合并] 写入合并数据文件失败: %s (header=%zu, entries=%zu/%d)",
                          merged_data_path, header_ok, entries_ok, merged_count);
            } else {
                log_info("[分支合并] 合并结果已持久化写入: %s (共%d条目, 分支=%s)",
                         merged_data_path, merged_count, target);
            }
            fclose(merged_fp);
        } else {
            log_error("[分支合并] 无法创建合并数据文件: %s (errno=%d)",
                      merged_data_path, errno);
        }
    }

    safe_free((void**)&src_entries);
    safe_free((void**)&tgt_entries);
    safe_free((void**)&merged_entries);
    return 0;
}

int kv_list_branches(const KnowledgeVersionManager* kvm, char* branches, int max_count) {
    if (!kvm || !branches) return 0;
    int pos = 0;
    for (int i = 0; i < kvm->branch_count && i < max_count; i++) {
        pos += snprintf(branches + pos, 128, "%s\n", kvm->branches[i]);
    }
    return kvm->branch_count;
}

int kv_trace_entry(const KnowledgeVersionManager* kvm, int entry_id, char* history, size_t max_len) {
    if (!kvm || !history) return -1;

    const KnowledgeEntryHistory* found = NULL;
    for (int i = 0; i < kvm->history_count; i++) {
        if (kvm->entry_histories[i].entry_id == entry_id) {
            found = &kvm->entry_histories[i];
            break;
        }
    }

    if (found) {
        int pos = snprintf(history, max_len,
            "条目 #%d 版本历史:\n"
            "  主体: %s\n  谓词: %s\n"
            "  首次出现: %s"
            "  最后修改: %s"
            "  修改次数: %d\n"
            "  涉及快照: ",
            found->entry_id, found->subject, found->predicate,
            ctime(&found->first_seen), ctime(&found->last_modified),
            found->modification_count);

        for (int i = 0; i < found->history_count && (size_t)pos < max_len; i++) {
            pos += snprintf(history + pos, max_len - (size_t)pos,
                           "#%d ", found->snapshot_ids[i]);
        }
        snprintf(history + pos, max_len - (size_t)pos, "\n");
    } else {
        snprintf(history, max_len, "条目#%d 版本历史追踪中", entry_id);
    }
    return 0;
}

int kv_set_auto_snapshot(KnowledgeVersionManager* kvm, int interval_minutes, int max_snapshots) {
    if (!kvm) return -1;
    kvm->auto_interval_min = interval_minutes;
    kvm->auto_max_snapshots = max_snapshots;
    return 0;
}

/* ===== P4.4 新增功能实现 ===== */

int kv_get_snapshot_entry_count(const KnowledgeVersionManager* kvm, int snapshot_id) {
    if (!kvm) return -1;

    for (int i = 0; i < kvm->snapshot_count; i++) {
        if (kvm->snapshots[i].snapshot_id == snapshot_id) {
            return (int)kvm->snapshots[i].entry_count;
        }
    }
    return -1;
}

int kv_compare_branch_snapshots(const KnowledgeVersionManager* kvm,
                                const char* branch_a, int snapshot_a_id,
                                const char* branch_b, int snapshot_b_id,
                                KnowledgeDiff* diff) {
    if (!kvm || !branch_a || !branch_b || !diff) return -1;

    int branch_a_found = 0, branch_b_found = 0;
    for (int i = 0; i < kvm->branch_count; i++) {
        if (strcmp(kvm->branches[i], branch_a) == 0) branch_a_found = 1;
        if (strcmp(kvm->branches[i], branch_b) == 0) branch_b_found = 1;
    }
    if (!branch_a_found || !branch_b_found) return -1;

    int actual_from = snapshot_a_id;
    int actual_to = snapshot_b_id;

    /* 如果ID为0，使用该分支的最新快照 */
    if (actual_from <= 0) {
        for (int i = kvm->snapshot_count - 1; i >= 0; i--) {
            if (strcmp(kvm->snapshots[i].branch, branch_a) == 0) {
                actual_from = kvm->snapshots[i].snapshot_id;
                break;
            }
        }
    }
    if (actual_to <= 0) {
        for (int i = kvm->snapshot_count - 1; i >= 0; i--) {
            if (strcmp(kvm->snapshots[i].branch, branch_b) == 0) {
                actual_to = kvm->snapshots[i].snapshot_id;
                break;
            }
        }
    }

    if (actual_from <= 0 || actual_to <= 0) return -1;

    /* 验证快照属于指定分支 */
    int from_ok = 0, to_ok = 0;
    for (int i = 0; i < kvm->snapshot_count; i++) {
        if (kvm->snapshots[i].snapshot_id == actual_from &&
            strcmp(kvm->snapshots[i].branch, branch_a) == 0)
            from_ok = 1;
        if (kvm->snapshots[i].snapshot_id == actual_to &&
            strcmp(kvm->snapshots[i].branch, branch_b) == 0)
            to_ok = 1;
    }
    if (!from_ok || !to_ok) return -1;

    return kv_diff_snapshots(kvm, actual_from, actual_to, diff);
}

int kv_get_entry_history(const KnowledgeVersionManager* kvm, int entry_id,
                         KnowledgeEntryHistory* history) {
    if (!kvm || !history) return -1;

    for (int i = 0; i < kvm->history_count; i++) {
        if (kvm->entry_histories[i].entry_id == entry_id) {
            memcpy(history, &kvm->entry_histories[i], sizeof(KnowledgeEntryHistory));
            return 0;
        }
    }
    return -1;
}

int kv_diff_export_report(const KnowledgeVersionManager* kvm, int from_id, int to_id,
                          char* report, size_t max_len) {
    if (!kvm || !report || max_len < 128) return -1;

    KnowledgeDiff diff;
    if (kv_diff_snapshots(kvm, from_id, to_id, &diff) != 0) {
        snprintf(report, max_len, "无法比较快照 #%d 和 #%d\n", from_id, to_id);
        return -1;
    }

    KnowledgeSnapshot from_snap, to_snap;
    char from_info[128] = "N/A", to_info[128] = "N/A";
    if (kv_get_snapshot(kvm, from_id, &from_snap) == 0) {
        snprintf(from_info, sizeof(from_info), "#%d %s",
                from_snap.snapshot_id, from_snap.message);
    }
    if (kv_get_snapshot(kvm, to_id, &to_snap) == 0) {
        snprintf(to_info, sizeof(to_info), "#%d %s",
                to_snap.snapshot_id, to_snap.message);
    }

    int pos = snprintf(report, max_len,
        "========================================\n"
        "  知识版本差异报告\n"
        "========================================\n"
        "  源快照: %s\n"
        "  目标快照: %s\n"
        "  分析时间: %s"
        "----------------------------------------\n"
        "  新增条目: %zu\n"
        "  删除条目: %zu\n"
        "  修改条目: %zu\n"
        "  变更总计: %d\n"
        "----------------------------------------\n",
        from_info, to_info, ctime(&diff.diff_time),
        diff.added_count, diff.removed_count, diff.modified_count,
        diff.detail_count);

    if (diff.added_count > 0 && (size_t)pos < max_len) {
        pos += snprintf(report + pos, max_len - (size_t)pos, "\n  --- 新增条目 ---\n");
        for (int i = 0; i < diff.detail_count && (size_t)pos < max_len; i++) {
            if (diff.details[i].change_type == KV_DIFF_ADD) {
                pos += snprintf(report + pos, max_len - (size_t)pos,
                    "  [+] #%d <%s, %s, %s> conf=%.2f\n",
                    diff.details[i].entry_id,
                    diff.details[i].subject,
                    diff.details[i].predicate,
                    diff.details[i].object_after,
                    diff.details[i].confidence_after);
            }
        }
    }

    if (diff.removed_count > 0 && (size_t)pos < max_len) {
        pos += snprintf(report + pos, max_len - (size_t)pos, "\n  --- 删除条目 ---\n");
        for (int i = 0; i < diff.detail_count && (size_t)pos < max_len; i++) {
            if (diff.details[i].change_type == KV_DIFF_REMOVE) {
                pos += snprintf(report + pos, max_len - (size_t)pos,
                    "  [-] #%d <%s, %s, %s>\n",
                    diff.details[i].entry_id,
                    diff.details[i].subject,
                    diff.details[i].predicate,
                    diff.details[i].object_before);
            }
        }
    }

    if (diff.modified_count > 0 && (size_t)pos < max_len) {
        pos += snprintf(report + pos, max_len - (size_t)pos, "\n  --- 修改条目 ---\n");
        for (int i = 0; i < diff.detail_count && (size_t)pos < max_len; i++) {
            if (diff.details[i].change_type == KV_DIFF_MODIFY) {
                pos += snprintf(report + pos, max_len - (size_t)pos,
                    "  [~] #%d <%s, %s>\n"
                    "       之前: %s (conf=%.2f)\n"
                    "       之后: %s (conf=%.2f)\n",
                    diff.details[i].entry_id,
                    diff.details[i].subject,
                    diff.details[i].predicate,
                    diff.details[i].object_before,
                    diff.details[i].confidence_before,
                    diff.details[i].object_after,
                    diff.details[i].confidence_after);
            }
        }
    }

    if ((size_t)pos < max_len) {
        snprintf(report + pos, max_len - (size_t)pos,
            "========================================\n");
    }

    return 0;
}

/* 快照按创建时间升序排序的比较函数 */
static int compare_snapshots_by_time(const void* a, const void* b) {
    const KnowledgeSnapshot* snap_a = (const KnowledgeSnapshot*)a;
    const KnowledgeSnapshot* snap_b = (const KnowledgeSnapshot*)b;
    if (snap_a->created_at < snap_b->created_at) return -1;
    if (snap_a->created_at > snap_b->created_at) return 1;
    return 0;
}

int kv_cleanup_old_snapshots(KnowledgeVersionManager* kvm, int keep_count) {
    if (!kvm || keep_count <= 0) return -1;

    if (kvm->snapshot_count <= keep_count) return 0;

    /* 按创建时间升序排序，确保删除最旧的快照 */
    qsort(kvm->snapshots, kvm->snapshot_count, sizeof(KnowledgeSnapshot), compare_snapshots_by_time);

    int to_remove = kvm->snapshot_count - keep_count;
    int removed = 0;
    char filepath[512];

    /* 遍历排序后的快照，移除最旧的，但保护检查点不被自动删除 */
    for (int i = 0; i < kvm->snapshot_count && removed < to_remove; i++) {
        /* 标记为检查点的快照不自动删除 */
        if (kvm->snapshots[i].is_checkpoint) {
            continue;
        }
        int snap_id = kvm->snapshots[i].snapshot_id;
        snprintf(filepath, sizeof(filepath), "%s/snapshot_%d.dat",
                 kvm->storage_dir, snap_id);
        remove(filepath);
        memset(&kvm->snapshots[i], 0, sizeof(KnowledgeSnapshot));
        removed++;
    }

    /* 紧凑数组：将已清空的快照移除，保留有效快照 */
    int write_idx = 0;
    for (int read_idx = 0; read_idx < kvm->snapshot_count; read_idx++) {
        if (kvm->snapshots[read_idx].snapshot_id != 0) {
            if (write_idx != read_idx) {
                memcpy(&kvm->snapshots[write_idx], &kvm->snapshots[read_idx], sizeof(KnowledgeSnapshot));
            }
            write_idx++;
        }
    }
    kvm->snapshot_count = write_idx;

    if (kvm->current_snapshot > 0 && kvm->snapshot_count > 0) {
        kvm->current_snapshot = kvm->snapshots[kvm->snapshot_count - 1].snapshot_id;
    }

    return removed;
}

/* 暴露当前条目快照供知识库更新时使用，并自动追踪条目历史 */
int kv_update_current_entries(KnowledgeVersionManager* kvm,
                               SnapshotEntryRecord* entries, int count) {
    if (!kvm || !entries || count <= 0) return -1;
    if (count > KV_MAX_ENTRIES_PER_SNAPSHOT) {
        count = KV_MAX_ENTRIES_PER_SNAPSHOT;
    }

    /* 对每个更新条目自动追踪历史 */
    for (int i = 0; i < count; i++) {
        int entry_id = entries[i].entry_id;
        int found = -1;
        for (int h = 0; h < kvm->history_count; h++) {
            if (kvm->entry_histories[h].entry_id == entry_id) {
                found = h;
                break;
            }
        }
        if (found >= 0) {
            /* 已有历史：追加新快照ID */
            KnowledgeEntryHistory* hist = &kvm->entry_histories[found];
            if (hist->history_count < KV_MAX_HISTORY) {
                hist->snapshot_ids[hist->history_count++] = kvm->current_snapshot;
            } else {
                /* 滑动窗口：移除最旧，追加最新 */
                memmove(hist->snapshot_ids, hist->snapshot_ids + 1,
                        (size_t)(KV_MAX_HISTORY - 1) * sizeof(int));
                hist->snapshot_ids[KV_MAX_HISTORY - 1] = kvm->current_snapshot;
            }
            hist->last_modified = time(NULL);
        } else if (kvm->history_count < KV_MAX_HISTORY) {
            /* 新建历史追踪 */
            KnowledgeEntryHistory* hist = &kvm->entry_histories[kvm->history_count++];
            memset(hist, 0, sizeof(KnowledgeEntryHistory));
            hist->entry_id = entry_id;
            hist->snapshot_ids[0] = kvm->current_snapshot;
            hist->history_count = 1;
            hist->first_seen = time(NULL);
            hist->last_modified = hist->first_seen;
        }
    }

    memcpy(kvm->current_entries, entries, (size_t)count * sizeof(SnapshotEntryRecord));
    kvm->current_entry_count = count;
    return 0;
}

/* ============================================================================
 * ZSFWS-031: 语义相似度合并策略
 *
 * 对两个快照中的实体进行语义级合并，根据相似度决定合并策略：
 *   - 相似度 >= 0.80: 自动合并（高置信度实体替换低置信度）
 *   - 相似度 0.50 ~ 0.80: 标记为待审核（记录到冲突日志）
 *   - 相似度 < 0.50: 视为不同实体，保留双方
 *
 * 策略背后的设计思想：
 *   1. 高相似度（>=0.8）的实体极可能是同一知识的不同表述，
 *      自动合并可减少知识库冗余而不损失信息。
 *   2. 中等相似度（0.5-0.8）的实体可能是相关但不完全等价的知识，
 *      需要人工或后续推理确认，标记为待审核。
 *   3. 低相似度（<0.5）的实体为不同的知识条目，各自保留。
 *
 * 注意：此函数是 kv_diff_snapshots 的补充而非替代。
 *      kv_diff_snapshots 保证精确版本追溯（基于(S,P)精确匹配），
 *      kv_semantic_merge_entities 提供语义级冗余消除能力。
 *
 * @param entries_a      第一个条目数组
 * @param count_a        第一个数组大小
 * @param entries_b      第二个条目数组  
 * @param count_b        第二个数组大小
 * @param merged_entries 合并结果输出（调用者分配，容量需 >= count_a + count_b）
 * @param max_merged     合并结果最大容量
 * @param merged_count   输出：实际合并后条目数
 * @param log_buffer     冲突日志输出缓冲区（可NULL）
 * @param log_max        日志缓冲区大小
 * @return 0成功, -1失败
 * ============================================================================ */
int kv_semantic_merge_entities(
    const SnapshotEntryRecord* entries_a, int count_a,
    const SnapshotEntryRecord* entries_b, int count_b,
    SnapshotEntryRecord* merged_entries, int max_merged,
    int* merged_count,
    char* log_buffer, size_t log_max)
{
    if (!entries_a || !entries_b || !merged_entries || !merged_count) return -1;
    if (count_a < 0 || count_b < 0 || max_merged < (count_a + count_b)) return -1;

    int out_count = 0;
    int auto_merged = 0;
    int pending_review = 0;
    int log_pos = 0;

    #define SEMANTIC_HIGH_THRESHOLD 0.80f
    #define SEMANTIC_MID_THRESHOLD 0.50f

    /* 使用标记数组追踪b中已合并的条目 */
    int* b_merged = (int*)safe_calloc((size_t)count_b, sizeof(int));
    if (!b_merged && count_b > 0) return -1;

    /* 第1阶段：将a中所有条目与b比较，相似度 >= 0.8 自动合并 */
    for (int ai = 0; ai < count_a && out_count < max_merged; ai++) {
        float best_sim = 0.0f;
        int best_idx = -1;

        for (int bi = 0; bi < count_b; bi++) {
            if (!b_merged) break;
            if (b_merged[bi]) continue;
            float sim = compute_entity_similarity(&entries_a[ai], &entries_b[bi]);
            if (sim > best_sim) {
                best_sim = sim;
                best_idx = bi;
            }
        }

        if (best_sim >= SEMANTIC_HIGH_THRESHOLD && best_idx >= 0) {
            /* 高相似度自动合并：取置信度更高的 */
            if (entries_a[ai].confidence >= entries_b[best_idx].confidence) {
                merged_entries[out_count] = entries_a[ai];
            } else {
                merged_entries[out_count] = entries_b[best_idx];
            }
            merged_entries[out_count].entry_id = out_count + 1;
            if (b_merged) b_merged[best_idx] = 1;
            auto_merged++;
            out_count++;
        } else if (best_sim >= SEMANTIC_MID_THRESHOLD && best_idx >= 0) {
            /* 中等相似度：两者都保留但标记为待审核 */
            merged_entries[out_count] = entries_a[ai];
            merged_entries[out_count].entry_id = out_count + 1;
            out_count++;

            if (out_count < max_merged && b_merged) {
                merged_entries[out_count] = entries_b[best_idx];
                merged_entries[out_count].entry_id = out_count + 1;
                b_merged[best_idx] = 1;
                out_count++;
                pending_review++;

                /* 记录待审核冲突到日志 */
                if (log_buffer && (size_t)log_pos < log_max) {
                    log_pos += snprintf(log_buffer + log_pos, log_max - (size_t)log_pos,
                        "[待审核] 相似度=%.3f: <%s,%s,%s>(conf=%.2f) vs <%s,%s,%s>(conf=%.2f)\n",
                        best_sim,
                        entries_a[ai].subject, entries_a[ai].predicate, entries_a[ai].object,
                        entries_a[ai].confidence,
                        entries_b[best_idx].subject, entries_b[best_idx].predicate,
                        entries_b[best_idx].object,
                        entries_b[best_idx].confidence);
                }
            } else {
                /* 空间不足，仅保留a条目 */
            }
        } else {
            /* 低相似度：视为不同实体，保留a条目 */
            merged_entries[out_count] = entries_a[ai];
            merged_entries[out_count].entry_id = out_count + 1;
            out_count++;
        }
    }

    /* 第2阶段：将b中剩余未合并的条目追加到结果 */
    for (int bi = 0; bi < count_b && out_count < max_merged; bi++) {
        if (b_merged && b_merged[bi]) continue;
        merged_entries[out_count] = entries_b[bi];
        merged_entries[out_count].entry_id = out_count + 1;
        out_count++;
    }

    /* 记录合并摘要 */
    log_info("[语义合并] 完成: a=%d条目, b=%d条目 → 合并后=%d条目, "
             "自动合并=%d, 待审核=%d (高阈值=%.2f, 中阈值=%.2f)",
             count_a, count_b, out_count, auto_merged, pending_review,
             SEMANTIC_HIGH_THRESHOLD, SEMANTIC_MID_THRESHOLD);

    safe_free((void**)&b_merged);
    *merged_count = out_count;
    return 0;

    #undef SEMANTIC_HIGH_THRESHOLD
    #undef SEMANTIC_MID_THRESHOLD
}

/* ============================================================================
 * H-003: 从knowledge_snapshot合并的标签和保存/加载功能
 * ============================================================================ */

/* 新增的save/load文件魔数（区别于SnapshotFileHeader的KV_FILE_MAGIC） */
#define KV_SAVE_FILE_MAGIC  0x4B564D47  /* "KVMG" - KnowledgeVersion ManaGer */
#define KV_SAVE_FILE_VERSION 1

/* save/load用文件头 */
typedef struct {
    uint32_t magic;
    uint32_t file_version;
    int32_t snapshot_count;
    int32_t current_snapshot;
    int32_t branch_count;
    int32_t auto_interval_min;
    int32_t auto_max_snapshots;
    int32_t next_snapshot_id;
    int32_t history_count;
    int32_t current_entry_count;
    int32_t reserved[4];
    /* 变长数据: storage_dir(512) + current_branch(64) + branch_names(16*64) + KnowledgeSnapshot数组 + entry_histories */
} KVMFileHeader;

/* 为指定快照添加标签 */
int kv_add_tag(KnowledgeVersionManager* kvm, int snapshot_id, const char* tag) {
    if (!kvm || !tag) return -1;
    for (int i = 0; i < kvm->snapshot_count; i++) {
        if (kvm->snapshots[i].snapshot_id == snapshot_id) {
            snprintf(kvm->snapshots[i].tag, sizeof(kvm->snapshots[i].tag), "%s", tag);
            return 0;
        }
    }
    return -1;
}

/* 将整个版本控制器序列化保存到文件 */
int kv_save(const KnowledgeVersionManager* kvm, const char* filepath) {
    if (!kvm || !filepath) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        log_error("[知识版本管理] 无法打开文件写入: %s", filepath);
        return -1;
    }

    KVMFileHeader header;
    memset(&header, 0, sizeof(KVMFileHeader));
    header.magic = KV_SAVE_FILE_MAGIC;
    header.file_version = KV_SAVE_FILE_VERSION;
    header.snapshot_count = (int32_t)kvm->snapshot_count;
    header.current_snapshot = (int32_t)kvm->current_snapshot;
    header.branch_count = (int32_t)kvm->branch_count;
    header.auto_interval_min = (int32_t)kvm->auto_interval_min;
    header.auto_max_snapshots = (int32_t)kvm->auto_max_snapshots;
    header.next_snapshot_id = (int32_t)kvm->next_snapshot_id;
    header.history_count = (int32_t)kvm->history_count;
    header.current_entry_count = (int32_t)kvm->current_entry_count;

    if (fwrite(&header, sizeof(KVMFileHeader), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    /* 写入storage_dir(512) */
    fwrite(kvm->storage_dir, sizeof(kvm->storage_dir), 1, fp);
    /* 写入current_branch(64) */
    fwrite(kvm->current_branch, sizeof(kvm->current_branch), 1, fp);
    /* 写入分支名称数组 */
    fwrite(kvm->branches, sizeof(kvm->branches), 1, fp);
    /* 写入快照数组 */
    fwrite(kvm->snapshots, sizeof(kvm->snapshots), 1, fp);
    /* 写入条目历史数组 */
    fwrite(kvm->entry_histories, sizeof(kvm->entry_histories), 1, fp);
    /* 写入当前条目数组 */
    fwrite(kvm->current_entries, sizeof(SnapshotEntryRecord), (size_t)kvm->current_entry_count, fp);

    fclose(fp);
    log_info("[知识版本管理] 已保存版本控制状态到: %s (快照数=%d, 分支数=%d)",
             filepath, kvm->snapshot_count, kvm->branch_count);
    return 0;
}

/* 从文件反序列化加载版本控制器 */
KnowledgeVersionManager* kv_load(const char* filepath) {
    if (!filepath) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        log_warning("[知识版本管理] 版本文件不存在: %s", filepath);
        return NULL;
    }

    KVMFileHeader header;
    if (fread(&header, sizeof(KVMFileHeader), 1, fp) != 1 ||
        header.magic != KV_SAVE_FILE_MAGIC) {
        log_error("[知识版本管理] 文件格式无效: %s (魔数不匹配)", filepath);
        fclose(fp);
        return NULL;
    }

    KnowledgeVersionManager* kvm = (KnowledgeVersionManager*)safe_calloc(1, sizeof(KnowledgeVersionManager));
    if (!kvm) { fclose(fp); return NULL; }

    kvm->snapshot_count = header.snapshot_count;
    kvm->current_snapshot = header.current_snapshot;
    kvm->branch_count = header.branch_count;
    kvm->auto_interval_min = header.auto_interval_min;
    kvm->auto_max_snapshots = header.auto_max_snapshots;
    kvm->next_snapshot_id = header.next_snapshot_id;
    kvm->history_count = header.history_count;
    kvm->current_entry_count = header.current_entry_count;
    kvm->last_auto_snapshot = 0;

    /* 读取storage_dir */
    fread(kvm->storage_dir, sizeof(kvm->storage_dir), 1, fp);
    /* 读取current_branch */
    fread(kvm->current_branch, sizeof(kvm->current_branch), 1, fp);
    /* 读取分支名称数组 */
    fread(kvm->branches, sizeof(kvm->branches), 1, fp);
    /* 读取快照数组 */
    fread(kvm->snapshots, sizeof(kvm->snapshots), 1, fp);
    /* 读取条目历史数组 */
    fread(kvm->entry_histories, sizeof(kvm->entry_histories), 1, fp);
    /* 读取当前条目数组 */
    if (kvm->current_entry_count > 0 && kvm->current_entry_count <= KV_MAX_ENTRIES_PER_SNAPSHOT) {
        fread(kvm->current_entries, sizeof(SnapshotEntryRecord), (size_t)kvm->current_entry_count, fp);
    }

    fclose(fp);
    log_info("[知识版本管理] 已从文件加载版本控制状态: %s (快照数=%d, 分支数=%d)",
             filepath, kvm->snapshot_count, kvm->branch_count);
    return kvm;
}

/* 获取指定分支的快照历史（按创建时间排序） */
int kv_get_history(const KnowledgeVersionManager* kvm, const char* branch_name,
                   KnowledgeSnapshot* out, int max_count) {
    if (!kvm || !out || max_count <= 0) return -1;

    /* 先收集匹配分支的快照到临时数组 */
    KnowledgeSnapshot temp[KV_MAX_SNAPSHOTS];
    int temp_count = 0;

    for (int i = 0; i < kvm->snapshot_count && temp_count < KV_MAX_SNAPSHOTS; i++) {
        if (!branch_name || strcmp(kvm->snapshots[i].branch, branch_name) == 0) {
            memcpy(&temp[temp_count], &kvm->snapshots[i], sizeof(KnowledgeSnapshot));
            temp_count++;
        }
    }

    /* 按创建时间升序排序（冒泡排序，快照数量通常不大） */
    for (int i = 0; i < temp_count - 1; i++) {
        for (int j = 0; j < temp_count - i - 1; j++) {
            if (temp[j].created_at > temp[j + 1].created_at) {
                KnowledgeSnapshot tmp = temp[j];
                temp[j] = temp[j + 1];
                temp[j + 1] = tmp;
            }
        }
    }

    /* 复制结果，不超过max_count */
    int result_count = temp_count < max_count ? temp_count : max_count;
    memcpy(out, temp, (size_t)result_count * sizeof(KnowledgeSnapshot));
    return result_count;
}
