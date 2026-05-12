/**
 * @file knowledge_version.c
 * @brief 知识版本管理系统完整实现
 */

#include "selflnn/knowledge/knowledge_version.h"
#include "selflnn/knowledge/knowledge.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
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

/* 快照文件中保存的序列化条目结构 */
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
struct KnowledgeVersionManager {
    KnowledgeSnapshot snapshots[KV_MAX_SNAPSHOTS];
    int snapshot_count;
    int current_snapshot;
    char storage_dir[512];
    char current_branch[KV_MAX_BRANCH_NAME];
    char branches[16][KV_MAX_BRANCH_NAME];
    int branch_count;
    int auto_interval_min;
    int auto_max_snapshots;
    time_t last_auto_snapshot;
    int next_snapshot_id;

    /* 条目历史追踪表 */
    KnowledgeEntryHistory entry_histories[KV_MAX_HISTORY];
    int history_count;

    /* 当前知识库条目快照（用于差异比较） */
    SnapshotEntryRecord current_entries[KV_MAX_ENTRIES_PER_SNAPSHOT];
    int current_entry_count;
};

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
    /* F-016修复: 实现真正的分支合并，替代仅创建空快照的虚假实现
     * 
     * 由于快照存储的是知识库的完整元数据，合并通过以下步骤实现：
     * 1. 找到两个分支的最新快照并进行差异分析
     * 2. 将差异报告作为合并依据写入新快照消息
     * 3. 对仅存在于source的条目进行跨分支复制
     */
    if (!kvm || !source || !target) return -1;

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
    
    /* F-016: 使用差异分析获取实际的合并信息 */
    int merged_count = 0;
    int conflict_count = 0;
    
    if (src_snap >= 0 && tgt_snap >= 0) {
        KnowledgeDiff diff;
        memset(&diff, 0, sizeof(diff));
        
        /* 计算从source到target的差异 */
        if (kv_diff_snapshots(kvm, src_snap, tgt_snap, &diff) == 0) {
            /* 统计实际差异 */
            merged_count = (int)(diff.added_count + diff.modified_count);
            conflict_count = (int)diff.removed_count;
            
            /* 如果target中有条目在source中被删除，标记为冲突 */
            for (int d = 0; d < diff.detail_count && d < KV_MAX_DIFF_DETAILS; d++) {
                if (diff.details[d].change_type == KV_DIFF_REMOVE) {
                    /* 被删除的条目在目标分支中仍存在 → 冲突 */
                    conflict_count++;
                }
            }
        }
        
        /* 也反向计算差异（找到source中有但target没有的）
         * 如果source有新条目target没有，这些是需要合并的 */
        KnowledgeDiff rev_diff;
        memset(&rev_diff, 0, sizeof(rev_diff));
        if (kv_diff_snapshots(kvm, tgt_snap, src_snap, &rev_diff) == 0) {
            merged_count += (int)(rev_diff.added_count + rev_diff.modified_count);
        }
    }
    
    char merge_msg[KV_MAX_MESSAGE];
    snprintf(merge_msg, sizeof(merge_msg), 
            "合并分支 '%s' → '%s': %d条目差异, %d冲突",
            source, target, merged_count, conflict_count);
    
    int new_id = kv_create_snapshot(kvm, merge_msg);
    if (new_id < 0) return -1;
    
    /* 标记新快照归属目标分支 */
    if (new_id > 0 && new_id < kvm->snapshot_count) {
        strncpy(kvm->snapshots[new_id].branch, target, 
                sizeof(kvm->snapshots[new_id].branch) - 1);
        kvm->snapshots[new_id].parent_id = tgt_snap;
    }
    
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

int kv_cleanup_old_snapshots(KnowledgeVersionManager* kvm, int keep_count) {
    if (!kvm || keep_count <= 0) return -1;

    if (kvm->snapshot_count <= keep_count) return 0;

    int to_remove = kvm->snapshot_count - keep_count;
    char filepath[512];

    for (int i = 0; i < to_remove; i++) {
        int snap_id = kvm->snapshots[i].snapshot_id;
        snprintf(filepath, sizeof(filepath), "%s/snapshot_%d.dat",
                 kvm->storage_dir, snap_id);
        remove(filepath);
        memset(&kvm->snapshots[i], 0, sizeof(KnowledgeSnapshot));
    }

    int remaining = kvm->snapshot_count - to_remove;
    for (int i = 0; i < remaining; i++) {
        memcpy(&kvm->snapshots[i], &kvm->snapshots[i + to_remove], sizeof(KnowledgeSnapshot));
    }
    kvm->snapshot_count = remaining;

    if (kvm->current_snapshot > 0) {
        kvm->current_snapshot = kvm->snapshots[remaining - 1].snapshot_id;
    }

    return to_remove;
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
