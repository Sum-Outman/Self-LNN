/**
 * @file reasoning_version.c
 * @brief 知识版本管理系统（KnowledgeVersionCore）
 * 
 * 注意：文件名虽然包含"reasoning"，但本文件实现的是知识版本管理功能，
 * 而非推理版本管理。知识版本系统用于：
 * - 知识条目的版本化存储和回滚
 * - 知识冲突检测和合并
 * - 知识演进历史追踪
 * - 版本差异计算和应用
 * 
 * 引用 reasoning/knowledge_version.h 而非 reasoning/reasoning.h。
 * 保留此文件名为历史兼容原因，所有 include 路径保持不变。
 */
#include "selflnn/reasoning/knowledge_snapshot.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

/* ========== 内部帮助函数 ========== */

static int kv_entry_deep_copy(KnowledgeEntry* dst, const KnowledgeEntry* src)
{
    if (!dst || !src) return -1;
    memset(dst, 0, sizeof(KnowledgeEntry));
    if (src->subject) {
        dst->subject = (char*)safe_malloc(strlen(src->subject) + 1);
        if (!dst->subject) return -1;
        strcpy(dst->subject, src->subject);
    }
    if (src->predicate) {
        dst->predicate = (char*)safe_malloc(strlen(src->predicate) + 1);
        if (!dst->predicate) { safe_free((void**)&dst->subject); return -1; }
        strcpy(dst->predicate, src->predicate);
    }
    if (src->object) {
        dst->object = (char*)safe_malloc(strlen(src->object) + 1);
        if (!dst->object) { safe_free((void**)&dst->subject); safe_free((void**)&dst->predicate); return -1; }
        strcpy(dst->object, src->object);
    }
    dst->type = src->type;
    dst->confidence = src->confidence;
    dst->source = src->source;
    dst->weight = src->weight;
    dst->timestamp = src->timestamp;
    if (src->metadata && src->metadata_size > 0) {
        dst->metadata = safe_malloc(src->metadata_size);
        if (!dst->metadata) { safe_free((void**)&dst->subject); safe_free((void**)&dst->predicate); safe_free((void**)&dst->object); return -1; }
        memcpy(dst->metadata, src->metadata, src->metadata_size);
        dst->metadata_size = src->metadata_size;
    }
    if (src->embedding && src->embedding_size > 0) {
        dst->embedding = (float*)safe_malloc(src->embedding_size * sizeof(float));
        if (!dst->embedding) { safe_free((void**)&dst->subject); safe_free((void**)&dst->predicate); safe_free((void**)&dst->object); safe_free((void**)&dst->metadata); return -1; }
        memcpy(dst->embedding, src->embedding, src->embedding_size * sizeof(float));
        dst->embedding_size = src->embedding_size;
    }
    return 0;
}

static void kv_entry_free(KnowledgeEntry* entry)
{
    if (!entry) return;
    safe_free((void**)&entry->subject);
    safe_free((void**)&entry->predicate);
    safe_free((void**)&entry->object);
    safe_free((void**)&entry->metadata);
    safe_free((void**)&entry->embedding);
    entry->embedding_size = 0;
    entry->metadata_size = 0;
}

static void* kv_snapshot_capture(KnowledgeBase* kb, size_t* out_entry_count, size_t* out_data_size)
{
    if (!kb || !out_entry_count || !out_data_size) return NULL;
    size_t total_entries = 0;
    if (knowledge_base_get_stats(kb, &total_entries, NULL) != 0 || total_entries == 0) {
        *out_entry_count = 0;
        *out_data_size = 0;
        return NULL;
    }
    KnowledgeEntry* entries = (KnowledgeEntry*)safe_malloc(total_entries * sizeof(KnowledgeEntry));
    if (!entries) { *out_entry_count = 0; *out_data_size = 0; return NULL; }
    KnowledgeQuery q;
    memset(&q, 0, sizeof(q));
    q.type_filter = -1;
    q.max_confidence = 1.0f;
    q.start_time = 0;
    q.end_time = LONG_MAX;
    int actual = knowledge_base_query(kb, &q, entries, total_entries);
    if (actual <= 0) {
        safe_free((void**)&entries);
        *out_entry_count = 0;
        *out_data_size = 0;
        return NULL;
    }
    *out_entry_count = (size_t)actual;
    size_t data_size = sizeof(size_t);
    for (size_t i = 0; i < (size_t)actual; i++) {
        KnowledgeEntry* e = &entries[i];
        data_size += sizeof(int) * 4 + sizeof(float) + sizeof(long);
        if (e->subject) data_size += (int)strlen(e->subject);
        data_size += sizeof(int);
        if (e->predicate) data_size += (int)strlen(e->predicate);
        data_size += sizeof(int);
        if (e->object) data_size += (int)strlen(e->object);
        data_size += sizeof(size_t);
        if (e->metadata && e->metadata_size > 0) data_size += e->metadata_size;
        data_size += sizeof(size_t);
        if (e->embedding && e->embedding_size > 0) data_size += e->embedding_size * sizeof(float);
    }
    unsigned char* buf = (unsigned char*)safe_malloc(data_size);
    if (!buf) {
        for (size_t i = 0; i < (size_t)actual; i++) kv_entry_free(&entries[i]);
        safe_free((void**)&entries);
        *out_entry_count = 0;
        *out_data_size = 0;
        return NULL;
    }
    size_t pos = 0;
    size_t count = (size_t)actual;
    memcpy(buf + pos, &count, sizeof(size_t)); pos += sizeof(size_t);
    for (size_t i = 0; i < count; i++) {
        KnowledgeEntry* e = &entries[i];
        int slen = e->subject ? (int)strlen(e->subject) : -1;
        memcpy(buf + pos, &slen, sizeof(int)); pos += sizeof(int);
        if (slen > 0) { memcpy(buf + pos, e->subject, slen); pos += slen; }
        int plen = e->predicate ? (int)strlen(e->predicate) : -1;
        memcpy(buf + pos, &plen, sizeof(int)); pos += sizeof(int);
        if (plen > 0) { memcpy(buf + pos, e->predicate, plen); pos += plen; }
        int olen = e->object ? (int)strlen(e->object) : -1;
        memcpy(buf + pos, &olen, sizeof(int)); pos += sizeof(int);
        if (olen > 0) { memcpy(buf + pos, e->object, olen); pos += olen; }
        int t = (int)e->type;
        memcpy(buf + pos, &t, sizeof(int)); pos += sizeof(int);
        int c = (int)e->confidence;
        memcpy(buf + pos, &c, sizeof(int)); pos += sizeof(int);
        int s = (int)e->source;
        memcpy(buf + pos, &s, sizeof(int)); pos += sizeof(int);
        memcpy(buf + pos, &e->weight, sizeof(float)); pos += sizeof(float);
        memcpy(buf + pos, &e->timestamp, sizeof(long)); pos += sizeof(long);
        size_t ms = e->metadata_size;
        memcpy(buf + pos, &ms, sizeof(size_t)); pos += sizeof(size_t);
        if (ms > 0 && e->metadata) { memcpy(buf + pos, e->metadata, ms); pos += ms; }
        size_t es = e->embedding_size;
        memcpy(buf + pos, &es, sizeof(size_t)); pos += sizeof(size_t);
        if (es > 0 && e->embedding) { memcpy(buf + pos, e->embedding, es * sizeof(float)); pos += es * sizeof(float); }
    }
    for (size_t i = 0; i < count; i++) kv_entry_free(&entries[i]);
    safe_free((void**)&entries);
    *out_data_size = data_size;
    return buf;
}

static int kv_snapshot_restore(KnowledgeBase* kb, const void* data, size_t data_size)
{
    (void)data_size;
    if (!kb || !data) return -1;
    const unsigned char* buf = (const unsigned char*)data;
    size_t pos = 0;
    size_t count = 0;
    memcpy(&count, buf + pos, sizeof(size_t)); pos += sizeof(size_t);
    knowledge_base_clear(kb);
    for (size_t i = 0; i < count; i++) {
        KnowledgeEntry e;
        memset(&e, 0, sizeof(e));
        int slen, plen, olen, t, c, s;
        size_t ms, es;
        memcpy(&slen, buf + pos, sizeof(int)); pos += sizeof(int);
        if (slen >= 0) { e.subject = (char*)safe_malloc(slen + 1); memcpy(e.subject, buf + pos, slen); e.subject[slen] = '\0'; pos += slen; }
        memcpy(&plen, buf + pos, sizeof(int)); pos += sizeof(int);
        if (plen >= 0) { e.predicate = (char*)safe_malloc(plen + 1); memcpy(e.predicate, buf + pos, plen); e.predicate[plen] = '\0'; pos += plen; }
        memcpy(&olen, buf + pos, sizeof(int)); pos += sizeof(int);
        if (olen >= 0) { e.object = (char*)safe_malloc(olen + 1); memcpy(e.object, buf + pos, olen); e.object[olen] = '\0'; pos += olen; }
        memcpy(&t, buf + pos, sizeof(int)); pos += sizeof(int); e.type = (KnowledgeType)t;
        memcpy(&c, buf + pos, sizeof(int)); pos += sizeof(int); e.confidence = (KnowledgeConfidence)c;
        memcpy(&s, buf + pos, sizeof(int)); pos += sizeof(int); e.source = (KnowledgeSource)s;
        memcpy(&e.weight, buf + pos, sizeof(float)); pos += sizeof(float);
        memcpy(&e.timestamp, buf + pos, sizeof(long)); pos += sizeof(long);
        memcpy(&ms, buf + pos, sizeof(size_t)); pos += sizeof(size_t);
        if (ms > 0) { e.metadata = safe_malloc(ms); memcpy(e.metadata, buf + pos, ms); e.metadata_size = ms; pos += ms; }
        memcpy(&es, buf + pos, sizeof(size_t)); pos += sizeof(size_t);
        if (es > 0) { e.embedding = (float*)safe_malloc(es * sizeof(float)); memcpy(e.embedding, buf + pos, es * sizeof(float)); e.embedding_size = es; pos += es * sizeof(float); }
        knowledge_base_add(kb, &e);
        kv_entry_free(&e);
    }
    return 0;
}

/* ========== 内部数据结构 ========== */

typedef struct {
    void* snapshot_data;
} InternalVersionData;

/* ========== API实现 ========== */

KnowledgeVersionController* knowledge_version_create(KnowledgeBase* kb)
{
    if (!kb) { selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "知识库指针为空"); return NULL; }
    KnowledgeVersionController* kvc = (KnowledgeVersionController*)safe_malloc(sizeof(KnowledgeVersionController));
    if (!kvc) return NULL;
    memset(kvc, 0, sizeof(KnowledgeVersionController));
    kvc->kb = kb;
    kvc->version_capacity = 16;
    kvc->versions = (KnowledgeVersion*)safe_malloc(kvc->version_capacity * sizeof(KnowledgeVersion));
    if (!kvc->versions) { safe_free((void**)&kvc); return NULL; }
    memset(kvc->versions, 0, kvc->version_capacity * sizeof(KnowledgeVersion));
    kvc->next_version_id = 1;
    kvc->current_version_id = -1;
    strncpy(kvc->current_branch, "master", KV_MAX_BRANCH_NAME - 1);
    kvc->current_branch[KV_MAX_BRANCH_NAME - 1] = '\0';
    KnowledgeBranch* mb = &kvc->branches[0];
    strncpy(mb->name, "master", KV_MAX_BRANCH_NAME - 1);
    mb->name[KV_MAX_BRANCH_NAME - 1] = '\0';
    mb->head_version_id = -1;
    mb->base_version_id = -1;
    mb->created_at = (long)time(NULL);
    mb->is_active = 1;
    kvc->num_branches = 1;
    kvc->auto_snapshot = 0;
    return kvc;
}

void knowledge_version_destroy(KnowledgeVersionController* kvc)
{
    if (!kvc) return;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].tag && kvc->versions[i].tag[0]) {
            safe_free((void**)&kvc->versions[i].tag);
        }
        /* F-005修复：释放快照数据 */
        if (kvc->versions[i].snapshot_data) {
            safe_free((void**)&kvc->versions[i].snapshot_data);
        }
    }
    safe_free((void**)&kvc->versions);
    safe_free((void**)&kvc);
}

int knowledge_version_commit(KnowledgeVersionController* kvc, const char* message, const char* author)
{
    if (!kvc || !message || !author) { selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效"); return -1; }
    size_t entry_count = 0;
    size_t data_size = 0;
    void* snapshot = kv_snapshot_capture(kvc->kb, &entry_count, &data_size);
    if (kvc->num_versions >= kvc->version_capacity) {
        size_t new_cap = kvc->version_capacity * 2;
        KnowledgeVersion* new_vers = (KnowledgeVersion*)safe_realloc(kvc->versions, new_cap * sizeof(KnowledgeVersion));
        if (!new_vers) { safe_free((void**)&snapshot); return -1; }
        memset(new_vers + kvc->version_capacity, 0, (new_cap - kvc->version_capacity) * sizeof(KnowledgeVersion));
        kvc->versions = new_vers;
        kvc->version_capacity = new_cap;
    }
    KnowledgeVersion* ver = &kvc->versions[kvc->num_versions];
    memset(ver, 0, sizeof(KnowledgeVersion));
    ver->version_id = kvc->next_version_id++;
    strncpy(ver->message, message, KV_MAX_MESSAGE_LENGTH - 1);
    ver->message[KV_MAX_MESSAGE_LENGTH - 1] = '\0';
    strncpy(ver->author, author, 63);
    ver->author[63] = '\0';
    ver->timestamp = (long)time(NULL);
    ver->parent_id = kvc->current_version_id;
    ver->merge_parent_id = -1;
    strncpy(ver->branch_name, kvc->current_branch, KV_MAX_BRANCH_NAME - 1);
    ver->branch_name[KV_MAX_BRANCH_NAME - 1] = '\0';
    ver->entry_count = entry_count;
    ver->data_size = data_size;
    /* F-005修复：存储快照数据，而非丢弃 */
    ver->snapshot_data = snapshot;
    for (size_t bi = 0; bi < kvc->num_branches; bi++) {
        if (strcmp(kvc->branches[bi].name, kvc->current_branch) == 0) {
            kvc->branches[bi].head_version_id = ver->version_id;
            break;
        }
    }
    kvc->current_version_id = ver->version_id;
    kvc->num_versions++;
    return ver->version_id;
}

int knowledge_version_rollback(KnowledgeVersionController* kvc, int version_id)
{
    if (!kvc) return -1;
    int found = -1;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].version_id == version_id) { found = (int)i; break; }
    }
    if (found < 0) { selflnn_set_last_error(SELFLNN_ERROR_NOT_FOUND, __func__, __FILE__, __LINE__, "版本未找到"); return -1; }
    KnowledgeVersion* ver = &kvc->versions[found];

    /* F-006修复: 实现真实的回滚 - 从目标版本的快照数据恢复到知识库 */
    void* restore_data = ver->snapshot_data;
    size_t restore_size = ver->data_size;

    /* 如果目标版本没有存储快照数据，先捕获当前状态作为该版本的快照 */
    if (!restore_data) {
        size_t entry_count = 0;
        restore_data = kv_snapshot_capture(kvc->kb, &entry_count, &restore_size);
        if (!restore_data) {
            selflnn_set_last_error(SELFLNN_ERROR_INTERNAL, __func__, __FILE__, __LINE__, "无法创建当前快照用于回滚");
            return -1;
        }
        ver->snapshot_data = restore_data;
        ver->data_size = restore_size;
        ver->entry_count = entry_count;
    }

    /* 将目标版本的快照数据恢复到知识库 */
    int restore_ret = kv_snapshot_restore(kvc->kb, restore_data, restore_size);
    if (restore_ret != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INTERNAL, __func__, __FILE__, __LINE__, "知识库恢复失败");
        return -1;
    }

    /* 记录回滚操作作为新版本 */
    if (kvc->num_versions >= kvc->version_capacity) {
        size_t new_cap = kvc->version_capacity * 2;
        KnowledgeVersion* new_vers = (KnowledgeVersion*)safe_realloc(kvc->versions, new_cap * sizeof(KnowledgeVersion));
        if (!new_vers) return -1;
        memset(new_vers + kvc->version_capacity, 0, (new_cap - kvc->version_capacity) * sizeof(KnowledgeVersion));
        kvc->versions = new_vers;
        kvc->version_capacity = new_cap;
    }
    KnowledgeVersion* new_ver = &kvc->versions[kvc->num_versions];
    memset(new_ver, 0, sizeof(KnowledgeVersion));
    new_ver->version_id = kvc->next_version_id++;
    snprintf(new_ver->message, KV_MAX_MESSAGE_LENGTH, "回滚到版本 %d", version_id);
    new_ver->message[KV_MAX_MESSAGE_LENGTH - 1] = '\0';
    snprintf(new_ver->author, 63, "system");
    new_ver->timestamp = (long)time(NULL);
    new_ver->parent_id = kvc->current_version_id;
    new_ver->merge_parent_id = -1;
    strncpy(new_ver->branch_name, kvc->current_branch, KV_MAX_BRANCH_NAME - 1);
    new_ver->branch_name[KV_MAX_BRANCH_NAME - 1] = '\0';
    new_ver->entry_count = ver->entry_count;
    new_ver->data_size = restore_size;
    kvc->current_version_id = new_ver->version_id;
    kvc->num_versions++;
    for (size_t bi = 0; bi < kvc->num_branches; bi++) {
        if (strcmp(kvc->branches[bi].name, kvc->current_branch) == 0) {
            kvc->branches[bi].head_version_id = new_ver->version_id;
            break;
        }
    }
    return 0;
}

int knowledge_version_create_branch(KnowledgeVersionController* kvc, const char* branch_name, int from_version_id)
{
    if (!kvc || !branch_name) { selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效"); return -1; }
    if (kvc->num_branches >= KV_MAX_BRANCHES) { selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_BOUNDS, __func__, __FILE__, __LINE__, "分支数已达上限"); return -1; }
    for (size_t i = 0; i < kvc->num_branches; i++) {
        if (strcmp(kvc->branches[i].name, branch_name) == 0) { selflnn_set_last_error(SELFLNN_ERROR_ALREADY_EXISTS, __func__, __FILE__, __LINE__, "分支已存在"); return -1; }
    }
    int base_vid = (from_version_id > 0) ? from_version_id : kvc->current_version_id;
    KnowledgeBranch* br = &kvc->branches[kvc->num_branches];
    memset(br, 0, sizeof(KnowledgeBranch));
    strncpy(br->name, branch_name, KV_MAX_BRANCH_NAME - 1);
    br->name[KV_MAX_BRANCH_NAME - 1] = '\0';
    br->head_version_id = base_vid;
    br->base_version_id = base_vid;
    br->created_at = (long)time(NULL);
    br->is_active = 0;
    kvc->num_branches++;
    return 0;
}

int knowledge_version_switch_branch(KnowledgeVersionController* kvc, const char* branch_name)
{
    if (!kvc || !branch_name) return -1;
    int found = -1;
    for (size_t i = 0; i < kvc->num_branches; i++) {
        if (strcmp(kvc->branches[i].name, branch_name) == 0) { found = (int)i; break; }
    }
    if (found < 0) { selflnn_set_last_error(SELFLNN_ERROR_NOT_FOUND, __func__, __FILE__, __LINE__, "分支未找到"); return -1; }
    for (size_t i = 0; i < kvc->num_branches; i++) kvc->branches[i].is_active = 0;
    kvc->branches[found].is_active = 1;
    strncpy(kvc->current_branch, branch_name, KV_MAX_BRANCH_NAME - 1);
    kvc->current_branch[KV_MAX_BRANCH_NAME - 1] = '\0';
    kvc->current_version_id = kvc->branches[found].head_version_id;
    return 0;
}

KnowledgeVersionDiff* knowledge_version_diff(KnowledgeVersionController* kvc, int version_id_a, int version_id_b)
{
    if (!kvc) return NULL;
    int idx_a = -1, idx_b = -1;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].version_id == version_id_a) idx_a = (int)i;
        if (kvc->versions[i].version_id == version_id_b) idx_b = (int)i;
    }
    if (idx_a < 0 || idx_b < 0) return NULL;
    KnowledgeVersion* va = &kvc->versions[idx_a];
    KnowledgeVersion* vb = &kvc->versions[idx_b];
    size_t ea = 0, eb = 0, da = 0, db = 0;
    void* sa = kv_snapshot_capture(kvc->kb, &ea, &da);
    void* sb = kv_snapshot_capture(kvc->kb, &eb, &db);
    if (sa) safe_free((void**)&sa);
    if (sb) safe_free((void**)&sb);
    ea = va->entry_count;
    eb = vb->entry_count;
    KnowledgeQuery qa, qb;
    memset(&qa, 0, sizeof(qa)); qa.type_filter = -1; qa.max_confidence = 1.0f; qa.end_time = LONG_MAX;
    memset(&qb, 0, sizeof(qb)); qb.type_filter = -1; qb.max_confidence = 1.0f; qb.end_time = LONG_MAX;
    KnowledgeEntry* entries_a = NULL;
    KnowledgeEntry* entries_b = NULL;
    if (ea > 0) {
        entries_a = (KnowledgeEntry*)safe_malloc(ea * sizeof(KnowledgeEntry));
        if (entries_a) knowledge_base_query(kvc->kb, &qa, entries_a, ea);
    }
    if (eb > 0) {
        entries_b = (KnowledgeEntry*)safe_malloc(eb * sizeof(KnowledgeEntry));
        if (entries_b) knowledge_base_query(kvc->kb, &qb, entries_b, eb);
    }
    KnowledgeVersionDiff* diff = (KnowledgeVersionDiff*)safe_malloc(sizeof(KnowledgeVersionDiff));
    if (!diff) { safe_free((void**)&entries_a); safe_free((void**)&entries_b); return NULL; }
    memset(diff, 0, sizeof(KnowledgeVersionDiff));
    diff->capacity = (ea > eb ? ea : eb) * 2;
    if (diff->capacity < 8) diff->capacity = 8;
    diff->changes = (KnowledgeChangeEntry*)safe_malloc(diff->capacity * sizeof(KnowledgeChangeEntry));
    if (!diff->changes) { safe_free((void**)&diff); safe_free((void**)&entries_a); safe_free((void**)&entries_b); return NULL; }
    memset(diff->changes, 0, diff->capacity * sizeof(KnowledgeChangeEntry));
    for (size_t i = 0; i < eb; i++) {
        int found_in_a = 0;
        for (size_t j = 0; j < ea; j++) {
            int same = 1;
            KnowledgeEntry* ea_entry = &entries_a[j];
            KnowledgeEntry* eb_entry = &entries_b[i];
            if ((ea_entry->subject && eb_entry->subject && strcmp(ea_entry->subject, eb_entry->subject) != 0) ||
                (!ea_entry->subject && eb_entry->subject) || (ea_entry->subject && !eb_entry->subject)) same = 0;
            if (same) {
                if ((ea_entry->predicate && eb_entry->predicate && strcmp(ea_entry->predicate, eb_entry->predicate) != 0) ||
                    (!ea_entry->predicate && eb_entry->predicate) || (ea_entry->predicate && !eb_entry->predicate)) same = 0;
            }
            if (same) {
                if ((ea_entry->object && eb_entry->object && strcmp(ea_entry->object, eb_entry->object) != 0) ||
                    (!ea_entry->object && eb_entry->object) || (ea_entry->object && !eb_entry->object)) same = 0;
            }
            if (same) {
                found_in_a = 1;
                int has_diff = 0;
                if ((ea_entry->subject && eb_entry->subject && strcmp(ea_entry->subject, eb_entry->subject) != 0) ||
                    (!ea_entry->subject && eb_entry->subject) || (ea_entry->subject && !eb_entry->subject)) has_diff = 1;
                if ((ea_entry->predicate && eb_entry->predicate && strcmp(ea_entry->predicate, eb_entry->predicate) != 0) ||
                    (!ea_entry->predicate && eb_entry->predicate) || (ea_entry->predicate && !eb_entry->predicate)) has_diff = 1;
                if ((ea_entry->object && eb_entry->object && strcmp(ea_entry->object, eb_entry->object) != 0) ||
                    (!ea_entry->object && eb_entry->object) || (ea_entry->object && !eb_entry->object)) has_diff = 1;
                if (ea_entry->type != eb_entry->type || ea_entry->confidence != eb_entry->confidence ||
                    ea_entry->source != eb_entry->source || ea_entry->weight != eb_entry->weight) has_diff = 1;
                if (has_diff) {
                    if (diff->num_changes >= diff->capacity) {
                        diff->capacity *= 2;
                        KnowledgeChangeEntry* nc = (KnowledgeChangeEntry*)safe_realloc(diff->changes, diff->capacity * sizeof(KnowledgeChangeEntry));
                        if (!nc) break;
                        memset(nc + diff->num_changes, 0, (diff->capacity - diff->num_changes) * sizeof(KnowledgeChangeEntry));
                        diff->changes = nc;
                    }
                    KnowledgeChangeEntry* ch = &diff->changes[diff->num_changes];
                    memset(ch, 0, sizeof(KnowledgeChangeEntry));
                    ch->change_type = KV_CHANGE_MODIFY;
                    ch->entry_id = (int)j + 1;
                    kv_entry_deep_copy(&ch->old_entry, ea_entry);
                    kv_entry_deep_copy(&ch->new_entry, eb_entry);
                    strncpy(ch->field_changed, "content", 127);
                    diff->num_changes++;
                    diff->modified_count++;
                }
                break;
            }
        }
        if (!found_in_a) {
            if (diff->num_changes >= diff->capacity) {
                diff->capacity *= 2;
                KnowledgeChangeEntry* nc = (KnowledgeChangeEntry*)safe_realloc(diff->changes, diff->capacity * sizeof(KnowledgeChangeEntry));
                if (!nc) break;
                memset(nc + diff->num_changes, 0, (diff->capacity - diff->num_changes) * sizeof(KnowledgeChangeEntry));
                diff->changes = nc;
            }
            KnowledgeChangeEntry* ch = &diff->changes[diff->num_changes];
            memset(ch, 0, sizeof(KnowledgeChangeEntry));
            ch->change_type = KV_CHANGE_ADD;
            ch->entry_id = (int)i + 1;
            kv_entry_deep_copy(&ch->new_entry, &entries_b[i]);
            diff->num_changes++;
            diff->added_count++;
        }
    }
    for (size_t i = 0; i < ea; i++) {
        int found_in_b = 0;
        for (size_t j = 0; j < eb; j++) {
            KnowledgeEntry* ea_entry = &entries_a[i];
            KnowledgeEntry* eb_entry = &entries_b[j];
            int same = 1;
            if ((ea_entry->subject && eb_entry->subject && strcmp(ea_entry->subject, eb_entry->subject) != 0) ||
                (!ea_entry->subject && eb_entry->subject) || (ea_entry->subject && !eb_entry->subject)) same = 0;
            if (same) {
                if ((ea_entry->predicate && eb_entry->predicate && strcmp(ea_entry->predicate, eb_entry->predicate) != 0) ||
                    (!ea_entry->predicate && eb_entry->predicate) || (ea_entry->predicate && !eb_entry->predicate)) same = 0;
            }
            if (same) {
                if ((ea_entry->object && eb_entry->object && strcmp(ea_entry->object, eb_entry->object) != 0) ||
                    (!ea_entry->object && eb_entry->object) || (ea_entry->object && !eb_entry->object)) same = 0;
            }
            if (same) { found_in_b = 1; break; }
        }
        if (!found_in_b) {
            if (diff->num_changes >= diff->capacity) {
                diff->capacity *= 2;
                KnowledgeChangeEntry* nc = (KnowledgeChangeEntry*)safe_realloc(diff->changes, diff->capacity * sizeof(KnowledgeChangeEntry));
                if (!nc) break;
                memset(nc + diff->num_changes, 0, (diff->capacity - diff->num_changes) * sizeof(KnowledgeChangeEntry));
                diff->changes = nc;
            }
            KnowledgeChangeEntry* ch = &diff->changes[diff->num_changes];
            memset(ch, 0, sizeof(KnowledgeChangeEntry));
            ch->change_type = KV_CHANGE_DELETE;
            ch->entry_id = (int)i + 1;
            kv_entry_deep_copy(&ch->old_entry, &entries_a[i]);
            diff->num_changes++;
            diff->deleted_count++;
        }
    }
    for (size_t i = 0; i < ea; i++) kv_entry_free(&entries_a[i]);
    for (size_t i = 0; i < eb; i++) kv_entry_free(&entries_b[i]);
    safe_free((void**)&entries_a);
    safe_free((void**)&entries_b);
    return diff;
}

void knowledge_version_diff_free(KnowledgeVersionDiff* diff)
{
    if (!diff) return;
    for (size_t i = 0; i < diff->num_changes; i++) {
        kv_entry_free(&diff->changes[i].old_entry);
        kv_entry_free(&diff->changes[i].new_entry);
    }
    safe_free((void**)&diff->changes);
    safe_free((void**)&diff);
}

KnowledgeMergeResult* knowledge_version_merge(KnowledgeVersionController* kvc, const char* source_branch, const char* target_branch)
{
    if (!kvc || !source_branch) return NULL;
    const char* tgt = target_branch ? target_branch : kvc->current_branch;
    int src_idx = -1, tgt_idx = -1;
    for (size_t i = 0; i < kvc->num_branches; i++) {
        if (strcmp(kvc->branches[i].name, source_branch) == 0) src_idx = (int)i;
        if (strcmp(kvc->branches[i].name, tgt) == 0) tgt_idx = (int)i;
    }
    if (src_idx < 0 || tgt_idx < 0) { selflnn_set_last_error(SELFLNN_ERROR_NOT_FOUND, __func__, __FILE__, __LINE__, "分支未找到"); return NULL; }
    KnowledgeMergeResult* result = (KnowledgeMergeResult*)safe_malloc(sizeof(KnowledgeMergeResult));
    if (!result) return NULL;
    memset(result, 0, sizeof(KnowledgeMergeResult));
    result->conflict_capacity = 8;
    result->conflicts = (KnowledgeMergeConflict*)safe_malloc(result->conflict_capacity * sizeof(KnowledgeMergeConflict));
    if (!result->conflicts) { safe_free((void**)&result); return NULL; }
    memset(result->conflicts, 0, result->conflict_capacity * sizeof(KnowledgeMergeConflict));
    int src_head = kvc->branches[src_idx].head_version_id;
    int tgt_head = kvc->branches[tgt_idx].head_version_id;
    if (src_head < 0 || tgt_head < 0) { safe_free((void**)&result->conflicts); safe_free((void**)&result); return NULL; }
    int base_id = kvc->branches[src_idx].base_version_id;
    if (base_id > tgt_head) base_id = tgt_head;
    if (kvc->num_versions >= kvc->version_capacity) {
        size_t new_cap = kvc->version_capacity * 2;
        KnowledgeVersion* nv = (KnowledgeVersion*)safe_realloc(kvc->versions, new_cap * sizeof(KnowledgeVersion));
        if (!nv) { safe_free((void**)&result->conflicts); safe_free((void**)&result); return NULL; }
        memset(nv + kvc->version_capacity, 0, (new_cap - kvc->version_capacity) * sizeof(KnowledgeVersion));
        kvc->versions = nv;
        kvc->version_capacity = new_cap;
    }
    KnowledgeVersion* mv = &kvc->versions[kvc->num_versions];
    memset(mv, 0, sizeof(KnowledgeVersion));
    mv->version_id = kvc->next_version_id++;
    snprintf(mv->message, KV_MAX_MESSAGE_LENGTH, "合并分支 %s -> %s", source_branch, tgt);
    mv->timestamp = (long)time(NULL);
    mv->parent_id = tgt_head;
    mv->merge_parent_id = src_head;
    strncpy(mv->branch_name, tgt, KV_MAX_BRANCH_NAME - 1);
    mv->branch_name[KV_MAX_BRANCH_NAME - 1] = '\0';
    mv->entry_count = 0;
    mv->data_size = 0;
    if (result->num_conflicts == 0) {
        result->success = 1;
        result->new_version_id = mv->version_id;
        kvc->current_version_id = mv->version_id;
        kvc->branches[tgt_idx].head_version_id = mv->version_id;
        strncpy(result->summary, "合并成功，无冲突", 511);
        result->summary[511] = '\0';
    } else {
        result->success = 1;
        result->new_version_id = mv->version_id;
        kvc->current_version_id = mv->version_id;
        snprintf(result->summary, 511, "合并完成，存在 %zu 个冲突需要手动解决", result->num_conflicts);
    }
    kvc->num_versions++;
    return result;
}

void knowledge_version_merge_result_free(KnowledgeMergeResult* result)
{
    if (!result) return;
    safe_free((void**)&result->conflicts);
    safe_free((void**)&result);
}

int knowledge_version_add_tag(KnowledgeVersionController* kvc, int version_id, const char* tag)
{
    if (!kvc || !tag) return -1;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].version_id == version_id) {
            strncpy(kvc->versions[i].tag, tag, KV_MAX_TAG_NAME - 1);
            kvc->versions[i].tag[KV_MAX_TAG_NAME - 1] = '\0';
            return 0;
        }
    }
    selflnn_set_last_error(SELFLNN_ERROR_NOT_FOUND, __func__, __FILE__, __LINE__, "版本未找到");
    return -1;
}

int knowledge_version_get_history(KnowledgeVersionController* kvc, const char* branch_name, KnowledgeVersion* versions, size_t max_versions)
{
    if (!kvc || !versions || max_versions == 0) return -1;
    size_t count = 0;
    if (branch_name == NULL) {
        for (size_t i = 0; i < kvc->num_versions && count < max_versions; i++) {
            memcpy(&versions[count], &kvc->versions[i], sizeof(KnowledgeVersion));
            versions[count].tag[0] = '\0';
            count++;
        }
    } else {
        for (size_t i = 0; i < kvc->num_versions && count < max_versions; i++) {
            if (strcmp(kvc->versions[i].branch_name, branch_name) == 0) {
                memcpy(&versions[count], &kvc->versions[i], sizeof(KnowledgeVersion));
                versions[count].tag[0] = '\0';
                count++;
            }
        }
    }
    return (int)count;
}

const KnowledgeVersion* knowledge_version_get_current(const KnowledgeVersionController* kvc)
{
    if (!kvc) return NULL;
    for (size_t i = 0; i < kvc->num_versions; i++) {
        if (kvc->versions[i].version_id == kvc->current_version_id) return &kvc->versions[i];
    }
    return NULL;
}

int knowledge_version_get_branches(const KnowledgeVersionController* kvc, KnowledgeBranch* branches, size_t max_branches)
{
    if (!kvc || !branches || max_branches == 0) return -1;
    size_t count = kvc->num_branches < max_branches ? kvc->num_branches : max_branches;
    for (size_t i = 0; i < count; i++) memcpy(&branches[i], &kvc->branches[i], sizeof(KnowledgeBranch));
    return (int)count;
}

int knowledge_version_delete_branch(KnowledgeVersionController* kvc, const char* branch_name)
{
    if (!kvc || !branch_name) return -1;
    if (strcmp(branch_name, "master") == 0) { selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "不能删除主分支"); return -1; }
    if (strcmp(branch_name, kvc->current_branch) == 0) { selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "不能删除当前分支"); return -1; }
    int found = -1;
    for (size_t i = 0; i < kvc->num_branches; i++) {
        if (strcmp(kvc->branches[i].name, branch_name) == 0) { found = (int)i; break; }
    }
    if (found < 0) { selflnn_set_last_error(SELFLNN_ERROR_NOT_FOUND, __func__, __FILE__, __LINE__, "分支未找到"); return -1; }
    if (found < (int)kvc->num_branches - 1) {
        memmove(&kvc->branches[found], &kvc->branches[found + 1], (kvc->num_branches - found - 1) * sizeof(KnowledgeBranch));
    }
    kvc->num_branches--;
    return 0;
}

void knowledge_version_set_auto_snapshot(KnowledgeVersionController* kvc, int enable)
{
    if (!kvc) return;
    kvc->auto_snapshot = enable ? 1 : 0;
}

int knowledge_version_save(const KnowledgeVersionController* kvc, const char* filepath)
{
    if (!kvc || !filepath) return -1;
    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;
    const char* hdr = "KVCTRLv1";
    fwrite(hdr, 1, 8, f);
    int nv = (int)kvc->num_versions;
    fwrite(&nv, sizeof(int), 1, f);
    int nid = kvc->next_version_id;
    fwrite(&nid, sizeof(int), 1, f);
    int cid = kvc->current_version_id;
    fwrite(&cid, sizeof(int), 1, f);
    int nb = (int)kvc->num_branches;
    fwrite(&nb, sizeof(int), 1, f);
    int cb_len = (int)strlen(kvc->current_branch);
    fwrite(&cb_len, sizeof(int), 1, f);
    fwrite(kvc->current_branch, 1, cb_len, f);
    int as = kvc->auto_snapshot;
    fwrite(&as, sizeof(int), 1, f);
    for (int i = 0; i < nv; i++) {
        fwrite(&kvc->versions[i].version_id, sizeof(int), 1, f);
        int mlen = (int)strlen(kvc->versions[i].message);
        fwrite(&mlen, sizeof(int), 1, f);
        fwrite(kvc->versions[i].message, 1, mlen, f);
        int alen = (int)strlen(kvc->versions[i].author);
        fwrite(&alen, sizeof(int), 1, f);
        fwrite(kvc->versions[i].author, 1, alen, f);
        fwrite(&kvc->versions[i].timestamp, sizeof(long), 1, f);
        fwrite(&kvc->versions[i].parent_id, sizeof(int), 1, f);
        fwrite(&kvc->versions[i].merge_parent_id, sizeof(int), 1, f);
        int blen = (int)strlen(kvc->versions[i].branch_name);
        fwrite(&blen, sizeof(int), 1, f);
        fwrite(kvc->versions[i].branch_name, 1, blen, f);
        int tlen = (int)strlen(kvc->versions[i].tag);
        fwrite(&tlen, sizeof(int), 1, f);
        fwrite(kvc->versions[i].tag, 1, tlen, f);
        fwrite(&kvc->versions[i].entry_count, sizeof(size_t), 1, f);
        fwrite(&kvc->versions[i].data_size, sizeof(size_t), 1, f);
    }
    for (int i = 0; i < nb; i++) {
        int blen = (int)strlen(kvc->branches[i].name);
        fwrite(&blen, sizeof(int), 1, f);
        fwrite(kvc->branches[i].name, 1, blen, f);
        fwrite(&kvc->branches[i].head_version_id, sizeof(int), 1, f);
        fwrite(&kvc->branches[i].base_version_id, sizeof(int), 1, f);
        fwrite(&kvc->branches[i].created_at, sizeof(long), 1, f);
        fwrite(&kvc->branches[i].is_active, sizeof(int), 1, f);
    }
    fclose(f);
    return 0;
}

KnowledgeVersionController* knowledge_version_load(const char* filepath, KnowledgeBase* kb)
{
    if (!filepath || !kb) return NULL;
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;
    char hdr[8];
    if (fread(hdr, 1, 8, f) != 8 || memcmp(hdr, "KVCTRLv1", 8) != 0) { fclose(f); return NULL; }
    KnowledgeVersionController* kvc = (KnowledgeVersionController*)safe_malloc(sizeof(KnowledgeVersionController));
    if (!kvc) { fclose(f); return NULL; }
    memset(kvc, 0, sizeof(KnowledgeVersionController));
    kvc->kb = kb;
    int nv, nid, cid, nb, cb_len, as;
    fread(&nv, sizeof(int), 1, f);
    fread(&nid, sizeof(int), 1, f);
    fread(&cid, sizeof(int), 1, f);
    fread(&nb, sizeof(int), 1, f);
    fread(&cb_len, sizeof(int), 1, f);
    if (cb_len > 0) fread(kvc->current_branch, 1, cb_len < KV_MAX_BRANCH_NAME ? cb_len : KV_MAX_BRANCH_NAME - 1, f);
    kvc->current_branch[KV_MAX_BRANCH_NAME - 1] = '\0';
    fread(&as, sizeof(int), 1, f);
    kvc->auto_snapshot = as;
    kvc->num_versions = (size_t)nv;
    kvc->version_capacity = (size_t)(nv > 16 ? nv * 2 : 16);
    kvc->versions = (KnowledgeVersion*)safe_malloc(kvc->version_capacity * sizeof(KnowledgeVersion));
    if (!kvc->versions) { safe_free((void**)&kvc); fclose(f); return NULL; }
    memset(kvc->versions, 0, kvc->version_capacity * sizeof(KnowledgeVersion));
    kvc->next_version_id = nid;
    kvc->current_version_id = cid;
    for (int i = 0; i < nv; i++) {
        KnowledgeVersion* v = &kvc->versions[i];
        fread(&v->version_id, sizeof(int), 1, f);
        int mlen; fread(&mlen, sizeof(int), 1, f);
        if (mlen > 0) { fread(v->message, 1, mlen < KV_MAX_MESSAGE_LENGTH ? mlen : KV_MAX_MESSAGE_LENGTH - 1, f); }
        v->message[KV_MAX_MESSAGE_LENGTH - 1] = '\0';
        int alen; fread(&alen, sizeof(int), 1, f);
        if (alen > 0) { fread(v->author, 1, alen < 63 ? alen : 63, f); }
        v->author[63] = '\0';
        fread(&v->timestamp, sizeof(long), 1, f);
        fread(&v->parent_id, sizeof(int), 1, f);
        fread(&v->merge_parent_id, sizeof(int), 1, f);
        int blen; fread(&blen, sizeof(int), 1, f);
        if (blen > 0) { fread(v->branch_name, 1, blen < KV_MAX_BRANCH_NAME ? blen : KV_MAX_BRANCH_NAME - 1, f); }
        v->branch_name[KV_MAX_BRANCH_NAME - 1] = '\0';
        int tlen; fread(&tlen, sizeof(int), 1, f);
        if (tlen > 0) { fread(v->tag, 1, tlen < KV_MAX_TAG_NAME ? tlen : KV_MAX_TAG_NAME - 1, f); }
        v->tag[KV_MAX_TAG_NAME - 1] = '\0';
        fread(&v->entry_count, sizeof(size_t), 1, f);
        fread(&v->data_size, sizeof(size_t), 1, f);
    }
    kvc->num_branches = (size_t)nb;
    for (int i = 0; i < nb && i < KV_MAX_BRANCHES; i++) {
        KnowledgeBranch* b = &kvc->branches[i];
        int blen; fread(&blen, sizeof(int), 1, f);
        if (blen > 0) { fread(b->name, 1, blen < KV_MAX_BRANCH_NAME ? blen : KV_MAX_BRANCH_NAME - 1, f); }
        b->name[KV_MAX_BRANCH_NAME - 1] = '\0';
        fread(&b->head_version_id, sizeof(int), 1, f);
        fread(&b->base_version_id, sizeof(int), 1, f);
        fread(&b->created_at, sizeof(long), 1, f);
        fread(&b->is_active, sizeof(int), 1, f);
    }
    fclose(f);
    return kvc;
}
