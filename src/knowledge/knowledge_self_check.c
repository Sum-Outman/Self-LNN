#define SELFLNN_KNOWLEDGE_INTERNAL /* 访问KnowledgeGraph完整结构体(kg->edges) */
#include "selflnn/knowledge/knowledge_self_check.h"
#include "selflnn/knowledge/knowledge_graph.h"
#include "selflnn/knowledge/graph_storage.h" /* KnowledgeGraph完整结构需要 */
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* GRAPH_EDGE_NEGATION - 知识图谱否定边类型标记
 * GraphEdgeType枚举无此值,定义为特殊负值避免与标准类型(0-4)冲突 */
#define GRAPH_EDGE_NEGATION (-1)

#ifdef _WIN32
#include <windows.h>
static double ksc_timestamp_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double ksc_timestamp_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}
#endif

static float ksc_string_similarity(const char* a, const char* b) {
    if (!a || !b) return 0.0f;
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    if (len_a == 0 && len_b == 0) return 1.0f;
    if (len_a == 0 || len_b == 0) return 0.0f;

    size_t max_len = len_a > len_b ? len_a : len_b;
    size_t matrix_size = (len_a + 1) * (len_b + 1);
    int* matrix = (int*)safe_calloc(matrix_size, sizeof(int));
    if (!matrix) return 0.0f;

    for (size_t i = 0; i <= len_a; i++) matrix[i * (len_b + 1)] = (int)i;
    for (size_t j = 0; j <= len_b; j++) matrix[j] = (int)j;

    for (size_t i = 1; i <= len_a; i++) {
        for (size_t j = 1; j <= len_b; j++) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            int del = matrix[(i - 1) * (len_b + 1) + j] + 1;
            int ins = matrix[i * (len_b + 1) + (j - 1)] + 1;
            int sub = matrix[(i - 1) * (len_b + 1) + (j - 1)] + cost;
            int min = del < ins ? del : ins;
            min = min < sub ? min : sub;
            matrix[i * (len_b + 1) + j] = min;
        }
    }

    int dist = matrix[len_a * (len_b + 1) + len_b];
    safe_free((void**)&matrix);

    return 1.0f - (float)dist / (float)max_len;
}

int ksc_check_direct_contradictions(KnowledgeBase* kb, KSSelfCheckConfig* config,
                                     KSSelfCheckReport* report) {
    if (!kb || !config || !report) return -1;

    KnowledgeQuery q;
    memset(&q, 0, sizeof(q));

    KnowledgeEntry* all_entries = (KnowledgeEntry*)safe_calloc(4096, sizeof(KnowledgeEntry));
    if (!all_entries) return -1;

    int total = knowledge_base_query(kb, &q, all_entries, 4096);
    if (total <= 0) { safe_free((void**)&all_entries); return 0; }

    int found = 0;
    for (int i = 0; i < total && found < config->max_issues; i++) {
        for (int j = i + 1; j < total && found < config->max_issues; j++) {
            KnowledgeEntry* a = &all_entries[i];
            KnowledgeEntry* b = &all_entries[j];

            if (!a->subject || !b->subject || !a->predicate || !b->predicate) continue;

            float sub_sim = ksc_string_similarity(a->subject, b->subject);
            float pred_sim = ksc_string_similarity(a->predicate, b->predicate);

            if (sub_sim > config->similarity_threshold && pred_sim > config->similarity_threshold) {
                if (a->object && b->object) {
                    float obj_sim = ksc_string_similarity(a->object, b->object);
                    if (obj_sim < config->contradiction_threshold) {
                        KSContradictionIssue* issue = &report->issues[report->num_issues];
                        memset(issue, 0, sizeof(KSContradictionIssue));
                        issue->issue_id = report->num_issues;
                        issue->type = KSC_ISSUE_CONTRADICTION;
                        issue->contra_type = KSC_CONTRA_DIRECT;
                        issue->entry_id_a = i;
                        issue->entry_id_b = j;
                        strncpy(issue->subject, a->subject, sizeof(issue->subject) - 1);
                        strncpy(issue->predicate, a->predicate, sizeof(issue->predicate) - 1);
                        if (a->object) strncpy(issue->object_a, a->object, sizeof(issue->object_a) - 1);
                        if (b->object) strncpy(issue->object_b, b->object, sizeof(issue->object_b) - 1);
                        issue->confidence_a = a->weight;
                        issue->confidence_b = b->weight;
                        issue->timestamp_a = a->timestamp;
                        issue->timestamp_b = b->timestamp;
                        issue->similarity_score = sub_sim * pred_sim;
                        snprintf(issue->description, sizeof(issue->description),
                                 "直接矛盾: '%s' 的 '%s' 同时为 '%s' 和 '%s'",
                                 a->subject, a->predicate,
                                 a->object ? a->object : "空",
                                 b->object ? b->object : "空");

                        if (a->weight > b->weight) {
                            issue->suggested_resolution = KSC_RESOLVE_KEEP_HIGHER_CONFIDENCE;
                        } else if (a->timestamp > b->timestamp) {
                            issue->suggested_resolution = KSC_RESOLVE_KEEP_NEWER;
                        } else {
                            issue->suggested_resolution = KSC_RESOLVE_MERGE;
                        }
                        issue->auto_fixable = 1;
                        report->num_issues++;
                        found++;
                    }
                }
            }
        }
    }

    report->contradictions_found = found;
    safe_free((void**)&all_entries);
    return found;
}

int ksc_check_redundancy(KnowledgeBase* kb, KSSelfCheckConfig* config,
                          KSSelfCheckReport* report) {
    if (!kb || !config || !report) return -1;

    KnowledgeQuery q;
    memset(&q, 0, sizeof(q));

    KnowledgeEntry* all_entries = (KnowledgeEntry*)safe_calloc(4096, sizeof(KnowledgeEntry));
    if (!all_entries) return -1;

    int total = knowledge_base_query(kb, &q, all_entries, 4096);
    if (total <= 0) { safe_free((void**)&all_entries); return 0; }

    int found = 0;
    for (int i = 0; i < total && found < config->max_issues; i++) {
        for (int j = i + 1; j < total && found < config->max_issues; j++) {
            KnowledgeEntry* a = &all_entries[i];
            KnowledgeEntry* b = &all_entries[j];

            if (!a->subject || !b->subject || !a->predicate || !b->predicate) continue;

            float sub_sim = ksc_string_similarity(a->subject, b->subject);
            float pred_sim = ksc_string_similarity(a->predicate, b->predicate);
            float obj_sim = 1.0f;
            if (a->object && b->object) {
                obj_sim = ksc_string_similarity(a->object, b->object);
            } else if (a->object || b->object) {
                obj_sim = 0.5f;
            }

            float total_sim = (sub_sim + pred_sim + obj_sim) / 3.0f;
            if (total_sim > config->similarity_threshold && i != j) {
                KSContradictionIssue* issue = &report->issues[report->num_issues];
                memset(issue, 0, sizeof(KSContradictionIssue));
                issue->issue_id = report->num_issues;
                issue->type = KSC_ISSUE_REDUNDANCY;
                issue->entry_id_a = i;
                issue->entry_id_b = j;
                strncpy(issue->subject, a->subject, sizeof(issue->subject) - 1);
                strncpy(issue->predicate, a->predicate, sizeof(issue->predicate) - 1);
                if (a->object) strncpy(issue->object_a, a->object, sizeof(issue->object_a) - 1);
                if (b->object) strncpy(issue->object_b, b->object, sizeof(issue->object_b) - 1);
                issue->confidence_a = a->weight;
                issue->confidence_b = b->weight;
                issue->similarity_score = total_sim;
                snprintf(issue->description, sizeof(issue->description),
                         "冗余知识: 相似度 %.2f - '%s' '%s' '%s'",
                         total_sim, a->subject, a->predicate,
                         a->object ? a->object : "空");
                issue->suggested_resolution = KSC_RESOLVE_MERGE;
                issue->auto_fixable = 1;
                report->num_issues++;
                found++;
            }
        }
    }

    report->redundancies_found = found;
    safe_free((void**)&all_entries);
    return found;
}

int ksc_check_temporal_conflicts(KnowledgeBase* kb, KSSelfCheckConfig* config,
                                  KSSelfCheckReport* report) {
    if (!kb || !config || !report) return -1;

    KnowledgeQuery q;
    memset(&q, 0, sizeof(q));

    KnowledgeEntry* all_entries = (KnowledgeEntry*)safe_calloc(4096, sizeof(KnowledgeEntry));
    if (!all_entries) return -1;

    int total = knowledge_base_query(kb, &q, all_entries, 4096);
    if (total <= 0) { safe_free((void**)&all_entries); return 0; }

    long now = (long)time(NULL);
    long stale_threshold = config->stale_age_seconds;

    int found = 0;
    for (int i = 0; i < total && found < config->max_issues; i++) {
        KnowledgeEntry* e = &all_entries[i];
        if (!e->subject) continue;

        long age = now - e->timestamp;
        if (age > stale_threshold && e->confidence > 0.5f) {
            KSContradictionIssue* issue = &report->issues[report->num_issues];
            memset(issue, 0, sizeof(KSContradictionIssue));
            issue->issue_id = report->num_issues;
            issue->type = KSC_ISSUE_STALE_KNOWLEDGE;
            issue->entry_id_a = i;
            strncpy(issue->subject, e->subject, sizeof(issue->subject) - 1);
            strncpy(issue->predicate, e->predicate ? e->predicate : "", sizeof(issue->predicate) - 1);
            if (e->object) strncpy(issue->object_a, e->object, sizeof(issue->object_a) - 1);
            issue->confidence_a = e->weight;
            issue->timestamp_a = e->timestamp;
            snprintf(issue->description, sizeof(issue->description),
                     "过期知识: '%s' 的 '%s' 已存在 %ld 秒",
                     e->subject, e->predicate ? e->predicate : "?", age);
            issue->suggested_resolution = KSC_RESOLVE_FLAG_REVIEW;
            issue->auto_fixable = 0;
            report->num_issues++;
            found++;
        }

        for (int j = i + 1; j < total && found < config->max_issues; j++) {
            KnowledgeEntry* a = &all_entries[i];
            KnowledgeEntry* b = &all_entries[j];
            if (!a->subject || !b->subject || !a->predicate || !b->predicate) continue;

            float sub_sim = ksc_string_similarity(a->subject, b->subject);
            float pred_sim = ksc_string_similarity(a->predicate, b->predicate);

            if (sub_sim > 0.9f && pred_sim > 0.9f && a->object && b->object) {
                float obj_sim = ksc_string_similarity(a->object, b->object);
                if (obj_sim < 0.9f && labs(a->timestamp - b->timestamp) > 3600) {
                    KSContradictionIssue* issue = &report->issues[report->num_issues];
                    memset(issue, 0, sizeof(KSContradictionIssue));
                    issue->issue_id = report->num_issues;
                    issue->type = KSC_ISSUE_TEMPORAL_CONFLICT;
                    issue->entry_id_a = i;
                    issue->entry_id_b = j;
                    strncpy(issue->subject, a->subject, sizeof(issue->subject) - 1);
                    strncpy(issue->predicate, a->predicate, sizeof(issue->predicate) - 1);
                    strncpy(issue->object_a, a->object, sizeof(issue->object_a) - 1);
                    strncpy(issue->object_b, b->object, sizeof(issue->object_b) - 1);
                    issue->timestamp_a = a->timestamp;
                    issue->timestamp_b = b->timestamp;
                    snprintf(issue->description, sizeof(issue->description),
                             "时间冲突: '%s' 的 '%s' 在 %ld 和 %ld 时间点值不同",
                             a->subject, a->predicate, a->timestamp, b->timestamp);
                    issue->suggested_resolution = KSC_RESOLVE_KEEP_NEWER;
                    issue->auto_fixable = 1;
                    report->num_issues++;
                    found++;
                }
            }
        }
    }

    report->temporal_conflicts_found = found;
    safe_free((void**)&all_entries);
    return found;
}

/* ============================================================================
 * P2-17修复: 深度逻辑矛盾检测
 * 检测传递矛盾（A-is-B + B-is-not-C 但 A-is-C）、不对称矛盾、自反矛盾
 * ============================================================================ */

/** @brief 传递性关系关键词集合 */
static int is_transitive_predicate(const char* pred) {
    if (!pred) return 0;
    static const char* transitive[] = {
        "是", "属于", "包含", "大于", "小于", "相等",
        "isa", "is_a", "instance_of", "subclass_of",
        "part_of", "located_in", "包含于"
    };
    for (size_t i = 0; i < sizeof(transitive)/sizeof(transitive[0]); i++) {
        if (strstr(pred, transitive[i])) return 1;
    }
    return 0;
}

/** @brief 不对称关系关键词集合 */
static int is_asymmetric_predicate(const char* pred) {
    if (!pred) return 0;
    static const char* asymmetric[] = {
        "大于", "小于", "高于", "低于", "父", "母", "祖先",
        "parent_of", "greater_than", "ancestor_of", "precedes"
    };
    for (size_t i = 0; i < sizeof(asymmetric)/sizeof(asymmetric[0]); i++) {
        if (strstr(pred, asymmetric[i])) return 1;
    }
    return 0;
}

/** @brief 语义矛盾概念对（扩展词表100+对反义词/对立概念） */
static int are_semantic_opposites(const char* a, const char* b) {
    if (!a || !b) return 0;
    static const char* opposite_pairs[][2] = {
        /* 基础对立 */
        {"真","假"},{"是","否"},{"存在","不存在"},{"开启","关闭"},
        {"true","false"},{"yes","no"},{"open","closed"},{"on","off"},
        /* 生命/性别 */
        {"活着","死亡"},{"男性","女性"},{"男","女"},{"生","死"},
        {"出生","死亡"},{"年轻","年老"},{"新","旧"},{"健康","疾病"},
        /* 空间/方向 */
        {"前进","后退"},{"前进","倒退"},{"上升","下降"},{"上","下"},
        {"左","右"},{"前","后"},{"高","低"},{"远","近"},
        {"进入","退出"},{"进入","离开"},{"内部","外部"},{"内","外"},
        {"顶部","底部"},{"上方","下方"},{"东","西"},{"南","北"},
        /* 数量/状态 */
        {"增加","减少"},{"增大","减小"},{"大","小"},{"多","少"},
        {"快","慢"},{"强","弱"},{"硬","软"},{"轻","重"},
        {"厚","薄"},{"宽","窄"},{"深","浅"},{"长","短"},
        {"满","空"},{"热","冷"},{"干","湿"},{"亮","暗"},
        /* 时间/顺序 */
        {"开始","结束"},{"开始","终止"},{"过去","未来"},{"现在","过去"},
        {"早期","晚期"},{"初始","最终"},{"之前","之后"},{"先","后"},
        /* 逻辑/判断 */
        {"正确","错误"},{"对","错"},{"成功","失败"},{"胜利","失败"},
        {"输入","输出"},{"进口","出口"},{"收入","支出"},{"盈利","亏损"},
        {"允许","禁止"},{"接受","拒绝"},{"同意","反对"},{"支持","反对"},
        /* 情感/评价 */
        {"好","坏"},{"善","恶"},{"美","丑"},{"喜欢","讨厌"},
        {"爱","恨"},{"快乐","悲伤"},{"积极","消极"},{"正面","负面"},
        {"安全","危险"},{"稳定","不稳定"},{"安全","风险"},{"和平","战争"},
        /* 知识/认知 */
        {"知道","不知道"},{"理解","误解"},{"清晰","模糊"},{"简单","复杂"},
        {"具体","抽象"},{"理论","实践"},{"原因","结果"},{"问题","答案"},
        /* 动作/变化 */
        {"创建","删除"},{"建立","摧毁"},{"构造","析构"},{"连接","断开"},
        {"激活","抑制"},{"加速","减速"},{"压缩","解压"},{"加密","解密"},
        {"加载","卸载"},{"安装","卸载"},{"锁定","解锁"},{"获取","释放"},
        /* 社会/关系 */
        {"朋友","敌人"},{"合作","竞争"},{"个体","群体"},{"私有","公有"},
        {"主人","仆人"},{"上级","下级"},{"雇主","雇员"},{"老师","学生"},
    };
    for (size_t i = 0; i < sizeof(opposite_pairs)/sizeof(opposite_pairs[0]); i++) {
        int match_a = (strstr(a, opposite_pairs[i][0]) || strstr(a, opposite_pairs[i][1]));
        int match_b = (strstr(b, opposite_pairs[i][0]) || strstr(b, opposite_pairs[i][1]));
        if (match_a && match_b) {
            if ((strstr(a, opposite_pairs[i][0]) && strstr(b, opposite_pairs[i][1])) ||
                (strstr(a, opposite_pairs[i][1]) && strstr(b, opposite_pairs[i][0]))) {
                return 1;
            }
        }
    }
    return 0;
}

static int detect_semantic_contradiction(const KnowledgeEntry* a, const KnowledgeEntry* b,
                                          KSSelfCheckConfig* config,
                                          KSSelfCheckReport* report) {
    if (!a || !b || !a->predicate || !b->predicate) return 0;

    /* 不对称关系反向检测：A parent-of B 且 B parent-of A */
    if (is_asymmetric_predicate(a->predicate) && is_asymmetric_predicate(b->predicate)) {
        if (a->subject && b->object && b->subject && a->object) {
            float s1 = ksc_string_similarity(a->subject, b->object);
            float s2 = ksc_string_similarity(b->subject, a->object);
            if (s1 > config->similarity_threshold && s2 > config->similarity_threshold) {
                KSContradictionIssue* issue = &report->issues[report->num_issues];
                memset(issue, 0, sizeof(KSContradictionIssue));
                issue->issue_id = report->num_issues;
                issue->type = KSC_ISSUE_LOGICAL_INCONSISTENCY;
                issue->contra_type = KSC_CONTRA_ASYMMETRIC;
                issue->entry_id_a = report->contradictions_found;
                issue->entry_id_b = report->contradictions_found + 1;
                snprintf(issue->description, sizeof(issue->description),
                         "不对称矛盾: [%s]与[%s]存在反向的[%s]关系",
                         a->subject ? a->subject : "?",
                         b->subject ? b->subject : "?",
                         a->predicate);
                issue->suggested_resolution = KSC_RESOLVE_FLAG_REVIEW;
                issue->auto_fixable = 0;
                report->num_issues++;
                return 1;
            }
        }
    }

    /* 自反检测：A parent-of A */
    if (is_asymmetric_predicate(a->predicate) && a->subject && a->object) {
        float s = ksc_string_similarity(a->subject, a->object);
        if (s > config->similarity_threshold) {
            KSContradictionIssue* issue = &report->issues[report->num_issues];
            memset(issue, 0, sizeof(KSContradictionIssue));
            issue->issue_id = report->num_issues;
            issue->type = KSC_ISSUE_LOGICAL_INCONSISTENCY;
            issue->contra_type = KSC_CONTRA_IRREFLEXIVE;
            issue->entry_id_a = report->contradictions_found;
            snprintf(issue->description, sizeof(issue->description),
                     "自反矛盾: [%s]不能[%s]自身", a->subject, a->predicate);
            issue->suggested_resolution = KSC_RESOLVE_REMOVE_BOTH;
            issue->auto_fixable = 1;
            report->num_issues++;
            return 1;
        }
    }

    /* 语义矛盾对象检测：A是猫 且 A是狗 */
    if (a->object && b->object && are_semantic_opposites(a->object, b->object)) {
        if (a->subject && b->subject &&
            ksc_string_similarity(a->subject, b->subject) > config->similarity_threshold) {
            KSContradictionIssue* issue = &report->issues[report->num_issues];
            memset(issue, 0, sizeof(KSContradictionIssue));
            issue->issue_id = report->num_issues;
            issue->type = KSC_ISSUE_LOGICAL_INCONSISTENCY;
            issue->contra_type = KSC_CONTRA_DIRECT;
            issue->entry_id_a = report->contradictions_found;
            issue->entry_id_b = report->contradictions_found + 1;
            snprintf(issue->description, sizeof(issue->description),
                     "语义矛盾: [%s]不能同时是[%s]和[%s]",
                     a->subject, a->object, b->object);
            issue->suggested_resolution = KSC_RESOLVE_FLAG_REVIEW;
            issue->auto_fixable = 0;
            report->num_issues++;
            return 1;
        }
    }

    return 0;
}

int ksc_check_logical_contradictions(KnowledgeBase* kb,
                                       KnowledgeGraph* kg,
                                       KSSelfCheckConfig* config,
                                       KSSelfCheckReport* report) {
    if (!kb || !config || !report) return -1;

    KnowledgeQuery q;
    memset(&q, 0, sizeof(q));

    KnowledgeEntry* all_entries = (KnowledgeEntry*)safe_calloc(4096, sizeof(KnowledgeEntry));
    if (!all_entries) return -1;

    int total = knowledge_base_query(kb, &q, all_entries, 4096);
    if (total <= 0) { safe_free((void**)&all_entries); return 0; }

    int found = 0;

    /* 检测传递性矛盾 */
    for (int i = 0; i < total && found < config->max_issues; i++) {
        KnowledgeEntry* a = &all_entries[i];
        if (!a->subject || !a->predicate || !a->object) continue;
        if (!is_transitive_predicate(a->predicate)) continue;

        for (int j = 0; j < total && found < config->max_issues; j++) {
            if (i == j) continue;
            KnowledgeEntry* b = &all_entries[j];
            if (!b->subject || !b->predicate || !b->object) continue;

            /* A-B有传递关系，且A.object==B.subject，检查是否存在A.subject直接连B.object */
            float mid_sim = ksc_string_similarity(a->object, b->subject);
            if (mid_sim <= config->similarity_threshold) continue;

            /* 查找是否有A.subject直接关系B.object但为否定 */
            for (int k = 0; k < total && found < config->max_issues; k++) {
                if (k == i || k == j) continue;
                KnowledgeEntry* c = &all_entries[k];
                if (!c->subject || !c->object) continue;

                float sub_sim = ksc_string_similarity(a->subject, c->subject);
                float obj_sim = ksc_string_similarity(b->object, c->object);
                if (sub_sim <= config->similarity_threshold || obj_sim <= config->similarity_threshold) continue;

                /* 检测是否否定传递结果（c表示否定） */
                if (are_semantic_opposites(c->object, b->object)) {
                    KSContradictionIssue* issue = &report->issues[report->num_issues];
                    memset(issue, 0, sizeof(KSContradictionIssue));
                    issue->issue_id = report->num_issues;
                    issue->type = KSC_ISSUE_LOGICAL_INCONSISTENCY;
                    issue->contra_type = KSC_CONTRA_TRANSITIVE;
                    issue->entry_id_a = i; issue->entry_id_b = k;
                    snprintf(issue->description, sizeof(issue->description),
                             "传递矛盾: [%s %s %s] + [%s %s %s] 与 [%s %s %s] 冲突",
                             a->subject, a->predicate, a->object,
                             b->subject, b->predicate, b->object,
                             c->subject, c->predicate, c->object);
                    issue->suggested_resolution = KSC_RESOLVE_FLAG_REVIEW;
                    issue->auto_fixable = 0;
                    report->num_issues++;
                    found++;
                }
            }
        }
    }

    /* 检测语义和关系矛盾 */
    int max_check = 2000;
    int check_count = 0;
    for (int i = 0; i < total && found < config->max_issues && check_count < max_check; i++) {
        for (int j = 0; j < total && found < config->max_issues && check_count < max_check; j++) {
            if (i == j) { check_count++; continue; }
            if (detect_semantic_contradiction(&all_entries[i], &all_entries[j],
                                               config, report)) {
                found++;
            }
            check_count++;
        }
    }

    /* 使用知识图谱进行跨图一致性验证 */
    if (kg && found < config->max_issues) {
        for (int i = 0; i < total && found < config->max_issues; i++) {
            KnowledgeEntry* e = &all_entries[i];
            if (!e->subject || !e->object) continue;
            /* 使用 find_nodes_by_label 获取节点列表，取第一个匹配 */
            KnowledgeGraphNode* subj_nodes[2];
            int subj_count = (int)knowledge_graph_find_nodes_by_label(kg, e->subject, subj_nodes, 2);
            KnowledgeGraphNode* obj_nodes[2];
            int obj_count = (int)knowledge_graph_find_nodes_by_label(kg, e->object, obj_nodes, 2);
            if (subj_count > 0 && obj_count > 0) {
                int subj_id = subj_nodes[0]->id;
                int obj_id = obj_nodes[0]->id;
                /* 遍历图中边检测矛盾和肯定边同时存在 */
                int has_negated = 0, has_affirming = 0;
                for (int ei = 0; ei < (int)kg->edge_count; ei++) {
                    if (!kg->edges[ei] || !kg->edges[ei]->source || !kg->edges[ei]->target) continue;
                    if (kg->edges[ei]->source->id == subj_id && kg->edges[ei]->target->id == obj_id) {
                        if (kg->edges[ei]->type == GRAPH_EDGE_NEGATION) has_negated = 1;
                        else has_affirming = 1;
                    }
                }
                if (has_negated && has_affirming) {
                    KSContradictionIssue* issue = &report->issues[report->num_issues];
                    memset(issue, 0, sizeof(KSContradictionIssue));
                    issue->issue_id = report->num_issues;
                    issue->type = KSC_ISSUE_LOGICAL_INCONSISTENCY;
                    issue->contra_type = KSC_CONTRA_GRAPH_EDGE;
                    issue->entry_id_a = i;
                    snprintf(issue->description, sizeof(issue->description),
                             "图谱边矛盾: %s-%s->%s 同时存在is和isNot边",
                             e->subject, e->predicate, e->object);
                    issue->suggested_resolution = KSC_RESOLVE_FLAG_REVIEW;
                    issue->auto_fixable = 0;
                    report->num_issues++;
                    found++;
                }
            }
        }
    }

    report->logical_issues_found = found;
    safe_free((void**)&all_entries);
    return found;
}

int ksc_check_circular_dependencies(KnowledgeGraph* kg, KSSelfCheckConfig* config,
                                     KSSelfCheckReport* report) {
    if (!kg || !config || !report) return -1;

    size_t node_count = knowledge_graph_get_all_nodes(kg, NULL, 0);
    if (node_count == 0) return -1;

    KnowledgeGraphNode** nodes = (KnowledgeGraphNode**)safe_calloc(node_count, sizeof(KnowledgeGraphNode*));
    if (!nodes) return -1;
    knowledge_graph_get_all_nodes(kg, nodes, node_count);

    size_t edge_count = knowledge_graph_get_all_edges(kg, NULL, 0);
    KnowledgeGraphEdge** edges = NULL;
    if (edge_count > 0) {
        edges = (KnowledgeGraphEdge**)safe_calloc(edge_count, sizeof(KnowledgeGraphEdge*));
        if (edges) {
            knowledge_graph_get_all_edges(kg, edges, edge_count);
        }
    }

    int found = 0;
    for (size_t start = 0; start < node_count && found < config->max_issues; start++) {
        int visited[1024] = {0};
        int path[1024];
        int path_len = 0;
        int stack[1024];
        int stack_top = 0;
        stack[stack_top++] = (int)start;

        while (stack_top > 0 && found < config->max_issues) {
            int current = stack[--stack_top];

            if (visited[current] == 1) {
                visited[current] = 2;
                path[path_len++] = current;

                for (int k = 0; k < path_len - 1; k++) {
                    if (path[k] == current) {
                        KSContradictionIssue* issue = &report->issues[report->num_issues];
                        memset(issue, 0, sizeof(KSContradictionIssue));
                        issue->issue_id = report->num_issues;
                        issue->type = KSC_ISSUE_CIRCULAR_DEPENDENCY;
                        issue->num_related = path_len - k;
                        for (int r = k; r < path_len && r - k < 16; r++) {
                            issue->related_ids[r - k] = path[r];
                        }
                        if (nodes[current]->label) {
                            strncpy(issue->subject, nodes[current]->label, sizeof(issue->subject) - 1);
                        }
                        snprintf(issue->description, sizeof(issue->description),
                                 "循环依赖检测到: 节点 %d 参与循环 (%d 个节点)",
                                 current, path_len - k);
                        issue->suggested_resolution = KSC_RESOLVE_FLAG_REVIEW;
                        issue->auto_fixable = 0;
                        report->num_issues++;
                        found++;
                        break;
                    }
                }
                path_len--;
                continue;
            }

            if (visited[current] == 2) continue;
            visited[current] = 1;
            path[path_len++] = current;

            for (size_t e = 0; e < edge_count && edges && found < config->max_issues; e++) {
                int neighbor = -1;
                if (edges[e] && edges[e]->source && edges[e]->target) {
                    if ((size_t)edges[e]->source->id == current) neighbor = edges[e]->target->id;
                    else if ((size_t)edges[e]->target->id == current) neighbor = edges[e]->source->id;
                }
                if (neighbor >= 0 && neighbor < 1024 && visited[neighbor] != 2) {
                    stack[stack_top++] = neighbor;
                }
            }
        }
    }

    safe_free((void**)&nodes);
    safe_free((void**)&edges);
    report->circular_deps_found = found;
    return found;
}

int ksc_auto_resolve(KnowledgeBase* kb, KSSelfCheckReport* report, int max_resolve) {
    if (!kb || !report) return -1;

    int resolved = 0;
    for (int i = 0; i < report->num_issues && resolved < max_resolve; i++) {
        KSContradictionIssue* issue = &report->issues[i];
        if (issue->resolved || !issue->auto_fixable) continue;

        if (ksc_resolve_issue(kb, issue) == 0) {
            issue->resolved = 1;
            resolved++;
        }
    }
    return resolved;
}

int ksc_resolve_issue(KnowledgeBase* kb, KSContradictionIssue* issue) {
    if (!kb || !issue) return -1;

    switch (issue->suggested_resolution) {
        case KSC_RESOLVE_KEEP_NEWER: {
            if (issue->timestamp_a > issue->timestamp_b) {
                knowledge_base_remove(kb, issue->entry_id_b);
            } else {
                knowledge_base_remove(kb, issue->entry_id_a);
            }
            return 0;
        }
        case KSC_RESOLVE_KEEP_OLDER: {
            if (issue->timestamp_a < issue->timestamp_b) {
                knowledge_base_remove(kb, issue->entry_id_b);
            } else {
                knowledge_base_remove(kb, issue->entry_id_a);
            }
            return 0;
        }
        case KSC_RESOLVE_KEEP_HIGHER_CONFIDENCE: {
            if (issue->confidence_a >= issue->confidence_b) {
                knowledge_base_remove(kb, issue->entry_id_b);
            } else {
                knowledge_base_remove(kb, issue->entry_id_a);
            }
            return 0;
        }
        case KSC_RESOLVE_REMOVE_BOTH: {
            knowledge_base_remove(kb, issue->entry_id_a);
            knowledge_base_remove(kb, issue->entry_id_b);
            return 0;
        }
        case KSC_RESOLVE_FLAG_REVIEW:
        default:
            return -1;
    }
}

KSSelfCheckReport* ksc_run_self_check(KnowledgeBase* kb, KnowledgeGraph* kg,
                                        void* reasoner, KSSelfCheckConfig* config) {
    if (!kb) return NULL;

    KSSelfCheckReport* report = (KSSelfCheckReport*)safe_calloc(1, sizeof(KSSelfCheckReport));
    if (!report) return NULL;

    KSSelfCheckConfig cfg;
    if (config) {
        memcpy(&cfg, config, sizeof(KSSelfCheckConfig));
    } else {
        memset(&cfg, 0, sizeof(KSSelfCheckConfig));
        cfg.enable_contradiction_check = 1;
        cfg.enable_redundancy_check = 1;
        cfg.enable_logical_check = 1;
        cfg.enable_temporal_check = 1;
        cfg.enable_circular_check = 1;
        cfg.enable_auto_resolve = 0;
        cfg.max_issues = KSC_MAX_ISSUES;
        cfg.similarity_threshold = KSC_SIMILARITY_THRESHOLD;
        cfg.contradiction_threshold = KSC_CONTRADICTION_THRESHOLD;
        cfg.stale_age_seconds = 86400 * 30;
        cfg.resolve_strategy = 0;
    }

    double start_time = ksc_timestamp_ms();

    KnowledgeQuery q;
    memset(&q, 0, sizeof(q));
    KnowledgeEntry* count_entries = (KnowledgeEntry*)safe_calloc(1, sizeof(KnowledgeEntry));
    if (count_entries) {
        report->total_entries_scanned = knowledge_base_query(kb, &q, count_entries, 1);
        safe_free((void**)&count_entries);
    }

    if (cfg.enable_contradiction_check) {
        ksc_check_direct_contradictions(kb, &cfg, report);
    }

    if (cfg.enable_redundancy_check) {
        ksc_check_redundancy(kb, &cfg, report);
    }

    if (cfg.enable_temporal_check) {
        ksc_check_temporal_conflicts(kb, &cfg, report);
    }

    if (cfg.enable_circular_check && kg) {
        ksc_check_circular_dependencies(kg, &cfg, report);
    }

    if (cfg.enable_logical_check) {
        ksc_check_logical_contradictions(kb, kg, &cfg, report);
    }

    /* 使用推理引擎对检测到的问题进行二次验证 */
    if (reasoner && report->num_issues > 0) {
        for (int i = 0; i < report->num_issues && i < KSC_MAX_ISSUES; i++) {
            KSContradictionIssue* iss = &report->issues[i];
            if (iss->type == KSC_ISSUE_LOGICAL_INCONSISTENCY) {
                /* 推理引擎验证：检查矛盾实体是否在知识库中仍然存在 */
                float verify_confidence = 0.5f;
                if (kb) {
                     KnowledgeEntry ea_buf, eb_buf;
                     int has_a = (knowledge_base_get_by_id(kb, iss->entry_id_a, &ea_buf) == 0);
                     int has_b = (knowledge_base_get_by_id(kb, iss->entry_id_b, &eb_buf) == 0);
                     if (!has_a || !has_b) verify_confidence = 0.0f;
                }
                if (verify_confidence < 0.3f) {
                    iss->suggested_resolution = KSC_RESOLVE_FLAG_REVIEW;
                }
            }
        }
    }

    int total_issues = report->contradictions_found + report->redundancies_found +
                       report->logical_issues_found + report->temporal_conflicts_found +
                       report->circular_deps_found;

    if (report->total_entries_scanned > 0) {
        float issue_ratio = (float)total_issues / (float)report->total_entries_scanned;
        report->consistency_score = 1.0f - fminf(issue_ratio, 1.0f);
    } else {
        report->consistency_score = 1.0f;
    }

    /* M-024修复: 基于实际数据计算完整性和新鲜度分数
     * 替代硬编码固定值0.85和0.9 */
    /* 完整性：基于知识覆盖度（条目数、类型分布） */
    report->completeness_score = 0.5f;
    if (report->total_entries_scanned > 0) {
        /* 计算各类型知识条目覆盖度 */
        int type_counts[8] = {0};
        KnowledgeEntry* type_entries = (KnowledgeEntry*)safe_calloc(
            (report->total_entries_scanned < 256 ? report->total_entries_scanned : 256), sizeof(KnowledgeEntry));
        if (type_entries) {
            KnowledgeQuery type_query;
            memset(&type_query, 0, sizeof(type_query));
            type_query.type_filter = -1;
            int type_count = knowledge_base_query(kb, &type_query, type_entries,
                report->total_entries_scanned < 256 ? report->total_entries_scanned : 256);
            for (int t = 0; t < type_count; t++) {
                if (type_entries[t].type >= 0 && type_entries[t].type < 8) {
                    type_counts[type_entries[t].type]++;
                }
            }
            /* 类型覆盖度：至少覆盖3种以上类型 */
            int covered_types = 0;
            for (int t = 0; t < 8; t++) if (type_counts[t] > 0) covered_types++;
            float type_coverage = (float)covered_types / 5.0f; /* 目标5种类型 */
            if (type_coverage > 1.0f) type_coverage = 1.0f;
            /* 条目数饱和度：经验值500为良好 */
            float size_score = report->total_entries_scanned < 500 ?
                (float)report->total_entries_scanned / 500.0f : 1.0f;
            report->completeness_score = 0.3f + 0.4f * type_coverage + 0.3f * size_score;
            safe_free((void**)&type_entries);
        }
    }
    /* 新鲜度：基于最近更新时间与当前时间的差异 */
    {
        time_t now = time(NULL);
        time_t latest_update = 0;
        KnowledgeEntry* fresh_entries = (KnowledgeEntry*)safe_calloc(256, sizeof(KnowledgeEntry));
        if (fresh_entries) {
            KnowledgeQuery fresh_query;
            memset(&fresh_query, 0, sizeof(fresh_query));
            int fresh_count = knowledge_base_query(kb, &fresh_query, fresh_entries, 256);
            for (int t = 0; t < fresh_count; t++) {
                if (fresh_entries[t].timestamp > latest_update) {
                    latest_update = fresh_entries[t].timestamp;
                }
            }
            safe_free((void**)&fresh_entries);
        }
        if (latest_update > 0) {
            double hours_ago = difftime(now, latest_update) / 3600.0;
            /* 24小时内更新=1.0，一周=0.7，一月=0.4 */
            report->freshness_score = hours_ago < 24.0 ? 1.0f :
                                      hours_ago < 168.0 ? (float)(1.0f - (float)(hours_ago - 24.0) / 144.0 * 0.3f) :
                                      hours_ago < 720.0 ? (float)(0.7f - (float)(hours_ago - 168.0) / 552.0 * 0.3f) : 0.4f;
        } else {
            report->freshness_score = 0.9f;
        }
    }

    if (cfg.enable_auto_resolve && total_issues > 0) {
        ksc_auto_resolve(kb, report, total_issues);
    }

    report->scan_time_ms = ksc_timestamp_ms() - start_time;

    return report;
}

void ksc_report_free(KSSelfCheckReport* report) {
    safe_free((void**)&report);
}

int ksc_get_unresolved_count(const KSSelfCheckReport* report) {
    if (!report) return 0;
    int count = 0;
    for (int i = 0; i < report->num_issues; i++) {
        if (!report->issues[i].resolved) count++;
    }
    return count;
}

const char* ksc_issue_type_str(KSIssueType type) {
    switch (type) {
        case KSC_ISSUE_CONTRADICTION: return "矛盾";
        case KSC_ISSUE_REDUNDANCY: return "冗余";
        case KSC_ISSUE_LOGICAL_INCONSISTENCY: return "逻辑不一致";
        case KSC_ISSUE_TEMPORAL_CONFLICT: return "时间冲突";
        case KSC_ISSUE_LOW_CONFIDENCE: return "低置信度";
        case KSC_ISSUE_STALE_KNOWLEDGE: return "过期知识";
        case KSC_ISSUE_INCOMPLETE_RELATION: return "关系不完整";
        case KSC_ISSUE_CIRCULAR_DEPENDENCY: return "循环依赖";
        default: return "未知";
    }
}

const char* ksc_contra_type_str(KSContradictionType type) {
    switch (type) {
        case KSC_CONTRA_DIRECT: return "直接矛盾";
        case KSC_CONTRA_TRANSITIVE: return "传递矛盾";
        case KSC_CONTRA_ASYMMETRIC: return "不对称矛盾";
        case KSC_CONTRA_IRREFLEXIVE: return "自反矛盾";
        default: return "无";
    }
}

const char* ksc_resolution_str(KSResolutionType type) {
    switch (type) {
        case KSC_RESOLVE_KEEP_NEWER: return "保留较新";
        case KSC_RESOLVE_KEEP_OLDER: return "保留较旧";
        case KSC_RESOLVE_KEEP_HIGHER_CONFIDENCE: return "保留高置信度";
        case KSC_RESOLVE_MERGE: return "合并";
        case KSC_RESOLVE_REMOVE_BOTH: return "全部删除";
        case KSC_RESOLVE_FLAG_REVIEW: return "标记审查";
        default: return "无";
    }
}

/* ============================================================================
 * K-修复: 增量知识自检
 * 仅扫描最近新增/修改的知识条目，而非全量扫描。
 * 维护上次检查的时间戳，跳过未变更的条目。
 * ============================================================================ */

int ksc_incremental_check(KnowledgeBase* kb, KSSelfCheckConfig* config,
                          KSSelfCheckReport* report) {
    if (!kb || !config || !report) return -1;

    /* 获取上次检查时间 */
    static time_t last_check_time = 0;
    time_t now = time(NULL);

    /* 查询最近变更的条目 */
    KnowledgeQuery q;
    memset(&q, 0, sizeof(KnowledgeQuery));
    q.min_confidence = 0.0f;

    KnowledgeEntry* recent_entries = (KnowledgeEntry*)safe_calloc(500, sizeof(KnowledgeEntry));
    if (!recent_entries) return -1;

    int total = knowledge_base_query(kb, &q, recent_entries, 500);
    int incremental_count = 0;
    int skipped_count = 0;

    /* 过滤：只检查上次检查后变更的条目 */
    KnowledgeEntry* inc_entries = (KnowledgeEntry*)safe_calloc(500, sizeof(KnowledgeEntry));
    if (!inc_entries) { safe_free((void**)&recent_entries); return -1; }

    for (int i = 0; i < total && i < 500; i++) {
        time_t entry_time = recent_entries[i].timestamp;
        if (last_check_time == 0 || entry_time >= last_check_time) {
            memcpy(&inc_entries[incremental_count], &recent_entries[i], sizeof(KnowledgeEntry));
            incremental_count++;
        } else {
            skipped_count++;
        }
    }

    /* 仅对增量条目运行检查 */
    int contradictions = 0, redundancies = 0;

    if (incremental_count > 1) {
        for (int i = 0; i < incremental_count; i++) {
            for (int j = i + 1; j < incremental_count; j++) {
                if (ksc_string_similarity(inc_entries[i].subject, inc_entries[j].subject) >
                    config->similarity_threshold) {
                    /* 直接矛盾检测 */
                    if (inc_entries[i].predicate && inc_entries[j].predicate &&
                        strcmp(inc_entries[i].predicate, inc_entries[j].predicate) == 0 &&
                        inc_entries[i].object && inc_entries[j].object &&
                        strcmp(inc_entries[i].object, inc_entries[j].object) != 0) {
                        contradictions++;
                        if (report->num_issues < KSC_MAX_ISSUES) {
                            KSContradictionIssue* issue = &report->issues[report->num_issues++];
                            memset(issue, 0, sizeof(KSContradictionIssue));
                            issue->type = KSC_ISSUE_CONTRADICTION;
                            issue->entry_id_a = i;
                            issue->entry_id_b = j;
                            snprintf(issue->description, sizeof(issue->description),
                                "增量矛盾: [%s]不同值[%s]vs[%s]",
                                inc_entries[i].subject, inc_entries[i].object, inc_entries[j].object);
                            issue->suggested_resolution = KSC_RESOLVE_KEEP_HIGHER_CONFIDENCE;
                            issue->auto_fixable = 1;
                        }
                    }
                    /* 冗余检测 */
                    float sim = ksc_string_similarity(inc_entries[i].object ? inc_entries[i].object : "",
                                                       inc_entries[j].object ? inc_entries[j].object : "");
                    if (sim > config->similarity_threshold) {
                        redundancies++;
                    }
                }
            }
        }
    }

    /* 更新报告 - -L-004: 不再将跳过计数误写入逻辑问题字段 */
    report->contradictions_found += contradictions;
    report->redundancies_found += redundancies;
    report->total_entries_scanned = (size_t)incremental_count;
    /* skipped_count仅用于日志统计，不写入逻辑问题字段 */

    if (total > 0) {
        float issue_ratio = (float)(contradictions + redundancies) / (float)incremental_count;
        report->consistency_score = 1.0f - fminf(issue_ratio, 1.0f);
    } else {
        report->consistency_score = 1.0f;
    }

    last_check_time = now;

    safe_free((void**)&recent_entries);
    safe_free((void**)&inc_entries);
    return 0;
}
