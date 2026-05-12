#ifndef SELFLNN_KNOWLEDGE_SELF_CHECK_H
#define SELFLNN_KNOWLEDGE_SELF_CHECK_H

#include "selflnn/knowledge/knowledge.h"
#include "selflnn/knowledge/knowledge_graph.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KSC_MAX_ISSUES 1024
#define KSC_MAX_PATH_LEN 64
#define KSC_SIMILARITY_THRESHOLD 0.85f
#define KSC_CONTRADICTION_THRESHOLD 0.15f

typedef enum {
    KSC_CONTRA_NONE = 0,
    KSC_CONTRA_DIRECT,
    KSC_CONTRA_TRANSITIVE,
    KSC_CONTRA_ASYMMETRIC,
    KSC_CONTRA_IRREFLEXIVE
} KSContradictionType;

typedef enum {
    KSC_ISSUE_NONE = 0,
    KSC_ISSUE_CONTRADICTION,
    KSC_ISSUE_REDUNDANCY,
    KSC_ISSUE_LOGICAL_INCONSISTENCY,
    KSC_ISSUE_TEMPORAL_CONFLICT,
    KSC_ISSUE_LOW_CONFIDENCE,
    KSC_ISSUE_STALE_KNOWLEDGE,
    KSC_ISSUE_INCOMPLETE_RELATION,
    KSC_ISSUE_CIRCULAR_DEPENDENCY
} KSIssueType;

typedef enum {
    KSC_RESOLVE_NONE = 0,
    KSC_RESOLVE_KEEP_NEWER,
    KSC_RESOLVE_KEEP_OLDER,
    KSC_RESOLVE_KEEP_HIGHER_CONFIDENCE,
    KSC_RESOLVE_MERGE,
    KSC_RESOLVE_REMOVE_BOTH,
    KSC_RESOLVE_FLAG_REVIEW
} KSResolutionType;

typedef struct {
    int issue_id;
    KSIssueType type;
    KSContradictionType contra_type;
    int entry_id_a;
    int entry_id_b;
    int related_ids[16];
    int num_related;
    char subject[256];
    char predicate[256];
    char object_a[256];
    char object_b[256];
    float confidence_a;
    float confidence_b;
    float similarity_score;
    long timestamp_a;
    long timestamp_b;
    char description[512];
    KSResolutionType suggested_resolution;
    float resolution_score;
    int auto_fixable;
    int resolved;
} KSContradictionIssue;

typedef struct {
    KSContradictionIssue issues[KSC_MAX_ISSUES];
    int num_issues;
    double scan_time_ms;
    int total_entries_scanned;
    int contradictions_found;
    int redundancies_found;
    int logical_issues_found;
    int temporal_conflicts_found;
    int circular_deps_found;
    float consistency_score;
    float completeness_score;
    float freshness_score;
} KSSelfCheckReport;

typedef struct {
    int enable_contradiction_check;
    int enable_redundancy_check;
    int enable_logical_check;
    int enable_temporal_check;
    int enable_circular_check;
    int enable_auto_resolve;
    int max_issues;
    float similarity_threshold;
    float contradiction_threshold;
    long stale_age_seconds;
    int resolve_strategy;
} KSSelfCheckConfig;

#define KS_SELF_CHECK_CONFIG_DEFAULT { 1, 1, 1, 1, 1, 0, KSC_MAX_ISSUES, KSC_SIMILARITY_THRESHOLD, KSC_CONTRADICTION_THRESHOLD, 86400 * 30, 0 }

KSSelfCheckReport* ksc_run_self_check(KnowledgeBase* kb, KnowledgeGraph* kg,
                                        void* reasoner, KSSelfCheckConfig* config);
void ksc_report_free(KSSelfCheckReport* report);

int ksc_check_direct_contradictions(KnowledgeBase* kb, KSSelfCheckConfig* config,
                                     KSSelfCheckReport* report);
int ksc_check_redundancy(KnowledgeBase* kb, KSSelfCheckConfig* config,
                          KSSelfCheckReport* report);
int ksc_check_temporal_conflicts(KnowledgeBase* kb, KSSelfCheckConfig* config,
                                  KSSelfCheckReport* report);
int ksc_check_circular_dependencies(KnowledgeGraph* kg, KSSelfCheckConfig* config,
                                     KSSelfCheckReport* report);

int ksc_auto_resolve(KnowledgeBase* kb, KSSelfCheckReport* report, int max_resolve);
int ksc_resolve_issue(KnowledgeBase* kb, KSContradictionIssue* issue);
int ksc_get_unresolved_count(const KSSelfCheckReport* report);
const char* ksc_issue_type_str(KSIssueType type);
const char* ksc_contra_type_str(KSContradictionType type);
const char* ksc_resolution_str(KSResolutionType type);

#ifdef __cplusplus
}
#endif

#endif
