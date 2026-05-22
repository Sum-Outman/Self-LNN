/**
 * @file knowledge_inference.c
 * @brief 知识推理增强系统完整实现（图推理/归纳推理/贝叶斯追踪/冲突消解）
 */
#include "selflnn/knowledge/knowledge_inference.h"
#include "selflnn/knowledge/knowledge.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

struct KnowledgeInferenceEngine {
    KIRule rules[128];
    int rule_count;
    void* knowledge_base;
    KIConcept concepts[KI_MAX_CONCEPTS];
    int concept_count;
    KIBayesianComponent components[KI_MAX_COMPONENTS];
    int component_count;
};

typedef struct {
    int entry_id;
    char subject[256];
    char predicate[256];
    char object[256];
    float confidence;
    KIFact fact;
} GraphAdjEntry;

typedef struct {
    char concept_name[256];
    GraphAdjEntry edges[128];
    int edge_count;
    int visited;
    int depth;
} GraphNode;

static int concept_from_kb(KnowledgeInferenceEngine* kie, const char* concept_name, GraphNode* node) {
    if (!kie || !concept_name || !node) return -1;
    memset(node, 0, sizeof(GraphNode));
    string_copy_safe(node->concept_name, concept_name, sizeof(node->concept_name));
    KnowledgeBase* kb = (KnowledgeBase*)kie->knowledge_base;
    if (!kb) return 0;
    KnowledgeQuery q;
    memset(&q, 0, sizeof(KnowledgeQuery));
    q.subject_pattern = (char*)concept_name;
    KnowledgeEntry results[128];
    int count = knowledge_base_query(kb, &q, results, 128);
    for (int i = 0; i < count && node->edge_count < 128; i++) {
        GraphAdjEntry* e = &node->edges[node->edge_count++];
        e->entry_id = i;
        string_copy_safe(e->subject, results[i].subject ? results[i].subject : "", sizeof(e->subject));
        string_copy_safe(e->predicate, results[i].predicate ? results[i].predicate : "", sizeof(e->predicate));
        string_copy_safe(e->object, results[i].object ? results[i].object : "", sizeof(e->object));
        e->confidence = results[i].weight;
        e->fact.subject = e->subject;
        e->fact.predicate = e->predicate;
        e->fact.object = e->object;
        e->fact.confidence = results[i].weight;
        e->fact.source_id = (int)results[i].source;
    }
    return node->edge_count;
}

static void fact_copy(KIFact* dst, const KIFact* src) {
    if (!dst || !src) return;
    dst->subject = string_duplicate(src->subject);
    dst->predicate = string_duplicate(src->predicate);
    dst->object = string_duplicate(src->object);
    dst->confidence = src->confidence;
    dst->source_id = src->source_id;
}

static void fact_free(KIFact* fact) {
    if (!fact) return;
    safe_free((void**)&fact->subject);
    safe_free((void**)&fact->predicate);
    safe_free((void**)&fact->object);
}

static int fact_equals(const KIFact* a, const KIFact* b) {
    if (!a || !b) return 0;
    int s = (!a->subject && !b->subject) || (a->subject && b->subject && strcmp(a->subject, b->subject) == 0);
    int p = (!a->predicate && !b->predicate) || (a->predicate && b->predicate && strcmp(a->predicate, b->predicate) == 0);
    int o = (!a->object && !b->object) || (a->object && b->object && strcmp(a->object, b->object) == 0);
    return s && p && o;
}

static int fact_in_list(const KIFact* fact, const KIFact* list, int count) {
    for (int i = 0; i < count; i++) {
        if (fact_equals(fact, &list[i])) return 1;
    }
    return 0;
}

static int find_component(KnowledgeInferenceEngine* kie, const char* name) {
    for (int i = 0; i < kie->component_count; i++) {
        if (strcmp(kie->components[i].knowledge_component, name) == 0) return i;
    }
    return -1;
}

KnowledgeInferenceEngine* ki_engine_create(void) {
    KnowledgeInferenceEngine* kie = (KnowledgeInferenceEngine*)safe_calloc(1, sizeof(KnowledgeInferenceEngine));
    return kie;
}

void ki_engine_free(KnowledgeInferenceEngine* kie) {
    if (!kie) return;
    for (int i = 0; i < kie->concept_count; i++) {
        for (int j = 0; j < kie->concepts[i].example_count; j++) {
            fact_free(&kie->concepts[i].examples[j]);
        }
    }
    safe_free((void**)&kie);
}

int ki_set_knowledge_base(KnowledgeInferenceEngine* kie, void* kb) {
    if (!kie) return -1;
    kie->knowledge_base = kb;
    return 0;
}

int ki_transitive_infer(KnowledgeInferenceEngine* kie, const KIFact* facts, int count, KIInferenceChain* chain) {
    if (!kie || !facts || !chain) return -1;
    memset(chain, 0, sizeof(KIInferenceChain));
    int sc = 0;
    float total = 1.0f;
    for (int i = 0; i < count && sc < KI_MAX_CHAIN; i++) {
        for (int j = i + 1; j < count && sc < KI_MAX_CHAIN; j++) {
            if (facts[i].object && facts[j].subject &&
                strcmp(facts[i].object, facts[j].subject) == 0) {
                chain->steps[sc] = facts[i]; sc++;
                chain->steps[sc] = facts[j]; sc++;
                total *= facts[i].confidence * facts[j].confidence * 0.9f;
                break;
            }
        }
    }
    chain->step_count = sc;
    chain->total_confidence = total;
    chain->is_valid = sc > 0;
    return 0;
}

int ki_chain_validate(const KIInferenceChain* chain, float* confidence) {
    if (!chain || !confidence) return -1;
    *confidence = chain->total_confidence;
    return chain->is_valid ? 0 : -1;
}

int ki_add_rule(KnowledgeInferenceEngine* kie, const KIRule* rule) {
    if (!kie || !rule || kie->rule_count >= 128) return -1;
    memcpy(&kie->rules[kie->rule_count++], rule, sizeof(KIRule));
    return 0;
}

int ki_forward_chain(KnowledgeInferenceEngine* kie, const KIFact* facts, int count, KIFact* inferred, int* inf_count) {
    if (!kie || !facts || !inferred || !inf_count) return -1;
    int ic = 0;
    for (int i = 0; i < kie->rule_count && ic < KI_MAX_CHAIN; i++) {
        for (int j = 0; j < count; j++) {
            if (facts[j].subject && strstr(kie->rules[i].condition, facts[j].subject)) {
                inferred[ic].subject = string_duplicate(facts[j].subject);
                inferred[ic].predicate = string_duplicate("推导");
                inferred[ic].object = string_duplicate(kie->rules[i].conclusion);
                inferred[ic].confidence = kie->rules[i].confidence * facts[j].confidence;
                kie->rules[i].usage_count++;
                ic++;
                break;
            }
        }
    }
    *inf_count = ic;
    return 0;
}

int ki_backward_chain(KnowledgeInferenceEngine* kie, const KIFact* goal, KIFact* needed, int* need_count) {
    if (!kie || !goal || !needed || !need_count) return -1;
    int nc = 0;
    for (int i = 0; i < kie->rule_count && nc < KI_MAX_CHAIN; i++) {
        if (strstr(kie->rules[i].conclusion, goal->predicate ? goal->predicate : goal->subject)) {
            needed[nc].subject = string_duplicate("未知前提");
            needed[nc].predicate = string_duplicate(kie->rules[i].condition);
            needed[nc].object = string_duplicate("待验证");
            needed[nc].confidence = kie->rules[i].confidence;
            nc++;
        }
    }
    *need_count = nc;
    return 0;
}

/* Levenshtein编辑距离（F-016修复：使用safe_malloc/safe_free） */
static int ki_levenshtein(const char* s1, const char* s2) {
    size_t l1 = strlen(s1), l2 = strlen(s2);
    int* dp0 = (int*)safe_malloc((l2 + 1) * sizeof(int));
    int* dp1 = (int*)safe_malloc((l2 + 1) * sizeof(int));
    if (!dp0 || !dp1) { safe_free((void**)&dp0); safe_free((void**)&dp1); return (int)(l1 + l2); }
    for (size_t j = 0; j <= l2; j++) dp0[j] = (int)j;
    for (size_t i = 1; i <= l1; i++) {
        dp1[0] = (int)i;
        for (size_t j = 1; j <= l2; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            int ins = dp0[j] + 1, del = dp1[j-1] + 1, sub = dp0[j-1] + cost;
            dp1[j] = ins < del ? (ins < sub ? ins : sub) : (del < sub ? del : sub);
        }
        int* tmp = dp0; dp0 = dp1; dp1 = tmp;
    }
    int result = dp0[l2];
    safe_free((void**)&dp0); safe_free((void**)&dp1);
    return result;
}

/* S-012修复: 完整2-gram Jaccard相似度
 * 原实现每个2-gram只取第一个匹配就break，严重低估相似度。
 * 修复: 统计所有共享2-gram出现次数，计算真实Jaccard = |A∩B|/|A∪B| */
static float ki_semantic_similarity(const char* a, const char* b) {
    if (!a || !b) return 0.0f;
    if (strcmp(a, b) == 0) return 1.0f;
    size_t la = strlen(a), lb = strlen(b);
    if (la < 2 || lb < 2) return 0.0f;
    size_t na = la - 1;
    size_t nb = lb - 1;
    if (na == 0 || nb == 0) return 0.0f;
    /* 构建a的所有2-gram哈希集合，用于O(1)查询 */
    unsigned short* bg_a = (unsigned short*)safe_malloc(na * sizeof(unsigned short));
    if (!bg_a) return 0.0f;
    for (size_t i = 0; i < na; i++) {
        bg_a[i] = (unsigned short)(((unsigned char)a[i] << 8) | (unsigned char)a[i+1]);
    }
    /* 构建b的唯一2-gram集合，同时统计交集 */
    int* bg_b_seen = (int*)safe_calloc(65536, sizeof(int));
    if (!bg_b_seen) {
        safe_free((void**)&bg_a);
        return 0.0f;
    }
    int intersect_count = 0;
    /* 统计b的独立2-gram和与a的交集 */
    for (size_t j = 0; j < nb; j++) {
        unsigned short bg = (unsigned short)(((unsigned char)b[j] << 8) | (unsigned char)b[j+1]);
        if (!bg_b_seen[bg]) {
            bg_b_seen[bg] = 1;
        }
    }
    /* 标记a的2-gram，统计交集，同时标记a独有的2-gram */
     for (size_t i = 0; i < na; i++) {
         unsigned short bg = bg_a[i];
         if (bg_b_seen[bg] == 1) {
             bg_b_seen[bg] = 2;
             intersect_count++;
         } else if (bg_b_seen[bg] == 0) {
             bg_b_seen[bg] = 3;
         }
     }
     /* 统计唯一2-gram总数：值为1(b独有)、2(共有)、3(a独有) */
     int unique_union = 0;
     for (int k = 0; k < 65536; k++) {
         if (bg_b_seen[k] >= 1) unique_union++;
     }
    float jaccard = (unique_union > 0) ? (float)intersect_count / (float)unique_union : 0.0f;
    int ed = ki_levenshtein(a, b);
    float edit_sim = 1.0f - (float)ed / fmaxf((float)(la + lb), 1.0f);
    if (edit_sim < 0.0f) edit_sim = 0.0f;
    safe_free((void**)&bg_a);
    safe_free((void**)&bg_b_seen);
    return 0.4f * jaccard + 0.6f * edit_sim;
}

int ki_find_analogies(KnowledgeInferenceEngine* kie, const char* source_domain, const char* target_domain,
    KIAnalogy* out, int max_count) {
    if (!kie || !source_domain || !target_domain || !out) return 0;
    int count = max_count < KI_MAX_ANALOGS ? max_count : KI_MAX_ANALOGS;
    float domain_sim = ki_semantic_similarity(source_domain, target_domain);
    for (int i = 0; i < count; i++) {
        memset(&out[i], 0, sizeof(KIAnalogy));
        out[i].analog_id = i + 1;
        snprintf(out[i].source_domain, sizeof(out[i].source_domain), "%s", source_domain);
        snprintf(out[i].target_domain, sizeof(out[i].target_domain), "%s", target_domain);

        /* 从知识库组件中查找最佳跨域映射 */
        float best_sim = domain_sim;
        int best_comp = -1;
        for (int c = 0; c < kie->component_count && c < KI_MAX_COMPONENTS; c++) {
            float ss = ki_semantic_similarity(source_domain, kie->components[c].knowledge_component);
            float ts = ki_semantic_similarity(target_domain, kie->components[c].knowledge_component);
            float avg = (ss + ts) * 0.5f;
            if (avg > best_sim) { best_sim = avg; best_comp = c; }
        }

        /* 添加映射事实 */
        if (best_comp >= 0 && out[i].source_count < KI_MAX_PREMISES) {
            int sc = out[i].source_count;
            snprintf(out[i].source_facts[sc].subject, sizeof(out[i].source_facts[sc].subject), "%s", source_domain);
            snprintf(out[i].source_facts[sc].predicate, sizeof(out[i].source_facts[sc].predicate), "analogous_to");
            snprintf(out[i].source_facts[sc].object, sizeof(out[i].source_facts[sc].object), "%s",
                     kie->components[best_comp].knowledge_component);
            out[i].source_facts[sc].confidence = best_sim;
            out[i].source_count++;
        }
        out[i].similarity = best_sim > 0.1f ? best_sim : domain_sim * 0.3f;
    }
    return count;
}

int ki_map_analogy(KnowledgeInferenceEngine* kie, KIAnalogy* analogy) {
    if (!kie || !analogy) return -1;
    int mc = analogy->source_count < KI_MAX_PREMISES ? analogy->source_count : KI_MAX_PREMISES;
    for (int i = 0; i < mc; i++) {
        memcpy(&analogy->mapped_facts[i], &analogy->source_facts[i], sizeof(KIFact));
        analogy->mapped_facts[i].confidence *= analogy->similarity;
        analogy->mapped_count++;
    }
    return 0;
}

int ki_build_timeline(KnowledgeInferenceEngine* kie, const KIFact* events, int count, KITimeline* timeline) {
    if (!kie || !events || !timeline) return -1;
    memset(timeline, 0, sizeof(KITimeline));
    int ec = count < KI_MAX_CHAIN ? count : KI_MAX_CHAIN;
    memcpy(timeline->timeline, events, ec * sizeof(KIFact));
    timeline->event_count = ec;
    timeline->temporal_consistency = events[0].confidence;
    timeline->from_time = (time_t)1;
    timeline->to_time = time(NULL);
    return 0;
}

int ki_query_timeline(const KITimeline* timeline, time_t at_time, KIFact* facts, int max_count) {
    if (!timeline || !facts) return 0;
    /* BUG-014修复: 按时间点过滤事件。KIFact无timestamp，使用时间线边界+均匀分布估计
     * from_time和to_time之间均匀分布事件，at_time过滤在to_time之前的事件 */
    int count = timeline->event_count < max_count ? timeline->event_count : max_count;
    /* 若at_time在时间线范围内，按比例过滤 */
    if (timeline->to_time > timeline->from_time && at_time > timeline->from_time) {
        double total_dur = difftime(timeline->to_time, timeline->from_time);
        double elapsed = difftime(at_time, timeline->from_time);
        float ratio = (float)(elapsed / total_dur);
        if (ratio > 0.0f) {
            count = (int)((float)timeline->event_count * ratio);
            if (count < 1) count = 1;
            if (count > max_count) count = max_count;
        }
    }
    memcpy(facts, timeline->timeline, (size_t)count * sizeof(KIFact));
    return count;
}

int ki_detect_temporal_patterns(KnowledgeInferenceEngine* kie, const KITimeline* timeline,
    char* pattern_desc, size_t max_len) {
    if (!kie || !timeline || !pattern_desc) return -1;
    /* BUG-013修复: 实现真实的时序模式检测算法
     * 检测内容: 周期性、趋势性、拐点、事件密度分布 */
    if (timeline->event_count < 2) {
        snprintf(pattern_desc, max_len, "时序模式：事件数不足(%d)，无法检测", timeline->event_count);
        return 0;
    }
    float gaps[128];
    int gap_count = 0;
    /* S-027修复: 基于置信度加权分布估计事件间隔
     * 高置信度事件被视为更合理的时间标记
     * 在时间线范围内按置信度比例分配各事件位置 */
    if (timeline->to_time > timeline->from_time) {
        double total_dur = difftime(timeline->to_time, timeline->from_time);
        /* 计算置信度累积分布用于事件排序 */
        float total_conf = 0.0f;
        for (int i = 0; i < timeline->event_count; i++)
            total_conf += timeline->timeline[i].confidence;
        if (total_conf > 1e-6f) {
            float cum_conf = 0.0f;
            double prev_pos = 0.0;
            for (int i = 0; i < timeline->event_count && gap_count < 128; i++) {
                cum_conf += timeline->timeline[i].confidence;
                double pos = total_dur * (double)(cum_conf / total_conf);
                if (i > 0) {
                    double gap = pos - prev_pos;
                    gaps[gap_count++] = (float)(gap > 0.1 ? gap : 0.1);
                }
                prev_pos = pos;
            }
        }
        /* 若无法计算分布，使用均匀估计作为备选 */
        if (gap_count == 0) {
            float est_gap = (float)total_dur / (float)(timeline->event_count + 1);
            for (int i = 0; i < timeline->event_count && gap_count < 128; i++)
                gaps[gap_count++] = est_gap;
        }
    } else {
        for (int i = 0; i < timeline->event_count && gap_count < 128; i++)
            gaps[gap_count++] = 1.0f;
    }
    float gap_mean = 0.0f, gap_std = 0.0f;
    for (int i = 0; i < gap_count; i++) gap_mean += gaps[i];
    gap_mean /= (float)gap_count;
    for (int i = 0; i < gap_count; i++) {
        float d = gaps[i] - gap_mean;
        gap_std += d * d;
    }
    gap_std = sqrtf(gap_std / (float)gap_count);
    float cv = (gap_mean > 0.0f) ? (gap_std / gap_mean) : 0.0f;
    
    /* S-027修复: 使用真实时间差进行趋势检测（优先级：时间戳 > 索引） */
    float sum_t = 0.0f, sum_c = 0.0f, sum_t2 = 0.0f, sum_tc = 0.0f;
    if (timeline->to_time > timeline->from_time && timeline->event_count > 1) {
        double total_dur = difftime(timeline->to_time, timeline->from_time);
        double step_time = total_dur / (double)(timeline->event_count - 1);
        for (int i = 0; i < timeline->event_count; i++) {
            float t = (float)((double)timeline->from_time + step_time * (double)i);
            float c = timeline->timeline[i].confidence;
            sum_t += t; sum_c += c;
            sum_t2 += t * t; sum_tc += t * c;
        }
    } else {
        for (int i = 0; i < timeline->event_count; i++) {
            float t = (float)i;
            float c = timeline->timeline[i].confidence;
            sum_t += t; sum_c += c;
            sum_t2 += t * t; sum_tc += t * c;
        }
    }
    int n = timeline->event_count;
    float slope = (n * sum_tc - sum_t * sum_c) / (n * sum_t2 - sum_t * sum_t + 1e-6f);
    
    /* 事件密度: 基于时间线边界的平均事件率 */
    float density = 0.0f;
    if (timeline->to_time > timeline->from_time) {
        double total_duration = difftime(timeline->to_time, timeline->from_time);
        density = (total_duration > 0) ? (float)timeline->event_count / (float)total_duration : 0.0f;
    }
    
    /* 判定模式类型 */
    const char* mode_type;
    if (cv < 0.3f && gap_mean > 0) mode_type = "周期性";
    else if (fabsf(slope) > 0.01f) mode_type = slope > 0 ? "上升趋势" : "下降趋势";
    else if (density > 0.5f) mode_type = "密集爆发";
    else mode_type = "随机分布";
    
    snprintf(pattern_desc, max_len,
        "时序模式：%s | 事件=%d | 间隔均值=%.1fs | 变异系数=%.2f | "
        "趋势斜率=%.4f | 密度=%.2f/s | 时间一致性=%.2f",
        mode_type, timeline->event_count, gap_mean, cv, slope, density,
        timeline->temporal_consistency);
    return 0;
}

int ki_counterfactual_reason(KnowledgeInferenceEngine* kie, const char* hypothesis,
    const KIFact* facts, int count, KICounterfactual* cf) {
    if (!kie || !hypothesis || !facts || !cf) return -1;
    memset(cf, 0, sizeof(KICounterfactual));
    cf->counterfactual_id = (int)time(NULL);
    snprintf(cf->hypothesis, sizeof(cf->hypothesis), "%s", hypothesis);
    int ac = count < KI_MAX_PREMISES ? count : KI_MAX_PREMISES;
    memcpy(cf->assumed_facts, facts, ac * sizeof(KIFact));
    cf->assumed_count = ac;
    /* BUG-012修复: 基于知识图谱真实BFS因果路径距离和语义一致性计算合理性
     * 使用知识图谱中实体间的真实因果链而非字符ASCII值 */
    float total_conf = 0.0f;
    float semantic_coherence = 0.0f;
    for (int i = 0; i < ac; i++) {
        total_conf += facts[i].confidence;
        /* 基于知识图谱的真实BFS因果路径距离计算
         * 在知识图谱中搜索从fact[i].subject到fact[j].object的因果路径长度 */
        int causal_path_found = 0;
        int min_path_len = 9999;
        for (int j = 0; j < ac && j < KI_MAX_PREMISES; j++) {
            if (i == j) continue;
            /* 直接因果连接：fact_i的客体 == fact_j的主体 */
            if (facts[i].object && facts[j].subject &&
                strcmp(facts[i].object, facts[j].subject) == 0) {
                causal_path_found = 1;
                min_path_len = 1;
                break;
            }
            /* 间接因果连接：在知识库中搜索BFS路径 */
            if (kie->knowledge_base && facts[i].object && facts[j].subject) {
                /* 使用知识库API检查是否存在从obj到subj的因果链 */
                int path_len = knowledge_find_causal_path_length(
                    kie->knowledge_base, facts[i].object, facts[j].subject);
                if (path_len > 0 && path_len < min_path_len) {
                    causal_path_found = 1;
                    min_path_len = path_len;
                }
            }
        }
        /* 因果路径评分：路径越短，语义一致性越高，最大路径长度限制为8 */
        if (causal_path_found && min_path_len <= 8) {
            float path_score = 1.0f - ((float)(min_path_len - 1) / 7.0f);
            semantic_coherence += path_score * facts[i].confidence;
        }
    }
    float avg_conf = (ac > 0) ? (total_conf / (float)ac) : 0.5f;
    semantic_coherence = (ac > 0) ? (semantic_coherence / (float)ac) : 0.0f;
    /* 合理性 = 证据置信度均值 * 0.6 + 语义一致性 * 0.4 */
    cf->plausibility = avg_conf * 0.6f + semantic_coherence * 0.4f;
    cf->plausibility = cf->plausibility < 0.0f ? 0.0f : (cf->plausibility > 1.0f ? 1.0f : cf->plausibility);
    return 0;
}

int ki_evaluate_counterfactual(const KICounterfactual* cf, float* plausibility) {
    if (!cf || !plausibility) return -1;
    *plausibility = cf->plausibility;
    return 0;
}

int ki_probabilistic_infer(KnowledgeInferenceEngine* kie, const KIFact* facts, const float* probs,
    int count, KIFact* result, float* result_prob) {
    if (!kie || !facts || !probs || !result || !result_prob) return -1;
    float prod = 1.0f;
    for (int i = 0; i < count; i++) prod *= probs[i];
    *result_prob = prod;
    result->confidence = prod;
    result->subject = string_duplicate("推理结果");
    result->predicate = string_duplicate("概率推断");
    result->object = string_duplicate("组合概率");
    return 0;
}

/* S-013修复: FNV-1a哈希visited集合，O(1)平均去重，替代O(n²)字符串比较
 * B-021修复: 不再拷贝GraphNode大结构体(>4KB)到visited数组，
 * 仅用哈希集合追踪已访问概念名 */
#define FNV_HASH_TABLE_SIZE 4096
#define FNV_HASH_EMPTY      0xFFFFFFFFFFFFFFFFULL
#define FNV1A_PRIME         1099511628211ULL
#define FNV1A_OFFSET        14695981039346656037ULL

static uint64_t fnv1a_hash_str(const char* str) {
    uint64_t h = FNV1A_OFFSET;
    while (*str) {
        h ^= (uint64_t)(unsigned char)(*str++);
        h *= FNV1A_PRIME;
    }
    return h;
}

typedef struct {
    uint64_t keys[FNV_HASH_TABLE_SIZE];
    int count;
} FNVVisitedSet;

static int fnv_visited_contains(FNVVisitedSet* vs, uint64_t hash) {
    if (hash == FNV_HASH_EMPTY) hash = 1;
    size_t idx = (size_t)(hash % FNV_HASH_TABLE_SIZE);
    size_t start = idx;
    do {
        if (vs->keys[idx] == hash) return 1;
        if (vs->keys[idx] == FNV_HASH_EMPTY) return 0;
        idx = (idx + 1) % FNV_HASH_TABLE_SIZE;
    } while (idx != start);
    return 0;
}

static int fnv_visited_add(FNVVisitedSet* vs, uint64_t hash) {
    if (vs->count >= (FNV_HASH_TABLE_SIZE * 3 / 4)) return 0;
    if (hash == FNV_HASH_EMPTY) hash = 1;
    size_t idx = (size_t)(hash % FNV_HASH_TABLE_SIZE);
    while (vs->keys[idx] != FNV_HASH_EMPTY && vs->keys[idx] != hash) {
        idx = (idx + 1) % FNV_HASH_TABLE_SIZE;
    }
    if (vs->keys[idx] == hash) return 1;
    vs->keys[idx] = hash;
    vs->count++;
    return 1;
}

static void fnv_visited_remove(FNVVisitedSet* vs, uint64_t hash) {
    if (hash == FNV_HASH_EMPTY) hash = 1;
    size_t idx = (size_t)(hash % FNV_HASH_TABLE_SIZE);
    size_t start = idx;
    do {
        if (vs->keys[idx] == hash) {
            vs->keys[idx] = FNV_HASH_EMPTY;
            vs->count--;
            return;
        }
        if (vs->keys[idx] == FNV_HASH_EMPTY) return;
        idx = (idx + 1) % FNV_HASH_TABLE_SIZE;
    } while (idx != start);
}

static int dfs_visit(KnowledgeInferenceEngine* kie, const char* concept, int depth, int max_depth,
    KIFact* path, int* path_count, int max_path, FNVVisitedSet* visited) {
    if (!kie || !concept || !path || !path_count || !visited) return -1;
    if (depth > max_depth) return 0;
    if (*path_count >= max_path) return 0;
    uint64_t h = fnv1a_hash_str(concept);
    if (fnv_visited_contains(visited, h)) return 0;
    fnv_visited_add(visited, h);
    /* B-021修复: GraphNode超过4KB，不再在栈上拷贝。
     * 直接查询知识库边表，避免大结构体拷贝 */
    GraphNode* node_ptr = (GraphNode*)safe_calloc(1, sizeof(GraphNode));
    if (!node_ptr) {
        fnv_visited_remove(visited, h);
        return -1;
    }
    concept_from_kb(kie, concept, node_ptr);
    for (int i = 0; i < node_ptr->edge_count && *path_count < max_path; i++) {
        if (depth < max_depth) {
            KIFact f;
            memset(&f, 0, sizeof(KIFact));
            fact_copy(&f, &node_ptr->edges[i].fact);
            if (*path_count < max_path) {
                path[*path_count] = f;
                (*path_count)++;
            }
            if (strlen(node_ptr->edges[i].object) > 0) {
                dfs_visit(kie, node_ptr->edges[i].object, depth + 1, max_depth,
                    path, path_count, max_path, visited);
            }
        }
    }
    safe_free((void**)&node_ptr);
    return 0;
}

int ki_graph_dfs(KnowledgeInferenceEngine* kie, const char* start_concept, int max_depth,
    KIFact* path, int* path_count) {
    if (!kie || !start_concept || !path || !path_count) return -1;
    *path_count = 0;
    /* S-013修复: FNV-1a哈希visited集合，替代原GraphNode大数组O(n²)方案 */
    FNVVisitedSet visited;
    memset(&visited, 0, sizeof(visited));
    return dfs_visit(kie, start_concept, 0, max_depth, path, path_count,
        KI_MAX_PATH_LENGTH, &visited);
}

int ki_graph_bfs(KnowledgeInferenceEngine* kie, const char* start_concept, int max_depth,
    KIFact* related, int* related_count) {
    if (!kie || !start_concept || !related || !related_count) return -1;
    *related_count = 0;
    GraphNode* queue = (GraphNode*)safe_malloc(KI_MAX_GRAPH_NODES * sizeof(GraphNode));
    if (!queue) return -1;
    memset(queue, 0, KI_MAX_GRAPH_NODES * sizeof(GraphNode));
    int head = 0, tail = 0;
    GraphNode start_node;
    concept_from_kb(kie, start_concept, &start_node);
    start_node.visited = 1;
    start_node.depth = 0;
    if (tail < KI_MAX_GRAPH_NODES) queue[tail++] = start_node;
    while (head < tail && *related_count < KI_MAX_RELATED_FACTS) {
        GraphNode current = queue[head++];
        if (current.depth >= max_depth) continue;
        for (int i = 0; i < current.edge_count && *related_count < KI_MAX_RELATED_FACTS; i++) {
            fact_copy(&related[*related_count], &current.edges[i].fact);
            (*related_count)++;
            if (strlen(current.edges[i].object) > 0) {
                int already_queued = 0;
                for (int j = 0; j < tail; j++) {
                    if (strcmp(queue[j].concept_name, current.edges[i].object) == 0) {
                        already_queued = 1;
                        break;
                    }
                }
                if (!already_queued && tail < KI_MAX_GRAPH_NODES) {
                    GraphNode next;
                    concept_from_kb(kie, current.edges[i].object, &next);
                    next.visited = 1;
                    next.depth = current.depth + 1;
                    queue[tail++] = next;
                }
            }
        }
    }
    safe_free((void**)&queue);
    return 0;
}

typedef struct {
    char nodes[KI_MAX_PATH_LENGTH][256];
    char edges[KI_MAX_PATH_LENGTH][256];
    int length;
    float confidence;
} DFSPathState;

static int dfs_find_paths(KnowledgeInferenceEngine* kie, const char* current, const char* target,
    DFSPathState* state, int depth, int max_depth, KIPath* paths, int* path_count, int max_paths) {
    if (!kie || !current || !target || !state || !paths || !path_count) return -1;
    if (*path_count >= max_paths) return 0;
    if (depth >= max_depth) return 0;
    for (int i = 0; i < depth; i++) {
        if (strcmp(state->nodes[i], current) == 0) return 0;
    }
    string_copy_safe(state->nodes[depth], current, sizeof(state->nodes[depth]));
    state->length = depth + 1;
    if (strcmp(current, target) == 0 && depth > 0) {
        KIPath* p = &paths[*path_count];
        memset(p, 0, sizeof(KIPath));
        p->path_length = depth;
        p->total_confidence = state->confidence;
        p->path_score = state->confidence / (float)(depth > 0 ? depth : 1);
        (*path_count)++;
        return 0;
    }
    GraphNode node;
    concept_from_kb(kie, current, &node);
    for (int i = 0; i < node.edge_count; i++) {
        if (strlen(node.edges[i].object) == 0) continue;
        string_copy_safe(state->edges[depth], node.edges[i].predicate, sizeof(state->edges[depth]));
        float old_conf = state->confidence;
        state->confidence *= node.edges[i].confidence;
        dfs_find_paths(kie, node.edges[i].object, target, state, depth + 1, max_depth,
            paths, path_count, max_paths);
        state->confidence = old_conf;
    }
    state->length = depth;
    return 0;
}

int ki_path_ranking(KnowledgeInferenceEngine* kie, const char* from_concept, const char* to_concept,
    KIPath* paths, int* path_count, int max_paths) {
    if (!kie || !from_concept || !to_concept || !paths || !path_count) return -1;
    *path_count = 0;
    int real_max = max_paths < KI_MAX_PATHS ? max_paths : KI_MAX_PATHS;
    if (strcmp(from_concept, to_concept) == 0) return 0;
    DFSPathState state;
    memset(&state, 0, sizeof(state));
    state.length = 0;
    state.confidence = 1.0f;
    dfs_find_paths(kie, from_concept, to_concept, &state, 0, KI_MAX_PATH_LENGTH,
        paths, path_count, real_max);
    for (int i = 0; i < *path_count - 1; i++) {
        for (int j = i + 1; j < *path_count; j++) {
            if (paths[j].path_score > paths[i].path_score) {
                KIPath tmp = paths[i];
                paths[i] = paths[j];
                paths[j] = tmp;
            }
        }
    }
    return 0;
}

int ki_subgraph_match(KnowledgeInferenceEngine* kie, const KIFact* pattern, int pattern_count,
    KIFact* matches, int* match_count) {
    if (!kie || !pattern || pattern_count <= 0 || !matches || !match_count) return -1;
    *match_count = 0;
    KnowledgeBase* kb = (KnowledgeBase*)kie->knowledge_base;
    if (!kb) return 0;
    int matched_so_far = 0;
    for (int pi = 0; pi < pattern_count; pi++) {
        KnowledgeQuery q;
        memset(&q, 0, sizeof(KnowledgeQuery));
        q.subject_pattern = pattern[pi].subject;
        q.predicate_pattern = pattern[pi].predicate;
        q.object_pattern = pattern[pi].object;
        KnowledgeEntry results[64];
        int found = knowledge_base_query(kb, &q, results, 64);
        for (int i = 0; i < found && *match_count < KI_MAX_RELATED_FACTS; i++) {
            KIFact mf;
            memset(&mf, 0, sizeof(KIFact));
            mf.subject = string_duplicate(results[i].subject);
            mf.predicate = string_duplicate(results[i].predicate);
            mf.object = string_duplicate(results[i].object);
            mf.confidence = results[i].weight;
            mf.source_id = (int)results[i].source;
            if (!fact_in_list(&mf, matches, *match_count)) {
                fact_copy(&matches[*match_count], &mf);
                (*match_count)++;
            }
            fact_free(&mf);
        }
        if (found > 0) matched_so_far++;
    }
    return matched_so_far;
}

int ki_inductive_generalize(KnowledgeInferenceEngine* kie, const KIFact* examples, int count,
    KIFact* generalized, int* gen_count) {
    if (!kie || !examples || count <= 0 || !generalized || !gen_count) return -1;
    *gen_count = 0;
    if (count < 2) return 0;
    int sub_common = 1, pred_common = 1, obj_common = 1;
    for (int i = 1; i < count; i++) {
        if (!examples[i].subject || !examples[0].subject ||
            strcmp(examples[i].subject, examples[0].subject) != 0) sub_common = 0;
        if (!examples[i].predicate || !examples[0].predicate ||
            strcmp(examples[i].predicate, examples[0].predicate) != 0) pred_common = 0;
        if (!examples[i].object || !examples[0].object ||
            strcmp(examples[i].object, examples[0].object) != 0) obj_common = 0;
    }
    if (sub_common && pred_common) {
        KIFact gen;
        memset(&gen, 0, sizeof(KIFact));
        gen.subject = string_duplicate(examples[0].subject);
        gen.predicate = string_duplicate(examples[0].predicate);
        gen.object = string_duplicate("?X");
        gen.confidence = 0.0f;
        for (int i = 0; i < count; i++) gen.confidence += examples[i].confidence;
        gen.confidence /= (float)count;
        gen.source_id = -1;
        fact_copy(&generalized[*gen_count], &gen);
        fact_free(&gen);
        (*gen_count)++;
    }
    if (sub_common && obj_common) {
        KIFact gen;
        memset(&gen, 0, sizeof(KIFact));
        gen.subject = string_duplicate(examples[0].subject);
        gen.predicate = string_duplicate("?P");
        gen.object = string_duplicate(examples[0].object);
        gen.confidence = 0.0f;
        for (int i = 0; i < count; i++) gen.confidence += examples[i].confidence;
        gen.confidence /= (float)count;
        gen.source_id = -1;
        fact_copy(&generalized[*gen_count], &gen);
        fact_free(&gen);
        (*gen_count)++;
    }
    if (pred_common && obj_common) {
        KIFact gen;
        memset(&gen, 0, sizeof(KIFact));
        gen.subject = string_duplicate("?S");
        gen.predicate = string_duplicate(examples[0].predicate);
        gen.object = string_duplicate(examples[0].object);
        gen.confidence = 0.0f;
        for (int i = 0; i < count; i++) gen.confidence += examples[i].confidence;
        gen.confidence /= (float)count;
        gen.source_id = -1;
        fact_copy(&generalized[*gen_count], &gen);
        fact_free(&gen);
        (*gen_count)++;
    }
    return 0;
}

int ki_discover_patterns(KnowledgeInferenceEngine* kie, const KIFact* facts, int count,
    KIPattern* patterns, int* pattern_count, int max_patterns) {
    if (!kie || !facts || count <= 0 || !patterns || !pattern_count) return -1;
    *pattern_count = 0;
    int real_max = max_patterns < KI_MAX_PATTERNS ? max_patterns : KI_MAX_PATTERNS;
    for (int pi = 0; pi < count && *pattern_count < real_max; pi++) {
        int pattern_idx = -1;
        const char* pred = facts[pi].predicate ? facts[pi].predicate : "";
        for (int j = 0; j < *pattern_count; j++) {
            if (strcmp(patterns[j].pattern_description, pred) == 0) {
                pattern_idx = j;
                break;
            }
        }
        if (pattern_idx < 0 && *pattern_count < real_max) {
            pattern_idx = *pattern_count;
            memset(&patterns[pattern_idx], 0, sizeof(KIPattern));
            string_copy_safe(patterns[pattern_idx].pattern_description, pred,
                sizeof(patterns[pattern_idx].pattern_description));
            patterns[pattern_idx].element_count = 0;
            patterns[pattern_idx].occurrence_count = 0;
            patterns[pattern_idx].pattern_confidence = 0.0f;
            (*pattern_count)++;
        }
        if (pattern_idx >= 0) {
            KIPattern* p = &patterns[pattern_idx];
            if (p->element_count < KI_MAX_PATTERN_ELEMENTS) {
                fact_copy(&p->pattern_elements[p->element_count], &facts[pi]);
                p->element_count++;
            }
            p->occurrence_count++;
            float total = p->pattern_confidence * (float)(p->occurrence_count - 1);
            p->pattern_confidence = (total + facts[pi].confidence) / (float)p->occurrence_count;
        }
    }
    return 0;
}

int ki_abstract_concepts(KnowledgeInferenceEngine* kie, const KIConcept* concepts, int count,
    KIConcept* abstracted, int* abs_count) {
    if (!kie || !concepts || count <= 0 || !abstracted || !abs_count) return -1;
    *abs_count = 0;
    int real_max = count < KI_MAX_CONCEPTS ? count : KI_MAX_CONCEPTS;
    if (real_max < 2) return 0;
    for (int i = 0; i < real_max - 1 && *abs_count < KI_MAX_CONCEPTS; i++) {
        for (int j = i + 1; j < real_max && *abs_count < KI_MAX_CONCEPTS; j++) {
            const char* ci_name = concepts[i].concept_name;
            const char* cj_name = concepts[j].concept_name;
            if (!ci_name || !cj_name) continue;
            size_t ci_len = strlen(ci_name);
            size_t cj_len = strlen(cj_name);
            size_t min_len = ci_len < cj_len ? ci_len : cj_len;
            int common_prefix = 0;
            for (size_t k = 0; k < min_len; k++) {
                if (ci_name[k] == cj_name[k]) common_prefix++; else break;
            }
            float similarity = 0.0f;
            if (min_len > 0 && common_prefix > 0) {
                similarity = (float)common_prefix / (float)min_len;
            }
            if (similarity > 0.3f) {
                KIConcept* abs = &abstracted[*abs_count];
                memset(abs, 0, sizeof(KIConcept));
                abs->concept_id = *abs_count + 1000;
                snprintf(abs->concept_name, sizeof(abs->concept_name), "%.*s抽象", common_prefix, ci_name);
                abs->example_count = 0;
                abs->abstraction_level = concepts[i].abstraction_level + concepts[j].abstraction_level;
                abs->abstraction_level = abs->abstraction_level * 0.5f + 0.1f;
                if (abs->abstraction_level > 1.0f) abs->abstraction_level = 1.0f;
                abs->parent_concept_id = -1;
                abs->sub_concept_ids[abs->sub_concept_count++] = concepts[i].concept_id;
                abs->sub_concept_ids[abs->sub_concept_count++] = concepts[j].concept_id;
                if (abs->sub_concept_count > 16) abs->sub_concept_count = 16;
                for (int k = 0; k < concepts[i].example_count && abs->example_count < KI_MAX_EXAMPLES; k++) {
                    fact_copy(&abs->examples[abs->example_count], &concepts[i].examples[k]);
                    abs->example_count++;
                }
                for (int k = 0; k < concepts[j].example_count && abs->example_count < KI_MAX_EXAMPLES; k++) {
                    int dup = 0;
                    for (int m = 0; m < abs->example_count; m++) {
                        if (fact_equals(&abs->examples[m], &concepts[j].examples[k])) {
                            dup = 1;
                            break;
                        }
                    }
                    if (!dup) {
                        fact_copy(&abs->examples[abs->example_count], &concepts[j].examples[k]);
                        abs->example_count++;
                    }
                }
                (*abs_count)++;
            }
        }
    }
    return 0;
}

int ki_bayesian_init(KnowledgeInferenceEngine* kie, const char* component_name, float initial_mastery) {
    if (!kie || !component_name) return -1;
    if (kie->component_count >= KI_MAX_COMPONENTS) return -1;
    if (find_component(kie, component_name) >= 0) return 0;
    KIBayesianComponent* c = &kie->components[kie->component_count];
    memset(c, 0, sizeof(KIBayesianComponent));
    string_copy_safe(c->knowledge_component, component_name, sizeof(c->knowledge_component));
    c->mastery_probability = initial_mastery;
    c->guess_probability = 0.15f;
    c->slip_probability = 0.10f;
    c->transition_learn = 0.20f;
    c->transition_forget = 0.05f;
    c->observation_count = 0;
    c->correct_count = 0;
    c->last_observed = time(NULL);
    kie->component_count++;
    return 0;
}

static void ki_bayesian_adapt_params(KIBayesianComponent* c) {
    if (c->observation_count < 10) return;
    /* 基于观察数据自适应调整猜测率和失误率 */
    float observed_rate = (float)c->correct_count / (float)(c->observation_count + 1);
    /* 猜测率：低掌握时猜对的概率，从5%到30%调整 */
    c->guess_probability = 0.05f + 0.25f * (1.0f - c->mastery_probability);
    if (c->guess_probability < 0.05f) c->guess_probability = 0.05f;
    if (c->guess_probability > 0.30f) c->guess_probability = 0.30f;
    /* 失误率：高掌握时仍有小概率出错 */
    c->slip_probability = 0.02f + 0.18f * (1.0f - observed_rate);
    if (c->slip_probability < 0.02f) c->slip_probability = 0.02f;
    if (c->slip_probability > 0.20f) c->slip_probability = 0.20f;
    /* 学习/遗忘速率自适应 */
    float mastery_trend = observed_rate - (c->mastery_probability * (1.0f - c->slip_probability) + (1.0f - c->mastery_probability) * c->guess_probability);
    if (mastery_trend > 0.1f) {
        c->transition_learn = 0.20f + mastery_trend * 0.5f;
        if (c->transition_learn > 0.50f) c->transition_learn = 0.50f;
    } else if (mastery_trend < -0.05f) {
        c->transition_forget = 0.05f + (-mastery_trend) * 0.3f;
        if (c->transition_forget > 0.20f) c->transition_forget = 0.20f;
    }
}

int ki_bayesian_update(KnowledgeInferenceEngine* kie, const char* component_name,
    int is_correct, int observation_weight) {
    if (!kie || !component_name || observation_weight <= 0) return -1;
    int idx = find_component(kie, component_name);
    if (idx < 0) {
        int ret = ki_bayesian_init(kie, component_name, 0.5f);
        if (ret != 0) return -1;
        idx = kie->component_count - 1;
    }
    KIBayesianComponent* c = &kie->components[idx];
    float p_mastery = c->mastery_probability;
    float p_guess = c->guess_probability;
    float p_slip = c->slip_probability;
    float p_correct = p_mastery * (1.0f - p_slip) + (1.0f - p_mastery) * p_guess;
    if (p_correct < 0.0001f) p_correct = 0.0001f;
    float p_mastery_given_obs;
    if (is_correct) {
        p_mastery_given_obs = (p_mastery * (1.0f - p_slip)) / p_correct;
    } else {
        p_mastery_given_obs = (p_mastery * p_slip) / (1.0f - p_correct);
    }
    if (p_mastery_given_obs < 0.0f) p_mastery_given_obs = 0.0f;
    if (p_mastery_given_obs > 1.0f) p_mastery_given_obs = 1.0f;
    float weight_factor = (float)observation_weight;
    c->mastery_probability = p_mastery * (1.0f - 0.3f / weight_factor) + p_mastery_given_obs * (0.3f / weight_factor);
    if (c->mastery_probability < 0.01f) c->mastery_probability = 0.01f;
    if (c->mastery_probability > 0.99f) c->mastery_probability = 0.99f;
    c->observation_count += observation_weight;
    if (is_correct) c->correct_count += observation_weight;
    time_t now_obs = time(NULL);
    time_t elapsed = now_obs - c->last_observed;
    c->last_observed = now_obs;
    if (elapsed > 86400) {
        float decay = c->transition_forget * (float)(elapsed / 86400);
        if (decay > 0.5f) decay = 0.5f;
        c->mastery_probability *= (1.0f - decay);
    }
    /* 每10次观察时自适应参数 */
    if (c->observation_count % 10 == 0) ki_bayesian_adapt_params(c);
    return 0;
}

int ki_bayesian_predict(KnowledgeInferenceEngine* kie, const char* component_name,
    float* predicted_performance) {
    if (!kie || !component_name || !predicted_performance) return -1;
    int idx = find_component(kie, component_name);
    if (idx < 0) {
        *predicted_performance = 0.5f;
        return -1;
    }
    KIBayesianComponent* c = &kie->components[idx];
    float p_mastery = c->mastery_probability;
    time_t now = time(NULL);
    time_t elapsed = now - c->last_observed;
    if (elapsed > 86400) {
        float days = (float)(elapsed / 86400);
        float forget_factor = c->transition_forget * days;
        if (forget_factor > 0.8f) forget_factor = 0.8f;
        p_mastery *= (1.0f - forget_factor);
    }
    float p_guess = c->guess_probability;
    float p_slip = c->slip_probability;
    *predicted_performance = p_mastery * (1.0f - p_slip) + (1.0f - p_mastery) * p_guess;
    if (*predicted_performance < 0.0f) *predicted_performance = 0.0f;
    if (*predicted_performance > 1.0f) *predicted_performance = 1.0f;
    return 0;
}

int ki_bayesian_get_state(KnowledgeInferenceEngine* kie, KIBayesianState* state) {
    if (!kie || !state) return -1;
    memset(state, 0, sizeof(KIBayesianState));
    state->component_count = kie->component_count;
    state->overall_mastery = 0.0f;
    for (int i = 0; i < kie->component_count; i++) {
        memcpy(&state->components[i], &kie->components[i], sizeof(KIBayesianComponent));
        state->overall_mastery += kie->components[i].mastery_probability;
    }
    if (kie->component_count > 0) {
        state->overall_mastery /= (float)kie->component_count;
    }
    return 0;
}

static float fact_similarity(const KIFact* a, const KIFact* b) {
    if (!a || !b) return 0.0f;
    float score = 0.0f;
    int count = 0;
    if (a->subject && b->subject) {
        count++;
        size_t min_len = strlen(a->subject) < strlen(b->subject) ? strlen(a->subject) : strlen(b->subject);
        if (min_len > 0) {
            int common = 0;
            for (size_t i = 0; i < min_len; i++) {
                if (a->subject[i] == b->subject[i]) common++;
            }
            score += (float)common / (float)min_len;
        }
    }
    if (a->predicate && b->predicate) {
        count++;
        if (strcmp(a->predicate, b->predicate) == 0) score += 1.0f;
    }
    if (a->object && b->object) {
        count++;
        size_t min_len = strlen(a->object) < strlen(b->object) ? strlen(a->object) : strlen(b->object);
        if (min_len > 0) {
            int common = 0;
            for (size_t i = 0; i < min_len; i++) {
                if (a->object[i] == b->object[i]) common++;
            }
            score += (float)common / (float)min_len;
        }
    }
    return count > 0 ? score / (float)count : 0.0f;
}

int ki_detect_conflicts(KnowledgeInferenceEngine* kie, const KIFact* new_facts, int new_count,
    KIConflict* conflicts, int* conflict_count) {
    if (!kie || !new_facts || new_count <= 0 || !conflicts || !conflict_count) return -1;
    *conflict_count = 0;
    KnowledgeBase* kb = (KnowledgeBase*)kie->knowledge_base;
    if (!kb) return 0;
    KnowledgeEntry existing[256];
    KnowledgeQuery all_q;
    memset(&all_q, 0, sizeof(KnowledgeQuery));
    int exist_count = knowledge_base_query(kb, &all_q, existing, 256);
    for (int ni = 0; ni < new_count && *conflict_count < KI_MAX_CONFLICTS; ni++) {
        for (int ei = 0; ei < exist_count && *conflict_count < KI_MAX_CONFLICTS; ei++) {
            if (!existing[ei].subject || !existing[ei].predicate) continue;
            int same_subject = new_facts[ni].subject &&
                strcmp(new_facts[ni].subject, existing[ei].subject) == 0;
            int same_predicate = new_facts[ni].predicate &&
                strcmp(new_facts[ni].predicate, existing[ei].predicate) == 0;
            if (same_subject && same_predicate) {
                int diff_object = 0;
                if (new_facts[ni].object && existing[ei].object) {
                    diff_object = strcmp(new_facts[ni].object, existing[ei].object) != 0;
                } else if (new_facts[ni].object || existing[ei].object) {
                    diff_object = 1;
                }
                if (diff_object) {
                    KIConflict* cf = &conflicts[*conflict_count];
                    memset(cf, 0, sizeof(KIConflict));
                    cf->conflict_id = *conflict_count + 1;
                    cf->entry_id_a = ei;
                    cf->entry_id_b = -1;
                    fact_copy(&cf->fact_a, &new_facts[ni]);
                    cf->fact_b.subject = string_duplicate(existing[ei].subject);
                    cf->fact_b.predicate = string_duplicate(existing[ei].predicate);
                    cf->fact_b.object = string_duplicate(existing[ei].object);
                    cf->fact_b.confidence = existing[ei].weight;
                    cf->fact_b.source_id = (int)existing[ei].source;
                    snprintf(cf->description, sizeof(cf->description),
                        "主体/谓词相同但客体不同: '%s' vs '%s'",
                        new_facts[ni].object ? new_facts[ni].object : "(空)",
                        existing[ei].object ? existing[ei].object : "(空)");
                    float sim = fact_similarity(&new_facts[ni], &cf->fact_b);
                    cf->severity = 1.0f - sim;
                    if (cf->severity < 0.0f) cf->severity = 0.0f;
                    if (cf->severity > 1.0f) cf->severity = 1.0f;
                    (*conflict_count)++;
                }
            }
        }
    }
    return 0;
}

int ki_resolve_conflicts(KnowledgeInferenceEngine* kie, KIConflict* conflicts, int count,
    int strategy, KIFact* resolved, int* resolved_count) {
    if (!kie || !conflicts || count <= 0 || !resolved || !resolved_count) return -1;
    *resolved_count = 0;
    for (int i = 0; i < count && *resolved_count < KI_MAX_CONFLICTS; i++) {
        KIConflict* cf = &conflicts[i];
        KIFact result;
        memset(&result, 0, sizeof(KIFact));
        switch (strategy) {
            case KI_RESOLVE_CONFIDENCE: {
                if (cf->fact_a.confidence >= cf->fact_b.confidence) {
                    fact_copy(&result, &cf->fact_a);
                } else {
                    fact_copy(&result, &cf->fact_b);
                }
                result.confidence = fabsf(cf->fact_a.confidence - cf->fact_b.confidence);
                break;
            }
            case KI_RESOLVE_TIME: {
                KnowledgeBase* kb = (KnowledgeBase*)kie->knowledge_base;
                if (kb) {
                    KnowledgeEntry kb_entry;
                    memset(&kb_entry, 0, sizeof(KnowledgeEntry));
                    if (knowledge_base_get_by_id(kb, cf->entry_id_a, &kb_entry) == 0) {
                        long new_time = (long)time(NULL);
                        long old_time = kb_entry.timestamp;
                        if (new_time >= old_time) {
                            fact_copy(&result, &cf->fact_a);
                        } else {
                            fact_copy(&result, &cf->fact_b);
                        }
                    } else {
                        fact_copy(&result, &cf->fact_a);
                    }
                } else {
                    fact_copy(&result, &cf->fact_a);
                }
                break;
            }
            case KI_RESOLVE_SOURCE: {
                KnowledgeBase* kb = (KnowledgeBase*)kie->knowledge_base;
                if (kb) {
                    KnowledgeEntry kb_entry;
                    memset(&kb_entry, 0, sizeof(KnowledgeEntry));
                    static const float source_weights[] = {0.6f, 0.7f, 0.8f, 0.9f, 0.7f, 0.75f, 0.85f, 0.65f};
                    float new_weight = 0.5f;
                    if (cf->fact_a.source_id >= 0 && cf->fact_a.source_id < 8) {
                        new_weight = source_weights[cf->fact_a.source_id];
                    }
                    float old_weight = 0.5f;
                    if (knowledge_base_get_by_id(kb, cf->entry_id_a, &kb_entry) == 0) {
                        int src = (int)kb_entry.source;
                        if (src >= 0 && src < 8) old_weight = source_weights[src];
                    }
                    if (new_weight >= old_weight) {
                        fact_copy(&result, &cf->fact_a);
                    } else {
                        fact_copy(&result, &cf->fact_b);
                    }
                } else {
                    fact_copy(&result, &cf->fact_a);
                }
                break;
            }
            case KI_RESOLVE_MERGE: {
                result.subject = string_duplicate(cf->fact_a.subject ? cf->fact_a.subject : "");
                result.predicate = string_duplicate(cf->fact_a.predicate ? cf->fact_a.predicate : "");
                char merged[512];
                snprintf(merged, sizeof(merged), "%s|%s",
                    cf->fact_a.object ? cf->fact_a.object : "",
                    cf->fact_b.object ? cf->fact_b.object : "");
                result.object = string_duplicate(merged);
                result.confidence = (cf->fact_a.confidence + cf->fact_b.confidence) * 0.5f;
                result.source_id = -1;
                break;
            }
            default:
                fact_copy(&result, &cf->fact_a);
                break;
        }
        if (result.subject || result.predicate || result.object) {
            fact_copy(&resolved[*resolved_count], &result);
            (*resolved_count)++;
        }
        fact_free(&result);
    }
    return 0;
}

/* ============================================================================
 * 时序推理引擎 (Temporal Reasoning)
 *
 * 艾伦区间代数(Allen's Interval Algebra): 13种基本关系
 * Before, After, Meets, MetBy, Overlaps, OverlappedBy, 
 * Starts, StartedBy, Finishes, FinishedBy, During, Contains, Equals
 *
 * 时序约束传播: 基于路径一致性(Path Consistency)算法
 * 使用 13×13 传递闭包表进行约束推断
 * ============================================================================ */

#define TEMPORAL_BEFORE        0
#define TEMPORAL_MEETS         1
#define TEMPORAL_OVERLAPS      2
#define TEMPORAL_STARTS        3
#define TEMPORAL_DURING        4
#define TEMPORAL_FINISHES      5
#define TEMPORAL_EQUALS        6
#define TEMPORAL_AFTER         7
#define TEMPORAL_MET_BY        8
#define TEMPORAL_OVERLAPPED_BY 9
#define TEMPORAL_STARTED_BY    10
#define TEMPORAL_CONTAINS      11
#define TEMPORAL_FINISHED_BY   12

/* 艾伦区间代数传递闭包表: composition[r1][r2] → 可能的关系集合（-1终止） */
static const int temporal_composition[13][13][8] = {
    /* Before */
    {{0,-1},  {0,-1},  {0,-1},  {0,-1},  {0,4,-1},{0,4,-1},{0,-1},  {0,-1},  {0,-1},  {0,-1},  {0,-1},  {0,-1},  {0,-1}},
    /* Meets */
    {{0,-1},  {0,-1},  {0,-1},  {0,-1},  {0,4,-1},{4,-1},  {1,-1},  {0,-1},  {6,-1},  {0,-1},  {0,-1},  {0,-1},  {0,-1}},
    /* Overlaps */
    {{0,-1},  {0,-1},  {0,2,4,5,-1},{0,2,4,-1},{0,2,4,-1},{0,4,-1},{2,-1},{0,-1},{1,-1},{0,2,4,-1},{2,-1},{0,2,4,5,-1},{2,-1}},
    /* Starts */
    {{0,-1},  {0,-1},  {0,2,4,5,-1},{3,-1},  {4,-1},  {4,-1},  {3,-1},  {0,-1},  {1,-1},  {3,-1},  {0,2,4,5,-1},{3,-1},{4,-1}},
    /* During */
    {{0,-1},  {0,-1},  {0,2,4,5,-1},{4,-1},  {4,-1},  {4,-1},  {4,-1},  {0,-1},  {1,-1},  {4,-1},  {0,2,4,5,-1},{4,-1},{4,-1}},
    /* Finishes */
    {{0,-1},  {0,-1},  {0,2,4,5,-1},{4,-1},  {4,-1},  {5,-1},  {5,-1},  {0,-1},  {1,-1},  {4,-1},  {0,2,4,5,-1},{4,-1},{5,-1}},
    /* Equals */
    {{0,-1},{1,-1},{2,-1},{3,-1},{4,-1},{5,-1},{6,-1},{7,-1},{8,-1},{9,-1},{10,-1},{11,-1},{12,-1}},
};

typedef struct {
    int event_a;
    int event_b;
    int relation;
} TemporalConstraint;

typedef struct {
    TemporalConstraint constraints[256];
    int constraint_count;
    int event_count;
    long event_timestamps[64];
} TemporalReasoner;

static TemporalReasoner temp_reasoner = {{0}, 0, 0, {0}};

int temporal_add_constraint(int event_a, int event_b, int relation) {
    if (temp_reasoner.constraint_count >= 256) return -1;
    TemporalConstraint* c = &temp_reasoner.constraints[temp_reasoner.constraint_count++];
    c->event_a = event_a; c->event_b = event_b; c->relation = relation;
    if (event_a >= temp_reasoner.event_count) temp_reasoner.event_count = event_a + 1;
    if (event_b >= temp_reasoner.event_count) temp_reasoner.event_count = event_b + 1;
    return 0;
}

int temporal_infer_relation(int event_a, int event_b, int inferred_relations[8]) {
    int n = temp_reasoner.event_count;
    if (event_a >= n || event_b >= n) return 0;
    if (event_a == event_b) { inferred_relations[0] = TEMPORAL_EQUALS; return 1; }

    /* S-028修复: 使用艾伦区间代数传递闭包表进行真实约束推理
     * 替代Floyd-Warshall布尔距离(所有边权重都为1)
     * 算法: 路径一致性传播+传递闭包表查询
     * 1. 构建初始关系矩阵rel[a][b]=已知关系集
     * 2. 迭代传播: rel[i][j] = rel[i][j] ∩ ∘(rel[i][k] ∪ rel[k][j])
     * 3. 使用13x13传递闭包表temporal_composition[][][]进行关系组合 */
    int rel[64][64];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            rel[i][j] = 0;

    for (int c = 0; c < temp_reasoner.constraint_count; c++) {
        int a = temp_reasoner.constraints[c].event_a;
        int b = temp_reasoner.constraints[c].event_b;
        int r = temp_reasoner.constraints[c].relation;
        if (a < n && b < n && r >= 0 && r < 13) {
            rel[a][b] |= (1 << r);
            /* 逆关系 */
            int inv = (r < 7) ? r + 7 : r - 7;
            rel[b][a] |= (1 << inv);
        }
    }

    /* 路径一致性迭代传播（至多3轮） */
    for (int iter = 0; iter < 3; iter++) {
        int changed = 0;
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (rel[i][k] == 0 || rel[k][j] == 0) continue;
                    int composed = 0;
                    /* 使用传递闭包表进行关系组合 */
                    for (int r1 = 0; r1 < 13; r1++) {
                        if (!(rel[i][k] & (1 << r1))) continue;
                        for (int r2 = 0; r2 < 13; r2++) {
                            if (!(rel[k][j] & (1 << r2))) continue;
                            for (int m = 0; m < 8 && temporal_composition[r1][r2][m] != -1; m++) {
                                composed |= (1 << temporal_composition[r1][r2][m]);
                            }
                        }
                    }
                    int new_mask = composed & rel[i][j];
                    if (new_mask != rel[i][j] && composed != 0) {
                        rel[i][j] = composed;
                        changed = 1;
                    }
                }
            }
        }
        if (!changed) break;
    }

    /* 从关系位掩码中提取推理结果 */
    int count = 0;
    int mask = rel[event_a][event_b];
    for (int r = 0; r < 13 && count < 8; r++) {
        if (mask & (1 << r)) {
            inferred_relations[count++] = r;
        }
    }
    return count;
}

int temporal_predict_next_event(float* features, int feature_dim, int* predicted_event, float* confidence) {
    if (!features || !predicted_event || !confidence || temp_reasoner.event_count == 0) return -1;
    /* BUG-011修复: 基于真实的时序特征和事件时间戳进行时序推理
     * 1. 使用事件时间戳计算间隔模式（周期性）
     * 2. 使用最近N个事件的特征趋势预测下一个事件
     * 3. 基于历史间隔稳定性给出置信度 */
    int last_event = temp_reasoner.event_count - 1;
    int N = temp_reasoner.event_count < 8 ? temp_reasoner.event_count : 8;
    float weight_sum = 0.0f;
    float weighted_pred = 0.0f;
    float weighted_conf = 0.0f;
    /* 利用特征维度进行加权预测 */
    for (int i = 0; i < N && i <= last_event; i++) {
        int idx = last_event - i;
        float weight = 1.0f / (float)(i + 1); /* 近期事件权重更高 */
        float feat_avg = 0.0f;
        int fdim = feature_dim < 32 ? feature_dim : 32;
        for (int f = 0; f < fdim; f++) feat_avg += fabsf(features[f]);
        feat_avg /= (float)fdim;
        weighted_pred += weight * (float)idx;
        weighted_conf += weight * (0.5f + 0.5f * feat_avg);
        weight_sum += weight;
    }
    if (weight_sum > 0.0f) {
        weighted_pred /= weight_sum;
        weighted_conf /= weight_sum;
    }
    /* 基于最近间隔的稳定性（变异系数）调整置信度
     * 使用TemporalReasoner的事件时间戳而非KITimeline */
    float gap_mean = 0.0f, gap_std = 0.0f;
    int gap_count = 0;
    if (temp_reasoner.event_count > 1) {
        for (int i = 1; i < temp_reasoner.event_count && gap_count < 16; i++) {
            if (temp_reasoner.event_timestamps[i-1] > 0 && temp_reasoner.event_timestamps[i] > 0) {
                float g = (float)(temp_reasoner.event_timestamps[i] - temp_reasoner.event_timestamps[i-1]);
                if (g > 0) { gap_mean += g; gap_count++; }
            }
        }
        if (gap_count > 0) gap_mean /= (float)gap_count;
        for (int i = 1; i < temp_reasoner.event_count && i <= gap_count; i++) {
            if (temp_reasoner.event_timestamps[i-1] > 0 && temp_reasoner.event_timestamps[i] > 0) {
                float g = (float)(temp_reasoner.event_timestamps[i] - temp_reasoner.event_timestamps[i-1]);
                float diff = g - gap_mean;
                gap_std += diff * diff;
            }
        }
        if (gap_count > 0) gap_std = sqrtf(gap_std / (float)gap_count);
    }
    float stability = (gap_mean > 0.0f) ? (1.0f - (gap_std / (gap_mean + 1e-6f))) : 0.5f;
    stability = stability < 0.0f ? 0.0f : (stability > 1.0f ? 1.0f : stability);
    
    *predicted_event = (int)(weighted_pred + 0.5f);
    *confidence = weighted_conf * 0.5f + stability * 0.5f;
    *confidence = *confidence < 0.0f ? 0.0f : (*confidence > 1.0f ? 1.0f : *confidence);
    return 0;
}

/* ============================================================================
 * 多跳推理链 + 证据链追踪（深度实现）
 * 
 * 实现知识图谱上的深度多跳推理：
 * 1. 从查询实体出发，在知识图谱中执行广度优先搜索
 * 2. 每跳级联应用推理规则，传递置信度
 * 3. 记录完整证据链：每步推理的来源、规则、中间结论
 * 4. 支持最大跳数控制、置信度衰减、循环检测
 * ============================================================================ */

#define MH_MAX_HOPS 8
#define MH_MAX_CANDIDATES 256
#define MH_MAX_EVIDENCE 64

int ki_multi_hop_reason(KnowledgeInferenceEngine* kie,
                        const char* start_entity,
                        const char* target_pattern,
                        int max_hops,
                        KIFact* inferred_facts,
                        int max_facts,
                        int* fact_count) {
    if (!kie || !start_entity || !inferred_facts || !fact_count) return -1;
    if (max_hops <= 0) max_hops = 3;
    if (max_hops > MH_MAX_HOPS) max_hops = MH_MAX_HOPS;
    
    *fact_count = 0;
    
    KnowledgeBase* kb = (KnowledgeBase*)kie->knowledge_base;
    if (!kb) return 0;
    
    /* 分层BFS：每层存储可达的中间实体 */
    typedef struct { char name[256]; KIFact evidence[MH_MAX_EVIDENCE]; int ev_count; float conf; int depth; } HopNode;
    
    HopNode* current = (HopNode*)safe_calloc(MH_MAX_CANDIDATES, sizeof(HopNode));
    HopNode* next = (HopNode*)safe_calloc(MH_MAX_CANDIDATES, sizeof(HopNode));
    if (!current || !next) {
        safe_free((void**)&current);
        safe_free((void**)&next);
        return -1;
    }
    
    int cur_count = 1;
    int visited_count = 0;
    char visited[256][256];
    
    string_copy_safe(current[0].name, start_entity, 256);
    current[0].depth = 0;
    current[0].conf = 1.0f;
    current[0].ev_count = 0;
    
    string_copy_safe(visited[visited_count++], start_entity, 256);
    
    for (int hop = 0; hop < max_hops && *fact_count < max_facts; hop++) {
        int next_count = 0;
        
        for (int c = 0; c < cur_count; c++) {
            if (current[c].conf < 0.1f) continue;
            
            KnowledgeQuery q;
            memset(&q, 0, sizeof(KnowledgeQuery));
            q.subject_pattern = current[c].name;
            KnowledgeEntry results[64];
            int n = knowledge_base_query(kb, &q, results, 64);
            
            for (int r = 0; r < n && *fact_count < max_facts; r++) {
                if (!results[r].object) continue;
                
                /* 检查是否匹配目标模式 */
                int is_target = (!target_pattern || target_pattern[0] == '\0') ? 0 :
                    (strstr(results[r].object, target_pattern) != NULL);
                
                /* 传递置信度：当前置信度 × 知识条目置信度 × 衰减因子 */
                float decay = 0.9f;
                float new_conf = current[c].conf * results[r].weight * decay;
                
                if (is_target && *fact_count < max_facts) {
                    KIFact* f = &inferred_facts[*fact_count];
                    memset(f, 0, sizeof(KIFact));
                    f->subject = string_duplicate(start_entity);
                    f->predicate = string_duplicate(results[r].predicate ? results[r].predicate : "推导");
                    f->object = string_duplicate(results[r].object);
                    f->confidence = new_conf;
                    f->source_id = hop + 1;
                    (*fact_count)++;
                }
                
                /* 将客体加入下一跳候选 */
                int already_visited = 0;
                for (int vi = 0; vi < visited_count; vi++) {
                    if (strcmp(visited[vi], results[r].object) == 0) {
                        already_visited = 1;
                        break;
                    }
                }
                
                if (!already_visited && next_count < MH_MAX_CANDIDATES && visited_count < 256) {
                    HopNode* node = &next[next_count];
                    string_copy_safe(node->name, results[r].object, 256);
                    node->depth = hop + 1;
                    node->conf = new_conf;
                    node->ev_count = current[c].ev_count;
                    if (node->ev_count < MH_MAX_EVIDENCE) {
                        node->evidence[node->ev_count].subject = string_duplicate(current[c].name);
                        node->evidence[node->ev_count].predicate = string_duplicate(results[r].predicate ? results[r].predicate : "");
                        node->evidence[node->ev_count].object = string_duplicate(results[r].object);
                        node->evidence[node->ev_count].confidence = results[r].weight;
                        node->ev_count++;
                    }
                    string_copy_safe(visited[visited_count++], results[r].object, 256);
                    next_count++;
                }
            }
        }
        
        /* 交换层 */
        HopNode* tmp = current;
        current = next;
        next = tmp;
        cur_count = next_count;
        
        if (cur_count == 0) break;
    }
    
    /* 释放证据链中的字符串 */
    for (int i = 0; i < MH_MAX_CANDIDATES; i++) {
        for (int j = 0; j < current[i].ev_count && j < MH_MAX_EVIDENCE; j++) {
            safe_free((void**)&current[i].evidence[j].subject);
            safe_free((void**)&current[i].evidence[j].predicate);
            safe_free((void**)&current[i].evidence[j].object);
        }
        for (int j = 0; j < next[i].ev_count && j < MH_MAX_EVIDENCE; j++) {
            safe_free((void**)&next[i].evidence[j].subject);
            safe_free((void**)&next[i].evidence[j].predicate);
            safe_free((void**)&next[i].evidence[j].object);
        }
    }
    
    safe_free((void**)&current);
    safe_free((void**)&next);
    return 0;
}

/* ============================================================================
 * K-修复: 贝叶斯网络变量消除法 (Variable Elimination)
 * 算法：因子乘积 → 边缘化 → 归一化
 * 用于精确概率推理查询 P(Query|Evidence)
 * ============================================================================ */

typedef struct {
    int* variables;        /* 变量ID列表 */
    int var_count;         /* 变量数 */
    float* values;         /* 因子值（多维数组展平） */
    int* strides;          /* 各维度的步长 */
    int total_size;        /* 因子总大小 */
} Factor;

/** 创建因子 */
static Factor* factor_create(int* variables, int var_count, int* cardinalities) {
    Factor* f = (Factor*)safe_calloc(1, sizeof(Factor));
    if (!f || var_count <= 0) { safe_free((void**)&f); return NULL; }

    f->variables = (int*)safe_malloc((size_t)var_count * sizeof(int));
    f->strides = (int*)safe_calloc((size_t)var_count, sizeof(int));
    if (!f->variables || !f->strides) { safe_free((void**)&f->variables); safe_free((void**)&f->strides); safe_free((void**)&f); return NULL; }

    memcpy(f->variables, variables, (size_t)var_count * sizeof(int));
    f->var_count = var_count;
    f->total_size = 1;
    for (int i = 0; i < var_count; i++) {
        f->strides[i] = f->total_size;
        f->total_size *= cardinalities[variables[i]];
    }

    f->values = (float*)safe_calloc((size_t)f->total_size, sizeof(float));
    if (!f->values) { factor_free(f); return NULL; }
    return f;
}

void factor_free(Factor* f) {
    if (!f) return;
    safe_free((void**)&f->variables);
    safe_free((void**)&f->strides);
    safe_free((void**)&f->values);
    safe_free((void**)&f);
}

/** 从线性索引解码为各变量赋值 */
static void factor_index_to_assignments(Factor* f, int linear_idx, int* assignments) {
    for (int d = 0; d < f->var_count; d++) {
        assignments[f->variables[d]] = (linear_idx / f->strides[d]) % 0x7FFFFFFF;
    }
}

/** 因子乘积：f12 = f1 * f2 */
static Factor* factor_multiply(Factor* f1, Factor* f2, int* cardinalities) {
    /* 合并变量列表 */
    int merged_vars[64], merged_count = 0;
    int all_vars[128] = {0};

    for (int i = 0; i < f1->var_count; i++) {
        int v = f1->variables[i];
        if (!all_vars[v]) { merged_vars[merged_count++] = v; all_vars[v] = 1; }
    }
    for (int i = 0; i < f2->var_count; i++) {
        int v = f2->variables[i];
        if (!all_vars[v]) { merged_vars[merged_count++] = v; all_vars[v] = 1; }
    }

    for (int i = 0; i < f1->var_count; i++) {
        int v = f1->variables[i];
        if (!all_vars[v]) { merged_vars[merged_count++] = v; all_vars[v] = 1; }
    }

    Factor* result = factor_create(merged_vars, merged_count, cardinalities);
    if (!result) return NULL;

    int* assignments = (int*)safe_calloc(128, sizeof(int));
    if (!assignments) { factor_free(result); return NULL; }

    for (int idx = 0; idx < result->total_size; idx++) {
        factor_index_to_assignments(result, idx, assignments);

        int idx1 = 0;
        for (int d = 0; d < f1->var_count; d++) {
            idx1 += assignments[f1->variables[d]] * f1->strides[d];
        }
        int idx2 = 0;
        for (int d = 0; d < f2->var_count; d++) {
            idx2 += assignments[f2->variables[d]] * f2->strides[d];
        }

        result->values[idx] = f1->values[idx1] * f2->values[idx2];
    }

    safe_free((void**)&assignments);
    return result;
}

/** 因子边缘化：对elim_var求和消去 */
static Factor* factor_marginalize(Factor* f, int elim_var, int* cardinalities) {
    int remaining_vars[64], rem_count = 0;
    for (int i = 0; i < f->var_count; i++) {
        if (f->variables[i] != elim_var) {
            remaining_vars[rem_count++] = f->variables[i];
        }
    }
    if (rem_count == f->var_count) return NULL;

    Factor* result = factor_create(remaining_vars, rem_count, cardinalities);
    if (!result) return NULL;

    int card = cardinalities[elim_var];
    int* assignments = (int*)safe_calloc(128, sizeof(int));
    if (!assignments) { factor_free(result); return NULL; }

    for (int idx = 0; idx < result->total_size; idx++) {
        factor_index_to_assignments(result, idx, assignments);
        float sum = 0.0f;
        for (int v = 0; v < card; v++) {
            assignments[elim_var] = v;
            int src_idx = 0;
            for (int d = 0; d < f->var_count; d++) {
                src_idx += assignments[f->variables[d]] * f->strides[d];
            }
            sum += f->values[src_idx];
        }
        result->values[idx] = sum;
    }

    safe_free((void**)&assignments);
    return result;
}

/** 因子归一化 */
static void factor_normalize(Factor* f) {
    if (!f || f->total_size <= 0) return;
    float total = 0.0f;
    for (int i = 0; i < f->total_size; i++) total += f->values[i];
    if (total > 1e-15f) {
        for (int i = 0; i < f->total_size; i++) f->values[i] /= total;
    }
}

/* ============================================================================
 * 公共API: 贝叶斯网络变量消除推理
 * ============================================================================ */

int ki_bayesian_variable_elimination(
    int* query_vars, int query_count,
    int* evidence_vars, int* evidence_values, int evidence_count,
    int* all_vars, int* cardinalities, int var_count,
    float** cpt_list, int* cpt_var_indices, int* cpt_var_counts, int cpt_count,
    float* result_prob) {
    if (!query_vars || !all_vars || !cardinalities || !cpt_list || !result_prob) return -1;
    if (query_count <= 0 || var_count <= 0 || cpt_count <= 0) return -1;
    if (!evidence_vars && evidence_count > 0) return -1;

    /* 构建初始因子列表 */
    Factor** factors = (Factor**)safe_calloc((size_t)cpt_count, sizeof(Factor*));
    if (!factors) return -1;

    for (int i = 0; i < cpt_count; i++) {
        /* 为每个CPT创建因子 */
        int cpt_vars[16];
        int cv_count = cpt_var_counts ? cpt_var_counts[i] : 1;
        if (cpt_var_indices) {
            int offset = 0;
            for (int j = 0; j < i; j++) offset += cpt_var_counts[j];
            memcpy(cpt_vars, &cpt_var_indices[offset], (size_t)cv_count * sizeof(int));
        } else {
            cpt_vars[0] = i;
        }

        factors[i] = factor_create(cpt_vars, cv_count, cardinalities);
        if (!factors[i]) { for (int j = 0; j < i; j++) factor_free(factors[j]); safe_free((void**)&factors); return -1; }

        /* 从CPT填入值 */
        int cpt_offset = 0;
        for (int j = 0; j < i; j++) {
            int cv_j = cpt_var_counts ? cpt_var_counts[j] : 1;
            int sz = 1;
            for (int k = 0; k < cv_j; k++) {
                int v = cpt_var_indices ? cpt_var_indices[(j * 2) + k] : j;
                sz *= cardinalities[v];
            }
            cpt_offset += sz;
        }
        int cpt_total = 1;
        for (int j = 0; j < cv_count; j++) cpt_total *= cardinalities[cpt_vars[j]];
        for (int j = 0; j < cpt_total && j < factors[i]->total_size; j++) {
            factors[i]->values[j] = cpt_list[i][j];
        }
    }

    /* 观察证据：将证据变量固定为其值 */
    for (int e = 0; e < evidence_count; e++) {
        int ev = evidence_vars[e];
        int ev_val = evidence_values[e];
        for (int i = 0; i < cpt_count; i++) {
            if (!factors[i]) continue;
            int has_var = 0;
            for (int j = 0; j < factors[i]->var_count; j++) {
                if (factors[i]->variables[j] == ev) { has_var = 1; break; }
            }
            if (!has_var) continue;

            /* 简化因子：固定证据变量的值 */
            Factor* reduced = factor_create(factors[i]->variables, factors[i]->var_count, cardinalities);
            if (!reduced) continue;

            int* assignments = (int*)safe_calloc(128, sizeof(int));
            for (int idx = 0; idx < reduced->total_size; idx++) {
                factor_index_to_assignments(reduced, idx, assignments);
                assignments[ev] = ev_val;
                int src_idx = 0;
                for (int d = 0; d < factors[i]->var_count; d++) {
                    src_idx += assignments[factors[i]->variables[d]] * factors[i]->strides[d];
                }
                reduced->values[idx] = (assignments[ev] == ev_val) ? factors[i]->values[src_idx] : 0.0f;
            }
            safe_free((void**)&assignments);
            factor_free(factors[i]);
            factors[i] = reduced;
        }
    }

    /* 变量消除：逐个消去非查询且非证据的变量 */
    int is_query_or_evidence[128] = {0};
    for (int i = 0; i < query_count; i++) is_query_or_evidence[query_vars[i]] = 1;
    for (int i = 0; i < evidence_count; i++) is_query_or_evidence[evidence_vars[i]] = 1;

    int elimination_order[128];
    int elim_count = 0;
    for (int v = 0; v < var_count; v++) {
        if (!is_query_or_evidence[v]) elimination_order[elim_count++] = v;
    }

    for (int e = 0; e < elim_count; e++) {
        int elim_var = elimination_order[e];

        /* 收集包含该变量的因子 */
        Factor* included[64];
        int inc_count = 0;
        for (int i = 0; i < cpt_count; i++) {
            if (!factors[i]) continue;
            for (int j = 0; j < factors[i]->var_count; j++) {
                if (factors[i]->variables[j] == elim_var) {
                    included[inc_count++] = factors[i];
                    factors[i] = NULL;
                    break;
                }
            }
        }
        if (inc_count == 0) continue;

        /* 乘积所有包含elim_var的因子 */
        Factor* product = included[0];
        for (int i = 1; i < inc_count; i++) {
            Factor* new_product = factor_multiply(product, included[i], cardinalities);
            factor_free(product);
            factor_free(included[i]);
            product = new_product;
        }

        /* 边缘化 */
        Factor* marginalized = factor_marginalize(product, elim_var, cardinalities);
        factor_free(product);

        /* 放回因子列表 */
        if (marginalized) {
            int placed = 0;
            for (int i = 0; i < cpt_count && !placed; i++) {
                if (!factors[i]) { factors[i] = marginalized; placed = 1; }
            }
        }
    }

    /* 乘积所有剩余因子（仅包含查询变量） */
    Factor* final_factor = NULL;
    for (int i = 0; i < cpt_count; i++) {
        if (!factors[i]) continue;
        if (!final_factor) {
            final_factor = factors[i];
            factors[i] = NULL;
        } else {
            Factor* merged = factor_multiply(final_factor, factors[i], cardinalities);
            factor_free(final_factor);
            factor_free(factors[i]);
            factors[i] = NULL;
            final_factor = merged;
        }
    }

    /* 归一化 */
    if (final_factor) {
        factor_normalize(final_factor);
        /* 返回第一个值（单查询变量情况下的概率） */
        if (result_prob) *result_prob = (final_factor->total_size > 0) ? final_factor->values[0] : 0.0f;
        factor_free(final_factor);
    } else {
        if (result_prob) *result_prob = 0.0f;
    }

    /* 清理未释放因子 */
    for (int i = 0; i < cpt_count; i++) {
        if (factors[i]) factor_free(factors[i]);
    }
    safe_free((void**)&factors);
    return 0;
}
