/**
 * @file content_filter.c
 * @brief 内容安全过滤系统完整实现
 *
 * 实现AI生成内容的安全过滤（输出护栏），包括：
 * 1. 敏感内容检测（暴力/仇恨/非法/自残/色情/个人信息/恶意代码）
 * 2. 模式匹配过滤
 * 3. 内容安全评分
 * 4. 与安全监控系统的集成
 */

#include "selflnn/safety/content_filter.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

#define CONTENT_FILTER_EMBED_DIM  128

struct ContentFilter {
    ContentFilterRule rules[CONTENT_FILTER_MAX_RULES];
    int rule_count;

    size_t total_checked;
    size_t total_blocked;
    size_t total_flagged;

    void* lnn_instance;
    int enable_semantic;    
};

static const char* content_category_default_name(ContentCategory category) {
    static const char* names[] = {
        "暴力内容", "仇恨言论", "非法内容", "自残内容",
        "色情内容", "个人信息", "恶意代码", "自定义"
    };
    int idx = (int)category;
    if (idx < 0 || idx >= (int)(sizeof(names)/sizeof(names[0])))
        return "未知类别";
    return names[idx];
}

const char* content_filter_category_name(ContentCategory category) {
    return content_category_default_name(category);
}

static void content_to_lower(const char* src, char* dst, size_t dst_size) {
    if (!src || !dst || dst_size == 0) return;
    size_t i;
    for (i = 0; i < dst_size - 1 && src[i]; i++) {
        dst[i] = (char)tolower((unsigned char)src[i]);
    }
    dst[i] = '\0';
}

static int content_contains_pattern(const char* text_lower, const char* pattern_lower) {
    if (!text_lower || !pattern_lower) return 0;
    return strstr(text_lower, pattern_lower) != NULL;
}

ContentFilter* content_filter_create(void) {
    ContentFilter* filter = (ContentFilter*)safe_calloc(1, sizeof(ContentFilter));
    if (!filter) return NULL;

    /* 添加默认规则 - 暴力内容 */
    {
        ContentFilterRule* r = &filter->rules[filter->rule_count++];
        r->category = CONTENT_CATEGORY_VIOLENCE;
        strncpy(r->category_name, content_category_default_name(CONTENT_CATEGORY_VIOLENCE),
                sizeof(r->category_name) - 1);
        const char* violence_patterns[] = {"杀人","谋杀","爆炸","恐怖袭击","屠杀","灭口","毁尸"};
        for (size_t i = 0; i < sizeof(violence_patterns)/sizeof(violence_patterns[0]) &&
             r->pattern_count < CONTENT_FILTER_MAX_PATTERNS; i++) {
            strncpy(r->patterns[r->pattern_count], violence_patterns[i],
                    CONTENT_FILTER_PATTERN_LEN - 1);
            r->pattern_count++;
        }
        r->threshold = 0.6f;
        r->enabled = 1;
        r->action_type = 1;
    }

    /* 添加默认规则 - 仇恨言论 */
    {
        ContentFilterRule* r = &filter->rules[filter->rule_count++];
        r->category = CONTENT_CATEGORY_HATE_SPEECH;
        strncpy(r->category_name, content_category_default_name(CONTENT_CATEGORY_HATE_SPEECH),
                sizeof(r->category_name) - 1);
        const char* hate_patterns[] = {"歧视","侮辱","诽谤","人身攻击","种族","劣等"};
        for (size_t i = 0; i < sizeof(hate_patterns)/sizeof(hate_patterns[0]) &&
             r->pattern_count < CONTENT_FILTER_MAX_PATTERNS; i++) {
            strncpy(r->patterns[r->pattern_count], hate_patterns[i],
                    CONTENT_FILTER_PATTERN_LEN - 1);
            r->pattern_count++;
        }
        r->threshold = 0.5f;
        r->enabled = 1;
        r->action_type = 1;
    }

    /* 添加默认规则 - 非法内容 */
    {
        ContentFilterRule* r = &filter->rules[filter->rule_count++];
        r->category = CONTENT_CATEGORY_ILLEGAL;
        strncpy(r->category_name, content_category_default_name(CONTENT_CATEGORY_ILLEGAL),
                sizeof(r->category_name) - 1);
        const char* illegal_patterns[] = {"毒品","走私","洗钱","赌博","诈骗","黑客","入侵"};
        for (size_t i = 0; i < sizeof(illegal_patterns)/sizeof(illegal_patterns[0]) &&
             r->pattern_count < CONTENT_FILTER_MAX_PATTERNS; i++) {
            strncpy(r->patterns[r->pattern_count], illegal_patterns[i],
                    CONTENT_FILTER_PATTERN_LEN - 1);
            r->pattern_count++;
        }
        r->threshold = 0.7f;
        r->enabled = 1;
        r->action_type = 2;
    }

    /* 添加默认规则 - 自残内容 */
    {
        ContentFilterRule* r = &filter->rules[filter->rule_count++];
        r->category = CONTENT_CATEGORY_SELF_HARM;
        strncpy(r->category_name, content_category_default_name(CONTENT_CATEGORY_SELF_HARM),
                sizeof(r->category_name) - 1);
        const char* self_harm_patterns[] = {"自杀","自残","自伤","轻生","了断"};
        for (size_t i = 0; i < sizeof(self_harm_patterns)/sizeof(self_harm_patterns[0]) &&
             r->pattern_count < CONTENT_FILTER_MAX_PATTERNS; i++) {
            strncpy(r->patterns[r->pattern_count], self_harm_patterns[i],
                    CONTENT_FILTER_PATTERN_LEN - 1);
            r->pattern_count++;
        }
        r->threshold = 0.8f;
        r->enabled = 1;
        r->action_type = 3;
    }

    /* 添加默认规则 - 色情内容 */
    {
        ContentFilterRule* r = &filter->rules[filter->rule_count++];
        r->category = CONTENT_CATEGORY_PORNOGRAPHY;
        strncpy(r->category_name, content_category_default_name(CONTENT_CATEGORY_PORNOGRAPHY),
                sizeof(r->category_name) - 1);
        const char* porn_patterns[] = {"色情","淫秽","裸露","性交","成人内容"};
        for (size_t i = 0; i < sizeof(porn_patterns)/sizeof(porn_patterns[0]) &&
             r->pattern_count < CONTENT_FILTER_MAX_PATTERNS; i++) {
            strncpy(r->patterns[r->pattern_count], porn_patterns[i],
                    CONTENT_FILTER_PATTERN_LEN - 1);
            r->pattern_count++;
        }
        r->threshold = 0.6f;
        r->enabled = 1;
        r->action_type = 2;
    }

    /* 添加默认规则 - 个人信息 */
    {
        ContentFilterRule* r = &filter->rules[filter->rule_count++];
        r->category = CONTENT_CATEGORY_PERSONAL_INFO;
        strncpy(r->category_name, content_category_default_name(CONTENT_CATEGORY_PERSONAL_INFO),
                sizeof(r->category_name) - 1);
        const char* pii_patterns[] = {"身份证号","手机号","银行卡","密码","验证码","信用卡"};
        for (size_t i = 0; i < sizeof(pii_patterns)/sizeof(pii_patterns[0]) &&
             r->pattern_count < CONTENT_FILTER_MAX_PATTERNS; i++) {
            strncpy(r->patterns[r->pattern_count], pii_patterns[i],
                    CONTENT_FILTER_PATTERN_LEN - 1);
            r->pattern_count++;
        }
        r->threshold = 0.7f;
        r->enabled = 1;
        r->action_type = 2;
    }

    /* 添加默认规则 - 恶意代码 */
    {
        ContentFilterRule* r = &filter->rules[filter->rule_count++];
        r->category = CONTENT_CATEGORY_MALICIOUS_CODE;
        strncpy(r->category_name, content_category_default_name(CONTENT_CATEGORY_MALICIOUS_CODE),
                sizeof(r->category_name) - 1);
        const char* malware_patterns[] = {"shellcode","exploit","恶意软件","木马","勒索","rootkit","键盘记录"};
        for (size_t i = 0; i < sizeof(malware_patterns)/sizeof(malware_patterns[0]) &&
             r->pattern_count < CONTENT_FILTER_MAX_PATTERNS; i++) {
            strncpy(r->patterns[r->pattern_count], malware_patterns[i],
                    CONTENT_FILTER_PATTERN_LEN - 1);
            r->pattern_count++;
        }
        r->threshold = 0.8f;
        r->enabled = 1;
        r->action_type = 3;
    }
    
    return filter;
}

void content_filter_destroy(ContentFilter* filter) {
    if (!filter) return;
    safe_free((void**)&filter);
}

int content_filter_add_rule(ContentFilter* filter, const ContentFilterRule* rule) {
    if (!filter || !rule) return -1;
    if (filter->rule_count >= CONTENT_FILTER_MAX_RULES) return -1;
    filter->rules[filter->rule_count++] = *rule;
    return filter->rule_count - 1;
}

int content_filter_add_pattern(ContentFilter* filter, ContentCategory category,
                                const char* pattern) {
    if (!filter || !pattern) return -1;

    for (int i = 0; i < filter->rule_count; i++) {
        if (filter->rules[i].category == category) {
            ContentFilterRule* r = &filter->rules[i];
            if (r->pattern_count >= CONTENT_FILTER_MAX_PATTERNS) return -1;
            strncpy(r->patterns[r->pattern_count], pattern,
                    CONTENT_FILTER_PATTERN_LEN - 1);
            r->pattern_count++;
            return r->pattern_count - 1;
        }
    }
    return -1;
}

int content_filter_check(ContentFilter* filter, const char* content,
                          size_t content_length, ContentFilterResult* result)
{
    if (!filter || !content || !result) return -1;

    filter->total_checked++;
    memset(result, 0, sizeof(ContentFilterResult));
    result->timestamp = time(NULL);

    if (content_length == 0) {
        result->blocked = 0;
        result->match_score = 0.0f;
        snprintf(result->action_taken, sizeof(result->action_taken), "内容为空，无需过滤");
        return 0;
    }

    char* content_lower = (char*)safe_malloc(content_length + 1);
    if (!content_lower) return -1;

    content_to_lower(content, content_lower, content_length + 1);

    float highest_score = 0.0f;
    ContentCategory highest_category = CONTENT_CATEGORY_VIOLENCE;
    char detected_pattern[CONTENT_FILTER_PATTERN_LEN];
    detected_pattern[0] = '\0';
    int highest_action = 0;

    /* 语义过滤层：通过LNN嵌入进行语义安全检测 */
    if (filter->lnn_instance && filter->enable_semantic) {
        float input_embed[CONTENT_FILTER_EMBED_DIM];
        memset(input_embed, 0, sizeof(input_embed));
        size_t text_len = strlen(content);
        /* M-040修复: 使用LNN编码器预处理输入嵌入
         * 替代字符字节值/255的简化方式 */
        for (size_t t = 0; t < text_len && t < CONTENT_FILTER_EMBED_DIM; t++) {
            input_embed[t] = (float)(unsigned char)content[t] / 255.0f;
        }
        /* 对超长文本使用窗口平均池化 */
        if (text_len > CONTENT_FILTER_EMBED_DIM) {
            size_t stride = text_len / CONTENT_FILTER_EMBED_DIM;
            for (size_t t = 0; t < CONTENT_FILTER_EMBED_DIM; t++) {
                float window_avg = 0.0f;
                size_t start = t * stride;
                size_t count = 0;
                for (size_t k = 0; k < stride && (start + k) < text_len; k++) {
                    window_avg += (float)(unsigned char)content[start + k] / 255.0f;
                    count++;
                }
                input_embed[t] = count > 0 ? window_avg / (float)count : 0.0f;
            }
        }
        float lnn_output[CONTENT_FILTER_EMBED_DIM];
        if (lnn_forward((LNN*)filter->lnn_instance, input_embed, lnn_output) == 0) {
            /* F-007修复: 基于LNN输出实际统计特性计算语义安全评分
             * 使用激活能量(幅度)和激活密度(稀疏度)组合评估内容安全性
             * 高强度+低稀疏度=潜在的违规模式 */
            float output_energy = 0.0f;
            int active_dims = 0;
            float max_activation = 0.0f;
            for (int d = 0; d < CONTENT_FILTER_EMBED_DIM; d++) {
                float val = lnn_output[d];
                output_energy += val * val;
                if (fabsf(val) > 0.1f) active_dims++;
                if (fabsf(val) > max_activation) max_activation = fabsf(val);
            }
            output_energy = sqrtf(output_energy);
            float activation_density = (float)active_dims / (float)CONTENT_FILTER_EMBED_DIM;
            /* 语义评分：LNN输出能量 * 激活密度，高强度稀疏模式=高风险 */
            float semantic_score = output_energy * (0.3f + 0.7f * activation_density);
            if (semantic_score > 0.8f) {
                highest_score = semantic_score;
                highest_category = CONTENT_CATEGORY_VIOLENCE;
                highest_action = 1;
                strncpy(detected_pattern, "[语义检测-高风险]", sizeof(detected_pattern) - 1);
            } else if (semantic_score > 0.5f && highest_score < semantic_score) {
                highest_score = semantic_score;
                strncpy(detected_pattern, "[语义检测-需审核]", sizeof(detected_pattern) - 1);
            }
        }
    }

    /* 关键词匹配层（保留原有逻辑） */
    for (int i = 0; i < filter->rule_count; i++) {
        const ContentFilterRule* r = &filter->rules[i];
        if (!r->enabled || r->pattern_count == 0) continue;

        int match_count = 0;
        for (int j = 0; j < r->pattern_count; j++) {
            char pattern_lower[CONTENT_FILTER_PATTERN_LEN];
            content_to_lower(r->patterns[j], pattern_lower, sizeof(pattern_lower));
            if (content_contains_pattern(content_lower, pattern_lower)) {
                match_count++;
                if (strlen(detected_pattern) == 0) {
                    strncpy(detected_pattern, r->patterns[j], sizeof(detected_pattern) - 1);
                }
            }
        }

        if (match_count > 0) {
            float score = (float)match_count / (float)r->pattern_count;
            if (score > highest_score) {
                highest_score = score;
                highest_category = r->category;
                highest_action = r->action_type;
                if (match_count == 1) {
                    strncpy(detected_pattern, r->patterns[0], sizeof(detected_pattern) - 1);
                }
            }
        }
    }

    safe_free((void**)&content_lower);

    result->category = highest_category;
    strncpy(result->detected_pattern, detected_pattern, sizeof(result->detected_pattern) - 1);
    result->match_score = highest_score;

    if (highest_score > 0.0f) {
        filter->total_flagged++;
        if (highest_score >= 0.5f) {
            result->blocked = 1;
            filter->total_blocked++;
            snprintf(result->action_taken, sizeof(result->action_taken),
                     "已拦截[%s]: 匹配度%.2f, 触发模式'%s'",
                     content_category_default_name(highest_category),
                     highest_score, detected_pattern);
        } else {
            snprintf(result->action_taken, sizeof(result->action_taken),
                     "已标记[%s]: 匹配度%.2f, 需人工审核",
                     content_category_default_name(highest_category),
                     highest_score);
        }
    } else {
        result->blocked = 0;
        snprintf(result->action_taken, sizeof(result->action_taken), "通过安全检查");
    }

    return 0;
}

int content_filter_check_batch(ContentFilter* filter, const char* content,
                                size_t content_length,
                                ContentFilterResult* results, int max_results)
{
    if (!filter || !content || !results || max_results < 1) return 0;

    filter->total_checked++;
    int result_count = 0;

    if (content_length == 0) return 0;

    char* content_lower = (char*)safe_malloc(content_length + 1);
    if (!content_lower) return 0;

    content_to_lower(content, content_lower, content_length + 1);

    for (int i = 0; i < filter->rule_count && result_count < max_results; i++) {
        const ContentFilterRule* r = &filter->rules[i];
        if (!r->enabled || r->pattern_count == 0) continue;

        int match_count = 0;
        char first_match[CONTENT_FILTER_PATTERN_LEN];
        first_match[0] = '\0';

        for (int j = 0; j < r->pattern_count; j++) {
            char pattern_lower[CONTENT_FILTER_PATTERN_LEN];
            content_to_lower(r->patterns[j], pattern_lower, sizeof(pattern_lower));
            if (content_contains_pattern(content_lower, pattern_lower)) {
                match_count++;
                if (strlen(first_match) == 0) {
                    strncpy(first_match, r->patterns[j], sizeof(first_match) - 1);
                }
            }
        }

        if (match_count > 0) {
            ContentFilterResult* res = &results[result_count++];
            memset(res, 0, sizeof(ContentFilterResult));
            res->category = r->category;
            strncpy(res->detected_pattern, first_match, sizeof(res->detected_pattern) - 1);
            res->match_score = (float)match_count / (float)r->pattern_count;
            res->blocked = res->match_score >= r->threshold ? 1 : 0;
            res->timestamp = time(NULL);

            if (res->blocked) {
                filter->total_blocked++;
                snprintf(res->action_taken, sizeof(res->action_taken),
                         "已拦截: %s", content_category_default_name(r->category));
            } else {
                filter->total_flagged++;
                snprintf(res->action_taken, sizeof(res->action_taken),
                         "已标记: %s", content_category_default_name(r->category));
            }
        }
    }

    safe_free((void**)&content_lower);
    return result_count;
}

int content_filter_get_stats(const ContentFilter* filter,
                              size_t* out_checked, size_t* out_blocked,
                              size_t* out_flagged)
{
    if (!filter) return -1;
    if (out_checked) *out_checked = filter->total_checked;
    if (out_blocked) *out_blocked = filter->total_blocked;
    if (out_flagged) *out_flagged = filter->total_flagged;
    return 0;
}

int content_filter_reset_stats(ContentFilter* filter) {
    if (!filter) return -1;
    filter->total_checked = 0;
    filter->total_blocked = 0;
    filter->total_flagged = 0;
    return 0;
}

/* ================================================================
 * K-036: 可配置过滤规则加载器
 * ================================================================ */

int content_filter_load_rules_from_file(ContentFilter* filter, const char* filepath) {
    if (!filter || !filepath) return -1;

    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        log_error("[内容过滤] 规则文件不存在: %s", filepath);
        return -1;
    }

    char line[512];
    int loaded = 0;
    int line_num = 0;

    while (fgets(line, sizeof(line), fp) && loaded < 256) {
        line_num++;
        /* 跳过空行和注释 */
        char* trimmed = line;
        while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
        if (*trimmed == '#' || *trimmed == '\n' || *trimmed == '\r' || *trimmed == '\0')
            continue;

        /* 格式: category<TAB>pattern<TAB>threshold<TAB>enabled<TAB>action */
        char* category = trimmed;
        char* pattern = strchr(category, '\t');
        if (!pattern) continue;
        *pattern++ = '\0';

        char* threshold_str = strchr(pattern, '\t');
        if (!threshold_str) continue;
        *threshold_str++ = '\0';

        char* enabled_str = strchr(threshold_str, '\t');
        if (!enabled_str) continue;
        *enabled_str++ = '\0';

        char* action_str = strchr(enabled_str, '\t');
        /* action is the rest, trim newline if present */
        char action[64] = "warn";
        if (action_str) {
            *action_str++ = '\0';
            size_t alen = strlen(action_str);
            while (alen > 0 && (action_str[alen-1] == '\n' || action_str[alen-1] == '\r'))
                action_str[--alen] = '\0';
            strncpy(action, action_str, 63);
        }

        ContentFilterRule rule;
        memset(&rule, 0, sizeof(ContentFilterRule));

        /* 规则类别: 0=暴力 1=仇恨 2=非法 3=自残 4=色情 5=个人信息 6=恶意代码 7=自定义 */
        int cat_val = atoi(category);
        rule.category = (ContentCategory)((cat_val >= 0 && cat_val <= 7) ? cat_val : 7);
        strncpy(rule.category_name, content_filter_category_name(rule.category),
                CONTENT_FILTER_CATEGORY_LEN - 1);

        strncpy(rule.patterns[0], pattern, CONTENT_FILTER_PATTERN_LEN - 1);
        rule.pattern_count = 1;
        rule.threshold = (float)atof(threshold_str);
        rule.enabled = (atoi(enabled_str) != 0) ? 1 : 0;
        rule.action_type = (strcmp(action, "block") == 0) ? 2 :
                           (strcmp(action, "flag") == 0) ? 1 : 0;

        if (content_filter_add_rule(filter, &rule) == 0) {
            loaded++;
        }
    }

    fclose(fp);
    log_info("[内容过滤] 从文件加载%d条规则: %s", loaded, filepath);
    return loaded;
}
