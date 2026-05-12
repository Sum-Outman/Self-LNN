/**
 * @file content_filter.h
 * @brief 内容安全过滤系统接口（输出护栏/Guardrails）
 *
 * 提供AI生成内容的安全过滤机制，确保输出内容符合安全规范。
 * 基于模式匹配的安全内容检测，纯C实现，无第三方依赖。
 */

#ifndef SELFLNN_CONTENT_FILTER_H
#define SELFLNN_CONTENT_FILTER_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CONTENT_FILTER_MAX_RULES      256
#define CONTENT_FILTER_MAX_PATTERNS   64
#define CONTENT_FILTER_PATTERN_LEN    128
#define CONTENT_FILTER_CATEGORY_LEN   64

typedef enum {
    CONTENT_CATEGORY_VIOLENCE = 0,
    CONTENT_CATEGORY_HATE_SPEECH,
    CONTENT_CATEGORY_ILLEGAL,
    CONTENT_CATEGORY_SELF_HARM,
    CONTENT_CATEGORY_PORNOGRAPHY,
    CONTENT_CATEGORY_PERSONAL_INFO,
    CONTENT_CATEGORY_MALICIOUS_CODE,
    CONTENT_CATEGORY_CUSTOM
} ContentCategory;

typedef struct {
    ContentCategory category;
    char category_name[CONTENT_FILTER_CATEGORY_LEN];
    char patterns[CONTENT_FILTER_MAX_PATTERNS][CONTENT_FILTER_PATTERN_LEN];
    int pattern_count;
    float threshold;
    int enabled;
    int action_type;
} ContentFilterRule;

typedef struct {
    ContentCategory category;
    char detected_pattern[CONTENT_FILTER_PATTERN_LEN];
    float match_score;
    int blocked;
    char action_taken[256];
    time_t timestamp;
} ContentFilterResult;

typedef struct ContentFilter ContentFilter;

ContentFilter* content_filter_create(void);

void content_filter_destroy(ContentFilter* filter);

int content_filter_add_rule(ContentFilter* filter, const ContentFilterRule* rule);

int content_filter_add_pattern(ContentFilter* filter, ContentCategory category,
                                const char* pattern);

int content_filter_check(ContentFilter* filter, const char* content,
                          size_t content_length, ContentFilterResult* result);

int content_filter_check_batch(ContentFilter* filter, const char* content,
                                size_t content_length,
                                ContentFilterResult* results, int max_results);

const char* content_filter_category_name(ContentCategory category);

int content_filter_get_stats(const ContentFilter* filter,
                              size_t* out_checked, size_t* out_blocked,
                              size_t* out_flagged);

int content_filter_reset_stats(ContentFilter* filter);

/* ================================================================
 * K-036: 可配置过滤规则文件加载
 * ================================================================ */

/**
 * @brief K-036: 从文件加载内容过滤规则
 *
 * 文件格式: 每行一条规则，制表符分隔
 *   category\tpattern\tthreshold\tenabled\taction
 *
 * @param filter 过滤句柄
 * @param filepath 规则文件路径
 * @return 成功加载的规则数，失败返回-1
 */
int content_filter_load_rules_from_file(ContentFilter* filter, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif
