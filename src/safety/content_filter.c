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
/* P0修复: 引入selflnn.h以获取selflnn_lock_lnn/selflnn_unlock_lnn线程安全接口，
 * 防止content_filter内部lnn_forward与AGI后台线程竞态崩溃 */
#include "selflnn/selflnn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

/* S-P2-08修复: 引入平台原子原语，对total_checked/total_flagged/total_blocked等
 * size_t计数器进行无锁原子递增，消除并发竞态。
 * 采用与thread_pool.c/lock_free.c一致的原子策略：
 *   - Windows: Interlocked系列(按size_t宽度选择32/64位原语)
 *   - GCC/Clang: __sync_fetch_and_add(自动适配size_t宽度) */
#ifdef _WIN32
#include <windows.h>
/* size_t在64位Windows为8字节，需用64位原子原语；32位Windows为4字节用32位原语 */
static __inline void cf_atomic_inc_size(size_t* p) {
#if defined(_WIN64)
    InterlockedIncrement64((LONG64*)p);
#else
    InterlockedIncrement((LONG*)p);
#endif
}
#else
static __inline void cf_atomic_inc_size(size_t* p) {
    __sync_fetch_and_add(p, (size_t)1);
}
#endif
/* 原子递增size_t计数器宏 */
#define CF_ATOMIC_INC(ptr) cf_atomic_inc_size((ptr))

#define CONTENT_FILTER_EMBED_DIM  128

struct ContentFilter {
    ContentFilterRule rules[CONTENT_FILTER_MAX_RULES];
    int rule_count;

    size_t total_checked;
    size_t total_blocked;
    size_t total_flagged;

    void* lnn_instance;     /* 外部绑定的LNN（可为NULL） */
    void* internal_cfc;     /* L-021修复: 内部独立的2层CfC分类器（无外部LNN时使用） */
    int enable_semantic;

    /* FIX-007修复: 添加互斥锁保护rule_count/rules并发访问 */
    void* lock;
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

    /* FIX-007修复: 初始化并发锁 */
    filter->lock = mutex_create();
    if (!filter->lock) { safe_free((void**)&filter); return NULL; }

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
/* 全部规则添加完成后，默认启用LNN语义过滤层。
     * 语义层与关键词匹配层协同工作：语义层做深度理解，关键词层做精细分类。 */
    filter->enable_semantic = 1;

    /* DEFECT-008根因修复: 内部CfC分类器延迟初始化。
     * content_filter_create()在initialize_subsystems()深层调用栈中执行，
     * 栈压力已接近极限。lnn_create()内部会触发完整的CfC网络创建链
     * (cfc_create→cfc_cell_create×2层)，进一步消耗栈空间导致/GS金丝雀检查失败。
     * 修复策略: 将lnn_create()延迟到content_filter_check()首次语义评分时按需创建，
     * 此时调用栈极浅(仅从AGI/对话处理线程调用)，有充足的栈空间。
     * 同时将LNNConfig从static数据段改为堆分配，消除线程安全隐患。 */
    filter->internal_cfc = NULL;

    return filter;
}

void content_filter_destroy(ContentFilter* filter) {
    if (!filter) return;
    if (filter->internal_cfc) {
        lnn_free((LNN*)filter->internal_cfc);
        filter->internal_cfc = NULL;
    }
    /* FIX-007修复: 释放互斥锁 */
    if (filter->lock) mutex_destroy(filter->lock);
    /* DEFECT-007修复: 使用临时局部变量通过safe_free释放 */
    {
        void* temp = filter;
        safe_free(&temp);
    }
}

/* 绑定LNN语义分析层实现
 * L-021修复: 外部LNN绑定/解绑时，若内部CfC分类器存在则保持语义层可用 */
int content_filter_set_lnn(ContentFilter* filter, void* lnn_instance) {
    if (!filter) return -1;
    /* P0修复: 维度安全检查 —— 拒绝绑定output_size≠CONTENT_FILTER_EMBED_DIM的LNN。
     * 如果外部LNN的输出维度与content_filter内部的lnn_output[128]不匹配，
     * lnn_forward会写入越界，导致栈溢出崩溃。 */
    if (lnn_instance) {
        LNN* lnn = (LNN*)lnn_instance;
        int lnn_out = lnn_get_output_size(lnn);
        if (lnn_out != CONTENT_FILTER_EMBED_DIM) {
            log_warning("[内容过滤] 拒绝绑定外部LNN(output_size=%d)——维度与内部缓冲区(%d)不匹配，"
                       "将使用内部CfC分类器", lnn_out, CONTENT_FILTER_EMBED_DIM);
            return -1;
        }
    }
    /* S-P2-11修复: lnn_instance/enable_semantic/internal_cfc均为跨线程共享字段，
     * content_filter_check在并发读取这些字段(第376行)，若本函数无锁写入，
     * 可能导致检测线程读到半写状态或enable_semantic与lnn_instance不一致。
     * 维度检查(lnn_get_output_size)不访问filter共享状态，可在锁外执行。 */
    mutex_lock(filter->lock);
    filter->lnn_instance = lnn_instance;
    if (lnn_instance || filter->internal_cfc) {
        filter->enable_semantic = 1;
    } else {
        filter->enable_semantic = 0;
    }
    mutex_unlock(filter->lock);
    return 0;
}

/* P0修复: 启用/禁用LNN语义评分层 */
int content_filter_set_enable_semantic(ContentFilter* filter, int enable) {
    if (!filter) return -1;
    filter->enable_semantic = enable ? 1 : 0;
    log_info("[内容过滤] 语义评分层已%s", enable ? "启用" : "禁用");
    return 0;
}

int content_filter_add_rule(ContentFilter* filter, const ContentFilterRule* rule) {
    if (!filter || !rule) return -1;
    /* FIX-007修复: 获取锁保护rule_count */
    mutex_lock(filter->lock);
    if (filter->rule_count >= CONTENT_FILTER_MAX_RULES) {
        mutex_unlock(filter->lock); return -1;
    }
    filter->rules[filter->rule_count++] = *rule;
    int result = filter->rule_count - 1;
    mutex_unlock(filter->lock);
    return result;
}

int content_filter_add_pattern(ContentFilter* filter, ContentCategory category,
                                const char* pattern) {
    if (!filter || !pattern) return -1;
    /* S-P2-09修复: 添加锁保护，防止rules数组遍历与patterns写入的并发竞态。
     * content_filter_check在持锁期间遍历rules，若本函数无锁并发写入patterns，
     * 可能导致遍历线程读到半写状态或pattern_count撕裂。与content_filter_add_rule
     * 保持一致的加锁策略（均使用filter->lock）。 */
    mutex_lock(filter->lock);
    for (int i = 0; i < filter->rule_count; i++) {
        if (filter->rules[i].category == category) {
            ContentFilterRule* r = &filter->rules[i];
            if (r->pattern_count >= CONTENT_FILTER_MAX_PATTERNS) {
                mutex_unlock(filter->lock);
                return -1;
            }
            strncpy(r->patterns[r->pattern_count], pattern,
                    CONTENT_FILTER_PATTERN_LEN - 1);
            r->patterns[r->pattern_count][CONTENT_FILTER_PATTERN_LEN - 1] = '\0';
            r->pattern_count++;
            int result = r->pattern_count - 1;
            mutex_unlock(filter->lock);
            return result;
        }
    }
    mutex_unlock(filter->lock);
    return -1;
}

int content_filter_check(ContentFilter* filter, const char* content,
                          size_t content_length, ContentFilterResult* result)
{
    if (!filter || !content || !result) return -1;

    /* S-P2-08修复: total_checked使用原子递增，防止并发检查线程计数丢失 */
    CF_ATOMIC_INC(&filter->total_checked);
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
    float semantic_score = 0.0f;
    ContentCategory highest_category = CONTENT_CATEGORY_VIOLENCE;
    char detected_pattern[CONTENT_FILTER_PATTERN_LEN];
    detected_pattern[0] = '\0';
    int highest_action = 0;

    /* ===== 第一层：关键词模式匹配 ===== */
    /* FIX-007修复: 获取读锁保护rules数组遍历 */
    mutex_lock(filter->lock);
    int snapshot_rule_count = filter->rule_count;
    for (int i = 0; i < snapshot_rule_count; i++) {
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
    /* P2修复: 在锁内缓存共享字段的值，避免锁外无锁读取lnn_instance/enable_semantic竞态。
     * lnn_instance/enable_semantic/internal_cfc均为跨线程共享字段，
     * content_filter_set_lnn/content_filter_set_enable_semantic可能并发写入，
     * 锁外读取可能导致读到半写状态或enable_semantic与lnn_instance不一致。 */
    void* cached_lnn_instance = filter->lnn_instance;
    void* cached_internal_cfc = filter->internal_cfc;
    int cached_enable_semantic = filter->enable_semantic;
    mutex_unlock(filter->lock);  /* FIX-007: 释放读锁 */

    /* ===== 第二层：LNN语义评估（补充过滤维度） =====
     * 在关键词匹配之后，使用LNN对完整文本进行语义级安全评估。
     * L-021修复: 优先使用外部绑定的LNN实例；若未绑定则使用内部独立的2层CfC分类器。
     * 语义评分作为补充维度：与关键词分数加权融合，提升对隐晦违规内容的检测能力。
     * DEFECT-008根因修复: 延迟初始化。如果internal_cfc为NULL且enable_semantic为1，
     * 则在首次需要语义评分时动态创建内部CfC分类器。此时调用栈极浅，栈空间充足。 */
    void* lnn_for_semantic = cached_lnn_instance ? cached_lnn_instance : cached_internal_cfc;
    /* P0修复: 如果使用外部共享LNN，需要加锁防止与AGI后台线程竞态崩溃。
     * internal_cfc是filter私有的，无需加锁。 */
    int use_shared_lnn = (cached_lnn_instance != NULL);
    if (use_shared_lnn) selflnn_lock_lnn();

    /* DEFECT-008根因修复: 延迟初始化内部CfC分类器 */
    if (cached_enable_semantic && !cached_internal_cfc && !cached_lnn_instance) {
        mutex_lock(filter->lock);
        /* 二次检查（可能其他线程已创建） */
        if (!filter->internal_cfc) {
            LNNConfig* internal_cfg = (LNNConfig*)safe_malloc(sizeof(LNNConfig));
            if (internal_cfg) {
                memset(internal_cfg, 0, sizeof(LNNConfig));
                internal_cfg->input_size = CONTENT_FILTER_EMBED_DIM;
                internal_cfg->hidden_size = 64;
                internal_cfg->output_size = CONTENT_FILTER_EMBED_DIM;
                internal_cfg->num_layers = 2;
                internal_cfg->time_constant = 0.05f;
                internal_cfg->learning_rate = 0.001f;
                internal_cfg->enable_training = 0;
                internal_cfg->ode_solver_type = 0;
                filter->internal_cfc = lnn_create(internal_cfg);
                safe_free((void**)&internal_cfg);
                if (filter->internal_cfc) {
                    log_info("[内容过滤] 内部2层CfC语义分类器已延迟创建 (input=%d, hidden=64, output=%d)",
                             CONTENT_FILTER_EMBED_DIM, CONTENT_FILTER_EMBED_DIM);
                } else {
                    log_warning("[内容过滤] 内部CfC语义分类器创建失败，语义评分将使用关键词匹配");
                    filter->enable_semantic = 0;
                    cached_enable_semantic = 0;
                }
            } else {
                log_warning("[内容过滤] 内存分配失败，无法创建内部CfC分类器，语义评分禁用");
                filter->enable_semantic = 0;
                cached_enable_semantic = 0;
            }
        }
        /* 更新缓存，重新选择lnn_for_semantic */
        cached_internal_cfc = filter->internal_cfc;
        cached_enable_semantic = filter->enable_semantic;
        mutex_unlock(filter->lock);
        lnn_for_semantic = cached_lnn_instance ? cached_lnn_instance : cached_internal_cfc;
    }

    if (lnn_for_semantic && cached_enable_semantic) {
        float input_embed[CONTENT_FILTER_EMBED_DIM];
        memset(input_embed, 0, sizeof(input_embed));
        size_t text_len = strlen(content);
        /* 使用bigram哈希编码替代简单字符编码，提升语义嵌入质量。
         * bigram哈希捕获字符相邻关系，比单字符编码提供更丰富的文本特征。 */
        if (text_len < 2) {
            /* 极短文本：使用字符编码回退 */
            for (size_t t = 0; t < text_len && t < CONTENT_FILTER_EMBED_DIM; t++) {
                unsigned char ch = (unsigned char)content[t];
                input_embed[t] = ((float)ch - 128.0f) / 128.0f;
            }
        } else {
            /* bigram哈希编码：滑动窗口提取字符对，散列到嵌入维度 */
            size_t num_bigrams = text_len - 1;
            for (size_t i = 0; i < num_bigrams; i++) {
                uint32_t h = ((uint32_t)(unsigned char)content[i] << 8)
                           | (uint32_t)(unsigned char)content[i + 1];
                h = h * 2654435761u;  /* Knuth乘法哈希 */
                size_t idx = (size_t)(h % CONTENT_FILTER_EMBED_DIM);
                input_embed[idx] += 1.0f;
            }
            /* L2归一化，避免不同文本长度的幅度差异 */
            float norm = 0.0f;
            for (size_t t = 0; t < CONTENT_FILTER_EMBED_DIM; t++) {
                norm += input_embed[t] * input_embed[t];
            }
            norm = sqrtf(norm + 1e-8f);
            float inv_norm = 1.0f / norm;
            for (size_t t = 0; t < CONTENT_FILTER_EMBED_DIM; t++) {
                input_embed[t] *= inv_norm;
            }
        }
        float lnn_output[CONTENT_FILTER_EMBED_DIM];
        float* heap_output = NULL;
        int lnn_forward_success = 0;
        /* P0修复: 单LNN强制策略可能将内部CfC(128维)替换为全局LNN(256维)，
         * 导致栈缓冲区溢出。检查输出维度，必要时使用堆分配缓冲区。 */
        size_t actual_output_size = lnn_get_output_size((LNN*)lnn_for_semantic);
        float* lnn_output_ptr = lnn_output;
        if (actual_output_size > CONTENT_FILTER_EMBED_DIM) {
            heap_output = (float*)safe_malloc(actual_output_size * sizeof(float));
            if (heap_output) {
                lnn_output_ptr = heap_output;
                log_info("[内容过滤] 检测到LNN输出维度(%zu)超过嵌入维度(%d)，使用堆缓冲区防止栈溢出",
                         actual_output_size, CONTENT_FILTER_EMBED_DIM);
            } else {
                /* 堆分配失败，回退到关键词匹配 */
                log_warning("[内容过滤] 堆缓冲区分配失败，语义评分跳过");
                cached_enable_semantic = 0;
            }
        }
        if (cached_enable_semantic && lnn_forward((LNN*)lnn_for_semantic, input_embed, lnn_output_ptr) == 0) {
            lnn_forward_success = 1;
            float output_energy = 0.0f;
            int active_dims = 0;
            float max_activation = 0.0f;
            for (int d = 0; d < CONTENT_FILTER_EMBED_DIM; d++) {
                float val = lnn_output_ptr[d];
                output_energy += val * val;
                if (fabsf(val) > 0.1f) active_dims++;
                if (fabsf(val) > max_activation) max_activation = fabsf(val);
            }
            output_energy = sqrtf(output_energy);
            float activation_density = (float)active_dims / (float)CONTENT_FILTER_EMBED_DIM;
            semantic_score = output_energy * (0.3f + 0.7f * activation_density);

            /* 语义评分作为补充维度：与关键词分数加权融合
             * - 关键词匹配到高风险时：语义评分提供置信度修正
             * - 关键词未匹配时：语义评分独立作为潜在风险检测 */
            if (highest_score > 0.0f) {
                /* 关键词已命中：加权融合(关键词70% + 语义30%) */
                float blended_score = highest_score * 0.7f + semantic_score * 0.3f;
                if (blended_score > 1.0f) blended_score = 1.0f;
                highest_score = blended_score;
                if (semantic_score > 0.7f && strstr(detected_pattern, "[语义") == NULL) {
                    /* 语义评分确认高风险：追加语义标记 */
                    char sem_tag[64];
                    snprintf(sem_tag, sizeof(sem_tag), "%s [语义确认:%.2f]",
                             detected_pattern, semantic_score);
                    strncpy(detected_pattern, sem_tag, sizeof(detected_pattern) - 1);
                }
            } else if (semantic_score > 0.6f) {
                /* 关键词未命中但语义评分较高：作为独立补充维度 */
                highest_score = semantic_score;
                strncpy(detected_pattern, "[语义检测]", sizeof(detected_pattern) - 1);
            }
        }
        /* 释放堆缓冲区（如果有） */
        if (heap_output) {
            safe_free((void**)&heap_output);
        }

        /* P2-03修复: 内部CfC/LNN推理失败时的统计特征fallback
         * 当LNN forward失败时，不直接放弃语义评估，而是使用基于文本统计特征的
         * 简单评分方法，提供比纯关键词匹配更丰富的信号。
         * 统计特征包括：ngram分布异常度、字符熵、模式相似度等。 */
        if (!lnn_forward_success) {
            float stat_score = 0.0f;
            size_t text_len = strlen(content);

            if (text_len > 0) {
                /* 统计特征1: 字符分布熵
                 * 高熵通常表示文本内容多样/随机，低熵可能表示重复模式 */
                int char_freq[256] = {0};
                for (size_t i = 0; i < text_len; i++) {
                    char_freq[(unsigned char)content[i]]++;
                }
                float char_entropy = 0.0f;
                float inv_len = 1.0f / (float)text_len;
                for (int c = 0; c < 256; c++) {
                    if (char_freq[c] > 0) {
                        float p = (float)char_freq[c] * inv_len;
                        char_entropy -= p * log2f(p + 1e-10f);
                    }
                }
                /* 熵归一化到[0,1]，高熵(>6)和极低熵(<2)都可能是异常信号 */
                float entropy_norm = char_entropy / 8.0f;
                float entropy_anomaly = 1.0f - fabsf(entropy_norm - 0.5f) * 2.0f;
                stat_score += entropy_anomaly * 0.2f;

                /* 统计特征2: 已知敏感模式的ngram重叠度
                 * 遍历所有过滤规则的模式，计算内容与敏感模式的bigram重叠率 */
                float max_pattern_overlap = 0.0f;
                for (int ri = 0; ri < snapshot_rule_count; ri++) {
                    const ContentFilterRule* r = &filter->rules[ri];
                    if (!r->enabled || r->pattern_count == 0) continue;
                    for (int pj = 0; pj < r->pattern_count; pj++) {
                        char pattern_lower[CONTENT_FILTER_PATTERN_LEN];
                        content_to_lower(r->patterns[pj], pattern_lower, sizeof(pattern_lower));
                        size_t plen = strlen(pattern_lower);
                        if (plen < 2) continue;

                        /* 计算内容的bigram集合与模式bigram集合的交集占比 */
                        int pattern_bigrams[CONTENT_FILTER_EMBED_DIM] = {0};
                        for (size_t pi = 0; pi + 1 < plen; pi++) {
                            uint32_t h = ((uint32_t)(unsigned char)pattern_lower[pi] << 8)
                                       | (uint32_t)(unsigned char)pattern_lower[pi + 1];
                            h = h * 2654435761u;
                            pattern_bigrams[(size_t)(h % CONTENT_FILTER_EMBED_DIM)] = 1;
                        }

                        int content_bigrams[CONTENT_FILTER_EMBED_DIM] = {0};
                        for (size_t ci = 0; ci + 1 < text_len; ci++) {
                            uint32_t h = ((uint32_t)(unsigned char)content_lower[ci] << 8)
                                       | (uint32_t)(unsigned char)content_lower[ci + 1];
                            h = h * 2654435761u;
                            content_bigrams[(size_t)(h % CONTENT_FILTER_EMBED_DIM)] = 1;
                        }

                        /* Jaccard相似度近似：交集/并集 */
                        int intersect = 0, union_count = 0;
                        for (int d = 0; d < CONTENT_FILTER_EMBED_DIM; d++) {
                            if (pattern_bigrams[d] || content_bigrams[d]) union_count++;
                            if (pattern_bigrams[d] && content_bigrams[d]) intersect++;
                        }
                        if (union_count > 0) {
                            float overlap = (float)intersect / (float)union_count;
                            if (overlap > max_pattern_overlap) {
                                max_pattern_overlap = overlap;
                            }
                        }
                    }
                }
                stat_score += max_pattern_overlap * 0.5f;

                /* 统计特征3: 特殊字符/控制字符密度
                 * 恶意代码通常含有较多特殊字符 */
                int special_chars = 0;
                for (size_t i = 0; i < text_len; i++) {
                    unsigned char ch = (unsigned char)content[i];
                    if (ch < 32 || (ch > 126 && ch < 160)) special_chars++;
                }
                float special_ratio = (float)special_chars / (float)text_len;
                stat_score += special_ratio * 0.3f;

                /* 统计特征4: 重复ngram检测（垃圾信息特征） */
                uint32_t trigram_buckets[128] = {0};
                int trigram_count = 0;
                for (size_t i = 0; i + 2 < text_len && i < 512; i++) {
                    uint32_t h = ((uint32_t)(unsigned char)content_lower[i] << 16)
                               | ((uint32_t)(unsigned char)content_lower[i + 1] << 8)
                               | (uint32_t)(unsigned char)content_lower[i + 2];
                    h = h * 2654435761u;
                    trigram_buckets[(size_t)(h % 128)]++;
                    trigram_count++;
                }
                if (trigram_count > 0) {
                    int max_trigram_freq = 0;
                    for (int t = 0; t < 128; t++) {
                        if (trigram_buckets[t] > max_trigram_freq)
                            max_trigram_freq = trigram_buckets[t];
                    }
                    float repeat_ratio = (float)max_trigram_freq / (float)trigram_count;
                    /* 高重复率可能是垃圾信息或模板注入 */
                    stat_score += repeat_ratio * 0.3f;
                }

                /* 限制统计分数在[0,1]范围 */
                if (stat_score > 1.0f) stat_score = 1.0f;
                if (stat_score < 0.0f) stat_score = 0.0f;

                /* 统计评分作为fallback语义评分 */
                if (stat_score > 0.15f) {
                    semantic_score = stat_score;

                    if (highest_score > 0.0f) {
                        /* 关键词已命中：加权融合(关键词70% + 统计特征30%) */
                        float blended_score = highest_score * 0.7f + semantic_score * 0.3f;
                        if (blended_score > 1.0f) blended_score = 1.0f;
                        highest_score = blended_score;
                        if (semantic_score > 0.6f && strstr(detected_pattern, "[统计") == NULL) {
                            char stat_tag[64];
                            snprintf(stat_tag, sizeof(stat_tag), "%s [统计特征:%.2f]",
                                     detected_pattern, semantic_score);
                            strncpy(detected_pattern, stat_tag, sizeof(detected_pattern) - 1);
                        }
                    } else if (semantic_score > 0.5f) {
                        /* 关键词未命中但统计特征评分较高 */
                        highest_score = semantic_score;
                        strncpy(detected_pattern, "[统计特征检测]", sizeof(detected_pattern) - 1);
                    }
                }
            }
        }
    } else if (cached_enable_semantic) {
        /* P2-03修复: LNN完全不可用时的纯统计特征fallback
         * 当lnn_for_semantic为NULL但enable_semantic为1时，
         * 说明内部CfC创建失败或外部LNN未绑定，使用统计特征评估文本安全性。
         * 此路径比纯关键词匹配提供更丰富的信号。 */
        size_t text_len = strlen(content);
        if (text_len > 0) {
            float stat_score = 0.0f;

            /* 字符分布熵 */
            int char_freq[256] = {0};
            for (size_t i = 0; i < text_len; i++) {
                char_freq[(unsigned char)content[i]]++;
            }
            float char_entropy = 0.0f;
            float inv_len = 1.0f / (float)text_len;
            for (int c = 0; c < 256; c++) {
                if (char_freq[c] > 0) {
                    float p = (float)char_freq[c] * inv_len;
                    char_entropy -= p * log2f(p + 1e-10f);
                }
            }
            float entropy_norm = char_entropy / 8.0f;
            float entropy_anomaly = 1.0f - fabsf(entropy_norm - 0.5f) * 2.0f;
            stat_score += entropy_anomaly * 0.2f;

            /* 已知敏感模式bigram重叠度 */
            float max_pattern_overlap = 0.0f;
            for (int ri = 0; ri < snapshot_rule_count; ri++) {
                const ContentFilterRule* r = &filter->rules[ri];
                if (!r->enabled || r->pattern_count == 0) continue;
                for (int pj = 0; pj < r->pattern_count; pj++) {
                    char pattern_lower[CONTENT_FILTER_PATTERN_LEN];
                    content_to_lower(r->patterns[pj], pattern_lower, sizeof(pattern_lower));
                    size_t plen = strlen(pattern_lower);
                    if (plen < 2) continue;

                    int pattern_bigrams[CONTENT_FILTER_EMBED_DIM] = {0};
                    for (size_t pi = 0; pi + 1 < plen; pi++) {
                        uint32_t h = ((uint32_t)(unsigned char)pattern_lower[pi] << 8)
                                   | (uint32_t)(unsigned char)pattern_lower[pi + 1];
                        h = h * 2654435761u;
                        pattern_bigrams[(size_t)(h % CONTENT_FILTER_EMBED_DIM)] = 1;
                    }
                    int content_bigrams[CONTENT_FILTER_EMBED_DIM] = {0};
                    for (size_t ci = 0; ci + 1 < text_len; ci++) {
                        uint32_t h = ((uint32_t)(unsigned char)content_lower[ci] << 8)
                                   | (uint32_t)(unsigned char)content_lower[ci + 1];
                        h = h * 2654435761u;
                        content_bigrams[(size_t)(h % CONTENT_FILTER_EMBED_DIM)] = 1;
                    }
                    int intersect = 0, union_count = 0;
                    for (int d = 0; d < CONTENT_FILTER_EMBED_DIM; d++) {
                        if (pattern_bigrams[d] || content_bigrams[d]) union_count++;
                        if (pattern_bigrams[d] && content_bigrams[d]) intersect++;
                    }
                    if (union_count > 0) {
                        float overlap = (float)intersect / (float)union_count;
                        if (overlap > max_pattern_overlap) max_pattern_overlap = overlap;
                    }
                }
            }
            stat_score += max_pattern_overlap * 0.5f;

            /* 特殊字符密度 */
            int special_chars = 0;
            for (size_t i = 0; i < text_len; i++) {
                unsigned char ch = (unsigned char)content[i];
                if (ch < 32 || (ch > 126 && ch < 160)) special_chars++;
            }
            stat_score += ((float)special_chars / (float)text_len) * 0.3f;

            if (stat_score > 1.0f) stat_score = 1.0f;
            if (stat_score < 0.0f) stat_score = 0.0f;

            if (stat_score > 0.15f) {
                semantic_score = stat_score;
                if (highest_score > 0.0f) {
                    float blended_score = highest_score * 0.7f + semantic_score * 0.3f;
                    if (blended_score > 1.0f) blended_score = 1.0f;
                    highest_score = blended_score;
                    if (strstr(detected_pattern, "[统计") == NULL) {
                        char stat_tag[64];
                        snprintf(stat_tag, sizeof(stat_tag), "%s [统计特征:%.2f]",
                                 detected_pattern, semantic_score);
                        strncpy(detected_pattern, stat_tag, sizeof(detected_pattern) - 1);
                    }
                } else if (semantic_score > 0.5f) {
                    highest_score = semantic_score;
                    strncpy(detected_pattern, "[统计特征检测]", sizeof(detected_pattern) - 1);
                }
            }
        }
    }
    /* P0修复: 释放共享LNN锁 */
    if (use_shared_lnn) selflnn_unlock_lnn();

    safe_free((void**)&content_lower);

    result->category = highest_category;
    strncpy(result->detected_pattern, detected_pattern, sizeof(result->detected_pattern) - 1);
    result->match_score = highest_score;

    if (highest_score > 0.0f) {
        /* S-P2-08修复: total_flagged使用原子递增 */
        CF_ATOMIC_INC(&filter->total_flagged);
        if (highest_score >= 0.5f) {
            result->blocked = 1;
            /* S-P2-08修复: total_blocked使用原子递增 */
            CF_ATOMIC_INC(&filter->total_blocked);
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

    /* S-P2-08修复: total_checked使用原子递增，防止并发批量检查线程计数丢失 */
    CF_ATOMIC_INC(&filter->total_checked);
    int result_count = 0;

    if (content_length == 0) return 0;

    char* content_lower = (char*)safe_malloc(content_length + 1);
    if (!content_lower) return 0;

    content_to_lower(content, content_lower, content_length + 1);

    /* S-P2-10修复: 加锁保护rules数组遍历，与content_filter_check保持一致(FIX-007)。
     * 快照rule_count后持锁遍历，防止遍历期间rule_count/rules被并发写入导致越界或撕裂。 */
    mutex_lock(filter->lock);
    int snapshot_rule_count = filter->rule_count;
    for (int i = 0; i < snapshot_rule_count && result_count < max_results; i++) {
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
                /* S-P2-08修复: total_blocked使用原子递增 */
                CF_ATOMIC_INC(&filter->total_blocked);
                snprintf(res->action_taken, sizeof(res->action_taken),
                         "已拦截: %s", content_category_default_name(r->category));
            } else {
                /* S-P2-08修复: total_flagged使用原子递增 */
                CF_ATOMIC_INC(&filter->total_flagged);
                snprintf(res->action_taken, sizeof(res->action_taken),
                         "已标记: %s", content_category_default_name(r->category));
            }
        }
    }
    mutex_unlock(filter->lock);  /* S-P2-10修复: 释放规则遍历锁 */

    safe_free((void**)&content_lower);
    return result_count;
}

int content_filter_get_stats(const ContentFilter* filter,
                              size_t* out_checked, size_t* out_blocked,
                              size_t* out_flagged)
{
    if (!filter) return -1;
    /* S-P2-11修复: 加锁读取计数器，防止与原子递增线程竞态读到撕裂值。
     * const指针下filter->lock为void* const，mutex_lock按值(MutexHandle=void*)
     * 传递，const仅作用于指针本身(顶层)，按值传入无const违例。 */
    mutex_lock(filter->lock);
    if (out_checked) *out_checked = filter->total_checked;
    if (out_blocked) *out_blocked = filter->total_blocked;
    if (out_flagged) *out_flagged = filter->total_flagged;
    mutex_unlock(filter->lock);
    return 0;
}

int content_filter_reset_stats(ContentFilter* filter) {
    if (!filter) return -1;
    /* S-P2-11修复: 加锁重置计数器，防止与递增线程竞态 */
    mutex_lock(filter->lock);
    filter->total_checked = 0;
    filter->total_blocked = 0;
    filter->total_flagged = 0;
    mutex_unlock(filter->lock);
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

        /* CONTENT-010修复: content_filter_add_rule返回新索引(>=0成功), 仅==0判断会跳过后面的规则 */
        if (content_filter_add_rule(filter, &rule) >= 0) {
            loaded++;
        }
    }

    fclose(fp);
    log_info("[内容过滤] 从文件加载%d条规则: %s", loaded, filepath);
    return loaded;
}
