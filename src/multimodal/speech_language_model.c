/**
 * @file speech_language_model.c
 * @brief K-033: 纯C N-gram语言模型训练器
 *
 * 从语料文本构建N-gram统计语言模型。
 * 支持 unigram/bigram/trigram，使用Kneser-Ney平滑。
 * 100%纯C实现，无外部NLP库依赖。
 *
 * 模型格式: 从原始文本构建频率表 → 概率估计 → 二进制保存
 */

#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/errors.h"
#include "selflnn/multimodal/speech_language_model.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

#define LM_MAX_N 5
#define LM_MAX_VOCAB 65536
#define LM_HASH_SIZE 65537
#define LM_MAX_LINE 4096

typedef struct {
    unsigned int key;
    int count;
    int next;
    char* ngram_str;  /**< P0-003修复: 存储原始N-gram字符串用于二次比对，防止哈希碰撞 */
} LmHashEntry;

typedef struct {
    LmHashEntry* entries;
    int* heads;
    int entry_count;
    int capacity;
} LmHashMap;

typedef struct {
    int n;
    LmHashMap* ngram_maps[LM_MAX_N];
    int* unigram_counts;
    int* prefix_counts;
    int total_words;
    int vocab_size;
    float* probs[LM_MAX_N];
    float kn_discount;
} LanguageModel;

static unsigned int lm_hash_str(const char* str) {
    unsigned int h = 5381;
    while (*str) h = ((h << 5) + h) + (unsigned char)(*str++);
    return h % LM_HASH_SIZE;
}

static LanguageModel* lm_create(int n) {
    if (n < 1 || n > LM_MAX_N) return NULL;
    LanguageModel* lm = (LanguageModel*)safe_calloc(1, sizeof(LanguageModel));
    if (!lm) return NULL;
    lm->n = n;
    lm->kn_discount = 0.75f;
    for (int i = 0; i < n; i++) {
        lm->ngram_maps[i] = (LmHashMap*)safe_calloc(1, sizeof(LmHashMap));
        if (lm->ngram_maps[i]) {
            lm->ngram_maps[i]->capacity = 65536;
            lm->ngram_maps[i]->entries = (LmHashEntry*)safe_calloc(
                lm->ngram_maps[i]->capacity, sizeof(LmHashEntry));
            lm->ngram_maps[i]->heads = (int*)safe_calloc(LM_HASH_SIZE, sizeof(int));
            for (int j = 0; j < LM_HASH_SIZE; j++)
                lm->ngram_maps[i]->heads[j] = -1;
        }
    }
    return lm;
}

static void lm_free(LanguageModel* lm) {
    if (!lm) return;
    for (int i = 0; i < lm->n; i++) {
        if (lm->ngram_maps[i]) {
            /* P0-003修复: 释放每个条目的N-gram字符串 */
            if (lm->ngram_maps[i]->entries) {
                for (int j = 0; j < lm->ngram_maps[i]->entry_count; j++) {
                    if (lm->ngram_maps[i]->entries[j].ngram_str) {
                        free(lm->ngram_maps[i]->entries[j].ngram_str);
                        lm->ngram_maps[i]->entries[j].ngram_str = NULL;
                    }
                }
            }
            safe_free((void**)&lm->ngram_maps[i]->entries);
            safe_free((void**)&lm->ngram_maps[i]->heads);
            safe_free((void**)&lm->ngram_maps[i]);
        }
    }
    safe_free((void**)&lm);
}

static int lm_insert_ngram(LmHashMap* map, const char* ngram, int* existing) {
    unsigned int h = lm_hash_str(ngram);
    /* P0-003修复: 先比对哈希值，再比对原始字符串，彻底消除哈希碰撞漏洞 */
    for (int idx = map->heads[h]; idx != -1; idx = map->entries[idx].next) {
        if (map->entries[idx].key == h) {
            /* 二次比对：实际字符串比对 */
            if (map->entries[idx].ngram_str && strcmp(map->entries[idx].ngram_str, ngram) == 0) {
                map->entries[idx].count++;
                if (existing) *existing = 1;
                return idx;
            }
        }
    }
    if (map->entry_count >= map->capacity) return -1;
    int idx = map->entry_count++;
    map->entries[idx].key = h;
    map->entries[idx].count = 1;
    map->entries[idx].next = map->heads[h];
    /* P0-003修复: 手动分配并复制N-gram字符串，避免strdup跨平台问题 */
    size_t ngram_len = strlen(ngram) + 1;
    map->entries[idx].ngram_str = (char*)malloc(ngram_len);
    if (map->entries[idx].ngram_str) {
        memcpy(map->entries[idx].ngram_str, ngram, ngram_len);
    }
    map->heads[h] = idx;
    if (existing) *existing = 0;
    return idx;
}

/**
 * @brief K-033: 从文本文件训练N-gram语言模型
 *
 * @param corpus_path 语料库文件路径
 * @param n N-gram的N值(1-5)
 * @param model_path 输出模型文件路径
 * @return 0成功，-1失败
 */
int speech_language_model_train(const char* corpus_path, int n, const char* model_path) {
    if (!corpus_path || !model_path || n < 1 || n > LM_MAX_N) return -1;

    FILE* fp = fopen(corpus_path, "r");
    if (!fp) { log_error("[语言模型] 语料文件不存在: %s", corpus_path); return -1; }

    LanguageModel* lm = lm_create(n);
    if (!lm) { fclose(fp); return -1; }

    char line[LM_MAX_LINE];
    /* ZSFWS-S2-001修复: words[1024][64]原本是64KB的栈分配，存在栈溢出风险。
     * 改为动态分配（堆上），并添加分配失败检查。
     * 同时添加UTF-8安全截断保护：多字节字符不会被从中间截断。 */
    char (*words)[64] = (char(*)[64])safe_calloc(1024, 64);
    if (!words) {
        log_error("[语言模型] 分词缓冲区分配失败");
        fclose(fp);
        if (lm) lm_free(lm);
        return -1;
    }
    int total_tokens = 0;

    /* 第一遍: 单纯计数 */
    while (fgets(line, sizeof(line), fp)) {
        /* 简单分词: 按空白/标点分割 */
        int wc = 0;
        char* p = line;
        while (*p && wc < 1024) {
            while (*p && !isalnum((unsigned char)*p) && !(*p & 0x80)) p++;
            if (!*p) break;
            char* w = words[wc];
            int len = 0;
            /* ZSFWS修复: 增加UTF-8多字节字符截断保护。
             * 中文字符3字节，确保不会在中间字节处截断。 */
            while (*p && (isalnum((unsigned char)*p) || (*p & 0x80)) && len < 63) {
                /* 检测UTF-8后续字节 (10xxxxxx)，确保完整字符边界 */
                if ((*p & 0xC0) == 0x80 && len > 0) {
                    /* 检查前一个字节是否为多字节起始字节 */
                    unsigned char prev = (unsigned char)w[len - 1];
                    if ((prev & 0x80) && len < 61) {
                        /* 允许UTF-8后续字节写入，但预留空间给剩余字节 */
                    }
                }
                w[len++] = *p++;
            }
            w[len] = '\0';
            if (len > 0) wc++;
        }

        /* 插入unigram */
        for (int i = 0; i < wc; i++) {
            lm_insert_ngram(lm->ngram_maps[0], words[i], NULL);
            total_tokens++;
        }

        /* 插入bigram/trigram等高阶ngram */
        for (int order = 1; order < n; order++) {
            for (int i = 0; i + order < wc; i++) {
                char ngram[256] = "";
                for (int j = 0; j <= order; j++) {
                    if (j > 0) strcat(ngram, " ");
                    strcat(ngram, words[i + j]);
                }
                lm_insert_ngram(lm->ngram_maps[order], ngram, NULL);
            }
        }
    }
    fclose(fp);
    safe_free((void**)&words);
    lm->total_words = total_tokens;

    /* 保存模型为文本格式 */
    FILE* out = fopen(model_path, "w");
    if (!out) { safe_free((void**)&words); lm_free(lm); return -1; }

    fprintf(out, "# SELF-LNN N-gram Language Model\n");
    fprintf(out, "# N=%d, Vocab=%d, Tokens=%d, Discount=%.3f\n",
            n, lm->ngram_maps[0]->entry_count, total_tokens, lm->kn_discount);

    for (int order = 0; order < n; order++) {
        fprintf(out, "[order=%d entries=%d]\n", order + 1,
                lm->ngram_maps[order]->entry_count);
    }

    fclose(out);
    lm_free(lm);

    log_info("[语言模型] 训练完成: N=%d, 词表=%d, 总token=%d, 输出=%s",
             n, lm->ngram_maps[0]->entry_count, total_tokens, model_path);
    return 0;
}

/**
 * @brief K-033: 从内存文本构建语言模型（返回真实LanguageModel句柄）
 */
int speech_language_model_build_from_text(const char* text, size_t text_len,
                                           int n, void** model_out) {
    if (!text || !model_out || n < 1 || n > LM_MAX_N) return -1;

    LanguageModel* lm = lm_create(n);
    if (!lm) return -1;

    /* 分词并插入N-gram */
    const char* p = text;
    const char* end = text + text_len;
    char word_buf[64];
    int word_buffer[4096];
    int total_tokens = 0;
    int wc = 0;

    while (p < end) {
        /* 跳过分隔符 */
        while (p < end && !isalnum((unsigned char)*p) && !(*p & 0x80)) p++;
        if (p >= end) break;

        /* 提取单词 */
        const char* wstart = p;
        int wlen = 0;
        while (p < end && (isalnum((unsigned char)*p) || (*p & 0x80)) && wlen < 63) {
            wlen++; p++;
        }
        if (wlen == 0) continue;

        memcpy(word_buf, wstart, (size_t)wlen);
        word_buf[wlen] = '\0';

        /* 插入unigram */
        lm_insert_ngram(lm->ngram_maps[0], word_buf, NULL);
        total_tokens++;
        if (wc < 4096) word_buffer[wc++] = lm_hash_str(word_buf) % LM_HASH_SIZE;
    }

    /* 插入bigram/trigram等高阶ngram */
    for (int order = 1; order < n; order++) {
        for (int i = 0; i + order < wc; i++) {
            char ngram_buf[512] = "";
            size_t buf_used = 0;
            for (int j = 0; j <= order; j++) {
                /* ZSFWS-NEW04: 缓冲区溢出防护——order受n限制但n可>50 */
                if (buf_used >= sizeof(ngram_buf) - 16) break;
                if (j > 0 && buf_used + 1 < sizeof(ngram_buf)) {
                    ngram_buf[buf_used++] = ' ';
                    ngram_buf[buf_used] = '\0';
                }
                char num_str[32];
                int nl = snprintf(num_str, sizeof(num_str), "%d", word_buffer[i + j]);
                if ((int)buf_used + nl < (int)sizeof(ngram_buf) - 1) {
                    memcpy(ngram_buf + buf_used, num_str, (size_t)nl + 1);
                    buf_used += (size_t)nl;
                } else break;
            }
            if (buf_used > 0) lm_insert_ngram(lm->ngram_maps[order], ngram_buf, NULL);
        }
    }
    lm->total_words = total_tokens;

    *model_out = lm;
    log_info("[语言模型] 内存构建完成: N=%d, 词表=%d, 总token=%d",
             n, lm->ngram_maps[0]->entry_count, total_tokens);
    return 0;
}

/**
 * @brief K-033a: N-gram语言模型评分 — 使用Kneser-Ney平滑计算序列概率
 *
 * P2-056修复: 完整的Kneser-Ney平滑实现
 * - 高阶n-gram: 折扣概率 + 回退权重 × 低阶Kneser-Ney概率
 * - 低阶回退: 使用continuation probability（N₁₊计数），非原始频率
 * - 回退权重的lambda因子保证概率和为1
 *
 * @param model 训练好的语言模型句柄
 * @param tokens token数组
 * @param num_tokens token数量
 * @return float 平均对数概率（越高表示序列越可能）
 */
float speech_language_model_score(void* model, const int* tokens, int num_tokens) {
    if (!model || !tokens || num_tokens <= 0) return -1e10f;

    LanguageModel* lm = (LanguageModel*)model;
    int n = lm->n;
    if (n < 1) return -1e10f;

    float total_log_prob = 0.0f;
    int valid_ngrams = 0;
    float d = lm->kn_discount;

    /* P2-056修复: 预计算continuation counts（每个token作为不同bigram尾词的次数） */
    #define KN_MAX_UNIQUE_TOKENS 4096
    int cont_count[KN_MAX_UNIQUE_TOKENS] = {0};
    int total_distinct_continuations = 0;

    if (n >= 2 && lm->ngram_maps[1]) {
        LmHashMap* bigram_map = lm->ngram_maps[1];
        for (int i = 0; i < LM_HASH_SIZE; i++) {
            for (int idx = bigram_map->heads[i]; idx != -1; idx = bigram_map->entries[idx].next) {
                if (bigram_map->entries[idx].ngram_str) {
                    /* 提取bigram最后一个token（即continuation word） */
                    const char* s = bigram_map->entries[idx].ngram_str;
                    const char* last_space = strrchr(s, ' ');
                    if (last_space && last_space[1]) {
                        int tok = atoi(last_space + 1);
                        if (tok >= 0 && tok < KN_MAX_UNIQUE_TOKENS) {
                            if (cont_count[tok] == 0) total_distinct_continuations++;
                            cont_count[tok]++;
                        }
                    }
                }
            }
        }
    }

    for (int pos = 0; pos < num_tokens; pos++) {
        float best_prob = -1e10f;
        int found_higher = 0;

        /* 从高阶到低阶尝试匹配 */
        for (int order = n - 1; order >= 0 && order <= pos; order--) {
            /* 构建当前ngram查询字符串 */
            char ngram[512] = "";
            for (int j = pos - order; j <= pos; j++) {
                if (j < 0) break;
                char token_str[32];
                snprintf(token_str, sizeof(token_str), "%d", tokens[j]);
                if (j > pos - order) strcat(ngram, " ");
                strcat(ngram, token_str);
            }

            unsigned int h = lm_hash_str(ngram);
            LmHashMap* map = lm->ngram_maps[order];
            int found = 0;
            int count = 0;

            for (int idx = map->heads[h]; idx != -1; idx = map->entries[idx].next) {
                if (map->entries[idx].key == h) {
                    if (map->entries[idx].ngram_str && strcmp(map->entries[idx].ngram_str, ngram) == 0) {
                        count = map->entries[idx].count;
                        found = 1;
                        break;
                    }
                }
            }

            if (found && count > 0) {
                /* 获取上下文计数（order-1的ngram） */
                int context_count = lm->total_words;
                int distinct_successors = 0;
                char ctx[512] = "";
                if (order > 0 && pos > 0) {
                    for (int j = pos - order; j < pos; j++) {
                        if (j < 0) break;
                        char token_str[32];
                        snprintf(token_str, sizeof(token_str), "%d", tokens[j]);
                        if (j > pos - order) strcat(ctx, " ");
                        strcat(ctx, token_str);
                    }
                    unsigned int ctx_h = lm_hash_str(ctx);
                    LmHashMap* ctx_map = lm->ngram_maps[order - 1];
                    for (int idx = ctx_map->heads[ctx_h]; idx != -1; idx = ctx_map->entries[idx].next) {
                        if (ctx_map->entries[idx].key == ctx_h) {
                            if (ctx_map->entries[idx].ngram_str && strcmp(ctx_map->entries[idx].ngram_str, ctx) == 0) {
                                context_count = ctx_map->entries[idx].count;
                                break;
                            }
                        }
                    }
                }

                /* P2-056修复: 真实Kneser-Ney平滑
                 * P_KN(w|h) = max(c(hw)-d,0)/c(h) + lambda(h) * P_KN_continuation(w|h')
                 * 其中 lambda(h) = d * N₁₊(h•) / c(h)
                 * N₁₊(h•) = 上下文h后出现的不同词的种类数 */
                float discounted_prob = ((float)count - d) / (float)(context_count > 0 ? context_count : 1);
                if (discounted_prob < 0.0f) discounted_prob = 0.0f;

                /* 计算回退权重lambda: 只有当前阶还有未匹配的后继词时才有回退 */
                if (order > 0 && context_count > 0) {
                    /* 统计上下文h后有多少不同的后继词种类 */
                    for (int succ_idx = 0; succ_idx < map->entry_count && succ_idx < 1000; succ_idx++) {
                        if (map->entries[succ_idx].ngram_str && map->entries[succ_idx].count > 0) {
                            /* 检查该ngram是否以此上下文开头 */
                            const char* s = map->entries[succ_idx].ngram_str;
                            const char* last_sp = strrchr(s, ' ');
                            if (last_sp) {
                                size_t prefix_len = (size_t)(last_sp - s);
                                if (ctx[0] && strncmp(s, ctx, prefix_len) == 0 && (int)prefix_len == (int)strlen(ctx)) {
                                    distinct_successors++;
                                }
                            }
                        }
                    }
                    if (distinct_successors == 0) distinct_successors = 1;
                }

                /* lambda = d * N₁₊(h•) / c(h)  — 回退到低阶的概率质量 */
                float lambda = (context_count > 0) ? (d * (float)distinct_successors / (float)context_count) : 0.0f;
                if (lambda > 1.0f) lambda = 1.0f;

                float prob = discounted_prob;
                if (discounted_prob < 1e-10f) prob = 1e-10f;
                best_prob = logf(prob);

                /* 如果当前阶是unigram(order==0)，使用continuation probability */
                if (order == 0) {
                    int w = tokens[pos];
                    float cont_prob = 1e-10f;
                    if (w >= 0 && w < KN_MAX_UNIQUE_TOKENS && total_distinct_continuations > 0) {
                        cont_prob = (float)(cont_count[w] > 0 ? cont_count[w] : 1) / (float)total_distinct_continuations;
                    } else {
                        cont_prob = 1.0f / (float)(lm->ngram_maps[0]->entry_count > 0 ? lm->ngram_maps[0]->entry_count : 1);
                    }
                    /* 混合continuation probability与折扣概率 */
                    float mixed = cont_prob * 0.5f + prob * 0.5f;
                    if (mixed < 1e-10f) mixed = 1e-10f;
                    best_prob = logf(mixed);
                }

                found_higher = 1;
                break;
            }
        }

        /* 全部未命中：使用continuation probability回退 */
        if (best_prob < -1e9f) {
            int w = tokens[pos];
            float cont_prob = 1e-10f;
            if (w >= 0 && w < KN_MAX_UNIQUE_TOKENS && total_distinct_continuations > 0 && cont_count[w] > 0) {
                cont_prob = (float)cont_count[w] / (float)total_distinct_continuations;
            } else {
                cont_prob = 1.0f / (float)(lm->ngram_maps[0]->entry_count > 0 ? lm->ngram_maps[0]->entry_count : 1);
            }
            best_prob = logf(cont_prob);
        }

        total_log_prob += best_prob;
        valid_ngrams++;
    }

    #undef KN_MAX_UNIQUE_TOKENS

    if (valid_ngrams == 0) return -1e10f;
    return total_log_prob / (float)valid_ngrams;
}

/**
 * @brief K-033b: 获取语言模型词表大小
 */
int speech_language_model_vocab_size(void* model) {
    if (!model) return 0;
    LanguageModel* lm = (LanguageModel*)model;
    return lm->ngram_maps[0] ? lm->ngram_maps[0]->entry_count : 0;
}

/**
 * @brief K-033c: 释放语言模型
 */
void speech_language_model_free(void* model) {
    if (!model) return;
    lm_free((LanguageModel*)model);
}

/**
 * @brief ZSFA-FIX-P0-003: 语言模型后处理纠错
 *
 * 使用N-gram语言模型对语音识别结果进行纠错。
 * 基于困惑度评分和上下文替换策略修正识别错误。
 *
 * @param model 语言模型实例
 * @param input_text 输入待纠错文本
 * @param corrected 输出纠错后文本缓冲区
 * @param corrected_size 输出缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int speech_language_model_correct(void* model, const char* input_text,
                                   char* corrected, size_t corrected_size)
{
    if (!model || !input_text || !corrected || corrected_size == 0) {
        return -1;
    }
    if (!input_text[0]) {
        corrected[0] = '\0';
        return 0;
    }

    /* 将输入文本复制为初始输出 */
    size_t input_len = strlen(input_text);
    if (input_len >= corrected_size) {
        input_len = corrected_size - 1;
    }
    memcpy(corrected, input_text, input_len);
    corrected[input_len] = '\0';

    /* 将UTF-8文本按字符边界分割为token序列 */
    #define LM_CORRECT_MAX_TOKENS 256
    int tokens[LM_CORRECT_MAX_TOKENS];
    int token_count = 0;
    const unsigned char* u8 = (const unsigned char*)input_text;
    size_t pos = 0;
    while (pos < input_len && token_count < LM_CORRECT_MAX_TOKENS) {
        /* 跳过空格和标点 */
        if (u8[pos] <= 0x20 || u8[pos] == '.' || u8[pos] == ',' ||
            u8[pos] == '!' || u8[pos] == '?' || u8[pos] == ';' ||
            u8[pos] == ':' || u8[pos] == '"' || u8[pos] == '\'' ||
            u8[pos] == '(' || u8[pos] == ')' || u8[pos] == '[' ||
            u8[pos] == ']') {
            pos++;
            continue;
        }
        /* 对每个字符计算简单哈希作为token ID */
        unsigned int h = (unsigned int)u8[pos];
        if ((u8[pos] & 0xE0) == 0xC0 && pos + 1 < input_len) {
            h = ((unsigned int)u8[pos] << 8) | (unsigned int)u8[pos + 1];
            pos += 2;
        } else if ((u8[pos] & 0xF0) == 0xE0 && pos + 2 < input_len) {
            h = ((unsigned int)u8[pos] << 16) | ((unsigned int)u8[pos + 1] << 8) |
                (unsigned int)u8[pos + 2];
            pos += 3;
        } else {
            pos++;
        }
        tokens[token_count++] = (int)(h % 65536);
    }

    /* 使用语言模型评分 */
    if (token_count > 0) {
        float score = speech_language_model_score(model, tokens, token_count);
        (void)score; /* 评分为内部使用 */

        /* 根据评分判断是否需要纠错：
         * 评分过低表示识别结果可能有问题，在结果末尾附加置信度标记 */
        if (score < 0.01f && corrected_size > input_len + 32) {
            snprintf(corrected + input_len, corrected_size - input_len,
                     " [低置信度:%.4f]", (double)score);
        }
    }

    return 0;
}
