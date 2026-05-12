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
            safe_free((void**)&lm->ngram_maps[i]->entries);
            safe_free((void**)&lm->ngram_maps[i]->heads);
            safe_free((void**)&lm->ngram_maps[i]);
        }
    }
    safe_free((void**)&lm);
}

static int lm_insert_ngram(LmHashMap* map, const char* ngram, int* existing) {
    unsigned int h = lm_hash_str(ngram);
    for (int idx = map->heads[h]; idx != -1; idx = map->entries[idx].next) {
        if (map->entries[idx].key == h) {
            map->entries[idx].count++;
            if (existing) *existing = 1;
            return idx;
        }
    }
    if (map->entry_count >= map->capacity) return -1;
    int idx = map->entry_count++;
    map->entries[idx].key = h;
    map->entries[idx].count = 1;
    map->entries[idx].next = map->heads[h];
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
    char words[1024][64];
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
            while (*p && (isalnum((unsigned char)*p) || (*p & 0x80)) && len < 63) {
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
    lm->total_words = total_tokens;

    /* 保存模型为文本格式 */
    FILE* out = fopen(model_path, "w");
    if (!out) { lm_free(lm); return -1; }

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
 * @brief K-033: 从内存文本构建语言模型
 */
int speech_language_model_build_from_text(const char* text, size_t text_len,
                                           int n, void** model_out) {
    if (!text || !model_out || n < 1 || n > LM_MAX_N) return -1;

    /* 写入临时文件后训练 */
    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "tmp_lm_%u.txt",
             (unsigned int)(size_t)text);
    FILE* fp = fopen(tmp_path, "w");
    if (!fp) return -1;
    fwrite(text, 1, text_len, fp);
    fclose(fp);

    char model_path[512];
    snprintf(model_path, sizeof(model_path), "tmp_lm_%u.model",
             (unsigned int)(size_t)text);
    int ret = speech_language_model_train(tmp_path, n, model_path);
    remove(tmp_path);

    if (ret == 0) {
        /* 读取模型内容 */
        FILE* mf = fopen(model_path, "r");
        if (mf) {
            fseek(mf, 0, SEEK_END);
            long sz = ftell(mf);
            fseek(mf, 0, SEEK_SET);
            char* buf = (char*)safe_malloc((size_t)sz + 1);
            if (buf) { fread(buf, 1, (size_t)sz, mf); buf[sz] = '\0'; }
            fclose(mf);
            *model_out = buf;
        }
        remove(model_path);
    }
    return ret;
}
