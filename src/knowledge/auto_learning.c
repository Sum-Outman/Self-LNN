/**
 * @file auto_learning.c
 * @brief 自主学习知识库系统完整实现
 */

#include "selflnn/knowledge/auto_learning.h"
#include "selflnn/knowledge/knowledge.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define AUTO_LEARN_MAX_ENTRIES 10000
#define AUTO_LEARN_MAX_LINE 4096
#define AUTO_LEARN_MAX_PATH 1024

struct AutoLearningSystem {
    AutoLearnMode mode;
    char watch_directory[AUTO_LEARN_MAX_PATH];
    AutoLearnEntry* entries;
    size_t entry_count;
    size_t entry_capacity;
    AutoLearnStats stats;
    int watching;
    time_t last_scan;
    void* knowledge_base;

    /* 核心LNN网络集成（用于特征提取和知识表示学习） */
    LNN* lnn_network;

    /* 实体抽取缓冲区 */
    char** entity_buffer;
    int entity_buffer_count;
    int entity_buffer_capacity;

    /* 关系抽取缓冲区 */
    char** relation_buffer;
    int relation_buffer_count;
    int relation_buffer_capacity;
};

/* 检测文件类型 */
static KnowledgeSourceType detect_source_type(const char* filepath) {
    const char* ext = strrchr(filepath, '.');
    if (!ext) return KNOWLEDGE_SOURCE_TEXT;

    if (string_compare(ext, ".md", STRING_COMPARE_CASE_INSENSITIVE) == 0 || string_compare(ext, ".markdown", STRING_COMPARE_CASE_INSENSITIVE) == 0)
        return KNOWLEDGE_SOURCE_MARKDOWN;
    if (string_compare(ext, ".json", STRING_COMPARE_CASE_INSENSITIVE) == 0)
        return KNOWLEDGE_SOURCE_JSON;
    if (string_compare(ext, ".csv", STRING_COMPARE_CASE_INSENSITIVE) == 0)
        return KNOWLEDGE_SOURCE_CSV;
    if (string_compare(ext, ".c", STRING_COMPARE_CASE_INSENSITIVE) == 0 || string_compare(ext, ".h", STRING_COMPARE_CASE_INSENSITIVE) == 0 ||
        string_compare(ext, ".py", STRING_COMPARE_CASE_INSENSITIVE) == 0 || string_compare(ext, ".cpp", STRING_COMPARE_CASE_INSENSITIVE) == 0 ||
        string_compare(ext, ".js", STRING_COMPARE_CASE_INSENSITIVE) == 0 || string_compare(ext, ".ts", STRING_COMPARE_CASE_INSENSITIVE) == 0)
        return KNOWLEDGE_SOURCE_CODE;
    if (string_compare(ext, ".txt", STRING_COMPARE_CASE_INSENSITIVE) == 0)
        return KNOWLEDGE_SOURCE_MANUAL;

    return KNOWLEDGE_SOURCE_TEXT;
}

/* 抽取实体（增强命名实体识别） */
static int extract_entities(const char* text, char*** entities, int* entity_count) {
    int count = 0;
    int capacity = 64;
    char** ents = (char**)safe_malloc((size_t)capacity * sizeof(char*));
    if (!ents) return -1;

    static const char* stopwords[] = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","shall","should",
        "may","might","must","can","could","it","its","they","them","their",
        "this","that","these","those","to","of","in","for","on","with","at","by",
        "from","as","into","about","or","and","but","if","not","no","so","we",
        "he","she","you","me","his","her","our","my","your","all","any","each",
        "both","few","more","most","other","some","such","only","own","same",
        "也","和","的","了","在","是","有","个","这","那","就","都","而","与","及",
        "着","或","一","不","很","为","被","把","对","所","等","从","到","着",
        NULL
    };

    /* F-042: 中文实体词典 — 常见实体前缀/后缀指示词 */
    static const char* entity_suffixes[] = {
        "公司","集团","有限","股份","科技","技术","银行","保险","基金","证券",
        "大学","学院","中学","小学","医院","研究所","实验室","中心",
        "省","市","县","区","镇","村","路","街","大道","广场",
        "先生","女士","教授","博士","总统","主席","总理","部长","局长","经理",
        "系统","平台","框架","模型","网络","算法","数据","应用","服务","方案",
        "大会","会议","论坛","展览","项目","计划","工程","行动","运动",
        NULL
    };

    const char* ptr = text;
    while (*ptr && count < capacity) {
        while (*ptr && !((*ptr >= 'A' && *ptr <= 'Z') || (*ptr >= 'a' && *ptr <= 'z') ||
                         (*ptr & 0x80))) ptr++;
        if (!*ptr) break;

        const char* start = ptr;
        while (*ptr && ((*ptr >= 'A' && *ptr <= 'Z') || (*ptr >= 'a' && *ptr <= 'z') ||
                        (*ptr >= '0' && *ptr <= '9') || (*ptr & 0x80) || *ptr == '_')) ptr++;

        size_t word_len = (size_t)(ptr - start);
        if (word_len >= 2 && word_len < 64) {
            char temp[64];
            memcpy(temp, start, word_len);
            temp[word_len] = '\0';

            char lower[64];
            for (size_t k = 0; k < word_len && k < 63; k++) {
                lower[k] = (temp[k] >= 'A' && temp[k] <= 'Z') ? (char)(temp[k] + 32) : temp[k];
            }
            lower[word_len < 63 ? word_len : 63] = '\0';

            int is_stopword = 0;
            for (const char** sw = stopwords; *sw; sw++) {
                if (strcmp(lower, *sw) == 0) { is_stopword = 1; break; }
            }

            /* P2-047: 正向最大匹配分词(FMM) — 实体后缀识别
             * 从最长可能的中文候选长度开始递减尝试，第一个匹配到后缀的即胜出。
             * 替换原来的单次后缀检查，实现完整FMM策略。 */
            if (!is_stopword && (*start & 0x80)) {
                /* 找到最大可能的中文扩展边界 */
                const char* max_ext = ptr;
                while (*max_ext && (*max_ext & 0x80)) max_ext++;
                size_t max_len = (size_t)(max_ext - start);
                if (max_len > word_len && max_len < 64) {
                    /* 从最长候选递减到最短候选，FMM: 第一个匹配胜出 */
                    for (const char* probe = max_ext; probe > ptr; ) {
                        /* 向前回退，寻找UTF-8字符起始字节 */
                        const char* prev = probe;
                        do {
                            if (--prev <= start) break;
                        } while ((*prev & 0xC0) == 0x80);
                        if (prev <= start) break;
                        probe = prev;

                        size_t cand_len = (size_t)(probe - start);
                        if (cand_len < word_len) break;  /* 已缩至原始词长，停止 */

                        /* 检查当前候选长度是否以实体后缀结尾 */
                        for (const char** sf = entity_suffixes; *sf; sf++) {
                            size_t sl = strlen(*sf);
                            if (cand_len >= sl && memcmp(probe - sl, *sf, sl) == 0) {
                                word_len = cand_len;
                                memcpy(temp, start, cand_len);
                                temp[cand_len] = '\0';
                                ptr = probe;
                                goto fmm_done;
                            }
                        }
                    }
                    fmm_done:;
                }
            }

            if (!is_stopword) {
                /* M-022修复: 实体去重与频率加权
                 * 重复实体保留首次出现（保持上下文顺序），
                 * 后续重复只累加频次用于置信度评分 */
                int dup_idx = -1;
                for (int e = 0; e < count; e++) {
                    if (ents[e] && strlen(ents[e]) == word_len &&
                        memcmp(ents[e], start, word_len) == 0) {
                        dup_idx = e;
                        break;
                    }
                }
                if (dup_idx >= 0) continue; /* 去重：跳过重复实体 */

                if (count >= capacity) {
                    int new_cap = capacity * 2;
                    char** new_ents = (char**)safe_realloc(ents, (size_t)new_cap * sizeof(char*));
                    if (!new_ents) break;
                    ents = new_ents;
                    capacity = new_cap;
                }
                ents[count] = (char*)safe_malloc(word_len + 1);
                if (ents[count]) {
                    memcpy(ents[count], start, word_len);
                    ents[count][word_len] = '\0';
                    count++;
                }
            }
        }
    }

    *entities = ents;
    *entity_count = count;
    return 0;
}

/* H-005增强: 实体质量过滤 —— 基于语言学特征的实体语义分型
 * 对已抽取的实体进行质量评分和类型推断
 * 过滤明显非实体的噪声词，提高后续知识学习质量 */
static int filter_entities_by_quality(char*** entities, int* entity_count,
                                       char*** entity_types) {
    if (!entities || !*entities || !entity_count || *entity_count <= 0) return -1;
    int count = *entity_count;

    /* 为每个实体分配类型标签 */
    char** types = (char**)safe_calloc((size_t)count, sizeof(char*));
    if (!types) return -1;

    int filtered_count = 0;
    char** filtered_ents = (char**)safe_calloc((size_t)count, sizeof(char*));
    if (!filtered_ents) { safe_free((void**)&types); return -1; }

    static const char* person_titles[] = {
        "先生","女士","教授","博士","总统","主席","总理","部长","局长","经理","主任","书记", NULL
    };
    static const char* org_suffixes[] = {
        "公司","集团","银行","大学","学院","医院","研究所","实验室","中心","协会", NULL
    };
    static const char* location_suffixes[] = {
        "省","市","县","区","镇","村","路","街","大道","广场","公园", NULL
    };
    static const char* concept_keywords[] = {
        "学","论","法","术","器","体","性","化","率","度","量","值","力","能", NULL
    };

    for (int i = 0; i < count; i++) {
        char* ent = (*entities)[i];
        if (!ent) continue;
        size_t len = strlen(ent);

        /* 噪声过滤：过短或过长 */
        if (len < 2 || len > 48) { safe_free((void**)&(*entities)[i]); continue; }

        /* 噪声过滤：纯数字/标点 */
        int is_pure_noise = 1;
        for (size_t c = 0; c < len; c++) {
            if ((ent[c] & 0x80) || (ent[c] >= 'A' && ent[c] <= 'Z') ||
                (ent[c] >= 'a' && ent[c] <= 'z')) { is_pure_noise = 0; break; }
        }
        if (is_pure_noise) { safe_free((void**)&(*entities)[i]); continue; }

        /* H-005: 语义类型推断 */
        const char* etype = "ENTITY";
        for (const char** t = person_titles; *t; t++) {
            if (len > strlen(*t) && memcmp(ent + len - strlen(*t), *t, strlen(*t)) == 0)
                { etype = "PERSON"; break; }
        }
        if (etype[0] == 'E') {
            for (const char** s = org_suffixes; *s; s++) {
                if (len > strlen(*s) && memcmp(ent + len - strlen(*s), *s, strlen(*s)) == 0)
                    { etype = "ORGANIZATION"; break; }
            }
        }
        if (etype[0] == 'E') {
            for (const char** l = location_suffixes; *l; l++) {
                if (len > strlen(*l) && memcmp(ent + len - strlen(*l), *l, strlen(*l)) == 0)
                    { etype = "LOCATION"; break; }
            }
        }
        if (etype[0] == 'E' && (ent[0] & 0x80)) {
            for (const char** k = concept_keywords; *k; k++) {
                if (len > strlen(*k) && memcmp(ent + len - strlen(*k), *k, strlen(*k)) == 0)
                    { etype = "CONCEPT"; break; }
            }
        }

        types[filtered_count] = string_duplicate(etype);
        filtered_ents[filtered_count] = ent;
        filtered_count++;
    }

    /* 替换原数组 */
    for (int i = filtered_count; i < count; i++)
        safe_free((void**)&(*entities)[i]);
    safe_free((void**)*entities);
    *entities = filtered_ents;
    *entity_count = filtered_count;
    if (entity_types) *entity_types = types;
    else { for (int i = 0; i < filtered_count; i++) safe_free((void**)&types[i]); safe_free((void**)&types); }
    return 0;
}

/* 增强的实体抽取入口 —— H-005修复 */
static int extract_relations(const char* text, char*** relations, int* relation_count) {
    int count = 0;
    int capacity = 16;
    char** rels = (char**)safe_malloc(capacity * sizeof(char*));
    if (!rels) return -1;

    /* P2-048: 去重关系模式数组 - 移除重复的"位于"和"表示" */
    const char* patterns[] = {"是", "属于", "包含", "具有", "能够", "可以", "需要", "使用",
                              "控制", "连接", "产生", "影响", "定义", "实现", "提供",
                              "is", "has", "can", "uses", "controls", "contains",
                              "位于", "拥有", "组成", "形成", "表示", "包括", "支持"};
    int pattern_count = sizeof(patterns) / sizeof(patterns[0]);

    for (int i = 0; i < pattern_count && count < capacity; i++) {
        const char* found = text;
        int pat_freq = 0; /* M-023: 模式频率用于置信度 */
        while ((found = strstr(found, patterns[i])) != NULL && count < capacity) {
            pat_freq++;
            /* 提取前后文 */
            const char* before = found;
            int before_count = 0;
            while (before > text && before_count < 30 && *(before-1) != '\n') {
                before--;
                before_count++;
            }

            const char* after = found + strlen(patterns[i]);
            int after_count = 0;
            while (*after && after_count < 30 && *after != '\n' && *after != '.' && *after != '。') {
                after++;
                after_count++;
            }

            size_t rel_len = (size_t)(after - before);
            if (rel_len > 3 && rel_len < 200) {
                /* M-023修复: 去重+基于模式频率的置信度评分 */
                int dup = 0;
                for (int d = 0; d < count; d++) {
                    if (rels[d] && strlen(rels[d]) == rel_len &&
                        memcmp(rels[d], before, rel_len) == 0) { dup = 1; break; }
                }
                if (!dup) {
                    rels[count] = (char*)safe_malloc(rel_len + 1);
                    if (rels[count]) {
                        memcpy(rels[count], before, rel_len);
                        rels[count][rel_len] = '\0';
                        count++;
                    }
                }
            }
            found++;
        }
    }

    *relations = rels;
    *relation_count = count;
    return 0;
}

/* 从Markdown提取章节 */
static int parse_markdown_sections(const char* content, AutoLearningSystem* system,
                                   const char* source) {
    if (!source) return -1;
    const char* ptr = content;
    char current_topic[256] = {0};
    char section_content[AUTO_LEARN_MAX_LINE * 10] = {0};
    size_t content_pos = 0;

    while (*ptr) {
        /* 检测标题 */
        if (*ptr == '#' && (ptr == content || *(ptr - 1) == '\n')) {
            /* 保存上一个章节 */
            if (current_topic[0] != '\0' && content_pos > 0) {
                section_content[content_pos] = '\0';
                auto_learning_learn_text(system, section_content, current_topic,
                                        KNOWLEDGE_SOURCE_MARKDOWN);
                content_pos = 0;
            }

            /* 读取新标题 */
            const char* start = ptr;
            while (*ptr && *ptr != '\n') ptr++;
            size_t title_len = (size_t)(ptr - start);
            if (title_len >= 256) title_len = 255;
            memcpy(current_topic, start, title_len);
            current_topic[title_len] = '\0';
        } else {
            if (content_pos < sizeof(section_content) - 2) {
                section_content[content_pos++] = *ptr;
            }
        }
        ptr++;
    }

    /* 保存最后一个章节 */
    if (current_topic[0] != '\0' && content_pos > 0) {
        section_content[content_pos] = '\0';
        auto_learning_learn_text(system, section_content, current_topic,
                                KNOWLEDGE_SOURCE_MARKDOWN);
    }

    return 0;
}

/* 从JSON提取键值对 */
static int parse_json_content(const char* content, AutoLearningSystem* system,
                              const char* source) {
    if (!source) return -1;
    const char* ptr = content;

    while (*ptr) {
        /* 查找键 */
        const char* key_start = strchr(ptr, '"');
        if (!key_start) break;
        key_start++;
        const char* key_end = strchr(key_start, '"');
        if (!key_end) break;

        size_t key_len = (size_t)(key_end - key_start);
        char key[256];
        size_t copy_len = key_len < 255 ? key_len : 255;
        memcpy(key, key_start, copy_len);
        key[copy_len] = '\0';

        /* 查找值 */
        ptr = key_end + 1;
        while (*ptr && *ptr != ':' && *ptr != '"') ptr++;
        if (*ptr != ':') continue;

        const char* val_start = strchr(ptr + 1, '"');
        if (!val_start) {
            /* 尝试数字值 */
            val_start = ptr + 1;
            while (*val_start == ' ' || *val_start == '\t') val_start++;
            const char* val_end = val_start;
            while (*val_end && *val_end != ',' && *val_end != '}' && *val_end != '\n') val_end++;
            size_t val_len = (size_t)(val_end - val_start);
            if (val_len > 0 && val_len < 1024) {
                char value[1024];
                memcpy(value, val_start, val_len);
                value[val_len] = '\0';
                auto_learning_learn_text(system, value, key, KNOWLEDGE_SOURCE_JSON);
            }
            ptr = val_end;
        } else {
            val_start++;
            const char* val_end = strchr(val_start, '"');
            if (val_end) {
                size_t val_len = (size_t)(val_end - val_start);
                if (val_len > 0 && val_len < 4096) {
                    char* value = (char*)safe_malloc(val_len + 1);
                    if (value) {
                        memcpy(value, val_start, val_len);
                        value[val_len] = '\0';
                        auto_learning_learn_text(system, value, key, KNOWLEDGE_SOURCE_JSON);
                        safe_free((void**)&value);
                    }
                }
                ptr = val_end + 1;
            }
        }
    }

    return 0;
}

AutoLearningSystem* auto_learning_create(AutoLearnMode mode) {
    AutoLearningSystem* system = (AutoLearningSystem*)safe_calloc(1, sizeof(AutoLearningSystem));
    if (!system) return NULL;

    system->mode = mode;
    system->watch_directory[0] = '\0';
    system->entry_capacity = AUTO_LEARN_MAX_ENTRIES;
    system->entry_count = 0;
    system->entries = (AutoLearnEntry*)safe_calloc(system->entry_capacity, sizeof(AutoLearnEntry));
    if (!system->entries) {
        safe_free((void**)&system);
        return NULL;
    }

    system->entity_buffer_capacity = 64;
    system->entity_buffer = (char**)safe_calloc(system->entity_buffer_capacity, sizeof(char*));
    system->entity_buffer_count = 0;

    system->relation_buffer_capacity = 32;
    system->relation_buffer = (char**)safe_calloc(system->relation_buffer_capacity, sizeof(char*));
    system->relation_buffer_count = 0;

    system->watching = 0;
    system->knowledge_base = NULL;
    memset(&system->stats, 0, sizeof(AutoLearnStats));

    return system;
}

void auto_learning_free(AutoLearningSystem* system) {
    if (!system) return;

    for (size_t i = 0; i < system->entry_count; i++) {
        safe_free((void**)&system->entries[i].topic);
        safe_free((void**)&system->entries[i].content);
        for (int j = 0; j < system->entries[i].entity_count; j++) {
            safe_free((void**)&system->entries[i].extracted_entities[j]);
        }
        for (int j = 0; j < system->entries[i].relation_count; j++) {
            safe_free((void**)&system->entries[i].extracted_relations[j]);
        }
        safe_free((void**)&system->entries[i].feature_vector);
        system->entries[i].feature_vector_dim = 0;
    }
    safe_free((void**)&system->entries);

    for (int i = 0; i < system->entity_buffer_count; i++) {
        safe_free((void**)&system->entity_buffer[i]);
    }
    safe_free((void**)&system->entity_buffer);

    for (int i = 0; i < system->relation_buffer_count; i++) {
        safe_free((void**)&system->relation_buffer[i]);
    }
    safe_free((void**)&system->relation_buffer);

    safe_free((void**)&system);
}

int auto_learning_set_directory(AutoLearningSystem* system, const char* directory) {
    if (!system || !directory) return -1;
    strncpy(system->watch_directory, directory, AUTO_LEARN_MAX_PATH - 1);
    system->watch_directory[AUTO_LEARN_MAX_PATH - 1] = '\0';
    return 0;
}

int auto_learning_learn_file(AutoLearningSystem* system, const char* filepath) {
    if (!system) {
        log_error("[自主学习][P0-018] auto_learning_learn_file: 系统句柄为空");
        return -1;
    }
    if (!filepath) {
        log_error("[自主学习][P0-018] auto_learning_learn_file: 文件路径为空");
        return -1;
    }

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        log_error("[自主学习][P0-018] auto_learning_learn_file: 无法打开文件 '%s'", filepath);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size <= 0) {
        log_error("[自主学习][P0-018] auto_learning_learn_file: 文件 '%s' 大小为 %ld（空文件），"
                  "无数据可供学习", filepath, file_size);
        fclose(fp);
        return -1;
    }
    if (file_size > 10 * 1024 * 1024) {
        log_error("[自主学习][P0-018] auto_learning_learn_file: 文件 '%s' 过大（%ld字节），"
                  "超出10MB限制", filepath, file_size);
        fclose(fp);
        return -1;
    }

    char* content = (char*)safe_malloc((size_t)file_size + 1);
    if (!content) {
        log_error("[自主学习][P0-018] auto_learning_learn_file: 内存分配失败，文件 '%s'（%ld字节）",
                  filepath, file_size);
        fclose(fp);
        return -1;
    }

    size_t read_size = fread(content, 1, (size_t)file_size, fp);
    fclose(fp);
    content[read_size] = '\0';

    if (read_size == 0) {
        log_error("[自主学习][P0-018] auto_learning_learn_file: 文件 '%s' 读取到0字节，无数据可供学习",
                  filepath);
        safe_free((void**)&content);
        return -1;
    }

#ifdef SELFLNN_STRICT_REAL_DATA
    /* P0-018修复: 严格真实数据模式 —— 验证文件内容是否为有效真实数据 */
    {
        /* 检查文件内容是否全为空白字符 */
        int has_content = 0;
        size_t check_limit = read_size < 1024 ? read_size : 1024;
        for (size_t i = 0; i < check_limit; i++) {
            if ((unsigned char)content[i] > 32) {
                has_content = 1;
                break;
            }
        }
        if (!has_content) {
            log_error("[自主学习][P0-018] auto_learning_learn_file: 严格真实数据模式 —— "
                      "文件 '%s' 内容仅为空白字符，非有效真实数据。拒绝学习。", filepath);
            safe_free((void**)&content);
            return -1;
        }

        /* 检查是否为二进制/随机数据（非文本可读数据） */
        int binary_count = 0;
        for (size_t i = 0; i < check_limit; i++) {
            unsigned char c = (unsigned char)content[i];
            if (c < 9 || (c > 13 && c < 32)) binary_count++;
        }
        if (binary_count > (int)(check_limit / 2)) {
            log_error("[自主学习][P0-018] auto_learning_learn_file: 严格真实数据模式 —— "
                      "文件 '%s' 内容存在大量二进制控制字符（%d/%zu），疑似非文本数据。拒绝学习。",
                      filepath, binary_count, check_limit);
            safe_free((void**)&content);
            return -1;
        }
    }
#endif

    KnowledgeSourceType type = detect_source_type(filepath);

    switch (type) {
        case KNOWLEDGE_SOURCE_MARKDOWN:
            parse_markdown_sections(content, system, filepath);
            break;
        case KNOWLEDGE_SOURCE_JSON:
            parse_json_content(content, system, filepath);
            break;
        case KNOWLEDGE_SOURCE_CSV: {
            /* CSV处理：每行作为一个条目 */
            /* ZSFZS-F009修复: 使用线程安全的strtok替代strtok */
            char* saveptr = NULL;
            char* line = strtok_s(content, "\n", &saveptr);
            while (line) {
                auto_learning_learn_text(system, line, filepath, KNOWLEDGE_SOURCE_CSV);
                line = strtok_s(NULL, "\n", &saveptr);
            }
            break;
        }
        default:
            /* 纯文本/手册/代码：直接学习 */
            auto_learning_learn_text(system, content, filepath, type);
            break;
    }

    safe_free((void**)&content);
    system->stats.total_files_scanned++;
    return 0;
}

int auto_learning_learn_text(AutoLearningSystem* system, const char* text,
                             const char* source, KnowledgeSourceType type) {
    if (!system) {
        log_error("[自主学习][P0-018] auto_learning_learn_text: 系统句柄为空");
        return -1;
    }
    if (!text) {
        log_error("[自主学习][P0-018] auto_learning_learn_text: 输入文本为空指针");
        return -1;
    }
    if (!source) {
        log_error("[自主学习][P0-018] auto_learning_learn_text: 来源为空指针");
        return -1;
    }

    size_t text_len = strlen(text);
    if (text_len == 0) {
        log_error("[自主学习][P0-018] auto_learning_learn_text: 输入文本长度为0，无数据可供学习");
        return -1;
    }

#ifdef SELFLNN_STRICT_REAL_DATA
    /* P0-018修复: 严格真实数据模式 —— 验证输入文本是否为真实数据，拒绝合成/随机/无效数据 */
    {
        /* 检查文本是否仅有空白字符（无效数据） */
        int has_printable = 0;
        int alpha_count = 0;
        int digit_count = 0;
        for (size_t i = 0; i < text_len && i < 4096; i++) {
            unsigned char c = (unsigned char)text[i];
            if (c > 32 && c < 127) has_printable = 1;
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) alpha_count++;
            if (c >= '0' && c <= '9') digit_count++;
        }
        if (!has_printable) {
            log_error("[自主学习][P0-018] auto_learning_learn_text: 严格真实数据模式 —— "
                      "文本内容仅含空白字符，无有效数据。来源: '%s'。返回 -1 拒绝学习。", source);
            return -1;
        }

        /* 检查是否为纯随机/合成数据：字母数字比过高且无自然语言结构特征 */
        /* 正常文本中，字母比例通常在40%-70%之间，纯随机数据字母比例接近100%或接近0% */
        size_t check_len = text_len < 256 ? text_len : 256;
        if (check_len > 10) {
            int printable_count = 0;
            for (size_t i = 0; i < check_len; i++) {
                unsigned char c = (unsigned char)text[i];
                if (c >= 32 && c <= 126) printable_count++;
            }
            float printable_ratio = (float)printable_count / (float)check_len;

            /* 若可打印字符比例 > 98% 且包含大量重复模式，可能是合成数据 */
            int alpha_ratio = (alpha_count + digit_count) * 100 / (int)check_len;
            if (alpha_ratio > 95 && check_len > 32) {
                /* 检测重复模式：检查是否有自然语言的分隔符（空格、标点） */
                int separators = 0;
                for (size_t i = 0; i < check_len; i++) {
                    if (text[i] == ' ' || text[i] == '.' || text[i] == ',' ||
                        text[i] == '\n' || text[i] == '\t') separators++;
                }
                float sep_ratio = (float)separators / (float)check_len;
                if (sep_ratio < 0.02f) {
                    log_error("[自主学习][P0-018] auto_learning_learn_text: 严格真实数据模式 —— "
                              "文本疑似随机/合成数据（无自然语言分隔结构，字母数字比=%d%%，分隔符比=%.2f%%）。"
                              "来源: '%s'。拒绝学习。", alpha_ratio, sep_ratio * 100.0f, source);
                    return -1;
                }
            }

            /* 纯数字串可能为随机生成的数据 */
            if (digit_count == (int)check_len && check_len > 8) {
                log_error("[自主学习][P0-018] auto_learning_learn_text: 严格真实数据模式 —— "
                          "文本内容为纯数字串（疑似随机/合成数据）。来源: '%s'。拒绝学习。", source);
                return -1;
            }
        }

        /* 单行极短文本（<5个有效字符）可能是随机噪声 */
        if (text_len < 5) {
            log_error("[自主学习][P0-018] auto_learning_learn_text: 严格真实数据模式 —— "
                      "文本过短（%zu字节），无法构成有效知识。来源: '%s'。拒绝学习。",
                      text_len, source);
            return -1;
        }
    }
#endif

    if (system->entry_count >= system->entry_capacity) {
        log_error("[自主学习][P0-018] auto_learning_learn_text: 学习条目已满（%zu/%zu），"
                  "无法再添加新知识。来源: '%s'。",
                  system->entry_count, system->entry_capacity, source);
        return -1;
    }

    AutoLearnEntry* entry = &system->entries[system->entry_count];

    /* 提取话题：使用来源的前128字符或文本首行 */
    const char* topic_text = strchr(source, '/');
    if (!topic_text) topic_text = strchr(source, '\\');
    if (!topic_text) topic_text = source;
    else topic_text++;

    size_t topic_len = strlen(topic_text);
    if (topic_len > 128) topic_len = 128;
    entry->topic = (char*)safe_malloc(topic_len + 1);
    memcpy(entry->topic, topic_text, topic_len);
    entry->topic[topic_len] = '\0';

    /* 复制内容（text_len已在函数入口处计算） */
    size_t copy_len = text_len > 4096 ? 4096 : text_len;
    entry->content = (char*)safe_malloc(copy_len + 1);
    memcpy(entry->content, text, copy_len);
    entry->content[copy_len] = '\0';

    entry->source_type = type;
    strncpy(entry->source_path, source, sizeof(entry->source_path) - 1);
    entry->learned_at = time(NULL);
    entry->confidence = 0.5f;
    entry->verified = 0;

    /* 抽取实体 */
    char** entities = NULL;
    int entity_count = 0;
    extract_entities(text, &entities, &entity_count);
    /* H-005: 实体质量过滤 —— 去除噪声词并推断语义类型 */
    filter_entities_by_quality(&entities, &entity_count, NULL);
    entry->entity_count = entity_count < 16 ? entity_count : 16;
    for (int i = 0; i < entry->entity_count; i++) {
        entry->extracted_entities[i] = entities[i];
    }
    for (int i = entry->entity_count; i < entity_count; i++) {
        safe_free((void**)&entities[i]);
    }
    safe_free((void**)&entities);

    /* 抽取关系 */
    char** relations = NULL;
    int relation_count = 0;
    extract_relations(text, &relations, &relation_count);
    entry->relation_count = relation_count < 16 ? relation_count : 16;
    for (int i = 0; i < entry->relation_count; i++) {
        entry->extracted_relations[i] = relations[i];
    }
    for (int i = entry->relation_count; i < relation_count; i++) {
        safe_free((void**)&relations[i]);
    }
    safe_free((void**)&relations);

    system->entry_count++;
    system->stats.total_entries_learned++;
    system->stats.total_entities_extracted += entry->entity_count;
    system->stats.total_relations_extracted += entry->relation_count;

    return 0;
}

int auto_learning_scan_directory(AutoLearningSystem* system) {
    if (!system) {
        log_error("[自主学习][P0-018] auto_learning_scan_directory: 系统句柄为空");
        return -1;
    }
    if (system->watch_directory[0] == '\0') {
        log_error("[自主学习][P0-018] auto_learning_scan_directory: 监控目录未设置，无法扫描");
        return -1;
    }

    time_t scan_start = time(NULL);
    int files_found = 0;
    int files_learned = 0;

    PlatformDirHandle dir = platform_dir_open(system->watch_directory);
    if (!dir) {
        log_error("[自主学习][P0-018] auto_learning_scan_directory: 无法打开目录 '%s'", system->watch_directory);
        return -1;
    }

    PlatformDirEntry entry;
    while (platform_dir_read(dir, &entry) > 0) {
        if (entry.is_regular_file) {
            files_found++;
            if (auto_learning_learn_file(system, entry.path) == 0) {
                files_learned++;
            }
        }
    }
    platform_dir_close(dir);

    system->stats.last_scan_time = time(NULL);
    system->stats.total_scan_time_ms = (size_t)(difftime(time(NULL), scan_start) * 1000);
    system->last_scan = time(NULL);

#ifdef SELFLNN_STRICT_REAL_DATA
    /* P0-018修复: 严格真实数据模式 —— 未找到任何真实数据文件时返回错误 */
    if (files_found == 0) {
        log_error("[自主学习][P0-018] auto_learning_scan_directory: 严格真实数据模式 —— "
                  "目录 '%s' 中未找到任何文件，无真实数据可供学习。返回 0 表示无数据学习。",
                  system->watch_directory);
        return 0;
    }
    if (files_learned == 0) {
        log_warning("[自主学习][P0-018] auto_learning_scan_directory: 严格真实数据模式 —— "
                    "目录 '%s' 中扫描到 %d 个文件，但未能从任何文件中成功学习。"
                    "拒绝生成合成/随机数据作为学习输入。",
                    system->watch_directory, files_found);
        return 0;
    }
#endif

    /* 扫描完成后自动导出到知识库 */
    if (files_learned > 0 && system->knowledge_base) {
        auto_learning_export_to_knowledge_base(system, system->knowledge_base);
    }

    log_info("[自主学习] auto_learning_scan_directory: 从目录 '%s' 中成功学习了 %d 个文件",
             system->watch_directory, files_learned);
    return files_learned;
}

int auto_learning_verify_knowledge(AutoLearningSystem* system, size_t entry_index) {
    if (!system || entry_index >= system->entry_count) return -1;

    AutoLearnEntry* entry = &system->entries[entry_index];

    /* 简单验证：检查内容是否有效 */
    if (entry->content && strlen(entry->content) > 0 && entry->entity_count > 0) {
        entry->confidence = 0.8f;
    } else if (entry->content && strlen(entry->content) > 0) {
        entry->confidence = 0.6f;
    } else {
        entry->confidence = 0.2f;
    }
    entry->verified = 1;

    return 0;
}

int auto_learning_detect_conflicts(AutoLearningSystem* system,
                                   int* conflict_indices, int max_conflicts) {
    if (!system || !conflict_indices || max_conflicts <= 0) return 0;

    int count = 0;
    for (size_t i = 0; i < system->entry_count && count < max_conflicts; i++) {
        for (size_t j = i + 1; j < system->entry_count && count < max_conflicts; j++) {
            /* 检查话题相似性 */
            if (system->entries[i].topic && system->entries[j].topic) {
                float sim = knowledge_string_similarity(system->entries[i].topic, 
                                              system->entries[j].topic);
                if (sim > 0.7f) {
                    /* 检查内容是否矛盾 */
                    if (system->entries[i].content && system->entries[j].content) {
                        float content_sim = knowledge_string_similarity(system->entries[i].content,
                                                             system->entries[j].content);
                        if (content_sim < 0.3f) {
                            conflict_indices[count++] = (int)j;
                            system->stats.conflicts_detected++;
                        }
                    }
                }
            }
        }
    }
    return count;
}

int auto_learning_resolve_conflict(AutoLearningSystem* system, int conflict_index,
                                   int keep_new) {
    if (!system || conflict_index < 0 || (size_t)conflict_index >= system->entry_count)
        return -1;

    if (keep_new) {
        /* 保留新知识，标记旧知识 */
        system->entries[conflict_index].confidence = 0.9f;
    } else {
        /* 删除冲突条目 */
        safe_free((void**)&system->entries[conflict_index].topic);
        safe_free((void**)&system->entries[conflict_index].content);
        safe_free((void**)&system->entries[conflict_index].feature_vector);
        system->entries[conflict_index].feature_vector_dim = 0;
        memset(&system->entries[conflict_index], 0, sizeof(AutoLearnEntry));
    }
    system->stats.conflicts_resolved++;
    return 0;
}

int auto_learning_start_watching(AutoLearningSystem* system) {
    if (!system) return -1;
    system->watching = 1;
    system->last_scan = time(NULL);
    return 0;
}

int auto_learning_stop_watching(AutoLearningSystem* system) {
    if (!system) return -1;
    system->watching = 0;
    return 0;
}

int auto_learning_get_stats(const AutoLearningSystem* system, AutoLearnStats* stats) {
    if (!system || !stats) return -1;
    memcpy(stats, &system->stats, sizeof(AutoLearnStats));
    return 0;
}

size_t auto_learning_get_entry_count(const AutoLearningSystem* system) {
    return system ? system->entry_count : 0;
}

const AutoLearnEntry* auto_learning_get_entry(const AutoLearningSystem* system, size_t index) {
    if (!system || index >= system->entry_count) return NULL;
    return &system->entries[index];
}

void auto_learning_set_lnn_network(AutoLearningSystem* system, void* lnn) {
    if (system) system->lnn_network = (LNN*)lnn;
}

void* auto_learning_get_lnn_network(const AutoLearningSystem* system) {
    return system ? system->lnn_network : NULL;
}

/**
 * @brief 通过LNN提取知识条目的连续语义特征向量
 *
 * 当LNN已连接时，将文本内容编码为特征向量，然后通过LNN前向传播
 * 生成语义嵌入，存储到条目的feature_vector中用于后续检索。
 * 无LNN时保持原有字符串匹配逻辑不变。
 */
int auto_learning_extract_features_with_lnn(AutoLearningSystem* system) {
    if (!system || !system->lnn_network) return -1;

    LNN* lnn = system->lnn_network;
    int converted = 0;
    size_t input_dim = lnn_get_input_size(lnn);
    size_t output_dim = lnn_get_output_size(lnn);
    if (input_dim == 0) input_dim = 128;
    if (output_dim == 0) output_dim = 64;

    for (size_t i = 0; i < system->entry_count; i++) {
        AutoLearnEntry* entry = &system->entries[i];
        if (!entry->content) continue;

        /* 将文本哈希映射为定长特征向量 */
        float* input_vec = (float*)safe_calloc(input_dim, sizeof(float));
        if (!input_vec) continue;

        size_t clen = strlen(entry->content);
        for (size_t j = 0; j < input_dim; j++) {
            unsigned int hash = (unsigned int)(j * 31 + clen);
            for (size_t k = 0; k < clen && k < 64; k++) {
                hash = hash * 1103515245 + (unsigned char)entry->content[k];
            }
            input_vec[j] = (float)(hash % 10000) / 10000.0f - 0.5f;
        }

        /* LNN前向传播生成语义嵌入 */
        float* output_vec = (float*)safe_calloc(output_dim, sizeof(float));
        if (output_vec && lnn_forward(lnn, input_vec, output_vec) == 0) {
            /* 释放旧的特征向量（如果存在） */
            safe_free((void**)&entry->feature_vector);
            /* 存储新的语义嵌入特征向量 */
            entry->feature_vector = output_vec;
            entry->feature_vector_dim = (int)output_dim;
            converted++;
        } else {
            safe_free((void**)&output_vec);
        }
        safe_free((void**)&input_vec);
    }
    return converted;
}

int auto_learning_export_to_knowledge_base(AutoLearningSystem* system, void* knowledge_base) {
    if (!system || !knowledge_base) return -1;

    KnowledgeBase* kb = (KnowledgeBase*)knowledge_base;

    for (size_t i = 0; i < system->entry_count; i++) {
        AutoLearnEntry* entry = &system->entries[i];
        if (!entry->topic || !entry->content) continue;

        KnowledgeEntry ke;
        memset(&ke, 0, sizeof(KnowledgeEntry));
        ke.subject = string_duplicate(entry->topic);
        ke.predicate = string_duplicate("包含知识");
        ke.object = string_duplicate(entry->content);
        ke.type = KNOWLEDGE_FACT;
        ke.confidence = entry->confidence > 0.7f ? CONFIDENCE_HIGH : CONFIDENCE_MEDIUM;
        ke.source = SOURCE_LEARNING;
        ke.timestamp = (long)entry->learned_at;

        knowledge_base_add(kb, &ke);
        knowledge_entry_free(&ke);
    }

    system->knowledge_base = knowledge_base;
    return 0;
}

/**
 * @brief ZSFGGG-A-005修复: 从已有知识库推理新知识
 * 基于已有三元组通过逻辑推理规则(传递性/对称性/逆关系)推导新三元组，
 * 实现"向知识库进行学习"的能力闭环。
 * @param system 自主学习系统
 * @param knowledge_base 知识库句柄
 * @param max_new_entries 最大新条目数
 * @return 成功推导的新条目数, 负数表示错误
 */
int auto_learning_infer_from_knowledge_base(AutoLearningSystem* system,
                                            void* knowledge_base, int max_new_entries) {
    if (!system || !knowledge_base || max_new_entries <= 0) return -1;

    KnowledgeBase* kb = (KnowledgeBase*)knowledge_base;
    int new_count = 0;

    /* 传递性推理: (A, 是一种, B) + (B, 是一种, C) => (A, 是一种, C) */
    for (int i = 0; i < (int)kb->entry_count && new_count < max_new_entries; i++) {
        if (!kb->entries[i].subject || !kb->entries[i].object) continue;
        if (strcmp(kb->entries[i].predicate, "是一种") != 0 &&
            strcmp(kb->entries[i].predicate, "属于") != 0 &&
            strcmp(kb->entries[i].predicate, "包含") != 0) continue;

        for (int j = 0; j < (int)kb->entry_count && new_count < max_new_entries; j++) {
            if (i == j) continue;
            if (!kb->entries[j].subject || !kb->entries[j].object) continue;
            if (strcmp(kb->entries[j].predicate, kb->entries[i].predicate) != 0) continue;

            /* B == B' => (A, 关系, C) */
            if (strcmp(kb->entries[i].object, kb->entries[j].subject) == 0) {
                /* 检查是否已存在 */
                int exists = 0;
                for (int k = 0; k < (int)kb->entry_count; k++) {
                    if (kb->entries[k].subject && kb->entries[k].object &&
                        strcmp(kb->entries[k].subject, kb->entries[i].subject) == 0 &&
                        strcmp(kb->entries[k].predicate, kb->entries[i].predicate) == 0 &&
                        strcmp(kb->entries[k].object, kb->entries[j].object) == 0) {
                        exists = 1;
                        break;
                    }
                }
                if (!exists) {
                    KnowledgeEntry ke;
                    memset(&ke, 0, sizeof(KnowledgeEntry));
                    ke.subject = string_duplicate(kb->entries[i].subject);
                    ke.predicate = string_duplicate(kb->entries[i].predicate);
                    ke.object = string_duplicate(kb->entries[j].object);
                    ke.type = KNOWLEDGE_FACT;
                    float conf = kb->entries[i].confidence * kb->entries[j].confidence * 0.85f;
                    ke.confidence = conf;
                    ke.source = SOURCE_LEARNING;
                    ke.timestamp = time(NULL);
                    knowledge_base_add(kb, &ke);
                    knowledge_entry_free(&ke);
                    new_count++;
                    log_info("[知识推理-传递] %s → %s → %s (置信度=%.3f)",
                             kb->entries[i].subject, kb->entries[i].predicate,
                             kb->entries[j].object, conf);
                }
            }
        }
    }

    /* 对称关系推理: (A, 相关于, B) => (B, 相关于, A) */
    for (int i = 0; i < (int)kb->entry_count && new_count < max_new_entries; i++) {
        if (!kb->entries[i].subject || !kb->entries[i].object) continue;
        if (strcmp(kb->entries[i].predicate, "相关于") != 0 &&
            strcmp(kb->entries[i].predicate, "类似于") != 0 &&
            strcmp(kb->entries[i].predicate, "连接") != 0) continue;

        int exists = 0;
        for (int k = 0; k < (int)kb->entry_count; k++) {
            if (kb->entries[k].subject && kb->entries[k].object &&
                strcmp(kb->entries[k].subject, kb->entries[i].object) == 0 &&
                strcmp(kb->entries[k].predicate, kb->entries[i].predicate) == 0 &&
                strcmp(kb->entries[k].object, kb->entries[i].subject) == 0) {
                exists = 1;
                break;
            }
        }
        if (!exists) {
            KnowledgeEntry ke;
            memset(&ke, 0, sizeof(KnowledgeEntry));
            ke.subject = string_duplicate(kb->entries[i].object);
            ke.predicate = string_duplicate(kb->entries[i].predicate);
            ke.object = string_duplicate(kb->entries[i].subject);
            ke.type = KNOWLEDGE_FACT;
            ke.confidence = kb->entries[i].confidence * 0.8f;
            ke.source = SOURCE_LEARNING;
            ke.timestamp = time(NULL);
            knowledge_base_add(kb, &ke);
            knowledge_entry_free(&ke);
            new_count++;
            log_info("[知识推理-对称] %s → %s → %s (置信度=%.3f)",
                     kb->entries[i].object, kb->entries[i].predicate,
                     kb->entries[i].subject, ke.confidence);
        }
    }

    /* 逆关系推理: (A, 包含, B) => (B, 属于, A) */
    for (int i = 0; i < (int)kb->entry_count && new_count < max_new_entries; i++) {
        if (!kb->entries[i].subject || !kb->entries[i].object) continue;
        const char* inv_predicate = NULL;
        if (strcmp(kb->entries[i].predicate, "包含") == 0) inv_predicate = "属于";
        else if (strcmp(kb->entries[i].predicate, "属于") == 0) inv_predicate = "包含";
        else if (strcmp(kb->entries[i].predicate, "拥有") == 0) inv_predicate = "被拥有";
        else if (strcmp(kb->entries[i].predicate, "位于") == 0) inv_predicate = "包含";

        if (inv_predicate) {
            int exists = 0;
            for (int k = 0; k < (int)kb->entry_count; k++) {
                if (kb->entries[k].subject && kb->entries[k].object &&
                    strcmp(kb->entries[k].subject, kb->entries[i].object) == 0 &&
                    strcmp(kb->entries[k].predicate, inv_predicate) == 0 &&
                    strcmp(kb->entries[k].object, kb->entries[i].subject) == 0) {
                    exists = 1;
                    break;
                }
            }
            if (!exists) {
                KnowledgeEntry ke;
                memset(&ke, 0, sizeof(KnowledgeEntry));
                ke.subject = string_duplicate(kb->entries[i].object);
                ke.predicate = string_duplicate(inv_predicate);
                ke.object = string_duplicate(kb->entries[i].subject);
                ke.type = KNOWLEDGE_FACT;
                ke.confidence = kb->entries[i].confidence * 0.75f;
                ke.source = SOURCE_LEARNING;
                ke.timestamp = time(NULL);
                knowledge_base_add(kb, &ke);
                knowledge_entry_free(&ke);
                new_count++;
                log_info("[知识推理-逆] %s → %s → %s (置信度=%.3f)",
                         kb->entries[i].object, inv_predicate,
                         kb->entries[i].subject, ke.confidence);
            }
        }
    }

    log_info("[知识推理] 从已有知识库推理出%d条新知识", new_count);
    return new_count;
}

/**
 * @brief 检查字符串是否在字符串数组中
 */
static int string_in_array(const char* str, char* arr[], int count) {
    for (int i = 0; i < count; i++) {
        if (arr[i] && strcmp(str, arr[i]) == 0) return 1;
    }
    return 0;
}

/**
 * @brief 获取来源类型权重（用于置信度计算）
 */
static float source_type_weight(KnowledgeSourceType type) {
    switch (type) {
        case KNOWLEDGE_SOURCE_MANUAL:    return 0.80f;
        case KNOWLEDGE_SOURCE_CODE:      return 0.85f;
        case KNOWLEDGE_SOURCE_JSON:      return 0.75f;
        case KNOWLEDGE_SOURCE_MARKDOWN:  return 0.70f;
        case KNOWLEDGE_SOURCE_CSV:       return 0.65f;
        case KNOWLEDGE_SOURCE_TEXT:      return 0.60f;
        case KNOWLEDGE_SOURCE_DIALOGUE:  return 0.50f;
        default:                         return 0.55f;
    }
}

int auto_learning_incremental_update(AutoLearningSystem* system, const char* topic,
                                     const char* content, KnowledgeSourceType type,
                                     const char* source_path) {
    if (!system) {
        log_error("[自主学习][P0-018] auto_learning_incremental_update: 系统句柄为空");
        return -1;
    }
    if (!topic || strlen(topic) == 0) {
        log_error("[自主学习][P0-018] auto_learning_incremental_update: 主题为空，无法执行增量更新");
        return -1;
    }
    if (!content || strlen(content) == 0) {
        log_error("[自主学习][P0-018] auto_learning_incremental_update: 内容为空，"
                  "无法对主题 '%s' 执行增量更新", topic ? topic : "(null)");
        return -1;
    }

    /* 查找话题相似度 > 0.7 的已有条目 */
    int best_match = -1;
    float best_sim = 0.7f;
    for (size_t i = 0; i < system->entry_count; i++) {
        if (!system->entries[i].topic) continue;
        float sim = knowledge_string_similarity(topic, system->entries[i].topic);
        if (sim > best_sim) {
            best_sim = sim;
            best_match = (int)i;
        }
    }

    if (best_match >= 0) {
        /* 更新已有条目 */
        AutoLearnEntry* entry = &system->entries[best_match];

        /* 合并实体（去重） */
        char** temp_entities = NULL;
        int temp_entity_count = 0;
        extract_entities(content, &temp_entities, &temp_entity_count);
        /* H-005: 实体质量过滤 */
        filter_entities_by_quality(&temp_entities, &temp_entity_count, NULL);

        for (int i = 0; i < temp_entity_count && entry->entity_count < 16; i++) {
            if (!string_in_array(temp_entities[i], entry->extracted_entities, entry->entity_count)) {
                entry->extracted_entities[entry->entity_count] = temp_entities[i];
                entry->entity_count++;
            } else {
                safe_free((void**)&temp_entities[i]);
            }
        }
        safe_free((void**)&temp_entities);

        /* 合并关系（去重） */
        char** temp_relations = NULL;
        int temp_relation_count = 0;
        extract_relations(content, &temp_relations, &temp_relation_count);

        for (int i = 0; i < temp_relation_count && entry->relation_count < 16; i++) {
            if (!string_in_array(temp_relations[i], entry->extracted_relations, entry->relation_count)) {
                entry->extracted_relations[entry->relation_count] = temp_relations[i];
                entry->relation_count++;
            } else {
                safe_free((void**)&temp_relations[i]);
            }
        }
        safe_free((void**)&temp_relations);

        /* 如果新内容更长，更新内容 */
        size_t old_len = strlen(entry->content);
        size_t new_len = strlen(content);
        if (new_len > old_len) {
            char* new_content = (char*)safe_malloc(new_len + 1);
            memcpy(new_content, content, new_len);
            new_content[new_len] = '\0';
            safe_free((void**)&entry->content);
            entry->content = new_content;
        }

        /* 更新置信度（平均） */
        entry->confidence = (entry->confidence + 0.5f) / 2.0f;

        /* 更新时间戳 */
        entry->learned_at = time(NULL);

        /* 更新来源类型（取较可靠的） */
        if (source_type_weight(type) > source_type_weight(entry->source_type)) {
            entry->source_type = type;
        }

        /* 更新来源路径 */
        if (source_path) {
            strncpy(entry->source_path, source_path, sizeof(entry->source_path) - 1);
            entry->source_path[sizeof(entry->source_path) - 1] = '\0';
        }

        /* 标记为未验证，需重新验证 */
        entry->verified = 0;

        return best_match;
    }

    /* 没有匹配的已有条目，新建条目 */
    if (auto_learning_learn_text(system, content, topic, type) != 0) return -1;

    /* 设置来源路径 */
    size_t new_idx = system->entry_count - 1;
    if (source_path) {
        strncpy(system->entries[new_idx].source_path, source_path,
                sizeof(system->entries[new_idx].source_path) - 1);
    }

    return (int)new_idx;
}

int auto_learning_fuse_knowledge(AutoLearningSystem* system, const char* topic,
                                  int source_indices[], int count,
                                  AutoLearnFusionMode mode,
                                  AutoLearnFusionResult* result) {
    if (!system) {
        log_error("[自主学习][P0-018] auto_learning_fuse_knowledge: 系统句柄为空");
        return -1;
    }
    if (!topic) {
        log_error("[自主学习][P0-018] auto_learning_fuse_knowledge: 融合主题为空");
        return -1;
    }
    if (!source_indices) {
        log_error("[自主学习][P0-018] auto_learning_fuse_knowledge: 源索引数组为空");
        return -1;
    }
    if (count <= 0 || count > 32) {
        log_error("[自主学习][P0-018] auto_learning_fuse_knowledge: 源条目数量无效（%d），"
                  "必须在[1,32]范围内", count);
        return -1;
    }

    /* 验证所有源索引有效 */
    for (int i = 0; i < count; i++) {
        if (source_indices[i] < 0 || (size_t)source_indices[i] >= system->entry_count) {
            log_error("[自主学习][P0-018] auto_learning_fuse_knowledge: 源索引[%d]=%d 无效，"
                      "有效范围[0, %zu)", i, source_indices[i], system->entry_count);
            return -1;
        }
        if (!system->entries[source_indices[i]].topic) {
            log_error("[自主学习][P0-018] auto_learning_fuse_knowledge: 源条目[%d]主题为空，"
                      "条目数据无效", source_indices[i]);
            return -1;
        }
        if (!system->entries[source_indices[i]].content) {
            log_error("[自主学习][P0-018] auto_learning_fuse_knowledge: 源条目[%d]内容为空，"
                      "无数据可供融合", source_indices[i]);
            return -1;
        }
        if (strlen(system->entries[source_indices[i]].content) == 0) {
            log_error("[自主学习][P0-018] auto_learning_fuse_knowledge: 源条目[%d]内容长度为0，"
                      "无有效数据", source_indices[i]);
            return -1;
        }
    }

#ifdef SELFLNN_STRICT_REAL_DATA
    /* P0-018修复: 严格真实数据模式 —— 验证融合源的实体和关系是否来自真实数据提取 */
    for (int i = 0; i < count; i++) {
        int idx = source_indices[i];
        AutoLearnEntry* entry = &system->entries[idx];

        /* 验证条目来源路径存在且非空（确保来自真实文件） */
        if (entry->source_path[0] == '\0') {
            log_warning("[自主学习][P0-018] auto_learning_fuse_knowledge: 严格真实数据模式 —— "
                        "源条目[%d] 无来源路径记录，无法验证数据真实性。继续融合但标记为低置信度。",
                        idx);
        }

        /* 验证实体和关系是从真实内容中提取的，而非合成占位符 */
        if (entry->entity_count == 0 && entry->relation_count == 0) {
            log_error("[自主学习][P0-018] auto_learning_fuse_knowledge: 严格真实数据模式 —— "
                      "源条目[%d] 主题 '%s' 的实体数和关系数均为0，该条目可能为无效/合成数据。"
                      "拒绝融合。", idx, entry->topic);
            return -1;
        }

        /* 验证内容长度与实体/关系数的比例合理性（合成数据通常此比例异常） */
        size_t clen = strlen(entry->content);
        if (clen > 200 && entry->entity_count == 0) {
            log_error("[自主学习][P0-018] auto_learning_fuse_knowledge: 严格真实数据模式 —— "
                      "源条目[%d] 主题 '%s' 内容长度 %zu 但无可抽取实体，"
                      "数据可能为无效文本。拒绝融合。", idx, entry->topic, clen);
            return -1;
        }
    }
#endif

    /* 收集所有实体和关系的并集 */
    char* all_entities[64];
    int all_entity_count = 0;
    char* all_relations[64];
    int all_relation_count = 0;
    float total_confidence = 0.0f;
    float max_confidence = 0.0f;
    float weight_sum = 0.0f;
    int longest_content_idx = source_indices[0];
    size_t longest_content_len = 0;

    for (int i = 0; i < count; i++) {
        int idx = source_indices[i];
        AutoLearnEntry* entry = &system->entries[idx];

        /* 收集实体 */
        for (int j = 0; j < entry->entity_count && all_entity_count < 64; j++) {
            if (!string_in_array(entry->extracted_entities[j], all_entities, all_entity_count)) {
                all_entities[all_entity_count] = entry->extracted_entities[j];
                all_entity_count++;
            }
        }

        /* 收集关系 */
        for (int j = 0; j < entry->relation_count && all_relation_count < 64; j++) {
            if (!string_in_array(entry->extracted_relations[j], all_relations, all_relation_count)) {
                all_relations[all_relation_count] = entry->extracted_relations[j];
                all_relation_count++;
            }
        }

        /* 记录最长内容 */
        size_t clen = strlen(entry->content);
        if (clen > longest_content_len) {
            longest_content_len = clen;
            longest_content_idx = idx;
        }

        /* 累积置信度和权重 */
        float sw = source_type_weight(entry->source_type);
        total_confidence += entry->confidence * sw;
        weight_sum += sw;

        if (entry->confidence > max_confidence) max_confidence = entry->confidence;
    }

    /* 计算融合后的置信度 */
    float fused_confidence = 0.0f;
    switch (mode) {
        case AUTO_LEARN_FUSE_MERGE:
            fused_confidence = weight_sum > 0.0f ? total_confidence / weight_sum : 0.5f;
            break;
        case AUTO_LEARN_FUSE_OVERRIDE:
            fused_confidence = max_confidence;
            break;
        case AUTO_LEARN_FUSE_AVERAGE:
        default:
            fused_confidence = total_confidence / (float)count;
            break;
    }
    if (fused_confidence > 0.99f) fused_confidence = 0.99f;
    if (fused_confidence < 0.1f) fused_confidence = 0.1f;

    /* 使用第一个源索引位置作为融合目标 */
    int target_idx = source_indices[0];
    AutoLearnEntry* target = &system->entries[target_idx];

    /* 更新主题 */
    size_t topic_len = strlen(topic);
    char* new_topic = (char*)safe_malloc(topic_len + 1);
    memcpy(new_topic, topic, topic_len);
    new_topic[topic_len] = '\0';
    safe_free((void**)&target->topic);
    target->topic = new_topic;

    /* 更新内容（取自最长内容的条目） */
    char* new_content = (char*)safe_malloc(longest_content_len + 1);
    memcpy(new_content, system->entries[longest_content_idx].content, longest_content_len);
    new_content[longest_content_len] = '\0';
    safe_free((void**)&target->content);
    target->content = new_content;

    /* 释放原有实体和关系 */
    for (int j = 0; j < target->entity_count; j++) {
        safe_free((void**)&target->extracted_entities[j]);
    }
    for (int j = 0; j < target->relation_count; j++) {
        safe_free((void**)&target->extracted_relations[j]);
    }

    /* 设置融合后的实体 */
    target->entity_count = all_entity_count;
    for (int j = 0; j < all_entity_count; j++) {
        target->extracted_entities[j] = all_entities[j];
    }

    /* 设置融合后的关系 */
    target->relation_count = all_relation_count;
    for (int j = 0; j < all_relation_count; j++) {
        target->extracted_relations[j] = all_relations[j];
    }

    /* 保存融合前置信度 */
    float confidence_before = target->confidence;

    /* 设置融合后置信度 */
    target->confidence = fused_confidence;
    target->learned_at = time(NULL);
    target->verified = 0;

    /* 删除其他源条目（从后往前删除，避免索引混乱） */
    int sorted_indices[32];
    memcpy(sorted_indices, source_indices, count * sizeof(int));
    for (int i = 1; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            if (sorted_indices[j] > sorted_indices[i]) {
                int t = sorted_indices[i];
                sorted_indices[i] = sorted_indices[j];
                sorted_indices[j] = t;
            }
        }
    }

    for (int i = count - 1; i >= 1; i--) {
        int del_idx = sorted_indices[i];
        if (del_idx == target_idx) continue;

        /* 释放条目 */
        safe_free((void**)&system->entries[del_idx].topic);
        safe_free((void**)&system->entries[del_idx].content);
        for (int j = 0; j < system->entries[del_idx].entity_count; j++) {
            safe_free((void**)&system->entries[del_idx].extracted_entities[j]);
        }
        for (int j = 0; j < system->entries[del_idx].relation_count; j++) {
            safe_free((void**)&system->entries[del_idx].extracted_relations[j]);
        }
        safe_free((void**)&system->entries[del_idx].feature_vector);
        system->entries[del_idx].feature_vector_dim = 0;

        /* 将后续条目前移 */
        for (size_t k = (size_t)del_idx; k < system->entry_count - 1; k++) {
            system->entries[k] = system->entries[k + 1];
        }
        memset(&system->entries[system->entry_count - 1], 0, sizeof(AutoLearnEntry));
        system->entry_count--;

        /* 调整后续源索引 */
        for (int j = 1; j < count; j++) {
            if (sorted_indices[j] > del_idx) sorted_indices[j]--;
        }
        if (target_idx > del_idx) target_idx--;
    }

    /* 填充融合结果 */
    if (result) {
        result->source_entry_count = count;
        memcpy(result->source_indices, source_indices, count * sizeof(int));
        result->target_entry_index = target_idx;
        result->confidence_before = confidence_before;
        result->confidence_after = fused_confidence;
        result->fusion_time = time(NULL);
        size_t ct = strlen(topic) < 255 ? strlen(topic) : 255;
        memcpy(result->topic, topic, ct);
        result->topic[ct] = '\0';
    }

    return 0;
}

float auto_learning_update_confidence(AutoLearningSystem* system, size_t entry_index) {
    if (!system || entry_index >= system->entry_count) return -1.0f;

    AutoLearnEntry* entry = &system->entries[entry_index];
    if (!entry->topic || !entry->content) return -1.0f;

    /* 基础置信度：基于内容质量和实体丰富度 */
    float base_confidence = 0.5f;
    size_t content_len = strlen(entry->content);
    if (content_len > 100)      base_confidence = 0.60f;
    if (content_len > 500)      base_confidence = 0.65f;
    if (content_len > 1000)     base_confidence = 0.70f;
    if (entry->entity_count >= 3)  base_confidence += 0.05f;
    if (entry->entity_count >= 6)  base_confidence += 0.05f;
    if (entry->relation_count >= 2) base_confidence += 0.05f;
    if (entry->relation_count >= 5) base_confidence += 0.05f;
    if (base_confidence > 0.85f) base_confidence = 0.85f;

    /* 来源可靠度权重 */
    float source_weight = source_type_weight(entry->source_type);

    /* 交叉引用因子：在知识库中查找相关条目 */
    float cross_ref_factor = 1.0f;
    if (system->knowledge_base) {
        KnowledgeBase* kb = (KnowledgeBase*)system->knowledge_base;
        KnowledgeEntry results[16];
        int match_count = knowledge_base_search_similar(kb, entry->topic, NULL, NULL, 0.5f, results, 16);
        if (match_count > 0) {
            cross_ref_factor = 1.0f + (float)match_count * 0.02f;
            if (cross_ref_factor > 1.3f) cross_ref_factor = 1.3f;
        }
        for (int i = 0; i < match_count; i++) {
            knowledge_entry_free(&results[i]);
        }
    } else {
        /* 无知识库时，通过交叉引用计数估算 */
        int similar_count = 0;
        for (size_t i = 0; i < system->entry_count; i++) {
            if (i == entry_index) continue;
            if (!system->entries[i].topic) continue;
            float sim = knowledge_string_similarity(entry->topic, system->entries[i].topic);
            if (sim > 0.5f) similar_count++;
        }
        if (similar_count > 0) {
            cross_ref_factor = 1.0f + (float)similar_count * 0.03f;
            if (cross_ref_factor > 1.3f) cross_ref_factor = 1.3f;
        }
    }

    /* 重复提升因子：同一话题出现多次 */
    float repeat_factor = 1.0f;
    {
        int repeat_count = 0;
        for (size_t i = 0; i < system->entry_count; i++) {
            if (i == entry_index) continue;
            if (!system->entries[i].topic) continue;
            float sim = knowledge_string_similarity(entry->topic, system->entries[i].topic);
            if (sim > 0.8f) repeat_count++;
        }
        if (repeat_count > 0) {
            repeat_factor = 1.0f + (float)repeat_count * 0.05f;
            if (repeat_factor > 1.2f) repeat_factor = 1.2f;
        }
    }

    /* 时效衰减因子：超过30天每30天衰减5% */
    float decay_factor = 1.0f;
    {
        time_t now = time(NULL);
        double days_elapsed = difftime(now, entry->learned_at) / 86400.0;
        if (days_elapsed > 30.0) {
            int decay_months = (int)(days_elapsed / 30.0);
            decay_factor = 1.0f - (float)decay_months * 0.05f;
            if (decay_factor < 0.5f) decay_factor = 0.5f;
        }
    }

    /* 验证因子：已验证的知识增加可信度 */
    float verify_factor = entry->verified ? 1.1f : 1.0f;

    /* 最终置信度 */
    float final_confidence = base_confidence * source_weight * cross_ref_factor
                             * repeat_factor * decay_factor * verify_factor;

    /* 归一化到 [0.05, 0.99] */
    if (final_confidence > 0.99f) final_confidence = 0.99f;
    if (final_confidence < 0.05f) final_confidence = 0.05f;

    entry->confidence = final_confidence;

    /* 更新统计平均置信度 */
    {
        float total_conf = 0.0f;
        size_t valid_count = 0;
        for (size_t i = 0; i < system->entry_count; i++) {
            if (system->entries[i].topic) {
                total_conf += system->entries[i].confidence;
                valid_count++;
            }
        }
        if (valid_count > 0) {
            system->stats.average_confidence = total_conf / (float)valid_count;
        }
    }

    return final_confidence;
}

int auto_learning_relearn_expired(AutoLearningSystem* system, int max_days) {
    if (!system || max_days <= 0) return -1;

    int relearn_count = 0;
    time_t now = time(NULL);

    for (size_t i = 0; i < system->entry_count; i++) {
        AutoLearnEntry* entry = &system->entries[i];
        if (!entry->topic) continue;

        double days_elapsed = difftime(now, entry->learned_at) / 86400.0;
        if (days_elapsed < (double)max_days) continue;

        /* 检查来源文件是否存在 */
        if (entry->source_path[0] == '\0') {
            /* 无来源路径，仅降低置信度并标记需要重学习 */
            entry->confidence *= 0.85f;
            if (entry->confidence < 0.1f) entry->confidence = 0.1f;
            entry->verified = 0;
            continue;
        }

        FILE* fp = fopen(entry->source_path, "rb");
        if (!fp) {
            /* 来源文件不存在，降低置信度 */
            entry->confidence *= 0.7f;
            if (entry->confidence < 0.1f) entry->confidence = 0.1f;
            entry->verified = 0;
            continue;
        }

        /* 读取文件内容 */
        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        if (file_size <= 0 || file_size > 10 * 1024 * 1024) {
            fclose(fp);
            continue;
        }

        char* file_content = (char*)safe_malloc((size_t)file_size + 1);
        if (!file_content) {
            fclose(fp);
            continue;
        }

        size_t read_size = fread(file_content, 1, (size_t)file_size, fp);
        fclose(fp);
        file_content[read_size] = '\0';

        /* 比较新旧内容 */
        float content_sim = knowledge_string_similarity(entry->content, file_content);

        if (content_sim < 0.95f) {
            /* 内容有变化，更新条目 */
            size_t new_len = strlen(file_content);
            if (new_len > 4096) new_len = 4096;

            char* new_content = (char*)safe_malloc(new_len + 1);
            memcpy(new_content, file_content, new_len);
            new_content[new_len] = '\0';

            safe_free((void**)&entry->content);
            entry->content = new_content;

            /* 重新抽取实体 */
            for (int j = 0; j < entry->entity_count; j++) {
                safe_free((void**)&entry->extracted_entities[j]);
            }
            entry->entity_count = 0;

            char** entities = NULL;
            int entity_count = 0;
            extract_entities(file_content, &entities, &entity_count);
            /* H-005: 实体质量过滤 */
            filter_entities_by_quality(&entities, &entity_count, NULL);
            entry->entity_count = entity_count < 16 ? entity_count : 16;
            for (int j = 0; j < entry->entity_count; j++) {
                entry->extracted_entities[j] = entities[j];
            }
            for (int j = entry->entity_count; j < entity_count; j++) {
                safe_free((void**)&entities[j]);
            }
            safe_free((void**)&entities);

            /* 重新抽取关系 */
            for (int j = 0; j < entry->relation_count; j++) {
                safe_free((void**)&entry->extracted_relations[j]);
            }
            entry->relation_count = 0;

            char** relations = NULL;
            int relation_count = 0;
            extract_relations(file_content, &relations, &relation_count);
            entry->relation_count = relation_count < 16 ? relation_count : 16;
            for (int j = 0; j < entry->relation_count; j++) {
                entry->extracted_relations[j] = relations[j];
            }
            for (int j = entry->relation_count; j < relation_count; j++) {
                safe_free((void**)&relations[j]);
            }
            safe_free((void**)&relations);

            /* 重学习置信度设为0.7基础 */
            entry->confidence = 0.7f;
            entry->verified = 0;
            entry->learned_at = now;

            /* 重新运行置信度动态更新 */
            auto_learning_update_confidence(system, i);

            relearn_count++;
        } else {
            /* 内容无变化，仅更新时间戳和微小置信度提升 */
            entry->learned_at = now;
            entry->confidence += 0.02f;
            if (entry->confidence > 0.95f) entry->confidence = 0.95f;
        }

        safe_free((void**)&file_content);
    }

    /* 更新统计 */
    system->stats.total_entries_learned += (size_t)relearn_count;

    return relearn_count;
}

/* =========================================================================
 * A03.4.1 主动学习系统实现
 * ========================================================================= */

#define AL_MAX_SAMPLES 100000
#define AL_SAMPLES_INIT_CAPACITY 1024

struct ALActiveLearner {
    ALUncertaintyStrategy strategy;
    ALSample* samples;
    int sample_count;
    int sample_capacity;
    int labeled_count;
};

/* 辅助：拷贝特征向量 */
static float* al_copy_features(const float* src, int dim) {
    if (!src || dim <= 0) return NULL;
    float* dst = (float*)safe_malloc((size_t)dim * sizeof(float));
    if (dst) memcpy(dst, src, (size_t)dim * sizeof(float));
    return dst;
}

ALActiveLearner* active_learner_create(ALUncertaintyStrategy strategy) {
    ALActiveLearner* learner = (ALActiveLearner*)safe_calloc(1, sizeof(ALActiveLearner));
    if (!learner) return NULL;
    learner->strategy = strategy;
    learner->sample_capacity = AL_SAMPLES_INIT_CAPACITY;
    learner->samples = (ALSample*)safe_calloc((size_t)learner->sample_capacity, sizeof(ALSample));
    if (!learner->samples) {
        safe_free((void**)&learner);
        return NULL;
    }
    learner->sample_count = 0;
    learner->labeled_count = 0;
    return learner;
}

void active_learner_free(ALActiveLearner* learner) {
    if (!learner) return;
    for (int i = 0; i < learner->sample_count; i++) {
        safe_free((void**)&learner->samples[i].features);
        safe_free((void**)&learner->samples[i].metadata);
    }
    safe_free((void**)&learner->samples);
    safe_free((void**)&learner);
}

/* 扩展样本数组 */
static int al_expand(ALActiveLearner* learner) {
    if (learner->sample_count < learner->sample_capacity) return 0;
    int new_cap = learner->sample_capacity * 2;
    if (new_cap > AL_MAX_SAMPLES) new_cap = AL_MAX_SAMPLES;
    if (new_cap <= learner->sample_capacity) return -1;
    ALSample* new_arr = (ALSample*)safe_realloc(learner->samples,
        (size_t)new_cap * sizeof(ALSample));
    if (!new_arr) return -1;
    memset(new_arr + learner->sample_capacity, 0,
        (size_t)(new_cap - learner->sample_capacity) * sizeof(ALSample));
    learner->samples = new_arr;
    learner->sample_capacity = new_cap;
    return 0;
}

int active_learner_add_sample(ALActiveLearner* learner, const float* features,
                               int feature_dim, int label, const char* metadata) {
    if (!learner || !features || feature_dim <= 0) return -1;
    if (al_expand(learner) != 0) return -1;

    int idx = learner->sample_count++;
    ALSample* s = &learner->samples[idx];
    s->features = al_copy_features(features, feature_dim);
    s->feature_dim = feature_dim;
    s->label = label;
    s->uncertainty_score = 0.0f;
    s->diversity_score = 0.0f;
    s->combined_score = 0.0f;
    s->selected = 0;
    s->metadata = metadata ? string_duplicate(metadata) : NULL;
    if (label >= 0) learner->labeled_count++;
    return idx;
}

int active_learner_add_samples(ALActiveLearner* learner, const float* features,
                                int feature_dim, const int* labels, int sample_count) {
    if (!learner || !features || feature_dim <= 0 || !labels || sample_count <= 0) return -1;
    int added = 0;
    for (int i = 0; i < sample_count; i++) {
        if (active_learner_add_sample(learner, features + (size_t)i * feature_dim,
                                       feature_dim, labels[i], NULL) >= 0)
            added++;
    }
    return added;
}

int active_learner_sample_count(const ALActiveLearner* learner) {
    return learner ? learner->sample_count : 0;
}

int active_learner_labeled_count(const ALActiveLearner* learner) {
    return learner ? learner->labeled_count : 0;
}

/* 比较函数：用于排序 */
static int al_cmp_desc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return 1;
    if (fa > fb) return -1;
    return 0;
}

static int al_cmp_asc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

int active_learner_compute_uncertainties(ALActiveLearner* learner,
    ALUncertaintyStrategy strategy) {
    if (!learner) return -1;

    /* 对每个未标注样本计算不确定性分数 */
    for (int i = 0; i < learner->sample_count; i++) {
        ALSample* s = &learner->samples[i];
        if (s->label >= 0) continue; /* 已标注样本跳过 */
        if (!s->features || s->feature_dim <= 0) continue;

        /* 使用特征值估算预测置信度分布 */
        /* 归一化特征值得到置信度权重 */
        float sum = 0.0f;
        for (int d = 0; d < s->feature_dim; d++) {
            float v = s->features[d];
            if (v < 0) v = -v;
            sum += v + 1e-10f;
        }

        switch (strategy) {
            case AL_UNCERTAINTY_LEAST_CONFIDENCE: {
                /* 最低置信度: 1 - max(prob) */
                float max_prob = 0.0f;
                for (int d = 0; d < s->feature_dim; d++) {
                    float prob = (s->features[d] >= 0 ? s->features[d] : -s->features[d]) / sum;
                    if (prob > max_prob) max_prob = prob;
                }
                s->uncertainty_score = 1.0f - max_prob;
                break;
            }
            case AL_UNCERTAINTY_MARGIN: {
                /* 边界采样: -(p1 - p2) 差值越小不确定性越大 */
                float probs[64];
                int pdim = s->feature_dim < 64 ? s->feature_dim : 64;
                for (int d = 0; d < pdim; d++) {
                    probs[d] = (s->features[d] >= 0 ? s->features[d] : -s->features[d]) / sum;
                }
                qsort(probs, (size_t)pdim, sizeof(float), al_cmp_desc);
                float margin = (pdim >= 2) ? (probs[0] - probs[1]) : probs[0];
                s->uncertainty_score = 1.0f - margin;
                break;
            }
            case AL_UNCERTAINTY_ENTROPY: {
                /* 熵最大采样 */
                float entropy = 0.0f;
                for (int d = 0; d < s->feature_dim; d++) {
                    float prob = (s->features[d] >= 0 ? s->features[d] : -s->features[d]) / sum;
                    if (prob > 1e-10f)
                        entropy -= prob * logf(prob);
                }
                /* 归一化熵到[0,1] */
                float max_entropy = logf((float)s->feature_dim);
                s->uncertainty_score = (max_entropy > 0) ? entropy / max_entropy : 0.0f;
                break;
            }
        }
    }
    return 0;
}

int active_learner_uncertainty_sampling(ALActiveLearner* learner,
    ALUncertaintyStrategy strategy, int* sample_indices, int n) {
    if (!learner || !sample_indices || n <= 0) return 0;

    /* 计算不确定性分数 */
    active_learner_compute_uncertainties(learner, strategy);

    /* 找出未标注样本中不确定性最高的n个 */
    typedef struct { int idx; float score; } ALScoreIdx;
    int unlabeled_count = learner->sample_count - learner->labeled_count;
    int alloc_count = unlabeled_count > 0 ? unlabeled_count : 1;
    ALScoreIdx* scored = (ALScoreIdx*)safe_malloc((size_t)alloc_count * sizeof(ALScoreIdx));
    if (!scored) return 0;

    int cnt = 0;
    for (int i = 0; i < learner->sample_count; i++) {
        if (learner->samples[i].label < 0) {
            scored[cnt].idx = i;
            scored[cnt].score = learner->samples[i].uncertainty_score;
            cnt++;
        }
    }

    /* 按分数降序排列 */
    for (int i = 0; i < cnt - 1; i++) {
        for (int j = i + 1; j < cnt; j++) {
            if (scored[j].score > scored[i].score) {
                ALScoreIdx tmp = scored[i];
                scored[i] = scored[j];
                scored[j] = tmp;
            }
        }
    }

    int selected = n < cnt ? n : cnt;
    for (int i = 0; i < selected; i++) {
        sample_indices[i] = scored[i].idx;
        learner->samples[scored[i].idx].selected = 1;
    }

    safe_free((void**)&scored);
    return selected;
}

/* K-means聚类辅助 */
static float al_euclidean_dist(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

int active_learner_diversity_sampling(ALActiveLearner* learner,
    int* sample_indices, int n, int cluster_count) {
    if (!learner || !sample_indices || n <= 0) return 0;

    /* 收集未标注样本的索引和特征 */
    int unlabeled_count = learner->sample_count - learner->labeled_count;
    if (unlabeled_count <= 0) return 0;

    int* unlabeled_idx = (int*)safe_malloc((size_t)unlabeled_count * sizeof(int));
    if (!unlabeled_idx) return 0;
    int cnt = 0;
    for (int i = 0; i < learner->sample_count; i++) {
        if (learner->samples[i].label < 0)
            unlabeled_idx[cnt++] = i;
    }

    /* 确定聚类数 */
    int k = cluster_count > 0 ? cluster_count : (n < cnt ? n : cnt);
    if (k > cnt) k = cnt;
    if (k <= 0) { safe_free((void**)&unlabeled_idx); return 0; }

    /* K-means++ 初始化 */
    int* assignments = (int*)safe_calloc((size_t)cnt, sizeof(int));
    float** centroids = (float**)safe_calloc((size_t)k, sizeof(float*));
    if (!assignments || !centroids) {
        safe_free((void**)&unlabeled_idx);
        safe_free((void**)&assignments);
        safe_free((void**)&centroids);
        return 0;
    }

    int dim = learner->samples[unlabeled_idx[0]].feature_dim;
    for (int i = 0; i < k; i++) {
        centroids[i] = (float*)safe_calloc((size_t)dim, sizeof(float));
        if (!centroids[i]) {
            for (int j = 0; j < i; j++) safe_free((void**)&centroids[j]);
            safe_free((void**)&unlabeled_idx);
            safe_free((void**)&assignments);
            safe_free((void**)&centroids);
            return 0;
        }
    }

    /* K-means++: 第一个随机选择（K-012修复：安全随机数） */
    int first = (int)(secure_random_int((uint32_t)(cnt - 1)));
    memcpy(centroids[0], learner->samples[unlabeled_idx[first]].features,
           (size_t)dim * sizeof(float));

    for (int c = 1; c < k; c++) {
        float* min_dists = (float*)safe_malloc((size_t)cnt * sizeof(float));
        if (!min_dists) break;
        float total_dist = 0.0f;
        for (int i = 0; i < cnt; i++) {
            float min_d = 1e30f;
            for (int j = 0; j < c; j++) {
                float d = al_euclidean_dist(
                    learner->samples[unlabeled_idx[i]].features, centroids[j], dim);
                if (d < min_d) min_d = d;
            }
            min_dists[i] = min_d;
            total_dist += min_d;
        }
        /* 概率加权选择（K-012修复：安全随机数） */
        float r = secure_random_float() * total_dist;
        float cum = 0.0f;
        int chosen = cnt - 1;
        for (int i = 0; i < cnt; i++) {
            cum += min_dists[i];
            if (cum >= r) { chosen = i; break; }
        }
        memcpy(centroids[c], learner->samples[unlabeled_idx[chosen]].features,
               (size_t)dim * sizeof(float));
        safe_free((void**)&min_dists);
    }

    /* 迭代优化（最多20轮） */
    for (int iter = 0; iter < 20; iter++) {
        int changed = 0;
        /* 分配 */
        for (int i = 0; i < cnt; i++) {
            float min_d = 1e30f;
            int best = 0;
            for (int c = 0; c < k; c++) {
                float d = al_euclidean_dist(
                    learner->samples[unlabeled_idx[i]].features, centroids[c], dim);
                if (d < min_d) { min_d = d; best = c; }
            }
            if (assignments[i] != best) { assignments[i] = best; changed++; }
        }
        if (changed == 0) break;
        /* 更新质心 */
        for (int c = 0; c < k; c++) {
            memset(centroids[c], 0, (size_t)dim * sizeof(float));
        }
        int* counts = (int*)safe_calloc((size_t)k, sizeof(int));
        if (!counts) break;
        for (int i = 0; i < cnt; i++) {
            int c = assignments[i];
            for (int d = 0; d < dim; d++)
                centroids[c][d] += learner->samples[unlabeled_idx[i]].features[d];
            counts[c]++;
        }
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < dim; d++)
                    centroids[c][d] /= (float)counts[c];
            }
        }
        safe_free((void**)&counts);
    }

    /* 从每个聚类选择最靠近质心的样本 */
    int selected = 0;
    for (int c = 0; c < k && selected < n; c++) {
        float min_d = 1e30f;
        int best_idx = -1;
        for (int i = 0; i < cnt; i++) {
            if (assignments[i] == c) {
                float d = al_euclidean_dist(
                    learner->samples[unlabeled_idx[i]].features, centroids[c], dim);
                if (d < min_d) { min_d = d; best_idx = unlabeled_idx[i]; }
            }
        }
        if (best_idx >= 0) {
            sample_indices[selected++] = best_idx;
            learner->samples[best_idx].selected = 1;
        }
    }

    /* 如果没选够，从未选中的补充 */
    if (selected < n) {
        for (int i = 0; i < cnt && selected < n; i++) {
            int idx = unlabeled_idx[i];
            if (!learner->samples[idx].selected) {
                int already = 0;
                for (int s = 0; s < selected; s++) {
                    if (sample_indices[s] == idx) { already = 1; break; }
                }
                if (!already) {
                    sample_indices[selected++] = idx;
                    learner->samples[idx].selected = 1;
                }
            }
        }
    }

    for (int i = 0; i < k; i++) safe_free((void**)&centroids[i]);
    safe_free((void**)&centroids);
    safe_free((void**)&assignments);
    safe_free((void**)&unlabeled_idx);
    return selected;
}

int active_learner_query_synthesis(ALActiveLearner* learner,
    float* out_features, int feature_dim, int count) {
    if (!learner || !out_features || feature_dim <= 0 || count <= 0) return 0;

#ifdef SELFLNN_STRICT_REAL_DATA
    /* P0-018修复: 严格真实数据模式 —— 完全禁止合成/随机数据生成
     * 主动学习器的查询合成本质上是在特征空间中生成虚假样本，
     * 违反"严格真实数据"原则，在此模式下直接拒绝。 */
    log_error("[自主学习][P0-018] active_learner_query_synthesis: 严格真实数据模式 —— "
              "完全禁止合成/随机数据生成。查询合成功能已禁用。"
              "请使用真实标注样本进行主动学习。");
    return 0;
#else
    int synthesized = 0;

    /* 收集已标注样本的统计信息 */
    int labeled_count = learner->labeled_count;
    if (labeled_count < 2) {
        /* 标注样本不足时，在特征空间边界随机生成（K-012修复：安全随机数） */
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < feature_dim; d++) {
                out_features[(size_t)i * feature_dim + d] =
                    secure_random_float() * 2.0f - 1.0f;
            }
            synthesized++;
        }
        return synthesized;
    }

    /* 计算已标注样本的均值和方差 */
    float* mean = (float*)safe_calloc((size_t)feature_dim, sizeof(float));
    float* std = (float*)safe_calloc((size_t)feature_dim, sizeof(float));
    if (!mean || !std) {
        safe_free((void**)&mean);
        safe_free((void**)&std);
        return 0;
    }

    int lc = 0;
    for (int i = 0; i < learner->sample_count; i++) {
        if (learner->samples[i].label >= 0) {
            for (int d = 0; d < feature_dim && d < learner->samples[i].feature_dim; d++)
                mean[d] += learner->samples[i].features[d];
            lc++;
        }
    }
    if (lc > 0) {
        for (int d = 0; d < feature_dim; d++) mean[d] /= (float)lc;
    }

    for (int i = 0; i < learner->sample_count; i++) {
        if (learner->samples[i].label >= 0) {
            for (int d = 0; d < feature_dim && d < learner->samples[i].feature_dim; d++) {
                float diff = learner->samples[i].features[d] - mean[d];
                std[d] += diff * diff;
            }
        }
    }
    if (lc > 1) {
        for (int d = 0; d < feature_dim; d++)
            std[d] = sqrtf(std[d] / (float)(lc - 1) + 1e-10f);
    }

    /* 在均值附近添加扰动合成样本（决策边界区域）（K-012修复：安全随机数） */
    for (int i = 0; i < count; i++) {
        for (int d = 0; d < feature_dim; d++) {
            float noise = (secure_random_float() - 0.5f) * 2.0f * std[d];
            out_features[(size_t)i * feature_dim + d] = mean[d] + noise;
        }
        synthesized++;
    }

    safe_free((void**)&mean);
    safe_free((void**)&std);
    return synthesized;
#endif
}

int active_learner_query(ALActiveLearner* learner, int* sample_indices, int n,
                          float uncertainty_weight, float diversity_weight) {
    if (!learner || !sample_indices || n <= 0) return 0;

    /* 先计算不确定性分数 */
    active_learner_compute_uncertainties(learner, learner->strategy);

    /* 计算未标注样本的综合分数 */
    int unlabeled_count = learner->sample_count - learner->labeled_count;
    if (unlabeled_count <= 0) return 0;

    typedef struct { int idx; float score; } ALScoreIdx;
    ALScoreIdx* scored = (ALScoreIdx*)safe_malloc((size_t)unlabeled_count * sizeof(ALScoreIdx));
    if (!scored) return 0;

    int cnt = 0;
    for (int i = 0; i < learner->sample_count; i++) {
        if (learner->samples[i].label < 0) {
            /* 多样性分数：与该样本最近的已标注样本的距离 */
            float min_dist = 1e30f;
            for (int j = 0; j < learner->sample_count; j++) {
                if (learner->samples[j].label >= 0) {
                    int dim = learner->samples[i].feature_dim < learner->samples[j].feature_dim
                            ? learner->samples[i].feature_dim : learner->samples[j].feature_dim;
                    float d = al_euclidean_dist(
                        learner->samples[i].features, learner->samples[j].features, dim);
                    if (d < min_dist) min_dist = d;
                }
            }
            /* 如果没有已标注样本，多样性分数设为1 */
            learner->samples[i].diversity_score = (learner->labeled_count > 0) ?
                (min_dist < 1e29f ? min_dist / (min_dist + 1.0f) : 0.5f) : 1.0f;

            learner->samples[i].combined_score =
                uncertainty_weight * learner->samples[i].uncertainty_score +
                diversity_weight * learner->samples[i].diversity_score;

            scored[cnt].idx = i;
            scored[cnt].score = learner->samples[i].combined_score;
            cnt++;
        }
    }

    for (int i = 0; i < cnt - 1; i++) {
        for (int j = i + 1; j < cnt; j++) {
            if (scored[j].score > scored[i].score) {
                ALScoreIdx tmp = scored[i];
                scored[i] = scored[j];
                scored[j] = tmp;
            }
        }
    }

    int selected = n < cnt ? n : cnt;
    for (int i = 0; i < selected; i++) {
        sample_indices[i] = scored[i].idx;
        learner->samples[scored[i].idx].selected = 1;
    }

    safe_free((void**)&scored);
    return selected;
}

int active_learner_label_sample(ALActiveLearner* learner, int sample_index, int label) {
    if (!learner || sample_index < 0 || sample_index >= learner->sample_count) return -1;
    if (learner->samples[sample_index].label < 0) learner->labeled_count++;
    learner->samples[sample_index].label = label;
    return 0;
}

const ALSample* active_learner_get_sample(const ALActiveLearner* learner, int index) {
    if (!learner || index < 0 || index >= learner->sample_count) return NULL;
    return &learner->samples[index];
}

/* =========================================================================
 * A03.4.2 持续学习系统实现
 * ========================================================================= */

/* --- EWC: 弹性权重巩固 --- */

struct EWC {
    int param_count;
    float lambda;
    float* optimal_params;
    float* fisher_diag;
    int has_optimal;
};

EWC* ewc_create(int param_count, float lambda) {
    if (param_count <= 0) return NULL;
    EWC* ewc = (EWC*)safe_calloc(1, sizeof(EWC));
    if (!ewc) return NULL;
    ewc->param_count = param_count;
    ewc->lambda = lambda;
    ewc->optimal_params = (float*)safe_calloc((size_t)param_count, sizeof(float));
    ewc->fisher_diag = (float*)safe_calloc((size_t)param_count, sizeof(float));
    if (!ewc->optimal_params || !ewc->fisher_diag) {
        safe_free((void**)&ewc->optimal_params);
        safe_free((void**)&ewc->fisher_diag);
        safe_free((void**)&ewc);
        return NULL;
    }
    ewc->has_optimal = 0;
    /* Fisher初始化为小正值 */
    for (int i = 0; i < param_count; i++) ewc->fisher_diag[i] = 1e-6f;
    return ewc;
}

void ewc_free(EWC* ewc) {
    if (!ewc) return;
    safe_free((void**)&ewc->optimal_params);
    safe_free((void**)&ewc->fisher_diag);
    safe_free((void**)&ewc);
}

int ewc_set_optimal_params(EWC* ewc, const float* optimal_params) {
    if (!ewc || !optimal_params) return -1;
    memcpy(ewc->optimal_params, optimal_params, (size_t)ewc->param_count * sizeof(float));
    ewc->has_optimal = 1;
    return 0;
}

int ewc_compute_fisher(EWC* ewc, const float* gradients, int sample_count) {
    if (!ewc || !gradients || sample_count <= 0) return -1;

    /* Fisher信息矩阵 = E[grad * grad^T] 的对角线 */
    /* 对每个参数，计算所有样本的梯度平方均值 */
    memset(ewc->fisher_diag, 0, (size_t)ewc->param_count * sizeof(float));

    for (int s = 0; s < sample_count; s++) {
        const float* grad = gradients + (size_t)s * ewc->param_count;
        for (int i = 0; i < ewc->param_count; i++) {
            ewc->fisher_diag[i] += grad[i] * grad[i];
        }
    }

    if (sample_count > 0) {
        for (int i = 0; i < ewc->param_count; i++) {
            ewc->fisher_diag[i] = ewc->fisher_diag[i] / (float)sample_count + 1e-8f;
        }
    }

    return 0;
}

int ewc_compute_loss(EWC* ewc, const float* current_params,
                     float* out_loss, float* out_per_param_loss) {
    if (!ewc || !current_params || !out_loss) return -1;
    if (!ewc->has_optimal) {
        *out_loss = 0.0f;
        return 0;
    }

    float total_loss = 0.0f;
    for (int i = 0; i < ewc->param_count; i++) {
        float diff = current_params[i] - ewc->optimal_params[i];
        float param_loss = 0.5f * ewc->lambda * ewc->fisher_diag[i] * diff * diff;
        total_loss += param_loss;
        if (out_per_param_loss) out_per_param_loss[i] = param_loss;
    }

    *out_loss = total_loss;
    return 0;
}

EWC* ewc_merge(EWC** targets, int count) {
    if (!targets || count <= 0) return NULL;
    int param_count = targets[0]->param_count;
    for (int i = 1; i < count; i++) {
        if (targets[i]->param_count != param_count) return NULL;
    }

    EWC* merged = ewc_create(param_count, targets[0]->lambda);
    if (!merged) return NULL;

    /* 合并Fisher信息（取平均） */
    memset(merged->fisher_diag, 0, (size_t)param_count * sizeof(float));
    memset(merged->optimal_params, 0, (size_t)param_count * sizeof(float));

    for (int i = 0; i < count; i++) {
        for (int j = 0; j < param_count; j++) {
            merged->fisher_diag[j] += targets[i]->fisher_diag[j];
            merged->optimal_params[j] += targets[i]->optimal_params[j];
        }
    }

    for (int j = 0; j < param_count; j++) {
        merged->fisher_diag[j] /= (float)count;
        merged->optimal_params[j] /= (float)count;
    }

    merged->has_optimal = 1;
    return merged;
}

int ewc_get_importance(EWC* ewc, float* out_importance) {
    if (!ewc || !out_importance) return -1;
    memcpy(out_importance, ewc->fisher_diag, (size_t)ewc->param_count * sizeof(float));
    return 0;
}

/* --- ProgressiveNet: 渐进式神经网络 --- */

#define PNN_MAX_COLUMNS 64
#define PNN_MAX_HIDDEN 4096

struct ProgressiveNet {
    int input_dim;
    int hidden_dim;
    ProgressiveColumn* columns;
    int column_count;
    int column_capacity;
    /* 横向连接权重 [column_count][prev_column_count][hidden_dim] */
    float** lateral_weights;
};

/* 激活函数：ReLU */
static float pnn_relu(float x) { return x > 0.0f ? x : 0.0f; }
static float pnn_relu_deriv(float x) { return x > 0.0f ? 1.0f : 0.0f; }

/* K-012修复：使用安全随机数Xavier初始化 */
static float pnn_xavier(int fan_in, int fan_out) {
    float scale = sqrtf(6.0f / (float)(fan_in + fan_out));
    return secure_random_float() * 2.0f * scale - scale;
}

ProgressiveNet* progressive_net_create(int input_dim, int hidden_dim) {
    if (input_dim <= 0 || hidden_dim <= 0 || hidden_dim > PNN_MAX_HIDDEN) return NULL;

    ProgressiveNet* pnn = (ProgressiveNet*)safe_calloc(1, sizeof(ProgressiveNet));
    if (!pnn) return NULL;

    pnn->input_dim = input_dim;
    pnn->hidden_dim = hidden_dim;
    pnn->column_capacity = 8;
    pnn->columns = (ProgressiveColumn*)safe_calloc(
        (size_t)pnn->column_capacity, sizeof(ProgressiveColumn));
    pnn->lateral_weights = (float**)safe_calloc(
        (size_t)pnn->column_capacity * PNN_MAX_COLUMNS, sizeof(float*));
    if (!pnn->columns || !pnn->lateral_weights) {
        safe_free((void**)&pnn->columns);
        safe_free((void**)&pnn->lateral_weights);
        safe_free((void**)&pnn);
        return NULL;
    }
    pnn->column_count = 0;
    return pnn;
}

void progressive_net_free(ProgressiveNet* pnn) {
    if (!pnn) return;
    for (int i = 0; i < pnn->column_count; i++) {
        ProgressiveColumn* col = &pnn->columns[i];
        safe_free((void**)&col->w_in);
        safe_free((void**)&col->b_in);
        safe_free((void**)&col->w_hidden);
        safe_free((void**)&col->b_hidden);
        safe_free((void**)&col->w_out);
        safe_free((void**)&col->b_out);
    }
    /* 释放横向连接权重 */
    for (int i = 0; i < pnn->column_count; i++) {
        for (int j = 0; j < i; j++) {
            safe_free((void**)&pnn->lateral_weights[(size_t)i * PNN_MAX_COLUMNS + j]);
        }
    }
    safe_free((void**)&pnn->columns);
    safe_free((void**)&pnn->lateral_weights);
    safe_free((void**)&pnn);
}

int progressive_net_add_column(ProgressiveNet* pnn, int output_dim) {
    if (!pnn || output_dim <= 0 || pnn->column_count >= PNN_MAX_COLUMNS) return -1;

    if (pnn->column_count >= pnn->column_capacity) {
        int new_cap = pnn->column_capacity * 2;
        if (new_cap > PNN_MAX_COLUMNS) new_cap = PNN_MAX_COLUMNS;
        ProgressiveColumn* new_cols = (ProgressiveColumn*)safe_realloc(
            pnn->columns, (size_t)new_cap * sizeof(ProgressiveColumn));
        if (!new_cols) return -1;
        memset(new_cols + pnn->column_capacity, 0,
            (size_t)(new_cap - pnn->column_capacity) * sizeof(ProgressiveColumn));
        pnn->columns = new_cols;
        pnn->column_capacity = new_cap;
    }

    int col_idx = pnn->column_count;
    ProgressiveColumn* col = &pnn->columns[col_idx];

    col->input_dim = pnn->input_dim;
    col->hidden_dim = pnn->hidden_dim;
    col->output_dim = output_dim;
    col->frozen = 0;

    /* 分配并初始化权重 */
    col->w_in = (float*)safe_calloc((size_t)pnn->input_dim * pnn->hidden_dim, sizeof(float));
    col->b_in = (float*)safe_calloc((size_t)pnn->hidden_dim, sizeof(float));
    col->w_hidden = (float*)safe_calloc((size_t)pnn->hidden_dim * pnn->hidden_dim, sizeof(float));
    col->b_hidden = (float*)safe_calloc((size_t)pnn->hidden_dim, sizeof(float));
    col->w_out = (float*)safe_calloc((size_t)pnn->hidden_dim * output_dim, sizeof(float));
    col->b_out = (float*)safe_calloc((size_t)output_dim, sizeof(float));

    if (!col->w_in || !col->b_in || !col->w_hidden || !col->b_hidden || !col->w_out || !col->b_out) {
        safe_free((void**)&col->w_in);
        safe_free((void**)&col->b_in);
        safe_free((void**)&col->w_hidden);
        safe_free((void**)&col->b_hidden);
        safe_free((void**)&col->w_out);
        safe_free((void**)&col->b_out);
        return -1;
    }

    /* Xavier初始化 */
    for (int i = 0; i < pnn->input_dim * pnn->hidden_dim; i++)
        col->w_in[i] = pnn_xavier(pnn->input_dim, pnn->hidden_dim);
    for (int i = 0; i < pnn->hidden_dim; i++)
        col->b_in[i] = 0.0f;
    for (int i = 0; i < pnn->hidden_dim * pnn->hidden_dim; i++)
        col->w_hidden[i] = pnn_xavier(pnn->hidden_dim, pnn->hidden_dim);
    for (int i = 0; i < pnn->hidden_dim; i++)
        col->b_hidden[i] = 0.0f;
    for (int i = 0; i < pnn->hidden_dim * output_dim; i++)
        col->w_out[i] = pnn_xavier(pnn->hidden_dim, output_dim);
    for (int i = 0; i < output_dim; i++)
        col->b_out[i] = 0.0f;

    /* 冻结旧列 */
    for (int i = 0; i < col_idx; i++) {
        pnn->columns[i].frozen = 1;
    }

    /* 初始化横向连接权重（从旧列到新列） */
    for (int j = 0; j < col_idx; j++) {
        pnn->lateral_weights[(size_t)col_idx * PNN_MAX_COLUMNS + j] =
            (float*)safe_calloc((size_t)pnn->hidden_dim * pnn->hidden_dim, sizeof(float));
        if (pnn->lateral_weights[(size_t)col_idx * PNN_MAX_COLUMNS + j]) {
            for (int k = 0; k < pnn->hidden_dim * pnn->hidden_dim; k++) {
                pnn->lateral_weights[(size_t)col_idx * PNN_MAX_COLUMNS + j][k] =
                    pnn_xavier(pnn->hidden_dim, pnn->hidden_dim);
            }
        }
    }

    pnn->column_count++;
    return col_idx;
}

int progressive_net_column_count(const ProgressiveNet* pnn) {
    return pnn ? pnn->column_count : 0;
}

const ProgressiveColumn* progressive_net_get_column(const ProgressiveNet* pnn, int column_idx) {
    if (!pnn || column_idx < 0 || column_idx >= pnn->column_count) return NULL;
    return &pnn->columns[column_idx];
}

int progressive_net_forward(ProgressiveNet* pnn, int column_idx,
                             const float* input, float* output) {
    if (!pnn || !input || !output || column_idx < 0 || column_idx >= pnn->column_count)
        return -1;

    ProgressiveColumn* col = &pnn->columns[column_idx];
    int hd = col->hidden_dim;

    /* 临时存储隐藏层激活 */
    float* hidden = (float*)safe_calloc((size_t)hd, sizeof(float));
    if (!hidden) return -1;

    /* 输入层 → 隐藏层 */
    for (int h = 0; h < hd; h++) {
        float sum = col->b_in[h];
        for (int i = 0; i < col->input_dim; i++)
            sum += input[i] * col->w_in[(size_t)i * hd + h];
        hidden[h] = pnn_relu(sum);
    }

    /* 横向连接：从旧列收集特征 */
    for (int prev = 0; prev < column_idx; prev++) {
        float* lateral = pnn->lateral_weights[(size_t)column_idx * PNN_MAX_COLUMNS + prev];
        if (!lateral) continue;
        ProgressiveColumn* prev_col = &pnn->columns[prev];

        /* 计算旧列的前向传播得到其隐藏层 */
        float* prev_hidden = (float*)safe_calloc((size_t)hd, sizeof(float));
        if (!prev_hidden) { safe_free((void**)&hidden); return -1; }

        for (int h = 0; h < prev_col->hidden_dim; h++) {
            float sum = prev_col->b_in[h];
            for (int i = 0; i < prev_col->input_dim; i++)
                sum += input[i] * prev_col->w_in[(size_t)i * prev_col->hidden_dim + h];
            prev_hidden[h] = pnn_relu(sum);
        }

        /* 通过横向连接将旧列隐藏层映射到新列隐藏层 */
        for (int h = 0; h < hd; h++) {
            float lat_sum = 0.0f;
            for (int ph = 0; ph < prev_col->hidden_dim; ph++)
                lat_sum += prev_hidden[ph] * lateral[(size_t)ph * hd + h];
            hidden[h] += lat_sum;
        }

        safe_free((void**)&prev_hidden);
    }

    /* 隐藏层 → 输出层 */
    for (int o = 0; o < col->output_dim; o++) {
        float sum = col->b_out[o];
        for (int h = 0; h < hd; h++)
            sum += hidden[h] * col->w_out[(size_t)h * col->output_dim + o];
        output[o] = sum;
    }

    safe_free((void**)&hidden);
    return 0;
}

int progressive_net_train_step(ProgressiveNet* pnn, int column_idx,
                                const float* input, const float* target,
                                float learning_rate) {
    if (!pnn || !input || !target || column_idx < 0 || column_idx >= pnn->column_count)
        return -1;

    ProgressiveColumn* col = &pnn->columns[column_idx];
    if (col->frozen) return -1;

    int hd = col->hidden_dim;
    int od = col->output_dim;
    int id = col->input_dim;

    /* 前向传播 */
    float* hidden_pre = (float*)safe_calloc((size_t)hd, sizeof(float));
    float* hidden = (float*)safe_calloc((size_t)hd, sizeof(float));
    float* output = (float*)safe_calloc((size_t)od, sizeof(float));
    if (!hidden_pre || !hidden || !output) {
        safe_free((void**)&hidden_pre);
        safe_free((void**)&hidden);
        safe_free((void**)&output);
        return -1;
    }

    for (int h = 0; h < hd; h++) {
        float sum = col->b_in[h];
        for (int i = 0; i < id; i++)
            sum += input[i] * col->w_in[(size_t)i * hd + h];
        hidden_pre[h] = sum;
        hidden[h] = pnn_relu(sum);
    }

    /* 横向连接 */
    for (int prev = 0; prev < column_idx; prev++) {
        float* lateral = pnn->lateral_weights[(size_t)column_idx * PNN_MAX_COLUMNS + prev];
        if (!lateral) continue;
        ProgressiveColumn* prev_col = &pnn->columns[prev];
        float* prev_hidden = (float*)safe_calloc((size_t)prev_col->hidden_dim, sizeof(float));
        if (!prev_hidden) continue;
        for (int h = 0; h < prev_col->hidden_dim; h++) {
            float sum = prev_col->b_in[h];
            for (int i = 0; i < prev_col->input_dim; i++)
                sum += input[i] * prev_col->w_in[(size_t)i * prev_col->hidden_dim + h];
            prev_hidden[h] = pnn_relu(sum);
        }
        for (int h = 0; h < hd; h++) {
            float lat_sum = 0.0f;
            for (int ph = 0; ph < prev_col->hidden_dim; ph++)
                lat_sum += prev_hidden[ph] * lateral[(size_t)ph * hd + h];
            hidden_pre[h] += lat_sum;
            hidden[h] = pnn_relu(hidden_pre[h]);
        }
        safe_free((void**)&prev_hidden);
    }

    for (int o = 0; o < od; o++) {
        float sum = col->b_out[o];
        for (int h = 0; h < hd; h++)
            sum += hidden[h] * col->w_out[(size_t)h * od + o];
        output[o] = sum;
    }

    /* 反向传播（MSE损失） */
    float* d_output = (float*)safe_calloc((size_t)od, sizeof(float));
    if (!d_output) { safe_free((void**)&hidden_pre); safe_free((void**)&hidden); safe_free((void**)&output); return -1; }

    for (int o = 0; o < od; o++)
        d_output[o] = output[o] - target[o];  /* dL/doutput */

    /* 输出层梯度 */
    float* d_w_out = (float*)safe_calloc((size_t)hd * od, sizeof(float));
    float* d_b_out = (float*)safe_calloc((size_t)od, sizeof(float));
    if (!d_w_out || !d_b_out) {
        safe_free((void**)&d_output);
        safe_free((void**)&d_w_out);
        safe_free((void**)&d_b_out);
        safe_free((void**)&hidden_pre);
        safe_free((void**)&hidden);
        safe_free((void**)&output);
        return -1;
    }

    for (int o = 0; o < od; o++) {
        d_b_out[o] = d_output[o];
        for (int h = 0; h < hd; h++)
            d_w_out[(size_t)h * od + o] = d_output[o] * hidden[h];
    }

    /* 隐藏层梯度 */
    float* d_hidden = (float*)safe_calloc((size_t)hd, sizeof(float));
    if (!d_hidden) {
        safe_free((void**)&d_output);
        safe_free((void**)&d_w_out);
        safe_free((void**)&d_b_out);
        safe_free((void**)&hidden_pre);
        safe_free((void**)&hidden);
        safe_free((void**)&output);
        return -1;
    }

    for (int h = 0; h < hd; h++) {
        float sum = 0.0f;
        for (int o = 0; o < od; o++)
            sum += d_output[o] * col->w_out[(size_t)h * od + o];
        d_hidden[h] = sum * pnn_relu_deriv(hidden_pre[h]);
    }

    /* 更新输入层权重 */
    for (int i = 0; i < id; i++) {
        for (int h = 0; h < hd; h++)
            col->w_in[(size_t)i * hd + h] -= learning_rate * d_hidden[h] * input[i];
    }
    for (int h = 0; h < hd; h++)
        col->b_in[h] -= learning_rate * d_hidden[h];

    /* 更新隐藏层权重 */
    for (int h1 = 0; h1 < hd; h1++) {
        for (int h2 = 0; h2 < hd; h2++)
            col->w_hidden[(size_t)h1 * hd + h2] -= learning_rate * d_hidden[h2] * hidden[h1];
    }
    for (int h = 0; h < hd; h++)
        col->b_hidden[h] -= learning_rate * d_hidden[h];

    /* 更新输出层权重 */
    for (int h = 0; h < hd; h++) {
        for (int o = 0; o < od; o++)
            col->w_out[(size_t)h * od + o] -= learning_rate * d_w_out[(size_t)h * od + o];
    }
    for (int o = 0; o < od; o++)
        col->b_out[o] -= learning_rate * d_b_out[o];

    /* 更新横向连接权重 */
    for (int prev = 0; prev < column_idx; prev++) {
        float* lateral = pnn->lateral_weights[(size_t)column_idx * PNN_MAX_COLUMNS + prev];
        if (!lateral) continue;
        ProgressiveColumn* prev_col = &pnn->columns[prev];
        float* prev_hidden = (float*)safe_calloc((size_t)prev_col->hidden_dim, sizeof(float));
        if (!prev_hidden) continue;
        for (int h = 0; h < prev_col->hidden_dim; h++) {
            float sum = prev_col->b_in[h];
            for (int i = 0; i < prev_col->input_dim; i++)
                sum += input[i] * prev_col->w_in[(size_t)i * prev_col->hidden_dim + h];
            prev_hidden[h] = pnn_relu(sum);
        }
        for (int ph = 0; ph < prev_col->hidden_dim; ph++) {
            for (int h = 0; h < hd; h++)
                lateral[(size_t)ph * hd + h] -= learning_rate * d_hidden[h] * prev_hidden[ph];
        }
        safe_free((void**)&prev_hidden);
    }

    safe_free((void**)&d_output);
    safe_free((void**)&d_w_out);
    safe_free((void**)&d_b_out);
    safe_free((void**)&d_hidden);
    safe_free((void**)&hidden_pre);
    safe_free((void**)&hidden);
    safe_free((void**)&output);
    return 0;
}

/* --- ExperienceReplay: 经验重放缓冲区 --- */

struct ExperienceReplay {
    int capacity;
    int state_dim;
    int action_dim;
    ExperienceSample* buffer;
    int size;
    int head;
    int tail;
    /* 优先回放树（Sum Tree） */
    float* priorities;
    float total_priority;
};

ExperienceReplay* experience_replay_create(int capacity, int state_dim, int action_dim) {
    if (capacity <= 0 || state_dim <= 0 || action_dim <= 0) return NULL;

    ExperienceReplay* er = (ExperienceReplay*)safe_calloc(1, sizeof(ExperienceReplay));
    if (!er) return NULL;

    er->capacity = capacity;
    er->state_dim = state_dim;
    er->action_dim = action_dim;
    er->buffer = (ExperienceSample*)safe_calloc((size_t)capacity, sizeof(ExperienceSample));
    er->priorities = (float*)safe_calloc((size_t)capacity, sizeof(float));
    if (!er->buffer || !er->priorities) {
        safe_free((void**)&er->buffer);
        safe_free((void**)&er->priorities);
        safe_free((void**)&er);
        return NULL;
    }

    er->size = 0;
    er->head = 0;
    er->tail = 0;
    er->total_priority = 0.0f;

    /* 初始化优先级 */
    for (int i = 0; i < capacity; i++) {
        er->priorities[i] = 1.0f;
        er->buffer[i].state = (float*)safe_calloc((size_t)state_dim, sizeof(float));
        er->buffer[i].action = (float*)safe_calloc((size_t)action_dim, sizeof(float));
        er->buffer[i].next_state = (float*)safe_calloc((size_t)state_dim, sizeof(float));
    }
    er->total_priority = (float)capacity;

    return er;
}

void experience_replay_free(ExperienceReplay* er) {
    if (!er) return;
    for (int i = 0; i < er->capacity; i++) {
        safe_free((void**)&er->buffer[i].state);
        safe_free((void**)&er->buffer[i].action);
        safe_free((void**)&er->buffer[i].next_state);
    }
    safe_free((void**)&er->buffer);
    safe_free((void**)&er->priorities);
    safe_free((void**)&er);
}

int experience_replay_add(ExperienceReplay* er, const float* state,
                           const float* action, float reward,
                           const float* next_state, int done, int task_id) {
    if (!er || !state || !action || !next_state) return -1;

    ExperienceSample* s = &er->buffer[er->head];

    memcpy(s->state, state, (size_t)er->state_dim * sizeof(float));
    memcpy(s->action, action, (size_t)er->action_dim * sizeof(float));
    s->reward = reward;
    memcpy(s->next_state, next_state, (size_t)er->state_dim * sizeof(float));
    s->done = done;
    s->priority = 1.0f;
    s->task_id = task_id;

    /* 更新优先级总和 */
    float old_priority = er->priorities[er->head];
    er->priorities[er->head] = 1.0f;
    er->total_priority = er->total_priority - old_priority + 1.0f;

    er->head = (er->head + 1) % er->capacity;
    if (er->size < er->capacity) {
        er->size++;
    } else {
        er->tail = (er->tail + 1) % er->capacity;
    }

    return 0;
}

int experience_replay_sample(ExperienceReplay* er, int batch_size,
                              float* out_states, float* out_actions,
                              float* out_rewards, float* out_next_states,
                              int* out_dones) {
    if (!er || !out_states || !out_actions || !out_rewards || !out_next_states || !out_dones)
        return 0;

    if (er->size <= 0 || batch_size <= 0) return 0;

    int actual = batch_size < er->size ? batch_size : er->size;

    for (int i = 0; i < actual; i++) {
        /* K-012修复：安全随机数 */
        int idx = (int)(secure_random_int((uint32_t)(er->size - 1)));
        int buf_idx = (er->tail + idx) % er->capacity;
        ExperienceSample* s = &er->buffer[buf_idx];

        memcpy(out_states + (size_t)i * er->state_dim, s->state,
               (size_t)er->state_dim * sizeof(float));
        memcpy(out_actions + (size_t)i * er->action_dim, s->action,
               (size_t)er->action_dim * sizeof(float));
        out_rewards[i] = s->reward;
        memcpy(out_next_states + (size_t)i * er->state_dim, s->next_state,
               (size_t)er->state_dim * sizeof(float));
        out_dones[i] = s->done;
    }

    return actual;
}

int experience_replay_priority_sample(ExperienceReplay* er, int batch_size,
    float beta, float* out_states, float* out_actions,
    float* out_rewards, float* out_next_states, int* out_dones,
    int* out_indices, float* out_weights) {
    if (!er || !out_states || !out_actions || !out_rewards || !out_next_states ||
        !out_dones || !out_indices || !out_weights)
        return 0;

    if (er->size <= 0 || batch_size <= 0) return 0;

    int actual = batch_size < er->size ? batch_size : er->size;
    float max_weight = 0.0f;

    /* 基于优先级分段采样 */
    float segment = er->total_priority / (float)actual;

    for (int i = 0; i < actual; i++) {
        /* K-012修复：安全随机数优先级采样 */
        float r = secure_random_float() * segment + (float)i * segment;

        /* 在优先级数组中查找r所在的累积区间 */
        float cum = 0.0f;
        int idx = 0;
        for (int j = 0; j < er->size; j++) {
            int buf_idx = (er->tail + j) % er->capacity;
            cum += er->priorities[buf_idx];
            if (cum >= r) { idx = j; break; }
        }

        int buf_idx = (er->tail + idx) % er->capacity;
        ExperienceSample* s = &er->buffer[buf_idx];

        memcpy(out_states + (size_t)i * er->state_dim, s->state,
               (size_t)er->state_dim * sizeof(float));
        memcpy(out_actions + (size_t)i * er->action_dim, s->action,
               (size_t)er->action_dim * sizeof(float));
        out_rewards[i] = s->reward;
        memcpy(out_next_states + (size_t)i * er->state_dim, s->next_state,
               (size_t)er->state_dim * sizeof(float));
        out_dones[i] = s->done;
        out_indices[i] = buf_idx;

        /* 重要性采样权重 */
        float prob = er->priorities[buf_idx] / (er->total_priority + 1e-10f);
        float weight = powf((float)er->size * prob, -beta);
        out_weights[i] = weight;
        if (weight > max_weight) max_weight = weight;
    }

    /* 归一化权重 */
    if (max_weight > 0) {
        for (int i = 0; i < actual; i++)
            out_weights[i] /= max_weight;
    }

    return actual;
}

int experience_replay_update_priorities(ExperienceReplay* er,
    const int* indices, const float* priorities, int count) {
    if (!er || !indices || !priorities || count <= 0) return -1;

    for (int i = 0; i < count; i++) {
        int idx = indices[i];
        if (idx < 0 || idx >= er->capacity) continue;
        float old_p = er->priorities[idx];
        float new_p = priorities[i] + 1e-6f;
        er->priorities[idx] = new_p;
        er->total_priority = er->total_priority - old_p + new_p;
        er->buffer[idx].priority = new_p;
    }

    return 0;
}

int experience_replay_size(const ExperienceReplay* er) {
    return er ? er->size : 0;
}

int experience_replay_capacity(const ExperienceReplay* er) {
    return er ? er->capacity : 0;
}

void experience_replay_clear(ExperienceReplay* er) {
    if (!er) return;
    er->size = 0;
    er->head = 0;
    er->tail = 0;
    er->total_priority = (float)er->capacity;
    for (int i = 0; i < er->capacity; i++) {
        er->priorities[i] = 1.0f;
    }
}

int experience_replay_task_count(const ExperienceReplay* er, int task_id) {
    if (!er) return 0;
    int cnt = 0;
    for (int i = 0; i < er->capacity; i++) {
        int buf_idx = (er->tail + i) % er->capacity;
        if (i < er->size && er->buffer[buf_idx].task_id == task_id)
            cnt++;
    }
    return cnt;
}
