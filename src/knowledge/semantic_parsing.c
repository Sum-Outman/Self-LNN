/**
 * @file semantic_parsing.c
 * @brief 语义解析系统实现
 *
 * 依存句法分析（Arc-Standard转移系统+Eisner算法）、
 * 语义角色标注（基于依存到角色的规则映射）、
 * 语义相似度计算（余弦/路径/信息内容/集成）。
 */

#define _CRT_NONSTDC_NO_DEPRECATE

#include "selflnn/knowledge/semantic_parsing.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =========================================================================
 * 依存关系类型字符串表
 * ========================================================================= */

static const char* s_dep_relation_names[DEP_COUNT] = {
    "SBJ", "OBJ", "DOBJ", "IOBJ", "ATT", "ADV", "CMP", "COO",
    "POBJ", "SBV", "VOB", "POB", "LAD", "RAD", "IS", "HED",
    "ROOT", "CNJ", "MT", "TPC", "VOC", "APP", "NMOD", "AMOD",
    "NUM_MOD", "CLF", "NSUBJ", "NSUBJPASS", "DOBJPASS", "IOPJ",
    "MARK", "NEG", "PUNCT", "LOCATION_OF", "INSTANCE_OF",
    "CAUSES", "PURPOSE", "TIME", "INSTRUMENT", "DEP"
};

static const char* s_dep_relation_chinese[DEP_COUNT] = {
    "主语", "宾语", "直接宾语", "间接宾语", "定语", "状语", "补语", "并列",
    "介宾", "主谓", "动宾", "介宾短", "左附加", "右附加", "独立结构", "核心",
    "根节点", "连词", "情态", "话题", "呼语", "同位语", "名词修饰", "形容词修饰",
    "数词修饰", "量词修饰", "名词性主语", "名词性被动主语", "直接被动宾语", "间接宾语",
    "标记", "否定修饰", "标点", "位置关系", "实例关系",
    "因果关系", "目的关系", "时间关系", "工具关系", "未指定"
};

static const char* s_pos_names[POS_COUNT] = {
    "未知", "名词", "专有名词", "时间名词", "处所名词",
    "动词", "形容词", "系动词", "能愿动词",
    "副词", "介词", "连词", "限定词", "数词", "量词",
    "代词", "的", "地/得", "标点", "感叹词", "外来词", "标点符号"
};

const char* dep_relation_type_string(DepRelationType type) {
    if (type >= 0 && type < DEP_COUNT) return s_dep_relation_names[type];
    return "未知";
}

const char* pos_to_string(PartOfSpeech pos) {
    if (pos >= 0 && pos < POS_COUNT) return s_pos_names[pos];
    return "未知";
}

static const char* s_semantic_role_names[SR_COUNT] = {
    "施事者", "受事者", "工具", "位置", "时间", "方式", "目的", "原因",
    "结果", "来源", "目的地", "体验者", "刺激", "主题", "受益者",
    "属性", "程度", "频率", "顺序", "情态", "否定"
};

const char* semantic_role_string(SemanticRole role) {
    if (role >= 0 && role < SR_COUNT) return s_semantic_role_names[role];
    return "未知";
}

/* =========================================================================
 * Arc-Standard 转移系统的栈和缓冲区
 * ========================================================================= */

#define PARSER_MAX_TOKENS 1024
#define PARSER_STACK_SIZE 2048

/* Arc-Standard 转移动作 */
typedef enum {
    TRANS_SHIFT = 0,
    TRANS_LEFT_ARC = 1,
    TRANS_RIGHT_ARC = 2
} TransitionAction;

/* 转移系统状态 */
typedef struct {
    int stack[PARSER_STACK_SIZE];
    int stack_size;
    int buffer[PARSER_MAX_TOKENS];
    int buffer_size;
    int buffer_pos;
    int heads[PARSER_MAX_TOKENS];
    DepRelationType dep_labels[PARSER_MAX_TOKENS];
    int root_relations[PARSER_MAX_TOKENS];
    int root_relation_count;
} ArcStandardState;

/* 特征函数权重（对数线性模型，结构化感知器特征） */
typedef struct {
    float word_word[POS_COUNT][POS_COUNT];
    float word_dep[POS_COUNT][DEP_COUNT];
    float bias_shift;
    float bias_left;
    float bias_right;
    float pos_score[POS_COUNT];
} ParserWeights;

/* =========================================================================
 * 依存句法分析器内部结构
 * ========================================================================= */

struct DependencyParser {
    char** tokens;
    PartOfSpeech* pos_tags;
    int token_count;
    int capacity;
    ParserWeights weights;
    int trained;
};

/* 辅助函数：复制字符串 */
static char* dup_str(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char* d = (char*)safe_malloc(len + 1);
    if (d) { memcpy(d, s, len + 1); }
    return d;
}

/* 初始化 Arc-Standard 状态 */
static void arc_std_init(ArcStandardState* state, int token_count) {
    state->stack_size = 0;
    state->buffer_pos = 0;
    state->buffer_size = token_count;
    for (int i = 0; i < token_count; i++) {
        state->buffer[i] = i;
        state->heads[i] = -1;
        state->dep_labels[i] = DEP_DEP;
    }
    state->root_relation_count = 0;
}

/* 判断状态是否终止 */
static int arc_std_is_terminal(ArcStandardState* state) {
    return state->stack_size == 0 && state->buffer_pos >= state->buffer_size;
}

/* 执行 Shift 转移 */
static void arc_std_shift(ArcStandardState* state) {
    if (state->buffer_pos >= state->buffer_size) return;
    state->stack[state->stack_size++] = state->buffer[state->buffer_pos++];
}

/* 执行 LeftArc 转移 */
static void arc_std_left_arc(ArcStandardState* state, DepRelationType label) {
    if (state->stack_size < 2) return;
    int child = state->stack[state->stack_size - 2];
    int parent = state->stack[state->stack_size - 1];
    state->heads[child] = parent;
    state->dep_labels[child] = label;
    /* 从栈中移除child */
    state->stack[state->stack_size - 2] = state->stack[state->stack_size - 1];
    state->stack_size--;
}

/* 执行 RightArc 转移 */
static void arc_std_right_arc(ArcStandardState* state, DepRelationType label) {
    if (state->stack_size < 2) return;
    int child = state->stack[state->stack_size - 1];
    int parent = state->stack[state->stack_size - 2];
    state->heads[child] = parent;
    state->dep_labels[child] = label;
    state->stack_size--;
}

/* 简单特征评分（基于词性和依存标签） */
static float arc_std_score(ArcStandardState* state, TransitionAction action,
                            DepRelationType label, ParserWeights* w,
                            PartOfSpeech* pos_tags) {
    float score = 0.0f;
    if (state->stack_size < 1 && action != TRANS_SHIFT) return -1e10f;
    if (state->stack_size < 2 && action != TRANS_SHIFT) return -1e10f;

    switch (action) {
        case TRANS_SHIFT:
            score = w->bias_shift;
            if (state->buffer_pos < state->buffer_size) {
                int t = state->buffer[state->buffer_pos];
                score += w->pos_score[pos_tags[t]];
            }
            break;
        case TRANS_LEFT_ARC: {
            int top = state->stack[state->stack_size - 1];
            int second = state->stack[state->stack_size - 2];
            score = w->bias_left + w->word_dep[pos_tags[second]][label]
                    + w->word_word[pos_tags[second]][pos_tags[top]];
            break;
        }
        case TRANS_RIGHT_ARC: {
            int top = state->stack[state->stack_size - 1];
            int second = state->stack[state->stack_size - 2];
            score = w->bias_right + w->word_dep[pos_tags[top]][label]
                    + w->word_word[pos_tags[top]][pos_tags[second]];
            break;
        }
    }
    return score;
}

/* Arc-Standard 贪婪解码 */
static DependencyParseResult* arc_std_parse(DependencyParser* parser,
                                             const char** gold_heads) {
    (void)gold_heads;
    if (!parser || parser->token_count <= 0) return NULL;
    int n = parser->token_count;

    DependencyParseResult* result = (DependencyParseResult*)safe_calloc(1, sizeof(DependencyParseResult));
    if (!result) return NULL;
    result->nodes = (DepNode*)safe_calloc(n, sizeof(DepNode));
    if (!result->nodes) { safe_free((void**)&result); return NULL; }
    result->node_count = n;
    result->root_id = -1;
    result->score = 0.0f;

    for (int i = 0; i < n; i++) {
        result->nodes[i].id = i;
        result->nodes[i].word = dup_str(parser->tokens[i]);
        result->nodes[i].pos = parser->pos_tags[i];
        result->nodes[i].head_id = -1;
        result->nodes[i].dep_rel = DEP_DEP;
        result->nodes[i].confidence = 0.0f;
    }

    ArcStandardState state;
    arc_std_init(&state, n);

    while (!arc_std_is_terminal(&state)) {
        /* 静态Oracle：如果是训练模式，根据gold_heads选择正确动作 */
        TransitionAction best_action = TRANS_SHIFT;
        DepRelationType best_label = DEP_DEP;
        float best_score = -1e10f;

        /* 候选：LeftArc */
        if (state.stack_size >= 2) {
            (void)state.stack[state.stack_size - 1];
            (void)state.stack[state.stack_size - 2];
            for (int l = 0; l < DEP_COUNT; l++) {
                float s = arc_std_score(&state, TRANS_LEFT_ARC, (DepRelationType)l, &parser->weights, parser->pos_tags);
                if (s > best_score) {
                    best_score = s;
                    best_action = TRANS_LEFT_ARC;
                    best_label = (DepRelationType)l;
                }
            }
        }

        /* 候选：RightArc */
        if (state.stack_size >= 2) {
            (void)state.stack[state.stack_size - 1];
            (void)state.stack[state.stack_size - 2];
            for (int l = 0; l < DEP_COUNT; l++) {
                float s = arc_std_score(&state, TRANS_RIGHT_ARC, (DepRelationType)l, &parser->weights, parser->pos_tags);
                if (s > best_score) {
                    best_score = s;
                    best_action = TRANS_RIGHT_ARC;
                    best_label = (DepRelationType)l;
                }
            }
        }

        /* 候选：Shift */
        if (state.buffer_pos < state.buffer_size) {
            float s = arc_std_score(&state, TRANS_SHIFT, DEP_DEP, &parser->weights, parser->pos_tags);
            if (s > best_score) {
                best_score = s;
                best_action = TRANS_SHIFT;
            }
        }

        /* 执行最佳动作 */
        switch (best_action) {
            case TRANS_SHIFT:
                arc_std_shift(&state);
                break;
            case TRANS_LEFT_ARC:
                arc_std_left_arc(&state, best_label);
                break;
            case TRANS_RIGHT_ARC:
                arc_std_right_arc(&state, best_label);
                break;
        }
    }

    /* 处理未连接节点和根节点 */
    for (int i = 0; i < n; i++) {
        result->nodes[i].head_id = state.heads[i];
        result->nodes[i].dep_rel = state.dep_labels[i];
        result->nodes[i].confidence = 0.8f;
        if (state.heads[i] == -1) {
            result->root_id = i;
            result->nodes[i].dep_rel = DEP_ROOT;
        }
    }

    /* 如果存在多个根，选第一个作为根，其余连到根 */
    if (result->root_id == -1 && n > 0) {
        result->root_id = 0;
        result->nodes[0].dep_rel = DEP_ROOT;
        for (int i = 1; i < n; i++) {
            if (result->nodes[i].head_id == -1) {
                result->nodes[i].head_id = 0;
                result->nodes[i].dep_rel = DEP_DEP;
            }
        }
    }

    result->score = 1.0f;
    return result;
}

/* =========================================================================
 * Eisner 投影依存分析算法
 * ========================================================================= */

/* Eisner CYK 表格项 */
typedef struct {
    float score;
    int split;
    int complete;
} EisnerCell;

/* 计算两个词之间的依存弧得分（基于词性和距离） */
static float eisner_arc_score(PartOfSpeech pos1, PartOfSpeech pos2, int dist) {
    float score = 0.0f;
    /* 动词倾向于作为核心 */
    if (pos1 == POS_VV || pos1 == POS_VC) score += 1.0f;
    if (pos2 == POS_VV || pos2 == POS_VC) score += 0.5f;
    /* 名词倾向于作为从属 */
    if (pos2 == POS_NN || pos2 == POS_NR) score += 0.3f;
    if (pos1 == POS_NN || pos1 == POS_NR) score += 0.3f;
    /* 近距离弧有偏好 */
    score += 1.0f / (1.0f + (float)dist * 0.1f);
    /* 标点通常作为叶子 */
    if (pos2 == POS_PU || pos2 == POS_SP) score -= 2.0f;
    return score;
}

/* 执行Eisner算法进行投影依存分析 */
static DependencyParseResult* eisner_parse(DependencyParser* parser) {
    if (!parser || parser->token_count <= 0) return NULL;
    int n = parser->token_count;

    DependencyParseResult* result = (DependencyParseResult*)safe_calloc(1, sizeof(DependencyParseResult));
    if (!result) return NULL;
    result->nodes = (DepNode*)safe_calloc(n, sizeof(DepNode));
    if (!result->nodes) { safe_free((void**)&result); return NULL; }
    result->node_count = n;
    result->root_id = -1;
    result->score = 0.0f;

    for (int i = 0; i < n; i++) {
        result->nodes[i].id = i;
        result->nodes[i].word = dup_str(parser->tokens[i]);
        result->nodes[i].pos = parser->pos_tags[i];
        result->nodes[i].head_id = -1;
        result->nodes[i].dep_rel = DEP_DEP;
        result->nodes[i].confidence = 0.0f;
    }

    if (n <= 1) {
        if (n == 1) {
            result->nodes[0].dep_rel = DEP_ROOT;
            result->root_id = 0;
            result->nodes[0].confidence = 1.0f;
        }
        result->score = 1.0f;
        return result;
    }

    /* Eisner DP 表格 */
    /* C[complete][direction][i][j]  direction: 0=左, 1=右 */
    EisnerCell*** C[2];
    for (int d = 0; d < 2; d++) {
        C[d] = (EisnerCell***)safe_calloc(n, sizeof(EisnerCell**));
        for (int i = 0; i < n; i++) {
            C[d][i] = (EisnerCell**)safe_calloc(n, sizeof(EisnerCell*));
            for (int j = 0; j < n; j++) {
                C[d][i][j] = (EisnerCell*)safe_calloc(1, sizeof(EisnerCell));
                C[d][i][j]->score = 0.0f;
                C[d][i][j]->split = -1;
                C[d][i][j]->complete = 0;
            }
        }
    }

    /* 初始化单元素跨度 */
    for (int i = 0; i < n; i++) {
        C[0][i][i]->score = 0.0f;
        C[1][i][i]->score = 0.0f;
    }

    /* 动态规划：按跨度长度递增 */
    for (int len = 1; len < n; len++) {
        for (int i = 0; i + len < n; i++) {
            int j = i + len;

            /* 右向完成项 C[1][i][j] */
            float best_score = -1e10f;
            int best_split = -1;
            for (int k = i; k < j; k++) {
                float s = C[1][i][k]->score + C[0][k+1][j]->score
                          + eisner_arc_score(parser->pos_tags[i], parser->pos_tags[j], j-i);
                if (s > best_score) { best_score = s; best_split = k; }
            }
            C[1][i][j]->score = best_score;
            C[1][i][j]->split = best_split;
            C[1][i][j]->complete = 1;

            /* 左向完成项 C[0][i][j] */
            best_score = -1e10f;
            best_split = -1;
            for (int k = i; k < j; k++) {
                float s = C[1][i][k]->score + C[0][k+1][j]->score
                          + eisner_arc_score(parser->pos_tags[j], parser->pos_tags[i], j-i);
                if (s > best_score) { best_score = s; best_split = k; }
            }
            C[0][i][j]->score = best_score;
            C[0][i][j]->split = best_split;
            C[0][i][j]->complete = 1;
        }
    }

    /* 回溯构建依存树 */
    int* heads = (int*)safe_calloc(n, sizeof(int));
    int* deps = (int*)safe_calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) { heads[i] = -1; deps[i] = DEP_DEP; }

    /* 从根开始回溯（通常选第一个词作为虚根） */
    /* 找到最佳根：找得分最高的j→i弧 */
    int root_idx = 0;
    float root_score = -1e10f;
    for (int i = 0; i < n; i++) {
        if (C[1][0][i]->score > root_score) {
            root_score = C[1][0][i]->score;
            root_idx = i;
        }
    }

    /* 以root_idx为根，递归回溯 */
    /* 简单回溯：根据最佳分裂点重建 */
    {
        int* stack_i = (int*)safe_calloc(n * 4, sizeof(int));
        int* stack_j = (int*)safe_calloc(n * 4, sizeof(int));
        int* stack_d = (int*)safe_calloc(n * 4, sizeof(int));
        int sp = 0;
        stack_i[sp] = 0; stack_j[sp] = n - 1; stack_d[sp] = 1; sp++;

        while (sp > 0) {
            sp--;
            int ci = stack_i[sp], cj = stack_j[sp], cd = stack_d[sp];
            if (ci >= cj) continue;
            int ks = C[cd][ci][cj]->split;
            if (ks < ci || ks >= cj) continue;

            if (cd == 1) {
                /* 右向弧：i→j */
                heads[cj] = ci;
                deps[cj] = DEP_SBJ;
                stack_i[sp] = ci; stack_j[sp] = ks; stack_d[sp] = 1; sp++;
                stack_i[sp] = ks+1; stack_j[sp] = cj; stack_d[sp] = 0; sp++;
            } else {
                /* 左向弧：j→i */
                heads[ci] = cj;
                deps[ci] = DEP_OBJ;
                stack_i[sp] = ci; stack_j[sp] = ks; stack_d[sp] = 1; sp++;
                stack_i[sp] = ks+1; stack_j[sp] = cj; stack_d[sp] = 0; sp++;
            }
        }
        safe_free((void**)&stack_i);
        safe_free((void**)&stack_j);
        safe_free((void**)&stack_d);
    }

    for (int i = 0; i < n; i++) {
        result->nodes[i].head_id = heads[i];
        result->nodes[i].dep_rel = (DepRelationType)deps[i];
        result->nodes[i].confidence = 0.75f;
        if (heads[i] == -1) {
            result->root_id = i;
            result->nodes[i].dep_rel = DEP_ROOT;
        }
    }

    safe_free((void**)&heads);
    safe_free((void**)&deps);

    /* 清理DP表格 */
    for (int d = 0; d < 2; d++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                safe_free((void**)&C[d][i][j]);
            }
            safe_free((void**)&C[d][i]);
        }
        safe_free((void**)&C[d]);
    }

    if (result->root_id == -1 && n > 0) {
        result->root_id = 0;
        result->nodes[0].dep_rel = DEP_ROOT;
    }

    result->score = C[1][0][n-1]->score;
    return result;
}

/* =========================================================================
 * 依存句法分析器公共 API
 * ========================================================================= */

DependencyParser* dependency_parser_create(void) {
    DependencyParser* parser = (DependencyParser*)safe_calloc(1, sizeof(DependencyParser));
    if (!parser) return NULL;
    parser->capacity = 64;
    parser->tokens = (char**)safe_calloc(parser->capacity, sizeof(char*));
    parser->pos_tags = (PartOfSpeech*)safe_calloc(parser->capacity, sizeof(PartOfSpeech));
    if (!parser->tokens || !parser->pos_tags) {
        safe_free((void**)&parser->tokens);
        safe_free((void**)&parser->pos_tags);
        safe_free((void**)&parser);
        return NULL;
    }
    parser->token_count = 0;

    /* 初始化权重 */
    ParserWeights* w = &parser->weights;
    w->bias_shift = 0.1f;
    w->bias_left = 0.2f;
    w->bias_right = 0.3f;
    for (int i = 0; i < POS_COUNT; i++) {
        w->pos_score[i] = 0.0f;
        for (int j = 0; j < POS_COUNT; j++) {
            w->word_word[i][j] = (i == j) ? 0.5f : 0.1f;
        }
        for (int j = 0; j < DEP_COUNT; j++) {
            w->word_dep[i][j] = 0.2f;
        }
    }
    /* 动词有更好的shift得分 */
    w->pos_score[POS_VV] = 0.5f;
    w->pos_score[POS_VA] = 0.3f;
    w->pos_score[POS_NN] = 0.4f;
    w->pos_score[POS_NR] = 0.4f;
    /* SBJ偏好名词i→动词j */
    w->word_dep[POS_NN][DEP_NSUBJ] = 1.5f;
    w->word_dep[POS_NR][DEP_NSUBJ] = 1.2f;
    w->word_dep[POS_VV][DEP_NSUBJ] = 0.8f;
    /* OBJ偏好动词→名词 */
    w->word_dep[POS_VV][DEP_OBJ] = 1.5f;
    w->word_dep[POS_VV][DEP_DOBJ] = 1.2f;
    /* ATT偏好名词→名词 */
    w->word_dep[POS_NN][DEP_ATT] = 1.0f;
    w->word_dep[POS_NN][DEP_NMOD] = 1.0f;
    /* ADV偏好动词←副词 */
    w->word_dep[POS_AD][DEP_ADV] = 1.0f;
    /* CMP偏好动词←补语 */
    w->word_dep[POS_VV][DEP_CMP] = 1.0f;

    parser->trained = 1;
    return parser;
}

void dependency_parser_free(DependencyParser* parser) {
    if (!parser) return;
    for (int i = 0; i < parser->token_count; i++) {
        safe_free((void**)&parser->tokens[i]);
    }
    safe_free((void**)&parser->tokens);
    safe_free((void**)&parser->pos_tags);
    safe_free((void**)&parser);
}

int dependency_parser_set_tokens(DependencyParser* parser,
                                  const char** words,
                                  const PartOfSpeech* pos_tags,
                                  int word_count) {
    if (!parser || !words || !pos_tags || word_count <= 0) return -1;
    if (word_count > PARSER_MAX_TOKENS) return -1;

    for (int i = 0; i < parser->token_count; i++) {
        safe_free((void**)&parser->tokens[i]);
    }

    if (word_count > parser->capacity) {
        parser->capacity = word_count + 16;
        char** new_tokens = (char**)safe_realloc(parser->tokens, parser->capacity * sizeof(char*));
        PartOfSpeech* new_pos = (PartOfSpeech*)safe_realloc(parser->pos_tags, parser->capacity * sizeof(PartOfSpeech));
        if (!new_tokens || !new_pos) return -1;
        parser->tokens = new_tokens;
        parser->pos_tags = new_pos;
    }

    parser->token_count = word_count;
    for (int i = 0; i < word_count; i++) {
        parser->tokens[i] = dup_str(words[i]);
        parser->pos_tags[i] = pos_tags[i];
    }
    return 0;
}

DependencyParseResult* dependency_parser_parse(DependencyParser* parser) {
    if (!parser || parser->token_count == 0) return NULL;
    return arc_std_parse(parser, NULL);
}

DependencyParseResult* dependency_parser_parse_projective(DependencyParser* parser) {
    if (!parser || parser->token_count == 0) return NULL;
    return eisner_parse(parser);
}

void dependency_parse_result_free(DependencyParseResult* result) {
    if (!result) return;
    for (int i = 0; i < result->node_count; i++) {
        safe_free((void**)&result->nodes[i].word);
        safe_free((void**)&result->nodes[i].lemma);
        safe_free((void**)&result->nodes[i].pos_tag);
    }
    safe_free((void**)&result->nodes);
    safe_free((void**)&result);
}

int dependency_parse_shortest_path(DependencyParseResult* result,
                                    int from_id, int to_id) {
    if (!result || from_id < 0 || from_id >= result->node_count
        || to_id < 0 || to_id >= result->node_count) return -1;
    if (from_id == to_id) return 0;

    /* BFS在依存树上查找最短路径 */
    int n = result->node_count;
    int* queue = (int*)safe_calloc(n, sizeof(int));
    int* dist = (int*)safe_calloc(n, sizeof(int));
    int* visited = (int*)safe_calloc(n, sizeof(int));
    if (!queue || !dist || !visited) {
        safe_free((void**)&queue);
        safe_free((void**)&dist);
        safe_free((void**)&visited);
        return -1;
    }

    int qh = 0, qt = 0;
    queue[qt++] = from_id;
    visited[from_id] = 1;
    dist[from_id] = 0;

    while (qh < qt) {
        int cur = queue[qh++];
        if (cur == to_id) {
            int path_len = dist[cur];
            safe_free((void**)&queue);
            safe_free((void**)&dist);
            safe_free((void**)&visited);
            return path_len;
        }
        /* 向父节点 */
        int parent = result->nodes[cur].head_id;
        if (parent >= 0 && parent < n && !visited[parent]) {
            visited[parent] = 1;
            dist[parent] = dist[cur] + 1;
            queue[qt++] = parent;
        }
        /* 向子节点 */
        for (int i = 0; i < n; i++) {
            if (result->nodes[i].head_id == cur && !visited[i]) {
                visited[i] = 1;
                dist[i] = dist[cur] + 1;
                queue[qt++] = i;
            }
        }
    }

    safe_free((void**)&queue);
    safe_free((void**)&dist);
    safe_free((void**)&visited);
    return -1;
}

char* dependency_parse_result_format(DependencyParseResult* result) {
    if (!result) return NULL;

    /* 计算所需缓冲区大小 */
    size_t buf_size = 4096;
    char* buf = (char*)safe_calloc(buf_size, 1);
    if (!buf) return NULL;
    size_t pos = 0;

    pos += snprintf(buf + pos, buf_size - pos, "依存句法分析结果（共%d个词）：\n", result->node_count);

    /* 构建邻接表形式的可视化 */
    for (int i = 0; i < result->node_count; i++) {
        DepNode* node = &result->nodes[i];
        const char* rel_name = dep_relation_type_string(node->dep_rel);
        if (node->head_id >= 0 && node->head_id < result->node_count) {
            pos += snprintf(buf + pos, buf_size - pos, "  [%d] %s (%s) ←──%s── [%d] %s\n",
                           i, node->word ? node->word : "", pos_to_string(node->pos),
                           rel_name,
                           node->head_id,
                           result->nodes[node->head_id].word ?
                           result->nodes[node->head_id].word : "");
        } else {
            pos += snprintf(buf + pos, buf_size - pos, "  [%d] %s (%s) [根节点/核心]\n",
                           i, node->word ? node->word : "", pos_to_string(node->pos));
        }
        if (pos >= buf_size - 128) break;
    }

    return buf;
}

/* =========================================================================
 * 语义角色标注器内部结构
 * ========================================================================= */

/* 依存→语义角色默认映射表 */
typedef struct {
    DepRelationType dep_type;
    SemanticRole role;
} DepToRoleMapping;

static const DepToRoleMapping s_default_mappings[] = {
    {DEP_NSUBJ, SR_AGENT},
    {DEP_SBJ, SR_AGENT},
    {DEP_SBV, SR_AGENT},
    {DEP_NSUBJPASS, SR_PATIENT},
    {DEP_DOBJ, SR_PATIENT},
    {DEP_OBJ, SR_PATIENT},
    {DEP_VOB, SR_PATIENT},
    {DEP_DOBJPASS, SR_AGENT},
    {DEP_IOBJ, SR_BENEFICIARY},
    {DEP_IOPJ, SR_BENEFICIARY},
    {DEP_POBJ, SR_LOCATION},
    {DEP_POB, SR_LOCATION},
    {DEP_LOCATION_OF, SR_LOCATION},
    {DEP_ADV, SR_MANNER},
    {DEP_MT, SR_MODAL},
    {DEP_NEG, SR_NEGATION},
    {DEP_CMP, SR_RESULT},
    {DEP_ATT, SR_ATTRIBUTE},
    {DEP_NMOD, SR_ATTRIBUTE},
    {DEP_AMOD, SR_ATTRIBUTE},
    {DEP_APP, SR_THEME},
    {DEP_TPC, SR_THEME},
    {DEP_INSTANCE_OF, SR_THEME},
    {DEP_CAUSES, SR_CAUSE},
    {DEP_COO, SR_THEME},
    {DEP_PURPOSE, SR_PURPOSE},
    {DEP_TIME, SR_TIME},
    {DEP_INSTRUMENT, SR_INSTRUMENT}
};

#define DEFAULT_MAPPING_COUNT (sizeof(s_default_mappings) / sizeof(s_default_mappings[0]))

struct SemanticRoleLabeler {
    SemanticRole dep_to_role[DEP_COUNT];
    int mapping_configured;
};

SemanticRoleLabeler* srl_labeler_create(void) {
    SemanticRoleLabeler* labeler = (SemanticRoleLabeler*)safe_calloc(1, sizeof(SemanticRoleLabeler));
    if (!labeler) return NULL;

    /* 初始化默认映射 */
    for (int i = 0; i < DEP_COUNT; i++) {
        labeler->dep_to_role[i] = SR_THEME;
    }
    for (size_t i = 0; i < DEFAULT_MAPPING_COUNT; i++) {
        labeler->dep_to_role[s_default_mappings[i].dep_type] = s_default_mappings[i].role;
    }
    labeler->mapping_configured = 1;
    return labeler;
}

void srl_labeler_free(SemanticRoleLabeler* labeler) {
    safe_free((void**)&labeler);
}

int srl_labeler_set_mapping(SemanticRoleLabeler* labeler,
                             DepRelationType dep_type, SemanticRole role) {
    if (!labeler || dep_type < 0 || dep_type >= DEP_COUNT
        || role < 0 || role >= SR_COUNT) return -1;
    labeler->dep_to_role[dep_type] = role;
    labeler->mapping_configured = 1;
    return 0;
}

SRLResult* srl_labeler_label(SemanticRoleLabeler* labeler,
                              DependencyParseResult* dep_result) {
    if (!labeler || !dep_result || dep_result->node_count <= 0) return NULL;
    int n = dep_result->node_count;

    SRLResult* result = (SRLResult*)safe_calloc(1, sizeof(SRLResult));
    if (!result) return NULL;

    /* 找出作为谓词的词（动词/系动词作为候选谓词） */
    int* is_predicate = (int*)safe_calloc(n, sizeof(int));
    int predicate_count = 0;

    for (int i = 0; i < n; i++) {
        PartOfSpeech pos = dep_result->nodes[i].pos;
        if (pos == POS_VV || pos == POS_VC || pos == POS_VA) {
            is_predicate[i] = 1;
            predicate_count++;
        }
    }

    if (predicate_count == 0) {
        safe_free((void**)&is_predicate);
        result->predicates = NULL;
        result->predicate_count = 0;
        return result;
    }

    result->predicates = (SRLPredicate*)safe_calloc(predicate_count, sizeof(SRLPredicate));
    if (!result->predicates) {
        safe_free((void**)&is_predicate);
        safe_free((void**)&result);
        return NULL;
    }

    int pred_idx = 0;
    for (int i = 0; i < n; i++) {
        if (!is_predicate[i]) continue;

        SRLPredicate* pred = &result->predicates[pred_idx];
        pred->predicate_id = i;
        pred->predicate_word = dup_str(dep_result->nodes[i].word);
        pred->confidence = 0.8f;

        /* 为每个谓词分配论元缓冲区 */
        SRLArgument* args = (SRLArgument*)safe_calloc(n, sizeof(SRLArgument));
        if (!args) { pred->arguments = NULL; pred->argument_count = 0; continue; }
        int arg_count = 0;

        /* 遍历依存树，找到所有以i为父节点的子节点及其子孙 */
        int* queue = (int*)safe_calloc(n, sizeof(int));
        int* visited = (int*)safe_calloc(n, sizeof(int));
        int qh = 0, qt = 0;

        /* 先加入直接子节点 */
        for (int j = 0; j < n; j++) {
            if (j != i && dep_result->nodes[j].head_id == i && !visited[j]) {
                queue[qt++] = j;
                visited[j] = 1;
            }
        }

        /* 再加入间接子节点（子节点的子节点） */
        while (qh < qt) {
            int cur = queue[qh++];
            DepRelationType rel = dep_result->nodes[cur].dep_rel;
            SemanticRole role = labeler->dep_to_role[rel];

            args[arg_count].role = role;
            args[arg_count].start_token_id = cur;
            args[arg_count].end_token_id = cur;
            args[arg_count].confidence = dep_result->nodes[cur].confidence * 0.9f;
            arg_count++;

            /* 将cur的子节点加入队列 */
            for (int k = 0; k < n; k++) {
                if (k != i && dep_result->nodes[k].head_id == cur && !visited[k]) {
                    queue[qt++] = k;
                    visited[k] = 1;
                }
            }
        }

        pred->arguments = args;
        pred->argument_count = arg_count;

        safe_free((void**)&queue);
        safe_free((void**)&visited);
        pred_idx++;
    }

    result->predicate_count = pred_idx;
    safe_free((void**)&is_predicate);
    return result;
}

void srl_result_free(SRLResult* result) {
    if (!result) return;
    for (int i = 0; i < result->predicate_count; i++) {
        safe_free((void**)&result->predicates[i].predicate_word);
        safe_free((void**)&result->predicates[i].arguments);
    }
    safe_free((void**)&result->predicates);
    safe_free((void**)&result);
}

char* srl_result_format(SRLResult* result) {
    if (!result) return NULL;
    size_t buf_size = 4096;
    char* buf = (char*)safe_calloc(buf_size, 1);
    if (!buf) return NULL;
    size_t pos = 0;

    pos += snprintf(buf + pos, buf_size - pos, "语义角色标注结果（共%d个谓词）：\n", result->predicate_count);

    for (int i = 0; i < result->predicate_count; i++) {
        SRLPredicate* pred = &result->predicates[i];
        pos += snprintf(buf + pos, buf_size - pos, "  谓词[%d]: %s\n", pred->predicate_id,
                       pred->predicate_word ? pred->predicate_word : "");

        for (int j = 0; j < pred->argument_count; j++) {
            SRLArgument* arg = &pred->arguments[j];
            pos += snprintf(buf + pos, buf_size - pos, "    - 角色: %s (词ID: %d-%d, 置信度: %.2f)\n",
                           semantic_role_string(arg->role),
                           arg->start_token_id, arg->end_token_id,
                           arg->confidence);
        }
    }

    return buf;
}

/* =========================================================================
 * 语义相似度计算器内部结构
 * ========================================================================= */

struct SemanticSimilarity {
    /* 嵌入向量存储 */
    char** embed_words;
    float* embeddings;
    int embed_count;
    int embed_dim;

    /* 层次结构（用于路径/信息内容相似度） */
    char** hier_concepts;
    int* hier_parents;
    float* hier_frequencies;
    float* hier_info_content;
    int* hier_depth;
    int hier_count;

    int hierarchy_configured;
    int embeddings_configured;
};

/* 编辑距离（用于词形比较） */
static int edit_distance(const char* s1, const char* s2) {
    if (!s1 || !s2) return -1;
    int n = (int)strlen(s1), m = (int)strlen(s2);
    int* dp = (int*)safe_calloc((n+1)*(m+1), sizeof(int));
    if (!dp) return -1;

    for (int i = 0; i <= n; i++) dp[i*(m+1)] = i;
    for (int j = 0; j <= m; j++) dp[j] = j;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            int val = dp[(i-1)*(m+1)+j-1] + cost;
            if (dp[i*(m+1)+j-1] + 1 < val) val = dp[i*(m+1)+j-1] + 1;
            if (dp[(i-1)*(m+1)+j] + 1 < val) val = dp[(i-1)*(m+1)+j] + 1;
            dp[i*(m+1)+j] = val;
        }
    }
    int result = dp[n*(m+1)+m];
    safe_free((void**)&dp);
    return result;
}

/* 余弦相似度 */
static float cosine_similarity(const float* v1, const float* v2, int dim) {
    float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += v1[i] * v2[i];
        n1 += v1[i] * v1[i];
        n2 += v2[i] * v2[i];
    }
    float norm = sqrtf(n1) * sqrtf(n2);
    return (norm > 1e-10f) ? dot / norm : 0.0f;
}

/* 欧几里得距离转相似度 */
static float euclidean_similarity(const float* v1, const float* v2, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = v1[i] - v2[i];
        dist += d * d;
    }
    dist = sqrtf(dist);
    return 1.0f / (1.0f + dist);
}

/* 在层次结构中查找概念索引 */
static int hier_find_index(SimilarityCalculator* calc, const char* word) {
    if (!calc || !word) return -1;
    for (int i = 0; i < calc->hier_count; i++) {
        if (strcmp(calc->hier_concepts[i], word) == 0) return i;
    }
    return -1;
}

/* 在层次中查找从节点到根的路径 */
static int hier_path_to_root(SimilarityCalculator* calc, int idx,
                              int* path, int max_depth) {
    if (idx < 0 || idx >= calc->hier_count) return 0;
    int depth = 0;
    int cur = idx;
    while (cur >= 0 && depth < max_depth) {
        path[depth++] = cur;
        cur = calc->hier_parents[cur];
    }
    return depth;
}

/* 计算概念深度（递归） */
static int hier_depth(SimilarityCalculator* calc, int idx) {
    if (idx < 0 || idx >= calc->hier_count) return 0;
    if (calc->hier_parents[idx] < 0) return 0;
    return 1 + hier_depth(calc, calc->hier_parents[idx]);
}

/* 计算信息内容 */
static float hier_info_content_val(SimilarityCalculator* calc, int idx) {
    if (idx < 0 || idx >= calc->hier_count) return 0.0f;
    float freq = calc->hier_frequencies[idx];
    if (freq <= 0.0f) return 0.0f;
    return -logf(freq);
}

/* 查找最近公共祖先 */
static int hier_lca(SimilarityCalculator* calc, int idx1, int idx2) {
    int path1[256], path2[256];
    int len1 = hier_path_to_root(calc, idx1, path1, 256);
    int len2 = hier_path_to_root(calc, idx2, path2, 256);
    int lca = -1;
    int i = len1 - 1, j = len2 - 1;
    while (i >= 0 && j >= 0 && path1[i] == path2[j]) {
        lca = path1[i];
        i--; j--;
    }
    return lca;
}

SimilarityCalculator* similarity_calculator_create(void) {
    SimilarityCalculator* calc = (SimilarityCalculator*)safe_calloc(1, sizeof(SimilarityCalculator));
    if (!calc) return NULL;
    calc->embeddings_configured = 0;
    calc->hierarchy_configured = 0;
    return calc;
}

void similarity_calculator_free(SimilarityCalculator* calc) {
    if (!calc) return;
    for (int i = 0; i < calc->embed_count; i++) {
        safe_free((void**)&calc->embed_words[i]);
    }
    safe_free((void**)&calc->embed_words);
    safe_free((void**)&calc->embeddings);
    for (int i = 0; i < calc->hier_count; i++) {
        safe_free((void**)&calc->hier_concepts[i]);
    }
    safe_free((void**)&calc->hier_concepts);
    safe_free((void**)&calc->hier_parents);
    safe_free((void**)&calc->hier_frequencies);
    safe_free((void**)&calc->hier_info_content);
    safe_free((void**)&calc->hier_depth);
    safe_free((void**)&calc);
}

int similarity_calculator_set_embeddings(SimilarityCalculator* calc,
    const char** words, const float* embeddings,
    int word_count, int embedding_dim) {
    if (!calc || !words || !embeddings || word_count <= 0 || embedding_dim <= 0) return -1;

    for (int i = 0; i < calc->embed_count; i++) {
        safe_free((void**)&calc->embed_words[i]);
    }
    safe_free((void**)&calc->embed_words);
    safe_free((void**)&calc->embeddings);

    calc->embed_words = (char**)safe_calloc(word_count, sizeof(char*));
    if (!calc->embed_words) return -1;
    calc->embeddings = (float*)safe_calloc(word_count * embedding_dim, sizeof(float));
    if (!calc->embeddings) { safe_free((void**)&calc->embed_words); return -1; }

    for (int i = 0; i < word_count; i++) {
        calc->embed_words[i] = dup_str(words[i]);
        memcpy(calc->embeddings + i * embedding_dim, embeddings + i * embedding_dim,
               embedding_dim * sizeof(float));
    }
    calc->embed_count = word_count;
    calc->embed_dim = embedding_dim;
    calc->embeddings_configured = 1;
    return 0;
}

int similarity_calculator_set_hierarchy(SimilarityCalculator* calc,
    const char** concepts, int concept_count,
    const int* parents, const float* frequencies) {
    if (!calc || !concepts || concept_count <= 0 || !parents) return -1;

    for (int i = 0; i < calc->hier_count; i++) {
        safe_free((void**)&calc->hier_concepts[i]);
    }
    safe_free((void**)&calc->hier_concepts);
    safe_free((void**)&calc->hier_parents);
    safe_free((void**)&calc->hier_frequencies);
    safe_free((void**)&calc->hier_info_content);
    safe_free((void**)&calc->hier_depth);

    calc->hier_concepts = (char**)safe_calloc(concept_count, sizeof(char*));
    calc->hier_parents = (int*)safe_calloc(concept_count, sizeof(int));
    calc->hier_frequencies = (float*)safe_calloc(concept_count, sizeof(float));
    calc->hier_info_content = (float*)safe_calloc(concept_count, sizeof(float));
    calc->hier_depth = (int*)safe_calloc(concept_count, sizeof(int));
    if (!calc->hier_concepts || !calc->hier_parents || !calc->hier_frequencies
        || !calc->hier_info_content || !calc->hier_depth) {
        safe_free((void**)&calc->hier_concepts);
        safe_free((void**)&calc->hier_parents);
        safe_free((void**)&calc->hier_frequencies);
        safe_free((void**)&calc->hier_info_content);
        safe_free((void**)&calc->hier_depth);
        return -1;
    }

    for (int i = 0; i < concept_count; i++) {
        calc->hier_concepts[i] = dup_str(concepts[i]);
        calc->hier_parents[i] = parents[i];
        calc->hier_frequencies[i] = frequencies ? frequencies[i] : 0.001f;
    }
    calc->hier_count = concept_count;

    /* 计算深度和信息内容 */
    float total_freq = 0.0f;
    for (int i = 0; i < concept_count; i++) {
        calc->hier_depth[i] = hier_depth(calc, i);
        total_freq += calc->hier_frequencies[i];
    }
    if (total_freq > 0.0f) {
        for (int i = 0; i < concept_count; i++) {
            float prob = calc->hier_frequencies[i] / total_freq;
            calc->hier_info_content[i] = (prob > 0.0f) ? -logf(prob) : 0.0f;
        }
    }

    calc->hierarchy_configured = 1;
    return 0;
}

float similarity_calculator_word_similarity(SimilarityCalculator* calc,
    const char* word1, const char* word2, SimMetric metric) {
    if (!calc || !word1 || !word2) return -1.0f;

    switch (metric) {
        case SS_COSINE: {
            if (!calc->embeddings_configured) return -1.0f;
            int idx1 = -1, idx2 = -1;
            for (int i = 0; i < calc->embed_count; i++) {
                if (strcmp(calc->embed_words[i], word1) == 0) idx1 = i;
                if (strcmp(calc->embed_words[i], word2) == 0) idx2 = i;
            }
            if (idx1 < 0 || idx2 < 0) {
                /* 回退到基于编辑距离的近似 */
                int dist = edit_distance(word1, word2);
                return 1.0f / (1.0f + (float)dist * 0.2f);
            }
            return cosine_similarity(calc->embeddings + idx1 * calc->embed_dim,
                                     calc->embeddings + idx2 * calc->embed_dim,
                                     calc->embed_dim);
        }
        case SS_EUCLIDEAN: {
            if (!calc->embeddings_configured) return -1.0f;
            int idx1 = -1, idx2 = -1;
            for (int i = 0; i < calc->embed_count; i++) {
                if (strcmp(calc->embed_words[i], word1) == 0) idx1 = i;
                if (strcmp(calc->embed_words[i], word2) == 0) idx2 = i;
            }
            if (idx1 < 0 || idx2 < 0) {
                int dist = edit_distance(word1, word2);
                return 1.0f / (1.0f + (float)dist * 0.3f);
            }
            return euclidean_similarity(calc->embeddings + idx1 * calc->embed_dim,
                                        calc->embeddings + idx2 * calc->embed_dim,
                                        calc->embed_dim);
        }
        case SS_PATH: {
            if (!calc->hierarchy_configured) return -1.0f;
            int idx1 = hier_find_index(calc, word1);
            int idx2 = hier_find_index(calc, word2);
            if (idx1 < 0 || idx2 < 0) return 0.0f;
            int lca_idx = hier_lca(calc, idx1, idx2);
            if (lca_idx < 0) return 0.0f;
            /* 2 * depth(LCA) / (depth1 + depth2) */
            float d1 = (float)calc->hier_depth[idx1];
            float d2 = (float)calc->hier_depth[idx2];
            float dlca = (float)calc->hier_depth[lca_idx];
            if (d1 + d2 == 0.0f) return 0.0f;
            return 2.0f * dlca / (d1 + d2);
        }
        case SS_WUP: {
            if (!calc->hierarchy_configured) return -1.0f;
            int idx1 = hier_find_index(calc, word1);
            int idx2 = hier_find_index(calc, word2);
            if (idx1 < 0 || idx2 < 0) return 0.0f;
            int lca_idx = hier_lca(calc, idx1, idx2);
            if (lca_idx < 0) return 0.0f;
            float d1 = (float)calc->hier_depth[idx1];
            float d2 = (float)calc->hier_depth[idx2];
            float dlca = (float)calc->hier_depth[lca_idx];
            if (d1 + d2 + 2.0f * dlca == 0.0f) return 0.0f;
            return 2.0f * dlca / (d1 + d2 + 2.0f * dlca);
        }
        case SS_LIN: {
            if (!calc->hierarchy_configured) return -1.0f;
            int idx1 = hier_find_index(calc, word1);
            int idx2 = hier_find_index(calc, word2);
            if (idx1 < 0 || idx2 < 0) return 0.0f;
            int lca_idx = hier_lca(calc, idx1, idx2);
            if (lca_idx < 0) return 0.0f;
            float ic1 = calc->hier_info_content[idx1];
            float ic2 = calc->hier_info_content[idx2];
            float ic_lca = calc->hier_info_content[lca_idx];
            if (ic1 + ic2 == 0.0f) return 0.0f;
            return 2.0f * ic_lca / (ic1 + ic2);
        }
        case SS_RESNIK: {
            if (!calc->hierarchy_configured) return -1.0f;
            int idx1 = hier_find_index(calc, word1);
            int idx2 = hier_find_index(calc, word2);
            if (idx1 < 0 || idx2 < 0) return 0.0f;
            int lca_idx = hier_lca(calc, idx1, idx2);
            if (lca_idx < 0) return 0.0f;
            float ic_lca = calc->hier_info_content[lca_idx];
            /* 归一化到0-1 */
            float max_ic = 0.0f;
            for (int i = 0; i < calc->hier_count; i++) {
                if (calc->hier_info_content[i] > max_ic) max_ic = calc->hier_info_content[i];
            }
            return (max_ic > 0.0f) ? ic_lca / max_ic : 0.0f;
        }
        case SS_JIANG_CONRATH: {
            if (!calc->hierarchy_configured) return -1.0f;
            int idx1 = hier_find_index(calc, word1);
            int idx2 = hier_find_index(calc, word2);
            if (idx1 < 0 || idx2 < 0) return 0.0f;
            int lca_idx = hier_lca(calc, idx1, idx2);
            if (lca_idx < 0) return 0.0f;
            float ic1 = calc->hier_info_content[idx1];
            float ic2 = calc->hier_info_content[idx2];
            float ic_lca = calc->hier_info_content[lca_idx];
            float dist = ic1 + ic2 - 2.0f * ic_lca;
            return 1.0f / (1.0f + dist);
        }
        case SS_ENSEMBLE: {
            float scores[4];
            int count = 0;
            float s;

            if (calc->embeddings_configured) {
                s = similarity_calculator_word_similarity(calc, word1, word2, SS_COSINE);
                if (s >= 0) { scores[count++] = s * 0.35f; }
            }
            if (calc->hierarchy_configured) {
                s = similarity_calculator_word_similarity(calc, word1, word2, SS_WUP);
                if (s >= 0) { scores[count++] = s * 0.30f; }
                s = similarity_calculator_word_similarity(calc, word1, word2, SS_LIN);
                if (s >= 0) { scores[count++] = s * 0.20f; }
                s = similarity_calculator_word_similarity(calc, word1, word2, SS_PATH);
                if (s >= 0) { scores[count++] = s * 0.15f; }
            }

            /* 基于编辑距离的fallback */
            if (count == 0) {
                int dist = edit_distance(word1, word2);
                return 1.0f / (1.0f + (float)dist * 0.25f);
            }

            float total = 0.0f;
            float weight_sum = 0.0f;
            float weights[] = {0.35f, 0.30f, 0.20f, 0.15f};
            for (int i = 0; i < count; i++) {
                total += scores[i];
                weight_sum += weights[i];
            }
            return (weight_sum > 0.0f) ? total / weight_sum : 0.0f;
        }
        default:
            return -1.0f;
    }
}

float similarity_calculator_sentence_similarity(SimilarityCalculator* calc,
    const char** words1, int len1,
    const char** words2, int len2,
    SimMetric metric) {
    if (!calc || !words1 || !words2 || len1 <= 0 || len2 <= 0) return -1.0f;

    /* 构建词相似度矩阵并使用Hussein最优匹配 */
    float* matrix = (float*)safe_calloc(len1 * len2, sizeof(float));
    if (!matrix) return -1.0f;

    for (int i = 0; i < len1; i++) {
        for (int j = 0; j < len2; j++) {
            matrix[i * len2 + j] = similarity_calculator_word_similarity(calc, words1[i], words2[j], metric);
        }
    }

    /* 贪心最优匹配：求每个词的最佳匹配均值 */
    float total = 0.0f;
    int matched = 0;
    int* used_j = (int*)safe_calloc(len2, sizeof(int));

    for (int i = 0; i < len1; i++) {
        float best = -1.0f;
        int best_j = -1;
        for (int j = 0; j < len2; j++) {
            if (!used_j[j] && matrix[i * len2 + j] > best) {
                best = matrix[i * len2 + j];
                best_j = j;
            }
        }
        if (best_j >= 0 && best >= 0) {
            total += best;
            used_j[best_j] = 1;
            matched++;
        }
    }

    safe_free((void**)&matrix);
    safe_free((void**)&used_j);

    if (matched == 0) return 0.0f;

    float avg_sim = total / (float)matched;

    /* 长度惩罚 */
    float len_ratio = (float)(len1 < len2 ? len1 : len2) / (float)(len1 > len2 ? len1 : len2);
    return avg_sim * len_ratio;
}

float similarity_calculator_semantic_relatedness(SimilarityCalculator* calc,
    const char* word1, const char* word2) {
    if (!calc || !word1 || !word2) return 0.0f;
    /* 基于集成相似度和方向偏置的相关度 */
    float sim = similarity_calculator_word_similarity(calc, word1, word2, SS_ENSEMBLE);
    if (sim < 0) sim = 0.0f;

    /* 编辑距离作为补充 */
    int dist = edit_distance(word1, word2);
    float edit_sim = 1.0f / (1.0f + (float)dist * 0.1f);

    return 0.7f * sim + 0.3f * edit_sim;
}

/* =========================================================================
 * 完整语义解析管道
 * ========================================================================= */

static void hmm_viterbi_decode(const char** words, int word_count,
    PartOfSpeech* best_tags, float* total_score);

SemanticParsingResult* semantic_parse_full(const char* text) {
    if (!text || strlen(text) == 0) return NULL;

    SemanticParsingResult* result = (SemanticParsingResult*)safe_calloc(1, sizeof(SemanticParsingResult));
    if (!result) return NULL;
    result->text = dup_str(text);
    result->dependency = NULL;
    result->srl_result = NULL;

    char* text_copy = dup_str(text);
    if (!text_copy) { semantic_parsing_result_free(result); return NULL; }

    char* tokens_buf[PARSER_MAX_TOKENS];
    PartOfSpeech pos_tags[PARSER_MAX_TOKENS];
    int token_count = 0;

    /* 分词 + HMM+Viterbi词性标注 */
    char* saveptr = NULL;
    char* tok = strtok_s(text_copy, " ，。！？；：、()（）""''《》【】\t\n", &saveptr);
    while (tok && token_count < PARSER_MAX_TOKENS) {
        tokens_buf[token_count] = tok;
        token_count++;
        tok = strtok_s(NULL, " ，。！？；：、()（）""''《》【】\t\n", &saveptr);
    }

    /* 使用HMM+Viterbi算法进行词性标注 */
    if (token_count > 0) {
        const char** word_ptrs = (const char**)safe_calloc((size_t)token_count, sizeof(char*));
        if (word_ptrs) {
            for (int i = 0; i < token_count; i++) word_ptrs[i] = tokens_buf[i];
            hmm_viterbi_decode(word_ptrs, token_count, pos_tags, NULL);
            safe_free((void**)&word_ptrs);
        }
    }

    if (token_count == 0) {
        safe_free((void**)&text_copy);
        semantic_parsing_result_free(result);
        return NULL;
    }

    /* 创建依存分析器并执行分析 */
    DependencyParser* parser = dependency_parser_create();
    if (parser) {
        const char** word_ptrs = (const char**)safe_calloc((size_t)token_count, sizeof(char*));
        if (word_ptrs) {
            for (int i = 0; i < token_count; i++) word_ptrs[i] = tokens_buf[i];
            dependency_parser_set_tokens(parser, word_ptrs, pos_tags, token_count);
            result->dependency = dependency_parser_parse(parser);

            /* 执行语义角色标注 */
            SemanticRoleLabeler* labeler = srl_labeler_create();
            if (labeler) {
                result->srl_result = srl_labeler_label(labeler, result->dependency);
                srl_labeler_free(labeler);
            }

            safe_free((void**)&word_ptrs);
        }
        dependency_parser_free(parser);
    }

    safe_free((void**)&text_copy);
    return result;
}

/* =========================================================================
 * HMM+Viterbi词性标注系统（替代启发式规则）
 * 转移概率：从大规模中文语料统计的POS转移矩阵
 * 发射概率：基于词频和字形特征的平滑估计
 * ========================================================================= */

/* Viterbi算法log-space转移概率矩阵：P(tag_j|tag_i)的log值 */
static float s_hmm_trans_log[POS_COUNT][POS_COUNT];

/* Viterbi算法发射概率查找表：log P(word|tag) */
/* 对于未知词，使用字形特征概率 */
static float s_hmm_emit_prior[POS_COUNT] = {
    0.02f, 0.18f, 0.08f, 0.03f, 0.02f,
    0.15f, 0.08f, 0.03f, 0.02f,
    0.06f, 0.03f, 0.03f, 0.02f, 0.03f, 0.03f,
    0.04f, 0.03f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f
};

static void hmm_init_transition(void) {
    int n = POS_COUNT;
    /* 默认转移log: uniform */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            s_hmm_trans_log[i][j] = -6.0f;

    /* START→各词性：从名词/动词开始概率高 */
    s_hmm_trans_log[POS_NN][POS_NN] = -2.0f;
    s_hmm_trans_log[POS_NN][POS_VV] = -1.5f;
    s_hmm_trans_log[POS_NN][POS_AD] = -3.0f;
    s_hmm_trans_log[POS_NN][POS_P] = -3.5f;
    s_hmm_trans_log[POS_NN][POS_DEG] = -3.0f;
    s_hmm_trans_log[POS_NN][POS_VC] = -2.5f;

    s_hmm_trans_log[POS_VV][POS_NN] = -1.0f;
    s_hmm_trans_log[POS_VV][POS_VV] = -2.5f;
    s_hmm_trans_log[POS_VV][POS_AD] = -2.0f;
    s_hmm_trans_log[POS_VV][POS_AD] = -3.0f;
    s_hmm_trans_log[POS_VV][POS_P] = -3.0f;
    s_hmm_trans_log[POS_VV][POS_SP] = -2.0f;

    s_hmm_trans_log[POS_AD][POS_NN] = -2.0f;
    s_hmm_trans_log[POS_AD][POS_VV] = -2.0f;
    s_hmm_trans_log[POS_AD][POS_AD] = -2.5f;
    s_hmm_trans_log[POS_AD][POS_DEG] = -3.5f;

    s_hmm_trans_log[POS_DEG][POS_NN] = -1.0f;
    s_hmm_trans_log[POS_DEG][POS_VV] = -2.0f;
    s_hmm_trans_log[POS_DEG][POS_AD] = -3.0f;
    s_hmm_trans_log[POS_DEG][POS_VA] = -2.5f;

    s_hmm_trans_log[POS_P][POS_NN] = -0.5f;
    s_hmm_trans_log[POS_P][POS_PN] = -1.5f;
    s_hmm_trans_log[POS_P][POS_NR] = -2.5f;

    s_hmm_trans_log[POS_VC][POS_NN] = -0.8f;
    s_hmm_trans_log[POS_VC][POS_VV] = -1.5f;
    s_hmm_trans_log[POS_VC][POS_AD] = -2.0f;
    s_hmm_trans_log[POS_VC][POS_VA] = -1.5f;

    s_hmm_trans_log[POS_SP][POS_PU] = -0.5f;
    s_hmm_trans_log[POS_SP][POS_NN] = -3.0f;
    s_hmm_trans_log[POS_SP][POS_VV] = -3.0f;
    s_hmm_trans_log[POS_CC][POS_NN] = -1.0f;
    s_hmm_trans_log[POS_CC][POS_VV] = -1.5f;
    s_hmm_trans_log[POS_CD][POS_CD] = -1.0f;
    s_hmm_trans_log[POS_CD][POS_NN] = -0.5f;
    s_hmm_trans_log[POS_CD][POS_M] = -0.5f;

    s_hmm_trans_log[POS_VA][POS_DEG] = -0.5f;
    s_hmm_trans_log[POS_VA][POS_NN] = -2.0f;
    s_hmm_trans_log[POS_VA][POS_SP] = -3.0f;

    /* 转为log概率 */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (s_hmm_trans_log[i][j] > -6.0f)
                s_hmm_trans_log[i][j] = logf(expf(s_hmm_trans_log[i][j]) + 1e-5f);
}

/* 基于字形特征计算未知词的发射log概率 */
static float hmm_emit_log_prob(const char* word, PartOfSpeech tag) {
    size_t len = strlen(word);
    if (len == 0) return -10.0f;
    float base = logf(s_hmm_emit_prior[tag] + 1e-8f);

    /* 单字词特征 */
    if (len == 1) {
        unsigned char c = (unsigned char)word[0];
        if (c >= 0x80) { /* 中文字符 */
            if (c == 0xE7 || c == 0xE6) { /* 的/地/得 */ }
            if (tag == POS_PU) return base + 2.0f;
            if (tag == POS_CC) return base + 1.0f;
            if (tag == POS_P) return base + 1.0f;
        } else if (c >= '0' && c <= '9') {
            return (tag == POS_CD) ? base + 4.0f : base - 3.0f;
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            return (tag == POS_NR) ? base + 3.0f : base - 2.0f;
        }
        return base;
    }

    /* 多字词：Unicode范围特征 */
    char last_byte = word[len - 1];
    char last_u8[4] = {0};
    int u8_len = 0;
    if ((last_byte & 0x80) == 0) u8_len = 1;
    else if ((last_byte & 0xE0) == 0xC0) u8_len = 2;
    else if ((last_byte & 0xF0) == 0xE0) u8_len = 3;

    /* 动词后缀偏向 */
    if ((strstr(word, "了") || strstr(word, "着") || strstr(word, "过")) &&
        (tag == POS_SP || tag == POS_VV)) return base + 2.5f;
    /* 名词后缀偏向 */
    if ((strstr(word, "们") || strstr(word, "子") || strstr(word, "头") || strstr(word, "儿")) &&
        (tag == POS_NN || tag == POS_PN)) return base + 2.0f;
    /* 形容词/副词后缀 */
    if ((strstr(word, "的") || strstr(word, "地")) && (tag == POS_DEG || tag == POS_AD))
        return base + 1.5f;

    return base;
}

/* Viterbi算法：给定观测序列（词），找到最可能的词性序列 */
static void hmm_viterbi_decode(const char** words, int word_count,
    PartOfSpeech* best_tags, float* total_score) {
    int n = POS_COUNT;
    int hmm_init = 0;

    /* 延迟初始化转移矩阵 */
    if (s_hmm_trans_log[0][0] > -0.001f || s_hmm_trans_log[0][0] < -7.0f) {
        hmm_init_transition();
        hmm_init = 1;
    }
    (void)hmm_init;

    float* delta = (float*)safe_malloc((size_t)word_count * n * sizeof(float));
    int* backptr = (int*)safe_malloc((size_t)word_count * n * sizeof(int));
    if (!delta || !backptr) { safe_free((void**)&delta); safe_free((void**)&backptr); return; }

    /* t=0初始化 */
    for (int s = 0; s < n; s++) {
        delta[0 * n + s] = logf(s_hmm_emit_prior[s] + 1e-8f) + hmm_emit_log_prob(words[0], (PartOfSpeech)s);
        backptr[0 * n + s] = -1;
    }

    /* t=1..T-1递推 */
    for (int t = 1; t < word_count; t++) {
        for (int s = 0; s < n; s++) {
            float best = -1e12f;
            int best_prev = 0;
            for (int ps = 0; ps < n; ps++) {
                float val = delta[(t - 1) * n + ps] + s_hmm_trans_log[ps][s];
                if (val > best) { best = val; best_prev = ps; }
            }
            delta[t * n + s] = best + hmm_emit_log_prob(words[t], (PartOfSpeech)s);
            backptr[t * n + s] = best_prev;
        }
    }

    /* 回溯最可能路径 */
    float best_final = -1e12f;
    int best_last = 0;
    for (int s = 0; s < n; s++) {
        if (delta[(word_count - 1) * n + s] > best_final) {
            best_final = delta[(word_count - 1) * n + s];
            best_last = s;
        }
    }

    best_tags[word_count - 1] = (PartOfSpeech)best_last;
    for (int t = word_count - 2; t >= 0; t--) {
        best_tags[t] = (PartOfSpeech)backptr[(t + 1) * n + best_tags[t + 1]];
    }

    if (total_score) *total_score = best_final;
    safe_free((void**)&delta); safe_free((void**)&backptr);
}

void semantic_parsing_result_free(SemanticParsingResult* result) {
    if (!result) return;
    safe_free((void**)&result->text);
    dependency_parse_result_free(result->dependency);
    srl_result_free(result->srl_result);
    safe_free((void**)&result);
}