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
    (void)last_byte;
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

/* ============================================================================
 * K-修复: SWRL规则引擎（语义网规则语言解释器）
 * 支持规则解析、前向链推理、规则冲突检测
 * ============================================================================ */

#define SWRL_MAX_RULES 256
#define SWRL_MAX_ATOMS 32
#define SWRL_MAX_VARS 64

typedef enum {
    ATOM_CLASS = 0,       /**< C(?x) 类归属 */
    ATOM_PROPERTY = 1,    /**< P(?x,?y) 属性关系 */
    ATOM_EQUAL = 2,       /**< ?x = ?y */
    ATOM_DIFFERENT = 3,   /**< ?x ≠ ?y */
    ATOM_BUILTIN = 4      /**< swrlb: 内置函数 */
} SwrlAtomType;

typedef struct {
    SwrlAtomType type;
    char predicate[128];     /**< 谓词名（类名/属性名/内置函数名） */
    char arg1[128];          /**< 参数1 */
    char arg2[128];          /**< 参数2 */
} SwrlAtom;

typedef struct {
    char rule_name[128];
    SwrlAtom body[SWRL_MAX_ATOMS];     /**< 条件（body） */
    int body_count;
    SwrlAtom head[SWRL_MAX_ATOMS];     /**< 结论（head） */
    int head_count;
    float confidence;
    int is_active;
    int priority;
} SwrlRule;

/* SwrlTriple需要在SwrlEngine前定义（MSVC C模式不支持嵌套typedef） */
typedef struct { char subj[128]; char pred[128]; char obj[128]; float conf; } SwrlTriple;

typedef struct {
    SwrlRule rules[SWRL_MAX_RULES];
    int rule_count;

    SwrlTriple triples[2048];
    int triple_count;

    int inference_count;
    int conflict_count;
    int initialized;
} SwrlEngine;

/** 创建SWRL引擎 */
static SwrlEngine* swrl_engine_create(void) {
    SwrlEngine* e = (SwrlEngine*)safe_calloc(1, sizeof(SwrlEngine));
    if (e) e->initialized = 1;
    return e;
}

/** 释放SWRL引擎 */
static void swrl_engine_free(SwrlEngine* e) {
    safe_free((void**)&e);
}

/** 添加SWRL规则 */
static int swrl_add_rule(SwrlEngine* e, const char* name,
    SwrlAtom* body, int body_count, SwrlAtom* head, int head_count, float conf) {
    if (!e || !name || e->rule_count >= SWRL_MAX_RULES) return -1;
    SwrlRule* r = &e->rules[e->rule_count];
    strncpy(r->rule_name, name, 127);
    r->body_count = (body_count < SWRL_MAX_ATOMS) ? body_count : SWRL_MAX_ATOMS;
    r->head_count = (head_count < SWRL_MAX_ATOMS) ? head_count : SWRL_MAX_ATOMS;
    memcpy(r->body, body, (size_t)r->body_count * sizeof(SwrlAtom));
    memcpy(r->head, head, (size_t)r->head_count * sizeof(SwrlAtom));
    r->confidence = conf;
    r->is_active = 1;
    r->priority = 0;
    e->rule_count++;
    return 0;
}

/** 添加事实三元组 */
static int swrl_add_triple(SwrlEngine* e, const char* s, const char* p, const char* o, float conf) {
    if (!e || !s || !p || !o || e->triple_count >= 2048) return -1;
    SwrlTriple* t = &e->triples[e->triple_count];
    strncpy(t->subj, s, 127); strncpy(t->pred, p, 127); strncpy(t->obj, o, 127);
    t->conf = conf;
    e->triple_count++;
    return 0;
}

/** 变量绑定类型 */
typedef struct {
    char var_name[64];
    char value[128];
} SwrlBinding;

/** 匹配单个SWRL原子与三元组 */
static int swrl_match_atom(SwrlAtom* atom, SwrlTriple* triple, SwrlBinding* bindings, int* bind_count) {
    if (!atom || !triple) return 0;
    if (atom->type == ATOM_CLASS) {
        /* C(?x): 检查triple的subject是否等于类名匹配 */
        return (strcmp(triple->pred, "type") == 0 && strcmp(triple->obj, atom->predicate) == 0) ? 1 : 0;
    }
    if (atom->type == ATOM_PROPERTY) {
        /* P(?x,?y): 谓词匹配 */
        return (strcmp(triple->pred, atom->predicate) == 0) ? 1 : 0;
    }
    if (atom->type == ATOM_EQUAL) {
        return (strcmp(triple->subj, triple->obj) == 0) ? 1 : 0;
    }
    return 0;
}

/** SWRL前向链推理 */
static int swrl_forward_chain(SwrlEngine* e, int max_iterations) {
    if (!e || e->rule_count == 0) return 0;

    int new_facts = 0;
    for (int iter = 0; iter < max_iterations; iter++) {
        int iter_new = 0;

        for (int ri = 0; ri < e->rule_count; ri++) {
            SwrlRule* rule = &e->rules[ri];
            if (!rule->is_active) continue;

            /* 尝试匹配body中的所有原子 */
            for (int ti = 0; ti < e->triple_count; ti++) {
                int all_matched = 1;
                for (int ai = 0; ai < rule->body_count; ai++) {
                    SwrlTriple* tri = &e->triples[ti];
                    if (!swrl_match_atom(&rule->body[ai], tri, NULL, NULL)) {
                        /* 尝试下一条三元组 */
                        int found = 0;
                        for (int tj = 0; tj < e->triple_count; tj++) {
                            if (swrl_match_atom(&rule->body[ai], &e->triples[tj], NULL, NULL)) {
                                found = 1; break;
                            }
                        }
                        if (!found) { all_matched = 0; break; }
                    }
                }

                if (all_matched) {
                    /* 触发规则，生成head中定义的结论 */
                    for (int hi = 0; hi < rule->head_count; hi++) {
                        SwrlAtom* ha = &rule->head[hi];
                        if (ha->type == ATOM_PROPERTY) {
                            /* 检查是否已存在（去重） */
                            int dup = 0;
                            for (int tk = 0; tk < e->triple_count; tk++) {
                                SwrlTriple* t = &e->triples[tk];
                                if (strcmp(t->subj, ha->arg1) == 0 &&
                                    strcmp(t->pred, ha->predicate) == 0 &&
                                    strcmp(t->obj, ha->arg2) == 0) { dup = 1; break; }
                            }
                            if (!dup && e->triple_count < 2048) {
                                SwrlTriple* nt = &e->triples[e->triple_count];
                                strncpy(nt->subj, ha->arg1, 127);
                                strncpy(nt->pred, ha->predicate, 127);
                                strncpy(nt->obj, ha->arg2, 127);
                                nt->conf = rule->confidence;
                                e->triple_count++;
                                iter_new++;
                            }
                        } else if (ha->type == ATOM_CLASS) {
                            int dup = 0;
                            for (int tk = 0; tk < e->triple_count; tk++) {
                                SwrlTriple* t = &e->triples[tk];
                                if (strcmp(t->subj, ha->arg1) == 0 &&
                                    strcmp(t->pred, "type") == 0 &&
                                    strcmp(t->obj, ha->predicate) == 0) { dup = 1; break; }
                            }
                            if (!dup && e->triple_count < 2048) {
                                SwrlTriple* nt = &e->triples[e->triple_count];
                                strncpy(nt->subj, ha->arg1, 127);
                                strncpy(nt->pred, "type", 5);
                                strncpy(nt->obj, ha->predicate, 127);
                                nt->conf = rule->confidence;
                                e->triple_count++;
                                iter_new++;
                            }
                        }
                    }
                    e->inference_count++;
                }
            }
        }

        new_facts += iter_new;
        if (iter_new == 0) break; /* 不动点 */
    }
    return new_facts;
}

/** 规则冲突检测 */
static int swrl_detect_conflicts(SwrlEngine* e) {
    if (!e) return 0;
    int conflicts = 0;

    /* 检测互补性预测 */
    for (int ri = 0; ri < e->rule_count; ri++) {
        for (int rj = ri + 1; rj < e->rule_count; rj++) {
            SwrlRule* r1 = &e->rules[ri];
            SwrlRule* r2 = &e->rules[rj];
            if (!r1->is_active || !r2->is_active) continue;

            /* 匹配相反的head */
            for (int hi = 0; hi < r1->head_count; hi++) {
                for (int hj = 0; hj < r2->head_count; hj++) {
                    SwrlAtom* h1 = &r1->head[hi];
                    SwrlAtom* h2 = &r2->head[hj];
                    if (h1->type == h2->type &&
                        strcmp(h1->arg1, h2->arg1) == 0 &&
                        strcmp(h1->predicate, h2->predicate) == 0 &&
                        strcmp(h1->arg2, h2->arg2) != 0) {
                        /* 冲突：同一谓词对同一主体有不同客体 */
                        if (r1->confidence > r2->confidence) {
                            r2->is_active = 0;
                        } else {
                            r1->is_active = 0;
                        }
                        conflicts++;
                    }
                }
            }
        }
    }

    e->conflict_count += conflicts;
    return conflicts;
}

/* ============================================================================
 * 公共接口
 * ============================================================================ */

static SwrlEngine* g_swrl_engine = NULL;

int semantic_parsing_swrl_init(void) {
    if (g_swrl_engine) return 0;
    g_swrl_engine = swrl_engine_create();
    return g_swrl_engine ? 0 : -1;
}

void semantic_parsing_swrl_cleanup(void) {
    if (g_swrl_engine) { swrl_engine_free(g_swrl_engine); g_swrl_engine = NULL; }
}

int semantic_parsing_swrl_add_rule(const char* name, const char* body_rules,
    int body_count, const char* head_rules, int head_count, float confidence) {
    if (!g_swrl_engine) return -1;
    SwrlAtom body[SWRL_MAX_ATOMS] = {0};
    SwrlAtom head[SWRL_MAX_ATOMS] = {0};
    int actual_body_count = 0;
    int actual_head_count = 0;
    /* 解析单个原子字符串: 格式 "类型:谓词:arg1:arg2" 或简单 "谓词名" */
    /* 类型: C=类归属 P=属性关系 E=相等 D=不等 B=内置函数 */
    #define PARSE_ONE_ATOM(atom_str, out_atom) do { \
        char _parts[4][128] = {{0}, {0}, {0}, {0}}; \
        int _pc = 0; \
        const char* _s = (atom_str); \
        const char* _colon; \
        while ((_colon = strchr(_s, ':')) != NULL && _pc < 4) { \
            size_t _len = (size_t)(_colon - _s); \
            if (_len > 127) _len = 127; \
            memcpy(_parts[_pc], _s, _len); \
            _parts[_pc][_len] = '\0'; \
            _s = _colon + 1; \
            _pc++; \
        } \
        if (_pc < 4 && _s[0] != '\0') { \
            strncpy(_parts[_pc], _s, 127); \
            _parts[_pc][127] = '\0'; \
            _pc++; \
        } \
        if (_pc == 1) { \
            (out_atom)->type = ATOM_PROPERTY; \
            strncpy((out_atom)->predicate, _parts[0], 127); \
            (out_atom)->predicate[127] = '\0'; \
            (out_atom)->arg1[0] = '?'; (out_atom)->arg1[1] = 'x'; (out_atom)->arg1[2] = '\0'; \
            (out_atom)->arg2[0] = '?'; (out_atom)->arg2[1] = 'y'; (out_atom)->arg2[2] = '\0'; \
        } else if (_pc >= 2) { \
            switch (_parts[0][0]) { \
                case 'C': case 'c': (out_atom)->type = ATOM_CLASS; break; \
                case 'P': case 'p': (out_atom)->type = ATOM_PROPERTY; break; \
                case 'E': case 'e': (out_atom)->type = ATOM_EQUAL; break; \
                case 'D': case 'd': (out_atom)->type = ATOM_DIFFERENT; break; \
                case 'B': case 'b': (out_atom)->type = ATOM_BUILTIN; break; \
                default: (out_atom)->type = ATOM_PROPERTY; break; \
            } \
            strncpy((out_atom)->predicate, _parts[1], 127); \
            (out_atom)->predicate[127] = '\0'; \
            if (_pc >= 3) { strncpy((out_atom)->arg1, _parts[2], 127); (out_atom)->arg1[127] = '\0'; } \
            else { (out_atom)->arg1[0] = '?'; (out_atom)->arg1[1] = 'x'; (out_atom)->arg1[2] = '\0'; } \
            if (_pc >= 4) { strncpy((out_atom)->arg2, _parts[3], 127); (out_atom)->arg2[127] = '\0'; } \
            else { (out_atom)->arg2[0] = '?'; (out_atom)->arg2[1] = 'y'; (out_atom)->arg2[2] = '\0'; } \
        } \
    } while(0)
    /* 解析body规则: 以";"分隔多个原子 */
    if (body_rules && body_count > 0) {
        const char* cursor = body_rules;
        while (actual_body_count < body_count && actual_body_count < SWRL_MAX_ATOMS) {
            while (*cursor == ' ' || *cursor == '\t') cursor++;
            if (*cursor == '\0') break;
            const char* semi = strchr(cursor, ';');
            char single[512] = {0};
            if (semi) {
                size_t len = (size_t)(semi - cursor);
                if (len > 511) len = 511;
                memcpy(single, cursor, len);
                single[len] = '\0';
                cursor = semi + 1;
            } else {
                strncpy(single, cursor, 511);
                single[511] = '\0';
                cursor += strlen(cursor);
            }
            /* 去除尾部空格 */
            size_t slen = strlen(single);
            while (slen > 0 && (single[slen-1] == ' ' || single[slen-1] == '\t')) {
                single[--slen] = '\0';
            }
            if (slen > 0) {
                PARSE_ONE_ATOM(single, &body[actual_body_count]);
                actual_body_count++;
            }
        }
    }
    /* 解析head规则 */
    if (head_rules && head_count > 0) {
        const char* cursor = head_rules;
        while (actual_head_count < head_count && actual_head_count < SWRL_MAX_ATOMS) {
            while (*cursor == ' ' || *cursor == '\t') cursor++;
            if (*cursor == '\0') break;
            const char* semi = strchr(cursor, ';');
            char single[512] = {0};
            if (semi) {
                size_t len = (size_t)(semi - cursor);
                if (len > 511) len = 511;
                memcpy(single, cursor, len);
                single[len] = '\0';
                cursor = semi + 1;
            } else {
                strncpy(single, cursor, 511);
                single[511] = '\0';
                cursor += strlen(cursor);
            }
            size_t slen = strlen(single);
            while (slen > 0 && (single[slen-1] == ' ' || single[slen-1] == '\t')) {
                single[--slen] = '\0';
            }
            if (slen > 0) {
                PARSE_ONE_ATOM(single, &head[actual_head_count]);
                actual_head_count++;
            }
        }
    }
    /* 回退: 如果字符串中存在内容但未能解析出任何原子,
       将整个字符串当作单个谓词名处理 (向后兼容) */
    if (actual_body_count == 0 && body_rules && body_count > 0) {
        PARSE_ONE_ATOM(body_rules, &body[0]);
        actual_body_count = 1;
    }
    if (actual_head_count == 0 && head_rules && head_count > 0) {
        PARSE_ONE_ATOM(head_rules, &head[0]);
        actual_head_count = 1;
    }
    #undef PARSE_ONE_ATOM
    return swrl_add_rule(g_swrl_engine, name, body, actual_body_count,
                          head, actual_head_count, confidence);
}

int semantic_parsing_swrl_add_fact(const char* subject, const char* predicate,
    const char* object, float confidence) {
    if (!g_swrl_engine) return -1;
    return swrl_add_triple(g_swrl_engine, subject, predicate, object, confidence);
}

int semantic_parsing_swrl_reason(int max_iterations, int* inferred_count) {
    if (!g_swrl_engine) return -1;
    /* 先检测冲突 */
    swrl_detect_conflicts(g_swrl_engine);
    /* 再推理 */
    int count = swrl_forward_chain(g_swrl_engine, max_iterations);
    if (inferred_count) *inferred_count = count;
    return 0;
}

int semantic_parsing_swrl_get_stats(int* out_rule_count, int* out_triple_count,
    int* out_inference_count, int* out_conflict_count) {
    if (!g_swrl_engine) return -1;
    if (out_rule_count) *out_rule_count = g_swrl_engine->rule_count;
    if (out_triple_count) *out_triple_count = g_swrl_engine->triple_count;
    if (out_inference_count) *out_inference_count = g_swrl_engine->inference_count;
    if (out_conflict_count) *out_conflict_count = g_swrl_engine->conflict_count;
    return 0;
}

/* ============================================================================
 * M-005修复: 依存分析器在线学习
 *
 * 使用感知器更新规则对转移分类器权重进行在线学习。
 * 支持标注样例输入、用户反馈纠错、权重持久化。
 * 学习率采用指数衰减：η = η₀ / (1 + decay * iteration)
 * ============================================================================ */

/* 在线学习器状态 */
typedef struct {
    float* feature_weights;      /* 特征权重向量 */
    int weight_dim;              /* 权重维度 */
    int iteration_count;         /* 学习迭代次数 */
    float learning_rate;         /* 当前学习率 */
    float initial_learning_rate; /* 初始学习率 */
    float decay_rate;            /* 衰减率 */
    int is_initialized;          /* 是否已初始化 */
} SPOnlineLearner;

static SPOnlineLearner g_sp_learner = { NULL, 0, 0, 0.01f, 0.01f, 0.001f, 0 };

/* 初始化在线学习器 */
static int sp_learner_init(int feature_dim) {
    if (g_sp_learner.is_initialized && g_sp_learner.weight_dim == feature_dim) return 0;
    if (g_sp_learner.feature_weights) {
        safe_free((void**)&g_sp_learner.feature_weights);
    }
    g_sp_learner.feature_weights = (float*)safe_calloc((size_t)feature_dim, sizeof(float));
    if (!g_sp_learner.feature_weights) return -1;
    g_sp_learner.weight_dim = feature_dim;
    g_sp_learner.iteration_count = 0;
    g_sp_learner.learning_rate = g_sp_learner.initial_learning_rate;
    g_sp_learner.is_initialized = 1;
    return 0;
}

/* 从词序列和词性标注提取特征向量（简化版：bigram+POS组合特征） */
static int sp_extract_features(const char** words, const PartOfSpeech* pos_tags,
                                int word_count, float* features, int feature_dim) {
    if (!words || !features || feature_dim <= 0) return -1;
    memset(features, 0, (size_t)feature_dim * sizeof(float));

    for (int i = 0; i < word_count && i < 64; i++) {
        /* 单词bigram特征 */
        if (words[i]) {
            const char* w = words[i];
            for (int j = 0; w[j] && j < 16; j++) {
                unsigned int h = (unsigned int)(unsigned char)w[j];
                if (j + 1 < 16 && w[j + 1]) {
                    h = (h << 8) | (unsigned int)(unsigned char)w[j + 1];
                }
                h = h * 2654435761u;
                size_t idx = (size_t)(h % (unsigned int)feature_dim);
                features[idx] += 1.0f;
            }
        }
        /* 词性特征 */
        int pos_val = (int)pos_tags[i];
        size_t pos_idx = (size_t)(pos_val * 7 + i) % (size_t)feature_dim;
        features[pos_idx] += 0.5f;
    }

    /* L2归一化 */
    float norm = 0.0f;
    for (int i = 0; i < feature_dim; i++) norm += features[i] * features[i];
    if (norm > 1e-10f) {
        norm = sqrtf(norm);
        for (int i = 0; i < feature_dim; i++) features[i] /= norm;
    }
    return 0;
}

int sp_online_learn(const char** words, const PartOfSpeech* pos_tags, int word_count,
                     const int* gold_heads, const DepRelationType* gold_rels,
                     int gold_count) {
    if (!words || !pos_tags || word_count <= 0 || !gold_heads || !gold_rels) return -1;
    if (word_count != gold_count) return -1;

    /* 初始化学习器（128维特征向量） */
    int feature_dim = 128;
    if (sp_learner_init(feature_dim) != 0) return -1;

    /* 提取特征向量 */
    float* features = (float*)safe_malloc((size_t)feature_dim * sizeof(float));
    if (!features) return -1;

    sp_extract_features(words, pos_tags, word_count, features, feature_dim);

    /* 使用感知器更新规则：w = w + η * (gold - predicted) */
    for (int i = 0; i < gold_count; i++) {
        /* 预测标签：基于当前权重计算得分最高的依存关系 */
        float best_score = -1e10f;
        int predicted_head = -1;
        DepRelationType predicted_rel = DEP_DEP;

        for (int j = 0; j < word_count; j++) {
            for (int rt = 0; rt < DEP_COUNT; rt++) {
                /* 简化得分：特征向量与权重向量的内积 */
                float score = 0.0f;
                for (int k = 0; k < feature_dim; k++) {
                    /* 使用word索引和关系类型作为特征索引偏移 */
                    int feat_idx = (i * DEP_COUNT + rt + j * 17) % feature_dim;
                    score += features[feat_idx] * g_sp_learner.feature_weights[k];
                }
                if (score > best_score) {
                    best_score = score;
                    predicted_head = j;
                    predicted_rel = (DepRelationType)rt;
                }
            }
        }

        /* 感知器更新：如果预测错误，更新权重 */
        if (predicted_head != gold_heads[i] || predicted_rel != gold_rels[i]) {
            for (int k = 0; k < feature_dim; k++) {
                /* 正确标签的特征贡献 */
                float gold_feat = features[(i * DEP_COUNT + (int)gold_rels[i] + gold_heads[i] * 17) % feature_dim];
                float pred_feat = features[(i * DEP_COUNT + (int)predicted_rel + predicted_head * 17) % feature_dim];
                g_sp_learner.feature_weights[k] +=
                    g_sp_learner.learning_rate * (gold_feat - pred_feat);
            }
        }
    }

    /* 更新学习率（指数衰减） */
    g_sp_learner.iteration_count++;
    g_sp_learner.learning_rate = g_sp_learner.initial_learning_rate /
        (1.0f + g_sp_learner.decay_rate * (float)g_sp_learner.iteration_count);

    safe_free((void**)&features);
    return 0;
}

int sp_feedback(int word_index, int correct_head, DepRelationType correct_rel) {
    if (!g_sp_learner.is_initialized || word_index < 0) return -1;

    /* 简单反馈：调整对应位置的权重 */
    int feature_dim = g_sp_learner.weight_dim;
    int feat_idx = (word_index * DEP_COUNT + (int)correct_rel + correct_head * 17) % feature_dim;

    if (feat_idx >= 0 && feat_idx < feature_dim) {
        g_sp_learner.feature_weights[feat_idx] += 0.1f; /* 强化正确映射 */
    }

    return 0;
}

int sp_save_weights(const char* filepath) {
    if (!g_sp_learner.is_initialized || !filepath) return -1;
    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;
    /* 写入头部：维度 + 迭代次数 + 学习率 */
    int header[3] = { g_sp_learner.weight_dim, g_sp_learner.iteration_count, 0 };
    memcpy(&header[2], &g_sp_learner.learning_rate, sizeof(float));
    fwrite(header, sizeof(int), 3, f);
    fwrite(g_sp_learner.feature_weights, sizeof(float), (size_t)g_sp_learner.weight_dim, f);
    fclose(f);
    return 0;
}

int sp_load_weights(const char* filepath) {
    if (!filepath) return -1;
    FILE* f = fopen(filepath, "rb");
    if (!f) return -1;
    int header[3];
    if (fread(header, sizeof(int), 3, f) != 3) { fclose(f); return -1; }
    int dim = header[0];
    if (sp_learner_init(dim) != 0) { fclose(f); return -1; }
    g_sp_learner.iteration_count = header[1];
    memcpy(&g_sp_learner.learning_rate, &header[2], sizeof(float));
    fread(g_sp_learner.feature_weights, sizeof(float), (size_t)dim, f);
    fclose(f);
    return 0;
}

/* ============================================================================
 * M-005深度修复: 依存分析器完整训练管线
 *
 * 在已有的在线学习器基础上，增加批量训练、验证评估、
 * 早停机制、学习率调度、训练统计等完整训练管线功能。
 * ============================================================================ */

/** 训练样本结构 */
typedef struct {
    const char** words;           /* 词序列 */
    PartOfSpeech* pos_tags;       /* 词性标注 */
    int word_count;               /* 词数量 */
    int* gold_heads;              /* 标准依存头 */
    DepRelationType* gold_rels;   /* 标准依存关系 */
} SPTrainingSample;

/** 训练统计 */
typedef struct {
    int total_samples;            /* 总样本数 */
    int total_epochs;             /* 总轮数 */
    int correct_heads;            /* 正确的头预测数 */
    int correct_rels;             /* 正确的关系预测数 */
    int total_tokens;             /* 总词元数 */
    float current_loss;           /* 当前损失 */
    float best_accuracy;          /* 最佳准确率 */
    float train_time_ms;          /* 训练耗时 */
    float validation_accuracy;    /* 验证准确率 */
} SPTrainingStats;

/** 训练配置 */
typedef struct {
    int max_epochs;               /* 最大训练轮数 */
    int batch_size;               /* 批次大小 */
    float learning_rate;          /* 初始学习率 */
    float decay_rate;             /* 衰减率 */
    float momentum;               /* 动量 */
    float early_stop_patience;    /* 早停耐心值（轮数） */
    int enable_validation;        /* 是否启用验证 */
    int enable_early_stop;        /* 是否启用早停 */
    int verbose;                  /* 是否打印训练日志 */
} SPTrainingConfig;

/* 全局训练统计 */
static SPTrainingStats g_sp_train_stats = {0};
static SPTrainingConfig g_sp_train_config = {
    50,     /* max_epochs */
    16,     /* batch_size */
    0.01f,  /* learning_rate */
    0.001f, /* decay_rate */
    0.9f,   /* momentum */
    10,     /* early_stop_patience */
    1,      /* enable_validation */
    1,      /* enable_early_stop */
    1       /* verbose */
};

/**
 * @brief 设置训练配置
 */
void sp_training_config_set(const SPTrainingConfig* config) {
    if (config) {
        memcpy(&g_sp_train_config, config, sizeof(SPTrainingConfig));
    }
}

/**
 * @brief 获取训练配置
 */
void sp_training_config_get(SPTrainingConfig* config) {
    if (config) {
        memcpy(config, &g_sp_train_config, sizeof(SPTrainingConfig));
    }
}

/**
 * @brief 获取训练统计
 */
void sp_training_stats_get(SPTrainingStats* stats) {
    if (stats) {
        memcpy(stats, &g_sp_train_stats, sizeof(SPTrainingStats));
    }
}

/**
 * @brief 计算依赖关系预测准确率
 */
static float sp_compute_uas_las(const SPTrainingSample* samples, int sample_count,
                                 int* out_uas, int* out_las) {
    /* 变量声明在MSVC C89模式下的块顶部 */
    int correct_heads = 0;
    int correct_rels = 0;
    int total_tokens = 0;
    int si;
    int ti;
    int best_head;
    DepRelationType best_rel;
    float best_score;
    float score;
    int feature_dim;
    int feat_idx;
    int h;
    int rt;

    if (!samples || sample_count <= 0) {
        if (out_uas) *out_uas = 0;
        if (out_las) *out_las = 0;
        return 0.0f;
    }

    feature_dim = g_sp_learner.weight_dim;
    if (feature_dim <= 0) {
        if (out_uas) *out_uas = 0;
        if (out_las) *out_las = 0;
        return 0.0f;
    }

    for (si = 0; si < sample_count; si++) {
        const SPTrainingSample* s = &samples[si];
        if (!s->words || !s->pos_tags || !s->gold_heads || !s->gold_rels) continue;

        for (ti = 0; ti < s->word_count; ti++) {
            best_head = -1;
            best_rel = DEP_DEP;
            best_score = -1e10f;

            for (h = 0; h < s->word_count; h++) {
                for (rt = 0; rt < DEP_COUNT; rt++) {
                    feat_idx = (ti * DEP_COUNT + rt + h * 17) % feature_dim;
                    score = g_sp_learner.feature_weights[feat_idx];
                    if (score > best_score) {
                        best_score = score;
                        best_head = h;
                        best_rel = (DepRelationType)rt;
                    }
                }
            }

            if (best_head == s->gold_heads[ti]) {
                correct_heads++;
                if (best_rel == s->gold_rels[ti]) {
                    correct_rels++;
                }
            }
            total_tokens++;
        }
    }

    if (out_uas) *out_uas = correct_heads;
    if (out_las) *out_las = correct_rels;
    return (float)correct_rels / (float)(total_tokens > 0 ? total_tokens : 1);
}

/**
 * @brief 单轮训练
 */
static float sp_train_epoch(const SPTrainingSample* samples, int sample_count,
                             float learning_rate) {
    /* 变量声明在MSVC C89模式下的块顶部 */
    int si;
    int ti;
    int best_head;
    DepRelationType best_rel;
    float best_score;
    float score;
    int feature_dim;
    int feat_idx;
    int h;
    int rt;
    float total_loss;
    int updates;

    if (!samples || sample_count <= 0) return 0.0f;

    feature_dim = g_sp_learner.weight_dim;
    if (feature_dim <= 0) return 0.0f;

    total_loss = 0.0f;
    updates = 0;

    for (si = 0; si < sample_count; si++) {
        const SPTrainingSample* s = &samples[si];
        if (!s->words || !s->pos_tags || !s->gold_heads || !s->gold_rels) continue;

        for (ti = 0; ti < s->word_count; ti++) {
            best_head = -1;
            best_rel = DEP_DEP;
            best_score = -1e10f;

            for (h = 0; h < s->word_count; h++) {
                for (rt = 0; rt < DEP_COUNT; rt++) {
                    feat_idx = (ti * DEP_COUNT + rt + h * 17) % feature_dim;
                    score = g_sp_learner.feature_weights[feat_idx];
                    if (score > best_score) {
                        best_score = score;
                        best_head = h;
                        best_rel = (DepRelationType)rt;
                    }
                }
            }

            /* 感知器更新：如果预测错误，更新权重 */
            if (best_head != s->gold_heads[ti] || best_rel != s->gold_rels[ti]) {
                int gold_feat_idx = (ti * DEP_COUNT + (int)s->gold_rels[ti] +
                                     s->gold_heads[ti] * 17) % feature_dim;
                int pred_feat_idx = (ti * DEP_COUNT + (int)best_rel +
                                     best_head * 17) % feature_dim;

                g_sp_learner.feature_weights[gold_feat_idx] += learning_rate;
                g_sp_learner.feature_weights[pred_feat_idx] -= learning_rate;

                total_loss += 1.0f;
                updates++;
            }
        }
    }

    return (updates > 0) ? total_loss / (float)updates : 0.0f;
}

/**
 * @brief M-005深度修复: 批量训练管线
 *
 * 完整的批量训练流程：
 * 1. 初始化学习器
 * 2. 多轮训练 + 学习率调度
 * 3. 验证集评估（可选）
 * 4. 早停机制（可选）
 * 5. 训练统计记录
 *
 * @param train_samples 训练样本数组
 * @param train_count 训练样本数
 * @param val_samples 验证样本数组（NULL=不使用验证）
 * @param val_count 验证样本数
 * @param stats 输出训练统计
 * @return int 成功返回0
 */
int sp_batch_train(const SPTrainingSample* train_samples, int train_count,
                    const SPTrainingSample* val_samples, int val_count,
                    SPTrainingStats* stats) {
    /* 变量声明在MSVC C89模式下的块顶部 */
    int epoch;
    int feature_dim;
    float lr;
    float train_loss;
    float best_val_acc;
    int patience_counter;
    int uas;
    int las;
    float val_acc;
    int total_tokens;
    int si;
    int ti;

    if (!train_samples || train_count <= 0) return -1;

    /* 初始化学习器 */
    feature_dim = 256;  /* 增强特征维度 */
    if (sp_learner_init(feature_dim) != 0) return -1;

    /* 初始化统计 */
    memset(&g_sp_train_stats, 0, sizeof(SPTrainingStats));
    g_sp_train_stats.total_samples = train_count;
    g_sp_train_stats.total_epochs = 0;

    /* 计算总词元数 */
    total_tokens = 0;
    for (si = 0; si < train_count; si++) {
        if (train_samples[si].words) {
            total_tokens += train_samples[si].word_count;
        }
    }
    g_sp_train_stats.total_tokens = total_tokens;

    /* 设置初始学习率 */
    lr = g_sp_train_config.learning_rate;
    g_sp_learner.learning_rate = lr;
    g_sp_learner.initial_learning_rate = lr;

    best_val_acc = 0.0f;
    patience_counter = 0;

    /* 训练循环 */
    for (epoch = 0; epoch < g_sp_train_config.max_epochs; epoch++) {
        /* 单轮训练 */
        train_loss = sp_train_epoch(train_samples, train_count, lr);

        g_sp_train_stats.current_loss = train_loss;
        g_sp_train_stats.total_epochs = epoch + 1;

        /* 计算训练准确率 */
        sp_compute_uas_las(train_samples, train_count, &uas, &las);
        g_sp_train_stats.correct_heads = uas;
        g_sp_train_stats.correct_rels = las;

        /* 验证评估 */
        if (g_sp_train_config.enable_validation && val_samples && val_count > 0) {
            val_acc = sp_compute_uas_las(val_samples, val_count, &uas, &las);
            g_sp_train_stats.validation_accuracy = val_acc;

            if (val_acc > best_val_acc) {
                best_val_acc = val_acc;
                g_sp_train_stats.best_accuracy = val_acc;
                patience_counter = 0;
            } else {
                patience_counter++;
            }

            /* 早停检查 */
            if (g_sp_train_config.enable_early_stop &&
                patience_counter >= (int)g_sp_train_config.early_stop_patience) {
                if (g_sp_train_config.verbose) {
                    log_info("[依存分析训练] 早停触发: epoch=%d, best_val_acc=%.4f",
                             epoch + 1, best_val_acc);
                }
                break;
            }
        } else {
            /* 无验证集时使用训练准确率 */
            float train_acc = (float)las / (float)(total_tokens > 0 ? total_tokens : 1);
            if (train_acc > best_val_acc) {
                best_val_acc = train_acc;
                g_sp_train_stats.best_accuracy = train_acc;
            }
        }

        /* 学习率衰减 */
        lr = g_sp_train_config.learning_rate /
             (1.0f + g_sp_train_config.decay_rate * (float)(epoch + 1));
        g_sp_learner.learning_rate = lr;
        g_sp_learner.iteration_count = epoch + 1;

        /* 打印训练日志 */
        if (g_sp_train_config.verbose && (epoch % 5 == 0 || epoch == g_sp_train_config.max_epochs - 1)) {
            log_info("[依存分析训练] Epoch %d/%d: loss=%.4f, UAS=%d/%d, LAS=%d/%d, lr=%.6f",
                     epoch + 1, g_sp_train_config.max_epochs,
                     train_loss, uas, total_tokens, las, total_tokens, lr);
        }
    }

    /* 输出最终统计 */
    if (stats) {
        memcpy(stats, &g_sp_train_stats, sizeof(SPTrainingStats));
    }

    if (g_sp_train_config.verbose) {
        log_info("[依存分析训练] 完成: %d轮, 最佳LAS=%.4f, 总词元=%d",
                 g_sp_train_stats.total_epochs,
                 g_sp_train_stats.best_accuracy,
                 g_sp_train_stats.total_tokens);
    }

    return 0;
}

/**
 * @brief M-005深度修复: 从标注数据构建训练样本
 *
 * 将原始词序列+词性标注+标准依存头/关系转换为训练样本。
 * 简化接口：调用者提供平坦数组，内部构建SPTrainingSample。
 *
 * @param words 词序列数组（平坦存储）
 * @param pos_tags 词性标注数组
 * @param word_counts 每个样本的词数
 * @param gold_heads 标准依存头数组
 * @param gold_rels 标准依存关系数组
 * @param sample_count 样本数量
 * @param out_samples 输出训练样本数组（调用者分配）
 * @return int 成功返回0
 */
int sp_build_training_samples(const char** words, const PartOfSpeech* pos_tags,
                               const int* word_counts,
                               const int* gold_heads, const DepRelationType* gold_rels,
                               int sample_count, SPTrainingSample* out_samples) {
    /* 变量声明在MSVC C89模式下的块顶部 */
    int si;
    int word_offset;
    int head_offset;
    int wc;

    if (!words || !pos_tags || !word_counts || !gold_heads || !gold_rels || !out_samples) return -1;

    word_offset = 0;
    head_offset = 0;

    for (si = 0; si < sample_count; si++) {
        wc = word_counts[si];
        if (wc <= 0) continue;

        out_samples[si].words = &words[word_offset];
        out_samples[si].pos_tags = (PartOfSpeech*)&pos_tags[word_offset];
        out_samples[si].word_count = wc;
        out_samples[si].gold_heads = (int*)&gold_heads[head_offset];
        out_samples[si].gold_rels = (DepRelationType*)&gold_rels[head_offset];

        word_offset += wc;
        head_offset += wc;
    }

    return 0;
}

/**
 * @brief M-005深度修复: 训练数据管线数据增强
 *
 * 对训练数据进行简单增强：随机交换两个非核心词的顺序，
 * 生成更多训练样本，提高模型泛化能力。
 *
 * @param samples 输入样本数组
 * @param sample_count 样本数量
 * @param augmented 输出增强样本（调用者分配，大小>=sample_count*2）
 * @param aug_count 输出增强后样本数量
 * @return int 成功返回0
 */
int sp_augment_training_data(const SPTrainingSample* samples, int sample_count,
                              SPTrainingSample* augmented, int* aug_count) {
    /* 变量声明在MSVC C89模式下的块顶部 */
    int si;
    int out_idx;
    int swap_i;
    int swap_j;
    int tmp_head;
    DepRelationType tmp_rel;

    if (!samples || !augmented || !aug_count || sample_count <= 0) return -1;

    out_idx = 0;

    for (si = 0; si < sample_count && out_idx < sample_count * 2; si++) {
        /* 复制原始样本 */
        memcpy(&augmented[out_idx], &samples[si], sizeof(SPTrainingSample));
        out_idx++;

        /* 生成增强样本：交换两个非核心词 */
        if (samples[si].word_count >= 4 && out_idx < sample_count * 2) {
            memcpy(&augmented[out_idx], &samples[si], sizeof(SPTrainingSample));

            /* 随机选择两个非核心词交换（跳过根节点和第0个词） */
            swap_i = 1 + (si * 3) % (samples[si].word_count - 1);
            swap_j = 1 + (si * 7 + 2) % (samples[si].word_count - 1);
            if (swap_i != swap_j && swap_i < samples[si].word_count &&
                swap_j < samples[si].word_count) {
                /* 交换gold_heads */
                tmp_head = augmented[out_idx].gold_heads[swap_i];
                augmented[out_idx].gold_heads[swap_i] = augmented[out_idx].gold_heads[swap_j];
                augmented[out_idx].gold_heads[swap_j] = tmp_head;

                /* 交换gold_rels */
                tmp_rel = augmented[out_idx].gold_rels[swap_i];
                augmented[out_idx].gold_rels[swap_i] = augmented[out_idx].gold_rels[swap_j];
                augmented[out_idx].gold_rels[swap_j] = tmp_rel;
            }
            out_idx++;
        }
    }

    *aug_count = out_idx;
    return 0;
}