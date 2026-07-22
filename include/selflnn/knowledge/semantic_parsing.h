/**
 * @file semantic_parsing.h
 * @brief 语义解析系统接口
 *
 * 依存句法分析、语义角色标注、语义相似度计算。
 * 基于规则和统计方法的自然语言语义分析引擎。
 */

#ifndef SELFLNN_SEMANTIC_PARSING_H
#define SELFLNN_SEMANTIC_PARSING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 依存关系类型（中文依存句法）
 */
typedef enum {
    DEP_SBJ = 0,           /* 主语 */
    DEP_OBJ = 1,           /* 宾语 */
    DEP_DOBJ = 2,          /* 直接宾语 */
    DEP_IOBJ = 3,          /* 间接宾语 */
    DEP_ATT = 4,           /* 定语 */
    DEP_ADV = 5,           /* 状语 */
    DEP_CMP = 6,           /* 补语 */
    DEP_COO = 7,           /* 并列 */
    DEP_POBJ = 8,          /* 介宾 */
    DEP_SBV = 9,           /* 主谓 */
    DEP_VOB = 10,          /* 动宾 */
    DEP_POB = 11,          /* 介宾(短) */
    DEP_LAD = 12,          /* 左附加 */
    DEP_RAD = 13,          /* 右附加 */
    DEP_IS = 14,           /* 独立结构 */
    DEP_HED = 15,          /* 核心 */
    DEP_ROOT = 16,         /* 根节点 */
    DEP_CNJ = 17,          /* 连词 */
    DEP_MT = 18,           /* 情态 */
    DEP_TPC = 19,          /* 话题 */
    DEP_VOC = 20,          /* 呼语 */
    DEP_APP = 21,          /* 同位语 */
    DEP_NMOD = 22,         /* 名词修饰 */
    DEP_AMOD = 23,         /* 形容词修饰 */
    DEP_NUM_MOD = 24,      /* 数词修饰 */
    DEP_CLF = 25,          /* 量词修饰 */
    DEP_NSUBJ = 26,        /* 名词性主语 */
    DEP_NSUBJPASS = 27,    /* 名词性被动主语 */
    DEP_DOBJPASS = 28,     /* 直接被动宾语 */
    DEP_IOPJ = 29,         /* 间接宾语 */
    DEP_MARK = 30,         /* 标记 */
    DEP_NEG = 31,          /* 否定修饰 */
    DEP_PUNCT = 32,        /* 标点 */
    DEP_LOCATION_OF = 33,  /* 位置关系 */
    DEP_INSTANCE_OF = 34,  /* 实例关系 */
    DEP_CAUSES = 35,       /* 因果关系 */
    DEP_PURPOSE = 36,      /* 目的关系 */
    DEP_TIME = 37,         /* 时间关系 */
    DEP_INSTRUMENT = 38,   /* 工具关系 */
    DEP_DEP = 39,          /* 未指定依存 */
    DEP_COUNT = 40
} DepRelationType;

/**
 * @brief 词性标注类型（简化中文词性集）
 */
typedef enum {
    POS_UNKNOWN = 0,
    POS_NN = 1,            /* 名词 */
    POS_NR = 2,            /* 专有名词 */
    POS_NT = 3,            /* 时间名词 */
    POS_NS = 4,            /* 处所名词 */
    POS_VV = 5,            /* 动词 */
    POS_VA = 6,            /* 形容词 */
    POS_VC = 7,            /* 系动词 */
    POS_VE = 8,            /* 能愿动词 */
    POS_AD = 9,            /* 副词 */
    POS_P = 10,            /* 介词 */
    POS_CC = 11,           /* 连词 */
    POS_DT = 12,           /* 限定词 */
    POS_CD = 13,           /* 数词 */
    POS_M = 14,            /* 量词 */
    POS_PN = 15,           /* 代词 */
    POS_DEG = 16,          /* 的 */
    POS_DE = 17,           /* 地/得 */
    POS_SP = 18,           /* 标点 */
    POS_IJ = 19,           /* 感叹词 */
    POS_FW = 20,           /* 外来词 */
    POS_PU = 21,           /* 标点符号 */
    POS_COUNT = 22
} PartOfSpeech;

/**
 * @brief 依存句法分析结果 - 一个词节点
 */
typedef struct {
    int id;                 /* 词序号（从0开始） */
    char* word;             /* 词文本 */
    char* lemma;            /* 词干/原型 */
    PartOfSpeech pos;       /* 词性 */
    char* pos_tag;          /* 词性标注字符串（扩展） */
    int head_id;            /* 依存父节点ID（-1表示根） */
    DepRelationType dep_rel; /* 与父节点的依存关系类型 */
    float confidence;       /* 分析置信度 */
} DepNode;

/**
 * @brief 依存句法分析结果
 */
typedef struct {
    DepNode* nodes;         /* 节点数组 */
    int node_count;         /* 节点数量 */
    int root_id;            /* 根节点ID */
    float score;            /* 整体分析得分 */
} DependencyParseResult;

/**
 * @brief 语义角色类型
 */
typedef enum {
    SR_AGENT = 0,          /* 施事者 */
    SR_PATIENT = 1,        /* 受事者 */
    SR_INSTRUMENT = 2,     /* 工具 */
    SR_LOCATION = 3,       /* 位置 */
    SR_TIME = 4,           /* 时间 */
    SR_MANNER = 5,         /* 方式 */
    SR_PURPOSE = 6,        /* 目的 */
    SR_CAUSE = 7,          /* 原因 */
    SR_RESULT = 8,         /* 结果 */
    SR_SOURCE = 9,         /* 来源 */
    SR_DESTINATION = 10,   /* 目的地 */
    SR_EXPERIENCER = 11,   /* 体验者 */
    SR_STIMULUS = 12,      /* 刺激 */
    SR_THEME = 13,         /* 主题 */
    SR_BENEFICIARY = 14,   /* 受益者 */
    SR_ATTRIBUTE = 15,     /* 属性 */
    SR_EXTENT = 16,        /* 程度 */
    SR_FREQUENCY = 17,     /* 频率 */
    SR_ORDER = 18,         /* 顺序 */
    SR_MODAL = 19,         /* 情态 */
    SR_NEGATION = 20,      /* 否定 */
    SR_COUNT = 21
} SemanticRole;

/**
 * @brief 语义角色标注 - 一个论元
 */
typedef struct {
    SemanticRole role;      /* 角色类型 */
    int start_token_id;     /* 起始词ID */
    int end_token_id;       /* 结束词ID */
    float confidence;       /* 标注置信度 */
} SRLArgument;

/**
 * @brief 语义角色标注 - 一个谓词及其论元
 */
typedef struct {
    int predicate_id;       /* 谓词词ID */
    char* predicate_word;   /* 谓词文本 */
    SRLArgument* arguments; /* 论元数组 */
    int argument_count;     /* 论元数量 */
    float confidence;       /* 标注置信度 */
} SRLPredicate;

/**
 * @brief 语义角色标注结果
 */
typedef struct {
    SRLPredicate* predicates; /* 谓词数组 */
    int predicate_count;      /* 谓词数量 */
} SRLResult;

/**
 * @brief 依存句法分析器句柄
 */
typedef struct DependencyParser DependencyParser;

/**
 * @brief 语义角色标注器句柄
 */
typedef struct SemanticRoleLabeler SemanticRoleLabeler;

/**
 * @brief 语义相似度计算器句柄
 */
typedef struct SemanticSimilarity SimilarityCalculator;

/**
 * @brief 语义句子结构（包含依存句法和语义角色信息）
 */
typedef struct {
    char* text;                        /* 原句文本 */
    DependencyParseResult* dependency;  /* 依存分析结果 */
    SRLResult* srl_result;             /* 语义角色标注结果 */
} SemanticParsingResult;

/* =========================================================================
 * 依存句法分析器 API
 * ========================================================================= */

/**
 * @brief 创建基于转移的依存句法分析器
 *
 * @return DependencyParser* 分析器句柄，失败返回NULL
 */
DependencyParser* dependency_parser_create(void);

/**
 * @brief 释放依存句法分析器
 *
 * @param parser 分析器句柄
 */
void dependency_parser_free(DependencyParser* parser);

/**
 * @brief 设置词性标注序列（分词+词性预处理）
 *
 * @param parser 分析器句柄
 * @param words 词序列
 * @param pos_tags 词性标注序列
 * @param word_count 词数量
 * @return int 成功返回0，失败返回-1
 */
int dependency_parser_set_tokens(DependencyParser* parser,
                                 const char** words,
                                 const PartOfSpeech* pos_tags,
                                 int word_count);

/**
 * @brief 执行依存句法分析（Arc-Standard 转移系统）
 *
 * 实现基于非投影Arc-Standard转移的依存分析：
 * Shift, LeftArc, RightArc 三种转移操作，
 * 使用静态oracle训练/推理。
 *
 * @param parser 分析器句柄
 * @return DependencyParseResult* 分析结果，调用者负责释放，失败返回NULL
 */
DependencyParseResult* dependency_parser_parse(DependencyParser* parser);

/**
 * @brief 基于约束满足的依存分析（投影）
 *
 * 使用Eisner算法进行投影依存分析，保证生成
 * 合法的投影依存树（无交叉弧）。
 *
 * @param parser 分析器句柄
 * @return DependencyParseResult* 分析结果，调用者负责释放，失败返回NULL
 */
DependencyParseResult* dependency_parser_parse_projective(DependencyParser* parser);

/**
 * @brief 释放依存分析结果
 *
 * @param result 分析结果指针
 */
void dependency_parse_result_free(DependencyParseResult* result);

/**
 * @brief 获取依存弧上的最短路径
 *
 * 在依存树中查找两个词之间的最短路径（经过的节点数）。
 *
 * @param result 依存分析结果
 * @param from_id 起始词ID
 * @param to_id 目标词ID
 * @return int 路径长度（节点数），失败返回-1
 */
int dependency_parse_shortest_path(DependencyParseResult* result,
                                   int from_id, int to_id);

/**
 * @brief 将依存关系类型转换为字符串
 *
 * @param type 依存关系类型
 * @return const char* 字符串表示
 */
const char* dep_relation_type_string(DepRelationType type);

/**
 * @brief 将词性枚举转换为字符串
 *
 * @param pos 词性
 * @return const char* 字符串表示
 */
const char* pos_to_string(PartOfSpeech pos);

/* =========================================================================
 * 语义角色标注 API
 * ========================================================================= */

/**
 * @brief 创建语义角色标注器
 *
 * @return SemanticRoleLabeler* 标注器句柄，失败返回NULL
 */
SemanticRoleLabeler* srl_labeler_create(void);

/**
 * @brief 释放语义角色标注器
 *
 * @param labeler 标注器句柄
 */
void srl_labeler_free(SemanticRoleLabeler* labeler);

/**
 * @brief 设置语义角色映射规则（自定义模式）
 *
 * 允许用户配置从依存关系到语义角色的映射规则。
 * 例如：nsubj→Agent, dobj→Patient
 *
 * @param labeler 标注器句柄
 * @param dep_type 依存关系类型
 * @param role 映射到的语义角色
 * @return int 成功返回0，失败返回-1
 */
int srl_labeler_set_mapping(SemanticRoleLabeler* labeler,
                            DepRelationType dep_type, SemanticRole role);

/**
 * @brief 基于依存分析结果执行语义角色标注
 *
 * @param labeler 标注器句柄
 * @param dep_result 依存句法分析结果
 * @return SRLResult* 角色标注结果，调用者负责释放，失败返回NULL
 */
SRLResult* srl_labeler_label(SemanticRoleLabeler* labeler,
                             DependencyParseResult* dep_result);

/**
 * @brief 释放语义角色标注结果
 *
 * @param result 标注结果指针
 */
void srl_result_free(SRLResult* result);

/**
 * @brief 将语义角色类型转换为字符串
 *
 * @param role 语义角色类型
 * @return const char* 字符串表示
 */
const char* semantic_role_string(SemanticRole role);

/* =========================================================================
 * 语义相似度计算 API
 * ========================================================================= */

/**
 * @brief 语义相似度度量方法
 */
typedef enum {
    SS_COSINE = 0,          /* 余弦相似度（基于嵌入向量） */
    SS_EUCLIDEAN = 1,       /* 欧几里得距离（转相似度） */
    SS_PATH = 2,            /* 基于路径的相似度（WordNet/同义词林风格） */
    SS_WUP = 3,             /* Wu-Palmer 相似度 */
    SS_LIN = 4,             /* Lin 相似度（基于信息内容） */
    SS_RESNIK = 5,          /* Resnik 相似度（基于信息内容） */
    SS_JIANG_CONRATH = 6,   /* Jiang-Conrath 相似度 */
    SS_ENSEMBLE = 7         /* 集成相似度（多种方法加权融合） */
} SimMetric;

/**
 * @brief 创建语义相似度计算器
 *
 * @return SimilarityCalculator* 计算器句柄，失败返回NULL
 */
SimilarityCalculator* similarity_calculator_create(void);

/**
 * @brief 释放语义相似度计算器
 *
 * @param calc 计算器句柄
 */
void similarity_calculator_free(SimilarityCalculator* calc);

/**
 * @brief 计算两个词之间的语义相似度
 *
 * @param calc 计算器句柄
 * @param word1 词1
 * @param word2 词2
 * @param metric 相似度度量方法
 * @return float 相似度得分（0-1），失败返回-1
 */
float similarity_calculator_word_similarity(SimilarityCalculator* calc,
    const char* word1, const char* word2, SimMetric metric);

/**
 * @brief 计算两个句子的语义相似度
 *
 * 使用词相似度矩阵和最优匹配策略计算句子级相似度。
 *
 * @param calc 计算器句柄
 * @param words1 句子1的词序列
 * @param len1 句子1的词数量
 * @param words2 句子2的词序列
 * @param len2 句子2的词数量
 * @param metric 词相似度度量方法
 * @return float 相似度得分（0-1），失败返回-1
 */
float similarity_calculator_sentence_similarity(SimilarityCalculator* calc,
    const char** words1, int len1,
    const char** words2, int len2,
    SimMetric metric);

/**
 * @brief 设置语义嵌入表
 *
 * 提供预训练的词嵌入向量用于余弦相似度计算。
 *
 * @param calc 计算器句柄
 * @param words 词数组
 * @param embeddings 嵌入向量数组（一维，按行存储）
 * @param word_count 词数量
 * @param embedding_dim 嵌入维度
 * @return int 成功返回0，失败返回-1
 */
int similarity_calculator_set_embeddings(SimilarityCalculator* calc,
    const char** words, const float* embeddings,
    int word_count, int embedding_dim);

/**
 * @brief 设置层次结构信息（用于路径/信息内容相似度）
 *
 * 提供概念层次树结构，类似WordNet的同义词集层次。
 *
 * @param calc 计算器句柄
 * @param concepts 概念名称数组
 * @param concept_count 概念数量
 * @param parents 父概念索引数组（-1表示根）
 * @param frequencies 概念频率数组（用于信息内容计算）
 * @return int 成功返回0，失败返回-1
 */
int similarity_calculator_set_hierarchy(SimilarityCalculator* calc,
    const char** concepts, int concept_count,
    const int* parents, const float* frequencies);

/**
 * @brief 计算语义相关度（非对称）
 *
 * 衡量word1到word2的语义关联强度，方向敏感。
 *
 * @param calc 计算器句柄
 * @param word1 词1
 * @param word2 词2
 * @return float 相关度得分（0-1）
 */
float similarity_calculator_semantic_relatedness(SimilarityCalculator* calc,
    const char* word1, const char* word2);

/* =========================================================================
 * 完整语义解析管道 API
 * ========================================================================= */

/**
 * @brief 执行完整语义解析（分词→依存分析→语义角色标注）
 *
 * @param text 输入文本
 * @return SemanticParsingResult* 解析结果，调用者负责释放
 */
SemanticParsingResult* semantic_parse_full(const char* text);

/**
 * @brief 释放完整语义解析结果
 *
 * @param result 解析结果指针
 */
void semantic_parsing_result_free(SemanticParsingResult* result);

/**
 * @brief 将依存分析结果格式化为中文可读字符串
 *
 * @param result 依存分析结果
 * @return char* 格式化字符串，调用者负责释放
 */
char* dependency_parse_result_format(DependencyParseResult* result);

/**
 * @brief 将语义角色标注结果格式化为中文可读字符串
 *
 * @param result 语义角色标注结果
 * @return char* 格式化字符串，调用者负责释放
 */
char* srl_result_format(SRLResult* result);

/* ============================================================================
 * K-修复: SWRL规则引擎接口
 * ============================================================================ */

/** 初始化SWRL规则引擎 */
int semantic_parsing_swrl_init(void);

/** 清理SWRL规则引擎 */
void semantic_parsing_swrl_cleanup(void);

/** 添加SWRL规则 */
int semantic_parsing_swrl_add_rule(const char* name, const char* body_rules,
    int body_count, const char* head_rules, int head_count, float confidence);

/** 添加SWRL事实三元组 */
int semantic_parsing_swrl_add_fact(const char* subject, const char* predicate,
    const char* object, float confidence);

/** 执行SWRL推理（前向链） */
int semantic_parsing_swrl_reason(int max_iterations, int* inferred_count);

/** 获取SWRL引擎统计 */
int semantic_parsing_swrl_get_stats(int* out_rule_count, int* out_triple_count,
    int* out_inference_count, int* out_conflict_count);

/* =========================================================================
 * M-005修复: 依存分析器在线学习 API
 * ========================================================================= */

/**
 * @brief 在线学习：使用标注样例更新依存分析器权重
 *
 * 基于感知器更新规则，接收标注的依存关系样例，
 * 更新转移分类器的特征权重。支持持续增量学习。
 *
 * @param words 词序列数组
 * @param pos_tags 词性标注序列
 * @param word_count 词数量
 * @param gold_heads 标准依存父节点ID数组
 * @param gold_rels 标准依存关系类型数组
 * @param gold_count 标注数量（应等于word_count）
 * @return int 成功返回0，失败返回-1
 */
int sp_online_learn(const char** words, const PartOfSpeech* pos_tags, int word_count,
                     const int* gold_heads, const DepRelationType* gold_rels,
                     int gold_count);

/**
 * @brief 用户反馈纠错：调整特定词位置的依存权重
 *
 * @param word_index 要纠正的词索引
 * @param correct_head 正确的父节点ID
 * @param correct_rel 正确的依存关系类型
 * @return int 成功返回0，失败返回-1
 */
int sp_feedback(int word_index, int correct_head, DepRelationType correct_rel);

/**
 * @brief 保存学习后的权重到文件
 *
 * @param filepath 权重文件路径
 * @return int 成功返回0，失败返回-1
 */
int sp_save_weights(const char* filepath);

/**
 * @brief 从文件加载学习后的权重
 *
 * @param filepath 权重文件路径
 * @return int 成功返回0，失败返回-1
 */
int sp_load_weights(const char* filepath);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SEMANTIC_PARSING_H */
