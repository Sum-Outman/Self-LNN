/**
 * @file knowledge.h
 * @brief 知识库系统接口
 * 
 * 知识表示、存储和检索接口。
 */

#ifndef SELFLNN_KNOWLEDGE_H
#define SELFLNN_KNOWLEDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 知识类型
 */
typedef enum {
    KNOWLEDGE_FACT = 0,      /**< 事实 */
    KNOWLEDGE_RULE = 1,      /**< 规则 */
    KNOWLEDGE_CONCEPT = 2,   /**< 概念 */
    KNOWLEDGE_RELATION = 3,  /**< 关系 */
    KNOWLEDGE_OBSERVATION = 4 /**< 观察 */
} KnowledgeType;

/**
 * @brief 知识置信度
 */
typedef enum {
    CONFIDENCE_LOW = 0,      /**< 低置信度 */
    CONFIDENCE_MEDIUM = 1,   /**< 中置信度 */
    CONFIDENCE_HIGH = 2      /**< 高置信度 */
} KnowledgeConfidence;

/**
 * @brief 知识源
 */
typedef enum {
    SOURCE_PERCEPTION = 0,   /**< 感知 */
    SOURCE_INFERENCE = 1,    /**< 推理 */
    SOURCE_LEARNING = 2,     /**< 学习 */
    SOURCE_USER = 3,         /**< 用户输入 */
    SOURCE_SEMANTIC_NETWORK = 4, /**< 语义网络 */
    SOURCE_KNOWLEDGE_GRAPH = 5,  /**< 知识图谱 */
    SOURCE_LOGIC_REASONING = 6,  /**< 逻辑推理 */
    SOURCE_MERGED = 7,       /**< 合并结果 */
    SOURCE_PRESET = 8,       /**< 系统预置知识（可由用户学习覆盖） */
    SOURCE_AUTO_LEARN = 9    /**< 自动从文件学习 */
} KnowledgeSource;

/**
 * @brief 知识条目
 */
typedef struct {
    char* subject;           /**< 主体 */
    char* predicate;         /**< 谓词/关系 */
    char* object;            /**< 客体/值 */
    
    KnowledgeType type;      /**< 知识类型 */
    KnowledgeConfidence confidence; /**< 置信度 */
    KnowledgeSource source;  /**< 来源 */
    
    float weight;            /**< 权重 */
    long timestamp;          /**< 时间戳 */
    
    void* metadata;          /**< 元数据 */
    size_t metadata_size;    /**< 元数据大小 */
    
    float* embedding;        /**< 嵌入向量 */
    size_t embedding_size;   /**< 嵌入向量维度 */
} KnowledgeEntry;

/**
 * @brief 查询条件
 */
typedef struct {
    char* subject_pattern;   /**< 主体模式（可为NULL） */
    char* predicate_pattern; /**< 谓词模式（可为NULL） */
    char* object_pattern;    /**< 客体模式（可为NULL） */
    
    KnowledgeType type_filter; /**< 类型过滤（-1表示不过滤） */
    float min_confidence;    /**< 最小置信度 */
    float max_confidence;    /**< 最大置信度 */
    
    long start_time;         /**< 开始时间 */
    long end_time;           /**< 结束时间 */
} KnowledgeQuery;

/**
 * @brief 知识库句柄
 */
typedef struct KnowledgeBase KnowledgeBase;

/**
 * @brief 知识库内部结构体（完整定义，供直接字段访问使用）
 * knowledge.c中定义SELFLNN_KNOWLEDGE_IMPL时使用完整定义，
 * 其他文件使用兼容的KnowledgeEntry*版本。
 */
#ifndef SELFLNN_KNOWLEDGE_IMPL
struct KnowledgeBase {
    KnowledgeEntry* entries;     /**< 条目数组 */
    size_t capacity;             /**< 数组容量 */
    size_t size;                 /**< 当前条目数 */
    size_t max_entries;          /**< 最大条目数（0=无限制） */
    size_t entry_count;          /**< 条目计数（与size等价，兼容不同命名） */
    int next_id;                 /**< 下一个ID */
    void* cfc_embed;             /**< CfC嵌入引擎句柄 */
    int cfc_embed_dim;           /**< 嵌入向量维度 */
};
#endif

/**
 * @brief 创建知识库
 * 
 * @param max_entries 最大条目数（0表示无限制）
 * @return KnowledgeBase* 知识库句柄，失败返回NULL
 */
KnowledgeBase* knowledge_base_create(size_t max_entries);

/**
 * @brief 创建知识库并自动加载预设常识知识
 * 
 * 包含数学、物理、化学、生物、地理、计算机科学、逻辑推理等领域的基础常识。
 * 满足需求"完整的知识库"的要求。
 *
 * @param max_entries 最大条目数（0表示无限制）
 * @return KnowledgeBase* 知识库句柄，失败返回NULL
 */
KnowledgeBase* knowledge_base_create_with_preset(size_t max_entries);

/**
 * @brief 向现有知识库填充预设常识知识
 *
 * @param kb 知识库句柄
 * @return int 成功添加的条目数
 */
int knowledge_base_populate_preset(KnowledgeBase* kb);

/**
 * @brief K-006: 从外部知识库文件加载知识条目
 * 
 * 文件格式为每行一条知识: subject\tpredicate\tobject\ttype\tconfidence\tweight
 * 其中type: 0=事实 1=规则 2=概念 3=关系 4=观察
 * confidence: 0=低 1=中 2=高
 * 空行和以#开头的行被忽略。
 * 
 * @param kb 知识库句柄
 * @param filepath 知识库文件路径
 * @return int 成功加载的条目数，失败返回-1
 */
int knowledge_base_load_from_file(KnowledgeBase* kb, const char* filepath);

/**
 * @brief 从JSON种子知识文件加载知识条目
 * 格式: {"version":1, "entries":[{"s":"主体","p":"谓词","o":"客体","t":"FACT","c":"HIGH","w":1.0},...]}
 * 使用紧凑键名(s/p/o/t/c/w)减少文件体积。
 * @param kb 知识库句柄
 * @param filepath JSON文件路径
 * @return int 成功加载的条目数，失败返回-1
 */
int knowledge_base_import_seed_json(KnowledgeBase* kb, const char* filepath);

/**
 * @brief 导出知识库到JSON文件
 * 格式与 knowledge_base_import_seed_json() 兼容，支持备份和迁移。
 * @param kb 知识库句柄
 * @param filepath 输出JSON文件路径
 * @return int 成功导出的条目数，失败返回-1
 */
int knowledge_base_export_json(KnowledgeBase* kb, const char* filepath);

/**
 * @brief 释放知识库
 * 
 * @param kb 知识库句柄
 */
void knowledge_base_free(KnowledgeBase* kb);

/**
 * @brief 添加知识条目
 * 
 * @param kb 知识库句柄
 * @param entry 知识条目
 * @return int 成功返回条目ID，失败返回-1
 */
int knowledge_base_add(KnowledgeBase* kb, const KnowledgeEntry* entry);

/**
 * @brief 删除知识条目
 * 
 * @param kb 知识库句柄
 * @param entry_id 条目ID
 * @return int 成功返回0，失败返回-1
 */
int knowledge_base_remove(KnowledgeBase* kb, int entry_id);

/**
 * @brief 更新知识条目
 * 
 * @param kb 知识库句柄
 * @param entry_id 条目ID
 * @param entry 新条目数据
 * @return int 成功返回0，失败返回-1
 */
int knowledge_base_update(KnowledgeBase* kb, int entry_id, const KnowledgeEntry* entry);

/**
 * @brief 查询知识条目
 * 
 * @param kb 知识库句柄
 * @param query 查询条件
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return int 返回匹配的条目数
 */
int knowledge_base_query(KnowledgeBase* kb, const KnowledgeQuery* query,
                        KnowledgeEntry* results, size_t max_results);

/**
 * @brief 状态感知语义查询——结合CfC嵌入语义相似度对结果排序
 *
 * 在基本查询基础上，使用状态向量作为上下文计算每个匹配条目的复合相关性得分：
 * score = embedding_similarity * 0.35 + text_match * 0.20 + confidence * 0.15 + weight * 0.15 + recency * 0.15
 * 并按照得分降序返回，使AGI认知循环能获取最相关的知识。
 *
 * @param kb 知识库句柄
 * @param query 查询条件（可为NULL，此时匹配所有条目）
 * @param context_embedding 上下文状态向量（可为NULL，用于语义相关性计算）
 * @param context_dim 状态向量维度（context_embedding为NULL时传0）
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @param result_scores 各结果得分输出缓冲区（可为NULL，长度应与max_results一致）
 * @return int 返回匹配的条目数
 */
int knowledge_base_query_state_aware(KnowledgeBase* kb,
                                     const KnowledgeQuery* query,
                                     const float* context_embedding,
                                     int context_dim,
                                     KnowledgeEntry* results,
                                     size_t max_results,
                                     float* result_scores);

/**
 * @brief 按来源过滤的知识查询（防止自主学习污染用户教学知识）
 *
 *: 在查询时按KnowledgeSource过滤，
 * source_filter=-1 表示不过滤（返回所有来源）。
 *
 * @param kb 知识库句柄
 * @param query 查询条件
 * @param source_filter 期望的知识来源（-1=全部, 0=用户教学, 1=自主学习等）
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return int 匹配条目数
 */
int knowledge_base_query_by_source(KnowledgeBase* kb, const KnowledgeQuery* query,
                                   int source_filter,
                                   KnowledgeEntry* results, size_t max_results);

/**
 * @brief 根据ID获取知识条目
 * 
 * @param kb 知识库句柄
 * @param entry_id 条目ID
 * @param entry 条目输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int knowledge_base_get_by_id(KnowledgeBase* kb, int entry_id, KnowledgeEntry* entry);

/**
 * @brief 获取知识库统计信息
 * 
 * @param kb 知识库句柄
 * @param total_entries 总条目数输出
 * @param memory_usage 内存使用量输出（字节）
 * @return int 成功返回0，失败返回-1
 */
int knowledge_base_get_stats(KnowledgeBase* kb, size_t* total_entries, size_t* memory_usage);

/**
 * @brief 保存知识库到文件
 * 
 * @param kb 知识库句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int knowledge_base_save(KnowledgeBase* kb, const char* filename);

/**
 * @brief 自动保存知识库到默认路径 knowledge_data/knowledge_base.skb
 * @param kb 知识库句柄
 * @return int 成功返回0，失败返回-1
 */
int knowledge_base_auto_save(KnowledgeBase* kb);

/**
 * @brief 从默认路径加载知识库（追加到现有知识库，不覆盖）
 * @param kb 知识库句柄
 * @return int 成功返回0，失败返回-1
 */
int knowledge_base_auto_load(KnowledgeBase* kb);

/**
 * @brief 获取知识库默认存储路径
 * @return const char* 默认路径字符串
 */
const char* knowledge_base_get_default_path(void);

/**
 * @brief 从文件加载知识库
 * 
 * @param filename 文件名
 * @return KnowledgeBase* 知识库句柄，失败返回NULL
 */
KnowledgeBase* knowledge_base_load(const char* filename);

/**
 * @brief 清空知识库
 * 
 * @param kb 知识库句柄
 */
void knowledge_base_clear(KnowledgeBase* kb);

/**
 * @brief 搜索相似知识
 * 
 * @param kb 知识库句柄
 * @param subject 主体
 * @param predicate 谓词
 * @param object 客体
 * @param similarity_threshold 相似度阈值
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return int 返回匹配的条目数
 */
int knowledge_base_search_similar(KnowledgeBase* kb,
                                 const char* subject, const char* predicate, const char* object,
                                 float similarity_threshold,
                                 KnowledgeEntry* results, size_t max_results);

/**
 * @brief 为知识库启用CfC嵌入引擎，使语义搜索可用
 *
 * 创建CfC嵌入引擎，为所有知识条目生成语义嵌入向量。
 * 之后可通过 knowledge_base_cfc_semantic_search 进行语义检索。
 *
 * @param kb 知识库句柄
 * @param embedding_dim 嵌入向量维度（建议64~256，传0则使用默认128）
 * @return int 成功返回0，失败返回-1
 */
int knowledge_base_enable_cfc_embedding(KnowledgeBase* kb, int embedding_dim);

/**
 * @brief 基于CfC嵌入向量的语义相似度搜索
 *
 * 使用余弦相似度比较查询文本与知识条目的嵌入向量，
 * 返回语义最相似的知识条目。
 *
 * @param kb 知识库句柄
 * @param query_text 查询文本
 * @param similarity_threshold 语义相似度阈值（0~1）
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return int 返回匹配的条目数，失败返回-1
 */
int knowledge_base_cfc_semantic_search(KnowledgeBase* kb,
                                       const char* query_text,
                                       float similarity_threshold,
                                       KnowledgeEntry* results,
                                       size_t max_results);

/**
 * @brief TF-IDF全文检索知识库
 *
 * 使用TF-IDF（词频-逆文档频率）+余弦相似度对知识库进行全文排序检索。
 * 支持中英文混合分词。Laplace平滑IDF。纯C实现，不依赖外部库。
 *
 * @param kb 知识库句柄
 * @param query_text 查询文本（自然语言查询字符串）
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @param scores 各结果TF-IDF分数输出（可为NULL）
 * @param min_score 最小分数阈值（0-1），建议0.01
 * @return int 返回匹配的条目数
 */
int knowledge_base_search_tfidf(KnowledgeBase* kb,
                                const char* query_text,
                                KnowledgeEntry* results, size_t max_results,
                                float* scores, float min_score);

/**
 * @brief 推理新知识
 * 
 * @param kb 知识库句柄
 * @param rule_pattern 规则模式
 * @param max_inferences 最大推理数量
 * @param inferred_entries 推理结果输出缓冲区
 * @param max_entries 最大条目数
 * @return int 返回推理出的条目数
 */
int knowledge_base_infer(KnowledgeBase* kb, const char* rule_pattern,
                        size_t max_inferences,
                        KnowledgeEntry* inferred_entries, size_t max_entries);

/**
 * @brief 在知识图谱中查找两个实体间的因果路径长度（BFS搜索）
 * @param kb 知识库句柄
 * @param from_entity 起始实体名
 * @param to_entity 目标实体名
 * @return int 路径长度（1=直接连接），0表示无路径，-1表示错误
 */
int knowledge_find_causal_path_length(KnowledgeBase* kb,
                                      const char* from_entity,
                                      const char* to_entity);

/**
 * @brief 合并两个知识库
 * 
 * @param dest 目标知识库
 * @param src 源知识库
 * @param conflict_resolution 冲突解决策略（0:保留目标, 1:保留源, 2:合并）
 * @return int 成功返回0，失败返回-1
 */
int knowledge_base_merge(KnowledgeBase* dest, KnowledgeBase* src, int conflict_resolution);

/**
 * @brief 学习结果结构体
 */
typedef struct {
    size_t new_knowledge_count;      /**< 新增知识条目数 */
    size_t updated_knowledge_count;  /**< 更新知识条目数 */
    char* learning_summary;          /**< 学习摘要 */
    float learning_score;            /**< 学习得分 */
    long learning_time_ms;           /**< 学习时间（毫秒） */
} LearningResult;

/**
 * @brief 演化结果结构体
 */
typedef struct {
    size_t concepts_evolved;         /**< 演化的概念数 */
    size_t rules_evolved;            /**< 演化的规则数 */
    float diversity_score;           /**< 多样性分数 */
    char* evolution_summary;         /**< 演化摘要 */
    long evolution_time_ms;          /**< 演化时间（毫秒） */
} EvolutionResult;

/**
 * @brief 知识库自我学习配置
 */
typedef struct {
    int max_new_knowledge;           /**< 最大新增知识条目数，默认32 */
    int min_frequency_threshold;     /**< 最小频率阈值，默认2 */
    char* topic_filter;              /**< 主题过滤字符串（可为NULL） */
} KnowledgeSelfLearnConfig;

/**
 * @brief 知识库自我演化配置
 */
typedef struct {
    int max_generations;             /**< 最大演化代数，默认10 */
    float mutation_rate;             /**< 变异率，默认0.1f */
    float crossover_rate;            /**< 交叉率，默认0.7f */
    int population_size;             /**< 种群大小，默认10 */
} KnowledgeSelfEvolveConfig;

/**
 * @brief 知识库自我修正配置
 */
typedef struct {
    int max_corrections;             /**< 最大修正条目数，默认100 */
    float low_confidence_threshold;  /**< 低置信度阈值，默认0.3f */
    int max_age_days;                /**< 旧知识最大天数阈值，默认60 */
} KnowledgeSelfCorrectConfig;

/**
 * @brief 知识库自我学习
 * 
 * @param kb 知识库句柄
 * @param config 学习配置
 * @param description 学习描述
 * @return LearningResult* 学习结果，失败返回NULL
 */
LearningResult* knowledge_self_learn(KnowledgeBase* kb, const void* config, const char* description);

/**
 * @brief 知识库自我演化
 * 
 * @param kb 知识库句柄
 * @param config 学习配置
 * @param description 演化描述
 * @return EvolutionResult* 演化结果，失败返回NULL
 */
EvolutionResult* knowledge_self_evolve(KnowledgeBase* kb, const void* config, const char* description);

void knowledge_set_lnn_network(KnowledgeBase* kb, void* lnn_network);
void* knowledge_get_lnn_network(const KnowledgeBase* kb);
int knowledge_has_lnn_integration(const KnowledgeBase* kb);

/* 知识库更新事件通知回调机制
  * 当 knowledge_base_add() 成功写入新知识后，通过此回调主动通知
  * 上层系统（如LNN重新进行知识嵌入编码、推理引擎刷新缓存等），
  * 替代原来main.c中定时轮询的被动方式，消除延迟。 */
 typedef void (*KnowledgeUpdateCallback)(void* user_data);
 void knowledge_base_set_update_callback(KnowledgeUpdateCallback callback, void* user_data);

 /**
  * @brief 触发CfC嵌入引擎对知识库全量重新训练
  *
  * 当知识库通过回调机制收到更新通知后，AGI后台循环调用此函数
  * 对新增的知识条目进行嵌入向量重新编码，替代原来的定时轮询方式。
  *
  * @param kb 知识库句柄
  * @param epochs 训练轮数（传0则使用默认2轮）
  * @return int 成功返回0，失败返回-1
  */
 int knowledge_base_retrain_embeddings(KnowledgeBase* kb, int epochs);

/**
 * @brief 知识库自我修正
 * 
 * @param kb 知识库句柄
 * @param config 学习配置
 * @param description 修正描述
 * @return LearningResult* 修正结果，失败返回NULL
 */
LearningResult* knowledge_self_correct(KnowledgeBase* kb, const void* config, const char* description);

/**
 * @brief 释放学习结果
 * 
 * @param result 学习结果
 */
void learning_result_free(LearningResult* result);

/**
 * @brief 释放演化结果
 * 
 * @param result 演化结果
 */
void evolution_result_free(EvolutionResult* result);

/**
 * @brief 推理结果结构体
 */
typedef struct {
    size_t result_count;           /**< 结果数量 */
    KnowledgeEntry* results;       /**< 结果数组 */
    float confidence_score;        /**< 总体置信度得分 */
    char* query_summary;           /**< 查询摘要 */
} InferenceResult;

/**
 * @brief 知识查询
 * 
 * @param kb 知识库句柄
 * @param query_text 查询文本
 * @param max_results 最大结果数
 * @param min_confidence 最小置信度
 * @return InferenceResult* 推理结果，失败返回NULL
 */
InferenceResult* knowledge_query(KnowledgeBase* kb, const char* query_text, 
                                int max_results, float min_confidence);

/**
 * @brief 释放推理结果
 * 
 * @param result 推理结果
 */
void inference_result_free(InferenceResult* result);

/**
 * @brief 前向链接推理 - 从已知事实出发，应用规则推导新事实
 * 
 * 实现完整的前向链接推理算法，支持多前提规则和递归推理。
 * 使用确定性算法避免无限循环，提供真实的推理能力。
 * 
 * @param kb 知识库句柄
 * @param max_iterations 最大迭代次数
 * @param max_new_facts 最大新事实数
 * @param inferred_entries 推理结果输出缓冲区
 * @param max_entries 最大条目数
 * @return int 返回推理出的新事实数
 */
int knowledge_base_forward_chaining(KnowledgeBase* kb,
                                   size_t max_iterations,
                                   size_t max_new_facts,
                                   KnowledgeEntry* inferred_entries,
                                   size_t max_entries);

/**
 * @brief 增强推理 - 结合多种推理方法的高级推理引擎
 * 
 * 整合前向链接、后向链接和概率推理，提供更强大的推理能力。
 * 支持规则优先级、置信度传播和矛盾检测。
 * 
 * @param kb 知识库句柄
 * @param query_pattern 查询模式（可选）
 * @param inference_mode 推理模式（0:自动, 1:前向, 2:后向, 3:混合）
 * @param max_inferences 最大推理数量
 * @param inferred_entries 推理结果输出缓冲区
 * @param max_entries 最大条目数
 * @return int 返回推理出的条目数
 */
int knowledge_base_enhanced_infer(KnowledgeBase* kb,
                                 const char* query_pattern,
                                 int inference_mode,
                                 size_t max_inferences,
                                 KnowledgeEntry* inferred_entries,
                                 size_t max_entries);

/**
 * @brief 时间排序方向
 */
typedef enum {
    TIME_ORDER_ASC = 0,   /**< 时间升序（从旧到新） */
    TIME_ORDER_DESC = 1   /**< 时间降序（从新到旧） */
} TemporalOrder;

/**
 * @brief 时序模式类型
 */
typedef enum {
    TEMPORAL_PATTERN_SEQUENCE = 0,  /**< 序列模式：A→B→C */
    TEMPORAL_PATTERN_CYCLE = 1,     /**< 周期模式：周期性重复 */
    TEMPORAL_PATTERN_TREND = 2,     /**< 趋势模式：单调变化 */
    TEMPORAL_PATTERN_CORRELATION = 3 /**< 相关模式：事件同时发生 */
} TemporalPatternType;

/**
 * @brief 时序模式
 */
typedef struct {
    TemporalPatternType type;        /**< 模式类型 */
    char** event_sequence;           /**< 事件序列（主体-谓词-客体三元组字符串） */
    size_t sequence_length;          /**< 序列长度 */
    float confidence;                /**< 模式置信度 (0-1) */
    float period_ms;                 /**< 周期（毫秒，仅周期模式） */
    long first_occurrence;           /**< 首次出现时间 */
    long last_occurrence;            /**< 最后出现时间 */
    size_t occurrence_count;         /**< 出现次数 */
    float trend_slope;               /**< 趋势斜率（仅趋势模式） */
} TemporalPattern;

/**
 * @brief 时序一致性冲突
 */
typedef struct {
    int entry_id_a;                  /**< 冲突条目A的ID */
    int entry_id_b;                  /**< 冲突条目B的ID */
    long timestamp_a;                /**< 条目A的时间戳 */
    long timestamp_b;                /**< 条目B的时间戳 */
    float conflict_score;            /**< 冲突严重程度 (0-1) */
    char* description;               /**< 冲突描述 */
} TemporalConflict;

/**
 * @brief 使用抽象能力系统对知识进行抽象处理
 * 
 * @param kb 知识库
 * @param input 输入数据
 * @param input_size 输入大小
 * @param target_level 目标抽象层次 (0-5)
 * @param output 抽象输出缓冲区
 * @param max_output_size 输出缓冲区最大大小
 * @return int 成功返回抽象输出大小，失败返回-1
 */
int knowledge_abstraction_process(KnowledgeBase* kb,
                                  const float* input, size_t input_size,
                                  int target_level,
                                  float* output, size_t max_output_size);

/**
 * @brief 基于知识库条目学习抽象概念
 * 
 * @param kb 知识库
 * @param entry_ids 条目ID数组
 * @param num_entries 条目数量
 * @param concept_name 概念名称
 * @return int 成功返回0，失败返回-1
 */
int knowledge_learn_concept(KnowledgeBase* kb, const int* entry_ids,
                           size_t num_entries, const char* concept_name);

/**
 * @brief 在知识库中执行类比推理
 * 
 * @param kb 知识库
 * @param source_domain 源领域数据
 * @param source_size 源领域大小
 * @param target_domain 目标领域数据
 * @param target_size 目标领域大小
 * @param mapping 类比映射结果缓冲区 (AnalogyMapping*)
 * @param max_mappings 最大映射数量
 * @return int 成功返回映射数量，失败返回-1
 */
int knowledge_analogical_reasoning(KnowledgeBase* kb,
                                   const float* source_domain, size_t source_size,
                                   const float* target_domain, size_t target_size,
                                   void* mapping, size_t max_mappings);

/**
 * @brief 从知识库中归纳模式
 * 
 * @param kb 知识库
 * @param pattern_ids 模式条目ID数组
 * @param num_patterns 模式数量
 * @param induced_pattern 归纳结果输出
 * @param max_pattern_size 输出缓冲区大小
 * @return int 成功返回归纳模式大小，失败返回-1
 */
int knowledge_pattern_induction(KnowledgeBase* kb, const int* pattern_ids,
                                size_t num_patterns,
                                float* induced_pattern, size_t max_pattern_size);

/**
 * @brief 获取知识库的抽象能力状态
 * 
 * @param kb 知识库
 * @param state 状态输出缓冲区 (AbstractionState*)
 * @return int 成功返回0，失败返回-1
 */
int knowledge_get_abstraction_state(KnowledgeBase* kb, void* state);

/**
 * @brief 时序知识查询 - 在指定时间范围内查询并按时间排序
 * 
 * @param kb 知识库句柄
 * @param query 查询条件（可为NULL查询所有）
 * @param start_time 开始时间戳（毫秒，0表示不限制）
 * @param end_time 结束时间戳（毫秒，0表示不限制）
 * @param time_order 排序方向（TIME_ORDER_ASC或TIME_ORDER_DESC）
 * @param results 结果输出缓冲区
 * @param max_results 最大结果数
 * @return int 返回匹配的条目数，失败返回-1
 */
int knowledge_base_temporal_query(KnowledgeBase* kb, const KnowledgeQuery* query,
                                  long start_time, long end_time,
                                  TemporalOrder time_order,
                                  KnowledgeEntry* results, size_t max_results);

/**
 * @brief 获取知识演化时间线 - 按时间区间的知识密度分布
 * 
 * @param kb 知识库句柄
 * @param start_time 开始时间戳
 * @param end_time 结束时间戳
 * @param interval_ms 时间区间宽度（毫秒）
 * @param timeline_data 时间线数据输出缓冲区（大小为num_points的float数组，每个元素表示该区间知识条目数）
 * @param num_points 时间点数量（输出区间的数量）
 * @return int 成功返回0，失败返回-1
 */
int knowledge_base_get_timeline(KnowledgeBase* kb,
                                long start_time, long end_time,
                                long interval_ms,
                                float* timeline_data, size_t num_points);

/**
 * @brief 发现时序模式 - 检测知识库中的周期性模式、序列模式等
 * 
 * @param kb 知识库句柄
 * @param min_occurrences 最小出现次数（过滤低频模式）
 * @param patterns 时序模式输出缓冲区（TemporalPattern数组）
 * @param max_patterns 最大模式数量
 * @return int 返回发现的模式数量，失败返回-1
 */
int knowledge_base_find_temporal_patterns(KnowledgeBase* kb,
                                          size_t min_occurrences,
                                          TemporalPattern* patterns, size_t max_patterns);

/**
 * @brief 时序一致性检查 - 检测知识库中的时序冲突
 * 
 * @param kb 知识库句柄
 * @param time_window_ms 时间窗口大小（毫秒，在此窗口内的相关知识会进行一致性检查）
 * @param conflicts 冲突输出缓冲区（TemporalConflict数组）
 * @param max_conflicts 最大冲突数量
 * @return int 返回发现的冲突数量，失败返回-1
 */
int knowledge_base_temporal_consistency_check(KnowledgeBase* kb,
                                              long time_window_ms,
                                              TemporalConflict* conflicts, size_t max_conflicts);

/**
 * @brief 释放知识条目内部的动态分配内存
 * 
 * 释放由知识条目中的 subject、predicate、object、metadata 等字段
 * 动态分配的内存。注意：此函数不释放 entry 结构体本身。
 * 
 * @param entry 知识条目
 */
void knowledge_entry_free(KnowledgeEntry* entry);

/**
 * @brief 释放时序模式数组
 * 
 * @param patterns 时序模式数组
 * @param count 数组大小
 */
void temporal_patterns_free(TemporalPattern* patterns, size_t count);

/**
 * @brief 释放时序冲突数组
 * 
 * @param conflicts 时序冲突数组
 * @param count 数组大小
 */
void temporal_conflicts_free(TemporalConflict* conflicts, size_t count);

/**
 * @brief 计算两个字符串的相似度（公共前缀比例）
 * 
 * @param str1 第一个字符串
 * @param str2 第二个字符串
 * @return float 相似度 (0-1)
 */
float knowledge_string_similarity(const char* str1, const char* str2);

/* 知识库辅助函数 */
size_t knowledge_base_get_total_facts(KnowledgeBase* kb);
float knowledge_base_output_consistency(KnowledgeBase* kb, const float* output, size_t dim);

/* 查找知识库中与给定向量的嵌入最相似的事实
 * 当LNN输出爆炸或异常时，用于找到知识库中最近的锚定事实约束修正 */
int knowledge_base_nearest_fact(KnowledgeBase* kb, const float* query_vec, size_t dim,
                                char* subject_out, size_t subj_size,
                                char* predicate_out, size_t pred_size,
                                char* object_out, size_t obj_size,
                                float* similarity_out);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_KNOWLEDGE_H