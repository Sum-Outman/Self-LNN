/**
 * @file ontology_engineering.h
 * @brief 本体工程系统接口
 *
 * 本体构建、本体对齐、本体演化三大核心功能。
 * 支持从知识图谱/文本构建本体，多本体间的概念对齐，
 * 以及本体的版本管理和变更传播。
 */

#ifndef SELFLNN_ONTOLOGY_ENGINEERING_H
#define SELFLNN_ONTOLOGY_ENGINEERING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 本体元素类型
 */
typedef enum {
    ONT_CLASS = 0,          /* 类 */
    ONT_OBJECT_PROPERTY = 1,/* 对象属性（关系） */
    ONT_DATA_PROPERTY = 2,  /* 数据属性 */
    ONT_INDIVIDUAL = 3,     /* 实例 */
    ONT_AXIOM = 4,          /* 公理 */
    ONT_ANNOTATION = 5      /* 注解 */
} OntElementType;

/**
 * @brief 公理类型
 */
typedef enum {
    AXIOM_SUBCLASS = 0,         /* 子类 */
    AXIOM_EQUIVALENT = 1,       /* 等价 */
    AXIOM_DISJOINT = 2,         /* 不相交 */
    AXIOM_DOMAIN = 3,           /* 定义域 */
    AXIOM_RANGE = 4,            /* 值域 */
    AXIOM_SYMMETRIC = 5,        /* 对称 */
    AXIOM_TRANSITIVE = 6,       /* 传递 */
    AXIOM_FUNCTIONAL = 7,       /* 函数 */
    AXIOM_INVERSE = 8,          /* 逆关系 */
    AXIOM_SUBPROPERTY = 9,      /* 子属性 */
    AXIOM_CARDINALITY = 10,     /* 基数约束 */
    AXIOM_VALUE_RESTRICTION = 11 /* 值约束 */
} OntAxiomType;

/**
 * @brief 本体元素
 */
typedef struct OntElement {
    int id;                         /* 元素ID */
    OntElementType type;            /* 元素类型 */
    char* name;                     /* 元素名称 */
    char* description;              /* 描述 */
    struct OntElement** related;    /* 关联元素数组 */
    int related_count;              /* 关联元素数量 */
    OntAxiomType* axiom_types;      /* 公理类型数组 */
    float* axiom_weights;           /* 公理权重数组 */
    float confidence;               /* 置信度 */
    void* user_data;                /* 用户数据 */
} OntElement;

/**
 * @brief 本体句柄
 */
typedef struct Ontology Ontology;

/**
 * @brief 本体对齐结果类型
 */
typedef enum {
    ALIGN_EQUIVALENT = 0,   /* 等价匹配 */
    ALIGN_SUBCLASS = 1,     /* 子类关系 */
    ALIGN_SUPERCLASS = 2,   /* 父类关系 */
    ALIGN_OVERLAP = 3,      /* 部分重叠 */
    ALIGN_RELATED = 4       /* 相关 */
} AlignmentType;

/**
 * @brief 对齐条目
 */
typedef struct {
    char* source_element;            /* 源本体元素名称 */
    char* target_element;            /* 目标本体元素名称 */
    AlignmentType type;              /* 对齐类型 */
    float similarity;                /* 相似度得分 */
    float confidence;               /* 对齐置信度 */
    char* mapping_property;          /* 映射属性 */
} AlignmentEntry;

/**
 * @brief 本体变更操作类型
 */
typedef enum {
    ONT_CHANGE_ADD = 0,             /* 添加 */
    ONT_CHANGE_MODIFY = 1,          /* 修改 */
    ONT_CHANGE_DELETE = 2,          /* 删除 */
    ONT_CHANGE_MERGE = 3,           /* 合并 */
    ONT_CHANGE_SPLIT = 4            /* 分裂 */
} OntChangeOp;

/**
 * @brief 本体变更条目
 */
typedef struct {
    OntChangeOp operation;           /* 操作类型 */
    OntElementType element_type;     /* 元素类型 */
    char* element_name;              /* 元素名称 */
    char* old_value;                 /* 旧值 */
    char* new_value;                 /* 新值 */
    char* description;               /* 变更描述 */
    long timestamp;                  /* 时间戳 */
} OntChangeEntry;

/* =========================================================================
 * 本体构建 API
 * ========================================================================= */

/**
 * @brief 创建空本体
 *
 * @param name 本体名称
 * @param description 本体描述
 * @return Ontology* 本体句柄，失败返回NULL
 */
Ontology* ontology_create(const char* name, const char* description);

/**
 * @brief 释放本体
 *
 * @param ont 本体句柄
 */
void ontology_free(Ontology* ont);

/**
 * @brief 添加本体类
 *
 * @param ont 本体句柄
 * @param name 类名称
 * @param description 类描述
 * @return OntElement* 创建的类元素指针，失败返回NULL
 */
OntElement* ontology_add_class(Ontology* ont, const char* name, const char* description);

/**
 * @brief 添加对象属性（关系）
 *
 * @param ont 本体句柄
 * @param name 属性名称
 * @param description 属性描述
 * @param domain 定义域类名
 * @param range 值域类名
 * @return OntElement* 创建的属性元素指针，失败返回NULL
 */
OntElement* ontology_add_object_property(Ontology* ont, const char* name,
    const char* description, const char* domain, const char* range);

/**
 * @brief 添加数据属性
 *
 * @param ont 本体句柄
 * @param name 属性名称
 * @param description 属性描述
 * @param data_type 数据类型（string/int/float/bool等）
 * @return OntElement* 创建的属性元素指针，失败返回NULL
 */
OntElement* ontology_add_data_property(Ontology* ont, const char* name,
    const char* description, const char* data_type);

/**
 * @brief 添加实例
 *
 * @param ont 本体句柄
 * @param name 实例名称
 * @param class_name 所属类名
 * @return OntElement* 创建的实例元素指针，失败返回NULL
 */
OntElement* ontology_add_individual(Ontology* ont, const char* name,
    const char* class_name);

/**
 * @brief 添加公理
 *
 * @param ont 本体句柄
 * @param axiom_type 公理类型
 * @param subject 主体元素名称
 * @param object 客体元素名称
 * @param weight 公理权重
 * @return int 成功返回0，失败返回-1
 */
int ontology_add_axiom(Ontology* ont, OntAxiomType axiom_type,
    const char* subject, const char* object, float weight);

/**
 * @brief 从三元组构建本体
 *
 * @param ont 本体句柄
 * @param subjects 主体数组
 * @param predicates 谓词数组
 * @param objects 客体数组
 * @param triple_count 三元组数量
 * @return int 成功返回导入的元素数量，失败返回-1
 */
int ontology_build_from_triples(Ontology* ont,
    const char** subjects, const char** predicates, const char** objects,
    int triple_count);

/**
 * @brief 从知识图谱构建本体
 *
 * @param ont 本体句柄
 * @param graph_src 知识图谱源（概念列表JSON格式字符串）
 * @param src_len 源字符串长度
 * @return int 成功返回导入的元素数量，失败返回-1
 */
int ontology_build_from_knowledge_graph(Ontology* ont,
    const char* graph_src, size_t src_len);

/**
 * @brief 本体一致性检查
 *
 * 检查本体中的逻辑一致性，包括：
 * - 循环继承检查
 * - 不相交类冲突检查
 * - 定义域/值域一致性检查
 * - 基数约束一致性检查
 *
 * @param ont 本体句柄
 * @param errors 错误信息输出缓冲区
 * @param max_errors 最大错误数量
 * @return int 返回检测到的错误数量，0表示一致
 */
int ontology_check_consistency(Ontology* ont, char** errors, int max_errors);

/**
 * @brief 按名称查找元素
 *
 * @param ont 本体句柄
 * @param name 元素名称
 * @return OntElement* 元素指针，未找到返回NULL
 */
OntElement* ontology_find_element(Ontology* ont, const char* name);

/**
 * @brief 获取本体中所有元素
 *
 * @param ont 本体句柄
 * @param type 元素类型过滤
 * @param count 输出参数：元素数量
 * @return OntElement** 元素数组，调用者负责释放
 */
OntElement** ontology_get_elements(Ontology* ont, OntElementType type, int* count);

/**
 * @brief 获取本体统计信息
 *
 * @param ont 本体句柄
 * @param out_class_count 类数量输出
 * @param out_property_count 属性数量输出
 * @param out_individual_count 实例数量输出
 * @param out_axiom_count 公理数量输出
 * @return int 成功返回0，失败返回-1
 */
int ontology_get_stats(Ontology* ont, int* out_class_count,
    int* out_property_count, int* out_individual_count, int* out_axiom_count);

/* =========================================================================
 * 本体对齐 API
 * ========================================================================= */

/**
 * @brief 本体对齐器句柄
 */
typedef struct OntologyAligner OntologyAligner;

/**
 * @brief 对齐策略配置
 */
typedef struct {
    float name_weight;           /* 名称相似度权重 */
    float structure_weight;      /* 结构相似度权重 */
    float semantic_weight;       /* 语义相似度权重 */
    float name_threshold;        /* 名称匹配阈值 */
    float structure_threshold;   /* 结构匹配阈值 */
    float semantic_threshold;    /* 语义匹配阈值 */
    float final_threshold;       /* 最终对齐阈值 */
    int use_edit_distance;       /* 是否使用编辑距离 */
    int use_semantic_matching;   /* 是否使用语义匹配 */
    int max_alignments;          /* 最大对齐数 */
} AlignmentConfig;

/**
 * @brief 创建本体对齐器
 *
 * @param config 对齐配置（NULL使用默认配置）
 * @return OntologyAligner* 对齐器句柄，失败返回NULL
 */
OntologyAligner* ontology_aligner_create(const AlignmentConfig* config);

/**
 * @brief 释放本体对齐器
 *
 * @param aligner 对齐器句柄
 */
void ontology_aligner_free(OntologyAligner* aligner);

/**
 * @brief 获取默认对齐配置
 *
 * @return AlignmentConfig 默认配置
 */
AlignmentConfig ontology_aligner_default_config(void);

/**
 * @brief 执行本体对齐
 *
 * 执行多策略本体对齐，综合名称、结构、语义三种匹配策略。
 *
 * @param aligner 对齐器句柄
 * @param source_ont 源本体
 * @param target_ont 目标本体
 * @param out_alignments 对齐结果数组输出
 * @param max_alignments 最大对齐数
 * @return int 返回实际对齐数量，失败返回-1
 */
int ontology_align(OntologyAligner* aligner, Ontology* source_ont,
    Ontology* target_ont, AlignmentEntry* out_alignments, int max_alignments);

/**
 * @brief 基于名称的对齐（编辑距离+子串匹配）
 *
 * @param aligner 对齐器句柄
 * @param source_name 源名称
 * @param target_name 目标名称
 * @return float 相似度得分（0-1）
 */
float ontology_align_name_similarity(OntologyAligner* aligner,
    const char* source_name, const char* target_name);

/**
 * @brief 基于结构的对齐（邻接结构相似度）
 *
 * @param aligner 对齐器句柄
 * @param source_ont 源本体
 * @param target_ont 目标本体
 * @param source_element 源元素名称
 * @param target_element 目标元素名称
 * @return float 相似度得分（0-1）
 */
float ontology_align_structure_similarity(OntologyAligner* aligner,
    Ontology* source_ont, Ontology* target_ont,
    const char* source_element, const char* target_element);

/**
 * @brief 对齐一致性检查
 *
 * 检查对齐结果是否满足一致性约束（无冲突对齐）。
 *
 * @param alignments 对齐结果数组
 * @param alignment_count 对齐数量
 * @return int 返回冲突数，0表示一致
 */
int ontology_align_check_consistency(AlignmentEntry* alignments, int alignment_count);

/* =========================================================================
 * 本体演化 API
 * ========================================================================= */

/**
 * @brief 本体演化管理器句柄
 */
typedef struct OntologyEvolution OntologyEvolution;

/**
 * @brief 创建本体演化管理器
 *
 * @param ont 要管理的本体
 * @return OntologyEvolution* 演化管理器句柄，失败返回NULL
 */
OntologyEvolution* ontology_evolution_create(Ontology* ont);

/**
 * @brief 释放本体演化管理器
 *
 * @param evo 演化管理器句柄
 */
void ontology_evolution_free(OntologyEvolution* evo);

/**
 * @brief 应用变更操作
 *
 * @param evo 演化管理器
 * @param entry 变更条目
 * @return int 成功返回0，失败返回-1
 */
int ontology_evolution_apply_change(OntologyEvolution* evo,
    const OntChangeEntry* entry);

/**
 * @brief 批量应用变更
 *
 * @param evo 演化管理器
 * @param entries 变更条目数组
 * @param entry_count 变更数量
 * @return int 成功返回应用的变更数，失败返回-1
 */
int ontology_evolution_apply_changes(OntologyEvolution* evo,
    const OntChangeEntry* entries, int entry_count);

/**
 * @brief 变更影响分析
 *
 * 分析变更对本体中其他元素的影响范围和程度。
 *
 * @param evo 演化管理器
 * @param entry 变更条目
 * @param affected 受影响元素名称数组输出
 * @param max_affected 最大输出数
 * @return int 返回受影响元素数量
 */
int ontology_evolution_impact_analysis(OntologyEvolution* evo,
    const OntChangeEntry* entry, char** affected, int max_affected);

/**
 * @brief 创建版本快照
 *
 * @param evo 演化管理器
 * @param version_label 版本标签
 * @return int 成功返回版本号，失败返回-1
 */
int ontology_evolution_create_snapshot(OntologyEvolution* evo,
    const char* version_label);

/**
 * @brief 回滚到指定版本
 *
 * @param evo 演化管理器
 * @param version 版本号
 * @return int 成功返回0，失败返回-1
 */
int ontology_evolution_rollback(OntologyEvolution* evo, int version);

/**
 * @brief 获取当前版本号
 *
 * @param evo 演化管理器
 * @return int 当前版本号，失败返回-1
 */
int ontology_evolution_get_current_version(OntologyEvolution* evo);

/**
 * @brief 获取版本历史
 *
 * @param evo 演化管理器
 * @param out_versions 版本号数组输出
 * @param out_labels 版本标签数组输出
 * @param max_versions 最大输出数
 * @return int 返回历史版本数量
 */
int ontology_evolution_get_history(OntologyEvolution* evo,
    int* out_versions, char** out_labels, int max_versions);

/**
 * @brief 比较两个版本的差异
 *
 * @param evo 演化管理器
 * @param version1 版本1
 * @param version2 版本2
 * @param out_diffs 差异条目数组输出
 * @param max_diffs 最大输出数
 * @return int 返回差异数量
 */
int ontology_evolution_diff(OntologyEvolution* evo, int version1, int version2,
    OntChangeEntry* out_diffs, int max_diffs);

/**
 * @brief 导出本体为OWL/XML格式字符串
 *
 * @param ont 本体句柄
 * @return char* OWL格式字符串，调用者负责释放
 */
char* ontology_export_owl(Ontology* ont);

/* =========================================================================
 * L-005修复: OWL/RDF 互操作层 — 导入/导出双向格式转换
 * ========================================================================= */

/**
 * @brief OWL/RDF格式类型枚举
 */
typedef enum {
    ONT_FORMAT_OWL_XML = 0,     /**< OWL/XML格式 */
    ONT_FORMAT_RDF_TURTLE = 1,  /**< RDF/Turtle (TTL) 格式 */
    ONT_FORMAT_RDF_XML = 2,     /**< RDF/XML格式 */
    ONT_FORMAT_AUTO = 3         /**< 自动检测格式 */
} OntologyFormat;

/**
 * @brief 从OWL/XML文件导入本体
 *
 * 解析OWL/XML格式文件，提取类、对象属性、数据属性、实例、
 * 以及公理(子类、等价、不相交、定义域、值域等)。
 *
 * @param ont 本体句柄（如果为NULL则自动创建）
 * @param owl_path OWL/XML文件路径
 * @return Ontology* 成功返回本体句柄，失败返回NULL
 */
Ontology* ontology_import_owl(Ontology* ont, const char* owl_path);

/**
 * @brief 从OWL/XML字符串导入本体
 *
 * @param ont 本体句柄（如果为NULL则自动创建）
 * @param owl_xml OWL/XML字符串
 * @param xml_len 字符串长度
 * @return Ontology* 成功返回本体句柄，失败返回NULL
 */
Ontology* ontology_import_owl_string(Ontology* ont, const char* owl_xml, size_t xml_len);

/**
 * @brief 从RDF/Turtle文件导入本体
 *
 * 解析Turtle (TTL) 格式文件，提取三元组并构建本体结构。
 * 支持@prefix声明、rdf:type、rdfs:subClassOf等标准谓词。
 *
 * @param ont 本体句柄（如果为NULL则自动创建）
 * @param ttl_path Turtle文件路径
 * @return Ontology* 成功返回本体句柄，失败返回NULL
 */
Ontology* ontology_import_rdf(Ontology* ont, const char* ttl_path);

/**
 * @brief 从RDF/Turtle字符串导入本体
 *
 * @param ont 本体句柄（如果为NULL则自动创建）
 * @param ttl_str Turtle格式字符串
 * @param ttl_len 字符串长度
 * @return Ontology* 成功返回本体句柄，失败返回NULL
 */
Ontology* ontology_import_rdf_string(Ontology* ont, const char* ttl_str, size_t ttl_len);

/**
 * @brief 导出本体为RDF/Turtle格式字符串
 *
 * 将本体的类、属性、实例、公理序列化为W3C标准Turtle格式。
 *
 * @param ont 本体句柄
 * @return char* Turtle格式字符串，调用者负责释放
 */
char* ontology_export_rdf(Ontology* ont);

/**
 * @brief 自动检测格式并导入本体文件
 *
 * 根据文件扩展名(.owl, .rdf, .ttl, .xml)或文件内容自动判断格式，
 * 然后调用对应的导入函数。
 *
 * @param ont 本体句柄（如果为NULL则自动创建）
 * @param file_path 文件路径
 * @return Ontology* 成功返回本体句柄，失败返回NULL
 */
Ontology* ontology_import_auto(Ontology* ont, const char* file_path);

/**
 * @brief 保存本体到文件（自动选择格式）
 *
 * 根据文件扩展名自动选择导出格式。
 *
 * @param ont 本体句柄
 * @param file_path 文件路径
 * @param format 导出格式（ONT_FORMAT_AUTO则根据扩展名决定）
 * @return int 成功返回0，失败返回-1
 */
int ontology_save_to_file(Ontology* ont, const char* file_path, OntologyFormat format);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_ONTOLOGY_ENGINEERING_H */
