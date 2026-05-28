/**
 * @file uncertainty_reasoning.h
 * @brief 不确定性推理引擎接口
 *
 * 支持模糊逻辑推理、概率逻辑推理、Dempster-Shafer证据理论。
 * 可与现有逻辑推理引擎集成使用。
 */

#ifndef SELFLNN_UNCERTAINTY_REASONING_H
#define SELFLNN_UNCERTAINTY_REASONING_H

#include "selflnn/knowledge/logic_reasoning.h"
#include "selflnn/reasoning/reasoning.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 模糊逻辑推理
 * =========================================================================== */

/** 最大前提数 */
#define UR_MAX_PREMISES 16
/** 模糊术语最大长度 */
#define UR_MAX_TERM_LEN 64
/** 模糊规则最大数量 */
#define UR_MAX_FUZZY_RULES 1024
/** 隶属度函数采样点数 */
#define UR_FUZZY_SAMPLES 256

/**
 * @brief 隶属度函数类型
 */
typedef enum {
    MF_TRIANGULAR = 0,      /**< 三角形隶属度函数 */
    MF_TRAPEZOIDAL,         /**< 梯形隶属度函数 */
    MF_GAUSSIAN,            /**< 高斯隶属度函数 */
    MF_BELL,                /**< 钟形隶属度函数 */
    MF_SIGMOID,             /**< S型隶属度函数 */
    MF_ZSHAPE               /**< Z型隶属度函数 */
} MembershipFuncType;

/**
 * @brief 隶属度函数
 */
typedef struct {
    MembershipFuncType type;       /**< 函数类型 */
    char name[UR_MAX_TERM_LEN];    /**< 语言术语名称（如"高"/"低"/"中等"） */
    float params[4];               /**< 参数数组（因类型而异） */
    float support_min;             /**< 支撑集下界 */
    float support_max;             /**< 支撑集上界 */
} MembershipFunction;

/**
 * @brief 模糊变量
 */
typedef struct {
    char name[UR_MAX_TERM_LEN];    /**< 变量名称 */
    MembershipFunction* terms;     /**< 语言术语（隶属度函数数组） */
    int term_count;                /**< 术语数量 */
    float min_value;               /**< 变量最小值 */
    float max_value;               /**< 变量最大值 */
} FuzzyVariable;

/**
 * @brief 模糊规则前提
 */
typedef struct {
    char var_name[UR_MAX_TERM_LEN];   /**< 变量名称 */
    char term_name[UR_MAX_TERM_LEN];  /**< 术语名称 */
    int is_negated;                   /**< 是否取非（NOT操作） */
} FuzzyAntecedent;

/**
 * @brief 模糊规则
 */
typedef struct {
    int rule_id;                          /**< 规则ID */
    FuzzyAntecedent premises[UR_MAX_PREMISES]; /**< 前提数组 */
    int premise_count;                    /**< 前提数量 */
    char conclusion_var[UR_MAX_TERM_LEN]; /**< 结论变量 */
    char conclusion_term[UR_MAX_TERM_LEN];/**< 结论术语 */
    float weight;                         /**< 规则权重 */
    float confidence;                     /**< 规则置信度 */
    int is_active;                        /**< 是否激活 */
} FuzzyRule;

/**
 * @brief 去模糊化方法
 */
typedef enum {
    DEFUZZ_CENTROID = 0,    /**< 重心法（COG） */
    DEFUZZ_BISECTOR,        /**< 平分线法 */
    DEFUZZ_MOM,             /**< 最大平均值法（MOM） */
    DEFUZZ_LOM,             /**< 最大左值法（LOM） */
    DEFUZZ_ROM              /**< 最大右值法（ROM） */
} DefuzzificationMethod;

/**
 * @brief 模糊推理方法
 */
typedef enum {
    FUZZY_INFER_MAMDANI = 0,  /**< Mamdani推理（min蕴含 + max聚合） */
    FUZZY_INFER_SUGENO        /**< Sugeno推理（加权平均） */
} FuzzyInferenceMethod;

/**
 * @brief 模糊推理引擎
 */
typedef struct FuzzyEngine FuzzyEngine;

/**
 * @brief 创建模糊推理引擎
 *
 * @param method 推理方法
 * @param defuzz 去模糊化方法
 * @return FuzzyEngine* 成功返回引擎句柄，失败返回NULL
 */
FuzzyEngine* fuzzy_engine_create(FuzzyInferenceMethod method, DefuzzificationMethod defuzz);

/**
 * @brief 释放模糊推理引擎
 *
 * @param engine 模糊推理引擎句柄
 */
void fuzzy_engine_free(FuzzyEngine* engine);

/**
 * @brief 定义模糊变量
 *
 * @param engine 模糊推理引擎句柄
 * @param var 模糊变量定义
 * @return int 成功返回0，失败返回-1
 */
int fuzzy_engine_add_variable(FuzzyEngine* engine, const FuzzyVariable* var);

/**
 * @brief 添加模糊规则
 *
 * @param engine 模糊推理引擎句柄
 * @param rule 模糊规则
 * @return int 成功返回规则ID，失败返回-1
 */
int fuzzy_engine_add_rule(FuzzyEngine* engine, const FuzzyRule* rule);

/**
 * @brief 设置输入值
 *
 * @param engine 模糊推理引擎句柄
 * @param var_name 变量名称
 * @param value 输入值
 * @return int 成功返回0，失败返回-1
 */
int fuzzy_engine_set_input(FuzzyEngine* engine, const char* var_name, float value);

/**
 * @brief 执行模糊推理
 *
 * 对每个规则：计算前提激活度（模糊AND/OR）→ 蕴含（剪切/缩放）
 * 聚合所有规则输出 → 去模糊化得到精确值
 *
 * @param engine 模糊推理引擎句柄
 * @return int 成功返回0，失败返回-1
 */
int fuzzy_engine_evaluate(FuzzyEngine* engine);

/**
 * @brief 获取模糊推理结果
 *
 * @param engine 模糊推理引擎句柄
 * @param var_name 输出变量名称
 * @param output 输出值缓冲区
 * @return int 成功返回0，失败返回-1
 */
int fuzzy_engine_get_output(FuzzyEngine* engine, const char* var_name, float* output);

/**
 * @brief 计算隶属度
 *
 * @param mf 隶属度函数
 * @param x 输入值
 * @return float 隶属度（0-1）
 */
float fuzzy_membership(const MembershipFunction* mf, float x);

/**
 * @brief 重置模糊引擎状态
 *
 * @param engine 模糊推理引擎句柄
 */
void fuzzy_engine_reset(FuzzyEngine* engine);

/**
 * @brief 获取模糊引擎统计信息
 *
 * @param engine 模糊推理引擎句柄
 * @param var_count 变量数量输出
 * @param rule_count 规则数量输出
 * @return int 成功返回0，失败返回-1
 */
int fuzzy_engine_get_stats(FuzzyEngine* engine, int* var_count, int* rule_count);

/* ============================================================================
 * 概率逻辑推理
 * =========================================================================== */

/** 最大证据变量数 */
#define UR_MAX_PROB_VARS 64
/** 最大条件概率表大小 */
#define UR_MAX_CPT_SIZE 256
/** 最大马尔可夫毯节点数 */
#define UR_MAX_MB_NODES 32

/**
 * @brief 概率变量类型
 */
typedef enum {
    PROB_VAR_BOOLEAN = 0,   /**< 布尔变量（True/False） */
    PROB_VAR_DISCRETE,      /**< 离散变量（多值） */
    PROB_VAR_CONTINUOUS     /**< 连续变量（高斯近似） */
} ProbVariableType;

/**
 * @brief 概率变量
 */
typedef struct {
    char name[UR_MAX_TERM_LEN];        /**< 变量名称 */
    ProbVariableType type;             /**< 变量类型 */
    int value_count;                    /**< 取值数量（离散） */
    char** value_names;                /**< 取值名称数组 */
    float* prior_probs;                /**< 先验概率数组 */
} ProbVariable;

/**
 * @brief 条件概率表条目
 */
typedef struct {
    int parent_values[UR_MAX_MB_NODES]; /**< 父节点取值索引 */
    int parent_count;                   /**< 父节点数量 */
    float* probs;                       /**< 条件概率分布 */
    int prob_count;                     /**< 概率分布大小 */
} CPTEntry;

/**
 * @brief 条件概率表（CPT）
 */
typedef struct {
    char var_name[UR_MAX_TERM_LEN];     /**< 变量名称 */
    char** parent_names;               /**< 父节点名称列表 */
    int parent_count;                   /**< 父节点数量 */
    CPTEntry* entries;                 /**< CPT条目数组 */
    int entry_count;                    /**< 条目数量 */
    int entry_capacity;                /**< 条目容量 */
} ConditionalProbabilityTable;

/**
 * @brief 贝叶斯网络
 * 
 * 注意：若 reasoning.h 已包含（定义了SELFLNN_REASONING_H），
 * 则跳过此定义使用 reasoning.h 中的 BayesianNetwork 兼容定义。
 */
#ifndef SELFLNN_REASONING_H
typedef struct {
    char name[UR_MAX_TERM_LEN];          /**< 网络名称 */
    ProbVariable* variables;             /**< 变量数组 */
    int var_count;                       /**< 变量数量 */
    int var_capacity;                    /**< 变量容量 */
    ConditionalProbabilityTable* cpts;   /**< CPT数组 */
    int cpt_count;                       /**< CPT数量 */
    int* adj_matrix;                     /**< 邻接矩阵（var_count x var_count） */
    int adj_capacity;                    /**< 邻接矩阵容量 */
} BayesianNetwork;
#endif /* !SELFLNN_REASONING_H */

/**
 * @brief 贝叶斯推理引擎
 */
typedef struct ProbEngine ProbEngine;

/**
 * @brief 创建概率逻辑推理引擎
 *
 * @return ProbEngine* 成功返回引擎句柄，失败返回NULL
 */
ProbEngine* prob_engine_create(void);

/**
 * @brief 释放概率逻辑推理引擎
 *
 * @param engine 概率推理引擎句柄
 */
void prob_engine_free(ProbEngine* engine);

/**
 * @brief 加载贝叶斯网络
 *
 * @param engine 概率推理引擎句柄
 * @param network 贝叶斯网络
 * @return int 成功返回0，失败返回-1
 */
int prob_engine_load_network(ProbEngine* engine, const BayesianNetwork* network);

/**
 * @brief 设置证据（观测到的变量值）
 *
 * @param engine 概率推理引擎句柄
 * @param var_name 变量名称
 * @param value_index 取值索引
 * @return int 成功返回0，失败返回-1
 */
int prob_engine_set_evidence(ProbEngine* engine, const char* var_name, int value_index);

/**
 * @brief 执行变量消元推理
 *
 * 使用变量消元算法计算查询变量的后验概率分布。
 *
 * @param engine 概率推理引擎句柄
 * @param query_var 查询变量名称
 * @param result_probs 结果概率数组输出缓冲区
 * @param max_probs 结果概率数组大小
 * @return int 成功返回概率分布大小，失败返回-1
 */
int prob_engine_infer(ProbEngine* engine, const char* query_var, float* result_probs, int max_probs);

/**
 * @brief 执行拒绝采样推理（近似）
 *
 * 使用拒绝采样算法进行近似推理，适用于大规模贝叶斯网络。
 *
 * @param engine 概率推理引擎句柄
 * @param query_var 查询变量名称
 * @param sample_count 采样次数
 * @param result_probs 结果概率数组输出缓冲区
 * @param max_probs 结果概率数组大小
 * @return int 成功返回概率分布大小，失败返回-1
 */
int prob_engine_rejection_sampling(ProbEngine* engine, const char* query_var,
                                    int sample_count, float* result_probs, int max_probs);

/**
 * @brief 执行似然加权采样推理（近似）
 *
 * 使用似然加权采样算法，比拒绝采样更高效。
 *
 * @param engine 概率推理引擎句柄
 * @param query_var 查询变量名称
 * @param sample_count 采样次数
 * @param result_probs 结果概率数组输出缓冲区
 * @param max_probs 结果概率数组大小
 * @return int 成功返回概率分布大小，失败返回-1
 */
int prob_engine_likelihood_weighting(ProbEngine* engine, const char* query_var,
                                      int sample_count, float* result_probs, int max_probs);

/**
 * @brief 重置概率推理引擎状态
 *
 * @param engine 概率推理引擎句柄
 */
void prob_engine_reset(ProbEngine* engine);

/**
 * @brief 获取概率引擎统计信息
 *
 * @param engine 概率推理引擎句柄
 * @param var_count 变量数量输出
 * @param cpt_count CPT数量输出
 * @return int 成功返回0，失败返回-1
 */
int prob_engine_get_stats(ProbEngine* engine, int* var_count, int* cpt_count);

/* ============================================================================
 * Dempster-Shafer 证据理论
 * =========================================================================== */

/** 最大辨识框架元素数 */
#define UR_MAX_FRAME_ELEMENTS 64
/** 最大焦元数 */
#define UR_MAX_FOCAL_ELEMENTS 256

/**
 * @brief Dempster-Shafer焦元（命题）
 */
typedef struct {
    int element_mask;                   /** 位掩码表示包含的辨识框架元素 */
    float mass;                         /** 基本概率分配（BPA） */
    float belief;                       /** 信度（累积） */
    float plausibility;                 /** 似真度（累积） */
    char description[256];              /** 命题描述 */
} DSElement;

/**
 * @brief 辨识框架
 */
typedef struct {
    char name[UR_MAX_TERM_LEN];          /** 框架名称 */
    char** element_names;               /** 元素名称数组 */
    int element_count;                   /** 元素数量 */
} DSFrame;

/**
 * @brief 证据体（BPA分布）
 */
typedef struct {
    DSFrame* frame;                      /** 所属辨识框架 */
    DSElement* focal_elements;          /** 焦元数组 */
    int focal_count;                     /** 焦元数量 */
    int focal_capacity;                  /** 焦元容量 */
    float total_mass;                   /** 总BPA（应接近1.0） */
} DSBodyOfEvidence;

/**
 * @brief Dempster-Shafer推理引擎
 */
typedef struct DSEngine DSEngine;

/**
 * @brief 创建Dempster-Shafer推理引擎
 *
 * @return DSEngine* 成功返回引擎句柄，失败返回NULL
 */
DSEngine* ds_engine_create(void);

/**
 * @brief 释放Dempster-Shafer推理引擎
 *
 * @param engine DS推理引擎句柄
 */
void ds_engine_free(DSEngine* engine);

/**
 * @brief 定义辨识框架
 *
 * @param engine DS推理引擎句柄
 * @param frame 辨识框架
 * @return int 成功返回0，失败返回-1
 */
int ds_engine_set_frame(DSEngine* engine, const DSFrame* frame);

/**
 * @brief 添加证据体
 *
 * @param engine DS推理引擎句柄
 * @param evidence 证据体
 * @return int 成功返回证据体ID，失败返回-1
 */
int ds_engine_add_evidence(DSEngine* engine, const DSBodyOfEvidence* evidence);

/**
 * @brief 执行Dempster组合规则
 *
 * 计算 m12(A) = (1/(1-K)) * Σ_{B∩C=A} m1(B) * m2(C)
 * 其中K = Σ_{B∩C=∅} m1(B) * m2(C) 为冲突系数
 *
 * @param engine DS推理引擎句柄
 * @param result 组合结果证据体输出
 * @return int 成功返回0，失败返回-1
 */
int ds_engine_combine(DSEngine* engine, DSBodyOfEvidence* result);

/**
 * @brief 计算信度函数
 *
 * Bel(A) = Σ_{B⊆A} m(B)
 *
 * @param element_mask 命题的位掩码
 * @param evidence 证据体
 * @return float 信度值
 */
float ds_belief(int element_mask, const DSBodyOfEvidence* evidence);

/**
 * @brief 计算似真度函数
 *
 * Pl(A) = Σ_{B∩A≠∅} m(B)
 *
 * @param element_mask 命题的位掩码
 * @param evidence 证据体
 * @return float 似真度值
 */
float ds_plausibility(int element_mask, const DSBodyOfEvidence* evidence);

/**
 * @brief 执行Dempster组合规则（静态函数）
 *
 * @param ev1 证据体1
 * @param ev2 证据体2
 * @param result 组合结果输出
 * @return int 成功返回0，失败返回-1
 */
int ds_combine_two(const DSBodyOfEvidence* ev1, const DSBodyOfEvidence* ev2,
                   DSBodyOfEvidence* result);

/**
 * @brief 执行Yager修改的组合规则
 *
 * Yager将冲突分配给全集，避免归一化导致的反直觉结果。
 *
 * @param ev1 证据体1
 * @param ev2 证据体2
 * @param result 组合结果输出
 * @return int 成功返回0，失败返回-1
 */
int ds_combine_yager(const DSBodyOfEvidence* ev1, const DSBodyOfEvidence* ev2,
                     DSBodyOfEvidence* result);

/**
 * @brief 计算证据冲突系数K
 *
 * K = Σ_{B∩C=∅} m1(B) * m2(C)
 *
 * @param ev1 证据体1
 * @param ev2 证据体2
 * @return float 冲突系数（0=完全一致，1=完全冲突）
 */
float ds_conflict_coefficient(const DSBodyOfEvidence* ev1, const DSBodyOfEvidence* ev2);

/**
 * @brief 计算信度区间 [Bel, Pl]
 *
 * @param element_mask 命题的位掩码
 * @param evidence 证据体
 * @param belief 信度值输出
 * @param plausibility 似真度值输出
 * @return int 成功返回0，失败返回-1
 */
int ds_belief_interval(int element_mask, const DSBodyOfEvidence* evidence,
                       float* belief, float* plausibility);

/**
 * @brief 决策规则：选择信度区间最大的命题
 *
 * @param evidence 证据体
 * @param decision 决策结果描述输出缓冲区
 * @param decision_size 缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int ds_decide(const DSBodyOfEvidence* evidence, char* decision, size_t decision_size);

/**
 * @brief 重置DS推理引擎
 *
 * @param engine DS推理引擎句柄
 */
void ds_engine_reset(DSEngine* engine);

/**
 * @brief 获取DS引擎统计信息
 *
 * @param engine DS推理引擎句柄
 * @param frame_size 辨识框架大小输出
 * @param evidence_count 证据体数量输出
 * @return int 成功返回0，失败返回-1
 */
int ds_engine_get_stats(DSEngine* engine, int* frame_size, int* evidence_count);

/* ============================================================================
 * 综合不确定性推理（集成三种方法）
 * =========================================================================== */

/**
 * @brief 不确定性推理方法
 */
typedef enum {
    UR_METHOD_FUZZY = 0,          /**< 模糊逻辑推理 */
    UR_METHOD_PROBABILISTIC,      /**< 概率逻辑推理 */
    UR_METHOD_DS_EVIDENCE,        /**< Dempster-Shafer证据推理 */
    UR_METHOD_HYBRID              /**< 混合推理 */
} URMethod;

/**
 * @brief 不确定性推理引擎配置
 */
typedef struct {
    URMethod method;                    /**< 推理方法 */
    FuzzyInferenceMethod fuzzy_method;  /**< 模糊推理方法 */
    DefuzzificationMethod defuzz_method;/**< 去模糊化方法 */
    int prob_sample_count;              /**< 概率采样次数 */
    float ds_conflict_threshold;        /**< DS冲突阈值 */
    char network_name[UR_MAX_TERM_LEN]; /**< 贝叶斯网络名称 */
} URConfig;

/**
 * @brief 不确定性推理结果
 */
typedef struct {
    float confidence;                    /**< 总体置信度 */
    float fuzzy_output;                  /**< 模糊推理输出 */
    float* prob_distribution;           /**< 概率分布 */
    int prob_size;                       /**< 概率分布大小 */
    DSBodyOfEvidence* ds_result;        /**< DS组合结果 */
    int inference_time_ms;              /**< 推理时间（毫秒） */
    char trace[2048];                   /**< 推理追踪 */
} URResult;

/**
 * @brief 不确定性推理引擎
 */
typedef struct UncertaintyEngine UncertaintyEngine;

/**
 * @brief 创建不确定性推理引擎
 *
 * @param config 引擎配置
 * @return UncertaintyEngine* 成功返回引擎句柄，失败返回NULL
 */
UncertaintyEngine* uncertainty_engine_create(const URConfig* config);

/**
 * @brief 释放不确定性推理引擎
 *
 * @param engine 不确定性推理引擎句柄
 */
void uncertainty_engine_free(UncertaintyEngine* engine);

/**
 * @brief 执行不确定性推理
 *
 * 根据配置的方法调用对应的推理引擎。
 *
 * @param engine 不确定性推理引擎句柄
 * @param facts 输入事实数组
 * @param fact_count 事实数量
 * @param result 推理结果输出
 * @return int 成功返回0，失败返回-1
 */
int uncertainty_engine_reason(UncertaintyEngine* engine, const char** facts,
                               size_t fact_count, URResult* result);

/**
 * @brief 与逻辑推理引擎集成执行推理
 *
 * 先将事实送入不确定性推理，再将结果传递给逻辑推理引擎。
 *
 * @param uncertainty_engine 不确定性推理引擎句柄
 * @param logic_engine 逻辑推理引擎句柄
 * @param facts 输入事实数组
 * @param fact_count 事实数量
 * @return LogicInferenceResult* 推理结果，调用者负责释放
 */
LogicInferenceResult* uncertainty_integrated_reason(UncertaintyEngine* uncertainty_engine,
                                                     ReasoningEngine* logic_engine,
                                                     const char** facts, size_t fact_count);

/**
 * @brief 释放不确定性推理结果
 *
 * @param result 推理结果
 */
void uncertainty_result_free(URResult* result);

/**
 * @brief 重置不确定性推理引擎
 *
 * @param engine 不确定性推理引擎句柄
 */
void uncertainty_engine_reset(UncertaintyEngine* engine);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_UNCERTAINTY_REASONING_H */
