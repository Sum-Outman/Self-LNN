/**
 * @file reasoning.h
 * @brief 推理引擎接口
 * 
 * 推理引擎核心接口，支持多种推理模式。
 */

#ifndef SELFLNN_REASONING_H
#define SELFLNN_REASONING_H

#include <stddef.h>
#include "selflnn/reasoning/causal_reasoning.h"
#include "selflnn/core/lnn.h"

/* 前置声明知识库类型 */
struct KnowledgeBase;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 推理模式枚举
 */
typedef enum {
    REASONING_DEDUCTIVE = 0,   /**< 演绎推理 */
    REASONING_INDUCTIVE = 1,   /**< 归纳推理 */
    REASONING_ABDUCTIVE = 2,   /**< 溯因推理 */
    REASONING_ANALOGICAL = 3,  /**< 类比推理 */
    REASONING_CAUSAL = 4       /**< 因果推理 */
} ReasoningMode;

/**
 * @brief 推理引擎配置
 */
typedef struct {
    ReasoningMode default_mode;   /**< 默认推理模式 */
    int max_iterations;           /**< 最大迭代次数 */
    float confidence_threshold;   /**< 置信度阈值 */
    int enable_multimodal;        /**< 是否启用多模态推理 */
} ReasoningConfig;

/**
 * @brief 推理引擎句柄
 */
typedef struct ReasoningEngine ReasoningEngine;

/**
 * @brief 创建推理引擎
 * 
 * @param config 推理配置
 * @return ReasoningEngine* 推理引擎句柄，失败返回NULL
 */
ReasoningEngine* reasoning_engine_create(const ReasoningConfig* config);

/**
 * @brief 释放推理引擎
 * 
 * @param engine 推理引擎句柄
 */
void reasoning_engine_free(ReasoningEngine* engine);

/**
 * @brief 执行推理
 * 
 * @param engine 推理引擎句柄
 * @param premises 前提条件数组
 * @param num_premises 前提条件数量
 * @param conclusion 结论输出缓冲区
 * @param max_conclusion_size 结论最大大小
 * @param mode 推理模式
 * @return int 成功返回0，失败返回-1
 */
int reasoning_infer(ReasoningEngine* engine,
                   const float* premises, size_t num_premises,
                   float* conclusion, size_t max_conclusion_size,
                   ReasoningMode mode);

/**
 * @brief 评估推理结果置信度
 * 
 * @param engine 推理引擎句柄
 * @param conclusion 结论
 * @param conclusion_size 结论大小
 * @param premises 前提条件数组
 * @param num_premises 前提条件数量
 * @return float 置信度分数 (0-1)
 */
float reasoning_evaluate_confidence(ReasoningEngine* engine,
                                   const float* conclusion, size_t conclusion_size,
                                   const float* premises, size_t num_premises);

/**
 * @brief 多模态推理
 * 
 * @param engine 推理引擎句柄
 * @param visual_data 视觉数据 (可为NULL)
 * @param visual_size 视觉数据大小
 * @param audio_data 音频数据 (可为NULL)
 * @param audio_size 音频数据大小
 * @param text_data 文本数据 (可为NULL)
 * @param text_size 文本数据大小
 * @param conclusion 结论输出缓冲区
 * @param max_conclusion_size 结论最大大小
 * @return int 成功返回0，失败返回-1
 */
int reasoning_multimodal_infer(ReasoningEngine* engine,
                              const float* visual_data, size_t visual_size,
                              const float* audio_data, size_t audio_size,
                              const float* text_data, size_t text_size,
                              float* conclusion, size_t max_conclusion_size);

/**
 * @brief 类比推理
 * 
 * @param engine 推理引擎句柄
 * @param source 源域数据
 * @param source_size 源域数据大小
 * @param target 目标域数据
 * @param target_size 目标域数据大小
 * @param mapping 映射输出缓冲区
 * @param max_mapping_size 映射最大大小
 * @return int 成功返回映射数，失败返回-1
 */
int reasoning_analogical_map(ReasoningEngine* engine,
                            const float* source, size_t source_size,
                            const float* target, size_t target_size,
                            float* mapping, size_t max_mapping_size);

/**
 * @brief 因果推理
 * 
 * @param engine 推理引擎句柄
 * @param cause 原因数据
 * @param cause_size 原因数据大小
 * @param effect 结果数据
 * @param effect_size 结果数据大小
 * @param causal_strength 因果强度输出缓冲区
 * @return float 因果强度分数 (0-1)
 */
float reasoning_causal_infer(ReasoningEngine* engine,
                            const float* cause, size_t cause_size,
                            const float* effect, size_t effect_size,
                            float* causal_strength);

/**
 * @brief 获取推理引擎配置
 * 
 * @param engine 推理引擎句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int reasoning_engine_get_config(const ReasoningEngine* engine, ReasoningConfig* config);

/**
 * @brief 设置推理引擎配置
 * 
 * @param engine 推理引擎句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int reasoning_engine_set_config(ReasoningEngine* engine, const ReasoningConfig* config);

/**
 * @brief 重置推理引擎
 * 
 * @param engine 推理引擎句柄
 */
void reasoning_engine_reset(ReasoningEngine* engine);

/**
 * @brief 构建因果图
 * 
 * @param engine 推理引擎句柄
 * @param data 观测数据矩阵（行：样本，列：变量）
 * @param num_samples 样本数量
 * @param num_variables 变量数量
 * @param variable_names 变量名称数组
 * @return int 成功返回0，失败返回-1
 */
int reasoning_build_causal_graph(ReasoningEngine* engine,
                                const float* data, size_t num_samples, size_t num_variables,
                                const char** variable_names);

/**
 * @brief 因果推理（高级接口，使用因果图）
 * 
 * @param engine 推理引擎句柄
 * @param cause_indices 原因变量索引数组
 * @param num_causes 原因数量
 * @param effect_indices 结果变量索引数组
 * @param num_effects 结果数量
 * @param result 推理结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int reasoning_causal_infer_ex(ReasoningEngine* engine,
                             const int* cause_indices, size_t num_causes,
                             const int* effect_indices, size_t num_effects,
                             CausalReasoningResult* result);

/**
 * @brief 估计因果效应
 * 
 * @param engine 推理引擎句柄
 * @param treatment_index 处理变量索引
 * @param outcome_index 结果变量索引
 * @param confounder_indices 混淆变量索引数组
 * @param num_confounders 混淆变量数量
 * @return float 因果效应估计值
 */
float reasoning_estimate_causal_effect(ReasoningEngine* engine,
                                      int treatment_index, int outcome_index,
                                      const int* confounder_indices, size_t num_confounders);

/**
 * @brief 执行反事实推理
 * 
 * @param engine 推理引擎句柄
 * @param factual_data 事实数据
 * @param factual_size 事实数据大小
 * @param intervention_value 干预值
 * @param counterfactual_result 反事实结果输出缓冲区
 * @param max_result_size 最大结果大小
 * @return int 成功返回0，失败返回-1
 */
int reasoning_counterfactual_infer(ReasoningEngine* engine,
                                  const float* factual_data, size_t factual_size,
                                  float intervention_value,
                                  float* counterfactual_result, size_t max_result_size);

/**
 * @brief 执行因果发现
 * 
 * @param engine 推理引擎句柄
 * @param data 观测数据矩阵
 * @param num_samples 样本数量
 * @param num_variables 变量数量
 * @param discovered_edges 发现的边输出缓冲区
 * @param max_edges 最大边数量
 * @return int 成功返回发现的边数量，失败返回-1
 */
int reasoning_discover_causal_structure(ReasoningEngine* engine,
                                       const float* data, size_t num_samples, size_t num_variables,
                                       CausalEdge* discovered_edges, size_t max_edges);

/**
 * @brief 验证因果假设
 * 
 * @param engine 推理引擎句柄
 * @param cause_index 假设原因索引
 * @param effect_index 假设结果索引
 * @param test_data 测试数据
 * @param num_test_samples 测试样本数量
 * @return float 假设验证分数（0-1）
 */
float reasoning_validate_causal_hypothesis(ReasoningEngine* engine,
                                          int cause_index, int effect_index,
                                          const float* test_data, size_t num_test_samples);

/**
 * @brief 关联知识库到推理引擎
 *
 * 将外部知识库连接到推理引擎，使推理过程可以使用知识库中的事实和规则，
 * 并将推理结果回写到知识库。
 *
 * @param engine 推理引擎句柄
 * @param kb 知识库指针（可传NULL解除关联）
 * @return int 成功返回0，失败返回-1
 */
int reasoning_engine_set_knowledge_base(ReasoningEngine* engine, struct KnowledgeBase* kb);

/**
 * @brief 获取关联的知识库
 *
 * @param engine 推理引擎句柄
 * @return struct KnowledgeBase* 知识库指针，未关联时返回NULL
 */
struct KnowledgeBase* reasoning_engine_get_knowledge_base(const ReasoningEngine* engine);

/**
 * @brief 使用知识库增强推理
 *
 * 结合外部知识库中的事实和规则执行推理，推理过程中自动查询知识库获取
 * 相关前提和规则，并将高置信度结论自动写入知识库。
 *
 * @param engine 推理引擎句柄
 * @param premises 前提条件数组
 * @param num_premises 前提条件数量
 * @param conclusion 结论输出缓冲区
 * @param max_conclusion_size 结论最大大小
 * @param mode 推理模式
 * @param knowledge_weight 知识库影响权重 (0.0=不使用知识库, 1.0=完全依赖知识库)
 * @return int 成功返回0，失败返回-1
 */
int reasoning_infer_with_knowledge(ReasoningEngine* engine,
                                  const float* premises, size_t num_premises,
                                  float* conclusion, size_t max_conclusion_size,
                                  ReasoningMode mode,
                                  float knowledge_weight);

/**
 * @brief 同步知识库到推理引擎内部表示
 *
 * 将外部知识库中的规则和事实同步到推理引擎的内部知识库缓冲区，
 * 确保推理引擎使用最新的知识库数据。
 *
 * @param engine 推理引擎句柄
 * @return int 成功返回同步的条目数，失败返回-1
 */
int reasoning_sync_knowledge(ReasoningEngine* engine);

/**
 * @brief P0-009: 检查推理引擎是否已从真实知识库同步数据
 * 
 * 引擎创建时内部知识缓冲区为零值占位，需通过
 * reasoning_engine_set_knowledge_base 关联真实知识库后自动同步。
 * 此函数检查引擎是否已加载真实知识数据。
 *
 * @param engine 推理引擎句柄
 * @return int 1=知识就绪可推理, 0=未同步/空白状态
 */
int reasoning_engine_is_knowledge_ready(const ReasoningEngine* engine);

/**
 * @brief 关联液态神经网络实例到推理引擎
 *
 * 将LNN实例连接到推理引擎，使推理引擎可以利用液态神经网络的
 * 连续动态系统进行推理过程的状态演化。
 *
 * @param engine 推理引擎句柄
 * @param lnn 液态神经网络实例指针（可传NULL解除关联）
 * @return int 成功返回0，失败返回-1
 */
int reasoning_engine_set_lnn(ReasoningEngine* engine, LNN* lnn);

/* ========== P2-2: 因果模型驱动的规划 ========== */

/**
 * @brief 因果模型变量类型
 */
typedef enum {
    CAUSAL_VAR_ENDOGENOUS = 0,      /**< 内生变量 */
    CAUSAL_VAR_EXOGENOUS = 1,       /**< 外生变量 */
    CAUSAL_VAR_LATENT = 2,          /**< 隐变量 */
    CAUSAL_VAR_INTERVENTION = 3     /**< 干预变量 */
} CausalVariableType;

/**
 * @brief 因果模型变量
 */
typedef struct {
    int var_id;                     /**< 变量ID */
    CausalVariableType var_type;    /**< 变量类型 */
    char name[64];                  /**< 变量名称 */
    float current_value;            /**< 当前值 */
    float* parent_weights;          /**< 父节点权重（结构方程系数） */
    int* parent_ids;                /**< 父节点ID数组 */
    size_t num_parents;             /**< 父节点数量 */
    float noise_mean;               /**< 噪声均值 */
    float noise_std;                /**< 噪声标准差 */
    float intervention_value;       /**< 干预值（当var_type=INTERVENTION时有效） */
    int is_intervened;              /**< 是否正在被干预 */
} CausalVariable;

/**
 * @brief 结构因果模型（SCM）
 */
typedef struct {
    CausalVariable* variables;      /**< 变量数组 */
    size_t num_variables;           /**< 变量数量 */
    size_t variable_capacity;       /**< 变量容量 */
    float** adjacency_matrix;       /**< 邻接矩阵（因果结构） */
} StructuralCausalModel;

/**
 * @brief do-演算操作类型
 */
typedef enum {
    DO_CALCULUS_DO = 0,            /**< P(y|do(x)) 干预操作 */
    DO_CALCULUS_CONDITION = 1,     /**< P(y|do(x),z) 条件干预 */
    DO_CALCULUS_DO_CONDITION = 2,  /**< P(y|do(x),do(z)) 多重干预 */
    DO_CALCULUS_BACKDOOR = 3,      /**< 后门准则调整 */
    DO_CALCULUS_FRONTDOOR = 4,     /**< 前门准则调整 */
    DO_CALCULUS_IV = 5,            /**< 工具变量估计 */
    DO_CALCULUS_MEDIATION = 6      /**< 中介分析 */
} DoCalculusOperator;

/**
 * @brief 因果效应估计结果
 */
typedef struct {
    float causal_effect;            /**< 因果效应估计值 */
    float effect_variance;          /**< 效应方差 */
    float confidence_interval_low;  /**< 置信区间下限（95%） */
    float confidence_interval_high; /**< 置信区间上限（95%） */
    int is_significant;             /**< 是否统计显著 */
    char explanation[256];          /**< 估计方法解释 */
} CausalEffectEstimate;

/**
 * @brief 中介分析结果
 */
typedef struct {
    float total_effect;             /**< 总效应 */
    float direct_effect;            /**< 直接效应 */
    float indirect_effect;          /**< 间接效应（中介效应） */
    float mediation_proportion;     /**< 中介比例（间接/总效应） */
    float direct_significance;      /**< 直接效应显著性 */
    float indirect_significance;    /**< 间接效应显著性 */
} MediationResult;

/**
 * @brief 创建结构因果模型
 *
 * @param max_variables 最大变量数
 * @return StructuralCausalModel* 模型句柄，失败返回NULL
 */
StructuralCausalModel* causal_model_create(size_t max_variables);

/**
 * @brief 释放结构因果模型
 *
 * @param model 模型句柄
 */
void causal_model_free(StructuralCausalModel* model);

/**
 * @brief 添加变量到因果模型
 *
 * @param model 模型句柄
 * @param name 变量名称
 * @param var_type 变量类型
 * @param noise_mean 噪声均值
 * @param noise_std 噪声标准差
 * @return int 成功返回变量ID，失败返回-1
 */
int causal_model_add_variable(StructuralCausalModel* model,
                              const char* name,
                              CausalVariableType var_type,
                              float noise_mean, float noise_std);

/**
 * @brief 设置结构方程（定义因果关系）
 *
 * 设置变量var_id的父节点和权重系数，即 var_id = Σ(weight_i × parent_i) + noise
 *
 * @param model 模型句柄
 * @param var_id 目标变量ID
 * @param parent_ids 父节点ID数组
 * @param parent_weights 父节点权重数组
 * @param num_parents 父节点数量
 * @return int 成功返回0，失败返回-1
 */
int causal_model_set_structural_equation(StructuralCausalModel* model,
                                        int var_id,
                                        const int* parent_ids,
                                        const float* parent_weights,
                                        size_t num_parents);

/**
 * @brief 执行do-演算干预 P(y|do(x))
 *
 * 对变量x施加干预（固定为x_value），观测对变量y的因果效应。
 *
 * @param model 模型句柄
 * @param x_var_id 干预变量ID
 * @param x_value 干预值
 * @param y_var_id 结果变量ID
 * @param result 效应估计结果输出
 * @return int 成功返回0，失败返回-1
 */
int causal_model_do_intervention(StructuralCausalModel* model,
                                 int x_var_id, float x_value,
                                 int y_var_id,
                                 CausalEffectEstimate* result);

/**
 * @brief 后门准则调整
 *
 * 在给定混淆变量集Z的条件下，估计X对Y的因果效应。
 * 公式: P(y|do(x)) = Σ_z P(y|x,z) P(z)
 *
 * @param model 模型句柄
 * @param treatment_id 处理变量ID
 * @param outcome_id 结果变量ID
 * @param backdoor_ids 后门变量ID数组（混淆变量）
 * @param num_backdoors 后门变量数量
 * @param result 效应估计结果输出
 * @return int 成功返回0，失败返回-1
 */
int causal_model_backdoor_adjustment(StructuralCausalModel* model,
                                     int treatment_id, int outcome_id,
                                     const int* backdoor_ids, size_t num_backdoors,
                                     CausalEffectEstimate* result);

/**
 * @brief 前门准则调整
 *
 * 当存在中介变量M时，通过前门准则估计X对Y的因果效应。
 * 公式: P(y|do(x)) = Σ_m P(m|x) Σ_x' P(y|x',m) P(x')
 *
 * @param model 模型句柄
 * @param treatment_id 处理变量ID
 * @param outcome_id 结果变量ID
 * @param mediator_id 中介变量ID
 * @param result 效应估计结果输出
 * @return int 成功返回0，失败返回-1
 */
int causal_model_frontdoor_adjustment(StructuralCausalModel* model,
                                      int treatment_id, int outcome_id,
                                      int mediator_id,
                                      CausalEffectEstimate* result);

/**
 * @brief 工具变量估计
 *
 * 使用工具变量Z估计X对Y的因果效应（存在未观测混淆时）。
 * 假设: Z→X→Y, Z⊥U, Z影响Y仅通过X
 *
 * @param model 模型句柄
 * @param instrument_id 工具变量ID
 * @param treatment_id 处理变量ID
 * @param outcome_id 结果变量ID
 * @param result 效应估计结果输出
 * @return int 成功返回0，失败返回-1
 */
int causal_model_instrumental_variable(StructuralCausalModel* model,
                                       int instrument_id,
                                       int treatment_id, int outcome_id,
                                       CausalEffectEstimate* result);

/**
 * @brief 中介分析
 *
 * 分解X对Y的总效应为直接效应和间接效应（通过中介变量M）。
 *
 * @param model 模型句柄
 * @param treatment_id 处理变量ID
 * @param mediator_id 中介变量ID
 * @param outcome_id 结果变量ID
 * @param result 中介分析结果输出
 * @return int 成功返回0，失败返回-1
 */
int causal_model_mediation_analysis(StructuralCausalModel* model,
                                    int treatment_id, int mediator_id, int outcome_id,
                                    MediationResult* result);

/**
 * @brief 因果模型驱动的规划
 *
 * 使用因果模型生成干预策略，依据因果效应估计选择最优行动。
 * 核心流程：
 * 1. 对每个候选行动（干预变量）执行do-演算
 * 2. 估计对目标变量的因果效应
 * 3. 选择因果效应最大的行动方案
 * 4. 生成包含因果解释的规划步骤
 *
 * @param model 模型句柄
 * @param action_ids 候选行动变量ID数组
 * @param num_actions 行动数量
 * @param target_ids 目标变量ID数组
 * @param num_targets 目标数量
 * @param plan 规划输出缓冲区（每步为[action_id, target_id, effect, confidence]）
 * @param max_plan_size 规划最大步数
 * @return int 成功返回规划步数，失败返回-1
 */
int causal_model_driven_plan(StructuralCausalModel* model,
                             const int* action_ids, size_t num_actions,
                             const int* target_ids, size_t num_targets,
                             float* plan, size_t max_plan_size);

/**
 * @brief 反事实推理（因果模型版）
 *
 * 基于事实观测和干预假设，计算反事实结果。
 * 公式: Y_{X=x}(U) = f_Y(x, U_Y) 其中U是外生变量
 *
 * @param model 模型句柄
 * @param factual_data 事实观测数据
 * @param factual_size 事实数据大小
 * @param counterfactual_intervention 反事实干预假设
 * @param intervened_var_id 被干预变量ID
 * @param counterfactual_result 反事实结果输出缓冲区
 * @param max_result_size 结果大小
 * @return int 成功返回0，失败返回-1
 */
int causal_model_counterfactual(StructuralCausalModel* model,
                                const float* factual_data, size_t factual_size,
                                float counterfactual_intervention,
                                int intervened_var_id,
                                float* counterfactual_result, size_t max_result_size);

/**
 * @brief 构建规划因果图
 *
 * 从分层规划系统中提取任务结构，构建因果模型描述规划变量间的因果关系。
 * 支持将HTN分解结果自动转化为结构因果模型。
 *
 * @param model 模型句柄
 * @param htn_domain HTN规划域（可为NULL）
 * @param task_variables 任务变量描述
 * @param num_task_vars 任务变量数量
 * @param goal_description 目标描述向量
 * @param goal_size 目标大小
 * @return int 成功返回0，失败返回-1
 */
int causal_model_build_plan_graph(StructuralCausalModel* model,
                                  const void* htn_domain,
                                  const float* task_variables, size_t num_task_vars,
                                  const float* goal_description, size_t goal_size);

/* ========== P2-3: 贝叶斯推理网络 ========== */

/**
 * @brief 贝叶斯网络节点（随机变量）
 */
typedef struct {
    int node_id;                    /**< 节点ID */
    char name[64];                  /**< 节点名称 */
    int is_observed;                /**< 是否已观测到值 */
    int current_state;              /**< 当前状态索引（观测值） */
} BayesianNode;

/**
 * @brief 贝叶斯网络边（条件依赖关系）
 */
typedef struct {
    int parent_id;                  /**< 父节点ID */
    int child_id;                   /**< 子节点ID */
    int is_active;                  /**< 边是否激活 */
} BayesianEdge;

/**
 * @brief 条件概率表（CPD）
 */
typedef struct {
    int variable_id;                /**< 变量ID */
    int num_parents;                /**< 父节点数量 */
    int* parent_ids;                /**< 父节点ID数组 */
    float* probabilities;           /**< 条件概率表（展开的数组） */
    int num_states;                 /**< 本变量状态数 */
    int* parent_states;             /**< 每个父节点的状态数 */
    size_t table_size;              /**< 概率表总大小 */
} BayesianCPD;

/**
 * @brief 贝叶斯网络
 */
typedef struct {
    BayesianNode* nodes;            /**< 节点数组 */
    size_t num_nodes;               /**< 节点数量 */
    size_t node_capacity;           /**< 节点容量 */
    BayesianEdge* edges;            /**< 边数组 */
    size_t num_edges;               /**< 边数量 */
    size_t edge_capacity;           /**< 边容量 */
    BayesianCPD* cpds;              /**< 条件概率表数组 */
    size_t num_cpds;                /**< 条件概率表数量 */
    size_t cpd_capacity;            /**< 条件概率表容量 */
    float** adjacency_matrix;       /**< 邻接矩阵（用于图算法） */
} BayesianNetwork;

/**
 * @brief 贝叶斯查询结果
 */
typedef struct {
    float* probabilities;           /**< 各状态的后验概率 */
    size_t num_states;              /**< 状态数量 */
    float confidence;               /**< 查询置信度 */
} BayesianQueryResult;

/**
 * @brief 创建贝叶斯网络
 *
 * @param max_nodes 最大节点数
 * @return BayesianNetwork* 网络句柄，失败返回NULL
 */
BayesianNetwork* bayesian_network_create(size_t max_nodes);

/**
 * @brief 释放贝叶斯网络
 *
 * @param network 网络句柄
 */
void bayesian_network_free(BayesianNetwork* network);

/**
 * @brief 添加节点（随机变量）
 *
 * @param network 网络句柄
 * @param name 节点名称
 * @param num_states 状态数（至少2）
 * @return int 成功返回节点ID，失败返回-1
 */
int bayesian_network_add_node(BayesianNetwork* network,
                              const char* name,
                              int num_states);

/**
 * @brief 添加有向边（条件依赖）
 *
 * 添加 parent_id → child_id 的有向边，表示child的条件概率依赖于parent。
 *
 * @param network 网络句柄
 * @param parent_id 父节点ID
 * @param child_id 子节点ID
 * @return int 成功返回0，失败返回-1
 */
int bayesian_network_add_edge(BayesianNetwork* network,
                              int parent_id, int child_id);

/**
 * @brief 设置条件概率表（CPD）
 *
 * probabilities数组的排列顺序为：最外层索引对应最后一个父节点，
 * 最内层索引对应本变量的状态。即 P(V|Pa1,Pa2,...) 的维度为
 * [num_states_Pa1][num_states_Pa2]...[num_states_V]。
 *
 * @param network 网络句柄
 * @param var_id 变量ID
 * @param parent_ids 父节点ID数组（可为NULL表示无父节点）
 * @param num_parents 父节点数量
 * @param probabilities 概率值数组（总和应为1）
 * @param prob_size 概率数组大小
 * @return int 成功返回0，失败返回-1
 */
int bayesian_network_set_cpd(BayesianNetwork* network,
                             int var_id,
                             const int* parent_ids, int num_parents,
                             const float* probabilities, size_t prob_size);

/**
 * @brief 验证贝叶斯网络是否为有向无环图（DAG）
 *
 * @param network 网络句柄
 * @return int 是DAG返回1，否则返回0或-1（错误）
 */
int bayesian_network_is_valid(BayesianNetwork* network);

/**
 * @brief 贝叶斯推理 - 变量消除法（精确推理）
 *
 * 使用变量消除算法计算查询变量的后验概率分布 P(Query|Evidence)。
 * 适用于中小型网络（节点数<50）。
 *
 * @param network 网络句柄
 * @param query_vars 查询变量ID数组
 * @param num_queries 查询变量数量
 * @param evidence_vars 证据变量ID数组（可为NULL表示无证据）
 * @param evidence_vals 证据变量状态值数组
 * @param num_evidence 证据数量
 * @param result 推理结果输出（需调用bayesian_query_result_free释放）
 * @return int 成功返回0，失败返回-1
 */
int bayesian_network_variable_elimination(BayesianNetwork* network,
                                          const int* query_vars, int num_queries,
                                          const int* evidence_vars, const int* evidence_vals,
                                          int num_evidence,
                                          BayesianQueryResult* result);

/**
 * @brief 贝叶斯推理 - 吉布斯采样法（近似推理）
 *
 * 使用MCMC吉布斯采样算法近似计算后验概率分布。
 * 适用于大型复杂网络。
 *
 * @param network 网络句柄
 * @param query_vars 查询变量ID数组
 * @param num_queries 查询变量数量
 * @param evidence_vars 证据变量ID数组（可为NULL表示无证据）
 * @param evidence_vals 证据变量状态值数组
 * @param num_evidence 证据数量
 * @param num_samples 采样总数（建议至少10000）
 * @param burn_in 烧入期样本数（建议至少1000）
 * @param result 推理结果输出（需调用bayesian_query_result_free释放）
 * @return int 成功返回0，失败返回-1
 */
int bayesian_network_gibbs_sampling(BayesianNetwork* network,
                                    const int* query_vars, int num_queries,
                                    const int* evidence_vars, const int* evidence_vals,
                                    int num_evidence,
                                    int num_samples, int burn_in,
                                    BayesianQueryResult* result);

/**
 * @brief 贝叶斯推理 - 环状置信传播（近似推理）
 *
 * 使用Loopy Belief Propagation算法在图中传播置信信息。
 * 适用于树状或近树状网络。
 *
 * @param network 网络句柄
 * @param evidence_vars 证据变量ID数组
 * @param evidence_vals 证据变量状态值数组
 * @param num_evidence 证据数量
 * @param max_iterations 最大迭代次数
 * @param convergence_threshold 收敛阈值
 * @param result 推理结果（需调用bayesian_query_result_free释放）
 * @param result_size 结果数组大小（等于节点数）
 * @return int 成功返回0，失败返回-1
 */
int bayesian_network_belief_propagation(BayesianNetwork* network,
                                        const int* evidence_vars, const int* evidence_vals,
                                        int num_evidence,
                                        int max_iterations, float convergence_threshold,
                                        BayesianQueryResult* result, size_t result_size);

/**
 * @brief 释放贝叶斯查询结果
 *
 * @param result 查询结果
 */
void bayesian_query_result_free(BayesianQueryResult* result);

/**
 * @brief 关联贝叶斯网络到推理引擎
 *
 * @param engine 推理引擎句柄
 * @param network 贝叶斯网络指针（可传NULL解除关联）
 * @return int 成功返回0，失败返回-1
 */
int reasoning_engine_set_bayesian_network(ReasoningEngine* engine,
                                          BayesianNetwork* network);

/**
 * @brief 获取关联的贝叶斯网络
 *
 * @param engine 推理引擎句柄
 * @return BayesianNetwork* 贝叶斯网络指针，未关联时返回NULL
 */
BayesianNetwork* reasoning_engine_get_bayesian_network(const ReasoningEngine* engine);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_REASONING_H