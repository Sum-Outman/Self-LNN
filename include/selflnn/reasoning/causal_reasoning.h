/**
 * @file causal_reasoning.h
 * @brief 因果推理引擎接口
 * 
 * 因果推理引擎完整实现，支持：
 * 1. 因果图构建与推理（Causal Graph）
 * 2. 因果发现算法（PC算法、FCI算法）
 * 3. 因果效应估计（干预、反事实推理）
 * 4. 时间因果分析（Granger因果、转移熵）
 * 5. 结构因果模型（SCM）
 * 6. 因果强化学习（Causal RL）
 * 7. 因果知识图谱（Causal Knowledge Graph）
 * 8. 多变量因果分析
 * 
 *  ，提供完整的因果推理算法。
 */

#ifndef SELFLNN_CAUSAL_REASONING_H
#define SELFLNN_CAUSAL_REASONING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 因果推理算法类型
 */
typedef enum {
    CAUSAL_REASONING_PC_ALGORITHM = 0,      /**< PC算法（Peter-Clark算法） */
    CAUSAL_REASONING_FCI_ALGORITHM = 1,     /**< FCI算法（Fast Causal Inference） */
    CAUSAL_REASONING_GRANGER = 2,           /**< Granger因果分析 */
    CAUSAL_REASONING_TRANSFER_ENTROPY = 3,  /**< 转移熵因果分析 */
    CAUSAL_REASONING_STRUCTURAL = 4,        /**< 结构因果模型（SCM） */
    CAUSAL_REASONING_BAYESIAN = 5,          /**< 贝叶斯因果网络 */
    CAUSAL_REASONING_COUNTERFACTUAL = 6,    /**< 反事实推理 */
    CAUSAL_REASONING_GES_ALGORITHM = 7,     /**< GES算法（Greedy Equivalence Search） */
    CAUSAL_REASONING_LINGAM_ALGORITHM = 8   /**< LiNGAM算法（线性非高斯模型） */
} CausalReasoningAlgorithm;

/**
 * @brief 因果图节点类型
 */
typedef enum {
    CAUSAL_NODE_OBSERVED = 0,      /**< 观测节点 */
    CAUSAL_NODE_LATENT = 1,        /**< 隐变量节点 */
    CAUSAL_NODE_INTERVENTION = 2,  /**< 干预节点 */
    CAUSAL_NODE_OUTCOME = 3        /**< 结果节点 */
} CausalNodeType;

/**
 * @brief 因果边类型
 */
typedef enum {
    CAUSAL_EDGE_DIRECTED = 0,      /**< 有向边（因果方向确定） */
    CAUSAL_EDGE_BIDIRECTED = 1,    /**< 双向边（存在隐变量） */
    CAUSAL_EDGE_UNDIRECTED = 2,    /**< 无向边（方向不确定） */
    CAUSAL_EDGE_PROBABILISTIC = 3  /**< 概率性边（不确定因果） */
} CausalEdgeType;

/**
 * @brief 因果推理配置
 */
typedef struct {
    CausalReasoningAlgorithm algorithm;  /**< 因果推理算法 */
    float significance_level;            /**< 显著性水平（默认0.05） */
    int max_conditioning_set_size;       /**< 最大条件集大小 */
    int enable_temporal_analysis;        /**< 是否启用时间分析 */
    int enable_counterfactual;           /**< 是否启用反事实推理 */
    int max_iterations;                  /**< 最大迭代次数 */
    float confidence_threshold;          /**< 置信度阈值 */
} CausalReasoningConfig;

/**
 * @brief 因果图节点
 */
typedef struct {
    int node_id;                    /**< 节点ID */
    CausalNodeType node_type;       /**< 节点类型 */
    float* node_features;           /**< 节点特征向量 */
    size_t feature_size;            /**< 特征大小 */
    char node_name[64];             /**< 节点名称 */
    float node_confidence;          /**< 节点置信度 */
} CausalNode;

/**
 * @brief 因果图边
 */
typedef struct {
    int source_node_id;             /**< 源节点ID */
    int target_node_id;             /**< 目标节点ID */
    CausalEdgeType edge_type;       /**< 边类型 */
    float edge_strength;            /**< 边强度（因果效应大小） */
    float edge_confidence;          /**< 边置信度 */
    float* edge_parameters;         /**< 边参数（如条件概率） */
    size_t param_size;              /**< 参数大小 */
} CausalEdge;

/**
 * @brief 因果推理结果
 */
typedef struct {
    int* causal_path;               /**< 因果路径（节点ID数组） */
    size_t path_length;             /**< 路径长度 */
    float causal_effect;            /**< 因果效应估计值 */
    float effect_confidence;        /**< 效应置信度 */
    float* counterfactual_values;   /**< 反事实值数组 */
    size_t counterfactual_size;     /**< 反事实值大小 */
    char explanation[256];          /**< 因果解释 */
    int requires_intervention;      /**< 是否需要干预 */
} CausalReasoningResult;

/**
 * @brief 因果推理引擎句柄
 */
typedef struct CausalReasoningEngine CausalReasoningEngine;

/**
 * @brief 创建因果推理引擎
 * 
 * @param config 因果推理配置
 * @return CausalReasoningEngine* 因果推理引擎句柄，失败返回NULL
 */
CausalReasoningEngine* causal_reasoning_engine_create(const CausalReasoningConfig* config);

/**
 * @brief 释放因果推理引擎
 * 
 * @param engine 因果推理引擎句柄
 */
void causal_reasoning_engine_free(CausalReasoningEngine* engine);

/**
 * @brief 构建因果图
 * 
 * @param engine 因果推理引擎句柄
 * @param data 观测数据矩阵（行：样本，列：变量）
 * @param num_samples 样本数量
 * @param num_variables 变量数量
 * @param variable_names 变量名称数组
 * @return int 成功返回0，失败返回-1
 */
int causal_reasoning_build_graph(CausalReasoningEngine* engine,
                                const float* data, size_t num_samples, size_t num_variables,
                                const char** variable_names);

/**
 * @brief 执行因果推理
 * 
 * @param engine 因果推理引擎句柄
 * @param cause_indices 原因变量索引数组
 * @param num_causes 原因数量
 * @param effect_indices 结果变量索引数组
 * @param num_effects 结果数量
 * @param result 推理结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int causal_reasoning_infer(CausalReasoningEngine* engine,
                          const int* cause_indices, size_t num_causes,
                          const int* effect_indices, size_t num_effects,
                          CausalReasoningResult* result);

/**
 * @brief 估计因果效应
 * 
 * @param engine 因果推理引擎句柄
 * @param treatment_index 处理变量索引
 * @param outcome_index 结果变量索引
 * @param confounder_indices 混淆变量索引数组
 * @param num_confounders 混淆变量数量
 * @return float 因果效应估计值，失败返回0.0f
 */
float causal_reasoning_estimate_effect(CausalReasoningEngine* engine,
                                      int treatment_index, int outcome_index,
                                      const int* confounder_indices, size_t num_confounders);

/**
 * @brief 执行反事实推理
 * 
 * @param engine 因果推理引擎句柄
 * @param factual_data 事实数据
 * @param factual_size 事实数据大小
 * @param intervention_value 干预值
 * @param counterfactual_result 反事实结果输出缓冲区
 * @param max_result_size 最大结果大小
 * @return int 成功返回0，失败返回-1
 */
int causal_reasoning_counterfactual(CausalReasoningEngine* engine,
                                   const float* factual_data, size_t factual_size,
                                   float intervention_value,
                                   float* counterfactual_result, size_t max_result_size);

/**
 * @brief 获取因果图节点
 * 
 * @param engine 因果推理引擎句柄
 * @param node_index 节点索引
 * @param node 节点输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int causal_reasoning_get_node(CausalReasoningEngine* engine,
                             int node_index, CausalNode* node);

/**
 * @brief 获取因果图边
 * 
 * @param engine 因果推理引擎句柄
 * @param source_index 源节点索引
 * @param target_index 目标节点索引
 * @param edge 边输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int causal_reasoning_get_edge(CausalReasoningEngine* engine,
                             int source_index, int target_index, CausalEdge* edge);

/**
 * @brief 执行因果发现（从数据中发现因果结构）
 * 
 * @param engine 因果推理引擎句柄
 * @param data 观测数据矩阵
 * @param num_samples 样本数量
 * @param num_variables 变量数量
 * @param discovered_edges 发现的边输出缓冲区
 * @param max_edges 最大边数量
 * @return int 成功返回发现的边数量，失败返回-1
 */
int causal_reasoning_discover(CausalReasoningEngine* engine,
                             const float* data, size_t num_samples, size_t num_variables,
                             CausalEdge* discovered_edges, size_t max_edges);

/**
 * @brief 验证因果假设
 * 
 * @param engine 因果推理引擎句柄
 * @param cause_index 假设原因索引
 * @param effect_index 假设结果索引
 * @param test_data 测试数据
 * @param num_test_samples 测试样本数量
 * @return float 假设验证分数（0-1），越高表示假设越可能成立
 */
float causal_reasoning_validate_hypothesis(CausalReasoningEngine* engine,
                                          int cause_index, int effect_index,
                                          const float* test_data, size_t num_test_samples);

/**
 * @brief 获取因果推理引擎配置
 * 
 * @param engine 因果推理引擎句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int causal_reasoning_engine_get_config(const CausalReasoningEngine* engine,
                                      CausalReasoningConfig* config);

/**
 * @brief 设置因果推理引擎配置
 * 
 * @param engine 因果推理引擎句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int causal_reasoning_engine_set_config(CausalReasoningEngine* engine,
                                      const CausalReasoningConfig* config);

/**
 * @brief 逆概率加权(IPW)因果效应估计
 *
 * 使用倾向性评分的逆概率加权估计平均处理效应(ATE)。
 * 先估计每个样本接受处理的概率P(T=1|X)，再用逆概率加权计算:
 * ATE = E[Y(1)] - E[Y(0)], 其中权重 = T/P + (1-T)/(1-P)
 *
 * @param engine 因果推理引擎句柄
 * @param data 观测数据矩阵 [样本×变量] 行优先
 * @param num_samples 样本数量
 * @param num_variables 变量数量
 * @param treatment_idx 处理变量列索引
 * @param outcome_idx 结果变量列索引
 * @param covariate_indices 协变量索引数组（用于倾向性评分模型）
 * @param num_covariates 协变量数量
 * @return float ATE估计值，失败返回NaN
 */
float causal_reasoning_estimate_ipw(CausalReasoningEngine* engine,
                                    const float* data, size_t num_samples, size_t num_variables,
                                    int treatment_idx, int outcome_idx,
                                    const int* covariate_indices, size_t num_covariates);

/**
 * @brief 双重稳健估计(DR)因果效应估计
 *
 * 同时使用倾向性评分和结果回归模型，只要其中一个模型正确指定，
 * 估计就是一致的。DR估计量:
 * DR = 1/n Σ[μ1(X) - μ0(X) + T(Y-μ1(X))/P - (1-T)(Y-μ0(X))/(1-P)]
 *
 * @param engine 因果推理引擎句柄
 * @param data 观测数据矩阵 [样本×变量] 行优先
 * @param num_samples 样本数量
 * @param num_variables 变量数量
 * @param treatment_idx 处理变量列索引
 * @param outcome_idx 结果变量列索引
 * @param covariate_indices 协变量索引数组
 * @param num_covariates 协变量数量
 * @return float ATE估计值，失败返回NaN
 */
float causal_reasoning_estimate_doubly_robust(CausalReasoningEngine* engine,
                                              const float* data, size_t num_samples, size_t num_variables,
                                              int treatment_idx, int outcome_idx,
                                              const int* covariate_indices, size_t num_covariates);

/**
 * @brief 工具变量(IV)两阶段最小二乘因果效应估计
 *
 * 使用工具变量处理未观测混淆变量。两阶段最小二乘(2SLS):
 * 第一阶段: T = Z*π + X*β + ε (工具变量Z预测处理T)
 * 第二阶段: Y = T̂*ρ + X*γ + δ (用预测值T̂回归结果Y)
 * IV效应 = ρ
 *
 * @param engine 因果推理引擎句柄
 * @param data 观测数据矩阵 [样本×变量] 行优先
 * @param num_samples 样本数量
 * @param num_variables 变量数量
 * @param treatment_idx 处理变量列索引
 * @param outcome_idx 结果变量列索引
 * @param instrument_idx 工具变量列索引
 * @param covariate_indices 额外协变量索引数组（可选，传NULL和0表示无）
 * @param num_covariates 协变量数量
 * @return float IV因果效应估计值，失败返回NaN
 */
float causal_reasoning_estimate_instrumental_variable(CausalReasoningEngine* engine,
                                                      const float* data, size_t num_samples, size_t num_variables,
                                                      int treatment_idx, int outcome_idx,
                                                      int instrument_idx,
                                                      const int* covariate_indices, size_t num_covariates);

/**
 * @brief 断点回归设计(RDD)因果效应估计
 *
 * 利用处理变量分配规则中的断点进行因果效应估计。
 * 假设在断点c附近，个体是"几乎随机"分配处理。
 *
 * 锐断点回归(Sharp RDD):
 *   处理T在X≥c时确定等于1，X<c时等于0。
 *   τ = lim_{x→c+} E[Y|X=x] - lim_{x→c-} E[Y|X=x]
 *   使用局部线性回归+三角核加权估计边界期望。
 *
 * 模糊断点回归(Fuzzy RDD):
 *   处理概率在断点处跳跃但不完全确定。
 *   使用两阶段方法: 断点指示作为处理变量的工具变量
 *   τ_Fuzzy = τ_Reduced / τ_FirstStage
 *
 * 带宽选择: Imbens-Kalyanaraman(2012)最优带宽或用户指定
 *
 * @param engine 因果推理引擎句柄
 * @param data 观测数据矩阵 [样本×变量] 行优先
 * @param num_samples 样本数量
 * @param num_variables 变量数量
 * @param running_idx 分配变量(Assignment/Running variable)列索引
 * @param treatment_idx 处理变量列索引
 * @param outcome_idx 结果变量列索引
 * @param cutoff 断点位置
 * @param is_fuzzy 是否为模糊断点回归(非0:模糊,0:锐)
 * @param bandwidth 带宽(传≤0则使用IK最优带宽估计)
 * @return float RDD因果效应估计值，失败返回NaN
 */
float causal_reasoning_estimate_rdd(CausalReasoningEngine* engine,
                                    const float* data, size_t num_samples, size_t num_variables,
                                    int running_idx, int treatment_idx, int outcome_idx,
                                    float cutoff, int is_fuzzy, float bandwidth);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_CAUSAL_REASONING_H */