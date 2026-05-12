/**
 * @file cfc_uncertainty_reasoning.h
 * @brief CfC不确定性推理引擎接口
 *
 * 基于CfC液态神经网络的不确定性推理系统，提供三种核心不确定性推理方法：
 * 1. CfC模糊逻辑推理 — 连续隶属度推理取代离散逻辑
 * 2. CfC概率逻辑推理 — 液态概率图模型
 * 3. CfC Dempster-Shafer推理 — 证据理论融合
 *
 * 100% 纯C实现，无第三方库依赖。
 */

#ifndef SELFLNN_CFC_UNCERTAINTY_REASONING_H
#define SELFLNN_CFC_UNCERTAINTY_REASONING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 配置常量
 * ============================================================================ */
#define CFC_UNCERTAIN_MAX_VARIABLES     65536   /**< 最大变量数 */
#define CFC_UNCERTAIN_MAX_RULES         262144  /**< 最大规则数 */
#define CFC_UNCERTAIN_MAX_EVIDENCE      1024    /**< 最大证据源数 */
#define CFC_UNCERTAIN_MAX_HYPOTHESES    4096    /**< 最大假设数 */
#define CFC_UNCERTAIN_MAX_DIM           512     /**< 最大推理维度 */

/* ============================================================================
 * 类型定义
 * ============================================================================ */

/**
 * @brief 不确定性推理算法类型
 */
typedef enum {
    CFC_UNCERTAIN_FUZZY_LOGIC = 0,          /**< CfC模糊逻辑推理 */
    CFC_UNCERTAIN_PROBABILISTIC_LOGIC,      /**< CfC概率逻辑推理 */
    CFC_UNCERTAIN_DEMPSTER_SHAFER,          /**< CfC Dempster-Shafer推理 */
    CFC_UNCERTAIN_HYBRID                    /**< 混合推理模式 */
} CfCUncertainAlgorithm;

/**
 * @brief 模糊隶属函数类型（A03.2.2.1）
 */
typedef enum {
    FUZZY_MF_GAUSSIAN = 0,                  /**< 高斯隶属函数: exp(-(x-c)^2/(2*σ^2)) */
    FUZZY_MF_TRIANGULAR,                    /**< 三角形隶属函数 */
    FUZZY_MF_TRAPEZOIDAL,                   /**< 梯形隶属函数 */
    FUZZY_MF_SIGMOID,                       /**< S型隶属函数: 1/(1+exp(-a*(x-c))) */
    FUZZY_MF_BELL,                          /**< 钟形隶属函数: 1/(1+|(x-c)/a|^(2*b)) */
    FUZZY_MF_LIQUID_CFC                     /**< CfC液态隶属函数（自适应演化） */
} FuzzyMembershipType;

/**
 * @brief 模糊推理去模糊化方法
 */
typedef enum {
    FUZZY_DEFUZZ_CENTROID = 0,              /**< 重心法 */
    FUZZY_DEFUZZ_BISECTOR,                  /**< 平分法 */
    FUZZY_DEFUZZ_MOM,                       /**< 最大平均值法 */
    FUZZY_DEFUZZ_LOM,                       /**< 最大最大值法 */
    FUZZY_DEFUZZ_SOM,                       /**< 最大最小值法 */
    FUZZY_DEFUZZ_CFC_LIQUID                 /**< CfC液态去模糊化 */
} FuzzyDefuzzMethod;

/**
 * @brief 概率逻辑推理类型（A03.2.2.2）
 */
typedef enum {
    PROB_LOGIC_BAYESIAN = 0,                /**< 贝叶斯概率逻辑 */
    PROB_LOGIC_MARKOV_LOGIC,                /**< 马尔可夫逻辑网络 */
    PROB_LOGIC_LIQUID_BN,                   /**< CfC液态贝叶斯网络 */
    PROB_LOGIC_CAUSAL_PROB                  /**< 因果概率逻辑 */
} ProbabilisticLogicType;

/**
 * @brief Dempster-Shafer融合规则（A03.2.2.3）
 */
typedef enum {
    DS_FUSE_DEMPSTER = 0,                   /**< Dempster组合规则 */
    DS_FUSE_YAGER,                          /**< Yager修改规则 */
    DS_FUSE_INAGAKI,                        /**< Inagaki统一组合 */
    DS_FUSE_DISCOUNTED,                     /**< 带折扣因子的组合 */
    DS_FUSE_CFC_LIQUID                      /**< CfC液态证据融合 */
} DSFusionRule;

/**
 * @brief 模糊变量
 */
typedef struct {
    int var_id;                             /**< 变量ID */
    char name[64];                          /**< 变量名称 */
    float min_value;                        /**< 最小值 */
    float max_value;                        /**< 最大值 */
    int num_terms;                          /**< 语言项数量 */
    char** term_names;                       /**< 语言项名称数组 */
    FuzzyMembershipType* term_types;         /**< 各语言项隶属函数类型 */
    float** term_params;                     /**< 各语言项隶属函数参数 [num_terms][3] */
} CfCUncertainFuzzyVar;

/**
 * @brief 模糊规则
 */
typedef struct {
    int rule_id;                            /**< 规则ID */
    float weight;                           /**< 规则权重 */
    int* antecedent_vars;                    /**< 前件变量ID数组 */
    int* antecedent_terms;                   /**< 前件语言项索引数组 */
    int antecedent_count;                   /**< 前件数量 */
    int consequent_var;                     /**< 后件变量ID */
    int consequent_term;                    /**< 后件语言项索引 */
    char description[256];                  /**< 规则描述 */
} CfCUncertainFuzzyRule;

/**
 * @brief 概率逻辑因子
 */
typedef struct {
    int factor_id;                          /**< 因子ID */
    int* variable_ids;                       /**< 变量ID数组 */
    int variable_count;                     /**< 变量数 */
    float* potential_values;                 /**< 势函数值数组 */
    size_t potential_size;                  /**< 势函数大小 */
    float weight;                           /**< 因子权重 */
} CfCUncertainProbFactor;

/**
 * @brief 证据结构（Dempster-Shafer）
 */
typedef struct {
    int hypothesis_id;                      /**< 假设ID */
    char name[64];                          /**< 假设名称 */
    float belief;                           /**< 信任度 Bel(A) */
    float plausibility;                     /**< 似然度 Pl(A) */
    float mass;                             /**< 基本概率分配 m(A) */
    float ignorance;                        /**< 无知度 */
} CfCUncertainEvidence;

/**
 * @brief 证据源
 */
typedef struct {
    int source_id;                          /**< 源ID */
    char source_name[64];                   /**< 源名称 */
    float reliability;                      /**< 源可靠性(0~1) */
    CfCUncertainEvidence* hypotheses;       /**< 假设数组 */
    int hypothesis_count;                   /**< 假设数 */
    float* mass_function;                    /**< 质量函数 [hypothesis_count] */
} CfCUncertainEvidenceSource;

/**
 * @brief 不确定性推理配置
 */
typedef struct {
    CfCUncertainAlgorithm algorithm;        /**< 推理算法 */
    FuzzyDefuzzMethod defuzz_method;         /**< 去模糊化方法 */
    ProbabilisticLogicType prob_type;       /**< 概率逻辑类型 */
    DSFusionRule ds_rule;                   /**< DS融合规则 */
    int max_iterations;                     /**< 最大迭代次数 */
    float convergence_threshold;            /**< 收敛阈值 */
    float default_uncertainty;              /**< 默认不确定度(0~1) */
    float learning_rate;                    /**< 参数学习率 */
    int enable_online_learning;             /**< 启用在线学习 */
    int enable_explanation;                 /**< 启用解释生成 */
    float cfc_tau;                          /**< CfC时间常数 */
    float cfc_dt;                           /**< CfC步长 */
    int cfc_steps;                          /**< CfC积分步数 */
} CfCUncertainConfig;

/**
 * @brief 不确定性推理状态
 */
typedef struct CfCUncertainReasoningState CfCUncertainReasoningState;

/* ============================================================================
 * A03.2.2.1 — CfC模糊逻辑推理核心API
 * ============================================================================ */

/**
 * @brief 创建不确定性推理引擎
 *
 * @param config 推理配置
 * @return CfCUncertainReasoningState* 成功返回状态指针，失败返回NULL
 */
CfCUncertainReasoningState* cfc_uncertain_create(const CfCUncertainConfig* config);

/**
 * @brief 销毁推理引擎
 *
 * @param state 推理状态指针
 */
void cfc_uncertain_destroy(CfCUncertainReasoningState* state);

/**
 * @brief 添加模糊变量
 *
 * @param state 推理状态
 * @param var 模糊变量定义
 * @return int 成功返回变量ID，失败返回-1
 */
int cfc_uncertain_add_fuzzy_var(CfCUncertainReasoningState* state,
                                 const CfCUncertainFuzzyVar* var);

/**
 * @brief 添加模糊规则
 *
 * @param state 推理状态
 * @param rule 模糊规则定义
 * @return int 成功返回规则ID，失败返回-1
 */
int cfc_uncertain_add_fuzzy_rule(CfCUncertainReasoningState* state,
                                  const CfCUncertainFuzzyRule* rule);

/**
 * @brief 计算隶属度
 *
 * 根据隶属函数类型计算输入值x的隶属度：
 * - FUZZY_MF_GAUSSIAN: exp(-(x-c)^2/(2*σ^2))
 * - FUZZY_MF_TRIANGULAR: max(0, min((x-a)/(b-a), (c-x)/(c-b)))
 * - FUZZY_MF_LIQUID_CFC: CfC ODE演化自适应隶属度
 *
 * @param mf_type 隶属函数类型
 * @param params 参数数组 [3] (如 [c, σ, 0])
 * @param x 输入值
 * @return float 隶属度(0~1)
 */
float cfc_uncertain_compute_membership(FuzzyMembershipType mf_type,
                                        const float* params, float x);

/**
 * @brief 执行模糊推理（Mamdani型）
 *
 * 步骤：
 * 1. 模糊化：计算每个规则前件的隶属度
 * 2. 规则评估：取前件隶属度的最小值/乘积作为规则激活度
 * 3. 聚合：将所有规则的后件隶属函数进行聚合
 * 4. 去模糊化：使用指定方法计算清晰输出值
 *
 * @param state 推理状态
 * @param inputs 输入变量值数组 [num_vars]
 * @param input_count 输入变量数
 * @param outputs 输出变量值数组 [num_vars]
 * @param output_count 输出变量数
 * @return int 成功返回0，失败返回-1
 */
int cfc_uncertain_fuzzy_infer(CfCUncertainReasoningState* state,
                               const float* inputs, int input_count,
                               float* outputs, int output_count);

/**
 * @brief CfC液态模糊推理（自适应隶属度演化）
 *
 * 使用CfC ODE演化隶属函数参数：
 *   dm/dt = -m/τ + f_cfc(m, x, θ)
 * 其中f_cfc是CfC液态门控函数，θ是隶属函数参数
 *
 * @param state 推理状态
 * @param inputs 输入值数组
 * @param input_count 输入数量
 * @param outputs 输出值数组
 * @param output_count 输出数量
 * @param evolution_steps 演化步数
 * @return int 成功返回0，失败返回-1
 */
int cfc_uncertain_liquid_fuzzy_infer(CfCUncertainReasoningState* state,
                                      const float* inputs, int input_count,
                                      float* outputs, int output_count,
                                      int evolution_steps);

/* ============================================================================
 * A03.2.2.2 — CfC概率逻辑推理API
 * ============================================================================ */

/**
 * @brief 添加概率因子
 *
 * @param state 推理状态
 * @param factor 概率因子定义
 * @return int 成功返回因子ID，失败返回-1
 */
int cfc_uncertain_add_prob_factor(CfCUncertainReasoningState* state,
                                   const CfCUncertainProbFactor* factor);

/**
 * @brief 执行概率逻辑推理（变量消除/信念传播）
 *
 * 使用CfC液态概率传播：
 *   1. 因子图构建：将概率因子转为因子图
 *   2. 液态信念传播：用CfC ODE模拟消息传递
 *      m_{i→j} = Σ_{xi} φ_i(x_i) Π_{k≠j} m_{k→i}
 *   3. 边际概率计算：P(x_i) ∝ Π m_{k→i}
 *
 * @param state 推理状态
 * @param query_vars 查询变量ID数组
 * @param num_queries 查询变量数
 * @param evidence_vars 证据变量ID数组
 * @param evidence_vals 证据变量值数组
 * @param num_evidence 证据变量数
 * @param results 边际概率输出 [num_queries]
 * @return int 成功返回0，失败返回-1
 */
int cfc_uncertain_prob_infer(CfCUncertainReasoningState* state,
                              const int* query_vars, int num_queries,
                              const int* evidence_vars, const float* evidence_vals,
                              int num_evidence, float* results);

/**
 * @brief 执行马尔可夫逻辑网推理
 *
 * 对于加权一阶逻辑公式(F_i, w_i)：
 *   P(x) = (1/Z) * exp(Σ w_i * n_i(x))
 * 其中n_i(x)是公式F_i在x上的真值赋值数
 *
 * @param state 推理状态
 * @param formula_weights 公式权重数组
 * @param formula_count 公式数
 * @param num_variables 变量数
 * @param query_index 查询变量索引
 * @param result 输出概率
 * @return int 成功返回0，失败返回-1
 */
int cfc_uncertain_markov_logic_infer(CfCUncertainReasoningState* state,
                                      const float* formula_weights,
                                      int formula_count, int num_variables,
                                      int query_index, float* result);

/* ============================================================================
 * A03.2.2.3 — CfC Dempster-Shafer推理API
 * ============================================================================ */

/**
 * @brief 添加证据源
 *
 * @param state 推理状态
 * @param source 证据源定义
 * @return int 成功返回源ID，失败返回-1
 */
int cfc_uncertain_add_evidence_source(CfCUncertainReasoningState* state,
                                       const CfCUncertainEvidenceSource* source);

/**
 * @brief 执行Dempster-Shafer证据融合
 *
 * Dempster组合规则：
 *   m₁₂(A) = (1/(1-K)) * Σ_{B∩C=A} m₁(B) * m₂(C)
 * 其中 K = Σ_{B∩C=∅} m₁(B) * m₂(C) 是冲突因子
 *
 * Yager修改规则：冲突部分归入未知 Θ
 *
 * @param state 推理状态
 * @param source_ids 证据源ID数组
 * @param num_sources 证据源数
 * @param result_hypotheses 结果假设数组 [num_hypotheses]
 * @param max_hypotheses 最大假设数
 * @return int 实际输出的假设数，失败返回-1
 */
int cfc_uncertain_ds_fuse(CfCUncertainReasoningState* state,
                           const int* source_ids, int num_sources,
                           CfCUncertainEvidence* result_hypotheses,
                           int max_hypotheses);

/**
 * @brief 计算信任函数和似然函数
 *
 * Bel(A) = Σ_{B⊆A} m(B)
 * Pl(A) = Σ_{B∩A≠∅} m(B)
 *
 * @param state 推理状态
 * @param hypothesis_id 假设ID
 * @param belief 输出信任度
 * @param plausibility 输出似然度
 * @return int 成功返回0，失败返回-1
 */
int cfc_uncertain_ds_belief_pl(CfCUncertainReasoningState* state,
                                int hypothesis_id,
                                float* belief, float* plausibility);

/**
 * @brief 带折扣因子的证据融合
 *
 * m_α(A) = α * m(A)  (对于A≠Θ)
 * m_α(Θ) = 1 - α + α*m(Θ)  (对于全集)
 *
 * @param state 推理状态
 * @param source_ids 证据源ID数组
 * @param discounts 各源折扣因子数组 [num_sources]
 * @param num_sources 证据源数
 * @param result 输出融合后假设数组
 * @param max_hypotheses 最大假设数
 * @return int 实际输出假设数，失败返回-1
 */
int cfc_uncertain_ds_discounted_fuse(CfCUncertainReasoningState* state,
                                      const int* source_ids,
                                      const float* discounts,
                                      int num_sources,
                                      CfCUncertainEvidence* result,
                                      int max_hypotheses);

/* ============================================================================
 * 不确定性决策与排序
 * ============================================================================ */

/**
 * @brief 在不确定性下决策
 *
 * 使用期望效用最大化：
 *   a* = argmax_a Σ_o P(o|a) * U(o)
 * 其中P(o|a)来自不确定性推理结果，U(o)是效用函数
 *
 * @param state 推理状态
 * @param action_ids 候选动作ID数组
 * @param num_actions 候选动作数
 * @param utility_weights 各结果效用权重
 * @param num_outcomes 结果数
 * @param best_action_idx 输出最优动作索引
 * @param expected_utility 输出最大期望效用
 * @return int 成功返回0，失败返回-1
 */
int cfc_uncertain_decision_making(CfCUncertainReasoningState* state,
                                   const int* action_ids, int num_actions,
                                   const float* utility_weights, int num_outcomes,
                                   int* best_action_idx, float* expected_utility);

/**
 * @brief 计算推断结果的不确定度
 *
 * H(P) = -Σ P(x) * log(P(x))  (信息熵)
 * 或 U(A) = 1 - (Bel(A) + Pl(A))/2  (DS不确定度)
 *
 * @param state 推理状态
 * @param result_id 结果ID
 * @return float 不确定度(0~1)
 */
float cfc_uncertain_compute_uncertainty(CfCUncertainReasoningState* state,
                                         int result_id);

/**
 * @brief 生成不确定性推理解释
 *
 * @param state 推理状态
 * @param result_id 结果ID
 * @param explanation 解释文本缓冲区
 * @param max_length 缓冲区长度
 * @return int 成功返回0，失败返回-1
 */
int cfc_uncertain_explain(CfCUncertainReasoningState* state,
                           int result_id, char* explanation, size_t max_length);

/* ============================================================================
 * 模型管理
 * ============================================================================ */

/**
 * @brief 保存不确定性推理模型
 *
 * @param state 推理状态
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int cfc_uncertain_save(const CfCUncertainReasoningState* state,
                        const char* filepath);

/**
 * @brief 加载不确定性推理模型
 *
 * @param filepath 文件路径
 * @return CfCUncertainReasoningState* 成功返回状态指针，失败返回NULL
 */
CfCUncertainReasoningState* cfc_uncertain_load(const char* filepath);

/**
 * @brief 获取默认不确定性推理配置
 *
 * @param algorithm 算法类型
 * @return CfCUncertainConfig 默认配置
 */
CfCUncertainConfig cfc_uncertain_default_config(CfCUncertainAlgorithm algorithm);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_CFC_UNCERTAINTY_REASONING_H */
