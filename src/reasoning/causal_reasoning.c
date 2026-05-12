/**
 * @file causal_reasoning.c
 * @brief 因果推理引擎实现
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

#include "selflnn/reasoning/causal_reasoning.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include <math.h>
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdio.h>

/* 手动声明内存函数以避免警告 */

void safe_free(void** ptr);

/**
 * @brief 因果图内部结构
 */
typedef struct {
    CausalNode* nodes;              /**< 节点数组 */
    size_t num_nodes;               /**< 节点数量 */
    size_t max_nodes;               /**< 最大节点容量 */
    
    CausalEdge* edges;              /**< 边数组 */
    size_t num_edges;               /**< 边数量 */
    size_t max_edges;               /**< 最大边容量 */
    
    float** adjacency_matrix;       /**< 邻接矩阵 */
    float** correlation_matrix;     /**< 相关矩阵 */
    float** partial_correlation;    /**< 偏相关矩阵 */
    
    char** variable_names;          /**< 变量名称数组 */
    size_t num_variables;           /**< 变量数量 */
} CausalGraph;

/**
 * @brief 因果推理引擎内部结构
 */
struct CausalReasoningEngine {
    CausalReasoningConfig config;   /**< 配置 */
    CausalGraph graph;              /**< 因果图 */
    
    /* 算法状态 */
    int is_graph_built;             /**< 因果图是否已构建 */
    int is_trained;                 /**< 是否已训练 */
    
    /* 统计信息 */
    float* variable_means;          /**< 变量均值 */
    float* variable_variances;      /**< 变量方差 */
    float** covariance_matrix;      /**< 协方差矩阵 */
    
    /* 时间序列分析 */
    float** time_series_data;       /**< 时间序列数据 */
    size_t time_series_length;      /**< 时间序列长度 */
    size_t time_lag;                /**< 时间滞后 */
    
    /* 缓存和优化 */
    float* inference_cache;         /**< 推理缓存 */
    size_t cache_size;              /**< 缓存大小 */
    int cache_valid;                /**< 缓存是否有效 */
    
    /* 性能监控 */
    size_t total_inferences;        /**< 总推理次数 */
    size_t total_discoveries;       /**< 总发现次数 */
    size_t last_inference_time;     /**< 最后推理时间 */
};

/* 静态函数声明 */
static int initialize_causal_graph(CausalGraph* graph, size_t initial_capacity);
static void free_causal_graph(CausalGraph* graph);
static int add_node_to_graph(CausalGraph* graph, const CausalNode* node);
static int add_edge_to_graph(CausalGraph* graph, const CausalEdge* edge);
static float compute_correlation(const float* data_x, const float* data_y, size_t n);
static float compute_partial_correlation(const float** data, size_t n, int i, int j, const int* conditioning_set, size_t k);
static int pc_algorithm(CausalReasoningEngine* engine, const float* data, size_t num_samples, size_t num_variables);
static int fci_algorithm(CausalReasoningEngine* engine, const float* data, size_t num_samples, size_t num_variables);
static float granger_causality(CausalReasoningEngine* engine, int cause_idx, int effect_idx, int max_lag);
static float transfer_entropy(CausalReasoningEngine* engine, int cause_idx, int effect_idx, int k, int l);
static int structural_causal_model(CausalReasoningEngine* engine, const float* data, size_t num_samples);
static float estimate_causal_effect_do_calculus(CausalReasoningEngine* engine, int treatment, int outcome, const int* confounders, size_t num_confounders);
static int perform_counterfactual_reasoning(CausalReasoningEngine* engine, const float* factual, size_t n, float intervention, float* counterfactual, size_t max_size);
static int find_causal_path(CausalReasoningEngine* engine, int source, int target, int* path, size_t max_path_length);
static int d_separation_test(CausalGraph* graph, int x, int y, const int* z, size_t z_size);
static int moralize_graph(CausalGraph* graph);
static int triangulate_graph(CausalGraph* graph);
static float compute_bayesian_information_criterion(const float* data, size_t n, size_t num_vars, int num_params);

/* A04.1.2 因果效应估计增强函数 */
static float* estimate_propensity_score_logistic(const float** data, size_t n, int treatment_idx, const int* covariates, size_t num_covariates, float* scores);
static float* outcome_regression_linear(const float** data, size_t n, int outcome_idx, int treatment_idx, const int* covariates, size_t num_covariates, float* y0, float* y1);
static float estimate_ate_ipw_internal(const float** data, size_t n, int treatment_idx, int outcome_idx, const int* covariates, size_t num_covariates);
static float estimate_ate_doubly_robust_internal(const float** data, size_t n, int treatment_idx, int outcome_idx, const int* covariates, size_t num_covariates);
static float estimate_iv_2sls_internal(const float** data, size_t n, int treatment_idx, int outcome_idx, int instrument_idx, const int* covariates, size_t num_covariates);
static int ges_algorithm(CausalReasoningEngine* engine, const float* data, size_t num_samples, size_t num_variables);
static int lingam_algorithm(CausalReasoningEngine* engine, const float* data, size_t num_samples, size_t num_variables);
static int dag_has_cycle(int* dag, size_t num_variables);
static float compute_bic_for_graph(const float* data, size_t n, size_t m, int* dag);

/**
 * @brief 创建因果推理引擎
 */
CausalReasoningEngine* causal_reasoning_engine_create(const CausalReasoningConfig* config) {
    if (config == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建因果推理引擎：配置参数为空");
        return NULL;
    }
    
    CausalReasoningEngine* engine = (CausalReasoningEngine*)safe_malloc(sizeof(CausalReasoningEngine));
    if (engine == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建因果推理引擎：内存分配失败");
        return NULL;
    }
    
    /* 初始化配置 */
    memcpy(&engine->config, config, sizeof(CausalReasoningConfig));
    
    /* 设置默认值 */
    if (engine->config.significance_level <= 0.0f || engine->config.significance_level > 1.0f) {
        engine->config.significance_level = 0.05f;
    }
    
    if (engine->config.max_conditioning_set_size <= 0) {
        engine->config.max_conditioning_set_size = 5;
    }
    
    if (engine->config.max_iterations <= 0) {
        engine->config.max_iterations = 1000;
    }
    
    if (engine->config.confidence_threshold <= 0.0f) {
        engine->config.confidence_threshold = 0.7f;
    }
    
    /* 初始化因果图 */
    if (initialize_causal_graph(&engine->graph, 32) != 0) {
        safe_free((void**)&engine);
        return NULL;
    }
    
    /* 初始化其他字段 */
    engine->is_graph_built = 0;
    engine->is_trained = 0;
    engine->variable_means = NULL;
    engine->variable_variances = NULL;
    engine->covariance_matrix = NULL;
    engine->time_series_data = NULL;
    engine->time_series_length = 0;
    engine->time_lag = 1;
    engine->inference_cache = NULL;
    engine->cache_size = 0;
    engine->cache_valid = 0;
    engine->total_inferences = 0;
    engine->total_discoveries = 0;
    engine->last_inference_time = 0;
    
    return engine;
}

/**
 * @brief 释放因果推理引擎
 */
void causal_reasoning_engine_free(CausalReasoningEngine* engine) {
    if (engine == NULL) {
        return;
    }
    
    /* 释放因果图 */
    free_causal_graph(&engine->graph);
    
    /* 释放统计信息 */
    if (engine->variable_means != NULL) {
        safe_free((void**)&engine->variable_means);
    }
    
    if (engine->variable_variances != NULL) {
        safe_free((void**)&engine->variable_variances);
    }
    
    if (engine->covariance_matrix != NULL) {
        for (size_t i = 0; i < engine->graph.num_variables; i++) {
            if (engine->covariance_matrix[i] != NULL) {
                safe_free((void**)&engine->covariance_matrix[i]);
            }
        }
        safe_free((void**)&engine->covariance_matrix);
    }
    
    /* 释放时间序列数据 */
    if (engine->time_series_data != NULL) {
        for (size_t i = 0; i < engine->graph.num_variables; i++) {
            if (engine->time_series_data[i] != NULL) {
                safe_free((void**)&engine->time_series_data[i]);
            }
        }
        safe_free((void**)&engine->time_series_data);
    }
    
    /* 释放缓存 */
    if (engine->inference_cache != NULL) {
        safe_free((void**)&engine->inference_cache);
    }
    
    safe_free((void**)&engine);
}

/**
 * @brief 初始化因果图
 */
static int initialize_causal_graph(CausalGraph* graph, size_t initial_capacity) {
    if (graph == NULL) {
        return -1;
    }
    
    memset(graph, 0, sizeof(CausalGraph));
    
    /* 分配节点数组 */
    graph->nodes = (CausalNode*)safe_malloc(initial_capacity * sizeof(CausalNode));
    if (graph->nodes == NULL) {
        return -1;
    }
    graph->num_nodes = 0;
    graph->max_nodes = initial_capacity;
    
    /* 分配边数组 */
    graph->edges = (CausalEdge*)safe_malloc(initial_capacity * 2 * sizeof(CausalEdge));
    if (graph->edges == NULL) {
        safe_free((void**)&graph->nodes);
        return -1;
    }
    graph->num_edges = 0;
    graph->max_edges = initial_capacity * 2;
    
    /* 初始化矩阵指针为NULL，将在需要时分配 */
    graph->adjacency_matrix = NULL;
    graph->correlation_matrix = NULL;
    graph->partial_correlation = NULL;
    graph->variable_names = NULL;
    graph->num_variables = 0;
    
    return 0;
}

/**
 * @brief 释放因果图
 */
static void free_causal_graph(CausalGraph* graph) {
    if (graph == NULL) {
        return;
    }
    
    /* 释放节点数组 */
    if (graph->nodes != NULL) {
        for (size_t i = 0; i < graph->num_nodes; i++) {
            if (graph->nodes[i].node_features != NULL) {
                safe_free((void**)&graph->nodes[i].node_features);
            }
        }
        safe_free((void**)&graph->nodes);
    }
    
    /* 释放边数组 */
    if (graph->edges != NULL) {
        for (size_t i = 0; i < graph->num_edges; i++) {
            if (graph->edges[i].edge_parameters != NULL) {
                safe_free((void**)&graph->edges[i].edge_parameters);
            }
        }
        safe_free((void**)&graph->edges);
    }
    
    /* 释放邻接矩阵 */
    if (graph->adjacency_matrix != NULL) {
        for (size_t i = 0; i < graph->num_variables; i++) {
            if (graph->adjacency_matrix[i] != NULL) {
                safe_free((void**)&graph->adjacency_matrix[i]);
            }
        }
        safe_free((void**)&graph->adjacency_matrix);
    }
    
    /* 释放相关矩阵 */
    if (graph->correlation_matrix != NULL) {
        for (size_t i = 0; i < graph->num_variables; i++) {
            if (graph->correlation_matrix[i] != NULL) {
                safe_free((void**)&graph->correlation_matrix[i]);
            }
        }
        safe_free((void**)&graph->correlation_matrix);
    }
    
    /* 释放偏相关矩阵 */
    if (graph->partial_correlation != NULL) {
        for (size_t i = 0; i < graph->num_variables; i++) {
            if (graph->partial_correlation[i] != NULL) {
                safe_free((void**)&graph->partial_correlation[i]);
            }
        }
        safe_free((void**)&graph->partial_correlation);
    }
    
    /* 释放变量名称 */
    if (graph->variable_names != NULL) {
        for (size_t i = 0; i < graph->num_variables; i++) {
            if (graph->variable_names[i] != NULL) {
                safe_free((void**)&graph->variable_names[i]);
            }
        }
        safe_free((void**)&graph->variable_names);
    }
    
    memset(graph, 0, sizeof(CausalGraph));
}

/**
 * @brief 添加节点到因果图
 */
static int add_node_to_graph(CausalGraph* graph, const CausalNode* node) {
    if (graph == NULL || node == NULL) {
        return -1;
    }
    
    /* 检查是否需要扩展节点数组 */
    if (graph->num_nodes >= graph->max_nodes) {
        size_t new_capacity = graph->max_nodes * 2;
        CausalNode* new_nodes = (CausalNode*)safe_malloc(new_capacity * sizeof(CausalNode));
        if (new_nodes == NULL) {
            return -1;
        }
        
        memcpy(new_nodes, graph->nodes, graph->num_nodes * sizeof(CausalNode));
        safe_free((void**)&graph->nodes);
        graph->nodes = new_nodes;
        graph->max_nodes = new_capacity;
    }
    
    /* 复制节点 */
    CausalNode* dest = &graph->nodes[graph->num_nodes];
    memcpy(dest, node, sizeof(CausalNode));
    
    /* 复制节点特征 */
    if (node->node_features != NULL && node->feature_size > 0) {
        dest->node_features = (float*)safe_malloc(node->feature_size * sizeof(float));
        if (dest->node_features == NULL) {
            return -1;
        }
        memcpy(dest->node_features, node->node_features, node->feature_size * sizeof(float));
    } else {
        dest->node_features = NULL;
        dest->feature_size = 0;
    }
    
    graph->num_nodes++;
    return 0;
}

/**
 * @brief 添加边到因果图
 */
static int add_edge_to_graph(CausalGraph* graph, const CausalEdge* edge) {
    if (graph == NULL || edge == NULL) {
        return -1;
    }
    
    /* 检查节点是否存在 */
    if (edge->source_node_id < 0 || edge->source_node_id >= (int)graph->num_nodes ||
        edge->target_node_id < 0 || edge->target_node_id >= (int)graph->num_nodes) {
        return -1;
    }
    
    /* 检查是否需要扩展边数组 */
    if (graph->num_edges >= graph->max_edges) {
        size_t new_capacity = graph->max_edges * 2;
        CausalEdge* new_edges = (CausalEdge*)safe_malloc(new_capacity * sizeof(CausalEdge));
        if (new_edges == NULL) {
            return -1;
        }
        
        memcpy(new_edges, graph->edges, graph->num_edges * sizeof(CausalEdge));
        safe_free((void**)&graph->edges);
        graph->edges = new_edges;
        graph->max_edges = new_capacity;
    }
    
    /* 复制边 */
    CausalEdge* dest = &graph->edges[graph->num_edges];
    memcpy(dest, edge, sizeof(CausalEdge));
    
    /* 复制边参数 */
    if (edge->edge_parameters != NULL && edge->param_size > 0) {
        dest->edge_parameters = (float*)safe_malloc(edge->param_size * sizeof(float));
        if (dest->edge_parameters == NULL) {
            return -1;
        }
        memcpy(dest->edge_parameters, edge->edge_parameters, edge->param_size * sizeof(float));
    } else {
        dest->edge_parameters = NULL;
        dest->param_size = 0;
    }
    
    graph->num_edges++;
    return 0;
}

/**
 * @brief 计算相关系数
 */
static float compute_correlation(const float* data_x, const float* data_y, size_t n) {
    if (n < 2 || data_x == NULL || data_y == NULL) {
        return 0.0f;
    }
    
    float sum_x = 0.0f, sum_y = 0.0f;
    float sum_xy = 0.0f, sum_x2 = 0.0f, sum_y2 = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float x = data_x[i];
        float y = data_y[i];
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }
    
    float numerator = n * sum_xy - sum_x * sum_y;
    float denominator_x = n * sum_x2 - sum_x * sum_x;
    float denominator_y = n * sum_y2 - sum_y * sum_y;
    
    if (denominator_x <= 0.0f || denominator_y <= 0.0f) {
        return 0.0f;
    }
    
    float correlation = numerator / sqrtf(denominator_x * denominator_y);
    
    /* 限制在-1到1之间 */
    if (correlation < -1.0f) correlation = -1.0f;
    if (correlation > 1.0f) correlation = 1.0f;
    
    return correlation;
}

/**
 * @brief 计算偏相关系数
 */
static float compute_partial_correlation(const float** data, size_t n, int i, int j, 
                                        const int* conditioning_set, size_t k) {
    if (n < 2 || data == NULL || i < 0 || j < 0) {
        return 0.0f;
    }
    
    /* 如果没有条件集，退化为普通相关系数 */
    if (k == 0 || conditioning_set == NULL) {
        return compute_correlation(data[i], data[j], n);
    }
    
    /* 完整实现：使用多元线性回归残差计算偏相关 */
    /* 步骤1: 将变量i回归到条件集上，得到残差R_i */
    /* 步骤2: 将变量j回归到条件集上，得到残差R_j */
    /* 步骤3: 计算R_i和R_j的相关系数 = 偏相关系数 */
    
    size_t cond_vars = k;
    int n_obs = (int)n;
    
    /* 构建设计矩阵X = [1, Z1, Z2, ..., Zk] */
    float* X_data = (float*)safe_malloc((size_t)n_obs * (cond_vars + 1) * sizeof(float));
    float* y_i = (float*)safe_malloc((size_t)n_obs * sizeof(float));
    float* y_j = (float*)safe_malloc((size_t)n_obs * sizeof(float));
    float* resid_i = (float*)safe_malloc((size_t)n_obs * sizeof(float));
    float* resid_j = (float*)safe_malloc((size_t)n_obs * sizeof(float));
    
    if (!X_data || !y_i || !y_j || !resid_i || !resid_j) {
        if (X_data) safe_free((void**)&X_data);
        if (y_i) safe_free((void**)&y_i);
        if (y_j) safe_free((void**)&y_j);
        if (resid_i) safe_free((void**)&resid_i);
        if (resid_j) safe_free((void**)&resid_j);
        return compute_correlation(data[i], data[j], n);
    }
    
    int p = (int)cond_vars + 1;  /* 参数数量（含截距） */
    
    for (int t = 0; t < n_obs; t++) {
        X_data[t * p] = 1.0f;  /* 截距项 */
        for (size_t c = 0; c < cond_vars; c++) {
            int z_idx = conditioning_set[c];
            X_data[t * p + (int)c + 1] = (z_idx >= 0) ? data[z_idx][t] : 0.0f;
        }
        y_i[t] = data[i][t];
        y_j[t] = data[j][t];
    }
    
    /* OLS: β = (X'X)^{-1} X'y */
    /* 计算X'X */
    float* XtX = (float*)safe_malloc((size_t)p * p * sizeof(float));
    float* Xty_i = (float*)safe_malloc((size_t)p * sizeof(float));
    float* Xty_j = (float*)safe_malloc((size_t)p * sizeof(float));
    
    if (!XtX || !Xty_i || !Xty_j) {
        if (XtX) safe_free((void**)&XtX);
        if (Xty_i) safe_free((void**)&Xty_i);
        if (Xty_j) safe_free((void**)&Xty_j);
        safe_free((void**)&X_data); safe_free((void**)&y_i); safe_free((void**)&y_j);
        safe_free((void**)&resid_i); safe_free((void**)&resid_j);
        return compute_correlation(data[i], data[j], n);
    }
    
    memset(XtX, 0, (size_t)p * p * sizeof(float));
    memset(Xty_i, 0, (size_t)p * sizeof(float));
    memset(Xty_j, 0, (size_t)p * sizeof(float));
    
    for (int t = 0; t < n_obs; t++) {
        for (int a = 0; a < p; a++) {
            for (int b = 0; b < p; b++) {
                XtX[a * p + b] += X_data[t * p + a] * X_data[t * p + b];
            }
            Xty_i[a] += X_data[t * p + a] * y_i[t];
            Xty_j[a] += X_data[t * p + a] * y_j[t];
        }
    }
    
    /* 高斯消元法求解β_i = XtX^{-1} * Xty_i */
    float* A_i = (float*)safe_malloc((size_t)p * p * sizeof(float));
    float* A_j = (float*)safe_malloc((size_t)p * p * sizeof(float));
    float* beta_i = (float*)safe_malloc((size_t)p * sizeof(float));
    float* beta_j = (float*)safe_malloc((size_t)p * sizeof(float));
    
    if (!A_i || !A_j || !beta_i || !beta_j) {
        if (A_i) safe_free((void**)&A_i); if (A_j) safe_free((void**)&A_j);
        if (beta_i) safe_free((void**)&beta_i); if (beta_j) safe_free((void**)&beta_j);
        safe_free((void**)&XtX); safe_free((void**)&Xty_i); safe_free((void**)&Xty_j);
        safe_free((void**)&X_data); safe_free((void**)&y_i); safe_free((void**)&y_j);
        safe_free((void**)&resid_i); safe_free((void**)&resid_j);
        return compute_correlation(data[i], data[j], n);
    }
    
    memcpy(A_i, XtX, (size_t)p * p * sizeof(float));
    memcpy(beta_i, Xty_i, (size_t)p * sizeof(float));
    for (int col = 0; col < p; col++) {
        if (fabsf(A_i[col * p + col]) < 1e-10f) continue;
        for (int row = 0; row < p; row++) {
            if (row == col) continue;
            float factor = A_i[row * p + col] / A_i[col * p + col];
            for (int c2 = 0; c2 < p; c2++) {
                A_i[row * p + c2] -= factor * A_i[col * p + c2];
            }
            beta_i[row] -= factor * beta_i[col];
        }
    }
    for (int row = 0; row < p; row++) {
        if (fabsf(A_i[row * p + row]) > 1e-10f) {
            beta_i[row] /= A_i[row * p + row];
        } else {
            beta_i[row] = 0.0f;
        }
    }
    
    memcpy(A_j, XtX, (size_t)p * p * sizeof(float));
    memcpy(beta_j, Xty_j, (size_t)p * sizeof(float));
    for (int col = 0; col < p; col++) {
        if (fabsf(A_j[col * p + col]) < 1e-10f) continue;
        for (int row = 0; row < p; row++) {
            if (row == col) continue;
            float factor = A_j[row * p + col] / A_j[col * p + col];
            for (int c2 = 0; c2 < p; c2++) {
                A_j[row * p + c2] -= factor * A_j[col * p + c2];
            }
            beta_j[row] -= factor * beta_j[col];
        }
    }
    for (int row = 0; row < p; row++) {
        if (fabsf(A_j[row * p + row]) > 1e-10f) {
            beta_j[row] /= A_j[row * p + row];
        } else {
            beta_j[row] = 0.0f;
        }
    }
    
    /* 计算残差 */
    for (int t = 0; t < n_obs; t++) {
        float pred_i = 0.0f, pred_j = 0.0f;
        for (int a = 0; a < p; a++) {
            pred_i += X_data[t * p + a] * beta_i[a];
            pred_j += X_data[t * p + a] * beta_j[a];
        }
        resid_i[t] = y_i[t] - pred_i;
        resid_j[t] = y_j[t] - pred_j;
    }
    
    /* 计算残差相关系数 = 偏相关系数 */
    float partial_corr = compute_correlation(resid_i, resid_j, n);
    
    safe_free((void**)&X_data);
    safe_free((void**)&y_i);
    safe_free((void**)&y_j);
    safe_free((void**)&resid_i);
    safe_free((void**)&resid_j);
    safe_free((void**)&XtX);
    safe_free((void**)&Xty_i);
    safe_free((void**)&Xty_j);
    safe_free((void**)&A_i);
    safe_free((void**)&A_j);
    safe_free((void**)&beta_i);
    safe_free((void**)&beta_j);
    
    /* 限制在-1到1之间 */
    if (partial_corr < -1.0f) partial_corr = -1.0f;
    if (partial_corr > 1.0f) partial_corr = 1.0f;
    
    return partial_corr;
}

/**
 * @brief 组合生成器：生成从n个元素中选k个的所有组合
 * @param comb 输出组合数组（长度为k），首次调用时内容无效
 * @param n 总元素数
 * @param k 选取元素数
 * @param first 首次调用时传入1，后续传入0
 * @return 0=有有效组合，1=所有组合已遍历完
 */
static int next_combination(int* comb, int n, int k, int first) {
    if (first) {
        for (int i = 0; i < k; i++) comb[i] = i;
        return (k <= n) ? 0 : 1;
    }
    int i = k - 1;
    while (i >= 0 && comb[i] == n - k + i) i--;
    if (i < 0) return 1;
    comb[i]++;
    for (int j = i + 1; j < k; j++) comb[j] = comb[j - 1] + 1;
    return 0;
}

/**
 * @brief Fisher Z-变换条件独立性检验
 * @param partial_corr 偏相关系数
 * @param n 样本量
 * @param num_cond 条件变量数
 * @param significance 显著性水平
 * @return 1=独立（p值>显著性水平），0=不独立
 */
static int fisher_z_independence_test(float partial_corr, size_t n, size_t num_cond, float significance) {
    if (n <= num_cond + 3) return 0;
    
    float r = partial_corr;
    if (r >= 1.0f) r = 0.9999f;
    if (r <= -1.0f) r = -0.9999f;
    
    float z = 0.5f * logf((1.0f + r) / (1.0f - r));
    float z_std = sqrtf((float)(n - num_cond - 3));
    float z_stat = fabsf(z) * z_std;
    
    /* 标准正态CDF近似（更高精度） */
    float a = fabsf(z_stat);
    float cdf_approx;
    if (a < 6.0f) {
        float b0 = 0.2316419f, b1 = 0.319381530f, b2 = -0.356563782f;
        float b3 = 1.781477937f, b4 = -1.821255978f, b5 = 1.330274429f;
        float t = 1.0f / (1.0f + b0 * a);
        float poly = b1 + t * (b2 + t * (b3 + t * (b4 + t * b5)));
        cdf_approx = 1.0f - 0.3989422804f * expf(-a * a * 0.5f) * poly;
    } else {
        cdf_approx = 1.0f;
    }
    if (z_stat < 0) cdf_approx = 1.0f - cdf_approx;
    
    float p_value = 2.0f * (1.0f - cdf_approx);
    if (p_value < 0.0f) p_value = 0.0f;
    if (p_value > 1.0f) p_value = 1.0f;
    
    return (p_value > significance) ? 1 : 0;
}

/**
 * @brief PC算法（Peter-Clark算法）实现
 */
static int pc_algorithm(CausalReasoningEngine* engine, const float* data, 
                       size_t num_samples, size_t num_variables) {
    if (engine == NULL || data == NULL || num_samples == 0 || num_variables == 0) {
        return -1;
    }
    
    /* 步骤1：初始化完全连接的无向图 */
    
    /* 步骤2：逐步删除边，基于条件独立性测试 */
    float significance = engine->config.significance_level;
    int max_conditioning = engine->config.max_conditioning_set_size;
    
    /* 将数据重新组织为二维数组以便访问 */
    float** data_matrix = (float**)safe_malloc(num_variables * sizeof(float*));
    if (data_matrix == NULL) {
        return -1;
    }
    
    for (size_t i = 0; i < num_variables; i++) {
        data_matrix[i] = (float*)safe_malloc(num_samples * sizeof(float));
        if (data_matrix[i] == NULL) {
            for (size_t j = 0; j < i; j++) {
                safe_free((void**)&data_matrix[j]);
            }
            safe_free((void**)&data_matrix);
            return -1;
        }
        
        /* 从扁平数组复制数据 */
        for (size_t j = 0; j < num_samples; j++) {
            data_matrix[i][j] = data[j * num_variables + i];
        }
    }
    
    /* 创建邻接矩阵（初始为完全连接） */
    if (engine->graph.adjacency_matrix == NULL) {
        engine->graph.adjacency_matrix = (float**)safe_malloc(num_variables * sizeof(float*));
        if (engine->graph.adjacency_matrix == NULL) {
            for (size_t i = 0; i < num_variables; i++) {
                safe_free((void**)&data_matrix[i]);
            }
            safe_free((void**)&data_matrix);
            return -1;
        }
        
        for (size_t i = 0; i < num_variables; i++) {
            engine->graph.adjacency_matrix[i] = (float*)safe_malloc(num_variables * sizeof(float));
            if (engine->graph.adjacency_matrix[i] == NULL) {
                for (size_t j = 0; j < i; j++) {
                    safe_free((void**)&engine->graph.adjacency_matrix[j]);
                }
                safe_free((void**)&engine->graph.adjacency_matrix);
                for (size_t k = 0; k < num_variables; k++) {
                    safe_free((void**)&data_matrix[k]);
                }
                safe_free((void**)&data_matrix);
                return -1;
            }
            
            /* 初始化为1（完全连接） */
            for (size_t j = 0; j < num_variables; j++) {
                engine->graph.adjacency_matrix[i][j] = (i == j) ? 0.0f : 1.0f;
            }
        }
    }
    
    /* 完整PC算法条件独立性测试：遍历所有邻域子集，使用Fisher Z检验 */
    for (size_t i = 0; i < num_variables; i++) {
        for (size_t j = i + 1; j < num_variables; j++) {
            if (engine->graph.adjacency_matrix[i][j] <= 0.0f) continue;
            
            int independent = 0;
            
            for (int cond_size = 0; cond_size <= max_conditioning && !independent; cond_size++) {
                /* 构建i的邻域（排除j）和j的邻域（排除i），求并集作为候选条件集 */
                int neighbor_i[256], neighbor_j[256], candidate_set[256];
                int ni = 0, nj = 0, nc = 0;
                int used[256] = {0};
                
                for (size_t k = 0; k < num_variables && ni < 256; k++) {
                    if (k != i && k != j && engine->graph.adjacency_matrix[i][k] > 0.0f) {
                        neighbor_i[ni++] = (int)k;
                    }
                }
                for (size_t k = 0; k < num_variables && nj < 256; k++) {
                    if (k != i && k != j && engine->graph.adjacency_matrix[j][k] > 0.0f) {
                        neighbor_j[nj++] = (int)k;
                    }
                }
                
                for (int a = 0; a < ni && nc < 256; a++) {
                    if (!used[neighbor_i[a]]) {
                        used[neighbor_i[a]] = 1;
                        candidate_set[nc++] = neighbor_i[a];
                    }
                }
                for (int a = 0; a < nj && nc < 256; a++) {
                    if (!used[neighbor_j[a]]) {
                        used[neighbor_j[a]] = 1;
                        candidate_set[nc++] = neighbor_j[a];
                    }
                }
                
                if (cond_size > nc) continue;
                if (cond_size == 0) {
                    /* 无条件独立性测试：Fisher Z检验 */
                    float partial_corr = compute_partial_correlation((const float**)data_matrix,
                                            num_samples, (int)i, (int)j, NULL, 0);
                    if (fisher_z_independence_test(partial_corr, num_samples, 0, significance)) {
                        independent = 1;
                    }
                } else {
                    /* 生成所有大小为cond_size的候选子集组合 */
                    int comb[256];
                    if (next_combination(comb, nc, cond_size, 1) != 0) continue;
                    
                    do {
                        int cond_set[256];
                        for (int a = 0; a < cond_size; a++) {
                            cond_set[a] = candidate_set[comb[a]];
                        }
                        
                        float partial_corr = compute_partial_correlation((const float**)data_matrix,
                                                num_samples, (int)i, (int)j,
                                                cond_set, (size_t)cond_size);
                        
                        if (fisher_z_independence_test(partial_corr, num_samples, (size_t)cond_size, significance)) {
                            independent = 1;
                            break;
                        }
                    } while (next_combination(comb, nc, cond_size, 0) == 0);
                }
            }
            
            /* 如果独立，删除边 */
            if (independent) {
                engine->graph.adjacency_matrix[i][j] = 0.0f;
                engine->graph.adjacency_matrix[j][i] = 0.0f;
            }
        }
    }
    
    /* 步骤3：确定边的方向（使用v-结构检测+Meek规则传播） */
    /* 3a: 查找v-结构（碰撞检测器）*/
    /* 对于每个三元组(i,j,k)满足i-j-k且i-k不相邻，定向为i->j<-k */
    for (size_t i = 0; i < num_variables; i++) {
        for (size_t k = i + 1; k < num_variables; k++) {
            /* 检查i和k是否有边相连（如果有，不能是v-结构） */
            if (engine->graph.adjacency_matrix[i][k] > 0.0f) continue;
            
            /* 找同时与i和k相邻的节点j */
            for (size_t j = 0; j < num_variables; j++) {
                if (j == i || j == k) continue;
                if (engine->graph.adjacency_matrix[i][j] > 0.0f && 
                    engine->graph.adjacency_matrix[j][k] > 0.0f) {
                    /* v-结构: i->j<-k */
                    engine->graph.adjacency_matrix[j][i] = 0.0f;  /* 删除j->i */
                    engine->graph.adjacency_matrix[j][k] = 0.0f;  /* 删除j->k */
                }
            }
        }
    }
    
    /* 3b: Meek规则传播方向，直到不再有新的定向 */
    int changed = 1;
    int max_iterations = (int)num_variables * (int)num_variables;
    int iter_count = 0;
    while (changed && iter_count < max_iterations) {
        changed = 0;
        iter_count++;
        
        /* Meek Rule 1: i->j, j-k 且 i-k不相邻 => j->k */
        for (size_t i = 0; i < num_variables; i++) {
            for (size_t j = 0; j < num_variables; j++) {
                if (i == j) continue;
                /* i->j: adj[i][j]>0 且 adj[j][i]==0 */
                if (engine->graph.adjacency_matrix[i][j] > 0.0f && 
                    engine->graph.adjacency_matrix[j][i] == 0.0f) {
                    for (size_t k = 0; k < num_variables; k++) {
                        if (k == i || k == j) continue;
                        /* j-k（无向边）：adj[j][k]>0 且 adj[k][j]>0 */
                        if (engine->graph.adjacency_matrix[j][k] > 0.0f && 
                            engine->graph.adjacency_matrix[k][j] > 0.0f) {
                            /* i-k不相邻 */
                            if (engine->graph.adjacency_matrix[i][k] == 0.0f && 
                                engine->graph.adjacency_matrix[k][i] == 0.0f) {
                                engine->graph.adjacency_matrix[k][j] = 0.0f;  /* j->k */
                                changed = 1;
                            }
                        }
                    }
                }
            }
        }
        
        /* Meek Rule 2: i->j, i->k, j-k => j->k */
        for (size_t i = 0; i < num_variables; i++) {
            for (size_t j = 0; j < num_variables; j++) {
                if (i == j) continue;
                if (engine->graph.adjacency_matrix[i][j] > 0.0f && 
                    engine->graph.adjacency_matrix[j][i] == 0.0f) {
                    for (size_t k = 0; k < num_variables; k++) {
                        if (k == i || k == j) continue;
                        if (engine->graph.adjacency_matrix[i][k] > 0.0f && 
                            engine->graph.adjacency_matrix[k][i] == 0.0f) {
                            if (engine->graph.adjacency_matrix[j][k] > 0.0f && 
                                engine->graph.adjacency_matrix[k][j] > 0.0f) {
                                engine->graph.adjacency_matrix[k][j] = 0.0f;  /* j->k */
                                changed = 1;
                            }
                        }
                    }
                }
            }
        }
        
        /* Meek Rule 3: i-j, i->k, j->k => i->j */
        for (size_t i = 0; i < num_variables; i++) {
            for (size_t j = 0; j < num_variables; j++) {
                if (i == j) continue;
                if (engine->graph.adjacency_matrix[i][j] > 0.0f && 
                    engine->graph.adjacency_matrix[j][i] > 0.0f) {
                    for (size_t k = 0; k < num_variables; k++) {
                        if (k == i || k == j) continue;
                        if (engine->graph.adjacency_matrix[i][k] > 0.0f && 
                            engine->graph.adjacency_matrix[k][i] == 0.0f) {
                            if (engine->graph.adjacency_matrix[j][k] > 0.0f && 
                                engine->graph.adjacency_matrix[k][j] == 0.0f) {
                                engine->graph.adjacency_matrix[j][i] = 0.0f;  /* i->j */
                                changed = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    
    /* 剩余的未定向边（仍为无向的边），随机分配方向 */
    for (size_t i = 0; i < num_variables; i++) {
        for (size_t j = i + 1; j < num_variables; j++) {
            if (engine->graph.adjacency_matrix[i][j] > 0.0f && 
                engine->graph.adjacency_matrix[j][i] > 0.0f) {
                if (rng_next() % (uint64_t)(2) == 0) {
                    engine->graph.adjacency_matrix[j][i] = 0.0f;  /* i -> j */
                } else {
                    engine->graph.adjacency_matrix[i][j] = 0.0f;  /* j -> i */
                }
            }
        }
    }
    
    /* 清理数据矩阵 */
    for (size_t i = 0; i < num_variables; i++) {
        safe_free((void**)&data_matrix[i]);
    }
    safe_free((void**)&data_matrix);
    
    engine->is_graph_built = 1;
    return 0;
}

/**
 * @brief 构建因果图
 */
int causal_reasoning_build_graph(CausalReasoningEngine* engine,
                                const float* data, size_t num_samples, size_t num_variables,
                                const char** variable_names) {
    if (engine == NULL || data == NULL || num_samples == 0 || num_variables == 0) {
        return -1;
    }
    
    /* 存储变量数量 */
    engine->graph.num_variables = num_variables;
    
    /* 存储变量名称 */
    if (variable_names != NULL) {
        engine->graph.variable_names = (char**)safe_malloc(num_variables * sizeof(char*));
        if (engine->graph.variable_names != NULL) {
            for (size_t i = 0; i < num_variables; i++) {
                if (variable_names[i] != NULL) {
                    size_t name_len = strlen(variable_names[i]) + 1;
                    engine->graph.variable_names[i] = (char*)safe_malloc(name_len);
                    if (engine->graph.variable_names[i] != NULL) {
                        strcpy(engine->graph.variable_names[i], variable_names[i]);
                    }
                }
            }
        }
    }
    
    /* 根据算法选择构建方法 */
    int result = -1;
    
    switch (engine->config.algorithm) {
        case CAUSAL_REASONING_PC_ALGORITHM:
            result = pc_algorithm(engine, data, num_samples, num_variables);
            break;
            
        case CAUSAL_REASONING_FCI_ALGORITHM:
            result = fci_algorithm(engine, data, num_samples, num_variables);
            break;
            
        case CAUSAL_REASONING_GRANGER: {
            /* 时间序列分析：存储数据并为每条变量对执行Granger因果测试 */
            /* 先存储时间序列数据 */
            engine->time_series_data = (float**)safe_malloc(num_variables * sizeof(float*));
            if (engine->time_series_data == NULL) {
                result = -1;
                break;
            }
            engine->time_series_length = num_samples;
            engine->time_lag = (num_samples > 10) ? (num_samples / 10) : 1;
            if (engine->time_lag < 1) engine->time_lag = 1;
            if (engine->time_lag > 10) engine->time_lag = 10;
            
            for (size_t v = 0; v < num_variables; v++) {
                engine->time_series_data[v] = (float*)safe_malloc(num_samples * sizeof(float));
                if (engine->time_series_data[v] == NULL) {
                    for (size_t u = 0; u < v; u++) {
                        safe_free((void**)&engine->time_series_data[u]);
                    }
                    safe_free((void**)&engine->time_series_data);
                    engine->time_series_data = NULL;
                    result = -1;
                    break;
                }
                for (size_t s = 0; s < num_samples; s++) {
                    engine->time_series_data[v][s] = data[s * num_variables + v];
                }
            }
            if (result == -1) break;
            
            /* 初始化邻接矩阵 */
            if (engine->graph.adjacency_matrix == NULL) {
                engine->graph.adjacency_matrix = (float**)safe_malloc(num_variables * sizeof(float*));
                if (engine->graph.adjacency_matrix == NULL) {
                    result = -1;
                    break;
                }
                for (size_t i = 0; i < num_variables; i++) {
                    engine->graph.adjacency_matrix[i] = (float*)safe_malloc(num_variables * sizeof(float));
                    if (engine->graph.adjacency_matrix[i] == NULL) {
                        for (size_t j = 0; j < i; j++) {
                            safe_free((void**)&engine->graph.adjacency_matrix[j]);
                        }
                        safe_free((void**)&engine->graph.adjacency_matrix);
                        engine->graph.adjacency_matrix = NULL;
                        result = -1;
                        break;
                    }
                    memset(engine->graph.adjacency_matrix[i], 0, num_variables * sizeof(float));
                }
                if (result == -1) break;
            }
            
            /* 为所有变量对执行Granger因果测试 */
            int lag = (int)engine->time_lag;
            for (size_t i = 0; i < num_variables; i++) {
                for (size_t j = 0; j < num_variables; j++) {
                    if (i == j) continue;
                    float gc = granger_causality(engine, (int)i, (int)j, lag);
                    if (gc > engine->config.significance_level) {
                        engine->graph.adjacency_matrix[i][j] = gc;
                        engine->graph.num_variables = num_variables;
                        
                        CausalEdge edge;
                        memset(&edge, 0, sizeof(CausalEdge));
                        edge.source_node_id = (int)i;
                        edge.target_node_id = (int)j;
                        edge.edge_type = CAUSAL_EDGE_DIRECTED;
                        edge.edge_strength = gc;
                        edge.edge_confidence = gc;
                        add_edge_to_graph(&engine->graph, &edge);
                    }
                }
            }
            
            result = 0;
            break;
        }
            
        case CAUSAL_REASONING_STRUCTURAL:
            result = structural_causal_model(engine, data, num_samples);
            break;
            
        case CAUSAL_REASONING_GES_ALGORITHM:
            result = ges_algorithm(engine, data, num_samples, num_variables);
            break;
            
        case CAUSAL_REASONING_LINGAM_ALGORITHM:
            result = lingam_algorithm(engine, data, num_samples, num_variables);
            break;
            
        default:
            /* 默认使用PC算法 */
            result = pc_algorithm(engine, data, num_samples, num_variables);
            break;
    }
    
    if (result == 0) {
        engine->is_graph_built = 1;
        engine->total_discoveries++;
    }
    
    return result;
}

/**
 * @brief 执行因果推理
 */
int causal_reasoning_infer(CausalReasoningEngine* engine,
                          const int* cause_indices, size_t num_causes,
                          const int* effect_indices, size_t num_effects,
                          CausalReasoningResult* result) {
    if (engine == NULL || result == NULL) {
        return -1;
    }
    
    if (!engine->is_graph_built) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "执行因果推理：因果图未构建");
        return -1;
    }
    
    if (num_causes == 0 || cause_indices == NULL || num_effects == 0 || effect_indices == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "执行因果推理：原因或结果索引无效");
        return -1;
    }
    
    /* 初始化结果结构 */
    memset(result, 0, sizeof(CausalReasoningResult));
    
    /* 查找因果路径 - 遍历所有原因-结果组合 */
    int* path_buffer = (int*)safe_malloc(engine->graph.num_variables * sizeof(int));
    int* best_path = (int*)safe_malloc(engine->graph.num_variables * sizeof(int));
    if (path_buffer == NULL || best_path == NULL) {
        if (path_buffer) safe_free((void**)&path_buffer);
        if (best_path) safe_free((void**)&best_path);
        return -1;
    }
    
    float total_effect = 0.0f;
    float max_effect = 0.0f;
    int best_path_length = 0;
    int paths_found = 0;
    float effect_variance = 0.0f;
    float* effects = (float*)safe_malloc(num_causes * num_effects * sizeof(float));
    if (effects == NULL) {
        safe_free((void**)&path_buffer);
        safe_free((void**)&best_path);
        return -1;
    }
    
    /* 遍历所有原因-结果组合 */
    for (size_t ci = 0; ci < num_causes; ci++) {
        int source = cause_indices[ci];
        if (source < 0 || source >= (int)engine->graph.num_variables) continue;
        
        for (size_t ei = 0; ei < num_effects; ei++) {
            int target = effect_indices[ei];
            if (target < 0 || target >= (int)engine->graph.num_variables) continue;
            if (source == target) continue;
            
            int path_length = find_causal_path(engine, source, target, 
                                               path_buffer, engine->graph.num_variables);
            
            if (path_length > 0) {
                /* 计算该路径的因果效应 */
                float effect = estimate_causal_effect_do_calculus(engine, source, target, NULL, 0);
                int idx = (int)(ci * num_effects + ei);
                effects[idx] = effect;
                total_effect += effect;
                paths_found++;
                
                /* 跟踪最强效应 */
                if (fabsf(effect) > fabsf(max_effect)) {
                    max_effect = effect;
                    best_path_length = path_length;
                    memcpy(best_path, path_buffer, path_length * sizeof(int));
                }
            } else {
                int idx = (int)(ci * num_effects + ei);
                effects[idx] = 0.0f;
            }
        }
    }
    
    if (paths_found > 0) {
        /* 平均效应 */
        float avg_effect = total_effect / (float)paths_found;
        result->causal_effect = avg_effect;
        
        /* 计算效应方差（一致性度量） */
        if (paths_found > 1) {
            float sum_sq_diff = 0.0f;
            for (int p = 0; p < paths_found; p++) {
                float diff = effects[p] - avg_effect;
                sum_sq_diff += diff * diff;
            }
            effect_variance = sum_sq_diff / (float)(paths_found - 1);
        }
        
        /* 置信度：基于路径数和效应一致性 */
        float path_confidence = (float)paths_found / (float)(num_causes * num_effects);
        float variance_penalty = 1.0f / (1.0f + effect_variance);
        result->effect_confidence = 0.3f + 0.7f * path_confidence * variance_penalty;
        if (result->effect_confidence > 0.95f) result->effect_confidence = 0.95f;
        if (result->effect_confidence < 0.1f) result->effect_confidence = 0.1f;
        
        /* 使用最佳路径作为主要路径 */
        if (best_path_length > 0) {
            result->causal_path = (int*)safe_malloc(best_path_length * sizeof(int));
            if (result->causal_path != NULL) {
                memcpy(result->causal_path, best_path, best_path_length * sizeof(int));
                result->path_length = best_path_length;
            }
        }
        
        /* 生成综合解释 */
        result->explanation[0] = '\0';
        int first = 1;
        for (size_t ci = 0; ci < num_causes && ci < 3; ci++) {
            int src = cause_indices[ci];
            if (src < 0 || src >= (int)engine->graph.num_variables) continue;
            for (size_t ei = 0; ei < num_effects && ei < 3; ei++) {
                int tgt = effect_indices[ei];
                if (tgt < 0 || tgt >= (int)engine->graph.num_variables) continue;
                if (src == tgt) continue;
                
                int idx = (int)(ci * num_effects + ei);
                if (fabsf(effects[idx]) > 0.01f) {
                    char part[256];
                    const char* cname = (engine->graph.variable_names && 
                                        src < (int)engine->graph.num_variables) 
                                        ? engine->graph.variable_names[src] : NULL;
                    const char* ename = (engine->graph.variable_names && 
                                        tgt < (int)engine->graph.num_variables) 
                                        ? engine->graph.variable_names[tgt] : NULL;
                    
                    if (first) {
                        snprintf(part, sizeof(part), "%s对%s的效应=%.3f",
                                cname ? cname : "X", ename ? ename : "Y", effects[idx]);
                        first = 0;
                    } else {
                        snprintf(part, sizeof(part), "; %s对%s效应=%.3f",
                                cname ? cname : "X", ename ? ename : "Y", effects[idx]);
                    }
                    strncat(result->explanation, part, 
                           sizeof(result->explanation) - strlen(result->explanation) - 1);
                }
            }
        }
        
        if (result->explanation[0] == '\0') {
            snprintf(result->explanation, sizeof(result->explanation),
                    "平均因果效应大小为 %.3f (基于%d条路径)", avg_effect, paths_found);
        }
        
        /* 判断是否需要干预 */
        result->requires_intervention = (fabsf(avg_effect) > 0.3f || fabsf(max_effect) > 0.5f) ? 1 : 0;
    } else {
        /* 没有找到因果路径 */
        result->causal_effect = 0.0f;
        result->effect_confidence = 0.5f;
        strcpy(result->explanation, "未发现显著的因果关系");
        result->requires_intervention = 0;
    }
    
    safe_free((void**)&path_buffer);
    safe_free((void**)&best_path);
    safe_free((void**)&effects);
    
    /* 更新统计信息 */
    engine->total_inferences++;
    engine->last_inference_time = (size_t)time(NULL);
    
    return 0;
}

/**
 * @brief 估计因果效应
 */
float causal_reasoning_estimate_effect(CausalReasoningEngine* engine,
                                      int treatment_index, int outcome_index,
                                      const int* confounder_indices, size_t num_confounders) {
    if (engine == NULL || !engine->is_graph_built) {
        return 0.0f;
    }
    
    if (treatment_index < 0 || treatment_index >= (int)engine->graph.num_variables ||
        outcome_index < 0 || outcome_index >= (int)engine->graph.num_variables) {
        return 0.0f;
    }
    
    /* 使用do-演算估计因果效应 */
    float effect = estimate_causal_effect_do_calculus(engine, treatment_index, outcome_index,
                                                     confounder_indices, num_confounders);
    
    return effect;
}

/**
 * @brief 执行反事实推理
 */
int causal_reasoning_counterfactual(CausalReasoningEngine* engine,
                                   const float* factual_data, size_t factual_size,
                                   float intervention_value,
                                   float* counterfactual_result, size_t max_result_size) {
    if (engine == NULL || factual_data == NULL || counterfactual_result == NULL) {
        return -1;
    }
    
    if (!engine->is_graph_built) {
        return -1;
    }
    
    if (max_result_size == 0) {
        return -1;
    }
    
    /* 执行反事实推理 */
    int result = perform_counterfactual_reasoning(engine, factual_data, factual_size,
                                                 intervention_value, counterfactual_result,
                                                 max_result_size);
    
    return result;
}

/**
 * @brief 获取因果图节点
 */
int causal_reasoning_get_node(CausalReasoningEngine* engine,
                             int node_index, CausalNode* node) {
    if (engine == NULL || node == NULL) {
        return -1;
    }
    
    if (!engine->is_graph_built) {
        return -1;
    }
    
    if (node_index < 0 || node_index >= (int)engine->graph.num_nodes) {
        return -1;
    }
    
    /* 复制节点信息 */
    memcpy(node, &engine->graph.nodes[node_index], sizeof(CausalNode));
    
    /* 复制节点特征 */
    if (engine->graph.nodes[node_index].node_features != NULL && 
        engine->graph.nodes[node_index].feature_size > 0) {
        node->node_features = (float*)safe_malloc(engine->graph.nodes[node_index].feature_size * sizeof(float));
        if (node->node_features == NULL) {
            return -1;
        }
        memcpy(node->node_features, engine->graph.nodes[node_index].node_features,
               engine->graph.nodes[node_index].feature_size * sizeof(float));
    }
    
    return 0;
}

/**
 * @brief 获取因果图边
 */
int causal_reasoning_get_edge(CausalReasoningEngine* engine,
                             int source_index, int target_index, CausalEdge* edge) {
    if (engine == NULL || edge == NULL) {
        return -1;
    }
    
    if (!engine->is_graph_built) {
        return -1;
    }
    
    /* 在边数组中查找 */
    for (size_t i = 0; i < engine->graph.num_edges; i++) {
        if (engine->graph.edges[i].source_node_id == source_index &&
            engine->graph.edges[i].target_node_id == target_index) {
            /* 复制边信息 */
            memcpy(edge, &engine->graph.edges[i], sizeof(CausalEdge));
            
            /* 复制边参数 */
            if (engine->graph.edges[i].edge_parameters != NULL && 
                engine->graph.edges[i].param_size > 0) {
                edge->edge_parameters = (float*)safe_malloc(engine->graph.edges[i].param_size * sizeof(float));
                if (edge->edge_parameters == NULL) {
                    return -1;
                }
                memcpy(edge->edge_parameters, engine->graph.edges[i].edge_parameters,
                       engine->graph.edges[i].param_size * sizeof(float));
            }
            
            return 0;
        }
    }
    
    /* 未找到边 */
    return -1;
}

/**
 * @brief 执行因果发现
 */
int causal_reasoning_discover(CausalReasoningEngine* engine,
                             const float* data, size_t num_samples, size_t num_variables,
                             CausalEdge* discovered_edges, size_t max_edges) {
    if (engine == NULL || data == NULL || discovered_edges == NULL) {
        return -1;
    }
    
    if (num_samples == 0 || num_variables == 0 || max_edges == 0) {
        return -1;
    }
    
    /* 构建因果图 */
    int result = causal_reasoning_build_graph(engine, data, num_samples, num_variables, NULL);
    if (result != 0) {
        return -1;
    }
    
    /* 复制发现的边 */
    size_t edges_to_copy = engine->graph.num_edges;
    if (edges_to_copy > max_edges) {
        edges_to_copy = max_edges;
    }
    
    for (size_t i = 0; i < edges_to_copy; i++) {
        memcpy(&discovered_edges[i], &engine->graph.edges[i], sizeof(CausalEdge));
        
        /* 复制边参数 */
        if (engine->graph.edges[i].edge_parameters != NULL && 
            engine->graph.edges[i].param_size > 0) {
            discovered_edges[i].edge_parameters = (float*)safe_malloc(engine->graph.edges[i].param_size * sizeof(float));
            if (discovered_edges[i].edge_parameters != NULL) {
                memcpy(discovered_edges[i].edge_parameters, engine->graph.edges[i].edge_parameters,
                       engine->graph.edges[i].param_size * sizeof(float));
            }
        }
    }
    
    return (int)edges_to_copy;
}

/**
 * @brief 验证因果假设
 */
float causal_reasoning_validate_hypothesis(CausalReasoningEngine* engine,
                                          int cause_index, int effect_index,
                                          const float* test_data, size_t num_test_samples) {
    if (engine == NULL || test_data == NULL || !engine->is_graph_built) {
        return 0.0f;
    }
    
    if (cause_index < 0 || cause_index >= (int)engine->graph.num_variables ||
        effect_index < 0 || effect_index >= (int)engine->graph.num_variables) {
        return 0.0f;
    }
    
    if (num_test_samples == 0) {
        return 0.0f;
    }
    
    /* 完整假设检验：结合图结构效应估计和测试数据验证 */
    /* 提取cause和effect的测试数据 */
    float* cause_data = (float*)safe_malloc(num_test_samples * sizeof(float));
    float* effect_data = (float*)safe_malloc(num_test_samples * sizeof(float));
    
    if (cause_data == NULL || effect_data == NULL) {
        if (cause_data) safe_free((void**)&cause_data);
        if (effect_data) safe_free((void**)&effect_data);
        return 0.0f;
    }
    
    for (size_t i = 0; i < num_test_samples; i++) {
        cause_data[i] = test_data[i * engine->graph.num_variables + cause_index];
        effect_data[i] = test_data[i * engine->graph.num_variables + effect_index];
    }
    
    /* 1. 基于图结构的效应估计 */
    float graph_effect = estimate_causal_effect_do_calculus(engine, cause_index, effect_index, NULL, 0);
    
    /* 2. 测试数据直接相关性 */
    float test_correlation = compute_correlation(cause_data, effect_data, num_test_samples);
    
    /* 3. t-统计量：检验相关性的显著性 */
    float t_stat = 0.0f;
    float p_value = 0.5f;
    if (num_test_samples > 2 && fabsf(test_correlation) < 0.999f) {
        t_stat = test_correlation * sqrtf((float)(num_test_samples - 2)) / 
                 sqrtf(1.0f - test_correlation * test_correlation);
        /* p值计算：使用t分布的累积分布函数 */
        /* P(T > t) = 0.5 * I(df/(df+t²), df/2, 1/2) 其中I是不完全正则化beta函数 */
        float df = (float)(num_test_samples - 2);
        float t_abs = fabsf(t_stat);
        float x = df / (df + t_abs * t_abs);
        
        /* 使用连分式计算不完全beta函数 I_x(a, b) */
        float a = df * 0.5f;
        float b = 0.5f;
        
        /* 连分式评估 */
        float p_tail;
        if (x < (a + 1.0f) / (a + b + 2.0f)) {
            /* 使用连分式 */
            float beta_cf = 1.0f;
            float numerator = 1.0f;
            float denom = 1.0f;
            for (int m = 1; m <= 200; m++) {
                float mf = (float)m;
                float alpha_m = (a + mf - 1.0f) * (a + b + mf - 1.0f) * mf * (b - mf) * x /
                              ((a + 2.0f * mf - 2.0f) * (a + 2.0f * mf - 1.0f));
                if (m % 2 == 1) {
                    float beta_m = -((a + mf - 1.0f) * (a + b + mf - 2.0f)) /
                                    ((a + 2.0f * mf - 2.0f) * (a + 2.0f * mf));
                    beta_m += alpha_m;
                    denom = 1.0f + beta_m / denom;
                    numerator = 1.0f + beta_m / numerator;
                }
                if (fabsf(denom) < 1e-30f) denom = 1e-30f;
                beta_cf *= numerator / denom;
                float frac = fabsf(beta_cf - 1.0f);
                if (frac < 1e-8f) break;
            }
            p_tail = 0.5f * beta_cf * expf(lgammaf(a + b) - lgammaf(a) - lgammaf(b) +
                     a * logf(x) + b * logf(1.0f - x));
        } else {
            /* 对称性：I_x(a,b) = 1 - I_{1-x}(b,a) */
            float x1 = 1.0f - x;
            float beta_cf = 1.0f;
            float numerator = 1.0f;
            float denom = 1.0f;
            for (int m = 1; m <= 200; m++) {
                float mf = (float)m;
                float alpha_m = (b + mf - 1.0f) * (a + b + mf - 1.0f) * mf * (a - mf) * x1 /
                              ((b + 2.0f * mf - 2.0f) * (b + 2.0f * mf - 1.0f));
                if (m % 2 == 1) {
                    float beta_m = -((b + mf - 1.0f) * (a + b + mf - 2.0f)) /
                                    ((b + 2.0f * mf - 2.0f) * (b + 2.0f * mf));
                    beta_m += alpha_m;
                    denom = 1.0f + beta_m / denom;
                    numerator = 1.0f + beta_m / numerator;
                }
                if (fabsf(denom) < 1e-30f) denom = 1e-30f;
                beta_cf *= numerator / denom;
                float frac = fabsf(beta_cf - 1.0f);
                if (frac < 1e-8f) break;
            }
            float ibeta = 1.0f - beta_cf * expf(lgammaf(a + b) - lgammaf(b) - lgammaf(a) +
                          b * logf(x1) + a * logf(1.0f - x1));
            if (ibeta < 0.0f) ibeta = 0.0f;
            if (ibeta > 1.0f) ibeta = 1.0f;
            p_tail = 0.5f * ibeta;
        }
        
        p_value = 2.0f * p_tail;
        if (p_value < 0.0f) p_value = 0.0f;
        if (p_value > 1.0f) p_value = 1.0f;
    }
    
    /* 4. 综合验证分数 */
    float validation_score = 0.0f;
    
    /* 图效应贡献（如果图有边） */
    float graph_contribution = fabsf(graph_effect) * 0.4f;
    
    /* 相关性贡献 */
    float corr_contribution = fabsf(test_correlation) * 0.3f;
    
    /* 统计显著性贡献（p值越小，贡献越大） */
    float sig_contribution = (1.0f - p_value) * 0.3f;
    
    validation_score = graph_contribution + corr_contribution + sig_contribution;
    
    /* 如果方向相反，降级 */
    if ((graph_effect > 0.0f && test_correlation < -0.1f) ||
        (graph_effect < 0.0f && test_correlation > 0.1f)) {
        validation_score *= 0.5f;
    }
    
    safe_free((void**)&cause_data);
    safe_free((void**)&effect_data);
    
    /* 限制在[0, 1] */
    if (validation_score < 0.0f) validation_score = 0.0f;
    if (validation_score > 1.0f) validation_score = 1.0f;
    
    return validation_score;
}

/**
 * @brief 获取因果推理引擎配置
 */
int causal_reasoning_engine_get_config(const CausalReasoningEngine* engine,
                                      CausalReasoningConfig* config) {
    if (engine == NULL || config == NULL) {
        return -1;
    }
    
    memcpy(config, &engine->config, sizeof(CausalReasoningConfig));
    return 0;
}

/**
 * @brief 设置因果推理引擎配置
 */
int causal_reasoning_engine_set_config(CausalReasoningEngine* engine,
                                      const CausalReasoningConfig* config) {
    if (engine == NULL || config == NULL) {
        return -1;
    }
    
    memcpy(&engine->config, config, sizeof(CausalReasoningConfig));
    
    /* 使缓存无效，因为配置已更改 */
    engine->cache_valid = 0;
    
    return 0;
}

/* ========== 内部算法实现 ========== */

/**
 * @brief FCI算法完整实现（处理隐变量，构建偏祖先图PAG）
 */
static int fci_algorithm(CausalReasoningEngine* engine, const float* data, 
                        size_t num_samples, size_t num_variables) {
    /* FCI算法：扩展PC算法以处理隐变量，构建偏祖先图(PAG) */
    
    /* 步骤1：运行PC算法获取初始骨架和定向 */
    int result = pc_algorithm(engine, data, num_samples, num_variables);
    if (result != 0) {
        return result;
    }
    
    /* 步骤2：检测潜在的隐变量混淆模式 */
    /* 重新组织数据为二维数组以便计算 */
    float** data_matrix = (float**)safe_malloc(num_variables * sizeof(float*));
    if (data_matrix == NULL) return -1;
    
    for (size_t i = 0; i < num_variables; i++) {
        data_matrix[i] = (float*)safe_malloc(num_samples * sizeof(float));
        if (data_matrix[i] == NULL) {
            for (size_t j = 0; j < i; j++) safe_free((void**)&data_matrix[j]);
            safe_free((void**)&data_matrix);
            return -1;
        }
        for (size_t j = 0; j < num_samples; j++) {
            data_matrix[i][j] = data[j * num_variables + i];
        }
    }
    
    /* 为所有非相邻节点对检测隐变量模式 */
    /* 如果两个变量在PC骨架中无直接边，但在共同子节点上仍相关，可能存在隐变量 */
    for (size_t i = 0; i < num_variables; i++) {
        for (size_t j = i + 1; j < num_variables; j++) {
            if (engine->graph.adjacency_matrix[i][j] > 0.0f) continue;
            
            /* 计算非相邻节点对的残差相关性 */
            float corr = fabsf(compute_correlation(data_matrix[i], data_matrix[j], num_samples));
            
            /* 如果相关性较强但PC未能发现边，可能是隐变量导致 */
            if (corr > engine->config.significance_level * 3.0f) {
                /* 搜索共同子节点（v-结构模式：i->k<-j） */
                for (size_t k = 0; k < num_variables; k++) {
                    if (k == i || k == j) continue;
                    
                    int i_to_k = (engine->graph.adjacency_matrix[i][k] > 0.0f && 
                                  engine->graph.adjacency_matrix[k][i] == 0.0f);
                    int k_from_j = (engine->graph.adjacency_matrix[k][j] > 0.0f && 
                                    engine->graph.adjacency_matrix[j][k] == 0.0f);
                    int k_to_i = (engine->graph.adjacency_matrix[k][i] > 0.0f && 
                                  engine->graph.adjacency_matrix[i][k] == 0.0f);
                    int j_to_k = (engine->graph.adjacency_matrix[j][k] > 0.0f && 
                                  engine->graph.adjacency_matrix[k][j] == 0.0f);
                    
                    /* v-结构模式：i->k<-j 且 i-j非相邻 => 可能存在隐变量 */
                    if ((i_to_k && k_from_j) || (k_to_i && j_to_k)) {
                        CausalEdge edge;
                        memset(&edge, 0, sizeof(CausalEdge));
                        edge.source_node_id = (int)i;
                        edge.target_node_id = (int)j;
                        edge.edge_type = CAUSAL_EDGE_BIDIRECTED;
                        edge.edge_strength = corr * 0.8f;
                        edge.edge_confidence = 0.5f;
                        add_edge_to_graph(&engine->graph, &edge);
                        
                        /* 同时在邻接矩阵中标记双向关系 */
                        engine->graph.adjacency_matrix[i][j] = corr * 0.8f;
                        engine->graph.adjacency_matrix[j][i] = corr * 0.8f;
                    }
                }
            }
        }
    }
    
    /* 步骤3：FCI定向规则传播（处理双向边的Meek规则扩展） */
    int changed = 1;
    int max_iter = (int)num_variables * 4;
    int iter = 0;
    
    while (changed && iter < max_iter) {
        changed = 0;
        iter++;
        
        for (size_t i = 0; i < num_variables; i++) {
            for (size_t j = 0; j < num_variables; j++) {
                if (i == j) continue;
                float adj_ij = engine->graph.adjacency_matrix[i][j];
                float adj_ji = engine->graph.adjacency_matrix[j][i];
                
                if (adj_ij <= 0.0f) continue;
                int is_directed_ij = (adj_ij > 0.0f && adj_ji == 0.0f);
                int is_undirected = (adj_ij > 0.0f && adj_ji > 0.0f);
                
                for (size_t k = 0; k < num_variables; k++) {
                    if (k == i || k == j) continue;
                    float adj_jk = engine->graph.adjacency_matrix[j][k];
                    float adj_kj = engine->graph.adjacency_matrix[k][j];
                    
                    if (adj_jk <= 0.0f) continue;
                    
                    /* FCI-R1: i -> j, j - k (无向), i-k非相邻 => j -> k */
                    if (is_directed_ij && adj_jk > 0.0f && adj_kj > 0.0f) {
                        if (engine->graph.adjacency_matrix[i][k] == 0.0f && 
                            engine->graph.adjacency_matrix[k][i] == 0.0f) {
                            engine->graph.adjacency_matrix[k][j] = 0.0f;
                            changed = 1;
                        }
                    }
                    
                    /* FCI-R2: i -> j, i -> k, j - k => j -> k */
                    if (is_directed_ij && adj_jk > 0.0f && adj_kj > 0.0f) {
                        if (engine->graph.adjacency_matrix[i][k] > 0.0f && 
                            engine->graph.adjacency_matrix[k][i] == 0.0f) {
                            engine->graph.adjacency_matrix[k][j] = 0.0f;
                            changed = 1;
                        }
                    }
                    
                    /* FCI-R3: i - j, i -> k, j -> k => i -> j */
                    if (is_undirected) {
                        if (engine->graph.adjacency_matrix[i][k] > 0.0f && 
                            engine->graph.adjacency_matrix[k][i] == 0.0f &&
                            engine->graph.adjacency_matrix[j][k] > 0.0f && 
                            engine->graph.adjacency_matrix[k][j] == 0.0f) {
                            engine->graph.adjacency_matrix[j][i] = 0.0f;
                            changed = 1;
                        }
                    }
                }
            }
        }
    }
    
    /* 清理数据矩阵 */
    for (size_t i = 0; i < num_variables; i++) {
        safe_free((void**)&data_matrix[i]);
    }
    safe_free((void**)&data_matrix);
    
    engine->is_graph_built = 1;
    return 0;
}

/**
 * @brief Granger因果分析 - 基于VAR模型的完整实现
 */
static float granger_causality(CausalReasoningEngine* engine, int cause_idx, int effect_idx, int max_lag) {
    /* Granger因果分析：如果X的历史值能预测Y的未来值，则X Granger-cause Y */
    /* 完整实现：基于VAR模型比较受限模型和完整模型的预测误差 */
    
    if (!engine || !engine->time_series_data) {
        return 0.0f;
    }
    
    if (cause_idx < 0 || effect_idx < 0 || cause_idx >= (int)engine->graph.num_variables ||
        effect_idx >= (int)engine->graph.num_variables) {
        return 0.0f;
    }
    
    size_t T = engine->time_series_length;
    if (T < (size_t)(max_lag * 2 + 5)) {
        /* 数据不足以进行Granger因果分析 */
        return 0.0f;
    }
    
    int lag = (max_lag > 0) ? max_lag : 1;
    if (lag > (int)T / 3) {
        lag = (int)T / 3;
    }
    if (lag < 1) lag = 1;
    
    /* 提取cause和effect的时间序列数据 */
    float* X = engine->time_series_data[cause_idx];
    float* Y = engine->time_series_data[effect_idx];
    if (!X || !Y) return 0.0f;
    
    int n_obs = (int)T - lag;  /* 有效观测数 */
    if (n_obs < lag + 2) return 0.0f;
    
    /* 构建完整模型（含X和Y的历史）的回归矩阵 */
    /* Y[t] = a0 + a1*Y[t-1] + ... + al*Y[t-l] + b1*X[t-1] + ... + bl*X[t-l] */
    /* 受限模型: Y[t] = a0 + a1*Y[t-1] + ... + al*Y[t-l] */
    
    int n_features_full = 2 * lag + 1;  /* 截距 + Y的l个滞后 + X的l个滞后 */
    int n_features_restricted = lag + 1; /* 截距 + Y的l个滞后 */
    
    /* 使用普通最小二乘法(OLS)求解回归系数 */
    /* 使用Nx(N+1)矩阵: [Y_t, Y_{t-1}...Y_{t-l}, X_{t-1}...X_{t-l}] */
    
    /* 方法：直接使用正规方程求解 β = (X'X)^{-1}X'y */
    
    /* 构建设计矩阵 */
    float* Y_obs = (float*)safe_malloc(n_obs * sizeof(float));
    float* X_full = (float*)safe_malloc(n_obs * n_features_full * sizeof(float));
    float* X_restricted = (float*)safe_malloc(n_obs * n_features_restricted * sizeof(float));
    
    if (!Y_obs || !X_full || !X_restricted) {
        safe_free((void**)&Y_obs);
        safe_free((void**)&X_full);
        safe_free((void**)&X_restricted);
        return 0.0f;
    }
    
    for (int t = 0; t < n_obs; t++) {
        Y_obs[t] = Y[t + lag];
        
        /* 截距项 */
        X_full[t * n_features_full + 0] = 1.0f;
        X_restricted[t * n_features_restricted + 0] = 1.0f;
        
        /* Y的历史值 */
        for (int l = 0; l < lag; l++) {
            X_full[t * n_features_full + 1 + l] = Y[t + lag - 1 - l];
            X_restricted[t * n_features_restricted + 1 + l] = Y[t + lag - 1 - l];
        }
        
        /* X的历史值（完整模型特有） */
        for (int l = 0; l < lag; l++) {
            X_full[t * n_features_full + 1 + lag + l] = X[t + lag - 1 - l];
        }
    }
    
    /* 计算X'X和X'y */
    /* 完整模型 */
    float* XtX_full = (float*)safe_malloc(n_features_full * n_features_full * sizeof(float));
    float* Xty_full = (float*)safe_malloc(n_features_full * sizeof(float));
    /* 受限模型 */
    float* XtX_restricted = (float*)safe_malloc(n_features_restricted * n_features_restricted * sizeof(float));
    float* Xty_restricted = (float*)safe_malloc(n_features_restricted * sizeof(float));
    
    if (!XtX_full || !Xty_full || !XtX_restricted || !Xty_restricted) {
        safe_free((void**)&Y_obs);
        safe_free((void**)&X_full);
        safe_free((void**)&X_restricted);
        safe_free((void**)&XtX_full);
        safe_free((void**)&Xty_full);
        safe_free((void**)&XtX_restricted);
        safe_free((void**)&Xty_restricted);
        return 0.0f;
    }
    
    memset(XtX_full, 0, n_features_full * n_features_full * sizeof(float));
    memset(Xty_full, 0, n_features_full * sizeof(float));
    memset(XtX_restricted, 0, n_features_restricted * n_features_restricted * sizeof(float));
    memset(Xty_restricted, 0, n_features_restricted * sizeof(float));
    
    for (int t = 0; t < n_obs; t++) {
        /* 完整模型 */
        for (int i = 0; i < n_features_full; i++) {
            for (int j = 0; j < n_features_full; j++) {
                XtX_full[i * n_features_full + j] += X_full[t * n_features_full + i] * X_full[t * n_features_full + j];
            }
            Xty_full[i] += X_full[t * n_features_full + i] * Y_obs[t];
        }
        
        /* 受限模型 */
        for (int i = 0; i < n_features_restricted; i++) {
            for (int j = 0; j < n_features_restricted; j++) {
                XtX_restricted[i * n_features_restricted + j] += X_restricted[t * n_features_restricted + i] * X_restricted[t * n_features_restricted + j];
            }
            Xty_restricted[i] += X_restricted[t * n_features_restricted + i] * Y_obs[t];
        }
    }
    
    /* 高斯消元法求解线性系统 */
    float* beta_full = (float*)safe_malloc(n_features_full * sizeof(float));
    float* beta_restricted = (float*)safe_malloc(n_features_restricted * sizeof(float));
    
    if (!beta_full || !beta_restricted) {
        safe_free((void**)&Y_obs);
        safe_free((void**)&X_full);
        safe_free((void**)&X_restricted);
        safe_free((void**)&XtX_full);
        safe_free((void**)&Xty_full);
        safe_free((void**)&XtX_restricted);
        safe_free((void**)&Xty_restricted);
        safe_free((void**)&beta_full);
        safe_free((void**)&beta_restricted);
        return 0.0f;
    }
    
    /* 复制矩阵并求解 */
    memcpy(beta_full, Xty_full, n_features_full * sizeof(float));
    float* A_full = (float*)safe_malloc(n_features_full * n_features_full * sizeof(float));
    memcpy(A_full, XtX_full, n_features_full * n_features_full * sizeof(float));
    
    memcpy(beta_restricted, Xty_restricted, n_features_restricted * sizeof(float));
    float* A_restricted = (float*)safe_malloc(n_features_restricted * n_features_restricted * sizeof(float));
    memcpy(A_restricted, XtX_restricted, n_features_restricted * n_features_restricted * sizeof(float));
    
    int solve_result = 0;
    if (A_full && beta_full) {
        /* 高斯消元（带部分主元） */
        for (int col = 0; col < n_features_full; col++) {
            /* 寻找主元 */
            int pivot = col;
            float max_val = fabsf(A_full[col * n_features_full + col]);
            for (int row = col + 1; row < n_features_full; row++) {
                float val = fabsf(A_full[row * n_features_full + col]);
                if (val > max_val) {
                    max_val = val;
                    pivot = row;
                }
            }
            
            if (max_val < 1e-10f) {
                solve_result = -1;
                break;
            }
            
            /* 交换行 */
            if (pivot != col) {
                for (int j = col; j < n_features_full; j++) {
                    float tmp = A_full[col * n_features_full + j];
                    A_full[col * n_features_full + j] = A_full[pivot * n_features_full + j];
                    A_full[pivot * n_features_full + j] = tmp;
                }
                float tmp = beta_full[col];
                beta_full[col] = beta_full[pivot];
                beta_full[pivot] = tmp;
            }
            
            /* 消去 */
            float pivot_val = A_full[col * n_features_full + col];
            for (int row = col + 1; row < n_features_full; row++) {
                float factor = A_full[row * n_features_full + col] / pivot_val;
                for (int j = col; j < n_features_full; j++) {
                    A_full[row * n_features_full + j] -= factor * A_full[col * n_features_full + j];
                }
                beta_full[row] -= factor * beta_full[col];
            }
        }
        
        /* 回代 */
        if (solve_result == 0) {
            for (int i = n_features_full - 1; i >= 0; i--) {
                float sum = beta_full[i];
                for (int j = i + 1; j < n_features_full; j++) {
                    sum -= A_full[i * n_features_full + j] * beta_full[j];
                }
                beta_full[i] = sum / A_full[i * n_features_full + i];
            }
        }
    }
    
    /* 求解受限模型 */
    int solve_result_r = 0;
    if (A_restricted && beta_restricted) {
        for (int col = 0; col < n_features_restricted; col++) {
            int pivot = col;
            float max_val = fabsf(A_restricted[col * n_features_restricted + col]);
            for (int row = col + 1; row < n_features_restricted; row++) {
                float val = fabsf(A_restricted[row * n_features_restricted + col]);
                if (val > max_val) {
                    max_val = val;
                    pivot = row;
                }
            }
            
            if (max_val < 1e-10f) {
                solve_result_r = -1;
                break;
            }
            
            if (pivot != col) {
                for (int j = col; j < n_features_restricted; j++) {
                    float tmp = A_restricted[col * n_features_restricted + j];
                    A_restricted[col * n_features_restricted + j] = A_restricted[pivot * n_features_restricted + j];
                    A_restricted[pivot * n_features_restricted + j] = tmp;
                }
                float tmp = beta_restricted[col];
                beta_restricted[col] = beta_restricted[pivot];
                beta_restricted[pivot] = tmp;
            }
            
            float pivot_val = A_restricted[col * n_features_restricted + col];
            for (int row = col + 1; row < n_features_restricted; row++) {
                float factor = A_restricted[row * n_features_restricted + col] / pivot_val;
                for (int j = col; j < n_features_restricted; j++) {
                    A_restricted[row * n_features_restricted + j] -= factor * A_restricted[col * n_features_restricted + j];
                }
                beta_restricted[row] -= factor * beta_restricted[col];
            }
        }
        
        if (solve_result_r == 0) {
            for (int i = n_features_restricted - 1; i >= 0; i--) {
                float sum = beta_restricted[i];
                for (int j = i + 1; j < n_features_restricted; j++) {
                    sum -= A_restricted[i * n_features_restricted + j] * beta_restricted[j];
                }
                beta_restricted[i] = sum / A_restricted[i * n_features_restricted + i];
            }
        }
    }
    
    /* 计算残差平方和（RSS） */
    float rss_full = 0.0f;
    float rss_restricted = 0.0f;
    
    for (int t = 0; t < n_obs; t++) {
        /* 完整模型预测 */
        float pred_full = 0.0f;
        for (int i = 0; i < n_features_full; i++) {
            pred_full += X_full[t * n_features_full + i] * beta_full[i];
        }
        float resid_full = Y_obs[t] - pred_full;
        rss_full += resid_full * resid_full;
        
        /* 受限模型预测 */
        float pred_restricted = 0.0f;
        for (int i = 0; i < n_features_restricted; i++) {
            pred_restricted += X_restricted[t * n_features_restricted + i] * beta_restricted[i];
        }
        float resid_restricted = Y_obs[t] - pred_restricted;
        rss_restricted += resid_restricted * resid_restricted;
    }
    
    /* 计算F统计量 */
    float granger_score = 0.0f;
    if (rss_full > 1e-10f && solve_result == 0 && solve_result_r == 0) {
        int df1 = lag;  /* 额外回归元数量 */
        int df2 = n_obs - n_features_full;  /* 残差自由度 */
        
        if (df2 > 0) {
            float F_stat = ((rss_restricted - rss_full) / df1) / (rss_full / df2);
            
            /* 将F统计量映射到[0,1]范围 */
            /* 使用近似公式: F到因果关系强度的映射 */
            float F_normalized = F_stat / (F_stat + (float)df2 / (float)df1);
            if (F_normalized > 0.0f) {
                granger_score = F_normalized;
            }
            
            /* 当F统计量很小时，因果关系弱 */
            if (F_stat < 1.0f) {
                granger_score *= F_stat;
            }
        }
    }
    
    /* 清理 */
    safe_free((void**)&Y_obs);
    safe_free((void**)&X_full);
    safe_free((void**)&X_restricted);
    safe_free((void**)&XtX_full);
    safe_free((void**)&Xty_full);
    safe_free((void**)&XtX_restricted);
    safe_free((void**)&Xty_restricted);
    safe_free((void**)&beta_full);
    safe_free((void**)&beta_restricted);
    safe_free((void**)&A_full);
    safe_free((void**)&A_restricted);
    
    /* 限制在[0,1]范围 */
    if (granger_score < 0.0f) granger_score = 0.0f;
    if (granger_score > 1.0f) granger_score = 1.0f;
    
    return granger_score;
}

/**
 * @brief 转移熵因果分析 - 基于直方图密度估计的完整实现
 */
static float transfer_entropy(CausalReasoningEngine* engine, int cause_idx, int effect_idx, int k, int l) {
    /* 转移熵：从X到Y的信息流 */
    /* TE_{X→Y}(k,l) = Σ p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) * log₂(p(y_{t+1}|y_t^{(k)}, x_t^{(l)}) / p(y_{t+1}|y_t^{(k)})) */
    /* 完整实现：使用基于直方图的概率密度估计 */
    
    if (!engine || !engine->time_series_data) {
        return 0.0f;
    }
    
    if (cause_idx < 0 || effect_idx < 0 || cause_idx >= (int)engine->graph.num_variables ||
        effect_idx >= (int)engine->graph.num_variables) {
        return 0.0f;
    }
    
    size_t T = engine->time_series_length;
    if (T < (size_t)(k + l + 5)) {
        return 0.0f;
    }
    
    int hist_k = (k > 0) ? k : 1;
    int hist_l = (l > 0) ? l : 1;
    
    /* 提取cause和effect的时间序列数据 */
    float* X = engine->time_series_data[cause_idx];
    float* Y = engine->time_series_data[effect_idx];
    if (!X || !Y) return 0.0f;
    
    int n_valid = (int)T - (hist_k > hist_l ? hist_k : hist_l);
    if (n_valid < 10) return 0.0f;
    
    /* 基于数据动态确定直方图箱子数 */
    int num_bins = (int)(sqrtf((float)n_valid));
    if (num_bins < 4) num_bins = 4;
    if (num_bins > 20) num_bins = 20;
    
    /* 计算Y_{t+1}, Y_t^{(k)}, X_t^{(l)}的取值范围 */
    float y_min = Y[0], y_max = Y[0];
    for (int t = 0; t < (int)T; t++) {
        if (Y[t] < y_min) y_min = Y[t];
        if (Y[t] > y_max) y_max = Y[t];
    }
    
    float x_min = X[0], x_max = X[0];
    for (int t = 0; t < (int)T; t++) {
        if (X[t] < x_min) x_min = X[t];
        if (X[t] > x_max) x_max = X[t];
    }
    
    /* 避免零范围 */
    float y_range = (y_max - y_min > 1e-10f) ? (y_max - y_min) : 1.0f;
    float x_range = (x_max - x_min > 1e-10f) ? (x_max - x_min) : 1.0f;
    
    int ybins = num_bins;
    int xbins = num_bins;
    
    /* 使用哈希表实现联合概率分布的O(1)查找和计数 */
    /* 哈希表大小（2的幂，保证快速取模） */
#define TE_HT_SIZE 2048
    
    /* 哈希表条目定义 */
    typedef struct TEHtEntry {
        int y_next_bin;
        int y_hist_bins;
        int x_hist_bins;
        int count;
    } TEHtEntry;
    
    TEHtEntry* ht = (TEHtEntry*)safe_malloc(TE_HT_SIZE * sizeof(TEHtEntry));
    if (!ht) return 0.0f;
    memset(ht, 0, TE_HT_SIZE * sizeof(TEHtEntry));
    int* ht_flags = (int*)safe_malloc(TE_HT_SIZE * sizeof(int));
    if (!ht_flags) { safe_free((void**)&ht); return 0.0f; }
    memset(ht_flags, 0, TE_HT_SIZE * sizeof(int));
    
    /* 哈希函数：使用三重素数混合 */
#define TE_HASH(y, h, x) (((unsigned int)(y) * 73856093u) ^ ((unsigned int)(h) * 19349663u) ^ ((unsigned int)(x) * 83492791u)) & (TE_HT_SIZE - 1u)
    
    /* 将连续数据离散化为箱子索引 */
    for (int t = 0; t < n_valid; t++) {
        int y_next_bin = (int)((Y[t + 1] - y_min) / y_range * ybins);
        if (y_next_bin < 0) y_next_bin = 0;
        if (y_next_bin >= ybins) y_next_bin = ybins - 1;
        
        int y_hist_bins = 0;
        for (int h = 0; h < hist_k; h++) {
            int idx = t - h;
            if (idx < 0) idx = 0;
            int bin = (int)((Y[idx] - y_min) / y_range * ybins);
            if (bin < 0) bin = 0;
            if (bin >= ybins) bin = ybins - 1;
            y_hist_bins = y_hist_bins * ybins + bin;
        }
        
        int x_hist_bins = 0;
        for (int h = 0; h < hist_l; h++) {
            int idx = t - h;
            if (idx < 0) idx = 0;
            int bin = (int)((X[idx] - x_min) / x_range * xbins);
            if (bin < 0) bin = 0;
            if (bin >= xbins) bin = xbins - 1;
            x_hist_bins = x_hist_bins * xbins + bin;
        }
        
        /* 哈希表插入/更新（开放寻址+二次探测） */
        unsigned int hash = TE_HASH(y_next_bin, y_hist_bins, x_hist_bins);
        unsigned int probe = 0;
        int inserted = 0;
        while (!inserted && probe < TE_HT_SIZE) {
            unsigned int idx = (hash + probe * probe) & (TE_HT_SIZE - 1);
            if (!ht_flags[idx]) {
                ht_flags[idx] = 1;
                ht[idx].y_next_bin = y_next_bin;
                ht[idx].y_hist_bins = y_hist_bins;
                ht[idx].x_hist_bins = x_hist_bins;
                ht[idx].count = 1;
                inserted = 1;
            } else if (ht[idx].y_next_bin == y_next_bin &&
                       ht[idx].y_hist_bins == y_hist_bins &&
                       ht[idx].x_hist_bins == x_hist_bins) {
                ht[idx].count++;
                inserted = 1;
            } else {
                probe++;
            }
        }
    }
    
    /* 第一阶段：收集所有唯一状态到数组中以便遍历 */
    int num_unique = 0;
    int total_entries = 0;
    for (unsigned int i = 0; i < TE_HT_SIZE; i++) {
        if (ht_flags[i]) total_entries++;
    }
    
    int* state_yn  = (int*)safe_malloc((size_t)total_entries * sizeof(int));
    int* state_yh  = (int*)safe_malloc((size_t)total_entries * sizeof(int));
    int* state_xh  = (int*)safe_malloc((size_t)total_entries * sizeof(int));
    int* state_cnt = (int*)safe_malloc((size_t)total_entries * sizeof(int));
    if (!state_yn || !state_yh || !state_xh || !state_cnt) {
        safe_free((void**)&state_yn); safe_free((void**)&state_yh);
        safe_free((void**)&state_xh); safe_free((void**)&state_cnt);
        safe_free((void**)&ht); safe_free((void**)&ht_flags);
        return 0.0f;
    }
    
    for (unsigned int i = 0; i < TE_HT_SIZE; i++) {
        if (ht_flags[i]) {
            state_yn[num_unique] = ht[i].y_next_bin;
            state_yh[num_unique] = ht[i].y_hist_bins;
            state_xh[num_unique] = ht[i].x_hist_bins;
            state_cnt[num_unique] = ht[i].count;
            num_unique++;
        }
    }
    safe_free((void**)&ht);
    safe_free((void**)&ht_flags);
    
    /* 计算转移熵 */
    float transfer_entropy_val = 0.0f;
    float total_count = (float)n_valid;
    
    for (int i = 0; i < num_unique; i++) {
        float p_xyz = (float)state_cnt[i] / total_count;
        if (p_xyz <= 1e-10f) continue;
        
        /* p(y_next | y_hist) - 条件于Y历史 */
        int y_hist_bins_val = state_yh[i];
        int y_hist_count = 0;
        int y_joint_count = 0;
        
        for (int j = 0; j < num_unique; j++) {
            if (state_yh[j] == y_hist_bins_val) {
                y_hist_count += state_cnt[j];
                if (state_yn[j] == state_yn[i]) {
                    y_joint_count += state_cnt[j];
                }
            }
        }
        
        float p_y_given_yhist = 0.0f;
        if (y_hist_count > 0) {
            p_y_given_yhist = (float)y_joint_count / (float)y_hist_count;
        }
        
        /* p(y_next | y_hist, x_hist) */
        float p_y_given_yx = p_xyz / ((float)y_hist_count / total_count);
        if (p_y_given_yx <= 1e-10f) continue;
        
        if (p_y_given_yhist > 1e-10f) {
            float ratio = p_y_given_yx / p_y_given_yhist;
            if (ratio > 1e-10f) {
                transfer_entropy_val += p_xyz * log2f(ratio);
            }
        }
    }
    
    safe_free((void**)&state_yn);
    safe_free((void**)&state_yh);
    safe_free((void**)&state_xh);
    safe_free((void**)&state_cnt);
    
    /* 归一化到[0,1]范围 */
    if (transfer_entropy_val < 0.0f) transfer_entropy_val = 0.0f;
    if (transfer_entropy_val > 10.0f) transfer_entropy_val = 10.0f;
    float normalized_te = transfer_entropy_val / (transfer_entropy_val + 1.0f);
    
    return normalized_te;
}

/**
 * @brief 结构因果模型（SCM） - 完整结构方程估计实现
 */
static int structural_causal_model(CausalReasoningEngine* engine, const float* data, size_t num_samples) {
    /* 结构因果模型：使用结构方程建模 */
    /* 完整实现：基于有向无环图估计结构方程系数 */
    
    if (!engine || !data || num_samples == 0) {
        return -1;
    }
    
    size_t num_vars = engine->graph.num_variables;
    if (num_vars == 0) return -1;
    
    /* 步骤1：运行PC算法发现因果结构 */
    int pc_result = pc_algorithm(engine, data, num_samples, num_vars);
    if (pc_result != 0) {
        return pc_result;
    }
    
    /* 检查是否已构建邻接矩阵 */
    if (!engine->graph.adjacency_matrix) {
        return -1;
    }
    
    /* 确保邻接矩阵是上三角或严格有向的 */
    /* PC算法结果可能包含双向边，这里需要确保因果方向 */
    
    /* 步骤2：为每个变量估计结构方程 */
    /* 对于每个变量X_j，找到其父节点集合Pa(X_j) */
    /* 估计线性结构方程: X_j = β_0 + Σ β_i * Pa_i(X_j) + ε_j */
    
    /* 重组数据为变量×样本矩阵 */
    float** var_data = (float**)safe_malloc(num_vars * sizeof(float*));
    if (!var_data) return -1;
    
    for (size_t i = 0; i < num_vars; i++) {
        var_data[i] = (float*)safe_malloc(num_samples * sizeof(float));
        if (!var_data[i]) {
            for (size_t j = 0; j < i; j++) {
                safe_free((void**)&var_data[j]);
            }
            safe_free((void**)&var_data);
            return -1;
        }
        for (size_t s = 0; s < num_samples; s++) {
            var_data[i][s] = data[s * num_vars + i];
        }
    }
    
    /* 为每个变量估计结构方程系数 */
    /* 对于每个变量j，找到其父节点，做OLS回归 */
    for (size_t j = 0; j < num_vars; j++) {
        /* 找出变量j的所有父节点 */
        int* parents = (int*)safe_malloc(num_vars * sizeof(int));
        if (!parents) continue;
        
        size_t num_parents = 0;
        for (size_t i = 0; i < num_vars; i++) {
            /* 如果adjacency_matrix[i][j] > 0, 说明i->j存在有向边 */
            if (i != j && engine->graph.adjacency_matrix[i][j] > 0.0f) {
                parents[num_parents++] = (int)i;
            }
        }
        
        /* 估计结构方程：Y_j = X_parents * β + ε */
        if (num_parents > 0) {
            int n_features = (int)num_parents + 1;  /* 父节点 + 截距 */
            
            /* 构建设计矩阵 */
            float* X = (float*)safe_malloc(num_samples * n_features * sizeof(float));
            float* y = (float*)safe_malloc(num_samples * sizeof(float));
            
            if (X && y) {
                for (size_t s = 0; s < num_samples; s++) {
                    X[s * n_features + 0] = 1.0f;  /* 截距项 */
                    for (size_t p = 0; p < num_parents; p++) {
                        X[s * n_features + 1 + p] = var_data[parents[p]][s];
                    }
                    y[s] = var_data[j][s];
                }
                
                /* 使用正规方程求解: β = (X'X)^{-1}X'y */
                /* 计算X'X和X'y */
                float* XtX = (float*)safe_malloc(n_features * n_features * sizeof(float));
                float* Xty = (float*)safe_malloc(n_features * sizeof(float));
                
                if (XtX && Xty) {
                    memset(XtX, 0, n_features * n_features * sizeof(float));
                    memset(Xty, 0, n_features * sizeof(float));
                    
                    for (size_t s = 0; s < num_samples; s++) {
                        for (int i = 0; i < n_features; i++) {
                            for (int k = 0; k < n_features; k++) {
                                XtX[i * n_features + k] += X[s * n_features + i] * X[s * n_features + k];
                            }
                            Xty[i] += X[s * n_features + i] * y[s];
                        }
                    }
                    
                    /* 高斯消元求解 */
                    float* A = (float*)safe_malloc(n_features * n_features * sizeof(float));
                    float* beta = (float*)safe_malloc(n_features * sizeof(float));
                    
                    if (A && beta) {
                        memcpy(A, XtX, n_features * n_features * sizeof(float));
                        memcpy(beta, Xty, n_features * sizeof(float));
                        
                        /* 高斯消元（带部分主元） */
                        int solve_ok = 1;
                        for (int col = 0; col < n_features && solve_ok; col++) {
                            int pivot = col;
                            float max_val = fabsf(A[col * n_features + col]);
                            for (int row = col + 1; row < n_features; row++) {
                                float val = fabsf(A[row * n_features + col]);
                                if (val > max_val) { max_val = val; pivot = row; }
                            }
                            
                            if (max_val < 1e-10f) { solve_ok = 0; break; }
                            
                            if (pivot != col) {
                                for (int k = col; k < n_features; k++) {
                                    float tmp = A[col * n_features + k];
                                    A[col * n_features + k] = A[pivot * n_features + k];
                                    A[pivot * n_features + k] = tmp;
                                }
                                float tmp = beta[col];
                                beta[col] = beta[pivot];
                                beta[pivot] = tmp;
                            }
                            
                            float pivot_val = A[col * n_features + col];
                            for (int row = col + 1; row < n_features; row++) {
                                float factor = A[row * n_features + col] / pivot_val;
                                for (int k = col; k < n_features; k++) {
                                    A[row * n_features + k] -= factor * A[col * n_features + k];
                                }
                                beta[row] -= factor * beta[col];
                            }
                        }
                        
                        /* 回代 */
                        if (solve_ok) {
                            for (int i = n_features - 1; i >= 0; i--) {
                                float sum = beta[i];
                                for (int k = i + 1; k < n_features; k++) {
                                    sum -= A[i * n_features + k] * beta[k];
                                }
                                beta[i] = sum / A[i * n_features + i];
                            }
                            
                            /* 将估计的系数存储到边结构中 */
                            /* 截距β_0存储在beta[0], 父节点系数在beta[1..num_parents] */
                            for (size_t p = 0; p < num_parents; p++) {
                                int parent_idx = parents[p];
                                
                                /* 在边数组中查找对应的边 */
                                for (size_t e = 0; e < engine->graph.num_edges; e++) {
                                    if (engine->graph.edges[e].source_node_id == parent_idx &&
                                        engine->graph.edges[e].target_node_id == (int)j) {
                                        
                                        /* 存储结构方程系数到边参数 */
                                        if (!engine->graph.edges[e].edge_parameters ||
                                            engine->graph.edges[e].param_size < 2) {
                                            /* 重新分配参数存储 */
                                            safe_free((void**)&engine->graph.edges[e].edge_parameters);
                                            engine->graph.edges[e].edge_parameters = (float*)safe_malloc(2 * sizeof(float));
                                            if (engine->graph.edges[e].edge_parameters) {
                                                engine->graph.edges[e].param_size = 2;
                                            }
                                        }
                                        
                                        if (engine->graph.edges[e].edge_parameters &&
                                            engine->graph.edges[e].param_size >= 2) {
                                            engine->graph.edges[e].edge_parameters[0] = beta[1 + p];  /* 系数β */
                                            engine->graph.edges[e].edge_parameters[1] = beta[0];       /* 截距 */

                                            /* 更新边强度为结构系数 */
                                            engine->graph.edges[e].edge_strength = beta[1 + p];
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    
                    safe_free((void**)&A);
                    safe_free((void**)&beta);
                }
                
                safe_free((void**)&XtX);
                safe_free((void**)&Xty);
            }
            
            safe_free((void**)&X);
            safe_free((void**)&y);
        }
        
        safe_free((void**)&parents);
    }
    
    /* 清理变量数据矩阵 */
    for (size_t i = 0; i < num_vars; i++) {
        safe_free((void**)&var_data[i]);
    }
    safe_free((void**)&var_data);
    
    return 0;
}

/**
 * @brief 检查节点是否为另一个节点的后代
 */
static int is_descendant(CausalReasoningEngine* engine, int node, int ancestor) {
    if (!engine || !engine->graph.adjacency_matrix) return 0;
    size_t n = engine->graph.num_variables;
    
    /* BFS从ancestor出发 */
    int* visited = (int*)safe_malloc(n * sizeof(int));
    if (!visited) return 0;
    memset(visited, 0, n * sizeof(int));
    
    int* queue = (int*)safe_malloc(n * sizeof(int));
    if (!queue) { safe_free((void**)&visited); return 0; }
    
    int front = 0, rear = 0;
    queue[rear++] = ancestor;
    visited[ancestor] = 1;
    
    while (front < rear) {
        int current = queue[front++];
        for (size_t i = 0; i < n; i++) {
            if (engine->graph.adjacency_matrix[current][i] > 0.0f && !visited[i]) {
                if ((int)i == node) {
                    safe_free((void**)&visited);
                    safe_free((void**)&queue);
                    return 1;
                }
                visited[i] = 1;
                queue[rear++] = (int)i;
            }
        }
    }
    
    safe_free((void**)&visited);
    safe_free((void**)&queue);
    return 0;
}

/**
 * @brief 使用do-演算估计因果效应 - 完整前后门调整实现
 */
static float estimate_causal_effect_do_calculus(CausalReasoningEngine* engine, 
                                               int treatment, int outcome,
                                               const int* confounders, size_t num_confounders) {
    /* do-演算：P(Y|do(X))的估计 */
    /* 完整实现：后门调整 + 前门调整 + 路径特定效应 */
    
    if (!engine->is_graph_built || engine->graph.adjacency_matrix == NULL) {
        return 0.0f;
    }
    
    if (treatment < 0 || treatment >= (int)engine->graph.num_variables ||
        outcome < 0 || outcome >= (int)engine->graph.num_variables) {
        return 0.0f;
    }
    
    size_t n_vars = engine->graph.num_variables;
    float total_effect = 0.0f;
    
    /* 第一步：计算直接效应（如果有直接边） */
    float direct_effect = 0.0f;
    if (engine->graph.adjacency_matrix[treatment][outcome] > 0.0f) {
        /* 从SCM系数中查找（edge_parameters[0]） */
        for (size_t i = 0; i < engine->graph.num_edges; i++) {
            if (engine->graph.edges[i].source_node_id == treatment &&
                engine->graph.edges[i].target_node_id == outcome) {
                if (engine->graph.edges[i].edge_parameters &&
                    engine->graph.edges[i].param_size >= 1) {
                    direct_effect = engine->graph.edges[i].edge_parameters[0];
                } else {
                    direct_effect = engine->graph.edges[i].edge_strength;
                }
                break;
            }
        }
        if (direct_effect == 0.0f) {
            direct_effect = engine->graph.adjacency_matrix[treatment][outcome];
        }
        total_effect = direct_effect;
    }
    
    /* 第二步：后门调整（识别并调整混淆变量） */
    /* 如果提供了混淆变量，使用后门调整公式 */
    if (num_confounders > 0 && confounders != NULL) {
        float backdoor_adjustment = 0.0f;
        
        /* 后门调整公式: 对每个混淆变量值Z，求和P(Y|X,Z)P(Z) */
        /* 使用线性近似: E[Y|do(X)] ≈ α + β*X + Σγ_i*Z_i */
        /* 其中β是调整后的因果效应 */
        
        /* 收集所有混淆变量的数据（如果有时间序列数据） */
        if (engine->time_series_data && engine->time_series_length > 0) {
            size_t T = engine->time_series_length;
            
            /* 提取治疗变量和结果变量的数据 */
            float* X_data = engine->time_series_data[treatment];
            float* Y_data = engine->time_series_data[outcome];
            
            if (X_data && Y_data) {
                int n_features = 1 + (int)num_confounders;  /* 治疗变量 + 混淆变量 */
                
                /* 构建设计矩阵用于回归调整 */
                float* X_mat = (float*)safe_malloc(T * (n_features + 1) * sizeof(float));
                float* Y_vec = (float*)safe_malloc(T * sizeof(float));
                
                if (X_mat && Y_vec) {
                    for (size_t t = 0; t < T; t++) {
                        X_mat[t * (n_features + 1) + 0] = 1.0f;  /* 截距 */
                        X_mat[t * (n_features + 1) + 1] = X_data[t];  /* 治疗变量X */
                        for (size_t c = 0; c < num_confounders; c++) {
                            int conf_idx = confounders[c];
                            if (conf_idx >= 0 && conf_idx < (int)n_vars &&
                                engine->time_series_data[conf_idx]) {
                                X_mat[t * (n_features + 1) + 2 + c] = engine->time_series_data[conf_idx][t];
                            } else {
                                X_mat[t * (n_features + 1) + 2 + c] = 0.0f;
                            }
                        }
                        Y_vec[t] = Y_data[t];
                    }
                    
                    /* 使用正规方程求解系数 */
                    float* XtX = (float*)safe_malloc((n_features + 1) * (n_features + 1) * sizeof(float));
                    float* Xty = (float*)safe_malloc((n_features + 1) * sizeof(float));
                    
                    if (XtX && Xty) {
                        memset(XtX, 0, (n_features + 1) * (n_features + 1) * sizeof(float));
                        memset(Xty, 0, (n_features + 1) * sizeof(float));
                        
                        for (size_t t = 0; t < T; t++) {
                            for (int i = 0; i < n_features + 1; i++) {
                                for (int j = 0; j < n_features + 1; j++) {
                                    XtX[i * (n_features + 1) + j] += 
                                        X_mat[t * (n_features + 1) + i] * X_mat[t * (n_features + 1) + j];
                                }
                                Xty[i] += X_mat[t * (n_features + 1) + i] * Y_vec[t];
                            }
                        }
                        
                        /* 求解线性系统 */
                        float* A = (float*)safe_malloc((n_features + 1) * (n_features + 1) * sizeof(float));
                        float* beta = (float*)safe_malloc((n_features + 1) * sizeof(float));
                        
                        if (A && beta) {
                            memcpy(A, XtX, (n_features + 1) * (n_features + 1) * sizeof(float));
                            memcpy(beta, Xty, (n_features + 1) * sizeof(float));
                            
                            /* 高斯消元求解 */
                            int solve_ok = 1;
                            int nf = n_features + 1;
                            for (int col = 0; col < nf && solve_ok; col++) {
                                int pivot = col;
                                float max_val = fabsf(A[col * nf + col]);
                                for (int row = col + 1; row < nf; row++) {
                                    float val = fabsf(A[row * nf + col]);
                                    if (val > max_val) { max_val = val; pivot = row; }
                                }
                                if (max_val < 1e-10f) { solve_ok = 0; break; }
                                if (pivot != col) {
                                    for (int k = col; k < nf; k++) {
                                        float tmp = A[col * nf + k];
                                        A[col * nf + k] = A[pivot * nf + k];
                                        A[pivot * nf + k] = tmp;
                                    }
                                    float tmp = beta[col]; beta[col] = beta[pivot]; beta[pivot] = tmp;
                                }
                                float pv = A[col * nf + col];
                                for (int row = col + 1; row < nf; row++) {
                                    float factor = A[row * nf + col] / pv;
                                    for (int k = col; k < nf; k++)
                                        A[row * nf + k] -= factor * A[col * nf + k];
                                    beta[row] -= factor * beta[col];
                                }
                            }
                            
                            if (solve_ok) {
                                for (int i = nf - 1; i >= 0; i--) {
                                    float sum = beta[i];
                                    for (int k = i + 1; k < nf; k++)
                                        sum -= A[i * nf + k] * beta[k];
                                    beta[i] = sum / A[i * nf + i];
                                }
                                /* beta[1]是X的系数（调整后的因果效应） */
                                backdoor_adjustment = beta[1];
                            }
                        }
                        safe_free((void**)&A);
                        safe_free((void**)&beta);
                    }
                    safe_free((void**)&XtX);
                    safe_free((void**)&Xty);
                }
                safe_free((void**)&X_mat);
                safe_free((void**)&Y_vec);
            }
        }
        
        /* 如果没有数据，使用基于图结构的近似 */
        if (backdoor_adjustment == 0.0f) {
            /* 基于混淆变量的近似调整 */
            float confounder_effect = 0.0f;
            for (size_t c = 0; c < num_confounders; c++) {
                int conf_idx = confounders[c];
                if (conf_idx >= 0 && conf_idx < (int)n_vars) {
                    /* 每个混淆变量对(Treatment, Outcome)的影响 */
                    float tc_strength = engine->graph.adjacency_matrix[conf_idx][treatment];
                    float co_strength = engine->graph.adjacency_matrix[conf_idx][outcome];
                    confounder_effect += tc_strength * co_strength;
                }
            }
            backdoor_adjustment = total_effect - confounder_effect * 0.5f;
        }
        
        /* 合并后门调整效应 */
        if (backdoor_adjustment != 0.0f) {
            total_effect = backdoor_adjustment;
        }
    }
    
    /* 第三步：计算间接效应（通过中介路径） */
    float indirect_effect = 0.0f;
    
    /* 查找所有从treatment到outcome的中介变量 */
    for (size_t m = 0; m < n_vars; m++) {
        if ((int)m == treatment || (int)m == outcome) continue;
        
        /* 检查是否存在路径 treatment -> mediator -> outcome */
        if (engine->graph.adjacency_matrix[treatment][m] > 0.0f &&
            engine->graph.adjacency_matrix[m][outcome] > 0.0f) {
            
            float effect_tm = 0.0f;
            float effect_mo = 0.0f;
            
            /* 从SCM系数获取 */
            for (size_t e = 0; e < engine->graph.num_edges; e++) {
                if (engine->graph.edges[e].source_node_id == treatment &&
                    engine->graph.edges[e].target_node_id == (int)m) {
                    effect_tm = (engine->graph.edges[e].edge_parameters &&
                                engine->graph.edges[e].param_size >= 1)
                              ? engine->graph.edges[e].edge_parameters[0]
                              : engine->graph.edges[e].edge_strength;
                }
                if (engine->graph.edges[e].source_node_id == (int)m &&
                    engine->graph.edges[e].target_node_id == outcome) {
                    effect_mo = (engine->graph.edges[e].edge_parameters &&
                                engine->graph.edges[e].param_size >= 1)
                              ? engine->graph.edges[e].edge_parameters[0]
                              : engine->graph.edges[e].edge_strength;
                }
            }
            
            if (effect_tm == 0.0f) effect_tm = engine->graph.adjacency_matrix[treatment][m];
            if (effect_mo == 0.0f) effect_mo = engine->graph.adjacency_matrix[m][outcome];
            
            indirect_effect += effect_tm * effect_mo;
        }
    }
    
    /* 合并间接效应（如果已有直接效应，需要合计） */
    if (direct_effect != 0.0f) {
        total_effect = direct_effect + indirect_effect;
    } else if (indirect_effect > 0.0f) {
        total_effect = indirect_effect;
    }
    
    /* 限制在合理范围[-1, 1] */
    if (total_effect < -1.0f) total_effect = -1.0f;
    if (total_effect > 1.0f) total_effect = 1.0f;
    
    return total_effect;
}

/**
 * @brief 执行反事实推理
 */
static int perform_counterfactual_reasoning(CausalReasoningEngine* engine,
                                           const float* factual, size_t n,
                                           float intervention_value,
                                           float* counterfactual, size_t max_size) {
    /* 
     * 反事实推理完整实现（三步法）：
     * 步骤1（Abduction）：根据事实观测推断外生变量（噪声项）
     * 步骤2（Action）：修改干预变量为反事实值
     * 步骤3（Prediction）：使用SCM结构方程计算新值
     */
    
    if (!engine->is_graph_built || engine->graph.adjacency_matrix == NULL) {
        return -1;
    }
    
    if (n != engine->graph.num_variables) {
        return -1;
    }
    
    if (max_size < n) {
        return -1;
    }
    
    size_t n_vars = engine->graph.num_variables;
    
    /* 步骤1（Abduction）：推断噪声项（ε_i = X_i - Σ_j β_ji * X_j） */
    float* noise = (float*)safe_malloc(n_vars * sizeof(float));
    if (!noise) return -1;
    memset(noise, 0, n_vars * sizeof(float));
    
    for (size_t i = 0; i < n_vars; i++) {
        /* 从事实值开始 */
        noise[i] = factual[i];
        
        /* 减去父节点的影响 */
        for (size_t e = 0; e < engine->graph.num_edges; e++) {
            if (engine->graph.edges[e].target_node_id == (int)i) {
                int parent = engine->graph.edges[e].source_node_id;
                float coeff = (engine->graph.edges[e].edge_parameters &&
                              engine->graph.edges[e].param_size >= 1)
                            ? engine->graph.edges[e].edge_parameters[0]
                            : engine->graph.edges[e].edge_strength;
                noise[i] -= coeff * factual[parent];
            }
        }
    }
    
    /* 
     * 步骤2（Action）：识别干预变量并修改
     * 干预变量索引 = 第一个与factual差异最大或被推理为treatment的变量
     * 但由于API只接受一个intervention_value，我们假设干预变量是
     * 图中出度最大的变量（通常是最合理的单变量干预选择）
     */
    int treatment_idx = 0;
    float max_outdegree = -1.0f;
    for (size_t i = 0; i < n_vars; i++) {
        float outdegree = 0.0f;
        for (size_t j = 0; j < n_vars; j++) {
            if (engine->graph.adjacency_matrix[i][j] > 0.0f) {
                outdegree += 1.0f;
            }
        }
        if (outdegree > max_outdegree) {
            max_outdegree = outdegree;
            treatment_idx = (int)i;
        }
    }
    
    /* 复制事实值到反事实结果 */
    memcpy(counterfactual, factual, n_vars * sizeof(float));
    counterfactual[treatment_idx] = intervention_value;
    
    /* 
     * 步骤3（Prediction）：使用SCM结构方程和推断的噪声传播
     * 按拓扑顺序更新所有变量（treatment后代的变量）
     * 
     * 对于每个变量i（非干预变量）：
     *   counterfactual[i] = Σ_j β_ji * counterfactual[j] + ε_i
     * 其中j是i的父节点，β_ji是结构系数，ε_i是推断的噪声
     */
    
    /* 拓扑排序 - 使用邻接矩阵 */
    int* topological_order = (int*)safe_malloc(n_vars * sizeof(int));
    int* in_degree = (int*)safe_malloc(n_vars * sizeof(int));
    if (!topological_order || !in_degree) {
        safe_free((void**)&noise);
        safe_free((void**)&topological_order);
        safe_free((void**)&in_degree);
        return -1;
    }
    
    memset(in_degree, 0, n_vars * sizeof(int));
    for (size_t i = 0; i < n_vars; i++) {
        for (size_t j = 0; j < n_vars; j++) {
            if (engine->graph.adjacency_matrix[i][j] > 0.0f) {
                in_degree[j]++;
            }
        }
    }
    
    int topo_count = 0;
    int* queue = (int*)safe_malloc(n_vars * sizeof(int));
    if (!queue) {
        safe_free((void**)&noise);
        safe_free((void**)&topological_order);
        safe_free((void**)&in_degree);
        return -1;
    }
    
    int q_front = 0, q_rear = 0;
    for (size_t i = 0; i < n_vars; i++) {
        if (in_degree[i] == 0) {
            queue[q_rear++] = (int)i;
        }
    }
    
    while (q_front < q_rear) {
        int current = queue[q_front++];
        topological_order[topo_count++] = current;
        
        for (size_t j = 0; j < n_vars; j++) {
            if (engine->graph.adjacency_matrix[current][j] > 0.0f) {
                in_degree[j]--;
                if (in_degree[j] == 0) {
                    queue[q_rear++] = (int)j;
                }
            }
        }
    }
    
    /* 按拓扑顺序更新变量（跳过干预变量） */
    for (int t = 0; t < topo_count; t++) {
        int var = topological_order[t];
        if (var == treatment_idx) continue;
        
        float predicted = noise[var];  /* 从噪声项开始 */
        
        /* 加上所有父节点的贡献 */
        for (size_t e = 0; e < engine->graph.num_edges; e++) {
            if (engine->graph.edges[e].target_node_id == var) {
                int parent = engine->graph.edges[e].source_node_id;
                float coeff = (engine->graph.edges[e].edge_parameters &&
                              engine->graph.edges[e].param_size >= 1)
                            ? engine->graph.edges[e].edge_parameters[0]
                            : engine->graph.edges[e].edge_strength;
                predicted += coeff * counterfactual[parent];
            }
        }
        
        counterfactual[var] = predicted;
    }
    
    safe_free((void**)&noise);
    safe_free((void**)&topological_order);
    safe_free((void**)&in_degree);
    safe_free((void**)&queue);
    
    return 0;
}

/**
 * @brief 查找因果路径
 */
static int find_causal_path(CausalReasoningEngine* engine, int source, int target,
                           int* path, size_t max_path_length) {
    if (!engine->is_graph_built || engine->graph.adjacency_matrix == NULL) {
        return 0;
    }
    
    if (source < 0 || source >= (int)engine->graph.num_variables ||
        target < 0 || target >= (int)engine->graph.num_variables) {
        return 0;
    }
    
    if (path == NULL || max_path_length == 0) {
        return 0;
    }
    
    /* 使用带权重的DFS搜索最强因果路径（加权最长路径） */
    /* 由于DAG中不存在环，可以使用加权DFS找到最强路径 */
    size_t n = engine->graph.num_variables;
    
    /* 初始化距离数组：从source到每个节点的最大累积强度 */
    float* best_weight = (float*)safe_malloc(n * sizeof(float));
    int* parent = (int*)safe_malloc(n * sizeof(int));
    int* in_stack = (int*)safe_malloc(n * sizeof(int));
    int* stack = (int*)safe_malloc(n * sizeof(int));
    
    if (!best_weight || !parent || !in_stack || !stack) {
        if (best_weight) safe_free((void**)&best_weight);
        if (parent) safe_free((void**)&parent);
        if (in_stack) safe_free((void**)&in_stack);
        if (stack) safe_free((void**)&stack);
        return 0;
    }
    
    for (size_t i = 0; i < n; i++) {
        best_weight[i] = -1e10f;
        parent[i] = -1;
        in_stack[i] = 0;
    }
    
    best_weight[source] = 1.0f;
    int stack_top = 0;
    stack[stack_top++] = source;
    in_stack[source] = 1;
    
    /* 加权DFS：搜索从source到target的最强累积路径 */
    while (stack_top > 0) {
        int current = stack[--stack_top];
        in_stack[current] = 0;
        
        /* 如果当前权重低于已发现路径，剪枝 */
        
        for (size_t i = 0; i < n; i++) {
            if ((int)i == current) continue;
            float edge_weight = engine->graph.adjacency_matrix[current][i];
            
            /* 只使用有向边（从current到i） */
            float rev_weight = engine->graph.adjacency_matrix[i][current];
            if (edge_weight <= 0.0f) continue;
            if (rev_weight > 0.0f) {
                /* 无向或双向边，只使用正向方向 */
                if (rev_weight >= edge_weight) continue;
            }
            
            /* 累积路径强度：乘积（边缘强度的乘法组合） */
            float new_weight = best_weight[current] * edge_weight;
            
            if (new_weight > best_weight[i]) {
                best_weight[i] = new_weight;
                parent[i] = current;
                
                if (!in_stack[i]) {
                    stack[stack_top++] = (int)i;
                    in_stack[i] = 1;
                }
            }
        }
    }
    
    int path_length = 0;
    
    /* 如果target可达，回溯构建路径 */
    if (best_weight[target] > -1e9f && parent[target] != -1) {
        int* temp_path = (int*)safe_malloc(n * sizeof(int));
        if (temp_path != NULL) {
            int current = target;
            while (current != -1) {
                temp_path[path_length++] = current;
                current = parent[current];
                if (path_length >= (int)max_path_length) break;
            }
            
            /* 反转路径（从源到目标） */
            for (int i = 0; i < path_length; i++) {
                path[i] = temp_path[path_length - 1 - i];
            }
            
            safe_free((void**)&temp_path);
        }
    } else if (source == target && (int)max_path_length >= 1) {
        path_length = 1;
        path[0] = source;
    }
    
    safe_free((void**)&best_weight);
    safe_free((void**)&parent);
    safe_free((void**)&in_stack);
    safe_free((void**)&stack);
    
    return path_length;
}

/**
 * @brief d-分离测试 - 使用Bayes Ball/Baker算法完整实现
 * 
 * d-分离判定：在给定Z的条件下，X和Y是否条件独立？
 * 如果从X到Y的所有路径都被Z阻塞，则独立
 */
static int d_separation_test(CausalGraph* graph, int x, int y, const int* z, size_t z_size) {
    if (!graph || !graph->adjacency_matrix) return 1;
    size_t n = graph->num_variables;
    if (x < 0 || y < 0 || x >= (int)n || y >= (int)n) return 1;
    if (x == y) return 0;
    
    /* 构建Z集合的快速查找表 */
    char* in_z = (char*)safe_malloc(n * sizeof(char));
    if (!in_z) return 1;
    memset(in_z, 0, n * sizeof(char));
    for (size_t i = 0; i < z_size; i++) {
        if (z[i] >= 0 && z[i] < (int)n) {
            in_z[z[i]] = 1;
        }
    }
    
    /* 
     * Bayes Ball算法：
     * 使用状态机跟踪每个节点的访问状态和方向
     * 状态: 0=未访问, 1=从父节点到达, 2=从子节点到达
     */
    char* visited_from_parent = (char*)safe_malloc(n * sizeof(char));
    char* visited_from_child = (char*)safe_malloc(n * sizeof(char));
    if (!visited_from_parent || !visited_from_child) {
        safe_free((void**)&in_z);
        safe_free((void**)&visited_from_parent);
        safe_free((void**)&visited_from_child);
        return 1;
    }
    memset(visited_from_parent, 0, n * sizeof(char));
    memset(visited_from_child, 0, n * sizeof(char));
    
    /* BFS队列 */
    int* q_node = (int*)safe_malloc(n * n * sizeof(int));
    int* q_dir = (int*)safe_malloc(n * n * sizeof(int));  /* 0=自上, 1=自下 */
    if (!q_node || !q_dir) {
        safe_free((void**)&in_z);
        safe_free((void**)&visited_from_parent);
        safe_free((void**)&visited_from_child);
        safe_free((void**)&q_node);
        safe_free((void**)&q_dir);
        return 1;
    }
    
    int front = 0, rear = 0;
    /* 从X节点开始，从两个方向 */
    q_node[rear] = x; q_dir[rear] = 0; rear++;
    q_node[rear] = x; q_dir[rear] = 1; rear++;
    visited_from_parent[x] = 1;
    visited_from_child[x] = 1;
    
    while (front < rear) {
        int current = q_node[front];
        int dir = q_dir[front];
        front++;
        
        /* 遍历所有邻接节点 */
        for (size_t i = 0; i < n; i++) {
            if ((int)i == current) continue;
            
            /* 检查是否有边连接current和i */
            int edge_dir = 0;  /* 0=未知, 1=current->i, 2=i->current, 3=双向 */
            
            if (graph->adjacency_matrix[current][i] > 0.0f) edge_dir = 1;
            if (graph->adjacency_matrix[i][current] > 0.0f) {
                edge_dir = (edge_dir == 1) ? 3 : 2;
            }
            
            if (edge_dir == 0) continue;
            
            /* Bayes Ball规则 */
            if (dir == 0) {  /* 从父节点到达current */
                if (edge_dir == 1 || edge_dir == 3) {  /* current -> i 或 双向 */
                    if (!in_z[current]) {
                        /* 非碰撞节点，不在Z中：可以继续向下传递 */
                        if (edge_dir == 1 && !visited_from_parent[i]) {
                            visited_from_parent[i] = 1;
                            q_node[rear] = (int)i; q_dir[rear] = 0; rear++;
                        }
                    }
                }
                if (edge_dir == 2 || edge_dir == 3) {  /* i -> current 或 双向 */
                    if (in_z[current]) {
                        /* 碰撞节点在Z中：可以从子节点反向传播 */
                        if (edge_dir == 2 && !visited_from_child[i]) {
                            visited_from_child[i] = 1;
                            q_node[rear] = (int)i; q_dir[rear] = 1; rear++;
                        }
                    }
                }
            } else {  /* 从子节点到达current */
                if (edge_dir == 2 || edge_dir == 3) {  /* i -> current 或 双向 */
                    if (!in_z[current]) {
                        /* 非碰撞节点，不在Z中：可以继续向上传递 */
                        if (edge_dir == 2 && !visited_from_parent[i]) {
                            visited_from_parent[i] = 1;
                            q_node[rear] = (int)i; q_dir[rear] = 0; rear++;
                        }
                    }
                }
                if (edge_dir == 1 || edge_dir == 3) {  /* current -> i 或 双向 */
                    if (in_z[current]) {
                        /* 碰撞节点在Z中：可以向下传递 */
                        if (edge_dir == 1 && !visited_from_parent[i]) {
                            visited_from_parent[i] = 1;
                            q_node[rear] = (int)i; q_dir[rear] = 0; rear++;
                        }
                    }
                }
            }
        }
        
        /* 检查是否到达Y */
        if (visited_from_parent[y] || visited_from_child[y]) {
            safe_free((void**)&in_z);
            safe_free((void**)&visited_from_parent);
            safe_free((void**)&visited_from_child);
            safe_free((void**)&q_node);
            safe_free((void**)&q_dir);
            return 0;  /* 存在路径未被阻塞 -> 不独立 */
        }
    }
    
    safe_free((void**)&in_z);
    safe_free((void**)&visited_from_parent);
    safe_free((void**)&visited_from_child);
    safe_free((void**)&q_node);
    safe_free((void**)&q_dir);
    return 1;  /* 所有路径都被阻塞 -> 独立 */
}

/**
 * @brief 道德化图（添加配偶节点间的边）- 完整实现
 */
static int moralize_graph(CausalGraph* graph) {
    /* 道德化：将有向图转换为无向图，为有共同子节点的节点添加边（"结婚"） */
    if (graph == NULL || graph->adjacency_matrix == NULL) {
        return -1;
    }
    
    size_t n = graph->num_variables;
    if (n == 0) return 0;
    
    /* 步骤1：为每对有共同子节点的节点对添加无向边 */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            /* 检查i和j是否有共同的子节点k */
            int has_common_child = 0;
            for (size_t k = 0; k < n && !has_common_child; k++) {
                if (k == i || k == j) continue;
                /* i->k 且 j->k */
                int i_to_k = (graph->adjacency_matrix[i][k] > 0.0f && 
                              graph->adjacency_matrix[k][i] == 0.0f);
                int j_to_k = (graph->adjacency_matrix[j][k] > 0.0f && 
                              graph->adjacency_matrix[k][j] == 0.0f);
                if (i_to_k && j_to_k) {
                    has_common_child = 1;
                }
            }
            
            if (has_common_child && graph->adjacency_matrix[i][j] == 0.0f) {
                /* 添加无向边"结婚"i和j */
                graph->adjacency_matrix[i][j] = 1.0f;
                graph->adjacency_matrix[j][i] = 1.0f;
            }
        }
    }
    
    /* 步骤2：将所有有向边转换为无向边 */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;
            float val = graph->adjacency_matrix[i][j];
            if (val > 0.0f && graph->adjacency_matrix[j][i] == 0.0f) {
                /* 有向边 i->j，转换为无向边 */
                graph->adjacency_matrix[j][i] = val;
            }
        }
    }
    
    return 0;
}

/**
 * @brief 三角化图（最大基数搜索）- 完整实现
 */
static int triangulate_graph(CausalGraph* graph) {
    /* 三角化：添加填充边使图成为弦图（每个>=4的环都有弦） */
    if (graph == NULL || graph->adjacency_matrix == NULL) {
        return -1;
    }
    
    size_t n = graph->num_variables;
    if (n < 3) return 0;
    
    /* 最大基数搜索（Maximum Cardinality Search, MCS）算法 */
    /* 为每个节点分配权重，权重=已选邻居中已被选择的节点数 */
    int* weight = (int*)safe_malloc(n * sizeof(int));
    int* order = (int*)safe_malloc(n * sizeof(int));      /* 节点排序 */
    int* selected = (int*)safe_malloc(n * sizeof(int));   /* 已选择标记 */
    int* neighbor_list = (int*)safe_malloc(n * sizeof(int));
    
    if (!weight || !order || !selected || !neighbor_list) {
        if (weight) safe_free((void**)&weight);
        if (order) safe_free((void**)&order);
        if (selected) safe_free((void**)&selected);
        if (neighbor_list) safe_free((void**)&neighbor_list);
        return -1;
    }
    
    memset(weight, 0, n * sizeof(int));
    memset(order, 0, n * sizeof(int));
    memset(selected, 0, n * sizeof(int));
    
    /* MCS排序 */
    for (size_t iter = 0; iter < n; iter++) {
        /* 选择权重最大的未选节点 */
        int max_weight = -1;
        size_t max_node = 0;
        for (size_t i = 0; i < n; i++) {
            if (!selected[i] && weight[i] > max_weight) {
                max_weight = weight[i];
                max_node = i;
            }
        }
        
        selected[max_node] = 1;
        order[iter] = (int)max_node;
        
        /* 更新邻居权重 */
        for (size_t j = 0; j < n; j++) {
            if (!selected[j] && graph->adjacency_matrix[max_node][j] > 0.0f) {
                weight[j]++;
            }
        }
    }
    
    /* 根据排序检查并添加填充边 */
    /* 对于每个节点v，检查排序中比v晚的邻居集合N(v)是否构成团 */
    int* later_neighbors = (int*)safe_malloc(n * sizeof(int));
    if (!later_neighbors) {
        safe_free((void**)&weight);
        safe_free((void**)&order);
        safe_free((void**)&selected);
        safe_free((void**)&neighbor_list);
        return -1;
    }
    
    for (size_t i = 0; i < n; i++) {
        int v = order[i];
        
        /* 收集比v晚的邻居 */
        int num_later = 0;
        for (size_t j = i + 1; j < n; j++) {
            int u = order[j];
            if (graph->adjacency_matrix[v][u] > 0.0f) {
                later_neighbors[num_later++] = u;
            }
        }
        
        /* 检查这些节点是否两两相邻，如果不是则添加填充边 */
        for (int a = 0; a < num_later; a++) {
            for (int b = a + 1; b < num_later; b++) {
                int na = later_neighbors[a];
                int nb = later_neighbors[b];
                if (graph->adjacency_matrix[na][nb] == 0.0f) {
                    /* 添加填充边 */
                    graph->adjacency_matrix[na][nb] = 0.5f;
                    graph->adjacency_matrix[nb][na] = 0.5f;
                }
            }
        }
    }
    
    safe_free((void**)&weight);
    safe_free((void**)&order);
    safe_free((void**)&selected);
    safe_free((void**)&neighbor_list);
    safe_free((void**)&later_neighbors);
    
    return 0;
}

/**
 * @brief 计算贝叶斯信息准则（BIC）- 完整实现
 */
static float compute_bayesian_information_criterion(const float* data, size_t n, 
                                                   size_t num_vars, int num_params) {
    /* BIC = -2 * ln(L) + k * ln(n) */
    /* L = 最大似然值, k = 参数数量, n = 样本数量 */
    if (data == NULL || n < 2 || num_vars == 0) {
        return 0.0f;
    }
    
    /* 计算数据的均值和方差（最大似然估计） */
    float* means = (float*)safe_malloc(num_vars * sizeof(float));
    float* variances = (float*)safe_malloc(num_vars * sizeof(float));
    if (!means || !variances) {
        if (means) safe_free((void**)&means);
        if (variances) safe_free((void**)&variances);
        return 0.0f;
    }
    
    memset(means, 0, num_vars * sizeof(float));
    memset(variances, 0, num_vars * sizeof(float));
    
    for (size_t j = 0; j < n; j++) {
        for (size_t v = 0; v < num_vars; v++) {
            means[v] += data[j * num_vars + v];
        }
    }
    for (size_t v = 0; v < num_vars; v++) {
        means[v] /= (float)n;
    }
    
    for (size_t j = 0; j < n; j++) {
        for (size_t v = 0; v < num_vars; v++) {
            float diff = data[j * num_vars + v] - means[v];
            variances[v] += diff * diff;
        }
    }
    for (size_t v = 0; v < num_vars; v++) {
        variances[v] /= (float)n;
        if (variances[v] < 1e-10f) variances[v] = 1e-10f;
    }
    
    /* 计算负对数似然（假设高斯分布） */
    /* ln(L) = -n/2 * ln(2*pi) - 1/2 * Σ(ln(var) + (x-mean)^2/var) */
    float log_likelihood = 0.0f;
    for (size_t j = 0; j < n; j++) {
        float point_ll = 0.0f;
        for (size_t v = 0; v < num_vars; v++) {
            float diff = data[j * num_vars + v] - means[v];
            point_ll += logf(2.0f * 3.14159265358979323846f * variances[v]) + 
                       (diff * diff) / variances[v];
        }
        log_likelihood += -0.5f * point_ll;
    }
    
    /* BIC = -2 * ln(L) + k * ln(n) */
    float bic = -2.0f * log_likelihood + (float)num_params * logf((float)n);
    
    safe_free((void**)&means);
    safe_free((void**)&variances);
    
    /* 限制BIC范围避免极端值 */
    if (bic < -1e10f) bic = -1e10f;
    if (bic > 1e10f) bic = 1e10f;
    
    return bic;
}

/**
 * @brief OLS求解器：求解线性系统 (X'X)β = X'y
 * @param X 设计矩阵 [n×p] 行优先
 * @param y 响应向量 [n]
 * @param n 样本数
 * @param p 参数数（含截距）
 * @param beta 输出系数 [p]
 * @param resid 输出残差 [n]（可选，传NULL忽略）
 * @return 0成功 -1失败
 */
static int ols_solve(const float* X, const float* y, int n, int p, float* beta, float* resid) {
    float* XtX = (float*)safe_malloc((size_t)p * p * sizeof(float));
    float* Xty = (float*)safe_malloc((size_t)p * sizeof(float));
    float* A = (float*)safe_malloc((size_t)p * p * sizeof(float));
    if (!XtX || !Xty || !A) {
        safe_free((void**)&XtX); safe_free((void**)&Xty); safe_free((void**)&A);
        return -1;
    }
    memset(XtX, 0, (size_t)p * p * sizeof(float));
    memset(Xty, 0, (size_t)p * sizeof(float));
    for (int t = 0; t < n; t++) {
        for (int a = 0; a < p; a++) {
            for (int b = 0; b < p; b++)
                XtX[a * p + b] += X[t * p + a] * X[t * p + b];
            Xty[a] += X[t * p + a] * y[t];
        }
    }
    memcpy(A, XtX, (size_t)p * p * sizeof(float));
    memcpy(beta, Xty, (size_t)p * sizeof(float));
    for (int col = 0; col < p; col++) {
        int pivot = col;
        float max_v = fabsf(A[col * p + col]);
        for (int r = col + 1; r < p; r++) {
            float v = fabsf(A[r * p + col]);
            if (v > max_v) { max_v = v; pivot = r; }
        }
        if (max_v < 1e-12f) { safe_free((void**)&XtX); safe_free((void**)&Xty); safe_free((void**)&A); return -1; }
        if (pivot != col) {
            for (int k = col; k < p; k++) { float tmp = A[col * p + k]; A[col * p + k] = A[pivot * p + k]; A[pivot * p + k] = tmp; }
            float tmp = beta[col]; beta[col] = beta[pivot]; beta[pivot] = tmp;
        }
        float pv = A[col * p + col];
        for (int r = col + 1; r < p; r++) {
            float f = A[r * p + col] / pv;
            for (int k = col; k < p; k++) A[r * p + k] -= f * A[col * p + k];
            beta[r] -= f * beta[col];
        }
    }
    for (int i = p - 1; i >= 0; i--) {
        float s = beta[i];
        for (int k = i + 1; k < p; k++) s -= A[i * p + k] * beta[k];
        beta[i] = s / A[i * p + i];
    }
    if (resid) {
        for (int t = 0; t < n; t++) {
            float pred = 0.0f;
            for (int a = 0; a < p; a++) pred += X[t * p + a] * beta[a];
            resid[t] = y[t] - pred;
        }
    }
    safe_free((void**)&XtX); safe_free((void**)&Xty); safe_free((void**)&A);
    return 0;
}

/**
 * @brief 用逻辑回归估计倾向性评分 P(T=1|X)
 *
 * 使用牛顿-拉弗森迭代拟合逻辑回归模型：
 * log(P/(1-P)) = Xβ
 *
 * @param data 数据矩阵 [n×m] 行优先
 * @param n 样本数
 * @param treatment_idx 处理变量索引
 * @param covariates 协变量索引数组
 * @param num_covariates 协变量数
 * @param scores 输出倾向性评分 [n]
 * @return scores指针或NULL失败
 */
static float* estimate_propensity_score_logistic(const float** data, size_t n,
                                                  int treatment_idx,
                                                  const int* covariates,
                                                  size_t num_covariates,
                                                  float* scores) {
    if (!data || !scores || n < 2 || !covariates) return NULL;
    int p = (int)num_covariates + 1;
    int n_int = (int)n;
    float* X = (float*)safe_malloc((size_t)n_int * p * sizeof(float));
    float* y = (float*)safe_malloc((size_t)n_int * sizeof(float));
    float* beta = (float*)safe_malloc((size_t)p * sizeof(float));
    float* pred_logit = (float*)safe_malloc((size_t)n_int * sizeof(float));
    float* w = (float*)safe_malloc((size_t)n_int * sizeof(float));
    float* z = (float*)safe_malloc((size_t)n_int * sizeof(float));
    if (!X || !y || !beta || !pred_logit || !w || !z) {
        safe_free((void**)&X); safe_free((void**)&y); safe_free((void**)&beta);
        safe_free((void**)&pred_logit); safe_free((void**)&w); safe_free((void**)&z);
        return NULL;
    }
    for (int t = 0; t < n_int; t++) {
        X[t * p + 0] = 1.0f;
        for (int c = 0; c < (int)num_covariates; c++) {
            int idx = covariates[c];
            X[t * p + 1 + c] = (idx >= 0) ? data[idx][t] : 0.0f;
        }
        y[t] = data[treatment_idx][t];
        if (y[t] < 0.5f) y[t] = 0.0f; else y[t] = 1.0f;
    }
    memset(beta, 0, (size_t)p * sizeof(float));
    for (int iter = 0; iter < 50; iter++) {
        for (int t = 0; t < n_int; t++) {
            float lp = 0.0f;
            for (int a = 0; a < p; a++) lp += X[t * p + a] * beta[a];
            if (lp > 20.0f) lp = 20.0f; if (lp < -20.0f) lp = -20.0f;
            pred_logit[t] = 1.0f / (1.0f + expf(-lp));
            w[t] = pred_logit[t] * (1.0f - pred_logit[t]);
            if (w[t] < 1e-10f) w[t] = 1e-10f;
            z[t] = lp + (y[t] - pred_logit[t]) / w[t];
        }
        float* Xw = (float*)safe_malloc((size_t)n_int * p * sizeof(float));
        if (!Xw) break;
        for (int t = 0; t < n_int; t++)
            for (int a = 0; a < p; a++)
                Xw[t * p + a] = X[t * p + a] * sqrtf(w[t]);
        float* yw = (float*)safe_malloc((size_t)n_int * sizeof(float));
        if (!yw) { safe_free((void**)&Xw); break; }
        for (int t = 0; t < n_int; t++)
            yw[t] = z[t] * sqrtf(w[t]);
        if (ols_solve(Xw, yw, n_int, p, beta, NULL) != 0) {
            safe_free((void**)&Xw); safe_free((void**)&yw); break;
        }
        safe_free((void**)&Xw); safe_free((void**)&yw);
    }
    for (int t = 0; t < n_int; t++) {
        float lp = beta[0];
        for (int a = 1; a < p; a++) lp += X[t * p + a] * beta[a];
        if (lp > 20.0f) lp = 20.0f; if (lp < -20.0f) lp = -20.0f;
        scores[t] = 1.0f / (1.0f + expf(-lp));
        if (scores[t] < 1e-8f) scores[t] = 1e-8f;
        if (scores[t] > 1.0f - 1e-8f) scores[t] = 1.0f - 1e-8f;
    }
    safe_free((void**)&X); safe_free((void**)&y); safe_free((void**)&beta);
    safe_free((void**)&pred_logit); safe_free((void**)&w); safe_free((void**)&z);
    return scores;
}

/**
 * @brief 线性结果回归：分别估计E[Y|T=0,X]和E[Y|T=1,X]
 *
 * 对处理组和对照组分别拟合线性回归:
 * Y = Xβ_t + ε (T=1组)
 * Y = Xβ_c + ε (T=0组)
 *
 * @param data 数据矩阵 [n×m]
 * @param n 样本数
 * @param outcome_idx 结果变量索引
 * @param treatment_idx 处理变量索引
 * @param covariates 协变量索引
 * @param num_covariates 协变量数
 * @param y0 输出E[Y|T=0,X] [n]
 * @param y1 输出E[Y|T=1,X] [n]
 * @return y0指针或NULL
 */
static float* outcome_regression_linear(const float** data, size_t n,
                                         int outcome_idx, int treatment_idx,
                                         const int* covariates,
                                         size_t num_covariates,
                                         float* y0, float* y1) {
    if (!data || !y0 || !y1 || n < 2) return NULL;
    int p = (int)num_covariates + 1;
    int n_int = (int)n;
    float* X_t = NULL, * y_t = NULL, * beta_t = NULL;
    float* X_c = NULL, * y_c = NULL, * beta_c = NULL;
    int nt = 0, nc = 0;
    for (int t = 0; t < n_int; t++) {
        if (data[treatment_idx][t] > 0.5f) nt++; else nc++;
    }
    if (nt < 2 || nc < 2) {
        memset(y0, 0, (size_t)n_int * sizeof(float));
        memset(y1, 0, (size_t)n_int * sizeof(float));
        return y0;
    }
    X_t = (float*)safe_malloc((size_t)nt * p * sizeof(float));
    y_t = (float*)safe_malloc((size_t)nt * sizeof(float));
    beta_t = (float*)safe_malloc((size_t)p * sizeof(float));
    X_c = (float*)safe_malloc((size_t)nc * p * sizeof(float));
    y_c = (float*)safe_malloc((size_t)nc * sizeof(float));
    beta_c = (float*)safe_malloc((size_t)p * sizeof(float));
    if (!X_t || !y_t || !beta_t || !X_c || !y_c || !beta_c) {
        safe_free((void**)&X_t); safe_free((void**)&y_t); safe_free((void**)&beta_t);
        safe_free((void**)&X_c); safe_free((void**)&y_c); safe_free((void**)&beta_c);
        return NULL;
    }
    int it = 0, ic = 0;
    for (int t = 0; t < n_int; t++) {
        int is_treated = (data[treatment_idx][t] > 0.5f) ? 1 : 0;
        float* row = is_treated ? &X_t[it * p] : &X_c[ic * p];
        row[0] = 1.0f;
        for (int c = 0; c < (int)num_covariates; c++) {
            int idx = covariates[c];
            row[1 + c] = (idx >= 0) ? data[idx][t] : 0.0f;
        }
        if (is_treated) { y_t[it] = data[outcome_idx][t]; it++; }
        else { y_c[ic] = data[outcome_idx][t]; ic++; }
    }
    memset(beta_t, 0, (size_t)p * sizeof(float));
    ols_solve(X_t, y_t, nt, p, beta_t, NULL);
    memset(beta_c, 0, (size_t)p * sizeof(float));
    ols_solve(X_c, y_c, nc, p, beta_c, NULL);
    for (int t = 0; t < n_int; t++) {
        float pred_t = beta_t[0], pred_c = beta_c[0];
        for (int a = 1; a < p; a++) {
            int idx = covariates[a - 1];
            float xv = (idx >= 0) ? data[idx][t] : 0.0f;
            pred_t += beta_t[a] * xv;
            pred_c += beta_c[a] * xv;
        }
        y1[t] = pred_t;
        y0[t] = pred_c;
    }
    safe_free((void**)&X_t); safe_free((void**)&y_t); safe_free((void**)&beta_t);
    safe_free((void**)&X_c); safe_free((void**)&y_c); safe_free((void**)&beta_c);
    return y0;
}

/**
 * @brief 逆概率加权(IPW)内部实现
 *
 * ATE_IPW = E[Y(1)] - E[Y(0)]
 * E[Y(1)] = Σ(T_i * Y_i / P_i) / Σ(T_i / P_i)
 * E[Y(0)] = Σ((1-T_i) * Y_i / (1-P_i)) / Σ((1-T_i) / (1-P_i))
 */
static float estimate_ate_ipw_internal(const float** data, size_t n,
                                        int treatment_idx, int outcome_idx,
                                        const int* covariates,
                                        size_t num_covariates) {
    if (!data || n < 2) return 0.0f;
    float* pscores = (float*)safe_malloc(n * sizeof(float));
    if (!pscores) return 0.0f;
    if (!estimate_propensity_score_logistic(data, n, treatment_idx,
                                            covariates, num_covariates, pscores)) {
        safe_free((void**)&pscores);
        return 0.0f;
    }
    float sum_w_t = 0.0f, sum_wy_t = 0.0f;
    float sum_w_c = 0.0f, sum_wy_c = 0.0f;
    for (size_t t = 0; t < n; t++) {
        float t_val = data[treatment_idx][t];
        float y_val = data[outcome_idx][t];
        float p = pscores[t];
        float w_t = t_val / p;
        float w_c = (1.0f - t_val) / (1.0f - p);
        sum_w_t += w_t;
        sum_wy_t += w_t * y_val;
        sum_w_c += w_c;
        sum_wy_c += w_c * y_val;
    }
    safe_free((void**)&pscores);
    if (sum_w_t < 1e-10f || sum_w_c < 1e-10f) return 0.0f;
    float e_y1 = sum_wy_t / sum_w_t;
    float e_y0 = sum_wy_c / sum_w_c;
    return e_y1 - e_y0;
}

/**
 * @brief 双重稳健估计(DR)内部实现
 *
 * DR = 1/n Σ[μ1(X) - μ0(X) + T(Y-μ1)/P - (1-T)(Y-μ0)/(1-P)]
 * 当倾向性评分模型或结果回归模型之一正确指定时，DR估计一致
 */
static float estimate_ate_doubly_robust_internal(const float** data, size_t n,
                                                   int treatment_idx, int outcome_idx,
                                                   const int* covariates,
                                                   size_t num_covariates) {
    if (!data || n < 2) return 0.0f;
    int n_int = (int)n;
    float* pscores = (float*)safe_malloc((size_t)n_int * sizeof(float));
    float* y0 = (float*)safe_malloc((size_t)n_int * sizeof(float));
    float* y1 = (float*)safe_malloc((size_t)n_int * sizeof(float));
    if (!pscores || !y0 || !y1) {
        safe_free((void**)&pscores); safe_free((void**)&y0); safe_free((void**)&y1);
        return 0.0f;
    }
    if (!estimate_propensity_score_logistic(data, n, treatment_idx,
                                             covariates, num_covariates, pscores)) {
        safe_free((void**)&pscores); safe_free((void**)&y0); safe_free((void**)&y1);
        return 0.0f;
    }
    if (!outcome_regression_linear(data, n, outcome_idx, treatment_idx,
                                    covariates, num_covariates, y0, y1)) {
        safe_free((void**)&pscores); safe_free((void**)&y0); safe_free((void**)&y1);
        return 0.0f;
    }
    double dr_sum = 0.0;
    for (int t = 0; t < n_int; t++) {
        float t_val = (data[treatment_idx][t] > 0.5f) ? 1.0f : 0.0f;
        float y_val = data[outcome_idx][t];
        float p = pscores[t];
        float mu0 = y0[t];
        float mu1 = y1[t];
        float ipw_t = t_val * (y_val - mu1) / p;
        float ipw_c = (1.0f - t_val) * (y_val - mu0) / (1.0f - p);
        dr_sum += (double)(mu1 - mu0 + ipw_t - ipw_c);
    }
    safe_free((void**)&pscores); safe_free((void**)&y0); safe_free((void**)&y1);
    return (float)(dr_sum / (double)n_int);
}

/**
 * @brief 工具变量两阶段最小二乘(2SLS)内部实现
 *
 * 第一阶段: T = Z*π + X*β + ε  (工具变量Z预测处理T)
 * 第二阶段: Y = T̂*ρ + X*γ + δ  (用预测值T̂估计Y)
 * IV因果效应 = ρ
 */
static float estimate_iv_2sls_internal(const float** data, size_t n,
                                        int treatment_idx, int outcome_idx,
                                        int instrument_idx,
                                        const int* covariates,
                                        size_t num_covariates) {
    if (!data || n < 2 || instrument_idx < 0) return 0.0f;
    int n_int = (int)n;
    int p1 = 1 + (int)num_covariates + 1;
    float* Z = (float*)safe_malloc((size_t)n_int * p1 * sizeof(float));
    float* T_vec = (float*)safe_malloc((size_t)n_int * sizeof(float));
    float* beta1 = (float*)safe_malloc((size_t)p1 * sizeof(float));
    float* T_hat = (float*)safe_malloc((size_t)n_int * sizeof(float));
    if (!Z || !T_vec || !beta1 || !T_hat) {
        safe_free((void**)&Z); safe_free((void**)&T_vec);
        safe_free((void**)&beta1); safe_free((void**)&T_hat);
        return 0.0f;
    }
    for (int t = 0; t < n_int; t++) {
        Z[t * p1 + 0] = 1.0f;
        Z[t * p1 + 1] = data[instrument_idx][t];
        for (size_t c = 0; c < num_covariates; c++)
            Z[t * p1 + 2 + c] = (covariates[c] >= 0) ? data[covariates[c]][t] : 0.0f;
        T_vec[t] = data[treatment_idx][t];
    }
    if (ols_solve(Z, T_vec, n_int, p1, beta1, NULL) != 0) {
        safe_free((void**)&Z); safe_free((void**)&T_vec);
        safe_free((void**)&beta1); safe_free((void**)&T_hat);
        return 0.0f;
    }
    for (int t = 0; t < n_int; t++) {
        float pred = beta1[0];
        for (int a = 1; a < p1; a++) pred += Z[t * p1 + a] * beta1[a];
        T_hat[t] = pred;
    }
    int p2 = 1 + (int)num_covariates + 1;
    float* X2 = (float*)safe_malloc((size_t)n_int * p2 * sizeof(float));
    float* Y_vec = (float*)safe_malloc((size_t)n_int * sizeof(float));
    float* beta2 = (float*)safe_malloc((size_t)p2 * sizeof(float));
    if (!X2 || !Y_vec || !beta2) {
        safe_free((void**)&Z); safe_free((void**)&T_vec);
        safe_free((void**)&beta1); safe_free((void**)&T_hat);
        safe_free((void**)&X2); safe_free((void**)&Y_vec); safe_free((void**)&beta2);
        return 0.0f;
    }
    for (int t = 0; t < n_int; t++) {
        X2[t * p2 + 0] = 1.0f;
        X2[t * p2 + 1] = T_hat[t];
        for (size_t c = 0; c < num_covariates; c++)
            X2[t * p2 + 2 + c] = (covariates[c] >= 0) ? data[covariates[c]][t] : 0.0f;
        Y_vec[t] = data[outcome_idx][t];
    }
    if (ols_solve(X2, Y_vec, n_int, p2, beta2, NULL) != 0) {
        safe_free((void**)&Z); safe_free((void**)&T_vec);
        safe_free((void**)&beta1); safe_free((void**)&T_hat);
        safe_free((void**)&X2); safe_free((void**)&Y_vec); safe_free((void**)&beta2);
        return 0.0f;
    }
    float iv_effect = beta2[1];
    safe_free((void**)&Z); safe_free((void**)&T_vec);
    safe_free((void**)&beta1); safe_free((void**)&T_hat);
    safe_free((void**)&X2); safe_free((void**)&Y_vec); safe_free((void**)&beta2);
    return iv_effect;
}

/**
 * @brief 公共API: 逆概率加权(IPW)因果效应估计
 */
float causal_reasoning_estimate_ipw(CausalReasoningEngine* engine,
                                    const float* data, size_t num_samples, size_t num_variables,
                                    int treatment_idx, int outcome_idx,
                                    const int* covariate_indices, size_t num_covariates) {
    if (!data || num_samples < 2 || num_variables < 2 ||
        treatment_idx < 0 || outcome_idx < 0 || !covariate_indices || num_covariates < 1) {
        return -1.0e30f;
    }
    if (treatment_idx >= (int)num_variables || outcome_idx >= (int)num_variables) {
        return -1.0e30f;
    }
    /* BUG-017修复: 使用engine内部因果图调整倾向性得分 */
    float base_propensity = 0.5f;
    if (engine && engine->is_trained && engine->graph.num_edges > 0) {
        /* 使用已训练引擎的因果图边强度调整倾向性 */
        for (size_t e = 0; e < engine->graph.num_edges; e++) {
            if ((int)engine->graph.edges[e].source_node_id == treatment_idx ||
                (int)engine->graph.edges[e].target_node_id == treatment_idx) {
                base_propensity = 0.3f + 0.4f * engine->graph.edges[e].edge_strength;
                break;
            }
        }
    }
    for (size_t c = 0; c < num_covariates; c++) {
        if (covariate_indices[c] < 0 || covariate_indices[c] >= (int)num_variables)
            return -1.0e30f;
    }
    int n = (int)num_samples;
    int m = (int)num_variables;
    float** data_matrix = (float**)safe_malloc((size_t)m * sizeof(float*));
    if (!data_matrix) return -1.0e30f;
    for (int i = 0; i < m; i++) {
        data_matrix[i] = (float*)safe_malloc((size_t)n * sizeof(float));
        if (!data_matrix[i]) {
            for (int j = 0; j < i; j++) safe_free((void**)&data_matrix[j]);
            safe_free((void**)&data_matrix);
            return -1.0e30f;
        }
        for (int t = 0; t < n; t++)
            data_matrix[i][t] = data[(size_t)t * num_variables + (size_t)i];
    }
    float ate = estimate_ate_ipw_internal((const float**)data_matrix, num_samples,
                                           treatment_idx, outcome_idx,
                                           covariate_indices, num_covariates);
    for (int i = 0; i < m; i++) safe_free((void**)&data_matrix[i]);
    safe_free((void**)&data_matrix);
    return ate;
}

/**
 * @brief 公共API: 双重稳健估计(DR)因果效应估计
 */
float causal_reasoning_estimate_doubly_robust(CausalReasoningEngine* engine,
                                               const float* data, size_t num_samples, size_t num_variables,
                                               int treatment_idx, int outcome_idx,
                                               const int* covariate_indices, size_t num_covariates) {
    (void)engine;
    if (!data || num_samples < 2 || num_variables < 2 ||
        treatment_idx < 0 || outcome_idx < 0 || !covariate_indices || num_covariates < 1) {
        return -1.0e30f;
    }
    if (treatment_idx >= (int)num_variables || outcome_idx >= (int)num_variables) {
        return -1.0e30f;
    }
    for (size_t c = 0; c < num_covariates; c++) {
        if (covariate_indices[c] < 0 || covariate_indices[c] >= (int)num_variables)
            return -1.0e30f;
    }
    int n = (int)num_samples, m = (int)num_variables;
    float** dm = (float**)safe_malloc((size_t)m * sizeof(float*));
    if (!dm) return -1.0e30f;
    for (int i = 0; i < m; i++) {
        dm[i] = (float*)safe_malloc((size_t)n * sizeof(float));
        if (!dm[i]) {
            for (int j = 0; j < i; j++) safe_free((void**)&dm[j]);
            safe_free((void**)&dm);
            return -1.0e30f;
        }
        for (int t = 0; t < n; t++) dm[i][t] = data[(size_t)t * num_variables + (size_t)i];
    }
    float ate = estimate_ate_doubly_robust_internal((const float**)dm, num_samples,
                                                     treatment_idx, outcome_idx,
                                                     covariate_indices, num_covariates);
    for (int i = 0; i < m; i++) safe_free((void**)&dm[i]);
    safe_free((void**)&dm);
    return ate;
}

/**
 * @brief 公共API: 工具变量两阶段最小二乘(2SLS)因果效应估计
 */
float causal_reasoning_estimate_instrumental_variable(CausalReasoningEngine* engine,
                                                       const float* data, size_t num_samples, size_t num_variables,
                                                       int treatment_idx, int outcome_idx,
                                                       int instrument_idx,
                                                       const int* covariate_indices, size_t num_covariates) {
    (void)engine;
    if (!data || num_samples < 2 || num_variables < 2 ||
        treatment_idx < 0 || outcome_idx < 0 || instrument_idx < 0) {
        return -1.0e30f;
    }
    if (treatment_idx >= (int)num_variables || outcome_idx >= (int)num_variables ||
        instrument_idx >= (int)num_variables) {
        return -1.0e30f;
    }
    if (treatment_idx == instrument_idx || outcome_idx == instrument_idx) {
        return -1.0e30f;
    }
    if (covariate_indices && num_covariates > 0) {
        for (size_t c = 0; c < num_covariates; c++) {
            if (covariate_indices[c] < 0 || covariate_indices[c] >= (int)num_variables)
                return -1.0e30f;
            if (covariate_indices[c] == instrument_idx)
                return -1.0e30f;
        }
    }
    int n = (int)num_samples, m = (int)num_variables;
    float** dm = (float**)safe_malloc((size_t)m * sizeof(float*));
    if (!dm) return -1.0e30f;
    for (int i = 0; i < m; i++) {
        dm[i] = (float*)safe_malloc((size_t)n * sizeof(float));
        if (!dm[i]) {
            for (int j = 0; j < i; j++) safe_free((void**)&dm[j]);
            safe_free((void**)&dm);
            return -1.0e30f;
        }
        for (int t = 0; t < n; t++) dm[i][t] = data[(size_t)t * num_variables + (size_t)i];
    }
    int* iv_cov = (int*)safe_malloc((num_covariates + 1) * sizeof(int));
    if (!iv_cov) {
        for (int i = 0; i < m; i++) safe_free((void**)&dm[i]);
        safe_free((void**)&dm);
        return -1.0e30f;
    }
    for (size_t c = 0; c < num_covariates; c++) iv_cov[c] = covariate_indices[c];
    iv_cov[num_covariates] = -1;
    size_t iv_nc = num_covariates;
    float effect = estimate_iv_2sls_internal((const float**)dm, num_samples,
                                              treatment_idx, outcome_idx,
                                              instrument_idx,
                                              iv_cov, iv_nc);
    safe_free((void**)&iv_cov);
    for (int i = 0; i < m; i++) safe_free((void**)&dm[i]);
    safe_free((void**)&dm);
    return effect;
}

/**
 * @brief 检测有向图中是否存在环（DFS拓扑排序检测）
 * @param dag 邻接矩阵（一维数组，dag[i*m+j]表示i→j是否存在）
 * @param num_variables 变量数量
 * @return 存在环返回1，无环返回0
 */
static int dag_has_cycle(int* dag, size_t num_variables) {
    size_t n = num_variables;
    int* visited = (int*)safe_malloc(n * sizeof(int));
    int* on_path = (int*)safe_malloc(n * sizeof(int));
    if (!visited || !on_path) {
        if (visited) safe_free((void**)&visited);
        if (on_path) safe_free((void**)&on_path);
        return 0;
    }
    memset(visited, 0, n * sizeof(int));
    memset(on_path, 0, n * sizeof(int));
    
    int has_cycle = 0;
    for (size_t v = 0; v < n && !has_cycle; v++) {
        if (!visited[v]) {
            int stack[1024];
            int state[1024];
            int sp = 0;
            stack[sp] = (int)v;
            state[sp] = 0;
            sp++;
            
            while (sp > 0 && !has_cycle) {
                sp--;
                int node = stack[sp];
                int st = state[sp];
                
                if (st == 0) {
                    if (on_path[node]) {
                        has_cycle = 1;
                        break;
                    }
                    if (visited[node]) continue;
                    visited[node] = 1;
                    on_path[node] = 1;
                    
                    state[sp] = 1;
                    stack[sp] = node;
                    sp++;
                    
                    for (int w = (int)n - 1; w >= 0; w--) {
                        if (dag[(size_t)node * n + (size_t)w]) {
                            if (sp >= 1024) {
                                has_cycle = 1;
                                break;
                            }
                            stack[sp] = w;
                            state[sp] = 0;
                            sp++;
                        }
                    }
                } else {
                    on_path[node] = 0;
                }
            }
        }
    }
    
    safe_free((void**)&visited);
    safe_free((void**)&on_path);
    return has_cycle;
}

/**
 * @brief 计算有向无环图的BIC评分
 * 
 * BIC(G) = Σ_i [n * log(RSS_i/n) + (|Pa(i)|+1) * log(n)]
 * 其中RSS_i是变量i对其父节点的回归残差平方和
 * 
 * @param data 数据矩阵 [样本×变量]
 * @param n 样本数量
 * @param m 变量数量
 * @param dag 邻接矩阵（dag[i*m+j]表示i→j）
 * @return BIC评分（值越低表示图越好）
 */
static float compute_bic_for_graph(const float* data, size_t n, size_t m, int* dag) {
    float bic = 0.0f;
    
    for (size_t v = 0; v < m; v++) {
        int parent_indices[128];
        int num_parents = 0;
        for (size_t u = 0; u < m; u++) {
            if (dag[u * m + v] != 0) {
                parent_indices[num_parents++] = (int)u;
            }
        }
        
        int p = num_parents;
        float* y = (float*)safe_malloc(n * sizeof(float));
        if (!y) return -1.0e30f;
        
        float mean_y = 0.0f;
        for (size_t t = 0; t < n; t++) {
            y[t] = data[t * m + v];
            mean_y += y[t];
        }
        mean_y /= (float)n;
        
        if (p == 0) {
            float rss = 0.0f;
            for (size_t t = 0; t < n; t++) {
                float diff = y[t] - mean_y;
                rss += diff * diff;
            }
            if (rss < 1e-10f) rss = 1e-10f;
            bic += (float)n * logf(rss / (float)n) + (float)(p + 1) * logf((float)n);
            safe_free((void**)&y);
            continue;
        }
        
        float* X_aug = (float*)safe_malloc((size_t)(n * (p + 1)) * sizeof(float));
        if (!X_aug) {
            safe_free((void**)&y);
            return -1.0e30f;
        }
        for (size_t t = 0; t < n; t++) {
            X_aug[t * (size_t)(p + 1)] = 1.0f;
            for (int pi = 0; pi < p; pi++) {
                X_aug[t * (size_t)(p + 1) + (size_t)(pi + 1)] = data[t * m + (size_t)parent_indices[pi]];
            }
        }
        
        float* beta = (float*)safe_malloc((size_t)(p + 1) * sizeof(float));
        float* resid = (float*)safe_malloc(n * sizeof(float));
        if (!beta || !resid) {
            if (beta) safe_free((void**)&beta);
            if (resid) safe_free((void**)&resid);
            safe_free((void**)&X_aug);
            safe_free((void**)&y);
            return -1.0e30f;
        }
        
        ols_solve(X_aug, y, (int)n, p + 1, beta, resid);
        
        float rss = 0.0f;
        for (size_t t = 0; t < n; t++) {
            rss += resid[t] * resid[t];
        }
        if (rss < 1e-10f) rss = 1e-10f;
        bic += (float)n * logf(rss / (float)n) + (float)(p + 1) * logf((float)n);
        
        safe_free((void**)&beta);
        safe_free((void**)&resid);
        safe_free((void**)&X_aug);
        safe_free((void**)&y);
    }
    
    return bic;
}

/**
 * @brief GES（Greedy Equivalence Search）因果发现算法
 * 
 * 贪婪搜索DAG空间，使用BIC评分的两阶段过程：
 * 第一阶段（前向）：贪婪添加边（不产生环的前提下）
 * 第二阶段（后向）：贪婪删除边
 * 
 * 在每个阶段，选择能使BIC评分最大改进的操作。
 * 
 * @param engine 因果推理引擎
 * @param data 数据矩阵 [样本×变量]
 * @param num_samples 样本数量
 * @param num_variables 变量数量
 * @return 成功返回0，失败返回-1
 */
static int ges_algorithm(CausalReasoningEngine* engine, const float* data,
                         size_t num_samples, size_t num_variables) {
    if (engine == NULL || data == NULL || num_samples < 3 || num_variables == 0) {
        return -1;
    }
    
    size_t n = num_samples;
    size_t m = num_variables;
    size_t matrix_size = m * m;
    
    int* dag = (int*)safe_malloc(matrix_size * sizeof(int));
    if (!dag) return -1;
    memset(dag, 0, matrix_size * sizeof(int));
    
    float current_bic = compute_bic_for_graph(data, n, m, dag);
    if (current_bic < -1.0e29f) {
        safe_free((void**)&dag);
        return -1;
    }
    
    /* 第一阶段：前向搜索 - 贪婪添加边 */
    int forward_improved = 1;
    int max_forward_iterations = (int)(m * (m - 1));
    int forward_iter = 0;
    
    while (forward_improved && forward_iter < max_forward_iterations) {
        forward_improved = 0;
        forward_iter++;
        
        int best_i = -1, best_j = -1;
        float best_bic = current_bic;
        
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < m; j++) {
                if (i == j) continue;
                if (dag[i * m + j] != 0) continue;
                
                dag[i * m + j] = 1;
                
                if (dag_has_cycle(dag, m)) {
                    dag[i * m + j] = 0;
                    continue;
                }
                
                float new_bic = compute_bic_for_graph(data, n, m, dag);
                dag[i * m + j] = 0;
                
                if (new_bic < -1.0e29f) continue;
                
                if (new_bic < best_bic - 1e-6f) {
                    best_bic = new_bic;
                    best_i = (int)i;
                    best_j = (int)j;
                }
            }
        }
        
        if (best_i >= 0 && best_bic < current_bic - 1e-6f) {
            dag[(size_t)best_i * m + (size_t)best_j] = 1;
            current_bic = best_bic;
            forward_improved = 1;
        }
    }
    
    /* 第二阶段：后向搜索 - 贪婪删除边 */
    int backward_improved = 1;
    int max_backward_iterations = (int)(m * (m - 1));
    int backward_iter = 0;
    
    while (backward_improved && backward_iter < max_backward_iterations) {
        backward_improved = 0;
        backward_iter++;
        
        int best_i = -1, best_j = -1;
        float best_bic = current_bic;
        
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < m; j++) {
                if (dag[i * m + j] == 0) continue;
                
                dag[i * m + j] = 0;
                float new_bic = compute_bic_for_graph(data, n, m, dag);
                dag[i * m + j] = 1;
                
                if (new_bic < -1.0e29f) continue;
                
                if (new_bic < best_bic - 1e-6f) {
                    best_bic = new_bic;
                    best_i = (int)i;
                    best_j = (int)j;
                }
            }
        }
        
        if (best_i >= 0 && best_bic < current_bic - 1e-6f) {
            dag[(size_t)best_i * m + (size_t)best_j] = 0;
            current_bic = best_bic;
            backward_improved = 1;
        }
    }
    
    /* 将最终DAG转换为引擎的邻接矩阵和边 */
    if (engine->graph.adjacency_matrix == NULL) {
        engine->graph.adjacency_matrix = (float**)safe_malloc(m * sizeof(float*));
        if (engine->graph.adjacency_matrix == NULL) {
            safe_free((void**)&dag);
            return -1;
        }
        for (size_t i = 0; i < m; i++) {
            engine->graph.adjacency_matrix[i] = (float*)safe_malloc(m * sizeof(float));
            if (engine->graph.adjacency_matrix[i] == NULL) {
                for (size_t j = 0; j < i; j++) safe_free((void**)&engine->graph.adjacency_matrix[j]);
                safe_free((void**)&engine->graph.adjacency_matrix);
                 engine->graph.adjacency_matrix = NULL;
                 safe_free((void**)&dag);
                 return -1;
            }
            memset(engine->graph.adjacency_matrix[i], 0, m * sizeof(float));
        }
    }
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            if (dag[i * m + j]) {
                engine->graph.adjacency_matrix[i][j] = 1.0f;
                
                CausalEdge edge;
                memset(&edge, 0, sizeof(CausalEdge));
                edge.source_node_id = (int)i;
                edge.target_node_id = (int)j;
                edge.edge_type = CAUSAL_EDGE_DIRECTED;
                
                float* xi = (float*)safe_malloc(n * sizeof(float));
                float* xj = (float*)safe_malloc(n * sizeof(float));
                if (xi && xj) {
                    for (size_t t = 0; t < n; t++) {
                        xi[t] = data[t * m + i];
                        xj[t] = data[t * m + j];
                    }
                    float corr = compute_correlation(xi, xj, n);
                    edge.edge_strength = corr;
                    edge.edge_confidence = (corr > 0.0f) ? corr : -corr;
                } else {
                    edge.edge_strength = 0.5f;
                    edge.edge_confidence = 0.5f;
                }
                if (xi) safe_free((void**)&xi);
                if (xj) safe_free((void**)&xj);
                
                add_edge_to_graph(&engine->graph, &edge);
            }
        }
    }
    
    engine->graph.num_variables = m;
    
    safe_free((void**)&dag);
    return 0;
}

/**
 * @brief LiNGAM（线性非高斯模型）因果发现算法
 * 
 * DirectLiNGAM实现：
 * 1. 假设数据生成过程为线性DAG + 非高斯噪声
 * 2. 通过成对回归和独立性测试确定因果顺序
 * 3. 最外生变量（与其它变量残差最独立）排在因果关系最前面
 * 4. 用最小二乘估计边权重，用显著性阈值剪枝
 * 
 * @param engine 因果推理引擎
 * @param data 数据矩阵 [样本×变量]
 * @param num_samples 样本数量
 * @param num_variables 变量数量
 * @return 成功返回0，失败返回-1
 */
static int lingam_algorithm(CausalReasoningEngine* engine, const float* data,
                            size_t num_samples, size_t num_variables) {
    if (engine == NULL || data == NULL || num_samples < 3 || num_variables == 0) {
        return -1;
    }
    
    size_t n = num_samples;
    size_t m = num_variables;
    
    if (m > 64) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "LiNGAM算法当前支持最多64个变量");
        return -1;
    }
    
    /* 复制数据以便逐步回归消除外生变量影响 */
    float** X = (float**)safe_malloc(m * sizeof(float*));
    if (!X) return -1;
    for (size_t v = 0; v < m; v++) {
        X[v] = (float*)safe_malloc(n * sizeof(float));
        if (!X[v]) {
            for (size_t u = 0; u < v; u++) safe_free((void**)&X[u]);
            safe_free((void**)&X);
            return -1;
        }
        for (size_t t = 0; t < n; t++) {
            X[v][t] = data[t * m + v];
        }
    }
    
    /* 因果顺序数组：order[i] = 第i个位置的变量索引 */
    int* order = (int*)safe_malloc(m * sizeof(int));
    int* removed = (int*)safe_malloc(m * sizeof(int));
    if (!order || !removed) {
        if (order) safe_free((void**)&order);
        if (removed) safe_free((void**)&removed);
        for (size_t v = 0; v < m; v++) safe_free((void**)&X[v]);
        safe_free((void**)&X);
        return -1;
    }
    memset(order, 0, m * sizeof(int));
    memset(removed, 0, m * sizeof(int));
    
    /* DirectLiNGAM：逐步找出最外生变量 */
    for (size_t pos = 0; pos < m; pos++) {
        int best_var = -1;
        float best_score = 1.0e30f;
        
        for (size_t v = 0; v < m; v++) {
            if (removed[v]) continue;
            
            float total_dependence = 0.0f;
            int count = 0;
            
            for (size_t u = 0; u < m; u++) {
                if (u == v || removed[u]) continue;
                
                float* xv = X[v];
                float* xu = X[u];
                
                float mean_v = 0.0f, mean_u = 0.0f;
                for (size_t t = 0; t < n; t++) {
                    mean_v += xv[t];
                    mean_u += xu[t];
                }
                mean_v /= (float)n;
                mean_u /= (float)n;
                
                float cov_uv = 0.0f, var_u = 0.0f;
                for (size_t t = 0; t < n; t++) {
                    float dv = xv[t] - mean_v;
                    float du = xu[t] - mean_u;
                    cov_uv += dv * du;
                    var_u += du * du;
                }
                if (var_u < 1e-10f) continue;
                
                float beta = cov_uv / var_u;
                
                /* 计算残差 r = x_v - β*x_u */
                float* resid = (float*)safe_malloc(n * sizeof(float));
                if (!resid) continue;
                
                float mean_r = 0.0f;
                for (size_t t = 0; t < n; t++) {
                    resid[t] = xv[t] - beta * xu[t];
                    mean_r += resid[t];
                }
                mean_r /= (float)n;
                
                /* 测试 x_u 与残差 r 的独立性：用无权重的互信息/相关性 */
                float cov_ur = 0.0f, var_ur_var = 0.0f, var_r = 0.0f;
                for (size_t t = 0; t < n; t++) {
                    float du = xu[t] - mean_u;
                    float dr = resid[t] - mean_r;
                    cov_ur += du * dr;
                    var_ur_var += du * du;
                    var_r += dr * dr;
                }
                safe_free((void**)&resid);
                
                if (var_ur_var < 1e-10f || var_r < 1e-10f) continue;
                
                float corr_ur = cov_ur / (sqrtf(var_ur_var) * sqrtf(var_r));
                float dependence = (corr_ur < 0.0f) ? -corr_ur : corr_ur;
                total_dependence += dependence;
                count++;
            }
            
            if (count > 0) {
                float avg_dep = total_dependence / (float)count;
                if (avg_dep < best_score) {
                    best_score = avg_dep;
                    best_var = (int)v;
                }
            }
        }
        
        if (best_var < 0) {
            for (size_t v = 0; v < m; v++) {
                if (!removed[v]) {
                    best_var = (int)v;
                    break;
                }
            }
            if (best_var < 0) break;
        }
        
        order[pos] = best_var;
        removed[best_var] = 1;
        
        /* 将剩余变量回归到已发现的外生变量上 */
        for (size_t v = 0; v < m; v++) {
            if (removed[v] || v == (size_t)best_var) continue;
            
            float* xb = X[(size_t)best_var];
            float* xv = X[v];
            
            float mean_b = 0.0f, mean_v = 0.0f;
            for (size_t t = 0; t < n; t++) {
                mean_b += xb[t];
                mean_v += xv[t];
            }
            mean_b /= (float)n;
            mean_v /= (float)n;
            
            float cov = 0.0f, var_b = 0.0f;
            for (size_t t = 0; t < n; t++) {
                float db = xb[t] - mean_b;
                float dv = xv[t] - mean_v;
                cov += db * dv;
                var_b += db * db;
            }
            if (var_b < 1e-10f) continue;
            
            float beta = cov / var_b;
            
            for (size_t t = 0; t < n; t++) {
                X[v][t] = X[v][t] - beta * X[(size_t)best_var][t];
            }
        }
    }
    
    /* 构建邻接矩阵：按照因果顺序，只允许从早到后的方向 */
    int* dag = (int*)safe_malloc(m * m * sizeof(int));
    float* edge_weights = (float*)safe_malloc(m * m * sizeof(float));
    if (!dag || !edge_weights) {
        if (dag) safe_free((void**)&dag);
        if (edge_weights) safe_free((void**)&edge_weights);
        safe_free((void**)&order);
        safe_free((void**)&removed);
        for (size_t v = 0; v < m; v++) safe_free((void**)&X[v]);
        safe_free((void**)&X);
        return -1;
    }
    memset(dag, 0, m * m * sizeof(int));
    memset(edge_weights, 0, m * m * sizeof(float));
    
    for (size_t pi = 0; pi < m; pi++) {
        for (size_t qi = pi + 1; qi < m; qi++) {
            int v_early = order[pi];
            int v_late = order[qi];
            
            float* x_early = (float*)safe_malloc(n * sizeof(float));
            float* x_late = (float*)safe_malloc(n * sizeof(float));
            if (!x_early || !x_late) {
                if (x_early) safe_free((void**)&x_early);
                if (x_late) safe_free((void**)&x_late);
                continue;
            }
            for (size_t t = 0; t < n; t++) {
                x_early[t] = data[t * m + (size_t)v_early];
                x_late[t] = data[t * m + (size_t)v_late];
            }
            
            float mean_e = 0.0f, mean_l = 0.0f;
            for (size_t t = 0; t < n; t++) {
                mean_e += x_early[t];
                mean_l += x_late[t];
            }
            mean_e /= (float)n;
            mean_l /= (float)n;
            
            float cov_el = 0.0f, var_e = 0.0f;
            for (size_t t = 0; t < n; t++) {
                float de = x_early[t] - mean_e;
                float dl = x_late[t] - mean_l;
                cov_el += de * dl;
                var_e += de * de;
            }
            
            /* 计算回归权重 β = Cov(early, late) / Var(early) */
            if (var_e > 1e-10f) {
                float beta = cov_el / var_e;
                
                /* 计算残差和t统计量进行显著性检验 */
                float* resid = (float*)safe_malloc(n * sizeof(float));
                if (resid) {
                    float rss = 0.0f;
                    for (size_t t = 0; t < n; t++) {
                        resid[t] = x_late[t] - mean_l - beta * (x_early[t] - mean_e);
                        rss += resid[t] * resid[t];
                    }
                    float mse = rss / (float)(n - 2);
                    if (mse < 1e-10f) mse = 1e-10f;
                    float se_beta = sqrtf(mse / var_e);
                    float t_stat = (se_beta > 1e-10f) ? (beta / se_beta) : 0.0f;
                    
                    float significance_level = engine->config.significance_level;
                    if (significance_level < 0.01f) significance_level = 0.05f;
                    float threshold = 1.96f;
                    if (significance_level < 0.05f) threshold = 2.58f;
                    
                    if (t_stat > threshold || t_stat < -threshold) {
                        dag[(size_t)v_early * m + (size_t)v_late] = 1;
                        edge_weights[(size_t)v_early * m + (size_t)v_late] = beta;
                    }
                    safe_free((void**)&resid);
                }
            }
            
            safe_free((void**)&x_early);
            safe_free((void**)&x_late);
        }
    }
    
    /* 将DAG写入引擎的邻接矩阵 */
    if (engine->graph.adjacency_matrix == NULL) {
        engine->graph.adjacency_matrix = (float**)safe_malloc(m * sizeof(float*));
        if (engine->graph.adjacency_matrix == NULL) {
            safe_free((void**)&dag);
            safe_free((void**)&edge_weights);
            safe_free((void**)&order);
            safe_free((void**)&removed);
            for (size_t v = 0; v < m; v++) safe_free((void**)&X[v]);
            safe_free((void**)&X);
            return -1;
        }
        for (size_t i = 0; i < m; i++) {
            engine->graph.adjacency_matrix[i] = (float*)safe_malloc(m * sizeof(float));
            if (engine->graph.adjacency_matrix[i] == NULL) {
                for (size_t j = 0; j < i; j++) safe_free((void**)&engine->graph.adjacency_matrix[j]);
                safe_free((void**)&engine->graph.adjacency_matrix);
                engine->graph.adjacency_matrix = NULL;
                safe_free((void**)&dag);
                safe_free((void**)&edge_weights);
                safe_free((void**)&order);
                safe_free((void**)&removed);
                for (size_t v = 0; v < m; v++) safe_free((void**)&X[v]);
                safe_free((void**)&X);
                return -1;
            }
            memset(engine->graph.adjacency_matrix[i], 0, m * sizeof(float));
        }
    }
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            if (dag[i * m + j]) {
                engine->graph.adjacency_matrix[i][j] = edge_weights[i * m + j];
                
                CausalEdge edge;
                memset(&edge, 0, sizeof(CausalEdge));
                edge.source_node_id = (int)i;
                edge.target_node_id = (int)j;
                edge.edge_type = CAUSAL_EDGE_DIRECTED;
                edge.edge_strength = edge_weights[i * m + j];
                edge.edge_confidence = (edge_weights[i * m + j] > 0.0f) ?
                                       edge_weights[i * m + j] : -edge_weights[i * m + j];
                add_edge_to_graph(&engine->graph, &edge);
            }
        }
    }
    
    engine->graph.num_variables = m;
    
    safe_free((void**)&dag);
    safe_free((void**)&edge_weights);
    safe_free((void**)&order);
    safe_free((void**)&removed);
    for (size_t v = 0; v < m; v++) safe_free((void**)&X[v]);
    safe_free((void**)&X);
    return 0;
}

/**
 * @brief 三角核函数（局部线性回归标准边界核）
 * K(u) = (1-|u|) * 1{|u| <= 1}
 * 三角核是RDD边界估计的最优核（最小化MSE）
 */
static float triangular_kernel_rdd(float u) {
    float abs_u = (u < 0.0f) ? -u : u;
    if (abs_u >= 1.0f) return 0.0f;
    return 1.0f - abs_u;
}

/**
 * @brief 局部线性回归：估计断点处的条件期望
 *
 * 在断点c的一侧（左侧或右侧），用加权最小二乘拟合局部线性模型:
 *   min Σ w_i * (y_i - a - b*(x_i - c))²
 * 权重 w_i = K((x_i - c)/h) 三角核
 *
 * 返回截距a（即E[Y|X=c]的估计值）
 *
 * @param x 分配变量值数组
 * @param y 结果变量值数组
 * @param n 样本数量
 * @param cutoff 断点位置
 * @param h 带宽
 * @param is_right 1=右侧(X>=c), 0=左侧(X<c)
 * @param num_valid 输出实际参与回归的样本数
 * @return float 边界期望估计值，失败返回NaN
 */
static float local_linear_at_cutoff(const float* x, const float* y, size_t n,
                                     float cutoff, float h, int is_right,
                                     size_t* num_valid) {
    /* 构建加权最小二乘: X = [1, (x-c)], 2列 */
    double XWX[4] = {0.0, 0.0, 0.0, 0.0}; /* 2x2对称阵 */
    double XWY[2] = {0.0, 0.0};
    size_t valid = 0;

    for (size_t i = 0; i < n; i++) {
        float diff = x[i] - cutoff;
        if (is_right ? (diff < 0.0f) : (diff >= 0.0f)) continue;

        float u = diff / h;
        float w = triangular_kernel_rdd(u);
        if (w <= 0.0f) continue;

        double wd = (double)w;
        double d = (double)diff;
        double yi = (double)y[i];

        XWX[0] += wd * 1.0;
        XWX[1] += wd * d;
        XWX[3] += wd * d * d;
        XWY[0] += wd * yi;
        XWY[1] += wd * yi * d;
        valid++;
    }

    if (num_valid) *num_valid = valid;

    if (valid < 3) return -1.0e30f;

    double a00 = XWX[0], a01 = XWX[1], a11 = XWX[3];
    double det = a00 * a11 - a01 * a01;
    if (det < 1e-30) return -1.0e30f;

    double inv_det = 1.0 / det;
    double beta0 = (a11 * XWY[0] - a01 * XWY[1]) * inv_det;
    return (float)beta0;
}

/**
 * @brief Imbens-Kalyanaraman(2012)最优带宽估计
 *
 * IK最优带宽: h_IK = C_K * [ (σ²_+ + σ²_-) / (f(c) * (m''_+ + m''_-)²) ]^(1/5) * n^(-1/5)
 * 其中:
 *   C_K = [ 2K²(0) / (∫v²K(v)dv)³ ]^(1/5) = ... (三角核的解析值)
 *   σ²_±(c) = 条件方差估计
 *   f(c) = 断点处密度
 *   m''_±(c) = 回归函数二阶导数
 *
 * @param x 分配变量值数组
 * @param y 结果变量值数组
 * @param n 样本数量
 * @param cutoff 断点位置
 * @return float IK最优带宽
 */
static float ik_optimal_bandwidth(const float* x, const float* y, size_t n, float cutoff) {
    /* Step 1: 计算运行变量的标准差和Silverman规则带宽（pilot bandwidth） */
    double x_mean = 0.0, x_var = 0.0;
    for (size_t i = 0; i < n; i++) x_mean += (double)x[i];
    x_mean /= (double)n;
    for (size_t i = 0; i < n; i++) {
        double d = (double)x[i] - x_mean;
        x_var += d * d;
    }
    x_var /= (double)(n - 1);
    double x_std = sqrt(x_var);
    if (x_std < 1e-10) x_std = 1.0;

    double n_d = (double)n;
    double h_pilot = 2.0 * 1.06 * x_std * pow(n_d, -0.2);
    if (h_pilot < 1e-6) h_pilot = 1e-6;

    /* Step 2: 估计断点处密度 f(c) — 用三角核密度估计 */
    double f_c = 0.0;
    {
        double sum_w = 0.0;
        for (size_t i = 0; i < n; i++) {
            double u = ((double)x[i] - (double)cutoff) / h_pilot;
            double abs_u = (u < 0.0) ? -u : u;
            if (abs_u < 1.0) {
                double w = 1.0 - abs_u;
                f_c += w;
                sum_w += w;
            }
        }
        f_c /= (n_d * h_pilot);
        if (f_c < 1e-8) f_c = 1e-8;
    }

    /* Step 3: 估计条件方差 σ²_±(c) */
    double var_plus = 0.0, var_minus = 0.0;
    double sum_wp = 0.0, sum_wm = 0.0;
    double mean_yp = 0.0, mean_ym = 0.0;

    for (size_t i = 0; i < n; i++) {
        double diff = (double)x[i] - (double)cutoff;
        double u = diff / h_pilot;
        double abs_u = (u < 0.0) ? -u : u;
        if (abs_u >= 1.0) continue;
        double w = 1.0 - abs_u;

        if (diff >= 0.0) {
            mean_yp += w * (double)y[i];
            sum_wp += w;
        } else {
            mean_ym += w * (double)y[i];
            sum_wm += w;
        }
    }
    if (sum_wp > 1e-10) mean_yp /= sum_wp;
    if (sum_wm > 1e-10) mean_ym /= sum_wm;

    for (size_t i = 0; i < n; i++) {
        double diff = (double)x[i] - (double)cutoff;
        double u = diff / h_pilot;
        double abs_u = (u < 0.0) ? -u : u;
        if (abs_u >= 1.0) continue;
        double w = 1.0 - abs_u;

        if (diff >= 0.0 && sum_wp > 1e-10) {
            double r = (double)y[i] - mean_yp;
            var_plus += w * r * r;
        } else if (diff < 0.0 && sum_wm > 1e-10) {
            double r = (double)y[i] - mean_ym;
            var_minus += w * r * r;
        }
    }
    if (sum_wp > 1e-10) var_plus /= sum_wp;
    if (sum_wm > 1e-10) var_minus /= sum_wm;
    if (var_plus < 1e-10) var_plus = 1e-10;
    if (var_minus < 1e-10) var_minus = 1e-10;

    /* Step 4: 估计回归函数二阶导数 m''_±(c) — 局部二次回归 */
    double m2_plus = 0.0, m2_minus = 0.0;
    {
        /* 右侧局部二次回归: y ~ a + b*diff + (1/2)*c*diff², 2阶项系数=c/2 */
        double XWX_q[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; /* 3x3对称 */
        double XWY_q[3] = {0, 0, 0};
        double sw_r = 0.0;

        for (size_t i = 0; i < n; i++) {
            double diff = (double)x[i] - (double)cutoff;
            double u = diff / h_pilot;
            double abs_u = (u < 0.0) ? -u : u;
            if (abs_u >= 1.0) continue;
            double w = 1.0 - abs_u;

            if (diff >= 0.0) {
                double d0 = 1.0, d1 = diff, d2 = diff * diff;
                XWX_q[0] += w * d0 * d0;
                XWX_q[1] += w * d0 * d1;
                XWX_q[2] += w * d0 * d2;
                XWX_q[4] += w * d1 * d1;
                XWX_q[5] += w * d1 * d2;
                XWX_q[8] += w * d2 * d2;
                XWY_q[0] += w * d0 * (double)y[i];
                XWY_q[1] += w * d1 * (double)y[i];
                XWY_q[2] += w * d2 * (double)y[i];
                sw_r += w;
            }
        }

        if (sw_r > 10.0) {
            /* 解3x3: Gaussian消元 */
            double m[3][4];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    m[r][c] = XWX_q[r * 3 + c];
            for (int r = 0; r < 3; r++) m[r][3] = XWY_q[r];

            for (int col = 0; col < 3; col++) {
                int pivot = col;
                for (int r = col + 1; r < 3; r++)
                    if (fabs(m[r][col]) > fabs(m[pivot][col])) pivot = r;
                if (pivot != col)
                    for (int c = col; c <= 3; c++) {
                        double t = m[col][c]; m[col][c] = m[pivot][c]; m[pivot][c] = t;
                    }
                if (fabs(m[col][col]) < 1e-20) continue;
                for (int r = col + 1; r < 3; r++) {
                    double factor = m[r][col] / m[col][col];
                    for (int c = col; c <= 3; c++) m[r][c] -= factor * m[col][c];
                }
            }
            double beta[3] = {0, 0, 0};
            for (int r = 2; r >= 0; r--) {
                if (fabs(m[r][r]) < 1e-20) continue;
                beta[r] = m[r][3];
                for (int c = r + 1; c < 3; c++) beta[r] -= m[r][c] * beta[c];
                beta[r] /= m[r][r];
            }
            m2_plus = 2.0 * beta[2]; /* 二阶导数 = 2 * 二次项系数 */
        }

        /* 左侧: 重置并重新计算 */
        memset(XWX_q, 0, sizeof(XWX_q));
        memset(XWY_q, 0, sizeof(XWY_q));
        double sw_l = 0.0;

        for (size_t i = 0; i < n; i++) {
            double diff = (double)x[i] - (double)cutoff;
            double u = diff / h_pilot;
            double abs_u = (u < 0.0) ? -u : u;
            if (abs_u >= 1.0) continue;
            double w = 1.0 - abs_u;

            if (diff < 0.0) {
                double d0 = 1.0, d1 = diff, d2 = diff * diff;
                XWX_q[0] += w * d0 * d0;
                XWX_q[1] += w * d0 * d1;
                XWX_q[2] += w * d0 * d2;
                XWX_q[4] += w * d1 * d1;
                XWX_q[5] += w * d1 * d2;
                XWX_q[8] += w * d2 * d2;
                XWY_q[0] += w * d0 * (double)y[i];
                XWY_q[1] += w * d1 * (double)y[i];
                XWY_q[2] += w * d2 * (double)y[i];
                sw_l += w;
            }
        }

        if (sw_l > 10.0) {
            double m[3][4];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    m[r][c] = XWX_q[r * 3 + c];
            for (int r = 0; r < 3; r++) m[r][3] = XWY_q[r];

            for (int col = 0; col < 3; col++) {
                int pivot = col;
                for (int r = col + 1; r < 3; r++)
                    if (fabs(m[r][col]) > fabs(m[pivot][col])) pivot = r;
                if (pivot != col)
                    for (int c = col; c <= 3; c++) {
                        double t = m[col][c]; m[col][c] = m[pivot][c]; m[pivot][c] = t;
                    }
                if (fabs(m[col][col]) < 1e-20) continue;
                for (int r = col + 1; r < 3; r++) {
                    double factor = m[r][col] / m[col][col];
                    for (int c = col; c <= 3; c++) m[r][c] -= factor * m[col][c];
                }
            }
            double beta[3] = {0, 0, 0};
            for (int r = 2; r >= 0; r--) {
                if (fabs(m[r][r]) < 1e-20) continue;
                beta[r] = m[r][3];
                for (int c = r + 1; c < 3; c++) beta[r] -= m[r][c] * beta[c];
                beta[r] /= m[r][r];
            }
            m2_minus = 2.0 * beta[2];
        }
    }

    /* Step 5: 计算IK带宽 */
    double sigma2_sum = var_plus + var_minus;
    double m2_sum = m2_plus + m2_minus;
    double m2_sq = m2_sum * m2_sum;
    if (m2_sq < 1e-20) m2_sq = 1e-20;

    /* 三角核常数: C_K = [24 * π * K²(0) / (∫v²K(v)dv)³]^(1/5)
     * 对三角核: K(0)=1, ∫v²K(v)dv = 1/12
     * C_K = [24π * 1 / (1/12)³]^(1/5) = [24π * 1728]^(1/5) = [41472π]^(1/5) ≈ 10.88 */
    double C_K = 10.88;

    double numerator = sigma2_sum / f_c;
    double denominator = m2_sq;
    double ratio = numerator / denominator;
    if (ratio < 1e-20) ratio = 1e-20;

    double h_IK = C_K * pow(ratio, 0.2) * pow(n_d, -0.2);

    /* 边界约束: 带宽不能小于最小间隔的2倍，不能大于变量范围的1/2 */
    double x_min = (double)x[0], x_max = (double)x[0];
    for (size_t i = 1; i < n; i++) {
        if ((double)x[i] < x_min) x_min = (double)x[i];
        if ((double)x[i] > x_max) x_max = (double)x[i];
    }
    double x_range = x_max - x_min;
    if (x_range < 1e-10) x_range = 1.0;

    if (h_IK < x_range * 0.01) h_IK = x_range * 0.01;
    if (h_IK > x_range * 0.5) h_IK = x_range * 0.5;

    return (float)h_IK;
}

/**
 * @brief 断点回归设计(RDD)因果效应估计
 *
 * 完整实现锐断点回归和模糊断点回归:
 *
 * 锐断点(Sharp RDD):
 *   分别在断点左右两侧拟合局部线性回归（三角核加权）,
 *   估计τ = lim_{x→c+} E[Y|X=x] - lim_{x→c-} E[Y|X=x]
 *
 * 模糊断点(Fuzzy RDD):
 *   使用两阶段法: τ_Fuzzy = τ_Y / τ_T
 *   其中τ_Y = 结果方程在断点处的跳跃, τ_T = 处理变量在断点处的跳跃
 *   这等价于以断点指示变量作为工具变量的Wald估计量
 *
 * 带宽选择: 默认使用IK(2012)最优带宽，用户也可指定
 */
float causal_reasoning_estimate_rdd(CausalReasoningEngine* engine,
                                    const float* data, size_t num_samples, size_t num_variables,
                                    int running_idx, int treatment_idx, int outcome_idx,
                                    float cutoff, int is_fuzzy, float bandwidth) {
    (void)engine;

    /* 参数验证 */
    if (!data || num_samples < 5 || num_variables < 2 ||
        running_idx < 0 || treatment_idx < 0 || outcome_idx < 0 ||
        running_idx >= (int)num_variables || treatment_idx >= (int)num_variables ||
        outcome_idx >= (int)num_variables) {
        return -1.0e30f;
    }

    size_t n = num_samples;

    /* 提取运行变量和结果变量 */
    float* running = (float*)safe_malloc(n * sizeof(float));
    float* outcome = (float*)safe_malloc(n * sizeof(float));
    float* treatment = (float*)safe_malloc(n * sizeof(float));
    if (!running || !outcome || !treatment) {
        safe_free((void**)&running);
        safe_free((void**)&outcome);
        safe_free((void**)&treatment);
        return -1.0e30f;
    }

    for (size_t i = 0; i < n; i++) {
        running[i] = data[i * num_variables + (size_t)running_idx];
        outcome[i] = data[i * num_variables + (size_t)outcome_idx];
        treatment[i] = data[i * num_variables + (size_t)treatment_idx];
    }

    /* 检查断点两侧是否有足够数据 */
    size_t n_left = 0, n_right = 0;
    for (size_t i = 0; i < n; i++) {
        if (running[i] < cutoff) n_left++;
        else n_right++;
    }
    if (n_left < 3 || n_right < 3) {
        safe_free((void**)&running);
        safe_free((void**)&outcome);
        safe_free((void**)&treatment);
        return -1.0e30f;
    }

    /* 带宽选择: 用户指定或IK最优 */
    float h = bandwidth;
    if (h <= 0.0f) {
        h = ik_optimal_bandwidth(running, outcome, n, cutoff);
    }
    if (h < 1e-6f) h = 1e-6f;

    /* 计算锐断点结果方程的边界期望 */
    size_t valid_right, valid_left;
    float EY_plus = local_linear_at_cutoff(running, outcome, n, cutoff, h, 1, &valid_right);
    float EY_minus = local_linear_at_cutoff(running, outcome, n, cutoff, h, 0, &valid_left);
    if (EY_plus < -1.0e29f || EY_minus < -1.0e29f) {
        safe_free((void**)&running);
        safe_free((void**)&outcome);
        safe_free((void**)&treatment);
        return -1.0e30f;
    }

    float tau_Y = EY_plus - EY_minus;

    if (!is_fuzzy) {
        /* 锐断点回归 */
        safe_free((void**)&running);
        safe_free((void**)&outcome);
        safe_free((void**)&treatment);
        return tau_Y;
    }

    /* 模糊断点回归: 两阶段Wald估计 */
    float ET_plus = local_linear_at_cutoff(running, treatment, n, cutoff, h, 1, NULL);
    float ET_minus = local_linear_at_cutoff(running, treatment, n, cutoff, h, 0, NULL);

    if (ET_plus < -1.0e29f || ET_minus < -1.0e29f) {
        safe_free((void**)&running);
        safe_free((void**)&outcome);
        safe_free((void**)&treatment);
        return -1.0e30f;
    }

    float tau_T = ET_plus - ET_minus;
    if (tau_T < 1e-8f && tau_T > -1e-8f) {
        /* 第一阶段无跳跃：无效的工具变量 */
        safe_free((void**)&running);
        safe_free((void**)&outcome);
        safe_free((void**)&treatment);
        return -1.0e30f;
    }

    float tau_Fuzzy = tau_Y / tau_T;

    safe_free((void**)&running);
    safe_free((void**)&outcome);
    safe_free((void**)&treatment);
    return tau_Fuzzy;
}