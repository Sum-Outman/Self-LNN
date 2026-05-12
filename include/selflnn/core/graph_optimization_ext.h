#ifndef SELFLNN_GRAPH_OPTIMIZATION_EXT_H
#define SELFLNN_GRAPH_OPTIMIZATION_EXT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==================== A01.4.1 非线性最小二乘优化器 ==================== */

/* ---------- Levenberg-Marquardt 优化器 ---------- */

typedef struct {
    int n_params;
    int n_residuals;
    float* params;
    float* residuals;
    float* jacobian;
    float* workspace;
    float lambda;
    float lambda_factor;
    float gtol;
    float ftol;
    float xtol;
    int max_iterations;
    int n_iterations;
    int status;
} LMOptimizer;

typedef int (*LMResidualFunc)(const float* params, float* residuals, void* user_data);
typedef int (*LMJacobianFunc)(const float* params, float* jacobian, void* user_data);

int lm_optimizer_create(LMOptimizer* opt, int n_params, int n_residuals);
void lm_optimizer_free(LMOptimizer* opt);
int lm_optimizer_solve(LMOptimizer* opt, float* initial_params,
                       LMResidualFunc res_func, LMJacobianFunc jac_func,
                       void* user_data);
void lm_optimizer_set_defaults(LMOptimizer* opt);

/* ---------- Dogleg 信赖域优化器 ---------- */

typedef struct {
    int n_params;
    int n_residuals;
    float* params;
    float* residuals;
    float* jacobian;
    float* gradient;
    float* gn_step;
    float* cauchy_step;
    float* workspace;
    float trust_radius;
    float delta_max;
    float gtol;
    float ftol;
    float xtol;
    int max_iterations;
    int n_iterations;
    int status;
} DoglegOptimizer;

int dogleg_optimizer_create(DoglegOptimizer* opt, int n_params, int n_residuals);
void dogleg_optimizer_free(DoglegOptimizer* opt);
int dogleg_optimizer_solve(DoglegOptimizer* opt, float* initial_params,
                           LMResidualFunc res_func, LMJacobianFunc jac_func,
                           void* user_data);
void dogleg_optimizer_set_defaults(DoglegOptimizer* opt);

/* ---------- Powell 无导数优化器 ---------- */

typedef int (*PowellFunc)(const float* params, float* value, void* user_data);

typedef struct {
    int n_params;
    float* params;
    float* directions;
    float* workspace;
    float* prev_params;
    float* step;
    float ftol;
    float xtol;
    int max_iterations;
    int n_iterations;
    int status;
    float initial_radius;
} PowellOptimizer;

int powell_optimizer_create(PowellOptimizer* opt, int n_params);
void powell_optimizer_free(PowellOptimizer* opt);
int powell_optimizer_solve(PowellOptimizer* opt, float* initial_params,
                           PowellFunc func, void* user_data);
void powell_optimizer_set_defaults(PowellOptimizer* opt);

/* ==================== A01.4.2 概率图模型 ==================== */

/* ---------- 因子图 (Factor Graph) ---------- */

typedef struct {
    int id;
    char name[64];
    int domain_size;
    float* belief;
    int* factor_ids;
    int n_factors;
} FgVariableNode;

typedef struct {
    int id;
    char name[64];
    int* connected_vars;
    int n_connected_vars;
    int* var_dimensions;
    float* values;
} FgFactorNode;

typedef struct {
    FgVariableNode* variables;
    int n_variables;
    FgFactorNode* factors;
    int n_factors;
    int max_variables;
    int max_factors;
    int* elim_order;
} FactorGraph;

int fg_create(FactorGraph* fg, int max_vars, int max_factors);
void fg_free(FactorGraph* fg);
int fg_add_variable(FactorGraph* fg, int id, const char* name, int domain_size);
int fg_add_factor(FactorGraph* fg, int id, const char* name,
                  const int* var_ids, int n_vars, const float* values);
int fg_connect(FactorGraph* fg, int factor_id, int var_id);
int fg_sum_product(FactorGraph* fg, int target_var, float* marginal, int max_iter);
int fg_max_product(FactorGraph* fg, int* map_assignment, int max_iter);
int fg_variable_elimination(FactorGraph* fg, int elim_var, int* new_factor_vars, float* new_factor_values);

/* ---------- 贝叶斯网络 (Bayesian Network) ---------- */

typedef struct {
    int id;
    char name[64];
    int domain_size;
    int* parent_ids;
    int n_parents;
    int* child_ids;
    int n_children;
    float* cpt;
    int cpt_size;
} BNNode;

typedef struct {
    BNNode* nodes;
    int n_nodes;
    int max_nodes;
    int* topological_order;
    int* sorted;
    int* visited;
} BayesianNetwork;

int bn_create(BayesianNetwork* bn, int max_nodes);
void bn_free(BayesianNetwork* bn);
int bn_add_node(BayesianNetwork* bn, int id, const char* name, int domain_size);
int bn_add_edge(BayesianNetwork* bn, int parent_id, int child_id);
int bn_set_cpt(BayesianNetwork* bn, int node_id, const float* cpt_values, int cpt_size);
int bn_topological_sort(BayesianNetwork* bn);
float bn_infer_marginal(const BayesianNetwork* bn, int query_var, int evidence_var, int evidence_val);
int bn_sample(const BayesianNetwork* bn, int* samples, int n_samples);

/* ---------- 马尔可夫随机场 (Markov Random Field) ---------- */

typedef struct {
    int* node_ids;
    int n_nodes;
    float* potential;
} Clique;

typedef struct {
    int id;
    char name[64];
    int domain_size;
    float* belief;
    int* clique_ids;
    int n_cliques;
} MrfNode;

typedef struct {
    MrfNode* nodes;
    int n_nodes;
    int max_nodes;
    Clique* cliques;
    int n_cliques;
    int max_cliques;
    int* edges;
    int edge_count;
    int* edge_pairs;
    float* pairwise_potentials;
} MarkovRandomField;

int mrf_create(MarkovRandomField* mrf, int max_nodes, int max_cliques);
void mrf_free(MarkovRandomField* mrf);
int mrf_add_node(MarkovRandomField* mrf, int id, const char* name, int domain_size);
int mrf_add_edge(MarkovRandomField* mrf, int node_i, int node_j, const float* potentials);
int mrf_add_clique(MarkovRandomField* mrf, const int* node_ids, int n_nodes, const float* potential);
int mrf_gibbs_sample(const MarkovRandomField* mrf, int* assignment, int n_burnin, int n_samples);
float mrf_infer_marginal(const MarkovRandomField* mrf, int node_id, int n_burnin, int n_samples, float* marginal);
int mrf_loopy_bp(MarkovRandomField* mrf, int max_iter, float tol);

#ifdef __cplusplus
}
#endif

#endif
