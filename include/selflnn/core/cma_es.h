#ifndef SELFLNN_CMA_ES_H
#define SELFLNN_CMA_ES_H

#include "selflnn/core/common.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CMAES_MAX_DIM 4096
#define CMAES_MAX_POP 1024
#define CMAES_DEFAULT_LAMBDA 100
#define CMAES_MIN_POP 4
#define CMAES_DEFAULT_STOP_FITNESS 1e-12f
#define CMAES_DEFAULT_MAX_GEN 1000
#define CMAES_DEFAULT_TOL_COV 1e-14f
#define CMAES_DEFAULT_TOL_X 1e-12f
#define CMAES_DEFAULT_TOL_FUN 1e-12f
#define CMAES_DEFAULT_TOL_HIST_FUN 1e-12f
#define CMAES_DEFAULT_SIGMA 0.3f

typedef enum {
    CMAES_TERM_NONE = 0,
    CMAES_TERM_MAXGEN,
    CMAES_TERM_TOLFUN,
    CMAES_TERM_TOLX,
    CMAES_TERM_TOLCOV,
    CMAES_TERM_TOLHISTFUN,
    CMAES_TERM_CONDITION,
    CMAES_TERM_USER
} CMAESTerminationReason;

typedef struct {
    float* mean;
    float sigma;
    float* covariance;
    float* evolution_path_sigma;
    float* evolution_path_cov;
    float* temp_vec;
    float* temp_vec2;
    float* weights;
    float* sample_pop;
    float* fitness;
    int* index_order;
    float mueff;
    float cc;
    float cs;
    float c1;
    float cmu;
    float ds;
    float chi_n;
    float eigen_value_sum;
    float* eigen_values;
    float* eigen_vectors;
    float* inv_sqrt_cov;
    size_t dimension;
    int lambda;
    int mu;
    int generation;
    int max_generations;
    float stop_fitness;
    float tol_x;
    float tol_cov;
    float tol_fun;
    float tol_hist_fun;
    float best_fitness;
    float best_fitness_history;
    float* best_solution;
    float* lower_bounds;
    float* upper_bounds;
    int use_boundary;
    CMAESTerminationReason termination_reason;
    int verbose;
    unsigned int rng_state;
    int enable_active;
    int ipop_restart_count;
} CMAESState;

typedef float (*CMAESFitnessFunction)(const float* x, size_t dim, void* user_data);

CMAESState* cmaes_alloc(size_t dimension, float sigma, int lambda, int seed);
void cmaes_free(CMAESState* state);
int cmaes_init(CMAESState* state, size_t dimension, float sigma, int lambda, int seed);
void cmaes_set_bounds(CMAESState* state, const float* lower, const float* upper);
void cmaes_set_stop_conditions(CMAESState* state, float stop_fitness, int max_generations,
                                float tol_x, float tol_cov, float tol_fun, float tol_hist_fun);
void cmaes_sample_population(CMAESState* state);
int cmaes_update(CMAESState* state, const float* fitness_values);
float* cmaes_get_solution(const CMAESState* state, int index);
int cmaes_get_population_size(const CMAESState* state);
int cmaes_get_generation(const CMAESState* state);
void cmaes_get_best_solution(const CMAESState* state, float* out_x, float* out_fitness);
float cmaes_get_current_sigma(const CMAESState* state);
int cmaes_test_stop_conditions(CMAESState* state);
void cmaes_reset(CMAESState* state);
int cmaes_optimize(CMAESState* state, CMAESFitnessFunction func, void* user_data);

/* === IPOP (增量种群) 重启策略 === */
#define CMAES_IPOP_MAX_RESTARTS 10
#define CMAES_IPOP_POP_MULTIPLIER 2.0f

int cmaes_ipop_optimize(CMAESState* state, CMAESFitnessFunction func, void* user_data,
                         int max_restarts);
void cmaes_enable_active_cma(CMAESState* state, int enable);

#ifdef __cplusplus
}
#endif

#endif
