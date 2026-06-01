#ifndef SELFLNN_CFC_ENHANCED_H
#define SELFLNN_CFC_ENHANCED_H

/* DUP-004: cfc_enhanced是cfc_cell的上层增强封装，无功能重复。
 * 本模块提供：自动求解器选择、刚度检测、多速率、并行RHS/SIMD加速。
 * cfc_cell提供：CfC单元核心创建/前向/反向/ODE求解器。
 * 两者为互补层级关系，cfc_enhanced依赖cfc_cell。 */
#define SELFLNN_CFC_ENHANCED_WRAPPER 1

#include <stdint.h>
#include "selflnn/core/cfc_cell.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CFC_ENHANCED_MAX_NEWTON_ITER 50
#define CFC_ENHANCED_DEFAULT_STIFFNESS_THRESHOLD 100.0f
#define CFC_ENHANCED_DEFAULT_STIFFNESS_CHECK_INTERVAL 100
#define CFC_ENHANCED_MIN_PARALLEL_SIZE 64
#define CFC_ENHANCED_POWER_ITER_EPS 1e-8f
#define CFC_ENHANCED_MAX_POWER_ITER 100

typedef enum {
    CFC_SOLVER_SELECT_FIXED = 0,
    CFC_SOLVER_SELECT_AUTO_STIFFNESS = 1,
    CFC_SOLVER_SELECT_HYBRID = 2,
    CFC_SOLVER_SELECT_PERFORMANCE = 3
} CfcSolverSelectStrategy;

typedef struct {
    int enable_multi_rate;
    float fast_ratio;
    float slow_ratio;
    int enable_adaptive_ratio;
    float min_fast_ratio;
    float max_slow_ratio;
    int enable_stiffness_adaptive;
    float stiffness_threshold;
} CfcMultiRateConfig;

typedef struct {
    CfcSolverSelectStrategy strategy;
    int enable_stiffness_detection;
    int stiffness_check_interval;
    float stiffness_ratio_threshold;
    int enable_solver_switching;
    int preferred_stiff_solver;
    int preferred_nonstiff_solver;
    int fallback_solver;
    CfcMultiRateConfig multi_rate;
} CfcAutoSolverConfig;

typedef struct {
    int enable_parallel_rhs;
    int enable_domain_parallel;
    int parallel_rhs_min_size;
    int num_parallel_domains;
    int enable_simd_optimization;
    int enable_dynamic_load_balance;
} CfcParallelEnhanceConfig;

typedef struct {
    int enable_auto_solver;
    CfcAutoSolverConfig auto_solver;
    int enable_parallel_enhance;
    CfcParallelEnhanceConfig parallel_enhance;
    int verbose;
} CfcEnhancedConfig;

typedef struct {
    int current_solver;
    int original_solver;
    int last_stiffness_solver;
    int stiffness_check_counter;
    float current_stiffness_ratio;
    float peak_stiffness_ratio;
    float avg_stiffness_ratio;
    int stiffness_samples;
    int parallel_rhs_enabled;
    int total_calls;
    int solver_switches;
    int stiffness_detected_count;
    int multi_rate_active;
    int saved_use_multi_timescale;  /**< P3-004修复: 保存原始多时间尺度标志 */
    float saved_fast_tau_ratio;     /**< P3-004修复: 保存原始快速时间常数比例 */
    float saved_slow_tau_ratio;     /**< P3-004修复: 保存原始慢速时间常数比例 */
    CfcEnhancedConfig active_config; /**< L-004修复: 存储当前活跃配置，供cfc_get_enhanced_config查询 */
    float* power_iter_buffer;
    float* power_iter_buffer2;
    int power_iter_buffer_size;
    int initialized;
    /* P3-002: 刚度估计结果缓存 */
    uint32_t stiffness_cache_input_hash;    /**< 上次缓存的输入哈希 */
    uint32_t stiffness_cache_state_hash;    /**< 上次缓存的状态哈希 */
    float stiffness_cache_ratio;            /**< 缓存的刚度比 */
    int stiffness_cache_age;                /**< 缓存年龄（步数） */
} CfcEnhancedState;

CfcEnhancedConfig cfc_enhanced_default_config(void);

CfcEnhancedState* cfc_enhanced_state_create(void);

void cfc_enhanced_state_free(CfcEnhancedState* state);

int cfc_enhanced_state_reset(CfcEnhancedState* state);

float cfc_estimate_stiffness_ratio(CfCCell* cell, const float* input,
                                    const float* state, CfcEnhancedState* estate);

int cfc_select_solver_by_stiffness(CfCCell* cell, const float* input,
                                    const float* state,
                                    const CfcAutoSolverConfig* config,
                                    CfcEnhancedState* estate);

int cfc_enhanced_forward(CfCCell* cell, const float* input, float* hidden_state,
                          const CfcEnhancedConfig* config,
                          CfcEnhancedState* state);

int cfc_enhanced_forward_with_dt(CfCCell* cell, const float* input, float delta_t,
                                   float* hidden_state,
                                   const CfcEnhancedConfig* config,
                                   CfcEnhancedState* state);

int cfc_multi_rate_forward(CfCCell* cell, const float* input, float delta_t,
                            float* hidden_state,
                            const CfcMultiRateConfig* mr_config,
                            CfcEnhancedState* state);

int cfc_parallel_rhs_forward(CfCCell* cell, const float* input, float delta_t,
                              float* hidden_state,
                              const CfcParallelEnhanceConfig* pe_config,
                              const ParallelODERHSConfig* base_pcfg,
                              CfcEnhancedState* state);

int cfc_domain_parallel_forward(CfCCell* cell, const float* input, float delta_t,
                                 float* hidden_state, int solver_type,
                                 const CfcParallelEnhanceConfig* pe_config,
                                 const ParallelODERHSConfig* base_pcfg,
                                 CfcEnhancedState* state);

int cfc_get_enhanced_stats(const CfcEnhancedState* state, int* current_solver,
                            float* stiffness_ratio, int* total_calls,
                            int* solver_switches, int* stiffness_count);

int cfc_get_enhanced_config(const CfcEnhancedState* state, CfcEnhancedConfig* config);

#ifdef __cplusplus
}
#endif

#endif
