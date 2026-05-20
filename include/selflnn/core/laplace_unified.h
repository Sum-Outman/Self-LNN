/**
 * @file laplace_unified.h
 * @brief 拉普拉斯变换统一接口（DUP-005合并）
 *
 * 统一合并以下三个模块的所有API：
 * - laplace_ai_framework.h — 拉普拉斯AI框架（频谱变换/特征提取/RL集成）
 * - laplace_enhanced.h   — 拉普拉斯增强系统（频域增强/稳定性/管道/监控/LNN保证器）
 * - laplace_integration.h — 拉普拉斯深度集成（CfC稳定性/频谱自适应/分数阶记忆/RLS/PID/门控）
 *
 * 所有原有函数签名和类型定义保持不变，完全向后兼容。
 */

#ifndef SELFLNN_LAPLACE_UNIFIED_H
#define SELFLNN_LAPLACE_UNIFIED_H

#include "selflnn/core/common.h"
#include "selflnn/core/laplace.h"
#include "selflnn/learning/reinforcement_learning.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * 第一部分：合并自 laplace_ai_framework.h — 拉普拉斯AI框架
 * =================================================================== */

#define LAI_MAX_CHANNELS 64
#define LAI_MAX_FILTER_BANDS 32
#define LAI_SPECTRUM_SIZE 1024

typedef enum {
    LAI_FILTER_NONE = 0,
    LAI_FILTER_LOWPASS,
    LAI_FILTER_HIGHPASS,
    LAI_FILTER_BANDPASS,
    LAI_FILTER_BANDSTOP,
    LAI_FILTER_NOTCH,
    LAI_FILTER_ADAPTIVE
} LAIFilterType;

typedef enum {
    LAI_WINDOW_NONE = 0,
    LAI_WINDOW_HANNING,
    LAI_WINDOW_HAMMING,
    LAI_WINDOW_BLACKMAN,
    LAI_WINDOW_KAISER
} LAIWindowType;

typedef struct {
    size_t fft_size;
    float sampling_rate;
    float low_cutoff_hz;
    float high_cutoff_hz;
    int filter_order;
    LAIFilterType filter_type;
    LAIWindowType window_type;
    float kaiser_beta;
    int enable_adaptive_filter;
    int enable_stability_monitor;
    int enable_feature_extraction;
    float feature_freq_bands[16];
    int num_feature_bands;
    float lr_adaptation_speed;
    float lr_min_scale;
    float lr_max_scale;
    size_t history_size;
} LaplaceAIConfig;

#define LAPLACE_AI_CONFIG_DEFAULT_INITIALIZER { \
    256, 100.0f, 0.5f, 45.0f, 4, LAI_FILTER_ADAPTIVE, \
    LAI_WINDOW_HANNING, 2.0f, 1, 1, 1, \
    {0.5f, 2.0f, 5.0f, 10.0f, 20.0f}, 5, \
    0.1f, 0.1f, 3.0f, 512 \
}

typedef struct LaplaceAI LaplaceAI;

typedef struct {
    float* magnitude;
    float* phase;
    float* frequency;
    size_t num_bins;
    float sampling_rate;
    float dominant_freq;
    float spectral_centroid;
    float spectral_rolloff;
    float spectral_flux;
    float total_energy;
    float band_energies[16];
    int num_bands;
    double processing_time_ms;
} LAISpectrumResult;

typedef struct {
    LAIFilterType type;
    float cutoff_low;
    float cutoff_high;
    int order;
    float* numerator;
    float* denominator;
    size_t num_coeffs;
    float ripple_db;
    float stopband_atten_db;
} LAIFilterCoeffs;

typedef struct {
    float stability_score;
    float phase_margin;
    float gain_margin;
    float dominant_pole_real;
    float noise_level;
    float oscillation_index;
    int is_stable;
    int warning_level;
    char description[128];
} LAIStabilityReport;

typedef struct {
    float* time_domain;
    float* frequency_domain_real;
    float* frequency_domain_imag;
    float* magnitude_spectrum;
    float* phase_spectrum;
    size_t signal_length;
    size_t spectrum_size;
    double transform_time_ms;
} LAITransformBuffer;

/* ---- LAI系统API ---- */
SELFLNN_API LaplaceAI* laplace_ai_create(const LaplaceAIConfig* config);
SELFLNN_API void laplace_ai_free(LaplaceAI* ai);

SELFLNN_API int laplace_ai_transform(LaplaceAI* ai, const float* input,
                                      size_t input_length, LAITransformBuffer* buffer);
SELFLNN_API int laplace_ai_inverse_transform(LaplaceAI* ai, const LAITransformBuffer* buffer,
                                              float* output, size_t output_length);
SELFLNN_API void laplace_ai_transform_buffer_free(LAITransformBuffer* buffer);

SELFLNN_API int laplace_ai_extract_features(LaplaceAI* ai, const float* signal,
                                              size_t signal_length, LAISpectrumResult* result);
SELFLNN_API void laplace_ai_spectrum_free(LAISpectrumResult* result);

SELFLNN_API int laplace_ai_filter_signal(LaplaceAI* ai, const float* input,
                                          size_t input_length, float* output,
                                          LAIFilterType filter_type,
                                          float cutoff_low, float cutoff_high);
SELFLNN_API int laplace_ai_filter_gradient(LaplaceAI* ai, const float* gradients,
                                            size_t num_gradients, float* filtered_gradients);

SELFLNN_API float laplace_ai_adaptive_learning_rate(LaplaceAI* ai,
                                                      const float* gradient_history,
                                                      size_t history_length,
                                                      float base_lr);

SELFLNN_API int laplace_ai_monitor_stability(LaplaceAI* ai, const float* signal,
                                              size_t signal_length,
                                              LAIStabilityReport* report);

SELFLNN_API int laplace_ai_frequency_reason(LaplaceAI* ai,
                                             const float* input_signal,
                                             size_t input_length,
                                             float* output_signal,
                                             size_t output_length,
                                             const float* target_spectrum,
                                             size_t spectrum_length);
SELFLNN_API int laplace_ai_set_config(LaplaceAI* ai, const LaplaceAIConfig* config);
SELFLNN_API int laplace_ai_get_config(const LaplaceAI* ai, LaplaceAIConfig* config);
SELFLNN_API void laplace_ai_reset(LaplaceAI* ai);

/* ---- 强化学习系统API（LAI集成） ---- */
SELFLNN_API RLAgent* laplace_ai_rl_create(RLAlgorithm algorithm, int state_dim, int action_dim);
SELFLNN_API void laplace_ai_rl_destroy(RLAgent* agent);
SELFLNN_API void laplace_ai_rl_destroy_all(void);
SELFLNN_API int laplace_ai_rl_select_action(RLAgent* agent, const float* state, int state_dim,
                                            float* action, int action_dim);
SELFLNN_API int laplace_ai_rl_store_experience(RLAgent* agent, const float* state, int state_dim,
                                               const float* action, int action_dim, float reward,
                                               const float* next_state, int next_state_dim, int done);
SELFLNN_API int laplace_ai_rl_train(RLAgent* agent, int batch_size);
SELFLNN_API int laplace_ai_rl_save(RLAgent* agent, const char* filepath);
SELFLNN_API RLAgent* laplace_ai_rl_load(const char* filepath);
SELFLNN_API int laplace_ai_rl_update_exploration(RLAgent* agent);
SELFLNN_API int laplace_ai_rl_reset(RLAgent* agent);
SELFLNN_API float laplace_ai_rl_get_exploration_rate(const RLAgent* agent);
SELFLNN_API int laplace_ai_rl_set_exploration_rate(RLAgent* agent, float rate);
SELFLNN_API int laplace_ai_rl_get_stats(const RLAgent* agent, int* total_steps,
                                        int* total_episodes, float* avg_return, float* best_return);
SELFLNN_API int laplace_ai_rl_get_q_values(RLAgent* agent, const float* state,
                                           int state_dim, float* q_values);

/* ===================================================================
 * 第二部分：合并自 laplace_enhanced.h — 拉普拉斯全面增强系统
 * =================================================================== */

/* 拉普拉斯算子类型 */
typedef enum {
    LAPLACE_OP_STANDARD = 0,
    LAPLACE_OP_GAUSSIAN = 1,
    LAPLACE_OP_FRACTIONAL = 2,
    LAPLACE_OP_ANISOTROPIC = 3,
    LAPLACE_OP_SPECTRAL = 4
} LaplaceOperatorType;

/* 增强目标模块 */
typedef enum {
    LAPLACE_TARGET_TRAINING = 0,
    LAPLACE_TARGET_REASONING = 1,
    LAPLACE_TARGET_VISION = 2,
    LAPLACE_TARGET_AUDIO = 3,
    LAPLACE_TARGET_CONTROL = 4,
    LAPLACE_TARGET_MEMORY = 5,
    LAPLACE_TARGET_ALL = 6
} LaplaceTarget;

/* 滤波配置 */
typedef struct {
    LaplaceOperatorType op_type;
    float cutoff_frequency;
    float filter_order;
    float smoothing_alpha;
    float noise_suppression;
    int use_adaptive_cutoff;
    int use_zero_phase;
    int kernel_size;
} LaplaceFilterConfig;

/* 频域特征 */
typedef struct {
    float* magnitude_spectrum;
    float* phase_spectrum;
    size_t spectrum_size;
    float dominant_frequency;
    float spectral_entropy;
    float spectral_centroid;
    float spectral_bandwidth;
    float spectral_flatness;
    float high_freq_energy_ratio;
    float stability_metric;
} LaplaceSpectralFeatures;

/* 系统稳定性分析 */
typedef struct {
    float damping_ratio;
    float natural_frequency;
    float gain_margin;
    float phase_margin;
    float settling_time;
    float overshoot;
    int is_stable;
    float stability_reserve;
} EnhancedStabilityAnalysis;

/* 拉普拉斯增强系统句柄 */
typedef struct LaplaceEnhancedSystem LaplaceEnhancedSystem;

/* ---- 增强系统API ---- */
LaplaceEnhancedSystem* laplace_enhanced_create(LaplaceTarget target);
void laplace_enhanced_free(LaplaceEnhancedSystem* system);
int laplace_set_filter_config(LaplaceEnhancedSystem* system, const LaplaceFilterConfig* config);
int laplace_spectral_analyze(LaplaceEnhancedSystem* system, const float* signal,
                            size_t signal_size, LaplaceSpectralFeatures* features);
int laplace_spectral_filter(LaplaceEnhancedSystem* system, const float* input,
                           float* output, size_t signal_size);
int laplace_enhance_gradients(LaplaceEnhancedSystem* system, float* gradients,
                             size_t grad_size, float learning_rate);
int laplace_denoise(LaplaceEnhancedSystem* system, const float* input,
                   float* output, size_t signal_size);
int laplace_stability_analyze(LaplaceEnhancedSystem* system, const float* system_matrix,
                             int matrix_size, EnhancedStabilityAnalysis* analysis);
int laplace_stabilize_control(LaplaceEnhancedSystem* system, const float* control_signal,
                             float* stabilized_signal, size_t signal_size);
int laplace_image_enhance(LaplaceEnhancedSystem* system, const float* image,
                         float* enhanced, int width, int height, int channels);
int laplace_memory_decay_model(LaplaceEnhancedSystem* system, float* memory_strength,
                              size_t memory_count, float time_elapsed);

/* ---- S06: 全系统拉普拉斯分析管道 ---- */

typedef enum {
    LAPLACE_PIPELINE_STAGE_FFT,
    LAPLACE_PIPELINE_STAGE_SPECTRAL,
    LAPLACE_PIPELINE_STAGE_STABILITY,
    LAPLACE_PIPELINE_STAGE_FILTERING,
    LAPLACE_PIPELINE_STAGE_CONTROL,
    LAPLACE_PIPELINE_STAGE_COUNT
} LaplacePipelineStage;

typedef struct {
    LaplacePipelineStage stage;
    int success;
    float execution_time_ms;
    float stage_metrics[10];
    int metric_count;
} PipelineStageResult;

typedef struct {
    LaplaceSpectralFeatures spectral;
    EnhancedStabilityAnalysis stability;
    float control_recommendation;
    float overall_health_score;
    int alarm_level;
    PipelineStageResult stage_results[LAPLACE_PIPELINE_STAGE_COUNT];
    int stage_count;
} PipelineResult;

int laplace_pipeline_execute(LaplaceEnhancedSystem* system,
                              const float* time_domain_signal,
                              size_t signal_size, float sampling_rate,
                              PipelineResult* result);

/* ---- S06: 实时稳定性监控 ---- */

typedef struct {
    float* history_buffer;
    size_t buffer_size;
    size_t write_pos;
    float* spectrum_accumulator;
    size_t accumulation_count;
    float stability_threshold;
    float anomaly_threshold;
    int consecutive_anomalies;
    float* feature_history;
    size_t feature_history_size;
    size_t feature_write_pos;
    int is_initialized;
} RealtimeMonitor;

RealtimeMonitor* laplace_monitor_create(size_t buffer_size,
                                         size_t feature_history_size);
void laplace_monitor_free(RealtimeMonitor* monitor);
int laplace_monitor_update(RealtimeMonitor* monitor, float new_sample,
                            EnhancedStabilityAnalysis* result);
int laplace_monitor_get_statistics(RealtimeMonitor* monitor,
                                    float* mean, float* variance,
                                    float* peak_frequency);

/* ---- S06: LNN稳定性保证器 ---- */

typedef struct {
    float max_eigenvalue_real;
    float correction_strength;
    float damping_factor;
    int enable_automatic_correction;
    int enable_adaptive_damping;
    int max_correction_iterations;
} LNNStabilityConfig;

typedef struct {
    LaplaceEnhancedSystem* parent_system;
    LNNStabilityConfig config;
    float* eigenvalue_real_parts;
    float* eigenvalue_imag_parts;
    size_t eigenvalue_count;
    float max_real_part;
    int correction_count;
    float* correction_history;
    size_t correction_history_size;
    size_t correction_write_pos;
} LNNStabilityGuarantor;

LNNStabilityGuarantor* laplace_guarantor_create(LaplaceEnhancedSystem* parent,
                                                  const LNNStabilityConfig* config,
                                                  size_t eigenvalue_history_size);
void laplace_guarantor_free(LNNStabilityGuarantor* guarantor);
int laplace_guarantor_check_and_correct(LNNStabilityGuarantor* guarantor,
                                          const float* state_matrix,
                                          size_t matrix_size,
                                          int* correction_applied);
int laplace_guarantor_get_report(LNNStabilityGuarantor* guarantor,
                                  float* max_eigenvalue,
                                  float* eigenvalue_spread,
                                  int* total_corrections,
                                  float* current_stability_margin);

/* ===================================================================
 * 第三部分：合并自 laplace_integration.h — 拉普拉斯深度集成
 * =================================================================== */

/* ---- CfC单元稳定性分析 ---- */

typedef struct {
    float stability_score;
    float phase_margin;
    float gain_margin;
    float natural_frequency;
    float damping_ratio;
    float dominant_pole_real;
    float dominant_pole_imag;
    int is_stable;
    int stability_warning;
} CfcStabilityAnalysis;

int cfc_cell_analyze_stability(const void* cell, CfcStabilityAnalysis* analysis);
int cfc_cell_optimize_stability(void* cell, float target_phase_margin, float target_gain_margin);
int cfc_cell_frequency_response(const void* cell,
                                const float* frequencies,
                                size_t num_frequencies,
                                float* magnitude_response,
                                float* phase_response);
int cfc_cell_design_lowpass_filter(void* cell, float cutoff_frequency, int filter_order);
int cfc_cell_system_identification(const void* cell,
                                   const float* input_signal,
                                   const float* output_signal,
                                   size_t num_samples,
                                   float sampling_rate,
                                   float* identified_params);

/* ---- 频域自适应学习率与频谱分析 ---- */

typedef struct {
    size_t fft_size;
    float sampling_rate;
    float min_frequency;
    float max_frequency;
    int enable_window;
    int enable_log_scale;
} SpectrumConfig;

typedef struct {
    float frequency;
    float magnitude;
    float phase;
} SpectrumPoint;

typedef struct {
    SpectrumPoint* points;
    size_t num_points;
    float dominant_frequency;
    float dominant_magnitude;
    float dc_component;
    float total_power;
    float high_freq_ratio;
    float low_freq_ratio;
    float spectral_centroid;
    float bandwidth_3db;
} SpectrumResult;

typedef struct {
    float base_learning_rate;
    float min_learning_rate;
    float max_learning_rate;
    float high_freq_threshold;
    float low_freq_threshold;
    float adaptation_speed;
    int use_spectral_centroid;
    float momentum;
} FreqAdaptiveLRConfig;

SpectrumConfig laplace_spectrum_config_default(void);
FreqAdaptiveLRConfig laplace_freq_adaptive_lr_config_default(float base_lr);
int laplace_compute_spectrum(const float* signal, size_t signal_length,
                              const SpectrumConfig* config, SpectrumResult* result);
float laplace_freq_adaptive_lr(const float* gradient_history, size_t history_length,
                                const FreqAdaptiveLRConfig* config,
                                const SpectrumResult* spectrum);
int laplace_spectrum_to_json(const SpectrumResult* result,
                              char* buffer, size_t buffer_size);
void laplace_spectrum_result_free(SpectrumResult* result);

/* ---- 拉普拉斯深度集成扩展 ---- */

void* lnn_laplace_create_default_analyzer(void);
int lnn_laplace_modulate_hidden(void* analyzer, float* hidden_state,
                                 size_t hidden_size, float modulation_strength);
int lnn_laplace_analyze_network_dynamics(void* analyzer,
                                          float time_constant,
                                          const float* hidden_state,
                                          size_t hidden_size,
                                          float* stability_score,
                                          float* recommended_cutoff,
                                          float* frequency_bandwidth);

/* ---- 分数阶拉普拉斯记忆模块 ---- */

typedef struct {
    float fractional_order;
    float memory_decay;
    size_t memory_length;
    int use_caputo_derivative;
    int enable_adaptive_order;
    float order_adaptation_rate;
    float min_fractional_order;
    float max_fractional_order;
} FractionalMemoryConfig;

FractionalMemoryConfig laplace_fractional_memory_config_default(void);
int laplace_fractional_integral(const float* signal, size_t length,
                                 float order, float dt, float* output);
int laplace_fractional_derivative(const float* signal, size_t length,
                                   float order, float dt, float* output);
int laplace_fractional_memory_filter(float* hidden_state, size_t hidden_size,
                                      const FractionalMemoryConfig* config,
                                      float dt, float* buffer);
int laplace_learn_fractional_order(const float* hidden_state_history,
                                    size_t hidden_size, size_t history_length,
                                    float* current_order,
                                    const FractionalMemoryConfig* config);

/* ---- 递归最小二乘在线系统辨识 ---- */

typedef struct {
    float theta[4];
    float P[16];
    float forgetting_factor;
    float delta;
    int model_order;
    float lambda;
    float phi_buffer[4];
    int is_initialized;
} RLSEstimator;

int laplace_rls_init(RLSEstimator* estimator, float forgetting_factor,
                     int model_order, float delta);
int laplace_rls_update(RLSEstimator* estimator, float y, float u);
int laplace_rls_get_continuous_params(const RLSEstimator* estimator,
                                       float dt, float* time_constant,
                                       float* gain, float* natural_freq,
                                       float* damping_ratio);

/* ---- 拉普拉斯域PID自动调谐 ---- */

typedef struct {
    float kp;
    float ki;
    float kd;
    float setpoint_weight;
    float derivative_filter;
    float output_limit_min;
    float output_limit_max;
    float integral_limit_min;
    float integral_limit_max;
} PIDParams;

int laplace_pid_auto_tune(float time_constant, float gain,
                           float dead_time, int controller_type,
                           PIDParams* params);
int laplace_lead_lag_design(float phase_margin_target,
                             float crossover_freq, int is_lead,
                             float lead_ratio,
                             float* zero_time, float* pole_time,
                             float* compensator_gain);

/* ---- 自适应频谱门控 ---- */

typedef struct {
    float noise_floor;
    float gate_threshold;
    float attack_time;
    float release_time;
    float frequency_smoothing;
    int use_wiener_filter;
    float wiener_alpha;
    int enable_noise_estimation;
    float noise_estimation_rate;
    float min_gain;
    float max_gain;
    float bandwidth_hz;
    float center_freq_hz;
} SpectralGateConfig;

SpectralGateConfig laplace_spectral_gate_config_default(void);
int laplace_spectral_gate(const float* input_signal, float* output_signal,
                           size_t signal_length,
                           const SpectralGateConfig* config,
                           const float* noise_spectrum);
void laplace_update_noise_spectrum(float* noise_spectrum,
                                    const float* current_spectrum,
                                    size_t num_bins, float update_rate);

/* ---- 多变量拉普拉斯稳定性（全LNN分析） ---- */

typedef struct {
    size_t num_cells;
    float* pole_real_parts;
    float* pole_imag_parts;
    int* pole_stability;
    float* cell_bandwidths;
    float overall_stability;
    float worst_case_margin;
    float system_bandwidth;
    int num_poles;
    float nyquist_stability;
    float characteristic_loci_margin;
} LNNStabilityResult;

int laplace_lnn_full_stability(const float* time_constants,
                                const float* feedback_strengths,
                                const float* coupling_matrix,
                                size_t num_cells,
                                LNNStabilityResult* result);
void laplace_lnn_stability_result_free(LNNStabilityResult* result);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LAPLACE_UNIFIED_H */
