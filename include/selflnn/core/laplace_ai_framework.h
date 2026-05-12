#ifndef SELFLNN_LAPLACE_AI_FRAMEWORK_H
#define SELFLNN_LAPLACE_AI_FRAMEWORK_H

#include "selflnn/core/common.h"
#include "selflnn/core/laplace.h"
#include "selflnn/learning/reinforcement_learning.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

/* ===== 强化学习系统 API ===== */
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

#ifdef __cplusplus
}
#endif

#endif
