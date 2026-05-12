#ifndef SELFLNN_UNIFIED_LNN_STATE_H
#define SELFLNN_UNIFIED_LNN_STATE_H

#include "selflnn/core/common.h"
#include "selflnn/core/cfc_cell.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UNIFIED_LNN_MAX_MODALITIES 9
#define UNIFIED_LNN_DEFAULT_STATE_DIM 512
#define UNIFIED_LNN_DEFAULT_OUTPUT_DIM 256
#define UNIFIED_LNN_MAX_RAW_DIM 4096

typedef enum {
    UNIFIED_MODALITY_VISION = 0,
    UNIFIED_MODALITY_AUDIO = 1,
    UNIFIED_MODALITY_TEXT = 2,
    UNIFIED_MODALITY_SENSOR = 3,
    UNIFIED_MODALITY_TACTILE = 4,
    UNIFIED_MODALITY_PROPRIOCEPTION = 5,
    UNIFIED_MODALITY_THERMAL = 6,
    UNIFIED_MODALITY_RADAR = 7,
    UNIFIED_MODALITY_MOTOR = 8
} UnifiedModalityType;

typedef struct {
    size_t raw_dimensions[UNIFIED_LNN_MAX_MODALITIES];
    size_t state_dimension;
    size_t output_dimension;
    float evolution_delta_t;
    float learning_rate;
    int ode_solver_type;
    int enable_online_learning;
    int enable_noise_injection;
    float noise_std;
} UnifiedLNNStateConfig;

typedef struct UnifiedLNNState UnifiedLNNState;

UnifiedLNNState* unified_lnn_state_create(const UnifiedLNNStateConfig* config);

void unified_lnn_state_free(UnifiedLNNState* state);

int unified_lnn_state_step(UnifiedLNNState* state,
                          const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES],
                          const size_t raw_sizes[UNIFIED_LNN_MAX_MODALITIES],
                          const int modality_present[UNIFIED_LNN_MAX_MODALITIES],
                          float* output, size_t max_output_size);

int unified_lnn_state_forward(UnifiedLNNState* state,
                            const float* combined_input,
                            float* hidden_state_out);

int unified_lnn_state_reset(UnifiedLNNState* state);

int unified_lnn_state_get_hidden_state(const UnifiedLNNState* state,
                                      float* hidden_out, size_t max_size);

int unified_lnn_state_set_learning_rate(UnifiedLNNState* state, float lr);

int unified_lnn_state_set_shared_lnn(UnifiedLNNState* state, void* lnn_instance);

int unified_lnn_state_get_config(const UnifiedLNNState* state,
                                UnifiedLNNStateConfig* config);

int unified_lnn_state_train_step(UnifiedLNNState* state,
                               const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES],
                               const size_t raw_sizes[UNIFIED_LNN_MAX_MODALITIES],
                               const int modality_present[UNIFIED_LNN_MAX_MODALITIES],
                               const float* target_output, size_t target_size,
                               float* loss_out);

int unified_lnn_state_save(const UnifiedLNNState* state, const char* filepath);

UnifiedLNNState* unified_lnn_state_load(const char* filepath);

UnifiedLNNStateConfig unified_lnn_state_get_default_config(void);

int unified_lnn_state_set_modality_raw_dim(UnifiedLNNState* state,
                                          UnifiedModalityType modality,
                                          size_t raw_dim);

#ifdef __cplusplus
}
#endif

#endif
