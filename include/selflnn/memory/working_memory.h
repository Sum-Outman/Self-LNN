#ifndef SELFLNN_WORKING_MEMORY_H
#define SELFLNN_WORKING_MEMORY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WM_MAX_SLOTS 16
#define WM_KEY_MAX 64
#define WM_CFC_GATE_DIM 8

typedef struct {
    size_t capacity;
    size_t feature_dim;
    float focus_decay_rate;
    float cfc_gate_threshold;
    float cfc_temperature;
    float update_gate_rate;
    int enable_cfc_gating;
    int enable_priority_evict;
    float min_focus_threshold;
    float max_age_before_evict;
    float rehearsal_boost;
    float cfc_input_weight[WM_CFC_GATE_DIM];
    float cfc_context_weight[WM_CFC_GATE_DIM];
    float cfc_bias[WM_CFC_GATE_DIM];
    int cfc_gate_dim;
    int enable_context_gating;
} WorkingMemoryConfig;

typedef struct {
    char key[WM_KEY_MAX];
    float* data;
    size_t data_size;
    float focus;
    float cfc_gate_value;
    float age;
    float rehearsal_count;
    float last_gate_input[WM_CFC_GATE_DIM];
    int is_active;
} WorkingMemorySlot;

typedef struct WorkingMemory WorkingMemory;

WorkingMemoryConfig working_memory_config_default(void);

WorkingMemory* working_memory_create(const WorkingMemoryConfig* config);
void working_memory_destroy(WorkingMemory* wm);

int working_memory_update(WorkingMemory* wm, const char* key,
                         const float* data, size_t data_size,
                         const float* gate_context, size_t context_dim);

int working_memory_retrieve(WorkingMemory* wm, const char* key,
                           float* data, size_t data_size, float* focus_out);

int working_memory_focus_on(WorkingMemory* wm, const char* key,
                           const float* gate_signal, size_t signal_dim);

int working_memory_remove(WorkingMemory* wm, const char* key);
int working_memory_clear(WorkingMemory* wm);

int working_memory_compute_cfc_gate(WorkingMemory* wm,
                                    const float* input, size_t input_dim,
                                    const float* context, size_t context_dim,
                                    float* gate_out, size_t gate_dim);

int working_memory_apply_gate_to_all(WorkingMemory* wm,
                                    const float* global_gate, size_t gate_dim);

int working_memory_decay_all(WorkingMemory* wm, float dt);

int working_memory_evict_lowest(WorkingMemory* wm);
int working_memory_evict_by_key(WorkingMemory* wm, const char* key);

int working_memory_rehearse(WorkingMemory* wm, const char* key);

int working_memory_get_slots(const WorkingMemory* wm,
                            WorkingMemorySlot* slots, size_t max_count,
                            size_t* actual_count);

int working_memory_get_config(const WorkingMemory* wm, WorkingMemoryConfig* config);
int working_memory_set_config(WorkingMemory* wm, const WorkingMemoryConfig* config);

int working_memory_get_stats(const WorkingMemory* wm,
                            size_t* active_count, float* avg_focus,
                            float* avg_gate, float* total_age);

int working_memory_get_attention_map(const WorkingMemory* wm,
                                    float* attention_out, size_t max_count,
                                    size_t* actual_count);

int working_memory_serialize(const WorkingMemory* wm,
                            unsigned char* buffer, size_t buffer_size,
                            size_t* written);

int working_memory_deserialize(WorkingMemory* wm,
                              const unsigned char* buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif
