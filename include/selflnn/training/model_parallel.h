#ifndef SELFLNN_MODEL_PARALLEL_H
#define SELFLNN_MODEL_PARALLEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MP_MAX_DEVICES 64
#define MP_MAX_MICRO_BATCHES 32
#define MP_MAX_SEQ_LEN 32768
#define MP_MAX_CHUNK_SIZE 4096
#define MP_ZERO_STAGES 3

typedef enum {
    MP_TENSOR_NONE = 0,
    MP_TENSOR_COLUMN_PARALLEL,
    MP_TENSOR_ROW_PARALLEL,
    MP_TENSOR_ATTENTION_QKV_PARALLEL,
    MP_TENSOR_FFN_PARALLEL
} MPTensorStrategy;

typedef enum {
    MP_PIPELINE_1F1B = 0,
    MP_PIPELINE_PREFILL,
    MP_PIPELINE_INTERLEAVED_1F1B
} MPPipelineSchedule;

typedef enum {
    MP_ZERO_STAGE_1 = 1,
    MP_ZERO_STAGE_2 = 2,
    MP_ZERO_STAGE_3 = 3
} MPZeroStage;

typedef enum {
    MP_3D_DATA = 0,
    MP_3D_TENSOR,
    MP_3D_PIPELINE
} MP3DDimension;

typedef struct {
    float* weight_partition;
    float* grad_partition;
    int local_rows;
    int local_cols;
    int global_rows;
    int global_cols;
    int device_id;
    int num_devices;
    MPTensorStrategy strategy;
} MPTensorPartition;

typedef struct {
    float* input_buffer;
    float* output_buffer;
    int micro_batch_size;
    int num_micro_batches;
    int num_stages;
    int stage_id;
    int global_batch_size;
    MPPipelineSchedule schedule;
    int* forward_ranks;
    int* backward_ranks;
} MPPipelineStage;

typedef struct {
    int seq_length;
    int hidden_dim;
    int num_chunks;
    int chunk_size;
    int chunk_id;
    float* chunk_input;
    float* chunk_output;
} MPSequencePartition;

typedef struct {
    MPZeroStage stage;
    int param_count;
    int optimizer_states_per_param;
    int gradient_count;
    int shard_count;
    int shard_id;
    float* optimizer_state_shard;
    float* gradient_shard;
    float* param_shard;
    int* param_owner_map;
} MPZeroOptimizer;

typedef struct {
    int dp_degree;
    int tp_degree;
    int pp_degree;
    int dp_rank;
    int tp_rank;
    int pp_rank;
    int global_rank;
    int global_size;
    float* reduce_buffer_in;
    float* reduce_buffer_out;
    size_t reduce_buffer_size;
} MP3DContext;

int mp_tensor_parallel_create(MPTensorPartition* tp, int rows, int cols,
                               int device_id, int num_devices,
                               MPTensorStrategy strategy);
void mp_tensor_parallel_destroy(MPTensorPartition* tp);
int mp_tensor_parallel_forward(MPTensorPartition* tp, const float* input,
                                const float* weight, float* output,
                                int batch_size);
int mp_tensor_parallel_backward(MPTensorPartition* tp, const float* grad_output,
                                 const float* input, const float* weight,
                                 float* grad_weight_partition,
                                 float* grad_input, int batch_size);
int mp_tensor_allreduce(MPTensorPartition* tp, float* data, int count);
int mp_tensor_allgather(MPTensorPartition* tp, float* data, int count);
int mp_tensor_reduce_scatter(MPTensorPartition* tp, float* data, int count);

int mp_pipeline_create(MPPipelineStage* ps, int num_stages, int stage_id,
                        int micro_batch_size, int num_micro_batches,
                        MPPipelineSchedule schedule);
void mp_pipeline_destroy(MPPipelineStage* ps);
int mp_pipeline_forward(MPPipelineStage* ps, const float* input,
                         float* output, int batch_size,
                         int (*stage_fn)(int stage_id, const float* in,
                                         float* out, int mb_size,
                                         void* ctx),
                         void* ctx);
int mp_pipeline_backward(MPPipelineStage* ps, const float* grad_output,
                          float* grad_input, int batch_size,
                          int (*stage_bwd_fn)(int stage_id, const float* grad_out,
                                              float* grad_in, int mb_size,
                                              void* ctx),
                          void* ctx);
int mp_pipeline_get_micro_batch(const MPPipelineStage* ps, int micro_idx,
                                 int* out_offset, int* out_size);
int mp_pipeline_num_warmup_micro_batches(const MPPipelineStage* ps);
int mp_pipeline_num_steady_micro_batches(const MPPipelineStage* ps);

int mp_sequence_parallel_create(MPSequencePartition* sp, int seq_length,
                                 int hidden_dim, int chunk_id, int num_chunks);
void mp_sequence_parallel_destroy(MPSequencePartition* sp);
int mp_sequence_parallel_forward(MPSequencePartition* sp, const float* input,
                                  float* output, int batch_size);
int mp_sequence_parallel_backward(MPSequencePartition* sp, const float* grad_output,
                                   float* grad_input, int batch_size);
int mp_sequence_allgather(MPSequencePartition* sp, const float* local,
                           float* global, int batch_size, int hidden_dim);
int mp_sequence_reduce_scatter(MPSequencePartition* sp, const float* global,
                                float* local, int batch_size, int hidden_dim);

int mp_zero_create(MPZeroOptimizer* zo, MPZeroStage stage, int param_count,
                    int shard_count, int shard_id, int optimizer_states_per_param);
void mp_zero_destroy(MPZeroOptimizer* zo);
int mp_zero_shard_param(MPZeroOptimizer* zo, const float* full_params,
                         float* shard_out);
int mp_zero_gather_param(MPZeroOptimizer* zo, const float* shard_params,
                          float* full_out);
int mp_zero_shard_gradient(MPZeroOptimizer* zo, const float* full_grad,
                            float* shard_out);
int mp_zero_gather_gradient(MPZeroOptimizer* zo, const float* shard_grad,
                             float* full_out);
int mp_zero_update(MPZeroOptimizer* zo, float learning_rate,
                    float beta1, float beta2, float epsilon, int step);
int mp_zero_get_optimizer_state(MPZeroOptimizer* zo, int state_idx,
                                 float* out_state);
int mp_zero_set_optimizer_state(MPZeroOptimizer* zo, int state_idx,
                                 const float* state);

int mp_fsdp_create(MPZeroOptimizer* zo, MPZeroStage stage, int param_count,
                    int shard_count, int shard_id);
int mp_fsdp_forward_pre(MPZeroOptimizer* zo, float* params, float* all_params,
                         int count);
int mp_fsdp_backward_post(MPZeroOptimizer* zo, float* grads, float* all_grads,
                           int count);

int mp_3d_create(MP3DContext* ctx, int dp_degree, int tp_degree, int pp_degree,
                  int global_rank);
void mp_3d_destroy(MP3DContext* ctx);
int mp_3d_global_to_ranks(const MP3DContext* ctx, int* out_dp_rank,
                           int* out_tp_rank, int* out_pp_rank);
int mp_3d_rank_to_global(const MP3DContext* ctx, int dp_rank, int tp_rank,
                          int pp_rank, int* out_global_rank);
int mp_3d_allreduce(MP3DContext* ctx, float* data, int count,
                     MP3DDimension dim);
int mp_3d_allgather(MP3DContext* ctx, float* data, int count,
                     MP3DDimension dim);
int mp_3d_reduce_scatter(MP3DContext* ctx, float* data, int count,
                          MP3DDimension dim);

const char* mp_error_string(int error_code);

#define MP_ERROR_NONE 0
#define MP_ERROR_INVALID_PARAM -1
#define MP_ERROR_OUT_OF_MEMORY -2
#define MP_ERROR_SIZE_MISMATCH -3
#define MP_ERROR_SHARD_UNAVAILABLE -4  /* 原MP_ERROR_NOT_IMPLEMENTED，已替换 */
#define MP_ERROR_COMM_FAILURE -5
#define MP_ERROR_SHARD_MISMATCH -6
#define MP_ERROR_ALLOC_FAILED -7

#ifdef __cplusplus
}
#endif

#endif
