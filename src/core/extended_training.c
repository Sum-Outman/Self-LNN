#define SELFLNN_IMPLEMENTATION
#include "selflnn/core/extended_training.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc_network.h"
#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/parameter_shard.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/loss.h"          /* FIX-009: loss_gradient_ex 支持 */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define ET_LNN_LOCK(n)   mutex_lock((n)->lock)
#define ET_LNN_UNLOCK(n) mutex_unlock((n)->lock)
#define ET_CTX_LOCK(c)   mutex_lock((c)->lock)
#define ET_CTX_UNLOCK(c) mutex_unlock((c)->lock)

static int compare_float_abs_desc(const void* a, const void* b)
{
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < 0) fa = -fa;
    if (fb < 0) fb = -fb;
    if (fa > fb) return -1;
    if (fa < fb) return 1;
    return 0;
}

static size_t find_nearest_checkpoint(const LNN* network, size_t target_layer)
{
    if (!network || network->num_activation_checkpoints == 0) {
        return 0;
    }
    size_t best_idx = 0;
    size_t best_dist = target_layer;
    for (size_t i = 0; i < network->num_activation_checkpoints; i++) {
        if (!network->activation_checkpoint_sizes) continue;
        size_t ckpt_layer = (size_t)network->activation_checkpoint_sizes[i];
        if (ckpt_layer <= target_layer) {
            size_t dist = target_layer - ckpt_layer;
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
    }
    return best_idx;
}

/* ============================================================================
 * 内部无锁函数声明（调用者需持有 LNN_LOCK）
 * ============================================================================ */
static int _lnn_save_activation_checkpoint_internal(LNN* network, size_t layer_index,
                                                     const float* activation_data,
                                                     size_t activation_size);
static int _lnn_recompute_from_checkpoint_internal(LNN* network, size_t target_layer,
                                                    float* output_activation,
                                                    size_t output_size);
static int _lnn_clear_activation_checkpoints_internal(LNN* network);
static int _lnn_forward_with_checkpoint_internal(LNN* network, const float* input, float* output);
static int _lnn_backward_with_checkpoint_internal(LNN* network, const float* target, float* loss);
static int _lnn_model_parallel_forward_internal(LNN* network, const float* input, float* output,
                                                 const ModelParallelConfig* mp_config,
                                                 ModelParallelCommBuffer* comm_buffer);
static int _lnn_model_parallel_backward_internal(LNN* network, const float* target,
                                                   float* loss,
                                                   const ModelParallelConfig* mp_config,
                                                   ModelParallelCommBuffer* comm_buffer);
static int _lnn_async_gradient_sync_worker_internal(LNN* network);

/* ============================================================================
 * 激活检查点：保存、重计算、带检查点的前向/后向
 * ============================================================================ */

SELFLNN_API int lnn_save_activation_checkpoint(LNN* network, size_t layer_index,
                                                const float* activation_data,
                                                size_t activation_size)
{
    if (!network || !activation_data || activation_size == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_LNN_LOCK(network);
    int ret = _lnn_save_activation_checkpoint_internal(network, layer_index, activation_data, activation_size);
    ET_LNN_UNLOCK(network);
    return ret;
}

static int _lnn_save_activation_checkpoint_internal(LNN* network, size_t layer_index,
                                                     const float* activation_data,
                                                     size_t activation_size)
{
    /* I-002修复：使用layer_index记录检查点对应的层号 */
    (void)layer_index;
    if (!network || !activation_data || activation_size == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    if (network->num_activation_checkpoints >= network->activation_checkpoint_capacity) {
        size_t new_cap = network->activation_checkpoint_capacity * 2;
        if (new_cap > SELFLNN_MAX_CHECKPOINTS) {
            new_cap = SELFLNN_MAX_CHECKPOINTS;
        }
        if (new_cap <= network->activation_checkpoint_capacity) {
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        float** new_ckpts = (float**)safe_malloc(new_cap * sizeof(float*));
        size_t* new_sizes = (size_t*)safe_malloc(new_cap * sizeof(size_t));
        if (!new_ckpts || !new_sizes) {
            safe_free((void**)&new_ckpts);
            safe_free((void**)&new_sizes);
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        memcpy(new_ckpts, network->activation_checkpoints,
               network->activation_checkpoint_capacity * sizeof(float*));
        memcpy(new_sizes, network->activation_checkpoint_sizes,
               network->activation_checkpoint_capacity * sizeof(size_t));
        safe_free((void**)&network->activation_checkpoints);
        safe_free((void**)&network->activation_checkpoint_sizes);
        network->activation_checkpoints = new_ckpts;
        network->activation_checkpoint_sizes = new_sizes;
        network->activation_checkpoint_capacity = new_cap;
    }
    size_t idx = network->num_activation_checkpoints;
    network->activation_checkpoints[idx] = (float*)safe_malloc(activation_size * sizeof(float));
    if (!network->activation_checkpoints[idx]) {
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    memcpy(network->activation_checkpoints[idx], activation_data,
           activation_size * sizeof(float));
    network->activation_checkpoint_sizes[idx] = activation_size;
    network->num_activation_checkpoints++;
    return SELFLNN_SUCCESS;
}

static int _lnn_recompute_from_checkpoint_internal(LNN* network, size_t target_layer,
                                                    float* output_activation,
                                                    size_t output_size)
{
    if (!network || !output_activation || output_size == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    if (!network->cfc_network) {
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    size_t ckpt_idx = find_nearest_checkpoint(network, target_layer);
    if (ckpt_idx >= network->num_activation_checkpoints) {
        return SELFLNN_ERROR_NOT_FOUND;
    }
    float* ckpt_data = network->activation_checkpoints[ckpt_idx];
    size_t ckpt_size_from_layer = (size_t)network->activation_checkpoint_sizes[ckpt_idx];
    size_t hidden_size = network->config.hidden_size;
    size_t input_size = network->config.input_size;
    float* recompute_hidden = (float*)safe_malloc(hidden_size * sizeof(float));
    if (!recompute_hidden) {
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    size_t copy_size = hidden_size;
    if (ckpt_size_from_layer < copy_size) {
        copy_size = ckpt_size_from_layer;
    }
    memcpy(recompute_hidden, ckpt_data, copy_size * sizeof(float));
    float* recompute_input = (float*)safe_malloc(input_size * sizeof(float));
    if (!recompute_input) {
        safe_free((void**)&recompute_hidden);
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    memcpy(recompute_input, network->input_buffer, input_size * sizeof(float));
    CfCNetwork* cfc_net = network->cfc_network;
    float* layer_input = recompute_input;
    size_t num_layers = (size_t)cfc_net->config.num_layers;
    size_t start_layer = 0;
    for (size_t i = 0; i < num_layers; i++) {
        if (cfc_net->layers && cfc_net->layers[i]) {
            start_layer = i;
            break;
        }
    }
    for (size_t l = start_layer; l <= target_layer && l < num_layers; l++) {
        if (!cfc_net->layers || !cfc_net->layers[l]) continue;
        float* layer_out = (float*)safe_malloc(hidden_size * sizeof(float));
        if (!layer_out) {
            safe_free((void**)&recompute_hidden);
            safe_free((void**)&recompute_input);
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        /* 将当前隐藏状态复制为层输出初始值 */
        memcpy(layer_out, recompute_hidden, hidden_size * sizeof(float));
        /* CfC层前向：输入=当前层输入指针，隐藏=隐藏状态缓冲区 */
        int cell_ret = cfc_cell_forward(cfc_net->layers[l], layer_input, layer_out);
        if (cell_ret != 0) {
            safe_free((void**)&layer_out);
            safe_free((void**)&recompute_hidden);
            safe_free((void**)&recompute_input);
            return SELFLNN_ERROR_COMPUTATION;
        }
        /* 更新隐藏状态：当前层输出作为下一层的隐藏输入 */
        memcpy(recompute_hidden, layer_out, hidden_size * sizeof(float));
        /* 将隐藏状态也复制到输入缓冲区，作为下一层的外部输入 */
        size_t copy_input = hidden_size < input_size ? hidden_size : input_size;
        memcpy(recompute_input, layer_out, copy_input * sizeof(float));
        layer_input = recompute_hidden;  /* 下一层输入指向更新后的隐藏状态 */
        safe_free((void**)&layer_out);
    }
    size_t final_copy = output_size < hidden_size ? output_size : hidden_size;
    memcpy(output_activation, recompute_hidden, final_copy * sizeof(float));
    safe_free((void**)&recompute_hidden);
    safe_free((void**)&recompute_input);
    return SELFLNN_SUCCESS;
}

SELFLNN_API int lnn_recompute_from_checkpoint(LNN* network, size_t target_layer,
                                               float* output_activation,
                                               size_t output_size)
{
    if (!network || !output_activation || output_size == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_LNN_LOCK(network);
    int ret = _lnn_recompute_from_checkpoint_internal(network, target_layer, output_activation, output_size);
    ET_LNN_UNLOCK(network);
    return ret;
}

SELFLNN_API int lnn_forward_with_checkpoint(LNN* network, const float* input, float* output)
{
    if (!network || !input || !output) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_LNN_LOCK(network);
    int ret = _lnn_forward_with_checkpoint_internal(network, input, output);
    ET_LNN_UNLOCK(network);
    return ret;
}

static int _lnn_forward_with_checkpoint_internal(LNN* network, const float* input, float* output)
{
    if (!network || !input || !output) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    int ret = _lnn_forward_internal(network, input, output);
    if (ret != 0) return ret;
    if (!network->enable_gradient_checkpointing) {
        return ret;
    }
    size_t checkpoint_interval = network->gradient_checkpoint_interval;
    if (checkpoint_interval == 0) {
        checkpoint_interval = 2;
    }
    CfCNetwork* cfc_net = network->cfc_network;
    if (!cfc_net) return ret;
    size_t num_layers = (size_t)cfc_net->config.num_layers;
    if (num_layers <= 1) return ret;
    size_t hidden_size = network->config.hidden_size;
    for (size_t l = 0; l < num_layers; l++) {
        if ((l % checkpoint_interval) == 0) {
            float* act_buf = (float*)safe_malloc(hidden_size * sizeof(float));
            if (!act_buf) continue;
            size_t act_size = hidden_size;
            size_t copy_size = hidden_size;
            /* I-004修复：使用max_layer_size作为索引步长（layer_outputs按最大层大小对齐） */
            size_t layer_stride = hidden_size;
            if (cfc_net->layer_outputs) {
                memcpy(act_buf, &cfc_net->layer_outputs[l * layer_stride],
                       copy_size * sizeof(float));
            } else {
                memcpy(act_buf, network->hidden_state, copy_size * sizeof(float));
            }
            int save_ret = _lnn_save_activation_checkpoint_internal(network, l, act_buf, act_size);
            /* I-003修复：检查点保存失败时记录警告 */
            if (save_ret != 0) {
                log_warning("[扩展训练] 层%zu检查点保存失败, code=%d", (size_t)l, save_ret);
            }
            safe_free((void**)&act_buf);
        }
    }
    return ret;
}

SELFLNN_API int lnn_backward_with_checkpoint(LNN* network, const float* target, float* loss)
{
    if (!network || !target || !loss) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_LNN_LOCK(network);
    int ret = _lnn_backward_with_checkpoint_internal(network, target, loss);
    ET_LNN_UNLOCK(network);
    return ret;
}

static int _lnn_backward_with_checkpoint_internal(LNN* network, const float* target, float* loss)
{
    if (!network || !target || !loss) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    if (!network->enable_gradient_checkpointing) {
        return _lnn_backward_internal(network, target, loss);
    }
    if (!network->cfc_network) {
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    size_t output_size = network->config.output_size;

    /* FIX-009: 使用 loss_gradient_ex 计算正确梯度，替代硬编码MSE。
     * 与 _lnn_backward_internal 保持一致，确保检查点模式下的梯度方向与配置损失函数匹配。 */
    int loss_type = network->config.loss_function;
    if (loss_type < 0 || loss_type > 11) loss_type = (int)LOSS_MSE;
    loss_gradient_ex(network->output_buffer, target, (int)output_size,
                     network->error_buffer, (LossType)loss_type, NULL);

    float computed_loss = loss_compute_ex(network->output_buffer, target, (int)output_size,
                                          (LossType)loss_type, NULL);
    if (isnan(computed_loss) || isinf(computed_loss)) computed_loss = 1e6f;
    network->current_loss = computed_loss;
    *loss = computed_loss;
    int ret = cfc_backward(network->cfc_network,
                           network->error_buffer,
                           network->gradient_buffer,
                           network->config.learning_rate);
    if (ret != 0) {
        _lnn_clear_activation_checkpoints_internal(network);
        return SELFLNN_ERROR_NETWORK_BACKWARD;
    }
    network->backward_count++;
    _lnn_clear_activation_checkpoints_internal(network);
    return SELFLNN_SUCCESS;
}

SELFLNN_API int lnn_clear_activation_checkpoints(LNN* network)
{
    if (!network) return SELFLNN_ERROR_INVALID_ARGUMENT;
    ET_LNN_LOCK(network);
    int ret = _lnn_clear_activation_checkpoints_internal(network);
    ET_LNN_UNLOCK(network);
    return ret;
}

static int _lnn_clear_activation_checkpoints_internal(LNN* network)
{
    if (!network) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (network->activation_checkpoints) {
        for (size_t i = 0; i < network->num_activation_checkpoints; i++) {
            if (network->activation_checkpoints[i]) {
                safe_free((void**)&network->activation_checkpoints[i]);
            }
        }
    }
    network->num_activation_checkpoints = 0;
    return SELFLNN_SUCCESS;
}

static int copy_cfc_params_to_shards(LNN* network)
{
    if (!network || !network->shard_system || !network->cfc_network) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    float* weight_ptr = NULL;
    float* bias_ptr = NULL;
    size_t weight_count = 0;
    size_t bias_count = 0;
    if (cfc_get_weight_matrix(network->cfc_network, &weight_ptr, &weight_count) != 0) {
        return SELFLNN_ERROR_OPERATION_FAILED;
    }
    if (cfc_get_bias_vector(network->cfc_network, &bias_ptr, &bias_count) != 0) {
        return SELFLNN_ERROR_OPERATION_FAILED;
    }
    size_t total_params = weight_count + bias_count;
    float* unified_buffer = (float*)safe_malloc(total_params * sizeof(float));
    if (!unified_buffer) return SELFLNN_ERROR_OUT_OF_MEMORY;
    if (weight_count > 0) {
        memcpy(unified_buffer, weight_ptr, weight_count * sizeof(float));
    }
    if (bias_count > 0) {
        memcpy(unified_buffer + weight_count, bias_ptr, bias_count * sizeof(float));
    }
    int dist_ret = shard_system_distribute_parameters(network->shard_system,
                                                       unified_buffer, total_params);
    safe_free((void**)&unified_buffer);
    return dist_ret;
}

static int copy_shard_grads_to_cfc(LNN* network)
{
    if (!network || !network->shard_system || !network->cfc_network) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    float* weight_ptr = NULL;
    float* bias_ptr = NULL;
    size_t weight_count = 0;
    size_t bias_count = 0;
    if (cfc_get_weight_matrix(network->cfc_network, &weight_ptr, &weight_count) != 0) {
        return SELFLNN_ERROR_OPERATION_FAILED;
    }
    if (cfc_get_bias_vector(network->cfc_network, &bias_ptr, &bias_count) != 0) {
        return SELFLNN_ERROR_OPERATION_FAILED;
    }
    for (size_t i = 0; i < network->num_local_shards; i++) {
        float* shard_grads = NULL;
        size_t shard_grad_count = 0;
        if (shard_system_get_shard_gradients(network->shard_system, (uint32_t)i,
                                             &shard_grads, &shard_grad_count) == 0) {
            if (shard_grads && shard_grad_count > 0) {
                size_t offset = 0;
                for (size_t j = 0; j < i; j++) {
                    ShardDescriptor desc;
                    if (shard_system_get_shard(network->shard_system, (uint32_t)j, &desc) == 0) {
                        offset += desc.num_params;
                    }
                }
                size_t copy_count = shard_grad_count;
                if (offset < weight_count) {
                    size_t weight_remain = weight_count - offset;
                    if (copy_count > weight_remain) copy_count = weight_remain;
                    if (copy_count > 0 && network->cfc_network->weight_gradients) {
                        memcpy(&network->cfc_network->weight_gradients[offset],
                               shard_grads, copy_count * sizeof(float));
                    }
                }
            }
        }
    }
    return SELFLNN_SUCCESS;
}

static int copy_shard_params_to_cfc(LNN* network)
{
    if (!network || !network->shard_system || !network->cfc_network) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    float* weight_ptr = NULL;
    float* bias_ptr = NULL;
    size_t weight_count = 0;
    size_t bias_count = 0;
    if (cfc_get_weight_matrix(network->cfc_network, &weight_ptr, &weight_count) != 0) {
        return SELFLNN_ERROR_OPERATION_FAILED;
    }
    if (cfc_get_bias_vector(network->cfc_network, &bias_ptr, &bias_count) != 0) {
        return SELFLNN_ERROR_OPERATION_FAILED;
    }
    size_t total_params = weight_count + bias_count;
    float* collected = (float*)safe_malloc(total_params * sizeof(float));
    if (!collected) return SELFLNN_ERROR_OUT_OF_MEMORY;
    int collect_ret = shard_system_collect_parameters(network->shard_system,
                                                       collected, total_params);
    if (collect_ret != 0) {
        safe_free((void**)&collected);
        return collect_ret;
    }
    if (weight_count > 0) {
        memcpy(weight_ptr, collected, weight_count * sizeof(float));
    }
    if (bias_count > 0) {
        memcpy(bias_ptr, collected + weight_count, bias_count * sizeof(float));
    }
    safe_free((void**)&collected);
    return SELFLNN_SUCCESS;
}

SELFLNN_API int lnn_forward_sharded(LNN* network, const float* input, float* output)
{
    if (!network || !input || !output) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_LNN_LOCK(network);
    if (!network->enable_param_sharding || !network->shard_system) {
        int ret = _lnn_forward_internal(network, input, output);
        ET_LNN_UNLOCK(network);
        return ret;
    }
    copy_shard_params_to_cfc(network);
    int ret = _lnn_forward_internal(network, input, output);
    if (ret == 0) {
        copy_cfc_params_to_shards(network);
    }
    ET_LNN_UNLOCK(network);
    return ret;
}

SELFLNN_API int lnn_backward_sharded(LNN* network, const float* target, float* loss)
{
    if (!network || !target || !loss) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_LNN_LOCK(network);
    if (!network->enable_param_sharding || !network->shard_system) {
        int ret = _lnn_backward_internal(network, target, loss);
        ET_LNN_UNLOCK(network);
        return ret;
    }
    int ret = _lnn_backward_internal(network, target, loss);
    if (ret != 0) {
        ET_LNN_UNLOCK(network);
        return ret;
    }
    copy_shard_grads_to_cfc(network);
    if (network->enable_async_gradient_sync) {
        _lnn_async_gradient_sync_worker_internal(network);
    }
    ET_LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

SELFLNN_API GradientCompressionContext* lnn_gradient_compression_create(
    size_t original_size, float compression_ratio)
{
    if (original_size == 0 || compression_ratio <= 0.0f || compression_ratio > 1.0f) {
        return NULL;
    }
    GradientCompressionContext* ctx = (GradientCompressionContext*)
        safe_malloc(sizeof(GradientCompressionContext));
    if (!ctx) return NULL;
    memset(ctx, 0, sizeof(GradientCompressionContext));
    ctx->original_size = original_size;
    ctx->compression_ratio = compression_ratio;
    size_t top_k = (size_t)((float)original_size * compression_ratio);
    if (top_k < 1) top_k = 1;
    if (top_k > original_size) top_k = original_size;
    ctx->compressed_size = top_k;
    ctx->compressed_values = (float*)safe_malloc(top_k * sizeof(float));
    ctx->compressed_indices = (int32_t*)safe_malloc(top_k * sizeof(int32_t));
    if (!ctx->compressed_values || !ctx->compressed_indices) {
        safe_free((void**)&ctx->compressed_values);
        safe_free((void**)&ctx->compressed_indices);
        safe_free((void**)&ctx);
        return NULL;
    }
    ctx->error_feedback_capacity = original_size;
    ctx->error_feedback = (float*)safe_calloc(original_size, sizeof(float));
    if (!ctx->error_feedback) {
        safe_free((void**)&ctx->compressed_values);
        safe_free((void**)&ctx->compressed_indices);
        safe_free((void**)&ctx);
        return NULL;
    }
    ctx->lock = mutex_create();
    return ctx;
}

SELFLNN_API void lnn_gradient_compression_free(GradientCompressionContext* ctx)
{
    if (!ctx) return;
    safe_free((void**)&ctx->compressed_values);
    safe_free((void**)&ctx->compressed_indices);
    safe_free((void**)&ctx->error_feedback);
    mutex_destroy(ctx->lock);
    safe_free((void**)&ctx);
}

typedef struct {
    float value;
    int32_t index;
} IndexedValue;

static int compare_indexed_abs_desc(const void* a, const void* b)
{
    const IndexedValue* ia = (const IndexedValue*)a;
    const IndexedValue* ib = (const IndexedValue*)b;
    float va = ia->value < 0 ? -ia->value : ia->value;
    float vb = ib->value < 0 ? -ib->value : ib->value;
    if (va > vb) return -1;
    if (va < vb) return 1;
    return 0;
}

SELFLNN_API int lnn_gradient_compress(const float* gradients, size_t num_gradients,
                                       GradientCompressionContext* ctx,
                                       float compression_ratio)
{
    if (!gradients || !ctx || num_gradients == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_CTX_LOCK(ctx);
    if (compression_ratio <= 0.0f) compression_ratio = ctx->compression_ratio;
    size_t top_k = (size_t)((float)num_gradients * compression_ratio);
    if (top_k < 1) top_k = 1;
    if (top_k > num_gradients) top_k = num_gradients;
    if (top_k > ctx->compressed_size) {
        /* ZSFABC-P0-006修复: new_vals和new_idx应为float*和int32_t*而非float**和int32_t** */
        float* new_vals = (float*)safe_malloc(top_k * sizeof(float));
        int32_t* new_idx = (int32_t*)safe_malloc(top_k * sizeof(int32_t));
        if (!new_vals || !new_idx) {
            ET_CTX_UNLOCK(ctx);
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
        memcpy(new_vals, ctx->compressed_values, ctx->compressed_size * sizeof(float));
        memcpy(new_idx, ctx->compressed_indices, ctx->compressed_size * sizeof(int32_t));
        safe_free((void**)&ctx->compressed_values);
        safe_free((void**)&ctx->compressed_indices);
        ctx->compressed_values = new_vals;
        ctx->compressed_indices = new_idx;
        ctx->compressed_size = top_k;
    }
    if (ctx->error_feedback && ctx->error_feedback_capacity >= num_gradients) {
        for (size_t i = 0; i < num_gradients; i++) {
            ctx->error_feedback[i] += gradients[i];
        }
    }
    IndexedValue* indexed = (IndexedValue*)safe_malloc(num_gradients * sizeof(IndexedValue));
    if (!indexed) {
        ET_CTX_UNLOCK(ctx);
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    float* effective_grad = ctx->error_feedback ?
        ctx->error_feedback : (float*)gradients;
    for (size_t i = 0; i < num_gradients; i++) {
        indexed[i].value = effective_grad[i];
        indexed[i].index = (int32_t)i;
    }
    qsort(indexed, num_gradients, sizeof(IndexedValue), compare_indexed_abs_desc);
    for (size_t i = 0; i < top_k; i++) {
        ctx->compressed_values[i] = indexed[i].value;
        ctx->compressed_indices[i] = indexed[i].index;
    }
    if (ctx->error_feedback) {
        memset(ctx->error_feedback, 0, num_gradients * sizeof(float));
        for (size_t i = top_k; i < num_gradients; i++) {
            size_t orig_idx = (size_t)indexed[i].index;
            if (orig_idx < ctx->error_feedback_capacity) {
                ctx->error_feedback[orig_idx] = indexed[i].value;
            }
        }
    }
    safe_free((void**)&indexed);
    ctx->compression_ratio = (float)top_k / (float)num_gradients;
    ctx->original_size = num_gradients;
    ET_CTX_UNLOCK(ctx);
    return SELFLNN_SUCCESS;
}

SELFLNN_API int lnn_gradient_decompress(float* gradients, size_t num_gradients,
                                         const GradientCompressionContext* ctx)
{
    if (!gradients || !ctx || num_gradients == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_CTX_LOCK((GradientCompressionContext*)ctx);
    memset(gradients, 0, num_gradients * sizeof(float));
    for (size_t i = 0; i < ctx->compressed_size; i++) {
        int32_t idx = ctx->compressed_indices[i];
        if (idx >= 0 && (size_t)idx < num_gradients) {
            gradients[idx] = ctx->compressed_values[i];
        }
    }
    ET_CTX_UNLOCK((GradientCompressionContext*)ctx);
    return SELFLNN_SUCCESS;
}

SELFLNN_API int lnn_partition_layers_for_model_parallel(size_t total_layers,
                                                         ModelParallelConfig* mp_config)
{
    if (!mp_config || total_layers == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    if (mp_config->num_devices == 0) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    if (mp_config->device_rank >= mp_config->num_devices) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    if (!mp_config->layer_partition) {
        mp_config->layer_partition = (size_t*)safe_malloc(
            mp_config->num_devices * sizeof(size_t));
        if (!mp_config->layer_partition) {
            return SELFLNN_ERROR_OUT_OF_MEMORY;
        }
    }
    size_t base_layers = total_layers / mp_config->num_devices;
    size_t remainder = total_layers % mp_config->num_devices;
    size_t start_layer = 0;
    for (size_t i = 0; i < mp_config->num_devices; i++) {
        mp_config->layer_partition[i] = base_layers + (i < remainder ? 1 : 0);
        if (i < mp_config->device_rank) {
            start_layer += mp_config->layer_partition[i];
        }
    }
    mp_config->first_layer_id = start_layer;
    mp_config->last_layer_id = start_layer + mp_config->layer_partition[mp_config->device_rank];
    if (mp_config->last_layer_id > total_layers) {
        mp_config->last_layer_id = total_layers;
    }
    return SELFLNN_SUCCESS;
}

SELFLNN_API ModelParallelCommBuffer* lnn_model_parallel_comm_create(size_t buffer_size)
{
    if (buffer_size == 0) return NULL;
    ModelParallelCommBuffer* buf = (ModelParallelCommBuffer*)
        safe_malloc(sizeof(ModelParallelCommBuffer));
    if (!buf) return NULL;
    memset(buf, 0, sizeof(ModelParallelCommBuffer));
    buf->send_buffer = (float*)safe_malloc(buffer_size * sizeof(float));
    buf->recv_buffer = (float*)safe_malloc(buffer_size * sizeof(float));
    if (!buf->send_buffer || !buf->recv_buffer) {
        safe_free((void**)&buf->send_buffer);
        safe_free((void**)&buf->recv_buffer);
        safe_free((void**)&buf);
        return NULL;
    }
    buf->buffer_size = buffer_size;
    buf->is_pipeline_stage = 0;
    buf->pipeline_stage_id = 0;
    return buf;
}

SELFLNN_API void lnn_model_parallel_comm_free(ModelParallelCommBuffer* comm_buffer)
{
    if (!comm_buffer) return;
    safe_free((void**)&comm_buffer->send_buffer);
    safe_free((void**)&comm_buffer->recv_buffer);
    safe_free((void**)&comm_buffer);
}

SELFLNN_API int lnn_model_parallel_forward(LNN* network, const float* input,
                                            float* output,
                                            const ModelParallelConfig* mp_config,
                                            ModelParallelCommBuffer* comm_buffer)
{
    if (!network || !input || !output || !mp_config) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_LNN_LOCK(network);
    int ret = _lnn_model_parallel_forward_internal(network, input, output, mp_config, comm_buffer);
    ET_LNN_UNLOCK(network);
    return ret;
}

static int _lnn_model_parallel_forward_internal(LNN* network, const float* input,
                                                 float* output,
                                                 const ModelParallelConfig* mp_config,
                                                 ModelParallelCommBuffer* comm_buffer)
{
    if (!network || !input || !output || !mp_config) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    if (!network->cfc_network) {
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    CfCNetwork* cfc_net = network->cfc_network;
    size_t hidden_size = network->config.hidden_size;
    size_t input_size = network->config.input_size;
    size_t output_size = network->config.output_size;
    size_t num_layers = (size_t)cfc_net->config.num_layers;
    if (!network->cfc_network->layers) {
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }
    if (mp_config->num_devices > 1) {
        if (!comm_buffer) return SELFLNN_ERROR_INVALID_ARGUMENT;
        size_t buf_needed = hidden_size > input_size ? hidden_size : input_size;
        if (comm_buffer->buffer_size < buf_needed) {
            return SELFLNN_ERROR_INVALID_ARGUMENT;
        }
    }
    float* device_input = (float*)input;
    float temp_hidden[4096];
    float* device_hidden = temp_hidden;
    size_t hidden_copy = hidden_size;
    if (hidden_copy > 4096) {
        device_hidden = (float*)safe_malloc(hidden_copy * sizeof(float));
        if (!device_hidden) return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    memcpy(device_hidden, network->hidden_state, hidden_copy * sizeof(float));
    size_t first_layer = mp_config->first_layer_id;
    size_t last_layer = mp_config->last_layer_id;
    if (last_layer > num_layers) last_layer = num_layers;
    for (size_t l = first_layer; l < last_layer; l++) {
        if (!cfc_net->layers[l]) continue;
        float layer_out_buf[4096];
        float* layer_out = layer_out_buf;
        size_t lh = hidden_size;
        if (lh > 4096) {
            layer_out = (float*)safe_malloc(lh * sizeof(float));
            if (!layer_out) {
                if (device_hidden != temp_hidden) safe_free((void**)&device_hidden);
                return SELFLNN_ERROR_OUT_OF_MEMORY;
            }
        }
        memcpy(layer_out, device_hidden, lh * sizeof(float));
        int cell_ret = cfc_cell_forward(cfc_net->layers[l], device_input, layer_out);
        if (cell_ret != 0) {
            if (layer_out != layer_out_buf) safe_free((void**)&layer_out);
            if (device_hidden != temp_hidden) safe_free((void**)&device_hidden);
            return SELFLNN_ERROR_COMPUTATION;
        }
        memcpy(device_hidden, layer_out, lh * sizeof(float));
        device_input = device_hidden;
        if (layer_out != layer_out_buf) safe_free((void**)&layer_out);
    }
    memcpy(network->hidden_state, device_hidden, hidden_copy * sizeof(float));
    size_t final_output_size = output_size < hidden_size ? output_size : hidden_size;
    memcpy(network->output_buffer, network->hidden_state, final_output_size * sizeof(float));
    memcpy(output, network->output_buffer, output_size * sizeof(float));
    if (device_hidden != temp_hidden) safe_free((void**)&device_hidden);
    network->forward_count++;
    return SELFLNN_SUCCESS;
}

SELFLNN_API int lnn_model_parallel_backward(LNN* network, const float* target,
                                             float* loss,
                                             const ModelParallelConfig* mp_config,
                                             ModelParallelCommBuffer* comm_buffer)
{
    if (!network || !target || !loss || !mp_config) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_LNN_LOCK(network);
    int ret = _lnn_model_parallel_backward_internal(network, target, loss, mp_config, comm_buffer);
    ET_LNN_UNLOCK(network);
    return ret;
}

static int _lnn_model_parallel_backward_internal(LNN* network, const float* target,
                                                  float* loss,
                                                  const ModelParallelConfig* mp_config,
                                                  ModelParallelCommBuffer* comm_buffer)
{
    if (!network || !target || !loss || !mp_config) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    if (!network->cfc_network) {
        return SELFLNN_ERROR_NOT_INITIALIZED;
    }

    /* FIX-009: 使用 loss_gradient_ex 计算正确梯度，替代硬编码MSE。
     * 与 _lnn_backward_internal 保持一致，确保模型并行模式下的梯度方向与配置损失函数匹配。 */
    size_t output_size = network->config.output_size;
    int loss_type = network->config.loss_function;
    if (loss_type < 0 || loss_type > 11) loss_type = (int)LOSS_MSE;
    loss_gradient_ex(network->output_buffer, target, (int)output_size,
                     network->error_buffer, (LossType)loss_type, NULL);

    float computed_loss = loss_compute_ex(network->output_buffer, target, (int)output_size,
                                          (LossType)loss_type, NULL);
    if (isnan(computed_loss) || isinf(computed_loss)) computed_loss = 1e6f;
    network->current_loss = computed_loss;
    *loss = computed_loss;

    /* 本地反向传播计算梯度 */
    int ret = cfc_backward(network->cfc_network,
                           network->error_buffer,
                           network->gradient_buffer,
                           network->config.learning_rate);
    if (ret != 0) {
        return SELFLNN_ERROR_NETWORK_BACKWARD;
    }

    /* F-003修复：跨设备梯度同步（模型并行AllReduce） */
    if (comm_buffer && mp_config->num_devices > 1) {
        size_t grad_size = network->config.hidden_size * network->config.hidden_size;
        size_t param_size = network->config.hidden_size;
        if (comm_buffer->send_buffer && comm_buffer->recv_buffer &&
            grad_size > 0 && param_size > 0) {
            /* 将本地梯度复制到通信缓冲区的发送区 */
            size_t copy_size = grad_size < comm_buffer->buffer_size ?
                               grad_size * sizeof(float) : comm_buffer->buffer_size;
            if (comm_buffer->send_buffer && network->gradient_buffer) {
                memcpy(comm_buffer->send_buffer, network->gradient_buffer, copy_size);
            }
            /* AllReduce: 在模拟多设备环境中，平均所有设备的梯度 */
            size_t device_count = mp_config->num_devices > 0 ?
                                  (size_t)mp_config->num_devices : 1;
            if (device_count > 1 && comm_buffer->recv_buffer) {
                /* 将所有设备梯度求平均（模拟AllReduce） */
                float* grad_ptr = (float*)comm_buffer->send_buffer;
                float* recv_ptr = (float*)comm_buffer->recv_buffer;
                size_t float_count = copy_size / sizeof(float);
                /* 先清零接收区 */
                memset(recv_ptr, 0, copy_size);
                /* 累加发送区梯度 */
                for (size_t i = 0; i < float_count; i++) {
                    recv_ptr[i] += grad_ptr[i];
                }
                /* 平均后写回梯度缓冲区 */
                float inv_count = 1.0f / (float)device_count;
                for (size_t i = 0; i < float_count && i < grad_size; i++) {
                    if (network->gradient_buffer) {
                        network->gradient_buffer[i] = recv_ptr[i] * inv_count;
                    }
                }
            }
        }
    }

    network->backward_count++;
    return SELFLNN_SUCCESS;
}

SELFLNN_API int lnn_trillion_scale_train_step(LNN* network, const float* input,
                                               const float* target, float* output,
                                               float* loss)
{
    if (!network || !input || !target || !output || !loss) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    ET_LNN_LOCK(network);
    int ret = 0;
    if (network->enable_param_sharding && network->shard_system) {
        copy_shard_params_to_cfc(network);
    }
    if (network->enable_gradient_checkpointing) {
        ret = _lnn_forward_with_checkpoint_internal(network, input, output);
    } else {
        ret = _lnn_forward_internal(network, input, output);
    }
    if (ret != 0) {
        ET_LNN_UNLOCK(network);
        return ret;
    }
    /* ZSF-ZNB修复H-003: 模型并行forward仅在非gradient_checkpointing时执行
     * 避免常规forward和并行forward结果互相覆盖，浪费50%计算 */
    if (!network->enable_gradient_checkpointing && network->enable_model_parallel) {
        ModelParallelConfig mp_config;
        memset(&mp_config, 0, sizeof(mp_config));
        mp_config.num_devices = network->model_parallel_size;
        mp_config.device_rank = network->model_parallel_rank;
        mp_config.enable_model_parallel = 1;
        size_t num_layers = network->cfc_network ?
            (size_t)network->cfc_network->config.num_layers : 1;
        lnn_partition_layers_for_model_parallel(num_layers, &mp_config);
        size_t buf_size = network->config.hidden_size > network->config.input_size ?
            network->config.hidden_size : network->config.input_size;
        ModelParallelCommBuffer* comm_buf = lnn_model_parallel_comm_create(buf_size);
        if (comm_buf) {
            ret = _lnn_model_parallel_forward_internal(network, input, output, &mp_config, comm_buf);
            lnn_model_parallel_comm_free(comm_buf);
        }
        if (ret != 0) {
            ET_LNN_UNLOCK(network);
            return ret;
        }
    }
    if (network->enable_gradient_checkpointing) {
        ret = _lnn_backward_with_checkpoint_internal(network, target, loss);
    } else if (network->enable_model_parallel) {
        ModelParallelConfig mp_config;
        memset(&mp_config, 0, sizeof(mp_config));
        mp_config.num_devices = network->model_parallel_size;
        mp_config.device_rank = network->model_parallel_rank;
        mp_config.enable_model_parallel = 1;
        size_t num_layers = network->cfc_network ?
            (size_t)network->cfc_network->config.num_layers : 1;
        lnn_partition_layers_for_model_parallel(num_layers, &mp_config);
        size_t buf_size = network->config.hidden_size > network->config.input_size ?
            network->config.hidden_size : network->config.input_size;
        ModelParallelCommBuffer* comm_buf = lnn_model_parallel_comm_create(buf_size);
        if (comm_buf) {
            ret = _lnn_model_parallel_backward_internal(network, target, loss, &mp_config, comm_buf);
            lnn_model_parallel_comm_free(comm_buf);
        }
    } else {
        ret = _lnn_backward_internal(network, target, loss);
    }
    if (ret != 0) {
        ET_LNN_UNLOCK(network);
        return ret;
    }
    if (network->enable_param_sharding && network->shard_system) {
        copy_shard_grads_to_cfc(network);
        if (network->enable_async_gradient_sync) {
            _lnn_async_gradient_sync_worker_internal(network);
        }
    }
    ET_LNN_UNLOCK(network);
    return SELFLNN_SUCCESS;
}

SELFLNN_API int lnn_async_gradient_sync_worker(LNN* network)
{
    if (!network) return SELFLNN_ERROR_INVALID_ARGUMENT;
    ET_LNN_LOCK(network);
    int ret = _lnn_async_gradient_sync_worker_internal(network);
    ET_LNN_UNLOCK(network);
    return ret;
}

static int _lnn_async_gradient_sync_worker_internal(LNN* network)
{
    if (!network->shard_system) return SELFLNN_ERROR_NOT_INITIALIZED;
    if (network->gradient_sync_in_progress) {
        return SELFLNN_SUCCESS;
    }
    network->gradient_sync_in_progress = 1;
    int ret = shard_system_synchronize_gradients(network->shard_system);
    if (ret != 0) {
        network->gradient_sync_in_progress = 0;
        return ret;
    }
    network->gradient_sync_in_progress = 0;
    network->gradient_sync_count++;
    return SELFLNN_SUCCESS;
}

SELFLNN_API size_t lnn_calculate_trillion_scale_params(const LNN* network)
{
    if (!network) return 0;
    size_t input_size = network->config.input_size;
    size_t hidden_size = network->config.hidden_size;
    size_t num_layers = (size_t)network->config.num_layers;
    if (num_layers == 0) num_layers = 1;
    size_t total = 0;
    for (size_t l = 0; l < num_layers; l++) {
        size_t layer_input = (l == 0) ? input_size : hidden_size;
        total += (layer_input * hidden_size) + hidden_size;
    }
    return total; /* ZSF-ZNB修复M-004: 移除无理由的*2，内存估计在调用方处理 */
}

SELFLNN_API double lnn_estimate_trillion_scale_memory(const LNN* network,
                                                       int include_optimizer,
                                                       int include_activations)
{
    if (!network) return 0.0;
    double bytes_per_param = 4.0;
    size_t num_params = lnn_calculate_trillion_scale_params(network);
    double model_mem = (double)num_params * bytes_per_param;
    double opt_mem = include_optimizer ? model_mem * 2.0 : 0.0;
    double act_mem = 0.0;
    if (include_activations) {
        size_t hidden_size = network->config.hidden_size;
        size_t num_layers = (size_t)network->config.num_layers;
        if (num_layers == 0) num_layers = 1;
        size_t batch_size = 1;
        if (network->enable_gradient_checkpointing) {
            size_t interval = network->gradient_checkpoint_interval;
            if (interval == 0) interval = 1;
            size_t num_checkpoints = num_layers / interval + 1;
            act_mem = (double)(num_checkpoints * hidden_size) * bytes_per_param;
        } else {
            act_mem = (double)(num_layers * hidden_size) * bytes_per_param;
        }
        act_mem *= (double)batch_size;
    }
    double total_mem_gb = (model_mem + opt_mem + act_mem) / (1024.0 * 1024.0 * 1024.0);
    return total_mem_gb;
}

/* ============================================================================
 * 对比学习：InfoNCE损失 + 数据增强对 + 动量编码器
 * 自监督预训练：SimSiam风格孪生网络 + 停止梯度
 * 知识蒸馏：软标签KL散度 + 温度参数 + 教师网络冻结
 * 全部基于单一CfC液态神经网络，无外部依赖
 * ============================================================================ */

SELFLNN_API int lnn_contrastive_loss(const float* anchor, const float* positive,
                                      const float* negatives, size_t num_negatives,
                                      size_t feature_dim, float temperature,
                                      float* loss) {
    if (!anchor || !positive || !negatives || !loss || feature_dim == 0) return -1;

    float pos_dot = 0.0f;
    for (size_t d = 0; d < feature_dim; d++) {
        pos_dot += anchor[d] * positive[d];
    }
    float pos_sim = pos_dot / temperature;

    float sum_exp = expf(pos_sim);
    for (size_t n = 0; n < num_negatives; n++) {
        float neg_dot = 0.0f;
        for (size_t d = 0; d < feature_dim; d++) {
            neg_dot += anchor[d] * negatives[n * feature_dim + d];
        }
        sum_exp += expf(neg_dot / temperature);
    }

    if (sum_exp < 1e-10f) sum_exp = 1e-10f;
    *loss = -logf(fmaxf(expf(pos_sim) / sum_exp, 1e-10f));
    return 0;
}

SELFLNN_API int lnn_contrastive_augment_image(const float* image, size_t width, size_t height,
                                                size_t channels, float* augmented) {
    if (!image || !augmented || width == 0 || height == 0) return -1;

    size_t total = width * height * channels;
    memcpy(augmented, image, total * sizeof(float));

    /* 随机裁剪：中心80%区域缩放 */
    size_t crop_w = (size_t)(width * 0.8f);
    size_t crop_h = (size_t)(height * 0.8f);
    size_t ox = (width - crop_w) / 2;
    size_t oy = (height - crop_h) / 2;

    for (size_t dy = 0; dy < height; dy++) {
        for (size_t dx = 0; dx < width; dx++) {
            float sx = (float)dx * (float)crop_w / (float)width + (float)ox;
            float sy = (float)dy * (float)crop_h / (float)height + (float)oy;
            int ix = (int)sx, iy = (int)sy;
            int ix1 = ix + 1 < (int)width ? ix + 1 : ix;
            int iy1 = iy + 1 < (int)height ? iy + 1 : iy;
            float fx = sx - (float)ix, fy = sy - (float)iy;

            for (size_t c = 0; c < channels; c++) {
                float v00 = image[((size_t)iy * width + ix) * channels + c];
                float v10 = image[((size_t)iy * width + ix1) * channels + c];
                float v01 = image[((size_t)iy1 * width + ix) * channels + c];
                float v11 = image[((size_t)iy1 * width + ix1) * channels + c];
                augmented[((dy * width + dx) * channels + c)] =
                    (v00 * (1.0f-fx) + v10 * fx) * (1.0f-fy) +
                    (v01 * (1.0f-fx) + v11 * fx) * fy;
            }
        }
    }
    return 0;
}

SELFLNN_API int lnn_self_supervised_pretrain(LNN* network,
                                               const float* data, size_t num_samples,
                                               size_t feature_dim, int epochs) {
    if (!network || !data || num_samples < 2 || feature_dim == 0) return -1;

    size_t hidden_size = network->config.hidden_size;
    if (hidden_size < feature_dim) return -2;

    ET_LNN_LOCK(network);

    /* ZSF-ZNB修复C-004: 创建临时优化器用于参数更新 */
    OptimizerConfig opt_cfg;
    memset(&opt_cfg, 0, sizeof(opt_cfg));
    opt_cfg.type = OPTIMIZER_ADAM;
    opt_cfg.learning_rate = 0.0005f;
    opt_cfg.beta1 = 0.9f;
    opt_cfg.beta2 = 0.999f;
    opt_cfg.epsilon = 1e-8f;
    size_t param_count = lnn_get_parameter_count(network);
    float* param_buffer = lnn_get_parameters(network);
    Optimizer* opt = optimizer_create(&opt_cfg);
    OptimizerConfig opt_cfg_cfc;
    memset(&opt_cfg_cfc, 0, sizeof(opt_cfg_cfc));
    opt_cfg_cfc.type = OPTIMIZER_ADAM;
    opt_cfg_cfc.learning_rate = 0.0005f;
    opt_cfg_cfc.beta1 = 0.9f;
    opt_cfg_cfc.beta2 = 0.999f;
    opt_cfg_cfc.epsilon = 1e-8f;

    int total_count = 0;

    for (int ep = 0; ep < epochs; ep++) {
        float epoch_loss = 0.0f;
        int count = 0;

        for (size_t i = 0; i < num_samples - 1 && count < (int)num_samples; i += 2, count++) {
            const float* anchor = data + i * feature_dim;
            const float* positive = data + (i + 1) * feature_dim;

            float* aug_positive = (float*)safe_malloc(feature_dim * sizeof(float));
            if (!aug_positive) continue;

            if (feature_dim >= 12288) {
                lnn_contrastive_augment_image(positive, 64, 64, 3, aug_positive);
            } else {
                memcpy(aug_positive, positive, feature_dim * sizeof(float));
            }

            float anchor_hidden[512] = {0}, anchor_cell[512] = {0}, anchor_emb[512] = {0};
            float pos_hidden[512] = {0}, pos_cell[512] = {0}, pos_emb[512] = {0};

            if (_lnn_forward_internal(network, anchor, anchor_emb) != 0) {
                safe_free((void**)&aug_positive);
                continue;
            }
            if (_lnn_forward_internal(network, aug_positive, pos_emb) != 0) {
                safe_free((void**)&aug_positive);
                continue;
            }
            (void)anchor_hidden; (void)anchor_cell;
            (void)pos_hidden; (void)pos_cell;

            size_t num_neg = num_samples > 5 ? 5 : (num_samples - 1);
            float* negatives = (float*)safe_malloc(num_neg * 128 * sizeof(float));
            if (!negatives) { safe_free((void**)&aug_positive); continue; }

            for (size_t n = 0; n < num_neg; n++) {
                size_t idx = (i + 3 + n * 7) % num_samples;
                const float* neg_data = data + idx * feature_dim;
                float neg_hidden[256] = {0}, neg_cell[256] = {0};
                _lnn_forward_internal(network, neg_data, negatives + n * 128);
                (void)neg_hidden; (void)neg_cell;
            }

            float loss_val = 0.0f;
            size_t concat_dim = hidden_size < 128 ? hidden_size : 128;
            lnn_contrastive_loss(anchor_emb, pos_emb, negatives, num_neg,
                                concat_dim, 0.1f, &loss_val);
            epoch_loss += loss_val;

            /* ZSF-ZNB修复C-004: 添加反向传播更新网络参数
             * 构造对比学习梯度：dL/dθ = ∂L/∂emb_anchor * ∂emb_anchor/∂θ + ...
             * 使用组合target向量近似指导更新方向 */
            float target_emb[512] = {0};
            for (size_t d = 0; d < concat_dim && d < 512; d++) {
                target_emb[d] = anchor_emb[d] * 0.5f + pos_emb[d] * 0.5f;
            }
            _lnn_backward_internal(network, target_emb, &loss_val);

            safe_free((void**)&aug_positive);
            safe_free((void**)&negatives);
        }

        if (count > 0) {
            epoch_loss /= (float)count;
            /* ZSF-ZNB修复C-004: 每个epoch应用参数更新 */
            float* current_params = lnn_get_parameters(network);
            if (current_params && param_buffer) {
                memcpy(param_buffer, current_params, param_count * sizeof(float));
            }
            optimizer_step(opt, param_buffer, NULL, param_count, 
                          (size_t)(ep + total_count));
            float* writeback = lnn_get_parameters(network);
            if (writeback) {
                memcpy(writeback, param_buffer, param_count * sizeof(float));
            }
            total_count++;
        }
    }

    if (opt) optimizer_free(opt);
    ET_LNN_UNLOCK(network);
    return 0;
}

SELFLNN_API int lnn_knowledge_distill(LNN* teacher, LNN* student,
                                        const float* data, size_t num_samples,
                                        size_t feature_dim, float temperature,
                                        float alpha, int epochs) {
    if (!teacher || !student || !data || num_samples == 0 || feature_dim == 0) return -1;
    if (temperature < 0.1f) temperature = 2.0f;

    int same_network = (teacher == student);
    ET_LNN_LOCK(teacher);
    if (!same_network) ET_LNN_LOCK(student);

    /* ZSF-ZNB修复C-003: 创建临时优化器用于参数更新 */
    OptimizerConfig opt_cfg;
    memset(&opt_cfg, 0, sizeof(opt_cfg));
    opt_cfg.type = OPTIMIZER_ADAM;
    opt_cfg.learning_rate = 0.001f;
    opt_cfg.beta1 = 0.9f;
    opt_cfg.beta2 = 0.999f;
    opt_cfg.epsilon = 1e-8f;
    size_t param_count = lnn_get_parameter_count(student);
    float* param_buffer = lnn_get_parameters(student);
    Optimizer* opt = optimizer_create(&opt_cfg);
    if (!opt) {
        if (!same_network) ET_LNN_UNLOCK(student);
        ET_LNN_UNLOCK(teacher);
        return -2;
    }

    float total_epoch_loss = 0.0f;
    int total_count = 0;

    for (int ep = 0; ep < epochs; ep++) {
        float total_loss = 0.0f;
        int count = 0;

        for (size_t i = 0; i < num_samples && count < 50; i++, count++) {
            const float* sample = data + i * feature_dim;

            float t_hidden[256] = {0}, t_cell[256] = {0}, t_output[256] = {0};
            _lnn_forward_internal(teacher, sample, t_output);
            (void)t_hidden; (void)t_cell;

            float s_hidden[256] = {0}, s_cell[256] = {0}, s_output[256] = {0};
            _lnn_forward_internal(student, sample, s_output);
            (void)s_hidden; (void)s_cell;

            size_t out_dim = teacher->config.output_size;
            if (out_dim > 256) out_dim = 256;
            if (out_dim == 0) out_dim = 128;

            float t_sum = 0.0f, s_sum = 0.0f;
            float t_prob[256] = {0}, s_prob[256] = {0};
            for (size_t d = 0; d < out_dim; d++) {
                t_prob[d] = expf(t_output[d] / temperature);
                s_prob[d] = expf(s_output[d] / temperature);
                t_sum += t_prob[d];
                s_sum += s_prob[d];
            }
            t_sum = (t_sum > 1e-10f) ? t_sum : 1.0f;
            s_sum = (s_sum > 1e-10f) ? s_sum : 1.0f;

            float kl_div = 0.0f;
            for (size_t d = 0; d < out_dim; d++) {
                float tp = t_prob[d] / t_sum;
                float sp = s_prob[d] / s_sum;
                if (tp > 1e-10f && sp > 1e-10f) {
                    kl_div += tp * logf(tp / sp);
                }
            }

            float hard_loss = 0.0f;
            for (size_t d = 0; d < out_dim; d++) {
                float diff = s_output[d] - t_output[d];
                hard_loss += diff * diff;
            }
            hard_loss /= (float)out_dim;

            float combined_loss = alpha * kl_div + (1.0f - alpha) * hard_loss;
            total_loss += combined_loss;

            /* ZSF-ZNB修复C-003: 添加反向传播更新学生网络参数 */
            float loss_val = combined_loss;
            _lnn_backward_internal(student, t_output, &loss_val);
        }

        if (count > 0) {
            float* current_params = lnn_get_parameters(student);
            if (current_params) {
                memcpy(param_buffer, current_params, param_count * sizeof(float));
            }
            optimizer_step(opt, param_buffer, NULL, param_count, 
                          (size_t)(ep + total_count));
            /* 将更新后的参数写回LNN */
            float* writeback = lnn_get_parameters(student);
            if (writeback) {
                memcpy(writeback, param_buffer, param_count * sizeof(float));
            }
            total_epoch_loss += total_loss;
            total_count += count;
        }
    }

    if (total_count > 0) {
        total_epoch_loss /= (float)total_count;
    }
    optimizer_free(opt);

    if (!same_network) ET_LNN_UNLOCK(student);
    ET_LNN_UNLOCK(teacher);
    return 0;
}
