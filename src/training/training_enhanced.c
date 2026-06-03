#define SELFLNN_IMPLEMENTATION
#include "selflnn/training/training_enhanced.h"
#include "selflnn/core/cfc_network.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/platform.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * 内部注册表：Trainer* → EnhancedTrainerState 映射
 * 使用静态数组实现，无需动态容器
 * ======================================================================== */

#define MAX_ENHANCED_TRAINERS 64

typedef struct {
    Trainer* trainer;
    EnhancedTrainerState state;
    int used;
} EnhancedTrainerEntry;

static EnhancedTrainerEntry g_enhanced_registry[MAX_ENHANCED_TRAINERS];
static int g_registry_initialized = 0;
static MutexHandle g_registry_lock = NULL;

#define REGISTRY_LOCK()    do { if (!g_registry_lock) g_registry_lock = mutex_create(); mutex_lock(g_registry_lock); } while(0)
#define REGISTRY_UNLOCK()  mutex_unlock(g_registry_lock)

static void registry_init(void)
{
    if (!g_registry_initialized) {
        memset(g_enhanced_registry, 0, sizeof(g_enhanced_registry));
        g_registry_initialized = 1;
    }
}

static EnhancedTrainerEntry* registry_find(Trainer* trainer)
{
    registry_init();
    for (int i = 0; i < MAX_ENHANCED_TRAINERS; i++) {
        if (g_enhanced_registry[i].used && g_enhanced_registry[i].trainer == trainer) {
            return &g_enhanced_registry[i];
        }
    }
    return NULL;
}

static EnhancedTrainerEntry* registry_alloc(Trainer* trainer)
{
    registry_init();
    for (int i = 0; i < MAX_ENHANCED_TRAINERS; i++) {
        if (!g_enhanced_registry[i].used) {
            memset(&g_enhanced_registry[i].state, 0, sizeof(EnhancedTrainerState));
            g_enhanced_registry[i].trainer = trainer;
            g_enhanced_registry[i].used = 1;
            return &g_enhanced_registry[i];
        }
    }
    return NULL;
}

static void registry_free(Trainer* trainer)
{
    for (int i = 0; i < MAX_ENHANCED_TRAINERS; i++) {
        EnhancedTrainerEntry* entry = &g_enhanced_registry[i];
        if (entry->used && entry->trainer == trainer) {
            if (entry->state.ema) {
                ema_weight_manager_free(entry->state.ema);
                entry->state.ema = NULL;
            }
            if (entry->state.warmup_cosine) {
                warmup_cosine_scheduler_free(entry->state.warmup_cosine);
                entry->state.warmup_cosine = NULL;
            }
            memset(&entry->state, 0, sizeof(EnhancedTrainerState));
            entry->used = 0;
            entry->trainer = NULL;
            break;
        }
    }
}

/* ========================================================================
 * 内部辅助函数：从LNN提取权重和偏置参数
 * 使用SELFLNN_IMPLEMENTATION访问内部结构
 * ======================================================================== */

static int lnn_get_weight_and_bias(LNN* network, float** weights, size_t* num_weights,
                                    float** biases, size_t* num_biases)
{
/* 使用公共API替代直接访问cfc_network内部字段 */
    if (!network) return -1;
    *weights = lnn_get_weight_matrix(network);
    *num_weights = lnn_get_weight_count(network);
    *biases = lnn_get_bias_vector(network);
    *num_biases = lnn_get_bias_count(network);
    if (!*weights || !*biases || *num_weights == 0 || *num_biases == 0) return -1;
    return 0;
}

static int lnn_set_weight_and_bias(LNN* network, const float* weights, const float* biases)
{
/* 使用公共API替代直接访问cfc_network内部字段 */
    return lnn_set_weights_and_biases(network, weights, biases);
}

static int lnn_get_hidden_size(LNN* network)
{
    if (!network || !network->cfc_network) return 0;
    return (int)network->cfc_network->config.hidden_size;
}

static int lnn_get_output_size_internal(LNN* network)
{
    if (!network || !network->cfc_network) return 0;
    return (int)network->cfc_network->config.output_size;
}

/* ========================================================================
 * 数值稳定函数
 * ======================================================================== */

static float stable_exp(float x)
{
    if (x < -80.0f) x = -80.0f;
    if (x > 80.0f) x = 80.0f;
    return expf(x);
}

static float softmax_scale(const float* input, size_t n)
{
    float max_val = input[0];
    for (size_t i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    return max_val;
}

static void apply_softmax(float* output, const float* input, size_t n)
{
    float max_val = softmax_scale(input, n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    if (sum > 1e-10f) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < n; i++) {
            output[i] *= inv_sum;
        }
    }
}

static float sigmoid(float x)
{
    if (x < -40.0f) return 0.0f;
    if (x > 40.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* ========================================================================
 * EMA权重管理器实现
 * ======================================================================== */

EMAWeightManager* ema_weight_manager_create(size_t num_weights, size_t num_biases, float decay)
{
    if (num_weights == 0 && num_biases == 0) return NULL;
    if (decay < 0.0f || decay >= 1.0f) decay = 0.999f;

    EMAWeightManager* mgr = (EMAWeightManager*)safe_malloc(sizeof(EMAWeightManager));
    if (!mgr) return NULL;

    mgr->ema_weights = NULL;
    mgr->ema_biases = NULL;
    mgr->decay = decay;
    mgr->num_weights = num_weights;
    mgr->num_biases = num_biases;
    mgr->initialized = 0;

    if (num_weights > 0) {
        mgr->ema_weights = (float*)safe_malloc(num_weights * sizeof(float));
        if (!mgr->ema_weights) {
            safe_free((void**)&mgr);
            return NULL;
        }
        memset(mgr->ema_weights, 0, num_weights * sizeof(float));
    }

    if (num_biases > 0) {
        mgr->ema_biases = (float*)safe_malloc(num_biases * sizeof(float));
        if (!mgr->ema_biases) {
            safe_free((void**)&mgr->ema_weights);
            safe_free((void**)&mgr);
            return NULL;
        }
        memset(mgr->ema_biases, 0, num_biases * sizeof(float));
    }

    return mgr;
}

void ema_weight_manager_update(EMAWeightManager* mgr, const float* weights, const float* biases)
{
    if (!mgr) return;

    if (!mgr->initialized) {
        if (weights && mgr->ema_weights && mgr->num_weights > 0) {
            memcpy(mgr->ema_weights, weights, mgr->num_weights * sizeof(float));
        }
        if (biases && mgr->ema_biases && mgr->num_biases > 0) {
            memcpy(mgr->ema_biases, biases, mgr->num_biases * sizeof(float));
        }
        mgr->initialized = 1;
        return;
    }

    float decay = mgr->decay;
    float one_minus_decay = 1.0f - decay;

    if (weights && mgr->ema_weights && mgr->num_weights > 0) {
        for (size_t i = 0; i < mgr->num_weights; i++) {
            mgr->ema_weights[i] = decay * mgr->ema_weights[i] + one_minus_decay * weights[i];
        }
    }

    if (biases && mgr->ema_biases && mgr->num_biases > 0) {
        for (size_t i = 0; i < mgr->num_biases; i++) {
            mgr->ema_biases[i] = decay * mgr->ema_biases[i] + one_minus_decay * biases[i];
        }
    }
}

void ema_weight_manager_apply(EMAWeightManager* mgr, float* weights, float* biases)
{
    if (!mgr || !mgr->initialized) return;

    if (weights && mgr->ema_weights && mgr->num_weights > 0) {
        memcpy(weights, mgr->ema_weights, mgr->num_weights * sizeof(float));
    }
    if (biases && mgr->ema_biases && mgr->num_biases > 0) {
        memcpy(biases, mgr->ema_biases, mgr->num_biases * sizeof(float));
    }
}

void ema_weight_manager_update_from_lnn(EMAWeightManager* mgr, LNN* network)
{
    if (!mgr || !network) return;
    float* weights = NULL;
    float* biases = NULL;
    size_t num_weights = 0, num_biases = 0;
    if (lnn_get_weight_and_bias(network, &weights, &num_weights, &biases, &num_biases) != 0) {
        return;
    }
    ema_weight_manager_update(mgr, weights, biases);
}

void ema_weight_manager_apply_to_lnn(EMAWeightManager* mgr, LNN* network)
{
    if (!mgr || !network) return;
    float* weights = NULL;
    float* biases = NULL;
    size_t num_weights = 0, num_biases = 0;
    if (lnn_get_weight_and_bias(network, &weights, &num_weights, &biases, &num_biases) != 0) {
        return;
    }
    ema_weight_manager_apply(mgr, weights, biases);
}

void ema_weight_manager_reset(EMAWeightManager* mgr)
{
    if (!mgr) return;
    if (mgr->ema_weights && mgr->num_weights > 0) {
        memset(mgr->ema_weights, 0, mgr->num_weights * sizeof(float));
    }
    if (mgr->ema_biases && mgr->num_biases > 0) {
        memset(mgr->ema_biases, 0, mgr->num_biases * sizeof(float));
    }
    mgr->initialized = 0;
}

void ema_weight_manager_free(EMAWeightManager* mgr)
{
    if (!mgr) return;
    if (mgr->ema_weights) {
        safe_free((void**)&mgr->ema_weights);
    }
    if (mgr->ema_biases) {
        safe_free((void**)&mgr->ema_biases);
    }
    safe_free((void**)&mgr);
}

/* ========================================================================
 * 标签平滑交叉熵损失实现
 * ======================================================================== */

float label_smoothing_cross_entropy(const float* predictions, const float* targets,
                                    size_t num_classes, float smoothing_factor)
{
    if (!predictions || !targets || num_classes == 0) return 0.0f;
    if (smoothing_factor < 0.0f) smoothing_factor = 0.0f;
    if (smoothing_factor >= 1.0f) smoothing_factor = 0.9f;

    float eps = smoothing_factor / (float)num_classes;
    float loss = 0.0f;

    for (size_t i = 0; i < num_classes; i++) {
        float p = predictions[i];
        if (p < 1e-10f) p = 1e-10f;
        if (p > 1.0f) p = 1.0f;

        float t = targets[i];
        float smoothed_target = t * (1.0f - smoothing_factor) + eps;
        loss -= smoothed_target * logf(p);
    }

    return loss;
}

void label_smoothing_cross_entropy_gradient(const float* predictions, const float* targets,
                                            size_t num_classes, float smoothing_factor,
                                            float* gradient)
{
    if (!predictions || !targets || !gradient || num_classes == 0) return;
    if (smoothing_factor < 0.0f) smoothing_factor = 0.0f;
    if (smoothing_factor >= 1.0f) smoothing_factor = 0.9f;

    float eps = smoothing_factor / (float)num_classes;

    for (size_t i = 0; i < num_classes; i++) {
        float p = predictions[i];
        if (p < 1e-10f) p = 1e-10f;
        float t = targets[i];
        float smoothed_target = t * (1.0f - smoothing_factor) + eps;
        gradient[i] = p - smoothed_target;
    }
}

/* ========================================================================
 * Focal Loss实现
 * ======================================================================== */

float focal_loss(const float* predictions, const float* targets,
                 size_t num_classes, float gamma, float alpha)
{
    if (!predictions || !targets || num_classes == 0) return 0.0f;
    if (gamma < 0.0f) gamma = 2.0f;
    if (alpha < 0.0f || alpha > 1.0f) alpha = 0.25f;

    float loss = 0.0f;

    for (size_t i = 0; i < num_classes; i++) {
        float p = sigmoid(predictions[i]);
        if (p < 1e-10f) p = 1e-10f;
        if (p > 1.0f - 1e-10f) p = 1.0f - 1e-10f;

        float t = targets[i];
        float p_t = t * p + (1.0f - t) * (1.0f - p);
        float alpha_t = t * alpha + (1.0f - t) * (1.0f - alpha);

        float modulating = powf(1.0f - p_t, gamma);
        loss -= alpha_t * modulating * logf(p_t);
    }

    return loss;
}

void focal_loss_gradient(const float* predictions, const float* targets,
                         size_t num_classes, float gamma, float alpha,
                         float* gradient)
{
    if (!predictions || !targets || !gradient || num_classes == 0) return;
    if (gamma < 0.0f) gamma = 2.0f;
    if (alpha < 0.0f || alpha > 1.0f) alpha = 0.25f;

    for (size_t i = 0; i < num_classes; i++) {
        float p = sigmoid(predictions[i]);
        if (p < 1e-10f) p = 1e-10f;
        if (p > 1.0f - 1e-10f) p = 1.0f - 1e-10f;

        float t = targets[i];
        float p_t = t * p + (1.0f - t) * (1.0f - p);
        float alpha_t = t * alpha + (1.0f - t) * (1.0f - alpha);

        float modulating = powf(1.0f - p_t, gamma);
        float gradient_mod = 0.0f;

        if (t > 0.5f) {
            gradient_mod = alpha_t * (gamma * p * powf(1.0f - p, gamma - 1.0f) * logf(p)
                                      - modulating);
        } else {
            gradient_mod = (1.0f - alpha_t) * (gamma * (1.0f - p) * powf(p, gamma - 1.0f) * logf(1.0f - p)
                                                - modulating);
        }

        gradient[i] = gradient_mod;
    }
}

/* ========================================================================
 * Warmup+余弦退火学习率调度器实现
 * ======================================================================== */

WarmupCosineScheduler* warmup_cosine_scheduler_create(float base_lr, float min_lr,
                                                      size_t warmup_steps, size_t total_steps)
{
    if (base_lr <= 0.0f || total_steps == 0) return NULL;
    if (min_lr < 0.0f) min_lr = 0.0f;
    if (min_lr > base_lr) min_lr = base_lr * 0.01f;

    WarmupCosineScheduler* scheduler = (WarmupCosineScheduler*)safe_malloc(sizeof(WarmupCosineScheduler));
    if (!scheduler) return NULL;

    scheduler->base_lr = base_lr;
    scheduler->min_lr = min_lr;
    scheduler->warmup_steps = warmup_steps;
    scheduler->total_steps = total_steps;
    scheduler->current_step = 0;
    scheduler->warmup_done = (warmup_steps == 0);

    return scheduler;
}

float warmup_cosine_scheduler_step(WarmupCosineScheduler* scheduler)
{
    if (!scheduler) return 0.0f;

    scheduler->current_step++;
    return warmup_cosine_scheduler_get_lr(scheduler);
}

float warmup_cosine_scheduler_get_lr(const WarmupCosineScheduler* scheduler)
{
    if (!scheduler) return 0.0f;

    size_t step = scheduler->current_step;
    size_t warmup_steps = scheduler->warmup_steps;
    size_t total_steps = scheduler->total_steps;

    if (step < warmup_steps && warmup_steps > 0) {
        float progress = (float)step / (float)warmup_steps;
        return scheduler->min_lr + (scheduler->base_lr - scheduler->min_lr) * progress;
    }

    float cosine_progress = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
    if (cosine_progress > 1.0f) cosine_progress = 1.0f;
    if (cosine_progress < 0.0f) cosine_progress = 0.0f;

    float cosine_decay = 0.5f * (1.0f + cosf((float)M_PI * cosine_progress));
    return scheduler->min_lr + (scheduler->base_lr - scheduler->min_lr) * cosine_decay;
}

void warmup_cosine_scheduler_reset(WarmupCosineScheduler* scheduler)
{
    if (!scheduler) return;
    scheduler->current_step = 0;
    scheduler->warmup_done = (scheduler->warmup_steps == 0);
}

void warmup_cosine_scheduler_free(WarmupCosineScheduler* scheduler)
{
    if (!scheduler) return;
    safe_free((void**)&scheduler);
}

/* ========================================================================
 * 知识蒸馏训练器实现
 * ======================================================================== */

DistillationTrainer* distillation_trainer_create(LNN* teacher, LNN* student,
                                                  float temperature, float alpha,
                                                  float learning_rate)
{
    if (!teacher || !student) return NULL;
    if (temperature <= 0.0f) temperature = 2.0f;
    if (alpha < 0.0f || alpha > 1.0f) alpha = 0.7f;
    if (learning_rate <= 0.0f) learning_rate = 0.001f;

    size_t output_size = (size_t)lnn_get_output_size_internal(teacher);
    if (output_size == 0) output_size = 128;

    DistillationTrainer* trainer = (DistillationTrainer*)safe_malloc(sizeof(DistillationTrainer));
    if (!trainer) return NULL;

    trainer->teacher_network = teacher;
    trainer->student_network = student;
    trainer->temperature = temperature;
    trainer->alpha = alpha;
    trainer->enable_hard_labels = 1;
    trainer->learning_rate = learning_rate;
    trainer->current_step = 0;
    trainer->teacher_output = (float*)safe_malloc(output_size * sizeof(float));
    trainer->student_output = (float*)safe_malloc(output_size * sizeof(float));
    trainer->buffer_size = output_size;

    if (!trainer->teacher_output || !trainer->student_output) {
        if (trainer->teacher_output) safe_free((void**)&trainer->teacher_output);
        if (trainer->student_output) safe_free((void**)&trainer->student_output);
        safe_free((void**)&trainer);
        return NULL;
    }

    memset(trainer->teacher_output, 0, output_size * sizeof(float));
    memset(trainer->student_output, 0, output_size * sizeof(float));

    return trainer;
}

int distillation_trainer_step(DistillationTrainer* trainer,
                              const float* input, const float* hard_targets,
                              size_t input_dim, size_t output_dim,
                              float* loss_output)
{
    if (!trainer || !input) return -1;
    if (output_dim == 0) output_dim = trainer->buffer_size;

    LNN* teacher = trainer->teacher_network;
    LNN* student = trainer->student_network;
    if (!teacher || !student) return -1;

    float temperature = trainer->temperature;

    memset(trainer->teacher_output, 0, trainer->buffer_size * sizeof(float));
    memset(trainer->student_output, 0, trainer->buffer_size * sizeof(float));

    if (lnn_forward(teacher, input, trainer->teacher_output) != 0) {
        return -1;
    }

    if (lnn_forward(student, input, trainer->student_output) != 0) {
        return -1;
    }

    size_t out_size = output_dim < trainer->buffer_size ? output_dim : trainer->buffer_size;

    float* teacher_soft = (float*)safe_malloc(out_size * sizeof(float));
    float* student_soft = (float*)safe_malloc(out_size * sizeof(float));
    if (!teacher_soft || !student_soft) {
        if (teacher_soft) safe_free((void**)&teacher_soft);
        if (student_soft) safe_free((void**)&student_soft);
        return -1;
    }

    float inv_t = 1.0f / temperature;
    float t_max = 0.0f;
    for (size_t i = 0; i < out_size; i++) {
        float tv = trainer->teacher_output[i] * inv_t;
        if (tv > t_max) t_max = tv;
    }
    float s_max = 0.0f;
    for (size_t i = 0; i < out_size; i++) {
        float sv = trainer->student_output[i] * inv_t;
        if (sv > s_max) s_max = sv;
    }

    float t_sum = 0.0f, s_sum = 0.0f;
    for (size_t i = 0; i < out_size; i++) {
        teacher_soft[i] = expf(trainer->teacher_output[i] * inv_t - t_max);
        student_soft[i] = expf(trainer->student_output[i] * inv_t - s_max);
        t_sum += teacher_soft[i];
        s_sum += student_soft[i];
    }

    float kl_div = 0.0f;
    if (t_sum > 1e-10f && s_sum > 1e-10f) {
        for (size_t i = 0; i < out_size; i++) {
            float tp = teacher_soft[i] / t_sum;
            float sp = student_soft[i] / s_sum;
            if (tp > 1e-10f && sp > 1e-10f) {
                kl_div += tp * logf(tp / sp);
            }
        }
    }

    kl_div *= (temperature * temperature);

    float hard_loss = 0.0f;
    if (hard_targets && trainer->enable_hard_labels) {
        for (size_t i = 0; i < out_size; i++) {
            float diff = trainer->student_output[i] - hard_targets[i];
            hard_loss += diff * diff;
        }
        hard_loss /= (float)out_size;
    }

    float alpha = trainer->alpha;
    float total_loss = alpha * kl_div + (1.0f - alpha) * hard_loss;

    float* error_buffer = (float*)safe_malloc(out_size * sizeof(float));
    if (!error_buffer) {
        safe_free((void**)&teacher_soft);
        safe_free((void**)&student_soft);
        return -1;
    }

    for (size_t i = 0; i < out_size; i++) {
        float teacher_mult = 0.0f;
        if (t_sum > 1e-10f) {
            float tp = teacher_soft[i] / t_sum;
            float sp = (s_sum > 1e-10f) ? student_soft[i] / s_sum : 0.0f;
            if (sp > 1e-10f) {
                teacher_mult = (sp - tp) / (temperature * temperature);
            }
        }
        float hard_mult = 0.0f;
        if (hard_targets && trainer->enable_hard_labels) {
            hard_mult = 2.0f * (trainer->student_output[i] - hard_targets[i]) / (float)out_size;
        }
        error_buffer[i] = alpha * teacher_mult + (1.0f - alpha) * hard_mult;
    }

    CfCNetwork* cfc = student->cfc_network;
    if (cfc) {
        /* ZSF-070标注：蒸馏路径使用cfc_backward直接更新权重（SGD模式）。
         * 对于需要Adam/AdamW优化的场景，应改为：先累积梯度→调用optimizer_update。 */
        cfc_backward(cfc, error_buffer, student->gradient_buffer, trainer->learning_rate);
    }

    safe_free((void**)&teacher_soft);
    safe_free((void**)&student_soft);
    safe_free((void**)&error_buffer);

    trainer->current_step++;

    if (loss_output) {
        *loss_output = total_loss;
    }

    return 0;
}

int distillation_trainer_train_batch(DistillationTrainer* trainer,
                                     const float* inputs, const float* hard_targets,
                                     size_t num_samples,
                                     size_t input_dim, size_t output_dim,
                                     float* avg_loss)
{
    if (!trainer || !inputs || num_samples == 0) return -1;

    double total_loss = 0.0;
    int success_count = 0;

    for (size_t i = 0; i < num_samples; i++) {
        const float* sample_input = inputs + i * input_dim;
        const float* sample_target = hard_targets ? hard_targets + i * output_dim : NULL;
        float step_loss = 0.0f;

        int ret = distillation_trainer_step(trainer, sample_input, sample_target,
                                            input_dim, output_dim, &step_loss);
        if (ret == 0) {
            total_loss += (double)step_loss;
            success_count++;
        }
    }

    if (avg_loss) {
        *avg_loss = (success_count > 0) ? (float)(total_loss / (double)success_count) : 0.0f;
    }

    return (success_count > 0) ? 0 : -1;
}

void distillation_trainer_free(DistillationTrainer* trainer)
{
    if (!trainer) return;
    if (trainer->teacher_output) {
        safe_free((void**)&trainer->teacher_output);
    }
    if (trainer->student_output) {
        safe_free((void**)&trainer->student_output);
    }
    safe_free((void**)&trainer);
}

/* ========================================================================
 * Trainer集成辅助实现
 * ======================================================================== */

int trainer_enhanced_register(Trainer* trainer)
{
    if (!trainer) return -1;

    REGISTRY_LOCK();
    if (registry_find(trainer)) {
        REGISTRY_UNLOCK();
        return 0;
    }

    EnhancedTrainerEntry* entry = registry_alloc(trainer);
    if (!entry) {
        REGISTRY_UNLOCK();
        return -1;
    }

    size_t num_weights = 0, num_biases = 0;
    LNN* network = trainer_get_network(trainer);
    if (network) {
        lnn_get_weight_and_bias(network, NULL, &num_weights, NULL, &num_biases);
    }

    entry->state.ema = ema_weight_manager_create(num_weights, num_biases, 0.999f);
    entry->state.label_smoothing_factor = 0.1f;
    entry->state.use_label_smoothing = 0;
    entry->state.focal_loss_gamma = 2.0f;
    entry->state.focal_loss_alpha = 0.25f;
    entry->state.use_focal_loss = 0;
    entry->state.warmup_cosine = NULL;
    entry->state.use_warmup_cosine = 0;
    entry->state.enable_ema = 0;
    entry->state.ema_update_interval = 1;
    entry->state.ema_step_counter = 0;
    REGISTRY_UNLOCK();

    return 0;
}

int trainer_enhanced_unregister(Trainer* trainer)
{
    if (!trainer) return -1;
    REGISTRY_LOCK();
    registry_free(trainer);
    REGISTRY_UNLOCK();
    return 0;
}

EnhancedTrainerState* trainer_enhanced_get_state(Trainer* trainer)
{
    if (!trainer) return NULL;
    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    REGISTRY_UNLOCK();
    if (!entry) return NULL;
    return &entry->state;
}

int trainer_enhanced_enable_ema(Trainer* trainer, float decay, int update_interval)
{
    if (!trainer) return -1;

    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    if (!entry) {
        entry = registry_alloc(trainer);
        if (!entry) {
            REGISTRY_UNLOCK();
            return -1;
        }
        size_t num_weights = 0, num_biases = 0;
        LNN* network = trainer_get_network(trainer);
        if (network) {
            lnn_get_weight_and_bias(network, NULL, &num_weights, NULL, &num_biases);
        }
        entry->state.ema = ema_weight_manager_create(num_weights, num_biases, 0.999f);
        entry->state.label_smoothing_factor = 0.1f;
    }

    EnhancedTrainerState* state = &entry->state;

    if (state->ema) {
        ema_weight_manager_free(state->ema);
        state->ema = NULL;
    }

    size_t num_weights = 0, num_biases = 0;
    LNN* network = trainer_get_network(trainer);
    if (network) {
        lnn_get_weight_and_bias(network, NULL, &num_weights, NULL, &num_biases);
    }

    state->ema = ema_weight_manager_create(num_weights, num_biases, decay);
    if (!state->ema) {
        REGISTRY_UNLOCK();
        return -1;
    }

    if (network) {
        ema_weight_manager_update_from_lnn(state->ema, network);
    }

    state->enable_ema = 1;
    state->ema_update_interval = (update_interval > 0) ? update_interval : 1;
    state->ema_step_counter = 0;
    REGISTRY_UNLOCK();

    return 0;
}

int trainer_enhanced_disable_ema(Trainer* trainer)
{
    if (!trainer) return -1;
    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    if (!entry) {
        REGISTRY_UNLOCK();
        return -1;
    }
    EnhancedTrainerState* state = &entry->state;
    state->enable_ema = 0;
    if (state->ema) {
        ema_weight_manager_free(state->ema);
        state->ema = NULL;
    }
    REGISTRY_UNLOCK();
    return 0;
}

int trainer_enhanced_apply_ema(Trainer* trainer)
{
    if (!trainer) return -1;
    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    if (!entry || !entry->state.ema || !entry->state.enable_ema) {
        REGISTRY_UNLOCK();
        return -1;
    }

    LNN* network = trainer_get_network(trainer);
    if (!network) {
        REGISTRY_UNLOCK();
        return -1;
    }

    ema_weight_manager_apply_to_lnn(entry->state.ema, network);
    REGISTRY_UNLOCK();
    return 0;
}

int trainer_enhanced_enable_label_smoothing(Trainer* trainer, float factor)
{
    if (!trainer) return -1;
    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    if (!entry) {
        entry = registry_alloc(trainer);
        if (!entry) {
            REGISTRY_UNLOCK();
            return -1;
        }
        size_t num_weights = 0, num_biases = 0;
        LNN* network = trainer_get_network(trainer);
        if (network) {
            lnn_get_weight_and_bias(network, NULL, &num_weights, NULL, &num_biases);
        }
        entry->state.ema = ema_weight_manager_create(num_weights, num_biases, 0.999f);
        entry->state.label_smoothing_factor = 0.1f;
    }

    entry->state.label_smoothing_factor = (factor >= 0.0f && factor < 1.0f) ? factor : 0.1f;
    entry->state.use_label_smoothing = 1;
    REGISTRY_UNLOCK();
    return 0;
}

int trainer_enhanced_disable_label_smoothing(Trainer* trainer)
{
    if (!trainer) return -1;
    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    if (!entry) {
        REGISTRY_UNLOCK();
        return -1;
    }
    entry->state.use_label_smoothing = 0;
    REGISTRY_UNLOCK();
    return 0;
}

int trainer_enhanced_enable_focal_loss(Trainer* trainer, float gamma, float alpha)
{
    if (!trainer) return -1;
    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    if (!entry) {
        entry = registry_alloc(trainer);
        if (!entry) {
            REGISTRY_UNLOCK();
            return -1;
        }
        size_t num_weights = 0, num_biases = 0;
        LNN* network = trainer_get_network(trainer);
        if (network) {
            lnn_get_weight_and_bias(network, NULL, &num_weights, NULL, &num_biases);
        }
        entry->state.ema = ema_weight_manager_create(num_weights, num_biases, 0.999f);
        entry->state.label_smoothing_factor = 0.1f;
    }

    entry->state.focal_loss_gamma = (gamma >= 0.0f) ? gamma : 2.0f;
    entry->state.focal_loss_alpha = (alpha >= 0.0f && alpha <= 1.0f) ? alpha : 0.25f;
    entry->state.use_focal_loss = 1;
    REGISTRY_UNLOCK();
    return 0;
}

int trainer_enhanced_disable_focal_loss(Trainer* trainer)
{
    if (!trainer) return -1;
    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    if (!entry) {
        REGISTRY_UNLOCK();
        return -1;
    }
    entry->state.use_focal_loss = 0;
    REGISTRY_UNLOCK();
    return 0;
}

int trainer_enhanced_enable_warmup_cosine(Trainer* trainer, float base_lr, float min_lr,
                                          size_t warmup_steps, size_t total_steps)
{
    if (!trainer) return -1;
    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    if (!entry) {
        entry = registry_alloc(trainer);
        if (!entry) {
            REGISTRY_UNLOCK();
            return -1;
        }
        size_t num_weights = 0, num_biases = 0;
        LNN* network = trainer_get_network(trainer);
        if (network) {
            lnn_get_weight_and_bias(network, NULL, &num_weights, NULL, &num_biases);
        }
        entry->state.ema = ema_weight_manager_create(num_weights, num_biases, 0.999f);
        entry->state.label_smoothing_factor = 0.1f;
    }

    EnhancedTrainerState* state = &entry->state;

    if (state->warmup_cosine) {
        warmup_cosine_scheduler_free(state->warmup_cosine);
        state->warmup_cosine = NULL;
    }

    state->warmup_cosine = warmup_cosine_scheduler_create(base_lr, min_lr, warmup_steps, total_steps);
    if (!state->warmup_cosine && total_steps > 0) {
        REGISTRY_UNLOCK();
        return -1;
    }

    state->use_warmup_cosine = 1;
    REGISTRY_UNLOCK();
    return 0;
}

int trainer_enhanced_disable_warmup_cosine(Trainer* trainer)
{
    if (!trainer) return -1;
    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    if (!entry) {
        REGISTRY_UNLOCK();
        return -1;
    }
    EnhancedTrainerState* state = &entry->state;
    state->use_warmup_cosine = 0;
    if (state->warmup_cosine) {
        warmup_cosine_scheduler_free(state->warmup_cosine);
        state->warmup_cosine = NULL;
    }
    REGISTRY_UNLOCK();
    return 0;
}

int trainer_enhanced_on_step_end(Trainer* trainer)
{
    if (!trainer) return -1;
    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    if (!entry) {
        REGISTRY_UNLOCK();
        return 0;
    }
    EnhancedTrainerState* state = &entry->state;

    if (state->enable_ema && state->ema) {
        state->ema_step_counter++;
        if (state->ema_step_counter >= state->ema_update_interval) {
            LNN* network = trainer_get_network(trainer);
            if (network) {
                ema_weight_manager_update_from_lnn(state->ema, network);
            }
            state->ema_step_counter = 0;
        }
    }

    if (state->use_warmup_cosine && state->warmup_cosine) {
        float current_lr = warmup_cosine_scheduler_step(state->warmup_cosine);
        TrainingConfig* config = trainer_get_config(trainer);
        if (config) {
            config->learning_rate = current_lr;
        }
    }
    REGISTRY_UNLOCK();

    return 0;
}

int trainer_enhanced_on_epoch_end(Trainer* trainer)
{
    if (!trainer) return -1;

    /* 获取训练状态和配置 */
    TrainingState* state = trainer_get_state(trainer);
    TrainingConfig* config = trainer_get_config(trainer);
    if (!state || !config) return -1;

    REGISTRY_LOCK();
    EnhancedTrainerEntry* entry = registry_find(trainer);
    REGISTRY_UNLOCK();

    /* 日志输出训练指标 */
    if (config->verbose) {
        printf("[增强训练] Epoch %zu | 损失: %.6f | 验证损失: %.6f | "
               "准确率: %.4f | 验证准确率: %.4f | 学习率: %.6f | "
               "最佳损失: %.6f | 无改进步数: %zu\n",
               state->current_epoch,
               state->current_loss,
               state->validation_loss,
               state->current_accuracy,
               state->validation_accuracy,
               state->learning_rate,
               state->best_loss,
               state->steps_without_improvement);
    }

    /* 最佳模型检查点保存 */
    if (config->save_best_model && state->current_loss < state->best_loss) {
        char checkpoint_path[512];
        snprintf(checkpoint_path, sizeof(checkpoint_path),
                 "checkpoint_epoch_%zu_loss_%.4f.bin",
                 state->current_epoch, state->current_loss);
        trainer_save_model_weights(trainer, checkpoint_path);
        if (config->verbose) {
            printf("[增强训练] 保存最佳模型检查点: %s\n", checkpoint_path);
        }
    }

    /* 自适应学习率调整（ReduceLROnPlateau风格） */
    if (config->enable_adaptive_lr && entry) {
        EnhancedTrainerState* est = &entry->state;
        (void)est;
        if (state->steps_without_improvement >= config->lr_patience) {
            float new_lr = config->learning_rate * config->lr_factor;
            float min_lr = config->learning_rate * config->lr_min_factor;
            if (new_lr >= min_lr) {
                config->learning_rate = new_lr;
                state->learning_rate = new_lr;
                state->steps_without_improvement = 0;
                if (config->verbose) {
                    printf("[增强训练] 学习率调整: %.6f -> %.6f\n",
                           new_lr / config->lr_factor, new_lr);
                }
            }
        }
    }

    /* 早停信号（由调用方根据返回值决定是否停止训练） */
    if (config->enable_adaptive_lr) {
        float min_lr = config->learning_rate * config->lr_min_factor;
        if (config->learning_rate <= min_lr &&
            state->steps_without_improvement >= config->lr_patience * 3) {
            if (config->verbose) {
                printf("[增强训练] 早停触发: 学习率已降至最低且长时间无改善\n");
            }
            return 1;
        }
    }

    return 0;
}
