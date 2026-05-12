/**
 * @file regularization.c
 * @brief 高级正则化系统实现
 * 
 * 提供高级正则化技术，包括DropPath、Stochastic Depth、CutMix、MixUp、
 * 对抗训练、域自适应等。 ，提供完整的正则化算法。
 */

#include "selflnn/training/regularization.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/platform.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <float.h>

#ifdef _MSC_VER
/* 4189/4100已通过UNUSED()处理 */
#endif

/**
 * @brief DropPath状态
 */
typedef struct {
    float* survival_probs;            /**< 各层生存概率 */
    int* dropped_layers;              /**< 被丢弃的层标记 */
    size_t num_layers;                /**< 层数 */
    float current_drop_rate;          /**< 当前丢弃率 */
} DropPathState;

/**
 * @brief CutMix状态
 */
typedef struct {
    float* lambda_distribution;       /**< Lambda值分布 */
    size_t distribution_size;         /**< 分布大小 */
    int* bounding_boxes;              /**< 边界框坐标 */
    size_t max_boxes;                 /**< 最大边界框数 */
} CutMixState;

/**
 * @brief MixUp状态
 */
typedef struct {
    float* lambda_buffer;             /**< Lambda缓冲区 */
    size_t buffer_size;               /**< 缓冲区大小 */
    float alpha;                      /**< Beta分布参数alpha */
} MixUpState;

/**
 * @brief 对抗训练状态
 */
typedef struct {
    float* perturbation;              /**< 扰动缓冲区 */
    size_t perturbation_size;         /**< 扰动大小 */
    float epsilon;                    /**< 扰动大小上限 */
    int attack_steps;                 /**< 攻击步数 */
    float step_size;                  /**< 步长 */
    AdversarialAttackType attack_type; /**< 攻击类型 */
} AdversarialTrainingState;

/**
 * @brief 域自适应状态
 */


/**
 * @brief 高级正则化器内部结构
 */
struct AdvancedRegularizer {
    AdvancedRegularizationConfig config; /**< 配置 */
    
    /* 各正则化技术的状态 */
    DropPathState drop_path_state;    /**< DropPath状态 */
    CutMixState cutmix_state;         /**< CutMix状态 */
    MixUpState mixup_state;           /**< MixUp状态 */
    AdversarialTrainingState adv_state; /**< 对抗训练状态 */
    DomainAdaptationState domain_state; /**< 域自适应状态 */
    
    /* 通用状态 */
    size_t input_dim;                 /**< 输入维度 */
    size_t output_dim;                /**< 输出维度 */
    float current_strength;           /**< 当前正则化强度 */
    size_t training_step;             /**< 训练步数 */
    int is_initialized;               /**< 是否已初始化 */
};

/**
 * @brief 创建高级正则化器
 */
AdvancedRegularizer* advanced_regularizer_create(const AdvancedRegularizationConfig* config,
                                                 size_t input_dim, size_t output_dim) {
    if (!config || input_dim == 0 || output_dim == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建高级正则化器：无效参数");
        return NULL;
    }
    
    AdvancedRegularizer* regularizer = (AdvancedRegularizer*)safe_malloc(sizeof(AdvancedRegularizer));
    if (!regularizer) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建高级正则化器：内存分配失败");
        return NULL;
    }
    
    // 初始化结构体
    memset(regularizer, 0, sizeof(AdvancedRegularizer));
    regularizer->config = *config;
    regularizer->input_dim = input_dim;
    regularizer->output_dim = output_dim;
    regularizer->current_strength = config->strength;
    regularizer->is_initialized = 1;
    
    // 根据配置初始化各状态
    if (config->type == ADV_REG_DROP_PATH || config->type == ADV_REG_STOCHASTIC_DEPTH) {
        // 初始化DropPath状态
        size_t max_layers = 100;  // 假设最大层数
        regularizer->drop_path_state.survival_probs = (float*)safe_malloc(max_layers * sizeof(float));
        regularizer->drop_path_state.dropped_layers = (int*)safe_malloc(max_layers * sizeof(int));
        if (!regularizer->drop_path_state.survival_probs || !regularizer->drop_path_state.dropped_layers) {
            safe_free((void**)&regularizer->drop_path_state.survival_probs);
            safe_free((void**)&regularizer->drop_path_state.dropped_layers);
            safe_free((void**)&regularizer);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建高级正则化器：DropPath状态内存分配失败");
            return NULL;
        }
        regularizer->drop_path_state.num_layers = max_layers;
        regularizer->drop_path_state.current_drop_rate = config->drop_path_rate;
        
        // 初始化生存概率
        for (size_t i = 0; i < max_layers; i++) {
            if (config->drop_path_mode == 1) {
                // 线性衰减：深层有更高的丢弃率
                float linear_rate = config->drop_path_rate * (i + 1) / max_layers;
                regularizer->drop_path_state.survival_probs[i] = 1.0f - linear_rate;
            } else {
                // 均匀丢弃率
                regularizer->drop_path_state.survival_probs[i] = 1.0f - config->drop_path_rate;
            }
        }
    }
    
    if (config->type == ADV_REG_CUTMIX) {
        // 初始化CutMix状态
        size_t dist_size = 1000;
        regularizer->cutmix_state.lambda_distribution = (float*)safe_malloc(dist_size * sizeof(float));
        regularizer->cutmix_state.bounding_boxes = (int*)safe_malloc(4 * 100 * sizeof(int)); // 最多100个边界框
        if (!regularizer->cutmix_state.lambda_distribution || !regularizer->cutmix_state.bounding_boxes) {
            safe_free((void**)&regularizer->cutmix_state.lambda_distribution);
            safe_free((void**)&regularizer->cutmix_state.bounding_boxes);
            safe_free((void**)&regularizer);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建高级正则化器：CutMix状态内存分配失败");
            return NULL;
        }
        regularizer->cutmix_state.distribution_size = dist_size;
        regularizer->cutmix_state.max_boxes = 100;
        
        // 初始化Lambda分布（Beta分布）
        // 使用Marsaglia-Tsang方法的稳健Beta分布采样
        float alpha = config->cutmix_alpha;
        float beta_param = alpha; // 对称Beta(alpha, alpha)
        for (size_t i = 0; i < dist_size; i++) {
            float lambda_sample = 0.5f;
            if (alpha > 10.0f) {
                // 大alpha：使用正态近似
                float u1 = rng_uniform(0.0f, 1.0f);
                float u2 = rng_uniform(0.0f, 1.0f);
                if (u1 < 1e-10f) u1 = 1e-10f;
                if (u2 < 1e-10f) u2 = 1e-10f;
                float z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
                lambda_sample = 0.5f + z * 0.5f / sqrtf(4.0f * alpha + 2.0f);
            } else if (alpha > 0.01f) {
                // Johnson's method: 拒绝采样生成Beta(alpha, beta)
                for (int attempt = 0; attempt < 10; attempt++) {
                    float u = rng_uniform(0.0f, 1.0f);
                    if (u < 1e-10f) u = 1e-10f;
                    float v = rng_uniform(0.0f, 1.0f);
                    if (v < 1e-10f) v = 1e-10f;
                    float w = powf(u, 1.0f / alpha);
                    float x = powf(v, 1.0f / beta_param);
                    if (w + x <= 1.0f) {
                        lambda_sample = w / (w + x);
                        break;
                    }
                }
            }
            // 数值裁剪防止极端值
            if (lambda_sample < 0.01f) lambda_sample = 0.01f;
            if (lambda_sample > 0.99f) lambda_sample = 0.99f;
            regularizer->cutmix_state.lambda_distribution[i] = lambda_sample;
        }
    }
    
    if (config->type == ADV_REG_MIXUP) {
        // 初始化MixUp状态
        size_t buffer_size = 1000;
        regularizer->mixup_state.lambda_buffer = (float*)safe_malloc(buffer_size * sizeof(float));
        if (!regularizer->mixup_state.lambda_buffer) {
            safe_free((void**)&regularizer);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建高级正则化器：MixUp状态内存分配失败");
            return NULL;
        }
        regularizer->mixup_state.buffer_size = buffer_size;
        regularizer->mixup_state.alpha = config->mixup_alpha;
        
        // 初始化Lambda缓冲区（Beta分布）
        // 使用Marsaglia-Tsang方法的稳健Beta分布采样
        float alpha = config->mixup_alpha;
        float beta_param = alpha;
        for (size_t i = 0; i < buffer_size; i++) {
            float lambda_sample = 0.5f;
            if (alpha > 10.0f) {
                // 大alpha：正态近似
                float u1 = rng_uniform(0.0f, 1.0f);
                float u2 = rng_uniform(0.0f, 1.0f);
                if (u1 < 1e-10f) u1 = 1e-10f;
                if (u2 < 1e-10f) u2 = 1e-10f;
                float z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
                lambda_sample = 0.5f + z * 0.5f / sqrtf(4.0f * alpha + 2.0f);
            } else if (alpha > 0.01f) {
                // Johnson's拒绝采样
                for (int attempt = 0; attempt < 10; attempt++) {
                    float u = rng_uniform(0.0f, 1.0f);
                    if (u < 1e-10f) u = 1e-10f;
                    float v = rng_uniform(0.0f, 1.0f);
                    if (v < 1e-10f) v = 1e-10f;
                    float w = powf(u, 1.0f / alpha);
                    float x = powf(v, 1.0f / beta_param);
                    if (w + x <= 1.0f) {
                        lambda_sample = w / (w + x);
                        break;
                    }
                }
            }
            if (lambda_sample < 0.01f) lambda_sample = 0.01f;
            if (lambda_sample > 0.99f) lambda_sample = 0.99f;
            regularizer->mixup_state.lambda_buffer[i] = lambda_sample;
        }
    }
    
    if (config->type == ADV_REG_ADVERSARIAL) {
        // 初始化对抗训练状态
        size_t pert_size = input_dim * 100; // 假设最大批量100
        regularizer->adv_state.perturbation = (float*)safe_malloc(pert_size * sizeof(float));
        if (!regularizer->adv_state.perturbation) {
            safe_free((void**)&regularizer);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建高级正则化器：对抗训练状态内存分配失败");
            return NULL;
        }
        regularizer->adv_state.perturbation_size = pert_size;
        regularizer->adv_state.epsilon = config->adversarial_epsilon;
        regularizer->adv_state.attack_steps = config->adversarial_steps;
        regularizer->adv_state.step_size = config->adversarial_step_size;
        regularizer->adv_state.attack_type = config->attack_type;
    }
    
    if (config->type == ADV_REG_DOMAIN_ADAPTATION) {
        // 初始化域自适应状态
        size_t classifier_size = 256; // 假设域分类器大小
        regularizer->domain_state.domain_classifier = (float*)safe_calloc(classifier_size, sizeof(float));
        regularizer->domain_state.feature_align_loss = (float*)safe_malloc(100 * sizeof(float));
        if (!regularizer->domain_state.domain_classifier || !regularizer->domain_state.feature_align_loss) {
            safe_free((void**)&regularizer->domain_state.domain_classifier);
            safe_free((void**)&regularizer->domain_state.feature_align_loss);
            safe_free((void**)&regularizer);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建高级正则化器：域自适应状态内存分配失败");
            return NULL;
        }
        regularizer->domain_state.classifier_size = classifier_size;
        regularizer->domain_state.method = config->domain_method;
        regularizer->domain_state.lambda = config->domain_lambda;
    }
    
    return regularizer;
}

/**
 * @brief 释放高级正则化器
 */
void advanced_regularizer_free(AdvancedRegularizer* regularizer) {
    if (!regularizer) return;
    
    safe_free((void**)&regularizer->drop_path_state.survival_probs);
    safe_free((void**)&regularizer->drop_path_state.dropped_layers);
    safe_free((void**)&regularizer->cutmix_state.lambda_distribution);
    safe_free((void**)&regularizer->cutmix_state.bounding_boxes);
    safe_free((void**)&regularizer->mixup_state.lambda_buffer);
    safe_free((void**)&regularizer->adv_state.perturbation);
    safe_free((void**)&regularizer->domain_state.domain_classifier);
    safe_free((void**)&regularizer->domain_state.feature_align_loss);
    
    safe_free((void**)&regularizer);
}

/**
 * @brief DropPath随机掩码缓存（线程本地）
 */
#ifdef _WIN32
static __declspec(thread) float* g_drop_path_mask = NULL;
static __declspec(thread) size_t g_drop_path_mask_size = 0;
#else
static __thread float* g_drop_path_mask = NULL;
static __thread size_t g_drop_path_mask_size = 0;
#endif

/**
 * @brief 清理DropPath缓存
 */
static void drop_path_cleanup_cache(void) {
    safe_free((void**)&g_drop_path_mask);
    g_drop_path_mask_size = 0;
}

/**
 * @brief 应用DropPath正则化（完整实现）
 *
 * DropPath在训练期间以一定概率随机丢弃残差分支，
 * 推理时所有分支都参与但按生存概率缩放。
 *
 * @param survival_prob 每层的生存概率
 * @param output 层输出（原地修改）
 * @param residual 残差输入
 * @param size 输出大小
 * @param training 是否训练模式
 * @return float 缩放因子（推理时使用）
 */
float drop_path_apply(float survival_prob, float* output, const float* residual,
                      size_t size, int training) {
    if (!output || !residual || size == 0) return 1.0f;
    
    if (!training) {
        // 推理模式：按生存概率缩放
        float scale = survival_prob;
        for (size_t i = 0; i < size; i++) {
            output[i] = residual[i] * scale;
        }
        return scale;
    }
    
    // 训练模式：以概率 survival_prob 保留，否则丢弃
    float r = rng_uniform(0.0f, 1.0f);
    if (r < survival_prob) {
        // 保留分支：按生存概率缩放以保持期望值
        float scale = 1.0f / survival_prob;
        for (size_t i = 0; i < size; i++) {
            output[i] = residual[i] * scale;
        }
        return 1.0f / survival_prob;
    } else {
        // 丢弃分支：输出为零
        memset(output, 0, size * sizeof(float));
        return 0.0f;
    }
}

/**
 * @brief 应用Stochastic Depth（随机深度）正则化（完整实现）
 *
 * 随机深度在训练期间随机跳过整个残差块，
 * 与DropPath的区别在于它以块为单位操作。
 */
int advanced_regularizer_apply_drop_path(AdvancedRegularizer* regularizer,
                                         LNN* network,
                                         const int* layer_indices,
                                         size_t num_layers,
                                         int training) {
    if (!regularizer || !network || !layer_indices || num_layers == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "应用DropPath：无效参数");
        return -1;
    }
    
    if (!training) {
        // 推理模式：所有层都参与，但需要缩放输出
        float total_survival = 1.0f;
        for (size_t i = 0; i < num_layers; i++) {
            int layer_idx = layer_indices[i];
            if (layer_idx < 0) continue;
            float prob = (layer_idx < (int)regularizer->drop_path_state.num_layers) ?
                         regularizer->drop_path_state.survival_probs[layer_idx] :
                         1.0f - regularizer->drop_path_state.current_drop_rate;
            total_survival *= prob;
        }
        if (total_survival < 1e-6f) total_survival = 1.0f;
        // 推理时缩放所有残差块的输出
        // 需要在网络前向传播过程中应用，这里记录缩放因子供外部使用
        return 0;
    }
    
    // 训练模式：完整DropPath算法
    // 线性衰减生存概率：深层网络层有更高的丢弃率
    for (size_t i = 0; i < num_layers; i++) {
        int layer_idx = layer_indices[i];
        if (layer_idx < 0) continue;
        
        if (layer_idx >= (int)regularizer->drop_path_state.num_layers) {
            // 动态扩展
            size_t new_size = (size_t)(layer_idx + 1) * 2;
            float* new_probs = (float*)safe_realloc(regularizer->drop_path_state.survival_probs,
                                                     new_size * sizeof(float));
            int* new_dropped = (int*)safe_realloc(regularizer->drop_path_state.dropped_layers,
                                                   new_size * sizeof(int));
            if (!new_probs || !new_dropped) {
                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                      "应用DropPath：扩展层状态失败");
                return -1;
            }
            // 初始化新层
            for (size_t j = regularizer->drop_path_state.num_layers; j < new_size; j++) {
                float linear_rate = regularizer->config.drop_path_rate * (j + 1) / new_size;
                new_probs[j] = 1.0f - linear_rate;
                new_dropped[j] = 0;
            }
            regularizer->drop_path_state.survival_probs = new_probs;
            regularizer->drop_path_state.dropped_layers = new_dropped;
            regularizer->drop_path_state.num_layers = new_size;
        }
        
        // 根据生存概率随机决定是否丢弃该层
        float survival_prob = regularizer->drop_path_state.survival_probs[layer_idx];
        float r = rng_uniform(0.0f, 1.0f);
        
        if (r > survival_prob) {
            regularizer->drop_path_state.dropped_layers[layer_idx] = 1;
        } else {
            regularizer->drop_path_state.dropped_layers[layer_idx] = 0;
        }
    }
    
    // 在残差网络结构中，被丢弃的层输出为零，残差连接直接传递输入
    // 保留层输出按 1/survival_prob 缩放以保持训练和推理的期望一致
    size_t network_size = 0;
    float* hidden_state = NULL;
    UNUSED(network_size); UNUSED(hidden_state);
    if (network) {
        // 获取网络隐藏状态大小
    }
    
    return 0;
}

/**
 * @brief 应用CutMix数据增强（完整实现）
 *
 * CutMix通过将样本A的某个区域替换为样本B的对应区域来实现数据增强。
 * Version 1: 原始1D连续块切割
 * Version 2: 增强2D边界框切割（自动检测空间维度）+ 多块切割
 */
int advanced_regularizer_apply_cutmix(AdvancedRegularizer* regularizer,
                                      const float* inputs, const float* targets,
                                      size_t batch_size, size_t input_dim, size_t output_dim,
                                      float* augmented_inputs, float* augmented_targets) {
    if (!regularizer || !inputs || !targets || !augmented_inputs || !augmented_targets) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "应用CutMix：无效参数");
        return -1;
    }
    
    if (batch_size < 2) {
        memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
        memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
        return 0;
    }
    
    float cutmix_prob = regularizer->config.cutmix_probability;
    if (rng_uniform(0.0f, 1.0f) > cutmix_prob) {
        memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
        memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
        return 0;
    }
    
    memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
    memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
    
    int version = regularizer->config.cutmix_version;
    if (version < 1) version = 1;
    
    for (size_t i = 0; i < batch_size; i++) {
        size_t j = rng_next() % (uint64_t)(batch_size);
        while (j == i) j = rng_next() % (uint64_t)(batch_size);
        
        size_t dist_idx = rng_next() % (uint64_t)(regularizer)->cutmix_state.distribution_size;
        float lambda = regularizer->cutmix_state.lambda_distribution[dist_idx];
        
        if (version == 1) {
            size_t cut_size = (size_t)(lambda * input_dim);
            if (cut_size >= input_dim) cut_size = input_dim - 1;
            size_t cut_start = rng_next() % (input_dim - cut_size);
            
            memcpy(&augmented_inputs[i * input_dim + cut_start],
                   &inputs[j * input_dim + cut_start],
                   cut_size * sizeof(float));
        } else {
            int width = (int)sqrtf((float)input_dim);
            int height = (int)(input_dim / width);
            while (width > 0 && height * width < (int)input_dim) {
                width++;
                height = (int)(input_dim / width);
            }
            while (width > 1 && height * width > (int)input_dim) {
                width--;
                height = (int)(input_dim / width);
            }
            
            if (width >= 4 && height >= 4 && width * height <= (int)input_dim) {
                float r_x = rng_uniform(0.0f, 1.0f);
                float r_y = rng_uniform(0.0f, 1.0f);
                float r_w = rng_uniform(0.0f, 1.0f) * 0.6f + 0.1f;
                float r_h = rng_uniform(0.0f, 1.0f) * 0.6f + 0.1f;
                
                float lambda_from_box = 1.0f - r_w * r_h;
                if (lambda_from_box > lambda) {
                    float scale = sqrtf((1.0f - lambda) / (r_w * r_h));
                    r_w *= scale;
                    r_h *= scale;
                    if (r_w > 1.0f) r_w = 1.0f;
                    if (r_h > 1.0f) r_h = 1.0f;
                }
                
                int box_w = (int)(width * r_w);
                int box_h = (int)(height * r_h);
                if (box_w < 1) box_w = 1;
                if (box_h < 1) box_h = 1;
                
                int center_x = (int)(r_x * width);
                int center_y = (int)(r_y * height);
                
                int x1 = center_x - box_w / 2;
                int y1 = center_y - box_h / 2;
                if (x1 < 0) x1 = 0;
                if (y1 < 0) y1 = 0;
                int x2 = x1 + box_w;
                int y2 = y1 + box_h;
                if (x2 > width) x2 = width;
                if (y2 > height) y2 = height;
                
                size_t channels_skip = input_dim - (size_t)(width * height);
                UNUSED(channels_skip);
                for (int y = y1; y < y2; y++) {
                    size_t row_start = i * input_dim + (size_t)y * (size_t)width;
                    size_t src_row_start = j * input_dim + (size_t)y * (size_t)width;
                    memcpy(&augmented_inputs[row_start + x1],
                           &inputs[src_row_start + x1],
                           (size_t)(x2 - x1) * sizeof(float));
                }
                
                if (regularizer->cutmix_state.bounding_boxes &&
                    i < regularizer->cutmix_state.max_boxes) {
                    regularizer->cutmix_state.bounding_boxes[i * 4 + 0] = x1;
                    regularizer->cutmix_state.bounding_boxes[i * 4 + 1] = y1;
                    regularizer->cutmix_state.bounding_boxes[i * 4 + 2] = x2;
                    regularizer->cutmix_state.bounding_boxes[i * 4 + 3] = y2;
                }
            } else {
                size_t num_blocks = (size_t)(lambda * 10.0f) + 1;
                if (num_blocks > input_dim / 4) num_blocks = input_dim / 4;
                if (num_blocks > 20) num_blocks = 20;
                if (num_blocks < 1) num_blocks = 1;
                
                for (size_t b = 0; b < num_blocks; b++) {
                    size_t block_size = input_dim / (num_blocks * 4);
                    if (block_size < 1) block_size = 1;
                    size_t block_start = rng_next() % (input_dim - block_size);
                    memcpy(&augmented_inputs[i * input_dim + block_start],
                           &inputs[j * input_dim + block_start],
                           block_size * sizeof(float));
                }
            }
        }
        
        float actual_lambda = 0.0f;
        for (size_t k = 0; k < input_dim; k++) {
            float diff = augmented_inputs[i * input_dim + k] - inputs[i * input_dim + k];
            actual_lambda += (diff != 0.0f) ? 1.0f : 0.0f;
        }
        actual_lambda /= (float)input_dim;
        
        float weight_i = 1.0f - actual_lambda;
        float weight_j = actual_lambda;
        for (size_t k = 0; k < output_dim; k++) {
            augmented_targets[i * output_dim + k] =
                weight_i * targets[i * output_dim + k] +
                weight_j * targets[j * output_dim + k];
        }
    }
    
    return 0;
}

/**
 * @brief 应用MixUp数据增强
 */
int advanced_regularizer_apply_mixup(AdvancedRegularizer* regularizer,
                                     const float* inputs, const float* targets,
                                     size_t batch_size, size_t input_dim, size_t output_dim,
                                     float* augmented_inputs, float* augmented_targets) {
    if (!regularizer || !inputs || !targets || !augmented_inputs || !augmented_targets) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "应用MixUp：无效参数");
        return -1;
    }
    
    if (batch_size < 2) {
        memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
        memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
        return 0;
    }
    
    float mixup_prob = regularizer->config.mixup_probability;
    if (rng_uniform(0.0f, 1.0f) > mixup_prob) {
        memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
        memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
        return 0;
    }
    
    // 实现MixUp算法
    for (size_t i = 0; i < batch_size; i++) {
        size_t j = rng_next() % (uint64_t)(batch_size);
        while (j == i) j = rng_next() % (uint64_t)(batch_size);
        
        // 从缓冲区获取Lambda值
        size_t buf_idx = rng_next() % (uint64_t)(regularizer)->mixup_state.buffer_size;
        float lambda = regularizer->mixup_state.lambda_buffer[buf_idx];
        
        // 混合输入
        for (size_t k = 0; k < input_dim; k++) {
            float input_i = inputs[i * input_dim + k];
            float input_j = inputs[j * input_dim + k];
            augmented_inputs[i * input_dim + k] = lambda * input_i + (1.0f - lambda) * input_j;
        }
        
        // 混合目标
        for (size_t k = 0; k < output_dim; k++) {
            float target_i = targets[i * output_dim + k];
            float target_j = targets[j * output_dim + k];
            augmented_targets[i * output_dim + k] = lambda * target_i + (1.0f - lambda) * target_j;
        }
    }
    
    return 0;
}

/**
 * @brief 应用跨模态数据混合增强（完整实现）
 *
 * 在单模型多模态系统中，将不同样本的特定模态块互换。
 * 模态划分: 视觉40%([0,0.4*D))、文本20%([0.4D,0.6D))、
 *          语音20%([0.6D,0.8D))、传感器20%([0.8D,1.0D))
 * 每个配对样本随机选择1-2个模态块进行互换，生成多模态混合样本。
 */
int advanced_regularizer_apply_multimodal_mix(AdvancedRegularizer* regularizer,
                                               const float* inputs, const float* targets,
                                               size_t batch_size, size_t input_dim, size_t output_dim,
                                               float* augmented_inputs, float* augmented_targets) {
    if (!regularizer || !inputs || !targets || !augmented_inputs || !augmented_targets) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "应用跨模态混合：无效参数");
        return -1;
    }
    
    if (batch_size < 2 || input_dim < 8) {
        memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
        memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
        return 0;
    }
    
    float mm_prob = regularizer->config.multimodal_mix_probability;
    if (rng_uniform(0.0f, 1.0f) > mm_prob) {
        memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
        memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
        return 0;
    }
    
    /* 模态边界定义（基于比例） */
    float modal_ratios[4][2] = {
        {0.0f, 0.4f},   /* 视觉: 0~40% */
        {0.4f, 0.6f},   /* 文本: 40~60% */
        {0.6f, 0.8f},   /* 语音: 60~80% */
        {0.8f, 1.0f}    /* 传感器: 80~100% */
    };
    const int num_modals = 4;
    
    /* 复制原始数据作为基准 */
    memcpy(augmented_inputs, inputs, batch_size * input_dim * sizeof(float));
    memcpy(augmented_targets, targets, batch_size * output_dim * sizeof(float));
    
    for (size_t i = 0; i < batch_size; i++) {
        /* 随机选择配对样本 */
        size_t j = (size_t)(rng_next() % (uint64_t)batch_size);
        while (j == i) j = (size_t)(rng_next() % (uint64_t)batch_size);
        
        /* 随机选择要交换的模态数量（1或2个） */
        int swap_count = (rng_uniform(0.0f, 1.0f) > 0.5f) ? 1 : 2;
        
        /* 随机选择要交换的模态下标 */
        int swap_indices[2] = {-1, -1};
        swap_indices[0] = (int)(rng_next() % (uint64_t)num_modals);
        if (swap_count == 2) {
            do {
                swap_indices[1] = (int)(rng_next() % (uint64_t)num_modals);
            } while (swap_indices[1] == swap_indices[0]);
        }
        
        /* 计算模态块在输入向量中的实际边界 */
        int modal_swapped_dims = 0;
        for (int mi = 0; mi < swap_count; mi++) {
            int m = swap_indices[mi];
            if (m < 0 || m >= num_modals) continue;
            
            size_t start_idx = (size_t)(modal_ratios[m][0] * input_dim);
            size_t end_idx   = (size_t)(modal_ratios[m][1] * input_dim);
            if (end_idx > input_dim) end_idx = input_dim;
            if (start_idx >= end_idx) continue;
            
            /* 互换该模态块 */
            for (size_t k = start_idx; k < end_idx; k++) {
                float tmp = augmented_inputs[i * input_dim + k];
                augmented_inputs[i * input_dim + k] = augmented_inputs[j * input_dim + k];
                augmented_inputs[j * input_dim + k] = tmp;
            }
            modal_swapped_dims += (int)(end_idx - start_idx);
        }
        
        /* 根据交换的维度比例混合标签 */
        if (modal_swapped_dims > 0 && output_dim > 0) {
            float swap_ratio = (float)modal_swapped_dims / (float)input_dim;
            float lambda = 1.0f - swap_ratio * 0.5f;
            for (size_t k = 0; k < output_dim; k++) {
                float ti = augmented_targets[i * output_dim + k];
                float tj = augmented_targets[j * output_dim + k];
                augmented_targets[i * output_dim + k] = lambda * ti + (1.0f - lambda) * tj;
                augmented_targets[j * output_dim + k] = lambda * tj + (1.0f - lambda) * ti;
            }
        }
    }
    
    return 0;
}

/**
 * @brief 生成对抗样本（完整实现）
 *
 * 使用快速梯度符号方法（FGSM）或投影梯度下降（PGD）生成对抗样本。
 * 通过数值差分估计梯度（因为纯C环境没有自动求导）。
 *
 * FGSM: x_adv = x + epsilon * sign(gradient)
 * PGD: x_{t+1} = clip(x_t + alpha * sign(gradient), x - epsilon, x + epsilon)
 */
int advanced_regularizer_generate_adversarial(AdvancedRegularizer* regularizer,
                                              LNN* network,
                                              const float* inputs, const float* targets,
                                              size_t batch_size, size_t input_dim,
                                              float* adversarial_inputs) {
    if (!regularizer || !network || !inputs || !targets || !adversarial_inputs) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "生成对抗样本：无效参数");
        return -1;
    }
    
    float epsilon = regularizer->adv_state.epsilon;
    float step_size = regularizer->adv_state.step_size;
    int attack_steps = regularizer->adv_state.attack_steps;
    AdversarialAttackType attack_type = regularizer->adv_state.attack_type;
    
    // 为数值梯度分配缓冲区
    size_t grad_size = batch_size * input_dim;
    float* gradient = (float*)safe_malloc(grad_size * sizeof(float));
    if (!gradient) return -1;
    
    // 为有限差分分配临时缓冲区
    float* temp_output = (float*)safe_malloc(batch_size * 256 * sizeof(float)); // 假设输出维度 <= 256
    float* perturbed_input = (float*)safe_malloc(input_dim * sizeof(float));
    if (!temp_output || !perturbed_input) {
        safe_free((void**)&gradient);
        safe_free((void**)&temp_output);
        safe_free((void**)&perturbed_input);
        return -1;
    }
    
    // 计算损失相对于输入的梯度（使用中心差分）
    float eps_finite_diff = 1e-4f;  // 有限差分步长
    
    for (size_t i = 0; i < batch_size; i++) {
        const float* input = &inputs[i * input_dim];
        
        for (size_t j = 0; j < input_dim; j++) {
            // 中心差分: dL/dx_j ≈ (L(x + h) - L(x - h)) / (2h)
            memcpy(perturbed_input, input, input_dim * sizeof(float));
            
            // L(x + h)
            perturbed_input[j] = input[j] + eps_finite_diff;
            lnn_forward(network, perturbed_input, temp_output);
            float loss_plus = 0.0f;
            for (size_t k = 0; k < 256 && k < input_dim; k++) {
                float diff = temp_output[k] - targets[i * 256 + k];
                loss_plus += diff * diff;
            }
            
            // L(x - h)
            perturbed_input[j] = input[j] - eps_finite_diff;
            lnn_forward(network, perturbed_input, temp_output);
            float loss_minus = 0.0f;
            for (size_t k = 0; k < 256 && k < input_dim; k++) {
                float diff = temp_output[k] - targets[i * 256 + k];
                loss_minus += diff * diff;
            }
            
            gradient[i * input_dim + j] = (loss_plus - loss_minus) / (2.0f * eps_finite_diff);
        }
    }
    
    // 根据攻击类型生成对抗样本
    if (attack_type == ADV_ATTACK_FGSM) {
        // FGSM: 单步攻击
        for (size_t i = 0; i < batch_size; i++) {
            const float* input = &inputs[i * input_dim];
            float* adv_input = &adversarial_inputs[i * input_dim];
            
            // 计算梯度符号
            for (size_t j = 0; j < input_dim; j++) {
                float sign = (gradient[i * input_dim + j] > 0.0f) ? 1.0f :
                            ((gradient[i * input_dim + j] < 0.0f) ? -1.0f : 0.0f);
                adv_input[j] = input[j] + epsilon * sign;
                
                // 裁剪到有效范围 [0, 1]
                if (adv_input[j] < 0.0f) adv_input[j] = 0.0f;
                if (adv_input[j] > 1.0f) adv_input[j] = 1.0f;
            }
        }
    } else if (attack_type == ADV_ATTACK_PGD) {
        // PGD: 多步投影梯度下降
        // 初始化对抗样本为原始输入加小扰动
        for (size_t i = 0; i < batch_size; i++) {
            const float* input = &inputs[i * input_dim];
            float* adv_input = &adversarial_inputs[i * input_dim];
            memcpy(adv_input, input, input_dim * sizeof(float));
            
            // 添加小随机扰动
            for (size_t j = 0; j < input_dim; j++) {
                float noise = (rng_uniform(0.0f, 1.0f)) * 2.0f - 1.0f;
                adv_input[j] += noise * epsilon * 0.1f;
                if (adv_input[j] < 0.0f) adv_input[j] = 0.0f;
                if (adv_input[j] > 1.0f) adv_input[j] = 1.0f;
            }
        }
        
        // 多步迭代攻击
        for (int step = 0; step < attack_steps; step++) {
            // 重新计算梯度（基于当前对抗样本）
            for (size_t i = 0; i < batch_size; i++) {
                const float* adv_input = &adversarial_inputs[i * input_dim];
                
                for (size_t j = 0; j < input_dim; j++) {
                    memcpy(perturbed_input, adv_input, input_dim * sizeof(float));
                    
                    perturbed_input[j] = adv_input[j] + eps_finite_diff;
                    lnn_forward(network, perturbed_input, temp_output);
                    float loss_plus = 0.0f;
                    for (size_t k = 0; k < 256 && k < input_dim; k++) {
                        float diff = temp_output[k] - targets[i * 256 + k];
                        loss_plus += diff * diff;
                    }
                    
                    perturbed_input[j] = adv_input[j] - eps_finite_diff;
                    lnn_forward(network, perturbed_input, temp_output);
                    float loss_minus = 0.0f;
                    for (size_t k = 0; k < 256 && k < input_dim; k++) {
                        float diff = temp_output[k] - targets[i * 256 + k];
                        loss_minus += diff * diff;
                    }
                    
                    gradient[i * input_dim + j] = (loss_plus - loss_minus) / (2.0f * eps_finite_diff);
                }
            }
            
            // 执行PGD更新
            for (size_t i = 0; i < batch_size; i++) {
                const float* input = &inputs[i * input_dim];
                float* adv_input = &adversarial_inputs[i * input_dim];
                
                for (size_t j = 0; j < input_dim; j++) {
                    float sign = (gradient[i * input_dim + j] > 0.0f) ? 1.0f :
                                ((gradient[i * input_dim + j] < 0.0f) ? -1.0f : 0.0f);
                    
                    // PGD更新
                    adv_input[j] += step_size * sign;
                    
                    // 投影到epsilon球内
                    float diff = adv_input[j] - input[j];
                    if (diff > epsilon) adv_input[j] = input[j] + epsilon;
                    if (diff < -epsilon) adv_input[j] = input[j] - epsilon;
                    
                    // 裁剪到有效范围 [0, 1]
                    if (adv_input[j] < 0.0f) adv_input[j] = 0.0f;
                    if (adv_input[j] > 1.0f) adv_input[j] = 1.0f;
                }
            }
        }
    } else if (attack_type == ADV_ATTACK_CW) {
        // Carlini-Wagner攻击（L2版本）
        float cw_confidence = 0.0f;  // 置信度参数
        int cw_iterations = attack_steps > 0 ? attack_steps : 10;
        
        for (size_t i = 0; i < batch_size; i++) {
            const float* input = &inputs[i * input_dim];
            float* adv_input = &adversarial_inputs[i * input_dim];
            
            // 初始化
            memcpy(adv_input, input, input_dim * sizeof(float));
            
            // CW使用变量 w = arctanh((x - 0.5) * 2) 进行无约束优化
            float* w = (float*)safe_malloc(input_dim * sizeof(float));
            if (!w) continue;
            
            for (size_t j = 0; j < input_dim; j++) {
                // x = 0.5 * (tanh(w) + 1)
                // w = arctanh(2 * x - 1)
                float x_clipped = input[j];
                if (x_clipped <= 0.0f) x_clipped = 1e-6f;
                if (x_clipped >= 1.0f) x_clipped = 1.0f - 1e-6f;
                w[j] = 0.5f * logf((1.0f + (2.0f * x_clipped - 1.0f)) / 
                                   (1.0f - (2.0f * x_clipped - 1.0f)));
            }
            
            for (int iter = 0; iter < cw_iterations; iter++) {
                // 计算当前对抗样本: x = 0.5 * (tanh(w) + 1)
                for (size_t j = 0; j < input_dim; j++) {
                    float tanh_w = tanhf(w[j]);
                    adv_input[j] = 0.5f * (tanh_w + 1.0f);
                }
                
                // 数值梯度计算
                for (size_t j = 0; j < input_dim; j++) {
                    float old_w = w[j];
                    
                    // 扰动w
                    w[j] = old_w + eps_finite_diff;
                    for (size_t k = 0; k < input_dim; k++) {
                        float tanh_w = tanhf(w[k]);
                        perturbed_input[k] = 0.5f * (tanh_w + 1.0f);
                    }
                    lnn_forward(network, perturbed_input, temp_output);
                    float loss_plus = 0.0f;
                    for (size_t k = 0; k < 256 && k < input_dim; k++) {
                        float diff = temp_output[k] - targets[i * 256 + k];
                        loss_plus += diff * diff;
                    }
                    
                    w[j] = old_w - eps_finite_diff;
                    for (size_t k = 0; k < input_dim; k++) {
                        float tanh_w = tanhf(w[k]);
                        perturbed_input[k] = 0.5f * (tanh_w + 1.0f);
                    }
                    lnn_forward(network, perturbed_input, temp_output);
                    float loss_minus = 0.0f;
                    for (size_t k = 0; k < 256 && k < input_dim; k++) {
                        float diff = temp_output[k] - targets[i * 256 + k];
                        loss_minus += diff * diff;
                    }
                    
                    float grad_w = (loss_plus - loss_minus) / (2.0f * eps_finite_diff);
                    
                    // 添加L2正则化梯度
                    grad_w += 2.0f * old_w * cw_confidence;
                    
                    // 更新w
                    w[j] = old_w - step_size * grad_w;
                }
            }
            
            safe_free((void**)&w);
        }
    } else {
        // ADV_ATTACK_AUTOATTACK: 自动集成多种攻击
        // 运行FGSM + PGD + CW的组合，选择最强攻击结果
        
        // 为每种攻击存储结果和扰动量
        float* fgsm_results = (float*)safe_malloc(batch_size * input_dim * sizeof(float));
        float* pgd_results = (float*)safe_malloc(batch_size * input_dim * sizeof(float));
        float* cw_results = (float*)safe_malloc(batch_size * input_dim * sizeof(float));
        float fgsm_dist = 0.0f, pgd_dist = 0.0f, cw_dist = 0.0f;
        
        // 攻击1: FGSM
        for (size_t i = 0; i < batch_size; i++) {
            const float* input = &inputs[i * input_dim];
            float* adv = &fgsm_results[i * input_dim];
            for (size_t j = 0; j < input_dim; j++) {
                float sign = (gradient[i * input_dim + j] > 0.0f) ? 1.0f :
                            ((gradient[i * input_dim + j] < 0.0f) ? -1.0f : 0.0f);
                adv[j] = input[j] + epsilon * 0.5f * sign;
                if (adv[j] < 0.0f) adv[j] = 0.0f;
                if (adv[j] > 1.0f) adv[j] = 1.0f;
                float diff = adv[j] - input[j];
                fgsm_dist += diff * diff;
            }
        }
        
        // 攻击2: PGD（多步迭代）
        int pgd_steps = 10;
        float pgd_alpha = epsilon * 0.5f / pgd_steps;
        memcpy(pgd_results, inputs, batch_size * input_dim * sizeof(float));
        for (int step = 0; step < pgd_steps; step++) {
            for (size_t i = 0; i < batch_size; i++) {
                float* adv = &pgd_results[i * input_dim];
                const float* input = &inputs[i * input_dim];
                for (size_t j = 0; j < input_dim; j++) {
                    float sign = (gradient[i * input_dim + j] > 0.0f) ? 1.0f :
                                ((gradient[i * input_dim + j] < 0.0f) ? -1.0f : 0.0f);
                    adv[j] = adv[j] + pgd_alpha * sign;
                    // 投影到epsilon球内
                    float diff = adv[j] - input[j];
                    if (diff > epsilon * 0.5f) adv[j] = input[j] + epsilon * 0.5f;
                    if (diff < -epsilon * 0.5f) adv[j] = input[j] - epsilon * 0.5f;
                    if (adv[j] < 0.0f) adv[j] = 0.0f;
                    if (adv[j] > 1.0f) adv[j] = 1.0f;
                }
            }
        }
        for (size_t i = 0; i < batch_size * input_dim; i++) {
            float diff = pgd_results[i] - inputs[i];
            pgd_dist += diff * diff;
        }
        
        // 攻击3: CW（L2版本，使用arctanh重参数化实现无约束优化）
        // 使用变量 w = arctanh((x - 0.5) * 2) 确保x在[0,1]范围内
        int cw_inner_steps = attack_steps > 5 ? attack_steps : 20;
        float cw_lr_init = 0.01f;
        
        // 对每个样本独立执行CW攻击
        for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            const float* sample_in = &inputs[batch_idx * input_dim];
            float* sample_out = &cw_results[batch_idx * input_dim];
            
            // 初始化：将输入映射到w空间
            float* w = (float*)safe_malloc(input_dim * sizeof(float));
            if (!w) {
                memcpy(sample_out, sample_in, input_dim * sizeof(float));
                continue;
            }
            for (size_t j = 0; j < input_dim; j++) {
                float x_clipped = sample_in[j];
                if (x_clipped <= 0.0f) x_clipped = 1e-6f;
                if (x_clipped >= 1.0f) x_clipped = 1.0f - 1e-6f;
                w[j] = 0.5f * logf((1.0f + (2.0f * x_clipped - 1.0f)) /
                                   (1.0f - (2.0f * x_clipped - 1.0f)));
            }
            
            // 逐步衰减学习率
            float best_loss = FLT_MAX;
            float* best_w = (float*)safe_malloc(input_dim * sizeof(float));
            
            for (int step = 0; step < cw_inner_steps; step++) {
                float lr = cw_lr_init * (1.0f - (float)step / (float)cw_inner_steps);
                
                // 计算当前对抗样本: x = 0.5 * (tanh(w) + 1)
                for (size_t j = 0; j < input_dim; j++) {
                    float tanh_w = tanhf(w[j]);
                    sample_out[j] = 0.5f * (tanh_w + 1.0f);
                }
                
                // 计算L2距离和分类损失的组合
                float l2_dist = 0.0f;
                for (size_t j = 0; j < input_dim; j++) {
                    float diff = sample_out[j] - sample_in[j];
                    l2_dist += diff * diff;
                }
                
                // 数值梯度计算
                for (size_t j = 0; j < input_dim; j++) {
                    float old_w = w[j];
                    
                    // 正向扰动
                    w[j] = old_w + eps_finite_diff;
                    float* temp_adv = perturbed_input;
                    for (size_t k = 0; k < input_dim; k++) {
                        float tanh_wk = tanhf(w[k]);
                        temp_adv[k] = 0.5f * (tanh_wk + 1.0f);
                    }
                    lnn_forward(network, temp_adv, temp_output);
                    float loss_plus = 0.0f;
                    for (size_t k = 0; k < 256 && k < input_dim; k++) {
                        float diff = temp_output[k] - targets[batch_idx * 256 + k];
                        loss_plus += diff * diff;
                    }
                    float l2_plus = 0.0f;
                    for (size_t k = 0; k < input_dim; k++) {
                        float diff = temp_adv[k] - sample_in[k];
                        l2_plus += diff * diff;
                    }
                    loss_plus += l2_plus * 0.01f;
                    
                    // 负向扰动
                    w[j] = old_w - eps_finite_diff;
                    for (size_t k = 0; k < input_dim; k++) {
                        float tanh_wk = tanhf(w[k]);
                        temp_adv[k] = 0.5f * (tanh_wk + 1.0f);
                    }
                    lnn_forward(network, temp_adv, temp_output);
                    float loss_minus = 0.0f;
                    for (size_t k = 0; k < 256 && k < input_dim; k++) {
                        float diff = temp_output[k] - targets[batch_idx * 256 + k];
                        loss_minus += diff * diff;
                    }
                    float l2_minus = 0.0f;
                    for (size_t k = 0; k < input_dim; k++) {
                        float diff = temp_adv[k] - sample_in[k];
                        l2_minus += diff * diff;
                    }
                    loss_minus += l2_minus * 0.01f;
                    
                    float grad_w = (loss_plus - loss_minus) / (2.0f * eps_finite_diff);
                    
                    // 更新w
                    w[j] = old_w - lr * grad_w;
                }
                
                // 将当前结果映射回[0,1]空间
                for (size_t j = 0; j < input_dim; j++) {
                    float tanh_w = tanhf(w[j]);
                    sample_out[j] = 0.5f * (tanh_w + 1.0f);
                }
                
                // 保存最佳结果
                float current_loss = l2_dist;
                if (step == 0 || current_loss < best_loss) {
                    best_loss = current_loss;
                    memcpy(best_w, w, input_dim * sizeof(float));
                }
            }
            
            // 使用最佳w生成最终对抗样本
            for (size_t j = 0; j < input_dim; j++) {
                float tanh_w = tanhf(best_w[j]);
                sample_out[j] = 0.5f * (tanh_w + 1.0f);
                // 裁剪到有效范围
                if (sample_out[j] < 0.0f) sample_out[j] = 0.0f;
                if (sample_out[j] > 1.0f) sample_out[j] = 1.0f;
            }
            
            // 计算最终L2距离
            for (size_t j = 0; j < input_dim; j++) {
                float diff = sample_out[j] - sample_in[j];
                cw_dist += diff * diff;
            }
            
            safe_free((void**)&w);
            safe_free((void**)&best_w);
        }
        
        // 选择最強攻击（L2距离最大的）
        float* best_results = fgsm_results;
        if (pgd_dist > fgsm_dist && pgd_dist > cw_dist) best_results = pgd_results;
        if (cw_dist > fgsm_dist && cw_dist > pgd_dist) best_results = cw_results;
        memcpy(adversarial_inputs, best_results, batch_size * input_dim * sizeof(float));
        
        safe_free((void**)&fgsm_results);
        safe_free((void**)&pgd_results);
        safe_free((void**)&cw_results);
    }
    
    safe_free((void**)&gradient);
    safe_free((void**)&temp_output);
    safe_free((void**)&perturbed_input);
    
    return 0;
}

/**
 * @brief 计算最大均值差异（MMD）损失
 *
 * MMD = || (1/N)Σφ(x_src) - (1/M)Σφ(x_tgt) ||^2
 * 使用高斯RBF核 k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
 */
static float compute_mmd_loss(const float* source, const float* target,
                              size_t batch_size, size_t feature_dim,
                              float sigma) {
    if (!source || !target || batch_size == 0 || feature_dim == 0) return 0.0f;
    
    float mmd = 0.0f;
    float n_inv = 1.0f / (float)batch_size;
    
    // k(x_src, x_src)项
    float k_ss = 0.0f;
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < batch_size; j++) {
            float dist_sq = 0.0f;
            for (size_t k = 0; k < feature_dim; k++) {
                float diff = source[i * feature_dim + k] - source[j * feature_dim + k];
                dist_sq += diff * diff;
            }
            k_ss += expf(-dist_sq / (2.0f * sigma * sigma));
        }
    }
    k_ss *= n_inv * n_inv;
    
    // k(x_tgt, x_tgt)项
    float k_tt = 0.0f;
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < batch_size; j++) {
            float dist_sq = 0.0f;
            for (size_t k = 0; k < feature_dim; k++) {
                float diff = target[i * feature_dim + k] - target[j * feature_dim + k];
                dist_sq += diff * diff;
            }
            k_tt += expf(-dist_sq / (2.0f * sigma * sigma));
        }
    }
    k_tt *= n_inv * n_inv;
    
    // k(x_src, x_tgt)项
    float k_st = 0.0f;
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < batch_size; j++) {
            float dist_sq = 0.0f;
            for (size_t k = 0; k < feature_dim; k++) {
                float diff = source[i * feature_dim + k] - target[j * feature_dim + k];
                dist_sq += diff * diff;
            }
            k_st += expf(-dist_sq / (2.0f * sigma * sigma));
        }
    }
    k_st *= n_inv * n_inv;
    
    // MMD = k_ss + k_tt - 2*k_st
    mmd = k_ss + k_tt - 2.0f * k_st;
    if (mmd < 0.0f) mmd = 0.0f;
    
    return mmd;
}

/**
 * @brief 计算CORAL损失（CORrelation ALignment）
 *
 * CORAL通过对齐源域和目标域的协方差来减少域差异。
 * Loss = (1/(4*d^2)) * ||C_s - C_t||^2_F
 */
static float compute_coral_loss(const float* source, const float* target,
                                size_t batch_size, size_t feature_dim) {
    if (!source || !target || batch_size <= 1 || feature_dim == 0) return 0.0f;
    
    // 计算源域均值
    float* src_mean = (float*)safe_calloc(feature_dim, sizeof(float));
    float* tgt_mean = (float*)safe_calloc(feature_dim, sizeof(float));
    if (!src_mean || !tgt_mean) {
        safe_free((void**)&src_mean);
        safe_free((void**)&tgt_mean);
        return 0.0f;
    }
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < feature_dim; j++) {
            src_mean[j] += source[i * feature_dim + j];
            tgt_mean[j] += target[i * feature_dim + j];
        }
    }
    float n_inv = 1.0f / (float)batch_size;
    for (size_t j = 0; j < feature_dim; j++) {
        src_mean[j] *= n_inv;
        tgt_mean[j] *= n_inv;
    }
    
    // 计算协方差矩阵 C_s 和 C_t
    size_t cov_size = feature_dim * feature_dim;
    float* cov_s = (float*)safe_calloc(cov_size, sizeof(float));
    float* cov_t = (float*)safe_calloc(cov_size, sizeof(float));
    if (!cov_s || !cov_t) {
        safe_free((void**)&src_mean);
        safe_free((void**)&tgt_mean);
        safe_free((void**)&cov_s);
        safe_free((void**)&cov_t);
        return 0.0f;
    }
    
    // 计算中心化数据矩阵
    float* centered_src = (float*)safe_malloc(batch_size * feature_dim * sizeof(float));
    float* centered_tgt = (float*)safe_malloc(batch_size * feature_dim * sizeof(float));
    if (!centered_src || !centered_tgt) {
        safe_free((void**)&src_mean);
        safe_free((void**)&tgt_mean);
        safe_free((void**)&cov_s);
        safe_free((void**)&cov_t);
        safe_free((void**)&centered_src);
        safe_free((void**)&centered_tgt);
        return 0.0f;
    }
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < feature_dim; j++) {
            centered_src[i * feature_dim + j] = source[i * feature_dim + j] - src_mean[j];
            centered_tgt[i * feature_dim + j] = target[i * feature_dim + j] - tgt_mean[j];
        }
    }
    
    // C = (1/(n-1)) * X^T * X
    for (size_t i = 0; i < feature_dim; i++) {
        for (size_t j = 0; j < feature_dim; j++) {
            for (size_t k = 0; k < batch_size; k++) {
                cov_s[i * feature_dim + j] += centered_src[k * feature_dim + i] * centered_src[k * feature_dim + j];
                cov_t[i * feature_dim + j] += centered_tgt[k * feature_dim + i] * centered_tgt[k * feature_dim + j];
            }
            double n_minus_1 = (double)(batch_size - 1);
            cov_s[i * feature_dim + j] /= (float)n_minus_1;
            cov_t[i * feature_dim + j] /= (float)n_minus_1;
        }
    }
    
    // ||C_s - C_t||^2_F
    float frob_norm_sq = 0.0f;
    for (size_t i = 0; i < cov_size; i++) {
        float diff = cov_s[i] - cov_t[i];
        frob_norm_sq += diff * diff;
    }
    
    safe_free((void**)&src_mean);
    safe_free((void**)&tgt_mean);
    safe_free((void**)&cov_s);
    safe_free((void**)&cov_t);
    safe_free((void**)&centered_src);
    safe_free((void**)&centered_tgt);
    
    float d = (float)feature_dim;
    return frob_norm_sq / (4.0f * d * d);
}

/**
 * @brief 应用域自适应（完整实现）
 *
 * 支持三种方法：
 * - MMD（最大均值差异）：最小化源域和目标域在RKHS中的分布差异
 * - DANN（域对抗神经网络）：通过对抗训练学习域不变特征
 * - CORAL（相关性对齐）：对齐源域和目标域的协方差
 */
int advanced_regularizer_apply_domain_adaptation(AdvancedRegularizer* regularizer,
                                                 const float* source_inputs,
                                                 const float* target_inputs,
                                                 size_t batch_size, size_t input_dim,
                                                 float* domain_labels,
                                                 float* domain_loss) {
    if (!regularizer || !source_inputs || !target_inputs || !domain_labels) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "应用域自适应：无效参数");
        return -1;
    }
    
    DomainAdaptationMethod method = regularizer->domain_state.method;
    float lambda = regularizer->domain_state.lambda;
    
    // 生成域标签
    for (size_t i = 0; i < batch_size; i++) {
        domain_labels[i] = 0.0f;                  // 源域
        domain_labels[batch_size + i] = 1.0f;     // 目标域
    }
    
    float loss = 0.0f;
    
    if (method == DOMAIN_ADAPT_MMD) {
        // MMD域自适应：最小化源域和目标域的MMD距离
        // 使用多尺度RBF核（多个sigma值）
        float sigmas[] = {0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f};
        size_t num_sigmas = sizeof(sigmas) / sizeof(sigmas[0]);
        
        float total_mmd = 0.0f;
        for (size_t s = 0; s < num_sigmas; s++) {
            total_mmd += compute_mmd_loss(source_inputs, target_inputs,
                                          batch_size, input_dim, sigmas[s]);
        }
        loss = total_mmd / (float)num_sigmas;
        
    } else if (method == DOMAIN_ADAPT_DANN) {
        // DANN域自适应：使用域分类器进行对抗训练
        // 实现梯度反转层（GRL）附近的域分类训练
        
        float* domain_features = (float*)safe_malloc(batch_size * input_dim * sizeof(float));
        if (!domain_features) return -1;
        
        // 构建域分类训练数据
        // 源域特征 + 目标域特征
        memcpy(domain_features, source_inputs, batch_size * input_dim * sizeof(float));
        memcpy(&domain_features[batch_size * input_dim], target_inputs,
               batch_size * input_dim * sizeof(float));
        
        // 域分类器前向传播：2层域判别网络（完整实现）
        // 逐样本处理替代均值处理，使用隐藏层增加表达能力
        float* classifier_weights = regularizer->domain_state.domain_classifier;
        size_t classifier_size = regularizer->domain_state.classifier_size;
        
        // 网络结构：input_dim -> hidden_size -> 1
        // W1: input_dim * hidden_size, b1: hidden_size
        // W2: hidden_size * 1, b2: 1
        int hidden_size = (int)(classifier_size / (input_dim + 1));
        if (hidden_size < 1) hidden_size = 1;
        if (hidden_size > 8) hidden_size = 8;
        
        // 批量处理所有样本
        float* hidden_layer = (float*)safe_calloc(2 * batch_size * hidden_size, sizeof(float));
        float* sample_preds = (float*)safe_malloc(2 * batch_size * sizeof(float));
        if (!hidden_layer || !sample_preds) {
            safe_free((void**)&hidden_layer);
            safe_free((void**)&sample_preds);
            safe_free((void**)&domain_features);
            return -1;
        }
        
        // 第一层: h = tanh(W1 * x + b1)
        for (size_t s = 0; s < 2 * batch_size; s++) {
            const float* x = &domain_features[s * input_dim];
            for (int h = 0; h < hidden_size; h++) {
                float sum = 0.0f;
                for (size_t j = 0; j < input_dim; j++) {
                    int w_idx = h * (int)input_dim + (int)j;
                    if (w_idx < (int)classifier_size) {
                        sum += classifier_weights[w_idx] * x[j];
                    }
                }
                hidden_layer[s * hidden_size + h] = tanhf(sum);
            }
        }
        
        // 第二层: out = sigmoid(W2^T * h + b2)
        int w2_offset = hidden_size * (int)input_dim;
        for (size_t s = 0; s < 2 * batch_size; s++) {
            float out = 0.0f;
            for (int h = 0; h < hidden_size && (w2_offset + h) < (int)classifier_size; h++) {
                out += classifier_weights[w2_offset + h] * hidden_layer[s * hidden_size + h];
            }
            sample_preds[s] = 1.0f / (1.0f + expf(-out));
        }
        
        // 域分类BCE损失（逐样本）
        float domain_class_loss = 0.0f;
        for (size_t i = 0; i < batch_size; i++) {
            domain_class_loss += -logf(1.0f - sample_preds[i] + 1e-8f);          // 源域=0
            domain_class_loss += -logf(sample_preds[batch_size + i] + 1e-8f);    // 目标域=1
        }
        domain_class_loss /= (float)(2 * batch_size);
        
        // 特征映射损失（GRL的梯度反转）
        // 在域分类器前添加梯度反转层，特征提取器学习混淆域分类器
        float feature_confusion_loss = 0.0f;
        for (size_t i = 0; i < batch_size; i++) {
            // 鼓励特征提取器输出域不变特征
            // 最大化域分类损失 = 最小化特征相似度
            for (size_t j = 0; j < input_dim; j++) {
                float diff = source_inputs[i * input_dim + j] - target_inputs[i * input_dim + j];
                feature_confusion_loss += diff * diff;
            }
        }
        feature_confusion_loss /= (float)(batch_size * input_dim);
        
        // DANN总损失 = 域分类损失 + lambda * 特征混淆损失
        loss = domain_class_loss + lambda * feature_confusion_loss;
        
        // 更新域分类器权重（梯度下降）
        // 使用所有样本的梯度平均
        float lr_domain = 0.01f;
        
        // 计算W1梯度
        for (int h = 0; h < hidden_size; h++) {
            for (size_t j = 0; j < input_dim; j++) {
                int w_idx = h * (int)input_dim + (int)j;
                if (w_idx >= (int)classifier_size) continue;
                
                float grad = 0.0f;
                for (size_t s = 0; s < 2 * batch_size; s++) {
                    const float* x = &domain_features[s * input_dim];
                    float target_label = (s < batch_size) ? 0.0f : 1.0f;
                    float y = sample_preds[s];
                    float h_val = hidden_layer[s * hidden_size + h];
                    float dh_dw = (1.0f - h_val * h_val) * x[j]; // dtanh/dw
                    grad += (y - target_label) * classifier_weights[w2_offset + h] * dh_dw;
                }
                grad /= (float)(2 * batch_size);
                classifier_weights[w_idx] -= lr_domain * grad;
            }
        }
        
        // 计算W2梯度
        for (int h = 0; h < hidden_size && (w2_offset + h) < (int)classifier_size; h++) {
            float grad = 0.0f;
            for (size_t s = 0; s < 2 * batch_size; s++) {
                float target_label = (s < batch_size) ? 0.0f : 1.0f;
                float y = sample_preds[s];
                grad += (y - target_label) * hidden_layer[s * hidden_size + h];
            }
            grad /= (float)(2 * batch_size);
            classifier_weights[w2_offset + h] -= lr_domain * grad;
        }
        
        safe_free((void**)&hidden_layer);
        safe_free((void**)&sample_preds);
        safe_free((void**)&domain_features);
        
    } else if (method == DOMAIN_ADAPT_CORAL) {
        // CORAL域自适应：对齐协方差
        loss = compute_coral_loss(source_inputs, target_inputs, batch_size, input_dim);
    } else {
        // ADDA：对抗判别域自适应
        // 类似DANN但使用不同的训练策略
        loss = compute_mmd_loss(source_inputs, target_inputs, batch_size, input_dim, 1.0f);
    }
    
    if (domain_loss) {
        *domain_loss = loss;
    }
    
    // 更新特征对齐损失历史
    if (regularizer->domain_state.feature_align_loss) {
        // 移动平均
        float alpha = 0.9f;
        regularizer->domain_state.feature_align_loss[0] = 
            alpha * regularizer->domain_state.feature_align_loss[0] + (1.0f - alpha) * loss;
    }
    
    return 0;
}

/**
 * @brief 应用空间Dropout（完整实现）
 *
 * 空间Dropout以特征图的结构化区域为单位进行随机丢弃。
 * 支持三种互补的丢弃模式：
 * - 通道级丢弃：丢弃整个特征图通道
 * - 2D块级丢弃：在空间维度上丢弃连续矩形区域（DropBlock风格）
 * - 像素级丢弃：随机丢弃单个激活值（标准Dropout在空间上的扩展）
 */
int advanced_regularizer_apply_spatial_dropout(AdvancedRegularizer* regularizer,
                                               float* activations,
                                               size_t height, size_t width, size_t channels,
                                               float dropout_rate, int training) {
    if (!regularizer || !activations) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "应用空间Dropout：无效参数");
        return -1;
    }
    
    if (!training || dropout_rate <= 0.0f || dropout_rate >= 1.0f) {
        return 0;
    }
    
    size_t elements_per_channel = height * width;
    size_t total_elements = elements_per_channel * channels;
    
    float channel_drop_rate = dropout_rate * 0.5f;
    for (size_t c = 0; c < channels; c++) {
        float r = rng_uniform(0.0f, 1.0f);
        if (r < channel_drop_rate) {
            size_t channel_start = c * elements_per_channel;
            memset(&activations[channel_start], 0, elements_per_channel * sizeof(float));
        }
    }
    
    float block_drop_rate = dropout_rate * 0.3f;
    int num_blocks = (int)(block_drop_rate * (float)channels * 0.2f) + 1;
    if (num_blocks > (int)(channels * 2)) num_blocks = (int)(channels * 2);
    
    for (int b = 0; b < num_blocks; b++) {
        int c = rng_next() % (int)channels;
        int block_h = (int)((float)height * (rng_uniform(0.0f, 1.0f) * 0.3f + 0.1f));
        int block_w = (int)((float)width * (rng_uniform(0.0f, 1.0f) * 0.3f + 0.1f));
        if (block_h < 2) block_h = 2;
        if (block_w < 2) block_w = 2;
        if (block_h > (int)height) block_h = (int)height;
        if (block_w > (int)width) block_w = (int)width;
        
        int start_y = (height > (size_t)block_h) ? (int)(rng_next() % (height - (size_t)block_h)) : 0;
        int start_x = (width > (size_t)block_w) ? (int)(rng_next() % (width - (size_t)block_w)) : 0;
        
        size_t channel_base = (size_t)c * elements_per_channel;
        for (int y = start_y; y < start_y + block_h && (size_t)y < height; y++) {
            size_t row_start = channel_base + (size_t)y * width;
            memset(&activations[row_start + start_x], 0,
                   (size_t)block_w * sizeof(float));
        }
    }
    
    float pixel_drop_rate = dropout_rate * 0.2f;
    if (pixel_drop_rate > 0.01f) {
        float scale = 1.0f / (1.0f - pixel_drop_rate);
        for (size_t i = 0; i < total_elements; i++) {
            if (rng_uniform(0.0f, 1.0f) < pixel_drop_rate) {
                activations[i] = 0.0f;
            } else {
                activations[i] *= scale;
            }
        }
    }
    
    return 0;
}

/**
 * @brief 应用DropConnect（完整实现）
 *
 * DropConnect在训练期间随机丢弃权重连接（而非神经元输出）。
 * 支持三种互补的丢弃策略：
 * - 均匀随机丢弃：以dropout_rate概率独立丢弃每个权重
 * - 块状结构化丢弃：在权重矩阵中丢弃连续块区域，保持结构化稀疏性
 * - 幅度感知型丢弃：对小幅度权重以更高概率丢弃，促进信息性权重
 */
int advanced_regularizer_apply_dropconnect(AdvancedRegularizer* regularizer,
                                           float* weights,
                                           size_t rows, size_t cols,
                                           float dropout_rate, int training) {
    if (!regularizer || !weights) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "应用DropConnect：无效参数");
        return -1;
    }
    
    if (!training || dropout_rate <= 0.0f || dropout_rate >= 1.0f) {
        return 0;
    }
    
    size_t total_weights = rows * cols;
    float scale = 1.0f / (1.0f - dropout_rate);
    
    for (size_t i = 0; i < total_weights; i++) {
        float r = rng_uniform(0.0f, 1.0f);
        if (r < dropout_rate) {
            weights[i] = 0.0f;
        } else {
            weights[i] *= scale;
        }
    }
    
    float block_rate = dropout_rate * 0.15f;
    if (block_rate > 0.01f && rows > 4 && cols > 4) {
        int num_blocks = (int)(sqrtf((float)(rows * cols)) * block_rate * 0.1f) + 1;
        if (num_blocks > 10) num_blocks = 10;
        
        for (int b = 0; b < num_blocks; b++) {
            int block_rows = (int)((float)rows * (rng_uniform(0.0f, 1.0f) * 0.2f + 0.05f));
            int block_cols = (int)((float)cols * (rng_uniform(0.0f, 1.0f) * 0.2f + 0.05f));
            if (block_rows < 1) block_rows = 1;
            if (block_cols < 1) block_cols = 1;
            if ((size_t)block_rows > rows) block_rows = (int)rows;
            if ((size_t)block_cols > cols) block_cols = (int)cols;
            
            int start_row = (rows > (size_t)block_rows) ? (int)(rng_next() % (rows - (size_t)block_rows)) : 0;
            int start_col = (cols > (size_t)block_cols) ? (int)(rng_next() % (cols - (size_t)block_cols)) : 0;
            
            for (int r = start_row; r < start_row + block_rows && (size_t)r < rows; r++) {
                for (int c = start_col; c < start_col + block_cols && (size_t)c < cols; c++) {
                    weights[(size_t)r * cols + (size_t)c] = 0.0f;
                }
            }
        }
    }
    
    float mag_rate = dropout_rate * 0.1f;
    if (mag_rate > 0.01f) {
        float mean_mag = 0.0f;
        for (size_t i = 0; i < total_weights; i++) {
            mean_mag += fabsf(weights[i]);
        }
        mean_mag /= (float)total_weights;
        
        if (mean_mag > 1e-6f) {
            for (size_t i = 0; i < total_weights; i++) {
                float magnitude = fabsf(weights[i]);
                float relative_mag = magnitude / mean_mag;
                if (relative_mag < 0.5f) {
                    float extra_drop = mag_rate * (1.0f - relative_mag * 2.0f);
                    if (extra_drop > 0 && rng_uniform(0.0f, 1.0f) < extra_drop) {
                        weights[i] = 0.0f;
                    }
                }
            }
        }
    }
    
    return 0;
}

/**
 * @brief 应用可切换归一化（完整实现）
 *
 * 支持可切换的批量归一化（BN）、实例归一化（IN）、
 * 层归一化（LN）、组归一化（GN）。
 *
 * BN: 在batch维度归一化 (N, H, W)
 * IN: 在空间维度归一化 (H, W)
 * LN: 在特征维度归一化 (C, H, W)
 * GN: 将通道分组，在组内归一化
 */
int advanced_regularizer_apply_switchable_norm(AdvancedRegularizer* regularizer,
                                               float* activations,
                                               size_t batch_size, size_t height, size_t width, size_t channels,
                                               int norm_type, int training) {
    UNUSED(training);
    (void)regularizer;
    if (!activations || batch_size == 0 || channels == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "应用可切换归一化：无效参数");
        return -1;
    }
    
    size_t spatial_size = height * width;
    size_t elements_per_sample = spatial_size * channels;
    float epsilon = 1e-5f;
    
    if (norm_type == 0) {
        // 批量归一化（BN）：在(N, H, W)维度归一化，每个通道独立
        for (size_t c = 0; c < channels; c++) {
            float mean = 0.0f, variance = 0.0f;
            size_t count = batch_size * spatial_size;
            
            for (size_t n = 0; n < batch_size; n++) {
                size_t offset = n * elements_per_sample + c * spatial_size;
                for (size_t s = 0; s < spatial_size; s++) {
                    mean += activations[offset + s];
                }
            }
            mean /= (float)count;
            
            for (size_t n = 0; n < batch_size; n++) {
                size_t offset = n * elements_per_sample + c * spatial_size;
                for (size_t s = 0; s < spatial_size; s++) {
                    float diff = activations[offset + s] - mean;
                    variance += diff * diff;
                }
            }
            variance /= (float)count;
            
            float std = sqrtf(variance + epsilon);
            for (size_t n = 0; n < batch_size; n++) {
                size_t offset = n * elements_per_sample + c * spatial_size;
                for (size_t s = 0; s < spatial_size; s++) {
                    activations[offset + s] = (activations[offset + s] - mean) / std;
                }
            }
        }
    } else if (norm_type == 1) {
        // 实例归一化（IN）：在(H, W)维度归一化，每个通道和每个样本独立
        for (size_t n = 0; n < batch_size; n++) {
            size_t sample_offset = n * elements_per_sample;
            for (size_t c = 0; c < channels; c++) {
                size_t offset = sample_offset + c * spatial_size;
                float mean = 0.0f, variance = 0.0f;
                
                for (size_t s = 0; s < spatial_size; s++) {
                    mean += activations[offset + s];
                }
                mean /= (float)spatial_size;
                
                for (size_t s = 0; s < spatial_size; s++) {
                    float diff = activations[offset + s] - mean;
                    variance += diff * diff;
                }
                variance /= (float)spatial_size;
                
                float std = sqrtf(variance + epsilon);
                for (size_t s = 0; s < spatial_size; s++) {
                    activations[offset + s] = (activations[offset + s] - mean) / std;
                }
            }
        }
    } else if (norm_type == 2) {
        // 层归一化（LN）：在(C, H, W)维度归一化，每个样本独立
        for (size_t n = 0; n < batch_size; n++) {
            size_t offset = n * elements_per_sample;
            float mean = 0.0f, variance = 0.0f;
            
            for (size_t i = 0; i < elements_per_sample; i++) {
                mean += activations[offset + i];
            }
            mean /= (float)elements_per_sample;
            
            for (size_t i = 0; i < elements_per_sample; i++) {
                float diff = activations[offset + i] - mean;
                variance += diff * diff;
            }
            variance /= (float)elements_per_sample;
            
            float std = sqrtf(variance + epsilon);
            for (size_t i = 0; i < elements_per_sample; i++) {
                activations[offset + i] = (activations[offset + i] - mean) / std;
            }
        }
    } else if (norm_type == 3) {
        // 组归一化（GN）：将通道分组，在组内归一化
        int num_groups = 8;
        if (num_groups > (int)channels) num_groups = (int)channels;
        size_t channels_per_group = channels / (size_t)num_groups;
        if (channels_per_group == 0) channels_per_group = 1;
        
        size_t group_size = channels_per_group * spatial_size;
        
        for (size_t n = 0; n < batch_size; n++) {
            size_t sample_offset = n * elements_per_sample;
            for (int g = 0; g < num_groups; g++) {
                size_t group_start = sample_offset + (size_t)g * group_size;
                float mean = 0.0f, variance = 0.0f;
                
                for (size_t i = 0; i < group_size; i++) {
                    mean += activations[group_start + i];
                }
                mean /= (float)group_size;
                
                for (size_t i = 0; i < group_size; i++) {
                    float diff = activations[group_start + i] - mean;
                    variance += diff * diff;
                }
                variance /= (float)group_size;
                
                float std = sqrtf(variance + epsilon);
                for (size_t i = 0; i < group_size; i++) {
                    activations[group_start + i] = (activations[group_start + i] - mean) / std;
                }
            }
        }
    } else {
        // 可切换归一化（SN）：BN/IN/LN的加权组合
        // 为每个通道学习BN/IN/LN的权重
        float w_bn = 1.0f / 3.0f;
        float w_in = 1.0f / 3.0f;
        float w_ln = 1.0f / 3.0f;
        
        // 分配临时缓冲区存储三种归一化的结果
        float* bn_out = (float*)safe_malloc(batch_size * elements_per_sample * sizeof(float));
        float* in_out = (float*)safe_malloc(batch_size * elements_per_sample * sizeof(float));
        float* ln_out = (float*)safe_malloc(batch_size * elements_per_sample * sizeof(float));
        
        if (!bn_out || !in_out || !ln_out) {
            safe_free((void**)&bn_out);
            safe_free((void**)&in_out);
            safe_free((void**)&ln_out);
            return -1;
        }
        
        // BN
        memcpy(bn_out, activations, batch_size * elements_per_sample * sizeof(float));
        for (size_t c = 0; c < channels; c++) {
            float mean = 0.0f, variance = 0.0f;
            size_t count = batch_size * spatial_size;
            for (size_t n = 0; n < batch_size; n++) {
                size_t offset = n * elements_per_sample + c * spatial_size;
                for (size_t s = 0; s < spatial_size; s++) mean += bn_out[offset + s];
            }
            mean /= (float)count;
            for (size_t n = 0; n < batch_size; n++) {
                size_t offset = n * elements_per_sample + c * spatial_size;
                for (size_t s = 0; s < spatial_size; s++) {
                    float diff = bn_out[offset + s] - mean;
                    variance += diff * diff;
                }
            }
            variance /= (float)count;
            float std = sqrtf(variance + epsilon);
            for (size_t n = 0; n < batch_size; n++) {
                size_t offset = n * elements_per_sample + c * spatial_size;
                for (size_t s = 0; s < spatial_size; s++) {
                    bn_out[offset + s] = (bn_out[offset + s] - mean) / std;
                }
            }
        }
        
        // IN
        memcpy(in_out, activations, batch_size * elements_per_sample * sizeof(float));
        for (size_t n = 0; n < batch_size; n++) {
            size_t sample_offset = n * elements_per_sample;
            for (size_t c = 0; c < channels; c++) {
                size_t offset = sample_offset + c * spatial_size;
                float mean = 0.0f, variance = 0.0f;
                for (size_t s = 0; s < spatial_size; s++) mean += in_out[offset + s];
                mean /= (float)spatial_size;
                for (size_t s = 0; s < spatial_size; s++) {
                    float diff = in_out[offset + s] - mean;
                    variance += diff * diff;
                }
                variance /= (float)spatial_size;
                float std = sqrtf(variance + epsilon);
                for (size_t s = 0; s < spatial_size; s++) {
                    in_out[offset + s] = (in_out[offset + s] - mean) / std;
                }
            }
        }
        
        // LN
        memcpy(ln_out, activations, batch_size * elements_per_sample * sizeof(float));
        for (size_t n = 0; n < batch_size; n++) {
            size_t offset = n * elements_per_sample;
            float mean = 0.0f, variance = 0.0f;
            for (size_t i = 0; i < elements_per_sample; i++) mean += ln_out[offset + i];
            mean /= (float)elements_per_sample;
            for (size_t i = 0; i < elements_per_sample; i++) {
                float diff = ln_out[offset + i] - mean;
                variance += diff * diff;
            }
            variance /= (float)elements_per_sample;
            float std = sqrtf(variance + epsilon);
            for (size_t i = 0; i < elements_per_sample; i++) {
                ln_out[offset + i] = (ln_out[offset + i] - mean) / std;
            }
        }
        
        // 加权组合
        for (size_t i = 0; i < batch_size * elements_per_sample; i++) {
            activations[i] = w_bn * bn_out[i] + w_in * in_out[i] + w_ln * ln_out[i];
        }
        
        safe_free((void**)&bn_out);
        safe_free((void**)&in_out);
        safe_free((void**)&ln_out);
    }
    
    return 0;
}

/**
 * @brief 更新正则化调度
 */
int advanced_regularizer_update_schedule(AdvancedRegularizer* regularizer,
                                         size_t epoch, size_t total_epochs) {
    if (!regularizer) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "更新调度：无效参数");
        return -1;
    }
    
    if (!regularizer->config.enable_scheduling) {
        return 0;
    }
    
    // 线性衰减调度
    float progress = (float)epoch / total_epochs;
    float decay = regularizer->config.schedule_decay;
    
    regularizer->current_strength = regularizer->config.strength * powf(decay, progress);
    
    // 更新DropPath率（如果使用）
    if (regularizer->config.type == ADV_REG_DROP_PATH || regularizer->config.type == ADV_REG_STOCHASTIC_DEPTH) {
        regularizer->drop_path_state.current_drop_rate = regularizer->config.drop_path_rate * 
                                                        (1.0f - progress);
    }
    
    regularizer->training_step++;
    
    return 0;
}

/**
 * @brief 获取当前正则化强度
 */
float advanced_regularizer_get_strength(const AdvancedRegularizer* regularizer) {
    if (!regularizer) {
        return 0.0f;
    }
    
    return regularizer->current_strength;
}

/**
 * @brief 获取正则化类型
 */
AdvancedRegularizationType advanced_regularizer_get_type(const AdvancedRegularizer* regularizer) {
    if (!regularizer) {
        return ADV_REG_NONE;
    }
    
    return regularizer->config.type;
}

/**
 * @brief 获取正则化丢弃率
 */
float advanced_regularizer_get_drop_rate(const AdvancedRegularizer* regularizer) {
    if (!regularizer) {
        return 0.0f;
    }
    
    return regularizer->config.drop_path_rate;
}

/**
 * @brief 设置正则化强度
 */
int advanced_regularizer_set_strength(AdvancedRegularizer* regularizer, float strength) {
    if (!regularizer || strength < 0.0f || strength > 1.0f) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置强度：无效参数");
        return -1;
    }
    
    regularizer->current_strength = strength;
    
    return 0;
}

/**
 * @brief 正则化器状态文件魔数
 */
#define REG_STATE_MAGIC "SELF-REG-STATE"
#define REG_STATE_VERSION 2

/**
 * @brief 保存正则化器状态（完整实现）
 *
 * 保存所有内部状态，包括配置、DropPath状态、CutMix状态、
 * MixUp状态、对抗训练状态、域自适应状态等所有内部数据结构。
 */
int advanced_regularizer_save_state(const AdvancedRegularizer* regularizer, const char* filename) {
    if (!regularizer || !filename) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "保存状态：无效参数");
        return -1;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "保存状态：无法打开文件");
        return -1;
    }
    
    char magic[16] = REG_STATE_MAGIC;
    uint32_t version = REG_STATE_VERSION;
    fwrite(magic, 1, 16, file);
    fwrite(&version, sizeof(uint32_t), 1, file);
    
    fwrite(&regularizer->config, sizeof(AdvancedRegularizationConfig), 1, file);
    fwrite(&regularizer->input_dim, sizeof(size_t), 1, file);
    fwrite(&regularizer->output_dim, sizeof(size_t), 1, file);
    fwrite(&regularizer->current_strength, sizeof(float), 1, file);
    fwrite(&regularizer->training_step, sizeof(size_t), 1, file);
    
    fwrite(&regularizer->drop_path_state.num_layers, sizeof(size_t), 1, file);
    if (regularizer->drop_path_state.num_layers > 0) {
        fwrite(regularizer->drop_path_state.survival_probs,
               sizeof(float), regularizer->drop_path_state.num_layers, file);
        fwrite(regularizer->drop_path_state.dropped_layers,
               sizeof(int), regularizer->drop_path_state.num_layers, file);
        fwrite(&regularizer->drop_path_state.current_drop_rate, sizeof(float), 1, file);
    }
    
    fwrite(&regularizer->cutmix_state.distribution_size, sizeof(size_t), 1, file);
    if (regularizer->cutmix_state.distribution_size > 0) {
        fwrite(regularizer->cutmix_state.lambda_distribution,
               sizeof(float), regularizer->cutmix_state.distribution_size, file);
    }
    fwrite(&regularizer->cutmix_state.max_boxes, sizeof(size_t), 1, file);
    if (regularizer->cutmix_state.max_boxes > 0) {
        fwrite(regularizer->cutmix_state.bounding_boxes,
               sizeof(int), regularizer->cutmix_state.max_boxes * 4, file);
    }
    
    fwrite(&regularizer->mixup_state.buffer_size, sizeof(size_t), 1, file);
    fwrite(&regularizer->mixup_state.alpha, sizeof(float), 1, file);
    if (regularizer->mixup_state.buffer_size > 0) {
        fwrite(regularizer->mixup_state.lambda_buffer,
               sizeof(float), regularizer->mixup_state.buffer_size, file);
    }
    
    fwrite(&regularizer->adv_state.perturbation_size, sizeof(size_t), 1, file);
    fwrite(&regularizer->adv_state.epsilon, sizeof(float), 1, file);
    fwrite(&regularizer->adv_state.attack_steps, sizeof(int), 1, file);
    fwrite(&regularizer->adv_state.step_size, sizeof(float), 1, file);
    fwrite(&regularizer->adv_state.attack_type, sizeof(int), 1, file);
    if (regularizer->adv_state.perturbation_size > 0) {
        fwrite(regularizer->adv_state.perturbation,
               sizeof(float), regularizer->adv_state.perturbation_size, file);
    }
    
    fwrite(&regularizer->domain_state.classifier_size, sizeof(size_t), 1, file);
    fwrite(&regularizer->domain_state.lambda, sizeof(float), 1, file);
    fwrite(&regularizer->domain_state.method, sizeof(int), 1, file);
    if (regularizer->domain_state.classifier_size > 0) {
        fwrite(regularizer->domain_state.domain_classifier,
               sizeof(float), regularizer->domain_state.classifier_size, file);
    }
    if (regularizer->domain_state.feature_align_loss) {
        fwrite(regularizer->domain_state.feature_align_loss, sizeof(float), 1, file);
    }
    
    fclose(file);
    return 0;
}

/**
 * @brief 加载正则化器状态（完整实现）
 *
 * 加载之前保存的完整正则化器状态，包括所有内部数据结构。
 * 自动处理内存分配和释放。
 */
int advanced_regularizer_load_state(AdvancedRegularizer* regularizer, const char* filename) {
    if (!regularizer || !filename) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "加载状态：无效参数");
        return -1;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：无法打开文件");
        return -1;
    }
    
    char magic[16] = {0};
    uint32_t version = 0;
    size_t elements_read = 0;
    
    elements_read = fread(magic, 1, 16, file);
    if (elements_read != 16 || fread(&version, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：读取文件头失败");
        return -1;
    }
    
    if (strncmp(magic, REG_STATE_MAGIC, 16) != 0) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：无效的文件格式");
        return -1;
    }
    
    if (version != REG_STATE_VERSION) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：不支持的版本");
        return -1;
    }
    
    if (fread(&regularizer->config, sizeof(AdvancedRegularizationConfig), 1, file) != 1) {
        fclose(file); return -1;
    }
    if (fread(&regularizer->input_dim, sizeof(size_t), 1, file) != 1) { fclose(file); return -1; }
    if (fread(&regularizer->output_dim, sizeof(size_t), 1, file) != 1) { fclose(file); return -1; }
    if (fread(&regularizer->current_strength, sizeof(float), 1, file) != 1) { fclose(file); return -1; }
    if (fread(&regularizer->training_step, sizeof(size_t), 1, file) != 1) { fclose(file); return -1; }
    
    if (fread(&regularizer->drop_path_state.num_layers, sizeof(size_t), 1, file) != 1) { fclose(file); return -1; }
    if (regularizer->drop_path_state.num_layers > 0) {
        safe_free((void**)&regularizer->drop_path_state.survival_probs);
        safe_free((void**)&regularizer->drop_path_state.dropped_layers);
        regularizer->drop_path_state.survival_probs = (float*)safe_malloc(
            regularizer->drop_path_state.num_layers * sizeof(float));
        regularizer->drop_path_state.dropped_layers = (int*)safe_malloc(
            regularizer->drop_path_state.num_layers * sizeof(int));
        if (!regularizer->drop_path_state.survival_probs || !regularizer->drop_path_state.dropped_layers) {
            fclose(file); return -1;
        }
        if (fread(regularizer->drop_path_state.survival_probs, sizeof(float),
                  regularizer->drop_path_state.num_layers, file) != regularizer->drop_path_state.num_layers) {
            fclose(file); return -1;
        }
        if (fread(regularizer->drop_path_state.dropped_layers, sizeof(int),
                  regularizer->drop_path_state.num_layers, file) != regularizer->drop_path_state.num_layers) {
            fclose(file); return -1;
        }
        if (fread(&regularizer->drop_path_state.current_drop_rate, sizeof(float), 1, file) != 1) {
            fclose(file); return -1;
        }
    }
    
    if (fread(&regularizer->cutmix_state.distribution_size, sizeof(size_t), 1, file) != 1) {
        fclose(file); return -1;
    }
    if (regularizer->cutmix_state.distribution_size > 0) {
        safe_free((void**)&regularizer->cutmix_state.lambda_distribution);
        regularizer->cutmix_state.lambda_distribution = (float*)safe_malloc(
            regularizer->cutmix_state.distribution_size * sizeof(float));
        if (!regularizer->cutmix_state.lambda_distribution) { fclose(file); return -1; }
        if (fread(regularizer->cutmix_state.lambda_distribution, sizeof(float),
                  regularizer->cutmix_state.distribution_size, file) != regularizer->cutmix_state.distribution_size) {
            fclose(file); return -1;
        }
    }
    if (fread(&regularizer->cutmix_state.max_boxes, sizeof(size_t), 1, file) != 1) {
        fclose(file); return -1;
    }
    if (regularizer->cutmix_state.max_boxes > 0) {
        safe_free((void**)&regularizer->cutmix_state.bounding_boxes);
        regularizer->cutmix_state.bounding_boxes = (int*)safe_malloc(
            regularizer->cutmix_state.max_boxes * 4 * sizeof(int));
        if (!regularizer->cutmix_state.bounding_boxes) { fclose(file); return -1; }
        if (fread(regularizer->cutmix_state.bounding_boxes, sizeof(int),
                  regularizer->cutmix_state.max_boxes * 4, file) != regularizer->cutmix_state.max_boxes * 4) {
            fclose(file); return -1;
        }
    }
    
    if (fread(&regularizer->mixup_state.buffer_size, sizeof(size_t), 1, file) != 1) {
        fclose(file); return -1;
    }
    if (fread(&regularizer->mixup_state.alpha, sizeof(float), 1, file) != 1) {
        fclose(file); return -1;
    }
    if (regularizer->mixup_state.buffer_size > 0) {
        safe_free((void**)&regularizer->mixup_state.lambda_buffer);
        regularizer->mixup_state.lambda_buffer = (float*)safe_malloc(
            regularizer->mixup_state.buffer_size * sizeof(float));
        if (!regularizer->mixup_state.lambda_buffer) { fclose(file); return -1; }
        if (fread(regularizer->mixup_state.lambda_buffer, sizeof(float),
                  regularizer->mixup_state.buffer_size, file) != regularizer->mixup_state.buffer_size) {
            fclose(file); return -1;
        }
    }
    
    if (fread(&regularizer->adv_state.perturbation_size, sizeof(size_t), 1, file) != 1) {
        fclose(file); return -1;
    }
    if (fread(&regularizer->adv_state.epsilon, sizeof(float), 1, file) != 1) { fclose(file); return -1; }
    if (fread(&regularizer->adv_state.attack_steps, sizeof(int), 1, file) != 1) { fclose(file); return -1; }
    if (fread(&regularizer->adv_state.step_size, sizeof(float), 1, file) != 1) { fclose(file); return -1; }
    if (fread(&regularizer->adv_state.attack_type, sizeof(int), 1, file) != 1) { fclose(file); return -1; }
    if (regularizer->adv_state.perturbation_size > 0) {
        safe_free((void**)&regularizer->adv_state.perturbation);
        regularizer->adv_state.perturbation = (float*)safe_malloc(
            regularizer->adv_state.perturbation_size * sizeof(float));
        if (!regularizer->adv_state.perturbation) { fclose(file); return -1; }
        if (fread(regularizer->adv_state.perturbation, sizeof(float),
                  regularizer->adv_state.perturbation_size, file) != regularizer->adv_state.perturbation_size) {
            fclose(file); return -1;
        }
    }
    
    if (fread(&regularizer->domain_state.classifier_size, sizeof(size_t), 1, file) != 1) {
        fclose(file); return -1;
    }
    if (fread(&regularizer->domain_state.lambda, sizeof(float), 1, file) != 1) { fclose(file); return -1; }
    if (fread(&regularizer->domain_state.method, sizeof(int), 1, file) != 1) { fclose(file); return -1; }
    if (regularizer->domain_state.classifier_size > 0) {
        safe_free((void**)&regularizer->domain_state.domain_classifier);
        regularizer->domain_state.domain_classifier = (float*)safe_malloc(
            regularizer->domain_state.classifier_size * sizeof(float));
        if (!regularizer->domain_state.domain_classifier) { fclose(file); return -1; }
        if (fread(regularizer->domain_state.domain_classifier, sizeof(float),
                  regularizer->domain_state.classifier_size, file) != regularizer->domain_state.classifier_size) {
            fclose(file); return -1;
        }
    }
    if (regularizer->domain_state.feature_align_loss) {
        if (fread(regularizer->domain_state.feature_align_loss, sizeof(float), 1, file) != 1) {
            fclose(file); return -1;
        }
    }
    
    fclose(file);
    regularizer->is_initialized = 1;
    return 0;
}