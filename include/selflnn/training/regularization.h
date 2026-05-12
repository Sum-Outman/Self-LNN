/**
 * @file regularization.h
 * @brief 高级正则化系统接口
 * 
 * 提供高级正则化技术，包括DropPath、Stochastic Depth、CutMix、MixUp、
 * 对抗训练、域自适应等。 ，提供完整的正则化算法。
 */

#ifndef SELFLNN_REGULARIZATION_H
#define SELFLNN_REGULARIZATION_H

#include "selflnn/core/lnn.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 高级正则化技术类型枚举
 */
typedef enum {
    ADV_REG_NONE = 0,                 /**< 无高级正则化 */
    ADV_REG_DROP_PATH = 1,            /**< DropPath（随机深度） */
    ADV_REG_STOCHASTIC_DEPTH = 2,     /**< 随机深度 */
    ADV_REG_CUTMIX = 3,               /**< CutMix数据增强 */
    ADV_REG_MIXUP = 4,                /**< MixUp数据增强 */
    ADV_REG_ADVERSARIAL = 5,          /**< 对抗训练 */
    ADV_REG_DOMAIN_ADAPTATION = 6,    /**< 域自适应 */
    ADV_REG_SPATIAL_DROPOUT = 7,      /**< 空间Dropout */
    ADV_REG_DROPCONNECT = 8,          /**< DropConnect */
    ADV_REG_SWITCHABLE_NORM = 9,      /**< 可切换归一化 */
    ADV_REG_ENSEMBLE = 10             /**< 集成正则化 */
} AdvancedRegularizationType;

/**
 * @brief 对抗训练攻击类型枚举
 */
typedef enum {
    ADV_ATTACK_FGSM = 0,              /**< 快速梯度符号方法 */
    ADV_ATTACK_PGD = 1,               /**< 投影梯度下降 */
    ADV_ATTACK_CW = 2,                /**< Carlini-Wagner攻击 */
    ADV_ATTACK_AUTOATTACK = 3         /**< 自动攻击 */
} AdversarialAttackType;

/**
 * @brief 域自适应方法枚举
 */
typedef enum {
    DOMAIN_ADAPT_DANN = 0,            /**< 域对抗神经网络 */
    DOMAIN_ADAPT_MMD = 1,             /**< 最大均值差异 */
    DOMAIN_ADAPT_CORAL = 2,           /**< CORAL对齐 */
    DOMAIN_ADAPT_ADDA = 3             /**< 对抗判别域自适应 */
} DomainAdaptationMethod;

/**
 * @brief 高级正则化配置
 */
typedef struct {
    AdvancedRegularizationType type;  /**< 正则化类型 */
    
    /* DropPath/Stochastic Depth参数 */
    float drop_path_rate;             /**< DropPath丢弃率 */
    float survival_probability;       /**< 生存概率（随机深度） */
    int drop_path_mode;               /**< DropPath模式（0=均匀，1=线性衰减） */
    
    /* CutMix参数 */
    float cutmix_alpha;               /**< CutMix Beta分布参数alpha */
    float cutmix_probability;         /**< CutMix应用概率 */
    int cutmix_version;               /**< CutMix版本（1=原始，2=增强） */
    
    /* MixUp参数 */
    float mixup_alpha;                /**< MixUp Beta分布参数alpha */
    float mixup_probability;          /**< MixUp应用概率 */
    
    /* 跨模态混合参数 */
    float multimodal_mix_probability; /**< 跨模态混合应用概率 */
    
    /* 对抗训练参数 */
    AdversarialAttackType attack_type; /**< 攻击类型 */
    float adversarial_epsilon;        /**< 对抗扰动大小 */
    int adversarial_steps;            /**< 对抗步数（PGD） */
    float adversarial_step_size;      /**< 对抗步长 */
    float adversarial_alpha;          /**< 对抗训练强度 */
    
    /* 域自适应参数 */
    DomainAdaptationMethod domain_method; /**< 域自适应方法 */
    float domain_lambda;              /**< 域损失权重 */
    int domain_adapt_iterations;      /**< 域自适应迭代次数 */
    
    /* 通用参数 */
    float strength;                   /**< 正则化强度（0-1） */
    int apply_during_training;        /**< 是否在训练期间应用 */
    int apply_during_validation;      /**< 是否在验证期间应用 */
    int enable_scheduling;            /**< 是否启用调度（随时间调整强度） */
    float schedule_decay;             /**< 调度衰减率 */
    
    /* 集成参数 */
    int ensemble_size;                /**< 集成模型数量 */
    float ensemble_diversity_weight;  /**< 集成多样性权重 */
} AdvancedRegularizationConfig;

/**
 * @brief CutMix样本对
 */
typedef struct {
    float* mixed_input;               /**< 混合输入 */
    float* mixed_target;              /**< 混合目标 */
    float lambda;                     /**< 混合系数 */
    int from_sample_a;                /**< 来自样本A的索引 */
    int from_sample_b;                /**< 来自样本B的索引 */
} CutMixPair;

/**
 * @brief MixUp样本对
 */
typedef struct {
    float* mixed_input;               /**< 混合输入 */
    float* mixed_target;              /**< 混合目标 */
    float lambda;                     /**< 混合系数 */
} MixUpPair;

/**
 * @brief 对抗样本
 */
typedef struct {
    float* adversarial_input;         /**< 对抗输入 */
    float* original_input;            /**< 原始输入 */
    float perturbation_norm;          /**< 扰动范数 */
    int attack_success;               /**< 攻击是否成功 */
} AdversarialSample;

/**
 * @brief 域自适应状态
 */
typedef struct {
    float* domain_classifier;         /**< 域分类器权重 */
    size_t classifier_size;           /**< 分类器大小 */
    float* feature_align_loss;        /**< 特征对齐损失 */
    DomainAdaptationMethod method;    /**< 域自适应方法 */
    float lambda;                     /**< 域损失权重 */
} DomainAdaptationState;

/**
 * @brief 高级正则化器句柄
 */
typedef struct AdvancedRegularizer AdvancedRegularizer;

/**
 * @brief 创建高级正则化器
 * 
 * @param config 正则化配置
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @return AdvancedRegularizer* 正则化器句柄，失败返回NULL
 */
AdvancedRegularizer* advanced_regularizer_create(const AdvancedRegularizationConfig* config,
                                                 size_t input_dim, size_t output_dim);

/**
 * @brief 释放高级正则化器
 * 
 * @param regularizer 正则化器句柄
 */
void advanced_regularizer_free(AdvancedRegularizer* regularizer);

/**
 * @brief 应用DropPath正则化
 * 
 * @param regularizer 正则化器句柄
 * @param network 神经网络
 * @param layer_indices 应用DropPath的层索引数组
 * @param num_layers 层数
 * @param training 是否在训练模式
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_apply_drop_path(AdvancedRegularizer* regularizer,
                                         LNN* network,
                                         const int* layer_indices,
                                         size_t num_layers,
                                         int training);

/**
 * @brief 应用CutMix数据增强
 * 
 * @param regularizer 正则化器句柄
 * @param inputs 输入数据（批量×特征）
 * @param targets 目标数据（批量×类别）
 * @param batch_size 批量大小
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param augmented_inputs 增强后的输入输出
 * @param augmented_targets 增强后的目标输出
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_apply_cutmix(AdvancedRegularizer* regularizer,
                                      const float* inputs, const float* targets,
                                      size_t batch_size, size_t input_dim, size_t output_dim,
                                      float* augmented_inputs, float* augmented_targets);

/**
 * @brief 应用MixUp数据增强
 * 
 * @param regularizer 正则化器句柄
 * @param inputs 输入数据
 * @param targets 目标数据
 * @param batch_size 批量大小
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param augmented_inputs 增强后的输入输出
 * @param augmented_targets 增强后的目标输出
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_apply_mixup(AdvancedRegularizer* regularizer,
                                     const float* inputs, const float* targets,
                                     size_t batch_size, size_t input_dim, size_t output_dim,
                                     float* augmented_inputs, float* augmented_targets);

/**
 * @brief 应用跨模态数据混合增强
 * 
 * 在单模型多模态系统中，将不同样本的不同模态块互换，
 * 生成混合了多种模态特征的训练样本。
 * 
 * 模态划分比例: 视觉40%, 文本20%, 语音20%, 传感器20%
 * 
 * @param regularizer 正则化器句柄
 * @param inputs 输入数据
 * @param targets 目标数据
 * @param batch_size 批量大小
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param augmented_inputs 增强后的输入输出
 * @param augmented_targets 增强后的目标输出
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_apply_multimodal_mix(AdvancedRegularizer* regularizer,
                                               const float* inputs, const float* targets,
                                               size_t batch_size, size_t input_dim, size_t output_dim,
                                               float* augmented_inputs, float* augmented_targets);

/**
 * @brief 生成对抗样本
 * 
 * @param regularizer 正则化器句柄
 * @param network 神经网络
 * @param inputs 原始输入
 * @param targets 目标
 * @param batch_size 批量大小
 * @param input_dim 输入维度
 * @param adversarial_inputs 对抗样本输出
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_generate_adversarial(AdvancedRegularizer* regularizer,
                                              LNN* network,
                                              const float* inputs, const float* targets,
                                              size_t batch_size, size_t input_dim,
                                              float* adversarial_inputs);

/**
 * @brief 应用域自适应
 * 
 * @param regularizer 正则化器句柄
 * @param source_inputs 源域输入
 * @param target_inputs 目标域输入
 * @param batch_size 批量大小
 * @param input_dim 输入维度
 * @param domain_labels 域标签输出（0=源域，1=目标域）
 * @param domain_loss 域损失输出
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_apply_domain_adaptation(AdvancedRegularizer* regularizer,
                                                 const float* source_inputs,
                                                 const float* target_inputs,
                                                 size_t batch_size, size_t input_dim,
                                                 float* domain_labels,
                                                 float* domain_loss);

/**
 * @brief 应用空间Dropout
 * 
 * @param regularizer 正则化器句柄
 * @param activations 激活值（空间维度：高度×宽度×通道）
 * @param height 高度
 * @param width 宽度
 * @param channels 通道数
 * @param dropout_rate Dropout率
 * @param training 是否在训练模式
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_apply_spatial_dropout(AdvancedRegularizer* regularizer,
                                               float* activations,
                                               size_t height, size_t width, size_t channels,
                                               float dropout_rate, int training);

/**
 * @brief 应用DropConnect
 * 
 * @param regularizer 正则化器句柄
 * @param weights 权重矩阵
 * @param rows 行数
 * @param cols 列数
 * @param dropout_rate Dropout率
 * @param training 是否在训练模式
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_apply_dropconnect(AdvancedRegularizer* regularizer,
                                           float* weights,
                                           size_t rows, size_t cols,
                                           float dropout_rate, int training);

/**
 * @brief 应用可切换归一化
 * 
 * @param regularizer 正则化器句柄
 * @param activations 激活值
 * @param batch_size 批量大小
 * @param height 高度
 * @param width 宽度
 * @param channels 通道数
 * @param norm_type 归一化类型（0=BN，1=IN，2=LN，3=GN）
 * @param training 是否在训练模式
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_apply_switchable_norm(AdvancedRegularizer* regularizer,
                                               float* activations,
                                               size_t batch_size, size_t height, size_t width, size_t channels,
                                               int norm_type, int training);

/**
 * @brief 更新正则化调度
 * 
 * @param regularizer 正则化器句柄
 * @param epoch 当前训练轮数
 * @param total_epochs 总训练轮数
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_update_schedule(AdvancedRegularizer* regularizer,
                                         size_t epoch, size_t total_epochs);

/**
 * @brief 获取当前正则化强度
 * 
 * @param regularizer 正则化器句柄
 * @return float 当前正则化强度
 */
float advanced_regularizer_get_strength(const AdvancedRegularizer* regularizer);

/**
 * @brief 获取正则化类型
 * 
 * @param regularizer 正则化器句柄
 * @return AdvancedRegularizationType 正则化类型
 */
AdvancedRegularizationType advanced_regularizer_get_type(const AdvancedRegularizer* regularizer);

/**
 * @brief 获取正则化丢弃率
 * 
 * @param regularizer 正则化器句柄
 * @return float 当前丢弃率
 */
float advanced_regularizer_get_drop_rate(const AdvancedRegularizer* regularizer);

/**
 * @brief 设置正则化强度
 * 
 * @param regularizer 正则化器句柄
 * @param strength 新的正则化强度
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_set_strength(AdvancedRegularizer* regularizer, float strength);

/**
 * @brief 保存正则化器状态
 * 
 * @param regularizer 正则化器句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_save_state(const AdvancedRegularizer* regularizer, const char* filename);

/**
 * @brief 加载正则化器状态
 * 
 * @param regularizer 正则化器句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int advanced_regularizer_load_state(AdvancedRegularizer* regularizer, const char* filename);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_REGULARIZATION_H */