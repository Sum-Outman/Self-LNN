/**
 * @file quaternion_lnn.h
 * @brief 四元数增强液态神经网络
 * 
 * 基于四元数数学的液态神经网络增强模块，提供旋转不变性、空间变换和
 * 四维超复数表示能力，显著增强模型在处理空间关系和旋转变换方面的能力。
 * 
 * 特性：
 * 1. 四元数隐藏状态表示：将传统标量隐藏状态升级为四元数表示
 * 2. 四元数激活函数：使用四元数非线性激活增强表达能力
 * 3. 旋转不变性：利用四元数实现旋转不变的特征学习
 * 4. 时空一致性：通过四元数保持时空变换的一致性
 * 5. 参数效率：四元数表示提供更紧凑的参数空间
 */

#ifndef SELFLNN_QUATERNION_LNN_H
#define SELFLNN_QUATERNION_LNN_H

#include <stddef.h>
#include "selflnn/utils/math_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 四元数增强液态神经网络配置
 */
typedef struct {
    size_t input_size;           /**< 输入向量大小（标量维度） */
    size_t quaternion_hidden_size; /**< 四元数隐藏状态大小（四元数数量） */
    size_t output_size;          /**< 输出向量大小（标量维度） */
    float learning_rate;         /**< 学习率 */
    float time_constant;         /**< 时间常数（秒） */
    float noise_std;             /**< 噪声标准差 */
    int enable_training;         /**< 是否启用训练 */
    int enable_adaptation;       /**< 是否启用参数自适应 */
    int enable_evolution;        /**< 是否启用演化 */
    
    // 四元数特定配置
    int use_quaternion_weights;      /**< 是否使用四元数权重 */
    int use_quaternion_activation;   /**< 是否使用四元数激活函数 */
    float rotation_invariance_strength; /**< 旋转不变性强度（0-1） */
    int preserve_norm;                /**< 是否保持四元数范数 */
    int auto_init_weights;           /**< 是否自动初始化权重 (1: 自动初始化, 0: 手动初始化) */
    
    // GPU加速配置
    int use_gpu_acceleration;        /**< 是否使用GPU加速 (0: CPU, 1: GPU) */
    int gpu_backend;                 /**< GPU后端类型 (0: CPU, 1: CUDA, 2: OpenCL, 3: Vulkan, 4: Metal) */
    int gpu_device_index;            /**< GPU设备索引 (默认: 0) */
    int gpu_max_batch_size;          /**< GPU最大批处理大小 (默认: 1024) */
    int gpu_enable_async;            /**< 是否启用异步计算 (0: 同步, 1: 异步) */
    
    // 旋转正则化配置
    float rotation_regularization_strength; /**< 旋转正则化强度（0=禁用，推荐0.01~0.5） */
    int rotation_samples_per_step;          /**< 每次训练旋转采样数（推荐3~10） */
    int use_adaptive_regularization;        /**< 是否自适应调整正则化强度 */
    
    // 四元数-拉普拉斯联合优化配置
    int enable_laplace_optimization;        /**< 是否启用拉普拉斯联合优化 */
    float laplace_cutoff_frequency;         /**< 拉普拉斯截止频率（Hz，0=自动） */
    float laplace_damping_target;           /**< 目标阻尼比（0.0~1.0，推荐0.7） */
} QuaternionLNNConfig;

/**
 * @brief 四元数增强液态神经网络句柄
 */
typedef struct QuaternionLNN QuaternionLNN;

/**
 * @brief 四元数网络状态
 */
typedef struct {
    Quaternion* hidden_state;        /**< 四元数隐藏状态数组 */
    Quaternion* cell_state;          /**< 四元数细胞状态数组 */
    size_t state_size;               /**< 状态大小（四元数数量） */
    float timestamp;                 /**< 时间戳 */
    float stability;                 /**< 稳定性指标 */
    float rotation_consistency;      /**< 旋转一致性指标 */
} QuaternionLNNState;

/**
 * @brief 四元数前向传播结果
 */
typedef struct {
    float* output;                   /**< 标量输出向量 */
    Quaternion* quaternion_output;   /**< 四元数输出（可选） */
    float loss;                      /**< 损失值 */
    float rotation_loss;             /**< 旋转一致性损失 */
    int iteration_count;             /**< 迭代次数 */
} QuaternionLNNResult;

/**
 * @brief 创建四元数增强液态神经网络
 * 
 * @param config 网络配置
 * @return QuaternionLNN* 网络句柄，失败返回NULL
 */
QuaternionLNN* quaternion_lnn_create(const QuaternionLNNConfig* config);

/**
 * @brief 释放四元数增强液态神经网络
 * 
 * @param network 网络句柄
 */
void quaternion_lnn_free(QuaternionLNN* network);

/**
 * @brief 四元数前向传播
 * 
 * 将标量输入转换为四元数表示，通过四元数网络进行处理，
 * 最后将四元数输出转换回标量表示。
 * 
 * @param network 网络句柄
 * @param input 标量输入向量（大小：input_size）
 * @param output 标量输出向量缓冲区（大小：output_size）
 * @param state 网络状态输出（可选）
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_forward(QuaternionLNN* network, const float* input, 
                          float* output, QuaternionLNNState* state);

/**
 * @brief 四元数反向传播（训练）
 * 
 * 使用四元数梯度进行训练，支持旋转不变性正则化。
 * 
 * @param network 网络句柄
 * @param target 目标输出向量
 * @param result 训练结果输出（可选）
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_backward(QuaternionLNN* network, const float* target,
                           QuaternionLNNResult* result);

/**
 * @brief 四元数批量训练
 * 
 * 对批量数据进行四元数增强训练，支持数据增强和旋转增强。
 * 
 * @param network 网络句柄
 * @param inputs 输入数据数组（batch_size x input_size）
 * @param targets 目标数据数组（batch_size x output_size）
 * @param batch_size 批量大小
 * @param result 训练结果输出（可选）
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_train_batch(QuaternionLNN* network, 
                              const float* inputs, const float* targets,
                              size_t batch_size, QuaternionLNNResult* result);

/**
 * @brief 应用旋转增强
 * 
 * 对输入数据应用随机旋转增强，提高模型的旋转鲁棒性。
 * 
 * @param network 网络句柄
 * @param input 输入向量
 * @param rotated_input 旋转后的输入输出缓冲区
 * @param rotation_angle 旋转角度（弧度，可选）
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_apply_rotation_augmentation(QuaternionLNN* network,
                                              const float* input, float* rotated_input,
                                              float* rotation_angle);

/**
 * @brief 获取四元数隐藏状态
 * 
 * @param network 网络句柄
 * @param state 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_get_state(QuaternionLNN* network, QuaternionLNNState* state);

/**
 * @brief 重置四元数网络状态
 * 
 * @param network 网络句柄
 */
void quaternion_lnn_reset(QuaternionLNN* network);

/**
 * @brief 保存四元数网络到文件
 * 
 * @param network 网络句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_save(QuaternionLNN* network, const char* filename);

/**
 * @brief 从文件加载四元数网络
 * 
 * @param network 网络句柄（可为NULL，将创建新网络）
 * @param filename 文件名
 * @return QuaternionLNN* 网络句柄，失败返回NULL
 */
QuaternionLNN* quaternion_lnn_load(const char* filename);

/**
 * @brief 获取四元数网络统计信息
 * 
 * @param network 网络句柄
 * @param avg_activation 平均激活度
 * @param max_activation 最大激活度
 * @param rotation_consistency 旋转一致性
 * @param norm_preservation 范数保持度
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_get_stats(QuaternionLNN* network,
                            float* avg_activation, float* max_activation,
                            float* rotation_consistency, float* norm_preservation);

/**
 * @brief 设置四元数网络配置
 * 
 * @param network 网络句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_set_config(QuaternionLNN* network, const QuaternionLNNConfig* config);

/**
 * @brief 获取四元数网络配置
 * 
 * @param network 网络句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_get_config(QuaternionLNN* network, QuaternionLNNConfig* config);

/**
 * @brief 设置四元数旋转不变性正则化强度
 * 
 * 增强四元数旋转不变性的训练正则化。该函数在训练过程中添加
 * 旋转一致性损失项，强制网络对旋转输入产生旋转等变输出。
 * 正则化方法：对输入施加随机旋转，计算旋转前后的输出差异，
 * 将差异作为额外损失加入训练。
 * 
 * @param network 网络句柄
 * @param strength 正则化强度（0=禁用，推荐0.01~0.5）
 * @param rotation_samples 每次训练采样的旋转数量（推荐3~10）
 * @param use_adaptive 是否自适应调整强度（基于旋转一致性指标）
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_set_rotation_regularization(QuaternionLNN* network,
                                               float strength,
                                               int rotation_samples,
                                               int use_adaptive);

/**
 * @brief 计算四元数旋转正则化损失
 * 
 * 对输入batch中的每个样本施加随机轴-角旋转，
 * 计算旋转前后网络输出的MSE作为正则化损失。
 * 
 * @param network 网络句柄
 * @param inputs 原始输入batch [batch_size][input_size]
 * @param batch_size batch大小
 * @param regularization_loss 输出正则化损失值
 * @param rotation_consistency 输出旋转一致性指标（0~1，越高越好）
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_compute_rotation_regularization(QuaternionLNN* network,
                                                    const float* inputs,
                                                    size_t batch_size,
                                                    float* regularization_loss,
                                                    float* rotation_consistency);

/**
 * @brief 四元数-拉普拉斯联合优化
 * 
 * 使用拉普拉斯频域分析增强四元数网络的训练稳定性。
 * 将四元数隐藏状态的演化视为连续动态系统，通过拉普拉斯变换
 * 分析其频率特性，自动调整时间常数和阻尼比以抑制振荡。
 * 
 * @param network 网络句柄
 * @param state_history 历史状态数组 [history_len][quaternion_hidden_size]
 * @param state_history_count 历史状态数量
 * @param laplace_cutoff_freq 拉普拉斯截止频率（Hz，0=自动）
 * @param damping_target 目标阻尼比（0.0~1.0，推荐0.7）
 * @param stability_score 输出稳定性评分（0~1）
 * @return int 成功返回0，失败返回-1
 */
int quaternion_lnn_laplace_optimize(QuaternionLNN* network,
                                     const QuaternionLNNState* state_history,
                                     size_t state_history_count,
                                     float laplace_cutoff_freq,
                                     float damping_target,
                                     float* stability_score);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_QUATERNION_LNN_H */