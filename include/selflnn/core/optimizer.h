/**
 * @file optimizer.h
 * @brief 优化器接口头文件
 * 
 * 提供各种优化器（梯度下降、Adam、RMSprop等）的接口。
 * 用于训练液态神经网络和其他模型。
 */

#ifndef SELFLNN_CORE_OPTIMIZER_H
#define SELFLNN_CORE_OPTIMIZER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 优化器类型枚举
 */
typedef enum {
    OPTIMIZER_SGD = 0,        /**< 随机梯度下降 */
    OPTIMIZER_MOMENTUM = 1,   /**< 动量梯度下降 */
    OPTIMIZER_ADAGRAD = 2,    /**< AdaGrad */
    OPTIMIZER_RMSPROP = 3,    /**< RMSprop */
    OPTIMIZER_ADAM = 4,       /**< Adam */
    OPTIMIZER_ADAMW = 5,      /**< AdamW */
    OPTIMIZER_ADADELTA = 6,   /**< AdaDelta */
    OPTIMIZER_LAMB = 7,       /**< LAMB: Layer-wise Adaptive Moments optimizer for Batch */
    OPTIMIZER_LARS = 8,       /**< LARS: Layer-wise Adaptive Rate Scaling */
    OPTIMIZER_RANGER = 9,     /**< Ranger: RAdam + LookAhead 融合优化器 */
    OPTIMIZER_NOVOGRAD = 10   /**< NovoGrad: 逐层二阶矩归一化优化器 */
} OptimizerType;

/**
 * @brief 优化器配置结构体
 */
typedef struct {
    OptimizerType type;       /**< 优化器类型 */
    float learning_rate;      /**< 学习率 */
    float momentum;           /**< 动量系数（仅对动量优化器有效） */
    float beta1;              /**< Adam beta1参数 */
    float beta2;              /**< Adam beta2参数 */
    float epsilon;            /**< 数值稳定性常数 */
    float weight_decay;       /**< 权重衰减（L2正则化） */
    int use_nesterov;         /**< 是否使用Nesterov动量 */
    int amsgrad;              /**< 是否使用AMSGrad变体 */
    float lars_trust_coef;    /**< LARS/LAMB信任系数（默认0.001） */
    int lookahead_k;          /**< Ranger LookAhead步数（默认5） */
    float lookahead_alpha;    /**< Ranger LookAhead平滑系数（默认0.5） */
} OptimizerConfig;

/**
 * @brief 多参数组结构体（ZSFWS-023修复：支持多组参数独立更新）
 * 
 * 优化器通过此结构一次性更新多个参数组（如输入权重、门控权重、
 * 隐藏权重、时间常数、偏置等），每组可独立提供自己的参数和梯度指针。
 * 优化器内部维护跨组的连续动量/速度缓冲区，确保所有参数组共享
 * 相同的优化器状态演化。
 */
typedef struct {
    float* parameters;         /**< 指向该参数组的参数数组（原地更新） */
    const float* gradients;    /**< 指向该参数组的梯度数组（只读） */
    size_t num_params;         /**< 该组参数的数量 */
} OptimizerParamGroup;

/**
 * @brief 优化器句柄（不透明类型）
 */
typedef struct Optimizer Optimizer;

/**
 * @brief 创建优化器实例
 * 
 * @param config 优化器配置
 * @return Optimizer* 优化器句柄，失败返回NULL
 */
Optimizer* optimizer_create(const OptimizerConfig* config);

/**
 * @brief 释放优化器实例
 * 
 * @param optimizer 优化器句柄
 */
void optimizer_free(Optimizer* optimizer);

/**
 * @brief 更新参数（一步优化）
 * 
 * @param optimizer 优化器句柄
 * @param parameters 参数数组
 * @param gradients 梯度数组
 * @param num_params 参数数量
 * @param step 当前步数（用于学习率调度）
 * @return int 成功返回0，失败返回-1
 */
int optimizer_step(Optimizer* optimizer, float* parameters, const float* gradients,
                   size_t num_params, size_t step);

/**
 * @brief 多参数组更新（ZSFWS-023修复：一次更新多组独立参数）
 * 
 * 该函数在一次调用中更新多个参数组，每组独立指定参数指针、梯度指针和大小。
 * 适用于CfC/LNN等具有多组参数（输入权重、门控权重、隐藏权重、时间常数、偏置等）
 * 的复杂网络结构，避免手动memcpy展平/回写。
 * 
 * 优化器内部将所有参数组映射到连续的动量/速度缓冲区上，
 * 确保Adam/RMSprop等自适应优化器的状态正确更新。
 * 
 * @param optimizer 优化器句柄
 * @param groups 参数组数组，每组独立指定params/grads/num_params
 * @param num_groups 参数组数量
 * @param step 当前步数（用于学习率调度和偏差修正）
 * @return int 成功返回0，失败返回-1
 */
int optimizer_update_multi_group(Optimizer* optimizer, OptimizerParamGroup* groups,
                                  int num_groups, size_t step);

/**
 * @brief 重置优化器状态（用于新训练周期）
 * 
 * @param optimizer 优化器句柄
 */
void optimizer_reset(Optimizer* optimizer);

/**
 * @brief 获取优化器配置
 * 
 * @param optimizer 优化器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int optimizer_get_config(const Optimizer* optimizer, OptimizerConfig* config);

/**
 * @brief 设置学习率
 * 
 * @param optimizer 优化器句柄
 * @param learning_rate 新的学习率
 * @return int 成功返回0，失败返回-1
 */
int optimizer_set_learning_rate(Optimizer* optimizer, float learning_rate);

/**
 * @brief AdamW权重衰减步进（含解耦权重衰减）
 * 
 * 直接操作底层数组的AdamW更新，适合外部独立调用。
 * @param params 参数数组（原地更新）
 * @param grads 梯度数组
 * @param m 一阶动量缓冲区
 * @param v 二阶动量缓冲区
 * @param n 参数数量
 * @param lr 学习率
 * @param beta1 一阶动量衰减因子
 * @param beta2 二阶动量衰减因子
 * @param eps 防除零常数
 * @param weight_decay 权重衰减系数
 * @param step 当前步数（用于偏差校正）
 * @return int 成功返回0
 */
int optimizer_adamw_step(float* params, float* grads, float* m, float* v,
                         size_t n, float lr, float beta1, float beta2,
                         float eps, float weight_decay, int step);

/**
 * @brief 余弦退火学习率调度（带热重启）
 * @param base_lr 基础学习率
 * @param min_lr 最小学习率
 * @param epoch 当前epoch
 * @param T_0 首次重启周期
 * @param T_mult 重启周期倍增因子
 * @return float 调整后的学习率
 */
float lr_cosine_annealing(float base_lr, float min_lr, int epoch, int T_0, int T_mult);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_CORE_OPTIMIZER_H