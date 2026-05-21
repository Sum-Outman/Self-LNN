#ifndef SELFLNN_LOSS_H
#define SELFLNN_LOSS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file loss.h
 * @brief 损失函数类型定义
 */

/**
 * @brief 损失函数类型枚举
 */
typedef enum {
    LOSS_MSE = 0,           /**< 均方误差损失 */
    LOSS_MAE = 1,           /**< 平均绝对误差损失 */
    LOSS_HUBER = 2,         /**< Huber损失 */
    LOSS_CATEGORICAL_CROSSENTROPY = 3, /**< 分类交叉熵损失 */
    LOSS_BINARY_CROSSENTROPY = 4, /**< 二元交叉熵损失 */
    LOSS_KLD = 5,           /**< KL散度损失 */
    LOSS_COSINE = 6,        /**< 余弦相似度损失 */
    LOSS_CONTRASTIVE = 7,   /**< 对比损失 */
    LOSS_FOCAL = 8,         /**< Focal损失：处理类别不平衡 */
    LOSS_DICE = 9,          /**< Dice损失：图像分割相似度 */
    LOSS_TRIPLET = 10,      /**< Triplet损失：度量学习 */
    LOSS_QUANTILE = 11      /**< 分位数损失：分位数回归 */
} LossType;

/**
 * @brief 损失函数配置结构体
 */
typedef struct {
    float focal_gamma;       /**< Focal Loss聚焦参数（默认2.0） */
    float focal_alpha;       /**< Focal Loss类别平衡参数（默认0.25） */
    float dice_smooth;       /**< Dice Loss平滑因子（默认1e-6） */
    float triplet_margin;    /**< Triplet Loss边界值（默认1.0） */
    float quantile_tau;      /**< Quantile Loss分位数（默认0.5） */
} LossConfig;

/**
 * @brief 计算损失函数值（带配置参数）
 * 
 * @param predictions 预测值数组
 * @param targets 目标值数组
 * @param n 数组长度
 * @param loss_type 损失函数类型
 * @param config 损失函数配置（可为NULL使用默认值）
 * @return float 损失值
 */
float loss_compute_ex(const float* predictions, const float* targets, int n,
                       LossType loss_type, const LossConfig* config);

/**
 * @brief 计算损失函数梯度（带配置参数）
 * 
 * @param predictions 预测值数组
 * @param targets 目标值数组
 * @param n 数组长度
 * @param gradients 梯度输出数组
 * @param loss_type 损失函数类型
 * @param config 损失函数配置（可为NULL使用默认值）
 */
void loss_gradient_ex(const float* predictions, const float* targets, int n,
                       float* gradients, LossType loss_type, const LossConfig* config);

/**
 * @brief 计算损失函数值
 * 
 * @param predictions 预测值数组
 * @param targets 目标值数组
 * @param n 数组长度
 * @param loss_type 损失函数类型
 * @return float 损失值
 */
float loss_compute(const float* predictions, const float* targets, int n, LossType loss_type);

/**
 * @brief 计算损失函数梯度
 * 
 * @param predictions 预测值数组
 * @param targets 目标值数组
 * @param n 数组长度
 * @param gradients 梯度输出数组
 * @param loss_type 损失函数类型
 */
void loss_gradient(const float* predictions, const float* targets, int n, float* gradients, LossType loss_type);

#ifdef __cplusplus
}
#endif

/**
 * @brief 设置Focal Loss默认gamma参数（聚焦参数）
 * @param gamma 新的默认gamma值
 */
void loss_set_default_focal_gamma(float gamma);

/**
 * @brief 设置Focal Loss默认alpha参数（类别平衡参数）
 * @param alpha 新的默认alpha值
 */
void loss_set_default_focal_alpha(float alpha);

/**
 * @brief 设置Dice Loss默认平滑因子
 * @param smooth 新的默认平滑值
 */
void loss_set_default_dice_smooth(float smooth);

/**
 * @brief 设置Triplet Loss默认边界值
 * @param margin 新的默认边界值
 */
void loss_set_default_triplet_margin(float margin);

/**
 * @brief 设置Quantile Loss默认分位数
 * @param tau 新的默认分位数值
 */
void loss_set_default_quantile_tau(float tau);

#endif // SELFLNN_LOSS_H