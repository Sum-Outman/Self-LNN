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
 * @brief 模态类型枚举（ZSFWS-024：多模态输出适配）
 * 
 * 不同模态的输出具有不同的数值范围和分布特性，
 * 需要不同的损失函数类型来度量预测误差。
 */
#ifndef SELFLNN_MODALITY_TYPE_DEFINED
#define SELFLNN_MODALITY_TYPE_DEFINED
typedef enum {
    MODALITY_VISUAL = 0,       /**< 视觉特征：连续浮点[-1,1]或[0,1]，适用MSE/Huber/Cosine */
    MODALITY_TEXT_LOGITS = 1,  /**< 文本logits：未归一化logit，适用CrossEntropy */
    MODALITY_TEXT_EMBED = 2,   /**< 文本嵌入：连续浮点向量，适用MSE/Cosine */
    MODALITY_SENSOR = 3,       /**< 传感器数据：范围各异，适用Huber/MAE/Quantile */
    MODALITY_CONTROL = 4,      /**< 控制信号：连续值，适用MSE/Huber */
    MODALITY_AUDIO = 5,        /**< 音频特征：频域/时域连续值，适用MSE/MAE */
    MODALITY_CUSTOM = 6        /**< 自定义模态：由外部指定LossType */
} ModalityType;
#endif

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
    float huber_delta;       /**< Huber Loss delta阈值（默认1.0） */
    float quantile_tau;      /**< Quantile Loss分位数（默认0.5） */
} LossConfig;

/**
 * @brief 多模态损失段描述符（ZSFWS-024）
 * 
 * 将统一输出张量划分为多个模态段，每段使用独立的损失函数。
 * 最终损失为各段损失的加权和。
 */
typedef struct {
    ModalityType modality;     /**< 模态类型（若为MODALITY_CUSTOM则使用custom_loss_type） */
    int start_index;           /**< 在预测/目标数组中的起始索引 */
    int length;                /**< 该模态段的元素数量 */
    float weight;              /**< 该模态在总损失中的权重（默认1.0） */
    LossType custom_loss_type; /**< 自定义损失类型（仅当modality==MODALITY_CUSTOM时生效） */
    LossConfig loss_config;    /**< 该段的损失配置（focal_gamma等，可为全零使用默认值） */
} MultimodalLossSegment;

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

/* ============ ZSFWS-024: 多模态损失函数 API ============ */

/**
 * @brief 计算多模态损失函数值（ZSFWS-024）
 * 
 * 将统一输出数组按模态段划分，每段使用最适合该模态的损失函数，
 * 最终返回加权总损失。有效解决了视觉[-1,1]、文本discrete、传感器varied ranges
 * 等多尺度差异问题。
 * 
 * 模态→损失映射规则：
 *   MODALITY_VISUAL        → LOSS_HUBER (鲁棒于异常像素)
 *   MODALITY_TEXT_LOGITS   → LOSS_CATEGORICAL_CROSSENTROPY
 *   MODALITY_TEXT_EMBED    → LOSS_COSINE (方向敏感)
 *   MODALITY_SENSOR        → LOSS_HUBER (鲁棒于传感器噪声)
 *   MODALITY_CONTROL       → LOSS_MSE (精确控制)
 *   MODALITY_AUDIO         → LOSS_MAE (鲁棒于频域尖峰)
 *   MODALITY_CUSTOM        → 使用segment.custom_loss_type
 * 
 * @param predictions 统一预测数组（所有模态拼接）
 * @param targets 统一目标数组（所有模态拼接）
 * @param total_length 数组总长度（用于边界校验）
 * @param segments 模态段描述符数组
 * @param num_segments 模态段数量
 * @return float 加权总损失值
 */
float loss_compute_multimodal(const float* predictions, const float* targets,
                               int total_length,
                               const MultimodalLossSegment* segments,
                               int num_segments);

/**
 * @brief 多模态损失自适应梯度平衡 (ZSFZX-FIX-R4-1)
 *
 * 与loss_compute_multimodal功能相同，但use_adaptive=1时自动调整各模态段权重，
 * 使梯度范数均衡。消除手动调权的需求，防止模态崩塌。
 *
 * @param gradient_buffer 梯度缓冲区（用于计算各段梯度范数）
 * @param use_adaptive 0=使用固定权重, 1=启用自适应平衡
 * @return float 自适应加权总损失
 */
float loss_compute_multimodal_adaptive(const float* predictions, const float* targets,
                                        int total_length,
                                        const MultimodalLossSegment* segments,
                                        int num_segments,
                                        const float* gradient_buffer,
                                        int use_adaptive);

/**
 * @brief 计算多模态损失函数梯度（ZSFWS-024）
 * 
 * 对每个模态段独立计算梯度，然后拼接回统一的梯度数组。
 * 各段梯度在写入前被清零，确保跨模态段无梯度泄漏。
 * 
 * @param predictions 统一预测数组
 * @param targets 统一目标数组
 * @param total_length 数组总长度
 * @param gradients 统一梯度输出数组（调用者分配，函数填充）
 * @param segments 模态段描述符数组
 * @param num_segments 模态段数量
 */
void loss_gradient_multimodal(const float* predictions, const float* targets,
                               int total_length, float* gradients,
                               const MultimodalLossSegment* segments,
                               int num_segments);

/**
 * @brief 解析模态类型对应的默认损失函数（ZSFWS-024）
 * 
 * 将ModalityType映射到最合适的LossType。
 * 
 * @param modality 模态类型
 * @return LossType 推荐的损失函数类型
 */
LossType loss_get_default_for_modality(ModalityType modality);

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

/**
 * @brief 设置Huber Loss默认delta阈值
 * @param delta 新的默认delta值
 */
void loss_set_default_huber_delta(float delta);

#endif // SELFLNN_LOSS_H