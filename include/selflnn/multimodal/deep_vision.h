/**
 * @file deep_vision.h
 * @brief 深度视觉特征提取（基于CfC ODE连续动态系统）
 *
 * 基于液态神经网络CfC（Closed-form Continuous-time）的深度视觉特征提取。
 * 不使用任何CNN卷积层，全部使用CfC ODE微分方程驱动的连续时间演化：
 *   τ dh/dt = -h + σ(gate) ⊙ tanh(W * input + b)
 * 严格遵循：所有模态通过同一液态神经网络框架完成特征提取。
 * 纯C实现，不依赖任何第三方库。
 */

#ifndef SELFLNN_DEEP_VISION_H
#define SELFLNN_DEEP_VISION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CfC ODE层配置（替代CNN卷积层）
 *
 * 使用CfC闭式连续时间单元替代传统卷积特征提取。
 * 图像数据通过补丁编码后，各补丁状态在ODE驱动下连续演化。
 */
typedef struct {
    int input_dim;               /**< 输入特征维度 */
    int hidden_dim;              /**< 隐藏状态维度 */
    int num_layers;              /**< ODE层数（串联深度） */
    float time_constant;         /**< CfC时间常数τ，默认0.1 */
    float delta_t;               /**< ODE数值积分步长，默认0.05 */
    int use_adaptive_tau;        /**< 是否使用自适应时间常数，默认1 */
    float min_tau;               /**< 最小时间常数，默认0.01 */
    float max_tau;               /**< 最大时间常数，默认1.0 */
    int use_bias;                /**< 是否使用偏置，默认1 */
    int use_liquid_gate;         /**< 是否使用液态门控，默认1 */
    float noise_std;             /**< 噪声标准差（训练用），默认0.0 */
} CfcOdeLayerConfig;

/**
 * @brief CfC ODE层句柄
 */
typedef struct CfcOdeLayer CfcOdeLayer;

/**
 * @brief 获取默认CfC ODE层配置
 *
 * @return CfcOdeLayerConfig 默认配置
 */
CfcOdeLayerConfig cfc_ode_layer_get_default_config(void);

/**
 * @brief 创建CfC ODE层
 *
 * @param config 层配置
 * @return CfcOdeLayer* 层句柄，失败返回NULL
 */
CfcOdeLayer* cfc_ode_layer_create(const CfcOdeLayerConfig* config);

/**
 * @brief 释放CfC ODE层
 *
 * @param layer 层句柄
 */
void cfc_ode_layer_free(CfcOdeLayer* layer);

/**
 * @brief CfC ODE层前向传播
 *
 * 使用CfC闭式解执行连续时间状态演化：
 *   h(t+Δt) = CfC(h(t), input, W, τ)
 * 其中CfC闭式解为：
 *   h(t+Δt) = (1 - σ(gate)) ⊙ h(t) + σ(gate) ⊙ tanh(W_input * input + W_hidden * h(t) + b)
 *
 * @param layer 层句柄
 * @param input 输入特征向量 [input_dim]
 * @param output 输出状态向量 [hidden_dim]
 * @return int 成功返回0，失败返回-1
 */
int cfc_ode_layer_forward(CfcOdeLayer* layer,
                          const float* input,
                          float* output);

/**
 * @brief 多层CfC ODE堆叠前向传播
 *
 * 将多个CfC ODE层串联，每个层接收前一层的输出作为输入。
 * 形成深度连续时间特征演化网络。
 *
 * @param layers CfC ODE层指针数组
 * @param num_layers 层数
 * @param input 输入特征向量
 * @param output 最终输出特征向量
 * @return int 成功返回0，失败返回-1
 */
int cfc_ode_layers_forward(CfcOdeLayer** layers, int num_layers,
                           const float* input, float* output);

/**
 * @brief CfC ODE层反向传播（训练）
 *
 * @param layer 层句柄
 * @param dL_doutput 损失对输出的梯度 [hidden_dim]
 * @param input 前向传播时的输入 [input_dim]
 * @param output 前向传播时的输出 [hidden_dim]
 * @param dL_dinput 损失对输入的梯度输出 [input_dim]（可为NULL）
 * @param learning_rate 学习率
 * @return int 成功返回0，失败返回-1
 */
int cfc_ode_layer_backward(CfcOdeLayer* layer,
                           const float* dL_doutput,
                           const float* input,
                           const float* output,
                           float* dL_dinput,
                           float learning_rate);

/**
 * @brief 深度视觉处理配置
 *
 * 使用CfC ODE层替代CNN进行视觉特征提取。
 */
typedef struct {
    int image_width;             /**< 输入图像宽度 */
    int image_height;            /**< 输入图像高度 */
    int image_channels;          /**< 输入图像通道数 (1=灰度, 3=RGB) */
    int patch_size;              /**< 补丁大小（对图像分块后逐个ODE演化） */
    int output_dim;              /**< 输出特征维度 */
    int num_ode_layers;          /**< CfC ODE层数 */
    float time_constant;         /**< ODE时间常数 */
    float delta_t;               /**< ODE时间步长 */
} CfcVisionConfig;

/**
 * @brief 深度视觉处理器句柄
 */
typedef struct CfcVisionProcessor CfcVisionProcessor;

/**
 * @brief 获取默认深度视觉配置
 *
 * @return CfcVisionConfig 默认配置
 */
CfcVisionConfig cfc_vision_get_default_config(void);

/**
 * @brief 创建深度视觉处理器（CfC ODE架构）
 *
 * @param config 处理器配置
 * @return CfcVisionProcessor* 处理器句柄，失败返回NULL
 */
CfcVisionProcessor* cfc_vision_processor_create(const CfcVisionConfig* config);

/**
 * @brief 销毁深度视觉处理器
 *
 * @param processor 处理器句柄
 */
void cfc_vision_processor_destroy(CfcVisionProcessor* processor);

/**
 * @brief 提取深度视觉特征（CfC ODE驱动）
 *
 * 图像数据通过以下管道处理：
 * 1. 补丁划分：将图像划分为固定大小的补丁
 * 2. 补丁展平：每个补丁展平为向量
 * 3. 线性投影：投影到隐藏维度
 * 4. CfC ODE演化：通过多层ODE连续时间演化
 * 5. 全局池化：所有补丁输出汇聚为统一特征向量
 *
 * 不使用任何CNN卷积、池化或注意力机制。
 * 全部由微分方程驱动：τ dh/dt = -h + σ(gate) ⊙ tanh(activation)
 *
 * @param processor 处理器句柄
 * @param image_data 图像数据（行优先，通道最后，float归一化[0,1]）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数
 * @param features 输出特征缓冲区
 * @param max_features 缓冲区最大特征数
 * @return int 成功返回特征数，失败返回-1
 */
int cfc_vision_extract_features(CfcVisionProcessor* processor,
                                const float* image_data,
                                int width, int height, int channels,
                                float* features, size_t max_features);

/**
 * @brief 获取处理器统计信息
 *
 * @param processor 处理器句柄
 * @param total_operations 总操作数（输出）
 * @param memory_usage_mb 内存使用量（MB，输出）
 * @param average_time_ms 平均处理时间（毫秒，输出）
 */
void cfc_vision_get_statistics(CfcVisionProcessor* processor,
                                size_t* total_operations,
                                float* memory_usage_mb,
                                float* average_time_ms);

/**
 * @brief 保存处理器到文件
 *
 * @param processor 处理器句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int cfc_vision_save_processor(CfcVisionProcessor* processor, const char* filename);

/**
 * @brief 从文件加载处理器
 *
 * @param filename 文件名
 * @return CfcVisionProcessor* 处理器句柄，失败返回NULL
 */
CfcVisionProcessor* cfc_vision_load_processor(const char* filename);

/* ============================================================================
 * 目标检测API（基于CfC特征 + 检测头）
 * ============================================================================ */

/**
 * @brief 最大检测类别数
 */
#define CFC_VISION_MAX_CLASSES 1000

/**
 * @brief 单次推理最大检测框数
 */
#define CFC_VISION_MAX_DETECTIONS 100

/**
 * @brief 检测结果结构体
 */
typedef struct {
    int class_id;           /**< 类别ID */
    float confidence;       /**< 置信度 [0,1] */
    float x, y;            /**< 检测框中心坐标（归一化 [0,1]） */
    float width, height;    /**< 检测框宽高（归一化 [0,1]） */
} CfcDetectionResult;

/**
 * @brief 检测头配置
 */
typedef struct {
    int num_classes;        /**< 目标类别数，默认80（COCO标准） */
    float conf_threshold;   /**< 置信度阈值，默认0.5 */
    float iou_threshold;    /**< NMS的IoU阈值，默认0.5 */
    int max_detections;     /**< 单次推理最大检测框数，默认100 */
} CfcDetectionConfig;

/**
 * @brief 获取默认检测配置
 *
 * @return CfcDetectionConfig 默认配置：80类，conf=0.5，iou=0.5，max=100
 */
CfcDetectionConfig cfc_vision_get_default_detection_config(void);

/**
 * @brief 检测API：对图像执行目标检测
 *
 * 流程：
 * 1. 调用 cfc_vision_extract_features 提取CfC特征
 * 2. 通过检测头线性层预测边界框和类别分数
 * 3. 对每个补丁网格位置生成检测候选
 * 4. 应用非极大值抑制（NMS）去除冗余框
 * 5. 按置信度排序输出 top-K 检测结果
 *
 * @param processor 处理器句柄
 * @param image_data 图像数据（行优先，通道最后，float归一化[0,1]）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数
 * @param results 检测结果输出缓冲区
 * @param max_results 最大结果数
 * @param num_detected 实际检测到的目标数（输出）
 * @return int 成功返回0，失败返回-1
 */
int cfc_vision_detect(CfcVisionProcessor* processor,
                       const float* image_data,
                       int width, int height, int channels,
                       CfcDetectionResult* results, int max_results,
                       int* num_detected);

/**
 * @brief 设置检测阈值
 *
 * @param processor 处理器句柄
 * @param conf_threshold 置信度阈值 [0,1]
 * @param iou_threshold IoU阈值 [0,1]
 * @return int 成功返回0，失败返回-1
 */
int cfc_vision_set_detection_threshold(CfcVisionProcessor* processor,
                                        float conf_threshold,
                                        float iou_threshold);

/**
 * @brief 获取检测统计信息
 *
 * @param processor 处理器句柄
 * @param total_detections 总检测数（输出）
 * @param avg_confidence 平均置信度（输出）
 * @return int 成功返回0，失败返回-1
 */
int cfc_vision_get_detection_stats(CfcVisionProcessor* processor,
                                    int* total_detections,
                                    float* avg_confidence);

/**
 * @brief 标记视觉处理器已完成训练
 *
 * 训练完成后调用此函数，标记CfC权重为已训练状态。
 * 未训练的处理器使用时将输出警告。
 *
 * @param processor 处理器句柄
 */
void cfc_vision_mark_trained(CfcVisionProcessor* processor);

/**
 * @brief 检查视觉处理器是否已完成训练
 *
 * @param processor 处理器句柄
 * @return int 1=已训练，0=未训练
 */
int cfc_vision_is_trained(const CfcVisionProcessor* processor);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_DEEP_VISION_H */
