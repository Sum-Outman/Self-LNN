#ifndef SELFLNN_UNIFIED_SIGNAL_PROCESSOR_ADVANCED_H
#define SELFLNN_UNIFIED_SIGNAL_PROCESSOR_ADVANCED_H

#include "selflnn/core/common.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SELFLNN_MAX_MODALITIES
#define SELFLNN_MAX_MODALITIES 4
#endif

/**
 * @file unified_signal_processor_advanced.h
 * @brief 模态自适应路由系统
 *
 * 基于信号质量动态计算各模态的路由权重。
 * 所有模态特征通过简单拼接后送入同一个CfC连续动态系统，
 * 路由权重仅用于信号质量感知和稳定性控制，不产生跨模态交互。
 * 严格遵循原则：不需要多模型融合、不需要跨模态注意力。
 */

/**
 * @brief 模态信号质量指标
 *
 * 描述单个模态的信号质量多维评估结果：
 * signal_energy - 信号能量（特征向量L2范数）
 * signal_variance - 信号方差（特征分布离散度）
 * signal_snr - 信噪比估计（最近N帧的信号稳定性）
 * temporal_consistency - 时序一致性（当前帧与前一帧的余弦相似度）
 * sparsity - 稀疏度（非零元素比例）
 * overall_quality - 综合质量评分（0.0低 ~ 1.0高）
 */
typedef struct {
    float signal_energy;           /**< 信号能量 */
    float signal_variance;         /**< 信号方差 */
    float signal_snr;              /**< 信噪比估计 */
    float temporal_consistency;    /**< 时序一致性 */
    float sparsity;                /**< 稀疏度 */
    float overall_quality;         /**< 综合质量评分（0.0-1.0） */
} ModalityQualityMetrics;

/**
 * @brief 模态自适应路由器配置
 *
 * softmax_temperature - Softmax温度参数（越低越硬路由，越高越均匀路由）
 * weight_momentum - 权重动量平滑系数（0=无平滑，0.9=强平滑）
 * min_weight / max_weight - 权重上下限约束
 * energy_weight / variance_weight / snr_weight / temporal_weight -
 *   各质量维度在综合评分中的权重系数
 * enable_adaptive_temperature - 是否启用自适应温度调节
 * temperature_decay - 温度衰减率（每次更新乘以该系数）
 * min_temperature - 最低温度下限
 * quality_window - 质量评估滑动窗口大小（帧数）
 */
typedef struct {
    float softmax_temperature;          /**< Softmax温度参数 */
    float weight_momentum;              /**< 权重动量平滑系数 */
    float min_weight;                   /**< 最小权重值 */
    float max_weight;                   /**< 最大权重值 */
    float energy_weight;                /**< 信号能量权重系数 */
    float variance_weight;              /**< 信号方差权重系数 */
    float snr_weight;                   /**< 信噪比权重系数 */
    float temporal_weight;              /**< 时序一致性权重系数 */
    int enable_adaptive_temperature;    /**< 启用自适应温度调节 */
    float temperature_decay;            /**< 温度衰减率 */
    float min_temperature;              /**< 最低温度下限 */
    int quality_window;                 /**< 质量评估滑动窗口大小 */
} AdaptiveRouterConfig;

/**
 * @brief 模态自适应路由器状态（不透明类型）
 *
 * 内部维护每个模态的平滑质量估计、历史权重和特征缓冲区。
 * 所有操作通过API函数完成，不暴露内部实现细节。
 */
typedef struct AdaptiveRouter AdaptiveRouter;

/**
 * @brief 获取默认自适应路由器配置
 * @return 默认配置结构体
 */
SELFLNN_API AdaptiveRouterConfig adaptive_router_get_default_config(void);

/**
 * @brief 创建模态自适应路由器实例
 * @param config 路由器配置（为NULL则使用默认配置）
 * @return 路由器句柄，失败返回NULL
 */
SELFLNN_API AdaptiveRouter* adaptive_router_create(const AdaptiveRouterConfig* config);

/**
 * @brief 销毁模态自适应路由器实例
 * @param router 路由器句柄
 */
SELFLNN_API void adaptive_router_free(AdaptiveRouter* router);

/**
 * @brief 计算指定模态的信号质量指标
 * @param router 路由器句柄
 * @param modality_idx 模态索引（0=视觉, 1=音频, 2=文本, 3=传感器）
 * @param features 当前帧特征向量
 * @param feature_dim 特征维度
 * @param prev_features 上一帧特征向量（用于时序一致性计算，可为NULL）
 * @param metrics [输出] 质量指标结果
 * @return 0=成功，非0=失败
 */
SELFLNN_API int adaptive_router_compute_quality(
    AdaptiveRouter* router, int modality_idx,
    const float* features, size_t feature_dim,
    const float* prev_features,
    ModalityQualityMetrics* metrics);

/**
 * @brief 基于各模态质量指标计算路由权重
 * @param router 路由器句柄
 * @param qualities 各模态的质量指标数组（长度为激活模态数）
 * @param modality_active 各模态是否激活的标志数组（长度为SELFLNN_MAX_MODALITIES）
 * @param weights [输出] 计算得到的路由权重数组
 * @return 0=成功，非0=失败
 */
SELFLNN_API int adaptive_router_compute_weights(
    AdaptiveRouter* router,
    const ModalityQualityMetrics* qualities,
    const int* modality_active,
    float* weights);

/**
 * @brief 对特征向量应用路由权重
 * @param router 路由器句柄
 * @param features 各模态的特征向量指针数组
 * @param feature_dims 各模态的特征维度数组
 * @param weights 各模态的路由权重数组
 * @param weighted_output [输出] 加权后的拼接输出向量
 * @param total_dim 输出总维度
 * @return 0=成功，非0=失败
 */
SELFLNN_API int adaptive_router_apply_weights(
    AdaptiveRouter* router,
    const float* features, const size_t* feature_dims,
    const float* weights,
    float* weighted_output, size_t total_dim);

/**
 * @brief 获取当前各模态的路由权重
 * @param router 路由器句柄
 * @param weights [输出] 权重数组（长度为SELFLNN_MAX_MODALITIES）
 * @return 0=成功，非0=失败
 */
SELFLNN_API int adaptive_router_get_weights(
    const AdaptiveRouter* router, float* weights);

/**
 * @brief 重置路由器状态（清除所有平滑缓冲区和历史记录）
 * @param router 路由器句柄
 */
SELFLNN_API void adaptive_router_reset(AdaptiveRouter* router);

#ifdef __cplusplus
}
#endif

#endif
