/**
 * @file metacognition.h
 * @brief 元认知系统接口
 * 
 * 元认知系统实现，支持元认知监控、自我模型在线更新、预测性自我认知、
 * 客观理性自我评估等功能。 ，提供完整的元认知算法。
 */

#ifndef SELFLNN_METACOGNITION_H
#define SELFLNN_METACOGNITION_H

#include "selflnn/core/lnn.h"
#include <stddef.h>

/* 前向声明（实际结构体在专用头文件中定义，避免循环包含） */
struct DeepReflectionEngine;
struct DTCSystem;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 元认知监控类型枚举
 */
typedef enum {
    METACOGNITION_MONITORING_PERFORMANCE = 0,   /**< 性能监控 */
    METACOGNITION_MONITORING_CONFIDENCE = 1,    /**< 置信度监控 */
    METACOGNITION_MONITORING_UNCERTAINTY = 2,   /**< 不确定性监控 */
    METACOGNITION_MONITORING_LEARNING = 3,      /**< 学习进展监控 */
    METACOGNITION_MONITORING_PLANNING = 4,      /**< 规划进展监控 */
    METACOGNITION_MONITORING_RESOURCE = 5,      /**< 资源使用监控 */
    METACOGNITION_MONITORING_ERROR = 6,         /**< 错误监控 */
    METACOGNITION_MONITORING_ADAPTATION = 7     /**< 自适应监控 */
} MetacognitionMonitoringType;

/**
 * @brief 自我模型更新方法枚举
 */
typedef enum {
    SELF_MODEL_UPDATE_BAYESIAN = 0,            /**< 贝叶斯更新 */
    SELF_MODEL_UPDATE_KALMAN = 1,              /**< 卡尔曼滤波更新 */
    SELF_MODEL_UPDATE_ONLINE_LEARNING = 2,     /**< 在线学习更新 */
    SELF_MODEL_UPDATE_REINFORCEMENT = 3,       /**< 强化学习更新 */
    SELF_MODEL_UPDATE_ENSEMBLE = 4             /**< 集成学习更新 */
} SelfModelUpdateMethod;

/**
 * @brief 预测性自我认知类型枚举
 */
typedef enum {
    PREDICTIVE_SELF_PERFORMANCE = 0,           /**< 性能预测 */
    PREDICTIVE_SELF_FAILURE = 1,               /**< 失败预测 */
    PREDICTIVE_SELF_LEARNING = 2,              /**< 学习进展预测 */
    PREDICTIVE_SELF_RESOURCE = 3,              /**< 资源使用预测 */
    PREDICTIVE_SELF_ADAPTATION = 4             /**< 自适应需求预测 */
} PredictiveSelfType;

/**
 * @brief 元认知监控配置
 */
typedef struct {
    MetacognitionMonitoringType monitoring_type; /**< 监控类型 */
    float update_frequency;                     /**< 更新频率（Hz） */
    float confidence_threshold;                 /**< 置信度阈值 */
    float uncertainty_threshold;                /**< 不确定性阈值 */
    int enable_real_time;                       /**< 是否启用实时监控 */
    int enable_prediction;                      /**< 是否启用预测 */
    int enable_self_correction;                 /**< 是否启用自我修正 */
    size_t history_buffer_size;                 /**< 历史缓冲区大小 */
    float adaptation_rate;                      /**< 自适应率 */
} MetacognitionMonitoringConfig;

/**
 * @brief 自我模型更新配置
 */
typedef struct {
    SelfModelUpdateMethod update_method;       /**< 更新方法 */
    float learning_rate;                       /**< 学习率 */
    float forgetting_factor;                   /**< 遗忘因子 */
    float uncertainty_weight;                  /**< 不确定性权重 */
    int enable_online_update;                  /**< 是否启用在线更新 */
    int enable_batch_update;                   /**< 是否启用批量更新 */
    size_t model_complexity;                   /**< 模型复杂度 */
    float regularization_strength;             /**< 正则化强度 */
} SelfModelUpdateConfig;

/**
 * @brief 预测性自我认知配置
 */
typedef struct {
    PredictiveSelfType prediction_type;        /**< 预测类型 */
    float prediction_horizon;                  /**< 预测时间范围（秒） */
    float confidence_level;                    /**< 置信水平 */
    int enable_multiple_horizons;              /**< 是否启用多时间范围 */
    int enable_uncertainty_quantification;     /**< 是否启用不确定性量化 */
    size_t prediction_model_size;              /**< 预测模型大小 */
    float adaptation_sensitivity;              /**< 自适应敏感度 */
} PredictiveSelfConfig;

/**
 * @brief 元认知监控结果
 */
typedef struct {
    MetacognitionMonitoringType type;          /**< 监控类型 */
    float current_value;                       /**< 当前值 */
    float predicted_value;                     /**< 预测值 */
    float confidence;                          /**< 置信度 */
    float uncertainty;                         /**< 不确定性 */
    float trend;                               /**< 趋势（变化率） */
    int requires_action;                       /**< 是否需要行动 */
    char action_recommendation[256];           /**< 行动建议 */
    size_t timestamp;                          /**< 时间戳 */
} MetacognitionMonitoringResult;

/**
 * @brief 自我模型状态
 */
typedef struct {
    float* model_parameters;                   /**< 模型参数 */
    size_t parameters_size;                    /**< 参数大小 */
    float* uncertainty_estimates;              /**< 不确定性估计 */
    size_t uncertainty_size;                   /**< 不确定性大小 */
    float* performance_history;                /**< 性能历史 */
    size_t history_size;                       /**< 历史大小 */
    float* prediction_errors;                  /**< 预测误差 */
    size_t errors_size;                        /**< 误差大小 */
    float model_confidence;                    /**< 模型置信度 */
    float model_accuracy;                      /**< 模型准确度 */
    size_t update_count;                       /**< 更新计数 */
    size_t last_update_time;                   /**< 最后更新时间 */
} MetacognitionModelState;

/**
 * @brief 预测性自我认知结果
 */
typedef struct {
    PredictiveSelfType type;                   /**< 预测类型 */
    float* predicted_values;                   /**< 预测值数组 */
    size_t values_size;                        /**< 值数组大小 */
    float* confidence_intervals;               /**< 置信区间 */
    size_t intervals_size;                     /**< 区间大小 */
    float* uncertainty_estimates;              /**< 不确定性估计 */
    size_t uncertainty_size;                   /**< 不确定性大小 */
    float expected_value;                      /**< 期望值 */
    float worst_case_value;                    /**< 最坏情况值 */
    float best_case_value;                     /**< 最好情况值 */
    int requires_intervention;                 /**< 是否需要干预 */
    char intervention_suggestion[256];         /**< 干预建议 */
    size_t prediction_time;                    /**< 预测时间 */
} PredictiveSelfResult;

/**
 * @brief 元认知系统句柄
 */
typedef struct MetacognitionSystem MetacognitionSystem;

/**
 * @brief 创建元认知系统
 * 
 * @param monitoring_config 监控配置
 * @param model_update_config 模型更新配置
 * @param prediction_config 预测配置
 * @return MetacognitionSystem* 元认知系统句柄，失败返回NULL
 */
MetacognitionSystem* metacognition_system_create(
    const MetacognitionMonitoringConfig* monitoring_config,
    const SelfModelUpdateConfig* model_update_config,
    const PredictiveSelfConfig* prediction_config);

/**
 * @brief 释放元认知系统
 * 
 * @param system 元认知系统句柄
 */
void metacognition_system_free(MetacognitionSystem* system);

/**
 * @brief 释放预测性自我认知结果
 * 
 * 释放由 metacognition_predictive_self 返回的预测结果中的动态分配内存。
 * 
 * @param result 预测结果指针
 */
void metacognition_prediction_result_free(PredictiveSelfResult* result);

/**
 * @brief 执行元认知监控
 * 
 * @param system 元认知系统句柄
 * @param input_data 输入数据
 * @param data_size 数据大小
 * @param result 监控结果输出
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_monitor(MetacognitionSystem* system,
                         const float* input_data, size_t data_size,
                         MetacognitionMonitoringResult* result);

/**
 * @brief 更新自我模型
 * 
 * @param system 元认知系统句柄
 * @param new_data 新数据
 * @param data_size 数据大小
 * @param ground_truth 真实值（可选）
 * @param truth_size 真实值大小
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_update_self_model(MetacognitionSystem* system,
                                   const float* new_data, size_t data_size,
                                   const float* ground_truth, size_t truth_size);

/**
 * @brief 执行预测性自我认知
 * 
 * @param system 元认知系统句柄
 * @param current_state 当前状态
 * @param state_size 状态大小
 * @param prediction_horizon 预测时间范围（秒）
 * @param result 预测结果输出
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_predictive_self(MetacognitionSystem* system,
                                 const float* current_state, size_t state_size,
                                 float prediction_horizon,
                                 PredictiveSelfResult* result);

/**
 * @brief 获取自我模型状态
 * 
 * @param system 元认知系统句柄
 * @param state 自我模型状态输出
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_get_self_model_state(MetacognitionSystem* system,
                                      MetacognitionModelState* state);

/**
 * @brief 执行客观理性自我评估（不涉及情感，纯系统状态分析）
 * 
 * @param system 元认知系统句柄
 * @param assessment_type 评估类型
 * @param assessment_result 评估结果输出缓冲区
 * @param result_size 结果缓冲区大小
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_neutral_self_assessment(MetacognitionSystem* system,
                                         int assessment_type,
                                         char* assessment_result,
                                         size_t result_size);

/**
 * @brief 执行元认知驱动的自我修正
 * 
 * @param system 元认知系统句柄
 * @param error_type 错误类型
 * @param error_data 错误数据
 * @param data_size 数据大小
 * @param correction_plan 修正计划输出缓冲区
 * @param plan_size 计划缓冲区大小
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_self_correction(MetacognitionSystem* system,
                                 int error_type,
                                 const float* error_data, size_t data_size,
                                 char* correction_plan, size_t plan_size);

/**
 * @brief 保存元认知系统状态
 * 
 * @param system 元认知系统句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_save_state(MetacognitionSystem* system, const char* filename);

/**
 * @brief 加载元认知系统状态
 * 
 * @param system 元认知系统句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_load_state(MetacognitionSystem* system, const char* filename);

/**
 * @brief 获取元认知系统统计信息
 * 
 * @param system 元认知系统句柄
 * @param stats_buffer 统计信息输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_get_statistics(MetacognitionSystem* system,
                                char* stats_buffer, size_t buffer_size);

/**
 * @brief 设置元认知系统的液态神经网络实例
 * 
 * 将LNN实例连接到元认知系统，使元认知监控和预测功能能够利用
 * LNN的连续动态系统进行深度处理。
 * 
 * @param system 元认知系统句柄
 * @param lnn 液态神经网络实例指针（可传NULL解除关联）
 * @return int 成功返回0，失败返回-1
 */
int metacognition_system_set_lnn(MetacognitionSystem* system, LNN* lnn);

/**
 * @brief 设置元认知系统的深度反思引擎
 * 
 * 将DeepReflectionEngine实例连接到元认知系统，使自我评估和监控
 * 能够利用深度反思链进行多层次、多视角的认知分析。
 * 
 * @param system 元认知系统句柄
 * @param engine 深度反思引擎实例指针（可传NULL解除关联）
 * @return int 成功返回0，失败返回-1
 */
int metacognition_system_set_deep_reflection(MetacognitionSystem* system,
                                              struct DeepReflectionEngine* engine);

/**
 * @brief 执行深度反思驱动的自我评估
 * 
 * 使用深度反思引擎对系统状态进行多层次反思分析，生成包含
 * 多视角评估、知识冲突检测、根本原因分析和假设生成的
 * 全面自我评估结果。
 * 
 * @param system 元认知系统句柄
 * @param engine 深度反思引擎实例（若为NULL则使用系统内部引擎）
 * @param assessment_type 评估类型（0=全面, 1=性能, 2=认知, 3=知识）
 * @param assessment_result 评估结果输出缓冲区
 * @param result_size 结果缓冲区大小
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_deep_assessment(MetacognitionSystem* system,
                                   struct DeepReflectionEngine* engine,
                                   int assessment_type,
                                   char* assessment_result,
                                   size_t result_size);

/**
 * @brief 执行反思增强监控
 * 
 * 在标准元认知监控基础上，使用深度思考链（DTC）进行多步骤
 * 推理分析，生成包含推理链、替代假设和置信度评估的增强监控结果。
 * 
 * @param system 元认知系统句柄
 * @param input_data 输入数据
 * @param data_size 数据大小
 * @param result 监控结果输出
 * @param reflection_detail 反思详细信息输出缓冲区（可选）
 * @param detail_size 反思详细信息缓冲区大小
 * @return int 成功返回0，失败返回错误代码
 */
int metacognition_reflective_monitor(MetacognitionSystem* system,
                                      const float* input_data,
                                      size_t data_size,
                                      MetacognitionMonitoringResult* result,
                                      char* reflection_detail,
                                      size_t detail_size);

/* ============================================================================
 * P1-05: 元认知→自我认知闭环桥接
 *
 * 将元认知系统的深度反思/深度思维链分析结果
 * 推送回自我认知系统进行闭环校准。
 * 实现 Metacognition ↔ DeepRelection ↔ SelfCognition 的完整三方闭环。
 * ============================================================================ */

/**
 * @brief 将元认知的深度分析结果桥接到自我认知闭环
 *
 * 自动从元认知的 dr_engine (深度反思引擎) 和 dtc_system (深度思维链)
 * 提取洞见，推送到 self_cognition 进行能力维度校准。
 *
 * 调用时机：每次元认知监控或预测后调用，确保认知闭环。
 *
 * @param system 元认知系统
 * @param self_cog 自我认知系统（SelfCognitionSystem*）
 * @param bridge_result 桥接结果描述（可NULL）
 * @param result_size 描述缓冲区大小
 * @return 0成功，-1失败
 */
int metacognition_bridge_to_self_cognition(MetacognitionSystem* system,
                                           void* self_cog,
                                           char* bridge_result,
                                           size_t result_size);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_METACOGNITION_H */