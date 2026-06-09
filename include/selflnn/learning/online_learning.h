/**
 * @file online_learning.h
 * @brief 在线学习系统接口
 * 
 * 实时学习、流式学习、增量学习等功能的接口。
 *  ，提供完整的在线学习算法。
 */

#ifndef SELFLNN_ONLINE_LEARNING_H
#define SELFLNN_ONLINE_LEARNING_H

#include <stddef.h>
#include <time.h>

/* 前向声明LNN类型，用于 online_learner_attach_lnn() */
typedef struct LNN LNN;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 在线学习算法类型枚举
 */
typedef enum {
    ONLINE_LEARNING_SGD = 0,          /**< 流式梯度下降 */
    ONLINE_LEARNING_KALMAN = 1,       /**< 卡尔曼滤波学习 */
    ONLINE_LEARNING_ADAPTIVE = 2,     /**< 自适应学习率 */
    ONLINE_LEARNING_FORGETTING = 3,   /**< 带遗忘机制的学习 */
    ONLINE_LEARNING_ENSEMBLE = 4      /**< 集成在线学习 */
} OnlineLearningType;

/**
 * @brief 概念漂移检测方法枚举
 */
typedef enum {
    CONCEPT_DRIFT_NONE = 0,           /**< 无概念漂移检测 */
    CONCEPT_DRIFT_ADWIN = 1,          /**< ADWIN算法 */
    CONCEPT_DRIFT_PAGE_HINKLEY = 2,   /**< Page-Hinkley测试 */
    CONCEPT_DRIFT_KSWIN = 3           /**< KSWIN算法 */
} ConceptDriftMethod;

/**
 * @brief 遗忘机制类型枚举
 */
typedef enum {
    FORGETTING_NONE = 0,              /**< 无遗忘机制 */
    FORGETTING_EXPONENTIAL = 1,       /**< 指数遗忘 */
    FORGETTING_LINEAR = 2,            /**< 线性遗忘 */
    FORGETTING_ADAPTIVE = 3           /**< 自适应遗忘 */
} ForgettingType;

/**
 * @brief 在线学习配置
 */
typedef struct {
    OnlineLearningType algorithm_type;    /**< 算法类型 */
    float learning_rate;                  /**< 基础学习率 */
    float forgetting_factor;              /**< 遗忘因子（0-1） */
    ForgettingType forgetting_type;       /**< 遗忘机制类型 */
    ConceptDriftMethod drift_method;      /**< 概念漂移检测方法 */
    float drift_detection_threshold;      /**< 漂移检测阈值 */
    int window_size;                      /**< 滑动窗口大小 */
    int enable_adaptive_rate;             /**< 是否启用自适应学习率 */
    float min_learning_rate;              /**< 最小学习率 */
    float max_learning_rate;              /**< 最大学习率 */
    int enable_momentum;                  /**< 是否启用动量 */
    float momentum_factor;                /**< 动量因子 */
    int enable_regularization;            /**< 是否启用正则化 */
    float regularization_strength;        /**< 正则化强度 */
    int buffer_size;                      /**< 样本缓冲区大小 */
    int enable_real_time_update;          /**< 是否启用实时更新 */
    int update_frequency_ms;              /**< 更新频率（毫秒） */
    int enable_ewc;                       /**< K-012: 是否启用EWC防止灾难性遗忘 */
    float ewc_strength;                   /**< K-012: EWC正则化强度（默认1.0） */
    float ewc_fisher_sample_count;        /**< K-012: Fisher信息矩阵采样数量 */
} OnlineLearningConfig;

/**
 * @brief 概念漂移检测结果
 */
typedef struct {
    int drift_detected;                   /**< 是否检测到漂移 */
    float confidence;                     /**< 检测置信度 */
    size_t drift_point;                   /**< 漂移点位置 */
    float magnitude;                      /**< 漂移幅度 */
    char description[256];                /**< 漂移描述 */
} ConceptDriftResult;

/**
 * @brief 在线学习状态
 */
typedef struct {
    size_t total_samples;                 /**< 总样本数 */
    size_t processed_samples;             /**< 已处理样本数 */
    float current_loss;                   /**< 当前损失 */
    float average_loss;                   /**< 平均损失 */
    float current_learning_rate;          /**< 当前学习率 */
    int concept_drift_count;              /**< 概念漂移检测次数 */
    time_t last_update_time;              /**< 最后更新时间 */
    int is_initialized;                   /**< 是否已初始化 */
} OnlineLearningStatus;

/**
 * @brief 在线学习器句柄
 */
typedef struct OnlineLearner OnlineLearner;

/**
 * @brief 创建在线学习器
 * 
 * @param config 学习配置
 * @param model_weights 模型权重数组（将被复制）
 * @param weights_size 权重数量
 * @return OnlineLearner* 在线学习器句柄，失败返回NULL
 */
OnlineLearner* online_learner_create(const OnlineLearningConfig* config,
                                     const float* model_weights,
                                     size_t weights_size);

/**
 * @brief 释放在线学习器
 * 
 * @param learner 在线学习器句柄
 */
void online_learner_free(OnlineLearner* learner);

/**
 * @brief 在线学习更新（单样本）
 * 
 * @param learner 在线学习器句柄
 * @param input 输入特征向量
 * @param input_size 输入特征维度
 * @param target 目标值/标签
 * @param target_size 目标维度
 * @param loss 输出损失值（可选）
 * @return int 成功返回0，失败返回-1
 */
int online_learner_update(OnlineLearner* learner,
                          const float* input, size_t input_size,
                          const float* target, size_t target_size,
                          float* loss);

/**
 * @brief 在线学习更新（批量样本）
 * 
 * @param learner 在线学习器句柄
 * @param inputs 输入特征矩阵（样本×特征）
 * @param num_samples 样本数量
 * @param input_size 输入特征维度
 * @param targets 目标矩阵（样本×目标）
 * @param target_size 目标维度
 * @param average_loss 输出平均损失（可选）
 * @return int 成功返回0，失败返回-1
 */
int online_learner_update_batch(OnlineLearner* learner,
                                const float* inputs, size_t num_samples, size_t input_size,
                                const float* targets, size_t target_size,
                                float* average_loss);

/**
 * @brief 检查概念漂移
 * 
 * @param learner 在线学习器句柄
 * @param result 漂移检测结果输出
 * @return int 成功返回0，失败返回-1
 */
int online_learner_check_concept_drift(OnlineLearner* learner,
                                       ConceptDriftResult* result);

/**
 * @brief 获取当前学习状态
 * 
 * @param learner 在线学习器句柄
 * @param status 状态输出
 * @return int 成功返回0，失败返回-1
 */
int online_learner_get_status(OnlineLearner* learner,
                              OnlineLearningStatus* status);

/**
 * @brief 重置学习器状态
 * 
 * @param learner 在线学习器句柄
 * @param keep_weights 是否保留当前权重
 * @return int 成功返回0，失败返回-1
 */
int online_learner_reset(OnlineLearner* learner, int keep_weights);

/**
 * @brief 获取当前模型权重数量
 * 
 * @param learner 在线学习器句柄
 * @return size_t 权重数量，失败返回0
 */
size_t online_learner_get_weights_size(OnlineLearner* learner);

/**
 * @brief 获取当前模型权重
 * 
 * @param learner 在线学习器句柄
 * @param weights 权重输出缓冲区
 * @param max_weights 最大权重数量
 * @return int 成功返回权重数量，失败返回-1
 */
int online_learner_get_weights(OnlineLearner* learner,
                               float* weights, size_t max_weights);

/**
 * @brief 设置模型权重
 * 
 * @param learner 在线学习器句柄
 * @param weights 新权重数组
 * @param weights_size 权重数量
 * @return int 成功返回0，失败返回-1
 */
int online_learner_set_weights(OnlineLearner* learner,
                               const float* weights, size_t weights_size);

/**
 * @brief 自适应调整学习率
 * 
 * @param learner 在线学习器句柄
 * @param recent_losses 最近损失数组
 * @param num_losses 损失数量
 * @return int 成功返回新的学习率（通过返回值），失败返回-1
 */
float online_learner_adjust_learning_rate(OnlineLearner* learner,
                                          const float* recent_losses,
                                          size_t num_losses);

/**
 * @brief 执行遗忘机制
 * 
 * @param learner 在线学习器句柄
 * @param forgetting_strength 遗忘强度（0-1）
 * @return int 成功返回0，失败返回-1
 */
int online_learner_apply_forgetting(OnlineLearner* learner,
                                    float forgetting_strength);

/**
 * @brief K-012: 计算EWC Fisher信息矩阵对角线
 *
 * 基于当前任务训练数据计算每个参数的Fisher信息量。
 * Fisher信息量高的参数对当前任务更重要，更新时需要更严格限制。
 *
 * Fisher信息矩阵估计: F_i = E[(∂log p(y|x,θ)/∂θ_i)²]
 * 简化实现: 使用梯度平方的指数移动平均近似
 *
 * @param learner 在线学习器句柄
 * @param input 输入样本
 * @param input_size 输入维度
 * @param target 目标值
 * @param target_size 目标维度
 * @return int 成功返回0，失败返回-1
 */
int online_learner_compute_ewc_fisher(OnlineLearner* learner,
                                       const float* input, size_t input_size,
                                       const float* target, size_t target_size);

/**
 * @brief K-012: 设置EWC锚定参数（保存当前最优权重作为前一任务参照）
 *
 * 调用此函数后，后续更新将受EWC正则化约束：
 * L_total(θ) = L_new(θ) + (λ/2) * Σ F_i * (θ_i - θ*_i)²
 *
 * @param learner 在线学习器句柄
 * @return int 成功返回0，失败返回-1
 */
int online_learner_ewc_set_anchor(OnlineLearner* learner);

/**
 * @brief K-012: 获取EWC状态信息
 *
 * @param learner 在线学习器句柄
 * @param ewc_enabled [输出] EWC是否启用
 * @param ewc_strength [输出] EWC强度
 * @param fisher_sum [输出] Fisher信息总和(norm)
 * @return int 成功返回0，失败返回-1
 */
int online_learner_ewc_get_status(OnlineLearner* learner,
                                   int* ewc_enabled,
                                   float* ewc_strength,
                                   float* fisher_sum);

/**
 * @brief 保存学习器状态到文件
 * 
 * @param learner 在线学习器句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int online_learner_save_state(OnlineLearner* learner, const char* filename);

/**
 * @brief 从文件加载学习器状态
 * 
 * @param learner 在线学习器句柄
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int online_learner_load_state(OnlineLearner* learner, const char* filename);

/**
 * @brief 将在线学习器附着到共享LNN实例
 *
 * 调用后学习器将直接读写LNN的权重矩阵，而非维护独立权重副本。
 * 学习器的梯度下降更新将直接作用于LNN参数，实现真正的在线学习。
 * 学习器的统计信息（loss、samples、概念漂移等）继续正常维护。
 *
 * @param learner 在线学习器句柄
 * @param lnn 共享LNN实例
 * @return int 成功返回0，失败返回-1
 */
int online_learner_attach_lnn(OnlineLearner* learner, LNN* lnn);

/**
 * @brief 设置在线学习器的模仿学习启用状态
 *
 * 由能力开关系统调用，控制模仿学习子系统的启停。
 *
 * @param learner 在线学习器句柄
 * @param enabled 1=启用模仿学习, 0=禁用模仿学习
 * @return int 成功返回0，失败返回-1
 */
int online_learner_set_imitation_enabled(OnlineLearner* learner, int enabled);

/**
 * @brief 设置在线学习器的探索率
 *
 * 由能力开关系统调用，控制好奇心/探索驱动力。
 * 探索率影响epsilon-greedy策略和噪声注入强度。
 *
 * @param learner 在线学习器句柄
 * @param rate 探索率（0.0=纯利用, 1.0=纯探索）
 * @return int 成功返回0，失败返回-1
 */
int online_learner_set_exploration(OnlineLearner* learner, float rate);

/**
 * @brief 获取在线学习器的模仿学习启用状态
 *
 * @param learner 在线学习器句柄
 * @return int 1=启用, 0=禁用, -1=错误
 */
int online_learner_get_imitation_enabled(OnlineLearner* learner);

/**
 * @brief 获取在线学习器的当前探索率
 *
 * @param learner 在线学习器句柄
 * @param rate [输出] 当前探索率
 * @return int 成功返回0，失败返回-1
 */
int online_learner_get_exploration(OnlineLearner* learner, float* rate);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_ONLINE_LEARNING_H */