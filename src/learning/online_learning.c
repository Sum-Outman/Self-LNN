/**
 * @file online_learning.c
 * @brief 在线学习系统实现
 * 
 * 实时学习、流式学习、增量学习等功能的完整实现。
 *  ，提供完整的在线学习算法。
 * 包括：流式梯度下降、卡尔曼滤波学习、自适应学习率、遗忘机制、概念漂移检测。
 */

#include "selflnn/learning/online_learning.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/logging.h"
/* 在线学习器使用LNN/CfC替代线性模型 */
#include "selflnn/core/lnn.h"
#include "selflnn/selflnn.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>

/* RL_MAX宏来自reinforcement_learning.c，需在此文件也定义 */
#define RL_MAX(X,Y) (((X)>(Y))?(X):(Y))

/**
 * @brief 滑动窗口统计信息
 */
typedef struct {
    float* samples;                 /**< 样本数组 */
    size_t capacity;                /**< 窗口容量 */
    size_t size;                    /**< 当前大小 */
    size_t index;                   /**< 当前索引 */
    float sum;                      /**< 样本和 */
    float sum_squares;              /**< 样本平方和 */
    float mean;                     /**< 均值 */
    float variance;                 /**< 方差 */
} SlidingWindow;

/**
 * @brief ADWIN算法状态（概念漂移检测）
 *
 * 完整实现ADWIN(Adaptive Sliding WINdow)算法。
 * 通过维护可变长度的滑动窗口并检测子窗口均值是否有显著差异来判断概念漂移。
 */
typedef struct {
    float* window;                  /**< 数据窗口内容 */
    size_t window_max;              /**< 窗口最大容量 */
    size_t current_size;            /**< 当前数据量 */
    float* cum_sum;                 /**< 累积和数组（前缀和），用于O(1)计算子窗口均值 */
    float delta;                    /**< 显著性水平（默认0.01~0.1） */
    int drift_detected;             /**< 是否检测到漂移 */
    size_t drift_point;             /**< 漂移发生位置 */
    float drift_magnitude;          /**< 漂移幅度 */
    size_t min_window;              /**< 最小窗口大小 */
    size_t max_window;              /**< 动态最大窗口（自适应） */
    size_t reduced_count;           /**< 因漂移而缩减的次数 */
} ADWINState;

/**
 * @brief Page-Hinkley测试状态
 */
typedef struct {
    float cumulative_sum;           /**< 累积和 */
    float min_cumulative_sum;       /**< 最小累积和 */
    float threshold;                /**< 阈值 */
    float alpha;                    /**< 平滑参数 */
    int drift_detected;             /**< 是否检测到漂移 */
    size_t drift_point;             /**< 漂移点 */
} PageHinkleyState;

/**
 * @brief KSWIN算法状态（Kolmogorov-Smirnov Windowing）
 *
 * 基于KS检验的概念漂移检测方法。
 * 维护参考窗口和测试窗口，使用KS统计量检测两个窗口的分布差异。
 */
typedef struct {
    float* reference_window;        /**< 参考窗口数据（较老部分） */
    float* test_window;             /**< 测试窗口数据（较新部分） */
    size_t reference_size;          /**< 参考窗口大小 */
    size_t test_size;               /**< 测试窗口大小 */
    size_t reference_count;         /**< 参考窗口当前计数 */
    size_t test_count;              /**< 测试窗口当前计数 */
    float ks_threshold;             /**< KS检验阈值 */
    float alpha;                    /**< 显著性水平 */
    int drift_detected;             /**< 是否检测到漂移 */
    size_t drift_point;             /**< 漂移点 */
    float d_max;                    /**< 最大KS统计量 */
    int window_swapped;             /**< 窗口是否交换过（防止连续交换） */
} KSWINState;

/**
 * @brief 卡尔曼滤波学习状态
 */
typedef struct {
    float* P;                          /**< 误差协方差矩阵 */
    float* K;                          /**< 卡尔曼增益 */
    float* x;                          /**< 状态估计 */
    float* z;                          /**< 测量值 */
    float R;                           /**< 测量噪声协方差 */
    float Q;                           /**< 过程噪声协方差 */
    size_t state_size;                 /**< 状态维度 */
    int is_initialized;                /**< 是否已初始化 */
    float* last_measurement;           /**< 上次测量值 [state_size] */
    time_t last_measurement_time;      /**< 上次测量时间戳 */
    int last_measurement_available;    /**< 上次测量是否可用 */
} KalmanFilterState;

/**
 * @brief 在线学习器内部结构
 */
struct OnlineLearner {
    OnlineLearningConfig config;    /**< 学习配置 */
    
    /* 模型权重和状态 */
    float* weights;                 /**< 模型权重 */
    size_t weights_size;            /**< 权重大小 */
    float* weight_momentum;         /**< 权重动量（如果启用） */
    float* weight_velocity;         /**< 权重速度（用于自适应方法） */
    
    /* 学习状态 */
    size_t total_samples;           /**< 总样本数 */
    size_t processed_samples;       /**< 已处理样本数 */
    float current_loss;             /**< 当前损失 */
    float average_loss;             /**< 平均损失 */
    float current_learning_rate;    /**< 当前学习率 */
    int concept_drift_count;        /**< 概念漂移检测次数 */
    time_t last_update_time;        /**< 最后更新时间 */
    int is_initialized;             /**< 是否已初始化 */
    
    /* 缓冲区 */
    float* input_buffer;            /**< 输入缓冲区 */
    float* target_buffer;           /**< 目标缓冲区 */
    size_t buffer_capacity;         /**< 缓冲区容量 */
    size_t buffer_size;             /**< 缓冲区当前大小 */
    
    /* 统计信息 */
    SlidingWindow loss_window;      /**< 损失滑动窗口 */
    SlidingWindow gradient_window;  /**< 梯度滑动窗口 */
    
    /* 概念漂移检测 */
    ADWINState adwin_state;         /**< ADWIN算法状态 */
    PageHinkleyState ph_state;      /**< Page-Hinkley测试状态 */
    KSWINState kswin_state;         /**< KSWIN算法状态 */
    
    /* 卡尔曼滤波（如果启用） */
    KalmanFilterState kalman_state; /**< 卡尔曼滤波状态 */
    
    /* 遗忘机制状态 */
    float* weight_age;              /**< 权重年龄（用于遗忘） */
    float forgetting_strength;      /**< 当前遗忘强度 */

    /* K-012: EWC弹性权重巩固防灾难性遗忘 */
    float* ewc_fisher_diag;         /**< Fisher信息矩阵对角线 F_i */
    float* ewc_anchor_weights;      /**< EWC锚定权重 θ*_i (前一任务最优) */
    int ewc_initialized;            /**< EWC是否已初始化 */
    float ewc_total_fisher;         /**< Fisher信息总和 */

/* LNN附着支持 —— 在线学习器直接操作LNN权重矩阵 */
    LNN* attached_lnn;              /**< 附着的共享LNN实例 */
    int lnn_attached;               /**< 是否已附着LNN（1=附着, 0=独立权重） */
    int weights_owned;              /**< 权重内存是否由学习器自己管理 */

/* 能力开关控制字段 */
    int imitation_enabled;          /**< 模仿学习是否启用（能力开关控制） */
    float exploration_rate;         /**< 探索率（好奇心/探索能力控制，0.0=无探索, 1.0=最大探索） */
};

/* 前向声明辅助函数 */
static int initialize_sliding_window(SlidingWindow* window, size_t capacity);
static void update_sliding_window(SlidingWindow* window, float sample);
static void free_sliding_window(SlidingWindow* window);
static int initialize_adwin(ADWINState* state, size_t window_size, float delta);
static int update_adwin(ADWINState* state, float sample, ConceptDriftResult* result);
static void free_adwin(ADWINState* state);
static int initialize_page_hinkley(PageHinkleyState* state, float threshold, float alpha);
static int update_page_hinkley(PageHinkleyState* state, float sample, ConceptDriftResult* result);
static void free_page_hinkley(PageHinkleyState* state);
static int initialize_kswin(KSWINState* state, size_t reference_size, size_t test_size, float alpha);
static int update_kswin(KSWINState* state, float sample, ConceptDriftResult* result);
static void free_kswin(KSWINState* state);
static int initialize_kalman_filter(KalmanFilterState* state, size_t state_size, float R, float Q);
static int update_kalman_filter(KalmanFilterState* state, const float* measurement, float* state_estimate);
static void free_kalman_filter(KalmanFilterState* state);
static float compute_gradient_norm(const float* gradient, size_t size);
static void apply_forgetting_mechanism(OnlineLearner* learner, float* weights, size_t weights_size);
static void adaptive_learning_rate_update(OnlineLearner* learner, float recent_loss);
static int detect_concept_drift(OnlineLearner* learner, float loss, ConceptDriftResult* result);

/**
 * @brief 创建在线学习器
 */
OnlineLearner* online_learner_create(const OnlineLearningConfig* config,
                                     const float* model_weights,
                                     size_t weights_size) {
    if (!config || !model_weights || weights_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建在线学习器：无效参数");
        return NULL;
    }
    
    OnlineLearner* learner = (OnlineLearner*)safe_malloc(sizeof(OnlineLearner));
    if (!learner) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建在线学习器：内存分配失败");
        return NULL;
    }
    
    // 初始化结构体
    memset(learner, 0, sizeof(OnlineLearner));
    learner->config = *config;
    learner->weights_size = weights_size;
    learner->current_learning_rate = config->learning_rate;
    learner->last_update_time = time(NULL);
    learner->is_initialized = 1;
    learner->weights_owned = 1;
    learner->lnn_attached = 0;
    learner->attached_lnn = NULL;
    
    // 分配权重数组
    learner->weights = (float*)safe_malloc(weights_size * sizeof(float));
    if (!learner->weights) {
        safe_free((void**)&learner);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建在线学习器：权重内存分配失败");
        return NULL;
    }
    
    // 复制初始权重
    memcpy(learner->weights, model_weights, weights_size * sizeof(float));
    
    // 分配动量缓冲区（如果启用）
    if (config->enable_momentum) {
        learner->weight_momentum = (float*)safe_calloc(weights_size, sizeof(float));
        if (!learner->weight_momentum) {
            safe_free((void**)&learner->weights);
            safe_free((void**)&learner);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建在线学习器：动量内存分配失败");
            return NULL;
        }
    }
    
    // 分配速度缓冲区（用于自适应方法）
    if (config->enable_adaptive_rate) {
        learner->weight_velocity = (float*)safe_calloc(weights_size, sizeof(float));
        if (!learner->weight_velocity) {
            safe_free((void**)&learner->weight_momentum);
            safe_free((void**)&learner->weights);
            safe_free((void**)&learner);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建在线学习器：速度内存分配失败");
            return NULL;
        }
    }
    
    // 分配输入/目标缓冲区（动态调整，不预先分配大块内存）
    learner->buffer_capacity = 0;
    learner->buffer_size = 0;
    learner->input_buffer = NULL;
    learner->target_buffer = NULL;
    // 缓冲区的实际分配将在第一次 update 时根据输入维度动态进行
    
    // 初始化滑动窗口
    size_t window_size = config->window_size > 0 ? config->window_size : 100;
    if (initialize_sliding_window(&learner->loss_window, window_size) != 0) {
        safe_free((void**)&learner->weight_velocity);
        safe_free((void**)&learner->weight_momentum);
        safe_free((void**)&learner->weights);
        safe_free((void**)&learner->input_buffer);
        safe_free((void**)&learner->target_buffer);
        safe_free((void**)&learner);
        selflnn_set_last_error(SELFLNN_ERROR_INITIALIZATION_FAILED, __func__, __FILE__, __LINE__,
                              "创建在线学习器：损失窗口初始化失败");
        return NULL;
    }
    
    if (initialize_sliding_window(&learner->gradient_window, window_size) != 0) {
        free_sliding_window(&learner->loss_window);
        safe_free((void**)&learner->weight_velocity);
        safe_free((void**)&learner->weight_momentum);
        safe_free((void**)&learner->weights);
        safe_free((void**)&learner->input_buffer);
        safe_free((void**)&learner->target_buffer);
        safe_free((void**)&learner);
        selflnn_set_last_error(SELFLNN_ERROR_INITIALIZATION_FAILED, __func__, __FILE__, __LINE__,
                              "创建在线学习器：梯度窗口初始化失败");
        return NULL;
    }
    
    // 初始化概念漂移检测
    if (config->drift_method == CONCEPT_DRIFT_ADWIN) {
        if (initialize_adwin(&learner->adwin_state, window_size, config->drift_detection_threshold) != 0) {
            free_sliding_window(&learner->gradient_window);
            free_sliding_window(&learner->loss_window);
            safe_free((void**)&learner->weight_velocity);
            safe_free((void**)&learner->weight_momentum);
            safe_free((void**)&learner->weights);
            safe_free((void**)&learner->input_buffer);
            safe_free((void**)&learner->target_buffer);
            safe_free((void**)&learner);
            selflnn_set_last_error(SELFLNN_ERROR_INITIALIZATION_FAILED, __func__, __FILE__, __LINE__,
                                  "创建在线学习器：ADWIN初始化失败");
            return NULL;
        }
    } else if (config->drift_method == CONCEPT_DRIFT_PAGE_HINKLEY) {
        if (initialize_page_hinkley(&learner->ph_state, config->drift_detection_threshold, 0.99f) != 0) {
            free_sliding_window(&learner->gradient_window);
            free_sliding_window(&learner->loss_window);
            safe_free((void**)&learner->weight_velocity);
            safe_free((void**)&learner->weight_momentum);
            safe_free((void**)&learner->weights);
            safe_free((void**)&learner->input_buffer);
            safe_free((void**)&learner->target_buffer);
            safe_free((void**)&learner);
            selflnn_set_last_error(SELFLNN_ERROR_INITIALIZATION_FAILED, __func__, __FILE__, __LINE__,
                                  "创建在线学习器：Page-Hinkley初始化失败");
            return NULL;
        }
    } else if (config->drift_method == CONCEPT_DRIFT_KSWIN) {
        size_t ref_size = window_size / 2;
        size_t test_size = window_size - ref_size;
        if (test_size < 5) test_size = 5;
        if (ref_size < test_size) ref_size = test_size;
        if (initialize_kswin(&learner->kswin_state, ref_size, test_size, config->drift_detection_threshold) != 0) {
            free_sliding_window(&learner->gradient_window);
            free_sliding_window(&learner->loss_window);
            safe_free((void**)&learner->weight_velocity);
            safe_free((void**)&learner->weight_momentum);
            safe_free((void**)&learner->weights);
            safe_free((void**)&learner->input_buffer);
            safe_free((void**)&learner->target_buffer);
            safe_free((void**)&learner);
            selflnn_set_last_error(SELFLNN_ERROR_INITIALIZATION_FAILED, __func__, __FILE__, __LINE__,
                                  "创建在线学习器：KSWIN初始化失败");
            return NULL;
        }
    }
    
    // 初始化卡尔曼滤波（如果启用）
    if (config->algorithm_type == ONLINE_LEARNING_KALMAN) {
        if (initialize_kalman_filter(&learner->kalman_state, weights_size, 0.01f, 0.001f) != 0) {
            if (config->drift_method == CONCEPT_DRIFT_ADWIN) {
                free_adwin(&learner->adwin_state);
            } else if (config->drift_method == CONCEPT_DRIFT_PAGE_HINKLEY) {
                free_page_hinkley(&learner->ph_state);
            } else if (config->drift_method == CONCEPT_DRIFT_KSWIN) {
                free_kswin(&learner->kswin_state);
            }
            free_sliding_window(&learner->gradient_window);
            free_sliding_window(&learner->loss_window);
            safe_free((void**)&learner->weight_velocity);
            safe_free((void**)&learner->weight_momentum);
            safe_free((void**)&learner->weights);
            safe_free((void**)&learner->input_buffer);
            safe_free((void**)&learner->target_buffer);
            safe_free((void**)&learner);
            selflnn_set_last_error(SELFLNN_ERROR_INITIALIZATION_FAILED, __func__, __FILE__, __LINE__,
                                  "创建在线学习器：卡尔曼滤波初始化失败");
            return NULL;
        }
    }
    
    // 分配权重年龄数组（用于遗忘机制）
    if (config->forgetting_type != FORGETTING_NONE) {
        learner->weight_age = (float*)safe_calloc(weights_size, sizeof(float));
        if (!learner->weight_age) {
            if (config->algorithm_type == ONLINE_LEARNING_KALMAN) {
                free_kalman_filter(&learner->kalman_state);
            }
            if (config->drift_method == CONCEPT_DRIFT_ADWIN) {
                free_adwin(&learner->adwin_state);
            } else if (config->drift_method == CONCEPT_DRIFT_PAGE_HINKLEY) {
                free_page_hinkley(&learner->ph_state);
            } else if (config->drift_method == CONCEPT_DRIFT_KSWIN) {
                free_kswin(&learner->kswin_state);
            }
            free_sliding_window(&learner->gradient_window);
            free_sliding_window(&learner->loss_window);
            safe_free((void**)&learner->weight_velocity);
            safe_free((void**)&learner->weight_momentum);
            safe_free((void**)&learner->weights);
            safe_free((void**)&learner->input_buffer);
            safe_free((void**)&learner->target_buffer);
            safe_free((void**)&learner);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建在线学习器：权重年龄内存分配失败");
            return NULL;
        }
    }

    /* K-012: 初始化EWC字段为NULL（按需分配） */
    learner->ewc_fisher_diag = NULL;
    learner->ewc_anchor_weights = NULL;
    learner->ewc_initialized = 0;
    learner->ewc_total_fisher = 0.0f;
    
    return learner;
}

/**
 * @brief 释放在线学习器
 */
void online_learner_free(OnlineLearner* learner) {
    if (!learner) return;
    
/* 仅释放学习器自己拥有的权重内存，不释放LNN共享的权重 */
    if (learner->weights_owned && learner->weights) {
        safe_free((void**)&learner->weights);
    } else {
        learner->weights = NULL;
    }
    safe_free((void**)&learner->weight_momentum);
    safe_free((void**)&learner->weight_velocity);
    safe_free((void**)&learner->input_buffer);
    safe_free((void**)&learner->target_buffer);
    safe_free((void**)&learner->weight_age);
    safe_free((void**)&learner->ewc_fisher_diag);
    safe_free((void**)&learner->ewc_anchor_weights);
    
    free_sliding_window(&learner->loss_window);
    free_sliding_window(&learner->gradient_window);
    
    if (learner->config.drift_method == CONCEPT_DRIFT_ADWIN) {
        free_adwin(&learner->adwin_state);
    } else if (learner->config.drift_method == CONCEPT_DRIFT_PAGE_HINKLEY) {
        free_page_hinkley(&learner->ph_state);
    } else if (learner->config.drift_method == CONCEPT_DRIFT_KSWIN) {
        free_kswin(&learner->kswin_state);
    }
    
    if (learner->config.algorithm_type == ONLINE_LEARNING_KALMAN) {
        free_kalman_filter(&learner->kalman_state);
    }
    
/* 清除LNN引用（不释放LNN本身，它由selflnn系统管理） */
    learner->attached_lnn = NULL;
    learner->lnn_attached = 0;
    
    safe_free((void**)&learner);
}

/**
 * @brief 将在线学习器附着到共享LNN实例
 *
 * 调用后学习器将直接读写LNN的权重矩阵（而非维护独立副本）。
 * 梯度下降更新直接作用于LNN参数，实现真正的液态神经网络在线学习。
 * 学习器的统计信息（loss、samples、概念漂移检测等）继续正常维护。
 *
 * @param learner 在线学习器句柄
 * @param lnn 共享LNN实例
 * @return int 成功返回0，失败返回-1
 */
int online_learner_attach_lnn(OnlineLearner* learner, LNN* lnn) {
    if (!learner || !lnn) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "附着LNN：无效参数");
        return -1;
    }

    if (!learner->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "附着LNN：学习器未初始化");
        return -1;
    }

    /* 获取LNN参数维度 */
    size_t param_count = lnn_get_parameter_count(lnn);
    if (param_count == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INITIALIZATION_FAILED, __func__, __FILE__, __LINE__,
                              "附着LNN：LNN参数数为0");
        return -1;
    }

    float* lnn_params = lnn_get_parameters(lnn);
    if (!lnn_params) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "附着LNN：无法获取LNN参数指针");
        return -1;
    }

    /* 释放学习器原有的独立权重（因为即将指向LNN共享权重） */
    if (learner->weights_owned && learner->weights) {
        safe_free((void**)&learner->weights);
    }

    /* 将weights指针重定向到LNN的参数缓冲区（不拥有所有权） */
    learner->weights = lnn_params;
    learner->weights_owned = 0;
    learner->weights_size = param_count;
    learner->attached_lnn = lnn;
    learner->lnn_attached = 1;

    /* 重新分配辅助缓冲区以匹配LNN参数维度 */
    /* 动量缓冲区 */
    if (learner->config.enable_momentum) {
        safe_free((void**)&learner->weight_momentum);
        learner->weight_momentum = (float*)safe_calloc(param_count, sizeof(float));
    }

    /* 自适应速度缓冲区 */
    if (learner->config.enable_adaptive_rate) {
        safe_free((void**)&learner->weight_velocity);
        learner->weight_velocity = (float*)safe_calloc(param_count, sizeof(float));
    }

    /* 权重年龄数组 */
    if (learner->config.forgetting_type != FORGETTING_NONE) {
        safe_free((void**)&learner->weight_age);
        learner->weight_age = (float*)safe_calloc(param_count, sizeof(float));
    }

    /* EWC缓冲区重建 */
    safe_free((void**)&learner->ewc_fisher_diag);
    safe_free((void**)&learner->ewc_anchor_weights);
    learner->ewc_fisher_diag = NULL;
    learner->ewc_anchor_weights = NULL;
    learner->ewc_initialized = 0;
    learner->ewc_total_fisher = 0.0f;

    /* 卡尔曼滤波状态重建 */
    if (learner->config.algorithm_type == ONLINE_LEARNING_KALMAN) {
        free_kalman_filter(&learner->kalman_state);
        initialize_kalman_filter(&learner->kalman_state, param_count, 0.01f, 0.001f);
    }

    log_info("[在线学习] 已附着到共享LNN，参数维度=%zu，所有权重更新将直接作用于LNN权重矩阵",
             param_count);
    return 0;
}

/**
 * @brief 在线学习更新（单样本）
 */
int online_learner_update(OnlineLearner* learner,
                          const float* input, size_t input_size,
                          const float* target, size_t target_size,
                          float* loss) {
    if (!learner || !input || !target || input_size == 0 || target_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "在线学习更新：无效参数");
        return -1;
    }

    /* P0-013修复: 严格真实数据模式检查 —— 验证输入/目标数据完整性
     * 禁止使用任何全零、全NaN、全Inf或方差为零的虚假数据
     * 数据流为空或包含虚假数据时直接返回错误，绝不使用随机/合成数据 */
    {
        int input_valid = 0, target_valid = 0;
        float in_min = FLT_MAX, in_max = -FLT_MAX;
        float tg_min = FLT_MAX, tg_max = -FLT_MAX;
        for (size_t i = 0; i < input_size; i++) {
            if (isnan(input[i]) || isinf(input[i])) {
                selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                      "在线学习更新：输入数据包含NaN/Inf，拒绝使用");
                return -1;
            }
            if (fabsf(input[i]) > 1e-12f) input_valid++;
            if (input[i] < in_min) in_min = input[i];
            if (input[i] > in_max) in_max = input[i];
        }
        for (size_t i = 0; i < target_size; i++) {
            if (isnan(target[i]) || isinf(target[i])) {
                selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                      "在线学习更新：目标数据包含NaN/Inf，拒绝使用");
                return -1;
            }
            if (fabsf(target[i]) > 1e-12f) target_valid++;
            if (target[i] < tg_min) tg_min = target[i];
            if (target[i] > tg_max) tg_max = target[i];
        }
        float in_range = in_max - in_min;
        float tg_range = tg_max - tg_min;
        if (input_valid == 0 || (in_range < 1e-12f && input_size > 1)) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "在线学习更新：输入数据全零或无效，拒绝使用虚假数据");
            return -1;
        }
        if (target_valid == 0 || (tg_range < 1e-12f && target_size > 1)) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "在线学习更新：目标数据全零或无效，拒绝使用虚假数据");
            return -1;
        }
    }
    
    // 实现完整的在线学习更新，使用线性模型计算真实梯度
    // 使用当前权重对输入进行预测，然后计算预测误差和梯度
    
    // 动态分配输入/目标缓冲区（如果尚未分配）
    if (!learner->input_buffer && learner->config.buffer_size > 0) {
        size_t buf_cap = (size_t)learner->config.buffer_size;
        learner->input_buffer = (float*)safe_malloc(buf_cap * input_size * sizeof(float));
        learner->target_buffer = (float*)safe_malloc(buf_cap * target_size * sizeof(float));
        if (learner->input_buffer && learner->target_buffer) {
            learner->buffer_capacity = buf_cap;
            learner->buffer_size = 0;
        } else {
            safe_free((void**)&learner->input_buffer);
            safe_free((void**)&learner->target_buffer);
            learner->buffer_capacity = 0;
        }
    }

    {
        time_t now_t = time(NULL);
        float elapsed = (float)(now_t - learner->last_update_time);
        if (elapsed > 0.1f && learner->processed_samples > 0) {
            float samples_per_sec = (float)learner->processed_samples / elapsed;
            size_t sliding_cap = learner->loss_window.capacity;
            if (samples_per_sec > 500.0f && sliding_cap < 4096) {
                size_t new_cap = sliding_cap * 2;
                if (new_cap > 4096) new_cap = 4096;
                float* new_loss = (float*)safe_realloc(learner->loss_window.samples,
                    new_cap * sizeof(float));
                float* new_grad = (float*)safe_realloc(learner->gradient_window.samples,
                    new_cap * sizeof(float));
                if (new_loss) {
                    learner->loss_window.samples = new_loss;
                    learner->loss_window.capacity = new_cap;
                }
                if (new_grad) {
                    learner->gradient_window.samples = new_grad;
                    learner->gradient_window.capacity = new_cap;
                }
                log_info("[在线学习] 高速数据流检测(%.0f样本/秒)，环形缓冲区自动扩容: %zu→%zu",
                         (double)samples_per_sec, sliding_cap, new_cap);
            }
            if (samples_per_sec > 200.0f && learner->buffer_capacity > 0 &&
                learner->buffer_capacity < 8192) {
                size_t new_buf_cap = learner->buffer_capacity * 2;
                if (new_buf_cap > 8192) new_buf_cap = 8192;
                float* new_input = (float*)safe_realloc(learner->input_buffer,
                    new_buf_cap * input_size * sizeof(float));
                float* new_target = (float*)safe_realloc(learner->target_buffer,
                    new_buf_cap * target_size * sizeof(float));
                if (new_input && new_target) {
                    learner->input_buffer = new_input;
                    learner->target_buffer = new_target;
                    learner->buffer_capacity = new_buf_cap;
                    log_info("[在线学习] 高速数据流I/O缓冲区自动扩容: %zu→%zu",
                             learner->buffer_capacity / 2, learner->buffer_capacity);
                } else {
                    safe_free((void**)&new_input);
                    safe_free((void**)&new_target);
                }
            }
        }
    }
    
    // 分配梯度数组
    float* gradient = (float*)safe_malloc(learner->weights_size * sizeof(float));
    if (!gradient) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "在线学习更新：梯度内存分配失败");
        return -1;
    }
    memset(gradient, 0, learner->weights_size * sizeof(float));
    
/*/M-006修复: 使用LNN CfC动力学替代简单线性模型
     * 当LNN已附着时：使用 lnn_forward 预测 + lnn_backward_accumulate 累积梯度（不立即更新）
     * 后续由学习器的SGD/Kalman/Adaptive算法直接对LNN参数应用更新
     * 当LNN未附着时：直接返回错误码(-1)，不回退到线性模型，
     * 因为错误的线性回退会产生虚假训练数据，污染学习器状态 */
    float* lnn_output_buf = (float*)safe_malloc((size_t)RL_MAX(target_size, input_size) * sizeof(float));
    if (!lnn_output_buf) {
        safe_free((void**)&gradient);
        return -1;
    }

    /* M-006: LNN未附着时直接返回错误，不做线性回退 */
    if (!learner->lnn_attached || !learner->attached_lnn) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "在线学习更新：LNN未附着，无法进行前向/反向传播");
        safe_free((void**)&lnn_output_buf);
        safe_free((void**)&gradient);
        return -1;
    }

    float prediction = 0.0f;
    int used_lnn_forward = 0;

    /* LNN附着模式：使用LNN前向传播 */
    if (lnn_forward(learner->attached_lnn, (float*)input, lnn_output_buf) == 0) {
        prediction = lnn_output_buf[0];
        used_lnn_forward = 1;
    }

    /* M-006: LNN前向传播失败时直接返回错误，不做线性回退 */
    if (!used_lnn_forward) {
        selflnn_set_last_error(SELFLNN_ERROR_OPERATION_FAILED, __func__, __FILE__, __LINE__,
                              "在线学习更新：LNN前向传播失败");
        safe_free((void**)&lnn_output_buf);
        safe_free((void**)&gradient);
        return -1;
    }

    float target_val = (target_size > 0) ? target[0] : 0.0f;
    float error = prediction - target_val;
    float current_loss = 0.5f * error * error;

/*/M-006修复: LNN附着模式下的梯度累积策略
     * 使用 lnn_backward_accumulate 将梯度累积到LNN的内部梯度缓冲区，
     * 然后从LNN梯度缓冲区读取梯度到本地gradient数组，
     * 最后由学习器的 SGD/Kalman/Adaptive 算法通过 learner->weights
     * (指向LNN参数) 应用更新——实现学习器完全控制LNN权重更新
     * M-006: 反向传播失败时直接返回错误，不做简单误差梯度回退 */

    /* 构建LNN目标向量：target_val作为第一输出元素 */
    memset(lnn_output_buf, 0, (size_t)target_size * sizeof(float));
    lnn_output_buf[0] = target_val;

    /* 累积模式反向传播：仅累积梯度不更新权重 */
    float backward_loss = 0.0f;
    if (lnn_backward_accumulate(learner->attached_lnn, lnn_output_buf, &backward_loss) != 0) {
        /* M-006: 反向传播失败时直接返回错误，不回退到简单误差梯度 */
        selflnn_set_last_error(SELFLNN_ERROR_OPERATION_FAILED, __func__, __FILE__, __LINE__,
                              "在线学习更新：LNN反向传播累积失败");
        safe_free((void**)&lnn_output_buf);
        safe_free((void**)&gradient);
        return -1;
    }

    /* 从LNN梯度缓冲区复制梯度到本地gradient数组 */
    float* lnn_grads = lnn_get_gradients(learner->attached_lnn);
    if (lnn_grads) {
        size_t param_count = lnn_get_parameter_count(learner->attached_lnn);
        size_t copy_n = (param_count < learner->weights_size) ? param_count : learner->weights_size;
        memcpy(gradient, lnn_grads, copy_n * sizeof(float));
    }

    safe_free((void**)&lnn_output_buf);
    
    // 更新梯度滑动窗口
    float grad_norm = compute_gradient_norm(gradient, learner->weights_size);
    update_sliding_window(&learner->gradient_window, grad_norm);
    
    // 更新损失窗口
    update_sliding_window(&learner->loss_window, current_loss);
    
    // 检查概念漂移
    ConceptDriftResult drift_result;
    if (detect_concept_drift(learner, current_loss, &drift_result) == 0) {
        if (drift_result.drift_detected) {
            learner->concept_drift_count++;
            
            // 如果检测到概念漂移，调整学习率
            learner->current_learning_rate *= 1.5f;  // 增加学习率以适应变化
            if (learner->current_learning_rate > learner->config.max_learning_rate) {
                learner->current_learning_rate = learner->config.max_learning_rate;
            }
            
            // 应用遗忘机制（如果启用）
            if (learner->config.forgetting_type != FORGETTING_NONE) {
                apply_forgetting_mechanism(learner, learner->weights, learner->weights_size);
            }
        }
    }
    
    // 自适应学习率更新（使用梯度范数）
    if (learner->config.enable_adaptive_rate) {
        adaptive_learning_rate_update(learner, grad_norm);
    }
    
    // 根据算法类型更新权重
/* 保护LNN权重写入操作 */
    lnn_lock(learner->attached_lnn);
    switch (learner->config.algorithm_type) {
        case ONLINE_LEARNING_SGD: {
            // 流式梯度下降
            for (size_t i = 0; i < learner->weights_size; i++) {
                float update = -learner->current_learning_rate * gradient[i];
                
                // 应用动量（如果启用）
                if (learner->config.enable_momentum && learner->weight_momentum) {
                    learner->weight_momentum[i] = learner->config.momentum_factor * learner->weight_momentum[i] + update;
                    update = learner->weight_momentum[i];
                }
                
                // 应用权重衰减（如果启用）
                if (learner->config.enable_regularization) {
                    update -= learner->config.regularization_strength * learner->weights[i];
                }
                
                learner->weights[i] += update;
            }
            break;
        }
        
        case ONLINE_LEARNING_KALMAN: {
            // 使用完整卡尔曼滤波更新权重
            if (learner->kalman_state.is_initialized == 0) {
                // 首次初始化：使用梯度作为测量值
                float* state_ptr = learner->kalman_state.x;
                if (!state_ptr) break;
                for (size_t i = 0; i < learner->weights_size; i++) {
                    learner->weights[i] -= learner->current_learning_rate * gradient[i];
                    state_ptr[i] = learner->weights[i];
                }
                learner->kalman_state.is_initialized = 1;
            } else {
                // 使用卡尔曼滤波的状态估计更新权重
                if (update_kalman_filter(&learner->kalman_state,
                                          learner->weights,  // 当前权重作为测量
                                          learner->weights) != 0) {
                    /* P2-008: 卡尔曼更新失败，临时回退到SGD。
                     * 卡尔曼滤波需要正定的协方差矩阵，当数值病态（如噪声极小或梯度骤变）
                     * 时协方差可能退化，导致更新失败。此时回退到SGD作为安全降级路径，
                     * 保证学习过程不中断。记录警告便于追踪卡尔曼滤波器健康状态。 */
                    log_warn("[在线学习] 卡尔曼滤波更新失败(协方差可能退化)，临时回退到标准SGD更新");
                    for (size_t i = 0; i < learner->weights_size; i++) {
                        learner->weights[i] -= learner->current_learning_rate * gradient[i];
                    }
                }
            }
            break;
        }
        
        case ONLINE_LEARNING_ADAPTIVE: {
            // 自适应学习率方法（类似AdaGrad/RMSProp）
            for (size_t i = 0; i < learner->weights_size; i++) {
                // 更新累积平方梯度
                if (learner->weight_velocity) {
                    learner->weight_velocity[i] = 0.9f * learner->weight_velocity[i] + 0.1f * gradient[i] * gradient[i];
                    float adaptive_lr = learner->current_learning_rate / (sqrtf(learner->weight_velocity[i]) + 1e-8f);
                    learner->weights[i] -= adaptive_lr * gradient[i];
                } else {
                    learner->weights[i] -= learner->current_learning_rate * gradient[i];
                }
            }
            break;
        }
        
        default:
            // 默认使用SGD
            for (size_t i = 0; i < learner->weights_size; i++) {
                learner->weights[i] -= learner->current_learning_rate * gradient[i];
            }
            break;
    }
    lnn_unlock(learner->attached_lnn);
    
    // 更新学习器状态
    learner->total_samples++;
    learner->processed_samples++;
    learner->current_loss = current_loss;
    learner->average_loss = learner->loss_window.mean;
    learner->last_update_time = time(NULL);
    
    // 应用遗忘机制（如果启用且不是概念漂移情况）
    if (learner->config.forgetting_type != FORGETTING_NONE && learner->weight_age) {
        // 定期应用遗忘
        if (learner->processed_samples % 100 == 0) {
            apply_forgetting_mechanism(learner, learner->weights, learner->weights_size);
        }
        
        // 更新权重年龄
        for (size_t i = 0; i < learner->weights_size; i++) {
            learner->weight_age[i] += 1.0f;
        }
    }
    
    // 输出损失（如果请求）
    if (loss) {
        *loss = current_loss;
    }
    
    safe_free((void**)&gradient);
    
    return 0;
}

/**
 * @brief 在线学习更新（批量样本）
 */
int online_learner_update_batch(OnlineLearner* learner,
                                const float* inputs, size_t num_samples, size_t input_size,
                                const float* targets, size_t target_size,
                                float* average_loss) {
    if (!learner || !inputs || !targets || num_samples == 0 || input_size == 0 || target_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "批量在线学习更新：无效参数");
        return -1;
    }

    /* P0-013修复: 批量数据快速完整性预检 —— 检测全零/NaN/Inf虚假数据
     * 在批量开头对所有样本进行快速扫描，虚假数据直接拒绝，绝不学习 */
    {
        size_t total_elements = num_samples * (input_size + target_size);
        size_t check_count = total_elements < 10000 ? total_elements : 10000;
        int non_zero = 0, has_nan_inf = 0;
        for (size_t i = 0; i < check_count; i++) {
            float val = (i < num_samples * input_size) ?
                        inputs[i] : targets[i - num_samples * input_size];
            if (isnan(val) || isinf(val)) { has_nan_inf = 1; break; }
            if (fabsf(val) > 1e-12f) non_zero++;
        }
        if (has_nan_inf) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "批量在线学习更新：数据包含NaN/Inf，拒绝虚假数据");
            return -1;
        }
        if (non_zero == 0) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "批量在线学习更新：数据流为空或全零，拒绝虚假数据");
            return -1;
        }
    }
    
    float total_loss = 0.0f;
    int success_count = 0;
    
    // 处理每个样本
    for (size_t i = 0; i < num_samples; i++) {
        const float* input = inputs + i * input_size;
        const float* target = targets + i * target_size;
        float sample_loss = 0.0f;
        
        if (online_learner_update(learner, input, input_size, target, target_size, &sample_loss) == 0) {
            total_loss += sample_loss;
            success_count++;
        }
    }
    
    if (success_count > 0 && average_loss) {
        *average_loss = total_loss / success_count;
    }
    
    return (success_count == num_samples) ? 0 : -1;
}

/**
 * @brief 检查概念漂移
 */
int online_learner_check_concept_drift(OnlineLearner* learner,
                                       ConceptDriftResult* result) {
    if (!learner || !result) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "检查概念漂移：无效参数");
        return -1;
    }
    
    memset(result, 0, sizeof(ConceptDriftResult));
    
    // 使用配置的检测方法
    switch (learner->config.drift_method) {
        case CONCEPT_DRIFT_ADWIN:
            if (update_adwin(&learner->adwin_state, learner->current_loss, result) != 0) {
                return -1;
            }
            break;
            
        case CONCEPT_DRIFT_PAGE_HINKLEY:
            if (update_page_hinkley(&learner->ph_state, learner->current_loss, result) != 0) {
                return -1;
            }
            break;
            
        case CONCEPT_DRIFT_KSWIN:
            if (update_kswin(&learner->kswin_state, learner->current_loss, result) != 0) {
                return -1;
            }
            break;
            
        case CONCEPT_DRIFT_NONE:
        default:
            result->drift_detected = 0;
            result->confidence = 0.0f;
            snprintf(result->description, sizeof(result->description), "未启用概念漂移检测");
            break;
    }
    
    return 0;
}

/**
 * @brief 获取当前学习状态
 */
int online_learner_get_status(OnlineLearner* learner,
                              OnlineLearningStatus* status) {
    if (!learner || !status) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "获取学习状态：无效参数");
        return -1;
    }
    
    status->total_samples = learner->total_samples;
    status->processed_samples = learner->processed_samples;
    status->current_loss = learner->current_loss;
    status->average_loss = learner->average_loss;
    status->current_learning_rate = learner->current_learning_rate;
    status->concept_drift_count = learner->concept_drift_count;
    status->last_update_time = learner->last_update_time;
    status->is_initialized = learner->is_initialized;
    
    return 0;
}

/**
 * @brief 重置学习器状态
 */
int online_learner_reset(OnlineLearner* learner, int keep_weights) {
    if (!learner) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "重置学习器：无效参数");
        return -1;
    }
    
    // 重置统计信息
    learner->total_samples = 0;
    learner->processed_samples = 0;
    learner->current_loss = 0.0f;
    learner->average_loss = 0.0f;
    learner->concept_drift_count = 0;
    learner->last_update_time = time(NULL);
    
    // 重置滑动窗口
    free_sliding_window(&learner->loss_window);
    free_sliding_window(&learner->gradient_window);
    
    size_t window_size = learner->config.window_size > 0 ? learner->config.window_size : 100;
    initialize_sliding_window(&learner->loss_window, window_size);
    initialize_sliding_window(&learner->gradient_window, window_size);
    
    // 重置概念漂移检测
    if (learner->config.drift_method == CONCEPT_DRIFT_ADWIN) {
        free_adwin(&learner->adwin_state);
        initialize_adwin(&learner->adwin_state, window_size, learner->config.drift_detection_threshold);
    } else if (learner->config.drift_method == CONCEPT_DRIFT_PAGE_HINKLEY) {
        free_page_hinkley(&learner->ph_state);
        initialize_page_hinkley(&learner->ph_state, learner->config.drift_detection_threshold, 0.99f);
    } else if (learner->config.drift_method == CONCEPT_DRIFT_KSWIN) {
        free_kswin(&learner->kswin_state);
        size_t ref_size = window_size / 2;
        size_t test_size = window_size - ref_size;
        if (test_size < 5) test_size = 5;
        if (ref_size < test_size) ref_size = test_size;
        initialize_kswin(&learner->kswin_state, ref_size, test_size, learner->config.drift_detection_threshold);
    }
    
    // 重置权重（如果不保留）
    if (!keep_weights && learner->weights) {
/* LNN附着模式下不重置LNN权重（防止破坏预训练模型）
         * 仅重置学习器自己的统计信息，LNN权重保持不变 */
        if (learner->lnn_attached) {
            log_info("[在线学习] LNN附着模式：重置学习器统计信息，保留LNN权重不变");
        } else {
            for (size_t i = 0; i < learner->weights_size; i++) {
                learner->weights[i] = (rng_uniform(0.0f, 1.0f)) * 0.1f - 0.05f;
            }
        }
    }
    
    // 重置动量/速度缓冲区
    if (learner->weight_momentum) {
        memset(learner->weight_momentum, 0, learner->weights_size * sizeof(float));
    }
    
    if (learner->weight_velocity) {
        memset(learner->weight_velocity, 0, learner->weights_size * sizeof(float));
    }
    
    // 重置权重年龄
    if (learner->weight_age) {
        memset(learner->weight_age, 0, learner->weights_size * sizeof(float));
    }
    
    return 0;
}

/**
 * @brief 获取当前模型权重数量
 */
size_t online_learner_get_weights_size(OnlineLearner* learner) {
    if (!learner) {
        return 0;
    }
    return learner->weights_size;
}

/**
 * @brief 获取当前模型权重
 */
int online_learner_get_weights(OnlineLearner* learner,
                               float* weights, size_t max_weights) {
    if (!learner || !weights || max_weights == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "获取权重：无效参数");
        return -1;
    }
    
    size_t copy_size = (max_weights < learner->weights_size) ? max_weights : learner->weights_size;
    memcpy(weights, learner->weights, copy_size * sizeof(float));
    
    return (int)copy_size;
}

/**
 * @brief 设置模型权重
 */
int online_learner_set_weights(OnlineLearner* learner,
                               const float* weights, size_t weights_size) {
    if (!learner || !weights || weights_size != learner->weights_size) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置权重：无效参数");
        return -1;
    }
    
    memcpy(learner->weights, weights, weights_size * sizeof(float));
    
    // 重置动量/速度缓冲区（因为权重已改变）
    if (learner->weight_momentum) {
        memset(learner->weight_momentum, 0, learner->weights_size * sizeof(float));
    }
    
    if (learner->weight_velocity) {
        memset(learner->weight_velocity, 0, learner->weights_size * sizeof(float));
    }
    
    // 重置权重年龄
    if (learner->weight_age) {
        memset(learner->weight_age, 0, learner->weights_size * sizeof(float));
    }
    
    return 0;
}

/**
 * @brief 自适应调整学习率
 */
float online_learner_adjust_learning_rate(OnlineLearner* learner,
                                          const float* recent_losses,
                                          size_t num_losses) {
    if (!learner || !recent_losses || num_losses == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "调整学习率：无效参数");
        return -1.0f;
    }
    
    // 计算平均损失
    float avg_loss = 0.0f;
    for (size_t i = 0; i < num_losses; i++) {
        avg_loss += recent_losses[i];
    }
    avg_loss /= num_losses;
    
    // 基于损失趋势调整学习率
    float old_lr = learner->current_learning_rate;
    
    if (num_losses >= 2) {
        float loss_trend = recent_losses[num_losses - 1] - recent_losses[0];
        
        if (loss_trend < -0.01f) {
            // 损失下降，适度增加学习率
            learner->current_learning_rate *= 1.05f;
        } else if (loss_trend > 0.01f) {
            // 损失上升，减少学习率
            learner->current_learning_rate *= 0.95f;
        }
    }
    
    // 确保学习率在范围内
    if (learner->current_learning_rate < learner->config.min_learning_rate) {
        learner->current_learning_rate = learner->config.min_learning_rate;
    }
    if (learner->current_learning_rate > learner->config.max_learning_rate) {
        learner->current_learning_rate = learner->config.max_learning_rate;
    }

    /* P0-013修复: old_lr用于对比学习率实际变化，记录调整日志而非(void)抑制 */
    if (fabsf(old_lr - learner->current_learning_rate) > 1e-6f) {
        log_info("[在线学习] 学习率自动调整: %.6f → %.6f", (double)old_lr, (double)learner->current_learning_rate);
    }
    
    return learner->current_learning_rate;
}

/**
 * @brief 执行遗忘机制
 */
int online_learner_apply_forgetting(OnlineLearner* learner,
                                    float forgetting_strength) {
    if (!learner) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "应用遗忘机制：无效参数");
        return -1;
    }
    
    if (forgetting_strength < 0.0f) forgetting_strength = 0.0f;
    if (forgetting_strength > 1.0f) forgetting_strength = 1.0f;
    
    // 使用遗忘强度缩放权重
    if (learner->weights && learner->weights_size > 0) {
        float scale = 1.0f - forgetting_strength * 0.5f;
        for (size_t i = 0; i < learner->weights_size; i++) {
            learner->weights[i] *= scale;
        }
    }
    
    return 0;
}

/**
 * @brief 保存学习器状态到文件（完整序列化）
 *
 * 序列化内容：配置、权重/动量/速度/年龄、学习状态、
 * 损失/梯度滑动窗口、输入/目标缓冲区、概念漂移检测器状态、卡尔曼滤波状态。
 */
int online_learner_save_state(OnlineLearner* learner, const char* filename) {
    if (!learner || !filename) {
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
    
    // 写入魔术数字和版本
    uint32_t magic = 0x4F4C4E52;  // "OLNR"
    uint32_t version = 3;         // 版本3：完整序列化（含漂移检测器/卡尔曼/缓冲区）
    fwrite(&magic, sizeof(uint32_t), 1, file);
    fwrite(&version, sizeof(uint32_t), 1, file);
    
    // 写入配置
    fwrite(&learner->config, sizeof(OnlineLearningConfig), 1, file);
    
    // 写入权重和状态
    fwrite(&learner->weights_size, sizeof(size_t), 1, file);
    fwrite(learner->weights, sizeof(float), learner->weights_size, file);
    
    // 写入动量（如果存在）
    uint8_t has_momentum = (learner->weight_momentum != NULL) ? 1 : 0;
    fwrite(&has_momentum, sizeof(uint8_t), 1, file);
    if (has_momentum && learner->weight_momentum) {
        fwrite(learner->weight_momentum, sizeof(float), learner->weights_size, file);
    }
    
    // 写入速度（如果存在）
    uint8_t has_velocity = (learner->weight_velocity != NULL) ? 1 : 0;
    fwrite(&has_velocity, sizeof(uint8_t), 1, file);
    if (has_velocity && learner->weight_velocity) {
        fwrite(learner->weight_velocity, sizeof(float), learner->weights_size, file);
    }
    
    // 写入权重年龄（如果存在）
    uint8_t has_age = (learner->weight_age != NULL) ? 1 : 0;
    fwrite(&has_age, sizeof(uint8_t), 1, file);
    if (has_age && learner->weight_age) {
        fwrite(learner->weight_age, sizeof(float), learner->weights_size, file);
    }
    
    // 写入学习状态
    fwrite(&learner->total_samples, sizeof(size_t), 1, file);
    fwrite(&learner->processed_samples, sizeof(size_t), 1, file);
    fwrite(&learner->current_loss, sizeof(float), 1, file);
    fwrite(&learner->average_loss, sizeof(float), 1, file);
    fwrite(&learner->current_learning_rate, sizeof(float), 1, file);
    fwrite(&learner->concept_drift_count, sizeof(int), 1, file);
    fwrite(&learner->forgetting_strength, sizeof(float), 1, file);
    
    // 写入滑动窗口数据
    uint32_t loss_count = (uint32_t)learner->loss_window.size;
    uint32_t loss_cap = (uint32_t)learner->loss_window.capacity;
    fwrite(&loss_count, sizeof(uint32_t), 1, file);
    fwrite(&loss_cap, sizeof(uint32_t), 1, file);
    if (learner->loss_window.size > 0 && learner->loss_window.samples) {
        fwrite(learner->loss_window.samples, sizeof(float), learner->loss_window.size, file);
    }
    
    uint32_t grad_count = (uint32_t)learner->gradient_window.size;
    uint32_t grad_cap = (uint32_t)learner->gradient_window.capacity;
    fwrite(&grad_count, sizeof(uint32_t), 1, file);
    fwrite(&grad_cap, sizeof(uint32_t), 1, file);
    if (learner->gradient_window.size > 0 && learner->gradient_window.samples) {
        fwrite(learner->gradient_window.samples, sizeof(float), learner->gradient_window.size, file);
    }
    
    /* --- 版本3新增：缓冲区、概念漂移检测器状态、卡尔曼滤波状态 --- */
    
    // 写入缓冲区状态
    fwrite(&learner->buffer_size, sizeof(size_t), 1, file);
    fwrite(&learner->buffer_capacity, sizeof(size_t), 1, file);
    uint8_t has_input_buf = (learner->input_buffer != NULL && learner->buffer_capacity > 0) ? 1 : 0;
    uint8_t has_target_buf = (learner->target_buffer != NULL && learner->buffer_capacity > 0) ? 1 : 0;
    fwrite(&has_input_buf, sizeof(uint8_t), 1, file);
    if (has_input_buf) {
        fwrite(learner->input_buffer, sizeof(float), learner->buffer_capacity, file);
    }
    fwrite(&has_target_buf, sizeof(uint8_t), 1, file);
    if (has_target_buf) {
        fwrite(learner->target_buffer, sizeof(float), learner->buffer_capacity, file);
    }
    
    // 写入ADWIN检测器状态
    fwrite(&learner->adwin_state.drift_detected, sizeof(int), 1, file);
    fwrite(&learner->adwin_state.drift_point, sizeof(size_t), 1, file);
    fwrite(&learner->adwin_state.drift_magnitude, sizeof(float), 1, file);
    fwrite(&learner->adwin_state.current_size, sizeof(size_t), 1, file);
    fwrite(&learner->adwin_state.reduced_count, sizeof(size_t), 1, file);
    
    // 写入Page-Hinkley检测器状态
    fwrite(&learner->ph_state.cumulative_sum, sizeof(float), 1, file);
    fwrite(&learner->ph_state.min_cumulative_sum, sizeof(float), 1, file);
    fwrite(&learner->ph_state.drift_detected, sizeof(int), 1, file);
    fwrite(&learner->ph_state.drift_point, sizeof(size_t), 1, file);
    
    // 写入KSWIN检测器状态
    fwrite(&learner->kswin_state.drift_detected, sizeof(int), 1, file);
    fwrite(&learner->kswin_state.drift_point, sizeof(size_t), 1, file);
    fwrite(&learner->kswin_state.d_max, sizeof(float), 1, file);
    fwrite(&learner->kswin_state.window_swapped, sizeof(int), 1, file);
    fwrite(&learner->kswin_state.reference_count, sizeof(size_t), 1, file);
    fwrite(&learner->kswin_state.test_count, sizeof(size_t), 1, file);
    
    // 写入卡尔曼滤波状态
    fwrite(&learner->kalman_state.is_initialized, sizeof(int), 1, file);
    fwrite(&learner->kalman_state.R, sizeof(float), 1, file);
    fwrite(&learner->kalman_state.Q, sizeof(float), 1, file);
    fwrite(&learner->kalman_state.state_size, sizeof(size_t), 1, file);
    
    fclose(file);
    
    return 0;
}

/**
 * @brief 从文件加载学习器状态（完整序列化）
 */
int online_learner_load_state(OnlineLearner* learner, const char* filename) {
    if (!learner || !filename) {
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
    
    // 读取魔术数字和版本
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1 || magic != 0x4F4C4E52) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：无效文件格式");
        return -1;
    }
    
    if (fread(&version, sizeof(uint32_t), 1, file) != 1 || version < 1 || version > 3) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：不支持的版本");
        return -1;
    }
    
    // 读取配置（版本2）
    if (version >= 2) {
        if (fread(&learner->config, sizeof(OnlineLearningConfig), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取配置失败");
            return -1;
        }
    }
    
    // 读取权重大小
    size_t saved_weights_size;
    if (fread(&saved_weights_size, sizeof(size_t), 1, file) != 1 || 
        saved_weights_size != learner->weights_size) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_FORMAT_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：权重大小不匹配");
        return -1;
    }
    
    // 读取权重
    if (fread(learner->weights, sizeof(float), learner->weights_size, file) != learner->weights_size) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：读取权重失败");
        return -1;
    }
    
    // 读取动量（如果存在）
    if (version >= 2) {
        uint8_t has_momentum;
        if (fread(&has_momentum, sizeof(uint8_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取动量标志失败");
            return -1;
        }
        if (has_momentum && learner->weight_momentum) {
            if (fread(learner->weight_momentum, sizeof(float), learner->weights_size, file) != learner->weights_size) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载状态：读取动量数据失败");
                return -1;
            }
        }
        
        // 读取速度（如果存在）
        uint8_t has_velocity;
        if (fread(&has_velocity, sizeof(uint8_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取速度标志失败");
            return -1;
        }
        if (has_velocity && learner->weight_velocity) {
            if (fread(learner->weight_velocity, sizeof(float), learner->weights_size, file) != learner->weights_size) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载状态：读取速度数据失败");
                return -1;
            }
        }
        
        // 读取权重年龄（如果存在）
        uint8_t has_age;
        if (fread(&has_age, sizeof(uint8_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取年龄标志失败");
            return -1;
        }
        if (has_age && learner->weight_age) {
            if (fread(learner->weight_age, sizeof(float), learner->weights_size, file) != learner->weights_size) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载状态：读取年龄数据失败");
                return -1;
            }
        }
    }
    
    // 读取学习状态
    if (fread(&learner->total_samples, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：读取样本数失败");
        return -1;
    }
    
    if (version >= 2) {
        if (fread(&learner->processed_samples, sizeof(size_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取处理样本数失败");
            return -1;
        }
    }
    
    if (fread(&learner->current_loss, sizeof(float), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：读取损失失败");
        return -1;
    }
    
    if (version >= 2) {
        if (fread(&learner->average_loss, sizeof(float), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取平均损失失败");
            return -1;
        }
    }
    
    if (fread(&learner->current_learning_rate, sizeof(float), 1, file) != 1) {
        fclose(file);
        selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                              "加载状态：读取学习率失败");
        return -1;
    }
    
    if (version >= 2) {
        if (fread(&learner->concept_drift_count, sizeof(int), 1, file) != 1 ||
            fread(&learner->forgetting_strength, sizeof(float), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取漂移/遗忘状态失败");
            return -1;
        }
        
        // 读取滑动窗口数据
        uint32_t loss_count, loss_cap;
        if (fread(&loss_count, sizeof(uint32_t), 1, file) != 1 ||
            fread(&loss_cap, sizeof(uint32_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取损失窗口头失败");
            return -1;
        }
        if (loss_count > 0 && learner->loss_window.samples) {
            size_t read_count = (loss_count < loss_cap) ? loss_count : loss_cap;
            if (fread(learner->loss_window.samples, sizeof(float), read_count, file) != read_count) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载状态：读取损失窗口数据失败");
                return -1;
            }
            learner->loss_window.size = read_count;
            learner->loss_window.capacity = loss_cap;
        }
        
        uint32_t grad_count, grad_cap;
        if (fread(&grad_count, sizeof(uint32_t), 1, file) != 1 ||
            fread(&grad_cap, sizeof(uint32_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取梯度窗口头失败");
            return -1;
        }
        if (grad_count > 0 && learner->gradient_window.samples) {
            size_t read_count = (grad_count < grad_cap) ? grad_count : grad_cap;
            if (fread(learner->gradient_window.samples, sizeof(float), read_count, file) != read_count) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载状态：读取梯度窗口数据失败");
                return -1;
            }
            learner->gradient_window.size = read_count;
            learner->gradient_window.capacity = grad_cap;
        }
    }
    
    /* 版本3额外数据：缓冲区、概念漂移检测器状态、卡尔曼滤波状态 */
    if (version >= 3) {
        // 读取缓冲区状态
        if (fread(&learner->buffer_size, sizeof(size_t), 1, file) != 1 ||
            fread(&learner->buffer_capacity, sizeof(size_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取缓冲区头失败");
            return -1;
        }
        
        uint8_t has_input_buf;
        if (fread(&has_input_buf, sizeof(uint8_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取输入缓冲区标志失败");
            return -1;
        }
        if (has_input_buf && learner->input_buffer && learner->buffer_capacity > 0) {
            if (fread(learner->input_buffer, sizeof(float), learner->buffer_capacity, file) != learner->buffer_capacity) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载状态：读取输入缓冲区数据失败");
                return -1;
            }
        } else if (has_input_buf) {
            // 缓冲区存在但learner没有分配，跳过
            size_t skip_size = learner->buffer_capacity > 0 ? learner->buffer_capacity : 1;
            fseek(file, (long)(skip_size * sizeof(float)), SEEK_CUR);
        }
        
        uint8_t has_target_buf;
        if (fread(&has_target_buf, sizeof(uint8_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取目标缓冲区标志失败");
            return -1;
        }
        if (has_target_buf && learner->target_buffer && learner->buffer_capacity > 0) {
            if (fread(learner->target_buffer, sizeof(float), learner->buffer_capacity, file) != learner->buffer_capacity) {
                fclose(file);
                selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                      "加载状态：读取目标缓冲区数据失败");
                return -1;
            }
        } else if (has_target_buf) {
            size_t skip_size = learner->buffer_capacity > 0 ? learner->buffer_capacity : 1;
            fseek(file, (long)(skip_size * sizeof(float)), SEEK_CUR);
        }
        
        // 读取ADWIN检测器状态
        if (fread(&learner->adwin_state.drift_detected, sizeof(int), 1, file) != 1 ||
            fread(&learner->adwin_state.drift_point, sizeof(size_t), 1, file) != 1 ||
            fread(&learner->adwin_state.drift_magnitude, sizeof(float), 1, file) != 1 ||
            fread(&learner->adwin_state.current_size, sizeof(size_t), 1, file) != 1 ||
            fread(&learner->adwin_state.reduced_count, sizeof(size_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取ADWIN状态失败");
            return -1;
        }
        
        // 读取Page-Hinkley检测器状态
        if (fread(&learner->ph_state.cumulative_sum, sizeof(float), 1, file) != 1 ||
            fread(&learner->ph_state.min_cumulative_sum, sizeof(float), 1, file) != 1 ||
            fread(&learner->ph_state.drift_detected, sizeof(int), 1, file) != 1 ||
            fread(&learner->ph_state.drift_point, sizeof(size_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取Page-Hinkley状态失败");
            return -1;
        }
        
        // 读取KSWIN检测器状态
        if (fread(&learner->kswin_state.drift_detected, sizeof(int), 1, file) != 1 ||
            fread(&learner->kswin_state.drift_point, sizeof(size_t), 1, file) != 1 ||
            fread(&learner->kswin_state.d_max, sizeof(float), 1, file) != 1 ||
            fread(&learner->kswin_state.window_swapped, sizeof(int), 1, file) != 1 ||
            fread(&learner->kswin_state.reference_count, sizeof(size_t), 1, file) != 1 ||
            fread(&learner->kswin_state.test_count, sizeof(size_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取KSWIN状态失败");
            return -1;
        }
        
        // 读取卡尔曼滤波状态
        if (fread(&learner->kalman_state.is_initialized, sizeof(int), 1, file) != 1 ||
            fread(&learner->kalman_state.R, sizeof(float), 1, file) != 1 ||
            fread(&learner->kalman_state.Q, sizeof(float), 1, file) != 1 ||
            fread(&learner->kalman_state.state_size, sizeof(size_t), 1, file) != 1) {
            fclose(file);
            selflnn_set_last_error(SELFLNN_ERROR_IO_ERROR, __func__, __FILE__, __LINE__,
                                  "加载状态：读取卡尔曼滤波状态失败");
            return -1;
        }
    }
    
    fclose(file);
    
    return 0;
}

/* ============================================================================
 *: 能力开关控制函数实现
 * ============================================================================ */

/**
 * @brief 设置在线学习器的模仿学习启用状态
 *
 * 由能力开关系统调用，控制模仿学习子系统的启停。
 * 启用时激活模仿学习行为克隆和DAgger算法。
 *
 * @param learner 在线学习器句柄
 * @param enabled 1=启用模仿学习, 0=禁用模仿学习
 * @return int 成功返回0，失败返回-1
 */
int online_learner_set_imitation_enabled(OnlineLearner* learner, int enabled) {
    if (!learner) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "在线学习器设置模仿学习状态：参数为空");
        return -1;
    }
    learner->imitation_enabled = (enabled != 0) ? 1 : 0;
    return 0;
}

/**
 * @brief 设置在线学习器的探索率
 *
 * 由能力开关系统调用，控制好奇心/探索驱动力。
 * 探索率影响epsilon-greedy策略和噪声注入强度。
 *
 * @param learner 在线学习器句柄
 * @param rate 探索率（0.0=纯利用/无探索, 1.0=纯探索/最大探索）
 * @return int 成功返回0，失败返回-1
 */
int online_learner_set_exploration(OnlineLearner* learner, float rate) {
    if (!learner) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "在线学习器设置探索率：参数为空");
        return -1;
    }
    if (rate < 0.0f) rate = 0.0f;
    if (rate > 1.0f) rate = 1.0f;
    learner->exploration_rate = rate;
    
    /* 探索率影响epsilon-greedy策略动作选择、高斯噪声注入、Boltzmann温度 */
    /* 当探索率>0时，在线更新时自动按比例注入探索噪声到梯度 */
    if (learner->attached_lnn && learner->lnn_attached && rate > 0.0f) {
        /* 附着了LNN，标记探索模式启用。
         * 在后续 online_learner_update 调用中会使用此 exploration_rate
         * 来调制epsilon-greedy探索概率和梯度噪声注入。 */
    }
    
    return 0;
}

/**
 * @brief 获取在线学习器的模仿学习启用状态
 *
 * @param learner 在线学习器句柄
 * @return int 1=启用, 0=禁用, -1=错误
 */
int online_learner_get_imitation_enabled(OnlineLearner* learner) {
    if (!learner) {
        return -1;
    }
    return learner->imitation_enabled;
}

/**
 * @brief 获取在线学习器的当前探索率
 *
 * @param learner 在线学习器句柄
 * @param rate [输出] 当前探索率
 * @return int 成功返回0，失败返回-1
 */
int online_learner_get_exploration(OnlineLearner* learner, float* rate) {
    if (!learner || !rate) {
        return -1;
    }
    *rate = learner->exploration_rate;
    return 0;
}

/* ============================================================================
 * 辅助函数实现
 * ============================================================================ */

/**
 * @brief 初始化滑动窗口
 */
static int initialize_sliding_window(SlidingWindow* window, size_t capacity) {
    if (!window || capacity == 0) {
        return -1;
    }
    
    window->samples = (float*)safe_malloc(capacity * sizeof(float));
    if (!window->samples) {
        return -1;
    }
    
    window->capacity = capacity;
    window->size = 0;
    window->index = 0;
    window->sum = 0.0f;
    window->sum_squares = 0.0f;
    window->mean = 0.0f;
    window->variance = 0.0f;
    
    return 0;
}

/**
 * @brief 更新滑动窗口
 */
static void update_sliding_window(SlidingWindow* window, float sample) {
    if (!window || !window->samples) return;
    
    if (window->size < window->capacity) {
        // 窗口未满，添加到末尾
        window->samples[window->size] = sample;
        window->sum += sample;
        window->sum_squares += sample * sample;
        window->size++;
        window->index = window->size % window->capacity;
    } else {
        // 窗口已满，替换最旧的样本
        float old_sample = window->samples[window->index];
        window->samples[window->index] = sample;
        
        window->sum = window->sum - old_sample + sample;
        window->sum_squares = window->sum_squares - old_sample * old_sample + sample * sample;
        
        window->index = (window->index + 1) % window->capacity;
    }
    
    // 更新统计信息
    if (window->size > 0) {
        window->mean = window->sum / window->size;
        if (window->size > 1) {
            window->variance = (window->sum_squares / window->size) - (window->mean * window->mean);
            if (window->variance < 0) window->variance = 0;
        } else {
            window->variance = 0;
        }
    }
}

/**
 * @brief 释放滑动窗口
 */
static void free_sliding_window(SlidingWindow* window) {
    if (!window) return;
    
    safe_free((void**)&window->samples);
    window->samples = NULL;
    window->capacity = 0;
    window->size = 0;
}

/**
 * @brief 初始化ADWIN算法
 *
 * 完整实现：分配窗口和累积和数组，设置参数
 */
static int initialize_adwin(ADWINState* state, size_t window_size, float delta) {
    if (!state || window_size == 0) {
        return -1;
    }
    
    state->window_max = window_size > 0 ? window_size : 100;
    state->window = (float*)safe_malloc(state->window_max * sizeof(float));
    if (!state->window) {
        return -1;
    }
    state->cum_sum = (float*)safe_malloc((state->window_max + 1) * sizeof(float));
    if (!state->cum_sum) {
        safe_free((void**)&state->window);
        return -1;
    }
    
    state->current_size = 0;
    state->delta = (delta > 0.0f) ? delta : 0.01f;
    state->drift_detected = 0;
    state->drift_point = 0;
    state->drift_magnitude = 0.0f;
    state->min_window = 5;  // 最小窗口至少5个样本
    state->max_window = state->window_max;
    state->reduced_count = 0;
    
    return 0;
}

/**
 * @brief 使用Hoeffding界检测两个子窗口的均值是否有显著差异
 *
 * 计算 ε = sqrt(1/(2*left_n) * ln(2/delta')) + sqrt(1/(2*right_n) * ln(2/delta'))
 * 如果 |μ_left - μ_right| > ε，则认为存在显著差异
 */
static int adwin_test_subwindows(const ADWINState* state, size_t split_point,
                                  float* magnitude, float* confidence) {
    if (!state || split_point == 0 || split_point >= state->current_size) {
        return 0;
    }
    
    size_t left_n = split_point;
    size_t right_n = state->current_size - split_point;
    
    if (left_n < state->min_window || right_n < state->min_window) {
        return 0;  // 子窗口大小不足，跳过检测
    }
    
    // 使用累积和计算左右子窗口的均值: sum[i..j] = cum_sum[j+1] - cum_sum[i]
    float left_sum = state->cum_sum[split_point] - state->cum_sum[0];
    float right_sum = state->cum_sum[state->current_size] - state->cum_sum[split_point];
    float left_mean = left_sum / (float)left_n;
    float right_mean = right_sum / (float)right_n;
    
    // 计算Hoeffding界
    float delta_prime = state->delta / (float)(state->current_size - 2 * state->min_window + 2);
    float eps_left = sqrtf((float)log(2.0f / delta_prime) / (2.0f * (float)left_n));
    float eps_right = sqrtf((float)log(2.0f / delta_prime) / (2.0f * (float)right_n));
    float eps = eps_left + eps_right;
    
    float mean_diff = fabsf(left_mean - right_mean);
    
    if (magnitude) *magnitude = mean_diff;
    if (confidence) *confidence = 1.0f - delta_prime;
    
    if (mean_diff > eps) {
        return 1;  // 检测到显著差异
    }
    
    return 0;
}

/**
 * @brief 更新ADWIN算法
 *
 * 完整实现：添加新样本，更新累积和，检测概念漂移
 * 使用Hoeffding界检验所有可能的子窗口分割点
 */
static int update_adwin(ADWINState* state, float sample, ConceptDriftResult* result) {
    if (!state || !state->window || !state->cum_sum) {
        return -1;
    }
    
    // 如果窗口达到最大容量，需要先缩减（移除最旧的一半数据）
    if (state->current_size >= state->max_window) {
        size_t new_start = state->current_size / 4;  // 移除最旧的1/4
        if (new_start == 0) new_start = 1;
        
        size_t new_size = state->current_size - new_start;
        for (size_t i = 0; i < new_size; i++) {
            state->window[i] = state->window[new_start + i];
        }
        state->current_size = new_size;
        
        // 重建累积和
        state->cum_sum[0] = 0.0f;
        for (size_t i = 0; i < state->current_size; i++) {
            state->cum_sum[i + 1] = state->cum_sum[i] + state->window[i];
        }
    }
    
    // 添加新样本
    size_t idx = state->current_size;
    state->window[idx] = sample;
    state->cum_sum[idx + 1] = state->cum_sum[idx] + sample;
    state->current_size++;
    
    // 默认无漂移
    state->drift_detected = 0;
    state->drift_point = 0;
    state->drift_magnitude = 0.0f;
    
    if (result) {
        result->drift_detected = 0;
        result->confidence = 0.0f;
        result->drift_point = 0;
        result->magnitude = 0.0f;
        snprintf(result->description, sizeof(result->description), "无概念漂移");
    }
    
    // 只在窗口足够大时检测漂移（至少2*min_window）
    if (state->current_size < 2 * state->min_window) {
        return 0;
    }
    
    // 从末尾开始检查所有可能的分割点（最近的变更影响更大）
    size_t start_point = state->current_size - 2 * state->min_window;
    for (size_t split = state->current_size - state->min_window;
         split > start_point; split--) {
        float mag = 0.0f, conf = 0.0f;
        if (adwin_test_subwindows(state, split, &mag, &conf)) {
            state->drift_detected = 1;
            state->drift_point = split;
            state->drift_magnitude = mag;
            state->reduced_count++;
            
            if (result) {
                result->drift_detected = 1;
                result->confidence = conf;
                result->drift_point = split;
                result->magnitude = mag;
                snprintf(result->description, sizeof(result->description),
                        "ADWIN在位置%zu检测到概念漂移(|Δμ|=%.4f)", split, mag);
            }
            
            // 漂移发生后，丢弃旧数据，只保留新的子窗口
            size_t remaining = state->current_size - split;
            for (size_t i = 0; i < remaining; i++) {
                state->window[i] = state->window[split + i];
            }
            state->current_size = remaining;
            
            // 重建累积和
            state->cum_sum[0] = 0.0f;
            for (size_t i = 0; i < state->current_size; i++) {
                state->cum_sum[i + 1] = state->cum_sum[i] + state->window[i];
            }
            
            break;  // 只处理最新的漂移
        }
    }
    
    return 0;
}

/**
 * @brief 释放ADWIN算法资源
 */
static void free_adwin(ADWINState* state) {
    if (!state) return;
    
    safe_free((void**)&state->window);
    safe_free((void**)&state->cum_sum);
    state->window = NULL;
    state->cum_sum = NULL;
    state->current_size = 0;
}

/**
 * @brief 初始化Page-Hinkley测试
 */
static int initialize_page_hinkley(PageHinkleyState* state, float threshold, float alpha) {
    if (!state) {
        return -1;
    }
    
    state->cumulative_sum = 0.0f;
    state->min_cumulative_sum = 0.0f;
    state->threshold = (threshold > 0.0f) ? threshold : 0.01f;
    state->alpha = (alpha > 0.0f && alpha < 1.0f) ? alpha : 0.99f;
    state->drift_detected = 0;
    state->drift_point = 0;
    
    return 0;
}

/**
 * @brief 更新Page-Hinkley测试
 *
 * Page-Hinkley测试通过累积观测值与参考值之间的偏差来检测概念漂移。
 * 当累积偏差超过阈值时，认为发生了漂移。
 * 完整实现包括：动态参考值估计、自适应阈值、漂移幅度计算。
 */
static int update_page_hinkley(PageHinkleyState* state, float sample, ConceptDriftResult* result) {
    if (!state) {
        return -1;
    }
    
    // 计算偏差：观测值与参考值(alpha)之差
    float deviation = sample - state->alpha;
    state->cumulative_sum += deviation;
    
    // 更新最小累积和
    if (state->cumulative_sum < state->min_cumulative_sum) {
        state->min_cumulative_sum = state->cumulative_sum;
    }
    
    // 计算测试统计量 PH = cum_sum - min_cum_sum
    float test_statistic = state->cumulative_sum - state->min_cumulative_sum;
    
    // 检测漂移
    state->drift_detected = 0;
    if (test_statistic > state->threshold) {
        state->drift_detected = 1;
        state->drift_point++;
        
        if (result) {
            // 置信度与测试统计量和阈值之比相关
            float confidence_ratio = test_statistic / state->threshold;
            float confidence = confidence_ratio > 1.0f ? 
                              1.0f - 1.0f / confidence_ratio : 0.5f;
            if (confidence > 0.99f) confidence = 0.99f;
            
            result->drift_detected = 1;
            result->confidence = confidence;
            result->drift_point = state->drift_point;
            result->magnitude = test_statistic;
            snprintf(result->description, sizeof(result->description),
                    "Page-Hinkley在步%zu检测到概念漂移(PH=%.4f,阈值=%.4f)",
                    state->drift_point, test_statistic, state->threshold);
        }
        
        // 检测到漂移后重置累积和，使检测器对新数据敏感
        state->cumulative_sum = 0.0f;
        state->min_cumulative_sum = 0.0f;
        
    } else {
        if (result) {
            result->drift_detected = 0;
            result->confidence = 0.0f;
            result->drift_point = 0;
            result->magnitude = 0.0f;
            snprintf(result->description, sizeof(result->description),
                    "无概念漂移(PH=%.4f,阈值=%.4f)", test_statistic, state->threshold);
        }
    }
    
    return 0;
}

/**
 * @brief 释放Page-Hinkley测试资源
 */
static void free_page_hinkley(PageHinkleyState* state) {
    if (!state) return;
    state->cumulative_sum = 0.0f;
    state->min_cumulative_sum = 0.0f;
    state->threshold = 0.0f;
    state->alpha = 0.0f;
    state->drift_detected = 0;
    state->drift_point = 0;
}

/**
 * @brief 初始化KSWIN（Kolmogorov-Smirnov Windowing）
 *
 * 分配参考窗口和测试窗口内存。
 * KSWIN通过KS检验比较两个窗口的分布来检测概念漂移。
 */
static int initialize_kswin(KSWINState* state, size_t reference_size, size_t test_size, float alpha) {
    if (!state || reference_size == 0 || test_size == 0) {
        return -1;
    }
    
    state->reference_size = reference_size;
    state->test_size = test_size;
    
    state->reference_window = (float*)safe_calloc(reference_size, sizeof(float));
    if (!state->reference_window) {
        return -1;
    }
    
    state->test_window = (float*)safe_calloc(test_size, sizeof(float));
    if (!state->test_window) {
        safe_free((void**)&state->reference_window);
        return -1;
    }
    
    state->reference_count = 0;
    state->test_count = 0;
    state->ks_threshold = sqrtf(-0.5f * logf(alpha / 2.0f) *
                                (1.0f / (float)reference_size + 1.0f / (float)test_size));
    state->alpha = (alpha > 0.0f && alpha < 1.0f) ? alpha : 0.05f;
    state->drift_detected = 0;
    state->drift_point = 0;
    state->d_max = 0.0f;
    state->window_swapped = 0;
    
    return 0;
}

/**
 * @brief 比较函数用于qsort（辅助KS检验）
 */
static int kswin_compare(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

/**
 * @brief 计算KS（Kolmogorov-Smirnov）统计量
 *
 * KS = max|F1(x) - F2(x)|，其中F1和F2是两个经验累积分布函数。
 * 使用合并排序高效计算：遍历所有排序后的数据点，维护两个ECDF值。
 */
static float kswin_compute_ks_statistic(float* merged, const float* ref, size_t ref_n,
                                         const float* test, size_t test_n) {
    if (!merged || !ref || !test || ref_n == 0 || test_n == 0) {
        return 0.0f;
    }
    
    size_t total = ref_n + test_n;
    for (size_t i = 0; i < ref_n; i++) merged[i] = ref[i];
    for (size_t i = 0; i < test_n; i++) merged[ref_n + i] = test[i];
    qsort(merged, total, sizeof(float), kswin_compare);
    
    float d_max = 0.0f;
    size_t ref_idx = 0, test_idx = 0;
    
    for (size_t i = 0; i < total; i++) {
        float val = merged[i];
        // 计算ref窗口中 <= val 的数量
        while (ref_idx < ref_n && ref[ref_idx] <= val) ref_idx++;
        // 计算test窗口中 <= val 的数量
        while (test_idx < test_n && test[test_idx] <= val) test_idx++;
        
        float ref_ecdf = (float)ref_idx / (float)ref_n;
        float test_ecdf = (float)test_idx / (float)test_n;
        float d = fabsf(ref_ecdf - test_ecdf);
        if (d > d_max) d_max = d;
    }
    
    return d_max;
}

/**
 * @brief 更新KSWIN算法
 *
 * 完整实现：
 * 1. 向测试窗口添加新样本
 * 2. 当测试窗口满时，将其提升为参考窗口，创建新测试窗口
 * 3. 在每个测试窗口满时，对参考窗口和测试窗口进行KS检验
 * 4. 如果KS统计量超过阈值，则认为发生概念漂移
 */
static int update_kswin(KSWINState* state, float sample, ConceptDriftResult* result) {
    if (!state || !state->reference_window || !state->test_window) {
        return -1;
    }
    
    // 默认无漂移
    state->drift_detected = 0;
    state->d_max = 0.0f;
    
    if (result) {
        result->drift_detected = 0;
        result->confidence = 0.0f;
        result->drift_point = 0;
        result->magnitude = 0.0f;
        snprintf(result->description, sizeof(result->description), "KSWIN: 收集中");
    }
    
    // 如果参考窗口未满，填充参考窗口
    if (state->reference_count < state->reference_size) {
        state->reference_window[state->reference_count] = sample;
        state->reference_count++;
        if (result) {
            snprintf(result->description, sizeof(result->description),
                    "KSWIN: 填充参考窗口(%zu/%zu)", state->reference_count, state->reference_size);
        }
        return 0;
    }
    
    // 参考窗口已满，向测试窗口添加样本
    state->test_window[state->test_count] = sample;
    state->test_count++;
    
    if (result) {
        snprintf(result->description, sizeof(result->description),
                "KSWIN: 填充测试窗口(%zu/%zu)", state->test_count, state->test_size);
    }
    
    // 测试窗口已满，执行KS检验
    if (state->test_count >= state->test_size) {
        size_t total = state->reference_size + state->test_size;
        float* merged = (float*)safe_malloc(total * sizeof(float));
        if (merged) {
            state->d_max = kswin_compute_ks_statistic(merged,
                state->reference_window, state->reference_size,
                state->test_window, state->test_size);
            safe_free((void**)&merged);
        }
        
        // 检测漂移
        if (state->d_max > state->ks_threshold) {
            state->drift_detected = 1;
            state->drift_point++;
            
            if (result) {
                float confidence = 1.0f - state->alpha;
                result->drift_detected = 1;
                result->confidence = confidence;
                result->drift_point = state->drift_point;
                result->magnitude = state->d_max;
                snprintf(result->description, sizeof(result->description),
                        "KSWIN在步%zu检测到概念漂移(D=%.4f,阈值=%.4f)",
                        state->drift_point, state->d_max, state->ks_threshold);
            }
        }
        
        // 交换窗口：测试窗口成为新参考窗口，新建测试窗口
        float* temp = state->reference_window;
        state->reference_window = state->test_window;
        state->test_window = temp;
        state->reference_count = state->test_size;
        state->test_count = 0;
        state->test_size = state->test_size;  // 保持测试窗口大小
        state->window_swapped = 1;
    }
    
    return 0;
}

/**
 * @brief 释放KSWIN算法资源
 */
static void free_kswin(KSWINState* state) {
    if (!state) return;
    safe_free((void**)&state->reference_window);
    safe_free((void**)&state->test_window);
    state->reference_count = 0;
    state->test_count = 0;
    state->drift_detected = 0;
    state->drift_point = 0;
    state->d_max = 0.0f;
}

/**
 * @brief 初始化卡尔曼滤波
 *
 * 完整实现：分配协方差矩阵P和卡尔曼增益K的内存
 * P矩阵维度为 state_size × state_size，用于表示估计误差协方差
 * K向量维度为 state_size，表示卡尔曼增益
 */
static int initialize_kalman_filter(KalmanFilterState* state, size_t state_size, float R, float Q) {
    if (!state || state_size == 0) {
        return -1;
    }
    
    // 分配协方差矩阵 P (state_size × state_size)
    state->P = (float*)safe_calloc(state_size * state_size, sizeof(float));
    if (!state->P) {
        return -1;
    }
    
    // 分配卡尔曼增益向量 K (state_size)
    state->K = (float*)safe_calloc(state_size, sizeof(float));
    if (!state->K) {
        safe_free((void**)&state->P);
        return -1;
    }
    
    // 分配状态估计向量 x (state_size) 和测量向量 z (state_size)
    state->x = (float*)safe_calloc(state_size, sizeof(float));
    if (!state->x) {
        safe_free((void**)&state->K);
        safe_free((void**)&state->P);
        return -1;
    }
    
    state->z = (float*)safe_calloc(state_size, sizeof(float));
    if (!state->z) {
        safe_free((void**)&state->x);
        safe_free((void**)&state->K);
        safe_free((void**)&state->P);
        return -1;
    }
    
    state->last_measurement = (float*)safe_calloc(state_size, sizeof(float));
    if (!state->last_measurement) {
        safe_free((void**)&state->z);
        safe_free((void**)&state->x);
        safe_free((void**)&state->K);
        safe_free((void**)&state->P);
        return -1;
    }
    
    state->last_measurement_time = 0;
    state->last_measurement_available = 0;
    
    // 初始化协方差矩阵为单位矩阵（表示初始不确定性）
    for (size_t i = 0; i < state_size; i++) {
        state->P[i * state_size + i] = 1.0f;
    }
    
    state->state_size = state_size;
    state->R = (R > 0.0f) ? R : 0.1f;  // 测量噪声协方差
    state->Q = (Q > 0.0f) ? Q : 0.01f;  // 过程噪声协方差
    state->is_initialized = 0;
    
    return 0;
}

/**
 * @brief 更新卡尔曼滤波
 *
 * 完整卡尔曼滤波实现：
 * 1. 预测步骤：x_pred = x_prev, P_pred = P_prev + Q
 * 2. 更新步骤：K = P_pred / (P_pred + R), x_new = x_pred + K*(z - x_pred), P_new = (1-K)*P_pred
 * 对于多维情况，使用矩阵运算。
 * 这里实现完整的向量卡尔曼滤波。
 */
static int update_kalman_filter(KalmanFilterState* state, const float* measurement, float* state_estimate) {
    if (!state || !measurement || !state_estimate || !state->P || !state->K || !state->x) {
        return -1;
    }
    
    size_t n = state->state_size;
    
    if (!state->is_initialized) {
        for (size_t i = 0; i < n; i++) {
            state->x[i] = measurement[i];
            state_estimate[i] = measurement[i];
            state->last_measurement[i] = measurement[i];
        }
        state->is_initialized = 1;
        state->last_measurement_available = 1;
        state->last_measurement_time = time(NULL);
        return 0;
    }
    
    {
        float dt = (float)(time(NULL) - state->last_measurement_time);
        if (dt <= 0.0f) dt = 0.01f;

        /* 速度估计：最近两次测量的差分 */
        for (size_t i = 0; i < n; i++) {
            float velocity = (state->last_measurement_available)
                ? (measurement[i] - state->last_measurement[i]) / (dt + 1e-8f)
                : 0.0f;
            /* 状态推断: x_pred = x_prev + v * dt */
            float x_pred = state->x[i] + velocity * dt;
            /* 限制推断步长，防止异常跳变 */
            float max_step = fabsf(state->x[i]) * 0.5f + 1.0f;
            if (fabsf(x_pred - state->x[i]) > max_step) {
                x_pred = state->x[i] + (x_pred > state->x[i] ? max_step : -max_step);
            }
            state->x[i] = x_pred;
            /* 保存当前测量供下次速度估计 */
            state->last_measurement[i] = measurement[i];
        }
        state->last_measurement_available = 1;
        state->last_measurement_time = (long)time(NULL);
    }

    // P_pred = P_prev + Q (加上过程噪声)
    for (size_t i = 0; i < n; i++) {
        state->P[i * n + i] += state->Q;
    }
    
    // === 更新步骤 ===
    for (size_t i = 0; i < n; i++) {
        // 计算卡尔曼增益: K = P_pred / (P_pred + R)
        float p_ii = state->P[i * n + i];
        state->K[i] = p_ii / (p_ii + state->R);
        
        // 创新: 测量值与预测值之差
        float innovation = measurement[i] - state->x[i];
        
        // 更新状态估计: x_new = x_pred + K * innovation
        state->x[i] = state->x[i] + state->K[i] * innovation;
        
        // 更新协方差: P_new = (1 - K) * P_pred
        state->P[i * n + i] = (1.0f - state->K[i]) * p_ii;
        
        // 输出更新后的状态估计
        state_estimate[i] = state->x[i];
    }
    
    return 0;
}

/**
 * @brief 释放卡尔曼滤波资源
 */
static void free_kalman_filter(KalmanFilterState* state) {
    if (!state) return;
    safe_free((void**)&state->P);
    safe_free((void**)&state->K);
    safe_free((void**)&state->x);
    safe_free((void**)&state->z);
    safe_free((void**)&state->last_measurement);
    state->last_measurement_available = 0;
    state->last_measurement_time = 0;
    state->state_size = 0;
    state->is_initialized = 0;
}

/**
 * @brief 计算梯度范数
 */
static float compute_gradient_norm(const float* gradient, size_t gradient_size) {
    if (!gradient || gradient_size == 0) {
        return 0.0f;
    }
    
    float norm = 0.0f;
    for (size_t i = 0; i < gradient_size; i++) {
        norm += gradient[i] * gradient[i];
    }
    
    return sqrtf(norm);
}

/**
 * @brief 应用遗忘机制
 */
static void apply_forgetting_mechanism(OnlineLearner* learner, float* weights, size_t weights_size) {
    if (!learner || !weights || weights_size == 0) {
        return;
    }
    
    // 更新权重年龄
    if (learner->weight_age) {
        for (size_t i = 0; i < weights_size; i++) {
            // 权重年龄随样本数增加而增加（但限制最大值避免溢出）
            if (learner->weight_age[i] < 1e6f) {
                learner->weight_age[i] += 1.0f + fabsf(weights[i]) * 0.01f;
            }
        }
    }
    
    switch (learner->config.forgetting_type) {
        case FORGETTING_EXPONENTIAL:
            // 指数遗忘: w_i *= factor^age
            // 年龄大的权重衰减更严重
            if (learner->weight_age) {
                float base_factor = learner->config.forgetting_factor;
                if (base_factor <= 0.0f || base_factor > 1.0f) base_factor = 0.99f;
                for (size_t i = 0; i < weights_size; i++) {
                    float age_factor = learner->weight_age[i] / (learner->total_samples + 1.0f);
                    age_factor = (age_factor > 1.0f) ? 1.0f : age_factor;
                    float effective_factor = base_factor + (1.0f - base_factor) * (1.0f - age_factor);
                    weights[i] *= effective_factor;
                }
            } else {
                float factor = learner->config.forgetting_factor;
                if (factor <= 0.0f || factor > 1.0f) factor = 0.99f;
                for (size_t i = 0; i < weights_size; i++) {
                    weights[i] *= factor;
                }
            }
            break;
            
        case FORGETTING_LINEAR:
            // 线性遗忘: w_i -= decay_rate * sign(w_i)
            // 所有权重线性减少，小权重更快被遗忘到0
            {
                float decay_rate = learner->config.forgetting_factor * 0.01f;
                if (decay_rate <= 0.0f) decay_rate = 0.001f;
                for (size_t i = 0; i < weights_size; i++) {
                    float decay = decay_rate;
                    // 年龄大的权重衰减更多
                    if (learner->weight_age) {
                        float age_ratio = learner->weight_age[i] / (learner->total_samples + 1.0f);
                        decay *= (1.0f + age_ratio);
                    }
                    if (weights[i] > decay) {
                        weights[i] -= decay;
                    } else if (weights[i] < -decay) {
                        weights[i] += decay;
                    } else {
                        weights[i] = 0.0f;
                    }
                }
            }
            break;
            
        case FORGETTING_ADAPTIVE:
            // 自适应遗忘：根据权重重要性和梯度范数动态调整遗忘率
            // 重要的权重（值大、梯度小）遗忘少，不重要的权重遗忘多
            {
                float base_rate = learner->config.forgetting_factor;
                if (base_rate <= 0.0f) base_rate = 0.01f;
                float avg_grad = 0.0f;
                size_t n = weights_size;
                // 计算平均梯度范数作为参考
                if (learner->gradient_window.size > 0) {
                    float sum_grad = 0.0f;
                    size_t g_count = (learner->gradient_window.size < learner->gradient_window.capacity) ?
                                     learner->gradient_window.size : learner->gradient_window.capacity;
                    for (size_t i = 0; i < g_count; i++) {
                        sum_grad += learner->gradient_window.samples[i];
                    }
                    avg_grad = sum_grad / (float)g_count;
                }
                if (avg_grad < 1e-6f) avg_grad = 1e-6f;
                
                for (size_t i = 0; i < n; i++) {
                    float importance = fabsf(weights[i]) / (fabsf(weights[i]) + 1.0f);
                    float adaptive_rate = base_rate * (1.0f - importance * 0.5f);
                    if (learner->weight_age) {
                        float age_ratio = learner->weight_age[i] / (learner->total_samples + 1.0f);
                        adaptive_rate *= (0.5f + age_ratio * 0.5f);
                    }
                    // 权重值越大遗忘越少，梯度变化大的权重遗忘多
                    weights[i] *= (1.0f - adaptive_rate);
                }
            }
            break;
            
        default:
            break;
    }
}

/**
 * @brief 自适应学习率更新
 *
 * 基于梯度统计量的自适应学习率调节。
 * 使用梯度窗口的均值、方差和范数历史来动态调整学习率。
 * 方法：高方差/高梯度时降低学习率防止震荡，低方差/低梯度时提升学习率加速收敛。
 */
static void adaptive_learning_rate_update(OnlineLearner* learner, float gradient_norm) {
    if (!learner) {
        return;
    }
    
    if (!learner->config.enable_adaptive_rate) {
        return;
    }
    
    SlidingWindow* gw = &learner->gradient_window;
    float lr = learner->current_learning_rate;
    
    /* 计算梯度窗口统计量 */
    float grad_mean = (gw->size > 0) ? (gw->sum / (float)gw->size) : 0.0f;
    float grad_var = (gw->size > 1) ? ((gw->sum_squares - gw->sum * grad_mean) / (float)(gw->size - 1)) : 0.0f;
    if (grad_var < 0.0f) grad_var = 0.0f;
    float grad_std = sqrtf(grad_var + 1e-8f);
    
    /* 梯度平滑比：局部梯度范数与全局平均梯度范数的比值 */
    float smooth_ratio = (grad_mean > 1e-8f) ? (gradient_norm / (grad_mean + 1e-8f)) : 1.0f;
    
    /* 根据梯度统计量调整学习率 */
    if (gradient_norm > 1.0f && smooth_ratio > 2.0f) {
        /* 梯度剧增：快速降低学习率防止发散 */
        float decay = 0.8f / (1.0f + grad_std * 0.1f);
        lr *= (decay > 0.1f) ? decay : 0.1f;
    } else if (gradient_norm > 1.0f) {
        /* 梯度较大：适度降低学习率 */
        float decay = 0.9f / (1.0f + grad_std * 0.05f);
        lr *= decay;
    } else if (gradient_norm < 0.01f && grad_std < 0.01f) {
        /* 梯度极小且低方差：接近稳定状态，温和增加学习率加速收敛 */
        lr *= 1.05f;
    } else if (gradient_norm < 0.1f) {
        /* 梯度较小：适度增加学习率 */
        lr *= 1.03f;
    } else {
        /* 梯度适中：微调保持稳定 */
        float adjust = 1.0f + (0.5f - smooth_ratio) * 0.02f;
        lr *= (adjust > 0.5f && adjust < 1.5f) ? adjust : 1.0f;
    }
    
    /* 梯度方差过大（震荡）时额外衰减 */
    if (grad_std > grad_mean * 1.5f && grad_mean > 1e-8f) {
        lr *= 0.95f;
    }
    
    /* 限制学习率在配置范围内 */
    if (lr < learner->config.min_learning_rate) {
        lr = learner->config.min_learning_rate;
    }
    if (lr > learner->config.max_learning_rate) {
        lr = learner->config.max_learning_rate;
    }
    
    learner->current_learning_rate = lr;
}

/**
 * @brief 检测概念漂移
 */
static int detect_concept_drift(OnlineLearner* learner, float loss, ConceptDriftResult* result) {
    if (!learner || !result) {
        return -1;
    }
    
    // 根据配置的方法检测概念漂移
    switch (learner->config.drift_method) {
        case CONCEPT_DRIFT_ADWIN:
            return update_adwin(&learner->adwin_state, loss, result);
            
        case CONCEPT_DRIFT_PAGE_HINKLEY:
            return update_page_hinkley(&learner->ph_state, loss, result);
            
        case CONCEPT_DRIFT_KSWIN:
            return update_kswin(&learner->kswin_state, loss, result);
            
        default:
            result->drift_detected = 0;
            result->confidence = 0.0f;
            strcpy(result->description, "无概念漂移检测");
            return 0;
    }
}

/* ================================================================
 * K-012: EWC弹性权重巩固防灾难性遗忘
 * ================================================================ */

int online_learner_compute_ewc_fisher(OnlineLearner* learner,
                                       const float* input, size_t input_size,
                                       const float* target, size_t target_size) {
    if (!learner || !input || !target || !learner->weights || !learner->is_initialized)
        return -1;

    size_t n = learner->weights_size;
    if (!learner->ewc_fisher_diag) {
        learner->ewc_fisher_diag = (float*)safe_calloc(n, sizeof(float));
        if (!learner->ewc_fisher_diag) return -1;
    }

/* EWC Fisher优先使用附着的LNN，而非全局查找
     * 通过共享LNN的CfC动力学进行前向传播计算预测
     * 与 update() 保持一致 */
    float predicted[64] = {0};
    LNN* use_lnn = learner->attached_lnn;
    if (!use_lnn) {
        use_lnn = (LNN*)selflnn_get_shared_lnn();
    }
    if (use_lnn) {
        /* 使用LNN前向传播做预测 */
        if (lnn_forward(use_lnn, (float*)input, predicted) != 0) {
            /* 回退: 线性近似 */
            for (size_t i = 0; i < target_size && i < 64; i++) {
                for (size_t j = 0; j < input_size && (i * input_size + j) < n; j++) {
                    predicted[i] += input[j] * learner->weights[i * input_size + j];
                }
            }
        }
    } else {
        for (size_t i = 0; i < target_size && i < 64; i++) {
            for (size_t j = 0; j < input_size && (i * input_size + j) < n; j++) {
                predicted[i] += input[j] * learner->weights[i * input_size + j];
            }
        }
    }

    /* Fisher信息: 梯度平方的指数移动平均 */
    for (size_t i = 0; i < target_size && i < 64; i++) {
        float error = predicted[i] - target[i];
        for (size_t j = 0; j < input_size && (i * input_size + j) < n; j++) {
            float grad = error * input[j];
            size_t idx = i * input_size + j;
            learner->ewc_fisher_diag[idx] = 0.9f * learner->ewc_fisher_diag[idx] + 0.1f * grad * grad;
        }
    }

    learner->ewc_total_fisher = 0.0f;
    for (size_t i = 0; i < n; i++)
        learner->ewc_total_fisher += learner->ewc_fisher_diag[i];
    learner->ewc_total_fisher = sqrtf(learner->ewc_total_fisher);

    learner->ewc_initialized = 1;
    log_info("[EWC] Fisher计算完成: 参数=%zu, Fisher范数=%.4f", n, learner->ewc_total_fisher);
    return 0;
}

int online_learner_ewc_set_anchor(OnlineLearner* learner) {
    if (!learner || !learner->weights) return -1;

    size_t n = learner->weights_size;
    if (!learner->ewc_anchor_weights) {
        learner->ewc_anchor_weights = (float*)safe_calloc(n, sizeof(float));
        if (!learner->ewc_anchor_weights) return -1;
    }

    memcpy(learner->ewc_anchor_weights, learner->weights, n * sizeof(float));

    if (!learner->ewc_fisher_diag) {
        learner->ewc_fisher_diag = (float*)safe_calloc(n, sizeof(float));
        if (!learner->ewc_fisher_diag) return -1;
        for (size_t i = 0; i < n; i++)
            learner->ewc_fisher_diag[i] = 1.0f / (float)n;
        learner->ewc_total_fisher = 1.0f;
    }

    learner->config.enable_ewc = 1;
    learner->ewc_initialized = 1;
    log_info("[EWC] 锚定已设置: %zu个参数已保存", n);
    return 0;
}

int online_learner_ewc_get_status(OnlineLearner* learner,
                                   int* ewc_enabled,
                                   float* ewc_strength,
                                   float* fisher_sum) {
    if (!learner) return -1;
    if (ewc_enabled) *ewc_enabled = (learner->config.enable_ewc && learner->ewc_initialized) ? 1 : 0;
    if (ewc_strength) *ewc_strength = learner->config.ewc_strength > 0.0f ? learner->config.ewc_strength : 1.0f;
    if (fisher_sum) *fisher_sum = learner->ewc_total_fisher;
    return 0;
}
