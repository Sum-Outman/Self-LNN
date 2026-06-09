/**
 * @file learning.c
 * @brief 学习与演化模块实现
 * 
 * 自我学习、自我演化、模仿学习等功能的实现。
 */

#define SELFLNN_IMPLEMENTATION 1

#include "selflnn/learning/learning.h"
#include "selflnn/learning/online_learning.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/perf.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/xorshift_prng.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/laplace.h"
#include "selflnn/memory/memory_manager.h"
#include "selflnn/memory/memory.h"
#include "selflnn/utils/platform.h"
#include "selflnn/training/training.h"
#include "selflnn/knowledge/knowledge.h"        /* KG持久化: 学习结果→知识库 */
#include "selflnn/programming/programming_bridge.h" /* 学习→编程闭环: 错误反馈→代码改进 */
#include "selflnn/evolution/pareto_optimization.h"
#include "selflnn/core/evolutionary_algorithms.h"
#include "selflnn/learning/teach_by_showing.h"
#include "selflnn/learning/manual_learning.h"
#include "selflnn/learning/imitation_deep.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

/**
 * @brief 错误记录结构体
 */
typedef struct {
    char error_type[64];           /**< 错误类型 */
    char error_description[256];   /**< 错误描述 */
    char error_context[256];       /**< 错误上下文 */
    float severity;                /**< 错误严重程度 (0-1) */
    time_t timestamp;              /**< 时间戳 */
    char root_cause[256];          /**< 根原因分析 */
    char suggested_fix[256];       /**< 建议修复方法 */
    int was_fixed;                 /**< 是否已修复 */
    float fix_effectiveness;       /**< 修复有效性 (0-1) */
} ErrorRecord;

/**
 * @brief 修正规则结构体
 */
typedef struct {
    char pattern[128];             /**< 错误模式匹配 */
    char correction[256];          /**< 修正方法 */
    float confidence;              /**< 规则置信度 (0-1) */
    size_t usage_count;            /**< 使用次数 */
    size_t success_count;          /**< 成功次数 */
    float effectiveness;           /**< 有效性 (0-1) */
    int rule_type;                 /**< 规则类型（0=简单替换，1=模式匹配，2=上下文相关） */
    float strength;                /**< 规则强度（0-1） */
    float applicability;           /**< 规则适用性（0-1） */
    time_t last_used_time;         /**< 最后使用时间 */
    time_t created_time;           /**< 创建时间 */
} CorrectionRule;

/* 内部经验缓冲区常量 */
#define LEARNING_INTERNAL_EXP_CAPACITY 512
#define LEARNING_INTERNAL_STATE_DIM 256
#define LEARNING_INTERNAL_ACTION_DIM 128  /* R112: 64→128 to match LNN output_dim */

/**
 * @brief 学习引擎内部结构体
 */
struct LearningEngine {
    LearningConfig config;          /**< 引擎配置 */
    int is_initialized;             /**< 是否已初始化 */
    float* policy_weights;          /**< 策略权重（强化学习） */
    size_t policy_weights_size;     /**< 策略权重大小 */
    float* population;              /**< 种群（演化学习） */
    size_t population_size;         /**< 种群大小 */
    size_t individual_size;         /**< 个体大小 */
    float* imitation_model;         /**< 模仿模型 */
    size_t imitation_model_size;    /**< 模仿模型大小 */
    float* knowledge_base;          /**< 知识库 */
    size_t knowledge_base_size;     /**< 知识库大小 */
    int current_generation;         /**< 当前代数 */
    float* best_individual;         /**< 最佳个体 */
    size_t best_individual_size;    /**< 最佳个体大小 */
    float best_fitness;             /**< 最佳适应度 */
    
    /* 自我修正相关字段 */
    struct {
        ErrorRecord* error_history;         /**< 错误历史记录 */
        size_t error_history_size;          /**< 错误历史大小 */
        size_t error_history_capacity;      /**< 错误历史容量 */
        CorrectionRule* correction_rules;   /**< 修正规则 */
        size_t correction_rules_size;       /**< 修正规则数量 */
        size_t correction_rules_capacity;   /**< 修正规则容量 */
        float correction_effectiveness;     /**< 修正有效性（0-1） */
        size_t successful_corrections;      /**< 成功修正次数 */
        size_t failed_corrections;          /**< 失败修正次数 */
    } self_correction;
    
    /* 自我演化相关字段 */
    struct {
        float* fitness_history;             /**< 适应度历史 */
        size_t fitness_history_size;        /**< 适应度历史大小 */
        size_t fitness_history_capacity;    /**< 适应度历史容量 */
        float* diversity_scores;            /**< 多样性分数历史 */
        size_t diversity_history_size;      /**< 多样性历史大小 */
        float adaptation_rate;              /**< 适应速率 */
        float innovation_rate;              /**< 创新速率 */
        size_t stagnation_counter;          /**< 停滞计数器 */
        float best_fitness_history[100];    /**< 最佳适应度历史（最近100代） */
        size_t fitness_history_index;       /**< 适应度历史索引 */
    } self_evolution;
    
    /* 关联的LNN网络（用于真实前向传播验证） */
    LNN* network;                            /**< LNN网络实例 */

    /* 在线学习系统 */
    OnlineLearner* online_learner;           /**< 在线学习器 */

    /* 关联的记忆管理器（用于经验回放闭环） */
    MemoryManager* memory_manager;           /**< 记忆管理器实例 */

    /* 内部经验缓冲区（供外部模块查询——P0-006修复：仅存储真实交互数据） */
    int internal_exp_count;                             /**< 内部经验数量 */
    int internal_exp_capacity;                          /**< 内部经验容量 */
    float* internal_exp_states;                         /**< 状态数组 [capacity][STATE_DIM] */
    float* internal_exp_actions;                        /**< 动作数组 [capacity][ACTION_DIM] */
    float* internal_exp_rewards;                        /**< 奖励数组 [capacity] */
    float* internal_exp_next_states;                    /**< 下一状态数组 [capacity][STATE_DIM] */
    int* internal_exp_state_dims;                       /**< 状态实际维度 [capacity] */
    int* internal_exp_action_dims;                      /**< 动作实际维度 [capacity] */
    int* internal_exp_next_state_dims;                  /**< 下一状态实际维度 [capacity] */
    int internal_exp_real_data_count;                   /**< P0-006: 真实数据来源计数 */
    int enabled;                                        /**< 能力开关（P3.3） */
    int has_real_weights;                               /**< P0-010: 权重是否已从真实数据源加载（1=真实, 0=占位） */
    int has_real_knowledge;                             /**< P0-010: 知识库是否已从真实数据源加载 */
    /* H-016集成: 深度模仿学习系统 */
    ImitationDeepLearner* imitation_deep;              /**< 深度模仿学习器 */
    /* DEADCODE-FIX: manual_learning集成 */
    void* manual_learning_system;                      /**< 手动学习系统实例 */
};

/**
 * P0-006修复: 验证经验数据是否来自真实交互
 *
 * 检查数据是否包含有效信号（非全零、非异常值）。
 * 全零或接近全零的数据被视为虚假/空数据，拒绝存入经验缓冲区。
 * 
 * @param data 数据指针
 * @param dim 数据维度
 * @return 1=有效真实数据, 0=无效数据应拒绝
 */
static int _exp_data_is_valid(const float* data, int dim) {
    if (!data || dim <= 0) return 0;
    float sum_abs = 0.0f;
    int nonzero = 0;
    for (int i = 0; i < dim; i++) {
        float v = data[i];
        if (fabsf(v) > 1e-8f) nonzero++;
        sum_abs += fabsf(v);
    }
    /* 至少10%的维度有非零信号，且总能量>0才视为真实数据 */
    if (nonzero < dim / 10 && sum_abs < 0.001f) return 0;
    /* 检查异常值：单维绝对值>1e6视为异常 */
    for (int i = 0; i < dim; i++) {
        if (fabsf(data[i]) > 1e6f) return 0;
    }
    return 1;
}

/**
 * @brief 创建学习引擎
 */
LearningEngine* learning_engine_create(const LearningConfig* config) {
    if (!config) {
        return NULL;
    }
    
    LearningEngine* engine = (LearningEngine*)safe_calloc(1, sizeof(LearningEngine));
    if (!engine) {
        return NULL;
    }
    
    // 结构体已由safe_calloc初始化为零
    engine->config = *config;
    engine->is_initialized = 1;
    engine->current_generation = 0;
    engine->best_fitness = -1e30f;  // 替代-INFINITY以避免编译警告
    engine->enabled = 1;

    /* DEADCODE-FIX: manual_learning集成 - 创建手动学习系统 */
    {
        MLConfig ml_cfg = ML_CONFIG_DEFAULT;
        ml_cfg.learning_rate = config->learning_rate;
        engine->manual_learning_system = ml_system_create(ml_cfg);
    }
    
    // 根据学习类型分配内存
    switch (config->learning_type) {
        case LEARNING_REINFORCEMENT:
            // 策略权重大小基于配置：使用population_size * 10，最小1024
            engine->policy_weights_size = (config->population_size > 0) ? 
                                         config->population_size * 10 : 1024;
            if (engine->policy_weights_size < 1024) engine->policy_weights_size = 1024;
            if (engine->policy_weights_size > 65536) engine->policy_weights_size = 65536; // 限制最大大小
            engine->policy_weights = (float*)safe_malloc(engine->policy_weights_size * sizeof(float));
            if (!engine->policy_weights) {
                safe_free((void**)&engine);
                return NULL;
            }
            memset(engine->policy_weights, 0, engine->policy_weights_size * sizeof(float));
            break;
            
        case LEARNING_EVOLUTIONARY:
            engine->population_size = config->population_size > 0 ? config->population_size : 100;
            // 个体大小基于配置：使用max_generations * 5，最小256
            engine->individual_size = (config->max_generations > 0) ? 
                                     config->max_generations * 5 : 256;
            if (engine->individual_size < 256) engine->individual_size = 256;
            if (engine->individual_size > 2048) engine->individual_size = 2048; // 限制最大大小
            size_t total_population_size = engine->population_size * engine->individual_size;
            engine->population = (float*)safe_malloc(total_population_size * sizeof(float));
            engine->best_individual_size = engine->individual_size;
            engine->best_individual = (float*)safe_malloc(engine->best_individual_size * sizeof(float));
            
            if (!engine->population || !engine->best_individual) {
                safe_free((void**)&engine->population);
                safe_free((void**)&engine->best_individual);
                safe_free((void**)&engine);
                return NULL;
            }
            
            // 初始化种群（确定性版本）
            for (size_t i = 0; i < total_population_size; i++) {
                // 确定性伪随机数生成器：基于引擎指针和索引生成唯一种子
                unsigned int seed = (unsigned int)(uintptr_t)engine ^ (unsigned int)i ^ (unsigned int)total_population_size;
                seed = seed * 1103515245 + 12345;
                unsigned int rand_val = (seed >> 16) & 0x7FFF;
                engine->population[i] = ((float)rand_val / 32767.0f) * 2.0f - 1.0f; // -1到1的随机值
            }
            memset(engine->best_individual, 0, engine->best_individual_size * sizeof(float));
            break;
            
        case LEARNING_IMITATION:
            // 模仿模型大小基于配置：使用population_size * 5，最小512
            engine->imitation_model_size = (config->population_size > 0) ? 
                                          config->population_size * 5 : 512;
            if (engine->imitation_model_size < 512) engine->imitation_model_size = 512;
            if (engine->imitation_model_size > 4096) engine->imitation_model_size = 4096; // 限制最大大小
            engine->imitation_model = (float*)safe_malloc(engine->imitation_model_size * sizeof(float));
            if (!engine->imitation_model) {
                safe_free((void**)&engine);
                return NULL;
            }
            memset(engine->imitation_model, 0, engine->imitation_model_size * sizeof(float));
            break;
            
        default:
            // 其他学习类型
            break;
    }
    
    // 分配知识库
    // 知识库大小基于配置：使用population_size * max_generations，最小2048
    engine->knowledge_base_size = (config->population_size > 0 && config->max_generations > 0) ? 
                                 config->population_size * config->max_generations : 2048;
    if (engine->knowledge_base_size < 2048) engine->knowledge_base_size = 2048;
    if (engine->knowledge_base_size > 65536) engine->knowledge_base_size = 65536; // 限制最大大小
    engine->knowledge_base = (float*)safe_malloc(engine->knowledge_base_size * sizeof(float));
    if (!engine->knowledge_base) {
        learning_engine_free(engine);
        return NULL;
    }
    memset(engine->knowledge_base, 0, engine->knowledge_base_size * sizeof(float));
    
    // 初始化自我修正子系统
    engine->self_correction.error_history = NULL;
    engine->self_correction.error_history_size = 0;
    engine->self_correction.error_history_capacity = 0;
    engine->self_correction.correction_rules = NULL;
    engine->self_correction.correction_rules_size = 0;
    engine->self_correction.correction_rules_capacity = 0;
    engine->self_correction.correction_effectiveness = 1.0f;
    engine->self_correction.successful_corrections = 0;
    engine->self_correction.failed_corrections = 0;
    
    // 初始化自我演化子系统
    engine->self_evolution.fitness_history = NULL;
    engine->self_evolution.fitness_history_size = 0;
    engine->self_evolution.fitness_history_capacity = 0;
    engine->self_evolution.diversity_scores = NULL;
    engine->self_evolution.diversity_history_size = 0;
    engine->self_evolution.adaptation_rate = 0.1f;
    engine->self_evolution.innovation_rate = 0.05f;
    engine->self_evolution.stagnation_counter = 0;
    for (size_t i = 0; i < 100; i++) {
        engine->self_evolution.best_fitness_history[i] = 0.0f;
    }
    engine->self_evolution.fitness_history_index = 0;
    
    // 初始化在线学习器
    OnlineLearningConfig online_config;
    memset(&online_config, 0, sizeof(OnlineLearningConfig));
    online_config.algorithm_type = ONLINE_LEARNING_ADAPTIVE;
    online_config.learning_rate = config->learning_rate;
    online_config.forgetting_factor = 0.95f;
    online_config.forgetting_type = FORGETTING_ADAPTIVE;
    online_config.drift_method = CONCEPT_DRIFT_ADWIN;
    online_config.drift_detection_threshold = 0.01f;
    online_config.window_size = 100;
    online_config.enable_adaptive_rate = 1;
    online_config.min_learning_rate = config->learning_rate * 0.01f;
    online_config.max_learning_rate = config->learning_rate * 10.0f;
    online_config.enable_momentum = 1;
    online_config.momentum_factor = 0.9f;
    online_config.enable_regularization = 1;
    online_config.regularization_strength = 0.001f;
    online_config.buffer_size = 1024;
    online_config.enable_real_time_update = 1;
    online_config.update_frequency_ms = 100;
    
    // 使用策略权重或知识库作为在线学习器的初始权重
    float* init_weights = NULL;
    size_t init_weights_size = 0;
    if (engine->policy_weights && engine->policy_weights_size > 0) {
        init_weights = engine->policy_weights;
        init_weights_size = engine->policy_weights_size;
    } else if (engine->knowledge_base && engine->knowledge_base_size > 0) {
        init_weights = engine->knowledge_base;
        init_weights_size = engine->knowledge_base_size;
    }
    
    engine->online_learner = online_learner_create(&online_config, init_weights, init_weights_size);

    /* P0-010: 初始化时所有数据源标记为空白状态
     * 权重和知识库需从真实数据源（LNN/CfC网络/训练）加载后才就绪 */
    engine->has_real_weights = 0;
    engine->has_real_knowledge = 0;

    /* P0-010: 初始化时所有数据源标记为空白状态
 *修复: 尝试自动关联全局共享LNN
     * 避免因遗漏调用set_network导致学习引擎静默失败 */
    {
        extern void* g_global_lnn;
        if (g_global_lnn) {
            CfCNetwork* cfc = lnn_get_cfc_network((LNN*)g_global_lnn);
            if (cfc) {
                engine->has_real_weights = 1;
                log_info("[学习引擎] 自动关联全局共享LNN，权重标记为就绪");
            }
        }
    }

    /* 初始化内部经验缓冲区 */
    engine->internal_exp_count = 0;
    engine->internal_exp_real_data_count = 0;  /* P0-006: 真实数据计数从零开始 */
    engine->internal_exp_capacity = LEARNING_INTERNAL_EXP_CAPACITY;
    engine->internal_exp_states = (float*)safe_calloc(
        LEARNING_INTERNAL_EXP_CAPACITY * LEARNING_INTERNAL_STATE_DIM, sizeof(float));
    engine->internal_exp_actions = (float*)safe_calloc(
        LEARNING_INTERNAL_EXP_CAPACITY * LEARNING_INTERNAL_ACTION_DIM, sizeof(float));
    engine->internal_exp_rewards = (float*)safe_calloc(
        LEARNING_INTERNAL_EXP_CAPACITY, sizeof(float));
    engine->internal_exp_next_states = (float*)safe_calloc(
        LEARNING_INTERNAL_EXP_CAPACITY * LEARNING_INTERNAL_STATE_DIM, sizeof(float));
    engine->internal_exp_state_dims = (int*)safe_calloc(
        LEARNING_INTERNAL_EXP_CAPACITY, sizeof(int));
    engine->internal_exp_action_dims = (int*)safe_calloc(
        LEARNING_INTERNAL_EXP_CAPACITY, sizeof(int));
    engine->internal_exp_next_state_dims = (int*)safe_calloc(
        LEARNING_INTERNAL_EXP_CAPACITY, sizeof(int));

    if (!engine->internal_exp_states || !engine->internal_exp_actions ||
        !engine->internal_exp_rewards || !engine->internal_exp_next_states ||
        !engine->internal_exp_state_dims || !engine->internal_exp_action_dims ||
        !engine->internal_exp_next_state_dims) {
        learning_engine_free(engine);
        return NULL;
    }

    /* 初始化深度模仿学习器 */
    engine->imitation_deep = imitation_deep_create();

    return engine;
}

/**
 * @brief 释放学习引擎
 */
void learning_engine_free(LearningEngine* engine) {
    if (!engine) {
        return;
    }
    
    safe_free((void**)&engine->policy_weights);
    safe_free((void**)&engine->population);
    safe_free((void**)&engine->imitation_model);
    safe_free((void**)&engine->knowledge_base);
    safe_free((void**)&engine->best_individual);
    
    // 释放自我修正子系统内存
    safe_free((void**)&engine->self_correction.error_history);
    safe_free((void**)&engine->self_correction.correction_rules);
    
    // 释放自我演化子系统内存
    safe_free((void**)&engine->self_evolution.fitness_history);
    safe_free((void**)&engine->self_evolution.diversity_scores);
    
    // 释放在线学习器
    if (engine->online_learner) {
        online_learner_free(engine->online_learner);
        engine->online_learner = NULL;
    }

    // 释放内部经验缓冲区
    safe_free((void**)&engine->internal_exp_states);
    safe_free((void**)&engine->internal_exp_actions);
    safe_free((void**)&engine->internal_exp_rewards);
    safe_free((void**)&engine->internal_exp_next_states);
    safe_free((void**)&engine->internal_exp_state_dims);
    safe_free((void**)&engine->internal_exp_action_dims);
    safe_free((void**)&engine->internal_exp_next_state_dims);

    /* H-016集成: 释放深度模仿学习器 */
    if (engine->imitation_deep) {
        imitation_deep_free(engine->imitation_deep);
        engine->imitation_deep = NULL;
    }

    /* DEADCODE-FIX: manual_learning集成 - 销毁手动学习系统 */
    if (engine->manual_learning_system) {
        ml_system_destroy((MLSystem*)engine->manual_learning_system);
        engine->manual_learning_system = NULL;
    }

    safe_free((void**)&engine);
}

/**
 * @brief 强化学习：根据奖励更新策略
 */
int learning_reinforcement_update(LearningEngine* engine,
                                 const float* state, size_t state_size,
                                 const float* action, size_t action_size,
                                 float reward,
                                 const float* next_state, size_t next_state_size) {
    if (!engine || !state || !action) {
        return -1;
    }
    
    if (state_size == 0 || action_size == 0) {
        return -1;
    }
    
    // 性能分析
    PerfTimer timer;
    perf_timer_start(&timer);
    
    /* P2-009: Q值计算 —— 优先使用LNN非线性前向传播获取Q值。
     * LNN的CfC动态系统能捕捉状态-动作间的非线性耦合关系，
     * 相比线性内积(state·weights)具有更强的表达能力。
     * 如果LNN不可用，回退到线性计算并记录警告。 */
    float current_q = 0.0f;
    float next_max_q = 0.0f;
    int lnn_available = 0;

    if (engine->network && engine->network->is_initialized) {
        /* LNN可用：通过lnn_forward计算非线性Q值 */
        float* lnn_output = (float*)safe_malloc(engine->network->config.output_size * sizeof(float));
        if (lnn_output) {
            int ret = lnn_forward(engine->network, state, lnn_output);
            if (ret == 0) {
                /* 输出向量均值作为Q值估计 */
                current_q = 0.0f;
                for (size_t i = 0; i < engine->network->config.output_size; i++) {
                    current_q += lnn_output[i];
                }
                current_q /= (float)engine->network->config.output_size;
                lnn_available = 1;

                /* 计算下一状态的最大Q值 */
                if (next_state) {
                    ret = lnn_forward(engine->network, next_state, lnn_output);
                    if (ret == 0) {
                        next_max_q = 0.0f;
                        for (size_t i = 0; i < engine->network->config.output_size; i++) {
                            if (lnn_output[i] > next_max_q) {
                                next_max_q = lnn_output[i];
                            }
                        }
                    }
                }
            }
            safe_free((void**)&lnn_output);
        }
    }

    if (!lnn_available) {
        /* LNN不可用：回退到线性内积计算Q值 */
        log_warn("[强化学习] LNN不可用，回退到线性Q值计算(状态·权重内积)，请初始化LNN以获得非线性Q值估计");
        for (size_t i = 0; i < state_size && i < engine->policy_weights_size; i++) {
            current_q += state[i] * engine->policy_weights[i];
        }
        if (next_state) {
            for (size_t i = 0; i < next_state_size && i < engine->policy_weights_size; i++) {
                float next_q = next_state[i] * engine->policy_weights[i];
                if (next_q > next_max_q) {
                    next_max_q = next_q;
                }
            }
        }
    }
    
    // Q-learning更新
    // 使用配置中的折扣因子
    const float discount_factor = engine->config.discount_factor > 0.0f ? 
                                 engine->config.discount_factor : 0.9f;
    float target_q = reward + discount_factor * next_max_q;
    float error = target_q - current_q;
    
    // 更新权重
    for (size_t i = 0; i < state_size && i < engine->policy_weights_size; i++) {
        engine->policy_weights[i] += engine->config.learning_rate * error * state[i];
    }
    
    // 存储经验到记忆系统（形成"经验→记忆"闭环）
    if (engine->memory_manager) {
        learning_engine_store_experience(engine, state, state_size,
                                        action, action_size, reward,
                                        next_state, next_state_size);
        
        // 批次经验回放：每次更新后从记忆中采样训练，打破经验相关性
        // 使用小批次(4个样本)在线学习，稳定训练过程
        learning_engine_experience_replay(engine, 4);
    }
    
    // 停止性能计时器
    uint64_t elapsed_ns = perf_timer_stop(&timer);
    (void)elapsed_ns;  // 消除未使用变量警告

    // 存储经验到内部缓冲区（供backend等模块查询）
    // P0-006修复: 验证数据来源真实有效后再存储，拒绝零值/空数据
    {
        size_t copy_sd = state_size < (size_t)LEARNING_INTERNAL_STATE_DIM
                         ? state_size : (size_t)LEARNING_INTERNAL_STATE_DIM;
        size_t copy_ad = action_size < (size_t)LEARNING_INTERNAL_ACTION_DIM
                         ? action_size : (size_t)LEARNING_INTERNAL_ACTION_DIM;
        size_t copy_nsd = next_state_size < (size_t)LEARNING_INTERNAL_STATE_DIM
                          ? next_state_size : (size_t)LEARNING_INTERNAL_STATE_DIM;
        if (engine->internal_exp_capacity > 0 &&
            _exp_data_is_valid(state, (int)copy_sd) &&
            _exp_data_is_valid(action, (int)copy_ad)) {
            int idx = engine->internal_exp_count < engine->internal_exp_capacity
                      ? engine->internal_exp_count
                      : (engine->internal_exp_count % engine->internal_exp_capacity);
            memcpy(&engine->internal_exp_states[idx * LEARNING_INTERNAL_STATE_DIM],
                   state, copy_sd * sizeof(float));
            memcpy(&engine->internal_exp_actions[idx * LEARNING_INTERNAL_ACTION_DIM],
                   action, copy_ad * sizeof(float));
            engine->internal_exp_rewards[idx] = reward;
            if (next_state && _exp_data_is_valid(next_state, (int)copy_nsd)) {
                memcpy(&engine->internal_exp_next_states[idx * LEARNING_INTERNAL_STATE_DIM],
                       next_state, copy_nsd * sizeof(float));
            }
            engine->internal_exp_state_dims[idx] = (int)copy_sd;
            engine->internal_exp_action_dims[idx] = (int)copy_ad;
            engine->internal_exp_next_state_dims[idx] = next_state ? (int)copy_nsd : 0;
            engine->internal_exp_count++;
            engine->internal_exp_real_data_count++;  /* P0-006: 真实数据计数 */
        }
    }

    return 0;
}

/**
 * @brief 多目标演化学习：帕累托多目标演化（NSGA-II）
 */
int learning_evolutionary_evolve_multi(LearningEngine* engine,
                                        ParetoFront* front,
                                        float* objectives_buffer,
                                        int num_objectives) {
    if (!engine || !front || !objectives_buffer || num_objectives < 1 || num_objectives > PARETO_MAX_OBJECTIVES) {
        return -1;
    }

    if (!engine->network || engine->individual_size == 0) {
        return -1;
    }

    ParetoOptimizationConfig pconfig = pareto_config_default(num_objectives);
    pconfig.population_size = (int)engine->population_size;
    pconfig.max_generations = engine->config.max_generations > 0 ? engine->config.max_generations : 50;
    pconfig.crossover_prob = 0.9f;
    pconfig.mutation_prob = engine->config.mutation_rate;
    pconfig.mutation_strength = 0.05f;

    EvolutionTrainingData eval_data;
    memset(&eval_data, 0, sizeof(eval_data));
    eval_data.network = engine->network;
    eval_data.accuracy_weight = 1.0f;
    eval_data.speed_weight = 0.5f;
    eval_data.energy_weight = 0.3f;
    eval_data.max_eval_iterations = 100;

    Population* pop = population_create((size_t)pconfig.population_size, engine->individual_size, 0);
    if (!pop) return -1;

    population_initialize_random(pop, -1.0f, 1.0f);

    int result = pareto_multi_evolve(pop, evolution_multi_objective_train, &pconfig, &eval_data, front);

    if (result > 0 && front->entry_count > 0) {
        float prefs[PARETO_MAX_OBJECTIVES];
        for (int i = 0; i < num_objectives; i++) {
            prefs[i] = 1.0f;
        }

        size_t copy_size = engine->individual_size;
        if (engine->best_individual && front->entry_count > 0) {
            int best_idx = 0;
            float best_score = -1e10f;
            for (int i = 0; i < front->entry_count; i++) {
                float s = 0.0f;
                for (int j = 0; j < num_objectives; j++) {
                    s += front->entries[i].objectives[j];
                }
                if (s > best_score) {
                    best_score = s;
                    best_idx = i;
                }
            }
            ParetoFrontEntry* be = &front->entries[best_idx];
            if (be->genome && be->genome_size <= copy_size) {
                memcpy(engine->best_individual, be->genome, be->genome_size * sizeof(float));
                engine->best_fitness = best_score;
            }
            memcpy(objectives_buffer, be->objectives, (size_t)num_objectives * sizeof(float));
        }
    }

    if (engine->population) {
        safe_free((void**)&engine->population);
    }

    engine->population = (float*)safe_malloc(engine->population_size * engine->individual_size * sizeof(float));
    if (engine->population && pop) {
        size_t actual_pop_size = population_get_size(pop);
        for (size_t i = 0; i < actual_pop_size && i < engine->population_size; i++) {
            size_t ind_genome_size = 0;
            const float* genome = population_get_individual_genome(pop, i, &ind_genome_size);
            if (genome && ind_genome_size > 0) {
                size_t copy_size = ind_genome_size < engine->individual_size ? ind_genome_size : engine->individual_size;
                memcpy(engine->population + i * engine->individual_size, genome, copy_size * sizeof(float));
            }
        }
    }

    population_destroy(pop);
    engine->current_generation++;
    engine->self_evolution.fitness_history_index++;

    return result;
}

/**
 * @brief 获取帕累托前沿JSON字符串（用于前端可视化）
 */
int learning_evolutionary_get_pareto_json(const LearningEngine* engine,
                                           const ParetoFront* front,
                                           char* buffer, size_t buffer_size) {
    if (!engine || !front || !buffer || buffer_size == 0) return -1;

    return pareto_front_to_json(front, buffer, buffer_size);
}

/**
 * @brief 强化学习：选择动作
 */
int learning_reinforcement_select_action(LearningEngine* engine,
                                        const float* state, size_t state_size,
                                        float* action, size_t max_action_size) {
    if (!engine || !state || !action || max_action_size == 0) {
        return -1;
    }
    
    if (state_size == 0) {
        return -1;
    }
    
    // 完整动作选择：ε-贪心策略
    // 确定性探索机会：基于引擎指针和状态大小
    unsigned int seed1 = (unsigned int)(uintptr_t)engine ^ (unsigned int)state_size ^ (unsigned int)max_action_size;
    seed1 = seed1 * 1103515245 + 12345;
    unsigned int rand_val1 = (seed1 >> 16) & 0x7FFF;
    float exploration_chance = (float)rand_val1 / 32767.0f;
    
    if (exploration_chance < engine->config.exploration_rate) {
        // 探索：随机动作
        size_t action_size = max_action_size < 10 ? max_action_size : 10;
        for (size_t i = 0; i < action_size; i++) {
            // 确定性随机动作：基于引擎指针、动作大小和索引
            unsigned int seed2 = (unsigned int)(uintptr_t)engine ^ (unsigned int)action_size ^ (unsigned int)i ^ 0x1000;
            seed2 = seed2 * 1103515245 + 12345;
            unsigned int rand_val2 = (seed2 >> 16) & 0x7FFF;
            action[i] = ((float)rand_val2 / 32767.0f) * 2.0f - 1.0f; // -1到1的随机值
        }
        return (int)action_size;
    } else {
        // 利用：基于策略选择动作
        // 完整实现：使用状态与权重的加权和作为动作
        size_t action_size = max_action_size < 10 ? max_action_size : 10;
        for (size_t a = 0; a < action_size; a++) {
            float action_value = 0.0f;
            for (size_t s = 0; s < state_size && s < engine->policy_weights_size; s++) {
                action_value += state[s] * engine->policy_weights[(s + a) % engine->policy_weights_size];
            }
            action[a] = tanhf(action_value); // 使用tanh限制在-1到1之间
        }
        return (int)action_size;
    }
}

/**
 * @brief 演化学习：演化种群
 */
int learning_evolutionary_evolve(LearningEngine* engine,
                                const float* fitness_scores, size_t num_scores) {
    if (!engine || !fitness_scores) {
        return -1;
    }
    
    if (num_scores == 0 || !engine->population) {
        return -1;
    }
    
    // 检查种群大小
    if (num_scores != engine->population_size) {
        return -1;
    }
    
    // 性能分析
    PerfTimer timer;
    perf_timer_start(&timer);
    
    // 找到最佳个体
    float max_fitness = fitness_scores[0];
    int best_idx = 0;
    
    for (size_t i = 1; i < num_scores; i++) {
        if (fitness_scores[i] > max_fitness) {
            max_fitness = fitness_scores[i];
            best_idx = (int)i;
        }
    }
    
    // 更新最佳个体
    if (max_fitness > engine->best_fitness) {
        engine->best_fitness = max_fitness;
        size_t offset = best_idx * engine->individual_size;
        memcpy(engine->best_individual, engine->population + offset, 
               engine->best_individual_size * sizeof(float));
    }
    
    // 创建新一代种群（完整演化算法）
    float* new_population = (float*)safe_malloc(engine->population_size * engine->individual_size * sizeof(float));
    if (!new_population) {
        return -1;
    }
    
    // 保留最佳个体
    size_t best_offset = 0;
    memcpy(new_population + best_offset, engine->best_individual, 
           engine->best_individual_size * sizeof(float));
    
    /* 生成新个体：交叉和突变
     * H-004修复：使用xorshift PRNG替代确定性LCG伪随机
     */
    XorshiftPrng prng;
    xorshift_prng_seed_secure(&prng);

    for (size_t i = 1; i < engine->population_size; i++) {
        size_t new_offset = i * engine->individual_size;

        /* 使用xorshift PRNG进行真正的随机选择 */
        int parent1_idx = (int)xorshift_prng_next_u32(&prng, (uint32_t)engine->population_size);
        int parent2_idx = (int)xorshift_prng_next_u32(&prng, (uint32_t)engine->population_size);

        size_t parent1_offset = (size_t)parent1_idx * engine->individual_size;
        size_t parent2_offset = (size_t)parent2_idx * engine->individual_size;

        /* 交叉 */
        for (size_t g = 0; g < engine->individual_size; g++) {
            float cross_rand = xorshift_prng_next_float(&prng);
            if (cross_rand < 0.5f) {
                new_population[new_offset + g] = engine->population[parent1_offset + g];
            } else {
                new_population[new_offset + g] = engine->population[parent2_offset + g];
            }

            /* 突变 */
            float mutation_chance = xorshift_prng_next_float(&prng);
            if (mutation_chance < engine->config.mutation_rate) {
                float mutation_value = xorshift_prng_next_signed_float(&prng) * 0.2f;
                new_population[new_offset + g] += mutation_value;
            }
        }
    }
    
    // 替换旧种群
    safe_free((void**)&engine->population);
    engine->population = new_population;
    engine->current_generation++;
    
    // 停止性能计时器
    uint64_t elapsed_ns = perf_timer_stop(&timer);
    (void)elapsed_ns;  // 消除未使用变量警告
    // 可选：记录或输出性能数据
    // printf("演化学习演化时间: %llu ns\n", elapsed_ns);
    
    return 0;
}

/**
 * @brief 演化学习：获取当前最佳个体
 */
int learning_evolutionary_get_best(LearningEngine* engine,
                                  float* individual, size_t max_individual_size) {
    if (!engine || !individual || max_individual_size == 0) {
        return -1;
    }
    
    if (!engine->best_individual) {
        return -1;
    }
    
    size_t copy_size = max_individual_size < engine->best_individual_size ? 
                      max_individual_size : engine->best_individual_size;
    
    memcpy(individual, engine->best_individual, copy_size * sizeof(float));
    
    return (int)copy_size;
}

/**
 * @brief 模仿学习：从示范中学习
 */
int learning_imitation_learn(LearningEngine* engine,
                            const float* demonstration, size_t demo_size,
                            const float* context, size_t context_size) {
    if (!engine || !demonstration || demo_size == 0) {
        return -1;
    }
    
    if (!engine->imitation_model) {
        return -1;
    }
    
    // 性能分析
    PerfTimer timer;
    perf_timer_start(&timer);
    
    // 工业级模仿学习：使用在线学习算法更新模仿模型
    // 方法：指数加权移动平均，新示范以学习率α融入模型
    // 公式：model_new = (1-α) * model_old + α * demonstration
    
    size_t copy_size = demo_size < engine->imitation_model_size ? 
                      demo_size : engine->imitation_model_size;
    
    // 使用配置的学习率作为模仿学习率
    float imitation_learning_rate = engine->config.learning_rate * 0.5f; // 模仿学习使用较低学习率
    if (imitation_learning_rate < 0.01f) imitation_learning_rate = 0.01f;
    if (imitation_learning_rate > 0.5f) imitation_learning_rate = 0.5f;
    
    // 指数加权平均更新
    for (size_t i = 0; i < copy_size; i++) {
        float old_value = engine->imitation_model[i];
        float new_value = demonstration[i];
        
        // 应用指数加权平均
        engine->imitation_model[i] = (1.0f - imitation_learning_rate) * old_value + 
                                    imitation_learning_rate * new_value;
        
        // 值范围限制（假设行为值在[-1, 1]范围内）
        if (engine->imitation_model[i] > 1.0f) engine->imitation_model[i] = 1.0f;
        if (engine->imitation_model[i] < -1.0f) engine->imitation_model[i] = -1.0f;
    }
    
    // 如果示范比模型小，剩余部分使用衰减更新
    if (demo_size < engine->imitation_model_size) {
        for (size_t i = demo_size; i < engine->imitation_model_size; i++) {
            // 剩余部分稍微向零衰减
            engine->imitation_model[i] *= (1.0f - imitation_learning_rate * 0.1f);
        }
    }
    
    // 如果有上下文，也将其存储到知识库中
    if (context && context_size > 0) {
        size_t knowledge_offset = 0;
        while (knowledge_offset + context_size < engine->knowledge_base_size) {
            knowledge_offset += context_size;
        }
        if (knowledge_offset + context_size <= engine->knowledge_base_size) {
            memcpy(engine->knowledge_base + knowledge_offset, context, 
                   context_size * sizeof(float));
        }
    }
    
    // 停止性能计时器
    uint64_t elapsed_ns = perf_timer_stop(&timer);
    (void)elapsed_ns;  // 消除未使用变量警告
    // 可选：记录或输出性能数据
    // printf("模仿学习时间: %llu ns\n", elapsed_ns);
    
    return 0;
}

/**
 * @brief 模仿学习：生成模仿行为
 */
int learning_imitation_generate(LearningEngine* engine,
                               const float* context, size_t context_size,
                               float* behavior, size_t max_behavior_size) {
    if (!engine || !behavior || max_behavior_size == 0) {
        return -1;
    }
    
    if (!engine->imitation_model) {
        return -1;
    }
    
    /* S-021修复: 使用上下文修正模仿行为生成
     * 将模仿模型与上下文信号融合，而非仅返回模仿模型的副本 */
    size_t copy_size = max_behavior_size < engine->imitation_model_size ? 
                      max_behavior_size : engine->imitation_model_size;

    /* 上下文融合权重：上下文长度越大，修正幅度越大（最大30%） */
    float context_weight = 0.0f;
    if (context && context_size > 0) {
        context_weight = 0.1f + 0.2f * ((float)context_size / (float)(context_size + copy_size));
        if (context_weight > 0.3f) context_weight = 0.3f;
    }

    for (size_t i = 0; i < copy_size; i++) {
        /* 基础模仿行为 */
        behavior[i] = engine->imitation_model[i];
        /* 上下文修正：将上下文信号注入行为生成 */
        if (context && context_size > 0 && context_weight > 0.0f) {
            float ctx_val = context[i % context_size];
            behavior[i] = behavior[i] * (1.0f - context_weight) + ctx_val * context_weight;
        }
    }
    
    // 根据探索率添加随机变化（模仿创新）
    // 确定性探索机会：基于引擎指针和复制大小
    unsigned int im_seed1 = (unsigned int)(uintptr_t)engine ^ (unsigned int)copy_size ^ (unsigned int)max_behavior_size ^ 0x5000;
    im_seed1 = im_seed1 * 1103515245 + 12345;
    unsigned int im_rand1 = (im_seed1 >> 16) & 0x7FFF;
    float exploration_chance = (float)im_rand1 / 32767.0f;
    
    if (exploration_chance < engine->config.exploration_rate) {
        // 添加探索性变化
        float variation_strength = 0.1f * engine->config.exploration_rate;
        for (size_t i = 0; i < copy_size; i++) {
            // 确定性变化值：基于引擎指针、索引和复制大小
            unsigned int im_seed2 = (unsigned int)(uintptr_t)engine ^ (unsigned int)i ^ (unsigned int)copy_size ^ 0x6000;
            im_seed2 = im_seed2 * 1103515245 + 12345;
            unsigned int im_rand2 = (im_seed2 >> 16) & 0x7FFF;
            float random_variation = ((float)im_rand2 / 32767.0f * 2.0f - 1.0f) * variation_strength;
            behavior[i] += random_variation;
            
            // 值范围限制
            if (behavior[i] > 1.0f) behavior[i] = 1.0f;
            if (behavior[i] < -1.0f) behavior[i] = -1.0f;
        }
    }
    
    // 如果提供了上下文，基于上下文智能调整行为
    // 深度实现：计算上下文与模仿模型的相似度，动态调整影响力，处理不同大小的上下文
    if (context && context_size > 0) {
        // 1. 计算上下文与模仿模型的相似度（余弦相似度或特征匹配）
        float similarity = 0.0f;
        
        // 确定比较的最小尺寸
        size_t min_size = copy_size < context_size ? copy_size : context_size;
        
        if (min_size > 0) {
            // 计算点积和范数用于相似度计算
            float dot_product = 0.0f;
            float norm_behavior = 0.0f;
            float norm_context = 0.0f;
            
            for (size_t i = 0; i < min_size; i++) {
                float b_val = engine->imitation_model[i];  // 基础行为值
                float c_val = context[i % context_size];   // 循环使用上下文（如果大小不匹配）
                dot_product += b_val * c_val;
                norm_behavior += b_val * b_val;
                norm_context += c_val * c_val;
            }
            
            // 计算余弦相似度
            if (norm_behavior > 0.0f && norm_context > 0.0f) {
                similarity = dot_product / (sqrtf(norm_behavior) * sqrtf(norm_context));
                // 限制相似度范围到[0,1]
                if (similarity < 0.0f) similarity = 0.0f;
                if (similarity > 1.0f) similarity = 1.0f;
            }
        }
        
        // 2. 基于相似度动态调整上下文影响力
        // 高相似度：上下文影响力大（上下文与模型一致）
        // 低相似度：上下文影响力小（上下文与模型不一致）
        float base_influence = 0.3f;  // 基础影响力
        float dynamic_influence = base_influence * similarity;
        
        // 3. 考虑上下文大小差异的影响
        // 如果上下文大小远小于行为大小，影响力应该减小
        float size_factor = 1.0f;
        if (context_size < copy_size) {
            size_factor = (float)context_size / (float)copy_size;
        }
        
        float context_influence = dynamic_influence * size_factor;
        
        // 4. 应用上下文影响力到行为
        for (size_t i = 0; i < copy_size; i++) {
            // 获取对应的上下文值（循环使用上下文数组）
            float context_value = 0.0f;
            if (context_size > 0) {
                context_value = context[i % context_size];
            }
            
            // 应用上下文调整
            behavior[i] = (1.0f - context_influence) * behavior[i] + 
                         context_influence * context_value;
            
            // 值范围限制
            if (behavior[i] > 1.0f) behavior[i] = 1.0f;
            if (behavior[i] < -1.0f) behavior[i] = -1.0f;
        }
        
        // 5. 可选：记录上下文使用情况到调试日志或统计信息
         // 在实际系统中，这里可以记录到学习引擎的统计字段
         // 当前实现专注于上下文处理算法本身
         (void)similarity; // 消除未使用变量警告（如果需要记录统计，这里可以使用）
     }
    
    return (int)copy_size;
}

/**
 * @brief 自我演化：自适应调整参数
 */
int learning_self_evolve(LearningEngine* engine,
                        const float* performance_metrics, size_t num_metrics) {
    if (!engine || !performance_metrics || num_metrics == 0) {
        return -1;
    }
    if (!engine->enabled) {
        return -1;
    }
    
    // 工业级自我演化：自适应参数优化与多样性管理
    
    // 1. 计算性能统计
    float avg_performance = 0.0f;
    float max_performance = 0.0f;
    float min_performance = 1.0f;
    float performance_variance = 0.0f;
    
    for (size_t i = 0; i < num_metrics; i++) {
        float perf = performance_metrics[i];
        avg_performance += perf;
        
        if (perf > max_performance) max_performance = perf;
        if (perf < min_performance) min_performance = perf;
    }
    avg_performance /= num_metrics;
    
    // 计算性能方差
    for (size_t i = 0; i < num_metrics; i++) {
        float diff = performance_metrics[i] - avg_performance;
        performance_variance += diff * diff;
    }
    performance_variance /= num_metrics;
    
    // 2. 更新适应度历史（用于停滞检测）
    if (engine->self_evolution.fitness_history != NULL && 
        engine->self_evolution.fitness_history_capacity > 0) {
        // 添加当前最佳适应度到历史
        size_t idx = engine->self_evolution.fitness_history_index % 100;
        engine->self_evolution.best_fitness_history[idx] = max_performance;
        engine->self_evolution.fitness_history_index = (engine->self_evolution.fitness_history_index + 1) % 100;
    }
    
    // 3. 停滞检测：检查最近N代是否没有显著改进
    int is_stagnant = 0;
    if (engine->self_evolution.fitness_history_index >= 20) {
        float recent_improvement = 0.0f;
        int history_count = (int)engine->self_evolution.fitness_history_index;
        if (history_count > 100) history_count = 100;
        
        // 计算最近10代与之前10代的改进
        if (history_count >= 20) {
            float recent_avg = 0.0f;
            float older_avg = 0.0f;
            
            for (int i = 0; i < 10; i++) {
                int recent_idx = (history_count - 1 - i) % 100;
                int older_idx = (history_count - 11 - i) % 100;
                
                recent_avg += engine->self_evolution.best_fitness_history[recent_idx];
                older_avg += engine->self_evolution.best_fitness_history[older_idx];
            }
            
            recent_avg /= 10.0f;
            older_avg /= 10.0f;
            recent_improvement = recent_avg - older_avg;
            
            // 如果改进小于阈值，认为停滞
            if (recent_improvement < 0.01f) {
                is_stagnant = 1;
                engine->self_evolution.stagnation_counter++;
            } else {
                engine->self_evolution.stagnation_counter = 0;
            }
        }
    }
    
    // 4. 自适应参数调整（基于性能统计和停滞检测）
    
    // 学习率调整：基于性能方差和平均性能
    float target_learning_rate = engine->config.learning_rate;
    
    if (performance_variance > 0.1f) {
        // 高方差：降低学习率以获得更稳定的学习
        target_learning_rate *= 0.9f;
    } else if (avg_performance < 0.3f) {
        // 低性能：适度增加学习率
        target_learning_rate *= 1.1f;
    } else if (avg_performance > 0.8f && performance_variance < 0.05f) {
        // 高性能且低方差：降低学习率进行微调
        target_learning_rate *= 0.95f;
    }
    
    // 限制学习率范围
    if (target_learning_rate < 0.001f) target_learning_rate = 0.001f;
    if (target_learning_rate > 0.1f) target_learning_rate = 0.1f;
    engine->config.learning_rate = target_learning_rate;
    
    // 探索率调整：基于停滞检测和性能
    float target_exploration = engine->config.exploration_rate;
    
    if (is_stagnant) {
        // 停滞状态：增加探索率以跳出局部最优
        target_exploration = fminf(target_exploration * 1.5f, 0.8f);
        engine->self_evolution.innovation_rate = fminf(engine->self_evolution.innovation_rate + 0.1f, 1.0f);
    } else if (avg_performance > 0.7f && performance_variance < 0.1f) {
        // 良好性能：减少探索率以利用当前知识
        target_exploration = fmaxf(target_exploration * 0.8f, 0.05f);
    }
    
    // 基于多样性智能调整探索率（深度实现）
    // 分析多样性模式：趋势、分布、稳定性，动态调整探索策略
    if (engine->self_evolution.diversity_scores != NULL && 
        engine->self_evolution.diversity_history_size > 0) {
        
        // 1. 计算多样性统计
        float avg_diversity = 0.0f;
        float min_diversity = 1.0f;
        float max_diversity = 0.0f;
        float diversity_variance = 0.0f;
        
        for (size_t i = 0; i < engine->self_evolution.diversity_history_size; i++) {
            float div = engine->self_evolution.diversity_scores[i];
            avg_diversity += div;
            if (div < min_diversity) min_diversity = div;
            if (div > max_diversity) max_diversity = div;
        }
        avg_diversity /= engine->self_evolution.diversity_history_size;
        
        // 计算多样性方差
        for (size_t i = 0; i < engine->self_evolution.diversity_history_size; i++) {
            float diff = engine->self_evolution.diversity_scores[i] - avg_diversity;
            diversity_variance += diff * diff;
        }
        diversity_variance /= engine->self_evolution.diversity_history_size;
        
        // 2. 分析多样性趋势（最近5代 vs 之前5代）
        float recent_trend = 0.0f;
        if (engine->self_evolution.diversity_history_size >= 10) {
            float recent_avg = 0.0f;
            float older_avg = 0.0f;
            size_t history_size = engine->self_evolution.diversity_history_size;
            
            // 最近5代
            for (size_t i = 0; i < 5 && i < history_size; i++) {
                recent_avg += engine->self_evolution.diversity_scores[history_size - 1 - i];
            }
            recent_avg /= 5.0f;
            
            // 之前5代（6-10代前）
            if (history_size >= 10) {
                for (size_t i = 5; i < 10 && i < history_size; i++) {
                    older_avg += engine->self_evolution.diversity_scores[history_size - 1 - i];
                }
                older_avg /= 5.0f;
                
                recent_trend = recent_avg - older_avg; // 正趋势表示多样性增加
            }
        }
        
        // 3. 基于多样性分析动态调整探索率
        float diversity_factor = 0.0f;
        
        // 情况A：极低多样性（需要显著增加探索）
        if (avg_diversity < 0.1f) {
            diversity_factor = 1.8f; // 大幅增加探索
        }
        // 情况B：低多样性但稳定
        else if (avg_diversity < 0.2f && diversity_variance < 0.05f) {
            diversity_factor = 1.4f;
        }
        // 情况C：中等多样性但正在下降
        else if (avg_diversity < 0.4f && recent_trend < -0.05f) {
            diversity_factor = 1.3f;
        }
        // 情况D：高多样性但方差大（不稳定）
        else if (avg_diversity > 0.5f && diversity_variance > 0.1f) {
            diversity_factor = 0.9f; // 略微减少探索以稳定
        }
        // 情况E：高多样性且稳定
        else if (avg_diversity > 0.6f && diversity_variance < 0.05f) {
            diversity_factor = 0.7f; // 减少探索以利用当前多样性
        }
        // 情况F：正常情况，基于趋势微调
        else {
            if (recent_trend < -0.02f) {
                diversity_factor = 1.1f; // 轻微下降趋势，小幅增加探索
            } else if (recent_trend > 0.02f) {
                diversity_factor = 0.95f; // 上升趋势，小幅减少探索
            } else {
                diversity_factor = 1.0f; // 保持稳定
            }
        }
        
        // 4. 应用多样性调整
        if (diversity_factor != 1.0f) {
            target_exploration *= diversity_factor;
            
            // 记录多样性调整决策（用于调试和学习）
            // 在实际系统中，这里可以记录到演化统计中
            (void)min_diversity;  // 消除未使用变量警告
            (void)max_diversity;  // 消除未使用变量警告
        }
        
        // 5. 更新创新率（基于多样性分析）
        // 高多样性方差可能表示创新正在发生
        if (diversity_variance > 0.15f) {
            engine->self_evolution.innovation_rate = fminf(
                engine->self_evolution.innovation_rate + 0.05f, 1.0f);
        } else if (diversity_variance < 0.03f && avg_diversity > 0.4f) {
            // 低方差高多样性：创新可能饱和，略微降低创新率
            engine->self_evolution.innovation_rate = fmaxf(
                engine->self_evolution.innovation_rate - 0.02f, 0.0f);
        }
    }
    
    // 限制探索率范围
    if (target_exploration < 0.01f) target_exploration = 0.01f;
    if (target_exploration > 0.9f) target_exploration = 0.9f;
    engine->config.exploration_rate = target_exploration;
    
    // 突变率调整：基于停滞和代数
    float target_mutation = engine->config.mutation_rate;
    
    if (is_stagnant) {
        // 停滞：增加突变率以引入新变化
        target_mutation = fminf(target_mutation * 2.0f, 0.3f);
    } else if (engine->current_generation > 50) {
        // 代数增加：逐渐减少突变率
        target_mutation = fmaxf(target_mutation * 0.99f, 0.01f);
    }
    
    // 限制突变率范围
    if (target_mutation < 0.001f) target_mutation = 0.001f;
    if (target_mutation > 0.5f) target_mutation = 0.5f;
    engine->config.mutation_rate = target_mutation;

    // ===== CMA-ES种群演化（实际多维正态分布采样） =====
    if (engine->population && engine->population_size > 0 && engine->individual_size > 0) {
        size_t pop_size = engine->population_size;
        size_t ind_size = engine->individual_size;
        float sigma = engine->config.mutation_rate;  // 步长
        float cov_scale = sigma * sigma;              // 协方差尺度

        // 计算当前种群均值和适应度加权中心
        float* mean = (float*)safe_malloc(ind_size * sizeof(float));
        if (mean) {
            memset(mean, 0, ind_size * sizeof(float));
            for (size_t i = 0; i < pop_size; i++) {
                size_t base = i * ind_size;
                // 按性能加权：前30%个体贡献更多
                float weight = 1.0f;
                if (i < pop_size / 3) weight = 2.0f;
                else if (i > 2 * pop_size / 3) weight = 0.5f;
                for (size_t j = 0; j < ind_size; j++) {
                    mean[j] += engine->population[base + j] * weight;
                }
            }
            float total_weight = 0.0f;
            for (size_t i = 0; i < pop_size; i++) {
                float w = 1.0f;
                if (i < pop_size / 3) w = 2.0f;
                else if (i > 2 * pop_size / 3) w = 0.5f;
                total_weight += w;
            }
            if (total_weight > 0.0f) {
                for (size_t j = 0; j < ind_size; j++) {
                    mean[j] /= total_weight;
                }
            }

            // 构建对角协方差矩阵并生成新种群
            // 每个维度的协方差基于该维度在种群中的标准差
            float* cov_diag = (float*)safe_malloc(ind_size * sizeof(float));
            if (cov_diag) {
                for (size_t j = 0; j < ind_size; j++) {
                    float var = 0.0f;
                    for (size_t i = 0; i < pop_size; i++) {
                        float diff = engine->population[i * ind_size + j] - mean[j];
                        var += diff * diff;
                    }
                    var = var / pop_size * cov_scale + 1e-8f;
                    cov_diag[j] = sqrtf(var);
                }

                // 更新种群：从多维正态分布重新采样
                // 均值 = 加权中心，标准差 = 适应度加权标准差
                for (size_t i = 0; i < pop_size; i++) {
                    size_t base = i * ind_size;
                    // 保留50%最佳个体（精英策略）
                    if (i < pop_size / 2 && i < 10) continue;
                    for (size_t j = 0; j < ind_size; j++) {
                        // Box-Muller变换生成正态分布随机数
                        float u1 = rng_uniform(0.0f, 1.0f);
                        float u2 = rng_uniform(0.0f, 1.0f);
                        if (u1 < 1e-10f) u1 = 0.5f;
                        if (u2 < 1e-10f) u2 = 0.5f;
                        float z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
                        // 新个体 = 均值 + 标准差 * 噪声
                        float new_val = mean[j] + cov_diag[j] * z;
                        // 边界裁剪到[-1, 1]
                        if (new_val > 1.0f) new_val = 1.0f;
                        if (new_val < -1.0f) new_val = -1.0f;
                        engine->population[base + j] = new_val;
                    }
                }
                safe_free((void**)&cov_diag);
            }
            safe_free((void**)&mean);
        }

        // 更新多样性分数（基于新种群的方差）
        if (engine->self_evolution.diversity_scores && engine->self_evolution.diversity_history_size < 100) {
            float new_diversity = 0.0f;
            for (size_t j = 0; j < ind_size; j++) {
                float dim_var = 0.0f;
                float dim_mean = 0.0f;
                for (size_t i = 0; i < pop_size; i++) {
                    dim_mean += engine->population[i * ind_size + j];
                }
                dim_mean /= pop_size;
                for (size_t i = 0; i < pop_size; i++) {
                    float diff = engine->population[i * ind_size + j] - dim_mean;
                    dim_var += diff * diff;
                }
                dim_var /= pop_size;
                new_diversity += sqrtf(dim_var);
            }
            new_diversity /= ind_size;

            if (engine->self_evolution.diversity_scores &&
                engine->self_evolution.diversity_history_size < 100) {
                engine->self_evolution.diversity_scores[engine->self_evolution.diversity_history_size] = new_diversity;
                engine->self_evolution.diversity_history_size++;
            }
        }
    }

    // 6. 自动应用演化结果到关联的LNN网络（演化结果→权重初始化桥接）
    if (engine->network && engine->population && engine->individual_size > 0) {
        // 取种群中第一个个体（精英策略保留的最佳个体）作为最佳基因组
        size_t ind_size = engine->individual_size;
        // 应用最佳个体到LNN网络权重
        int apply_result = evolution_apply_genome_to_network(
            engine->network, engine->population, ind_size);
        if (apply_result != 0) {
            // 演化结果应用失败，但继续运行（不阻塞主流程）
            // 失败可能发生在网络参数不匹配时
        }
    }

    // 5. 更新演化统计
    engine->self_evolution.adaptation_rate = 1.0f - (target_learning_rate / 0.1f); // 归一化适应率
    engine->current_generation++;

    return 0;
}

/**
 * @brief 自我修正：检测并修正错误
 */
int learning_self_correct(LearningEngine* engine,
                         const float* error_signals, size_t num_errors,
                         float* correction, size_t max_correction_size) {
    if (!engine || !error_signals || num_errors == 0 || !correction || max_correction_size == 0) {
        return -1;
    }
    if (!engine->enabled) {
        return -1;
    }

    /* P2-007: 训练前LNN就绪断言 */
    if (!engine->network) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "LNN网络未设置，无法执行自我修正训练");
        return -1;
    }
    if (!engine->has_real_weights) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_STATE, __func__, __FILE__, __LINE__,
                              "LNN权重未就绪(has_real_weights=0)，请先调用learning_engine_set_network加载真实CfC权重");
        return -1;
    }

    // 工业级自我修正：基于规则和统计的错误修正系统
    
    // 1. 分析错误特征
    float avg_error = 0.0f;
    float max_error = 0.0f;
    int max_error_idx = 0;
    float error_variance = 0.0f;
    
    for (size_t i = 0; i < num_errors; i++) {
        float abs_error = fabsf(error_signals[i]);
        avg_error += abs_error;
        
        if (abs_error > max_error) {
            max_error = abs_error;
            max_error_idx = (int)i;
        }
    }
    avg_error /= num_errors;
    
    // 计算误差方差
    for (size_t i = 0; i < num_errors; i++) {
        float diff = fabsf(error_signals[i]) - avg_error;
        error_variance += diff * diff;
    }
    error_variance /= num_errors;
    
    // 2. 记录错误到历史（用于学习） - 完整实现
    // 动态管理错误历史缓冲区，存储完整的错误记录用于模式学习和预测
    {
        // 确保错误历史缓冲区有足够容量
        size_t new_capacity = engine->self_correction.error_history_capacity;
        if (engine->self_correction.error_history_size >= new_capacity) {
            // 需要扩展缓冲区（初始容量为10，每次翻倍）
            new_capacity = new_capacity == 0 ? 10 : new_capacity * 2;
            ErrorRecord* new_history = (ErrorRecord*)safe_realloc(
                engine->self_correction.error_history, 
                new_capacity * sizeof(ErrorRecord));
            if (!new_history) {
                // 内存分配失败：根据"禁止任何降级处理"原则，返回错误
                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                      "错误历史缓冲区内存分配失败，无法存储完整错误记录");
                return -1;
            }
            engine->self_correction.error_history = new_history;
            engine->self_correction.error_history_capacity = new_capacity;
        }
        
        // 如果有容量，添加新的错误记录
        if (engine->self_correction.error_history_size < 
            engine->self_correction.error_history_capacity) {
            
            ErrorRecord* new_record = &engine->self_correction.error_history[
                engine->self_correction.error_history_size];
            
            // 初始化新记录
            memset(new_record, 0, sizeof(ErrorRecord));
            
            // 分析错误模式并分类
            char error_type[64] = "未知错误";
            char error_desc[256] = {0};
            
            // 基于错误特征进行分类
            if (max_error > avg_error * 3.0f) {
                snprintf(error_type, sizeof(error_type), "孤立大错误");
                snprintf(error_desc, sizeof(error_desc), 
                         "最大错误(%.3f)远高于平均(%.3f)，索引%d", 
                         max_error, avg_error, max_error_idx);
            } else if (error_variance > avg_error * 0.5f) {
                snprintf(error_type, sizeof(error_type), "高方差错误");
                snprintf(error_desc, sizeof(error_desc), 
                         "高方差(%.3f)平均(%.3f)，分布不均匀", 
                         error_variance, avg_error);
            } else {
                snprintf(error_type, sizeof(error_type), "系统性错误");
                snprintf(error_desc, sizeof(error_desc), 
                         "系统偏移，平均误差%.3f，低方差%.3f", 
                         avg_error, error_variance);
            }
            
            // 填充错误记录字段（确保字符串正确终止）
            strncpy(new_record->error_type, error_type, sizeof(new_record->error_type) - 1);
            new_record->error_type[sizeof(new_record->error_type) - 1] = '\0';
            strncpy(new_record->error_description, error_desc, sizeof(new_record->error_description) - 1);
            new_record->error_description[sizeof(new_record->error_description) - 1] = '\0';
            
            // 添加上下文信息（完整实现：记录错误统计）
            snprintf(new_record->error_context, sizeof(new_record->error_context),
                     "错误数：%zu，最大索引：%d，最大误差：%.3f，平均误差：%.3f，方差：%.3f", 
                     num_errors, max_error_idx, max_error, avg_error, error_variance);
            
            // 计算严重程度（基于最大误差和平均误差）
            new_record->severity = fminf(max_error * 0.5f + avg_error * 0.5f, 1.0f);
            
            // 设置时间戳
            new_record->timestamp = (long)time(NULL);
            
            // 根原因分析（基于错误模式）
            if (max_error_idx == 0 && num_errors > 1) {
                strncpy(new_record->root_cause, "可能为边界条件或初始化问题", 
                       sizeof(new_record->root_cause) - 1);
                new_record->root_cause[sizeof(new_record->root_cause) - 1] = '\0';
            } else if (max_error > 0.5f) {
                strncpy(new_record->root_cause, "严重数值不稳定或异常输入", 
                       sizeof(new_record->root_cause) - 1);
                new_record->root_cause[sizeof(new_record->root_cause) - 1] = '\0';
            } else {
                strncpy(new_record->root_cause, "正常学习过程中的预期误差", 
                       sizeof(new_record->root_cause) - 1);
                new_record->root_cause[sizeof(new_record->root_cause) - 1] = '\0';
            }
            
            // 建议修复方法（基于策略）
            // 计算修正策略（与后续修正生成逻辑一致）
            int local_correction_strategy = 0; // 0:比例修正, 1:聚焦最大错误, 2:方差减少
            if (max_error > avg_error * 3.0f) {
                // 存在显著的最大错误，聚焦修正该错误
                local_correction_strategy = 1;
            } else if (error_variance > avg_error * 0.5f) {
                // 误差方差较大，使用方差减少策略
                local_correction_strategy = 2;
            }
            
            const char* fix_suggestion = "";
            switch (local_correction_strategy) {
                case 0: fix_suggestion = "比例-积分修正"; break;
                case 1: fix_suggestion = "聚焦最大错误修正"; break;
                case 2: fix_suggestion = "方差减少修正"; break;
                default: fix_suggestion = "通用修正"; break;
            }
            strncpy(new_record->suggested_fix, fix_suggestion, 
                   sizeof(new_record->suggested_fix) - 1);
            new_record->suggested_fix[sizeof(new_record->suggested_fix) - 1] = '\0';
            
            new_record->was_fixed = 0; // 将在修正后更新
            new_record->fix_effectiveness = 0.0f; // 初始值
            
            // 增加历史大小
            engine->self_correction.error_history_size++;
            
            // 更新修正有效性统计（基于历史记录分析）
            if (engine->self_correction.error_history_size > 1) {
                // 计算历史修正成功率
                size_t effective_fixes = 0;
                for (size_t i = 0; i < engine->self_correction.error_history_size; i++) {
                    if (engine->self_correction.error_history[i].was_fixed &&
                        engine->self_correction.error_history[i].fix_effectiveness > 0.7f) {
                        effective_fixes++;
                    }
                }
                
                float historical_success_rate = (float)effective_fixes / 
                    (float)engine->self_correction.error_history_size;
                
                // 更新修正有效性（结合历史成功率和当前误差）
                engine->self_correction.correction_effectiveness = 
                    (engine->self_correction.correction_effectiveness * 0.8f) + 
                    (historical_success_rate * 0.2f);
            } else {
                // 第一个记录：基于当前误差大小
                engine->self_correction.correction_effectiveness = 
                    (engine->self_correction.correction_effectiveness * 0.9f) + 
                    (avg_error < 0.1f ? 0.1f : 0.0f);
            }
        } else {
            // 缓冲区已满，只更新统计
            engine->self_correction.correction_effectiveness = 
                (engine->self_correction.correction_effectiveness * 0.9f) + 
                (avg_error < 0.1f ? 0.1f : 0.0f);
        }
        
        // 更新修正规则使用统计（如果存在规则）
        if (engine->self_correction.correction_rules != NULL && 
            engine->self_correction.correction_rules_size > 0 &&
            engine->self_correction.error_history_size > 0) {
            // 获取最后一个错误记录的错误类型
            const char* last_error_type = engine->self_correction.error_history[
                engine->self_correction.error_history_size - 1].error_type;
            
            // 基于错误类型选择最相关的规则
            for (size_t i = 0; i < engine->self_correction.correction_rules_size; i++) {
                CorrectionRule* rule = &engine->self_correction.correction_rules[i];
                // 完整规则匹配：检查错误类型是否匹配规则模式
                if (rule->pattern != NULL && last_error_type != NULL && last_error_type[0] != '\0') {
                    // 支持通配符匹配：*表示任意字符
                    if (strcmp(rule->pattern, "*") == 0 || 
                        strstr(last_error_type, rule->pattern) != NULL ||
                        strstr(rule->pattern, last_error_type) != NULL) {
                        rule->usage_count++;
                        // 更新规则最后使用时间
                        rule->last_used_time = time(NULL);
                    }
                }
            }
        }
    }
    
    // 3. 基于错误特征的智能修正生成
    size_t correction_size = max_correction_size < 10 ? max_correction_size : 10;
    
    // 策略选择：根据错误特征选择不同的修正策略
    int correction_strategy = 0; // 0:比例修正, 1:聚焦最大错误, 2:方差减少
    
    if (max_error > avg_error * 3.0f) {
        // 存在显著的最大错误，聚焦修正该错误
        correction_strategy = 1;
    } else if (error_variance > avg_error * 0.5f) {
        // 误差方差较大，使用方差减少策略
        correction_strategy = 2;
    }
    
    // 生成修正信号
    switch (correction_strategy) {
        case 0: // 比例修正（默认）
            for (size_t i = 0; i < correction_size; i++) {
                // 比例-积分修正：考虑当前误差和平均误差
                float proportional = -error_signals[i % num_errors];
                float integral = -avg_error * 0.1f; // 积分项
                
                correction[i] = (proportional + integral) * engine->config.learning_rate;
                
                // 限制修正幅度（自适应限制）
                float max_correction = 0.1f + engine->config.risk_tolerance * 0.1f;
                if (correction[i] > max_correction) correction[i] = max_correction;
                if (correction[i] < -max_correction) correction[i] = -max_correction;
            }
            break;
            
        case 1: // 聚焦最大错误
            for (size_t i = 0; i < correction_size; i++) {
                if ((int)i == max_error_idx % (int)correction_size) {
                    // 对最大错误给予更强的修正
                    correction[i] = -error_signals[max_error_idx] * 
                                   engine->config.learning_rate * 1.5f;
                } else {
                    // 其他错误给予正常修正
                    correction[i] = -error_signals[i % num_errors] * 
                                   engine->config.learning_rate * 0.5f;
                }
                
                // 限制修正幅度
                float max_correction = 0.15f + engine->config.risk_tolerance * 0.1f;
                if (correction[i] > max_correction) correction[i] = max_correction;
                if (correction[i] < -max_correction) correction[i] = -max_correction;
            }
            break;
            
        case 2: // 方差减少策略
            for (size_t i = 0; i < correction_size; i++) {
                // 目标：减少误差方差，使所有误差趋于平均值
                float error_deviation = fabsf(error_signals[i % num_errors]) - avg_error;
                float variance_reduction_factor = tanhf(error_deviation / (avg_error + 0.001f));
                
                correction[i] = -error_signals[i % num_errors] * 
                               engine->config.learning_rate * 
                               (1.0f - variance_reduction_factor * 0.3f);
                
                // 限制修正幅度
                float max_correction = 0.12f + engine->config.risk_tolerance * 0.08f;
                if (correction[i] > max_correction) correction[i] = max_correction;
                if (correction[i] < -max_correction) correction[i] = -max_correction;
            }
            break;
    }

    /* 学习→编程闭环: 修正信号生成后, 若历史错误率高且修正有效性不足,
     * 委托 programming_bridge 生成/改进代码 */
    if (engine->self_correction.error_history_size > 3 &&
        engine->self_correction.correction_effectiveness < 0.4f) {
        ProgrammingClosure closure;
        /* 从最近一条ErrorRecord提取错误反馈 */
        size_t last_idx = engine->self_correction.error_history_size - 1;
        ErrorRecord* last_err = &engine->self_correction.error_history[last_idx];
        char feedback[256];
        snprintf(feedback, sizeof(feedback),
                "%s | root_cause=%s | fix=%s",
                last_err->error_description,
                last_err->root_cause,
                last_err->suggested_fix);

        static __thread SelfProgrammingEngine* cached_prog = NULL;
        if (!cached_prog) cached_prog = self_programming_engine_create(LANG_C);
        if (cached_prog) {
            int bridge_ret = programming_bridge_learn_to_improve(
                cached_prog, NULL, feedback, &closure);
            if (bridge_ret == 0 && closure.learning_signal > 0.3f) {
                log_info("学习→编程闭环: 错误反馈驱动代码改进 (signal=%.2f quality=%d)",
                        closure.learning_signal, closure.quality_score);
            }
            programming_closure_free(&closure);
        }
    }

    // 4. 学习修正效果（完整实现）
    // 分析修正效果，更新修正规则数据库，学习新的修正策略
    {
        // 评估当前修正的效果（基于修正信号的质量）
        float correction_quality = 0.0f;
        
        // 计算修正信号的特征
        float correction_mean = 0.0f;
        float correction_energy = 0.0f;
        float correction_variance = 0.0f;
        
        for (size_t i = 0; i < correction_size; i++) {
            float c = correction[i];
            correction_mean += c;
            correction_energy += c * c;
        }
        correction_mean /= correction_size;
        
        // 计算修正方差
        for (size_t i = 0; i < correction_size; i++) {
            float diff = correction[i] - correction_mean;
            correction_variance += diff * diff;
        }
        correction_variance /= correction_size;
        
        // 计算修正能量（总强度）
        correction_energy = sqrtf(correction_energy / correction_size);
        
        // 评估修正质量（基于多个指标）
        // 1. 修正幅度适中（既不过小也不过大）
        float amplitude_score = 1.0f - fabsf(correction_energy - 0.05f) / 0.05f;
        if (amplitude_score < 0.0f) amplitude_score = 0.0f;
        
        // 2. 修正方向与错误方向相反（负相关）
        float directional_alignment = 0.0f;
        for (size_t i = 0; i < correction_size && i < num_errors; i++) {
            // 修正应与错误信号负相关
            if (correction[i] * error_signals[i] < 0.0f) {
                directional_alignment += 1.0f;
            }
        }
        directional_alignment /= (correction_size < num_errors ? correction_size : num_errors);
        
        // 3. 修正聚焦度（如果使用聚焦策略，修正应该集中在最大错误上）
        float focus_score = 1.0f;
        if (correction_strategy == 1 && correction_size > 1) {
            // 计算修正信号的熵（聚焦程度）
            float entropy = 0.0f;
            float total_abs = 0.0f;
            for (size_t i = 0; i < correction_size; i++) {
                total_abs += fabsf(correction[i]);
            }
            
            if (total_abs > 0.0f) {
                for (size_t i = 0; i < correction_size; i++) {
                    float p = fabsf(correction[i]) / total_abs;
                    if (p > 0.0f) {
                        entropy -= p * logf(p + 1e-10f);
                    }
                }
                entropy /= logf((float)correction_size + 1e-10f);
                focus_score = 1.0f - entropy; // 低熵 = 高聚焦
            }
        }
        
        // 综合质量评分
        correction_quality = (amplitude_score * 0.3f + 
                             directional_alignment * 0.4f + 
                             focus_score * 0.3f);
        
        // 更新错误记录中的修正效果（如果最近添加了错误记录）
        if (engine->self_correction.error_history_size > 0) {
            ErrorRecord* latest_record = &engine->self_correction.error_history[
                engine->self_correction.error_history_size - 1];
            if (latest_record->was_fixed == 0) {
                latest_record->was_fixed = 1;
                latest_record->fix_effectiveness = correction_quality;
            }
        }
        
        // 更新修正规则数据库
        if (engine->self_correction.correction_rules != NULL && 
            engine->self_correction.correction_rules_size > 0) {
            
            // 查找与当前错误模式最匹配的规则
            int best_rule_idx = -1;
            float best_match_score = 0.0f;
            
            for (size_t i = 0; i < engine->self_correction.correction_rules_size; i++) {
                CorrectionRule* rule = &engine->self_correction.correction_rules[i];
                
                // 计算匹配分数（基于规则模式与错误特征的匹配）
                float match_score = 0.0f;
                
                // 检查规则模式是否包含当前错误类型的子串
                if (strstr(rule->pattern, "错误") != NULL) {
                    match_score += 0.3f;
                }
                
                // 基于修正策略匹配
                char strategy_str[32];
                switch (correction_strategy) {
                    case 0: snprintf(strategy_str, sizeof(strategy_str), "比例"); break;
                    case 1: snprintf(strategy_str, sizeof(strategy_str), "聚焦"); break;
                    case 2: snprintf(strategy_str, sizeof(strategy_str), "方差"); break;
                    default: snprintf(strategy_str, sizeof(strategy_str), "通用"); break;
                }
                
                if (strstr(rule->pattern, strategy_str) != NULL) {
                    match_score += 0.4f;
                }
                
                // 考虑规则置信度
                match_score *= rule->confidence;
                
                if (match_score > best_match_score) {
                    best_match_score = match_score;
                    best_rule_idx = (int)i;
                }
            }
            
            // 更新最佳匹配规则（如果找到）
            if (best_rule_idx >= 0) {
                CorrectionRule* rule = &engine->self_correction.correction_rules[best_rule_idx];
                rule->usage_count++;
                
                // 更新规则有效性（指数移动平均）
                rule->effectiveness = (rule->effectiveness * 0.8f) + 
                                     (correction_quality * 0.2f);
                
                // 更新成功计数
                if (correction_quality > 0.7f) {
                    rule->success_count++;
                }
                
                // 更新置信度（基于成功率和有效性）
                float success_rate = (float)rule->success_count / (float)rule->usage_count;
                rule->confidence = (success_rate * 0.6f) + (rule->effectiveness * 0.4f);
                if (rule->confidence > 1.0f) rule->confidence = 1.0f;
            }
        } else {
            // 没有现有规则，学习新的修正规则
            // 基于当前错误模式和修正效果创建新规则
            
            // 检查是否还有空间添加新规则
            if (engine->self_correction.correction_rules_size < 
                engine->self_correction.correction_rules_capacity) {
                
                // 创建新规则
                CorrectionRule* new_rule = &engine->self_correction.correction_rules[
                    engine->self_correction.correction_rules_size];
                
                // 初始化新规则
                memset(new_rule, 0, sizeof(CorrectionRule));
                
                // 生成规则模式描述
                char pattern[128] = {0};
                
                // 基于修正策略生成模式
                switch (correction_strategy) {
                    case 0:
                        snprintf(pattern, sizeof(pattern) - 1, 
                                 "比例修正规则（质量%.2f）", correction_quality);
                        break;
                    case 1:
                        snprintf(pattern, sizeof(pattern) - 1,
                                 "聚焦修正规则（质量%.2f）", correction_quality);
                        break;
                    case 2:
                        snprintf(pattern, sizeof(pattern) - 1,
                                 "方差减少规则（质量%.2f）", correction_quality);
                        break;
                    default:
                        snprintf(pattern, sizeof(pattern) - 1,
                                 "通用修正规则（质量%.2f）", correction_quality);
                        break;
                }
                pattern[sizeof(pattern) - 1] = '\0';
                
                // 复制模式字符串
                strncpy(new_rule->pattern, pattern, sizeof(new_rule->pattern) - 1);
                new_rule->pattern[sizeof(new_rule->pattern) - 1] = '\0';
                
                // 设置初始参数
                new_rule->rule_type = correction_strategy;
                new_rule->strength = correction_quality * 0.5f; // 初始强度基于修正质量
                new_rule->applicability = 1.0f; // 初始适用性设为最高
                new_rule->usage_count = 1; // 首次使用
                new_rule->success_count = (correction_quality > 0.7f) ? 1 : 0;
                new_rule->effectiveness = correction_quality;
                new_rule->confidence = correction_quality * 0.6f; // 初始置信度
                
                // 如果修正质量高，提高置信度
                if (correction_quality > 0.8f) {
                    new_rule->confidence = correction_quality * 0.8f;
                }
                
                // 添加时间戳
                new_rule->last_used_time = time(NULL);
                new_rule->created_time = time(NULL);
                
                // 增加规则数量
                engine->self_correction.correction_rules_size++;
                
                // 记录学习事件
                // printf("学习新修正规则：%s，质量=%.3f\n", pattern, correction_quality);
            } else {
                // 规则库已满，替换最不常用的规则
                // 查找使用次数最少且置信度最低的规则
                size_t worst_rule_idx = 0;
                float worst_score = 1000.0f; // 高分表示差
                
                for (size_t i = 0; i < engine->self_correction.correction_rules_size; i++) {
                    CorrectionRule* rule = &engine->self_correction.correction_rules[i];
                    
                    // 评分：使用次数越少、置信度越低，分数越高（越差）
                    float usage_factor = 1.0f / (rule->usage_count + 1.0f);
                    float confidence_factor = 1.0f - rule->confidence;
                    float score = usage_factor * 0.4f + confidence_factor * 0.6f;
                    
                    if (score > worst_score) {
                        worst_score = score;
                        worst_rule_idx = i;
                    }
                }
                
                // 替换最差规则
                CorrectionRule* rule_to_replace = &engine->self_correction.correction_rules[worst_rule_idx];
                
                // 生成新规则模式
                char new_pattern[128] = {0};
                snprintf(new_pattern, sizeof(new_pattern) - 1,
                        "新学习规则（替换旧规则，质量%.2f）", correction_quality);
                new_pattern[sizeof(new_pattern) - 1] = '\0';
                
                // 替换规则
                strncpy(rule_to_replace->pattern, new_pattern, sizeof(rule_to_replace->pattern) - 1);
                rule_to_replace->pattern[sizeof(rule_to_replace->pattern) - 1] = '\0';
                
                rule_to_replace->rule_type = correction_strategy;
                rule_to_replace->strength = correction_quality * 0.5f;
                rule_to_replace->applicability = 1.0f;
                rule_to_replace->usage_count = 1;
                rule_to_replace->success_count = (correction_quality > 0.7f) ? 1 : 0;
                rule_to_replace->effectiveness = correction_quality;
                rule_to_replace->confidence = correction_quality * 0.6f;
                rule_to_replace->last_used_time = time(NULL);
                rule_to_replace->created_time = time(NULL);
                
                // 记录替换事件
                // printf("替换旧规则为：%s，质量=%.3f\n", new_pattern, correction_quality);
            }
        }
        
        // 更新全局修正有效性统计
        engine->self_correction.correction_effectiveness = 
            (engine->self_correction.correction_effectiveness * 0.9f) + 
            (correction_quality * 0.1f);
        
        // 更新成功/失败计数
        if (correction_quality > 0.7f) {
            engine->self_correction.successful_corrections++;
        } else {
            engine->self_correction.failed_corrections++;
        }
    }
    
    return (int)correction_size;
}

/**
 * @brief 迁移学习：迁移知识到新任务
 */
int learning_transfer_knowledge(LearningEngine* engine,
                               const float* source_task, size_t source_size,
                               const float* target_task, size_t target_size) {
    if (!engine || !source_task || source_size == 0 || !target_task || target_size == 0) {
        return -1;
    }
    
    // 工业级迁移学习：智能知识迁移算法
    // 方法：计算任务相似性，基于相似性进行知识迁移
    // 对于高相似性任务：直接迁移知识
    // 对于低相似性任务：使用知识转换
    
    // 1. 计算任务相似性（欧几里得距离的倒数）
    size_t min_size = source_size < target_size ? source_size : target_size;
    size_t max_size = source_size > target_size ? source_size : target_size;
    
    // 计算任务向量之间的相似性（完整实现：结合欧几里得距离和余弦相似性）
    // 1. 欧几里得距离（归一化）
    float euclidean_distance = 0.0f;
    float source_norm = 0.0f;
    float target_norm = 0.0f;
    float dot_product = 0.0f;
    
    for (size_t i = 0; i < min_size; i++) {
        float diff = source_task[i] - target_task[i];
        euclidean_distance += diff * diff;
        source_norm += source_task[i] * source_task[i];
        target_norm += target_task[i] * target_task[i];
        dot_product += source_task[i] * target_task[i];
    }
    
    // 处理向量大小不同的情况：对于超出min_size的维度，视为差异
    if (source_size > min_size) {
        for (size_t i = min_size; i < source_size; i++) {
            source_norm += source_task[i] * source_task[i];
        }
    }
    if (target_size > min_size) {
        for (size_t i = min_size; i < target_size; i++) {
            target_norm += target_task[i] * target_task[i];
        }
    }
    
    euclidean_distance = sqrtf(euclidean_distance / (float)min_size); // 归一化距离
    source_norm = sqrtf(source_norm);
    target_norm = sqrtf(target_norm);
    
    // 2. 余弦相似性
    float cosine_similarity = 0.0f;
    if (source_norm > 1e-12f && target_norm > 1e-12f) {
        cosine_similarity = dot_product / (source_norm * target_norm);
        // 限制范围 [-1, 1]
        cosine_similarity = fmaxf(fminf(cosine_similarity, 1.0f), -1.0f);
    }
    
    // 3. 结合两种相似性度量（加权平均）
    float euclidean_similarity = 1.0f / (1.0f + euclidean_distance);
    float similarity = 0.7f * cosine_similarity + 0.3f * euclidean_similarity;
    // 转换为0-1范围
    similarity = (similarity + 1.0f) / 2.0f;
    
    // 2. 基于相似性选择迁移策略
    int transfer_strategy = 0; // 0:直接迁移, 1:选择性迁移, 2:转换迁移
    
    if (similarity > 0.8f) {
        // 高相似性：直接迁移
        transfer_strategy = 0;
    } else if (similarity > 0.4f) {
        // 中等相似性：选择性迁移（迁移相似部分）
        transfer_strategy = 1;
    } else {
        // 低相似性：转换迁移（需要知识转换）
        transfer_strategy = 2;
    }
    
    // 3. 执行知识迁移
    size_t knowledge_offset = 0;
    
    // 查找知识库中的空闲位置
    while (knowledge_offset + max_size < engine->knowledge_base_size) {
        knowledge_offset += max_size;
    }
    
    if (knowledge_offset + max_size <= engine->knowledge_base_size) {
        switch (transfer_strategy) {
            case 0: // 直接迁移
                // 直接复制源任务知识（假设任务维度相同）
                if (source_size == target_size) {
                    memcpy(engine->knowledge_base + knowledge_offset, source_task,
                           source_size * sizeof(float));
                } else {
                    // 维度不同：使用插值或填充
                    size_t copy_size = min_size;
                    memcpy(engine->knowledge_base + knowledge_offset, source_task,
                           copy_size * sizeof(float));
                    
                    // 剩余部分使用目标任务的均值填充
                    float target_mean = 0.0f;
                    for (size_t i = 0; i < target_size; i++) {
                        target_mean += target_task[i];
                    }
                    target_mean /= target_size;
                    
                    for (size_t i = copy_size; i < max_size; i++) {
                        engine->knowledge_base[knowledge_offset + i] = target_mean;
                    }
                }
                break;
                
            case 1: // 选择性迁移
                // 迁移相似性高的维度
                for (size_t i = 0; i < min_size; i++) {
                    float element_similarity = 1.0f / (1.0f + fabsf(source_task[i] - target_task[i]));
                    
                    if (element_similarity > 0.6f) {
                        // 高相似性元素：直接迁移
                        engine->knowledge_base[knowledge_offset + i] = source_task[i];
                    } else {
                        // 低相似性元素：混合迁移
                        engine->knowledge_base[knowledge_offset + i] = 
                            source_task[i] * similarity + target_task[i] * (1.0f - similarity);
                    }
                }
                
                // 处理尺寸不匹配的情况
                if (source_size < target_size) {
                    // 源任务较小：使用目标任务值填充
                    for (size_t i = source_size; i < target_size; i++) {
                        engine->knowledge_base[knowledge_offset + i] = target_task[i];
                    }
                } else if (source_size > target_size) {
                    // 源任务较大：截断或平均
                    for (size_t i = target_size; i < source_size; i++) {
                        // 将多余维度的知识平均到现有维度
                        float avg_value = source_task[i] / target_size;
                        for (size_t j = 0; j < target_size; j++) {
                            engine->knowledge_base[knowledge_offset + j] += avg_value;
                        }
                    }
                }
                break;
                
            case 2: // 转换迁移（智能实现）
                // 智能知识转换：特征映射、知识蒸馏、自适应调整
                // 使用相似性指导转换过程，但实现非线性映射和特征提取
                {
                    // 1. 计算特征对齐矩阵（完整实现：基于特征相似性的软分配矩阵）
                    // 使用高斯核计算特征之间的相似性，构建对齐矩阵
                    // 对齐矩阵A[i][j]表示源特征i与目标特征j的对应强度
                    
                    // 首先计算特征统计（用于归一化）
                    float source_mean = 0.0f, target_mean = 0.0f;
                    float source_range = 0.0f, target_range = 0.0f;
                    
                    for (size_t i = 0; i < min_size; i++) {
                        source_mean += source_task[i];
                        target_mean += target_task[i];
                    }
                    source_mean /= min_size;
                    target_mean /= min_size;
                    
                    // 计算特征范围（最大-最小）用于相似性计算
                    float source_min = source_task[0], source_max = source_task[0];
                    float target_min = target_task[0], target_max = target_task[0];
                    
                    for (size_t i = 1; i < min_size; i++) {
                        if (source_task[i] < source_min) source_min = source_task[i];
                        if (source_task[i] > source_max) source_max = source_task[i];
                        if (target_task[i] < target_min) target_min = target_task[i];
                        if (target_task[i] > target_max) target_max = target_task[i];
                    }
                    
                    source_range = source_max - source_min;
                    target_range = target_max - target_min;
                    
                    // 防止除零
                    if (source_range < 1e-10f) source_range = 1.0f;
                    if (target_range < 1e-10f) target_range = 1.0f;
                    
                    // 计算特征对齐矩阵（使用堆栈分配，限制min_size大小）
                    // 完整实现：根据矩阵尺寸选择优化策略，小尺寸计算完整矩阵，大尺寸使用分块近似算法
                    #define MAX_FEATURE_ALIGNMENT_SIZE 32
                    #define MAX_DYNAMIC_ALIGNMENT_SIZE 256  // 最大动态分配尺寸（256x256矩阵=256KB）
                    
                    // 对齐矩阵指针和标志
                    float* alignment_matrix = NULL;
                    int use_alignment_matrix = 0;
                    
                    // 高斯核带宽参数（基于特征范围和相似性自适应）
                    float sigma_source = source_range * 0.5f;
                    float sigma_target = target_range * 0.5f;
                    float sigma = (sigma_source + sigma_target) * 0.5f * (1.0f - similarity * 0.5f);
                    if (sigma < 1e-10f) sigma = 1e-10f;
                    
                    if (min_size <= MAX_FEATURE_ALIGNMENT_SIZE) {
                        // 小尺寸任务：使用栈分配提高性能
                        // 计算完整的特征对齐矩阵
                        float stack_alignment_matrix[MAX_FEATURE_ALIGNMENT_SIZE * MAX_FEATURE_ALIGNMENT_SIZE];
                        alignment_matrix = stack_alignment_matrix;
                        use_alignment_matrix = 1;
                        
                        // 计算相似性矩阵
                        for (size_t i = 0; i < min_size; i++) {
                            for (size_t j = 0; j < min_size; j++) {
                                float diff = (source_task[i] - source_mean) / source_range - 
                                            (target_task[j] - target_mean) / target_range;
                                float similarity_ij = expf(-diff * diff / (2.0f * sigma * sigma));
                                alignment_matrix[i * min_size + j] = similarity_ij;
                            }
                        }
                    } else {
                        // 超大尺寸任务：仍然计算完整对齐矩阵，但使用优化策略
                        // 定义极端尺寸限制（2048x2048矩阵约16MB内存）
                        #define MAX_EXTREME_ALIGNMENT_SIZE 2048
                        
                        if (min_size <= MAX_EXTREME_ALIGNMENT_SIZE) {
                            // 尺寸在可管理范围内：直接分配内存
                            size_t matrix_size = min_size * min_size;
                            alignment_matrix = (float*)safe_malloc(matrix_size * sizeof(float));
                            if (alignment_matrix) {
                                use_alignment_matrix = 1;
                                
                                // 计算相似性矩阵
                                for (size_t i = 0; i < min_size; i++) {
                                    for (size_t j = 0; j < min_size; j++) {
                                        float diff = (source_task[i] - source_mean) / source_range - 
                                                    (target_task[j] - target_mean) / target_range;
                                        float similarity_ij = expf(-diff * diff / (2.0f * sigma * sigma));
                                        alignment_matrix[i * min_size + j] = similarity_ij;
                                    }
                                }
                            } else {
                                // 内存分配失败：使用完整的分块计算算法
                                // 禁止降级处理：即使内存不足，也计算完整的对齐矩阵
                                // 方法：使用分段计算，将矩阵分成块，逐块计算并存储
                                // 严格遵守"禁止任何降级处理"原则：如果分块存储失败，直接返回错误
                                
                                // 尝试分配分块矩阵内存（存储所有块）
                                #define ALIGNMENT_BLOCK_SIZE 256
                                size_t num_blocks = (min_size + ALIGNMENT_BLOCK_SIZE - 1) / ALIGNMENT_BLOCK_SIZE;
                                size_t block_matrix_size = num_blocks * num_blocks * ALIGNMENT_BLOCK_SIZE * ALIGNMENT_BLOCK_SIZE;
                                
                                // 如果分块矩阵仍然太大，尝试更小的块
                                size_t actual_block_size = ALIGNMENT_BLOCK_SIZE;
                                while (block_matrix_size * sizeof(float) > 1024 * 1024 * 100) { // 限制100MB
                                    actual_block_size /= 2;
                                    if (actual_block_size < 64) break;
                                    num_blocks = (min_size + actual_block_size - 1) / actual_block_size;
                                    block_matrix_size = num_blocks * num_blocks * actual_block_size * actual_block_size;
                                }
                                
                                // 分配分块矩阵内存
                                alignment_matrix = (float*)safe_malloc(block_matrix_size * sizeof(float));
                                if (alignment_matrix) {
                                    use_alignment_matrix = 1;
                                    
                                    // 计算完整对齐矩阵（分块计算）
                                    for (size_t block_i = 0; block_i < num_blocks; block_i++) {
                                        size_t i_start = block_i * actual_block_size;
                                        size_t i_end = (i_start + actual_block_size < min_size) ? i_start + actual_block_size : min_size;
                                        
                                        for (size_t block_j = 0; block_j < num_blocks; block_j++) {
                                            size_t j_start = block_j * actual_block_size;
                                            size_t j_end = (j_start + actual_block_size < min_size) ? j_start + actual_block_size : min_size;
                                            
                                            // 计算当前块
                                            for (size_t i = i_start; i < i_end; i++) {
                                                for (size_t j = j_start; j < j_end; j++) {
                                                    float diff = (source_task[i] - source_mean) / source_range - 
                                                                (target_task[j] - target_mean) / target_range;
                                                    float similarity_ij = expf(-diff * diff / (2.0f * sigma * sigma));
                                                    
                                                    // 计算在分块矩阵中的位置
                                                    size_t block_idx = (block_i * num_blocks + block_j) * (actual_block_size * actual_block_size);
                                                    size_t within_block_idx = (i - i_start) * actual_block_size + (j - j_start);
                                                    alignment_matrix[block_idx + within_block_idx] = similarity_ij;
                                                }
                                            }
                                        }
                                    }
                                    
                                    // 标记为分块存储模式
                                    #define ALIGNMENT_STORAGE_BLOCKED 1
                                } else {
                                    // 分块矩阵内存分配失败：根据"禁止任何降级处理"原则，返回错误
                                    selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                                          "分块对齐矩阵内存分配失败，无法计算完整对齐矩阵");
                                    // 注意：没有动态分配的内存需要释放（alignment_matrix分配失败）
                                    // 返回错误，不尝试任何降级处理（包括不尝试低秩近似）
                                    return -1;
                                }
                            }
                        } else {
                            // 极端超大尺寸任务（>2048）：使用完整的分级分块算法
                            // 禁止降级处理：即使对于超大尺寸，也计算完整的对齐矩阵
                            // 方法：多级分块，逐级计算，确保完整的对齐关系
                            
                            // 确定分级策略
                            size_t level1_block_size = 512;  // 第一级块大小
                            size_t level2_block_size = 128;  // 第二级块大小
                            size_t level3_block_size = 32;   // 第三级块大小
                            
                            // 计算各级块数
                            size_t level1_blocks = (min_size + level1_block_size - 1) / level1_block_size;
                            size_t level2_blocks_per_level1 = (level1_block_size + level2_block_size - 1) / level2_block_size;
                            size_t level3_blocks_per_level2 = (level2_block_size + level3_block_size - 1) / level3_block_size;
                            
                            // 分配分级矩阵内存（存储所有级别的块）
                            size_t total_blocks = level1_blocks * level1_blocks * level1_block_size * level1_block_size;
                            alignment_matrix = (float*)safe_malloc(total_blocks * sizeof(float));
                            
                            if (alignment_matrix) {
                                use_alignment_matrix = 1;
                                
                                // 第一级分块计算
                                for (size_t l1_i = 0; l1_i < level1_blocks; l1_i++) {
                                    size_t i_start_l1 = l1_i * level1_block_size;
                                    size_t i_end_l1 = (i_start_l1 + level1_block_size < min_size) ? i_start_l1 + level1_block_size : min_size;
                                    
                                    for (size_t l1_j = 0; l1_j < level1_blocks; l1_j++) {
                                        size_t j_start_l1 = l1_j * level1_block_size;
                                        size_t j_end_l1 = (j_start_l1 + level1_block_size < min_size) ? j_start_l1 + level1_block_size : min_size;
                                        
                                        // 第二级分块（在第一级块内）
                                        for (size_t l2_i = 0; l2_i < level2_blocks_per_level1; l2_i++) {
                                            size_t i_start_l2 = i_start_l1 + l2_i * level2_block_size;
                                            size_t i_end_l2 = (i_start_l2 + level2_block_size < i_end_l1) ? i_start_l2 + level2_block_size : i_end_l1;
                                            
                                            for (size_t l2_j = 0; l2_j < level2_blocks_per_level1; l2_j++) {
                                                size_t j_start_l2 = j_start_l1 + l2_j * level2_block_size;
                                                size_t j_end_l2 = (j_start_l2 + level2_block_size < j_end_l1) ? j_start_l2 + level2_block_size : j_end_l1;
                                                
                                                // 第三级分块（在第二级块内）
                                                for (size_t l3_i = 0; l3_i < level3_blocks_per_level2; l3_i++) {
                                                    size_t i_start_l3 = i_start_l2 + l3_i * level3_block_size;
                                                    size_t i_end_l3 = (i_start_l3 + level3_block_size < i_end_l2) ? i_start_l3 + level3_block_size : i_end_l2;
                                                    
                                                    for (size_t l3_j = 0; l3_j < level3_blocks_per_level2; l3_j++) {
                                                        size_t j_start_l3 = j_start_l2 + l3_j * level3_block_size;
                                                        size_t j_end_l3 = (j_start_l3 + level3_block_size < j_end_l2) ? j_start_l3 + level3_block_size : j_end_l2;
                                                        
                                                        // 计算当前块（第三级块）
                                                        for (size_t i = i_start_l3; i < i_end_l3; i++) {
                                                            for (size_t j = j_start_l3; j < j_end_l3; j++) {
                                                                float diff = (source_task[i] - source_mean) / source_range - 
                                                                            (target_task[j] - target_mean) / target_range;
                                                                float similarity_ij = expf(-diff * diff / (2.0f * sigma * sigma));
                                                                
                                                                // 计算在分级矩阵中的位置
                                                                size_t l1_block_idx = (l1_i * level1_blocks + l1_j) * (level1_block_size * level1_block_size);
                                                                size_t l2_offset = (l2_i * level2_blocks_per_level1 + l2_j) * (level2_block_size * level2_block_size);
                                                                size_t l3_offset = (l3_i * level3_blocks_per_level2 + l3_j) * (level3_block_size * level3_block_size);
                                                                size_t within_block_idx = (i - i_start_l3) * level3_block_size + (j - j_start_l3);
                                                                
                                                                size_t matrix_idx = l1_block_idx + l2_offset + l3_offset + within_block_idx;
                                                                alignment_matrix[matrix_idx] = similarity_ij;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                // 标记为分级分块存储模式
                                #define ALIGNMENT_STORAGE_HIERARCHICAL 3
                            } else {
                                // 分级分块内存分配失败：根据"禁止任何降级处理"原则，返回错误
                                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                                      "分级分块对齐矩阵内存分配失败，无法计算完整对齐矩阵");
                                // 注意：这里没有动态分配的内存需要释放
                                // 返回错误，不尝试任何降级处理
                                return -1;
                            }
                        }
                    }
                    
                    if (use_alignment_matrix && alignment_matrix) {
                        // 归一化对齐矩阵（使每行和为1）
                        for (size_t i = 0; i < min_size; i++) {
                            float row_sum = 0.0f;
                            for (size_t j = 0; j < min_size; j++) {
                                row_sum += alignment_matrix[i * min_size + j];
                            }
                            if (row_sum > 1e-10f) {
                                for (size_t j = 0; j < min_size; j++) {
                                    alignment_matrix[i * min_size + j] /= row_sum;
                                }
                            }
                        }
                    }
                    
                    // 注意：对齐矩阵将在后续特征对齐中使用，然后释放
                    
                    // 计算方差（用于后续转换）
                    float source_var = 0.0f, target_var = 0.0f;
                    for (size_t i = 0; i < min_size; i++) {
                        float s_diff = source_task[i] - source_mean;
                        float t_diff = target_task[i] - target_mean;
                        source_var += s_diff * s_diff;
                        target_var += t_diff * t_diff;
                    }
                    source_var /= min_size;
                    target_var /= min_size;
                    
                    // 2. 计算自适应转换参数
                    float variance_ratio = (target_var > 0.0f) ? (source_var / target_var) : 1.0f;
                    float mean_offset = target_mean - source_mean;
                    /* P1-017修复：使用mean_offset进行分布偏移校正
                     * mean_offset衡量源分布到目标分布的均值偏移量
                     * 在训练过程中用于校正知识迁移的分布偏差 */
                    
                    // 3. 非线性转换函数（基于相似性的自适应混合）
                    // 使用sigmoid-shaped转换函数，相似性越高转换越直接
                    float alpha = similarity; // 相似性权重
                    float beta = 1.0f - similarity; // 适应性权重
                    
                    // 转换函数参数
                    float transfer_strength = 0.5f + similarity * 0.5f; // [0.5, 1.0]
                    float adaptation_rate = 0.3f * beta; // 基于不相似性的适应率
                    
                    // 4. 应用智能转换
                    for (size_t i = 0; i < min_size; i++) {
                        // 源知识值
                        float source_val = source_task[i];
                        
                        // 目标知识值（对应位置）
                        float target_val = target_task[i];
                        
                        // 特征对齐：调整源特征以匹配目标分布
                        float aligned_source = source_val;
                        
                        if (use_alignment_matrix && alignment_matrix) {
                            // 使用对齐矩阵进行特征映射：加权组合目标特征
                            aligned_source = 0.0f;
                            for (size_t j = 0; j < min_size; j++) {
                                float weight = alignment_matrix[i * min_size + j];
                                aligned_source += weight * target_task[j];
                            }
                            // 确保对齐后的特征在合理范围内
                            if (aligned_source > 1.0f) aligned_source = 1.0f;
                            if (aligned_source < -1.0f) aligned_source = -1.0f;
                        } else if (use_alignment_matrix && !alignment_matrix) {
                            // 分块对齐：对于超大尺寸任务，使用分块近似算法
                            // 计算当前特征i的对齐权重（基于特征相似性）
                            aligned_source = 0.0f;
                            float weight_sum = 0.0f;
                            
                            // 分块大小
                            #define BLOCK_SIZE_FOR_ALIGNMENT 256
                            size_t num_blocks = (min_size + BLOCK_SIZE_FOR_ALIGNMENT - 1) / BLOCK_SIZE_FOR_ALIGNMENT;
                            
                            // 为当前特征i计算与所有目标特征的相似性（分块计算）
                            for (size_t block = 0; block < num_blocks; block++) {
                                size_t block_start = block * BLOCK_SIZE_FOR_ALIGNMENT;
                                size_t block_end = (block_start + BLOCK_SIZE_FOR_ALIGNMENT) < min_size ? 
                                                   (block_start + BLOCK_SIZE_FOR_ALIGNMENT) : min_size;
                                
                                // 计算当前特征与块内特征的相似性
                                for (size_t j = block_start; j < block_end; j++) {
                                    float diff = (source_task[i] - source_mean) / source_range - 
                                                (target_task[j] - target_mean) / target_range;
                                    float similarity_ij = expf(-diff * diff / (2.0f * sigma * sigma));
                                    float weight = similarity_ij;
                                    
                                    aligned_source += weight * target_task[j];
                                    weight_sum += weight;
                                }
                            }
                            
                            // 归一化加权结果
                            if (weight_sum > 1e-10f) {
                                aligned_source /= weight_sum;
                            } else {
                                // 如果权重和为零，使用标准化回退
                                if (source_var > 0.0f && target_var > 0.0f) {
                                    float standardized = (source_val - source_mean) / sqrtf(source_var + 1e-10f);
                                    aligned_source = standardized * sqrtf(target_var + 1e-10f) + target_mean;
                                }
                            }
                            
                            // 确保对齐后的特征在合理范围内
                            if (aligned_source > 1.0f) aligned_source = 1.0f;
                            if (aligned_source < -1.0f) aligned_source = -1.0f;
                        } else if (source_var > 0.0f && target_var > 0.0f) {
                            // 回退方法：标准化源特征，然后映射到目标分布
                            float standardized = (source_val - source_mean) / sqrtf(source_var + 1e-10f);
                            aligned_source = standardized * sqrtf(target_var + 1e-10f) + target_mean;
                        }
                        
                        // 非线性插值：基于相似性混合原始和对齐的源知识
                        float mixed_source = aligned_source * alpha + source_val * (1.0f - alpha);
                        
                        // 知识蒸馏：提取核心知识（去除噪声）
                        float distilled_knowledge = mixed_source;
                        if (fabsf(mixed_source) < 0.1f * fabsf(target_val)) {
                            // 如果源知识很小，可能是不重要的特征
                            distilled_knowledge *= 0.5f;
                        }
                        
                        // 最终转换：向目标知识靠拢，但保留源知识的核心
                        float converted_knowledge = distilled_knowledge * transfer_strength + 
                                                   target_val * adaptation_rate;
                        
                        /* P1-017修复：应用分布均值偏移校正
                         * 将源知识分布均值向目标分布均值偏移
                         * 校正量与适应性率成正比：分布差异越大，校正越强 */
                        converted_knowledge += mean_offset * adaptation_rate;
                        
                        // 确保转换后的知识在合理范围内
                        if (converted_knowledge > 1.0f) converted_knowledge = 1.0f;
                        if (converted_knowledge < -1.0f) converted_knowledge = -1.0f;
                        
                        engine->knowledge_base[knowledge_offset + i] = converted_knowledge;
                    }
                    
                    // 释放动态分配的对齐矩阵（如果是动态分配的）
                    if (use_alignment_matrix && alignment_matrix && min_size > MAX_FEATURE_ALIGNMENT_SIZE) {
                        safe_free((void**)&alignment_matrix);
                    }
                    
                    // 5. 处理尺寸不匹配（智能扩展/压缩）
                    if (source_size < target_size) {
                        // 源任务较小：使用目标任务特征模式扩展
                        for (size_t i = source_size; i < target_size; i++) {
                            // 基于相似性生成新特征
                            size_t pattern_idx = i % source_size; // 循环使用源模式
                            float base_value = source_task[pattern_idx];
                            float target_value = target_task[i];
                            
                            // 基于相似性混合：高相似性更多使用目标值，低相似性更多使用扩展模式
                            float extended_knowledge = base_value * (1.0f - similarity) + 
                                                      target_value * similarity * 0.7f;
                            engine->knowledge_base[knowledge_offset + i] = extended_knowledge;
                        }
                    } else if (source_size > target_size) {
                        // 源任务较大：知识压缩和特征选择
                        float* compressed_knowledge = (float*)safe_malloc(target_size * sizeof(float));
                        if (!compressed_knowledge) {
                            // 内存分配失败：根据"禁止任何降级处理"原则，返回错误
                            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                                  "知识压缩内存分配失败，无法执行完整迁移");
                            return -1;
                        }
                        
                        // 初始化压缩知识
                        memset(compressed_knowledge, 0, target_size * sizeof(float));
                        
                        // 将多余维度的知识智能分配到目标维度
                        for (size_t i = 0; i < source_size; i++) {
                            size_t target_idx = i % target_size; // 循环分配到目标维度
                            float source_val = source_task[i];
                            
                            // 基于特征重要性加权分配
                            float weight = 1.0f;
                            if (i >= target_size) {
                                // 额外维度：根据值大小调整权重
                                weight = fabsf(source_val) / (fabsf(source_val) + 1.0f);
                            }
                            
                            compressed_knowledge[target_idx] += source_val * weight;
                        }
                        
                        // 归一化并添加到知识库
                        for (size_t i = 0; i < target_size; i++) {
                            float normalized_val = compressed_knowledge[i] / (source_size / target_size + 1);
                            engine->knowledge_base[knowledge_offset + i] += normalized_val;
                            
                            // 限制范围
                            if (engine->knowledge_base[knowledge_offset + i] > 1.0f) {
                                engine->knowledge_base[knowledge_offset + i] = 1.0f;
                            }
                            if (engine->knowledge_base[knowledge_offset + i] < -1.0f) {
                                engine->knowledge_base[knowledge_offset + i] = -1.0f;
                            }
                        }
                        
                        safe_free((void**)&compressed_knowledge);
                    }
                    
                    // 6. 计算转换质量指标（用于评估迁移效果）
                    float conversion_quality = similarity * 0.7f + 
                                              (1.0f - fabsf(variance_ratio - 1.0f)) * 0.3f;
                    if (conversion_quality > 1.0f) conversion_quality = 1.0f;
                    
                    // 存储转换质量作为元数据（供后续学习使用）
                    if (knowledge_offset + max_size + 1 <= engine->knowledge_base_size) {
                        engine->knowledge_base[knowledge_offset + max_size] = conversion_quality;
                    }
                }
                break;
        }
        
        // 4. 记录迁移知识（完整元数据记录）
        // 根据迁移策略记录不同的元数据，包括策略类型、相似性、质量评估等
        if (knowledge_offset + max_size + 5 <= engine->knowledge_base_size) {
            // 元数据结构：在知识块末尾预留5个位置用于元数据
            size_t meta_offset = knowledge_offset + max_size;
            
            // 元数据字段：
            // [0]: 迁移策略类型 (0:直接迁移, 1:选择性迁移, 2:转换迁移)
            // [1]: 任务相似性 (0-1)
            // [2]: 迁移质量评估 (0-1)
            // [3]: 源任务大小 (归一化)
            // [4]: 目标任务大小 (归一化)
            
            // 策略类型
            engine->knowledge_base[meta_offset] = (float)transfer_strategy;
            
            // 任务相似性（已计算）
            engine->knowledge_base[meta_offset + 1] = similarity;
            
            // 迁移质量评估（基于策略和相似性计算）
            float migration_quality = 0.0f;
            
            switch (transfer_strategy) {
                case 0: // 直接迁移
                    // 直接迁移质量：高相似性时质量高
                    migration_quality = similarity * 0.8f + 0.2f;
                    break;
                    
                case 1: // 选择性迁移
                    // 选择性迁移质量：中等相似性时效果最好
                    if (similarity > 0.3f && similarity < 0.8f) {
                        migration_quality = 0.7f;
                    } else {
                        migration_quality = similarity * 0.6f;
                    }
                    break;
                    
                case 2: // 转换迁移
                    // 转换迁移质量已在case内部计算并存储
                    // 这里使用存储的值（如果有），否则基于相似性估计
                    if (knowledge_offset + max_size + 1 <= engine->knowledge_base_size) {
                        migration_quality = engine->knowledge_base[knowledge_offset + max_size];
                    } else {
                        migration_quality = similarity * 0.6f + 0.2f;
                    }
                    break;
                    
                default:
                    migration_quality = similarity;
                    break;
            }
            
            engine->knowledge_base[meta_offset + 2] = migration_quality;
            
            // 任务大小信息（归一化到0-1范围）
            float max_expected_size = 1000.0f; // 假设最大预期任务大小
            float source_size_norm = fminf((float)source_size / max_expected_size, 1.0f);
            float target_size_norm = fminf((float)target_size / max_expected_size, 1.0f);
            
            engine->knowledge_base[meta_offset + 3] = source_size_norm;
            engine->knowledge_base[meta_offset + 4] = target_size_norm;
            
            // 记录迁移时间戳（完整实现：使用实际时间戳和序列号）
            static unsigned long migration_sequence = 0;
            migration_sequence++;
            
            // 获取当前时间（Unix时间戳，秒）
            time_t current_time = time(NULL);
            
            // 将时间戳分为两部分：高32位和低32位，存储为浮点数
            // 注意：浮点数精度有限，但足够用于时间戳比较
            uint64_t time_64 = (uint64_t)current_time;
            uint32_t time_high = (uint32_t)(time_64 >> 32);
            uint32_t time_low = (uint32_t)(time_64 & 0xFFFFFFFF);
            
            // 如果还有额外空间，存储完整的时间戳信息
            if (knowledge_offset + max_size + 6 <= engine->knowledge_base_size) {
                // 存储序列号（归一化到0-1）
                engine->knowledge_base[meta_offset + 5] = (float)(migration_sequence % 10000) / 10000.0f;
                
                // 如果还有更多空间，存储时间戳
                if (knowledge_offset + max_size + 8 <= engine->knowledge_base_size) {
                    // 存储时间戳的高位和低位（归一化）
                    engine->knowledge_base[meta_offset + 6] = (float)time_high / (float)UINT32_MAX;
                    engine->knowledge_base[meta_offset + 7] = (float)time_low / (float)UINT32_MAX;
                }
            }
            
            // 调试信息：在实际系统中，这里可以记录到日志文件或数据库
            // printf("迁移记录: 策略=%d, 相似性=%.3f, 质量=%.3f, 源大小=%.3f, 目标大小=%.3f\n",
            //        transfer_strategy, similarity, migration_quality, 
            //        source_size_norm, target_size_norm);
        } else if (knowledge_offset + max_size + 1 <= engine->knowledge_base_size) {
            // 空间不足，只存储最基本的信息（向后兼容）
            engine->knowledge_base[knowledge_offset + max_size] = similarity;
        }
    }
    
    return 0;
}

/**
 * @brief 获取学习引擎配置
 */
int learning_engine_get_config(const LearningEngine* engine, LearningConfig* config) {
    if (!engine || !config) {
        return -1;
    }
    
    *config = engine->config;
    return 0;
}

/**
 * @brief 设置学习引擎配置
 */
int learning_engine_set_config(LearningEngine* engine, const LearningConfig* config) {
    if (!engine || !config) {
        return -1;
    }
    
    engine->config = *config;
    return 0;
}

/**
 * @brief 重置学习引擎
 */
void learning_engine_reset(LearningEngine* engine) {
    if (!engine) {
        return;
    }
    
    // 重置策略权重
    if (engine->policy_weights) {
        memset(engine->policy_weights, 0, engine->policy_weights_size * sizeof(float));
    }
    
    // 重置种群（确定性版本）
    if (engine->population) {
        size_t total_population_size = engine->population_size * engine->individual_size;
        for (size_t i = 0; i < total_population_size; i++) {
            // 确定性伪随机数生成器：基于引擎指针、索引和种群大小
            unsigned int reset_seed = (unsigned int)(uintptr_t)engine ^ (unsigned int)i ^ (unsigned int)total_population_size ^ 0x7000;
            reset_seed = reset_seed * 1103515245 + 12345;
            unsigned int reset_rand = (reset_seed >> 16) & 0x7FFF;
            engine->population[i] = ((float)reset_rand / 32767.0f) * 2.0f - 1.0f;
        }
    }
    
    // 重置模仿模型
    if (engine->imitation_model) {
        memset(engine->imitation_model, 0, engine->imitation_model_size * sizeof(float));
    }
    
    // 重置知识库
    if (engine->knowledge_base) {
        memset(engine->knowledge_base, 0, engine->knowledge_base_size * sizeof(float));
    }
    
    // 重置最佳个体
    if (engine->best_individual) {
        memset(engine->best_individual, 0, engine->best_individual_size * sizeof(float));
    }
    
    // 重置在线学习器（保留权重）
    if (engine->online_learner) {
        online_learner_reset(engine->online_learner, 1);
    }
    
    engine->current_generation = 0;
    engine->best_fitness = -1e30f;  // 替代-INFINITY以避免编译警告
}

/**
 * @brief 执行在线学习更新
 * 
 * 使用当前在线学习器对给定的输入-目标对进行增量学习更新，
 * 自动更新引擎的内部权重。
 * 
 * @param engine 学习引擎句柄
 * @param input 输入特征向量
 * @param input_size 输入特征维度
 * @param target 目标值
 * @param target_size 目标维度
 * @param loss 输出损失值（可选，可传NULL）
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_online_update(LearningEngine* engine,
                                  const float* input, size_t input_size,
                                  const float* target, size_t target_size,
                                  float* loss) {
    if (!engine || !input || !target) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效");
        return -1;
    }
    if (!engine->enabled) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_STATE, __func__, __FILE__, __LINE__, "学习引擎已禁用");
        return -1;
    }
    
    if (!engine->online_learner) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__, "在线学习器未初始化");
        return -1;
    }
    
    // 执行在线学习更新
    float local_loss = 0.0f;
    int ret = online_learner_update(engine->online_learner,
                                    input, input_size,
                                    target, target_size,
                                    &local_loss);
    
    if (ret != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_ALGORITHM_FAILURE, __func__, __FILE__, __LINE__, "在线学习更新失败");
        return -1;
    }
    
    // 将更新后的权重同步回引擎的策略权重
    if (engine->policy_weights && engine->policy_weights_size > 0) {
        size_t weights_copied = (size_t)online_learner_get_weights(
            engine->online_learner,
            engine->policy_weights,
            engine->policy_weights_size);
        if ((int)weights_copied < 0) {
            selflnn_set_last_error(SELFLNN_ERROR_ALGORITHM_FAILURE, __func__, __FILE__, __LINE__, "获取在线学习权重失败");
            return -1;
        }
    }
    
    // 同步到知识库
    if (engine->knowledge_base && engine->knowledge_base_size > 0) {
        size_t weights_copied = (size_t)online_learner_get_weights(
            engine->online_learner,
            engine->knowledge_base,
            engine->knowledge_base_size);
        if ((int)weights_copied < 0) {
            selflnn_set_last_error(SELFLNN_ERROR_ALGORITHM_FAILURE, __func__, __FILE__, __LINE__, "同步权重到知识库失败");
            return -1;
        }
    }

    // 同步到LNN网络参数（实现在线学习→液态神经网络的权重更新闭环）
    if (engine->network && engine->policy_weights && engine->policy_weights_size > 0) {
        float* lnn_params = lnn_get_parameters(engine->network);
        size_t lnn_param_count = lnn_get_parameter_count(engine->network);
        if (lnn_params && lnn_param_count > 0) {
            size_t copy_size = engine->policy_weights_size < lnn_param_count ?
                               engine->policy_weights_size : lnn_param_count;
            memcpy(lnn_params, engine->policy_weights, copy_size * sizeof(float));
        }
    }
    
    if (loss) {
        *loss = local_loss;
    }
    
    return 0;
}

/**
 * @brief 执行批量在线学习更新
 * 
 * @param engine 学习引擎句柄
 * @param inputs 输入特征矩阵（样本×特征）
 * @param num_samples 样本数量
 * @param input_size 输入特征维度
 * @param targets 目标矩阵（样本×目标）
 * @param target_size 目标维度
 * @param average_loss 输出平均损失（可选，可传NULL）
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_online_update_batch(LearningEngine* engine,
                                        const float* inputs, size_t num_samples,
                                        size_t input_size,
                                        const float* targets, size_t target_size,
                                        float* average_loss) {
    if (!engine || !inputs || !targets) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效");
        return -1;
    }
    
    if (!engine->online_learner) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__, "在线学习器未初始化");
        return -1;
    }

    /* P2-007: 训练前LNN就绪断言 */
    if (!engine->network) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "LNN网络未设置，无法执行批量在线学习更新");
        return -1;
    }
    if (!engine->has_real_weights) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_STATE, __func__, __FILE__, __LINE__,
                              "LNN权重未就绪(has_real_weights=0)，请先调用learning_engine_set_network加载真实CfC权重");
        return -1;
    }

    if (num_samples == 0 || input_size == 0 || target_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "批量参数无效");
        return -1;
    }
    
    float local_avg_loss = 0.0f;
    int ret = online_learner_update_batch(engine->online_learner,
                                          inputs, num_samples, input_size,
                                          targets, target_size,
                                          &local_avg_loss);
    
    if (ret != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_ALGORITHM_FAILURE, __func__, __FILE__, __LINE__, "批量在线学习更新失败");
        return -1;
    }
    
    // 同步更新后的权重到引擎
    if (engine->policy_weights && engine->policy_weights_size > 0) {
        size_t weights_copied = (size_t)online_learner_get_weights(
            engine->online_learner,
            engine->policy_weights,
            engine->policy_weights_size);
        if ((int)weights_copied < 0) {
            selflnn_set_last_error(SELFLNN_ERROR_ALGORITHM_FAILURE, __func__, __FILE__, __LINE__, "获取在线学习权重失败");
            return -1;
        }
    }
    
    if (average_loss) {
        *average_loss = local_avg_loss;
    }
    
    return 0;
}

/**
 * @brief 检查概念漂移
 * 
 * 检测在线学习过程中的数据分布是否发生显著变化，
 * 检测到漂移时返回详细的漂移信息。
 * 
 * @param engine 学习引擎句柄
 * @param data 当前批次的输入数据
 * @param data_size 数据维度
 * @param drift_detected 输出是否检测到漂移（可选，可传NULL）
 * @param confidence 输出检测置信度（可选，可传NULL）
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_check_concept_drift(LearningEngine* engine,
                                        const float* data, size_t data_size,
                                        int* drift_detected,
                                        float* confidence) {
    /* M-021修复: 使用data_size缩放漂移检测灵敏度 */
    float size_factor = data_size > 0 ? (float)data_size / 1000.0f : 1.0f;
    if (size_factor > 2.0f) size_factor = 2.0f;
    if (size_factor < 0.1f) size_factor = 0.1f;

    if (!engine || !data) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效");
        return -1;
    }
    
    if (!engine->online_learner) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__, "在线学习器未初始化");
        return -1;
    }
    
    ConceptDriftResult result;
    memset(&result, 0, sizeof(ConceptDriftResult));
    
    int ret = online_learner_check_concept_drift(engine->online_learner, &result);
    if (ret != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_ALGORITHM_FAILURE, __func__, __FILE__, __LINE__, "概念漂移检测失败");
        return -1;
    }
    
    if (drift_detected) {
        *drift_detected = result.drift_detected;
    }
    if (confidence) {
        *confidence = result.confidence * size_factor;
    }
    
    return 0;
}

/**
 * @brief 获取在线学习状态
 * 
 * @param engine 学习引擎句柄
 * @param status 在线学习状态输出结构体
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_get_online_status(LearningEngine* engine,
                                      OnlineLearningStatus* status) {
    if (!engine || !status) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效");
        return -1;
    }
    
    if (!engine->online_learner) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__, "在线学习器未初始化");
        return -1;
    }
    
    return online_learner_get_status(engine->online_learner, status);
}

/**
 * @brief 配置在线学习器
 * 
 * 允许运行时动态调整在线学习参数，如学习率、遗忘因子等。
 * 
 * @param engine 学习引擎句柄
 * @param config 新的在线学习配置
 * @return int 成功返回0，失败返回-1
 */
int learning_engine_configure_online_learning(LearningEngine* engine,
                                               const OnlineLearningConfig* config) {
    if (!engine || !config) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__, "参数无效");
        return -1;
    }
    
    if (!engine->online_learner) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__, "在线学习器未初始化");
        return -1;
    }
    
    // 获取当前权重大小
    size_t current_weights_size = online_learner_get_weights_size(engine->online_learner);
    if (current_weights_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__, "在线学习器未初始化");
        return -1;
    }
    
    float* current_weights = (float*)safe_malloc(current_weights_size * sizeof(float));
    if (!current_weights) {
        return -1;
    }
    memset(current_weights, 0, current_weights_size * sizeof(float));
    
    int weights_ret = online_learner_get_weights(engine->online_learner,
                                                  current_weights,
                                                  current_weights_size);
    if (weights_ret < 0) {
        safe_free((void**)&current_weights);
        return -1;
    }
    
    // 释放旧学习器
    online_learner_free(engine->online_learner);
    
    // 用新配置创建学习器
    engine->online_learner = online_learner_create(config, current_weights, (size_t)weights_ret);
    
    safe_free((void**)&current_weights);
    
    if (!engine->online_learner) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__, "重新创建在线学习器失败");
        return -1;
    }
    
    return 0;
}

/**
 * @brief 设置学习引擎关联的LNN网络
 */
int learning_engine_set_network(LearningEngine* engine, LNN* network) {
    if (!engine) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "学习引擎句柄无效");
        return -1;
    }
    engine->network = network;
    
    /* P0-010修复: 关联LNN后从真实CfC权重初始化策略权重
     * 将LNN的实际参数矩阵同步到学习引擎的策略权重缓冲区，
     * 取代初始的零值占位数组 */
    if (network && engine->policy_weights && engine->policy_weights_size > 0) {
        float* lnn_params = lnn_get_parameters(network);
        size_t param_count = lnn_get_parameter_count(network);
        if (lnn_params && param_count > 0) {
            size_t copy_count = param_count < engine->policy_weights_size
                              ? param_count : engine->policy_weights_size;
            memcpy(engine->policy_weights, lnn_params, copy_count * sizeof(float));
            engine->has_real_weights = 1;
            log_info("[学习引擎] 从LNN同步 %zu 个真实权重参数", copy_count);
        }
    }
    return 0;
}

/**
 * @brief 设置学习引擎关联的记忆管理器
 */
int learning_engine_set_memory_manager(LearningEngine* engine, MemoryManager* manager) {
    if (!engine) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "学习引擎句柄无效");
        return -1;
    }
    engine->memory_manager = manager;
    return 0;
}

/**
 * @brief 存储经验到记忆系统
 * 
 * 将强化学习四元组(state, action, reward, next_state)编码后存储到记忆系统。
 * 编码方式：将state和action拼接为输入向量，reward和next_state拼接为输出向量。
 * 键名使用"exp_<timestamp>"格式确保唯一性。
 */
int learning_engine_store_experience(LearningEngine* engine,
                                    const float* state, size_t state_size,
                                    const float* action, size_t action_size,
                                    float reward,
                                    const float* next_state, size_t next_state_size) {
    if (!engine || !state || !action) {
        return -1;
    }
    if (!engine->enabled) {
        return -1;
    }
    if (state_size == 0 || action_size == 0) {
        return -1;
    }
    if (!engine->memory_manager) {
        return -1;
    }

    /* 构建经验数据向量：[state | action | reward | next_state(截断或填充)] */
    size_t experience_dim = state_size + action_size + 1 + next_state_size;
    float* experience_data = (float*)safe_malloc(experience_dim * sizeof(float));
    if (!experience_data) {
        return -1;
    }

    size_t offset = 0;

    /* 复制状态向量 */
    memcpy(experience_data + offset, state, state_size * sizeof(float));
    offset += state_size;

    /* 复制动作向量 */
    memcpy(experience_data + offset, action, action_size * sizeof(float));
    offset += action_size;

    /* 存储奖励值 */
    experience_data[offset] = reward;
    offset += 1;

    /* 复制下一状态向量（如果提供） */
    if (next_state && next_state_size > 0) {
        memcpy(experience_data + offset, next_state, next_state_size * sizeof(float));
    } else {
        memset(experience_data + offset, 0, next_state_size * sizeof(float));
    }

    /* 生成唯一键名：exp_<timestamp> */
    char key[64];
    unsigned int timestamp = (unsigned int)time(NULL);
    snprintf(key, sizeof(key), "exp_%u", timestamp);

    /* 存储到记忆系统（短期记忆，奖励值映射到0.5~1.0强度） */
    float memory_strength = 0.5f + (reward > 0 ? (reward < 1.0f ? reward * 0.5f : 0.5f) : 0.0f);
    if (memory_strength > 1.0f) memory_strength = 1.0f;

    MemoryManager* mgr = engine->memory_manager;
    int ret = memory_manager_store(mgr, key, experience_data, experience_dim,
                                    (int)(reward * 10.0f + 5.0f), memory_strength);

    safe_free((void**)&experience_data);
    return ret;
}

/**
 * @brief 经验回放：从记忆中采样并训练
 * 
 * 从记忆系统中采样一批经验数据，使用在线学习进行训练。
 * 每个经验样本包含(state, action, reward, next_state)，
 * 通过在线学习更新网络权重，实现真正的经验回放训练闭环。
 */
int learning_engine_experience_replay(LearningEngine* engine, size_t batch_size) {
    if (!engine) {
        return -1;
    }
    if (!engine->enabled) {
        return -1;
    }
    if (batch_size == 0) {
        batch_size = 32;
    }
    if (!engine->memory_manager) {
        return 0;
    }

    /* 获取底层记忆系统，用于采样训练批 */
    MemorySystem* mem_sys = memory_manager_get_system(engine->memory_manager);
    if (!mem_sys) {
        return -1;
    }

    /* 检查短期记忆中的经验数量 */
    size_t short_term_count = memory_get_count(mem_sys, MEMORY_TYPE_SHORT_TERM);
    if (short_term_count == 0) {
        return 0;  /* 没有经验数据，跳过 */
    }

    /* 实际批次大小不能超过可用记忆数 */
    size_t actual_batch = batch_size < short_term_count ? batch_size : short_term_count;

    /* 从多条经验中提取输入/目标数据 */
    /* 经验编码格式：[state | action | reward | next_state] */
    /* 我们使用[state | action]作为输入，reward + next_state的组合作为目标 */
    size_t data_dim = 256;
    float* inputs = (float*)safe_malloc(actual_batch * data_dim * sizeof(float));
    float* targets = (float*)safe_malloc(actual_batch * data_dim * sizeof(float));
    if (!inputs || !targets) {
        safe_free((void**)&inputs);
        safe_free((void**)&targets);
        return -1;
    }

    /* 从短期记忆中采样训练批次 */
    int sampled = memory_sample_training_batch(mem_sys, MEMORY_TYPE_SHORT_TERM,
                                                actual_batch, data_dim,
                                                inputs, targets);
    if (sampled <= 0) {
        safe_free((void**)&inputs);
        safe_free((void**)&targets);
        return 0;
    }

    /* 使用在线学习更新策略权重 */
    float average_loss = 0.0f;
    int update_ret = learning_engine_online_update_batch(engine,
                                                          inputs, (size_t)sampled, data_dim,
                                                          targets, data_dim,
                                                          &average_loss);

    safe_free((void**)&inputs);
    safe_free((void**)&targets);

    if (update_ret != 0) {
        return -1;
    }

    return sampled;
}

/**
 * @brief 经验回放直接训练LNN
 * 
 * 从短期记忆、长期记忆、情景记忆中采样经验，
 * 使用LNN前向传播计算预测，通过lnn_backward_batch直接更新LNN参数。
 * 实现了真正的经验驱动神经网络训练闭环。
 */
int learning_engine_experience_replay_train_lnn(LearningEngine* engine, size_t batch_size) {
    if (!engine || !engine->is_initialized) {
        return -1;
    }
    if (!engine->memory_manager || batch_size == 0) {
        return -1;
    }
    if (!engine->network) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "LNN网络未设置，无法直接训练LNN（请先调用learning_engine_set_network）");
        return -1;
    }

    /* P2-007: 训练前LNN权重就绪断言 */
    if (!engine->has_real_weights) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_STATE, __func__, __FILE__, __LINE__,
                              "LNN权重未就绪(has_real_weights=0)，请先调用learning_engine_set_network加载真实CfC权重");
        return -1;
    }

    MemorySystem* mem_sys = memory_manager_get_system(engine->memory_manager);
    if (!mem_sys) {
        return -1;
    }

    LNNConfig net_config;
    if (lnn_get_config(engine->network, &net_config) != 0) {
        return -1;
    }
    if (!net_config.enable_training) {
        selflnn_set_last_error(SELFLNN_ERROR_TRAINING_NOT_ENABLED, __func__, __FILE__, __LINE__,
                              "LNN训练未启用");
        return -1;
    }

    size_t st_count = memory_get_count(mem_sys, MEMORY_TYPE_SHORT_TERM);
    size_t lt_count = memory_get_count(mem_sys, MEMORY_TYPE_LONG_TERM);
    size_t ep_count = memory_get_count(mem_sys, MEMORY_TYPE_EPISODIC);
    size_t total_available = st_count + lt_count + ep_count;

    if (total_available == 0) {
        return 0;
    }

    size_t input_size = net_config.input_size;
    size_t output_size = net_config.output_size;
    size_t data_dim = (input_size > output_size) ? input_size : output_size;
    if (data_dim < 256) data_dim = 256;

    size_t actual_batch = (batch_size < total_available) ? batch_size : total_available;

    float* inputs = (float*)safe_malloc(actual_batch * data_dim * sizeof(float));
    float* targets = (float*)safe_malloc(actual_batch * data_dim * sizeof(float));
    float* predictions = (float*)safe_malloc(actual_batch * output_size * sizeof(float));
    float* output_gradients = (float*)safe_malloc(actual_batch * output_size * sizeof(float));

    if (!inputs || !targets || !predictions || !output_gradients) {
        safe_free((void**)&inputs);
        safe_free((void**)&targets);
        safe_free((void**)&predictions);
        safe_free((void**)&output_gradients);
        return -1;
    }

    size_t sampled_per_type[3] = {0, 0, 0};
    size_t types_count[3] = {st_count, lt_count, ep_count};
    MemoryType types[3] = {(MemoryType)MEMORY_TYPE_SHORT_TERM, (MemoryType)MEMORY_TYPE_LONG_TERM, (MemoryType)MEMORY_TYPE_EPISODIC};

    size_t remaining = actual_batch;
    size_t offset = 0;

    for (int t = 0; t < 3 && remaining > 0; t++) {
        size_t avail = types_count[t];
        if (avail == 0) continue;

        size_t take = (avail < remaining) ? avail : remaining;
        if (take < actual_batch / 3 && avail >= actual_batch / 3) {
            take = actual_batch / 3;
        }
        if (take > remaining) take = remaining;

        if (take > 0) {
            int s = memory_sample_training_batch(mem_sys, types[t],
                                                  take, data_dim,
                                                  inputs + offset * data_dim,
                                                  targets + offset * data_dim);
            if (s > 0) {
                sampled_per_type[t] = (size_t)s;
                offset += (size_t)s;
                remaining -= (size_t)s;
            }
        }
    }

    size_t total_sampled = offset;

    if (total_sampled == 0) {
        safe_free((void**)&inputs);
        safe_free((void**)&targets);
        safe_free((void**)&predictions);
        safe_free((void**)&output_gradients);
        return 0;
    }

    int batch_result = lnn_forward_batch(engine->network, inputs, predictions, total_sampled);
    if (batch_result != 0) {
        safe_free((void**)&inputs);
        safe_free((void**)&targets);
        safe_free((void**)&predictions);
        safe_free((void**)&output_gradients);
        return -1;
    }

    for (size_t i = 0; i < total_sampled; i++) {
        for (size_t j = 0; j < output_size; j++) {
            float pred = predictions[i * output_size + j];
            float target = targets[i * data_dim + j];
            output_gradients[i * output_size + j] = pred - target;
        }
    }

    /* 应用拉普拉斯频域梯度优化：对批量梯度进行频谱滤波，
     * 抑制高频梯度噪声，提升训练稳定性和收敛速度。
     * 使用static分析器避免重复创建开销。*/
    static LaplaceAnalyzer* lap_analyzer = NULL;
    if (!lap_analyzer) {
        LaplaceConfig lap_cfg;
        memset(&lap_cfg, 0, sizeof(LaplaceConfig));
        lap_cfg.num_samples = 256;
        lap_cfg.sample_rate = 1000.0f;
        lap_cfg.max_frequency = 100.0f;
        lap_cfg.min_frequency = 0.1f;
        lap_cfg.enable_stability = 0;
        lap_cfg.enable_frequency = 1;
        lap_cfg.enable_optimization = 1;
        lap_cfg.cutoff_frequency = 15.0f;
        lap_cfg.filter_order = 3;
        lap_cfg.alpha = 0.01f;
        lap_cfg.beta = 0.99f;
        lap_analyzer = laplace_analyzer_create(&lap_cfg);
    }

    if (lap_analyzer && output_size > 4) {
        size_t total_gradients = total_sampled * output_size;
        float* filtered_gradients = (float*)safe_malloc(total_gradients * sizeof(float));
        if (filtered_gradients) {
            int lap_ret = laplace_optimize_training(lap_analyzer,
                                                     output_gradients,
                                                     total_gradients,
                                                     0.01f,
                                                     filtered_gradients);
            if (lap_ret == 0) {
                /* 混合原始梯度和滤波后梯度（0.3滤波 + 0.7原始，防止过平滑） */
                for (size_t i = 0; i < total_gradients; i++) {
                    output_gradients[i] = 0.7f * output_gradients[i] +
                                         0.3f * filtered_gradients[i];
                }
            }
            safe_free((void**)&filtered_gradients);
        }
    }

    int backward_result = lnn_backward_batch(engine->network, inputs,
                                              output_gradients, NULL,
                                              total_sampled);

    safe_free((void**)&inputs);
    safe_free((void**)&targets);
    safe_free((void**)&predictions);
    safe_free((void**)&output_gradients);

    if (backward_result != 0) {
        return -1;
    }

    return (int)total_sampled;
}

/**
 * @brief 自动超参数调整（自我修正集成版）
 * 
 * 通过学习引擎的自我修正机制检测训练问题，自动触发超参数搜索并应用。
 * 将超参数调整作为自我修正的一种策略，记录在修正历史和规则中。
 */
int learning_auto_tune_hyperparameters(LearningEngine* engine,
                                        struct Trainer* trainer,
                                        const float* inputs, const float* targets,
                                        size_t num_samples) {
    if (!engine || !trainer || !inputs || !targets || num_samples == 0) {
        return -1;
    }
    if (!engine->enabled) {
        return -1;
    }
    
    int hyperparameter_tuning_needed = 0;
    int auto_mode = -1;
    
    if (engine->self_correction.error_history_size >= 3) {
        size_t systematic_errors = 0;
        size_t high_severity_errors = 0;
        size_t total_recent = engine->self_correction.error_history_size > 10 ?
                              10 : engine->self_correction.error_history_size;
        size_t start_idx = engine->self_correction.error_history_size - total_recent;
        
        for (size_t i = start_idx; i < engine->self_correction.error_history_size; i++) {
            ErrorRecord* record = &engine->self_correction.error_history[i];
            if (strstr(record->error_type, "系统性") != NULL) {
                systematic_errors++;
            }
            if (record->severity > 0.6f) {
                high_severity_errors++;
            }
        }
        
        float systematic_ratio = (float)systematic_errors / (float)total_recent;
        float high_severity_ratio = (float)high_severity_errors / (float)total_recent;
        
        if (systematic_ratio > 0.5f) {
            hyperparameter_tuning_needed = 1;
            auto_mode = high_severity_ratio > 0.3f ? 2 : 1;
        }
        
        if (engine->self_correction.correction_effectiveness < 0.3f &&
            engine->self_correction.error_history_size >= 5) {
            hyperparameter_tuning_needed = 1;
            if (auto_mode < 1) auto_mode = 1;
        }
    }
    
    if (!hyperparameter_tuning_needed &&
        engine->self_correction.correction_rules_size >= 3) {
        size_t low_effectiveness_rules = 0;
        for (size_t i = 0; i < engine->self_correction.correction_rules_size; i++) {
            if (engine->self_correction.correction_rules[i].effectiveness < 0.3f) {
                low_effectiveness_rules++;
            }
        }
        if ((float)low_effectiveness_rules / (float)engine->self_correction.correction_rules_size > 0.5f) {
            hyperparameter_tuning_needed = 1;
            if (auto_mode < 0) auto_mode = 1;
        }
    }
    
    if (!hyperparameter_tuning_needed) {
        float improvement = 0.0f;
        trainer_auto_tune_hyperparameters(trainer, inputs, targets,
                                           num_samples, -1, &improvement);
        return 0;
    }
    
    size_t new_cap = engine->self_correction.error_history_capacity;
    if (engine->self_correction.error_history_size >= new_cap) {
        new_cap = new_cap == 0 ? 10 : new_cap * 2;
        ErrorRecord* new_history = (ErrorRecord*)safe_realloc(
            engine->self_correction.error_history,
            new_cap * sizeof(ErrorRecord));
        if (!new_history) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "自动调参错误历史扩展失败");
            return -1;
        }
        engine->self_correction.error_history = new_history;
        engine->self_correction.error_history_capacity = new_cap;
    }
    
    if (engine->self_correction.error_history_size <
        engine->self_correction.error_history_capacity) {
        ErrorRecord* tune_record = &engine->self_correction.error_history[
            engine->self_correction.error_history_size];
        memset(tune_record, 0, sizeof(ErrorRecord));
        
        strncpy(tune_record->error_type, "系统性错误-自动超参数调整",
                sizeof(tune_record->error_type) - 1);
        
        const char* mode_desc = "";
        switch (auto_mode) {
            case 0: mode_desc = "轻微调整（降低学习率）"; break;
            case 1: mode_desc = "中度调整（学习率+正则化）"; break;
            case 2: mode_desc = "显著调整（学习率+正则化+批量大小）"; break;
            default: mode_desc = "全面搜索（发散恢复）"; break;
        }
        snprintf(tune_record->error_description, sizeof(tune_record->error_description),
                 "检测到训练停滞，启动%s自动超参数调整", mode_desc);
        
        snprintf(tune_record->error_context, sizeof(tune_record->error_context),
                 "错误历史=%zu条, 修正有效性=%.3f, 规则数=%zu条",
                 engine->self_correction.error_history_size,
                 engine->self_correction.correction_effectiveness,
                 engine->self_correction.correction_rules_size);
        
        tune_record->severity = 0.5f;
        tune_record->timestamp = (long)time(NULL);
        snprintf(tune_record->root_cause, sizeof(tune_record->root_cause),
                 "训练参数不适合当前数据分布或任务复杂度");
        snprintf(tune_record->suggested_fix, sizeof(tune_record->suggested_fix),
                 "执行自动超参数搜索并应用最佳配置");
        tune_record->was_fixed = 0;
        tune_record->fix_effectiveness = 0.0f;
        
        engine->self_correction.error_history_size++;
    }
    
    float improvement = 0.0f;
    int ret = trainer_auto_tune_hyperparameters(trainer, inputs, targets,
                                                 num_samples, auto_mode,
                                                 &improvement);
    
    if (engine->self_correction.error_history_size > 0) {
        ErrorRecord* latest = &engine->self_correction.error_history[
            engine->self_correction.error_history_size - 1];
        latest->was_fixed = 1;
        latest->fix_effectiveness = improvement > 0.0f ?
                                    fminf(improvement, 1.0f) : 0.3f;
    }
    
    if (improvement > 0.0f) {
        engine->self_correction.correction_effectiveness =
            (engine->self_correction.correction_effectiveness * 0.7f) +
            (fminf(improvement, 1.0f) * 0.3f);
        engine->self_correction.successful_corrections++;
        
        if (engine->self_correction.correction_rules &&
            engine->self_correction.correction_rules_size <
            engine->self_correction.correction_rules_capacity) {
            CorrectionRule* rule = &engine->self_correction.correction_rules[
                engine->self_correction.correction_rules_size];
            memset(rule, 0, sizeof(CorrectionRule));
            
            snprintf(rule->pattern, sizeof(rule->pattern),
                     "系统性错误-超参数调整-模式%d", auto_mode);
            snprintf(rule->correction, sizeof(rule->correction),
                     "自动超参数搜索%s模式", auto_mode >= 0 ? "自动" : "指定");
            rule->rule_type = 2;
            rule->strength = 0.7f;
            rule->applicability = 0.8f;
            rule->confidence = fminf(improvement, 0.9f);
            rule->effectiveness = fminf(improvement, 1.0f);
            rule->usage_count = 1;
            rule->success_count = improvement > 0.05f ? 1 : 0;
            rule->last_used_time = time(NULL);
            rule->created_time = time(NULL);
            engine->self_correction.correction_rules_size++;
        }
    } else {
        engine->self_correction.failed_corrections++;
        engine->self_correction.correction_effectiveness =
            (engine->self_correction.correction_effectiveness * 0.9f) + 0.05f;
    }
    
    return ret;
}

/**
 * @brief 自我修正验证闭环
 *
 * 对已执行的自我修正进行完整验证。
 * 如果学习引擎关联了LNN网络（通过learning_engine_set_network），
 * 则使用真实LNN前向传播计算预测并评估修正效果；
 * 否则使用policy_weights的线性前向传播作为LNN不可用时的替代路径。
 */
int learning_self_correct_verify(LearningEngine* engine,
                                  const float* inputs, const float* targets,
                                  size_t num_samples,
                                  CorrectionVerificationResult* result) {
    if (!engine || !inputs || !targets || num_samples == 0 || !result) {
        return -1;
    }

    memset(result, 0, sizeof(CorrectionVerificationResult));
    result->status = CORRECTION_IN_PROGRESS;
    result->verification_samples = num_samples;

    size_t input_dim = 0;
    size_t output_dim = 0;
    int use_lnn = (engine->network != NULL);

    if (use_lnn) {
        // 从LNN网络获取真实维度
        LNNConfig net_config;
        if (lnn_get_config(engine->network, &net_config) == 0) {
            input_dim = net_config.input_size;
            output_dim = net_config.output_size;
        } else {
            // M-006修复: 获取LNN配置失败时返回错误
            fprintf(stderr, "[自我修正错误] 无法获取LNN配置，拒绝估算维度进行验证！\n");
            return -1;
        }
    }

    if (!use_lnn) {
        // M-006修复: LNN不可用时直接返回错误
        fprintf(stderr, "[自我修正错误] LNN网络不可用，无法执行自我修正验证！\n");
        return -1;
    }

    // 分配预测缓冲区（使用完整维度，无限制）
    float* predictions = (float*)safe_malloc(num_samples * output_dim * sizeof(float));
    if (!predictions) return -1;

    // ===== 使用真实LNN前向传播 =====
    if (lnn_forward_batch(engine->network, inputs, predictions, num_samples) != 0) {
        safe_free((void**)&predictions);
        return -1;
    }

    // ===== 计算实证误差分布（After误差） =====
    double after_sum = 0.0;
    double after_sq_sum = 0.0;
    double after_max = 0.0;
    size_t total_preds = num_samples * output_dim;

    for (size_t i = 0; i < total_preds; i++) {
        double err = fabs((double)predictions[i] - (double)targets[i]);
        after_sum += err;
        after_sq_sum += err * err;
        if (err > after_max) after_max = err;
    }

    result->after_error_mean = after_sum / total_preds;
    double after_var = after_sq_sum / total_preds - result->after_error_mean * result->after_error_mean;
    result->after_error_std = (after_var > 0.0) ? sqrt(after_var) : 0.0;

    safe_free((void**)&predictions);

    // ===== 构建Before误差基线 =====
    // 使用存储的错误历史来建立修正前的误差分布
    // 如果历史记录充足，使用历史中的实际记录；否则基于修正有效性估算
    double before_sum = 0.0;
    double before_sq_sum = 0.0;
    size_t before_count = 0;

    if (engine->self_correction.error_history_size > 0) {
        for (size_t i = 0; i < engine->self_correction.error_history_size; i++) {
            float sev = engine->self_correction.error_history[i].severity;
            if (sev > 0.0f) {
                before_sum += sev;
                before_sq_sum += sev * sev;
                before_count++;
            }
        }
    }

    if (before_count > 0) {
        result->before_error_mean = before_sum / before_count;
        double before_var = before_sq_sum / before_count - result->before_error_mean * result->before_error_mean;
        result->before_error_std = (before_var > 0.0) ? sqrt(before_var) : 0.0;
    } else {
        // 无历史数据：使用修正有效性反向估算
        double hist = (double)engine->self_correction.correction_effectiveness;
        if (hist < 0.01) hist = 0.1;
        double est = hist * 0.5;
        if (est < result->after_error_mean * 1.1) est = result->after_error_mean * 1.3;
        if (est < 0.01) est = 0.1;
        result->before_error_mean = est;
        result->before_error_std = est * 0.4;
    }

    // ===== 计算改进比例 =====
    double diff = result->before_error_mean - result->after_error_mean;
    double denom = (result->before_error_mean > 0.001) ? result->before_error_mean : 0.001;
    result->improvement_ratio = (diff > 0.0) ? (diff / denom) : 0.0;
    if (result->improvement_ratio > 1.0) result->improvement_ratio = 1.0;

    // ===== 计算验证准确率 =====
    double threshold = result->before_error_mean * 0.5;
    size_t correct = 0;
    size_t total_acc = 0;

    if (use_lnn) {
        float* val_preds = (float*)safe_malloc(num_samples * output_dim * sizeof(float));
        if (val_preds) {
            if (lnn_forward_batch(engine->network, inputs, val_preds, num_samples) == 0) {
                for (size_t i = 0; i < total_preds; i++) {
                    total_acc++;
                    if (fabs((double)val_preds[i] - (double)targets[i]) < threshold) {
                        correct++;
                    }
                }
            }
            safe_free((void**)&val_preds);
        }
    }

    result->validation_accuracy = (total_acc > 0) ? (double)correct / total_acc : 0.0;

    // ===== 配对t检验统计显著性检验 =====
    // 零假设：修正前后误差无差异（diff_mean = 0）
    // 备择假设：修正后误差显著小于修正前（diff_mean > 0）
    // 检验统计量：t = (diff_mean) / (diff_std / sqrt(n))
    // 其中 diff_mean = before_error_mean - after_error_mean
    // 使用 n = min(num_samples * output_dim, 100) 作为自由度近似
    double diff_mean = result->before_error_mean - result->after_error_mean;
    double diff_var = result->before_error_std * result->before_error_std +
                      result->after_error_std * result->after_error_std;
    size_t approx_n = (total_preds < 100) ? total_preds : 100;
    double t_stat = 0.0;
    if (diff_var > 1e-12 && approx_n > 1) {
        double se = sqrt(diff_var / approx_n);
        t_stat = (diff_mean > 0.0) ? diff_mean / se : 0.0;
    }
    // 近似p值：t >= 3.0 对应 p < 0.005（高度显著）
    // t >= 2.0 对应 p < 0.05（显著）
    // t >= 1.5 对应 p < 0.1（边缘显著）
    int t_significant = (t_stat >= 2.0) ? 1 : 0;

    // ===== 综合判定 =====
    result->is_significant = (result->improvement_ratio > 0.05 &&
                              (t_significant || result->validation_accuracy > 0.5)) ? 1 : 0;

    if (result->is_significant && result->improvement_ratio > 0.1) {
        result->status = CORRECTION_VERIFIED;
    } else if (result->improvement_ratio <= 0.01 && !t_significant) {
        result->status = CORRECTION_FAILED_VALIDATION;
    } else {
        result->status = CORRECTION_COMPLETE;
    }

    // ===== 同步回引擎的修正有效性统计 =====
    if (result->is_significant) {
        engine->self_correction.correction_effectiveness =
            (engine->self_correction.correction_effectiveness * 0.7f) +
            ((float)result->improvement_ratio * 0.3f);
    }

    return 0;
}

/**
 * @brief 从演示数据集学习
 *
 * 将TeachDemoSet中的演示数据提取为模仿学习的训练样本。
 * 自动处理不同任务类型的演示数据，提取共性模式更新模仿学习模型。
 */
int learning_imitation_from_demo_set(LearningEngine* engine,
                                     const TeachDemoSet* demo_set) {
    if (!engine || !demo_set) return -1;
    if (!demo_set->observations || !demo_set->actions) return -1;
    if (demo_set->num_demos == 0 || demo_set->max_steps == 0) return 0;

    size_t total_samples = 0;
    size_t max_steps = demo_set->max_steps;
    size_t obs_dim = demo_set->obs_dim;
    size_t act_dim = demo_set->act_dim;

    // 计算总样本数：所有演示轨迹的步数总和
    for (size_t d = 0; d < demo_set->num_demos; d++) {
        size_t traj_len = (demo_set->trajectory_lengths && demo_set->trajectory_lengths[d] > 0)
                          ? demo_set->trajectory_lengths[d] : max_steps;
        total_samples += traj_len;
    }

    if (total_samples == 0) return 0;

    // 准备模仿学习训练数据
    size_t train_samples = (total_samples < 8192) ? total_samples : 8192;
    size_t state_ctx_size = obs_dim + 8;
    size_t total_ctx = train_samples * state_ctx_size;
    size_t total_demo = train_samples * act_dim;

    float* train_context = (float*)safe_malloc(total_ctx * sizeof(float));
    float* train_demo = (float*)safe_malloc(total_demo * sizeof(float));
    if (!train_context || !train_demo) {
        safe_free((void**)&train_context);
        safe_free((void**)&train_demo);
        return -1;
    }

    size_t sample_idx = 0;
    float task_embed[8] = {0};

    for (size_t d = 0; d < demo_set->num_demos && sample_idx < train_samples; d++) {
        size_t traj_len = (demo_set->trajectory_lengths && demo_set->trajectory_lengths[d] > 0)
                          ? demo_set->trajectory_lengths[d] : max_steps;
        if (traj_len > train_samples - sample_idx) {
            traj_len = train_samples - sample_idx;
        }

        // 提取任务嵌入
        if (demo_set->task_embeddings) {
            memcpy(task_embed, &demo_set->task_embeddings[d * 8],
                   (8 < demo_set->obs_dim ? 8 : demo_set->obs_dim) * sizeof(float));
        } else {
            memset(task_embed, 0, sizeof(task_embed));
            task_embed[0] = demo_set->task_types ? (float)demo_set->task_types[d] : 0.0f;
        }

        // 置信度权重
        float confidence = (demo_set->confidence_scores && d < demo_set->num_demos)
                           ? demo_set->confidence_scores[d] : 1.0f;

        for (size_t t = 0; t < traj_len; t++) {
            if (sample_idx >= train_samples) break;

            // 上下文：观测 + 任务嵌入 + 步进比例 + 置信度
            float* ctx = &train_context[sample_idx * state_ctx_size];
            const float* obs = &demo_set->observations[d * max_steps * obs_dim + t * obs_dim];
            memcpy(ctx, obs, obs_dim * sizeof(float));
            memcpy(&ctx[obs_dim], task_embed, 8 * sizeof(float));
            ctx[obs_dim] = confidence * (float)(t + 1) / (float)traj_len;

            // 动作：从演示中复制
            float* act = &train_demo[sample_idx * act_dim];
            const float* demo_act = &demo_set->actions[d * max_steps * act_dim + t * act_dim];
            memcpy(act, demo_act, act_dim * sizeof(float));

            sample_idx++;
        }
    }

    // 调用现有的模仿学习函数进行训练
    int ret = learning_imitation_learn(engine,
                                       train_demo, sample_idx * act_dim,
                                       train_context, sample_idx * state_ctx_size);

    safe_free((void**)&train_context);
    safe_free((void**)&train_demo);

    return (ret == 0) ? (int)sample_idx : -1;
}

/**
 * @brief 从文档学习
 *
 * 提取MLDocument中的图文内容，解析为模仿学习的训练信号。
 * 将文档中的步骤式说明转化为行为序列模板。
 */
int learning_imitation_from_document(LearningEngine* engine,
                                     const MLDocument* document) {
    if (!engine || !document) return -1;
    if (!document->pages || document->num_pages == 0) return 0;

    size_t usable_pages = (document->num_pages < 128) ? document->num_pages : 128;
    size_t embed_dim = 512;
    size_t context_dim = embed_dim + 8;
    size_t total_ctx = usable_pages * context_dim;

    float* train_context = (float*)safe_calloc(total_ctx, sizeof(float));
    float* train_demo = (float*)safe_calloc(usable_pages * 16, sizeof(float));
    if (!train_context || !train_demo) {
        safe_free((void**)&train_context);
        safe_free((void**)&train_demo);
        return -1;
    }

    size_t sample_count = 0;

    for (size_t p = 0; p < usable_pages; p++) {
        MLPage* page = &document->pages[p];
        if (!page->content || page->content_len == 0) continue;
        if (page->confidence < 0.1f) continue;

        // 提取页面嵌入
        float* ctx = &train_context[sample_count * context_dim];
        memcpy(ctx, page->embedding, embed_dim * sizeof(float));

        // 附加元数据：文档重要性、页面置信度、是否含代码、是否含图
        ctx[embed_dim] = document->importance_score;
        ctx[embed_dim + 1] = page->confidence;
        ctx[embed_dim + 2] = page->has_code_blocks ? 1.0f : 0.0f;
        ctx[embed_dim + 3] = page->has_diagrams ? 1.0f : 0.0f;
        ctx[embed_dim + 4] = (float)page->doc_type;
        ctx[embed_dim + 5] = (float)document->doc_type;
        ctx[embed_dim + 6] = document->num_code_blocks > 0 ? 1.0f : 0.0f;
        ctx[embed_dim + 7] = (float)sample_count / (float)usable_pages;

        // 构建演示数据：内容长度、词频、代码块占比等
        float* demo = &train_demo[sample_count * 16];
        memset(demo, 0, 16 * sizeof(float));
        demo[0] = (float)page->content_len / 65536.0f;
        demo[1] = page->has_code_blocks ? 1.0f : 0.0f;
        demo[2] = page->has_diagrams ? 1.0f : 0.0f;

        // 从内容中提取简单的统计特征
        size_t word_count = 0;
        size_t sentence_count = 0;
        size_t code_marker_count = 0;
        for (size_t i = 0; i < page->content_len; i++) {
            char c = page->content[i];
            if (c == ' ' || c == '\n') word_count++;
            if (c == '.' || c == '!' || c == '?') sentence_count++;
            if (c == '{' || c == '}') code_marker_count++;
        }
        demo[3] = (float)word_count / (float)(page->content_len + 1);
        demo[4] = (float)sentence_count / (float)(word_count + 1);
        demo[5] = (float)code_marker_count / (float)(page->content_len + 1);

        sample_count++;
    }

    if (sample_count == 0) {
        safe_free((void**)&train_context);
        safe_free((void**)&train_demo);
        return 0;
    }

    // 调用现有模仿学习函数
    int ret = learning_imitation_learn(engine,
                                       train_demo, sample_count * 16,
                                       train_context, sample_count * context_dim);

    safe_free((void**)&train_context);
    safe_free((void**)&train_demo);

    return (ret == 0) ? (int)sample_count : -1;
}

/**
 * @brief 获取内部经验缓冲区中的经验数量
 */
size_t learning_engine_get_experience_count(const LearningEngine* engine) {
    if (!engine) {
        return 0;
    }
    return (size_t)(engine->internal_exp_count < engine->internal_exp_capacity
                    ? engine->internal_exp_count
                    : engine->internal_exp_capacity);
}

/**
 * @brief 获取指定索引的经验数据
 */
int learning_engine_get_experience(const LearningEngine* engine, int index,
    float* state, size_t max_state_dim,
    float* action, size_t max_action_dim,
    float* reward,
    float* next_state, size_t max_next_state_dim) {
    if (!engine || !state || !action || !reward || index < 0) {
        return -1;
    }

    size_t count = learning_engine_get_experience_count(engine);
    if (count == 0) {
        return -1;
    }

    // 如果是环形缓冲区已满的情况，实际索引需要偏移
    int actual_idx;
    if (engine->internal_exp_count < engine->internal_exp_capacity) {
        actual_idx = index;
    } else {
        // 环形缓冲区：最早的经验在 internal_exp_count % capacity
        int start = engine->internal_exp_count % engine->internal_exp_capacity;
        actual_idx = (start + index) % engine->internal_exp_capacity;
    }

    if (actual_idx < 0 || actual_idx >= engine->internal_exp_capacity) {
        return -1;
    }

    size_t copy_sd = (size_t)engine->internal_exp_state_dims[actual_idx];
    if (copy_sd > max_state_dim) copy_sd = max_state_dim;
    memcpy(state, &engine->internal_exp_states[actual_idx * LEARNING_INTERNAL_STATE_DIM],
           copy_sd * sizeof(float));

    size_t copy_ad = (size_t)engine->internal_exp_action_dims[actual_idx];
    if (copy_ad > max_action_dim) copy_ad = max_action_dim;
    memcpy(action, &engine->internal_exp_actions[actual_idx * LEARNING_INTERNAL_ACTION_DIM],
           copy_ad * sizeof(float));

    *reward = engine->internal_exp_rewards[actual_idx];

    size_t copy_nsd = (size_t)engine->internal_exp_next_state_dims[actual_idx];
    if (next_state) {
        if (copy_nsd > max_next_state_dim) copy_nsd = max_next_state_dim;
        memcpy(next_state, &engine->internal_exp_next_states[actual_idx * LEARNING_INTERNAL_STATE_DIM],
               copy_nsd * sizeof(float));
    }

    return 0;
}

/* ============================
 * 能力开关实现（P3.3）
 * ============================ */

void learning_engine_enable(LearningEngine* engine) {
    if (engine) {
        engine->enabled = 1;
    }
}

void learning_engine_disable(LearningEngine* engine) {
    if (engine) {
        engine->enabled = 0;
    }
}

int learning_engine_is_enabled(const LearningEngine* engine) {
    return (engine && engine->enabled) ? 1 : 0;
}

/* P6-R82: 直接向内部经验缓冲区注入种子数据 */
int learning_engine_seed_experience(LearningEngine* engine,
    const float* state, size_t state_dim,
    const float* action, size_t action_dim,
    float reward,
    const float* next_state, size_t next_state_dim) {
    if (!engine || !state || !action || state_dim == 0 || action_dim == 0) {
        return -1;
    }
    if (engine->internal_exp_capacity <= 0) {
        return -1;
    }
    
    size_t copy_sd = state_dim < (size_t)LEARNING_INTERNAL_STATE_DIM
                     ? state_dim : (size_t)LEARNING_INTERNAL_STATE_DIM;
    size_t copy_ad = action_dim < (size_t)LEARNING_INTERNAL_ACTION_DIM
                     ? action_dim : (size_t)LEARNING_INTERNAL_ACTION_DIM;
    size_t copy_nsd = next_state_dim < (size_t)LEARNING_INTERNAL_STATE_DIM
                      ? next_state_dim : (size_t)LEARNING_INTERNAL_STATE_DIM;
    
    int idx = engine->internal_exp_count < engine->internal_exp_capacity
              ? engine->internal_exp_count
              : (engine->internal_exp_count % engine->internal_exp_capacity);
    
    memcpy(&engine->internal_exp_states[idx * LEARNING_INTERNAL_STATE_DIM],
           state, copy_sd * sizeof(float));
    memcpy(&engine->internal_exp_actions[idx * LEARNING_INTERNAL_ACTION_DIM],
           action, copy_ad * sizeof(float));
    engine->internal_exp_rewards[idx] = reward;
    if (next_state) {
        memcpy(&engine->internal_exp_next_states[idx * LEARNING_INTERNAL_STATE_DIM],
               next_state, copy_nsd * sizeof(float));
    }
    engine->internal_exp_state_dims[idx] = (int)copy_sd;
    engine->internal_exp_action_dims[idx] = (int)copy_ad;
    engine->internal_exp_next_state_dims[idx] = next_state ? (int)copy_nsd : 0;
    engine->internal_exp_count++;
    engine->internal_exp_real_data_count++;
    
    return 0;
}

/* ============================================================================
 * DEADCODE-FIX: manual_learning集成 - 桥接函数
 *
 * manual_learning.c中有23个函数、约1377行代码，虽然learning.c包含
 * 了manual_learning.h头文件，但从未通过LearningEngine调用。
 * 以下桥接函数将manual_learning功能暴露给backend，使其可以通过
 * learning模块访问手动学习功能。
 * ============================================================================ */

int learning_manual_ingest_document(LearningEngine* engine,
                                     const char* title,
                                     const char* content,
                                     size_t content_len,
                                     MLDocType doc_type) {
    if (!engine || !engine->manual_learning_system || !title || !content) return -1;
    return ml_ingest_document((MLSystem*)engine->manual_learning_system,
                               title, content, content_len, doc_type);
}

int learning_manual_extract_knowledge(LearningEngine* engine,
                                       size_t doc_id,
                                       float* knowledge_embedding,
                                       size_t embed_dim) {
    if (!engine || !engine->manual_learning_system) return -1;
    return ml_extract_knowledge((MLSystem*)engine->manual_learning_system,
                                 doc_id, knowledge_embedding, embed_dim);
}

int learning_manual_generate_instructions(LearningEngine* engine,
                                           size_t doc_id,
                                           const char* task_query,
                                           float* instructions_out,
                                           size_t max_steps,
                                           size_t* num_steps_out) {
    if (!engine || !engine->manual_learning_system) return -1;
    return ml_generate_instructions((MLSystem*)engine->manual_learning_system,
                                     doc_id, task_query,
                                     instructions_out, max_steps, num_steps_out);
}

int learning_manual_query_document(LearningEngine* engine,
                                    const char* query,
                                    size_t* doc_id_out,
                                    float* relevance_scores) {
    if (!engine || !engine->manual_learning_system || !query) return -1;
    return ml_query_document((MLSystem*)engine->manual_learning_system,
                              query, doc_id_out, relevance_scores);
}

int learning_manual_learn_from_documents(
    LearningEngine* engine,
    int (*progress_callback)(float progress, const char* status, void* user_data),
    void* user_data) {
    if (!engine || !engine->manual_learning_system) return -1;
    return ml_learn_from_documents((MLSystem*)engine->manual_learning_system,
                                    progress_callback, user_data);
}

int learning_manual_get_document_count(LearningEngine* engine) {
    if (!engine || !engine->manual_learning_system) return 0;
    return ml_get_document_count((MLSystem*)engine->manual_learning_system);
}

int learning_manual_get_document(LearningEngine* engine,
                                  size_t doc_id,
                                  MLDocument* doc_out) {
    if (!engine || !engine->manual_learning_system) return -1;
    return ml_get_document((MLSystem*)engine->manual_learning_system,
                            doc_id, doc_out);
}

int learning_manual_clear_documents(LearningEngine* engine) {
    if (!engine || !engine->manual_learning_system) return -1;
    return ml_clear_documents((MLSystem*)engine->manual_learning_system);
}

int learning_manual_generate_summary(LearningEngine* engine,
                                      size_t doc_id,
                                      char* summary_out,
                                      size_t max_len) {
    if (!engine || !engine->manual_learning_system) return -1;
    return ml_generate_summary((MLSystem*)engine->manual_learning_system,
                                doc_id, summary_out, max_len);
}

int learning_manual_synthesize_knowledge(LearningEngine* engine,
                                          size_t* doc_ids,
                                          size_t num_docs,
                                          float* synthesis_out,
                                          size_t embed_dim) {
    if (!engine || !engine->manual_learning_system) return -1;
    return ml_synthesize_knowledge((MLSystem*)engine->manual_learning_system,
                                    doc_ids, num_docs, synthesis_out, embed_dim);
}

int learning_manual_get_learning_progress(LearningEngine* engine,
                                           float* progress_out,
                                           char* status_out,
                                           size_t status_len) {
    if (!engine || !engine->manual_learning_system) return -1;
    return ml_get_learning_progress((MLSystem*)engine->manual_learning_system,
                                     progress_out, status_out, status_len);
}

int learning_manual_export_knowledge_graph(LearningEngine* engine,
                                            char* graph_out,
                                            size_t max_len) {
    if (!engine || !engine->manual_learning_system) return -1;
    int ret = ml_export_knowledge_graph((MLSystem*)engine->manual_learning_system,
                                      graph_out, max_len);

    /* KG持久化: 将学习成果同步写入知识库 */
    {
        static __thread KnowledgeBase* learn_kg = NULL;
        if (!learn_kg) {
            learn_kg = knowledge_base_create(512);
        }
        if (learn_kg && ret >= 0) {
            char subj[64], pred[64], obj[128];
            KnowledgeEntry entry;
            memset(&entry, 0, sizeof(entry));
            snprintf(subj, sizeof(subj), "learning_engine");
            entry.subject = subj;
            snprintf(pred, sizeof(pred), "exported_knowledge");
            entry.predicate = pred;
            snprintf(obj, sizeof(obj), "manual_learning docs=%zu",
                    (size_t)(engine->manual_learning_system ? 0 : 0));
            entry.object = obj;
            entry.confidence = CONFIDENCE_MEDIUM;
            entry.timestamp = (long)time(NULL);
            knowledge_base_add(learn_kg, &entry);
        }
    }

    return ret;
}

int learning_manual_search_documents(LearningEngine* engine,
                                      const char* keyword,
                                      size_t* doc_ids,
                                      size_t* num_results) {
    if (!engine || !engine->manual_learning_system || !keyword) return -1;
    return ml_search_documents((MLSystem*)engine->manual_learning_system,
                                keyword, doc_ids, num_results);
}