/**
 * @file swarm_intelligence.c
 * @brief 群体智能系统实现
 * 
 * 实现多种群体智能算法，用于多机器人协调、优化和决策。
 *  ，提供完整的群体智能算法实现。
 */

#include "selflnn/multisystem/swarm_intelligence.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/laplace.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

/* ========== 内部数据结构定义 ========== */

/* ========== 停滞检测锁（保护函数内static变量） ========== */
#ifdef _WIN32
static CRITICAL_SECTION g_swarm_stag_lock;
static int g_swarm_stag_lock_init = 0;
static void swarm_stag_lock_init_func(void) {
    if (!g_swarm_stag_lock_init) {
        InitializeCriticalSection(&g_swarm_stag_lock);
        g_swarm_stag_lock_init = 1;
    }
}
#define SWARM_STAG_LOCK() do { swarm_stag_lock_init_func(); EnterCriticalSection(&g_swarm_stag_lock); } while(0)
#define SWARM_STAG_UNLOCK() LeaveCriticalSection(&g_swarm_stag_lock)
#else
static pthread_mutex_t g_swarm_stag_lock = PTHREAD_MUTEX_INITIALIZER;
#define SWARM_STAG_LOCK() pthread_mutex_lock(&g_swarm_stag_lock)
#define SWARM_STAG_UNLOCK() pthread_mutex_unlock(&g_swarm_stag_lock)
#endif

/** @brief 群体内部结构 */
struct Swarm {
    SwarmConfig config;                /**< 配置参数 */
    
    /* 群体数据 */
    SwarmIndividual* individuals;      /**< 个体数组 */
    float* global_best_position;       /**< 全局最佳位置 */
    float global_best_fitness;         /**< 全局最佳适应度 */
    int global_best_id;                /**< 全局最佳个体ID */
    
    /* 状态信息 */
    int current_iteration;             /**< 当前迭代次数 */
    int is_initialized;                /**< 是否已初始化 */
    int is_converged;                  /**< 是否已收敛 */
    int convergence_reason;            /**< 收敛原因 */
    
    /* 算法特定数据 */
    void* algorithm_data;              /**< 算法特定数据指针 */
    
    /* 性能统计 */
    clock_t start_time;                /**< 开始时间 */
    float total_computation_time_ms;   /**< 总计算时间（毫秒） */
    
    /* 并行计算 */
#ifdef _WIN32
    HANDLE* threads;                   /**< 线程句柄数组 */
#else
    pthread_t* threads;                /**< 线程ID数组 */
#endif
    int* thread_indices;               /**< 线程索引数组 */
    void* thread_data;                 /**< 线程数据 */
};

/** @brief PSO算法特定数据 */
typedef struct {
    float* personal_best_positions;    /**< 个体历史最佳位置数组 */
    float* personal_best_fitnesses;    /**< 个体历史最佳适应度数组 */
    float* neighborhood_best_positions; /**< 邻居最佳位置数组 */
    float* neighborhood_best_fitnesses; /**< 邻居最佳适应度数组 */
    int* neighborhood_indices;         /**< 邻居索引数组 */
    float inertia_weight;              /**< 当前惯性权重 */
    float inertia_weight_max;          /**< 惯性权重最大值 */
    float inertia_weight_min;          /**< 惯性权重最小值 */
    float cognitive_weight;            /**< 认知权重 */
    float social_weight;               /**< 社会权重 */
    float velocity_max;                /**< 最大速度 */
} PSOData;

/** @brief ACO算法特定数据 */
typedef struct {
    float** pheromone_matrix;          /**< 信息素矩阵 */
    float** heuristic_matrix;          /**< 启发式矩阵 */
    float** probability_matrix;        /**< 概率矩阵 */
    int** ant_paths;                   /**< 蚂蚁路径数组 */
    float* ant_path_lengths;           /**< 蚂蚁路径长度数组 */
    float pheromone_evaporation;       /**< 信息素蒸发率 */
    float pheromone_deposit;           /**< 信息素沉积量 */
    float alpha;                       /**< 信息素重要性 */
    float beta;                        /**< 启发式重要性 */
    float q0;                          /**< 探索参数 */
    int num_ants;                      /**< 蚂蚁数量 */
    int path_length;                   /**< 路径长度 */
} ACOData;

/** @brief ABC算法特定数据 */
typedef struct {
    float* trial_counts;               /**< 试验计数数组 */
    float* probabilities;              /**< 选择概率数组 */
    int* employed_bees;                /**< 雇佣蜂数组 */
    int* onlooker_bees;                /**< 观察蜂数组 */
    int* scout_bees;                   /**< 侦查蜂数组 */
    float limit;                       /**< 试验限制 */
    float abandonment_probability;     /**< 放弃概率 */
} ABCData;

/** @brief FSS（鱼群算法）算法特定数据 */
typedef struct {
    float* individual_step_sizes;      /**< 个体步长数组 */
    float* weights;                    /**< 权重数组 */
    float* prev_positions;             /**< 前一步位置数组 */
    float step_size_initial;           /**< 初始步长 */
    float step_size_final;             /**< 最终步长 */
    float weight_min;                  /**< 最小权重 */
    float weight_max;                  /**< 最大权重 */
    float weight_scale;                /**< 权重缩放因子 */
    float instictive_acceleration;     /**< 本能加速度累积 */
    float volitive_acceleration;       /**< 集体加速度累积 */
    float barycenter[3];               /**< 群体质心（最多3维，扩展时使用） */
    int barycenter_dim;                /**< 质心维度 */
} FSSData;

/** @brief BAT（蝙蝠算法）算法特定数据 */
typedef struct {
    float* frequencies;                /**< 频率数组 */
    float* velocities;                 /**< 速度数组 */
    float* loudness;                   /**< 响度数组 */
    float* pulse_rates;               /**< 脉冲发射率数组 */
    float* pulse_rates_initial;        /**< 初始脉冲发射率数组 */
    float* best_positions;             /**< 个体最佳位置数组 */
    float* best_fitnesses;             /**< 个体最佳适应度数组 */
    float frequency_min;               /**< 最小频率 */
    float frequency_max;               /**< 最大频率 */
    float loudness_initial;            /**< 初始响度 */
    float pulse_rate_initial;          /**< 初始脉冲率 */
    float alpha;                       /**< 响度衰减系数 */
    float gamma;                       /**< 脉冲率增强系数 */
} BATData;

/** @brief Firefly（萤火虫算法）算法特定数据 */
typedef struct {
    float* brightness;                 /**< 亮度数组 */
    float attractiveness_base;         /**< 基础吸引力 */
    float light_absorption;            /**< 光吸收系数 */
    float randomization_alpha;         /**< 随机化参数 */
    float randomization_decay;         /**< 随机化衰减系数 */
    float max_attraction_distance;     /**< 最大吸引距离 */
} FireflyData;

/** @brief Cuckoo（布谷鸟搜索）算法特定数据 */
typedef struct {
    float* levy_steps;                 /**< Levy步长数组 */
    float* prev_positions;             /**< 前一步位置数组 */
    float* best_positions;             /**< 个体最佳位置数组 */
    float* best_fitnesses;             /**< 个体最佳适应度数组 */
    float step_size;                   /**< 步长缩放因子 */
    float discovery_probability;       /**< 发现概率（丢弃概率） */
    float levy_exponent;               /**< Levy分布指数（β，1<β≤3） */
    float levy_scale;                  /**< Levy分布缩放因子 */
    int use_levy_flights;              /**< 是否使用Levy飞行（1=是，0=否） */
} CuckooData;

/** @brief GWO（灰狼优化器）算法特定数据 */
typedef struct {
    float* alpha_position;             /**< α狼位置 */
    float* beta_position;              /**< β狼位置 */
    float* delta_position;             /**< δ狼位置 */
    float alpha_fitness;               /**< α狼适应度 */
    float beta_fitness;                /**< β狼适应度 */
    float delta_fitness;               /**< δ狼适应度 */
    float a_linear;                    /**< 线性衰减参数a */
    float a_initial;                   /**< a的初始值（通常为2） */
    float a_final;                     /**< a的最终值（通常为0） */
    int alpha_idx;                     /**< α狼个体索引 */
    int beta_idx;                      /**< β狼个体索引 */
    int delta_idx;                     /**< δ狼个体索引 */
} GWOData;

/** @brief WOA（鲸鱼优化算法）算法特定数据 */
typedef struct {
    float* best_positions;             /**< 鲸鱼个体最佳位置数组 */
    float* best_fitnesses;             /**< 鲸鱼个体最佳适应度数组 */
    float spiral_constant;             /**< 螺旋常数b */
    float a_linear;                    /**< 线性衰减参数a */
    float a2_linear;                   /**< 第二线性衰减参数a2 */
    float bubble_net_probability;      /**< 气泡网攻击概率 */
    float leader_position[3];          /**< 领导者位置（最多3维，扩展时使用） */
    int leader_dim;                    /**< 领导者位置维度 */
} WOAData;

/** @brief MARL（多智能体强化学习）算法特定数据 */
typedef struct {
    float* q_table;                    /**< Q表扁平化存储 */
    float* eligibility_traces;         /**< 资格迹 */
    size_t q_table_size;               /**< Q表总大小 */
    int* state_indices;                /**< 状态索引数组 */
    int* action_indices;               /**< 动作索引数组 */
    int* prev_states;                  /**< 前一步状态数组 */
    int* prev_actions;                 /**< 前一步动作数组 */
    float learning_rate;               /**< 学习率α */
    float discount_factor;             /**< 折扣因子γ */
    float exploration_rate;            /**< 探索率ε */
    float exploration_decay;           /**< 探索率衰减 */
    float exploration_min;             /**< 最小探索率 */
    float lambda;                      /**< 资格迹衰减参数λ */
    int num_states;                    /**< 状态数量 */
    int num_actions;                   /**< 动作数量 */
    int use_softmax_policy;            /**< 是否使用Softmax策略（1=是，0=ε-greedy） */
    float temperature;                 /**< Softmax温度参数 */
} MARLData;

/* ========== 静态函数声明 ========== */

static Swarm* swarm_create_internal(const SwarmConfig* config);
static int swarm_initialize_internal(Swarm* swarm, const float* initial_positions);
static void swarm_free_internal(Swarm* swarm);
static int swarm_validate_config(const SwarmConfig* config);
static float swarm_calculate_fitness(Swarm* swarm, int individual_id);
static void swarm_update_velocity_position_pso(Swarm* swarm, int individual_id);
static void swarm_update_pheromone_aco(Swarm* swarm);
static void swarm_select_neighbors(Swarm* swarm, int individual_id, int* neighbors, int* num_neighbors);
static float swarm_calculate_diversity(Swarm* swarm, float** positions);
static void swarm_apply_constraints(Swarm* swarm, int individual_id);
static void swarm_log_state(Swarm* swarm, int iteration);
static int swarm_check_convergence(Swarm* swarm);
static void swarm_update_global_best(Swarm* swarm);
static void swarm_generate_initial_positions(Swarm* swarm, float* positions);

/* PSO算法函数 */
static PSOData* pso_data_create(const SwarmConfig* config);
static void pso_data_free(PSOData* data);
static void pso_initialize(Swarm* swarm);
static void pso_iterate(Swarm* swarm);

/* ACO算法函数 */
static ACOData* aco_data_create(const SwarmConfig* config);
static void aco_data_free(ACOData* data);
static void aco_initialize(Swarm* swarm);
static void aco_iterate(Swarm* swarm);
static void aco_construct_solution(Swarm* swarm, int ant_id);

/* ABC算法函数 */
static ABCData* abc_data_create(const SwarmConfig* config);
static void abc_data_free(ABCData* data);
static void abc_initialize(Swarm* swarm);
static void abc_iterate(Swarm* swarm);
static void abc_employed_phase(Swarm* swarm);
static void abc_onlooker_phase(Swarm* swarm);
static void abc_scout_phase(Swarm* swarm);

/* FSS算法函数 */
static FSSData* fss_data_create(const SwarmConfig* config);
static void fss_data_free(FSSData* data);
static void fss_initialize(Swarm* swarm);
static void fss_iterate(Swarm* swarm);

/* BAT算法函数 */
static BATData* bat_data_create(const SwarmConfig* config);
static void bat_data_free(BATData* data);
static void bat_initialize(Swarm* swarm);
static void bat_iterate(Swarm* swarm);

/* Firefly算法函数 */
static FireflyData* firefly_data_create(const SwarmConfig* config);
static void firefly_data_free(FireflyData* data);
static void firefly_initialize(Swarm* swarm);
static void firefly_iterate(Swarm* swarm);

/* Cuckoo算法函数 */
static CuckooData* cuckoo_data_create(const SwarmConfig* config);
static void cuckoo_data_free(CuckooData* data);
static void cuckoo_initialize(Swarm* swarm);
static void cuckoo_iterate(Swarm* swarm);

/* GWO算法函数 */
static GWOData* gwo_data_create(const SwarmConfig* config);
static void gwo_data_free(GWOData* data);
static void gwo_initialize(Swarm* swarm);
static void gwo_iterate(Swarm* swarm);

/* WOA算法函数 */
static WOAData* woa_data_create(const SwarmConfig* config);
static void woa_data_free(WOAData* data);
static void woa_initialize(Swarm* swarm);
static void woa_iterate(Swarm* swarm);

/* MARL算法函数 */
static MARLData* marl_data_create(const SwarmConfig* config);
static void marl_data_free(MARLData* data);
static void marl_initialize(Swarm* swarm);
static void marl_iterate(Swarm* swarm);

/* 通用工具函数 */
static float random_float(float min, float max);
static int random_int(int min, int max);
static void clamp_position(float* position, const float* lower_bounds, const float* upper_bounds, int dimensions);
static float euclidean_distance(const float* a, const float* b, int dimensions);
static void copy_position(float* dest, const float* src, int dimensions);

/* ========== 公共函数实现 ========== */

/**
 * @brief 创建群体智能系统
 */
Swarm* swarm_create(const SwarmConfig* config) {
    if (config == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建群体智能系统：配置参数为空");
        return NULL;
    }
    
    /* 验证配置 */
    if (swarm_validate_config(config) != 0) {
        return NULL;
    }
    
    /* 创建群体 */
    Swarm* swarm = swarm_create_internal(config);
    if (swarm == NULL) {
        return NULL;
    }
    
    /* 初始化算法特定数据 */
    switch (config->algorithm_type) {
        case SWARM_ALGORITHM_PSO:
            swarm->algorithm_data = pso_data_create(config);
            break;
        case SWARM_ALGORITHM_ACO:
            swarm->algorithm_data = aco_data_create(config);
            break;
        case SWARM_ALGORITHM_ABC:
            swarm->algorithm_data = abc_data_create(config);
            break;
        case SWARM_ALGORITHM_FSS:
            swarm->algorithm_data = fss_data_create(config);
            break;
        case SWARM_ALGORITHM_BAT:
            swarm->algorithm_data = bat_data_create(config);
            break;
        case SWARM_ALGORITHM_FIREFLY:
            swarm->algorithm_data = firefly_data_create(config);
            break;
        case SWARM_ALGORITHM_CUCKOO:
            swarm->algorithm_data = cuckoo_data_create(config);
            break;
        case SWARM_ALGORITHM_GWO:
            swarm->algorithm_data = gwo_data_create(config);
            break;
        case SWARM_ALGORITHM_WOA:
            swarm->algorithm_data = woa_data_create(config);
            break;
        case SWARM_ALGORITHM_MARL:
            swarm->algorithm_data = marl_data_create(config);
            break;
        default:
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "创建群体智能系统：未知算法类型");
            swarm_free_internal(swarm);
            return NULL;
    }
    
    if (swarm->algorithm_data == NULL) {
        swarm_free_internal(swarm);
        return NULL;
    }
    
    return swarm;
}

/**
 * @brief 创建群体内部结构
 */
static Swarm* swarm_create_internal(const SwarmConfig* config) {
    Swarm* swarm = (Swarm*)safe_malloc(sizeof(Swarm));
    if (swarm == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建群体智能系统：内存分配失败");
        return NULL;
    }
    
    /* 复制配置 */
    memcpy(&swarm->config, config, sizeof(SwarmConfig));
    
    /* 分配个体数组 */
    swarm->individuals = (SwarmIndividual*)safe_malloc(sizeof(SwarmIndividual) * config->swarm_size);
    if (swarm->individuals == NULL) {
        safe_free((void**)&swarm);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建群体智能系统：个体数组内存分配失败");
        return NULL;
    }
    
    /* 初始化个体 */
    for (int i = 0; i < config->swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        
        ind->position = (float*)safe_malloc(sizeof(float) * config->dimensions);
        ind->velocity = (float*)safe_malloc(sizeof(float) * config->dimensions);
        ind->best_position = (float*)safe_malloc(sizeof(float) * config->dimensions);
        
        if (ind->position == NULL || ind->velocity == NULL || ind->best_position == NULL) {
            /* 清理已分配的内存 */
            for (int j = 0; j <= i; j++) {
                SwarmIndividual* ind2 = &swarm->individuals[j];
                safe_free((void**)&ind2->position);
                safe_free((void**)&ind2->velocity);
                safe_free((void**)&ind2->best_position);
            }
            safe_free((void**)&swarm->individuals);
            safe_free((void**)&swarm);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建群体智能系统：个体内存分配失败");
            return NULL;
        }
        
        ind->id = i;
        ind->is_active = 1;
        ind->best_fitness = FLT_MAX;
        ind->current_fitness = FLT_MAX;
        ind->custom_data = NULL;
        
        /* 初始化为零 */
        memset(ind->position, 0, sizeof(float) * config->dimensions);
        memset(ind->velocity, 0, sizeof(float) * config->dimensions);
        memset(ind->best_position, 0, sizeof(float) * config->dimensions);
    }
    
    /* 分配全局最佳位置 */
    swarm->global_best_position = (float*)safe_malloc(sizeof(float) * config->dimensions);
    if (swarm->global_best_position == NULL) {
        /* 清理已分配的内存 */
        for (int i = 0; i < config->swarm_size; i++) {
            SwarmIndividual* ind = &swarm->individuals[i];
            safe_free((void**)&ind->position);
            safe_free((void**)&ind->velocity);
            safe_free((void**)&ind->best_position);
        }
        safe_free((void**)&swarm->individuals);
        safe_free((void**)&swarm);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建群体智能系统：全局最佳位置内存分配失败");
        return NULL;
    }
    
    /* 初始化状态 */
    swarm->global_best_fitness = FLT_MAX;
    swarm->global_best_id = -1;
    swarm->current_iteration = 0;
    swarm->is_initialized = 0;
    swarm->is_converged = 0;
    swarm->convergence_reason = -1;
    swarm->algorithm_data = NULL;
    swarm->start_time = 0;
    swarm->total_computation_time_ms = 0.0f;
    
    /* 初始化并行计算相关 */
    swarm->threads = NULL;
    swarm->thread_indices = NULL;
    swarm->thread_data = NULL;
    
    if (config->enable_parallel_evaluation && config->num_threads > 1) {
        /* 分配线程索引数组 */
        swarm->thread_indices = (int*)safe_malloc(sizeof(int) * config->swarm_size);
        if (swarm->thread_indices == NULL) {
            /* 清理已分配的内存 */
            safe_free((void**)&swarm->global_best_position);
            for (int i = 0; i < config->swarm_size; i++) {
                SwarmIndividual* ind = &swarm->individuals[i];
                safe_free((void**)&ind->position);
                safe_free((void**)&ind->velocity);
                safe_free((void**)&ind->best_position);
            }
            safe_free((void**)&swarm->individuals);
            safe_free((void**)&swarm);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建群体智能系统：线程索引内存分配失败");
            return NULL;
        }
        
        /* 分配线程句柄数组 */
#ifdef _WIN32
        swarm->threads = (HANDLE*)safe_malloc(sizeof(HANDLE) * config->num_threads);
#else
        swarm->threads = (pthread_t*)safe_malloc(sizeof(pthread_t) * config->num_threads);
#endif
        if (swarm->threads == NULL) {
            safe_free((void**)&swarm->thread_indices);
            safe_free((void**)&swarm->global_best_position);
            for (int i = 0; i < config->swarm_size; i++) {
                SwarmIndividual* ind = &swarm->individuals[i];
                safe_free((void**)&ind->position);
                safe_free((void**)&ind->velocity);
                safe_free((void**)&ind->best_position);
            }
            safe_free((void**)&swarm->individuals);
            safe_free((void**)&swarm);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "创建群体智能系统：线程句柄内存分配失败");
            return NULL;
        }
    }
    
    return swarm;
}

/**
 * @brief 验证配置参数
 */
static int swarm_validate_config(const SwarmConfig* config) {
    if (config->swarm_size <= 0 || config->swarm_size > SWARM_MAX_SIZE) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "验证配置：群体大小无效（范围：1-%d）", SWARM_MAX_SIZE);
        return -1;
    }
    
    if (config->dimensions <= 0 || config->dimensions > SWARM_MAX_DIMENSIONS) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "验证配置：维度数无效（范围：1-%d）", SWARM_MAX_DIMENSIONS);
        return -1;
    }
    
    if (config->max_iterations <= 0 || config->max_iterations > SWARM_MAX_ITERATIONS) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "验证配置：最大迭代次数无效（范围：1-%d）", SWARM_MAX_ITERATIONS);
        return -1;
    }
    
    if (config->fitness_func == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "验证配置：适应度函数为空");
        return -1;
    }
    
    if (config->lower_bounds == NULL || config->upper_bounds == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "验证配置：边界数组为空");
        return -1;
    }
    
    /* 检查边界有效性 */
    for (int i = 0; i < config->dimensions; i++) {
        if (config->lower_bounds[i] >= config->upper_bounds[i]) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "验证配置：下界[%d] >= 上界[%d]", i, i);
            return -1;
        }
    }
    
    return 0;
}

/**
 * @brief 释放群体智能系统
 */
void swarm_free(Swarm* swarm) {
    if (swarm == NULL) {
        return;
    }
    
    swarm_free_internal(swarm);
}

/**
 * @brief 释放群体内部结构
 */
static void swarm_free_internal(Swarm* swarm) {
    if (swarm->individuals != NULL) {
        for (int i = 0; i < swarm->config.swarm_size; i++) {
            SwarmIndividual* ind = &swarm->individuals[i];
            safe_free((void**)&ind->position);
            safe_free((void**)&ind->velocity);
            safe_free((void**)&ind->best_position);
            if (ind->custom_data != NULL) {
                safe_free((void**)&ind->custom_data);
            }
        }
        safe_free((void**)&swarm->individuals);
    }
    
    safe_free((void**)&swarm->global_best_position);
    
    /* 释放算法特定数据 */
    if (swarm->algorithm_data != NULL) {
        switch (swarm->config.algorithm_type) {
            case SWARM_ALGORITHM_PSO:
                pso_data_free((PSOData*)swarm->algorithm_data);
                break;
            case SWARM_ALGORITHM_ACO:
                aco_data_free((ACOData*)swarm->algorithm_data);
                break;
            case SWARM_ALGORITHM_ABC:
                abc_data_free((ABCData*)swarm->algorithm_data);
                break;
            case SWARM_ALGORITHM_FSS:
                fss_data_free((FSSData*)swarm->algorithm_data);
                break;
            case SWARM_ALGORITHM_BAT:
                bat_data_free((BATData*)swarm->algorithm_data);
                break;
            case SWARM_ALGORITHM_FIREFLY:
                firefly_data_free((FireflyData*)swarm->algorithm_data);
                break;
            case SWARM_ALGORITHM_CUCKOO:
                cuckoo_data_free((CuckooData*)swarm->algorithm_data);
                break;
            case SWARM_ALGORITHM_GWO:
                gwo_data_free((GWOData*)swarm->algorithm_data);
                break;
            case SWARM_ALGORITHM_WOA:
                woa_data_free((WOAData*)swarm->algorithm_data);
                break;
            case SWARM_ALGORITHM_MARL:
                marl_data_free((MARLData*)swarm->algorithm_data);
                break;
            default:
                safe_free((void**)&swarm->algorithm_data);
                break;
        }
    }
    
    /* 释放并行计算相关资源 */
    if (swarm->thread_indices != NULL) {
        safe_free((void**)&swarm->thread_indices);
    }
    
    if (swarm->threads != NULL) {
        safe_free((void**)&swarm->threads);
    }
    
    if (swarm->thread_data != NULL) {
        safe_free((void**)&swarm->thread_data);
    }
    
    safe_free((void**)&swarm);
}

/**
 * @brief 初始化群体
 */
int swarm_initialize(Swarm* swarm, const float* initial_positions) {
    if (swarm == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "初始化群体：群体指针为空");
        return -1;
    }
    
    if (swarm->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_ALREADY_INITIALIZED, __func__, __FILE__, __LINE__,
                              "初始化群体：群体已初始化");
        return -1;
    }
    
    int result = swarm_initialize_internal(swarm, initial_positions);
    if (result == 0) {
        swarm->is_initialized = 1;
        swarm->start_time = clock();
    }
    
    return result;
}

/**
 * @brief 初始化群体内部实现
 */
static int swarm_initialize_internal(Swarm* swarm, const float* initial_positions) {
    const SwarmConfig* config = &swarm->config;
    
    /* 生成或使用初始位置 */
    float* positions = NULL;
    if (initial_positions != NULL) {
        positions = (float*)initial_positions;
    } else {
        /* 生成随机初始位置 */
        positions = (float*)safe_malloc(sizeof(float) * config->swarm_size * config->dimensions);
        if (positions == NULL) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "初始化群体：初始位置内存分配失败");
            return -1;
        }
        swarm_generate_initial_positions(swarm, positions);
    }
    
    /* 初始化个体位置和速度 */
    for (int i = 0; i < config->swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        
        /* 设置初始位置 */
        copy_position(ind->position, &positions[i * config->dimensions], config->dimensions);
        copy_position(ind->best_position, ind->position, config->dimensions);
        
        /* 生成随机初始速度（在边界范围内） */
        for (int d = 0; d < config->dimensions; d++) {
            float range = config->upper_bounds[d] - config->lower_bounds[d];
            ind->velocity[d] = random_float(-range * 0.1f, range * 0.1f);
        }
        
        /* 应用约束 */
        swarm_apply_constraints(swarm, i);
        
        /* 计算初始适应度 */
        ind->current_fitness = swarm_calculate_fitness(swarm, i);
        ind->best_fitness = ind->current_fitness;
        
        /* 更新全局最佳 */
        if (ind->current_fitness < swarm->global_best_fitness) {
            swarm->global_best_fitness = ind->current_fitness;
            swarm->global_best_id = i;
            copy_position(swarm->global_best_position, ind->position, config->dimensions);
        }
    }
    
    /* 如果分配了内存，则释放 */
    if (initial_positions == NULL && positions != NULL) {
        safe_free((void**)&positions);
    }
    
    /* 算法特定初始化 */
    switch (config->algorithm_type) {
        case SWARM_ALGORITHM_PSO:
            pso_initialize(swarm);
            break;
        case SWARM_ALGORITHM_ACO:
            aco_initialize(swarm);
            break;
        case SWARM_ALGORITHM_ABC:
            abc_initialize(swarm);
            break;
        default:
            /* 其他算法 */
            break;
    }
    
    /* 记录初始状态 */
    if (config->enable_logging) {
        swarm_log_state(swarm, 0);
    }
    
    return 0;
}

/**
 * @brief 生成初始位置
 */
static void swarm_generate_initial_positions(Swarm* swarm, float* positions) {
    const SwarmConfig* config = &swarm->config;
    
    for (int i = 0; i < config->swarm_size; i++) {
        for (int d = 0; d < config->dimensions; d++) {
            positions[i * config->dimensions + d] = 
                random_float(config->lower_bounds[d], config->upper_bounds[d]);
        }
    }
}

/**
 * @brief 计算适应度
 */
static float swarm_calculate_fitness(Swarm* swarm, int individual_id) {
    if (individual_id < 0 || individual_id >= swarm->config.swarm_size) {
        return FLT_MAX;
    }
    
    SwarmIndividual* ind = &swarm->individuals[individual_id];
    return swarm->config.fitness_func(ind->position, swarm->config.dimensions, 
                                     swarm->config.user_data);
}

/**
 * @brief 应用约束
 */
static void swarm_apply_constraints(Swarm* swarm, int individual_id) {
    if (individual_id < 0 || individual_id >= swarm->config.swarm_size) {
        return;
    }
    
    SwarmIndividual* ind = &swarm->individuals[individual_id];
    
    /* 应用边界约束 */
    clamp_position(ind->position, swarm->config.lower_bounds, 
                  swarm->config.upper_bounds, swarm->config.dimensions);
    
    /* 应用用户自定义约束 */
    if (swarm->config.constraint_func != NULL) {
        swarm->config.constraint_func(ind->position, swarm->config.dimensions, 
                                     swarm->config.user_data);
    }
}

/**
 * @brief 执行群体智能优化
 */
int swarm_optimize(Swarm* swarm, SwarmResult* result) {
    if (swarm == NULL || result == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "执行优化：参数为空");
        return -1;
    }
    
    if (!swarm->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "执行优化：群体未初始化");
        return -1;
    }
    
    /* 初始化结果 */
    swarm_result_init(result);
    
    /* 记录开始时间 */
    clock_t start_time = clock();
    
    /* 主优化循环 */
    while (!swarm->is_converged && 
           swarm->current_iteration < swarm->config.max_iterations) {
        
        /* 执行单次迭代 */
        if (swarm_iterate(swarm) != 0) {
            strcpy(result->error_message, "迭代失败");
            return -1;
        }
        
        /* 检查收敛 */
        swarm->is_converged = swarm_check_convergence(swarm);
        
        /* 更新迭代计数 */
        swarm->current_iteration++;
        
        /* 记录状态 */
        if (swarm->config.enable_logging && 
            (swarm->current_iteration % swarm->config.log_frequency == 0)) {
            swarm_log_state(swarm, swarm->current_iteration);
        }
        
        /* 调用迭代回调 */
        if (swarm->config.iteration_callback != NULL) {
            swarm->config.iteration_callback(swarm, swarm->current_iteration, 
                                           swarm->config.user_data);
        }
    }
    
    /* 记录结束时间 */
    clock_t end_time = clock();
    float total_time_ms = (float)(end_time - start_time) * 1000.0f / CLOCKS_PER_SEC;
    swarm->total_computation_time_ms = total_time_ms;
    
    /* 设置收敛原因 */
    if (swarm->is_converged) {
        swarm->convergence_reason = swarm_check_convergence(swarm);
    } else {
        swarm->convergence_reason = SWARM_CONVERGENCE_MAX_ITERATIONS;
    }
    
    /* 填充结果 */
    result->best_fitness = swarm->global_best_fitness;
    result->iterations = swarm->current_iteration;
    result->total_time_ms = total_time_ms;
    result->convergence_reason = swarm->convergence_reason;
    
    /* 复制最佳位置 */
    if (result->best_position == NULL) {
        result->best_position = (float*)safe_malloc(sizeof(float) * swarm->config.dimensions);
        if (result->best_position == NULL) {
            strcpy(result->error_message, "分配最佳位置内存失败");
            return -1;
        }
    }
    copy_position(result->best_position, swarm->global_best_position, swarm->config.dimensions);
    
    /* 获取最终状态 */
    if (swarm_get_state(swarm, &result->final_state) != 0) {
        strcpy(result->error_message, "获取最终状态失败");
        return -1;
    }
    
    /* 调用收敛回调 */
    if (swarm->config.convergence_callback != NULL) {
        swarm->config.convergence_callback(swarm, swarm->current_iteration, 
                                         swarm->config.user_data);
    }
    
    return 0;
}

/**
 * @brief 执行单次迭代
 */
int swarm_iterate(Swarm* swarm) {
    if (swarm == NULL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "执行迭代：群体指针为空");
        return -1;
    }
    
    if (!swarm->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "执行迭代：群体未初始化");
        return -1;
    }
    
    /* 根据算法类型执行迭代 */
    switch (swarm->config.algorithm_type) {
        case SWARM_ALGORITHM_PSO:
            pso_iterate(swarm);
            break;
        case SWARM_ALGORITHM_ACO:
            aco_iterate(swarm);
            break;
        case SWARM_ALGORITHM_ABC:
            abc_iterate(swarm);
            break;
        case SWARM_ALGORITHM_FSS:
            fss_iterate(swarm);
            break;
        case SWARM_ALGORITHM_BAT:
            bat_iterate(swarm);
            break;
        case SWARM_ALGORITHM_FIREFLY:
            firefly_iterate(swarm);
            break;
        case SWARM_ALGORITHM_CUCKOO:
            cuckoo_iterate(swarm);
            break;
        case SWARM_ALGORITHM_GWO:
            gwo_iterate(swarm);
            break;
        case SWARM_ALGORITHM_WOA:
            woa_iterate(swarm);
            break;
        case SWARM_ALGORITHM_MARL:
            marl_iterate(swarm);
            break;
        default:
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "执行迭代：未知算法类型");
            return -1;
    }
    
    /* 更新全局最佳 */
    swarm_update_global_best(swarm);
    
    return 0;
}

/**
 * @brief 更新全局最佳解
 */
static void swarm_update_global_best(Swarm* swarm) {
    for (int i = 0; i < swarm->config.swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        
        if (ind->current_fitness < ind->best_fitness) {
            ind->best_fitness = ind->current_fitness;
            copy_position(ind->best_position, ind->position, swarm->config.dimensions);
        }
        
        if (ind->current_fitness < swarm->global_best_fitness) {
            swarm->global_best_fitness = ind->current_fitness;
            swarm->global_best_id = i;
            copy_position(swarm->global_best_position, ind->position, swarm->config.dimensions);
        }
    }
}

/**
 * @brief 检查收敛
 */
static int swarm_check_convergence(Swarm* swarm) {
    const SwarmConfig* config = &swarm->config;
    
    switch (config->convergence_condition) {
        case SWARM_CONVERGENCE_MAX_ITERATIONS:
            if (swarm->current_iteration >= config->max_iterations) {
                return SWARM_CONVERGENCE_MAX_ITERATIONS;
            }
            break;
            
        case SWARM_CONVERGENCE_FITNESS_THRESHOLD:
            if (swarm->global_best_fitness <= config->fitness_threshold) {
                return SWARM_CONVERGENCE_FITNESS_THRESHOLD;
            }
            break;
            
        case SWARM_CONVERGENCE_POSITION_CHANGE: {
            float total_movement = 0.0f;
            int affected_count = 0;
            for (int i = 0; i < swarm->config.swarm_size; i++) {
                SwarmIndividual* ind = &swarm->individuals[i];
                float movement = 0.0f;
                for (int d = 0; d < swarm->config.dimensions; d++) {
                    float diff = ind->position[d] - ind->best_position[d];
                    movement += diff * diff;
                }
                movement = sqrtf(movement / (float)swarm->config.dimensions);
                total_movement += movement;
                if (movement < 0.001f) affected_count++;
            }
            float avg_movement = total_movement / (float)swarm->config.swarm_size;
            float convergence_ratio = (float)affected_count / (float)swarm->config.swarm_size;
            if (avg_movement < 0.0001f && convergence_ratio > 0.9f) {
                return SWARM_CONVERGENCE_POSITION_CHANGE;
            }
            break;
        }
            
        case SWARM_CONVERGENCE_VELOCITY_CHANGE: {
            float total_velocity = 0.0f;
            float max_velocity = 0.0f;
            int near_zero_count = 0;
            for (int i = 0; i < swarm->config.swarm_size; i++) {
                SwarmIndividual* ind = &swarm->individuals[i];
                float vel_magnitude = 0.0f;
                for (int d = 0; d < swarm->config.dimensions; d++) {
                    vel_magnitude += ind->velocity[d] * ind->velocity[d];
                }
                vel_magnitude = sqrtf(vel_magnitude / (float)swarm->config.dimensions);
                total_velocity += vel_magnitude;
                if (vel_magnitude > max_velocity) max_velocity = vel_magnitude;
                if (vel_magnitude < 0.0001f) near_zero_count++;
            }
            float avg_velocity = total_velocity / (float)swarm->config.swarm_size;
            float zero_ratio = (float)near_zero_count / (float)swarm->config.swarm_size;
            if (avg_velocity < 0.0005f && zero_ratio > 0.8f && max_velocity < 0.001f) {
                return SWARM_CONVERGENCE_VELOCITY_CHANGE;
            }
            break;
        }
            
        case SWARM_CONVERGENCE_STAGNATION: {
            static float prev_best_fitness = FLT_MAX;
            static int stagnation_counter = 0;
            SWARM_STAG_LOCK();
            if (swarm->current_iteration == 0) {
                prev_best_fitness = swarm->global_best_fitness;
                stagnation_counter = 0;
            }
            float improvement = prev_best_fitness - swarm->global_best_fitness;
            if (improvement > 0.0f && improvement < 0.0001f) {
                stagnation_counter++;
            } else if (improvement > 0.0001f) {
                stagnation_counter = 0;
                prev_best_fitness = swarm->global_best_fitness;
            } else {
                stagnation_counter++;
            }
            float diversity = 0.0f;
            float* center = (float*)safe_calloc(swarm->config.dimensions, sizeof(float));
            if (center) {
                for (int i = 0; i < swarm->config.swarm_size; i++) {
                    for (int d = 0; d < swarm->config.dimensions; d++) {
                        center[d] += swarm->individuals[i].position[d];
                    }
                }
                for (int d = 0; d < swarm->config.dimensions; d++) {
                    center[d] /= (float)swarm->config.swarm_size;
                }
                for (int i = 0; i < swarm->config.swarm_size; i++) {
                    for (int d = 0; d < swarm->config.dimensions; d++) {
                        float diff = swarm->individuals[i].position[d] - center[d];
                        diversity += diff * diff;
                    }
                }
                diversity = sqrtf(diversity / (float)(swarm->config.swarm_size * swarm->config.dimensions));
                safe_free((void**)&center);
            }
            int stagnation_threshold = config ? config->max_iterations / 10 : 20;
            if (stagnation_threshold < 5) stagnation_threshold = 5;
            int result = 0;
            if (stagnation_counter >= stagnation_threshold && diversity < 0.01f) {
                stagnation_counter = 0;
                result = SWARM_CONVERGENCE_STAGNATION;
            }
            SWARM_STAG_UNLOCK();
            if (result) return result;
            break;
        }
    }
    
    return 0; /* 未收敛 */
}

/**
 * @brief 记录状态
 */
static void swarm_log_state(Swarm* swarm, int iteration) {
    if (swarm == NULL || !swarm->config.enable_logging) {
        return;
    }
    
    SwarmState state;
    if (swarm_get_state(swarm, &state) == 0) {
        log_info("群体智能迭代 %d: 最佳适应度=%f, 平均适应度=%f, 多样性=%f", 
                        iteration, state.best_fitness, state.average_fitness, 
                        state.position_diversity);
    }
}

/**
 * @brief 获取群体状态
 */
int swarm_get_state(const Swarm* swarm, SwarmState* state) {
    if (swarm == NULL || state == NULL) {
        return -1;
    }
    
    state->iteration = swarm->current_iteration;
    state->best_fitness = swarm->global_best_fitness;
    state->average_fitness = 0.0f;
    state->fitness_std_dev = 0.0f;
    state->position_diversity = 0.0f;
    state->velocity_diversity = 0.0f;
    state->is_converged = swarm->is_converged;
    state->convergence_reason = swarm->convergence_reason;
    state->computation_time_ms = swarm->total_computation_time_ms;
    
    /* 计算平均适应度和标准差 */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < swarm->config.swarm_size; i++) {
        float fitness = swarm->individuals[i].current_fitness;
        sum += fitness;
        sum_sq += fitness * fitness;
    }
    
    state->average_fitness = sum / swarm->config.swarm_size;
    if (swarm->config.swarm_size > 1) {
        float variance = (sum_sq - sum * sum / swarm->config.swarm_size) / (swarm->config.swarm_size - 1);
        state->fitness_std_dev = sqrtf(fmaxf(variance, 0.0f));
    }
    
    /* 计算位置多样性 */
    float** positions = (float**)safe_malloc(sizeof(float*) * swarm->config.swarm_size);
    if (positions != NULL) {
        for (int i = 0; i < swarm->config.swarm_size; i++) {
            positions[i] = ((Swarm*)swarm)->individuals[i].position;
        }
        state->position_diversity = swarm_calculate_diversity((Swarm*)swarm, positions);
        safe_free((void**)&positions);
    }
    
    /* 复制最佳位置 */
    if (state->best_position == NULL) {
        state->best_position = (float*)safe_malloc(sizeof(float) * swarm->config.dimensions);
        if (state->best_position == NULL) {
            return -1;
        }
    }
    copy_position(state->best_position, swarm->global_best_position, swarm->config.dimensions);
    
    return 0;
}

/**
 * @brief 计算多样性
 */
static float swarm_calculate_diversity(Swarm* swarm, float** positions) {
    if (positions == NULL || swarm->config.swarm_size <= 1) {
        return 0.0f;
    }
    
    /* 计算所有位置的中心 */
    float* center = (float*)safe_malloc(sizeof(float) * swarm->config.dimensions);
    if (center == NULL) {
        return 0.0f;
    }
    
    memset(center, 0, sizeof(float) * swarm->config.dimensions);
    for (int i = 0; i < swarm->config.swarm_size; i++) {
        for (int d = 0; d < swarm->config.dimensions; d++) {
            center[d] += positions[i][d];
        }
    }
    
    for (int d = 0; d < swarm->config.dimensions; d++) {
        center[d] /= swarm->config.swarm_size;
    }
    
    /* 计算平均距离 */
    float total_distance = 0.0f;
    for (int i = 0; i < swarm->config.swarm_size; i++) {
        total_distance += euclidean_distance(positions[i], center, swarm->config.dimensions);
    }
    
    safe_free((void**)&center);
    return total_distance / swarm->config.swarm_size;
}

/**
 * @brief 获取个体信息
 */
int swarm_get_individual(const Swarm* swarm, int individual_id, SwarmIndividual* individual) {
    if (swarm == NULL || individual == NULL) {
        return -1;
    }
    
    if (individual_id < 0 || individual_id >= swarm->config.swarm_size) {
        return -1;
    }
    
    SwarmIndividual* src = &swarm->individuals[individual_id];
    
    /* 复制基本信息 */
    individual->id = src->id;
    individual->is_active = src->is_active;
    individual->best_fitness = src->best_fitness;
    individual->current_fitness = src->current_fitness;
    
    /* 分配并复制位置 */
    if (individual->position == NULL) {
        individual->position = (float*)safe_malloc(sizeof(float) * swarm->config.dimensions);
        if (individual->position == NULL) {
            return -1;
        }
    }
    copy_position(individual->position, src->position, swarm->config.dimensions);
    
    /* 分配并复制速度 */
    if (individual->velocity == NULL) {
        individual->velocity = (float*)safe_malloc(sizeof(float) * swarm->config.dimensions);
        if (individual->velocity == NULL) {
            safe_free((void**)&individual->position);
            return -1;
        }
    }
    copy_position(individual->velocity, src->velocity, swarm->config.dimensions);
    
    /* 分配并复制最佳位置 */
    if (individual->best_position == NULL) {
        individual->best_position = (float*)safe_malloc(sizeof(float) * swarm->config.dimensions);
        if (individual->best_position == NULL) {
            safe_free((void**)&individual->position);
            safe_free((void**)&individual->velocity);
            return -1;
        }
    }
    copy_position(individual->best_position, src->best_position, swarm->config.dimensions);
    
    /* 自定义数据（浅拷贝） */
    individual->custom_data = src->custom_data;
    
    return 0;
}

/**
 * @brief 更新个体位置
 */
int swarm_update_individual_position(Swarm* swarm, int individual_id, const float* new_position) {
    if (swarm == NULL || new_position == NULL) {
        return -1;
    }
    
    if (individual_id < 0 || individual_id >= swarm->config.swarm_size) {
        return -1;
    }
    
    SwarmIndividual* ind = &swarm->individuals[individual_id];
    copy_position(ind->position, new_position, swarm->config.dimensions);
    
    /* 应用约束 */
    swarm_apply_constraints(swarm, individual_id);
    
    /* 重新计算适应度 */
    ind->current_fitness = swarm_calculate_fitness(swarm, individual_id);
    
    /* 更新个体最佳 */
    if (ind->current_fitness < ind->best_fitness) {
        ind->best_fitness = ind->current_fitness;
        copy_position(ind->best_position, ind->position, swarm->config.dimensions);
    }
    
    /* 更新全局最佳 */
    if (ind->current_fitness < swarm->global_best_fitness) {
        swarm->global_best_fitness = ind->current_fitness;
        swarm->global_best_id = individual_id;
        copy_position(swarm->global_best_position, ind->position, swarm->config.dimensions);
    }
    
    return 0;
}

/**
 * @brief 获取最佳解
 */
int swarm_get_best_solution(const Swarm* swarm, float* position, float* fitness) {
    if (swarm == NULL) {
        return -1;
    }
    
    if (position != NULL) {
        copy_position(position, swarm->global_best_position, swarm->config.dimensions);
    }
    
    if (fitness != NULL) {
        *fitness = swarm->global_best_fitness;
    }
    
    return 0;
}

/**
 * @brief 重置群体
 */
int swarm_reset(Swarm* swarm) {
    if (swarm == NULL) {
        return -1;
    }
    
    /* 重置状态 */
    swarm->current_iteration = 0;
    swarm->is_converged = 0;
    swarm->convergence_reason = -1;
    swarm->global_best_fitness = FLT_MAX;
    swarm->global_best_id = -1;
    memset(swarm->global_best_position, 0, sizeof(float) * swarm->config.dimensions);
    
    /* 重置个体 */
    for (int i = 0; i < swarm->config.swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        ind->best_fitness = FLT_MAX;
        ind->current_fitness = FLT_MAX;
        memset(ind->position, 0, sizeof(float) * swarm->config.dimensions);
        memset(ind->velocity, 0, sizeof(float) * swarm->config.dimensions);
        memset(ind->best_position, 0, sizeof(float) * swarm->config.dimensions);
    }
    
    swarm->is_initialized = 0;
    return 0;
}

/**
 * @brief 保存群体状态到文件
 */
int swarm_save_state(const Swarm* swarm, const char* filename) {
    if (!swarm || !filename) return -1;
    
    FILE* file = fopen(filename, "wb");
    if (!file) return -1;
    
    if (fwrite(&swarm->config, sizeof(SwarmConfig), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    
    if (fwrite(&swarm->current_iteration, sizeof(int), 1, file) != 1 ||
        fwrite(&swarm->global_best_fitness, sizeof(float), 1, file) != 1 ||
        fwrite(&swarm->global_best_id, sizeof(int), 1, file) != 1 ||
        fwrite(&swarm->convergence_reason, sizeof(int), 1, file) != 1 ||
        fwrite(&swarm->is_initialized, sizeof(int), 1, file) != 1 ||
        fwrite(&swarm->config.dimensions, sizeof(int), 1, file) != 1 ||
        fwrite(&swarm->config.swarm_size, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    
    int dims = swarm->config.dimensions;
    int size = swarm->config.swarm_size;
    
    if (dims > 0 && fwrite(swarm->global_best_position, sizeof(float), dims, file) != (size_t)dims) {
        fclose(file);
        return -1;
    }
    
    for (int i = 0; i < size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        if (fwrite(ind->position, sizeof(float), dims, file) != (size_t)dims ||
            fwrite(ind->velocity, sizeof(float), dims, file) != (size_t)dims ||
            fwrite(ind->best_position, sizeof(float), dims, file) != (size_t)dims ||
            fwrite(&ind->best_fitness, sizeof(float), 1, file) != 1 ||
            fwrite(&ind->current_fitness, sizeof(float), 1, file) != 1) {
            fclose(file);
            return -1;
        }
    }
    
    fclose(file);
    return 0;
}

/**
 * @brief 从文件加载群体状态
 */
int swarm_load_state(Swarm* swarm, const char* filename) {
    if (!swarm || !filename) return -1;
    
    FILE* file = fopen(filename, "rb");
    if (!file) return -1;
    
    SwarmConfig loaded_config;
    if (fread(&loaded_config, sizeof(SwarmConfig), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    
    int loaded_iteration, loaded_gbest_id, loaded_convergence, loaded_initialized;
    int loaded_dims, loaded_size;
    if (fread(&loaded_iteration, sizeof(int), 1, file) != 1 ||
        fread(&swarm->global_best_fitness, sizeof(float), 1, file) != 1 ||
        fread(&loaded_gbest_id, sizeof(int), 1, file) != 1 ||
        fread(&loaded_convergence, sizeof(int), 1, file) != 1 ||
        fread(&loaded_initialized, sizeof(int), 1, file) != 1 ||
        fread(&loaded_dims, sizeof(int), 1, file) != 1 ||
        fread(&loaded_size, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    
    if (loaded_dims > SWARM_MAX_DIMENSIONS || loaded_size > SWARM_MAX_SIZE ||
        loaded_dims <= 0 || loaded_size <= 0) {
        fclose(file);
        return -1;
    }
    
    swarm->current_iteration = loaded_iteration;
    swarm->global_best_id = loaded_gbest_id;
    swarm->convergence_reason = loaded_convergence;
    swarm->is_initialized = loaded_initialized;
    swarm->config.dimensions = loaded_dims;
    swarm->config.swarm_size = loaded_size;
    
    if (fread(swarm->global_best_position, sizeof(float), loaded_dims, file) != (size_t)loaded_dims) {
        fclose(file);
        return -1;
    }
    
    for (int i = 0; i < loaded_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        if (fread(ind->position, sizeof(float), loaded_dims, file) != (size_t)loaded_dims ||
            fread(ind->velocity, sizeof(float), loaded_dims, file) != (size_t)loaded_dims ||
            fread(ind->best_position, sizeof(float), loaded_dims, file) != (size_t)loaded_dims ||
            fread(&ind->best_fitness, sizeof(float), 1, file) != 1 ||
            fread(&ind->current_fitness, sizeof(float), 1, file) != 1) {
            fclose(file);
            return -1;
        }
    }
    
    fclose(file);
    return 0;
}

/**
 * @brief 设置算法参数
 */
int swarm_set_parameter(Swarm* swarm, const char* param_name, float value) {
    if (swarm == NULL || param_name == NULL) {
        return -1;
    }
    
    /* 根据参数名设置值 */
    if (strcmp(param_name, "inertia_weight") == 0) {
        if (swarm->config.algorithm_type == SWARM_ALGORITHM_PSO) {
            PSOData* pso_data = (PSOData*)swarm->algorithm_data;
            if (pso_data != NULL) {
                pso_data->inertia_weight = value;
            }
        }
        swarm->config.inertia_weight = value;
    } else if (strcmp(param_name, "cognitive_weight") == 0) {
        swarm->config.cognitive_weight = value;
    } else if (strcmp(param_name, "social_weight") == 0) {
        swarm->config.social_weight = value;
    } else if (strcmp(param_name, "exploration_factor") == 0) {
        swarm->config.exploration_factor = value;
    } else if (strcmp(param_name, "exploitation_factor") == 0) {
        swarm->config.exploitation_factor = value;
    } else {
        return -1; /* 未知参数 */
    }
    
    return 0;
}

/**
 * @brief 获取算法参数
 */
int swarm_get_parameter(const Swarm* swarm, const char* param_name, float* value) {
    if (swarm == NULL || param_name == NULL || value == NULL) {
        return -1;
    }
    
    /* 根据参数名获取值 */
    if (strcmp(param_name, "inertia_weight") == 0) {
        *value = swarm->config.inertia_weight;
    } else if (strcmp(param_name, "cognitive_weight") == 0) {
        *value = swarm->config.cognitive_weight;
    } else if (strcmp(param_name, "social_weight") == 0) {
        *value = swarm->config.social_weight;
    } else if (strcmp(param_name, "exploration_factor") == 0) {
        *value = swarm->config.exploration_factor;
    } else if (strcmp(param_name, "exploitation_factor") == 0) {
        *value = swarm->config.exploitation_factor;
    } else {
        return -1; /* 未知参数 */
    }
    
    return 0;
}

/**
 * @brief 获取算法类型名称
 */
const char* swarm_algorithm_name(SwarmAlgorithmType algorithm_type) {
    switch (algorithm_type) {
        case SWARM_ALGORITHM_PSO: return "粒子群优化";
        case SWARM_ALGORITHM_ACO: return "蚁群算法";
        case SWARM_ALGORITHM_ABC: return "人工蜂群算法";
        case SWARM_ALGORITHM_FSS: return "鱼群算法";
        case SWARM_ALGORITHM_BAT: return "蝙蝠算法";
        case SWARM_ALGORITHM_FIREFLY: return "萤火虫算法";
        case SWARM_ALGORITHM_CUCKOO: return "布谷鸟搜索";
        case SWARM_ALGORITHM_GWO: return "灰狼优化器";
        case SWARM_ALGORITHM_WOA: return "鲸鱼优化算法";
        case SWARM_ALGORITHM_MARL: return "多智能体强化学习";
        case SWARM_ALGORITHM_HYBRID: return "混合算法";
        default: return "未知算法";
    }
}

/**
 * @brief 获取拓扑类型名称
 */
const char* swarm_topology_name(SwarmTopologyType topology_type) {
    switch (topology_type) {
        case SWARM_TOPOLOGY_GLOBAL: return "全局拓扑";
        case SWARM_TOPOLOGY_RING: return "环状拓扑";
        case SWARM_TOPOLOGY_STAR: return "星形拓扑";
        case SWARM_TOPOLOGY_MESH: return "网状拓扑";
        case SWARM_TOPOLOGY_RANDOM: return "随机拓扑";
        case SWARM_TOPOLOGY_HIERARCHICAL: return "分层拓扑";
        default: return "未知拓扑";
    }
}

/**
 * @brief 获取收敛条件名称
 */
const char* swarm_convergence_condition_name(SwarmConvergenceCondition condition) {
    switch (condition) {
        case SWARM_CONVERGENCE_MAX_ITERATIONS: return "最大迭代次数";
        case SWARM_CONVERGENCE_FITNESS_THRESHOLD: return "适应度阈值";
        case SWARM_CONVERGENCE_POSITION_CHANGE: return "位置变化阈值";
        case SWARM_CONVERGENCE_VELOCITY_CHANGE: return "速度变化阈值";
        case SWARM_CONVERGENCE_STAGNATION: return "停滞迭代次数";
        default: return "未知条件";
    }
}

/**
 * @brief 初始化群体结果
 */
void swarm_result_init(SwarmResult* result) {
    if (result == NULL) {
        return;
    }
    
    memset(result, 0, sizeof(SwarmResult));
    result->best_fitness = FLT_MAX;
    result->iterations = 0;
    result->total_time_ms = 0.0f;
    result->convergence_reason = -1;
    result->best_position = NULL;
    result->error_message[0] = '\0';
}

/**
 * @brief 释放群体结果内存
 */
void swarm_result_free(SwarmResult* result) {
    if (result == NULL) {
        return;
    }
    
    if (result->best_position != NULL) {
        safe_free((void**)&result->best_position);
    }
    
    swarm_state_free(&result->final_state);
}

/**
 * @brief 初始化群体状态
 */
void swarm_state_init(SwarmState* state) {
    if (state == NULL) {
        return;
    }
    
    memset(state, 0, sizeof(SwarmState));
    state->best_fitness = FLT_MAX;
    state->best_position = NULL;
}

/**
 * @brief 释放群体状态内存
 */
void swarm_state_free(SwarmState* state) {
    if (state == NULL) {
        return;
    }
    
    if (state->best_position != NULL) {
        safe_free((void**)&state->best_position);
    }
}

/* ========== PSO算法实现 ========== */

/**
 * @brief 创建PSO算法数据
 */
static PSOData* pso_data_create(const SwarmConfig* config) {
    PSOData* data = (PSOData*)safe_malloc(sizeof(PSOData));
    if (data == NULL) {
        return NULL;
    }
    
    int swarm_size = config->swarm_size;
    int dimensions = config->dimensions;
    
    /* 分配个体历史最佳位置数组 */
    data->personal_best_positions = (float*)safe_malloc(sizeof(float) * swarm_size * dimensions);
    data->personal_best_fitnesses = (float*)safe_malloc(sizeof(float) * swarm_size);
    data->neighborhood_best_positions = (float*)safe_malloc(sizeof(float) * swarm_size * dimensions);
    data->neighborhood_best_fitnesses = (float*)safe_malloc(sizeof(float) * swarm_size);
    data->neighborhood_indices = (int*)safe_malloc(sizeof(int) * swarm_size * swarm_size);
    
    if (data->personal_best_positions == NULL || data->personal_best_fitnesses == NULL ||
        data->neighborhood_best_positions == NULL || data->neighborhood_best_fitnesses == NULL ||
        data->neighborhood_indices == NULL) {
        pso_data_free(data);
        return NULL;
    }
    
    /* 初始化参数 */
    data->inertia_weight = config->inertia_weight;
    data->inertia_weight_max = config->inertia_weight * 1.2f;
    data->inertia_weight_min = config->inertia_weight * 0.8f;
    data->cognitive_weight = config->cognitive_weight;
    data->social_weight = config->social_weight;
    
    /* 计算最大速度（边界范围的20%） */
    float max_range = 0.0f;
    for (int d = 0; d < dimensions; d++) {
        float range = config->upper_bounds[d] - config->lower_bounds[d];
        if (range > max_range) max_range = range;
    }
    data->velocity_max = max_range * 0.2f;
    
    return data;
}

/**
 * @brief 释放PSO算法数据
 */
static void pso_data_free(PSOData* data) {
    if (data == NULL) {
        return;
    }
    
    safe_free((void**)&data->personal_best_positions);
    safe_free((void**)&data->personal_best_fitnesses);
    safe_free((void**)&data->neighborhood_best_positions);
    safe_free((void**)&data->neighborhood_best_fitnesses);
    safe_free((void**)&data->neighborhood_indices);
    safe_free((void**)&data);
}

/**
 * @brief 初始化PSO算法
 */
static void pso_initialize(Swarm* swarm) {
    PSOData* data = (PSOData*)swarm->algorithm_data;
    if (data == NULL) {
        return;
    }
    
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    
    /* 初始化个体历史最佳 */
    for (int i = 0; i < swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        
        /* 复制当前位置到历史最佳 */
        float* personal_best = &data->personal_best_positions[i * dimensions];
        copy_position(personal_best, ind->position, dimensions);
        data->personal_best_fitnesses[i] = ind->current_fitness;
        
        /* 初始化邻居最佳 */
        float* neighborhood_best = &data->neighborhood_best_positions[i * dimensions];
        copy_position(neighborhood_best, ind->position, dimensions);
        data->neighborhood_best_fitnesses[i] = ind->current_fitness;
    }
    
    /* 初始化邻居索引（全连接拓扑） */
    for (int i = 0; i < swarm_size; i++) {
        for (int j = 0; j < swarm_size; j++) {
            data->neighborhood_indices[i * swarm_size + j] = j;
        }
    }
}

/**
 * @brief PSO迭代
 */
static void pso_iterate(Swarm* swarm) {
    PSOData* data = (PSOData*)swarm->algorithm_data;
    if (data == NULL) {
        return;
    }
    
    int swarm_size = swarm->config.swarm_size;
    
    /* 更新惯性权重（线性递减） */
    float progress = (float)swarm->current_iteration / (float)swarm->config.max_iterations;
    data->inertia_weight = data->inertia_weight_max - 
                          (data->inertia_weight_max - data->inertia_weight_min) * progress;
    
    /* 更新每个粒子 */
    for (int i = 0; i < swarm_size; i++) {
        swarm_update_velocity_position_pso(swarm, i);
    }
}

/**
 * @brief 更新PSO粒子速度和位置
 */
static void swarm_update_velocity_position_pso(Swarm* swarm, int individual_id) {
    PSOData* data = (PSOData*)swarm->algorithm_data;
    if (data == NULL) {
        return;
    }
    
    SwarmIndividual* ind = &swarm->individuals[individual_id];
    int dimensions = swarm->config.dimensions;
    
    /* 更新速度 */
    for (int d = 0; d < dimensions; d++) {
        float r1 = random_float(0.0f, 1.0f);
        float r2 = random_float(0.0f, 1.0f);
        
        float cognitive_component = data->cognitive_weight * r1 * 
                                   (data->personal_best_positions[individual_id * dimensions + d] - ind->position[d]);
        float social_component = data->social_weight * r2 * 
                                (data->neighborhood_best_positions[individual_id * dimensions + d] - ind->position[d]);
        
        ind->velocity[d] = data->inertia_weight * ind->velocity[d] + 
                          cognitive_component + social_component;
        
        /* 限制速度 */
        if (ind->velocity[d] > data->velocity_max) ind->velocity[d] = data->velocity_max;
        if (ind->velocity[d] < -data->velocity_max) ind->velocity[d] = -data->velocity_max;
    }
    
    /* 更新位置 */
    for (int d = 0; d < dimensions; d++) {
        ind->position[d] += ind->velocity[d];
    }
    
    /* 应用约束 */
    swarm_apply_constraints(swarm, individual_id);
    
    /* 计算新适应度 */
    ind->current_fitness = swarm_calculate_fitness(swarm, individual_id);
    
    /* 更新个体历史最佳 */
    if (ind->current_fitness < data->personal_best_fitnesses[individual_id]) {
        data->personal_best_fitnesses[individual_id] = ind->current_fitness;
        copy_position(&data->personal_best_positions[individual_id * dimensions], 
                     ind->position, dimensions);
    }
    
    /* 更新邻居最佳（环形拓扑） */
    int swarm_size = swarm->config.swarm_size;
    dimensions = swarm->config.dimensions;
    int neighborhood_size = 3;
    int best_neighbor_id = individual_id;
    float best_neighbor_fitness = data->neighborhood_best_fitnesses[individual_id];
    
    for (int n = -neighborhood_size; n <= neighborhood_size; n++) {
        if (n == 0) continue;
        int neighbor_id = (individual_id + n + swarm_size) % swarm_size;
        float* nb_pos = &data->neighborhood_best_positions[neighbor_id * dimensions];
        float nb_fitness = data->neighborhood_best_fitnesses[neighbor_id];
        if (nb_fitness < best_neighbor_fitness) {
            float updated_nb_fitness = 0.0f;
            for (int d = 0; d < dimensions; d++) {
                updated_nb_fitness += nb_pos[d];
            }
            if (nb_fitness < best_neighbor_fitness) {
                best_neighbor_fitness = nb_fitness;
                best_neighbor_id = neighbor_id;
            }
        }
    }
    
    if (best_neighbor_id != individual_id &&
        data->neighborhood_best_fitnesses[best_neighbor_id] < data->neighborhood_best_fitnesses[individual_id]) {
        data->neighborhood_best_fitnesses[individual_id] = data->neighborhood_best_fitnesses[best_neighbor_id];
        copy_position(&data->neighborhood_best_positions[individual_id * dimensions],
                     &data->neighborhood_best_positions[best_neighbor_id * dimensions], dimensions);
    }
    
    if (ind->current_fitness < data->neighborhood_best_fitnesses[individual_id]) {
        data->neighborhood_best_fitnesses[individual_id] = ind->current_fitness;
        copy_position(&data->neighborhood_best_positions[individual_id * dimensions],
                     ind->position, dimensions);
    }
}


/* ========== ACO算法函数实现 ========== */

/**
 * @brief 创建ACO算法数据
 */
static ACOData* aco_data_create(const SwarmConfig* config) {
    if (config == NULL || config->dimensions <= 0 || config->swarm_size <= 0) {
        return NULL;
    }

    ACOData* data = (ACOData*)safe_malloc(sizeof(ACOData));
    if (data == NULL) {
        return NULL;
    }

    data->num_ants = config->swarm_size;
    data->path_length = config->dimensions;
    data->pheromone_evaporation = (config->pheromone_evaporation > 0.0f) ?
                                   config->pheromone_evaporation : 0.1f;
    data->pheromone_deposit = 1.0f;
    data->alpha = (config->exploration_factor > 0.0f) ?
                   config->exploration_factor : 1.0f;
    data->beta = (config->exploitation_factor > 0.0f) ?
                  config->exploitation_factor : 2.0f;
    data->q0 = 0.9f;

    int dim = config->dimensions;
    int num_ants = data->num_ants;
    int path_len = data->path_length;

    /* 分配信息素矩阵（dim x dim） */
    data->pheromone_matrix = (float**)safe_malloc(sizeof(float*) * dim);
    data->heuristic_matrix = (float**)safe_malloc(sizeof(float*) * dim);
    data->probability_matrix = (float**)safe_malloc(sizeof(float*) * dim);

    if (!data->pheromone_matrix || !data->heuristic_matrix || !data->probability_matrix) {
        safe_free((void**)&data->probability_matrix);
        safe_free((void**)&data->heuristic_matrix);
        safe_free((void**)&data->pheromone_matrix);
        safe_free((void**)&data);
        return NULL;
    }

    for (int i = 0; i < dim; i++) {
        data->pheromone_matrix[i] = (float*)safe_calloc(dim, sizeof(float));
        data->heuristic_matrix[i] = (float*)safe_calloc(dim, sizeof(float));
        data->probability_matrix[i] = (float*)safe_calloc(dim, sizeof(float));
        if (!data->pheromone_matrix[i] || !data->heuristic_matrix[i] || !data->probability_matrix[i]) {
            for (int j = 0; j <= i; j++) {
                safe_free((void**)&data->probability_matrix[j]);
                safe_free((void**)&data->heuristic_matrix[j]);
                safe_free((void**)&data->pheromone_matrix[j]);
            }
            safe_free((void**)&data->probability_matrix);
            safe_free((void**)&data->heuristic_matrix);
            safe_free((void**)&data->pheromone_matrix);
            safe_free((void**)&data);
            return NULL;
        }
        /* 初始化信息素和启发式信息 */
        for (int j = 0; j < dim; j++) {
            if (i != j) {
                data->pheromone_matrix[i][j] = 0.1f;
                data->heuristic_matrix[i][j] = 1.0f / (float)(abs(i - j) + 1);
            }
        }
    }

    /* 分配蚂蚁路径（num_ants x path_len） */
    data->ant_paths = (int**)safe_malloc(sizeof(int*) * num_ants);
    if (data->ant_paths == NULL) {
        for (int i = 0; i < dim; i++) {
            safe_free((void**)&data->probability_matrix[i]);
            safe_free((void**)&data->heuristic_matrix[i]);
            safe_free((void**)&data->pheromone_matrix[i]);
        }
        safe_free((void**)&data->probability_matrix);
        safe_free((void**)&data->heuristic_matrix);
        safe_free((void**)&data->pheromone_matrix);
        safe_free((void**)&data);
        return NULL;
    }

    for (int i = 0; i < num_ants; i++) {
        data->ant_paths[i] = (int*)safe_calloc(path_len, sizeof(int));
        if (data->ant_paths[i] == NULL) {
            for (int j = 0; j < i; j++) {
                safe_free((void**)&data->ant_paths[j]);
            }
            safe_free((void**)&data->ant_paths);
            for (int j = 0; j < dim; j++) {
                safe_free((void**)&data->probability_matrix[j]);
                safe_free((void**)&data->heuristic_matrix[j]);
                safe_free((void**)&data->pheromone_matrix[j]);
            }
            safe_free((void**)&data->probability_matrix);
            safe_free((void**)&data->heuristic_matrix);
            safe_free((void**)&data->pheromone_matrix);
            safe_free((void**)&data);
            return NULL;
        }
    }

    data->ant_path_lengths = (float*)safe_calloc(num_ants, sizeof(float));
    if (data->ant_path_lengths == NULL) {
        for (int i = 0; i < num_ants; i++) {
            safe_free((void**)&data->ant_paths[i]);
        }
        safe_free((void**)&data->ant_paths);
        for (int i = 0; i < dim; i++) {
            safe_free((void**)&data->probability_matrix[i]);
            safe_free((void**)&data->heuristic_matrix[i]);
            safe_free((void**)&data->pheromone_matrix[i]);
        }
        safe_free((void**)&data->probability_matrix);
        safe_free((void**)&data->heuristic_matrix);
        safe_free((void**)&data->pheromone_matrix);
        safe_free((void**)&data);
        return NULL;
    }

    return data;
}

/**
 * @brief 释放ACO算法数据
 */
static void aco_data_free(ACOData* data) {
    if (data == NULL) {
        return;
    }

    int dim = data->path_length;
    int num_ants = data->num_ants;

    if (data->pheromone_matrix) {
        for (int i = 0; i < dim; i++) {
            safe_free((void**)&data->pheromone_matrix[i]);
        }
        safe_free((void**)&data->pheromone_matrix);
    }

    if (data->heuristic_matrix) {
        for (int i = 0; i < dim; i++) {
            safe_free((void**)&data->heuristic_matrix[i]);
        }
        safe_free((void**)&data->heuristic_matrix);
    }

    if (data->probability_matrix) {
        for (int i = 0; i < dim; i++) {
            safe_free((void**)&data->probability_matrix[i]);
        }
        safe_free((void**)&data->probability_matrix);
    }

    if (data->ant_paths) {
        for (int i = 0; i < num_ants; i++) {
            safe_free((void**)&data->ant_paths[i]);
        }
        safe_free((void**)&data->ant_paths);
    }

    safe_free((void**)&data->ant_path_lengths);
    safe_free((void**)&data);
}

/**
 * @brief 初始化ACO算法
 */
static void aco_initialize(Swarm* swarm) {
    ACOData* data = (ACOData*)swarm->algorithm_data;
    if (data == NULL) {
        return;
    }

    int dim = swarm->config.dimensions;

    /* 重置信息素矩阵 */
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i != j) {
                data->pheromone_matrix[i][j] = 0.1f;
                data->heuristic_matrix[i][j] = 1.0f / (float)(abs(i - j) + 1);
            } else {
                data->pheromone_matrix[i][j] = 0.0f;
                data->heuristic_matrix[i][j] = 0.0f;
            }
            data->probability_matrix[i][j] = 0.0f;
        }
    }

    /* 重置蚂蚁路径 */
    for (int i = 0; i < data->num_ants; i++) {
        memset(data->ant_paths[i], 0, sizeof(int) * data->path_length);
        data->ant_path_lengths[i] = 0.0f;
    }

    /* 初始化每个个体的位置为随机起点 */
    for (int i = 0; i < swarm->config.swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        int start_node = rng_next() % (uint64_t)(dim);
        ind->position[0] = (float)start_node;
        for (int d = 1; d < dim; d++) {
            ind->position[d] = (float)((start_node + d) % dim);
        }
    }
}

/**
 * @brief 为单只蚂蚁构建路径（轮盘赌选择）
 */
static void aco_construct_solution_internal(Swarm* swarm, ACOData* data, int ant_id) {
    int dim = swarm->config.dimensions;
    int* path = data->ant_paths[ant_id];
    float* path_length = &data->ant_path_lengths[ant_id];

    /* 已访问节点标记 */
    int* visited = (int*)safe_calloc(dim, sizeof(int));
    if (visited == NULL) {
        return;
    }

    /* 随机选择起始节点 */
    int current = rng_next() % (uint64_t)(dim);
    path[0] = current;
    visited[current] = 1;
    int step = 1;

    while (step < dim) {
        /* 计算选择概率 */
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            if (!visited[j]) {
                float tau = data->pheromone_matrix[current][j];
                float eta = data->heuristic_matrix[current][j];
                data->probability_matrix[current][j] = powf(tau, data->alpha) * powf(eta, data->beta);
                sum += data->probability_matrix[current][j];
            } else {
                data->probability_matrix[current][j] = 0.0f;
            }
        }

        /* 轮盘赌选择下一节点 */
        if (sum > 0.0f) {
            float r = rng_uniform(0.0f, 1.0f) * sum;
            float cum = 0.0f;
            int next = -1;
            for (int j = 0; j < dim; j++) {
                if (!visited[j]) {
                    cum += data->probability_matrix[current][j];
                    if (cum >= r) {
                        next = j;
                        break;
                    }
                }
            }
            if (next == -1) {
                /* 回退：选择第一个未访问节点 */
                for (int j = 0; j < dim; j++) {
                    if (!visited[j]) {
                        next = j;
                        break;
                    }
                }
            }
            current = next;
        } else {
            /* 回退：选择第一个未访问节点 */
            for (int j = 0; j < dim; j++) {
                if (!visited[j]) {
                    current = j;
                    break;
                }
            }
        }

        path[step] = current;
        visited[current] = 1;
        step++;
    }

    /* 计算路径长度 */
    *path_length = 0.0f;
    for (int i = 0; i < dim - 1; i++) {
        *path_length += (float)abs(path[i] - path[i + 1]);
    }
    /* 回到起点形成环路 */
    *path_length += (float)abs(path[dim - 1] - path[0]);

    safe_free((void**)&visited);
}

/**
 * @brief ACO迭代
 */
static void aco_iterate(Swarm* swarm) {
    ACOData* data = (ACOData*)swarm->algorithm_data;
    if (data == NULL) {
        return;
    }

    int dim = swarm->config.dimensions;
    int num_ants = data->num_ants;

    /* 每只蚂蚁构建路径 */
    for (int i = 0; i < num_ants && i < swarm->config.swarm_size; i++) {
        aco_construct_solution_internal(swarm, data, i);
        /* 更新个体适应度（路径长度越短，适应度越好） */
        swarm->individuals[i].current_fitness = data->ant_path_lengths[i];
    }

    /* 信息素蒸发 */
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i != j) {
                data->pheromone_matrix[i][j] *= (1.0f - data->pheromone_evaporation);
                if (data->pheromone_matrix[i][j] < 0.0001f) {
                    data->pheromone_matrix[i][j] = 0.0001f;
                }
            }
        }
    }

    /* 信息素沉积 */
    for (int ant = 0; ant < num_ants && ant < swarm->config.swarm_size; ant++) {
        float deposit = data->pheromone_deposit / (data->ant_path_lengths[ant] + 0.001f);
        for (int i = 0; i < dim - 1; i++) {
            int from = data->ant_paths[ant][i];
            int to = data->ant_paths[ant][i + 1];
            data->pheromone_matrix[from][to] += deposit;
            data->pheromone_matrix[to][from] += deposit;
        }
        /* 环路连接 */
        int last = data->ant_paths[ant][dim - 1];
        int first = data->ant_paths[ant][0];
        data->pheromone_matrix[last][first] += deposit;
        data->pheromone_matrix[first][last] += deposit;
    }

    /* 更新全局最佳（最小化路径长度） */
    for (int i = 0; i < num_ants && i < swarm->config.swarm_size; i++) {
        if (data->ant_path_lengths[i] < swarm->global_best_fitness) {
            swarm->global_best_fitness = data->ant_path_lengths[i];
            copy_position(swarm->global_best_position, swarm->individuals[i].position, dim);
        }
    }
}

/* ========== ABC算法函数实现 ========== */

/**
 * @brief 创建ABC算法数据
 */
static ABCData* abc_data_create(const SwarmConfig* config) {
    if (config == NULL || config->swarm_size <= 0 || config->dimensions <= 0) {
        return NULL;
    }

    ABCData* data = (ABCData*)safe_malloc(sizeof(ABCData));
    if (data == NULL) {
        return NULL;
    }

    int swarm_size = config->swarm_size;

    data->trial_counts = (float*)safe_calloc(swarm_size, sizeof(float));
    data->probabilities = (float*)safe_calloc(swarm_size, sizeof(float));
    data->employed_bees = (int*)safe_calloc(swarm_size, sizeof(int));
    data->onlooker_bees = (int*)safe_calloc(swarm_size, sizeof(int));
    data->scout_bees = (int*)safe_calloc(swarm_size, sizeof(int));
    data->limit = (float)(swarm_size * config->dimensions);  /* 试验限制 = 群体大小 * 维度 */
    data->abandonment_probability = 0.5f;

    if (!data->trial_counts || !data->probabilities ||
        !data->employed_bees || !data->onlooker_bees || !data->scout_bees) {
        safe_free((void**)&data->scout_bees);
        safe_free((void**)&data->onlooker_bees);
        safe_free((void**)&data->employed_bees);
        safe_free((void**)&data->probabilities);
        safe_free((void**)&data->trial_counts);
        safe_free((void**)&data);
        return NULL;
    }

    /* 初始化：所有蜜蜂都是雇佣蜂 */
    for (int i = 0; i < swarm_size; i++) {
        data->employed_bees[i] = 1;
        data->onlooker_bees[i] = 0;
        data->scout_bees[i] = 0;
        data->trial_counts[i] = 0.0f;
        data->probabilities[i] = 0.0f;
    }

    return data;
}

/**
 * @brief 释放ABC算法数据
 */
static void abc_data_free(ABCData* data) {
    if (data == NULL) {
        return;
    }

    safe_free((void**)&data->trial_counts);
    safe_free((void**)&data->probabilities);
    safe_free((void**)&data->employed_bees);
    safe_free((void**)&data->onlooker_bees);
    safe_free((void**)&data->scout_bees);
    safe_free((void**)&data);
}

/**
 * @brief 初始化ABC算法
 */
static void abc_initialize(Swarm* swarm) {
    ABCData* data = (ABCData*)swarm->algorithm_data;
    if (data == NULL) {
        return;
    }

    int swarm_size = swarm->config.swarm_size;

    /* 重置试验计数 */
    memset(data->trial_counts, 0, sizeof(float) * swarm_size);

    /* 初始化概率 */
    for (int i = 0; i < swarm_size; i++) {
        data->probabilities[i] = 0.0f;
        data->employed_bees[i] = 1;
        data->onlooker_bees[i] = 0;
        data->scout_bees[i] = 0;
    }

    /* 初始化个体适应度 */
    for (int i = 0; i < swarm_size; i++) {
        swarm->individuals[i].current_fitness = swarm_calculate_fitness(swarm, i);
    }
}

/**
 * @brief ABC雇佣蜂阶段
 */
static void abc_employed_phase(Swarm* swarm) {
    ABCData* data = (ABCData*)swarm->algorithm_data;
    if (data == NULL) {
        return;
    }

    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;

    for (int i = 0; i < swarm_size; i++) {
        if (!data->employed_bees[i]) {
            continue;
        }

        SwarmIndividual* ind = &swarm->individuals[i];

        /* 生成邻域解 */
        int neighbor_id = rng_next() % (uint64_t)(swarm_size);
        if (neighbor_id == i) {
            neighbor_id = (i + 1) % swarm_size;
        }

        int dim = rng_next() % (uint64_t)(dimensions);
        float phi = (float)(rng_next() % (uint64_t)(2001) - 1000) / 1000.0f;  /* [-1, 1] */

        float new_position = ind->position[dim] + phi *
                            (ind->position[dim] - swarm->individuals[neighbor_id].position[dim]);

        /* 边界检查 */
        float lower = swarm->config.lower_bounds ? swarm->config.lower_bounds[dim] : -100.0f;
        float upper = swarm->config.upper_bounds ? swarm->config.upper_bounds[dim] : 100.0f;
        if (new_position < lower) new_position = lower;
        if (new_position > upper) new_position = upper;

        /* 计算新适应度 */
        float old_value = ind->position[dim];
        ind->position[dim] = new_position;
        float new_fitness = swarm_calculate_fitness(swarm, i);

        if (new_fitness < ind->current_fitness) {
            ind->current_fitness = new_fitness;
            data->trial_counts[i] = 0.0f;
        } else {
            ind->position[dim] = old_value;
            data->trial_counts[i] += 1.0f;
        }
    }
}

/**
 * @brief ABC观察蜂阶段
 */
static void abc_onlooker_phase(Swarm* swarm) {
    ABCData* data = (ABCData*)swarm->algorithm_data;
    if (data == NULL) {
        return;
    }

    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;

    /* 计算选择概率（基于适应度） */
    float max_fitness = -1e38f;
    for (int i = 0; i < swarm_size; i++) {
        if (swarm->individuals[i].current_fitness > max_fitness) {
            max_fitness = swarm->individuals[i].current_fitness;
        }
    }

    float sum_fitness = 0.0f;
    for (int i = 0; i < swarm_size; i++) {
        /* 适用于最小化问题的适应度转换 */
        float fitness_val = max_fitness - swarm->individuals[i].current_fitness + 0.001f;
        if (fitness_val < 0.001f) fitness_val = 0.001f;
        data->probabilities[i] = fitness_val;
        sum_fitness += fitness_val;
    }

    if (sum_fitness > 0.0f) {
        for (int i = 0; i < swarm_size; i++) {
            data->probabilities[i] /= sum_fitness;
        }
    }

    /* 观察蜂选择蜜源 */
    for (int i = 0; i < swarm_size; i++) {
        if (!data->onlooker_bees[i]) {
            continue;
        }

        /* 轮盘赌选择 */
        float r = rng_uniform(0.0f, 1.0f);
        float cum = 0.0f;
        int selected = 0;
        for (int j = 0; j < swarm_size; j++) {
            cum += data->probabilities[j];
            if (cum >= r) {
                selected = j;
                break;
            }
        }

        /* 在选中的蜜源附近搜索 */
        SwarmIndividual* target = &swarm->individuals[selected];
        int neighbor_id = rng_next() % (uint64_t)(swarm_size);
        if (neighbor_id == selected) {
            neighbor_id = (selected + 1) % swarm_size;
        }

        int dim = rng_next() % (uint64_t)(dimensions);
        float phi = (float)(rng_next() % (uint64_t)(2001) - 1000) / 1000.0f;
        float new_position = target->position[dim] + phi *
                            (target->position[dim] - swarm->individuals[neighbor_id].position[dim]);

        float lower = swarm->config.lower_bounds ? swarm->config.lower_bounds[dim] : -100.0f;
        float upper = swarm->config.upper_bounds ? swarm->config.upper_bounds[dim] : 100.0f;
        if (new_position < lower) new_position = lower;
        if (new_position > upper) new_position = upper;

        float old_value = target->position[dim];
        target->position[dim] = new_position;
        float new_fitness = swarm_calculate_fitness(swarm, selected);

        if (new_fitness < target->current_fitness) {
            target->current_fitness = new_fitness;
            data->trial_counts[selected] = 0.0f;
        } else {
            target->position[dim] = old_value;
            data->trial_counts[selected] += 1.0f;
        }
    }
}

/**
 * @brief ABC侦查蜂阶段
 */
static void abc_scout_phase(Swarm* swarm) {
    ABCData* data = (ABCData*)swarm->algorithm_data;
    if (data == NULL) {
        return;
    }

    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;

    for (int i = 0; i < swarm_size; i++) {
        if (!data->scout_bees[i]) {
            continue;
        }

        /* 检查是否超出限制 */
        if (data->trial_counts[i] >= data->limit) {
            /* 随机重新初始化位置 */
            for (int d = 0; d < dimensions; d++) {
                float lower = swarm->config.lower_bounds ? swarm->config.lower_bounds[d] : -100.0f;
                float upper = swarm->config.upper_bounds ? swarm->config.upper_bounds[d] : 100.0f;
                swarm->individuals[i].position[d] = lower + rng_uniform(0.0f, 1.0f) * (upper - lower);
            }

            /* 重新计算适应度 */
            swarm->individuals[i].current_fitness = swarm_calculate_fitness(swarm, i);
            data->trial_counts[i] = 0.0f;

            /* 转换为雇佣蜂 */
            data->employed_bees[i] = 1;
            data->scout_bees[i] = 0;
        }
    }
}

/**
 * @brief ABC迭代
 */
static void abc_iterate(Swarm* swarm) {
    ABCData* data = (ABCData*)swarm->algorithm_data;
    if (data == NULL) {
        return;
    }

    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;

    /* 更新蜂群角色：部分雇佣蜂和观察蜂转为侦查蜂 */
    int num_scouts = (int)(swarm_size * 0.1f);  /* 10% 转为侦查蜂 */
    if (num_scouts < 1) num_scouts = 1;

    for (int i = 0; i < swarm_size; i++) {
        data->employed_bees[i] = 1;
        data->onlooker_bees[i] = 0;
        data->scout_bees[i] = 0;
    }

    /* 随机选择侦查蜂 */
    for (int s = 0; s < num_scouts; s++) {
        int idx = rng_next() % (uint64_t)(swarm_size);
        data->employed_bees[idx] = 0;
        data->onlooker_bees[idx] = 0;
        data->scout_bees[idx] = 1;
    }

    /* 剩余蜜蜂中一半为观察蜂 */
    int onlooker_count = 0;
    int target_onlookers = (swarm_size - num_scouts) / 2;
    for (int i = 0; i < swarm_size && onlooker_count < target_onlookers; i++) {
        if (!data->scout_bees[i]) {
            data->employed_bees[i] = 0;
            data->onlooker_bees[i] = 1;
            onlooker_count++;
        }
    }

    /* 三个阶段 */
    abc_employed_phase(swarm);
    abc_onlooker_phase(swarm);
    abc_scout_phase(swarm);

    /* 更新全局最佳 */
    for (int i = 0; i < swarm_size; i++) {
        float fitness = swarm->individuals[i].current_fitness;
        if (fitness < swarm->global_best_fitness) {
            swarm->global_best_fitness = fitness;
            copy_position(swarm->global_best_position, swarm->individuals[i].position, dimensions);
        }
    }
}


/* ========== FSS（鱼群算法）函数实现 ========== */

static FSSData* fss_data_create(const SwarmConfig* config) {
    FSSData* data = (FSSData*)safe_malloc(sizeof(FSSData));
    if (!data) return NULL;
    
    int swarm_size = config->swarm_size;
    int dimensions = config->dimensions;
    
    data->individual_step_sizes = (float*)safe_malloc(sizeof(float) * swarm_size);
    data->weights = (float*)safe_malloc(sizeof(float) * swarm_size);
    data->prev_positions = (float*)safe_malloc(sizeof(float) * swarm_size * dimensions);
    
    if (!data->individual_step_sizes || !data->weights || !data->prev_positions) {
        fss_data_free(data);
        return NULL;
    }
    
    data->step_size_initial = config->exploration_factor > 0.0f ? config->exploration_factor : 0.1f;
    data->step_size_final = config->exploitation_factor > 0.0f ? config->exploitation_factor : 0.001f;
    data->weight_min = 0.0f;
    data->weight_max = 1.0f;
    data->weight_scale = 1.0f;
    data->instictive_acceleration = 0.0f;
    data->volitive_acceleration = 0.0f;
    data->barycenter_dim = dimensions > 3 ? 3 : dimensions;
    memset(data->barycenter, 0, sizeof(float) * 3);
    
    return data;
}

static void fss_data_free(FSSData* data) {
    if (!data) return;
    safe_free((void**)&data->individual_step_sizes);
    safe_free((void**)&data->weights);
    safe_free((void**)&data->prev_positions);
    safe_free((void**)&data);
}

static void fss_initialize(Swarm* swarm) {
    FSSData* data = (FSSData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    
    for (int i = 0; i < swarm_size; i++) {
        data->individual_step_sizes[i] = data->step_size_initial;
        data->weights[i] = data->weight_max;
        memcpy(&data->prev_positions[i * dimensions], swarm->individuals[i].position,
               sizeof(float) * dimensions);
    }
}

static void fss_iterate(Swarm* swarm) {
    FSSData* data = (FSSData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    float current_step = data->step_size_initial -
        (data->step_size_initial - data->step_size_final) *
        (float)swarm->current_iteration / (float)swarm->config.max_iterations;
    
    float total_weight = 0.0f;
    float barycenter[3] = {0.0f, 0.0f, 0.0f};
    
    for (int i = 0; i < swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        
        memcpy(&data->prev_positions[i * dimensions], ind->position, sizeof(float) * dimensions);
        
        float delta_weight = -ind->current_fitness * data->weight_scale;
        data->weights[i] += delta_weight;
        if (data->weights[i] < data->weight_min) data->weights[i] = data->weight_min;
        if (data->weights[i] > data->weight_max) data->weights[i] = data->weight_max;
        
        total_weight += data->weights[i];
        for (int d = 0; d < dimensions && d < 3; d++) {
            barycenter[d] += ind->position[d] * data->weights[i];
        }
    }
    if (total_weight > 1e-10f) {
        for (int d = 0; d < data->barycenter_dim; d++) barycenter[d] /= total_weight;
    }
    memcpy(data->barycenter, barycenter, sizeof(float) * 3);
    
    for (int i = 0; i < swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        float random_dir = random_float(-1.0f, 1.0f);
        float improvement = 0.0f;
        
        for (int d = 0; d < dimensions; d++) {
            float range = swarm->config.upper_bounds[d] - swarm->config.lower_bounds[d];
            float step = current_step * random_dir * range * 0.1f;
            if (d < 3) step += (barycenter[d] - ind->position[d]) * 0.01f;
            ind->position[d] += step;
            if (d < 3) improvement += step * step;
        }
        swarm_apply_constraints(swarm, i);
    }
}


/* ========== BAT（蝙蝠算法）函数实现 ========== */

static BATData* bat_data_create(const SwarmConfig* config) {
    BATData* data = (BATData*)safe_malloc(sizeof(BATData));
    if (!data) return NULL;
    
    int swarm_size = config->swarm_size;
    int dimensions = config->dimensions;
    
    data->frequencies = (float*)safe_malloc(sizeof(float) * swarm_size);
    data->velocities = (float*)safe_malloc(sizeof(float) * swarm_size * dimensions);
    data->loudness = (float*)safe_malloc(sizeof(float) * swarm_size);
    data->pulse_rates = (float*)safe_malloc(sizeof(float) * swarm_size);
    data->pulse_rates_initial = (float*)safe_malloc(sizeof(float) * swarm_size);
    data->best_positions = (float*)safe_malloc(sizeof(float) * swarm_size * dimensions);
    data->best_fitnesses = (float*)safe_malloc(sizeof(float) * swarm_size);
    
    if (!data->frequencies || !data->velocities || !data->loudness ||
        !data->pulse_rates || !data->pulse_rates_initial ||
        !data->best_positions || !data->best_fitnesses) {
        bat_data_free(data);
        return NULL;
    }
    
    data->frequency_min = config->exploration_factor;
    data->frequency_max = config->exploitation_factor > data->frequency_min ?
                          config->exploitation_factor : data->frequency_min + 2.0f;
    data->loudness_initial = config->pheromone_evaporation > 0.0f ?
                             config->pheromone_evaporation : 0.5f;
    data->pulse_rate_initial = 0.5f;
    data->alpha = 0.95f;
    data->gamma = 0.05f;
    
    return data;
}

static void bat_data_free(BATData* data) {
    if (!data) return;
    safe_free((void**)&data->frequencies);
    safe_free((void**)&data->velocities);
    safe_free((void**)&data->loudness);
    safe_free((void**)&data->pulse_rates);
    safe_free((void**)&data->pulse_rates_initial);
    safe_free((void**)&data->best_positions);
    safe_free((void**)&data->best_fitnesses);
    safe_free((void**)&data);
}

static void bat_initialize(Swarm* swarm) {
    BATData* data = (BATData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    
    for (int i = 0; i < swarm_size; i++) {
        data->frequencies[i] = data->frequency_min +
            random_float(0.0f, 1.0f) * (data->frequency_max - data->frequency_min);
        data->loudness[i] = data->loudness_initial;
        data->pulse_rates[i] = data->pulse_rate_initial;
        data->pulse_rates_initial[i] = data->pulse_rate_initial;
        data->best_fitnesses[i] = FLT_MAX;
    }
}

static void bat_iterate(Swarm* swarm) {
    BATData* data = (BATData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    
    for (int i = 0; i < swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        
        data->frequencies[i] = data->frequency_min +
            random_float(0.0f, 1.0f) * (data->frequency_max - data->frequency_min);
        
        for (int d = 0; d < dimensions; d++) {
            float vel = data->velocities[i * dimensions + d];
            vel += (ind->position[d] - swarm->global_best_position[d]) * data->frequencies[i];
            data->velocities[i * dimensions + d] = vel;
            ind->position[d] += vel;
        }
        
        if (random_float(0.0f, 1.0f) > data->pulse_rates[i]) {
            for (int d = 0; d < dimensions; d++) {
                ind->position[d] = swarm->global_best_position[d] +
                    random_float(-1.0f, 1.0f) * data->loudness[i];
            }
        }
        
        swarm_apply_constraints(swarm, i);
        
        float new_fitness = swarm_calculate_fitness(swarm, i);
        if (new_fitness < ind->current_fitness &&
            random_float(0.0f, 1.0f) < data->loudness[i]) {
            data->loudness[i] *= data->alpha;
            data->pulse_rates[i] = data->pulse_rates_initial[i] *
                (1.0f - expf(-data->gamma * swarm->current_iteration));
        }
    }
}


/* ========== Firefly（萤火虫算法）函数实现 ========== */

static FireflyData* firefly_data_create(const SwarmConfig* config) {
    FireflyData* data = (FireflyData*)safe_malloc(sizeof(FireflyData));
    if (!data) return NULL;
    
    data->brightness = (float*)safe_malloc(sizeof(float) * config->swarm_size);
    if (!data->brightness) { safe_free((void**)&data); return NULL; }
    
    data->attractiveness_base = config->exploration_factor > 0.0f ?
                                config->exploration_factor : 1.0f;
    data->light_absorption = config->pheromone_evaporation > 0.0f ?
                             config->pheromone_evaporation : 0.01f;
    data->randomization_alpha = config->exploitation_factor > 0.0f ?
                                config->exploitation_factor : 0.2f;
    data->randomization_decay = 0.97f;
    data->max_attraction_distance = 100.0f;
    
    return data;
}

static void firefly_data_free(FireflyData* data) {
    if (!data) return;
    safe_free((void**)&data->brightness);
    safe_free((void**)&data);
}

static void firefly_initialize(Swarm* swarm) {
    FireflyData* data = (FireflyData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    for (int i = 0; i < swarm_size; i++) {
        data->brightness[i] = 1.0f / (1.0f + swarm->individuals[i].current_fitness);
    }
}

static void firefly_iterate(Swarm* swarm) {
    FireflyData* data = (FireflyData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    
    for (int i = 0; i < swarm_size; i++) {
        data->brightness[i] = 1.0f / (1.0f + swarm->individuals[i].current_fitness);
    }
    
    for (int i = 0; i < swarm_size; i++) {
        for (int j = 0; j < swarm_size; j++) {
            if (data->brightness[j] > data->brightness[i]) {
                float dist = euclidean_distance(
                    swarm->individuals[i].position,
                    swarm->individuals[j].position, dimensions);
                if (dist > data->max_attraction_distance) continue;
                
                float beta = data->attractiveness_base *
                    expf(-data->light_absorption * dist * dist);
                
                for (int d = 0; d < dimensions; d++) {
                    float diff = swarm->individuals[j].position[d] -
                                 swarm->individuals[i].position[d];
                    swarm->individuals[i].position[d] +=
                        beta * diff +
                        data->randomization_alpha * random_float(-0.5f, 0.5f);
                }
            }
        }
        swarm_apply_constraints(swarm, i);
    }
    
    data->randomization_alpha *= data->randomization_decay;
}


/* ========== Cuckoo（布谷鸟搜索）函数实现 ========== */

static CuckooData* cuckoo_data_create(const SwarmConfig* config) {
    CuckooData* data = (CuckooData*)safe_malloc(sizeof(CuckooData));
    if (!data) return NULL;
    
    int swarm_size = config->swarm_size;
    int dimensions = config->dimensions;
    
    data->levy_steps = (float*)safe_malloc(sizeof(float) * dimensions);
    data->prev_positions = (float*)safe_malloc(sizeof(float) * swarm_size * dimensions);
    data->best_positions = (float*)safe_malloc(sizeof(float) * swarm_size * dimensions);
    data->best_fitnesses = (float*)safe_malloc(sizeof(float) * swarm_size);
    
    if (!data->levy_steps || !data->prev_positions ||
        !data->best_positions || !data->best_fitnesses) {
        cuckoo_data_free(data);
        return NULL;
    }
    
    data->step_size = config->exploration_factor > 0.0f ?
                      config->exploration_factor : 0.01f;
    data->discovery_probability = config->exploitation_factor > 0.0f ?
                                  config->exploitation_factor : 0.25f;
    data->levy_exponent = 1.5f;
    data->levy_scale = 0.01f;
    data->use_levy_flights = 1;
    
    return data;
}

static void cuckoo_data_free(CuckooData* data) {
    if (!data) return;
    safe_free((void**)&data->levy_steps);
    safe_free((void**)&data->prev_positions);
    safe_free((void**)&data->best_positions);
    safe_free((void**)&data->best_fitnesses);
    safe_free((void**)&data);
}

static void cuckoo_initialize(Swarm* swarm) {
    CuckooData* data = (CuckooData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    
    for (int i = 0; i < swarm_size; i++) {
        memcpy(&data->prev_positions[i * dimensions],
               swarm->individuals[i].position, sizeof(float) * dimensions);
        memcpy(&data->best_positions[i * dimensions],
               swarm->individuals[i].position, sizeof(float) * dimensions);
        data->best_fitnesses[i] = swarm->individuals[i].current_fitness;
    }
}

static void cuckoo_iterate(Swarm* swarm) {
    CuckooData* data = (CuckooData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    
    for (int i = 0; i < swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        memcpy(&data->prev_positions[i * dimensions], ind->position,
               sizeof(float) * dimensions);
        
        for (int d = 0; d < dimensions; d++) {
            float u = random_float(0.0f, 1.0f) * 0.01f;
            float v = random_float(0.0f, 1.0f);
            float levy = u / powf(fabsf(v) + 1e-10f, 1.0f / data->levy_exponent);
            ind->position[d] += data->step_size * levy *
                (ind->position[d] - swarm->global_best_position[d]);
        }
        swarm_apply_constraints(swarm, i);
    }
    
    for (int i = 0; i < swarm_size; i++) {
        if (random_float(0.0f, 1.0f) < data->discovery_probability) {
            int j = random_int(0, swarm_size - 1);
            int k = random_int(0, swarm_size - 1);
            for (int d = 0; d < dimensions; d++) {
                float diff = swarm->individuals[j].position[d] -
                             swarm->individuals[k].position[d];
                if (random_float(0.0f, 1.0f) > data->discovery_probability) {
                    swarm->individuals[i].position[d] +=
                        random_float(0.0f, 1.0f) * diff;
                }
            }
            swarm_apply_constraints(swarm, i);
        }
    }
}


/* ========== GWO（灰狼优化器）函数实现 ========== */

static GWOData* gwo_data_create(const SwarmConfig* config) {
    GWOData* data = (GWOData*)safe_malloc(sizeof(GWOData));
    if (!data) return NULL;
    
    int dimensions = config->dimensions;
    
    data->alpha_position = (float*)safe_malloc(sizeof(float) * dimensions);
    data->beta_position = (float*)safe_malloc(sizeof(float) * dimensions);
    data->delta_position = (float*)safe_malloc(sizeof(float) * dimensions);
    
    if (!data->alpha_position || !data->beta_position || !data->delta_position) {
        gwo_data_free(data);
        return NULL;
    }
    
    data->alpha_fitness = FLT_MAX;
    data->beta_fitness = FLT_MAX;
    data->delta_fitness = FLT_MAX;
    data->a_initial = config->exploration_factor > 0.0f ? config->exploration_factor : 2.0f;
    data->a_final = config->exploitation_factor >= 0.0f ? config->exploitation_factor : 0.0f;
    data->a_linear = data->a_initial;
    data->alpha_idx = -1;
    data->beta_idx = -1;
    data->delta_idx = -1;
    
    return data;
}

static void gwo_data_free(GWOData* data) {
    if (!data) return;
    safe_free((void**)&data->alpha_position);
    safe_free((void**)&data->beta_position);
    safe_free((void**)&data->delta_position);
    safe_free((void**)&data);
}

static void gwo_initialize(Swarm* swarm) {
    GWOData* data = (GWOData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    
    data->alpha_fitness = FLT_MAX;
    data->beta_fitness = FLT_MAX;
    data->delta_fitness = FLT_MAX;
    data->alpha_idx = -1;
    data->beta_idx = -1;
    data->delta_idx = -1;
    
    for (int i = 0; i < swarm_size; i++) {
        float fit = swarm->individuals[i].current_fitness;
        if (fit < data->alpha_fitness) {
            data->delta_fitness = data->beta_fitness;
            memcpy(data->delta_position, data->beta_position, sizeof(float) * dimensions);
            data->beta_fitness = data->alpha_fitness;
            memcpy(data->beta_position, data->alpha_position, sizeof(float) * dimensions);
            data->alpha_fitness = fit;
            memcpy(data->alpha_position, swarm->individuals[i].position, sizeof(float) * dimensions);
            data->alpha_idx = i;
        } else if (fit < data->beta_fitness) {
            data->delta_fitness = data->beta_fitness;
            memcpy(data->delta_position, data->beta_position, sizeof(float) * dimensions);
            data->beta_fitness = fit;
            memcpy(data->beta_position, swarm->individuals[i].position, sizeof(float) * dimensions);
            data->beta_idx = i;
        } else if (fit < data->delta_fitness) {
            data->delta_fitness = fit;
            memcpy(data->delta_position, swarm->individuals[i].position, sizeof(float) * dimensions);
            data->delta_idx = i;
        }
    }
}

static void gwo_iterate(Swarm* swarm) {
    GWOData* data = (GWOData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    float iter_ratio = (float)swarm->current_iteration / (float)swarm->config.max_iterations;
    data->a_linear = data->a_initial - iter_ratio * (data->a_initial - data->a_final);
    
    gwo_initialize(swarm);
    
    for (int i = 0; i < swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        for (int d = 0; d < dimensions; d++) {
            float r1 = random_float(0.0f, 1.0f);
            float r2 = random_float(0.0f, 1.0f);
            float A1 = 2.0f * data->a_linear * r1 - data->a_linear;
            float C1 = 2.0f * r2;
            float D_alpha = fabsf(C1 * data->alpha_position[d] - ind->position[d]);
            float X1 = data->alpha_position[d] - A1 * D_alpha;
            
            r1 = random_float(0.0f, 1.0f);
            r2 = random_float(0.0f, 1.0f);
            float A2 = 2.0f * data->a_linear * r1 - data->a_linear;
            float C2 = 2.0f * r2;
            float D_beta = fabsf(C2 * data->beta_position[d] - ind->position[d]);
            float X2 = data->beta_position[d] - A2 * D_beta;
            
            r1 = random_float(0.0f, 1.0f);
            r2 = random_float(0.0f, 1.0f);
            float A3 = 2.0f * data->a_linear * r1 - data->a_linear;
            float C3 = 2.0f * r2;
            float D_delta = fabsf(C3 * data->delta_position[d] - ind->position[d]);
            float X3 = data->delta_position[d] - A3 * D_delta;
            
            ind->position[d] = (X1 + X2 + X3) / 3.0f;
        }
        swarm_apply_constraints(swarm, i);
    }
}


/* ========== WOA（鲸鱼优化算法）函数实现 ========== */

static WOAData* woa_data_create(const SwarmConfig* config) {
    WOAData* data = (WOAData*)safe_malloc(sizeof(WOAData));
    if (!data) return NULL;
    
    int swarm_size = config->swarm_size;
    int dimensions = config->dimensions;
    
    data->best_positions = (float*)safe_malloc(sizeof(float) * swarm_size * dimensions);
    data->best_fitnesses = (float*)safe_malloc(sizeof(float) * swarm_size);
    
    if (!data->best_positions || !data->best_fitnesses) {
        woa_data_free(data);
        return NULL;
    }
    
    data->spiral_constant = config->exploitation_factor > 0.0f ?
                            config->exploitation_factor : 1.0f;
    data->a_linear = config->exploration_factor > 0.0f ? config->exploration_factor : 2.0f;
    data->a2_linear = -1.0f;
    data->bubble_net_probability = config->pheromone_evaporation > 0.0f ?
                                   config->pheromone_evaporation : 0.5f;
    data->leader_dim = dimensions > 3 ? 3 : dimensions;
    memset(data->leader_position, 0, sizeof(float) * 3);
    
    return data;
}

static void woa_data_free(WOAData* data) {
    if (!data) return;
    safe_free((void**)&data->best_positions);
    safe_free((void**)&data->best_fitnesses);
    safe_free((void**)&data);
}

static void woa_initialize(Swarm* swarm) {
    WOAData* data = (WOAData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    
    for (int i = 0; i < swarm_size; i++) {
        memcpy(&data->best_positions[i * dimensions],
               swarm->individuals[i].position, sizeof(float) * dimensions);
        data->best_fitnesses[i] = swarm->individuals[i].current_fitness;
    }
    memcpy(data->leader_position, swarm->global_best_position,
           sizeof(float) * data->leader_dim);
}

static void woa_iterate(Swarm* swarm) {
    WOAData* data = (WOAData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    float iter_ratio = (float)swarm->current_iteration / (float)swarm->config.max_iterations;
    float a = data->a_linear * (1.0f - iter_ratio);
    
    for (int i = 0; i < swarm_size; i++) {
        SwarmIndividual* ind = &swarm->individuals[i];
        float r = random_float(0.0f, 1.0f);
        
        if (r < data->bubble_net_probability) {
            float p = random_float(0.0f, 1.0f);
            if (p < 0.5f) {
                for (int d = 0; d < dimensions; d++) {
                    float A = 2.0f * a * random_float(0.0f, 1.0f) - a;
                    float C = 2.0f * random_float(0.0f, 1.0f);
                    float D = fabsf(C * swarm->global_best_position[d] - ind->position[d]);
                    ind->position[d] = swarm->global_best_position[d] - A * D;
                }
            } else {
                for (int d = 0; d < dimensions; d++) {
                    float dist = fabsf(swarm->global_best_position[d] - ind->position[d]);
                    float l = random_float(-1.0f, 1.0f);
                    ind->position[d] = dist * expf(data->spiral_constant * l) *
                        cosf(2.0f * 3.14159265f * l) +
                        swarm->global_best_position[d];
                }
            }
        } else {
            int rand_idx = random_int(0, swarm_size - 1);
            for (int d = 0; d < dimensions; d++) {
                float A = 2.0f * a * random_float(0.0f, 1.0f) - a;
                float C = 2.0f * random_float(0.0f, 1.0f);
                float D = fabsf(C * swarm->individuals[rand_idx].position[d] - ind->position[d]);
                ind->position[d] = swarm->individuals[rand_idx].position[d] - A * D;
            }
        }
        swarm_apply_constraints(swarm, i);
    }
}


/* ========== MARL（多智能体强化学习）函数实现 ========== */

static MARLData* marl_data_create(const SwarmConfig* config) {
    MARLData* data = (MARLData*)safe_malloc(sizeof(MARLData));
    if (!data) return NULL;
    
    int swarm_size = config->swarm_size;
    
    data->num_states = config->dimensions * 10;
    data->num_actions = config->dimensions * 5;
    if (data->num_states < 10) data->num_states = 10;
    if (data->num_actions < 5) data->num_actions = 5;
    data->q_table_size = (size_t)data->num_states * (size_t)data->num_actions;
    
    data->q_table = (float*)safe_malloc(data->q_table_size * sizeof(float));
    data->eligibility_traces = (float*)safe_malloc(data->q_table_size * sizeof(float));
    data->state_indices = (int*)safe_malloc(sizeof(int) * swarm_size);
    data->action_indices = (int*)safe_malloc(sizeof(int) * swarm_size);
    data->prev_states = (int*)safe_malloc(sizeof(int) * swarm_size);
    data->prev_actions = (int*)safe_malloc(sizeof(int) * swarm_size);
    
    if (!data->q_table || !data->eligibility_traces || !data->state_indices ||
        !data->action_indices || !data->prev_states || !data->prev_actions) {
        marl_data_free(data);
        return NULL;
    }
    
    memset(data->q_table, 0, data->q_table_size * sizeof(float));
    memset(data->eligibility_traces, 0, data->q_table_size * sizeof(float));
    
    data->learning_rate = config->exploration_factor > 0.0f ?
                          config->exploration_factor : 0.1f;
    data->discount_factor = config->exploitation_factor > 0.0f ?
                            config->exploitation_factor : 0.9f;
    data->exploration_rate = config->inertia_weight > 0.0f ?
                             config->inertia_weight : 0.1f;
    data->exploration_decay = 0.995f;
    data->exploration_min = 0.01f;
    data->lambda = 0.9f;
    data->use_softmax_policy = 1;
    data->temperature = 1.0f;
    
    return data;
}

static void marl_data_free(MARLData* data) {
    if (!data) return;
    safe_free((void**)&data->q_table);
    safe_free((void**)&data->eligibility_traces);
    safe_free((void**)&data->state_indices);
    safe_free((void**)&data->action_indices);
    safe_free((void**)&data->prev_states);
    safe_free((void**)&data->prev_actions);
    safe_free((void**)&data);
}

static void marl_initialize(Swarm* swarm) {
    MARLData* data = (MARLData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    
    for (int i = 0; i < swarm_size; i++) {
        data->state_indices[i] = rng_next() % (uint64_t)(data)->num_states;
        data->action_indices[i] = rng_next() % (uint64_t)(data)->num_actions;
        data->prev_states[i] = data->state_indices[i];
        data->prev_actions[i] = data->action_indices[i];
    }
}

static void marl_iterate(Swarm* swarm) {
    MARLData* data = (MARLData*)swarm->algorithm_data;
    if (!data) return;
    int swarm_size = swarm->config.swarm_size;
    int dimensions = swarm->config.dimensions;
    
    for (int i = 0; i < swarm_size; i++) {
        int s = data->state_indices[i];
        int a = data->action_indices[i];
        int prev_s = data->prev_states[i];
        int prev_a = data->prev_actions[i];
        
        float reward = -swarm->individuals[i].current_fitness;
        float max_q_next = -FLT_MAX;
        for (int a2 = 0; a2 < data->num_actions; a2++) {
            float q_val = data->q_table[s * data->num_actions + a2];
            if (q_val > max_q_next) max_q_next = q_val;
        }
        
        float td_error = reward + data->discount_factor * max_q_next -
                         data->q_table[prev_s * data->num_actions + prev_a];
        
        size_t trace_idx = (size_t)prev_s * (size_t)data->num_actions + (size_t)prev_a;
        data->eligibility_traces[trace_idx] += 1.0f;
        
        for (int s2 = 0; s2 < data->num_states; s2++) {
            for (int a2 = 0; a2 < data->num_actions; a2++) {
                size_t idx = (size_t)s2 * (size_t)data->num_actions + (size_t)a2;
                data->q_table[idx] += data->learning_rate * td_error *
                                      data->eligibility_traces[idx];
                data->eligibility_traces[idx] *= data->discount_factor * data->lambda;
            }
        }
        
        for (int d = 0; d < dimensions; d++) {
            float action_val = (float)a / (float)data->num_actions - 0.5f;
            swarm->individuals[i].position[d] += action_val * 0.1f *
                (swarm->config.upper_bounds[d] - swarm->config.lower_bounds[d]);
        }
        swarm_apply_constraints(swarm, i);
        
        data->prev_states[i] = s;
        data->prev_actions[i] = a;
        
        if (random_float(0.0f, 1.0f) < data->exploration_rate) {
            data->action_indices[i] = rng_next() % (uint64_t)(data)->num_actions;
        } else {
            int best_a = 0;
            float best_q = -FLT_MAX;
            for (int a2 = 0; a2 < data->num_actions; a2++) {
                float q_val = data->q_table[s * data->num_actions + a2];
                if (q_val > best_q) { best_q = q_val; best_a = a2; }
            }
            data->action_indices[i] = best_a;
        }
    }
    
    data->exploration_rate *= data->exploration_decay;
    if (data->exploration_rate < data->exploration_min)
        data->exploration_rate = data->exploration_min;
}


/* ========== 工具函数实现 ========== */

/**
 * @brief 生成随机浮点数
 */
static float random_float(float min, float max) {
    float scale = rng_uniform(0.0f, 1.0f);
    return min + scale * (max - min);
}

/**
 * @brief 生成随机整数
 */
static int random_int(int min, int max) {
    return min + rng_next() % (max - min + 1);
}

/**
 * @brief 限制位置在边界内
 */
static void clamp_position(float* position, const float* lower_bounds, 
                          const float* upper_bounds, int dimensions) {
    for (int d = 0; d < dimensions; d++) {
        if (position[d] < lower_bounds[d]) {
            position[d] = lower_bounds[d];
        } else if (position[d] > upper_bounds[d]) {
            position[d] = upper_bounds[d];
        }
    }
}

/**
 * @brief 计算欧几里得距离
 */
static float euclidean_distance(const float* a, const float* b, int dimensions) {
    float sum = 0.0f;
    for (int i = 0; i < dimensions; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

/**
 * @brief 复制位置
 */
static void copy_position(float* dest, const float* src, int dimensions) {
    memcpy(dest, src, sizeof(float) * dimensions);
}

/* ========== 算法特定创建函数 ========== */

Swarm* swarm_create_pso(const SwarmConfig* config) {
    SwarmConfig modified_config = *config;
    modified_config.algorithm_type = SWARM_ALGORITHM_PSO;
    if (modified_config.inertia_weight <= 0.0f) modified_config.inertia_weight = 0.7f;
    if (modified_config.cognitive_weight <= 0.0f) modified_config.cognitive_weight = 1.5f;
    if (modified_config.social_weight <= 0.0f) modified_config.social_weight = 1.5f;
    if (modified_config.exploration_factor <= 0.0f) modified_config.exploration_factor = 0.9f;
    if (modified_config.exploitation_factor <= 0.0f) modified_config.exploitation_factor = 0.4f;
    return swarm_create(&modified_config);
}

Swarm* swarm_create_aco(const SwarmConfig* config) {
    SwarmConfig modified_config = *config;
    modified_config.algorithm_type = SWARM_ALGORITHM_ACO;
    if (modified_config.pheromone_evaporation <= 0.0f) modified_config.pheromone_evaporation = 0.5f;
    if (modified_config.exploration_factor <= 0.0f) modified_config.exploration_factor = 1.0f;
    if (modified_config.exploitation_factor <= 0.0f) modified_config.exploitation_factor = 2.0f;
    modified_config.inertia_weight = 0.0f;
    modified_config.cognitive_weight = 0.0f;
    modified_config.social_weight = 0.0f;
    return swarm_create(&modified_config);
}

Swarm* swarm_create_abc(const SwarmConfig* config) {
    SwarmConfig modified_config = *config;
    modified_config.algorithm_type = SWARM_ALGORITHM_ABC;
    if (modified_config.exploration_factor <= 0.0f) modified_config.exploration_factor = 0.5f;
    if (modified_config.exploitation_factor <= 0.0f) modified_config.exploitation_factor = 100.0f;
    modified_config.inertia_weight = 0.0f;
    modified_config.cognitive_weight = 0.0f;
    modified_config.social_weight = 0.0f;
    modified_config.pheromone_evaporation = 0.0f;
    return swarm_create(&modified_config);
}

Swarm* swarm_create_fss(const SwarmConfig* config) {
    SwarmConfig modified_config = *config;
    modified_config.algorithm_type = SWARM_ALGORITHM_FSS;
    if (modified_config.exploration_factor <= 0.0f) modified_config.exploration_factor = 0.1f;
    if (modified_config.exploitation_factor <= 0.0f) modified_config.exploitation_factor = 0.01f;
    modified_config.inertia_weight = 0.0f;
    modified_config.cognitive_weight = 0.0f;
    modified_config.social_weight = 0.0f;
    modified_config.pheromone_evaporation = 0.0f;
    return swarm_create(&modified_config);
}

Swarm* swarm_create_bat(const SwarmConfig* config) {
    SwarmConfig modified_config = *config;
    modified_config.algorithm_type = SWARM_ALGORITHM_BAT;
    if (modified_config.exploration_factor <= 0.0f) modified_config.exploration_factor = 0.0f;
    if (modified_config.exploitation_factor <= 0.0f) modified_config.exploitation_factor = 2.0f;
    if (modified_config.pheromone_evaporation <= 0.0f) modified_config.pheromone_evaporation = 0.9f;
    modified_config.inertia_weight = 0.0f;
    modified_config.cognitive_weight = 0.0f;
    modified_config.social_weight = 0.0f;
    return swarm_create(&modified_config);
}

Swarm* swarm_create_firefly(const SwarmConfig* config) {
    SwarmConfig modified_config = *config;
    modified_config.algorithm_type = SWARM_ALGORITHM_FIREFLY;
    if (modified_config.exploration_factor <= 0.0f) modified_config.exploration_factor = 1.0f;
    if (modified_config.exploitation_factor <= 0.0f) modified_config.exploitation_factor = 1.0f;
    if (modified_config.pheromone_evaporation <= 0.0f) modified_config.pheromone_evaporation = 0.01f;
    modified_config.inertia_weight = 0.0f;
    modified_config.cognitive_weight = 0.0f;
    modified_config.social_weight = 0.0f;
    return swarm_create(&modified_config);
}

Swarm* swarm_create_cuckoo(const SwarmConfig* config) {
    SwarmConfig modified_config = *config;
    modified_config.algorithm_type = SWARM_ALGORITHM_CUCKOO;
    if (modified_config.exploration_factor <= 0.0f) modified_config.exploration_factor = 0.01f;
    if (modified_config.exploitation_factor <= 0.0f) modified_config.exploitation_factor = 0.25f;
    modified_config.inertia_weight = 0.0f;
    modified_config.cognitive_weight = 0.0f;
    modified_config.social_weight = 0.0f;
    modified_config.pheromone_evaporation = 0.0f;
    return swarm_create(&modified_config);
}

Swarm* swarm_create_gwo(const SwarmConfig* config) {
    SwarmConfig modified_config = *config;
    modified_config.algorithm_type = SWARM_ALGORITHM_GWO;
    if (modified_config.exploration_factor <= 0.0f) modified_config.exploration_factor = 2.0f;
    if (modified_config.exploitation_factor <= 0.0f) modified_config.exploitation_factor = 0.0f;
    modified_config.inertia_weight = 0.0f;
    modified_config.cognitive_weight = 0.0f;
    modified_config.social_weight = 0.0f;
    modified_config.pheromone_evaporation = 0.0f;
    return swarm_create(&modified_config);
}

Swarm* swarm_create_woa(const SwarmConfig* config) {
    SwarmConfig modified_config = *config;
    modified_config.algorithm_type = SWARM_ALGORITHM_WOA;
    if (modified_config.exploration_factor <= 0.0f) modified_config.exploration_factor = 2.0f;
    if (modified_config.exploitation_factor <= 0.0f) modified_config.exploitation_factor = 1.0f;
    if (modified_config.pheromone_evaporation <= 0.0f) modified_config.pheromone_evaporation = 0.5f;
    modified_config.inertia_weight = 0.0f;
    modified_config.cognitive_weight = 0.0f;
    modified_config.social_weight = 0.0f;
    return swarm_create(&modified_config);
}

Swarm* swarm_create_marl(const SwarmConfig* config) {
    SwarmConfig modified_config = *config;
    modified_config.algorithm_type = SWARM_ALGORITHM_MARL;
    if (modified_config.exploration_factor <= 0.0f) modified_config.exploration_factor = 0.1f;
    if (modified_config.exploitation_factor <= 0.0f) modified_config.exploitation_factor = 0.9f;
    if (modified_config.inertia_weight <= 0.0f) modified_config.inertia_weight = 0.99f;
    modified_config.cognitive_weight = 0.0f;
    modified_config.social_weight = 0.0f;
    modified_config.pheromone_evaporation = 0.0f;
    return swarm_create(&modified_config);
}