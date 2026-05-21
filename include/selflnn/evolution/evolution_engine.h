/**
 * @file evolution_engine.h
 * @brief 增强自我演化进化引擎接口
 */

#ifndef SELFLNN_EVOLUTION_ENGINE_H
#define SELFLNN_EVOLUTION_ENGINE_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 遗传操作类型 */
typedef enum {
    EVO_SELECTION_ROULETTE = 0,      /* 轮盘赌选择 */
    EVO_SELECTION_TOURNAMENT = 1,    /* 锦标赛选择 */
    EVO_SELECTION_RANK = 2,          /* 排名选择 */
    EVO_SELECTION_ELITE = 3,         /* 精英选择 */
    EVO_SELECTION_BOLTZMANN = 4      /* 玻尔兹曼选择 */
} SelectionMethod;

typedef enum {
    EVO_CROSSOVER_SINGLE = 0,        /* 单点交叉 */
    EVO_CROSSOVER_MULTI = 1,         /* 多点交叉 */
    EVO_CROSSOVER_UNIFORM = 2,       /* 均匀交叉 */
    EVO_CROSSOVER_ARITHMETIC = 3,    /* 算术交叉 */
    EVO_CROSSOVER_SBX = 4,           /* 模拟二进制交叉 */
    EVO_CROSSOVER_BLEND = 5          /* 混合交叉 */
} CrossoverMethod;

typedef enum {
    EVO_MUTATION_GAUSSIAN = 0,       /* 高斯变异 */
    EVO_MUTATION_UNIFORM = 1,        /* 均匀变异 */
    EVO_MUTATION_POLYNOMIAL = 2,     /* 多项式变异 */
    EVO_MUTATION_SWAP = 3,           /* 交换变异 */
    EVO_MUTATION_INVERSION = 4,      /* 反转变异 */
    EVO_MUTATION_SCRAMBLE = 5,       /* 打乱变异 */
    EVO_MUTATION_ADAPTIVE = 6        /* 自适应变异 */
} MutationMethod;

/* 适应度类型 */
typedef enum {
    EVO_FITNESS_MAXIMIZE = 0,        /* 最大化 */
    EVO_FITNESS_MINIMIZE = 1         /* 最小化 */
} FitnessDirection;

/* 个体 */
typedef struct {
    float* chromosome;               /* 染色体/基因 */
    size_t chromosome_size;          /* 染色体大小 */
    float fitness;                   /* 适应度 */
    float* objectives;               /* 多目标适应度 */
    int objective_count;             /* 目标数量 */
    float crowding_distance;         /* 拥挤距离 */
    int pareto_rank;                 /* 帕累托等级 */
    int age;                         /* 年龄 */
    time_t birth_time;               /* 出生时间 */
    int id;                          /* 个体ID */
} EvolutionIndividual;

/* 种群 */
typedef struct {
    EvolutionIndividual* individuals; /* 个体数组 */
    size_t size;                     /* 当前大小 */
    size_t capacity;                 /* 容量 */
    int generation;                  /* 当前代数 */
    float best_fitness;              /* 历史最佳适应度 */
    float avg_fitness;               /* 平均适应度 */
    float std_fitness;               /* 适应度标准差 */
    float diversity;                 /* 种群多样性 */
    int stagnated_generations;       /* 停滞代数 */
    int total_evaluations;           /* 总评估次数 */
} EvolutionPopulation;

/* 演化配置 */
typedef struct {
    SelectionMethod selection;       /* 选择方法 */
    CrossoverMethod crossover;       /* 交叉方法 */
    MutationMethod mutation;         /* 变异方法 */
    FitnessDirection direction;      /* 适应度方向 */

    size_t population_size;          /* 种群大小 */
    size_t chromosome_size;          /* 染色体大小 */
    int max_generations;             /* 最大代数 */
    int elite_count;                 /* 精英保留数 */

    float chromosome_min;            /* 染色体取值范围下限 */
    float chromosome_max;            /* 染色体取值范围上限 */

    float crossover_rate;            /* 交叉率 (0-1) */
    float mutation_rate;             /* 初始变异率 (0-1) */
    float mutation_rate_min;         /* 最小变异率 */
    float mutation_rate_max;         /* 最大变异率 */
    float mutation_decay;            /* 变异率衰减 */

    float tournament_size_ratio;     /* 锦标赛大小比例 */
    float sbx_eta;                   /* SBX分布指数 */
    float polynomial_eta;            /* 多项式变异分布指数 */
    float gaussian_sigma;            /* 高斯变异标准差 */
    float gaussian_sigma_decay;      /* 高斯变异标准差衰减 */

    int use_adaptive_mutation;       /* 自适应变异 */
    int use_restart_stagnation;      /* 停滞重启 */
    int restart_stagnation_generations; /* 重启停滞代数 */
    int use_island_model;            /* 岛模型 */
    int island_count;                /* 岛屿数量 */
    int migration_interval;          /* 迁移间隔 */
    int migration_size;              /* 迁移大小 */
} EvolutionConfig;

/* 演化统计 */
typedef struct {
    size_t total_generations;        /* 总代数 */
    size_t total_evaluations;        /* 总评估次数 */
    float initial_best_fitness;      /* 初始最佳适应度 */
    float final_best_fitness;        /* 最终最佳适应度 */
    float improvement;               /* 改进幅度 */
    float convergence_speed;         /* 收敛速度 */
    float* fitness_history;          /* 适应度历史 */
    size_t history_size;             /* 历史大小 */
    float* diversity_history;        /* 多样性历史 */
    size_t diversity_size;           /* 多样性历史大小 */
    time_t start_time;               /* 开始时间 */
    time_t end_time;                 /* 结束时间 */
    double elapsed_seconds;          /* 耗时 */
} EvolutionStats;

/* 演化引擎句柄 */
typedef struct EvolutionEngine EvolutionEngine;

/* 适应度评估回调 */
typedef float (*FitnessFunction)(const float* chromosome, size_t size, void* user_data);

/* 多目标适应度评估回调 */
typedef void (*MultiFitnessFunction)(const float* chromosome, size_t size, 
                                     float* objectives, int obj_count, void* user_data);

/**
 * @brief 创建演化引擎
 */
EvolutionEngine* evolution_engine_create(const EvolutionConfig* config);

/**
 * @brief 释放演化引擎
 */
void evolution_engine_free(EvolutionEngine* engine);

/**
 * @brief 设置适应度函数
 */
int evolution_set_fitness_function(EvolutionEngine* engine, FitnessFunction func, void* user_data);

/**
 * @brief 设置多目标适应度函数
 */
int evolution_set_multi_fitness_function(EvolutionEngine* engine, MultiFitnessFunction func, 
                                         int obj_count, void* user_data);

/* F-010/F-019: CMA-ES模式控制API */
/**
 * @brief 启用CMA-ES（协方差矩阵自适应进化策略）模式
 *
 * 启用后evolution_step将使用core/cma_es.c的CMA-ES算法替代GA，
 * 包含协方差矩阵自适应更新、演化路径(pc/pσ)和步长控制。
 *
 * @param engine 演化引擎
 * @param initial_sigma 初始步长，默认CMAES_DEFAULT_SIGMA(0.3)
 * @param lambda 种群大小，0表示自动选择(4+3*ln(dim))
 * @return 0成功，-1失败
 */
int evolution_engine_enable_cmaes(EvolutionEngine* engine, float initial_sigma, int lambda);

/**
 * @brief 禁用CMA-ES模式，回退到GA
 * @param engine 演化引擎
 * @return 0成功，-1失败
 */
int evolution_engine_disable_cmaes(EvolutionEngine* engine);

/**
 * @brief 初始化种群
 */
int evolution_initialize_population(EvolutionEngine* engine, float min_val, float max_val);

/**
 * @brief 用已有个体初始化种群
 */
int evolution_initialize_from_existing(EvolutionEngine* engine, 
                                       const float** chromosomes, size_t count);

/**
 * @brief 执行一代演化
 */
int evolution_step(EvolutionEngine* engine);

/**
 * @brief 执行完整演化
 */
int evolution_run(EvolutionEngine* engine, int max_generations);

/**
 * @brief 获取最佳个体
 */
const EvolutionIndividual* evolution_get_best(const EvolutionEngine* engine);

/**
 * @brief 获取帕累托前沿
 */
int evolution_get_pareto_front(const EvolutionEngine* engine, 
                               EvolutionIndividual** front, size_t* front_size);

/**
 * @brief 获取种群
 */
const EvolutionPopulation* evolution_get_population(const EvolutionEngine* engine);

/**
 * @brief 获取演化统计
 */
int evolution_get_stats(const EvolutionEngine* engine, EvolutionStats* stats);

/**
 * @brief 精英个体注入
 */
int evolution_inject_elite(EvolutionEngine* engine, const float* chromosome, size_t size);

/**
 * @brief 调整变异率
 */
int evolution_set_mutation_rate(EvolutionEngine* engine, float rate);

/**
 * @brief 保存种群到文件
 */
int evolution_save_population(const EvolutionEngine* engine, const char* filepath);

/**
 * @brief 从文件加载种群
 */
int evolution_load_population(EvolutionEngine* engine, const char* filepath);

/**
 * @brief 重置演化引擎
 */
int evolution_reset(EvolutionEngine* engine);

/**
 * @brief 设置演化目标LNN（将演化结果写回液态神经网络）
 * @param engine 演化引擎句柄
 * @param lnn 目标LNN实例指针
 * @return 成功返回0，失败返回-1
 */
int evolution_engine_set_target_lnn(EvolutionEngine* engine, void* lnn);

/**
 * @brief 将当前最优个体的染色体写入LNN权重
 * 调用此函数前必须先调用 evolution_engine_set_target_lnn() 设置目标LNN
 * @param engine 演化引擎句柄
 * @return 成功返回0，失败返回-1
 */
int evolution_engine_apply_best_to_lnn(EvolutionEngine* engine);

/* 能力开关 (P1-05) */
int evolution_engine_enable(EvolutionEngine* engine);
int evolution_engine_disable(EvolutionEngine* engine);
int evolution_engine_is_enabled(const EvolutionEngine* engine);
int evolution_evaluate_environment(EvolutionEngine* engine, const float* environment_state,
                                    size_t state_dim, float* fitness_out);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_EVOLUTION_ENGINE_H */
