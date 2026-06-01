#ifndef SELFLNN_EVOLUTIONARY_ALGORITHMS_H
#define SELFLNN_EVOLUTIONARY_ALGORITHMS_H

#include "selflnn/core/common.h"
#include "selflnn/evolution/neural_architecture_search.h"

/* 前向声明LNN类型，避免循环依赖 */
#ifndef SELFLNN_LNN_H
typedef struct LNN LNN;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file evolutionary_algorithms.h
 * @brief 自我演化进化算法接口
 * 
 * 实现遗传算法、神经进化和进化策略，支持AGI系统的自我演化和进化。
 */

/**
 * @brief 适应度函数类型
 * 
 * @param genome 基因组数组
 * @param genome_size 基因组大小
 * @param user_data 用户数据
 * @return float 适应度分数（越高越好）
 */
typedef float (*FitnessFunction)(const float* genome, size_t genome_size, void* user_data);

/**
 * @brief 个体句柄（不透明类型）
 */
typedef struct Individual Individual;

/**
 * @brief 种群句柄（不透明类型）
 */
typedef struct Population Population;

/**
 * @brief 种群统计信息
 */
typedef struct {
    float best_fitness;             /**< 最佳适应度 */
    float average_fitness;          /**< 平均适应度 */
    float fitness_stddev;           /**< 适应度标准差 */
    float diversity_score;          /**< 多样性分数 */
    int current_generation;         /**< 当前代数 */
    size_t population_size;         /**< 种群大小 */
} PopulationStatistics;

/**
 * @brief 创建个体
 * 
 * @param genome_size 基因组大小
 * @param auxiliary_size 辅助数据大小
 * @return Individual* 个体指针，失败返回NULL
 */
Individual* individual_create(size_t genome_size, size_t auxiliary_size);

/**
 * @brief 销毁个体
 * 
 * @param ind 个体指针
 */
void individual_destroy(Individual* ind);

/**
 * @brief 初始化个体基因组（随机初始化）
 * 
 * @param ind 个体指针
 * @param min_val 最小值
 * @param max_val 最大值
 * @return int 成功返回0，失败返回-1
 */
int individual_initialize_random(Individual* ind, float min_val, float max_val);

/**
 * @brief 复制个体
 * 
 * @param src 源个体
 * @return Individual* 新个体指针，失败返回NULL
 */
Individual* individual_clone(const Individual* src);

/**
 * @brief 突变个体
 * 
 * @param ind 个体指针
 * @param mutation_rate 突变率
 * @param mutation_strength 突变强度
 * @return int 成功返回0，失败返回-1
 */
int individual_mutate(Individual* ind, float mutation_rate, float mutation_strength);

/**
 * @brief 交叉两个个体产生子代
 * 
 * @param parent1 父代1
 * @param parent2 父代2
 * @param crossover_rate 交叉率
 * @return Individual* 子代个体指针，失败返回NULL
 */
Individual* individual_crossover(const Individual* parent1, const Individual* parent2, 
                                 float crossover_rate);

/**
 * @brief 创建种群
 * 
 * @param population_size 种群大小
 * @param genome_size 基因组大小
 * @param auxiliary_size 辅助数据大小
 * @return Population* 种群指针，失败返回NULL
 */
Population* population_create(size_t population_size, size_t genome_size, 
                              size_t auxiliary_size);

/**
 * @brief 销毁种群
 * 
 * @param pop 种群指针
 */
void population_destroy(Population* pop);

/**
 * @brief 设置种群突变率
 * 
 * @param pop 种群指针
 * @param rate 突变率（0.0-1.0）
 */
void population_set_mutation_rate(Population* pop, float rate);

/**
 * @brief 设置种群交叉率
 * 
 * @param pop 种群指针
 * @param rate 交叉率（0.0-1.0）
 */
void population_set_crossover_rate(Population* pop, float rate);

/**
 * @brief 设置种群精英保留率
 * 
 * @param pop 种群指针
 * @param rate 精英保留率（0.0-1.0）
 */
void population_set_elitism_rate(Population* pop, float rate);

/**
 * @brief 初始化种群（随机初始化）
 * 
 * @param pop 种群指针
 * @param min_val 最小值
 * @param max_val 最大值
 * @return int 成功返回0，失败返回-1
 */
int population_initialize_random(Population* pop, float min_val, float max_val);

/**
 * @brief 评估种群适应度
 * 
 * @param pop 种群指针
 * @param fitness_func 适应度函数
 * @param user_data 用户数据
 * @return int 成功返回0，失败返回-1
 */
int population_evaluate(Population* pop, FitnessFunction fitness_func, void* user_data);

/**
 * @brief 计算种群多样性
 * 
 * @param pop 种群指针
 * @return int 成功返回0，失败返回-1
 */
int population_compute_diversity(Population* pop);

/**
 * @brief 选择个体（锦标赛选择）
 * 
 * @param pop 种群指针
 * @param tournament_size 锦标赛大小
 * @return Individual* 选择的个体指针，失败返回NULL
 */
Individual* population_tournament_selection(Population* pop, int tournament_size);

/**
 * @brief 执行一代进化
 * 
 * @param pop 种群指针
 * @param fitness_func 适应度函数
 * @param user_data 用户数据
 * @return int 成功返回0，失败返回-1
 */
int population_evolve(Population* pop, FitnessFunction fitness_func, void* user_data);

/**
 * @brief 获取种群统计信息
 * 
 * @param pop 种群指针
 * @param stats 统计信息输出
 * @return int 成功返回0，失败返回-1
 */
int population_get_statistics(const Population* pop, PopulationStatistics* stats);

/**
 * @brief 获取最佳个体基因组
 * 
 * @param pop 种群指针
 * @param genome_size 输出基因组大小（可选）
 * @return const float* 基因组数组指针，失败返回NULL
 */
const float* population_get_best_genome(const Population* pop, size_t* genome_size);

const float* individual_get_genome(const Individual* ind, size_t* genome_size);

size_t population_get_size(const Population* pop);

const float* population_get_individual_genome(const Population* pop, size_t index, size_t* genome_size);

float population_get_individual_fitness(const Population* pop, size_t index);

// ==================== NAS系统集成接口 ====================

/**
 * @brief 执行神经架构搜索
 * 
 * @param pop 种群指针
 * @param max_generations 最大搜索代数（0表示使用默认值）
 * @return int 成功返回最佳架构索引，失败返回-1
 */
int population_nas_search(Population* pop, int max_generations);

/**
 * @brief 获取NAS系统发现的最佳架构
 * 
 * @param pop 种群指针
 * @param architecture 架构描述输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int population_get_best_architecture(Population* pop, ArchitectureDescription* architecture);

/**
 * @brief 获取NAS搜索状态
 * 
 * @param pop 种群指针
 * @param state 搜索状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int population_nas_get_state(Population* pop, NASSearchState* state);

/**
 * @brief 生成随机架构描述
 * 
 * @param pop 种群指针
 * @param architecture 架构描述输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int population_nas_generate_architecture(Population* pop, ArchitectureDescription* architecture);

/**
 * @brief 评估指定架构
 * 
 * @param pop 种群指针
 * @param architecture 架构描述
 * @param evaluation 评估结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int population_nas_evaluate_architecture(Population* pop,
                                        const ArchitectureDescription* architecture,
                                        ArchitectureEvaluation* evaluation);

// ==================== 多目标演化（NSGA-II） ====================

/**
 * @brief 多目标适应度函数类型
 * 
 * @param genome 基因组数组
 * @param genome_size 基因组大小
 * @param objectives 目标值输出数组
 * @param num_objectives 目标数量
 * @param user_data 用户数据
 */
typedef void (*MultiObjectiveFunction)(const float* genome, size_t genome_size,
                                       float* objectives, int num_objectives, void* user_data);

/**
 * @brief NSGA-II配置
 */
typedef struct {
    int num_objectives;             /**< 目标数量 */
    float crossover_prob;           /**< 交叉概率 */
    float mutation_prob;            /**< 突变概率 */
    float mutation_strength;        /**< 突变强度 */
    float tournament_size_ratio;    /**< 锦标赛比例 (0-1) */
} NSGA2Config;

/**
 * @brief NSGA-II个体数据（多目标信息，与Individual分开存储）
 */
typedef struct {
    float* objectives;              /**< 目标值数组 */
    int rank;                       /**< 非支配排序等级(0=最优前沿) */
    float crowding_distance;        /**< 拥挤距离 */
} NSGA2IndividualData;

/**
 * @brief 获取NSGA-II默认配置
 * 
 * @param num_objectives 目标数量
 * @return NSGA2Config 默认配置
 */
NSGA2Config nsga2_config_default(int num_objectives);

/**
 * @brief 执行NSGA-II一代演化
 * 
 * 非支配排序遗传算法II，同时优化多个目标函数。
 * 
 * @param pop 种群指针
 * @param obj_func 多目标函数
 * @param nsga2_config NSGA-II配置
 * @param user_data 用户数据
 * @return int 成功返回0，失败返回-1
 */
int population_nsga2_evolve(Population* pop, MultiObjectiveFunction obj_func,
                            const NSGA2Config* nsga2_config, void* user_data);

/**
 * @brief 计算非支配排序（NSGA-II核心）
 * 
 * @param pop 种群指针
 * @param obj_data NSGA2个体数据数组（每个个体一个，预分配）
 * @param num_objectives 目标数量
 * @return int 成功返回前沿层级总数，失败返回-1
 */
int population_compute_non_dominated_sort(Population* pop,
                                          NSGA2IndividualData* obj_data,
                                          int num_objectives);

/**
 * @brief 计算拥挤距离
 * 
 * @param pop 种群指针
 * @param obj_data NSGA2个体数据数组
 * @param front_indices 当前前沿的个体索引数组
 * @param front_size 前沿大小
 * @param num_objectives 目标数量
 */
void population_compute_crowding_distance(Population* pop,
                                          NSGA2IndividualData* obj_data,
                                          const int* front_indices, int front_size,
                                          int num_objectives);

/**
 * @brief 提取帕累托前沿解（rank=0的非支配解）
 * 
 * @param pop 种群指针
 * @param obj_data NSGA2个体数据数组
 * @param pareto_indices 输出帕累托前沿个体索引数组（预分配pop->population_size个int）
 * @return int 帕累托前沿解的数量
 */
int population_extract_pareto_front(Population* pop,
                                    NSGA2IndividualData* obj_data,
                                    int* pareto_indices);

// ==================== 训练系统集成（P1-7: 多目标演化优化） ====================

/**
 * @brief 多目标训练评估数据（传递到MultiObjectiveFunction的user_data）
 */
typedef struct {
    LNN* network;                       /**< 要评估的神经网络 */
    const float* train_inputs;          /**< 训练输入数据 */
    const float* train_targets;         /**< 训练目标数据 */
    size_t num_samples;                 /**< 样本数 */
    float accuracy_weight;              /**< 精度目标权重（默认1.0） */
    float speed_weight;                 /**< 速度目标权重（默认0.5） - 推理时间倒数 */
    float energy_weight;                /**< 能耗目标权重（默认0.3） - 参数数量倒数 */
    int max_eval_iterations;            /**< 每次评估最大训练迭代数（默认100） */
    float learning_rate;                /**< 评估时使用的学习率（默认0.01） */
    size_t input_dim;                   /**< 输入维度 */
    size_t output_dim;                  /**< 输出维度 */
} EvolutionTrainingData;

/**
 * @brief 将基因组应用到神经网络权重
 *
 * 将演化算法找到的基因组（参数数组）复制到LNN网络的权重中。
 * 用于演化结果自动应用到网络。
 *
 * @param network 神经网络
 * @param genome 基因组数组
 * @param genome_size 基因组大小
 * @return int 成功返回0，失败返回-1
 */
int evolution_apply_genome_to_network(LNN* network, const float* genome, size_t genome_size);

/**
 * @brief 从神经网络提取权重到基因组
 *
 * 将LNN网络的当前权重提取为一维浮点数组，用于初始化或比较。
 *
 * @param network 神经网络
 * @param genome_size 输出基因组大小
 * @return float* 新分配的基因组数组，调用者负责释放，失败返回NULL
 */
float* evolution_extract_network_weights(const LNN* network, size_t* genome_size);

/**
 * @brief 三目标演化评估函数（精度↑ / 速度↑ / 能耗↓）
 *
 * 同时对精度、推理速度、能耗三个目标进行评估：
 * - 目标0: 精度 = 验证准确率（最大化）
 * - 目标1: 速度 = 1/(推理时间ms) 归一化（最大化）
 * - 目标2: 能耗 = 1/(参数量) 归一化（最大化）
 *
 * @param genome 基因组数组
 * @param genome_size 基因组大小
 * @param objectives 三个目标值输出数组（精度, 速度, 能耗）
 * @param num_objectives 目标数量（必须为3）
 * @param user_data EvolutionTrainingData指针
 */
void evolution_multi_objective_train(const float* genome, size_t genome_size,
                                      float* objectives, int num_objectives,
                                      void* user_data);

/**
 * @brief 演化后自动应用最佳基因组到网络
 *
 * 在种群演化完成后，提取最佳个体基因组自动应用到指定网络。
 *
 * @param pop 演化完成的种群
 * @param network 要应用的目标网络
 * @return int 成功返回0，失败返回-1
 */
int evolution_evolve_and_apply(Population* pop, LNN* network);

// ==================== 协同演化 ====================

/**
 * @brief 协同演化类型
 */
typedef enum {
    COEVOLUTION_COOPERATIVE,        /**< 合作协同演化：各子种群合作解决同一问题 */
    COEVOLUTION_COMPETITIVE         /**< 竞争协同演化：子种群之间相互竞争 */
} CoevolutionType;

/**
 * @brief 协同演化配置
 */
typedef struct {
    CoevolutionType type;           /**< 协同演化类型 */
    int num_subpopulations;         /**< 子种群数量 */
    int* subpopulation_sizes;       /**< 各子种群大小数组 */
    int* genome_sizes;              /**< 各子种群基因组大小数组 */
    float* mutation_rates;          /**< 各子种群突变率数组 */
    float* crossover_rates;         /**< 各子种群交叉率数组 */
    float interaction_rate;         /**< 种群间交互频率 (0-1) */
    int enable_shared_fitness;      /**< 启用共享适应度评估 */
} CoevolutionConfig;

/**
 * @brief 协同演化系统（不透明类型）
 */
typedef struct CoevolutionSystem CoevolutionSystem;

/**
 * @brief 创建协同演化系统
 * 
 * @param config 协同演化配置
 * @return CoevolutionSystem* 协同演化系统指针，失败返回NULL
 */
CoevolutionSystem* coevolution_system_create(const CoevolutionConfig* config);

/**
 * @brief 销毁协同演化系统
 * 
 * @param system 协同演化系统指针
 */
void coevolution_system_destroy(CoevolutionSystem* system);

/**
 * @brief 初始化协同演化系统子种群
 * 
 * @param system 协同演化系统指针
 * @param min_val 基因组初始化最小值
 * @param max_val 基因组初始化最大值
 * @return int 成功返回0，失败返回-1
 */
int coevolution_system_initialize(CoevolutionSystem* system, float min_val, float max_val);

/**
 * @brief 执行协同演化一代进化
 * 
 * @param system 协同演化系统指针
 * @param obj_func 多目标适应度函数（接受subpopulation_index参数区分子种群）
 * @param user_data 用户数据
 * @return int 成功返回0，失败返回-1
 */
int coevolution_system_evolve(CoevolutionSystem* system,
                              MultiObjectiveFunction obj_func, void* user_data);

/**
 * @brief 获取协同演化系统中指定子种群的统计信息
 * 
 * @param system 协同演化系统指针
 * @param subpop_index 子种群索引
 * @param stats 统计信息输出
 * @return int 成功返回0，失败返回-1
 */
int coevolution_system_get_statistics(const CoevolutionSystem* system, int subpop_index,
                                      PopulationStatistics* stats);

/**
 * @brief 获取协同演化系统中指定子种群的最佳个体基因组
 * 
 * @param system 协同演化系统指针
 * @param subpop_index 子种群索引
 * @param genome_size 输出基因组大小
 * @return const float* 基因组指针，失败返回NULL
 */
const float* coevolution_system_get_best_genome(const CoevolutionSystem* system,
                                                 int subpop_index, size_t* genome_size);

// ==================== 开放式演化（新颖性搜索） ====================

/**
 * @brief 新颖性搜索配置
 */
typedef struct {
    float novelty_threshold;        /**< 新颖性阈值（判断行为是否为新颖） */
    int archive_growth_interval;    /**< 归档增长间隔（每N代检查归档） */
    int k_nearest_neighbors;        /**< K近邻数（用于计算新颖性分数） */
    float novelty_vs_fitness;       /**< 新颖性vs适应度权重 (0=纯适应度, 1=纯新颖性) */
    int max_archive_size;           /**< 最大归档大小 */
    int behavior_dimension;         /**< 行为特征维度 */
} OpenEndedConfig;

/**
 * @brief 开放演化状态
 */
typedef struct {
    float** behavior_archive;       /**< 行为特征归档 */
    size_t archive_size;            /**< 当前归档大小 */
    size_t archive_capacity;        /**< 归档容量 */
    float* novelty_scores;          /**< 当前种群新颖性分数 */
    float* behavior_features;       /**< 当前种群行为特征（一维数组，population_size * behavior_dim） */
    float avg_novelty;              /**< 平均新颖性 */
    float max_novelty;              /**< 最大新颖性 */
    int generation_counter;         /**< 代数计数器 */
    int behavior_dim;               /**< 行为特征维度 */
    size_t population_size;         /**< 种群大小 */
} OpenEndedState;

/**
 * @brief 创建开放演化状态
 * 
 * @param config 开放式演化配置
 * @param population_size 种群大小
 * @return OpenEndedState* 状态指针，失败返回NULL
 */
OpenEndedState* open_ended_state_create(const OpenEndedConfig* config,
                                        size_t population_size);

/**
 * @brief 销毁开放演化状态
 * 
 * @param state 状态指针
 */
void open_ended_state_destroy(OpenEndedState* state);

/**
 * @brief 执行新颖性搜索一代演化
 * 
 * @param pop 种群指针
 * @param fitness_func 适应度函数
 * @param behavior_func 行为特征函数
 * @param oe_state 开放演化状态
 * @param config 开放式演化配置
 * @param user_data 用户数据
 * @return int 成功返回0，失败返回-1
 */
int population_novelty_evolve(Population* pop,
                              FitnessFunction fitness_func,
                              void (*behavior_func)(const float* genome, size_t genome_size,
                                                    float* behavior, int behavior_dim, void* user_data),
                              OpenEndedState* oe_state,
                              const OpenEndedConfig* config,
                              void* user_data);

/**
 * @brief 更新新颖性归档
 * 
 * @param oe_state 开放演化状态
 * @param behavior_func 行为特征函数
 * @param pop 种群指针
 * @param config 开放式演化配置
 * @param user_data 用户数据
 * @return int 成功返回新增归档数量，失败返回-1
 */
int open_ended_update_archive(OpenEndedState* oe_state,
                              void (*behavior_func)(const float*, size_t, float*, int, void*),
                              Population* pop,
                              const OpenEndedConfig* config,
                              void* user_data);

/* ============================
 * 能力开关控制（P3.3）
 * ============================ */

/**
 * @brief 启用演化算法种群
 * 
 * @param pop 种群指针
 */
void population_enable(Population* pop);

/**
 * @brief 禁用演化算法种群
 * 
 * @param pop 种群指针
 */
void population_disable(Population* pop);

/**
 * @brief 检查种群是否已启用
 * 
 * @param pop 种群指针
 * @return 1表示启用，0表示禁用
 */
int population_is_enabled(const Population* pop);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_EVOLUTIONARY_ALGORITHMS_H */