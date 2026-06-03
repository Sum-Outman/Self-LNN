/**
 * @file neural_architecture_search.h
 * @brief 神经架构搜索（NAS）系统接口
 * 
 * 神经架构搜索（Neural Architecture Search）系统实现，支持：
 * 1. 基于强化学习的NAS（RL-NAS）
 * 2. 基于进化算法的NAS（EA-NAS）
 * 3. 基于梯度的NAS（DARTS）
 * 4. 一次性NAS（One-Shot NAS）
 * 5. 可微分架构搜索（Differentiable NAS）
 * 6. 多目标NAS（Multi-Objective NAS）
 * 7. 硬件感知NAS（Hardware-Aware NAS）
 * 8. 渐进式NAS（Progressive NAS）
 * 
 *  ，提供完整的神经架构搜索算法。
 */

#ifndef SELFLNN_NEURAL_ARCHITECTURE_SEARCH_H
#define SELFLNN_NEURAL_ARCHITECTURE_SEARCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NAS搜索策略
 */
typedef enum {
    NAS_STRATEGY_RL = 0,            /**< 强化学习策略 */
    NAS_STRATEGY_EVOLUTIONARY = 1,  /**< 进化算法策略 */
    NAS_STRATEGY_GRADIENT = 2,      /**< 梯度策略（DARTS） */
    NAS_STRATEGY_ONE_SHOT = 3,      /**< 一次性策略 */
    NAS_STRATEGY_PROGRESSIVE = 4,   /**< 渐进式策略 */
    NAS_STRATEGY_RANDOM = 5,        /**< 随机搜索策略 */
    NAS_STRATEGY_BAYESIAN = 6,      /**< 贝叶斯优化策略 */
    NAS_STRATEGY_MIXED = 7          /**< 混合策略 */
} NASStrategy;

/**
 * @brief 架构编码类型
 */
typedef enum {
    ARCH_ENCODING_DIRECT = 0,       /**< 直接编码 */
    ARCH_ENCODING_CELL_BASED = 1,   /**< 基于单元的编码 */
    ARCH_ENCODING_GRAPH = 2,        /**< 图编码 */
    ARCH_ENCODING_SEQUENCE = 3,     /**< 序列编码 */
    ARCH_ENCODING_TREE = 4,         /**< 树编码 */
    ARCH_ENCODING_HYPERNET = 5      /**< 超网络编码 */
} ArchitectureEncoding;

/**
 * @brief 搜索空间维度
 */
typedef enum {
    SEARCH_DIM_LAYERS = 0,          /**< 层数 */
    SEARCH_DIM_WIDTH = 1,           /**< 宽度（通道数） */
    SEARCH_DIM_KERNEL = 2,          /**< 卷积核大小 */
    SEARCH_DIM_OPERATIONS = 3,      /**< 操作类型 */
    SEARCH_DIM_CONNECTIONS = 4,     /**< 连接模式 */
    SEARCH_DIM_ACTIVATION = 5,      /**< 激活函数 */
    SEARCH_DIM_NORMALIZATION = 6,   /**< 归一化层 */
    SEARCH_DIM_ATTENTION = 7        /**< 注意力机制 */
} SearchDimension;

/**
 * @brief 架构评估指标
 */
typedef enum {
    METRIC_ACCURACY = 0,            /**< 准确率 */
    METRIC_LOSS = 1,                /**< 损失值 */
    METRIC_PARAMETERS = 2,          /**< 参数量 */
    METRIC_FLOPs = 3,               /**< 浮点运算数 */
    METRIC_LATENCY = 4,             /**< 延迟 */
    METRIC_ENERGY = 5,              /**< 能耗 */
    METRIC_MEMORY = 6,              /**< 内存使用 */
    METRIC_COMPLEXITY = 7,          /**< 复杂度 */
    METRIC_ROBUSTNESS = 8,          /**< 鲁棒性 */
    METRIC_GENERALIZATION = 9       /**< 泛化能力 */
} ArchitectureMetric;

/**
 * @brief NAS配置
 */
typedef struct {
    NASStrategy strategy;           /**< 搜索策略 */
    ArchitectureEncoding encoding;  /**< 架构编码 */
    int search_dimensions[8];       /**< 搜索维度标志 */
    int max_layers;                 /**< 最大层数 */
    int min_layers;                 /**< 最小层数 */
    int max_width;                  /**< 最大宽度 */
    int min_width;                  /**< 最小宽度 */
    float mutation_rate;            /**< 变异率 */
    float crossover_rate;           /**< 交叉率 */
    int population_size;            /**< 种群大小 */
    int max_generations;            /**< 最大代数 */
    float exploration_rate;         /**< 探索率 */
    float learning_rate;            /**< 学习率 */
    int enable_multi_objective;     /**< 是否启用多目标优化 */
    int enable_hardware_aware;      /**< 是否启用硬件感知 */
    int enable_progressive;         /**< 是否启用渐进式搜索 */
    int enable_transfer_learning;   /**< 是否启用迁移学习 */
    int enable_ensemble;            /**< 是否启用集成学习 */
    float target_accuracy;          /**< 目标准确率 */
    float max_complexity;           /**< 最大复杂度 */
    float max_latency;              /**< 最大延迟 */
    int enable_parallel_evaluation; /**< 是否启用并行评估 */
    int max_parallel_evaluations;   /**< 最大并行评估数 */
} NASConfig;

/**
 * @brief 架构描述
 */
typedef struct {
    int* layer_types;               /**< 层类型数组 */
    int* layer_widths;              /**< 层宽度数组 */
    int* kernel_sizes;              /**< 卷积核大小数组 */
    int* operations;                /**< 操作类型数组 */
    int* connections;               /**< 连接矩阵 */
    int* activations;               /**< 激活函数数组 */
    int layer_count;                /**< 层数 */
    int total_parameters;           /**< 总参数量 */
    float estimated_flops;          /**< 估计FLOPs */
    float estimated_latency;        /**< 估计延迟 */
    float estimated_memory;         /**< 估计内存使用 */
    float* genome;                  /**< 基因组表示 */
    size_t genome_size;             /**< 基因组大小 */
    int is_evaluated;               /**< 是否已评估 */
    float fitness_score;            /**< 适应度分数 */
    float* metrics;                 /**< 指标数组 */
    size_t metrics_count;           /**< 指标数量 */
} ArchitectureDescription;

/**
 * @brief 架构评估结果
 */
typedef struct {
    ArchitectureDescription* architecture; /**< 架构描述 */
    float accuracy;                /**< 准确率 */
    float loss;                    /**< 损失值 */
    float training_time;           /**< 训练时间 */
    float inference_time;          /**< 推理时间 */
    float memory_usage;            /**< 内存使用 */
    float energy_consumption;      /**< 能耗 */
    float robustness_score;        /**< 鲁棒性得分 */
    float generalization_score;    /**< 泛化得分 */
    float complexity_score;        /**< 复杂度得分 */
    float overall_score;           /**< 总体得分 */
    int evaluation_status;         /**< 评估状态 */
    char* evaluation_log;          /**< 评估日志 */
} ArchitectureEvaluation;

/**
 * @brief NAS搜索状态
 */
typedef struct {
    int current_generation;        /**< 当前代数 */
    int architectures_evaluated;   /**< 已评估架构数 */
    int architectures_generated;   /**< 已生成架构数 */
    float best_fitness;            /**< 最佳适应度 */
    float average_fitness;         /**< 平均适应度 */
    float fitness_stddev;          /**< 适应度标准差 */
    float diversity_score;         /**< 多样性分数 */
    float exploration_score;       /**< 探索分数 */
    float exploitation_score;      /**< 开发分数 */
    float search_progress;         /**< 搜索进度 */
    ArchitectureDescription* best_architecture; /**< 最佳架构 */
    int is_searching;              /**< 是否正在搜索 */
    int search_complete;           /**< 搜索是否完成 */
} NASSearchState;

/**
 * @brief NAS系统句柄
 */
typedef struct NASSystem NASSystem;

/**
 * @brief 架构评估回调函数
 * 
 * @param architecture 架构描述
 * @param user_data 用户数据
 * @return ArchitectureEvaluation* 评估结果，失败返回NULL
 */
typedef ArchitectureEvaluation* (*ArchitectureEvaluator)(
    const ArchitectureDescription* architecture, void* user_data);

/**
 * @brief 创建NAS系统
 * 
 * @param config NAS配置
 * @param evaluator 架构评估器
 * @param user_data 用户数据
 * @return NASSystem* NAS系统句柄，失败返回NULL
 */
NASSystem* nas_system_create(const NASConfig* config,
                            ArchitectureEvaluator evaluator,
                            void* user_data);

/**
 * @brief 释放NAS系统
 * 
 * @param system NAS系统句柄
 */
void nas_system_free(NASSystem* system);

/**
 * @brief 初始化搜索空间
 * 
 * @param system NAS系统句柄
 * @return int 成功返回0，失败返回-1
 */
int nas_initialize_search_space(NASSystem* system);

/**
 * @brief 执行一代搜索
 * 
 * @param system NAS系统句柄
 * @return int 成功返回评估的架构数量，失败返回-1
 */
int nas_search_generation(NASSystem* system);

/**
 * @brief 执行完整搜索
 * 
 * @param system NAS系统句柄
 * @param max_generations 最大代数（0表示使用配置中的值）
 * @return int 成功返回找到的最佳架构索引，失败返回-1
 */
int nas_search_complete(NASSystem* system, int max_generations,
                        void* arch_ctrl, void** lnn);

/**
 * @brief 生成随机架构
 * 
 * @param system NAS系统句柄
 * @param architecture 架构描述输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int nas_generate_random_architecture(NASSystem* system,
                                    ArchitectureDescription* architecture);

/**
 * @brief 变异架构
 * 
 * @param system NAS系统句柄
 * @param parent 父架构
 * @param child 子架构输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int nas_mutate_architecture(NASSystem* system,
                           const ArchitectureDescription* parent,
                           ArchitectureDescription* child);

/**
 * @brief 交叉架构
 * 
 * @param system NAS系统句柄
 * @param parent1 父架构1
 * @param parent2 父架构2
 * @param child 子架构输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int nas_crossover_architectures(NASSystem* system,
                               const ArchitectureDescription* parent1,
                               const ArchitectureDescription* parent2,
                               ArchitectureDescription* child);

/**
 * @brief 评估架构
 * 
 * @param system NAS系统句柄
 * @param architecture 架构描述
 * @param evaluation 评估结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int nas_evaluate_architecture(NASSystem* system,
                             const ArchitectureDescription* architecture,
                             ArchitectureEvaluation* evaluation);

/**
 * @brief 获取最佳架构
 * 
 * @param system NAS系统句柄
 * @param architecture 架构描述输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int nas_get_best_architecture(NASSystem* system,
                             ArchitectureDescription* architecture);

/**
 * @brief 获取搜索状态
 * 
 * @param system NAS系统句柄
 * @param state 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int nas_get_search_state(NASSystem* system, NASSearchState* state);

/**
 * @brief 重置NAS系统
 * 
 * @param system NAS系统句柄
 * @return int 成功返回0，失败返回-1
 */
int nas_reset(NASSystem* system);

/**
 * @brief 保存搜索状态
 * 
 * @param system NAS系统句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int nas_save_state(NASSystem* system, const char* filepath);

/**
 * @brief 加载搜索状态
 * 
 * @param system NAS系统句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int nas_load_state(NASSystem* system, const char* filepath);

/**
 * @brief 导出最佳架构
 * 
 * @param system NAS系统句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int nas_export_best_architecture(NASSystem* system, const char* filepath);

/**
 * @brief P1-002修复: 将NAS搜索到的最优架构部署到运行中的LNN
 *
 * 打通NAS→LNN断裂点。将搜索到的最优架构通过架构控制器
 * 真正部署到生产LNN实例中，实现知识迁移和原子交换。
 *
 * @param system NAS系统句柄
 * @param arch_ctrl 架构控制器实例
 * @param lnn 当前LNN实例指针的指针（部署成功后被替换为新架构）
 * @param min_improvement 最小性能提升阈值（推荐0.1=10%）
 * @param confidence 部署置信度（0-1）
 * @return SELFLNN_SUCCESS(0)=成功，非0=失败
 */
int nas_deploy_best_architecture(NASSystem* system,
                                  void* arch_ctrl,
                                  void** lnn,
                                  float min_improvement,
                                  float confidence);

/**
 * @brief 获取架构统计信息
 * 
 * @param system NAS系统句柄
 * @param generation 代数（-1表示所有代数）
 * @param statistics 统计信息输出缓冲区
 * @param max_statistics 最大统计信息数
 * @return int 成功返回统计信息数量，失败返回-1
 */
int nas_get_statistics(NASSystem* system, int generation,
                      float* statistics, size_t max_statistics);

// ============================================================================
// CfC液态神经架构搜索（A05.4.2）
// ============================================================================

/**
 * @brief 层类型常量（扩展）
 */
#ifndef LAYER_TYPE_CFC
#define LAYER_TYPE_CFC          6   /**< CfC液态层 */
#define LAYER_TYPE_CFC_CELL     7   /**< CfC细胞单元 */
#endif

/**
 * @brief CfC-NAS配置
 */
typedef struct {
    int num_cfc_layers;                 /**< CfC层数搜索范围（默认1-6） */
    int min_hidden_size;                /**< 最小隐藏层维度（默认8） */
    int max_hidden_size;                /**< 最大隐藏层维度（默认256） */
    float min_time_constant;            /**< 最小时间常数（默认0.001） */
    float max_time_constant;            /**< 最大时间常数（默认1.0） */
    int min_num_layers;                 /**< 最小Liquid层数（默认1） */
    int max_num_layers;                 /**< 最大Liquid层数（默认5） */
    int ode_solver_options[4];          /**< ODE求解器选项 */
    int ode_solver_count;               /**< ODE求解器选项数量 */
    int use_batch_norm_options[2];      /**< 批归一化选项 */
    int enable_skip_connections;        /**< 是否启用跳接搜索 */
    int enable_mixed_precision_search;  /**< 是否启用混合精度搜索 */
    float mutation_strength;            /**< 变异强度（默认0.2） */
    float temperature;                  /**< 搜索温度（默认1.0） */
    int controller_hidden_size;         /**< 控制器隐藏层维度（ENAS用，默认64） */
    float controller_lr;                /**< 控制器学习率（ENAS用，默认0.0005） */
    float entropy_coef;                 /**< 熵正则系数（默认0.01） */
    int num_architecture_samples;       /**< 每次采样架构数（默认1） */
    int max_shared_steps;               /**< 最大参数共享步数（默认100） */
} CfcNASConfig;

/**
 * @brief CfC-ENAS搜索器句柄（不透明类型）
 */
typedef struct CfcENASSearch CfcENASSearch;

/**
 * @brief CfC-DARTS搜索器句柄（不透明类型）
 */
typedef struct CfcDARTSearch CfcDARTSearch;

/**
 * @brief 液态CfC进化搜索器句柄（不透明类型）
 */
typedef struct LiquidCfcSearch LiquidCfcSearch;

// ==================== 通用CfC-NAS ====================

/**
 * @brief 获取默认CfC-NAS配置
 */
void cfc_nas_default_config(CfcNASConfig* config);

/**
 * @brief 将CfC层类型注册到现有NAS系统的搜索空间
 * 
 * @param system NAS系统句柄
 * @param cfc_config CfC-NAS配置
 * @return int 成功返回0，失败返回-1
 */
int nas_register_cfc_layers(NASSystem* system, const CfcNASConfig* cfc_config);

/**
 * @brief 生成CfC架构描述（在已有NAS架构中插入CfC层）
 * 
 * @param system NAS系统句柄
 * @param cfc_config CfC-NAS配置
 * @param architecture 架构描述输出
 * @return int 成功返回0，失败返回-1
 */
int nas_generate_cfc_architecture(NASSystem* system,
                                 const CfcNASConfig* cfc_config,
                                 ArchitectureDescription* architecture);

// ==================== CfC-ENAS ====================

/**
 * @brief 创建CfC-ENAS搜索器
 * 
 * CfC-ENAS：使用CfC网络作为控制器生成子架构，
 * 子架构之间共享权重。控制器通过策略梯度训练。
 * 
 * @param cfc_config CfC-NAS配置
 * @return CfcENASSearch* 成功返回搜索器句柄，失败返回NULL
 */
CfcENASSearch* cfc_enas_create(const CfcNASConfig* cfc_config);

/**
 * @brief 销毁CfC-ENAS搜索器
 */
void cfc_enas_destroy(CfcENASSearch* searcher);

/**
 * @brief CfC-ENAS搜索一步
 * 
 * @param searcher CfC-ENAS搜索器
 * @param architecture 输出的子架构
 * @param reward 当前步的奖励（用于更新控制器）
 * @return int 成功返回0，失败返回-1
 */
int cfc_enas_step(CfcENASSearch* searcher,
                  ArchitectureDescription* architecture,
                  float reward);

/**
 * @brief 采样一组子架构（用于批量训练）
 * 
 * @param searcher CfC-ENAS搜索器
 * @param architectures 架构数组输出
 * @param num_samples 采样数量
 * @return int 成功返回采样数，失败返回-1
 */
int cfc_enas_sample_architectures(CfcENASSearch* searcher,
                                 ArchitectureDescription* architectures,
                                 int num_samples);

/**
 * @brief 更新CfC-ENAS控制器参数
 * 
 * @param searcher CfC-ENAS搜索器
 * @param rewards 奖励数组
 * @param num_samples 奖励数
 * @return int 成功返回0，失败返回-1
 */
int cfc_enas_update_controller(CfcENASSearch* searcher,
                              const float* rewards,
                              int num_samples);

/**
 * @brief 共享子架构之间的权重（参数共享）
 * 
 * @param searcher CfC-ENAS搜索器
 * @param architectures 架构数组
 * @param num_archs 架构数
 * @return int 成功返回0，失败返回-1
 */
int cfc_enas_share_weights(CfcENASSearch* searcher,
                          const ArchitectureDescription* architectures,
                          int num_archs);

/**
 * @brief 获取CfC-ENAS搜索到的最佳架构
 * 
 * @param searcher CfC-ENAS搜索器
 * @param architecture 最佳架构输出
 * @return int 成功返回0，失败返回-1
 */
int cfc_enas_get_best_architecture(CfcENASSearch* searcher,
                                  ArchitectureDescription* architecture);

// ==================== CfC-DARTS ====================

/**
 * @brief 创建CfC-DARTS搜索器
 * 
 * CfC-DARTS：将CfC细胞结构和操作选择松弛为连续参数，
 * 通过双层优化（架构参数+网络权重）进行可微搜索。
 * 
 * @param cfc_config CfC-NAS配置
 * @return CfcDARTSearch* 成功返回搜索器句柄，失败返回NULL
 */
CfcDARTSearch* cfc_darts_create(const CfcNASConfig* cfc_config);

/**
 * @brief 销毁CfC-DARTS搜索器
 */
void cfc_darts_destroy(CfcDARTSearch* searcher);

/**
 * @brief CfC-DARTS前向传播
 * 
 * @param searcher CfC-DARTS搜索器
 * @param input 输入数据
 * @param input_size 输入维度
 * @param output 输出数据缓冲区
 * @param output_size 输出维度
 * @return int 成功返回0，失败返回-1
 */
int cfc_darts_forward(CfcDARTSearch* searcher,
                     const float* input, size_t input_size,
                     float* output, size_t output_size);

/**
 * @brief CfC-DARTS反向传播（更新架构参数+网络权重）
 * 
 * @param searcher CfC-DARTS搜索器
 * @param gradient 损失梯度
 * @param gradient_size 梯度维度
 * @return int 成功返回0，失败返回-1
 */
int cfc_darts_backward(CfcDARTSearch* searcher,
                      const float* gradient, size_t gradient_size);

/**
 * @brief 执行一步CfC-DARTS搜索（前向+反向）
 * 
 * @param searcher CfC-DARTS搜索器
 * @param input 输入数据
 * @param input_size 输入维度
 * @param target 目标数据
 * @param target_size 目标维度
 * @param loss 输出损失值
 * @return int 成功返回0，失败返回-1
 */
int cfc_darts_step(CfcDARTSearch* searcher,
                  const float* input, size_t input_size,
                  const float* target, size_t target_size,
                  float* loss);

/**
 * @brief 从松弛架构参数解码离散架构
 * 
 * @param searcher CfC-DARTS搜索器
 * @param architecture 离散架构输出
 * @return int 成功返回0，失败返回-1
 */
int cfc_darts_discretize(CfcDARTSearch* searcher,
                        ArchitectureDescription* architecture);

/**
 * @brief 获取CfC-DARTS架构参数
 * 
 * @param searcher CfC-DARTS搜索器
 * @param alpha 架构参数输出缓冲区
 * @param alpha_size 输出缓冲区大小
 * @return int 成功返回参数数量，失败返回-1
 */
int cfc_darts_get_alphas(CfcDARTSearch* searcher,
                        float* alpha, size_t alpha_size);

// ==================== 液态CfC进化搜索 ====================

/**
 * @brief 液态CfC进化搜索配置
 */
typedef struct {
    int population_size;                /**< 种群大小（默认50） */
    int max_generations;                /**< 最大代数（默认100） */
    float mutation_rate;                /**< 变异率（默认0.3） */
    float crossover_rate;               /**< 交叉率（默认0.5） */
    float elite_ratio;                  /**< 精英保留比例（默认0.2） */
    int tournament_size;                /**< 锦标赛选择大小（默认3） */
    int enable_novelty_search;          /**< 是否启用新颖性搜索 */
    float novelty_weight;               /**< 新颖性权重（默认0.3） */
    int enable_multi_objective;         /**< 是否启用多目标优化 */
    float complexity_weight;            /**< 复杂度权重（默认0.1） */
    float latency_weight;               /**< 延迟权重（默认0.1） */
    int memory_size;                    /**< 归档记忆大小（默认100） */
    CfcNASConfig cfc_config;            /**< CfC子配置 */
} LiquidCfcSearchConfig;

/**
 * @brief 获取液态CfC进化搜索默认配置
 */
void liquid_cfc_search_default_config(LiquidCfcSearchConfig* config);

/**
 * @brief 创建液态CfC进化搜索器
 * 
 * 液态CfC进化搜索：使用进化算法搜索CfC网络的
 * 超参数和拓扑结构（隐藏层维度、时间常数、ODE求解器类型、
 * 层数、跳接模式等），支持多目标和新颖性搜索。
 * 
 * @param config 液态CfC进化搜索配置
 * @return LiquidCfcSearch* 成功返回搜索器句柄，失败返回NULL
 */
LiquidCfcSearch* liquid_cfc_search_create(const LiquidCfcSearchConfig* config);

/**
 * @brief 销毁液态CfC进化搜索器
 */
void liquid_cfc_search_destroy(LiquidCfcSearch* search);

/**
 * @brief 执行一代液态CfC进化搜索
 * 
 * @param search 液态CfC进化搜索器
 * @param evaluator 评估回调函数
 * @param user_data 用户数据
 * @return int 成功返回评估的个体数，失败返回-1
 */
int liquid_cfc_search_generation(LiquidCfcSearch* search,
                                ArchitectureEvaluator evaluator,
                                void* user_data);

/**
 * @brief 执行完整液态CfC进化搜索
 * 
 * @param search 液态CfC进化搜索器
 * @param evaluator 评估回调函数
 * @param user_data 用户数据
 * @param max_generations 最大代数（0使用配置值）
 * @return int 成功返回0，失败返回-1
 */
int liquid_cfc_search_complete(LiquidCfcSearch* search,
                              ArchitectureEvaluator evaluator,
                              void* user_data,
                              int max_generations);

/**
 * @brief 获取液态CfC进化搜索最佳个体
 * 
 * @param search 液态CfC进化搜索器
 * @param architecture 最佳架构输出
 * @return int 成功返回0，失败返回-1
 */
int liquid_cfc_get_best_architecture(LiquidCfcSearch* search,
                                    ArchitectureDescription* architecture);

/**
 * @brief 获取液态CfC进化搜索当前种群
 * 
 * @param search 液态CfC进化搜索器
 * @param architectures 种群架构输出缓冲区
 * @param max_count 最大输出数量
 * @return int 成功返回实际种群数量，失败返回-1
 */
int liquid_cfc_get_population(LiquidCfcSearch* search,
                            ArchitectureDescription* architectures,
                            int max_count);

/**
 * @brief 获取液态CfC进化搜索状态
 * 
 * @param search 液态CfC进化搜索器
 * @param current_generation 当前代数输出
 * @param best_fitness 最佳适应度输出
 * @param average_fitness 平均适应度输出
 * @return int 成功返回0，失败返回-1
 */
int liquid_cfc_get_search_state(LiquidCfcSearch* search,
                               int* current_generation,
                               float* best_fitness,
                               float* average_fitness);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_NEURAL_ARCHITECTURE_SEARCH_H */