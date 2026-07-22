/**
 * @file product_design.h
 * @brief 产品设计能力接口 — 【基础功能实现】
 *
 * P3-004 功能边界说明:
 *   ✅ 本文件: 核心产品设计能力（需求解析、规格生成、设计评估、优化）
 *   ✅ 本文件: 基础产品类型管理（硬件/软件/系统/自定义）
 *   ✅ 本文件: 产品规格创建/评估/比较/优化生命周期管理
 *   ❌ 非本文件: 需求分析引擎(需求追踪矩阵)、多目标优化(NSGA-II)、系统工程(V模型)、
 *                设计验证自动化、需求变更管理、版本控制
 *                → 请使用 product_design_enhanced.h（高级增强功能，与本基础版互补，非替代）
 *
 *   基础版 (product_design.h)         → 核心产品设计 + 基础规格管理
 *   增强版 (product_design_enhanced.h) → 需求分析引擎 + NSGA-II多目标优化 + 系统工程V模型
 */

#ifndef SELFLNN_PRODUCT_DESIGN_H
#define SELFLNN_PRODUCT_DESIGN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 产品类型枚举
 */
typedef enum {
    PRODUCT_TYPE_HARDWARE = 0,   /**< 硬件产品 */
    PRODUCT_TYPE_SOFTWARE = 1,   /**< 软件产品 */
    PRODUCT_TYPE_SYSTEM = 2,     /**< 系统产品 */
    PRODUCT_TYPE_CUSTOM = 3,     /**< 自定义产品 */
    PRODUCT_TYPE_SERVICE = 4     /**< 服务产品（M-010: LNN生成式扩展） */
} ProductType;

/**
 * @brief 产品规格结构体
 */
typedef struct {
    char* name;                  /**< 产品名称 */
    ProductType type;            /**< 产品类型 */
    char* description;           /**< 产品描述 */
    char** features;             /**< 功能特性数组 */
    size_t feature_count;        /**< 功能特性数量 */
    double estimated_cost;       /**< 预估成本 */
    double development_time;     /**< 开发时间（月） */
    double complexity_score;     /**< 复杂度评分 */
    double feasibility_score;    /**< 可行性评分 */
    /* M-010: LNN生成式扩展字段 */
    double size_factor;          /**< 尺寸系数 (0.5-5.0) */
    double weight_factor;        /**< 重量系数 (0.1-10.0) */
    double cost_factor;          /**< 成本系数 (0.5-5.0) */
    char* material_name;         /**< 材料名称 */
} ProductSpec;

/**
 * @brief 产品需求结构体
 */
typedef struct {
    char* name;                  /**< 需求名称 */
    char* requirement_text;      /**< 需求文本 */
    char** keywords;             /**< 关键词数组 */
    size_t keyword_count;        /**< 关键词数量 */
    ProductType preferred_type;  /**< 偏好产品类型 */
    double max_cost;             /**< 最大成本限制 */
    double max_time;             /**< 最长时间限制 */
} ProductRequirement;

/**
 * @brief 设计评估结果
 */
typedef struct {
    double feasibility;          /**< 可行性评分（0-1） */
    double cost_effectiveness;   /**< 成本效益评分（0-1） */
    double innovation_level;     /**< 创新程度评分（0-1） */
    double market_potential;     /**< 市场潜力评分（0-1） */
    char** strengths;            /**< 优势列表 */
    size_t strength_count;       /**< 优势数量 */
    char** weaknesses;           /**< 劣势列表 */
    size_t weakness_count;       /**< 劣势数量 */
    char** recommendations;      /**< 改进建议列表 */
    size_t recommendation_count; /**< 建议数量 */
} DesignEvaluation;

/* ============================================================================
 * F-10: 参数化生成设计数据结构
 * =========================================================================== */

/**
 * @brief 设计参数类型
 */
typedef enum {
    PARAM_TYPE_FLOAT = 0,       /**< 浮点参数 */
    PARAM_TYPE_INT = 1,         /**< 整数参数 */
    PARAM_TYPE_BOOL = 2         /**< 布尔参数 */
} DesignParamType;

/**
 * @brief 设计参数定义
 */
typedef struct {
    char* name;                  /**< 参数名称 */
    DesignParamType param_type;  /**< 参数类型 */
    double min_value;            /**< 最小值 */
    double max_value;            /**< 最大值 */
    double step;                 /**< 步长 */
    double current_value;        /**< 当前值 */
} DesignParameter;

/**
 * @brief 设计变体
 */
typedef struct {
    DesignParameter* parameters; /**< 参数数组 */
    size_t param_count;          /**< 参数数量 */
    ProductSpec* spec;           /**< 对应的产品规格 */
    DesignEvaluation* evaluation; /**< 评估结果 */
    double composite_score;      /**< 综合评分 */
} DesignVariant;

/**
 * @brief 约束类型
 */
typedef enum {
    CONSTRAINT_LINEAR_LE = 0,   /**< 线性 <= 约束 */
    CONSTRAINT_LINEAR_GE = 1,   /**< 线性 >= 约束 */
    CONSTRAINT_LINEAR_EQ = 2,   /**< 线性 == 约束 */
    CONSTRAINT_BOUND = 3        /**< 边界约束 */
} ConstraintType;

/**
 * @brief 约束规格
 */
typedef struct {
    ConstraintType type;         /**< 约束类型 */
    double* coefficients;        /**< 线性约束系数数组 */
    size_t coefficient_count;    /**< 系数数量 */
    double rhs;                  /**< 约束右侧值 */
    char* description;           /**< 约束描述 */
} ConstraintSpec;

/* ============================================================================
 * F-10: 拓扑优化数据结构
 * =========================================================================== */

/**
 * @brief 拓扑优化状态
 */
typedef struct {
    double* densities;           /**< 单元密度数组 (nelx * nely) */
    double* densities_new;       /**< 更新后密度数组 */
    int nelx;                    /**< X方向单元数 */
    int nely;                    /**< Y方向单元数 */
    double volfrac;              /**< 体分比目标 */
    double penal;                /**< SIMP惩罚因子 */
    double rmin;                 /**< 过滤半径 */
    double* compliance;          /**< 各单元柔顺度 */
    double total_compliance;     /**< 总柔顺度 */
    double* sensitivity;         /**< 灵敏度数组 */
    double* sensitivity_filter;  /**< 过滤后灵敏度 */
    int iteration;               /**< 当前迭代次数 */
    int max_iterations;          /**< 最大迭代次数 */
    double change_threshold;     /**< 收敛阈值 */
    double* ke;                  /**< 单元刚度矩阵 (8x8) */
    double* f;                   /**< 力向量 */
    int* fixed_dofs;             /**< 固定自由度索引 */
    int fixed_count;             /**< 固定自由度数量 */
} TopologyOptimizationState;

/**
 * @brief 产品设计引擎句柄
 */
typedef struct ProductDesignEngine ProductDesignEngine;

/**
 * @brief 创建产品设计引擎
 * 
 * @return ProductDesignEngine* 引擎句柄，失败返回NULL
 */
ProductDesignEngine* product_design_engine_create(void);

/**
 * @brief 销毁产品设计引擎
 * 
 * @param engine 引擎句柄
 */
void product_design_engine_destroy(ProductDesignEngine* engine);

/**
 * @brief 解析产品需求
 * 
 * @param engine 引擎句柄
 * @param requirement_text 需求文本
 * @return ProductRequirement* 解析后的需求，失败返回NULL
 */
ProductRequirement* parse_product_requirement(ProductDesignEngine* engine,
                                              const char* requirement_text);

/**
 * @brief 销毁产品需求
 * 
 * @param requirement 需求句柄
 */
void product_requirement_destroy(ProductRequirement* requirement);

/**
 * @brief 生成产品规格
 * 
 * @param engine 引擎句柄
 * @param requirement 产品需求
 * @return ProductSpec* 产品规格，失败返回NULL
 */
ProductSpec* generate_product_spec(ProductDesignEngine* engine,
                                   const ProductRequirement* requirement);

/**
 * @brief 销毁产品规格
 * 
 * @param spec 产品规格
 */
void product_spec_destroy(ProductSpec* spec);

/**
 * @brief 释放产品规格（别名，同product_spec_destroy）
 * 
 * M-010: 供LNN生成式内部使用
 * 
 * @param spec 产品规格
 */
void free_product_spec(ProductSpec* spec);

/* ============================================================================
 * M-010: LNN生成式产品设计API
 * ============================================================================ */

/**
 * @brief 使用LNN液态神经网络生成产品规格
 * 
 * 流程：需求文本→bigram哈希特征向量(128维)→LNN前向传播→输出解码→产品参数
 * 贝叶斯融合：规则匹配作为先验(权重0.4)，LNN输出作为后验(权重0.6)
 * LNN不可用时返回NULL，调用方应回退到generate_product_spec规则匹配。
 * 
 * @param engine 产品设计引擎句柄
 * @param requirement 产品需求
 * @return ProductSpec* 产品规格，失败或LNN不可用时返回NULL
 */
ProductSpec* generate_product_spec_lnn(ProductDesignEngine* engine,
                                        const ProductRequirement* requirement);

/**
 * @brief 设置产品设计引擎的LNN实例
 * 
 * 允许外部注入LNN实例用于产品规格生成。
 * 传入NULL则使用selflnn_get_lnn()获取全局实例。
 * 
 * @param engine 产品设计引擎句柄
 * @param lnn LNN实例（可为NULL使用全局实例）
 * @return int 成功返回0，失败返回-1
 */
int product_design_set_lnn(ProductDesignEngine* engine, void* lnn);

/**
 * @brief 评估产品设计
 * 
 * @param engine 引擎句柄
 * @param spec 产品规格
 * @return DesignEvaluation* 评估结果，失败返回NULL
 */
DesignEvaluation* evaluate_product_design(ProductDesignEngine* engine,
                                          const ProductSpec* spec);

/**
 * @brief 销毁设计评估
 * 
 * @param evaluation 评估结果
 */
void design_evaluation_destroy(DesignEvaluation* evaluation);

/**
 * @brief 优化产品设计
 * 
 * @param engine 引擎句柄
 * @param spec 产品规格（将被修改）
 * @param evaluation 设计评估
 * @return int 成功返回0，失败返回错误码
 */
int optimize_product_design(ProductDesignEngine* engine,
                            ProductSpec* spec,
                            const DesignEvaluation* evaluation);

/**
 * @brief 自我改进：基于反馈迭代优化设计
 * 
 * @param engine 引擎句柄
 * @param requirement 产品需求
 * @param iterations 迭代次数
 * @return ProductSpec* 优化后的产品规格，失败返回NULL
 */
ProductSpec* self_improve_design(ProductDesignEngine* engine,
                                 const ProductRequirement* requirement,
                                 int iterations);

/* ============================================================================
 * F-10: 参数化生成设计API
 * =========================================================================== */

/**
 * @brief 创建设计参数
 * 
 * @param name 参数名称
 * @param param_type 参数类型
 * @param min_value 最小值
 * @param max_value 最大值
 * @param step 步长
 * @return DesignParameter 创建的参数
 */
DesignParameter create_design_parameter(const char* name, DesignParamType param_type,
                                        double min_value, double max_value, double step);

/**
 * @brief 基于参数网格生成设计变体
 * 
 * 对每个参数在其范围内按步长生成离散值，取笛卡尔积生成所有变体。
 * 使用并行枚举策略，对每个变体调用generate_product_spec生成规格。
 * 
 * @param engine 引擎句柄
 * @param requirement 产品需求
 * @param parameters 参数数组
 * @param param_count 参数数量
 * @param variant_count 输出：生成的变体数量
 * @return DesignVariant* 设计变体数组，失败返回NULL
 */
DesignVariant* generate_design_variants(ProductDesignEngine* engine,
                                        const ProductRequirement* requirement,
                                        const DesignParameter* parameters,
                                        size_t param_count,
                                        size_t* variant_count);

/**
 * @brief 检查设计参数是否满足所有约束
 * 
 * @param parameters 设计参数数组
 * @param param_count 参数数量
 * @param constraints 约束数组
 * @param constraint_count 约束数量
 * @param violated_idx 输出：第一个违反的约束索引（未违反设为constraint_count）
 * @return int 全部满足返回1，否则返回0
 */
int constraint_satisfaction_check(const DesignParameter* parameters,
                                  size_t param_count,
                                  const ConstraintSpec* constraints,
                                  size_t constraint_count,
                                  size_t* violated_idx);

/**
 * @brief 对设计变体按综合评分排序（从优到劣）
 * 
 * 综合评分 = w1 * feasibility + w2 * (1 - cost_ratio) + w3 * innovation
 * 
 * @param variants 设计变体数组（将被排序）
 * @param variant_count 变体数量
 */
void rank_design_variants(DesignVariant* variants, size_t variant_count);

/**
 * @brief 销毁设计变体及其内部资源
 * 
 * @param variant 设计变体指针
 */
void design_variant_destroy(DesignVariant* variant);

/* ============================================================================
 *: 参考设计案例管理API
 * ============================================================================ */

/**
 * @brief 向产品设计引擎注入参考设计案例
 *
 * 将跨领域工业设计先验案例注入引擎内部案例库，
 * 供设计优化、类比推理和设计空间探索时作为参考基线。
 * 应在引擎初始化后、产品设计任务开始前调用。
 *
 * @param engine 引擎句柄
 * @param case_name 案例名称
 * @param category_name 类别名称（如"消费电子"、"机械设计"等）
 * @param feature_tags 关键特征标签（逗号分隔字符串）
 * @param constraint_summary 设计约束摘要
 * @param category 类别枚举值
 * @return int 成功返回0，失败返回-1
 */
int product_design_engine_add_reference_case(struct ProductDesignEngine* engine,
    const char* case_name, const char* category_name,
    const char* feature_tags, const char* constraint_summary, int category);

/**
 * @brief 获取引擎中已加载的参考案例数量
 *
 * @param engine 引擎句柄
 * @return int 参考案例数量，engine为NULL返回0
 */
int product_design_engine_get_reference_case_count(struct ProductDesignEngine* engine);

/* ============================================================================
 * F-10: 拓扑优化API
 * ============================================================================ */

/**
 * @brief 创建拓扑优化状态
 * 
 * 初始化SIMP拓扑优化所需的全部数据结构。
 * 使用矩形设计域，左边界固定，右下角施加单位集中力。
 * 
 * @param nelx X方向单元数
 * @param nely Y方向单元数
 * @param volfrac 目标体分比 (0-1)
 * @param penal SIMP惩罚因子（通常3.0）
 * @param rmin 灵敏度过滤半径
 * @return TopologyOptimizationState* 优化状态，失败返回NULL
 */
TopologyOptimizationState* topology_optimization_create(int nelx, int nely,
                                                         double volfrac,
                                                         double penal,
                                                         double rmin);

/**
 * @brief 销毁拓扑优化状态
 * 
 * @param state 优化状态
 */
void topology_optimization_destroy(TopologyOptimizationState* state);

/**
 * @brief 执行单次SIMP迭代
 * 
 * 包含：1) 有限元组装与求解 2) 柔顺度计算 3) 灵敏度分析
 * 4) 灵敏度过滤 5) OC法密度更新
 * 
 * @param state 优化状态
 * @return double 本次迭代后的最大密度变化量，收敛依据
 */
double topology_optimization_simp_iteration(TopologyOptimizationState* state);

/**
 * @brief 运行完整拓扑优化流程
 * 
 * 循环调用simp_iteration直到收敛或达到最大迭代次数。
 * 
 * @param state 优化状态
 * @return int 成功返回0，失败返回错误码
 */
int topology_optimization_run(TopologyOptimizationState* state);

/**
 * @brief 获取当前拓扑优化的密度分布
 * 
 * @param state 优化状态
 * @param density_count 输出：密度数组长度 (nelx*nely)
 * @return const double* 密度数组指针，state生命周期内有效
 */
const double* topology_optimization_get_densities(TopologyOptimizationState* state,
                                                   int* density_count);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_PRODUCT_DESIGN_H */