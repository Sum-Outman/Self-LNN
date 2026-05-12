/**
 * @file math_physics_reasoning.h
 * @brief 数学物理推理引擎接口
 * 
 * 数学物理推理引擎核心接口，支持数学表达式解析、方程求解、物理模拟等。
 */

#ifndef SELFLNN_MATH_PHYSICS_REASONING_H
#define SELFLNN_MATH_PHYSICS_REASONING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 数学表达式类型枚举
 */
typedef enum {
    EXPR_NUMBER = 0,           /**< 数字常量 */
    EXPR_VARIABLE = 1,         /**< 变量 */
    EXPR_ADDITION = 2,         /**< 加法 */
    EXPR_SUBTRACTION = 3,      /**< 减法 */
    EXPR_MULTIPLICATION = 4,   /**< 乘法 */
    EXPR_DIVISION = 5,         /**< 除法 */
    EXPR_POWER = 6,            /**< 幂运算 */
    EXPR_FUNCTION = 7,         /**< 函数调用 */
    EXPR_NEGATION = 8,         /**< 取负 */
    EXPR_PARENTHESES = 9       /**< 括号表达式 */
} MathExpressionType;

/**
 * @brief 数学函数类型枚举
 */
typedef enum {
    FUNC_SIN = 0,              /**< 正弦函数 */
    FUNC_COS = 1,              /**< 余弦函数 */
    FUNC_TAN = 2,              /**< 正切函数 */
    FUNC_EXP = 3,              /**< 指数函数 */
    FUNC_LOG = 4,              /**< 自然对数 */
    FUNC_LOG10 = 5,            /**< 常用对数 */
    FUNC_SQRT = 6,             /**< 平方根 */
    FUNC_ABS = 7               /**< 绝对值 */
} MathFunctionType;

/**
 * @brief 物理量类型枚举
 */
typedef enum {
    PHYS_LENGTH = 0,           /**< 长度 */
    PHYS_MASS = 1,             /**< 质量 */
    PHYS_TIME = 2,             /**< 时间 */
    PHYS_VELOCITY = 3,         /**< 速度 */
    PHYS_ACCELERATION = 4,     /**< 加速度 */
    PHYS_FORCE = 5,            /**< 力 */
    PHYS_ENERGY = 6,           /**< 能量 */
    PHYS_POWER = 7             /**< 功率 */
} PhysicalQuantityType;

/**
 * @brief 数学表达式节点结构体
 */
typedef struct MathExpressionNode MathExpressionNode;

/**
 * @brief 数学表达式节点
 */
struct MathExpressionNode {
    MathExpressionType type;   /**< 表达式类型 */
    MathExpressionNode* left;  /**< 左子节点 */
    MathExpressionNode* right; /**< 右子节点 */
    union {
        double number_value;   /**< 数字值 */
        char* variable_name;   /**< 变量名 */
        MathFunctionType func_type; /**< 函数类型 */
    } value;                   /**< 节点值 */
};

/**
 * @brief 数学表达式求值上下文
 */
typedef struct {
    const char** variable_names;   /**< 变量名数组 */
    double* variable_values;       /**< 变量值数组 */
    size_t variable_count;         /**< 变量数量 */
} MathEvalContext;

/**
 * @brief 方程求解结果
 */
typedef struct {
    double* solutions;             /**< 解数组 */
    size_t solution_count;         /**< 解数量 */
    int has_solution;              /**< 是否有解 */
    int is_exact;                  /**< 是否精确解 */
} EquationSolution;

/**
 * @brief 物理模拟配置
 */
typedef struct {
    double time_step;              /**< 时间步长 */
    double simulation_time;        /**< 模拟总时间 */
    int max_iterations;            /**< 最大迭代次数 */
    double tolerance;              /**< 容差 */
} PhysicsSimConfig;

/**
 * @brief 数学物理推理引擎句柄
 */
typedef struct MathPhysicsReasoningEngine MathPhysicsReasoningEngine;

/**
 * @brief 创建数学物理推理引擎
 * 
 * @return MathPhysicsReasoningEngine* 引擎句柄，失败返回NULL
 */
MathPhysicsReasoningEngine* math_physics_reasoning_engine_create(void);

/**
 * @brief 销毁数学物理推理引擎
 * 
 * @param engine 引擎句柄
 */
void math_physics_reasoning_engine_destroy(MathPhysicsReasoningEngine* engine);

/**
 * @brief 解析数学表达式
 * 
 * @param expression 表达式字符串
 * @return MathExpressionNode* 表达式语法树根节点，失败返回NULL
 */
MathExpressionNode* math_expression_parse(const char* expression);

/**
 * @brief 销毁数学表达式语法树
 * 
 * @param node 表达式语法树根节点
 */
void math_expression_destroy(MathExpressionNode* node);

/**
 * @brief 求值数学表达式
 * 
 * @param node 表达式语法树根节点
 * @param context 求值上下文
 * @return double 求值结果
 */
double math_expression_evaluate(const MathExpressionNode* node, const MathEvalContext* context);

/**
 * @brief 求解一元方程
 * 
 * @param equation 方程表达式字符串
 * @param variable 变量名
 * @param initial_guess 初始猜测值
 * @return EquationSolution* 方程解，失败返回NULL
 */
EquationSolution* solve_equation(const char* equation, const char* variable, double initial_guess);

/**
 * @brief 销毁方程解
 * 
 * @param solution 方程解
 */
void equation_solution_destroy(EquationSolution* solution);

/**
 * @brief 执行物理模拟
 * 
 * @param engine 推理引擎句柄
 * @param config 模拟配置
 * @param initial_conditions 初始条件表达式数组
 * @param condition_count 初始条件数量
 * @return double* 模拟结果时间序列，失败返回NULL
 */
double* physics_simulation_run(MathPhysicsReasoningEngine* engine,
                               const PhysicsSimConfig* config,
                               const char** initial_conditions,
                               size_t condition_count);

/* ========== P2-3: 符号数学引擎增强 ========== */

/**
 * @brief 符号微分：对表达式树进行符号微分
 *
 * @param node 原表达式树根节点
 * @param variable_name 微分变量名
 * @return MathExpressionNode* 微分后的表达式树，失败返回NULL
 */
MathExpressionNode* symbolic_differentiate(const MathExpressionNode* node,
                                           const char* variable_name);

/**
 * @brief 符号化简：代数化简表达式树
 *
 * 化简规则包括：
 * - 0 + x → x, x + 0 → x
 * - 0 * x → 0, x * 0 → 0, 1 * x → x, x * 1 → x
 * - x - 0 → x, x - x → 0
 * - x / 1 → x, 0 / x → 0
 * - x ^ 0 → 1, x ^ 1 → x
 * - 常量折叠：计算所有常量子表达式的值
 * - 负号化简：--x → x
 *
 * @param node 原表达式树根节点
 * @return MathExpressionNode* 化简后的表达式树，失败返回NULL
 */
MathExpressionNode* symbolic_simplify(const MathExpressionNode* node);

/**
 * @brief 泰勒展开：在指定点展开到n阶
 *
 * f(x) = Σ_{k=0}^{n} f^{(k)}(a) / k! * (x-a)^k
 *
 * @param node 原表达式树根节点
 * @param variable_name 展开变量名
 * @param a 展开点
 * @param order 展开阶数
 * @return MathExpressionNode* 展开后的表达式树，失败返回NULL
 */
MathExpressionNode* taylor_expand(const MathExpressionNode* node,
                                  const char* variable_name,
                                  double a, size_t order);

/**
 * @brief 复合Simpson数值积分
 *
 * ∫_a^b f(x) dx ≈ h/3 * [f(a) + 4*Σf(x_{2k-1}) + 2*Σf(x_{2k}) + f(b)]
 *
 * @param node 表达式树根节点
 * @param variable_name 积分变量名
 * @param a 积分下限
 * @param b 积分上限
 * @param n 子区间数（必须为偶数）
 * @return double 积分结果
 */
double definite_integral(const MathExpressionNode* node,
                         const char* variable_name,
                         double a, double b, size_t n);

/**
 * @brief 偏微分：多元函数对指定变量的偏导数
 *
 * @param node 表达式树根节点
 * @param variable_names 所有变量名数组
 * @param variable_values 所有变量值数组
 * @param num_variables 变量数量
 * @param diff_var_name 求导变量名
 * @return double 偏导数值（通过数值微分）
 */
double partial_derivative(const MathExpressionNode* node,
                          const char** variable_names,
                          const double* variable_values,
                          size_t num_variables,
                          const char* diff_var_name);

/**
 * @brief 梯度计算：多元函数对所有变量的梯度向量
 *
 * @param node 表达式树根节点
 * @param variable_names 所有变量名数组
 * @param variable_values 所有变量值数组
 * @param num_variables 变量数量
 * @param gradient 梯度输出缓冲区（大小至少num_variables）
 * @return int 成功返回0，失败返回-1
 */
int gradient(const MathExpressionNode* node,
             const char** variable_names,
             const double* variable_values,
             size_t num_variables,
             double* gradient);

/**
 * @brief 表达式替换：将表达式中的指定变量替换为另一个表达式
 *
 * @param node 原表达式树根节点
 * @param variable_name 被替换的变量名
 * @param replacement 替换表达式
 * @return MathExpressionNode* 替换后的新表达式树，失败返回NULL
 */
MathExpressionNode* expression_substitute(const MathExpressionNode* node,
                                          const char* variable_name,
                                          const MathExpressionNode* replacement);

/**
 * @brief 方程求解：符号求解（支持线性、二次方程）
 *
 * 求解标准形式 ax + b = 0 和 ax² + bx + c = 0
 *
 * @param equation 方程表达式字符串（如 "x^2 - 3*x + 2 = 0"）
 * @param variable 变量名
 * @return EquationSolution* 符号解，失败返回NULL
 */
EquationSolution* solve_equation_symbolic(const char* equation, const char* variable);

/* ========== A04.4.2: 物理量纲分析 ========== */

/**
 * @brief 物理量纲结构体（7个基本量纲的指数）
 *
 * 基于国际单位制(SI)的7个基本量纲：
 * M(质量), L(长度), T(时间), I(电流), Θ(温度), N(物质的量), J(发光强度)
 */
typedef struct {
    int M;  /**< 质量指数 */
    int L;  /**< 长度指数 */
    int T;  /**< 时间指数 */
    int I;  /**< 电流指数 */
    int Th; /**< 温度指数(Theta) */
    int N;  /**< 物质的量指数 */
    int J;  /**< 发光强度指数 */
} PhysicalDimension;

/**
 * @brief 物理单位结构体
 */
typedef struct {
    const char* name;           /**< 单位名称（中文） */
    const char* symbol;         /**< 单位符号 */
    PhysicalDimension dim;      /**< 量纲 */
    double si_conversion;       /**< 到SI单位的转换因子 */
} PhysicalUnit;

/**
 * @brief 物理量结构体
 */
typedef struct {
    double value;               /**< 数值 */
    PhysicalDimension dim;      /**< 量纲 */
    char name[64];              /**< 物理量名称 */
    double uncertainty;         /**< 不确定度 */
} PhysicalQuantity;

/**
 * @brief 物理约束类型枚举
 */
typedef enum {
    CONSTRAINT_ENERGY_CONSERVATION = 0,  /**< 能量守恒 */
    CONSTRAINT_MOMENTUM_CONSERVATION,    /**< 动量守恒 */
    CONSTRAINT_ANGULAR_MOMENTUM_CONSERV, /**< 角动量守恒 */
    CONSTRAINT_NEWTON_SECOND,            /**< 牛顿第二定律 */
    CONSTRAINT_THERMAL_EQUILIBRIUM,      /**< 热平衡 */
    CONSTRAINT_CONTINUITY,               /**< 连续性 */
    CONSTRAINT_CUSTOM                     /**< 自定义约束 */
} PhysicalConstraintType;

/**
 * @brief 物理约束结构体
 */
typedef struct {
    PhysicalConstraintType type; /**< 约束类型 */
    char expression[256];        /**< 约束表达式 */
    double tolerance;            /**< 容差 */
    int is_active;              /**< 是否激活 */
} PhysicalConstraint;

/**
 * @brief 创建物理量
 *
 * @param value 数值
 * @param dim 量纲
 * @param name 物理量名称
 * @return PhysicalQuantity 物理量结构体
 */
PhysicalQuantity physical_quantity_create(double value, PhysicalDimension dim, const char* name);

/**
 * @brief 创建量纲（从指数）
 *
 * @param M 质量指数
 * @param L 长度指数
 * @param T 时间指数
 * @return PhysicalDimension 量纲
 */
PhysicalDimension physical_dimension_create(int M, int L, int T);

/**
 * @brief 创建完整量纲（7个基本量纲）
 */
PhysicalDimension physical_dimension_create_full(int M, int L, int T, int I, int Th, int N, int J);

/**
 * @brief 量纲乘法：两个物理量相乘时量纲指数相加
 *
 * @param a 量纲A
 * @param b 量纲B
 * @return PhysicalDimension 结果量纲
 */
PhysicalDimension physical_dimension_multiply(PhysicalDimension a, PhysicalDimension b);

/**
 * @brief 量纲除法：两个物理量相除时量纲指数相减
 *
 * @param a 量纲A（分子）
 * @param b 量纲B（分母）
 * @return PhysicalDimension 结果量纲
 */
PhysicalDimension physical_dimension_divide(PhysicalDimension a, PhysicalDimension b);

/**
 * @brief 量纲幂运算
 *
 * @param dim 量纲
 * @param power 幂次
 * @return PhysicalDimension 结果量纲
 */
PhysicalDimension physical_dimension_power(PhysicalDimension dim, int power);

/**
 * @brief 检查两个量纲是否一致（可比较/相加）
 *
 * @param a 量纲A
 * @param b 量纲B
 * @return int 一致返回1，否则返回0
 */
int physical_dimension_check_consistent(PhysicalDimension a, PhysicalDimension b);

/**
 * @brief 检查物理量是否符合指定量纲类型
 *
 * @param q 物理量
 * @param expected_dims 期望量纲数组
 * @param dim_count 期望量纲数量
 * @return int 符合返回1，否则返回0
 */
int physical_quantity_check_dimension(const PhysicalQuantity* q,
                                      const PhysicalDimension* expected_dims,
                                      size_t dim_count);

/**
 * @brief 物理量相加（量纲必须一致）
 *
 * @param a 物理量A
 * @param b 物理量B
 * @return PhysicalQuantity 结果物理量（量纲与A一致）
 */
PhysicalQuantity physical_quantity_add(PhysicalQuantity a, PhysicalQuantity b);

/**
 * @brief 物理量相减
 *
 * @param a 物理量A
 * @param b 物理量B
 * @return PhysicalQuantity 结果物理量
 */
PhysicalQuantity physical_quantity_subtract(PhysicalQuantity a, PhysicalQuantity b);

/**
 * @brief 物理量相乘
 *
 * @param a 物理量A
 * @param b 物理量B
 * @return PhysicalQuantity 结果物理量（量纲为A×B）
 */
PhysicalQuantity physical_quantity_multiply(PhysicalQuantity a, PhysicalQuantity b);

/**
 * @brief 物理量相除
 *
 * @param a 物理量A（分子）
 * @param b 物理量B（分母）
 * @return PhysicalQuantity 结果物理量（量纲为A/B）
 */
PhysicalQuantity physical_quantity_divide(PhysicalQuantity a, PhysicalQuantity b);

/**
 * @brief 物理量单位转换
 *
 * @param q 物理量
 * @param target_dim 目标量纲
 * @return PhysicalQuantity 转换后的物理量
 */
PhysicalQuantity physical_quantity_convert(PhysicalQuantity q, PhysicalDimension target_dim);

/**
 * @brief 量纲转为可读字符串
 *
 * @param dim 量纲
 * @param buffer 输出缓冲区
 * @param buffer_size 缓冲区大小
 */
void physical_dimension_to_string(PhysicalDimension dim, char* buffer, size_t buffer_size);

/* ========== A04.4.2: 物理约束推理 ========== */

/**
 * @brief 检查能量守恒约束
 *
 * 检查系统总能量（动能+势能）是否守恒：
 * E_total = E_kinetic + E_potential ≈ constant
 *
 * @param kinetic_energy 动能序列
 * @param potential_energy 势能序列
 * @param count 序列长度
 * @param tolerance 容差
 * @return int 守恒返回1，不守恒返回0
 */
int physics_check_energy_conservation(const double* kinetic_energy,
                                      const double* potential_energy,
                                      size_t count, double tolerance);

/**
 * @brief 检查动量守恒约束
 *
 * @param momentum_before 碰撞前动量
 * @param momentum_after 碰撞后动量
 * @param tolerance 容差
 * @return int 守恒返回1，不守恒返回0
 */
int physics_check_momentum_conservation(double momentum_before,
                                        double momentum_after,
                                        double tolerance);

/**
 * @brief 检查角动量守恒约束
 *
 * @param ang_momentum_before 碰撞前角动量
 * @param ang_momentum_after 碰撞后角动量
 * @param tolerance 容差
 * @return int 守恒返回1，不守恒返回0
 */
int physics_check_angular_momentum_conservation(double ang_momentum_before,
                                                double ang_momentum_after,
                                                double tolerance);

/**
 * @brief 验证物理量是否符合牛顿第二定律 F=ma
 *
 * @param force 力
 * @param mass 质量
 * @param acceleration 加速度
 * @param tolerance 容差
 * @return int 符合返回1，否则返回0
 */
int physics_check_newton_second_law(double force, double mass,
                                    double acceleration, double tolerance);

/**
 * @brief 添加物理约束
 *
 * @param constraints 约束数组指针
 * @param count 当前约束数量
 * @param capacity 约束数组容量
 * @param type 约束类型
 * @param expression 约束表达式
 * @param tolerance 容差
 * @return int 成功返回0，失败返回-1
 */
int physics_add_constraint(PhysicalConstraint** constraints, int* count, int* capacity,
                           PhysicalConstraintType type, const char* expression, double tolerance);

/**
 * @brief 执行所有激活的约束检查
 *
 * @param constraints 约束数组
 * @param count 约束数量
 * @param results 检查结果数组（大小至少count）
 * @return int 通过检查的约束数量
 */
int physics_validate_constraints(const PhysicalConstraint* constraints, int count, int* results);

/**
 * @brief 带约束的物理模拟
 *
 * 在模拟过程中检查并强制执行物理约束
 *
 * @param engine 推理引擎
 * @param config 模拟配置
 * @param initial_conditions 初始条件
 * @param condition_count 初始条件数量
 * @param constraints 物理约束数组
 * @param constraint_count 约束数量
 * @param constraint_results 约束检查结果数组（输出）
 * @return double* 模拟结果，失败返回NULL
 */
double* physics_simulation_with_constraints(MathPhysicsReasoningEngine* engine,
                                            const PhysicsSimConfig* config,
                                            const char** initial_conditions,
                                            size_t condition_count,
                                            const PhysicalConstraint* constraints,
                                            int constraint_count,
                                            int* constraint_results);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MATH_PHYSICS_REASONING_H */