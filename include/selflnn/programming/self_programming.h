/**
 * @file self_programming.h
 * @brief 自我编程能力接口
 * 
 * 自我编程能力核心接口，支持代码解析、生成、分析和自我优化。
 */

#ifndef SELFLNN_SELF_PROGRAMMING_H
#define SELFLNN_SELF_PROGRAMMING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 编程语言类型枚举
 */
typedef enum {
    LANG_C = 0,                 /**< C语言 */
    LANG_CPP = 1,               /**< C++语言 */
    LANG_PYTHON = 2,            /**< Python语言 */
    LANG_CUSTOM = 3             /**< 自定义语言（用于LNN配置） */
} ProgrammingLanguage;

/**
 * @brief AST节点类型枚举
 */
typedef enum {
    AST_PROGRAM = 0,            /**< 程序节点 */
    AST_FUNCTION = 1,           /**< 函数定义 */
    AST_VARIABLE_DECL = 2,      /**< 变量声明 */
    AST_VARIABLE_REF = 3,       /**< 变量引用 */
    AST_LITERAL = 4,            /**< 字面量 */
    AST_BINARY_OP = 5,          /**< 二元操作 */
    AST_UNARY_OP = 6,           /**< 一元操作 */
    AST_CALL = 7,               /**< 函数调用 */
    AST_RETURN = 8,             /**< 返回语句 */
    AST_IF = 9,                 /**< 条件语句 */
    AST_LOOP = 10,              /**< 循环语句 */
    AST_ASSIGNMENT = 11,        /**< 赋值语句 */
    AST_BLOCK = 12              /**< 代码块 */
} ASTNodeType;

/**
 * @brief 二元操作符枚举
 */
typedef enum {
    OP_ADD = 0,                 /**< 加法 */
    OP_SUB = 1,                 /**< 减法 */
    OP_MUL = 2,                 /**< 乘法 */
    OP_DIV = 3,                 /**< 除法 */
    OP_MOD = 4,                 /**< 取模 */
    OP_EQ = 5,                  /**< 等于 */
    OP_NE = 6,                  /**< 不等于 */
    OP_LT = 7,                  /**< 小于 */
    OP_GT = 8,                  /**< 大于 */
    OP_LE = 9,                  /**< 小于等于 */
    OP_GE = 10,                 /**< 大于等于 */
    OP_AND = 11,                /**< 逻辑与 */
    OP_OR = 12                  /**< 逻辑或 */
} BinaryOperator;

/**
 * @brief 一元操作符枚举
 */
typedef enum {
    UNARY_PLUS = 0,             /**< 正号 */
    UNARY_MINUS = 1,            /**< 负号 */
    UNARY_NOT = 2               /**< 逻辑非 */
} UnaryOperator;

/**
 * @brief 数据类型枚举
 */
#ifndef SELFLNN_DATA_TYPE_DEFINED
#define SELFLNN_DATA_TYPE_DEFINED
typedef enum {
    TYPE_VOID = 0,              /**< void类型 */
    TYPE_INT = 1,               /**< 整数类型 */
    TYPE_FLOAT = 2,             /**< 浮点类型 */
    TYPE_BOOL = 3,              /**< 布尔类型 */
    TYPE_STRING = 4,            /**< 字符串类型 */
    TYPE_CUSTOM = 5             /**< 自定义类型 */
} DataType;
#endif /* SELFLNN_DATA_TYPE_DEFINED */

/**
 * @brief AST节点结构体
 */
typedef struct ASTNode ASTNode;

/**
 * @brief AST节点
 */
struct ASTNode {
    ASTNodeType type;           /**< 节点类型 */
    ASTNode* left;              /**< 左子节点 */
    ASTNode* right;             /**< 右子节点 */
    ASTNode* next;              /**< 下一个节点（用于语句序列） */
    union {
        char* string_value;     /**< 字符串值 */
        int int_value;          /**< 整数值 */
        float float_value;      /**< 浮点值 */
        BinaryOperator bin_op;  /**< 二元操作符 */
        UnaryOperator unary_op; /**< 一元操作符 */
        DataType data_type;     /**< 数据类型 */
    } value;                    /**< 节点值 */
    char* name;                 /**< 节点名称（变量名、函数名等） */
    int line;                   /**< 源代码行号 */
    int column;                 /**< 源代码列号 */
    void* data;                 /**< 通用数据指针 */
    int child_count;            /**< 子节点数量 */
    ASTNode** children;         /**< 子节点数组 */
};

/**
 * @brief 代码分析结果
 */
typedef struct {
    int cyclomatic_complexity;  /**< 圈复杂度 */
    int line_count;             /**< 代码行数 */
    int function_count;         /**< 函数数量 */
    int variable_count;         /**< 变量数量 */
    int max_nesting_depth;      /**< 最大嵌套深度 */
    float maintainability_index; /**< 可维护性指数 */
    int error_count;            /**< 错误数量 */
    int warning_count;          /**< 警告数量 */
} CodeAnalysisResult;

/**
 * @brief 代码优化建议
 */
typedef struct {
    char** suggestions;         /**< 优化建议数组 */
    size_t suggestion_count;    /**< 建议数量 */
    float* confidence;          /**< 置信度数组 */
    int* priority;              /**< 优先级数组 */
} OptimizationSuggestions;

/**
 * @brief 自我编程引擎句柄
 */
typedef struct SelfProgrammingEngine SelfProgrammingEngine;

/**
 * @brief 创建自我编程引擎
 * 
 * @param language 目标编程语言
 * @return SelfProgrammingEngine* 引擎句柄，失败返回NULL
 */
SelfProgrammingEngine* self_programming_engine_create(ProgrammingLanguage language);

/**
 * @brief 销毁自我编程引擎
 * 
 * @param engine 引擎句柄
 */
void self_programming_engine_destroy(SelfProgrammingEngine* engine);

/**
 * @brief 解析源代码为AST
 * 
 * @param engine 引擎句柄
 * @param source_code 源代码字符串
 * @return ASTNode* AST根节点，失败返回NULL
 */
ASTNode* parse_source_code(SelfProgrammingEngine* engine, const char* source_code);

/**
 * @brief 销毁AST
 * 
 * @param ast AST根节点
 */
void ast_destroy(ASTNode* ast);

/**
 * @brief 从AST生成源代码
 * 
 * @param engine 引擎句柄
 * @param ast AST根节点
 * @return char* 生成的源代码，失败返回NULL
 */
char* generate_source_code(SelfProgrammingEngine* engine, const ASTNode* ast);

/**
 * @brief 分析代码复杂度
 * 
 * @param engine 引擎句柄
 * @param ast AST根节点
 * @param result 分析结果输出
 * @return int 成功返回0，失败返回错误码
 */
int analyze_code_complexity(SelfProgrammingEngine* engine, 
                            const ASTNode* ast, 
                            CodeAnalysisResult* result);

/**
 * @brief 生成优化建议
 * 
 * @param engine 引擎句柄
 * @param ast AST根节点
 * @param analysis 代码分析结果
 * @return OptimizationSuggestions* 优化建议，失败返回NULL
 */
OptimizationSuggestions* generate_optimization_suggestions(
    SelfProgrammingEngine* engine,
    const ASTNode* ast,
    const CodeAnalysisResult* analysis);

/**
 * @brief 销毁优化建议
 * 
 * @param suggestions 优化建议
 */
void optimization_suggestions_destroy(OptimizationSuggestions* suggestions);

/**
 * @brief 应用代码优化
 * 
 * @param engine 引擎句柄
 * @param ast AST根节点（将被修改）
 * @param suggestions 优化建议
 * @return int 成功返回0，失败返回错误码
 */
int apply_code_optimizations(SelfProgrammingEngine* engine,
                             ASTNode* ast,
                             const OptimizationSuggestions* suggestions);

/**
 * @brief 自我改进：基于分析自动优化代码
 * 
 * @param engine 引擎句柄
 * @param source_code 源代码
 * @param iterations 迭代次数
 * @return char* 优化后的源代码，失败返回NULL
 */
char* self_improve_code(SelfProgrammingEngine* engine,
                        const char* source_code,
                        int iterations);

/**
 * @brief 编译验证结果
 */
typedef struct {
    int success;                 /**< 编译是否成功 */
    char error_message[4096];    /**< 编译错误信息 */
    int warning_count;           /**< 编译警告数量 */
    int error_count;             /**< 编译错误数量 */
} CompilationResult;

/**
 * @brief 验证代码是否能编译通过
 * 
 * 将源代码写入临时文件，调用系统编译器编译验证。
 * 根据平台自动选择编译器：MSVC(cl.exe)、GCC、Clang。
 * 
 * @param engine 引擎句柄
 * @param source_code 源代码
 * @return CompilationResult 编译验证结果
 */
CompilationResult verify_code_compilation(SelfProgrammingEngine* engine,
                                          const char* source_code);

/**
 * @brief 编译错误反馈 — 将编译失败信息反馈回引擎用于代码质量改进
 * 
 * 当 verify_code_compilation 返回失败时，此函数将错误信息注入引擎的
 * 错误历史记录，使后续代码生成能避开已知的编译错误模式。
 *修复: 闭合编译→生成反馈循环。
 * 
 * @param engine 引擎句柄
 * @param source_code 编译失败的源代码
 * @param error_message 编译器错误信息
 */
void self_programming_feedback_compile_error(SelfProgrammingEngine* engine,
                                              const char* source_code,
                                              const char* error_message);

/**
 * @brief 在沙箱环境中执行代码片段
 * 
 * 创建隔离的执行环境：
 * - Windows: 使用作业对象+限制令牌
 * - Linux: 使用fork()+seccomp+namespace
 * - macOS: 使用sandbox_init()+XPC
 * 
 * @param engine 引擎句柄
 * @param code 要执行的代码
 * @param input 输入参数（可为NULL）
 * @param output 输出缓冲区（可为NULL）
 * @param output_size 输出缓冲区大小
 * @return int 成功返回0，执行异常返回-1
 */
int execute_code_sandboxed(SelfProgrammingEngine* engine,
                           const char* code,
                           const char* input,
                           char* output,
                           size_t output_size);

/**
 * @brief 代码质量自检结果
 */
typedef struct {
    int has_syntax_errors;       /**< 是否有语法错误 */
    int has_undefined_symbols;   /**< 是否有未定义符号 */
    int has_type_mismatch;       /**< 是否有类型不匹配 */
    int recursion_depth_safe;    /**< 递归深度是否安全 */
    int memory_safe;             /**< 内存访问是否安全 */
    float quality_score;         /**< 综合质量评分 0.0-100.0 */
    int issue_count;             /**< 问题总数 */
    char details[8192];          /**< 质量检查详细报告 */
} CodeQualityResult;

/**
 * @brief 自检代码生成质量
 * 
 * 对生成的代码进行全面的质量检查：
 * - 语法正确性验证
 * - 符号引用完整性
 * - 类型一致性检查
 * - 递归深度分析
 * - 内存安全初步分析
 * 
 * @param engine 引擎句柄
 * @param source_code 源代码
 * @return CodeQualityResult 质量检查结果
 */
CodeQualityResult self_check_code_gen_quality(SelfProgrammingEngine* engine,
                                               const char* source_code);

/**
 * @brief Python代码生成类型
 */
typedef enum {
    PYGEN_SCRIPT = 0,           /**< 通用Python脚本 */
    PYGEN_MODULE = 1,           /**< Python模块 */
    PYGEN_CLASS = 2,            /**< Python类 */
    PYGEN_FUNCTION = 3,         /**< Python函数 */
    PYGEN_PYBULLET = 4,         /**< PyBullet仿真脚本 */
    PYGEN_NETWORK = 5,          /**< 神经网络训练脚本 */
    PYGEN_DATA_ANALYSIS = 6,    /**< 数据分析脚本 */
    PYGEN_WEB_API = 7,          /**< Web API服务 */
    PYGEN_HARDWARE = 8          /**< 硬件控制脚本 */
} PythonGenType;

/**
 * @brief Python生成配置
 */
typedef struct {
    PythonGenType gen_type;      /**< 生成类型 */
    const char* module_name;     /**< 模块名称（可选） */
    const char* class_name;      /**< 类名称（可选） */
    const char* function_name;   /**< 函数名称（可选） */
    const char* description;     /**< 功能描述 */
    const char* author;          /**< 作者（可选） */
    int include_type_hints;      /**< 是否包含类型提示 */
    int include_docstrings;      /**< 是否包含文档字符串 */
    int include_tests;           /**< 是否包含测试代码 */
    int include_main;            /**< 是否包含__main__入口 */
    int include_logging;         /**< 是否包含日志配置 */
    int async_mode;              /**< 是否使用异步模式 */
    float version;               /**< 版本号 */
} PythonGenConfig;

#define PYTHON_GEN_CONFIG_DEFAULT { \
    PYGEN_SCRIPT, NULL, NULL, NULL, NULL, NULL, 1, 1, 0, 0, 0, 0, 1.0f \
}

/**
 * @brief 生成Python代码
 * 
 * 根据配置生成Python代码，支持多种生成类型：
 * - PYGEN_PYBULLET: 生成PyBullet机器人仿真控制脚本
 * - PYGEN_NETWORK: 生成神经网络训练/推理脚本
 * - PYGEN_HARDWARE: 生成硬件接口控制脚本
 * - PYGEN_DATA_ANALYSIS: 生成数据分析/可视化脚本
 * - PYGEN_WEB_API: 生成Web API服务脚本
 * 
 * @param engine 自我编程引擎句柄
 * @param config Python生成配置
 * @return char* 生成的Python代码，失败返回NULL（调用者需safe_free）
 */
char* self_programming_generate_python(SelfProgrammingEngine* engine,
                                       const PythonGenConfig* config);

/* ============================================================================
 * F-11: 代码合成数据结构（前置：CodeSpecification前向声明区域）
 * ============================================================================ */

/**
 * @brief I/O示例对（用于代码合成）
 */
typedef struct {
    double* inputs;              /**< 输入值数组 */
    size_t input_count;          /**< 输入值数量 */
    double expected_output;      /**< 期望输出值 */
} IOExample;

/**
 * @brief 代码规格说明
 */
typedef struct {
    char* function_name;         /**< 函数名称 */
    char* description;           /**< 功能描述 */
    DataType return_type;        /**< 返回类型 */
    DataType* param_types;       /**< 参数类型数组 */
    char** param_names;          /**< 参数名称数组 */
    size_t param_count;          /**< 参数数量 */
    char* precondition;          /**< 前置条件（自然语言描述） */
    char* postcondition;         /**< 后置条件（自然语言描述） */
    IOExample* examples;         /**< I/O示例数组 */
    size_t example_count;        /**< 示例数量 */
} CodeSpecification;

/**
 * @brief 生成C语言代码（移到CodeSpecification定义之后）
 * 
 * 从代码规格说明生成完整的C语言函数实现。
 * 
 * @param engine 自我编程引擎句柄
 * @param spec 代码规格说明
 * @return char* 生成的C代码字符串，失败返回NULL（调用者需safe_free）
 */
char* self_programming_generate_c(SelfProgrammingEngine* engine,
                                  const CodeSpecification* spec);

/* ============================================================================
 * F-11: 语义分析数据结构
 * =========================================================================== */

/**
 * @brief 符号作用域
 */
typedef enum {
    SCOPE_GLOBAL = 0,           /**< 全局作用域 */
    SCOPE_FUNCTION = 1,         /**< 函数作用域 */
    SCOPE_BLOCK = 2,            /**< 块作用域 */
    SCOPE_PARAM = 3             /**< 参数作用域 */
} SymbolScope;

/**
 * @brief 符号信息
 */
typedef struct {
    char* name;                  /**< 符号名称 */
    DataType type;               /**< 符号类型 */
    SymbolScope scope;           /**< 定义作用域 */
    int declared_line;           /**< 声明行号 */
    int is_initialized;          /**< 是否已初始化 */
    int is_used;                 /**< 是否被使用 */
    int ref_count;               /**< 引用计数 */
} SymbolInfo;

/**
 * @brief 语义分析结果
 */
typedef struct {
    SymbolInfo* symbols;         /**< 符号表 */
    size_t symbol_count;         /**< 符号数量 */
    char** type_errors;          /**< 类型错误列表 */
    size_t type_error_count;     /**< 类型错误数量 */
    char** scope_errors;         /**< 作用域错误列表 */
    size_t scope_error_count;    /**< 作用域错误数量 */
    int has_uninitialized_var;   /**< 是否有未初始化的变量 */
    int has_unused_var;          /**< 是否有未使用的变量 */
    char details[8192];          /**< 分析详细报告 */
} SemanticAnalysisResult;

/**
 * @brief 数据流分析结果
 */
typedef struct {
    int** def_use_chains;        /**< 定值-使用链矩阵 [var_count x ref_count] */
    size_t var_count;            /**< 变量数量 */
    size_t ref_count;            /**< 引用数量 */
    int* live_variables;         /**< 活跃变量标记 */
    int has_uninitialized_read;  /**< 是否有未初始化读取 */
    int has_dead_assignment;     /**< 是否有死赋值 */
    char details[4096];          /**< 数据流分析详细报告 */
} DataFlowResult;

/* ============================================================================
 * F-11: 代码转换数据结构
 * =========================================================================== */

/**
 * @brief 转换规则类型
 */
typedef enum {
    TRANSFORM_CONSTANT_FOLDING = 0,    /**< 常量折叠 */
    TRANSFORM_DEAD_CODE_ELIM = 1,      /**< 死代码消除 */
    TRANSFORM_LOOP_INVARIANT = 2,      /**< 循环不变式外提 */
    TRANSFORM_STRENGTH_REDUCTION = 3,  /**< 强度削弱 */
    TRANSFORM_CSE = 4                  /**< 公共子表达式消除 */
} TransformRuleType;

/**
 * @brief 转换结果
 */
typedef struct {
    ASTNode* transformed_ast;    /**< 转换后的AST */
    char** applied_rules;        /**< 已应用的规则说明 */
    size_t rule_count;           /**< 规则数量 */
    char details[4096];          /**< 转换详细报告 */
} TransformResult;

/* ============================================================================
 * F-11: API声明 - 代码合成
 * =========================================================================== */

/**
 * @brief 创建代码规格说明
 * 
 * @param function_name 函数名称
 * @param description 功能描述
 * @param return_type 返回类型
 * @return CodeSpecification* 创建的规格说明，失败返回NULL
 */
CodeSpecification* create_code_specification(const char* function_name,
                                             const char* description,
                                             DataType return_type);

/**
 * @brief 向规格说明添加参数
 * 
 * @param spec 规格说明
 * @param name 参数名称
 * @param type 参数类型
 * @return int 成功返回0，失败返回错误码
 */
int code_spec_add_parameter(CodeSpecification* spec, const char* name, DataType type);

/**
 * @brief 向规格说明添加I/O示例
 * 
 * @param spec 规格说明
 * @param inputs 输入值数组
 * @param input_count 输入值数量
 * @param expected_output 期望输出
 * @return int 成功返回0，失败返回错误码
 */
int code_spec_add_example(CodeSpecification* spec, const double* inputs,
                          size_t input_count, double expected_output);

/**
 * @brief 设置前置条件
 * 
 * @param spec 规格说明
 * @param precondition 前置条件文本
 * @return int 成功返回0，失败返回错误码
 */
int code_spec_set_precondition(CodeSpecification* spec, const char* precondition);

/**
 * @brief 设置后置条件
 * 
 * @param spec 规格说明
 * @param postcondition 后置条件文本
 * @return int 成功返回0，失败返回错误码
 */
int code_spec_set_postcondition(CodeSpecification* spec, const char* postcondition);

/**
 * @brief 销毁代码规格说明
 * 
 * @param spec 规格说明
 */
void code_specification_destroy(CodeSpecification* spec);

/**
 * @brief 从规格说明合成代码
 * 
 * 根据规格说明（函数签名、条件、I/O示例）自动合成C代码。
 * 使用模式匹配和实例学习策略：
 * 1. 从示例中提取操作模式（加减乘除比较）
 * 2. 根据条件约束筛选有效操作
 * 3. 生成完整的函数定义代码
 * 
 * @param engine 引擎句柄
 * @param spec 规格说明
 * @return char* 合成的源代码，失败返回NULL
 */
char* synthesize_code(SelfProgrammingEngine* engine, const CodeSpecification* spec);

/* ============================================================================
 * F-11: API声明 - 语义分析
 * =========================================================================== */

/**
 * @brief 执行语义分析
 * 
 * 对AST进行全面的语义分析：
 * - 符号表构建（多作用域）
 * - 类型一致性检查
 * - 作用域规则验证
 * - 变量初始化状态跟踪
 * 
 * @param engine 引擎句柄
 * @param ast AST根节点
 * @return SemanticAnalysisResult* 分析结果，失败返回NULL
 */
SemanticAnalysisResult* semantic_analyze(SelfProgrammingEngine* engine,
                                         const ASTNode* ast);

/**
 * @brief 销毁语义分析结果
 * 
 * @param result 语义分析结果
 */
void semantic_analysis_destroy(SemanticAnalysisResult* result);

/**
 * @brief 执行数据流分析
 * 
 * 构建定值-使用链，识别：
 * - 未初始化变量读取
 * - 死赋值（定值后从未使用）
 * - 活跃变量
 * 
 * @param engine 引擎句柄
 * @param ast AST根节点
 * @return DataFlowResult* 数据流分析结果，失败返回NULL
 */
DataFlowResult* analyze_data_flow(SelfProgrammingEngine* engine,
                                  const ASTNode* ast);

/**
 * @brief 销毁数据流分析结果
 * 
 * @param result 数据流分析结果
 */
void data_flow_destroy(DataFlowResult* result);

/* ============================================================================
 * F-11: API声明 - 代码转换
 * =========================================================================== */

/**
 * @brief 对AST应用代码转换
 * 
 * @param engine 引擎句柄
 * @param ast 输入AST（不会被修改，只读分析用）
 * @param rules 要应用的转换规则类型数组
 * @param rule_count 规则数量
 * @return TransformResult* 转换结果（含转换后的AST和规则说明），失败返回NULL
 */
TransformResult* transform_code(SelfProgrammingEngine* engine,
                                const ASTNode* ast,
                                const TransformRuleType* rules,
                                size_t rule_count);

/**
 * @brief 常量折叠：将编译时可计算的常量表达式替换为计算结果
 * 
 * 遍历AST，识别并计算纯常量表达式（如 3+5 → 8, 4*2 → 8）。
 * 
 * @param engine 引擎句柄
 * @param ast AST根节点（将被修改）
 * @return int 成功返回0，失败返回错误码
 */
int constant_folding(SelfProgrammingEngine* engine, ASTNode* ast);

/**
 * @brief 死代码消除：移除不可达代码片段
 * 
 * 识别并删除return/break后的不可达语句、if(false)分支等。
 * 
 * @param engine 引擎句柄
 * @param ast AST根节点（将被修改）
 * @return int 成功返回0，失败返回错误码
 */
int dead_code_elimination(SelfProgrammingEngine* engine, ASTNode* ast);

/**
 * @brief 循环不变式外提：将循环内不随迭代变化的计算移到循环外
 * 
 * 识别循环体内变量引用模式，将与循环变量无关的计算外提到循环之前。
 * 
 * @param engine 引擎句柄
 * @param ast AST根节点（将被修改）
 * @return int 成功返回0，失败返回错误码
 */
int loop_invariant_hoisting(SelfProgrammingEngine* engine, ASTNode* ast);

/**
 * @brief 销毁转换结果
 * 
 * @param result 转换结果
 */
void transform_result_destroy(TransformResult* result);

/* ============================================================================
 * P1-05: AST级变换与优化（新增）
 * ============================================================================ */

/**
 * @brief 深拷贝AST子树
 *
 * @param node AST节点
 * @return 拷贝后的AST节点，失败返回NULL
 */
ASTNode* ast_clone(const ASTNode* node);

/**
 * @brief AST优化管道：常量折叠 + 死代码消除
 *
 * 依次应用常量折叠和死代码消除优化，最多5轮迭代直至收敛。
 *
 * @param ast AST根节点
 * @return 优化后的AST（可能与输入相同或为新AST）
 */
ASTNode* ast_optimize_pipeline(ASTNode* ast);

/**
 * @brief 查找AST中指定名称的变量引用节点
 *
 * @param ast AST根节点
 * @param var_name 变量名
 * @return 第一个匹配的AST_VARIABLE_REF节点，未找到返回NULL
 */
ASTNode* ast_find_variable_ref(ASTNode* ast, const char* var_name);

/* ================================================================
 * K-007: 纯C表达式解释器（当外部编译器不可用时备选执行）
 * ================================================================ */

/**
 * @brief K-007: 纯C表达式解释执行
 *
 * 当外部C编译器不可用时，使用内置解释器执行表达式。
 * 支持: 算术运算(+ - * /)、变量赋值(=)、数学函数(sin/cos/sqrt/abs/rand)、
 *       比较运算(< > ==)
 *
 * @param code 待执行的代码字符串
 * @param result [输出] 计算结果
 * @param error_msg [输出] 错误信息(256字节缓冲区，可NULL)
 * @return 0成功，-1失败
 */
int self_programming_interpret_expr(const char* code, float* result, char* error_msg);

/**
 * @brief K-007: 检查内置解释器是否可用
 *
 * 解释器始终可用(100%纯C实现)。
 * 能力有限仅支持表达式求值(非完整C程序执行)。
 *
 * @return 1可用
 */
int self_programming_interpreter_available(void);

/**
 * @brief K-007: 获取解释器能力描述
 *
 * @return 描述字符串
 */
const char* self_programming_interpreter_capability(void);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SELF_PROGRAMMING_H */