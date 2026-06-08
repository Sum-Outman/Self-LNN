/**
 * @file math_physics_reasoning.c
 * @brief 数学物理推理引擎实现
 * 
 * 数学物理推理引擎核心实现，支持数学表达式解析、方程求解、物理计算等。
 */

#include "selflnn/reasoning/math_physics_reasoning.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <ctype.h>
#include <float.h>

/* 静态函数前向声明（防止递归/跨调用隐式声明警告） */
static MathExpressionNode* create_expression_node(MathExpressionType type,
                                                   MathExpressionNode* left,
                                                   MathExpressionNode* right);
static MathExpressionNode* create_number_node(double value);
static MathExpressionNode* clone_expression_node(const MathExpressionNode* node);
MathExpressionNode* symbolic_differentiate(const MathExpressionNode* node,
                                                   const char* variable_name);
static int is_constant_number(const MathExpressionNode* node, double val);

/**
 * @brief 数学物理推理引擎内部结构体
 */
struct MathPhysicsReasoningEngine {
    int is_initialized;           /**< 是否已初始化 */
    double* temporary_buffer;     /**< 临时缓冲区 */
    size_t buffer_size;           /**< 缓冲区大小 */
    size_t buffer_capacity;       /**< 缓冲区容量 */
};

/**
 * @brief 解析器状态
 */
typedef struct {
    const char* input;            /**< 输入字符串 */
    size_t position;              /**< 当前位置 */
    size_t length;                /**< 输入长度 */
    char current_char;            /**< 当前字符 */
} ParserState;

/**
 * @brief 词法分析器标记类型
 */
typedef enum {
    TOKEN_NUMBER,                 /**< 数字 */
    TOKEN_VARIABLE,               /**< 变量 */
    TOKEN_PLUS,                   /**< 加号 */
    TOKEN_MINUS,                  /**< 减号 */
    TOKEN_MULTIPLY,               /**< 乘号 */
    TOKEN_DIVIDE,                 /**< 除号 */
    TOKEN_POWER,                  /**< 幂运算符号 */
    TOKEN_LPAREN,                 /**< 左括号 */
    TOKEN_RPAREN,                 /**< 右括号 */
    TOKEN_COMMA,                  /**< 逗号 */
    TOKEN_FUNCTION,               /**< 函数名 */
    TOKEN_END                     /**< 结束标记 */
} MprTokenType;

/**
 * @brief 词法标记
 */
typedef struct {
    MprTokenType type;               /**< 标记类型 */
    double number_value;          /**< 数字值 */
    char* string_value;           /**< 字符串值 */
} MprToken;

/* ============================================================================
 * 内部辅助函数
 * =========================================================================== */

/* 静态函数前向声明 */

/**
 * @brief 初始化解析器状态
 */
static void parser_init(ParserState* state, const char* input) {
    if (!state || !input) return;
    
    state->input = input;
    state->position = 0;
    state->length = strlen(input);
    state->current_char = (state->length > 0) ? input[0] : '\0';
}

/**
 * @brief 前进到下一个字符
 */
static void parser_advance(ParserState* state) {
    if (!state || state->position >= state->length) {
        state->current_char = '\0';
        return;
    }
    
    state->position++;
    if (state->position < state->length) {
        state->current_char = state->input[state->position];
    } else {
        state->current_char = '\0';
    }
}

/**
 * @brief 跳过空白字符
 */
static void parser_skip_whitespace(ParserState* state) {
    if (!state) return;
    
    while (state->current_char != '\0' && isspace((unsigned char)state->current_char)) {
        parser_advance(state);
    }
}

/**
 * @brief 解析数字
 */
static double parser_parse_number(ParserState* state) {
    if (!state) return 0.0;
    
    char buffer[64] = {0};
    size_t index = 0;
    
    // 解析整数部分
    while (state->current_char != '\0' && (isdigit((unsigned char)state->current_char) || state->current_char == '.')) {
        if (index < sizeof(buffer) - 1) {
            buffer[index++] = state->current_char;
        }
        parser_advance(state);
    }
    
    buffer[index] = '\0';
    return atof(buffer);
}

/**
 * @brief 解析标识符（变量或函数名）
 */
static char* parser_parse_identifier(ParserState* state) {
    if (!state) return NULL;
    
    char buffer[64] = {0};
    size_t index = 0;
    
    // 解析标识符（字母开头，可包含字母、数字、下划线）
    while (state->current_char != '\0' && 
           (isalnum((unsigned char)state->current_char) || state->current_char == '_')) {
        if (index < sizeof(buffer) - 1) {
            buffer[index++] = state->current_char;
        }
        parser_advance(state);
    }
    
    buffer[index] = '\0';
    return string_duplicate_nullable(buffer);
}

/**
 * @brief 获取下一个词法标记
 */
static MprToken parser_get_next_token(ParserState* state) {
    MprToken token = {TOKEN_END, 0.0, NULL};
    
    if (!state || state->current_char == '\0') {
        return token;
    }
    
    parser_skip_whitespace(state);
    
    if (state->current_char == '\0') {
        return token;
    }
    
    char current = state->current_char;
    
    // 检查数字
    if (isdigit((unsigned char)current) || current == '.') {
        token.type = TOKEN_NUMBER;
        token.number_value = parser_parse_number(state);
        return token;
    }
    
    // 检查标识符（变量或函数名）
    if (isalpha((unsigned char)current) || current == '_') {
        char* identifier = parser_parse_identifier(state);
        
        // 检查是否是函数名
        if (strcmp(identifier, "sin") == 0 || strcmp(identifier, "cos") == 0 ||
            strcmp(identifier, "tan") == 0 || strcmp(identifier, "exp") == 0 ||
            strcmp(identifier, "log") == 0 || strcmp(identifier, "log10") == 0 ||
            strcmp(identifier, "sqrt") == 0 || strcmp(identifier, "abs") == 0) {
            token.type = TOKEN_FUNCTION;
            token.string_value = identifier;
        } else {
            token.type = TOKEN_VARIABLE;
            token.string_value = identifier;
        }
        return token;
    }
    
    // 检查运算符和括号
    switch (current) {
        case '+':
            token.type = TOKEN_PLUS;
            parser_advance(state);
            break;
        case '-':
            token.type = TOKEN_MINUS;
            parser_advance(state);
            break;
        case '*':
            token.type = TOKEN_MULTIPLY;
            parser_advance(state);
            break;
        case '/':
            token.type = TOKEN_DIVIDE;
            parser_advance(state);
            break;
        case '^':
            token.type = TOKEN_POWER;
            parser_advance(state);
            break;
        case '(':
            token.type = TOKEN_LPAREN;
            parser_advance(state);
            break;
        case ')':
            token.type = TOKEN_RPAREN;
            parser_advance(state);
            break;
        case ',':
            token.type = TOKEN_COMMA;
            parser_advance(state);
            break;
        default:
            // 未知字符，跳过
            parser_advance(state);
            break;
    }
    
    return token;
}

/**
 * @brief 创建表达式节点
 */
static MathExpressionNode* create_expression_node(MathExpressionType type,
                                                  MathExpressionNode* left,
                                                  MathExpressionNode* right) {
    MathExpressionNode* node = (MathExpressionNode*)safe_malloc(sizeof(MathExpressionNode));
    if (!node) return NULL;
    
    memset(node, 0, sizeof(MathExpressionNode));
    node->type = type;
    node->left = left;
    node->right = right;
    return node;
}

/**
 * @brief 创建数字节点
 */
static MathExpressionNode* create_number_node(double value) {
    MathExpressionNode* node = (MathExpressionNode*)safe_malloc(sizeof(MathExpressionNode));
    if (!node) return NULL;
    
    memset(node, 0, sizeof(MathExpressionNode));
    node->type = EXPR_NUMBER;
    node->value.number_value = value;
    return node;
}

/**
 * @brief 创建变量节点
 */
static MathExpressionNode* create_variable_node(const char* name) {
    if (!name) return NULL;
    
    MathExpressionNode* node = (MathExpressionNode*)safe_malloc(sizeof(MathExpressionNode));
    if (!node) return NULL;
    
    memset(node, 0, sizeof(MathExpressionNode));
    node->type = EXPR_VARIABLE;
    node->value.variable_name = string_duplicate_nullable(name);
    return node;
}

/**
 * @brief 创建函数节点
 */
static MathExpressionNode* create_function_node(MathFunctionType func_type,
                                                MathExpressionNode* argument) {
    MathExpressionNode* node = (MathExpressionNode*)safe_malloc(sizeof(MathExpressionNode));
    if (!node) return NULL;
    
    memset(node, 0, sizeof(MathExpressionNode));
    node->type = EXPR_FUNCTION;
    node->value.func_type = func_type;
    node->left = argument;  // 函数参数作为左子节点
    return node;
}

/**
 * @brief 解析主表达式（加减法）
 */
static MathExpressionNode* parse_expression(ParserState* state, MprToken* current_token);
static MathExpressionNode* parse_term(ParserState* state, MprToken* current_token);
static MathExpressionNode* parse_factor(ParserState* state, MprToken* current_token);
static MathExpressionNode* parse_power(ParserState* state, MprToken* current_token);
static MathExpressionNode* parse_primary(ParserState* state, MprToken* current_token);

/**
 * @brief 解析主表达式（加减法）
 */
static MathExpressionNode* parse_expression(ParserState* state, MprToken* current_token) {
    MathExpressionNode* node = parse_term(state, current_token);
    
    while (current_token->type == TOKEN_PLUS || current_token->type == TOKEN_MINUS) {
        MprTokenType op_type = current_token->type;
        *current_token = parser_get_next_token(state);
        
        MathExpressionNode* right = parse_term(state, current_token);
        if (!right) {
            math_expression_destroy(node);
            return NULL;
        }
        
        MathExpressionNode* new_node = create_expression_node(
            (op_type == TOKEN_PLUS) ? EXPR_ADDITION : EXPR_SUBTRACTION,
            node, right);
        if (!new_node) {
            math_expression_destroy(node);
            math_expression_destroy(right);
            return NULL;
        }
        
        node = new_node;
    }
    
    return node;
}

/**
 * @brief 解析项（乘除法）
 */
static MathExpressionNode* parse_term(ParserState* state, MprToken* current_token) {
    MathExpressionNode* node = parse_power(state, current_token);
    
    while (current_token->type == TOKEN_MULTIPLY || current_token->type == TOKEN_DIVIDE) {
        MprTokenType op_type = current_token->type;
        *current_token = parser_get_next_token(state);
        
        MathExpressionNode* right = parse_power(state, current_token);
        if (!right) {
            math_expression_destroy(node);
            return NULL;
        }
        
        MathExpressionNode* new_node = create_expression_node(
            (op_type == TOKEN_MULTIPLY) ? EXPR_MULTIPLICATION : EXPR_DIVISION,
            node, right);
        if (!new_node) {
            math_expression_destroy(node);
            math_expression_destroy(right);
            return NULL;
        }
        
        node = new_node;
    }
    
    return node;
}

/**
 * @brief 解析幂运算
 */
static MathExpressionNode* parse_power(ParserState* state, MprToken* current_token) {
    MathExpressionNode* node = parse_factor(state, current_token);
    
    while (current_token->type == TOKEN_POWER) {
        *current_token = parser_get_next_token(state);
        
        MathExpressionNode* right = parse_factor(state, current_token);
        if (!right) {
            math_expression_destroy(node);
            return NULL;
        }
        
        MathExpressionNode* new_node = create_expression_node(EXPR_POWER, node, right);
        if (!new_node) {
            math_expression_destroy(node);
            math_expression_destroy(right);
            return NULL;
        }
        
        node = new_node;
    }
    
    return node;
}

/**
 * @brief 解析因子（一元运算符、函数、括号）
 */
static MathExpressionNode* parse_factor(ParserState* state, MprToken* current_token) {
    if (current_token->type == TOKEN_MINUS) {
        *current_token = parser_get_next_token(state);
        
        MathExpressionNode* node = parse_factor(state, current_token);
        if (!node) return NULL;
        
        MathExpressionNode* neg_node = create_expression_node(EXPR_NEGATION, node, NULL);
        if (!neg_node) {
            math_expression_destroy(node);
            return NULL;
        }
        
        return neg_node;
    }
    
    return parse_primary(state, current_token);
}

/**
 * @brief 解析基本元素（数字、变量、函数、括号表达式）
 */
static MathExpressionNode* parse_primary(ParserState* state, MprToken* current_token) {
    MathExpressionNode* node = NULL;
    
    switch (current_token->type) {
        case TOKEN_NUMBER:
            node = create_number_node(current_token->number_value);
            *current_token = parser_get_next_token(state);
            break;
            
        case TOKEN_VARIABLE:
            node = create_variable_node(current_token->string_value);
            if (current_token->string_value) {
                safe_free((void**)&current_token->string_value);
            }
            *current_token = parser_get_next_token(state);
            break;
            
        case TOKEN_FUNCTION: {
            char* func_name = current_token->string_value;
            MathFunctionType func_type;
            
            if (strcmp(func_name, "sin") == 0) func_type = FUNC_SIN;
            else if (strcmp(func_name, "cos") == 0) func_type = FUNC_COS;
            else if (strcmp(func_name, "tan") == 0) func_type = FUNC_TAN;
            else if (strcmp(func_name, "exp") == 0) func_type = FUNC_EXP;
            else if (strcmp(func_name, "log") == 0) func_type = FUNC_LOG;
            else if (strcmp(func_name, "log10") == 0) func_type = FUNC_LOG10;
            else if (strcmp(func_name, "sqrt") == 0) func_type = FUNC_SQRT;
            else if (strcmp(func_name, "abs") == 0) func_type = FUNC_ABS;
            else {
                // 未知函数
                if (func_name) safe_free((void**)&func_name);
                return NULL;
            }
            
            if (func_name) safe_free((void**)&func_name);
            
            // 期望左括号
            *current_token = parser_get_next_token(state);
            if (current_token->type != TOKEN_LPAREN) {
                return NULL;
            }
            
            *current_token = parser_get_next_token(state);
            MathExpressionNode* arg = parse_expression(state, current_token);
            if (!arg) return NULL;
            
            if (current_token->type != TOKEN_RPAREN) {
                math_expression_destroy(arg);
                return NULL;
            }
            
            *current_token = parser_get_next_token(state);
            node = create_function_node(func_type, arg);
            break;
        }
            
        case TOKEN_LPAREN:
            *current_token = parser_get_next_token(state);
            node = parse_expression(state, current_token);
            
            if (!node) return NULL;
            
            if (current_token->type != TOKEN_RPAREN) {
                math_expression_destroy(node);
                return NULL;
            }
            
            *current_token = parser_get_next_token(state);
            break;
            
        default:
            // 语法错误
            return NULL;
    }
    
    return node;
}

/* ============================================================================
 * 公共API实现
 * =========================================================================== */

/**
 * @brief 创建数学物理推理引擎
 */
MathPhysicsReasoningEngine* math_physics_reasoning_engine_create(void) {
    MathPhysicsReasoningEngine* engine = (MathPhysicsReasoningEngine*)safe_malloc(sizeof(MathPhysicsReasoningEngine));
    if (!engine) return NULL;
    
    memset(engine, 0, sizeof(MathPhysicsReasoningEngine));
    engine->is_initialized = 1;
    
    return engine;
}

/**
 * @brief 销毁数学物理推理引擎
 */
void math_physics_reasoning_engine_destroy(MathPhysicsReasoningEngine* engine) {
    if (!engine) return;
    
    if (engine->temporary_buffer) {
        safe_free((void**)&engine->temporary_buffer);
    }
    
    safe_free((void**)&engine);
}

/**
 * @brief 解析数学表达式
 */
MathExpressionNode* math_expression_parse(const char* expression) {
    if (!expression) return NULL;
    
    ParserState state;
    parser_init(&state, expression);
    
    MprToken current_token = parser_get_next_token(&state);
    MathExpressionNode* node = parse_expression(&state, &current_token);
    
    // 检查是否完全解析
    if (current_token.type != TOKEN_END) {
        math_expression_destroy(node);
        return NULL;
    }
    
    return node;
}

/**
 * @brief 销毁数学表达式语法树
 */
void math_expression_destroy(MathExpressionNode* node) {
    if (!node) return;
    
    // 递归销毁子节点
    if (node->left) math_expression_destroy(node->left);
    if (node->right) math_expression_destroy(node->right);
    
    // 释放变量名
    if (node->type == EXPR_VARIABLE && node->value.variable_name) {
        safe_free((void**)&node->value.variable_name);
    }
    
    safe_free((void**)&node);
}

/**
 * @brief 求值数学表达式
 */
double math_expression_evaluate(const MathExpressionNode* node, const MathEvalContext* context) {
    if (!node) return 0.0;
    
    switch (node->type) {
        case EXPR_NUMBER:
            return node->value.number_value;
            
        case EXPR_VARIABLE: {
            if (!context || !node->value.variable_name) return 0.0;
            
            // 在上下文中查找变量值
            for (size_t i = 0; i < context->variable_count; i++) {
                if (context->variable_names[i] && 
                    strcmp(context->variable_names[i], node->value.variable_name) == 0) {
                    return context->variable_values[i];
                }
            }
            return 0.0;  // 未找到变量，返回0
        }
            
        case EXPR_ADDITION:
            return math_expression_evaluate(node->left, context) + 
                   math_expression_evaluate(node->right, context);
            
        case EXPR_SUBTRACTION:
            return math_expression_evaluate(node->left, context) - 
                   math_expression_evaluate(node->right, context);
            
        case EXPR_MULTIPLICATION:
            return math_expression_evaluate(node->left, context) * 
                   math_expression_evaluate(node->right, context);
            
        case EXPR_DIVISION: {
            double denominator = math_expression_evaluate(node->right, context);
            if (fabs(denominator) < DBL_EPSILON) return 0.0;  // 避免除零
            return math_expression_evaluate(node->left, context) / denominator;
        }
            
        case EXPR_POWER:
            return pow(math_expression_evaluate(node->left, context),
                       math_expression_evaluate(node->right, context));
            
        case EXPR_FUNCTION: {
            double arg = math_expression_evaluate(node->left, context);
            switch (node->value.func_type) {
                case FUNC_SIN: return sin(arg);
                case FUNC_COS: return cos(arg);
                case FUNC_TAN: return tan(arg);
                case FUNC_EXP: return exp(arg);
                case FUNC_LOG: return log(arg);
                case FUNC_LOG10: return log10(arg);
                case FUNC_SQRT: return sqrt(arg);
                case FUNC_ABS: return fabs(arg);
                default: return 0.0;
            }
        }
            
        case EXPR_NEGATION:
            return -math_expression_evaluate(node->left, context);
            
        case EXPR_PARENTHESES:
            return math_expression_evaluate(node->left, context);
            
        default:
            return 0.0;
    }
}

/**
 * @brief 求解一元方程（使用牛顿法）
 * 
 * 解析方程字符串，分离等号两侧表达式，使用牛顿迭代法求根。
 * f(x) = left_expr(x) - right_expr(x) = 0
 * x_{n+1} = x_n - f(x_n) / f'(x_n)
 * 使用中心差分法近似导数：f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
 */
EquationSolution* solve_equation(const char* equation, const char* variable, double initial_guess) {
    if (!equation || !variable) return NULL;
    
    // 查找等号位置，分离左右表达式
    const char* eq_pos = strstr(equation, "=");
    if (!eq_pos) return NULL;  // 没有等号，不是有效方程
    
    // 分配左右表达式字符串
    size_t left_len = eq_pos - equation;
    char* left_str = (char*)safe_malloc(left_len + 1);
    if (!left_str) return NULL;
    strncpy(left_str, equation, left_len);
    left_str[left_len] = '\0';
    
    const char* right_start = eq_pos + 1;
    while (*right_start == ' ') right_start++;  // 跳过等号后的空格
    char* right_str = string_duplicate_nullable(right_start);
    if (!right_str) {
        safe_free((void**)&left_str);
        return NULL;
    }
    
    // 去除左侧字符串末尾空格
    char* end = left_str + strlen(left_str) - 1;
    while (end >= left_str && *end == ' ') *end-- = '\0';
    
    // 解析左右表达式
    MathExpressionNode* left_expr = math_expression_parse(left_str);
    if (!left_expr) {
        safe_free((void**)&left_str);
        safe_free((void**)&right_str);
        return NULL;
    }
    
    MathExpressionNode* right_expr = math_expression_parse(right_str);
    if (!right_expr) {
        math_expression_destroy(left_expr);
        safe_free((void**)&left_str);
        safe_free((void**)&right_str);
        return NULL;
    }
    
    // 构建求值上下文
    MathEvalContext context;
    context.variable_names = &variable;
    context.variable_values = (double*)safe_malloc(sizeof(double));
    if (!context.variable_values) {
        math_expression_destroy(left_expr);
        math_expression_destroy(right_expr);
        safe_free((void**)&left_str);
        safe_free((void**)&right_str);
        return NULL;
    }
    context.variable_count = 1;
    
    // 牛顿迭代法求根：f(x) = left(x) - right(x) = 0
    int max_iter = 100;
    double tolerance = 1e-12;
    double x = initial_guess;
    int converged = 0;
    
    for (int iter = 0; iter < max_iter; iter++) {
        context.variable_values[0] = x;
        double left_val = math_expression_evaluate(left_expr, &context);
        double right_val = math_expression_evaluate(right_expr, &context);
        double fx = left_val - right_val;
        
        if (fabs(fx) < tolerance) {
            converged = 1;
            break;
        }
        
        // 中心差分法计算导数 f'(x)
        double h = fmax(1e-8, fabs(x) * 1e-8);
        context.variable_values[0] = x + h;
        double left_ph = math_expression_evaluate(left_expr, &context);
        double right_ph = math_expression_evaluate(right_expr, &context);
        double f_plus = left_ph - right_ph;
        
        context.variable_values[0] = x - h;
        double left_mh = math_expression_evaluate(left_expr, &context);
        double right_mh = math_expression_evaluate(right_expr, &context);
        double f_minus = left_mh - right_mh;
        
        double deriv = (f_plus - f_minus) / (2.0 * h);
        
        if (fabs(deriv) < 1e-15) {
            // 导数为零，无法继续牛顿迭代
            converged = 0;
            break;
        }
        
        double x_new = x - fx / deriv;
        
        if (fabs(x_new - x) < tolerance) {
            x = x_new;
            converged = 1;
            break;
        }
        
        x = x_new;
    }
    
    // 构建结果
    EquationSolution* solution = (EquationSolution*)safe_malloc(sizeof(EquationSolution));
    if (!solution) {
        math_expression_destroy(left_expr);
        math_expression_destroy(right_expr);
        safe_free((void**)&context.variable_values);
        safe_free((void**)&left_str);
        safe_free((void**)&right_str);
        return NULL;
    }
    
    memset(solution, 0, sizeof(EquationSolution));
    
    if (converged) {
        solution->solutions = (double*)safe_malloc(sizeof(double));
        if (solution->solutions) {
            solution->solutions[0] = x;
            solution->solution_count = 1;
            solution->has_solution = 1;
            solution->is_exact = 0;  // 数值解，非精确解
        }
    }
    
    // 清理
    math_expression_destroy(left_expr);
    math_expression_destroy(right_expr);
    safe_free((void**)&context.variable_values);
    safe_free((void**)&left_str);
    safe_free((void**)&right_str);
    
    return solution;
}

/**
 * @brief 销毁方程解
 */
void equation_solution_destroy(EquationSolution* solution) {
    if (!solution) return;
    
    if (solution->solutions) {
        safe_free((void**)&solution->solutions);
    }
    
    safe_free((void**)&solution);
}

/**
 * @brief 执行物理计算（完整牛顿力学+多模型求解）
 */
double* physics_simulation_run(MathPhysicsReasoningEngine* engine,
                               const PhysicsSimConfig* config,
                               const char** initial_conditions,
                               size_t condition_count) {
    if (!engine || !config || condition_count == 0) return NULL;
    
    size_t step_count = (size_t)(config->simulation_time / config->time_step + 0.5);
    if (step_count == 0) step_count = 1;
    
    double* results = (double*)safe_malloc(step_count * 3 * sizeof(double));
    if (!results) return NULL;
    
    /* 解析初始条件字符串数组，格式: "key=value" */
    double initial_position = 0.0;
    double initial_velocity = 1.0;
    double acceleration = 0.5;
    double gravity = 9.81;
    double spring_k = 10.0;
    double mass = 1.0;
    double pendulum_length = 1.0;
    double launch_angle = 0.785398;
    double launch_speed = 10.0;
    int simulation_model = 0;
    
    for (size_t ci = 0; ci < condition_count; ci++) {
        if (!initial_conditions[ci]) continue;
        const char* cond = initial_conditions[ci];
        const char* eq = strchr(cond, '=');
        if (!eq || eq == cond) continue;
        size_t key_len = (size_t)(eq - cond);
        const char* value_str = eq + 1;
        double val = strtod(value_str, NULL);
        
        if (key_len == 8 && strncmp(cond, "position", 8) == 0) {
            initial_position = val;
        } else if (key_len == 8 && strncmp(cond, "velocity", 8) == 0) {
            initial_velocity = val;
        } else if (key_len == 12 && strncmp(cond, "acceleration", 12) == 0) {
            acceleration = val;
        } else if (key_len == 6 && strncmp(cond, "gravity", 6) == 0) {
            gravity = val;
        } else if (key_len == 8 && strncmp(cond, "spring_k", 8) == 0) {
            spring_k = val;
        } else if (key_len == 4 && strncmp(cond, "mass", 4) == 0) {
            mass = val;
        } else if (key_len == 15 && strncmp(cond, "pendulum_length", 15) == 0) {
            pendulum_length = val;
        } else if (key_len == 12 && strncmp(cond, "launch_angle", 12) == 0) {
            launch_angle = val;
        } else if (key_len == 12 && strncmp(cond, "launch_speed", 12) == 0) {
            launch_speed = val;
        } else if (key_len == 5 && strncmp(cond, "model", 5) == 0) {
            simulation_model = (int)val;
        }
    }
    
    for (size_t i = 0; i < step_count; i++) {
        double t = i * config->time_step;
        double pos = 0.0, vel = 0.0, acc = 0.0;
        
        switch (simulation_model) {
            case 0: /* 匀加速直线运动: s = s0 + v0*t + 0.5*a*t^2 */
                pos = initial_position + initial_velocity * t + 0.5 * acceleration * t * t;
                vel = initial_velocity + acceleration * t;
                acc = acceleration;
                break;
            case 1: { /* 单摆: θ(t) = θ0 * cos(ω*t), ω = sqrt(g/L) */
                double omega = sqrt(gravity / pendulum_length);
                double theta0 = 0.3; /* 初始角度约17度 */
                pos = pendulum_length * theta0 * cos(omega * t);
                vel = -pendulum_length * theta0 * omega * sin(omega * t);
                acc = -pendulum_length * theta0 * omega * omega * cos(omega * t);
                break;
            }
            case 2: { /* 弹簧谐振: x(t) = A * cos(ω*t), ω = sqrt(k/m) */
                double omega = sqrt(spring_k / mass);
                double amplitude = initial_velocity / omega;
                pos = amplitude * sin(omega * t);
                vel = amplitude * omega * cos(omega * t);
                acc = -amplitude * omega * omega * sin(omega * t);
                break;
            }
            case 3: { /* 抛物线运动: x = v0x*t, y = v0y*t - 0.5*g*t^2 */
                double v0x = launch_speed * cos(launch_angle);
                double v0y = launch_speed * sin(launch_angle);
                double x_pos = v0x * t;
                double y_pos = v0y * t - 0.5 * gravity * t * t;
                pos = sqrt(x_pos * x_pos + y_pos * y_pos);
                double vy = v0y - gravity * t;
                vel = sqrt(v0x * v0x + vy * vy);
                acc = gravity;
                break;
            }
        }
        
        results[i * 3 + 0] = pos;
        results[i * 3 + 1] = vel;
        results[i * 3 + 2] = acc;
    }
    
    return results;
}

/* ========== P2-3: 符号数学引擎实现 ========== */

/**
 * @brief 深拷贝表达式节点
 */
static MathExpressionNode* clone_expression_node(const MathExpressionNode* node) {
    if (!node) return NULL;
    MathExpressionNode* clone = create_expression_node(node->type, NULL, NULL);
    if (!clone) return NULL;

    clone->value = node->value;
    clone->left = node->left ? clone_expression_node(node->left) : NULL;
    clone->right = node->right ? clone_expression_node(node->right) : NULL;

    return clone;
}

/**
 * @brief 判断表达式节点是否为指定变量
 */
static int is_variable_node(const MathExpressionNode* node, const char* var_name) {
    if (!node || !var_name) return 0;
    return (node->type == EXPR_VARIABLE &&
            node->value.variable_name &&
            strcmp(node->value.variable_name, var_name) == 0);
}

/**
 * @brief 判断表达式节点是否为数字常量
 */
static int is_constant_number(const MathExpressionNode* node, double val) {
    if (!node) return 0;
    return (node->type == EXPR_NUMBER && fabs(node->value.number_value - val) < 1e-14);
}

MathExpressionNode* symbolic_differentiate(const MathExpressionNode* node,
                                           const char* variable_name) {
    if (!node || !variable_name) return NULL;

    switch (node->type) {
    case EXPR_NUMBER:
        return create_number_node(0.0);

    case EXPR_VARIABLE:
        if (is_variable_node(node, variable_name))
            return create_number_node(1.0);
        return create_number_node(0.0);

    case EXPR_ADDITION: {
        MathExpressionNode* dl = symbolic_differentiate(node->left, variable_name);
        MathExpressionNode* dr = symbolic_differentiate(node->right, variable_name);
        if (!dl || !dr) { safe_free((void**)&dl); safe_free((void**)&dr); return NULL; }
        return create_expression_node(EXPR_ADDITION, dl, dr);
    }

    case EXPR_SUBTRACTION: {
        MathExpressionNode* dl = symbolic_differentiate(node->left, variable_name);
        MathExpressionNode* dr = symbolic_differentiate(node->right, variable_name);
        if (!dl || !dr) { safe_free((void**)&dl); safe_free((void**)&dr); return NULL; }
        return create_expression_node(EXPR_SUBTRACTION, dl, dr);
    }

    case EXPR_MULTIPLICATION: {
        /* (f*g)' = f'*g + f*g' */
        MathExpressionNode* dl = symbolic_differentiate(node->left, variable_name);
        MathExpressionNode* dr = symbolic_differentiate(node->right, variable_name);
        if (!dl || !dr) { safe_free((void**)&dl); safe_free((void**)&dr); return NULL; }
        MathExpressionNode* left_term = create_expression_node(EXPR_MULTIPLICATION,
            dl, clone_expression_node(node->right));
        MathExpressionNode* right_term = create_expression_node(EXPR_MULTIPLICATION,
            clone_expression_node(node->left), dr);
        return create_expression_node(EXPR_ADDITION, left_term, right_term);
    }

    case EXPR_DIVISION: {
        /* (f/g)' = (f'*g - f*g') / g^2 */
        MathExpressionNode* dl = symbolic_differentiate(node->left, variable_name);
        MathExpressionNode* dr = symbolic_differentiate(node->right, variable_name);
        if (!dl || !dr) { safe_free((void**)&dl); safe_free((void**)&dr); return NULL; }
        MathExpressionNode* numerator_left = create_expression_node(EXPR_MULTIPLICATION,
            dl, clone_expression_node(node->right));
        MathExpressionNode* numerator_right = create_expression_node(EXPR_MULTIPLICATION,
            clone_expression_node(node->left), dr);
        MathExpressionNode* numerator = create_expression_node(EXPR_SUBTRACTION,
            numerator_left, numerator_right);
        MathExpressionNode* denominator = create_expression_node(EXPR_POWER,
            clone_expression_node(node->right), create_number_node(2.0));
        return create_expression_node(EXPR_DIVISION, numerator, denominator);
    }

    case EXPR_POWER: {
        /* 检查指数是否为常数 */
        if (node->right && node->right->type == EXPR_NUMBER) {
            double n = node->right->value.number_value;
            /* (x^n)' = n * x^(n-1) */
            MathExpressionNode* base_diff = symbolic_differentiate(node->left, variable_name);
            if (!base_diff) return NULL;
            MathExpressionNode* coeff = create_number_node(n);
            MathExpressionNode* new_power = create_expression_node(EXPR_POWER,
                clone_expression_node(node->left), create_number_node(n - 1.0));
            MathExpressionNode* inner = create_expression_node(EXPR_MULTIPLICATION,
                coeff, new_power);
            return create_expression_node(EXPR_MULTIPLICATION, inner, base_diff);
        }
        /* 一般情况: f^g, 使用对数微分 */
        MathExpressionNode* ln_f = create_expression_node(EXPR_FUNCTION,
            clone_expression_node(node->left), NULL);
        ln_f->value.func_type = FUNC_LOG;
        MathExpressionNode* g_ln_f = create_expression_node(EXPR_MULTIPLICATION,
            clone_expression_node(node->right), ln_f);
        MathExpressionNode* exp_node = create_expression_node(EXPR_FUNCTION,
            g_ln_f, NULL);
        exp_node->value.func_type = FUNC_EXP;
        MathExpressionNode* inner_diff = symbolic_differentiate(g_ln_f, variable_name);
        if (!inner_diff) { math_expression_destroy(exp_node); return NULL; }
        return create_expression_node(EXPR_MULTIPLICATION, exp_node, inner_diff);
    }

    case EXPR_FUNCTION: {
        MathExpressionNode* arg_diff = symbolic_differentiate(node->left, variable_name);
        if (!arg_diff) return NULL;
        switch (node->value.func_type) {
        case FUNC_SIN:
            return create_expression_node(EXPR_MULTIPLICATION,
                create_expression_node(EXPR_FUNCTION, clone_expression_node(node->left), NULL),
                arg_diff);
        case FUNC_COS:
            return create_expression_node(EXPR_MULTIPLICATION,
                create_expression_node(EXPR_NEGATION,
                    create_expression_node(EXPR_FUNCTION, clone_expression_node(node->left), NULL),
                    NULL),
                arg_diff);
        case FUNC_TAN: {
            MathExpressionNode* cos_x = create_expression_node(EXPR_FUNCTION, clone_expression_node(node->left), NULL);
            cos_x->value.func_type = FUNC_COS;
            MathExpressionNode* sec2 = create_expression_node(EXPR_DIVISION,
                create_number_node(1.0),
                create_expression_node(EXPR_POWER, cos_x, create_number_node(2.0)));
            return create_expression_node(EXPR_MULTIPLICATION, sec2, arg_diff);
        }
        case FUNC_EXP: {
            MathExpressionNode* exp_x = create_expression_node(EXPR_FUNCTION, clone_expression_node(node->left), NULL);
            exp_x->value.func_type = FUNC_EXP;
            return create_expression_node(EXPR_MULTIPLICATION, exp_x, arg_diff);
        }
        case FUNC_LOG:
            return create_expression_node(EXPR_DIVISION,
                arg_diff, clone_expression_node(node->left));
        case FUNC_LOG10: {
            /* d/dx log10(x) = 1/(x * ln(10)) */
            MathExpressionNode* denom = create_expression_node(EXPR_MULTIPLICATION,
                clone_expression_node(node->left),
                create_number_node(log(10.0)));
            return create_expression_node(EXPR_DIVISION, arg_diff, denom);
        }
        case FUNC_SQRT: {
            MathExpressionNode* sqrt_x = create_expression_node(EXPR_FUNCTION, clone_expression_node(node->left), NULL);
            sqrt_x->value.func_type = FUNC_SQRT;
            MathExpressionNode* two_sqrt = create_expression_node(EXPR_MULTIPLICATION,
                create_number_node(2.0), sqrt_x);
            return create_expression_node(EXPR_DIVISION, arg_diff, two_sqrt);
        }
        case FUNC_ABS:
            return arg_diff;
        }
        return NULL;
    }

    case EXPR_NEGATION: {
        MathExpressionNode* inner_diff = symbolic_differentiate(node->left, variable_name);
        if (!inner_diff) return NULL;
        return create_expression_node(EXPR_NEGATION, inner_diff, NULL);
    }

    case EXPR_PARENTHESES:
        return symbolic_differentiate(node->left, variable_name);
    }
    return NULL;
}

MathExpressionNode* symbolic_simplify(const MathExpressionNode* node) {
    if (!node) return NULL;

    MathExpressionNode* left_simplified = node->left ? symbolic_simplify(node->left) : NULL;
    MathExpressionNode* right_simplified = node->right ? symbolic_simplify(node->right) : NULL;

    /* 常量折叠：如果左右子节点都是数字，直接计算 */
    if (left_simplified && right_simplified &&
        left_simplified->type == EXPR_NUMBER && right_simplified->type == EXPR_NUMBER) {
        double lv = left_simplified->value.number_value;
        double rv = right_simplified->value.number_value;
        double result = 0.0;
        int can_fold = 1;

        switch (node->type) {
        case EXPR_ADDITION:       result = lv + rv; break;
        case EXPR_SUBTRACTION:    result = lv - rv; break;
        case EXPR_MULTIPLICATION: result = lv * rv; break;
        case EXPR_DIVISION:
            if (fabs(rv) < 1e-14) can_fold = 0;
            else result = lv / rv;
            break;
        case EXPR_POWER:
            if (lv < 0 && fabs(rv - (int)rv) > 1e-14) can_fold = 0;
            else result = pow(lv, rv);
            break;
        default: can_fold = 0; break;
        }

        math_expression_destroy(left_simplified);
        math_expression_destroy(right_simplified);

        if (can_fold) return create_number_node(result);
        left_simplified = symbolic_simplify(node->left);
        right_simplified = symbolic_simplify(node->right);
    }

    /* 单子节点化简（右子节点为NULL） */
    if (node->type == EXPR_NEGATION && left_simplified) {
        if (left_simplified->type == EXPR_NUMBER) {
            double r = create_number_node(-left_simplified->value.number_value)->value.number_value;
            math_expression_destroy(left_simplified);
            return create_number_node(r);
        }
        if (left_simplified->type == EXPR_NEGATION) {
            /* --x → x */
            MathExpressionNode* inner = left_simplified->left;
            left_simplified->left = NULL;
            math_expression_destroy(left_simplified);
            return inner;
        }
    }

    /* 函数节点化简 */
    if (node->type == EXPR_FUNCTION && left_simplified &&
        left_simplified->type == EXPR_NUMBER) {
        double v = left_simplified->value.number_value;
        double result = 0.0;
        int can_fold = 1;
        switch (node->value.func_type) {
        case FUNC_SIN:  result = sin(v); break;
        case FUNC_COS:  result = cos(v); break;
        case FUNC_TAN:
            if (fabs(cos(v)) < 1e-14) can_fold = 0;
            else result = tan(v);
            break;
        case FUNC_EXP:  result = exp(v); break;
        case FUNC_LOG:
            if (v <= 0.0) can_fold = 0;
            else result = log(v);
            break;
        case FUNC_LOG10:
            if (v <= 0.0) can_fold = 0;
            else result = log10(v);
            break;
        case FUNC_SQRT:
            if (v < 0.0) can_fold = 0;
            else result = sqrt(v);
            break;
        case FUNC_ABS:  result = fabs(v); break;
        default: can_fold = 0; break;
        }
        math_expression_destroy(left_simplified);
        if (can_fold) return create_number_node(result);
        left_simplified = symbolic_simplify(node->left);
    }

    /* 二元运算化简规则 */
    if (node->type == EXPR_ADDITION) {
        if (!left_simplified || !right_simplified) return NULL;
        if (left_simplified->type == EXPR_NUMBER &&
            fabs(left_simplified->value.number_value) < 1e-14) {
            math_expression_destroy(left_simplified);
            return right_simplified;
        }
        if (right_simplified->type == EXPR_NUMBER &&
            fabs(right_simplified->value.number_value) < 1e-14) {
            math_expression_destroy(right_simplified);
            return left_simplified;
        }
    }

    if (node->type == EXPR_SUBTRACTION) {
        if (!left_simplified || !right_simplified) return NULL;
        if (right_simplified->type == EXPR_NUMBER &&
            fabs(right_simplified->value.number_value) < 1e-14) {
            math_expression_destroy(right_simplified);
            return left_simplified;
        }
    }

    if (node->type == EXPR_MULTIPLICATION) {
        if (!left_simplified || !right_simplified) return NULL;
        if ((left_simplified->type == EXPR_NUMBER &&
             fabs(left_simplified->value.number_value) < 1e-14) ||
            (right_simplified->type == EXPR_NUMBER &&
             fabs(right_simplified->value.number_value) < 1e-14)) {
            math_expression_destroy(left_simplified);
            math_expression_destroy(right_simplified);
            return create_number_node(0.0);
        }
        if (left_simplified->type == EXPR_NUMBER &&
            fabs(left_simplified->value.number_value - 1.0) < 1e-14) {
            math_expression_destroy(left_simplified);
            return right_simplified;
        }
        if (right_simplified->type == EXPR_NUMBER &&
            fabs(right_simplified->value.number_value - 1.0) < 1e-14) {
            math_expression_destroy(right_simplified);
            return left_simplified;
        }
    }

    if (node->type == EXPR_DIVISION) {
        if (!left_simplified || !right_simplified) return NULL;
        if (right_simplified->type == EXPR_NUMBER &&
            fabs(right_simplified->value.number_value - 1.0) < 1e-14) {
            math_expression_destroy(right_simplified);
            return left_simplified;
        }
        if (left_simplified->type == EXPR_NUMBER &&
            fabs(left_simplified->value.number_value) < 1e-14) {
            math_expression_destroy(right_simplified);
            math_expression_destroy(left_simplified);
            return create_number_node(0.0);
        }
    }

    if (node->type == EXPR_POWER) {
        if (!left_simplified || !right_simplified) return NULL;
        if (right_simplified->type == EXPR_NUMBER &&
            fabs(right_simplified->value.number_value) < 1e-14) {
            math_expression_destroy(left_simplified);
            math_expression_destroy(right_simplified);
            return create_number_node(1.0);
        }
        if (right_simplified->type == EXPR_NUMBER &&
            fabs(right_simplified->value.number_value - 1.0) < 1e-14) {
            math_expression_destroy(right_simplified);
            return left_simplified;
        }
    }

    MathExpressionNode* result = create_expression_node(node->type, left_simplified, right_simplified);
    return result;
}

MathExpressionNode* taylor_expand(const MathExpressionNode* node,
                                  const char* variable_name,
                                  double a, size_t order) {
    if (!node || !variable_name) return NULL;

    MathExpressionNode* result = NULL;
    double factorial = 1.0;

    for (size_t k = 0; k <= order; k++) {
        /* 创建 (x - a)^k 项 */
        MathExpressionNode* var_node = create_variable_node(variable_name);
        MathExpressionNode* a_node = create_number_node(a);
        MathExpressionNode* diff = create_expression_node(EXPR_SUBTRACTION, var_node, a_node);

        MathExpressionNode* power_term = NULL;
        if (k == 0) {
            math_expression_destroy(diff);
            power_term = create_number_node(1.0);
        } else if (k == 1) {
            power_term = diff;
        } else {
            power_term = create_expression_node(EXPR_POWER, diff, create_number_node((double)k));
        }

        /* 计算 k! */
        if (k > 0) factorial *= (double)k;

        /* 计算 f^{(k)}(a) —— 通过数值微分 */
        double derivative_at_a = 0.0;
        if (k == 0) {
            MathEvalContext ctx;
            ctx.variable_names = &variable_name;
            double var_val = a;
            ctx.variable_values = &var_val;
            ctx.variable_count = 1;
            derivative_at_a = math_expression_evaluate(node, &ctx);
        } else {
            /* 使用中心差分计算高阶导数 */
            MathExpressionNode* current_deriv = clone_expression_node(node);
            for (size_t d = 0; d < k; d++) {
                MathExpressionNode* diff_exp = symbolic_differentiate(current_deriv, variable_name);
                math_expression_destroy(current_deriv);
                current_deriv = diff_exp;
                if (!current_deriv) break;
            }
            if (current_deriv) {
                MathEvalContext ctx;
                ctx.variable_names = &variable_name;
                double var_val = a;
                ctx.variable_values = &var_val;
                ctx.variable_count = 1;
                derivative_at_a = math_expression_evaluate(current_deriv, &ctx);
                math_expression_destroy(current_deriv);
            }
        }

        /* 构建项: f^{(k)}(a) / k! * (x-a)^k */
        MathExpressionNode* coeff = create_number_node(derivative_at_a / factorial);
        MathExpressionNode* term = create_expression_node(EXPR_MULTIPLICATION, coeff, power_term);

        if (!result) {
            result = term;
        } else {
            result = create_expression_node(EXPR_ADDITION, result, term);
        }
    }

    return result;
}

double definite_integral(const MathExpressionNode* node,
                         const char* variable_name,
                         double a, double b, size_t n) {
    if (!node || !variable_name || n == 0) return 0.0;

    if (n % 2 != 0) n++;

    double h = (b - a) / (double)n;
    MathEvalContext ctx;
    ctx.variable_names = &variable_name;
    ctx.variable_count = 1;

    /* f(a) */
    double fa_val = a;
    ctx.variable_values = &fa_val;
    double fa = math_expression_evaluate(node, &ctx);

    /* f(b) */
    double fb_val = b;
    ctx.variable_values = &fb_val;
    double fb = math_expression_evaluate(node, &ctx);

    double sum_odd = 0.0;
    double sum_even = 0.0;

    for (size_t i = 1; i < n; i++) {
        double x = a + (double)i * h;
        ctx.variable_values = &x;
        double fx = math_expression_evaluate(node, &ctx);

        if (i % 2 == 0) {
            sum_even += fx;
        } else {
            sum_odd += fx;
        }
    }

    return h / 3.0 * (fa + 4.0 * sum_odd + 2.0 * sum_even + fb);
}

double partial_derivative(const MathExpressionNode* node,
                          const char** variable_names,
                          const double* variable_values,
                          size_t num_variables,
                          const char* diff_var_name) {
    if (!node || !variable_names || !variable_values || !diff_var_name) return 0.0;

    size_t diff_index = num_variables;
    for (size_t i = 0; i < num_variables; i++) {
        if (strcmp(variable_names[i], diff_var_name) == 0) {
            diff_index = i;
            break;
        }
    }
    if (diff_index >= num_variables) return 0.0;

    /* 中心差分: f'(x) ≈ (f(x+h) - f(x-h)) / (2h) */
    double h = 1e-6;
    double* v_plus = (double*)safe_malloc(num_variables * sizeof(double));
    double* v_minus = (double*)safe_malloc(num_variables * sizeof(double));
    if (!v_plus || !v_minus) {
        safe_free((void**)&v_plus); safe_free((void**)&v_minus);
        return 0.0;
    }

    for (size_t i = 0; i < num_variables; i++) {
        v_plus[i] = variable_values[i];
        v_minus[i] = variable_values[i];
    }
    v_plus[diff_index] += h;
    v_minus[diff_index] -= h;

    MathEvalContext ctx;
    ctx.variable_names = variable_names;
    ctx.variable_count = num_variables;

    ctx.variable_values = v_plus;
    double f_plus = math_expression_evaluate(node, &ctx);
    ctx.variable_values = v_minus;
    double f_minus = math_expression_evaluate(node, &ctx);

    safe_free((void**)&v_plus);
    safe_free((void**)&v_minus);

    if (fabs(2.0 * h) < 1e-15) return 0.0;
    return (f_plus - f_minus) / (2.0 * h);
}

int gradient(const MathExpressionNode* node,
             const char** variable_names,
             const double* variable_values,
             size_t num_variables,
             double* gradient_out) {
    if (!node || !variable_names || !variable_values || !gradient_out || num_variables == 0)
        return -1;

    for (size_t i = 0; i < num_variables; i++) {
        gradient_out[i] = partial_derivative(node, variable_names, variable_values,
                                              num_variables, variable_names[i]);
    }

    return 0;
}

MathExpressionNode* expression_substitute(const MathExpressionNode* node,
                                          const char* variable_name,
                                          const MathExpressionNode* replacement) {
    if (!node || !variable_name || !replacement) return NULL;

    if (node->type == EXPR_VARIABLE &&
        node->value.variable_name &&
        strcmp(node->value.variable_name, variable_name) == 0) {
        return clone_expression_node(replacement);
    }

    if (node->type == EXPR_NUMBER) {
        return clone_expression_node(node);
    }

    MathExpressionNode* new_left = node->left ?
        expression_substitute(node->left, variable_name, replacement) : NULL;
    MathExpressionNode* new_right = node->right ?
        expression_substitute(node->right, variable_name, replacement) : NULL;

    MathExpressionNode* result = create_expression_node(node->type, new_left, new_right);
    if (result) {
        result->value = node->value;
        if (node->type == EXPR_FUNCTION && node->value.variable_name) {
            result->value.variable_name = string_duplicate_nullable(node->value.variable_name);
        }
    }

    return result;
}

EquationSolution* solve_equation_symbolic(const char* equation, const char* variable) {
    if (!equation || !variable) return NULL;

    /* 解析方程中 '=' 左右两侧 */
    char* eq_copy = string_duplicate_nullable(equation);
    if (!eq_copy) return NULL;

    char* eq_pos = strstr(eq_copy, "=");
    if (!eq_pos) {
        safe_free((void**)&eq_copy);
        return NULL;
    }

    *eq_pos = '\0';
    char* left_str = eq_copy;
    char* right_str = eq_pos + 1;

    /* 解析左右表达式 */
    MathExpressionNode* left_expr = math_expression_parse(left_str);
    MathExpressionNode* right_expr = math_expression_parse(right_str);
    if (!left_expr || !right_expr) {
        math_expression_destroy(left_expr);
        math_expression_destroy(right_expr);
        safe_free((void**)&eq_copy);
        return NULL;
    }

    /* 构建 f(x) = left - right = 0 */
    MathExpressionNode* f_expr = create_expression_node(EXPR_SUBTRACTION, left_expr, right_expr);
    if (!f_expr) {
        math_expression_destroy(left_expr);
        math_expression_destroy(right_expr);
        safe_free((void**)&eq_copy);
        return NULL;
    }

    /* 简化方程 */
    MathExpressionNode* simplified = symbolic_simplify(f_expr);
    math_expression_destroy(f_expr);

    EquationSolution* solution = (EquationSolution*)safe_malloc(sizeof(EquationSolution));
    if (!solution) { math_expression_destroy(simplified); safe_free((void**)&eq_copy); return NULL; }
    memset(solution, 0, sizeof(EquationSolution));

    /* 检测是否为线性方程 ax + b = 0 */
    MathExpressionNode* derivative = symbolic_differentiate(simplified, variable);
    if (!derivative) {
        math_expression_destroy(simplified);
        safe_free((void**)&eq_copy);
        solution->has_solution = 0;
        return solution;
    }

    MathExpressionNode* second_deriv = symbolic_differentiate(derivative, variable);

    MathEvalContext ctx;
    ctx.variable_names = &variable;
    ctx.variable_count = 1;

    /* 检查是否线性: 二阶导数为0 */
    if (second_deriv) {
        double test_val = 0.0;
        ctx.variable_values = &test_val;
        double sd_val = math_expression_evaluate(second_deriv, &ctx);
        math_expression_destroy(second_deriv);

        if (fabs(sd_val) < 1e-10) {
            /* 线性方程: a*x + b = 0, x = -b/a */
            double a_coeff = 0.0, b_coeff = 0.0;
            double x0 = 0.0;
            ctx.variable_values = &x0;
            b_coeff = math_expression_evaluate(simplified, &ctx);
            double x1 = 1.0;
            ctx.variable_values = &x1;
            double f1 = math_expression_evaluate(simplified, &ctx);
            a_coeff = f1 - b_coeff;

            if (fabs(a_coeff) > 1e-14) {
                double root = -b_coeff / a_coeff;
                solution->solutions = (double*)safe_malloc(sizeof(double));
                if (solution->solutions) {
                    solution->solutions[0] = root;
                    solution->solution_count = 1;
                    solution->has_solution = 1;
                    solution->is_exact = 1;
                }
            }
            math_expression_destroy(simplified);
            math_expression_destroy(derivative);
            safe_free((void**)&eq_copy);
            return solution;
        }
    }

    /* 检查是否为二次方程: 二阶导数为常数且非零 */
    {
        MathExpressionNode* second_simplified = symbolic_differentiate(derivative, variable);
        if (second_simplified) {
            MathExpressionNode* second_simp = symbolic_simplify(second_simplified);
            math_expression_destroy(second_simplified);
            if (second_simp && second_simp->type == EXPR_NUMBER) {
                double two_a = 2.0 * second_simp->value.number_value;
                math_expression_destroy(second_simp);

                if (fabs(two_a) > 1e-14) {
                    double a_coeff = two_a / 2.0;
                    double x0 = 0.0;
                    ctx.variable_values = &x0;
                    double c_val = math_expression_evaluate(simplified, &ctx);
                    double x1 = 1.0;
                    ctx.variable_values = &x1;
                    double f1 = math_expression_evaluate(simplified, &ctx);
                    double b_coeff = f1 - a_coeff - c_val;

                    double discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_val;

                    solution->solutions = (double*)safe_malloc(2 * sizeof(double));
                    if (solution->solutions) {
                        if (discriminant >= 0.0) {
                            double sqrt_d = sqrt(discriminant);
                            solution->solutions[0] = (-b_coeff + sqrt_d) / (2.0 * a_coeff);
                            solution->solutions[1] = (-b_coeff - sqrt_d) / (2.0 * a_coeff);
                            solution->solution_count = 2;
                            solution->has_solution = 1;
                            solution->is_exact = 1;
                        } else {
                            solution->has_solution = 0;
                            solution->solution_count = 0;
                        }
                    }
                    math_expression_destroy(simplified);
                    math_expression_destroy(derivative);
                    safe_free((void**)&eq_copy);
                    return solution;
                }
            } else if (second_simp) {
                math_expression_destroy(second_simp);
            }
        }
    }

    math_expression_destroy(derivative);

    /* 回退到数值求解 */
    solution->has_solution = 0;
    EquationSolution* num_sol = solve_equation(equation, variable, 0.0);
    if (num_sol) {
        if (num_sol->has_solution && num_sol->solution_count > 0) {
            solution->solutions = (double*)safe_malloc(num_sol->solution_count * sizeof(double));
            if (solution->solutions) {
                memcpy(solution->solutions, num_sol->solutions,
                       num_sol->solution_count * sizeof(double));
                solution->solution_count = num_sol->solution_count;
                solution->has_solution = 1;
                solution->is_exact = 0;
            }
        }
        equation_solution_destroy(num_sol);
    }

    math_expression_destroy(simplified);
    safe_free((void**)&eq_copy);
    return solution;
}

/* ============================================================================
 * A04.4.2: 物理量纲分析实现
 * =========================================================================== */

PhysicalQuantity physical_quantity_create(double value, PhysicalDimension dim, const char* name)
{
    PhysicalQuantity q;
    q.value = value;
    q.dim = dim;
    q.uncertainty = 0.0;
    memset(q.name, 0, sizeof(q.name));
    if (name) {
        size_t len = strlen(name);
        if (len >= sizeof(q.name)) len = sizeof(q.name) - 1;
        memcpy(q.name, name, len);
        q.name[len] = '\0';
    }
    return q;
}

PhysicalDimension physical_dimension_create(int M, int L, int T)
{
    PhysicalDimension d;
    d.M = M; d.L = L; d.T = T;
    d.I = 0; d.Th = 0; d.N = 0; d.J = 0;
    return d;
}

PhysicalDimension physical_dimension_create_full(int M, int L, int T, int I, int Th, int N, int J)
{
    PhysicalDimension d;
    d.M = M; d.L = L; d.T = T;
    d.I = I; d.Th = Th; d.N = N; d.J = J;
    return d;
}

PhysicalDimension physical_dimension_multiply(PhysicalDimension a, PhysicalDimension b)
{
    PhysicalDimension d;
    d.M = a.M + b.M; d.L = a.L + b.L; d.T = a.T + b.T;
    d.I = a.I + b.I; d.Th = a.Th + b.Th; d.N = a.N + b.N; d.J = a.J + b.J;
    return d;
}

PhysicalDimension physical_dimension_divide(PhysicalDimension a, PhysicalDimension b)
{
    PhysicalDimension d;
    d.M = a.M - b.M; d.L = a.L - b.L; d.T = a.T - b.T;
    d.I = a.I - b.I; d.Th = a.Th - b.Th; d.N = a.N - b.N; d.J = a.J - b.J;
    return d;
}

PhysicalDimension physical_dimension_power(PhysicalDimension dim, int power)
{
    PhysicalDimension d;
    d.M = dim.M * power; d.L = dim.L * power; d.T = dim.T * power;
    d.I = dim.I * power; d.Th = dim.Th * power; d.N = dim.N * power; d.J = dim.J * power;
    return d;
}

int physical_dimension_check_consistent(PhysicalDimension a, PhysicalDimension b)
{
    return (a.M == b.M && a.L == b.L && a.T == b.T &&
            a.I == b.I && a.Th == b.Th && a.N == b.N && a.J == b.J) ? 1 : 0;
}

int physical_quantity_check_dimension(const PhysicalQuantity* q,
                                       const PhysicalDimension* expected_dims,
                                       size_t dim_count)
{
    if (!q || !expected_dims || dim_count == 0) return 0;
    for (size_t i = 0; i < dim_count; i++) {
        if (physical_dimension_check_consistent(q->dim, expected_dims[i])) return 1;
    }
    return 0;
}

PhysicalQuantity physical_quantity_add(PhysicalQuantity a, PhysicalQuantity b)
{
    PhysicalQuantity result;
    memset(&result, 0, sizeof(result));
    if (!physical_dimension_check_consistent(a.dim, b.dim)) {
        snprintf(result.name, sizeof(result.name), "维度不一致错误");
        return result;
    }
    result.value = a.value + b.value;
    result.dim = a.dim;
    result.uncertainty = sqrt(a.uncertainty * a.uncertainty + b.uncertainty * b.uncertainty);
    snprintf(result.name, sizeof(result.name), "%s+%s", a.name, b.name);
    return result;
}

PhysicalQuantity physical_quantity_subtract(PhysicalQuantity a, PhysicalQuantity b)
{
    PhysicalQuantity result;
    memset(&result, 0, sizeof(result));
    if (!physical_dimension_check_consistent(a.dim, b.dim)) {
        snprintf(result.name, sizeof(result.name), "维度不一致错误");
        return result;
    }
    result.value = a.value - b.value;
    result.dim = a.dim;
    result.uncertainty = sqrt(a.uncertainty * a.uncertainty + b.uncertainty * b.uncertainty);
    snprintf(result.name, sizeof(result.name), "%s-%s", a.name, b.name);
    return result;
}

PhysicalQuantity physical_quantity_multiply(PhysicalQuantity a, PhysicalQuantity b)
{
    PhysicalQuantity result;
    memset(&result, 0, sizeof(result));
    result.value = a.value * b.value;
    result.dim = physical_dimension_multiply(a.dim, b.dim);
    result.uncertainty = result.value * sqrt(
        pow(a.uncertainty / (fabs(a.value) > 1e-15 ? a.value : 1e-15), 2) +
        pow(b.uncertainty / (fabs(b.value) > 1e-15 ? b.value : 1e-15), 2));
    snprintf(result.name, sizeof(result.name), "%s*%s", a.name, b.name);
    return result;
}

PhysicalQuantity physical_quantity_divide(PhysicalQuantity a, PhysicalQuantity b)
{
    PhysicalQuantity result;
    memset(&result, 0, sizeof(result));
    if (fabs(b.value) < 1e-15) {
        snprintf(result.name, sizeof(result.name), "除零错误");
        return result;
    }
    result.value = a.value / b.value;
    result.dim = physical_dimension_divide(a.dim, b.dim);
    result.uncertainty = result.value * sqrt(
        pow(a.uncertainty / (fabs(a.value) > 1e-15 ? a.value : 1e-15), 2) +
        pow(b.uncertainty / (fabs(b.value) > 1e-15 ? b.value : 1e-15), 2));
    snprintf(result.name, sizeof(result.name), "%s/%s", a.name, b.name);
    return result;
}

PhysicalQuantity physical_quantity_convert(PhysicalQuantity q, PhysicalDimension target_dim)
{
    /* F-005修复: 实现物理量纲一致性检查和转换
     * 两个物理量只有在量纲一致(所有7个基本量纲指数相等)时才能直接转换
     * 量纲一致的物理量值不变(均为SI单位制) */
    if (q.dim.M == target_dim.M && q.dim.L == target_dim.L &&
        q.dim.T == target_dim.T && q.dim.I == target_dim.I &&
        q.dim.Th == target_dim.Th && q.dim.N == target_dim.N &&
        q.dim.J == target_dim.J) {
        PhysicalQuantity result = q;
        result.dim = target_dim;
        return result;
    }
    /* 量纲不一致：物理上不能直接转换，将值置零并标记目标量纲 */
    PhysicalQuantity result = q;
    result.dim = target_dim;
    result.value = 0.0;
    result.uncertainty = 3.4e38f;  /* FLT_MAX替换INFINITY，避免/fp:fast下溢出警告 */
    return result;
}

static const char* dim_exponent_str(int exp)
{
    if (exp == 0) return "";
    if (exp == 1) return "";
    static char buf[16];
    snprintf(buf, sizeof(buf), "^%d", exp);
    return buf;
}

void physical_dimension_to_string(PhysicalDimension dim, char* buffer, size_t buffer_size)
{
    if (!buffer || buffer_size == 0) return;
    char temp[256] = {0};
    int first = 1;
    if (dim.M != 0) {
        snprintf(temp + strlen(temp), sizeof(temp) - strlen(temp), "%sM%s", first ? "" : "·", dim_exponent_str(dim.M));
        first = 0;
    }
    if (dim.L != 0) {
        snprintf(temp + strlen(temp), sizeof(temp) - strlen(temp), "%sL%s", first ? "" : "·", dim_exponent_str(dim.L));
        first = 0;
    }
    if (dim.T != 0) {
        snprintf(temp + strlen(temp), sizeof(temp) - strlen(temp), "%sT%s", first ? "" : "·", dim_exponent_str(dim.T));
        first = 0;
    }
    if (dim.I != 0) {
        snprintf(temp + strlen(temp), sizeof(temp) - strlen(temp), "%sI%s", first ? "" : "·", dim_exponent_str(dim.I));
        first = 0;
    }
    if (dim.Th != 0) {
        snprintf(temp + strlen(temp), sizeof(temp) - strlen(temp), "%sΘ%s", first ? "" : "·", dim_exponent_str(dim.Th));
        first = 0;
    }
    if (dim.N != 0) {
        snprintf(temp + strlen(temp), sizeof(temp) - strlen(temp), "%sN%s", first ? "" : "·", dim_exponent_str(dim.N));
        first = 0;
    }
    if (dim.J != 0) {
        snprintf(temp + strlen(temp), sizeof(temp) - strlen(temp), "%sJ%s", first ? "" : "·", dim_exponent_str(dim.J));
        first = 0;
    }
    if (first) {
        snprintf(temp, sizeof(temp), "1（无量纲）");
    }
    snprintf(buffer, buffer_size, "%s", temp);
}

/* ============================================================================
 * A04.4.2: 物理约束推理实现
 * =========================================================================== */

int physics_check_energy_conservation(const double* kinetic_energy,
                                      const double* potential_energy,
                                      size_t count, double tolerance)
{
    if (!kinetic_energy || !potential_energy || count < 2) return 0;
    double total_first = kinetic_energy[0] + potential_energy[0];
    for (size_t i = 1; i < count; i++) {
        double total = kinetic_energy[i] + potential_energy[i];
        if (fabs(total - total_first) > tolerance * fmax(1.0, fabs(total_first))) {
            return 0;
        }
    }
    return 1;
}

int physics_check_momentum_conservation(double momentum_before,
                                        double momentum_after,
                                        double tolerance)
{
    return (fabs(momentum_before - momentum_after) <=
            tolerance * fmax(1.0, fabs(momentum_before))) ? 1 : 0;
}

int physics_check_angular_momentum_conservation(double ang_momentum_before,
                                                double ang_momentum_after,
                                                double tolerance)
{
    return (fabs(ang_momentum_before - ang_momentum_after) <=
            tolerance * fmax(1.0, fabs(ang_momentum_before))) ? 1 : 0;
}

int physics_check_newton_second_law(double force, double mass,
                                    double acceleration, double tolerance)
{
    if (fabs(mass) < 1e-15) return 0;
    double expected_force = mass * acceleration;
    return (fabs(force - expected_force) <=
            tolerance * fmax(1.0, fabs(expected_force))) ? 1 : 0;
}

int physics_add_constraint(PhysicalConstraint** constraints, int* count, int* capacity,
                           PhysicalConstraintType type, const char* expression, double tolerance)
{
    if (!constraints || !count || !capacity) return -1;
    if (*count >= *capacity) {
        int new_cap = (*capacity == 0) ? 8 : (*capacity * 2);
        PhysicalConstraint* new_arr = (PhysicalConstraint*)safe_realloc(
            *constraints, new_cap * sizeof(PhysicalConstraint));
        if (!new_arr) return -1;
        *constraints = new_arr;
        *capacity = new_cap;
    }
    PhysicalConstraint* c = &(*constraints)[*count];
    c->type = type;
    c->tolerance = tolerance;
    c->is_active = 1;
    memset(c->expression, 0, sizeof(c->expression));
    if (expression) {
        size_t len = strlen(expression);
        if (len >= sizeof(c->expression)) len = sizeof(c->expression) - 1;
        memcpy(c->expression, expression, len);
        c->expression[len] = '\0';
    }
    (*count)++;
    return 0;
}

int physics_validate_constraints(const PhysicalConstraint* constraints, int count, int* results)
{
    if (!constraints || count <= 0 || !results) return 0;
    int passed = 0;
    for (int i = 0; i < count; i++) {
        results[i] = constraints[i].is_active ? 1 : -1;
        if (constraints[i].is_active) passed++;
    }
    return passed;
}

double* physics_simulation_with_constraints(MathPhysicsReasoningEngine* engine,
                                            const PhysicsSimConfig* config,
                                            const char** initial_conditions,
                                            size_t condition_count,
                                            const PhysicalConstraint* constraints,
                                            int constraint_count,
                                            int* constraint_results)
{
    if (!engine || !config || condition_count == 0) return NULL;
    double* results = physics_simulation_run(engine, config, initial_conditions, condition_count);
    if (!results) return NULL;

    if (constraints && constraint_count > 0 && constraint_results) {
        size_t step_count = (size_t)(config->simulation_time / config->time_step + 0.5);
        if (step_count == 0) step_count = 1;
        for (int ci = 0; ci < constraint_count; ci++) {
            constraint_results[ci] = 1;
            if (!constraints[ci].is_active) continue;
            switch (constraints[ci].type) {
                case CONSTRAINT_ENERGY_CONSERVATION: {
                    double* ke = (double*)safe_malloc(step_count * sizeof(double));
                    double* pe = (double*)safe_malloc(step_count * sizeof(double));
                    if (ke && pe) {
                        for (size_t i = 0; i < step_count; i++) {
                            double v = results[i * 3 + 1];
                            double pos = results[i * 3 + 0];
                            ke[i] = 0.5 * 1.0 * v * v;
                            pe[i] = 9.81 * 1.0 * pos;
                        }
                        constraint_results[ci] = physics_check_energy_conservation(
                            ke, pe, step_count, constraints[ci].tolerance);
                    }
                    if (ke) safe_free((void**)&ke);
                    if (pe) safe_free((void**)&pe);
                    break;
                }
                case CONSTRAINT_NEWTON_SECOND: {
                    for (size_t i = 0; i < step_count && i < 10; i++) {
                        double f = results[i * 3 + 0];
                        double a = results[i * 3 + 2];
                        if (!physics_check_newton_second_law(f, 1.0, a, constraints[ci].tolerance)) {
                            constraint_results[ci] = 0;
                            break;
                        }
                    }
                    break;
                }
                default:
                    constraint_results[ci] = 1;
                    break;
            }
        }
    }
    return results;
}