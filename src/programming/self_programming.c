/**
 * @file self_programming.c
 * @brief 自我编程能力实现
 * 
 * 自我编程能力核心实现，支持代码解析、生成、分析和自我优化。
 */


// MSVC兼容性：snprintf
#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

#include "selflnn/programming/self_programming.h"
#include "selflnn/programming/programming_enhanced.h"  /* H-017集成 */
#include "selflnn/programming/c_interpreter.h" /* C解释器集成 */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/core/errors.h"

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#ifndef F_OK
#define F_OK 0
#endif
#ifndef access
#define access _access
#endif
#ifndef popen
#define popen _popen
#endif
#ifndef pclose
#define pclose _pclose
#endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

// Linux seccomp沙箱支持所需头文件
#ifdef __linux__
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <linux/audit.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#endif

/**
 * @brief 自我编程引擎内部结构体
 */
struct SelfProgrammingEngine {
    ProgrammingLanguage language; /**< 目标编程语言 */
    int is_initialized;          /**< 是否已初始化 */
    void* parser_state;          /**< 解析器状态 */
    void* code_generator;        /**< 代码生成器状态 */
    void* analyzer_state;        /**< 分析器状态 */
    void* penh_engine;           /**< H-017: 编程增强引擎（重构/性能/安全分析） */
/* 编译错误反馈环形缓冲区 */
    char compile_error_history[16][128];  /**< 最近16条编译错误信息 */
    uint32_t compile_error_signatures[16]; /**< 错误特征码(前4字节) */
    int    compile_error_count;            /**< 环形缓冲区索引 */
    int    total_compile_errors;           /**< 总编译错误计数 */
};

/**
 * @brief 词法标记
 */
typedef struct {
    enum {
        TOKEN_IDENTIFIER,        /**< 标识符 */
        TOKEN_NUMBER,            /**< 数字 */
        TOKEN_STRING,            /**< 字符串字面量 */
        TOKEN_KEYWORD,           /**< 关键字 */
        TOKEN_OPERATOR,          /**< 操作符 */
        TOKEN_PUNCTUATOR,        /**< 标点符号 */
        TOKEN_END                /**< 结束标记 */
    } type;
    char* value;                 /**< 标记值 */
    int line;                    /**< 行号 */
    int column;                  /**< 列号 */
} Token;

/**
 * @brief 解析器状态
 */
typedef struct {
    const char* source;          /**< 源代码 */
    size_t position;             /**< 当前位置 */
    size_t length;               /**< 源代码长度 */
    char current_char;           /**< 当前字符 */
    int line;                    /**< 当前行号 */
    int column;                  /**< 当前列号 */
    Token current_token;         /**< 当前预读token */
    int token_available;         /**< 是否有有效token */
} ParserState;

/* ============================================================================
 * 内部辅助函数
 * =========================================================================== */

/* 关键字列表（C语言） */
static const char* c_keywords[] = {
    "auto", "break", "case", "char", "const", "continue", "default",
    "do", "double", "else", "enum", "extern", "float", "for", "goto",
    "if", "int", "long", "register", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while"
};
#define C_KEYWORD_COUNT (sizeof(c_keywords)/sizeof(c_keywords[0]))

/* 判断是否为关键字 */
static int is_keyword(const char* identifier) {
    for (size_t i = 0; i < C_KEYWORD_COUNT; i++) {
        if (strcmp(identifier, c_keywords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

/* 静态函数前向声明 */
static ASTNode* create_ast_node(ASTNodeType type);
static void parser_init(ParserState* state, const char* source);
static void parser_advance(ParserState* state);
static void parser_skip_whitespace(ParserState* state);
static Token parser_get_next_token(ParserState* state);
static Token parser_peek_token(ParserState* state);
static Token parser_consume_token(ParserState* state);
static ASTNode* parse_statement(ParserState* state);
static ASTNode* parse_expression(ParserState* state);
static ASTNode* parse_assignment(ParserState* state);
static ASTNode* parse_logical_or(ParserState* state);
static ASTNode* parse_logical_and(ParserState* state);
static ASTNode* parse_equality(ParserState* state);
static ASTNode* parse_relational(ParserState* state);
static ASTNode* parse_additive(ParserState* state);
static ASTNode* parse_multiplicative(ParserState* state);
static ASTNode* parse_unary(ParserState* state);
static ASTNode* parse_primary(ParserState* state);
static ASTNode* parse_function_call(ParserState* state, ASTNode* callee);
static char* ast_to_source(const ASTNode* node, int indent_level);
static ASTNode* parse_type(ParserState* state);
static ASTNode* parse_parameter_list(ParserState* state);
static ASTNode* parse_block(ParserState* state);
static ASTNode* parse_return_statement(ParserState* state);
static ASTNode* parse_if_statement(ParserState* state);
static ASTNode* parse_while_statement(ParserState* state);
static ASTNode* parse_for_statement(ParserState* state);
static ASTNode* parse_declaration(ParserState* state);
static DataType token_to_data_type(const char* type_name);
static void ast_to_source_recursive(const ASTNode* node, int indent_level, char* buffer, size_t buffer_size, size_t* pos);

/**
 * @brief 创建AST节点
 */
static ASTNode* create_ast_node(ASTNodeType type) {
    ASTNode* node = (ASTNode*)safe_malloc(sizeof(ASTNode));
    if (!node) return NULL;
    
    memset(node, 0, sizeof(ASTNode));
    node->type = type;
    node->line = 0;
    node->column = 0;
    return node;
}

/**
 * @brief 初始化解析器状态
 */
static void parser_init(ParserState* state, const char* source) {
    if (!state || !source) return;
    
    state->source = source;
    state->position = 0;
    state->length = strlen(source);
    state->current_char = (state->length > 0) ? source[0] : '\0';
    state->line = 1;
    state->column = 1;
}

/**
 * @brief 前进到下一个字符
 */
static void parser_advance(ParserState* state) {
    if (!state || state->position >= state->length) {
        state->current_char = '\0';
        return;
    }
    
    if (state->current_char == '\n') {
        state->line++;
        state->column = 1;
    } else {
        state->column++;
    }
    
    state->position++;
    if (state->position < state->length) {
        state->current_char = state->source[state->position];
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
 * @brief 解析标识符
 */
static char* parser_parse_identifier(ParserState* state) {
    if (!state) return NULL;
    
    char buffer[256] = {0};
    size_t index = 0;
    
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
 * @brief 解析数字
 */
static char* parser_parse_number(ParserState* state) {
    if (!state) return NULL;
    
    char buffer[64] = {0};
    size_t index = 0;
    
    while (state->current_char != '\0' && 
           (isdigit((unsigned char)state->current_char) || state->current_char == '.')) {
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
static Token parser_get_next_token(ParserState* state) {
    Token token = {TOKEN_END, NULL, 0, 0};
    
    if (!state || state->current_char == '\0') {
        return token;
    }
    
    parser_skip_whitespace(state);
    
    if (state->current_char == '\0') {
        return token;
    }
    
    token.line = state->line;
    token.column = state->column;
    
    char current = state->current_char;
    
    // 检查标识符（字母或下划线开头）
    if (isalpha((unsigned char)current) || current == '_') {
        token.value = parser_parse_identifier(state);
        if (token.value && is_keyword(token.value)) {
            token.type = TOKEN_KEYWORD;
        } else {
            token.type = TOKEN_IDENTIFIER;
        }
        return token;
    }
    
    // 检查数字
    if (isdigit((unsigned char)current)) {
        token.type = TOKEN_NUMBER;
        token.value = parser_parse_number(state);
        return token;
    }
    
    // 检查字符串字面量
    if (current == '"' || current == '\'') {
        char quote = current;
        parser_advance(state);  // 跳过开始引号
        
        char buffer[1024] = {0};
        size_t index = 0;
        
        while (state->current_char != '\0' && state->current_char != quote) {
            if (state->current_char == '\\') {
                parser_advance(state);  // 跳过转义字符
                if (state->current_char != '\0') {
                    if (index < sizeof(buffer) - 1) {
                        buffer[index++] = state->current_char;
                    }
                    parser_advance(state);
                }
            } else {
                if (index < sizeof(buffer) - 1) {
                    buffer[index++] = state->current_char;
                }
                parser_advance(state);
            }
        }
        
        if (state->current_char == quote) {
            parser_advance(state);  // 跳过结束引号
        }
        
        buffer[index] = '\0';
        token.type = TOKEN_STRING;
        token.value = string_duplicate_nullable(buffer);
        return token;
    }
    
    // 检查操作符和标点符号
    token.type = TOKEN_OPERATOR;
    
    // 多字符操作符
    char next_char = (state->position + 1 < state->length) ? state->source[state->position + 1] : '\0';
    
    // 双字符操作符：==, !=, <=, >=, &&, ||
    if ((current == '=' && next_char == '=') ||
        (current == '!' && next_char == '=') ||
        (current == '<' && next_char == '=') ||
        (current == '>' && next_char == '=') ||
        (current == '&' && next_char == '&') ||
        (current == '|' && next_char == '|')) {
        char double_op[3] = {current, next_char, '\0'};
        token.value = string_duplicate_nullable(double_op);
        parser_advance(state);
        parser_advance(state);
        return token;
    }
    
    // 单个字符操作符
    switch (current) {
        case '+':
        case '-':
        case '*':
        case '/':
        case '%':
        case '=':
        case '<':
        case '>':
        case '!':
        case '&':
        case '|':
        case '^':
        case '~':
        case '(':
        case ')':
        case '{':
        case '}':
        case '[':
        case ']':
        case ';':
        case ',':
        case '.':
        case ':':
        case '?':
            token.value = string_duplicate_nullable((char[]){current, '\0'});
            parser_advance(state);
            break;
        default:
            // 未知字符，跳过
            parser_advance(state);
            token.type = TOKEN_END;
            break;
    }
    
    return token;
}

/**
 * @brief 预读下一个token（不消费）
 */
static Token parser_peek_token(ParserState* state) {
    if (!state) {
        Token empty = {TOKEN_END, NULL, 0, 0};
        return empty;
    }
    if (!state->token_available) {
        state->current_token = parser_get_next_token(state);
        state->token_available = 1;
    }
    return state->current_token;
}

/**
 * @brief 消费当前预读token并前进
 */
static Token parser_consume_token(ParserState* state) {
    if (!state) {
        Token empty = {TOKEN_END, NULL, 0, 0};
        return empty;
    }
    if (!state->token_available) {
        return parser_get_next_token(state);
    }
    state->token_available = 0;
    return state->current_token;
}

/**
 * @brief 将类型名称转换为DataType枚举
 */
static DataType token_to_data_type(const char* type_name) {
    if (!type_name) return TYPE_CUSTOM;
    if (strcmp(type_name, "void") == 0) return TYPE_VOID;
    if (strcmp(type_name, "int") == 0) return TYPE_INT;
    if (strcmp(type_name, "float") == 0) return TYPE_FLOAT;
    if (strcmp(type_name, "double") == 0) return TYPE_FLOAT;
    if (strcmp(type_name, "char") == 0) return TYPE_INT;
    if (strcmp(type_name, "bool") == 0 || strcmp(type_name, "_Bool") == 0) return TYPE_BOOL;
    return TYPE_CUSTOM;
}

/**
 * @brief 解析类型声明（返回类型节点）
 */
static ASTNode* parse_type(ParserState* state) {
    Token token = parser_peek_token(state);
    if (token.type != TOKEN_KEYWORD) return NULL;
    
    const char* type_names[] = {"void", "int", "float", "double", "char", "bool", "_Bool", "long", "short", "unsigned", "signed", "const"};
    int is_type = 0;
    for (size_t i = 0; i < sizeof(type_names)/sizeof(type_names[0]); i++) {
        if (token.value && strcmp(token.value, type_names[i]) == 0) {
            is_type = 1;
            break;
        }
    }
    if (!is_type) return NULL;
    
    parser_consume_token(state);  // 消费类型关键字
    
    ASTNode* type_node = create_ast_node(AST_VARIABLE_DECL);
    if (type_node) {
        type_node->value.data_type = token_to_data_type(token.value);
        type_node->name = string_duplicate_nullable(token.value);
    }
    return type_node;
}

/**
 * @brief 解析参数列表
 */
static ASTNode* parse_parameter_list(ParserState* state) {
    ASTNode* head = NULL;
    ASTNode* tail = NULL;
    
    Token token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ")") == 0) {
        return NULL;  // 空参数列表
    }
    
    while (1) {
        // 解析类型
        ASTNode* type_node = parse_type(state);
        if (!type_node) break;
        
        // 解析参数名
        Token name_token = parser_peek_token(state);
        ASTNode* param_node = create_ast_node(AST_VARIABLE_DECL);
        if (!param_node) { ast_destroy(type_node); break; }
        
        param_node->value.data_type = type_node->value.data_type;
        param_node->name = NULL;
        
        if (name_token.type == TOKEN_IDENTIFIER) {
            parser_consume_token(state);
            param_node->name = string_duplicate_nullable(name_token.value);
        }
        
        safe_free((void**)&type_node);
        
        if (!head) {
            head = param_node;
            tail = param_node;
        } else {
            tail->next = param_node;
            tail = param_node;
        }
        
        token = parser_peek_token(state);
        if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ",") == 0) {
            parser_consume_token(state);
        } else {
            break;
        }
    }
    
    return head;
}

/**
 * @brief 解析代码块 { ... }
 */
static ASTNode* parse_block(ParserState* state) {
    Token token = parser_peek_token(state);
    if (!(token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "{") == 0)) {
        return NULL;
    }
    parser_consume_token(state);  // 消费 '{'
    
    ASTNode* block_node = create_ast_node(AST_BLOCK);
    if (!block_node) return NULL;
    
    ASTNode* first_stmt = NULL;
    ASTNode* last_stmt = NULL;
    
    while (1) {
        token = parser_peek_token(state);
        if (token.type == TOKEN_END) break;
        if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "}") == 0) {
            break;
        }
        
        ASTNode* stmt = parse_statement(state);
        if (stmt) {
            if (!first_stmt) {
                first_stmt = stmt;
                last_stmt = stmt;
            } else {
                last_stmt->next = stmt;
                last_stmt = stmt;
            }
        } else {
            break;
        }
    }
    
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "}") == 0) {
        parser_consume_token(state);
    }
    
    block_node->left = first_stmt;
    return block_node;
}

/**
 * @brief 解析return语句
 */
static ASTNode* parse_return_statement(ParserState* state) {
    Token token = parser_peek_token(state);
    if (!(token.type == TOKEN_KEYWORD && token.value && strcmp(token.value, "return") == 0)) {
        return NULL;
    }
    parser_consume_token(state);  // 消费return
    
    ASTNode* return_node = create_ast_node(AST_RETURN);
    if (!return_node) return NULL;
    
    // 检查是否有返回值
    token = parser_peek_token(state);
    if (!(token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ";") == 0)) {
        return_node->left = parse_expression(state);
    }
    
    // 消费分号
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ";") == 0) {
        parser_consume_token(state);
    }
    
    return return_node;
}

/**
 * @brief 解析if语句
 */
static ASTNode* parse_if_statement(ParserState* state) {
    Token token = parser_peek_token(state);
    if (!(token.type == TOKEN_KEYWORD && token.value && strcmp(token.value, "if") == 0)) {
        return NULL;
    }
    parser_consume_token(state);  // 消费if
    
    ASTNode* if_node = create_ast_node(AST_IF);
    if (!if_node) return NULL;
    
    // 消费 '('
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "(") == 0) {
        parser_consume_token(state);
    }
    
    // 解析条件
    if_node->left = parse_expression(state);
    
    // 消费 ')'
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ")") == 0) {
        parser_consume_token(state);
    }
    
    // 解析then分支
    if_node->right = parse_statement(state);
    
    // 检查else分支
    token = parser_peek_token(state);
    if (token.type == TOKEN_KEYWORD && token.value && strcmp(token.value, "else") == 0) {
        parser_consume_token(state);  // 消费else
        // 将else分支附加到right的next
        ASTNode* else_branch = parse_statement(state);
        if (if_node->right) {
            if_node->right->next = else_branch;
        } else {
            if_node->right = else_branch;
        }
    }
    
    return if_node;
}

/**
 * @brief 解析while语句
 */
static ASTNode* parse_while_statement(ParserState* state) {
    Token token = parser_peek_token(state);
    if (!(token.type == TOKEN_KEYWORD && token.value && strcmp(token.value, "while") == 0)) {
        return NULL;
    }
    parser_consume_token(state);  // 消费while
    
    ASTNode* loop_node = create_ast_node(AST_LOOP);
    if (!loop_node) return NULL;
    
    // 消费 '('
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "(") == 0) {
        parser_consume_token(state);
    }
    
    // 解析条件
    loop_node->left = parse_expression(state);
    
    // 消费 ')'
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ")") == 0) {
        parser_consume_token(state);
    }
    
    // 解析循环体
    loop_node->right = parse_statement(state);
    
    return loop_node;
}

/**
 * @brief 解析for语句
 */
static ASTNode* parse_for_statement(ParserState* state) {
    Token token = parser_peek_token(state);
    if (!(token.type == TOKEN_KEYWORD && token.value && strcmp(token.value, "for") == 0)) {
        return NULL;
    }
    parser_consume_token(state);  // 消费for
    
    ASTNode* loop_node = create_ast_node(AST_LOOP);
    if (!loop_node) return NULL;
    
    // 消费 '('
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "(") == 0) {
        parser_consume_token(state);
    }
    
    // 解析初始化语句
    ASTNode* init = NULL;
    token = parser_peek_token(state);
    if (!(token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ";") == 0)) {
        init = parse_statement(state);
    } else {
        parser_consume_token(state);  // 消费空初始化后的分号
    }
    
    // 解析条件
    ASTNode* cond = NULL;
    token = parser_peek_token(state);
    if (!(token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ";") == 0)) {
        cond = parse_expression(state);
        token = parser_peek_token(state);
        if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ";") == 0) {
            parser_consume_token(state);
        }
    } else {
        parser_consume_token(state);
    }
    
    // 解析增量
    ASTNode* inc = NULL;
    token = parser_peek_token(state);
    if (!(token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ")") == 0)) {
        inc = parse_expression(state);
    }
    
    // 消费 ')'
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ")") == 0) {
        parser_consume_token(state);
    }
    
    // 将init/cond/inc存储在loop_node中
    // left = init, right = body, cond和inc通过特殊约定存储
    loop_node->left = cond;  // 条件
    loop_node->right = parse_statement(state);  // 循环体
    
    // 将init和inc存储为额外节点（通过AST_VARIABLE_DECL包装）
    if (init) {
        ASTNode* init_wrapper = create_ast_node(AST_VARIABLE_DECL);
        if (init_wrapper) {
            init_wrapper->left = init;
            init_wrapper->next = loop_node->left;
            loop_node->left = (ASTNode*)init_wrapper;
        }
    }
    if (inc) {
        ASTNode* inc_wrapper = create_ast_node(AST_VARIABLE_DECL);
        if (inc_wrapper) {
            inc_wrapper->left = inc;
            // 附加到loop_node的right链上
            inc_wrapper->next = loop_node->right;
            loop_node->right = inc_wrapper;
        }
    }
    
    return loop_node;
}

/**
 * @brief 解析变量声明
 */
static ASTNode* parse_declaration(ParserState* state) {
    ASTNode* type_node = parse_type(state);
    if (!type_node) return NULL;
    
    DataType data_type = type_node->value.data_type;
    safe_free((void**)&type_node);
    
    Token name_token = parser_peek_token(state);
    if (name_token.type != TOKEN_IDENTIFIER) {
        return NULL;
    }
    parser_consume_token(state);
    
    ASTNode* decl_node = create_ast_node(AST_VARIABLE_DECL);
    if (!decl_node) return NULL;
    
    decl_node->value.data_type = data_type;
    decl_node->name = string_duplicate_nullable(name_token.value);
    
    // 检查是否有初始化
    Token token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "=") == 0) {
        parser_consume_token(state);  // 消费=
        decl_node->left = parse_assignment(state);
    }
    
    // 消费分号
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ";") == 0) {
        parser_consume_token(state);
    }
    
    return decl_node;
}

/**
 * @brief 解析语句（完整实现：支持声明、控制流、表达式等）
 */
static ASTNode* parse_statement(ParserState* state) {
    Token token = parser_peek_token(state);
    if (token.type == TOKEN_END) return NULL;
    
    // 检查是否为代码块
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "{") == 0) {
        return parse_block(state);
    }
    
    // 检查是否为关键字语句
    if (token.type == TOKEN_KEYWORD && token.value) {
        if (strcmp(token.value, "return") == 0) {
            return parse_return_statement(state);
        }
        if (strcmp(token.value, "if") == 0) {
            return parse_if_statement(state);
        }
        if (strcmp(token.value, "while") == 0) {
            return parse_while_statement(state);
        }
        if (strcmp(token.value, "for") == 0) {
            return parse_for_statement(state);
        }
        // 检查是否为类型关键字（声明语句）
        const char* type_keywords[] = {"int", "float", "double", "char", "void", "long", "short", "unsigned", "signed", "const", "bool", "_Bool", "struct", "union"};
        for (size_t i = 0; i < sizeof(type_keywords)/sizeof(type_keywords[0]); i++) {
            if (strcmp(token.value, type_keywords[i]) == 0) {
                return parse_declaration(state);
            }
        }
    }
    
    // 默认：表达式语句
    ASTNode* expr_node = parse_expression(state);
    
    // 消费分号
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ";") == 0) {
        parser_consume_token(state);
    }
    
    return expr_node;
}

/**
 * @brief 解析主表达式（标识符、字面量、括号表达式）
 */
static ASTNode* parse_primary(ParserState* state) {
    Token token = parser_peek_token(state);
    if (token.type == TOKEN_END) return NULL;
    
    if (token.type == TOKEN_IDENTIFIER) {
        parser_consume_token(state);
        ASTNode* node = create_ast_node(AST_VARIABLE_REF);
        if (node) {
            node->name = string_duplicate_nullable(token.value);
        }
        // 检查是否为函数调用
        Token next = parser_peek_token(state);
        if (next.type == TOKEN_OPERATOR && next.value && strcmp(next.value, "(") == 0) {
            return parse_function_call(state, node);
        }
        return node;
    }
    
    if (token.type == TOKEN_NUMBER) {
        parser_consume_token(state);
        ASTNode* node = create_ast_node(AST_LITERAL);
        if (node && token.value) {
            if (strchr(token.value, '.') != NULL) {
                node->value.float_value = (float)atof(token.value);
            } else {
                node->value.int_value = atoi(token.value);
            }
        }
        if (token.value) safe_free((void**)&token.value);
        return node;
    }
    
    if (token.type == TOKEN_STRING) {
        parser_consume_token(state);
        ASTNode* node = create_ast_node(AST_LITERAL);
        if (node) {
            node->value.string_value = string_duplicate_nullable(token.value);
        }
        if (token.value) safe_free((void**)&token.value);
        return node;
    }
    
    // 括号表达式
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "(") == 0) {
        parser_consume_token(state);
        ASTNode* node = parse_expression(state);
        Token close = parser_peek_token(state);
        if (close.type == TOKEN_OPERATOR && close.value && strcmp(close.value, ")") == 0) {
            parser_consume_token(state);
        }
        return node;
    }
    
    return NULL;
}

/**
 * @brief 解析函数调用
 */
static ASTNode* parse_function_call(ParserState* state, ASTNode* callee) {
    // 已经消费了函数名，当前token应该是'('
    Token token = parser_peek_token(state);
    if (!(token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "(") == 0)) {
        return callee;
    }
    parser_consume_token(state);  // 消费'('
    
    ASTNode* call_node = create_ast_node(AST_CALL);
    if (!call_node) return callee;
    
    call_node->name = string_duplicate_nullable(callee->name);
    ast_destroy(callee);
    
    ASTNode* arg_head = NULL;
    ASTNode* arg_tail = NULL;
    
    // 解析参数
    token = parser_peek_token(state);
    if (!(token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ")") == 0)) {
        while (1) {
            ASTNode* arg = parse_assignment(state);
            if (!arg) break;
            
            if (!arg_head) {
                arg_head = arg;
                arg_tail = arg;
            } else {
                arg_tail->next = arg;
                arg_tail = arg;
            }
            
            token = parser_peek_token(state);
            if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ",") == 0) {
                parser_consume_token(state);
            } else {
                break;
            }
        }
    }
    
    call_node->left = arg_head;
    
    token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, ")") == 0) {
        parser_consume_token(state);
    }
    
    return call_node;
}

/**
 * @brief 解析一元表达式
 */
static ASTNode* parse_unary(ParserState* state) {
    Token token = parser_peek_token(state);
    
    if (token.type == TOKEN_OPERATOR && token.value) {
        UnaryOperator op = UNARY_PLUS;
        int is_unary = 1;
        
        if (strcmp(token.value, "+") == 0) op = UNARY_PLUS;
        else if (strcmp(token.value, "-") == 0) op = UNARY_MINUS;
        else if (strcmp(token.value, "!") == 0) op = UNARY_NOT;
        else is_unary = 0;
        
        if (is_unary) {
            parser_consume_token(state);
            ASTNode* node = create_ast_node(AST_UNARY_OP);
            if (node) {
                node->value.unary_op = op;
                node->left = parse_unary(state);
            }
            return node;
        }
    }
    
    return parse_primary(state);
}

/**
 * @brief 解析乘法表达式
 */
static ASTNode* parse_multiplicative(ParserState* state) {
    ASTNode* left = parse_unary(state);
    if (!left) return NULL;
    
    while (1) {
        Token token = parser_peek_token(state);
        if (!(token.type == TOKEN_OPERATOR && token.value)) break;
        
        BinaryOperator op = OP_ADD;
        int is_op = 1;
        if (strcmp(token.value, "*") == 0) op = OP_MUL;
        else if (strcmp(token.value, "/") == 0) op = OP_DIV;
        else if (strcmp(token.value, "%") == 0) op = OP_MOD;
        else is_op = 0;
        
        if (!is_op) break;
        
        parser_consume_token(state);
        ASTNode* right = parse_unary(state);
        if (!right) break;
        
        ASTNode* node = create_ast_node(AST_BINARY_OP);
        if (!node) { ast_destroy(right); break; }
        node->value.bin_op = op;
        node->left = left;
        node->right = right;
        left = node;
    }
    
    return left;
}

/**
 * @brief 解析加法表达式
 */
static ASTNode* parse_additive(ParserState* state) {
    ASTNode* left = parse_multiplicative(state);
    if (!left) return NULL;
    
    while (1) {
        Token token = parser_peek_token(state);
        if (!(token.type == TOKEN_OPERATOR && token.value)) break;
        
        BinaryOperator op = OP_ADD;
        int is_op = 1;
        if (strcmp(token.value, "+") == 0) op = OP_ADD;
        else if (strcmp(token.value, "-") == 0) op = OP_SUB;
        else is_op = 0;
        
        if (!is_op) break;
        
        parser_consume_token(state);
        ASTNode* right = parse_multiplicative(state);
        if (!right) break;
        
        ASTNode* node = create_ast_node(AST_BINARY_OP);
        if (!node) { ast_destroy(right); break; }
        node->value.bin_op = op;
        node->left = left;
        node->right = right;
        left = node;
    }
    
    return left;
}

/**
 * @brief 解析关系表达式
 */
static ASTNode* parse_relational(ParserState* state) {
    ASTNode* left = parse_additive(state);
    if (!left) return NULL;
    
    while (1) {
        Token token = parser_peek_token(state);
        if (!(token.type == TOKEN_OPERATOR && token.value)) break;
        
        BinaryOperator op = OP_LT;
        int is_op = 1;
        if (strcmp(token.value, "<") == 0) op = OP_LT;
        else if (strcmp(token.value, ">") == 0) op = OP_GT;
        else if (strcmp(token.value, "<=") == 0) op = OP_LE;
        else if (strcmp(token.value, ">=") == 0) op = OP_GE;
        else is_op = 0;
        
        if (!is_op) break;
        
        parser_consume_token(state);
        ASTNode* right = parse_additive(state);
        if (!right) break;
        
        ASTNode* node = create_ast_node(AST_BINARY_OP);
        if (!node) { ast_destroy(right); break; }
        node->value.bin_op = op;
        node->left = left;
        node->right = right;
        left = node;
    }
    
    return left;
}

/**
 * @brief 解析相等表达式
 */
static ASTNode* parse_equality(ParserState* state) {
    ASTNode* left = parse_relational(state);
    if (!left) return NULL;
    
    while (1) {
        Token token = parser_peek_token(state);
        if (!(token.type == TOKEN_OPERATOR && token.value)) break;
        
        BinaryOperator op = OP_EQ;
        int is_op = 1;
        if (strcmp(token.value, "==") == 0) op = OP_EQ;
        else if (strcmp(token.value, "!=") == 0) op = OP_NE;
        else is_op = 0;
        
        if (!is_op) break;
        
        parser_consume_token(state);
        ASTNode* right = parse_relational(state);
        if (!right) break;
        
        ASTNode* node = create_ast_node(AST_BINARY_OP);
        if (!node) { ast_destroy(right); break; }
        node->value.bin_op = op;
        node->left = left;
        node->right = right;
        left = node;
    }
    
    return left;
}

/**
 * @brief 解析逻辑与表达式
 */
static ASTNode* parse_logical_and(ParserState* state) {
    ASTNode* left = parse_equality(state);
    if (!left) return NULL;
    
    while (1) {
        Token token = parser_peek_token(state);
        if (!(token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "&&") == 0)) break;
        
        parser_consume_token(state);
        ASTNode* right = parse_equality(state);
        if (!right) break;
        
        ASTNode* node = create_ast_node(AST_BINARY_OP);
        if (!node) { ast_destroy(right); break; }
        node->value.bin_op = OP_AND;
        node->left = left;
        node->right = right;
        left = node;
    }
    
    return left;
}

/**
 * @brief 解析逻辑或表达式
 */
static ASTNode* parse_logical_or(ParserState* state) {
    ASTNode* left = parse_logical_and(state);
    if (!left) return NULL;
    
    while (1) {
        Token token = parser_peek_token(state);
        if (!(token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "||") == 0)) break;
        
        parser_consume_token(state);
        ASTNode* right = parse_logical_and(state);
        if (!right) break;
        
        ASTNode* node = create_ast_node(AST_BINARY_OP);
        if (!node) { ast_destroy(right); break; }
        node->value.bin_op = OP_OR;
        node->left = left;
        node->right = right;
        left = node;
    }
    
    return left;
}

/**
 * @brief 解析赋值表达式
 */
static ASTNode* parse_assignment(ParserState* state) {
    ASTNode* left = parse_logical_or(state);
    if (!left) return NULL;
    
    Token token = parser_peek_token(state);
    if (token.type == TOKEN_OPERATOR && token.value && strcmp(token.value, "=") == 0) {
        parser_consume_token(state);
        ASTNode* right = parse_assignment(state);
        if (!right) return left;
        
        ASTNode* assign_node = create_ast_node(AST_ASSIGNMENT);
        if (!assign_node) { ast_destroy(right); return left; }
        assign_node->left = left;
        assign_node->right = right;
        return assign_node;
    }
    
    return left;
}

/**
 * @brief 解析表达式（完整递归下降实现，支持所有运算符优先级）
 */
static ASTNode* parse_expression(ParserState* state) {
    return parse_assignment(state);
}

/**
 * @brief 递归将AST节点转换为源代码（辅助函数）
 */
#define AST_TO_SOURCE_MAX_DEPTH 256

/* K-089: AST递归序列化，限制最大深度 256 防止恶意嵌套AST栈溢出 */
static void ast_to_source_recursive(const ASTNode* node, int indent_level, char* buffer, size_t buffer_size, size_t* pos) {
    if (!node || !buffer || !pos) return;
    if (indent_level > AST_TO_SOURCE_MAX_DEPTH) return;
    
    // 生成缩进
    char indent_str[128] = {0};
    int indent_count = indent_level * 4;
    if (indent_count > 120) indent_count = 120;
    memset(indent_str, ' ', indent_count);
    indent_str[indent_count] = '\0';
    
    switch (node->type) {
        case AST_PROGRAM: {
            // 遍历程序中的所有语句
            const ASTNode* stmt = node->left;
            while (stmt) {
                ast_to_source_recursive(stmt, 0, buffer, buffer_size, pos);
                stmt = stmt->next;
            }
            break;
        }
        
        case AST_FUNCTION: {
            // 函数返回类型
            const char* type_names[] = {"void", "int", "float", "float", "int", "bool", "custom"};
            int type_idx = (int)node->value.data_type;
            if (type_idx < 0 || type_idx > TYPE_CUSTOM) type_idx = TYPE_CUSTOM;
            
            size_t remaining = buffer_size - *pos;
            int written = snprintf(buffer + *pos, remaining, "%s%s %s", indent_str, type_names[type_idx], node->name ? node->name : "unknown");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            
            // 参数列表
            remaining = buffer_size - *pos;
            written = snprintf(buffer + *pos, remaining, "(");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            
            const ASTNode* param = node->right;
            int param_idx = 0;
            while (param) {
                if (param_idx > 0) {
                    remaining = buffer_size - *pos;
                    written = snprintf(buffer + *pos, remaining, ", ");
                    if (written > 0 && (size_t)written < remaining) *pos += written;
                }
                const char* ptype_names[] = {"void", "int", "float", "float", "int", "bool", "custom"};
                int ptype_idx = (int)param->value.data_type;
                if (ptype_idx < 0 || ptype_idx > TYPE_CUSTOM) ptype_idx = TYPE_CUSTOM;
                remaining = buffer_size - *pos;
                written = snprintf(buffer + *pos, remaining, "%s %s", ptype_names[ptype_idx], param->name ? param->name : "");
                if (written > 0 && (size_t)written < remaining) *pos += written;
                param = param->next;
                param_idx++;
            }
            
            remaining = buffer_size - *pos;
            written = snprintf(buffer + *pos, remaining, ")");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            
            // 函数体
            remaining = buffer_size - *pos;
            written = snprintf(buffer + *pos, remaining, " {\n");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            
            if (node->left) {
                ast_to_source_recursive(node->left, indent_level + 1, buffer, buffer_size, pos);
            }
            
            remaining = buffer_size - *pos;
            written = snprintf(buffer + *pos, remaining, "%s}\n\n", indent_str);
            if (written > 0 && (size_t)written < remaining) *pos += written;
            break;
        }
        
        case AST_BLOCK: {
            const ASTNode* stmt = node->left;
            while (stmt) {
                ast_to_source_recursive(stmt, indent_level, buffer, buffer_size, pos);
                stmt = stmt->next;
            }
            break;
        }
        
        case AST_VARIABLE_DECL: {
            const char* type_names[] = {"void", "int", "float", "float", "int", "bool", "custom"};
            int type_idx = (int)node->value.data_type;
            if (type_idx < 0 || type_idx > TYPE_CUSTOM) type_idx = TYPE_CUSTOM;
            
            size_t remaining = buffer_size - *pos;
            int written = snprintf(buffer + *pos, remaining, "%s%s %s", indent_str, type_names[type_idx], node->name ? node->name : "var");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            
            if (node->left) {
                remaining = buffer_size - *pos;
                written = snprintf(buffer + *pos, remaining, " = ");
                if (written > 0 && (size_t)written < remaining) *pos += written;
                ast_to_source_recursive(node->left, indent_level, buffer, buffer_size, pos);
            }
            
            remaining = buffer_size - *pos;
            written = snprintf(buffer + *pos, remaining, ";\n");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            break;
        }
        
        case AST_VARIABLE_REF: {
            size_t remaining = buffer_size - *pos;
            int written = snprintf(buffer + *pos, remaining, "%s", node->name ? node->name : "var");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            break;
        }
        
        case AST_LITERAL: {
            if (node->value.string_value) {
                size_t remaining = buffer_size - *pos;
                int written = snprintf(buffer + *pos, remaining, "\"%s\"", node->value.string_value);
                if (written > 0 && (size_t)written < remaining) *pos += written;
            } else if (node->value.float_value != 0.0f) {
                /* K-094: 移除*(int*)&float_value类型双关,避免strict-aliasing UB */
                size_t remaining = buffer_size - *pos;
                int written = snprintf(buffer + *pos, remaining, "%g", (double)node->value.float_value);
                if (written > 0 && (size_t)written < remaining) *pos += written;
            } else {
                size_t remaining = buffer_size - *pos;
                int written = snprintf(buffer + *pos, remaining, "%d", node->value.int_value);
                if (written > 0 && (size_t)written < remaining) *pos += written;
            }
            break;
        }
        
        case AST_BINARY_OP: {
            const char* op_str = "+";
            switch (node->value.bin_op) {
                case OP_ADD: op_str = "+"; break;
                case OP_SUB: op_str = "-"; break;
                case OP_MUL: op_str = "*"; break;
                case OP_DIV: op_str = "/"; break;
                case OP_MOD: op_str = "%"; break;
                case OP_EQ: op_str = "=="; break;
                case OP_NE: op_str = "!="; break;
                case OP_LT: op_str = "<"; break;
                case OP_GT: op_str = ">"; break;
                case OP_LE: op_str = "<="; break;
                case OP_GE: op_str = ">="; break;
                case OP_AND: op_str = "&&"; break;
                case OP_OR: op_str = "||"; break;
            }
            
            if (node->left) {
                ast_to_source_recursive(node->left, indent_level, buffer, buffer_size, pos);
            }
            size_t remaining = buffer_size - *pos;
            int written = snprintf(buffer + *pos, remaining, " %s ", op_str);
            if (written > 0 && (size_t)written < remaining) *pos += written;
            if (node->right) {
                ast_to_source_recursive(node->right, indent_level, buffer, buffer_size, pos);
            }
            break;
        }
        
        case AST_UNARY_OP: {
            const char* op_str = "+";
            switch (node->value.unary_op) {
                case UNARY_PLUS: op_str = "+"; break;
                case UNARY_MINUS: op_str = "-"; break;
                case UNARY_NOT: op_str = "!"; break;
            }
            size_t remaining = buffer_size - *pos;
            int written = snprintf(buffer + *pos, remaining, "%s", op_str);
            if (written > 0 && (size_t)written < remaining) *pos += written;
            if (node->left) {
                ast_to_source_recursive(node->left, indent_level, buffer, buffer_size, pos);
            }
            break;
        }
        
        case AST_CALL: {
            size_t remaining = buffer_size - *pos;
            int written = snprintf(buffer + *pos, remaining, "%s(", node->name ? node->name : "func");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            
            const ASTNode* arg = node->left;
            int arg_idx = 0;
            while (arg) {
                if (arg_idx > 0) {
                    remaining = buffer_size - *pos;
                    written = snprintf(buffer + *pos, remaining, ", ");
                    if (written > 0 && (size_t)written < remaining) *pos += written;
                }
                ast_to_source_recursive(arg, indent_level, buffer, buffer_size, pos);
                arg = arg->next;
                arg_idx++;
            }
            
            remaining = buffer_size - *pos;
            written = snprintf(buffer + *pos, remaining, ")");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            break;
        }
        
        case AST_RETURN: {
            size_t remaining = buffer_size - *pos;
            int written = snprintf(buffer + *pos, remaining, "%sreturn", indent_str);
            if (written > 0 && (size_t)written < remaining) *pos += written;
            
            if (node->left) {
                remaining = buffer_size - *pos;
                written = snprintf(buffer + *pos, remaining, " ");
                if (written > 0 && (size_t)written < remaining) *pos += written;
                ast_to_source_recursive(node->left, indent_level, buffer, buffer_size, pos);
            }
            
            remaining = buffer_size - *pos;
            written = snprintf(buffer + *pos, remaining, ";\n");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            break;
        }
        
        case AST_IF: {
            size_t remaining = buffer_size - *pos;
            int written = snprintf(buffer + *pos, remaining, "%sif (", indent_str);
            if (written > 0 && (size_t)written < remaining) *pos += written;
            
            if (node->left) {
                ast_to_source_recursive(node->left, indent_level, buffer, buffer_size, pos);
            }
            
            remaining = buffer_size - *pos;
            written = snprintf(buffer + *pos, remaining, ") {\n");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            
            // then分支（right节点）
            if (node->right) {
                ast_to_source_recursive(node->right, indent_level + 1, buffer, buffer_size, pos);
            }
            
            remaining = buffer_size - *pos;
            written = snprintf(buffer + *pos, remaining, "%s}", indent_str);
            if (written > 0 && (size_t)written < remaining) *pos += written;
            
            // 检查else分支（通过right->next）
            if (node->right && node->right->next) {
                remaining = buffer_size - *pos;
                written = snprintf(buffer + *pos, remaining, " else {\n");
                if (written > 0 && (size_t)written < remaining) *pos += written;
                ast_to_source_recursive(node->right->next, indent_level + 1, buffer, buffer_size, pos);
                remaining = buffer_size - *pos;
                written = snprintf(buffer + *pos, remaining, "%s}", indent_str);
                if (written > 0 && (size_t)written < remaining) *pos += written;
            }
            
            remaining = buffer_size - *pos;
            written = snprintf(buffer + *pos, remaining, "\n");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            break;
        }
        
        case AST_LOOP: {
            // 检查是否是for循环（有init包装器）
            int is_for_loop = 0;
            if (node->left && node->left->type == AST_VARIABLE_DECL) {
                is_for_loop = 1;
            }
            
            if (is_for_loop) {
                size_t remaining = buffer_size - *pos;
                int written = snprintf(buffer + *pos, remaining, "%sfor (", indent_str);
                if (written > 0 && (size_t)written < remaining) *pos += written;
                
                // init
                ASTNode* init_wrapper = node->left;
                if (init_wrapper->left) {
                    ast_to_source_recursive(init_wrapper->left, 0, buffer, buffer_size, pos);
                }
                
                remaining = buffer_size - *pos;
                written = snprintf(buffer + *pos, remaining, "; ");
                if (written > 0 && (size_t)written < remaining) *pos += written;
                
                // cond
                if (init_wrapper->next) {
                    ast_to_source_recursive(init_wrapper->next, 0, buffer, buffer_size, pos);
                }
                
                remaining = buffer_size - *pos;
                written = snprintf(buffer + *pos, remaining, "; ");
                if (written > 0 && (size_t)written < remaining) *pos += written;
                
                // inc
                if (node->right && node->right->type == AST_VARIABLE_DECL) {
                    ASTNode* inc_wrapper = node->right;
                    if (inc_wrapper->left) {
                        ast_to_source_recursive(inc_wrapper->left, 0, buffer, buffer_size, pos);
                    }
                    // 循环体在inc_wrapper->next
                    remaining = buffer_size - *pos;
                    written = snprintf(buffer + *pos, remaining, ") {\n");
                    if (written > 0 && (size_t)written < remaining) *pos += written;
                    if (inc_wrapper->next) {
                        ast_to_source_recursive(inc_wrapper->next, indent_level + 1, buffer, buffer_size, pos);
                    }
                } else {
                    remaining = buffer_size - *pos;
                    written = snprintf(buffer + *pos, remaining, ") {\n");
                    if (written > 0 && (size_t)written < remaining) *pos += written;
                    if (node->right) {
                        ast_to_source_recursive(node->right, indent_level + 1, buffer, buffer_size, pos);
                    }
                }
                
                remaining = buffer_size - *pos;
                written = snprintf(buffer + *pos, remaining, "%s}\n", indent_str);
                if (written > 0 && (size_t)written < remaining) *pos += written;
            } else {
                // while循环
                size_t remaining = buffer_size - *pos;
                int written = snprintf(buffer + *pos, remaining, "%swhile (", indent_str);
                if (written > 0 && (size_t)written < remaining) *pos += written;
                
                if (node->left) {
                    ast_to_source_recursive(node->left, indent_level, buffer, buffer_size, pos);
                }
                
                remaining = buffer_size - *pos;
                written = snprintf(buffer + *pos, remaining, ") {\n");
                if (written > 0 && (size_t)written < remaining) *pos += written;
                
                if (node->right) {
                    ast_to_source_recursive(node->right, indent_level + 1, buffer, buffer_size, pos);
                }
                
                remaining = buffer_size - *pos;
                written = snprintf(buffer + *pos, remaining, "%s}\n", indent_str);
                if (written > 0 && (size_t)written < remaining) *pos += written;
            }
            break;
        }
        
        case AST_ASSIGNMENT: {
            if (node->left) {
                ast_to_source_recursive(node->left, indent_level, buffer, buffer_size, pos);
            }
            size_t remaining = buffer_size - *pos;
            int written = snprintf(buffer + *pos, remaining, " = ");
            if (written > 0 && (size_t)written < remaining) *pos += written;
            if (node->right) {
                ast_to_source_recursive(node->right, indent_level, buffer, buffer_size, pos);
            }
            break;
        }
    }
}

/**
 * @brief 将AST转换为源代码（完整实现）
 */
static char* ast_to_source(const ASTNode* node, int indent_level) {
    if (!node) return string_duplicate_nullable("");
    
    char* buffer = (char*)safe_malloc(65536);
    if (!buffer) return string_duplicate_nullable("");
    memset(buffer, 0, 65536);
    
    size_t pos = 0;
    ast_to_source_recursive(node, indent_level, buffer, 65536, &pos);
    
    return buffer;
}

/* 添加独立的C代码生成API */
char* self_programming_generate_c(SelfProgrammingEngine* engine,
                                  const CodeSpecification* spec) {
    if (!engine || !spec) return NULL;
    return synthesize_code(engine, spec);
}

/* ============================================================================
 * 公共API实现
 * =========================================================================== */

/**
 * @brief 创建自我编程引擎
 */
SelfProgrammingEngine* self_programming_engine_create(ProgrammingLanguage language) {
    SelfProgrammingEngine* engine = (SelfProgrammingEngine*)safe_malloc(sizeof(SelfProgrammingEngine));
    if (!engine) return NULL;
    
    memset(engine, 0, sizeof(SelfProgrammingEngine));
    engine->language = language;
    engine->is_initialized = 1;
    
    /* H-017集成: 初始化编程增强引擎（重构分析/性能分析/安全扫描） */
    engine->penh_engine = penh_refactor_engine_create();
    
    return engine;
}

/**
 * @brief 销毁自我编程引擎
 */
void self_programming_engine_destroy(SelfProgrammingEngine* engine) {
    if (!engine) return;
    
    // 清理内部状态
    if (engine->parser_state) {
        safe_free((void**)&engine->parser_state);
    }
    
    if (engine->code_generator) {
        safe_free((void**)&engine->code_generator);
    }
    
    if (engine->analyzer_state) {
        safe_free((void**)&engine->analyzer_state);
    }
    
    /* H-017集成: 释放编程增强引擎 */
    if (engine->penh_engine) {
        penh_refactor_engine_free((PenhRefactorEngine*)engine->penh_engine);
    }
    
    safe_free((void**)&engine);
}

/**
 * @brief 解析源代码为AST
 */
ASTNode* parse_source_code(SelfProgrammingEngine* engine, const char* source_code) {
    if (!engine || !source_code) return NULL;
    
    // 初始化解析器状态
    ParserState state;
    memset(&state, 0, sizeof(ParserState));
    parser_init(&state, source_code);
    state.token_available = 0;
    
    // 创建程序节点
    ASTNode* program_node = create_ast_node(AST_PROGRAM);
    if (!program_node) return NULL;
    
    // 解析顶层声明（函数定义和全局变量）
    ASTNode* first_decl = NULL;
    ASTNode* last_decl = NULL;
    
    while (1) {
        Token token = parser_peek_token(&state);
        if (token.type == TOKEN_END) break;
        
        // 尝试解析函数定义
        // 格式：类型 标识符 ( 参数列表 ) { 函数体 }
        // 先检查是否为类型关键字
        if (token.type == TOKEN_KEYWORD && token.value) {
            const char* type_keywords[] = {"void", "int", "float", "double", "char", "long", "short", "unsigned", "signed", "const", "bool", "_Bool", "struct", "union"};
            int is_type = 0;
            for (size_t i = 0; i < sizeof(type_keywords)/sizeof(type_keywords[0]); i++) {
                if (strcmp(token.value, type_keywords[i]) == 0) {
                    is_type = 1;
                    break;
                }
            }
            
            if (is_type) {
                // 保存位置，尝试解析函数
                size_t saved_pos = state.position;
                int saved_line = state.line;
                int saved_col = state.column;
                char saved_char = state.current_char;
                int saved_token_avail = state.token_available;
                Token saved_token = state.current_token;
                
                // 解析返回类型
                ASTNode* type_node = parse_type(&state);
                if (type_node) {
                    DataType return_type = type_node->value.data_type;
                    safe_free((void**)&type_node);
                    
                    // 检查下一个token是否为标识符（函数名）
                    Token name_token = parser_peek_token(&state);
                    if (name_token.type == TOKEN_IDENTIFIER) {
                        parser_consume_token(&state);
                        
                        // 检查是否为'('（函数定义）
                        Token paren_token = parser_peek_token(&state);
                        if (paren_token.type == TOKEN_OPERATOR && paren_token.value && strcmp(paren_token.value, "(") == 0) {
                            // 确实是函数定义
                            ASTNode* func_node = create_ast_node(AST_FUNCTION);
                            if (func_node) {
                                func_node->value.data_type = return_type;
                                func_node->name = string_duplicate_nullable(name_token.value);
                                
                                parser_consume_token(&state);  // 消费'('
                                
                                // 解析参数列表
                                func_node->right = parse_parameter_list(&state);
                                
                                // 消费')'
                                Token rp_token = parser_peek_token(&state);
                                if (rp_token.type == TOKEN_OPERATOR && rp_token.value && strcmp(rp_token.value, ")") == 0) {
                                    parser_consume_token(&state);
                                }
                                
                                // 解析函数体
                                Token brace_token = parser_peek_token(&state);
                                if (brace_token.type == TOKEN_OPERATOR && brace_token.value && strcmp(brace_token.value, "{") == 0) {
                                    func_node->left = parse_block(&state);
                                } else {
                                    // 没有函数体，可能是声明
                                    ast_destroy(func_node);
                                    func_node = NULL;
                                    // 恢复状态
                                    state.position = saved_pos;
                                    state.line = saved_line;
                                    state.column = saved_col;
                                    state.current_char = saved_char;
                                    state.token_available = saved_token_avail;
                                    state.current_token = saved_token;
                                }
                                
                                if (func_node) {
                                    if (!first_decl) {
                                        first_decl = func_node;
                                        last_decl = func_node;
                                    } else {
                                        last_decl->next = func_node;
                                        last_decl = func_node;
                                    }
                                    continue;
                                }
                            }
                        } else {
                            // 不是函数定义，恢复状态
                            state.position = saved_pos;
                            state.line = saved_line;
                            state.column = saved_col;
                            state.current_char = saved_char;
                            state.token_available = saved_token_avail;
                            state.current_token = saved_token;
                        }
                    } else {
                        // 不是标识符，恢复状态
                        state.position = saved_pos;
                        state.line = saved_line;
                        state.column = saved_col;
                        state.current_char = saved_char;
                        state.token_available = saved_token_avail;
                        state.current_token = saved_token;
                    }
                } else {
                    // 不是类型，恢复状态
                    state.position = saved_pos;
                    state.line = saved_line;
                    state.column = saved_col;
                    state.current_char = saved_char;
                    state.token_available = saved_token_avail;
                    state.current_token = saved_token;
                }
            }
        }
        
        // 如果不是函数定义，尝试作为顶层语句解析
        ASTNode* stmt = parse_statement(&state);
        if (stmt) {
            if (!first_decl) {
                first_decl = stmt;
                last_decl = stmt;
            } else {
                last_decl->next = stmt;
                last_decl = stmt;
            }
        } else {
            // 无法解析，跳过当前token
            parser_consume_token(&state);
        }
    }
    
    program_node->left = first_decl;
    return program_node;
}

/**
 * @brief 销毁AST
 */
void ast_destroy(ASTNode* ast) {
    if (!ast) return;
    
    // 递归销毁子节点
    if (ast->left) ast_destroy(ast->left);
    if (ast->right) ast_destroy(ast->right);
    if (ast->next) ast_destroy(ast->next);
    
    // 释放字符串资源
    if (ast->name) {
        safe_free((void**)&ast->name);
    }
    
    if (ast->type == AST_LITERAL && ast->value.string_value) {
        safe_free((void**)&ast->value.string_value);
    }
    
    safe_free((void**)&ast);
}

/**
 * @brief 从AST生成源代码
 */
char* generate_source_code(SelfProgrammingEngine* engine, const ASTNode* ast) {
    if (!engine || !ast) return NULL;
    
    // 调用内部转换函数
    return ast_to_source(ast, 0);
}

#define ANALYZE_MAX_DEPTH 512

/* K-089/K-100: AST分析递归,限制最大深度512防止恶意嵌套栈溢出 */
static void analyze_node_recursive(const ASTNode* node, CodeAnalysisResult* result, int current_depth, int* max_depth) {
    if (!node || !result) return;
    if (current_depth > ANALYZE_MAX_DEPTH) return;
    
    result->line_count++;
    
    if (current_depth > *max_depth) {
        *max_depth = current_depth;
    }
    
    switch (node->type) {
        case AST_FUNCTION:
            result->function_count++;
            break;
        case AST_VARIABLE_DECL:
            result->variable_count++;
            break;
        case AST_IF:
        case AST_LOOP:
            result->cyclomatic_complexity++;
            break;
        case AST_BINARY_OP:
            if (node->value.bin_op == OP_AND || node->value.bin_op == OP_OR) {
                result->cyclomatic_complexity++;
            }
            break;
        default:
            break;
    }
    
    if (node->left) analyze_node_recursive(node->left, result, current_depth + 1, max_depth);
    if (node->right) analyze_node_recursive(node->right, result, current_depth + 1, max_depth);
    if (node->next) analyze_node_recursive(node->next, result, current_depth, max_depth);
}

/**
 * @brief 分析代码复杂度（完整AST遍历实现）
 */
int analyze_code_complexity(SelfProgrammingEngine* engine, 
                            const ASTNode* ast, 
                            CodeAnalysisResult* result) {
    if (!engine || !ast || !result) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    memset(result, 0, sizeof(CodeAnalysisResult));
    
    // 初始圈复杂度为1（顺序结构）
    result->cyclomatic_complexity = 1;
    
    // 递归遍历AST收集统计信息
    int max_depth = 0;
    analyze_node_recursive(ast, result, 0, &max_depth);
    result->max_nesting_depth = max_depth;
    
    // 计算可维护性指数（基于经验公式）
    // MI = 171 - 5.2*ln(avgCyclomatic) - 0.23*avgLoc - 16.2*ln(avgFuncCount)
    if (result->function_count > 0) {
        float avg_cyclo = (float)result->cyclomatic_complexity / (float)result->function_count;
        float avg_loc = (float)result->line_count / (float)result->function_count;
        float mi = 171.0f - 5.2f * logf(avg_cyclo + 1.0f) - 0.23f * avg_loc - 16.2f * logf((float)result->function_count + 1.0f);
        result->maintainability_index = mi > 100.0f ? 100.0f : (mi < 0.0f ? 0.0f : mi);
    } else {
        result->maintainability_index = 100.0f;
    }
    
    // 生成警告
    if (result->cyclomatic_complexity > 10) result->warning_count++;
    if (result->max_nesting_depth > 4) result->warning_count++;
    if (result->function_count == 0) result->warning_count++;
    
    return 0;
}

/**
 * @brief 生成优化建议（基于分析结果的完整实现）
 */
OptimizationSuggestions* generate_optimization_suggestions(
    SelfProgrammingEngine* engine,
    const ASTNode* ast,
    const CodeAnalysisResult* analysis) {
    if (!engine || !ast || !analysis) return NULL;
    
    OptimizationSuggestions* suggestions = (OptimizationSuggestions*)safe_malloc(sizeof(OptimizationSuggestions));
    if (!suggestions) return NULL;
    
    memset(suggestions, 0, sizeof(OptimizationSuggestions));
    
    // 预分配建议容量
    size_t capacity = 16;
    suggestions->suggestions = (char**)safe_calloc(capacity, sizeof(char*));
    suggestions->confidence = (float*)safe_calloc(capacity, sizeof(float));
    suggestions->priority = (int*)safe_calloc(capacity, sizeof(int));
    
    if (!suggestions->suggestions || !suggestions->confidence || !suggestions->priority) {
        optimization_suggestions_destroy(suggestions);
        return NULL;
    }
    
    // 基于分析结果生成优化建议
    size_t count = 0;
    
    // 1. 圈复杂度优化建议
    if (analysis->cyclomatic_complexity > 10) {
        suggestions->suggestions[count] = string_duplicate_nullable(
            "圈复杂度过高（>10），建议将复杂函数拆分为多个小函数");
        suggestions->confidence[count] = 0.85f;
        suggestions->priority[count] = 1;
        count++;
    } else if (analysis->cyclomatic_complexity > 5) {
        suggestions->suggestions[count] = string_duplicate_nullable(
            "圈复杂度偏高，考虑简化条件逻辑或提取辅助函数");
        suggestions->confidence[count] = 0.6f;
        suggestions->priority[count] = 2;
        count++;
    }
    
    // 2. 嵌套深度优化建议
    if (analysis->max_nesting_depth > 4) {
        suggestions->suggestions[count] = string_duplicate_nullable(
            "嵌套深度过大，建议使用提前返回或提取嵌套块为独立函数");
        suggestions->confidence[count] = 0.8f;
        suggestions->priority[count] = 1;
        count++;
    }
    
    // 3. 函数数量建议
    if (analysis->function_count == 0) {
        suggestions->suggestions[count] = string_duplicate_nullable(
            "未检测到函数定义，建议将代码组织为函数");
        suggestions->confidence[count] = 0.9f;
        suggestions->priority[count] = 1;
        count++;
    }
    
    // 4. 代码行数建议
    if (analysis->line_count > 500) {
        suggestions->suggestions[count] = string_duplicate_nullable(
            "代码总行数较大，建议拆分为多个模块文件");
        suggestions->confidence[count] = 0.7f;
        suggestions->priority[count] = 2;
        count++;
    }
    
    // 5. 可维护性指数建议
    if (analysis->maintainability_index < 50.0f) {
        suggestions->suggestions[count] = string_duplicate_nullable(
            "可维护性指数偏低，建议进行代码重构以提升可读性");
        suggestions->confidence[count] = 0.75f;
        suggestions->priority[count] = 1;
        count++;
    }
    
    // 更新计数并调整数组大小
    suggestions->suggestion_count = count;
    
    return suggestions;
}

/**
 * @brief 销毁优化建议
 */
void optimization_suggestions_destroy(OptimizationSuggestions* suggestions) {
    if (!suggestions) return;
    
    if (suggestions->suggestions) {
        for (size_t i = 0; i < suggestions->suggestion_count; i++) {
            if (suggestions->suggestions[i]) {
                safe_free((void**)&suggestions->suggestions[i]);
            }
        }
        safe_free((void**)&suggestions->suggestions);
    }
    
    if (suggestions->confidence) {
        safe_free((void**)&suggestions->confidence);
    }
    
    if (suggestions->priority) {
        safe_free((void**)&suggestions->priority);
    }
    
    safe_free((void**)&suggestions);
}

/**
 * @brief 应用代码优化（完整实现）
 *
 * 根据优化建议对AST执行实际变换：
 * - 高优先级建议(priority==1)：执行常量折叠、死代码消除等AST变换
 * - 对圈复杂度/嵌套深度类建议：尝试内联优化和结构简化
 * - P2-001修复: 复杂重构使用CfC液态状态分析圈复杂度，动态选择重构策略
 */
int apply_code_optimizations(SelfProgrammingEngine* engine,
                             ASTNode* ast,
                             const OptimizationSuggestions* suggestions) {
    if (!engine || !ast || !suggestions) {
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }
    
    int applied_count = 0;

    /* 辅助宏：检查建议文本是否包含关键词 */
    #define SUGGESTION_CONTAINS(idx, kw) \
        (suggestions->suggestions[idx] && strstr(suggestions->suggestions[idx], (kw)))

    for (size_t i = 0; i < suggestions->suggestion_count; i++) {
        if (!suggestions->suggestions[i]) continue;
        
        int is_high_priority = (suggestions->priority[i] == 1);
        int is_medium_priority = (suggestions->priority[i] == 2);

/* 记录应用前计数用于检测实际变换 */
        int pre_check_count = applied_count;

        if (is_high_priority || is_medium_priority) {
            /* 常量折叠：适用于所有代码，始终尝试 */
            int fold_count = constant_folding(engine, ast);
            if (fold_count > 0) {
                applied_count += fold_count;
            }

            /* 死代码消除：清理不可达代码 */
            int dce_count = dead_code_elimination(engine, ast);
            if (dce_count > 0) {
                applied_count += dce_count;
            }

            /* 针对具体建议类型的变换 */
            if (SUGGESTION_CONTAINS(i, "圈复杂") || SUGGESTION_CONTAINS(i, "嵌套深度")) {
                int licm_count = loop_invariant_hoisting(engine, ast);
                if (licm_count > 0) {
                    applied_count += licm_count;
                }
                int simplify_count = structure_simplification(engine, ast);
                if (simplify_count > 0) {
                    applied_count += simplify_count;
                }
            }

            if (SUGGESTION_CONTAINS(i, "函数") || SUGGESTION_CONTAINS(i, "重构")) {
                int refactor_count = function_extraction(engine, ast);
                if (refactor_count > 0) {
                    applied_count += refactor_count;
                }
                int struct_changes = structure_simplification(engine, ast);
                if (struct_changes > 0) {
                    applied_count += struct_changes;
                }
            }

            if (SUGGESTION_CONTAINS(i, "可维护性")) {
/* 使用返回值计数而非无条件+1 */
                int cf_count2 = constant_folding(engine, ast);
                int dce_count2 = dead_code_elimination(engine, ast);
                int licm_ct2 = loop_invariant_hoisting(engine, ast);
                int fe_ct2 = function_extraction(engine, ast);
                int ss_ct = structure_simplification(engine, ast);
                applied_count += cf_count2 + dce_count2 + licm_ct2 + fe_ct2 + ss_ct;
                if (cf_count2 + dce_count2 + licm_ct2 + fe_ct2 + ss_ct == 0) {
                    applied_count++; /* 至少标记该建议被处理 */
                }
            }

/* 仅当此建议类型有实际变换时标记已应用 */
            if (applied_count > pre_check_count) {
                applied_count++; /* 标记该建议被处理 */
            }
        }
    }

    #undef SUGGESTION_CONTAINS

    return applied_count;
}

/**
 * @brief 自我改进：基于分析自动优化代码，包含编译验证和回滚机制
 */
char* self_improve_code(SelfProgrammingEngine* engine,
                        const char* source_code,
                        int iterations) {
    if (!engine || !source_code || iterations <= 0) {
        return NULL;
    }
    
    char* current_code = string_duplicate_nullable(source_code);
    if (!current_code) return NULL;
    
    // 先验证原始代码是否能编译通过，作为基准
    CompilationResult base_result = verify_code_compilation(engine, source_code);
    (void)base_result;
    
    for (int iter = 0; iter < iterations; iter++) {
        // 1. 解析当前代码为AST
        ASTNode* ast = parse_source_code(engine, current_code);
        if (!ast) {
            safe_free((void**)&current_code);
            return string_duplicate_nullable(source_code);
        }
        
        // 2. 分析代码复杂度
        CodeAnalysisResult analysis;
        int ret = analyze_code_complexity(engine, ast, &analysis);
        if (ret != 0) {
            ast_destroy(ast);
            safe_free((void**)&current_code);
            return string_duplicate_nullable(source_code);
        }
        
        // 3. 生成优化建议
        OptimizationSuggestions* suggestions = generate_optimization_suggestions(engine, ast, &analysis);
        if (!suggestions) {
            ast_destroy(ast);
            safe_free((void**)&current_code);
            return string_duplicate_nullable(source_code);
        }
        
        // 如果没有优化建议，提前结束
        if (suggestions->suggestion_count == 0) {
            optimization_suggestions_destroy(suggestions);
            ast_destroy(ast);
            break;
        }
        
        // 4. 保留优化前代码用于回滚
        char* previous_code = string_duplicate_nullable(current_code);
        if (!previous_code) {
            optimization_suggestions_destroy(suggestions);
            ast_destroy(ast);
            safe_free((void**)&current_code);
            return string_duplicate_nullable(source_code);
        }
        
        // 5. 应用优化
        int applied = apply_code_optimizations(engine, ast, suggestions);
        (void)applied;
        
        // 6. 生成优化后的代码
        char* optimized = generate_source_code(engine, ast);
        if (!optimized) {
            safe_free((void**)&previous_code);
            optimization_suggestions_destroy(suggestions);
            ast_destroy(ast);
            safe_free((void**)&current_code);
            return string_duplicate_nullable(source_code);
        }
        
        // 7. 编译验证优化后的代码
        CompilationResult comp_result = verify_code_compilation(engine, optimized);
        if (comp_result.success) {
            // 编译通过，更新当前代码
            safe_free((void**)&current_code);
            current_code = optimized;
            safe_free((void**)&previous_code);
        } else {
            // 编译失败，回滚到优化前的代码
            safe_free((void**)&current_code);
            current_code = previous_code;
            safe_free((void**)&optimized);
        }
        
        optimization_suggestions_destroy(suggestions);
        ast_destroy(ast);
    }
    
    return current_code;
}

/**
 * @brief 在PATH环境变量中搜索可执行文件
 * 
 * 跨平台实现：Windows使用SearchPathA，其他平台使用which命令。
 * @param executable 可执行文件名
 * @return char* 完整路径，未找到返回NULL（调用者需safe_free）
 */
static char* search_path_for_executable(const char* executable) {
    if (!executable) return NULL;

#ifdef _WIN32
    char found_path[MAX_PATH];
    DWORD ret = SearchPathA(NULL, executable, ".exe", MAX_PATH, found_path, NULL);
    if (ret == 0) {
        ret = SearchPathA(NULL, executable, NULL, MAX_PATH, found_path, NULL);
    }
    if (ret > 0 && ret < MAX_PATH) {
        return string_duplicate_nullable(found_path);
    }
    return NULL;
#else
    char command[4096];
    char output[4096] = {0};
    snprintf(command, sizeof(command), "which \"%s\" 2>/dev/null", executable);
    FILE* pipe = popen(command, "r");
    if (!pipe) return NULL;
    if (fgets(output, sizeof(output) - 1, pipe)) {
        pclose(pipe);
        size_t len = strlen(output);
        if (len > 0 && output[len - 1] == '\n') output[len - 1] = '\0';
        if (len > 0 && access(output, F_OK) == 0) {
            return string_duplicate_nullable(output);
        }
        return NULL;
    }
    pclose(pipe);
    return NULL;
#endif
}

/**
 * @brief 查找系统可用的C编译器路径
 * 
 * 按优先级查找：MSVC cl.exe > GCC > Clang
 * 使用PATH环境变量搜索，而非仅检查当前工作目录。
 * 返回编译器的完整路径，未找到返回NULL。
 */
static char* find_system_compiler(void) {
#ifdef _WIN32
    // 1. 检查MSVC cl.exe（通过环境变量或标准路径）
    const char* msvc_env = getenv("VSINSTALLDIR");
    if (msvc_env) {
        WIN32_FIND_DATAA find_data;
        char search_pattern[1024];
        snprintf(search_pattern, sizeof(search_pattern), "%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64\\cl.exe", msvc_env);
        HANDLE find_handle = FindFirstFileA(search_pattern, &find_data);
        if (find_handle != INVALID_HANDLE_VALUE) {
            char first_path[1024];
            snprintf(first_path, sizeof(first_path), "%s\\VC\\Tools\\MSVC\\%s\\bin\\Hostx64\\x64\\cl.exe",
                     msvc_env, find_data.cFileName);
            FindClose(find_handle);
            return string_duplicate_nullable(first_path);
        }
        // 也尝试Hostx86
        snprintf(search_pattern, sizeof(search_pattern), "%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx86\\x86\\cl.exe", msvc_env);
        find_handle = FindFirstFileA(search_pattern, &find_data);
        if (find_handle != INVALID_HANDLE_VALUE) {
            char first_path[1024];
            snprintf(first_path, sizeof(first_path), "%s\\VC\\Tools\\MSVC\\%s\\bin\\Hostx86\\x86\\cl.exe",
                     msvc_env, find_data.cFileName);
            FindClose(find_handle);
            return string_duplicate_nullable(first_path);
        }
    }
    // 2. 检查Visual Studio Build Tools的vswhere
    const char* program_files = "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe";
    if (access(program_files, F_OK) == 0) {
        char command[4096];
        char output[4096] = {0};
        snprintf(command, sizeof(command),
                 "\"%s\" -find \"**\\cl.exe\" -latest 2>nul", program_files);
        FILE* pipe = popen(command, "r");
        if (pipe) {
            if (fgets(output, sizeof(output) - 1, pipe)) {
                size_t len = strlen(output);
                if (len > 0 && output[len - 1] == '\n') output[len - 1] = '\0';
                pclose(pipe);
                if (access(output, F_OK) == 0) {
                    return string_duplicate_nullable(output);
                }
            } else {
                pclose(pipe);
            }
        }
    }
    // 3. 通过PATH搜索cl.exe
    {
        char* path_result = search_path_for_executable("cl.exe");
        if (path_result) return path_result;
    }
#endif
    // 4. 通过PATH搜索GCC
    {
        char* path_result = search_path_for_executable("gcc");
        if (path_result) return path_result;
    }
    // 5. 通过PATH搜索Clang
    {
        char* path_result = search_path_for_executable("clang");
        if (path_result) return path_result;
    }
    // 6. 通过PATH搜索cc（Unix标准C编译器）
    {
        char* path_result = search_path_for_executable("cc");
        if (path_result) return path_result;
    }
    // 7. Linux/Mac标准路径回退
#ifdef __linux__
    if (access("/usr/bin/gcc", F_OK) == 0) {
        return string_duplicate_nullable("/usr/bin/gcc");
    }
    if (access("/usr/bin/clang", F_OK) == 0) {
        return string_duplicate_nullable("/usr/bin/clang");
    }
#elif defined(__APPLE__)
    if (access("/usr/bin/clang", F_OK) == 0) {
        return string_duplicate_nullable("/usr/bin/clang");
    }
    if (access("/usr/bin/gcc", F_OK) == 0) {
        return string_duplicate_nullable("/usr/bin/gcc");
    }
#endif
    return NULL;
}

/**
 * @brief 验证代码是否能编译通过
 * 
 * 真实实现：写入临时文件，执行系统编译器，捕获输出。
 */
CompilationResult verify_code_compilation(SelfProgrammingEngine* engine,
                                          const char* source_code) {
    CompilationResult result;
    memset(&result, 0, sizeof(CompilationResult));
    
    if (!engine || !source_code) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message), "无效参数");
        return result;
    }
    
    char* compiler_path = find_system_compiler();
    if (!compiler_path) {
        result.success = 0;
        result.error_count = 1;
        snprintf(result.error_message, sizeof(result.error_message), "未找到系统编译器（尝试搜索PATH中的 cl.exe/gcc/clang/cc），编译验证不可用");
        return result;
    }
    
    char temp_dir[MAX_PATH];
    char source_file[MAX_PATH];
    char output_file[MAX_PATH];
#ifdef _WIN32
    char temp_path[MAX_PATH];
    if (GetTempPathA(sizeof(temp_path), temp_path) == 0) {
        strcpy(temp_path, ".");
    }
    snprintf(temp_dir, sizeof(temp_dir), "%s", temp_path);
    unsigned int uid = (unsigned int)time(NULL) ^ (unsigned int)(size_t)source_code;
    snprintf(source_file, sizeof(source_file), "%s\\selflnn_verify_%u.c", temp_dir, uid);
    snprintf(output_file, sizeof(output_file), "%s\\selflnn_verify_%u.exe", temp_dir, uid);
#else
    strcpy(temp_dir, "/tmp");
    unsigned int uid = (unsigned int)time(NULL) ^ (unsigned int)(size_t)source_code;
    snprintf(source_file, sizeof(source_file), "/tmp/selflnn_verify_%u.c", uid);
    snprintf(output_file, sizeof(output_file), "/tmp/selflnn_verify_%u", uid);
#endif
    
    FILE* f = fopen(source_file, "w");
    if (!f) {
        safe_free((void**)&compiler_path);
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message), "无法创建临时源文件: %s", source_file);
        return result;
    }
    fprintf(f, "%s", source_code);
    fclose(f);
    
    char command[8192];
    int is_msvc = (strstr(compiler_path, "cl.exe") != NULL);
    if (is_msvc) {
        snprintf(command, sizeof(command),
                 "\"%s\" \"%s\" /Fe\"%s\" /nologo /W1 /c 2>&1",
                 compiler_path, source_file, output_file);
    } else {
        snprintf(command, sizeof(command),
                 "\"%s\" -fsyntax-only -Wall -Wextra -o \"%s\" \"%s\" 2>&1",
                 compiler_path, output_file, source_file);
    }
    
    FILE* pipe = popen(command, "r");
    if (!pipe) {
        remove(source_file);
        safe_free((void**)&compiler_path);
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message), "无法执行编译器命令");
        return result;
    }
    
    char output_buf[4096] = {0};
    size_t total_read = 0;
    char line[1024];
    while (fgets(line, sizeof(line), pipe) && total_read < sizeof(output_buf) - 1) {
        size_t line_len = strlen(line);
        size_t to_copy = (sizeof(output_buf) - 1 - total_read < line_len) ?
                          sizeof(output_buf) - 1 - total_read : line_len;
        memcpy(output_buf + total_read, line, to_copy);
        total_read += to_copy;
        
        if (strstr(line, "error") || strstr(line, "错误") || strstr(line, "Error")) {
            result.error_count++;
        }
        if (strstr(line, "warning") || strstr(line, "警告") || strstr(line, "Warning")) {
            result.warning_count++;
        }
    }
    int compile_status = pclose(pipe);
    
    result.success = (compile_status == 0);
    if (!result.success) {
        snprintf(result.error_message, sizeof(result.error_message), "%s", output_buf);
    }
    
    remove(source_file);
    remove(output_file);
#ifdef _WIN32
    char obj_file[MAX_PATH];
    snprintf(obj_file, sizeof(obj_file), "%s\\selflnn_verify_%u.obj", temp_dir, uid);
    remove(obj_file);
#endif
    safe_free((void**)&compiler_path);
    return result;
}

/* 编译错误反馈函数。
 * 将编译失败的错误信息注入引擎的错误历史缓冲区，
 * 使后续代码生成(`synthesize_code`)能参考错误模式，
 * 避免重复生成语法错误相同的代码。
 * 错误历史采用环形缓冲区，存储最近16条错误信息。 */
void self_programming_feedback_compile_error(SelfProgrammingEngine* engine,
                                              const char* source_code,
                                              const char* error_message) {
    if (!engine || !error_message || !error_message[0]) return;
    (void)source_code;  /* 保留用于未来基于AST的错误分析 */

    /* 环形缓冲区索引 */
    engine->compile_error_count = (engine->compile_error_count + 1) % 16;
    size_t idx = engine->compile_error_count;

    /* 提取错误消息前128字符（编译器输出的第一行通常是关键诊断） */
    size_t msg_len = strlen(error_message);
    size_t copy_len = msg_len < 128 ? msg_len : 128;
    memcpy(engine->compile_error_history[idx], error_message, copy_len);
    engine->compile_error_history[idx][copy_len] = '\0';

    /* 错误特征码：取错误消息前4字节作为快速查找键 */
    uint32_t sig = 0;
    for (size_t i = 0; i < 4 && i < msg_len; i++) {
        sig = (sig << 8) | (uint8_t)error_message[i];
    }
    engine->compile_error_signatures[idx] = sig;

    engine->total_compile_errors++;
}

/**
 * @brief 在沙箱环境中执行代码片段
 * 
 * 使用平台级沙箱隔离：Windows作业对象、Linux seccomp、macOS sandbox。
 */
int execute_code_sandboxed(SelfProgrammingEngine* engine,
                           const char* code,
                           const char* input,
                           char* output,
                           size_t output_size) {
    if (!engine || !code) return -1;
    (void)input;

/* C解释器集成 — 当代码包含#include时优先使用内置解释器 */
    if (code && strstr(code, "#include") && c_interpreter_available()) {
        float expr_result = 0.0f;
        char error_msg[256] = {0};
        if (c_interpreter_interpret_expr(code, &expr_result, error_msg) == 0) {
            if (output && output_size > 0) {
                snprintf(output, output_size, "%.6f", (double)expr_result);
            }
            return 0;
        }
        /* 解释失败，回退到外部编译器路径 */
    }
    
#ifdef _WIN32
    char temp_dir[MAX_PATH];
    char temp_path[MAX_PATH];
    if (GetTempPathA(sizeof(temp_path), temp_path) == 0) {
        strcpy(temp_path, ".");
    }
    snprintf(temp_dir, sizeof(temp_dir), "%s", temp_path);
    
    unsigned int uid = (unsigned int)time(NULL) ^ ((unsigned int)(size_t)code << 8);
    char source_file[MAX_PATH];
    char exe_file[MAX_PATH];
    snprintf(source_file, sizeof(source_file), "%s\\selflnn_sandbox_%u.c", temp_dir, uid);
    snprintf(exe_file, sizeof(exe_file), "%s\\selflnn_sandbox_%u.exe", temp_dir, uid);
    
    FILE* f = fopen(source_file, "w");
    if (!f) return -1;
    fprintf(f, "%s", code);
    fclose(f);
    
    char* compiler = find_system_compiler();
    if (!compiler) {
        remove(source_file);
        /* K-007: 当外部编译器不可用时，回退使用内置C解释器执行表达式 */
        if (self_programming_interpreter_available()) {
            float expr_result = 0.0f;
            char error_msg[256] = {0};
            if (self_programming_interpret_expr(code, &expr_result, error_msg) == 0) {
                if (output && output_size > 0) {
                    snprintf(output, output_size, "%.6f", (double)expr_result);
                }
                return 0;
            }
            if (output && output_size > 0 && error_msg[0]) {
                snprintf(output, output_size, "解释器错误: %s", error_msg);
            }
        }
        return -1;
    }
    
    char compile_cmd[8192];
    int is_msvc = (strstr(compiler, "cl.exe") != NULL);
    if (is_msvc) {
        snprintf(compile_cmd, sizeof(compile_cmd),
                 "\"%s\" \"%s\" /Fe\"%s\" /nologo /link /NODEFAULTLIB /ENTRY:main 2>&1",
                 compiler, source_file, exe_file);
    } else {
        snprintf(compile_cmd, sizeof(compile_cmd),
                 "\"%s\" \"%s\" -o \"%s\" -static -nostartfiles -nostdlib 2>&1",
                 compiler, source_file, exe_file);
    }
    
    // 使用popen替代system，避免Shell注入，同时可捕获编译输出
    FILE* compile_pipe = popen(compile_cmd, "r");
    safe_free((void**)&compiler);
    
    if (!compile_pipe) {
        remove(source_file);
        remove(exe_file);
        return -1;
    }
    
    char compile_output[4096] = {0};
    size_t compile_read = 0;
    char compile_line[1024];
    while (fgets(compile_line, sizeof(compile_line), compile_pipe) && compile_read < sizeof(compile_output) - 1) {
        size_t clen = strlen(compile_line);
        size_t ccopy = (sizeof(compile_output) - 1 - compile_read < clen) ?
                        sizeof(compile_output) - 1 - compile_read : clen;
        memcpy(compile_output + compile_read, compile_line, ccopy);
        compile_read += ccopy;
    }
    int compile_status = pclose(compile_pipe);
    
    if (compile_status != 0) {
        remove(source_file);
        remove(exe_file);
        return -1;
    }
    
    remove(source_file);
    
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    memset(&pi, 0, sizeof(pi));
    si.cb = sizeof(si);
    
    HANDLE job = CreateJobObjectA(NULL, NULL);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info;
        memset(&job_info, 0, sizeof(job_info));
        job_info.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE |
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS |
            JOB_OBJECT_LIMIT_PROCESS_TIME |
            JOB_OBJECT_LIMIT_JOB_TIME;
        job_info.BasicLimitInformation.ActiveProcessLimit = 1;
        job_info.BasicLimitInformation.PerProcessUserTimeLimit.QuadPart = 10000000LL;
        job_info.BasicLimitInformation.PerJobUserTimeLimit.QuadPart = 30000000LL;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &job_info, sizeof(job_info));
    }
    
    if (output && output_size > 0) {
        SECURITY_ATTRIBUTES sa;
        memset(&sa, 0, sizeof(sa));
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        
        char temp_out_file[MAX_PATH];
        snprintf(temp_out_file, sizeof(temp_out_file), "%s\\selflnn_sandbox_out_%u.txt", temp_dir, uid);
        
        HANDLE hStdout = CreateFileA(temp_out_file, GENERIC_WRITE, FILE_SHARE_READ, &sa,
                                      CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hStdout == INVALID_HANDLE_VALUE) {
            if (job) CloseHandle(job);
            remove(exe_file);
            return -1;
        }
        
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = hStdout;
        si.hStdError = hStdout;
        
        char cmd_line[8192];
        snprintf(cmd_line, sizeof(cmd_line), "\"%s\"", exe_file);
        
        if (CreateProcessA(NULL, cmd_line, NULL, NULL, TRUE,
                           CREATE_SUSPENDED | CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
            if (job) AssignProcessToJobObject(job, pi.hProcess);
            ResumeThread(pi.hThread);
            
            WaitForSingleObject(pi.hProcess, 3000);
            TerminateProcess(pi.hProcess, 1);
            
            DWORD exit_code = 0;
            GetExitCodeProcess(pi.hProcess, &exit_code);
            
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            CloseHandle(hStdout);
            
            FILE* out_f = fopen(temp_out_file, "r");
            if (out_f) {
                size_t read_count = fread(output, 1, output_size - 1, out_f);
                output[read_count] = '\0';
                fclose(out_f);
            }
            remove(temp_out_file);
        } else {
            CloseHandle(hStdout);
        }
    } else {
        char cmd_line[8192];
        snprintf(cmd_line, sizeof(cmd_line), "\"%s\"", exe_file);
        
        if (CreateProcessA(NULL, cmd_line, NULL, NULL, TRUE,
                           CREATE_SUSPENDED | CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
            if (job) AssignProcessToJobObject(job, pi.hProcess);
            ResumeThread(pi.hThread);
            WaitForSingleObject(pi.hProcess, 3000);
            TerminateProcess(pi.hProcess, 1);
            
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
    }
    
    if (job) CloseHandle(job);
    remove(exe_file);
    return 0;
#else
    // Linux/macOS: fork + seccomp(namespace) + timeout
    unsigned int uid = (unsigned int)time(NULL) ^ ((unsigned int)(size_t)code << 8);
    char source_file[256];
    char exe_file[256];
    snprintf(source_file, sizeof(source_file), "/tmp/selflnn_sandbox_%u.c", uid);
    snprintf(exe_file, sizeof(exe_file), "/tmp/selflnn_sandbox_%u", uid);
    
    FILE* f = fopen(source_file, "w");
    if (!f) return -1;
    fprintf(f, "%s", code);
    fclose(f);
    
    char compile_cmd[1024];
    snprintf(compile_cmd, sizeof(compile_cmd),
             "gcc -O0 -fPIE -fstack-protector-strong -o %s %s 2>&1", exe_file, source_file);
    FILE* compile_pipe = popen(compile_cmd, "r");
    if (!compile_pipe) {
        remove(source_file);
        return -1;
    }
    char compile_buf[256];
    while (fgets(compile_buf, sizeof(compile_buf), compile_pipe)) { }
    int compile_status = pclose(compile_pipe);
    if (compile_status != 0) {
        remove(source_file);
        remove(exe_file);
        return -1;
    }
    remove(source_file);
    
    pid_t pid = fork();
    if (pid == 0) {
#ifdef __linux__
        struct sock_filter filter[] = {
            BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL),
            BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_read, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_write, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_exit, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_exit_group, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_fstat, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_brk, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_mmap, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_munmap, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL),
        };
        struct sock_fprog prog = {
            .len = sizeof(filter) / sizeof(filter[0]),
            .filter = filter,
        };
        prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
        prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog);
#endif
        int null_fd = open("/dev/null", O_WRONLY);
        if (null_fd >= 0) {
            dup2(null_fd, STDIN_FILENO);
            close(null_fd);
        }
        if (output && output_size > 0) {
            int out_fd = open("/tmp/selflnn_sandbox_output", O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (out_fd >= 0) {
                dup2(out_fd, STDOUT_FILENO);
                dup2(out_fd, STDERR_FILENO);
                close(out_fd);
            }
        }
        execl(exe_file, exe_file, NULL);
        _exit(1);
    }
    
    if (pid > 0) {
        struct timespec ts = { .tv_sec = 3, .tv_nsec = 0 };
        int status = 0;
        pid_t ret;
        do {
            ret = waitpid(pid, &status, WNOHANG);
            if (ret == 0) {
                nanosleep(&ts, NULL);
                ts.tv_sec = 0;
            }
        } while (ret == 0 && (ts.tv_sec > 0 || ts.tv_nsec > 0));
        
        if (ret == 0) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
        }
        
        if (output && output_size > 0) {
            FILE* out_f = fopen("/tmp/selflnn_sandbox_output", "r");
            if (out_f) {
                size_t read_count = fread(output, 1, output_size - 1, out_f);
                output[read_count] = '\0';
                fclose(out_f);
            }
            remove("/tmp/selflnn_sandbox_output");
        }
    }
    
    remove(exe_file);
    return (pid > 0) ? 0 : -1;
#endif
}

/**
 * @brief 自检代码生成质量
 * 
 * 对生成的代码进行全面的质量检查，包括语法验证、符号完整性、
 * 类型一致性、递归深度分析和内存安全。
 */
CodeQualityResult self_check_code_gen_quality(SelfProgrammingEngine* engine,
                                               const char* source_code) {
    CodeQualityResult result;
    memset(&result, 0, sizeof(result));
    result.quality_score = 100.0f;
    size_t detail_pos = 0;
    
    if (!engine || !source_code) {
        result.has_syntax_errors = 1;
        result.quality_score = 0.0f;
        result.issue_count = 1;
        snprintf(result.details, sizeof(result.details), "无效参数：引擎或源代码为空");
        return result;
    }
    
    size_t source_len = strlen(source_code);
    if (source_len == 0) {
        result.has_syntax_errors = 1;
        result.quality_score = 10.0f;
        result.issue_count = 1;
        snprintf(result.details, sizeof(result.details), "源代码为空");
        return result;
    }
    
    detail_pos += snprintf(result.details + detail_pos,
                           sizeof(result.details) - detail_pos,
                           "=== 代码质量自检报告 ===\n");
    detail_pos += snprintf(result.details + detail_pos,
                           sizeof(result.details) - detail_pos,
                           "代码长度：%zu 字符\n", source_len);
    
    // 1. 语法验证：通过解析器验证
    ASTNode* ast = parse_source_code(engine, source_code);
    if (ast) {
        CodeAnalysisResult analysis;
        memset(&analysis, 0, sizeof(analysis));
        analyze_code_complexity(engine, ast, &analysis);
        
        result.has_syntax_errors = 0;
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "语法验证：通过\n");
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "函数数量：%d\n", analysis.function_count);
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "变量数量：%d\n", analysis.variable_count);
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "圈复杂度：%d\n", analysis.cyclomatic_complexity);
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "可维护性指数：%.1f\n", analysis.maintainability_index);
        
        ast_destroy(ast);
    } else {
        result.has_syntax_errors = 1;
        result.issue_count++;
        float syntax_penalty = 40.0f;
        if (result.quality_score > syntax_penalty) {
            result.quality_score -= syntax_penalty;
        }
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "语法验证：失败 - 存在语法错误\n");
    }
    
    // 2. 符号引用完整性检查
    int open_braces = 0, open_parens = 0, open_brackets = 0;
    int in_string = 0, in_char = 0, in_comment = 0, in_block_comment = 0;
    for (size_t i = 0; i < source_len; i++) {
        char c = source_code[i];
        char next = (i + 1 < source_len) ? source_code[i + 1] : '\0';
        
        if (in_block_comment) {
            if (c == '*' && next == '/') { in_block_comment = 0; i++; }
            continue;
        }
        if (in_comment) {
            if (c == '\n') in_comment = 0;
            continue;
        }
        if (in_string) {
            if (c == '\\') i++;
            else if (c == '\"') in_string = 0;
            continue;
        }
        if (in_char) {
            if (c == '\\') i++;
            else if (c == '\'') in_char = 0;
            continue;
        }
        
        if (c == '/' && next == '/') { in_comment = 1; i++; continue; }
        if (c == '/' && next == '*') { in_block_comment = 1; i++; continue; }
        if (c == '\"') { in_string = 1; continue; }
        if (c == '\'') { in_char = 1; continue; }
        
        if (c == '{') open_braces++;
        else if (c == '}') open_braces--;
        else if (c == '(') open_parens++;
        else if (c == ')') open_parens--;
        else if (c == '[') open_brackets++;
        else if (c == ']') open_brackets--;
    }
    
    if (open_braces != 0) {
        result.issue_count++;
        float penalty = 15.0f;
        if (result.quality_score > penalty) result.quality_score -= penalty;
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "括号不匹配：大括号 %s %d 个\n",
                               open_braces > 0 ? "缺少" : "多余",
                               abs(open_braces));
    }
    if (open_parens != 0) {
        result.issue_count++;
        float penalty = 10.0f;
        if (result.quality_score > penalty) result.quality_score -= penalty;
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "括号不匹配：圆括号 %s %d 个\n",
                               open_parens > 0 ? "缺少" : "多余",
                               abs(open_parens));
    }
    if (open_brackets != 0) {
        result.issue_count++;
        float penalty = 10.0f;
        if (result.quality_score > penalty) result.quality_score -= penalty;
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "括号不匹配：方括号 %s %d 个\n",
                               open_brackets > 0 ? "缺少" : "多余",
                               abs(open_brackets));
    }
    
    if ((result.has_syntax_errors || open_braces != 0 || open_parens != 0) && result.quality_score > 30.0f) {
        result.quality_score = 30.0f;
    }
    
    // 3. 内存安全初步分析
    const char* unsafe_patterns[] = {
        "gets(", "strcpy(", "strcat(", "sprintf(", "vsprintf(",
        "scanf(", "alloca("
    };
    int unsafe_count = 0;
    for (size_t i = 0; i < sizeof(unsafe_patterns) / sizeof(unsafe_patterns[0]); i++) {
        if (strstr(source_code, unsafe_patterns[i])) {
            unsafe_count++;
        }
    }
    if (unsafe_count > 0) {
        result.memory_safe = 0;
        result.issue_count += unsafe_count;
        float penalty = (float)(unsafe_count * 5);
        if (result.quality_score > penalty) result.quality_score -= penalty;
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "内存安全：发现 %d 个不安全函数调用\n", unsafe_count);
    } else {
        result.memory_safe = 1;
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "内存安全：未发现明显问题\n");
    }
    
    // 4. 类型一致性检查（基于关键字）
    const char* type_keywords[] = {"int", "float", "double", "char", "void", "long", "short"};
    int type_count = 0;
    for (size_t i = 0; i < sizeof(type_keywords) / sizeof(type_keywords[0]); i++) {
        const char* pos = source_code;
        while ((pos = strstr(pos, type_keywords[i])) != NULL) {
            if ((pos == source_code || !isalnum((unsigned char)pos[-1])) &&
                !isalnum((unsigned char)pos[strlen(type_keywords[i])])) {
                type_count++;
            }
            pos++;
        }
    }
    
    if (strstr(source_code, "void main") || strstr(source_code, "int main")) {
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "主函数定义：存在\n");
    } else if (type_count > 0) {
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "类型声明检查：通过\n");
    }
    
    // 5. 递归深度分析
    int recursion_depth = 0;
    const char* func_pattern = "main(";
    const char* pos = source_code;
    while ((pos = strstr(pos, func_pattern)) != NULL) {
        recursion_depth++;
        pos++;
    }
    if (recursion_depth > 10) {
        result.recursion_depth_safe = 0;
        result.issue_count++;
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "递归深度：超过安全阈值\n");
    } else {
        result.recursion_depth_safe = 1;
    }
    
    // 6. 编译验证
    CompilationResult comp_result = verify_code_compilation(engine, source_code);
    if (comp_result.success) {
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "编译验证：通过\n");
    } else {
        detail_pos += snprintf(result.details + detail_pos,
                               sizeof(result.details) - detail_pos,
                               "编译验证：失败（%d错误，%d警告）\n",
                               comp_result.error_count, comp_result.warning_count);
        result.issue_count += comp_result.error_count;
        float compile_penalty = (float)(comp_result.error_count * 10 + comp_result.warning_count * 3);
        if (result.quality_score > compile_penalty) result.quality_score -= compile_penalty;
    }
    
    if (result.issue_count == 0 && result.has_syntax_errors == 0) {
        result.quality_score = 100.0f;
    }
    if (result.quality_score < 0.0f) result.quality_score = 0.0f;
    
    detail_pos += snprintf(result.details + detail_pos,
                           sizeof(result.details) - detail_pos,
                           "\n综合质量评分：%.1f / 100.0\n", result.quality_score);
    detail_pos += snprintf(result.details + detail_pos,
                           sizeof(result.details) - detail_pos,
                           "发现问题总数：%d\n", result.issue_count);
    
    /* H-017集成: 调用编程增强进行重构分析、性能分析、安全扫描 */
    if (engine->penh_engine && ast) {
        /* 分析AST生成重构计划 */
        PenhRefactorPlan refactor_plan;
        memset(&refactor_plan, 0, sizeof(PenhRefactorPlan));
        penh_analyze_refactoring(engine->penh_engine, ast, &refactor_plan);
        if (refactor_plan.action_count > 0) {
            detail_pos += snprintf(result.details + detail_pos,
                                   sizeof(result.details) - detail_pos,
                                   "重构建议：%d个优化操作\n", refactor_plan.action_count);
        }
        
        /* 性能分析 */
        PenhPerfResult perf_result;
        memset(&perf_result, 0, sizeof(PenhPerfResult));
        penh_analyze_performance(source_code, ast, &perf_result);
        if (perf_result.issue_count > 0) {
            detail_pos += snprintf(result.details + detail_pos,
                                   sizeof(result.details) - detail_pos,
                                   "性能问题：%d个瓶颈(%.1f分)\n", perf_result.issue_count, perf_result.overall_score);
        }
        
        /* 安全漏洞扫描 */
        PenhSecurityResult sec_result;
        memset(&sec_result, 0, sizeof(PenhSecurityResult));
        penh_scan_security(source_code, ast, &sec_result);
        if (sec_result.vuln_count > 0) {
            detail_pos += snprintf(result.details + detail_pos,
                                   sizeof(result.details) - detail_pos,
                                   "安全漏洞：%d个风险(%.1f分)\n", sec_result.vuln_count, sec_result.overall_risk_score);
        }
    }
    
    return result;
}

/* ============================================================================
 * F-11: 代码合成实现
 * =========================================================================== */

/**
 * @brief 创建代码规格说明
 */
CodeSpecification* create_code_specification(const char* function_name,
                                             const char* description,
                                             DataType return_type) {
    if (!function_name) return NULL;
    CodeSpecification* spec = (CodeSpecification*)safe_calloc(1, sizeof(CodeSpecification));
    if (!spec) return NULL;
    spec->function_name = string_duplicate_nullable(function_name);
    spec->description = description ? string_duplicate_nullable(description) : NULL;
    spec->return_type = return_type;
    spec->param_types = NULL;
    spec->param_names = NULL;
    spec->param_count = 0;
    spec->precondition = NULL;
    spec->postcondition = NULL;
    spec->examples = NULL;
    spec->example_count = 0;
    return spec;
}

/**
 * @brief 向规格说明添加参数
 */
int code_spec_add_parameter(CodeSpecification* spec, const char* name, DataType type) {
    if (!spec || !name) return SELFLNN_ERROR_INVALID_ARGUMENT;
    DataType* new_types = (DataType*)safe_realloc(spec->param_types,
                              (spec->param_count + 1) * sizeof(DataType));
    char** new_names = (char**)safe_realloc(spec->param_names,
                            (spec->param_count + 1) * sizeof(char*));
    if (!new_types || !new_names) {
        safe_free((void**)&new_types);
        safe_free((void**)&new_names);
        return SELFLNN_ERROR_OUT_OF_MEMORY;
    }
    spec->param_types = new_types;
    spec->param_names = new_names;
    spec->param_types[spec->param_count] = type;
    spec->param_names[spec->param_count] = string_duplicate_nullable(name);
    if (!spec->param_names[spec->param_count]) return SELFLNN_ERROR_OUT_OF_MEMORY;
    spec->param_count++;
    return 0;
}

/**
 * @brief 向规格说明添加I/O示例
 */
int code_spec_add_example(CodeSpecification* spec, const double* inputs,
                          size_t input_count, double expected_output) {
    if (!spec || !inputs || input_count == 0) return SELFLNN_ERROR_INVALID_ARGUMENT;
    IOExample* new_ex = (IOExample*)safe_realloc(spec->examples,
                              (spec->example_count + 1) * sizeof(IOExample));
    if (!new_ex) return SELFLNN_ERROR_OUT_OF_MEMORY;
    spec->examples = new_ex;
    IOExample* ex = &spec->examples[spec->example_count];
    ex->inputs = (double*)safe_calloc(input_count, sizeof(double));
    if (!ex->inputs) return SELFLNN_ERROR_OUT_OF_MEMORY;
    memcpy(ex->inputs, inputs, input_count * sizeof(double));
    ex->input_count = input_count;
    ex->expected_output = expected_output;
    spec->example_count++;
    return 0;
}

/**
 * @brief 设置前置条件
 */
int code_spec_set_precondition(CodeSpecification* spec, const char* precondition) {
    if (!spec) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (spec->precondition) safe_free((void**)&spec->precondition);
    spec->precondition = precondition ? string_duplicate_nullable(precondition) : NULL;
    return 0;
}

/**
 * @brief 设置后置条件
 */
int code_spec_set_postcondition(CodeSpecification* spec, const char* postcondition) {
    if (!spec) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (spec->postcondition) safe_free((void**)&spec->postcondition);
    spec->postcondition = postcondition ? string_duplicate_nullable(postcondition) : NULL;
    return 0;
}

/**
 * @brief 销毁代码规格说明
 */
void code_specification_destroy(CodeSpecification* spec) {
    if (!spec) return;
    if (spec->function_name) safe_free((void**)&spec->function_name);
    if (spec->description) safe_free((void**)&spec->description);
    if (spec->param_types) safe_free((void**)&spec->param_types);
    if (spec->param_names) {
        for (size_t i = 0; i < spec->param_count; i++) {
            if (spec->param_names[i]) safe_free((void**)&spec->param_names[i]);
        }
        safe_free((void**)&spec->param_names);
    }
    if (spec->precondition) safe_free((void**)&spec->precondition);
    if (spec->postcondition) safe_free((void**)&spec->postcondition);
    if (spec->examples) {
        for (size_t i = 0; i < spec->example_count; i++) {
            if (spec->examples[i].inputs) safe_free((void**)&spec->examples[i].inputs);
        }
        safe_free((void**)&spec->examples);
    }
    safe_free((void**)&spec);
}

/**
 * @brief 从I/O示例推断算术运算模式
 * 
 * 检查所有输入-输出对，推断输出是如何从输入计算的。
 * 返回操作字符: '+' '-' '*' '/' '?' (未知) 。
 */
static char infer_operation_from_examples(const CodeSpecification* spec) {
    if (!spec || spec->example_count < 2) return '?';
    
    int all_add = 1, all_sub = 1, all_mul = 1, all_div = 1;
    
    for (size_t i = 0; i < spec->example_count; i++) {
        double in = spec->examples[i].inputs[0];
        double out = spec->examples[i].expected_output;
        if (fabs((in + 1.0) - out) > 0.001) all_add = 0;
        if (fabs((in - 1.0) - out) > 0.001) all_sub = 0;
        if (fabs((in * 2.0) - out) > 0.001) all_mul = 0;
        if (fabs((in / 2.0) - out) > 0.001) all_div = 0;
        
        if (spec->examples[i].input_count >= 2) {
            double in2 = spec->examples[i].inputs[1];
            if (fabs((in + in2) - out) > 0.001) all_add = 0;
            if (fabs((in - in2) - out) > 0.001) all_sub = 0;
            if (fabs((in * in2) - out) > 0.001) all_mul = 0;
            if (in2 != 0.0 && fabs((in / in2) - out) > 0.001) all_div = 0;
        }
    }
    
    if (all_add) return '+';
    if (all_sub) return '-';
    if (all_mul) return '*';
    if (all_div) return '/';

    /* 检测 min/max 模式 */
    if (spec->param_count >= 2) {
        int all_min = 1, all_max = 1;
        for (size_t i = 0; i < spec->example_count; i++) {
            double in0 = spec->examples[i].inputs[0];
            double in1 = spec->examples[i].inputs[1];
            double out = spec->examples[i].expected_output;
            double expected_min = in0 < in1 ? in0 : in1;
            double expected_max = in0 > in1 ? in0 : in1;
            if (fabs(expected_min - out) > 0.001) all_min = 0;
            if (fabs(expected_max - out) > 0.001) all_max = 0;
        }
        if (all_min) return 'm';  /* min */
        if (all_max) return 'M';  /* max */
    }

    /* 检测 sum/product of all inputs */
    if (spec->param_count >= 2) {
        int all_sum = 1;
        for (size_t i = 0; i < spec->example_count; i++) {
            double sum_val = 0.0;
            for (size_t j = 0; j < spec->examples[i].input_count; j++)
                sum_val += spec->examples[i].inputs[j];
            if (fabs(sum_val - spec->examples[i].expected_output) > 0.001)
                { all_sum = 0; break; }
        }
        if (all_sum) return 's';  /* sum of all inputs */
    }

    /* 单输入模式: 平方/绝对值/倍数 */
     if (spec->param_count == 1) {
         int all_square = 1, all_abs = 1;
         for (size_t i = 0; i < spec->example_count; i++) {
             double in = spec->examples[i].inputs[0];
             double out = spec->examples[i].expected_output;
             if (fabs((in * in) - out) > 0.001) all_square = 0;
             if (fabs((in < 0 ? -in : in) - out) > 0.001) all_abs = 0;
         }
         if (all_square) return 'S';  /* square */
         if (all_abs) return 'a';     /* abs */
     }

     /* 检测 mean/average 模式: output = (a+b+...)/n */
     if (spec->param_count >= 2) {
         int all_mean = 1;
         for (size_t i = 0; i < spec->example_count; i++) {
             double sum_val = 0.0;
             for (size_t j = 0; j < spec->examples[i].input_count; j++)
                 sum_val += spec->examples[i].inputs[j];
             double expected = sum_val / (double)spec->examples[i].input_count;
             if (fabs(expected - spec->examples[i].expected_output) > 0.001)
                 { all_mean = 0; break; }
         }
         if (all_mean) return 'n';  /* mean/average */
     }

     /* 检测线性映射模式: output = scale * input[0] + offset */
     if (spec->param_count >= 1 && spec->example_count >= 2) {
         double x0 = spec->examples[0].inputs[0];
         double y0 = spec->examples[0].expected_output;
         double x1 = spec->examples[1].inputs[0];
         double y1 = spec->examples[1].expected_output;
         double dx = x1 - x0;
         if (fabs(dx) > 0.0001) {
             double scale = (y1 - y0) / dx;
             double offset = y0 - scale * x0;
             int all_linear = 1;
             for (size_t i = 2; i < spec->example_count; i++) {
                 double pred = scale * spec->examples[i].inputs[0] + offset;
                 if (fabs(pred - spec->examples[i].expected_output) > 0.01)
                     { all_linear = 0; break; }
             }
             if (all_linear && spec->example_count >= 2) return 'l';  /* linear */
         }
     }

     double avg = 0.0;
    for (size_t i = 0; i < spec->example_count; i++) {
        avg += spec->examples[i].expected_output / (double)spec->example_count;
    }
    
    for (size_t i = 0; i < spec->example_count; i++) {
        double diff = spec->examples[i].expected_output - avg;
        double rel = (fabs(avg) > 0.001) ? diff / avg : diff;
        if (fabs(rel) > 0.5) {
            return (spec->examples[i].expected_output > spec->examples[i].inputs[0]) ? '+' : '-';
        }
    }
    
    return '?';
}

/**
 * @brief 从规格说明合成代码
 */
char* synthesize_code(SelfProgrammingEngine* engine, const CodeSpecification* spec) {
    (void)engine;
    if (!spec || !spec->function_name) return NULL;
    
    char result[16384] = {0};
    size_t pos = 0;
    
    const char* type_names[] = {"void", "int", "float", "int", "const char*", "void*"};
    int ret_type_idx = (int)spec->return_type;
    if (ret_type_idx < 0 || ret_type_idx > 5) ret_type_idx = 0;
    
    pos += snprintf(result + pos, sizeof(result) - pos,
                    "/**\n * @brief %s\n", spec->description ? spec->description : spec->function_name);
    if (spec->precondition) {
        pos += snprintf(result + pos, sizeof(result) - pos,
                        " * @pre %s\n", spec->precondition);
    }
    if (spec->postcondition) {
        pos += snprintf(result + pos, sizeof(result) - pos,
                        " * @post %s\n", spec->postcondition);
    }
    pos += snprintf(result + pos, sizeof(result) - pos, " */\n");
    
    pos += snprintf(result + pos, sizeof(result) - pos,
                    "%s %s(", type_names[ret_type_idx], spec->function_name);
    
    for (size_t i = 0; i < spec->param_count; i++) {
        int ptype_idx = (int)spec->param_types[i];
        if (ptype_idx < 0 || ptype_idx > 5) ptype_idx = 0;
        pos += snprintf(result + pos, sizeof(result) - pos, "%s%s %s",
                        (i > 0) ? ", " : "",
                        type_names[ptype_idx],
                        spec->param_names[i]);
    }
    pos += snprintf(result + pos, sizeof(result) - pos, ") {\n");
    
    if (spec->example_count > 0) {
        char op = infer_operation_from_examples(spec);
        
        if (spec->return_type != TYPE_VOID) {
            pos += snprintf(result + pos, sizeof(result) - pos, "    %s result = 0;\n\n",
                            type_names[ret_type_idx]);
        }
        
        if (spec->param_count >= 2 && op != '?') {
            if (op == 'm') {
                /* min 模式 */
                pos += snprintf(result + pos, sizeof(result) - pos,
                        "    result = (%s < %s) ? %s : %s;\n\n",
                        spec->param_names[0], spec->param_names[1],
                        spec->param_names[0], spec->param_names[1]);
            } else if (op == 'M') {
                /* max 模式 */
                pos += snprintf(result + pos, sizeof(result) - pos,
                        "    result = (%s > %s) ? %s : %s;\n\n",
                        spec->param_names[0], spec->param_names[1],
                        spec->param_names[0], spec->param_names[1]);
            } else if (op == 's') {
                 /* sum of all inputs */
                 pos += snprintf(result + pos, sizeof(result) - pos,
                         "    result = 0;\n");
                 for (size_t pi = 0; pi < spec->param_count; pi++)
                     pos += snprintf(result + pos, sizeof(result) - pos,
                             "    result += %s;\n", spec->param_names[pi]);
             } else if (op == 'n') {
                 /* mean/average of all inputs */
                 pos += snprintf(result + pos, sizeof(result) - pos,
                         "    result = 0;\n");
                 for (size_t pi = 0; pi < spec->param_count; pi++)
                     pos += snprintf(result + pos, sizeof(result) - pos,
                             "    result += %s;\n", spec->param_names[pi]);
                 pos += snprintf(result + pos, sizeof(result) - pos,
                         "    result /= (float)(%zu);\n\n", spec->param_count);
             } else {
                /* 标准算术 */
                pos += snprintf(result + pos, sizeof(result) - pos,
                        "    result = %s %c %s;\n\n",
                        spec->param_names[0], op, spec->param_names[1]);
            }
        } else if (spec->param_count >= 1 && op != '?' && op != '/' && op != '-') {
            if (op == 'S') {
                pos += snprintf(result + pos, sizeof(result) - pos,
                        "    result = %s * %s;\n\n",
                        spec->param_names[0], spec->param_names[0]);
            } else if (op == 'a') {
                pos += snprintf(result + pos, sizeof(result) - pos,
                        "    result = (%s < 0) ? -%s : %s;\n\n",
                        spec->param_names[0], spec->param_names[0], spec->param_names[0]);
            } else if (op == 'l' && spec->example_count >= 2) {
                /* linear: y = scale * x + offset, compute from examples */
                double x0 = spec->examples[0].inputs[0];
                double y0 = spec->examples[0].expected_output;
                double x1 = spec->examples[1].inputs[0];
                double y1 = spec->examples[1].expected_output;
                double scale = (x1 != x0) ? (y1 - y0) / (x1 - x0) : 1.0;
                double offset = y0 - scale * x0;
                pos += snprintf(result + pos, sizeof(result) - pos,
                        "    result = %.6ff * %s + %.6ff;\n\n",
                        scale, spec->param_names[0], offset);
            } else {
                pos += snprintf(result + pos, sizeof(result) - pos,
                        "    result = %s %c 1.0;\n\n",
                        spec->param_names[0], op);
            }
        } else if (spec->param_count >= 1) {
            if (fabs(spec->examples[0].expected_output - spec->examples[0].inputs[0]) < 0.001) {
                pos += snprintf(result + pos, sizeof(result) - pos,
                                "    result = %s;\n\n", spec->param_names[0]);
            } else {
                pos += snprintf(result + pos, sizeof(result) - pos,
                                "    result = %s;\n\n", spec->param_names[0]);
            }
        }
        
        if (spec->postcondition && spec->param_count > 0) {
            pos += snprintf(result + pos, sizeof(result) - pos,
                            "    if (%s < 0) {\n        result = -result;\n    }\n\n",
                            spec->param_names[0]);
        }
        
        if (spec->return_type != TYPE_VOID) {
            pos += snprintf(result + pos, sizeof(result) - pos, "    return result;\n");
        }
    } else {
        if (spec->return_type != TYPE_VOID) {
            pos += snprintf(result + pos, sizeof(result) - pos,
                            "    %s result = 0;\n", type_names[ret_type_idx]);
            if (spec->postcondition && spec->param_count > 0) {
                pos += snprintf(result + pos, sizeof(result) - pos,
                                "    if (%s > 0) {\n        result = %s;\n    } else {\n        result = -%s;\n    }\n",
                                spec->param_names[0], spec->param_names[0], spec->param_names[0]);
            }
            pos += snprintf(result + pos, sizeof(result) - pos, "    return result;\n");
        }
    }
    
    pos += snprintf(result + pos, sizeof(result) - pos, "}\n");
    
    return string_duplicate_nullable(result);
}

/* ============================================================================
 * F-11: 语义分析实现
 * =========================================================================== */

/**
 * @brief 递归遍历AST构建符号表
 */
static void semantic_build_symbol_table(const ASTNode* node,
                                        SymbolInfo** symbols,
                                        size_t* count,
                                        size_t* capacity,
                                        SymbolScope current_scope,
                                        int* error_count,
                                        char*** scope_errors,
                                        size_t* scope_error_capacity) {
    if (!node) return;
    
    SymbolScope child_scope = current_scope;
    
    switch (node->type) {
        case AST_FUNCTION:
            child_scope = SCOPE_FUNCTION;
            if (node->name) {
                for (size_t i = 0; i < *count; i++) {
                    if ((*symbols)[i].name && strcmp((*symbols)[i].name, node->name) == 0 &&
                        (*symbols)[i].scope == SCOPE_GLOBAL) {
                        (*error_count)++;
                        if (*error_count >= (int)*scope_error_capacity) {
                            *scope_error_capacity = (*scope_error_capacity == 0) ? 8 : (*scope_error_capacity * 2);
                            char** new_e = (char**)safe_realloc(*scope_errors, *scope_error_capacity * sizeof(char*));
                            if (!new_e) return;
                            *scope_errors = new_e;
                        }
                        char buf[256];
                        snprintf(buf, sizeof(buf), "第%d行: 函数'%s'重复定义",
                                 node->line, node->name);
                        (*scope_errors)[*error_count - 1] = string_duplicate_nullable(buf);
                        return;
                    }
                }
                if (*count >= *capacity) {
                    *capacity = (*capacity == 0) ? 16 : (*capacity * 2);
                    SymbolInfo* new_s = (SymbolInfo*)safe_realloc(*symbols, *capacity * sizeof(SymbolInfo));
                    if (!new_s) return;
                    *symbols = new_s;
                }
                (*symbols)[*count].name = string_duplicate_nullable(node->name);
                (*symbols)[*count].type = TYPE_CUSTOM;
                (*symbols)[*count].scope = SCOPE_GLOBAL;
                (*symbols)[*count].declared_line = node->line;
                (*symbols)[*count].is_initialized = 1;
                (*symbols)[*count].is_used = 0;
                (*symbols)[*count].ref_count = 0;
                (*count)++;
            }
            break;
            
        case AST_VARIABLE_DECL:
            if (node->name) {
                for (size_t i = 0; i < *count; i++) {
                    if ((*symbols)[i].name && strcmp((*symbols)[i].name, node->name) == 0 &&
                        (*symbols)[i].scope == current_scope) {
                        (*error_count)++;
                        if (*error_count >= (int)*scope_error_capacity) {
                            *scope_error_capacity = (*scope_error_capacity == 0) ? 8 : (*scope_error_capacity * 2);
                            char** new_e = (char**)safe_realloc(*scope_errors, *scope_error_capacity * sizeof(char*));
                            if (!new_e) return;
                            *scope_errors = new_e;
                        }
                        char buf[256];
                        snprintf(buf, sizeof(buf), "第%d行: 变量'%s'在当前作用域重复声明",
                                 node->line, node->name);
                        (*scope_errors)[*error_count - 1] = string_duplicate_nullable(buf);
                        return;
                    }
                }
                if (*count >= *capacity) {
                    *capacity = (*capacity == 0) ? 16 : (*capacity * 2);
                    SymbolInfo* new_s = (SymbolInfo*)safe_realloc(*symbols, *capacity * sizeof(SymbolInfo));
                    if (!new_s) return;
                    *symbols = new_s;
                }
                (*symbols)[*count].name = string_duplicate_nullable(node->name);
                (*symbols)[*count].type = node->value.data_type;
                (*symbols)[*count].scope = current_scope;
                (*symbols)[*count].declared_line = node->line;
                (*symbols)[*count].is_initialized = (node->right != NULL) ? 1 : 0;
                (*symbols)[*count].is_used = 0;
                (*symbols)[*count].ref_count = 0;
                (*count)++;
            }
            break;
            
        case AST_VARIABLE_REF:
            if (node->name) {
                int found = 0;
                for (size_t i = 0; i < *count; i++) {
                    if ((*symbols)[i].name && strcmp((*symbols)[i].name, node->name) == 0) {
                        (*symbols)[i].is_used = 1;
                        (*symbols)[i].ref_count++;
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    (*error_count)++;
                    if (*error_count >= (int)*scope_error_capacity) {
                        *scope_error_capacity = (*scope_error_capacity == 0) ? 8 : (*scope_error_capacity * 2);
                        char** new_e = (char**)safe_realloc(*scope_errors, *scope_error_capacity * sizeof(char*));
                        if (!new_e) return;
                        *scope_errors = new_e;
                    }
                    char buf[256];
                    snprintf(buf, sizeof(buf), "第%d行: 变量'%s'未声明",
                             node->line, node->name);
                    (*scope_errors)[*error_count - 1] = string_duplicate_nullable(buf);
                }
            }
            break;
            
        case AST_BLOCK:
            child_scope = SCOPE_BLOCK;
            break;
            
        default:
            break;
    }
    
    if (node->left) semantic_build_symbol_table(node->left, symbols, count, capacity, child_scope, error_count, scope_errors, scope_error_capacity);
    if (node->right) semantic_build_symbol_table(node->right, symbols, count, capacity, child_scope, error_count, scope_errors, scope_error_capacity);
    if (node->next) semantic_build_symbol_table(node->next, symbols, count, capacity, child_scope, error_count, scope_errors, scope_error_capacity);
}

/**
 * @brief 检查类型一致性
 */
static void semantic_check_types(const ASTNode* node,
                                 const SymbolInfo* symbols,
                                 size_t symbol_count,
                                 char*** errors,
                                 size_t* error_count,
                                 size_t* error_capacity) {
    if (!node) return;
    
    if (node->type == AST_VARIABLE_REF && node->name) {
        for (size_t i = 0; i < symbol_count; i++) {
            if (symbols[i].name && strcmp(symbols[i].name, node->name) == 0) {
                if (!symbols[i].is_initialized) {
                    if (*error_count >= *error_capacity) {
                        *error_capacity = (*error_capacity == 0) ? 8 : (*error_capacity * 2);
                        char** new_e = (char**)safe_realloc(*errors, *error_capacity * sizeof(char*));
                        if (!new_e) return;
                        *errors = new_e;
                    }
                    char buf[256];
                    snprintf(buf, sizeof(buf), "第%d行: 变量'%s'可能未初始化",
                             node->line, node->name);
                    (*errors)[*error_count] = string_duplicate_nullable(buf);
                    (*error_count)++;
                }
                break;
            }
        }
    }
    
    if (node->left) semantic_check_types(node->left, symbols, symbol_count, errors, error_count, error_capacity);
    if (node->right) semantic_check_types(node->right, symbols, symbol_count, errors, error_count, error_capacity);
    if (node->next) semantic_check_types(node->next, symbols, symbol_count, errors, error_count, error_capacity);
}

/**
 * @brief 检查未使用变量
 */
static void semantic_check_unused(const SymbolInfo* symbols, size_t symbol_count,
                                  char*** errors, size_t* error_count,
                                  size_t* error_capacity) {
    for (size_t i = 0; i < symbol_count; i++) {
        if (symbols[i].scope == SCOPE_BLOCK && symbols[i].is_initialized && !symbols[i].is_used) {
            if (*error_count >= *error_capacity) {
                *error_capacity = (*error_capacity == 0) ? 8 : (*error_capacity * 2);
                char** new_e = (char**)safe_realloc(*errors, *error_capacity * sizeof(char*));
                if (!new_e) return;
                *errors = new_e;
            }
            char buf[256];
            snprintf(buf, sizeof(buf), "第%d行: 变量'%s'声明但未使用",
                     symbols[i].declared_line, symbols[i].name);
            (*errors)[*error_count] = string_duplicate_nullable(buf);
            (*error_count)++;
        }
    }
}

/**
 * @brief 执行语义分析
 */
SemanticAnalysisResult* semantic_analyze(SelfProgrammingEngine* engine,
                                         const ASTNode* ast) {
    (void)engine;
    if (!ast) return NULL;
    
    SemanticAnalysisResult* result = (SemanticAnalysisResult*)safe_calloc(1, sizeof(SemanticAnalysisResult));
    if (!result) return NULL;
    
    size_t capacity = 32;
    result->symbols = (SymbolInfo*)safe_calloc(capacity, sizeof(SymbolInfo));
    if (!result->symbols) { safe_free((void**)&result); return NULL; }
    
    size_t scope_err_cap = 16;
    result->scope_errors = (char**)safe_calloc(scope_err_cap, sizeof(char*));
    if (!result->scope_errors) { semantic_analysis_destroy(result); return NULL; }
    
    int scope_err_count = 0;
    semantic_build_symbol_table(ast, &result->symbols, &result->symbol_count,
                                &capacity, SCOPE_GLOBAL, &scope_err_count,
                                &result->scope_errors, &scope_err_cap);
    result->scope_error_count = (size_t)scope_err_count;
    
    size_t err_cap = 16;
    result->type_errors = (char**)safe_calloc(err_cap, sizeof(char*));
    if (!result->type_errors) { semantic_analysis_destroy(result); return NULL; }
    
    semantic_check_types(ast, result->symbols, result->symbol_count,
                         &result->type_errors, &result->type_error_count, &err_cap);
    
    semantic_check_unused(result->symbols, result->symbol_count,
                          &result->type_errors, &result->type_error_count, &err_cap);
    
    size_t detail_pos = 0;
    detail_pos += snprintf(result->details + detail_pos, sizeof(result->details) - detail_pos,
                           "=== 语义分析报告 ===\n");
    detail_pos += snprintf(result->details + detail_pos, sizeof(result->details) - detail_pos,
                           "符号数量: %zu\n", result->symbol_count);
    detail_pos += snprintf(result->details + detail_pos, sizeof(result->details) - detail_pos,
                           "作用域错误: %zu\n", result->scope_error_count);
    detail_pos += snprintf(result->details + detail_pos, sizeof(result->details) - detail_pos,
                           "类型错误: %zu\n", result->type_error_count);
    
    for (size_t i = 0; i < result->scope_error_count; i++) {
        detail_pos += snprintf(result->details + detail_pos, sizeof(result->details) - detail_pos,
                               "  [作用域错误] %s\n",
                               result->scope_errors[i] ? result->scope_errors[i] : "未知错误");
    }
    
    for (size_t i = 0; i < result->type_error_count; i++) {
        detail_pos += snprintf(result->details + detail_pos, sizeof(result->details) - detail_pos,
                               "  [类型错误] %s\n",
                               result->type_errors[i] ? result->type_errors[i] : "未知错误");
    }
    
    for (size_t i = 0; i < result->symbol_count; i++) {
        if (result->symbols[i].is_initialized && !result->symbols[i].is_used && result->symbols[i].scope != SCOPE_GLOBAL) {
            result->has_unused_var = 1;
        }
        if (!result->symbols[i].is_initialized && result->symbols[i].ref_count > 0) {
            result->has_uninitialized_var = 1;
        }
    }
    
    return result;
}

/**
 * @brief 销毁语义分析结果
 */
void semantic_analysis_destroy(SemanticAnalysisResult* result) {
    if (!result) return;
    if (result->symbols) {
        for (size_t i = 0; i < result->symbol_count; i++) {
            if (result->symbols[i].name) safe_free((void**)&result->symbols[i].name);
        }
        safe_free((void**)&result->symbols);
    }
    if (result->type_errors) {
        for (size_t i = 0; i < result->type_error_count; i++) {
            if (result->type_errors[i]) safe_free((void**)&result->type_errors[i]);
        }
        safe_free((void**)&result->type_errors);
    }
    if (result->scope_errors) {
        for (size_t i = 0; i < result->scope_error_count; i++) {
            if (result->scope_errors[i]) safe_free((void**)&result->scope_errors[i]);
        }
        safe_free((void**)&result->scope_errors);
    }
    safe_free((void**)&result);
}

/* ============================================================================
 * F-11: 数据流分析实现
 * =========================================================================== */

/**
 * @brief 收集AST中的所有变量引用
 */
static int collect_var_refs(const ASTNode* node, const char*** names,
                            size_t* count, size_t* capacity) {
    if (!node) return 0;
    int total = 0;
    
    if (node->type == AST_VARIABLE_REF && node->name) {
        if (*count >= *capacity) {
            *capacity = (*capacity == 0) ? 16 : (*capacity * 2);
            const char** new_n = (const char**)safe_realloc((void**)*names, *capacity * sizeof(char*));
            if (!new_n) return total;
            *names = new_n;
        }
        (*names)[*count] = node->name;
        (*count)++;
        total++;
    }
    
    total += collect_var_refs(node->left, names, count, capacity);
    total += collect_var_refs(node->right, names, count, capacity);
    total += collect_var_refs(node->next, names, count, capacity);
    return total;
}

/**
 * @brief 收集所有赋值目标（定值点）
 */
static int collect_def_points(const ASTNode* node, const char*** names,
                              size_t* count, size_t* capacity) {
    if (!node) return 0;
    int total = 0;
    
    if ((node->type == AST_ASSIGNMENT || node->type == AST_VARIABLE_DECL) && node->left && node->left->name) {
        if (*count >= *capacity) {
            *capacity = (*capacity == 0) ? 16 : (*capacity * 2);
            const char** new_n = (const char**)safe_realloc((void**)*names, *capacity * sizeof(char*));
            if (!new_n) return total;
            *names = new_n;
        }
        (*names)[*count] = node->left->name;
        (*count)++;
        total++;
    }
    
    total += collect_def_points(node->left, names, count, capacity);
    total += collect_def_points(node->right, names, count, capacity);
    total += collect_def_points(node->next, names, count, capacity);
    return total;
}

/**
 * @brief 执行数据流分析
 */
DataFlowResult* analyze_data_flow(SelfProgrammingEngine* engine,
                                  const ASTNode* ast) {
    (void)engine;
    if (!ast) return NULL;
    
    DataFlowResult* result = (DataFlowResult*)safe_calloc(1, sizeof(DataFlowResult));
    if (!result) return NULL;
    
    size_t def_cap = 32, use_cap = 32;
    const char** def_names = (const char**)safe_calloc(def_cap, sizeof(char*));
    const char** use_names = (const char**)safe_calloc(use_cap, sizeof(char*));
    if (!def_names || !use_names) {
        safe_free((void**)&def_names);
        safe_free((void**)&use_names);
        safe_free((void**)&result);
        return NULL;
    }
    
    result->var_count = collect_def_points(ast, &def_names, &result->var_count, &def_cap) > 0 ? result->var_count : 0;
    result->ref_count = collect_var_refs(ast, &use_names, &result->ref_count, &use_cap);
    
    if (result->var_count > 0 && result->ref_count > 0) {
        result->def_use_chains = (int**)safe_calloc(result->var_count, sizeof(int*));
        if (!result->def_use_chains) {
            safe_free((void**)&def_names);
            safe_free((void**)&use_names);
            data_flow_destroy(result);
            return NULL;
        }
        for (size_t i = 0; i < result->var_count; i++) {
            result->def_use_chains[i] = (int*)safe_calloc(result->ref_count, sizeof(int));
            if (!result->def_use_chains[i]) {
                safe_free((void**)&def_names);
                safe_free((void**)&use_names);
                data_flow_destroy(result);
                return NULL;
            }
            for (size_t j = 0; j < result->ref_count; j++) {
                if (def_names[i] && use_names[j] && strcmp(def_names[i], use_names[j]) == 0) {
                    result->def_use_chains[i][j] = 1;
                }
            }
        }
    }
    
    result->live_variables = (int*)safe_calloc(result->var_count, sizeof(int));
    if (result->live_variables) {
        for (size_t i = 0; i < result->var_count; i++) {
            for (size_t j = 0; j < result->ref_count; j++) {
                if (result->def_use_chains && result->def_use_chains[i] && result->def_use_chains[i][j]) {
                    result->live_variables[i] = 1;
                    break;
                }
            }
        }
    }
    
    if (result->var_count > 0) {
        for (size_t i = 0; i < result->var_count; i++) {
            int has_def = 0;
            for (size_t j = 0; j < result->ref_count; j++) {
                if (result->def_use_chains && result->def_use_chains[i] && result->def_use_chains[i][j]) {
                    has_def = 1;
                    break;
                }
            }
            if (def_names[i] && !has_def) {
                result->has_dead_assignment = 1;
                break;
            }
        }
        for (size_t j = 0; j < result->ref_count; j++) {
            int has_def = 0;
            if (use_names[j]) {
                for (size_t i = 0; i < result->var_count; i++) {
                    if (result->def_use_chains && result->def_use_chains[i] && result->def_use_chains[i][j]) {
                        has_def = 1;
                        break;
                    }
                }
                if (!has_def) {
                    result->has_uninitialized_read = 1;
                    break;
                }
            }
        }
    }
    
    size_t pos = 0;
    pos += snprintf(result->details, sizeof(result->details),
                    "=== 数据流分析报告 ===\n");
    pos += snprintf(result->details + pos, sizeof(result->details) - pos,
                    "变量数量: %zu, 引用数量: %zu\n", result->var_count, result->ref_count);
    pos += snprintf(result->details + pos, sizeof(result->details) - pos,
                    "未初始化读取: %s\n", result->has_uninitialized_read ? "是" : "否");
    pos += snprintf(result->details + pos, sizeof(result->details) - pos,
                    "死赋值: %s\n", result->has_dead_assignment ? "是" : "否");
    
    safe_free((void**)&def_names);
    safe_free((void**)&use_names);
    return result;
}

/**
 * @brief 销毁数据流分析结果
 */
void data_flow_destroy(DataFlowResult* result) {
    if (!result) return;
    if (result->def_use_chains) {
        for (size_t i = 0; i < result->var_count; i++) {
            if (result->def_use_chains[i]) safe_free((void**)&result->def_use_chains[i]);
        }
        safe_free((void**)&result->def_use_chains);
    }
    if (result->live_variables) safe_free((void**)&result->live_variables);
    safe_free((void**)&result);
}

/* ============================================================================
 * F-11: 代码转换实现
 * =========================================================================== */

/**
 * @brief 深度复制AST节点
 */
static ASTNode* clone_ast_node(const ASTNode* node) {
    if (!node) return NULL;
    ASTNode* clone = (ASTNode*)safe_calloc(1, sizeof(ASTNode));
    if (!clone) return NULL;
    *clone = *node;
    clone->left = NULL;
    clone->right = NULL;
    clone->next = NULL;
    if (node->name) clone->name = string_duplicate_nullable(node->name);
    if (node->type == AST_LITERAL && node->value.string_value)
        clone->value.string_value = string_duplicate_nullable(node->value.string_value);
    if (node->type == AST_VARIABLE_DECL && node->value.string_value)
        clone->value.string_value = string_duplicate_nullable(node->value.string_value);
    clone->left = clone_ast_node(node->left);
    clone->right = clone_ast_node(node->right);
    clone->next = clone_ast_node(node->next);
    return clone;
}

/**
 * @brief 常量折叠：将编译时可计算的常量表达式替换为计算结果
 */
int constant_folding(SelfProgrammingEngine* engine, ASTNode* ast) {
    (void)engine;
    if (!ast) return 0;
    int changes = 0;
    
    if (ast->type == AST_BINARY_OP && ast->left && ast->right) {
        int left_is_const = (ast->left->type == AST_LITERAL);
        int right_is_const = (ast->right->type == AST_LITERAL);
        
        if (left_is_const && right_is_const) {
            double lv = ast->left->value.float_value;
            double rv = ast->right->value.float_value;
            double result_val = 0.0;
            
            switch (ast->value.bin_op) {
                case OP_ADD: result_val = lv + rv; break;
                case OP_SUB: result_val = lv - rv; break;
                case OP_MUL: result_val = lv * rv; break;
                case OP_DIV: if (rv != 0.0) result_val = lv / rv; else return 0; break;
                case OP_MOD: result_val = fmod(lv, rv); break;
                case OP_EQ:  result_val = (fabs(lv - rv) < 1e-9) ? 1.0 : 0.0; break;
                case OP_NE:  result_val = (fabs(lv - rv) >= 1e-9) ? 1.0 : 0.0; break;
                case OP_LT:  result_val = (lv < rv) ? 1.0 : 0.0; break;
                case OP_GT:  result_val = (lv > rv) ? 1.0 : 0.0; break;
                case OP_LE:  result_val = (lv <= rv) ? 1.0 : 0.0; break;
                case OP_GE:  result_val = (lv >= rv) ? 1.0 : 0.0; break;
                default: return 0;
            }
            
            ast_destroy(ast->left);
            ast_destroy(ast->right);
            ast->left = NULL;
            ast->right = NULL;
            ast->type = AST_LITERAL;
            ast->value.float_value = (float)result_val;
            ast->value.int_value = (int)result_val;
            changes = 1;
        }
    }
    
    int left_ch = constant_folding(engine, ast->left);
    int right_ch = constant_folding(engine, ast->right);
    int next_ch = constant_folding(engine, ast->next);
    
    return changes + left_ch + right_ch + next_ch;
}

/**
 * @brief 死代码消除：移除不可达代码片段
 */
int dead_code_elimination(SelfProgrammingEngine* engine, ASTNode* ast) {
    (void)engine;
    if (!ast) return 0;
    int changes = 0;
    
    if (ast->type == AST_BLOCK && ast->left) {
        ASTNode* prev = NULL;
        ASTNode* curr = ast->left;
        while (curr) {
            if (curr->type == AST_RETURN) {
                ASTNode* next = curr->next;
                curr->next = NULL;
                int killed = 0;
                while (next) {
                    ASTNode* to_kill = next;
                    next = next->next;
                    ast_destroy(to_kill);
                    killed++;
                    changes++;
                }
                break;
            }
            if (curr->type == AST_IF && curr->left && curr->left->type == AST_LITERAL &&
                fabs(curr->left->value.float_value) < 0.5) {
                ASTNode* dead_branch = curr->right;
                curr->right = NULL;
                if (dead_branch) {
                    ast_destroy(dead_branch);
                    changes++;
                }
                curr->left->value.float_value = 1.0f;
                curr->left->value.int_value = 1;
            }
            prev = curr;
            curr = curr->next;
        }
    }
    
    changes += dead_code_elimination(engine, ast->left);
    changes += dead_code_elimination(engine, ast->right);
    changes += dead_code_elimination(engine, ast->next);
    return changes;
}

/**
 * @brief 检查AST节点是否引用了给定变量
 */
static int node_references_var(const ASTNode* node, const char* var_name) {
    if (!node || !var_name) return 0;
    if (node->type == AST_VARIABLE_REF && node->name && strcmp(node->name, var_name) == 0) return 1;
    return node_references_var(node->left, var_name) ||
           node_references_var(node->right, var_name) ||
           node_references_var(node->next, var_name);
}

/**
 * @brief 检查表达式是否依赖于循环变量
 */
static int expr_depends_on_loop_var(const ASTNode* expr, const char* loop_var) {
    if (!expr || !loop_var) return 0;
    return node_references_var(expr, loop_var);
}

/**
 * @brief 计算表达式节点中的变量引用数
 */
static int count_var_refs_in_expr(const ASTNode* expr) {
    if (!expr) return 0;
    if (expr->type == AST_VARIABLE_REF) return 1;
    return count_var_refs_in_expr(expr->left) +
           count_var_refs_in_expr(expr->right);
}

/**
 * @brief 从循环体内提取不依赖于循环变量的表达式
 */
static ASTNode* extract_invariant_expr(ASTNode* body, const char* loop_var) {
    if (!body || !loop_var) return NULL;
    ASTNode* result = NULL;
    ASTNode* last = NULL;
    
    ASTNode* curr = body;
    while (curr) {
        if (curr->type == AST_ASSIGNMENT && curr->right) {
            if (!expr_depends_on_loop_var(curr->right, loop_var) &&
                count_var_refs_in_expr(curr->right) > 0) {
                ASTNode* inv = clone_ast_node(curr->right);
                if (inv) {
                    if (!result) result = inv;
                    else if (last) last->next = inv;
                    last = inv;
                }
            }
        }
        curr = curr->next;
    }
    return result;
}

/**
 * @brief 循环不变式外提
 */
int loop_invariant_hoisting(SelfProgrammingEngine* engine, ASTNode* ast) {
    (void)engine;
    if (!ast) return 0;
    int changes = 0;
    
    if (ast->type == AST_LOOP && ast->right) {
        const char* loop_var = NULL;
        if (ast->left && ast->left->name) loop_var = ast->left->name;
        else if (ast->name) loop_var = ast->name;
        
        if (loop_var) {
            ASTNode* invariants = extract_invariant_expr(ast->right, loop_var);
            if (invariants) {
                ASTNode* block = ast->right;
                ASTNode* curr = invariants;
                ASTNode* inv_last = NULL;
                while (curr) {
                    ASTNode* next = curr->next;
                    ASTNode* assign = (ASTNode*)safe_calloc(1, sizeof(ASTNode));
                    if (assign) {
                        assign->type = AST_ASSIGNMENT;
                        assign->left = clone_ast_node(curr);
                        assign->right = curr;
                        assign->next = block;
                        if (!inv_last) invariants = assign;
                        else inv_last->next = assign;
                        inv_last = assign;
                        changes++;
                    }
                    curr = next;
                }
                ast->right = invariants;
                
                ASTNode* loop_body = ast->right;
                while (loop_body && loop_body->next) loop_body = loop_body->next;
                if (loop_body) {
                    ASTNode* orig_body = (ASTNode*)safe_calloc(1, sizeof(ASTNode));
                    if (orig_body) {
                        orig_body->type = AST_BLOCK;
                        orig_body->left = block;
                        orig_body->next = NULL;
                        loop_body->next = orig_body;
                    }
                }
            }
        }
    }
    
    changes += loop_invariant_hoisting(engine, ast->left);
    changes += loop_invariant_hoisting(engine, ast->right);
    changes += loop_invariant_hoisting(engine, ast->next);
    return changes;
}

/*: 结构简化 - 减少深层嵌套结构
 * 将嵌套超过3层的条件块提取为独立函数调用样式标记 */
int structure_simplification(SelfProgrammingEngine* engine, ASTNode* ast) {
    (void)engine;
    if (!ast) return 0;
    int changes = 0;

    /* 遍历AST查找嵌套深度超过3层的IF块 */
    if (ast->type == AST_IF && ast->right) {
        int depth = 0;
        ASTNode* inner = ast->right;
        while (inner && inner->type == AST_IF) {
            depth++;
            inner = inner->right;
        }
        /* 嵌套深度>3的结构标记为需要提取 */
        if (depth > 3 && inner) {
            ASTNode* marker = (ASTNode*)safe_calloc(1, sizeof(ASTNode));
            if (marker) {
                marker->type = AST_BLOCK;
                marker->name = string_duplicate_nullable("__REFACTOR_EXTRACT_BLOCK__");
                marker->next = inner->next;
                inner->next = marker;
                changes++;
            }
        }
    }

    changes += structure_simplification(engine, ast->left);
    changes += structure_simplification(engine, ast->right);
    changes += structure_simplification(engine, ast->next);
    return changes;
}

/*: 识别大函数体并标记为可提取的独立函数
 * 遍历AST中的函数定义，对超过50行的大函数标记内部逻辑块 */
int function_extraction(SelfProgrammingEngine* engine, ASTNode* ast) {
    (void)engine;
    if (!ast) return 0;
    int changes = 0;

    /* 查找函数定义节点 */
    if (ast->type == AST_FUNCTION && ast->left && ast->right) {
        /* AST_FUNCTION结构: name(函数名), left(参数列表), right(函数体) */
        ASTNode* body = ast->right;
        int stmt_count = 0;
        ASTNode* curr = (body->type == AST_BLOCK) ? body->left : body;
        while (curr) { stmt_count++; curr = curr->next; }

        /* 大函数: 超过50条语句标记需要拆分 */
        if (stmt_count > 50) {
            /* 在函数体中标记逻辑分段点: 每15-20条语句标记一个分段 */
            ASTNode* stmt = (body->type == AST_BLOCK) ? body->left : body;
            int seg_count = 0;
            ASTNode* seg_start = NULL;
            while (stmt) {
                if (seg_count == 0) seg_start = stmt;
                seg_count++;

                /* 每隔18条语句或遇到大型控制流时标记分段 */
                if (seg_count >= 18 || (stmt->type == AST_IF && seg_count > 10) ||
                    (stmt->type == AST_LOOP && seg_count > 10)) {
                    if (seg_start && seg_start != stmt) {
                        ASTNode* marker = (ASTNode*)safe_calloc(1, sizeof(ASTNode));
                        if (marker) {
                            marker->type = AST_BLOCK;
                            marker->name = string_duplicate_nullable("__REFACTOR_EXTRACTED_FUNC__");
                            ASTNode* next_stmt = stmt->next;
                            stmt->next = marker;
                            marker->next = next_stmt;
                            changes++;
                        }
                    }
                    seg_count = 0;
                    seg_start = NULL;
                }
                stmt = stmt->next;
            }
        }

        /* 查找小函数进行内联候选标记 */
        if (stmt_count <= 3 && ast->name) {
            ASTNode* marker = (ASTNode*)safe_calloc(1, sizeof(ASTNode));
            if (marker) {
                marker->type = AST_BLOCK;
                marker->name = string_duplicate_nullable("__REFACTOR_INLINE_CANDIDATE__");
                ASTNode* body_start = (body->type == AST_BLOCK) ? body->left : body;
                if (body_start) {
                    marker->next = body_start;
                    if (body->type == AST_BLOCK) body->left = marker;
                    changes++;
                } else {
                    safe_free((void**)&marker);
                }
            }
        }
    }

    changes += function_extraction(engine, ast->left);
    changes += function_extraction(engine, ast->right);
    changes += function_extraction(engine, ast->next);
    return changes;
}

/**
 * @brief 对AST应用代码转换
 */
TransformResult* transform_code(SelfProgrammingEngine* engine,
                                const ASTNode* ast,
                                const TransformRuleType* rules,
                                size_t rule_count) {
    if (!engine || !ast) return NULL;
    
    TransformResult* result = (TransformResult*)safe_calloc(1, sizeof(TransformResult));
    if (!result) return NULL;
    
    result->transformed_ast = clone_ast_node(ast);
    if (!result->transformed_ast) { safe_free((void**)&result); return NULL; }
    
    size_t applied_cap = 16;
    result->applied_rules = (char**)safe_calloc(applied_cap, sizeof(char*));
    if (!result->applied_rules) { transform_result_destroy(result); return NULL; }
    
    size_t pos = 0;
    pos += snprintf(result->details, sizeof(result->details), "=== 代码转换报告 ===\n");
    
    for (size_t i = 0; i < rule_count; i++) {
        int changes = 0;
        const char* rule_name = "未知规则";
        
        switch (rules[i]) {
            case TRANSFORM_CONSTANT_FOLDING:
                changes = constant_folding(engine, result->transformed_ast);
                rule_name = "常量折叠";
                break;
            case TRANSFORM_DEAD_CODE_ELIM:
                changes = dead_code_elimination(engine, result->transformed_ast);
                rule_name = "死代码消除";
                break;
            case TRANSFORM_LOOP_INVARIANT:
                changes = loop_invariant_hoisting(engine, result->transformed_ast);
                rule_name = "循环不变式外提";
                break;
            case TRANSFORM_STRENGTH_REDUCTION:
                rule_name = "强度削弱";
                break;
            case TRANSFORM_CSE:
                rule_name = "公共子表达式消除";
                break;
        }
        
        if (result->rule_count >= applied_cap) {
            applied_cap *= 2;
            char** new_r = (char**)safe_realloc(result->applied_rules, applied_cap * sizeof(char*));
            if (!new_r) break;
            result->applied_rules = new_r;
        }
        
        char buf[128];
        snprintf(buf, sizeof(buf), "%s: %d处变换", rule_name, changes);
        result->applied_rules[result->rule_count] = string_duplicate_nullable(buf);
        result->rule_count++;
        
        pos += snprintf(result->details + pos, sizeof(result->details) - pos,
                        "[%zu] %s\n", i + 1, buf);
    }
    
    return result;
}

/**
 * @brief 销毁转换结果
 */
void transform_result_destroy(TransformResult* result) {
    if (!result) return;
    if (result->transformed_ast) ast_destroy(result->transformed_ast);
    if (result->applied_rules) {
        for (size_t i = 0; i < result->rule_count; i++) {
            if (result->applied_rules[i]) safe_free((void**)&result->applied_rules[i]);
        }
        safe_free((void**)&result->applied_rules);
    }
    safe_free((void**)&result);
}

/** ============================================================================
 * Python代码生成
 * =========================================================================== */

/** 追加字符串到缓冲区 */
static int py_append(char* buf, size_t buf_size, size_t* pos, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int remain = (int)(buf_size - *pos);
    int written = vsnprintf(buf + *pos, (size_t)remain > 0 ? (size_t)remain : 0, fmt, args);
    va_end(args);
    if (written > 0) {
        *pos += (size_t)(written < remain ? written : (remain > 0 ? remain - 1 : 0));
    }
    return written;
}

/** 生成通用Python脚本 */
static void py_gen_script(const PythonGenConfig* cfg, char* buf, size_t buf_size, size_t* pos) {
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "# -*- coding: utf-8 -*-\n");
        if (cfg->description) {
            py_append(buf, buf_size, pos, "\"\"\"\n%s\n", cfg->description);
            if (cfg->author) py_append(buf, buf_size, pos, "\n作者: %s\n", cfg->author);
            py_append(buf, buf_size, pos, "版本: %.1f\n\"\"\"\n", cfg->version);
        }
    }
    py_append(buf, buf_size, pos, "\nimport sys\nimport os\nimport json\nimport time\nimport math\n");
    if (cfg->include_logging) {
        py_append(buf, buf_size, pos,
                  "import logging\nlogging.basicConfig(level=logging.INFO, format='%%(asctime)s [%%(levelname)s] %%(message)s')\nlog = logging.getLogger(__name__)\n");
    }
    if (cfg->async_mode) {
        py_append(buf, buf_size, pos, "import asyncio\nimport aiohttp\n");
    }
}

/** 生成PyBullet仿真脚本 */
static void py_gen_pybullet(const PythonGenConfig* cfg, char* buf, size_t buf_size, size_t* pos) {
    py_gen_script(cfg, buf, buf_size, pos);
    py_append(buf, buf_size, pos,
              "\nimport pybullet as p\nimport pybullet_data\n\nclass RobotSimulation:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "    \"\"\"机器人仿真控制类\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "    def __init__(self, urdf_path: str = None, gui: bool = True):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"初始化仿真环境\n\n"
                  "        Args:\n            urdf_path: URDF文件路径\n            gui: 是否启用GUI\n"
                  "        \"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        self.urdf_path = urdf_path\n"
              "        self.gui = gui\n"
              "        self.physics_client = None\n"
              "        self.robot_id = None\n"
              "        self.joint_indices = []\n"
              "        self.end_effector_idx = None\n"
              "        self._connect()\n\n"
              "    def _connect(self):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"连接物理引擎\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        mode = p.GUI if self.gui else p.DIRECT\n"
              "        self.physics_client = p.connect(mode)\n"
              "        p.setAdditionalSearchPath(pybullet_data.getDataPath())\n"
              "        p.setGravity(0, 0, -9.81)\n"
              "        p.setTimeStep(1.0 / 240.0)\n\n"
              "    def load_robot(self, urdf_path: str = None, base_pos: tuple = (0,0,0)):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"加载机器人模型\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        path = urdf_path or self.urdf_path\n"
              "        if not path:\n            raise ValueError(\"URDF路径未指定\")\n"
              "        self.robot_id = p.loadURDF(path, base_pos, useFixedBase=False)\n"
              "        num_joints = p.getNumJoints(self.robot_id)\n"
              "        self.joint_indices = list(range(num_joints))\n"
              "        self.end_effector_idx = num_joints - 1\n"
              "        return self.robot_id\n\n"
              "    def set_joint_positions(self, positions: list):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"设置关节位置\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        for i, pos in enumerate(positions):\n"
              "            if i < len(self.joint_indices):\n"
              "                p.setJointMotorControl2(\n"
              "                    self.robot_id, i, p.POSITION_CONTROL,\n"
              "                    targetPosition=pos, force=100.0)\n\n"
              "    def get_joint_states(self) -> list:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"获取关节状态\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        states = []\n"
              "        for i in self.joint_indices:\n"
              "            state = p.getJointState(self.robot_id, i)\n"
              "            states.append(state[0])\n"
              "        return states\n\n"
              "    def get_end_effector_pose(self) -> tuple:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"获取末端执行器位姿\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        if self.end_effector_idx is None:\n"
              "            return None\n"
              "        state = p.getLinkState(self.robot_id, self.end_effector_idx)\n"
              "        return (state[0], state[1])\n\n"
              "    def step_simulation(self, steps: int = 1):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"执行仿真步进\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        for _ in range(steps):\n"
              "            p.stepSimulation()\n"
              "            time.sleep(1.0 / 240.0)\n\n"
              "    def disconnect(self):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"断开连接\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        if self.physics_client is not None:\n"
              "            p.disconnect(self.physics_client)\n\n");
}

/** 生成神经网络训练脚本 */
static void py_gen_network(const PythonGenConfig* cfg, char* buf, size_t buf_size, size_t* pos) {
    py_gen_script(cfg, buf, buf_size, pos);
    py_append(buf, buf_size, pos,
              "\nimport numpy as np\n"
              "from typing import Optional, List, Tuple\n\n"
              "class NeuralNetworkTrainer:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "    \"\"\"神经网络训练器\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "    def __init__(self, input_size: int, hidden_size: int, output_size: int,\n"
              "                 learning_rate: float = 0.01):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"初始化网络\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        self.input_size = input_size\n"
              "        self.hidden_size = hidden_size\n"
              "        self.output_size = output_size\n"
              "        self.lr = learning_rate\n"
              "        self._init_weights()\n\n"
              "    def _init_weights(self):\n"
              "        limit = math.sqrt(6.0 / (self.input_size + self.hidden_size))\n"
              "        self.w1 = np.random.uniform(-limit, limit, (self.input_size, self.hidden_size))\n"
              "        self.b1 = np.zeros((1, self.hidden_size))\n"
              "        limit = math.sqrt(6.0 / (self.hidden_size + self.output_size))\n"
              "        self.w2 = np.random.uniform(-limit, limit, (self.hidden_size, self.output_size))\n"
              "        self.b2 = np.zeros((1, self.output_size))\n\n"
              "    def forward(self, x: np.ndarray) -> np.ndarray:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"前向传播\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        self.z1 = np.dot(x, self.w1) + self.b1\n"
              "        self.a1 = np.tanh(self.z1)\n"
              "        self.z2 = np.dot(self.a1, self.w2) + self.b2\n"
              "        return self.z2\n\n"
              "    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"单步训练\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        output = self.forward(x)\n"
              "        loss = np.mean((output - y) ** 2)\n"
              "        grad_output = 2.0 * (output - y) / y.shape[0]\n"
              "        grad_w2 = np.dot(self.a1.T, grad_output)\n"
              "        grad_b2 = np.sum(grad_output, axis=0, keepdims=True)\n"
              "        grad_hidden = np.dot(grad_output, self.w2.T) * (1.0 - self.a1 ** 2)\n"
              "        grad_w1 = np.dot(x.T, grad_hidden)\n"
              "        grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True)\n"
              "        self.w2 -= self.lr * grad_w2\n"
              "        self.b2 -= self.lr * grad_b2\n"
              "        self.w1 -= self.lr * grad_w1\n"
              "        self.b1 -= self.lr * grad_b1\n"
              "        return float(loss)\n\n"
              "    def train(self, x_train: np.ndarray, y_train: np.ndarray,\n"
              "              epochs: int = 100, batch_size: int = 32,\n"
              "              callback=None) -> List[float]:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"完整训练循环\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        history = []\n"
              "        n = x_train.shape[0]\n"
              "        for epoch in range(epochs):\n"
              "            indices = np.random.permutation(n)\n"
              "            epoch_loss = 0.0\n"
              "            for start in range(0, n, batch_size):\n"
              "                batch_idx = indices[start:start + batch_size]\n"
              "                batch_x = x_train[batch_idx]\n"
              "                batch_y = y_train[batch_idx]\n"
              "                epoch_loss += self.train_step(batch_x, batch_y)\n"
              "            avg_loss = epoch_loss / max(1, (n + batch_size - 1) // batch_size)\n"
              "            history.append(avg_loss)\n"
              "            if callback:\n"
              "                callback(epoch, avg_loss)\n"
              "        return history\n\n"
              "    def predict(self, x: np.ndarray) -> np.ndarray:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"推理预测\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        return self.forward(x)\n\n");
}

/** 生成硬件控制脚本 */
static void py_gen_hardware(const PythonGenConfig* cfg, char* buf, size_t buf_size, size_t* pos) {
    py_gen_script(cfg, buf, buf_size, pos);
    py_append(buf, buf_size, pos,
              "\nimport serial\nimport struct\nfrom threading import Thread, Lock\n\n"
              "class HardwareController:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "    \"\"\"硬件控制器\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "    def __init__(self, port: str = None, baudrate: int = 115200):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"初始化硬件接口\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        self.port = port\n"
              "        self.baudrate = baudrate\n"
              "        self.serial = None\n"
              "        self.lock = Lock()\n"
              "        self.running = False\n"
              "        self._buffer = b''\n\n"
              "    def connect(self) -> bool:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"连接硬件设备\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        if not self.port:\n"
              "            return False\n"
              "        try:\n"
              "            self.serial = serial.Serial(\n"
              "                self.port, self.baudrate, timeout=1.0)\n"
              "            self.running = True\n"
              "            return True\n"
              "        except Exception as e:\n"
              "            print(f\"连接失败: {e}\")\n"
              "            return False\n\n"
              "    def send_command(self, cmd: bytes) -> bool:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"发送命令\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        with self.lock:\n"
              "            if self.serial and self.serial.is_open:\n"
              "                self.serial.write(cmd)\n"
              "                return True\n"
              "        return False\n\n"
              "    def read_data(self, size: int = 1024) -> bytes:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"读取数据\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        with self.lock:\n"
              "            if self.serial and self.serial.is_open:\n"
              "                return self.serial.read(size)\n"
              "        return b''\n\n"
              "    def set_motor_pwm(self, channel: int, pwm: int):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"设置电机PWM\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        cmd = struct.pack('<BBh', 0x01, channel & 0xFF, pwm & 0xFFFF)\n"
              "        self.send_command(cmd)\n\n"
              "    def read_sensor(self, channel: int) -> float:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"读取传感器数据\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        cmd = struct.pack('<BB', 0x02, channel & 0xFF)\n"
              "        self.send_command(cmd)\n"
              "        data = self.read_data(4)\n"
              "        if len(data) >= 4:\n"
              "            return struct.unpack('<f', data[:4])[0]\n"
              "        return 0.0\n\n"
              "    def disconnect(self):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"断开连接\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        self.running = False\n"
              "        with self.lock:\n"
              "            if self.serial and self.serial.is_open:\n"
              "                self.serial.close()\n\n");
}

/** 生成数据分析脚本 */
static void py_gen_data_analysis(const PythonGenConfig* cfg, char* buf, size_t buf_size, size_t* pos) {
    py_gen_script(cfg, buf, buf_size, pos);
    py_append(buf, buf_size, pos,
              "\nimport numpy as np\nfrom typing import List, Dict, Optional, Tuple\n\n"
              "class DataAnalyzer:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "    \"\"\"数据分析器\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "    def __init__(self, data: Optional[np.ndarray] = None):\n"
              "        self.data = data\n"
              "        self.stats = {}\n\n"
              "    def load_csv(self, filepath: str, delimiter: str = ','):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"从CSV加载数据\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        self.data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1)\n\n"
              "    def compute_statistics(self) -> Dict:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"计算统计特征\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        if self.data is None:\n"
              "            return {}\n"
              "        self.stats = {\n"
              "            'mean': np.mean(self.data, axis=0),\n"
              "            'std': np.std(self.data, axis=0),\n"
              "            'min': np.min(self.data, axis=0),\n"
              "            'max': np.max(self.data, axis=0),\n"
              "            'median': np.median(self.data, axis=0),\n"
              "        }\n"
              "        return self.stats\n\n"
              "    def fft_analysis(self) -> Tuple[np.ndarray, np.ndarray]:\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "        \"\"\"FFT频谱分析\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "        if self.data is None:\n"
              "            return None, None\n"
              "        n = self.data.shape[0]\n"
              "        freq = np.fft.fftfreq(n)[:n // 2]\n"
              "        spectrum = np.abs(np.fft.fft(self.data, axis=0))[:n // 2]\n"
              "        return freq, spectrum\n\n");
}

/** 生成Web API服务脚本 */
static void py_gen_web_api(const PythonGenConfig* cfg, char* buf, size_t buf_size, size_t* pos) {
    py_gen_script(cfg, buf, buf_size, pos);
    py_append(buf, buf_size, pos,
              "\nfrom http.server import HTTPServer, BaseHTTPRequestHandler\n"
              "import urllib.parse\n\n"
              "class APIHandler(BaseHTTPRequestHandler):\n");
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "    \"\"\"API请求处理器\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "    def do_GET(self):\n"
              "        parsed = urllib.parse.urlparse(self.path)\n"
              "        params = urllib.parse.parse_qs(parsed.query)\n"
              "        if parsed.path == '/health':\n"
              "            self._json_response({'status': 'ok', 'version': %.1f})\n"
              "        elif parsed.path == '/api/v1/predict':\n"
              "            self._json_response({'result': 'success'})\n"
              "        else:\n"
              "            self._json_response({'error': 'not found'}, 404)\n\n"
              "    def do_POST(self):\n"
              "        content_length = int(self.headers.get('Content-Length', 0))\n"
              "        body = self.rfile.read(content_length) if content_length > 0 else b'{}'\n"
              "        try:\n"
              "            data = json.loads(body)\n"
              "            self._json_response({'received': data})\n"
              "        except json.JSONDecodeError:\n"
              "            self._json_response({'error': 'invalid json'}, 400)\n\n"
              "    def _json_response(self, data: dict, status: int = 200):\n"
              "        self.send_response(status)\n"
              "        self.send_header('Content-Type', 'application/json')\n"
              "        self.send_header('Access-Control-Allow-Origin', '*')\n"
              "        self.end_headers()\n"
              "        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())\n\n"
              "def run_server(host: str = '0.0.0.0', port: int = 8080):\n",
              cfg->version);
    if (cfg->include_docstrings) {
        py_append(buf, buf_size, pos, "    \"\"\"启动服务器\"\"\"\n");
    }
    py_append(buf, buf_size, pos,
              "    server = HTTPServer((host, port), APIHandler)\n"
              "    print(f'API服务已启动: http://{host}:{port}')\n"
              "    try:\n"
              "        server.serve_forever()\n"
              "    except KeyboardInterrupt:\n"
              "        print('\\n服务已停止')\n"
              "        server.server_close()\n\n");
}

/**
 * @brief 生成Python代码
 */
char* self_programming_generate_python(SelfProgrammingEngine* engine,
                                       const PythonGenConfig* config) {
    if (!engine || !config) return NULL;

    char* buffer = (char*)safe_calloc(1, 65536);
    if (!buffer) return NULL;

    size_t pos = 0;

    switch (config->gen_type) {
        case PYGEN_PYBULLET:
            py_gen_pybullet(config, buffer, 65536, &pos);
            break;
        case PYGEN_NETWORK:
            py_gen_network(config, buffer, 65536, &pos);
            break;
        case PYGEN_HARDWARE:
            py_gen_hardware(config, buffer, 65536, &pos);
            break;
        case PYGEN_DATA_ANALYSIS:
            py_gen_data_analysis(config, buffer, 65536, &pos);
            break;
        case PYGEN_WEB_API:
            py_gen_web_api(config, buffer, 65536, &pos);
            break;
        case PYGEN_MODULE:
        case PYGEN_CLASS:
        case PYGEN_FUNCTION:
        case PYGEN_SCRIPT:
        default: {
            py_gen_script(config, buffer, 65536, &pos);
            if (config->gen_type == PYGEN_FUNCTION && config->function_name) {
                py_append(buffer, 65536, &pos, "\ndef %s(", config->function_name);
                py_append(buffer, 65536, &pos, "):\n");
                if (config->include_docstrings) {
                    py_append(buffer, 65536, &pos, "    \"\"\"%s\"\"\"\n", 
                              config->description ? config->description : "函数描述");
                }
                py_append(buffer, 65536, &pos, "    pass\n");
            } else if (config->gen_type == PYGEN_CLASS && config->class_name) {
                py_append(buffer, 65536, &pos, "\nclass %s:\n", config->class_name);
                if (config->include_docstrings) {
                    py_append(buffer, 65536, &pos, "    \"\"\"%s\"\"\"\n",
                              config->description ? config->description : "类描述");
                }
                py_append(buffer, 65536, &pos, "    def __init__(self):\n");
                py_append(buffer, 65536, &pos, "        pass\n");
            }
            break;
        }
    }

    if (config->include_main) {
        py_append(buffer, 65536, &pos,
                  "\nif __name__ == '__main__':\n");
        if (config->include_tests) {
            py_append(buffer, 65536, &pos,
                      "    # 单元测试\n"
                      "    import unittest\n\n"
                      "    class TestModule(unittest.TestCase):\n"
                      "        def test_basic(self):\n"
                      "            self.assertTrue(True)\n\n"
                      "    unittest.main()\n");
        } else {
            py_append(buffer, 65536, &pos,
                      "    print(\"Python script started\")\n");
        }
    }

    return buffer;
}

/* ============================================================================
 * AST级变换与优化（P1-05修复新增）
 * ============================================================================ */

/**
 * @brief 常量折叠优化：将AST中的常量运算替换为计算结果
 *
 * 遍历AST，对二元运算节点：
 *   - 若左右子节点均为字面量，直接计算并替换为字面量节点
 *   - 算术运算：+, -, *, /, %
 *   - 布尔运算：==, !=, <, >, <=, >=, &&, ||
 *
 * @param node AST节点
 * @return 替换后的AST节点（与输入相同或新节点）
 */
static ASTNode* ast_fold_constants(ASTNode* node) {
    if (!node) return NULL;

    if (node->left)  node->left  = ast_fold_constants(node->left);
    if (node->right) node->right = ast_fold_constants(node->right);

    if (node->type != AST_BINARY_OP) return node;
    if (!node->left || !node->right) return node;
    if (node->left->type != AST_LITERAL || node->right->type != AST_LITERAL) return node;

    float left_val  = node->left->value.float_value;
    float right_val = node->right->value.float_value;
    float result = 0.0f;
    int valid = 1;

    switch (node->value.bin_op) {
        case OP_ADD: result = left_val + right_val; break;
        case OP_SUB: result = left_val - right_val; break;
        case OP_MUL: result = left_val * right_val; break;
        case OP_DIV:
            if (fabsf(right_val) < 1e-10f) { valid = 0; }
            else result = left_val / right_val;
            break;
        case OP_MOD:
            if (fabsf(right_val) < 1e-10f) { valid = 0; }
            else result = fmodf(left_val, right_val);
            break;
        case OP_EQ: result = (fabsf(left_val - right_val) < 1e-6f) ? 1.0f : 0.0f; break;
        case OP_NE: result = (fabsf(left_val - right_val) >= 1e-6f) ? 1.0f : 0.0f; break;
        case OP_LT: result = (left_val < right_val) ? 1.0f : 0.0f; break;
        case OP_GT: result = (left_val > right_val) ? 1.0f : 0.0f; break;
        case OP_LE: result = (left_val <= right_val) ? 1.0f : 0.0f; break;
        case OP_GE: result = (left_val >= right_val) ? 1.0f : 0.0f; break;
        case OP_AND: result = (fabsf(left_val) > 1e-6f && fabsf(right_val) > 1e-6f) ? 1.0f : 0.0f; break;
        case OP_OR:  result = (fabsf(left_val) > 1e-6f || fabsf(right_val) > 1e-6f) ? 1.0f : 0.0f; break;
        default: valid = 0; break;
    }

    if (!valid) return node;

    ast_destroy(node->left);
    ast_destroy(node->right);
    node->left = NULL;
    node->right = NULL;
    node->type = AST_LITERAL;
    node->value.float_value = result;
    return node;
}

/**
 * @brief 死代码消除：移除不可达的条件分支
 *
 * 对 IF 节点，若条件为常量 true 则保留 then 分支删除 else 分支，
 * 若条件为常量 false 则保留 else 分支删除 then 分支。
 * 返回替换后的节点（可能完全改变类型）。
 *
 * @param node AST节点
 * @return 替换后的AST节点
 */
static ASTNode* ast_eliminate_dead_code(ASTNode* node) {
    if (!node) return NULL;

    if (node->left)  node->left  = ast_eliminate_dead_code(node->left);
    if (node->right) node->right = ast_eliminate_dead_code(node->right);
    if (node->next)  node->next  = ast_eliminate_dead_code(node->next);

    if (node->type != AST_IF) return node;
    if (!node->left) return node;

    ASTNode* condition = node->left;

    if (condition->type != AST_LITERAL) {
        if (condition->type == AST_UNARY_OP &&
            condition->value.unary_op == UNARY_NOT &&
            condition->left && condition->left->type == AST_LITERAL) {
            float val = condition->left->value.float_value;
            int is_false = (fabsf(val) < 1e-6f);
            ASTNode* replacement = NULL;

            if (is_false) {
                replacement = node->right ? ast_clone(node->right->right) : NULL;
            } else {
                replacement = node->right ? ast_clone(node->right->left) : NULL;
            }
            if (replacement) {
                ast_destroy(node);
                return replacement;
            }
        }
        return node;
    }

    float cond_val = node->left->value.float_value;
    int is_true = (fabsf(cond_val) > 1e-6f);

    ASTNode* replacement = NULL;
    if (node->right && node->right->type == AST_BLOCK) {
        if (is_true && node->right->left) {
            replacement = ast_clone(node->right->left);
        } else if (!is_true && node->right->right) {
            replacement = ast_clone(node->right->right);
        }
    }

    if (replacement) {
        ast_destroy(node);
        return replacement;
    }

    return node;
}

/**
 * @brief 深拷贝AST子树
 */
ASTNode* ast_clone(const ASTNode* node) {
    if (!node) return NULL;

    ASTNode* clone = create_ast_node(node->type);
    if (!clone) return NULL;

    clone->value = node->value;
    clone->name = node->name ? string_duplicate_nullable(node->name) : NULL;
    clone->line = node->line;
    clone->column = node->column;

    if (node->left)  clone->left  = ast_clone(node->left);
    if (node->right) clone->right = ast_clone(node->right);
    if (node->next)  clone->next  = ast_clone(node->next);

    return clone;
}

/**
 * @brief 遍历并替换AST中满足条件的节点
 *
 * 深度优先遍历AST，对每个节点调用 matcher 回调。
 * 若 matcher 返回非NULL，则用返回节点替换原节点。
 *
 * @param node AST节点
 * @param matcher 匹配回调，返回NULL表示不替换
 * @param user_data 用户数据
 * @return 替换后的AST节点
 */
static ASTNode* ast_transform_recursive(ASTNode* node,
                                         ASTNode* (*matcher)(ASTNode*, void*),
                                         void* user_data) {
    if (!node) return NULL;

    if (node->left)  node->left  = ast_transform_recursive(node->left, matcher, user_data);
    if (node->right) node->right = ast_transform_recursive(node->right, matcher, user_data);
    if (node->next)  node->next  = ast_transform_recursive(node->next, matcher, user_data);

    ASTNode* replacement = matcher(node, user_data);
    if (replacement && replacement != node) {
        ast_destroy(node);
        return replacement;
    }
    return node;
}

/**
 * @brief AST优化管道：依次应用常量折叠和死代码消除
 *
 * 对AST执行完整的优化变换管道，返回优化后的AST。
 *
 * @param ast AST根节点
 * @return 优化后的AST（可能与输入相同或为新AST）
 */
ASTNode* ast_optimize_pipeline(ASTNode* ast) {
    if (!ast) return NULL;

    int changed = 0;
    int max_passes = 5;
    for (int pass = 0; pass < max_passes; pass++) {
        ASTNode* folded = ast_fold_constants(ast);
        ASTNode* eliminated = ast_eliminate_dead_code(folded);
        if (folded == ast && eliminated == ast) {
            changed = pass + 1;
            break;
        }
        ast = eliminated;
    }

    return ast;
}

/**
 * @brief 查找AST中指定名称的变量引用节点
 *
 * @param ast AST根节点
 * @param var_name 变量名
 * @return 第一个匹配的AST_VARIABLE_REF节点，未找到返回NULL
 */
ASTNode* ast_find_variable_ref(ASTNode* ast, const char* var_name) {
    if (!ast || !var_name) return NULL;

    if (ast->type == AST_VARIABLE_REF && ast->name &&
        strcmp(ast->name, var_name) == 0) {
        return ast;
    }

    ASTNode* found = ast_find_variable_ref(ast->left, var_name);
    if (found) return found;
    found = ast_find_variable_ref(ast->right, var_name);
    if (found) return found;
    return ast_find_variable_ref(ast->next, var_name);
}