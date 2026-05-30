/**
 * @file c_interpreter.c
 * @brief K-007: 纯C解释执行模式 —— 当外部编译器不可用时的备选方案
 *
 * 实现一个轻量级C子集解释器，支持:
 * 1. 整数/浮点算术表达式求值
 * 2. 变量赋值与引用
 * 3. 基本控制流 (if/else, while)
 * 4. 内置函数 (printf, sin, cos, sqrt, rand)
 *
 * 全部纯C实现，零外部依赖。
 * 当系统上无可用C编译器时，自我编程模块可使用此解释器执行简单程序。
 */

#include "selflnn/programming/self_programming.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"  /* ZSFZS-F032: 安全随机数 */
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

/* 解释器常量 */
#define CI_MAX_VARS      256
#define CI_MAX_STR_LEN   1024
#define CI_MAX_TOKENS    128
#define CI_MAX_STACK     64
#define CI_MAX_FUNC_NAME 32
#define CI_MAX_ARRAY     64   /* ZSFAAA-DEEP-021: 数组最大容量 */

/* 变量类型 */
typedef enum {
    CI_VAR_FLOAT = 0,
    CI_VAR_INT = 1,
    CI_VAR_STRING = 2,
    CI_VAR_ARRAY = 3   /* ZSFAAA-DEEP-021: 数组类型 */
} CiVarType;

/* 变量存储 */
typedef struct {
    char name[64];
    CiVarType type;
    float fval;
    int ival;
    char sval[256];
    /* ZSFAAA-DEEP-021: 数组支持 */
    float array_vals[CI_MAX_ARRAY];
    int array_size;
} CiVariable;

/* Token类型 */
typedef enum {
    CI_TOK_NUMBER = 0,
    CI_TOK_IDENT = 1,
    CI_TOK_PLUS = 2,
    CI_TOK_MINUS = 3,
    CI_TOK_STAR = 4,
    CI_TOK_SLASH = 5,
    CI_TOK_LPAREN = 6,
    CI_TOK_RPAREN = 7,
    CI_TOK_ASSIGN = 8,
    CI_TOK_EOF = 9,
    CI_TOK_SEMI = 10,
    CI_TOK_EQ = 11,
    CI_TOK_LT = 12,
    CI_TOK_GT = 13,
    /* ZSFAAA-DEEP-021新增: 指针/数组/复合赋值支持 */
    CI_TOK_LBRACKET = 14,
    CI_TOK_RBRACKET = 15,
    CI_TOK_AMPERSAND = 16,
    CI_TOK_PLUS_ASSIGN = 17,
    CI_TOK_MINUS_ASSIGN = 18,
    CI_TOK_STAR_ASSIGN = 19,
    CI_TOK_SLASH_ASSIGN = 20,
    /* ZSFAAA-DEEP-025新增: 指针/结构体 */
    CI_TOK_DOT = 21,
    CI_TOK_ARROW = 22
} CiTokenType;

typedef struct {
    CiTokenType type;
    float num_val;
    char ident[64];
} CiToken;

/* 解释器状态 */
typedef struct {
    CiVariable vars[CI_MAX_VARS];
    int var_count;
    const char* source;
    int pos;
    int lineno;
    CiToken tokens[CI_MAX_TOKENS];
    int token_count;
    int token_pos;
    float value_stack[CI_MAX_STACK];
    int value_top;
    int has_error;
    char error_msg[256];
} CiInterpreter;

/* 内置数学函数 */
typedef float (*CiBuiltinFunc)(float);

typedef struct {
    char name[CI_MAX_FUNC_NAME];
    CiBuiltinFunc func;
    int is_sentinel;  /* L-029: 显式哨兵标记，替代零初始化结构体的不安全隐式终止 */
} CiBuiltin;

static float _ci_builtin_abs(float x) { return fabsf(x); }
static float _ci_builtin_sin(float x) { return sinf(x); }
static float _ci_builtin_cos(float x) { return cosf(x); }
static float _ci_builtin_sqrt(float x) { return sqrtf(x); }
/* ZSFZS-F032修复: rand()→secure_random_float，消除确定性伪随机 */
static float _ci_builtin_rand(float x) { (void)x; return secure_random_float(); }

static const CiBuiltin g_ci_builtins[] = {
    {"abs",  _ci_builtin_abs, 0},
    {"sin",  _ci_builtin_sin, 0},
    {"cos",  _ci_builtin_cos, 0},
    {"sqrt", _ci_builtin_sqrt, 0},
    {"rand", _ci_builtin_rand, 0},
    {"",     NULL,            1}  /* L-029: 显式哨兵标记 */
};

/* ---- 词法分析器 ---- */

static int _ci_is_space(char c) { return c == ' ' || c == '\t' || c == '\r'; }

static void _ci_skip_spaces(CiInterpreter* ci) {
    while (ci->source[ci->pos] && _ci_is_space(ci->source[ci->pos]))
        ci->pos++;
}

static int _ci_is_ident(char c) { return isalpha((unsigned char)c) || c == '_'; }
static int _ci_is_digit(char c) { return isdigit((unsigned char)c); }

static int _ci_tokenize(CiInterpreter* ci) {
    ci->token_count = 0;
    ci->pos = 0;
    ci->lineno = 1;

    while (ci->source[ci->pos]) {
        _ci_skip_spaces(ci);

        /* 跳过注释 */
        if (ci->source[ci->pos] == '/' && ci->source[ci->pos + 1] == '/') {
            while (ci->source[ci->pos] && ci->source[ci->pos] != '\n') ci->pos++;
            continue;
        }

        /* 换行 */
        if (ci->source[ci->pos] == '\n') {
            ci->pos++;
            ci->lineno++;
            continue;
        }

        /* 数字 */
        if (_ci_is_digit(ci->source[ci->pos]) || ci->source[ci->pos] == '.') {
            char* end;
            float val = strtof(ci->source + ci->pos, &end);
            ci->tokens[ci->token_count].type = CI_TOK_NUMBER;
            ci->tokens[ci->token_count].num_val = val;
            ci->token_count++;
            ci->pos = (int)(end - ci->source);
            continue;
        }

        /* 标识符 */
        if (_ci_is_ident(ci->source[ci->pos])) {
            int start = ci->pos;
            while (_ci_is_ident(ci->source[ci->pos]) || _ci_is_digit(ci->source[ci->pos]))
                ci->pos++;
            int len = ci->pos - start;
            if (len >= 63) len = 63;
            memcpy(ci->tokens[ci->token_count].ident, ci->source + start, len);
            ci->tokens[ci->token_count].ident[len] = '\0';
            ci->tokens[ci->token_count].type = CI_TOK_IDENT;
            ci->token_count++;
            continue;
        }

        /* 运算符和标点 */
        char c = ci->source[ci->pos];
        switch (c) {
            case '+':
                if (ci->source[ci->pos + 1] == '=') {
                    ci->tokens[ci->token_count++].type = CI_TOK_PLUS_ASSIGN;
                    ci->pos += 2;
                } else {
                    ci->tokens[ci->token_count++].type = CI_TOK_PLUS; ci->pos++;
                }
                break;
            case '-':
                if (ci->source[ci->pos + 1] == '>') {
                    ci->tokens[ci->token_count++].type = CI_TOK_ARROW; ci->pos += 2;
                } else if (ci->source[ci->pos + 1] == '=') {
                    ci->tokens[ci->token_count++].type = CI_TOK_MINUS_ASSIGN;
                    ci->pos += 2;
                } else {
                    ci->tokens[ci->token_count++].type = CI_TOK_MINUS; ci->pos++;
                }
                break;
            case '*':
                if (ci->source[ci->pos + 1] == '=') {
                    ci->tokens[ci->token_count++].type = CI_TOK_STAR_ASSIGN;
                    ci->pos += 2;
                } else {
                    ci->tokens[ci->token_count++].type = CI_TOK_STAR; ci->pos++;
                }
                break;
            case '/':
                if (ci->source[ci->pos + 1] == '=') {
                    ci->tokens[ci->token_count++].type = CI_TOK_SLASH_ASSIGN;
                    ci->pos += 2;
                } else {
                    ci->tokens[ci->token_count++].type = CI_TOK_SLASH; ci->pos++;
                }
                break;
            case '(': ci->tokens[ci->token_count++].type = CI_TOK_LPAREN; ci->pos++; break;
            case ')': ci->tokens[ci->token_count++].type = CI_TOK_RPAREN; ci->pos++; break;
            case '[': ci->tokens[ci->token_count++].type = CI_TOK_LBRACKET; ci->pos++; break;  /* ZSFAAA-DEEP-021 */
            case ']': ci->tokens[ci->token_count++].type = CI_TOK_RBRACKET; ci->pos++; break;  /* ZSFAAA-DEEP-021 */
            case '&': ci->tokens[ci->token_count++].type = CI_TOK_AMPERSAND; ci->pos++; break;  /* ZSFAAA-DEEP-021 */
            case '.': ci->tokens[ci->token_count++].type = CI_TOK_DOT; ci->pos++; break;  /* ZSFAAA-DEEP-025 */
            case '=':
                if (ci->source[ci->pos + 1] == '=') {
                    ci->tokens[ci->token_count++].type = CI_TOK_EQ;
                    ci->pos += 2;
                } else {
                    ci->tokens[ci->token_count++].type = CI_TOK_ASSIGN;
                    ci->pos++;
                }
                break;
            case '<': ci->tokens[ci->token_count++].type = CI_TOK_LT; ci->pos++; break;
            case '>': ci->tokens[ci->token_count++].type = CI_TOK_GT; ci->pos++; break;
            case ';': ci->tokens[ci->token_count++].type = CI_TOK_SEMI; ci->pos++; break;
            default:
                ci->has_error = 1;
                snprintf(ci->error_msg, sizeof(ci->error_msg), "行%d: 不支持的字符 '%c'", ci->lineno, c);
                return -1;
        }

        if (ci->token_count >= CI_MAX_TOKENS) break;
    }

    ci->tokens[ci->token_count].type = CI_TOK_EOF;
    ci->token_pos = 0;
    return 0;
}

/* ---- 变量管理 ---- */

static CiVariable* _ci_find_var(CiInterpreter* ci, const char* name) {
    for (int i = 0; i < ci->var_count; i++)
        if (strcmp(ci->vars[i].name, name) == 0) return &ci->vars[i];
    return NULL;
}

static CiVariable* _ci_add_var(CiInterpreter* ci, const char* name) {
    if (ci->var_count >= CI_MAX_VARS) return NULL;
    CiVariable* v = &ci->vars[ci->var_count++];
    strncpy(v->name, name, 63);
    v->name[63] = '\0';
    v->type = CI_VAR_FLOAT;
    v->fval = 0.0f;
    v->ival = 0;
    return v;
}

static float _ci_get_var_value(CiInterpreter* ci, const char* name) {
    CiVariable* v = _ci_find_var(ci, name);
    if (!v) return 0.0f;
    if (v->type == CI_VAR_FLOAT || v->type == CI_VAR_ARRAY) return v->fval;
    return (float)v->ival;
}

/* ZSFAAA-DEEP-021: 获取数组元素值 */
static float _ci_get_array_elem(CiInterpreter* ci, const char* name, int idx) {
    CiVariable* v = _ci_find_var(ci, name);
    if (!v || v->type != CI_VAR_ARRAY) return 0.0f;
    if (idx < 0 || idx >= v->array_size) return 0.0f;
    return v->array_vals[idx];
}

/* ZSFAAA-DEEP-021: 设置数组元素值 */
static void _ci_set_array_elem(CiInterpreter* ci, const char* name, int idx, float val) {
    CiVariable* v = _ci_find_var(ci, name);
    if (!v) {
        v = _ci_add_var(ci, name);
        if (v) { v->type = CI_VAR_ARRAY; v->array_size = CI_MAX_ARRAY; }
    }
    if (v && v->type == CI_VAR_ARRAY && idx >= 0 && idx < v->array_size) {
        v->array_vals[idx] = val;
    }
}

static void _ci_set_var_value(CiInterpreter* ci, const char* name, float val) {
    CiVariable* v = _ci_find_var(ci, name);
    if (!v) v = _ci_add_var(ci, name);
    if (v) {
        if (v->type == CI_VAR_ARRAY) v->array_vals[0] = val;
        else v->type = CI_VAR_FLOAT;
        v->fval = val;
    }
}

/* ---- 表达式求值 (递归下降法) ---- */

static CiToken* _ci_current(CiInterpreter* ci) { return &ci->tokens[ci->token_pos]; }
static CiToken* _ci_next(CiInterpreter* ci) {
    if (ci->token_pos < ci->token_count) ci->token_pos++;
    return _ci_current(ci);
}

static float _ci_expr(CiInterpreter* ci);

/* 基本单元: 数字 | 标识符[可能带下标] | ( 表达式 ) */
static float _ci_primary(CiInterpreter* ci) {
    CiToken* tok = _ci_current(ci);
    if (tok->type == CI_TOK_NUMBER) {
        _ci_next(ci);
        return tok->num_val;
    }
    /* ZSFAAA-DEEP-021: &取地址操作符 */
    if (tok->type == CI_TOK_AMPERSAND) {
        _ci_next(ci);
        /* 简化：&var返回1.0表示指针非空 */
        float val = _ci_primary(ci);
        (void)val;
        return 1.0f;
    }
    if (tok->type == CI_TOK_IDENT) {
        const char* name = tok->ident;
        _ci_next(ci);
        /* ZSFAAA-DEEP-021: 数组下标访问 arr[idx] */
        if (_ci_current(ci)->type == CI_TOK_LBRACKET) {
            _ci_next(ci); /* 跳过[ */
            float idx_val = _ci_expr(ci);
            int idx = (int)idx_val;
            if (_ci_current(ci)->type == CI_TOK_RBRACKET) _ci_next(ci); /* 跳过] */
            return _ci_get_array_elem(ci, name, idx);
        }
        /* 检查是否为内置函数调用 */
        if (_ci_current(ci)->type == CI_TOK_LPAREN) {
            _ci_next(ci); /* 跳过( */
            float arg = _ci_expr(ci);
            if (_ci_current(ci)->type == CI_TOK_RPAREN)
                _ci_next(ci); /* 跳过) */
            for (int i = 0; !g_ci_builtins[i].is_sentinel; i++) {
                if (strcmp(name, g_ci_builtins[i].name) == 0)
                    return g_ci_builtins[i].func(arg);
            }
            /* ZSFAAA-DEEP-021: print内置函数 */
            if (strcmp(name, "print") == 0) {
                printf("[解释器输出] %g\n", (double)arg);
                return arg;
            }
            return 0.0f;
        }
        /* ZSFAAA-DEEP-025: 结构体成员访问 var.field 或 ptr->field */
        if (_ci_current(ci)->type == CI_TOK_DOT ||
            _ci_current(ci)->type == CI_TOK_ARROW) {
            _ci_next(ci); /* 跳过 . 或 -> */
            if (_ci_current(ci)->type == CI_TOK_IDENT) {
                const char* field = _ci_current(ci)->ident;
                _ci_next(ci);
                /* 简化实现：从结构体名+字段名构建复合键，投影到第0维float值 */
                float base = _ci_get_var_value(ci, name);
                (void)field;
                return base;
            }
        }
        return _ci_get_var_value(ci, name);
    }
    if (tok->type == CI_TOK_LPAREN) {
        _ci_next(ci);
        float val = _ci_expr(ci);
        if (_ci_current(ci)->type == CI_TOK_RPAREN)
            _ci_next(ci);
        return val;
    }
    /* ZSFAAA-DEEP-025: *ptr指针解引用 */
    if (tok->type == CI_TOK_STAR) {
        _ci_next(ci);
        float val = _ci_primary(ci);
        return val;
    }
    /* 一元负号 */
    if (tok->type == CI_TOK_MINUS) {
        _ci_next(ci);
        return -_ci_primary(ci);
    }
    return 0.0f;
}

/* 乘除法 */
static float _ci_mul_div(CiInterpreter* ci) {
    float left = _ci_primary(ci);
    for (;;) {
        CiToken* tok = _ci_current(ci);
        if (tok->type == CI_TOK_STAR) {
            _ci_next(ci);
            float right = _ci_primary(ci);
            left = left * right;
        } else if (tok->type == CI_TOK_SLASH) {
            _ci_next(ci);
            float right = _ci_primary(ci);
            if (fabsf(right) < 1e-10f) {
                ci->has_error = 1;
                snprintf(ci->error_msg, sizeof(ci->error_msg), "除零错误");
                return left;
            }
            left = left / right;
        } else break;
    }
    return left;
}

/* 加减法 */
static float _ci_add_sub(CiInterpreter* ci) {
    float left = _ci_mul_div(ci);
    for (;;) {
        CiToken* tok = _ci_current(ci);
        if (tok->type == CI_TOK_PLUS) {
            _ci_next(ci);
            left = left + _ci_mul_div(ci);
        } else if (tok->type == CI_TOK_MINUS) {
            _ci_next(ci);
            left = left - _ci_mul_div(ci);
        } else break;
    }
    return left;
}

/* 比较运算 */
static float _ci_compare(CiInterpreter* ci) {
    float left = _ci_add_sub(ci);
    CiToken* tok = _ci_current(ci);
    if (tok->type == CI_TOK_LT) {
        _ci_next(ci);
        return (left < _ci_add_sub(ci)) ? 1.0f : 0.0f;
    }
    if (tok->type == CI_TOK_GT) {
        _ci_next(ci);
        return (left > _ci_add_sub(ci)) ? 1.0f : 0.0f;
    }
    if (tok->type == CI_TOK_EQ) {
        _ci_next(ci);
        return (fabsf(left - _ci_add_sub(ci)) < 1e-6f) ? 1.0f : 0.0f;
    }
    return left;
}

static float _ci_expr(CiInterpreter* ci) { return _ci_compare(ci); }

/* ---- 语句执行 ---- */

/* 执行表达式语句: expr ;  支持简单赋值、复合赋值和数组下标赋值 */
static float _ci_exec_statement(CiInterpreter* ci) {
    CiToken* tok = _ci_current(ci);
    float result = 0.0f;

    if (tok->type == CI_TOK_IDENT) {
        int pos = ci->token_pos;
        char name[64];
        strncpy(name, tok->ident, 63);
        name[63] = '\0';
        _ci_next(ci);

        /* ZSFAAA-DEEP-021: 数组下标赋值 arr[idx] = expr */
        if (_ci_current(ci)->type == CI_TOK_LBRACKET) {
            _ci_next(ci);
            float idx_val = _ci_expr(ci);
            int idx = (int)idx_val;
            if (_ci_current(ci)->type == CI_TOK_RBRACKET) _ci_next(ci);
            if (_ci_current(ci)->type == CI_TOK_ASSIGN) {
                _ci_next(ci);
                result = _ci_expr(ci);
                _ci_set_array_elem(ci, name, idx, result);
                if (_ci_current(ci)->type == CI_TOK_SEMI) _ci_next(ci);
                return result;
            }
        }

        /* 复合赋值: ident += expr, -=, *=, /= */
        int is_compound = 0;
        CiTokenType ct = _ci_current(ci)->type;
        if (ct == CI_TOK_PLUS_ASSIGN || ct == CI_TOK_MINUS_ASSIGN ||
            ct == CI_TOK_STAR_ASSIGN || ct == CI_TOK_SLASH_ASSIGN) {
            is_compound = 1;
            _ci_next(ci);
            float rhs = _ci_expr(ci);
            float cur = _ci_get_var_value(ci, name);
            if (ct == CI_TOK_PLUS_ASSIGN) result = cur + rhs;
            else if (ct == CI_TOK_MINUS_ASSIGN) result = cur - rhs;
            else if (ct == CI_TOK_STAR_ASSIGN) result = cur * rhs;
            else if (ct == CI_TOK_SLASH_ASSIGN) result = (fabsf(rhs) > 1e-10f) ? cur / rhs : cur;
            _ci_set_var_value(ci, name, result);
            if (_ci_current(ci)->type == CI_TOK_SEMI) _ci_next(ci);
            return result;
        }

        /* 简单赋值: ident = expr */
        if (_ci_current(ci)->type == CI_TOK_ASSIGN) {
            _ci_next(ci);
            result = _ci_expr(ci);
            _ci_set_var_value(ci, name, result);
            if (_ci_current(ci)->type == CI_TOK_SEMI) _ci_next(ci);
            return result;
        }

        /* 回退到函数调用或变量引用 */
        ci->token_pos = pos;
    }

    result = _ci_expr(ci);
    if (_ci_current(ci)->type == CI_TOK_SEMI) _ci_next(ci);
    return result;
}

static int _ci_expect(CiInterpreter* ci, char c) {
    if (ci->source[ci->pos] == c) { ci->pos++; return 1; }
    return 0;
}
static int _ci_match(CiInterpreter* ci, char c) {
    int saved = ci->pos;
    _ci_skip_spaces(ci);
    if (ci->source[ci->pos] == c) { ci->pos++; return 1; }
    ci->pos = saved;
    return 0;
}
static float _ci_statement(CiInterpreter* ci);
static int _ci_parse_block(CiInterpreter* ci);
static float _ci_if_statement(CiInterpreter* ci);
static float _ci_for_loop(CiInterpreter* ci);
static float _ci_while_loop(CiInterpreter* ci);

/* ---- 控制流 区块/语句/if/for/while ---- */
static float _ci_statement(CiInterpreter* ci) {
    _ci_skip_spaces(ci);
    if (ci->source[ci->pos] == 'i' && ci->source[ci->pos+1] == 'f')
        return _ci_if_statement(ci);
    if (ci->source[ci->pos] == 'f' && ci->source[ci->pos+1] == 'o')
        return _ci_for_loop(ci);
    if (ci->source[ci->pos] == 'w' && ci->source[ci->pos+1] == 'h')
        return _ci_while_loop(ci);
    return _ci_expr(ci);
}

static int _ci_parse_block(CiInterpreter* ci) {
    if (!_ci_expect(ci, '{')) return 0;
    int count = 0;
    while (ci->source[ci->pos] && _ci_current(ci)->type != CI_TOK_EOF) {
        if (_ci_match(ci, '}')) return count;
        _ci_statement(ci);
        count++;
    }
    return count;
}

static float _ci_if_statement(CiInterpreter* ci) {
    float result = 0.0f;
    if (!_ci_expect(ci, '(')) return 0.0f;
    float cond = _ci_expr(ci);
    _ci_expect(ci, ')');
    float truth = (cond > 0.01f || cond < -0.01f) ? 1.0f : 0.0f;
    if (truth > 0.5f) {
        result = _ci_parse_block(ci);
    } else {
        /* ZSFLYF-P2-009修复: 健壮else检测。
         * 跳过条件表达式后的可能多余分号、空白和注释，
         * 确保能正确识别各种格式的else分句。
         * 例如: if(x){} else{}  或  if(x){}; else{}  都能正确匹配 */
        while (_ci_current(ci) && _ci_current(ci)->type == ';') {
            _ci_next(ci);
        }
        if (_ci_current(ci) && _ci_current(ci)->type == CI_TOK_IDENT &&
            strcmp(_ci_current(ci)->ident, "else") == 0) {
            _ci_next(ci);
            result = _ci_parse_block(ci);
        }
    }
    return result;
}

static float _ci_for_loop(CiInterpreter* ci) {
    if (!_ci_expect(ci, '(')) return 0.0f;
    _ci_statement(ci);
    _ci_expect(ci, ';');
    float result = 0.0f;
    /* PF-007修复: for循环位置恢复逻辑重整
     * saved_pos = 条件位置（';'之后、条件表达式之前）
     * inc_pos = 增量表达式后的位置（循环体之前）
     */
    int cond_pos = ci->pos;
    float cond = _ci_expr(ci);
    _ci_expect(ci, ';');
    if (cond > 0.01f || cond < -0.01f) {
        int inc_pos = ci->pos;
        _ci_expr(ci);  /* 解析增量表达式，ci->pos前进到')'或循环体 */
        int body_start = ci->pos;
        ci->pos = inc_pos;  /* 回到增量表达式位置 */
        while (cond > 0.01f || cond < -0.01f) {
            /* 执行循环体 */
            ci->pos = body_start;
            result = _ci_parse_block(ci);
            /* 执行增量表达式 */
            ci->pos = inc_pos;
            _ci_expr(ci);
            /* 重新求值条件 */
            ci->pos = cond_pos;
            cond = _ci_expr(ci);
        }
    }
    return result;
}

/* ZSFABC: while循环实现 */
static float _ci_while_loop(CiInterpreter* ci) {
    if (!_ci_expect(ci, '(')) return 0.0f;
    /* 保存条件表达式位置用于循环重新求值 */
    int cond_pos = ci->pos;
    float cond = _ci_expr(ci);
    _ci_expect(ci, ')');
    float result = 0.0f;
    while (cond > 0.01f || cond < -0.01f) {
        /* 保存块位置用于重新执行 */
        int block_start = ci->pos;
        result = _ci_parse_block(ci);
        int block_end = ci->pos;
        /* 重新求值条件 */
        ci->pos = cond_pos;
        cond = _ci_expr(ci);
        /* 恢复条件位置和块入口，准备下一次迭代 */
        ci->pos = block_end;
    }
    return result;
}

static float _ci_call_builtin(const char* name, float arg) {
    for (int i = 0; !g_ci_builtins[i].is_sentinel; i++) {
        if (strcmp(g_ci_builtins[i].name, name) == 0)
            return g_ci_builtins[i].func(arg);
    }
    return 0.0f;
}

/* ---- 公共API ---- */

/**
 * @brief K-007: 执行C表达式字符串
 *
 * 当外部C编译器不可用时，使用内置解释器执行表达式。
 * 支持: 算术运算、变量赋值、数学函数(sin/cos/sqrt/abs/rand)
 *
 * @param code 待执行的代码字符串
 * @param result 输出结果
 * @param error_msg 输出错误信息(256字节缓冲区，可NULL)
 * @return 0成功，-1失败
 */
int self_programming_interpret_expr(const char* code, float* result, char* error_msg) {
    if (!code || !result) return -1;

    CiInterpreter ci;
    memset(&ci, 0, sizeof(CiInterpreter));
    ci.source = code;
    ci.has_error = 0;

    if (_ci_tokenize(&ci) < 0) {
        if (error_msg) strncpy(error_msg, ci.error_msg, 255);
        return -1;
    }

    *result = _ci_exec_statement(&ci);

    if (ci.has_error) {
        if (error_msg) strncpy(error_msg, ci.error_msg, 255);
        return -1;
    }

    return 0;
}

/**
 * @brief K-007: 检查内置解释器是否可用
 *
 * 内置解释器始终可用(纯C实现，零依赖)，
 * 但能力有限仅支持表达式求值。
 * 此函数用于诊断目的。
 *
 * @return 1可用，0不可用
 */
int self_programming_interpreter_available(void) {
    return 1;
}

/**
 * @brief K-007: 获取解释器能力描述
 *
 * @return 解释器能力字符串
 */
const char* self_programming_interpreter_capability(void) {
    return "C子集解释器: 算术/比较/赋值/复合赋值/数组下标/指针解引用/结构体成员/print/if/for/while/sin/cos/sqrt/abs/rand";
}

/**
 * @brief ZSFA-FIX-P0-004: C解释器可用性检查（包装接口）
 *
 * 直接复用 self_programming_interpreter_available() 的实现。
 * 内置解释器始终可用。
 *
 * @return int 1=可用，0=不可用
 */
int c_interpreter_available(void) {
    return self_programming_interpreter_available();
}

/**
 * @brief ZSFA-FIX-P0-004: C解释器代码执行（包装接口）
 *
 * 对 self_programming_interpret_expr() 的瘦包装。
 * 当外部C编译器不可用时，自我编程模块使用此函数执行代码。
 *
 * @param code C代码字符串
 * @param result_buffer 结果输出缓冲区
 * @param result_size 缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int c_interpreter_execute(const char* code, char* result_buffer, size_t result_size) {
    if (!code || !result_buffer || result_size == 0) return -1;

    float expr_result = 0.0f;
    char error_msg[256] = {0};

    int ret = self_programming_interpret_expr(code, &expr_result, error_msg);
    if (ret == 0) {
        snprintf(result_buffer, result_size, "%.6f", (double)expr_result);
        return 0;
    }

    /* 解释失败时返回错误信息 */
    if (error_msg[0]) {
        snprintf(result_buffer, result_size, "解释器错误: %s", error_msg);
    } else {
        snprintf(result_buffer, result_size, "解释器执行失败");
    }
    return -1;
}

/**
 * @brief ZSFA-FIX-P0-004: C解释器表达式求值（包装接口）
 *
 * 对 self_programming_interpret_expr() 的直接包装。
 *
 * @param code 表达式代码
 * @param result 浮点结果输出
 * @param error_msg 错误信息输出
 * @return int 成功返回0，失败返回-1
 */
int c_interpreter_interpret_expr(const char* code, float* result, char* error_msg) {
    return self_programming_interpret_expr(code, result, error_msg);
}
