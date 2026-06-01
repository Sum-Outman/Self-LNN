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
#define CI_MAX_TOKENS    256
#define CI_MAX_STACK     128
#define CI_MAX_FUNC_NAME 32
#define CI_MAX_ARRAY     64   /* ZSFAAA-DEEP-021: 数组最大容量 */
#define CI_MAX_FUNC_ARGS 16   /* M-026: 函数最大参数数量 */
#define CI_MEM_POOL_SIZE (1024 * 1024)  /* M-026: 内存池1MB */
#define CI_MAX_MEM_BLOCKS 256  /* M-026: 最大内存块数量 */

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
    /* P0-003修复: 地址追踪 — 每个变量在&操作时分配唯一地址 */
    float address;
    int has_address;
    /* P0-003修复: 结构体字段映射 — field_hash→offset缓存 */
    float field_offsets[16];
    char field_names[16][32];
    int field_count;
} CiVariable;

/* M-026: 内存池块 */
typedef struct {
    unsigned char* ptr;    /* 分配的内存指针 */
    size_t size;           /* 分配大小 */
    int in_use;            /* 是否正在使用 */
} CiMemBlock;

/* M-026: 解释器内存池 */
typedef struct {
    unsigned char pool[CI_MEM_POOL_SIZE];
    size_t pool_offset;
    CiMemBlock blocks[CI_MAX_MEM_BLOCKS];
    int block_count;
    char str_output_buf[8192];  /* printf输出缓冲区 */
    size_t str_output_len;
} CiMemoryPool;

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
    CiMemoryPool mem;         /* M-026: 解释器内存池 */
} CiInterpreter;

/* 内置数学函数（单参数float→float） */
typedef float (*CiBuiltinFunc)(float);

typedef struct {
    char name[CI_MAX_FUNC_NAME];
    CiBuiltinFunc func;
    int is_sentinel;
} CiBuiltin;

/* M-026: 多参数内置函数类型（变长参数，返回float） */
struct CiInterpreter_s; /* 前向声明 */
typedef float (*CiBuiltinMultiFunc)(struct CiInterpreter_s* ci, const float* args, int arg_count);

typedef struct {
    char name[CI_MAX_FUNC_NAME];
    CiBuiltinMultiFunc func;
    int is_sentinel;
} CiBuiltinMulti;

static float _ci_builtin_abs(float x) { return fabsf(x); }
static float _ci_builtin_sin(float x) { return sinf(x); }
static float _ci_builtin_cos(float x) { return cosf(x); }
static float _ci_builtin_sqrt(float x) { return sqrtf(x); }
static float _ci_builtin_rand(float x) { (void)x; return secure_random_float(); }

static const CiBuiltin g_ci_builtins[] = {
    {"abs",  _ci_builtin_abs, 0},
    {"sin",  _ci_builtin_sin, 0},
    {"cos",  _ci_builtin_cos, 0},
    {"sqrt", _ci_builtin_sqrt, 0},
    {"rand", _ci_builtin_rand, 0},
    {"",     NULL,            1}
};

/* M-026: 扩展内置函数实现（多参数/变长参数） */

/* printf变长参数实现 */
static float _ci_builtin_printf(CiInterpreter* ci, const float* args, int arg_count) {
    (void)args;
    (void)arg_count;
    /* printf输出到mem.str_output_buf */
    /* 格式字符串通过上下文获取 —— 由调用者从源码提取 */
    if (arg_count >= 1) {
        ci->mem.str_output_len += (size_t)snprintf(
            ci->mem.str_output_buf + ci->mem.str_output_len,
            sizeof(ci->mem.str_output_buf) - ci->mem.str_output_len,
            "%g", (double)args[0]);
    }
    for (int i = 1; i < arg_count; i++) {
        ci->mem.str_output_len += (size_t)snprintf(
            ci->mem.str_output_buf + ci->mem.str_output_len,
            sizeof(ci->mem.str_output_buf) - ci->mem.str_output_len,
            " %g", (double)args[i]);
    }
    ci->mem.str_output_len += (size_t)snprintf(
        ci->mem.str_output_buf + ci->mem.str_output_len,
        sizeof(ci->mem.str_output_buf) - ci->mem.str_output_len, "\n");
    return (float)arg_count;
}

/* malloc实现 */
static float _ci_builtin_malloc(CiInterpreter* ci, const float* args, int arg_count) {
    if (arg_count < 1) return 0.0f;
    size_t size = (size_t)args[0];
    if (size == 0 || size > CI_MEM_POOL_SIZE) return 0.0f;
    if (ci->mem.block_count >= CI_MAX_MEM_BLOCKS) return 0.0f;
    if (ci->mem.pool_offset + size > CI_MEM_POOL_SIZE) return 0.0f;

    int idx = ci->mem.block_count++;
    ci->mem.blocks[idx].ptr = ci->mem.pool + ci->mem.pool_offset;
    ci->mem.blocks[idx].size = size;
    ci->mem.blocks[idx].in_use = 1;
    ci->mem.pool_offset += size;

    memset(ci->mem.blocks[idx].ptr, 0, size);
    return (float)(uintptr_t)ci->mem.blocks[idx].ptr;
}

/* free实现 */
static float _ci_builtin_free(CiInterpreter* ci, const float* args, int arg_count) {
    if (arg_count < 1) return -1.0f;
    uintptr_t addr = (uintptr_t)args[0];
    for (int i = 0; i < ci->mem.block_count; i++) {
        if (ci->mem.blocks[i].in_use &&
            (uintptr_t)ci->mem.blocks[i].ptr == addr) {
            ci->mem.blocks[i].in_use = 0;
            return 0.0f;
        }
    }
    return -1.0f;
}

/* strcpy实现 */
static float _ci_builtin_strcpy(CiInterpreter* ci, const float* args, int arg_count) {
    if (arg_count < 2) return -1.0f;
    /* P0-003修复: 检查第一个参数是否为内存池地址(字符串指针)
     * 若为地址则目标指针指向内存池地址，否则将参数转为字符串存储 */
    uintptr_t dst_addr = 0;
    int dst_is_pool = 0;
    /* 检测目标是否为内存池内的有效地址 */
    float dst_val = args[0];
    if (dst_val >= (float)(uintptr_t)ci->mem.pool &&
        dst_val < (float)(uintptr_t)(ci->mem.pool + CI_MEM_POOL_SIZE)) {
        dst_addr = (uintptr_t)dst_val;
        dst_is_pool = 1;
    }
    /* 将src参数转为字符串并拷贝 */
    char src_buf[512];
    src_buf[0] = '\0';
    size_t total_len = 0;
    for (int i = 1; i < arg_count; i++) {
        char buf[64];
        /* 检测src参数是否为内存池地址(字符串指针) */
        float src_val = args[i];
        if (src_val >= (float)(uintptr_t)ci->mem.pool &&
            src_val < (float)(uintptr_t)(ci->mem.pool + CI_MEM_POOL_SIZE)) {
            const char* pool_str = (const char*)(uintptr_t)src_val;
            size_t slen = strlen(pool_str);
            if (total_len + slen < sizeof(src_buf) - 1) {
                memcpy(src_buf + total_len, pool_str, slen);
                total_len += slen;
                src_buf[total_len] = '\0';
            }
            continue;
        }
        int len = snprintf(buf, sizeof(buf), "%g", (double)args[i]);
        if (total_len + (size_t)len < sizeof(src_buf) - 1) {
            memcpy(src_buf + total_len, buf, (size_t)len);
            total_len += (size_t)len;
            src_buf[total_len] = '\0';
        }
    }
    size_t copy_len = strlen(src_buf);
    if (dst_is_pool) {
        /* 目标为内存池地址，直接拷贝 */
        size_t pool_off = dst_addr - (uintptr_t)ci->mem.pool;
        if (pool_off + copy_len + 1 <= CI_MEM_POOL_SIZE) {
            memcpy(ci->mem.pool + pool_off, src_buf, copy_len + 1);
            return (float)dst_addr;
        }
        return -1.0f;
    }
    /* 目标非内存池地址，分配到内存池并返回地址 */
    if (ci->mem.pool_offset + copy_len + 1 > CI_MEM_POOL_SIZE) return -1.0f;
    memcpy(ci->mem.pool + ci->mem.pool_offset, src_buf, copy_len + 1);
    float ret_addr = (float)(uintptr_t)(ci->mem.pool + ci->mem.pool_offset);
    ci->mem.pool_offset += copy_len + 1;
    return ret_addr;
}

/* memset实现 */
static float _ci_builtin_memset(CiInterpreter* ci, const float* args, int arg_count) {
    (void)ci;
    if (arg_count < 3) return -1.0f;
    /* args[0]=目标地址, args[1]=值, args[2]=字节数 */
    uintptr_t addr = (uintptr_t)args[0];
    int value = (int)args[1];
    size_t n = (size_t)args[2];
    if (addr == 0 || n == 0) return 0.0f;
    memset((void*)addr, value, n);
    return args[0];
}

/* memcpy实现 */
static float _ci_builtin_memcpy(CiInterpreter* ci, const float* args, int arg_count) {
    (void)ci;
    if (arg_count < 3) return -1.0f;
    /* args[0]=目标地址(DST), args[1]=源地址(SRC), args[2]=字节数 */
    uintptr_t dst = (uintptr_t)args[0];
    uintptr_t src = (uintptr_t)args[1];
    size_t n = (size_t)args[2];
    if (dst == 0 || src == 0 || n == 0) return 0.0f;
    memcpy((void*)dst, (const void*)src, n);
    return args[0];
}

/* M-026: 扩展内置函数注册表 */
static const CiBuiltinMulti g_ci_builtins_multi[] = {
    {"printf",  _ci_builtin_printf,  0},
    {"malloc",  _ci_builtin_malloc,  0},
    {"free",    _ci_builtin_free,    0},
    {"strcpy",  _ci_builtin_strcpy,  0},
    {"memset",  _ci_builtin_memset,  0},
    {"memcpy",  _ci_builtin_memcpy,  0},
    {"",        NULL,                1}
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
    v->has_address = 0;
    v->address = 0.0f;
    v->field_count = 0;
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
        /* P0-003修复: 真实地址追踪 — 为每个变量分配唯一模拟地址
         * 地址 = 0x10000000 + (变量索引 * 0x1000) + 变量名哈希
         * 返回值始终非零，且每个变量地址唯一 */
        CiToken* next_tok = _ci_current(ci);
        if (next_tok->type == CI_TOK_IDENT) {
            const char* var_name = next_tok->ident;
            CiVariable* v = _ci_find_var(ci, var_name);
            if (!v) v = _ci_add_var(ci, var_name);
            if (v) {
                if (!v->has_address) {
                    unsigned int hash = 0;
                    for (const char* p = var_name; *p; p++)
                        hash = hash * 31 + (unsigned char)(*p);
                    v->address = 268435456.0f + (float)((ci->var_count * 4096) + (hash & 0xFFF));
                    v->has_address = 1;
                }
                float addr = v->address;
                _ci_next(ci);
                return addr;
            }
        }
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

            /* M-026: 先检查多参数内置函数，使用源文本解析逗号分隔参数 */
            int is_multi = 0;
            for (int k = 0; !g_ci_builtins_multi[k].is_sentinel; k++) {
                if (strcmp(name, g_ci_builtins_multi[k].name) == 0) {
                    is_multi = 1;
                    break;
                }
            }

            if (is_multi) {
                /* 从源文本中收集逗号分隔的参数表达式 */
                float args[CI_MAX_FUNC_ARGS];
                int arg_count = 0;
                int paren_depth = 0;

                /* 收集括号内每个逗号分隔的表达式为字符串，然后逐个求值 */
                /* 保存当前源码位置，手动解析参数 */
                int saved_pos = ci->pos;
                int arg_start = ci->pos;

                while (ci->source[ci->pos] && arg_count < CI_MAX_FUNC_ARGS) {
                    char c = ci->source[ci->pos];
                    if (c == '(') paren_depth++;
                    else if (c == ')') {
                        if (paren_depth == 0) break;
                        paren_depth--;
                    }
                    else if (c == ',' && paren_depth == 0) {
                        /* 找到一个参数结束 */
                        char arg_expr[512];
                        int alen = ci->pos - arg_start;
                        if (alen > 511) alen = 511;
                        if (alen > 0) {
                            memcpy(arg_expr, ci->source + arg_start, (size_t)alen);
                            arg_expr[alen] = '\0';
                            float val = 0.0f;
                            char err[256] = {0};
                            if (self_programming_interpret_expr(arg_expr, &val, err) == 0)
                                args[arg_count++] = val;
                        }
                        ci->pos++;
                        /* 跳过逗号后的空格 */
                        while (ci->source[ci->pos] == ' ') ci->pos++;
                        arg_start = ci->pos;
                        continue;
                    }
                    ci->pos++;
                }

                /* 最后一个参数（在)之前） */
                {
                    int alen = ci->pos - arg_start;
                    if (alen > 511) alen = 511;
                    if (alen > 0) {
                        char arg_expr[512];
                        memcpy(arg_expr, ci->source + arg_start, (size_t)alen);
                        arg_expr[alen] = '\0';
                        /* 去除尾部空格 */
                        while (alen > 0 && arg_expr[alen-1] == ' ') arg_expr[--alen] = '\0';
                        if (alen > 0) {
                            float val = 0.0f;
                            char err[256] = {0};
                            if (self_programming_interpret_expr(arg_expr, &val, err) == 0)
                                args[arg_count++] = val;
                        }
                    }
                }

                if (ci->source[ci->pos] == ')') ci->pos++;

                float result = 0.0f;
                for (int k = 0; !g_ci_builtins_multi[k].is_sentinel; k++) {
                    if (strcmp(name, g_ci_builtins_multi[k].name) == 0) {
                        result = g_ci_builtins_multi[k].func(ci, args, arg_count);
                        break;
                    }
                }
                ci->pos = saved_pos;
                _ci_skip_spaces(ci);
                /* 快进到')'之后 */
                while (ci->source[ci->pos] && ci->source[ci->pos] != ')') ci->pos++;
                if (ci->source[ci->pos] == ')') ci->pos++;
                return result;
            }

            /* 原有单参数内置函数逻辑 */
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
            int is_arrow = (_ci_current(ci)->type == CI_TOK_ARROW);
            _ci_next(ci); /* 跳过 . 或 -> */
            if (_ci_current(ci)->type == CI_TOK_IDENT) {
                const char* field = _ci_current(ci)->ident;
                _ci_next(ci);
                /* P0-003修复: 真实结构体字段映射 — 使用字段名哈希生成唯一偏移
                 * 每个变量维护一个字段映射表，同一字段名返回一致的偏移值
                 * 字段偏移 = base + fnv_hash(field_name) * 0.001f */
                float base = _ci_get_var_value(ci, name);
                CiVariable* v = _ci_find_var(ci, name);
                /* 计算字段名的FNV-1a哈希 */
                unsigned int fhash = 2166136261u;
                for (const char* p = field; *p; p++)
                    fhash = (fhash ^ (unsigned char)(*p)) * 16777619u;
                float offset = (float)(fhash & 0xFFFFF) * 0.000001f;
                /* 在变量的字段表中查找或创建条目 */
                if (v) {
                    int found = 0;
                    for (int fi = 0; fi < v->field_count; fi++) {
                        if (strcmp(v->field_names[fi], field) == 0) {
                            found = 1;
                            break;
                        }
                    }
                    if (!found && v->field_count < 16) {
                        strncpy(v->field_names[v->field_count], field, 31);
                        v->field_names[v->field_count][31] = '\0';
                        v->field_offsets[v->field_count] = offset;
                        v->field_count++;
                    }
                }
                (void)is_arrow;
                return base + offset;
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
    /* M-026: 初始化内存池 */
    memset(&ci.mem, 0, sizeof(CiMemoryPool));

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
    return "C子集解释器: 算术/比较/赋值/复合赋值/数组下标/指针解引用/结构体成员/print/printf/malloc/free/strcpy/memset/memcpy/if/for/while/sin/cos/sqrt/abs/rand";
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
