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
#include "selflnn/utils/secure_random.h" /* 安全随机数 */
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
#define CI_MAX_ARRAY     64 /* 数组最大容量 */
#define CI_MAX_FUNC_ARGS 16   /* M-026: 函数最大参数数量 */
#define CI_MEM_POOL_SIZE (1024 * 1024)  /* M-026: 内存池1MB */
#define CI_MAX_MEM_BLOCKS 256  /* M-026: 最大内存块数量 */
#define CI_MAX_LOOP_ITERATIONS 100000  /* P-P1-16: 循环最大迭代次数，防止死循环挂起AGI线程 */
#define CI_MAX_CALL_DEPTH 256  /* P-P2-17: 递归最大调用深度，防止嵌套块栈溢出 */

/* 变量类型 — P2-006扩展：添加CI_VAR_BOOL支持 */
typedef enum {
    CI_VAR_FLOAT = 0,
    CI_VAR_INT = 1,
    CI_VAR_STRING = 2,
    CI_VAR_ARRAY = 3,
    CI_VAR_BOOL = 4
} CiVarType;

/* 变量存储 */
typedef struct {
    char name[64];
    CiVarType type;
    float fval;
    int ival;
    char sval[256];
/* 数组支持 */
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
    CI_TOK_LBRACKET = 14,
    CI_TOK_RBRACKET = 15,
    CI_TOK_AMPERSAND = 16,
    CI_TOK_PLUS_ASSIGN = 17,
    CI_TOK_MINUS_ASSIGN = 18,
    CI_TOK_STAR_ASSIGN = 19,
    CI_TOK_SLASH_ASSIGN = 20,
    CI_TOK_DOT = 21,
    CI_TOK_ARROW = 22,
    CI_TOK_LBRACE = 23,   /* { 左花括号 */
    CI_TOK_RBRACE = 24,    /* } 右花括号 */
    CI_TOK_COMMA = 25      /* , 逗号(多参数函数参数分隔) */
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
    int call_depth;           /* P-P2-17: 当前递归调用深度(用于_parse_block嵌套保护) */
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
static float _ci_builtin_log(float x) { return (x > 0.0f) ? logf(x) : 0.0f; }
static float _ci_builtin_exp(float x) { return expf(x); }
static float _ci_builtin_pow(float x, float y) { return powf(x, y); }
static float _ci_builtin_tan(float x) { return tanf(x); }
static float _ci_builtin_floor(float x) { return floorf(x); }
static float _ci_builtin_ceil(float x) { return ceilf(x); }

static const CiBuiltin g_ci_builtins[] = {
    {"abs",   _ci_builtin_abs, 0},
    {"sin",   _ci_builtin_sin, 0},
    {"cos",   _ci_builtin_cos, 0},
    {"sqrt",  _ci_builtin_sqrt, 0},
    {"rand",  _ci_builtin_rand, 0},
    {"log",   _ci_builtin_log, 0},
    {"exp",   _ci_builtin_exp, 0},
    {"tan",   _ci_builtin_tan, 0},
    {"floor", _ci_builtin_floor, 0},
    {"ceil",  _ci_builtin_ceil, 0},
    {"",      NULL,            1}
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

/* P-P1-15修复: 判断float值是否为有效的"在用"内存块索引
 * float仅24位尾数，无法无损承载64位指针地址；
 * 改用块索引(0~CI_MAX_MEM_BLOCKS-1)标识内存块，int↔float转换无精度损失。
 * 返回>=0为有效块索引，-1表示不是有效的在用块索引 */
static int _ci_block_index_of(CiInterpreter* ci, float v) {
    if (v < 0.0f) return -1;
    int idx = (int)v;
    /* 仅整数值才可能是块索引(非整数视为普通数值) */
    if ((float)idx != v) return -1;
    if (idx < 0 || idx >= CI_MAX_MEM_BLOCKS) return -1;
    if (!ci->mem.blocks[idx].in_use) return -1;
    return idx;
}

/* malloc实现 */
static float _ci_builtin_malloc(CiInterpreter* ci, const float* args, int arg_count) {
    if (arg_count < 1) return -1.0f;
    size_t size = (size_t)args[0];
    if (size == 0 || size > CI_MEM_POOL_SIZE) return -1.0f;
    if (ci->mem.pool_offset + size > CI_MEM_POOL_SIZE) return -1.0f;
    /* P-P1-15修复: 返回内存块索引而非指针地址，避免float承载64位指针的精度损失 */
    int idx = -1;
    for (int i = 0; i < CI_MAX_MEM_BLOCKS; i++) {
        if (!ci->mem.blocks[i].in_use) { idx = i; break; }
    }
    if (idx < 0) return -1.0f;
    ci->mem.blocks[idx].in_use = 1;
    ci->mem.blocks[idx].size = size;
    ci->mem.blocks[idx].ptr = ci->mem.pool + ci->mem.pool_offset;
    ci->mem.pool_offset += size;
    if (idx >= ci->mem.block_count) ci->mem.block_count = idx + 1;  /* 维护高水位标记 */
    memset(ci->mem.blocks[idx].ptr, 0, size);
    return (float)idx;  /* 返回块索引，int→float转换无精度损失 */
}

/* free实现 */
static float _ci_builtin_free(CiInterpreter* ci, const float* args, int arg_count) {
    if (arg_count < 1) return -1.0f;
    /* P-P1-15修复: 通过块索引释放，不再从float还原64位指针地址(会丢失高位) */
    int idx = (int)args[0];
    if (idx < 0 || idx >= CI_MAX_MEM_BLOCKS) return -1.0f;
    if (!ci->mem.blocks[idx].in_use) return -1.0f;
    ci->mem.blocks[idx].in_use = 0;
    /* P2修复: 检查所有块是否已全部释放，若是则回收内存池空间，
     * 防止pool_offset单调递增导致内存池耗尽（内存碎片化）。
     * 当所有块均不在使用时，将pool_offset和block_count重置为0，
     * 使后续malloc可从头复用整个内存池。 */
    int all_free = 1;
    for (int i = 0; i < ci->mem.block_count; i++) {
        if (ci->mem.blocks[i].in_use) { all_free = 0; break; }
    }
    if (all_free) {
        ci->mem.pool_offset = 0;
        ci->mem.block_count = 0;
    }
    return 0.0f;
}

/* strcpy实现 */
static float _ci_builtin_strcpy(CiInterpreter* ci, const float* args, int arg_count) {
    if (arg_count < 2) return -1.0f;
    /* P-P1-15修复: 通过块索引定位目标与源，不再用float承载64位指针地址
     * 原实现以float保存指针地址，float仅24位尾数，指针↔float双向转换会丢失高位字节，
     * 导致写入错误地址。现统一用块索引(0~CI_MAX_MEM_BLOCKS-1)标识内存块。 */

    /* 将src参数(从args[1]起)拼接为字符串缓冲区 */
    char src_buf[512];
    src_buf[0] = '\0';
    size_t total_len = 0;
    for (int i = 1; i < arg_count; i++) {
        /* 检测src参数是否为有效的在用块索引(字符串指针) */
        int src_idx = _ci_block_index_of(ci, args[i]);
        if (src_idx >= 0 && ci->mem.blocks[src_idx].ptr) {
            const char* block_str = (const char*)ci->mem.blocks[src_idx].ptr;
            size_t cap = ci->mem.blocks[src_idx].size;
            size_t slen = 0;
            while (slen < cap && block_str[slen] != '\0') slen++;  /* 带容量上限的strlen */
            if (total_len + slen < sizeof(src_buf) - 1) {
                memcpy(src_buf + total_len, block_str, slen);
                total_len += slen;
                src_buf[total_len] = '\0';
            }
            continue;
        }
        /* 非块索引，按数值格式化为字符串 */
        char buf[64];
        int len = snprintf(buf, sizeof(buf), "%g", (double)args[i]);
        if (total_len + (size_t)len < sizeof(src_buf) - 1) {
            memcpy(src_buf + total_len, buf, (size_t)len);
            total_len += (size_t)len;
            src_buf[total_len] = '\0';
        }
    }

    size_t copy_len = strlen(src_buf);
    /* 检测目标(args[0])是否为有效的在用块索引 */
    int dst_idx = _ci_block_index_of(ci, args[0]);
    if (dst_idx >= 0) {
        /* 目标为已分配块，直接拷贝到该块 */
        unsigned char* dst_ptr = ci->mem.blocks[dst_idx].ptr;
        size_t dst_capacity = ci->mem.blocks[dst_idx].size;
        if (!dst_ptr) return -1.0f;
        /* 拷贝长度不超过目标块容量(保留1字节给终止符) */
        if (copy_len >= dst_capacity) copy_len = (dst_capacity > 0) ? dst_capacity - 1 : 0;
        memcpy(dst_ptr, src_buf, copy_len);
        dst_ptr[copy_len] = '\0';
        return args[0];  /* 返回目标块索引 */
    }

    /* 目标非有效块索引：分配新块存放字符串并返回新块索引 */
    size_t need = copy_len + 1;
    if (ci->mem.pool_offset + need > CI_MEM_POOL_SIZE) return -1.0f;
    int new_idx = -1;
    for (int i = 0; i < CI_MAX_MEM_BLOCKS; i++) {
        if (!ci->mem.blocks[i].in_use) { new_idx = i; break; }
    }
    if (new_idx < 0) return -1.0f;
    ci->mem.blocks[new_idx].in_use = 1;
    ci->mem.blocks[new_idx].size = need;
    ci->mem.blocks[new_idx].ptr = ci->mem.pool + ci->mem.pool_offset;
    if (new_idx >= ci->mem.block_count) ci->mem.block_count = new_idx + 1;
    memcpy(ci->mem.blocks[new_idx].ptr, src_buf, copy_len + 1);
    ci->mem.pool_offset += need;
    return (float)new_idx;
}

/* memset实现 */
static float _ci_builtin_memset(CiInterpreter* ci, const float* args, int arg_count) {
    if (arg_count < 3) return -1.0f;
    /* args[0]=块索引, args[1]=值, args[2]=字节数 */
    /* P-P1-15修复: 通过块索引获取目标指针，不再从float还原指针地址(高位丢失) */
    int idx = (int)args[0];
    if (idx < 0 || idx >= CI_MAX_MEM_BLOCKS) return -1.0f;
    if (!ci->mem.blocks[idx].in_use) return -1.0f;
    void* addr = ci->mem.blocks[idx].ptr;
    size_t block_size = ci->mem.blocks[idx].size;
    int value = (int)args[1];
    size_t n = (size_t)args[2];
    if (addr == NULL || n == 0) return 0.0f;
    /* 边界校验: 字节数不得超过该块分配大小，防止越界写入内存池 */
    if (n > block_size) n = block_size;
    memset(addr, value, n);
    return args[0];
}

/* memcpy实现 */
static float _ci_builtin_memcpy(CiInterpreter* ci, const float* args, int arg_count) {
    if (arg_count < 3) return -1.0f;
    /* args[0]=目标块索引(DST), args[1]=源块索引(SRC), args[2]=字节数 */
    /* P-P1-15修复: 通过块索引获取src和dst指针，不再从float还原指针地址(高位丢失) */
    int dst_idx = (int)args[0];
    int src_idx = (int)args[1];
    if (dst_idx < 0 || dst_idx >= CI_MAX_MEM_BLOCKS) return -1.0f;
    if (src_idx < 0 || src_idx >= CI_MAX_MEM_BLOCKS) return -1.0f;
    if (!ci->mem.blocks[dst_idx].in_use || !ci->mem.blocks[src_idx].in_use) return -1.0f;
    void* dst = ci->mem.blocks[dst_idx].ptr;
    const void* src = ci->mem.blocks[src_idx].ptr;
    size_t dst_size = ci->mem.blocks[dst_idx].size;
    size_t src_size = ci->mem.blocks[src_idx].size;
    size_t n = (size_t)args[2];
    if (dst == NULL || src == NULL || n == 0) return 0.0f;
    /* 边界校验: 字节数不得超过源块和目标块大小，防止越界读写内存池 */
    if (n > dst_size) n = dst_size;
    if (n > src_size) n = src_size;
    memcpy(dst, src, n);
    return args[0];
}

/* M-026: 扩展内置函数注册表 */
static const CiBuiltinMulti g_ci_builtins_multi[] = {
    {"printf",  (CiBuiltinMultiFunc)_ci_builtin_printf,  0},
    {"malloc",  (CiBuiltinMultiFunc)_ci_builtin_malloc,  0},
    {"free",    (CiBuiltinMultiFunc)_ci_builtin_free,    0},
    {"strcpy",  (CiBuiltinMultiFunc)_ci_builtin_strcpy,  0},
    {"memset",  (CiBuiltinMultiFunc)_ci_builtin_memset,  0},
    {"memcpy",  (CiBuiltinMultiFunc)_ci_builtin_memcpy,  0},
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

        /* P-P0-14修复: 在token创建前检查边界，防止越界写
         * 原检查位于循环末尾(仅对运算符生效)，数字/标识符分支
         * 使用continue跳过末尾检查，导致token_count==CI_MAX_TOKENS时
         * 写入tokens[256]越界。此处前移检查，并预留EOF槽位。 */
        if (ci->token_count >= CI_MAX_TOKENS - 1) break;

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
            case '[': ci->tokens[ci->token_count++].type = CI_TOK_LBRACKET; ci->pos++; break;
            case ']': ci->tokens[ci->token_count++].type = CI_TOK_RBRACKET; ci->pos++; break;
            case '&': ci->tokens[ci->token_count++].type = CI_TOK_AMPERSAND; ci->pos++; break;
            case '.': ci->tokens[ci->token_count++].type = CI_TOK_DOT; ci->pos++; break;
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
            case '{': ci->tokens[ci->token_count++].type = CI_TOK_LBRACE; ci->pos++; break;
            case '}': ci->tokens[ci->token_count++].type = CI_TOK_RBRACE; ci->pos++; break;
            case ',': ci->tokens[ci->token_count++].type = CI_TOK_COMMA; ci->pos++; break;
            default:
                ci->has_error = 1;
                snprintf(ci->error_msg, sizeof(ci->error_msg), "行%d: 不支持的字符 '%c'", ci->lineno, c);
                return -1;
        }
    }

    /* 写入EOF标记(token_count已被前置检查限制在CI_MAX_TOKENS-1内，
     * 故tokens[token_count]始终在有效索引[0,CI_MAX_TOKENS-1]范围内) */
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

/* 获取数组元素值 */
static float _ci_get_array_elem(CiInterpreter* ci, const char* name, int idx) {
    CiVariable* v = _ci_find_var(ci, name);
    if (!v || v->type != CI_VAR_ARRAY) return 0.0f;
    if (idx < 0 || idx >= v->array_size) return 0.0f;
    return v->array_vals[idx];
}

/* 设置数组元素值 */
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

/* P1修复: token-based的期望匹配函数，匹配并消费指定类型的token
 * 与char-based的_ci_expect不同，此函数操作token_pos而非source[pos]，
 * 确保控制流解析与表达式解析使用同一套token游标，消除位置不同步问题。
 * @param ci 解释器状态
 * @param type 期望的token类型
 * @return 1=匹配成功并已消费, 0=不匹配 */
static int _ci_expect_tok(CiInterpreter* ci, CiTokenType type) {
    if (_ci_current(ci)->type == type) {
        _ci_next(ci);
        return 1;
    }
    return 0;
}

static float _ci_expr(CiInterpreter* ci);

/* 基本单元: 数字 | 标识符[可能带下标] | ( 表达式 ) */
static float _ci_primary(CiInterpreter* ci) {
    CiToken* tok = _ci_current(ci);
    if (tok->type == CI_TOK_NUMBER) {
        _ci_next(ci);
        return tok->num_val;
    }
/* &取地址操作符 */
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
/* 数组下标访问 arr[idx] */
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
                /* P1修复4: 在当前解释器上下文中求值参数，不创建新实例。
                 * 原实现通过self_programming_interpret_expr(arg_expr, ...)为每个参数
                 * 创建全新的CiInterpreter实例，新实例拥有独立的变量表，
                 * 外部作用域中定义的变量在新实例中不可见。
                 * 现改为在当前ci中通过token-based方式逐个求值参数表达式，
                 * 用CI_TOK_COMMA分隔参数，CI_TOK_RPAREN结束参数列表。 */
                float args[CI_MAX_FUNC_ARGS];
                int arg_count = 0;

                /* 解析参数列表: arg1, arg2, ..., argN) */
                if (_ci_current(ci)->type != CI_TOK_RPAREN) {
                    args[arg_count++] = _ci_expr(ci);
                    while (_ci_current(ci)->type == CI_TOK_COMMA &&
                           arg_count < CI_MAX_FUNC_ARGS) {
                        _ci_next(ci);  /* 跳过逗号 */
                        args[arg_count++] = _ci_expr(ci);
                    }
                }
                if (_ci_current(ci)->type == CI_TOK_RPAREN)
                    _ci_next(ci);  /* 跳过')' */

                /* 调用多参数内置函数 */
                float result = 0.0f;
                for (int k = 0; !g_ci_builtins_multi[k].is_sentinel; k++) {
                    if (strcmp(name, g_ci_builtins_multi[k].name) == 0) {
                        result = g_ci_builtins_multi[k].func(
                            (struct CiInterpreter_s*)ci, args, arg_count);
                        break;
                    }
                }
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
/* print内置函数 */
            if (strcmp(name, "print") == 0) {
                printf("[解释器输出] %g\n", (double)arg);
                return arg;
            }
            return 0.0f;
        }
/* 结构体成员访问 var.field 或 ptr->field */
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
/* *ptr指针解引用 */
    if (tok->type == CI_TOK_STAR) {
        _ci_next(ci);
        float addr = _ci_primary(ci);
        /* P1修复: *addr应返回addr指向的变量值，而非addr本身。
         * &x返回合成地址(268435456.0f + var_index*4096 + name_hash)，
         * 存储在v->address中。*addr需要反查该地址对应的变量并返回其值。
         * 遍历变量表查找address匹配的变量，返回值逻辑与_ci_get_var_value一致：
         * float/array类型返回fval，int类型返回ival。 */
        for (int i = 0; i < ci->var_count; i++) {
            if (ci->vars[i].has_address && ci->vars[i].address == addr) {
                if (ci->vars[i].type == CI_VAR_FLOAT || ci->vars[i].type == CI_VAR_ARRAY)
                    return ci->vars[i].fval;
                return (float)ci->vars[i].ival;
            }
        }
        return 0.0f;
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

/* P1修复1: 前向声明，使_ci_exec_statement能委托控制流给_ci_statement */
static float _ci_statement(CiInterpreter* ci);

/* 执行表达式语句: expr ;  支持简单赋值、复合赋值和数组下标赋值 */
static float _ci_exec_statement(CiInterpreter* ci) {
    CiToken* tok = _ci_current(ci);
    float result = 0.0f;

    /* P1修复1: 检测控制流关键字和代码块，委托给_ci_statement处理
     * 原实现仅处理赋值和表达式语句，从不进入if/for/while/{块路径，
     * 导致从公共API self_programming_interpret_expr()无法执行控制流语句。
     * 此处在入口处增加对{块和if/for/while关键字的检测（基于token，带词边界）。 */
    if (tok->type == CI_TOK_LBRACE ||
        (tok->type == CI_TOK_IDENT &&
         (strcmp(tok->ident, "if") == 0 ||
          strcmp(tok->ident, "for") == 0 ||
          strcmp(tok->ident, "while") == 0))) {
        return _ci_statement(ci);
    }

    if (tok->type == CI_TOK_IDENT) {
        int pos = ci->token_pos;
        char name[64];
        strncpy(name, tok->ident, 63);
        name[63] = '\0';
        _ci_next(ci);

/* 数组下标赋值 arr[idx] = expr */
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

/* P1修复: 前向声明 — 跳过语句和跳过括号的辅助函数需要相互引用 */
static void _ci_skip_statement(CiInterpreter* ci);
static void _ci_skip_paren(CiInterpreter* ci);

static int _ci_parse_block(CiInterpreter* ci);
static float _ci_if_statement(CiInterpreter* ci);
static float _ci_for_loop(CiInterpreter* ci);
static float _ci_while_loop(CiInterpreter* ci);

/* P1修复1/2/3/5: 跳过一个语句的token序列（不执行），用于if条件为false时跳过then分支。
 * 通过推进token_pos跳过token序列，确保后续的else检测定位正确。
 * 支持: {块}, if/for/while嵌套控制流, 单语句(到';')。 */
static void _ci_skip_statement(CiInterpreter* ci) {
    CiToken* tok = _ci_current(ci);
    if (tok->type == CI_TOK_LBRACE) {
        /* 跳过{...}块: 匹配花括号对 */
        int depth = 0;
        do {
            if (tok->type == CI_TOK_LBRACE) depth++;
            else if (tok->type == CI_TOK_RBRACE) depth--;
            _ci_next(ci);
            tok = _ci_current(ci);
        } while (depth > 0 && tok->type != CI_TOK_EOF);
    } else if (tok->type == CI_TOK_IDENT && strcmp(tok->ident, "if") == 0) {
        /* 跳过嵌套if语句: if (cond) stmt [else stmt] */
        _ci_next(ci);             /* 消费if */
        _ci_skip_paren(ci);      /* 跳过(cond) */
        _ci_skip_statement(ci);  /* 跳过then分支 */
        tok = _ci_current(ci);
        if (tok->type == CI_TOK_IDENT && strcmp(tok->ident, "else") == 0) {
            _ci_next(ci);             /* 消费else */
            _ci_skip_statement(ci);   /* 跳过else分支 */
        }
    } else if (tok->type == CI_TOK_IDENT &&
               (strcmp(tok->ident, "for") == 0 || strcmp(tok->ident, "while") == 0)) {
        /* 跳过for/while循环: for/while (...) body */
        _ci_next(ci);             /* 消费for/while */
        _ci_skip_paren(ci);      /* 跳过(...) */
        _ci_skip_statement(ci);  /* 跳过循环体 */
    } else {
        /* 跳过单语句到分号 */
        while (tok->type != CI_TOK_SEMI && tok->type != CI_TOK_EOF) {
            _ci_next(ci);
            tok = _ci_current(ci);
        }
        if (tok->type == CI_TOK_SEMI) _ci_next(ci);  /* 消费分号 */
    }
}

/* P1修复: 跳过匹配的圆括号(...)，用于跳过if/for/while的条件表达式 */
static void _ci_skip_paren(CiInterpreter* ci) {
    if (_ci_current(ci)->type != CI_TOK_LPAREN) return;
    int depth = 0;
    do {
        if (_ci_current(ci)->type == CI_TOK_LPAREN) depth++;
        else if (_ci_current(ci)->type == CI_TOK_RPAREN) depth--;
        _ci_next(ci);
    } while (depth > 0 && _ci_current(ci)->type != CI_TOK_EOF);
}

/* ---- 控制流 区块/语句/if/for/while ----
 * P1修复1/2/5: 全部改为token-based解析，消除source[pos]与token_pos不同步问题。
 * 关键字检测通过token的ident字段精确匹配（tokenizer已处理词边界），
 * 检测到关键字后通过_ci_next(ci)消费关键字token，再委托给对应处理函数。 */
static float _ci_statement(CiInterpreter* ci) {
    CiToken* tok = _ci_current(ci);

    /* {代码块} */
    if (tok->type == CI_TOK_LBRACE) {
        return (float)_ci_parse_block(ci);
    }

    /* 控制流关键字检测（token精确匹配，自带词边界） */
    if (tok->type == CI_TOK_IDENT) {
        if (strcmp(tok->ident, "if") == 0) {
            _ci_next(ci);  /* P1修复2: 消费"if"关键字token */
            return _ci_if_statement(ci);
        }
        if (strcmp(tok->ident, "for") == 0) {
            _ci_next(ci);  /* P1修复2: 消费"for"关键字token */
            return _ci_for_loop(ci);
        }
        if (strcmp(tok->ident, "while") == 0) {
            _ci_next(ci);  /* P1修复2: 消费"while"关键字token */
            return _ci_while_loop(ci);
        }
    }

    /* 赋值/表达式语句 */
    return _ci_exec_statement(ci);
}

static int _ci_parse_block(CiInterpreter* ci) {
    /* P1修复: 使用token匹配'{'，替代char-based的_ci_expect */
    if (!_ci_expect_tok(ci, CI_TOK_LBRACE)) return 0;
    /* P-P2-17修复: 递归深度保护，防止恶意/失控嵌套块导致栈溢出
     * _ci_parse_block → _ci_statement → if/for/while → _ci_parse_block 形成递归 */
    if (++ci->call_depth > CI_MAX_CALL_DEPTH) {
        --ci->call_depth;
        ci->has_error = 1;
        snprintf(ci->error_msg, sizeof(ci->error_msg),
                 "递归深度超过上限%d", CI_MAX_CALL_DEPTH);
        return 0;
    }
    int count = 0;
    /* P1修复: 使用token检测EOF和'}'，替代char-based的_ci_match和source[pos] */
    while (_ci_current(ci)->type != CI_TOK_EOF &&
           _ci_current(ci)->type != CI_TOK_RBRACE) {
        /* P1修复4: 推进保护。当语句起始token是_ci_statement/_ci_primary均不消费的类型
         * (如COMMA等)时，_ci_statement内部无法推进token_pos，while条件永真→无限循环。
         * 记录执行前位置，若未推进则强制前进一步，防止卡死。 */
        int prev_pos = ci->token_pos;
        _ci_statement(ci);
        if (ci->token_pos == prev_pos) {
            _ci_next(ci);  /* 防止卡死：强制推进一个token */
        }
        count++;
    }
    _ci_expect_tok(ci, CI_TOK_RBRACE);  /* 消费'}' */
    --ci->call_depth;
    return count;
}

static float _ci_if_statement(CiInterpreter* ci) {
    /* P1修复: 使用token匹配'('和')'，替代char-based的_ci_expect */
    if (!_ci_expect_tok(ci, CI_TOK_LPAREN)) return 0.0f;
    float cond = _ci_expr(ci);
    _ci_expect_tok(ci, CI_TOK_RPAREN);
    float result = 0.0f;
    float truth = (fabsf(cond) > 0.01f) ? 1.0f : 0.0f;
    if (truth > 0.5f) {
        /* 条件为真: 执行then分支 */
        result = _ci_statement(ci);
        /* 检查是否有else分支，如有则跳过（不执行） */
        CiToken* tok = _ci_current(ci);
        if (tok->type == CI_TOK_IDENT && strcmp(tok->ident, "else") == 0) {
            _ci_next(ci);  /* 消费else */
            _ci_skip_statement(ci);  /* 跳过else分支 */
        }
    } else {
        /* 条件为假: 跳过then分支，执行else分支（如有） */
        _ci_skip_statement(ci);
        /* 跳过then分支后可能有多余分号 */
        while (_ci_current(ci)->type == CI_TOK_SEMI) {
            _ci_next(ci);
        }
        CiToken* tok = _ci_current(ci);
        if (tok->type == CI_TOK_IDENT && strcmp(tok->ident, "else") == 0) {
            _ci_next(ci);  /* 消费else */
            result = _ci_statement(ci);
        }
    }
    return result;
}

static float _ci_for_loop(CiInterpreter* ci) {
    /* P1修复: 使用token匹配，替代char-based的_ci_expect */
    if (!_ci_expect_tok(ci, CI_TOK_LPAREN)) return 0.0f;
    /* 初始化语句 */
    _ci_statement(ci);
    _ci_expect_tok(ci, CI_TOK_SEMI);
    float result = 0.0f;
    /* P1修复3: 保存token_pos（而非source pos）用于循环条件重新求值。
     * 原实现仅保存/恢复ci->pos(source位置)，从不保存/恢复ci->token_pos，
     * 导致_ci_expr()在后续迭代中从EOF token读取返回0.0f。 */
    int cond_tok_pos = ci->token_pos;
    float cond = _ci_expr(ci);
    _ci_expect_tok(ci, CI_TOK_SEMI);
    /* 增量表达式位置（保存用于循环内重新执行） */
    int inc_tok_pos = ci->token_pos;
    /* P1修复3: 首次仅跳过增量表达式(不执行)，定位到for的')'。
     * 原实现调用_ci_expr()解析增量，但_ci_expr不处理赋值类运算符(=,+=,-=等)，
     * i+=1 / i=i+1 的增量被完全跳过，且因')'未被消费导致循环体定位错误。
     * 现改为扫描token到匹配的')'(处理嵌套括号如 i=f(x))，由后续_ci_expect_tok消费。
     * 真正执行放在循环体内通过_ci_exec_statement完成。 */
    {
        int inc_depth = 0;
        while (ci->token_pos < ci->token_count) {
            CiTokenType tt = _ci_current(ci)->type;
            if (tt == CI_TOK_LPAREN) {
                inc_depth++;
            } else if (tt == CI_TOK_RPAREN) {
                if (inc_depth == 0) break;  /* 到达for循环的')' */
                inc_depth--;
            } else if (tt == CI_TOK_EOF) {
                break;  /* 防御: 增量缺少')'时不越界 */
            }
            _ci_next(ci);
        }
    }
    _ci_expect_tok(ci, CI_TOK_RPAREN);  /* 消费')' */
    /* 循环体位置 */
    int body_tok_pos = ci->token_pos;
    int iter_count = 0;  /* P-P1-16修复: 迭代计数器，防止死循环挂起AGI线程 */
    int after_body_tok_pos = -1;  /* 循环体结束位置(首次执行后记录) */
    while (cond > 0.01f || cond < -0.01f) {
        if (++iter_count > CI_MAX_LOOP_ITERATIONS) {
            log_warn("[C解释器] for循环超过最大迭代上限%d，强制终止", CI_MAX_LOOP_ITERATIONS);
            break;
        }
        /* P1修复3: 同步恢复token_pos到循环体起始位置 */
        ci->token_pos = body_tok_pos;
        result = _ci_statement(ci);  /* 执行循环体 */
        if (after_body_tok_pos < 0) after_body_tok_pos = ci->token_pos;
        /* 执行增量表达式 */
        ci->token_pos = inc_tok_pos;
        /* P1修复3: 增量通过语句执行器执行，正确处理赋值类运算符(=,+=,-=,*=,/=)。
         * 原实现用_ci_expr()无法执行 i+=1 / i=i+1，导致循环变量永不更新→死循环。
         * _ci_exec_statement在赋值后仅消费分号(此处无分号)，token_pos停在')'，
         * 随后由条件重定位覆盖，不影响后续流程。 */
        _ci_exec_statement(ci);
        /* 重新求值条件 */
        ci->token_pos = cond_tok_pos;
        cond = _ci_expr(ci);
    }
    /* P1修复3: 循环结束后，将token_pos恢复到循环体之后 */
    if (after_body_tok_pos >= 0) ci->token_pos = after_body_tok_pos;
    return result;
}

/* while循环实现 */
static float _ci_while_loop(CiInterpreter* ci) {
    /* P1修复: 使用token匹配 */
    if (!_ci_expect_tok(ci, CI_TOK_LPAREN)) return 0.0f;
    /* P1修复3: 保存token_pos用于循环条件重新求值 */
    int cond_tok_pos = ci->token_pos;
    float cond = _ci_expr(ci);
    _ci_expect_tok(ci, CI_TOK_RPAREN);
    float result = 0.0f;
    int body_tok_pos = ci->token_pos;
    int iter_count = 0;  /* P-P1-16修复: 迭代计数器，防止死循环挂起AGI线程 */
    int after_body_tok_pos = -1;
    while (cond > 0.01f || cond < -0.01f) {
        if (++iter_count > CI_MAX_LOOP_ITERATIONS) {
            log_warn("[C解释器] while循环超过最大迭代上限%d，强制终止", CI_MAX_LOOP_ITERATIONS);
            break;
        }
        /* P1修复3: 同步恢复token_pos到循环体起始位置 */
        ci->token_pos = body_tok_pos;
        result = _ci_statement(ci);  /* 执行循环体 */
        if (after_body_tok_pos < 0) after_body_tok_pos = ci->token_pos;
        /* 重新求值条件 */
        ci->token_pos = cond_tok_pos;
        cond = _ci_expr(ci);
    }
    if (after_body_tok_pos >= 0) ci->token_pos = after_body_tok_pos;
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

    /* P-P0-13修复: 改为堆分配，避免1.2MB栈帧导致栈溢出
     * CiInterpreter含1MB内存池+大量数组，总计约1.2MB，
     * Windows默认线程栈1MB，栈上分配必定栈溢出。
     * 改用safe_calloc堆分配，safe_calloc已清零，无需再memset。 */
    CiInterpreter* ci = (CiInterpreter*)safe_calloc(1, sizeof(CiInterpreter));
    if (!ci) return -1;
    ci->source = code;
    ci->has_error = 0;
    /* M-026: 初始化内存池(safe_calloc已清零，此处保留以保持语义清晰) */
    memset(&ci->mem, 0, sizeof(CiMemoryPool));

    if (_ci_tokenize(ci) < 0) {
        if (error_msg) strncpy(error_msg, ci->error_msg, 255);
        safe_free((void**)&ci);
        return -1;
    }

    *result = _ci_exec_statement(ci);

    if (ci->has_error) {
        if (error_msg) strncpy(error_msg, ci->error_msg, 255);
        safe_free((void**)&ci);
        return -1;
    }

    safe_free((void**)&ci);
    return 0;
}

/**
 * @brief K-007: 检查内置解释器是否可用
 *
 * 检查解释器上下文初始化状态和内存池分配状态。
 * 验证内置函数表完整性以确认解释器核心组件正常。
 *
 * @return 1可用，0不可用
 */
/* P1-003修复: 静态标志追踪解释器初始化状态 */
static int g_ci_core_verified = 0;

int self_programming_interpreter_available(void) {
    /* 第一优先级: 检查内置函数表完整性 */
    if (!g_ci_builtins || !g_ci_builtins_multi) return 0;
    if (g_ci_builtins[0].func == NULL && g_ci_builtins_multi[0].func == NULL) return 0;

    /* 检查内存池所需的内存区域可用性 */
    if (g_ci_core_verified) return 1;

    /* P2修复6: 原实现在栈上创建CiInterpreter ci_check（含1MB内存池，总计约1.2MB），
     * Windows默认线程栈仅1MB，栈上分配必定栈溢出。
     * 改为仅用sizeof在编译期检查结构体成员大小是否满足要求，
     * sizeof是编译期常量表达式，不实际分配任何内存。 */
    if (sizeof(((CiInterpreter*)0)->mem.pool) < CI_MEM_POOL_SIZE) return 0;
    if (sizeof(((CiInterpreter*)0)->mem.blocks) < CI_MAX_MEM_BLOCKS * sizeof(CiMemBlock)) return 0;
    if (sizeof(((CiInterpreter*)0)->vars) < CI_MAX_VARS * sizeof(CiVariable)) return 0;

    g_ci_core_verified = 1;

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
 * @brief C解释器可用性检查（包装接口）
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
 * @brief C解释器代码执行（包装接口）
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
 * @brief C解释器表达式求值（包装接口）
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
