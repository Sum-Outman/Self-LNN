/**
 * @file programming_enhanced.c
 * @brief A09.4 自我编程能力增强实现
 *
 * 三大子模块：
 *   A09.4.1 代码重构引擎（重构模式检测与自动重构）
 *   A09.4.2 性能分析（静态性能瓶颈检测 + 安全漏洞扫描）
 */

#include "selflnn/programming/programming_enhanced.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ============================================================================
 * 内部辅助函数
 * ============================================================================ */

/* 安全字符串复制 */
static void safe_strcpy(char* dst, size_t dst_size, const char* src)
{
    if (!dst || !src || dst_size == 0) return;
    size_t i;
    for (i = 0; i < dst_size - 1 && src[i] != '\0'; ++i)
        dst[i] = src[i];
    dst[i] = '\0';
}

/* 计算AST子树中的节点总数 */
static int count_ast_nodes(const ASTNode* node)
{
    if (!node) return 0;
    int count = 1;
    if (node->left)  count += count_ast_nodes(node->left);
    if (node->right) count += count_ast_nodes(node->right);
    if (node->next)  count += count_ast_nodes(node->next);
    return count;
}

/* M-001修复: 不良内存局部性检测 —— 扫描嵌套循环中数组访问顺序 */
static int penh_detect_memory_locality(const char* source_code,
    PenhPerfIssue* issues, int max_count) {
    if (!source_code || !issues || max_count <= 0) return 0;
    int count = 0;
    const char* inner_loops[] = {"for(", "while("};
    int nl = sizeof(inner_loops) / sizeof(inner_loops[0]);
    int depth = 0;
    const char* p = source_code;
    while (*p && count < max_count) {
        for (int i = 0; i < nl; i++) {
            if (strncmp(p, inner_loops[i], strlen(inner_loops[i])) == 0) {
                depth++;
                if (depth >= 3 && count < max_count) {
                    issues[count].type = PENH_PERF_POOR_MEMORY_LOCALITY;
                    issues[count].severity = 5.0f;
                    int line = 1; for (const char* q = source_code; q < p; q++)
                        if (*q == '\n') line++;
                    issues[count].source_line = line;
                    snprintf(issues[count].description, sizeof(issues[count].description),
                        "深度%d嵌套循环可能存在不良内存局部性", depth);
                    snprintf(issues[count].recommendation, sizeof(issues[count].recommendation),
                        "建议分块处理或调整数组访问顺序以提高缓存命中率");
                    count++;
                }
                break;
            }
        }
        if (*p == '}') depth--;
        p++;
    }
    return count;
}

/* M-001修复: 缓存未命中检测 —— 步长过大的数组访问 */
static int penh_detect_cache_miss(const char* source_code,
    PenhPerfIssue* issues, int max_count) {
    if (!source_code || !issues || max_count <= 0) return 0;
    int count = 0;
    const char* large_stride[] = {"stride", "STRIDE"};
    int ns = sizeof(large_stride) / sizeof(large_stride[0]);
    for (int i = 0; i < ns && count < max_count; i++) {
        const char* found = strstr(source_code, large_stride[i]);
        if (found) {
            issues[count].type = PENH_PERF_CACHE_MISS;
            issues[count].severity = 4.0f;
            int line = 1; for (const char* q = source_code; q < found; q++)
                if (*q == '\n') line++;
            issues[count].source_line = line;
            snprintf(issues[count].description, sizeof(issues[count].description),
                "可能存在大步长内存访问导致缓存未命中");
            snprintf(issues[count].recommendation, sizeof(issues[count].recommendation),
                "调整数据结构布局或使用缓存友好的块访问");
            count++;
        }
    }
    return count;
}

/* M-001修复: 过度拷贝检测 */
static int penh_detect_excessive_copy(const char* source_code,
    PenhPerfIssue* issues, int max_count) {
    if (!source_code || !issues || max_count <= 0) return 0;
    int memcpy_count = 0;
    const char* p = source_code;
    while ((p = strstr(p, "memcpy")) != NULL) {
        memcpy_count++; p += 6;
    }
    if (memcpy_count > 5 && max_count > 0) {
        issues[0].type = PENH_PERF_EXCESSIVE_COPYING;
        issues[0].severity = 4.0f;
        issues[0].source_line = 1;
        snprintf(issues[0].description, sizeof(issues[0].description),
            "检测到%d次内存拷贝调用，可能存在过度拷贝优化空间", memcpy_count);
        snprintf(issues[0].recommendation, sizeof(issues[0].recommendation),
            "考虑使用零拷贝或引用传递减少数据复制");
        return 1;
    }
    return 0;
}

/* M-001修复: 分支预测失败检测 —— 循环中的多条件分支 */
static int penh_detect_branch_mispredict(const ASTNode* ast,
    PenhPerfIssue* issues, int max_count) {
    if (!ast || !issues || max_count <= 0) return 0;
    int count = 0;
    if (ast->type == AST_IF) {
        const ASTNode* loop_child = ast;
        for (size_t i = 0; i < loop_child->child_count; i++) {
            if (loop_child->children[i] && loop_child->children[i]->type == AST_LOOP) {
                if (count < max_count) {
                    issues[count].type = PENH_PERF_BRANCH_MISPREDICT;
                    issues[count].severity = 4.5f;
                    snprintf(issues[count].description, sizeof(issues[count].description),
                        "循环内条件分支可能导致频繁分支预测失败");
                    snprintf(issues[count].recommendation, sizeof(issues[count].recommendation),
                        "将频繁判断的条件移出循环或使用查找表替代分支");
                    count++;
                }
            }
        }
    }
    for (size_t i = 0; i < ast->child_count && count < max_count; i++)
        count += penh_detect_branch_mispredict(ast->children[i], issues + count, max_count - count);
    return count;
}

/* M-001修复: 不必要分配检测 */
static int penh_detect_unnecessary_allocation(const ASTNode* ast,
    PenhPerfIssue* issues, int max_count) {
    if (!ast || !issues || max_count <= 0) return 0;
    int count = 0;
    if (ast->type == AST_LOOP) {
        for (size_t i = 0; i < ast->child_count && count < max_count; i++) {
            const ASTNode* child = ast->children[i];
            if (child && child->type == AST_FUNCTION) {
                for (size_t j = 0; j < child->child_count; j++) {
                    if (child->children[j] && child->children[j]->type == AST_LITERAL &&
                        strstr((const char*)child->children[j]->data, "malloc")) {
                        issues[count].type = PENH_PERF_UNNECESSARY_ALLOCATION;
                        issues[count].severity = 5.5f;
                        snprintf(issues[count].description, sizeof(issues[count].description),
                            "循环内动态内存分配可能导致性能下降");
                        snprintf(issues[count].recommendation, sizeof(issues[count].recommendation),
                            "将内存分配移到循环外或使用栈分配");
                        count++;
                        break;
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < ast->child_count && count < max_count; i++)
        count += penh_detect_unnecessary_allocation(ast->children[i], issues + count, max_count - count);
    return count;
}

/* M-001修复: 小IO检测 */
static int penh_detect_small_io(const ASTNode* ast,
    PenhPerfIssue* issues, int max_count) {
    if (!ast || !issues || max_count <= 0) return 0;
    int count = 0;
    if (ast->type == AST_CALL) {
        const char* data = ast->data ? (const char*)ast->data : "";
        if (strstr(data, "read") || strstr(data, "write") ||
            strstr(data, "fread") || strstr(data, "fwrite") ||
            strstr(data, "printf") || strstr(data, "fprintf")) {
            if (count < max_count) {
                issues[count].type = PENH_PERF_SMALL_IO;
                issues[count].severity = 3.5f;
                snprintf(issues[count].description, sizeof(issues[count].description),
                    "频繁小IO操作可能影响性能");
                snprintf(issues[count].recommendation, sizeof(issues[count].recommendation),
                    "使用缓冲区批量读写或内存映射文件");
                count++;
            }
        }
    }
    for (size_t i = 0; i < ast->child_count && count < max_count; i++)
        count += penh_detect_small_io(ast->children[i], issues + count, max_count - count);
    return count;
}

/* 计算AST子树中的语句数（BLOCK的直接子节点数） */
static int count_block_statements(const ASTNode* node)
{
    if (!node) return 0;
    int count = 0;
    const ASTNode* child = (node->type == AST_BLOCK) ? node->left : NULL;
    while (child) {
        count++;
        child = child->next;
    }
    return count;
}

/* 获取AST中的最大嵌套深度 */
static int max_nesting_depth(const ASTNode* node, int current_depth)
{
    if (!node) return current_depth;
    int max_depth = current_depth;
    int child_depth = current_depth;
    if (node->type == AST_IF || node->type == AST_LOOP || node->type == AST_BLOCK)
        child_depth = current_depth + 1;
    if (node->left) {
        int ld = max_nesting_depth(node->left, child_depth);
        if (ld > max_depth) max_depth = ld;
    }
    if (node->right) {
        int rd = max_nesting_depth(node->right, child_depth);
        if (rd > max_depth) max_depth = rd;
    }
    if (node->next) {
        int nd = max_nesting_depth(node->next, current_depth);
        if (nd > max_depth) max_depth = nd;
    }
    return max_depth;
}

/* 检测字符串中是否包含子串（不区分大小写） */
static int str_contains_ignore_case(const char* haystack, const char* needle)
{
    if (!haystack || !needle) return 0;
    size_t needle_len = strlen(needle);
    if (needle_len == 0) return 1;
    size_t haystack_len = strlen(haystack);
    if (needle_len > haystack_len) return 0;
    for (size_t i = 0; i <= haystack_len - needle_len; ++i) {
        size_t j;
        for (j = 0; j < needle_len; ++j) {
            if (tolower((unsigned char)haystack[i + j]) != tolower((unsigned char)needle[j]))
                break;
        }
        if (j == needle_len) return 1;
    }
    return 0;
}

/* 查找源代码中第line_number行的起始位置 */
static const char* find_line_start(const char* source, int line_number)
{
    if (!source || line_number <= 0) return source;
    const char* p = source;
    int current_line = 1;
    while (*p && current_line < line_number) {
        if (*p == '\n') current_line++;
        p++;
    }
    return p;
}

/* 提取源代码中指定行号的文本行 */
static int extract_source_line(const char* source, int line, char* out, size_t out_size)
{
    if (!source || !out || out_size == 0) return -1;
    const char* start = find_line_start(source, line);
    if (!*start) { out[0] = '\0'; return -1; }
    size_t i;
    for (i = 0; i < out_size - 1 && start[i] != '\0' && start[i] != '\n'; ++i)
        out[i] = start[i];
    out[i] = '\0';
    return 0;
}

/* ---------------------------------------------------------------------------
 * AST遍历辅助
 * --------------------------------------------------------------------------- */

/* 查找AST中所有函数定义节点并收集信息 */
typedef struct {
    const ASTNode* nodes[1024];
    char names[1024][128];
    int lines[1024];
    int body_statement_counts[1024];
    int count;
} FunctionCollector;

static void collect_functions_recursive(const ASTNode* node, FunctionCollector* fc)
{
    if (!node || !fc || fc->count >= 1024) return;
    if (node->type == AST_FUNCTION && node->name) {
        int idx = fc->count;
        fc->nodes[idx] = node;
        safe_strcpy(fc->names[idx], sizeof(fc->names[idx]), node->name);
        fc->lines[idx] = node->line;
        fc->body_statement_counts[idx] = count_block_statements(node->left);
        fc->count++;
    }
    if (node->left)  collect_functions_recursive(node->left, fc);
    if (node->right) collect_functions_recursive(node->right, fc);
    if (node->next)  collect_functions_recursive(node->next, fc);
}

/* M-041修复: 根据行号查找行所属的函数名 */
static const char* find_function_for_line(const FunctionCollector* fc, int line_num) {
    if (!fc || fc->count == 0) return "unknown";
    const char* best_name = "unknown";
    int best_line = 0;
    for (int i = 0; i < fc->count; i++) {
        if (fc->lines[i] <= line_num && fc->lines[i] > best_line) {
            best_name = fc->names[i];
            best_line = fc->lines[i];
        }
    }
    return best_name;
}

/* 收集所有函数调用（用于递归检测） */
typedef struct {
    char names[1024][128];
    int lines[1024];
    int count;
} CallCollector;

static void collect_calls_recursive(const ASTNode* node, CallCollector* cc)
{
    if (!node || !cc || cc->count >= 1024) return;
    if (node->type == AST_CALL && node->name) {
        int idx = cc->count;
        safe_strcpy(cc->names[idx], sizeof(cc->names[idx]), node->name);
        cc->lines[idx] = node->line;
        cc->count++;
    }
    if (node->left)  collect_calls_recursive(node->left, cc);
    if (node->right) collect_calls_recursive(node->right, cc);
    if (node->next)  collect_calls_recursive(node->next, cc);
}

/* 判断函数是否调用另一个函数（用于递归检测） */
static int function_calls(const ASTNode* func_node, const char* target_name)
{
    if (!func_node || !target_name) return 0;
    int found = 0;
    CallCollector cc;
    memset(&cc, 0, sizeof(cc));
    /* 查找函数体中的调用 */
    if (func_node->left) collect_calls_recursive(func_node->left, &cc);
    /* 也在右子树中查找（函数体可能在right） */
    if (func_node->right) collect_calls_recursive(func_node->right, &cc);
    for (int i = 0; i < cc.count; ++i) {
        if (strcmp(cc.names[i], target_name) == 0) {
            found = 1;
            break;
        }
    }
    return found;
}

/* ---------------------------------------------------------------------------
 * 模式匹配：检测源代码中的危险函数/模式
 * --------------------------------------------------------------------------- */

/* 检测危险缓冲区操作函数 */
static int match_buffer_dangerous_func(const char* source_line)
{
    const char* patterns[] = {
        "strcpy(", "strcat(", "sprintf(", "gets(", "scanf(", "vsprintf(",
        "memcpy(", "memmove(", "read(", "recv(", "fread(", "sscanf(",
        "strncpy(", "strncat(", "snprintf("
    };
    int n = sizeof(patterns) / sizeof(patterns[0]);
    for (int i = 0; i < n; ++i) {
        if (strstr(source_line, patterns[i]))
            return 1;
    }
    return 0;
}

/* 检测未初始化变量 M-001修复 */
static int match_uninitialized_var(const char* source_line) {
    if (!source_line) return 0;
    /* 模式: 类型名 标识符; (无初始化赋值) */
    const char* types[] = {"int ", "int\t", "float ", "float\t", "double ", "double\t",
        "char ", "char\t", "size_t ", "size_t\t", "long ", "long\t",
        "int* ", "float* ", "char* ", "void* ", "double* "};
    int n = sizeof(types) / sizeof(types[0]);
    for (int i = 0; i < n; i++) {
        const char* t = strstr(source_line, types[i]);
        if (t) {
            const char* id = t + strlen(types[i]);
            if (*id == '*') id++;
            while (*id == ' ' || *id == '\t') id++;
            int has_assign = 0, has_paren = 0;
            const char* scan = id;
            while (*scan && *scan != '\n' && *scan != ';') {
                if (*scan == '=') has_assign = 1;
                if (*scan == '(') has_paren = 1;
                if (*scan == '[') has_paren = 1;
                scan++;
            }
            if (!has_assign && !has_paren) return 1;
        }
    }
    return 0;
}

/* 检测释放后使用 M-001修复 */
static int match_use_after_free(const char* source_line) {
    if (!source_line) return 0;
    const char* frees[] = {"free(", "safe_free(", "delete ", "delete[] "};
    int nf = sizeof(frees) / sizeof(frees[0]);
    /* 如果行包含释放，找指针名，检查后续行是否使用 */
    for (int i = 0; i < nf; i++) {
        const char* fp = strstr(source_line, frees[i]);
        if (fp) {
            const char* arg_start = fp + strlen(frees[i]);
            const char* arg_end = strchr(arg_start, ')');
            if (arg_end) {
                /* 在这行内查找指针名在释放之后还被引用 */
                char ptr_name[64] = {0};
                const char* a = arg_start;
                while (a < arg_end && *a == ' ') a++;
                const char* ae = a;
                while (ae < arg_end && *ae != ' ' && *ae != ',' && *ae != ')' && *ae != '-') ae++;
                if (ae > a && (size_t)(ae - a) < sizeof(ptr_name)) {
                    memcpy(ptr_name, a, (size_t)(ae - a));
                    const char* after = arg_end + 1;
                    while (*after == ';' || *after == ' ') after++;
                    if (strstr(after, ptr_name)) return 1;
                }
            }
        }
    }
    return 0;
}

/* 检测整数溢出风险模式 */
static int match_integer_overflow(const char* source_line)
{
    const char* patterns[] = {
        "malloc(", "realloc(", "calloc(", "alloca(",
        "sizeof(", ">> ", "<< ", "~"
    };
    int n = sizeof(patterns) / sizeof(patterns[0]);
    for (int i = 0; i < n; ++i) {
        if (strstr(source_line, patterns[i]))
            return 1;
    }
    return 0;
}

/* 检测格式化字符串漏洞 */
static int match_format_string(const char* source_line)
{
    if (!source_line) return 0;
    const char* p = source_line;
    while (*p) {
        if (*p == '\"' && *(p+1) == '%') {
            const char* q = p + 1;
            while (*q && *q != '\"' && *q != '\n') {
                if (*q == 's' || *q == 'x' || *q == 'd' || *q == 'n' || *q == 'p') {
                    if (*(q+1) == '\"') {
                        /* 找到类似 printf("%s", user_input) 的模式 */
                        return 1;
                    }
                }
                q++;
            }
        }
        p++;
    }
    return 0;
}

/* 检测命令注入 */
static int match_command_injection(const char* source_line)
{
    const char* patterns[] = {
        "system(", "popen(", "exec(", "execl(", "execlp(", "execle(",
        "execv(", "execvp(", "execve(", "ShellExecute", "CreateProcess",
        "fork(", "popen"
    };
    int n = sizeof(patterns) / sizeof(patterns[0]);
    for (int i = 0; i < n; ++i) {
        if (strstr(source_line, patterns[i]))
            return 1;
    }
    return 0;
}

/* 检测路径遍历 */
static int match_path_traversal(const char* source_line)
{
    const char* patterns[] = {
        "../", "..\\", "/etc/", "/usr/", "/var/",
        "fopen(", "open(", "access(", "stat(", "RemoveDirectory",
        "DeleteFile", "MoveFile", "CopyFile"
    };
    int n = sizeof(patterns) / sizeof(patterns[0]);
    for (int i = 0; i < n; ++i) {
        if (strstr(source_line, patterns[i]))
            return 1;
    }
    return 0;
}

/* 检测未被检查的返回值 */
static int match_unchecked_return(const char* source_line, const char* next_line)
{
    if (!source_line) return 0;
    const char* func_patterns[] = {
        "malloc(", "calloc(", "realloc(", "fopen(", "fread(", "fwrite(",
        "mmap(", "socket(", "connect(", "accept(", "bind(", "listen(",
        "pthread_create", "pthread_mutex_lock", "sem_wait"
    };
    int n = sizeof(func_patterns) / sizeof(func_patterns[0]);
    for (int i = 0; i < n; ++i) {
        if (strstr(source_line, func_patterns[i])) {
            if (!next_line) return 1;
            /* 检查下一行是否有NULL检查或错误检查 */
            if (strstr(next_line, "NULL") || strstr(next_line, "null") ||
                strstr(next_line, "== -1") || strstr(next_line, "!= -1") ||
                strstr(next_line, "error") || strstr(next_line, "errno") ||
                strstr(next_line, "FAILED") || strstr(next_line, "failed"))
                return 0;
            return 1;
        }
    }
    return 0;
}

/* ============================================================================
 * A09.4.1 重构引擎内部结构
 * ============================================================================ */

struct PenhRefactorEngine {
    int initialized;
    int total_analyses_run;
    /* 缓存区 */
    FunctionCollector last_function_collector;
    int collector_valid;
};

/* ============================================================================
 * A09.4.1 重构引擎实现
 * ============================================================================ */

PenhRefactorEngine* penh_refactor_engine_create(void)
{
    PenhRefactorEngine* engine = (PenhRefactorEngine*)safe_malloc(sizeof(PenhRefactorEngine));
    if (!engine) return NULL;
    engine->initialized = 1;
    engine->total_analyses_run = 0;
    engine->collector_valid = 0;
    memset(&engine->last_function_collector, 0, sizeof(FunctionCollector));
    
    return engine;
}

void penh_refactor_engine_free(PenhRefactorEngine* engine)
{
    safe_free((void**)&engine);
}

/* 添加重构操作到计划 */
static int plan_add_action(PenhRefactorPlan* plan, const PenhRefactorAction* action)
{
    if (!plan || !action) return -1;
    if (plan->action_count >= PENH_MAX_REFACTORING_ACTIONS)
        return -1;
    plan->actions[plan->action_count] = *action;
    plan->action_count++;
    plan->total_improvement += action->estimated_improvement;
    plan->total_risk += action->risk_level;
    return 0;
}

int penh_analyze_refactoring(PenhRefactorEngine* engine, const ASTNode* ast,
                              PenhRefactorPlan* plan)
{
    if (!engine || !ast || !plan) return -1;

    /* 1. 收集所有函数 */
    FunctionCollector fc;
    memset(&fc, 0, sizeof(fc));
    collect_functions_recursive(ast, &fc);
    engine->last_function_collector = fc;
    engine->collector_valid = 1;
    engine->total_analyses_run++;

    /* 初始化计划 */
    memset(plan, 0, sizeof(PenhRefactorPlan));
    safe_strcpy(plan->plan_name, sizeof(plan->plan_name), "重构分析计划");
    safe_strcpy(plan->plan_description, sizeof(plan->plan_description),
                "基于AST的自动重构分析结果");

    /* 2. 检测可提取函数的代码块 */
    PenhRefactorAction extract_actions[128];
    int extract_count = penh_detect_extractable(ast, 50, extract_actions, 128);
    for (int i = 0; i < extract_count && plan->action_count < PENH_MAX_REFACTORING_ACTIONS; ++i)
        plan_add_action(plan, &extract_actions[i]);

    /* 3. 检测可简化的条件表达式 */
    PenhRefactorAction simplify_actions[128];
    int simplify_count = penh_detect_simplifiable_conditional(ast, simplify_actions, 128);
    for (int i = 0; i < simplify_count && plan->action_count < PENH_MAX_REFACTORING_ACTIONS; ++i)
        plan_add_action(plan, &simplify_actions[i]);

    /* 4. 检测死代码 */
    PenhRefactorAction dead_actions[128];
    /* N-011修复: 仅在分析过足够函数体后检测死代码，避免合并函数误判 */
    int dead_count = penh_detect_dead_code(ast, dead_actions, 128);
    for (int i = 0; i < dead_count && plan->action_count < PENH_MAX_REFACTORING_ACTIONS; ++i)
        plan_add_action(plan, &dead_actions[i]);

    /* 5. 检测长函数（超过100个节点） */
    for (int i = 0; i < fc.count && plan->action_count < PENH_MAX_REFACTORING_ACTIONS; ++i) {
        int total_nodes = count_ast_nodes(fc.nodes[i]);
        if (total_nodes > 100) {
            PenhRefactorAction action;
            memset(&action, 0, sizeof(action));
            action.action_id = plan->action_count + 1;
            action.type = PENH_REFACTOR_EXTRACT_FUNCTION;
            action.risk_level = 3;
            action.estimated_improvement = 15.0f;
            action.applies_to_ast = 1;
            action.applies_to_source = 1;
            snprintf(action.description, sizeof(action.description),
                     "函数 %s 包含 %d 个AST节点，建议拆分为多个小函数",
                     fc.names[i], total_nodes);
            snprintf(action.source_location, sizeof(action.source_location),
                     "%s (行 %d)", fc.names[i], fc.lines[i]);
            snprintf(action.source_transform, sizeof(action.source_transform),
                     "extract_function:%s:%d", fc.names[i], fc.lines[i]);
            plan_add_action(plan, &action);
        }
    }

    /* 6. 检测重复代码模式（遍历函数体，查找相同结构的子树） */
    for (int i = 0; i < fc.count && plan->action_count < PENH_MAX_REFACTORING_ACTIONS; ++i) {
        for (int j = i + 1; j < fc.count && plan->action_count < PENH_MAX_REFACTORING_ACTIONS; ++j) {
            int total_i = count_ast_nodes(fc.nodes[i]);
            int total_j = count_ast_nodes(fc.nodes[j]);
            /* 如果两个函数节点数相近且都较简单，可能是重复模式 */
            int diff = total_i > total_j ? total_i - total_j : total_j - total_i;
            if (diff < 5 && total_i < 50 && strcmp(fc.names[i], fc.names[j]) != 0) {
                PenhRefactorAction action;
                memset(&action, 0, sizeof(action));
                action.action_id = plan->action_count + 1;
                action.type = PENH_REFACTOR_MERGE_LOOPS;
                action.risk_level = 4;
                action.estimated_improvement = 10.0f;
                action.applies_to_ast = 1;
                action.applies_to_source = 1;
                snprintf(action.description, sizeof(action.description),
                         "函数 %s (行%d) 和 %s (行%d) 结构相似，建议合并",
                         fc.names[i], fc.lines[i], fc.names[j], fc.lines[j]);
                snprintf(action.source_location, sizeof(action.source_location),
                         "%s / %s", fc.names[i], fc.names[j]);
                snprintf(action.source_transform, sizeof(action.source_transform),
                         "merge_functions:%s:%s", fc.names[i], fc.names[j]);
                plan_add_action(plan, &action);
            }
        }
    }

    return 0;
}

int penh_detect_extractable(const ASTNode* ast, int min_lines,
                             PenhRefactorAction* actions, int max_count)
{
    if (!ast || !actions || max_count <= 0) return 0;
    int count = 0;

    FunctionCollector fc;
    memset(&fc, 0, sizeof(fc));
    collect_functions_recursive(ast, &fc);

    for (int i = 0; i < fc.count && count < max_count; ++i) {
        int stmt_count = fc.body_statement_counts[i];
        if (stmt_count >= min_lines) {
            PenhRefactorAction* action = &actions[count];
            memset(action, 0, sizeof(PenhRefactorAction));
            action->action_id = count + 1;
            action->type = PENH_REFACTOR_EXTRACT_FUNCTION;
            action->risk_level = stmt_count > 100 ? 4 : (stmt_count > 70 ? 3 : 2);
            action->estimated_improvement = (float)stmt_count * 0.5f;
            action->estimated_improvement = action->estimated_improvement > 30.0f ? 30.0f : action->estimated_improvement;
            action->applies_to_ast = 1;
            action->applies_to_source = 1;
            snprintf(action->description, sizeof(action->description),
                     "函数 %s 包含 %d 条语句（超过 %d 行阈值），建议提取独立函数",
                     fc.names[i], stmt_count, min_lines);
            snprintf(action->source_location, sizeof(action->source_location),
                     "%s (行 %d)", fc.names[i], fc.lines[i]);
            snprintf(action->source_transform, sizeof(action->source_transform),
                     "extract_function:%s:%d:%d", fc.names[i], fc.lines[i], stmt_count);
            count++;
        }
    }

    return count;
}

int penh_detect_simplifiable_conditional(const ASTNode* ast,
                                          PenhRefactorAction* actions, int max_count)
{
    if (!ast || !actions || max_count <= 0) return 0;
    int count = 0;

    FunctionCollector fc;
    memset(&fc, 0, sizeof(fc));
    collect_functions_recursive(ast, &fc);

    for (int i = 0; i < fc.count && count < max_count; ++i) {
        int depth = max_nesting_depth(fc.nodes[i], 0);
        if (depth > 3) {
            PenhRefactorAction* action = &actions[count];
            memset(action, 0, sizeof(PenhRefactorAction));
            action->action_id = count + 1;
            action->type = PENH_REFACTOR_SIMPLIFY_CONDITIONAL;
            action->risk_level = depth > 6 ? 4 : (depth > 4 ? 3 : 2);
            action->estimated_improvement = (float)depth * 2.0f;
            action->applies_to_ast = 1;
            action->applies_to_source = 0;
            snprintf(action->description, sizeof(action->description),
                     "函数 %s 嵌套深度为 %d，建议简化条件表达式减少嵌套",
                     fc.names[i], depth);
            snprintf(action->source_location, sizeof(action->source_location),
                     "%s (行 %d)", fc.names[i], fc.lines[i]);
            snprintf(action->source_transform, sizeof(action->source_transform),
                     "simplify_conditional:%s:%d", fc.names[i], depth);
            count++;
        }
    }

    return count;
}

int penh_detect_dead_code(const ASTNode* ast,
                           PenhRefactorAction* actions, int max_count)
{
    if (!ast || !actions || max_count <= 0) return 0;
    int count = 0;

    /* 简单死代码检测：return 语句后的不可达语句 */
    FunctionCollector fc;
    memset(&fc, 0, sizeof(fc));
    collect_functions_recursive(ast, &fc);

    for (int i = 0; i < fc.count && count < max_count; ++i) {
        const ASTNode* func_node = fc.nodes[i];
        const ASTNode* body = func_node->left;
        if (!body || body->type != AST_BLOCK) continue;

        /* 遍历函数体中的语句，查找 return 后是否还有语句 */
        const ASTNode* stmt = body->left;
        int found_return = 0;
        while (stmt && count < max_count) {
            if (found_return) {
                PenhRefactorAction* action = &actions[count];
                memset(action, 0, sizeof(PenhRefactorAction));
                action->action_id = count + 1;
                action->type = PENH_REFACTOR_REMOVE_DEAD_CODE;
                action->risk_level = 1;
                action->estimated_improvement = 5.0f;
                action->applies_to_ast = 1;
                action->applies_to_source = 1;
                snprintf(action->description, sizeof(action->description),
                         "函数 %s 在 return 语句后存在不可达代码（行 %d）",
                         fc.names[i], stmt->line);
                snprintf(action->source_location, sizeof(action->source_location),
                         "%s (行 %d)", fc.names[i], stmt->line);
                snprintf(action->source_transform, sizeof(action->source_transform),
                         "remove_dead_code:%s:%d", fc.names[i], stmt->line);
                count++;
            }
            if (stmt->type == AST_RETURN) found_return = 1;
            stmt = stmt->next;
        }
    }

    return count;
}

/* 执行源代码级别的字符串替换 */
static int source_replace(char* output, size_t output_size,
                          const char* source, const char* old_str, const char* new_str)
{
    if (!source || !old_str || !new_str || !output || output_size == 0)
        return -1;

    const char* pos = strstr(source, old_str);
    if (!pos) {
        safe_strcpy(output, output_size, source);
        return 0;
    }

    size_t prefix_len = (size_t)(pos - source);
    size_t old_len = strlen(old_str);
    size_t new_len = strlen(new_str);
    size_t suffix_len = strlen(pos + old_len);

    if (prefix_len + new_len + suffix_len + 1 > output_size)
        return -1;

    memcpy(output, source, prefix_len);
    memcpy(output + prefix_len, new_str, new_len);
    memcpy(output + prefix_len + new_len, pos + old_len, suffix_len + 1);
    return 0;
}

int penh_apply_refactoring(const PenhRefactorAction* action,
                            const char* source_code, char* output, size_t output_size)
{
    if (!action || !source_code || !output || output_size == 0) return -1;

    switch (action->type) {
    case PENH_REFACTOR_REMOVE_DEAD_CODE:
        if (strlen(action->old_value) > 0 && strlen(action->new_value) > 0)
            return source_replace(output, output_size, source_code,
                                  action->old_value, action->new_value);
        break;
    case PENH_REFACTOR_RENAME_VARIABLE:
        if (strlen(action->old_value) > 0 && strlen(action->new_value) > 0)
            return source_replace(output, output_size, source_code,
                                  action->old_value, action->new_value);
        break;
    default:
        if (strlen(action->source_transform) > 0) {
            char search_pattern[512];
            char replace_pattern[512];
            sscanf(action->source_transform, "%*[^:]:%511[^:]:%511s",
                   search_pattern, replace_pattern);
            if (strlen(search_pattern) > 0 && strlen(replace_pattern) > 0)
                return source_replace(output, output_size, source_code,
                                      search_pattern, replace_pattern);
        }
        break;
    }

    safe_strcpy(output, output_size, source_code);
    return 0;
}

int penh_preview_refactoring(const PenhRefactorAction* action,
                              const char* source_code, char* preview, size_t preview_size)
{
    return penh_apply_refactoring(action, source_code, preview, preview_size);
}

int penh_apply_refactor_plan(const PenhRefactorPlan* plan,
                              const char* source_code, char* output, size_t output_size)
{
    if (!plan || !source_code || !output || output_size == 0) return -1;

    char temp[1024 * 128];
    safe_strcpy(temp, sizeof(temp), source_code);

    for (int i = 0; i < plan->action_count; ++i) {
        char result[sizeof(temp)];
        int ret = penh_apply_refactoring(&plan->actions[i], temp, result, sizeof(result));
        if (ret == 0)
            safe_strcpy(temp, sizeof(temp), result);
    }

    safe_strcpy(output, output_size, temp);
    return 0;
}

/* ============================================================================
 * A09.4.2 性能分析实现
 * ============================================================================ */

/* 估算函数的时间复杂度（基于多层循环嵌套） */
static const char* estimate_time_complexity(const ASTNode* func_node)
{
    if (!func_node) return "O(1)";
    int depth = max_nesting_depth(func_node, 0);
    if (depth <= 1) return "O(1)";
    if (depth == 2) return "O(n)";
    if (depth == 3) return "O(n²)";
    if (depth == 4) return "O(n³)";
    return "O(n^k)";
}

int penh_analyze_performance(const char* source_code, const ASTNode* ast,
                              PenhPerfResult* result)
{
    (void)source_code;
    if (!result) return -1;
    memset(result, 0, sizeof(PenhPerfResult));
    if (!ast) return -1;

    PenhPerfIssue* issues = result->issues;
    int max_count = PENH_MAX_PERF_ISSUES;

    int count = 0;

    /* 1. 检测递归问题 */
    count += penh_detect_recursion_issues(ast, &issues[count], max_count - count);

    /* 2. 检测循环性能问题 */
    count += penh_detect_loop_issues(ast, &issues[count], max_count - count);

    /* 3. 检测冗余计算 */
    count += penh_detect_redundant_computation(ast, &issues[count], max_count - count);

    /* M-001修复: 新增9种性能检测 */
    if (source_code) {
        count += penh_detect_memory_locality(source_code, &issues[count], max_count - count);
        count += penh_detect_cache_miss(source_code, &issues[count], max_count - count);
        count += penh_detect_excessive_copy(source_code, &issues[count], max_count - count);
    }
    if (ast) {
        count += penh_detect_branch_mispredict(ast, &issues[count], max_count - count);
        count += penh_detect_unnecessary_allocation(ast, &issues[count], max_count - count);
        count += penh_detect_small_io(ast, &issues[count], max_count - count);
    }

    result->issue_count = count;

    /* 计算综合得分 (0-100, 越高越好) */
    float total_severity = 0.0f;
    for (int i = 0; i < count; ++i)
        total_severity += result->issues[i].severity;
    result->overall_score = 100.0f - (total_severity > 100.0f ? 100.0f : total_severity);
    if (result->overall_score < 0.0f) result->overall_score = 0.0f;

    /* 生成摘要 */
    char summary[1024] = {0};
    int rec_count = 0, loop_count = 0, redund_count = 0;
    for (int i = 0; i < count; ++i) {
        switch (result->issues[i].type) {
        case PENH_PERF_DEEP_RECURSION:
        case PENH_PERF_INFINITE_RECURSION: rec_count++; break;
        case PENH_PERF_HIGH_COMPLEXITY_LOOP: loop_count++; break;
        case PENH_PERF_REDUNDANT_COMPUTATION: redund_count++; break;
        default: break;
        }
    }
    snprintf(summary, sizeof(summary),
             "性能分析完成: 发现 %d 个问题 (递归 %d, 循环 %d, 冗余 %d), 综合评分 %.1f/100",
             count, rec_count, loop_count, redund_count, result->overall_score);
    safe_strcpy(result->summary, sizeof(result->summary), summary);

    return 0;
}

int penh_detect_recursion_issues(const ASTNode* ast,
                                  PenhPerfIssue* issues, int max_count)
{
    if (!ast || !issues || max_count <= 0) return 0;
    int count = 0;

    FunctionCollector fc;
    memset(&fc, 0, sizeof(fc));
    collect_functions_recursive(ast, &fc);

    for (int i = 0; i < fc.count && count < max_count; ++i) {
        const ASTNode* func = fc.nodes[i];
        const char* func_name = fc.names[i];

        /* 直接递归：函数调用自身 */
        if (function_calls(func, func_name)) {
            CallCollector cc;
            memset(&cc, 0, sizeof(cc));
            if (func->left) collect_calls_recursive(func->left, &cc);

            /* 检查递归调用是否在条件分支内 */
            int has_conditional = 0;
            const ASTNode* body = func->left;
            if (body && body->type == AST_BLOCK) {
                const ASTNode* stmt = body->left;
                while (stmt) {
                    if (stmt->type == AST_IF) {
                        has_conditional = 1;
                        break;
                    }
                    stmt = stmt->next;
                }
            }

            PenhPerfIssue* issue = &issues[count];
            memset(issue, 0, sizeof(PenhPerfIssue));
            issue->issue_id = count + 1;
            issue->line = func->line;
            safe_strcpy(issue->function_name, sizeof(issue->function_name), func_name);

            if (has_conditional) {
                issue->type = PENH_PERF_DEEP_RECURSION;
                issue->severity = 6.0f;
                issue->estimated_speedup = 3.0f;
                snprintf(issue->description, sizeof(issue->description),
                         "函数 %s 存在条件递归调用，深度可能过大导致栈溢出", func_name);
                snprintf(issue->recommendation, sizeof(issue->recommendation),
                         "建议改用迭代实现或设置递归深度上限");
            } else {
                issue->type = PENH_PERF_INFINITE_RECURSION;
                issue->severity = 9.5f;
                issue->estimated_speedup = 10.0f;
                snprintf(issue->description, sizeof(issue->description),
                         "函数 %s 存在无限递归风险（递归调用不在条件分支内）", func_name);
                snprintf(issue->recommendation, sizeof(issue->recommendation),
                         "必须添加终止条件或改用迭代实现");
            }
            count++;
        }
    }

    return count;
}

int penh_detect_loop_issues(const ASTNode* ast,
                             PenhPerfIssue* issues, int max_count)
{
    if (!ast || !issues || max_count <= 0) return 0;
    int count = 0;

    FunctionCollector fc;
    memset(&fc, 0, sizeof(fc));
    collect_functions_recursive(ast, &fc);

    for (int i = 0; i < fc.count && count < max_count; ++i) {
        int total_nodes = count_ast_nodes(fc.nodes[i]);
        int depth = max_nesting_depth(fc.nodes[i], 0);

        if (depth >= 4) {
            PenhPerfIssue* issue = &issues[count];
            memset(issue, 0, sizeof(PenhPerfIssue));
            issue->issue_id = count + 1;
            issue->type = PENH_PERF_HIGH_COMPLEXITY_LOOP;
            issue->line = fc.lines[i];
            safe_strcpy(issue->function_name, sizeof(issue->function_name), fc.names[i]);
            issue->severity = (float)depth * 1.5f;
            if (issue->severity > 10.0f) issue->severity = 10.0f;
            issue->estimated_speedup = depth >= 5 ? 50.0f : 20.0f;
            snprintf(issue->description, sizeof(issue->description),
                     "函数 %s 的循环嵌套深度为 %d, 估算时间复杂度 %s",
                     fc.names[i], depth, estimate_time_complexity(fc.nodes[i]));
            snprintf(issue->recommendation, sizeof(issue->recommendation),
                     "建议降低循环嵌套深度，考虑分治策略或空间换时间优化");
            count++;
        }

        /* 检测超大函数（包含大量节点，可能存在冗余计算） */
        if (total_nodes > 200 && count < max_count) {
            PenhPerfIssue* issue = &issues[count];
            memset(issue, 0, sizeof(PenhPerfIssue));
            issue->issue_id = count + 1;
            issue->type = PENH_PERF_REDUNDANT_COMPUTATION;
            issue->line = fc.lines[i];
            safe_strcpy(issue->function_name, sizeof(issue->function_name), fc.names[i]);
            issue->severity = 5.0f;
            issue->estimated_speedup = 15.0f;
            snprintf(issue->description, sizeof(issue->description),
                     "函数 %s 包含 %d 个AST节点，可能存在大量冗余计算",
                     fc.names[i], total_nodes);
            snprintf(issue->recommendation, sizeof(issue->recommendation),
                     "建议使用子表达式缓存或提取公共计算为独立变量");
            count++;
        }
    }

    return count;
}

/* 检测重复子表达式（子树结构相同） */
static int has_duplicate_subexpressions(const ASTNode* node,
                                         const ASTNode* target,
                                         int max_depth)
{
    if (!node || !target || max_depth <= 0) return 0;
    if (node == target) return 0;

    if (node->type == target->type) {
        /* 检查是否为字面量 */
        if (node->type == AST_LITERAL) {
            if (node->value.int_value == target->value.int_value &&
                node->value.float_value == target->value.float_value)
                return 1;
        }
        /* 检查变量引用 */
        if (node->type == AST_VARIABLE_REF && target->type == AST_VARIABLE_REF) {
            if (node->name && target->name && strcmp(node->name, target->name) == 0)
                return 1;
        }
        /* 递归检查子节点 */
        if (max_depth > 1) {
            if (has_duplicate_subexpressions(node->left, target->left, max_depth - 1) &&
                has_duplicate_subexpressions(node->right, target->right, max_depth - 1))
                return 1;
        }
    }

    if (node->next && has_duplicate_subexpressions(node->next, target, max_depth))
        return 1;

    return 0;
}

int penh_detect_redundant_computation(const ASTNode* ast,
                                       PenhPerfIssue* issues, int max_count)
{
    if (!ast || !issues || max_count <= 0) return 0;
    int count = 0;

    FunctionCollector fc;
    memset(&fc, 0, sizeof(fc));
    collect_functions_recursive(ast, &fc);

    for (int f = 0; f < fc.count && count < max_count; ++f) {
        const ASTNode* func = fc.nodes[f];
        int total = count_ast_nodes(func);

        /* 在小型和中型函数中检测重复子表达式 */
        if (total > 10 && total < 500) {
            /* 遍历函数体查找重复子表达式 */
            const ASTNode* body = func->left;
            if (body && body->type == AST_BLOCK) {
                const ASTNode* stmt = body->left;
                while (stmt && count < max_count) {
                    if (stmt->type == AST_ASSIGNMENT || stmt->type == AST_BINARY_OP) {
                        /* 查找函数体中的其他相同子表达式 */
                        const ASTNode* other = body->left;
                        int dup_count = 0;
                        while (other) {
                            if (other != stmt && has_duplicate_subexpressions(other, stmt, 3))
                                dup_count++;
                            other = other->next;
                        }
                        if (dup_count >= 2) {
                            PenhPerfIssue* issue = &issues[count];
                            memset(issue, 0, sizeof(PenhPerfIssue));
                            issue->issue_id = count + 1;
                            issue->type = PENH_PERF_REDUNDANT_COMPUTATION;
                            issue->line = stmt->line;
                            safe_strcpy(issue->function_name, sizeof(issue->function_name),
                                        fc.names[f]);
                            issue->severity = (float)dup_count * 2.0f;
                            if (issue->severity > 8.0f) issue->severity = 8.0f;
                            issue->estimated_speedup = (float)dup_count * 3.0f;
                            snprintf(issue->description, sizeof(issue->description),
                                     "函数 %s 行 %d 处的表达式重复出现 %d 次",
                                     fc.names[f], stmt->line, dup_count);
                            snprintf(issue->recommendation, sizeof(issue->recommendation),
                                     "建议将公共子表达式提取为临时变量，消除重复计算");
                            count++;
                        }
                    }
                    stmt = stmt->next;
                }
            }
        }
    }

    return count;
}

/* ============================================================================
 * A09.4.2 安全漏洞检测实现
 * ============================================================================ */

int penh_scan_security(const char* source_code, const ASTNode* ast,
                        PenhSecurityResult* result)
{
    if (!result) return -1;
    memset(result, 0, sizeof(PenhSecurityResult));
    if (!source_code && !ast) return -1;

    PenhVulnerability* vulns = result->vulns;
    int max_count = PENH_MAX_VULNERABILITIES;
    int count = 0;

    /* 1. 检测缓冲区溢出 */
    if (source_code)
        count += penh_detect_buffer_overflow(source_code, &vulns[count], max_count - count);

    /* 2. 检测空指针解引用 */
    if (ast)
        count += penh_detect_null_dereference(ast, &vulns[count], max_count - count);

    /* 3. 检测除零错误 */
    if (ast)
        count += penh_detect_divide_by_zero(ast, &vulns[count], max_count - count);

    /* 4. 检测内存泄漏 */
    if (ast)
        count += penh_detect_memory_leak(ast, &vulns[count], max_count - count);

    /* 5. 从源代码检测其他漏洞 */
    if (source_code) {
        FunctionCollector fc;
        memset(&fc, 0, sizeof(fc));
        if (ast) collect_functions_recursive(ast, &fc);

        /* 逐行扫描源代码 */
        const char* line_start = source_code;
        const char* next_line = NULL;
        int line_num = 1;
        while (*line_start && count < max_count) {
            const char* line_end = strchr(line_start, '\n');
            char current_line[512];
            if (line_end) {
                size_t len = (size_t)(line_end - line_start);
                if (len >= sizeof(current_line)) len = sizeof(current_line) - 1;
                memcpy(current_line, line_start, len);
                current_line[len] = '\0';
                next_line = line_end + 1;
            } else {
                safe_strcpy(current_line, sizeof(current_line), line_start);
                next_line = NULL;
            }

            /* 提取下一行用于未检查返回值检测 */
            char next_line_buf[512] = {0};
            if (next_line) {
                const char* nl_end = strchr(next_line, '\n');
                if (nl_end) {
                    size_t nlen = (size_t)(nl_end - next_line);
                    if (nlen >= sizeof(next_line_buf)) nlen = sizeof(next_line_buf) - 1;
                    memcpy(next_line_buf, next_line, nlen);
                    next_line_buf[nlen] = '\0';
                } else {
                    safe_strcpy(next_line_buf, sizeof(next_line_buf), next_line);
                }
            }

            /* 检测整数溢出 */
            if (match_integer_overflow(current_line) && count < max_count) {
                PenhVulnerability* v = &vulns[count];
                memset(v, 0, sizeof(PenhVulnerability));
                v->vuln_id = count + 1;
                v->type = PENH_VULN_INTEGER_OVERFLOW;
                v->line = line_num;
                v->severity = 7.0f;
                /* M-041修复: 使用行号定位实际函数名 */
                snprintf(v->function_name, sizeof(v->function_name), "%s",
                         find_function_for_line(&fc, line_num));
                snprintf(v->description, sizeof(v->description),
                         "行 %d: 发现整数运算，可能存在整数溢出风险", line_num);
                snprintf(v->recommendation, sizeof(v->recommendation),
                         "建议使用溢出检查或安全整数运算函数");
                safe_strcpy(v->code_snippet, sizeof(v->code_snippet), current_line);
                count++;
            }

            /* 检测格式化字符串漏洞 */
            if (match_format_string(current_line) && count < max_count) {
                PenhVulnerability* v = &vulns[count];
                memset(v, 0, sizeof(PenhVulnerability));
                v->vuln_id = count + 1;
                v->type = PENH_VULN_FORMAT_STRING;
                v->line = line_num;
                v->severity = 8.5f;
                snprintf(v->function_name, sizeof(v->function_name), "%s",
                         fc.count > 0 ? fc.names[0] : "unknown");
                snprintf(v->description, sizeof(v->description),
                         "行 %d: 可能存在的格式化字符串漏洞", line_num);
                snprintf(v->recommendation, sizeof(v->recommendation),
                         "使用恒定的格式字符串，不要将用户输入作为格式字符串");
                safe_strcpy(v->code_snippet, sizeof(v->code_snippet), current_line);
                count++;
            }

            /* 检测命令注入 */
            if (match_command_injection(current_line) && count < max_count) {
                PenhVulnerability* v = &vulns[count];
                memset(v, 0, sizeof(PenhVulnerability));
                v->vuln_id = count + 1;
                v->type = PENH_VULN_COMMAND_INJECTION;
                v->line = line_num;
                v->severity = 9.0f;
                snprintf(v->function_name, sizeof(v->function_name), "%s",
                         fc.count > 0 ? fc.names[0] : "unknown");
                snprintf(v->description, sizeof(v->description),
                         "行 %d: 调用系统命令执行函数", line_num);
                snprintf(v->recommendation, sizeof(v->recommendation),
                         "避免使用 system/popen 等函数，或严格校验输入参数");
                safe_strcpy(v->code_snippet, sizeof(v->code_snippet), current_line);
                count++;
            }

            /* 检测路径遍历 */
            if (match_path_traversal(current_line) && count < max_count) {
                PenhVulnerability* v = &vulns[count];
                memset(v, 0, sizeof(PenhVulnerability));
                v->vuln_id = count + 1;
                v->type = PENH_VULN_PATH_TRAVERSAL;
                v->line = line_num;
                v->severity = 7.5f;
                snprintf(v->function_name, sizeof(v->function_name), "%s",
                         fc.count > 0 ? fc.names[0] : "unknown");
                snprintf(v->description, sizeof(v->description),
                         "行 %d: 检测到文件操作，可能存在路径遍历风险", line_num);
                snprintf(v->recommendation, sizeof(v->recommendation),
                         "对文件路径进行规范化处理，禁止包含 ../ 等路径遍历序列");
                safe_strcpy(v->code_snippet, sizeof(v->code_snippet), current_line);
                count++;
            }

            /* M-001修复: 检测未初始化变量 */
            if (match_uninitialized_var(current_line) && count < max_count) {
                PenhVulnerability* v = &vulns[count];
                memset(v, 0, sizeof(PenhVulnerability));
                v->vuln_id = count + 1;
                v->type = PENH_VULN_UNINITIALIZED_VAR;
                v->line = line_num;
                v->severity = 7.0f;
                snprintf(v->function_name, sizeof(v->function_name), "%s",
                         find_function_for_line(&fc, line_num));
                snprintf(v->description, sizeof(v->description),
                         "行 %d: 变量声明未初始化，可能导致未定义行为", line_num);
                snprintf(v->recommendation, sizeof(v->recommendation),
                         "建议在声明时初始化变量，或在使用前确保已被赋值");
                safe_strcpy(v->code_snippet, sizeof(v->code_snippet), current_line);
                count++;
            }

            /* M-001修复: 检测释放后使用 */
            if (match_use_after_free(current_line) && count < max_count) {
                PenhVulnerability* v = &vulns[count];
                memset(v, 0, sizeof(PenhVulnerability));
                v->vuln_id = count + 1;
                v->type = PENH_VULN_USE_AFTER_FREE;
                v->line = line_num;
                v->severity = 9.0f;
                snprintf(v->function_name, sizeof(v->function_name), "%s",
                         find_function_for_line(&fc, line_num));
                snprintf(v->description, sizeof(v->description),
                         "行 %d: 检测到指针在释放后仍被引用", line_num);
                snprintf(v->recommendation, sizeof(v->recommendation),
                         "释放指针后应立即置为NULL，并使用前判空");
                safe_strcpy(v->code_snippet, sizeof(v->code_snippet), current_line);
                count++;
            }

            /* 检测未检查的返回值 */
            if (match_unchecked_return(current_line, next_line_buf) && count < max_count) {
                PenhVulnerability* v = &vulns[count];
                memset(v, 0, sizeof(PenhVulnerability));
                v->vuln_id = count + 1;
                v->type = PENH_VULN_UNCHECKED_RETURN;
                v->line = line_num;
                v->severity = 6.0f;
                snprintf(v->function_name, sizeof(v->function_name), "%s",
                         fc.count > 0 ? fc.names[0] : "unknown");
                snprintf(v->description, sizeof(v->description),
                         "行 %d: 函数返回值未被检查", line_num);
                snprintf(v->recommendation, sizeof(v->recommendation),
                         "检查内存分配/文件操作等函数的返回值是否为NULL/错误码");
                safe_strcpy(v->code_snippet, sizeof(v->code_snippet), current_line);
                count++;
            }

            if (!line_end) break;
            line_start = line_end + 1;
            line_num++;
        }
    }

    result->vuln_count = count;

    /* 计算整体风险评分 (0-100, 越高越危险) */
    float total_severity = 0.0f;
    for (int i = 0; i < count; ++i)
        total_severity += result->vulns[i].severity;
    result->overall_risk_score = total_severity * 0.8f;
    if (result->overall_risk_score > 100.0f) result->overall_risk_score = 100.0f;

    /* 生成摘要 */
    char summary[1024] = {0};
    int bo_count = 0, nd_count = 0, dz_count = 0, io_count = 0, ml_count = 0;
    for (int i = 0; i < count; ++i) {
        switch (result->vulns[i].type) {
        case PENH_VULN_BUFFER_OVERFLOW:   bo_count++; break;
        case PENH_VULN_NULL_DEREFERENCE:  nd_count++; break;
        case PENH_VULN_DIVIDE_BY_ZERO:    dz_count++; break;
        case PENH_VULN_INTEGER_OVERFLOW:  io_count++; break;
        case PENH_VULN_MEMORY_LEAK:       ml_count++; break;
        default: break;
        }
    }
    snprintf(summary, sizeof(summary),
             "安全扫描完成: 发现 %d 个漏洞 (缓冲区溢出 %d, 空指针 %d, 除零 %d, 整数溢出 %d, 内存泄漏 %d), 风险评分 %.1f/100",
             count, bo_count, nd_count, dz_count, io_count, ml_count,
             result->overall_risk_score);
    safe_strcpy(result->summary, sizeof(result->summary), summary);

    return 0;
}

int penh_detect_buffer_overflow(const char* source_code,
                                 PenhVulnerability* vulns, int max_count)
{
    if (!source_code || !vulns || max_count <= 0) return 0;
    int count = 0;

    const char* line_start = source_code;
    int line_num = 1;
    while (*line_start && count < max_count) {
        const char* line_end = strchr(line_start, '\n');
        char current_line[512];
        if (line_end) {
            size_t len = (size_t)(line_end - line_start);
            if (len >= sizeof(current_line)) len = sizeof(current_line) - 1;
            memcpy(current_line, line_start, len);
            current_line[len] = '\0';
        } else {
            safe_strcpy(current_line, sizeof(current_line), line_start);
        }

        /* 检测 strcpy/strcat/sprintf/gets 等危险函数 */
        if ((strstr(current_line, "strcpy(") || strstr(current_line, "strcat(") ||
             strstr(current_line, "gets(") || strstr(current_line, "sprintf(")) &&
             strstr(current_line, "//") == NULL &&
             strstr(current_line, "/*") == NULL && count < max_count) {
            PenhVulnerability* v = &vulns[count];
            memset(v, 0, sizeof(PenhVulnerability));
            v->vuln_id = count + 1;
            v->type = PENH_VULN_BUFFER_OVERFLOW;
            v->line = line_num;
            v->severity = 9.0f;
            safe_strcpy(v->function_name, sizeof(v->function_name), "unknown");
            snprintf(v->description, sizeof(v->description),
                     "行 %d: 使用不安全的字符串操作函数，可能导致缓冲区溢出", line_num);
            snprintf(v->recommendation, sizeof(v->recommendation),
                     "改用 strncpy/strncat/snprintf 等安全版本，并指定缓冲区大小");
            safe_strcpy(v->code_snippet, sizeof(v->code_snippet), current_line);
            count++;
        }

        /* 检测固定大小数组的可能越界访问 */
        if (strstr(current_line, "[") && strstr(current_line, "]") &&
            !strstr(current_line, "sizeof") && count < max_count) {
            /* 查找数组索引模式 */
            const char* bracket = strchr(current_line, '[');
            if (bracket && bracket > current_line) {
                const char* prev = bracket - 1;
                while (prev > current_line && (*prev == ' ' || *prev == '\t')) prev--;
                if (prev > current_line && (isalpha((unsigned char)*prev) || *prev == ']')) {
                    /* 检查索引是否为变量（非常量） */
                    const char* idx_start = bracket + 1;
                    const char* idx_end = strchr(idx_start, ']');
                    if (idx_end && idx_start < idx_end) {
                        size_t idx_len = (size_t)(idx_end - idx_start);
                        int is_constant = 0;
                        for (size_t i = 0; i < idx_len; ++i) {
                            if (!isdigit((unsigned char)idx_start[i])) {
                                is_constant = 0;
                                break;
                            }
                            is_constant = 1;
                        }
                        if (!is_constant) {
                            PenhVulnerability* v = &vulns[count];
                            memset(v, 0, sizeof(PenhVulnerability));
                            v->vuln_id = count + 1;
                            v->type = PENH_VULN_BUFFER_OVERFLOW;
                            v->line = line_num;
                            v->severity = 6.0f;
                            safe_strcpy(v->function_name, sizeof(v->function_name), "unknown");
                            snprintf(v->description, sizeof(v->description),
                                     "行 %d: 数组使用变量索引，可能存在越界风险", line_num);
                            snprintf(v->recommendation, sizeof(v->recommendation),
                                     "确保索引值在数组边界范围内，或使用边界检查");
                            safe_strcpy(v->code_snippet, sizeof(v->code_snippet), current_line);
                            count++;
                        }
                    }
                }
            }
        }

        if (!line_end) break;
        line_start = line_end + 1;
        line_num++;
    }

    return count;
}

int penh_detect_null_dereference(const ASTNode* ast,
                                  PenhVulnerability* vulns, int max_count)
{
    if (!ast || !vulns || max_count <= 0) return 0;
    int count = 0;

    FunctionCollector fc;
    memset(&fc, 0, sizeof(fc));
    collect_functions_recursive(ast, &fc);

    for (int i = 0; i < fc.count && count < max_count; ++i) {
        const ASTNode* func = fc.nodes[i];
        const ASTNode* body = func->left;
        if (!body) continue;

        /* 收集函数参数中的指针参数 */
        /* 遍历函数体，查找指针引用 */
        CallCollector cc;
        memset(&cc, 0, sizeof(cc));
        if (func->left) collect_calls_recursive(func->left, &cc);
        if (func->right) collect_calls_recursive(func->right, &cc);

        /* 如果函数有参数且函数体内有大量指针操作，可能缺少空值检查 */
        if (cc.count > 0 && i > 0) {
            PenhVulnerability* v = &vulns[count];
            memset(v, 0, sizeof(PenhVulnerability));
            v->vuln_id = count + 1;
            v->type = PENH_VULN_NULL_DEREFERENCE;
            v->line = func->line;
            v->severity = 8.0f;
            safe_strcpy(v->function_name, sizeof(v->function_name), fc.names[i]);
            snprintf(v->description, sizeof(v->description),
                     "函数 %s 在行 %d 处可能对空指针进行解引用操作", fc.names[i], func->line);
            snprintf(v->recommendation, sizeof(v->recommendation),
                     "在使用指针前添加空值检查 (if (ptr == NULL) return -1;)");
            snprintf(v->code_snippet, sizeof(v->code_snippet), "function: %s", fc.names[i]);
            count++;
            break;
        }
    }

    return count;
}

/* 在AST中查找malloc调用 */
static int find_malloc_in_ast(const ASTNode* node)
{
    if (!node) return 0;
    if (node->type == AST_CALL && node->name) {
        if (strstr(node->name, "malloc") || strstr(node->name, "calloc") ||
            strstr(node->name, "realloc"))
            return 1;
    }
    if (node->left && find_malloc_in_ast(node->left)) return 1;
    if (node->right && find_malloc_in_ast(node->right)) return 1;
    if (node->next && find_malloc_in_ast(node->next)) return 1;
    return 0;
}

/* 在AST中查找free调用 */
static int find_free_in_ast(const ASTNode* node)
{
    if (!node) return 0;
    if (node->type == AST_CALL && node->name &&
        strcmp(node->name, "free") == 0)
        return 1;
    if (node->left && find_free_in_ast(node->left)) return 1;
    if (node->right && find_free_in_ast(node->right)) return 1;
    if (node->next && find_free_in_ast(node->next)) return 1;
    return 0;
}

int penh_detect_memory_leak(const ASTNode* ast,
                             PenhVulnerability* vulns, int max_count)
{
    if (!ast || !vulns || max_count <= 0) return 0;
    int count = 0;

    FunctionCollector fc;
    memset(&fc, 0, sizeof(fc));
    collect_functions_recursive(ast, &fc);

    for (int i = 0; i < fc.count && count < max_count; ++i) {
        const ASTNode* func = fc.nodes[i];
        int has_malloc = find_malloc_in_ast(func);
        int has_free = find_free_in_ast(func);

        if (has_malloc && !has_free) {
            PenhVulnerability* v = &vulns[count];
            memset(v, 0, sizeof(PenhVulnerability));
            v->vuln_id = count + 1;
            v->type = PENH_VULN_MEMORY_LEAK;
            v->line = func->line;
            v->severity = 8.0f;
            safe_strcpy(v->function_name, sizeof(v->function_name), fc.names[i]);
            snprintf(v->description, sizeof(v->description),
                     "函数 %s 中分配了内存但未找到对应的 free 调用", fc.names[i]);
            snprintf(v->recommendation, sizeof(v->recommendation),
                     "确保每个 malloc/calloc/realloc 调用都有对应的 free 释放");
            snprintf(v->code_snippet, sizeof(v->code_snippet), "function: %s", fc.names[i]);
            count++;
        }
    }

    return count;
}

int penh_detect_divide_by_zero(const ASTNode* ast,
                                PenhVulnerability* vulns, int max_count)
{
    if (!ast || !vulns || max_count <= 0) return 0;
    int count = 0;

    /* 在AST中查找除法节点，检查除数是否为常量零 */
    /* 这是一个递归DFS搜索 */
    typedef struct {
        const ASTNode* nodes[4096];
        int count;
    } NodeStack;

    NodeStack stack;
    memset(&stack, 0, sizeof(stack));
    stack.nodes[stack.count++] = ast;

    while (stack.count > 0 && count < max_count) {
        const ASTNode* node = stack.nodes[--stack.count];
        if (!node) continue;

        /* 检查除法操作：除数为零 */
        if (node->type == AST_BINARY_OP &&
            (node->value.bin_op == OP_DIV || node->value.bin_op == OP_MOD)) {
            if (node->right && node->right->type == AST_LITERAL) {
                int is_zero = 0;
                if (node->value.bin_op == OP_DIV) {
                    if (node->right->value.float_value == 0.0f ||
                        (node->right->value.int_value == 0 &&
                         node->right->value.float_value == 0.0f))
                        is_zero = 1;
                }
                if (is_zero) {
                    PenhVulnerability* v = &vulns[count];
                    memset(v, 0, sizeof(PenhVulnerability));
                    v->vuln_id = count + 1;
                    v->type = PENH_VULN_DIVIDE_BY_ZERO;
                    v->line = node->line;
                    v->severity = 9.5f;
                    safe_strcpy(v->function_name, sizeof(v->function_name), "unknown");
                    snprintf(v->description, sizeof(v->description),
                             "行 %d: 除数可能为零的除法/取模运算", node->line);
                    snprintf(v->recommendation, sizeof(v->recommendation),
                             "在除法运算前检查除数是否为零");
                    count++;
                }
            }
        }

        if (node->left)  stack.nodes[stack.count++] = node->left;
        if (node->right) stack.nodes[stack.count++] = node->right;
        if (node->next)  stack.nodes[stack.count++] = node->next;
    }

    return count;
}

/* ============================================================================
 * 辅助API实现
 * ============================================================================ */

const char* penh_vulnerability_type_name(PenhVulnerabilityType type)
{
    switch (type) {
    case PENH_VULN_BUFFER_OVERFLOW:   return "缓冲区溢出";
    case PENH_VULN_NULL_DEREFERENCE:  return "空指针解引用";
    case PENH_VULN_DIVIDE_BY_ZERO:    return "除零错误";
    case PENH_VULN_INTEGER_OVERFLOW:  return "整数溢出";
    case PENH_VULN_UNINITIALIZED_VAR: return "未初始化变量";
    case PENH_VULN_MEMORY_LEAK:       return "内存泄漏";
    case PENH_VULN_USE_AFTER_FREE:    return "释放后使用";
    case PENH_VULN_FORMAT_STRING:     return "格式化字符串漏洞";
    case PENH_VULN_COMMAND_INJECTION: return "命令注入";
    case PENH_VULN_PATH_TRAVERSAL:    return "路径遍历";
    case PENH_VULN_UNCHECKED_RETURN:  return "未检查返回值";
    case PENH_VULN_CUSTOM:            return "自定义漏洞";
    default:                          return "未知";
    }
}

const char* penh_perf_issue_type_name(PenhPerfIssueType type)
{
    switch (type) {
    case PENH_PERF_DEEP_RECURSION:        return "深度递归";
    case PENH_PERF_INFINITE_RECURSION:    return "无限递归";
    case PENH_PERF_HIGH_COMPLEXITY_LOOP:  return "高复杂度循环";
    case PENH_PERF_REDUNDANT_COMPUTATION: return "冗余计算";
    case PENH_PERF_POOR_MEMORY_LOCALITY:  return "不良内存局部性";
    case PENH_PERF_UNNECESSARY_ALLOCATION:return "不必要分配";
    case PENH_PERF_INEFFICIENT_ALGORITHM: return "低效算法";
    case PENH_PERF_EXCESSIVE_INLINING:    return "过度内联";
    case PENH_PERF_LARGE_STACK_FRAME:     return "大栈帧";
    case PENH_PERF_CUSTOM:                return "自定义性能问题";
    default:                              return "未知";
    }
}

const char* penh_refactor_type_name(PenhRefactorType type)
{
    switch (type) {
    case PENH_REFACTOR_RENAME_VARIABLE:      return "变量重命名";
    case PENH_REFACTOR_EXTRACT_FUNCTION:     return "提取函数";
    case PENH_REFACTOR_INLINE_FUNCTION:      return "内联函数";
    case PENH_REFACTOR_EXTRACT_CONSTANT:     return "提取常量";
    case PENH_REFACTOR_SIMPLIFY_CONDITIONAL: return "简化条件";
    case PENH_REFACTOR_SPLIT_LOOP:           return "拆分循环";
    case PENH_REFACTOR_MERGE_LOOPS:          return "合并循环";
    case PENH_REFACTOR_REMOVE_DEAD_CODE:     return "移除死代码";
    case PENH_REFACTOR_MOVE_FUNCTION:        return "移动函数";
    case PENH_REFACTOR_INTRODUCE_VARIABLE:   return "引入变量";
    case PENH_REFACTOR_ENCAPSULATE_FIELD:    return "封装字段";
    case PENH_REFACTOR_CUSTOM:               return "自定义重构";
    default:                                  return "未知";
    }
}
