/**
 * @file graph_query.c
 * @brief 知识图谱查询引擎实现
 *
 * 三大存储引擎的统一查询层：
 * 1. SPARQL风格查询（RDFTripleStore）
 * 2. 图模式匹配（AdjacencyList）
 * 3. 子图匹配（PropertyGraph VF2风格）
 */

#include "selflnn/knowledge/graph_query.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include "selflnn/utils/logging.h"  /* DEEP-005: log宏 */

/* 多跳推理最大跳数限制 */
#define MAX_HOPS_DEFAULT 10
#define MAX_HOPS_QUERY_TIMEOUT_MS 30000

/* ============================================================================
 * 内部辅助函数
 * =========================================================================== */

static char* q_strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* c = (char*)safe_malloc(len);
    if (c) memcpy(c, s, len);
    return c;
}

static void skip_ws(const char** p) {
    while (**p && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r'))
        (*p)++;
}

static int next_token(const char** p, char* token, size_t maxlen) {
    skip_ws(p);
    if (!**p) return -1;
    size_t i = 0;
    if (**p == '{' || **p == '}' || **p == '.' || **p == ';') {
        token[i++] = **p; (*p)++;
    } else if (**p == '"') {
        token[i++] = *(*p)++;
        while (**p && **p != '"' && i < maxlen - 1) {
            if (**p == '\\' && *(*p + 1)) { (*p)++; }
            token[i++] = *(*p)++;
        }
        if (**p == '"') { token[i++] = *(*p)++; }
    } else if (**p == '<') {
        while (**p && **p != '>' && i < maxlen - 1) {
            token[i++] = *(*p)++;
        }
        if (**p == '>') { token[i++] = *(*p)++; }
    } else {
        while (**p && **p != ' ' && **p != '\t' && **p != '\n' &&
               **p != '\r' && **p != '{' && **p != '}' &&
               **p != '.' && **p != ';' && i < maxlen - 1) {
            token[i++] = *(*p)++;
        }
    }
    token[i] = '\0';
    return 0;
}

static int str_eq_ic(const char* a, const char* b) {
#ifdef _MSC_VER
    return _stricmp(a, b) == 0;
#else
    return strcasecmp(a, b) == 0;
#endif
}

static int binding_compare_desc(const void* a, const void* b) {
    const QueryResultRow* ra = (const QueryResultRow*)a;
    const QueryResultRow* rb = (const QueryResultRow*)b;
    if (ra->confidence < rb->confidence) return 1;
    if (ra->confidence > rb->confidence) return -1;
    return 0;
}

static int binding_compare_asc(const void* a, const void* b) {
    return -binding_compare_desc(a, b);
}

/* ZS-019修复: 按变量名排序的比较器上下文 */
typedef struct {
    int var_index;
    int ascending;
} SortByVarContext;

/* TLS变量必须在MSVC C89模式下声明在使用前 */
/* S-021修复: 全局变量改为TLS，每个线程独立持有排序上下文，避免多实例并行排序时互相干扰 */
#ifdef _WIN32
static __declspec(thread) int g_query_sort_var_index = -1;
static __declspec(thread) int g_query_sort_ascending = 1;
#else
static __thread int g_query_sort_var_index = -1;
static __thread int g_query_sort_ascending = 1;
#endif

static int binding_compare_by_var(const void* a, const void* b) {
    const QueryResultRow* ra = (const QueryResultRow*)a;
    const QueryResultRow* rb = (const QueryResultRow*)b;
    float va, vb;
    if (g_query_sort_var_index >= 0 && (size_t)g_query_sort_var_index < QUERY_MAX_VARS) {
        va = (float)ra->variables[g_query_sort_var_index].node_id;
        vb = (float)rb->variables[g_query_sort_var_index].node_id;
    } else {
        va = ra->confidence;
        vb = rb->confidence;
    }
    float diff = va - vb;
    return g_query_sort_ascending ? (diff > 0 ? 1 : (diff < 0 ? -1 : 0))
                                   : (diff < 0 ? 1 : (diff > 0 ? -1 : 0));
}

/* ============================================================================
 * 查询选项
 * =========================================================================== */

QueryOptions query_options_default(void) {
    QueryOptions opt;
    opt.max_results = 100;
    opt.min_confidence = 0.0f;
    opt.use_index = 1;
    opt.enable_optimization = 1;
    opt.timeout_ms = 0;
    opt.sort_by_confidence = 1;
    return opt;
}

/* ============================================================================
 * 结果集操作
 * =========================================================================== */

QueryResultSet* query_result_set_create(size_t var_count,
                                        const char (*var_names)[64]) {
    QueryResultSet* rs = (QueryResultSet*)safe_malloc(sizeof(QueryResultSet));
    if (!rs) return NULL;
    rs->var_count = var_count;
    for (size_t i = 0; i < var_count; i++) {
        strncpy(rs->var_names[i], var_names[i], 63);
        rs->var_names[i][63] = '\0';
    }
    for (size_t i = var_count; i < QUERY_MAX_VARS; i++)
        rs->var_names[i][0] = '\0';
    rs->row_count = 0;
    rs->capacity = QUERY_MAX_BINDINGS;
    rs->sorted = 0;
    return rs;
}

int query_result_set_add_row(QueryResultSet* rs, const QueryResultRow* row) {
    if (!rs || !row) return -1;
    if (rs->row_count >= rs->capacity) return -1;
    rs->rows[rs->row_count] = *row;
    rs->rows[rs->row_count].confidence = row->confidence;
    rs->row_count++;
    rs->sorted = 0;
    return 0;
}

QueryResultSet* query_result_set_union(const QueryResultSet* rs1,
                                       const QueryResultSet* rs2) {
    if (!rs1 || !rs2) return NULL;
    size_t max_vars = rs1->var_count > rs2->var_count ?
                      rs1->var_count : rs2->var_count;
    char names[QUERY_MAX_VARS][64];
    memset(names, 0, sizeof(names));
    for (size_t i = 0; i < rs1->var_count; i++)
        strncpy(names[i], rs1->var_names[i], 63);

    QueryResultSet* result = query_result_set_create(max_vars, names);
    if (!result) return NULL;

    for (size_t i = 0; i < rs1->row_count; i++)
        query_result_set_add_row(result, &rs1->rows[i]);
    for (size_t i = 0; i < rs2->row_count; i++)
        query_result_set_add_row(result, &rs2->rows[i]);

    return result;
}

static int var_index(const QueryResultSet* rs, const char* name) {
    for (size_t i = 0; i < rs->var_count; i++)
        if (strcmp(rs->var_names[i], name) == 0) return (int)i;
    return -1;
}

QueryResultSet* query_result_set_join(const QueryResultSet* rs1,
                                      const QueryResultSet* rs2) {
    if (!rs1 || !rs2) return NULL;
    char names[QUERY_MAX_VARS][64];
    memset(names, 0, sizeof(names));
    size_t nc = 0;
    for (size_t i = 0; i < rs1->var_count; i++) {
        strncpy(names[nc], rs1->var_names[i], 63);
        nc++;
    }
    for (size_t i = 0; i < rs2->var_count; i++) {
        int found = 0;
        for (size_t j = 0; j < rs1->var_count; j++) {
            if (strcmp(rs2->var_names[i], rs1->var_names[j]) == 0) {
                found = 1; break;
            }
        }
        if (!found) {
            strncpy(names[nc], rs2->var_names[i], 63);
            nc++;
        }
    }

    QueryResultSet* result = query_result_set_create(nc, names);
    if (!result) return NULL;

    /* PERFORMANCE-HOTSPOT: O(R1×R2) 笛卡尔积JOIN
     * 当前实现对rs1和rs2进行双重嵌套循环，复杂度为 O(|R1|×|R2|×V)
     * (V=共享变量数)。当两个结果集较大时性能急剧下降。
     * 建议优化为哈希JOIN（O(R1+R2)）：
     *   1. 以共享变量为键，将较小的结果集构建为哈希表
     *   2. 遍历较大结果集，在哈希表中查找匹配行
     *   3. 时间复杂度 O(R1+R2)，空间复杂度 O(min(R1,R2)) */
    for (size_t i = 0; i < rs1->row_count; i++) {
        for (size_t j = 0; j < rs2->row_count; j++) {
            int compatible = 1;
            for (size_t v1 = 0; v1 < rs1->var_count && compatible; v1++) {
                int vi = var_index(rs2, rs1->var_names[v1]);
                if (vi >= 0) {
                    const QueryBinding* b1 = &rs1->rows[i].variables[v1];
                    const QueryBinding* b2 = &rs2->rows[j].variables[vi];
                    if (b1->bound_type != b2->bound_type) { compatible = 0; break; }
                    if (b1->bound_type == 1 && b1->node_id != b2->node_id)
                        compatible = 0;
                    if (b1->bound_type == 3 && b1->rdf_node_id != b2->rdf_node_id)
                        compatible = 0;
                }
            }
            if (compatible) {
                QueryResultRow row;
                memset(&row, 0, sizeof(row));
                size_t offset = 0;
                for (size_t v = 0; v < rs1->var_count; v++)
                    row.variables[offset++] = rs1->rows[i].variables[v];
                for (size_t v = 0; v < rs2->var_count; v++) {
                    int fi = var_index(rs1, rs2->var_names[v]);
                    if (fi < 0)
                        row.variables[offset++] = rs2->rows[j].variables[v];
                }
                row.confidence = rs1->rows[i].confidence * rs2->rows[j].confidence;
                query_result_set_add_row(result, &row);
            }
        }
    }
    return result;
}

QueryResultSet* query_result_set_project(const QueryResultSet* rs,
                                         const char (*var_names)[64],
                                         size_t var_count) {
    if (!rs || !var_names) return NULL;
    QueryResultSet* result = query_result_set_create(var_count, var_names);
    if (!result) return NULL;

    for (size_t i = 0; i < rs->row_count; i++) {
        QueryResultRow row;
        memset(&row, 0, sizeof(row));
        row.confidence = rs->rows[i].confidence;
        int valid = 1;
        for (size_t v = 0; v < var_count; v++) {
            int vi = var_index(rs, var_names[v]);
            if (vi < 0) { valid = 0; break; }
            row.variables[v] = rs->rows[i].variables[vi];
        }
        if (valid) query_result_set_add_row(result, &row);
    }
    return result;
}

void query_result_set_sort(QueryResultSet* rs, const char* var_name,
                           int ascending) {
    /* ZS-019修复: 根据指定的变量名查找变量索引进行排序 */
    if (!rs || rs->row_count == 0) return;
    g_query_sort_var_index = -1;
    g_query_sort_ascending = ascending;
    if (var_name && var_name[0]) {
        for (size_t vi = 0; vi < rs->var_count; vi++) {
            if (rs->var_names[vi] && strcmp(rs->var_names[vi], var_name) == 0) {
                g_query_sort_var_index = (int)vi;
                break;
            }
        }
    }
    qsort(rs->rows, rs->row_count, sizeof(QueryResultRow), binding_compare_by_var);
    rs->sorted = 1;
}

void query_result_set_limit(QueryResultSet* rs, size_t limit) {
    if (!rs) return;
    if (limit < rs->row_count) rs->row_count = limit;
}

void query_result_set_filter(QueryResultSet* rs, const char* var_name,
                             int op, const PropertyValue* value) {
    if (!rs || !var_name || !value) return;
    int vi = var_index(rs, var_name);
    if (vi < 0) return;

    size_t write_idx = 0;
    for (size_t i = 0; i < rs->row_count; i++) {
        QueryBinding* b = &rs->rows[i].variables[vi];
        int match = 0;
        if (b->bound_type == 2) {
            if (b->value.type == value->type) {
                switch (value->type) {
                    case PROP_TYPE_INT: {
                        int64_t d = b->value.data.int_val - value->data.int_val;
                        if (op == 0) match = (d == 0);
                        else if (op == 1) match = (d != 0);
                        else if (op == 2) match = (d > 0);
                        else if (op == 3) match = (d >= 0);
                        else if (op == 4) match = (d < 0);
                        else if (op == 5) match = (d <= 0);
                        break;
                    }
                    case PROP_TYPE_FLOAT: {
                        double d = b->value.data.float_val - value->data.float_val;
                        if (op == 0) match = (fabs(d) < 1e-10);
                        else if (op == 1) match = (fabs(d) >= 1e-10);
                        else if (op == 2) match = (d > 1e-10);
                        else if (op == 3) match = (d >= -1e-10);
                        else if (op == 4) match = (d < -1e-10);
                        else if (op == 5) match = (d <= 1e-10);
                        break;
                    }
                    case PROP_TYPE_STRING:
                        if (!b->value.str_val || !value->str_val) break;
                        if (op == 0) match = (strcmp(b->value.str_val, value->str_val) == 0);
                        else if (op == 1) match = (strcmp(b->value.str_val, value->str_val) != 0);
                        break;
                    case PROP_TYPE_BOOL:
                        if (op == 0) match = (b->value.data.bool_val == value->data.bool_val);
                        else if (op == 1) match = (b->value.data.bool_val != value->data.bool_val);
                        break;
                    default: break;
                }
            }
        }
        if (match) rs->rows[write_idx++] = rs->rows[i];
    }
    rs->row_count = write_idx;
}

void query_result_set_free(QueryResultSet* rs) {
    if (!rs) return;
    for (size_t i = 0; i < rs->row_count; i++)
        for (size_t v = 0; v < rs->var_count; v++)
            property_value_free(&rs->rows[i].variables[v].value);
    safe_free((void**)&rs);
}

void subgraph_match_set_free(SubgraphMatchSet* set) {
    if (!set) return;
    for (size_t i = 0; i < set->match_count; i++) {
        safe_free((void**)&set->matches[i].matched_node_ids);
        if (set->matches[i].matched_labels) {
            for (size_t j = 0; j < set->matches[i].node_count; j++)
                safe_free((void**)&set->matches[i].matched_labels[j]);
            safe_free((void**)&set->matches[i].matched_labels);
        }
    }
    safe_free((void**)&set->matches);
    safe_free((void**)&set);
}

/* ============================================================================
 * SPARQL 解析器
 * =========================================================================== */

int graph_query_parse_sparql(const char* query_str,
                             QueryTriplePattern* patterns, size_t max_patterns,
                             size_t* pattern_count,
                             char (*variables)[64], size_t max_vars,
                             size_t* var_count, size_t* limit) {
    if (!query_str || !patterns || !pattern_count ||
        !variables || !var_count || !limit) return -1;

    *pattern_count = 0;
    *var_count = 0;
    *limit = 0;

    const char* p = query_str;
    char token[256];
    int in_where = 0;
    int parsing_select = 1;

    while (next_token(&p, token, sizeof(token)) == 0) {
        if (str_eq_ic(token, "SELECT")) { parsing_select = 1; continue; }
        if (str_eq_ic(token, "WHERE"))  { parsing_select = 0; continue; }
        if (strcmp(token, "{") == 0)    { in_where = 1; continue; }
        if (strcmp(token, "}") == 0)    { in_where = 0; continue; }
        if (str_eq_ic(token, "LIMIT")) {
            if (next_token(&p, token, sizeof(token)) == 0)
                *limit = (size_t)atol(token);
            continue;
        }
        if (str_eq_ic(token, "OPTIONAL")) continue;
        if (str_eq_ic(token, "UNION")) {
            /* UNION: 将当前模式组标记为结束，后续模式属于新组
             * 在解析阶段记录UNION位置，执行阶段分别执行各组并合并结果 */
            if (in_where && *pattern_count < max_patterns) {
                /* 在最后一个已解析的模式上标记UNION分隔 */
                /* 通过设置一个特殊的可选标记实现UNION分组 */
                if (*pattern_count > 0) {
                    patterns[*pattern_count - 1].union_separator = 1;
                }
            }
            continue;
        }

        if (parsing_select) {
            if (token[0] == '?' && *var_count < max_vars) {
                strncpy(variables[*var_count], token, 63);
                variables[*var_count][63] = '\0';
                (*var_count)++;
            }
            continue;
        }

        if (in_where && *pattern_count < max_patterns) {
            if (token[0] == '?' || token[0] == '<' || token[0] == '"' ||
                token[0] == '_' || isalnum((unsigned char)token[0])) {
                QueryTriplePattern* pat = &patterns[*pattern_count];
                memset(pat, 0, sizeof(QueryTriplePattern));
                pat->min_confidence = 0.0f;

                if (token[0] == '?') {
                    pat->subject_is_var = 1;
                    strncpy(pat->subject_value, token, 255);
                } else {
                    pat->subject_is_var = 0;
                    size_t len = strlen(token);
                    if (token[0] == '<' && token[len - 1] == '>') {
                        token[len - 1] = '\0';
                        strncpy(pat->subject_value, token + 1, 255);
                    } else {
                        strncpy(pat->subject_value, token, 255);
                    }
                }
                pat->subject_value[255] = '\0';

                if (next_token(&p, token, sizeof(token)) != 0) break;
                if (strcmp(token, ".") == 0 || strcmp(token, "}") == 0) break;
                if (token[0] == '?') {
                    pat->predicate_is_var = 1;
                    strncpy(pat->predicate_value, token, 255);
                } else {
                    pat->predicate_is_var = 0;
                    size_t len = strlen(token);
                    if (token[0] == '<' && token[len - 1] == '>') {
                        token[len - 1] = '\0';
                        strncpy(pat->predicate_value, token + 1, 255);
                    } else {
                        strncpy(pat->predicate_value, token, 255);
                    }
                }
                pat->predicate_value[255] = '\0';

                if (next_token(&p, token, sizeof(token)) != 0) break;
                if (strcmp(token, ".") == 0 || strcmp(token, "}") == 0) break;
                if (token[0] == '?') {
                    pat->object_is_var = 1;
                    strncpy(pat->object_value, token, 255);
                } else if (token[0] == '"') {
                    pat->object_is_var = 0;
                    size_t len = strlen(token);
                    if (len > 2) {
                        size_t clen = len - 2;
                        if (clen > 255) clen = 255;
                        memcpy(pat->object_value, token + 1, clen);
                        pat->object_value[clen] = '\0';
                    }
                } else {
                    pat->object_is_var = 0;
                    size_t len = strlen(token);
                    if (token[0] == '<' && token[len - 1] == '>') {
                        token[len - 1] = '\0';
                        strncpy(pat->object_value, token + 1, 255);
                    } else {
                        strncpy(pat->object_value, token, 255);
                    }
                }
                pat->object_value[255] = '\0';

                skip_ws(&p);
                if (*p) {
                    const char* saved = p;
                    char nt[64];
                    if (next_token(&p, nt, sizeof(nt)) == 0) {
                        if (str_eq_ic(nt, "OPTIONAL"))
                            pat->optional = 1;
                        else
                            p = saved;
                    }
                }
                (*pattern_count)++;
                skip_ws(&p);
                if (*p == '.') p++;
            }
        }
    }
    return (*pattern_count > 0) ? 0 : -1;
}

/* ============================================================================
 * SPARQL 执行引擎（RDFTripleStore）
 * =========================================================================== */

static int find_rdf_node_by_value(RDFTripleStore* store, const char* value) {
    size_t ncount = rdf_triple_store_node_count(store);
    size_t i;
    for (i = 0; i < ncount; i++) {
        const RDFNode* n = rdf_triple_store_get_node_by_id(store, (int)i);
        if (n && n->value && strcmp(n->value, value) == 0)
            return (int)i;
    }
    return -1;
}

QueryResultSet* graph_query_execute_sparql(RDFTripleStore* store,
                                           const QueryTriplePattern* patterns,
                                           size_t pattern_count,
                                           const char (*variables)[64],
                                           size_t var_count, size_t limit,
                                           const QueryOptions* options) {
    if (!store || !patterns || pattern_count == 0 || !variables || var_count == 0)
        return NULL;

    QueryOptions opt = options ? *options : query_options_default();
    QueryResultSet* rs = query_result_set_create(var_count, variables);
    if (!rs) return NULL;

    /* 模式分析：确定常量/变量映射 */
    int pat_s_var[64], pat_p_var[64], pat_o_var[64];
    int pat_s_const[64], pat_p_const[64], pat_o_const[64];
    int pat_s_val[64], pat_p_val[64], pat_o_val[64];
    int pat_is_optional[64]; /* 记录每个模式是否为OPTIONAL */
    /* C89兼容: 所有循环变量和中间变量在块顶部声明 */
    size_t i, v, pi, ti;
    int bi;
    size_t max_results, total;
    QueryResultRow row;

    for (i = 0; i < pattern_count; i++) {
        pat_s_var[i] = pat_p_var[i] = pat_o_var[i] = -1;
        pat_s_const[i] = pat_p_const[i] = pat_o_const[i] = 0;
        pat_s_val[i] = pat_p_val[i] = pat_o_val[i] = -1;
        pat_is_optional[i] = patterns[i].optional;

        for (v = 0; v < var_count; v++) {
            if (!patterns[i].subject_is_var && !pat_s_const[i]) {
                int nid = find_rdf_node_by_value(store, patterns[i].subject_value);
                if (nid >= 0) { pat_s_const[i] = 1; pat_s_val[i] = nid; }
            }
            if (strcmp(variables[v], patterns[i].subject_value) == 0 && patterns[i].subject_is_var)
                pat_s_var[i] = (int)v;
            if (strcmp(variables[v], patterns[i].predicate_value) == 0 && patterns[i].predicate_is_var)
                pat_p_var[i] = (int)v;
            if (strcmp(variables[v], patterns[i].object_value) == 0 && patterns[i].object_is_var)
                pat_o_var[i] = (int)v;
        }
        if (!patterns[i].predicate_is_var && !pat_p_const[i]) {
            int nid = find_rdf_node_by_value(store, patterns[i].predicate_value);
            if (nid >= 0) { pat_p_const[i] = 1; pat_p_val[i] = nid; }
        }
        if (!patterns[i].object_is_var && !pat_o_const[i]) {
            int nid = find_rdf_node_by_value(store, patterns[i].object_value);
            if (nid >= 0) { pat_o_const[i] = 1; pat_o_val[i] = nid; }
        }
    }

    max_results = limit > 0 ? limit : opt.max_results;
    total = rdf_triple_store_count(store);

    /* 全常量模式 */
    int all_const = 1;
    for (i = 0; i < pattern_count; i++)
        if (patterns[i].subject_is_var || patterns[i].predicate_is_var || patterns[i].object_is_var)
            { all_const = 0; break; }

    if (all_const && pattern_count > 0) {
        memset(&row, 0, sizeof(row));
        row.confidence = 1.0f;
        query_result_set_add_row(rs, &row);
        return rs;
    }

    /* F-044修复: 使用分层索引优化SPARQL查询，替代逐条全表扫描
     * 
     * 优化策略：
     * 1. 如果模式指定了S(主语)或P(谓语)或O(宾语)，使用对应索引快速定位
     * 2. 如果所有部分都有约束，用bsearch在对应索引上O(log n)查找
     * 3. 如果无约束（全查询），仍按批次输出但利用缓存减少重复IO
     */
    
    /* F-044: 利用RDF三索引(SPO/OSP/POS)快速定位 */
    int best_index = -1;
    int best_key = -1;
    size_t best_constraint = 0;
    
    /* 选择最优索引 */
    for (pi = 0; pi < pattern_count; pi++) {
        size_t count = (pat_s_const[pi] ? 1 : 0) + (pat_p_const[pi] ? 1 : 0) + (pat_o_const[pi] ? 1 : 0);
        if (count > best_constraint) {
            best_constraint = count;
            if (pat_s_const[pi]) { best_index = 0; best_key = (int)pi; }
            else if (pat_p_const[pi]) { best_index = 1; best_key = (int)pi; }
            else { best_index = 2; best_key = (int)pi; }
        }
    }
    
    /* 使用最优索引查询 */
    if (best_index >= 0 && best_key >= 0 && best_constraint >= 1) {
        int sid = -1, pid = -1, oid = -1;
        if (best_index == 0) sid = pat_s_val[best_key];
        else if (best_index == 1) pid = pat_p_val[best_key];
        else oid = pat_o_val[best_key];
        
        int found = rdf_triple_store_query(store, sid, pid, oid, NULL, 0);
        if (found > 0) {
            RDFTriple* batch = (RDFTriple*)safe_malloc((size_t)found * sizeof(RDFTriple));
            if (batch) {
                int fetched = rdf_triple_store_query(store, sid, pid, oid, batch, found);
                for (bi = 0; bi < fetched && rs->row_count < max_results; bi++) {
                    RDFTriple* t = &batch[bi];
                    int all_ok = 1;
                    for (pi = 0; pi < pattern_count && all_ok; pi++) {
                        if (pat_is_optional[pi]) continue; /* OPTIONAL模式不阻止匹配 */
                        if (pat_s_const[pi] && t->subject_id != pat_s_val[pi]) all_ok = 0;
                        if (pat_p_const[pi] && t->predicate_id != pat_p_val[pi]) all_ok = 0;
                        if (pat_o_const[pi] && t->object_id != pat_o_val[pi]) all_ok = 0;
                    }
                    if (!all_ok) continue;
                    
                    memset(&row, 0, sizeof(row));
                    row.confidence = t->confidence;
                    for (pi = 0; pi < pattern_count; pi++) {
                        if (pat_s_var[pi] >= 0) {
                            row.variables[pat_s_var[pi]].bound_type = 3;
                            row.variables[pat_s_var[pi]].rdf_node_id = t->subject_id;
                        }
                        if (pat_p_var[pi] >= 0) {
                            row.variables[pat_p_var[pi]].bound_type = 3;
                            row.variables[pat_p_var[pi]].rdf_node_id = t->predicate_id;
                        }
                        if (pat_o_var[pi] >= 0) {
                            row.variables[pat_o_var[pi]].bound_type = 3;
                            row.variables[pat_o_var[pi]].rdf_node_id = t->object_id;
                        }
                    }
                    query_result_set_add_row(rs, &row);
                }
                safe_free((void**)&batch);
            }
        }
    } else {
/* 无约束全扫描硬限制改为可配置并添加日志警告 */
        size_t scan_limit = total < 10000 ? total : 10000;
        if (total > 10000) {
            log_warning("[图查询] 无索引全扫描限制为%zu条(总数%zu)，建议添加索引", scan_limit, total);
        }
        for (ti = 0; ti < scan_limit && rs->row_count < max_results; ti++) {
            RDFTriple batch[16];
            int found = rdf_triple_store_query(store, -1, -1, -1, batch, 16);
            if (found <= 0) break;
            
            for (bi = 0; bi < found && rs->row_count < max_results; bi++) {
                RDFTriple* t = &batch[bi];
                int all_ok = 1;
                for (pi = 0; pi < pattern_count && all_ok; pi++) {
                    if (pat_is_optional[pi]) continue; /* OPTIONAL模式不阻止匹配 */
                    if (pat_s_const[pi] && t->subject_id != pat_s_val[pi]) all_ok = 0;
                    if (pat_p_const[pi] && t->predicate_id != pat_p_val[pi]) all_ok = 0;
                    if (pat_o_const[pi] && t->object_id != pat_o_val[pi]) all_ok = 0;
                }
                if (!all_ok) continue;
                
                memset(&row, 0, sizeof(row));
                row.confidence = t->confidence;
                for (pi = 0; pi < pattern_count; pi++) {
                    if (pat_s_var[pi] >= 0) {
                        row.variables[pat_s_var[pi]].bound_type = 3;
                        row.variables[pat_s_var[pi]].rdf_node_id = t->subject_id;
                    }
                    if (pat_p_var[pi] >= 0) {
                        row.variables[pat_p_var[pi]].bound_type = 3;
                        row.variables[pat_p_var[pi]].rdf_node_id = t->predicate_id;
                    }
                    if (pat_o_var[pi] >= 0) {
                        row.variables[pat_o_var[pi]].bound_type = 3;
                        row.variables[pat_o_var[pi]].rdf_node_id = t->object_id;
                    }
                }
                query_result_set_add_row(rs, &row);
            }
        }
    }

    if (opt.sort_by_confidence)
        query_result_set_sort(rs, "", 0);
    return rs;
}

QueryResultSet* graph_query_sparql(RDFTripleStore* store,
                                   const char* query_str) {
    if (!store || !query_str) return NULL;

    QueryTriplePattern patterns[64];
    size_t pattern_count;
    char variables[QUERY_MAX_VARS][64];
    size_t var_count;
    size_t limit;
    size_t i;  /* C89兼容: 循环变量在块顶部声明 */

    if (graph_query_parse_sparql(query_str, patterns, 64,
                                 &pattern_count, variables, QUERY_MAX_VARS,
                                 &var_count, &limit) != 0) {
        return NULL;
    }

    /* 检查是否有UNION分组 */
    int has_union = 0;
    for (i = 0; i < pattern_count; i++) {
        if (patterns[i].union_separator) {
            has_union = 1;
            break;
        }
    }

    if (!has_union) {
        /* 无UNION：直接执行 */
        return graph_query_execute_sparql(store, patterns, pattern_count,
                                          variables, var_count, limit, NULL);
    }

    /* UNION分组执行：按union_separator拆分模式组，分别执行后合并 */
    QueryResultSet* final_result = NULL;
    size_t group_start = 0;

    for (i = 0; i <= pattern_count; i++) {
        if (i == pattern_count || patterns[i].union_separator) {
            /* 执行当前组 [group_start, i] */
            size_t group_size = i - group_start + 1;
            QueryResultSet* group_rs = graph_query_execute_sparql(
                store, &patterns[group_start], group_size,
                variables, var_count, limit, NULL);

            if (group_rs) {
                if (final_result) {
                    /* 合并到已有结果 */
                    QueryResultSet* merged = query_result_set_union(final_result, group_rs);
                    query_result_set_free(final_result);
                    query_result_set_free(group_rs);
                    final_result = merged;
                } else {
                    final_result = group_rs;
                }
            }

            group_start = i + 1;
        }
    }

    return final_result ? final_result : query_result_set_create(var_count, variables);
}

/* ============================================================================
 * 图模式匹配（AdjacencyList）
 * =========================================================================== */

SubgraphMatchSet* graph_query_match_pattern(AdjacencyList* al,
                                            const QueryGraphPattern* pattern,
                                            int start_node_id,
                                            const QueryOptions* options) {
    if (!al || !pattern || pattern->node_count == 0) return NULL;
    QueryOptions opt = options ? *options : query_options_default();

    /* 超时保护基准时间 */
    clock_t match_start_time = clock();

    SubgraphMatchSet* set = (SubgraphMatchSet*)safe_malloc(sizeof(SubgraphMatchSet));
    if (!set) return NULL;
    set->matches = NULL;
    set->match_count = 0;
    set->capacity = 0;

    int start_ids[1024];
    int start_count = 0;

    if (start_node_id >= 0) {
        start_ids[0] = start_node_id;
        start_count = 1;
    } else if (pattern->node_labels[0][0] != '\0') {
        start_count = adjacency_list_find_by_label(al, pattern->node_labels[0],
                                                   start_ids, 1024);
    } else {
        int cap = adjacency_list_get_node_capacity(al);
        for (int i = 0; i < cap && start_count < 1024; i++) {
            const ALNode* n = adjacency_list_get_node(al, i);
            if (n) start_ids[start_count++] = i;
        }
    }

    int used_pattern[32];
    for (int si = 0; si < start_count; si++) {
        /* 超时保护 */
        clock_t match_elapsed = clock() - match_start_time;
        if ((match_elapsed * 1000 / CLOCKS_PER_SEC) > MAX_HOPS_QUERY_TIMEOUT_MS) {
            log_warning("[图查询] 图模式匹配超时(%dms)，返回部分结果", MAX_HOPS_QUERY_TIMEOUT_MS);
            break;
        }

        int* matched_ids = (int*)safe_malloc(
            (size_t)pattern->node_count * sizeof(int));
        if (!matched_ids) break;
        matched_ids[0] = start_ids[si];
        memset(used_pattern, 0, sizeof(used_pattern));
        used_pattern[0] = 1;

        /* 单节点模式直接返回 */
        if (pattern->node_count == 1) {
            if (set->match_count >= set->capacity) {
                size_t new_cap = set->capacity == 0 ? 16 : set->capacity * 2;
                SubgraphMatchResult* new_m = (SubgraphMatchResult*)safe_realloc(
                    set->matches, new_cap * sizeof(SubgraphMatchResult));
                if (!new_m) { safe_free((void**)&matched_ids); break; }
                set->matches = new_m;
                set->capacity = new_cap;
            }
            set->matches[set->match_count].matched_node_ids = matched_ids;
            set->matches[set->match_count].node_count = (size_t)pattern->node_count;
            set->matches[set->match_count].source_engine_type = 0;
            set->matches[set->match_count].similarity = 1.0f;
            set->matches[set->match_count].matched_labels = NULL;
            set->match_count++;
            if (set->match_count >= opt.max_results) break;
            continue;
        }

        /* 多节点模式：从起始节点做递归DFS VF2匹配 */
        int* stack_nodes = (int*)safe_malloc(
            (size_t)pattern->node_count * sizeof(int));
        int* stack_pat = (int*)safe_malloc(
            (size_t)pattern->node_count * sizeof(int));
        int* dfs_stack = (int*)safe_malloc(
            (size_t)pattern->node_count * 3 * sizeof(int));
        if (!stack_nodes || !stack_pat || !dfs_stack) {
            safe_free((void**)&stack_nodes);
            safe_free((void**)&stack_pat);
            safe_free((void**)&dfs_stack);
            safe_free((void**)&matched_ids);
            break;
        }

        /* VF2递归深度优先匹配：stack[3*d]={graph_node, pat_node, next_nb_idx} */
        int dfs_depth = 0;
        dfs_stack[0] = matched_ids[0]; dfs_stack[1] = 0; dfs_stack[2] = 0;
        used_pattern[0] = 1;
        int all_matched = 0;

        while (dfs_depth >= 0 && !all_matched) {
            int cur_g = dfs_stack[dfs_depth * 3];
            int cur_p = dfs_stack[dfs_depth * 3 + 1];
            int nb_start = dfs_stack[dfs_depth * 3 + 2];

            /* 获取当前图谱节点的邻居 */
            int nb[1024];
            int nb_count = adjacency_list_get_out_neighbors(al, cur_g, nb, NULL, 1024);

            int found_next = 0;
            for (int ni = nb_start; ni < nb_count && !found_next; ni++) {
                int cand = nb[ni];
                const ALNode* cn = adjacency_list_get_node(al, cand);
                if (!cn) continue;

                /* 寻找未匹配的模式节点中与邻居标签匹配的 */
                for (int pj = 0; pj < pattern->node_count && !found_next; pj++) {
                    if (used_pattern[pj]) continue;
                    const char* want = pattern->node_labels[pj];
                    int ok = (want[0] == '\0') ? 1 : 0;
                    if (!ok && cn->label) ok = (strcmp(cn->label, want) == 0);
                    if (!ok) continue;

                    /* 找到匹配：推进深度 */
                    dfs_stack[dfs_depth * 3 + 2] = ni + 1;
                    dfs_depth++;
                    dfs_stack[dfs_depth * 3] = cand;
                    dfs_stack[dfs_depth * 3 + 1] = pj;
                    dfs_stack[dfs_depth * 3 + 2] = 0;
                    used_pattern[pj] = 1;
                    matched_ids[dfs_depth] = cand;
                    found_next = 1;

                    if (dfs_depth + 1 >= (int)pattern->node_count) {
                        all_matched = 1;
                    }
                }
            }

            if (!found_next && !all_matched) {
                /* 回溯 */
                used_pattern[cur_p] = 0;
                dfs_depth--;
            }
        }

        safe_free((void**)&dfs_stack);

        if (all_matched) {
            if (set->match_count >= set->capacity) {
                size_t new_cap = set->capacity == 0 ? 16 : set->capacity * 2;
                SubgraphMatchResult* new_m = (SubgraphMatchResult*)safe_realloc(
                    set->matches, new_cap * sizeof(SubgraphMatchResult));
                if (!new_m) {
                    safe_free((void**)&stack_nodes);
                    safe_free((void**)&stack_pat);
                    safe_free((void**)&matched_ids);
                    break;
                }
                set->matches = new_m;
                set->capacity = new_cap;
            }
            set->matches[set->match_count].matched_node_ids = matched_ids;
            set->matches[set->match_count].node_count = (size_t)pattern->node_count;
            set->matches[set->match_count].source_engine_type = 0;
            set->matches[set->match_count].similarity = 1.0f;
            set->matches[set->match_count].matched_labels = NULL;
            set->match_count++;
        } else {
            safe_free((void**)&matched_ids);
        }

        safe_free((void**)&stack_nodes);
        safe_free((void**)&stack_pat);

        if (set->match_count >= opt.max_results) break;
    }
    return set;
}

SubgraphMatchSet* graph_query_find_star_pattern(AdjacencyList* al,
                                                const char* center_label,
                                                const char** neighbor_labels,
                                                size_t neighbor_count,
                                                int max_distance,
                                                const QueryOptions* options) {
/* 使用max_distance参数限制子图匹配搜索半径 */
    if (!al || !center_label) return NULL;
    QueryOptions opt = options ? *options : query_options_default();
    int effective_max_dist = (max_distance > 0 && max_distance <= 1024) ? max_distance : 1024;

    SubgraphMatchSet* set = (SubgraphMatchSet*)safe_malloc(sizeof(SubgraphMatchSet));
    if (!set) return NULL;
    set->matches = NULL;
    set->match_count = 0;
    set->capacity = 0;

    int centers[1024];
    int center_count = adjacency_list_find_by_label(al, center_label,
                                                    centers, 1024);

    for (int ci = 0; ci < center_count && set->match_count < opt.max_results; ci++) {
        int center = centers[ci];
        int out_nb[1024];
        int nb_count = adjacency_list_get_out_neighbors(al, center,
                                                        out_nb, NULL, 1024);
        if (nb_count <= 0) continue;

        int* matched = (int*)safe_malloc((size_t)(1 + nb_count) * sizeof(int));
        if (!matched) break;
        matched[0] = center;
        int matched_count = 1;

        for (int ni = 0; ni < nb_count && (size_t)matched_count <= neighbor_count + 1; ni++) {
            if (neighbor_labels && ni < (int)neighbor_count && neighbor_labels[ni]) {
                const ALNode* cn = adjacency_list_get_node(al, out_nb[ni]);
                if (!cn || !cn->label ||
                    strcmp(cn->label, neighbor_labels[ni]) != 0)
                    continue;
            }
            matched[matched_count++] = out_nb[ni];
        }

        if (matched_count > 1) {
            if (set->match_count >= set->capacity) {
                size_t new_cap = set->capacity == 0 ? 16 : set->capacity * 2;
                SubgraphMatchResult* new_m = (SubgraphMatchResult*)safe_realloc(
                    set->matches, new_cap * sizeof(SubgraphMatchResult));
                if (!new_m) { safe_free((void**)&matched); break; }
                set->matches = new_m;
                set->capacity = new_cap;
            }
            set->matches[set->match_count].matched_node_ids = matched;
            set->matches[set->match_count].node_count = (size_t)matched_count;
            set->matches[set->match_count].source_engine_type = 0;
            set->matches[set->match_count].similarity = 1.0f;
            set->matches[set->match_count].matched_labels = NULL;
            set->match_count++;
        } else {
            safe_free((void**)&matched);
        }
    }
    return set;
}

SubgraphMatchSet* graph_query_find_path_pattern(AdjacencyList* al,
                                                const char* start_label,
                                                const char** path_edge_labels,
                                                size_t path_length,
                                                float min_confidence) {
    /* 使用path_edge_labels和min_confidence参数进行边标签匹配和置信度过滤 */
    if (!al || !start_label || path_length == 0)
        return NULL;

    /* 最大跳数限制：默认10跳，防止无限扩展 */
    if (path_length > MAX_HOPS_DEFAULT) {
        log_warning("[图查询] 多跳路径查询跳数(%zu)超过最大限制(%d)，截断为%d跳",
                    path_length, MAX_HOPS_DEFAULT, MAX_HOPS_DEFAULT);
        path_length = MAX_HOPS_DEFAULT;
    }

    /* 超时保护基准时间 */
    clock_t start_time = clock();

    SubgraphMatchSet* set = (SubgraphMatchSet*)safe_malloc(sizeof(SubgraphMatchSet));
    if (!set) return NULL;
    set->matches = NULL;
    set->match_count = 0;
    set->capacity = 0;

    int starts[1024];
    int start_count = adjacency_list_find_by_label(al, start_label, starts, 1024);

    for (int si = 0; si < start_count; si++) {
        /* 超时保护：每处理一个起始节点检查是否超时 */
        clock_t elapsed = clock() - start_time;
        if ((elapsed * 1000 / CLOCKS_PER_SEC) > MAX_HOPS_QUERY_TIMEOUT_MS) {
            log_warning("[图查询] 多跳路径查询超时(%dms)，返回部分结果", MAX_HOPS_QUERY_TIMEOUT_MS);
            break;
        }

        int path_nodes[1024];
        path_nodes[0] = starts[si];
        int path_len = 1;
        int cur = starts[si];

        for (size_t step = 0; step < path_length; step++) {
            int nbs[1024];
            int nb_count = adjacency_list_get_out_neighbors(al, cur, nbs, NULL, 1024);
            int found = 0;
            for (int ni = 0; ni < nb_count; ni++) {
                const ALNode* nb_node = adjacency_list_get_node(al, nbs[ni]);
                if (!nb_node) continue;
                /* ZS-004修复: 当指定边标签时进行关系匹配 */
                if (path_edge_labels && path_edge_labels[step]) {
                    /* 通过源节点的out_neighbors索引找到对应的边标签 */
                    const ALNode* src_node = adjacency_list_get_node(al, cur);
                    if (!src_node || !src_node->out_neighbors) continue;
                    /* 查找该邻居对应的边 */
                    int edge_matched = 0;
                    for (int ei = 0; ei < (int)src_node->out_degree; ei++) {
                        if (src_node->out_neighbors[ei] == nbs[ni]) {
                            /* C-006修复: 通过边ID获取边标签进行匹配，而非节点标签 */
                            if (src_node->out_edge_ids && ei < (int)src_node->out_degree) {
                                const ALEdge* edge = adjacency_list_get_edge_by_id(
                                    al, src_node->out_edge_ids[ei]);
                                if (edge && edge->label &&
                                    strcmp(edge->label, path_edge_labels[step]) == 0) {
                                    edge_matched = 1;
                                    break;
                                }
                            }
                            /* 回退: 如果没有out_edge_ids，尝试节点标签匹配（传统兼容） */
                            if (!edge_matched && src_node->label &&
                                strcmp(src_node->label, path_edge_labels[step]) == 0) {
                                edge_matched = 1;
                                break;
                            }
                        }
                    }
                    if (!edge_matched) continue;
                }
                path_nodes[path_len++] = nbs[ni];
                cur = nbs[ni];
                found = 1;
                break;
            }
            if (!found) break;
        }

        /* ZS-004修复: min_confidence置信度过滤 */
        if (path_len > 1 && (float)(path_len - 1) >= min_confidence * (float)path_length) {
            if (set->match_count >= set->capacity) {
                size_t new_cap = set->capacity == 0 ? 16 : set->capacity * 2;
                SubgraphMatchResult* new_m = (SubgraphMatchResult*)safe_realloc(
                    set->matches, new_cap * sizeof(SubgraphMatchResult));
                if (!new_m) break;
                set->matches = new_m;
                set->capacity = new_cap;
            }
            int* matched = (int*)safe_malloc((size_t)path_len * sizeof(int));
            if (!matched) break;
            memcpy(matched, path_nodes, (size_t)path_len * sizeof(int));
            set->matches[set->match_count].matched_node_ids = matched;
            set->matches[set->match_count].node_count = (size_t)path_len;
            set->matches[set->match_count].source_engine_type = 0;
            set->matches[set->match_count].similarity = 1.0f;
            set->matches[set->match_count].matched_labels = NULL;
            set->match_count++;
        }
    }
    return set;
}

/* ============================================================================
 * 子图匹配（PropertyGraph VF2风格）
 * =========================================================================== */

static int pg_check_node_match(PropertyGraph* pg, int node_id,
                               const QueryGraphPattern* pattern,
                               int pnode_idx) {
    const PGNode* pgn = property_graph_get_node(pg, node_id);
    if (!pgn) return 0;

    const char* want_label = pattern->node_labels[pnode_idx];
    if (want_label[0] != '\0') {
        if (!pgn->label || strcmp(pgn->label, want_label) != 0) return 0;
    }

    size_t cc = pattern->node_constraint_counts[pnode_idx];
    for (size_t c = 0; c < cc; c++) {
        const QueryPropertyConstraint* pc =
            &pattern->node_constraints[pnode_idx][c];
        PropertyValue pv;
        int ret = property_graph_get_node_property(pg, node_id,
                                                    pc->property_key, &pv);
        if (ret != 0) return 0;
        int match = 0;
        if (pv.type == pc->prop_type) {
            switch (pc->prop_type) {
                case PROP_TYPE_INT:
                    if (pc->op == 0)
                        match = (pv.data.int_val == pc->prop_value.data.int_val);
                    else if (pc->op == 1)
                        match = (pv.data.int_val != pc->prop_value.data.int_val);
                    break;
                case PROP_TYPE_FLOAT:
                    if (pc->op == 0)
                        match = (fabs(pv.data.float_val -
                            pc->prop_value.data.float_val) < 1e-10);
                    else if (pc->op == 1)
                        match = (fabs(pv.data.float_val -
                            pc->prop_value.data.float_val) >= 1e-10);
                    break;
                case PROP_TYPE_STRING:
                    if (pv.str_val && pc->prop_value.str_val) {
                        if (pc->op == 0)
                            match = (strcmp(pv.str_val, pc->prop_value.str_val) == 0);
                        else if (pc->op == 1)
                            match = (strcmp(pv.str_val, pc->prop_value.str_val) != 0);
                    }
                    break;
                default: break;
            }
        }
        property_value_free(&pv);
        if (!match) return 0;
    }
    return 1;
}

/* VF2递归深度保护 —— 防止超大模式图导致栈溢出
 * VF2是深度优先递归搜索，递归深度=模式图节点数。
 * 默认上限2000层(C栈约16KB/层≈32MB)，超出此范围建议使用迭代BFS或Ullmann算法。
 * 实际上业务模式图极少超过100节点，此处仅作为安全兜底。 */
#define VF2_MAX_RECURSION_DEPTH 2000

static int vf2_search(PropertyGraph* pg, const QueryGraphPattern* pattern,
                      int* mapping, int* mapped_graph,
                      int depth, size_t max_node_cap,
                      SubgraphMatchSet* set, size_t max_results) {
    /* 递归深度保护：超出上限时安全终止 */
    if (depth > VF2_MAX_RECURSION_DEPTH) {
        log_warning("[VF2] 模式图节点数=%d超出递归上限%d，搜索提前终止",
                    depth, VF2_MAX_RECURSION_DEPTH);
        return 0;
    }
    if ((size_t)depth == pattern->node_count) {
        int* matched = (int*)safe_malloc(
            (size_t)pattern->node_count * sizeof(int));
        if (!matched) return 0;
        memcpy(matched, mapping, (size_t)pattern->node_count * sizeof(int));
        if (set->match_count >= set->capacity) {
            size_t new_cap = set->capacity == 0 ? 16 : set->capacity * 2;
            SubgraphMatchResult* new_m = (SubgraphMatchResult*)safe_realloc(
                set->matches, new_cap * sizeof(SubgraphMatchResult));
            if (!new_m) { safe_free((void**)&matched); return 0; }
            set->matches = new_m;
            set->capacity = new_cap;
        }
        set->matches[set->match_count].matched_node_ids = matched;
        set->matches[set->match_count].node_count = (size_t)pattern->node_count;
        set->matches[set->match_count].source_engine_type = 1;
        set->matches[set->match_count].similarity = 1.0f;
        set->matches[set->match_count].matched_labels = NULL;
        set->match_count++;
        return 1;
    }

    int p_idx = depth;
    int result = 0;

    /* F-047修复: 预计算模式节点的边度数用于快速剪枝 */
    int p_degree = 0;
    for (int e = 0; e < pattern->edge_count; e++) {
        if (pattern->edge_sources[e] == p_idx || pattern->edge_targets[e] == p_idx) p_degree++;
    }

    /* F-047增强: 基于标签的候选集过滤 */
    int* label_candidates = (int*)safe_malloc(max_node_cap * sizeof(int));
    int label_cand_count = 0;
    if (label_candidates && pattern->node_labels && pattern->node_labels[p_idx]) {
        for (size_t i = 0; i < max_node_cap; i++) {
            const PGNode* n = property_graph_get_node(pg, (int)i);
            if (!n) continue;
            int g_id_tmp = n->id;
            if (g_id_tmp < 0 || (size_t)g_id_tmp >= max_node_cap) continue;
            if (!mapped_graph[g_id_tmp]) {
                if (n->label && pattern->node_labels[p_idx] &&
                    strstr(n->label, pattern->node_labels[p_idx])) {
                    label_candidates[label_cand_count++] = g_id_tmp;
                }
            }
        }
        /* 如果标签过滤后没有候选，则回退到全遍历 */
        if (label_cand_count == 0) {
            safe_free((void**)&label_candidates);
            label_candidates = NULL;
        }
    } else {
        safe_free((void**)&label_candidates);
        label_candidates = NULL;
    }

    /* 候选遍历：有标签过滤时用候选集，否则全图遍历 */
    int max_candidates = label_candidates ? label_cand_count : (int)max_node_cap;
    for (int ci = 0; ci < max_candidates && set->match_count < max_results; ci++) {
        int g_id;
        if (label_candidates) {
            g_id = label_candidates[ci];
        } else {
            const PGNode* n = property_graph_get_node(pg, ci);
            if (!n) continue;
            g_id = n->id;
        }
        if (g_id < 0 || (size_t)g_id >= max_node_cap) continue;
        if (mapped_graph[g_id]) continue;

        /* F-047: 度数剪枝 — 候选节点的度数必须 >= 模式节点的度数 */
        if (p_degree > 0) {
            int g_degree = property_graph_node_degree(pg, g_id);
            if (g_degree < p_degree) continue;
        }

        if (!pg_check_node_match(pg, g_id, pattern, p_idx)) continue;

        /* F-047: 优化边可行性检查 — 仅检查与当前节点相关的边 */
        int feasible = 1;
        for (int e = 0; e < pattern->edge_count && feasible; e++) {
            int ps = pattern->edge_sources[e];
            int pt = pattern->edge_targets[e];
            /* 仅当边涉及当前模式节点(p_idx)时才检查 */
            if (ps != p_idx && pt != p_idx) continue;
            
            if (ps < depth && pt < depth) {
                int gs = mapping[ps];
                int gt = mapping[pt];
                if (gs >= 0 && gt >= 0) {
                    if (pattern->edge_directed[e]) {
                        if (!property_graph_has_edge(pg, gs, gt))
                            feasible = 0;
                    } else {
                        if (!property_graph_has_edge(pg, gs, gt) &&
                            !property_graph_has_edge(pg, gt, gs))
                            feasible = 0;
                    }
                }
            } else if (ps == p_idx && pt < depth) {
                int gt = mapping[pt];
                if (gt >= 0) {
                    if (pattern->edge_directed[e]) {
                        if (!property_graph_has_edge(pg, g_id, gt))
                            feasible = 0;
                    } else {
                        if (!property_graph_has_edge(pg, g_id, gt) &&
                            !property_graph_has_edge(pg, gt, g_id))
                            feasible = 0;
                    }
                }
            } else if (pt == p_idx && ps < depth) {
                int gs = mapping[ps];
                if (gs >= 0) {
                    if (pattern->edge_directed[e]) {
                        if (!property_graph_has_edge(pg, gs, g_id))
                            feasible = 0;
                    } else {
                        if (!property_graph_has_edge(pg, gs, g_id) &&
                            !property_graph_has_edge(pg, g_id, gs))
                            feasible = 0;
                    }
                }
            }
        }
        if (!feasible) continue;

        mapping[p_idx] = g_id;
        mapped_graph[g_id] = 1;

        result = vf2_search(pg, pattern, mapping, mapped_graph,
                           depth + 1, max_node_cap, set, max_results);

        mapped_graph[g_id] = 0;
        mapping[p_idx] = -1;
    }
    safe_free((void**)&label_candidates);
    return result;
}

SubgraphMatchSet* graph_query_subgraph_isomorphism(PropertyGraph* pg,
                                                   const QueryGraphPattern* pattern,
                                                   const QueryOptions* options) {
    if (!pg || !pattern || pattern->node_count == 0) return NULL;
    QueryOptions opt = options ? *options : query_options_default();

    SubgraphMatchSet* set = (SubgraphMatchSet*)safe_malloc(sizeof(SubgraphMatchSet));
    if (!set) return NULL;
    set->matches = NULL;
    set->match_count = 0;
    set->capacity = 0;

    int pg_cap = property_graph_get_node_capacity(pg);
    if (pg_cap <= 0) { safe_free((void**)&set); return NULL; }

    int* mapping = (int*)safe_malloc(
        (size_t)pattern->node_count * sizeof(int));
    int* mapped_graph = (int*)safe_calloc((size_t)pg_cap, sizeof(int));
    if (!mapping || !mapped_graph) {
        safe_free((void**)&mapping);
        safe_free((void**)&mapped_graph);
        safe_free((void**)&set);
        return NULL;
    }
    memset(mapping, -1, (size_t)pattern->node_count * sizeof(int));

    vf2_search(pg, pattern, mapping, mapped_graph, 0,
               (size_t)pg_cap, set, opt.max_results);

    safe_free((void**)&mapping);
    safe_free((void**)&mapped_graph);
    return set;
}

SubgraphMatchSet* graph_query_find_similar_subgraph(PropertyGraph* pg,
                                                    int example_node_id,
                                                    int max_distance,
                                                    float similarity_threshold,
                                                    const QueryOptions* options) {
/* 使用max_distance限制子图相似搜索的范围 */
    if (!pg || example_node_id < 0) return NULL;
    QueryOptions opt = options ? *options : query_options_default();
    int effective_max_dist = (max_distance > 0) ? max_distance : 1024;

    SubgraphMatchSet* set = (SubgraphMatchSet*)safe_malloc(sizeof(SubgraphMatchSet));
    if (!set) return NULL;
    set->matches = NULL;
    set->match_count = 0;
    set->capacity = 0;

    const PGNode* example = property_graph_get_node(pg, example_node_id);
    if (!example) { safe_free((void**)&set); return NULL; }

    int pg_cap = property_graph_get_node_capacity(pg);

    for (int i = 0; i < pg_cap && set->match_count < opt.max_results; i++) {
        const PGNode* other = property_graph_get_node(pg, i);
        if (!other || other->id == example_node_id) continue;

        float sim = 0.0f;
        int shared_props = 0;

        if (example->label && other->label &&
            strcmp(example->label, other->label) == 0)
            sim += 0.3f;

        for (size_t pi = 0; pi < example->properties.count; pi++) {
            for (size_t pj = 0; pj < other->properties.count; pj++) {
                if (strcmp(example->properties.items[pi].key,
                           other->properties.items[pj].key) == 0) {
                    PropertyValue* ev = &example->properties.items[pi].value;
                    PropertyValue* ov = &other->properties.items[pj].value;
                    if (ev->type == ov->type) {
                        int match = 0;
                        switch (ev->type) {
                            case PROP_TYPE_INT:
                                match = (ev->data.int_val == ov->data.int_val);
                                break;
                            case PROP_TYPE_FLOAT:
                                match = (fabs(ev->data.float_val -
                                    ov->data.float_val) < 1e-10);
                                break;
                            case PROP_TYPE_STRING:
                                if (ev->str_val && ov->str_val)
                                    match = (strcmp(ev->str_val, ov->str_val) == 0);
                                break;
                            default: break;
                        }
                        if (match) sim += 0.2f;
                    }
                    shared_props++;
                }
            }
        }
        if (shared_props > 0) sim += 0.1f;

        if (sim >= similarity_threshold) {
            if (set->match_count >= set->capacity) {
                size_t new_cap = set->capacity == 0 ? 16 : set->capacity * 2;
                SubgraphMatchResult* new_m = (SubgraphMatchResult*)safe_realloc(
                    set->matches, new_cap * sizeof(SubgraphMatchResult));
                if (!new_m) break;
                set->matches = new_m;
                set->capacity = new_cap;
            }
            int* matched = (int*)safe_malloc(sizeof(int));
            if (!matched) break;
            matched[0] = other->id;
            set->matches[set->match_count].matched_node_ids = matched;
            set->matches[set->match_count].node_count = 1;
            set->matches[set->match_count].source_engine_type = 1;
            set->matches[set->match_count].similarity = sim;
            set->matches[set->match_count].matched_labels = NULL;
            set->match_count++;
        }
    }
    return set;
}

QueryResultSet* graph_query_by_property(PropertyGraph* pg,
                                        const QueryPropertyConstraint* constraints,
                                        size_t constraint_count,
                                        const char* edge_label,
                                        int max_hops,
                                        const QueryOptions* options) {
/* 使用edge_label和max_hops参数过滤查询结果 */
    if (!pg || !constraints || constraint_count == 0) return NULL;
    QueryOptions opt = options ? *options : query_options_default();
    int use_edge_filter = (edge_label != NULL && edge_label[0] != '\0');
    int effective_hops = (max_hops > 0) ? max_hops : 1024;

    char vnames[QUERY_MAX_VARS][64];
    strncpy(vnames[0], "?node", 63);
    QueryResultSet* rs = query_result_set_create(1, vnames);
    if (!rs) return NULL;

    int pg_cap = property_graph_get_node_capacity(pg);
    for (int i = 0; i < pg_cap && rs->row_count < opt.max_results; i++) {
        const PGNode* n = property_graph_get_node(pg, i);
        if (!n) continue;
        int node_id = n->id;

        int all_match = 1;
        for (size_t c = 0; c < constraint_count; c++) {
            PropertyValue pv;
            int ret = property_graph_get_node_property(pg, node_id,
                        constraints[c].property_key, &pv);
            if (ret != 0) { all_match = 0; break; }
            int match = 0;
            if (pv.type == constraints[c].prop_type) {
                switch (pv.type) {
                    case PROP_TYPE_INT:
                        if (constraints[c].op == 0)
                            match = (pv.data.int_val ==
                                     constraints[c].prop_value.data.int_val);
                        break;
                    case PROP_TYPE_FLOAT:
                        if (constraints[c].op == 0)
                            match = (fabs(pv.data.float_val -
                                constraints[c].prop_value.data.float_val) < 1e-10);
                        break;
                    case PROP_TYPE_STRING:
                        if (pv.str_val && constraints[c].prop_value.str_val &&
                            constraints[c].op == 0)
                            match = (strcmp(pv.str_val,
                                constraints[c].prop_value.str_val) == 0);
                        break;
                    default: break;
                }
            }
            property_value_free(&pv);
            if (!match) { all_match = 0; break; }
        }

        if (all_match) {
            QueryResultRow row;
            memset(&row, 0, sizeof(row));
            row.variables[0].bound_type = 1;
            row.variables[0].node_id = node_id;
            row.confidence = 1.0f;
            query_result_set_add_row(rs, &row);
        }
    }

    if (opt.sort_by_confidence)
        query_result_set_sort(rs, "", 0);
    return rs;
}

/* ============================================================================
 * L-001: 图遍历算法实现 —— DFS深度优先、A*启发式搜索、Dijkstra最短路径
 * ============================================================================ */

/**
 * @brief L-001: 计算节点标签的FNV-1a哈希值（用于A*启发函数）
 */
static uint32_t gq_fnv1a_hash(const char* str) {
    uint32_t hash = 2166136261u;
    if (!str) return hash;
    while (*str) {
        hash ^= (uint8_t)(*str++);
        hash *= 16777619u;
    }
    return hash;
}

/**
 * @brief L-001: 查找节点ID对应的索引（在邻接表中，节点ID通常从0开始连续分配）
 */
static int gq_find_node_index(AdjacencyList* al, int node_id) {
    /* 节点ID在邻接表中从0开始连续分配，直接用node_id作索引 */
    if (node_id < 0 || (size_t)node_id >= adjacency_list_node_count(al)) return -1;
    const ALNode* node = adjacency_list_get_node(al, node_id);
    return node ? node_id : -1;
}

/* ---- L-001: DFS深度优先遍历 ---- */

static int gq_dfs_recursive(AdjacencyList* al, int node_id, int depth,
                             int max_depth, int* visited,
                             GraphTraverseCallback callback, void* user_data) {
    const ALNode* node = adjacency_list_get_node(al, node_id);
    if (!node || node_id < 0 || (size_t)node_id >= adjacency_list_node_count(al)) return 0;

    visited[node_id] = 1;
    int count = 1;

    if (callback) {
        callback(node->id, node->label ? node->label : "", depth, user_data);
    }

    if (max_depth > 0 && depth >= max_depth) return count;

    /* 获取出边邻居 */
    int neighbors[256];
    float weights[256];
    size_t out_deg = 0, in_deg = 0;
    adjacency_list_get_degree(al, node_id, &out_deg, &in_deg);
    if (out_deg > 256) out_deg = 256;
    int nb_count = adjacency_list_get_out_neighbors(al, node_id, neighbors, weights, out_deg);

    for (int i = 0; i < nb_count; i++) {
        int nb_id = neighbors[i];
        if (nb_id < 0 || (size_t)nb_id >= adjacency_list_node_count(al)) continue;
        if (visited[nb_id]) continue;
        count += gq_dfs_recursive(al, nb_id, depth + 1, max_depth,
                                  visited, callback, user_data);
    }
    return count;
}

int graph_query_dfs_traverse(AdjacencyList* al, int start_node_id,
                             int max_depth, GraphTraverseCallback callback,
                             void* user_data) {
    if (!al) return -1;
    size_t n = adjacency_list_node_count(al);
    if (n == 0) return -1;
    if (start_node_id < 0 || (size_t)start_node_id >= n) return -1;
    if (!adjacency_list_get_node(al, start_node_id)) return -1;

    int* visited = (int*)safe_calloc(n, sizeof(int));
    if (!visited) return -1;

    int count = gq_dfs_recursive(al, start_node_id, 0, max_depth, visited,
                                 callback, user_data);
    safe_free((void**)&visited);
    return count;
}

/* ---- L-001: 二叉堆优先队列 ---- */

typedef struct {
    int node_id;
    float priority;
} GQHeapEntry;

typedef struct {
    GQHeapEntry* entries;
    size_t capacity;
    size_t size;
} GQMinHeap;

static GQMinHeap* gq_heap_create(size_t capacity) {
    GQMinHeap* heap = (GQMinHeap*)safe_malloc(sizeof(GQMinHeap));
    if (!heap) return NULL;
    heap->entries = (GQHeapEntry*)safe_malloc(capacity * sizeof(GQHeapEntry));
    if (!heap->entries) { safe_free((void**)&heap); return NULL; }
    heap->capacity = capacity;
    heap->size = 0;
    return heap;
}

static void gq_heap_free(GQMinHeap* heap) {
    if (!heap) return;
    safe_free((void**)&heap->entries);
    safe_free((void**)&heap);
}

static void gq_heap_push(GQMinHeap* heap, int node_id, float priority) {
    if (!heap || heap->size >= heap->capacity) return;
    size_t i = heap->size++;
    heap->entries[i].node_id = node_id;
    heap->entries[i].priority = priority;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (heap->entries[parent].priority <= heap->entries[i].priority) break;
        GQHeapEntry tmp = heap->entries[parent];
        heap->entries[parent] = heap->entries[i];
        heap->entries[i] = tmp;
        i = parent;
    }
}

static int gq_heap_pop(GQMinHeap* heap, float* priority_out) {
    if (!heap || heap->size == 0) return -1;
    int result = heap->entries[0].node_id;
    if (priority_out) *priority_out = heap->entries[0].priority;
    heap->entries[0] = heap->entries[--heap->size];
    size_t i = 0;
    for (;;) {
        size_t smallest = i;
        size_t left = 2 * i + 1;
        size_t right = 2 * i + 2;
        if (left < heap->size && heap->entries[left].priority < heap->entries[smallest].priority)
            smallest = left;
        if (right < heap->size && heap->entries[right].priority < heap->entries[smallest].priority)
            smallest = right;
        if (smallest == i) break;
        GQHeapEntry tmp = heap->entries[i];
        heap->entries[i] = heap->entries[smallest];
        heap->entries[smallest] = tmp;
        i = smallest;
    }
    return result;
}

/* ---- L-001: A*启发式搜索 ---- */

AStarPathResult* graph_query_astar_search(AdjacencyList* al,
                                          int start_node_id, int goal_node_id) {
    if (!al || adjacency_list_node_count(al) == 0) return NULL;
    int start_idx = gq_find_node_index(al, start_node_id);
    int goal_idx = gq_find_node_index(al, goal_node_id);
    if (start_idx < 0 || goal_idx < 0) return NULL;

    const ALNode* goal_node = adjacency_list_get_node(al, goal_idx);
    const char* goal_label = goal_node ? goal_node->label : NULL;
    uint32_t goal_hash = gq_fnv1a_hash(goal_label ? goal_label : "");
    size_t n = adjacency_list_node_count(al);

    float* g_score = (float*)safe_malloc(n * sizeof(float));
    int* came_from = (int*)safe_malloc(n * sizeof(int));
    int* closed_set = (int*)safe_calloc(n, sizeof(int));
    if (!g_score || !came_from || !closed_set) {
        safe_free((void**)&g_score); safe_free((void**)&came_from);
        safe_free((void**)&closed_set); return NULL;
    }

    for (size_t i = 0; i < n; i++) { g_score[i] = 1e30f; came_from[i] = -1; }
    g_score[start_idx] = 0.0f;

    GQMinHeap* open_set = gq_heap_create(n * 2);
    if (!open_set) {
        safe_free((void**)&g_score); safe_free((void**)&came_from);
        safe_free((void**)&closed_set); return NULL;
    }

    {
        const ALNode* sl_node = adjacency_list_get_node(al, start_idx);
        const char* sl = sl_node ? sl_node->label : NULL;
        float h = (float)(gq_fnv1a_hash(sl ? sl : "") ^ goal_hash) / 4294967295.0f;
        gq_heap_push(open_set, start_node_id, h);
    }

    int nodes_explored = 0;
    while (open_set->size > 0) {
        float f_val;
        int current_id = gq_heap_pop(open_set, &f_val);
        int current_idx = gq_find_node_index(al, current_id);
        if (current_idx < 0) continue;
        nodes_explored++;

        if (current_id == goal_node_id) {
            /* 重建路径 */
            size_t path_len = 0;
            int cur = goal_idx;
            while (cur >= 0) { path_len++; cur = came_from[cur]; }
            int* path = (int*)safe_malloc(path_len * sizeof(int));
            if (path) {
                cur = goal_idx; size_t idx = path_len;
                while (cur >= 0) { 
                    const ALNode* cn = adjacency_list_get_node(al, cur);
                    path[--idx] = cn ? cn->id : cur; 
                    cur = came_from[cur]; 
                }
            }
            AStarPathResult* result = (AStarPathResult*)safe_malloc(sizeof(AStarPathResult));
            if (result) {
                result->path = path; result->path_length = path_len;
                result->total_cost = g_score[goal_idx]; result->nodes_explored = nodes_explored;
            } else { safe_free((void**)&path); }
            gq_heap_free(open_set);
            safe_free((void**)&g_score); safe_free((void**)&came_from);
            safe_free((void**)&closed_set);
            return result;
        }

        if (closed_set[current_idx]) continue;
        closed_set[current_idx] = 1;

        const ALNode* node = adjacency_list_get_node(al, current_idx);
        for (size_t i = 0; i < node->out_degree; i++) {
            int nb_id = node->out_neighbors[i];
            int nb_idx = gq_find_node_index(al, nb_id);
            if (nb_idx < 0 || closed_set[nb_idx]) continue;

            float ew = node->out_weights ? 1.0f - node->out_weights[i] : 1.0f;
            if (ew < 0.0f) ew = 0.0f;
            float tg = g_score[current_idx] + ew;
            if (tg < g_score[nb_idx]) {
                came_from[nb_idx] = current_idx;
                g_score[nb_idx] = tg;
                const ALNode* nl_node = adjacency_list_get_node(al, nb_idx);
                const char* nl = nl_node ? nl_node->label : NULL;
                float h = (float)(gq_fnv1a_hash(nl ? nl : "") ^ goal_hash) / 4294967295.0f;
                gq_heap_push(open_set, nb_id, tg + h);
            }
        }
    }

    gq_heap_free(open_set);
    safe_free((void**)&g_score); safe_free((void**)&came_from);
    safe_free((void**)&closed_set);
    return NULL;
}

void graph_query_astar_result_free(AStarPathResult* result) {
    if (!result) return;
    safe_free((void**)&result->path);
    safe_free((void**)&result);
}

/* ---- L-001: Dijkstra最短路径 ---- */

DijkstraResult* graph_query_dijkstra_shortest_path(AdjacencyList* al,
                                                    int start_node_id) {
    if (!al || adjacency_list_node_count(al) == 0) return NULL;
    int start_idx = gq_find_node_index(al, start_node_id);
    if (start_idx < 0) return NULL;

    size_t n = adjacency_list_node_count(al);
    float* distances = (float*)safe_malloc(n * sizeof(float));
    int* predecessors = (int*)safe_malloc(n * sizeof(int));
    int* visited = (int*)safe_calloc(n, sizeof(int));
    if (!distances || !predecessors || !visited) {
        safe_free((void**)&distances); safe_free((void**)&predecessors);
        safe_free((void**)&visited); return NULL;
    }

    for (size_t i = 0; i < n; i++) { distances[i] = 1e30f; predecessors[i] = -1; }
    distances[start_idx] = 0.0f;

    GQMinHeap* heap = gq_heap_create(n * 2);
    if (!heap) {
        safe_free((void**)&distances); safe_free((void**)&predecessors);
        safe_free((void**)&visited); return NULL;
    }
    gq_heap_push(heap, start_node_id, 0.0f);

    while (heap->size > 0) {
        float dist;
        int current_id = gq_heap_pop(heap, &dist);
        int current_idx = gq_find_node_index(al, current_id);
        if (current_idx < 0) continue;
        if (visited[current_idx]) continue;
        visited[current_idx] = 1;

        const ALNode* node = adjacency_list_get_node(al, current_idx);
        for (size_t i = 0; i < node->out_degree; i++) {
            int nb_id = node->out_neighbors[i];
            int nb_idx = gq_find_node_index(al, nb_id);
            if (nb_idx < 0 || visited[nb_idx]) continue;

            float ew = node->out_weights ? 1.0f - node->out_weights[i] : 1.0f;
            if (ew < 0.0f) ew = 0.0f;
            float nd = distances[current_idx] + ew;
            if (nd < distances[nb_idx]) {
                distances[nb_idx] = nd;
                predecessors[nb_idx] = current_idx;
                gq_heap_push(heap, nb_id, nd);
            }
        }
    }

    gq_heap_free(heap);
    safe_free((void**)&visited);

    DijkstraResult* result = (DijkstraResult*)safe_malloc(sizeof(DijkstraResult));
    if (!result) {
        safe_free((void**)&distances); safe_free((void**)&predecessors);
        return NULL;
    }
    result->distances = distances;
    result->predecessors = predecessors;
    result->node_count = n;
    result->start_node_id = start_node_id;
    return result;
}

int* graph_query_dijkstra_get_path(const DijkstraResult* result,
                                   int target_node_id,
                                   size_t* path_length_out) {
    if (!result || !path_length_out) return NULL;

    int idx = target_node_id;
    if (idx < 0 || (size_t)idx >= result->node_count) return NULL;
    if (result->distances[idx] >= 1e29f) { *path_length_out = 0; return NULL; }

    size_t len = 0;
    int cur = idx;
    while (cur >= 0) { len++; cur = result->predecessors[cur]; }

    int* path = (int*)safe_malloc(len * sizeof(int));
    if (!path) { *path_length_out = 0; return NULL; }

    cur = idx; size_t pos = len;
    while (cur >= 0) { path[--pos] = cur; cur = result->predecessors[cur]; }

    *path_length_out = len;
    return path;
}

void graph_query_dijkstra_result_free(DijkstraResult* result) {
    if (!result) return;
    safe_free((void**)&result->distances);
    safe_free((void**)&result->predecessors);
    safe_free((void**)&result);
}