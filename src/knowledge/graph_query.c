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
#include "selflnn/core/safe_memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

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
    (void)var_name;
    if (!rs || rs->row_count == 0) return;
    if (ascending)
        qsort(rs->rows, rs->row_count, sizeof(QueryResultRow), binding_compare_asc);
    else
        qsort(rs->rows, rs->row_count, sizeof(QueryResultRow), binding_compare_desc);
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
    for (size_t i = 0; i < ncount; i++) {
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

    for (size_t i = 0; i < pattern_count; i++) {
        pat_s_var[i] = pat_p_var[i] = pat_o_var[i] = -1;
        pat_s_const[i] = pat_p_const[i] = pat_o_const[i] = 0;
        pat_s_val[i] = pat_p_val[i] = pat_o_val[i] = -1;

        for (size_t v = 0; v < var_count; v++) {
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

    size_t max_results = limit > 0 ? limit : opt.max_results;
    size_t total = rdf_triple_store_count(store);

    /* 全常量模式 */
    int all_const = 1;
    for (size_t i = 0; i < pattern_count; i++)
        if (patterns[i].subject_is_var || patterns[i].predicate_is_var || patterns[i].object_is_var)
            { all_const = 0; break; }

    if (all_const && pattern_count > 0) {
        QueryResultRow row;
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
    for (size_t pi = 0; pi < pattern_count; pi++) {
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
                for (int bi = 0; bi < fetched && rs->row_count < max_results; bi++) {
                    RDFTriple* t = &batch[bi];
                    int all_ok = 1;
                    for (size_t pi = 0; pi < pattern_count && all_ok; pi++) {
                        if (pat_s_const[pi] && t->subject_id != pat_s_val[pi]) all_ok = 0;
                        if (pat_p_const[pi] && t->predicate_id != pat_p_val[pi]) all_ok = 0;
                        if (pat_o_const[pi] && t->object_id != pat_o_val[pi]) all_ok = 0;
                    }
                    if (!all_ok) continue;
                    
                    QueryResultRow row;
                    memset(&row, 0, sizeof(row));
                    row.confidence = t->confidence;
                    for (size_t pi = 0; pi < pattern_count; pi++) {
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
        /* 无约束全扫描（保留原有逐批查询但限制总量） */
        size_t scan_limit = total < 10000 ? total : 10000;
        for (size_t ti = 0; ti < scan_limit && rs->row_count < max_results; ti++) {
            RDFTriple batch[16];
            int found = rdf_triple_store_query(store, -1, -1, -1, batch, 16);
            if (found <= 0) break;
            
            for (int bi = 0; bi < found && rs->row_count < max_results; bi++) {
                RDFTriple* t = &batch[bi];
                int all_ok = 1;
                for (size_t pi = 0; pi < pattern_count && all_ok; pi++) {
                    if (pat_s_const[pi] && t->subject_id != pat_s_val[pi]) all_ok = 0;
                    if (pat_p_const[pi] && t->predicate_id != pat_p_val[pi]) all_ok = 0;
                    if (pat_o_const[pi] && t->object_id != pat_o_val[pi]) all_ok = 0;
                }
                if (!all_ok) continue;
                
                QueryResultRow row;
                memset(&row, 0, sizeof(row));
                row.confidence = t->confidence;
                for (size_t pi = 0; pi < pattern_count; pi++) {
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

    if (graph_query_parse_sparql(query_str, patterns, 64,
                                 &pattern_count, variables, QUERY_MAX_VARS,
                                 &var_count, &limit) != 0) {
        return NULL;
    }

    return graph_query_execute_sparql(store, patterns, pattern_count,
                                      variables, var_count, limit, NULL);
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

        /* 多节点模式：从起始节点做DFS匹配 */
        int depth_unused = 1;
        (void)depth_unused;
        int* stack_nodes = (int*)safe_malloc(
            (size_t)pattern->node_count * sizeof(int));
        int* stack_pat = (int*)safe_malloc(
            (size_t)pattern->node_count * sizeof(int));
        int* stack_ni = (int*)safe_malloc(
            (size_t)pattern->node_count * sizeof(int));
        if (!stack_nodes || !stack_pat || !stack_ni) {
            safe_free((void**)&stack_nodes);
            safe_free((void**)&stack_pat);
            safe_free((void**)&stack_ni);
            safe_free((void**)&matched_ids);
            break;
        }

        /* 选择下一个未匹配的模式节点 */
        int next_pat = -1;
        for (int j = 1; j < pattern->node_count; j++) {
            if (!used_pattern[j]) { next_pat = j; break; }
        }

        if (next_pat < 0) {
            safe_free((void**)&stack_nodes);
            safe_free((void**)&stack_pat);
            safe_free((void**)&stack_ni);
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
            safe_free((void**)&stack_nodes);
            safe_free((void**)&stack_pat);
            safe_free((void**)&stack_ni);
            continue;
        }

        int found_match = 0;
        int nb[1024];
        int nb_count = adjacency_list_get_out_neighbors(al, start_ids[si],
                                                        nb, NULL, 1024);
        for (int ni = 0; ni < nb_count && !found_match; ni++) {
            int cand = nb[ni];
            const ALNode* cn = adjacency_list_get_node(al, cand);
            if (!cn) continue;
            const char* want_label = pattern->node_labels[next_pat];
            int label_match = (want_label[0] == '\0') ? 1 : 0;
            if (!label_match && cn->label)
                label_match = (strcmp(cn->label, want_label) == 0);
            if (!label_match) continue;

            matched_ids[1] = cand;
            used_pattern[next_pat] = 1;
            found_match = 1;
        }

        if (found_match) {
            if (set->match_count >= set->capacity) {
                size_t new_cap = set->capacity == 0 ? 16 : set->capacity * 2;
                SubgraphMatchResult* new_m = (SubgraphMatchResult*)safe_realloc(
                    set->matches, new_cap * sizeof(SubgraphMatchResult));
                if (!new_m) {
                    safe_free((void**)&stack_nodes);
                    safe_free((void**)&stack_pat);
                    safe_free((void**)&stack_ni);
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
        safe_free((void**)&stack_ni);

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
    (void)max_distance;
    if (!al || !center_label) return NULL;
    QueryOptions opt = options ? *options : query_options_default();

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
    (void)min_confidence;
    if (!al || !start_label || !path_edge_labels || path_length == 0)
        return NULL;

    SubgraphMatchSet* set = (SubgraphMatchSet*)safe_malloc(sizeof(SubgraphMatchSet));
    if (!set) return NULL;
    set->matches = NULL;
    set->match_count = 0;
    set->capacity = 0;

    int starts[1024];
    int start_count = adjacency_list_find_by_label(al, start_label, starts, 1024);

    for (int si = 0; si < start_count; si++) {
        int path_nodes[1024];
        path_nodes[0] = starts[si];
        int path_len = 1;
        int cur = starts[si];

        for (size_t step = 0; step < path_length; step++) {
            int nbs[1024];
            int nb_count = adjacency_list_get_out_neighbors(al, cur, nbs, NULL, 1024);
            int found = 0;
            if (nb_count > 0) {
                path_nodes[path_len++] = nbs[0];
                cur = nbs[0];
                found = 1;
            }
            if (!found) break;
        }

        if (path_len > 1) {
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

static int vf2_search(PropertyGraph* pg, const QueryGraphPattern* pattern,
                      int* mapping, int* mapped_graph,
                      int depth, size_t max_node_cap,
                      SubgraphMatchSet* set, size_t max_results) {
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
    (void)max_distance;
    if (!pg || example_node_id < 0) return NULL;
    QueryOptions opt = options ? *options : query_options_default();

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
    (void)edge_label; (void)max_hops;
    if (!pg || !constraints || constraint_count == 0) return NULL;
    QueryOptions opt = options ? *options : query_options_default();

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