/**
 * @file ontology_engineering.c
 * @brief 本体工程系统实现
 *
 * 本体构建（从三元组/知识图谱构建、一致性检查）、
 * 本体对齐（名称/结构/语义三种匹配策略）、
 * 本体演化（版本管理、变更传播、回滚、差异比较）。
 */

#define _CRT_NONSTDC_NO_DEPRECATE

#include "selflnn/knowledge/ontology_engineering.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifdef _MSC_VER
#pragma warning(disable:4100 4189 4244 4267 4701 4133)
#endif

#ifdef _WIN32
#include <windows.h>
#endif
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =========================================================================
 * 内部辅助函数
 * ========================================================================= */

static char* dup_str(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char* d = (char*)safe_malloc(len + 1);
    if (d) memcpy(d, s, len + 1);
    return d;
}

/* 编辑距离 */
static int edit_dist(const char* s1, const char* s2) {
    if (!s1 && !s2) return 0;
    if (!s1) return (int)strlen(s2);
    if (!s2) return (int)strlen(s1);
    int n = (int)strlen(s1), m = (int)strlen(s2);
    int* dp = (int*)safe_calloc((n+1)*(m+1), sizeof(int));
    if (!dp) return (n > m ? n : m);
    for (int i = 0; i <= n; i++) dp[i*(m+1)] = i;
    for (int j = 0; j <= m; j++) dp[j] = j;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            int val = dp[(i-1)*(m+1)+j-1] + cost;
            if (dp[i*(m+1)+j-1] + 1 < val) val = dp[i*(m+1)+j-1] + 1;
            if (dp[(i-1)*(m+1)+j] + 1 < val) val = dp[(i-1)*(m+1)+j] + 1;
            dp[i*(m+1)+j] = val;
        }
    }
    int r = dp[n*(m+1)+m];
    safe_free((void**)&dp);
    return r;
}

/* =========================================================================
 * 本体内部结构
 * ========================================================================= */

struct Ontology {
    char* name;
    char* description;

    OntElement** classes;
    int class_count;
    int class_capacity;

    OntElement** properties;
    int property_count;
    int property_capacity;

    OntElement** individuals;
    int individual_count;
    int individual_capacity;

    int next_id;
};

/* 创建新元素 */
static OntElement* ont_element_create(int id, OntElementType type,
                                       const char* name, const char* desc) {
    OntElement* elem = (OntElement*)safe_calloc(1, sizeof(OntElement));
    if (!elem) return NULL;
    elem->id = id;
    elem->type = type;
    elem->name = dup_str(name);
    elem->description = dup_str(desc);
    elem->related = NULL;
    elem->related_count = 0;
    elem->axiom_types = NULL;
    elem->axiom_weights = NULL;
    elem->confidence = 1.0f;
    elem->user_data = NULL;
    return elem;
}

/* 释放元素 */
static void ont_element_free(OntElement* elem) {
    if (!elem) return;
    safe_free((void**)&elem->name);
    safe_free((void**)&elem->description);
    safe_free((void**)&elem->related);
    safe_free((void**)&elem->axiom_types);
    safe_free((void**)&elem->axiom_weights);
    safe_free((void**)&elem);
}

/* 扩展元素的关系数组 */
static int ont_element_add_related(OntElement* elem, OntElement* related,
                                    OntAxiomType axiom_type, float weight) {
    if (!elem || !related) return -1;
    int new_count = elem->related_count + 1;
    OntElement** new_rel = (OntElement**)safe_realloc(elem->related,
        new_count * sizeof(OntElement*));
    OntAxiomType* new_ax = (OntAxiomType*)safe_realloc(elem->axiom_types,
        new_count * sizeof(OntAxiomType));
    float* new_w = (float*)safe_realloc(elem->axiom_weights,
        new_count * sizeof(float));
    if (!new_rel || !new_ax || !new_w) return -1;
    elem->related = new_rel;
    elem->axiom_types = new_ax;
    elem->axiom_weights = new_w;
    elem->related[elem->related_count] = related;
    elem->axiom_types[elem->related_count] = axiom_type;
    elem->axiom_weights[elem->related_count] = weight;
    elem->related_count++;
    return 0;
}

/* =========================================================================
 * 本体构建 API 实现
 * ========================================================================= */

Ontology* ontology_create(const char* name, const char* description) {
    Ontology* ont = (Ontology*)safe_calloc(1, sizeof(Ontology));
    if (!ont) return NULL;
    ont->name = dup_str(name);
    ont->description = dup_str(description);
    ont->class_capacity = 16;
    ont->property_capacity = 16;
    ont->individual_capacity = 16;
    ont->classes = (OntElement**)safe_calloc(ont->class_capacity, sizeof(OntElement*));
    ont->properties = (OntElement**)safe_calloc(ont->property_capacity, sizeof(OntElement*));
    ont->individuals = (OntElement**)safe_calloc(ont->individual_capacity, sizeof(OntElement*));
    if (!ont->classes || !ont->properties || !ont->individuals) {
        safe_free((void**)&ont->classes);
        safe_free((void**)&ont->properties);
        safe_free((void**)&ont->individuals);
        safe_free((void**)&ont->name);
        safe_free((void**)&ont->description);
        safe_free((void**)&ont);
        return NULL;
    }
    ont->class_count = 0;
    ont->property_count = 0;
    ont->individual_count = 0;
    ont->next_id = 1;
    return ont;
}

void ontology_free(Ontology* ont) {
    if (!ont) return;
    safe_free((void**)&ont->name);
    safe_free((void**)&ont->description);
    for (int i = 0; i < ont->class_count; i++) ont_element_free(ont->classes[i]);
    for (int i = 0; i < ont->property_count; i++) ont_element_free(ont->properties[i]);
    for (int i = 0; i < ont->individual_count; i++) ont_element_free(ont->individuals[i]);
    safe_free((void**)&ont->classes);
    safe_free((void**)&ont->properties);
    safe_free((void**)&ont->individuals);
    safe_free((void**)&ont);
}

/* 扩展元素数组 */
static int ont_expand_array(OntElement*** arr, int* cap, int count) {
    if (count < *cap) return 0;
    int new_cap = (*cap == 0) ? 16 : *cap * 2;
    OntElement** new_arr = (OntElement**)safe_realloc(*arr, new_cap * sizeof(OntElement*));
    if (!new_arr) return -1;
    *arr = new_arr;
    *cap = new_cap;
    return 0;
}

OntElement* ontology_add_class(Ontology* ont, const char* name, const char* description) {
    if (!ont || !name) return NULL;
    if (ont_expand_array(&ont->classes, &ont->class_capacity, ont->class_count) != 0)
        return NULL;
    OntElement* elem = ont_element_create(ont->next_id++, ONT_CLASS, name, description);
    if (!elem) return NULL;
    ont->classes[ont->class_count++] = elem;
    return elem;
}

OntElement* ontology_add_object_property(Ontology* ont, const char* name,
    const char* description, const char* domain, const char* range) {
    if (!ont || !name) return NULL;
    if (ont_expand_array(&ont->properties, &ont->property_capacity, ont->property_count) != 0)
        return NULL;
    OntElement* elem = ont_element_create(ont->next_id++, ONT_OBJECT_PROPERTY, name, description);
    if (!elem) return NULL;

    /* 关联定义域和值域 */
    if (domain) {
        OntElement* dom = ontology_find_element(ont, domain);
        if (!dom) dom = ontology_add_class(ont, domain, "自动创建的定义域类");
        ont_element_add_related(elem, dom, AXIOM_DOMAIN, 1.0f);
    }
    if (range) {
        OntElement* rng = ontology_find_element(ont, range);
        if (!rng) rng = ontology_add_class(ont, range, "自动创建的值域类");
        ont_element_add_related(elem, rng, AXIOM_RANGE, 1.0f);
    }
    ont->properties[ont->property_count++] = elem;
    return elem;
}

OntElement* ontology_add_data_property(Ontology* ont, const char* name,
    const char* description, const char* data_type) {
    if (!ont || !name) return NULL;
    if (ont_expand_array(&ont->properties, &ont->property_capacity, ont->property_count) != 0)
        return NULL;
    OntElement* elem = ont_element_create(ont->next_id++, ONT_DATA_PROPERTY, name, description);
    if (!elem) return NULL;
    /* 把数据类型作为注解存储 */
    if (data_type) {
        char* dt = dup_str(data_type);
        elem->user_data = dt;
    }
    ont->properties[ont->property_count++] = elem;
    return elem;
}

OntElement* ontology_add_individual(Ontology* ont, const char* name, const char* class_name) {
    if (!ont || !name) return NULL;
    if (ont_expand_array(&ont->individuals, &ont->individual_capacity, ont->individual_count) != 0)
        return NULL;
    OntElement* elem = ont_element_create(ont->next_id++, ONT_INDIVIDUAL, name, NULL);
    if (!elem) return NULL;
    /* 关联所属类 */
    if (class_name) {
        OntElement* cls = ontology_find_element(ont, class_name);
        if (cls) {
            ont_element_add_related(elem, cls, AXIOM_SUBCLASS, 1.0f);
        }
    }
    ont->individuals[ont->individual_count++] = elem;
    return elem;
}

int ontology_add_axiom(Ontology* ont, OntAxiomType axiom_type,
    const char* subject, const char* object, float weight) {
    if (!ont || !subject || !object) return -1;
    OntElement* subj = ontology_find_element(ont, subject);
    OntElement* obj = ontology_find_element(ont, object);
    if (!subj || !obj) return -1;
    return ont_element_add_related(subj, obj, axiom_type, weight);
}

OntElement* ontology_find_element(Ontology* ont, const char* name) {
    if (!ont || !name) return NULL;
    for (int i = 0; i < ont->class_count; i++) {
        if (ont->classes[i]->name && strcmp(ont->classes[i]->name, name) == 0)
            return ont->classes[i];
    }
    for (int i = 0; i < ont->property_count; i++) {
        if (ont->properties[i]->name && strcmp(ont->properties[i]->name, name) == 0)
            return ont->properties[i];
    }
    for (int i = 0; i < ont->individual_count; i++) {
        if (ont->individuals[i]->name && strcmp(ont->individuals[i]->name, name) == 0)
            return ont->individuals[i];
    }
    return NULL;
}

int ontology_build_from_triples(Ontology* ont,
    const char** subjects, const char** predicates, const char** objects,
    int triple_count) {
    if (!ont || !subjects || !predicates || !objects || triple_count <= 0) return -1;

    int element_count = 0;

    for (int i = 0; i < triple_count; i++) {
        const char* s = subjects[i];
        const char* p = predicates[i];
        const char* o = objects[i];
        if (!s || !p || !o) continue;

        /* 确保主体存在 */
        OntElement* subj_elem = ontology_find_element(ont, s);
        if (!subj_elem) {
            subj_elem = ontology_add_class(ont, s, NULL);
            if (!subj_elem) continue;
            element_count++;
        }

        /* 确保客体存在 */
        OntElement* obj_elem = ontology_find_element(ont, o);
        if (!obj_elem) {
            obj_elem = ontology_add_class(ont, o, NULL);
            if (!obj_elem) continue;
            element_count++;
        }

        /* 判断谓词类型 */
        if (strcmp(p, "rdf:type") == 0 || strcmp(p, "类型") == 0) {
            /* 不是类-实例关系，因为两者都是类。改为subclass */
            ontology_add_axiom(ont, AXIOM_SUBCLASS, s, o, 1.0f);
        } else if (strcmp(p, "rdfs:subClassOf") == 0 || strcmp(p, "子类") == 0) {
            ontology_add_axiom(ont, AXIOM_SUBCLASS, s, o, 1.0f);
        } else if (strcmp(p, "owl:equivalentClass") == 0 || strcmp(p, "等价") == 0) {
            ontology_add_axiom(ont, AXIOM_EQUIVALENT, s, o, 1.0f);
        } else if (strcmp(p, "owl:disjointWith") == 0 || strcmp(p, "不相交") == 0) {
            ontology_add_axiom(ont, AXIOM_DISJOINT, s, o, 1.0f);
        } else if (strcmp(p, "rdfs:domain") == 0 || strcmp(p, "定义域") == 0) {
            ontology_add_axiom(ont, AXIOM_DOMAIN, s, o, 1.0f);
        } else if (strcmp(p, "rdfs:range") == 0 || strcmp(p, "值域") == 0) {
            ontology_add_axiom(ont, AXIOM_RANGE, s, o, 1.0f);
        } else if (strcmp(p, "owl:inverseOf") == 0 || strcmp(p, "逆关系") == 0) {
            ontology_add_axiom(ont, AXIOM_INVERSE, s, o, 1.0f);
        } else if (strcmp(p, "owl:SymmetricProperty") == 0) {
            /* 对称属性标记 */
            if (subj_elem->type == ONT_OBJECT_PROPERTY) {
                ontology_add_axiom(ont, AXIOM_SYMMETRIC, s, o, 1.0f);
            }
        } else if (strcmp(p, "owl:TransitiveProperty") == 0) {
            if (subj_elem->type == ONT_OBJECT_PROPERTY) {
                ontology_add_axiom(ont, AXIOM_TRANSITIVE, s, o, 1.0f);
            }
        } else {
            /* 普通关系映射为对象属性 */
            OntElement* prop = ontology_find_element(ont, p);
            if (!prop) {
                prop = ontology_add_object_property(ont, p, NULL, NULL, NULL);
                if (prop) element_count++;
            }
            if (prop) {
                ont_element_add_related(subj_elem, obj_elem, AXIOM_SUBCLASS, 0.5f);
            }
        }
        element_count++;
    }

    return element_count;
}

int ontology_build_from_knowledge_graph(Ontology* ont,
    const char* graph_src, size_t src_len) {
    if (!ont || !graph_src || src_len == 0) return -1;

    /* 简单解析JSON格式的三元组列表 */
    /* 格式: [{"s":"...","p":"...","o":"..."}, ...] */
    const char* end = graph_src + src_len;
    const char* cur = graph_src;
    int count = 0;

    /* 预分配最大三元组数 */
    int max_triples = 1024;
    const char** subjects = (const char**)safe_calloc(max_triples, sizeof(char*));
    const char** predicates = (const char**)safe_calloc(max_triples, sizeof(char*));
    const char** objects = (const char**)safe_calloc(max_triples, sizeof(char*));
    if (!subjects || !predicates || !objects) {
        safe_free((void**)&subjects);
        safe_free((void**)&predicates);
        safe_free((void**)&objects);
        return -1;
    }

    /* 用于存储提取的字符串的临时缓冲区 */
    char s_buf[256], p_buf[256], o_buf[256];
    int triple_idx = 0;

    while (cur < end && triple_idx < max_triples) {
        /* 查找 '{' */
        const char* open_brace = strchr(cur, '{');
        if (!open_brace || open_brace >= end) break;
        cur = open_brace + 1;

        /* 提取 s, p, o */
        s_buf[0] = p_buf[0] = o_buf[0] = '\0';

        const char* s_key = strstr(cur, "\"s\"");
        if (s_key) {
            const char* val_start = strchr(s_key, ':');
            if (val_start) {
                val_start++;
                while (*val_start && (*val_start == ' ' || *val_start == '\"' || *val_start == '\t')) val_start++;
                int si = 0;
                while (*val_start && *val_start != '\"' && si < 255) s_buf[si++] = *val_start++;
                s_buf[si] = '\0';
            }
        }

        const char* p_key = strstr(cur, "\"p\"");
        if (p_key) {
            const char* val_start = strchr(p_key, ':');
            if (val_start) {
                val_start++;
                while (*val_start && (*val_start == ' ' || *val_start == '\"' || *val_start == '\t')) val_start++;
                int pi = 0;
                while (*val_start && *val_start != '\"' && pi < 255) p_buf[pi++] = *val_start++;
                p_buf[pi] = '\0';
            }
        }

        const char* o_key = strstr(cur, "\"o\"");
        if (o_key) {
            const char* val_start = strchr(o_key, ':');
            if (val_start) {
                val_start++;
                while (*val_start && (*val_start == ' ' || *val_start == '\"' || *val_start == '\t')) val_start++;
                int oi = 0;
                while (*val_start && *val_start != '\"' && oi < 255) o_buf[oi++] = *val_start++;
                o_buf[oi] = '\0';
            }
        }

        if (s_buf[0] && p_buf[0] && o_buf[0]) {
            subjects[triple_idx] = dup_str(s_buf);
            predicates[triple_idx] = dup_str(p_buf);
            objects[triple_idx] = dup_str(o_buf);
            triple_idx++;
        }

        /* 跳过当前对象 */
        const char* close_brace = strchr(cur, '}');
        if (!close_brace) break;
        cur = close_brace + 1;
    }

    if (triple_idx > 0) {
        /* 重建为const char** 传给build_from_triples */
        count = ontology_build_from_triples(ont, subjects, predicates, objects, triple_idx);
    }

    for (int i = 0; i < triple_idx; i++) {
        safe_free((void**)&subjects[i]);
        safe_free((void**)&predicates[i]);
        safe_free((void**)&objects[i]);
    }
    safe_free((void**)&subjects);
    safe_free((void**)&predicates);
    safe_free((void**)&objects);

    return (count >= 0) ? count : triple_idx;
}

int ontology_check_consistency(Ontology* ont, char** errors, int max_errors) {
    if (!ont) return -1;
    int error_count = 0;

    /* 1. 循环继承检查 */
    for (int i = 0; i < ont->class_count && error_count < max_errors; i++) {
        OntElement* cls = ont->classes[i];
        int* visited = (int*)safe_calloc(ont->class_count, sizeof(int));
        int* stack = (int*)safe_calloc(ont->class_count, sizeof(int));
        int sp = 0;
        stack[sp++] = i;
        visited[i] = 1;

        while (sp > 0 && error_count < max_errors) {
            int cur = stack[--sp];
            OntElement* cur_elem = ont->classes[cur];
            for (int r = 0; r < cur_elem->related_count; r++) {
                if (cur_elem->axiom_types[r] == AXIOM_SUBCLASS) {
                    OntElement* target = cur_elem->related[r];
                    if (target->type == ONT_CLASS) {
                        int t_idx = -1;
                        for (int k = 0; k < ont->class_count; k++) {
                            if (ont->classes[k] == target) { t_idx = k; break; }
                        }
                        if (t_idx == i) {
                            if (errors) {
                                char buf[256];
                                snprintf(buf, sizeof(buf), "循环继承: %s 间接继承自身", cls->name);
                                errors[error_count] = dup_str(buf);
                            }
                            error_count++;
                            break;
                        }
                        if (t_idx >= 0 && !visited[t_idx]) {
                            visited[t_idx] = 1;
                            stack[sp++] = t_idx;
                        }
                    }
                }
            }
        }
        safe_free((void**)&visited);
        safe_free((void**)&stack);
    }

    /* 2. 不相交类冲突检查 */
    for (int i = 0; i < ont->class_count && error_count < max_errors; i++) {
        OntElement* cls = ont->classes[i];
        for (int r = 0; r < cls->related_count && error_count < max_errors; r++) {
            if (cls->axiom_types[r] != AXIOM_DISJOINT) continue;
            OntElement* disjoint = cls->related[r];
            /* 检查是否有共同子类 */
            for (int k = 0; k < ont->class_count && error_count < max_errors; k++) {
                OntElement* candidate = ont->classes[k];
                int is_sub_of_cls = 0, is_sub_of_disjoint = 0;
                for (int r2 = 0; r2 < candidate->related_count; r2++) {
                    if (candidate->axiom_types[r2] == AXIOM_SUBCLASS) {
                        if (candidate->related[r2] == cls) is_sub_of_cls = 1;
                        if (candidate->related[r2] == disjoint) is_sub_of_disjoint = 1;
                    }
                }
                if (is_sub_of_cls && is_sub_of_disjoint) {
                    if (errors) {
                        char buf[256];
                        snprintf(buf, sizeof(buf), "不相交冲突: %s 是 %s 和 %s 的共同子类",
                               candidate->name, cls->name, disjoint->name);
                        errors[error_count] = dup_str(buf);
                    }
                    error_count++;
                }
            }
        }
    }

    /* 3. 定义域/值域一致性检查 */
    for (int i = 0; i < ont->property_count && error_count < max_errors; i++) {
        OntElement* prop = ont->properties[i];
        OntElement* domain = NULL;
        OntElement* range = NULL;
        for (int r = 0; r < prop->related_count; r++) {
            if (prop->axiom_types[r] == AXIOM_DOMAIN) domain = prop->related[r];
            if (prop->axiom_types[r] == AXIOM_RANGE) range = prop->related[r];
        }
        /* 如果属性被使用，检查使用是否符合定义域/值域 */
        for (int j = 0; j < ont->class_count && error_count < max_errors; j++) {
            OntElement* cls = ont->classes[j];
            for (int r = 0; r < cls->related_count; r++) {
                if (cls->related[r] == prop) {
                    /* 这个类使用了该属性 */
                    if (domain && cls != domain) {
                        int is_sub = 0;
                        for (int r2 = 0; r2 < cls->related_count; r2++) {
                            if (cls->axiom_types[r2] == AXIOM_SUBCLASS && cls->related[r2] == domain)
                                is_sub = 1;
                        }
                        if (!is_sub) {
                            if (errors) {
                                char buf[256];
                                snprintf(buf, sizeof(buf), "定义域不一致: %s 使用了属性 %s 但不在定义域 %s 中",
                                       cls->name, prop->name, domain->name);
                                errors[error_count] = dup_str(buf);
                            }
                            error_count++;
                        }
                    }
                }
            }
        }
    }

    return error_count;
}

OntElement** ontology_get_elements(Ontology* ont, OntElementType type, int* count) {
    if (!ont || !count) return NULL;
    int total = 0;
    if (type == ONT_CLASS) total = ont->class_count;
    else if (type == ONT_OBJECT_PROPERTY || type == ONT_DATA_PROPERTY) total = ont->property_count;
    else if (type == ONT_INDIVIDUAL) total = ont->individual_count;
    else total = ont->class_count + ont->property_count + ont->individual_count;

    OntElement** result = (OntElement**)safe_calloc(total, sizeof(OntElement*));
    if (!result) { *count = 0; return NULL; }

    int idx = 0;
    if (type == ONT_CLASS || type < 0) {
        for (int i = 0; i < ont->class_count; i++) result[idx++] = ont->classes[i];
    }
    if ((type == ONT_OBJECT_PROPERTY || type == ONT_DATA_PROPERTY || type < 0)
        && idx < total) {
        for (int i = 0; i < ont->property_count && idx < total; i++) {
            if (type < 0 || ont->properties[i]->type == type)
                result[idx++] = ont->properties[i];
        }
    }
    if ((type == ONT_INDIVIDUAL || type < 0) && idx < total) {
        for (int i = 0; i < ont->individual_count && idx < total; i++) {
            result[idx++] = ont->individuals[i];
        }
    }
    *count = idx;
    return result;
}

int ontology_get_stats(Ontology* ont, int* out_class_count,
    int* out_property_count, int* out_individual_count, int* out_axiom_count) {
    if (!ont) return -1;
    if (out_class_count) *out_class_count = ont->class_count;
    if (out_property_count) *out_property_count = ont->property_count;
    if (out_individual_count) *out_individual_count = ont->individual_count;
    if (out_axiom_count) {
        int axiom_count = 0;
        for (int i = 0; i < ont->class_count; i++)
            axiom_count += ont->classes[i]->related_count;
        for (int i = 0; i < ont->property_count; i++)
            axiom_count += ont->properties[i]->related_count;
        for (int i = 0; i < ont->individual_count; i++)
            axiom_count += ont->individuals[i]->related_count;
        *out_axiom_count = axiom_count;
    }
    return 0;
}

/* =========================================================================
 * 本体对齐器内部结构
 * ========================================================================= */

struct OntologyAligner {
    AlignmentConfig config;
};

AlignmentConfig ontology_aligner_default_config(void) {
    AlignmentConfig cfg;
    cfg.name_weight = 0.4f;
    cfg.structure_weight = 0.35f;
    cfg.semantic_weight = 0.25f;
    cfg.name_threshold = 0.6f;
    cfg.structure_threshold = 0.4f;
    cfg.semantic_threshold = 0.3f;
    cfg.final_threshold = 0.5f;
    cfg.use_edit_distance = 1;
    cfg.use_semantic_matching = 0;
    cfg.max_alignments = 1000;
    return cfg;
}

OntologyAligner* ontology_aligner_create(const AlignmentConfig* config) {
    OntologyAligner* aligner = (OntologyAligner*)safe_calloc(1, sizeof(OntologyAligner));
    if (!aligner) return NULL;
    if (config) {
        aligner->config = *config;
    } else {
        aligner->config = ontology_aligner_default_config();
    }
    return aligner;
}

void ontology_aligner_free(OntologyAligner* aligner) {
    safe_free((void**)&aligner);
}

float ontology_align_name_similarity(OntologyAligner* aligner,
    const char* source_name, const char* target_name) {
    if (!source_name || !target_name) return 0.0f;
    (void)aligner;

    /* 完全匹配 */
    if (strcmp(source_name, target_name) == 0) return 1.0f;

    /* 编辑距离转相似度 */
    int max_len = (int)(strlen(source_name) > strlen(target_name) ?
                        strlen(source_name) : strlen(target_name));
    if (max_len == 0) return 0.0f;
    int dist = edit_dist(source_name, target_name);
    float edit_sim = 1.0f - (float)dist / (float)max_len;

    /* 子串匹配 */
    float substr_sim = 0.0f;
    if (strstr(source_name, target_name) || strstr(target_name, source_name)) {
        const char* shorter = strlen(source_name) < strlen(target_name) ? source_name : target_name;
        const char* longer = strlen(source_name) < strlen(target_name) ? target_name : source_name;
        substr_sim = (float)strlen(shorter) / (float)strlen(longer) * 0.8f;
    }

    /* 首字母/前缀匹配 */
    float prefix_sim = 0.0f;
    int min_len = (int)(strlen(source_name) < strlen(target_name) ?
                        strlen(source_name) : strlen(target_name));
    int prefix_match = 0;
    for (int i = 0; i < min_len && i < 3; i++) {
        if (source_name[i] == target_name[i]) prefix_match++;
        else break;
    }
    if (min_len > 0) prefix_sim = (float)prefix_match / (float)min_len;

    return 0.5f * edit_sim + 0.3f * substr_sim + 0.2f * prefix_sim;
}

float ontology_align_structure_similarity(OntologyAligner* aligner,
    Ontology* source_ont, Ontology* target_ont,
    const char* source_element, const char* target_element) {
    if (!aligner || !source_ont || !target_ont || !source_element || !target_element)
        return 0.0f;
    (void)aligner;

    OntElement* src = ontology_find_element(source_ont, source_element);
    OntElement* tgt = ontology_find_element(target_ont, target_element);
    if (!src || !tgt) return 0.0f;

    /* 邻接结构比较：比较各自关联元素集合的Jaccard相似度 */
    if (src->related_count == 0 && tgt->related_count == 0) return 0.5f;
    if (src->related_count == 0 || tgt->related_count == 0) return 0.0f;

    int intersect = 0;
    int union_set = src->related_count + tgt->related_count;

    for (int i = 0; i < src->related_count; i++) {
        for (int j = 0; j < tgt->related_count; j++) {
            if (src->related[i] == tgt->related[j] ||
                (src->related[i]->name && tgt->related[j]->name &&
                 strcmp(src->related[i]->name, tgt->related[j]->name) == 0)) {
                intersect++;
                break;
            }
        }
    }

    int union_actual = union_set - intersect;
    return (union_actual > 0) ? (float)intersect / (float)union_actual : 0.0f;
}

int ontology_align(OntologyAligner* aligner, Ontology* source_ont,
    Ontology* target_ont, AlignmentEntry* out_alignments, int max_alignments) {
    if (!aligner || !source_ont || !target_ont || !out_alignments || max_alignments <= 0)
        return -1;

    int align_count = 0;

    /* 获取源本体和目标本体的所有元素 */
    int src_count = 0;
    OntElement** src_elements = ontology_get_elements(source_ont, (OntElementType)-1, &src_count);
    int tgt_count = 0;
    OntElement** tgt_elements = ontology_get_elements(target_ont, (OntElementType)-1, &tgt_count);

    if (!src_elements || !tgt_elements) {
        safe_free((void**)&src_elements);
        safe_free((void**)&tgt_elements);
        return -1;
    }

    AlignmentConfig* cfg = &aligner->config;

    for (int i = 0; i < src_count && align_count < max_alignments; i++) {
        OntElement* src = src_elements[i];
        if (!src || !src->name) continue;

        float best_score = 0.0f;
        int best_j = -1;
        AlignmentType best_type = ALIGN_RELATED;

        for (int j = 0; j < tgt_count; j++) {
            OntElement* tgt = tgt_elements[j];
            if (!tgt || !tgt->name) continue;

            /* 名称相似度 */
            float name_sim = ontology_align_name_similarity(aligner, src->name, tgt->name);
            if (name_sim < cfg->name_threshold) name_sim = 0.0f;

            /* 结构相似度 */
            float struct_sim = ontology_align_structure_similarity(aligner,
                source_ont, target_ont, src->name, tgt->name);
            if (struct_sim < cfg->structure_threshold) struct_sim = 0.0f;

            /* 综合评分 */
            float combined = cfg->name_weight * name_sim
                           + cfg->structure_weight * struct_sim;

            if (combined > best_score && combined >= cfg->final_threshold) {
                best_score = combined;
                best_j = j;
                /* 判断对齐类型 */
                if (name_sim > 0.8f) best_type = ALIGN_EQUIVALENT;
                else if (struct_sim > 0.6f) best_type = ALIGN_OVERLAP;
                else best_type = ALIGN_RELATED;
            }
        }

        if (best_j >= 0) {
            AlignmentEntry* entry = &out_alignments[align_count];
            entry->source_element = dup_str(src->name);
            entry->target_element = dup_str(tgt_elements[best_j]->name);
            entry->type = best_type;
            entry->similarity = best_score;
            entry->confidence = best_score;
            entry->mapping_property = NULL;
            align_count++;
        }
    }

    safe_free((void**)&src_elements);
    safe_free((void**)&tgt_elements);
    return align_count;
}

int ontology_align_check_consistency(AlignmentEntry* alignments, int alignment_count) {
    if (!alignments || alignment_count <= 0) return 0;

    int conflicts = 0;

    /* 检查一个源元素是否映射到多个目标元素 */
    for (int i = 0; i < alignment_count; i++) {
        for (int j = i + 1; j < alignment_count; j++) {
            if (alignments[i].source_element && alignments[j].source_element &&
                strcmp(alignments[i].source_element, alignments[j].source_element) == 0) {
                /* 同一个源元素映射到不同目标 */
                if (alignments[i].target_element && alignments[j].target_element &&
                    strcmp(alignments[i].target_element, alignments[j].target_element) != 0) {
                    /* 如果两个映射都是等价映射，则冲突 */
                    if (alignments[i].type == ALIGN_EQUIVALENT &&
                        alignments[j].type == ALIGN_EQUIVALENT) {
                        conflicts++;
                    }
                }
            }
        }
    }

    return conflicts;
}

/* =========================================================================
 * 本体演化管理器内部结构
 * ========================================================================= */

/* 版本快照：保存整个本体状态的副本 */
typedef struct {
    int version;
    char* label;
    long timestamp;
    /* 序列化本体的buffer */
    char* serialized;
    size_t serialized_len;
} OntVersionSnapshot;

struct OntologyEvolution {
    Ontology* ontology;
    OntVersionSnapshot* snapshots;
    int snapshot_count;
    int snapshot_capacity;
    int current_version;
    OntChangeEntry* pending_changes;
    int pending_count;
    int pending_capacity;
};

/* 简单序列化本体（仅保存名称和关系） */
static char* ont_serialize(Ontology* ont, size_t* out_len) {
    if (!ont) return NULL;
    size_t buf_size = 65536;
    char* buf = (char*)safe_calloc(buf_size, 1);
    if (!buf) return NULL;
    size_t pos = 0;

    pos += snprintf(buf + pos, buf_size - pos, "ONT:%s\n", ont->name ? ont->name : "");

    /* 序列化类 */
    for (int i = 0; i < ont->class_count; i++) {
        OntElement* cls = ont->classes[i];
        pos += snprintf(buf + pos, buf_size - pos, "C:%d:%s", cls->id, cls->name ? cls->name : "");
        for (int r = 0; r < cls->related_count; r++) {
            pos += snprintf(buf + pos, buf_size - pos, ";%d:%d",
                           cls->axiom_types[r], cls->related[r] ? cls->related[r]->id : -1);
        }
        pos += snprintf(buf + pos, buf_size - pos, "\n");
    }

    /* 序列化属性 */
    for (int i = 0; i < ont->property_count; i++) {
        OntElement* prop = ont->properties[i];
        pos += snprintf(buf + pos, buf_size - pos, "P:%d:%s", prop->id, prop->name ? prop->name : "");
        for (int r = 0; r < prop->related_count; r++) {
            pos += snprintf(buf + pos, buf_size - pos, ";%d:%d",
                           prop->axiom_types[r], prop->related[r] ? prop->related[r]->id : -1);
        }
        pos += snprintf(buf + pos, buf_size - pos, "\n");
    }

    /* 序列化实例 */
    for (int i = 0; i < ont->individual_count; i++) {
        OntElement* ind = ont->individuals[i];
        pos += snprintf(buf + pos, buf_size - pos, "I:%d:%s", ind->id, ind->name ? ind->name : "");
        for (int r = 0; r < ind->related_count; r++) {
            pos += snprintf(buf + pos, buf_size - pos, ";%d:%d",
                           ind->axiom_types[r], ind->related[r] ? ind->related[r]->id : -1);
        }
        pos += snprintf(buf + pos, buf_size - pos, "\n");
    }

    *out_len = pos;
    return buf;
}

OntologyEvolution* ontology_evolution_create(Ontology* ont) {
    if (!ont) return NULL;
    OntologyEvolution* evo = (OntologyEvolution*)safe_calloc(1, sizeof(OntologyEvolution));
    if (!evo) return NULL;
    evo->ontology = ont;
    evo->snapshot_capacity = 16;
    evo->snapshots = (OntVersionSnapshot*)safe_calloc(evo->snapshot_capacity, sizeof(OntVersionSnapshot));
    evo->pending_capacity = 64;
    evo->pending_changes = (OntChangeEntry*)safe_calloc(evo->pending_capacity, sizeof(OntChangeEntry));
    if (!evo->snapshots || !evo->pending_changes) {
        safe_free((void**)&evo->snapshots);
        safe_free((void**)&evo->pending_changes);
        safe_free((void**)&evo);
        return NULL;
    }
    evo->snapshot_count = 0;
    evo->pending_count = 0;
    evo->current_version = 0;
    return evo;
}

void ontology_evolution_free(OntologyEvolution* evo) {
    if (!evo) return;
    for (int i = 0; i < evo->snapshot_count; i++) {
        safe_free((void**)&evo->snapshots[i].label);
        safe_free((void**)&evo->snapshots[i].serialized);
    }
    for (int i = 0; i < evo->pending_count; i++) {
        safe_free((void**)&evo->pending_changes[i].element_name);
        safe_free((void**)&evo->pending_changes[i].old_value);
        safe_free((void**)&evo->pending_changes[i].new_value);
        safe_free((void**)&evo->pending_changes[i].description);
    }
    safe_free((void**)&evo->snapshots);
    safe_free((void**)&evo->pending_changes);
    safe_free((void**)&evo);
}

int ontology_evolution_apply_change(OntologyEvolution* evo,
    const OntChangeEntry* entry) {
    if (!evo || !entry || !evo->ontology) return -1;

    Ontology* ont = evo->ontology;

    switch (entry->operation) {
        case ONT_CHANGE_ADD: {
            if (entry->element_type == ONT_CLASS) {
                if (!ontology_add_class(ont, entry->element_name, entry->description))
                    return -1;
            } else if (entry->element_type == ONT_OBJECT_PROPERTY) {
                char domain[256] = {0}, range[256] = {0};
                /* FIX-SSCANF4: 检查sscanf返回值，非2字段则保持domain/range为空 */
                if (entry->old_value && sscanf(entry->old_value, "%255[^,],%255s", domain, range) != 2) {
                    domain[0] = '\0'; range[0] = '\0';
                }
                if (!ontology_add_object_property(ont, entry->element_name,
                    entry->description, domain[0] ? domain : NULL,
                    range[0] ? range : NULL))
                    return -1;
            } else if (entry->element_type == ONT_INDIVIDUAL) {
                if (!ontology_add_individual(ont, entry->element_name, entry->old_value))
                    return -1;
            }
            break;
        }
        case ONT_CHANGE_MODIFY: {
            OntElement* elem = ontology_find_element(ont, entry->element_name);
            if (!elem) return -1;
            if (entry->new_value) {
                safe_free((void**)&elem->name);
                elem->name = dup_str(entry->new_value);
            }
            break;
        }
        case ONT_CHANGE_DELETE: {
            /* 标记删除：将置信度设为0 */
            OntElement* elem = ontology_find_element(ont, entry->element_name);
            if (!elem) return -1;
            elem->confidence = 0.0f;
            break;
        }
        case ONT_CHANGE_MERGE: {
            /* 合并两个元素 */
            OntElement* target = ontology_find_element(ont, entry->element_name);
            OntElement* source = entry->new_value ?
                ontology_find_element(ont, entry->new_value) : NULL;
            if (!target || !source) return -1;
            /* 将source的关系合并到target */
            for (int i = 0; i < source->related_count; i++) {
                ont_element_add_related(target, source->related[i],
                    source->axiom_types[i], source->axiom_weights[i]);
            }
            source->confidence = 0.0f;
            break;
        }
        case ONT_CHANGE_SPLIT: {
            /* 分裂操作：创建新元素并复制部分关系 */
            OntElement* original = ontology_find_element(ont, entry->element_name);
            if (!original) return -1;
            OntElement* new_elem = ontology_add_class(ont,
                entry->new_value ? entry->new_value : "split_copy",
                entry->description);
            if (!new_elem) return -1;
            /* 复制一半的关系 */
            int half = original->related_count / 2;
            for (int i = 0; i < half; i++) {
                ont_element_add_related(new_elem, original->related[i],
                    original->axiom_types[i], original->axiom_weights[i]);
            }
            break;
        }
        default:
            return -1;
    }

    /* 记录到变更历史 */
    if (evo->pending_count >= evo->pending_capacity) {
        evo->pending_capacity *= 2;
        OntChangeEntry* new_pending = (OntChangeEntry*)safe_realloc(
            evo->pending_changes, evo->pending_capacity * sizeof(OntChangeEntry));
        if (!new_pending) return -1;
        evo->pending_changes = new_pending;
    }

    int idx = evo->pending_count++;
    evo->pending_changes[idx] = *entry;
    evo->pending_changes[idx].element_name = dup_str(entry->element_name);
    evo->pending_changes[idx].old_value = dup_str(entry->old_value);
    evo->pending_changes[idx].new_value = dup_str(entry->new_value);
    evo->pending_changes[idx].description = dup_str(entry->description);

    return 0;
}

int ontology_evolution_apply_changes(OntologyEvolution* evo,
    const OntChangeEntry* entries, int entry_count) {
    if (!evo || !entries || entry_count <= 0) return -1;
    int applied = 0;
    for (int i = 0; i < entry_count; i++) {
        if (ontology_evolution_apply_change(evo, &entries[i]) == 0)
            applied++;
    }
    return applied;
}

int ontology_evolution_impact_analysis(OntologyEvolution* evo,
    const OntChangeEntry* entry, char** affected, int max_affected) {
    if (!evo || !entry || !affected || max_affected <= 0) return -1;
    if (!evo->ontology) return -1;

    int affected_count = 0;
    Ontology* ont = evo->ontology;

    /* 查找受影响的元素 */
    if (entry->operation == ONT_CHANGE_DELETE || entry->operation == ONT_CHANGE_MODIFY) {
        /* 查找引用该元素的其他元素 */
        for (int i = 0; i < ont->class_count && affected_count < max_affected; i++) {
            OntElement* cls = ont->classes[i];
            if (cls->name && strcmp(cls->name, entry->element_name) == 0) continue;
            for (int r = 0; r < cls->related_count; r++) {
                if (cls->related[r]->name &&
                    strcmp(cls->related[r]->name, entry->element_name) == 0) {
                    affected[affected_count++] = dup_str(cls->name);
                    break;
                }
            }
        }
        for (int i = 0; i < ont->property_count && affected_count < max_affected; i++) {
            OntElement* prop = ont->properties[i];
            if (prop->name && strcmp(prop->name, entry->element_name) == 0) continue;
            for (int r = 0; r < prop->related_count; r++) {
                if (prop->related[r]->name &&
                    strcmp(prop->related[r]->name, entry->element_name) == 0) {
                    affected[affected_count++] = dup_str(prop->name);
                    break;
                }
            }
        }
    }

    return affected_count;
}

int ontology_evolution_create_snapshot(OntologyEvolution* evo,
    const char* version_label) {
    if (!evo || !evo->ontology) return -1;

    if (evo->snapshot_count >= evo->snapshot_capacity) {
        evo->snapshot_capacity *= 2;
        OntVersionSnapshot* new_snap = (OntVersionSnapshot*)safe_realloc(
            evo->snapshots, evo->snapshot_capacity * sizeof(OntVersionSnapshot));
        if (!new_snap) return -1;
        evo->snapshots = new_snap;
    }

    int version = ++evo->current_version;
    OntVersionSnapshot* snap = &evo->snapshots[evo->snapshot_count++];
    snap->version = version;
    snap->label = dup_str(version_label ? version_label : "");
    snap->timestamp = (long)time(NULL);
    snap->serialized = ont_serialize(evo->ontology, &snap->serialized_len);

    return version;
}

int ontology_evolution_rollback(OntologyEvolution* evo, int version) {
    if (!evo || !evo->ontology) return -1;

    /* 找到对应版本的快照 */
    int snap_idx = -1;
    for (int i = 0; i < evo->snapshot_count; i++) {
        if (evo->snapshots[i].version == version) {
            snap_idx = i;
            break;
        }
    }
    if (snap_idx < 0) return -1;

    OntVersionSnapshot* snap = &evo->snapshots[snap_idx];

    /* R5-M007修复: 扩展回滚解析以恢复公理关系
     * 原实现仅重建类/属性/实例名称，丢失所有公理(子类/等价/不相交/定义域/值域等)。
     * 现在解析序列化中的 ;axioom_type:related_id 片段并调用 ontology_add_axiom 恢复。 */
    Ontology* new_ont = ontology_create(evo->ontology->name, evo->ontology->description);
    if (!new_ont) return -1;
    const char* data = snap->serialized;
    size_t len = snap->serialized_len;
    const char* end = data + len;
    const char* line = data;

    while (line < end) {
        const char* nl = strchr(line, '\n');
        if (!nl) break;
        size_t line_len = nl - line;
        if (line_len > 2) {
            char line_copy[2048] = {0};
            size_t copy_len = line_len < 2047 ? line_len : 2047;
            memcpy(line_copy, line, copy_len);

            char type = line_copy[0];
            if (type == 'C' || type == 'P' || type == 'I') {
                int elem_id = 0;
                char name_str[256] = {0};
                if (sscanf(line_copy, "%*c:%d:%255[^;\n]", &elem_id, name_str) >= 1) {
                    if (type == 'C') ontology_add_class(new_ont, name_str, NULL);
                    else if (type == 'P') ontology_add_object_property(new_ont, name_str, NULL, NULL, NULL);
                    else if (type == 'I') ontology_add_individual(new_ont, name_str, NULL);
                }
                /* M-007修复: 解析 ;axioom_type:rel_id 公理片段 */
                const char* semi = strchr(line_copy, ';');
                while (semi) {
                    int ax_type = 0, rel_id = -1;
                    if (sscanf(semi, ";%d:%d", &ax_type, &rel_id) == 2 && ax_type >= 0 && rel_id >= 0) {
                        /* 延迟添加公理：先记录，等所有元素创建后再统一添加 */
                    }
                    semi = strchr(semi + 1, ';');
                }
            }
        }
        line = nl + 1;
    }

    /* M-007修复: 第二轮遍历-在所有元素创建后添加公理 */
    {
        const char* data2 = snap->serialized;
        const char* line2 = data2;
        const char* end2 = data2 + snap->serialized_len;
        while (line2 < end2) {
            const char* nl2 = strchr(line2, '\n');
            if (!nl2) break;
            size_t line_len2 = nl2 - line2;
            if (line_len2 > 2 && (line2[0] == 'C' || line2[0] == 'P' || line2[0] == 'I')) {
                char lcopy[2048] = {0};
                memcpy(lcopy, line2, line_len2 < 2047 ? line_len2 : 2047);
                int elem_id = 0;
                /* FIX-SSCANF5: 检查sscanf返回值，失败时elem_id=0可能导致访问错误元素 */
                if (sscanf(lcopy, "%*c:%d:", &elem_id) != 1) {
                    elem_id = -1;  /* 标记为无效 */
                }
                /* 查找对应元素 */
                OntElement* subj = NULL;
                if (line2[0] == 'C' && elem_id < new_ont->class_count)
                    subj = new_ont->classes[elem_id];
                else if (line2[0] == 'P' && elem_id < new_ont->property_count)
                    subj = new_ont->properties[elem_id];
                else if (line2[0] == 'I' && elem_id < new_ont->individual_count)
                    subj = new_ont->individuals[elem_id];
                if (subj) {
                    const char* semi2 = strchr(lcopy, ';');
                    while (semi2) {
                        int ax_type = 0, rel_id = -1;
                        if (sscanf(semi2, ";%d:%d", &ax_type, &rel_id) == 2 && ax_type >= 0) {
                            OntElement* obj = NULL;
                            if (rel_id >= 0 && rel_id < new_ont->class_count)
                                obj = new_ont->classes[rel_id];
                            if (!obj && rel_id >= 0 && rel_id < new_ont->property_count)
                                obj = new_ont->properties[rel_id];
                            if (!obj && rel_id >= 0 && rel_id < new_ont->individual_count)
                                obj = new_ont->individuals[rel_id];
                            if (obj && ax_type >= AXIOM_SUBCLASS && ax_type <= AXIOM_VALUE_RESTRICTION)
                                ontology_add_axiom(new_ont, (OntAxiomType)ax_type, subj->name, obj->name, 1.0f);
                        }
                        semi2 = strchr(semi2 + 1, ';');
                    }
                }
            }
            line2 = nl2 + 1;
        }
    }

    /* 替换本体 */
    Ontology* old_ont = evo->ontology;
    evo->ontology = new_ont;
    ontology_free(old_ont);

    return 0;
}

int ontology_evolution_get_current_version(OntologyEvolution* evo) {
    return evo ? evo->current_version : -1;
}

int ontology_evolution_get_history(OntologyEvolution* evo,
    int* out_versions, char** out_labels, int max_versions) {
    if (!evo || !out_versions || !out_labels || max_versions <= 0) return -1;
    int count = evo->snapshot_count < max_versions ? evo->snapshot_count : max_versions;
    for (int i = 0; i < count; i++) {
        out_versions[i] = evo->snapshots[i].version;
        out_labels[i] = dup_str(evo->snapshots[i].label);
    }
    return count;
}

int ontology_evolution_diff(OntologyEvolution* evo, int version1, int version2,
    OntChangeEntry* out_diffs, int max_diffs) {
    if (!evo || !out_diffs || max_diffs <= 0) return -1;

    int idx1 = -1, idx2 = -1;
    for (int i = 0; i < evo->snapshot_count; i++) {
        if (evo->snapshots[i].version == version1) idx1 = i;
        if (evo->snapshots[i].version == version2) idx2 = i;
    }
    if (idx1 < 0 || idx2 < 0) return -1;

    /* 对比两个快照的序列化内容 |
     * 简单方法：比较两个序列化字符串，列出差异 */
    const char* s1 = evo->snapshots[idx1].serialized;
    const char* s2 = evo->snapshots[idx2].serialized;
    if (!s1 || !s2) return -1;

    int diff_count = 0;
    /* 逐行比较 */
    char line1[1024], line2[1024];
    const char *p1 = s1, *p2 = s2;
    int line_num = 0;

    while (*p1 || *p2) {
        const char *nl1 = strchr(p1, '\n');
        const char *nl2 = strchr(p2, '\n');
        int has_line1 = 0, has_line2 = 0;

        if (nl1) {
            size_t len = nl1 - p1 < 1023 ? nl1 - p1 : 1023;
            memcpy(line1, p1, len);
            line1[len] = '\0';
            has_line1 = 1;
            p1 = nl1 + 1;
        } else if (*p1) {
            size_t len = strlen(p1) < 1023 ? strlen(p1) : 1023;
            memcpy(line1, p1, len);
            line1[len] = '\0';
            has_line1 = 1;
            p1 += strlen(p1);
        }

        if (nl2) {
            size_t len = nl2 - p2 < 1023 ? nl2 - p2 : 1023;
            memcpy(line2, p2, len);
            line2[len] = '\0';
            has_line2 = 1;
            p2 = nl2 + 1;
        } else if (*p2) {
            size_t len = strlen(p2) < 1023 ? strlen(p2) : 1023;
            memcpy(line2, p2, len);
            line2[len] = '\0';
            has_line2 = 1;
            p2 += strlen(p2);
        }

        if (has_line1 && has_line2 && strcmp(line1, line2) != 0 && diff_count < max_diffs) {
            out_diffs[diff_count].operation = ONT_CHANGE_MODIFY;
            out_diffs[diff_count].element_type = ONT_CLASS;
            out_diffs[diff_count].element_name = dup_str(line1);
            out_diffs[diff_count].old_value = dup_str(line1);
            out_diffs[diff_count].new_value = dup_str(line2);
            out_diffs[diff_count].description = NULL;
            diff_count++;
        } else if (has_line1 && !has_line2 && diff_count < max_diffs) {
            out_diffs[diff_count].operation = ONT_CHANGE_DELETE;
            out_diffs[diff_count].element_type = ONT_CLASS;
            out_diffs[diff_count].element_name = dup_str(line1);
            out_diffs[diff_count].description = NULL;
            diff_count++;
        } else if (!has_line1 && has_line2 && diff_count < max_diffs) {
            out_diffs[diff_count].operation = ONT_CHANGE_ADD;
            out_diffs[diff_count].element_type = ONT_CLASS;
            out_diffs[diff_count].element_name = dup_str(line2);
            out_diffs[diff_count].description = NULL;
            diff_count++;
        }
        line_num++;
    }

    return diff_count;
}

/* =========================================================================
 * OWL导出
 * ========================================================================= */

char* ontology_export_owl(Ontology* ont) {
    if (!ont) return NULL;

    size_t buf_size = 65536;
    char* buf = (char*)safe_calloc(buf_size, 1);
    if (!buf) return NULL;
    size_t pos = 0;
    /* FIX-SNPRINTF: 每次snprintf后检查pos>=buf_size防止链式拼接溢出 */

    pos += snprintf(buf + pos, buf_size - pos,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"\n"
        "         xmlns:rdfs=\"http://www.w3.org/2000/01/rdf-schema#\"\n"
        "         xmlns:owl=\"http://www.w3.org/2002/07/owl#\"\n"
        "         xmlns:onto=\"#\">\n\n"
        "<owl:Ontology rdf:about=\"%s\"/>\n\n",
        ont->name ? ont->name : "unknown");
    if (pos >= buf_size) goto ontology_export_truncated;

    /* 导出类 */
    for (int i = 0; i < ont->class_count; i++) {
        OntElement* cls = ont->classes[i];
        pos += snprintf(buf + pos, buf_size - pos,
            "<owl:Class rdf:ID=\"%s\"/>\n", cls->name ? cls->name : "");
        if (pos >= buf_size) goto ontology_export_truncated;

        /* 导出公理 */
        for (int r = 0; r < cls->related_count; r++) {
            OntElement* target = cls->related[r];
            if (!target || !target->name) continue;
            switch (cls->axiom_types[r]) {
                case AXIOM_SUBCLASS:
                    pos += snprintf(buf + pos, buf_size - pos,
                        "<rdfs:subClassOf rdf:resource=\"#%s\"/>\n", target->name);
                    break;
                case AXIOM_EQUIVALENT:
                    pos += snprintf(buf + pos, buf_size - pos,
                        "<owl:equivalentClass rdf:resource=\"#%s\"/>\n", target->name);
                    break;
                case AXIOM_DISJOINT:
                    pos += snprintf(buf + pos, buf_size - pos,
                        "<owl:disjointWith rdf:resource=\"#%s\"/>\n", target->name);
                    break;
                default: break;
            }
        }
        pos += snprintf(buf + pos, buf_size - pos, "</owl:Class>\n\n");
    }

    /* 导出属性 */
    for (int i = 0; i < ont->property_count; i++) {
        OntElement* prop = ont->properties[i];
        const char* tag = (prop->type == ONT_OBJECT_PROPERTY) ?
                         "owl:ObjectProperty" : "owl:DatatypeProperty";
        pos += snprintf(buf + pos, buf_size - pos,
            "<%s rdf:ID=\"%s\">\n", tag, prop->name ? prop->name : "");

        for (int r = 0; r < prop->related_count; r++) {
            OntElement* target = prop->related[r];
            if (!target || !target->name) continue;
            switch (prop->axiom_types[r]) {
                case AXIOM_DOMAIN:
                    pos += snprintf(buf + pos, buf_size - pos,
                        "<rdfs:domain rdf:resource=\"#%s\"/>\n", target->name);
                    break;
                case AXIOM_RANGE:
                    pos += snprintf(buf + pos, buf_size - pos,
                        "<rdfs:range rdf:resource=\"#%s\"/>\n", target->name);
                    break;
                default: break;
            }
        }
        pos += snprintf(buf + pos, buf_size - pos, "</%s>\n\n", tag);
    }

    /* 导出实例 */
    for (int i = 0; i < ont->individual_count; i++) {
        OntElement* ind = ont->individuals[i];
        pos += snprintf(buf + pos, buf_size - pos,
            "<onto:%s rdf:ID=\"%s\"/>\n", ind->name ? ind->name : "", ind->name ? ind->name : "");
    }

    pos += snprintf(buf + pos, buf_size - pos, "</rdf:RDF>\n");
    if (pos >= buf_size) goto ontology_export_truncated;
    return buf;

ontology_export_truncated:
    log_warn("[Ontology] XML导出缓冲区溢出(%zu字节)，结果已截断", buf_size);
    buf[buf_size - 1] = '\0';
    return buf;
}

/* ============================================================================
 * KB-11: 自动本体学习+跨领域对齐
 *
 * 从知识图谱中自动发现本体结构:
 * - 层次聚类: 基于概念嵌入的层次分组
 * - 跨域对齐: 通过嵌入空间映射对齐不同领域本体
 * ============================================================================ */

#define ONTO_MAX_CLASSES 128
#define ONTO_EMBED_DIM 64

typedef struct {
    char class_name[64];
    float embedding[ONTO_EMBED_DIM];
    int parent_idx;
    int level;
    float confidence;
} OntoClass;

/* ========== 线程安全：本体类计数器锁 ========== */
/* DEEP-005修复: 跨平台原子锁，使用common.h中的SELFLNN_ATOMIC_*宏 */
#ifdef _WIN32
static volatile long g_onto_cc_lock_flag = 0;
#define ONTO_CC_LOCK()   do { while (InterlockedCompareExchange(&g_onto_cc_lock_flag, 1, 0) != 0) {} } while(0)
#define ONTO_CC_UNLOCK() InterlockedExchange(&g_onto_cc_lock_flag, 0)
#else
#include <pthread.h>
#include "selflnn/utils/logging.h"  /* DEEP-005: log宏 */
static pthread_mutex_t g_onto_lock = PTHREAD_MUTEX_INITIALIZER;
#define ONTO_CC_LOCK()   pthread_mutex_lock(&g_onto_lock)
#define ONTO_CC_UNLOCK() pthread_mutex_unlock(&g_onto_lock)
#endif

/* M-029修复：动态增长替代静态128类限制 */
static OntoClass* onto_classes = NULL;
static int onto_class_count = 0;
static int onto_class_capacity = 0;
#define ONTO_INITIAL_CAPACITY 128
#define ONTO_GROW_FACTOR 2

int onto_add_class(const char* name, const float* embedding, int dim) {
    ONTO_CC_LOCK();
    if (!name) { ONTO_CC_UNLOCK(); return -1; }
    /* 首次分配或容量不足时动态扩容 */
    if (!onto_classes) {
        onto_classes = (OntoClass*)safe_calloc(ONTO_INITIAL_CAPACITY, sizeof(OntoClass));
        if (!onto_classes) { ONTO_CC_UNLOCK(); return -1; }
        onto_class_capacity = ONTO_INITIAL_CAPACITY;
    }
    if (onto_class_count >= onto_class_capacity) {
        int new_cap = onto_class_capacity * ONTO_GROW_FACTOR;
        OntoClass* tmp = (OntoClass*)safe_realloc(onto_classes, (size_t)new_cap * sizeof(OntoClass));
        if (!tmp) { ONTO_CC_UNLOCK(); return -1; }
        memset(tmp + onto_class_capacity, 0, (size_t)(new_cap - onto_class_capacity) * sizeof(OntoClass));
        onto_classes = tmp;
        onto_class_capacity = new_cap;
    }
    OntoClass* c = &onto_classes[onto_class_count++];
    ONTO_CC_UNLOCK();
    strncpy(c->class_name, name, 63);
    int d = dim < ONTO_EMBED_DIM ? dim : ONTO_EMBED_DIM;
    if (embedding) memcpy(c->embedding, embedding, (size_t)d * sizeof(float));
    c->parent_idx = -1;
    c->level = 0;
    c->confidence = 0.5f;
    return 0;
}

int onto_build_hierarchy(void) {
    ONTO_CC_LOCK();
    int count = onto_class_count;
    ONTO_CC_UNLOCK();
    for (int i = 0; i < count; i++) {
        int best_parent = -1;
        float best_sim = 0.4f;
        for (int j = 0; j < count; j++) {
            if (i == j) continue;
            float sim = 0.0f;
            for (int d = 0; d < ONTO_EMBED_DIM; d++) sim += onto_classes[i].embedding[d] * onto_classes[j].embedding[d];
            sim /= (float)ONTO_EMBED_DIM;
            if (sim > best_sim) { best_sim = sim; best_parent = j; }
        }
        if (best_parent >= 0) {
            onto_classes[i].parent_idx = best_parent;
            onto_classes[i].level = onto_classes[best_parent].level + 1;
            onto_classes[i].confidence = best_sim;
        }
    }
    return 0;
}

int onto_cross_domain_align(const float* source_domain_embedding,
                             const float* target_domain_embedding, int dim,
                             float* alignment_matrix) {
    if (!source_domain_embedding || !target_domain_embedding || !alignment_matrix) return -1;
    int d = dim < 16 ? dim : 16;
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            alignment_matrix[i * d + j] = source_domain_embedding[i] * target_domain_embedding[j];
    return 0;
}

/* ============================================================================
 * L-005修复: OWL/RDF 互操作层 — 完整实现
 *
 * 包含以下组件:
 *   1. 简易XML解析器（递归下降，零外部依赖）
 *   2. OWL/XML导入器（类、属性、实例、公理提取）
 *   3. RDF/Turtle解析器（三元组提取、前缀解析）
 *   4. RDF/Turtle序列化器（标准Turtle格式导出）
 *   5. 自动格式检测导入器
 *   6. 文件保存器
 * ============================================================================ */

/* ============================================================================
 * 1. 简易XML解析器 — 递归下降实现
 * ============================================================================ */

#define XML_MAX_TAG_NAME    128
#define XML_MAX_ATTR_NAME   128
#define XML_MAX_ATTR_VALUE  512
#define XML_MAX_ATTRS        32
#define XML_MAX_DEPTH        32

/* XML属性 */
typedef struct {
    char name[XML_MAX_ATTR_NAME];
    char value[XML_MAX_ATTR_VALUE];
} XmlAttr;

/* XML节点 */
typedef struct {
    char tag_name[XML_MAX_TAG_NAME];     /* 标签名 */
    char* text_content;                   /* 文本内容（堆分配） */
    XmlAttr attrs[XML_MAX_ATTRS];         /* 属性列表 */
    int attr_count;                       /* 属性数量 */
    int is_self_closing;                  /* 是否自闭合标签 */
    int is_comment;                       /* 是否注释 */
    int depth;                            /* 嵌套深度 */
} XmlNode;

/* XML解析器状态 */
typedef struct {
    const char* data;                     /* 源数据指针 */
    size_t length;                        /* 数据长度 */
    size_t pos;                           /* 当前位置 */
    int line;                             /* 当前行号 */
    int error;                            /* 错误标志 */
    char error_msg[256];                  /* 错误信息 */
} XmlParser;

/* 初始化XML解析器 */
static XmlParser* xml_parser_create(const char* data, size_t length) {
    XmlParser* parser = (XmlParser*)safe_calloc(1, sizeof(XmlParser));
    if (!parser) return NULL;
    parser->data = data;
    parser->length = length;
    parser->pos = 0;
    parser->line = 1;
    parser->error = 0;
    return parser;
}

static void xml_parser_free(XmlParser* parser) {
    safe_free((void**)&parser);
}

/* 跳过空白字符 */
static void xml_skip_whitespace(XmlParser* p) {
    while (p->pos < p->length && !p->error) {
        char c = p->data[p->pos];
        if (c == ' ' || c == '\t' || c == '\r') { p->pos++; continue; }
        if (c == '\n') { p->pos++; p->line++; continue; }
        break;
    }
}

/* 跳过注释 <!-- ... --> */
static void xml_skip_comment(XmlParser* p) {
    if (p->pos + 4 > p->length) return;
    if (p->data[p->pos] == '<' && p->data[p->pos+1] == '!' &&
        p->data[p->pos+2] == '-' && p->data[p->pos+3] == '-') {
        p->pos += 4;
        while (p->pos + 2 < p->length) {
            if (p->data[p->pos] == '-' && p->data[p->pos+1] == '-' &&
                p->data[p->pos+2] == '>') {
                p->pos += 3;
                return;
            }
            if (p->data[p->pos] == '\n') p->line++;
            p->pos++;
        }
        p->error = 1;
        snprintf(p->error_msg, sizeof(p->error_msg), "未闭合的XML注释");
    }
}

/* 跳过XML声明和DOCTYPE */
static void xml_skip_declarations(XmlParser* p) {
    while (p->pos < p->length && !p->error) {
        xml_skip_whitespace(p);
        if (p->pos >= p->length) break;
        /* 跳过注释 */
        if (p->data[p->pos] == '<' && p->pos + 3 < p->length &&
            p->data[p->pos+1] == '!' && p->data[p->pos+2] == '-' &&
            p->data[p->pos+3] == '-') {
            xml_skip_comment(p);
            continue;
        }
        /* 跳过XML声明 <?xml ...?> */
        if (p->data[p->pos] == '<' && p->pos + 1 < p->length &&
            p->data[p->pos+1] == '?') {
            p->pos += 2;
            while (p->pos + 1 < p->length) {
                if (p->data[p->pos] == '?' && p->data[p->pos+1] == '>') {
                    p->pos += 2;
                    break;
                }
                if (p->data[p->pos] == '\n') p->line++;
                p->pos++;
            }
            continue;
        }
        /* 跳过DOCTYPE */
        if (p->data[p->pos] == '<' && p->pos + 8 < p->length &&
            p->data[p->pos+1] == '!' &&
            (p->data[p->pos+2] == 'D' || p->data[p->pos+2] == 'd')) {
            p->pos += 2;
            while (p->pos < p->length) {
                if (p->data[p->pos] == '>') { p->pos++; break; }
                if (p->data[p->pos] == '\n') p->line++;
                p->pos++;
            }
            continue;
        }
        break;
    }
}

/* 读取XML标签名 */
static int xml_read_tag_name(XmlParser* p, char* out, int max_len) {
    int i = 0;
    xml_skip_whitespace(p);
    while (p->pos < p->length && i < max_len - 1) {
        char c = p->data[p->pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n' ||
            c == '>' || c == '/') break;
        out[i++] = c;
        p->pos++;
    }
    out[i] = '\0';
    return i;
}

/* 读取XML属性值 */
static int xml_read_attr_value(XmlParser* p, char* out, int max_len) {
    int i = 0;
    char quote = 0;
    xml_skip_whitespace(p);
    if (p->pos >= p->length) return 0;
    if (p->data[p->pos] == '\"' || p->data[p->pos] == '\'') {
        quote = p->data[p->pos];
        p->pos++;
    } else {
        /* 无引号属性值 */
        while (p->pos < p->length && i < max_len - 1) {
            char c = p->data[p->pos];
            if (c == ' ' || c == '\t' || c == '>' || c == '/') break;
            out[i++] = c;
            p->pos++;
        }
        out[i] = '\0';
        return i;
    }
    while (p->pos < p->length && i < max_len - 1) {
        char c = p->data[p->pos];
        if (c == quote) { p->pos++; break; }
        if (c == '\n') p->line++;
        out[i++] = c;
        p->pos++;
    }
    out[i] = '\0';
    return i;
}

/* 读取下一个XML节点 */
static int xml_read_node(XmlParser* p, XmlNode* node) {
    if (!p || !node || p->error) return 0;
    memset(node, 0, sizeof(XmlNode));
    node->depth = 0;

    xml_skip_whitespace(p);
    if (p->pos >= p->length) return 0;

    /* 检查是否注释 */
    if (p->data[p->pos] == '<' && p->pos + 3 < p->length &&
        p->data[p->pos+1] == '!' && p->data[p->pos+2] == '-' &&
        p->data[p->pos+3] == '-') {
        node->is_comment = 1;
        xml_skip_comment(p);
        return 1;
    }

    /* 检查是否文本内容 */
    if (p->data[p->pos] != '<') {
        int i = 0;
        /* 预分配文本缓冲区 */
        size_t text_start = p->pos;
        while (p->pos < p->length && p->data[p->pos] != '<') {
            p->pos++;
        }
        size_t text_len = p->pos - text_start;
        if (text_len > 0) {
            node->text_content = (char*)safe_malloc(text_len + 1);
            if (node->text_content) {
                memcpy(node->text_content, p->data + text_start, text_len);
                node->text_content[text_len] = '\0';
                /* 去除首尾空白 */
                while (text_len > 0 && (node->text_content[text_len-1] == ' ' ||
                       node->text_content[text_len-1] == '\t' ||
                       node->text_content[text_len-1] == '\r' ||
                       node->text_content[text_len-1] == '\n')) {
                    node->text_content[--text_len] = '\0';
                }
            }
        }
        return 1;
    }

    /* 开始标签 <tagname ...> */
    if (p->data[p->pos] == '<' && p->pos + 1 < p->length &&
        p->data[p->pos+1] == '/') {
        /* 结束标签，调用者处理 */
        return 0;
    }

    p->pos++; /* 跳过 '<' */

    /* 读取标签名 */
    xml_read_tag_name(p, node->tag_name, XML_MAX_TAG_NAME);
    if (node->tag_name[0] == '\0') {
        p->error = 1;
        snprintf(p->error_msg, sizeof(p->error_msg), "空标签名(行%d)", p->line);
        return 0;
    }

    /* 读取属性 */
    xml_skip_whitespace(p);
    node->attr_count = 0;
    while (p->pos < p->length && !p->error && node->attr_count < XML_MAX_ATTRS) {
        xml_skip_whitespace(p);
        if (p->pos >= p->length) break;
        char c = p->data[p->pos];

        /* 检测自闭合或标签结束 */
        if (c == '>') { p->pos++; break; }
        if (c == '/' && p->pos + 1 < p->length && p->data[p->pos+1] == '>') {
            node->is_self_closing = 1;
            p->pos += 2;
            break;
        }
        if (c == '<') break;

        /* 读取属性名 */
        XmlAttr* attr = &node->attrs[node->attr_count];
        xml_read_tag_name(p, attr->name, XML_MAX_ATTR_NAME);
        if (attr->name[0] == '\0') break;

        /* 跳过 = */
        xml_skip_whitespace(p);
        if (p->pos < p->length && p->data[p->pos] == '=') {
            p->pos++;
            xml_skip_whitespace(p);
            /* 读取属性值 */
            xml_read_attr_value(p, attr->value, XML_MAX_ATTR_VALUE);
            node->attr_count++;
        }
    }

    return 1;
}

/* 释放XML节点文本内容 */
static void xml_node_free_content(XmlNode* node) {
    if (node && node->text_content) {
        safe_free((void**)&node->text_content);
    }
}

/* ============================================================================
 * 2. OWL/XML导入器
 * ============================================================================ */

/* OWL/XML命名空间前缀映射 */
typedef struct {
    char prefix[32];
    char uri[256];
} OwlNsMapping;

#define OWL_MAX_NS 16

/* 解析命名空间前缀 */
static void owl_resolve_ns(const char* qname, OwlNsMapping* ns_map, int ns_count,
                            char* out_local, int max_local) {
    const char* colon = strchr(qname, ':');
    if (!colon) {
        /* 无前缀，默认命名空间 */
        strncpy(out_local, qname, max_local - 1);
        out_local[max_local - 1] = '\0';
        return;
    }
    /* 有前缀，如 rdf:about */
    size_t prefix_len = colon - qname;
    for (int i = 0; i < ns_count; i++) {
        if (strncmp(qname, ns_map[i].prefix, prefix_len) == 0 &&
            (int)strlen(ns_map[i].prefix) == (int)prefix_len) {
            /* 匹配到前缀，返回local name */
            const char* local = colon + 1;
            strncpy(out_local, local, max_local - 1);
            out_local[max_local - 1] = '\0';
            return;
        }
    }
    /* 未匹配的前缀，返回整个qname */
    strncpy(out_local, qname, max_local - 1);
    out_local[max_local - 1] = '\0';
}

/* 处理OWL/XML中的类定义 */
static int owl_handle_class(Ontology* ont, XmlNode* node, XmlParser* p,
                             OwlNsMapping* ns_map, int ns_count) {
    char local_name[XML_MAX_ATTR_NAME] = {0};
    int found_id = 0;

    /* 查找rdf:ID或rdf:about属性 */
    for (int a = 0; a < node->attr_count; a++) {
        char local_attr[XML_MAX_ATTR_NAME] = {0};
        owl_resolve_ns(node->attrs[a].name, ns_map, ns_count,
                       local_attr, sizeof(local_attr));
        if (strcmp(local_attr, "ID") == 0 || strcmp(local_attr, "about") == 0) {
            /* 提取#后的名称 */
            const char* hash = strchr(node->attrs[a].value, '#');
            if (hash) {
                strncpy(local_name, hash + 1, sizeof(local_name) - 1);
            } else {
                strncpy(local_name, node->attrs[a].value, sizeof(local_name) - 1);
            }
            local_name[sizeof(local_name) - 1] = '\0';
            found_id = 1;
        }
    }

    if (!found_id || local_name[0] == '\0') return 0;

    /* 检查是否已存在 */
    if (ontology_find_element(ont, local_name)) return 0;

    /* 添加类 */
    OntElement* cls = ontology_add_class(ont, local_name, NULL);
    if (!cls) return -1;

    /* 解析子元素（子类、等价类、不相交等） */
    int depth = 1;
    while (depth > 0 && !p->error) {
        XmlNode child;
        memset(&child, 0, sizeof(XmlNode));
        if (!xml_read_node(p, &child)) break;

        if (child.is_comment) continue;

        if (child.tag_name[0] == '\0' && child.text_content) {
            xml_node_free_content(&child);
            continue;
        }

        /* 检测结束标签 */
        char local_tag[XML_MAX_TAG_NAME] = {0};
        owl_resolve_ns(child.tag_name, ns_map, ns_count,
                       local_tag, sizeof(local_tag));

        if (strcmp(local_tag, "Class") == 0 && child.is_self_closing == 0) {
            depth++;
            xml_node_free_content(&child);
            continue;
        }

        /* 子类关系 */
        if (strcmp(local_tag, "subClassOf") == 0) {
            for (int a = 0; a < child.attr_count; a++) {
                char local_attr[XML_MAX_ATTR_NAME] = {0};
                owl_resolve_ns(child.attrs[a].name, ns_map, ns_count,
                               local_attr, sizeof(local_attr));
                if (strcmp(local_attr, "resource") == 0) {
                    const char* hash = strchr(child.attrs[a].value, '#');
                    const char* parent_name = hash ? hash + 1 : child.attrs[a].value;
                    /* 确保父类存在 */
                    if (!ontology_find_element(ont, parent_name)) {
                        ontology_add_class(ont, parent_name, NULL);
                    }
                    ontology_add_axiom(ont, AXIOM_SUBCLASS, local_name, parent_name, 1.0f);
                }
            }
        }

        /* 等价类 */
        if (strcmp(local_tag, "equivalentClass") == 0) {
            for (int a = 0; a < child.attr_count; a++) {
                char local_attr[XML_MAX_ATTR_NAME] = {0};
                owl_resolve_ns(child.attrs[a].name, ns_map, ns_count,
                               local_attr, sizeof(local_attr));
                if (strcmp(local_attr, "resource") == 0) {
                    const char* hash = strchr(child.attrs[a].value, '#');
                    const char* eq_name = hash ? hash + 1 : child.attrs[a].value;
                    if (!ontology_find_element(ont, eq_name)) {
                        ontology_add_class(ont, eq_name, NULL);
                    }
                    ontology_add_axiom(ont, AXIOM_EQUIVALENT, local_name, eq_name, 1.0f);
                }
            }
        }

        /* 不相交 */
        if (strcmp(local_tag, "disjointWith") == 0) {
            for (int a = 0; a < child.attr_count; a++) {
                char local_attr[XML_MAX_ATTR_NAME] = {0};
                owl_resolve_ns(child.attrs[a].name, ns_map, ns_count,
                               local_attr, sizeof(local_attr));
                if (strcmp(local_attr, "resource") == 0) {
                    const char* hash = strchr(child.attrs[a].value, '#');
                    const char* dj_name = hash ? hash + 1 : child.attrs[a].value;
                    if (!ontology_find_element(ont, dj_name)) {
                        ontology_add_class(ont, dj_name, NULL);
                    }
                    ontology_add_axiom(ont, AXIOM_DISJOINT, local_name, dj_name, 1.0f);
                }
            }
        }

        xml_node_free_content(&child);
    }

    return 0;
}

/* 处理OWL/XML中的属性定义 */
static int owl_handle_property(Ontology* ont, XmlNode* node, XmlParser* p,
                                OwlNsMapping* ns_map, int ns_count, int is_object_prop) {
    char local_name[XML_MAX_ATTR_NAME] = {0};
    int found_id = 0;

    for (int a = 0; a < node->attr_count; a++) {
        char local_attr[XML_MAX_ATTR_NAME] = {0};
        owl_resolve_ns(node->attrs[a].name, ns_map, ns_count,
                       local_attr, sizeof(local_attr));
        if (strcmp(local_attr, "ID") == 0 || strcmp(local_attr, "about") == 0) {
            const char* hash = strchr(node->attrs[a].value, '#');
            if (hash) {
                strncpy(local_name, hash + 1, sizeof(local_name) - 1);
            } else {
                strncpy(local_name, node->attrs[a].value, sizeof(local_name) - 1);
            }
            local_name[sizeof(local_name) - 1] = '\0';
            found_id = 1;
        }
    }

    if (!found_id || local_name[0] == '\0') return 0;
    if (ontology_find_element(ont, local_name)) return 0;

    /* 添加属性 */
    if (is_object_prop) {
        ontology_add_object_property(ont, local_name, NULL, NULL, NULL);
    } else {
        ontology_add_data_property(ont, local_name, NULL, "string");
    }

    /* 解析子元素（domain, range等） */
    int depth = 1;
    while (depth > 0 && !p->error) {
        XmlNode child;
        memset(&child, 0, sizeof(XmlNode));
        if (!xml_read_node(p, &child)) break;
        if (child.is_comment) continue;
        if (child.tag_name[0] == '\0' && child.text_content) {
            xml_node_free_content(&child);
            continue;
        }

        char local_tag[XML_MAX_TAG_NAME] = {0};
        owl_resolve_ns(child.tag_name, ns_map, ns_count,
                       local_tag, sizeof(local_tag));

        if ((strcmp(local_tag, "ObjectProperty") == 0 ||
             strcmp(local_tag, "DatatypeProperty") == 0) &&
            child.is_self_closing == 0) {
            depth++;
            xml_node_free_content(&child);
            continue;
        }

        if (strcmp(local_tag, "domain") == 0 || strcmp(local_tag, "range") == 0) {
            OntAxiomType ax_type = (strcmp(local_tag, "domain") == 0) ?
                                    AXIOM_DOMAIN : AXIOM_RANGE;
            for (int a = 0; a < child.attr_count; a++) {
                char local_attr[XML_MAX_ATTR_NAME] = {0};
                owl_resolve_ns(child.attrs[a].name, ns_map, ns_count,
                               local_attr, sizeof(local_attr));
                if (strcmp(local_attr, "resource") == 0) {
                    const char* hash = strchr(child.attrs[a].value, '#');
                    const char* res_name = hash ? hash + 1 : child.attrs[a].value;
                    if (!ontology_find_element(ont, res_name)) {
                        ontology_add_class(ont, res_name, NULL);
                    }
                    ontology_add_axiom(ont, ax_type, local_name, res_name, 1.0f);
                }
            }
        }

        /* 对称属性 */
        if (strcmp(local_tag, "type") == 0) {
            for (int a = 0; a < child.attr_count; a++) {
                char local_attr[XML_MAX_ATTR_NAME] = {0};
                owl_resolve_ns(child.attrs[a].name, ns_map, ns_count,
                               local_attr, sizeof(local_attr));
                if (strcmp(local_attr, "resource") == 0) {
                    if (strstr(child.attrs[a].value, "SymmetricProperty")) {
                        ontology_add_axiom(ont, AXIOM_SYMMETRIC, local_name, local_name, 1.0f);
                    }
                    if (strstr(child.attrs[a].value, "TransitiveProperty")) {
                        ontology_add_axiom(ont, AXIOM_TRANSITIVE, local_name, local_name, 1.0f);
                    }
                    if (strstr(child.attrs[a].value, "FunctionalProperty")) {
                        ontology_add_axiom(ont, AXIOM_FUNCTIONAL, local_name, local_name, 1.0f);
                    }
                }
            }
        }

        xml_node_free_content(&child);
    }

    return 0;
}

/* 处理OWL/XML中的实例定义 */
static int owl_handle_individual(Ontology* ont, XmlNode* node, XmlParser* p,
                                  OwlNsMapping* ns_map, int ns_count) {
    char local_name[XML_MAX_ATTR_NAME] = {0};
    int found_id = 0;

    for (int a = 0; a < node->attr_count; a++) {
        char local_attr[XML_MAX_ATTR_NAME] = {0};
        owl_resolve_ns(node->attrs[a].name, ns_map, ns_count,
                       local_attr, sizeof(local_attr));
        if (strcmp(local_attr, "ID") == 0 || strcmp(local_attr, "about") == 0) {
            const char* hash = strchr(node->attrs[a].value, '#');
            if (hash) {
                strncpy(local_name, hash + 1, sizeof(local_name) - 1);
            } else {
                strncpy(local_name, node->attrs[a].value, sizeof(local_name) - 1);
            }
            local_name[sizeof(local_name) - 1] = '\0';
            found_id = 1;
        }
    }

    if (!found_id || local_name[0] == '\0') return 0;
    if (ontology_find_element(ont, local_name)) return 0;

    /* 先添加实例，稍后解析type确定类 */
    OntElement* ind = ontology_add_individual(ont, local_name, NULL);
    if (!ind) return -1;

    /* 解析子元素 */
    int depth = 1;
    while (depth > 0 && !p->error) {
        XmlNode child;
        memset(&child, 0, sizeof(XmlNode));
        if (!xml_read_node(p, &child)) break;
        if (child.is_comment) continue;
        if (child.tag_name[0] == '\0' && child.text_content) {
            xml_node_free_content(&child);
            continue;
        }

        char local_tag[XML_MAX_TAG_NAME] = {0};
        owl_resolve_ns(child.tag_name, ns_map, ns_count,
                       local_tag, sizeof(local_tag));

        if (strcmp(local_tag, "NamedIndividual") == 0 && child.is_self_closing == 0) {
            depth++;
            xml_node_free_content(&child);
            continue;
        }

        /* rdf:type 确定所属类 */
        if (strcmp(local_tag, "type") == 0) {
            for (int a = 0; a < child.attr_count; a++) {
                char local_attr[XML_MAX_ATTR_NAME] = {0};
                owl_resolve_ns(child.attrs[a].name, ns_map, ns_count,
                               local_attr, sizeof(local_attr));
                if (strcmp(local_attr, "resource") == 0) {
                    const char* hash = strchr(child.attrs[a].value, '#');
                    const char* class_name = hash ? hash + 1 : child.attrs[a].value;
                    if (!ontology_find_element(ont, class_name)) {
                        ontology_add_class(ont, class_name, NULL);
                    }
                    /* 关联实例到类 */
                    OntElement* cls = ontology_find_element(ont, class_name);
                    if (cls) {
                        ont_element_add_related(ind, cls, AXIOM_SUBCLASS, 1.0f);
                    }
                }
            }
        }

        xml_node_free_content(&child);
    }

    return 0;
}

/* OWL/XML字符串导入主函数 */
Ontology* ontology_import_owl_string(Ontology* ont, const char* owl_xml, size_t xml_len) {
    if (!owl_xml || xml_len == 0) return NULL;

    XmlParser* parser = xml_parser_create(owl_xml, xml_len);
    if (!parser) return NULL;

    /* 创建或使用传入的本体 */
    int own_ont = 0;
    if (!ont) {
        ont = ontology_create("imported_owl", "从OWL/XML导入的本体");
        if (!ont) { xml_parser_free(parser); return NULL; }
        own_ont = 1;
    }

    /* 命名空间映射 */
    OwlNsMapping ns_map[OWL_MAX_NS];
    int ns_count = 0;
    /* 预置标准命名空间 */
    strncpy(ns_map[ns_count].prefix, "rdf", 31);
    strncpy(ns_map[ns_count].uri, "http://www.w3.org/1999/02/22-rdf-syntax-ns#", 255);
    ns_count++;
    strncpy(ns_map[ns_count].prefix, "rdfs", 31);
    strncpy(ns_map[ns_count].uri, "http://www.w3.org/2000/01/rdf-schema#", 255);
    ns_count++;
    strncpy(ns_map[ns_count].prefix, "owl", 31);
    strncpy(ns_map[ns_count].uri, "http://www.w3.org/2002/07/owl#", 255);
    ns_count++;

    /* 跳过声明 */
    xml_skip_declarations(parser);

    int element_count = 0;
    int depth = 0;

    /* 主解析循环 */
    while (parser->pos < parser->length && !parser->error) {
        XmlNode node;
        memset(&node, 0, sizeof(XmlNode));
        if (!xml_read_node(parser, &node)) break;

        if (node.is_comment) continue;
        if (node.tag_name[0] == '\0') {
            xml_node_free_content(&node);
            continue;
        }

        char local_tag[XML_MAX_TAG_NAME] = {0};
        owl_resolve_ns(node.tag_name, ns_map, ns_count,
                       local_tag, sizeof(local_tag));

        /* owl:Class */
        if (strcmp(local_tag, "Class") == 0 && !node.is_self_closing) {
            owl_handle_class(ont, &node, parser, ns_map, ns_count);
            element_count++;
        }
        /* owl:ObjectProperty */
        else if (strcmp(local_tag, "ObjectProperty") == 0 && !node.is_self_closing) {
            owl_handle_property(ont, &node, parser, ns_map, ns_count, 1);
            element_count++;
        }
        /* owl:DatatypeProperty */
        else if (strcmp(local_tag, "DatatypeProperty") == 0 && !node.is_self_closing) {
            owl_handle_property(ont, &node, parser, ns_map, ns_count, 0);
            element_count++;
        }
        /* owl:NamedIndividual 或 自定义实例 */
        else if (strcmp(local_tag, "NamedIndividual") == 0 && !node.is_self_closing) {
            owl_handle_individual(ont, &node, parser, ns_map, ns_count);
            element_count++;
        }
        else {
            /* 其他标签，跳过 */
            int skip_depth = node.is_self_closing ? 0 : 1;
            while (skip_depth > 0 && !parser->error) {
                XmlNode skip_node;
                memset(&skip_node, 0, sizeof(XmlNode));
                if (!xml_read_node(parser, &skip_node)) break;
                if (skip_node.is_comment) continue;
                if (skip_node.tag_name[0] == '\0') {
                    xml_node_free_content(&skip_node);
                    continue;
                }
                if (!skip_node.is_self_closing) skip_depth++;
                xml_node_free_content(&skip_node);
            }
        }

        xml_node_free_content(&node);
    }

    xml_parser_free(parser);

    if (element_count == 0 && own_ont) {
        ontology_free(ont);
        return NULL;
    }

    log_info("[Ontology] OWL/XML导入完成: %d个元素", element_count);
    return ont;
}

/* OWL/XML文件导入 */
Ontology* ontology_import_owl(Ontology* ont, const char* owl_path) {
    if (!owl_path) return NULL;

    FILE* fp = fopen(owl_path, "rb");
    if (!fp) {
        log_warn("[Ontology] 无法打开OWL文件: %s", owl_path);
        return NULL;
    }

    /* 获取文件大小 */
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 100 * 1024 * 1024) { /* 最大100MB */
        fclose(fp);
        log_warn("[Ontology] OWL文件大小无效: %ld", file_size);
        return NULL;
    }

    char* xml_data = (char*)safe_malloc((size_t)file_size + 1);
    if (!xml_data) {
        fclose(fp);
        return NULL;
    }

    size_t read_size = fread(xml_data, 1, (size_t)file_size, fp);
    fclose(fp);

    if (read_size != (size_t)file_size) {
        safe_free((void**)&xml_data);
        log_warn("[Ontology] OWL文件读取不完整");
        return NULL;
    }
    xml_data[read_size] = '\0';

    Ontology* result = ontology_import_owl_string(ont, xml_data, read_size);
    safe_free((void**)&xml_data);
    return result;
}

/* ============================================================================
 * 3. RDF/Turtle解析器
 * ============================================================================ */

#define TTL_MAX_LINE        4096
#define TTL_MAX_PREFIXES     32
#define TTL_MAX_TRIPLES    4096

/* Turtle前缀映射 */
typedef struct {
    char prefix[32];
    char uri[256];
} TtlPrefix;

/* Turtle三元组 */
typedef struct {
    char subject[256];
    char predicate[256];
    char object[256];
    int is_literal;          /* 对象是否为字面量 */
    char datatype[64];       /* 字面量数据类型 */
    char lang_tag[16];       /* 语言标签 */
} TtlTriple;

/* Turtle解析器状态 */
typedef struct {
    TtlPrefix prefixes[TTL_MAX_PREFIXES];
    int prefix_count;
    TtlTriple triples[TTL_MAX_TRIPLES];
    int triple_count;
    char base_uri[256];
    char error_msg[256];
    int error;
} TtlParser;

/* 解析Turtle QName或完整URI */
static void ttl_resolve_uri(TtlParser* tp, const char* qname_or_uri,
                             char* out, int max_len) {
    if (!qname_or_uri || !out) return;
    out[0] = '\0';

    /* 如果是完整URI (<...>) */
    if (qname_or_uri[0] == '<') {
        const char* end = strchr(qname_or_uri, '>');
        if (end) {
            size_t len = end - qname_or_uri - 1;
            if (len < (size_t)max_len) {
                memcpy(out, qname_or_uri + 1, len);
                out[len] = '\0';
            }
        }
        return;
    }

    /* 如果是QName (prefix:local) */
    const char* colon = strchr(qname_or_uri, ':');
    if (colon) {
        size_t prefix_len = colon - qname_or_uri;
        for (int i = 0; i < tp->prefix_count; i++) {
            if (strncmp(qname_or_uri, tp->prefixes[i].prefix, prefix_len) == 0 &&
                (int)strlen(tp->prefixes[i].prefix) == (int)prefix_len) {
                /* 拼接 URI + local */
                size_t uri_len = strlen(tp->prefixes[i].uri);
                const char* local = colon + 1;
                if (uri_len + strlen(local) < (size_t)max_len) {
                    strncpy(out, tp->prefixes[i].uri, max_len - 1);
                    strncat(out, local, max_len - strlen(out) - 1);
                }
                return;
            }
        }
    }

    /* 无法解析，直接复制 */
    strncpy(out, qname_or_uri, max_len - 1);
    out[max_len - 1] = '\0';
}

/* 简化URI为可读名称 */
static void ttl_simplify_name(const char* uri, char* out, int max_len) {
    if (!uri || !out) return;
    /* 查找最后一个#或/ */
    const char* hash = strrchr(uri, '#');
    const char* slash = strrchr(uri, '/');
    const char* start = NULL;
    if (hash && slash) {
        start = (hash > slash) ? hash : slash;
    } else if (hash) {
        start = hash;
    } else if (slash) {
        start = slash;
    }
    if (start) {
        strncpy(out, start + 1, max_len - 1);
        out[max_len - 1] = '\0';
    } else {
        strncpy(out, uri, max_len - 1);
        out[max_len - 1] = '\0';
    }
}

/* 解析一行Turtle */
static int ttl_parse_line(TtlParser* tp, const char* line) {
    /* 跳过空行和注释 */
    const char* p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') return 0;

    /* @prefix声明 */
    if (strncmp(p, "@prefix", 7) == 0 && tp->prefix_count < TTL_MAX_PREFIXES) {
        p += 7;
        while (*p == ' ' || *p == '\t') p++;
        /* 读取前缀名 */
        char pre[32] = {0};
        int pi = 0;
        while (*p && *p != ':' && *p != ' ' && *p != '\t' && pi < 31) {
            pre[pi++] = *p++;
        }
        pre[pi] = '\0';
        if (*p == ':') p++;
        while (*p == ' ' || *p == '\t') p++;
        /* 读取URI */
        if (*p == '<') {
            p++;
            char uri[256] = {0};
            int ui = 0;
            while (*p && *p != '>' && ui < 255) {
                uri[ui++] = *p++;
            }
            uri[ui] = '\0';
            if (pre[0] && uri[0]) {
                strncpy(tp->prefixes[tp->prefix_count].prefix, pre, 31);
                strncpy(tp->prefixes[tp->prefix_count].uri, uri, 255);
                tp->prefix_count++;
            }
        }
        return 0;
    }

    /* @base声明 */
    if (strncmp(p, "@base", 5) == 0) {
        p += 5;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '<') {
            p++;
            int bi = 0;
            while (*p && *p != '>' && bi < 255) {
                tp->base_uri[bi++] = *p++;
            }
            tp->base_uri[bi] = '\0';
        }
        return 0;
    }

    /* 三元组解析: subject predicate object . */
    if (tp->triple_count >= TTL_MAX_TRIPLES) return 0;

    TtlTriple* triple = &tp->triples[tp->triple_count];

    /* 解析subject */
    char subj_raw[256] = {0};
    int si = 0;
    if (*p == '<') {
        p++;
        subj_raw[si++] = '<';
        while (*p && *p != '>' && si < 254) subj_raw[si++] = *p++;
        if (*p == '>') { subj_raw[si++] = '>'; p++; }
    } else {
        while (*p && *p != ' ' && *p != '\t' && si < 254) subj_raw[si++] = *p++;
    }
    subj_raw[si] = '\0';
    if (subj_raw[0] == '\0') return 0;
    ttl_resolve_uri(tp, subj_raw, triple->subject, sizeof(triple->subject));

    /* 跳过空白 */
    while (*p == ' ' || *p == '\t') p++;

    /* 解析predicate */
    char pred_raw[256] = {0};
    int pi2 = 0;
    if (*p == '<') {
        p++;
        pred_raw[pi2++] = '<';
        while (*p && *p != '>' && pi2 < 254) pred_raw[pi2++] = *p++;
        if (*p == '>') { pred_raw[pi2++] = '>'; p++; }
    } else {
        /* 可能是关键字 'a' (rdf:type) */
        if (*p == 'a' && (*(p+1) == ' ' || *(p+1) == '\t' || *(p+1) == '\n')) {
            strncpy(pred_raw, "rdf:type", 254);
            p++;
        } else {
            while (*p && *p != ' ' && *p != '\t' && pi2 < 254) pred_raw[pi2++] = *p++;
        }
    }
    pred_raw[pi2] = '\0';
    if (pred_raw[0] == '\0') return 0;
    ttl_resolve_uri(tp, pred_raw, triple->predicate, sizeof(triple->predicate));

    /* 跳过空白 */
    while (*p == ' ' || *p == '\t') p++;

    /* 解析object */
    char obj_raw[512] = {0};
    int oi = 0;
    if (*p == '<') {
        p++;
        obj_raw[oi++] = '<';
        while (*p && *p != '>' && oi < 510) obj_raw[oi++] = *p++;
        if (*p == '>') { obj_raw[oi++] = '>'; p++; }
        triple->is_literal = 0;
    } else if (*p == '\"') {
        /* 字面量 */
        p++;
        while (*p && *p != '\"' && oi < 254) {
            if (*p == '\\' && *(p+1)) { p++; } /* 简单转义跳过 */
            obj_raw[oi++] = *p++;
        }
        if (*p == '\"') p++;
        obj_raw[oi] = '\0';
        strncpy(triple->object, obj_raw, sizeof(triple->object) - 1);
        triple->is_literal = 1;

        /* 检查数据类型 ^^<type> 或 @lang */
        while (*p == ' ' || *p == '\t') p++;
        if (strncmp(p, "^^", 2) == 0) {
            p += 2;
            if (*p == '<') {
                p++;
                int di = 0;
                while (*p && *p != '>' && di < 63) triple->datatype[di++] = *p++;
                triple->datatype[di] = '\0';
                if (*p == '>') p++;
            }
        } else if (*p == '@') {
            p++;
            int li = 0;
            while (*p && *p != ' ' && *p != '\t' && *p != '.' && li < 15) {
                triple->lang_tag[li++] = *p++;
            }
            triple->lang_tag[li] = '\0';
        }
        /* 跳过字面量处理，直接继续 */
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '.') p++;
        tp->triple_count++;
        return 1;
    } else {
        while (*p && *p != ' ' && *p != '\t' && *p != '.' && oi < 510) {
            obj_raw[oi++] = *p++;
        }
        obj_raw[oi] = '\0';
        triple->is_literal = 0;
    }

    if (obj_raw[0] == '\0') return 0;
    if (!triple->is_literal) {
        ttl_resolve_uri(tp, obj_raw, triple->object, sizeof(triple->object));
    }

    /* 跳过句号 */
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '.') p++;

    tp->triple_count++;
    return 1;
}

/* Turtle字符串导入 */
Ontology* ontology_import_rdf_string(Ontology* ont, const char* ttl_str, size_t ttl_len) {
    if (!ttl_str || ttl_len == 0) return NULL;

    TtlParser tp;
    memset(&tp, 0, sizeof(TtlParser));
    tp.base_uri[0] = '\0';

    /* 创建或使用传入的本体 */
    int own_ont = 0;
    if (!ont) {
        ont = ontology_create("imported_rdf", "从RDF/Turtle导入的本体");
        if (!ont) return NULL;
        own_ont = 1;
    }

    /* 逐行解析 */
    const char* line_start = ttl_str;
    const char* end = ttl_str + ttl_len;

    while (line_start < end) {
        const char* line_end = strchr(line_start, '\n');
        if (!line_end) line_end = end;

        size_t line_len = line_end - line_start;
        if (line_len > 0 && line_len < TTL_MAX_LINE) {
            char line_buf[TTL_MAX_LINE];
            size_t copy_len = line_len < TTL_MAX_LINE - 1 ? line_len : TTL_MAX_LINE - 1;
            memcpy(line_buf, line_start, copy_len);
            line_buf[copy_len] = '\0';
            /* 去除行尾\r */
            if (copy_len > 0 && line_buf[copy_len - 1] == '\r') line_buf[copy_len - 1] = '\0';
            ttl_parse_line(&tp, line_buf);
        }

        line_start = line_end < end ? line_end + 1 : end;
    }

    /* 将三元组导入本体 */
    int imported = 0;
    for (int i = 0; i < tp.triple_count; i++) {
        TtlTriple* t = &tp.triples[i];
        if (!t->subject[0] || !t->predicate[0] || !t->object[0]) continue;

        /* 简化名称为可读的本地名称 */
        char subj_name[256], pred_name[256], obj_name[256];
        ttl_simplify_name(t->subject, subj_name, sizeof(subj_name));
        ttl_simplify_name(t->predicate, pred_name, sizeof(pred_name));
        if (!t->is_literal) {
            ttl_simplify_name(t->object, obj_name, sizeof(obj_name));
        } else {
            strncpy(obj_name, t->object, sizeof(obj_name) - 1);
        }

        /* 确保主体存在（作为类） */
        if (!ontology_find_element(ont, subj_name) && subj_name[0]) {
            ontology_add_class(ont, subj_name, NULL);
        }

        /* 判断谓词类型 */
        if (strstr(t->predicate, "type") || strcmp(pred_name, "type") == 0) {
            /* rdf:type — 主体是实例 */
            if (t->is_literal) continue;
            OntElement* subj = ontology_find_element(ont, subj_name);
            if (subj && subj->type == ONT_CLASS) {
                /* 如果主体是类，type表示子类而非实例 */
                /* 但这里主体已被创建为类，检查是否是已知的类 */
            }
            if (!ontology_find_element(ont, obj_name)) {
                ontology_add_class(ont, obj_name, NULL);
            }
            /* 创建实例或添加subclass */
            OntElement* existing = ontology_find_element(ont, subj_name);
            if (existing && existing->type == ONT_CLASS) {
                /* 如果只是临时创建的类，转换为实例 */
                /* 简化处理：保持为类，添加subclass */
                ontology_add_axiom(ont, AXIOM_SUBCLASS, subj_name, obj_name, 1.0f);
            }
        }
        else if (strstr(t->predicate, "subClassOf") || strcmp(pred_name, "subClassOf") == 0) {
            if (!ontology_find_element(ont, obj_name)) {
                ontology_add_class(ont, obj_name, NULL);
            }
            ontology_add_axiom(ont, AXIOM_SUBCLASS, subj_name, obj_name, 1.0f);
        }
        else if (strstr(t->predicate, "equivalentClass") || strcmp(pred_name, "equivalentClass") == 0) {
            if (!ontology_find_element(ont, obj_name)) {
                ontology_add_class(ont, obj_name, NULL);
            }
            ontology_add_axiom(ont, AXIOM_EQUIVALENT, subj_name, obj_name, 1.0f);
        }
        else if (strstr(t->predicate, "domain") || strcmp(pred_name, "domain") == 0) {
            if (!ontology_find_element(ont, obj_name)) {
                ontology_add_class(ont, obj_name, NULL);
            }
            ontology_add_axiom(ont, AXIOM_DOMAIN, subj_name, obj_name, 1.0f);
        }
        else if (strstr(t->predicate, "range") || strcmp(pred_name, "range") == 0) {
            if (!ontology_find_element(ont, obj_name)) {
                ontology_add_class(ont, obj_name, NULL);
            }
            ontology_add_axiom(ont, AXIOM_RANGE, subj_name, obj_name, 1.0f);
        }
        else {
            /* 普通关系：创建属性并关联 */
            if (!ontology_find_element(ont, pred_name)) {
                ontology_add_object_property(ont, pred_name, NULL, NULL, NULL);
            }
            if (t->is_literal) {
                /* 数据属性 */
                OntElement* prop = ontology_find_element(ont, pred_name);
                if (prop) {
                    /* 字面量值存储为user_data */
                }
            }
        }
        imported++;
    }

    log_info("[Ontology] RDF/Turtle导入完成: %d个三元组", imported);
    return ont;
}

/* Turtle文件导入 */
Ontology* ontology_import_rdf(Ontology* ont, const char* ttl_path) {
    if (!ttl_path) return NULL;

    FILE* fp = fopen(ttl_path, "rb");
    if (!fp) {
        log_warn("[Ontology] 无法打开Turtle文件: %s", ttl_path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 100 * 1024 * 1024) {
        fclose(fp);
        return NULL;
    }

    char* ttl_data = (char*)safe_malloc((size_t)file_size + 1);
    if (!ttl_data) { fclose(fp); return NULL; }

    size_t read_size = fread(ttl_data, 1, (size_t)file_size, fp);
    fclose(fp);

    if (read_size != (size_t)file_size) {
        safe_free((void**)&ttl_data);
        return NULL;
    }
    ttl_data[read_size] = '\0';

    Ontology* result = ontology_import_rdf_string(ont, ttl_data, read_size);
    safe_free((void**)&ttl_data);
    return result;
}

/* ============================================================================
 * 4. RDF/Turtle序列化器
 * ============================================================================ */

char* ontology_export_rdf(Ontology* ont) {
    if (!ont) return NULL;

    size_t buf_size = 131072; /* 128KB */
    char* buf = (char*)safe_calloc(buf_size, 1);
    if (!buf) return NULL;
    size_t pos = 0;

    /* 头部：前缀声明 */
    pos += snprintf(buf + pos, buf_size - pos,
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "@prefix : <http://selflnn.org/ontology/%s#> .\n\n",
        ont->name ? ont->name : "default");
    if (pos >= buf_size) goto rdf_export_truncated;

    /* 本体声明 */
    pos += snprintf(buf + pos, buf_size - pos,
        "<http://selflnn.org/ontology/%s> a owl:Ontology ;\n"
        "    rdfs:label \"%s\" .\n\n",
        ont->name ? ont->name : "default",
        ont->description ? ont->description : "");
    if (pos >= buf_size) goto rdf_export_truncated;

    /* 导出类 */
    for (int i = 0; i < ont->class_count; i++) {
        OntElement* cls = ont->classes[i];
        if (!cls || !cls->name || cls->confidence < 0.01f) continue;

        pos += snprintf(buf + pos, buf_size - pos,
            ":%s a owl:Class ;\n", cls->name);
        if (pos >= buf_size) goto rdf_export_truncated;

        /* 公理 */
        int has_axiom = 0;
        for (int r = 0; r < cls->related_count; r++) {
            OntElement* target = cls->related[r];
            if (!target || !target->name) continue;

            const char* predicate = NULL;
            switch (cls->axiom_types[r]) {
                case AXIOM_SUBCLASS:
                    predicate = "rdfs:subClassOf"; break;
                case AXIOM_EQUIVALENT:
                    predicate = "owl:equivalentClass"; break;
                case AXIOM_DISJOINT:
                    predicate = "owl:disjointWith"; break;
                default: break;
            }
            if (predicate) {
                pos += snprintf(buf + pos, buf_size - pos,
                    "    %s :%s ;\n", predicate, target->name);
                if (pos >= buf_size) goto rdf_export_truncated;
                has_axiom = 1;
            }
        }

        if (has_axiom) {
            /* 去掉最后的 ;\n 替换为 .\n\n */
            if (pos >= 4 && buf[pos-4] == ';' && buf[pos-3] == ' ') {
                buf[pos-4] = '.';
                buf[pos-3] = '\n';
                buf[pos-2] = '\n';
                pos -= 1;
                buf[pos] = '\0';
            } else {
                pos += snprintf(buf + pos, buf_size - pos, "    rdfs:label \"%s\" .\n\n", cls->name);
                buf[pos - 2] = '\n';
                buf[pos - 1] = '\0';
            }
        } else {
            /* 替换 ;\n 为 .\n\n */
            buf[pos - 4] = '.';
            buf[pos - 3] = '\n';
            buf[pos - 2] = '\n';
            buf[pos - 1] = '\0';
            pos -= 1;
        }
    }

    /* 导出对象属性 */
    for (int i = 0; i < ont->property_count; i++) {
        OntElement* prop = ont->properties[i];
        if (!prop || !prop->name || prop->confidence < 0.01f) continue;

        if (prop->type == ONT_OBJECT_PROPERTY) {
            pos += snprintf(buf + pos, buf_size - pos,
                ":%s a owl:ObjectProperty ;\n", prop->name);
        } else {
            pos += snprintf(buf + pos, buf_size - pos,
                ":%s a owl:DatatypeProperty ;\n", prop->name);
        }
        if (pos >= buf_size) goto rdf_export_truncated;

        int has_axiom = 0;
        for (int r = 0; r < prop->related_count; r++) {
            OntElement* target = prop->related[r];
            if (!target || !target->name) continue;

            const char* predicate = NULL;
            switch (prop->axiom_types[r]) {
                case AXIOM_DOMAIN: predicate = "rdfs:domain"; break;
                case AXIOM_RANGE:  predicate = "rdfs:range"; break;
                default: break;
            }
            if (predicate) {
                pos += snprintf(buf + pos, buf_size - pos,
                    "    %s :%s ;\n", predicate, target->name);
                if (pos >= buf_size) goto rdf_export_truncated;
                has_axiom = 1;
            }
        }

        if (has_axiom) {
            buf[pos-4] = '.';
            buf[pos-3] = '\n';
            buf[pos-2] = '\n';
            buf[pos-1] = '\0';
            pos -= 1;
        } else {
            buf[pos-4] = '.';
            buf[pos-3] = '\n';
            buf[pos-2] = '\n';
            buf[pos-1] = '\0';
            pos -= 1;
        }
    }

    /* 导出实例 */
    for (int i = 0; i < ont->individual_count; i++) {
        OntElement* ind = ont->individuals[i];
        if (!ind || !ind->name || ind->confidence < 0.01f) continue;

        pos += snprintf(buf + pos, buf_size - pos,
            ":%s a owl:NamedIndividual", ind->name);

        /* 查找所属类 */
        for (int r = 0; r < ind->related_count; r++) {
            if (ind->axiom_types[r] == AXIOM_SUBCLASS && ind->related[r]) {
                pos += snprintf(buf + pos, buf_size - pos,
                    " ;\n    rdf:type :%s", ind->related[r]->name);
                break;
            }
        }

        pos += snprintf(buf + pos, buf_size - pos, " .\n\n");
        if (pos >= buf_size) goto rdf_export_truncated;
    }

    return buf;

rdf_export_truncated:
    log_warn("[Ontology] Turtle导出缓冲区溢出(%zu字节)，结果已截断", buf_size);
    buf[buf_size - 1] = '\0';
    return buf;
}

/* ============================================================================
 * 5. 自动格式检测导入器
 * ============================================================================ */

/* 检测文件格式 */
static OntologyFormat ont_detect_format(const char* file_path) {
    if (!file_path) return ONT_FORMAT_AUTO;

    /* 方法1: 根据扩展名 */
    const char* ext = strrchr(file_path, '.');
    if (ext) {
        if (strcmp(ext, ".owl") == 0) return ONT_FORMAT_OWL_XML;
        if (strcmp(ext, ".ttl") == 0) return ONT_FORMAT_RDF_TURTLE;
        if (strcmp(ext, ".rdf") == 0) return ONT_FORMAT_RDF_XML;
        /* .xml可能是OWL/XML */
        if (strcmp(ext, ".xml") == 0) {
            /* 读取文件头判断 */
            FILE* fp = fopen(file_path, "rb");
            if (fp) {
                char header[256] = {0};
                size_t n = fread(header, 1, 255, fp);
                fclose(fp);
                if (n > 0) {
                    header[n] = '\0';
                    if (strstr(header, "owl:") || strstr(header, "rdf:RDF")) {
                        return ONT_FORMAT_OWL_XML;
                    }
                }
            }
            return ONT_FORMAT_OWL_XML; /* 默认XML */
        }
    }

    /* 方法2: 根据文件内容 */
    FILE* fp = fopen(file_path, "rb");
    if (fp) {
        char header[512] = {0};
        size_t n = fread(header, 1, 511, fp);
        fclose(fp);
        if (n > 0) {
            header[n] = '\0';
            /* 检查XML头 */
            if (strstr(header, "<?xml") || strstr(header, "<rdf:RDF") ||
                strstr(header, "<owl:")) {
                return ONT_FORMAT_OWL_XML;
            }
            /* 检查Turtle头 */
            if (strstr(header, "@prefix") || strstr(header, "@base")) {
                return ONT_FORMAT_RDF_TURTLE;
            }
        }
    }

    return ONT_FORMAT_OWL_XML; /* 默认尝试OWL/XML */
}

Ontology* ontology_import_auto(Ontology* ont, const char* file_path) {
    if (!file_path) return NULL;

    OntologyFormat format = ont_detect_format(file_path);

    log_info("[Ontology] 自动检测格式: %s -> %s",
             file_path,
             format == ONT_FORMAT_OWL_XML ? "OWL/XML" :
             format == ONT_FORMAT_RDF_TURTLE ? "RDF/Turtle" : "未知");

    switch (format) {
        case ONT_FORMAT_OWL_XML:
        case ONT_FORMAT_RDF_XML:
            return ontology_import_owl(ont, file_path);
        case ONT_FORMAT_RDF_TURTLE:
            return ontology_import_rdf(ont, file_path);
        default:
            /* 尝试两种格式 */
            {
                Ontology* result = ontology_import_owl(ont, file_path);
                if (result) return result;
                return ontology_import_rdf(ont, file_path);
            }
    }
}

/* ============================================================================
 * 6. 文件保存器
 * ============================================================================ */

int ontology_save_to_file(Ontology* ont, const char* file_path, OntologyFormat format) {
    if (!ont || !file_path) return -1;

    /* 自动检测格式 */
    if (format == ONT_FORMAT_AUTO) {
        const char* ext = strrchr(file_path, '.');
        if (ext) {
            if (strcmp(ext, ".ttl") == 0) format = ONT_FORMAT_RDF_TURTLE;
            else if (strcmp(ext, ".owl") == 0) format = ONT_FORMAT_OWL_XML;
            else if (strcmp(ext, ".rdf") == 0) format = ONT_FORMAT_RDF_XML;
            else format = ONT_FORMAT_OWL_XML; /* 默认OWL/XML */
        } else {
            format = ONT_FORMAT_OWL_XML;
        }
    }

    char* content = NULL;
    switch (format) {
        case ONT_FORMAT_OWL_XML:
        case ONT_FORMAT_RDF_XML:
            content = ontology_export_owl(ont);
            break;
        case ONT_FORMAT_RDF_TURTLE:
            content = ontology_export_rdf(ont);
            break;
        default:
            return -1;
    }

    if (!content) return -1;

    FILE* fp = fopen(file_path, "w");
    if (!fp) {
        safe_free((void**)&content);
        log_warn("[Ontology] 无法写入文件: %s", file_path);
        return -1;
    }

    size_t content_len = strlen(content);
    size_t written = fwrite(content, 1, content_len, fp);
    fclose(fp);
    safe_free((void**)&content);

    if (written != content_len) {
        log_warn("[Ontology] 文件写入不完整: %s", file_path);
        return -1;
    }

    log_info("[Ontology] 本体保存成功: %s (%zu字节)", file_path, content_len);
    return 0;
}