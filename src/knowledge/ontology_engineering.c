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
                if (entry->old_value) sscanf(entry->old_value, "%255[^,],%255s", domain, range);
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
                sscanf(lcopy, "%*c:%d:", &elem_id);
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
                                ontology_add_axiom(new_ont, (OntAxiomType)ax_type, subj, obj, 1.0f);
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

    pos += snprintf(buf + pos, buf_size - pos,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"\n"
        "         xmlns:rdfs=\"http://www.w3.org/2000/01/rdf-schema#\"\n"
        "         xmlns:owl=\"http://www.w3.org/2002/07/owl#\"\n"
        "         xmlns:onto=\"#\">\n\n"
        "<owl:Ontology rdf:about=\"%s\"/>\n\n",
        ont->name ? ont->name : "unknown");

    /* 导出类 */
    for (int i = 0; i < ont->class_count; i++) {
        OntElement* cls = ont->classes[i];
        pos += snprintf(buf + pos, buf_size - pos,
            "<owl:Class rdf:ID=\"%s\"/>\n", cls->name ? cls->name : "");

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
#ifdef _WIN32
static CRITICAL_SECTION g_onto_cc_lock;
static int g_onto_cc_lock_init = 0;
static void onto_cc_lock_init(void) {
    if (!g_onto_cc_lock_init) {
        InitializeCriticalSection(&g_onto_cc_lock);
        g_onto_cc_lock_init = 1;
    }
}
#define ONTO_CC_LOCK() do { onto_cc_lock_init(); EnterCriticalSection(&g_onto_cc_lock); } while(0)
#define ONTO_CC_UNLOCK() LeaveCriticalSection(&g_onto_cc_lock)
#else
#include <pthread.h>
static pthread_mutex_t g_onto_cc_lock = PTHREAD_MUTEX_INITIALIZER;
#define ONTO_CC_LOCK() pthread_mutex_lock(&g_onto_cc_lock)
#define ONTO_CC_UNLOCK() pthread_mutex_unlock(&g_onto_cc_lock)
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
