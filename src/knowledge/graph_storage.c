/**
 * @file graph_storage.c
 * @brief 知识图谱存储引擎实现
 *
 * 三种互补的图存储模型完整实现：
 * 1. 邻接表存储（AdjacencyList）
 * 2. 属性图存储（PropertyGraph）
 * 3. RDF三元组存储（RDFTripleStore）
 */

#include "selflnn/knowledge/graph_storage.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ============================================================================
 * 内部辅助函数
 * =========================================================================== */

static char* graph_strdup(const char* str) {
    if (!str) return NULL;
    size_t len = strlen(str) + 1;
    char* copy = (char*)safe_malloc(len);
    if (copy) memcpy(copy, str, len);
    return copy;
}

static int expand_array(void** array, size_t* capacity, size_t element_size,
                        size_t min_needed) {
    if (!array || !capacity) return -1;
    size_t new_cap = (*capacity == 0) ? 16 : *capacity;
    while (new_cap < min_needed) new_cap *= 2;
    void* new_arr = safe_realloc(*array, new_cap * element_size);
    if (!new_arr) return -1;
    *array = new_arr;
    *capacity = new_cap;
    return 0;
}

static int ensure_capacity(void** array, size_t* capacity, size_t element_size,
                           size_t index) {
    if (index >= *capacity) {
        return expand_array(array, capacity, element_size, index + 1);
    }
    return 0;
}

/* ============================================================================
 * 属性值操作
 * =========================================================================== */

void property_value_free(PropertyValue* value) {
    if (!value) return;
    if (value->type == PROP_TYPE_STRING && value->str_val) {
        safe_free((void**)&value->str_val);
    }
    value->type = PROP_TYPE_NONE;
    memset(&value->data, 0, sizeof(value->data));
    value->str_val = NULL;
}

static PropertyValue property_value_copy(const PropertyValue* src) {
    PropertyValue dst;
    dst.type = src->type;
    dst.data = src->data;
    dst.str_val = NULL;
    if (src->type == PROP_TYPE_STRING && src->str_val) {
        dst.str_val = graph_strdup(src->str_val);
    }
    return dst;
}

static void property_list_init(PropertyList* pl) {
    pl->items = NULL;
    pl->count = 0;
    pl->capacity = 0;
}

static void property_list_free(PropertyList* pl) {
    if (!pl) return;
    for (size_t i = 0; i < pl->count; i++) {
        safe_free((void**)&pl->items[i].key);
        property_value_free(&pl->items[i].value);
    }
    safe_free((void**)&pl->items);
    pl->items = NULL;
    pl->count = 0;
    pl->capacity = 0;
}

static int property_list_set(PropertyList* pl, const char* key,
                             const PropertyValue* value) {
    for (size_t i = 0; i < pl->count; i++) {
        if (strcmp(pl->items[i].key, key) == 0) {
            property_value_free(&pl->items[i].value);
            pl->items[i].value = property_value_copy(value);
            return 0;
        }
    }
    if (pl->count >= pl->capacity) {
        size_t new_cap = pl->capacity == 0 ? 8 : pl->capacity * 2;
        Property* new_items = (Property*)safe_realloc(
            pl->items, new_cap * sizeof(Property));
        if (!new_items) return -1;
        pl->items = new_items;
        pl->capacity = new_cap;
    }
    pl->items[pl->count].key = graph_strdup(key);
    pl->items[pl->count].value = property_value_copy(value);
    pl->count++;
    return 0;
}

static int property_list_get(const PropertyList* pl, const char* key,
                             PropertyValue* out) {
    for (size_t i = 0; i < pl->count; i++) {
        if (strcmp(pl->items[i].key, key) == 0) {
            *out = property_value_copy(&pl->items[i].value);
            return 0;
        }
    }
    return 1;
}

/* ============================================================================
 * 邻接表存储引擎
 * =========================================================================== */

struct AdjacencyList {
    ALNode** nodes;
    size_t node_capacity;
    size_t node_count;
    ALEdge** edges;
    size_t edge_capacity;
    size_t edge_count;
    int next_node_id;
    int next_edge_id;
};

AdjacencyList* adjacency_list_create(size_t initial_node_capacity) {
    AdjacencyList* al = (AdjacencyList*)safe_malloc(sizeof(AdjacencyList));
    if (!al) return NULL;
    al->node_capacity = initial_node_capacity > 0 ? initial_node_capacity : 64;
    al->nodes = (ALNode**)safe_calloc(al->node_capacity, sizeof(ALNode*));
    if (!al->nodes) {
        safe_free((void**)&al);
        return NULL;
    }
    al->node_count = 0;
    al->edge_capacity = 64;
    al->edges = (ALEdge**)safe_calloc(al->edge_capacity, sizeof(ALEdge*));
    if (!al->edges) {
        safe_free((void**)&al->nodes);
        safe_free((void**)&al);
        return NULL;
    }
    al->edge_count = 0;
    al->next_node_id = 0;
    al->next_edge_id = 0;
    return al;
}

static void al_free_node(ALNode* node) {
    if (!node) return;
    safe_free((void**)&node->label);
    safe_free((void**)&node->out_neighbors);
    safe_free((void**)&node->out_weights);
    safe_free((void**)&node->out_edge_ids);
    safe_free((void**)&node->in_neighbors);
    safe_free((void**)&node->in_weights);
    safe_free((void**)&node->in_edge_ids);
    safe_free((void**)&node);
}

void adjacency_list_free(AdjacencyList* al) {
    if (!al) return;
    for (size_t i = 0; i < al->node_capacity; i++) {
        if (al->nodes[i]) al_free_node(al->nodes[i]);
    }
    safe_free((void**)&al->nodes);
    for (size_t i = 0; i < al->edge_count; i++) {
        if (al->edges[i]) {
            safe_free((void**)&al->edges[i]->label);
            safe_free((void**)&al->edges[i]);
        }
    }
    safe_free((void**)&al->edges);
    safe_free((void**)&al);
}

static int al_ensure_node(AdjacencyList* al, int node_id) {
    if (node_id < 0) return -1;
    if ((size_t)node_id >= al->node_capacity) {
        size_t new_cap = al->node_capacity;
        while (new_cap <= (size_t)node_id) new_cap *= 2;
        ALNode** new_nodes = (ALNode**)safe_realloc(
            al->nodes, new_cap * sizeof(ALNode*));
        if (!new_nodes) return -1;
        memset(new_nodes + al->node_capacity, 0,
               (new_cap - al->node_capacity) * sizeof(ALNode*));
        al->nodes = new_nodes;
        al->node_capacity = new_cap;
    }
    return 0;
}

int adjacency_list_add_node(AdjacencyList* al, const char* label) {
    if (!al) return -1;
    int node_id = al->next_node_id++;
    if (al_ensure_node(al, node_id) != 0) return -1;
    ALNode* node = (ALNode*)safe_calloc(1, sizeof(ALNode));
    if (!node) return -1;
    node->id = node_id;
    node->label = graph_strdup(label);
    node->out_capacity = 8;
    node->out_neighbors = (int*)safe_malloc(8 * sizeof(int));
    node->out_weights = (float*)safe_malloc(8 * sizeof(float));
    node->out_edge_ids = (int*)safe_malloc(8 * sizeof(int));
    if (!node->out_neighbors || !node->out_weights || !node->out_edge_ids) {
        al_free_node(node);
        return -1;
    }
    node->in_capacity = 8;
    node->in_neighbors = (int*)safe_malloc(8 * sizeof(int));
    node->in_weights = (float*)safe_malloc(8 * sizeof(float));
    node->in_edge_ids = (int*)safe_malloc(8 * sizeof(int));
    if (!node->in_neighbors || !node->in_weights || !node->in_edge_ids) {
        al_free_node(node);
        return -1;
    }
    al->nodes[node_id] = node;
    al->node_count++;
    return node_id;
}

int adjacency_list_add_edge(AdjacencyList* al, int source_id, int target_id,
                            const char* label, float weight) {
    if (!al) return -1;
    if (source_id < 0 || target_id < 0) return -1;
    if ((size_t)source_id >= al->node_capacity || !al->nodes[source_id]) return -1;
    if ((size_t)target_id >= al->node_capacity || !al->nodes[target_id]) return -1;

    int edge_id = al->next_edge_id++;
    if (al->edge_count >= al->edge_capacity) {
        size_t new_cap = al->edge_capacity * 2;
        ALEdge** new_edges = (ALEdge**)safe_realloc(
            al->edges, new_cap * sizeof(ALEdge*));
        if (!new_edges) return -1;
        al->edges = new_edges;
        al->edge_capacity = new_cap;
    }
    ALEdge* edge = (ALEdge*)safe_malloc(sizeof(ALEdge));
    if (!edge) return -1;
    edge->id = edge_id;
    edge->source_id = source_id;
    edge->target_id = target_id;
    edge->label = graph_strdup(label);
    edge->weight = weight;
    edge->directed = 1;
    al->edges[al->edge_count++] = edge;

    ALNode* src = al->nodes[source_id];
    if (src->out_degree >= src->out_capacity) {
        size_t new_cap = src->out_capacity * 2;
        int* new_nb = (int*)safe_realloc(src->out_neighbors, new_cap * sizeof(int));
        float* new_w = (float*)safe_realloc(src->out_weights, new_cap * sizeof(float));
        int* new_eid = (int*)safe_realloc(src->out_edge_ids, new_cap * sizeof(int));
        if (!new_nb || !new_w || !new_eid) return -1;
        src->out_neighbors = new_nb;
        src->out_weights = new_w;
        src->out_edge_ids = new_eid;
        src->out_capacity = new_cap;
    }
    src->out_neighbors[src->out_degree] = target_id;
    src->out_weights[src->out_degree] = weight;
    src->out_edge_ids[src->out_degree] = edge_id;
    src->out_degree++;

    ALNode* tgt = al->nodes[target_id];
    if (tgt->in_degree >= tgt->in_capacity) {
        size_t new_cap = tgt->in_capacity * 2;
        int* new_nb = (int*)safe_realloc(tgt->in_neighbors, new_cap * sizeof(int));
        float* new_w = (float*)safe_realloc(tgt->in_weights, new_cap * sizeof(float));
        int* new_eid = (int*)safe_realloc(tgt->in_edge_ids, new_cap * sizeof(int));
        if (!new_nb || !new_w || !new_eid) return -1;
        tgt->in_neighbors = new_nb;
        tgt->in_weights = new_w;
        tgt->in_edge_ids = new_eid;
        tgt->in_capacity = new_cap;
    }
    tgt->in_neighbors[tgt->in_degree] = source_id;
    tgt->in_weights[tgt->in_degree] = weight;
    tgt->in_edge_ids[tgt->in_degree] = edge_id;
    tgt->in_degree++;

    return edge_id;
}

int adjacency_list_add_undirected_edge(AdjacencyList* al, int node_a, int node_b,
                                       const char* label, float weight) {
    int eid = adjacency_list_add_edge(al, node_a, node_b, label, weight);
    if (eid < 0) return -1;
    /* 标记为无向边 */
    if (eid >= 0 && al->edges && (size_t)eid < al->edge_count) {
        if (al->edges[eid]) al->edges[eid]->directed = 0;
    }
    return eid;
}

int adjacency_list_get_out_neighbors(AdjacencyList* al, int node_id,
                                     int* neighbors, float* weights,
                                     size_t max_count) {
    if (!al || !neighbors) return -1;
    if (node_id < 0 || (size_t)node_id >= al->node_capacity) return -1;
    ALNode* node = al->nodes[node_id];
    if (!node) return -1;
    size_t count = node->out_degree < max_count ? node->out_degree : max_count;
    memcpy(neighbors, node->out_neighbors, count * sizeof(int));
    if (weights) {
        memcpy(weights, node->out_weights, count * sizeof(float));
    }
    return (int)count;
}

int adjacency_list_get_in_neighbors(AdjacencyList* al, int node_id,
                                    int* neighbors, float* weights,
                                    size_t max_count) {
    if (!al || !neighbors) return -1;
    if (node_id < 0 || (size_t)node_id >= al->node_capacity) return -1;
    ALNode* node = al->nodes[node_id];
    if (!node) return -1;
    size_t count = node->in_degree < max_count ? node->in_degree : max_count;
    memcpy(neighbors, node->in_neighbors, count * sizeof(int));
    if (weights) {
        memcpy(weights, node->in_weights, count * sizeof(float));
    }
    return (int)count;
}

int adjacency_list_get_degree(AdjacencyList* al, int node_id,
                              size_t* out_degree, size_t* in_degree) {
    if (!al || !out_degree || !in_degree) return -1;
    if (node_id < 0 || (size_t)node_id >= al->node_capacity) return -1;
    ALNode* node = al->nodes[node_id];
    if (!node) return -1;
    *out_degree = node->out_degree;
    *in_degree = node->in_degree;
    return 0;
}

size_t adjacency_list_node_count(AdjacencyList* al) {
    return al ? al->node_count : 0;
}

size_t adjacency_list_edge_count(AdjacencyList* al) {
    return al ? al->edge_count : 0;
}

int adjacency_list_find_by_label(AdjacencyList* al, const char* label,
                                 int* results, size_t max_results) {
    if (!al || !results || !label) return -1;
    int count = 0;
    for (size_t i = 0; i < al->node_capacity && (size_t)count < max_results; i++) {
        if (al->nodes[i] && al->nodes[i]->label &&
            strcmp(al->nodes[i]->label, label) == 0) {
            results[count++] = al->nodes[i]->id;
        }
    }
    return count;
}

int adjacency_list_dfs(AdjacencyList* al, int start_id,
                       int (*callback)(int, void*), void* user_data) {
    if (!al || !callback) return -1;
    if (start_id < 0 || (size_t)start_id >= al->node_capacity) return -1;
    if (!al->nodes[start_id]) return -1;

    size_t max_nodes = al->node_capacity;
    int* visited = (int*)safe_calloc(max_nodes, sizeof(int));
    if (!visited) return -1;

    int* stack = (int*)safe_malloc(max_nodes * sizeof(int));
    if (!stack) {
        safe_free((void**)&visited);
        return -1;
    }
    int stack_top = 0;
    stack[stack_top++] = start_id;

    while (stack_top > 0) {
        int cur = stack[--stack_top];
        if (visited[cur]) continue;
        visited[cur] = 1;
        if (callback(cur, user_data) != 0) break;

        ALNode* node = al->nodes[cur];
        for (size_t i = 0; i < node->out_degree; i++) {
            int nb = node->out_neighbors[i];
            if (!visited[nb]) {
                if ((size_t)stack_top >= max_nodes) break;
                stack[stack_top++] = nb;
            }
        }
    }

    safe_free((void**)&visited);
    safe_free((void**)&stack);
    return 0;
}

int adjacency_list_bfs(AdjacencyList* al, int start_id,
                       int (*callback)(int, void*), void* user_data) {
    if (!al || !callback) return -1;
    if (start_id < 0 || (size_t)start_id >= al->node_capacity) return -1;
    if (!al->nodes[start_id]) return -1;

    size_t max_nodes = al->node_capacity;
    int* visited = (int*)safe_calloc(max_nodes, sizeof(int));
    if (!visited) return -1;

    int* queue = (int*)safe_malloc(max_nodes * sizeof(int));
    if (!queue) {
        safe_free((void**)&visited);
        return -1;
    }
    int q_head = 0, q_tail = 0;
    queue[q_tail++] = start_id;
    visited[start_id] = 1;

    while (q_head < q_tail) {
        int cur = queue[q_head++];
        if (callback(cur, user_data) != 0) break;

        ALNode* node = al->nodes[cur];
        for (size_t i = 0; i < node->out_degree; i++) {
            int nb = node->out_neighbors[i];
            if (!visited[nb] && (size_t)q_tail < max_nodes) {
                visited[nb] = 1;
                queue[q_tail++] = nb;
            }
        }
    }

    safe_free((void**)&visited);
    safe_free((void**)&queue);
    return 0;
}

int adjacency_list_has_edge(AdjacencyList* al, int source_id, int target_id) {
    if (!al) return -1;
    if (source_id < 0 || (size_t)source_id >= al->node_capacity) return -1;
    ALNode* node = al->nodes[source_id];
    if (!node) return -1;
    for (size_t i = 0; i < node->out_degree; i++) {
        if (node->out_neighbors[i] == target_id) return 1;
    }
    return 0;
}

const ALEdge* adjacency_list_get_edge_by_id(AdjacencyList* al, int edge_id) {
    if (!al) return NULL;
    if (edge_id < 0 || (size_t)edge_id >= al->edge_capacity) return NULL;
    return al->edges[edge_id];
}

/* ============================================================================
 * 属性图存储引擎
 * =========================================================================== */

struct PropertyGraph {
    PGNode** nodes;
    size_t node_capacity;
    size_t node_count;
    PGEdge** edges;
    size_t edge_capacity;
    size_t edge_count;
    int next_node_id;
    int next_edge_id;
};

PropertyGraph* property_graph_create(size_t initial_capacity) {
    PropertyGraph* pg = (PropertyGraph*)safe_malloc(sizeof(PropertyGraph));
    if (!pg) return NULL;
    size_t cap = initial_capacity > 0 ? initial_capacity : 64;
    pg->nodes = (PGNode**)safe_calloc(cap, sizeof(PGNode*));
    if (!pg->nodes) {
        safe_free((void**)&pg);
        return NULL;
    }
    pg->node_capacity = cap;
    pg->node_count = 0;
    pg->edge_capacity = 64;
    pg->edges = (PGEdge**)safe_calloc(pg->edge_capacity, sizeof(PGEdge*));
    if (!pg->edges) {
        safe_free((void**)&pg->nodes);
        safe_free((void**)&pg);
        return NULL;
    }
    pg->edge_count = 0;
    pg->next_node_id = 0;
    pg->next_edge_id = 0;
    return pg;
}

static void pg_free_node(PGNode* node) {
    if (!node) return;
    safe_free((void**)&node->label);
    property_list_free(&node->properties);
    safe_free((void**)&node);
}

static void pg_free_edge(PGEdge* edge) {
    if (!edge) return;
    safe_free((void**)&edge->label);
    property_list_free(&edge->properties);
    safe_free((void**)&edge);
}

void property_graph_free(PropertyGraph* pg) {
    if (!pg) return;
    for (size_t i = 0; i < pg->node_capacity; i++) {
        if (pg->nodes[i]) pg_free_node(pg->nodes[i]);
    }
    safe_free((void**)&pg->nodes);
    for (size_t i = 0; i < pg->edge_count; i++) {
        if (pg->edges[i]) pg_free_edge(pg->edges[i]);
    }
    safe_free((void**)&pg->edges);
    safe_free((void**)&pg);
}

int property_graph_add_node(PropertyGraph* pg, const char* label) {
    if (!pg) return -1;
    int node_id = pg->next_node_id++;
    if ((size_t)node_id >= pg->node_capacity) {
        size_t new_cap = pg->node_capacity * 2;
        PGNode** new_nodes = (PGNode**)safe_realloc(
            pg->nodes, new_cap * sizeof(PGNode*));
        if (!new_nodes) return -1;
        memset(new_nodes + pg->node_capacity, 0,
               (new_cap - pg->node_capacity) * sizeof(PGNode*));
        pg->nodes = new_nodes;
        pg->node_capacity = new_cap;
    }
    PGNode* node = (PGNode*)safe_malloc(sizeof(PGNode));
    if (!node) return -1;
    node->id = node_id;
    node->label = graph_strdup(label);
    property_list_init(&node->properties);
    pg->nodes[node_id] = node;
    pg->node_count++;
    return node_id;
}

int property_graph_add_edge(PropertyGraph* pg, int source_id, int target_id,
                            const char* label, float weight, int directed) {
    if (!pg) return -1;
    if (source_id < 0 || (size_t)source_id >= pg->node_capacity) return -1;
    if (target_id < 0 || (size_t)target_id >= pg->node_capacity) return -1;
    if (!pg->nodes[source_id] || !pg->nodes[target_id]) return -1;

    int edge_id = pg->next_edge_id++;
    if (pg->edge_count >= pg->edge_capacity) {
        size_t new_cap = pg->edge_capacity * 2;
        PGEdge** new_edges = (PGEdge**)safe_realloc(
            pg->edges, new_cap * sizeof(PGEdge*));
        if (!new_edges) return -1;
        pg->edges = new_edges;
        pg->edge_capacity = new_cap;
    }
    PGEdge* edge = (PGEdge*)safe_malloc(sizeof(PGEdge));
    if (!edge) return -1;
    edge->id = edge_id;
    edge->source_id = source_id;
    edge->target_id = target_id;
    edge->label = graph_strdup(label);
    edge->weight = weight;
    edge->directed = directed;
    property_list_init(&edge->properties);
    pg->edges[pg->edge_count++] = edge;
    return edge_id;
}

static int pg_set_property(PropertyList* pl, const char* key,
                           PropertyType type, const void* val) {
    PropertyValue pv;
    pv.type = type;
    pv.str_val = NULL;
    memset(&pv.data, 0, sizeof(pv.data));
    switch (type) {
        case PROP_TYPE_INT:
            pv.data.int_val = *(const int64_t*)val;
            break;
        case PROP_TYPE_FLOAT:
            pv.data.float_val = *(const double*)val;
            break;
        case PROP_TYPE_STRING:
            pv.str_val = graph_strdup((const char*)val);
            break;
        case PROP_TYPE_BOOL:
            pv.data.bool_val = *(const int*)val;
            break;
        default:
            return -1;
    }
    return property_list_set(pl, key, &pv);
}

int property_graph_set_node_property_int(PropertyGraph* pg, int node_id,
                                         const char* key, int64_t value) {
    if (!pg || !key) return -1;
    if (node_id < 0 || (size_t)node_id >= pg->node_capacity) return -1;
    if (!pg->nodes[node_id]) return -1;
    return pg_set_property(&pg->nodes[node_id]->properties, key,
                           PROP_TYPE_INT, &value);
}

int property_graph_set_node_property_float(PropertyGraph* pg, int node_id,
                                           const char* key, double value) {
    if (!pg || !key) return -1;
    if (node_id < 0 || (size_t)node_id >= pg->node_capacity) return -1;
    if (!pg->nodes[node_id]) return -1;
    return pg_set_property(&pg->nodes[node_id]->properties, key,
                           PROP_TYPE_FLOAT, &value);
}

int property_graph_set_node_property_string(PropertyGraph* pg, int node_id,
                                            const char* key, const char* value) {
    if (!pg || !key) return -1;
    if (node_id < 0 || (size_t)node_id >= pg->node_capacity) return -1;
    if (!pg->nodes[node_id]) return -1;
    return pg_set_property(&pg->nodes[node_id]->properties, key,
                           PROP_TYPE_STRING, value);
}

int property_graph_get_node_property(PropertyGraph* pg, int node_id,
                                     const char* key, PropertyValue* value) {
    if (!pg || !key || !value) return -1;
    if (node_id < 0 || (size_t)node_id >= pg->node_capacity) return -1;
    if (!pg->nodes[node_id]) return -1;
    return property_list_get(&pg->nodes[node_id]->properties, key, value);
}

int property_graph_set_edge_property_int(PropertyGraph* pg, int edge_id,
                                         const char* key, int64_t value) {
    if (!pg || !key) return -1;
    if (edge_id < 0 || (size_t)edge_id >= pg->edge_count) return -1;
    if (!pg->edges[edge_id]) return -1;
    return pg_set_property(&pg->edges[edge_id]->properties, key,
                           PROP_TYPE_INT, &value);
}

int property_graph_set_edge_property_float(PropertyGraph* pg, int edge_id,
                                           const char* key, double value) {
    if (!pg || !key) return -1;
    if (edge_id < 0 || (size_t)edge_id >= pg->edge_count) return -1;
    if (!pg->edges[edge_id]) return -1;
    return pg_set_property(&pg->edges[edge_id]->properties, key,
                           PROP_TYPE_FLOAT, &value);
}

int property_graph_set_edge_property_string(PropertyGraph* pg, int edge_id,
                                            const char* key, const char* value) {
    if (!pg || !key) return -1;
    if (edge_id < 0 || (size_t)edge_id >= pg->edge_count) return -1;
    if (!pg->edges[edge_id]) return -1;
    return pg_set_property(&pg->edges[edge_id]->properties, key,
                           PROP_TYPE_STRING, value);
}

int property_graph_get_edge_property(PropertyGraph* pg, int edge_id,
                                     const char* key, PropertyValue* value) {
    if (!pg || !key || !value) return -1;
    if (edge_id < 0 || (size_t)edge_id >= pg->edge_count) return -1;
    if (!pg->edges[edge_id]) return -1;
    return property_list_get(&pg->edges[edge_id]->properties, key, value);
}

int property_graph_find_nodes_by_property(PropertyGraph* pg, const char* key,
                                          const PropertyValue* value,
                                          int* results, size_t max_results) {
    if (!pg || !key || !value || !results) return -1;
    int count = 0;
    for (size_t i = 0; i < pg->node_capacity && (size_t)count < max_results; i++) {
        if (!pg->nodes[i]) continue;
        PropertyValue pv;
        if (property_list_get(&pg->nodes[i]->properties, key, &pv) == 0) {
            int match = 0;
            if (pv.type == value->type) {
                switch (pv.type) {
                    case PROP_TYPE_INT:
                        match = (pv.data.int_val == value->data.int_val);
                        break;
                    case PROP_TYPE_FLOAT:
                        match = (fabs(pv.data.float_val - value->data.float_val) < 1e-10);
                        break;
                    case PROP_TYPE_STRING:
                        match = (pv.str_val && value->str_val &&
                                 strcmp(pv.str_val, value->str_val) == 0);
                        break;
                    case PROP_TYPE_BOOL:
                        match = (pv.data.bool_val == value->data.bool_val);
                        break;
                    default:
                        break;
                }
            }
            property_value_free(&pv);
            if (match) {
                results[count++] = pg->nodes[i]->id;
            }
        }
    }
    return count;
}

int property_graph_find_nodes_by_label(PropertyGraph* pg, const char* label,
                                       int* results, size_t max_results) {
    if (!pg || !label || !results) return -1;
    int count = 0;
    for (size_t i = 0; i < pg->node_capacity && (size_t)count < max_results; i++) {
        if (pg->nodes[i] && pg->nodes[i]->label &&
            strcmp(pg->nodes[i]->label, label) == 0) {
            results[count++] = pg->nodes[i]->id;
        }
    }
    return count;
}

int property_graph_get_stats(PropertyGraph* pg, size_t* node_count,
                             size_t* edge_count) {
    if (!pg || !node_count || !edge_count) return -1;
    *node_count = pg->node_count;
    *edge_count = pg->edge_count;
    return 0;
}

int property_graph_save(PropertyGraph* pg, const char* filename) {
    if (!pg || !filename) return -1;
    FILE* fp = fopen(filename, "wb");
    if (!fp) return -1;

    uint32_t magic = 0x50524700;
    uint32_t version = 1;
    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);

    /* ZSFWS-M025修复: 保存实际存在的节点数（遍历capacity跳过空洞）
     * 防止节点ID不连续时丢失高ID节点 */
    uint32_t actual_count = 0;
    for (size_t i = 0; i < pg->node_capacity; i++) {
        if (pg->nodes[i]) actual_count++;
    }
    fwrite(&actual_count, sizeof(actual_count), 1, fp);
    for (size_t i = 0; i < pg->node_capacity; i++) {
        PGNode* n = pg->nodes[i];
        if (!n) continue;
        int32_t nid = (int32_t)n->id;
        fwrite(&nid, sizeof(nid), 1, fp);
        uint8_t has_label = (n->label != NULL) ? 1 : 0;
        fwrite(&has_label, 1, 1, fp);
        if (has_label) {
            uint32_t llen = (uint32_t)strlen(n->label) + 1;
            fwrite(&llen, sizeof(llen), 1, fp);
            fwrite(n->label, 1, llen, fp);
        }
        uint32_t pcount = (uint32_t)n->properties.count;
        fwrite(&pcount, sizeof(pcount), 1, fp);
        for (size_t j = 0; j < n->properties.count; j++) {
            Property* p = &n->properties.items[j];
            uint32_t klen = (uint32_t)strlen(p->key) + 1;
            fwrite(&klen, sizeof(klen), 1, fp);
            fwrite(p->key, 1, klen, fp);
            uint32_t ptype = (uint32_t)p->value.type;
            fwrite(&ptype, sizeof(ptype), 1, fp);
            switch (p->value.type) {
                case PROP_TYPE_INT:
                    fwrite(&p->value.data.int_val, sizeof(int64_t), 1, fp);
                    break;
                case PROP_TYPE_FLOAT:
                    fwrite(&p->value.data.float_val, sizeof(double), 1, fp);
                    break;
                case PROP_TYPE_BOOL:
                    fwrite(&p->value.data.bool_val, sizeof(int), 1, fp);
                    break;
                case PROP_TYPE_STRING:
                    if (p->value.str_val) {
                        uint32_t svlen = (uint32_t)strlen(p->value.str_val) + 1;
                        fwrite(&svlen, sizeof(svlen), 1, fp);
                        fwrite(p->value.str_val, 1, svlen, fp);
                    } else {
                        uint32_t svlen = 0;
                        fwrite(&svlen, sizeof(svlen), 1, fp);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    uint32_t ec = (uint32_t)pg->edge_count;
    fwrite(&ec, sizeof(ec), 1, fp);
    for (size_t i = 0; i < pg->edge_count; i++) {
        PGEdge* e = pg->edges[i];
        int32_t eid = (int32_t)e->id;
        int32_t src = (int32_t)e->source_id;
        int32_t tgt = (int32_t)e->target_id;
        fwrite(&eid, sizeof(eid), 1, fp);
        fwrite(&src, sizeof(src), 1, fp);
        fwrite(&tgt, sizeof(tgt), 1, fp);
        fwrite(&e->weight, sizeof(float), 1, fp);
        uint8_t dir = (uint8_t)e->directed;
        fwrite(&dir, 1, 1, fp);
        uint8_t has_label = (e->label != NULL) ? 1 : 0;
        fwrite(&has_label, 1, 1, fp);
        if (has_label) {
            uint32_t llen = (uint32_t)strlen(e->label) + 1;
            fwrite(&llen, sizeof(llen), 1, fp);
            fwrite(e->label, 1, llen, fp);
        }
        uint32_t pcount = (uint32_t)e->properties.count;
        fwrite(&pcount, sizeof(pcount), 1, fp);
        for (size_t j = 0; j < e->properties.count; j++) {
            Property* p = &e->properties.items[j];
            uint32_t klen = (uint32_t)strlen(p->key) + 1;
            fwrite(&klen, sizeof(klen), 1, fp);
            fwrite(p->key, 1, klen, fp);
            uint32_t ptype = (uint32_t)p->value.type;
            fwrite(&ptype, sizeof(ptype), 1, fp);
            switch (p->value.type) {
                case PROP_TYPE_INT:
                    fwrite(&p->value.data.int_val, sizeof(int64_t), 1, fp);
                    break;
                case PROP_TYPE_FLOAT:
                    fwrite(&p->value.data.float_val, sizeof(double), 1, fp);
                    break;
                case PROP_TYPE_BOOL:
                    fwrite(&p->value.data.bool_val, sizeof(int), 1, fp);
                    break;
                case PROP_TYPE_STRING:
                    if (p->value.str_val) {
                        uint32_t svlen = (uint32_t)strlen(p->value.str_val) + 1;
                        fwrite(&svlen, sizeof(svlen), 1, fp);
                        fwrite(p->value.str_val, 1, svlen, fp);
                    } else {
                        uint32_t svlen = 0;
                        fwrite(&svlen, sizeof(svlen), 1, fp);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    fclose(fp);
    return 0;
}

PropertyGraph* property_graph_load(const char* filename) {
    if (!filename) return NULL;
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;

    uint32_t magic, version;
    if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != 0x50524700) {
        fclose(fp);
        return NULL;
    }
    if (fread(&version, sizeof(version), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    PropertyGraph* pg = property_graph_create(64);
    if (!pg) { fclose(fp); return NULL; }

    uint32_t nc;
    if (fread(&nc, sizeof(nc), 1, fp) != 1) { fclose(fp); property_graph_free(pg); return NULL; }
    for (uint32_t i = 0; i < nc; i++) {
        int32_t nid;
        if (fread(&nid, sizeof(nid), 1, fp) != 1) break;
        char label_buf[256] = {0};
        uint8_t has_label;
        fread(&has_label, 1, 1, fp);
        if (has_label) {
            uint32_t llen;
            fread(&llen, sizeof(llen), 1, fp);
            if (llen > 0) {
                size_t read_len = llen < sizeof(label_buf) ? llen : sizeof(label_buf) - 1;
                fread(label_buf, 1, read_len, fp);
            }
        }
        int added_id = property_graph_add_node(pg, label_buf);
        if (added_id != nid) continue;
        uint32_t pcount;
        fread(&pcount, sizeof(pcount), 1, fp);
        for (uint32_t j = 0; j < pcount; j++) {
            char key_buf[128] = {0};
            uint32_t klen;
            fread(&klen, sizeof(klen), 1, fp);
            if (klen > 0) {
                size_t read_len = klen < sizeof(key_buf) ? klen : sizeof(key_buf) - 1;
                fread(key_buf, 1, read_len, fp);
            }
            uint32_t ptype;
            fread(&ptype, sizeof(ptype), 1, fp);
            switch ((PropertyType)ptype) {
                case PROP_TYPE_INT: {
                    int64_t val;
                    fread(&val, sizeof(val), 1, fp);
                    property_graph_set_node_property_int(pg, nid, key_buf, val);
                    break;
                }
                case PROP_TYPE_FLOAT: {
                    double val;
                    fread(&val, sizeof(val), 1, fp);
                    property_graph_set_node_property_float(pg, nid, key_buf, val);
                    break;
                }
                case PROP_TYPE_BOOL: {
                    int val;
                    fread(&val, sizeof(val), 1, fp);
                    PropertyValue pv;
                    pv.type = PROP_TYPE_BOOL;
                    pv.data.bool_val = val;
                    pv.str_val = NULL;
                    property_list_set(&pg->nodes[nid]->properties, key_buf, &pv);
                    break;
                }
                case PROP_TYPE_STRING: {
                    uint32_t svlen;
                    fread(&svlen, sizeof(svlen), 1, fp);
                    if (svlen > 0) {
                        char* sv = (char*)safe_malloc(svlen);
                        if (sv) {
                            fread(sv, 1, svlen - 1, fp);
                            sv[svlen - 1] = '\0';
                            property_graph_set_node_property_string(pg, nid, key_buf, sv);
                            safe_free((void**)&sv);
                        }
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    uint32_t ec;
    if (fread(&ec, sizeof(ec), 1, fp) != 1) { fclose(fp); return pg; }
    for (uint32_t i = 0; i < ec; i++) {
        int32_t eid, src, tgt;
        float w;
        uint8_t dir;
        if (fread(&eid, sizeof(eid), 1, fp) != 1) break;
        fread(&src, sizeof(src), 1, fp);
        fread(&tgt, sizeof(tgt), 1, fp);
        fread(&w, sizeof(w), 1, fp);
        fread(&dir, 1, 1, fp);
        char label_buf[256] = {0};
        uint8_t has_label;
        fread(&has_label, 1, 1, fp);
        if (has_label) {
            uint32_t llen;
            fread(&llen, sizeof(llen), 1, fp);
            if (llen > 0) {
                size_t read_len = llen < sizeof(label_buf) ? llen : sizeof(label_buf) - 1;
                fread(label_buf, 1, read_len, fp);
            }
        }
        int added_eid = property_graph_add_edge(pg, src, tgt, label_buf, w, dir);
        if (added_eid < 0) continue;
        uint32_t pcount;
        fread(&pcount, sizeof(pcount), 1, fp);
        for (uint32_t j = 0; j < pcount; j++) {
            char key_buf[128] = {0};
            uint32_t klen;
            fread(&klen, sizeof(klen), 1, fp);
            if (klen > 0) {
                size_t read_len = klen < sizeof(key_buf) ? klen : sizeof(key_buf) - 1;
                fread(key_buf, 1, read_len, fp);
            }
            uint32_t ptype;
            fread(&ptype, sizeof(ptype), 1, fp);
            switch ((PropertyType)ptype) {
                case PROP_TYPE_INT: {
                    int64_t val;
                    fread(&val, sizeof(val), 1, fp);
                    property_graph_set_edge_property_int(pg, added_eid, key_buf, val);
                    break;
                }
                case PROP_TYPE_FLOAT: {
                    double val;
                    fread(&val, sizeof(val), 1, fp);
                    property_graph_set_edge_property_float(pg, added_eid, key_buf, val);
                    break;
                }
                case PROP_TYPE_STRING: {
                    uint32_t svlen;
                    fread(&svlen, sizeof(svlen), 1, fp);
                    if (svlen > 0) {
                        char* sv = (char*)safe_malloc(svlen);
                        if (sv) {
                            fread(sv, 1, svlen - 1, fp);
                            sv[svlen - 1] = '\0';
                            property_graph_set_edge_property_string(pg, added_eid, key_buf, sv);
                            safe_free((void**)&sv);
                        }
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    fclose(fp);
    return pg;
}

const PGNode* property_graph_get_node(PropertyGraph* pg, int node_id) {
    if (!pg || node_id < 0 || (size_t)node_id >= pg->node_capacity)
        return NULL;
    return pg->nodes[node_id];
}

int property_graph_has_edge(PropertyGraph* pg, int src_node_id, int tgt_node_id) {
    if (!pg) return -1;
    for (size_t i = 0; i < pg->edge_count; i++) {
        if (pg->edges[i] &&
            ((pg->edges[i]->source_id == src_node_id &&
              pg->edges[i]->target_id == tgt_node_id) ||
             (!pg->edges[i]->directed &&
              pg->edges[i]->source_id == tgt_node_id &&
              pg->edges[i]->target_id == src_node_id))) {
            return 1;
        }
    }
    return 0;
}

const PGEdge* property_graph_get_edge_between(PropertyGraph* pg,
                                               int src_node_id,
                                               int tgt_node_id) {
    if (!pg) return NULL;
    for (size_t i = 0; i < pg->edge_count; i++) {
        if (pg->edges[i] &&
            ((pg->edges[i]->source_id == src_node_id &&
              pg->edges[i]->target_id == tgt_node_id) ||
             (!pg->edges[i]->directed &&
              pg->edges[i]->source_id == tgt_node_id &&
              pg->edges[i]->target_id == src_node_id))) {
            return pg->edges[i];
        }
    }
    return NULL;
}

const ALNode* adjacency_list_get_node(AdjacencyList* al, int node_id) {
    if (!al || node_id < 0 || (size_t)node_id >= al->node_capacity)
        return NULL;
    return al->nodes[node_id];
}

int adjacency_list_get_node_capacity(AdjacencyList* al) {
    if (!al) return 0;
    return (int)al->node_capacity;
}

int property_graph_get_node_capacity(PropertyGraph* pg) {
    if (!pg) return 0;
    return (int)pg->node_capacity;
}

/* F-047: 节点边度数计算 — VF2子图同构度数剪枝 */
int property_graph_node_degree(PropertyGraph* pg, int node_id) {
    if (!pg || node_id < 0) return -1;
    int degree = 0;
    for (size_t i = 0; i < pg->edge_count; i++) {
        if (pg->edges[i] && (pg->edges[i]->source_id == node_id ||
                              pg->edges[i]->target_id == node_id)) {
            degree++;
        }
    }
    return degree;
}

/* ============================================================================
 * RDF三元组存储引擎
 * ============================================================================ */

typedef struct {
    int subject_id;
    int predicate_id;
    int object_id;
} SPOEntry;

typedef struct {
    int subject_id;
    int predicate_id;
    int object_id;
} OSPEntry;

typedef struct {
    int predicate_id;
    int object_id;
    int subject_id;
} POSEntry;

struct RDFTripleStore {
    RDFNode* nodes;
    size_t node_capacity;
    size_t node_count;

    RDFTriple* triples;
    size_t triple_capacity;
    size_t triple_count;

    SPOEntry* spo_index;
    size_t spo_capacity;
    size_t spo_count;

    OSPEntry* osp_index;
    size_t osp_capacity;
    size_t osp_count;

    POSEntry* pos_index;
    size_t pos_capacity;
    size_t pos_count;

    RDFNamespace* namespaces;
    size_t namespace_capacity;
    size_t namespace_count;
};

RDFTripleStore* rdf_triple_store_create(void) {
    RDFTripleStore* store = (RDFTripleStore*)safe_malloc(sizeof(RDFTripleStore));
    if (!store) return NULL;

    store->node_capacity = 128;
    store->nodes = (RDFNode*)safe_calloc(store->node_capacity, sizeof(RDFNode));
    if (!store->nodes) { safe_free((void**)&store); return NULL; }
    store->node_count = 0;

    store->triple_capacity = 256;
    store->triples = (RDFTriple*)safe_calloc(store->triple_capacity, sizeof(RDFTriple));
    if (!store->triples) { safe_free((void**)&store->nodes); safe_free((void**)&store); return NULL; }
    store->triple_count = 0;

    store->spo_capacity = 256;
    store->spo_index = (SPOEntry*)safe_malloc(store->spo_capacity * sizeof(SPOEntry));
    store->spo_count = 0;

    store->osp_capacity = 256;
    store->osp_index = (OSPEntry*)safe_malloc(store->osp_capacity * sizeof(OSPEntry));
    store->osp_count = 0;

    store->pos_capacity = 256;
    store->pos_index = (POSEntry*)safe_malloc(store->pos_capacity * sizeof(POSEntry));
    store->pos_count = 0;

    store->namespace_capacity = 16;
    store->namespaces = (RDFNamespace*)safe_calloc(store->namespace_capacity, sizeof(RDFNamespace));
    store->namespace_count = 0;

    return store;
}

void rdf_triple_store_free(RDFTripleStore* store) {
    if (!store) return;
    for (size_t i = 0; i < store->node_count; i++) {
        safe_free((void**)&store->nodes[i].value);
        safe_free((void**)&store->nodes[i].datatype_iri);
        safe_free((void**)&store->nodes[i].lang_tag);
    }
    safe_free((void**)&store->nodes);
    safe_free((void**)&store->triples);
    safe_free((void**)&store->spo_index);
    safe_free((void**)&store->osp_index);
    safe_free((void**)&store->pos_index);
    for (size_t i = 0; i < store->namespace_count; i++) {
        safe_free((void**)&store->namespaces[i].prefix);
        safe_free((void**)&store->namespaces[i].iri);
    }
    safe_free((void**)&store->namespaces);
    safe_free((void**)&store);
}

int rdf_triple_store_add_prefix(RDFTripleStore* store, const char* prefix,
                                const char* iri) {
    if (!store || !prefix || !iri) return -1;
    if (store->namespace_count >= store->namespace_capacity) {
        size_t new_cap = store->namespace_capacity * 2;
        RDFNamespace* new_ns = (RDFNamespace*)safe_realloc(
            store->namespaces, new_cap * sizeof(RDFNamespace));
        if (!new_ns) return -1;
        store->namespaces = new_ns;
        store->namespace_capacity = new_cap;
    }
    store->namespaces[store->namespace_count].prefix = graph_strdup(prefix);
    store->namespaces[store->namespace_count].iri = graph_strdup(iri);
    store->namespace_count++;
    return 0;
}

const char* rdf_triple_store_get_prefix_iri(RDFTripleStore* store,
                                            const char* prefix) {
    if (!store || !prefix) return NULL;
    for (size_t i = 0; i < store->namespace_count; i++) {
        if (strcmp(store->namespaces[i].prefix, prefix) == 0) {
            return store->namespaces[i].iri;
        }
    }
    return NULL;
}

static int rdf_find_node(const RDFTripleStore* store, RDFNodeType node_type,
                         const char* value, const char* datatype_iri,
                         const char* lang_tag) {
    for (size_t i = 0; i < store->node_count; i++) {
        if (store->nodes[i].node_type != node_type) continue;
        if (strcmp(store->nodes[i].value, value) != 0) continue;
        if (node_type == RDF_NODE_LITERAL) {
            int dt_match = (!datatype_iri && !store->nodes[i].datatype_iri) ||
                           (datatype_iri && store->nodes[i].datatype_iri &&
                            strcmp(datatype_iri, store->nodes[i].datatype_iri) == 0);
            int lang_match = (!lang_tag && !store->nodes[i].lang_tag) ||
                             (lang_tag && store->nodes[i].lang_tag &&
                              strcmp(lang_tag, store->nodes[i].lang_tag) == 0);
            if (dt_match && lang_match) return (int)i;
        } else {
            return (int)i;
        }
    }
    return -1;
}

int rdf_triple_store_get_node(RDFTripleStore* store, RDFNodeType node_type,
                              const char* value, const char* datatype_iri,
                              const char* lang_tag) {
    if (!store || !value) return -1;
    int existing = rdf_find_node(store, node_type, value, datatype_iri, lang_tag);
    if (existing >= 0) return existing;

    if (store->node_count >= store->node_capacity) {
        size_t new_cap = store->node_capacity * 2;
        RDFNode* new_nodes = (RDFNode*)safe_realloc(
            store->nodes, new_cap * sizeof(RDFNode));
        if (!new_nodes) return -1;
        memset(new_nodes + store->node_capacity, 0,
               (new_cap - store->node_capacity) * sizeof(RDFNode));
        store->nodes = new_nodes;
        store->node_capacity = new_cap;
    }
    size_t idx = store->node_count++;
    store->nodes[idx].id = (int)idx;
    store->nodes[idx].node_type = node_type;
    store->nodes[idx].value = graph_strdup(value);
    store->nodes[idx].datatype_iri = graph_strdup(datatype_iri);
    store->nodes[idx].lang_tag = graph_strdup(lang_tag);
    return (int)idx;
}

static int spo_compare(const void* a, const void* b) {
    const SPOEntry* ea = (const SPOEntry*)a;
    const SPOEntry* eb = (const SPOEntry*)b;
    if (ea->subject_id != eb->subject_id) return ea->subject_id - eb->subject_id;
    if (ea->predicate_id != eb->predicate_id) return ea->predicate_id - eb->predicate_id;
    return ea->object_id - eb->object_id;
}

/* ZSFABC-P0-008修复: OSP索引必须按(object,subject,predicate)排序，不可复用spo_compare */
static int osp_compare(const void* a, const void* b) {
    const OSPEntry* ea = (const OSPEntry*)a;
    const OSPEntry* eb = (const OSPEntry*)b;
    if (ea->object_id != eb->object_id) return ea->object_id - eb->object_id;
    if (ea->subject_id != eb->subject_id) return ea->subject_id - eb->subject_id;
    return ea->predicate_id - eb->predicate_id;
}

static int pos_compare(const void* a, const void* b) {
    const POSEntry* ea = (const POSEntry*)a;
    const POSEntry* eb = (const POSEntry*)b;
    if (ea->predicate_id != eb->predicate_id) return ea->predicate_id - eb->predicate_id;
    if (ea->object_id != eb->object_id) return ea->object_id - eb->object_id;
    return ea->subject_id - eb->subject_id;
}

int rdf_triple_store_add_triple(RDFTripleStore* store, int subject_id,
                                int predicate_id, int object_id,
                                float confidence) {
    if (!store) return -1;
    if (subject_id < 0 || (size_t)subject_id >= store->node_count) return -1;
    if (predicate_id < 0 || (size_t)predicate_id >= store->node_count) return -1;
    if (object_id < 0 || (size_t)object_id >= store->node_count) return -1;

    if (store->triple_count >= store->triple_capacity) {
        size_t new_cap = store->triple_capacity * 2;
        RDFTriple* new_triples = (RDFTriple*)safe_realloc(
            store->triples, new_cap * sizeof(RDFTriple));
        if (!new_triples) return -1;
        store->triples = new_triples;
        store->triple_capacity = new_cap;
    }
    int tid = (int)store->triple_count;
    store->triples[tid].id = tid;
    store->triples[tid].subject_id = subject_id;
    store->triples[tid].predicate_id = predicate_id;
    store->triples[tid].object_id = object_id;
    store->triples[tid].confidence = confidence;
    store->triple_count++;

    if (store->spo_count >= store->spo_capacity) {
        size_t new_cap = store->spo_capacity * 2;
        SPOEntry* new_spo = (SPOEntry*)safe_realloc(
            store->spo_index, new_cap * sizeof(SPOEntry));
        if (!new_spo) return -1;
        store->spo_index = new_spo;
        store->spo_capacity = new_cap;
    }
    /* B-018修复: 二分查找插入位置 + memmove，避免每次qsort全量O(n log n) */
    {
        SPOEntry key;
        key.subject_id = subject_id;
        key.predicate_id = predicate_id;
        key.object_id = object_id;
        size_t lo = 0, hi = store->spo_count;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            int cmp = spo_compare(&key, &store->spo_index[mid]);
            if (cmp < 0) hi = mid;
            else if (cmp > 0) lo = mid + 1;
            else { lo = mid; hi = mid; break; }
        }
        if (lo < store->spo_count) {
            memmove(&store->spo_index[lo + 1], &store->spo_index[lo],
                    (store->spo_count - lo) * sizeof(SPOEntry));
        }
        store->spo_index[lo] = key;
        store->spo_count++;
    }

    if (store->osp_count >= store->osp_capacity) {
        size_t new_cap = store->osp_capacity * 2;
        OSPEntry* new_osp = (OSPEntry*)safe_realloc(
            store->osp_index, new_cap * sizeof(OSPEntry));
        if (!new_osp) return -1;
        store->osp_index = new_osp;
        store->osp_capacity = new_cap;
    }
    /* B-018修复: OSP索引二分查找插入位置 + memmove */
    {
        OSPEntry key;
        key.subject_id = subject_id;
        key.predicate_id = predicate_id;
        key.object_id = object_id;
        size_t lo = 0, hi = store->osp_count;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            int cmp = osp_compare(&key, &store->osp_index[mid]);
            if (cmp < 0) hi = mid;
            else if (cmp > 0) lo = mid + 1;
            else { lo = mid; hi = mid; break; }
        }
        if (lo < store->osp_count) {
            memmove(&store->osp_index[lo + 1], &store->osp_index[lo],
                    (store->osp_count - lo) * sizeof(OSPEntry));
        }
        store->osp_index[lo] = key;
        store->osp_count++;
    }

    if (store->pos_count >= store->pos_capacity) {
        size_t new_cap = store->pos_capacity * 2;
        POSEntry* new_pos = (POSEntry*)safe_realloc(
            store->pos_index, new_cap * sizeof(POSEntry));
        if (!new_pos) return -1;
        store->pos_index = new_pos;
        store->pos_capacity = new_cap;
    }
    /* B-018修复: POS索引二分查找插入位置 + memmove */
    {
        POSEntry key;
        key.predicate_id = predicate_id;
        key.object_id = object_id;
        key.subject_id = subject_id;
        size_t lo = 0, hi = store->pos_count;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            int cmp = pos_compare(&key, &store->pos_index[mid]);
            if (cmp < 0) hi = mid;
            else if (cmp > 0) lo = mid + 1;
            else { lo = mid; hi = mid; break; }
        }
        if (lo < store->pos_count) {
            memmove(&store->pos_index[lo + 1], &store->pos_index[lo],
                    (store->pos_count - lo) * sizeof(POSEntry));
        }
        store->pos_index[lo] = key;
        store->pos_count++;
    }

    return tid;
}

int rdf_triple_store_query(RDFTripleStore* store, int subject_id,
                           int predicate_id, int object_id,
                           RDFTriple* results, size_t max_results) {
    if (!store || !results) return -1;
    int count = 0;

    if (subject_id >= 0 && predicate_id >= 0 && object_id >= 0) {
        /* SPO全匹配：二分查找SPO索引 */
        SPOEntry key;
        key.subject_id = subject_id;
        key.predicate_id = predicate_id;
        key.object_id = object_id;
        SPOEntry* found = (SPOEntry*)bsearch(&key, store->spo_index,
            store->spo_count, sizeof(SPOEntry), spo_compare);
        if (found) {
            size_t idx = found - store->spo_index;
            for (size_t i = idx; i < store->spo_count && (size_t)count < max_results; i++) {
                if (store->spo_index[i].subject_id != subject_id ||
                    store->spo_index[i].predicate_id != predicate_id ||
                    store->spo_index[i].object_id != object_id) break;
                /* ZSFWS-L-017: 索引命中后全表回查——性能优化空间：
                 * 可在SPOEntry中直接存储triple索引避免O(n)全扫描 */
                for (size_t t = 0; t < store->triple_count; t++) {
                    if (store->triples[t].subject_id == subject_id &&
                        store->triples[t].predicate_id == predicate_id &&
                        store->triples[t].object_id == object_id) {
                        results[count++] = store->triples[t];
                        break;
                    }
                }
            }
        }
        return count;
    }

    if (subject_id >= 0) {
        /* 按主语查询：SPO索引范围扫描 */
        SPOEntry key;
        key.subject_id = subject_id;
        key.predicate_id = predicate_id >= 0 ? predicate_id : -1;
        key.object_id = object_id >= 0 ? object_id : -1;
        int use_pred = (predicate_id >= 0);
        int use_obj = (object_id >= 0);

        int found_start = 0;
        SPOEntry* base = NULL;
        if (predicate_id >= 0) {
            base = (SPOEntry*)bsearch(&key, store->spo_index,
                store->spo_count, sizeof(SPOEntry), spo_compare);
        }
        if (base) {
            size_t start_idx = base - store->spo_index;
            found_start = 1;
            for (size_t i = start_idx; i < store->spo_count && (size_t)count < max_results; i++) {
                if (store->spo_index[i].subject_id != subject_id) break;
                if (use_pred && store->spo_index[i].predicate_id != predicate_id) continue;
                if (use_obj && store->spo_index[i].object_id != object_id) continue;
                int s = store->spo_index[i].subject_id;
                int p = store->spo_index[i].predicate_id;
                int o = store->spo_index[i].object_id;
                for (size_t t = 0; t < store->triple_count; t++) {
                    if (store->triples[t].subject_id == s &&
                        store->triples[t].predicate_id == p &&
                        store->triples[t].object_id == o) {
                        results[count++] = store->triples[t];
                        break;
                    }
                }
            }
        }
        if (!found_start) {
            for (size_t i = 0; i < store->spo_count && (size_t)count < max_results; i++) {
                if (store->spo_index[i].subject_id != subject_id) continue;
                if (use_pred && store->spo_index[i].predicate_id != predicate_id) continue;
                if (use_obj && store->spo_index[i].object_id != object_id) continue;
                int s = store->spo_index[i].subject_id;
                int p = store->spo_index[i].predicate_id;
                int o = store->spo_index[i].object_id;
                for (size_t t = 0; t < store->triple_count; t++) {
                    if (store->triples[t].subject_id == s &&
                        store->triples[t].predicate_id == p &&
                        store->triples[t].object_id == o) {
                        results[count++] = store->triples[t];
                        break;
                    }
                }
            }
        }
        return count;
    }

    if (predicate_id >= 0) {
        POSEntry key;
        key.predicate_id = predicate_id;
        key.object_id = object_id >= 0 ? object_id : -1;
        key.subject_id = -1;
        int use_obj = (object_id >= 0);

        int found_start = 0;
        if (object_id >= 0) {
            POSEntry* base = (POSEntry*)bsearch(&key, store->pos_index,
                store->pos_count, sizeof(POSEntry), pos_compare);
            if (base) {
                size_t start_idx = base - store->pos_index;
                found_start = 1;
                for (size_t i = start_idx; i < store->pos_count && (size_t)count < max_results; i++) {
                    if (store->pos_index[i].predicate_id != predicate_id) break;
                    if (use_obj && store->pos_index[i].object_id != object_id) continue;
                    int s = store->pos_index[i].subject_id;
                    int p = store->pos_index[i].predicate_id;
                    int o = store->pos_index[i].object_id;
                    for (size_t t = 0; t < store->triple_count; t++) {
                        if (store->triples[t].subject_id == s &&
                            store->triples[t].predicate_id == p &&
                            store->triples[t].object_id == o) {
                            results[count++] = store->triples[t];
                            break;
                        }
                    }
                }
            }
        }
        if (!found_start) {
            for (size_t i = 0; i < store->pos_count && (size_t)count < max_results; i++) {
                if (store->pos_index[i].predicate_id != predicate_id) continue;
                if (use_obj && store->pos_index[i].object_id != object_id) continue;
                int s = store->pos_index[i].subject_id;
                int p = store->pos_index[i].predicate_id;
                int o = store->pos_index[i].object_id;
                for (size_t t = 0; t < store->triple_count; t++) {
                    if (store->triples[t].subject_id == s &&
                        store->triples[t].predicate_id == p &&
                        store->triples[t].object_id == o) {
                        results[count++] = store->triples[t];
                        break;
                    }
                }
            }
        }
        return count;
    }

    if (object_id >= 0) {
        for (size_t i = 0; i < store->osp_count && (size_t)count < max_results; i++) {
            if (store->osp_index[i].object_id != object_id) continue;
            if (predicate_id >= 0 && store->osp_index[i].predicate_id != predicate_id) continue;
            int s = store->osp_index[i].subject_id;
            int p = store->osp_index[i].predicate_id;
            int o = store->osp_index[i].object_id;
            for (size_t t = 0; t < store->triple_count; t++) {
                if (store->triples[t].subject_id == s &&
                    store->triples[t].predicate_id == p &&
                    store->triples[t].object_id == o) {
                    results[count++] = store->triples[t];
                    break;
                }
            }
        }
        return count;
    }

    /* 无约束：返回所有 */
    size_t max = max_results < store->triple_count ? max_results : store->triple_count;
    memcpy(results, store->triples, max * sizeof(RDFTriple));
    return (int)max;
}

int rdf_triple_store_add_triple_by_values(RDFTripleStore* store,
                                          const char* subject,
                                          const char* predicate,
                                          const char* object_val,
                                          float confidence) {
    if (!store || !subject || !predicate || !object_val) return -1;
    int s_id = rdf_triple_store_get_node(store, RDF_NODE_IRI, subject, NULL, NULL);
    int p_id = rdf_triple_store_get_node(store, RDF_NODE_IRI, predicate, NULL, NULL);
    int o_id = rdf_triple_store_get_node(store, RDF_NODE_IRI, object_val, NULL, NULL);
    if (s_id < 0 || p_id < 0 || o_id < 0) return -1;
    return rdf_triple_store_add_triple(store, s_id, p_id, o_id, confidence);
}

int rdf_triple_store_query_by_values(RDFTripleStore* store,
                                     const char* subject,
                                     const char* predicate,
                                     const char* object_val,
                                     RDFTriple* results, size_t max_results) {
    if (!store || !results) return -1;
    int s_id = -1, p_id = -1, o_id = -1;
    if (subject) {
        int idx = rdf_find_node(store, RDF_NODE_IRI, subject, NULL, NULL);
        if (idx < 0) return 0;
        s_id = idx;
    }
    if (predicate) {
        int idx = rdf_find_node(store, RDF_NODE_IRI, predicate, NULL, NULL);
        if (idx < 0) return 0;
        p_id = idx;
    }
    if (object_val) {
        int idx = rdf_find_node(store, RDF_NODE_IRI, object_val, NULL, NULL);
        if (idx < 0) return 0;
        o_id = idx;
    }
    return rdf_triple_store_query(store, s_id, p_id, o_id, results, max_results);
}

const RDFNode* rdf_triple_store_get_node_by_id(RDFTripleStore* store,
                                                int node_id) {
    if (!store) return NULL;
    if (node_id < 0 || (size_t)node_id >= store->node_count) return NULL;
    return &store->nodes[node_id];
}

size_t rdf_triple_store_count(RDFTripleStore* store) {
    return store ? store->triple_count : 0;
}

size_t rdf_triple_store_node_count(RDFTripleStore* store) {
    return store ? store->node_count : 0;
}

int rdf_triple_store_export_ntriples(RDFTripleStore* store,
                                     const char* filename) {
    if (!store || !filename) return -1;
    FILE* fp = fopen(filename, "w");
    if (!fp) return -1;

    for (size_t i = 0; i < store->triple_count; i++) {
        RDFTriple* t = &store->triples[i];
        const RDFNode* s = &store->nodes[t->subject_id];
        const RDFNode* p = &store->nodes[t->predicate_id];
        const RDFNode* o = &store->nodes[t->object_id];

        if (s->node_type == RDF_NODE_IRI)
            fprintf(fp, "<%s> ", s->value);
        else if (s->node_type == RDF_NODE_BLANK)
            fprintf(fp, "_:%s ", s->value);
        else
            fprintf(fp, "\"%s\" ", s->value);

        if (p->node_type == RDF_NODE_IRI)
            fprintf(fp, "<%s> ", p->value);
        else
            fprintf(fp, "\"%s\" ", p->value);

        if (o->node_type == RDF_NODE_IRI)
            fprintf(fp, "<%s>", o->value);
        else if (o->node_type == RDF_NODE_BLANK)
            fprintf(fp, "_:%s", o->value);
        else {
            fprintf(fp, "\"%s\"", o->value);
            if (o->datatype_iri)
                fprintf(fp, "^^<%s>", o->datatype_iri);
            if (o->lang_tag)
                fprintf(fp, "@%s", o->lang_tag);
        }
        fprintf(fp, " .\n");
    }

    fclose(fp);
    return 0;
}

int rdf_triple_store_export_turtle(RDFTripleStore* store,
                                   const char* filename) {
    if (!store || !filename) return -1;
    FILE* fp = fopen(filename, "w");
    if (!fp) return -1;

    for (size_t i = 0; i < store->namespace_count; i++) {
        fprintf(fp, "@prefix %s: <%s> .\n",
                store->namespaces[i].prefix,
                store->namespaces[i].iri);
    }

    for (size_t i = 0; i < store->triple_count; i++) {
        RDFTriple* t = &store->triples[i];
        const RDFNode* s = &store->nodes[t->subject_id];
        const RDFNode* p = &store->nodes[t->predicate_id];
        const RDFNode* o = &store->nodes[t->object_id];

        int s_abbrev = 0, p_abbrev = 0, o_abbrev = 0;
        for (size_t j = 0; j < store->namespace_count; j++) {
            size_t iri_len = strlen(store->namespaces[j].iri);
            if (s->node_type == RDF_NODE_IRI &&
                strncmp(s->value, store->namespaces[j].iri, iri_len) == 0) {
                fprintf(fp, "%s:%s ", store->namespaces[j].prefix,
                        s->value + iri_len);
                s_abbrev = 1;
                break;
            }
        }
        if (!s_abbrev) {
            if (s->node_type == RDF_NODE_IRI)
                fprintf(fp, "<%s> ", s->value);
            else if (s->node_type == RDF_NODE_BLANK)
                fprintf(fp, "_:%s ", s->value);
            else
                fprintf(fp, "\"%s\" ", s->value);
        }

        for (size_t j = 0; j < store->namespace_count; j++) {
            size_t iri_len = strlen(store->namespaces[j].iri);
            if (p->node_type == RDF_NODE_IRI &&
                strncmp(p->value, store->namespaces[j].iri, iri_len) == 0) {
                fprintf(fp, "%s:%s ", store->namespaces[j].prefix,
                        p->value + iri_len);
                p_abbrev = 1;
                break;
            }
        }
        if (!p_abbrev) {
            fprintf(fp, "<%s> ", p->value);
        }

        for (size_t j = 0; j < store->namespace_count; j++) {
            size_t iri_len = strlen(store->namespaces[j].iri);
            if (o->node_type == RDF_NODE_IRI &&
                strncmp(o->value, store->namespaces[j].iri, iri_len) == 0) {
                fprintf(fp, "%s:%s", store->namespaces[j].prefix,
                        o->value + iri_len);
                o_abbrev = 1;
                break;
            }
        }
        if (!o_abbrev) {
            if (o->node_type == RDF_NODE_IRI)
                fprintf(fp, "<%s>", o->value);
            else if (o->node_type == RDF_NODE_BLANK)
                fprintf(fp, "_:%s", o->value);
            else {
                fprintf(fp, "\"%s\"", o->value);
                if (o->datatype_iri)
                    fprintf(fp, "^^<%s>", o->datatype_iri);
                if (o->lang_tag)
                    fprintf(fp, "@%s", o->lang_tag);
            }
        }
        fprintf(fp, " .\n");
    }

    fclose(fp);
    return 0;
}

int rdf_triple_store_import_ntriples(RDFTripleStore* store,
                                     const char* filename) {
    if (!store || !filename) return -1;
    FILE* fp = fopen(filename, "r");
    if (!fp) return -1;

    char line[4096];
    int import_count = 0;

    while (fgets(line, sizeof(line), fp)) {
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\0' || *p == '\n' || *p == '\r') continue;

        char subject[1024] = {0};
        char predicate[1024] = {0};
        char object_val[1024] = {0};
        int s_blank = 0, o_blank = 0;
        int s_literal = 0, o_literal = 0;
        char* token;

        /* ZSFZS-F040: strtok→strtok_s线程安全 */
        char* saveptr = NULL;
        token = strtok_s(p, " \t", &saveptr);
        if (!token) continue;

        if (token[0] == '<') {
            size_t len = strlen(token);
            if (len > 2 && token[len - 1] == '>') {
                token[len - 1] = '\0';
                strncpy(subject, token + 1, sizeof(subject) - 1);
            }
        } else if (strncmp(token, "_:", 2) == 0) {
            strncpy(subject, token + 2, sizeof(subject) - 1);
            s_blank = 1;
        } else {
            strncpy(subject, token, sizeof(subject) - 1);
            s_literal = 1;
        }

        token = strtok_s(NULL, " \t", &saveptr);
        if (!token) continue;  /* 跳过无谓词的行 */
        if (token[0] == '<') {
            size_t len = strlen(token);
            if (len > 2 && token[len - 1] == '>') {
                token[len - 1] = '\0';
                strncpy(predicate, token + 1, sizeof(predicate) - 1);
            }
        } else {
            strncpy(predicate, token, sizeof(predicate) - 1);
        }

        token = strtok(NULL, " \t");
        if (!token) continue;
        if (token[0] == '<') {
            size_t len = strlen(token);
            if (len > 2 && token[len - 1] == '>') {
                token[len - 1] = '\0';
                strncpy(object_val, token + 1, sizeof(object_val) - 1);
            }
        } else if (strncmp(token, "_:", 2) == 0) {
            strncpy(object_val, token + 2, sizeof(object_val) - 1);
            o_blank = 1;
        } else if (token[0] == '\"') {
            char* end = strrchr(token, '\"');
            if (end) {
                size_t val_len = end - token - 1;
                if (val_len < sizeof(object_val)) {
                    strncpy(object_val, token + 1, val_len);
                    object_val[val_len] = '\0';
                }
            }
            o_literal = 1;
        } else {
            strncpy(object_val, token, sizeof(object_val) - 1);
        }

        int s_id = rdf_triple_store_get_node(
            store, s_blank ? RDF_NODE_BLANK : (s_literal ? RDF_NODE_LITERAL : RDF_NODE_IRI),
            subject, NULL, NULL);
        int p_id = rdf_triple_store_get_node(
            store, RDF_NODE_IRI, predicate, NULL, NULL);
        int o_id = rdf_triple_store_get_node(
            store, o_blank ? RDF_NODE_BLANK : (o_literal ? RDF_NODE_LITERAL : RDF_NODE_IRI),
            object_val, NULL, NULL);

        if (s_id >= 0 && p_id >= 0 && o_id >= 0) {
            if (rdf_triple_store_add_triple(store, s_id, p_id, o_id, 1.0f) >= 0) {
                import_count++;
            }
        }
    }

    fclose(fp);
    return import_count;
}

int rdf_triple_store_save(RDFTripleStore* store, const char* filename) {
    if (!store || !filename) return -1;
    FILE* fp = fopen(filename, "wb");
    if (!fp) return -1;

    uint32_t magic = 0x52444653;
    uint32_t version = 1;
    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);

    uint32_t nc = (uint32_t)store->node_count;
    fwrite(&nc, sizeof(nc), 1, fp);
    for (size_t i = 0; i < store->node_count; i++) {
        RDFNode* n = &store->nodes[i];
        uint32_t ntype = (uint32_t)n->node_type;
        fwrite(&ntype, sizeof(ntype), 1, fp);
        uint32_t vlen = (uint32_t)strlen(n->value) + 1;
        fwrite(&vlen, sizeof(vlen), 1, fp);
        fwrite(n->value, 1, vlen, fp);
        uint8_t has_dt = (n->datatype_iri != NULL) ? 1 : 0;
        fwrite(&has_dt, 1, 1, fp);
        if (has_dt) {
            uint32_t dtlen = (uint32_t)strlen(n->datatype_iri) + 1;
            fwrite(&dtlen, sizeof(dtlen), 1, fp);
            fwrite(n->datatype_iri, 1, dtlen, fp);
        }
        uint8_t has_lang = (n->lang_tag != NULL) ? 1 : 0;
        fwrite(&has_lang, 1, 1, fp);
        if (has_lang) {
            uint32_t llen = (uint32_t)strlen(n->lang_tag) + 1;
            fwrite(&llen, sizeof(llen), 1, fp);
            fwrite(n->lang_tag, 1, llen, fp);
        }
    }

    uint32_t tc = (uint32_t)store->triple_count;
    fwrite(&tc, sizeof(tc), 1, fp);
    fwrite(store->triples, sizeof(RDFTriple), store->triple_count, fp);

    uint32_t nsc = (uint32_t)store->namespace_count;
    fwrite(&nsc, sizeof(nsc), 1, fp);
    for (size_t i = 0; i < store->namespace_count; i++) {
        uint32_t plen = (uint32_t)strlen(store->namespaces[i].prefix) + 1;
        fwrite(&plen, sizeof(plen), 1, fp);
        fwrite(store->namespaces[i].prefix, 1, plen, fp);
        uint32_t ilen = (uint32_t)strlen(store->namespaces[i].iri) + 1;
        fwrite(&ilen, sizeof(ilen), 1, fp);
        fwrite(store->namespaces[i].iri, 1, ilen, fp);
    }

    fclose(fp);
    return 0;
}

RDFTripleStore* rdf_triple_store_load(const char* filename) {
    if (!filename) return NULL;
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;

    uint32_t magic, version;
    if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != 0x52444653) {
        fclose(fp);
        return NULL;
    }
    if (fread(&version, sizeof(version), 1, fp) != 1 || version != 1) {
        fclose(fp);
        return NULL;
    }

    RDFTripleStore* store = rdf_triple_store_create();
    if (!store) { fclose(fp); return NULL; }

    uint32_t nc;
    if (fread(&nc, sizeof(nc), 1, fp) != 1) { fclose(fp); rdf_triple_store_free(store); return NULL; }
    for (uint32_t i = 0; i < nc; i++) {
        uint32_t ntype;
        if (fread(&ntype, sizeof(ntype), 1, fp) != 1) break;
        char val_buf[1024] = {0};
        uint32_t vlen;
        fread(&vlen, sizeof(vlen), 1, fp);
        if (vlen > 0) {
            size_t read_len = vlen < sizeof(val_buf) ? vlen : sizeof(val_buf) - 1;
            fread(val_buf, 1, read_len, fp);
        }
        char dt_buf[256] = {0};
        uint8_t has_dt;
        fread(&has_dt, 1, 1, fp);
        if (has_dt) {
            uint32_t dtlen;
            fread(&dtlen, sizeof(dtlen), 1, fp);
            if (dtlen > 0) {
                size_t read_len = dtlen < sizeof(dt_buf) ? dtlen : sizeof(dt_buf) - 1;
                fread(dt_buf, 1, read_len, fp);
            }
        }
        char lang_buf[64] = {0};
        uint8_t has_lang;
        fread(&has_lang, 1, 1, fp);
        if (has_lang) {
            uint32_t llen;
            fread(&llen, sizeof(llen), 1, fp);
            if (llen > 0) {
                size_t read_len = llen < sizeof(lang_buf) ? llen : sizeof(lang_buf) - 1;
                fread(lang_buf, 1, read_len, fp);
            }
        }
        rdf_triple_store_get_node(store, (RDFNodeType)ntype, val_buf,
                                  has_dt ? dt_buf : NULL,
                                  has_lang ? lang_buf : NULL);
    }

    uint32_t tc;
    if (fread(&tc, sizeof(tc), 1, fp) != 1) { fclose(fp); return store; }
    for (uint32_t i = 0; i < tc; i++) {
        RDFTriple t;
        if (fread(&t, sizeof(RDFTriple), 1, fp) != 1) break;
        /* 直接添加到triples数组，跳过re-sort开销 */
        if (store->triple_count >= store->triple_capacity) {
            size_t new_cap = store->triple_capacity * 2;
            RDFTriple* new_t = (RDFTriple*)safe_realloc(
                store->triples, new_cap * sizeof(RDFTriple));
            if (!new_t) break;
            store->triples = new_t;
            store->triple_capacity = new_cap;
        }
        store->triples[store->triple_count] = t;
        store->triple_count++;

        /* 重建索引 */
        if (store->spo_count >= store->spo_capacity) {
            size_t new_cap = store->spo_capacity * 2;
            SPOEntry* new_spo = (SPOEntry*)safe_realloc(
                store->spo_index, new_cap * sizeof(SPOEntry));
            if (!new_spo) break;
            store->spo_index = new_spo;
            store->spo_capacity = new_cap;
        }
        store->spo_index[store->spo_count].subject_id = t.subject_id;
        store->spo_index[store->spo_count].predicate_id = t.predicate_id;
        store->spo_index[store->spo_count].object_id = t.object_id;
        store->spo_count++;

        if (store->osp_count >= store->osp_capacity) {
            size_t new_cap = store->osp_capacity * 2;
            OSPEntry* new_osp = (OSPEntry*)safe_realloc(
                store->osp_index, new_cap * sizeof(OSPEntry));
            if (!new_osp) break;
            store->osp_index = new_osp;
            store->osp_capacity = new_cap;
        }
        store->osp_index[store->osp_count].subject_id = t.subject_id;
        store->osp_index[store->osp_count].predicate_id = t.predicate_id;
        store->osp_index[store->osp_count].object_id = t.object_id;
        store->osp_count++;

        if (store->pos_count >= store->pos_capacity) {
            size_t new_cap = store->pos_capacity * 2;
            POSEntry* new_pos = (POSEntry*)safe_realloc(
                store->pos_index, new_cap * sizeof(POSEntry));
            if (!new_pos) break;
            store->pos_index = new_pos;
            store->pos_capacity = new_cap;
        }
        store->pos_index[store->pos_count].predicate_id = t.predicate_id;
        store->pos_index[store->pos_count].object_id = t.object_id;
        store->pos_index[store->pos_count].subject_id = t.subject_id;
        store->pos_count++;
    }
    qsort(store->spo_index, store->spo_count, sizeof(SPOEntry), spo_compare);
    qsort(store->osp_index, store->osp_count, sizeof(OSPEntry), osp_compare);
    qsort(store->pos_index, store->pos_count, sizeof(POSEntry), pos_compare);

    uint32_t nsc;
    if (fread(&nsc, sizeof(nsc), 1, fp) == 1) {
        for (uint32_t i = 0; i < nsc; i++) {
            char prefix_buf[128] = {0};
            uint32_t plen;
            fread(&plen, sizeof(plen), 1, fp);
            if (plen > 0) {
                size_t read_len = plen < sizeof(prefix_buf) ? plen : sizeof(prefix_buf) - 1;
                fread(prefix_buf, 1, read_len, fp);
            }
            char iri_buf[512] = {0};
            uint32_t ilen;
            fread(&ilen, sizeof(ilen), 1, fp);
            if (ilen > 0) {
                size_t read_len = ilen < sizeof(iri_buf) ? ilen : sizeof(iri_buf) - 1;
                fread(iri_buf, 1, read_len, fp);
            }
            if (plen > 0 && ilen > 0) {
                rdf_triple_store_add_prefix(store, prefix_buf, iri_buf);
            }
        }
    }

    fclose(fp);
    return store;
}