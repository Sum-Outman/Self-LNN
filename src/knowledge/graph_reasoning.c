/**
 * @file graph_reasoning.c
 * @brief 知识图谱推理引擎完整实现（A03.1.3）
 *
 * 三种推理模式：
 * 1. CfC ODE连续空间嵌入推理（替代TransE）
 * 2. CfC复数域旋转嵌入推理（替代RotatE）
 * 3. CfC图液态推理（替代GNN）
 *
 * 集成 graph_storage 引擎 + cfc_knowledge_embedding 引擎。
 */

#include "selflnn/knowledge/graph_reasoning.h"
#include "selflnn/knowledge/graph_storage.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h" /* 统一安全随机数 */
/* CfC知识库集成——通过cfc_knowledge_embedding引擎进行语义嵌入与推理 */
/* #include "selflnn/core/cfc_cell.h" —— 所有连续演化由主LNN统一管理 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define GR_EPSILON          1e-8f
#define GR_LOSS_HISTORY_MAX 200
#define GR_MAX_NEG_SAMPLES  100
#define GR_ENERGY_NORMALIZE 10.0f

/* 已改用secure_random，移除旧LCG锁机制 */

static float gr_rand_float(void) {
    return secure_random_float();
}

static int gr_rand_int(int min, int max) {
    return min + (int)(gr_rand_float() * (max - min + 1));
}

static float gr_rand_normal(void) {
    float u1 = gr_rand_float();
    float u2 = gr_rand_float();
    if (u1 < GR_EPSILON) u1 = GR_EPSILON;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/** @brief 第K个关系路径生成（用于规则挖掘的枚举） */
typedef struct {
    int relations[GR_MAX_PATH_LENGTH];
    int length;
    int entities[GR_MAX_PATH_LENGTH + 1];
    int entity_count;
} RelationPath;

/** @brief 关系路径缓存 */
typedef struct {
    RelationPath paths[GR_MAX_PATHS];
    int count;
} RelationPathCache;

/** @brief 修复建议 */
typedef struct {
    int entity_id;
    int relation_id;
    int target_entity_id;
    float confidence;
    int action_type;
} RepairAction;

/* ============ GraphReasoner 内部结构（不透明） ============ */

struct GraphReasoner {
    GraphReasonConfig config;

    CfCEmbedState* embed_state;
    int embed_owns;

    PropertyGraph* property_graph;
    int pg_owns;

    AdjacencyList* adjacency_list;
    int al_owns;

    RDFTripleStore* rdf_store;
    int rdf_owns;

    int is_trained;
    int entity_count;
    int relation_count;

    float loss_history[GR_LOSS_HISTORY_MAX];
    int loss_count;

    int entity_to_cfc_id[GR_MAX_ENTITIES];
    int relation_to_cfc_id[GR_MAX_RELATIONS];
    int cfc_to_entity_id[GR_MAX_ENTITIES];
    int cfc_to_rel_id[GR_MAX_RELATIONS];

    /* cfc_cell_instance已移除——由主LNN统一处理连续动态演化 */
};

/* ============ 默认配置 ============ */

static GraphReasonConfig graph_reasoner_default_config(void) {
    GraphReasonConfig cfg;
    memset(&cfg, 0, sizeof(GraphReasonConfig));
    cfg.reason_mode = GR_REASON_CONTINUOUS;
    cfg.embedding_dim = 128;
    cfg.learning_rate = 0.001f;
    cfg.margin = 1.0f;
    cfg.num_negative_samples = 10;
    cfg.batch_size = 128;
    cfg.max_epochs = 200;
    cfg.cfc_tau = 2.0f;
    cfg.cfc_dt = 0.1f;
    cfg.cfc_steps = 10;
    cfg.graph_prop_steps = 3;
    cfg.graph_prop_dropout = 0.1f;
    cfg.top_k_predictions = 10;
    cfg.consistency_threshold = 0.5f;
    cfg.rule_min_confidence = 0.3f;
    cfg.rule_min_support = 2.0f;
    cfg.enable_self_diagnosis = 1;
    cfg.enable_auto_repair = 0;
    return cfg;
}

/* ============ CfC ODE深度步进（RK4自适应步进，替代Euler离散化） ============ */

/* CfC ODE右端项：dh/dt = -h/τ + gate*sigmoid(liquid) - inhibit*cross */
static void gr_cfc_rhs(const float* h, int dim, float tau, float* dh) {
    for (int i = 0; i < dim; i++) {
        float gate_sig = 1.0f / (1.0f + expf(-h[i]));
        float liquid_act = tanhf(0.8f * h[i] + 0.2f * h[(i + 1) % dim]);
        float inhibit = 1.0f - gate_sig;
        dh[i] = -h[i] / (tau + GR_EPSILON)
                + gate_sig * liquid_act
                - 0.1f * inhibit * h[(i + dim - 1) % dim];
    }
}

/* D-010修复: RK4 ODE步进失败时返回错误码而非静默降级Euler法
 * 内存分配失败时上层可重试或优雅处理 */
static int gr_cfc_step_deep(float* h, int dim, float tau, float dt, int ode_steps) {
    float* k1 = (float*)safe_malloc((size_t)dim * sizeof(float));
    float* k2 = (float*)safe_malloc((size_t)dim * sizeof(float));
    float* k3 = (float*)safe_malloc((size_t)dim * sizeof(float));
    float* k4 = (float*)safe_malloc((size_t)dim * sizeof(float));
    float* tmp = (float*)safe_malloc((size_t)dim * sizeof(float));
    if (!k1 || !k2 || !k3 || !k4 || !tmp) {
        safe_free((void**)&k1); safe_free((void**)&k2); safe_free((void**)&k3);
        safe_free((void**)&k4); safe_free((void**)&tmp);
        return -1;
    }
    float sub_dt = dt / (float)ode_steps;
    for (int s = 0; s < ode_steps; s++) {
        /* k1 = f(t, y) */
        gr_cfc_rhs(h, dim, tau, k1);
        /* k2 = f(t+dt/2, y+dt*k1/2) */
        for (int i = 0; i < dim; i++) tmp[i] = h[i] + 0.5f * sub_dt * k1[i];
        gr_cfc_rhs(tmp, dim, tau, k2);
        /* k3 = f(t+dt/2, y+dt*k2/2) */
        for (int i = 0; i < dim; i++) tmp[i] = h[i] + 0.5f * sub_dt * k2[i];
        gr_cfc_rhs(tmp, dim, tau, k3);
        /* k4 = f(t+dt, y+dt*k3) */
        for (int i = 0; i < dim; i++) tmp[i] = h[i] + sub_dt * k3[i];
        gr_cfc_rhs(tmp, dim, tau, k4);
        /* y_{n+1} = y_n + dt/6*(k1+2k2+2k3+k4) */
        for (int i = 0; i < dim; i++)
            h[i] += sub_dt * (k1[i] + 2.0f*k2[i] + 2.0f*k3[i] + k4[i]) / 6.0f;
    }
    safe_free((void**)&k1); safe_free((void**)&k2); safe_free((void**)&k3);
    safe_free((void**)&k4); safe_free((void**)&tmp);
    return 0;
}

/* ============ 图传播能量计算 ============ */

static float gr_compute_node_energy(const float* emb, int dim) {
    float energy = 0.0f;
    for (int i = 0; i < dim; i++) {
        energy += emb[i] * emb[i];
    }
    return sqrtf(energy / (float)dim);
}

/* ============ 生命周期实现 ============ */

GraphReasoner* graph_reasoner_create(const GraphReasonConfig* config,
                                     PropertyGraph* property_graph,
                                     AdjacencyList* adjacency_list,
                                     RDFTripleStore* rdf_store) {
    GraphReasoner* reasoner = (GraphReasoner*)
        safe_calloc(1, sizeof(GraphReasoner));
    if (!reasoner) return NULL;

    if (config) {
        reasoner->config = *config;
    } else {
        reasoner->config = graph_reasoner_default_config();
    }

    int dim = reasoner->config.embedding_dim;
    if (dim <= 0) dim = 128;
    if (dim > CFC_EMBED_MAX_DIM) dim = CFC_EMBED_MAX_DIM;
    reasoner->config.embedding_dim = dim;

    CfCEmbedType embed_type;
    switch (reasoner->config.reason_mode) {
        case GR_REASON_QUATERNION:
            embed_type = CFC_EMBED_QUATERNION;
            break;
        case GR_REASON_LIQUID_GRAPH:
            embed_type = CFC_EMBED_LIQUID_GRAPH;
            break;
        case GR_REASON_HYBRID:
            embed_type = CFC_EMBED_HYBRID;
            break;
        default:
            embed_type = CFC_EMBED_CONTINUOUS;
            break;
    }

    CfCEmbedConfig emb_cfg = cfc_embed_default_config();
    emb_cfg.embedding_dim = dim;
    emb_cfg.embed_type = embed_type;
    emb_cfg.learning_rate = reasoner->config.learning_rate;
    emb_cfg.margin = reasoner->config.margin;
    emb_cfg.num_negative_samples = reasoner->config.num_negative_samples;
    emb_cfg.batch_size = reasoner->config.batch_size;
    emb_cfg.max_epochs = reasoner->config.max_epochs;
    emb_cfg.cfc_tau = reasoner->config.cfc_tau;
    emb_cfg.cfc_dt = reasoner->config.cfc_dt;
    emb_cfg.cfc_steps = reasoner->config.cfc_steps;
    emb_cfg.enable_graph_propagation = 1;
    emb_cfg.graph_prop_steps = reasoner->config.graph_prop_steps;
    emb_cfg.graph_prop_dropout = reasoner->config.graph_prop_dropout;

    reasoner->embed_state = cfc_embed_create(&emb_cfg);
    if (!reasoner->embed_state) {
        safe_free((void**)&reasoner);
        return NULL;
    }
    reasoner->embed_owns = 1;

    for (int i = 0; i < GR_MAX_ENTITIES; i++) {
        reasoner->entity_to_cfc_id[i] = -1;
        reasoner->cfc_to_entity_id[i] = -1;
    }
    for (int i = 0; i < GR_MAX_RELATIONS; i++) {
        reasoner->relation_to_cfc_id[i] = -1;
        reasoner->cfc_to_rel_id[i] = -1;
    }

    if (property_graph) {
        reasoner->property_graph = property_graph;
        reasoner->pg_owns = 0;
    }
    if (adjacency_list) {
        reasoner->adjacency_list = adjacency_list;
        reasoner->al_owns = 0;
    }
    if (rdf_store) {
        reasoner->rdf_store = rdf_store;
        reasoner->rdf_owns = 0;
    }

    /* cfc_cell_instance 创建已移除——图液态推理使用 gr_cfc_step_deep 连续演化，
     * 无需独立CfC实例，所有连续动态统一由主LNN管理 */

    reasoner->is_trained = 0;
    reasoner->entity_count = 0;
    reasoner->relation_count = 0;
    reasoner->loss_count = 0;

    return reasoner;
}

void graph_reasoner_destroy(GraphReasoner* reasoner) {
    if (!reasoner) return;

    if (reasoner->embed_owns && reasoner->embed_state) {
        cfc_embed_destroy(reasoner->embed_state);
    }

    /* cfc_cell_instance 释放已移除——实例已不再创建 */

    if (reasoner->pg_owns && reasoner->property_graph) {
        property_graph_free(reasoner->property_graph);
    }
    if (reasoner->al_owns && reasoner->adjacency_list) {
        adjacency_list_free(reasoner->adjacency_list);
    }
    if (reasoner->rdf_owns && reasoner->rdf_store) {
        rdf_triple_store_free(reasoner->rdf_store);
    }

    safe_free((void**)&reasoner);
}

int graph_reasoner_get_config(const GraphReasoner* reasoner, GraphReasonConfig* config) {
    if (!reasoner || !config) return -1;
    *config = reasoner->config;
    return 0;
}

int graph_reasoner_set_config(GraphReasoner* reasoner, const GraphReasonConfig* config) {
    if (!reasoner || !config) return -1;
    reasoner->config = *config;
    return 0;
}

/* ============ 数据加载 ============ */

static int gr_load_from_property_graph(GraphReasoner* reasoner) {
    if (!reasoner || !reasoner->property_graph) return -1;
    if (reasoner->entity_count > 0) return 0;

    int max_id = property_graph_get_node_capacity(reasoner->property_graph);
    if (max_id <= 0) return 0;

    for (int i = 0; i < max_id; i++) {
        const PGNode* node = property_graph_get_node(reasoner->property_graph, i);
        if (!node) continue;

        const char* label = node->label ? node->label : "unknown";
        int cfc_id = cfc_embed_add_entity(reasoner->embed_state, label);
        if (cfc_id >= 0 && cfc_id < GR_MAX_ENTITIES) {
            reasoner->entity_to_cfc_id[i] = cfc_id;
            reasoner->cfc_to_entity_id[cfc_id] = i;
            reasoner->entity_count++;
        }
    }

/* 属性图的边数据通过RDF存储路径加载（gr_load_from_rdf_store）
     * property_graph当前无公开边迭代API，边的关系注册由上层调用
     * graph_reasoner_train_from_triples提供，确保链路预测和规则挖掘正常工作 */

    return 0;
}

static int gr_load_from_adjacency_list(GraphReasoner* reasoner) {
    if (!reasoner || !reasoner->adjacency_list) return -1;
    if (reasoner->entity_count > 0) return 0;

    int max_id = adjacency_list_get_node_capacity(reasoner->adjacency_list);
    if (max_id <= 0) return 0;

    for (int i = 0; i < max_id; i++) {
        const ALNode* node = adjacency_list_get_node(reasoner->adjacency_list, i);
        if (!node) continue;

        const char* label = node->label ? node->label : "unknown";
        int cfc_id = cfc_embed_add_entity(reasoner->embed_state, label);
        if (cfc_id >= 0 && cfc_id < GR_MAX_ENTITIES) {
            reasoner->entity_to_cfc_id[i] = cfc_id;
            reasoner->cfc_to_entity_id[cfc_id] = i;
            reasoner->entity_count++;
        }

/* 从邻接表出边加载关系
         * M-003修复: adjacency_list_get_out_neighbors需要int*缓冲区，
         * 不能将int**强转为int*传入。使用栈分配缓冲区替代。 */
        int out_neighbors_buf[64];
        memset(out_neighbors_buf, 0, sizeof(out_neighbors_buf));
        int out_cnt = adjacency_list_get_out_neighbors(reasoner->adjacency_list, i,
                                                        out_neighbors_buf, NULL, 64);
        for (int oe = 0; oe < out_cnt && oe < 64; oe++) {
            int tgt_id = out_neighbors_buf[oe];
            int cfc_rel_id = cfc_embed_add_relation(reasoner->embed_state, "adjacent");
            if (cfc_rel_id >= 0) {
                int tgt_cfc = reasoner->entity_to_cfc_id[tgt_id];
                if (tgt_cfc >= 0) {
                    cfc_embed_add_triple(reasoner->embed_state, cfc_id, cfc_rel_id, tgt_cfc);
                    reasoner->relation_count++;
                }
            }
        }
    }

    return 0;
}

static int gr_load_from_rdf_store(GraphReasoner* reasoner) {
    if (!reasoner || !reasoner->rdf_store) return -1;
    if (reasoner->entity_count > 0) return 0;

    size_t node_cnt = rdf_triple_store_node_count(reasoner->rdf_store);
    size_t triple_cnt = rdf_triple_store_count(reasoner->rdf_store);

    for (size_t i = 0; i < node_cnt && i < (size_t)GR_MAX_ENTITIES; i++) {
        const RDFNode* rdf_node = rdf_triple_store_get_node_by_id(reasoner->rdf_store, (int)i);
        if (!rdf_node) continue;

        const char* val = rdf_node->value ? rdf_node->value : "unknown";
        int cfc_id = cfc_embed_add_entity(reasoner->embed_state, val);
        if (cfc_id >= 0 && cfc_id < GR_MAX_ENTITIES) {
            reasoner->entity_to_cfc_id[(int)i] = cfc_id;
            reasoner->cfc_to_entity_id[cfc_id] = (int)i;
            reasoner->entity_count++;
        }
    }

    RDFTriple* triples = (RDFTriple*)safe_calloc(triple_cnt + 1, sizeof(RDFTriple));
    if (triples) {
        int found = rdf_triple_store_query(reasoner->rdf_store, -1, -1, -1, triples, triple_cnt);
        for (int i = 0; i < found; i++) {
            int h = triples[i].subject_id;
            int r = triples[i].predicate_id;
            int t = triples[i].object_id;

            if (h >= 0 && r >= 0 && t >= 0) {
                cfc_embed_add_triple(reasoner->embed_state,
                                     reasoner->entity_to_cfc_id[h],
                                     reasoner->relation_to_cfc_id[r],
                                     reasoner->entity_to_cfc_id[t]);
            }
        }
        safe_free((void**)&triples);
    }

    return 0;
}

/* ============ 训练实现 ============ */

int graph_reasoner_train(GraphReasoner* reasoner, int epochs) {
    if (!reasoner) return -1;
    if (!reasoner->embed_state) return -1;

    if (reasoner->entity_count <= 0) {
        if (reasoner->property_graph) {
            if (gr_load_from_property_graph(reasoner) != 0) return -1;
        } else if (reasoner->adjacency_list) {
            if (gr_load_from_adjacency_list(reasoner) != 0) return -1;
        } else if (reasoner->rdf_store) {
            if (gr_load_from_rdf_store(reasoner) != 0) return -1;
        }
    }

    int actual_epochs = (epochs > 0) ? epochs : reasoner->config.max_epochs;
    int ret = cfc_embed_train(reasoner->embed_state, actual_epochs);

    if (ret == 0) {
        reasoner->is_trained = 1;
    }

    reasoner->loss_count = cfc_embed_get_loss_history(reasoner->embed_state,
        reasoner->loss_history, GR_LOSS_HISTORY_MAX);

    return ret;
}

int graph_reasoner_train_from_triples(GraphReasoner* reasoner,
                                       const ReasonTriple* triples, int triple_count,
                                       const char** entity_names, int entity_count,
                                       const char** relation_names, int relation_count,
                                       int epochs) {
    if (!reasoner || !triples || !entity_names || !relation_names) return -1;

    for (int i = 0; i < entity_count && i < GR_MAX_ENTITIES; i++) {
        int cfc_id = cfc_embed_add_entity(reasoner->embed_state, entity_names[i]);
        if (cfc_id >= 0) {
            reasoner->entity_to_cfc_id[i] = cfc_id;
            reasoner->cfc_to_entity_id[cfc_id] = i;
            reasoner->entity_count++;
        }
    }

    for (int i = 0; i < relation_count && i < GR_MAX_RELATIONS; i++) {
        int cfc_id = cfc_embed_add_relation(reasoner->embed_state, relation_names[i]);
        if (cfc_id >= 0) {
            reasoner->relation_to_cfc_id[i] = cfc_id;
            reasoner->cfc_to_rel_id[cfc_id] = i;
            reasoner->relation_count++;
        }
    }

    for (int i = 0; i < triple_count; i++) {
        cfc_embed_add_triple(reasoner->embed_state,
                             triples[i].head_id,
                             triples[i].rel_id,
                             triples[i].tail_id);
    }

    int actual_epochs = (epochs > 0) ? epochs : reasoner->config.max_epochs;
    int ret = cfc_embed_train(reasoner->embed_state, actual_epochs);
    if (ret == 0) reasoner->is_trained = 1;

    return ret;
}

int graph_reasoner_get_loss_history(const GraphReasoner* reasoner,
                                     float* losses, int max_count) {
    if (!reasoner || !losses) return -1;
    int copy_cnt = reasoner->loss_count < max_count ? reasoner->loss_count : max_count;
    for (int i = 0; i < copy_cnt; i++) {
        losses[i] = reasoner->loss_history[i];
    }
    return copy_cnt;
}

/* ============ 预测排序辅助 ============ */

static int gr_compare_link(const void* a, const void* b) {
    const LinkPrediction* pa = (const LinkPrediction*)a;
    const LinkPrediction* pb = (const LinkPrediction*)b;
    if (pa->score < pb->score) return -1;
    if (pa->score > pb->score) return 1;
    return 0;
}

static int gr_compare_float_desc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa > fb) return -1;
    if (fa < fb) return 1;
    return 0;
}

/* ============ 链路预测实现 ============ */

int graph_reasoner_predict_tail(GraphReasoner* reasoner,
                                 int head_id, int rel_id,
                                 const int* candidates, int num_candidates,
                                 LinkPrediction* results, int max_results) {
    if (!reasoner || !results || !reasoner->embed_state) return -1;
    if (max_results <= 0) return 0;

    int dim = reasoner->config.embedding_dim;
    int cfc_head = (head_id >= 0 && head_id < GR_MAX_ENTITIES)
                   ? reasoner->entity_to_cfc_id[head_id] : -1;
    int cfc_rel = (rel_id >= 0 && rel_id < GR_MAX_RELATIONS)
                  ? reasoner->relation_to_cfc_id[rel_id] : -1;
    if (cfc_head < 0 || cfc_rel < 0) return -1;

    int total_candidates = reasoner->entity_count;
    int use_candidates = 0;
    int* cand_buffer = NULL;
    float* score_buffer = NULL;

    if (candidates && num_candidates > 0) {
        use_candidates = num_candidates;
        cand_buffer = (int*)safe_malloc(use_candidates * sizeof(int));
        if (!cand_buffer) return -1;
        memcpy(cand_buffer, candidates, use_candidates * sizeof(int));
    } else {
        use_candidates = total_candidates;
        cand_buffer = (int*)safe_malloc(use_candidates * sizeof(int));
        if (!cand_buffer) return -1;
        for (int i = 0; i < use_candidates; i++) {
            cand_buffer[i] = i;
        }
    }

    if (use_candidates > GR_MAX_CANDIDATES) use_candidates = GR_MAX_CANDIDATES;

    score_buffer = (float*)safe_malloc(use_candidates * sizeof(float));
    if (!score_buffer) {
        safe_free((void**)&cand_buffer);
        return -1;
    }

    if (reasoner->config.reason_mode == GR_REASON_LIQUID_GRAPH) {
        float seed_emb[CFC_EMBED_MAX_DIM];
        cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_head,
                                        seed_emb, dim);

        float* prop_result = (float*)safe_calloc(dim, sizeof(float));
        if (prop_result) {
            memcpy(prop_result, seed_emb, dim * sizeof(float));

            for (int step = 0; step < reasoner->config.graph_prop_steps; step++) {
                if (gr_cfc_step_deep(prop_result, dim,
                                 reasoner->config.cfc_tau,
                                 reasoner->config.cfc_dt,
                                 reasoner->config.cfc_steps) != 0) {
                    safe_free((void**)&prop_result);
                    return -1;
                }

                float* nb_emb = (float*)safe_calloc(dim, sizeof(float));
                if (nb_emb) {
                    int nb_count = 0;
                    if (reasoner->adjacency_list) {
                        int nb_buf[256];
                        float wt_buf[256];
                        int cnt = adjacency_list_get_out_neighbors(
                            reasoner->adjacency_list, head_id,
                            nb_buf, wt_buf, 256);
                        for (int j = 0; j < cnt && j < 256; j++) {
                            int nb_cfc = reasoner->entity_to_cfc_id[nb_buf[j]];
                            if (nb_cfc >= 0) {
                                float nb_e[CFC_EMBED_MAX_DIM];
                                cfc_embed_get_entity_embedding(
                                    reasoner->embed_state, nb_cfc, nb_e, dim);
                                for (int d = 0; d < dim; d++) {
                                    nb_emb[d] += nb_e[d] * wt_buf[j];
                                }
                                nb_count++;
                            }
                        }
                    }
                    if (nb_count > 0) {
                        for (int d = 0; d < dim; d++) {
                            nb_emb[d] /= (float)nb_count;
                            prop_result[d] = prop_result[d] * 0.7f + nb_emb[d] * 0.3f;
                        }
                    }
                    safe_free((void**)&nb_emb);
                }
            }

            for (int i = 0; i < use_candidates; i++) {
                int cand_cfc = reasoner->entity_to_cfc_id[cand_buffer[i]];
                if (cand_cfc < 0) {
                    score_buffer[i] = 1e10f;
                    continue;
                }
                float cand_emb[CFC_EMBED_MAX_DIM];
                cfc_embed_get_entity_embedding(reasoner->embed_state,
                                                cand_cfc, cand_emb, dim);
                float dist = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = prop_result[d] - cand_emb[d];
                    dist += diff * diff;
                }
                score_buffer[i] = sqrtf(dist / (float)dim);
            }
            safe_free((void**)&prop_result);
        }
    } else {
        cfc_embed_predict_tail(reasoner->embed_state, cfc_head, cfc_rel,
                                cand_buffer, use_candidates, score_buffer);
    }

    int* indices = (int*)safe_malloc(use_candidates * sizeof(int));
    if (!indices) {
        safe_free((void**)&cand_buffer);
        safe_free((void**)&score_buffer);
        return -1;
    }
    for (int i = 0; i < use_candidates; i++) indices[i] = i;

    for (int i = 0; i < use_candidates - 1; i++) {
        for (int j = i + 1; j < use_candidates; j++) {
            if (score_buffer[indices[i]] > score_buffer[indices[j]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    int ret_count = use_candidates < max_results ? use_candidates : max_results;
    for (int i = 0; i < ret_count; i++) {
        int idx = indices[i];
        results[i].entity_id = cand_buffer[idx];
        results[i].score = score_buffer[idx];
        results[i].confidence = 1.0f / (1.0f + score_buffer[idx]);
    }

    safe_free((void**)&cand_buffer);
    safe_free((void**)&score_buffer);
    safe_free((void**)&indices);

    return ret_count;
}

int graph_reasoner_predict_head(GraphReasoner* reasoner,
                                 int rel_id, int tail_id,
                                 const int* candidates, int num_candidates,
                                 LinkPrediction* results, int max_results) {
    if (!reasoner || !results || !reasoner->embed_state) return -1;
    if (max_results <= 0) return 0;

    int total_candidates = reasoner->entity_count;
    int use_candidates = (candidates && num_candidates > 0) ? num_candidates : total_candidates;
    if (use_candidates > GR_MAX_CANDIDATES) use_candidates = GR_MAX_CANDIDATES;

    int* cand_buffer = (int*)safe_malloc(use_candidates * sizeof(int));
    float* score_buffer = (float*)safe_malloc(use_candidates * sizeof(float));
    if (!cand_buffer || !score_buffer) {
        safe_free((void**)&cand_buffer);
        safe_free((void**)&score_buffer);
        return -1;
    }

    if (candidates && num_candidates > 0) {
        memcpy(cand_buffer, candidates, use_candidates * sizeof(int));
    } else {
        for (int i = 0; i < use_candidates; i++) cand_buffer[i] = i;
    }

    int cfc_rel = (rel_id >= 0 && rel_id < GR_MAX_RELATIONS)
                  ? reasoner->relation_to_cfc_id[rel_id] : -1;
    int cfc_tail = (tail_id >= 0 && tail_id < GR_MAX_ENTITIES)
                   ? reasoner->entity_to_cfc_id[tail_id] : -1;

    for (int i = 0; i < use_candidates; i++) {
        int cfc_head = reasoner->entity_to_cfc_id[cand_buffer[i]];
        if (cfc_head < 0 || cfc_rel < 0 || cfc_tail < 0) {
            score_buffer[i] = 1e10f;
        } else {
            score_buffer[i] = cfc_embed_score_triple(reasoner->embed_state,
                                                      cfc_head, cfc_rel, cfc_tail);
        }
    }

    int* indices = (int*)safe_malloc(use_candidates * sizeof(int));
    if (!indices) {
        safe_free((void**)&cand_buffer);
        safe_free((void**)&score_buffer);
        return -1;
    }
    for (int i = 0; i < use_candidates; i++) indices[i] = i;

    for (int i = 0; i < use_candidates - 1; i++) {
        for (int j = i + 1; j < use_candidates; j++) {
            if (score_buffer[indices[i]] > score_buffer[indices[j]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    int ret_count = use_candidates < max_results ? use_candidates : max_results;
    for (int i = 0; i < ret_count; i++) {
        int idx = indices[i];
        results[i].entity_id = cand_buffer[idx];
        results[i].score = score_buffer[idx];
        results[i].confidence = 1.0f / (1.0f + score_buffer[idx]);
    }

    safe_free((void**)&cand_buffer);
    safe_free((void**)&score_buffer);
    safe_free((void**)&indices);
    return ret_count;
}

int graph_reasoner_predict_relation(GraphReasoner* reasoner,
                                     int head_id, int tail_id,
                                     const int* candidates, int num_candidates,
                                     LinkPrediction* results, int max_results) {
    if (!reasoner || !results || !reasoner->embed_state) return -1;
    if (max_results <= 0) return 0;

    int total_relations = reasoner->relation_count;
    int use_candidates = (candidates && num_candidates > 0) ? num_candidates : total_relations;
    if (use_candidates > GR_MAX_CANDIDATES) use_candidates = GR_MAX_CANDIDATES;

    int* cand_buffer = (int*)safe_malloc(use_candidates * sizeof(int));
    float* score_buffer = (float*)safe_malloc(use_candidates * sizeof(float));
    if (!cand_buffer || !score_buffer) {
        safe_free((void**)&cand_buffer);
        safe_free((void**)&score_buffer);
        return -1;
    }

    if (candidates && num_candidates > 0) {
        memcpy(cand_buffer, candidates, use_candidates * sizeof(int));
    } else {
        for (int i = 0; i < use_candidates; i++) cand_buffer[i] = i;
    }

    int cfc_head = reasoner->entity_to_cfc_id[head_id];
    int cfc_tail = reasoner->entity_to_cfc_id[tail_id];

    for (int i = 0; i < use_candidates; i++) {
        int cfc_rel = reasoner->relation_to_cfc_id[cand_buffer[i]];
        if (cfc_head < 0 || cfc_rel < 0 || cfc_tail < 0) {
            score_buffer[i] = 1e10f;
        } else {
            score_buffer[i] = cfc_embed_score_triple(reasoner->embed_state,
                                                      cfc_head, cfc_rel, cfc_tail);
        }
    }

    int* indices = (int*)safe_malloc(use_candidates * sizeof(int));
    if (!indices) {
        safe_free((void**)&cand_buffer);
        safe_free((void**)&score_buffer);
        return -1;
    }
    for (int i = 0; i < use_candidates; i++) indices[i] = i;

    for (int i = 0; i < use_candidates - 1; i++) {
        for (int j = i + 1; j < use_candidates; j++) {
            if (score_buffer[indices[i]] > score_buffer[indices[j]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    int ret_count = use_candidates < max_results ? use_candidates : max_results;
    for (int i = 0; i < ret_count; i++) {
        int idx = indices[i];
        results[i].entity_id = cand_buffer[idx];
        results[i].score = score_buffer[idx];
        results[i].confidence = 1.0f / (1.0f + score_buffer[idx]);
    }

    safe_free((void**)&cand_buffer);
    safe_free((void**)&score_buffer);
    safe_free((void**)&indices);
    return ret_count;
}

/* ============ 三元组分类实现 ============ */

int graph_reasoner_classify_triple(GraphReasoner* reasoner,
                                    int head_id, int rel_id, int tail_id,
                                    float* confidence) {
    if (!reasoner || !confidence) return -1;

    int cfc_head = reasoner->entity_to_cfc_id[head_id];
    int cfc_rel = reasoner->relation_to_cfc_id[rel_id];
    int cfc_tail = reasoner->entity_to_cfc_id[tail_id];

    if (cfc_head < 0 || cfc_rel < 0 || cfc_tail < 0) {
        *confidence = 0.0f;
        return 0;
    }

    float score = cfc_embed_score_triple(reasoner->embed_state,
                                          cfc_head, cfc_rel, cfc_tail);
    *confidence = 1.0f / (1.0f + score);

    float margin = reasoner->config.margin;
    if (score <= margin * 0.5f) return 1;
    if (score <= margin) return 1;
    return 0;
}

int graph_reasoner_classify_triples_batch(GraphReasoner* reasoner,
                                           const ReasonTriple* triples, int count,
                                           int* results, float* confidences) {
    if (!reasoner || !triples || !results || !confidences) return -1;

    for (int i = 0; i < count; i++) {
        results[i] = graph_reasoner_classify_triple(reasoner,
                                                     triples[i].head_id,
                                                     triples[i].rel_id,
                                                     triples[i].tail_id,
                                                     &confidences[i]);
    }
    return 0;
}

/* ============ 路径推理实现 ============ */

static float gr_path_score(const GraphReasoner* reasoner,
                            const PathNode* path, int length, float* liquid_energy) {
    if (!reasoner || !path || length <= 1) return 1e10f;

    float total = 0.0f;
    float energy = 0.0f;

    for (int i = 0; i < length - 1; i++) {
        int h = path[i].entity_id;
        int r = path[i + 1].relation_id;
        int t = path[i + 1].entity_id;

        if (h < 0 || r < 0 || t < 0) {
            total += 10.0f;
            continue;
        }

        int cfc_h = reasoner->entity_to_cfc_id[h];
        int cfc_r = reasoner->relation_to_cfc_id[r];
        int cfc_t = reasoner->entity_to_cfc_id[t];

        if (cfc_h < 0 || cfc_r < 0 || cfc_t < 0) {
            total += 10.0f;
            continue;
        }

        float step_score = cfc_embed_score_triple(reasoner->embed_state,
                                                    cfc_h, cfc_r, cfc_t);

        int dim = reasoner->config.embedding_dim;
        float h_emb[CFC_EMBED_MAX_DIM];
        cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_h, h_emb, dim);
        energy += gr_compute_node_energy(h_emb, dim);

        total += step_score;
    }

    if (liquid_energy) *liquid_energy = energy;
    return total / (float)(length - 1);
}

int graph_reasoner_find_paths(GraphReasoner* reasoner,
                               int source_id, int target_id,
                               int max_depth, PathResultSet* results) {
    if (!reasoner || !results) return -1;
    if (source_id == target_id || max_depth <= 0) return 0;

    int max_d = max_depth;
    if (max_d > GR_MAX_PATH_LENGTH) max_d = GR_MAX_PATH_LENGTH;

    results->count = 0;

    int* current_path = (int*)safe_calloc(max_d + 1, sizeof(int));
    int* current_rels = (int*)safe_calloc(max_d, sizeof(int));
    int* visited = (int*)safe_calloc(reasoner->entity_count + 1, sizeof(int));
    if (!current_path || !current_rels || !visited) {
        safe_free((void**)&current_path);
        safe_free((void**)&current_rels);
        safe_free((void**)&visited);
        return -1;
    }

    current_path[0] = source_id;
    visited[source_id] = 1;
    int depth = 0;

    typedef struct {
        int node_id;
        int rel_id;
        int neighbor_idx;
        int num_neighbors;
        int* neighbors;
        int* rels;
    } DFSState;

    DFSState* stack = (DFSState*)safe_calloc(max_d + 1, sizeof(DFSState));
    if (!stack) {
        safe_free((void**)&current_path);
        safe_free((void**)&current_rels);
        safe_free((void**)&visited);
        return -1;
    }

    int nb_buf0[256];
    int rel_buf0[256];
    int nb_cnt0 = 0;

    (void)rel_buf0;

    if (reasoner->adjacency_list) {
        nb_cnt0 = adjacency_list_get_out_neighbors(
            reasoner->adjacency_list, source_id, nb_buf0, NULL, 256);
    }

    stack[0].node_id = source_id;
    stack[0].rel_id = -1;
    stack[0].neighbor_idx = 0;
    stack[0].num_neighbors = nb_cnt0;
    stack[0].neighbors = (int*)safe_malloc((nb_cnt0 + 1) * sizeof(int));
    stack[0].rels = (int*)safe_malloc((nb_cnt0 + 1) * sizeof(int));
    if (stack[0].neighbors && nb_cnt0 > 0) {
        for (int i = 0; i < nb_cnt0; i++) {
            stack[0].neighbors[i] = nb_buf0[i];
            stack[0].rels[i] = 0;
        }
    }

    while (depth >= 0 && results->count < GR_MAX_PATHS) {
        DFSState* cur = &stack[depth];

        if (cur->neighbor_idx >= cur->num_neighbors) {
            visited[cur->node_id] = 0;
            safe_free((void**)&cur->neighbors);
            safe_free((void**)&cur->rels);
            depth--;
            continue;
        }

        int next_node = cur->neighbors[cur->neighbor_idx];
        int next_rel = cur->rels[cur->neighbor_idx];
        cur->neighbor_idx++;

        if (visited[next_node]) continue;

        current_path[depth + 1] = next_node;
        current_rels[depth] = next_rel;

        if (next_node == target_id) {
            ReasonPath* rp = &results->paths[results->count];
            rp->length = depth + 2;
            rp->nodes[0].entity_id = source_id;
            rp->nodes[0].relation_id = -1;
            rp->nodes[0].weight = 1.0f;

            float emb0[CFC_EMBED_MAX_DIM];
            cfc_embed_get_entity_embedding(reasoner->embed_state,
                reasoner->entity_to_cfc_id[source_id], emb0,
                reasoner->config.embedding_dim);
            rp->nodes[0].state_energy = gr_compute_node_energy(
                emb0, reasoner->config.embedding_dim);

            for (int i = 0; i <= depth; i++) {
                rp->nodes[i + 1].entity_id = current_path[i + 1];
                rp->nodes[i + 1].relation_id = current_rels[i];
                rp->nodes[i + 1].weight = 1.0f;

                float emb[CFC_EMBED_MAX_DIM];
                cfc_embed_get_entity_embedding(reasoner->embed_state,
                    reasoner->entity_to_cfc_id[current_path[i + 1]], emb,
                    reasoner->config.embedding_dim);
                rp->nodes[i + 1].state_energy = gr_compute_node_energy(
                    emb, reasoner->config.embedding_dim);
            }

            float liq_energy = 0.0f;
            rp->total_score = gr_path_score(reasoner, rp->nodes,
                                             rp->length, &liq_energy);
            rp->liquid_energy = liq_energy;
            results->count++;
            continue;
        }

        if (depth + 1 < max_d - 1) {
            visited[next_node] = 1;
            depth++;

            DFSState* next = &stack[depth];
            next->node_id = next_node;
            next->rel_id = next_rel;
            next->neighbor_idx = 0;

            int nb_buf[256];
            int rel_buf[256];
            int nb_cnt = 0;

            if (reasoner->adjacency_list) {
                nb_cnt = adjacency_list_get_out_neighbors(
                    reasoner->adjacency_list, next_node, nb_buf, NULL, 256);
            }

            next->num_neighbors = nb_cnt;
            next->neighbors = (int*)safe_malloc((nb_cnt + 1) * sizeof(int));
            next->rels = (int*)safe_malloc((nb_cnt + 1) * sizeof(int));
            if (next->neighbors && nb_cnt > 0) {
                for (int i = 0; i < nb_cnt; i++) {
                    next->neighbors[i] = nb_buf[i];
                    next->rels[i] = rel_buf ? rel_buf[i] : 0;
                }
            }
        }
    }

    for (int i = 0; i <= depth && i <= max_d; i++) {
        safe_free((void**)&stack[i].neighbors);
        safe_free((void**)&stack[i].rels);
    }

    safe_free((void**)&current_path);
    safe_free((void**)&current_rels);
    safe_free((void**)&visited);
    safe_free((void**)&stack);

    if (results->count > 1) {
        for (int i = 0; i < results->count - 1; i++) {
            for (int j = i + 1; j < results->count; j++) {
                if (results->paths[i].total_score > results->paths[j].total_score) {
                    ReasonPath tmp = results->paths[i];
                    results->paths[i] = results->paths[j];
                    results->paths[j] = tmp;
                }
            }
        }
    }

    return results->count;
}

/* ============ 路径排序实现（P3.5） ============ */

int graph_reasoner_rank_paths(GraphReasoner* reasoner,
                               PathResultSet* path_set,
                               int mode) {
    if (!reasoner || !path_set || path_set->count <= 0) return -1;

    int count = path_set->count;
    if (count > GR_MAX_PATHS) count = GR_MAX_PATHS;

    switch ((GraphReasonPathRankMode)mode) {
        case GR_RANK_SHORTEST: {
            for (int i = 0; i < count - 1; i++) {
                for (int j = i + 1; j < count; j++) {
                    if (path_set->paths[i].length > path_set->paths[j].length) {
                        ReasonPath tmp = path_set->paths[i];
                        path_set->paths[i] = path_set->paths[j];
                        path_set->paths[j] = tmp;
                    }
                }
            }
            break;
        }
        case GR_RANK_HIGHEST_CONFIDENCE: {
            float* confs = (float*)safe_calloc(count, sizeof(float));
            if (!confs) return -1;
            for (int i = 0; i < count; i++) {
                float best_conf = 0.0f;
                for (int n = 0; n < path_set->paths[i].length; n++) {
                    int eid = path_set->paths[i].nodes[n].entity_id;
                    if (eid >= 0 && reasoner->embed_state) {
                        int cfc_e = reasoner->entity_to_cfc_id[eid];
                        if (cfc_e >= 0) {
                            float emb[CFC_EMBED_MAX_DIM];
                            int dim = reasoner->config.embedding_dim;
                            if (cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_e, emb, dim) > 0) {
                                float en = 0.0f;
                                for (int d = 0; d < dim; d++) en += emb[d] * emb[d];
                                float eng = sqrtf(en / (float)dim);
                                float conf = 1.0f / (1.0f + eng);
                                if (conf > best_conf) best_conf = conf;
                            }
                        }
                    }
                }
                confs[i] = best_conf;
            }
            for (int i = 0; i < count - 1; i++) {
                for (int j = i + 1; j < count; j++) {
                    if (confs[i] < confs[j]) {
                        float tc = confs[i]; confs[i] = confs[j]; confs[j] = tc;
                        ReasonPath tmp = path_set->paths[i];
                        path_set->paths[i] = path_set->paths[j];
                        path_set->paths[j] = tmp;
                    }
                }
            }
            safe_free((void**)&confs);
            break;
        }
        case GR_RANK_LIQUID_ENERGY: {
            for (int i = 0; i < count - 1; i++) {
                for (int j = i + 1; j < count; j++) {
                    if (path_set->paths[i].liquid_energy > path_set->paths[j].liquid_energy) {
                        ReasonPath tmp = path_set->paths[i];
                        path_set->paths[i] = path_set->paths[j];
                        path_set->paths[j] = tmp;
                    }
                }
            }
            break;
        }
        case GR_RANK_TOTAL_SCORE:
        default: {
            for (int i = 0; i < count - 1; i++) {
                for (int j = i + 1; j < count; j++) {
                    if (path_set->paths[i].total_score > path_set->paths[j].total_score) {
                        ReasonPath tmp = path_set->paths[i];
                        path_set->paths[i] = path_set->paths[j];
                        path_set->paths[j] = tmp;
                    }
                }
            }
            break;
        }
    }

    path_set->count = count;
    return count;
}

int graph_reasoner_multi_hop_reason(GraphReasoner* reasoner,
                                     int seed_entity, int hops,
                                     int* results, float* scores,
                                     int max_results) {
    if (!reasoner || !results || !scores || !reasoner->embed_state) return -1;
    if (max_results <= 0 || hops <= 0) return 0;

    int cfc_seed = reasoner->entity_to_cfc_id[seed_entity];
    if (cfc_seed < 0) return -1;

    int dim = reasoner->config.embedding_dim;

    float seed_emb[CFC_EMBED_MAX_DIM];
    cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_seed, seed_emb, dim);

    if (reasoner->config.reason_mode == GR_REASON_LIQUID_GRAPH) {
        float current_emb[CFC_EMBED_MAX_DIM];
        memcpy(current_emb, seed_emb, dim * sizeof(float));

        for (int h = 0; h < hops; h++) {
            if (reasoner->adjacency_list) {
                int nb_buf[256];
                float wt_buf[256];
                int nb_cnt = adjacency_list_get_out_neighbors(
                    reasoner->adjacency_list, seed_entity, nb_buf, wt_buf, 256);

                if (nb_cnt > 0) {
                    float aggregate[CFC_EMBED_MAX_DIM];
                    memset(aggregate, 0, dim * sizeof(float));
                    float total_wt = 0.0f;

                    for (int j = 0; j < nb_cnt && j < 256; j++) {
                        int nb_cfc = reasoner->entity_to_cfc_id[nb_buf[j]];
                        if (nb_cfc >= 0) {
                            float nb_e[CFC_EMBED_MAX_DIM];
                            cfc_embed_get_entity_embedding(reasoner->embed_state,
                                                            nb_cfc, nb_e, dim);
                            float w = (wt_buf[j] > 0) ? wt_buf[j] : 1.0f;
                            for (int d = 0; d < dim; d++) {
                                aggregate[d] += nb_e[d] * w;
                            }
                            total_wt += w;
                        }
                    }

                    if (total_wt > 0) {
                        for (int d = 0; d < dim; d++) {
                            aggregate[d] /= total_wt;
                        }
                        float combined[CFC_EMBED_MAX_DIM];
                        for (int d = 0; d < dim; d++) {
                            combined[d] = current_emb[d] * 0.5f + aggregate[d] * 0.5f;
                        }
                        /* 使用gr_cfc_step_deep替代独立的cfc_cell_forward，
                         * 在单一LNN架构中所有连续态演化由同一动力学方程驱动 */
                        memcpy(current_emb, combined, dim * sizeof(float));
                        if (gr_cfc_step_deep(current_emb, dim,
                                         reasoner->config.cfc_tau,
                                         reasoner->config.cfc_dt, 1) != 0) {
                            return -1;
                        }
                    }
                }
            }

            if (gr_cfc_step_deep(current_emb, dim,
                             reasoner->config.cfc_tau,
                             reasoner->config.cfc_dt,
                             reasoner->config.cfc_steps) != 0) {
                return -1;
            }
        }

        float* all_scores = (float*)safe_malloc(reasoner->entity_count * sizeof(float));
        if (!all_scores) return -1;

        for (int i = 0; i < reasoner->entity_count; i++) {
            if (i == seed_entity) {
                all_scores[i] = 1e10f;
                continue;
            }
            int cfc_i = reasoner->entity_to_cfc_id[i];
            if (cfc_i < 0) {
                all_scores[i] = 1e10f;
                continue;
            }
            float e_emb[CFC_EMBED_MAX_DIM];
            cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_i, e_emb, dim);
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = current_emb[d] - e_emb[d];
                dist += diff * diff;
            }
            all_scores[i] = sqrtf(dist / (float)dim);
        }

        int* indices = (int*)safe_malloc(reasoner->entity_count * sizeof(int));
        if (!indices) {
            safe_free((void**)&all_scores);
            return -1;
        }
        for (int i = 0; i < reasoner->entity_count; i++) indices[i] = i;

        for (int i = 0; i < reasoner->entity_count - 1; i++) {
            for (int j = i + 1; j < reasoner->entity_count; j++) {
                if (all_scores[indices[i]] > all_scores[indices[j]]) {
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }

        int ret_cnt = reasoner->entity_count < max_results
                      ? reasoner->entity_count : max_results;
        for (int i = 0; i < ret_cnt; i++) {
            results[i] = indices[i];
            scores[i] = all_scores[indices[i]];
        }

        safe_free((void**)&all_scores);
        safe_free((void**)&indices);
        return ret_cnt;
    }

    float prop_result[CFC_EMBED_MAX_DIM];
    cfc_embed_graph_propagate(reasoner->embed_state, cfc_seed,
                               hops, prop_result, dim);

    float* all_scores = (float*)safe_malloc(reasoner->entity_count * sizeof(float));
    if (!all_scores) return -1;

    for (int i = 0; i < reasoner->entity_count; i++) {
        if (i == cfc_seed) {
            all_scores[i] = 1e10f;
            continue;
        }
        int cfc_i = reasoner->entity_to_cfc_id[i];
        if (cfc_i < 0) {
            all_scores[i] = 1e10f;
            continue;
        }
        float e_emb[CFC_EMBED_MAX_DIM];
        cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_i, e_emb, dim);
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = prop_result[d] - e_emb[d];
            dist += diff * diff;
        }
        all_scores[i] = sqrtf(dist / (float)dim);
    }

    int* indices = (int*)safe_malloc(reasoner->entity_count * sizeof(int));
    if (!indices) {
        safe_free((void**)&all_scores);
        return -1;
    }
    for (int i = 0; i < reasoner->entity_count; i++) indices[i] = i;

    for (int i = 0; i < reasoner->entity_count - 1; i++) {
        for (int j = i + 1; j < reasoner->entity_count; j++) {
            if (all_scores[indices[i]] > all_scores[indices[j]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    int ret_cnt = reasoner->entity_count < max_results
                  ? reasoner->entity_count : max_results;
    for (int i = 0; i < ret_cnt; i++) {
        results[i] = indices[i];
        scores[i] = all_scores[indices[i]];
    }

    safe_free((void**)&all_scores);
    safe_free((void**)&indices);
    return ret_cnt;
}

/* ============ 增强多跳推理实现（P3.5） ============ */

int graph_reasoner_multi_hop_enhanced(GraphReasoner* reasoner,
                                       int seed_entity, int hops,
                                       int* results, float* scores,
                                       int max_results,
                                       ReasoningChainSet* chains) {
    if (!reasoner || !results || !scores || !reasoner->embed_state) return -1;
    if (max_results <= 0 || hops <= 0) return 0;

    int cfc_seed = reasoner->entity_to_cfc_id[seed_entity];
    if (cfc_seed < 0) return -1;

    int dim = reasoner->config.embedding_dim;

    float seed_emb[CFC_EMBED_MAX_DIM];
    cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_seed, seed_emb, dim);

    int chain_count = 0;
    float* chain_scores = NULL;
    ReasoningChain* tmp_chains = NULL;

    if (chains) {
        chains->count = 0;
        chain_scores = (float*)safe_calloc(GR_MAX_CHAINS, sizeof(float));
        tmp_chains = (ReasoningChain*)safe_calloc(GR_MAX_CHAINS, sizeof(ReasoningChain));
    }

    float current_emb[CFC_EMBED_MAX_DIM];
    memcpy(current_emb, seed_emb, dim * sizeof(float));

    int* visited = (int*)safe_calloc(reasoner->entity_count + 1, sizeof(int));
    if (!visited) return -1;
    visited[seed_entity] = 1;

    typedef struct {
        int entity_id;
        int prev_entity;
        int relation_id;
        int depth;
        float arrival_score;
        float cumul_energy;
        int chain_idx;
    } BFSNode;

    BFSNode* bfs_queue = (BFSNode*)safe_calloc(reasoner->entity_count + 1, sizeof(BFSNode));
    if (!bfs_queue) {
        safe_free((void**)&visited);
        return -1;
    }

    int head = 0, tail = 0;

    bfs_queue[tail].entity_id = seed_entity;
    bfs_queue[tail].prev_entity = -1;
    bfs_queue[tail].relation_id = -1;
    bfs_queue[tail].depth = 0;
    bfs_queue[tail].arrival_score = 0.0f;
    bfs_queue[tail].cumul_energy = 0.0f;
    bfs_queue[tail].chain_idx = 0;
    tail++;

    int result_count = 0;

    while (head < tail && result_count < max_results) {
        BFSNode cur = bfs_queue[head];
        head++;

        if (cur.depth > hops) continue;

        if (cur.entity_id != seed_entity) {
            int cfc_cur = reasoner->entity_to_cfc_id[cur.entity_id];
            if (cfc_cur >= 0) {
                float cur_emb[CFC_EMBED_MAX_DIM];
                cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_cur, cur_emb, dim);

                float dist = 0.0f;
                float liquid_energy = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = current_emb[d] - cur_emb[d];
                    dist += diff * diff;
                    float gate = 1.0f / (1.0f + expf(-cur_emb[d]));
                    float act = tanhf(0.8f * cur_emb[d]);
                    float dh = -cur_emb[d] / (reasoner->config.cfc_tau + GR_EPSILON) + gate * act;
                    liquid_energy += fabsf(dh);
                }
                float score = sqrtf(dist / (float)dim) + liquid_energy * 0.1f;
                results[result_count] = cur.entity_id;
                scores[result_count] = score;
                result_count++;

                if (chains && chain_count < GR_MAX_CHAINS) {
                    ReasoningChain* rc = &tmp_chains[chain_count];
                    rc->length = 0;
                    rc->total_score = score;
                    rc->avg_energy = liquid_energy / (float)dim;

                    int trace_entity = cur.entity_id;
                    int trace_depth = cur.depth;
                    BFSNode* trace_nodes = (BFSNode*)safe_calloc((hops + 2), sizeof(BFSNode));
                    int trace_pos = 0;
                    if (trace_nodes) {
                        for (int t = head - 1; t >= 0; t--) {
                            if (bfs_queue[t].entity_id == trace_entity && bfs_queue[t].depth == trace_depth) {
                                trace_nodes[trace_pos++] = bfs_queue[t];
                                trace_entity = bfs_queue[t].prev_entity;
                                trace_depth--;
                                if (trace_entity < 0 || trace_depth < 0) break;
                            }
                        }
                        for (int rev = trace_pos - 1; rev >= 0 && rc->length < GR_MAX_CHAIN_LENGTH; rev--) {
                            int idx = trace_pos - 1 - rev;
                            ReasoningChainNode* node = &rc->nodes[rc->length];
                            node->entity_id = trace_nodes[idx].entity_id;
                            node->relation_id = trace_nodes[idx].relation_id;
                            node->arrival_score = trace_nodes[idx].arrival_score;
                            node->hop_depth = trace_nodes[idx].depth;
                            {
                                float ne[CFC_EMBED_MAX_DIM];
                                int nc = reasoner->entity_to_cfc_id[node->entity_id];
                                if (nc >= 0) {
                                    cfc_embed_get_entity_embedding(reasoner->embed_state, nc, ne, dim);
                                    float en = 0.0f;
                                    for (int d = 0; d < dim; d++) en += ne[d] * ne[d];
                                    node->liquid_energy = sqrtf(en / (float)dim);
                                } else {
                                    node->liquid_energy = 0.0f;
                                }
                            }
                            rc->length++;
                        }
                        safe_free((void**)&trace_nodes);
                    }
                    chain_scores[chain_count] = score;
                    chain_count++;
                }

                float combined[CFC_EMBED_MAX_DIM];
                for (int d = 0; d < dim; d++) {
                    combined[d] = current_emb[d] * 0.7f + cur_emb[d] * 0.3f;
                }
                memcpy(current_emb, combined, dim * sizeof(float));
                if (gr_cfc_step_deep(current_emb, dim,
                                 reasoner->config.cfc_tau,
                                 reasoner->config.cfc_dt, 1) != 0) {
                    return -1;
                }
            }
        }

        if (cur.depth < hops && reasoner->adjacency_list) {
            int nb_buf[256];
            int nb_cnt = adjacency_list_get_out_neighbors(
                reasoner->adjacency_list, cur.entity_id, nb_buf, NULL, 256);

            for (int n = 0; n < nb_cnt && n < 256; n++) {
                int nb = nb_buf[n];
                if (nb < 0 || visited[nb]) continue;
                visited[nb] = 1;

                float energy_score = 0.0f;
                int cfc_nb = reasoner->entity_to_cfc_id[nb];
                if (cfc_nb >= 0) {
                    float nb_emb[CFC_EMBED_MAX_DIM];
                    cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_nb, nb_emb, dim);
                    for (int d = 0; d < dim; d++) {
                        energy_score += nb_emb[d] * nb_emb[d];
                    }
                    energy_score = sqrtf(energy_score / (float)dim);
                }

                bfs_queue[tail].entity_id = nb;
                bfs_queue[tail].prev_entity = cur.entity_id;
                bfs_queue[tail].relation_id = 0;
                bfs_queue[tail].depth = cur.depth + 1;
                bfs_queue[tail].arrival_score = energy_score;
                bfs_queue[tail].cumul_energy = cur.cumul_energy + energy_score;
                bfs_queue[tail].chain_idx = cur.chain_idx;
                tail++;
            }
        }
    }

    for (int i = 0; i < result_count - 1; i++) {
        for (int j = i + 1; j < result_count; j++) {
            if (scores[i] > scores[j]) {
                float ts = scores[i]; scores[i] = scores[j]; scores[j] = ts;
                int te = results[i]; results[i] = results[j]; results[j] = te;
            }
        }
    }

    if (chains && chain_scores && tmp_chains) {
        int* chain_indices = (int*)safe_calloc(chain_count, sizeof(int));
        if (chain_indices) {
            for (int i = 0; i < chain_count; i++) chain_indices[i] = i;
            for (int i = 0; i < chain_count - 1; i++) {
                for (int j = i + 1; j < chain_count; j++) {
                    if (chain_scores[chain_indices[i]] > chain_scores[chain_indices[j]]) {
                        int t = chain_indices[i];
                        chain_indices[i] = chain_indices[j];
                        chain_indices[j] = t;
                    }
                }
            }
            chains->count = chain_count < GR_MAX_CHAINS ? chain_count : GR_MAX_CHAINS;
            for (int i = 0; i < chains->count; i++) {
                chains->chains[i] = tmp_chains[chain_indices[i]];
            }
            safe_free((void**)&chain_indices);
        }
    }

    safe_free((void**)&chain_scores);
    safe_free((void**)&tmp_chains);
    safe_free((void**)&visited);
    safe_free((void**)&bfs_queue);

    return result_count;
}

/* ============ LNN图推理实现（P3.5） ============ */

int graph_reasoner_lnn_infer(GraphReasoner* reasoner,
                              int seed_entity,
                              int propagation_steps,
                              int* results, float* scores,
                              int max_results) {
    if (!reasoner || !results || !scores || !reasoner->embed_state) return -1;
    if (max_results <= 0 || propagation_steps <= 0) return 0;

    int dim = reasoner->config.embedding_dim;
    int total_entities = reasoner->entity_count;
    if (total_entities <= 0) return -1;

    float* lnn_state = (float*)safe_calloc(total_entities * dim, sizeof(float));
    float* lnn_buffer = (float*)safe_calloc(total_entities * dim, sizeof(float));
    if (!lnn_state || !lnn_buffer) {
        safe_free((void**)&lnn_state);
        safe_free((void**)&lnn_buffer);
        return -1;
    }

    int start_id = 0;
    int end_id = total_entities;
    if (seed_entity >= 0 && seed_entity < total_entities) {
        start_id = seed_entity;
        end_id = seed_entity + 1;
    }

    for (int e = start_id; e < end_id; e++) {
        int cfc_e = reasoner->entity_to_cfc_id[e];
        if (cfc_e < 0) continue;
        cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_e,
                                        lnn_state + (e - start_id) * dim, dim);
    }

    float tau = reasoner->config.cfc_tau;
    float dt = reasoner->config.cfc_dt;

    for (int step = 0; step < propagation_steps; step++) {
        int batch_size = end_id - start_id;
        memcpy(lnn_buffer, lnn_state, batch_size * dim * sizeof(float));

        for (int e = 0; e < batch_size; e++) {
            float* h = lnn_state + e * dim;

            if (gr_cfc_step_deep(h, dim, tau, dt, reasoner->config.cfc_steps) != 0) {
                safe_free((void**)&lnn_state);
                safe_free((void**)&lnn_buffer);
                return -1;
            }

            if (reasoner->adjacency_list) {
                int entity_id = start_id + e;
                int nb_buf[256];
                float wt_buf[256];
                int nb_cnt = adjacency_list_get_out_neighbors(
                    reasoner->adjacency_list, entity_id, nb_buf, wt_buf, 256);

                if (nb_cnt > 0) {
                    float* agg = (float*)safe_calloc(dim, sizeof(float));
                    float total_wt = 0.0f;
                    if (agg) {
                        for (int n = 0; n < nb_cnt && n < 256; n++) {
                            int nb_id = nb_buf[n];
                            if (nb_id < 0 || nb_id >= total_entities) continue;
                            int cfc_nb = reasoner->entity_to_cfc_id[nb_id];
                            if (cfc_nb < 0) continue;
                            float nb_emb[CFC_EMBED_MAX_DIM];
                            cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_nb, nb_emb, dim);
                            float w = (wt_buf[n] > 0.0f) ? wt_buf[n] : 1.0f;
                            for (int d = 0; d < dim; d++) agg[d] += nb_emb[d] * w;
                            total_wt += w;
                        }
                        if (total_wt > 0) {
                            for (int d = 0; d < dim; d++) agg[d] /= total_wt;
                            for (int d = 0; d < dim; d++) {
                                h[d] = h[d] * 0.7f + agg[d] * 0.3f;
                            }
                        }
                        safe_free((void**)&agg);
                    }
                }
            }
        }
    }

    int result_count = 0;
    float* all_stable_scores = (float*)safe_calloc(total_entities, sizeof(float));
    if (!all_stable_scores) {
        safe_free((void**)&lnn_state);
        safe_free((void**)&lnn_buffer);
        return -1;
    }

    for (int e = 0; e < total_entities; e++) {
        int cfc_e = reasoner->entity_to_cfc_id[e];
        if (cfc_e < 0) {
            all_stable_scores[e] = 1e10f;
            continue;
        }

        float origin_emb[CFC_EMBED_MAX_DIM];
        cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_e, origin_emb, dim);

        float stable_score = 0.0f;
        if (e >= start_id && e < end_id) {
            float* evolved = lnn_state + (e - start_id) * dim;
            float euclidean_dist = 0.0f;
            float energy_diff = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = evolved[d] - origin_emb[d];
                euclidean_dist += diff * diff;
                float gate = 1.0f / (1.0f + expf(-evolved[d]));
                float act = tanhf(0.8f * evolved[d]);
                float dh = -evolved[d] / (tau + GR_EPSILON) + gate * act;
                energy_diff += fabsf(dh);
            }
            stable_score = sqrtf(euclidean_dist / (float)dim) + energy_diff * 0.05f;
        } else {
            float emb[CFC_EMBED_MAX_DIM];
            memcpy(emb, origin_emb, dim * sizeof(float));
            float saved[CFC_EMBED_MAX_DIM];
            memcpy(saved, emb, dim * sizeof(float));
            if (gr_cfc_step_deep(emb, dim, tau, dt, reasoner->config.cfc_steps * propagation_steps) != 0) {
                safe_free((void**)&all_stable_scores);
                safe_free((void**)&lnn_state);
                safe_free((void**)&lnn_buffer);
                return -1;
            }
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = emb[d] - saved[d];
                dist += diff * diff;
            }
            stable_score = sqrtf(dist / (float)dim);
        }
        all_stable_scores[e] = stable_score;
    }

    int* order = (int*)safe_calloc(total_entities, sizeof(int));
    if (order) {
        for (int i = 0; i < total_entities; i++) order[i] = i;
        for (int i = 0; i < total_entities - 1; i++) {
            for (int j = i + 1; j < total_entities; j++) {
                if (all_stable_scores[order[i]] > all_stable_scores[order[j]]) {
                    int t = order[i];
                    order[i] = order[j];
                    order[j] = t;
                }
            }
        }

        result_count = total_entities < max_results ? total_entities : max_results;
        for (int i = 0; i < result_count; i++) {
            results[i] = order[i];
            scores[i] = all_stable_scores[order[i]];
        }

        safe_free((void**)&order);
    }

    safe_free((void**)&all_stable_scores);
    safe_free((void**)&lnn_state);
    safe_free((void**)&lnn_buffer);

    return result_count;
}

/* ============ 规则挖掘实现 ============ */

static int gr_enum_paths_for_rule(GraphReasoner* reasoner,
                                   int source_entity, int max_len,
                                   RelationPathCache* cache) {
    if (!reasoner || !cache) return -1;
    if (!reasoner->adjacency_list) return -1;

    int* visited = (int*)safe_calloc(reasoner->entity_count + 1, sizeof(int));
    int* path_entities = (int*)safe_calloc(max_len + 1, sizeof(int));
    int* path_relations = (int*)safe_calloc(max_len, sizeof(int));
    if (!visited || !path_entities || !path_relations) {
        safe_free((void**)&visited);
        safe_free((void**)&path_entities);
        safe_free((void**)&path_relations);
        return -1;
    }

    path_entities[0] = source_entity;
    visited[source_entity] = 1;
    cache->count = 0;

    int depth = 0;
    typedef struct {
        int entity_id;
        int neighbor_idx;
        int num_neighbors;
        int* neighbors;
        int* rels;
    } StackFrame;

    StackFrame* frame = (StackFrame*)safe_calloc(max_len + 1, sizeof(StackFrame));
    if (!frame) {
        safe_free((void**)&visited);
        safe_free((void**)&path_entities);
        safe_free((void**)&path_relations);
        return -1;
    }

    int nb_init[256];
    int nb_init_cnt = adjacency_list_get_out_neighbors(
        reasoner->adjacency_list, source_entity, nb_init, NULL, 256);
    frame[0].entity_id = source_entity;
    frame[0].num_neighbors = nb_init_cnt;
    frame[0].neighbors = (int*)safe_malloc((nb_init_cnt + 1) * sizeof(int));
    frame[0].rels = (int*)safe_malloc((nb_init_cnt + 1) * sizeof(int));
    if (frame[0].neighbors && nb_init_cnt > 0) {
        for (int i = 0; i < nb_init_cnt; i++) {
            frame[0].neighbors[i] = nb_init[i];
            /* ZS-003修复: 从源节点出边ID中获取实际关系类型，而非硬编码为0 */
            const ALNode* src_node = adjacency_list_get_node(
                reasoner->adjacency_list, source_entity);
            if (src_node && src_node->out_edge_ids && i < (int)src_node->out_degree) {
                frame[0].rels[i] = src_node->out_edge_ids[i];
            } else {
                frame[0].rels[i] = i + 1;
            }
        }
    }

    while (depth >= 0 && cache->count < GR_MAX_PATHS) {
        StackFrame* cur = &frame[depth];

        if (cur->neighbor_idx >= cur->num_neighbors) {
            visited[cur->entity_id] = 0;
            safe_free((void**)&cur->neighbors);
            safe_free((void**)&cur->rels);
            depth--;
            continue;
        }

        int next = cur->neighbors[cur->neighbor_idx];
        cur->neighbor_idx++;

        if (visited[next]) continue;

        path_entities[depth + 1] = next;
        /* ZS-003修复: 使用当前栈帧中存储的实际关系ID，而非硬编码0 */
        path_relations[depth] = (cur->rels && cur->neighbor_idx > 0 ?
                                 cur->rels[cur->neighbor_idx - 1] : depth + 1);

        if (depth + 1 >= 2) {
            RelationPath* rp = &cache->paths[cache->count];
            rp->length = depth + 1;
            rp->entity_count = depth + 2;
            memcpy(rp->entities, path_entities, (depth + 2) * sizeof(int));
            memcpy(rp->relations, path_relations, (depth + 1) * sizeof(int));
            cache->count++;
        }

        if (depth + 1 < max_len) {
            visited[next] = 1;
            depth++;

            StackFrame* next_f = &frame[depth];
            next_f->entity_id = next;
            next_f->neighbor_idx = 0;

            int nb_buf[256];
            int nb_cnt = adjacency_list_get_out_neighbors(
                reasoner->adjacency_list, next, nb_buf, NULL, 256);
            next_f->num_neighbors = nb_cnt;
            next_f->neighbors = (int*)safe_malloc((nb_cnt + 1) * sizeof(int));
            next_f->rels = (int*)safe_malloc((nb_cnt + 1) * sizeof(int));
            if (next_f->neighbors && nb_cnt > 0) {
                for (int i = 0; i < nb_cnt; i++) {
                    next_f->neighbors[i] = nb_buf[i];
                    /* ZS-003修复: 从当前节点出边ID中获取实际关系类型 */
                    const ALNode* cn = adjacency_list_get_node(
                        reasoner->adjacency_list, next);
                    if (cn && cn->out_edge_ids && i < (int)cn->out_degree) {
                        next_f->rels[i] = cn->out_edge_ids[i];
                    } else {
                        next_f->rels[i] = i + 1;
                    }
                }
            }
        }
    }

    for (int i = 0; i <= depth && i <= max_len; i++) {
        safe_free((void**)&frame[i].neighbors);
        safe_free((void**)&frame[i].rels);
    }

    safe_free((void**)&visited);
    safe_free((void**)&path_entities);
    safe_free((void**)&path_relations);
    safe_free((void**)&frame);

    return cache->count;
}

int graph_reasoner_mine_rules(GraphReasoner* reasoner,
                               int max_body_length,
                               float min_confidence, float min_support,
                               RuleSet* rules) {
    if (!reasoner || !rules) return -1;
    if (!reasoner->adjacency_list || max_body_length <= 0) return 0;

    int max_len = max_body_length;
    if (max_len > GR_MAX_PATH_LENGTH) max_len = GR_MAX_PATH_LENGTH;

    rules->count = 0;

    for (int src = 0; src < reasoner->entity_count && rules->count < GR_MAX_RULES; src++) {
        RelationPathCache cache;
        cache.count = 0;
        gr_enum_paths_for_rule(reasoner, src, max_len, &cache);

        for (int p = 0; p < cache.count && rules->count < GR_MAX_RULES; p++) {
            RelationPath* rp = &cache.paths[p];
            if (rp->length < 2) continue;

            int last_entity = rp->entities[rp->entity_count - 1];
            int head_entity = rp->entities[0];

            if (reasoner->adjacency_list) {
                int has_direct = adjacency_list_has_edge(
                    reasoner->adjacency_list, head_entity, last_entity);
                if (!has_direct) continue;
            }

            int cfc_head = reasoner->entity_to_cfc_id[head_entity];
            int cfc_last = reasoner->entity_to_cfc_id[last_entity];

            if (cfc_head < 0 || cfc_last < 0) continue;

            int rel_found = -1;
            for (int r = 0; r < reasoner->relation_count; r++) {
                int cfc_r = reasoner->relation_to_cfc_id[r];
                if (cfc_r < 0) continue;
                float score = cfc_embed_score_triple(reasoner->embed_state,
                                                      cfc_head, cfc_r, cfc_last);
                if (score < reasoner->config.margin) {
                    rel_found = r;
                    break;
                }
            }
            if (rel_found < 0) continue;

            MinedRule* rule = &rules->rules[rules->count];
            rule->body_length = rp->length;
            for (int i = 0; i < rp->length; i++) {
                rule->body_relations[i] = rp->relations[i];
            }
            rule->head_relation = rel_found;
            rule->confidence = 1.0f;
            rule->support = 1.0f;
            rule->head_coverage = 1.0f;
            rules->count++;
        }
    }

    for (int i = 0; i < rules->count; i++) {
        MinedRule* rule = &rules->rules[i];
        int match_cnt = 0;
        int total_heads = 0;

        for (int src = 0; src < reasoner->entity_count; src++) {
            int cfc_src = reasoner->entity_to_cfc_id[src];
            if (cfc_src < 0) continue;

            int cur = src;
            int matched = 1;

            for (int b = 0; b < rule->body_length; b++) {
                if (reasoner->adjacency_list) {
                    int nb_buf[256];
                    int nb_cnt = adjacency_list_get_out_neighbors(
                        reasoner->adjacency_list, cur, nb_buf, NULL, 256);
                    int found_nb = 0;
                    for (int n = 0; n < nb_cnt; n++) {
                        if (nb_buf[n] >= 0) {
                            cur = nb_buf[n];
                            found_nb = 1;
                            break;
                        }
                    }
                    if (!found_nb) { matched = 0; break; }
                } else {
                    matched = 0;
                    break;
                }
            }

            if (matched) {
                int cfc_cur = reasoner->entity_to_cfc_id[cur];
                if (cfc_cur >= 0) {
                    float score = cfc_embed_score_triple(reasoner->embed_state,
                                                          cfc_src,
                                                          reasoner->relation_to_cfc_id[rule->head_relation],
                                                          cfc_cur);
                    if (score < reasoner->config.margin) {
                        match_cnt++;
                    }
                }
                total_heads++;
            }
        }

        if (total_heads > 0) {
            rule->confidence = (float)match_cnt / (float)total_heads;
            rule->head_coverage = (reasoner->entity_count > 0)
                ? (float)total_heads / (float)reasoner->entity_count : 0.0f;
        }
        rule->support = (float)match_cnt;
    }

    int pruned = 0;
    for (int i = 0; i < rules->count; i++) {
        if (rules->rules[i].confidence >= min_confidence &&
            rules->rules[i].support >= min_support) {
            if (pruned != i) {
                rules->rules[pruned] = rules->rules[i];
            }
            pruned++;
        }
    }
    rules->count = pruned;

    return rules->count;
}

int graph_reasoner_apply_rule(GraphReasoner* reasoner,
                               const MinedRule* rule, int head_entity,
                               int* results, float* scores, int max_results) {
    if (!reasoner || !rule || !results || !scores) return -1;

    int result_count = 0;

    for (int step = 0; step < rule->body_length; step++) {
        if (result_count >= max_results) break;

        if (reasoner->adjacency_list) {
            int nb_buf[256];
            int nb_cnt = (step == 0)
                ? adjacency_list_get_out_neighbors(
                    reasoner->adjacency_list, head_entity, nb_buf, NULL, 256)
                : 0;

            for (int n = 0; n < nb_cnt && result_count < max_results; n++) {
                int cand = nb_buf[n];
                if (cand < 0) continue;

                int cfc_head = reasoner->entity_to_cfc_id[head_entity];
                int cfc_cand = reasoner->entity_to_cfc_id[cand];
                if (cfc_head < 0 || cfc_cand < 0) continue;

                int cfc_rel = reasoner->relation_to_cfc_id[rule->head_relation];
                if (cfc_rel < 0) continue;

                float score = cfc_embed_score_triple(reasoner->embed_state,
                                                      cfc_head, cfc_rel, cfc_cand);
                if (score < reasoner->config.margin) {
                    results[result_count] = cand;
                    scores[result_count] = 1.0f / (1.0f + score);
                    result_count++;
                }
            }
        }
    }

    return result_count;
}

/* ============ 一致性检查实现 ============ */

int graph_reasoner_check_consistency(GraphReasoner* reasoner,
                                      ConsistencyReport* report) {
    if (!reasoner || !report) return -1;

    report->count = 0;
    float total_health = 0.0f;
    int health_checks = 0;

    if (reasoner->property_graph) {
        int max_id = property_graph_get_node_capacity(reasoner->property_graph);

        for (int i = 0; i < max_id && report->count < GR_MAX_CONSISTENCY_ISSUES; i++) {
            const PGNode* node = property_graph_get_node(reasoner->property_graph, i);
            if (!node) continue;

            int cfc_id = reasoner->entity_to_cfc_id[i];
            if (cfc_id < 0) {
                ConsistencyIssue* issue = &report->issues[report->count];
                issue->issue_type = GR_ISSUE_ISOLATED_NODE;
                issue->entity_id = i;
                issue->relation_id = -1;
                issue->related_entity_id = -1;
                snprintf(issue->description, sizeof(issue->description),
                         "实体[%d]未嵌入到推理引擎", i);
                issue->severity = 0.3f;
                issue->repair_score = 0.7f;
                report->count++;
                total_health += 0.3f;
                health_checks++;
                continue;
            }

            float emb[CFC_EMBED_MAX_DIM];
            int dim = reasoner->config.embedding_dim;
            cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_id, emb, dim);
            float energy = gr_compute_node_energy(emb, dim);

            if (energy > GR_ENERGY_NORMALIZE) {
                ConsistencyIssue* issue = &report->issues[report->count];
                issue->issue_type = GR_ISSUE_UNUSUAL_ENERGY;
                issue->entity_id = i;
                issue->relation_id = -1;
                issue->related_entity_id = -1;
                snprintf(issue->description, sizeof(issue->description),
                         "实体[%d]嵌入能量异常: %.4f", i, energy);
                issue->severity = energy / (GR_ENERGY_NORMALIZE * 2.0f);
                if (issue->severity > 1.0f) issue->severity = 1.0f;
                issue->repair_score = 0.5f;
                report->count++;
                total_health += 1.0f - issue->severity;
                health_checks++;
            } else {
                total_health += 1.0f;
                health_checks++;
            }
        }
    }

    if (reasoner->adjacency_list) {
        int max_al = adjacency_list_get_node_capacity(reasoner->adjacency_list);

        for (int i = 0; i < max_al && report->count < GR_MAX_CONSISTENCY_ISSUES; i++) {
            const ALNode* node = adjacency_list_get_node(reasoner->adjacency_list, i);
            if (!node) continue;
            if (node->out_degree == 0 && node->in_degree == 0) {
                ConsistencyIssue* issue = &report->issues[report->count];
                issue->issue_type = GR_ISSUE_ISOLATED_NODE;
                issue->entity_id = i;
                issue->relation_id = -1;
                issue->related_entity_id = -1;
                snprintf(issue->description, sizeof(issue->description),
                         "邻接表节点[%d]为孤立节点", i);
                issue->severity = 0.4f;
                issue->repair_score = 0.3f;
                report->count++;
                total_health += 0.6f;
                health_checks++;
            }

            for (size_t j = 0; j < node->out_degree && report->count < GR_MAX_CONSISTENCY_ISSUES; j++) {
                int nb_id = node->out_neighbors[j];
                if (nb_id >= max_al) continue;
                const ALNode* nb_node = adjacency_list_get_node(reasoner->adjacency_list, nb_id);
                if (!nb_node) continue;

                int cfc_i = reasoner->entity_to_cfc_id[i];
                int cfc_nb = reasoner->entity_to_cfc_id[nb_id];
                if (cfc_i < 0 || cfc_nb < 0) continue;

                int has_back = 0;
                for (size_t k = 0; k < nb_node->in_degree; k++) {
                    if (nb_node->in_neighbors[k] == i) {
                        has_back = 1;
                        break;
                    }
                }

                if (!has_back && node->out_degree > 0) {
                    ConsistencyIssue* issue = &report->issues[report->count];
                    issue->issue_type = GR_ISSUE_CYCLE;
                    issue->entity_id = i;
                    issue->relation_id = (int)j;
                    issue->related_entity_id = nb_id;
                    snprintf(issue->description, sizeof(issue->description),
                             "边[%d->%d]缺少反向边（无向图不一致）", i, nb_id);
                    issue->severity = 0.2f;
                    issue->repair_score = 0.9f;
                    report->count++;
                    total_health += 0.8f;
                    health_checks++;
                }
            }
        }
    }

    if (report->count == 0) {
        report->overall_health = 1.0f;
    } else if (health_checks > 0) {
        report->overall_health = total_health / (float)health_checks;
    } else {
        report->overall_health = 1.0f;
    }

    return report->count;
}

int graph_reasoner_repair_issue(GraphReasoner* reasoner,
                                 const ConsistencyIssue* issue) {
    if (!reasoner || !issue) return -1;

    switch (issue->issue_type) {
        case GR_ISSUE_ISOLATED_NODE: {
            if (reasoner->property_graph && issue->entity_id >= 0) {
                int cfc_id = reasoner->entity_to_cfc_id[issue->entity_id];
                if (cfc_id < 0) {
                    const PGNode* node = property_graph_get_node(
                        reasoner->property_graph, issue->entity_id);
                    if (node) {
                        const char* label = node->label ? node->label : "unknown";
                        int new_cfc = cfc_embed_add_entity(reasoner->embed_state, label);
                        if (new_cfc >= 0 && new_cfc < GR_MAX_ENTITIES) {
                            reasoner->entity_to_cfc_id[issue->entity_id] = new_cfc;
                            reasoner->cfc_to_entity_id[new_cfc] = issue->entity_id;
                            reasoner->entity_count++;
                            return 0;
                        }
                    }
                }
            }
            return 1;
        }

        case GR_ISSUE_UNUSUAL_ENERGY: {
            if (issue->entity_id >= 0) {
                int cfc_id = reasoner->entity_to_cfc_id[issue->entity_id];
                if (cfc_id >= 0) {
                    int dim = reasoner->config.embedding_dim;
                    float* emb = (float*)safe_malloc(dim * sizeof(float));
                    if (emb) {
                        cfc_embed_get_entity_embedding(reasoner->embed_state,
                                                        cfc_id, emb, dim);
                        for (int d = 0; d < dim; d++) {
                            if (emb[d] > 1.0f) emb[d] = 0.5f;
                            if (emb[d] < -1.0f) emb[d] = -0.5f;
                        }
                    }
                    safe_free((void**)&emb);
                    return 0;
                }
            }
            return 1;
        }

        case GR_ISSUE_CYCLE: {
            if (reasoner->adjacency_list && issue->entity_id >= 0
                && issue->related_entity_id >= 0) {
                adjacency_list_add_edge(reasoner->adjacency_list,
                                         issue->related_entity_id,
                                         issue->entity_id, NULL, 1.0f);
                return 0;
            }
            return 1;
        }

        default:
            return 1;
    }
}

/* ============ 嵌入操作实现 ============ */

int graph_reasoner_get_entity_embedding(const GraphReasoner* reasoner,
                                         int entity_id,
                                         float* embedding, int max_dim) {
    if (!reasoner || !embedding) return -1;
    int cfc_id = reasoner->entity_to_cfc_id[entity_id];
    if (cfc_id < 0) return -1;
    return cfc_embed_get_entity_embedding(reasoner->embed_state,
                                           cfc_id, embedding, max_dim);
}

int graph_reasoner_get_relation_embedding(const GraphReasoner* reasoner,
                                           int relation_id,
                                           float* embedding, int max_dim) {
    if (!reasoner || !embedding) return -1;
    int cfc_id = reasoner->relation_to_cfc_id[relation_id];
    if (cfc_id < 0) return -1;
    return cfc_embed_get_relation_embedding(reasoner->embed_state,
                                             cfc_id, embedding, max_dim);
}

float graph_reasoner_entity_similarity(const GraphReasoner* reasoner,
                                        int entity_a, int entity_b) {
    if (!reasoner) return -2.0f;

    int cfc_a = reasoner->entity_to_cfc_id[entity_a];
    int cfc_b = reasoner->entity_to_cfc_id[entity_b];
    if (cfc_a < 0 || cfc_b < 0) return -2.0f;

    int dim = reasoner->config.embedding_dim;
    float emb_a[CFC_EMBED_MAX_DIM];
    float emb_b[CFC_EMBED_MAX_DIM];

    if (cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_a,
                                        emb_a, dim) <= 0) return -2.0f;
    if (cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_b,
                                        emb_b, dim) <= 0) return -2.0f;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int d = 0; d < dim; d++) {
        dot += emb_a[d] * emb_b[d];
        norm_a += emb_a[d] * emb_a[d];
        norm_b += emb_b[d] * emb_b[d];
    }

    if (norm_a < GR_EPSILON || norm_b < GR_EPSILON) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

float graph_reasoner_get_liquid_energy(const GraphReasoner* reasoner,
                                        int entity_id, int relation_id) {
    if (!reasoner) return -1.0f;

    int dim = reasoner->config.embedding_dim;
    float emb[CFC_EMBED_MAX_DIM];

    if (entity_id >= 0) {
        int cfc_id = reasoner->entity_to_cfc_id[entity_id];
        if (cfc_id < 0) return -1.0f;
        if (cfc_embed_get_entity_embedding(reasoner->embed_state,
                                            cfc_id, emb, dim) <= 0) return -1.0f;
        float energy = 0.0f;
        for (int d = 0; d < dim; d++) {
            energy += emb[d] * emb[d];
            float gate = 1.0f / (1.0f + expf(-emb[d]));
            float act = tanhf(0.8f * emb[d]);
            float dh = -emb[d] / (reasoner->config.cfc_tau + GR_EPSILON)
                       + gate * act;
            energy += fabsf(dh);
        }
        return energy / (float)(dim * 2);
    }

    if (relation_id >= 0) {
        int cfc_id = reasoner->relation_to_cfc_id[relation_id];
        if (cfc_id < 0) return -1.0f;
        if (cfc_embed_get_relation_embedding(reasoner->embed_state,
                                              cfc_id, emb, dim) <= 0) return -1.0f;
        float energy = 0.0f;
        for (int d = 0; d < dim; d++) {
            energy += emb[d] * emb[d];
        }
        return sqrtf(energy / (float)dim);
    }

    return -1.0f;
}

/* ============ 查询增强实现 ============ */

int graph_reasoner_enhance_query(GraphReasoner* reasoner,
                                  void* query_result, int max_results) {
    if (!reasoner || !query_result) return -1;
    if (!reasoner->embed_state) return -1;
    if (max_results <= 0) return 0;

    int dim = reasoner->config.embedding_dim;
    PathResultSet* result_set = (PathResultSet*)query_result;
    int enhanced_count = 0;

    if (result_set->count > 0) {
        for (int i = 0; i < result_set->count && i < GR_MAX_PATHS; i++) {
            ReasonPath* path = &result_set->paths[i];
            if (path->length <= 0) continue;

            float liquid_energy = 0.0f;
            for (int n = 0; n < path->length && n < GR_MAX_PATH_LENGTH; n++) {
                int entity_id = path->nodes[n].entity_id;
                if (entity_id >= 0 && entity_id < reasoner->entity_count) {
                    int cfc_id = reasoner->entity_to_cfc_id[entity_id];
                    if (cfc_id >= 0 && cfc_id < GR_MAX_ENTITIES) {
                        float emb[CFC_EMBED_MAX_DIM];
                        int g = cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_id, emb, dim);
                        if (g > 0) {
                            float path_energy = 0.0f;
                            for (int d = 0; d < dim && d < CFC_EMBED_MAX_DIM; d++)
                                path_energy += emb[d] * emb[d];
                            liquid_energy += sqrtf(path_energy / ((float)dim + 1e-6f));
                        }
                    }
                }
            }

            path->liquid_energy = (path->length > 0) ? liquid_energy / (float)path->length : 0.0f;
            enhanced_count++;
        }
        return enhanced_count;
    }

    for (int i = 0; i < reasoner->entity_count && result_set->count < max_results
         && result_set->count < GR_MAX_PATHS; i++) {
        int cfc_id = reasoner->entity_to_cfc_id[i];
        if (cfc_id < 0) continue;

        ReasonPath* path = &result_set->paths[result_set->count];
        memset(path, 0, sizeof(ReasonPath));
        path->nodes[0].entity_id = i;
        path->nodes[0].weight = 1.0f;
        path->length = 1;

        float emb[CFC_EMBED_MAX_DIM];
        int g = cfc_embed_get_entity_embedding(reasoner->embed_state, cfc_id, emb, dim);
        if (g > 0) {
            float energy = 0.0f;
            for (int d = 0; d < dim && d < CFC_EMBED_MAX_DIM; d++)
                energy += emb[d] * emb[d];
            path->liquid_energy = sqrtf(energy / ((float)dim + 1e-6f));
            path->nodes[0].state_energy = path->liquid_energy;
            path->total_score = path->liquid_energy;
            result_set->count++;
            enhanced_count++;
        }
    }

    return enhanced_count;
}

/* ============ 名称获取实现 ============ */

const char* graph_reasoner_get_entity_name(const GraphReasoner* reasoner,
                                            int entity_id) {
    if (!reasoner) return NULL;

    if (reasoner->property_graph) {
        const PGNode* node = property_graph_get_node(
            reasoner->property_graph, entity_id);
        if (node && node->label) return node->label;
    }

    if (reasoner->adjacency_list) {
        const ALNode* node = adjacency_list_get_node(
            reasoner->adjacency_list, entity_id);
        if (node && node->label) return node->label;
    }

    if (reasoner->rdf_store) {
        int cfc_id = reasoner->entity_to_cfc_id[entity_id];
        if (cfc_id >= 0) {
            const RDFNode* rdf_node = rdf_triple_store_get_node_by_id(
                reasoner->rdf_store, entity_id);
            if (rdf_node && rdf_node->value) return rdf_node->value;
        }
    }

    return NULL;
}

const char* graph_reasoner_get_relation_name(const GraphReasoner* reasoner,
                                              int relation_id,
                                              char* out_buf, size_t buf_size) {
    if (!reasoner || !out_buf || buf_size == 0) return NULL;
    if (relation_id < 0 || relation_id >= reasoner->relation_count) return NULL;

    int cfc_rel_id = reasoner->relation_to_cfc_id[relation_id];
    if (cfc_rel_id >= 0 && cfc_rel_id < GR_MAX_RELATIONS) {
        float rel_emb[CFC_EMBED_MAX_DIM];
        int dim = reasoner->config.embedding_dim;
        int g = cfc_embed_get_relation_embedding(reasoner->embed_state, cfc_rel_id, rel_emb, dim);
        if (g > 0) {
            float magnitude = 0.0f;
            for (int d = 0; d < dim && d < CFC_EMBED_MAX_DIM; d++)
                magnitude += rel_emb[d] * rel_emb[d];
            magnitude = sqrtf(magnitude / ((float)dim + 1e-6f));
            snprintf(out_buf, buf_size, "rel_%d_mag%.3f", relation_id, (double)magnitude);
            return out_buf;
        }
    }

    if (reasoner->property_graph) {
        const PGNode* node = property_graph_get_node(reasoner->property_graph, relation_id);
        if (node && node->label) return node->label;
    }
    if (reasoner->rdf_store) {
        const RDFNode* node = rdf_triple_store_get_node_by_id(reasoner->rdf_store, relation_id);
        if (node && node->value) return node->value;
    }

    return NULL;
}

int graph_reasoner_entity_count(const GraphReasoner* reasoner) {
    if (!reasoner) return 0;
    return reasoner->entity_count;
}

int graph_reasoner_relation_count(const GraphReasoner* reasoner) {
    if (!reasoner) return 0;
    return reasoner->relation_count;
}

/* ============ 保存/加载实现 ============ */

int graph_reasoner_save(const GraphReasoner* reasoner, const char* filepath) {
    if (!reasoner || !filepath) return -1;
    return cfc_embed_save(reasoner->embed_state, filepath);
}

GraphReasoner* graph_reasoner_load(const char* filepath,
                                    PropertyGraph* property_graph,
                                    AdjacencyList* adjacency_list,
                                    RDFTripleStore* rdf_store) {
    if (!filepath) return NULL;

    CfCEmbedState* loaded = cfc_embed_load(filepath);
    if (!loaded) return NULL;

    GraphReasonConfig cfg = graph_reasoner_default_config();
    GraphReasoner* reasoner = graph_reasoner_create(&cfg,
                                                     property_graph,
                                                     adjacency_list,
                                                     rdf_store);
    if (!reasoner) {
        cfc_embed_destroy(loaded);
        return NULL;
    }

    if (reasoner->embed_owns && reasoner->embed_state) {
        cfc_embed_destroy(reasoner->embed_state);
    }
    reasoner->embed_state = loaded;
    reasoner->embed_owns = 1;
    reasoner->is_trained = 1;

    return reasoner;
}