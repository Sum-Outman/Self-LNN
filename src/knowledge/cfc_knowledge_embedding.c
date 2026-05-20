/**
 * @file cfc_knowledge_embedding.c
 * @brief CfC知识图谱嵌入引擎完整实现
 *
 * 实现三种嵌入机制：
 * 1. 连续空间嵌入 — 用CfC ODE学习实体/关系嵌入向量
 * 2. 四元数旋转嵌入 — 用四元数旋转建模对称/反对称/组合关系
 * 3. 液态图传播 — 用CfC ODE在知识图谱上信息传播
 */

#include "selflnn/knowledge/cfc_knowledge_embedding.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/cfc_network.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define CFC_EMBED_EPSILON 1e-8f

static unsigned int embed_seed = 12345;

/* B-022修复: 全局RNG互斥锁，保护embed_seed的多线程安全访问 */
static MutexHandle g_embed_rng_mutex = NULL;
static int g_embed_rng_mutex_inited = 0;

static void embed_rng_ensure_mutex(void) {
    if (!g_embed_rng_mutex_inited) {
        g_embed_rng_mutex = mutex_create();
        g_embed_rng_mutex_inited = 1;
    }
}

static float embed_rand_float(void) {
    embed_rng_ensure_mutex();
    mutex_lock(g_embed_rng_mutex);
    embed_seed = embed_seed * 1103515245 + 12345;
    float result = (float)((embed_seed >> 16) & 0x7FFF) / 32768.0f;
    mutex_unlock(g_embed_rng_mutex);
    return result;
}

static int embed_rand_int(int min, int max) {
    return min + (int)(embed_rand_float() * (max - min + 1));
}

static float embed_rand_normal(void) {
    float u1 = embed_rand_float();
    float u2 = embed_rand_float();
    if (u1 < 1e-8f) u1 = 1e-8f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

typedef struct {
    char name[256];
    int id;
} EntityEntry;

typedef struct {
    char name[256];
    int id;
} RelationEntry;

typedef struct {
    int head_id;
    int rel_id;
    int tail_id;
    float weight;
} TripleEntry;

struct CfCEmbedState {
    CfCEmbedConfig config;
    int is_initialized;

    /* 实体和关系表 */
    EntityEntry* entities;
    int entity_count;
    int entity_capacity;

    RelationEntry* relations;
    int relation_count;
    int relation_capacity;

    /* 三元组 */
    TripleEntry* triples;
    int triple_count;
    int triple_capacity;

    /* 嵌入向量 */
    float* entity_embeddings;
    float* relation_embeddings;

    /* Adam优化器状态 */
    float* entity_momentum;
    float* entity_velocity;
    float* relation_momentum;
    float* relation_velocity;

    /* 四元数嵌入（用于旋转） */
    float* entity_quaternion_r;
    float* entity_quaternion_i;
    float* entity_quaternion_j;
    float* entity_quaternion_k;

    /* CfC ODE传播状态 */
    float* cfc_state_buffer;
    float* cfc_graph_state;

    /* 核心LNN网络集成 */
    CfCNetwork* lnn_network;

    /* 训练状态 */
    int current_epoch;
    float loss_history[100];
    int loss_count;
};

/* CfC ODE步进 — 集成核心LNN网络
 * 当LNN网络已连接时，通过核心LNN的连续动态系统进行状态演化；
 * 未连接时使用数学等价的自包含CfC封闭形式解（σ(-f)·g + σ(f)·h）
 */
static void embed_cfc_step(CfCEmbedState* state, float* h, int dim, float tau, float dt) {
    CfCNetwork* net = state->lnn_network;
    if (net) {
        float* cell_buf = (float*)safe_malloc((size_t)dim * sizeof(float));
        float* temp_out = (float*)safe_malloc((size_t)dim * sizeof(float));
        if (cell_buf && temp_out) {
            int ret = cfc_forward(net, h, h, cell_buf, temp_out);
            if (ret == 0) {
                memcpy(h, temp_out, (size_t)dim * sizeof(float));
            }
        }
        safe_free((void**)&temp_out);
        safe_free((void**)&cell_buf);
        return;
    }

    /* 无LNN连接时的自包含CfC */
    for (int i = 0; i < dim; i++) {
        float f_val = h[i] * 0.8f;
        float g_val = tanhf(h[i]);
        float sigmoid_f = 1.0f / (1.0f + expf(-f_val / tau));
        float cfc_h = sigmoid_f * h[i] + (1.0f - sigmoid_f) * g_val;
        float dh = (cfc_h - h[i]) / (tau + CFC_EMBED_EPSILON) * dt;
        h[i] += dh;
    }
}

/* 四元数旋转：h' = h ⊗ r （四元数乘法） */
static void quaternion_rotate(const float* h_r, const float* h_i,
                               const float* h_j, const float* h_k,
                               const float* r_r, const float* r_i,
                               const float* r_j, const float* r_k,
                               float* out_r, float* out_i,
                               float* out_j, float* out_k,
                               int dim) {
    for (int d = 0; d < dim; d++) {
        float hr = h_r[d], hi = h_i[d], hj = h_j[d], hk = h_k[d];
        float rr = r_r[d], ri = r_i[d], rj = r_j[d], rk = r_k[d];
        out_r[d] = hr * rr - hi * ri - hj * rj - hk * rk;
        out_i[d] = hr * ri + hi * rr + hj * rk - hk * rj;
        out_j[d] = hr * rj - hi * rk + hj * rr + hk * ri;
        out_k[d] = hr * rk + hi * rj - hj * ri + hk * rr;
    }
}

CfCEmbedState* cfc_embed_create(const CfCEmbedConfig* config) {
    if (!config) return NULL;

    CfCEmbedState* state = (CfCEmbedState*)
        safe_calloc(1, sizeof(CfCEmbedState));
    if (!state) return NULL;

    state->config = *config;
    int dim = config->embedding_dim > 0 ? config->embedding_dim : CFC_EMBED_DEFAULT_DIM;
    if (dim > CFC_EMBED_MAX_DIM) dim = CFC_EMBED_MAX_DIM;
    state->config.embedding_dim = dim;

    state->entity_capacity = 1024;
    state->entities = (EntityEntry*)
        safe_calloc(state->entity_capacity, sizeof(EntityEntry));
    state->relation_capacity = 256;
    state->relations = (RelationEntry*)
        safe_calloc(state->relation_capacity, sizeof(RelationEntry));
    state->triple_capacity = 4096;
    state->triples = (TripleEntry*)
        safe_calloc(state->triple_capacity, sizeof(TripleEntry));

    state->entity_embeddings = (float*)
        safe_calloc(CFC_EMBED_MAX_ENTITIES * dim, sizeof(float));

    if (config->embed_type == CFC_EMBED_QUATERNION ||
        config->embed_type == CFC_EMBED_HYBRID) {
        state->entity_quaternion_r = (float*)
            safe_calloc(CFC_EMBED_MAX_ENTITIES * dim, sizeof(float));
        state->entity_quaternion_i = (float*)
            safe_calloc(CFC_EMBED_MAX_ENTITIES * dim, sizeof(float));
        state->entity_quaternion_j = (float*)
            safe_calloc(CFC_EMBED_MAX_ENTITIES * dim, sizeof(float));
        state->entity_quaternion_k = (float*)
            safe_calloc(CFC_EMBED_MAX_ENTITIES * dim, sizeof(float));
    }

    state->relation_embeddings = (float*)
        safe_calloc(CFC_EMBED_MAX_RELATIONS * dim, sizeof(float));

    state->cfc_state_buffer = (float*)safe_calloc(dim, sizeof(float));
    state->cfc_graph_state = (float*)safe_calloc(dim, sizeof(float));

    /* Adam优化器状态初始化 */
    state->entity_momentum = (float*)safe_calloc(CFC_EMBED_MAX_ENTITIES * dim, sizeof(float));
    state->entity_velocity = (float*)safe_calloc(CFC_EMBED_MAX_ENTITIES * dim, sizeof(float));
    state->relation_momentum = (float*)safe_calloc(CFC_EMBED_MAX_RELATIONS * dim, sizeof(float));
    state->relation_velocity = (float*)safe_calloc(CFC_EMBED_MAX_RELATIONS * dim, sizeof(float));

    if (!state->entity_embeddings || !state->relation_embeddings) {
        cfc_embed_destroy(state);
        return NULL;
    }

    /* 随机初始化嵌入 */
    for (int i = 0; i < CFC_EMBED_MAX_ENTITIES * dim; i++)
        state->entity_embeddings[i] = embed_rand_normal() * 0.1f;

    for (int i = 0; i < CFC_EMBED_MAX_RELATIONS * dim; i++)
        state->relation_embeddings[i] = embed_rand_normal() * 0.1f;

    if (state->entity_quaternion_r) {
        int total = CFC_EMBED_MAX_ENTITIES * dim;
        for (int i = 0; i < total; i++) {
            state->entity_quaternion_r[i] = embed_rand_normal() * 0.1f;
            state->entity_quaternion_i[i] = embed_rand_normal() * 0.1f;
            state->entity_quaternion_j[i] = embed_rand_normal() * 0.1f;
            state->entity_quaternion_k[i] = embed_rand_normal() * 0.1f;
        }
    }

    state->is_initialized = 1;
    return state;
}

void cfc_embed_set_lnn_network(CfCEmbedState* state, void* lnn_network) {
    if (!state) return;
    state->lnn_network = (CfCNetwork*)lnn_network;
}

void* cfc_embed_get_lnn_network(const CfCEmbedState* state) {
    if (!state) return NULL;
    return state->lnn_network;
}

void cfc_embed_destroy(CfCEmbedState* state) {
    if (!state) return;
    safe_free((void**)&state->entities);
    safe_free((void**)&state->relations);
    safe_free((void**)&state->triples);
    safe_free((void**)&state->entity_embeddings);
    safe_free((void**)&state->relation_embeddings);
    safe_free((void**)&state->entity_quaternion_r);
    safe_free((void**)&state->entity_quaternion_i);
    safe_free((void**)&state->entity_quaternion_j);
    safe_free((void**)&state->entity_quaternion_k);
    safe_free((void**)&state->cfc_state_buffer);
    safe_free((void**)&state->cfc_graph_state);
    safe_free((void**)&state->entity_momentum);
    safe_free((void**)&state->entity_velocity);
    safe_free((void**)&state->relation_momentum);
    safe_free((void**)&state->relation_velocity);
    safe_free((void**)&state);
}

int cfc_embed_add_entity(CfCEmbedState* state, const char* entity_name) {
    if (!state || !entity_name) return -1;

    if (state->entity_count >= state->entity_capacity) {
        int new_cap = state->entity_capacity * 2;
        if (new_cap > CFC_EMBED_MAX_ENTITIES) new_cap = CFC_EMBED_MAX_ENTITIES;
        EntityEntry* new_ents = (EntityEntry*)
            safe_realloc(state->entities, new_cap * sizeof(EntityEntry));
        if (!new_ents) return -1;
        state->entities = new_ents;
        state->entity_capacity = new_cap;
    }

    int id = state->entity_count;
    state->entities[id].id = id;
    strncpy(state->entities[id].name, entity_name,
            sizeof(state->entities[id].name) - 1);
    state->entity_count++;
    return id;
}

int cfc_embed_add_relation(CfCEmbedState* state, const char* relation_name) {
    if (!state || !relation_name) return -1;

    if (state->relation_count >= state->relation_capacity) {
        int new_cap = state->relation_capacity * 2;
        if (new_cap > CFC_EMBED_MAX_RELATIONS) new_cap = CFC_EMBED_MAX_RELATIONS;
        RelationEntry* new_rels = (RelationEntry*)
            safe_realloc(state->relations, new_cap * sizeof(RelationEntry));
        if (!new_rels) return -1;
        state->relations = new_rels;
        state->relation_capacity = new_cap;
    }

    int id = state->relation_count;
    state->relations[id].id = id;
    strncpy(state->relations[id].name, relation_name,
            sizeof(state->relations[id].name) - 1);
    state->relation_count++;
    return id;
}

int cfc_embed_add_triple(CfCEmbedState* state, int head_id, int rel_id, int tail_id) {
    if (!state) return -1;
    if (head_id < 0 || rel_id < 0 || tail_id < 0) return -1;

    if (state->triple_count >= state->triple_capacity) {
        int new_cap = state->triple_capacity * 2;
        TripleEntry* new_tri = (TripleEntry*)
            safe_realloc(state->triples, new_cap * sizeof(TripleEntry));
        if (!new_tri) return -1;
        state->triples = new_tri;
        state->triple_capacity = new_cap;
    }

    int idx = state->triple_count;
    state->triples[idx].head_id = head_id;
    state->triples[idx].rel_id = rel_id;
    state->triples[idx].tail_id = tail_id;
    state->triples[idx].weight = 1.0f;
    state->triple_count++;
    return 0;
}

int cfc_embed_get_entity_id(const CfCEmbedState* state, const char* entity_name) {
    if (!state || !entity_name) return -1;
    for (int i = 0; i < state->entity_count; i++) {
        if (strcmp(state->entities[i].name, entity_name) == 0)
            return i;
    }
    return -1;
}

int cfc_embed_get_relation_id(const CfCEmbedState* state, const char* relation_name) {
    if (!state || !relation_name) return -1;
    for (int i = 0; i < state->relation_count; i++) {
        if (strcmp(state->relations[i].name, relation_name) == 0)
            return i;
    }
    return -1;
}

int cfc_embed_get_entity_embedding(const CfCEmbedState* state, int entity_id,
                                    float* embedding, int max_dim) {
    if (!state || !embedding) return -1;
    if (entity_id < 0 || entity_id >= state->entity_count) return -1;

    int dim = state->config.embedding_dim;
    int copy_dim = dim < max_dim ? dim : max_dim;
    memcpy(embedding, &state->entity_embeddings[entity_id * dim],
           copy_dim * sizeof(float));
    return copy_dim;
}

int cfc_embed_get_relation_embedding(const CfCEmbedState* state, int relation_id,
                                      float* embedding, int max_dim) {
    if (!state || !embedding) return -1;
    if (relation_id < 0 || relation_id >= state->relation_count) return -1;

    int dim = state->config.embedding_dim;
    int copy_dim = dim < max_dim ? dim : max_dim;
    memcpy(embedding, &state->relation_embeddings[relation_id * dim],
           copy_dim * sizeof(float));
    return copy_dim;
}

float cfc_embed_score_triple(CfCEmbedState* state, int head_id, int rel_id, int tail_id) {
    if (!state) return 1e10f;
    if (head_id < 0 || rel_id < 0 || tail_id < 0) return 1e10f;

    int dim = state->config.embedding_dim;
    CfCEmbedType type = state->config.embed_type;

    float score = 0.0f;

    if (type == CFC_EMBED_CONTINUOUS || type == CFC_EMBED_HYBRID) {
        float* h = &state->entity_embeddings[head_id * dim];
        float* r = &state->relation_embeddings[rel_id * dim];
        float* t = &state->entity_embeddings[tail_id * dim];

        for (int d = 0; d < dim; d++) {
            float diff = h[d] + r[d] - t[d];
            score += diff * diff;
        }
        score = sqrtf(score / (float)dim);
    }

    if (type == CFC_EMBED_QUATERNION || type == CFC_EMBED_HYBRID) {
        float out_r[CFC_EMBED_MAX_DIM];
        float out_i[CFC_EMBED_MAX_DIM];
        float out_j[CFC_EMBED_MAX_DIM];
        float out_k[CFC_EMBED_MAX_DIM];

        quaternion_rotate(
            &state->entity_quaternion_r[head_id * dim],
            &state->entity_quaternion_i[head_id * dim],
            &state->entity_quaternion_j[head_id * dim],
            &state->entity_quaternion_k[head_id * dim],
            &state->entity_quaternion_r[rel_id * dim],
            &state->entity_quaternion_i[rel_id * dim],
            &state->entity_quaternion_j[rel_id * dim],
            &state->entity_quaternion_k[rel_id * dim],
            out_r, out_i, out_j, out_k, dim);

        float qscore = 0.0f;
        float* tr = &state->entity_quaternion_r[tail_id * dim];
        float* ti = &state->entity_quaternion_i[tail_id * dim];
        float* tj = &state->entity_quaternion_j[tail_id * dim];
        float* tk = &state->entity_quaternion_k[tail_id * dim];

        for (int d = 0; d < dim; d++) {
            float dr = out_r[d] - tr[d];
            float di = out_i[d] - ti[d];
            float dj = out_j[d] - tj[d];
            float dk = out_k[d] - tk[d];
            qscore += dr * dr + di * di + dj * dj + dk * dk;
        }

        if (type == CFC_EMBED_QUATERNION)
            score = sqrtf(qscore / (float)(dim * 4));
        else
            score = (score + sqrtf(qscore / (float)(dim * 4))) * 0.5f;
    }

    return score;
}

int cfc_embed_predict_tail(CfCEmbedState* state, int head_id, int rel_id,
                            const int* candidates, int num_candidates,
                            float* scores) {
    if (!state || !candidates || !scores) return -1;

    for (int i = 0; i < num_candidates; i++) {
        scores[i] = cfc_embed_score_triple(state, head_id, rel_id, candidates[i]);
    }

    return 0;
}

int cfc_embed_graph_propagate(CfCEmbedState* state, int seed_entity,
                               int steps, float* result, int max_dim) {
    if (!state || !result) return -1;
    if (seed_entity < 0 || seed_entity >= state->entity_count) return -1;

    int dim = state->config.embedding_dim;
    int copy_dim = dim < max_dim ? dim : max_dim;

    memcpy(result, &state->entity_embeddings[seed_entity * dim],
           copy_dim * sizeof(float));

    for (int step = 0; step < steps; step++) {
        memcpy(state->cfc_graph_state, result, copy_dim * sizeof(float));

        for (int t = 0; t < state->triple_count; t++) {
            if (state->triples[t].head_id == seed_entity) {
                int tail = state->triples[t].tail_id;
                float alpha = 1.0f / (float)(state->triple_count + 1);
                for (int d = 0; d < copy_dim; d++) {
                    float tail_emb = state->entity_embeddings[tail * dim + d];
                    state->cfc_graph_state[d] += alpha * (tail_emb - state->cfc_graph_state[d]);
                }
            }
        }

        for (int s = 0; s < state->config.cfc_steps; s++)
            embed_cfc_step(state, state->cfc_graph_state, copy_dim,
                           state->config.cfc_tau, state->config.cfc_dt);

        memcpy(result, state->cfc_graph_state, copy_dim * sizeof(float));
    }

    return 0;
}

int cfc_embed_train(CfCEmbedState* state, int epochs) {
    if (!state) return -1;

    int dim = state->config.embedding_dim;
    float lr = state->config.learning_rate;
    float margin = state->config.margin;
    int batch_size = state->config.batch_size;
    int num_neg = state->config.num_negative_samples;
    float total_loss = 0.0f;
    int batch_count = 0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        total_loss = 0.0f;
        batch_count = 0;

        for (int b = 0; b < state->triple_count; b += batch_size) {
            int end = b + batch_size;
            if (end > state->triple_count) end = state->triple_count;
            float batch_loss = 0.0f;

            for (int t = b; t < end; t++) {
                int h = state->triples[t].head_id;
                int r = state->triples[t].rel_id;
                int tail = state->triples[t].tail_id;

                float pos_score = cfc_embed_score_triple(state, h, r, tail);

                for (int n = 0; n < num_neg; n++) {
                    int neg_tail = embed_rand_int(0, state->entity_count - 1);
                    float neg_score = cfc_embed_score_triple(state, h, r, neg_tail);
                    float hinge = margin + pos_score - neg_score;
                    if (hinge > 0) batch_loss += hinge;

                    /* Adam优化器更新（beta1=0.9, beta2=0.999, epsilon=1e-8） */
                    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
                    float* h_emb = &state->entity_embeddings[h * dim];
                    float* r_emb = &state->relation_embeddings[r * dim];
                    float* pos_emb = &state->entity_embeddings[tail * dim];
                    float* neg_emb = &state->entity_embeddings[neg_tail * dim];

                    float* h_m = &state->entity_momentum[h * dim];
                    float* r_m = &state->relation_momentum[r * dim];
                    float* h_v = &state->entity_velocity[h * dim];
                    float* r_v = &state->relation_velocity[r * dim];
                    float* pos_m = &state->entity_momentum[tail * dim];
                    float* pos_v = &state->entity_velocity[tail * dim];
                    float* neg_m = &state->entity_momentum[neg_tail * dim];
                    float* neg_v = &state->entity_velocity[neg_tail * dim];

                    for (int d = 0; d < dim; d++) {
                        float grad_p = 2.0f * (h_emb[d] + r_emb[d] - pos_emb[d]);
                        float grad_n = -2.0f * (h_emb[d] + r_emb[d] - neg_emb[d]);

                        /* 实体head: 正样本梯度和负样本梯度累加 */
                        float gh = grad_p + grad_n * 0.1f;
                        h_m[d] = beta1 * h_m[d] + (1.0f - beta1) * gh;
                        h_v[d] = beta2 * h_v[d] + (1.0f - beta2) * gh * gh;
                        float h_m_hat = h_m[d] / (1.0f - beta1);
                        float h_v_hat = h_v[d] / (1.0f - beta2);
                        h_emb[d] -= lr * h_m_hat / (sqrtf(h_v_hat) + eps);

                        /* 关系 */
                        float gr = grad_p * 0.5f + grad_n * 0.05f;
                        r_m[d] = beta1 * r_m[d] + (1.0f - beta1) * gr;
                        r_v[d] = beta2 * r_v[d] + (1.0f - beta2) * gr * gr;
                        float r_m_hat = r_m[d] / (1.0f - beta1);
                        float r_v_hat = r_v[d] / (1.0f - beta2);
                        r_emb[d] -= lr * r_m_hat / (sqrtf(r_v_hat) + eps);

                        /* 实体tail (正向) */
                        float gp = -grad_p;
                        pos_m[d] = beta1 * pos_m[d] + (1.0f - beta1) * gp;
                        pos_v[d] = beta2 * pos_v[d] + (1.0f - beta2) * gp * gp;
                        float p_m_hat = pos_m[d] / (1.0f - beta1);
                        float p_v_hat = pos_v[d] / (1.0f - beta2);
                        pos_emb[d] -= lr * p_m_hat / (sqrtf(p_v_hat) + eps);

                        /* 实体tail (负样本) */
                        float gn = -grad_n * 0.1f;
                        neg_m[d] = beta1 * neg_m[d] + (1.0f - beta1) * gn;
                        neg_v[d] = beta2 * neg_v[d] + (1.0f - beta2) * gn * gn;
                        float n_m_hat = neg_m[d] / (1.0f - beta1);
                        float n_v_hat = neg_v[d] / (1.0f - beta2);
                        neg_emb[d] -= lr * n_m_hat / (sqrtf(n_v_hat) + eps);
                    }
                }
            }

            total_loss += batch_loss;
            batch_count++;
        }

        if (state->loss_count < 100)
            state->loss_history[state->loss_count++] = total_loss / (batch_count + 1);
        state->current_epoch = epoch + 1;
    }

    return 0;
}

int cfc_embed_get_loss_history(const CfCEmbedState* state, float* losses, int max_count) {
    if (!state || !losses || max_count <= 0) return 0;
    int cnt = state->loss_count < max_count ? state->loss_count : max_count;
    for (int i = 0; i < cnt; i++) {
        losses[i] = state->loss_history[i];
    }
    return cnt;
}

/* ================================================================
 * 多跳推理：在嵌入空间中进行关系路径行走
 *
 * 给定起始实体e0和关系路径[r1, r2, ..., rN]，在嵌入空间中
 * 沿着关系路径逐步投影，每步：
 *   e_{k+1}  = e_k  + r_k （TransE平移）
 * 或：
 *   e_{k+1} = Rot(r_k, e_k) （四元数旋转，如果启用四元数）
 *
 * 返回最终预测的实体嵌入，以及每一步的中间嵌入。
 * ================================================================ */

int cfc_embed_multi_hop_infer(const CfCEmbedState* state,
                               int start_entity_id,
                               const int* relation_path,
                               int path_length,
                               float* result_embedding,
                               float** intermediate_embeddings,
                               float* path_confidence) {
    if (!state || !state->is_initialized || !result_embedding || !relation_path)
        return -1;
    if (path_length <= 0 || path_length > 10) return -1;
    if (start_entity_id < 0 || start_entity_id >= state->entity_count) return -1;

    int dim = state->config.embedding_dim;
    float* curr_emb = (float*)safe_malloc((size_t)dim * sizeof(float));
    if (!curr_emb) return -1;

    memcpy(curr_emb, &state->entity_embeddings[(size_t)start_entity_id * dim],
           (size_t)dim * sizeof(float));

    if (intermediate_embeddings) {
        for (int h = 0; h <= path_length; h++) {
            intermediate_embeddings[h] = (float*)safe_malloc((size_t)dim * sizeof(float));
        }
        if (intermediate_embeddings[0]) {
            memcpy(intermediate_embeddings[0], curr_emb, (size_t)dim * sizeof(float));
        }
    }

    float total_confidence = 1.0f;

    for (int step = 0; step < path_length; step++) {
        int rel_id = relation_path[step];
        if (rel_id < 0 || rel_id >= state->relation_count) {
            safe_free((void**)&curr_emb);
            return -1;
        }

        const float* r_emb = &state->relation_embeddings[(size_t)rel_id * dim];

        if (state->config.use_quaternion && dim >= 4) {
            /* 四元数旋转: e_out = q(r) * e_in * q(r)^-1 */
            for (int d = 0; d < dim; d += 4) {
                float r_w = r_emb[d], r_x = r_emb[d+1], r_y = r_emb[d+2], r_z = r_emb[d+3];
                float rn = sqrtf(r_w*r_w + r_x*r_x + r_y*r_y + r_z*r_z + 1e-8f);
                r_w /= rn; r_x /= rn; r_y /= rn; r_z /= rn;

                float e_w = curr_emb[d], e_x = curr_emb[d+1], e_y = curr_emb[d+2], e_z = curr_emb[d+3];
                float o_w = r_w*e_w - r_x*e_x - r_y*e_y - r_z*e_z;
                float o_x = r_w*e_x + r_x*e_w + r_y*e_z - r_z*e_y;
                float o_y = r_w*e_y - r_x*e_z + r_y*e_w + r_z*e_x;
                float o_z = r_w*e_z + r_x*e_y - r_y*e_x + r_z*e_w;
                curr_emb[d] = o_w; curr_emb[d+1] = o_x;
                curr_emb[d+2] = o_y; curr_emb[d+3] = o_z;
            }
        } else {
            /* TransE平移: e_out = e_in + r */
            for (int d = 0; d < dim; d++) {
                curr_emb[d] += r_emb[d];
            }
            /* L2归一化保持嵌入在单位球内 */
            float norm = 0.0f;
            for (int d = 0; d < dim; d++) norm += curr_emb[d] * curr_emb[d];
            norm = sqrtf(norm + 1e-8f);
            if (norm > 1e-6f) {
                float inv = 1.0f / norm;
                for (int d = 0; d < dim; d++) curr_emb[d] *= inv;
            }
        }

        if (intermediate_embeddings && intermediate_embeddings[step + 1]) {
            memcpy(intermediate_embeddings[step + 1], curr_emb,
                   (size_t)dim * sizeof(float));
        }

        /* 每步置信度衰减 */
        total_confidence *= 0.85f;
    }

    memcpy(result_embedding, curr_emb, (size_t)dim * sizeof(float));
    if (path_confidence) *path_confidence = total_confidence;

    safe_free((void**)&curr_emb);
    return 0;
}

/* 在知识图谱中执行多跳推理查询并返回top-k匹配实体 */
int cfc_embed_multi_hop_query(const CfCEmbedState* state,
                               int start_entity_id,
                               const int* relation_path,
                               int path_length,
                               int top_k,
                               int* matched_entity_ids,
                               float* match_scores) {
    if (!state || !relation_path || !matched_entity_ids || !match_scores)
        return -1;
    if (top_k <= 0 || top_k > 100) return -1;

    int dim = state->config.embedding_dim;
    float* result_emb = (float*)safe_malloc((size_t)dim * sizeof(float));
    if (!result_emb) return -1;

    float confidence;
    if (cfc_embed_multi_hop_infer(state, start_entity_id, relation_path,
                                   path_length, result_emb, NULL, &confidence) != 0) {
        safe_free((void**)&result_emb);
        return -1;
    }

    /* 计算与所有实体的余弦相似度，选择top-k */
    int* candidates = (int*)safe_malloc((size_t)state->entity_count * sizeof(int));
    float* scores = (float*)safe_malloc((size_t)state->entity_count * sizeof(float));
    if (!candidates || !scores) {
        safe_free((void**)&result_emb);
        safe_free((void**)&candidates);
        safe_free((void**)&scores);
        return -1;
    }

    for (int e = 0; e < state->entity_count; e++) {
        candidates[e] = e;
        const float* e_emb = &state->entity_embeddings[(size_t)e * dim];
        float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            dot += result_emb[d] * e_emb[d];
            n1 += result_emb[d] * result_emb[d];
            n2 += e_emb[d] * e_emb[d];
        }
        scores[e] = dot / (sqrtf(n1 * n2) + 1e-8f);
    }

    /* 简单选择排序 top-k */
    for (int k = 0; k < top_k && k < state->entity_count; k++) {
        int best = k;
        for (int e = k + 1; e < state->entity_count; e++) {
            if (scores[e] > scores[best]) best = e;
        }
        int tmp_i = candidates[k]; candidates[k] = candidates[best]; candidates[best] = tmp_i;
        float tmp_f = scores[k]; scores[k] = scores[best]; scores[best] = tmp_f;
    }

    int result_count = top_k < state->entity_count ? top_k : state->entity_count;
    for (int k = 0; k < result_count; k++) {
        matched_entity_ids[k] = candidates[k];
        match_scores[k] = scores[k] * confidence;
    }

    safe_free((void**)&result_emb);
    safe_free((void**)&candidates);
    safe_free((void**)&scores);
    return result_count;
}

int cfc_embed_save(const CfCEmbedState* state, const char* filepath) {
    if (!state || !filepath) return -1;
    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    fwrite(&state->config, sizeof(CfCEmbedConfig), 1, f);
    fwrite(&state->entity_count, sizeof(int), 1, f);
    fwrite(&state->relation_count, sizeof(int), 1, f);
    fwrite(&state->triple_count, sizeof(int), 1, f);
    fwrite(state->entity_embeddings, sizeof(float),
           CFC_EMBED_MAX_ENTITIES * state->config.embedding_dim, f);
    fwrite(state->relation_embeddings, sizeof(float),
           CFC_EMBED_MAX_RELATIONS * state->config.embedding_dim, f);

    fclose(f);
    return 0;
}

CfCEmbedState* cfc_embed_load(const char* filepath) {
    if (!filepath) return NULL;
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    CfCEmbedConfig config;
    if (fread(&config, sizeof(CfCEmbedConfig), 1, f) != 1) {
        fclose(f); return NULL;
    }

    CfCEmbedState* state = cfc_embed_create(&config);
    if (!state) { fclose(f); return NULL; }

    fread(&state->entity_count, sizeof(int), 1, f);
    fread(&state->relation_count, sizeof(int), 1, f);
    fread(&state->triple_count, sizeof(int), 1, f);
    fread(state->entity_embeddings, sizeof(float),
          CFC_EMBED_MAX_ENTITIES * config.embedding_dim, f);
    fread(state->relation_embeddings, sizeof(float),
          CFC_EMBED_MAX_RELATIONS * config.embedding_dim, f);

    fclose(f);
    return state;
}

CfCEmbedConfig cfc_embed_default_config(void) {
    CfCEmbedConfig config;
    memset(&config, 0, sizeof(CfCEmbedConfig));
    config.num_entities = 0;
    config.num_relations = 0;
    config.embedding_dim = CFC_EMBED_DEFAULT_DIM;
    config.embed_type = CFC_EMBED_CONTINUOUS;
    config.learning_rate = 0.01f;
    config.margin = 1.0f;
    config.num_negative_samples = 5;
    config.batch_size = 128;
    config.max_epochs = 100;
    config.cfc_tau = 2.0f;
    config.cfc_dt = 0.1f;
    config.cfc_steps = 5;
    config.enable_graph_propagation = 1;
    config.graph_prop_steps = 3;
    config.graph_prop_dropout = 0.1f;
    return config;
}
