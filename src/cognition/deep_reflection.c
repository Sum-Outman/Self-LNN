#include "selflnn/cognition/deep_reflection.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/laplace.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define DR_SIGMOID(x) (1.0f / (1.0f + expf(-(x))))
#define DR_TANH(x) (tanhf(x))
#define DR_MAX(a,b) (((a)>(b))?(a):(b))
#define DR_MIN(a,b) (((a)<(b))?(a):(b))
#define DR_CLAMP(v,lo,hi) DR_MIN(DR_MAX((v),(lo)),(hi))
#define DR_ABS(x) (((x)>=0)?(x):(-(x)))

struct DeepReflectionEngine {
    DRConfig config;
    LNN* reflection_net;
    LNN* synthesis_net;
    LNN* conflict_net;
    LNN* epistemic_net;
    LNN* hypothesis_net;
    int owns_reflection;   /* 所有权标记 */
    int owns_synthesis;
    int owns_conflict;
    int owns_epistemic;
    int owns_hypothesis;
    float* layer_embeddings;
    size_t layer_embed_count;
    float* knowledge_base;
    size_t knowledge_size;
    float* conflict_history;
    size_t conflict_count;
    int initialized;
};

static float cosine_sim_dr(const float* a, const float* b, size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

static float euclidean_dist_dr(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

static float bayesian_conflict_score(const float* belief_a, const float* belief_b,
                                      float prior_a, float prior_b, size_t dim) {
    float sim = cosine_sim_dr(belief_a, belief_b, dim);
    float evidence_ratio = (1.0f - sim) / (sim + 1e-10f);
    float posterior_a = (prior_a * (1.0f - sim)) / (prior_a * (1.0f - sim) + (1.0f - prior_a) * sim + 1e-10f);
    float posterior_b = (prior_b * (1.0f - sim)) / (prior_b * (1.0f - sim) + (1.0f - prior_b) * sim + 1e-10f);
    float divergence = DR_ABS(posterior_a - prior_a) + DR_ABS(posterior_b - prior_b);
    return DR_CLAMP(divergence * evidence_ratio, 0.0f, 1.0f);
}

static void weighted_embedding_fusion(const float** embeddings, const float* weights,
                                       size_t count, size_t dim, float* out) {
    memset(out, 0, dim * sizeof(float));
    float total_weight = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float w = (weights) ? weights[i] : 1.0f;
        for (size_t j = 0; j < dim; j++) {
            out[j] += embeddings[i][j] * w;
        }
        total_weight += w;
    }
    if (total_weight > 1e-10f) {
        float inv = 1.0f / total_weight;
        for (size_t j = 0; j < dim; j++) out[j] *= inv;
    }
}

static void generate_perspective_embedding(const float* base, int perspective_idx,
                                            float diversity, size_t dim, float* out) {
    for (size_t j = 0; j < dim; j++) {
        float noise = ((float)(perspective_idx * 7 + j * 31) / 997.0f) - 0.5f;
        float shift = sinf((float)perspective_idx * 1.5f + (float)j * 0.1f) * diversity;
        out[j] = base[j] + noise * diversity * 2.0f + shift;
    }
    float norm = 0.0f;
    for (size_t j = 0; j < dim; j++) norm += out[j] * out[j];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (size_t j = 0; j < dim; j++) out[j] /= norm;
    }
}

static const char* layer_names[] = {
    "描述性分析",
    "分析性分析",
    "批判性分析",
    "因果性分析",
    "认知论分析",
    "元认知分析",
    "变革性重构"
};

static const char* perspective_names[] = {
    "客观事实视角",
    "逻辑推理视角",
    "经验归纳视角",
    "因果推断视角",
    "反事实假设视角",
    "多智能体协商视角",
    "第一性原理视角",
    "系统整体视角"
};

DeepReflectionEngine* dr_engine_create(DRConfig config) {
    DeepReflectionEngine* engine = (DeepReflectionEngine*)safe_calloc(1, sizeof(DeepReflectionEngine));
    if (!engine) return NULL;

    engine->config = config;
    engine->initialized = 1;
    engine->layer_embed_count = 0;
    engine->knowledge_size = 0;
    engine->conflict_count = 0;

    size_t edim = (config.embedding_dim > 0) ? config.embedding_dim : DR_EMBED_DIM;

    engine->owns_reflection = 1;
    engine->owns_synthesis = 1;
    engine->owns_conflict = 1;
    engine->owns_epistemic = 1;
    engine->owns_hypothesis = 1;

    LNNConfig ref_cfg = {0};
    ref_cfg.input_size = (int)edim;
    ref_cfg.hidden_size = 512;
    ref_cfg.output_size = (int)edim;
    ref_cfg.num_layers = 2;
    engine->reflection_net = lnn_create(&ref_cfg);

    LNNConfig syn_cfg = {0};
    syn_cfg.input_size = (int)(edim * 2);
    syn_cfg.hidden_size = 512;
    syn_cfg.output_size = (int)edim;
    syn_cfg.num_layers = 2;
    engine->synthesis_net = lnn_create(&syn_cfg);
    engine->owns_synthesis = 1;

    LNNConfig con_cfg = {0};
    con_cfg.input_size = (int)(edim * 2);
    con_cfg.hidden_size = 256;
    con_cfg.output_size = 1;
    con_cfg.num_layers = 1;
    engine->conflict_net = lnn_create(&con_cfg);
    engine->owns_conflict = 1;

    LNNConfig epi_cfg = {0};
    epi_cfg.input_size = (int)edim;
    epi_cfg.hidden_size = 256;
    epi_cfg.output_size = 3;
    epi_cfg.num_layers = 1;
    engine->epistemic_net = lnn_create(&epi_cfg);
    engine->owns_epistemic = 1;

    LNNConfig hyp_cfg = {0};
    hyp_cfg.input_size = (int)edim;
    hyp_cfg.hidden_size = 512;
    hyp_cfg.output_size = (int)edim;
    hyp_cfg.num_layers = 2;
    engine->hypothesis_net = lnn_create(&hyp_cfg);
    engine->owns_hypothesis = 1;

    engine->layer_embeddings = (float*)safe_calloc(DR_MAX_CHAIN_LEN * edim, sizeof(float));
    engine->knowledge_base = (float*)safe_calloc(1024 * edim, sizeof(float));
    engine->conflict_history = (float*)safe_calloc(DR_MAX_CONTRADICTIONS * 4, sizeof(float));
    
    return engine;
}

void dr_engine_destroy(DeepReflectionEngine* engine) {
    if (!engine) return;
    if (engine->reflection_net && engine->owns_reflection) lnn_free(engine->reflection_net);
    if (engine->synthesis_net && engine->owns_synthesis) lnn_free(engine->synthesis_net);
    if (engine->conflict_net && engine->owns_conflict) lnn_free(engine->conflict_net);
    if (engine->epistemic_net && engine->owns_epistemic) lnn_free(engine->epistemic_net);
    if (engine->hypothesis_net && engine->owns_hypothesis) lnn_free(engine->hypothesis_net);
    safe_free((void**)&engine->layer_embeddings);
    safe_free((void**)&engine->knowledge_base);
    safe_free((void**)&engine->conflict_history);
    safe_free((void**)&engine);
}

int dr_reflect(DeepReflectionEngine* engine,
                const char* topic,
                const float* context_data,
                size_t context_size,
                DRReflectionChain* chain_out) {
    if (!engine || !chain_out) return -1;
    if (!topic && !context_data) return -2;

    memset(chain_out, 0, sizeof(DRReflectionChain));

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;

    float base_embed[DR_EMBED_DIM] = {0};
    if (context_data && context_size > 0) {
        size_t copy_dim = (context_size < DR_EMBED_DIM) ? context_size : DR_EMBED_DIM;
        memcpy(base_embed, context_data, copy_dim * sizeof(float));
    } else if (topic) {
        size_t tlen = strlen(topic);
        for (size_t i = 0; i < tlen && i < DR_EMBED_DIM; i++) {
            base_embed[i] = (float)topic[i] / 255.0f;
        }
    }

    int max_layers = engine->config.max_iterations;
    if (max_layers > DR_MAX_CHAIN_LEN) max_layers = DR_MAX_CHAIN_LEN;

    memcpy(engine->layer_embeddings, base_embed, edim * sizeof(float));
    engine->layer_embed_count = 1;

    const float* prev_embeddings[DR_MAX_CHAIN_LEN];
    float prev_weights[DR_MAX_CHAIN_LEN];

    for (int layer = 0; layer < max_layers; layer++) {
        DRLayerResult* lr = &chain_out->layers[layer];
        lr->layer = (DRReflectionLayer)(layer % 7);

        float layer_input[DR_EMBED_DIM] = {0};
        for (size_t i = 0; i < engine->layer_embed_count; i++) {
            for (size_t j = 0; j < edim; j++) {
                layer_input[j] += engine->layer_embeddings[i * edim + j];
            }
        }
        float inv_count = 1.0f / (float)(engine->layer_embed_count > 0 ? engine->layer_embed_count : 1);
        for (size_t j = 0; j < edim; j++) {
            layer_input[j] *= inv_count;
        }

        float perspective_count = 3.0f + (float)(layer % 3);
        float multi_persp_in[DR_EMBED_DIM] = {0};
        for (size_t p = 0; p < (size_t)perspective_count; p++) {
            float persp_emb[DR_EMBED_DIM];
            generate_perspective_embedding(layer_input, (int)p, 0.3f + (float)layer * 0.05f, edim, persp_emb);
            float persp_out[DR_EMBED_DIM] = {0};
            lnn_forward(engine->reflection_net, persp_emb, persp_out);
            for (size_t j = 0; j < edim; j++) {
                multi_persp_in[j] += persp_out[j];
            }
        }
        for (size_t j = 0; j < edim; j++) {
            multi_persp_in[j] /= perspective_count;
            lr->content_embedding[j] = multi_persp_in[j];
        }

        if (layer < 7) {
            lr->depth_score = (float)(layer + 1) / 7.0f;
        } else {
            lr->depth_score = 1.0f;
        }

        float sim_to_prev = 0.0f;
        if (layer > 0) {
            sim_to_prev = cosine_sim_dr(
                chain_out->layers[layer - 1].content_embedding,
                lr->content_embedding, edim);
        }
        lr->novelty_score = 1.0f - sim_to_prev;

        lr->coherence_score = 0.0f;
        for (size_t i = 0; i < engine->layer_embed_count; i++) {
            lr->coherence_score += cosine_sim_dr(
                engine->layer_embeddings + i * edim,
                lr->content_embedding, edim);
        }
        if (engine->layer_embed_count > 0) {
            lr->coherence_score /= (float)engine->layer_embed_count;
        }

        float epistemic_out[3] = {0};
        lnn_forward(engine->epistemic_net, lr->content_embedding, epistemic_out);
        float epistemic_conf = DR_SIGMOID(epistemic_out[0]);
        float epistemic_uncertainty = DR_SIGMOID(epistemic_out[1]);
        float epistemic_novelty = DR_SIGMOID(epistemic_out[2]);

        lr->reflection_text = (char*)safe_calloc(2048, 1);
        int perspective_idx = layer % 8;
        snprintf(lr->reflection_text, 2048,
            "[%s][%s] 深度:%.2f 新颖性:%.2f 一致性:%.2f "
            "认知置信度:%.2f 不确定性:%.2f 认知新颖性:%.2f - %s",
            layer_names[lr->layer], perspective_names[perspective_idx],
            lr->depth_score, lr->novelty_score, lr->coherence_score,
            epistemic_conf, epistemic_uncertainty, epistemic_novelty,
            topic ? topic : "上下文反思");
        lr->text_len = strlen(lr->reflection_text);

        lr->contradiction_flag = (lr->coherence_score < 0.25f) ? 1.0f : 0.0f;
        lr->needs_deeper = (lr->depth_score < engine->config.depth_threshold ||
                           lr->novelty_score > engine->config.novelty_threshold);

        prev_embeddings[layer] = lr->content_embedding;
        prev_weights[layer] = 1.0f - lr->novelty_score * 0.5f;

        if (engine->layer_embed_count < DR_MAX_CHAIN_LEN) {
            memcpy(engine->layer_embeddings + engine->layer_embed_count * edim,
                   lr->content_embedding, edim * sizeof(float));
            engine->layer_embed_count++;
        }

        chain_out->insight_scores[layer] = lr->novelty_score * 0.4f +
                                           lr->depth_score * 0.3f +
                                           epistemic_conf * 0.3f;
        chain_out->num_insights = (size_t)(layer + 1);

        chain_out->num_layers++;
    }

    float total_depth = 0.0f;
    for (size_t i = 0; i < chain_out->num_layers; i++) {
        total_depth += chain_out->layers[i].depth_score;
    }
    chain_out->overall_depth = total_depth / (float)(chain_out->num_layers > 0 ? chain_out->num_layers : 1);

    float transformative = 0.0f;
    float novelty_sum = 0.0f;
    for (size_t i = 0; i < chain_out->num_layers; i++) {
        novelty_sum += chain_out->layers[i].novelty_score;
        if (i > 0) {
            float change = DR_ABS(chain_out->layers[i].depth_score - chain_out->layers[i - 1].depth_score);
            transformative += change;
        }
    }
    chain_out->transformative_potential = DR_CLAMP(
        (novelty_sum / (float)chain_out->num_layers) * 0.6f +
        (transformative / (float)(chain_out->num_layers > 1 ? chain_out->num_layers - 1 : 1)) * 0.4f,
        0.0f, 1.0f);

    chain_out->self_consistency = 0.0f;
    int pair_count = 0;
    for (size_t i = 0; i < chain_out->num_layers && i < 12; i++) {
        for (size_t j = i + 1; j < chain_out->num_layers && j < 12; j++) {
            chain_out->self_consistency += cosine_sim_dr(
                chain_out->layers[i].content_embedding,
                chain_out->layers[j].content_embedding, edim);
            pair_count++;
        }
    }
    if (pair_count > 0) chain_out->self_consistency /= (float)pair_count;

    if (engine->config.enable_causal_analysis) {
        float root_cause_score = 0.0f;
        for (size_t i = 1; i < chain_out->num_layers; i++) {
            float cause_effect = cosine_sim_dr(
                chain_out->layers[i - 1].content_embedding,
                chain_out->layers[i].content_embedding, edim);
            root_cause_score += cause_effect * chain_out->layers[i].depth_score;
        }
        if (chain_out->num_layers > 1) {
            float avg_cause = root_cause_score / (float)(chain_out->num_layers - 1);
            chain_out->self_consistency = (chain_out->self_consistency + avg_cause) * 0.5f;
        }
    }

    chain_out->synthesis = (char*)safe_calloc(4096, 1);
    char insight_buffer[2048] = {0};
    size_t insight_offset = 0;
    for (size_t i = 0; i < chain_out->num_layers && i < 5; i++) {
        if (chain_out->layers[i].novelty_score > engine->config.novelty_threshold) {
            int n = snprintf(insight_buffer + insight_offset,
                sizeof(insight_buffer) - insight_offset,
                "  洞察%zu: %s (新颖性:%.2f 深度:%.2f)\n",
                i + 1, layer_names[chain_out->layers[i].layer],
                chain_out->layers[i].novelty_score,
                chain_out->layers[i].depth_score);
            if (n > 0) insight_offset += (size_t)n;
            if (insight_offset >= sizeof(insight_buffer) - 128) break;
        }
    }

    snprintf(chain_out->synthesis, 4096,
        "深度反思综合结果:\n"
        "  反思层数: %zu  |  整体深度: %.2f  |  变革潜力: %.2f  |  自我一致性: %.2f\n"
        "  ---\n"
        "  主题: %s\n"
        "  === 关键洞察 ===\n"
        "%s"
        "  === 总结 ===\n"
        "  多视角分析覆盖 %zu 个认知层面, "
        "最高深度 %.2f, "
        "认知稳定性 %s",
        chain_out->num_layers, chain_out->overall_depth,
        chain_out->transformative_potential,
        chain_out->self_consistency,
        topic ? topic : "（无主题）",
        insight_buffer,
        chain_out->num_layers,
        chain_out->overall_depth,
        (chain_out->self_consistency > 0.6f) ? "高" :
        (chain_out->self_consistency > 0.3f) ? "中" : "需修正");
    chain_out->synthesis_len = strlen(chain_out->synthesis);

    return 0;
}

int dr_reflect_multi_passage(DeepReflectionEngine* engine,
                              const char* topic,
                              const float** passages, size_t num_passages,
                              const size_t* passage_sizes,
                              DRReflectionChain* chain_out) {
    if (!engine || !chain_out) return -1;
    if (!topic && (!passages || num_passages == 0)) return -2;

    memset(chain_out, 0, sizeof(DRReflectionChain));

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;

    float fused_embed[DR_EMBED_DIM] = {0};
    if (passages && num_passages > 0) {
        const float** emb_ptr = (const float**)safe_calloc(num_passages, sizeof(float*));
        float* weights = (float*)safe_calloc(num_passages, sizeof(float));
        for (size_t p = 0; p < num_passages; p++) {
            emb_ptr[p] = passages[p];
            float mag = 0.0f;
            for (size_t j = 0; j < (passage_sizes ? passage_sizes[p] : edim) && j < edim; j++) {
                mag += passages[p][j] * passages[p][j];
            }
            weights[p] = sqrtf(mag) + 0.1f;
        }
        weighted_embedding_fusion(emb_ptr, weights, num_passages, edim, fused_embed);
        safe_free((void**)&emb_ptr);
        safe_free((void**)&weights);
    } else if (topic) {
        size_t tlen = strlen(topic);
        for (size_t i = 0; i < tlen && i < edim; i++) {
            fused_embed[i] = (float)topic[i] / 255.0f;
        }
    }

    int max_layers = engine->config.max_iterations;
    if (max_layers > DR_MAX_CHAIN_LEN) max_layers = DR_MAX_CHAIN_LEN;

    memcpy(engine->layer_embeddings, fused_embed, edim * sizeof(float));
    engine->layer_embed_count = 1;

    unsigned int seed = (unsigned int)(topic ? strlen(topic) : 42);

    for (int layer = 0; layer < max_layers; layer++) {
        DRLayerResult* lr = &chain_out->layers[layer];
        lr->layer = (DRReflectionLayer)(layer % 7);

        float layer_input[DR_EMBED_DIM] = {0};
        for (size_t i = 0; i < engine->layer_embed_count; i++) {
            for (size_t j = 0; j < edim; j++) {
                layer_input[j] += engine->layer_embeddings[i * edim + j];
            }
        }
        float inv_count = 1.0f / (float)(engine->layer_embed_count > 0 ? engine->layer_embed_count : 1);
        for (size_t j = 0; j < edim; j++) {
            layer_input[j] *= inv_count;
        }

        float hyp_input[DR_EMBED_DIM];
        memcpy(hyp_input, layer_input, edim * sizeof(float));
        for (size_t j = 0; j < edim; j++) {
            hyp_input[j] += ((float)((seed + layer * 137) % 1000) / 1000.0f - 0.5f) * 0.2f;
        }

        float hyp_out[DR_EMBED_DIM] = {0};
        lnn_forward(engine->hypothesis_net, hyp_input, hyp_out);

        float persp_out[DR_EMBED_DIM] = {0};
        lnn_forward(engine->reflection_net, hyp_out, persp_out);

        for (size_t j = 0; j < edim; j++) {
            lr->content_embedding[j] = persp_out[j] * 0.7f + hyp_out[j] * 0.3f;
        }

        lr->depth_score = (layer < 7) ? (float)(layer + 1) / 7.0f : 1.0f;

        float sim_prev = (layer > 0) ? cosine_sim_dr(
            chain_out->layers[layer - 1].content_embedding,
            lr->content_embedding, edim) : 0.0f;
        lr->novelty_score = 1.0f - sim_prev;

        lr->coherence_score = 0.0f;
        for (size_t i = 0; i < engine->layer_embed_count; i++) {
            lr->coherence_score += cosine_sim_dr(
                engine->layer_embeddings + i * edim,
                lr->content_embedding, edim);
        }
        if (engine->layer_embed_count > 0)
            lr->coherence_score /= (float)engine->layer_embed_count;

        lr->reflection_text = (char*)safe_calloc(2048, 1);
        snprintf(lr->reflection_text, 2048,
            "[%s][多段落反射] 深度:%.2f 新颖性:%.2f 一致性:%.2f - %s",
            layer_names[lr->layer], lr->depth_score,
            lr->novelty_score, lr->coherence_score,
            topic ? topic : "多段落反思");
        lr->text_len = strlen(lr->reflection_text);

        lr->contradiction_flag = (lr->coherence_score < 0.25f) ? 1.0f : 0.0f;
        lr->needs_deeper = (lr->depth_score < engine->config.depth_threshold);

        if (engine->layer_embed_count < DR_MAX_CHAIN_LEN) {
            memcpy(engine->layer_embeddings + engine->layer_embed_count * edim,
                   lr->content_embedding, edim * sizeof(float));
            engine->layer_embed_count++;
        }

        chain_out->insight_scores[layer] = lr->novelty_score * 0.5f + lr->depth_score * 0.5f;
        chain_out->num_insights = (size_t)(layer + 1);
        chain_out->num_layers++;
    }

    float total_depth = 0.0f;
    for (size_t i = 0; i < chain_out->num_layers; i++)
        total_depth += chain_out->layers[i].depth_score;
    chain_out->overall_depth = total_depth / (float)(chain_out->num_layers > 0 ? chain_out->num_layers : 1);

    chain_out->self_consistency = 0.0f;
    int pc = 0;
    for (size_t i = 0; i < chain_out->num_layers && i < 10; i++) {
        for (size_t j = i + 1; j < chain_out->num_layers && j < 10; j++) {
            chain_out->self_consistency += cosine_sim_dr(
                chain_out->layers[i].content_embedding,
                chain_out->layers[j].content_embedding, edim);
            pc++;
        }
    }
    if (pc > 0) chain_out->self_consistency /= (float)pc;

    float tv = 0.0f;
    for (size_t i = 1; i < chain_out->num_layers; i++) {
        tv += DR_ABS(chain_out->layers[i].depth_score - chain_out->layers[i - 1].depth_score);
    }
    chain_out->transformative_potential = DR_CLAMP(tv / (float)(chain_out->num_layers > 1 ? chain_out->num_layers - 1 : 1), 0.0f, 1.0f);

    chain_out->synthesis = (char*)safe_calloc(4096, 1);
    snprintf(chain_out->synthesis, 4096,
        "多段落深度反思:\n  段落数: %zu  |  反思层数: %zu  |  深度: %.2f  |  变革潜力: %.2f\n  主题: %s",
        num_passages, chain_out->num_layers, chain_out->overall_depth,
        chain_out->transformative_potential, topic ? topic : "（无主题）");
    chain_out->synthesis_len = strlen(chain_out->synthesis);

    return 0;
}

int dr_epistemic_assessment(DeepReflectionEngine* engine,
                             DRReflectionChain* chain,
                             float* knowledge_certainty_out,
                             float* knowledge_coverage_out,
                             float* knowledge_boundary_out) {
    if (!engine || !chain) return -1;

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;

    float certainty = 0.0f;
    float coverage = 0.0f;
    float boundary = 0.0f;

    for (size_t i = 0; i < chain->num_layers; i++) {
        float epi_in[DR_EMBED_DIM];
        memcpy(epi_in, chain->layers[i].content_embedding, edim * sizeof(float));
        float epi_out[3] = {0};
        lnn_forward(engine->epistemic_net, epi_in, epi_out);

        float c = DR_SIGMOID(epi_out[0]);
        float cov = DR_SIGMOID(epi_out[1]);
        float b = 1.0f - DR_SIGMOID(epi_out[2]);

        certainty += c * chain->layers[i].depth_score;
        coverage += cov;
        boundary += b;
    }

    float inv = 1.0f / (float)(chain->num_layers > 0 ? chain->num_layers : 1);
    if (knowledge_certainty_out) *knowledge_certainty_out = DR_CLAMP(certainty * inv, 0.0f, 1.0f);
    if (knowledge_coverage_out) *knowledge_coverage_out = DR_CLAMP(coverage * inv, 0.0f, 1.0f);
    if (knowledge_boundary_out) *knowledge_boundary_out = DR_CLAMP(boundary * inv, 0.0f, 1.0f);

    return 0;
}

int dr_detect_knowledge_conflicts(DeepReflectionEngine* engine,
                                    DRReflectionChain* chain,
                                    DRConflictInfo* conflicts_out,
                                    size_t* num_conflicts) {
    if (!engine || !chain || !conflicts_out || !num_conflicts) return -1;
    if (!engine->config.enable_contradiction_detection) {
        *num_conflicts = 0;
        return 0;
    }

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;
    size_t count = 0;
    size_t max_conflicts = (*num_conflicts < DR_MAX_CONTRADICTIONS) ? *num_conflicts : DR_MAX_CONTRADICTIONS;

    for (size_t i = 0; i < chain->num_layers && count < max_conflicts; i++) {
        float prior_i = chain->layers[i].depth_score * 0.5f + chain->layers[i].coherence_score * 0.5f;

        for (size_t j = i + 1; j < chain->num_layers && count < max_conflicts; j++) {
            float prior_j = chain->layers[j].depth_score * 0.5f + chain->layers[j].coherence_score * 0.5f;

            float bcs = bayesian_conflict_score(
                chain->layers[i].content_embedding,
                chain->layers[j].content_embedding,
                prior_i, prior_j, edim);

            float con_input[DR_EMBED_DIM * 2];
            memcpy(con_input, chain->layers[i].content_embedding, edim * sizeof(float));
            memcpy(con_input + edim, chain->layers[j].content_embedding, edim * sizeof(float));
            float con_out[1] = {0};
            lnn_forward(engine->conflict_net, con_input, con_out);
            float nn_conflict = DR_SIGMOID(con_out[0]);

            float combined_conflict = bcs * 0.6f + nn_conflict * 0.4f;

            if (combined_conflict > 0.35f) {
                DRConflictInfo* ci = &conflicts_out[count];
                ci->layer_a = i;
                ci->layer_b = j;
                ci->conflict_score = combined_conflict;
                ci->bayesian_score = bcs;
                ci->nn_score = nn_conflict;
                ci->similarity = cosine_sim_dr(
                    chain->layers[i].content_embedding,
                    chain->layers[j].content_embedding, edim);

                snprintf(ci->description, sizeof(ci->description),
                    "知识冲突: [层%zu:%s] vs [层%zu:%s] "
                    "(冲突:%.2f 贝叶斯:%.2f 网络:%.2f 相似度:%.2f)",
                    i, layer_names[chain->layers[i].layer],
                    j, layer_names[chain->layers[j].layer],
                    combined_conflict, bcs, nn_conflict, ci->similarity);

                ci->resolution = DR_RESOLVE_UNRESOLVED;

                if (engine->conflict_count < DR_MAX_CONTRADICTIONS) {
                    engine->conflict_history[engine->conflict_count * 4 + 0] = (float)i;
                    engine->conflict_history[engine->conflict_count * 4 + 1] = (float)j;
                    engine->conflict_history[engine->conflict_count * 4 + 2] = combined_conflict;
                    engine->conflict_history[engine->conflict_count * 4 + 3] = (float)ci->resolution;
                    engine->conflict_count++;
                }

                count++;
            }
        }
    }

    *num_conflicts = count;
    return 0;
}

int dr_resolve_conflicts(DeepReflectionEngine* engine,
                          DRConflictInfo* conflicts,
                          size_t num_conflicts,
                          DRReflectionChain* chain_out) {
    if (!engine || !conflicts || !chain_out) return -1;

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;
    int resolved_count = 0;

    for (size_t c = 0; c < num_conflicts; c++) {
        DRConflictInfo* ci = &conflicts[c];
        if (ci->resolution != DR_RESOLVE_UNRESOLVED) continue;

        size_t idx_a = ci->layer_a;
        size_t idx_b = ci->layer_b;

        float belief_a[DR_EMBED_DIM], belief_b[DR_EMBED_DIM];
        memcpy(belief_a, chain_out->layers[idx_a].content_embedding, edim * sizeof(float));
        memcpy(belief_b, chain_out->layers[idx_b].content_embedding, edim * sizeof(float));

        float confidence_a = chain_out->layers[idx_a].depth_score;
        float confidence_b = chain_out->layers[idx_b].depth_score;
        float total_conf = confidence_a + confidence_b + 1e-10f;

        float merged[DR_EMBED_DIM];
        for (size_t j = 0; j < edim; j++) {
            merged[j] = (belief_a[j] * confidence_a + belief_b[j] * confidence_b) / total_conf;
        }

        memcpy(chain_out->layers[idx_a].content_embedding, merged, edim * sizeof(float));
        chain_out->layers[idx_a].contradiction_flag = 0.0f;
        chain_out->layers[idx_a].coherence_score =
            (chain_out->layers[idx_a].coherence_score + chain_out->layers[idx_b].coherence_score) * 0.5f;
        chain_out->layers[idx_b].contradiction_flag = 0.0f;

        ci->resolution = DR_RESOLVE_MERGED;

        if (engine->conflict_count > 0 && c < engine->conflict_count) {
            engine->conflict_history[c * 4 + 3] = (float)DR_RESOLVE_MERGED;
        }

        resolved_count++;
    }

    float merged_depth = 0.0f;
    for (size_t i = 0; i < chain_out->num_layers; i++) {
        merged_depth += chain_out->layers[i].depth_score;
    }
    chain_out->overall_depth = merged_depth / (float)(chain_out->num_layers > 0 ? chain_out->num_layers : 1);

    return resolved_count;
}

int dr_generate_hypotheses(DeepReflectionEngine* engine,
                            DRReflectionChain* chain,
                            float* hypotheses_out,
                            size_t* num_hypotheses,
                            size_t max_hypotheses) {
    if (!engine || !chain || !hypotheses_out || !num_hypotheses) return -1;

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;
    size_t count = 0;
    size_t max_h = (max_hypotheses < 16) ? max_hypotheses : 16;

    unsigned int seed = 42;
    for (size_t i = 0; i < chain->num_layers && count < max_h; i++) {
        if (chain->layers[i].novelty_score < 0.3f) continue;

        float base[DR_EMBED_DIM];
        memcpy(base, chain->layers[i].content_embedding, edim * sizeof(float));

        for (size_t variant = 0; variant < 2 && count < max_h; variant++) {
            float hyp_in[DR_EMBED_DIM];
            for (size_t j = 0; j < edim; j++) {
                float noise = ((float)((seed + i * 37 + variant * 73) % 1000) / 1000.0f - 0.5f) * 0.3f;
                hyp_in[j] = base[j] + noise;
            }

            float hyp_out[DR_EMBED_DIM] = {0};
            lnn_forward(engine->hypothesis_net, hyp_in, hyp_out);

            float* out = hypotheses_out + count * edim;
            memcpy(out, hyp_out, edim * sizeof(float));
            count++;
        }
    }

    *num_hypotheses = count;
    return 0;
}

int dr_root_cause_analysis(DeepReflectionEngine* engine,
                            DRReflectionChain* chain,
                            size_t* root_layer_idx,
                            float* root_confidence) {
    if (!engine || !chain || !root_layer_idx || !root_confidence) return -1;
    if (chain->num_layers == 0) return -2;

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;
    float best_cause_score = -1.0f;
    size_t best_idx = 0;

    for (size_t i = 0; i < chain->num_layers; i++) {
        float downstream_influence = 0.0f;
        int influence_count = 0;

        for (size_t j = i + 1; j < chain->num_layers; j++) {
            float sim = cosine_sim_dr(
                chain->layers[i].content_embedding,
                chain->layers[j].content_embedding, edim);
            float influence = sim * chain->layers[j].depth_score * (1.0f - chain->layers[j].novelty_score);
            downstream_influence += influence;
            influence_count++;
        }

        for (size_t j = 0; j < i; j++) {
            float sim = cosine_sim_dr(
                chain->layers[i].content_embedding,
                chain->layers[j].content_embedding, edim);
            float influence = sim * chain->layers[i].depth_score * (1.0f - chain->layers[i].novelty_score);
            downstream_influence += influence * 0.3f;
            influence_count++;
        }

        float avg_influence = (influence_count > 0) ?
            downstream_influence / (float)influence_count : 0.0f;

        float cause_score = avg_influence * chain->layers[i].depth_score *
                           (1.0f + chain->layers[i].coherence_score) * 0.5f;

        if (cause_score > best_cause_score) {
            best_cause_score = cause_score;
            best_idx = i;
        }
    }

    *root_layer_idx = best_idx;
    *root_confidence = DR_CLAMP(best_cause_score, 0.0f, 1.0f);
    return 0;
}

int dr_get_insights(DeepReflectionEngine* engine,
                     DRReflectionChain* chain,
                     char** insights_out,
                     size_t* num_insights) {
    if (!engine || !chain || !insights_out || !num_insights) return -1;

    size_t count = 0;
    for (size_t i = 0; i < chain->num_layers && count < DR_MAX_INSIGHTS; i++) {
        if (chain->insight_scores[i] > engine->config.novelty_threshold) {
            insights_out[count] = chain->layers[i].reflection_text;
            count++;
        }
    }

    if (count == 0 && chain->num_layers > 0) {
        insights_out[0] = chain->layers[chain->num_layers - 1].reflection_text;
        count = 1;
    }

    *num_insights = count;
    return 0;
}

int dr_detect_contradictions(DeepReflectionEngine* engine,
                              DRReflectionChain* chain,
                              char** contradictions_out,
                              size_t* num_contradictions) {
    if (!engine || !chain || !contradictions_out || !num_contradictions) return -1;
    if (!engine->config.enable_contradiction_detection) {
        *num_contradictions = 0;
        return 0;
    }

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;
    size_t count = 0;
    size_t max_out = (*num_contradictions < DR_MAX_CONTRADICTIONS) ? *num_contradictions : DR_MAX_CONTRADICTIONS;

    for (size_t i = 0; i < chain->num_layers && count < max_out; i++) {
        for (size_t j = i + 1; j < chain->num_layers && count < max_out; j++) {
            float sim = cosine_sim_dr(
                chain->layers[i].content_embedding,
                chain->layers[j].content_embedding, edim);

            float bayesian = bayesian_conflict_score(
                chain->layers[i].content_embedding,
                chain->layers[j].content_embedding,
                chain->layers[i].depth_score,
                chain->layers[j].depth_score, edim);

            float combined_conflict = bayesian * 0.7f + (1.0f - sim) * 0.3f;

            if (combined_conflict > 0.4f && chain->layers[i].depth_score > 0.3f &&
                chain->layers[j].depth_score > 0.3f) {
                char* buf = (char*)safe_calloc(512, 1);
                snprintf(buf, 512, "矛盾: 层%zu[%s:%.2f] vs 层%zu[%s:%.2f] "
                         "(贝叶斯冲突:%.2f 相似度:%.2f 综合:%.2f)",
                         i, layer_names[chain->layers[i].layer],
                         chain->layers[i].depth_score,
                         j, layer_names[chain->layers[j].layer],
                         chain->layers[j].depth_score,
                         bayesian, sim, combined_conflict);
                contradictions_out[count] = buf;
                count++;
            }
        }
    }

    *num_contradictions = count;
    return 0;
}

int dr_generate_synthesis(DeepReflectionEngine* engine,
                           DRReflectionChain* chain,
                           char* synthesis_out,
                           size_t max_len) {
    if (!engine || !chain || !synthesis_out) return -1;

    if (chain->synthesis) {
        strncpy(synthesis_out, chain->synthesis, max_len - 1);
        synthesis_out[max_len - 1] = '\0';
        return 0;
    }

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;

    const float* top_embs[3];
    float top_weights[3];
    size_t num_top = (chain->num_layers < 3) ? chain->num_layers : 3;
    for (size_t i = 0; i < num_top; i++) {
        top_embs[i] = chain->layers[i].content_embedding;
        top_weights[i] = chain->layers[i].depth_score;
    }

    float syn_input[DR_EMBED_DIM] = {0};
    weighted_embedding_fusion(top_embs, top_weights, num_top, edim, syn_input);

    float syn_output[DR_EMBED_DIM] = {0};
    lnn_forward(engine->synthesis_net, syn_input, syn_output);

    snprintf(synthesis_out, max_len,
        "深度反思综合 (%zu层, 深度%.2f, 变革潜力%.2f, 一致性%.2f)",
        chain->num_layers, chain->overall_depth,
        chain->transformative_potential, chain->self_consistency);

    return 0;
}

int dr_chain_to_plan(DeepReflectionEngine* engine,
                      DRReflectionChain* chain,
                      float* plan_actions,
                      size_t* num_actions) {
    if (!engine || !chain || !plan_actions || !num_actions) return -1;

    size_t actions = (chain->num_layers < 16) ? chain->num_layers : 16;

    size_t root_idx = 0;
    float root_conf = 0.0f;
    dr_root_cause_analysis(engine, chain, &root_idx, &root_conf);

    for (size_t i = 0; i < actions; i++) {
        size_t layer_idx = i * chain->num_layers / actions;
        if (layer_idx >= chain->num_layers) layer_idx = chain->num_layers - 1;

        DRLayerResult* lr = &chain->layers[layer_idx];
        float* action = plan_actions + i * 5;
        action[0] = (float)lr->layer;
        action[1] = lr->depth_score;
        action[2] = lr->novelty_score;
        action[3] = lr->coherence_score;
        action[4] = (layer_idx == root_idx) ? 1.0f : 0.0f;
    }

    *num_actions = actions;
    return 0;
}

int dr_integrate_with_metacognition(DeepReflectionEngine* engine,
                                     MetacognitionSystem* meta_system,
                                     DRReflectionChain* chain) {
    if (!engine || !meta_system || !chain) return -1;

    float ref_quality = chain->overall_depth * 0.4f +
                        chain->self_consistency * 0.3f +
                        chain->transformative_potential * 0.3f;

    size_t root_idx = 0;
    float root_conf = 0.0f;
    dr_root_cause_analysis(engine, chain, &root_idx, &root_conf);

    float certainty = 0.0f, coverage = 0.0f, boundary = 0.0f;
    dr_epistemic_assessment(engine, chain, &certainty, &coverage, &boundary);

    char assessment[2048];
    snprintf(assessment, sizeof(assessment),
        "深度反思整合评估:\n"
        "  反思质量: %.2f  |  认知确定性: %.2f  |  知识覆盖率: %.2f  |  知识边界: %.2f\n"
        "  根因层: %zu (置信度:%.2f)\n"
        "  整体深度: %.2f  |  一致性: %.2f  |  变革潜力: %.2f",
        ref_quality, certainty, coverage, boundary,
        root_idx, root_conf,
        chain->overall_depth, chain->self_consistency,
        chain->transformative_potential);

    metacognition_neutral_self_assessment(meta_system, 0, assessment, strlen(assessment));

    return 0;
}

void dr_chain_free(DRReflectionChain* chain) {
    if (!chain) return;
    for (size_t i = 0; i < chain->num_layers; i++) {
        safe_free((void**)&chain->layers[i].reflection_text);
    }
    safe_free((void**)&chain->synthesis);
    memset(chain, 0, sizeof(DRReflectionChain));
}
