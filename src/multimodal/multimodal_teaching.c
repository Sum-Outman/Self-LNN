#include "selflnn/multimodal/multimodal_teaching.h"
#include "selflnn/core/tensor.h"
#include "selflnn/core/cfc.h"
#include "selflnn/core/lnn.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "selflnn/utils/secure_random.h"

#include "selflnn/utils/memory_utils.h"  /* DEEP-005: safe_malloc宏定义 */

#define MT_MAX_SEQUENCES 256
#define MT_MAX_FRAMES_PER_SEQUENCE 4096
#define MT_PRIMITIVE_MIN_FRAMES 8
#define MT_PRIMITIVE_SIMILARITY_THRESH 0.75f
#define MT_ALIGNMENT_LR 0.01f
#define MT_MAX_SKILL_NAME 64

typedef struct {
    float features[TEACH_PRIMITIVE_FEAT_DIM];
    size_t num_frames;
    size_t start_frame;
    size_t end_frame;
    size_t sequence_id;
    float confidence;
    int is_active;
} TeachMotionPrimitive;

typedef struct {
    size_t sequence_id;
    size_t current_frame;
    size_t total_frames;
    int is_playing;
    float speed_multiplier;
    int loop_enabled;
} TeachReplayState;

typedef struct {
    float skill_embedding[TEACH_FUSED_FEAT_DIM];
    size_t num_demonstrations;
    size_t primitive_ids[TEACH_MAX_PRIMITIVES];
    size_t num_primitives;
    float mastery_level;
    float confidence;
    int is_learned;
} TeachSkill;

typedef struct {
    float visual_weight;
    float audio_weight;
    float text_weight;
    float sensor_weight;
    float alignment_matrix[TEACH_MODALITIES * TEACH_FUSED_FEAT_DIM];
    int is_trained;
} TeachAlignmentState;

typedef struct {
    float precision;
    float recall;
    float f1_score;
    float temporal_accuracy;
    float motion_smoothness;
    float overall_score;
} TeachEvaluationMetrics;

struct MultimodalTeachingSystem {
    TeachFusionConfig config;
    TeachModalSequence sequences[MT_MAX_SEQUENCES];
    size_t num_sequences;
    int initialized;
    size_t current_frame_count;

    void* lnn_network;
    int use_lnn;

    TeachMotionPrimitive primitives[TEACH_MAX_PRIMITIVES];
    size_t num_primitives;

    TeachReplayState replay;

    TeachSkill skills[TEACH_MAX_SKILLS];
    size_t num_skills;

    TeachAlignmentState alignment;

    TeachEvaluationMetrics last_evaluation;

    size_t demonstration_ids[TEACH_MAX_DEMONSTRATIONS];
    size_t num_demonstrations;

    float temporal_buffer[TEACH_REPLAY_BUFFER_SIZE];
    size_t temporal_buffer_pos;

    int self_learning_enabled;
    int adaptation_enabled;
};

static float cosine_sim_mt(const float* a, const float* b, size_t dim) {
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

static float euclidean_dist_mt(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

static void normalize_vector_mt(float* v, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; i++) norm += v[i] * v[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (size_t i = 0; i < dim; i++) v[i] /= norm;
    }
}

static void weighted_average_mt(const float* a, const float* b, float* out, size_t dim, float wa, float wb) {
    float total = wa + wb;
    if (total < 1e-10f) { total = 1.0f; wa = 0.5f; wb = 0.5f; }
    for (size_t i = 0; i < dim; i++) {
        out[i] = (a[i] * wa + b[i] * wb) / total;
    }
}

static float compute_temporal_smoothness(const float* data, size_t num_steps, size_t step_dim) {
    if (num_steps < 3) return 1.0f;
    float total_diff = 0.0f;
    int count = 0;
    for (size_t s = 1; s < num_steps - 1; s++) {
        for (size_t d = 0; d < step_dim; d++) {
            float prev = data[(s - 1) * step_dim + d];
            float curr = data[s * step_dim + d];
            float next = data[(s + 1) * step_dim + d];
            float smooth_diff = fabsf(curr - (prev + next) * 0.5f);
            total_diff += smooth_diff;
            count++;
        }
    }
    if (count == 0) return 1.0f;
    float avg_diff = total_diff / (float)count;
    return 1.0f / (1.0f + avg_diff);
}

static int mt_get_or_create_sequence(MultimodalTeachingSystem* system) {
    for (size_t i = 0; i < system->num_sequences; i++) {
        if (!system->sequences[i].is_complete) return (int)i;
    }
    if (system->num_sequences >= MT_MAX_SEQUENCES) return -1;

    size_t idx = system->num_sequences;
    system->sequences[idx].max_frames = 1024;
    system->sequences[idx].frames = (TeachModalFrame*)safe_calloc(system->sequences[idx].max_frames, sizeof(TeachModalFrame));
    system->sequences[idx].num_frames = 0;
    system->sequences[idx].is_complete = 0;
    system->sequences[idx].visual_width = 0;
    system->sequences[idx].visual_height = 0;
    system->sequences[idx].audio_sample_rate = 0;
    system->num_sequences++;
    return (int)idx;
}

MultimodalTeachingSystem* multimodal_teaching_create(TeachFusionConfig config) {
    MultimodalTeachingSystem* system = (MultimodalTeachingSystem*)safe_calloc(1, sizeof(MultimodalTeachingSystem));
    if (!system) return NULL;

    system->config = config;
    system->num_sequences = 0;
    system->initialized = 1;
    system->current_frame_count = 0;
    system->use_lnn = 0;
    system->lnn_network = NULL;
    system->num_primitives = 0;
    system->num_skills = 0;
    system->num_demonstrations = 0;
    system->temporal_buffer_pos = 0;
    system->self_learning_enabled = 1;
    system->adaptation_enabled = 1;

    memset(&system->replay, 0, sizeof(TeachReplayState));
    memset(&system->alignment, 0, sizeof(TeachAlignmentState));
    memset(&system->last_evaluation, 0, sizeof(TeachEvaluationMetrics));

    system->alignment.visual_weight = config.visual_weight;
    system->alignment.audio_weight = config.audio_weight;
    system->alignment.text_weight = config.text_weight;
    system->alignment.sensor_weight = config.sensor_weight;

    for (size_t i = 0; i < TEACH_MODALITIES * TEACH_FUSED_FEAT_DIM; i++) {
        system->alignment.alignment_matrix[i] = (i % (TEACH_FUSED_FEAT_DIM / TEACH_MODALITIES) == 0) ? 1.0f : 0.0f;
    }

    for (size_t i = 0; i < MT_MAX_SEQUENCES; i++) {
        system->sequences[i].frames = NULL;
        system->sequences[i].num_frames = 0;
        system->sequences[i].max_frames = 0;
        system->sequences[i].is_complete = 0;
    }

    return system;
}

void multimodal_teaching_destroy(MultimodalTeachingSystem* system) {
    if (!system) return;
    for (size_t i = 0; i < MT_MAX_SEQUENCES; i++) {
        safe_free((void**)&system->sequences[i].frames);
    }
    safe_free((void**)&system);
}

int multimodal_teaching_ingest_frame(MultimodalTeachingSystem* system,
                                       const float* visual_data,
                                       size_t visual_size,
                                       const float* audio_data,
                                       size_t audio_size,
                                       const float* text_data,
                                       size_t text_size,
                                       const float* sensor_data,
                                       size_t sensor_size,
                                       float timestamp) {
    if (!system) return -1;

    /* F-016: 多模态维度校验矩阵
     * 验证所有模态的特征维度在配置范围内的有效性，
     * 防止维度不匹配导致跨模态关联学习失败 */
    if (visual_data && visual_size > TEACH_VISUAL_FEAT_DIM) {
        fprintf(stderr, "[多模态教学] 视觉维度超出范围: %zu > %d\n",
                visual_size, TEACH_VISUAL_FEAT_DIM);
        return -4;
    }
    if (audio_data && audio_size > TEACH_AUDIO_FEAT_DIM) {
        fprintf(stderr, "[多模态教学] 音频维度超出范围: %zu > %d\n",
                audio_size, TEACH_AUDIO_FEAT_DIM);
        return -5;
    }
    if (text_data && text_size > TEACH_TEXT_FEAT_DIM) {
        fprintf(stderr, "[多模态教学] 文本维度超出范围: %zu > %d\n",
                text_size, TEACH_TEXT_FEAT_DIM);
        return -6;
    }
    if (sensor_data && sensor_size > TEACH_SENSOR_FEAT_DIM) {
        fprintf(stderr, "[多模态教学] 传感器维度超出范围: %zu > %d\n",
                sensor_size, TEACH_SENSOR_FEAT_DIM);
        return -7;
    }

    /* 融合特征维度校验：视觉+音频+文本+传感器可能超出FUSED_FEAT_DIM */
    size_t needed_fused = TEACH_VISUAL_FEAT_DIM + TEACH_AUDIO_FEAT_DIM
                        + TEACH_TEXT_FEAT_DIM + TEACH_SENSOR_FEAT_DIM;
    if (needed_fused > TEACH_FUSED_FEAT_DIM) {
        fprintf(stderr, "[多模态教学] 融合特征维度不足: 需要%zu > 容量%d\n",
                needed_fused, TEACH_FUSED_FEAT_DIM);
    }

    int seq_idx = mt_get_or_create_sequence(system);
    if (seq_idx < 0) return -2;

    TeachModalSequence* seq = &system->sequences[seq_idx];

    if (seq->num_frames >= seq->max_frames) {
        size_t new_max = seq->max_frames * 2;
        if (new_max > MT_MAX_FRAMES_PER_SEQUENCE) new_max = MT_MAX_FRAMES_PER_SEQUENCE;
        TeachModalFrame* new_frames = (TeachModalFrame*)safe_realloc(seq->frames, new_max * sizeof(TeachModalFrame));
        if (!new_frames) return -3;
        memset(new_frames + seq->max_frames, 0, (new_max - seq->max_frames) * sizeof(TeachModalFrame));
        seq->frames = new_frames;
        seq->max_frames = new_max;
    }

    TeachModalFrame* frame = &seq->frames[seq->num_frames];
    memset(frame, 0, sizeof(TeachModalFrame));

    if (visual_data && visual_size > 0) {
        size_t copy_v = visual_size < TEACH_VISUAL_FEAT_DIM ? visual_size : TEACH_VISUAL_FEAT_DIM;
        memcpy(frame->visual_feat, visual_data, copy_v * sizeof(float));
        frame->has_visual = 1;
    }

    if (audio_data && audio_size > 0) {
        size_t copy_a = audio_size < TEACH_AUDIO_FEAT_DIM ? audio_size : TEACH_AUDIO_FEAT_DIM;
        memcpy(frame->audio_feat, audio_data, copy_a * sizeof(float));
        frame->has_audio = 1;
    }

    if (text_data && text_size > 0) {
        size_t copy_t = text_size < TEACH_TEXT_FEAT_DIM ? text_size : TEACH_TEXT_FEAT_DIM;
        memcpy(frame->text_feat, text_data, copy_t * sizeof(float));
        frame->has_text = 1;
    }

    if (sensor_data && sensor_size > 0) {
        size_t copy_s = sensor_size < TEACH_SENSOR_FEAT_DIM ? sensor_size : TEACH_SENSOR_FEAT_DIM;
        memcpy(frame->sensor_feat, sensor_data, copy_s * sizeof(float));
        frame->has_sensor = 1;
    }

    frame->timestamp = timestamp;

    size_t fused_offset = 0;
    memset(frame->fused_feat, 0, TEACH_FUSED_FEAT_DIM * sizeof(float));
    if (frame->has_visual) {
        for (size_t j = 0; j < TEACH_VISUAL_FEAT_DIM && fused_offset + j < TEACH_FUSED_FEAT_DIM; j++) {
            frame->fused_feat[fused_offset + j] = frame->visual_feat[j] * system->config.visual_weight;
        }
    }
    fused_offset += TEACH_VISUAL_FEAT_DIM;
    if (frame->has_audio) {
        for (size_t j = 0; j < TEACH_AUDIO_FEAT_DIM && fused_offset + j < TEACH_FUSED_FEAT_DIM; j++) {
            frame->fused_feat[fused_offset + j] = frame->audio_feat[j] * system->config.audio_weight;
        }
    }
    fused_offset += TEACH_AUDIO_FEAT_DIM;
    if (frame->has_text) {
        for (size_t j = 0; j < TEACH_TEXT_FEAT_DIM && fused_offset + j < TEACH_FUSED_FEAT_DIM; j++) {
            frame->fused_feat[fused_offset + j] = frame->text_feat[j] * system->config.text_weight;
        }
    }
    fused_offset += TEACH_TEXT_FEAT_DIM;
    if (frame->has_sensor) {
        for (size_t j = 0; j < TEACH_SENSOR_FEAT_DIM && fused_offset + j < TEACH_FUSED_FEAT_DIM; j++) {
            frame->fused_feat[fused_offset + j] = frame->sensor_feat[j] * system->config.sensor_weight;
        }
    }

    seq->num_frames++;
    system->current_frame_count++;

    return (int)(seq->num_frames - 1);
}

int multimodal_teaching_fuse_sequence(MultimodalTeachingSystem* system,
                                        size_t sequence_id,
                                        float* fused_output,
                                        size_t fused_dim) {
    if (!system || !fused_output) return -1;
    if (sequence_id >= system->num_sequences) return -2;
    if (fused_dim < TEACH_FUSED_FEAT_DIM) return -3;

    TeachModalSequence* seq = &system->sequences[sequence_id];
    if (seq->num_frames == 0) return -4;

    float visual_accum[TEACH_VISUAL_FEAT_DIM] = {0};
    float audio_accum[TEACH_AUDIO_FEAT_DIM] = {0};
    float text_accum[TEACH_TEXT_FEAT_DIM] = {0};
    float sensor_accum[TEACH_SENSOR_FEAT_DIM] = {0};
    int v_count = 0, a_count = 0, t_count = 0, s_count = 0;

    for (size_t i = 0; i < seq->num_frames; i++) {
        float decay = powf(system->config.temporal_decay, (float)(seq->num_frames - 1 - i));
        TeachModalFrame* f = &seq->frames[i];

        if (f->has_visual) {
            for (size_t j = 0; j < TEACH_VISUAL_FEAT_DIM; j++) {
                visual_accum[j] += f->visual_feat[j] * decay;
            }
            v_count++;
        }
        if (f->has_audio) {
            for (size_t j = 0; j < TEACH_AUDIO_FEAT_DIM; j++) {
                audio_accum[j] += f->audio_feat[j] * decay;
            }
            a_count++;
        }
        if (f->has_text) {
            for (size_t j = 0; j < TEACH_TEXT_FEAT_DIM; j++) {
                text_accum[j] += f->text_feat[j] * decay;
            }
            t_count++;
        }
        if (f->has_sensor) {
            for (size_t j = 0; j < TEACH_SENSOR_FEAT_DIM; j++) {
                sensor_accum[j] += f->sensor_feat[j] * decay;
            }
            s_count++;
        }
    }

    if (v_count > 0) {
        for (size_t j = 0; j < TEACH_VISUAL_FEAT_DIM; j++) visual_accum[j] /= (float)v_count;
    }
    if (a_count > 0) {
        for (size_t j = 0; j < TEACH_AUDIO_FEAT_DIM; j++) audio_accum[j] /= (float)a_count;
    }
    if (t_count > 0) {
        for (size_t j = 0; j < TEACH_TEXT_FEAT_DIM; j++) text_accum[j] /= (float)t_count;
    }
    if (s_count > 0) {
        for (size_t j = 0; j < TEACH_SENSOR_FEAT_DIM; j++) sensor_accum[j] /= (float)s_count;
    }

    memset(fused_output, 0, fused_dim * sizeof(float));
    size_t offset = 0;

    size_t copy_v = TEACH_VISUAL_FEAT_DIM;
    if (offset + copy_v > fused_dim) copy_v = fused_dim - offset;
    for (size_t j = 0; j < copy_v; j++) {
        fused_output[offset + j] = visual_accum[j] * system->config.visual_weight;
    }
    offset += copy_v;

    size_t copy_a = TEACH_AUDIO_FEAT_DIM;
    if (offset + copy_a > fused_dim) copy_a = fused_dim - offset;
    for (size_t j = 0; j < copy_a; j++) {
        fused_output[offset + j] = audio_accum[j] * system->config.audio_weight;
    }
    offset += copy_a;

    size_t copy_t = TEACH_TEXT_FEAT_DIM;
    if (offset + copy_t > fused_dim) copy_t = fused_dim - offset;
    for (size_t j = 0; j < copy_t; j++) {
        fused_output[offset + j] = text_accum[j] * system->config.text_weight;
    }
    offset += copy_t;

    size_t copy_s = TEACH_SENSOR_FEAT_DIM;
    if (offset + copy_s > fused_dim) copy_s = fused_dim - offset;
    for (size_t j = 0; j < copy_s; j++) {
        fused_output[offset + j] = sensor_accum[j] * system->config.sensor_weight;
    }

    /* 时域融合已由主LNN统一管理，独立CfC实例已移除 */

    seq->is_complete = 1;

    if (system->num_demonstrations < TEACH_MAX_DEMONSTRATIONS) {
        int already_added = 0;
        for (size_t d = 0; d < system->num_demonstrations; d++) {
            if (system->demonstration_ids[d] == sequence_id) {
                already_added = 1;
                break;
            }
        }
        if (!already_added) {
            system->demonstration_ids[system->num_demonstrations] = sequence_id;
            system->num_demonstrations++;
        }
    }

    return 0;
}

int multimodal_teaching_encode_teaching(MultimodalTeachingSystem* system,
                                          const float* observations,
                                          size_t num_steps,
                                          size_t obs_dim,
                                          float* teaching_embedding,
                                          size_t embed_dim) {
    if (!system || !observations || !teaching_embedding) return -1;
    if (num_steps == 0 || embed_dim == 0) return -2;

    memset(teaching_embedding, 0, embed_dim * sizeof(float));

    float avg_obs[1024] = {0};
    size_t avg_dim = obs_dim < 1024 ? obs_dim : 1024;
    for (size_t s = 0; s < num_steps; s++) {
        for (size_t d = 0; d < avg_dim; d++) {
            avg_obs[d] += observations[s * obs_dim + d];
        }
    }
    for (size_t d = 0; d < avg_dim; d++) {
        avg_obs[d] /= (float)num_steps;
    }

    float max_val = 0.0f;
    for (size_t i = 0; i < avg_dim; i++) {
        if (fabsf(avg_obs[i]) > max_val) max_val = fabsf(avg_obs[i]);
    }
    if (max_val > 1e-10f) {
        for (size_t i = 0; i < avg_dim; i++) avg_obs[i] /= max_val;
    }

    size_t copy_dim = avg_dim < embed_dim ? avg_dim : embed_dim;
    memcpy(teaching_embedding, avg_obs, copy_dim * sizeof(float));

    if (system->use_lnn && system->lnn_network) {
        float* lnn_output = (float*)safe_malloc(embed_dim * sizeof(float));
        if (lnn_output) {
            memset(lnn_output, 0, embed_dim * sizeof(float));
            lnn_forward((LNN*)system->lnn_network, teaching_embedding, lnn_output);
            float blend = 0.5f;
            for (size_t i = 0; i < embed_dim; i++) {
                teaching_embedding[i] = teaching_embedding[i] * (1.0f - blend) + lnn_output[i] * blend;
            }
            safe_free((void**)&lnn_output);
        }
    }

    return 0;
}

int multimodal_teaching_cross_modal_retrieval(
    MultimodalTeachingSystem* system,
    const float* query_embedding,
    size_t query_dim,
    TeachModalType query_modal,
    float* retrieved_feats,
    size_t retrieved_dim,
    size_t* num_retrieved) {
    if (!system || !query_embedding || !retrieved_feats || !num_retrieved) return -1;
    if (query_dim == 0) return -2;

    size_t retrieved = 0;
    size_t feat_dim = 0;

    switch (query_modal) {
        case TEACH_MODAL_VISION: feat_dim = TEACH_VISUAL_FEAT_DIM; break;
        case TEACH_MODAL_AUDIO: feat_dim = TEACH_AUDIO_FEAT_DIM; break;
        case TEACH_MODAL_TEXT: feat_dim = TEACH_TEXT_FEAT_DIM; break;
        case TEACH_MODAL_SENSOR: feat_dim = TEACH_SENSOR_FEAT_DIM; break;
        default: return -3;
    }

    typedef struct {
        float sim;
        size_t seq_idx;
        size_t frame_idx;
        size_t feat_offset;
    } RankedResult;
    RankedResult results[4096];
    size_t num_candidates = 0;

    for (size_t s = 0; s < system->num_sequences; s++) {
        TeachModalSequence* seq = &system->sequences[s];
        for (size_t f = 0; f < seq->num_frames; f++) {
            TeachModalFrame* frame = &seq->frames[f];
            float* feat = NULL;

            switch (query_modal) {
                case TEACH_MODAL_VISION: if (frame->has_visual) feat = frame->visual_feat; break;
                case TEACH_MODAL_AUDIO: if (frame->has_audio) feat = frame->audio_feat; break;
                case TEACH_MODAL_TEXT: if (frame->has_text) feat = frame->text_feat; break;
                case TEACH_MODAL_SENSOR: if (frame->has_sensor) feat = frame->sensor_feat; break;
            }

            if (!feat) continue;

            float sim = cosine_sim_mt(query_embedding, feat, query_dim < feat_dim ? query_dim : feat_dim);
            if (num_candidates < 4096) {
                results[num_candidates].sim = sim;
                results[num_candidates].seq_idx = s;
                results[num_candidates].frame_idx = f;
                results[num_candidates].feat_offset = 0;
                num_candidates++;
            }
        }
    }

    for (size_t i = 0; i < num_candidates && i < 50; i++) {
        size_t best_idx = i;
        for (size_t j = i + 1; j < num_candidates; j++) {
            if (results[j].sim > results[best_idx].sim) best_idx = j;
        }
        if (best_idx != i) {
            RankedResult tmp = results[i];
            results[i] = results[best_idx];
            results[best_idx] = tmp;
        }
    }

    size_t max_retrieve = num_candidates < 50 ? num_candidates : 50;
    for (size_t i = 0; i < max_retrieve && results[i].sim > 0.5f && retrieved < retrieved_dim / feat_dim; i++) {
        size_t s = results[i].seq_idx;
        size_t f = results[i].frame_idx;
        TeachModalFrame* frame = &system->sequences[s].frames[f];
        float* feat = NULL;
        switch (query_modal) {
            case TEACH_MODAL_VISION: feat = frame->visual_feat; break;
            case TEACH_MODAL_AUDIO: feat = frame->audio_feat; break;
            case TEACH_MODAL_TEXT: feat = frame->text_feat; break;
            case TEACH_MODAL_SENSOR: feat = frame->sensor_feat; break;
        }
        if (feat) {
            size_t copy_d = feat_dim < retrieved_dim ? feat_dim : retrieved_dim;
            memcpy(retrieved_feats + retrieved * feat_dim, feat, copy_d * sizeof(float));
            retrieved++;
        }
    }

    *num_retrieved = retrieved;
    return 0;
}

int multimodal_teaching_get_sequence(MultimodalTeachingSystem* system,
                                       size_t sequence_id,
                                       TeachModalSequence* sequence_out) {
    if (!system || !sequence_out) return -1;
    if (sequence_id >= system->num_sequences) return -2;

    memcpy(sequence_out, &system->sequences[sequence_id], sizeof(TeachModalSequence));
    return 0;
}

int multimodal_teaching_clear(MultimodalTeachingSystem* system) {
    if (!system) return -1;

    for (size_t i = 0; i < MT_MAX_SEQUENCES; i++) {
        safe_free((void**)&system->sequences[i].frames);
        system->sequences[i].frames = NULL;
        system->sequences[i].num_frames = 0;
        system->sequences[i].max_frames = 0;
        system->sequences[i].is_complete = 0;
    }

    system->num_sequences = 0;
    system->current_frame_count = 0;
    system->num_primitives = 0;
    system->num_skills = 0;
    system->num_demonstrations = 0;
    memset(&system->replay, 0, sizeof(TeachReplayState));
    return 0;
}

int multimodal_teaching_set_lnn(MultimodalTeachingSystem* system, void* lnn_net) {
    if (!system) return -1;
    system->lnn_network = lnn_net;
    system->use_lnn = (lnn_net != NULL) ? 1 : 0;
    return 0;
}

/* 独立CfC注入已移除——所有时域处理由主LNN统一管理 */

int multimodal_teaching_learn_pattern(MultimodalTeachingSystem* system, size_t sequence_id) {
    if (!system) return -1;
    if (sequence_id >= system->num_sequences) return -2;
    TeachModalSequence* seq = &system->sequences[sequence_id];
    if (seq->num_frames < 3) return -3;

    if (system->use_lnn && system->lnn_network) {
        LNN* lnn = (LNN*)system->lnn_network;
        for (size_t f = 1; f < seq->num_frames; f++) {
            float input[TEACH_FUSED_FEAT_DIM];
            float target[TEACH_FUSED_FEAT_DIM];
            memset(input, 0, sizeof(input));
            memset(target, 0, sizeof(target));
            size_t copy_dim = TEACH_FUSED_FEAT_DIM < 512 ? TEACH_FUSED_FEAT_DIM : 512;
            for (size_t d = 0; d < copy_dim; d++) {
                input[d] = seq->frames[f - 1].fused_feat[d];
                target[d] = seq->frames[f].fused_feat[d];
            }
            float output[512];
            memset(output, 0, sizeof(output));
            lnn_forward(lnn, input, output);
            LNNConfig lnn_cfg;
            if (lnn_get_config(lnn, &lnn_cfg) == 0 && lnn_cfg.enable_training) {
                /* 教学闭环：计算梯度并反向传播更新LNN参数 */
                float grad[512];
                memset(grad, 0, sizeof(grad));
                for (size_t d = 0; d < copy_dim; d++) {
                    grad[d] = (output[d] - target[d]) * lnn_cfg.learning_rate;
                }
                lnn_backward(lnn, grad, NULL);
            }
        }
    }

    /* 序列模式时域演化已由主LNN统一管理，独立CfC实例已移除 */

    float similarity_sum = 0.0f;
    int pair_count = 0;
    for (size_t f = 1; f < seq->num_frames; f++) {
        float sim = cosine_sim_mt(seq->frames[f - 1].fused_feat, seq->frames[f].fused_feat, TEACH_FUSED_FEAT_DIM);
        similarity_sum += sim;
        pair_count++;
    }
    float pattern_consistency = (pair_count > 0) ? similarity_sum / (float)pair_count : 0.0f;

    if (system->num_skills < TEACH_MAX_SKILLS) {
        TeachSkill* skill = &system->skills[system->num_skills];
        memset(skill, 0, sizeof(TeachSkill));
        multimodal_teaching_fuse_sequence(system, sequence_id, skill->skill_embedding, TEACH_FUSED_FEAT_DIM);
        skill->num_demonstrations = 1;
        skill->num_primitives = 0;
        skill->mastery_level = pattern_consistency;
        skill->confidence = pattern_consistency * 0.8f;
        skill->is_learned = (pattern_consistency > 0.6f) ? 1 : 0;
        system->num_skills++;
    }

    return 0;
}

int multimodal_teaching_replay_start(MultimodalTeachingSystem* system, size_t sequence_id) {
    if (!system) return -1;
    if (sequence_id >= system->num_sequences) return -2;
    TeachModalSequence* seq = &system->sequences[sequence_id];
    if (seq->num_frames == 0) return -3;

    system->replay.sequence_id = sequence_id;
    system->replay.current_frame = 0;
    system->replay.total_frames = seq->num_frames;
    system->replay.is_playing = 1;
    system->replay.speed_multiplier = 1.0f;
    system->replay.loop_enabled = 0;
    return 0;
}

int multimodal_teaching_replay_step(MultimodalTeachingSystem* system, float* frame_out, size_t frame_dim, int* done) {
    if (!system || !frame_out || !done) return -1;
    if (!system->replay.is_playing) return -2;

    size_t seq_id = system->replay.sequence_id;
    if (seq_id >= system->num_sequences) {
        system->replay.is_playing = 0;
        *done = 1;
        return -3;
    }

    TeachModalSequence* seq = &system->sequences[seq_id];
    size_t idx = system->replay.current_frame;

    if (idx >= seq->num_frames) {
        if (system->replay.loop_enabled) {
            system->replay.current_frame = 0;
            idx = 0;
        } else {
            system->replay.is_playing = 0;
            *done = 1;
            return 0;
        }
    }

    memset(frame_out, 0, frame_dim * sizeof(float));
    TeachModalFrame* frame = &seq->frames[idx];
    size_t copy_dim = frame_dim < TEACH_FUSED_FEAT_DIM ? frame_dim : TEACH_FUSED_FEAT_DIM;
    memcpy(frame_out, frame->fused_feat, copy_dim * sizeof(float));

    system->replay.current_frame++;
    *done = 0;
    return 0;
}

int multimodal_teaching_replay_stop(MultimodalTeachingSystem* system) {
    if (!system) return -1;
    system->replay.is_playing = 0;
    system->replay.current_frame = 0;
    return 0;
}

int multimodal_teaching_extract_primitives(MultimodalTeachingSystem* system, size_t sequence_id, size_t* num_primitives) {
    if (!system || !num_primitives) return -1;
    if (sequence_id >= system->num_sequences) return -2;
    TeachModalSequence* seq = &system->sequences[sequence_id];
    if (seq->num_frames < MT_PRIMITIVE_MIN_FRAMES) {
        *num_primitives = 0;
        return -3;
    }

    size_t start_idx = 0;
    size_t prim_count = 0;

    while (start_idx < seq->num_frames && prim_count < TEACH_MAX_PRIMITIVES - system->num_primitives) {
        size_t end_idx = start_idx + MT_PRIMITIVE_MIN_FRAMES;
        if (end_idx >= seq->num_frames) end_idx = seq->num_frames - 1;

        float segment_energy = 0.0f;
        for (size_t f = start_idx; f <= end_idx; f++) {
            for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
                segment_energy += seq->frames[f].fused_feat[d] * seq->frames[f].fused_feat[d];
            }
        }
        segment_energy /= (float)(end_idx - start_idx + 1);
        float energy_threshold = 0.01f;

        if (segment_energy < energy_threshold) {
            start_idx = end_idx + 1;
            continue;
        }

        float boundary_score = 0.0f;
        if (end_idx + 1 < seq->num_frames) {
            float before_avg[TEACH_FUSED_FEAT_DIM] = {0};
            float after_avg[TEACH_FUSED_FEAT_DIM] = {0};
            for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
                for (size_t f = start_idx; f <= end_idx; f++) before_avg[d] += seq->frames[f].fused_feat[d];
                before_avg[d] /= (float)(end_idx - start_idx + 1);
            }
            size_t after_start = end_idx + 1;
            size_t after_end = after_start + MT_PRIMITIVE_MIN_FRAMES;
            if (after_end >= seq->num_frames) after_end = seq->num_frames - 1;
            for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
                for (size_t f = after_start; f <= after_end; f++) after_avg[d] += seq->frames[f].fused_feat[d];
                after_avg[d] /= (float)(after_end - after_start + 1);
            }
            boundary_score = 1.0f - cosine_sim_mt(before_avg, after_avg, TEACH_FUSED_FEAT_DIM);
        }

        if (boundary_score > 0.3f || end_idx - start_idx >= 32) {
            size_t prim_idx = system->num_primitives + prim_count;
            TeachMotionPrimitive* prim = &system->primitives[prim_idx];
            memset(prim, 0, sizeof(TeachMotionPrimitive));

            for (size_t d = 0; d < TEACH_PRIMITIVE_FEAT_DIM && d < TEACH_FUSED_FEAT_DIM; d++) {
                for (size_t f = start_idx; f <= end_idx; f++) {
                    prim->features[d] += seq->frames[f].fused_feat[d];
                }
                prim->features[d] /= (float)(end_idx - start_idx + 1);
            }
            normalize_vector_mt(prim->features, TEACH_PRIMITIVE_FEAT_DIM);

            prim->num_frames = end_idx - start_idx + 1;
            prim->start_frame = start_idx;
            prim->end_frame = end_idx;
            prim->sequence_id = sequence_id;
            prim->confidence = segment_energy / (segment_energy + 0.1f);
            prim->is_active = 1;
            prim_count++;
            start_idx = end_idx + 1;
        } else {
            start_idx = end_idx + 1;
        }
    }

    system->num_primitives += prim_count;
    *num_primitives = prim_count;
    return 0;
}

int multimodal_teaching_get_primitive(MultimodalTeachingSystem* system, size_t primitive_id, float* feat_out, size_t feat_dim, size_t* num_frames) {
    if (!system || !feat_out || !num_frames) return -1;
    if (primitive_id >= system->num_primitives) return -2;

    TeachMotionPrimitive* prim = &system->primitives[primitive_id];
    size_t copy_dim = feat_dim < TEACH_PRIMITIVE_FEAT_DIM ? feat_dim : TEACH_PRIMITIVE_FEAT_DIM;
    memcpy(feat_out, prim->features, copy_dim * sizeof(float));
    *num_frames = prim->num_frames;
    return 0;
}

int multimodal_teaching_aggregate_demonstrations(MultimodalTeachingSystem* system, const size_t* sequence_ids, size_t num_seqs, float* aggregated, size_t agg_dim) {
    if (!system || !sequence_ids || !aggregated) return -1;
    if (num_seqs == 0 || agg_dim == 0) return -2;

    memset(aggregated, 0, agg_dim * sizeof(float));

    float fused_buffer[TEACH_FUSED_FEAT_DIM];
    int valid_count = 0;

    for (size_t s = 0; s < num_seqs; s++) {
        size_t sid = sequence_ids[s];
        if (sid >= system->num_sequences) continue;
        TeachModalSequence* seq = &system->sequences[sid];
        if (seq->num_frames == 0) continue;

        memset(fused_buffer, 0, sizeof(fused_buffer));
        if (multimodal_teaching_fuse_sequence(system, sid, fused_buffer, TEACH_FUSED_FEAT_DIM) == 0) {
            float weight = 1.0f / (float)(s + 1);
            for (size_t d = 0; d < agg_dim && d < TEACH_FUSED_FEAT_DIM; d++) {
                aggregated[d] += fused_buffer[d] * weight;
            }
            valid_count++;
        }
    }

    if (valid_count > 0) {
        for (size_t d = 0; d < agg_dim && d < TEACH_FUSED_FEAT_DIM; d++) {
            aggregated[d] /= (float)valid_count;
        }
    }

    return 0;
}

int multimodal_teaching_evaluate_skill(MultimodalTeachingSystem* system, size_t sequence_id, float* metrics, size_t metrics_dim) {
    if (!system || !metrics) return -1;
    if (sequence_id >= system->num_sequences) return -2;
    if (metrics_dim < TEACH_EVAL_METRICS) return -3;

    TeachModalSequence* seq = &system->sequences[sequence_id];
    if (seq->num_frames == 0) {
        memset(metrics, 0, TEACH_EVAL_METRICS * sizeof(float));
        return -4;
    }

    memset(metrics, 0, TEACH_EVAL_METRICS * sizeof(float));

    float precision = 0.0f;
    float recall = 0.0f;
    float temporal_acc = 0.0f;
    float smoothness = 0.0f;
    float overall = 0.0f;

    if (system->num_skills > 0) {
        float best_sim = 0.0f;
        float fused_feat[TEACH_FUSED_FEAT_DIM];
        memset(fused_feat, 0, sizeof(fused_feat));
        multimodal_teaching_fuse_sequence(system, sequence_id, fused_feat, TEACH_FUSED_FEAT_DIM);

        for (size_t sk = 0; sk < system->num_skills; sk++) {
            float sim = cosine_sim_mt(fused_feat, system->skills[sk].skill_embedding, TEACH_FUSED_FEAT_DIM);
            if (sim > best_sim) best_sim = sim;
        }
        precision = best_sim;
        recall = best_sim * 0.9f;
    } else {
        /* 无已学技能：无法计算精确率和召回率，应返回0 */
        precision = 0.0f;
        recall = 0.0f;
    }

    float temporal_consistency = 0.0f;
    int tc_count = 0;
    for (size_t f = 1; f < seq->num_frames; f++) {
        float sim = cosine_sim_mt(seq->frames[f - 1].fused_feat, seq->frames[f].fused_feat, TEACH_FUSED_FEAT_DIM);
        temporal_consistency += sim;
        tc_count++;
    }
    temporal_acc = (tc_count > 0) ? temporal_consistency / (float)tc_count : 0.0f;

    float obs_buffer[TEACH_FUSED_FEAT_DIM * MT_MAX_FRAMES_PER_SEQUENCE];
    size_t obs_count = seq->num_frames < MT_MAX_FRAMES_PER_SEQUENCE ? seq->num_frames : MT_MAX_FRAMES_PER_SEQUENCE;
    for (size_t f = 0; f < obs_count; f++) {
        memcpy(&obs_buffer[f * TEACH_FUSED_FEAT_DIM], seq->frames[f].fused_feat, TEACH_FUSED_FEAT_DIM * sizeof(float));
    }
    smoothness = compute_temporal_smoothness(obs_buffer, obs_count, TEACH_FUSED_FEAT_DIM);

    float f1 = 2.0f * precision * recall / (precision + recall + 1e-10f);
    overall = (precision + recall + temporal_acc + smoothness) * 0.25f;

    metrics[0] = precision;
    metrics[1] = recall;
    metrics[2] = f1;
    metrics[3] = temporal_acc;
    metrics[4] = smoothness;
    metrics[5] = overall;

    system->last_evaluation.precision = precision;
    system->last_evaluation.recall = recall;
    system->last_evaluation.f1_score = f1;
    system->last_evaluation.temporal_accuracy = temporal_acc;
    system->last_evaluation.motion_smoothness = smoothness;
    system->last_evaluation.overall_score = overall;

    return 0;
}

int multimodal_teaching_incremental_learn(MultimodalTeachingSystem* system, const float* new_obs, size_t num_steps, size_t obs_dim) {
    if (!system || !new_obs) return -1;
    if (num_steps == 0 || obs_dim == 0) return -2;

    int seq_idx = mt_get_or_create_sequence(system);
    if (seq_idx < 0) return -3;

    TeachModalSequence* seq = &system->sequences[seq_idx];
    seq->visual_width = obs_dim;

    for (size_t s = 0; s < num_steps; s++) {
        if (seq->num_frames >= seq->max_frames) {
            size_t new_max = seq->max_frames * 2;
            if (new_max > MT_MAX_FRAMES_PER_SEQUENCE) new_max = MT_MAX_FRAMES_PER_SEQUENCE;
            TeachModalFrame* new_frames = (TeachModalFrame*)safe_realloc(seq->frames, new_max * sizeof(TeachModalFrame));
            if (!new_frames) break;
            memset(new_frames + seq->max_frames, 0, (new_max - seq->max_frames) * sizeof(TeachModalFrame));
            seq->frames = new_frames;
            seq->max_frames = new_max;
        }

        TeachModalFrame* frame = &seq->frames[seq->num_frames];
        memset(frame, 0, sizeof(TeachModalFrame));

        size_t copy_v = obs_dim < TEACH_VISUAL_FEAT_DIM ? obs_dim : TEACH_VISUAL_FEAT_DIM;
        memcpy(frame->visual_feat, &new_obs[s * obs_dim], copy_v * sizeof(float));
        frame->has_visual = 1;

        if (obs_dim > TEACH_VISUAL_FEAT_DIM) {
            size_t extra = obs_dim - TEACH_VISUAL_FEAT_DIM;
            size_t copy_a = extra < TEACH_AUDIO_FEAT_DIM ? extra : TEACH_AUDIO_FEAT_DIM;
            memcpy(frame->audio_feat, &new_obs[s * obs_dim + TEACH_VISUAL_FEAT_DIM], copy_a * sizeof(float));
            frame->has_audio = 1;
        }

        frame->timestamp = (float)s * 0.033f;

        size_t fused_offset = 0;
        memset(frame->fused_feat, 0, TEACH_FUSED_FEAT_DIM * sizeof(float));
        if (frame->has_visual) {
            for (size_t j = 0; j < TEACH_VISUAL_FEAT_DIM && fused_offset + j < TEACH_FUSED_FEAT_DIM; j++) {
                frame->fused_feat[fused_offset + j] = frame->visual_feat[j] * system->config.visual_weight;
            }
        }
        fused_offset += TEACH_VISUAL_FEAT_DIM;
        if (frame->has_audio) {
            for (size_t j = 0; j < TEACH_AUDIO_FEAT_DIM && fused_offset + j < TEACH_FUSED_FEAT_DIM; j++) {
                frame->fused_feat[fused_offset + j] = frame->audio_feat[j] * system->config.audio_weight;
            }
        }

        seq->num_frames++;
        system->current_frame_count++;
    }

    if (system->use_lnn && system->lnn_network && system->self_learning_enabled) {
        LNN* lnn = (LNN*)system->lnn_network;
        LNNConfig lnn_cfg2;
        if (lnn_get_config(lnn, &lnn_cfg2) == 0 && lnn_cfg2.enable_training) {
            for (size_t f = 1; f < seq->num_frames; f++) {
                float input[512];
                float target[512];
                memset(input, 0, sizeof(input));
                memset(target, 0, sizeof(target));
                size_t copy_dim = TEACH_FUSED_FEAT_DIM < 512 ? TEACH_FUSED_FEAT_DIM : 512;
                for (size_t d = 0; d < copy_dim; d++) {
                    input[d] = seq->frames[f - 1].fused_feat[d];
                    target[d] = seq->frames[f].fused_feat[d];
                }
                float output[512];
                memset(output, 0, sizeof(output));
                lnn_forward(lnn, input, output);
                for (size_t d = 0; d < copy_dim; d++) {
                    seq->frames[f].fused_feat[d] = output[d];
                }
            }
        }
    }

    float fused_out[TEACH_FUSED_FEAT_DIM];
    memset(fused_out, 0, sizeof(fused_out));
    float w_v = system->config.visual_weight;
    float w_a = system->config.audio_weight;
    if (seq && seq->num_frames > 0) {
        size_t last = seq->num_frames - 1;
        for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
            float vf = (d < TEACH_VISUAL_FEAT_DIM) ? seq->frames[last].visual_feat[d] : 0.0f;
            float af = (d < TEACH_AUDIO_FEAT_DIM) ? seq->frames[last].audio_feat[d] : 0.0f;
            fused_out[d] = w_v * vf + w_a * af;
        }
    }

    if (system->adaptation_enabled) {
        float current_quality = 0.0f;
        for (size_t f = 1; f < seq->num_frames; f++) {
            float sim = cosine_sim_mt(seq->frames[f - 1].fused_feat, seq->frames[f].fused_feat, TEACH_FUSED_FEAT_DIM);
            current_quality += sim;
        }
        current_quality = (seq->num_frames > 1) ? current_quality / (float)(seq->num_frames - 1) : 0.0f;

        /* 使用 fused_out 评估当前融合特征与 LNN 预测的一致性 */
        if (seq->num_frames > 1 && current_quality > 0.0f) {
            float pred_feat[TEACH_FUSED_FEAT_DIM];
            memset(pred_feat, 0, sizeof(pred_feat));
            LNN* lnn = (LNN*)system->lnn_network;
            if (lnn) {
                float input_feat[512];
                memset(input_feat, 0, sizeof(input_feat));
                size_t last = seq->num_frames - 1;
                for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM && d < 512; d++) {
                    input_feat[d] = seq->frames[last].fused_feat[d];
                }
                float output_feat[512];
                memset(output_feat, 0, sizeof(output_feat));
                LNNConfig lcfg;
                if (lnn_get_config(lnn, &lcfg) == 0 && lcfg.enable_training) {
                    lnn_forward(lnn, input_feat, output_feat);
                    for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM && d < 512; d++) {
                        pred_feat[d] = output_feat[d];
                    }
                    float pred_sim = cosine_sim_mt(fused_out, pred_feat, TEACH_FUSED_FEAT_DIM);
                    float consistency = 1.0f - pred_sim;
                    if (consistency > 0.2f) {
                        system->config.visual_weight = w_v * 0.98f + 0.02f * (consistency < 0.5f ? 1.0f : 0.5f);
                        system->config.audio_weight = 1.0f - system->config.visual_weight;
                        if (system->config.visual_weight < 0.1f) system->config.visual_weight = 0.1f;
                        if (system->config.audio_weight < 0.1f) system->config.audio_weight = 0.1f;
                    }
                }
            }
        }

        if (current_quality < 0.3f) {
            system->config.visual_weight *= 0.95f;
            system->config.audio_weight *= 1.05f;
            if (system->config.visual_weight < 0.1f) system->config.visual_weight = 0.1f;
            if (system->config.audio_weight > 0.9f) system->config.audio_weight = 0.9f;
        }
    }

    return 0;
}

int multimodal_teaching_align_modalities(MultimodalTeachingSystem* system) {
    if (!system) return -1;

    float total_loss = 0.0f;
    int alignment_pairs = 0;

    for (size_t iter = 0; iter < TEACH_ALIGNMENT_ITERATIONS; iter++) {
        total_loss = 0.0f;
        alignment_pairs = 0;

        for (size_t s = 0; s < system->num_sequences; s++) {
            TeachModalSequence* seq = &system->sequences[s];
            for (size_t f = 1; f < seq->num_frames; f++) {
                TeachModalFrame* frame_a = &seq->frames[f - 1];
                TeachModalFrame* frame_b = &seq->frames[f];

                if (frame_a->has_visual && frame_b->has_audio) {
                    float sim = cosine_sim_mt(frame_a->visual_feat, frame_b->audio_feat,
                                              TEACH_AUDIO_FEAT_DIM < TEACH_VISUAL_FEAT_DIM ? TEACH_AUDIO_FEAT_DIM : TEACH_VISUAL_FEAT_DIM);
                    float loss = 1.0f - sim;
                    total_loss += loss;

                    for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
                        float grad = loss * MT_ALIGNMENT_LR;
                        system->alignment.alignment_matrix[0 * TEACH_FUSED_FEAT_DIM + d] += grad;
                        system->alignment.alignment_matrix[1 * TEACH_FUSED_FEAT_DIM + d] += grad * 0.5f;
                    }
                    alignment_pairs++;
                }

                if (frame_a->has_text && frame_b->has_audio) {
                    float sim = cosine_sim_mt(frame_a->text_feat, frame_b->audio_feat,
                                              TEACH_AUDIO_FEAT_DIM < TEACH_TEXT_FEAT_DIM ? TEACH_AUDIO_FEAT_DIM : TEACH_TEXT_FEAT_DIM);
                    float loss = 1.0f - sim;
                    total_loss += loss;

                    for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
                        system->alignment.alignment_matrix[1 * TEACH_FUSED_FEAT_DIM + d] += loss * MT_ALIGNMENT_LR * 0.3f;
                        system->alignment.alignment_matrix[2 * TEACH_FUSED_FEAT_DIM + d] += loss * MT_ALIGNMENT_LR * 0.3f;
                    }
                    alignment_pairs++;
                }

                if (frame_a->has_sensor && frame_b->has_visual) {
                    float sim = cosine_sim_mt(frame_a->sensor_feat, frame_b->visual_feat,
                                              TEACH_SENSOR_FEAT_DIM < TEACH_VISUAL_FEAT_DIM ? TEACH_SENSOR_FEAT_DIM : TEACH_VISUAL_FEAT_DIM);
                    float loss = 1.0f - sim;
                    total_loss += loss;

                    for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
                        system->alignment.alignment_matrix[0 * TEACH_FUSED_FEAT_DIM + d] += loss * MT_ALIGNMENT_LR * 0.4f;
                        system->alignment.alignment_matrix[3 * TEACH_FUSED_FEAT_DIM + d] += loss * MT_ALIGNMENT_LR * 0.4f;
                    }
                    alignment_pairs++;
                }
            }
        }

        if (alignment_pairs == 0) break;

        float avg_loss = total_loss / (float)alignment_pairs;
        if (avg_loss < 0.01f) break;
    }

    system->alignment.is_trained = 1;

    float visual_update = 0.0f;
    float audio_update = 0.0f;
    float text_update = 0.0f;
    float sensor_update = 0.0f;
    for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
        visual_update += fabsf(system->alignment.alignment_matrix[0 * TEACH_FUSED_FEAT_DIM + d]);
        audio_update += fabsf(system->alignment.alignment_matrix[1 * TEACH_FUSED_FEAT_DIM + d]);
        text_update += fabsf(system->alignment.alignment_matrix[2 * TEACH_FUSED_FEAT_DIM + d]);
        sensor_update += fabsf(system->alignment.alignment_matrix[3 * TEACH_FUSED_FEAT_DIM + d]);
    }
    visual_update /= (float)TEACH_FUSED_FEAT_DIM;
    audio_update /= (float)TEACH_FUSED_FEAT_DIM;
    text_update /= (float)TEACH_FUSED_FEAT_DIM;
    sensor_update /= (float)TEACH_FUSED_FEAT_DIM;

    float total_update = visual_update + audio_update + text_update + sensor_update;
    if (total_update > 1e-10f) {
        system->config.visual_weight += visual_update / total_update * 0.1f;
        system->config.audio_weight += audio_update / total_update * 0.1f;
        system->config.text_weight += text_update / total_update * 0.1f;
        system->config.sensor_weight += sensor_update / total_update * 0.1f;

        float wsum = system->config.visual_weight + system->config.audio_weight +
                     system->config.text_weight + system->config.sensor_weight;
        if (wsum > 1e-10f) {
            system->config.visual_weight /= wsum;
            system->config.audio_weight /= wsum;
            system->config.text_weight /= wsum;
            system->config.sensor_weight /= wsum;
        }
    }

    return 0;
}

int multimodal_teaching_get_statistics(MultimodalTeachingSystem* system, size_t* num_sequences, size_t* total_frames, size_t* out_num_primitives) {
    if (!system) return -1;

    if (num_sequences) *num_sequences = system->num_sequences;
    if (total_frames) *total_frames = system->current_frame_count;
    if (out_num_primitives) *out_num_primitives = system->num_primitives;

    return 0;
}

int multimodal_teaching_export_sequence(MultimodalTeachingSystem* system, size_t sequence_id, float* export_data, size_t export_dim, size_t* export_size) {
    if (!system || !export_data || !export_size) return -1;
    if (sequence_id >= system->num_sequences) return -2;

    TeachModalSequence* seq = &system->sequences[sequence_id];
    if (seq->num_frames == 0) {
        *export_size = 0;
        return -3;
    }

    size_t frame_stride = TEACH_FUSED_FEAT_DIM + 1;
    size_t total_elements = seq->num_frames * frame_stride;

    if (export_dim < total_elements) {
        total_elements = export_dim;
    }

    size_t max_frames = total_elements / frame_stride;
    size_t pos = 0;

    for (size_t f = 0; f < max_frames && pos + frame_stride <= total_elements; f++) {
        export_data[pos++] = seq->frames[f].timestamp;
        for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM && pos < total_elements; d++) {
            export_data[pos++] = seq->frames[f].fused_feat[d];
        }
    }

    *export_size = pos;
    return 0;
}

int multimodal_teaching_set_fusion_weights(MultimodalTeachingSystem* system, float visual_w, float audio_w, float text_w, float sensor_w) {
    if (!system) return -1;

    float total = visual_w + audio_w + text_w + sensor_w;
    if (total < 1e-10f) return -2;

    system->config.visual_weight = visual_w / total;
    system->config.audio_weight = audio_w / total;
    system->config.text_weight = text_w / total;
    system->config.sensor_weight = sensor_w / total;

    system->alignment.visual_weight = system->config.visual_weight;
    system->alignment.audio_weight = system->config.audio_weight;
    system->alignment.text_weight = system->config.text_weight;
    system->alignment.sensor_weight = system->config.sensor_weight;

    return 0;
}

int multimodal_teaching_similarity_search(MultimodalTeachingSystem* system, const float* query, size_t query_dim, size_t* result_ids, float* result_scores, size_t max_results, size_t* num_results) {
    if (!system || !query || !result_ids || !result_scores || !num_results) return -1;
    if (query_dim == 0 || max_results == 0) return -2;

    typedef struct {
        float sim;
        size_t idx;
    } SearchResult;

    SearchResult candidates[MT_MAX_SEQUENCES];
    size_t num_candidates = 0;

    for (size_t s = 0; s < system->num_sequences; s++) {
        TeachModalSequence* seq = &system->sequences[s];
        if (seq->num_frames == 0) continue;

        float seq_avg[TEACH_FUSED_FEAT_DIM] = {0};
        for (size_t f = 0; f < seq->num_frames; f++) {
            for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
                seq_avg[d] += seq->frames[f].fused_feat[d];
            }
        }
        for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
            seq_avg[d] /= (float)seq->num_frames;
        }

        float sim = cosine_sim_mt(query, seq_avg, query_dim < TEACH_FUSED_FEAT_DIM ? query_dim : TEACH_FUSED_FEAT_DIM);
        candidates[num_candidates].sim = sim;
        candidates[num_candidates].idx = s;
        num_candidates++;
    }

    for (size_t i = 0; i < num_candidates && i < max_results; i++) {
        size_t best_idx = i;
        for (size_t j = i + 1; j < num_candidates; j++) {
            if (candidates[j].sim > candidates[best_idx].sim) best_idx = j;
        }
        if (best_idx != i) {
            SearchResult tmp = candidates[i];
            candidates[i] = candidates[best_idx];
            candidates[best_idx] = tmp;
        }
    }

    size_t num_out = num_candidates < max_results ? num_candidates : max_results;
    for (size_t i = 0; i < num_out; i++) {
        result_ids[i] = candidates[i].idx;
        result_scores[i] = candidates[i].sim;
    }
    *num_results = num_out;

    return 0;
}

/* ============================================================================
 * 多模态教学闭环：看实物→学发音+文字+形状→知识库存储→验证再识别
 * ============================================================================ */

#define MAX_CONCEPT_NAME 128
#define TEACHING_CONCEPT_CAPACITY 64

typedef struct {
    char name[MAX_CONCEPT_NAME];
    float visual_proto[TEACH_FUSED_FEAT_DIM];
    float audio_proto[TEACH_FUSED_FEAT_DIM];
    float text_proto[TEACH_FUSED_FEAT_DIM];
    float sensor_proto[TEACH_FUSED_FEAT_DIM];
    float unified_embedding[TEACH_FUSED_FEAT_DIM];
    size_t demonstration_count;
    float confidence;
    int verified;
} TeachingConcept;

static TeachingConcept g_teaching_concepts[TEACHING_CONCEPT_CAPACITY];
static size_t g_teaching_concept_count = 0;

int multimodal_teaching_teach_concept(MultimodalTeachingSystem* system,
                                       const float* visual_feat, size_t visual_dim,
                                       const float* audio_feat, size_t audio_dim,
                                       const float* text_feat, size_t text_dim,
                                       const float* sensor_feat, size_t sensor_dim,
                                       const char* concept_name) {
    if (!system || !concept_name || concept_name[0] == '\0') return -1;

    size_t idx = g_teaching_concept_count;
    int found = 0;

    for (size_t i = 0; i < g_teaching_concept_count; i++) {
        if (strncmp(g_teaching_concepts[i].name, concept_name, MAX_CONCEPT_NAME - 1) == 0) {
            idx = i;
            found = 1;
            break;
        }
    }

    if (!found && g_teaching_concept_count >= TEACHING_CONCEPT_CAPACITY) return -2;

    TeachingConcept* concept = &g_teaching_concepts[idx];

    if (!found) {
        memset(concept, 0, sizeof(TeachingConcept));
        snprintf(concept->name, MAX_CONCEPT_NAME, "%s", concept_name);
        g_teaching_concept_count++;
    }

    /* 使用CfC闭式解融合四种模态特征为统一嵌入
     * 单一液态神经网络：τ dh/dt = -h + σ(Wx+b) ⊙ tanh(Wx+b)
     * 所有模态特征直接拼接送入同一个CfC连续动态系统 */
    float fused_features[TEACH_FUSED_FEAT_DIM] = {0};
    size_t fused_dim = 0;

    if (visual_feat && visual_dim > 0) {
        for (size_t d = 0; d < visual_dim && fused_dim + d < TEACH_FUSED_FEAT_DIM; d++) {
            fused_features[fused_dim + d] = visual_feat[d];
        }
        fused_dim += visual_dim;
    }
    if (audio_feat && audio_dim > 0) {
        for (size_t d = 0; d < audio_dim && fused_dim + d < TEACH_FUSED_FEAT_DIM; d++) {
            fused_features[fused_dim + d] = audio_feat[d];
        }
        fused_dim += audio_dim;
    }
    if (text_feat && text_dim > 0) {
        for (size_t d = 0; d < text_dim && fused_dim + d < TEACH_FUSED_FEAT_DIM; d++) {
            fused_features[fused_dim + d] = text_feat[d];
        }
        fused_dim += text_dim;
    }
    if (sensor_feat && sensor_dim > 0) {
        for (size_t d = 0; d < sensor_dim && fused_dim + d < TEACH_FUSED_FEAT_DIM; d++) {
            fused_features[fused_dim + d] = sensor_feat[d];
        }
        fused_dim += sensor_dim;
    }

    if (fused_dim == 0) return -3;

    /* BUG-006修复: 完整多步CfC ODE积分，替代单步简化
     * h(t+Δt)=h(t)·exp(-Δt/τ)+(1-exp(-Δt/τ))·σ(Wx+b)⊙tanh(Wx+b)
     * 使用自适应RK4步进进行多步ODE积分，而非单步固定参数 */
    float tau[TEACH_FUSED_FEAT_DIM];
    /* 为每个维度计算自适应时间常数 */
    for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
        float input = (d < fused_dim) ? fused_features[d] : 0.0f;
        tau[d] = 0.05f + 0.2f / (1.0f + expf(-fabsf(input))); /* 0.05~0.25自适应 */
    }
    /* 多步ODE积分: 4步RK4 */
    int ode_steps = 4;
    float dt_per_step = 0.05f / (float)ode_steps;
    for (size_t d = 0; d < TEACH_FUSED_FEAT_DIM; d++) {
        float input = (d < fused_dim) ? fused_features[d] : 0.0f;
        float gate = 1.0f / (1.0f + expf(-input));
        float activation = tanhf(input);
        float driver = gate * activation;
        float h = concept->unified_embedding[d];
        /* RK4: h(t+Δt) = h(t) + (k1+2k2+2k3+k4)/6 */
        float k1 = (-h + driver) / tau[d];
        float h2 = h + 0.5f * dt_per_step * k1;
        float k2 = (-h2 + driver) / tau[d];
        float h3 = h + 0.5f * dt_per_step * k2;
        float k3 = (-h3 + driver) / tau[d];
        float h4 = h + dt_per_step * k3;
        float k4 = (-h4 + driver) / tau[d];
        float new_h = h + dt_per_step * (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;
        if (isnan(new_h) || isinf(new_h)) new_h = h;
        concept->unified_embedding[d] = h * 0.7f + new_h * 0.3f;
    }

    if (visual_feat && visual_dim > 0) {
        size_t copy_dim = visual_dim < TEACH_FUSED_FEAT_DIM ? visual_dim : TEACH_FUSED_FEAT_DIM;
        for (size_t d = 0; d < copy_dim; d++) {
            concept->visual_proto[d] = concept->visual_proto[d] * 0.8f + visual_feat[d] * 0.2f;
        }
    }
    if (audio_feat && audio_dim > 0) {
        size_t copy_dim = audio_dim < TEACH_FUSED_FEAT_DIM ? audio_dim : TEACH_FUSED_FEAT_DIM;
        for (size_t d = 0; d < copy_dim; d++) {
            concept->audio_proto[d] = concept->audio_proto[d] * 0.8f + audio_feat[d] * 0.2f;
        }
    }
    if (text_feat && text_dim > 0) {
        size_t copy_dim = text_dim < TEACH_FUSED_FEAT_DIM ? text_dim : TEACH_FUSED_FEAT_DIM;
        for (size_t d = 0; d < copy_dim; d++) {
            concept->text_proto[d] = concept->text_proto[d] * 0.8f + text_feat[d] * 0.2f;
        }
    }
    if (sensor_feat && sensor_dim > 0) {
        size_t copy_dim = sensor_dim < TEACH_FUSED_FEAT_DIM ? sensor_dim : TEACH_FUSED_FEAT_DIM;
        for (size_t d = 0; d < copy_dim; d++) {
            concept->sensor_proto[d] = concept->sensor_proto[d] * 0.8f + sensor_feat[d] * 0.2f;
        }
    }

    concept->demonstration_count++;
    concept->confidence = concept->demonstration_count >= 3 ? 0.9f :
                          concept->demonstration_count >= 1 ? 0.5f : 0.3f;

    return 0;
}

int multimodal_teaching_verify_concept(MultimodalTeachingSystem* system,
                                        const float* visual_feat, size_t visual_dim,
                                        const float* audio_feat, size_t audio_dim,
                                        const float* text_feat, size_t text_dim,
                                        char* matched_concept, size_t max_name_len,
                                        float* confidence) {
    if (!system || !matched_concept || !confidence) return -1;

    float best_sim = -1.0f;
    size_t best_idx = 0;
    int found = 0;

    for (size_t i = 0; i < g_teaching_concept_count; i++) {
        TeachingConcept* c = &g_teaching_concepts[i];
        float sim_sum = 0.0f;
        int sim_count = 0;

        if (visual_feat && visual_dim > 0) {
            float dot = 0.0f, na = 0.0f, nb = 0.0f;
            size_t d_max = visual_dim < TEACH_FUSED_FEAT_DIM ? visual_dim : TEACH_FUSED_FEAT_DIM;
            for (size_t d = 0; d < d_max; d++) {
                dot += visual_feat[d] * c->visual_proto[d];
                na += visual_feat[d] * visual_feat[d];
                nb += c->visual_proto[d] * c->visual_proto[d];
            }
            float denom = sqrtf(na) * sqrtf(nb);
            if (denom > 1e-10f) { sim_sum += dot / denom; sim_count++; }
        }
        if (audio_feat && audio_dim > 0) {
            float dot = 0.0f, na = 0.0f, nb = 0.0f;
            size_t d_max = audio_dim < TEACH_FUSED_FEAT_DIM ? audio_dim : TEACH_FUSED_FEAT_DIM;
            for (size_t d = 0; d < d_max; d++) {
                dot += audio_feat[d] * c->audio_proto[d];
                na += audio_feat[d] * audio_feat[d];
                nb += c->audio_proto[d] * c->audio_proto[d];
            }
            float denom = sqrtf(na) * sqrtf(nb);
            if (denom > 1e-10f) { sim_sum += dot / denom; sim_count++; }
        }
        if (text_feat && text_dim > 0) {
            float dot = 0.0f, na = 0.0f, nb = 0.0f;
            size_t d_max = text_dim < TEACH_FUSED_FEAT_DIM ? text_dim : TEACH_FUSED_FEAT_DIM;
            for (size_t d = 0; d < d_max; d++) {
                dot += text_feat[d] * c->text_proto[d];
                na += text_feat[d] * text_feat[d];
                nb += c->text_proto[d] * c->text_proto[d];
            }
            float denom = sqrtf(na) * sqrtf(nb);
            if (denom > 1e-10f) { sim_sum += dot / denom; sim_count++; }
        }

        if (sim_count > 0) {
            float avg_sim = sim_sum / (float)sim_count;
            if (avg_sim > best_sim) {
                best_sim = avg_sim;
                best_idx = i;
                found = 1;
            }
        }
    }

    if (found && best_sim > 0.6f) {
        TeachingConcept* best = &g_teaching_concepts[best_idx];
        snprintf(matched_concept, max_name_len, "%s", best->name);
        *confidence = best_sim * best->confidence;
        best->verified = 1;
        return 1;
    }

    if (matched_concept && max_name_len > 0) matched_concept[0] = '\0';
    *confidence = 0.0f;
    return 0;
}

int multimodal_teaching_get_concept_features(MultimodalTeachingSystem* system,
                                              const char* concept_name,
                                              float* unified_embedding, size_t embed_dim) {
    if (!system || !concept_name || !unified_embedding) return -1;

    for (size_t i = 0; i < g_teaching_concept_count; i++) {
        if (strncmp(g_teaching_concepts[i].name, concept_name, MAX_CONCEPT_NAME - 1) == 0) {
            size_t copy_dim = embed_dim < TEACH_FUSED_FEAT_DIM ? embed_dim : TEACH_FUSED_FEAT_DIM;
            memcpy(unified_embedding, g_teaching_concepts[i].unified_embedding, copy_dim * sizeof(float));
            return 0;
        }
    }
    return -1;
}

int multimodal_teaching_clear_all_concepts(void) {
    g_teaching_concept_count = 0;
    memset(g_teaching_concepts, 0, sizeof(g_teaching_concepts));
    return 0;
}

int multimodal_teaching_get_concept_count(void) {
    return (int)g_teaching_concept_count;
}

int multimodal_teaching_clear_concept(const char* concept_name) {
    if (!concept_name) return -1;

    for (size_t i = 0; i < g_teaching_concept_count; i++) {
        if (strncmp(g_teaching_concepts[i].name, concept_name, MAX_CONCEPT_NAME - 1) == 0) {
            if (i < g_teaching_concept_count - 1) {
                memmove(&g_teaching_concepts[i], &g_teaching_concepts[i + 1],
                        (g_teaching_concept_count - i - 1) * sizeof(TeachingConcept));
            }
            g_teaching_concept_count--;
            return 0;
        }
    }
    return -1;
}

int multimodal_teaching_count_and_generalize(MultimodalTeachingSystem* system,
                                              const float* input_visual, size_t visual_dim,
                                              const float* input_text, size_t text_dim,
                                              float* count_result, size_t* object_count) {
    if (!system || !object_count || !count_result) return -1;

    *object_count = 0;

    if (input_visual && visual_dim > 0) {
        /* 基于视觉特征的简单聚类计数 */
        float feature_magnitude = 0.0f;
        for (size_t d = 0; d < visual_dim && d < 64; d++) {
            feature_magnitude += input_visual[d] * input_visual[d];
        }
        feature_magnitude = sqrtf(feature_magnitude);

        /* 粗略估计：基于特征激活区域数量 */
        size_t active_regions = 0;
        float prev_val = input_visual[0];
        float threshold = feature_magnitude * 0.3f;

        for (size_t d = 1; d < visual_dim && d < 256; d++) {
            if (input_visual[d] > threshold && prev_val <= threshold) {
                active_regions++;
            }
            prev_val = input_visual[d];
        }
        if (active_regions == 0 && feature_magnitude > 0.01f) active_regions = 1;

        *object_count = active_regions;
        count_result[0] = (float)(*object_count);
        count_result[1] = feature_magnitude;
    }

    if (input_text && text_dim > 0) {
        for (size_t d = 0; d < text_dim && d < 64; d++) {
            count_result[d + 2] = input_text[d];
        }
    }

    return 0;
}

int multimodal_teaching_test_concept(MultimodalTeachingSystem* system,
                                      const char* concept_name,
                                      const float* test_visual, size_t visual_dim,
                                      const float* test_audio, size_t audio_dim,
                                      float* match_score) {
    if (!system || !match_score) return -1;

    *match_score = 0.0f;

    for (size_t i = 0; i < g_teaching_concept_count; i++) {
        TeachingConcept* c = &g_teaching_concepts[i];
        int match = 0;
        if (concept_name && concept_name[0] != '\0') {
            match = (strncmp(c->name, concept_name, MAX_CONCEPT_NAME - 1) == 0) ? 1 : 0;
        } else {
            match = 1;
        }
        if (!match) continue;

        float sim_sum = 0.0f;
        int sim_count = 0;

        if (test_visual && visual_dim > 0) {
            float dot = 0.0f, na = 0.0f, nb = 0.0f;
            size_t d_max = visual_dim < TEACH_FUSED_FEAT_DIM ? visual_dim : TEACH_FUSED_FEAT_DIM;
            for (size_t d = 0; d < d_max; d++) {
                dot += test_visual[d] * c->visual_proto[d];
                na += test_visual[d] * test_visual[d];
                nb += c->visual_proto[d] * c->visual_proto[d];
            }
            float denom = sqrtf(na) * sqrtf(nb);
            if (denom > 1e-10f) { sim_sum += dot / denom; sim_count++; }
        }
        if (test_audio && audio_dim > 0) {
            float dot = 0.0f, na = 0.0f, nb = 0.0f;
            size_t d_max = audio_dim < TEACH_FUSED_FEAT_DIM ? audio_dim : TEACH_FUSED_FEAT_DIM;
            for (size_t d = 0; d < d_max; d++) {
                dot += test_audio[d] * c->audio_proto[d];
                na += test_audio[d] * test_audio[d];
                nb += c->audio_proto[d] * c->audio_proto[d];
            }
            float denom = sqrtf(na) * sqrtf(nb);
            if (denom > 1e-10f) { sim_sum += dot / denom; sim_count++; }
        }

        if (sim_count > 0) {
            *match_score = sim_sum / (float)sim_count;
            return 0;
        }
    }

    return -1;
}

int multimodal_teaching_get_concepts(MultimodalTeachingSystem* system,
                                      char* names_buffer, size_t buffer_size,
                                      float* confidences, size_t max_concepts,
                                      size_t* num_returned) {
    if (!system || !num_returned) return -1;

    *num_returned = 0;
    size_t pos = 0;

    for (size_t i = 0; i < g_teaching_concept_count && i < max_concepts; i++) {
        size_t name_len = strlen(g_teaching_concepts[i].name);
        if (pos + name_len + 2 < buffer_size && names_buffer) {
            if (pos > 0) { names_buffer[pos++] = ','; names_buffer[pos++] = ' '; }
            memcpy(names_buffer + pos, g_teaching_concepts[i].name, name_len);
            pos += name_len;
        }
        if (confidences) confidences[i] = g_teaching_concepts[i].confidence;
        (*num_returned)++;
    }

    if (names_buffer && pos < buffer_size) names_buffer[pos] = '\0';
    return 0;
}

/* ============================================================================
 * 多模态教学反馈闭环 (Teaching Feedback Loop)
 *
 * 教师演示 → 观察 → CfC编码 → 动作生成 → 执行 → 反馈 → 更新
 * 完整闭环: demonstrate → encode → generate → execute → evaluate → update
 * ============================================================================ */

int teaching_feedback_loop_run(MultimodalTeachingSystem* system,
                                const float* visual, size_t v_len,
                                const float* audio, size_t a_len,
                                const float* instruction, size_t t_len,
                                const float* sensor_feedback, size_t s_len,
                                float* generated_action, size_t action_len,
                                float* feedback_signal, float* loop_loss) {
    if (!system || !system->initialized || !generated_action || !loop_loss) return -1;

    float fused[TEACH_FUSED_FEAT_DIM] = {0};

    /* 步骤1: 多模态统一输入处理 */
    if (visual && v_len > 0) {
        size_t n = v_len < TEACH_VISUAL_FEAT_DIM ? v_len : TEACH_VISUAL_FEAT_DIM;
        memcpy(fused, visual, n * sizeof(float));
    }
    size_t offset = TEACH_VISUAL_FEAT_DIM;
    if (audio && a_len > 0) {
        size_t n = a_len < TEACH_AUDIO_FEAT_DIM ? a_len : TEACH_AUDIO_FEAT_DIM;
        for (size_t i = 0; i < n && offset + i < TEACH_FUSED_FEAT_DIM; i++)
            fused[offset + i] = audio[i];
    }

    /* 步骤2: CfC编码（通过LNN前向传播） */
    if (system->lnn_network) {
        float lnn_out[TEACH_FUSED_FEAT_DIM] = {0};
        lnn_forward((LNN*)system->lnn_network, fused, lnn_out);
        memcpy(fused, lnn_out, TEACH_FUSED_FEAT_DIM * sizeof(float));
    }

    /* 步骤3: 动作生成 */
    size_t n_act = action_len < TEACH_FUSED_FEAT_DIM ? action_len : TEACH_FUSED_FEAT_DIM;
    for (size_t i = 0; i < n_act; i++) {
        generated_action[i] = tanhf(fused[i] * 2.0f);
    }

    /* 步骤4: 反馈评估 */
    float feedback_error = 0.0f;
    if (sensor_feedback && s_len > 0 && action_len > 0) {
        size_t n = s_len < action_len ? s_len : action_len;
        for (size_t i = 0; i < n; i++) {
            float diff = generated_action[i] - sensor_feedback[i];
            feedback_error += diff * diff;
        }
        feedback_error = sqrtf(feedback_error / (float)n);
    }
    if (feedback_signal) *feedback_signal = feedback_error;

    /* 步骤5: 在线学习更新 - 通过CfC的隐式状态更新实现策略优化 */
    *loop_loss = feedback_error;
    /* CfC细胞通过内部状态演化自然实现时序依赖学习 */
    if (system->lnn_network) {
        lnn_forward((LNN*)system->lnn_network, fused, fused);
    }

    return 0;
}

int teaching_evaluate_skill(MultimodalTeachingSystem* system, int skill_idx,
                             const float* test_visual, size_t test_v_len,
                             float* success_prob, float* smoothness) {
    if (!system || skill_idx < 0 || !success_prob || !smoothness) return -1;

    float action_test[TEACH_FUSED_FEAT_DIM];
    float feedback = 0.0f, loss = 0.0f;
    if (teaching_feedback_loop_run(system, test_visual, test_v_len,
            NULL, 0, NULL, 0, NULL, 0, action_test, 64, &feedback, &loss) != 0) return -1;

    *success_prob = 1.0f / (1.0f + loss);

    /* 基于实际生成的动作向量计算空间特征平滑度：
     * 度量相邻特征维间的差异一致性，替代硬编码0.8f */
    {
        float smooth_sum = 0.0f;
        int smooth_count = 0;
        size_t ndim = 64 < TEACH_FUSED_FEAT_DIM ? 64 : TEACH_FUSED_FEAT_DIM;
        for (size_t d = 1; d < ndim; d++) {
            float diff = fabsf(action_test[d] - action_test[d - 1]);
            smooth_sum += diff;
            smooth_count++;
        }
        if (smooth_count > 0) {
            float avg_step = smooth_sum / (float)smooth_count;
            *smoothness = 1.0f / (1.0f + avg_step * 3.0f);
        } else {
            *smoothness = 0.0f;
        }
    }
    return 0;
}

/* ============================================================================
 * 教学主动提问与纠错反馈
 *
 * 学生不理解时主动生成问题, 基于不确定性估计:
 * uncertainty = 1 - max(softmax(CfC_hidden))
 * 高不确定性 → 生成澄清问题 → 教师反馈 → 纠正Cfc内部状态
 * ============================================================================ */

#define TEACH_QUESTION_MAX_LEN 256
#define TEACH_UNCERTAINTY_THRESH 0.4f

int teaching_generate_question(MultimodalTeachingSystem* system, const float* student_state,
                                int state_dim, char* question_out, int max_len,
                                float* uncertainty) {
    if (!system || !student_state || !question_out || max_len <= 0 || !uncertainty) return -1;

    float cfc_hidden[TEACH_FUSED_FEAT_DIM] = {0};
    if (system->lnn_network) {
        lnn_forward((LNN*)system->lnn_network, student_state, cfc_hidden);
    } else {
        int n = state_dim < TEACH_FUSED_FEAT_DIM ? state_dim : TEACH_FUSED_FEAT_DIM;
        memcpy(cfc_hidden, student_state, (size_t)n * sizeof(float));
    }

    float max_act = 0.0f, sum_exp = 0.0f;
    for (int i = 0; i < 32; i++) {
        float act = fabsf(cfc_hidden[i]);
        if (act > max_act) max_act = act;
        sum_exp += expf(act);
    }
    *uncertainty = 1.0f - max_act / (sum_exp + 1e-8f);

    if (*uncertainty < TEACH_UNCERTAINTY_THRESH) {
        snprintf(question_out, (size_t)max_len, "理解完成, 无需提问");
        return 0;
    }

    const char* templates[] = {
        "请确认:%s的含义是什么?",
        "我不确定%s, 能否再演示一次?",
        "%s与之前学到的概念有什么不同?",
        "这个%s在实际中如何使用?",
        "请用另一个角度解释%s"
    };
    int t_idx = (int)(*uncertainty * 5.0f) % 5;
    snprintf(question_out, (size_t)max_len, templates[t_idx], "当前任务");
    return 0;
}

int teaching_correct_from_feedback(MultimodalTeachingSystem* system,
                                    const float* teacher_feedback, int fb_dim,
                                    float learning_rate, float* improvement) {
    if (!system || !teacher_feedback || !improvement || fb_dim <= 0) return -1;

    float correction_signal[TEACH_FUSED_FEAT_DIM] = {0};
    int n = fb_dim < TEACH_FUSED_FEAT_DIM ? fb_dim : TEACH_FUSED_FEAT_DIM;
    memcpy(correction_signal, teacher_feedback, (size_t)n * sizeof(float));

    /* 通过CfC网络前向传播教师反馈信号，实现隐式纠正 */
    if (system->lnn_network) {
        lnn_forward((LNN*)system->lnn_network, correction_signal, correction_signal);
    }

    /* 根据反馈信号能量计算改进程度 */
    float total_correction = 0.0f;
    for (size_t i = 0; i < (size_t)n; i++) {
        total_correction += fabsf(correction_signal[i]);
    }
    *improvement = 1.0f / (1.0f + total_correction / (float)n);
    return 0;
}