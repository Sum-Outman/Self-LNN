#define SELFLNN_IMPLEMENTATION 1
#include "selflnn/learning/teach_by_showing.h"
#include "selflnn/core/tensor.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc.h"
#include "selflnn/multimodal/multimodal_unified_input.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/memory_utils.h"

struct TeachSystem {
    TeachDemoSet demos;
    TeachTrainConfig config;
    LNN* policy_net;
    LNN* task_encoder;
    float* task_embedding_cache;
    size_t obs_dim;
    size_t act_dim;
    size_t hidden_dim;
    int is_trained;
    float* attention_weights;
    size_t* reproduction_counts;
    float* task_confidence;
    int initialized;
    float cached_accuracy;

    TeachConcept concepts[TEACH_MAX_CONCEPTS];
    size_t num_concepts;
    LNN* visual_encoder;
    LNN* audio_encoder;
    LNN* tactile_encoder;
    LNN* text_encoder;
    LNN* concept_fuser;
    float* scratch_buffer;
};

static float gaussian_rand(float mean, float std) {
    float u1 = secure_random_float();
    float u2 = secure_random_float();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return mean + std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static float cosine_similarity_teach(const float* a, const float* b, size_t dim) {
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

/* P2-038修复: n-gram哈希嵌入——将文本转为有语义区分度的嵌入向量
 * 使用2-gram滑动窗口 + djb2哈希 + L2归一化
 * 相比简单的ASCII/255映射，不同文本可产生有实质区分度的嵌入 */
static void teach_text_hash_embed(const char* text, float* embedding, int dim) {
    if (!text || !embedding || dim <= 0) return;
    size_t len = strlen(text);
    if (len < 2) {
        /* 文本太短，使用单字符哈希填充 */
        for (int i = 0; i < dim; i++) {
            unsigned long h = 5381;
            h = ((h << 5) + h) + (unsigned char)text[0];
            h = ((h << 5) + h) + ((unsigned int)i * 2654435761u);
            embedding[i] = (float)(h % 1000007) / 500003.5f - 1.0f;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            /* 在文本上滑动选择2-gram窗口 */
            int pos = (i * 127 + 31) % (int)(len - 1);
            unsigned long h = 5381;
            h = ((h << 5) + h) + (unsigned char)text[pos];
            h = ((h << 5) + h) + (unsigned char)text[pos + 1];
            h = ((h << 5) + h) + ((unsigned int)i * 2654435761u);
            embedding[i] = (float)(h % 1000007) / 500003.5f - 1.0f;
        }
    }
    /* L2归一化 */
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += embedding[i] * embedding[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (int i = 0; i < dim; i++) embedding[i] /= norm;
    }
}

TeachSystem* teach_system_create(size_t obs_dim, size_t act_dim,
                                  TeachTrainConfig config) {
    TeachSystem* system = (TeachSystem*)safe_calloc(1, sizeof(TeachSystem));
    if (!system) return NULL;

    system->config = config;
    system->obs_dim = obs_dim;
    system->act_dim = act_dim;
    system->hidden_dim = 512;
    system->is_trained = 0;
    system->initialized = 1;
    system->cached_accuracy = 0.0f;

    system->demos.num_demos = 0;
    system->demos.max_steps = TEACH_MAX_STEPS_PER_DEMO;
    system->demos.obs_dim = obs_dim;
    system->demos.act_dim = act_dim;

    system->demos.observations = (float*)safe_calloc(TEACH_MAX_DEMOS * TEACH_MAX_STEPS_PER_DEMO * obs_dim, sizeof(float));
    system->demos.actions = (float*)safe_calloc(TEACH_MAX_DEMOS * TEACH_MAX_STEPS_PER_DEMO * act_dim, sizeof(float));
    system->demos.rewards = (float*)safe_calloc(TEACH_MAX_DEMOS * TEACH_MAX_STEPS_PER_DEMO, sizeof(float));
    system->demos.task_embeddings = (float*)safe_calloc(TEACH_MAX_DEMOS * TEACH_MODALITY_DIM, sizeof(float));
    system->demos.labels = (char**)safe_calloc(TEACH_MAX_DEMOS, sizeof(char*));
    system->demos.task_types = (TeachTaskType*)safe_calloc(TEACH_MAX_DEMOS, sizeof(TeachTaskType));
    system->demos.timestamps = (float*)safe_calloc(TEACH_MAX_DEMOS, sizeof(float));
    system->demos.confidence_scores = (float*)safe_calloc(TEACH_MAX_DEMOS, sizeof(float));
    system->demos.trajectory_lengths = (size_t*)safe_calloc(TEACH_MAX_DEMOS, sizeof(size_t));

    system->task_embedding_cache = (float*)safe_calloc(TEACH_MAX_TASKS * TEACH_MODALITY_DIM, sizeof(float));
    system->attention_weights = (float*)safe_calloc(TEACH_MAX_DEMOS * TEACH_MAX_STEPS_PER_DEMO, sizeof(float));
    system->reproduction_counts = (size_t*)safe_calloc(TEACH_MAX_TASKS, sizeof(size_t));
    system->task_confidence = (float*)safe_calloc(TEACH_MAX_TASKS, sizeof(float));

    for (size_t i = 0; i < TEACH_MAX_DEMOS; i++) {
        system->demos.labels[i] = (char*)safe_calloc(TEACH_LABEL_LEN, 1);
    }

    LNNConfig lnn_cfg = {0};
    lnn_cfg.input_size = (int)(obs_dim + act_dim + 64);
    lnn_cfg.hidden_size = (int)system->hidden_dim;
    lnn_cfg.output_size = (int)act_dim;
    lnn_cfg.num_layers = 1;
    lnn_cfg.enable_training = 1;
    lnn_cfg.learning_rate = config.learning_rate;
    system->policy_net = lnn_create(&lnn_cfg);
    if (!system->policy_net) {
        teach_system_destroy(system);
        return NULL;
    }

    LNNConfig enc_cfg = {0};
    enc_cfg.input_size = (int)obs_dim;
    enc_cfg.hidden_size = 256;
    enc_cfg.output_size = 128;
    enc_cfg.num_layers = 1;
    system->task_encoder = lnn_create(&enc_cfg);
    if (!system->task_encoder) {
        teach_system_destroy(system);
        return NULL;
    }

    system->num_concepts = 0;
    memset(system->concepts, 0, sizeof(system->concepts));

    LNNConfig vis_cfg = {0};
    vis_cfg.input_size = TEACH_VISUAL_DIM;
    vis_cfg.hidden_size = 512;
    vis_cfg.output_size = TEACH_VISUAL_DIM;
    vis_cfg.num_layers = 1;
    system->visual_encoder = lnn_create(&vis_cfg);

    LNNConfig aud_cfg = {0};
    aud_cfg.input_size = TEACH_AUDIO_DIM;
    aud_cfg.hidden_size = 256;
    aud_cfg.output_size = TEACH_AUDIO_DIM;
    aud_cfg.num_layers = 1;
    system->audio_encoder = lnn_create(&aud_cfg);

    LNNConfig tac_cfg = {0};
    tac_cfg.input_size = TEACH_TACTILE_DIM;
    tac_cfg.hidden_size = 256;
    tac_cfg.output_size = TEACH_TACTILE_DIM;
    tac_cfg.num_layers = 1;
    system->tactile_encoder = lnn_create(&tac_cfg);

    LNNConfig txt_cfg = {0};
    txt_cfg.input_size = TEACH_TEXT_DIM;
    txt_cfg.hidden_size = 256;
    txt_cfg.output_size = TEACH_TEXT_DIM;
    txt_cfg.num_layers = 1;
    system->text_encoder = lnn_create(&txt_cfg);

    LNNConfig fus_cfg = {0};
    fus_cfg.input_size = TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM + TEACH_TEXT_DIM;
    fus_cfg.hidden_size = 1024;
    fus_cfg.output_size = TEACH_UNIFIED_DIM;
    fus_cfg.num_layers = 2;
    system->concept_fuser = lnn_create(&fus_cfg);

    system->scratch_buffer = (float*)safe_calloc(TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM + TEACH_TEXT_DIM + TEACH_UNIFIED_DIM + 1024, sizeof(float));

    return system;
}

void teach_system_destroy(TeachSystem* system) {
    if (!system) return;

    for (size_t i = 0; i < TEACH_MAX_DEMOS; i++) {
        safe_free((void**)&system->demos.labels[i]);
    }
    safe_free((void**)&system->demos.labels);
    safe_free((void**)&system->demos.observations);
    safe_free((void**)&system->demos.actions);
    safe_free((void**)&system->demos.rewards);
    safe_free((void**)&system->demos.task_embeddings);
    safe_free((void**)&system->demos.task_types);
    safe_free((void**)&system->demos.timestamps);
    safe_free((void**)&system->demos.confidence_scores);
    safe_free((void**)&system->demos.trajectory_lengths);
    safe_free((void**)&system->task_embedding_cache);
    safe_free((void**)&system->attention_weights);
    safe_free((void**)&system->reproduction_counts);
    safe_free((void**)&system->task_confidence);

    if (system->policy_net) lnn_free(system->policy_net);
    if (system->task_encoder) lnn_free(system->task_encoder);
    if (system->visual_encoder) lnn_free(system->visual_encoder);
    if (system->audio_encoder) lnn_free(system->audio_encoder);
    if (system->tactile_encoder) lnn_free(system->tactile_encoder);
    if (system->text_encoder) lnn_free(system->text_encoder);
    if (system->concept_fuser) lnn_free(system->concept_fuser);
    safe_free((void**)&system->scratch_buffer);
    safe_free((void**)&system);
}

int teach_record_demonstration(TeachSystem* system,
                                const float* observations,
                                const float* actions,
                                size_t num_steps,
                                size_t obs_dim,
                                size_t act_dim,
                                const char* label,
                                TeachTaskType task_type) {
    if (!system || !observations || !actions || !label) return -1;
    if (system->demos.num_demos >= TEACH_MAX_DEMOS) return -2;
    if (num_steps == 0 || num_steps > TEACH_MAX_STEPS_PER_DEMO) return -3;
    if (obs_dim != system->obs_dim || act_dim != system->act_dim) return -4;

    size_t idx = system->demos.num_demos;

    size_t obs_offset = idx * TEACH_MAX_STEPS_PER_DEMO * obs_dim;
    size_t act_offset = idx * TEACH_MAX_STEPS_PER_DEMO * act_dim;

    memcpy(system->demos.observations + obs_offset, observations,
           num_steps * obs_dim * sizeof(float));
    memcpy(system->demos.actions + act_offset, actions,
           num_steps * act_dim * sizeof(float));

    system->demos.trajectory_lengths[idx] = num_steps;
    system->demos.task_types[idx] = task_type;
    system->demos.timestamps[idx] = (float)time(NULL);
    system->demos.confidence_scores[idx] = 1.0f;

    strncpy(system->demos.labels[idx], label, TEACH_LABEL_LEN - 1);
    system->demos.labels[idx][TEACH_LABEL_LEN - 1] = '\0';

    for (size_t s = 0; s < num_steps; s++) {
        size_t pos = idx * TEACH_MAX_STEPS_PER_DEMO + s;
        float prev_action[64] = {0};
        if (s > 0) {
            memcpy(prev_action, actions + (s - 1) * act_dim, act_dim * sizeof(float));
        }
        float dist = 0.0f;
        for (size_t d = 0; d < act_dim; d++) {
            float diff = actions[s * act_dim + d] - prev_action[d];
            dist += diff * diff;
        }
        system->demos.rewards[pos] = expf(-dist * 0.1f);
    }

    float enc_input[256] = {0};
    size_t enc_dim = obs_dim < 256 ? obs_dim : 256;
    memcpy(enc_input, observations, enc_dim * sizeof(float));
    float enc_out[128] = {0};
    lnn_forward(system->task_encoder, enc_input, enc_out);
    memcpy(system->demos.task_embeddings + idx * TEACH_MODALITY_DIM,
           enc_out, 128 * sizeof(float));

    system->demos.num_demos++;
    system->is_trained = 0;
    system->cached_accuracy = 0.0f;
    return (int)idx;
}

int teach_encode_demonstrations(TeachSystem* system,
                                 TeachEncodeType encode_type) {
    if (!system || system->demos.num_demos == 0) return -1;

    for (size_t i = 0; i < system->demos.num_demos; i++) {
        size_t len = system->demos.trajectory_lengths[i];
        float feat[128] = {0};
        size_t obs_off = i * TEACH_MAX_STEPS_PER_DEMO * system->obs_dim;

        if (encode_type == TEACH_ENCODE_LNN || encode_type == TEACH_ENCODE_HYBRID) {
            float avg_obs[256] = {0};
            size_t avg_dim = system->obs_dim < 256 ? system->obs_dim : 256;
            for (size_t s = 0; s < len; s++) {
                for (size_t d = 0; d < avg_dim; d++) {
                    avg_obs[d] += system->demos.observations[obs_off + s * system->obs_dim + d];
                }
            }
            for (size_t d = 0; d < avg_dim; d++) {
                avg_obs[d] /= (float)len;
            }
            lnn_forward(system->task_encoder, avg_obs, feat);
        }

        memcpy(system->demos.task_embeddings + i * TEACH_MODALITY_DIM,
               feat, 128 * sizeof(float));

        if (i > 0 && encode_type >= TEACH_ENCODE_HYBRID) {
            float sim = cosine_similarity_teach(
                system->demos.task_embeddings + (i - 1) * TEACH_MODALITY_DIM,
                system->demos.task_embeddings + i * TEACH_MODALITY_DIM, 128);
            if (system->demos.task_types[i] == system->demos.task_types[i - 1]) {
                system->demos.confidence_scores[i] += sim * 0.5f;
            }
        }

        size_t task_id = (size_t)system->demos.task_types[i];
        memcpy(system->task_embedding_cache + task_id * TEACH_MODALITY_DIM,
               feat, 128 * sizeof(float));
    }

    return 0;
}

int teach_train_from_demos(TeachSystem* system,
                            int (*progress_callback)(float progress,
                                                     float loss,
                                                     void* user_data),
                            void* user_data) {
    if (!system || system->demos.num_demos == 0) return -1;

    size_t total_steps = 0;
    for (size_t i = 0; i < system->demos.num_demos; i++) {
        total_steps += system->demos.trajectory_lengths[i];
    }
    if (total_steps == 0) return -2;

    size_t batch_size = system->config.batch_size;
    if (batch_size > total_steps) batch_size = total_steps;

    float* combined_input = (float*)safe_malloc(batch_size * (system->obs_dim + system->act_dim + 64) * sizeof(float));
    float* target_actions = (float*)safe_malloc(batch_size * system->act_dim * sizeof(float));
    float* bc_losses = (float*)safe_malloc(system->config.num_epochs * sizeof(float));
    if (!combined_input || !target_actions || !bc_losses) {
        safe_free((void**)&combined_input);
        safe_free((void**)&target_actions);
        safe_free((void**)&bc_losses);
        return -3;
    }

    for (size_t epoch = 0; epoch < system->config.num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        size_t batches = total_steps / batch_size;

        for (size_t b = 0; b < batches; b++) {
            size_t sampled = 0;
            for (size_t n = 0; n < batch_size && sampled < batch_size; n++) {
                size_t demo_idx = secure_random_int((uint32_t)system->demos.num_demos);
                size_t len = system->demos.trajectory_lengths[demo_idx];
                size_t step = secure_random_int((uint32_t)len);
                size_t obs_off = demo_idx * TEACH_MAX_STEPS_PER_DEMO * system->obs_dim + step * system->obs_dim;
                size_t act_off = demo_idx * TEACH_MAX_STEPS_PER_DEMO * system->act_dim + step * system->act_dim;

                float* inp = combined_input + sampled * (system->obs_dim + system->act_dim + 64);
                memcpy(inp, system->demos.observations + obs_off, system->obs_dim * sizeof(float));
                memcpy(inp + system->obs_dim, system->demos.actions + act_off, system->act_dim * sizeof(float));
                memcpy(inp + system->obs_dim + system->act_dim,
                       system->demos.task_embeddings + demo_idx * TEACH_MODALITY_DIM, 64 * sizeof(float));
                memcpy(target_actions + sampled * system->act_dim,
                       system->demos.actions + act_off, system->act_dim * sizeof(float));
                sampled++;
            }

            if (sampled == 0) continue;

            float temp_pred[256] = {0};
            float batch_loss = 0.0f;
            for (size_t n = 0; n < sampled; n++) {
                float* inp = combined_input + n * (system->obs_dim + system->act_dim + 64);
                float* tgt = target_actions + n * system->act_dim;
                lnn_forward(system->policy_net, inp, temp_pred);

                float sample_loss = 0.0f;
                lnn_backward(system->policy_net, tgt, &sample_loss);
                batch_loss += sample_loss;
            }
            batch_loss /= (float)sampled;

            epoch_loss += batch_loss;
        }

        if (batches > 0) epoch_loss /= (float)batches;
        bc_losses[epoch] = epoch_loss;

        float progress = (float)(epoch + 1) / (float)system->config.num_epochs;
        if (progress_callback) {
            int should_stop = progress_callback(progress, epoch_loss, user_data);
            if (should_stop) break;
        }
    }

    system->is_trained = 1;

    /* M-003修复: 训练完成后计算真实准确率并缓存 */
    {
        float total_accuracy = 0.0f;
        size_t eval_count = 0;
        for (size_t i = 0; i < system->demos.num_demos; i++) {
            size_t demo_len = system->demos.trajectory_lengths[i];
            if (demo_len == 0) continue;
            size_t act_off = i * TEACH_MAX_STEPS_PER_DEMO * system->act_dim;
            float accuracy = 0.0f, similarity = 0.0f;
            int ret = teach_evaluate_reproduction(system,
                system->demos.labels[i],
                system->demos.actions + act_off,
                demo_len, system->act_dim,
                &accuracy, &similarity);
            if (ret == 0) {
                total_accuracy += accuracy;
                eval_count++;
            }
        }
        if (eval_count > 0) {
            system->cached_accuracy = total_accuracy / (float)eval_count;
        } else {
            system->cached_accuracy = 0.0f;
        }
    }

    safe_free((void**)&combined_input);
    safe_free((void**)&target_actions);
    safe_free((void**)&bc_losses);
    return 0;
}

int teach_reproduce_task(TeachSystem* system,
                          const char* task_label,
                          float* current_obs,
                          size_t obs_dim,
                          float* actions_out,
                          size_t* num_steps_out,
                          float temperature) {
    if (!system || !task_label || !current_obs || !actions_out || !num_steps_out) return -1;
    if (!system->is_trained) return -2;

    size_t best_demo = TEACH_MAX_DEMOS;
    for (size_t i = 0; i < system->demos.num_demos; i++) {
        if (strcmp(system->demos.labels[i], task_label) == 0) {
            best_demo = i;
            break;
        }
    }
    if (best_demo >= TEACH_MAX_DEMOS) return -3;

    size_t demo_len = system->demos.trajectory_lengths[best_demo];
    size_t max_steps = demo_len > TEACH_MAX_STEPS_PER_DEMO ? TEACH_MAX_STEPS_PER_DEMO : demo_len;

    float obs_buf[256] = {0};
    size_t copy_obs = obs_dim < 256 ? obs_dim : 256;
    memcpy(obs_buf, current_obs, copy_obs * sizeof(float));

    for (size_t step = 0; step < max_steps; step++) {
        float inp[256 + 64 + 64] = {0};
        memcpy(inp, obs_buf, copy_obs * sizeof(float));
        float* act_src = system->demos.actions + best_demo * TEACH_MAX_STEPS_PER_DEMO * system->act_dim + step * system->act_dim;
        memcpy(inp + copy_obs, act_src, system->act_dim * sizeof(float));
        memcpy(inp + copy_obs + system->act_dim,
               system->demos.task_embeddings + best_demo * TEACH_MODALITY_DIM, 64 * sizeof(float));

        float action[64] = {0};
        lnn_forward(system->policy_net, inp, action);

        for (size_t d = 0; d < system->act_dim && d < 64; d++) {
            actions_out[step * system->act_dim + d] = action[d];
            if (temperature > 0.0f) {
                actions_out[step * system->act_dim + d] += gaussian_rand(0.0f, temperature);
            }
        }
    }

    *num_steps_out = max_steps;
    return 0;
}

int teach_recognize_task(TeachSystem* system,
                          const float* observations,
                          size_t num_steps,
                          size_t obs_dim,
                          char* task_label_out,
                          size_t label_buf_size,
                          float* confidence_out) {
    if (!system || !observations || !task_label_out || !confidence_out) return -1;
    if (system->demos.num_demos == 0) return -2;

    float avg_obs[256] = {0};
    size_t avg_dim = obs_dim < 256 ? obs_dim : 256;
    for (size_t s = 0; s < num_steps; s++) {
        for (size_t d = 0; d < avg_dim; d++) {
            avg_obs[d] += observations[s * obs_dim + d];
        }
    }
    for (size_t d = 0; d < avg_dim; d++) {
        avg_obs[d] /= (float)num_steps;
    }

    float query_embed[128] = {0};
    lnn_forward(system->task_encoder, avg_obs, query_embed);

    float best_sim = -1.0f;
    size_t best_idx = 0;
    for (size_t i = 0; i < system->demos.num_demos; i++) {
        float sim = cosine_similarity_teach(
            query_embed,
            system->demos.task_embeddings + i * TEACH_MODALITY_DIM, 128);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }

    strncpy(task_label_out, system->demos.labels[best_idx], label_buf_size - 1);
    task_label_out[label_buf_size - 1] = '\0';
    *confidence_out = best_sim;

    return 0;
}

void teach_get_stats(TeachSystem* system, TeachSystemStats* stats) {
    if (!system || !stats) return;
    memset(stats, 0, sizeof(TeachSystemStats));

    stats->total_demonstrations = system->demos.num_demos;
    stats->num_tasks_learned = 0;

    TeachTaskType last_type = TEACH_TASK_MAX;
    for (size_t i = 0; i < system->demos.num_demos; i++) {
        if (system->demos.task_types[i] != last_type) {
            stats->num_tasks_learned++;
            last_type = system->demos.task_types[i];
        }
    }

    /* M-003修复: 优先使用缓存的真实准确率，否则使用confidence_scores均值作为近似 */
    if (system->cached_accuracy > 0.0f) {
        stats->reproduction_accuracy = system->cached_accuracy;
    } else if (system->is_trained) {
        float total_conf = 0.0f;
        for (size_t i = 0; i < system->demos.num_demos; i++) {
            total_conf += system->demos.confidence_scores[i];
        }
        if (system->demos.num_demos > 0) {
            stats->reproduction_accuracy = total_conf / (float)system->demos.num_demos;
        } else {
            stats->reproduction_accuracy = 0.0f;
        }
    } else {
        stats->reproduction_accuracy = 0.0f;
    }

    stats->task_recognition_confidence = 0.0f;

    float total_conf = 0.0f;
    for (size_t i = 0; i < system->demos.num_demos; i++) {
        total_conf += system->demos.confidence_scores[i];
    }
    if (system->demos.num_demos > 0) {
        stats->task_recognition_confidence = total_conf / (float)system->demos.num_demos;
    }
}

int teach_export_demos(TeachSystem* system, const char* file_path) {
    if (!system || !file_path) return -1;

    FILE* fp = fopen(file_path, "wb");
    if (!fp) return -2;

    fwrite(&system->demos.num_demos, sizeof(size_t), 1, fp);
    fwrite(&system->obs_dim, sizeof(size_t), 1, fp);
    fwrite(&system->act_dim, sizeof(size_t), 1, fp);

    for (size_t i = 0; i < system->demos.num_demos; i++) {
        fwrite(&system->demos.trajectory_lengths[i], sizeof(size_t), 1, fp);
        fwrite(&system->demos.task_types[i], sizeof(TeachTaskType), 1, fp);
        fwrite(system->demos.labels[i], TEACH_LABEL_LEN, 1, fp);

        size_t len = system->demos.trajectory_lengths[i];
        size_t obs_off = i * TEACH_MAX_STEPS_PER_DEMO * system->obs_dim;
        size_t act_off = i * TEACH_MAX_STEPS_PER_DEMO * system->act_dim;

        fwrite(system->demos.observations + obs_off, sizeof(float), len * system->obs_dim, fp);
        fwrite(system->demos.actions + act_off, sizeof(float), len * system->act_dim, fp);
        fwrite(system->demos.task_embeddings + i * TEACH_MODALITY_DIM, sizeof(float), 128, fp);
    }

    fclose(fp);
    return 0;
}

int teach_import_demos(TeachSystem* system, const char* file_path) {
    if (!system || !file_path) return -1;

    FILE* fp = fopen(file_path, "rb");
    if (!fp) return -2;

    size_t num_import, import_obs_dim, import_act_dim;
    fread(&num_import, sizeof(size_t), 1, fp);
    fread(&import_obs_dim, sizeof(size_t), 1, fp);
    fread(&import_act_dim, sizeof(size_t), 1, fp);

    if (num_import + system->demos.num_demos > TEACH_MAX_DEMOS) {
        fclose(fp);
        return -3;
    }

    for (size_t i = 0; i < num_import; i++) {
        size_t idx = system->demos.num_demos + i;
        fread(&system->demos.trajectory_lengths[idx], sizeof(size_t), 1, fp);
        fread(&system->demos.task_types[idx], sizeof(TeachTaskType), 1, fp);
        fread(system->demos.labels[idx], TEACH_LABEL_LEN, 1, fp);

        size_t len = system->demos.trajectory_lengths[idx];
        size_t obs_off = idx * TEACH_MAX_STEPS_PER_DEMO * system->obs_dim;
        size_t act_off = idx * TEACH_MAX_STEPS_PER_DEMO * system->act_dim;

        fread(system->demos.observations + obs_off, sizeof(float), len * system->obs_dim, fp);
        fread(system->demos.actions + act_off, sizeof(float), len * system->act_dim, fp);
        fread(system->demos.task_embeddings + idx * TEACH_MODALITY_DIM, sizeof(float), 128, fp);

        system->demos.timestamps[idx] = (float)time(NULL);
        system->demos.confidence_scores[idx] = 0.8f;
    }

    system->demos.num_demos += num_import;
    system->is_trained = 0;
    system->cached_accuracy = 0.0f;
    fclose(fp);
    return 0;
}

int teach_clear_demos(TeachSystem* system, size_t keep_latest) {
    if (!system) return -1;

    if (keep_latest >= system->demos.num_demos) return 0;

    size_t to_remove = system->demos.num_demos - keep_latest;
    size_t keep_start = to_remove;

    for (size_t i = 0; i < keep_latest; i++) {
        size_t src = keep_start + i;
        safe_free((void**)&system->demos.labels[i]);
        system->demos.labels[i] = (char*)safe_calloc(TEACH_LABEL_LEN, 1);
        strncpy(system->demos.labels[i], system->demos.labels[src], TEACH_LABEL_LEN - 1);

        system->demos.trajectory_lengths[i] = system->demos.trajectory_lengths[src];
        system->demos.task_types[i] = system->demos.task_types[src];
        system->demos.timestamps[i] = system->demos.timestamps[src];
        system->demos.confidence_scores[i] = system->demos.confidence_scores[src];

        size_t len = system->demos.trajectory_lengths[src];
        size_t dst_obs = i * TEACH_MAX_STEPS_PER_DEMO * system->obs_dim;
        size_t dst_act = i * TEACH_MAX_STEPS_PER_DEMO * system->act_dim;
        size_t src_obs = src * TEACH_MAX_STEPS_PER_DEMO * system->obs_dim;
        size_t src_act = src * TEACH_MAX_STEPS_PER_DEMO * system->act_dim;

        memmove(system->demos.observations + dst_obs,
                system->demos.observations + src_obs,
                len * system->obs_dim * sizeof(float));
        memmove(system->demos.actions + dst_act,
                system->demos.actions + src_act,
                len * system->act_dim * sizeof(float));
        memcpy(system->demos.task_embeddings + i * TEACH_MODALITY_DIM,
               system->demos.task_embeddings + src * TEACH_MODALITY_DIM, 128 * sizeof(float));
    }

    system->demos.num_demos = keep_latest;
    system->is_trained = 0;
    system->cached_accuracy = 0.0f;
    return (int)to_remove;
}

int teach_evaluate_reproduction(TeachSystem* system,
                                 const char* task_label,
                                 const float* ground_truth_actions,
                                 size_t num_steps,
                                 size_t act_dim,
                                 float* accuracy_out,
                                 float* similarity_out) {
    float* demo_actions = NULL;
    if (!system || !task_label || !ground_truth_actions || !accuracy_out || !similarity_out) return -1;

    size_t best_demo = TEACH_MAX_DEMOS;
    for (size_t i = 0; i < system->demos.num_demos; i++) {
        if (strcmp(system->demos.labels[i], task_label) == 0) {
            best_demo = i;
            break;
        }
    }
    if (best_demo >= TEACH_MAX_DEMOS) return -2;

    demo_actions = (float*)safe_calloc(8192 * 64, sizeof(float));
    size_t demo_len = 0;
    size_t obs_buf_size = system->obs_dim < 256 ? system->obs_dim : 256;
    float demo_obs[256] = {0};
    size_t demo_obs_off = best_demo * TEACH_MAX_STEPS_PER_DEMO * system->obs_dim;
    memcpy(demo_obs, system->demos.observations + demo_obs_off, obs_buf_size * sizeof(float));

    teach_reproduce_task(system, task_label, demo_obs, system->obs_dim,
                         demo_actions, &demo_len, 0.0f);

    size_t eval_steps = num_steps < demo_len ? num_steps : demo_len;
    float total_mse = 0.0f;
    float total_sim = 0.0f;

    for (size_t s = 0; s < eval_steps; s++) {
        float mse = 0.0f;
        float n1 = 0.0f, n2 = 0.0f, dot = 0.0f;
        for (size_t d = 0; d < act_dim; d++) {
            float diff = demo_actions[s * act_dim + d] - ground_truth_actions[s * act_dim + d];
            mse += diff * diff;
            dot += demo_actions[s * act_dim + d] * ground_truth_actions[s * act_dim + d];
            n1 += demo_actions[s * act_dim + d] * demo_actions[s * act_dim + d];
            n2 += ground_truth_actions[s * act_dim + d] * ground_truth_actions[s * act_dim + d];
        }
        total_mse += mse / (float)act_dim;
        float denom = sqrtf(n1 * n2);
        if (denom > 1e-10f) {
            total_sim += dot / denom;
        }
    }

    if (eval_steps > 0) {
        float rmse = sqrtf(total_mse / (float)eval_steps);
        *accuracy_out = 1.0f / (1.0f + rmse);
        *similarity_out = total_sim / (float)eval_steps;
    } else {
        *accuracy_out = 0.0f;
        *similarity_out = 0.0f;
    }

    safe_free((void**)&demo_actions);
    return 0;
}

int teach_incremental_update(TeachSystem* system,
                              const float* new_observations,
                              const float* new_actions,
                              size_t num_steps,
                              size_t obs_dim,
                              size_t act_dim,
                              const char* label,
                              float learning_rate_scale) {
    if (!system || !new_observations || !new_actions || !label) return -1;
    if (!system->is_trained) return teach_record_demonstration(system,
        new_observations, new_actions, num_steps, obs_dim, act_dim, label, TEACH_TASK_CUSTOM);

    int demo_idx = teach_record_demonstration(system,
        new_observations, new_actions, num_steps, obs_dim, act_dim, label, TEACH_TASK_CUSTOM);
    if (demo_idx < 0) return demo_idx;

    float lr = system->config.learning_rate * learning_rate_scale;
    if (lr > 0.01f) lr = 0.01f;

    size_t num_mini_epochs = 5;
    for (size_t e = 0; e < num_mini_epochs; e++) {
        for (size_t s = 0; s < num_steps; s++) {
            float inp[256 + 64 + 64] = {0};
            size_t obs_off = (size_t)demo_idx * TEACH_MAX_STEPS_PER_DEMO * obs_dim + s * obs_dim;
            size_t act_off = (size_t)demo_idx * TEACH_MAX_STEPS_PER_DEMO * act_dim + s * act_dim;
            memcpy(inp, system->demos.observations + obs_off, obs_dim * sizeof(float));
            memcpy(inp + obs_dim, system->demos.actions + act_off, act_dim * sizeof(float));
            memcpy(inp + obs_dim + act_dim,
                   system->demos.task_embeddings + (size_t)demo_idx * TEACH_MODALITY_DIM, 64 * sizeof(float));

            float pred[64] = {0};
            float target[64] = {0};
            float loss_val = 0.0f;

            /* 前向传播 */
            lnn_forward(system->policy_net, inp, pred);

            /* 构建目标: 使用演示动作作为学习目标 */
            if (act_dim <= 64) {
                memcpy(target, system->demos.actions + act_off, act_dim * sizeof(float));
                /* 其余维度用预测值自身填充（无损） */
                for (size_t d = act_dim; d < 64; d++) {
                    target[d] = pred[d];
                }
            } else {
                for (size_t d = 0; d < 64; d++) {
                    target[d] = pred[d];
                }
            }

            /* 反向传播和权重更新（lnn_backward内部完成参数更新） */
            lnn_backward(system->policy_net, target, &loss_val);
        }
    }

    return 0;
}

TeachDemoSet* teach_get_demo_set(TeachSystem* system) {
    if (!system) return NULL;
    return &system->demos;
}

void teach_free_demo_set(TeachDemoSet* set) {
    if (!set) return;

    /* 释放所有演示数据 */
    safe_free((void**)&set->observations);
    safe_free((void**)&set->actions);
    safe_free((void**)&set->task_embeddings);
    safe_free((void**)&set->labels);
    safe_free((void**)&set->timestamps);

    set->num_demos = 0;
    set->max_steps = 0;
    set->obs_dim = 0;
    set->act_dim = 0;
}

/* ==================== 5大核心教学算法实现 ==================== */

static int teach_find_concept(TeachSystem* system, const char* name) {
    for (size_t i = 0; i < system->num_concepts; i++) {
        if (strcmp(system->concepts[i].concept_name, name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static void teach_normalize_vector(float* v, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; i++) norm += v[i] * v[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (size_t i = 0; i < dim; i++) v[i] /= norm;
    }
}

static void teach_weighted_fusion(float* dst, const float* src, size_t dim, float weight) {
    for (size_t i = 0; i < dim; i++) {
        dst[i] = dst[i] * (1.0f - weight) + src[i] * weight;
    }
}

int teach_bind_concept(TeachSystem* system,
                        const float* visual_feat, size_t visual_dim,
                        const float* audio_feat, size_t audio_dim,
                        const float* tactile_feat, size_t tactile_dim,
                        const char* concept_name) {
    if (!system || !concept_name) return -1;
    if (strlen(concept_name) == 0) return -2;

    if (system->num_concepts >= TEACH_MAX_CONCEPTS) return -3;

    float fused_input[TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM + TEACH_TEXT_DIM];
    memset(fused_input, 0, sizeof(fused_input));

    float* scratch = system->scratch_buffer;
    memset(scratch, 0, (TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM + TEACH_TEXT_DIM) * sizeof(float));

    size_t visual_copy = visual_dim < TEACH_VISUAL_DIM ? visual_dim : TEACH_VISUAL_DIM;
    size_t audio_copy = audio_dim < TEACH_AUDIO_DIM ? audio_dim : TEACH_AUDIO_DIM;
    size_t tactile_copy = tactile_dim < TEACH_TACTILE_DIM ? tactile_dim : TEACH_TACTILE_DIM;

    if (visual_feat && visual_dim > 0) {
        memcpy(scratch, visual_feat, visual_copy * sizeof(float));
        teach_normalize_vector(scratch, visual_copy);
        lnn_forward(system->visual_encoder, scratch, fused_input);
    }

    if (audio_feat && audio_dim > 0) {
        memset(scratch, 0, TEACH_AUDIO_DIM * sizeof(float));
        memcpy(scratch, audio_feat, audio_copy * sizeof(float));
        teach_normalize_vector(scratch, audio_copy);
        lnn_forward(system->audio_encoder, scratch, fused_input + TEACH_VISUAL_DIM);
    }

    if (tactile_feat && tactile_dim > 0) {
        memset(scratch, 0, TEACH_TACTILE_DIM * sizeof(float));
        memcpy(scratch, tactile_feat, tactile_copy * sizeof(float));
        teach_normalize_vector(scratch, tactile_copy);
        lnn_forward(system->tactile_encoder, scratch, fused_input + TEACH_VISUAL_DIM + TEACH_AUDIO_DIM);
    }

    size_t text_offset = TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM;
    memset(scratch, 0, TEACH_TEXT_DIM * sizeof(float));
    teach_text_hash_embed(concept_name, scratch, TEACH_TEXT_DIM);
    lnn_forward(system->text_encoder, scratch, fused_input + text_offset);

    float unified[TEACH_UNIFIED_DIM];
    memset(unified, 0, sizeof(unified));
    lnn_forward(system->concept_fuser, fused_input, unified);
    teach_normalize_vector(unified, TEACH_UNIFIED_DIM);

    int existing = teach_find_concept(system, concept_name);
    if (existing >= 0) {
        TeachConcept* c = &system->concepts[existing];
        if (visual_feat) {
            teach_weighted_fusion(c->visual_embedding, fused_input, TEACH_VISUAL_DIM, 0.3f);
        }
        if (audio_feat) {
            teach_weighted_fusion(c->audio_embedding, fused_input + TEACH_VISUAL_DIM, TEACH_AUDIO_DIM, 0.3f);
        }
        if (tactile_feat) {
            teach_weighted_fusion(c->tactile_embedding, fused_input + TEACH_VISUAL_DIM + TEACH_AUDIO_DIM, TEACH_TACTILE_DIM, 0.3f);
        }
        teach_weighted_fusion(c->text_embedding, fused_input + text_offset, TEACH_TEXT_DIM, 0.3f);
        teach_weighted_fusion(c->unified_embedding, unified, TEACH_UNIFIED_DIM, 0.3f);
        c->num_examples++;
        c->confidence = 1.0f - 1.0f / (float)(c->num_examples + 1);
        return existing;
    }

    size_t idx = system->num_concepts;
    TeachConcept* c = &system->concepts[idx];
    memset(c, 0, sizeof(TeachConcept));
    strncpy(c->concept_name, concept_name, TEACH_LABEL_LEN - 1);
    if (visual_feat) memcpy(c->visual_embedding, fused_input, TEACH_VISUAL_DIM * sizeof(float));
    if (audio_feat) memcpy(c->audio_embedding, fused_input + TEACH_VISUAL_DIM, TEACH_AUDIO_DIM * sizeof(float));
    if (tactile_feat) memcpy(c->tactile_embedding, fused_input + TEACH_VISUAL_DIM + TEACH_AUDIO_DIM, TEACH_TACTILE_DIM * sizeof(float));
    memcpy(c->text_embedding, fused_input + text_offset, TEACH_TEXT_DIM * sizeof(float));
    memcpy(c->unified_embedding, unified, TEACH_UNIFIED_DIM * sizeof(float));
    c->num_examples = 1;
    c->confidence = 0.5f;
    c->category = TEACH_CATEGORY_OBJECT;
    system->num_concepts++;
    return (int)idx;
}

int teach_look_and_learn(TeachSystem* system,
                          const float* visual_data, size_t width, size_t height,
                          const char* concept_name,
                          float* concept_embedding, size_t embed_dim) {
    if (!system || !visual_data || !concept_name) return -1;
    if (width == 0 || height == 0) return -2;

    float* scratch = system->scratch_buffer;
    memset(scratch, 0, TEACH_VISUAL_DIM * sizeof(float));

    float scale_x = (float)width / 32.0f;
    float scale_y = (float)height / 32.0f;
    int grid_w = 32, grid_h = 32;
    if (grid_w * grid_h > TEACH_VISUAL_DIM) {
        grid_w = 16;
        grid_h = 16;
    }

    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            int sx = (int)(x * scale_x);
            int sy = (int)(y * scale_y);
            if (sx >= (int)width) sx = (int)width - 1;
            if (sy >= (int)height) sy = (int)height - 1;
            float val = 0.0f;
            for (size_t c = 0; c < 3; c++) {
                val += visual_data[(sy * width + sx) * 3 + c];
            }
            scratch[y * grid_w + x] = val / 3.0f;
        }
    }

    float visual_feat[TEACH_VISUAL_DIM];
    memset(visual_feat, 0, sizeof(visual_feat));
    lnn_forward(system->visual_encoder, scratch, visual_feat);
    teach_normalize_vector(visual_feat, TEACH_VISUAL_DIM);

    int existing = teach_find_concept(system, concept_name);
    if (existing >= 0) {
        TeachConcept* c = &system->concepts[existing];
        teach_weighted_fusion(c->visual_embedding, visual_feat, TEACH_VISUAL_DIM, 0.2f);
        c->num_examples++;
        c->confidence = 1.0f - 1.0f / (float)(c->num_examples + 1);

        size_t copy_dim = embed_dim < TEACH_UNIFIED_DIM ? embed_dim : TEACH_UNIFIED_DIM;
        memcpy(concept_embedding, c->unified_embedding, copy_dim * sizeof(float));
        return existing;
    }

    if (system->num_concepts >= TEACH_MAX_CONCEPTS) return -3;

    size_t text_offset = TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM;
    float fused_input[TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM + TEACH_TEXT_DIM];
    memset(fused_input, 0, sizeof(fused_input));
    memcpy(fused_input, visual_feat, TEACH_VISUAL_DIM * sizeof(float));

    memset(scratch, 0, TEACH_TEXT_DIM * sizeof(float));
    teach_text_hash_embed(concept_name, scratch, TEACH_TEXT_DIM);
    lnn_forward(system->text_encoder, scratch, fused_input + text_offset);

    float unified[TEACH_UNIFIED_DIM];
    memset(unified, 0, sizeof(unified));
    lnn_forward(system->concept_fuser, fused_input, unified);
    teach_normalize_vector(unified, TEACH_UNIFIED_DIM);

    size_t idx = system->num_concepts;
    TeachConcept* c = &system->concepts[idx];
    memset(c, 0, sizeof(TeachConcept));
    strncpy(c->concept_name, concept_name, TEACH_LABEL_LEN - 1);
    memcpy(c->visual_embedding, visual_feat, TEACH_VISUAL_DIM * sizeof(float));
    memcpy(c->text_embedding, fused_input + text_offset, TEACH_TEXT_DIM * sizeof(float));
    memcpy(c->unified_embedding, unified, TEACH_UNIFIED_DIM * sizeof(float));
    c->num_examples = 1;
    c->confidence = 0.4f;
    c->category = TEACH_CATEGORY_OBJECT;
    system->num_concepts++;

    size_t copy_dim = embed_dim < TEACH_UNIFIED_DIM ? embed_dim : TEACH_UNIFIED_DIM;
    memcpy(concept_embedding, unified, copy_dim * sizeof(float));
    return (int)idx;
}

int teach_say_and_associate(TeachSystem* system,
                             const float* audio_data, size_t audio_len,
                             const char* text,
                             const float* context_visual, size_t visual_dim,
                             const char* concept_name,
                             float* association_embedding, size_t embed_dim) {
    if (!system || !concept_name) return -1;
    if (!audio_data && !text) return -2;

    float* scratch = system->scratch_buffer;
    size_t text_offset = TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM;
    float fused_input[TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM + TEACH_TEXT_DIM];
    memset(fused_input, 0, sizeof(fused_input));

    if (context_visual && visual_dim > 0) {
        size_t cv = visual_dim < TEACH_VISUAL_DIM ? visual_dim : TEACH_VISUAL_DIM;
        memcpy(fused_input, context_visual, cv * sizeof(float));
    }

    if (audio_data && audio_len > 0) {
        memset(scratch, 0, TEACH_AUDIO_DIM * sizeof(float));
        size_t ca = audio_len < TEACH_AUDIO_DIM ? audio_len : TEACH_AUDIO_DIM;
        float max_amp = 0.001f;
        for (size_t i = 0; i < audio_len; i++) {
            float amp = fabsf(audio_data[i]);
            if (amp > max_amp) max_amp = amp;
        }
        for (size_t i = 0; i < ca; i++) {
            scratch[i] = audio_data[i] / max_amp;
        }
        float aud_feat[TEACH_AUDIO_DIM];
        memset(aud_feat, 0, sizeof(aud_feat));
        lnn_forward(system->audio_encoder, scratch, aud_feat);
        memcpy(fused_input + TEACH_VISUAL_DIM, aud_feat, TEACH_AUDIO_DIM * sizeof(float));
    }

    const char* source_text = text ? text : concept_name;
    memset(scratch, 0, TEACH_TEXT_DIM * sizeof(float));
    teach_text_hash_embed(source_text, scratch, TEACH_TEXT_DIM);
    lnn_forward(system->text_encoder, scratch, fused_input + text_offset);

    float unified[TEACH_UNIFIED_DIM];
    memset(unified, 0, sizeof(unified));
    lnn_forward(system->concept_fuser, fused_input, unified);
    teach_normalize_vector(unified, TEACH_UNIFIED_DIM);

    int existing = teach_find_concept(system, concept_name);
    if (existing >= 0) {
        TeachConcept* c = &system->concepts[existing];
        if (audio_data) teach_weighted_fusion(c->audio_embedding, fused_input + TEACH_VISUAL_DIM, TEACH_AUDIO_DIM, 0.4f);
        teach_weighted_fusion(c->text_embedding, fused_input + text_offset, TEACH_TEXT_DIM, 0.4f);
        teach_weighted_fusion(c->unified_embedding, unified, TEACH_UNIFIED_DIM, 0.4f);
        c->num_examples++;
        c->confidence = 1.0f - 1.0f / (float)(c->num_examples + 1);
        size_t copy_dim = embed_dim < TEACH_UNIFIED_DIM ? embed_dim : TEACH_UNIFIED_DIM;
        memcpy(association_embedding, c->unified_embedding, copy_dim * sizeof(float));
        return existing;
    }

    if (system->num_concepts >= TEACH_MAX_CONCEPTS) return -3;

    size_t idx = system->num_concepts;
    TeachConcept* c = &system->concepts[idx];
    memset(c, 0, sizeof(TeachConcept));
    strncpy(c->concept_name, concept_name, TEACH_LABEL_LEN - 1);
    c->category = TEACH_CATEGORY_OBJECT;
    if (context_visual) memcpy(c->visual_embedding, fused_input, TEACH_VISUAL_DIM * sizeof(float));
    if (audio_data) memcpy(c->audio_embedding, fused_input + TEACH_VISUAL_DIM, TEACH_AUDIO_DIM * sizeof(float));
    memcpy(c->text_embedding, fused_input + text_offset, TEACH_TEXT_DIM * sizeof(float));
    memcpy(c->unified_embedding, unified, TEACH_UNIFIED_DIM * sizeof(float));
    c->num_examples = 1;
    c->confidence = 0.45f;
    system->num_concepts++;

    size_t copy_dim = embed_dim < TEACH_UNIFIED_DIM ? embed_dim : TEACH_UNIFIED_DIM;
    memcpy(association_embedding, unified, copy_dim * sizeof(float));
    return (int)idx;
}

int teach_touch_and_understand(TeachSystem* system,
                                const float* sensor_data, size_t sensor_dim,
                                const char* concept_name,
                                float* property_vector, size_t prop_dim) {
    if (!system || !sensor_data || !concept_name || !property_vector) return -1;
    if (sensor_dim == 0) return -2;

    float* scratch = system->scratch_buffer;
    memset(scratch, 0, TEACH_TACTILE_DIM * sizeof(float));

    size_t cs = sensor_dim < TEACH_TACTILE_DIM ? sensor_dim : TEACH_TACTILE_DIM;
    float max_val = 0.001f;
    for (size_t i = 0; i < sensor_dim; i++) {
        if (fabsf(sensor_data[i]) > max_val) max_val = fabsf(sensor_data[i]);
    }
    for (size_t i = 0; i < cs; i++) {
        scratch[i] = sensor_data[i] / max_val;
    }

    float tactile_feat[TEACH_TACTILE_DIM];
    memset(tactile_feat, 0, sizeof(tactile_feat));
    lnn_forward(system->tactile_encoder, scratch, tactile_feat);
    teach_normalize_vector(tactile_feat, TEACH_TACTILE_DIM);

    float physical_props[TEACH_MAX_PROPERTIES];
    memset(physical_props, 0, sizeof(physical_props));

    size_t num_props = prop_dim < TEACH_MAX_PROPERTIES ? prop_dim : TEACH_MAX_PROPERTIES;
    if (sensor_dim >= 6) {
        float force_mag = 0.0f;
        for (size_t i = 0; i < 3 && i < sensor_dim; i++) {
            force_mag += sensor_data[i] * sensor_data[i];
        }
        physical_props[0] = sqrtf(force_mag); /* 硬度估计 */
        physical_props[1] = sensor_data[sensor_dim > 1 ? 1 : 0]; /* 温度 */
        float variance = 0.0f;
        float mean = 0.0f;
        for (size_t i = 0; i < sensor_dim; i++) mean += sensor_data[i];
        mean /= (float)sensor_dim;
        for (size_t i = 0; i < sensor_dim; i++) variance += (sensor_data[i] - mean) * (sensor_data[i] - mean);
        physical_props[2] = sqrtf(variance / (float)sensor_dim); /* 纹理粗糙度 */
        physical_props[3] = mean; /* 平均力度/重量 */
    }

    size_t text_offset = TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM;
    float fused_input[TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM + TEACH_TEXT_DIM];
    memset(fused_input, 0, sizeof(fused_input));
    memcpy(fused_input + TEACH_VISUAL_DIM + TEACH_AUDIO_DIM, tactile_feat, TEACH_TACTILE_DIM * sizeof(float));

    memset(scratch, 0, TEACH_TEXT_DIM * sizeof(float));
    teach_text_hash_embed(concept_name, scratch, TEACH_TEXT_DIM);
    lnn_forward(system->text_encoder, scratch, fused_input + text_offset);

    float unified[TEACH_UNIFIED_DIM];
    memset(unified, 0, sizeof(unified));
    lnn_forward(system->concept_fuser, fused_input, unified);
    teach_normalize_vector(unified, TEACH_UNIFIED_DIM);

    int existing = teach_find_concept(system, concept_name);
    if (existing >= 0) {
        TeachConcept* c = &system->concepts[existing];
        teach_weighted_fusion(c->tactile_embedding, tactile_feat, TEACH_TACTILE_DIM, 0.35f);
        if (c->num_properties == 0) {
            memcpy(c->physical_properties, physical_props, num_props * sizeof(float));
            c->num_properties = (int)num_props;
        } else {
            for (size_t i = 0; i < num_props && i < (size_t)c->num_properties; i++) {
                c->physical_properties[i] = c->physical_properties[i] * 0.7f + physical_props[i] * 0.3f;
            }
        }
        c->num_examples++;
        c->confidence = 1.0f - 1.0f / (float)(c->num_examples + 1);
        c->category = TEACH_CATEGORY_PROPERTY;
        memcpy(property_vector, c->physical_properties, num_props * sizeof(float));
        return existing;
    }

    if (system->num_concepts >= TEACH_MAX_CONCEPTS) return -3;

    size_t idx = system->num_concepts;
    TeachConcept* c = &system->concepts[idx];
    memset(c, 0, sizeof(TeachConcept));
    strncpy(c->concept_name, concept_name, TEACH_LABEL_LEN - 1);
    memcpy(c->tactile_embedding, tactile_feat, TEACH_TACTILE_DIM * sizeof(float));
    memcpy(c->unified_embedding, unified, TEACH_UNIFIED_DIM * sizeof(float));
    memcpy(c->physical_properties, physical_props, num_props * sizeof(float));
    c->num_properties = (int)num_props;
    c->num_examples = 1;
    c->confidence = 0.5f;
    c->category = TEACH_CATEGORY_PROPERTY;
    system->num_concepts++;

    memcpy(property_vector, physical_props, num_props * sizeof(float));
    return (int)idx;
}

int teach_count_and_generalize(TeachSystem* system,
                                const float* visual_sequence, size_t num_frames,
                                size_t width, size_t height,
                                const char* count_concept, int count_value,
                                float* abstraction_embedding, size_t embed_dim) {
    if (!system || !visual_sequence || !count_concept || !abstraction_embedding) return -1;
    if (num_frames == 0 || count_value <= 0) return -2;

    float* scratch = system->scratch_buffer;
    float* frame_features = scratch + 1024;

    int grid_w = 16, grid_h = 16;
    int detected = 0;
    float total_energy = 0.0f;

    for (size_t f = 0; f < num_frames; f++) {
        const float* frame = visual_sequence + f * width * height * 3;
        float frame_energy = 0.0f;
        memset(frame_features, 0, TEACH_VISUAL_DIM * sizeof(float));

        for (int y = 0; y < grid_h; y++) {
            for (int x = 0; x < grid_w; x++) {
                int sx = (int)(x * width / grid_w);
                int sy = (int)(y * height / grid_h);
                if (sx >= (int)width) sx = (int)width - 1;
                if (sy >= (int)height) sy = (int)height - 1;
                float val = 0.0f;
                for (size_t c = 0; c < 3; c++) {
                    val += frame[(sy * width + sx) * 3 + c];
                }
                frame_features[y * grid_w + x] = val / 3.0f;
                frame_energy += fabsf(val);
            }
        }
        total_energy += frame_energy;

        float diff = 0.0f;
        if (f > 0) {
            const float* prev_frame = visual_sequence + (f - 1) * width * height * 3;
            for (int y = 0; y < grid_h; y++) {
                for (int x = 0; x < grid_w; x++) {
                    int sx = (int)(x * width / grid_w);
                    int sy = (int)(y * height / grid_h);
                    float curr_val = 0.0f;
                    float prev_val = 0.0f;
                    for (size_t c = 0; c < 3; c++) {
                        curr_val += frame[(sy * width + sx) * 3 + c];
                        prev_val += prev_frame[(sy * width + sx) * 3 + c];
                    }
                    diff += fabsf(curr_val / 3.0f - prev_val / 3.0f);
                }
            }
            if (diff > 10.0f) detected++;
        }
    }

    int estimated_count = detected + 1;
    if (estimated_count < 1) estimated_count = 1;

    memset(scratch, 0, TEACH_TEXT_DIM * sizeof(float));
    teach_text_hash_embed(count_concept, scratch, TEACH_TEXT_DIM);
    float text_feat[TEACH_TEXT_DIM];
    memset(text_feat, 0, sizeof(text_feat));
    lnn_forward(system->text_encoder, scratch, text_feat);

    float count_embed[TEACH_UNIFIED_DIM];
    memset(count_embed, 0, sizeof(count_embed));
    for (size_t i = 0; i < TEACH_TEXT_DIM && i < TEACH_UNIFIED_DIM; i++) {
        count_embed[i] = text_feat[i];
    }
    float count_norm = (float)count_value / 100.0f;
    if (count_norm > 1.0f) count_norm = 1.0f;
    count_embed[TEACH_UNIFIED_DIM - 1] = count_norm;
    count_embed[TEACH_UNIFIED_DIM - 2] = (float)estimated_count / 100.0f;

    teach_normalize_vector(count_embed, TEACH_UNIFIED_DIM);

    int existing = teach_find_concept(system, count_concept);
    if (existing >= 0) {
        TeachConcept* c = &system->concepts[existing];
        teach_weighted_fusion(c->unified_embedding, count_embed, TEACH_UNIFIED_DIM, 0.3f);
        c->num_examples++;
        c->confidence = 1.0f - 1.0f / (float)(c->num_examples + 1);
        c->is_abstract = 1;
        c->abstraction_level = (c->abstraction_level + (float)count_value / 10.0f) * 0.5f;
        if (c->abstraction_level > 1.0f) c->abstraction_level = 1.0f;
        size_t copy_dim = embed_dim < TEACH_UNIFIED_DIM ? embed_dim : TEACH_UNIFIED_DIM;
        memcpy(abstraction_embedding, c->unified_embedding, copy_dim * sizeof(float));
        return existing;
    }

    if (system->num_concepts >= TEACH_MAX_CONCEPTS) return -3;

    size_t idx = system->num_concepts;
    TeachConcept* c = &system->concepts[idx];
    memset(c, 0, sizeof(TeachConcept));
    strncpy(c->concept_name, count_concept, TEACH_LABEL_LEN - 1);
    memcpy(c->unified_embedding, count_embed, TEACH_UNIFIED_DIM * sizeof(float));
    c->num_examples = 1;
    c->confidence = 0.6f;
    c->category = TEACH_CATEGORY_QUANTITY;
    c->is_abstract = 1;
    c->abstraction_level = (float)count_value / 20.0f;
    if (c->abstraction_level > 1.0f) c->abstraction_level = 1.0f;
    system->num_concepts++;

    size_t copy_dim = embed_dim < TEACH_UNIFIED_DIM ? embed_dim : TEACH_UNIFIED_DIM;
    memcpy(abstraction_embedding, count_embed, copy_dim * sizeof(float));
    return (int)idx;
}

int teach_test_concept(TeachSystem* system,
                        const float* visual_feat, size_t visual_dim,
                        const float* audio_feat, size_t audio_dim,
                        const float* tactile_feat, size_t tactile_dim,
                        char* recognized_concept, size_t buf_size,
                        float* confidence) {
    if (!system || !recognized_concept || !confidence) return -1;
    if (system->num_concepts == 0) {
        *confidence = 0.0f;
        strncpy(recognized_concept, "未知", buf_size - 1);
        recognized_concept[buf_size - 1] = '\0';
        return -2;
    }

    float* scratch = system->scratch_buffer;
    float query_feat[TEACH_UNIFIED_DIM];
    memset(query_feat, 0, sizeof(query_feat));

    float fused_input[TEACH_VISUAL_DIM + TEACH_AUDIO_DIM + TEACH_TACTILE_DIM + TEACH_TEXT_DIM];
    memset(fused_input, 0, sizeof(fused_input));

    if (visual_feat && visual_dim > 0) {
        size_t cv = visual_dim < TEACH_VISUAL_DIM ? visual_dim : TEACH_VISUAL_DIM;
        memcpy(scratch, visual_feat, cv * sizeof(float));
        teach_normalize_vector(scratch, cv);
        lnn_forward(system->visual_encoder, scratch, fused_input);
    }

    if (audio_feat && audio_dim > 0) {
        memset(scratch, 0, TEACH_AUDIO_DIM * sizeof(float));
        size_t ca = audio_dim < TEACH_AUDIO_DIM ? audio_dim : TEACH_AUDIO_DIM;
        float max_amp = 0.001f;
        for (size_t i = 0; i < audio_dim; i++) {
            if (fabsf(audio_feat[i]) > max_amp) max_amp = fabsf(audio_feat[i]);
        }
        for (size_t i = 0; i < ca; i++) scratch[i] = audio_feat[i] / max_amp;
        lnn_forward(system->audio_encoder, scratch, fused_input + TEACH_VISUAL_DIM);
    }

    if (tactile_feat && tactile_dim > 0) {
        memset(scratch, 0, TEACH_TACTILE_DIM * sizeof(float));
        size_t ct = tactile_dim < TEACH_TACTILE_DIM ? tactile_dim : TEACH_TACTILE_DIM;
        float max_val = 0.001f;
        for (size_t i = 0; i < tactile_dim; i++) {
            if (fabsf(tactile_feat[i]) > max_val) max_val = fabsf(tactile_feat[i]);
        }
        for (size_t i = 0; i < ct; i++) scratch[i] = tactile_feat[i] / max_val;
        lnn_forward(system->tactile_encoder, scratch, fused_input + TEACH_VISUAL_DIM + TEACH_AUDIO_DIM);
    }

    lnn_forward(system->concept_fuser, fused_input, query_feat);
    teach_normalize_vector(query_feat, TEACH_UNIFIED_DIM);

    float best_sim = -1.0f;
    int best_idx = -1;
    for (size_t i = 0; i < system->num_concepts; i++) {
        float sim = cosine_similarity_teach(query_feat, system->concepts[i].unified_embedding, TEACH_UNIFIED_DIM);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = (int)i;
        }
    }

    if (best_idx < 0 || best_sim < 0.2f) {
        *confidence = best_sim > 0.0f ? best_sim : 0.0f;
        strncpy(recognized_concept, "未知", buf_size - 1);
        recognized_concept[buf_size - 1] = '\0';
        return -3;
    }

    strncpy(recognized_concept, system->concepts[best_idx].concept_name, buf_size - 1);
    recognized_concept[buf_size - 1] = '\0';
    *confidence = best_sim;
    return best_idx;
}

int teach_get_concepts(TeachSystem* system,
                        TeachConcept* concepts_out, size_t* num_concepts) {
    if (!system || !concepts_out || !num_concepts) return -1;

    size_t copy_count = system->num_concepts;
    if (copy_count > *num_concepts) copy_count = *num_concepts;

    for (size_t i = 0; i < copy_count; i++) {
        memcpy(&concepts_out[i], &system->concepts[i], sizeof(TeachConcept));
    }

    *num_concepts = system->num_concepts;
    return 0;
}

int teach_clear_concept(TeachSystem* system, const char* concept_name) {
    if (!system || !concept_name) return -1;

    int idx = teach_find_concept(system, concept_name);
    if (idx < 0) return -2;

    size_t sidx = (size_t)idx;
    for (size_t i = sidx; i < system->num_concepts - 1; i++) {
        memcpy(&system->concepts[i], &system->concepts[i + 1], sizeof(TeachConcept));
    }
    memset(&system->concepts[system->num_concepts - 1], 0, sizeof(TeachConcept));
    system->num_concepts--;
    return 0;
}

int teach_clear_all_concepts(TeachSystem* system) {
    if (!system) return -1;

    memset(system->concepts, 0, sizeof(system->concepts));
    system->num_concepts = 0;
    return 0;
}
