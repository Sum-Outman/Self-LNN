/**
 * @file teaching_loop.c
 * @brief 多模态教学闭环系统完整实现
 */
#include "selflnn/multimodal/teaching_loop.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "selflnn/utils/secure_random.h"
#include "selflnn/learning/imitation_deep.h"  /* H-016集成: 深度模仿学习 */
#include "selflnn/core/lnn.h"                  /* LNN演化集成 */
#include "selflnn/core/cfc_cell.h"             /* CfC细胞 */
#ifdef _WIN32
#include <windows.h>
#endif

/* 前向声明 */
struct TeachingLoopSystem {
    int session_counter;
    LNN* lnn;                   /**< 关联的液态神经网络（教学时动态演化） */
    int lnn_owned;              /**< 是否由本系统创建（1=需要释放） */
};

TeachingLoopSystem* teaching_loop_create(void) {
    TeachingLoopSystem* tls = (TeachingLoopSystem*)safe_calloc(1, sizeof(TeachingLoopSystem));
    if (!tls) return NULL;

    /* 创建教学专用的轻量级LNN网络（64-128-64架构） */
    LNNConfig net_cfg;
    memset(&net_cfg, 0, sizeof(LNNConfig));
    net_cfg.input_size = TL_MAX_FEATURES;    /* 接受多模态特征 */
    net_cfg.hidden_size = 128;                /* 隐藏层128个CfC神经元 */
    net_cfg.output_size = 64;                 /* 输出嵌入向量 */
    net_cfg.learning_rate = 0.001f;
    net_cfg.time_constant = 0.05f;            /* 快速时间常数，适合在线学习 */
    net_cfg.enable_training = 1;
    net_cfg.enable_adaptation = 1;

    tls->lnn = lnn_create(&net_cfg);
    if (tls->lnn) {
        tls->lnn_owned = 1;  /* 自身创建，负责释放 */
    }

    return tls;
}

void teaching_loop_free(TeachingLoopSystem* tls) {
    if (!tls) return;
    /* 验证指针是否为规范x64地址，防止在损坏指针上解引用 */
    {
        uintptr_t val = (uintptr_t)tls;
        uintptr_t high = (val >> 47) & 0x1FFFF;
        if (high != 0 && high != 0x1FFFF) {
            log_error("teaching_loop_free: 指针损坏(0x%016llX)，跳过释放",
                      (unsigned long long)val);
            return;
        }
    }
#ifdef _WIN32
    __try {
#endif
    if (tls->lnn_owned && tls->lnn) {
        lnn_free(tls->lnn);
    }
#ifdef _WIN32
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        log_error("teaching_loop_free 子资源访问异常 (0x%08lX), 跳过清理", GetExceptionCode());
    }
#endif
    safe_free((void**)&tls);
}

/* 绑定外部LNN网络（外部管理生命周期） */
int teaching_loop_bind_lnn(TeachingLoopSystem* tls, void* external_lnn) {
    if (!tls || !external_lnn) return -1;
    /* 释放自身创建的LNN */
    if (tls->lnn_owned && tls->lnn) {
        lnn_free(tls->lnn);
    }
    tls->lnn = external_lnn;
    tls->lnn_owned = 0;
    return 0;
}

int tl_add_concept(TeachingLoopSystem* tls, TeachingSession* session, const TeachingConcept* concept) {
    if (!tls || !session || !concept || session->concept_count >= TL_MAX_CONCEPTS) return -1;
    TeachingConcept* c = &session->concepts[session->concept_count];
    memcpy(c, concept, sizeof(TeachingConcept));
    c->concept_id = session->concept_count + 1;
    c->created_at = time(NULL);
    c->mastery_level = 0.0f;
    c->forgetting_rate = 0.02f;
    session->concept_count++;
    return c->concept_id;
}

int tl_link_concepts(TeachingLoopSystem* tls, TeachingSession* session, int parent_id, int child_id) {
    if (!tls || !session || parent_id < 1 || child_id < 1) return -1;
    if (parent_id > session->concept_count || child_id > session->concept_count) return -1;
    TeachingConcept* parent = &session->concepts[parent_id - 1];
    if (parent->child_count >= 16) return -1;
    parent->children[parent->child_count++] = child_id;
    session->concepts[child_id - 1].parent_id = parent_id;
    return 0;
}

int tl_get_concept(const TeachingSession* session, int concept_id, TeachingConcept* out) {
    if (!session || !out || concept_id < 1 || concept_id > session->concept_count) return -1;
    memcpy(out, &session->concepts[concept_id - 1], sizeof(TeachingConcept));
    return 0;
}

int tl_teach_object(TeachingLoopSystem* tls, TeachingSession* session, const char* name,
    const float* visual, int vdim, const float* audio, int adim, const float* haptic, int hdim) {
    if (!tls || !session || !name) return -1;
    TeachingConcept concept;
    memset(&concept, 0, sizeof(TeachingConcept));
    snprintf(concept.name, TL_MAX_NAME, "%s", name);
    concept.type = CONCEPT_OBJECT;

    if (visual && vdim > 0) {
        concept.visual_dim = vdim < TL_MAX_FEATURES ? vdim : TL_MAX_FEATURES;
        memcpy(concept.visual_features, visual, concept.visual_dim * sizeof(float));
    }
    if (audio && adim > 0) {
        concept.audio_dim = adim < TL_MAX_FEATURES ? adim : TL_MAX_FEATURES;
        memcpy(concept.audio_features, audio, concept.audio_dim * sizeof(float));
    }
    if (haptic && hdim > 0) {
        concept.haptic_dim = hdim < TL_MAX_FEATURES ? hdim : TL_MAX_FEATURES;
        memcpy(concept.haptic_features, haptic, concept.haptic_dim * sizeof(float));
    }

    return tl_add_concept(tls, session, &concept);
}

int tl_cross_modal_associate(TeachingLoopSystem* tls, TeachingSession* session, int concept_id) {
    if (!tls || !session || concept_id < 1 || concept_id > session->concept_count) return -1;
    TeachingConcept* c = &session->concepts[concept_id - 1];

    /* 使用LNN进行跨模态特征融合与演化
     * 将概念的多模态特征输入LNN，通过CfC ODE连续动态演化
     * 增强跨模态关联权重 */
    if (tls->lnn) {
        /* 构建多模态融合输入向量 */
        float fused_input[TL_MAX_FEATURES];
        memset(fused_input, 0, sizeof(fused_input));

        int total_dim = c->visual_dim + c->audio_dim + c->haptic_dim + c->sensor_dim;
        if (total_dim > TL_MAX_FEATURES) total_dim = TL_MAX_FEATURES;

        /* 拼合多模态特征（视觉→音频→触觉→传感器） */
        int offset = 0;
        if (c->visual_dim > 0 && offset + c->visual_dim <= TL_MAX_FEATURES) {
            memcpy(fused_input + offset, c->visual_features, 
                   c->visual_dim * sizeof(float));
            offset += c->visual_dim;
        }
        if (c->audio_dim > 0 && offset + c->audio_dim <= TL_MAX_FEATURES) {
            memcpy(fused_input + offset, c->audio_features,
                   c->audio_dim * sizeof(float));
            offset += c->audio_dim;
        }
        if (c->haptic_dim > 0 && offset + c->haptic_dim <= TL_MAX_FEATURES) {
            memcpy(fused_input + offset, c->haptic_features,
                   c->haptic_dim * sizeof(float));
            offset += c->haptic_dim;
        }
        if (c->sensor_dim > 0 && offset + c->sensor_dim <= TL_MAX_FEATURES) {
            memcpy(fused_input + offset, c->sensor_features,
                   c->sensor_dim * sizeof(float));
        }

        /* 执行LNN前向传播：通过CfC ODE动态系统融合跨模态特征 */
        /* P2修复: 使用lnn_get_output_size()动态获取输出维度，替代硬编码256 */
        size_t lnn_out_size = lnn_get_output_size(tls->lnn);
        if (lnn_out_size == 0) lnn_out_size = 256; /* 安全回退 */
        float* lnn_output = (float*)safe_malloc(lnn_out_size * sizeof(float));
        if (!lnn_output) return -1;
        memset(lnn_output, 0, lnn_out_size * sizeof(float));
        int result = lnn_forward(tls->lnn, fused_input, lnn_output);

        if (result == 0) {
            /* LNN成功融合，计算输出激活强度 */
            float activation_sum = 0.0f;
            for (size_t i = 0; i < lnn_out_size; i++) {
                activation_sum += fabsf(lnn_output[i]);
            }
            float lnn_activation = activation_sum / (float)lnn_out_size;

            /* LNN激活度贡献到掌握度（0-1映射） */
            float lnn_gain = tanhf(lnn_activation) * 0.3f;
            c->mastery_level += 0.10f + lnn_gain;
            
            /* 根据LNN输出更新概念特征（在线学习） */
            if (c->visual_dim > 0) {
                for (int i = 0; i < c->visual_dim && i < 64; i++) {
                    c->visual_features[i] = 0.9f * c->visual_features[i] 
                        + 0.1f * lnn_output[i];
                }
            }
        } else {
            /* LNN不可用时使用基础增益 */
            c->mastery_level += 0.10f;
        }
        /* P2修复: 释放动态分配的LNN输出缓冲区 */
        safe_free((void**)&lnn_output);
    } else {
        /* 无LNN时使用基础增益（向后兼容） */
        c->mastery_level += 0.10f;
    }

    if (c->mastery_level > 1.0f) c->mastery_level = 1.0f;
    return 0;
}

int tl_test_concept(TeachingLoopSystem* tls, TeachingSession* session, int concept_id, int* correct) {
    if (!tls || !session || !correct) return -1;
    session->questions_asked++;
    /* S-017修复: mastery_level阈值从0.3降低到0.15，并增加review_count>=1作为严格条件
     * 仅当概念至少被复习过一次且掌握度达到基础阈值时才判断为正确 */
    int mastery_ok = (concept_id > 0 && concept_id <= session->concept_count &&
                      session->concepts[concept_id - 1].mastery_level > 0.15f);
    /* 额外严格条件：必须至少复习过一次才判对（防止未经复习的概念被判正确） */
    int has_been_reviewed = (concept_id > 0 && concept_id <= session->concept_count &&
                             session->concepts[concept_id - 1].review_count >= 1);
    *correct = (mastery_ok && has_been_reviewed) ? 1 : 0;
    if (*correct) session->correct_answers++;

    if (concept_id >= 1 && concept_id <= session->concept_count) {
        TeachingConcept* c = &session->concepts[concept_id - 1];
        c->review_count++;
        c->last_reviewed = time(NULL);
        if (*correct) c->mastery_level += 0.1f; else c->mastery_level -= 0.05f;
        if (c->mastery_level < 0.0f) c->mastery_level = 0.0f;
        if (c->mastery_level > 1.0f) c->mastery_level = 1.0f;
    }
    return 0;
}

int tl_assess_mastery(const TeachingSession* session, int concept_id, float* mastery) {
    if (!session || !mastery || concept_id < 1 || concept_id > session->concept_count) return -1;
    *mastery = session->concepts[concept_id - 1].mastery_level;
    return 0;
}

int tl_update_forgetting(TeachingLoopSystem* tls, TeachingSession* session) {
    if (!tls || !session) return -1;
    time_t now = time(NULL);
    for (int i = 0; i < session->concept_count; i++) {
        TeachingConcept* c = &session->concepts[i];
        if (c->last_reviewed == 0) continue;
        float days_elapsed = (float)difftime(now, c->last_reviewed) / 86400.0f;
        if (days_elapsed > 0.0f) {
            c->mastery_level -= c->forgetting_rate * days_elapsed;
            if (c->mastery_level < 0.0f) c->mastery_level = 0.0f;
        }
    }
    return 0;
}

int tl_schedule_review(TeachingLoopSystem* tls, TeachingSession* session, int* concepts_to_review, int max_count) {
    if (!tls || !session || !concepts_to_review) return 0;
    tl_update_forgetting(tls, session);
    int count = 0;
    for (int i = 0; i < session->concept_count && count < max_count; i++) {
        if (session->concepts[i].mastery_level < 0.5f &&
            difftime(time(NULL), session->concepts[i].last_reviewed) > 300.0) {
            concepts_to_review[count++] = session->concepts[i].concept_id;
        }
    }
    return count;
}

int tl_start_session(TeachingLoopSystem* tls, TeachingSession* session, const char* topic) {
    if (!tls || !session || !topic) return -1;
    memset(session, 0, sizeof(TeachingSession));
    session->session_id = ++tls->session_counter;
    snprintf(session->topic, TL_MAX_NAME, "%s", topic);
    session->started_at = time(NULL);
    session->is_active = 1;
    return 0;
}

int tl_end_session(TeachingLoopSystem* tls, TeachingSession* session) {
    if (!tls || !session) return -1;
    session->is_active = 0;
    
    /* H-016集成: 教学结束后触发深度模仿学习 */
    if (session->concept_count > 0) {
        ImitationDeepLearner* idl = imitation_deep_create();
        if (idl) {
            /* 提取教学中的动作数据进行行为克隆训练 */
            float* action_data = (float*)safe_malloc(TL_MAX_FEATURES * sizeof(float));
            if (action_data) {
                for (int i = 0; i < session->concept_count; i++) {
                    TeachingConcept* c = &session->concepts[i];
                    if (c->visual_dim > 0 && c->visual_features) {
                        int dim = c->visual_dim < TL_MAX_FEATURES ? c->visual_dim : TL_MAX_FEATURES;
                        memcpy(action_data, c->visual_features, dim * sizeof(float));
                    }
                }
                /* 从教学概念中学习行为策略 */
                ImDemonstration demo;
                memset(&demo, 0, sizeof(ImDemonstration));
                if (session->concept_count >= 1) {
                    TeachingConcept* first = &session->concepts[0];
                    int joints = first->visual_dim > 0 ? (first->visual_dim < IM_MAX_JOINTS ? first->visual_dim : IM_MAX_JOINTS) : 1;
                    im_load_demonstration(idl, action_data, session->concept_count, joints, &demo);
                    /* 提取关键帧和轨迹编码 */
                    ImKeyframe kf[IM_MAX_KEYFRAMES];
                    int kf_count = im_extract_keyframes(idl, &demo, kf, IM_MAX_KEYFRAMES);
                    if (kf_count > 0) {
                        im_encode_trajectory(idl, &demo);
                    }
                }
                safe_free((void**)&action_data);
            }
            imitation_deep_free(idl);
        }
    }
    
    return 0;
}

int tl_get_next_question(TeachingLoopSystem* tls, TeachingSession* session, char* question, size_t max_len) {
    if (!tls || !session || !question) return -1;

    /* 优先复习最弱的概念 */
    int weakest = -1;
    float min_mastery = 2.0f;
    for (int i = 0; i < session->concept_count; i++) {
        if (session->concepts[i].mastery_level < min_mastery) {
            min_mastery = session->concepts[i].mastery_level;
            weakest = i;
        }
    }
    if (weakest >= 0) {
        snprintf(question, max_len, "关于\"%s\"，请描述它的特征。", session->concepts[weakest].name);
        session->current_concept = session->concepts[weakest].concept_id;
    } else if (session->concept_count > 0) {
        int idx = (int)secure_random_int((uint32_t)session->concept_count);
        snprintf(question, max_len, "请确认你对\"%s\"的理解。", session->concepts[idx].name);
        session->current_concept = session->concepts[idx].concept_id;
    } else {
        snprintf(question, max_len, "请描述你学到了什么？");
        session->current_concept = 0;
    }
    return 0;
}

/* S-017修复: 字符嵌入+余弦相似度匹配
 * 提取回答和期望概念的字符N-gram频率特征向量，计算余弦相似度
 * 同时计算与概念名称的编辑距离归一化分数，综合判断正确性 */
static float tl_calculate_semantic_similarity(const char* answer, const TeachingConcept* concept) {
    if (!answer || !concept || strlen(answer) == 0) return 0.0f;

    /* 提取回答的字符bigram频率特征（64维） */
    float ans_feat[64];
    memset(ans_feat, 0, sizeof(ans_feat));
    size_t alen = strlen(answer);
    if (alen > 256) alen = 256;
    float weight = 1.0f;
    for (size_t i = 0; i + 1 < alen; i++) {
        unsigned int hash = ((unsigned char)answer[i] * 31 + (unsigned char)answer[i+1]) * 2654435761u;
        int idx = (int)(hash % 64);
        ans_feat[idx] += weight;
        weight *= 0.92f;
    }
    /* L2归一化回答特征 */
    float norm = 0.0f;
    for (int d = 0; d < 64; d++) norm += ans_feat[d] * ans_feat[d];
    if (norm > 1e-6f) { norm = sqrtf(norm); for (int d = 0; d < 64; d++) ans_feat[d] /= norm; }

    /* 提取概念名称的bigram特征 */
    float name_feat[64];
    memset(name_feat, 0, sizeof(name_feat));
    size_t nlen = strlen(concept->name);
    if (nlen > 128) nlen = 128;
    for (size_t i = 0; i + 1 < nlen; i++) {
        unsigned int hash = ((unsigned char)concept->name[i] * 31 + (unsigned char)concept->name[i+1]) * 2654435761u;
        int idx = (int)(hash % 64);
        name_feat[idx] += 1.0f;
    }
    norm = 0.0f;
    for (int d = 0; d < 64; d++) norm += name_feat[d] * name_feat[d];
    if (norm > 1e-6f) { norm = sqrtf(norm); for (int d = 0; d < 64; d++) name_feat[d] /= norm; }

    /* 余弦相似度 */
    float cos_sim = 0.0f;
    for (int d = 0; d < 64; d++) cos_sim += ans_feat[d] * name_feat[d];
    if (cos_sim < 0.0f) cos_sim = 0.0f;

    /* 编辑距离归一化分数 */
    float edit_score = 0.0f;
    {
        size_t la = strlen(answer);
        size_t lb = strlen(concept->name);
        if (la > 0 && lb > 0) {
            /* 动态规划计算编辑距离 */
            int dp[128][128];
            int max_a = (int)(la < 127 ? la : 127);
            int max_b = (int)(lb < 127 ? lb : 127);
            for (int i = 0; i <= max_a; i++) dp[i][0] = i;
            for (int j = 0; j <= max_b; j++) dp[0][j] = j;
            for (int i = 1; i <= max_a; i++) {
                for (int j = 1; j <= max_b; j++) {
                    int cost = (answer[i-1] == concept->name[j-1]) ? 0 : 1;
                    int del = dp[i-1][j] + 1;
                    int ins = dp[i][j-1] + 1;
                    int sub = dp[i-1][j-1] + cost;
                    dp[i][j] = del < ins ? (del < sub ? del : sub) : (ins < sub ? ins : sub);
                }
            }
            int max_len = max_a > max_b ? max_a : max_b;
            if (max_len > 0) {
                edit_score = 1.0f - (float)dp[max_a][max_b] / (float)max_len;
                if (edit_score < 0.0f) edit_score = 0.0f;
            }
        }
    }

    /* 综合评分：余弦相似度 (0.6) + 编辑距离 (0.4) */
    float combined = cos_sim * 0.6f + edit_score * 0.4f;
    return combined;
}

int tl_submit_answer(TeachingLoopSystem* tls, TeachingSession* session, const char* answer, int* correct) {
    if (!tls || !session || !correct) return -1;
    /* S-017修复: 使用语义相似度匹配替代简单的字符长度判断
     * 提取回答的特征向量与当前概念的语义特征比较，
     * 仅当余弦相似度+编辑距离综合得分超过阈值时判对 */
    int is_correct = 0;
    if (answer && strlen(answer) > 0 && session->current_concept > 0 &&
        session->current_concept <= session->concept_count) {
        TeachingConcept* c = &session->concepts[session->current_concept - 1];
        float similarity = tl_calculate_semantic_similarity(answer, c);
        /* 综合相似度阈值：0.25（平衡精确匹配和宽松语义匹配） */
        is_correct = (similarity > 0.25f) ? 1 : 0;
    }
    *correct = is_correct;

    if (session->current_concept > 0) {
        TeachingConcept* c = &session->concepts[session->current_concept - 1];
        if (*correct) c->mastery_level += 0.08f; else c->mastery_level -= 0.03f;
        if (c->mastery_level < 0.0f) c->mastery_level = 0.0f;
        if (c->mastery_level > 1.0f) c->mastery_level = 1.0f;
        c->last_reviewed = time(NULL);
        c->review_count++;
    }
    session->questions_asked++;
    if (*correct) session->correct_answers++;
    session->session_progress = session->concept_count > 0 ?
        (float)session->questions_asked / (float)(session->concept_count * 3 + 1) : 0.5f;
    if (session->session_progress > 1.0f) session->session_progress = 1.0f;
    return 0;
}

float tl_get_session_progress(const TeachingSession* session) {
    return session ? session->session_progress : 0.0f;
}
