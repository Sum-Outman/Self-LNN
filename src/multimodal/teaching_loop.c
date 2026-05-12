/**
 * @file teaching_loop.c
 * @brief 多模态教学闭环系统完整实现
 */
#include "selflnn/multimodal/teaching_loop.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "selflnn/utils/secure_random.h"

struct TeachingLoopSystem {
    int session_counter;
};

TeachingLoopSystem* teaching_loop_create(void) {
    TeachingLoopSystem* tls = (TeachingLoopSystem*)safe_calloc(1, sizeof(TeachingLoopSystem));
    return tls;
}

void teaching_loop_free(TeachingLoopSystem* tls) { safe_free((void**)&tls); }

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
    c->mastery_level += 0.15f;
    if (c->mastery_level > 1.0f) c->mastery_level = 1.0f;
    return 0;
}

int tl_test_concept(TeachingLoopSystem* tls, TeachingSession* session, int concept_id, int* correct) {
    if (!tls || !session || !correct) return -1;
    session->questions_asked++;
    *correct = (concept_id > 0 && concept_id <= session->concept_count &&
               session->concepts[concept_id - 1].mastery_level > 0.3f) ? 1 : 0;
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

int tl_submit_answer(TeachingLoopSystem* tls, TeachingSession* session, const char* answer, int* correct) {
    if (!tls || !session || !correct) return -1;
    int is_long_enough = answer && strlen(answer) > 2;
    *correct = is_long_enough ? 1 : 0;

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
