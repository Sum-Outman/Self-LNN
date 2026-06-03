/**
 * @file teaching_loop.h
 * @brief 多模态教学闭环系统接口
 */

#ifndef SELFLNN_TEACHING_LOOP_H
#define SELFLNN_TEACHING_LOOP_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TL_MAX_CONCEPTS 256
#define TL_MAX_SAMPLES 100
#define TL_MAX_FEATURES 512
#define TL_MAX_NAME 128

typedef enum {
    CONCEPT_OBJECT = 0,
    CONCEPT_ACTION = 1,
    CONCEPT_PROPERTY = 2,
    CONCEPT_RELATION = 3,
    CONCEPT_ABSTRACT = 4
} ConceptType;

typedef struct {
    int concept_id;
    char name[TL_MAX_NAME];
    ConceptType type;
    int parent_id;
    int children[16];
    int child_count;
    float visual_features[TL_MAX_FEATURES];
    float audio_features[TL_MAX_FEATURES];
    float haptic_features[TL_MAX_FEATURES];
    float sensor_features[TL_MAX_FEATURES];
    int visual_dim;
    int audio_dim;
    int haptic_dim;
    int sensor_dim;
    float mastery_level;
    time_t last_reviewed;
    time_t created_at;
    int review_count;
    float forgetting_rate;
    int sample_ids[TL_MAX_SAMPLES];
    int sample_count;
} TeachingConcept;

typedef struct {
    int sample_id;
    int concept_id;
    char label[TL_MAX_NAME];
    float features[TL_MAX_FEATURES];
    int feature_dim;
    int modality_mask;
    time_t presented_at;
    int correct_responses;
    int total_responses;
} TeachingSample;

typedef struct {
    int session_id;
    char topic[TL_MAX_NAME];
    TeachingConcept concepts[TL_MAX_CONCEPTS];
    int concept_count;
    TeachingSample samples[TL_MAX_SAMPLES * TL_MAX_CONCEPTS];
    int sample_count;
    int current_concept;
    float session_progress;
    time_t started_at;
    int is_active;
    int questions_asked;
    int correct_answers;
} TeachingSession;

typedef struct TeachingLoopSystem TeachingLoopSystem;

TeachingLoopSystem* teaching_loop_create(void);
void teaching_loop_free(TeachingLoopSystem* tls);

/* LNN绑定：可为教学闭环绑定外部液态神经网络 */
int teaching_loop_bind_lnn(TeachingLoopSystem* tls, void* external_lnn);

/* 概念管理 */
int tl_add_concept(TeachingLoopSystem* tls, TeachingSession* session, const TeachingConcept* concept);
int tl_link_concepts(TeachingLoopSystem* tls, TeachingSession* session, int parent_id, int child_id);
int tl_get_concept(const TeachingSession* session, int concept_id, TeachingConcept* out);

/* 联合感知学习 */
int tl_teach_object(TeachingLoopSystem* tls, TeachingSession* session, const char* name, const float* visual, int vdim, const float* audio, int adim, const float* haptic, int hdim);
int tl_cross_modal_associate(TeachingLoopSystem* tls, TeachingSession* session, int concept_id);
int tl_test_concept(TeachingLoopSystem* tls, TeachingSession* session, int concept_id, int* correct);
int tl_assess_mastery(const TeachingSession* session, int concept_id, float* mastery);

/* 遗忘曲线 */
int tl_update_forgetting(TeachingLoopSystem* tls, TeachingSession* session);
int tl_schedule_review(TeachingLoopSystem* tls, TeachingSession* session, int* concepts_to_review, int max_count);

/* 教学流程 */
int tl_start_session(TeachingLoopSystem* tls, TeachingSession* session, const char* topic);
int tl_end_session(TeachingLoopSystem* tls, TeachingSession* session);
int tl_get_next_question(TeachingLoopSystem* tls, TeachingSession* session, char* question, size_t max_len);
int tl_submit_answer(TeachingLoopSystem* tls, TeachingSession* session, const char* answer, int* correct);
float tl_get_session_progress(const TeachingSession* session);

#ifdef __cplusplus
}
#endif
#endif
