/**
 * @file dialogue_memory.h
 * @brief 对话记忆增强系统接口
 */

#ifndef SELFLNN_DIALOGUE_MEMORY_H
#define SELFLNN_DIALOGUE_MEMORY_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DM_MAX_CONTEXT 20
#define DM_MAX_TOPICS 10
#define DM_MAX_REFERENCES 50
#define DM_MAX_SUMMARY 2048

typedef struct {
    int turn_id;
    char speaker[32];
    char text[1024];
    time_t timestamp;
    float importance;
    int topic_id;
    char* entities[8];
    int entity_count;
} DialogueTurn;

typedef struct {
    int topic_id;
    char name[64];
    int turn_count;
    time_t first_mentioned;
    time_t last_mentioned;
    int active;
    float relevance;
} DialogueTopic;

typedef struct {
    int ref_id;
    int source_turn;
    int target_turn;
    char ref_type[32];
    float confidence;
} DialogueReference;

typedef struct {
    DialogueTurn* turns;
    int turn_count;
    int turn_capacity;
    DialogueTopic topics[DM_MAX_TOPICS];
    int topic_count;
    DialogueReference references[DM_MAX_REFERENCES];
    int reference_count;
    char summary[DM_MAX_SUMMARY];
    char user_model[512];
    time_t session_start;
    time_t last_update;
    int session_id;
} DialogueMemory;

typedef struct DialogueMemoryManager DialogueMemoryManager;

DialogueMemoryManager* dlg_memory_create(void);
void dialogue_memory_free(DialogueMemoryManager* dmm);

int dm_add_turn(DialogueMemoryManager* dmm, const char* speaker, const char* text, float importance);
int dm_add_entity_to_turn(DialogueMemoryManager* dmm, int turn_id, const char* entity);
int dm_detect_topics(DialogueMemoryManager* dmm);
int dm_resolve_references(DialogueMemoryManager* dmm);
int dm_generate_summary(DialogueMemoryManager* dmm);

int dm_get_current_session(const DialogueMemoryManager* dmm, DialogueMemory* session);
int dm_get_turn(const DialogueMemoryManager* dmm, int turn_id, DialogueTurn* turn);
int dm_get_topics(const DialogueMemoryManager* dmm, DialogueTopic* topics, int max_count);
int dm_get_context_window(const DialogueMemoryManager* dmm, int window_size, DialogueTurn* turns);

/* 用户建模 */
int dm_update_user_model(DialogueMemoryManager* dmm, const char* preference_update);
const char* dm_get_user_model(const DialogueMemoryManager* dmm);

/* 会话边界 */
int dm_new_session(DialogueMemoryManager* dmm);
int dm_end_session(DialogueMemoryManager* dmm);
int dm_save_session(const DialogueMemoryManager* dmm, const char* filepath);
int dm_load_session(DialogueMemoryManager* dmm, const char* filepath);

#ifdef __cplusplus
}
#endif
#endif
