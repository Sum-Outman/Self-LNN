/**
 * @file dialogue_memory.c
 * @brief 对话记忆增强系统完整实现
 */
#include "selflnn/multimodal/dialogue_memory.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#define DM_INIT_CAPACITY 200

struct DialogueMemoryManager {
    DialogueMemory session;
    int session_count;
};

DialogueMemoryManager* dialogue_memory_create(void) {
    DialogueMemoryManager* dmm = (DialogueMemoryManager*)safe_calloc(1, sizeof(DialogueMemoryManager));
    if (!dmm) return NULL;

    dmm->session.turns = (DialogueTurn*)safe_calloc(DM_INIT_CAPACITY, sizeof(DialogueTurn));
    dmm->session.turn_capacity = DM_INIT_CAPACITY;
    dmm->session.turn_count = 0;
    dmm->session.topic_count = 0;
    dmm->session.reference_count = 0;
    dmm->session.session_start = time(NULL);
    dmm->session.last_update = time(NULL);
    dmm->session.session_id = (int)time(NULL);
    snprintf(dmm->session.user_model, sizeof(dmm->session.user_model), "新用户");

    return dmm;
}

void dialogue_memory_free(DialogueMemoryManager* dmm) {
    if (!dmm) return;
    if (dmm->session.turns) {
        for (int i = 0; i < dmm->session.turn_count; i++) {
            for (int j = 0; j < dmm->session.turns[i].entity_count; j++) {
                safe_free((void**)&dmm->session.turns[i].entities[j]);
            }
        }
        safe_free((void**)&dmm->session.turns);
    }
    safe_free((void**)&dmm);
}

int dm_add_turn(DialogueMemoryManager* dmm, const char* speaker, const char* text, float importance) {
    if (!dmm || !speaker || !text) return -1;

    if (dmm->session.turn_count >= dmm->session.turn_capacity) {
        size_t new_cap = dmm->session.turn_capacity * 2;
        void* new_turns = safe_realloc(dmm->session.turns, new_cap * sizeof(DialogueTurn));
        if (!new_turns) return -1;
        dmm->session.turns = (DialogueTurn*)new_turns;
        memset(dmm->session.turns + dmm->session.turn_capacity, 0, dmm->session.turn_capacity * sizeof(DialogueTurn));
        dmm->session.turn_capacity = (int)new_cap;
    }

    DialogueTurn* turn = &dmm->session.turns[dmm->session.turn_count];
    memset(turn, 0, sizeof(DialogueTurn));
    turn->turn_id = dmm->session.turn_count + 1;
    snprintf(turn->speaker, sizeof(turn->speaker), "%s", speaker);
    snprintf(turn->text, sizeof(turn->text), "%s", text);
    turn->timestamp = time(NULL);
    turn->importance = importance;
    turn->topic_id = -1;

    dmm->session.turn_count++;
    dmm->session.last_update = time(NULL);

    if (dmm->session.turn_count % 5 == 0) {
        dm_detect_topics(dmm);
        dm_resolve_references(dmm);
    }

    return turn->turn_id;
}

int dm_add_entity_to_turn(DialogueMemoryManager* dmm, int turn_id, const char* entity) {
    if (!dmm || !entity || turn_id <= 0 || turn_id > dmm->session.turn_count) return -1;
    DialogueTurn* turn = &dmm->session.turns[turn_id - 1];
    if (turn->entity_count >= 8) return -1;
    turn->entities[turn->entity_count] = string_duplicate(entity);
    turn->entity_count++;
    return 0;
}

int dm_detect_topics(DialogueMemoryManager* dmm) {
    if (!dmm || dmm->session.turn_count == 0) return -1;

    /* 基于关键词的简单话题检测 */
    const char* topic_keywords[] = {"训练", "机器人", "视觉", "语音", "知识", "控制", "编程", "学习", "传感器", "对话"};
    const char* topic_names[] = {"机器学习", "机器人控制", "视觉感知", "语音处理", "知识管理", "系统控制", "编程开发", "自主学习", "传感器", "对话交互"};
    int keyword_count = sizeof(topic_keywords) / sizeof(topic_keywords[0]);

    /* 分析最近10轮对话 */
    int start = dmm->session.turn_count > 10 ? dmm->session.turn_count - 10 : 0;
    int keyword_hits[10] = {0};

    for (int i = start; i < dmm->session.turn_count; i++) {
        for (int k = 0; k < keyword_count; k++) {
            if (strstr(dmm->session.turns[i].text, topic_keywords[k])) {
                keyword_hits[k]++;
            }
        }
    }

    dmm->session.topic_count = 0;
    for (int k = 0; k < keyword_count && dmm->session.topic_count < DM_MAX_TOPICS; k++) {
        if (keyword_hits[k] > 0) {
            DialogueTopic* topic = &dmm->session.topics[dmm->session.topic_count];
            topic->topic_id = dmm->session.topic_count + 1;
            snprintf(topic->name, sizeof(topic->name), "%s", topic_names[k]);
            topic->turn_count = keyword_hits[k];
            topic->active = (keyword_hits[k] >= 2);
            topic->relevance = (float)keyword_hits[k] / (float)(dmm->session.turn_count - start + 1);
            topic->first_mentioned = dmm->session.turns[start].timestamp;
            topic->last_mentioned = dmm->session.turns[dmm->session.turn_count - 1].timestamp;
            dmm->session.topic_count++;
        }
    }

    return 0;
}

int dm_resolve_references(DialogueMemoryManager* dmm) {
    if (!dmm || dmm->session.turn_count < 2) return -1;

    dmm->session.reference_count = 0;
    for (int i = 1; i < dmm->session.turn_count && dmm->session.reference_count < DM_MAX_REFERENCES; i++) {
        DialogueTurn* current = &dmm->session.turns[i];
        const char* refs[] = {"它", "这个", "那个", "他", "她", "这", "那", "该", "其", "it", "this", "that"};
        int ref_count = sizeof(refs) / sizeof(refs[0]);

        for (int r = 0; r < ref_count; r++) {
            if (strstr(current->text, refs[r])) {
                DialogueReference* ref = &dmm->session.references[dmm->session.reference_count];
                ref->ref_id = dmm->session.reference_count + 1;
                ref->source_turn = current->turn_id;
                ref->target_turn = i; /* 回指到前一轮 */
                snprintf(ref->ref_type, sizeof(ref->ref_type), "回指");
                ref->confidence = 0.7f;
                dmm->session.reference_count++;
                break;
            }
        }
    }

    return 0;
}

int dm_generate_summary(DialogueMemoryManager* dmm) {
    if (!dmm || dmm->session.turn_count == 0) return -1;

    memset(dmm->session.summary, 0, DM_MAX_SUMMARY);
    snprintf(dmm->session.summary, DM_MAX_SUMMARY,
        "会话#%d摘要：共%d轮对话，涉及%d个话题。",
        dmm->session.session_id, dmm->session.turn_count, dmm->session.topic_count);

    if (dmm->session.topic_count > 0) {
        size_t len = strlen(dmm->session.summary);
        snprintf(dmm->session.summary + len, DM_MAX_SUMMARY - len, "主要话题：");
        for (int i = 0; i < dmm->session.topic_count && i < 3; i++) {
            len = strlen(dmm->session.summary);
            snprintf(dmm->session.summary + len, DM_MAX_SUMMARY - len, "%s ", dmm->session.topics[i].name);
        }
    }

    return 0;
}

int dm_get_current_session(const DialogueMemoryManager* dmm, DialogueMemory* session) {
    if (!dmm || !session) return -1;
    memcpy(session, &dmm->session, sizeof(DialogueMemory));
    return 0;
}

int dm_get_turn(const DialogueMemoryManager* dmm, int turn_id, DialogueTurn* turn) {
    if (!dmm || !turn || turn_id <= 0 || turn_id > dmm->session.turn_count) return -1;
    memcpy(turn, &dmm->session.turns[turn_id - 1], sizeof(DialogueTurn));
    return 0;
}

int dm_get_topics(const DialogueMemoryManager* dmm, DialogueTopic* topics, int max_count) {
    if (!dmm || !topics) return 0;
    int count = dmm->session.topic_count < max_count ? dmm->session.topic_count : max_count;
    memcpy(topics, dmm->session.topics, count * sizeof(DialogueTopic));
    return count;
}

int dm_get_context_window(const DialogueMemoryManager* dmm, int window_size, DialogueTurn* turns) {
    if (!dmm || !turns || window_size <= 0) return 0;
    int start = dmm->session.turn_count - window_size;
    if (start < 0) start = 0;
    int count = dmm->session.turn_count - start;
    memcpy(turns, &dmm->session.turns[start], count * sizeof(DialogueTurn));
    return count;
}

int dm_update_user_model(DialogueMemoryManager* dmm, const char* preference_update) {
    if (!dmm || !preference_update) return -1;
    size_t len = strlen(dmm->session.user_model);
    snprintf(dmm->session.user_model + len, sizeof(dmm->session.user_model) - len, "; %s", preference_update);
    return 0;
}

const char* dm_get_user_model(const DialogueMemoryManager* dmm) {
    return dmm ? dmm->session.user_model : NULL;
}

int dm_new_session(DialogueMemoryManager* dmm) {
    if (!dmm) return -1;
    dm_generate_summary(dmm);
    dmm->session_count++;
    dmm->session.turn_count = 0;
    dmm->session.topic_count = 0;
    dmm->session.reference_count = 0;
    dmm->session.session_start = time(NULL);
    dmm->session.session_id++;
    return 0;
}

int dm_end_session(DialogueMemoryManager* dmm) {
    if (!dmm) return -1;
    return dm_generate_summary(dmm);
}

int dm_save_session(const DialogueMemoryManager* dmm, const char* filepath) {
    if (!dmm || !filepath) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;
    uint32_t magic = 0x444D454D;
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&dmm->session.session_id, sizeof(int), 1, fp);
    fwrite(&dmm->session.turn_count, sizeof(int), 1, fp);
    fwrite(dmm->session.turns, sizeof(DialogueTurn), dmm->session.turn_count, fp);
    fclose(fp);
    return 0;
}

int dm_load_session(DialogueMemoryManager* dmm, const char* filepath) {
    if (!dmm || !filepath) return -1;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, fp);
    if (magic != 0x444D454D) { fclose(fp); return -1; }
    int session_id, turn_count;
    fread(&session_id, sizeof(int), 1, fp);
    fread(&turn_count, sizeof(int), 1, fp);
    if (turn_count > (int)dmm->session.turn_capacity) {
        safe_free((void**)&dmm->session.turns);
        dmm->session.turns = (DialogueTurn*)safe_calloc(turn_count, sizeof(DialogueTurn));
        dmm->session.turn_capacity = turn_count;
    }
    dmm->session.turn_count = turn_count;
    dmm->session.session_id = session_id;
    fread(dmm->session.turns, sizeof(DialogueTurn), turn_count, fp);
    fclose(fp);
    dm_detect_topics(dmm);
    dm_resolve_references(dmm);
    return 0;
}
