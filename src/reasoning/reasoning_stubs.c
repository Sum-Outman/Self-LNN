/**
 * @file reasoning_stubs.c
 * @brief MSVC链接存根 — reasoning.c中函数因含GCC扩展语法无法通过/WX编译，
 * 提取关键接口为独立存根以满足链接需求。
 */

#include "selflnn/reasoning/reasoning.h"
#include "selflnn/core/lnn.h"
#include "selflnn/knowledge/knowledge.h"
#include <stdlib.h>
#include <string.h>

/* ---- reasoning引擎核心API ---- */

ReasoningEngine* reasoning_engine_create(const ReasoningConfig* config) {
    (void)config;
    return (ReasoningEngine*)calloc(1, 256);
}

void reasoning_engine_free(ReasoningEngine* engine) {
    free(engine);
}

void reasoning_engine_reset(ReasoningEngine* engine) {
    (void)engine;
}

int reasoning_engine_get_config(const ReasoningEngine* engine, ReasoningConfig* config) {
    (void)engine;
    if (config) memset(config, 0, sizeof(ReasoningConfig));
    return 0;
}

int reasoning_engine_set_config(ReasoningEngine* engine, const ReasoningConfig* config) {
    (void)engine; (void)config;
    return 0;
}

int reasoning_engine_set_knowledge_base(ReasoningEngine* engine, struct KnowledgeBase* kb) {
    (void)engine; (void)kb;
    return 0;
}

int reasoning_engine_set_lnn(ReasoningEngine* engine, LNN* lnn) {
    (void)engine; (void)lnn;
    return 0;
}

int reasoning_infer_with_knowledge(ReasoningEngine* engine,
    const char** premises, int num_premises,
    char** conclusions, int max_conclusions,
    float* confidences) {
    (void)engine; (void)premises; (void)num_premises;
    (void)conclusions; (void)max_conclusions; (void)confidences;
    return 0;
}

/* ---- main.c AGI后台循环缺失的selflnn API ---- */

int selflnn_get_recent_state(void* lnn, float* state, int dim) {
    int i;
    (void)lnn;
    if (!state || dim <= 0) return -1;
    for (i = 0; i < dim && i < 128; i++) state[i] = 0.0f;
    return 0;
}

int selflnn_get_recent_output(void* lnn, float* output, int dim) {
    int i;
    (void)lnn;
    if (!output || dim <= 0) return -1;
    for (i = 0; i < dim && i < 128; i++) output[i] = 0.5f;
    return 0;
}

void* selflnn_get_knowledge_base(void) {
    return NULL;
}

int selflnn_get_active_goal(void* kb, float* goal, int dim) {
    int i;
    (void)kb;
    if (!goal || dim <= 0) return -1;
    for (i = 0; i < dim && i < 64; i++) goal[i] = 0.0f;
    return 0;
}
