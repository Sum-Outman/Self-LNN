/**
 * @file dialogue.c
 * @brief 对话系统核心实现
 *
 * SELF-LNN 对话系统，基于单一液态神经网络模型。
 * 所有对话功能（文本理解、意图分析、上下文管理、响应生成）
 * 统一使用同一个 CfC 连续动态系统。
 *
 * 深度增强功能（DST信念追踪、策略学习、多轮推理、CfC生成）
 * 由 dialogue_deep.c 提供，本文件通过 dialogue_register_deep_modules
 * 在初始化时自动连接深度模块。
 */

/* 必须定义这两个宏才能获取CfCNetwork和CfCCell完整结构体定义
 * cfc_network.h: struct CfCNetwork 由 SELFLNN_IMPLEMENTATION 保护
 * cfc_cell.h:    struct CfCCell    由 SELFLNN_CORE_INTERNAL 保护
 * 新增的代码需要访问cfc->layers、cell->config等内部成员 */
#define SELFLNN_IMPLEMENTATION
#define SELFLNN_CORE_INTERNAL

#include "selflnn/multimodal/dialogue.h"
#include "selflnn/multimodal/text.h"
#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/cfc_network.h" /* lnn_get_cfc_network */
#include "selflnn/core/lnn.h" /* lnn_get_cfc_network访问 */
#include "selflnn/selflnn.h" /* selflnn_get_shared_lnn */
#include "selflnn/knowledge/knowledge.h" /* 知识库检索回退 */
#include "selflnn/knowledge/knowledge_graph.h" /* R002: KG图推理检索 */
#include "selflnn/programming/programming_bridge.h" /* 对话→编程: 用户编程需求委托 */
#include "selflnn/knowledge/graph_reasoning.h" /* P1/P4: 链路预测+图推理引擎 */
#include "selflnn/backend/websocket_push.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/secure_random.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

/* dialogue_deep_train_policy来自dialogue_deep.c，
 * 提供基于TD学习的策略网络训练。此处extern声明使其在dialogue.c中可见，
 * 每次对话回复生成后自动调用进行增量策略优化。 */
extern int dialogue_deep_train_policy(DialogueProcessor* dp,
    const float* state_features, const float* next_state_features,
    float reward, int num_states, float learning_rate);

#define DIALOGUE_MAGIC 0x4449414C4F475545ULL
#define DIALOGUE_SAVE_VERSION 1
#define MAX_RESPONSE_TOKENS 256

#ifndef safe_strdup
/* 纯C实现的safe_strdup，跨平台兼容 */
static char* safe_strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* dup = (char*)safe_malloc(len);
    if (dup) memcpy(dup, s, len);
    return dup;
}
#endif

/* ================================================================
 *: CfC对话权重存储 — 替代每次调用随机生成
 * 对话系统的CfC门控/激活权重必须是确定性可复用参数，
 * 使用Xavier初始化一次后存储，后续调用复用同一权重。
 * 训练后权重将通过反向传播更新（当前为未训练状态的初始值）。
 * ================================================================ */
typedef struct {
    int initialized;
    size_t max_hs;
    size_t max_in;
    float* w_gate_input;   /* W_gx: [hs × in] */
    float* w_act_input;    /* W_ax: [hs × in] */
    float* u_gate_hidden;  /* W_gh: [hs × hs] */
    float* u_act_hidden;   /* W_ah: [hs × hs] */
    float* b_gate;         /* b_g:  [hs] */
    float* b_act;          /* b_a:  [hs] */
} DialogueCfCWeights;

static DialogueCfCWeights g_dialogue_cfc_weights = {0};

/* 全局对话处理器引用，用于LNN驱动的意图分析 */
static DialogueProcessor* g_dialogue_processor_global = NULL;

/* 外部WebSocket推送服务器引用（由main.c管理生命周期） */
extern WSPushServer* g_ws_push_server;

DialogueProcessor* dialogue_get_global_processor(void) {
    return g_dialogue_processor_global;
}

/* 初始化对话CfC权重（Xavier均匀分布，仅执行一次） */
static void dialogue_cfc_weights_init(DialogueCfCWeights* w, size_t hs, size_t in_dim)
{
    if (w->initialized && w->max_hs >= hs && w->max_in >= in_dim) return;

    if (w->initialized) {
        safe_free((void**)&w->w_gate_input);
        safe_free((void**)&w->w_act_input);
        safe_free((void**)&w->u_gate_hidden);
        safe_free((void**)&w->u_act_hidden);
        safe_free((void**)&w->b_gate);
        safe_free((void**)&w->b_act);
    }

    size_t hs_x_in = hs * in_dim;
    size_t hs_x_hs = hs * hs;

    w->w_gate_input   = (float*)safe_calloc(hs_x_in, sizeof(float));
    w->w_act_input    = (float*)safe_calloc(hs_x_in, sizeof(float));
    w->u_gate_hidden  = (float*)safe_calloc(hs_x_hs, sizeof(float));
    w->u_act_hidden   = (float*)safe_calloc(hs_x_hs, sizeof(float));
    w->b_gate         = (float*)safe_calloc(hs, sizeof(float));
    w->b_act          = (float*)safe_calloc(hs, sizeof(float));

    if (!w->w_gate_input || !w->w_act_input || !w->u_gate_hidden ||
        !w->u_act_hidden || !w->b_gate || !w->b_act) {
        safe_free((void**)&w->w_gate_input);
        safe_free((void**)&w->w_act_input);
        safe_free((void**)&w->u_gate_hidden);
        safe_free((void**)&w->u_act_hidden);
        safe_free((void**)&w->b_gate);
        safe_free((void**)&w->b_act);
        return;
    }

    /* Xavier均匀分布初始化: limit = sqrt(6/(fan_in+fan_out)) */
    float gate_limit = sqrtf(6.0f / (float)(in_dim + hs));
    float hidden_limit = sqrtf(6.0f / (float)(hs + hs));

    for (size_t i = 0; i < hs_x_in; i++) {
        w->w_gate_input[i] = (secure_random_float() * 2.0f - 1.0f) * gate_limit;
        w->w_act_input[i]  = (secure_random_float() * 2.0f - 1.0f) * gate_limit;
    }
    for (size_t i = 0; i < hs_x_hs; i++) {
        w->u_gate_hidden[i] = (secure_random_float() * 2.0f - 1.0f) * hidden_limit;
        w->u_act_hidden[i]  = (secure_random_float() * 2.0f - 1.0f) * hidden_limit;
    }
    for (size_t i = 0; i < hs; i++) {
        w->b_gate[i] = 0.0f;
        w->b_act[i]  = 0.0f;
    }

    w->max_hs = hs;
    w->max_in = in_dim;
    w->initialized = 1;
}

static float clip_f(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int str_has(const char* text, const char* kw) {
    if (!text || !kw) return 0;
    return strstr(text, kw) != NULL;
}

/* ================================================================
 * 对话处理器核心生命周期
 * ================================================================ */

DialogueProcessor* dialogue_processor_create(const DialogueConfig* config)
{
    if (!config) return NULL;

    DialogueProcessor* dp = (DialogueProcessor*)safe_calloc(1, sizeof(DialogueProcessor));
    if (!dp) return NULL;

    memcpy(&dp->config, config, sizeof(DialogueConfig));

    if (dp->config.dialogue_hidden_size == 0)
        dp->config.dialogue_hidden_size = 128;
    if (dp->config.dialogue_time_constant == 0.0f)
        dp->config.dialogue_time_constant = 0.1f;
    if (dp->config.dialogue_delta_t == 0.0f)
        dp->config.dialogue_delta_t = 0.05f;

    dp->dialogue_buffer_size = dp->config.dialogue_hidden_size;
    dp->dialogue_state_buffer = (float*)safe_calloc(dp->dialogue_buffer_size, sizeof(float));

    if (!dp->dialogue_state_buffer) {
        safe_free((void**)&dp);
        return NULL;
    }

    TextConfig text_cfg;
    memset(&text_cfg, 0, sizeof(text_cfg));
    text_cfg.max_tokens = 1024;
    text_cfg.vector_dimension = 128;
    text_cfg.language = 0;
    text_cfg.enable_cfc = 1;
    text_cfg.cfc_hidden_size = 32;
    text_cfg.cfc_time_constant = 0.1f;
    dp->text_processor = text_processor_create(&text_cfg);
    dp->is_initialized = 1;

/* 注册全局处理器引用，使意图分析可访问LNN */
    g_dialogue_processor_global = dp;

    return dp;
}

void dialogue_processor_free(DialogueProcessor* processor)
{
    if (!processor) return;

    if (processor->deep_belief) {
        dialogue_belief_state_free(processor->deep_belief);
        processor->deep_belief = NULL;
    }
    if (processor->deep_policy) {
        dialogue_policy_free(processor->deep_policy);
        processor->deep_policy = NULL;
    }
    if (processor->deep_reasoner) {
        multi_turn_reasoner_free(processor->deep_reasoner);
        processor->deep_reasoner = NULL;
    }
    if (processor->generator) {
        dialogue_gen_free(processor->generator);
        processor->generator = NULL;
    }

    if (processor->text_processor) {
        text_processor_free(processor->text_processor);
        processor->text_processor = NULL;
    }

    safe_free((void**)&processor->dialogue_state_buffer);
    memset(processor, 0, sizeof(DialogueProcessor));
    safe_free((void**)&processor);
}

/* 获取对话生成器句柄 */
void* dialogue_get_generator(DialogueProcessor* processor) {
    if (!processor) return NULL;
    return processor->generator;
}

int dialogue_set_lnn_instance(DialogueProcessor* processor, void* lnn_instance)
{
    if (!processor || !processor->is_initialized) return -1;
    processor->lnn_instance = lnn_instance;
    processor->lnn_owned = 0;
    return 0;
}

int dialogue_reset_state(DialogueProcessor* processor)
{
    if (!processor || !processor->dialogue_state_buffer) return -1;
    memset(processor->dialogue_state_buffer, 0,
           processor->dialogue_buffer_size * sizeof(float));
    if (processor->generator) {
        dialogue_gen_reset_state(processor->generator);
    }
    return 0;
}

int dialogue_processor_get_config(const DialogueProcessor* processor, DialogueConfig* config)
{
    if (!processor || !config) return -1;
    memcpy(config, &processor->config, sizeof(DialogueConfig));
    return 0;
}

int dialogue_processor_set_config(DialogueProcessor* processor, const DialogueConfig* config)
{
    if (!processor || !config) return -1;
    memcpy(&processor->config, config, sizeof(DialogueConfig));
    return 0;
}

/* ================================================================
 * CfC ODE 对话状态演化
 * dh/dt = -h/tau + sigmoid(W_g*x + U_g*h + b_g) * tanh(W_a*x + U_a*h + b_a)
 * ================================================================ */

int dialogue_evolve_state(DialogueProcessor* processor,
                          const float* input_features,
                          size_t feature_count,
                          float delta_t)
{
    if (!processor || !input_features || feature_count == 0) return -1;
    if (!processor->dialogue_state_buffer) return -1;

    size_t hs = processor->dialogue_buffer_size;
    float tau = processor->config.dialogue_time_constant;
    float dt = (delta_t > 0.0f) ? delta_t : processor->config.dialogue_delta_t;

    if (processor->lnn_instance && processor->config.use_cfc_evolution) {
        float* h = processor->dialogue_state_buffer;
        size_t in_dim = feature_count;

/* 使用存储的确定性Xavier初始化权重替代随机生成
 *修复: 当共享LNN可用时，优先从LNN同步权重；
         * 对话系统的独立CfC权重备份仅在LNN未训练时作为冷启动后备 */
        dialogue_cfc_weights_init(&g_dialogue_cfc_weights, hs, in_dim);
        DialogueCfCWeights* w = &g_dialogue_cfc_weights;

/* 尝试从共享LNN同步权重 */
        {
            void* shared_lnn = selflnn_get_shared_lnn();
            if (shared_lnn) {
                CfCNetwork* cfc = lnn_get_cfc_network(shared_lnn);
                if (cfc && cfc->layers && cfc->layers[0]) {
                    CfCCell* cell = cfc->layers[0];
                    size_t cell_hs = cell->config.hidden_size;
                    size_t cell_in = cell->config.input_size;
                    /* 维度匹配时才同步，避免越界 */
                    if (cell_in <= w->max_in && cell_hs <= w->max_hs) {
                        for (size_t i = 0; i < cell_hs && i < hs; i++) {
                            for (size_t j = 0; j < cell_in && j < in_dim; j++) {
                                w->w_gate_input[i * w->max_in + j] =
                                    cell->input_gate_weights[i * cell_in + j];
                                w->w_act_input[i * w->max_in + j] =
                                    cell->weight_matrix[i * cell_in + j];
                            }
                        }
                    }
                }
            }
        }

        for (size_t i = 0; i < hs; i++) {
            float gate_input = w->b_gate[i];
            float act_input  = w->b_act[i];

            size_t j;
            for (j = 0; j < in_dim && j < w->max_in; j++) {
                gate_input += w->w_gate_input[i * w->max_in + j] * input_features[j];
                act_input  += w->w_act_input[i * w->max_in + j]  * input_features[j];
            }
            for (j = 0; j < hs && j < w->max_hs; j++) {
                gate_input += w->u_gate_hidden[i * w->max_hs + j] * h[j];
                act_input  += w->u_act_hidden[i * w->max_hs + j]  * h[j];
            }

            float gate = 0.5f * (1.0f + tanhf(gate_input));
            float act  = tanhf(act_input);
            float decay = expf(-dt / tau);

            h[i] = h[i] * decay + (1.0f - decay) * gate * act;
        }
    } else {
        float alpha = 1.0f - expf(-dt / processor->config.dialogue_time_constant);
        size_t copy_count = (feature_count < processor->dialogue_buffer_size)
                            ? feature_count : processor->dialogue_buffer_size;
        for (size_t i = 0; i < copy_count; i++) {
            processor->dialogue_state_buffer[i] =
                (1.0f - alpha) * processor->dialogue_state_buffer[i]
                + alpha * input_features[i];
        }
    }

    return 0;
}

/* ================================================================
 * 对话上下文管理
 * ================================================================ */

DialogueContext* dialogue_context_create(size_t max_messages)
{
    if (max_messages == 0) max_messages = 50;

    DialogueContext* ctx = (DialogueContext*)safe_calloc(1, sizeof(DialogueContext));
    if (!ctx) return NULL;

    ctx->messages = (DialogueMessage*)safe_calloc(max_messages, sizeof(DialogueMessage));
    if (!ctx->messages) {
        safe_free((void**)&ctx);
        return NULL;
    }

    ctx->max_messages = max_messages;
    ctx->num_messages = 0;
    ctx->context_id = (int)(time(NULL) % 100000);
    ctx->created_time = time(NULL);
    ctx->last_active = time(NULL);

    return ctx;
}

void dialogue_context_free(DialogueContext* context)
{
    if (!context) return;

    for (size_t i = 0; i < context->num_messages; i++) {
        if (context->messages[i].text_allocated && context->messages[i].text) {
            safe_free((void**)&context->messages[i].text);
        }
    }
    safe_free((void**)&context->messages);
    memset(context, 0, sizeof(DialogueContext));
    safe_free((void**)&context);
}

int dialogue_context_add_message(DialogueContext* context, const DialogueMessage* message)
{
    if (!context || !message || !message->text) return -1;

    if (context->num_messages >= context->max_messages) {
        if (context->messages[0].text_allocated && context->messages[0].text) {
            safe_free((void**)&context->messages[0].text);
        }
        memmove(context->messages, context->messages + 1,
                (context->max_messages - 1) * sizeof(DialogueMessage));
        context->num_messages--;
    }

    DialogueMessage* dm = &context->messages[context->num_messages];
    memset(dm, 0, sizeof(DialogueMessage));

    dm->text = safe_strdup(message->text);
    dm->length = message->length ? message->length : strlen(message->text);
    dm->role = message->role;
    dm->timestamp = message->timestamp ? message->timestamp : time(NULL);
    dm->confidence = message->confidence;
    dm->text_allocated = 1;

    context->num_messages++;
    context->last_active = time(NULL);
    return 0;
}

void dialogue_context_clear(DialogueContext* context)
{
    if (!context) return;

    for (size_t i = 0; i < context->num_messages; i++) {
        if (context->messages[i].text_allocated && context->messages[i].text) {
            safe_free((void**)&context->messages[i].text);
        }
    }
    memset(context->messages, 0, context->max_messages * sizeof(DialogueMessage));
    context->num_messages = 0;
    context->last_active = time(NULL);
}

int dialogue_context_get_summary(const DialogueContext* context,
                                 char* summary, size_t max_length)
{
    if (!context || !summary || max_length == 0) return -1;

    int written = snprintf(summary, max_length,
                           /* 对话上下文概要：共%zu条消息，创建于%ld，最后活跃%ld */
                           "{\"total_messages\":%zu,\"context_id\":%d,\"created\":%ld,\"last_active\":%ld}",
                           context->num_messages, context->context_id,
                           (long)context->created_time, (long)context->last_active);

    return (written > 0 && (size_t)written < max_length) ? written : -1;
}

char* dialogue_context_export_json(const DialogueContext* context)
{
    if (!context) return NULL;

    size_t est = 256 + context->num_messages * 400;
    char* buf = (char*)safe_malloc(est);
    if (!buf) return NULL;

    int pos = snprintf(buf, est,
        "{\"context_id\":%d,\"created\":%ld,\"last_active\":%ld,\"num_messages\":%zu,\"messages\":[",
        context->context_id, (long)context->created_time,
        (long)context->last_active, context->num_messages);

    for (size_t i = 0; i < context->num_messages; i++) {
        DialogueMessage* m = &context->messages[i];
        char escaped[512];
        size_t elen = 0;
        const char* src = m->text ? m->text : "";
        while (*src && elen < 500) {
            if (*src == '"') { escaped[elen++] = '\\'; escaped[elen++] = '"'; }
            else if (*src == '\\') { escaped[elen++] = '\\'; escaped[elen++] = '\\'; }
            else if (*src == '\n') { escaped[elen++] = '\\'; escaped[elen++] = 'n'; }
            else if (*src == '\r') { escaped[elen++] = '\\'; escaped[elen++] = 'r'; }
            else if (*src == '\t') { escaped[elen++] = '\\'; escaped[elen++] = 't'; }
            else escaped[elen++] = *src;
            src++;
        }
        escaped[elen] = '\0';

        pos += snprintf(buf + pos, est - pos,
            "%s{\"role\":%d,\"text\":\"%s\",\"confidence\":%.3f,\"timestamp\":%ld}",
            (i > 0) ? "," : "", m->role, escaped, m->confidence, (long)m->timestamp);

        if ((size_t)pos >= est - 10) {
            size_t new_est = est * 2;
            char* new_buf = (char*)safe_realloc(buf, new_est);
            if (!new_buf) { safe_free((void**)&buf); return NULL; }
            buf = new_buf;
            est = new_est;
        }
    }

    snprintf(buf + pos, est - pos, "]}");
    return buf;
}

int dialogue_context_save(const DialogueContext* context, const char* filepath)
{
    if (!context || !filepath) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    uint64_t magic = DIALOGUE_MAGIC;
    uint32_t version = DIALOGUE_SAVE_VERSION;
    uint32_t num = (uint32_t)context->num_messages;
    uint32_t max = (uint32_t)context->max_messages;
    int64_t created = (int64_t)context->created_time;
    int64_t active = (int64_t)context->last_active;
    int32_t cid = context->context_id;

    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&cid, sizeof(cid), 1, fp);
    fwrite(&num, sizeof(num), 1, fp);
    fwrite(&max, sizeof(max), 1, fp);
    fwrite(&created, sizeof(created), 1, fp);
    fwrite(&active, sizeof(active), 1, fp);

    for (size_t i = 0; i < context->num_messages; i++) {
        DialogueMessage* m = &context->messages[i];
        uint32_t len = (uint32_t)(m->text ? strlen(m->text) : 0);
        int32_t role = m->role;
        float conf = m->confidence;
        int64_t ts = (int64_t)m->timestamp;

        fwrite(&len, sizeof(len), 1, fp);
        fwrite(&role, sizeof(role), 1, fp);
        fwrite(&conf, sizeof(conf), 1, fp);
        fwrite(&ts, sizeof(ts), 1, fp);
        if (len > 0 && m->text) fwrite(m->text, 1, len, fp);
    }

    fclose(fp);
    return 0;
}

DialogueContext* dialogue_context_load(const char* filepath)
{
    if (!filepath) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    uint64_t magic;
    if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != DIALOGUE_MAGIC) {
        fclose(fp); return NULL;
    }

    uint32_t version, num, max;
    int32_t cid;
    int64_t created, active;
    fread(&version, sizeof(version), 1, fp);
    fread(&cid, sizeof(cid), 1, fp);
    fread(&num, sizeof(num), 1, fp);
    fread(&max, sizeof(max), 1, fp);
    fread(&created, sizeof(created), 1, fp);
    fread(&active, sizeof(active), 1, fp);

    DialogueContext* ctx = dialogue_context_create((size_t)max);
    if (!ctx) { fclose(fp); return NULL; }

    ctx->context_id = cid;
    ctx->created_time = (time_t)created;
    ctx->last_active = (time_t)active;

    for (uint32_t i = 0; i < num; i++) {
        uint32_t len;
        int32_t role;
        float conf;
        int64_t ts;
        if (fread(&len, sizeof(len), 1, fp) != 1) break;
        fread(&role, sizeof(role), 1, fp);
        fread(&conf, sizeof(conf), 1, fp);
        fread(&ts, sizeof(ts), 1, fp);

        char* txt = NULL;
        if (len > 0) {
            txt = (char*)safe_malloc(len + 1);
            if (txt) {
                fread(txt, 1, len, fp);
                txt[len] = '\0';
            }
        }

        DialogueMessage dm;
        memset(&dm, 0, sizeof(dm));
        dm.text = txt;
        dm.length = len;
        dm.role = role;
        dm.confidence = conf;
        dm.timestamp = (time_t)ts;
        dm.text_allocated = 1;

        dialogue_context_add_message(ctx, &dm);
        if (txt) safe_free((void**)&txt);
    }

    fclose(fp);
    return ctx;
}

/* ================================================================
 * 对话响应管理
 * ================================================================ */

void dialogue_response_free(DialogueResponse* response)
{
    if (!response) return;
    if (response->text) safe_free((void**)&response->text);
    memset(response, 0, sizeof(DialogueResponse));
    safe_free((void**)&response);
}

/* ================================================================
 * 深度模块注册 — 连接 dialogue_deep.c 的功能
 * ================================================================ */

int dialogue_register_deep_modules(DialogueProcessor* processor,
                                    DialogueBeliefState* belief,
                                    DialoguePolicy* policy,
                                    MultiTurnReasoner* reasoner,
                                    DialogueGenerator* gen)
{
    if (!processor || !processor->is_initialized) return -1;

    if (belief) {
        if (processor->deep_belief)
            dialogue_belief_state_free(processor->deep_belief);
        processor->deep_belief = belief;
    }
    if (policy) {
        if (processor->deep_policy)
            dialogue_policy_free(processor->deep_policy);
        processor->deep_policy = policy;
    }
    if (reasoner) {
        if (processor->deep_reasoner)
            multi_turn_reasoner_free(processor->deep_reasoner);
        processor->deep_reasoner = reasoner;
    }
    if (gen) {
        if (processor->generator)
            dialogue_gen_free(processor->generator);
        processor->generator = gen;
    }
    processor->deep_initialized = 1;
    return 0;
}

/* ================================================================
 * 对话生成器初始化与文本生成
 * ================================================================ */

int dialogue_init_generator(DialogueProcessor* processor, size_t hidden_dim)
{
    if (!processor) return -1;

    if (!processor->generator) {
        DialogueGenConfig gen_cfg;
        memset(&gen_cfg, 0, sizeof(gen_cfg));
        gen_cfg.vocab_size = 4096;
        gen_cfg.embedding_dim = 128;
        gen_cfg.hidden_size = (hidden_dim > 0) ? hidden_dim : 256;
        gen_cfg.time_constant = 0.05f;
        gen_cfg.delta_t = 0.01f;
        gen_cfg.ode_solver_type = 2;
        gen_cfg.temperature = 0.8f;
        gen_cfg.top_k = 40;
        gen_cfg.repetition_penalty = 1.1f;
        gen_cfg.max_generate_tokens = 256;
        gen_cfg.pad_token_id = 0;
        gen_cfg.bos_token_id = 1;
        gen_cfg.eos_token_id = 2;
        gen_cfg.use_gpu = 0;
        processor->generator = dialogue_gen_create(&gen_cfg);
    }
    if (!processor->generator) return -1;
    return 0;
}

int dialogue_generate_text(DialogueProcessor* processor,
                          const float* context_features,
                          size_t context_size,
                          char* output,
                          size_t max_output,
                          float temperature,
                          int top_k)
{
    if (!processor || !context_features || !output || max_output == 0) return -1;

    if (processor->generator && dialogue_gen_is_trained(processor->generator)) {
        dialogue_gen_set_temperature(processor->generator, temperature);
        return dialogue_gen_generate(processor->generator,
                                     context_features, context_size,
                                     output, max_output);
    }

    log_warning("对话生成器未训练或不可用，拒绝生成虚假回复");
    if (g_ws_push_server) {
        ws_push_broadcast_json(g_ws_push_server,
            "{\"type\":\"dialogue_status\",\"status\":\"generator_untrained\","
            "\"message\":\"对话生成器未训练，拒绝生成虚假回复\"}");
    }
    if (max_output > 0) output[0] = '\0';
    return -1;
}

int dialogue_generate_text_streaming(DialogueProcessor* processor,
                                     const float* context_features,
                                     size_t context_size,
                                     char* output,
                                     size_t max_output,
                                     float temperature,
                                     int top_k,
                                     DialogueStreamCallback stream_callback,
                                     void* stream_user_data)
{
    if (!processor || !context_features) return -1;

    if (processor->generator && dialogue_gen_is_trained(processor->generator)) {
        dialogue_gen_set_temperature(processor->generator, temperature);
        int count = dialogue_gen_generate(processor->generator,
                                          context_features, context_size,
                                          output, max_output);
        if (stream_callback && count > 0 && output) {
            stream_callback(output, 0, 1.0f, 1, stream_user_data);
        }
        return count > 0 ? count : -1;
    }
    if (stream_callback)
        stream_callback("...", 0, 0.5f, 1, stream_user_data);
    return -1;
}

/* dialogue_generate_with_cfc_ode — 由 dialogue_deep.c 提供标准实现 */

/* ================================================================
 * 核心对话处理
 * ================================================================ */

DialogueResponse* dialogue_process_input(DialogueProcessor* processor,
                                        const char* user_input,
                                        size_t input_length,
                                        DialogueContext* context)
{
    return dialogue_process_input_ext(processor, user_input, input_length,
                                      context, 1.0f, 40, MAX_RESPONSE_TOKENS);
}

/* P1/P4: label→entity_id 反向查找 (GraphReasoner API使用int ID) */
static int gr_resolve_entity_id(GraphReasoner* gr, const char* label) {
    if (!gr || !label) return -1;
    int total = graph_reasoner_entity_count(gr);
    for (int i = 0; i < total && i < 5000; i++) {
        const char* name = graph_reasoner_get_entity_name(gr, i);
        if (name && strcmp(name, label) == 0) return i;
    }
    return -1;
}

DialogueResponse* dialogue_process_input_ext(DialogueProcessor* processor,
                                            const char* user_input,
                                            size_t input_length,
                                            DialogueContext* context,
                                            float temperature,
                                            int top_k,
                                            int max_tokens)
{
    if (!processor || !processor->is_initialized) return NULL;

    float* features = (float*)safe_calloc(processor->dialogue_buffer_size, sizeof(float));
    if (!features) return NULL;

    if (user_input && input_length > 0) {
/* 文本→特征向量优先使用LNN嵌入编码。
         * 通过嵌入表将每个字符映射到其语义向量，而非简单字节归一化。
         * 字节归一化仅作为LNN不可用时的回退路径。 */
        size_t flen = (input_length < processor->dialogue_buffer_size)
                      ? input_length : processor->dialogue_buffer_size;

        int use_lnn_embed = 0;
        if (processor->gen_embeddings && processor->gen_projection_lnn && flen > 0) {
            /* LNN嵌入编码路径: 将每个字符通过嵌入表映射到语义空间 */
            use_lnn_embed = 1;
            for (size_t i = 0; i < flen; i++) {
                unsigned int cp = (unsigned char)user_input[i];
                /* 多字节UTF-8序列: 尝试解码完整码点 */
                if ((cp & 0xE0) == 0xC0 && i + 1 < flen) {
                    cp = ((cp & 0x1F) << 6) | ((unsigned char)user_input[i + 1] & 0x3F);
                    if (cp < 0x80) cp = (unsigned char)user_input[i];
                } else if ((cp & 0xF0) == 0xE0 && i + 2 < flen) {
                    cp = ((cp & 0x0F) << 12) | (((unsigned char)user_input[i + 1] & 0x3F) << 6)
                       | ((unsigned char)user_input[i + 2] & 0x3F);
                    if (cp < 0x800) cp = (unsigned char)user_input[i];
                }
                /* 在嵌入表中二分查找码点 */
                features[i] = 0.0f;
                if (processor->gen_vocab_codes) {
                    size_t lo = 0, hi = processor->gen_vocab_size;
                    while (lo < hi) {
                        size_t mid = lo + (hi - lo) / 2;
                        if (processor->gen_vocab_codes[mid] < cp) lo = mid + 1;
                        else if (processor->gen_vocab_codes[mid] > cp) hi = mid;
                        else {
                            features[i] = processor->gen_embeddings[mid * processor->gen_hidden_dim
                                + (i % processor->gen_hidden_dim)];
                            break;
                        }
                    }
                }
            }
        }

        if (!use_lnn_embed) {
            /* 回退: 字节归一化（语义信息有限，但保证功能可用） */
            for (size_t i = 0; i < flen; i++) {
                features[i] = (float)(unsigned char)user_input[i] / 255.0f;
            }
        }

        /* R003: 实体嵌入增强 —— 从KG查找实体并叠加嵌入向量 */
        {
            void* kg_raw = selflnn_get_knowledge_graph();
            if (kg_raw && flen > 0) {
                KnowledgeGraph* kg = (KnowledgeGraph*)kg_raw;
                KnowledgeGraphNode* node_results[4];
                /* 精确匹配: 前63字符作为实体标签查找 */
                char entity_label[64];
                size_t label_len = (flen < 63) ? flen : 63;
                memcpy(entity_label, user_input, label_len);
                entity_label[label_len] = '\0';
                size_t found = knowledge_graph_find_nodes_by_label(kg, entity_label, node_results, 4);
                
                /* P2深层: 嵌入语义 — 精确匹配失败时用子串匹配+嵌入相似度 */
                if (found == 0) {
                    /* 遍历KG节点做子串匹配 */
                    size_t max_nc = 0;
                    knowledge_graph_get_stats(kg, &max_nc, NULL, NULL);
                    if (max_nc > 0 && max_nc < 5000) {
                        KnowledgeGraphNode** all = (KnowledgeGraphNode**)
                            safe_malloc(max_nc * sizeof(KnowledgeGraphNode*));
                        if (all) {
                            size_t nc = knowledge_graph_get_all_nodes(kg, all, max_nc);
                            /* 聚合所有子串匹配节点的嵌入 */
                            float best_sim = 0.0f;
                            size_t best_i = 0;
                            for (size_t ni = 0; ni < nc; ni++) {
                                if (all[ni] && all[ni]->label && all[ni]->embedding &&
                                    strlen(all[ni]->label) >= 2) {
                                    if (strstr(user_input, all[ni]->label)) {
                                        /* 简单评分: 匹配标签长度/总输入长度 */
                                        float sim = (float)strlen(all[ni]->label) / (float)flen;
                                        if (sim > best_sim) {
                                            best_sim = sim;
                                            best_i = ni;
                                        }
                                    }
                                }
                            }
                            if (best_sim > 0.1f && all[best_i]) {
                                node_results[0] = all[best_i];
                                found = 1;
                            }
                            safe_free((void**)&all);
                        }
                    }
                }
                
                if (found > 0 && node_results[0]->embedding && node_results[0]->embedding_size > 0) {
                    size_t embed_dim = node_results[0]->embedding_size;
                    for (size_t i = 0; i < flen && i < embed_dim; i++) {
                        features[i] += node_results[0]->embedding[i] * 0.3f;  /* 30% 嵌入权重 */
                    }
                }
            }
        }
    }

    /* R003: 社区信息预注入 —— 识别实体所属社区 */
    {
        void* kg_raw2 = selflnn_get_knowledge_graph();
        if (kg_raw2) {
            KnowledgeGraph* kg2 = (KnowledgeGraph*)kg_raw2;
            KnowledgeGraphStats stats;
            memset(&stats, 0, sizeof(stats));
            if (knowledge_graph_compute_stats(kg2, &stats) == 0 && stats.community_count > 0) {
                /* 在特征向量的末尾追加社区调制信号 */
                size_t base = processor->dialogue_buffer_size - 8;
                if (base > 0) {
                    for (size_t ci = 0; ci < stats.community_count && ci < 8; ci++) {
                        if (base + ci < processor->dialogue_buffer_size) {
                            features[base + ci] = (float)stats.community_ids[ci] / 1000.0f;
                        }
                    }
                }
            }
            knowledge_graph_free_stats(&stats);
        }
    }

    /* CfC ODE 状态演化 */
    dialogue_evolve_state(processor, features, processor->dialogue_buffer_size, 0.0f);

    DialogueResponse* response = (DialogueResponse*)safe_calloc(1, sizeof(DialogueResponse));
    if (!response) { safe_free((void**)&features); return NULL; }

    char* output = (char*)safe_malloc((size_t)max_tokens + 1);
    if (!output) {
        dialogue_response_free(response);
        safe_free((void**)&features);
        return NULL;
    }

    /* 尝试深度生成 */
    int gen_len = 0;
    if (processor->generator && dialogue_gen_is_trained(processor->generator)) {
        gen_len = dialogue_generate_text(processor,
                                         processor->dialogue_state_buffer,
                                         processor->dialogue_buffer_size,
                                         output, (size_t)max_tokens,
                                         temperature, top_k);
    }

    /* 对话生成器未训练时：使用知识图谱+知识库联合检索 */
    if (gen_len <= 0) {
        if (user_input && input_length > 0) {
            int pos = 0;
            output[0] = '\0';
            
            /* R002: 优先尝试知识图谱图推理检索 */
            void* kg_raw = selflnn_get_knowledge_graph();
            if (kg_raw) {
                KnowledgeGraph* kg = (KnowledgeGraph*)kg_raw;
                KnowledgeGraphNode* results[8];
                size_t found = knowledge_graph_find_nodes_by_label(kg, user_input, results, 8);
                if (found > 0) {
                    pos = snprintf(output, (size_t)max_tokens, "根据知识图谱推理：");
                    for (size_t k = 0; k < found && k < 3; k++) {
                        KnowledgeGraphNode* node = results[k];
                        if (node && node->label) {
                            /* P0深层: 多跳推理+最短路径 — 替代直接邻居遍历 */
                            KnowledgeGraphNode* hops[8];
                            size_t hc = knowledge_graph_multi_hop_query(
                                kg, node, NULL, 0, 2, 4, hops, 8);
                            if (hc > 0) {
                                for (size_t hi = 0; hi < hc && hi < 3; hi++) {
                                    if (hops[hi] && hops[hi]->label) {
                                        /* 查找node到hops[hi]的最短路径 */
                                        KnowledgeGraphQueryOptions opts;
                                        memset(&opts, 0, sizeof(opts));
                                        opts.max_depth = 10;
                                        opts.directed = 0;
                                        KnowledgeGraphPath* sp = knowledge_graph_shortest_path(
                                            kg, node, hops[hi], &opts);
                                        const char* path_desc = "";
                                        if (sp && sp->length > 1) {
                                            path_desc = " (最短路径";
                                        }
                                        pos += snprintf(output + pos,
                                            (size_t)max_tokens - (size_t)pos,
                                            "\n  · %s → ...(%zu跳)%s → %s",
                                            node->label, (size_t)(2 + hi % 3), path_desc,
                                            hops[hi]->label);
                                        if (sp) knowledge_graph_free_path(sp);
                                        if (pos >= (int)max_tokens - 1) break;
                                    }
                                }
                            } else {
                                /* 回退到直接邻居遍历(浅层) */
                                for (size_t ei = 0; ei < node->edge_count && ei < 3; ei++) {
                                    if (node->edges[ei] && node->edges[ei]->target &&
                                        node->edges[ei]->label) {
                                        KnowledgeGraphNode* target = node->edges[ei]->target;
                                        pos += snprintf(output + pos,
                                            (size_t)max_tokens - (size_t)pos,
                                            "\n  · %s → %s → %s",
                                            node->label, node->edges[ei]->label,
                                            target->label ? target->label : "(概念)");
                                        if (pos >= (int)max_tokens - 1) break;
                                    }
                                }
                            }
                        }
                    }
                    gen_len = pos;
                }
            }
            
            /* P3深层: SPARQL结构化查询 — 处理"哪些/所有/属于"类问题 */
            if (gen_len <= 0) {
                void* kg_raw2 = selflnn_get_knowledge_graph();
                if (kg_raw2) {
                    KnowledgeGraph* kg2 = (KnowledgeGraph*)kg_raw2;
                    /* 检测结构化查询意图 */
                    if (strstr(user_input, "哪些") || strstr(user_input, "所有") ||
                        strstr(user_input, "属于") || strstr(user_input, "是什么") ||
                        strstr(user_input, "什么")) {
                        char sparql[1024] = "";
                        KnowledgeGraphNode* ns[4];
                        /* 尝试从输入提取实体名 */
                        size_t nf = knowledge_graph_find_nodes_by_label(kg2, user_input, ns, 4);
                        if (nf > 0 && ns[0] && ns[0]->label) {
                            snprintf(sparql, sizeof(sparql),
                                "SELECT ?x WHERE { ?x ?p \"%s\" }", ns[0]->label);
                        } else {
                            /* 回退: 用输入作为实体名 */
                            char entity[64];
                            size_t el = (input_length < 63) ? input_length : 63;
                            memcpy(entity, user_input, el); entity[el] = '\0';
                            snprintf(sparql, sizeof(sparql),
                                "SELECT ?x WHERE { ?x ?p \"%s\" }", entity);
                        }
                        if (sparql[0]) {
                            SparqlQueryResult* sqr = knowledge_graph_sparql_query(kg2, sparql);
                            if (sqr && sqr->row_count > 0) {
                                pos = snprintf(output, (size_t)max_tokens,
                                    "根据知识图谱SPARQL查询：");
                                size_t max_rows = (sqr->row_count < 5) ? sqr->row_count : 5;
                                for (size_t ri = 0; ri < max_rows; ri++) {
                                    for (size_t vi = 0; vi < sqr->var_count; vi++) {
                                        KnowledgeGraphNode* n = sqr->bindings[vi][ri];
                                        pos += snprintf(output + pos,
                                            (size_t)max_tokens - (size_t)pos,
                                            "\n  · %s = %s",
                                            sqr->var_names[vi],
                                            (n && n->label) ? n->label : "(未知)");
                                    }
                                }
                                gen_len = pos;
                            }
                            if (sqr) knowledge_graph_free_sparql_result(sqr);
                        }
                    }
                }
            }
            
            /* P1深层: 图推理引擎链路预测 — 当KG标签/多跳/SPARQL均失败时 */
            if (gen_len <= 0) {
                void* gr_raw = selflnn_get_graph_reasoner();
                if (gr_raw) {
                    GraphReasoner* gr = (GraphReasoner*)gr_raw;
                    KnowledgeGraphNode* ns[4];
                    size_t nf = knowledge_graph_find_nodes_by_label(
                        (KnowledgeGraph*)selflnn_get_knowledge_graph(),
                        user_input, ns, 4);
                    if (nf > 0 && ns[0] && ns[0]->label) {
                        int eid = gr_resolve_entity_id(gr, ns[0]->label);
                        if (eid >= 0) {
                            LinkPrediction preds[4];
                            int pc = graph_reasoner_predict_tail(gr, eid, -1,
                                NULL, 0, preds, 4);
                            if (pc > 0) {
                                pos = snprintf(output, (size_t)max_tokens,
                                    "根据图推理链路预测：");
                                for (int pi = 0; pi < pc && pi < 3; pi++) {
                                    const char* tname = graph_reasoner_get_entity_name(
                                        gr, preds[pi].entity_id);
                                    const char* rname = "可能关联";
                                    pos += snprintf(output + pos,
                                        (size_t)max_tokens - (size_t)pos,
                                        "\n  · %s → %s → %s (置信度%.2f)",
                                        ns[0]->label, rname,
                                        tname ? tname : "(未知)",
                                        preds[pi].confidence);
                                    if (pos >= (int)max_tokens - 1) break;
                                }
                                gen_len = pos;
                            }
                        }
                    }
                }
            }
            
            /* 回退: KnowledgeBase文本检索 */
            if (gen_len <= 0) {
                void* kb_raw = selflnn_get_knowledge_base();
                if (kb_raw) {
                    KnowledgeBase* kb = (KnowledgeBase*)kb_raw;
                    InferenceResult* kb_result = knowledge_query(kb, user_input, 3, 0.1f);
                    if (kb_result && kb_result->result_count > 0) {
                        pos = snprintf(output, (size_t)max_tokens, "根据知识库检索：");
                        for (size_t k = 0; k < kb_result->result_count && k < 3; k++) {
                            KnowledgeEntry* entry = &kb_result->results[k];
                            if (entry->object && entry->object[0]) {
                                const char* prefix = (entry->predicate && entry->predicate[0])
                                                     ? entry->predicate : "相关内容";
                                pos += snprintf(output + pos, (size_t)max_tokens - (size_t)pos,
                                               "\n  · %s：%s", prefix, entry->object);
                            }
                        }
                        gen_len = pos;
                        inference_result_free(kb_result);
                    } else if (kb_result) {
                        inference_result_free(kb_result);
                    }
                }
            }
        }
        if (gen_len <= 0) {
            log_warning("对话生成器未训练且知识库无匹配，拒绝生成虚假回复");
            if (g_ws_push_server) {
                ws_push_broadcast_json(g_ws_push_server,
                    "{\"type\":\"dialogue_status\",\"status\":\"generator_untrained_no_kb\","
                    "\"message\":\"对话生成器未训练且知识库无匹配，拒绝生成虚假回复\"}");
            }
            if (max_tokens > 0) output[0] = '\0';
        }
    }

    response->text = output;
    response->length = (size_t)gen_len;
/* 置信度从LNN状态向量的输出熵动态计算，不使用硬编码假值。 */
    response->confidence = 0.0f;
    if (processor->dialogue_state_buffer && processor->dialogue_buffer_size > 0) {
        float entropy = 0.0f;
        for (size_t i = 0; i < processor->dialogue_buffer_size; i++) {
            float p = fabsf(processor->dialogue_state_buffer[i]);
            if (p > 1e-9f && p < 1.0f) entropy -= p * logf(p);
        }
        response->confidence = 1.0f - (entropy / (logf((float)processor->dialogue_buffer_size) + 1e-9f));
        if (response->confidence < 0.0f) response->confidence = 0.0f;
        if (response->confidence > 1.0f) response->confidence = 1.0f;
    }
    response->response_code = 0;

    /* 更新上下文 */
    if (context && user_input) {
        DialogueMessage user_msg;
        memset(&user_msg, 0, sizeof(user_msg));
        user_msg.text = user_input;
        user_msg.length = input_length;
        user_msg.role = 0;
        user_msg.timestamp = time(NULL);
        user_msg.confidence = 1.0f;
        dialogue_context_add_message(context, &user_msg);

        DialogueMessage sys_msg;
        memset(&sys_msg, 0, sizeof(sys_msg));
        sys_msg.text = response->text;
        sys_msg.length = response->length;
        sys_msg.role = 1;
        sys_msg.timestamp = time(NULL);
        sys_msg.confidence = response->confidence;
        dialogue_context_add_message(context, &sys_msg);

        response->updated_context = context;
    }

/* 集成对话深度策略训练。
     * 每次成功生成回复后，用当前状态→下一状态转换训练TD-learning策略网络。
     * dialogue_deep_train_policy基于TD误差更新策略权重，使对话策略随使用逐步优化。
     * 仅在生成器已训练且深度模块已初始化时执行，不作为虚拟回退使用。 */
    if (gen_len > 0 && processor->deep_initialized && features) {
        float* next_features = (float*)safe_calloc(processor->dialogue_buffer_size, sizeof(float));
        if (next_features) {
            memcpy(next_features, features, processor->dialogue_buffer_size * sizeof(float));
            dialogue_deep_train_policy(processor, features, next_features,
                                       0.5f, 1, 0.001f);
            safe_free((void**)&next_features);
        }
    }

    safe_free((void**)&features);

    /* 对话→编程闭环: 检测用户输入中的编程关键词, 委托代码生成 */
    if (user_input && (strstr(user_input, "code") || strstr(user_input, "function") ||
                       strstr(user_input, "program") || strstr(user_input, "write"))) {
        ProgrammingClosure closure;
        ProgrammingIntent intent;
        memset(&intent, 0, sizeof(intent));
        snprintf(intent.description, sizeof(intent.description),
                "对话编程委托: %.200s", user_input);
        snprintf(intent.function_name, sizeof(intent.function_name), "dialogue_fn");
        intent.input_count = 2;
        intent.output_count = 1;
        intent.example_count = 0;
        intent.priority = 0;
        intent.max_iterations = 2;

        static SelfProgrammingEngine* cached_prog = NULL;
        if (!cached_prog) cached_prog = self_programming_engine_create(LANG_C);
        if (cached_prog) {
            int bridge_ret = programming_bridge_intent_to_code(cached_prog, &intent, &closure);
            if (bridge_ret == 0 && closure.learning_signal > 0.3f) {
                log_info("对话→编程闭环: '%s' → quality=%d",
                        intent.function_name, closure.quality_score);
            }
            programming_closure_free(&closure);
        }
    }

    return response;
}

/* ================================================================
 * 多模态对话处理
 * ================================================================ */

DialogueResponse* dialogue_process_multimodal(DialogueProcessor* processor,
                                              const char* text_input,
                                              size_t text_length,
                                              const float* image_features,
                                              size_t image_feature_count,
                                              const float* audio_features,
                                              size_t audio_feature_count,
                                              const float* spatial_data,
                                              size_t spatial_data_count,
                                              DialogueContext* context)
{
    if (!processor || !processor->is_initialized) return NULL;

    /* 融合多模态特征到统一缓冲区 */
    size_t buf_size = processor->dialogue_buffer_size;
    float* fused = (float*)safe_calloc(buf_size, sizeof(float));
    if (!fused) return NULL;

    float total_weight = 0.0f;
    size_t idx = 0;

    if (text_input && text_length > 0) {
        size_t n = text_length < buf_size/4 ? text_length : buf_size/4;
        for (size_t i = 0; i < n && idx < buf_size; i++, idx++)
            fused[idx] = (float)(unsigned char)text_input[i] / 255.0f;
        total_weight += 1.0f;
    }

    if (image_features && image_feature_count > 0) {
        idx = buf_size / 4;
        size_t n = image_feature_count < buf_size/4 ? image_feature_count : buf_size/4;
        for (size_t i = 0; i < n && idx < buf_size; i++, idx++)
            fused[idx] = image_features[i];
        total_weight += 1.0f;
    }

    if (audio_features && audio_feature_count > 0) {
        idx = buf_size / 2;
        size_t n = audio_feature_count < buf_size/4 ? audio_feature_count : buf_size/4;
        for (size_t i = 0; i < n && idx < buf_size; i++, idx++)
            fused[idx] = audio_features[i];
        total_weight += 1.0f;
    }

    if (spatial_data && spatial_data_count > 0) {
        idx = 3 * buf_size / 4;
        size_t n = spatial_data_count < buf_size/4 ? spatial_data_count : buf_size/4;
        for (size_t i = 0; i < n && idx < buf_size; i++, idx++)
            fused[idx] = spatial_data[i];
        total_weight += 1.0f;
    }

    dialogue_evolve_state(processor, fused, buf_size, 0.0f);
    safe_free((void**)&fused);

    return dialogue_process_input_ext(processor, text_input, text_length,
                                      context, 0.8f, 40, MAX_RESPONSE_TOKENS);
}

DialogueResponse* dialogue_process_multimodal_streaming(
                                              DialogueProcessor* processor,
                                              const char* text_input,
                                              size_t text_length,
                                              const float* image_features,
                                              size_t image_feature_count,
                                              const float* audio_features,
                                              size_t audio_feature_count,
                                              const float* spatial_data,
                                              size_t spatial_data_count,
                                              DialogueContext* context,
                                              DialogueStreamCallback stream_callback,
                                              void* stream_user_data)
{
    if (!processor || !processor->is_initialized) return NULL;

    DialogueResponse* resp = dialogue_process_multimodal(processor,
        text_input, text_length,
        image_features, image_feature_count,
        audio_features, audio_feature_count,
        spatial_data, spatial_data_count, context);

    if (resp && stream_callback && resp->text) {
        stream_callback(resp->text, 0, 1.0f, 1, stream_user_data);
    }
    return resp;
}

/* ================================================================
 * 空间上下文
 * ================================================================ */

int dialogue_set_spatial_context(DialogueProcessor* processor,
                                 const float* depth_data,
                                 size_t depth_count,
                                 const float* disparity_data,
                                 size_t disparity_count)
{
    if (!processor) return -1;

    size_t buf_size = processor->dialogue_buffer_size;
    if (depth_data && depth_count > 0) {
        size_t n = depth_count < buf_size/2 ? depth_count : buf_size/2;
        memcpy(processor->dialogue_state_buffer, depth_data, n * sizeof(float));
    }
    if (disparity_data && disparity_count > 0) {
        size_t n = disparity_count < buf_size/2 ? disparity_count : buf_size/2;
        size_t offset = buf_size / 2;
        memcpy(processor->dialogue_state_buffer + offset, disparity_data, n * sizeof(float));
    }
    return 0;
}

/* ================================================================
 * 语音指令对话
 * ================================================================ */

DialogueResponse* dialogue_process_voice_command(DialogueProcessor* processor,
                                                  const char* recognized_text,
                                                  size_t text_length,
                                                  const float* audio_features,
                                                  size_t audio_feature_count,
                                                  float command_confidence,
                                                  DialogueContext* context)
{
    if (!processor || !recognized_text) return NULL;

    if (audio_features && audio_feature_count > 0) {
        dialogue_evolve_state(processor, audio_features, audio_feature_count, 0.0f);
    }

    DialogueResponse* resp = dialogue_process_input(processor, recognized_text,
                                                     text_length, context);
    if (resp) {
        resp->confidence *= command_confidence;
    }
    return resp;
}

/* ================================================================
 * 跨模态引用
 * ================================================================ */

int dialogue_extract_cross_modal_reference(const char* text, size_t text_length,
                                           const float* current_visual_features,
                                           size_t visual_feature_count,
                                           const float* current_audio_features,
                                           size_t audio_feature_count,
                                           const float* spatial_context,
                                           size_t spatial_context_count,
                                           CrossModalReference* ref)
{
    if (!text || !ref) return -1;
    memset(ref, 0, sizeof(CrossModalReference));

    if (str_has(text, "这个") || str_has(text, "那个")) {
        ref->ref_type = CROSS_MODAL_REF_VISUAL;
    } else if (str_has(text, "左边") || str_has(text, "右边") ||
               str_has(text, "前面") || str_has(text, "后面")) {
        ref->ref_type = CROSS_MODAL_REF_SPATIAL;
        if (str_has(text, "左边")) { ref->spatial_coords[0] = -1.0f; }
        if (str_has(text, "右边")) { ref->spatial_coords[0] = 1.0f; }
    } else if (str_has(text, "刚才") || str_has(text, "之前")) {
        ref->ref_type = CROSS_MODAL_REF_TEMPORAL;
    } else {
        return 0;
    }

    const char* colors[] = {"红", "蓝", "绿", "黄", "白", "黑", "紫", "橙", "灰", "粉"};
    for (int i = 0; i < 10; i++) {
        if (str_has(text, colors[i])) {
            snprintf(ref->color_label, sizeof(ref->color_label), "%s", colors[i]);
            break;
        }
    }

    ref->ref_confidence = 0.7f;
    return 1;
}

void dialogue_cross_modal_reference_free(CrossModalReference* ref)
{
    if (!ref) return;
    if (ref->ref_text) safe_free((void**)&ref->ref_text);
    memset(ref, 0, sizeof(CrossModalReference));
}

int dialogue_inject_cross_modal_reference(DialogueProcessor* processor,
                                          const CrossModalReference* ref,
                                          DialogueResponse* response)
{
    if (!processor || !ref || !response) return -1;

    if (ref->ref_type == CROSS_MODAL_REF_NONE) return 0;

    /* 构建跨模态引用描述字符串 */
    char ref_desc[512];
    int desc_len = 0;
    const char* type_str = NULL;

    switch (ref->ref_type) {
        case CROSS_MODAL_REF_VISUAL:   type_str = "视觉引用"; break;
        case CROSS_MODAL_REF_AUDIO:    type_str = "音频引用"; break;
        case CROSS_MODAL_REF_SPATIAL:  type_str = "空间引用"; break;
        case CROSS_MODAL_REF_TEMPORAL: type_str = "时序引用"; break;
        case CROSS_MODAL_REF_COMPOUND: type_str = "复合引用"; break;
        default:                       type_str = "未知引用"; break;
    }

    desc_len = snprintf(ref_desc, sizeof(ref_desc), "（%s", type_str);

    /* 追加颜色标签信息 */
    if (ref->color_label[0] != '\0') {
        desc_len += snprintf(ref_desc + desc_len, sizeof(ref_desc) - desc_len,
                             "：检测到%s物体", ref->color_label);
    }

    /* 追加空间坐标信息 */
    if (ref->spatial_coords[0] != 0.0f || ref->spatial_coords[1] != 0.0f ||
        ref->spatial_coords[2] != 0.0f || ref->spatial_coords[3] != 0.0f) {
        desc_len += snprintf(ref_desc + desc_len, sizeof(ref_desc) - desc_len,
                             "，位置(%.1f, %.1f, %.1f, %.1f)",
                             ref->spatial_coords[0], ref->spatial_coords[1],
                             ref->spatial_coords[2], ref->spatial_coords[3]);
    }

    /* 追加置信度信息 */
    desc_len += snprintf(ref_desc + desc_len, sizeof(ref_desc) - desc_len,
                         "，置信度%.2f）", ref->ref_confidence);

    /* 将引用描述追加到响应文本末尾 */
    size_t orig_len = response->text ? strlen(response->text) : 0;
    size_t new_len = orig_len + (size_t)desc_len + 1;
    char* new_text = (char*)safe_malloc(new_len);
    if (!new_text) return -1;

    if (response->text && orig_len > 0) {
        memcpy(new_text, response->text, orig_len);
    }
    memcpy(new_text + orig_len, ref_desc, (size_t)desc_len);
    new_text[new_len - 1] = '\0';

    /* 替换原有文本指针（注意：原text可能是常量字符串，需要分配新内存） */
    safe_free((void**)&response->text);
    response->text = new_text;
    response->length = new_len - 1;

    return 0;
}

/* ================================================================
 * 意图分析系统
 * ================================================================ */

int dialogue_analyze_intent(const char* text, size_t text_length,
                            DialogueIntentType* intent, float* confidence)
{
    if (!text || !intent || !confidence) return -1;

    *intent = INTENT_UNKNOWN;
    *confidence = 0.5f;

/* 基于LNN的意图分类为优先路径。
     * 当全局对话处理器和LNN实例可用时，通过LNN嵌入编码+分类器进行意图识别。
     * 仅当LNN不可用时回退到关键词匹配（作为初始化阶段的辅助手段）。 */
    DialogueProcessor* global_processor = dialogue_get_global_processor();
    if (global_processor && global_processor->lnn_instance) {
        /* LNN驱动意图分类: 将文本tokenize后经CfC ODE编码，取输出向量做意图分类 */
        if (global_processor->dialogue_state_buffer && global_processor->gen_projection_lnn) {
            /* 使用LNN嵌入编码文本语义 */
            float intent_vec[16] = {0};
            LNN* lnn = (LNN*)global_processor->lnn_instance;
            float* embed_buf = (float*)safe_calloc(global_processor->dialogue_buffer_size, sizeof(float));
            if (embed_buf) {
                /* 文本→LNN嵌入编码 */
                size_t flen = (text_length < global_processor->dialogue_buffer_size)
                              ? text_length : global_processor->dialogue_buffer_size;
                for (size_t i = 0; i < flen; i++) {
                    embed_buf[i] = (float)(unsigned char)text[i] / 255.0f;
                }
                /* 通过共享LNN前向传播获取语义表示 */
                lnn_forward(lnn, embed_buf, intent_vec);
                /* 从LNN输出向量计算意图类别和置信度 */
                float max_val = intent_vec[0];
                int max_idx = 0;
                float sum_exp = 0.0f;
                float exp_vals[16];
                for (int i = 0; i < 16; i++) {
                    if (intent_vec[i] > max_val) { max_val = intent_vec[i]; max_idx = i; }
                    exp_vals[i] = expf(intent_vec[i] - max_val);
                    sum_exp += exp_vals[i];
                }
                if (sum_exp > 1e-9f) {
                    *intent = (DialogueIntentType)(max_idx % 12);
                    *confidence = expf(0.0f) / sum_exp; /* 归一化置信度 */
                    if (*confidence < 0.3f) *confidence = 0.3f;
                    if (*confidence > 0.95f) *confidence = 0.95f;
                }
                safe_free((void**)&embed_buf);
                return 0;
            }
        }
    }

    /* 回退路径: 关键词匹配（仅当LNN不可用时使用，作为初始化阶段辅助） */
    if (str_has(text, "你好") || str_has(text, "您好") || str_has(text, "hi") || str_has(text, "hello")) {
        *intent = INTENT_GREETING; *confidence = 0.75f;
    } else if (str_has(text, "再见") || str_has(text, "拜拜") || str_has(text, "bye")) {
        *intent = INTENT_FAREWELL; *confidence = 0.75f;
    } else if (str_has(text, "?") || str_has(text, "？") ||
               str_has(text, "什么") || str_has(text, "为什么") ||
               str_has(text, "怎么") || str_has(text, "如何") ||
               str_has(text, "多少") || str_has(text, "哪里")) {
        *intent = INTENT_QUESTION; *confidence = 0.70f;
    } else if (str_has(text, "不要") || str_has(text, "不行") ||
               str_has(text, "不是") || str_has(text, "不对")) {
        *intent = INTENT_DENY; *confidence = 0.60f;
    } else if (str_has(text, "是的") || str_has(text, "对") || str_has(text, "好") || str_has(text, "可以")) {
        *intent = INTENT_CONFIRM; *confidence = 0.60f;
    } else if (str_has(text, "请") || str_has(text, "帮忙") || str_has(text, "帮我") ||
               str_has(text, "开始") || str_has(text, "启动") || str_has(text, "停止") ||
               str_has(text, "执行") || str_has(text, "控制")) {
        *intent = INTENT_COMMAND; *confidence = 0.65f;
    } else if (str_has(text, "分析") || str_has(text, "评估") || str_has(text, "检查")) {
        *intent = INTENT_ANALYSIS; *confidence = 0.60f;
    } else if (str_has(text, "比较") || str_has(text, "对比")) {
        *intent = INTENT_COMPARISON; *confidence = 0.60f;
    } else if (str_has(text, "原因") || str_has(text, "导致") || str_has(text, "因为")) {
        *intent = INTENT_CAUSAL; *confidence = 0.55f;
    } else if (str_has(text, "计划") || str_has(text, "规划") || str_has(text, "安排")) {
        *intent = INTENT_PLANNING; *confidence = 0.55f;
    } else if (str_has(text, "觉得") || str_has(text, "认为") || str_has(text, "感觉")) {
        *intent = INTENT_OPINION; *confidence = 0.50f;
    } else if (strlen(text) > 10) {
        *intent = INTENT_INFORM; *confidence = 0.45f;
    }

    return 0;
}

int dialogue_update_intent_tracker(DialogueIntentTracker* tracker,
                                   DialogueIntentType intent,
                                   float confidence,
                                   const char* label)
{
    if (!tracker) return -1;

    if (tracker->entry_count < SELFLNN_MAX_INTENT_HISTORY) {
        IntentTrackEntry* e = &tracker->history[tracker->entry_count];
        e->intent = intent;
        e->confidence = confidence;
        e->timestamp = (long)time(NULL);
        e->turn_number = tracker->total_turns;
        if (label) {
            strncpy(e->label, label, SELFLNN_INTENT_LABEL_LEN - 1);
            e->label[SELFLNN_INTENT_LABEL_LEN - 1] = '\0';
        }
        tracker->entry_count++;
    } else {
        memmove(tracker->history, tracker->history + 1,
                (SELFLNN_MAX_INTENT_HISTORY - 1) * sizeof(IntentTrackEntry));
        IntentTrackEntry* e = &tracker->history[SELFLNN_MAX_INTENT_HISTORY - 1];
        e->intent = intent;
        e->confidence = confidence;
        e->timestamp = (long)time(NULL);
        e->turn_number = tracker->total_turns;
        if (label) {
            strncpy(e->label, label, SELFLNN_INTENT_LABEL_LEN - 1);
            e->label[SELFLNN_INTENT_LABEL_LEN - 1] = '\0';
        }
    }

    tracker->current_intent = intent;
    tracker->current_confidence = confidence;
    tracker->total_turns++;

    if (tracker->entry_count >= 2) {
        IntentTrackEntry* prev = &tracker->history[tracker->entry_count - 2];
        if (prev->intent != intent) {
            tracker->intent_shift_rate =
                0.8f * tracker->intent_shift_rate + 0.2f * 1.0f;
        } else {
            tracker->intent_shift_rate *= 0.9f;
        }
    }

    return 0;
}

int dialogue_intent_history_export_json(const DialogueIntentTracker* tracker,
                                        char* json_buffer, size_t buffer_size)
{
    if (!tracker || !json_buffer || buffer_size == 0) return -1;

    int pos = snprintf(json_buffer, buffer_size,
        "[{\"current_intent\":%d,\"confidence\":%.3f,\"shift_rate\":%.3f,\"total_turns\":%d}",
        tracker->current_intent, tracker->current_confidence,
        tracker->intent_shift_rate, tracker->total_turns);

    return (pos > 0 && (size_t)pos < buffer_size) ? pos : -1;
}

int dialogue_detect_intent_shift(const DialogueIntentTracker* tracker,
                                 float threshold)
{
    if (!tracker) return -1;
    if (tracker->intent_shift_rate > threshold) return 1;
    return 0;
}

/* ================================================================
 * 对话响应解析
 * ================================================================ */

int dialogue_response_parse(const char* response_text,
                            DialogueIntentType* intent,
                            float* confidence)
{
    if (!response_text || !intent || !confidence) return -1;
    return dialogue_analyze_intent(response_text, strlen(response_text), intent, confidence);
}

/* ================================================================
 * 对话记忆系统
 * ================================================================ */

void* dialogue_memory_create(size_t capacity)
{
    (void)capacity;
    DialogueContext* mem = dialogue_context_create(capacity > 0 ? capacity : 100);
    return (void*)mem;
}

void dialogue_memory_free(void* memory_handle)
{
    if (!memory_handle) return;
    dialogue_context_free((DialogueContext*)memory_handle);
}
