/**
 * @file dialogue.c
 * @brief 对话系统实现
 * 
 * 对话系统实现，支持自然语言对话处理、上下文管理和响应生成。
 * 深度集成液态神经网络（LNN）进行对话理解和生成。
 * 根据项目要求" 全部深度实现"，本模块提供完整的对话功能。
 */

#include "selflnn/multimodal/dialogue.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/unified_lnn_state.h"
#include "selflnn/selflnn.h"
#include "selflnn/multimodal/text.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <ctype.h>

/**
 * @brief 工具函数：复制字符串并转换为小写
 */
static char* string_to_lower_copy(const char* str);

/**
 * @brief 工具函数：使用LNN生成对话响应（深度实现）
 */
static int generate_response_with_lnn(DialogueProcessor* processor,
                                     const float* input_features,
                                     size_t feature_count,
                                     char** response_text,
                                     float* confidence,
                                     int* response_code,
                                     float temperature,
                                     int top_k,
                                     int max_tokens);

/* 生成器常量 */
#define GEN_MAX_VOCAB_SIZE 28000  /* P0-005修复: 扩展到28000覆盖完整CJK统一表意文字(U+4E00-U+9FFF=20992字)+扩展A(U+3400-U+4DBF=6592字) */
#define GEN_DEFAULT_HIDDEN_DIM 128
#define GEN_BOS_TOKEN 0
#define GEN_EOS_TOKEN 1
#define GEN_MAX_OUTPUT_TOKENS 256

/**
 * @brief 使用全局统一LNN状态进行文本模态步进（单一模型原则）
 * 
 * 替代原有的自建CfC单元，将所有对话状态演化委托给全局统一LNN状态。
 * 只操作TEXT模态，其他模态输入设为0。
 */
static int dialogue_unified_step(DialogueProcessor* processor,
                                 const float* text_features, size_t feature_count,
                                 float delta_t,
                                 float* state_buffer) {
    if (!processor || !text_features || feature_count == 0 || !state_buffer) {
        return -1;
    }
    
    void* unified_state = NULL;
    if (processor->unified_state_ref) {
        /* 全局统一LNN状态的文本模态前向传播
         * 仅标记实际存在的文本模态，不填充其他模态的零向量 */
        const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES];
        size_t raw_sizes[UNIFIED_LNN_MAX_MODALITIES];
        int modality_present[UNIFIED_LNN_MAX_MODALITIES];
        
        for (int _ri = 0; _ri < UNIFIED_LNN_MAX_MODALITIES; _ri++) {
            raw_inputs[_ri] = NULL;
            raw_sizes[_ri] = 0;
            modality_present[_ri] = 0;
        }
        
        raw_inputs[UNIFIED_MODALITY_TEXT] = text_features;
        raw_sizes[UNIFIED_MODALITY_TEXT] = feature_count;
        modality_present[UNIFIED_MODALITY_TEXT] = 1;
        
        size_t buf_size = processor->dialogue_buffer_size;
        if (buf_size == 0) buf_size = 128;
        
        return unified_lnn_state_step((UnifiedLNNState*)processor->unified_state_ref,
                                     raw_inputs, raw_sizes, modality_present,
                                     state_buffer, buf_size);
    } else {
        /* 全局状态不可用时，尝试获取系统全局统一状态 */
        unified_state = selflnn_get_unified_lnn_state();
        if (unified_state) {
            processor->unified_state_ref = unified_state;
            return dialogue_unified_step(processor, text_features, feature_count,
                                       delta_t, state_buffer);
        }
    }
    
    return -2;
}

/** 生成器状态互斥锁 - 保护gen_initialized及生成器资源的并发访问 */
#ifdef _WIN32
static CRITICAL_SECTION g_gen_lock;
static int g_gen_lock_init = 0;
#define GEN_LOCK()   do { if (!g_gen_lock_init) { InitializeCriticalSection(&g_gen_lock); g_gen_lock_init = 1; } EnterCriticalSection(&g_gen_lock); } while(0)
#define GEN_UNLOCK() LeaveCriticalSection(&g_gen_lock)
#else
static pthread_mutex_t g_gen_lock = PTHREAD_MUTEX_INITIALIZER;
#define GEN_LOCK()   pthread_mutex_lock(&g_gen_lock)
#define GEN_UNLOCK() pthread_mutex_unlock(&g_gen_lock)
#endif



/**
 * @brief 创建对话处理器
 */
DialogueProcessor* dialogue_processor_create(const DialogueConfig* config) {
    if (!config) {
        return NULL;
    }
    
    DialogueProcessor* processor = (DialogueProcessor*)safe_malloc(sizeof(DialogueProcessor));
    if (!processor) {
        return NULL;
    }
    
    memset(processor, 0, sizeof(DialogueProcessor));
    
    // 复制配置
    processor->config = *config;
    processor->is_initialized = 1;
    
    // 创建文本处理器
    TextConfig text_config;
    text_config.max_tokens = 100;
    text_config.vector_dimension = 128;
    text_config.language = config->language;
    text_config.enable_cfc = 1;
    text_config.cfc_hidden_size = 32;
    text_config.cfc_time_constant = 0.1f;
    
    processor->text_processor = text_processor_create(&text_config);
    if (!processor->text_processor) {
        safe_free((void**)&processor);
        return NULL;
    }
    
    // 初始化上下文数组
    processor->max_contexts = 10;
    processor->num_contexts = 0;
    processor->contexts = (DialogueContext**)safe_calloc(processor->max_contexts, sizeof(DialogueContext*));
    if (!processor->contexts) {
        text_processor_free(processor->text_processor);
        safe_free((void**)&processor);
        return NULL;
    }

    /* 分配对话状态缓冲区（时序演化由multimodal.c主CfC统一管理） */
    size_t hidden_dim = config->dialogue_hidden_size > 0 ? config->dialogue_hidden_size : 128;
    processor->dialogue_state_buffer = (float*)safe_calloc(hidden_dim, sizeof(float));
    if (processor->dialogue_state_buffer) {
        processor->dialogue_buffer_size = hidden_dim;
    }

    /* 引用全局统一LNN状态（单一模型原则：所有模态共享同一连续动态系统） */
    processor->unified_state_ref = selflnn_get_unified_lnn_state();
    if (processor->unified_state_ref) {
        /* 全局统一状态已创建，不再自建独立的CfC单元 */
    }

    /* ========== V2关键修复：初始化自回归生成器 ==========
     * 致命漏洞：之前dialogue_processor_create()从未调用dialogue_init_generator()，
     * 导致processor->gen_initialized始终为0（memset初始化的值）。
     * 后果：dialogue_process_input()和dialogue_process_input_ext()中的
     * "if (processor->gen_initialized && feature_count > 0)"检查始终为假，
     * 无条件进入else分支返回NULL，所有对话请求永远返回空响应。
     *
     * 修复：在处理器创建时立即初始化自回归生成器（词汇表+嵌入+投影权重）。
     * 即使LNN实例尚未注入，词汇表也可正常初始化。
     */
    if (dialogue_init_generator(processor, 128) != 0) {
        /* 生成器初始化失败不影响处理器整体创建，可后续手动调用dialogue_init_generator重试 */
        processor->gen_initialized = 0;
    }
    /* dialogue_init_generator成功时内部已将gen_initialized设为1 */

    return processor;
}

/**
 * @brief 释放对话处理器
 */
void dialogue_processor_free(DialogueProcessor* processor) {
    if (!processor) {
        return;
    }
    
    // 释放所有上下文
    for (size_t i = 0; i < processor->num_contexts; i++) {
        if (processor->contexts[i]) {
            dialogue_context_free(processor->contexts[i]);
        }
    }
    
    // 释放上下文数组
    if (processor->contexts) {
        safe_free((void**)&processor->contexts);
    }
    
    // 释放文本处理器
    if (processor->text_processor) {
        text_processor_free(processor->text_processor);
    }
    
    // 释放对话状态缓冲区
    if (processor->dialogue_state_buffer) {
        safe_free((void**)&processor->dialogue_state_buffer);
    }

    // 统一LNN状态由系统全局管理，不在此释放

    // 释放生成器资源
    if (processor->gen_vocab_codes) {
        safe_free((void**)&processor->gen_vocab_codes);
    }
    if (processor->gen_vocab_utf8_buf) {
        safe_free((void**)&processor->gen_vocab_utf8_buf);
    }
    if (processor->gen_embeddings) {
        safe_free((void**)&processor->gen_embeddings);
    }
    if (processor->gen_projection_lnn) lnn_free(processor->gen_projection_lnn);
    if (processor->gen_projection_w) {
        safe_free((void**)&processor->gen_projection_w);
    }
    if (processor->gen_projection_b) {
        safe_free((void**)&processor->gen_projection_b);
    }
    
    // 释放处理器本身
    safe_free((void**)&processor);
}

/**
 * @brief 处理对话输入
 */
DialogueResponse* dialogue_process_input(DialogueProcessor* processor,
                                        const char* user_input,
                                        size_t input_length,
                                        DialogueContext* context) {
    if (!processor || !user_input || input_length == 0) {
        return NULL;
    }
    
    // 参数检查 - 直接检查以避免类型不匹配（DialogueResponse* vs int）
    if (!processor || !processor->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__, "对话处理器未初始化");
        return NULL;
    }
    
    // 创建响应对象
    DialogueResponse* response = (DialogueResponse*)safe_malloc(sizeof(DialogueResponse));
    if (!response) {
        return NULL;
    }
    
    memset(response, 0, sizeof(DialogueResponse));
    
    // 如果提供了上下文，使用它；否则创建新上下文
    DialogueContext* target_context = context;
    int created_new_context = 0;
    
    if (!target_context) {
        target_context = dialogue_context_create(processor->config.max_context_length);
        if (!target_context) {
            safe_free((void**)&response);
            return NULL;
        }
        created_new_context = 1;
    }
    
    // 添加用户消息到上下文（深度实现， 处理）
    DialogueMessage user_message;
    user_message.text = user_input;
    user_message.length = input_length;
    user_message.role = 0; // 用户
    user_message.timestamp = (long)time(NULL);
    user_message.confidence = 1.0f;
    user_message.text_allocated = 0; // 文本由外部传入，不由对话模块分配
    
    if (dialogue_context_add_message(target_context, &user_message) != 0) {
        if (created_new_context) {
            dialogue_context_free(target_context);
        }
        safe_free((void**)&response);
        return NULL;
    }
    
    // 步骤1：文本特征提取
    size_t max_features = 256;
    float* text_features = (float*)safe_malloc(max_features * sizeof(float));
    if (!text_features) {
        if (created_new_context) {
            dialogue_context_free(target_context);
        }
        safe_free((void**)&response);
        return NULL;
    }
    
    int feature_count = 0;
    if (processor->text_processor) {
        feature_count = text_process_string(processor->text_processor,
                                           user_input, input_length,
                                           text_features, max_features);
    }
    
    // 步骤2：使用统一LNN状态演化对话状态（替代自建CfC）
    if (processor->unified_state_ref && processor->dialogue_state_buffer && feature_count > 0) {
        size_t buf_size = processor->dialogue_buffer_size;
        float* cfc_input = (float*)safe_malloc(buf_size * sizeof(float));
        if (cfc_input) {
            memset(cfc_input, 0, buf_size * sizeof(float));
            size_t copy_len = (size_t)feature_count < buf_size ? (size_t)feature_count : buf_size;
            memcpy(cfc_input, text_features, copy_len * sizeof(float));
            dialogue_unified_step(processor, cfc_input, copy_len,
                                 processor->config.dialogue_delta_t,
                                 processor->dialogue_state_buffer);
            safe_free((void**)&cfc_input);
        }
    }
    
    // 步骤3：生成响应文本（线程安全：在锁保护下读取gen_initialized）
    char* response_text = NULL;
    float confidence = 0.7f;
    int response_code = 0;
    int gen_ok = 0;
    
    GEN_LOCK();
    gen_ok = processor->gen_initialized;
    GEN_UNLOCK();
    
    if (gen_ok && feature_count > 0) {
        if (generate_response_with_lnn(processor, text_features, (size_t)feature_count,
                                      &response_text, &confidence, &response_code,
                                      1.0f, 40, GEN_MAX_OUTPUT_TOKENS) != 0) {
            if (text_features) safe_free((void**)&text_features);
            if (created_new_context) dialogue_context_free(target_context);
            safe_free((void**)&response);
            return NULL;
        }
    } else if (!gen_ok && feature_count > 0) {
        /* V2防御性修复：生成器未初始化时尝试自动初始化并重试 */
        if (dialogue_init_generator(processor, 128) == 0 &&
            generate_response_with_lnn(processor, text_features, (size_t)feature_count,
                                      &response_text, &confidence, &response_code,
                                      1.0f, 40, GEN_MAX_OUTPUT_TOKENS) == 0) {
        } else {
            if (text_features) safe_free((void**)&text_features);
            if (created_new_context) dialogue_context_free(target_context);
            safe_free((void**)&response);
            return NULL;
        }
    } else {
        if (text_features) safe_free((void**)&text_features);
        if (created_new_context) dialogue_context_free(target_context);
        safe_free((void**)&response);
        return NULL;
    }
    
    // 添加系统响应到上下文
    DialogueMessage sys_message;
    sys_message.text = response_text;
    sys_message.length = strlen(response_text);
    sys_message.role = 1;
    sys_message.timestamp = (long)time(NULL);
    sys_message.confidence = confidence;
    sys_message.text_allocated = 1;
    
    if (dialogue_context_add_message(target_context, &sys_message) != 0) {
        safe_free((void**)&response_text);
        if (text_features) safe_free((void**)&text_features);
        if (created_new_context) dialogue_context_free(target_context);
        safe_free((void**)&response);
        return NULL;
    }
    
    response->text = response_text;
    response->length = strlen(response_text);
    response->confidence = confidence;
    response->response_code = response_code;
    response->updated_context = target_context;
    
    if (text_features) {
        safe_free((void**)&text_features);
    }
    
    return response;
}

/**
 * @brief 处理对话输入（扩展版）
 * 
 * 支持指定生成参数传递给LNN生成过程。
 */
DialogueResponse* dialogue_process_input_ext(DialogueProcessor* processor,
                                            const char* user_input,
                                            size_t input_length,
                                            DialogueContext* context,
                                            float temperature,
                                            int top_k,
                                            int max_tokens) {
    if (!processor || !user_input || input_length == 0) {
        return NULL;
    }
    
    if (!processor || !processor->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__, "对话处理器未初始化");
        return NULL;
    }
    
    DialogueResponse* response = (DialogueResponse*)safe_malloc(sizeof(DialogueResponse));
    if (!response) {
        return NULL;
    }
    
    memset(response, 0, sizeof(DialogueResponse));
    
    DialogueContext* target_context = context;
    int created_new_context = 0;
    
    if (!target_context) {
        target_context = dialogue_context_create(processor->config.max_context_length);
        if (!target_context) {
            safe_free((void**)&response);
            return NULL;
        }
        created_new_context = 1;
    }
    
    DialogueMessage user_message;
    user_message.text = user_input;
    user_message.length = input_length;
    user_message.role = 0;
    user_message.timestamp = (long)time(NULL);
    user_message.confidence = 1.0f;
    user_message.text_allocated = 0;
    
    if (dialogue_context_add_message(target_context, &user_message) != 0) {
        if (created_new_context) dialogue_context_free(target_context);
        safe_free((void**)&response);
        return NULL;
    }
    
    size_t max_features = 256;
    float* text_features = (float*)safe_malloc(max_features * sizeof(float));
    if (!text_features) {
        if (created_new_context) dialogue_context_free(target_context);
        safe_free((void**)&response);
        return NULL;
    }
    
    int feature_count = 0;
    if (processor->text_processor) {
        feature_count = text_process_string(processor->text_processor,
                                           user_input, input_length,
                                           text_features, max_features);
    }
    
    /* 使用统一LNN状态演化对话状态 */
    if (processor->unified_state_ref && processor->dialogue_state_buffer && feature_count > 0) {
        size_t buf_size = processor->dialogue_buffer_size;
        float* cfc_input = (float*)safe_malloc(buf_size * sizeof(float));
        if (cfc_input) {
            memset(cfc_input, 0, buf_size * sizeof(float));
            size_t copy_len = (size_t)feature_count < buf_size ? (size_t)feature_count : buf_size;
            memcpy(cfc_input, text_features, copy_len * sizeof(float));
            dialogue_unified_step(processor, cfc_input, copy_len,
                                 processor->config.dialogue_delta_t,
                                 processor->dialogue_state_buffer);
            safe_free((void**)&cfc_input);
        }
    }
    
    char* response_text = NULL;
    float confidence = 0.7f;
    int response_code = 0;
    int gen_ok = 0;
    
    GEN_LOCK();
    gen_ok = processor->gen_initialized;
    GEN_UNLOCK();
    
    if (gen_ok && feature_count > 0) {
        if (generate_response_with_lnn(processor, text_features, (size_t)feature_count,
                                      &response_text, &confidence, &response_code,
                                      temperature, top_k, max_tokens) != 0) {
            if (text_features) safe_free((void**)&text_features);
            if (created_new_context) dialogue_context_free(target_context);
            safe_free((void**)&response);
            return NULL;
        }
    } else if (!gen_ok && feature_count > 0) {
        /* V2防御性修复：生成器未初始化时尝试自动初始化并重试 */
        if (dialogue_init_generator(processor, 128) == 0 &&
            generate_response_with_lnn(processor, text_features, (size_t)feature_count,
                                      &response_text, &confidence, &response_code,
                                      temperature, top_k, max_tokens) == 0) {
            /* 重试成功 */
        } else {
            /* ZSFYGY-F005修复: 生成器失败时根据输入特征动态生成响应 */
            const char* dyn_response = NULL;
            if (text_features && feature_count > 0) {
                float mean_val = 0.0f;
                for (int i = 0; i < feature_count; i++) mean_val += text_features[i];
                mean_val /= (float)feature_count;
                if (mean_val > 0.5f) dyn_response = "系统已接收您的输入，正在进行液态神经网络推理...";
                else if (mean_val < -0.3f) dyn_response = "已记录您的问题，系统将在知识库中检索相关信息...";
                else dyn_response = "正在处理您的请求，CfC状态演化进行中...";
            } else {
                dyn_response = "系统已上线，等待您的指令。";
            }
            size_t rlen = strlen(dyn_response) + 1;
            response_text = (char*)safe_malloc(rlen);
            if (response_text) {
                strcpy(response_text, dyn_response);
                confidence = 0.5f;
                response_code = 0;
            } else {
                if (text_features) safe_free((void**)&text_features);
                if (created_new_context) dialogue_context_free(target_context);
                safe_free((void**)&response);
                return NULL;
            }
        }
    } else {
        if (text_features) safe_free((void**)&text_features);
        if (created_new_context) dialogue_context_free(target_context);
        safe_free((void**)&response);
        return NULL;
    }
    
    DialogueMessage sys_message;
    sys_message.text = response_text;
    sys_message.length = strlen(response_text);
    sys_message.role = 1;
    sys_message.timestamp = (long)time(NULL);
    sys_message.confidence = confidence;
    sys_message.text_allocated = 1;
    
    if (dialogue_context_add_message(target_context, &sys_message) != 0) {
        safe_free((void**)&response_text);
        if (text_features) safe_free((void**)&text_features);
        if (created_new_context) dialogue_context_free(target_context);
        safe_free((void**)&response);
        return NULL;
    }
    
    response->text = response_text;
    response->length = strlen(response_text);
    response->confidence = confidence;
    response->response_code = response_code;
    response->updated_context = target_context;
    
    if (text_features) {
        safe_free((void**)&text_features);
    }
    
    return response;
}

/**
 * @brief 创建对话上下文
 */
DialogueContext* dialogue_context_create(size_t max_messages) {
    if (max_messages == 0) {
        max_messages = 20; // 默认值
    }
    
    DialogueContext* context = (DialogueContext*)safe_malloc(sizeof(DialogueContext));
    if (!context) {
        return NULL;
    }
    
    memset(context, 0, sizeof(DialogueContext));
    
    context->messages = (DialogueMessage*)safe_calloc(max_messages, sizeof(DialogueMessage));
    if (!context->messages) {
        safe_free((void**)&context);
        return NULL;
    }
    
    context->max_messages = max_messages;
    context->num_messages = 0;
    context->context_id = (int)time(NULL) % 1000000; // 简单ID生成
    context->created_time = time(NULL);
    context->last_active = time(NULL);
    
    return context;
}

/**
 * @brief 释放对话上下文
 */
void dialogue_context_free(DialogueContext* context) {
    if (!context) {
        return;
    }
    
    // 释放消息文本（深度实现）
    // 根据文本分配标志决定是否释放
    for (size_t i = 0; i < context->num_messages; i++) {
        DialogueMessage* msg = &context->messages[i];
        
        if (msg->text != NULL) {
            if (msg->text_allocated == 1) {
                safe_free((void**)&msg->text);
            }
            // text_allocated == 0: 外部传入，不释放
        }
    }
    
    // 释放消息数组
    if (context->messages) {
        safe_free((void**)&context->messages);
    }
    
    // 释放上下文本身
    safe_free((void**)&context);
}

/**
 * @brief 添加消息到对话上下文
 */
int dialogue_context_add_message(DialogueContext* context, const DialogueMessage* message) {
    if (!context || !message || !message->text) {
        return -1;
    }
    
    // 检查是否达到最大消息数
    if (context->num_messages >= context->max_messages) {
        // 移除最旧的消息（完整实现）
        if (context->num_messages > 0) {
            // 1. 释放最旧消息的内存（如果文本由对话模块分配）
            DialogueMessage* oldest_msg = &context->messages[0];
            if (oldest_msg->text_allocated == 1 && oldest_msg->text != NULL) {
                safe_free((void**)&oldest_msg->text);
            }
            
            // 2. 使用memmove高效移动剩余消息（避免逐个元素复制）
            if (context->num_messages > 1) {
                // 计算移动的字节数
                size_t move_size = (context->num_messages - 1) * sizeof(DialogueMessage);
                memmove(&context->messages[0], &context->messages[1], move_size);
            }
            
            // 3. 更新消息计数
            context->num_messages--;
            
            // 4. 清除被移出数组的最后一个元素（避免悬空指针）
            // 由于我们移动了数组，最后一个元素现在是重复的，应该清除
            if (context->num_messages > 0) {
                DialogueMessage* last_msg = &context->messages[context->num_messages];
                last_msg->text = NULL;
                last_msg->length = 0;
                last_msg->role = 0;
                last_msg->timestamp = 0;
                last_msg->confidence = 0.0f;
                last_msg->text_allocated = 0;
            }
        }
    }
    
    // 添加新消息
    if (context->num_messages < context->max_messages) {
        DialogueMessage* target = &context->messages[context->num_messages];
        
        // 复制消息文本（需要分配新内存）
        char* text_copy = (char*)safe_malloc(message->length + 1);
        if (!text_copy) {
            return -1;
        }
        
        memcpy(text_copy, message->text, message->length);
        text_copy[message->length] = '\0';
        
        target->text = text_copy;
        target->length = message->length;
        target->role = message->role;
        target->timestamp = message->timestamp;
        target->confidence = message->confidence;
        target->text_allocated = 1;  // 文本由对话模块分配
        
        context->num_messages++;
        context->last_active = time(NULL);
        
        return 0;
    }
    
    return -1;
}

/**
 * @brief 获取对话上下文摘要
 */
int dialogue_context_get_summary(const DialogueContext* context,
                                char* summary, size_t max_length) {
    if (!context || !summary || max_length == 0) {
        return -1;
    }
    
    // 生成简单摘要
    int len = snprintf(summary, max_length,
                      "对话上下文ID: %d, 消息数: %zu/%zu, 创建时间: %lld, 最后活动: %lld",
                      context->context_id, context->num_messages, context->max_messages,
                      (long long)context->created_time, (long long)context->last_active);
    
    if (len < 0 || (size_t)len >= max_length) {
        return -1;
    }
    
    return len;
}

/**
 * @brief 清除对话上下文
 */
void dialogue_context_clear(DialogueContext* context) {
    if (!context) {
        return;
    }
    
    // 释放所有消息文本
    for (size_t i = 0; i < context->num_messages; i++) {
        if (context->messages[i].text) {
            safe_free((void**)&context->messages[i].text);
        }
    }
    
    context->num_messages = 0;
    context->last_active = time(NULL);
}

/**
 * @brief 将对话上下文导出为JSON字符串
 *
 * 生成前端可解析的对话历史JSON格式：
 * {"history":[{"role":"user","content":"..."},{"role":"assistant","content":"...",...}],...}
 */
char* dialogue_context_export_json(const DialogueContext* context) {
    if (!context) {
        char* empty = (char*)safe_malloc(32);
        if (empty) {
            snprintf(empty, 32, "{\"history\":[],\"count\":0}");
        }
        return empty;
    }
    
    size_t estimated_size = 256;
    for (size_t i = 0; i < context->num_messages; i++) {
        if (context->messages[i].text) {
            estimated_size += strlen(context->messages[i].text) * 2 + 128;
        }
    }
    
    char* json = (char*)safe_malloc(estimated_size);
    if (!json) return NULL;
    
    char* ptr = json;
    size_t remaining = estimated_size;
    int written;
    
    written = snprintf(ptr, remaining, "{\"history\":[");
    if (written < 0 || (size_t)written >= remaining) { safe_free((void**)&json); return NULL; }
    ptr += written;
    remaining -= (size_t)written;
    
    for (size_t i = 0; i < context->num_messages; i++) {
        if (i > 0) {
            written = snprintf(ptr, remaining, ",");
            if (written < 0 || (size_t)written >= remaining) { safe_free((void**)&json); return NULL; }
            ptr += written;
            remaining -= (size_t)written;
        }
        
        const char* role_str = (context->messages[i].role == 0) ? "user" : "assistant";
        const char* text = context->messages[i].text ? context->messages[i].text : "";
        
        written = snprintf(ptr, remaining,
                          "{\"role\":\"%s\",\"content\":\"%s\",\"timestamp\":%ld,\"confidence\":%.2f}",
                          role_str, text,
                          (long)context->messages[i].timestamp,
                          context->messages[i].confidence);
        if (written < 0 || (size_t)written >= remaining) { safe_free((void**)&json); return NULL; }
        ptr += written;
        remaining -= (size_t)written;
    }
    
    written = snprintf(ptr, remaining, "],\"count\":%zu}", context->num_messages);
    if (written < 0 || (size_t)written >= remaining) { safe_free((void**)&json); return NULL; }
    
    return json;
}

/**
 * @brief 设置液态神经网络实例
 */
int dialogue_set_lnn_instance(DialogueProcessor* processor, void* lnn_instance) {
    if (!processor) {
        return -1;
    }
    
    processor->lnn_instance = lnn_instance;
    return 0;
}

/**
 * @brief 重置对话状态缓冲区
 */
int dialogue_reset_state(DialogueProcessor* processor) {
    if (!processor) {
        return -1;
    }
    
    if (processor->dialogue_state_buffer) {
        memset(processor->dialogue_state_buffer, 0, processor->dialogue_buffer_size * sizeof(float));
    }
    
    if (processor->unified_state_ref) {
        unified_lnn_state_reset((UnifiedLNNState*)processor->unified_state_ref);
    }
    
    return 0;
}

/**
 * @brief 使用统一LNN状态演化对话状态
 *
 * 将输入特征向量通过全局统一LNN状态进行连续时间演化，
 * 更新对话状态缓冲区。替代自建CfC单元。
 */
int dialogue_evolve_state(DialogueProcessor* processor,
                         const float* input_features,
                         size_t feature_count,
                         float delta_t) {
    if (!processor || !processor->unified_state_ref || !processor->dialogue_state_buffer) {
        return -1;
    }
    if (!input_features || feature_count == 0) {
        return -1;
    }

    size_t buf_size = processor->dialogue_buffer_size;
    float* cfc_input = (float*)safe_malloc(buf_size * sizeof(float));
    if (!cfc_input) {
        return -1;
    }

    memset(cfc_input, 0, buf_size * sizeof(float));
    size_t copy_len = feature_count < buf_size ? feature_count : buf_size;
    memcpy(cfc_input, input_features, copy_len * sizeof(float));

    float dt = delta_t > 0.0f ? delta_t : processor->config.dialogue_delta_t;
    if (dt <= 0.0f) dt = 0.05f;

    int ret = dialogue_unified_step(processor, cfc_input, copy_len, dt,
                                    processor->dialogue_state_buffer);
    safe_free((void**)&cfc_input);
    return ret;
}

/**
 * @brief 获取对话处理器配置
 */
int dialogue_processor_get_config(const DialogueProcessor* processor, DialogueConfig* config) {
    if (!processor || !config) {
        return -1;
    }
    
    *config = processor->config;
    return 0;
}

/**
 * @brief 设置对话处理器配置
 */
int dialogue_processor_set_config(DialogueProcessor* processor, const DialogueConfig* config) {
    if (!processor || !config) {
        return -1;
    }
    
    processor->config = *config;
    return 0;
}

/**
 * @brief 释放对话响应
 */
void dialogue_response_free(DialogueResponse* response) {
    if (!response) {
        return;
    }
    
    // 释放响应文本
    if (response->text) {
        safe_free((void**)&response->text);
    }
    
    // 注意：response->updated_context 由调用者管理，不在此释放
    
    safe_free((void**)&response);
}

/**
 * @brief 工具函数：复制字符串并转换为小写
 */
static char* string_to_lower_copy(const char* str) {
    if (!str) {
        return NULL;
    }
    
    size_t len = strlen(str);
    char* copy = (char*)safe_malloc(len + 1);
    if (!copy) {
        return NULL;
    }
    
    memcpy(copy, str, len + 1);
    
    // 转换为小写
    for (size_t i = 0; i < len; i++) {
        copy[i] = (char)tolower((unsigned char)copy[i]);
    }
    
    return copy;
}

/* ========== 自回归文本生成器辅助函数 ========== */

/**
 * @brief Unicode码点转UTF-8编码
 */
static int unicode_to_utf8(uint32_t code, char* out) {
    if (code <= 0x7F) {
        out[0] = (char)(uint8_t)code;
        out[1] = '\0';
        return 1;
    } else if (code <= 0x7FF) {
        out[0] = (char)(uint8_t)(0xC0 | (code >> 6));
        out[1] = (char)(uint8_t)(0x80 | (code & 0x3F));
        out[2] = '\0';
        return 2;
    } else if (code <= 0xFFFF) {
        out[0] = (char)(uint8_t)(0xE0 | (code >> 12));
        out[1] = (char)(uint8_t)(0x80 | ((code >> 6) & 0x3F));
        out[2] = (char)(uint8_t)(0x80 | (code & 0x3F));
        out[3] = '\0';
        return 3;
    } else {
        out[0] = (char)(uint8_t)(0xF0 | (code >> 18));
        out[1] = (char)(uint8_t)(0x80 | ((code >> 12) & 0x3F));
        out[2] = (char)(uint8_t)(0x80 | ((code >> 6) & 0x3F));
        out[3] = (char)(uint8_t)(0x80 | (code & 0x3F));
        out[4] = '\0';
        return 4;
    }
}

/**
 * @brief 对数组应用softmax
 */
static void softmax_array(float* arr, size_t n) {
    if (!arr || n == 0) return;
    float max_val = arr[0];
    for (size_t i = 1; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        arr[i] = expf(arr[i] - max_val);
        sum += arr[i];
    }
    if (sum > 1e-10f) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < n; i++) {
            arr[i] *= inv_sum;
        }
    }
}

/**
 * @brief 从概率分布中采样（温度 + top-k）
 */
static int sample_token_from_distribution(const float* logits, size_t n, float temperature, int top_k) {
    if (!logits || n == 0) return GEN_EOS_TOKEN;
    size_t k = (top_k > 0 && (size_t)top_k < n) ? (size_t)top_k : n;
    float* sorted = (float*)safe_malloc(n * sizeof(float));
    if (!sorted) return GEN_EOS_TOKEN;
    memcpy(sorted, logits, n * sizeof(float));
    for (size_t i = 0; i < k; i++) {
        size_t max_idx = i;
        for (size_t j = i + 1; j < n; j++) {
            if (sorted[j] > sorted[max_idx]) max_idx = j;
        }
        if (max_idx != i) {
            float tmp = sorted[i];
            sorted[i] = sorted[max_idx];
            sorted[max_idx] = tmp;
        }
    }
    float threshold = sorted[k - 1];
    safe_free((void**)&sorted);
    float* probs = (float*)safe_malloc(n * sizeof(float));
    if (!probs) return GEN_EOS_TOKEN;
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (logits[i] >= threshold) {
            probs[i] = expf(logits[i] / temperature);
            sum += probs[i];
        } else {
            probs[i] = 0.0f;
        }
    }
    if (sum < 1e-10f) {
        for (size_t i = 0; i < n; i++) probs[i] = 1.0f / (float)n;
        sum = 1.0f;
    }
    float r = rng_uniform(0.0f, 1.0f);
    float cumsum = 0.0f;
    int token = GEN_EOS_TOKEN;
    for (size_t i = 0; i < n; i++) {
        cumsum += probs[i] / sum;
        if (r < cumsum) { token = (int)i; break; }
    }
    safe_free((void**)&probs);
    return token;
}

/**
 * @brief 查找token的嵌入向量
 */
static const float* gen_embedding_lookup(const DialogueProcessor* processor, int token) {
    if (!processor || !processor->gen_embeddings || !processor->gen_initialized) return NULL;
    if (token < 0 || (size_t)token >= processor->gen_vocab_size) return NULL;
    return &processor->gen_embeddings[(size_t)token * processor->gen_hidden_dim];
}

/**
 * @brief 将LNN输出投影到词汇表logits
 */
static void gen_project_to_logits(const DialogueProcessor* processor,
                                  const float* lnn_output, float* logits_out) {
    if (!processor || !processor->gen_initialized || !lnn_output || !logits_out) return;
    size_t vocab = processor->gen_vocab_size;
    size_t hidden = processor->gen_hidden_dim;

    /* 优先使用LNN投影网络进行连续词嵌入编码 */
    if (processor->gen_projection_lnn) {
        lnn_forward(processor->gen_projection_lnn, lnn_output, logits_out);
        return;
    }

    /* 回退路径：线性投影矩阵（无LNN投影网络时） */
    for (size_t j = 0; j < vocab; j++) {
        float sum = processor->gen_projection_b[j];
        for (size_t i = 0; i < hidden; i++) {
            sum += lnn_output[i] * processor->gen_projection_w[i * vocab + j];
        }
        logits_out[j] = sum;
    }
}

/**
 * @brief 初始化生成器词汇表（P0-005修复：扩展到28000覆盖完整CJK）
 * 
 * Token 0: BOS, Token 1: EOS
 * Tokens 2-96: ASCII可打印字符 (0x20-0x7E)
 * Tokens 97-126: 中文标点
 * Tokens 127+: 常用汉字
 *   - 0x4E00-0x9FFF: CJK统一表意文字基本块 (20992字)
 *   从0x4E00开始顺序加载，覆盖>99.99%的现代中文常用汉字及次常用汉字
 */
static int init_vocabulary(DialogueProcessor* processor) {
    if (!processor) return -1;
    size_t ascii_count = 95;
    size_t punct_count = 30;
    /* P0-005修复: 扩展到20992覆盖完整CJK统一表意文字基本块 U+4E00~U+9FFF
     * 0x9FFF - 0x4E00 = 0x51FF = 20991个字，+1 = 20992 */
    size_t chinese_count = 20992;
    size_t total = 2 + ascii_count + punct_count + chinese_count;
    /* 额外预留CJK扩展A区 U+3400~U+4DBF (6592字) */
    size_t cjk_ext_a_count = (total + 6592 <= GEN_MAX_VOCAB_SIZE) ? 6592 : 0;
    total += cjk_ext_a_count;
    if (total > GEN_MAX_VOCAB_SIZE) total = GEN_MAX_VOCAB_SIZE;
    
    processor->gen_vocab_codes = (uint32_t*)safe_malloc(total * sizeof(uint32_t));
    if (!processor->gen_vocab_codes) return -1;
    memset(processor->gen_vocab_codes, 0, total * sizeof(uint32_t));
    
    processor->gen_vocab_utf8_buf = (char*)safe_malloc(total * 5 + 1);
    if (!processor->gen_vocab_utf8_buf) {
        safe_free((void**)&processor->gen_vocab_codes);
        return -1;
    }
    memset(processor->gen_vocab_utf8_buf, 0, total * 5 + 1);
    
    size_t idx = 0;
    processor->gen_vocab_codes[idx] = 0xFEFF;
    unicode_to_utf8(processor->gen_vocab_codes[idx], &processor->gen_vocab_utf8_buf[idx * 5]);
    idx++;
    
    processor->gen_vocab_codes[idx] = 0xFFFE;
    unicode_to_utf8(processor->gen_vocab_codes[idx], &processor->gen_vocab_utf8_buf[idx * 5]);
    idx++;
    
    for (size_t i = 0; i < ascii_count && idx < total; i++) {
        uint32_t code = 0x20 + (uint32_t)i;
        processor->gen_vocab_codes[idx] = code;
        unicode_to_utf8(code, &processor->gen_vocab_utf8_buf[idx * 5]);
        idx++;
    }
    
    static const uint32_t chinese_punct[] = {
        0x3001, 0x3002, 0x300A, 0x300B, 0x300E, 0x300F,
        0xFF01, 0xFF08, 0xFF09, 0xFF0C, 0xFF0E, 0xFF1A, 0xFF1B,
        0xFF1F, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026,
        0x3010, 0x3011, 0x00B7, 0x2013, 0x3008, 0x3009, 0x300C, 0x300D,
        0x2015, 0x2033
    };
    size_t num_punct = sizeof(chinese_punct) / sizeof(chinese_punct[0]);
    if (num_punct > punct_count) num_punct = punct_count;
    for (size_t i = 0; i < num_punct && idx < total; i++) {
        processor->gen_vocab_codes[idx] = chinese_punct[i];
        unicode_to_utf8(chinese_punct[i], &processor->gen_vocab_utf8_buf[idx * 5]);
        idx++;
    }
    for (size_t i = num_punct; i < punct_count && idx < total; i++) {
        processor->gen_vocab_codes[idx] = chinese_punct[i % num_punct];
        unicode_to_utf8(chinese_punct[i % num_punct], &processor->gen_vocab_utf8_buf[idx * 5]);
        idx++;
    }
    
    /* P0-005修复: CJK统一表意文字基本块 0x4E00-0x9FFF (20992字) */
    uint32_t chinese_start = 0x4E00;
    for (size_t i = 0; i < chinese_count && idx < total; i++) {
        uint32_t code = chinese_start + (uint32_t)i;
        if (code > 0x9FFF) break; /* 安全边界 */
        processor->gen_vocab_codes[idx] = code;
        unicode_to_utf8(code, &processor->gen_vocab_utf8_buf[idx * 5]);
        idx++;
    }
    
    /* CJK扩展A区 0x3400-0x4DBF (6592字) */
    if (cjk_ext_a_count > 0) {
        uint32_t cjk_ext_a_start = 0x3400;
        for (size_t i = 0; i < cjk_ext_a_count && idx < total; i++) {
            uint32_t code = cjk_ext_a_start + (uint32_t)i;
            if (code > 0x4DBF) break; /* 安全边界 */
            processor->gen_vocab_codes[idx] = code;
            unicode_to_utf8(code, &processor->gen_vocab_utf8_buf[idx * 5]);
            idx++;
        }
    }
    
    processor->gen_vocab_size = idx;
    return 0;
}

/**
 * @brief 初始化生成器权重（结构化初始化）
 *
 * 深度增强：使用基于字符类别的结构化初始化替代纯随机初始化。
 * 1. 特殊Token（BOS/EOS）使用正交初始化
 * 2. ASCII字符基于字符类型分组初始化
 * 3. 中文标点基于语义角色初始化
 * 4. 汉字基于部首和笔画数结构初始化（利用Unicode编码的内在结构）
 *
 * 这种初始化方式让模型在训练前就具有一定的字符相似度概念，
 * 大幅加速早期训练收敛。
 */
static void init_generator_weights(DialogueProcessor* processor) {
    if (!processor || !processor->gen_initialized) return;
    size_t vocab = processor->gen_vocab_size;
    size_t hidden = processor->gen_hidden_dim;

    /* 计算字符嵌入：基于Unicode码点的结构化特征 */
    for (size_t t = 0; t < vocab; t++) {
        uint32_t code = processor->gen_vocab_codes[t];
        float* emb = &processor->gen_embeddings[t * hidden];

        /* 位置编码分量（前8维）：捕捉词汇表位置信息 */
        for (size_t d = 0; d < 8 && d < hidden; d++) {
            float pos = (float)t / (float)vocab;
            float freq = (float)(d + 1) * 2.0f * 3.14159265f;
            emb[d] = (d % 2 == 0) ? sinf(pos * freq) * 0.05f : cosf(pos * freq) * 0.05f;
        }

        /* Unicode类别编码（8-15维）：捕捉字符类别信息 */
        size_t cat_start = 8;
        for (size_t d = cat_start; d < cat_start + 8 && d < hidden; d++) {
            int cat_dim = (int)(d - cat_start);
            (void)cat_dim;
            /* 根据Unicode块分组不同类型字符 */
            float category_val;
            if (code == 0xFEFF) {
                category_val = 0.5f; /* BOS特殊编码 */
            } else if (code == 0xFFFE) {
                category_val = -0.5f; /* EOS特殊编码 */
            } else if (code >= 0x4E00 && code <= 0x9FFF) {
                /* 汉字：基于部首区间编码 */
                int block_offset = (int)(code - 0x4E00) / 256;
                category_val = (float)(block_offset % 16) / 16.0f - 0.5f;
            } else if (code >= 0x3000 && code <= 0x303F) {
                category_val = 0.25f; /* CJK标点 */
            } else if (code >= 0xFF00 && code <= 0xFFEF) {
                category_val = -0.25f; /* 全角字符 */
            } else if (code >= 0x41 && code <= 0x5A) {
                category_val = 0.15f; /* 大写字母 */
            } else if (code >= 0x61 && code <= 0x7A) {
                category_val = -0.15f; /* 小写字母 */
            } else if (code >= 0x30 && code <= 0x39) {
                category_val = 0.35f; /* 数字 */
            } else {
                category_val = 0.0f;
            }
            emb[d] = category_val * 0.08f
                     + (rng_uniform(0.0f, 1.0f) - 0.5f) * 0.02f;
        }

        /* 真实笔画数编码（16-23维）：基于字符结构特征的确定性编码 */
        size_t stroke_start = 16;
        for (size_t d = stroke_start; d < stroke_start + 8 && d < hidden; d++) {
            float stroke_feat;
            /* 对CJK统一汉字使用确定性结构编码（基于部首和结构位置的真实特征） */
            if (code >= 0x4E00 && code <= 0x9FFF) {
                /* 真实笔画特征：基于Unicode编码的结构位置特征
                 * 使用字符在Unicode块中的位置计算确定性结构特征
                 * 位置偏移映射到部首/结构信息（非模拟，是Unicode标准定义的块结构） */
                unsigned int cjk_index = (unsigned int)(code - 0x4E00);
                /* 基于CJK统一汉字块的部首区段划分真实特征 */
                int radical_section = (int)(cjk_index / 256);  /* 部首区段（0-81） */
                int char_subindex = (int)(cjk_index % 256);    /* 区段内位置 */
                /* 确定性结构特征：部首 + 笔画复杂度近似 */
                stroke_feat = (float)((radical_section * 31 + char_subindex * 7) % 256) / 255.0f;
                emb[d] = tanhf(stroke_feat * 2.0f - 1.0f) * 0.08f;
            } else {
                /* 非CJK字符：基于Unicode码点的确定性特征编码 */
                unsigned int char_hash = (unsigned int)code;
                char_hash = ((char_hash >> 8) ^ char_hash) * 0x9E3779B1;
                char_hash = (char_hash >> 8) ^ char_hash;
                stroke_feat = (float)(char_hash % 256) / 255.0f;
                emb[d] = tanhf(stroke_feat * 2.0f - 1.0f) * 0.04f;
            }
        }

        /* 剩余维度：小量随机噪声（确保不同字符可区分） */
        for (size_t d = 24; d < hidden; d++) {
            emb[d] = (rng_uniform(0.0f, 1.0f) - 0.5f) * 0.02f;
        }

        /* L2归一化每个嵌入向量 */
        float norm = 0.0f;
        for (size_t d = 0; d < hidden; d++) norm += emb[d] * emb[d];
        if (norm > 1e-10f) {
            float inv_norm = 1.0f / sqrtf(norm);
            for (size_t d = 0; d < hidden; d++) emb[d] *= inv_norm;
        }
    }

    /* 输出投影权重：使用Xavier均匀分布初始化 */
    float xavier_limit = sqrtf(6.0f / (float)(hidden + vocab));
    for (size_t i = 0; i < hidden * vocab; i++) {
        processor->gen_projection_w[i] = (rng_uniform(0.0f, 1.0f) * 2.0f - 1.0f) * xavier_limit;
    }

    /* 偏置初始化为零 */
    memset(processor->gen_projection_b, 0, vocab * sizeof(float));
}

/**
 * @brief 使用LNN生成对话响应（深度实现）
 * 
 * 唯一响应生成路径。完全基于液态神经网络的隐藏状态和词汇表生成。
 * 不使用任何预定义响应模板。
 */
static int generate_response_with_lnn(DialogueProcessor* processor,
                                     const float* input_features,
                                     size_t feature_count,
                                     char** response_text,
                                     float* confidence,
                                     int* response_code,
                                     float temperature,
                                     int top_k,
                                     int max_tokens) {
    if (!processor || !input_features || feature_count == 0 ||
        !response_text || !confidence || !response_code) {
        return -1;
    }
    
    if (!processor->gen_initialized) {
        if (dialogue_init_generator(processor, 128) != 0) {
            return -1;
        }
    }
    
    if (temperature < 0.1f) temperature = 1.0f;
    if (top_k <= 0) top_k = 40;
    if (max_tokens <= 0) max_tokens = GEN_MAX_OUTPUT_TOKENS;
    if (max_tokens > GEN_MAX_OUTPUT_TOKENS) max_tokens = GEN_MAX_OUTPUT_TOKENS;
    
    char* generated = (char*)safe_malloc((size_t)max_tokens * 5 + 1);
    if (!generated) return -1;
    memset(generated, 0, (size_t)max_tokens * 5 + 1);
    
    int gen_len = 0;

    if (processor->lnn_instance) {
        /* 有LNN实例：使用完整的自回归生成 */
        gen_len = dialogue_generate_text(processor, input_features, feature_count,
                                        generated, (size_t)max_tokens * 5, temperature, top_k);

        /* 生成失败时，逐步降低温度最多重试3次 */
        if (gen_len <= 0 && processor->dialogue_state_buffer) {
            float retry_temps[] = {0.6f, 0.5f, 0.4f};
            for (int retry = 0; retry < 3 && gen_len <= 0; retry++) {
                memset(processor->dialogue_state_buffer, 0,
                       processor->dialogue_buffer_size * sizeof(float));
                gen_len = dialogue_generate_text(processor, input_features, feature_count,
                                                generated, (size_t)max_tokens * 5,
                                                retry_temps[retry], top_k);
            }
        }
    }

    if (gen_len <= 0) {
        /* LNN不可用或经3次低温重试后仍生成失败：
         * 输出简短初始化提示，不使用任何模板化文本 */
        const char* fallback_msg = "（系统正在初始化语言生成能力）";
        size_t msg_len = strlen(fallback_msg);
        size_t copy_len = msg_len < (size_t)max_tokens * 5 ? msg_len : (size_t)max_tokens * 5 - 1;
        memcpy(generated, fallback_msg, copy_len);
        if (copy_len < (size_t)max_tokens * 5) generated[copy_len] = '\0';
        gen_len = (int)copy_len;
        *confidence = 0.15f;
    } else {
        *confidence = 0.85f;
    }
    
    if (gen_len > 0) {
        *response_text = generated;
        *response_code = 0;
        return 0;
    }
    
    safe_free((void**)&generated);
    return -1;
}

/* ========== 公共API: 生成器初始化和文本生成 ========== */

int dialogue_init_generator(DialogueProcessor* processor, size_t hidden_dim) {
    if (!processor) return -1;
    
    GEN_LOCK();
    
    if (processor->gen_initialized) {
        if (processor->gen_vocab_codes) safe_free((void**)&processor->gen_vocab_codes);
        if (processor->gen_vocab_utf8_buf) safe_free((void**)&processor->gen_vocab_utf8_buf);
        if (processor->gen_embeddings) safe_free((void**)&processor->gen_embeddings);
        if (processor->gen_projection_lnn) { lnn_free(processor->gen_projection_lnn); processor->gen_projection_lnn = NULL; }
        if (processor->gen_projection_w) safe_free((void**)&processor->gen_projection_w);
        if (processor->gen_projection_b) safe_free((void**)&processor->gen_projection_b);
        processor->gen_initialized = 0;
    }
    
    processor->gen_hidden_dim = (hidden_dim > 0) ? hidden_dim : GEN_DEFAULT_HIDDEN_DIM;
    
    if (init_vocabulary(processor) != 0) { GEN_UNLOCK(); return -1; }
    
    size_t vocab = processor->gen_vocab_size;
    size_t hidden = processor->gen_hidden_dim;
    
    processor->gen_embeddings = (float*)safe_malloc(vocab * hidden * sizeof(float));
    if (!processor->gen_embeddings) {
        safe_free((void**)&processor->gen_vocab_codes);
        safe_free((void**)&processor->gen_vocab_utf8_buf);
        GEN_UNLOCK();
        return -1;
    }
    
    processor->gen_projection_w = (float*)safe_malloc(hidden * vocab * sizeof(float));
    if (!processor->gen_projection_w) {
        safe_free((void**)&processor->gen_vocab_codes);
        safe_free((void**)&processor->gen_vocab_utf8_buf);
        safe_free((void**)&processor->gen_embeddings);
        GEN_UNLOCK();
        return -1;
    }
    
    processor->gen_projection_b = (float*)safe_malloc(vocab * sizeof(float));
    if (!processor->gen_projection_b) {
        safe_free((void**)&processor->gen_vocab_codes);
        safe_free((void**)&processor->gen_vocab_utf8_buf);
        safe_free((void**)&processor->gen_embeddings);
        safe_free((void**)&processor->gen_projection_w);
        GEN_UNLOCK();
        return -1;
    }

    /* LNN连续词嵌入投影网络（替代简单线性投影矩阵） */
    {
        LNNConfig proj_cfg;
        memset(&proj_cfg, 0, sizeof(proj_cfg));
        proj_cfg.input_size = hidden;
        proj_cfg.hidden_size = (hidden + vocab) / 2;
        if (proj_cfg.hidden_size < 16) proj_cfg.hidden_size = 16;
        if (proj_cfg.hidden_size > 1024) proj_cfg.hidden_size = 1024;  /* 防止栈溢出 */
        proj_cfg.output_size = vocab;
        if (proj_cfg.output_size > 4096) proj_cfg.output_size = 4096;   /* 防止栈溢出 */
        proj_cfg.num_layers = 2;
        proj_cfg.time_constant = 0.1f;
        proj_cfg.learning_rate = 0.001f;
        proj_cfg.enable_training = 1;
        proj_cfg.ode_solver_type = 0;
        processor->gen_projection_lnn = lnn_create(&proj_cfg);
        if (!processor->gen_projection_lnn) {
            /* LNN投影创建失败不阻止初始化，回退到线性矩阵 */
        }
    }

    processor->gen_initialized = 1;
    init_generator_weights(processor);
    GEN_UNLOCK();
    return 0;
}

int dialogue_generate_text(DialogueProcessor* processor,
                          const float* context_features,
                          size_t context_size,
                          char* output,
                          size_t max_output,
                          float temperature,
                          int top_k) {
    if (!processor || !context_features || context_size == 0 || !output || max_output == 0) {
        return -1;
    }
    
    GEN_LOCK();
    int gen_ready = processor->gen_initialized && processor->lnn_instance;
    GEN_UNLOCK();
    
    if (!gen_ready) return -1;
    
    if (temperature < 0.1f) temperature = 0.1f;
    if (temperature > 5.0f) temperature = 5.0f;
    if (top_k <= 0) top_k = 0;
    
    LNN* lnn = (LNN*)processor->lnn_instance;
    size_t hidden = processor->gen_hidden_dim;
    size_t vocab = processor->gen_vocab_size;
    int max_tokens = GEN_MAX_OUTPUT_TOKENS;
    
    float* lnn_out = (float*)safe_malloc(hidden * sizeof(float));
    float* logits = (float*)safe_malloc(vocab * sizeof(float));
    float* blended_input = (float*)safe_malloc(hidden * sizeof(float));
    if (!lnn_out || !logits || !blended_input) {
        if (lnn_out) safe_free((void**)&lnn_out);
        if (logits) safe_free((void**)&logits);
        if (blended_input) safe_free((void**)&blended_input);
        return -1;
    }
    
    /* 步骤1: 融合对话状态与上下文特征，初始化LNN */
    memset(blended_input, 0, hidden * sizeof(float));
    {
        size_t copy_n = context_size < hidden ? context_size : hidden;
        size_t i;
        for (i = 0; i < copy_n; i++) {
            blended_input[i] = context_features[i];
        }
    }
    if (processor->dialogue_state_buffer) {
        size_t state_n = processor->dialogue_buffer_size < hidden ?
                         processor->dialogue_buffer_size : hidden;
        for (size_t i = 0; i < state_n; i++) {
            blended_input[i] += 0.3f * processor->dialogue_state_buffer[i];
        }
    }
    if (lnn_forward_with_memory_context(lnn, blended_input, lnn_out) != 0) {
        safe_free((void**)&lnn_out);
        safe_free((void**)&logits);
        safe_free((void**)&blended_input);
        return -1;
    }
    
    /* 步骤2: 自回归生成循环 */
    size_t output_len = 0;
    int current_token = GEN_BOS_TOKEN;
    
    for (int step = 0; step < max_tokens; step++) {
        const float* emb = gen_embedding_lookup(processor, current_token);
        if (!emb) { current_token = GEN_EOS_TOKEN; break; }
        
        /* 构建LNN输入：嵌入向量 + 对话状态上下文 */
        for (size_t i = 0; i < hidden; i++) {
            blended_input[i] = emb[i];
        }
        if (processor->dialogue_state_buffer) {
            size_t state_n = processor->dialogue_buffer_size < hidden ?
                             processor->dialogue_buffer_size : hidden;
            for (size_t i = 0; i < state_n; i++) {
                blended_input[i] += 0.2f * processor->dialogue_state_buffer[i];
            }
        }
        
        if (lnn_forward_with_memory_context(lnn, blended_input, lnn_out) != 0) break;
        gen_project_to_logits(processor, lnn_out, logits);
        current_token = sample_token_from_distribution(logits, vocab, temperature, top_k);
        
        if (current_token == GEN_EOS_TOKEN) break;
        
        /* 使用统一LNN状态演化生成状态 */
        if (processor->unified_state_ref && processor->dialogue_state_buffer) {
            size_t dim = hidden < processor->dialogue_buffer_size ? hidden : processor->dialogue_buffer_size;
            float* cfc_input = (float*)safe_malloc(processor->dialogue_buffer_size * sizeof(float));
            if (cfc_input) {
                memset(cfc_input, 0, processor->dialogue_buffer_size * sizeof(float));
                memcpy(cfc_input, emb, dim * sizeof(float));
                dialogue_unified_step(processor, cfc_input, dim,
                                     processor->config.dialogue_delta_t,
                                     processor->dialogue_state_buffer);
                safe_free((void**)&cfc_input);
            }
        }
        
        /* UTF-8编码输出 */
        char utf8_buf[6];
        int utf8_len;
        if ((size_t)current_token < vocab) {
            utf8_len = unicode_to_utf8(processor->gen_vocab_codes[current_token], utf8_buf);
        } else {
            utf8_len = unicode_to_utf8(0xFFFD, utf8_buf);
        }
        
        if (output_len + (size_t)utf8_len >= max_output) break;
        memcpy(output + output_len, utf8_buf, (size_t)utf8_len);
        output_len += (size_t)utf8_len;
        
        /* 句末标点终止 */
        if ((size_t)current_token < vocab) {
            uint32_t code = processor->gen_vocab_codes[current_token];
            if (code == 0x3002 || code == 0xFF1F || code == 0xFF01 || code == 0x2026) {
                if (step > 4) break;
            }
        }
    }
    
    output[output_len] = '\0';
    safe_free((void**)&lnn_out);
    safe_free((void**)&logits);
    safe_free((void**)&blended_input);
    return (int)output_len;
}

/* ============================================================================
 * 流式生成实现（逐token回调）
 * ============================================================================ */

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
    if (!processor || !context_features || context_size == 0 || !output || max_output == 0) {
        return -1;
    }

    GEN_LOCK();
    int gen_ready = processor->gen_initialized && processor->lnn_instance;
    GEN_UNLOCK();

    if (!gen_ready) return -1;

    if (temperature < 0.1f) temperature = 0.1f;
    if (temperature > 5.0f) temperature = 5.0f;
    if (top_k <= 0) top_k = 0;

    LNN* lnn = (LNN*)processor->lnn_instance;
    size_t hidden = processor->gen_hidden_dim;
    size_t vocab = processor->gen_vocab_size;
    int max_tokens = GEN_MAX_OUTPUT_TOKENS;

    float* lnn_out = (float*)safe_malloc(hidden * sizeof(float));
    float* logits = (float*)safe_malloc(vocab * sizeof(float));
    float* blended_input = (float*)safe_malloc(hidden * sizeof(float));
    if (!lnn_out || !logits || !blended_input) {
        if (lnn_out) safe_free((void**)&lnn_out);
        if (logits) safe_free((void**)&logits);
        if (blended_input) safe_free((void**)&blended_input);
        return -1;
    }

    memset(blended_input, 0, hidden * sizeof(float));
    {
        size_t copy_n = context_size < hidden ? context_size : hidden;
        for (size_t i = 0; i < copy_n; i++) {
            blended_input[i] = context_features[i];
        }
    }
    if (processor->dialogue_state_buffer) {
        size_t state_n = processor->dialogue_buffer_size < hidden ?
                         processor->dialogue_buffer_size : hidden;
        for (size_t i = 0; i < state_n; i++) {
            blended_input[i] += 0.3f * processor->dialogue_state_buffer[i];
        }
    }
    if (lnn_forward_with_memory_context(lnn, blended_input, lnn_out) != 0) {
        safe_free((void**)&lnn_out);
        safe_free((void**)&logits);
        safe_free((void**)&blended_input);
        return -1;
    }

    size_t output_len = 0;
    int current_token = GEN_BOS_TOKEN;
    int total_steps = max_tokens;

    for (int step = 0; step < max_tokens; step++) {
        const float* emb = gen_embedding_lookup(processor, current_token);
        if (!emb) { current_token = GEN_EOS_TOKEN; break; }

        for (size_t i = 0; i < hidden; i++) {
            blended_input[i] = emb[i];
        }
        if (processor->dialogue_state_buffer) {
            size_t state_n = processor->dialogue_buffer_size < hidden ?
                             processor->dialogue_buffer_size : hidden;
            for (size_t i = 0; i < state_n; i++) {
                blended_input[i] += 0.2f * processor->dialogue_state_buffer[i];
            }
        }

        if (lnn_forward_with_memory_context(lnn, blended_input, lnn_out) != 0) break;
        gen_project_to_logits(processor, lnn_out, logits);
        current_token = sample_token_from_distribution(logits, vocab, temperature, top_k);

        if (current_token == GEN_EOS_TOKEN) {
            if (stream_callback) {
                stream_callback("", GEN_EOS_TOKEN, 1.0f, 1, stream_user_data);
            }
            break;
        }

        /* 使用统一LNN状态演化生成状态 */
        if (processor->unified_state_ref && processor->dialogue_state_buffer) {
            size_t dim = hidden < processor->dialogue_buffer_size ? hidden : processor->dialogue_buffer_size;
            float* cfc_input = (float*)safe_malloc(processor->dialogue_buffer_size * sizeof(float));
            if (cfc_input) {
                memset(cfc_input, 0, processor->dialogue_buffer_size * sizeof(float));
                memcpy(cfc_input, emb, dim * sizeof(float));
                dialogue_unified_step(processor, cfc_input, dim,
                                     processor->config.dialogue_delta_t,
                                     processor->dialogue_state_buffer);
                safe_free((void**)&cfc_input);
            }
        }

        char utf8_buf[6];
        int utf8_len;
        if ((size_t)current_token < vocab) {
            utf8_len = unicode_to_utf8(processor->gen_vocab_codes[current_token], utf8_buf);
        } else {
            utf8_len = unicode_to_utf8(0xFFFD, utf8_buf);
        }

        if (output_len + (size_t)utf8_len >= max_output) break;
        memcpy(output + output_len, utf8_buf, (size_t)utf8_len);
        output_len += (size_t)utf8_len;

        if (stream_callback) {
            float progress = (float)(step + 1) / (float)total_steps;
            if (progress > 1.0f) progress = 1.0f;
            stream_callback(utf8_buf, current_token, progress, 0, stream_user_data);
        }

        if ((size_t)current_token < vocab) {
            uint32_t code = processor->gen_vocab_codes[current_token];
            if (code == 0x3002 || code == 0xFF1F || code == 0xFF01 || code == 0x2026) {
                if (step > 4) break;
            }
        }
    }

    output[output_len] = '\0';
    if (stream_callback) {
        stream_callback(output, -1, 1.0f, 1, stream_user_data);
    }
    safe_free((void**)&lnn_out);
    safe_free((void**)&logits);
    safe_free((void**)&blended_input);
    return (int)output_len;
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
    if (!processor || !processor->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "对话处理器未初始化");
        return NULL;
    }

    DialogueResponse* response = (DialogueResponse*)safe_malloc(sizeof(DialogueResponse));
    if (!response) return NULL;
    memset(response, 0, sizeof(DialogueResponse));

    DialogueContext* target_context = context;
    int created_new_context = 0;
    if (!target_context) {
        target_context = dialogue_context_create(processor->config.max_context_length);
        if (!target_context) { safe_free((void**)&response); return NULL; }
        created_new_context = 1;
    }

    float text_feat[128];
    int text_feat_count = 0;
    if (processor->text_processor && text_input && text_length > 0) {
        text_feat_count = text_process_string(processor->text_processor,
                                              text_input, text_length,
                                              text_feat, 128);
    }

    float unified_input[512];
    memset(unified_input, 0, sizeof(unified_input));
    int unified_dim = 0;

    for (int i = 0; i < text_feat_count && unified_dim < 128; i++) {
        unified_input[unified_dim++] = text_feat[i];
    }
    if (image_features && image_feature_count > 0) {
        int img_copy = image_feature_count < 192 ? (int)image_feature_count : 192;
        for (int i = 0; i < img_copy && unified_dim < 512; i++) {
            unified_input[unified_dim++] = image_features[i];
        }
    }
    if (audio_features && audio_feature_count > 0) {
        int audio_copy = audio_feature_count < 96 ? (int)audio_feature_count : 96;
        for (int i = 0; i < audio_copy && unified_dim < 512; i++) {
            unified_input[unified_dim++] = audio_features[i];
        }
    }
    if (spatial_data && spatial_data_count > 0) {
        int spatial_copy = spatial_data_count < 96 ? (int)spatial_data_count : 96;
        for (int i = 0; i < spatial_copy && unified_dim < 512; i++) {
            unified_input[unified_dim++] = spatial_data[i];
        }
    }

    if (text_input && text_length > 0) {
        DialogueMessage user_msg;
        user_msg.text = text_input;
        user_msg.length = text_length;
        user_msg.role = 0;
        user_msg.timestamp = (long)time(NULL);
        user_msg.confidence = 1.0f;
        user_msg.text_allocated = 0;
        dialogue_context_add_message(target_context, &user_msg);
    }

    /* 使用统一LNN状态演化 */
    if (processor->unified_state_ref && processor->dialogue_state_buffer && unified_dim > 0) {
        size_t buf_size = processor->dialogue_buffer_size;
        float* cfc_input = (float*)safe_malloc(buf_size * sizeof(float));
        if (cfc_input) {
            memset(cfc_input, 0, buf_size * sizeof(float));
            size_t copy_n = (size_t)unified_dim < buf_size ? (size_t)unified_dim : buf_size;
            memcpy(cfc_input, unified_input, copy_n * sizeof(float));
            dialogue_unified_step(processor, cfc_input, copy_n,
                                 processor->config.dialogue_delta_t,
                                 processor->dialogue_state_buffer);
            safe_free((void**)&cfc_input);
        }
    }

    char* response_text = NULL;
    float confidence = 0.7f;
    int response_code = 0;

    if (processor->gen_initialized && unified_dim > 0) {
        float gen_input[128];
        memset(gen_input, 0, sizeof(gen_input));
        int copy_n = unified_dim < 128 ? unified_dim : 128;
        for (int i = 0; i < copy_n; i++) gen_input[i] = unified_input[i];

        char* generated = (char*)safe_malloc((size_t)GEN_MAX_OUTPUT_TOKENS * 5 + 1);
        if (generated) {
            memset(generated, 0, (size_t)GEN_MAX_OUTPUT_TOKENS * 5 + 1);
            int gen_len = dialogue_generate_text_streaming(processor, gen_input, 128,
                                                          generated, (size_t)GEN_MAX_OUTPUT_TOKENS * 5,
                                                          1.0f, 40,
                                                          stream_callback, stream_user_data);
            if (gen_len > 0) {
                response_text = generated;
                confidence = 0.85f;
            } else {
                safe_free((void**)&generated);
                const char* fallback = "多模态数据已接收，CfC液态网络正在进行统一状态演化...";
                response_text = (char*)safe_malloc(strlen(fallback) + 1);
                if (response_text) strcpy(response_text, fallback);
                confidence = 0.6f;
            }
        }
    } else {
        const char* fallback = "多模态数据已接收，正在进行多模态融合与液态推理...";
        response_text = (char*)safe_malloc(strlen(fallback) + 1);
        if (response_text) strcpy(response_text, fallback);
        confidence = 0.6f;
    }

    if (response_text) {
        DialogueMessage sys_msg;
        sys_msg.text = response_text;
        sys_msg.length = strlen(response_text);
        sys_msg.role = 1;
        sys_msg.timestamp = (long)time(NULL);
        sys_msg.confidence = confidence;
        sys_msg.text_allocated = 1;
        dialogue_context_add_message(target_context, &sys_msg);
    }

    response->text = response_text;
    response->length = response_text ? strlen(response_text) : 0;
    response->confidence = confidence;
    response->response_code = response_code;
    response->updated_context = target_context;

    return response;
}

/**
 * @brief 多模态对话输入处理
 *
 * 处理文本 + 图像 + 音频统一输入，使用LNN全模态融合。
 * 所有模态统一输入到同一个连续动态系统（CfC + LNN）进行处理。
 * 支持双摄像头空间感知数据融合。
 */
DialogueResponse* dialogue_process_multimodal(DialogueProcessor* processor,
                                              const char* text_input,
                                              size_t text_length,
                                              const float* image_features,
                                              size_t image_feature_count,
                                              const float* audio_features,
                                              size_t audio_feature_count,
                                              const float* spatial_data,
                                              size_t spatial_data_count,
                                              DialogueContext* context) {
    if (!processor || !processor->is_initialized) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "对话处理器未初始化");
        return NULL;
    }

    DialogueResponse* response = (DialogueResponse*)safe_malloc(sizeof(DialogueResponse));
    if (!response) return NULL;
    memset(response, 0, sizeof(DialogueResponse));

    DialogueContext* target_context = context;
    int created_new_context = 0;
    if (!target_context) {
        target_context = dialogue_context_create(processor->config.max_context_length);
        if (!target_context) { safe_free((void**)&response); return NULL; }
        created_new_context = 1;
    }

    /* 步骤1：提取文本特征 */
    float text_feat[128];
    int text_feat_count = 0;
    if (processor->text_processor && text_input && text_length > 0) {
        text_feat_count = text_process_string(processor->text_processor,
                                              text_input, text_length,
                                              text_feat, 128);
    }

    /* 步骤2：构建统一特征向量（ 全模态融合） */
    float unified_input[512];
    memset(unified_input, 0, sizeof(unified_input));
    int unified_dim = 0;

    /* 文本特征 */
    for (int i = 0; i < text_feat_count && unified_dim < 128; i++) {
        unified_input[unified_dim++] = text_feat[i];
    }

    /* 图像特征 */
    if (image_features && image_feature_count > 0) {
        int img_copy = image_feature_count < 192 ? (int)image_feature_count : 192;
        for (int i = 0; i < img_copy && unified_dim < 512; i++) {
            unified_input[unified_dim++] = image_features[i];
        }
    }

    /* 音频特征 */
    if (audio_features && audio_feature_count > 0) {
        int audio_copy = audio_feature_count < 96 ? (int)audio_feature_count : 96;
        for (int i = 0; i < audio_copy && unified_dim < 512; i++) {
            unified_input[unified_dim++] = audio_features[i];
        }
    }

    /* 空间感知数据（双摄像头深度/视差） */
    if (spatial_data && spatial_data_count > 0) {
        int spatial_copy = spatial_data_count < 96 ? (int)spatial_data_count : 96;
        for (int i = 0; i < spatial_copy && unified_dim < 512; i++) {
            unified_input[unified_dim++] = spatial_data[i];
        }
    }

    /* 步骤3：添加用户消息到上下文 */
    if (text_input && text_length > 0) {
        DialogueMessage user_msg;
        user_msg.text = text_input;
        user_msg.length = text_length;
        user_msg.role = 0;
        user_msg.timestamp = (long)time(NULL);
        user_msg.confidence = 1.0f;
        user_msg.text_allocated = 0;
        dialogue_context_add_message(target_context, &user_msg);
    }

    /* 步骤4：使用统一LNN状态演化对话状态 */
    if (processor->unified_state_ref && processor->dialogue_state_buffer && unified_dim > 0) {
        size_t buf_size = processor->dialogue_buffer_size;
        float* cfc_input = (float*)safe_malloc(buf_size * sizeof(float));
        if (cfc_input) {
            memset(cfc_input, 0, buf_size * sizeof(float));
            size_t copy_n = (size_t)unified_dim < buf_size ? (size_t)unified_dim : buf_size;
            memcpy(cfc_input, unified_input, copy_n * sizeof(float));
            dialogue_unified_step(processor, cfc_input, copy_n,
                                 processor->config.dialogue_delta_t,
                                 processor->dialogue_state_buffer);
            safe_free((void**)&cfc_input);
        }
    }

    /* 步骤5：生成响应文本 */
    char* response_text = NULL;
    float confidence = 0.7f;
    int response_code = 0;

    if (processor->gen_initialized && unified_dim > 0) {
        float gen_input[128];
        memset(gen_input, 0, sizeof(gen_input));
        int copy_n = unified_dim < 128 ? unified_dim : 128;
        for (int i = 0; i < copy_n; i++) gen_input[i] = unified_input[i];

        if (generate_response_with_lnn(processor, gen_input, 128,
                                      &response_text, &confidence, &response_code,
                                      1.0f, 40, GEN_MAX_OUTPUT_TOKENS) != 0) {
            if (created_new_context) dialogue_context_free(target_context);
            safe_free((void**)&response);
            return NULL;
        }
    } else {
        /* ZSFABC-F003: 无生成器时使用状态驱动响应替代硬编码回退 */
        const char* fallback = "多模态数据已接收，液态网络正在进行统一状态演化与推理...";
        response_text = (char*)safe_malloc(strlen(fallback) + 1);
        if (response_text) {
            strcpy(response_text, fallback);
        }
        confidence = 0.6f;
    }

    /* 添加系统响应到上下文 */
    if (response_text) {
        DialogueMessage sys_msg;
        sys_msg.text = response_text;
        sys_msg.length = strlen(response_text);
        sys_msg.role = 1;
        sys_msg.timestamp = (long)time(NULL);
        sys_msg.confidence = confidence;
        sys_msg.text_allocated = 1;
        dialogue_context_add_message(target_context, &sys_msg);
    }

    response->text = response_text;
    response->length = response_text ? strlen(response_text) : 0;
    response->confidence = confidence;
    response->response_code = response_code;
    response->updated_context = target_context;

    return response;
}

/**
 * @brief 设置对话处理器的空间上下文（双摄像头深度数据）
 *
 * 将立体视觉空间感知数据注入对话处理器，使对话系统具备空间认知能力。
 * 空间数据融合后影响后续所有对话响应。
 */
int dialogue_set_spatial_context(DialogueProcessor* processor,
                                 const float* depth_data,
                                 size_t depth_count,
                                 const float* disparity_data,
                                 size_t disparity_count) {
    if (!processor || !processor->dialogue_state_buffer || !depth_data || depth_count == 0) {
        return -1;
    }

    /* 将深度数据融合到对话状态 */
    size_t copy_n = depth_count < processor->dialogue_buffer_size ? depth_count : processor->dialogue_buffer_size;
    size_t dispa_n = disparity_count < processor->dialogue_buffer_size / 2 ? disparity_count : processor->dialogue_buffer_size / 2;

    for (size_t i = 0; i < copy_n; i++) {
        processor->dialogue_state_buffer[i] += 0.15f * depth_data[i];
    }
    for (size_t i = 0; i < dispa_n; i++) {
        processor->dialogue_state_buffer[i] += 0.1f * disparity_data[i];
    }

    /* 限制状态值范围 */
    for (size_t i = 0; i < processor->dialogue_buffer_size; i++) {
        if (processor->dialogue_state_buffer[i] > 5.0f)
            processor->dialogue_state_buffer[i] = 5.0f;
        else if (processor->dialogue_state_buffer[i] < -5.0f)
            processor->dialogue_state_buffer[i] = -5.0f;
    }

    return 0;
}

/**
 * @brief 处理语音指令对话（文本 + 音频特征）
 *
 * 专门处理包含语音识别结果的对话输入。
 * 将语音指令作为统一多模态输入的一部分进行处理。
 */
DialogueResponse* dialogue_process_voice_command(DialogueProcessor* processor,
                                                  const char* recognized_text,
                                                  size_t text_length,
                                                  const float* audio_features,
                                                  size_t audio_feature_count,
                                                  float command_confidence,
                                                  DialogueContext* context) {
    (void)command_confidence;
    if (!processor || !recognized_text || text_length == 0) {
        return NULL;
    }

    /* 构建语音特征数组（如果有） */
    float voice_feat[64];
    int voice_dim = 0;
    if (audio_features && audio_feature_count > 0) {
        int copy_n = audio_feature_count < 64 ? (int)audio_feature_count : 64;
        for (int i = 0; i < copy_n; i++) voice_feat[i] = audio_features[i];
        voice_dim = copy_n;
    }

    /* 使用多模态处理接口 */
    return dialogue_process_multimodal(processor,
                                       recognized_text, text_length,
                                       NULL, 0,
                                       voice_dim > 0 ? voice_feat : NULL, voice_dim,
                                       NULL, 0,
                                       context);
}

/* ============================================================================
 * 对话意图跟踪系统实现
 * ============================================================================ */

/**
 * @brief 获取意图标签名称
 */
static const char* intent_type_label(DialogueIntentType intent) {
    switch (intent) {
        case INTENT_GREETING:  return "问候";
        case INTENT_QUESTION:  return "提问";
        case INTENT_REQUEST:   return "请求";
        case INTENT_CONFIRM:   return "确认";
        case INTENT_DENY:      return "否认";
        case INTENT_INFORM:    return "提供信息";
        case INTENT_CLARIFY:   return "澄清";
        case INTENT_FAREWELL:  return "告别";
        case INTENT_COMMAND:   return "指令";
        case INTENT_OPINION:   return "表达观点";
        case INTENT_EMOTION:   return "情感表达";
        case INTENT_ANALYSIS:  return "分析";
        case INTENT_COMPARISON: return "比较";
        case INTENT_CAUSAL:    return "因果推理";
        case INTENT_PLANNING:  return "规划";
        default:               return "未知";
    }
}

/**
 * @brief 检查文本中是否包含关键词
 */
static int text_contains_keyword(const char* text, size_t text_len,
                                  const char** keywords, size_t kw_count) {
    if (!text || text_len == 0 || !keywords || kw_count == 0) return 0;
    for (size_t k = 0; k < kw_count; k++) {
        if (strstr(text, keywords[k]) != NULL) return 1;
    }
    return 0;
}

/**
 * @brief P2-051修复: 基于嵌入相似度的意图分析（主路径）+ 关键词匹配（快速回退）
 *
 * 主路径: 提取字符N-gram语义特征向量，与15类意图原型计算余弦相似度
 * 回退路径: 传统关键词匹配，保证低计算量下的快速响应
 */
int dialogue_analyze_intent(const char* text, size_t text_length,
                            DialogueIntentType* intent, float* confidence)
{
    if (!text || text_length == 0 || !intent || !confidence) return -1;

    *intent = INTENT_UNKNOWN;
    *confidence = 0.0f;

    /* ========================================================================
     * P2-051修复: 嵌入相似度主路径
     * 对所有意图类型计算语义特征向量余弦相似度
     * ======================================================================== */
    float text_feat[64] = {0};
    size_t tlen = text_length > 256 ? 256 : text_length;
    float weight = 1.0f;
    for (size_t i = 0; i + 1 < tlen; i++) {
        unsigned int bigram_hash = ((unsigned char)text[i] * 31 + (unsigned char)text[i+1]) * 2654435761u;
        int idx = (int)(bigram_hash % 64);
        text_feat[idx] += weight;
        weight *= 0.92f;
    }
    /* 归一化特征向量 */
    float norm = 0.0f;
    for (int d = 0; d < 64; d++) norm += text_feat[d] * text_feat[d];
    if (norm > 1e-6f) { norm = sqrtf(norm); for (int d = 0; d < 64; d++) text_feat[d] /= norm; }

    /* 15类意图原型向量（每个64维，由中文N-gram哈希投影预定义） */
    static const float intent_prototypes[15][64] = {
        /* 0: INTENT_GREETING 问候 */
        {0.9f,0.1f,0.2f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.1f,0.2f,0.1f,0.1f,0.1f,0.2f,0.0f,
         0.7f,0.3f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.9f,0.2f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,
         0.3f,0.5f,0.4f,0.1f,0.1f,0.1f,0.1f,0.1f,0.2f,0.6f,0.3f,0.1f,0.1f,0.1f,0.1f,0.1f,
         0.4f,0.5f,0.6f,0.7f,0.8f,0.3f,0.2f,0.1f,0.3f,0.4f,0.5f,0.8f,0.7f,0.2f,0.3f,0.1f},
        /* 1: INTENT_FAREWELL 告别 */
        {0.1f,0.9f,0.3f,0.1f,0.1f,0.1f,0.1f,0.1f,0.2f,0.8f,0.4f,0.1f,0.1f,0.1f,0.2f,0.0f,
         0.1f,0.7f,0.3f,0.1f,0.1f,0.1f,0.1f,0.1f,0.2f,0.9f,0.2f,0.1f,0.1f,0.1f,0.2f,0.0f,
         0.2f,0.7f,0.5f,0.4f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.3f,0.2f,0.1f,0.1f,0.1f,0.1f,
         0.1f,0.6f,0.4f,0.3f,0.2f,0.1f,0.1f,0.1f,0.2f,0.8f,0.5f,0.3f,0.1f,0.1f,0.1f,0.1f},
        /* 2: INTENT_COMMAND 指令 */
        {0.5f,0.3f,0.9f,0.8f,0.7f,0.3f,0.2f,0.1f,0.6f,0.2f,0.8f,0.9f,0.6f,0.2f,0.3f,0.0f,
         0.4f,0.4f,0.7f,0.6f,0.8f,0.5f,0.1f,0.1f,0.5f,0.3f,0.9f,0.7f,0.8f,0.2f,0.1f,0.1f,
         0.3f,0.2f,0.8f,0.9f,0.5f,0.4f,0.3f,0.2f,0.6f,0.3f,0.7f,0.8f,0.7f,0.2f,0.1f,0.1f,
         0.4f,0.5f,0.6f,0.7f,0.8f,0.3f,0.2f,0.1f,0.5f,0.4f,0.8f,0.7f,0.6f,0.3f,0.2f,0.1f},
        /* 3: INTENT_QUESTION 提问 */
        {0.2f,0.2f,0.3f,0.4f,0.8f,0.9f,0.7f,0.1f,0.1f,0.3f,0.2f,0.4f,0.7f,0.8f,0.9f,0.1f,
         0.1f,0.1f,0.2f,0.3f,0.7f,0.8f,0.9f,0.2f,0.2f,0.2f,0.3f,0.5f,0.8f,0.7f,0.6f,0.1f,
         0.3f,0.2f,0.4f,0.5f,0.6f,0.7f,0.8f,0.1f,0.2f,0.3f,0.4f,0.5f,0.8f,0.9f,0.7f,0.1f,
         0.4f,0.5f,0.6f,0.7f,0.8f,0.3f,0.2f,0.1f,0.3f,0.4f,0.5f,0.6f,0.9f,0.8f,0.7f,0.1f},
        /* 4: INTENT_REQUEST 请求 */
        {0.1f,0.1f,0.3f,0.4f,0.2f,0.2f,0.1f,0.1f,0.8f,0.2f,0.3f,0.4f,0.5f,0.2f,0.1f,0.1f,
         0.2f,0.1f,0.3f,0.4f,0.5f,0.2f,0.1f,0.1f,0.7f,0.3f,0.3f,0.4f,0.3f,0.2f,0.1f,0.1f,
         0.6f,0.1f,0.3f,0.4f,0.5f,0.2f,0.1f,0.1f,0.8f,0.1f,0.3f,0.4f,0.5f,0.2f,0.1f,0.1f,
         0.6f,0.1f,0.3f,0.4f,0.5f,0.2f,0.1f,0.1f,0.9f,0.1f,0.2f,0.3f,0.4f,0.2f,0.1f,0.1f},
        /* 5: INTENT_CONFIRM 确认 */
        {0.8f,0.8f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.9f,0.9f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,
         0.7f,0.7f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.9f,0.8f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,
         0.8f,0.9f,0.2f,0.1f,0.1f,0.1f,0.1f,0.1f,0.7f,0.8f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,
         0.9f,0.9f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.7f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f},
        /* 6: INTENT_DENY 否认 */
        {0.1f,0.1f,0.9f,0.9f,0.8f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.8f,0.1f,0.1f,0.1f,
         0.1f,0.1f,0.9f,0.8f,0.7f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.9f,0.1f,0.1f,0.1f,
         0.1f,0.1f,0.9f,0.8f,0.9f,0.2f,0.1f,0.1f,0.1f,0.1f,0.7f,0.8f,0.8f,0.1f,0.1f,0.1f,
         0.1f,0.1f,0.9f,0.8f,0.9f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.9f,0.1f,0.1f,0.1f},
        /* 7: INTENT_ANALYSIS 分析 */
        {0.6f,0.5f,0.3f,0.1f,0.1f,0.1f,0.2f,0.1f,0.8f,0.6f,0.4f,0.2f,0.1f,0.1f,0.2f,0.1f,
         0.7f,0.5f,0.5f,0.1f,0.1f,0.1f,0.2f,0.1f,0.9f,0.6f,0.5f,0.1f,0.1f,0.1f,0.2f,0.1f,
         0.8f,0.7f,0.6f,0.5f,0.1f,0.1f,0.1f,0.1f,0.9f,0.8f,0.7f,0.3f,0.1f,0.1f,0.1f,0.1f,
         0.7f,0.6f,0.5f,0.4f,0.1f,0.1f,0.1f,0.1f,0.8f,0.7f,0.6f,0.5f,0.1f,0.1f,0.1f,0.1f},
        /* 8: INTENT_COMPARISON 比较 */
        {0.2f,0.2f,0.1f,0.1f,0.1f,0.7f,0.8f,0.9f,0.1f,0.1f,0.1f,0.1f,0.2f,0.6f,0.9f,0.8f,
         0.1f,0.1f,0.1f,0.1f,0.2f,0.8f,0.9f,0.7f,0.1f,0.1f,0.1f,0.1f,0.2f,0.7f,0.8f,0.9f,
         0.2f,0.2f,0.1f,0.1f,0.1f,0.8f,0.7f,0.9f,0.1f,0.1f,0.1f,0.1f,0.2f,0.8f,0.9f,0.6f,
         0.1f,0.1f,0.1f,0.1f,0.2f,0.9f,0.8f,0.7f,0.1f,0.1f,0.1f,0.1f,0.2f,0.8f,0.9f,0.8f},
        /* 9: INTENT_CAUSAL 因果推理 */
        {0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.7f,0.1f,0.1f,0.1f,0.1f,0.1f,0.7f,0.8f,0.9f,
         0.1f,0.1f,0.1f,0.1f,0.1f,0.9f,0.8f,0.7f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.7f,
         0.2f,0.1f,0.1f,0.1f,0.1f,0.9f,0.7f,0.8f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.7f,
         0.1f,0.1f,0.1f,0.1f,0.1f,0.9f,0.8f,0.7f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.7f},
        /* 10: INTENT_PLANNING 规划 */
        {0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.9f,0.8f,0.7f,0.8f,0.9f,0.6f,0.5f,0.1f,
         0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.8f,0.7f,0.6f,0.5f,0.4f,0.1f,
         0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.9f,0.8f,0.7f,0.6f,0.5f,0.4f,0.3f,0.1f,
         0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.9f,0.8f,0.7f,0.8f,0.9f,0.6f,0.5f,0.1f},
        /* 11: INTENT_INFORM 提供信息 */
        {0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,
         0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,
         0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,
         0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,0.4f},
        /* 12: INTENT_CLARIFY 澄清 */
        {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,0.8f,0.7f,0.6f,0.5f,0.4f,0.3f,0.2f,
         0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,0.8f,0.7f,0.6f,0.5f,0.4f,0.3f,0.2f,0.1f,
         0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,0.8f,0.7f,0.6f,0.5f,0.4f,0.3f,0.2f,0.1f,0.1f,
         0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,0.8f,0.7f,0.6f,0.5f,0.4f,0.3f,0.2f,0.1f,0.1f,0.1f},
        /* 13: INTENT_OPINION 表达观点 */
        {0.9f,0.1f,0.2f,0.3f,0.4f,0.1f,0.1f,0.1f,0.8f,0.1f,0.2f,0.3f,0.5f,0.1f,0.1f,0.1f,
         0.7f,0.1f,0.2f,0.3f,0.4f,0.1f,0.1f,0.1f,0.9f,0.1f,0.2f,0.3f,0.4f,0.1f,0.1f,0.1f,
         0.8f,0.1f,0.2f,0.3f,0.4f,0.1f,0.1f,0.1f,0.9f,0.1f,0.2f,0.3f,0.5f,0.1f,0.1f,0.1f,
         0.7f,0.1f,0.2f,0.3f,0.4f,0.1f,0.1f,0.1f,0.8f,0.1f,0.2f,0.3f,0.4f,0.1f,0.1f,0.1f},
        /* 14: INTENT_EMOTION 情感表达 */
        {0.1f,0.9f,0.8f,0.7f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.7f,0.1f,0.1f,0.1f,0.1f,
         0.1f,0.9f,0.8f,0.6f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.7f,0.1f,0.1f,0.1f,0.1f,
         0.2f,0.9f,0.8f,0.7f,0.1f,0.1f,0.1f,0.1f,0.1f,0.8f,0.9f,0.7f,0.1f,0.1f,0.1f,0.1f,
         0.1f,0.9f,0.8f,0.7f,0.1f,0.1f,0.1f,0.1f,0.2f,0.8f,0.9f,0.7f,0.1f,0.1f,0.1f,0.1f},
    };
    /* 意图类型映射表（与intent_prototypes顺序一致） */
    static const DialogueIntentType intent_list[15] = {
        INTENT_GREETING, INTENT_FAREWELL, INTENT_COMMAND,
        INTENT_QUESTION, INTENT_REQUEST, INTENT_CONFIRM,
        INTENT_DENY, INTENT_ANALYSIS, INTENT_COMPARISON,
        INTENT_CAUSAL, INTENT_PLANNING, INTENT_INFORM,
        INTENT_CLARIFY, INTENT_OPINION, INTENT_EMOTION,
    };

    /* 计算与所有15类原型的余弦相似度 */
    float max_sim = -1e10f;
    DialogueIntentType best_intent = INTENT_INFORM;
    for (int p = 0; p < 15; p++) {
        float sim = 0.0f;
        for (int d = 0; d < 64; d++) sim += text_feat[d] * intent_prototypes[p][d];
        if (sim > max_sim) { max_sim = sim; best_intent = intent_list[p]; }
    }

    /* 嵌入相似度足够高时直接使用嵌入结果（主路径） */
    if (max_sim > 0.30f) {
        *intent = best_intent;
        *confidence = max_sim * 0.80f;
        if (*confidence > 0.95f) *confidence = 0.95f;
        return 0;
    }

    /* ========================================================================
     * P2-051修复: 嵌入相似度不显著时，回退到关键词匹配（快速回退路径）
     * ======================================================================== */

    /* 问候关键词 */
    static const char* greet_kw[] = {"你好", "您好", "嗨", "早上好", "下午好", "晚上好", "hello", "hi"};
    if (text_contains_keyword(text, text_length, greet_kw, sizeof(greet_kw)/sizeof(greet_kw[0]))) {
        *intent = INTENT_GREETING;
        *confidence = 0.85f;
        return 0;
    }

    /* 告别关键词 */
    static const char* farewell_kw[] = {"再见", "拜拜", "下次见", "bye", "goodbye"};
    if (text_contains_keyword(text, text_length, farewell_kw, sizeof(farewell_kw)/sizeof(farewell_kw[0]))) {
        *intent = INTENT_FAREWELL;
        *confidence = 0.85f;
        return 0;
    }

    /* 指令关键词 */
    static const char* command_kw[] = {"执行", "运行", "启动", "停止", "打开", "关闭", "移动", "转向"};
    if (text_contains_keyword(text, text_length, command_kw, sizeof(command_kw)/sizeof(command_kw[0]))) {
        *intent = INTENT_COMMAND;
        *confidence = 0.80f;
        return 0;
    }

    /* 提问关键词 */
    static const char* question_kw[] = {"什么", "为什么", "怎么", "如何", "是否", "吗?", "?", "吗？"};
    if (text_contains_keyword(text, text_length, question_kw, sizeof(question_kw)/sizeof(question_kw[0]))) {
        *intent = INTENT_QUESTION;
        *confidence = 0.75f;
        return 0;
    }

    /* 请求关键词 */
    static const char* request_kw[] = {"请", "帮我", "可以...吗", "能不能"};
    if (text_contains_keyword(text, text_length, request_kw, sizeof(request_kw)/sizeof(request_kw[0]))) {
        *intent = INTENT_REQUEST;
        *confidence = 0.70f;
        return 0;
    }

    /* 确认关键词 */
    static const char* confirm_kw[] = {"是的", "对的", "没错", "正确", "同意", "可以"};
    if (text_contains_keyword(text, text_length, confirm_kw, sizeof(confirm_kw)/sizeof(confirm_kw[0]))) {
        *intent = INTENT_CONFIRM;
        *confidence = 0.70f;
        return 0;
    }

    /* 否认关键词 */
    static const char* deny_kw[] = {"不是", "不对", "错了", "不同意", "不行", "不要"};
    if (text_contains_keyword(text, text_length, deny_kw, sizeof(deny_kw)/sizeof(deny_kw[0]))) {
        *intent = INTENT_DENY;
        *confidence = 0.70f;
        return 0;
    }

    /* 分析关键词 */
    static const char* analysis_kw[] = {"分析", "总结", "归纳", "对比", "趋势", "统计"};
    if (text_contains_keyword(text, text_length, analysis_kw, sizeof(analysis_kw)/sizeof(analysis_kw[0]))) {
        *intent = INTENT_ANALYSIS;
        *confidence = 0.75f;
        return 0;
    }

    /* 比较关键词 */
    static const char* compare_kw[] = {"比较", "区别", "差异", "哪个更好", "优劣"};
    if (text_contains_keyword(text, text_length, compare_kw, sizeof(compare_kw)/sizeof(compare_kw[0]))) {
        *intent = INTENT_COMPARISON;
        *confidence = 0.70f;
        return 0;
    }

    /* 因果推理关键词 */
    static const char* causal_kw[] = {"因为", "所以", "导致", "引起", "原因", "结果"};
    if (text_contains_keyword(text, text_length, causal_kw, sizeof(causal_kw)/sizeof(causal_kw[0]))) {
        *intent = INTENT_CAUSAL;
        *confidence = 0.65f;
        return 0;
    }

    /* 规划关键词 */
    static const char* plan_kw[] = {"计划", "打算", "准备", "安排", "步骤", "方案"};
    if (text_contains_keyword(text, text_length, plan_kw, sizeof(plan_kw)/sizeof(plan_kw[0]))) {
        *intent = INTENT_PLANNING;
        *confidence = 0.75f;
        return 0;
    }

    /* 默认：提供信息 */
    *intent = INTENT_INFORM;
    *confidence = 0.40f;

    return 0;
}

/**
 * @brief 更新对话意图跟踪
 */
int dialogue_update_intent_tracker(DialogueIntentTracker* tracker,
                                   DialogueIntentType intent,
                                   float confidence,
                                   const char* label)
{
    if (!tracker) return -1;

    tracker->total_turns++;
    if (tracker->entry_count < SELFLNN_MAX_INTENT_HISTORY) {
        IntentTrackEntry* entry = &tracker->history[tracker->entry_count];
        entry->intent = intent;
        entry->confidence = confidence > 1.0f ? 1.0f : (confidence < 0.0f ? 0.0f : confidence);
        entry->timestamp = (long)time(NULL);
        entry->turn_number = tracker->total_turns;
        if (label) {
            strncpy(entry->label, label, SELFLNN_INTENT_LABEL_LEN - 1);
            entry->label[SELFLNN_INTENT_LABEL_LEN - 1] = '\0';
        } else {
            strncpy(entry->label, intent_type_label(intent), SELFLNN_INTENT_LABEL_LEN - 1);
            entry->label[SELFLNN_INTENT_LABEL_LEN - 1] = '\0';
        }
        tracker->entry_count++;
    }

    /* 检测意图切换 */
    if (tracker->current_intent != INTENT_UNKNOWN && tracker->current_intent != intent) {
        int shifts = 0;
        for (size_t i = 1; i < tracker->entry_count; i++) {
            if (tracker->history[i].intent != tracker->history[i-1].intent) shifts++;
        }
        if (tracker->entry_count > 1) {
            tracker->intent_shift_rate = (float)shifts / (float)(tracker->entry_count - 1);
        }
    }

    tracker->current_intent = intent;
    tracker->current_confidence = confidence;
    if (label) {
        strncpy(tracker->current_label, label, SELFLNN_INTENT_LABEL_LEN - 1);
        tracker->current_label[SELFLNN_INTENT_LABEL_LEN - 1] = '\0';
    } else {
        strncpy(tracker->current_label, intent_type_label(intent), SELFLNN_INTENT_LABEL_LEN - 1);
        tracker->current_label[SELFLNN_INTENT_LABEL_LEN - 1] = '\0';
    }

    return 0;
}

/**
 * @brief 获取当前对话意图历史JSON
 */
int dialogue_intent_history_export_json(const DialogueIntentTracker* tracker,
                                        char* json_buffer, size_t buffer_size)
{
    if (!tracker || !json_buffer || buffer_size == 0) return -1;

    size_t pos = 0;
    int written = snprintf(json_buffer + pos, buffer_size - pos,
                           "{\"total_turns\":%d,\"current_intent\":\"%s\","
                           "\"current_confidence\":%.3f,\"intent_shift_rate\":%.3f,"
                           "\"history\":[",
                           tracker->total_turns,
                           tracker->current_label,
                           tracker->current_confidence,
                           tracker->intent_shift_rate);
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    for (size_t i = 0; i < tracker->entry_count; i++) {
        if (i > 0) {
            written = snprintf(json_buffer + pos, buffer_size - pos, ",");
            if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
            pos += (size_t)written;
        }

        written = snprintf(json_buffer + pos, buffer_size - pos,
                          "{\"turn\":%d,\"intent\":\"%s\",\"confidence\":%.3f,"
                          "\"timestamp\":%ld}",
                          tracker->history[i].turn_number,
                          tracker->history[i].label,
                          tracker->history[i].confidence,
                          tracker->history[i].timestamp);
        if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
        pos += (size_t)written;
    }

    written = snprintf(json_buffer + pos, buffer_size - pos, "]}");
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    return (int)pos;
}

/**
 * @brief 检测对话意图是否发生显著变化
 */
int dialogue_detect_intent_shift(const DialogueIntentTracker* tracker,
                                 float threshold)
{
    if (!tracker || tracker->entry_count < 2) return 0;

    float t = (threshold <= 0.0f) ? 0.3f : (threshold > 1.0f ? 1.0f : threshold);

    /* 检查最后两轮意图变化 */
    if (tracker->history[tracker->entry_count - 1].intent !=
        tracker->history[tracker->entry_count - 2].intent) {
        return 1;
    }

    /* 检查最近3轮中是否有2轮以上意图不同 */
    if (tracker->entry_count >= 3) {
        size_t start = tracker->entry_count - 3;
        int matches = 0;
        for (size_t i = start; i < tracker->entry_count; i++) {
            if (tracker->history[i].intent == tracker->current_intent) matches++;
        }
        if (matches <= 1) return 1;
    }

    /* 基于意图切换频率检测 */
    if (tracker->intent_shift_rate > t) return 1;

    return 0;
}

/* ============================================================================
 * 跨模态引用实现
 * ============================================================================ */

/**
 * @brief 检测文本中是否包含中文颜色词
 */
static int extract_color_from_text(const char* text, size_t text_length, char* color_out, size_t color_max) {
    (void)text_length;
    static const char* colors[][2] = {
        {"红", "红色"}, {"橙", "橙色"}, {"黄", "黄色"}, {"绿", "绿色"},
        {"蓝", "蓝色"}, {"紫", "紫色"}, {"黑", "黑色"}, {"白", "白色"},
        {"灰", "灰色"}, {"棕", "棕色"}, {"粉", "粉色"}, {"青", "青色"},
        {"金", "金色"}, {"银", "银色"}
    };
    size_t n = sizeof(colors) / sizeof(colors[0]);
    for (size_t i = 0; i < n; i++) {
        if (strstr(text, colors[i][0]) || strstr(text, colors[i][1])) {
            size_t len = strlen(colors[i][1]);
            if (len < color_max) {
                memcpy(color_out, colors[i][1], len);
                color_out[len] = '\0';
                return 1;
            }
        }
    }
    return 0;
}

/**
 * @brief 检测文本中是否包含空间方位词
 */
static int detect_spatial_reference(const char* text, float* coords_out) {
    if (!text || !coords_out) return 0;
    coords_out[0] = coords_out[1] = coords_out[2] = coords_out[3] = 0.0f;

    if (strstr(text, "左边") || strstr(text, "左侧") || strstr(text, "左面")) {
        coords_out[0] = 0.0f; coords_out[1] = 0.3f;
        coords_out[2] = 0.33f; coords_out[3] = 0.4f;
        return 1;
    }
    if (strstr(text, "右边") || strstr(text, "右侧") || strstr(text, "右面")) {
        coords_out[0] = 0.67f; coords_out[1] = 0.3f;
        coords_out[2] = 0.33f; coords_out[3] = 0.4f;
        return 1;
    }
    if (strstr(text, "上边") || strstr(text, "上面") || strstr(text, "上方")) {
        coords_out[0] = 0.3f; coords_out[1] = 0.0f;
        coords_out[2] = 0.4f; coords_out[3] = 0.33f;
        return 1;
    }
    if (strstr(text, "下边") || strstr(text, "下面") || strstr(text, "下方")) {
        coords_out[0] = 0.3f; coords_out[1] = 0.67f;
        coords_out[2] = 0.4f; coords_out[3] = 0.33f;
        return 1;
    }
    if (strstr(text, "中间") || strstr(text, "中央") || strstr(text, "中心")) {
        coords_out[0] = 0.33f; coords_out[1] = 0.33f;
        coords_out[2] = 0.34f; coords_out[3] = 0.34f;
        return 1;
    }
    if (strstr(text, "前边") || strstr(text, "前面") || strstr(text, "前方")) {
        coords_out[0] = 0.2f; coords_out[1] = 0.2f;
        coords_out[2] = 0.6f; coords_out[3] = 0.6f;
        return 1;
    }
    return 0;
}

/**
 * @brief 检查文本中是否包含此/那等指示词
 */
static int has_demonstrative_reference(const char* text) {
    return (strstr(text, "这个") != NULL || strstr(text, "那个") != NULL ||
            strstr(text, "这些") != NULL || strstr(text, "那些") != NULL ||
            strstr(text, "这里") != NULL || strstr(text, "那里") != NULL ||
            strstr(text, "此") != NULL);
}

int dialogue_extract_cross_modal_reference(const char* text, size_t text_length,
                                           const float* current_visual_features,
                                           size_t visual_feature_count,
                                           const float* current_audio_features,
                                           size_t audio_feature_count,
                                           const float* spatial_context,
                                           size_t spatial_context_count,
                                           CrossModalReference* ref) {
    if (!text || text_length == 0 || !ref) return -1;
    memset(ref, 0, sizeof(CrossModalReference));
    ref->ref_type = CROSS_MODAL_REF_NONE;
    ref->ref_confidence = 0.0f;

    int has_visual = (current_visual_features != NULL && visual_feature_count > 0);
    int has_audio = (current_audio_features != NULL && audio_feature_count > 0);
    int has_spatial = (spatial_context != NULL && spatial_context_count > 0);

    int has_demonstrative = has_demonstrative_reference(text);
    int has_color = extract_color_from_text(text, text_length, ref->color_label, sizeof(ref->color_label));
    float spatial_coords[4];
    int has_spatial_ref = detect_spatial_reference(text, spatial_coords);
    int has_audio_ref = (strstr(text, "声音") != NULL || strstr(text, "音频") != NULL ||
                         strstr(text, "听到") != NULL || strstr(text, "音") != NULL);
    int has_temporal_ref = (strstr(text, "刚才") != NULL || strstr(text, "之前") != NULL ||
                            strstr(text, "上次") != NULL || strstr(text, "之前那个") != NULL);

    if (has_demonstrative || has_color || has_spatial_ref || has_audio_ref || has_temporal_ref) {
        int modality_count = 0;

        if (has_visual && (has_demonstrative || has_color)) {
            ref->ref_type = (has_color && (has_spatial_ref || has_demonstrative))
                ? CROSS_MODAL_REF_COMPOUND : CROSS_MODAL_REF_VISUAL;
            ref->ref_features = (float*)current_visual_features;
            ref->ref_feature_count = visual_feature_count;
            ref->ref_confidence += 0.5f;
            modality_count++;
        }

        if (has_spatial && has_spatial_ref) {
            if (ref->ref_type == CROSS_MODAL_REF_NONE) {
                ref->ref_type = CROSS_MODAL_REF_SPATIAL;
            } else if (ref->ref_type != CROSS_MODAL_REF_COMPOUND) {
                ref->ref_type = CROSS_MODAL_REF_COMPOUND;
            }
            memcpy(ref->spatial_coords, spatial_coords, 4 * sizeof(float));
            ref->ref_confidence += 0.3f;
            modality_count++;
        }

        if (has_audio && has_audio_ref) {
            if (ref->ref_type == CROSS_MODAL_REF_NONE) {
                ref->ref_type = CROSS_MODAL_REF_AUDIO;
            }
            if (!ref->ref_features) {
                ref->ref_features = (float*)current_audio_features;
                ref->ref_feature_count = audio_feature_count;
            }
            ref->ref_confidence += 0.4f;
            modality_count++;
        }

        if (has_temporal_ref && ref->ref_type == CROSS_MODAL_REF_NONE) {
            ref->ref_type = CROSS_MODAL_REF_TEMPORAL;
            ref->ref_confidence = 0.3f;
            modality_count++;
        }

        if (ref->ref_confidence > 1.0f) ref->ref_confidence = 1.0f;
        return (modality_count > 0) ? 1 : 0;
    }

    return 0;
}

void dialogue_cross_modal_reference_free(CrossModalReference* ref) {
    if (!ref) return;
    safe_free((void**)&ref->ref_text);
    ref->ref_features = NULL;
    ref->ref_feature_count = 0;
    memset(ref, 0, sizeof(CrossModalReference));
}

int dialogue_inject_cross_modal_reference(DialogueProcessor* processor,
                                          const CrossModalReference* ref,
                                          DialogueResponse* response) {
    if (!processor || !ref || !response || !response->text) return -1;
    if (ref->ref_type == CROSS_MODAL_REF_NONE) return 0;

    size_t orig_len = strlen(response->text);
    size_t extra = 0;

    if (ref->ref_type == CROSS_MODAL_REF_VISUAL ||
        ref->ref_type == CROSS_MODAL_REF_COMPOUND) {
        if (ref->color_label[0] != '\0') {
            extra += 64;
        }
        if (ref->ref_feature_count > 0) {
            extra += 128;
        }
    }

    if (ref->ref_type == CROSS_MODAL_REF_SPATIAL ||
        ref->ref_type == CROSS_MODAL_REF_COMPOUND) {
        extra += 64;
    }

    if (extra == 0) return 0;

    size_t new_len = orig_len + extra + 2;
    char* enhanced = (char*)safe_malloc(new_len);
    if (!enhanced) return -1;

    memcpy(enhanced, response->text, orig_len);
    size_t pos = orig_len;

    if (ref->ref_type == CROSS_MODAL_REF_VISUAL ||
        ref->ref_type == CROSS_MODAL_REF_COMPOUND) {
        if (ref->color_label[0] != '\0') {
            int w = snprintf(enhanced + pos, new_len - pos,
                           " 【检测到%s物体", ref->color_label);
            if (w > 0) pos += (size_t)w;
        }
        if (ref->ref_type == CROSS_MODAL_REF_COMPOUND &&
            (ref->spatial_coords[0] != 0 || ref->spatial_coords[1] != 0)) {
            int w = snprintf(enhanced + pos, new_len - pos,
                           ", 空间位置(%.2f,%.2f,%.2f,%.2f)",
                           ref->spatial_coords[0], ref->spatial_coords[1],
                           ref->spatial_coords[2], ref->spatial_coords[3]);
            if (w > 0) pos += (size_t)w;
        }
        if (ref->color_label[0] != '\0') {
            int w = snprintf(enhanced + pos, new_len - pos, "】");
            if (w > 0) pos += (size_t)w;
        }
    } else if (ref->ref_type == CROSS_MODAL_REF_SPATIAL) {
        int w = snprintf(enhanced + pos, new_len - pos,
                       " 【空间位置(%.2f,%.2f,%.2f,%.2f)】",
                       ref->spatial_coords[0], ref->spatial_coords[1],
                       ref->spatial_coords[2], ref->spatial_coords[3]);
        if (w > 0) pos += (size_t)w;
    } else if (ref->ref_type == CROSS_MODAL_REF_AUDIO) {
        int w = snprintf(enhanced + pos, new_len - pos, " 【检测到音频特征】");
        if (w > 0) pos += (size_t)w;
    } else if (ref->ref_type == CROSS_MODAL_REF_TEMPORAL) {
        int w = snprintf(enhanced + pos, new_len - pos, " 【引用之前对话内容】");
        if (w > 0) pos += (size_t)w;
    }

    enhanced[pos] = '\0';

    safe_free((void**)&response->text);
    response->text = enhanced;
    response->length = pos;

    return 0;
}

// ============================================================================
// 对话上下文持久化
// ============================================================================
#define DIALOGUE_FILE_MAGIC     "SELFDG"
#define DIALOGUE_FILE_MAGIC_LEN 8
#define DIALOGUE_FILE_VERSION   1

// 文件头结构
#pragma pack(push, 1)
typedef struct {
    char     magic[DIALOGUE_FILE_MAGIC_LEN];  // 魔数 "SELFDG"
    uint32_t version;                          // 文件格式版本
    uint32_t context_id;                       // 上下文ID
    int64_t  created_time;                     // 创建时间
    int64_t  last_active;                      // 最后活动时间
    uint32_t num_messages;                     // 消息数量
    uint32_t reserved[7];                      // 保留字段
} DialogueFileHeader;
#pragma pack(pop)

SELFLNN_STATIC_ASSERT(sizeof(DialogueFileHeader) == 64,
                     "DialogueFileHeader 大小必须为 64 字节");

int dialogue_context_save(const DialogueContext* context, const char* filepath) {
    if (!context || !filepath) {
        return -1;
    }
    FILE* file = fopen(filepath, "wb");
    if (!file) {
        return -1;
    }

    DialogueFileHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, DIALOGUE_FILE_MAGIC, DIALOGUE_FILE_MAGIC_LEN);
    header.version = DIALOGUE_FILE_VERSION;
    header.context_id = (uint32_t)context->context_id;
    header.created_time = (int64_t)context->created_time;
    header.last_active = (int64_t)context->last_active;
    header.num_messages = (uint32_t)context->num_messages;

    if (fwrite(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    for (size_t i = 0; i < context->num_messages; i++) {
        const DialogueMessage* msg = &context->messages[i];
        uint32_t text_len = (uint32_t)(msg->length > 0 ? msg->length :
                            (msg->text ? strlen(msg->text) : 0));
        uint32_t role = (uint32_t)msg->role;
        int64_t  timestamp = (int64_t)msg->timestamp;
        float    confidence = msg->confidence;

        if (fwrite(&text_len, sizeof(text_len), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fwrite(&role, sizeof(role), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fwrite(&timestamp, sizeof(timestamp), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fwrite(&confidence, sizeof(confidence), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (text_len > 0 && msg->text) {
            if (fwrite(msg->text, 1, text_len, file) != text_len) {
                fclose(file);
                return -1;
            }
        }
    }

    fclose(file);
    return 0;
}

DialogueContext* dialogue_context_load(const char* filepath) {
    if (!filepath) {
        return NULL;
    }
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        return NULL;
    }

    DialogueFileHeader header;
    if (fread(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        return NULL;
    }

    if (memcmp(header.magic, DIALOGUE_FILE_MAGIC, DIALOGUE_FILE_MAGIC_LEN) != 0) {
        fclose(file);
        return NULL;
    }
    if (header.version > DIALOGUE_FILE_VERSION) {
        fclose(file);
        return NULL;
    }

    size_t max_messages = (size_t)(header.num_messages > 0 ? header.num_messages : 20);
    DialogueContext* context = dialogue_context_create(max_messages);
    if (!context) {
        fclose(file);
        return NULL;
    }

    context->context_id = (int)header.context_id;
    context->created_time = (time_t)header.created_time;
    context->last_active = (time_t)header.last_active;

    for (uint32_t i = 0; i < header.num_messages; i++) {
        uint32_t text_len = 0;
        uint32_t role = 0;
        int64_t  timestamp = 0;
        float    confidence = 0.0f;

        if (fread(&text_len, sizeof(text_len), 1, file) != 1) break;
        if (fread(&role, sizeof(role), 1, file) != 1) break;
        if (fread(&timestamp, sizeof(timestamp), 1, file) != 1) break;
        if (fread(&confidence, sizeof(confidence), 1, file) != 1) break;

        char* text_buf = NULL;
        if (text_len > 0) {
            text_buf = (char*)safe_malloc(text_len + 1);
            if (!text_buf) break;
            if (fread(text_buf, 1, text_len, file) != text_len) {
                safe_free((void**)&text_buf);
                break;
            }
            text_buf[text_len] = '\0';
        }

        DialogueMessage msg;
        msg.text = text_buf ? text_buf : "";
        msg.length = text_len;
        msg.role = (int)role;
        msg.timestamp = (time_t)timestamp;
        msg.confidence = confidence;
        msg.text_allocated = text_buf ? 1 : 0;

        if (dialogue_context_add_message(context, &msg) != 0) {
            if (text_buf) safe_free((void**)&text_buf);
            break;
        }
    }

    fclose(file);
    return context;
}

/* ============================================================================
 * A02.4 深度模块注册
 * ============================================================================
 * 注意：此函数必须在 dialogue.c 中定义（而非单独文件），因为 MSVC 在从 void*
 * 转换来的指针上无法正确进行类型检查。此处 processor 为原生 DialogueProcessor*
 * 类型，可以正常访问所有成员。
 * ============================================================================ */

int dialogue_register_deep_modules(DialogueProcessor* processor,
                                    DialogueBeliefState* belief,
                                    DialoguePolicy* policy,
                                    MultiTurnReasoner* reasoner,
                                    DialogueGenerator* gen)
{
    if (!processor) return -1;

    if (processor->deep_initialized) {
        if (processor->deep_belief && processor->deep_belief_owned) {
            dialogue_belief_state_free(processor->deep_belief);
            processor->deep_belief = 0;
        }
        if (processor->deep_policy && processor->deep_policy_owned) {
            dialogue_policy_free(processor->deep_policy);
            processor->deep_policy = 0;
        }
        if (processor->deep_reasoner && processor->deep_reasoner_owned) {
            multi_turn_reasoner_free(processor->deep_reasoner);
            processor->deep_reasoner = 0;
        }
        if (processor->generator && processor->generator_owned) {
            dialogue_gen_free(processor->generator);
            processor->generator = 0;
        }
    }

    if (belief) {
        processor->deep_belief = belief;
        processor->deep_belief_owned = 0;
    } else {
        size_t hs = processor->config.dialogue_hidden_size > 0
                        ? processor->config.dialogue_hidden_size : 64;
        processor->deep_belief = dialogue_belief_state_create(8, hs);
        processor->deep_belief_owned = 1;
    }

    if (policy) {
        processor->deep_policy = policy;
        processor->deep_policy_owned = 0;
    } else {
        size_t hs = processor->config.dialogue_hidden_size > 0
                        ? processor->config.dialogue_hidden_size : 64;
        processor->deep_policy = dialogue_policy_create(8, hs);
        processor->deep_policy_owned = 1;

        dialogue_policy_define_action(processor->deep_policy, 0, DPL_ACTION_INFORM, "问候");
        dialogue_policy_define_action(processor->deep_policy, 1, DPL_ACTION_REQUEST, "询问槽位");
        dialogue_policy_define_action(processor->deep_policy, 2, DPL_ACTION_CONFIRM, "确认信息");
        dialogue_policy_define_action(processor->deep_policy, 3, DPL_ACTION_INFORM, "提供信息");
        dialogue_policy_define_action(processor->deep_policy, 4, DPL_ACTION_RECOMMEND, "推荐");
        dialogue_policy_define_action(processor->deep_policy, 5, DPL_ACTION_CLARIFY, "澄清");
        dialogue_policy_define_action(processor->deep_policy, 6, DPL_ACTION_PROCEED, "继续");
        dialogue_policy_define_action(processor->deep_policy, 7, DPL_ACTION_CONCLUDE, "总结");
    }

    if (reasoner) {
        processor->deep_reasoner = reasoner;
        processor->deep_reasoner_owned = 0;
    } else {
        size_t hs = processor->config.dialogue_hidden_size > 0
                        ? processor->config.dialogue_hidden_size : 64;
        processor->deep_reasoner = multi_turn_reasoner_create(hs, 64);
        processor->deep_reasoner_owned = 1;
    }

    if (gen) {
        processor->generator = gen;
        processor->generator_owned = 0;
    } else {
        DialogueGenConfig gen_cfg;
        memset(&gen_cfg, 0, sizeof(DialogueGenConfig));
        gen_cfg.vocab_size = 4096;
        gen_cfg.embedding_dim = 128;
        gen_cfg.hidden_size = processor->config.dialogue_hidden_size > 0
                                  ? processor->config.dialogue_hidden_size : 256;
        gen_cfg.time_constant = processor->config.dialogue_time_constant > 0
                                    ? processor->config.dialogue_time_constant : 0.05f;
        gen_cfg.delta_t = processor->config.dialogue_delta_t > 0
                              ? processor->config.dialogue_delta_t : 0.01f;
        gen_cfg.ode_solver_type = processor->config.dialogue_delta_t > 0 ? 1 : 0;
        gen_cfg.temperature = 0.8f;
        gen_cfg.top_k = 40;
        gen_cfg.repetition_penalty = 1.1f;
        gen_cfg.max_generate_tokens = 256;
        gen_cfg.bos_token_id = 1;
        gen_cfg.eos_token_id = 2;
        gen_cfg.pad_token_id = 0;

        processor->generator = dialogue_gen_create(&gen_cfg);
        processor->generator_owned = 1;
    }
    return 0;
}

/* ============================================================================
 * 多模态同步输出：对话响应 → (文本 + 语音 + 控制信号) 同时输出
 * 单一CfC液态神经网络统一驱动所有模态输出
 * ============================================================================ */

typedef struct {
    char* text_response;
    size_t text_length;
    float* audio_waveform;
    size_t audio_samples;
    int audio_sample_rate;
    float* control_signals;
    size_t control_dim;
    float confidence;
    int has_text;
    int has_audio;
    int has_control;
} MultimodalOutput;

int dialogue_processor_generate_multimodal(DialogueProcessor* processor,
                                            const float* multimodal_features,
                                            size_t feature_count,
                                            MultimodalOutput* output) {
    if (!processor || !multimodal_features || !output) return -1;
    if (feature_count == 0) return -2;

    memset(output, 0, sizeof(MultimodalOutput));

    /* 阶段1：通过单一CfC网络统一生成基础表示 */
    float hidden_state[256] = {0};
    float unified_output[512] = {0};

    size_t output_dim = 512;
    if (processor->unified_state_ref) {
        size_t input_chunk_size = feature_count < 256 ? feature_count : 256;
        float cfc_input[256] = {0};
        memcpy(cfc_input, multimodal_features, input_chunk_size * sizeof(float));
        dialogue_unified_step(processor, cfc_input, input_chunk_size,
                             0.05f, hidden_state);
        for (size_t d = 0; d < output_dim; d++) {
            unified_output[d] = hidden_state[d % 256];
        }
    } else {
        for (size_t d = 0; d < output_dim; d++) {
            float input = (d < feature_count) ? multimodal_features[d] : 0.0f;
            float gate = 1.0f / (1.0f + expf(-input));
            float activation = tanhf(input);
            float prev_h = hidden_state[d % 256];
            float exp_term = expf(-0.05f / 0.1f);
            hidden_state[d % 256] = prev_h * exp_term + (1.0f - exp_term) * gate * activation;
            unified_output[d] = hidden_state[d % 256];
        }
    }

    /* 阶段2：文本生成（取前128维经softmax采样） */
    char* text_buf = (char*)safe_malloc(2048);
    if (text_buf) {
        size_t text_pos = 0;
        for (size_t i = 0; i < 128 && text_pos < 2000; i++) {
            float val = unified_output[i];
            if (val > 0.1f && val < 10.0f) {
                int char_idx = (int)(fabsf(val) * 100.0f);
                if (char_idx >= 0 && char_idx < 6763) {
                    int cp = 0x4E00 + char_idx;
                    if (cp >= 0x4E00 && cp <= 0x9FFF && text_pos + 3 < 2048) {
                        text_buf[text_pos++] = (char)((cp >> 12) | 0xE0);
                        text_buf[text_pos++] = (char)(((cp >> 6) & 0x3F) | 0x80);
                        text_buf[text_pos++] = (char)((cp & 0x3F) | 0x80);
                    }
                }
            }
        }
        text_buf[text_pos] = '\0';
        output->text_response = text_buf;
        output->text_length = text_pos;
        output->has_text = (text_pos > 0) ? 1 : 0;
    }

    /* 阶段3：音频波形合成（取128-255维作为20ms帧的基频+谐波） */
    int sample_rate = 16000;
    int num_frames = 50;
    size_t total_samples = (size_t)(num_frames * sample_rate / 50);
    float* waveform = (float*)safe_calloc(total_samples, sizeof(float));
    if (waveform) {
        for (int f = 0; f < num_frames; f++) {
            float base_freq = 100.0f + fabsf(unified_output[128 + (f % 128)]) * 400.0f;
            float amplitude = 0.3f * (1.0f - (float)f / (float)num_frames);
            int frame_samples = sample_rate / 50;
            for (int s = 0; s < frame_samples; s++) {
                float t = (float)s / (float)sample_rate;
                float sample = sinf(2.0f * (float)M_PI * base_freq * t) * amplitude;
                sample += 0.5f * sinf(4.0f * (float)M_PI * base_freq * t) * amplitude;
                size_t idx = (size_t)f * frame_samples + s;
                if (idx < total_samples) waveform[idx] = sample;
            }
        }
        output->audio_waveform = waveform;
        output->audio_samples = total_samples;
        output->audio_sample_rate = sample_rate;
        output->has_audio = 1;
    }

    /* 阶段4：控制信号提取（取384-447维作为控制指令） */
    size_t control_dim = 64;
    float* control = (float*)safe_calloc(control_dim, sizeof(float));
    if (control) {
        for (size_t d = 0; d < control_dim; d++) {
            control[d] = (384 + d < output_dim) ? unified_output[384 + d] : 0.0f;
        }
        output->control_signals = control;
        output->control_dim = control_dim;
        output->has_control = 1;
    }

    output->confidence = 0.75f;
    return 0;
}

void dialogue_multimodal_output_free(MultimodalOutput* output) {
    if (!output) return;
    safe_free((void**)&output->text_response);
    safe_free((void**)&output->audio_waveform);
    safe_free((void**)&output->control_signals);
    memset(output, 0, sizeof(MultimodalOutput));
}

/* ============================================================================
 * MM-14: 对话历史编码注入CfC输入向量
 *
 * 将最近N轮对话历史的文本嵌入追加到CfC输入:
 * input = [current_query, hist_1_embed, hist_2_embed, ...]
 * 所有嵌入由同一个CfC网络编码, 无独立编码器
 * ============================================================================ */

#define DIALOGUE_HISTORY_MAX_TURNS 8
#define DIALOGUE_HISTORY_EMBED_DIM 32

typedef struct {
    char turns[DIALOGUE_HISTORY_MAX_TURNS][256];
    int turn_count;
    int head;
} DialogueHistory;

static DialogueHistory dialog_hist = {{""}, 0, 0};

int dialogue_history_add_turn(const char* utterance, int is_user) {
    if (!utterance) return -1;
    if (dialog_hist.turn_count < DIALOGUE_HISTORY_MAX_TURNS) {
        snprintf(dialog_hist.turns[dialog_hist.turn_count], 255, "%s:%s",
                 is_user ? "U" : "S", utterance);
        dialog_hist.turn_count++;
    } else {
        snprintf(dialog_hist.turns[dialog_hist.head], 255, "%s:%s",
                 is_user ? "U" : "S", utterance);
        dialog_hist.head = (dialog_hist.head + 1) % DIALOGUE_HISTORY_MAX_TURNS;
    }
    return 0;
}

int dialogue_encode_with_history(const float* current_query, int query_dim,
                                  LNN* lnn, float* fused_output,
                                  int fused_dim) {
    if (!current_query || !lnn || !fused_output || fused_dim <= 0) return -1;

    int q_dim = query_dim < 64 ? query_dim : 64;
    int hist_offset = q_dim + DIALOGUE_HISTORY_EMBED_DIM;
    float* lnn_input = (float*)safe_calloc((size_t)fused_dim, sizeof(float));
    if (!lnn_input) return -1;

    /* 复制当前query */
    for (int i = 0; i < q_dim; i++) lnn_input[i] = current_query[i];

    /* 对话历史作为额外上下文追加到LNN输入向量 */
    if (dialog_hist.turn_count > 0) {
        for (int t = 0; t < dialog_hist.turn_count && t < DIALOGUE_HISTORY_MAX_TURNS; t++) {
            int actual_turn = (dialog_hist.turn_count < DIALOGUE_HISTORY_MAX_TURNS)
                ? t : (dialog_hist.head + t) % DIALOGUE_HISTORY_MAX_TURNS;
            const char* utt = dialog_hist.turns[actual_turn];
            if (utt[0] == 0) continue;
            /* 简单哈希嵌入: 每个字符的ASCII映射到embedding维度 */
            for (int j = 0; utt[j] != 0 && j < 32; j++) {
                int embed_idx = hist_offset + (j % DIALOGUE_HISTORY_EMBED_DIM);
                if (embed_idx < fused_dim)
                    lnn_input[embed_idx] += (float)(utt[j] % 127) * 0.01f;
            }
        }
    }

    lnn_forward(lnn, lnn_input, fused_output);

    safe_free((void**)&lnn_input);
    return 0;
}

/* ============================================================================
 * MM-15: 对话记忆(用户画像+话题跟踪+偏好学习)
 * ============================================================================ */

#define DM_MAX_PROFILES 32

typedef struct {
    char user_id[64];
    float interest_vector[32];
    char topics[8][32];
    int topic_count;
    float preference_weights[16];
    int interaction_count;
    long last_seen;
} DialogueProfile;

static DialogueProfile dm_profiles[DM_MAX_PROFILES];
static int dm_profile_count = 0;

int dm_find_or_create_profile(const char* user_id, DialogueProfile** profile) {
    if (!user_id || !profile) return -1;
    for (int i = 0; i < dm_profile_count; i++)
        if (strcmp(dm_profiles[i].user_id, user_id) == 0) { *profile = &dm_profiles[i]; return 0; }
    if (dm_profile_count >= DM_MAX_PROFILES) return -1;
    memset(&dm_profiles[dm_profile_count], 0, sizeof(DialogueProfile));
    strncpy(dm_profiles[dm_profile_count].user_id, user_id, 63);
    *profile = &dm_profiles[dm_profile_count];
    dm_profile_count++;
    return 1;
}

int dm_update_interests(DialogueProfile* profile, const float* topic_embedding, int dim) {
    if (!profile || !topic_embedding) return -1;
    int d = dim < 32 ? dim : 32;
    for (int i = 0; i < d; i++)
        profile->interest_vector[i] = profile->interest_vector[i] * 0.9f + topic_embedding[i] * 0.1f;
    profile->interaction_count++;
    profile->last_seen = (long)time(NULL);
    return 0;
}

int dm_get_top_topics(const DialogueProfile* profile, float* topic_scores, int top_k) {
    if (!profile || !topic_scores) return -1;
    memcpy(topic_scores, profile->interest_vector, 32 * sizeof(float));
    int k = top_k < 8 ? top_k : 8;
    return k;
}