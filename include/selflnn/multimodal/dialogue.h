/**
 * @file dialogue.h
 * @brief 对话系统接口
 * 
 * 对话系统接口，支持自然语言对话处理、上下文管理和响应生成。
 * 使用CfC（Closed-form Continuous-time）细胞单元进行连续时间对话状态演化，
 * 液态神经网络（LNN）进行语义特征提取。
 * 根据项目要求"使用单一液态神经网络模型"，对话通过统一动态系统进行处理。
 */

#ifndef SELFLNN_DIALOGUE_H
#define SELFLNN_DIALOGUE_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

/* 前向声明LNN类型，避免循环依赖 */
typedef struct LNN LNN;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 对话消息结构体
 */
typedef struct {
    const char* text;           /**< 消息文本 */
    size_t length;              /**< 文本长度 */
    int role;                   /**< 角色：0=用户，1=系统 */
    time_t timestamp;           /**< 时间戳 */
    float confidence;           /**< 置信度 (0-1) */
    int text_allocated;         /**< 文本是否由对话模块分配：1=是，0=否 */
} DialogueMessage;

/**
 * @brief 对话上下文结构体
 */
typedef struct {
    DialogueMessage* messages;  /**< 消息数组 */
    size_t max_messages;        /**< 最大消息数 */
    size_t num_messages;        /**< 当前消息数 */
    int context_id;             /**< 上下文ID */
    time_t created_time;        /**< 创建时间 */
    time_t last_active;         /**< 最后活动时间 */
} DialogueContext;

/**
 * @brief 对话配置结构体
 */
typedef struct {
    int max_context_length;     /**< 最大上下文长度（消息数） */
    int enable_context_memory;  /**< 是否启用上下文记忆 */
    int response_generation_mode; /**< 响应生成模式：0=简单，1=深度 */
    float confidence_threshold; /**< 置信度阈值 */
    int language;               /**< 语言：0=中文，1=英文 */
    size_t dialogue_hidden_size; /**< 对话状态隐藏维度（默认128） */
    float dialogue_time_constant;/**< 对话时间常数（默认0.1） */
    float dialogue_delta_t;     /**< 对话时间步长（默认0.05） */
    int use_cfc_evolution;     /**< 是否使用CfC ODE演化对话状态（默认1） */
} DialogueConfig;

/**
 * @brief 对话响应结构体
 */
typedef struct {
    const char* text;           /**< 响应文本 */
    size_t length;              /**< 响应长度 */
    float confidence;           /**< 置信度 (0-1) */
    int response_code;          /**< 响应代码：0=成功，1=不理解，2=需要更多信息 */
    DialogueContext* updated_context; /**< 更新后的上下文（可选） */
} DialogueResponse;

/**
 * @brief 对话处理器句柄
 */
typedef struct DialogueProcessor DialogueProcessor;

#include "selflnn/multimodal/text.h"

/* 前向声明 */
typedef struct DialogueBeliefState DialogueBeliefState;
typedef struct DialoguePolicy DialoguePolicy;
typedef struct MultiTurnReasoner MultiTurnReasoner;
typedef struct DialogueGenerator DialogueGenerator;

/**
 * @brief 对话处理器内部结构
 */
struct DialogueProcessor {
    DialogueConfig config;           /**< 处理器配置 */
    int is_initialized;              /**< 是否已初始化 */
    void* lnn_instance;              /**< 液态神经网络实例句柄 */
    int lnn_owned;                   /**< LNN所有权标记 */

    float* dialogue_state_buffer;    /**< 对话状态缓冲区 */
    size_t dialogue_buffer_size;     /**< 对话状态缓冲区维度 */

    TextProcessor* text_processor;   /**< 文本处理器句柄 */
    DialogueContext** contexts;      /**< 对话上下文数组 */
    size_t max_contexts;             /**< 最大上下文数 */
    size_t num_contexts;             /**< 当前上下文数 */

    void* unified_state_ref;         /**< 全局统一LNN状态引用（单一模型原则） */

    int gen_initialized;             /**< 生成器是否已初始化 */
    size_t gen_vocab_size;           /**< 词汇表大小 */
    size_t gen_hidden_dim;           /**< 嵌入/隐藏层维度 */
    uint32_t* gen_vocab_codes;       /**< Unicode码点数组 */
    char* gen_vocab_utf8_buf;        /**< UTF-8编码字符串 */
    float* gen_embeddings;           /**< Token嵌入 */
    LNN* gen_projection_lnn;         /**< LNN词嵌入投影网络（替代线性矩阵） */
    float* gen_projection_w;         /**< 投影权重（回退路径） */
    float* gen_projection_b;         /**< 投影偏置（回退路径） */

    /* Phase1: 对话私有ODE状态 — 消除对共享LNN hidden_state的污染 */
    float* gen_private_hidden;       /**< 对话私有ODE隐藏状态 */
    float gen_private_tau;           /**< 私有ODE时间常数 */
    int gen_private_dim;             /**< 私有状态维度 */

    DialogueBeliefState* deep_belief;     /**< DST信念状态 */
    int deep_belief_owned;
    DialoguePolicy* deep_policy;          /**< 对话策略 */
    int deep_policy_owned;
    MultiTurnReasoner* deep_reasoner;     /**< 多轮推理器 */
    int deep_reasoner_owned;
    DialogueGenerator* generator; /**< 对话生成器 */
    int generator_owned;
    int deep_initialized;
};

/**
 * @brief 创建对话处理器
 * 
 * @param config 对话配置
 * @return DialogueProcessor* 对话处理器句柄，失败返回NULL
 */
DialogueProcessor* dialogue_processor_create(const DialogueConfig* config);

/**
 * @brief 释放对话处理器
 * 
 * @param processor 对话处理器句柄
 */
void dialogue_processor_free(DialogueProcessor* processor);

/**
 * @brief 处理对话输入
 * 
 * 处理用户输入，生成系统响应。深度集成液态神经网络进行对话理解和生成。
 * 
 * @param processor 对话处理器句柄
 * @param user_input 用户输入文本
 * @param input_length 输入文本长度
 * @param context 对话上下文（可为NULL，表示新对话）
 * @return DialogueResponse* 对话响应，调用者负责释放，失败返回NULL
 */
DialogueResponse* dialogue_process_input(DialogueProcessor* processor,
                                        const char* user_input,
                                        size_t input_length,
                                        DialogueContext* context);

/**
 * @brief 处理对话输入（扩展版）
 * 
 * 与dialogue_process_input相同，但支持指定生成参数。
 * 
 * @param processor 对话处理器句柄
 * @param user_input 用户输入文本
 * @param input_length 输入文本长度
 * @param context 对话上下文（可为NULL，表示新对话）
 * @param temperature 采样温度（0.1-2.0，默认1.0）
 * @param top_k top-k采样参数（默认40，0表示禁用）
 * @param max_tokens 最大生成token数（默认256）
 * @return DialogueResponse* 对话响应，调用者负责释放，失败返回NULL
 */
DialogueResponse* dialogue_process_input_ext(DialogueProcessor* processor,
                                            const char* user_input,
                                            size_t input_length,
                                            DialogueContext* context,
                                            float temperature,
                                            int top_k,
                                            int max_tokens);

/**
 * @brief 将对话上下文导出为JSON字符串
 * 
 * 用于前端获取对话历史。
 * 
 * @param context 对话上下文
 * @return char* JSON字符串，调用者负责释放，失败返回NULL
 */
char* dialogue_context_export_json(const DialogueContext* context);

/**
 * @brief 创建对话上下文
 * 
 * @param max_messages 最大消息数
 * @return DialogueContext* 对话上下文，调用者负责释放，失败返回NULL
 */
DialogueContext* dialogue_context_create(size_t max_messages);

/**
 * @brief 释放对话上下文
 * 
 * @param context 对话上下文
 */
void dialogue_context_free(DialogueContext* context);

/**
 * @brief 添加消息到对话上下文
 * 
 * @param context 对话上下文
 * @param message 消息
 * @return int 成功返回0，失败返回-1
 */
int dialogue_context_add_message(DialogueContext* context, const DialogueMessage* message);

/**
 * @brief 获取对话上下文摘要
 * 
 * @param context 对话上下文
 * @param summary 摘要输出缓冲区
 * @param max_length 最大长度
 * @return int 成功返回摘要长度，失败返回-1
 */
int dialogue_context_get_summary(const DialogueContext* context,
                                char* summary, size_t max_length);

/**
 * @brief 清除对话上下文
 * 
 * @param context 对话上下文
 */
void dialogue_context_clear(DialogueContext* context);

/**
 * @brief 设置液态神经网络实例
 * 
 * 将对话处理器连接到液态神经网络实例，用于深度对话理解和生成。
 * 
 * @param processor 对话处理器句柄
 * @param lnn_instance 液态神经网络实例句柄
 * @return int 成功返回0，失败返回-1
 */
int dialogue_set_lnn_instance(DialogueProcessor* processor, void* lnn_instance);

/**
 * @brief 重置对话状态缓冲区
 * 
 * @param processor 对话处理器句柄
 * @return int 成功返回0，失败返回-1
 */
int dialogue_reset_state(DialogueProcessor* processor);

/**
 * @brief 使用CfC ODE演化对话状态
 *
 * 将输入特征向量通过多模态CfC单元进行连续时间演化，
 * 更新对话状态缓冲区。替代简单EMA更新方式。
 *
 * @param processor 对话处理器句柄
 * @param input_features 输入特征向量
 * @param feature_count 特征维度
 * @param delta_t 时间步长（<=0时使用配置默认值）
 * @return int 成功返回0，失败返回-1
 */
int dialogue_evolve_state(DialogueProcessor* processor,
                         const float* input_features,
                         size_t feature_count,
                         float delta_t);

/**
 * @brief 获取对话处理器配置
 * 
 * @param processor 对话处理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int dialogue_processor_get_config(const DialogueProcessor* processor, DialogueConfig* config);

/**
 * @brief 设置对话处理器配置
 * 
 * @param processor 对话处理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int dialogue_processor_set_config(DialogueProcessor* processor, const DialogueConfig* config);

/**
 * @brief 初始化对话生成器（词汇表和嵌入）
 * 
 * 初始化自回归文本生成所需的词汇表和嵌入矩阵。
 * 在启用深度响应生成模式（response_generation_mode=1）前必须调用。
 * 
 * @param processor 对话处理器句柄
 * @param hidden_dim LNN隐藏层维度（通常与LNN输出维度一致）
 * @return int 成功返回0，失败返回-1
 */
int dialogue_init_generator(DialogueProcessor* processor, size_t hidden_dim);

/**
 * @brief 使用LNN自回归生成文本
 * 
 * 基于上下文特征向量，使用LNN的隐藏状态进行自回归文本生成。
 * 生成过程中使用温度采样和top-k采样策略。
 * 
 * @param processor 对话处理器句柄
 * @param context_features 上下文特征向量
 * @param context_size 上下文特征大小
 * @param output 输出文本缓冲区
 * @param max_output 输出缓冲区最大长度
 * @param temperature 采样温度（0.1-2.0，默认1.0）
 * @param top_k top-k采样参数（默认40，0表示禁用）
 * @return int 成功返回生成文本长度，失败返回-1
 */
int dialogue_generate_text(DialogueProcessor* processor,
                          const float* context_features,
                          size_t context_size,
                          char* output,
                          size_t max_output,
                          float temperature,
                          int top_k);

/**
 * @brief 释放对话响应
 * 
 * @param response 对话响应
 */
void dialogue_response_free(DialogueResponse* response);

/**
 * @brief 多模态对话输入处理
 *
 * 处理文本 + 图像 + 音频统一输入，使用LNN全模态融合。
 * 所有模态统一输入到同一个连续动态系统（CfC + LNN）进行处理。
 * 支持双摄像头空间感知数据融合。
 *
 * @param processor 对话处理器句柄
 * @param text_input 文本输入（可为NULL）
 * @param text_length 文本长度
 * @param image_features 图像特征向量（可为NULL）
 * @param image_feature_count 图像特征数量
 * @param audio_features 音频特征向量（可为NULL）
 * @param audio_feature_count 音频特征数量
 * @param spatial_data 空间感知数据（深度/视差，可为NULL）
 * @param spatial_data_count 空间数据数量
 * @param context 对话上下文（可为NULL）
 * @return DialogueResponse* 对话响应，调用者负责释放，失败返回NULL
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
                                              DialogueContext* context);

/**
 * @brief 设置对话处理器的空间上下文（双摄像头深度数据）
 *
 * 将立体视觉空间感知数据注入对话处理器，使对话系统具备空间认知能力。
 *
 * @param processor 对话处理器句柄
 * @param depth_data 深度数据数组
 * @param depth_count 深度数据数量
 * @param disparity_data 视差数据数组（可为NULL）
 * @param disparity_count 视差数据数量
 * @return int 成功返回0，失败返回-1
 */
int dialogue_set_spatial_context(DialogueProcessor* processor,
                                 const float* depth_data,
                                 size_t depth_count,
                                 const float* disparity_data,
                                 size_t disparity_count);

/**
 * @brief 处理语音指令对话（文本 + 音频特征）
 *
 * 专门处理包含语音识别结果的对话输入。
 * 将语音指令作为统一多模态输入的一部分进行处理。
 *
 * @param processor 对话处理器句柄
 * @param recognized_text 语音识别文本
 * @param text_length 文本长度
 * @param audio_features 音频特征向量（可为NULL）
 * @param audio_feature_count 音频特征数量
 * @param command_confidence 语音指令置信度
 * @param context 对话上下文（可为NULL）
 * @return DialogueResponse* 对话响应，调用者负责释放，失败返回NULL
 */
DialogueResponse* dialogue_process_voice_command(DialogueProcessor* processor,
                                                  const char* recognized_text,
                                                  size_t text_length,
                                                  const float* audio_features,
                                                  size_t audio_feature_count,
                                                  float command_confidence,
                                                  DialogueContext* context);

/* ============================================================================
 * 跨模态引用系统
 *
 * 支持多模态对话中的跨模态引用解析。
 * 处理"这个红色的物体是什么？""左边的那个东西"等包含视觉/空间引用的自然语言。
 * ============================================================================ */

/**
 * @brief 跨模态引用类型
 */
typedef enum {
    CROSS_MODAL_REF_NONE = 0,       /**< 无跨模态引用 */
    CROSS_MODAL_REF_VISUAL = 1,     /**< 视觉引用（"这个红色的物体"） */
    CROSS_MODAL_REF_AUDIO = 2,      /**< 音频引用（"这个声音"） */
    CROSS_MODAL_REF_SPATIAL = 3,    /**< 空间引用（"左边那个"） */
    CROSS_MODAL_REF_TEMPORAL = 4,   /**< 时序引用（"刚才那个"） */
    CROSS_MODAL_REF_COMPOUND = 5    /**< 复合引用（"左边那个红色的"） */
} CrossModalRefType;

/**
 * @brief 跨模态引用解析结果
 */
typedef struct {
    CrossModalRefType ref_type;     /**< 引用类型 */
    char* ref_text;                 /**< 引用文本片段 */
    float* ref_features;            /**< 引用的模态特征向量指针 */
    size_t ref_feature_count;       /**< 引用特征数量 */
    float spatial_coords[4];        /**< 空间坐标 [x, y, width, height] */
    char color_label[32];           /**< 颜色标签（如"红色""蓝色"） */
    int ref_frame_id;               /**< 引用帧ID */
    float ref_confidence;           /**< 引用匹配置信度 */
} CrossModalReference;

/**
 * @brief 从文本中提取跨模态引用信息
 *
 * 分析用户输入文本，提取"这个""那个""红色的""左边的"等跨模态引用关键词，
 * 返回解析后的引用类型和关联特征。
 *
 * @param text 用户输入文本
 * @param text_length 文本长度
 * @param current_visual_features 当前视觉特征缓冲区（可为NULL）
 * @param visual_feature_count 视觉特征数量
 * @param current_audio_features 当前音频特征缓冲区（可为NULL）
 * @param audio_feature_count 音频特征数量
 * @param spatial_context 空间上下文数据（可为NULL）
 * @param spatial_context_count 空间上下文数量
 * @param ref [out] 解析出的跨模态引用
 * @return int 找到引用返回1，未找到返回0，失败返回-1
 */
int dialogue_extract_cross_modal_reference(const char* text, size_t text_length,
                                           const float* current_visual_features,
                                           size_t visual_feature_count,
                                           const float* current_audio_features,
                                           size_t audio_feature_count,
                                           const float* spatial_context,
                                           size_t spatial_context_count,
                                           CrossModalReference* ref);

/**
 * @brief 释放跨模态引用资源
 *
 * @param ref 跨模态引用
 */
void dialogue_cross_modal_reference_free(CrossModalReference* ref);

/**
 * @brief 将跨模态引用信息注入对话响应
 *
 * 在对话响应中嵌入引用对象的特征信息，使系统能回答"这个红色的物体是什么"等问题。
 *
 * @param processor 对话处理器句柄
 * @param ref 跨模态引用
 * @param response 对话响应（将被修改添加引用信息）
 * @return int 成功返回0，失败返回-1
 */
int dialogue_inject_cross_modal_reference(DialogueProcessor* processor,
                                          const CrossModalReference* ref,
                                          DialogueResponse* response);

/* ============================================================================
 * 对话意图跟踪系统
 *
 * 支持对话意图识别、跟踪、轮次管理。
 * 意图类型涵盖 10 轮以上的自然对话。
 * ============================================================================ */

#define SELFLNN_MAX_INTENT_HISTORY 32
#define SELFLNN_INTENT_LABEL_LEN 64

/**
 * @brief 对话意图类型
 */
typedef enum {
    INTENT_UNKNOWN = 0,           /**< 未知意图 */
    INTENT_GREETING = 1,          /**< 问候 */
    INTENT_QUESTION = 2,          /**< 提问 */
    INTENT_REQUEST = 3,           /**< 请求 */
    INTENT_CONFIRM = 4,           /**< 确认 */
    INTENT_DENY = 5,              /**< 否认/拒绝 */
    INTENT_INFORM = 6,            /**< 提供信息 */
    INTENT_CLARIFY = 7,           /**< 澄清 */
    INTENT_FAREWELL = 8,          /**< 告别 */
    INTENT_COMMAND = 9,           /**< 指令 */
    INTENT_OPINION = 10,          /**< 表达观点 */
    INTENT_EMOTION = 11,          /**< 情感表达 */
    INTENT_ANALYSIS = 12,         /**< 分析 */
    INTENT_COMPARISON = 13,       /**< 比较 */
    INTENT_CAUSAL = 14,           /**< 因果推理 */
    INTENT_PLANNING = 15,         /**< 规划 */
} DialogueIntentType;

/**
 * @brief 意图跟踪条目
 */
typedef struct {
    DialogueIntentType intent;                    /**< 意图类型 */
    char label[SELFLNN_INTENT_LABEL_LEN];         /**< 意图标签 */
    float confidence;                             /**< 置信度 (0-1) */
    long timestamp;                               /**< 时间戳 */
    int turn_number;                              /**< 对话轮次号 */
} IntentTrackEntry;

/**
 * @brief 对话意图跟踪状态
 */
typedef struct {
    IntentTrackEntry history[SELFLNN_MAX_INTENT_HISTORY]; /**< 意图历史 */
    size_t entry_count;                                   /**< 条目数 */
    DialogueIntentType current_intent;                    /**< 当前意图 */
    char current_label[SELFLNN_INTENT_LABEL_LEN];         /**< 当前意图标签 */
    float current_confidence;                             /**< 当前置信度 */
    float intent_shift_rate;                              /**< 意图切换频率 (0-1) */
    int total_turns;                                      /**< 总对话轮次 */
} DialogueIntentTracker;

/**
 * @brief 分析用户输入的意图
 *
 * 基于关键词和上下文分析用户意图。
 *
 * @param text 用户输入文本
 * @param text_length 文本长度
 * @param intent [out] 识别出的意图类型
 * @param confidence [out] 识别置信度
 * @return int 成功返回0，失败返回-1
 */
int dialogue_analyze_intent(const char* text, size_t text_length,
                            DialogueIntentType* intent, float* confidence);

/**
 * @brief 获取全局对话处理器引用（供LNN驱动意图分析使用）
 * @return DialogueProcessor* 全局处理器句柄，未创建时返回NULL
 */
DialogueProcessor* dialogue_get_global_processor(void);

/**
 * @brief 解析对话响应文本，提取意图和置信度
 * @param response_text 响应文本
 * @param intent [out] 解析出的意图类型
 * @param confidence [out] 解析置信度
 * @return int 成功返回0，失败返回-1
 */
int dialogue_response_parse(const char* response_text,
                            DialogueIntentType* intent, float* confidence);

/**
 * @brief 创建对话记忆系统
 * @param capacity 记忆容量
 * @return void* 记忆系统句柄，失败返回NULL
 */
void* dialogue_memory_create(size_t capacity);

/**
 * @brief 释放对话记忆系统（兼容void*接口）
 * @param memory_handle 记忆系统句柄
 */
void dialogue_memory_free(void* memory_handle);

/**
 * @brief 更新对话意图跟踪
 *
 * 将当前轮次的意图更新到跟踪器中。
 *
 * @param tracker 意图跟踪器句柄
 * @param intent 识别的意图类型
 * @param confidence 置信度
 * @param label 意图标签字符串
 * @return int 成功返回0，失败返回-1
 */
int dialogue_update_intent_tracker(DialogueIntentTracker* tracker,
                                   DialogueIntentType intent,
                                   float confidence,
                                   const char* label);

/**
 * @brief 获取当前对话意图历史JSON
 *
 * @param tracker 意图跟踪器句柄
 * @param json_buffer [out] JSON字符串缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回写入字符数，失败返回-1
 */
int dialogue_intent_history_export_json(const DialogueIntentTracker* tracker,
                                        char* json_buffer, size_t buffer_size);

/**
 * @brief 检测对话意图是否发生显著变化
 *
 * @param tracker 意图跟踪器句柄
 * @param threshold 变化检测阈值 (0-1)，默认0.3
 * @return int 发生显著变化返回1，未变化返回0，失败返回-1
 */
int dialogue_detect_intent_shift(const DialogueIntentTracker* tracker,
                                 float threshold);

/**
 * @brief 保存对话上下文到文件
 *
 * 将对话上下文序列化到文件，支持后续恢复。
 * 存储格式：魔数(8B) + 版本(4B) + 上下文字段 + 消息数组。
 *
 * @param context 对话上下文
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int dialogue_context_save(const DialogueContext* context, const char* filepath);

/**
 * @brief 从文件加载对话上下文
 *
 * 从文件恢复之前保存的对话上下文。
 *
 * @param filepath 文件路径
 * @return DialogueContext* 对话上下文，调用者负责释放，失败返回NULL
 */
DialogueContext* dialogue_context_load(const char* filepath);

/* ============================================================================
 * A02.4.1 对话状态追踪（DST）深度实现
 *
 * 基于CfC连续时间状态演化的对话状态追踪。
 * 维护槽位-值对分布、信念状态、多轮累积更新。
 * ============================================================================ */

#define DST_MAX_SLOTS 64
#define DST_MAX_VALUES_PER_SLOT 32
#define DST_VALUE_LEN 64
#define DST_SLOT_NAME_LEN 48
#define DST_MAX_BELIEF_HISTORY 50

/**
 * @brief 对话槽位类型
 */
typedef enum {
    DST_SLOT_CATEGORICAL = 0,   /**< 类别型（如做饭、唱歌） */
    DST_SLOT_NUMERICAL = 1,     /**< 数值型（如人数、温度） */
    DST_SLOT_BINARY = 2,        /**< 二值型（是/否） */
    DST_SLOT_LIST = 3           /**< 列表型（多个值） */
} DSTSlotType;

/**
 * @brief 对话槽位定义
 */
typedef struct {
    char name[DST_SLOT_NAME_LEN];                    /**< 槽位名称 */
    DSTSlotType slot_type;                           /**< 槽位类型 */
    char possible_values[DST_MAX_VALUES_PER_SLOT][DST_VALUE_LEN]; /**< 候选取值 */
    int num_possible_values;                         /**< 候选取值数量 */
    float value_weights[DST_MAX_VALUES_PER_SLOT];    /**< 值权重分布 */
    float confidence;                                /**< 槽位填充置信度 (0-1) */
    int is_filled;                                   /**< 是否已填充 */
    int turn_filled;                                 /**< 填充时的对话轮次 */
    float decay_factor;                              /**< 随时间衰减因子 (0-1) */
} DSTSlot;

/**
 * @brief 对话信念状态
 */
struct DialogueBeliefState {
    DSTSlot slots[DST_MAX_SLOTS];                    /**< 槽位数组 */
    int num_slots;                                   /**< 槽位数 */
    int num_filled_slots;                            /**< 已填充槽位数 */
    float belief_entropy;                            /**< 信念熵（不确定性度量） */
    float state_coherence;                           /**< 状态一致性 (0-1) */
    float* belief_state_vector;                      /**< 连续信念状态向量 */
    size_t belief_state_size;                        /**< 信念状态维度 */
    float* belief_state_history[DST_MAX_BELIEF_HISTORY]; /**< 信念状态历史 */
    int history_count;                               /**< 历史帧数 */
    float belief_change_rate;                        /**< 信念变化速率 */
};

/**
 * @brief 创建对话信念状态
 *
 * @param num_slots 槽位数量（不超过 DST_MAX_SLOTS）
 * @param belief_hidden_size 信念状态维度
 * @return DialogueBeliefState* 信念状态句柄，失败返回NULL
 */
DialogueBeliefState* dialogue_belief_state_create(int num_slots, size_t belief_hidden_size);

/**
 * @brief 释放对话信念状态
 */
void dialogue_belief_state_free(DialogueBeliefState* belief);

/**
 * @brief 定义槽位
 *
 * @param belief 信念状态句柄
 * @param slot_index 槽位索引 (0 ~ num_slots-1)
 * @param name 槽位名称
 * @param slot_type 槽位类型
 * @param possible_values 候选取值数组，可为NULL
 * @param num_values 候选取值数量
 * @return int 成功返回0，失败返回-1
 */
int dialogue_belief_define_slot(DialogueBeliefState* belief, int slot_index,
                                 const char* name, DSTSlotType slot_type,
                                 const char* const* possible_values, int num_values);

/**
 * @brief 更新槽位分布（基于当前用户输入）
 *
 * 根据用户输入更新指定槽位的值分布。
 * 内部通过语义匹配计算各候选取值置信度。
 *
 * @param belief 信念状态句柄
 * @param slot_index 槽位索引
 * @param values 更新后的值数组
 * @param weights 对应的置信度权重数组
 * @param num_values 值数量
 * @param turn_number 当前对话轮次
 * @return int 成功返回0，失败返回-1
 */
int dialogue_belief_update_slot(DialogueBeliefState* belief, int slot_index,
                                 const char* const* values, const float* weights,
                                 int num_values, int turn_number);

/**
 * @brief 基于文本分析更新所有槽位
 *
 * 自动从用户输入文本中提取槽位信息并更新信念状态。
 * 使用基于CfC的语义匹配进行槽位填充。
 *
 * @param belief 信念状态句柄
 * @param user_input 用户输入文本
 * @param turn_number 当前对话轮次
 * @return int 成功返回0，失败返回-1
 */
int dialogue_belief_update_from_text(DialogueBeliefState* belief,
                                     const char* user_input, int turn_number);

/**
 * @brief 获取槽位的最优填充值
 *
 * @param belief 信念状态句柄
 * @param slot_index 槽位索引
 * @param value [out] 最优值缓冲区
 * @param value_max 缓冲区最大长度
 * @param confidence [out] 对应置信度
 * @return int 成功返回0，未填充返回-1
 */
int dialogue_belief_get_best_value(const DialogueBeliefState* belief,
                                   int slot_index, char* value, size_t value_max,
                                   float* confidence);

/**
 * @brief 计算信念状态熵（不确定性度量）
 *
 * @param belief 信念状态句柄
 * @return float 信念熵值
 */
float dialogue_belief_compute_entropy(const DialogueBeliefState* belief);

/**
 * @brief 应用信念状态的时间衰减
 *
 * 根据对话轮次推进，对已填充槽位施加时间衰减，
 * 模拟对话状态随时间的遗忘效应。
 *
 * @param belief 信念状态句柄
 * @param elapsed_turns 经过的对话轮次
 */
void dialogue_belief_decay(DialogueBeliefState* belief, int elapsed_turns);

/**
 * @brief 导出信念状态为JSON
 *
 * @param belief 信念状态句柄
 * @param json [out] JSON字符串缓冲区
 * @param max_len 缓冲区最大长度
 * @return int 成功返回写入字符数，失败返回-1
 */
int dialogue_belief_export_json(const DialogueBeliefState* belief,
                                char* json, size_t max_len);

/* ============================================================================
 * A02.4.1 对话策略学习
 *
 * 基于CfC动态系统的对话策略学习与决策。
 * 策略网络将当前信念状态映射到下一个系统动作。
 * ============================================================================ */

#define DPL_MAX_ACTIONS 32
#define DPL_ACTION_PARAM_LEN 64
#define DPL_MAX_POLICY_HISTORY 100

/**
 * @brief 对话策略动作类型
 */
typedef enum {
    DPL_ACTION_REQUEST = 0,      /**< 询问缺失槽位 */
    DPL_ACTION_CONFIRM = 1,      /**< 确认信息 */
    DPL_ACTION_INFORM = 2,       /**< 提供信息 */
    DPL_ACTION_RECOMMEND = 3,    /**< 推荐 */
    DPL_ACTION_CLARIFY = 4,      /**< 澄清需求 */
    DPL_ACTION_PROCEED = 5,      /**< 继续执行 */
    DPL_ACTION_REPEAT = 6,       /**< 重复说明 */
    DPL_ACTION_CONCLUDE = 7,     /**< 总结确认 */
    DPL_ACTION_TERMINATE = 8,    /**< 终止当前话题 */
    DPL_ACTION_EXPLORE = 9,      /**< 探索性提问 */
    DPL_ACTION_ELABORATE = 10    /**< 详细说明 */
} DPLActionType;

/**
 * @brief 对话策略动作
 */
typedef struct {
    DPLActionType action_type;   /**< 动作类型 */
    char params[DPL_ACTION_PARAM_LEN]; /**< 动作参数 */
    float confidence;            /**< 动作置信度 (0-1) */
    int target_slot;             /**< 关联槽位索引（-1表示无） */
} DPLAction;

/**
 * @brief 对话策略状态
 */
struct DialoguePolicy {
    DPLAction action_space[DPL_MAX_ACTIONS]; /**< 动作空间 */
    int action_count;                        /**< 动作数量 */
    float* policy_weights;                   /**< 策略权重向量 */
    size_t policy_size;                      /**< 策略权重维度 */
    float* policy_hidden_state;              /**< 策略隐藏状态 */
    size_t policy_hidden_size;               /**< 策略隐藏维度 */
    float exploration_rate;                  /**< 探索率 epsilon (0-1) */
    float discount_factor;                   /**< 折扣因子 gamma */
    float reward_history[DPL_MAX_POLICY_HISTORY]; /**< 奖励历史 */
    int reward_count;                        /**< 奖励记录数 */
    DPLAction last_action;                   /**< 上一步动作 */
    float last_action_value;                 /**< 上一步动作价值 */
    int is_learning;                         /**< 是否启用学习 */
};

/**
 * @brief 创建对话策略
 *
 * @param action_count 动作空间大小
 * @param policy_hidden_size 策略隐藏维度
 * @return DialoguePolicy* 策略句柄，失败返回NULL
 */
DialoguePolicy* dialogue_policy_create(int action_count, size_t policy_hidden_size);

/**
 * @brief 释放对话策略
 */
void dialogue_policy_free(DialoguePolicy* policy);

/**
 * @brief 定义策略动作
 *
 * @param policy 策略句柄
 * @param action_index 动作索引
 * @param action_type 动作类型
 * @param params 动作参数字符串
 * @return int 成功返回0，失败返回-1
 */
int dialogue_policy_define_action(DialoguePolicy* policy, int action_index,
                                   DPLActionType action_type, const char* params);

/**
 * @brief 基于当前信念状态选择动作
 *
 * 使用CfC动态系统对信念状态进行演化，然后通过策略网络
 * 计算各动作的Q值，采用epsilon-greedy策略选择动作。
 *
 * @param policy 策略句柄
 * @param belief 当前信念状态
 * @param action [out] 选中的动作
 * @return int 成功返回0，失败返回-1
 */
int dialogue_policy_decide(DialoguePolicy* policy,
                           const DialogueBeliefState* belief, DPLAction* action);

/**
 * @brief 策略学习更新
 *
 * 基于时序差分（TD）学习更新策略权重。
 * 使用当前奖励和下一状态价值更新Q值估计。
 *
 * @param policy 策略句柄
 * @param reward 当前奖励值
 * @param next_belief 下一时刻信念状态
 * @param next_action 下一时刻动作
 * @return int 成功返回0，失败返回-1
 */
int dialogue_policy_update(DialoguePolicy* policy, float reward,
                            const DialogueBeliefState* next_belief,
                            const DPLAction* next_action);

/**
 * @brief 设置策略探索率
 *
 * @param policy 策略句柄
 * @param rate 探索率 (0-1)
 */
void dialogue_policy_set_exploration(DialoguePolicy* policy, float rate);

/**
 * @brief 重置策略状态（清空历史）
 *
 * @param policy 策略句柄
 */
void dialogue_policy_reset(DialoguePolicy* policy);

/* ============================================================================
 * A02.4.1 多轮对话推理
 *
 * 基于连续时间状态演化的多轮对话推理。
 * 将多轮对话历史编码为状态轨迹，实现深度上下文理解。
 * ============================================================================ */

#define MTR_MAX_TRACKING 30

/**
 * @brief 多轮推理状态
 */
struct MultiTurnReasoner {
    float* trace_state;                          /**< 多轮状态轨迹 */
    size_t state_size;                           /**< 状态维度 */
    float context_weights[MTR_MAX_TRACKING];     /**< 各轮次上下文权重 */
    float* turn_embeddings[MTR_MAX_TRACKING];    /**< 各轮次嵌入向量 */
    size_t embedding_size;                       /**< 嵌入维度 */
    int num_tracks;                              /**< 跟踪轮次数 */
    int current_turn;                            /**< 当前轮次 */
    float turn_continuity;                       /**< 对话连续性度量 (0-1) */
    float topic_drift;                           /**< 话题漂移度量 (0-1) */
};

/**
 * @brief 创建多轮推理器
 *
 * @param state_size 状态维度
 * @param embedding_size 嵌入维度
 * @return MultiTurnReasoner* 推理器句柄，失败返回NULL
 */
MultiTurnReasoner* multi_turn_reasoner_create(size_t state_size,
                                               size_t embedding_size);

/**
 * @brief 释放多轮推理器
 */
void multi_turn_reasoner_free(MultiTurnReasoner* reasoner);

/**
 * @brief 添加对话轮次到推理状态
 *
 * 将当前轮次的状态和文本嵌入添加到多轮轨迹中。
 *
 * @param reasoner 推理器句柄
 * @param state 当前隐藏状态
 * @param text_embedding 文本嵌入向量
 * @return int 成功返回0，失败返回-1
 */
int multi_turn_reasoner_add_turn(MultiTurnReasoner* reasoner,
                                  const float* state,
                                  const float* text_embedding);

/**
 * @brief 计算推理状态（综合所有历史轮次）
 *
 * 基于加权历史计算综合推理表示。
 * 权重由注意力机制计算，近期和重要轮次获得更高权重。
 *
 * @param reasoner 推理器句柄
 * @param reasoning_output [out] 推理输出向量
 * @param output_size 输出向量大小
 * @return int 成功返回推理向量维度，失败返回-1
 */
int multi_turn_reasoner_reason(MultiTurnReasoner* reasoner,
                                float* reasoning_output, size_t output_size);

/**
 * @brief 检测话题切换
 *
 * 使用状态距离和文本语义相似度检测话题切换。
 *
 * @param reasoner 推理器句柄
 * @param new_state 新输入状态
 * @return int 检测到切换返回1，未检测到返回0，失败返回-1
 */
int multi_turn_reasoner_detect_topic_shift(MultiTurnReasoner* reasoner,
                                            const float* new_state);

/**
 * @brief 获取当前对话连贯性评分
 *
 * @param reasoner 推理器句柄
 * @return float 连贯性评分 (0-1)
 */
float multi_turn_reasoner_get_coherence(const MultiTurnReasoner* reasoner);

/* ============================================================================
 * A02.4.2 对话生成深度实现
 *
 * 使用单一液态神经网络进行连续时间对话状态演化，
 * 从演化状态直接生成对话响应。
 * ============================================================================ */

#define DG_MAX_VOCAB 8192
#define DG_EMBED_DIM 128
#define DG_MAX_GEN_TOKENS 512

/**
 * @brief 对话生成器配置
 */
typedef struct {
    size_t vocab_size;            /**< 词汇表大小（默认4096） */
    size_t embedding_dim;         /**< 词嵌入维度（默认128） */
    size_t hidden_size;           /**< 隐藏维度（默认256） */
    float time_constant;          /**< 时间常数（默认0.05） */
    float delta_t;                /**< 时间步长（默认0.01） */
    int ode_solver_type;          /**< ODE求解器类型（0=euler, 1=rk4, 2=rk45） */
    float temperature;            /**< 生成温度（默认0.8） */
    int top_k;                    /**< Top-K采样数（默认40） */
    float repetition_penalty;     /**< 重复惩罚（默认1.1） */
    int max_generate_tokens;      /**< 最大生成token数（默认256） */
    int pad_token_id;             /**< 填充token ID（默认0） */
    int bos_token_id;             /**< 起始token ID（默认1） */
    int eos_token_id;             /**< 结束token ID（默认2） */
    int use_gpu;                  /**< 是否使用GPU */
} DialogueGenConfig;

/**
 * @brief 对话生成器句柄
 */
typedef struct DialogueGenerator DialogueGenerator;

/**
 * @brief 创建对话生成器
 *
 * @param config 生成器配置
 * @return DialogueGenerator* 生成器句柄，失败返回NULL
 */
DialogueGenerator* dialogue_gen_create(const DialogueGenConfig* config);

/**
 * @brief 释放对话生成器
 */
void dialogue_gen_free(DialogueGenerator* gen);

/**
 * ZSFWS-M005: 标记对话生成器已训练
 * 权重加载或训练完成后调用，解除未训练保护
 */
void dialogue_gen_mark_trained(DialogueGenerator* gen);

/**
 * ZSFWS-M005: 查询对话生成器训练状态
 * @return 1=已训练, 0=未训练或句柄无效
 */
int dialogue_gen_is_trained(const DialogueGenerator* gen);

/**
 * @brief 基于状态生成对话文本
 *
 * 将上下文特征作为初始条件，生成token序列。
 *
 * @param gen 生成器句柄
 * @param context_embedding 上下文嵌入向量
 * @param context_size 上下文嵌入维度
 * @param output_text [out] 生成文本缓冲区
 * @param max_output 缓冲区最大长度
 * @return int 成功返回生成文本长度，失败返回-1
 */
int dialogue_gen_generate(DialogueGenerator* gen,
                               const float* context_embedding,
                               size_t context_size,
                               char* output_text, size_t max_output);

/**
 * @brief 带条件信号的对话生成
 *
 * 将条件信号注入，实现可控对话生成（风格/情感/主题控制）。
 *
 * @param gen 生成器句柄
 * @param context_embedding 上下文嵌入
 * @param context_size 上下文维度
 * @param condition_signal 条件信号向量
 * @param condition_size 条件信号维度
 * @param condition_strength 条件注入强度 (0-1)
 * @param output_text [out] 生成文本缓冲区
 * @param max_output 缓冲区最大长度
 * @return int 成功返回生成文本长度，失败返回-1
 */
int dialogue_gen_generate_conditional(DialogueGenerator* gen,
                                           const float* context_embedding,
                                           size_t context_size,
                                           const float* condition_signal,
                                           size_t condition_size,
                                           float condition_strength,
                                           char* output_text, size_t max_output);

/**
 * @brief 知识增强的对话生成
 *
 * 将知识库嵌入向量作为偏置信号注入，
 * 实现知识感知的对话生成。
 *
 * @param gen 生成器句柄
 * @param context_embedding 上下文嵌入
 * @param context_size 上下文维度
 * @param knowledge_embedding 知识嵌入向量
 * @param knowledge_size 知识嵌入维度
 * @param knowledge_bias_strength 知识偏置强度 (0-1)
 * @param output_text [out] 生成文本缓冲区
 * @param max_output 缓冲区最大长度
 * @return int 成功返回生成文本长度，失败返回-1
 */
int dialogue_gen_generate_knowledge(DialogueGenerator* gen,
                                         const float* context_embedding,
                                         size_t context_size,
                                         const float* knowledge_embedding,
                                         size_t knowledge_size,
                                         float knowledge_bias_strength,
                                         char* output_text, size_t max_output);

/**
 * @brief 重置生成器内部状态
 *
 * @param gen 生成器句柄
 */
void dialogue_gen_reset_state(DialogueGenerator* gen);

/**
 * @brief 设置生成温度
 *
 * @param gen 生成器句柄
 * @param temperature 温度值 (0.1-2.0)
 */
void dialogue_gen_set_temperature(DialogueGenerator* gen, float temperature);

/**
 * @brief 获取生成器配置
 *
 * @param gen 生成器句柄
 * @param config [out] 配置输出
 * @return int 成功返回0，失败返回-1
 */
int dialogue_gen_get_config(const DialogueGenerator* gen,
                                 DialogueGenConfig* config);

/**
 * @brief 设置知识库嵌入（知识增强模式）
 *
 * 为知识增强对话生成设置知识库嵌入矩阵。
 *
 * @param gen 生成器句柄
 * @param knowledge_embeddings 知识嵌入矩阵 [num_entries x embedding_dim]
 * @param num_entries 知识条目数
 * @param embedding_dim 嵌入维度
 * @return int 成功返回0，失败返回-1
 */
int dialogue_gen_set_knowledge_base(DialogueGenerator* gen,
                                         const float* knowledge_embeddings,
                                         int num_entries, size_t embedding_dim);

/* ============================================================================
 * dg_generate_response: 基于CfC ODE的神经对话生成（核心函数）
 *
 * 使用CfC连续时间ODE进行对话状态编码，结合知识库检索增强上下文，
 * 通过自回归解码器逐token生成回复文本。取代原有的模板匹配+规则填充方式。
 *
 * 上下文编码流程：
 *   用户输入 → 文本向量化 → CfC ODE编码 → 对话状态h
 *   知识检索 → 事实向量嵌入 → 与h融合 → 增强状态h'
 *   增强状态h' → 自回归解码 → 逐词生成回复
 *
 * 回复生成优先级：
 *   1. 知识库检索增强（检索结果→模板填充）
 *   2. CfC ODE神经生成（模型权重已加载时）
 *   3. 模板匹配回退（无模型时）
 * ============================================================================ */

/**
 * @brief CfC ODE神经对话生成主入口
 *
 * 接收用户输入和对话上下文，通过CfC连续时间动态系统生成回复。
 * 自动从知识库中检索与用户输入最相关的5个事实，注入到编码状态中。
 *
 * @param gen 对话生成器句柄（必须已初始化CfC单元）
 * @param user_input 用户输入文本
 * @param input_length 输入文本长度
 * @param dialogue_history 对话历史文本（可为NULL表示新对话）
 * @param history_length 对话历史长度
 * @param output_text [out] 生成的回复文本缓冲区
 * @param max_output 输出缓冲区最大字节数
 * @param temperature 生成温度（0.1-2.0）
 * @param top_k Top-K采样参数
 * @param confidence [out] 生成置信度输出
 * @return int 成功返回生成文本长度，失败返回-1
 */
int dg_generate_response(DialogueGenerator* gen,
                         const char* user_input,
                         size_t input_length,
                         const char* dialogue_history,
                         size_t history_length,
                         char* output_text,
                         size_t max_output,
                         float temperature,
                         int top_k,
                         float* confidence);

/**
 * @brief 保存CfC对话模型权重到二进制文件
 *
 * 保存生成器的全部权重：词嵌入表、输出投影权重W和b、
 * 知识库嵌入（如已初始化）。使用魔数+版本+校验和格式。
 *
 * @param gen 对话生成器句柄
 * @param filepath 保存文件路径
 * @return int 成功返回0，失败返回-1
 */
int dg_save_dialogue_model(DialogueGenerator* gen, const char* filepath);

/**
 * @brief 从二进制文件加载CfC对话模型权重
 *
 * 加载由dg_save_dialogue_model保存的模型权重。
 * 验证魔数、版本匹配和校验和，确保文件完整性。
 *
 * @param gen 对话生成器句柄（必须已初始化词汇表）
 * @param filepath 模型文件路径
 * @return int 成功返回0，失败返回-1
 */
int dg_load_dialogue_model(DialogueGenerator* gen, const char* filepath);

/**
 * @brief 检查对话生成器是否已初始化
 *
 * @param gen 对话生成器句柄
 * @return int 已初始化返回1，否则返回0
 */
int dialogue_gen_is_initialized(const DialogueGenerator* gen);

/**
 * @brief 使用CfC ODE对话生成器生成回复（集成入口）
 *
 * 结合DialogueProcessor的上下文，使用CfC ODE路径生成对话回复。
 * 自动处理对话历史拼接和知识库检索增强。
 *
 * @param processor 对话处理器句柄
 * @param user_input 用户输入文本
 * @param input_length 输入文本长度
 * @param context 对话上下文（可为NULL）
 * @param output_text [out] 生成的回复文本缓冲区
 * @param max_output 输出缓冲区最大字节数
 * @param temperature 生成温度（0.1-2.0）
 * @param top_k Top-K采样参数
 * @param confidence [out] 生成置信度输出
 * @return int 成功返回生成文本长度，失败返回-1
 */
int dialogue_generate_with_cfc_ode(DialogueProcessor* processor,
                                   const char* user_input,
                                   size_t input_length,
                                   const DialogueContext* context,
                                   char* output_text,
                                   size_t max_output,
                                   float temperature,
                                   int top_k,
                                   float* confidence);

/**
 * @brief 将对话深度增强系统集成到主对话处理器
 *
 * 将DST信念状态、策略学习、多轮推理器、生成器
 * 注册到主对话处理器中，增强 dialogue_process_input 的能力。
 *
 * @param processor 对话处理器句柄
 * @param belief DST信念状态（可为NULL）
 * @param policy 对话策略（可为NULL）
 * @param reasoner 多轮推理器（可为NULL）
 * @param gen 对话生成器（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int dialogue_register_deep_modules(DialogueProcessor* processor,
                                    DialogueBeliefState* belief,
                                    DialoguePolicy* policy,
                                    MultiTurnReasoner* reasoner,
                                    DialogueGenerator* gen);

/**
 * @brief 流式生成token回调函数类型
 *
 * 每次生成一个新token时调用此回调，用于实时推送生成过程。
 *
 * @param token_text 当前生成的UTF-8文本片段
 * @param token_id 当前生成的token ID
 * @param progress 生成进度 (0.0-1.0)
 * @param is_final 是否为最后一个token
 * @param user_data 用户自定义数据指针
 */
typedef void (*DialogueStreamCallback)(const char* token_text, int token_id,
                                       float progress, int is_final,
                                       void* user_data);

/**
 * @brief 使用LNN自回归流式生成文本（逐token回调）
 *
 * 与dialogue_generate_text相同，但每次生成一个token时调用回调函数。
 * 支持实时流式推送至前端。
 *
 * @param processor 对话处理器句柄
 * @param context_features 上下文特征向量
 * @param context_size 上下文特征大小
 * @param output 输出文本缓冲区
 * @param max_output 输出缓冲区最大长度
 * @param temperature 采样温度（0.1-2.0，默认1.0）
 * @param top_k top-k采样参数（默认40，0表示禁用）
 * @param stream_callback 逐token回调函数（可为NULL）
 * @param stream_user_data 回调用户数据指针
 * @return int 成功返回生成文本长度，失败返回-1
 */
int dialogue_generate_text_streaming(DialogueProcessor* processor,
                                    const float* context_features,
                                    size_t context_size,
                                    char* output,
                                    size_t max_output,
                                    float temperature,
                                    int top_k,
                                    DialogueStreamCallback stream_callback,
                                    void* stream_user_data);

/**
 * @brief 流式多模态对话处理（逐token回调）
 *
 * 与dialogue_process_multimodal相同，但支持逐token流式回调。
 *
 * @param processor 对话处理器句柄
 * @param text_input 文本输入（可为NULL）
 * @param text_length 文本长度
 * @param image_features 图像特征向量（可为NULL）
 * @param image_feature_count 图像特征数量
 * @param audio_features 音频特征向量（可为NULL）
 * @param audio_feature_count 音频特征数量
 * @param spatial_data 空间感知数据（可为NULL）
 * @param spatial_data_count 空间数据数量
 * @param context 对话上下文（可为NULL）
 * @param stream_callback 逐token回调函数（可为NULL）
 * @param stream_user_data 回调用户数据指针
 * @return DialogueResponse* 对话响应，调用者负责释放，失败返回NULL
 */
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
                                              void* stream_user_data);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_DIALOGUE_H