/**
 * @file audio_semantic.h
 * @brief 音频特征到语义理解的映射接口
 * 
 * 将音频特征（MFCC、频谱特征等）映射到语义理解结果：
 * - 语音识别结果（文本）
 * - 情感分析（情感类别和强度）
 * - 意图识别（用户意图分类）
 * - 关键词提取（关键概念识别）
 * - 语义槽填充（结构化信息提取）
 * 
 * 基于现有语义网络和语义记忆系统实现深度语义理解。
 */

#ifndef SELFLNN_AUDIO_SEMANTIC_H
#define SELFLNN_AUDIO_SEMANTIC_H

#include "selflnn/multimodal/audio.h"
#include "selflnn/knowledge/semantic_network.h"
#include "selflnn/memory/semantic.h"
#include <stddef.h>

/**
 * @brief 最大支持说话人数
 */
#define AUDIO_SEMANTIC_MAX_SPEAKERS 32

/**
 * @brief 说话人嵌入向量维度
 */
#define AUDIO_SEMANTIC_SPEAKER_EMBEDDING_SIZE 64

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 情感类别枚举
 */
typedef enum {
    EMOTION_NEUTRAL = 0,      /**< 中性 */
    EMOTION_HAPPY = 1,        /**< 快乐 */
    EMOTION_SAD = 2,          /**< 悲伤 */
    EMOTION_ANGRY = 3,        /**< 愤怒 */
    EMOTION_FEAR = 4,         /**< 恐惧 */
    EMOTION_SURPRISE = 5,     /**< 惊讶 */
    EMOTION_DISGUST = 6,      /**< 厌恶 */
    EMOTION_EXCITED = 7,      /**< 兴奋 */
    EMOTION_CALM = 8,         /**< 平静 */
    EMOTION_FRUSTRATED = 9,   /**< 沮丧 */
    EMOTION_CURIOUS = 10      /**< 好奇 */
} EmotionCategory;

/**
 * @brief 意图类别枚举
 */
typedef enum {
    INTENT_UNKNOWN = 0,       /**< 未知意图 */
    INTENT_QUESTION = 1,      /**< 提问 */
    INTENT_COMMAND = 2,       /**< 命令 */
    INTENT_STATEMENT = 3,     /**< 陈述 */
    INTENT_GREETING = 4,      /**< 问候 */
    INTENT_FAREWELL = 5,      /**< 告别 */
    INTENT_REQUEST = 6,       /**< 请求 */
    INTENT_COMPLAINT = 7,     /**< 投诉 */
    INTENT_COMPLIMENT = 8,    /**< 赞美 */
    INTENT_CONFIRMATION = 9,  /**< 确认 */
    INTENT_DENIAL = 10,       /**< 否认 */
    INTENT_HELP = 11,         /**< 求助 */
    INTENT_INFORMATION = 12,  /**< 信息查询 */
    INTENT_THANKS = 13,        /**< 感谢 */
    INTENT_APOLOGY = 14,       /**< 道歉 */
    INTENT_EXPRESSION = 15,    /**< 表达 */
    INTENT_DESCRIPTION = 16,   /**< 描述 */
    INTENT_NARRATION = 17,     /**< 叙述 */
    INTENT_EXPLANATION = 18,   /**< 解释 */
    INTENT_ARGUMENT = 19,      /**< 论证 */
    INTENT_OTHER = 20          /**< 其他 */
} IntentCategory;

/**
 * @brief 语义槽类型枚举
 */
typedef enum {
    SLOT_TYPE_TIME = 0,       /**< 时间 */
    SLOT_TYPE_DATE = 1,       /**< 日期 */
    SLOT_TYPE_LOCATION = 2,   /**< 地点 */
    SLOT_TYPE_PERSON = 3,     /**< 人名 */
    SLOT_TYPE_ORGANIZATION = 4, /**< 组织机构 */
    SLOT_TYPE_NUMBER = 5,     /**< 数字 */
    SLOT_TYPE_CURRENCY = 6,   /**< 货币 */
    SLOT_TYPE_PERCENTAGE = 7, /**< 百分比 */
    SLOT_TYPE_MEASUREMENT = 8, /**< 度量衡 */
    SLOT_TYPE_ACTION = 9,     /**< 动作 */
    SLOT_TYPE_UNKNOWN = 10    /**< 未知 */
} SlotType;

/**
 * @brief 音频语义模型类型枚举
 */
typedef enum {
    AUDIO_SEMANTIC_MODEL_EMOTION = 0,   /**< 情感分析模型 */
    AUDIO_SEMANTIC_MODEL_INTENT = 1,    /**< 意图识别模型 */
    AUDIO_SEMANTIC_MODEL_KEYWORD = 2    /**< 关键词提取模型 */
} AudioSemanticModelType;

/**
 * @brief 说话人识别结果
 */
typedef struct {
    int speaker_id;                  /**< 说话人ID（-1表示未知） */
    float confidence;                /**< 说话人识别置信度 (0-1) */
    char* speaker_name;              /**< 说话人名称（动态分配） */
    float embedding[AUDIO_SEMANTIC_SPEAKER_EMBEDDING_SIZE]; /**< 说话人嵌入向量 */
    float all_speaker_probs[AUDIO_SEMANTIC_MAX_SPEAKERS]; /**< 所有说话人概率 */
    int has_new_speaker;             /**< 是否可能为新说话人 */
} SpeakerRecognitionResult;

/**
 * @brief 语义槽结构体
 */
typedef struct {
    SlotType type;            /**< 槽类型 */
    char* value;              /**< 槽值字符串 */
    float confidence;         /**< 置信度 (0-1) */
    int start_pos;            /**< 在文本中的起始位置 */
    int end_pos;              /**< 在文本中的结束位置 */
} SemanticSlot;

/**
 * @brief 情感分析结果结构体
 */
typedef struct {
    EmotionCategory category; /**< 主要情感类别 */
    float intensity;          /**< 情感强度 (0-1) */
    float probabilities[10];  /**< 各类别概率（与EmotionCategory对应） */
    float valence;            /**< 情感效价（负向-正向）(-1到1) */
    float arousal;            /**< 情感唤醒度（低-高）(0-1) */
    float dominance;          /**< 情感支配度（弱-强）(0-1) */
} EmotionAnalysis;

/**
 * @brief 意图识别结果结构体
 */
typedef struct {
    IntentCategory category;  /**< 主要意图类别 */
    float confidence;         /**< 置信度 (0-1) */
    char* intent_name;        /**< 意图名称（字符串） */
    char* intent_description; /**< 意图描述 */
    float probabilities[13];  /**< 各类别概率（与IntentCategory对应） */
} IntentAnalysis;

/**
 * @brief 关键词结构体
 */
typedef struct {
    char* keyword;            /**< 关键词字符串 */
    float relevance;          /**< 相关性得分 (0-1) */
    float salience;           /**< 显著性得分 (0-1) */
    int frequency;            /**< 频率（在当前话语中） */
    Concept* concept;         /**< 关联的语义网络概念（可选） */
} Keyword;

/**
 * @brief 音频语义理解结果结构体
 */
typedef struct {
    /* 语音识别结果 */
    char* recognized_text;    /**< 识别的文本（如果进行语音识别） */
    float recognition_confidence; /**< 识别置信度 */
    
    /* 情感分析 */
    EmotionAnalysis emotion;  /**< 情感分析结果 */
    
    /* 意图识别 */
    IntentAnalysis intent;    /**< 意图识别结果 */
    
    /* 说话人识别 */
    SpeakerRecognitionResult speaker; /**< 说话人识别结果 */
    
    /* 关键词提取 */
    Keyword* keywords;        /**< 关键词数组 */
    int num_keywords;         /**< 关键词数量 */
    int max_keywords;         /**< 最大关键词数 */
    
    /* 语义槽填充 */
    SemanticSlot* slots;      /**< 语义槽数组 */
    int num_slots;            /**< 语义槽数量 */
    int max_slots;            /**< 最大语义槽数 */
    
    /* 语义网络关联 */
    Concept** related_concepts; /**< 相关概念数组 */
    int num_related_concepts; /**< 相关概念数量 */
    
    /* 轨迹和地图数据（用于空间语义理解） */
    void* trajectory;          /**< 运动轨迹数据（可选） */
    void* local_map_points;    /**< 局部地图点云数据（可选） */
    
    /* 元数据 */
    float processing_time_ms; /**< 处理时间（毫秒） */
    float audio_duration_ms;  /**< 音频时长（毫秒） */
    int sample_rate;          /**< 采样率 */
    int num_features;         /**< 使用的特征数量 */
    
    /* MFCC特征数据 */
    float* mfcc_features;     /**< MFCC特征向量 */
    int feature_dim;          /**< 特征维度 */
    
    /* 置信度分数 */
    float overall_confidence; /**< 整体置信度 */
    float semantic_coherence; /**< 语义连贯性得分 */
    float contextual_fit;     /**< 上下文适应性得分 */
} AudioSemanticResult;

/**
 * @brief 音频语义理解配置结构体
 */
typedef struct {
    /* 特征提取配置 */
    int use_mfcc;             /**< 是否使用MFCC特征 */
    int use_spectral;         /**< 是否使用频谱特征 */
    int use_prosodic;         /**< 是否使用韵律特征（音高、能量等） */
    int mfcc_coefficients;    /**< MFCC系数数量 */
    int spectral_bands;       /**< 频谱频带数量 */
    
    /* 模型配置 */
    int enable_emotion_analysis;   /**< 是否启用情感分析 */
    int enable_speaker_recognition; /**< 是否启用说话人识别 */
    int enable_intent_recognition; /**< 是否启用意图识别 */
    int enable_keyword_extraction; /**< 是否启用关键词提取 */
    int enable_slot_filling;       /**< 是否启用语义槽填充 */
    int enable_semantic_linking;   /**< 是否启用语义网络链接 */
    
    /* 语义网络配置 */
    int max_related_concepts; /**< 最大相关概念数 */
    float min_concept_similarity; /**< 最小概念相似度阈值 */
    
    /* 性能配置 */
    int use_gpu;              /**< 是否使用GPU加速 */
    int max_parallel_requests; /**< 最大并行请求数 */
    int cache_enabled;        /**< 是否启用缓存 */
    
    /* 阈值配置 */
    float confidence_threshold; /**< 置信度阈值 */
    float emotion_intensity_threshold; /**< 情感强度阈值 */
    float keyword_relevance_threshold; /**< 关键词相关性阈值 */
} AudioSemanticConfig;

/**
 * @brief 音频语义理解器句柄
 */
typedef struct AudioSemanticProcessor AudioSemanticProcessor;

/**
 * @brief 获取默认的音频语义理解配置
 * 
 * @return AudioSemanticConfig 默认配置
 */
AudioSemanticConfig audio_semantic_get_default_config(void);

/**
 * @brief 创建音频语义理解器
 * 
 * @param config 配置参数
 * @param audio_processor 音频处理器句柄（可选，如果为NULL则内部创建）
 * @param semantic_network 语义网络句柄（可选，如果为NULL则内部创建）
 * @param semantic_memory 语义记忆句柄（可选，如果为NULL则内部创建）
 * @return AudioSemanticProcessor* 处理器句柄，失败返回NULL
 */
AudioSemanticProcessor* audio_semantic_processor_create(
    const AudioSemanticConfig* config,
    AudioProcessor* audio_processor,
    SemanticNetwork* semantic_network,
    SemanticMemory* semantic_memory);

/**
 * @brief 释放音频语义理解器
 * 
 * @param processor 处理器句柄
 */
void audio_semantic_processor_free(AudioSemanticProcessor* processor);

/**
 * @brief 处理音频数据并生成语义理解结果
 * 
 * @param processor 处理器句柄
 * @param audio_data 音频数据（浮点数组）
 * @param num_samples 样本数
 * @param sample_rate 采样率
 * @param num_channels 通道数
 * @param result 语义理解结果输出
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_process(AudioSemanticProcessor* processor,
                          const float* audio_data,
                          int num_samples,
                          int sample_rate,
                          int num_channels,
                          AudioSemanticResult* result);

/**
 * @brief 处理音频特征（跳过特征提取步骤）
 * 
 * @param processor 处理器句柄
 * @param features 音频特征数组
 * @param num_features 特征数量
 * @param feature_dimension 特征维度
 * @param result 语义理解结果输出
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_process_features(AudioSemanticProcessor* processor,
                                   const float* features,
                                   int num_features,
                                   int feature_dimension,
                                   AudioSemanticResult* result);

/**
 * @brief 处理文本输入（用于纯语义分析，不包含音频）
 * 
 * @param processor 处理器句柄
 * @param text 输入文本
 * @param result 语义理解结果输出（不包含音频相关字段）
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_process_text(AudioSemanticProcessor* processor,
                               const char* text,
                               AudioSemanticResult* result);

/**
 * @brief 批量处理音频数据
 * 
 * @param processor 处理器句柄
 * @param audio_data_list 音频数据指针数组
 * @param num_samples_list 样本数数组
 * @param sample_rate 采样率（所有音频相同）
 * @param num_channels 通道数（所有音频相同）
 * @param num_audio 音频数量
 * @param results 语义理解结果数组输出
 * @return int 成功返回处理的音频数量，失败返回-1
 */
int audio_semantic_process_batch(AudioSemanticProcessor* processor,
                                const float** audio_data_list,
                                const int* num_samples_list,
                                int sample_rate,
                                int num_channels,
                                int num_audio,
                                AudioSemanticResult* results);

/**
 * @brief 更新音频语义理解器配置
 * 
 * @param processor 处理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_processor_set_config(AudioSemanticProcessor* processor,
                                       const AudioSemanticConfig* config);

/**
 * @brief 获取音频语义理解器配置
 * 
 * @param processor 处理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_processor_get_config(const AudioSemanticProcessor* processor,
                                       AudioSemanticConfig* config);

/**
 * @brief 获取音频语义理解器统计信息
 * 
 * @param processor 处理器句柄
 * @param total_processed 总处理音频数输出
 * @param avg_processing_time 平均处理时间输出（毫秒）
 * @param accuracy_emotion 情感分析准确率输出
 * @param accuracy_intent 意图识别准确率输出
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_processor_get_stats(const AudioSemanticProcessor* processor,
                                      int* total_processed,
                                      float* avg_processing_time,
                                      float* accuracy_emotion,
                                      float* accuracy_intent);

/**
 * @brief 重置音频语义理解器状态
 * 
 * @param processor 处理器句柄
 */
void audio_semantic_processor_reset(AudioSemanticProcessor* processor);

/**
 * @brief 加载预训练模型
 * 
 * @param processor 处理器句柄
 * @param model_path 模型文件路径
 * @param model_type 模型类型：0=情感模型，1=意图模型，2=关键词模型
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_load_model(AudioSemanticProcessor* processor,
                             const char* model_path,
                             AudioSemanticModelType model_type);

/**
 * @brief 保存当前模型状态
 * 
 * @param processor 处理器句柄
 * @param model_path 模型文件路径
 * @param model_type 模型类型：0=情感模型，1=意图模型，2=关键词模型
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_save_model(AudioSemanticProcessor* processor,
                             const char* model_path,
                             AudioSemanticModelType model_type);

/**
 * @brief 训练音频语义理解模型
 * 
 * @param processor 处理器句柄
 * @param training_data 训练数据（音频特征或原始音频）
 * @param labels 标签数据
 * @param num_samples 样本数量
 * @param model_type 模型类型：0=情感模型，1=意图模型，2=关键词模型
 * @param epochs 训练轮数
 * @param learning_rate 学习率
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_train_model(AudioSemanticProcessor* processor,
                              const float* training_data,
                              const int* labels,
                              int num_samples,
                              int model_type,
                              int epochs,
                              float learning_rate);

/**
 * @brief 释放音频语义理解结果内存
 * 
 * @param result 语义理解结果
 */
void audio_semantic_result_free(AudioSemanticResult* result);

/**
 * @brief 初始化音频语义理解结果（分配内存）
 * 
 * @param result 语义理解结果指针
 * @param max_keywords 最大关键词数
 * @param max_slots 最大语义槽数
 * @param max_concepts 最大相关概念数
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_result_init(AudioSemanticResult* result,
                              int max_keywords,
                              int max_slots,
                              int max_concepts);

/**
 * @brief 复制音频语义理解结果
 * 
 * @param src 源结果
 * @param dst 目标结果
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_result_copy(const AudioSemanticResult* src,
                              AudioSemanticResult* dst);

/**
 * @brief 序列化音频语义理解结果为JSON字符串
 * 
 * @param result 语义理解结果
 * @return char* JSON字符串，调用者负责释放，失败返回NULL
 */
char* audio_semantic_result_to_json(const AudioSemanticResult* result);

/**
 * @brief 从JSON字符串解析音频语义理解结果
 * 
 * @param json_str JSON字符串
 * @param result 语义理解结果输出
 * @return int 成功返回0，失败返回-1
 */
int audio_semantic_result_from_json(const char* json_str,
                                   AudioSemanticResult* result);

/**
 * @brief 打印音频语义理解结果（调试用）
 * 
 * @param result 语义理解结果
 */
void audio_semantic_result_print(const AudioSemanticResult* result);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_AUDIO_SEMANTIC_H