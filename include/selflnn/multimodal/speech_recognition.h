/**
 * @file speech_recognition.h
 * @brief 语音识别 — 液态状态时序动态系统
 *
 * 纯液态神经网络语音识别模块。
 * 输入音频 → 梅尔滤波器组特征 → 液态状态演化 → 直接输出文本token。
 * 无外部声学模型、无外部语言模型、无独立解码器。
 * 所有语音到文本的映射由单个连续时间液态状态动态系统完成。
 */

#ifndef SELFLNN_SPEECH_RECOGNITION_H
#define SELFLNN_SPEECH_RECOGNITION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 最大词汇表大小 */
#define SR_MAX_VOCAB_SIZE 65536

/** @brief 最大文本长度 */
#define SR_MAX_TEXT_LEN 4096

/** @brief 默认采样率 */
#define SR_DEFAULT_SAMPLE_RATE 16000

/** @brief 默认帧长（样本数） */
#define SR_DEFAULT_FRAME_LENGTH 400

/** @brief 默认帧移（样本数） */
#define SR_DEFAULT_FRAME_SHIFT 160

/** @brief 默认梅尔滤波器组数量 */
#define SR_DEFAULT_NUM_MEL_BINS 80

/** @brief 默认隐藏状态大小 */
#define SR_DEFAULT_HIDDEN_SIZE 256

/** @brief 最大token序列长度 */
#define SR_MAX_TOKEN_SEQ 2048

/** @brief 波束搜索宽度 */
#define SR_DEFAULT_BEAM_WIDTH 5

/**
 * @brief 语音识别配置结构体
 * 所有参数均为单一液态状态动态系统的配置，无独立子模型。
 */
typedef struct {
    int sample_rate;                     /**< 音频采样率（Hz），默认16000 */
    int frame_length;                    /**< 帧长度（样本数），默认400（25ms） */
    int frame_shift;                     /**< 帧移（样本数），默认160（10ms） */
    int num_mel_bins;                    /**< 梅尔滤波器组数量，默认80 */
    int feature_dimension;               /**< 特征维度（含一阶/二阶差分），默认240 */
    int vocab_size;                      /**< 词汇表大小（含空白token） */
    int beam_width;                      /**< 波束搜索宽度，默认5 */
    float beam_threshold;                /**< 波束搜索阈值，默认5.0 */
    float decoding_temperature;          /**< 解码温度，默认1.0 */

    /* 液态状态核心参数 */
    size_t hidden_size;                  /**< 隐藏状态大小，默认256 */
    float time_constant;                 /**< 时间常数，默认0.1 */
    int use_bias;                        /**< 是否使用偏置，默认1 */
    float noise_std;                     /**< 训练噪声标准差，默认0.01 */
    int ode_solver_type;                 /**< 求解器类型：0=封闭形式解，1=RK4 */
    int input_gate;                      /**< 是否开启输入门控，默认1 */
    int forget_gate;                     /**< 是否开启遗忘门控，默认1 */

    /* 训练参数 */
    float learning_rate;                 /**< 学习率，默认0.001 */
    int batch_size;                      /**< 批次大小，默认32 */
    int num_epochs;                      /**< 训练轮数，默认100 */
    float gradient_clip_norm;            /**< 梯度裁剪范数，默认5.0 */
    int use_ctc_loss;                    /**< 是否使用CTC损失，默认1 */

    /* 硬件加速 */
    int use_gpu;                         /**< 是否使用GPU加速 */
} SpeechRecognitionConfig;

/**
 * @brief 语音识别结果结构体
 */
typedef struct {
    char* text;                          /**< 识别的文本 */
    float confidence;                    /**< 置信度（0.0~1.0） */
    float processing_time_ms;            /**< 处理时间（毫秒） */
    int num_characters;                  /**< 字符数 */
    float* token_confidences;            /**< 每个token的置信度 */
    int* token_boundaries;               /**< token边界（帧索引） */
    int* token_ids;                      /**< token ID序列 */
    int num_tokens;                      /**< token数量 */
} SpeechRecognitionResult;

/**
 * @brief 语音识别处理器句柄（不透明指针）
 */
typedef struct SpeechRecognizer SpeechRecognizer;

/**
 * @brief 获取默认的语音识别配置
 * @return SpeechRecognitionConfig 默认配置
 */
SpeechRecognitionConfig speech_recognition_get_default_config(void);

/**
 * @brief 创建语音识别处理器
 * @param config 语音识别配置
 * @return SpeechRecognizer* 处理器句柄，失败返回NULL
 */
SpeechRecognizer* speech_recognizer_create(const SpeechRecognitionConfig* config);

/**
 * @brief 释放语音识别处理器
 * @param recognizer 处理器句柄
 */
void speech_recognizer_free(SpeechRecognizer* recognizer);

/**
 * @brief 执行语音识别（音频→文本）
 *
 * 液态状态动态系统处理：
 * 输入音频 → 梅尔滤波器组特征提取 → 液态状态演化 → 字符概率输出 → 波束搜索解码
 * @param recognizer 处理器句柄
 * @param audio_data 音频数据（float PCM，范围[-1.0, 1.0]）
 * @param num_samples 样本数
 * @param result 识别结果输出
 * @return int 成功返回0，失败返回-1
 */
int speech_recognizer_recognize(SpeechRecognizer* recognizer,
                                const float* audio_data, int num_samples,
                                SpeechRecognitionResult* result);

/**
 * @brief 执行流式语音识别
 * @param recognizer 处理器句柄
 * @param audio_chunk 音频数据块
 * @param chunk_samples 块样本数
 * @param is_final 是否是最终块
 * @param result 识别结果输出（增量）
 * @return int 成功返回0，失败返回-1
 */
int speech_recognizer_recognize_stream(SpeechRecognizer* recognizer,
                                       const float* audio_chunk, int chunk_samples,
                                       int is_final,
                                       SpeechRecognitionResult* result);

/**
 * @brief 提取音频特征（梅尔滤波器组 + 差分）
 * @param recognizer 处理器句柄
 * @param audio_data 音频数据
 * @param num_samples 样本数
 * @param features 特征输出缓冲区（[num_frames][feature_dimension]）
 * @param max_features 最大特征数
 * @return int 成功返回帧数，失败返回-1
 */
int speech_recognizer_extract_features(SpeechRecognizer* recognizer,
                                       const float* audio_data, int num_samples,
                                       float* features, int max_features);

/**
 * @brief 重置语音识别处理器（清空内部状态）
 * @param recognizer 处理器句柄
 */
void speech_recognizer_reset(SpeechRecognizer* recognizer);

/**
 * @brief 设置词汇表
 * @param recognizer 处理器句柄
 * @param vocab 词汇表数组（UTF-8字符串）
 * @param vocab_size 词汇表大小
 * @return int 成功返回0，失败返回-1
 */
int speech_recognizer_set_vocabulary(SpeechRecognizer* recognizer,
                                     const char** vocab, int vocab_size);

/**
 * @brief 获取当前词汇表
 * @param recognizer 处理器句柄
 * @param vocab 词汇表输出缓冲区
 * @param max_vocab_size 最大词汇表大小
 * @return int 成功返回词汇表大小，失败返回-1
 */
int speech_recognizer_get_vocabulary(SpeechRecognizer* recognizer,
                                     char** vocab, int max_vocab_size);

/**
 * @brief 计算词错误率（WER）
 * @param reference 参考文本
 * @param hypothesis 识别文本
 * @param wer 词错误率输出
 * @param substitutions 替换错误数输出
 * @param deletions 删除错误数输出
 * @param insertions 插入错误数输出
 * @return int 成功返回0，失败返回-1
 */
int speech_recognizer_compute_wer(const char* reference, const char* hypothesis,
                                  float* wer, int* substitutions,
                                  int* deletions, int* insertions);

/**
 * @brief 计算字符错误率（CER）
 * @param reference 参考文本
 * @param hypothesis 识别文本
 * @param cer 字符错误率输出
 * @param substitutions 替换错误数输出
 * @param deletions 删除错误数输出
 * @param insertions 插入错误数输出
 * @return int 成功返回0，失败返回-1
 */
int speech_recognizer_compute_cer(const char* reference, const char* hypothesis,
                                  float* cer, int* substitutions,
                                  int* deletions, int* insertions);

/**
 * @brief 获取语音识别配置
 * @param recognizer 处理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int speech_recognizer_get_config(const SpeechRecognizer* recognizer,
                                 SpeechRecognitionConfig* config);

/**
 * @brief 设置语音识别配置
 * @param recognizer 处理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int speech_recognizer_set_config(SpeechRecognizer* recognizer,
                                 const SpeechRecognitionConfig* config);

/**
 * @brief 训练语音识别模型（端到端训练）
 * @param recognizer 处理器句柄
 * @param training_audio 训练音频数组
 * @param training_transcripts 训练转录文本数组
 * @param num_samples 样本数量
 * @param num_epochs 训练轮数
 * @param learning_rate 学习率
 * @param batch_size 批次大小
 * @return int 成功返回0，失败返回-1
 */
int speech_recognizer_train(SpeechRecognizer* recognizer,
                            const float** training_audio,
                            const char** training_transcripts,
                            const int* training_audio_lengths,
                            int num_samples, int num_epochs,
                            float learning_rate, int batch_size,
                            float* out_loss);

/**
 * @brief 保存语音识别模型
 * @param recognizer 处理器句柄
 * @param directory 保存目录
 * @return int 成功返回0，失败返回-1
 */
int speech_recognizer_save_model(SpeechRecognizer* recognizer,
                                 const char* directory);

/**
 * @brief 加载语音识别模型
 * @param recognizer 处理器句柄
 * @param directory 加载目录
 * @return int 成功返回0，失败返回-1
 */
int speech_recognizer_load_model(SpeechRecognizer* recognizer,
                                 const char* directory);

/**
 * @brief CTC损失函数 — 连接时序分类前向后向算法
 * Graves et al. 2006
 * @param logits [num_frames][vocab_size] log概率矩阵
 * @param num_frames 时间帧数
 * @param labels 目标标签序列
 * @param num_labels 标签长度
 * @param vocab_size 词汇表大小
 * @return float CTC损失值
 */
float speech_ctc_compute_loss(const float* logits, int num_frames,
                               const int* labels, int num_labels, int vocab_size);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SPEECH_RECOGNITION_H */
