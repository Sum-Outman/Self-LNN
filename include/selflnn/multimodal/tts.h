/**
 * @file tts.h
 * @brief 语音合成（TTS）— 液态状态生成动态系统
 *
 * 纯液态神经网络语音合成模块。
 * 输入文本 → 拼音/字符嵌入 → 液态状态时序演化 → 直接输出波形。
 * 无Tacotron2注意力、无WaveGlow声码器、无独立F0/时长/韵律模型。
 * 所有文本到语音的映射由单个液态状态连续时间生成动态系统完成。
 */

#ifndef SELFLNN_TTS_H
#define SELFLNN_TTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 最大输入文本长度 */
#define TTS_MAX_TEXT_LEN 4096

/** @brief 最大音素数量 */
#define TTS_MAX_PHONES 8192

/** @brief 默认采样率（Hz） */
#define TTS_SAMPLE_RATE_DEFAULT 24000

/** @brief 最大WAV大小 */
#define TTS_MAX_WAV_SIZE (1024 * 1024 * 4)

/** @brief 默认隐藏状态大小 */
#define TTS_DEFAULT_HIDDEN_SIZE 512

/** @brief 默认时间常数 */
#define TTS_DEFAULT_TIME_CONSTANT 0.05f

/** @brief 拼音表大小 */
#define TTS_PINYIN_TABLE_SIZE 7200

/** @brief 最大波形生成长度 */
#define TTS_MAX_WAVEFORM_LENGTH (TTS_SAMPLE_RATE_DEFAULT * 60)

/**
 * @brief 拼音条目结构体（Unicode码点到拼音编码的映射）
 */
typedef struct {
    uint16_t codepoint;
    uint16_t pinyin_code;
} TTS_PinyinEntry;

/**
 * @brief 声母枚举
 */
typedef enum {
    TTS_INITIAL_NONE = 0,
    TTS_INITIAL_B, TTS_INITIAL_P, TTS_INITIAL_M, TTS_INITIAL_F,
    TTS_INITIAL_D, TTS_INITIAL_T, TTS_INITIAL_N, TTS_INITIAL_L,
    TTS_INITIAL_G, TTS_INITIAL_K, TTS_INITIAL_H,
    TTS_INITIAL_J, TTS_INITIAL_Q, TTS_INITIAL_X,
    TTS_INITIAL_ZH, TTS_INITIAL_CH, TTS_INITIAL_SH, TTS_INITIAL_R,
    TTS_INITIAL_Z, TTS_INITIAL_C, TTS_INITIAL_S,
    TTS_INITIAL_Y, TTS_INITIAL_W,
    TTS_INITIAL_COUNT
} TTS_Initial;

/**
 * @brief 韵母枚举
 */
typedef enum {
    TTS_FINAL_NONE = 0,
    TTS_FINAL_A, TTS_FINAL_O, TTS_FINAL_E, TTS_FINAL_I, TTS_FINAL_U, TTS_FINAL_V,
    TTS_FINAL_AI, TTS_FINAL_EI, TTS_FINAL_UI, TTS_FINAL_AO, TTS_FINAL_OU,
    TTS_FINAL_IU, TTS_FINAL_IE, TTS_FINAL_VE, TTS_FINAL_ER,
    TTS_FINAL_AN, TTS_FINAL_EN, TTS_FINAL_IN, TTS_FINAL_UN, TTS_FINAL_VN,
    TTS_FINAL_ANG, TTS_FINAL_ENG, TTS_FINAL_ING, TTS_FINAL_ONG,
    TTS_FINAL_IA, TTS_FINAL_IAO, TTS_FINAL_IAN, TTS_FINAL_IANG,
    TTS_FINAL_IONG, TTS_FINAL_UA, TTS_FINAL_UO, TTS_FINAL_UAI,
    TTS_FINAL_UAN, TTS_FINAL_UANG, TTS_FINAL_UE,
    TTS_FINAL_UENG, TTS_FINAL_VAN, TTS_FINAL_VANG,
    TTS_FINAL_COUNT
} TTS_Final;

/**
 * @brief 拼音结构体
 */
typedef struct {
    TTS_Initial initial;
    TTS_Final final_part;
    int tone;
} TTS_Pinyin;

/**
 * @brief 语音合成配置结构体
 * 所有参数均为单一液态状态生成系统的配置。
 * 无需独立F0模型、时长模型、韵律模型、LPC合成器。
 */
typedef struct {
    int sample_rate;                     /**< 输出采样率（Hz），默认24000 */
    float speed;                         /**< 语速倍率（0.5~2.0），默认1.0 */
    float pitch_shift;                   /**< 音高偏移（半音），默认0.0 */
    float volume;                        /**< 音量（0.0~1.0），默认1.0 */

    /* 液态状态生成核心参数 */
    size_t hidden_size;                  /**< 隐藏状态大小，默认512 */
    float time_constant;                 /**< 时间常数，默认0.05 */
    int ode_solver_type;                 /**< ODE求解器类型：0=封闭形式解，1=RK4 */
    float noise_std;                     /**< 生成噪声标准差，默认0.005 */

    /* 文本嵌入参数 */
    int vocab_size;                      /**< 文本词汇表大小（含特殊token），默认128 */
    int embedding_dim;                   /**< 字符嵌入维度，默认64 */

    /* M-8修复: 质量守卫参数 */
    float quality_threshold;             /**< 质量阈值（0.0~1.0），低于此阈值时合成返回错误码；默认0.0（不限制） */
} TTSConfig;

/**
 * @brief 语音合成引擎句柄（不透明指针）
 */
typedef struct TTSEngine TTSEngine;

/**
 * @brief 合成音频结构体
 */
typedef struct {
    float* samples;                      /**< PCM音频样本（范围[-1.0, 1.0]） */
    int num_samples;                     /**< 样本数 */
    int sample_rate;                     /**< 采样率 */
    float confidence;                    /**< S-007修复: 合成置信度（0.0~1.0），未训练时为0.3 */
} TTSAudio;

/**
 * @brief 创建语音合成引擎
 * @param config 合成配置
 * @return TTSEngine* 引擎句柄，失败返回NULL
 */
TTSEngine* tts_engine_create(const TTSConfig* config);

/**
 * @brief 释放语音合成引擎
 * @param engine 引擎句柄
 */
void tts_engine_free(TTSEngine* engine);

/**
 * @brief 执行语音合成（文本→波形）
 *
 * 液态状态生成系统处理：
 * 输入文本 → 拼音/字符嵌入 → 液态状态时序演化 → 直接波形采样输出
 * @param engine 引擎句柄
 * @param text 输入文本（UTF-8编码）
 * @return TTSAudio* 合成音频，失败返回NULL
 */
TTSAudio* tts_synthesize(TTSEngine* engine, const char* text);

/**
 * @brief 释放合成音频
 * @param audio 音频结构体指针
 */
void tts_audio_free(TTSAudio* audio);

/**
 * @brief 合成音频并直接写入WAV文件
 * @param engine 引擎句柄
 * @param text 输入文本
 * @param wav_path WAV文件路径
 * @return int 成功返回0，失败返回-1
 */
int tts_synthesize_to_wav(TTSEngine* engine, const char* text, const char* wav_path);

/**
 * @brief CfC驱动端到端语音合成（深度声学模型管线）
 *
 * 完整流水线：
 * 文本 → 字符嵌入 → 编码器(3层CfC ODE) → 编码隐藏状态
 * → 解码器(自回归CfC ODE) → 梅尔频谱图
 * → 神经声码器(WaveNet风格扩张卷积) → 音频波形
 *
 * 当声学模型未初始化时自动回退到确定性共振峰合成路径。
 *
 * @param engine TTS引擎
 * @param text 输入文本（UTF-8）
 * @param wav_path 输出WAV文件路径
 * @return int 0成功，-1失败
 */
int tts_synthesize_cfc_end_to_end(TTSEngine* engine, const char* text,
                                   const char* wav_path);

/**
 * @brief 重置合成引擎状态
 * @param engine 引擎句柄
 * @return int 成功返回0，失败返回-1
 */
int tts_engine_reset(TTSEngine* engine);

/**
 * @brief 设置语速
 * @param engine 引擎句柄
 * @param speed 语速倍率（0.5~2.0）
 * @return int 成功返回0，失败返回-1
 */
int tts_engine_set_speed(TTSEngine* engine, float speed);

/**
 * @brief 标记TTS引擎为已训练
 * 在系统加载检查点或完成引导训练后调用，使TTS可以正常合成语音
 * @param engine 引擎句柄
 */
void tts_engine_mark_trained(TTSEngine* engine);

/**
 * @brief 检查TTS引擎完整性
 * 在调用 tts_synthesize 前验证引擎内部状态完整，避免深层崩溃
 * @param engine 引擎句柄
 * @return int 1=引擎安全可用，0=内部状态不完整
 */
int tts_engine_is_healthy(TTSEngine* engine);

/**
 * @brief 设置音高偏移
 * @param engine 引擎句柄
 * @param pitch 音高偏移（半音，范围[-12, 12]）
 * @return int 成功返回0，失败返回-1
 */
int tts_engine_set_pitch(TTSEngine* engine, float pitch);

/**
 * @brief 设置音量
 * @param engine 引擎句柄
 * @param volume 音量（0.0~1.0）
 * @return int 成功返回0，失败返回-1
 */
int tts_engine_set_volume(TTSEngine* engine, float volume);

/* F-012: TTS模型权重保存/加载 */
/**
 * @brief 保存TTS字符嵌入权重到文件
 * @param engine 引擎句柄
 * @param filepath 文件路径
 * @return 0成功, -1失败
 */
int tts_save_weights(TTSEngine* engine, const char* filepath);

/**
 * @brief 从文件加载TTS字符嵌入权重
 * @param engine 引擎句柄
 * @param filepath 文件路径
 * @return 0成功, -1失败
 */
int tts_load_weights(TTSEngine* engine, const char* filepath);

/**
 * @brief 保存完整TTS模型权重（编码器+解码器+神经声码器）到二进制文件
 *
 * 包含：
 * - 3层CfC ODE编码器权重（W, U, bias, tau）
 * - 3层自回归CfC ODE解码器权重（W, U, bias, tau, output_w, output_b）
 * - 8层WaveNet风格扩张卷积声码器权重
 * - 字符嵌入表
 *
 * @param engine 引擎句柄
 * @param filepath 文件路径
 * @return 0成功, -1失败
 */
int tts_save_model(TTSEngine* engine, const char* filepath);

/**
 * @brief 从二进制文件加载完整TTS模型权重
 *
 * 自动初始化权重缓冲区（如果尚未初始化），然后从文件填充。
 * 维度不匹配时返回错误。
 *
 * @param engine 引擎句柄
 * @param filepath 文件路径
 * @return 0成功, -1失败
 */
int tts_load_model(TTSEngine* engine, const char* filepath);

/**
 * @brief 获取拼音字符串表示
 * @param pinyin 拼音结构体
 * @return const char* 拼音字符串
 */
const char* tts_get_pinyin_string(const TTS_Pinyin* pinyin);

/**
 * @brief 将文本转换为拼音序列
 * @param engine 引擎句柄
 * @param text 输入文本
 * @param output 拼音输出缓冲区
 * @param max_output 最大输出数量
 * @return int 成功返回拼音数量，失败返回-1
 */
int tts_text_to_pinyin(TTSEngine* engine, const char* text, TTS_Pinyin* output, int max_output);

/**
 * @brief Unicode码点→拼音查找（二分查找精确表 + 启发式回退）
 * @param codepoint Unicode码点
 * @param out_init 输出声母索引(TTS_Initial枚举值)
 * @param out_final 输出韵母索引(TTS_Final枚举值)
 * @param out_tone 输出声调(0-4)
 * @return 1=精确匹配, 0=启发式近似
 */
int tts_pinyin_lookup(uint16_t codepoint, int* out_init, int* out_final, int* out_tone);

/**
 * @brief 从GB2312编码查找拼音（快速辅助函数）
 * @param gb_hi GB2312高字节(0xA1-0xF7)
 * @param gb_lo GB2312低字节(0xA1-0xFE)
 * @param out_init 输出声母索引
 * @param out_final 输出韵母索引
 * @param out_tone 输出声调(0-4)
 * @return 1=成功, 0=失败
 */
int tts_pinyin_lookup_gb2312(unsigned char gb_hi, unsigned char gb_lo,
                              int* out_init, int* out_final, int* out_tone);

/**
 * @brief 获取拼音精确表大小
 * @return 表条目数
 */
int tts_pinyin_table_size(void);

/* ================================================================
 * 源-滤波器模型语音合成接口（LF声门脉冲 + LPC全极点声道滤波器）
 * ================================================================ */

/** @brief LPC最大阶数 */
#define SF_LPC_MAX_ORDER 48

/** @brief 最大共振峰数量 */
#define SF_MAX_FORMANTS 6

/**
 * @brief 说话人特征参数结构体（声源+声道完整描述）
 */
typedef struct {
    /* 声源参数（LF模型） */
    float f0_mean;
    float f0_min;
    float f0_max;
    float f0_std;
    
    /* LF声门模型参数 */
    float lf_open_quotient;
    float lf_speed_quotient;
    float lf_return_quotient;
    float lf_spectral_tilt;
    
    /* 声道滤波器参数 */
    int lpc_order;
    float lpc_coeffs[SF_LPC_MAX_ORDER];
    float lpc_gain;
    
    /* 共振峰参数 */
    float formant_freq[SF_MAX_FORMANTS];
    float formant_bw[SF_MAX_FORMANTS];
    float formant_amp[SF_MAX_FORMANTS];
    
    /* 声道特征 */
    float vocal_tract_length;
    float mouth_opening;
    
    /* 声调特征 */
    float tone_range_semitones;
    float tone_register_base;
    
    /* 韵律特征 */
    float speaking_rate;
    float mean_energy;
    float energy_variation;
    
    /* 模型元数据 */
    char speaker_name[64];
    char speaker_gender;
    int sample_rate;
    uint32_t magic;
    uint32_t version;
} VoiceModelParams;

/**
 * @brief 获取默认声音模型参数（普通话女性）
 * @param params 输出参数结构体（需由调用者分配内存）
 */
void tts_pinyin_get_default_voice_model(VoiceModelParams* params);

/**
 * @brief 评估声音模型参数的有效性
 * @param params 声音模型参数
 * @return 1=有效, 0=无效
 */
int tts_pinyin_validate_voice_model(const VoiceModelParams* params);

/**
 * @brief 保存说话人特征参数到二进制文件
 * @param params 声音模型参数
 * @param filepath 文件路径
 * @return 0成功, -1失败
 */
int tts_pinyin_save_voice_model(const VoiceModelParams* params, const char* filepath);

/**
 * @brief 从二进制文件加载说话人特征参数
 * @param params 输出声音模型参数（需由调用者分配内存）
 * @param filepath 文件路径
 * @return 0成功, -1失败
 */
int tts_pinyin_load_voice_model(VoiceModelParams* params, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_TTS_H */
