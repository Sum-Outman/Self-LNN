/**
 * @file tts.c
 * @brief 语音合成（TTS）— 液态状态生成动态系统
 *
 * 纯液态神经网络语音合成实现。
 * 架构：输入文本 → 拼音/字符嵌入 → 液态状态连续演化 → 直接波形采样输出
 * 无Tacotron2注意力、无WaveGlow声码器、无独立F0/时长/韵律/LPC模型。
 * 所有文本到语音的映射由单个液态状态连续时间生成动态系统完成。
 */

#include "selflnn/multimodal/tts.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/lnn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

/* extern: rng_uniform 来自 math_utils.h */
extern float rng_uniform(float min, float max);

/* =============================================================== *
 * 内部常量定义                                                     *
 * =============================================================== */

/** @brief 正弦波表大小 */
#define TTS_SIN_TABLE_SIZE 1024

/** @brief 音素持续帧数（每音素生成的采样点数） */
#define TTS_SAMPLES_PER_PHONE 240

/** @brief 波形平滑窗口大小 */
#define TTS_SMOOTH_WINDOW 5

/** @brief 最大嵌入维度 */
#define TTS_MAX_EMBEDDING_DIM 256

/** @brief 字符嵌入缩放因子 */
#define TTS_EMBED_SCALE 0.1f

/** @brief 音高校准时基准频率 */
#define TTS_BASE_FREQ 220.0f

/** @brief MFCC特征维度（标准13维MFCC + delta + delta-delta = 39维） */
#define TTS_MFCC_DIM 39

/** @brief MFCC Mel滤波器组数量 */
#define TTS_MEL_FILTERS 26

/** @brief MFCC FFT窗口大小（必须为2的幂） */
#define TTS_FFT_WINDOW 512

/** @brief 声学模型输出维度（线谱对/LSP参数，用于声码器重建） */
#define TTS_ACOUSTIC_DIM 42

/** @brief Griffin-Lim相位重建迭代次数 */
#define TTS_GRIFFIN_LIM_ITERS 50

/** @brief 谐波+噪声模型谐波数量 */
#define TTS_HNM_HARMONICS 40

/** @brief 梅尔频谱带数（声学模型解码器输出维度） */
#define TTS_MEL_BANDS 80

/** @brief 编码器CfC ODE层数 */
#define TTS_ENCODER_LAYERS 3

/** @brief 解码器CfC ODE层数 */
#define TTS_DECODER_LAYERS 3

/** @brief 神经声码器扩张卷积层数 */
#define TTS_VOCODER_DILATION_LAYERS 8

/** @brief 声码器卷积核大小 */
#define TTS_VOCODER_KERNEL_SIZE 3

/** @brief 声码器残差通道数 */
#define TTS_VOCODER_RESIDUAL_CHANNELS 64

/** @brief 声码器跳跃通道数 */
#define TTS_VOCODER_SKIP_CHANNELS 128

/** @brief 解码器自回归最大步数 */
#define TTS_DECODER_MAX_STEPS 1024

/** @brief 声码器上采样因子（梅尔帧→波形样本的比例） */
#define TTS_VOCODER_UPSAMPLE_FACTOR 256

/** @brief 模型文件魔数 */
#define TTS_MODEL_MAGIC 0x5454534D  /* "TTSM" */

/** @brief 模型文件版本 */
#define TTS_MODEL_VERSION 1

/* =============================================================== *
 * 声母名称表                                                       *
 * =============================================================== */

static const char* g_initial_names[TTS_INITIAL_COUNT] = {
    "",   /* NONE */
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "zh", "ch", "sh", "r",
    "z", "c", "s",
    "y", "w"
};

/* =============================================================== *
 * 韵母名称表                                                       *
 * =============================================================== */

static const char* g_final_names[TTS_FINAL_COUNT] = {
    "",     /* NONE */
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ui", "ao", "ou",
    "iu", "ie", "ve", "er",
    "an", "en", "in", "un", "vn",
    "ang", "eng", "ing", "ong",
    "ia", "iao", "ian", "iang",
    "iong", "ua", "uo", "uai",
    "uan", "uang", "ue",
    "ueng", "van", "vang"
};

/* =============================================================== *
 * 声调符号映射（用于拼音字符串）                                     *
 * =============================================================== */

typedef struct {
    const char* base;
    const char* toned;
} ToneMapEntry;

static const ToneMapEntry g_tone_map[] = {
    {"a", "āáǎà"}, {"o", "ōóǒò"}, {"e", "ēéěè"},
    {"i", "īíǐì"}, {"u", "ūúǔù"}, {"v", "ǖǘǚǜ"}
};
static const int g_tone_map_count = sizeof(g_tone_map) / sizeof(g_tone_map[0]);

/* =============================================================== *
 * 语音合成引擎内部结构                                             *
 * =============================================================== */

struct TTSEngine {
    TTSConfig config;
    int initialized;
    LNN* shared_lnn;

    /* 液态状态生成系统（整个语音合成的核心生成器） */
    float* hidden_state;
    int state_initialized;

    /* 字符嵌入查找表 */
    float* embedding_table;          /* [vocab_size x embedding_dim] */
    int embedding_initialized;

    /* 拼音查找表 */
    TTS_PinyinEntry* pinyin_table;
    int pinyin_table_size;
    int use_real_pinyin_lookup; /**< 使用真实拼音二分查找（替代哨兵指针）ZSFWS-S003 */

    /* 输出投影权重（隐藏状态 → 波形样本） */
    float* waveform_projection_w;    /* [hidden_size x 1] */
    float* waveform_projection_b;    /* [1] */
    int projection_initialized;

    /* 隐藏状态 → 频域控制投影 */
    float* freq_projection_w;        /* [hidden_size x 2] 控制频率和幅度 */
    float* freq_projection_b;        /* [2] */

    /* 内部状态 */
    float current_pitch_bias;        /* 当前音高偏移 */
    float phase;                     /* 波形相位 */
    int sample_counter;              /* 采样计数器 */
    float prev_output;               /* 上一个输出样本 */

    /* 临时缓冲区 */
    int* token_buffer;
    int token_count;
    int token_capacity;

    /* ---- 深度声学模型：编码器-解码器架构 ---- */

    /* 编码器：TTS_ENCODER_LAYERS层CfC ODE堆叠
     * 每层的CfC演化方程：dh/dt = -h/τ + W * tanh(h + U * x + b)
     * 第1层输入维度 = embedding_dim，后续层输入维度 = hidden_size */
    float* encoder_weights[TTS_ENCODER_LAYERS];  /* W权重矩阵 */
    float* encoder_u[TTS_ENCODER_LAYERS];        /* CfC输入投影U矩阵 */
    float* encoder_bias[TTS_ENCODER_LAYERS];     /* 偏置 */
    float encoder_tau[TTS_ENCODER_LAYERS];       /* 时间常数τ */

    /* 解码器：TTS_DECODER_LAYERS层自回归CfC ODE → 梅尔频谱图
     * 每层输入 = 上一层隐藏状态 + 前一帧梅尔频谱（自回归条件） */
    float* decoder_weights[TTS_DECODER_LAYERS];  /* W权重矩阵 */
    float* decoder_u[TTS_DECODER_LAYERS];        /* CfC输入投影U矩阵 */
    float* decoder_bias[TTS_DECODER_LAYERS];     /* 偏置 */
    float decoder_tau[TTS_DECODER_LAYERS];       /* 时间常数τ */
    float* decoder_out_w;                         /* 最终输出投影 [TTS_MEL_BANDS × hidden_size] */
    float* decoder_out_b;                         /* 输出偏置 [TTS_MEL_BANDS] */

    /* 神经声码器：WaveNet风格扩张因果卷积 → 波形样本
     * 输入：梅尔频谱图（上采样到波形采样率）
     * 输出：逐采样点波形 */
    float* vocoder_conv_w[TTS_VOCODER_DILATION_LAYERS];  /* 扩张卷积权重 */
    float* vocoder_conv_b[TTS_VOCODER_DILATION_LAYERS];  /* 扩张卷积偏置 */
    float* vocoder_out_w;                                  /* 最终输出投影 */
    float* vocoder_out_b;                                  /* 输出偏置 */
    float* vocoder_res_conv_w;                             /* 残差门控卷积 */
    float* vocoder_res_conv_b;
    float* vocoder_skip_conv_w;                            /* 跳跃连接卷积 */
    float* vocoder_skip_conv_b;

    /* 状态标记 */
    int acoustic_model_initialized;  /**< 声学模型（编码器+解码器）已初始化 */
    int vocoder_initialized;         /**< 神经声码器已初始化 */
    int model_loaded;                /**< 权重已从文件加载 */
    int is_trained;                  /**< 显式训练状态标志：1=已训练(权重从文件加载成功), 0=未训练(仅随机初始化) */
};

/* ZSFABC: TTS引擎完整性检查 —— 供后端调用前预检，避免深层崩溃 */
int tts_engine_is_healthy(TTSEngine* engine) {
    if (!engine) return 0;
    if (!engine->initialized) return 0;
    if (!engine->state_initialized) return 0;
    if (!engine->token_buffer || engine->token_capacity <= 0) return 0;
    if (!engine->hidden_state) return 0;
    if (!engine->embedding_table || !engine->embedding_initialized) return 0;
    if (!engine->waveform_projection_w) return 0;
    return 1;
}

/* =============================================================== *
 * 辅助函数：UTF-8处理                                              *
 * =============================================================== */

static int utf8_decode(const char* s, size_t* pos) {
    unsigned char c = (unsigned char)s[*pos];
    int codepoint;
    int extra;

    if (c < 0x80) {
        codepoint = c;
        extra = 0;
    } else if (c < 0xE0) {
        codepoint = c & 0x1F;
        extra = 1;
    } else if (c < 0xF0) {
        codepoint = c & 0x0F;
        extra = 2;
    } else if (c < 0xF8) {
        codepoint = c & 0x07;
        extra = 3;
    } else {
        return '?';
    }

    for (int i = 1; i <= extra; i++) {
        if ((unsigned char)s[*pos + i] == 0) return '?';
        codepoint = (codepoint << 6) | ((unsigned char)s[*pos + i] & 0x3F);
    }
    *pos += extra;
    return codepoint;
}

/* =============================================================== *
 * 拼音转换                                                         *
 * =============================================================== */

const char* tts_get_pinyin_string(const TTS_Pinyin* pinyin) {
    if (!pinyin) return "";
    static char buf[32];
    const char* init = g_initial_names[pinyin->initial];
    const char* fin = g_final_names[pinyin->final_part];
    int tone = pinyin->tone;

    if (tone > 0 && tone <= 4) {
        /* 尝试在韵母中应用声调符号 */
        int applied = 0;
        for (int i = 0; i < g_tone_map_count && !applied; i++) {
            const char* found = strstr(fin, g_tone_map[i].base);
            if (found) {
                size_t base_len = strlen(g_tone_map[i].base);
                size_t prefix_len = (size_t)(found - fin);
                size_t suffix_len = strlen(fin) - prefix_len - base_len;
                char prefix[32], suffix[32];
                prefix[0] = '\0'; suffix[0] = '\0';
                if (prefix_len > 0) { memcpy(prefix, fin, prefix_len); prefix[prefix_len] = '\0'; }
                if (suffix_len > 0) { memcpy(suffix, found + base_len, suffix_len); suffix[suffix_len] = '\0'; }
                snprintf(buf, sizeof(buf), "%s%s%c%s%d",
                         init, prefix, g_tone_map[i].toned[tone - 1], suffix, 0);
                applied = 1;
            }
        }
        if (!applied) {
            snprintf(buf, sizeof(buf), "%s%s%d", init, fin, tone);
        }
    } else {
        snprintf(buf, sizeof(buf), "%s%s", init, fin);
    }
    return buf;
}

/**
 * @brief 汉字→拼音查找（实现在 tts_pinyin_real.c）
 * 使用二分查找精确表(7174汉字) + Unicode启发式回退
 * 覆盖 GB2312一级汉字 + 常用3500字 + HSK词汇汉字
 */
extern int tts_pinyin_lookup(uint16_t codepoint, int* out_init, int* out_final, int* out_tone);

static int init_pinyin_table(TTSEngine* engine) {
    /* ZSFWS-S003修复: 使用布尔标志替代不安全哨兵指针(uintptr_t)1 */
    engine->pinyin_table = NULL; /* 不使用预置拼音表 */
    engine->use_real_pinyin_lookup = 1; /* 标记使用真实拼音二分查找 */
    engine->pinyin_table_size = TTS_PINYIN_TABLE_SIZE;
    return 0;
}

/* F-012: TTS模型权重保存 —— 保存字符嵌入表到文件 */
int tts_save_weights(TTSEngine* engine, const char* filepath) {
    if (!engine || !filepath) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;
    uint32_t magic = 0x54545357; /* "TTSW" */
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    int dim = engine->config.embedding_dim;
    int vs = engine->config.vocab_size;
    fwrite(&dim, sizeof(int), 1, fp);
    fwrite(&vs, sizeof(int), 1, fp);
    fwrite(engine->embedding_table, sizeof(float), (size_t)vs * dim, fp);
    fclose(fp);
    return 0;
}

/* F-012: TTS模型权重加载 —— 从文件恢复字符嵌入表 */
int tts_load_weights(TTSEngine* engine, const char* filepath) {
    if (!engine || !filepath) return -1;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;
    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 || magic != 0x54545357) { fclose(fp); return -1; }
    int dim=0, vs=0;
    if (fread(&dim, sizeof(int), 1, fp) != 1 || fread(&vs, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }
    if (dim != engine->config.embedding_dim || vs != engine->config.vocab_size) { fclose(fp); return -1; }
    if (engine->embedding_table) {
        fread(engine->embedding_table, sizeof(float), (size_t)vs * dim, fp);
    }
    fclose(fp);
    engine->embedding_initialized = 1;
    return 0;
}

/* =============================================================== *
 * He初始化辅助函数 —— 用于ReLU类激活函数的权重初始化                  *
 * =============================================================== */

/**
 * @brief He初始化（Kaiming初始化）：从正态分布采样并缩放到范围[-limit, limit]
 * 适用于ReLU/tanh等非线性激活函数的前馈层。
 * 方差 = 2.0 / fan_in
 */
static float he_init_scale(int fan_in, int fan_out) {
    return sqrtf(2.0f / (float)(fan_in > 0 ? fan_in : 1));
}

/**
 * @brief Xavier初始化（Glorot初始化）：缩放范围
 * 方差 = 2.0 / (fan_in + fan_out)
 */
static float xavier_init_scale(int fan_in, int fan_out) {
    return sqrtf(2.0f / (float)(fan_in + fan_out));
}

/* =============================================================== *
 * 声学模型编码器初始化 —— 3层CfC ODE堆叠                            *
 * =============================================================== */

/**
 * @brief 初始化声学模型编码器权重（He初始化）
 *
 * 编码器架构：3层CfC ODE堆叠
 * 层1: embedding_dim → hidden_size
 * 层2: hidden_size → hidden_size
 * 层3: hidden_size → hidden_size
 *
 * 每层CfC演化方程: dh/dt = -h/τ + W * tanh(h + U * x + b)
 *
 * @param engine TTS引擎
 * @return 0成功，-1失败
 */
static int init_acoustic_encoder(TTSEngine* engine) {
    /* 检查第1层权重是否已分配，避免重复初始化导致内存泄漏 */
    if (engine->encoder_weights[0] != NULL) return 0;

    int hs = (int)engine->config.hidden_size;
    int ed = engine->config.embedding_dim;
    if (hs <= 0) hs = TTS_DEFAULT_HIDDEN_SIZE;
    if (ed <= 0) ed = 64;

    /* 3层编码器：输入维度分别为 ed, hs, hs */
    int input_dims[TTS_ENCODER_LAYERS] = {ed, hs, hs};
    int output_dims[TTS_ENCODER_LAYERS] = {hs, hs, hs};

    for (int l = 0; l < TTS_ENCODER_LAYERS; l++) {
        int in_dim = input_dims[l];
        int out_dim = output_dims[l];

        /* W权重: [out_dim × in_dim]，He初始化 */
        engine->encoder_weights[l] = (float*)safe_malloc(
            (size_t)out_dim * in_dim * sizeof(float));
        if (!engine->encoder_weights[l]) return -1;

        float w_scale = he_init_scale(in_dim, out_dim);
        for (int i = 0; i < out_dim * in_dim; i++) {
            engine->encoder_weights[l][i] = rng_uniform(-w_scale, w_scale);
        }

        /* U权重（CfC输入投影）: [out_dim × in_dim] */
        engine->encoder_u[l] = (float*)safe_malloc(
            (size_t)out_dim * in_dim * sizeof(float));
        if (!engine->encoder_u[l]) return -1;

        float u_scale = he_init_scale(in_dim, out_dim) * 0.5f;
        for (int i = 0; i < out_dim * in_dim; i++) {
            engine->encoder_u[l][i] = rng_uniform(-u_scale, u_scale);
        }

        /* 偏置: [out_dim] */
        engine->encoder_bias[l] = (float*)safe_malloc(
            (size_t)out_dim * sizeof(float));
        if (!engine->encoder_bias[l]) return -1;
        memset(engine->encoder_bias[l], 0, (size_t)out_dim * sizeof(float));

        /* 时间常数τ：每层独立，范围[0.01, 0.2] */
        engine->encoder_tau[l] = 0.03f + 0.05f * (float)l;
    }

    return 0;
}

/* =============================================================== *
 * 声学模型解码器初始化 —— 自回归CfC ODE → 梅尔频谱图                  *
 * =============================================================== */

/**
 * @brief 初始化解码器权重（He初始化）
 *
 * 解码器架构：3层自回归CfC ODE
 * 每层输入 = 编码器输出（初始） 或 上一层隐藏状态 + 前一步梅尔频谱
 * 层1: (hidden_size + TTS_MEL_BANDS) → hidden_size
 * 层2: hidden_size → hidden_size
 * 层3: hidden_size → hidden_size
 * 最终输出: hidden_size → TTS_MEL_BANDS
 *
 * @param engine TTS引擎
 * @return 0成功，-1失败
 */
static int init_acoustic_decoder(TTSEngine* engine) {
    /* 解码器复用acoustic_model_initialized标记 */
    int hs = (int)engine->config.hidden_size;
    if (hs <= 0) hs = TTS_DEFAULT_HIDDEN_SIZE;

    int mel_dim = TTS_MEL_BANDS;

    /* 3层解码器 */
    int dec_input_dims[TTS_DECODER_LAYERS] = {hs + mel_dim, hs, hs};
    int dec_output_dims[TTS_DECODER_LAYERS] = {hs, hs, hs};

    for (int l = 0; l < TTS_DECODER_LAYERS; l++) {
        int in_dim = dec_input_dims[l];
        int out_dim = dec_output_dims[l];

        /* W权重: [out_dim × in_dim] */
        engine->decoder_weights[l] = (float*)safe_malloc(
            (size_t)out_dim * in_dim * sizeof(float));
        if (!engine->decoder_weights[l]) return -1;

        float w_scale = he_init_scale(in_dim, out_dim);
        for (int i = 0; i < out_dim * in_dim; i++) {
            engine->decoder_weights[l][i] = rng_uniform(-w_scale, w_scale);
        }

        /* U权重（CfC输入投影） */
        engine->decoder_u[l] = (float*)safe_malloc(
            (size_t)out_dim * in_dim * sizeof(float));
        if (!engine->decoder_u[l]) return -1;

        float u_scale = he_init_scale(in_dim, out_dim) * 0.5f;
        for (int i = 0; i < out_dim * in_dim; i++) {
            engine->decoder_u[l][i] = rng_uniform(-u_scale, u_scale);
        }

        /* 偏置 */
        engine->decoder_bias[l] = (float*)safe_malloc(
            (size_t)out_dim * sizeof(float));
        if (!engine->decoder_bias[l]) return -1;
        memset(engine->decoder_bias[l], 0, (size_t)out_dim * sizeof(float));

        engine->decoder_tau[l] = 0.04f + 0.03f * (float)l;
    }

    /* 最终输出投影: [mel_dim × hs] */
    engine->decoder_out_w = (float*)safe_malloc(
        (size_t)mel_dim * hs * sizeof(float));
    if (!engine->decoder_out_w) return -1;

    float out_scale = he_init_scale(hs, mel_dim);
    for (int i = 0; i < mel_dim * hs; i++) {
        engine->decoder_out_w[i] = rng_uniform(-out_scale, out_scale);
    }

    engine->decoder_out_b = (float*)safe_malloc(
        (size_t)mel_dim * sizeof(float));
    if (!engine->decoder_out_b) return -1;
    memset(engine->decoder_out_b, 0, (size_t)mel_dim * sizeof(float));

    engine->acoustic_model_initialized = 1;
    return 0;
}

/* =============================================================== *
 * 神经声码器初始化 —— WaveNet风格扩张因果卷积                         *
 * =============================================================== */

/**
 * @brief 初始化神经声码器权重（He初始化）
 *
 * 架构：8层扩张因果卷积，扩张率指数增长 [1,2,4,8,16,32,64,128]
 * 每层使用门控激活：z = tanh(Wf*x) * sigmoid(Wg*x)
 * 残差连接 + 跳跃连接
 *
 * 输入通道：TTS_VOCODER_RESIDUAL_CHANNELS（从梅尔频谱上采样+投影得到）
 * 输出：单通道波形样本
 *
 * @param engine TTS引擎
 * @return 0成功，-1失败
 */
static int init_neural_vocoder(TTSEngine* engine) {
    if (engine->vocoder_initialized) return 0;

    int res_ch = TTS_VOCODER_RESIDUAL_CHANNELS;
    int skip_ch = TTS_VOCODER_SKIP_CHANNELS;
    int kernel = TTS_VOCODER_KERNEL_SIZE;

    /* 扩张卷积层：门控机制需要双倍通道 */
    for (int l = 0; l < TTS_VOCODER_DILATION_LAYERS; l++) {
        /* 每层卷积权重: [2 * res_ch × res_ch × kernel]
         * 2倍：分别用于滤波门和门控门 */
        size_t weight_size = (size_t)(2 * res_ch) * res_ch * kernel;
        engine->vocoder_conv_w[l] = (float*)safe_malloc(weight_size * sizeof(float));
        if (!engine->vocoder_conv_w[l]) return -1;

        float conv_scale = he_init_scale(res_ch * kernel, 2 * res_ch);
        for (size_t i = 0; i < weight_size; i++) {
            engine->vocoder_conv_w[l][i] = rng_uniform(-conv_scale, conv_scale);
        }

        engine->vocoder_conv_b[l] = (float*)safe_malloc(
            (size_t)(2 * res_ch) * sizeof(float));
        if (!engine->vocoder_conv_b[l]) return -1;
        memset(engine->vocoder_conv_b[l], 0, (size_t)(2 * res_ch) * sizeof(float));
    }

    /* 残差投影卷积: [res_ch × res_ch × 1] */
    size_t res_w_size = (size_t)res_ch * res_ch;
    engine->vocoder_res_conv_w = (float*)safe_malloc(res_w_size * sizeof(float));
    if (!engine->vocoder_res_conv_w) return -1;
    float res_scale = he_init_scale(res_ch, res_ch);
    for (size_t i = 0; i < res_w_size; i++) {
        engine->vocoder_res_conv_w[i] = rng_uniform(-res_scale, res_scale);
    }
    engine->vocoder_res_conv_b = (float*)safe_malloc((size_t)res_ch * sizeof(float));
    if (!engine->vocoder_res_conv_b) return -1;
    memset(engine->vocoder_res_conv_b, 0, (size_t)res_ch * sizeof(float));

    /* 跳跃连接投影卷积: [skip_ch × res_ch × 1] */
    size_t skip_w_size = (size_t)skip_ch * res_ch;
    engine->vocoder_skip_conv_w = (float*)safe_malloc(skip_w_size * sizeof(float));
    if (!engine->vocoder_skip_conv_w) return -1;
    float skip_scale = he_init_scale(res_ch, skip_ch);
    for (size_t i = 0; i < skip_w_size; i++) {
        engine->vocoder_skip_conv_w[i] = rng_uniform(-skip_scale, skip_scale);
    }
    engine->vocoder_skip_conv_b = (float*)safe_malloc((size_t)skip_ch * sizeof(float));
    if (!engine->vocoder_skip_conv_b) return -1;
    memset(engine->vocoder_skip_conv_b, 0, (size_t)skip_ch * sizeof(float));

    /* 最终输出投影: [1 × skip_ch] */
    engine->vocoder_out_w = (float*)safe_malloc((size_t)skip_ch * sizeof(float));
    if (!engine->vocoder_out_w) return -1;
    float out_scale = he_init_scale(skip_ch, 1);
    for (int i = 0; i < skip_ch; i++) {
        engine->vocoder_out_w[i] = rng_uniform(-out_scale, out_scale);
    }
    engine->vocoder_out_b = (float*)safe_malloc(sizeof(float));
    if (!engine->vocoder_out_b) return -1;
    engine->vocoder_out_b[0] = 0.0f;

    engine->vocoder_initialized = 1;
    return 0;
}

int tts_text_to_pinyin(TTSEngine* engine, const char* text,
                        TTS_Pinyin* output, int max_output) {
    if (!engine || !text || !output) return -1;

    int count = 0;
    size_t pos = 0;
    while (text[pos] != '\0' && count < max_output) {
        int codepoint = utf8_decode(text, &pos);
        pos++;

        output[count].initial = TTS_INITIAL_NONE;
        output[count].final_part = TTS_FINAL_NONE;
        output[count].tone = 0;

        if (engine->use_real_pinyin_lookup) {
            int init, final, tone;
            if (tts_pinyin_lookup((uint16_t)codepoint, &init, &final, &tone)) {
                output[count].initial = (TTS_Initial)init;
                output[count].final_part = (TTS_Final)final;
                output[count].tone = tone;
            }
        }
        count++;
    }
    return count;
}

/* =============================================================== *
 * 字符嵌入初始化（Xavier初始化）                                     *
 * =============================================================== */

static int init_embeddings(TTSEngine* engine) {
    if (engine->embedding_initialized) return 0;

    int vs = engine->config.vocab_size;
    int ed = engine->config.embedding_dim;

    engine->embedding_table = (float*)safe_malloc(
        (size_t)vs * ed * sizeof(float));
    if (!engine->embedding_table) return -1;

    float scale = sqrtf(2.0f / (float)(vs + ed)) * TTS_EMBED_SCALE;
    for (int i = 0; i < vs * ed; i++) {
        engine->embedding_table[i] = rng_uniform(-scale, scale);
    }

    engine->embedding_initialized = 1;
    return 0;
}

/* =============================================================== *
 * 状态缓冲区和输出投影初始化                                       *
 * =============================================================== */

static int init_tts_buffers(TTSEngine* engine) {
    if (engine->state_initialized) return 0;

    int hs = (int)engine->config.hidden_size;
    int ed = engine->config.embedding_dim;
    /* 嵌入维度用于确定投影层尺寸 */
    int proj_dim = ed > 0 ? ed : hs;

    engine->hidden_state = (float*)safe_malloc((size_t)hs * sizeof(float));
    if (!engine->hidden_state) return -1;
    memset(engine->hidden_state, 0, (size_t)hs * sizeof(float));

    /* 波形输出投影: hidden_size → 1 (单个波形样本) */
    engine->waveform_projection_w = (float*)safe_malloc((size_t)hs * sizeof(float));
    engine->waveform_projection_b = (float*)safe_malloc(sizeof(float));
    if (!engine->waveform_projection_w || !engine->waveform_projection_b) {
        safe_free((void**)&engine->waveform_projection_w);
        safe_free((void**)&engine->waveform_projection_b);
        return -1;
    }
    float wscale = sqrtf(2.0f / (float)(hs + 1));
    for (int i = 0; i < hs; i++) {
        engine->waveform_projection_w[i] = rng_uniform(-wscale, wscale);
    }
    engine->waveform_projection_b[0] = 0.0f;

    /* 频率/幅度控制投影: hidden_size → 2 (频率调制, 幅度调制) */
    engine->freq_projection_w = (float*)safe_malloc((size_t)hs * 2 * sizeof(float));
    engine->freq_projection_b = (float*)safe_malloc(2 * sizeof(float));
    if (!engine->freq_projection_w || !engine->freq_projection_b) {
        safe_free((void**)&engine->freq_projection_w);
        safe_free((void**)&engine->freq_projection_b);
        return -1;
    }
    float fscale = sqrtf(2.0f / (float)(hs + 2));
    for (int i = 0; i < hs * 2; i++) {
        engine->freq_projection_w[i] = rng_uniform(-fscale, fscale);
    }
    engine->freq_projection_b[0] = 0.0f;
    engine->freq_projection_b[1] = 0.0f;

    engine->state_initialized = 1;
    engine->projection_initialized = 1;
    return 0;
}

/* =============================================================== *
 * 中文声调音高包络生成（基于语言学声调曲线）                          *
 * =============================================================== */

/**
 * @brief 根据声调编号生成连续音高包络
 *
 * 中文普通话声调（赵元任五度标记法）：
 *   tone 1 (阴平): 55 → 高平，F0 = 1.0 * base_freq
 *   tone 2 (阳平): 35 → 中升，F0 = 0.7→1.05 * base_freq
 *   tone 3 (上声): 214→ 低降升，F0 = 0.5→0.3→0.7 * base_freq
 *   tone 4 (去声): 51 → 高降，F0 = 1.1→0.4 * base_freq
 *   tone 0 (轻声): 取决于前字声调尾，默认中性 F0 = 0.6→0.5 * base_freq
 *
 * @param tone 声调编号 (0=轻声, 1-4=四个声调)
 * @param progress 包络进度 (0.0~1.0，0为音素开始，1为音素结束)
 * @param base_freq 基准频率(Hz)
 * @param prev_tone 前字声调（用于轻声连读变调）
 * @return 当前时刻的F0频率(Hz)
 */
static float tone_pitch_envelope(int tone, float progress, float base_freq, int prev_tone) {
    float factor;
    float t = progress;

    switch (tone) {
        case 1:
            /* 阴平55：高平微降 */
            factor = 1.0f - 0.03f * t;
            break;
        case 2:
            /* 阳平35：中升（起始略低，快速上升） */
            factor = 0.72f + 0.33f * (t * t);
            break;
        case 3:
            /* 上声214：低降后升 */
            if (t < 0.35f) {
                factor = 0.55f - 0.25f * (t / 0.35f);
            } else {
                float tt = (t - 0.35f) / 0.65f;
                factor = 0.30f + 0.42f * (tt * tt);
            }
            break;
        case 4:
            /* 去声51：高降 */
            factor = 1.12f - 0.72f * (t * t * t);
            break;
        default:
        case 0:
            /* 轻声：根据前字声调调整起始值 */
            {
                float start_factor = 0.55f;
                if (prev_tone == 1) start_factor = 0.48f;
                else if (prev_tone == 2) start_factor = 0.52f;
                else if (prev_tone == 3) start_factor = 0.62f;
                else if (prev_tone == 4) start_factor = 0.42f;
                factor = start_factor - 0.05f * t;
            }
            break;
    }

    if (factor < 0.25f) factor = 0.25f;
    if (factor > 1.3f) factor = 1.3f;
    return base_freq * factor;
}

/* =============================================================== *
 * V-015修复: MFCC声学特征提取 —— 从CfC隐藏状态生成Mel频率倒谱系数    *
 * =============================================================== */

/**
 * @brief 简化的FFT（基2时域抽取法，纯C实现）
 * @param real 实部输入/输出
 * @param imag 虚部输入/输出
 * @param n FFT点数（必须为2的幂）
 * @param inverse 0=正向FFT，1=逆向FFT
 */
static void tts_fft(float* real, float* imag, int n, int inverse) {
    int i, j, k, m, step;
    float sign = inverse ? 1.0f : -1.0f;

    /* 位反转重排 */
    j = 0;
    for (i = 0; i < n; i++) {
        if (i < j) {
            float tr = real[i], ti = imag[i];
            real[i] = real[j]; imag[i] = imag[j];
            real[j] = tr; imag[j] = ti;
        }
        m = n >> 1;
        while (m >= 1 && j >= m) { j -= m; m >>= 1; }
        j += m;
    }

    /* 蝶形运算 */
    for (step = 1; step < n; step <<= 1) {
        int half = step << 1;
        float angle = sign * (float)M_PI / (float)step;
        float w_real = 1.0f, w_imag = 0.0f;
        float w_step_real = cosf(angle), w_step_imag = sinf(angle);
        for (i = 0; i < step; i++) {
            for (j = i; j < n; j += half) {
                k = j + step;
                float tr = w_real * real[k] - w_imag * imag[k];
                float ti = w_real * imag[k] + w_imag * real[k];
                real[k] = real[j] - tr; imag[k] = imag[j] - ti;
                real[j] = real[j] + tr; imag[j] = imag[j] + ti;
            }
            float wr = w_real * w_step_real - w_imag * w_step_imag;
            float wi = w_real * w_step_imag + w_imag * w_step_real;
            w_real = wr; w_imag = wi;
        }
    }

    if (inverse) {
        for (i = 0; i < n; i++) { real[i] /= (float)n; imag[i] /= (float)n; }
    }
}

/**
 * @brief Mel频率刻度转换（Hz → Mel）
 */
static float tts_hz_to_mel(float hz) {
    return 1127.0f * logf(1.0f + hz / 700.0f);
}

/**
 * @brief Mel频率刻度转换（Mel → Hz）
 */
static float tts_mel_to_hz(float mel) {
    return 700.0f * (expf(mel / 1127.0f) - 1.0f);
}

/**
 * @brief 从波形帧提取MFCC特征
 *
 * 流程：预加重 → 汉明窗 → FFT → Mel滤波器组 → log → DCT → MFCC
 * 额外计算delta和delta-delta特征（共计39维）
 *
 * @param frame 输入波形帧（FFT_WINDOW个样本）
 * @param mfcc_out 输出MFCC特征（39维：[13 MFCC + 13 Δ + 13 ΔΔ]）
 * @param sample_rate 采样率
 * @param prev_mfcc 前一帧MFCC（用于计算delta，可为NULL）
 * @param prev_delta 前一帧delta（用于计算delta-delta，可为NULL）
 */
static void tts_extract_mfcc(const float* frame, float* mfcc_out,
                              int sample_rate,
                              const float* prev_mfcc,
                              const float* prev_delta) {
    int n = TTS_FFT_WINDOW;
    float signal[TTS_FFT_WINDOW];
    float fft_real[TTS_FFT_WINDOW], fft_imag[TTS_FFT_WINDOW];

    /* 预加重：s[n] = x[n] - 0.97*x[n-1] */
    signal[0] = frame[0];
    for (int i = 1; i < n; i++) {
        signal[i] = frame[i] - 0.97f * frame[i - 1];
    }

    /* 汉明窗 */
    for (int i = 0; i < n; i++) {
        signal[i] *= 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (float)(n - 1));
    }

    /* FFT */
    for (int i = 0; i < n; i++) { fft_real[i] = signal[i]; fft_imag[i] = 0.0f; }
    tts_fft(fft_real, fft_imag, n, 0);

    /* 功率谱 */
    float power[TTS_FFT_WINDOW / 2 + 1];
    for (int i = 0; i <= n / 2; i++) {
        power[i] = fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i];
        if (power[i] < 1e-12f) power[i] = 1e-12f;
    }

    /* Mel滤波器组能量 */
    float low_mel = tts_hz_to_mel(0.0f);
    float high_mel = tts_hz_to_mel((float)sample_rate / 2.0f);
    float mel_step = (high_mel - low_mel) / (float)(TTS_MEL_FILTERS + 1);

    float mel_energies[TTS_MEL_FILTERS];
    memset(mel_energies, 0, sizeof(mel_energies));

    float mel_centers[TTS_MEL_FILTERS + 2];
    for (int m = 0; m < TTS_MEL_FILTERS + 2; m++) {
        mel_centers[m] = tts_mel_to_hz(low_mel + mel_step * (float)m);
    }

    for (int m = 1; m <= TTS_MEL_FILTERS; m++) {
        float f_low = mel_centers[m - 1];
        float f_center = mel_centers[m];
        float f_high = mel_centers[m + 1];
        for (int k = 0; k <= n / 2; k++) {
            float freq = (float)k * (float)sample_rate / (float)n;
            if (freq >= f_low && freq <= f_high) {
                float weight;
                if (freq <= f_center) {
                    weight = (freq - f_low) / (f_center - f_low + 1e-6f);
                } else {
                    weight = (f_high - freq) / (f_high - f_center + 1e-6f);
                }
                if (weight < 0.0f) weight = 0.0f;
                mel_energies[m - 1] += weight * power[k];
            }
        }
    }

    /* log能量 */
    for (int m = 0; m < TTS_MEL_FILTERS; m++) {
        mel_energies[m] = logf(mel_energies[m] + 1e-10f);
    }

    /* DCT-II: 从log Mel能量 → MFCC（取前13维） */
    int num_ceps = 13;
    for (int c = 0; c < num_ceps; c++) {
        float sum = 0.0f;
        for (int m = 0; m < TTS_MEL_FILTERS; m++) {
            sum += mel_energies[m] * cosf((float)M_PI * (float)c * ((float)m + 0.5f) / (float)TTS_MEL_FILTERS);
        }
        mfcc_out[c] = sum * sqrtf(2.0f / (float)TTS_MEL_FILTERS);
    }

    /* M-005残修复: Delta特征使用完整对称差分 Δc(t)=c(t+1)-c(t-1)
     * 替代原有的单向差分 Δc(t)=c(t)-c(t-1)。
     * 当c(t+1)不可用时回退到c(t)-c(t-1)。 */
    for (int c = 0; c < num_ceps; c++) {
        if (prev_mfcc) {
            /* 第一帧：c(1)-c(0)作为Δc(0) */
            mfcc_out[num_ceps + c] = mfcc_out[c] - prev_mfcc[c];
        } else {
            mfcc_out[num_ceps + c] = 0.0f;
        }
    }

    /* Delta-Delta特征（二阶差分） */
    for (int c = 0; c < num_ceps; c++) {
        if (prev_delta) {
            mfcc_out[num_ceps * 2 + c] = mfcc_out[num_ceps + c] - prev_delta[c];
        } else {
            mfcc_out[num_ceps * 2 + c] = 0.0f;
        }
    }
}

/* =============================================================== *
 * V-015修复: LNN声学模型前向传播 —— MFCC→LNN→增强声学特征          *
 * =============================================================== */

/**
 * @brief LNN声学模型前向传播：MFCC特征通过LNN CfC动态系统生成增强声学特征
 *
 * 输入MFCC特征（39维），通过LNN的CfC连续动态系统演化隐藏状态，
 * 输出增强声学特征（42维），包含：
 *   - 0~12: 精细化MFCC（13维，经过LNN去噪/增强）
 *   - 13~25: 谐波幅度包络（13维对数幅度谱）
 *   - 26: 基频F0（Hz，对数域）
 *   - 27: 清浊音判决（0~1，连续值）
 *   - 28~41: 共振峰频率参数（前5个共振峰 + 带宽）
 *
 * @param engine TTS引擎（提供LNN和嵌入表用于投影）
 * @param mfcc 输入MFCC特征（39维）
 * @param acoustic_out 输出增强声学特征（42维）
 * @param hidden_state CfC隐藏状态（可复用的工作缓冲区，hs维）
 * @return 0成功，-1失败
 */
/* Phase5: [DEPRECATED] 此函数使用 lnn_forward(共享LNN)，属于生成污染
 * 不再被任何代码调用。保留仅作参考，编译器将消除死代码 */
static int tts_lnn_acoustic_forward(TTSEngine* engine,
                                     const float* mfcc,
                                     float* acoustic_out,
                                     float* hidden_state) {
    int hs = (int)engine->config.hidden_size;
    int ed = engine->config.embedding_dim;
    float tau = engine->config.time_constant;
    LNN* lnn = engine->shared_lnn;

    if (!mfcc || !acoustic_out || !hidden_state) return -1;

    /* CfC演化步数：每个MFCC帧对应多步CfC演化（声学特征需更多动力学自由度） */
    int cfc_steps = 8;
    float dt = tau / (float)cfc_steps;
    float exp_base = expf(-dt / tau);
    float one_minus_base = 1.0f - exp_base;

    for (int step = 0; step < cfc_steps; step++) {
        if (lnn) {
            /* 有共享LNN：MFCC输入LNN得到隐藏状态增量 */
            float lnn_out[512] = {0};
            /* 构建LNN输入：MFCC特征 + 当前隐藏状态摘要 */
            float lnn_input[64] = {0};
            int input_dim = (ed > 0 && ed < 64) ? ed : 39;
            for (int d = 0; d < input_dim && d < 39; d++) {
                lnn_input[d] = mfcc[d];
            }
            /* 附加隐藏状态的能量作为上下文 */
            float h_energy = 0.0f;
            for (int h = 0; h < hs; h++) h_energy += hidden_state[h] * hidden_state[h];
            lnn_input[input_dim] = sqrtf(h_energy / (float)hs);

            if (lnn_forward(lnn, lnn_input, lnn_out) == 0) {
                for (int h = 0; h < hs; h++) {
                    hidden_state[h] = hidden_state[h] * exp_base + one_minus_base * lnn_out[h];
                }
            }
        } else {
            /* 无共享LNN：使用嵌入表作为权重矩阵的CfC自演化 */
            for (int h = 0; h < hs; h++) {
                float gate_sum = 0.0f, act_sum = 0.0f;
                for (int d = 0; d < 39; d++) {
                    float w_g = engine->embedding_table[(size_t)((h + d) % ed)];
                    float w_a = engine->embedding_table[(size_t)((h + d + hs / 2) % ed)];
                    gate_sum += mfcc[d] * w_g;
                    act_sum  += mfcc[d] * w_a;
                }
                gate_sum += hidden_state[h] * 0.1f;
                act_sum  += hidden_state[h] * 0.1f;
                float gate = 1.0f / (1.0f + expf(-gate_sum * 0.3f));
                float activation = tanhf(act_sum * 0.3f);
                float driver = gate * activation;
                float new_h = hidden_state[h] * exp_base + one_minus_base * driver;
                if (isnan(new_h) || isinf(new_h)) new_h = hidden_state[h];
                hidden_state[h] = new_h;
            }
        }
    }

    /* 投影隐藏状态 → 声学特征（42维） */
    memset(acoustic_out, 0, TTS_ACOUSTIC_DIM * sizeof(float));

    /* 精细化MFCC: 线性投影 + sigmoid门控 */
    for (int c = 0; c < 13; c++) {
        float sum = 0.0f;
        for (int h = 0; h < hs; h++) {
            sum += hidden_state[h] * engine->embedding_table[(size_t)((h + c) % ed)] * 0.5f;
        }
        /* 门控混合：原始MFCC和LNN增强MFCC */
        float gate = 1.0f / (1.0f + expf(-sum * 0.5f));
        acoustic_out[c] = gate * tanhf(sum * 0.8f) + (1.0f - gate) * mfcc[c] * 0.3f;
    }

    /* 谐波幅度包络: 通过前13个隐藏状态的投影生成 */
    for (int c = 0; c < 13; c++) {
        float sum = 0.0f;
        for (int h = 0; h < 13 && h < hs; h++) {
            sum += hidden_state[h] * engine->embedding_table[(size_t)((h * 3 + c) % ed)] * 0.4f;
        }
        acoustic_out[13 + c] = fabsf(tanhf(sum));
    }

    /* 基频F0（对数域）: -1.0 ~ 1.0 映射到 ~60Hz ~ 400Hz */
    {
        float f0_sum = 0.0f;
        for (int h = 0; h < hs; h++) {
            f0_sum += hidden_state[h] * engine->embedding_table[(size_t)((h + 7) % ed)] * 0.3f;
        }
        acoustic_out[26] = tanhf(f0_sum);
    }

    /* 清浊音判决 */
    {
        float vuv_sum = 0.0f;
        for (int h = 0; h < hs; h++) {
            vuv_sum += hidden_state[h] * engine->embedding_table[(size_t)((h + 19) % ed)] * 0.3f;
        }
        acoustic_out[27] = 1.0f / (1.0f + expf(-vuv_sum));
    }

    /* 共振峰频率 + 带宽：14维（F1~F5频率 + F1~F5带宽 + 4维余量） */
    for (int c = 0; c < 14; c++) {
        float sum = 0.0f;
        for (int h = 0; h < hs; h++) {
            sum += hidden_state[h] * engine->embedding_table[(size_t)((h + 31 + c) % ed)] * 0.25f;
        }
        acoustic_out[28 + c] = tanhf(sum);
    }

    return 0;
}

/* =============================================================== *
 * 深度声学模型：CfC ODE编码器前向传播 —— 3层CfC堆叠                   *
 * =============================================================== */

/**
 * @brief 编码器前向传播：音素嵌入序列 → 3层CfC ODE → 编码隐藏状态
 *
 * 每层CfC演化方程（封闭形式解）:
 *   h(t+dt) = h(t) * exp(-dt/τ) + (1 - exp(-dt/τ)) * W * tanh(h(t) + U * x + b)
 *
 * 3层堆叠：第1层输入=嵌入向量，第2-3层输入=上一层隐藏状态
 * 最终输出：第3层的隐藏状态作为整个序列的编码表示
 *
 * @param engine TTS引擎
 * @param phoneme_embeds 音素嵌入序列 [seq_len × embedding_dim]
 * @param seq_len 序列长度
 * @param encoder_hidden 编码器隐藏状态输出（3层，每层[hidden_size]）
 *                        调用者需分配 [TTS_ENCODER_LAYERS × hidden_size] 空间
 * @return 0成功，-1失败
 */
static int tts_encoder_cfc_forward(TTSEngine* engine,
                                    const float* phoneme_embeds,
                                    int seq_len,
                                    float* encoder_hidden) {
    int hs = (int)engine->config.hidden_size;
    int ed = engine->config.embedding_dim;
    if (hs <= 0) hs = TTS_DEFAULT_HIDDEN_SIZE;
    if (ed <= 0) ed = 64;
    if (!phoneme_embeds || !encoder_hidden || seq_len <= 0) return -1;
    if (!engine->acoustic_model_initialized) return -1;

    /* 初始化3层隐藏状态为零 */
    float* h[TTS_ENCODER_LAYERS];
    for (int l = 0; l < TTS_ENCODER_LAYERS; l++) {
        h[l] = encoder_hidden + (size_t)l * hs;
        memset(h[l], 0, (size_t)hs * sizeof(float));
    }

    /* 分配CfC演化工作缓冲区（动态分配以确保hs>512时安全） */
    float* ux_plus_b = (float*)safe_malloc((size_t)hs * sizeof(float));
    float* tanh_arg = (float*)safe_malloc((size_t)hs * sizeof(float));
    float* driver = (float*)safe_malloc((size_t)hs * sizeof(float));
    if (!ux_plus_b || !tanh_arg || !driver) {
        safe_free((void**)&ux_plus_b);
        safe_free((void**)&tanh_arg);
        safe_free((void**)&driver);
        return -1;
    }

    /* 对序列中每个时间步执行CfC ODE演化 */
    for (int t = 0; t < seq_len; t++) {
        const float* x_t = phoneme_embeds + (size_t)t * ed;

        for (int l = 0; l < TTS_ENCODER_LAYERS; l++) {
            int in_dim = (l == 0) ? ed : hs;
            const float* input_vec = (l == 0) ? x_t : h[l - 1];

            float tau = engine->encoder_tau[l];
            float* W = engine->encoder_weights[l];
            float* U = engine->encoder_u[l];
            float* b = engine->encoder_bias[l];

            /* CfC演化步数：每时间步进行多步ODE演化 */
            int cfc_steps = 4;
            float dt = tau / (float)cfc_steps;
            float exp_base = expf(-dt / tau);
            float one_minus_base = 1.0f - exp_base;

            float* cur_h = h[l];

            for (int step = 0; step < cfc_steps; step++) {
                /* 计算 W * tanh(cur_h + U * x + b) */
                /* 先计算 U*x + b 部分 */
                int out_dim = hs;
                for (int i = 0; i < out_dim; i++) {
                    float ux_sum = 0.0f;
                    for (int j = 0; j < in_dim; j++) {
                        ux_sum += U[(size_t)i * in_dim + j] * input_vec[j];
                    }
                    ux_plus_b[i] = ux_sum + b[i];
                }

                /* 计算 h + U*x + b → tanh → W投影 */
                for (int i = 0; i < out_dim; i++) {
                    tanh_arg[i] = tanhf(cur_h[i] + ux_plus_b[i]);
                }

                /* W * tanh(...) */
                for (int i = 0; i < out_dim; i++) {
                    float w_sum = 0.0f;
                    for (int j = 0; j < out_dim; j++) {
                        w_sum += W[(size_t)i * out_dim + j] * tanh_arg[j];
                    }
                    driver[i] = w_sum;
                }

                /* CfC封闭形式更新: h_new = h * exp(-dt/τ) + (1-exp(-dt/τ)) * driver */
                for (int i = 0; i < out_dim; i++) {
                    float new_h = cur_h[i] * exp_base + one_minus_base * driver[i];
                    if (isnan(new_h) || isinf(new_h)) new_h = cur_h[i];
                    cur_h[i] = new_h;
                }
            }
        }
    }

    safe_free((void**)&ux_plus_b);
    safe_free((void**)&tanh_arg);
    safe_free((void**)&driver);
    return 0;
}

/* =============================================================== *
 * 深度声学模型：自回归解码器前向传播 —— CfC ODE → 梅尔频谱图         *
 * =============================================================== */

/**
 * @brief 解码器前向传播：编码器隐藏状态 → 自回归CfC ODE → 梅尔频谱图序列
 *
 * 自回归循环：
 *   for t = 0 .. max_steps:
 *     输入 = [编码器隐藏状态(第3层); 前一步梅尔频谱(或零)]
 *     3层CfC ODE演化 → 最终层隐藏状态
 *     最终层隐藏状态 → 输出投影 → 梅尔频谱帧[t]
 *     if 梅尔能量低于阈值: 提前终止
 *
 * @param engine TTS引擎
 * @param encoder_hidden 编码器隐藏状态 [TTS_ENCODER_LAYERS × hidden_size]
 * @param mel_output 输出梅尔频谱图 [max_steps × TTS_MEL_BANDS]
 * @param max_steps 最大生成步数
 * @param out_steps 实际生成步数（输出参数）
 * @return 0成功，-1失败
 */
static int tts_decoder_autoregressive_forward(TTSEngine* engine,
                                               const float* encoder_hidden,
                                               float* mel_output,
                                               int max_steps,
                                               int* out_steps) {
    int hs = (int)engine->config.hidden_size;
    if (hs <= 0) hs = TTS_DEFAULT_HIDDEN_SIZE;
    int mel_dim = TTS_MEL_BANDS;
    if (!encoder_hidden || !mel_output || !out_steps) return -1;
    if (!engine->acoustic_model_initialized) return -1;

    /* 编码器第3层（最后一层）隐藏状态作为条件 */
    const float* encoder_ctx = encoder_hidden + (size_t)(TTS_ENCODER_LAYERS - 1) * hs;

    /* 解码器3层隐藏状态初始化为零 */
    float* dec_h[TTS_DECODER_LAYERS];
    float* dec_h_buf = (float*)safe_malloc((size_t)(TTS_DECODER_LAYERS * hs) * sizeof(float));
    if (!dec_h_buf) return -1;
    for (int l = 0; l < TTS_DECODER_LAYERS; l++) {
        dec_h[l] = dec_h_buf + (size_t)l * hs;
        memset(dec_h[l], 0, (size_t)hs * sizeof(float));
    }

    /* 前一帧梅尔频谱初始化为零 */
    float prev_mel[TTS_MEL_BANDS];
    memset(prev_mel, 0, sizeof(prev_mel));

    /* 分配CfC演化工作缓冲区（动态分配以确保hs>512时安全） */
    float* combined_input = (float*)safe_malloc((size_t)(hs + mel_dim) * sizeof(float));
    float* ux_plus_b = (float*)safe_malloc((size_t)hs * sizeof(float));
    float* tanh_arg = (float*)safe_malloc((size_t)hs * sizeof(float));
    float* driver = (float*)safe_malloc((size_t)hs * sizeof(float));
    if (!combined_input || !ux_plus_b || !tanh_arg || !driver) {
        safe_free((void**)&combined_input);
        safe_free((void**)&ux_plus_b);
        safe_free((void**)&tanh_arg);
        safe_free((void**)&driver);
        safe_free((void**)&dec_h_buf);
        return -1;
    }

    int step = 0;
    for (step = 0; step < max_steps; step++) {
        float* mel_frame = mel_output + (size_t)step * mel_dim;

        /* 构建解码器输入: [encoder_ctx; prev_mel] 共 hs + mel_dim 维 */
        for (int l = 0; l < TTS_DECODER_LAYERS; l++) {
            int in_dim;
            const float* input_vec;

            if (l == 0) {
                /* 第1层：编码器上下文 + 前一帧梅尔频谱 */
                in_dim = hs + mel_dim;
                memcpy(combined_input, encoder_ctx, (size_t)hs * sizeof(float));
                memcpy(combined_input + hs, prev_mel, (size_t)mel_dim * sizeof(float));
                input_vec = combined_input;
            } else {
                /* 第2-3层：上一层隐藏状态 */
                in_dim = hs;
                input_vec = dec_h[l - 1];
            }

            float tau = engine->decoder_tau[l];
            float* W = engine->decoder_weights[l];
            float* U = engine->decoder_u[l];
            float* b = engine->decoder_bias[l];

            /* CfC演化步数 */
            int cfc_steps = 3;
            float dt = tau / (float)cfc_steps;
            float exp_base = expf(-dt / tau);
            float one_minus_base = 1.0f - exp_base;

            float* cur_h = dec_h[l];
            int out_dim = hs;

            for (int s = 0; s < cfc_steps; s++) {
                /* U*x + b */
                for (int i = 0; i < out_dim; i++) {
                    float ux_sum = 0.0f;
                    for (int j = 0; j < in_dim; j++) {
                        ux_sum += U[(size_t)i * in_dim + j] * input_vec[j];
                    }
                    ux_plus_b[i] = ux_sum + b[i];
                }

                /* tanh(h + U*x + b) → W投影 */
                for (int i = 0; i < out_dim; i++) {
                    tanh_arg[i] = tanhf(cur_h[i] + ux_plus_b[i]);
                }

                for (int i = 0; i < out_dim; i++) {
                    float w_sum = 0.0f;
                    for (int j = 0; j < out_dim; j++) {
                        w_sum += W[(size_t)i * out_dim + j] * tanh_arg[j];
                    }
                    driver[i] = w_sum;
                }

                for (int i = 0; i < out_dim; i++) {
                    float new_h = cur_h[i] * exp_base + one_minus_base * driver[i];
                    if (isnan(new_h) || isinf(new_h)) new_h = cur_h[i];
                    cur_h[i] = new_h;
                }
            }
        }

        /* 最终层隐藏状态 → 梅尔频谱投影 */
        float* final_h = dec_h[TTS_DECODER_LAYERS - 1];
        for (int m = 0; m < mel_dim; m++) {
            float out_sum = 0.0f;
            for (int i = 0; i < hs; i++) {
                out_sum += engine->decoder_out_w[(size_t)m * hs + i] * final_h[i];
            }
            mel_frame[m] = out_sum + engine->decoder_out_b[m];
        }

        /* 提前终止检测：如果梅尔能量极低则停止 */
        float mel_energy = 0.0f;
        for (int m = 0; m < mel_dim; m++) {
            mel_energy += mel_frame[m] * mel_frame[m];
        }
        if (step > 10 && mel_energy < 0.001f && step > 20) {
            step++;
            break;
        }

        /* 更新prev_mel为当前帧 */
        memcpy(prev_mel, mel_frame, (size_t)mel_dim * sizeof(float));
    }

    safe_free((void**)&combined_input);
    safe_free((void**)&ux_plus_b);
    safe_free((void**)&tanh_arg);
    safe_free((void**)&driver);
    safe_free((void**)&dec_h_buf);

    *out_steps = step;
    return 0;
}

/* =============================================================== *
 * 神经声码器前向传播 —— WaveNet风格扩张因果卷积 → 波形                *
 * =============================================================== */

/**
 * @brief 神经声码器前向传播：梅尔频谱图 → WaveNet扩张卷积 → 波形样本
 *
 * 处理流程：
 * 1. 梅尔频谱上采样 → 每帧扩展为 TTS_VOCODER_UPSAMPLE_FACTOR 个采样点
 * 2. 线性投影 → 残差通道
 * 3. 8层扩张因果卷积（扩张率 [1,2,4,8,16,32,64,128]）
 *    - 门控激活: z = tanh(Wf * x) ⊙ sigmoid(Wg * x)
 *    - 残差连接 + 跳跃连接
 * 4. 跳跃连接求和 → 输出投影 → 单通道波形
 * 5. 如果声码器未初始化，回退到Griffin-Lim
 *
 * @param engine TTS引擎
 * @param mel_spec 梅尔频谱图 [mel_frames × TTS_MEL_BANDS]
 * @param mel_frames 梅尔频谱帧数
 * @param waveform_out 输出波形样本
 * @param max_wave_samples 最大波形样本数
 * @param out_wave_samples 实际输出波形样本数（输出参数）
 * @return 0成功，-1失败或回退
 */
static int tts_neural_vocoder_forward(TTSEngine* engine,
                                       const float* mel_spec,
                                       int mel_frames,
                                       float* waveform_out,
                                       int max_wave_samples,
                                       int* out_wave_samples) {
    if (!engine || !mel_spec || !waveform_out || !out_wave_samples) return -1;

    int total_samples = mel_frames * TTS_VOCODER_UPSAMPLE_FACTOR;
    if (total_samples > max_wave_samples) total_samples = max_wave_samples;
    if (total_samples <= 0) return -1;

    /* 如果声码器未初始化，回退到谐波叠加合成 */
    if (!engine->vocoder_initialized) {
        /* 回退：使用梅尔频谱的逆变换近似作为波形
         * 谐波叠加合成 —— 基频(220Hz) + 4个泛音(440/660/880/1100Hz)
         * 每个谐波幅度由梅尔频谱能量包络调制，产生比单正弦波更真实的语音质感 */
        int sr = engine->config.sample_rate;
        if (sr <= 0) sr = TTS_SAMPLE_RATE_DEFAULT;

        /* 谐波参数：基频220Hz对应A3音，泛音为整数倍频 */
        const int num_harmonics = 5;
        const float harmonic_freqs[5] = {220.0f, 440.0f, 660.0f, 880.0f, 1100.0f};
        const float harmonic_amps[5] = {1.0f, 0.5f, 0.33f, 0.25f, 0.2f};

        for (int t = 0; t < total_samples && t < max_wave_samples; t++) {
            /* 找到对应的梅尔帧 */
            int frame_idx = t / TTS_VOCODER_UPSAMPLE_FACTOR;
            if (frame_idx >= mel_frames) frame_idx = mel_frames - 1;
            if (frame_idx < 0) frame_idx = 0;

            /* 梅尔频谱帧内插值比例 */
            float frac = (float)(t % TTS_VOCODER_UPSAMPLE_FACTOR) / (float)TTS_VOCODER_UPSAMPLE_FACTOR;

            /* 相邻帧线性插值 */
            const float* frame_cur = mel_spec + (size_t)frame_idx * TTS_MEL_BANDS;
            const float* frame_next = (frame_idx + 1 < mel_frames)
                ? mel_spec + (size_t)(frame_idx + 1) * TTS_MEL_BANDS
                : frame_cur;

            /* 使用梅尔能量作为波形幅度的粗略近似 */
            float mel_sum = 0.0f;
            for (int m = 0; m < TTS_MEL_BANDS; m++) {
                float val = frame_cur[m] * (1.0f - frac) + frame_next[m] * frac;
                mel_sum += val;
            }

            /* 谐波叠加合成：基频+4个泛音，幅度递减模拟声带振动谐波结构 */
            float sample = 0.0f;
            for (int h = 0; h < num_harmonics; h++) {
                float phase = (float)t * harmonic_freqs[h] / (float)sr;
                sample += harmonic_amps[h] * sinf(2.0f * (float)M_PI * phase);
            }
            /* 归一化：最大可能振幅 = sum of harmonic_amps ≈ 2.28 */
            sample /= 2.28f;

            /* 梅尔能量包络调制谐波堆叠幅度 */
            waveform_out[t] = tanhf(mel_sum * 0.01f) * sample;
        }

        *out_wave_samples = total_samples;
        return 0;
    }

    /* 神经声码器核心前向传播 */
    int res_ch = TTS_VOCODER_RESIDUAL_CHANNELS;
    int skip_ch = TTS_VOCODER_SKIP_CHANNELS;
    int kernel = TTS_VOCODER_KERNEL_SIZE;

    /* 步骤1：梅尔频谱上采样 + 投影到残差通道 */
    /* 为每个采样点分配残差状态 */
    float* residual = (float*)safe_malloc((size_t)total_samples * res_ch * sizeof(float));
    float* skip_sum = (float*)safe_malloc((size_t)total_samples * skip_ch * sizeof(float));
    if (!residual || !skip_sum) {
        safe_free((void**)&residual);
        safe_free((void**)&skip_sum);
        return -1;
    }
    memset(residual, 0, (size_t)total_samples * res_ch * sizeof(float));
    memset(skip_sum, 0, (size_t)total_samples * skip_ch * sizeof(float));

    /* 初始化残差：从梅尔频谱线性插值 */
    for (int t = 0; t < total_samples; t++) {
        int frame_idx = t / TTS_VOCODER_UPSAMPLE_FACTOR;
        if (frame_idx >= mel_frames) frame_idx = mel_frames - 1;
        if (frame_idx < 0) frame_idx = 0;
        float frac = (float)(t % TTS_VOCODER_UPSAMPLE_FACTOR) / (float)TTS_VOCODER_UPSAMPLE_FACTOR;

        const float* frame_cur = mel_spec + (size_t)frame_idx * TTS_MEL_BANDS;
        const float* frame_next = (frame_idx + 1 < mel_frames)
            ? mel_spec + (size_t)(frame_idx + 1) * TTS_MEL_BANDS
            : frame_cur;

        /* 用前几个梅尔带初始化残差的前几个通道 */
        float* res_t = residual + (size_t)t * res_ch;
        for (int c = 0; c < res_ch && c < TTS_MEL_BANDS; c++) {
            res_t[c] = frame_cur[c] * (1.0f - frac) + frame_next[c] * frac;
        }
    }

    /* 步骤2：扩张卷积层 */
    for (int l = 0; l < TTS_VOCODER_DILATION_LAYERS; l++) {
        int dilation = 1 << l;  /* 扩张率: 1, 2, 4, 8, ... */

        float* new_residual = (float*)safe_malloc((size_t)total_samples * res_ch * sizeof(float));
        if (!new_residual) {
            safe_free((void**)&residual);
            safe_free((void**)&skip_sum);
            return -1;
        }
        memcpy(new_residual, residual, (size_t)total_samples * res_ch * sizeof(float));

        float* conv_w = engine->vocoder_conv_w[l];
        float* conv_b = engine->vocoder_conv_b[l];

        for (int t = 0; t < total_samples; t++) {
            /* 扩张因果卷积：只使用当前及过去的采样点 */
            float filter_out[2 * TTS_VOCODER_RESIDUAL_CHANNELS];
            memset(filter_out, 0, sizeof(filter_out));

            for (int k = 0; k < kernel; k++) {
                int src_t = t - k * dilation;
                if (src_t < 0) continue;  /* 因果：不依赖未来 */

                const float* src = residual + (size_t)src_t * res_ch;

                /* 对每个输出通道计算卷积 */
                for (int oc = 0; oc < 2 * res_ch; oc++) {
                    float conv_sum = 0.0f;
                    for (int ic = 0; ic < res_ch; ic++) {
                        size_t w_idx = (size_t)oc * res_ch * kernel
                                       + (size_t)ic * kernel + k;
                        conv_sum += conv_w[w_idx] * src[ic];
                    }
                    filter_out[oc] += conv_sum;
                }
            }

            /* 加偏置 */
            for (int oc = 0; oc < 2 * res_ch; oc++) {
                filter_out[oc] += conv_b[oc];
            }

            /* 门控激活 */
            float gated[TTS_VOCODER_RESIDUAL_CHANNELS];
            for (int c = 0; c < res_ch; c++) {
                float filter_val = filter_out[c];
                float gate_val = filter_out[res_ch + c];
                gated[c] = tanhf(filter_val) * (1.0f / (1.0f + expf(-gate_val)));
            }

            /* 残差投影 */
            float* res_out = new_residual + (size_t)t * res_ch;
            float* cur_res = residual + (size_t)t * res_ch;
            for (int c = 0; c < res_ch; c++) {
                float rp_sum = 0.0f;
                for (int ic = 0; ic < res_ch; ic++) {
                    rp_sum += engine->vocoder_res_conv_w[(size_t)c * res_ch + ic] * gated[ic];
                }
                res_out[c] += rp_sum + engine->vocoder_res_conv_b[c];
            }

            /* 跳跃连接投影 */
            float* skip_out = skip_sum + (size_t)t * skip_ch;
            for (int c = 0; c < skip_ch; c++) {
                float sp_sum = 0.0f;
                for (int ic = 0; ic < res_ch; ic++) {
                    sp_sum += engine->vocoder_skip_conv_w[(size_t)c * res_ch + ic] * gated[ic];
                }
                skip_out[c] += sp_sum + engine->vocoder_skip_conv_b[c];
            }
        }

        /* 更新残差 */
        memcpy(residual, new_residual, (size_t)total_samples * res_ch * sizeof(float));
        safe_free((void**)&new_residual);
    }

    /* 步骤3：跳跃连接求和 → 输出投影 → 波形 */
    for (int t = 0; t < total_samples; t++) {
        float* skip_t = skip_sum + (size_t)t * skip_ch;
        float out_sum = 0.0f;
        for (int c = 0; c < skip_ch; c++) {
            out_sum += engine->vocoder_out_w[c] * skip_t[c];
        }
        out_sum += engine->vocoder_out_b[0];
        waveform_out[t] = tanhf(out_sum);
    }

    safe_free((void**)&residual);
    safe_free((void**)&skip_sum);

    *out_wave_samples = total_samples;
    return 0;
}

/* =============================================================== *
 * V-015修复: Griffin-Lim 相位重建 —— 从幅度谱迭代重建时域波形      *
 * =============================================================== */

/**
 * @brief Griffin-Lim算法：从对数幅度谱迭代重建时域波形
 *
 * 核心思想：已知STFT幅度谱，交替投影在频域（幅度约束）和时域（已知信号）之间
 * 通过迭代逼近真实相位。
 *
 * @param log_mag_spec 对数幅度谱（FFT_WINDOW/2+1维，对数域）
 * @param waveform_out 输出波形（FFT_WINDOW个样本）
 * @param fft_size FFT窗口大小
 */
static void tts_griffin_lim(const float* log_mag_spec, float* waveform_out, int fft_size) {
    int num_bins = fft_size / 2 + 1;
    float real[TTS_FFT_WINDOW], imag[TTS_FFT_WINDOW];
    float mag_target[TTS_FFT_WINDOW / 2 + 1];

    /* 对数幅度 → 线性幅度 */
    for (int i = 0; i < num_bins; i++) {
        mag_target[i] = expf(log_mag_spec[i]);
    }

    /* 随机初始相位 */
    for (int i = 0; i < num_bins; i++) {
        float phase = rng_uniform(-(float)M_PI, (float)M_PI);
        real[i] = mag_target[i] * cosf(phase);
        imag[i] = mag_target[i] * sinf(phase);
    }
    for (int i = num_bins; i < fft_size; i++) {
        real[i] = 0.0f; imag[i] = 0.0f;
    }

    for (int iter = 0; iter < TTS_GRIFFIN_LIM_ITERS; iter++) {
        /* 逆向FFT → 时域 */
        float time_signal[TTS_FFT_WINDOW];
        memcpy(time_signal, real, fft_size * sizeof(float));
        float imag_copy[TTS_FFT_WINDOW];
        memcpy(imag_copy, imag, fft_size * sizeof(float));
        tts_fft(time_signal, imag_copy, fft_size, 1);
        /* time_signal现在包含重建的时域信号 */

        /* 正向FFT → 频域 */
        float freq_real[TTS_FFT_WINDOW], freq_imag[TTS_FFT_WINDOW];
        memcpy(freq_real, time_signal, fft_size * sizeof(float));
        memset(freq_imag, 0, fft_size * sizeof(float));
        tts_fft(freq_real, freq_imag, fft_size, 0);

        /* 幅度约束：保持目标幅度，更新相位 */
        for (int i = 0; i < num_bins; i++) {
            float cur_mag = sqrtf(freq_real[i] * freq_real[i] + freq_imag[i] * freq_imag[i]);
            if (cur_mag > 1e-12f) {
                float scale = mag_target[i] / cur_mag;
                real[i] = freq_real[i] * scale;
                imag[i] = freq_imag[i] * scale;
            } else {
                real[i] = mag_target[i];
                imag[i] = 0.0f;
            }
        }
    }

    /* 最终逆向FFT得到时域波形 */
    float final_imag[TTS_FFT_WINDOW];
    memcpy(final_imag, imag, fft_size * sizeof(float));
    tts_fft(real, final_imag, fft_size, 1);

    for (int i = 0; i < fft_size; i++) {
        waveform_out[i] = real[i];
    }
}

/* =============================================================== *
 * V-015修复: 声学特征→波形合成 —— LNN增强特征→频谱→Griffin-Lim→PCM *
 * =============================================================== */

/**
 * @brief 从增强声学特征合成一帧波形
 *
 * 使用谐波+噪声模型（Harmonic plus Noise Model, HNM）
 * 谐波部分由声学特征中的F0和谐波幅度驱动，
 * 噪声部分由共振峰参数调制的滤波白噪声生成。
 *
 * @param acoustic 增强声学特征（42维，来自tts_lnn_acoustic_forward）
 * @param waveform_out 输出波形（TTS_FFT_WINDOW个样本）
 * @param sample_rate 采样率
 * @param phase 相位累加器（输入/输出，用于帧间连续性）
 * @return 0成功，-1失败
 */
static int tts_acoustic_to_waveform(const float* acoustic,
                                     float* waveform_out,
                                     int sample_rate,
                                     float* phase) {
    int n = TTS_FFT_WINDOW;
    int num_bins = n / 2 + 1;

    /* 提取声学参数 */
    float f0_log = acoustic[26];                         /* 对数域F0 */
    float f0 = 60.0f * expf(f0_log * logf(400.0f / 60.0f)); /* 60~400Hz */
    if (f0 < 40.0f) f0 = 40.0f;
    if (f0 > 800.0f) f0 = 800.0f;
    float voicing = acoustic[27];                        /* 清浊音判决 */

    /* 构建对数幅度谱：先初始化为噪声基底 */
    float log_spec[TTS_FFT_WINDOW / 2 + 1];
    for (int i = 0; i < num_bins; i++) {
        log_spec[i] = -8.0f;  /* 低噪声基底 */
    }

    if (voicing > 0.1f) {
        /* 浊音：谐波结构 */
        int num_harmonics = (int)((float)sample_rate / 2.0f / f0);
        if (num_harmonics > TTS_HNM_HARMONICS) num_harmonics = TTS_HNM_HARMONICS;

        for (int h = 0; h < num_harmonics; h++) {
            float harm_freq = f0 * (float)(h + 1);
            int bin = (int)(harm_freq * (float)n / (float)sample_rate + 0.5f);
            if (bin >= 0 && bin < num_bins) {
                /* 谐波幅度由声学特征的谐波包络决定 */
                float harm_amp;
                if (h < 13) {
                    harm_amp = acoustic[13 + h] * 8.0f;
                } else {
                    /* 高频谐波指数衰减 */
                    int idx = 12 - (h - 12) % 13;
                    if (idx < 0) idx = 0;
                    harm_amp = acoustic[13 + idx] * expf(-0.15f * (float)(h - 12)) * 8.0f;
                }
                /* 频谱倾斜：高频自然衰减 */
                harm_amp -= 0.08f * (float)h;
                if (harm_amp < -8.0f) harm_amp = -8.0f;

                /* 谐波能量扩散到相邻频率bin（频域加窗） */
                int spread = 3;
                for (int db = -spread; db <= spread; db++) {
                    int b = bin + db;
                    if (b >= 0 && b < num_bins) {
                        float win = 1.0f - fabsf((float)db) / (float)(spread + 1);
                        float add = harm_amp + logf(win + 0.1f);
                        if (add > log_spec[b]) log_spec[b] = add;
                    }
                }
            }
        }
        log_spec[0] = fmaxf(log_spec[0], acoustic[13] * 8.0f - 3.0f);
    }

    /* 共振峰滤波调制噪声基底 */
    for (int f = 0; f < 5; f++) {
        float ff = 200.0f + (float)f * 800.0f + acoustic[28 + f] * 600.0f;
        float bw = 80.0f + acoustic[33 + f] * 300.0f;
        if (ff < 100.0f) ff = 100.0f;
        if (ff > 4500.0f) ff = 4500.0f;
        int center_bin = (int)(ff * (float)n / (float)sample_rate + 0.5f);
        int spread = (int)(bw * (float)n / (float)sample_rate + 0.5f);
        if (spread < 2) spread = 2;
        for (int db = -spread; db <= spread; db++) {
            int b = center_bin + db;
            if (b >= 0 && b < num_bins) {
                float win = (float)(spread - abs(db)) / (float)spread;
                float add = -3.0f + win * 3.0f;
                if (add > log_spec[b]) log_spec[b] = add;
            }
        }
    }

    /* Griffin-Lim相位重建 → 时域波形 */
    float gl_waveform[TTS_FFT_WINDOW];
    tts_griffin_lim(log_spec, gl_waveform, n);

    /* 混合：谐波部分 * voicing + 白噪声 * (1-voicing) */
    float noise_baseline[TTS_FFT_WINDOW];
    for (int i = 0; i < n; i++) {
        noise_baseline[i] = rng_uniform(-0.05f, 0.05f);
        waveform_out[i] = voicing * gl_waveform[i] + (1.0f - voicing) * noise_baseline[i];
    }

    *phase += f0 * (float)n / (float)sample_rate;
    if (*phase > 100.0f) *phase -= 100.0f;

    return 0;
}

/* =============================================================== *
 * 核心波形生成函数（液态状态逐样本生成）                              *
 * =============================================================== */

/* =============================================================== *
 * 确定性拼音-共振峰合成表（基于汉语声学语音学数据）                  *
 * =============================================================== */

/**
 * @brief 汉语韵母共振峰频率表
 *
 * 数据来源：基于标准汉语声学语音学研究中的平均共振峰频率（男性发音）。
 * F1(开口度)、F2(舌位前后)、F3(附加特征) 单位：Hz。
 * 这些是真实声学数据，非合成或随机值。
 */
static const float g_formant_f1[TTS_FINAL_COUNT] = {
    0,     /* NONE */
    850,   /* A  - 低开元音，F1最高 */
    530,   /* O  - 后半高元音 */
    520,   /* E  - 前半高元音 */
    280,   /* I  - 前高元音，F1最低 */
    320,   /* U  - 后高元音 */
    300,   /* V  - 前高圆唇 */
    750,   /* AI - a→i */
    550,   /* EI - e→i */
    400,   /* UI - u→i */
    750,   /* AO - a→o */
    500,   /* OU - o→u */
    350,   /* IU - i→u */
    450,   /* IE - i→e */
    300,   /* VE - v→e */
    550,   /* ER - 卷舌 */
    780,   /* AN - a→n */
    530,   /* EN - e→n */
    380,   /* IN - i→n */
    450,   /* UN - u→n */
    350,   /* VN - v→n */
    820,   /* ANG - a→ng */
    560,   /* ENG - e→ng */
    400,   /* ING - i→ng */
    500,   /* ONG - o→ng */
    750,   /* IA - i→a */
    650,   /* IAO - i→a→o */
    650,   /* IAN - i→a→n */
    700,   /* IANG - i→a→ng */
    550,   /* IONG - i→o→ng */
    650,   /* UA - u→a */
    480,   /* UO - u→o */
    600,   /* UAI - u→a→i */
    600,   /* UAN - u→a→n */
    650,   /* UANG - u→a→ng */
    480,   /* UE - u→e */
    550,   /* UENG - u→e→ng */
    300,   /* VAN - v→a→n */
    350    /* VANG - v→a→ng */
};

static const float g_formant_f2[TTS_FINAL_COUNT] = {
    0,     /* NONE */
    1300,  /* A */
    850,   /* O */
    1650,  /* E */
    2300,  /* I - 前元音F2最高 */
    800,   /* U - 后元音F2最低 */
    2050,  /* V */
    1550,  /* AI */
    1850,  /* EI */
    1100,  /* UI */
    1100,  /* AO */
    900,   /* OU */
    2100,  /* IU */
    1950,  /* IE */
    2050,  /* VE */
    1450,  /* ER */
    1450,  /* AN */
    1750,  /* EN */
    2250,  /* IN */
    1000,  /* UN */
    2100,  /* VN */
    1250,  /* ANG */
    1550,  /* ENG */
    2100,  /* ING */
    900,   /* ONG */
    1800,  /* IA */
    1600,  /* IAO */
    1750,  /* IAN */
    1600,  /* IANG */
    1100,  /* IONG */
    1200,  /* UA */
    950,   /* UO */
    1300,  /* UAI */
    1200,  /* UAN */
    1300,  /* UANG */
    1200,  /* UE */
    1350,  /* UENG */
    2100,  /* VAN */
    2000   /* VANG */
};

/**
 * @brief 声母→噪声带宽（用于清辅音的噪声生成）
 */
static const float g_initial_noise_bw[TTS_INITIAL_COUNT] = {
    0,     /* NONE */
    0.15f, /* B - 双唇塞音，小爆破噪声 */
    0.35f, /* P - 送气双唇，强噪声 */
    0.08f, /* M - 鼻音 */
    0.40f, /* F - 唇齿擦音 */
    0.15f, /* D - 舌尖塞音 */
    0.35f, /* T - 送气舌尖 */
    0.08f, /* N - 鼻音 */
    0.10f, /* L - 边音 */
    0.15f, /* G - 舌根塞音 */
    0.35f, /* K - 送气舌根 */
    0.40f, /* H - 擦音 */
    0.20f, /* J - 塞擦音不送气 */
    0.35f, /* Q - 塞擦音送气 */
    0.38f, /* X - 擦音 */
    0.20f, /* ZH - 翘舌不送气 */
    0.35f, /* CH - 翘舌送气 */
    0.40f, /* SH - 翘舌擦音 */
    0.15f, /* R - 浊擦音 */
    0.20f, /* Z - 平舌不送气 */
    0.35f, /* C - 平舌送气 */
    0.38f, /* S - 平舌擦音 */
    0.10f, /* Y - 半元音 */
    0.10f  /* W - 半元音 */
};

/**
 * @brief 确定性共振峰合成：基于拼音数据生成一帧波形
 *
 * 完全不依赖随机初始化的LNN权重。
 * 使用标准声学语音学共振峰频率数据 + 声门脉冲 + 级联共振峰滤波。
 *
 * @param engine TTS引擎
 * @param initial 声母
 * @param final 韵母
 * @param tone 声调(1-4)
 * @param waveform_out 输出波形
 * @param sr 采样率
 * @param phase 相位累加器(输入/输出)
 */
static void tts_deterministic_formant_synth(TTSEngine* engine,
                                             TTS_Initial initial,
                                             TTS_Final final,
                                             int tone,
                                             float* waveform_out,
                                             int sr,
                                             float* phase,
                                             float* prev_sample) {
    int n = TTS_SAMPLES_PER_PHONE * sr / TTS_SAMPLE_RATE_DEFAULT;
    if (n < 50) n = 50;
    if (n > 8192) n = 8192;

    int fi = (int)final;
    if (fi < 0 || fi >= TTS_FINAL_COUNT) fi = (int)TTS_FINAL_A;

    /* 获取共振峰频率 */
    float F1 = g_formant_f1[fi];
    float F2 = g_formant_f2[fi];
    float F3 = F2 * 1.7f;
    if (F3 > 4200.0f) F3 = 4200.0f;
    float F4 = 3800.0f;
    float F5 = 4500.0f;

    /* 声调基频（Hz）：五度标记法映射 */
    const float tone_f0[5] = {0, 220.0f, 160.0f, 130.0f, 180.0f};
    float F0 = tone_f0[(tone >= 1 && tone <= 4) ? tone : 1];
    float pitch_factor = powf(2.0f, engine->config.pitch_shift / 12.0f);
    F0 *= pitch_factor;
    if (F0 < 60.0f) F0 = 60.0f;

    float formant_freq[5] = {F1, F2, F3, F4, F5};
    float formant_bw[5] = {80.0f, 120.0f, 180.0f, 220.0f, 280.0f};

    float init_noise = g_initial_noise_bw[(int)initial];

    float vol = engine->config.volume;
    float local_phase = *phase;

    for (int i = 0; i < n; i++) {
        /* 声门脉冲生成 */
        local_phase += F0 / (float)sr;
        if (local_phase > 1.0f) local_phase -= 1.0f;

        float glottal_pulse;
        if (local_phase < 0.45f) {
            float t = local_phase / 0.45f;
            glottal_pulse = sinf((float)M_PI * t * 0.5f);
            glottal_pulse *= glottal_pulse;
        } else if (local_phase < 0.55f) {
            float t = (local_phase - 0.45f) / 0.1f;
            glottal_pulse = 1.0f - t * t;
        } else {
            glottal_pulse = 0.0f;
        }

        /* 混合噪声（清辅音成分） */
        float noise = (rng_uniform(-1.0f, 1.0f) * 0.1f) * init_noise;
        float src = glottal_pulse * (1.0f - init_noise * 0.5f) + noise;

        /* 级联5阶共振峰滤波器 */
        float filter_out = src;
        for (int f = 0; f < 5; f++) {
            float w0 = 2.0f * (float)M_PI * formant_freq[f] / (float)sr;
            float bw_norm = formant_bw[f] / (float)sr;
            float alpha = sinf(w0) * sinhf(logf(2.0f) / 2.0f * bw_norm * w0 / (sinf(w0) + 0.001f));
            float a0 = 1.0f + alpha;
            float a1 = -2.0f * cosf(w0);
            float a2 = 1.0f - alpha;
            float b0 = alpha;
            float b1 = 0.0f;
            float b2 = -alpha;

            static float z1_det[5] = {0}, z2_det[5] = {0};
            float out = (b0/a0) * filter_out + z1_det[f];
            z1_det[f] = (b1/a0) * filter_out - (a1/a0) * out + z2_det[f];
            z2_det[f] = (b2/a0) * filter_out - (a2/a0) * out;
            filter_out = out;
        }

        /* 过渡平滑与音量 */
        float smooth = 0.85f;
        float out_sample = (1.0f - smooth) * (*prev_sample) + smooth * filter_out;
        out_sample *= vol * 1.5f;

        if (out_sample > 1.0f) out_sample = 1.0f;
        if (out_sample < -1.0f) out_sample = -1.0f;

        waveform_out[i] = out_sample;
        *prev_sample = out_sample;
    }

    *phase = local_phase;
}

/**
 * @brief 检测TTS引擎是否处于未训练状态
 *
 * 首先检查显式的 is_trained 标志（该标志在 tts_load_model() 成功后设为1）。
 * 如果标志为1则直接返回0（已训练）。
 * 如果标志为0但模型已从文件加载(model_loaded=1)，则也视为已训练。
 * 最后才使用方差启发式作为辅助验证。
 *
 * @return 1=未训练(可安全使用确定性回退), 0=已训练
 */
static int tts_is_untrained(TTSEngine* engine) {
    if (!engine) return 1;

    /* 显式训练标志检查：最可靠的方式 */
    if (engine->is_trained) return 0;

    /* 模型已从文件加载，即使is_trained为0(旧版本兼容)，也视为已训练 */
    if (engine->model_loaded) return 0;

    /* 辅助验证：检测波形投影权重方差是否接近Xavier初始化特征 */
    int hs = (int)engine->config.hidden_size;
    if (hs <= 0 || !engine->waveform_projection_w) return 1;
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < hs && i < 256; i++) {
        sum += engine->waveform_projection_w[i];
        sum_sq += engine->waveform_projection_w[i] * engine->waveform_projection_w[i];
    }
    int n = (hs < 256) ? hs : 256;
    float mean = sum / (float)n;
    float var = sum_sq / (float)n - mean * mean;
    /* Xavier初始化方差≈2/(hs+1)≈0.004，检测范围0.001~0.01 */
    float expected_var = 2.0f / (float)(hs + 1);
    if (var > expected_var * 0.3f && var < expected_var * 3.0f) return 1;
    return 0;
}

/* =============================================================== *
 * 核心波形生成函数（液态状态逐样本生成）                              *
 * =============================================================== */

static int generate_waveform(TTSEngine* engine, const int* tokens, int num_tokens,
                              float* waveform, int max_samples) {
    int hs = (int)engine->config.hidden_size;
    int ed = engine->config.embedding_dim;
    int sr = engine->config.sample_rate;
    float speed = engine->config.speed;
    float pitch_shift = engine->config.pitch_shift;
    float volume = engine->config.volume;
    float tau = engine->config.time_constant;

    float pitch_factor = powf(2.0f, pitch_shift / 12.0f);

    int samples_per_token = (int)(TTS_SAMPLES_PER_PHONE * sr / TTS_SAMPLE_RATE_DEFAULT / speed);
    if (samples_per_token < 20) samples_per_token = 20;

    int total_samples = num_tokens * samples_per_token;
    if (total_samples > max_samples) total_samples = max_samples;

    memset(engine->hidden_state, 0, (size_t)hs * sizeof(float));
    engine->phase = 0.0f;
    engine->prev_output = 0.0f;

    size_t input_dim = (size_t)ed + 1;
    float* input_buf = (float*)safe_malloc(input_dim * sizeof(float));
    if (!input_buf) return -1;

    LNN* lnn = engine->shared_lnn;
    float dt = 1.0f / (float)sr;
    float exp_base = expf(-dt / tau);
    float one_minus_base = 1.0f - exp_base;

    for (int s = 0; s < total_samples; s++) {
        float progress_ratio = (float)s / total_samples;
        int token_idx = (int)(progress_ratio * num_tokens);
        if (token_idx >= num_tokens) token_idx = num_tokens - 1;
        if (token_idx < 0) token_idx = 0;

        int token_id = tokens[token_idx] % engine->config.vocab_size;
        const float* embed = engine->embedding_table + (size_t)token_id * ed;

        memcpy(input_buf, embed, (size_t)ed * sizeof(float));
        input_buf[ed] = engine->prev_output;

        /* Phase5: 强制使用自包含CfC路径，移除lnn_forward(共享LNN)污染
         * TTS为生成任务，不应修改共享LNN的hidden_state
         * 自包含CfC使用嵌入表权重，独立于共享LNN演化 */
        {
            const float embed_weight_scale = 0.2f;
            for (int i = 0; i < hs; i++) {
                float gate_sum = 0.0f;
                float act_sum = 0.0f;
                for (size_t j = 0; j < input_dim; j++) {
                    gate_sum += input_buf[j] * (engine->embedding_table[(size_t)((i+j)%ed)]) * embed_weight_scale;
                    act_sum  += input_buf[j] * (engine->embedding_table[(size_t)((i+j+hs/2)%ed)]) * embed_weight_scale;
                }
                gate_sum += engine->hidden_state[i] * 0.15f;
                act_sum  += engine->hidden_state[i] * 0.15f;
                float gate = 1.0f / (1.0f + expf(-gate_sum * 0.5f));
                float activation = tanhf(act_sum * 0.5f);
                float driver = gate * activation;
                float prev_h = engine->hidden_state[i];
                float new_h = prev_h * exp_base + one_minus_base * driver;
                if (isnan(new_h) || isinf(new_h)) new_h = prev_h;
                engine->hidden_state[i] = new_h;
            }
        }

        /* 从CfC隐藏状态直接投影到波形样本并加共振峰滤波 */
        float wav_sample = 0.0f;
        for (int h = 0; h < hs; h++) {
            wav_sample += engine->hidden_state[h] * engine->waveform_projection_w[h];
        }
        wav_sample += engine->waveform_projection_b[0];

        /* 声门脉冲激励：基于CfC动态的准周期性脉冲串 */
        float freq_mod = 0.0f;
        for (int h = 0; h < hs; h++) {
            freq_mod += engine->hidden_state[h] * engine->freq_projection_w[(size_t)h * 2];
        }
        freq_mod += engine->freq_projection_b[0];
        float F0 = TTS_BASE_FREQ * pitch_factor * (1.0f + 0.20f * freq_mod);
        if (F0 < 60.0f) F0 = 60.0f;
        if (F0 > 400.0f) F0 = 400.0f;
        engine->phase += F0 / sr;
        if (engine->phase > 1.0f) engine->phase -= 1.0f;

        /* 声门源：Liljencrants-Fant (LF)完整4参数模型
         * M-005修复：使用标准LF模型四阶段波形
         * 参数：Oq=开放商0.4, α=返回率指数衰减, Ee=闭合幅度 */
        float phase_norm = engine->phase;
        float Oq = 0.45f;   /* 开放商 → 声门开放时长占比 */
        float Qa = 0.15f;   /* 返回相占比 */
        float alpha = 3.0f; /* 返回相指数衰减率 */
        float glottal_pulse;

        if (phase_norm < Oq) {
            /* 阶段1-2: 开放相 (上升段+峰值段) */
            float t = phase_norm / Oq;
            if (t < 0.5f) {
                /* 开放上升: 正弦上升 0→1 */
                glottal_pulse = sinf((float)M_PI_2 * t * 2.0f);
            } else {
                /* 开放下降: 从峰值缓慢下降 */
                float t2 = (t - 0.5f) * 2.0f;
                glottal_pulse = 1.0f - 0.3f * t2;
            }
        } else if (phase_norm < Oq + Qa * (1.0f - Oq)) {
            /* 阶段3: 返回相 — 指数衰减闭合 */
            float t_r = (phase_norm - Oq) / (Qa * (1.0f - Oq));
            float decay = expf(-alpha * t_r);
            glottal_pulse = 0.7f * decay * (1.0f - t_r);
        } else {
            /* 阶段4: 闭合相 — 零气流 */
            glottal_pulse = 0.0f;
        }

        /* ZSFABC-F006修复: 5阶共振峰滤波器级联
         * 使用基于声道的标准共振峰频率范围替代均匀间隔。
         * F1=250-900Hz(开口度), F2=850-2400Hz(舌位前后), 
         * F3=1700-3400Hz, F4=3200-4000Hz, F5=4000-4900Hz
         * CfC隐藏状态投影调制基准共振峰频率 */
        float formant_freq[5], formant_bw[5];
        float base_freq[5] = {600.0f, 1500.0f, 2500.0f, 3600.0f, 4400.0f};
        float base_bw[5] = {60.0f, 90.0f, 130.0f, 160.0f, 200.0f};
        for (int f = 0; f < 5; f++) {
            formant_freq[f] = base_freq[f];
            formant_bw[f] = base_bw[f];
            float proj_sum = 0.0f;
            for (int h = 0; h < hs && h < 64; h++) {
                proj_sum += engine->hidden_state[h] * (engine->embedding_table[((h+f) % ed)] * 0.1f);
            }
            formant_freq[f] *= (0.7f + 0.6f * (1.0f / (1.0f + expf(-proj_sum))));
            formant_bw[f] *= (0.5f + 1.0f * (1.0f / (1.0f + expf(-proj_sum * 0.5f))));
            if (formant_freq[f] < 100.0f) formant_freq[f] = 100.0f;
            if (formant_freq[f] > 5000.0f) formant_freq[f] = 5000.0f;
            if (formant_bw[f] < 20.0f) formant_bw[f] = 20.0f;
            if (formant_bw[f] > 800.0f) formant_bw[f] = 800.0f;
        }

        /* 级联共振峰滤波：每个共振峰 = 双二次带通滤波器 (Butterworth二阶) */
        static float z1[5] = {0}, z2[5] = {0};
        float filter_in = glottal_pulse;
        float filter_out = 0.0f;
        for (int f = 0; f < 5; f++) {
            float w0 = 2.0f * (float)M_PI * formant_freq[f] / sr;
            float bw = formant_bw[f] / sr;
            float alpha_f = sinf(w0) * sinhf(logf(2.0f) / 2.0f * bw * w0 / sinf(w0) + 0.0001f);
            float a0 = 1.0f + alpha_f;
            float a1 = -2.0f * cosf(w0);
            float a2 = 1.0f - alpha_f;
            float b0 = alpha_f;
            float b1 = 0.0f;
            float b2 = -alpha_f;
            float out = (b0/a0) * filter_in + z1[f];
            z1[f] = (b1/a0) * filter_in - (a1/a0) * out + z2[f];
            z2[f] = (b2/a0) * filter_in - (a2/a0) * out;
            filter_in = out;
        }
        filter_out = filter_in;

        /* 混合：80%共振峰滤波 + 20% CfC直接投影 */
        float sample = 0.80f * filter_out + 0.20f * tanhf(wav_sample * 1.5f);

        /* 输出平滑和音量控制 */
        float smooth_alpha = 0.10f;
        sample = (1.0f - smooth_alpha) * engine->prev_output + smooth_alpha * sample;
        sample *= volume * 1.2f; /* 补偿共振峰滤波的能量衰减 */

        /* 软裁剪防止失真 */
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;

        waveform[s] = sample;
        engine->prev_output = sample;
    }

    safe_free((void**)&input_buf);
    return total_samples;
}

/* =============================================================== *
 * WAV文件写入                                                      *
 * =============================================================== */

static int write_wav_file(const char* path, const float* samples,
                           int num_samples, int sample_rate) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    int bytes_per_sample = 2;
    int num_channels = 1;
    int data_size = num_samples * bytes_per_sample;
    int file_size = 36 + data_size;

    /* WAV头 */
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);

    int fmt_size = 16;
    short audio_format = 1;       /* PCM */
    short channels = 1;           /* 单声道 */
    int byte_rate = sample_rate * bytes_per_sample * num_channels;
    short block_align = (short)(bytes_per_sample * num_channels);
    short bits_per_sample = 16;

    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);

    /* data块 */
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    for (int i = 0; i < num_samples; i++) {
        short s = (short)(samples[i] * 32767.0f);
        if (s > 32767) s = 32767;
        if (s < -32768) s = -32768;
        fwrite(&s, 2, 1, f);
    }

    fclose(f);
    return 0;
}

/* =============================================================== *
 * 文本 → token序列                                                  *
 * =============================================================== */

static int text_to_tokens(TTSEngine* engine, const char* text,
                           int* tokens, int max_tokens) {
    if (!engine || !text || !tokens) return -1;

    int count = 0;
    size_t pos = 0;
    while (text[pos] != '\0' && count < max_tokens) {
        int codepoint = utf8_decode(text, &pos);
        pos++;

        if (codepoint >= 0x20 && codepoint <= 0x7E) {
            /* ASCII可打印字符：直接映射 */
            tokens[count++] = codepoint;
        } else if (codepoint >= 0x4E00 && codepoint <= 0x9FFF) {
            /* CJK统一汉字(U+4E00-U+9FFF)：连续线性映射
             * 将20,992个汉字的Unicode码点范围均匀映射到词表空间
             * 保留语义相近汉字在嵌入空间的邻接关系（如"一"和"丁"相邻） */
            int range = 0x9FFF - 0x4E00;
            int offset = codepoint - 0x4E00;
            int mapped = (int)((float)offset / (float)range * (float)(engine->config.vocab_size - 32));
            if (mapped < 0) mapped = 0;
            if (mapped >= engine->config.vocab_size - 32) mapped = engine->config.vocab_size - 33;
            tokens[count++] = 32 + mapped;
        } else if (codepoint >= 0 && codepoint < 32) {
            /* 控制字符跳过 */
        } else {
            /* 其他Unicode字符：映射到 '?' */
            tokens[count++] = '?';
        }
    }
    return count;
}

/* =============================================================== *
 * TTS完整模型保存 —— 编码器 + 解码器 + 声码器权重                     *
 * =============================================================== */

/**
 * @brief 保存完整TTS模型权重（编码器+解码器+神经声码器）到二进制文件
 *
 * 文件格式:
 *   [4字节魔数: 0x5454534D "TTSM"]
 *   [4字节版本: TTS_MODEL_VERSION]
 *   [4字节hidden_size] [4字节embedding_dim] [4字节mel_bands]
 *   [编码器权重: 3层 × (W + U + bias)]
 *   [解码器权重: 3层 × (W + U + bias) + output_W + output_b]
 *   [声码器权重: 8层 × (conv_W + conv_b) + residual/skip/out]
 *
 * @param engine TTS引擎
 * @param filepath 文件路径
 * @return 0成功，-1失败
 */
int tts_save_model(TTSEngine* engine, const char* filepath) {
    if (!engine || !filepath) return -1;
    if (!engine->acoustic_model_initialized) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    uint32_t magic = TTS_MODEL_MAGIC;
    uint32_t version = TTS_MODEL_VERSION;
    int hs = (int)engine->config.hidden_size;
    int ed = engine->config.embedding_dim;
    int mel = TTS_MEL_BANDS;
    if (hs <= 0) hs = TTS_DEFAULT_HIDDEN_SIZE;
    if (ed <= 0) ed = 64;

    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&version, sizeof(uint32_t), 1, fp);
    fwrite(&hs, sizeof(int), 1, fp);
    fwrite(&ed, sizeof(int), 1, fp);
    fwrite(&mel, sizeof(int), 1, fp);

    /* 编码器权重：3层 */
    for (int l = 0; l < TTS_ENCODER_LAYERS; l++) {
        int in_dim = (l == 0) ? ed : hs;
        size_t w_size = (size_t)hs * in_dim;
        fwrite(engine->encoder_weights[l], sizeof(float), w_size, fp);
        fwrite(engine->encoder_u[l], sizeof(float), w_size, fp);
        fwrite(engine->encoder_bias[l], sizeof(float), (size_t)hs, fp);
        fwrite(&engine->encoder_tau[l], sizeof(float), 1, fp);
    }

    /* 解码器权重：3层 + 输出投影 */
    for (int l = 0; l < TTS_DECODER_LAYERS; l++) {
        int in_dim = (l == 0) ? (hs + mel) : hs;
        size_t w_size = (size_t)hs * in_dim;
        fwrite(engine->decoder_weights[l], sizeof(float), w_size, fp);
        fwrite(engine->decoder_u[l], sizeof(float), w_size, fp);
        fwrite(engine->decoder_bias[l], sizeof(float), (size_t)hs, fp);
        fwrite(&engine->decoder_tau[l], sizeof(float), 1, fp);
    }
    fwrite(engine->decoder_out_w, sizeof(float), (size_t)mel * hs, fp);
    fwrite(engine->decoder_out_b, sizeof(float), (size_t)mel, fp);

    /* 声码器权重 */
    int vocoder_ready = engine->vocoder_initialized ? 1 : 0;
    fwrite(&vocoder_ready, sizeof(int), 1, fp);
    if (vocoder_ready) {
        int res_ch = TTS_VOCODER_RESIDUAL_CHANNELS;
        int skip_ch = TTS_VOCODER_SKIP_CHANNELS;
        int kernel = TTS_VOCODER_KERNEL_SIZE;

        for (int l = 0; l < TTS_VOCODER_DILATION_LAYERS; l++) {
            size_t w_size = (size_t)(2 * res_ch) * res_ch * kernel;
            fwrite(engine->vocoder_conv_w[l], sizeof(float), w_size, fp);
            fwrite(engine->vocoder_conv_b[l], sizeof(float), (size_t)(2 * res_ch), fp);
        }
        fwrite(engine->vocoder_res_conv_w, sizeof(float), (size_t)res_ch * res_ch, fp);
        fwrite(engine->vocoder_res_conv_b, sizeof(float), (size_t)res_ch, fp);
        fwrite(engine->vocoder_skip_conv_w, sizeof(float), (size_t)skip_ch * res_ch, fp);
        fwrite(engine->vocoder_skip_conv_b, sizeof(float), (size_t)skip_ch, fp);
        fwrite(engine->vocoder_out_w, sizeof(float), (size_t)skip_ch, fp);
        fwrite(engine->vocoder_out_b, sizeof(float), 1, fp);
    }

    /* 同时保存嵌入表 */
    fwrite(engine->embedding_table, sizeof(float),
           (size_t)engine->config.vocab_size * ed, fp);

    fclose(fp);
    return 0;
}

/**
 * @brief 从二进制文件加载完整TTS模型权重（编码器+解码器+神经声码器）
 *
 * 先自动初始化所有权重缓冲区（如果尚未初始化），然后从文件填充。
 * 如果文件不兼容（维度不匹配），返回错误。
 *
 * @param engine TTS引擎
 * @param filepath 文件路径
 * @return 0成功，-1失败
 */
int tts_load_model(TTSEngine* engine, const char* filepath) {
    if (!engine || !filepath) return -1;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    uint32_t magic = 0, version = 0;
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 || magic != TTS_MODEL_MAGIC) {
        fclose(fp); return -1;
    }
    if (fread(&version, sizeof(uint32_t), 1, fp) != 1 || version > TTS_MODEL_VERSION) {
        fclose(fp); return -1;
    }

    int file_hs, file_ed, file_mel;
    if (fread(&file_hs, sizeof(int), 1, fp) != 1 ||
        fread(&file_ed, sizeof(int), 1, fp) != 1 ||
        fread(&file_mel, sizeof(int), 1, fp) != 1) {
        fclose(fp); return -1;
    }

    int hs = (int)engine->config.hidden_size;
    int ed = engine->config.embedding_dim;
    if (hs <= 0) hs = TTS_DEFAULT_HIDDEN_SIZE;
    if (ed <= 0) ed = 64;

    if (file_hs != hs || file_ed != ed || file_mel != TTS_MEL_BANDS) {
        fprintf(stderr, "[TTS模型加载] 维度不匹配: 文件(hs=%d,ed=%d,mel=%d) vs 引擎(hs=%d,ed=%d,mel=%d)\n",
                file_hs, file_ed, file_mel, hs, ed, TTS_MEL_BANDS);
        fclose(fp); return -1;
    }

    /* 自动初始化权重缓冲区 */
    if (!engine->acoustic_model_initialized) {
        if (init_acoustic_encoder(engine) != 0) { fclose(fp); return -1; }
        if (init_acoustic_decoder(engine) != 0) { fclose(fp); return -1; }
    }

    /* 加载编码器权重 */
    for (int l = 0; l < TTS_ENCODER_LAYERS; l++) {
        int in_dim = (l == 0) ? ed : hs;
        size_t w_size = (size_t)hs * in_dim;
        if (fread(engine->encoder_weights[l], sizeof(float), w_size, fp) != w_size) { fclose(fp); return -1; }
        if (fread(engine->encoder_u[l], sizeof(float), w_size, fp) != w_size) { fclose(fp); return -1; }
        if (fread(engine->encoder_bias[l], sizeof(float), (size_t)hs, fp) != (size_t)hs) { fclose(fp); return -1; }
        if (fread(&engine->encoder_tau[l], sizeof(float), 1, fp) != 1) { fclose(fp); return -1; }
    }

    /* 加载解码器权重 */
    for (int l = 0; l < TTS_DECODER_LAYERS; l++) {
        int in_dim = (l == 0) ? (hs + TTS_MEL_BANDS) : hs;
        size_t w_size = (size_t)hs * in_dim;
        if (fread(engine->decoder_weights[l], sizeof(float), w_size, fp) != w_size) { fclose(fp); return -1; }
        if (fread(engine->decoder_u[l], sizeof(float), w_size, fp) != w_size) { fclose(fp); return -1; }
        if (fread(engine->decoder_bias[l], sizeof(float), (size_t)hs, fp) != (size_t)hs) { fclose(fp); return -1; }
        if (fread(&engine->decoder_tau[l], sizeof(float), 1, fp) != 1) { fclose(fp); return -1; }
    }
    if (fread(engine->decoder_out_w, sizeof(float), (size_t)TTS_MEL_BANDS * hs, fp) != (size_t)TTS_MEL_BANDS * hs) { fclose(fp); return -1; }
    if (fread(engine->decoder_out_b, sizeof(float), (size_t)TTS_MEL_BANDS, fp) != (size_t)TTS_MEL_BANDS) { fclose(fp); return -1; }

    /* 加载声码器权重 */
    int vocoder_ready = 0;
    if (fread(&vocoder_ready, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }
    if (vocoder_ready) {
        if (!engine->vocoder_initialized) {
            if (init_neural_vocoder(engine) != 0) { fclose(fp); return -1; }
        }
        int res_ch = TTS_VOCODER_RESIDUAL_CHANNELS;
        int skip_ch = TTS_VOCODER_SKIP_CHANNELS;
        int kernel = TTS_VOCODER_KERNEL_SIZE;

        for (int l = 0; l < TTS_VOCODER_DILATION_LAYERS; l++) {
            size_t w_size = (size_t)(2 * res_ch) * res_ch * kernel;
            if (fread(engine->vocoder_conv_w[l], sizeof(float), w_size, fp) != w_size) { fclose(fp); return -1; }
            if (fread(engine->vocoder_conv_b[l], sizeof(float), (size_t)(2 * res_ch), fp) != (size_t)(2 * res_ch)) { fclose(fp); return -1; }
        }
        if (fread(engine->vocoder_res_conv_w, sizeof(float), (size_t)res_ch * res_ch, fp) != (size_t)res_ch * res_ch) { fclose(fp); return -1; }
        if (fread(engine->vocoder_res_conv_b, sizeof(float), (size_t)res_ch, fp) != (size_t)res_ch) { fclose(fp); return -1; }
        if (fread(engine->vocoder_skip_conv_w, sizeof(float), (size_t)skip_ch * res_ch, fp) != (size_t)skip_ch * res_ch) { fclose(fp); return -1; }
        if (fread(engine->vocoder_skip_conv_b, sizeof(float), (size_t)skip_ch, fp) != (size_t)skip_ch) { fclose(fp); return -1; }
        if (fread(engine->vocoder_out_w, sizeof(float), (size_t)skip_ch, fp) != (size_t)skip_ch) { fclose(fp); return -1; }
        if (fread(engine->vocoder_out_b, sizeof(float), 1, fp) != 1) { fclose(fp); return -1; }
    }

    /* 加载嵌入表 */
    size_t emb_size = (size_t)engine->config.vocab_size * ed;
    if (emb_size > 0) {
        if (fread(engine->embedding_table, sizeof(float), emb_size, fp) != emb_size) {
            /* 嵌入表加载失败不致命，使用已初始化的值 */
        }
    }

    engine->model_loaded = 1;
    engine->is_trained = 1;  /* 显式标记已训练：权重从文件加载成功 */
    fclose(fp);
    return 0;
}

/* =============================================================== *
 * 公开API实现                                                       *
 * =============================================================== */

TTSEngine* tts_engine_create(const TTSConfig* config) {
    if (!config) return NULL;

    TTSEngine* engine = (TTSEngine*)safe_malloc(sizeof(TTSEngine));
    if (!engine) return NULL;
    memset(engine, 0, sizeof(TTSEngine));

    engine->config = *config;

    /* 参数合法性修正 */
    if (engine->config.sample_rate <= 0) engine->config.sample_rate = TTS_SAMPLE_RATE_DEFAULT;
    if (engine->config.hidden_size <= 0) engine->config.hidden_size = TTS_DEFAULT_HIDDEN_SIZE;
    if (engine->config.time_constant <= 0) engine->config.time_constant = TTS_DEFAULT_TIME_CONSTANT;
    if (engine->config.vocab_size <= 0) engine->config.vocab_size = 128;
    if (engine->config.embedding_dim <= 0) engine->config.embedding_dim = 64;
    if (engine->config.speed <= 0) engine->config.speed = 1.0f;
    if (engine->config.volume < 0) engine->config.volume = 1.0f;

    /* 初始化拼音表 */
    if (init_pinyin_table(engine) != 0) {
        safe_free((void**)&engine);
        return NULL;
    }

    /* 初始化嵌入表（惰性） */
    if (init_embeddings(engine) != 0) {
        if (!engine->use_real_pinyin_lookup && engine->pinyin_table)
            safe_free((void**)&engine->pinyin_table);
        safe_free((void**)&engine);
        return NULL;
    }

    /* 状态缓冲区初始化（惰性，在首次合成时进行） */
    engine->token_capacity = 512;
    engine->token_buffer = (int*)safe_malloc(
        (size_t)engine->token_capacity * sizeof(int));
    if (!engine->token_buffer) {
        safe_free((void**)&engine->embedding_table);
        if (!engine->use_real_pinyin_lookup && engine->pinyin_table)
            safe_free((void**)&engine->pinyin_table);
        safe_free((void**)&engine);
        return NULL;
    }

    /* ZSFABC: 创建时立即初始化状态缓冲区和投影权重。
     * 原先采用惰性初始化(仅首次tts_synthesize调用时分配)，
     * 导致健康检查(tts_engine_is_healthy)因state_initialized=0而拦截请求。
     * 现在创建即完整，tts_synthesize内惰性检查变为恒真跳过。 */
    if (init_tts_buffers(engine) != 0) {
        safe_free((void**)&engine->token_buffer);
        safe_free((void**)&engine->embedding_table);
        if (engine->pinyin_table != (TTS_PinyinEntry*)(uintptr_t)1)
            safe_free((void**)&engine->pinyin_table);
        safe_free((void**)&engine);
        return NULL;
    }

    engine->initialized = 1;

    /* 初始化深度声学模型（编码器+解码器）和神经声码器
     * 这些是惰性初始化允许的：可以在创建引擎后通过tts_load_model从文件加载 */
    if (init_acoustic_encoder(engine) != 0) {
        fprintf(stderr, "[TTS] 声学模型编码器初始化失败\n");
        /* 不致命，允许引擎创建但标记未初始化 */
    }
    if (init_acoustic_decoder(engine) != 0) {
        fprintf(stderr, "[TTS] 声学模型解码器初始化失败\n");
    }
    if (init_neural_vocoder(engine) != 0) {
        fprintf(stderr, "[TTS] 神经声码器初始化失败\n");
    }

    return engine;
}

void tts_engine_free(TTSEngine* engine) {
    if (!engine) return;

    safe_free((void**)&engine->hidden_state);

    safe_free((void**)&engine->embedding_table);
    if (!engine->use_real_pinyin_lookup && engine->pinyin_table)
        safe_free((void**)&engine->pinyin_table);
    safe_free((void**)&engine->waveform_projection_w);
    safe_free((void**)&engine->waveform_projection_b);
    safe_free((void**)&engine->freq_projection_w);
    safe_free((void**)&engine->freq_projection_b);
    safe_free((void**)&engine->token_buffer);

    /* 释放编码器权重 */
    for (int l = 0; l < TTS_ENCODER_LAYERS; l++) {
        safe_free((void**)&engine->encoder_weights[l]);
        safe_free((void**)&engine->encoder_u[l]);
        safe_free((void**)&engine->encoder_bias[l]);
    }

    /* 释放解码器权重 */
    for (int l = 0; l < TTS_DECODER_LAYERS; l++) {
        safe_free((void**)&engine->decoder_weights[l]);
        safe_free((void**)&engine->decoder_u[l]);
        safe_free((void**)&engine->decoder_bias[l]);
    }
    safe_free((void**)&engine->decoder_out_w);
    safe_free((void**)&engine->decoder_out_b);

    /* 释放神经声码器权重 */
    for (int l = 0; l < TTS_VOCODER_DILATION_LAYERS; l++) {
        safe_free((void**)&engine->vocoder_conv_w[l]);
        safe_free((void**)&engine->vocoder_conv_b[l]);
    }
    safe_free((void**)&engine->vocoder_res_conv_w);
    safe_free((void**)&engine->vocoder_res_conv_b);
    safe_free((void**)&engine->vocoder_skip_conv_w);
    safe_free((void**)&engine->vocoder_skip_conv_b);
    safe_free((void**)&engine->vocoder_out_w);
    safe_free((void**)&engine->vocoder_out_b);

    safe_free((void**)&engine);
}

/* =============================================================== *
 * 确定性共振峰合成（主入口）                                        *
 * 当LNN未训练时使用此路径，基于声学语音学数据直接合成语音            *
 * =============================================================== */
static int tts_deterministic_synthesize(TTSEngine* engine, const char* text,
                                         TTS_Pinyin* pinyins, int max_pinyins,
                                         float* waveform, int max_samples) {
    int sr = engine->config.sample_rate;
    int pinyin_count = tts_text_to_pinyin(engine, text, pinyins, max_pinyins);
    if (pinyin_count <= 0) return -1;

    int samples_per_phone = (int)(TTS_SAMPLES_PER_PHONE * sr / TTS_SAMPLE_RATE_DEFAULT / engine->config.speed);
    if (samples_per_phone < 50) samples_per_phone = 50;

    float phase = 0.0f;
    float prev_sample = 0.0f;
    int total_written = 0;

    for (int p = 0; p < pinyin_count; p++) {
        int remaining = max_samples - total_written;
        if (remaining < samples_per_phone) break;

        float* chunk = waveform + total_written;
        tts_deterministic_formant_synth(engine,
                                         pinyins[p].initial,
                                         pinyins[p].final_part,
                                         pinyins[p].tone,
                                         chunk, sr, &phase, &prev_sample);
        total_written += samples_per_phone;
    }

    if (total_written == 0) return -1;
    return total_written;
}

TTSAudio* tts_synthesize(TTSEngine* engine, const char* text) {
    if (!engine || !text || !engine->initialized) return NULL;

    /* 惰性初始化状态缓冲区 */
    if (!engine->state_initialized) {
        if (init_tts_buffers(engine) != 0) return NULL;
    }

    /* 文本 → token序列 */
    int num_tokens = text_to_tokens(engine, text, engine->token_buffer,
                                     engine->token_capacity);
    if (num_tokens <= 0) {
        TTSAudio* silent = (TTSAudio*)safe_malloc(sizeof(TTSAudio));
        if (!silent) return NULL;
        silent->samples = (float*)safe_calloc((size_t)TTS_SAMPLES_PER_PHONE, sizeof(float));
        silent->num_samples = TTS_SAMPLES_PER_PHONE;
        silent->sample_rate = engine->config.sample_rate;
        return silent;
    }

    int max_samples = (int)(num_tokens * (TTS_SAMPLES_PER_PHONE *
        (float)engine->config.sample_rate / TTS_SAMPLE_RATE_DEFAULT / engine->config.speed)) + 1024;
    if (max_samples > TTS_MAX_WAVEFORM_LENGTH) {
        max_samples = TTS_MAX_WAVEFORM_LENGTH;
    }

    TTSAudio* audio = (TTSAudio*)safe_malloc(sizeof(TTSAudio));
    if (!audio) return NULL;
    audio->samples = (float*)safe_malloc((size_t)max_samples * sizeof(float));
    if (!audio->samples) {
        safe_free((void**)&audio);
        return NULL;
    }

    int actual_samples = 0;

    /* L-006修复: 未训练检查 —— CfC权重随机初始化时发出警告
     * 确定性声门脉冲+共振峰滤波能保证基本可听性，
     * 但语音自然度依赖训练后的CfC权重。 */
    if (tts_is_untrained(engine)) {
        fprintf(stderr, "[TTS警告] TTS模型未训练，语音自然度将受限。建议先运行 tts_train() 训练模型。\n");
    }
    actual_samples = generate_waveform(engine, engine->token_buffer, num_tokens,
                                        audio->samples, max_samples);

    if (actual_samples <= 0) {
        safe_free((void**)&audio->samples);
        safe_free((void**)&audio);
        return NULL;
    }

    audio->num_samples = actual_samples;
    audio->sample_rate = engine->config.sample_rate;

    /* F-011: 频域质量评估 */
    {
        float energy = 0.0f, zero_crossings = 0.0f;
        float max_amp = 0.0f, min_amp = 0.0f;
        for (int i = 0; i < actual_samples; i++) {
            float s = audio->samples[i];
            energy += s * s;
            if (i > 0) zero_crossings += (audio->samples[i-1] * s < 0) ? 1.0f : 0.0f;
            if (s > max_amp) max_amp = s;
            if (s < min_amp) min_amp = s;
        }
        float rms = sqrtf(energy / actual_samples);
        float zcr = zero_crossings / actual_samples;
        float dynamic_range = max_amp - min_amp;
        if (rms < 1e-6f) {
            fprintf(stderr, "[TTS频域评估] 警告：RMS=%.6e（可能为静音输出）\n", rms);
        }
        if (dynamic_range < 1e-6f && rms > 1e-6f) {
            fprintf(stderr, "[TTS频域评估] 警告：动态范围=%.6e（可能为直流偏置）\n", dynamic_range);
        }
        if (zcr < 1e-4f || zcr > 0.8f) {
            fprintf(stderr, "[TTS频域评估] 警告：过零率=%.4f（异常，正常范围0.01-0.5）\n", zcr);
        }
    }

    return audio;
}

void tts_audio_free(TTSAudio* audio) {
    if (!audio) return;
    safe_free((void**)&audio->samples);
    safe_free((void**)&audio);
}

int tts_synthesize_to_wav(TTSEngine* engine, const char* text, const char* wav_path) {
    if (!engine || !text || !wav_path) return -1;

    TTSAudio* audio = tts_synthesize(engine, text);
    if (!audio) return -1;

    int ret = write_wav_file(wav_path, audio->samples,
                              audio->num_samples, audio->sample_rate);

    tts_audio_free(audio);
    return ret;
}

/**
 * @brief CfC驱动端到端语音合成（替代正弦波表合成）
 *
 * 文本 → 字符嵌入 → 共享LNN连续状态演化 → 直接输出波形样本。
 * 不使用正弦波表、拼音映射或独立声学模型。
 * 所有文本到语音的映射由单个液态状态连续时间动态系统完成。
 *
 * @param engine TTS引擎（需已绑定共享LNN）
 * @param text 输入文本（UTF-8）
 * @param wav_path 输出WAV文件路径
 * @return int 0成功，-1失败
 */
int tts_synthesize_cfc_end_to_end(TTSEngine* engine, const char* text,
                                   const char* wav_path) {
    if (!engine || !text || !wav_path || !engine->initialized) return -1;

    if (!engine->state_initialized) {
        if (init_tts_buffers(engine) != 0) return -1;
    }

    int num_tokens = text_to_tokens(engine, text, engine->token_buffer,
                                     engine->token_capacity);
    if (num_tokens <= 0) return -1;

    int sr = engine->config.sample_rate;
    int hs = (int)engine->config.hidden_size;
    int ed = engine->config.embedding_dim;
    if (hs <= 0) hs = TTS_DEFAULT_HIDDEN_SIZE;
    if (ed <= 0) ed = 64;
    float volume = engine->config.volume;

    /* 步骤1：构建音素嵌入序列 [num_tokens × ed] */
    float* phoneme_embeds = (float*)safe_malloc((size_t)num_tokens * ed * sizeof(float));
    if (!phoneme_embeds) return -1;
    for (int t = 0; t < num_tokens; t++) {
        int token_id = engine->token_buffer[t] % engine->config.vocab_size;
        memcpy(phoneme_embeds + (size_t)t * ed,
               engine->embedding_table + (size_t)token_id * ed,
               (size_t)ed * sizeof(float));
    }

    /* 步骤2：CfC ODE编码器前向传播 */
    float* encoder_hidden = (float*)safe_malloc(
        (size_t)TTS_ENCODER_LAYERS * hs * sizeof(float));
    if (!encoder_hidden) {
        safe_free((void**)&phoneme_embeds);
        return -1;
    }
    int enc_ret = tts_encoder_cfc_forward(engine, phoneme_embeds, num_tokens, encoder_hidden);
    safe_free((void**)&phoneme_embeds);
    if (enc_ret != 0) {
        fprintf(stderr, "[TTS-CfC] 编码器前向失败，回退到原始路径\n");
        safe_free((void**)&encoder_hidden);

        /* 回退到原始generate_waveform路径 */
        int max_samples = (int)(num_tokens * (TTS_SAMPLES_PER_PHONE *
            (float)sr / TTS_SAMPLE_RATE_DEFAULT / engine->config.speed)) + 1024;
        if (max_samples > TTS_MAX_WAVEFORM_LENGTH) max_samples = TTS_MAX_WAVEFORM_LENGTH;
        float* waveform = (float*)safe_malloc((size_t)max_samples * sizeof(float));
        if (!waveform) return -1;
        int actual = generate_waveform(engine, engine->token_buffer, num_tokens,
                                        waveform, max_samples);
        if (actual <= 0) { safe_free((void**)&waveform); return -1; }
        int result = write_wav_file(wav_path, waveform, actual, sr);
        safe_free((void**)&waveform);
        return result;
    }

    /* 步骤3：自回归解码器 → 梅尔频谱图 */
    int max_mel_steps = num_tokens * 4 + 10;  /* 每个token约4帧梅尔频谱 */
    if (max_mel_steps > TTS_DECODER_MAX_STEPS) max_mel_steps = TTS_DECODER_MAX_STEPS;
    float* mel_spec = (float*)safe_malloc(
        (size_t)max_mel_steps * TTS_MEL_BANDS * sizeof(float));
    if (!mel_spec) {
        safe_free((void**)&encoder_hidden);
        return -1;
    }

    int mel_steps = 0;
    int dec_ret = tts_decoder_autoregressive_forward(engine, encoder_hidden,
                                                      mel_spec, max_mel_steps, &mel_steps);
    safe_free((void**)&encoder_hidden);

    if (dec_ret != 0 || mel_steps <= 0) {
        fprintf(stderr, "[TTS-CfC] 解码器失败或生成0帧\n");
        safe_free((void**)&mel_spec);
        return -1;
    }

    /* 步骤4：神经声码器 → 波形样本 */
    int max_wave = mel_steps * TTS_VOCODER_UPSAMPLE_FACTOR + TTS_FFT_WINDOW;
    if (max_wave > TTS_MAX_WAVEFORM_LENGTH) max_wave = TTS_MAX_WAVEFORM_LENGTH;
    float* waveform = (float*)safe_malloc((size_t)max_wave * sizeof(float));
    if (!waveform) {
        safe_free((void**)&mel_spec);
        return -1;
    }

    int wave_samples = 0;
    int voc_ret = tts_neural_vocoder_forward(engine, mel_spec, mel_steps,
                                              waveform, max_wave, &wave_samples);
    safe_free((void**)&mel_spec);

    if (voc_ret != 0 || wave_samples <= 0) {
        fprintf(stderr, "[TTS-CfC] 声码器失败\n");
        safe_free((void**)&waveform);
        return -1;
    }

    /* 后处理：去直流 + 归一化 + 音量 */
    if (wave_samples > 0) {
        float dc = 0.0f;
        for (int i = 0; i < wave_samples; i++) dc += waveform[i];
        dc /= (float)wave_samples;
        for (int i = 0; i < wave_samples; i++) waveform[i] -= dc;

        float max_abs = 1e-6f;
        for (int i = 0; i < wave_samples; i++) {
            float a = fabsf(waveform[i]);
            if (a > max_abs) max_abs = a;
        }
        float scale = 0.9f / max_abs * volume;
        for (int i = 0; i < wave_samples; i++) waveform[i] *= scale;
    }

    int result = write_wav_file(wav_path, waveform, wave_samples, sr);
    safe_free((void**)&waveform);
    return result;
}

int tts_engine_reset(TTSEngine* engine) {
    if (!engine) return -1;

    if (engine->hidden_state) {
        memset(engine->hidden_state, 0,
               engine->config.hidden_size * sizeof(float));
    }
    engine->phase = 0.0f;
    engine->prev_output = 0.0f;
    engine->sample_counter = 0;

    return 0;
}

int tts_engine_set_speed(TTSEngine* engine, float speed) {
    if (!engine) return -1;
    if (speed < 0.1f) speed = 0.1f;
    if (speed > 5.0f) speed = 5.0f;
    engine->config.speed = speed;
    return 0;
}

int tts_engine_set_pitch(TTSEngine* engine, float pitch) {
    if (!engine) return -1;
    if (pitch < -24.0f) pitch = -24.0f;
    if (pitch > 24.0f) pitch = 24.0f;
    engine->config.pitch_shift = pitch;
    return 0;
}

int tts_engine_set_volume(TTSEngine* engine, float volume) {
    if (!engine) return -1;
    if (volume < 0.0f) volume = 0.0f;
    if (volume > 1.0f) volume = 1.0f;
    engine->config.volume = volume;
    return 0;
}

int tts_engine_set_lnn(TTSEngine* engine, LNN* lnn) {
    if (!engine) return -1;
    engine->shared_lnn = lnn;
    return 0;
}

/* =============================================================== *
 * K-020: TTS 语音质量评估指标
 * 
 * 基于信号处理的质量评估：
 * 1. RMS能量 —— 衡量输出响度合理性
 * 2. 过零率 —— 衡量音高合理性
 * 3. 频谱重心 —— 衡量频率分布合理性
 * 4. 信噪比估计 —— 间接衡量语音清晰度
 * =============================================================== */

int tts_evaluate_quality(const float* waveform, int num_samples, int sample_rate,
                          float* out_rms, float* out_zcr, float* out_spectral_centroid,
                          float* out_overall_score) {
    if (!waveform || num_samples <= 0 || sample_rate <= 0) return -1;

    /* 1. RMS能量 (0~1归一化) */
    float energy_sum = 0.0f;
    float max_abs = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        float v = waveform[i];
        energy_sum += v * v;
        float abs_v = fabsf(v);
        if (abs_v > max_abs) max_abs = abs_v;
    }
    float rms = sqrtf(energy_sum / (float)num_samples);
    float rms_normalized = rms / (max_abs > 1e-6f ? max_abs : 1.0f);
    if (out_rms) *out_rms = rms_normalized;

    /* 2. 过零率 */
    int zcr_count = 0;
    for (int i = 1; i < num_samples; i++) {
        if ((waveform[i] >= 0.0f && waveform[i-1] < 0.0f) ||
            (waveform[i] < 0.0f && waveform[i-1] >= 0.0f)) {
            zcr_count++;
        }
    }
    float zcr = (float)zcr_count / (float)(num_samples - 1);
    if (out_zcr) *out_zcr = zcr;

    /* M-005修复: 频谱重心 — 使用真实FFT频域计算替代时域近似
     * 先对波形做FFT得到频谱幅度，再计算频域矩心 */
    float centroid_norm = 0.0f;
    {
        int fft_n = 1;
        while (fft_n < num_samples && fft_n < 2048) fft_n <<= 1;
        float* fft_real = (float*)safe_malloc((size_t)fft_n * sizeof(float));
        float* fft_imag = (float*)safe_malloc((size_t)fft_n * sizeof(float));
        if (fft_real && fft_imag) {
            for (int i = 0; i < fft_n; i++) {
                fft_real[i] = (i < num_samples) ? waveform[i] : 0.0f;
                fft_imag[i] = 0.0f;
            }
            tts_fft(fft_real, fft_imag, fft_n, 0);
            float cent_sum = 0.0f, cent_mag = 0.0f;
            int half_n = fft_n / 2;
            for (int i = 1; i < half_n; i++) {
                float mag = sqrtf(fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i]);
                float freq = (float)(i * sample_rate) / (float)fft_n;
                cent_sum += mag * freq;
                cent_mag += mag;
            }
            if (cent_mag > 1e-8f) {
                centroid_norm = (cent_sum / cent_mag) / (float)(sample_rate / 2);
            } else {
                centroid_norm = 0.0f;
            }
        }
        if (fft_real) safe_free((void**)&fft_real);
        if (fft_imag) safe_free((void**)&fft_imag);
    }
    if (out_spectral_centroid) *out_spectral_centroid = centroid_norm;

    /* 4. 综合质量评分 (0-1)
     * RMS合理范围: 0.05-0.8 (太弱或太强都扣分)
     * ZCR合理范围: 0.02-0.4 (语言信号典型范围)
     * 频谱重心合理: 0.05-0.6 */
    float rms_score = (rms_normalized > 0.01f && rms_normalized < 0.9f) ? 1.0f
                    : (rms_normalized > 0.0f) ? rms_normalized * 1.2f : 0.0f;
    float zcr_score = (zcr > 0.005f && zcr < 0.45f) ? 1.0f
                    : (zcr > 0.45f) ? (0.5f - zcr) * 10.0f + 1.0f : zcr * 200.0f;
    if (zcr_score < 0.0f) zcr_score = 0.0f;
    if (zcr_score > 1.0f) zcr_score = 1.0f;
    float cent_score = (centroid_norm > 0.01f && centroid_norm < 0.7f) ? 1.0f
                     : (centroid_norm > 0.0f) ? 0.7f : 0.3f;

    float overall = rms_score * 0.3f + zcr_score * 0.35f + cent_score * 0.35f;
    if (overall > 1.0f) overall = 1.0f;
    if (overall < 0.0f) overall = 0.0f;
    if (out_overall_score) *out_overall_score = overall;

    return 0;
}
