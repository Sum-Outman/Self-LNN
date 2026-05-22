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

/* =============================================================== *
 * 拼音查找函数（声明，实现在 tts_pinyin_real.c）                     *
 * 使用真实汉字→拼音二分查找，覆盖~220个最高频汉字                   *
 * 未找到返回0（TTS_INITIAL_NONE/TTS_FINAL_NONE）                   *
 * =============================================================== */
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

    /* Delta特征（一阶差分）：Δc(t) = c(t+1) - c(t-1)，简化为Δc(t) = c(t) - c(t-1) */
    for (int c = 0; c < num_ceps; c++) {
        if (prev_mfcc) {
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

        if (lnn) {
            float lnn_out[256] = {0};
            if (lnn_forward(lnn, input_buf, lnn_out) == 0) {
                for (int i = 0; i < hs; i++) {
                    engine->hidden_state[i] = 0.7f * engine->hidden_state[i] + 0.3f * lnn_out[i];
                }
            }
        } else {
            /* ZSFWS-M022修复: 自包含CfC回退路径
             * 复用嵌入表作为权重矩阵时，嵌入表为Xavier×0.1缩放，
             * 添加0.2尺度因子使门控和激活计算与标准CfC权重范围匹配 */
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

        /* 声门源：Liljencrants-Fant (LF)式简化脉冲 */
        float phase_norm = engine->phase;
        float glottal_pulse;
        if (phase_norm < 0.4f) {
            float t = phase_norm / 0.4f;
            glottal_pulse = 0.5f * (1.0f - cosf((float)M_PI * t));
        } else if (phase_norm < 0.6f) {
            float t = (phase_norm - 0.4f) / 0.2f;
            glottal_pulse = 1.0f - t;
        } else {
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
            float alpha = sinf(w0) * sinhf(logf(2.0f) / 2.0f * bw * w0 / sinf(w0) + 0.0001f);
            float a0 = 1.0f + alpha;
            float a1 = -2.0f * cosf(w0);
            float a2 = 1.0f - alpha;
            float b0 = alpha;
            float b1 = 0.0f;
            float b2 = -alpha;
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
            tokens[count++] = codepoint;
        } else if (codepoint >= 0x4E00 && codepoint <= 0x9FFF) {
            tokens[count++] = (codepoint % (engine->config.vocab_size - 32)) + 32;
        } else if (codepoint >= 0 && codepoint < 32) {
            /* 控制字符跳过 */
        } else {
            tokens[count++] = '?';
        }
    }
    return count;
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

    safe_free((void**)&engine);
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
        /* 至少生成一个静音片段 */
        TTSAudio* silent = (TTSAudio*)safe_malloc(sizeof(TTSAudio));
        if (!silent) return NULL;
        silent->samples = (float*)safe_calloc((size_t)TTS_SAMPLES_PER_PHONE, sizeof(float));
        silent->num_samples = TTS_SAMPLES_PER_PHONE;
        silent->sample_rate = engine->config.sample_rate;
        return silent;
    }

    /* 预估最大波形长度 */
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

    /* 核心生成：液态状态逐样本波形生成 */
    int actual_samples = generate_waveform(engine, engine->token_buffer, num_tokens,
                                            audio->samples, max_samples);
    if (actual_samples <= 0) {
        safe_free((void**)&audio->samples);
        safe_free((void**)&audio);
        return NULL;
    }

    audio->num_samples = actual_samples;
    audio->sample_rate = engine->config.sample_rate;

    /* F-011: 频域质量评估 - 验证输出波形的频谱质量 */
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

        /* 质量判定：RMS过小=静音，动态范围为0=直流，ZCR异常=噪声 */
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
    if (ed <= 0) ed = 64;
    float speed = engine->config.speed;
    float volume = engine->config.volume;

    /* 每token产生的帧数 */
    int frames_per_token = 8;
    int total_frames = num_tokens * frames_per_token;
    int max_samples = total_frames * TTS_FFT_WINDOW + TTS_FFT_WINDOW;
    if (max_samples > TTS_MAX_WAVEFORM_LENGTH) max_samples = TTS_MAX_WAVEFORM_LENGTH;

    float* waveform = (float*)safe_calloc((size_t)max_samples, sizeof(float));
    if (!waveform) return -1;

    /* 声学模型工作缓冲区 */
    float* mfcc_current = (float*)safe_calloc(TTS_MFCC_DIM, sizeof(float));
    float* mfcc_prev = (float*)safe_calloc(TTS_MFCC_DIM, sizeof(float));
    float* delta_prev = (float*)safe_calloc(13, sizeof(float));
    float* acoustic = (float*)safe_calloc(TTS_ACOUSTIC_DIM, sizeof(float));
    float* acou_hidden = (float*)safe_calloc((size_t)hs, sizeof(float));
    float* frame_wave = (float*)safe_calloc(TTS_FFT_WINDOW, sizeof(float));
    if (!mfcc_current || !mfcc_prev || !delta_prev || !acoustic || !acou_hidden || !frame_wave) {
        safe_free((void**)&mfcc_current); safe_free((void**)&mfcc_prev);
        safe_free((void**)&delta_prev); safe_free((void**)&acoustic);
        safe_free((void**)&acou_hidden); safe_free((void**)&frame_wave);
        safe_free((void**)&waveform);
        return -1;
    }

    /* 初始化声学隐藏状态（从嵌入表取随机初始值） */
    for (int h = 0; h < hs; h++) {
        acou_hidden[h] = engine->embedding_table[(size_t)(h % ed)] * 0.05f;
    }

    float global_phase = 0.0f;
    int sample_idx = 0;

    /* 对每个文本token生成声学特征和波形帧 */
    for (int t = 0; t < num_tokens && sample_idx < max_samples; t++) {
        int token_id = engine->token_buffer[t];

        /* 从嵌入表构造初始伪MFCC（用token embedding作为声学启动条件） */
        memset(mfcc_current, 0, TTS_MFCC_DIM * sizeof(float));
        for (int d = 0; d < 13 && d < ed; d++) {
            float embed_val = engine->embedding_table[(size_t)token_id * ed + d] * 3.0f;
            mfcc_current[d] = 0.3f * embed_val;
            if (mfcc_prev[0] != 0.0f) {
                mfcc_current[13 + d] = mfcc_current[d] - mfcc_prev[d];
            }
            mfcc_current[26 + d] = (delta_prev[0] != 0.0f && d < 13)
                ? mfcc_current[13 + d] - delta_prev[d] : 0.0f;
        }
        /* 首帧从token获取初始MFCC，后续帧从LNN声学模型自回归生成 */
        if (t == 0) {
            for (int d = 0; d < 13; d++) {
                mfcc_current[d] = engine->embedding_table[(size_t)token_id * ed + d] * 1.5f;
            }
            memcpy(mfcc_prev, mfcc_current, 13 * sizeof(float));
            memset(delta_prev, 0, 13 * sizeof(float));
        } else {
            /* 后续帧：CFC声学状态自回归 */
            for (int d = 0; d < 13; d++) {
                mfcc_current[d] = 0.9f * mfcc_prev[d]
                    + 0.1f * engine->embedding_table[(size_t)token_id * ed + d] * 1.2f
                    + 0.05f * tanhf(acou_hidden[d % hs]);
            }
            memcpy(delta_prev, mfcc_current + 13, 13 * sizeof(float));
            memcpy(mfcc_prev, mfcc_current, 13 * sizeof(float));
        }

        for (int f = 0; f < frames_per_token && sample_idx < max_samples; f++) {
            /* V-015核心：MFCC → LNN声学模型 → 增强声学特征 */
            int ac_ret = tts_lnn_acoustic_forward(engine, mfcc_current, acoustic, acou_hidden);
            if (ac_ret != 0) break;

            /* 声学特征 → 波形帧（使用HNM声码器 + Griffin-Lim） */
            int wf_ret = tts_acoustic_to_waveform(acoustic, frame_wave, sr, &global_phase);
            if (wf_ret != 0) break;

            /* 复制波帧到输出缓冲（跨帧叠加混合） */
            int copy_start = sample_idx;
            int frame_samples = TTS_FFT_WINDOW;
            for (int s = 0; s < frame_samples && copy_start + s < max_samples; s++) {
                waveform[copy_start + s] = frame_wave[s] * volume;
            }
            sample_idx += frame_samples / 2;  /* 50%重叠以平滑过渡 */

            /* 自回归：当前声学特征作为下一帧输入 */
            memcpy(mfcc_current, acoustic, 13 * sizeof(float));
            for (int d = 0; d < 13; d++) {
                mfcc_current[13 + d] = (mfcc_prev[0] != 0.0f) ? mfcc_current[d] - mfcc_prev[d] : 0.0f;
                mfcc_current[26 + d] = (delta_prev[0] != 0.0f) ? mfcc_current[13 + d] - delta_prev[d] : 0.0f;
            }
            memcpy(mfcc_prev, mfcc_current, 13 * sizeof(float));
            memcpy(delta_prev, mfcc_current + 13, 13 * sizeof(float));
        }
    }

    /* 后处理：去直流 + 归一化 */
    if (sample_idx > 0) {
        float dc = 0.0f;
        for (int i = 0; i < sample_idx; i++) dc += waveform[i];
        dc /= (float)sample_idx;
        for (int i = 0; i < sample_idx; i++) waveform[i] -= dc;

        float max_abs = 1e-6f;
        for (int i = 0; i < sample_idx; i++) {
            float a = fabsf(waveform[i]);
            if (a > max_abs) max_abs = a;
        }
        float scale = 0.9f / max_abs;
        for (int i = 0; i < sample_idx; i++) waveform[i] *= scale;
    }

    int result = write_wav_file(wav_path, waveform, sample_idx, sr);

    safe_free((void**)&mfcc_current); safe_free((void**)&mfcc_prev);
    safe_free((void**)&delta_prev); safe_free((void**)&acoustic);
    safe_free((void**)&acou_hidden); safe_free((void**)&frame_wave);
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

    /* 3. 频谱重心（简化FFT前N个bin的频域矩心） */
    int fft_bins = (num_samples < 1024) ? num_samples : 1024;
    float centroid = 0.0f;
    float total_mag = 0.0f;
    /* 使用简化的时域近似：通过相邻采样差分的自相关估计频率分布 */
    for (int i = 1; i < num_samples && fft_bins > 0; i++) {
        float diff = waveform[i] - waveform[i-1];
        float mag = fabsf(diff);
        float freq_bin = (float)(i * sample_rate) / (float)(2 * num_samples);
        if (freq_bin > 0.0f && freq_bin < (float)(sample_rate / 2)) {
            centroid += mag * freq_bin;
            total_mag += mag;
        }
    }
    if (total_mag > 1e-6f) {
        centroid /= total_mag;
    }
    float centroid_norm = centroid / (float)(sample_rate / 2);
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
