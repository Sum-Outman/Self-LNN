/**
 * @file audio_loader.c
 * @brief 纯C WAV音频文件加载器
 *
 * K-004补充: 实现纯C的WAV文件解析，从原始WAV文件解码为float数组。
 * 支持8/16/24/32位PCM格式，支持单声道和立体声(可选混合为单声道)。
 * 无任何第三方库依赖，纯C标准库实现。
 */

#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/**
 * @brief WAV文件头结构 (RIFF格式)
 */
#pragma pack(push, 1)
typedef struct {
    char     riff_id[4];      /* "RIFF" */
    uint32_t riff_size;       /* 文件大小-8 */
    char     wave_id[4];      /* "WAVE" */
    char     fmt_id[4];       /* "fmt " */
    uint32_t fmt_size;        /* fmt块大小(16=PCM) */
    uint16_t audio_format;    /* 音频格式(1=PCM) */
    uint16_t num_channels;    /* 声道数 */
    uint32_t sample_rate;     /* 采样率(Hz) */
    uint32_t byte_rate;       /* 每秒字节数 */
    uint16_t block_align;     /* 每帧字节数 */
    uint16_t bits_per_sample; /* 每采样点位数 */
    /* 后面可能有额外fmt数据，然后是"data"块 */
} WavHeader;
#pragma pack(pop)

/**
 * @brief 从WAV文件加载音频数据为float数组
 *
 * 自动检测声道数，立体声混合为单声道。
 * 支持8位无符号、16位有符号、24位有符号、32位有符号PCM格式。
 *
 * @param filepath WAV文件路径
 * @param sample_rate_out 输出采样率 (Hz)
 * @param num_samples_out 输出采样点数
 * @param channels_out 输出声道数(实际加载的声道数)
 * @return float数组(需调用者以safe_free释放), 失败返回NULL
 */
float* audio_load_wav(const char* filepath, int* sample_rate_out,
                       int* num_samples_out, int* channels_out) {
    if (!filepath || !sample_rate_out || !num_samples_out) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        log_error("[音频加载] 无法打开WAV文件: %s", filepath);
        return NULL;
    }

    *sample_rate_out = 0;
    *num_samples_out = 0;
    if (channels_out) *channels_out = 0;

    /* 读取WAV文件头 */
    WavHeader header;
    if (fread(&header, sizeof(WavHeader), 1, fp) != 1) {
        log_error("[音频加载] WAV文件头读取失败");
        fclose(fp);
        return NULL;
    }

    /* 验证RIFF/WAVE魔数 */
    if (memcmp(header.riff_id, "RIFF", 4) != 0 ||
        memcmp(header.wave_id, "WAVE", 4) != 0) {
        log_error("[音频加载] 无效的WAV文件格式");
        fclose(fp);
        return NULL;
    }

    /* 仅支持PCM格式 */
    if (header.audio_format != 1) {
        log_error("[音频加载] 不支持的WAV音频格式: %d (仅支持PCM=1)", header.audio_format);
        fclose(fp);
        return NULL;
    }

    int sample_rate = (int)header.sample_rate;
    int num_channels = (int)header.num_channels;
    int bits_per_sample = (int)header.bits_per_sample;

    if (sample_rate <= 0 || num_channels < 1 || bits_per_sample < 8) {
        log_error("[音频加载] WAV参数无效: %dHz, %d声道, %d位", sample_rate, num_channels, bits_per_sample);
        fclose(fp);
        return NULL;
    }
    /* P0修复: 限制num_channels上限，防止后续乘积(total_raw_samples*num_channels*sizeof)
     * 在32位平台发生size_t溢出。256声道已覆盖7.1.4等所有实际音频场景。 */
    if (num_channels > 256) {
        log_error("[音频加载] 声道数过大: %d (上限256)", num_channels);
        fclose(fp);
        return NULL;
    }

    /* 跳过fmt块剩余数据(如果有) */
    if (header.fmt_size > 16) {
        if (fseek(fp, (long)(header.fmt_size - 16), SEEK_CUR) != 0) {
            log_error("[音频加载] 跳过fmt块剩余数据失败");
            fclose(fp);
            return NULL;
        }
    }

    /* 查找"data"块 */
    char chunk_id[4] = {0};
    uint32_t chunk_size = 0;

    for (int attempt = 0; attempt < 20; attempt++) {
        if (fread(chunk_id, 1, 4, fp) != 4) {
            log_error("[音频加载] 未找到WAV data块");
            fclose(fp);
            return NULL;
        }
        if (fread(&chunk_size, 4, 1, fp) != 1) {
            fclose(fp);
            return NULL;
        }
        if (memcmp(chunk_id, "data", 4) == 0) break;
        /* 跳过非data块 */
        if (fseek(fp, (long)chunk_size, SEEK_CUR) != 0) {
            log_error("[音频加载] 跳过非data块失败");
            fclose(fp);
            return NULL;
        }
    }

    if (memcmp(chunk_id, "data", 4) != 0) {
        log_error("[音频加载] 扫描后未找到data块");
        fclose(fp);
        return NULL;
    }

    int bytes_per_sample = bits_per_sample / 8;
    int total_raw_samples = (int)(chunk_size / (uint32_t)(num_channels * bytes_per_sample));

    if (total_raw_samples <= 0) {
        log_error("[音频加载] WAV数据大小为0");
        fclose(fp);
        return NULL;
    }

    /* 限制最大加载样本数(防止内存耗尽) */
    int max_load = 48000 * 600; /* 10分钟@48kHz */
    if (total_raw_samples > max_load) {
        log_warning("[音频加载] WAV文件过大, 截取前%d样本", max_load);
        total_raw_samples = max_load;
    }

    /* 分配输出缓冲区 (总是输出单声道float) */
    size_t float_count = (size_t)total_raw_samples;
    float* audio_data = (float*)safe_calloc(float_count, sizeof(float));
    if (!audio_data) {
        fclose(fp);
        return NULL;
    }

    /* 读取并转换PCM数据到float */
    if (bits_per_sample == 16) {
        /* 16位有符号PCM */
        /* P0修复: 乘积溢出检查 — total_raw_samples*num_channels*sizeof(int16_t) */
        size_t alloc16 = (size_t)total_raw_samples * (size_t)num_channels * sizeof(int16_t);
        if (num_channels > 0 && alloc16 / (size_t)num_channels / sizeof(int16_t) != (size_t)total_raw_samples) {
            log_error("[音频加载] 16位PCM缓冲区大小计算溢出");
            safe_free((void**)&audio_data); fclose(fp); return NULL;
        }
        int16_t* raw_buf = (int16_t*)safe_malloc(alloc16);
        if (!raw_buf) { safe_free((void**)&audio_data); fclose(fp); return NULL; }
        size_t raw_count = (size_t)total_raw_samples * num_channels;
        size_t bytes_read = fread(raw_buf, sizeof(int16_t), raw_count, fp);
        if (bytes_read != raw_count) {
            log_warning("[音频加载] 实际读取%d采样点(期望%d)", (int)bytes_read, (int)raw_count);
            /* P1修复: 根据实际读取数量修正样本数，避免循环访问未初始化内存区域 */
            total_raw_samples = (int)(bytes_read / (size_t)num_channels);
        }
        for (int i = 0; i < total_raw_samples; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < num_channels; ch++) {
                sum += (float)raw_buf[i * num_channels + ch] / 32768.0f;
            }
            audio_data[i] = sum / (float)num_channels;
        }
        safe_free((void**)&raw_buf);
    } else if (bits_per_sample == 8) {
        /* 8位无符号PCM */
        /* P0修复: 乘积溢出检查 — total_raw_samples*num_channels (8位每采样1字节) */
        size_t alloc8 = (size_t)total_raw_samples * (size_t)num_channels;
        if (num_channels > 0 && alloc8 / (size_t)num_channels != (size_t)total_raw_samples) {
            log_error("[音频加载] 8位PCM缓冲区大小计算溢出");
            safe_free((void**)&audio_data); fclose(fp); return NULL;
        }
        uint8_t* raw_buf = (uint8_t*)safe_malloc(alloc8);
        if (!raw_buf) { safe_free((void**)&audio_data); fclose(fp); return NULL; }
        size_t raw_count = (size_t)total_raw_samples * num_channels;
        size_t bytes_read = fread(raw_buf, 1, raw_count, fp);
        if (bytes_read != raw_count) {
            log_warning("[音频加载] 8位实际读取%d字节(期望%d)", (int)bytes_read, (int)raw_count);
            /* P1修复: 根据实际读取字节数修正样本数，避免循环访问未初始化内存区域 */
            total_raw_samples = (int)(bytes_read / (size_t)num_channels);
        }
        for (int i = 0; i < total_raw_samples; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < num_channels; ch++) {
                sum += ((float)raw_buf[i * num_channels + ch] - 128.0f) / 128.0f;
            }
            audio_data[i] = sum / (float)num_channels;
        }
        safe_free((void**)&raw_buf);
    } else if (bits_per_sample == 24) {
        /* 24位有符号PCM (3字节) */
        /* P0修复: 乘积溢出检查 — total_raw_samples*num_channels*3 (24位每采样3字节) */
        size_t raw_bytes = (size_t)total_raw_samples * (size_t)num_channels * 3;
        if (num_channels > 0 && raw_bytes / (size_t)num_channels / 3 != (size_t)total_raw_samples) {
            log_error("[音频加载] 24位PCM缓冲区大小计算溢出");
            safe_free((void**)&audio_data); fclose(fp); return NULL;
        }
        uint8_t* raw_buf = (uint8_t*)safe_malloc(raw_bytes);
        if (!raw_buf) { safe_free((void**)&audio_data); fclose(fp); return NULL; }
        size_t bytes_read = fread(raw_buf, 1, raw_bytes, fp);
        if (bytes_read != raw_bytes) {
            log_warning("[音频加载] 24位实际读取%d字节(期望%d)", (int)bytes_read, (int)raw_bytes);
            /* P1修复: 根据实际读取字节数修正样本数，避免循环访问未初始化内存区域 */
            total_raw_samples = (int)(bytes_read / ((size_t)num_channels * 3));
        }
        for (int i = 0; i < total_raw_samples; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < num_channels; ch++) {
                int idx = (i * num_channels + ch) * 3;
                int32_t val = (int32_t)(raw_buf[idx] | (raw_buf[idx + 1] << 8) |
                                       ((int8_t)raw_buf[idx + 2] << 16));
                sum += (float)val / 8388608.0f;
            }
            audio_data[i] = sum / (float)num_channels;
        }
        safe_free((void**)&raw_buf);
    } else if (bits_per_sample == 32) {
        /* 32位有符号PCM */
        /* P0修复: 乘积溢出检查 — total_raw_samples*num_channels*sizeof(int32_t) */
        size_t alloc32 = (size_t)total_raw_samples * (size_t)num_channels * sizeof(int32_t);
        if (num_channels > 0 && alloc32 / (size_t)num_channels / sizeof(int32_t) != (size_t)total_raw_samples) {
            log_error("[音频加载] 32位PCM缓冲区大小计算溢出");
            safe_free((void**)&audio_data); fclose(fp); return NULL;
        }
        int32_t* raw_buf = (int32_t*)safe_malloc(alloc32);
        if (!raw_buf) { safe_free((void**)&audio_data); fclose(fp); return NULL; }
        size_t raw_count = (size_t)total_raw_samples * num_channels;
        size_t bytes_read = fread(raw_buf, sizeof(int32_t), raw_count, fp);
        if (bytes_read != raw_count) {
            log_warning("[音频加载] 32位实际读取%d采样点(期望%d)", (int)bytes_read, (int)raw_count);
            /* P1修复: 根据实际读取数量修正样本数，避免循环访问未初始化内存区域 */
            total_raw_samples = (int)(bytes_read / (size_t)num_channels);
        }
        for (int i = 0; i < total_raw_samples; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < num_channels; ch++) {
                sum += (float)raw_buf[i * num_channels + ch] / 2147483648.0f;
            }
            audio_data[i] = sum / (float)num_channels;
        }
        safe_free((void**)&raw_buf);
    } else {
        log_error("[音频加载] 不支持的位深: %d", bits_per_sample);
        safe_free((void**)&audio_data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);

    *sample_rate_out = sample_rate;
    *num_samples_out = total_raw_samples;
    if (channels_out) *channels_out = num_channels;

    float duration = (float)total_raw_samples / (float)sample_rate;
    log_info("[音频加载] WAV解码成功: %s (%dHz, %d声道, %d位, %.1f秒)",
             filepath, sample_rate, num_channels, bits_per_sample, duration);

    return audio_data;
}

/**
 * @brief 释放audio_load_wav加载的音频数据
 */
void audio_wav_free(float* data) {
    if (data) safe_free((void**)&data);
}

/**
 * @brief 获取WAV文件信息(不加载全部数据)
 *
 * @param filepath WAV文件路径
 * @param sample_rate_out 输出采样率
 * @param num_channels_out 输出声道数
 * @param bits_per_sample_out 输出位深
 * @param duration_sec_out 输出时长(秒)
 * @return 0成功, -1失败
 */
int audio_wav_info(const char* filepath, int* sample_rate_out,
                    int* num_channels_out, int* bits_per_sample_out,
                    float* duration_sec_out) {
    if (!filepath) return -1;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    WavHeader header;
    if (fread(&header, sizeof(WavHeader), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    if (memcmp(header.riff_id, "RIFF", 4) != 0 ||
        memcmp(header.wave_id, "WAVE", 4) != 0) {
        fclose(fp);
        return -1;
    }

    /* P2修复: 验证头部字段有效性，与audio_load_wav保持一致 */
    int sample_rate = (int)header.sample_rate;
    int num_channels = (int)header.num_channels;
    int bits_per_sample = (int)header.bits_per_sample;
    if (sample_rate <= 0 || num_channels < 1 || bits_per_sample < 8) {
        fclose(fp);
        return -1;
    }
    if (num_channels > 256) {
        fclose(fp);
        return -1;
    }

    if (sample_rate_out) *sample_rate_out = sample_rate;
    if (num_channels_out) *num_channels_out = num_channels;
    if (bits_per_sample_out) *bits_per_sample_out = bits_per_sample;

    /* 计算时长 */
    if (duration_sec_out && header.byte_rate > 0) {
        int bytes_per_sample = (int)header.num_channels * (int)header.bits_per_sample / 8;
        /* 查找data块 */
        char chunk_id[4];
        uint32_t chunk_size = 0;
        long data_start = 0;
        int found = 0;

        for (int attempt = 0; attempt < 20; attempt++) {
            if (fread(chunk_id, 1, 4, fp) != 4) break;
            if (fread(&chunk_size, 4, 1, fp) != 1) break;
            if (memcmp(chunk_id, "data", 4) == 0) {
                data_start = ftell(fp);
                found = 1;
                break;
            }
            /* P2修复: 检查fseek返回值，失败时终止扫描 */
            if (fseek(fp, (long)chunk_size, SEEK_CUR) != 0) break;
        }

        if (found && bytes_per_sample > 0) {
            int total_samples = (int)(chunk_size / (uint32_t)bytes_per_sample);
            *duration_sec_out = (float)total_samples / (float)header.sample_rate;
        } else {
            *duration_sec_out = 0.0f;
        }
    }

    fclose(fp);
    return 0;
}
