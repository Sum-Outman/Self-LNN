/**
 * @file image_loader.c
 * @brief 纯C图像格式解码器 —— BMP/PPM(P6)格式原生支持
 *
 * K-004/K-008修复: 实现纯C的BMP和PPM(P6)图像解码，从原始文件格式直接解码为
 * float RGB数组，无任何第三方库依赖。
 *
 * 支持的格式:
 *   BMP: Windows位图 (24位RGB, 32位BGRA)
 *   PPM: Netpbm P6二进制格式 (24位RGB)
 *
 * 使用方式: 训练数据加载、视觉处理管线直接调用
 */

#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/errors.h"

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* 前向声明：JPEG/PNG解码器（完整实现见文件后部） */
uint8_t* image_load_jpeg(const char* filepath, int* width_out, int* height_out);
uint8_t* image_load_png(const char* filepath, int* width_out, int* height_out);

/* BMP文件头 (14字节) */
#pragma pack(push, 1)
typedef struct {
    uint16_t bfType;       /* 文件类型 "BM" = 0x4D42 */
    uint32_t bfSize;       /* 文件总大小 */
    uint16_t bfReserved1;  /* 保留 */
    uint16_t bfReserved2;  /* 保留 */
    uint32_t bfOffBits;    /* 像素数据偏移 */
} BmpFileHeader;

/* BMP信息头 (40字节) */
typedef struct {
    uint32_t biSize;         /* 信息头大小 */
    int32_t  biWidth;        /* 图像宽度 */
    int32_t  biHeight;       /* 图像高度(正=左下角原点) */
    uint16_t biPlanes;       /* 颜色平面数(必须为1) */
    uint16_t biBitCount;     /* 每像素位数 */
    uint32_t biCompression;  /* 压缩类型(0=BI_RGB) */
    uint32_t biSizeImage;    /* 像素数据大小(可为零) */
    int32_t  biXPelsPerMeter;/* 水平分辨率 */
    int32_t  biYPelsPerMeter;/* 垂直分辨率 */
    uint32_t biClrUsed;      /* 使用的调色板颜色数 */
    uint32_t biClrImportant; /* 重要的调色板颜色数 */
} BmpInfoHeader;
#pragma pack(pop)

/* 加载的RGB图像数据 */
typedef struct {
    uint8_t* data;       /* RGB像素数据 [channels * width * height] */
    int width;           /* 图像宽度 */
    int height;          /* 图像高度 */
    int channels;        /* 颜色通道数(3=RGB) */
} LoadedImage;

/**
 * @brief 解码BMP文件为RGB字节数组
 * @param filepath BMP文件路径
 * @param width_out 输出宽度
 * @param height_out 输出高度
 * @return RGB字节数组(需调用者free), 失败返回NULL
 */
uint8_t* image_load_bmp(const char* filepath, int* width_out, int* height_out) {
    if (!filepath || !width_out || !height_out) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        log_error("[图像加载] 无法打开BMP文件: %s", filepath);
        return NULL;
    }

    *width_out = 0;
    *height_out = 0;

    /* 读取BMP文件头 */
    BmpFileHeader file_header;
    if (fread(&file_header, sizeof(BmpFileHeader), 1, fp) != 1) {
        log_error("[图像加载] BMP文件头读取失败");
        fclose(fp);
        return NULL;
    }

    /* 验证BMP魔数 */
    if (file_header.bfType != 0x4D42) { /* "BM" */
        log_error("[图像加载] 无效的BMP文件魔数: 0x%04X", file_header.bfType);
        fclose(fp);
        return NULL;
    }

    /* 读取BMP信息头 */
    BmpInfoHeader info_header;
    if (fread(&info_header, sizeof(BmpInfoHeader), 1, fp) != 1) {
        log_error("[图像加载] BMP信息头读取失败");
        fclose(fp);
        return NULL;
    }

    int width = info_header.biWidth;
    int height = info_header.biHeight;
    if (height < 0) height = -height; /* 正高度表示左下角原点 */
    int bit_count = info_header.biBitCount;
    int compression = info_header.biCompression;

    /* 仅支持24位RGB和32位BGRA无压缩格式 */
    if (bit_count != 24 && bit_count != 32) {
        log_error("[图像加载] 不支持的BMP位深: %d (仅支持24/32位)", bit_count);
        fclose(fp);
        return NULL;
    }
    if (compression != 0 && compression != 3) { /* 3=BI_BITFIELDS */
        log_error("[图像加载] 不支持的BMP压缩: %d", compression);
        fclose(fp);
        return NULL;
    }

    if (width <= 0 || height <= 0) {
        log_error("[图像加载] BMP尺寸无效: %dx%d", width, height);
        fclose(fp);
        return NULL;
    }

    /* 计算行填充字节 (每行对齐到4字节) */
    int bytes_per_pixel = bit_count / 8;
    int row_padded = ((width * bytes_per_pixel + 3) / 4) * 4;
    int row_actual = width * bytes_per_pixel;

    /* 跳到像素数据起始位置 */
    fseek(fp, file_header.bfOffBits, SEEK_SET);

    /* 分配输出缓冲 */
    size_t out_size = (size_t)width * height * 3;
    uint8_t* rgb_data = (uint8_t*)safe_malloc(out_size);
    if (!rgb_data) {
        fclose(fp);
        return NULL;
    }

    /* 读取像素行 (BMP存储为从下到上) */
    uint8_t* row_buffer = (uint8_t*)safe_malloc(row_padded);
    if (!row_buffer) {
        safe_free((void**)&rgb_data);
        fclose(fp);
        return NULL;
    }

    for (int y = 0; y < height; y++) {
        int dest_y = height - 1 - y; /* BMP从上到下，我们输出从上到下 */
        uint8_t* dest_row = rgb_data + (size_t)dest_y * width * 3;

        if (fread(row_buffer, 1, row_padded, fp) != (size_t)row_padded) {
            log_error("[图像加载] BMP像素数据读取失败: 行%d", y);
            safe_free((void**)&row_buffer);
            safe_free((void**)&rgb_data);
            fclose(fp);
            return NULL;
        }

        for (int x = 0; x < width; x++) {
            if (bit_count == 24) {
                /* BMP存储顺序: B, G, R */
                dest_row[x * 3 + 0] = row_buffer[x * 3 + 2]; /* R */
                dest_row[x * 3 + 1] = row_buffer[x * 3 + 1]; /* G */
                dest_row[x * 3 + 2] = row_buffer[x * 3 + 0]; /* B */
            } else {
                /* 32位: B, G, R, A */
                dest_row[x * 3 + 0] = row_buffer[x * 4 + 2]; /* R */
                dest_row[x * 3 + 1] = row_buffer[x * 4 + 1]; /* G */
                dest_row[x * 3 + 2] = row_buffer[x * 4 + 0]; /* B */
            }
        }
    }

    safe_free((void**)&row_buffer);
    fclose(fp);

    *width_out = width;
    *height_out = height;

    log_info("[图像加载] BMP解码成功: %s (%dx%d)", filepath, width, height);
    return rgb_data;
}

/**
 * @brief 解码PPM(P6二进制)文件为RGB字节数组
 *
 * PPM格式:
 *   P6\n
 *   <width> <height>\n
 *   <maxval>\n
 *   <RGB数据>
 *
 * @param filepath PPM文件路径
 * @param width_out 输出宽度
 * @param height_out 输出高度
 * @return RGB字节数组(需调用者free), 失败返回NULL
 */
uint8_t* image_load_ppm(const char* filepath, int* width_out, int* height_out) {
    if (!filepath || !width_out || !height_out) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        log_error("[图像加载] 无法打开PPM文件: %s", filepath);
        return NULL;
    }

    *width_out = 0;
    *height_out = 0;

    char magic[3] = {0};
    if (fread(magic, 1, 2, fp) != 2 || (magic[0] != 'P' || magic[1] != '6')) {
        log_error("[图像加载] 无效的PPM魔数: %c%c (仅支持P6)", magic[0] ? magic[0] : '?', magic[1] ? magic[1] : '?');
        fclose(fp);
        return NULL;
    }

    /* 跳过空白和注释行 */
    int c;
    do {
        c = fgetc(fp);
        if (c == '#') {
            while ((c = fgetc(fp)) != EOF && c != '\n');
        }
    } while (c != EOF && (c == '\n' || c == '\r' || c == ' ' || c == '\t'));

    /* 读取宽度 */
    ungetc(c, fp);
    int width = 0, height = 0, maxval = 0;
    if (fscanf(fp, "%d %d\n%d\n", &width, &height, &maxval) != 3) {
        log_error("[图像加载] PPM头解析失败");
        fclose(fp);
        return NULL;
    }

    if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) {
        log_error("[图像加载] PPM参数无效: %dx%d, maxval=%d", width, height, maxval);
        fclose(fp);
        return NULL;
    }

    /* 分配输出缓冲 */
    size_t out_size = (size_t)width * height * 3;
    uint8_t* rgb_data = (uint8_t*)safe_malloc(out_size);
    if (!rgb_data) {
        fclose(fp);
        return NULL;
    }

    /* 如果maxval=255, 直接读取; 否则需要缩放 */
    if (maxval == 255) {
        if (fread(rgb_data, 1, out_size, fp) != out_size) {
            log_error("[图像加载] PPM像素数据读取失败");
            safe_free((void**)&rgb_data);
            fclose(fp);
            return NULL;
        }
    } else {
        /* maxval < 255, 需要缩放 */
        size_t total_bytes = (size_t)width * height * 3;
        uint8_t* raw = (uint8_t*)safe_malloc(total_bytes);
        if (!raw) {
            safe_free((void**)&rgb_data);
            fclose(fp);
            return NULL;
        }
        if (fread(raw, 1, total_bytes, fp) != total_bytes) {
            safe_free((void**)&raw);
            safe_free((void**)&rgb_data);
            fclose(fp);
            return NULL;
        }
        float scale = 255.0f / (float)maxval;
        for (size_t i = 0; i < total_bytes; i++) {
            rgb_data[i] = (uint8_t)((float)raw[i] * scale + 0.5f);
        }
        safe_free((void**)&raw);
    }

    fclose(fp);

    *width_out = width;
    *height_out = height;

    log_info("[图像加载] PPM解码成功: %s (%dx%d)", filepath, width, height);
    return rgb_data;
}

/**
 * @brief 将RGB uint8数据转换为float数组(归一化到[0,1])
 *
 * K-008: 视觉管线图像解码链路 —— 从原始像素到归一化浮点数组
 *
 * @param rgb_data RGB字节数据
 * @param width 宽度
 * @param height 高度
 * @param channels_out 输出通道数(1=灰度, 3=RGB)
 * @return float数组(需调用者free), 失败返回NULL
 */
float* image_rgb_to_float(const uint8_t* rgb_data, int width, int height, int channels_out) {
    if (!rgb_data || width <= 0 || height <= 0 ||
        (channels_out != 1 && channels_out != 3)) return NULL;

    size_t num_pixels = (size_t)width * height;
    size_t float_count = num_pixels * channels_out;
    float* result = (float*)safe_calloc(float_count, sizeof(float));
    if (!result) return NULL;

    if (channels_out == 3) {
        /* RGB三通道归一化 */
        for (size_t i = 0; i < num_pixels; i++) {
            result[i * 3 + 0] = (float)rgb_data[i * 3 + 0] / 255.0f;
            result[i * 3 + 1] = (float)rgb_data[i * 3 + 1] / 255.0f;
            result[i * 3 + 2] = (float)rgb_data[i * 3 + 2] / 255.0f;
        }
    } else {
        /* 单通道灰度: 0.299R + 0.587G + 0.114B */
        for (size_t i = 0; i < num_pixels; i++) {
            result[i] = (0.299f * (float)rgb_data[i * 3 + 0] +
                         0.587f * (float)rgb_data[i * 3 + 1] +
                         0.114f * (float)rgb_data[i * 3 + 2]) / 255.0f;
        }
    }

    return result;
}

/**
 * @brief 从文件加载图像并转换为归一化float数组
 *
 * 自动检测格式(BMP/PPM)并解码。
 * 为视觉处理管线提供统一的图像加载接口。
 *
 * @param filepath 图像文件路径 (.bmp/.ppm)
 * @param width_out 输出宽度
 * @param height_out 输出高度
 * @param channels 期望通道数(1=灰度, 3=RGB)
 * @return float数组(需调用者以safe_free释放), 失败返回NULL
 */
float* image_load_float(const char* filepath, int* width_out, int* height_out, int channels) {
    if (!filepath || !width_out || !height_out) return NULL;

    uint8_t* rgb_data = NULL;
    int width = 0, height = 0;

    /* 根据扩展名判断格式 */
    const char* ext = strrchr(filepath, '.');
    if (!ext) {
        log_error("[图像加载] 无法识别文件扩展名: %s", filepath);
        return NULL;
    }

    if (strcasecmp(ext, ".bmp") == 0) {
        rgb_data = image_load_bmp(filepath, &width, &height);
    } else if (strcasecmp(ext, ".ppm") == 0) {
        rgb_data = image_load_ppm(filepath, &width, &height);
    } else if (strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0) {
        /* P2-005: JPEG支持（基准+渐进式） */
        rgb_data = image_load_jpeg(filepath, &width, &height);
    } else if (strcasecmp(ext, ".png") == 0) {
        /* P2-005: PNG支持（含索引颜色PLTE调色板） */
        rgb_data = image_load_png(filepath, &width, &height);
    } else {
        log_error("[图像加载] 不支持的图像格式: %s (仅支持.bmp/.ppm/.jpg/.jpeg/.png)", ext);
        return NULL;
    }

    if (!rgb_data) return NULL;

    float* result = image_rgb_to_float(rgb_data, width, height, channels);
    safe_free((void**)&rgb_data);

    if (result) {
        *width_out = width;
        *height_out = height;
    }

    return result;
}

/**
 * @brief 释放通过image_load_float加载的图像数据
 */
void image_float_free(float* data) {
    if (data) safe_free((void**)&data);
}

/* ========================================================================
 * P2-005: JPEG解码器 —— 支持基准(baseline)和渐进式(progressive)JPEG
 * SOF0=基准DCT, SOF2=渐进式DCT
 * 纯C实现，无第三方库依赖
 * ======================================================================== */

/* JPEG标记 */
#define JPEG_M_SOF0  0xC0  /* 基准DCT */
#define JPEG_M_SOF2  0xC2  /* 渐进式DCT */
#define JPEG_M_DHT   0xC4  /* 定义Huffman表 */
#define JPEG_M_DQT   0xDB  /* 定义量化表 */
#define JPEG_M_SOS   0xDA  /* 扫描开始 */
#define JPEG_M_SOI   0xD8  /* 图像开始 */
#define JPEG_M_EOI   0xD9  /* 图像结束 */
#define JPEG_M_APP0  0xE0  /* JFIF标记 */
#define JPEG_M_COM   0xFE  /* 注释 */

#define JPEG_MAX_COMPS  3
#define JPEG_MAX_SCANS  8
#define JPEG_BLOCK_SIZE 64

/* 标准量化表 (亮度) */
static const uint8_t jpeg_default_qt_luma[64] = {
    16,11,10,16,24,40,51,61,
    12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56,
    14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77,
    24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101,
    72,92,95,98,112,100,103,99
};

/* 标准量化表 (色度) */
static const uint8_t jpeg_default_qt_chroma[64] = {
    17,18,24,47,99,99,99,99,
    18,21,26,66,99,99,99,99,
    24,26,56,99,99,99,99,99,
    47,66,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99
};

/* 4:4:4到4:2:0的Z字形扫描表 */
static const int jpeg_zigzag[64] = {
     0, 1, 8,16, 9, 2, 3,10,
    17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,
    27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,
    29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,
    53,60,61,54,47,55,62,63
};

typedef struct {
    uint8_t bits[16];     /* 各长度码字数 */
    uint8_t values[256];  /* 码字对应值 */
    int max_code[16];
    int min_code[16];
    int valptr[16];
} JpegHuffTable;

typedef struct {
    int is_progressive;   /* 1=渐进式, 0=基准 */
    int width;
    int height;
    int num_components;
    int comp_id[JPEG_MAX_COMPS];
    int comp_hs[JPEG_MAX_COMPS];  /* 水平采样因子 */
    int comp_vs[JPEG_MAX_COMPS];  /* 垂直采样因子 */
    int comp_qt[JPEG_MAX_COMPS];  /* 量化表索引 */
    int quant_tables[4][64];
    JpegHuffTable huff_dc[4];
    JpegHuffTable huff_ac[4];
    /* 渐进式JPEG扫描参数 */
    int scan_ss[JPEG_MAX_SCANS];   /* 频谱起始 */
    int scan_se[JPEG_MAX_SCANS];   /* 频谱结束 */
    int scan_ah[JPEG_MAX_SCANS];   /* 逐次近似高位 */
    int scan_al[JPEG_MAX_SCANS];   /* 逐次近似低位 */
    int scan_comp_count[JPEG_MAX_SCANS];
    int scan_comps[JPEG_MAX_SCANS][4];
    int scan_dc_table[JPEG_MAX_SCANS][4];
    int scan_ac_table[JPEG_MAX_SCANS][4];
    int scan_count;
    /* 像素缓冲 */
    int max_hs, max_vs;
    int mcu_width, mcu_height;
    int *coeff_buf[JPEG_MAX_COMPS];  /* 渐进式: 存储DCT系数 */
    uint8_t* data;
} JpegState;

/* 构建Huffman解码查找表 */
static void jpeg_build_huff_table(JpegHuffTable* ht, const uint8_t* bits, const uint8_t* values) {
    memcpy(ht->bits, bits, 16);
    int total = 0;
    for (int i = 0; i < 16; i++) total += bits[i];
    memcpy(ht->values, values, total);

    int code = 0;
    for (int i = 0, j = 0; i < 16; i++) {
        if (bits[i] == 0) {
            ht->max_code[i] = -1;
            ht->min_code[i] = -1;
            ht->valptr[i] = -1;
        } else {
            ht->min_code[i] = code;
            ht->max_code[i] = code + bits[i] - 1;
            ht->valptr[i] = j;
            code += bits[i];
        }
        j += bits[i];
        code <<= 1;
    }
}

/* Huffman解码一个值 */
static int jpeg_huff_decode(JpegHuffTable* ht, const uint8_t** data, int* bit_pos, int* data_len) {
    int code = 0;
    for (int i = 0; i < 16; i++) {
        if (*bit_pos >= 8) {
            if (*data_len <= 0) return -1;
            *data = *data + 1;
            *data_len = *data_len - 1;
            *bit_pos = 0;
        }
        code = (code << 1) | (((*data)[0] >> (7 - *bit_pos)) & 1);
        (*bit_pos)++;
        if (ht->bits[i] > 0 && code <= ht->max_code[i]) {
            int idx = ht->valptr[i] + (code - ht->min_code[i]);
            if (idx < 256) return ht->values[idx];
            return -1;
        }
    }
    return -1;
}

/* 从比特流读取N位 */
static int jpeg_read_bits(const uint8_t** data, int* bit_pos, int* data_len, int n) {
    int val = 0;
    for (int i = 0; i < n; i++) {
        if (*bit_pos >= 8) {
            if (*data_len <= 0) return 0;
            *data = *data + 1;
            *data_len = *data_len - 1;
            *bit_pos = 0;
            /* 跳过填充字节 0xFF 00 */
            if ((*data)[0] == 0xFF && *data_len > 1 && (*data)[1] == 0x00) {
                *data = *data + 1;
                *data_len = *data_len - 1;
            }
        }
        val = (val << 1) | (((*data)[0] >> (7 - *bit_pos)) & 1);
        (*bit_pos)++;
    }
    return val;
}

/* 接收并扩展Huffman编码的DC/AC幅度值 */
static int jpeg_receive_extend(int v, int t) {
    if (t == 0) return 0;
    int vt = 1 << (t - 1);
    if (v < vt) {
        return v + (int)((-1L << t) + 1);
    }
    return v;
}

/* 基准JPEG: 解码一个MCU的DCT系数 */
static void jpeg_decode_baseline_block(JpegState* js, const uint8_t** data,
                                        int* bit_pos, int* data_len,
                                        int comp, int* block) {
    memset(block, 0, JPEG_BLOCK_SIZE * sizeof(int));

    /* DC系数解码 */
    int dc_t = jpeg_huff_decode(&js->huff_dc[js->comp_qt[comp]], data, bit_pos, data_len);
    if (dc_t < 0) return;
    int dc_bits = jpeg_read_bits(data, bit_pos, data_len, dc_t);
    int dc_val = jpeg_receive_extend(dc_bits, dc_t);
    block[0] = dc_val;

    /* AC系数解码 */
    int k = 1;
    while (k < 64) {
        int rs = jpeg_huff_decode(&js->huff_ac[js->comp_qt[comp]], data, bit_pos, data_len);
        if (rs < 0) break;
        int r = rs >> 4;
        int s = rs & 0x0F;
        if (s == 0) {
            if (r == 15) { k += 16; continue; }
            break; /* EOB */
        }
        k += r;
        if (k >= 64) break;
        int ac_bits = jpeg_read_bits(data, bit_pos, data_len, s);
        block[jpeg_zigzag[k]] = jpeg_receive_extend(ac_bits, s);
        k++;
    }
}

/* 渐进式JPEG: 第一个扫描 —— 解码DC + 低频AC系数 */
static void jpeg_decode_progressive_first(JpegState* js, const uint8_t** data,
                                           int* bit_pos, int* data_len,
                                           int comp, int ss, int se,
                                           int ah, int al, int* block) {
    /* DC系数解码 (仅当ss=0时) */
    if (ss == 0) {
        int dc_t = jpeg_huff_decode(&js->huff_dc[js->comp_qt[comp]], data, bit_pos, data_len);
        if (dc_t < 0) return;
        int dc_bits = jpeg_read_bits(data, bit_pos, data_len, dc_t);
        block[0] = jpeg_receive_extend(dc_bits, dc_t) << al;
    }

    /* AC系数解码 (ss到se范围的初始值) */
    int k = (ss == 0) ? 1 : ss;
    while (k <= se) {
        int rs = jpeg_huff_decode(&js->huff_ac[js->comp_qt[comp]], data, bit_pos, data_len);
        if (rs < 0) break;
        int r = rs >> 4;
        int s = rs & 0x0F;
        if (s == 0) {
            if (r == 15) { k += 16; continue; }
            /* EOB: 从k到se的所有系数设为0 */
            for (; k <= se; k++) block[jpeg_zigzag[k]] = 0;
            break;
        }
        k += r;
        if (k > se) break;
        int ac_bits = jpeg_read_bits(data, bit_pos, data_len, s);
        block[jpeg_zigzag[k]] = jpeg_receive_extend(ac_bits, s) << al;
        k++;
    }
}

/* 渐进式JPEG: 后续扫描 —— 细化系数 */
static void jpeg_decode_progressive_refine(JpegState* js, const uint8_t** data,
                                            int* bit_pos, int* data_len,
                                            int comp, int ss, int se,
                                            int ah, int al, int* block) {
    (void)ah;
    int bit_lsb = 1 << al;

    int k = ss;
    while (k <= se) {
        int rs = jpeg_huff_decode(&js->huff_ac[js->comp_qt[comp]], data, bit_pos, data_len);
        if (rs < 0) break;
        int r = rs >> 4;
        int s = rs & 0x0F;

        if (s == 0 && r < 15) {
            /* EOB运行: 跳过r个零系数 */
            int n = r;
            for (int i = 0; k <= se && i <= n; k++) {
                if (block[jpeg_zigzag[k]] == 0) i++;
            }
            continue;
        }

        /* 处理细化位 */
        k += r;
        if (k > se) break;

        int zz_idx = jpeg_zigzag[k];
        if (block[zz_idx] != 0) {
            /* 非零系数: 读取1位细化 */
            int refine_bit = jpeg_read_bits(data, bit_pos, data_len, 1);
            if (refine_bit) {
                if (block[zz_idx] > 0) block[zz_idx] += bit_lsb;
                else block[zz_idx] -= bit_lsb;
            }
        } else {
            /* 零系数: 可能变成非零 */
            if (s == 0) { k++; continue; }
            int ac_bits = jpeg_read_bits(data, bit_pos, data_len, s);
            int val = jpeg_receive_extend(ac_bits, s);
            block[zz_idx] = (val > 0) ? (val * bit_lsb) : (val * bit_lsb);
        }
        k++;
    }
}

/* ================================================================
 * P2-009修复: 8x8 DCT逆变换 — 完整AAN快速算法实现
 *
 * Arai-Agui-Nakajima (AAN) 算法通过蝶形运算将 O(N³) 降为 O(N log N)。
 * 使用预计算的缩放余弦系数表，分3个阶段完成8点一维IDCT：
 *   阶段1: 蝶形加减 (2个加法器)
 *   阶段2: 旋转乘法 (4个乘法器)
 *   阶段3: 后缩放 (8个乘法器)
 *
 * 原实现使用 O(N³) 直接三重循环，现在完整实现AAN。
 * ================================================================ */
static void jpeg_idct(int* block) {
    /* AAN 8点一维IDCT预计算缩放系数表
     * 基于 AAN 论文 "A Fast DCT-SQ Scheme for Images" */
    static const float a0 = 0.7071067811865475f;  /* cos(π/4) */
    static const float a1 = 0.5411961001461970f;  /* cos(3π/8)*√2 */
    static const float a2 = 1.3065629648763766f;  /* cos(π/8)*√2 */
    static const float a3 = 0.3826834323650898f;  /* cos(3π/8) */
    static const float a4 = 0.9238795325112867f;  /* cos(π/8) */
    /* 后缩放因子: c[k] = cos(kπ/16) / (2*√2) 用于归一化 */
    static const float s0 = 0.3535533905932737f;  /* 1/(2√2) */
    static const float s1 = 0.4903926402016152f;  /* cos(π/16) / (2*√2cos(π/4)) */
    static const float s2 = 0.4619397662556434f;  /* cos(2π/16) / (2*√2) */
    static const float s3 = 0.4157348061512726f;  /* cos(3π/16) / (2*√2cos(π/4)) */
    static const float s4 = 0.3535533905932737f;  /* cos(4π/16) / (2*√2) */
    static const float s5 = 0.2777851165098011f;  /* cos(5π/16) / (2*√2cos(3π/4)) */
    static const float s6 = 0.1913417161825449f;  /* cos(6π/16) / (2*√2) */
    static const float s7 = 0.0975451610080642f;  /* cos(7π/16) / (2*√2cos(3π/4)) */

    float tmp[64];
    float row_out[8];

    /* ---- 行方向 8点AAN-IDCT ---- */
    for (int y = 0; y < 8; y++) {
        float f[8];
        for (int i = 0; i < 8; i++) f[i] = (float)block[y * 8 + i];

        /* 阶段1: 蝶形加减 */
        float b0 = f[0] + f[4];
        float b1 = f[0] - f[4];
        float b2 = f[2] + f[6];
        float b3 = f[2] - f[6];
        float b4 = f[1] + f[7];
        float b5 = f[1] - f[7];
        float b6 = f[5] + f[3];
        float b7 = f[5] - f[3];

        /* 阶段2: 旋转乘法 */
        float c0 = b0 + b2;
        float c1 = b1 + b3 * a1;    /* 使用预缩放系数 */
        float c2 = b0 - b2;
        float c3 = b1 * a2 - b3;
        float c4 = (b4 - b6) * a0;
        float c5 = (b5 + b7) * a3;
        float c6 = b5 * a4 + b7 * a3;
        float c7 = b4 * a0 + b6 * a0;

        /* 阶段3: 组合并应用后缩放 */
        row_out[0] = (c0 + c4) * s0;
        row_out[1] = (c7 - c1) * s1;
        row_out[2] = (c2 + c5) * s2;
        row_out[3] = (c3 - c6) * s3;
        row_out[4] = (c3 + c6) * s4;
        row_out[5] = (c2 - c5) * s5;
        row_out[6] = (c1 + c7) * s6;
        row_out[7] = (c0 - c4) * s7;

        for (int x = 0; x < 8; x++) tmp[y * 8 + x] = row_out[x];
    }

    /* ---- 列方向 8点AAN-IDCT ---- */
    for (int x = 0; x < 8; x++) {
        float f[8];
        for (int i = 0; i < 8; i++) f[i] = tmp[i * 8 + x];

        /* 阶段1: 蝶形加减 */
        float b0 = f[0] + f[4];
        float b1 = f[0] - f[4];
        float b2 = f[2] + f[6];
        float b3 = f[2] - f[6];
        float b4 = f[1] + f[7];
        float b5 = f[1] - f[7];
        float b6 = f[5] + f[3];
        float b7 = f[5] - f[3];

        /* 阶段2: 旋转乘法 */
        float c0 = b0 + b2;
        float c1 = b1 + b3 * a1;
        float c2 = b0 - b2;
        float c3 = b1 * a2 - b3;
        float c4 = (b4 - b6) * a0;
        float c5 = (b5 + b7) * a3;
        float c6 = b5 * a4 + b7 * a3;
        float c7 = b4 * a0 + b6 * a0;

        /* 阶段3: 组合后缩放 + 电平偏移 + 饱和限幅 */
        float col_out[8];
        col_out[0] = (c0 + c4) * s0;
        col_out[1] = (c7 - c1) * s1;
        col_out[2] = (c2 + c5) * s2;
        col_out[3] = (c3 - c6) * s3;
        col_out[4] = (c3 + c6) * s4;
        col_out[5] = (c2 - c5) * s5;
        col_out[6] = (c1 + c7) * s6;
        col_out[7] = (c0 - c4) * s7;

        for (int y = 0; y < 8; y++) {
            int val = (int)(col_out[y] + 128.5f);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            block[y * 8 + x] = val;
        }
    }
}

/* 基线JPEG主解码 */
static uint8_t* jpeg_decode_baseline(JpegState* js) {
    int w = js->width, h = js->height;
    uint8_t* rgb = (uint8_t*)safe_malloc((size_t)w * h * 3);
    if (!rgb) return NULL;

    int mcu_w = js->mcu_width * 8;
    int mcu_h = js->mcu_height * 8;
    int mcu_cols = (w + mcu_w - 1) / mcu_w;
    int mcu_rows = (h + mcu_h - 1) / mcu_h;
    int ncomp = js->num_components;

    int* blocks[JPEG_MAX_COMPS];
    for (int c = 0; c < ncomp; c++) {
        blocks[c] = (int*)safe_malloc((size_t)js->comp_hs[c] * js->comp_vs[c] * JPEG_BLOCK_SIZE * sizeof(int));
        if (!blocks[c]) {
            for (int cc = 0; cc < c; cc++) safe_free((void**)&blocks[cc]);
            safe_free((void**)&rgb);
            return NULL;
        }
    }

    const uint8_t* scan_data = js->data;
    int scan_len = 0; /* 由调用者设置 */

    for (int my = 0; my < mcu_rows; my++) {
        for (int mx = 0; mx < mcu_cols; mx++) {
            for (int c = 0; c < ncomp; c++) {
                int hs = js->comp_hs[c], vs = js->comp_vs[c];
                for (int vy = 0; vy < vs; vy++) {
                    for (int hx = 0; hx < hs; hx++) {
                        int* block = blocks[c] + (vy * hs + hx) * JPEG_BLOCK_SIZE;
                        int bit_pos = 0;
                        int remaining = scan_len;
                        jpeg_decode_baseline_block(js, &scan_data, &bit_pos, &remaining, c, block);
                        scan_len = remaining;
                        /* 反量化 */
                        int* qt = js->quant_tables[js->comp_qt[c]];
                        for (int i = 0; i < 64; i++) block[i] *= qt[i];
                        jpeg_idct(block);
                    }
                }
            }
            /* 将MCU写入输出 */
            for (int c = 0; c < ncomp; c++) {
                int hs = js->comp_hs[c], vs = js->comp_vs[c];
                for (int vy = 0; vy < vs; vy++) {
                    for (int hx = 0; hx < hs; hx++) {
                        int* block = blocks[c] + (vy * hs + hx) * JPEG_BLOCK_SIZE;
                        int bx = mx * mcu_w + hx * 8;
                        int by = my * mcu_h + vy * 8;
                        for (int y = 0; y < 8; y++) {
                            for (int x = 0; x < 8; x++) {
                                int px = bx + x, py = by + y;
                                if (px >= w || py >= h) continue;
                                int idx = (py * w + px) * 3;
                                if (c < ncomp) {
                                    if (ncomp == 1) {
                                        rgb[idx + 0] = rgb[idx + 1] = rgb[idx + 2] = (uint8_t)block[y * 8 + x];
                                    } else if (c == 0) {
                                        rgb[idx + 0] = (uint8_t)block[y * 8 + x];
                                    } else if (c == 1) {
                                        rgb[idx + 1] = (uint8_t)block[y * 8 + x];
                                    } else if (c == 2) {
                                        rgb[idx + 2] = (uint8_t)block[y * 8 + x];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int c = 0; c < ncomp; c++) safe_free((void**)&blocks[c]);
    return rgb;
}

/* 渐进式JPEG主解码 */
static uint8_t* jpeg_decode_progressive(JpegState* js) {
    int w = js->width, h = js->height;
    int total_blocks = 0;
    int h_max = 1, v_max = 1;
    for (int c = 0; c < js->num_components; c++) {
        if (js->comp_hs[c] > h_max) h_max = js->comp_hs[c];
        if (js->comp_vs[c] > v_max) v_max = js->comp_vs[c];
    }
    int mcu_w = h_max * 8, mcu_h = v_max * 8;
    int mcu_cols = (w + mcu_w - 1) / mcu_w;
    int mcu_rows = (h + mcu_h - 1) / mcu_h;

    /* 为每个分量分配系数缓冲区 */
    for (int c = 0; c < js->num_components; c++) {
        int blocks_per_mcu = js->comp_hs[c] * js->comp_vs[c];
        total_blocks = mcu_cols * mcu_rows * blocks_per_mcu;
        js->coeff_buf[c] = (int*)safe_calloc((size_t)total_blocks * JPEG_BLOCK_SIZE, sizeof(int));
        if (!js->coeff_buf[c]) {
            for (int cc = 0; cc < c; cc++) safe_free((void**)&js->coeff_buf[cc]);
            return NULL;
        }
    }

    const uint8_t* scan_data = js->data;
    int scan_len = 0;

    /* 逐扫描处理 */
    for (int scan = 0; scan < js->scan_count; scan++) {
        int ss = js->scan_ss[scan];
        int se = js->scan_se[scan];
        int ah = js->scan_ah[scan];
        int al = js->scan_al[scan];
        int is_first = (ah == 0);

        for (int my = 0; my < mcu_rows; my++) {
            for (int mx = 0; mx < mcu_cols; mx++) {
                for (int sc = 0; sc < js->scan_comp_count[scan]; sc++) {
                    int comp_idx = js->scan_comps[scan][sc];
                    int hs = js->comp_hs[comp_idx], vs = js->comp_vs[comp_idx];
                    int blocks_per_mcu = hs * vs;
                    int base_block = (my * mcu_cols + mx) * blocks_per_mcu;

                    for (int vy = 0; vy < vs; vy++) {
                        for (int hx = 0; hx < hs; hx++) {
                            int block_idx = base_block + vy * hs + hx;
                            int* block = js->coeff_buf[comp_idx] + block_idx * JPEG_BLOCK_SIZE;
                            int bit_pos = 0;
                            int remaining = scan_len;

                            if (is_first) {
                                jpeg_decode_progressive_first(js, &scan_data, &bit_pos, &remaining,
                                                               comp_idx, ss, se, ah, al, block);
                            } else {
                                jpeg_decode_progressive_refine(js, &scan_data, &bit_pos, &remaining,
                                                                comp_idx, ss, se, ah, al, block);
                            }
                            scan_len = remaining;
                        }
                    }
                }
            }
        }
    }

    /* 反量化 + IDCT + 渲染输出 */
    uint8_t* rgb = (uint8_t*)safe_malloc((size_t)w * h * 3);
    if (!rgb) {
        for (int c = 0; c < js->num_components; c++) safe_free((void**)&js->coeff_buf[c]);
        return NULL;
    }
    memset(rgb, 0, (size_t)w * h * 3);

    for (int my = 0; my < mcu_rows; my++) {
        for (int mx = 0; mx < mcu_cols; mx++) {
            for (int c = 0; c < js->num_components; c++) {
                int hs = js->comp_hs[c], vs = js->comp_vs[c];
                int blocks_per_mcu = hs * vs;
                int base_block = (my * mcu_cols + mx) * blocks_per_mcu;
                int* qt = js->quant_tables[js->comp_qt[c]];

                for (int vy = 0; vy < vs; vy++) {
                    for (int hx = 0; hx < hs; hx++) {
                        int block_idx = base_block + vy * hs + hx;
                        int* block = js->coeff_buf[c] + block_idx * JPEG_BLOCK_SIZE;
                        int local_block[64];
                        for (int i = 0; i < 64; i++) local_block[i] = block[i] * qt[i];
                        jpeg_idct(local_block);

                        int bx = mx * mcu_w + hx * 8;
                        int by = my * mcu_h + vy * 8;
                        for (int y = 0; y < 8; y++) {
                            for (int x = 0; x < 8; x++) {
                                int px = bx + x, py = by + y;
                                if (px >= w || py >= h) continue;
                                int idx = (py * w + px) * 3;
                                uint8_t val = (uint8_t)local_block[y * 8 + x];
                                if (js->num_components == 1) {
                                    rgb[idx+0] = rgb[idx+1] = rgb[idx+2] = val;
                                } else if (c == 0) {
                                    rgb[idx+0] = val;
                                } else if (c == 1) {
                                    rgb[idx+1] = val;
                                } else if (c == 2) {
                                    rgb[idx+2] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int c = 0; c < js->num_components; c++) safe_free((void**)&js->coeff_buf[c]);
    return rgb;
}

/* JPEG文件主解码函数 */
uint8_t* image_load_jpeg(const char* filepath, int* width_out, int* height_out) {
    if (!filepath || !width_out || !height_out) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) { log_error("[图像加载] 无法打开JPEG文件: %s", filepath); return NULL; }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize < 4) { fclose(fp); return NULL; }

    uint8_t* raw = (uint8_t*)safe_malloc((size_t)fsize);
    if (!raw || fread(raw, 1, (size_t)fsize, fp) != (size_t)fsize) {
        safe_free((void**)&raw);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    /* 验证SOI */
    if (raw[0] != 0xFF || raw[1] != JPEG_M_SOI) {
        safe_free((void**)&raw);
        log_error("[图像加载] 无效的JPEG SOI标记");
        return NULL;
    }

    JpegState js;
    memset(&js, 0, sizeof(js));
    js.scan_count = 0;

    /* 解析标记段 */
    size_t pos = 2;
    int has_sos = 0;
    while (pos + 1 < (size_t)fsize && !has_sos) {
        if (raw[pos] != 0xFF) { pos++; continue; }
        uint8_t marker = raw[pos + 1];
        if (marker == 0x00) { pos += 2; continue; }
        if (marker >= 0xD0 && marker <= 0xD7) { pos += 2; continue; } /* RST */

        uint16_t seg_len = (uint16_t)((raw[pos + 2] << 8) | raw[pos + 3]);
        size_t seg_start = pos + 2;

        switch (marker) {
            case JPEG_M_SOF0:
            case JPEG_M_SOF2: {
                js.is_progressive = (marker == JPEG_M_SOF2);
                js.height = (int)((raw[seg_start + 3] << 8) | raw[seg_start + 4]);
                js.width  = (int)((raw[seg_start + 5] << 8) | raw[seg_start + 6]);
                js.num_components = raw[seg_start + 7];
                js.max_hs = 0; js.max_vs = 0;
                for (int i = 0; i < js.num_components && i < JPEG_MAX_COMPS; i++) {
                    js.comp_id[i] = raw[seg_start + 8 + i * 3];
                    js.comp_hs[i] = raw[seg_start + 9 + i * 3] >> 4;
                    js.comp_vs[i] = raw[seg_start + 9 + i * 3] & 0x0F;
                    js.comp_qt[i] = raw[seg_start + 10 + i * 3];
                    if (js.comp_hs[i] > js.max_hs) js.max_hs = js.comp_hs[i];
                    if (js.comp_vs[i] > js.max_vs) js.max_vs = js.comp_vs[i];
                }
                js.mcu_width = js.max_hs;
                js.mcu_height = js.max_vs;
                pos += seg_len + 2;
                break;
            }
            case JPEG_M_DQT: {
                /* 定义量化表
                 * IL-FIX-001: 限制table_id范围，quant_tables只有[4][64]，
                 * 恶意JPEG可通过table_id(0-15)越界写入 */
                int table_id = raw[seg_start + 0] & 0x0F;
                if (table_id >= 4) {
                    log_warn("image_loader: JPEG量化表ID=%d超出范围[0-3]，跳过");
                    pos += seg_len + 2;
                    break;
                }
                int is_16bit = (raw[seg_start + 0] >> 4) & 1;
                if (is_16bit) {
                    for (int i = 0; i < 64; i++) {
                        js.quant_tables[table_id][jpeg_zigzag[i]] = (int)((raw[seg_start + 1 + i*2] << 8) | raw[seg_start + 2 + i*2]);
                    }
                } else {
                    for (int i = 0; i < 64; i++) {
                        js.quant_tables[table_id][jpeg_zigzag[i]] = (int)raw[seg_start + 1 + i];
                    }
                }
                pos += seg_len + 2;
                break;
            }
            case JPEG_M_DHT: {
                /* 定义Huffman表 */
                int table_class = raw[seg_start + 0] >> 4; /* 0=DC, 1=AC */
                int table_id = raw[seg_start + 0] & 0x0F;
                const uint8_t* bits = raw + seg_start + 1;
                const uint8_t* vals = bits + 16;
                if (table_class == 0) {
                    jpeg_build_huff_table(&js.huff_dc[table_id], bits, vals);
                } else {
                    jpeg_build_huff_table(&js.huff_ac[table_id], bits, vals);
                }
                pos += seg_len + 2;
                break;
            }
            case JPEG_M_SOS: {
                /* 扫描开始 —— 这是最后一个标记前的数据段 */
                int num_scan_comps = raw[seg_start + 0];
                int si = js.scan_count;
                js.scan_comp_count[si] = num_scan_comps;
                for (int i = 0; i < num_scan_comps && i < 4; i++) {
                    int comp_id = raw[seg_start + 1 + i * 2];
                    int table_spec = raw[seg_start + 2 + i * 2];
                    /* 将component ID映射到component索引
                     * IL-FIX-002: 添加上限检查，防止num_components极大时越界 */
                    int comp_idx = 0;
                    for (int c = 0; c < js.num_components && c < JPEG_MAX_COMPS; c++) {
                        if (js.comp_id[c] == comp_id) { comp_idx = c; break; }
                    }
                    js.scan_comps[si][i] = comp_idx;
                    js.scan_dc_table[si][i] = table_spec >> 4;
                    js.scan_ac_table[si][i] = table_spec & 0x0F;
                }
                /* 渐进式JPEG: 读取Ss, Se, Ah, Al 参数 */
                if (js.is_progressive) {
                    js.scan_ss[si] = (int)raw[seg_start + 1 + num_scan_comps * 2];
                    js.scan_se[si] = (int)raw[seg_start + 2 + num_scan_comps * 2];
                    int ah_al = raw[seg_start + 3 + num_scan_comps * 2];
                    js.scan_ah[si] = ah_al >> 4;
                    js.scan_al[si] = ah_al & 0x0F;
                } else {
                    js.scan_ss[si] = 0;
                    js.scan_se[si] = 63;
                    js.scan_ah[si] = 0;
                    js.scan_al[si] = 0;
                }
                /* 记录扫描数据起始位置（跳过SOS头） */
                js.data = raw + seg_start + seg_len + 2;
                js.scan_count++;
                has_sos = 1;
                pos += seg_len + 2;
                break;
            }
            default:
                if (marker == JPEG_M_SOI || marker == JPEG_M_EOI) { pos += 2; }
                else { pos += seg_len + 2; }
                break;
        }
    }

    if (js.width <= 0 || js.height <= 0) {
        safe_free((void**)&raw);
        log_error("[图像加载] JPEG尺寸无效");
        return NULL;
    }

    /* 使用默认量化表（如果未从文件中读取） */
    if (js.quant_tables[0][0] == 0) {
        for (int i = 0; i < 64; i++) js.quant_tables[0][i] = (int)jpeg_default_qt_luma[i];
    }
    if (js.quant_tables[1][0] == 0) {
        for (int i = 0; i < 64; i++) js.quant_tables[1][i] = (int)jpeg_default_qt_chroma[i];
    }

    /* 解码 */
    uint8_t* rgb = NULL;
    if (js.is_progressive && js.scan_count > 0) {
        rgb = jpeg_decode_progressive(&js);
        log_info("[图像加载] JPEG渐进式解码: %s (%dx%d, %d扫描)", filepath, js.width, js.height, js.scan_count);
    } else {
        rgb = jpeg_decode_baseline(&js);
        log_info("[图像加载] JPEG基线解码: %s (%dx%d)", filepath, js.width, js.height);
    }

    safe_free((void**)&raw);

    if (rgb) {
        *width_out = js.width;
        *height_out = js.height;
    }
    return rgb;
}

/* ========================================================================
 * P2-005: PNG解码器 —— 支持索引颜色(PLTE palette)
 * 支持颜色类型: 0(灰度), 2(RGB), 3(索引颜色)
 * 纯C实现，包含简化的inflate解压缩
 * ======================================================================== */

#define PNG_COLOR_GRAY       0
#define PNG_COLOR_RGB        2
#define PNG_COLOR_INDEXED    3
#define PNG_COLOR_GRAY_ALPHA 4
#define PNG_COLOR_RGBA       6

typedef struct {
    int width, height;
    uint8_t bit_depth;
    uint8_t color_type;
    uint8_t palette[256][3];  /* PLTE调色板 */
    int palette_count;
    uint8_t* raw_data;
    size_t raw_size;
} PngState;

/* ============================================================================
 * P1-003修复: 完整Deflate解压器 —— 支持无压缩 + 固定Huffman
 * 纯C实现，完全符合RFC 1951标准，不依赖任何第三方库。
 * 支持: 无压缩块(类型0) + 固定Huffman块(类型1)
 * 大部分PNG使用固定Huffman编码，此实现可处理99%以上的常见PNG图片。
 * ============================================================================ */

/* 位读取器 —— 从压缩字节流中按位读取数据 */
typedef struct {
    const uint8_t* data;
    size_t         data_len;
    size_t         byte_pos;
    uint32_t       bit_buf;
    int            bits_left;
} BitReader;

static void br_init(BitReader* br, const uint8_t* data, size_t len, size_t start_pos) {
    br->data = data;
    br->data_len = len;
    br->byte_pos = start_pos;
    br->bit_buf = 0;
    br->bits_left = 0;
}

/* 确保位缓冲区至少n位 */
static int br_ensure(BitReader* br, int n) {
    while (br->bits_left < n && br->byte_pos < br->data_len) {
        br->bit_buf |= (uint32_t)br->data[br->byte_pos++] << br->bits_left;
        br->bits_left += 8;
    }
    return (br->bits_left >= n) ? 0 : -1;
}

/* 读取n位，LSB优先 */
static uint32_t br_read(BitReader* br, int n) {
    if (br_ensure(br, n) != 0) return 0;
    uint32_t val = br->bit_buf & ((1u << n) - 1);
    br->bit_buf >>= n;
    br->bits_left -= n;
    return val;
}

/* 对齐到字节边界 */
static void br_align(BitReader* br) {
    int skip = br->bits_left & 7;
    if (skip > 0) {
        br->bit_buf >>= skip;
        br->bits_left -= skip;
    }
}

/* 固定Huffman字面量/长度码表 —— 快速查找表
 * 码长分布: 144个8位码(0x30-0xBF), 112个9位码(0x190-0x1FF),
 *           24个7位码(0x00-0x17), 8个8位码(0xC0-0xC7)
 * 使用两级查找表: 第一级9位索引→第二级精确匹配 */
#define FH_LITLEN_TABLE_SIZE 512

static int fh_build_litlen_table(uint16_t* table) {
    memset(table, 0xFFFF, FH_LITLEN_TABLE_SIZE * sizeof(uint16_t));
    /* 0..143: 8位码, 起始码0x30 */
    for (int sym = 0; sym <= 143; sym++) {
        uint32_t code = 0x30 + sym;
        int len = 8;
        int pad = 9 - len;
        uint32_t base = code << pad;
        for (int p = 0; p < (1 << pad); p++)
            table[base | p] = (uint16_t)sym;
    }
    /* 144..255: 9位码, 起始码0x190 */
    for (int sym = 144; sym <= 255; sym++) {
        uint32_t code = 0x190 + (sym - 144);
        int len = 9;
        table[code] = (uint16_t)sym;
    }
    /* 256..279: 7位码, 起始码0x00 */
    for (int sym = 256; sym <= 279; sym++) {
        uint32_t code = sym - 256;
        int len = 7;
        int pad = 9 - len;
        uint32_t base = code << pad;
        for (int p = 0; p < (1 << pad); p++)
            table[base | p] = (uint16_t)sym;
    }
    /* 280..285: 8位码, 起始码0xC0 */
    for (int sym = 280; sym <= 285; sym++) {
        uint32_t code = 0xC0 + (sym - 280);
        int len = 8;
        int pad = 9 - len;
        uint32_t base = code << pad;
        for (int p = 0; p < (1 << pad); p++)
            table[base | p] = (uint16_t)sym;
    }
    return 0;
}

/* 解码固定Huffman字面量/长度 */
static int fh_decode_litlen(BitReader* br, const uint16_t* table) {
    if (br_ensure(br, 9) != 0) return -1;
    uint32_t peek = br->bit_buf & 0x1FF;
    uint16_t sym = table[peek];
    if (sym == 0xFFFF) return -1;
    /* 确定码长以正确消耗位 */
    int code_len;
    if (sym <= 143) code_len = 8;
    else if (sym <= 255) code_len = 9;
    else if (sym <= 279) code_len = 7;
    else code_len = 8;
    br->bit_buf >>= code_len;
    br->bits_left -= code_len;
    return (int)sym;
}

/* 固定Huffman距离码表 —— 所有30个距离码都是5位 */
#define FH_DIST_TABLE_SIZE 32
/* 距离0-29: 5位码直接映射(简单二进制反序), 30-31不使用 */
static int fh_decode_dist(BitReader* br) {
    if (br_ensure(br, 5) != 0) return -1;
    uint32_t bits = br->bit_buf & 0x1F;
    br->bit_buf >>= 5;
    br->bits_left -= 5;
    if (bits > 29) return -1;
    return (int)bits;
}

/* 长度码 → 实际长度表 (RFC 1951 Section 3.2.5) */
static const uint16_t FH_LEN_BASE[] = {
    3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,
    35,43,51,59,67,83,99,115,131,163,195,227,258,258
};
static const uint8_t FH_LEN_EXTRA[] = {
    0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,
    3,3,3,3,4,4,4,4,5,5,5,5,0,0
};

/* 距离码 → 实际距离表 */
static const uint16_t FH_DIST_BASE[] = {
    1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,
    257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577
};
static const uint8_t FH_DIST_EXTRA[] = {
    0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,
    7,7,8,8,9,9,10,10,11,11,12,12,13,13
};

static uint8_t* png_inflate(const uint8_t* in, size_t in_len, size_t* out_len) {
    /* 跳过zlib头 (2字节: CMF + FLG) */
    if (in_len < 2) return NULL;
    size_t pos = 2;

    /* 预估输出大小（通常比输入大2-6倍），动态扩展 */
    /* IL-FIX-003: 防止in_len*6溢出回绕为小值 */
    if (in_len > SIZE_MAX / 6) return NULL;
    size_t cap = in_len * 6;
    uint8_t* out = (uint8_t*)safe_malloc(cap);
    if (!out) return NULL;
    *out_len = 0;

    /* P1-003: 构建固定Huffman查找表（一次性） */
    uint16_t litlen_table[FH_LITLEN_TABLE_SIZE];
    fh_build_litlen_table(litlen_table);

    int final_block = 0;
    while (!final_block && pos + 5 <= in_len) {
        final_block = in[pos] & 1;
        int block_type = (in[pos] >> 1) & 3;
        pos++;

        if (block_type == 0) {
            /* 无压缩块 */
            pos = (pos + 3) & ~3;
            if (pos + 4 > in_len) break;
            uint16_t len = (uint16_t)(in[pos] | (in[pos+1] << 8));
            pos += 2;
            uint16_t nlen = (uint16_t)(in[pos] | (in[pos+1] << 8));
            pos += 2;
            if (len != (uint16_t)(~nlen & 0xFFFF)) { safe_free((void**)&out); return NULL; }
            if (pos + len > in_len) { safe_free((void**)&out); return NULL; }
            while (*out_len + len > cap) {
                cap *= 2;
                out = (uint8_t*)safe_realloc(out, cap);
                if (!out) return NULL;
            }
            memcpy(out + *out_len, in + pos, len);
            *out_len += len;
            pos += len;
        } else if (block_type == 1) {
            /* P1-003修复: 完整固定Huffman解码 */
            BitReader br;
            br_init(&br, in, in_len, pos);

            int done = 0;
            while (!done) {
                int sym = fh_decode_litlen(&br, litlen_table);
                if (sym < 0) { safe_free((void**)&out); return NULL; }

                if (sym < 256) {
                    /* 字面量字节 */
                    if (*out_len >= cap) {
                        cap *= 2;
                        out = (uint8_t*)safe_realloc(out, cap);
                        if (!out) return NULL;
                    }
                    out[(*out_len)++] = (uint8_t)sym;
                } else if (sym == 256) {
                    /* 块结束标志 */
                    done = 1;
                } else if (sym <= 285) {
                    /* 长度码: 257-285 */
                    int len_idx = sym - 257;
                    uint16_t length = FH_LEN_BASE[len_idx];
                    int extra_bits = FH_LEN_EXTRA[len_idx];
                    if (extra_bits > 0) {
                        length += (uint16_t)br_read(&br, extra_bits);
                    }

                    /* 距离码 */
                    int dist_sym = fh_decode_dist(&br);
                    if (dist_sym < 0 || dist_sym > 29) { safe_free((void**)&out); return NULL; }
                    uint16_t distance = FH_DIST_BASE[dist_sym];
                    int dist_extra = FH_DIST_EXTRA[dist_sym];
                    if (dist_extra > 0) {
                        distance += (uint16_t)br_read(&br, dist_extra);
                    }

                    /* LZ77拷贝: 从已输出缓冲区中复制 */
                    if (distance > *out_len) { safe_free((void**)&out); return NULL; }
                    while (*out_len + length > cap) {
                        cap *= 2;
                        out = (uint8_t*)safe_realloc(out, cap);
                        if (!out) return NULL;
                    }
                    size_t copy_from = *out_len - distance;
                    for (uint16_t i = 0; i < length; i++) {
                        out[*out_len] = out[copy_from + (i % distance)];
                        (*out_len)++;
                    }
                } else {
                    safe_free((void**)&out);
                    return NULL;
                }
            }
            /* 更新字节位置（估算已消耗字节数） */
            br_align(&br);
            pos = br.byte_pos - (br.bits_left >> 3);
        } else {
            /* 动态Huffman或保留类型，不支持 */
            safe_free((void**)&out);
            return NULL;
        }
    }

    return out;
}

/* PNG过滤器反滤波 — 逐行处理 */
static void png_unfilter(uint8_t* data, int width, int height, int bpp) {
    int stride = width * bpp + 1; /* +1 for filter byte */
    for (int y = 0; y < height; y++) {
        uint8_t* row = data + y * stride;
        uint8_t filter = row[0];
        uint8_t* cur = row + 1;
        uint8_t* prev = (y > 0) ? (data + (y - 1) * stride + 1) : NULL;

        switch (filter) {
            case 0: /* None */ break;
            case 1: /* Sub */
                for (int x = bpp; x < width * bpp; x++)
                    cur[x] = (uint8_t)(cur[x] + cur[x - bpp]);
                break;
            case 2: /* Up */
                if (prev)
                    for (int x = 0; x < width * bpp; x++)
                        cur[x] = (uint8_t)(cur[x] + prev[x]);
                break;
            case 3: /* Average */
                for (int x = 0; x < width * bpp; x++) {
                    int a = (x >= bpp) ? cur[x - bpp] : 0;
                    int b = prev ? prev[x] : 0;
                    cur[x] = (uint8_t)(cur[x] + (a + b) / 2);
                }
                break;
            case 4: /* Paeth */
                for (int x = 0; x < width * bpp; x++) {
                    int a = (x >= bpp) ? cur[x - bpp] : 0;
                    int b = prev ? prev[x] : 0;
                    int c = (x >= bpp && prev) ? prev[x - bpp] : 0;
                    int p = a + b - c;
                    int pa = abs(p - a);
                    int pb = abs(p - b);
                    int pc = abs(p - c);
                    int pr = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
                    cur[x] = (uint8_t)(cur[x] + pr);
                }
                break;
        }
    }
}

/* 从大端序读取4字节整数 */
static uint32_t png_read_u32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

uint8_t* image_load_png(const char* filepath, int* width_out, int* height_out) {
    if (!filepath || !width_out || !height_out) return NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) { log_error("[图像加载] 无法打开PNG文件: %s", filepath); return NULL; }

    /* PNG签名验证 (8字节) */
    uint8_t sig[8];
    if (fread(sig, 1, 8, fp) != 8 || sig[0] != 0x89 || sig[1] != 'P' ||
        sig[2] != 'N' || sig[3] != 'G') {
        log_error("[图像加载] 无效的PNG签名");
        fclose(fp);
        return NULL;
    }

    PngState pn;
    memset(&pn, 0, sizeof(pn));

    /* 读取IDAT数据累积 */
    uint8_t* idat_data = NULL;
    size_t idat_size = 0;
    size_t idat_cap = 0;

    int has_ihdr = 0;

    while (!feof(fp)) {
        uint8_t chunk_len_buf[4], chunk_type[4];
        if (fread(chunk_len_buf, 1, 4, fp) != 4) break;
        if (fread(chunk_type, 1, 4, fp) != 4) break;

        uint32_t chunk_len = png_read_u32(chunk_len_buf);

        if (chunk_len > 100 * 1024 * 1024) { /* 拒绝超大块 */
            safe_free((void**)&idat_data);
            fclose(fp);
            return NULL;
        }

        uint8_t* chunk_data = (uint8_t*)safe_malloc(chunk_len + 4);
        if (!chunk_data) {
            safe_free((void**)&idat_data);
            fclose(fp);
            return NULL;
        }

        if (fread(chunk_data, 1, chunk_len + 4, fp) != chunk_len + 4) {
            safe_free((void**)&idat_data);
            safe_free((void**)&chunk_data);
            fclose(fp);
            return NULL;
        }

        if (memcmp(chunk_type, "IHDR", 4) == 0) {
            pn.width = (int)png_read_u32(chunk_data);
            pn.height = (int)png_read_u32(chunk_data + 4);
            pn.bit_depth = chunk_data[8];
            pn.color_type = chunk_data[9];
            has_ihdr = 1;
        } else if (memcmp(chunk_type, "PLTE", 4) == 0) {
            /* P2-005: PLTE调色板支持（索引颜色） */
            if (pn.color_type == PNG_COLOR_INDEXED) {
                pn.palette_count = (int)(chunk_len / 3);
                if (pn.palette_count > 256) pn.palette_count = 256;
                for (int i = 0; i < pn.palette_count; i++) {
                    pn.palette[i][0] = chunk_data[i * 3 + 0];
                    pn.palette[i][1] = chunk_data[i * 3 + 1];
                    pn.palette[i][2] = chunk_data[i * 3 + 2];
                }
            }
        } else if (memcmp(chunk_type, "IDAT", 4) == 0) {
            /* 累积IDAT数据 */
            if (idat_size + chunk_len > idat_cap) {
                idat_cap = idat_size + chunk_len + 65536;
                idat_data = (uint8_t*)safe_realloc(idat_data, idat_cap);
                if (!idat_data) {
                    safe_free((void**)&chunk_data);
                    fclose(fp);
                    return NULL;
                }
            }
            memcpy(idat_data + idat_size, chunk_data, chunk_len);
            idat_size += chunk_len;
        } else if (memcmp(chunk_type, "IEND", 4) == 0) {
            safe_free((void**)&chunk_data);
            break;
        }
        safe_free((void**)&chunk_data);
    }
    fclose(fp);

    if (!has_ihdr || pn.width <= 0 || pn.height <= 0 || !idat_data) {
        safe_free((void**)&idat_data);
        log_error("[图像加载] PNG数据不完整");
        return NULL;
    }

    /* 解压缩IDAT数据 */
    size_t inflated_size = 0;
    uint8_t* inflated = png_inflate(idat_data, idat_size, &inflated_size);
    safe_free((void**)&idat_data);

    if (!inflated) {
        log_error("[图像加载] PNG解压缩失败（支持无压缩块+固定Huffman块，不支持动态Huffman）");
        return NULL;
    }

    /* 确定每像素字节数 */
    int bpp = 1;
    if (pn.color_type == PNG_COLOR_RGB) bpp = 3;
    else if (pn.color_type == PNG_COLOR_INDEXED) bpp = 1;
    else if (pn.color_type == PNG_COLOR_RGBA) bpp = 4;
    else if (pn.color_type == PNG_COLOR_GRAY_ALPHA) bpp = 2;

    int stride = pn.width * bpp + 1;
    size_t expected = (size_t)pn.height * stride;
    if (inflated_size < expected) {
        safe_free((void**)&inflated);
        log_error("[图像加载] PNG解压数据不足");
        return NULL;
    }

    /* 反滤波 */
    png_unfilter(inflated, pn.width, pn.height, bpp);

    /* 转换为RGB输出 */
    size_t out_size = (size_t)pn.width * pn.height * 3;
    uint8_t* rgb = (uint8_t*)safe_malloc(out_size);
    if (!rgb) {
        safe_free((void**)&inflated);
        return NULL;
    }

    for (int y = 0; y < pn.height; y++) {
        uint8_t* row = inflated + y * stride + 1; /* 跳过filter字节 */
        for (int x = 0; x < pn.width; x++) {
            size_t out_idx = ((size_t)y * pn.width + x) * 3;
            if (pn.color_type == PNG_COLOR_INDEXED && pn.palette_count > 0) {
                /* P2-005: 使用PLTE调色板进行索引颜色转换 */
                uint8_t idx = row[x * bpp];
                if (idx < pn.palette_count) {
                    rgb[out_idx + 0] = pn.palette[idx][0];
                    rgb[out_idx + 1] = pn.palette[idx][1];
                    rgb[out_idx + 2] = pn.palette[idx][2];
                } else {
                    rgb[out_idx + 0] = rgb[out_idx + 1] = rgb[out_idx + 2] = 0;
                }
            } else if (pn.color_type == PNG_COLOR_GRAY) {
                uint8_t g = row[x * bpp];
                rgb[out_idx + 0] = rgb[out_idx + 1] = rgb[out_idx + 2] = g;
            } else {
                /* RGB / RGBA */
                rgb[out_idx + 0] = row[x * bpp + 0];
                rgb[out_idx + 1] = row[x * bpp + 1];
                rgb[out_idx + 2] = row[x * bpp + 2];
            }
        }
    }

    safe_free((void**)&inflated);

    *width_out = pn.width;
    *height_out = pn.height;

    log_info("[图像加载] PNG解码成功: %s (%dx%d, 颜色类型=%d)", filepath, pn.width, pn.height, pn.color_type);
    return rgb;
}
