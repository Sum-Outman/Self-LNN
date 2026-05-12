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
    } else {
        log_error("[图像加载] 不支持的图像格式: %s (仅支持.bmp和.ppm)", ext);
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
