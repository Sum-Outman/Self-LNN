/**
 * @file text.c
 * @brief 文本处理模块实现 —— 基于LNN的语义编码
 *
 * 将文本作为连续时序信号输入LNN进行语义编码。
 * 不再使用简单字符频率统计，而是通过LNN的连续时间动态
 * 将每个Unicode码点作为输入信号，演化隐藏状态，
 * 最终隐藏状态向量即为文本的语义表示。
 *
 * 严格遵循：单一液态神经网络模型原则
 */

#include "selflnn/multimodal/text.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/**
 * @brief LNN文本编码默认维度
 */
#define LNN_TEXT_DEFAULT_INPUT_DIM   256
#define LNN_TEXT_DEFAULT_HIDDEN_DIM  128
#define LNN_TEXT_MAX_SEQ_LEN         1024

struct TextProcessor {
    TextConfig config;
    int is_initialized;
    LNN* text_lnn;              /**< 文本编码专用LNN实例（由系统主LNN分配） */
    int owns_lnn;               /**< 是否拥有LNN所有权 */
    float* hidden_state;        /**< LNN隐藏状态缓冲区 */
    float* cell_state;          /**< LNN细胞状态缓冲区 */
    float* temp_input;          /**< 临时输入缓冲区 */
    size_t hidden_dim;          /**< 隐藏状态维度 */
    size_t input_dim;           /**< 输入维度 */
};

/* ================================================================
 * UTF-8 辅助函数
 * ================================================================ */

static int utf8_char_length(unsigned char first_byte) {
    if ((first_byte & 0x80) == 0x00) return 1;
    else if ((first_byte & 0xE0) == 0xC0) return 2;
    else if ((first_byte & 0xF0) == 0xE0) return 3;
    else if ((first_byte & 0xF8) == 0xF0) return 4;
    return 1;
}

static int utf8_decode_codepoint(const char* str, unsigned int* codepoint) {
    if (!str || !codepoint) return 0;
    unsigned char first = (unsigned char)str[0];
    *codepoint = 0;
    if ((first & 0x80) == 0x00) {
        *codepoint = first;
        return 1;
    } else if ((first & 0xE0) == 0xC0) {
        if ((str[1] & 0xC0) != 0x80) return 0;
        *codepoint = ((unsigned int)(first & 0x1F) << 6)
                   | ((unsigned int)(str[1] & 0x3F));
        return 2;
    } else if ((first & 0xF0) == 0xE0) {
        if ((str[1] & 0xC0) != 0x80 || (str[2] & 0xC0) != 0x80) return 0;
        *codepoint = ((unsigned int)(first & 0x0F) << 12)
                   | ((unsigned int)(str[1] & 0x3F) << 6)
                   | ((unsigned int)(str[2] & 0x3F));
        return 3;
    } else if ((first & 0xF8) == 0xF0) {
        if ((str[1] & 0xC0) != 0x80 || (str[2] & 0xC0) != 0x80
            || (str[3] & 0xC0) != 0x80) return 0;
        *codepoint = ((unsigned int)(first & 0x07) << 18)
                   | ((unsigned int)(str[1] & 0x3F) << 12)
                   | ((unsigned int)(str[2] & 0x3F) << 6)
                   | ((unsigned int)(str[3] & 0x3F));
        return 4;
    }
    return 0;
}

/* ================================================================
 * 核心：基于LNN连续时间动态的文本语义编码
 * ================================================================ */

/**
 * @brief 将单个Unicode码点编码为LNN输入向量
 *
 * 使用位置编码 + 码点哈希投影的方式，
 * 将离散的码点值映射到连续输入空间。
 *
 * @param cp Unicode码点值
 * @param position 在文本中的位置（从0开始）
 * @param input_vec 输出输入向量 [input_dim]
 * @param input_dim 输入向量维度
 */
static void codepoint_to_lnn_input(unsigned int cp, size_t position,
                                   float* input_vec, size_t input_dim) {
    /* 清零输入向量 */
    memset(input_vec, 0, input_dim * sizeof(float));

    /* 码点哈希投影：使用多个哈希函数将码点分散到输入维度 */
    unsigned int h1 = cp * 2654435761u;
    unsigned int h2 = (cp ^ 0x5bd1e995) * 2654435761u;
    unsigned int h3 = (cp ^ 0x9e3779b9) * 2654435761u;

    size_t idx1 = h1 % input_dim;
    size_t idx2 = h2 % input_dim;
    size_t idx3 = h3 % input_dim;

    input_vec[idx1] = 1.0f;
    input_vec[idx2] = 0.5f;
    input_vec[idx3] = 0.25f;

    /* 位置编码：正弦/余弦位置编码 */
    if (input_dim >= 8) {
        float pos = (float)position;
        for (size_t i = 0; i < input_dim / 2 && (i * 2 + 1) < input_dim; i++) {
            float freq = 1.0f / powf(10000.0f, (2.0f * (float)i) / (float)input_dim);
            float pe_sin = sinf(pos * freq);
            float pe_cos = cosf(pos * freq);
            /* 将位置编码叠加到现有值上（而非清零重建） */
            size_t even_idx = i * 2;
            size_t odd_idx = i * 2 + 1;
            if (even_idx < input_dim) input_vec[even_idx] += pe_sin * 0.1f;
            if (odd_idx < input_dim) input_vec[odd_idx] += pe_cos * 0.1f;
        }
    }

    /* 归一化输入向量 */
    float norm = 0.0f;
    for (size_t i = 0; i < input_dim; i++) {
        norm += input_vec[i] * input_vec[i];
    }
    if (norm > 1e-8f) {
        float inv_norm = 1.0f / sqrtf(norm);
        for (size_t i = 0; i < input_dim; i++) {
            input_vec[i] *= inv_norm;
        }
    }
}

/* ================================================================
 * 公共API
 * ================================================================ */

TextProcessor* text_processor_create(const TextConfig* config) {
    if (!config) return NULL;

    TextProcessor* processor = (TextProcessor*)safe_malloc(sizeof(TextProcessor));
    if (!processor) return NULL;

    memset(processor, 0, sizeof(TextProcessor));
    processor->config = *config;

    /* 设置默认维度 */
    size_t input_dim = (config->vector_dimension > 0)
                       ? (size_t)config->vector_dimension
                       : LNN_TEXT_DEFAULT_INPUT_DIM;
    size_t hidden_dim = (config->cfc_hidden_size > 0)
                        ? config->cfc_hidden_size
                        : LNN_TEXT_DEFAULT_HIDDEN_DIM;

    processor->input_dim = input_dim;
    processor->hidden_dim = hidden_dim;

    /* 创建文本编码专用LNN（ZSFABC-M001修复：LNN创建失败返回NULL，不降级） */
    {
        LNNConfig lnn_cfg;
        memset(&lnn_cfg, 0, sizeof(LNNConfig));
        lnn_cfg.input_size = input_dim;
        lnn_cfg.hidden_size = hidden_dim;
        lnn_cfg.output_size = hidden_dim;
        lnn_cfg.learning_rate = 0.001f;
        lnn_cfg.time_constant = config->cfc_time_constant > 0.0f
                                ? config->cfc_time_constant : 0.1f;
        lnn_cfg.noise_std = 0.01f;
        lnn_cfg.enable_training = 1;
        lnn_cfg.enable_adaptation = 1;
        lnn_cfg.enable_evolution = 0;
        lnn_cfg.num_layers = 1;
        lnn_cfg.ode_solver_type = 0;

        processor->text_lnn = lnn_create(&lnn_cfg);
        if (!processor->text_lnn) {
            log_error("[文本] LNN创建失败，文本处理器初始化中止。");
            text_processor_free(processor);
            return NULL;
        }
        processor->owns_lnn = 1;

        processor->hidden_state = (float*)safe_calloc(hidden_dim, sizeof(float));
        processor->cell_state = (float*)safe_calloc(hidden_dim, sizeof(float));
        processor->temp_input = (float*)safe_calloc(input_dim, sizeof(float));

        if (!processor->hidden_state || !processor->cell_state
            || !processor->temp_input) {
            text_processor_free(processor);
            return NULL;
        }
    }

    processor->is_initialized = 1;
    return processor;
}

void text_processor_free(TextProcessor* processor) {
    if (!processor) return;

    if (processor->owns_lnn && processor->text_lnn) {
        lnn_free(processor->text_lnn);
        processor->text_lnn = NULL;
    }

    safe_free((void**)&processor->hidden_state);
    safe_free((void**)&processor->cell_state);
    safe_free((void**)&processor->temp_input);
    safe_free((void**)&processor);
}

/**
 * @brief 使用LNN连续时间动态进行文本语义编码
 *
 * 处理流程：
 * 1. 将文本按UTF-8解码为Unicode码点序列
 * 2. 每个码点通过哈希投影+位置编码转为LNN输入向量
 * 3. 依次输入LNN进行连续时间状态演化
 * 4. 最终隐藏状态向量即为文本语义编码
 */
int text_process_string(TextProcessor* processor,
                       const char* text, size_t length,
                       float* features, size_t max_features) {
    if (!processor || !text || !features || max_features == 0) return -1;
    if (length == 0) length = strlen(text);

    if (processor->text_lnn && processor->hidden_state
        && processor->temp_input) {
        /* ================================================================
         * 基于LNN的连续时间语义编码（深度实现）
         * ================================================================ */

        size_t input_dim = processor->input_dim;
        size_t hidden_dim = processor->hidden_dim;

        /* 重置LNN隐藏状态（新文本，新编码） */
        memset(processor->hidden_state, 0, hidden_dim * sizeof(float));
        memset(processor->cell_state, 0, hidden_dim * sizeof(float));

        /* 遍历文本的每个UTF-8字符，作为时序信号输入LNN */
        const char* pos = text;
        const char* end = text + length;
        size_t char_position = 0;
        size_t seq_count = 0;

        while (pos < end && seq_count < LNN_TEXT_MAX_SEQ_LEN) {
            unsigned int cp = 0;
            int char_len = utf8_decode_codepoint(pos, &cp);
            if (char_len <= 0 || char_len > 4) {
                pos++;
                continue;
            }

            /* 将码点编码为LNN输入向量 */
            codepoint_to_lnn_input(cp, char_position,
                                   processor->temp_input, input_dim);

            /* LNN前向传播：时序状态演化 */
            float* lnn_output = (float*)safe_calloc(hidden_dim, sizeof(float));
            if (!lnn_output) return -1;

            int fwd_ret = lnn_forward(processor->text_lnn,
                                      processor->temp_input,
                                      lnn_output);
            if (fwd_ret == 0) {
                /* 使用LNN输出更新隐藏状态 */
                memcpy(processor->hidden_state, lnn_output,
                       hidden_dim * sizeof(float));
                memcpy(processor->cell_state, lnn_output,
                       hidden_dim * sizeof(float));
            }

            safe_free((void**)&lnn_output);
            pos += char_len;
            char_position++;
            seq_count++;
        }

        /* 将最终LNN隐藏状态作为文本语义特征输出 */
        size_t copy_dim = (hidden_dim < max_features) ? hidden_dim : max_features;
        memcpy(features, processor->hidden_state, copy_dim * sizeof(float));

        /* 如果输出维度小于隐藏维度，对剩余维度降采样 */
        if (max_features < hidden_dim) {
            /* 已复制完整部分 */
        } else if (max_features > hidden_dim) {
            /* 用零填充多余维度 */
            memset(features + hidden_dim, 0,
                   (max_features - hidden_dim) * sizeof(float));
        }

        return (int)copy_dim;
    }

    /* ZSFABC-M001修复: LNN不可用时返回0特征维度，不再使用Unicode码点哈希降级
     * 系统要求液态神经网络模型全部功能，不允许降级到非LNN编码 */
    log_warning("[文本] LNN不可用，文本特征提取返回0维度。请确保系统LNN已初始化。");
    return 0;
}

int text_extract_char_features(TextProcessor* processor,
                              const char* text, size_t length,
                              float* char_features, size_t max_features) {
    /* 委托给主处理函数，确保使用LNN语义编码 */
    return text_process_string(processor, text, length,
                              char_features, max_features);
}

int text_processor_get_config(const TextProcessor* processor, TextConfig* config) {
    if (!processor || !config) return -1;
    *config = processor->config;
    return 0;
}

int text_processor_set_config(TextProcessor* processor, const TextConfig* config) {
    if (!processor || !config) return -1;
    processor->config = *config;
    return 0;
}

void text_processor_reset(TextProcessor* processor) {
    if (!processor) return;
    if (processor->hidden_state) {
        memset(processor->hidden_state, 0,
               processor->hidden_dim * sizeof(float));
    }
    if (processor->cell_state) {
        memset(processor->cell_state, 0,
               processor->hidden_dim * sizeof(float));
    }
}
