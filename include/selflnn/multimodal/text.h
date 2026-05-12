/**
 * @file text.h
 * @brief 文本处理模块接口
 * 
 * 文本数据处理接口，提供基本的字符级文本特征提取。
 * 所有文本通过统一CfC信号处理器进行连续时间动态演化。
 */

#ifndef SELFLNN_TEXT_H
#define SELFLNN_TEXT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 文本处理配置
 */
typedef struct {
    int max_tokens;         /**< 最大字符数 */
    int vector_dimension;   /**< 向量维度 */
    int language;           /**< 语言类型 (0=中文, 1=英文) */
    int enable_cfc;         /**< 是否启用CfC连续时间演化 */
    size_t cfc_hidden_size; /**< CfC隐藏状态维度（默认32） */
    float cfc_time_constant;/**< CfC时间常数（默认0.1） */
} TextConfig;

/**
 * @brief 文本处理器句柄
 */
typedef struct TextProcessor TextProcessor;

/**
 * @brief 创建文本处理器
 * 
 * @param config 文本配置
 * @return TextProcessor* 处理器句柄，失败返回NULL
 */
TextProcessor* text_processor_create(const TextConfig* config);

/**
 * @brief 释放文本处理器
 * 
 * @param processor 处理器句柄
 */
void text_processor_free(TextProcessor* processor);

/**
 * @brief 处理文本数据
 *
 * 提取字符级频率特征并归一化，
 * 由 multimodal.c 中的主CfC统一处理时序演化。
 *
 * @param processor 处理器句柄
 * @param text 文本字符串
 * @param length 文本长度
 * @param features 特征输出缓冲区
 * @param max_features 最大特征数
 * @return int 成功返回特征数，失败返回-1
 */
int text_process_string(TextProcessor* processor,
                       const char* text, size_t length,
                       float* features, size_t max_features);

/**
 * @brief 提取字符级特征
 *
 * 统计输入文本的字符频率分布，归一化后输出。
 * 作为文本的浅层特征供CfC处理。
 *
 * @param processor 处理器句柄
 * @param text 文本字符串
 * @param length 文本长度
 * @param char_features 字符特征输出缓冲区
 * @param max_features 最大特征数
 * @return int 成功返回特征数，失败返回-1
 */
int text_extract_char_features(TextProcessor* processor,
                              const char* text, size_t length,
                              float* char_features, size_t max_features);

/**
 * @brief 获取文本处理器配置
 * 
 * @param processor 处理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int text_processor_get_config(const TextProcessor* processor, TextConfig* config);

/**
 * @brief 设置文本处理器配置
 * 
 * @param processor 处理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int text_processor_set_config(TextProcessor* processor, const TextConfig* config);

/**
 * @brief 重置文本处理器
 * 
 * @param processor 处理器句柄
 */
void text_processor_reset(TextProcessor* processor);

/**
 * @brief 生成文本嵌入向量（基于字符n-gram的轻量级实现）
 * 
 * 将文本字符串转换为固定维度的嵌入向量，使用字符n-gram哈希嵌入技术。
 * 支持n=1,2,3的字符n-gram，通过FNV-1a哈希函数映射到嵌入维度。
 * 输出的嵌入向量直接作为CfC信号处理器的输入序列。
 * 
 * @param processor 文本处理器句柄
 * @param text 文本字符串
 * @param length 文本长度（0表示自动计算）
 * @param embedding_dim 嵌入维度
 * @param embeddings 嵌入向量输出缓冲区（调用者负责分配内存）
 * @return int 成功返回0，失败返回-1
 */
int text_generate_embeddings(TextProcessor* processor,
                            const char* text, size_t length,
                            size_t embedding_dim,
                            float* embeddings);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_TEXT_H
