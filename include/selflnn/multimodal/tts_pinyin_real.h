/**
 * @file tts_pinyin_real.h
 * @brief 真实汉字→拼音映射接口
 * 
 * 覆盖GB2312一级汉字(3755字) + 常用3500字 + HSK词汇汉字
 * 提供Unicode码点→拼音(声母/韵母/声调)查询、GB2312编码查询、
 * 嗓音模型存取与验证。
 */
#ifndef SELFLNN_TTS_PINYIN_REAL_H
#define SELFLNN_TTS_PINYIN_REAL_H

#include "selflnn/multimodal/tts.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 根据Unicode码点查拼音
 * @param codepoint Unicode码点 (如 '一'=0x4E00)
 * @param out_init 输出声母索引 (1-23)
 * @param out_final 输出韵母索引 (1-38)
 * @param out_tone 输出声调 (0-4, 0=轻声)
 * @return 0=成功，-1=未找到
 */
int tts_pinyin_lookup(uint16_t codepoint, int* out_init, int* out_final, int* out_tone);

/**
 * @brief 根据GB2312编码查拼音
 * @param gb_hi GB2312高字节 (区码)
 * @param gb_lo GB2312低字节 (位码)
 * @param out_init 输出声母索引
 * @param out_final 输出韵母索引
 * @param out_tone 输出声调
 * @return 0=成功，-1=未找到
 */
int tts_pinyin_lookup_gb2312(unsigned char gb_hi, unsigned char gb_lo,
                             int* out_init, int* out_final, int* out_tone);

/**
 * @brief 获取拼音表大小
 * @return 拼音表条目数量
 */
int tts_pinyin_table_size(void);

/**
 * @brief 获取默认嗓音模型参数
 * @param params 输出嗓音模型参数
 */
void tts_pinyin_get_default_voice_model(VoiceModelParams* params);

/**
 * @brief 验证嗓音模型参数有效性
 * @param params 嗓音模型参数
 * @return 0=有效，负数=无效的参数索引
 */
int tts_pinyin_validate_voice_model(const VoiceModelParams* params);

/**
 * @brief 保存嗓音模型到文件
 * @param params 嗓音模型参数
 * @param filepath 保存路径
 * @return 0=成功，-1=失败
 */
int tts_pinyin_save_voice_model(const VoiceModelParams* params, const char* filepath);

/**
 * @brief 从文件加载嗓音模型
 * @param params 输出嗓音模型参数
 * @param filepath 文件路径
 * @return 0=成功，-1=失败
 */
int tts_pinyin_load_voice_model(VoiceModelParams* params, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_TTS_PINYIN_REAL_H */
