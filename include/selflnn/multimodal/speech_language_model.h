/**
 * @file speech_language_model.h
 * @brief K-033: 纯C N-gram语言模型训练器接口
 */

#ifndef SELFLNN_SPEECH_LANGUAGE_MODEL_H
#define SELFLNN_SPEECH_LANGUAGE_MODEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int speech_language_model_train(const char* corpus_path, int n, const char* model_path);

int speech_language_model_build_from_text(const char* text, size_t text_len,
                                           int n, void** model_out);

float speech_language_model_score(void* model, const int* tokens, int num_tokens);
int speech_language_model_vocab_size(void* model);
void speech_language_model_free(void* model);

/**
 * @brief 语言模型后处理纠错
 * 使用N-gram语言模型对语音识别结果进行置信度评估和纠错处理。
 * @param model 语言模型实例
 * @param input_text 输入待纠错文本
 * @param corrected 输出纠错后文本缓冲区
 * @param corrected_size 输出缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int speech_language_model_correct(void* model, const char* input_text,
                                   char* corrected, size_t corrected_size);

#ifdef __cplusplus
}
#endif

#endif
