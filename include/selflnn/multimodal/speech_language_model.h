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

#ifdef __cplusplus
}
#endif

#endif
