/**
 * @file environment_sound.h
 * @brief K-034: 环境声音CfC液态分类器接口
 */

#ifndef SELFLNN_ENVIRONMENT_SOUND_H
#define SELFLNN_ENVIRONMENT_SOUND_H

#ifdef __cplusplus
extern "C" {
#endif

void* environment_sound_classifier_create(int num_classes);

int environment_sound_classify(void* classifier,
                                const float* audio_samples, int num_samples,
                                int sample_rate,
                                char* class_name, int max_name_len,
                                float* confidence);

void environment_sound_classifier_free(void* classifier);

const char* environment_sound_class_name(int class_id);

#ifdef __cplusplus
}
#endif

#endif
