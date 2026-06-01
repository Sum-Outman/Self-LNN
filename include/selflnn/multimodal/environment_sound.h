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

/* ZSFZS-F020: 环境声音权重保存与加载
 * 权重文件格式：魔数0x45535357("ESSW") + 版本号(uint32_t) + 类别数(int) + 各层权重数据
 * mel_filters(ESC_FREQ_BINS*128) + cfc_w(ESC_HIDDEN_DIM*ESC_HIDDEN_DIM) + cfc_b(ESC_HIDDEN_DIM)
 * + fc_w(ESC_MAX_CLASSES*ESC_HIDDEN_DIM) + fc_b(ESC_MAX_CLASSES)
 */
int env_sound_save_weights(void* classifier, const char* filepath);
int env_sound_load_weights(void* classifier, const char* filepath);

/* ZSFZS-F020: 检查分类器是否已加载训练好的权重 */
int env_sound_is_trained(void* classifier);

/* ZSFXXXQ-P0-001: 标记分类器为已训练状态（引导训练模式注册） */
void env_sound_classifier_mark_trained(void* classifier);

#ifdef __cplusplus
}
#endif

#endif
