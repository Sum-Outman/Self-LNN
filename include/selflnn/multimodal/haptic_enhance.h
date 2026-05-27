#ifndef SELFLNN_HAPTIC_ENHANCE_H
#define SELFLNN_HAPTIC_ENHANCE_H

/**
 * @file haptic_enhance.h
 * @brief 触觉感知增强接口 — CfC触觉处理器+纹理分析+物体识别+抓取学习
 *
 * R9-B修复: 修正为匹配haptic_enhance.c实际实现。
 * 原R2-07版本使用了错误的类型名和函数名。
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== HapticCfc 核心处理器 ===== */

typedef struct HapticCfcConfig {
    int input_dim;
    int hidden_dim;
    int output_dim;
    float time_constant;
    float learning_rate;
    int num_materials;
    int enable_fusion;
} HapticCfcConfig;

typedef struct HapticCfcProcessor HapticCfcProcessor;

HapticCfcConfig haptic_cfc_get_default_config(void);
HapticCfcProcessor* haptic_cfc_create(const HapticCfcConfig* config);
void haptic_cfc_free(HapticCfcProcessor* proc);
int haptic_cfc_process(HapticCfcProcessor* proc,
    const float* raw_signal, int signal_len,
    float* features, int feature_dim);
void haptic_cfc_reset(HapticCfcProcessor* proc);
int haptic_cfc_get_state(const HapticCfcProcessor* proc,
    float* state, int state_dim);

/* 全局处理器桥接（供multimodal_manager/haptic_learning跨模块调用） */
void haptic_enhance_set_global_processor(HapticCfcProcessor* proc);
HapticCfcProcessor* haptic_enhance_get_global_processor(void);

/* ===== 触觉纹理分析器 ===== */

typedef struct HapticTextureAnalyzer HapticTextureAnalyzer;

HapticTextureAnalyzer* haptic_texture_create(void);
void haptic_texture_free(HapticTextureAnalyzer* ta);
int haptic_texture_analyze(HapticTextureAnalyzer* ta,
    const float* force_signal, int signal_len,
    float* texture_features, int feature_dim);
int haptic_texture_learn(HapticTextureAnalyzer* ta,
    const float* features, const char* label);
int haptic_texture_list(const HapticTextureAnalyzer* ta,
    char labels[][64], int max_labels);
void haptic_texture_reset(HapticTextureAnalyzer* ta);

/* ===== 触觉物体识别器 ===== */

typedef struct HapticObjectRecognizer HapticObjectRecognizer;

HapticObjectRecognizer* haptic_object_recognizer_create(void);
void haptic_object_recognizer_free(HapticObjectRecognizer* rec);
int haptic_object_recognize(HapticObjectRecognizer* rec,
    const float* features, int feature_dim,
    char* label, int label_max);
int haptic_object_learn(HapticObjectRecognizer* rec,
    const float* features, int feature_dim, const char* label);
void haptic_object_recognizer_reset(HapticObjectRecognizer* rec);

/* ===== 触觉抓取学习器 ===== */

typedef struct HapticGraspLearner HapticGraspLearner;

HapticGraspLearner* haptic_grasp_learner_create(void);
void haptic_grasp_learner_free(HapticGraspLearner* gl);
int haptic_grasp_control(HapticGraspLearner* gl,
    const float* force_feedback, int feedback_dim,
    float* grip_force, float* grip_position);
int haptic_grasp_learn_from_demo(HapticGraspLearner* gl,
    const float* force_sequence, int seq_len,
    const float* position_sequence, int pos_dim);
float haptic_grasp_evaluate(HapticGraspLearner* gl,
    const float* force_feedback, int feedback_dim);
void haptic_grasp_learner_reset(HapticGraspLearner* gl);

#ifdef __cplusplus
}
#endif

#endif
