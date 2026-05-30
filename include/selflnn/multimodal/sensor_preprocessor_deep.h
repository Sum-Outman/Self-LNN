#ifndef SELFLNN_SENSOR_PREPROCESSOR_DEEP_H
#define SELFLNN_SENSOR_PREPROCESSOR_DEEP_H

#include "selflnn/multimodal/sensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* SensorDeepPreprocessor 不透明类型，定义在 sensor_preprocessor_deep.c 中 */
typedef struct SensorDeepPreprocessor SensorDeepPreprocessor;

/* 创建深度传感器预处理器 */
SensorDeepPreprocessor* sensor_deep_preprocessor_create(size_t state_dim, size_t obs_dim);

/* 释放深度传感器预处理器 */
void sensor_deep_preprocessor_free(SensorDeepPreprocessor* sdp);

/* 深度预处理单帧 */
int sensor_deep_preprocess_frame(SensorDeepPreprocessor* sdp,
                                 const float* raw_data,
                                 const float* gyro,
                                 const float* acc,
                                 float dt,
                                 float* output);

/* 预处理质量评估 */
int sensor_deep_quality_assess(SensorDeepPreprocessor* sdp,
                                float* quality_score, float* innovation_norm,
                                float* consistency_ratio);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SENSOR_PREPROCESSOR_DEEP_H */
