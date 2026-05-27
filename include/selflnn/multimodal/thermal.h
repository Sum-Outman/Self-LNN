#ifndef SELFLNN_THERMAL_H
#define SELFLNN_THERMAL_H

/**
 * @file thermal.h
 * @brief 热感预处理器 — 热像仪温度标定、热点检测、温度梯度分析
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define THERMAL_MAX_PIXELS        4096
#define THERMAL_FEATURE_DIM       64
#define THERMAL_MAX_HOTSPOTS      16

typedef struct {
    float min_temp;
    float max_temp;
    float mean_temp;
    float median_temp;
    float std_temp;
    float gradient_magnitude;
    float gradient_direction;
    int pixel_count;
} ThermalStats;

typedef struct {
    float center_x;
    float center_y;
    float peak_temp;
    float area;
    float perimeter;
    float circularity;
    float mean_intensity;
} HotSpot;

typedef struct ThermalProcessor ThermalProcessor;

ThermalProcessor* thermal_processor_create(void);
void thermal_processor_free(ThermalProcessor* tp);

int thermal_calibrate(ThermalProcessor* tp,
    const float* raw_thermal, int width, int height,
    float emissivity, float ambient_temp, float* calibrated);
int thermal_compute_stats(ThermalProcessor* tp,
    const float* thermal_image, int width, int height,
    ThermalStats* stats);
int thermal_detect_hotspots(ThermalProcessor* tp,
    const float* thermal_image, int width, int height,
    float threshold_temp, HotSpot* hotspots, int max_hotspots,
    int* num_detected);
int thermal_compute_feature_vector(ThermalProcessor* tp,
    const float* thermal_image, int width, int height,
    float* feature_vector, size_t feature_dim);
int thermal_compute_gradient(ThermalProcessor* tp,
    const float* thermal_image, int width, int height,
    float* grad_x, float* grad_y, int grad_size);

#ifdef __cplusplus
}
#endif

#endif
