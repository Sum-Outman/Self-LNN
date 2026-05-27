#ifndef SELFLNN_RADAR_H
#define SELFLNN_RADAR_H

/**
 * @file radar.h
 * @brief 雷达信号预处理器 — FMCW距离-多普勒、CFAR检测、微多普勒分析
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RADAR_MAX_TARGETS          64
#define RADAR_FEATURE_DIM          128
#define RADAR_FFT_MAX_SIZE         4096

typedef struct {
    float range;
    float velocity;
    float azimuth;
    float elevation;
    float snr;
    float rcs;
    int track_id;
} RadarTarget;

typedef struct {
    float* range_profile;
    float* doppler_spectrum;
    float* micro_doppler;
    int num_range_bins;
    int num_doppler_bins;
    int num_micro_doppler_bins;
} RadarSpectrum;

typedef struct RadarProcessor RadarProcessor;

RadarProcessor* radar_processor_create(int num_samples, int num_chirps,
    float start_freq, float bandwidth, float chirp_duration,
    float sample_rate);
void radar_processor_free(RadarProcessor* rp);

int radar_range_doppler(RadarProcessor* rp,
    const float* raw_adc, int num_samples, int num_chirps,
    RadarSpectrum* spectrum);
int radar_cfar_detection(RadarProcessor* rp,
    const float* range_doppler_map, int num_range, int num_doppler,
    float pfa, int guard_cells, int ref_cells,
    RadarTarget* targets, int max_targets, int* num_detected);
int radar_micro_doppler(RadarProcessor* rp,
    const float* raw_iq, int num_samples,
    float* micro_doppler, int md_size);
int radar_compute_feature_vector(RadarProcessor* rp,
    const RadarSpectrum* spectrum, const RadarTarget* targets,
    int num_targets, float* feature_vector, size_t feature_dim);
int radar_doa_estimation(RadarProcessor* rp,
    const float* antenna_snapshots, int num_antennas, int num_snapshots,
    float* azimuth, float* elevation);

#ifdef __cplusplus
}
#endif

#endif
