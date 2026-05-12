#ifndef SELFLNN_STEREO_DEPTH_ENHANCE_H
#define SELFLNN_STEREO_DEPTH_ENHANCE_H

#include <stddef.h>
#include "stereo_calibration.h"
#include "stereo_3d_reconstruction.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SDE_MAX_DISPARITY 256
#define SDE_MAX_PATH_DIRECTIONS 8
#define SDE_MAX_FUSION_VIEWS 16
#define SDE_TEMPORAL_BUFFER_SIZE 16
#define SDE_MAX_FILTER_KERNEL 31

typedef enum {
    SDE_COST_CENSUS_SAD = 0,
    SDE_COST_CENSUS_SSD = 1,
    SDE_COST_GRADIENT_CENSUS = 2,
    SDE_COST_BT = 3
} SDECostType;

typedef enum {
    SDE_FUSION_WEIGHTED_AVERAGE = 0,
    SDE_FUSION_CONFIDENCE_MAX = 1,
    SDE_FUSION_MEDIAN = 2,
    SDE_FUSION_ADAPTIVE = 3
} SDEFusionMethod;

typedef struct {
    int min_disparity;
    int max_disparity;
    int window_size;
    int num_paths;
    float p1;
    float p2;
    SDECostType cost_type;
    int enable_lr_check;
    float lr_check_threshold;
    int enable_speckle_filter;
    int speckle_window_size;
    float speckle_range;
    int enable_subpixel;
    int enable_median_filter;
    int median_filter_size;
    int enable_bilateral_filter;
    float bilateral_sigma_color;
    float bilateral_sigma_space;
    int bilateral_kernel_size;
} SDEStereoConfig;

typedef struct {
    float fx;
    float fy;
    float cx;
    float cy;
    float baseline;
} SDECameraParams;

typedef struct {
    float* data;
    int width;
    int height;
    float* confidence;
} SDEDepthMap;

typedef struct {
    SDEFusionMethod method;
    float confidence_threshold;
    float temporal_alpha;
    int num_fusion_views;
    int enable_hole_filling;
    int hole_filling_radius;
    int enable_edge_preserve;
    float edge_preserve_sigma;
    int enable_temporal_fusion;
    float temporal_smoothing_factor;
} SDEFusionConfig;

typedef struct SDEHandler SDEHandler;

typedef struct SDEFusionHandler SDEFusionHandler;

SDEHandler* sde_create(const SDEStereoConfig* config);
void sde_free(SDEHandler* handler);

int sde_set_calibration(SDEHandler* handler,
                         const float left_k[9], const float right_k[9],
                         const float rotation[9], const float translation[3],
                         const float dist_left[5], const float dist_right[5]);

int sde_set_camera_params(SDEHandler* handler, const SDECameraParams* params);

int sde_compute_disparity(SDEHandler* handler,
                           const float* left_image, const float* right_image,
                           int width, int height, int channels,
                           float* disparity_out, float* confidence_out);

int sde_compute_depth(SDEHandler* handler,
                       const float* disparity, int width, int height,
                       float* depth_out, float* confidence_out);

int sde_generate_point_cloud(SDEHandler* handler,
                              const float* left_image,
                              const float* depth_map,
                              int width, int height, int channels,
                              float* point_cloud_out, int max_points,
                              int* out_point_count);

int sde_compute_stereo_pipeline(SDEHandler* handler,
                                 const float* left_image, const float* right_image,
                                 int width, int height, int channels,
                                 float* disparity_out, float* depth_out,
                                 float* point_cloud_out, int max_points,
                                 int* out_point_count,
                                 float* confidence_out);

SDEStereoConfig sde_get_default_config(void);

int sde_set_config(SDEHandler* handler, const SDEStereoConfig* config);
int sde_get_config(SDEHandler* handler, SDEStereoConfig* config);

SDEFusionHandler* sde_fusion_create(const SDEFusionConfig* config);
void sde_fusion_free(SDEFusionHandler* fh);

int sde_fusion_add_depth(SDEFusionHandler* fh,
                          const float* depth_map, const float* confidence,
                          int width, int height,
                          const float* pose_matrix);

int sde_fusion_fuse(SDEFusionHandler* fh,
                     float* fused_depth_out, float* fused_confidence_out);

int sde_fusion_fuse_multi_view(SDEFusionHandler* fh,
                                 const float** depth_maps, const float** confidence_maps,
                                 int num_views, int width, int height,
                                 float* fused_out, float* confidence_out);

int sde_fusion_temporal(SDEFusionHandler* fh,
                         const float* current_depth, const float* current_confidence,
                         int width, int height,
                         float* smoothed_out, float* confidence_out);

int sde_fusion_hole_fill(const float* depth_map, const float* confidence,
                          int width, int height, int radius,
                          float* filled_out);

int sde_fusion_edge_preserve_filter(const float* depth_map, const float* guide_image,
                                      int width, int height,
                                      float sigma_depth, float sigma_image,
                                      int kernel_size,
                                      float* filtered_out);

SDEFusionConfig sde_fusion_get_default_config(void);

int sde_save_disparity(const float* disparity, int width, int height,
                        const char* filepath);
int sde_save_depth(const float* depth, int width, int height,
                    const char* filepath);
int sde_load_disparity(float* disparity, int width, int height,
                        const char* filepath);
int sde_load_depth(float* depth, int width, int height,
                    const char* filepath);

#ifdef __cplusplus
}
#endif

#endif
