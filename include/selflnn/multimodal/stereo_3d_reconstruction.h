/**
 * @file stereo_3d_reconstruction.h
 * @brief 双摄像头空间3D重建增强系统接口
 */

#ifndef SELFLNN_STEREO_3D_RECONSTRUCTION_H
#define SELFLNN_STEREO_3D_RECONSTRUCTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SR3D_MAX_POINTS 500000
#define SR3D_MAX_TRIANGLES 1000000
#define SR3D_MAX_SEGMENTS 64
#define SR3D_MAX_MEASUREMENTS 32

typedef struct {
    float x, y, z;
    float nx, ny, nz;
    unsigned char r, g, b;
    float confidence;
    int segment_id;
} SR3DPoint;

typedef struct {
    int v1, v2, v3;
    float normal[3];
    int segment_id;
} SR3DTriangle;

typedef struct {
    int segment_id;
    char label[64];
    float center[3];
    float extents[3];
    float volume;
    int point_count;
    float confidence;
} SR3DSegment;

typedef struct {
    float width_m;
    float height_m;
    float depth_m;
    float area_m2;
    float volume_m3;
    char unit[8];
} SR3DMeasurement;

typedef struct {
    float camera_matrix_left[9];
    float camera_matrix_right[9];
    float dist_coeffs_left[5];
    float dist_coeffs_right[5];
    float rotation[9];
    float translation[3];
    float fundamental[9];
    float essential[9];
    float baseline_m;
} SR3DCalibration;

typedef struct SR3DReconstructor SR3DReconstructor;

SR3DReconstructor* sr3d_create(void);
void sr3d_free(SR3DReconstructor* sr);

int sr3d_set_calibration(SR3DReconstructor* sr, const SR3DCalibration* calib);

/* 稠密匹配与点云生成 */
int sr3d_compute_dense_disparity(SR3DReconstructor* sr, const float* left, const float* right, int w, int h, float* disparity);
int sr3d_generate_point_cloud(SR3DReconstructor* sr, const float* left, const float* right, const float* disparity, int w, int h, SR3DPoint* points, int* count);
int sr3d_filter_point_cloud(SR3DReconstructor* sr, SR3DPoint* points, int count, int* filtered_count);

/* 表面重建 */
int sr3d_reconstruct_surface(SR3DReconstructor* sr, const SR3DPoint* points, int count, SR3DTriangle* triangles, int* tri_count);
int sr3d_texture_map(SR3DReconstructor* sr, SR3DPoint* points, int count, SR3DTriangle* triangles, int tri_count, const float* image, int w, int h);

/* 物体分割 */
int sr3d_segment_objects(SR3DReconstructor* sr, const SR3DPoint* points, int count, SR3DSegment* segments, int* seg_count);

/* 空间测量 */
int sr3d_measure_object(SR3DReconstructor* sr, const SR3DPoint* points, int count, int segment_id, SR3DMeasurement* measurement);
int sr3d_measure_distance(SR3DReconstructor* sr, const float* p1, const float* p2, float* distance);

/* 场景标注 */
int sr3d_label_scene(SR3DReconstructor* sr, const SR3DPoint* points, int count, const SR3DSegment* segments, int seg_count, char* scene_desc, size_t max_len);

/* Delaunay三角剖分网格生成 */
int sr3d_generate_mesh(SR3DReconstructor* sr, const SR3DPoint* points, int point_count,
                        int* p_tri_count, int* p_tri_indices, int max_tri_indices);

#ifdef __cplusplus
}
#endif
#endif
