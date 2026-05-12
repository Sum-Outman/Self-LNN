#ifndef SELFLNN_ROBOT_VISION_ODOMETRY_H
#define SELFLNN_ROBOT_VISION_ODOMETRY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VO_MAX_FEATURES 2000
#define VO_MAX_MATCHES 5000
#define VO_MAX_INLIERS 1000
#define VO_GRID_SIZE 32
#define VO_HISTOGRAM_BINS 8
#define VO_DESC_LENGTH 128

typedef struct {
    float x;
    float y;
    float response;
    int octave;
    float angle;
    float descriptor[VO_DESC_LENGTH];
    int tracked;
    int age;
} FeaturePoint;

typedef struct {
    int idx_src;
    int idx_dst;
    float distance;
    float ratio;
    int is_inlier;
} FeatureMatch;

typedef struct {
    float K[9];
    float K_inv[9];
    float fx;
    float fy;
    float cx;
    float cy;
    float width;
    float height;
} CameraIntrinsics;

typedef struct {
    float position[3];
    float orientation[4];
    float velocity[3];
    float angular_velocity[3];
    float covariance[36];
    int initialized;
    unsigned long long timestamp;
} CameraPose;

typedef struct {
    float E[9];
    float R[9];
    float t[3];
    int num_inliers;
    float reprojection_error;
} EssentialMatrix;

typedef struct {
    CameraIntrinsics intrinsics;
    FeaturePoint features[VO_MAX_FEATURES];
    int feature_count;
    FeaturePoint prev_features[VO_MAX_FEATURES];
    int prev_feature_count;
    FeatureMatch matches[VO_MAX_MATCHES];
    int match_count;
    FeatureMatch inliers[VO_MAX_INLIERS];
    int inlier_count;
    CameraPose pose;
    CameraPose prev_pose;
    EssentialMatrix essential;
    float trajectory[10000][3];
    int trajectory_length;
    int max_features;
    float max_response;
    float min_distance;
    float max_distance;
    float ransac_threshold;
    int ransac_iterations;
    int initialized;
    int frame_count;
    float total_translation;
    char name[64];
} VisualOdometry;

extern const CameraIntrinsics CAMERA_INTRINSICS_DEFAULT;

VisualOdometry* visual_odometry_create(const CameraIntrinsics* intrinsics);
void visual_odometry_free(VisualOdometry* vo);
int visual_odometry_init(VisualOdometry* vo, const CameraIntrinsics* intrinsics);
int visual_odometry_reset(VisualOdometry* vo);

int visual_odometry_extract_features(VisualOdometry* vo,
                                      const unsigned char* image,
                                      int width, int height, int stride);
int visual_odometry_compute_descriptors(VisualOdometry* vo,
                                         const unsigned char* image,
                                         int width, int height, int stride);

int visual_odometry_match_features(VisualOdometry* vo);
int visual_odometry_compute_essential_matrix(VisualOdometry* vo);

int visual_odometry_estimate_pose(VisualOdometry* vo);
int visual_odometry_update(VisualOdometry* vo,
                            const unsigned char* image,
                            int width, int height, int stride);

int visual_odometry_get_pose(const VisualOdometry* vo,
                              float* position, float* orientation);
int visual_odometry_get_trajectory(const VisualOdometry* vo,
                                    float* trajectory, size_t* length);
int visual_odometry_get_status(const VisualOdometry* vo,
                                char* buffer, size_t size);

int visual_odometry_triangulate(const VisualOdometry* vo,
                                 const FeatureMatch* match,
                                 float* point_3d);
int visual_odometry_compute_reprojection_error(const VisualOdometry* vo,
                                                 float* mean_error,
                                                 float* max_error);

#ifdef __cplusplus
}
#endif

#endif
