/**
 * @file slam_internal.h
 * @brief SLAM系统内部数据结构与函数声明（模块间共享）
 *
 * 本文件包含SLAM系统各模块间共享的内部数据结构、常量和函数声明。
 * 用于将庞大的 slam.c（~10657行）拆分为多个可维护的模块文件。
 * 100%纯C实现，无外部依赖。
 */

#ifndef SELFLNN_SLAM_INTERNAL_H
#define SELFLNN_SLAM_INTERNAL_H

#include "selflnn/multimodal/slam.h"
#include "selflnn/multimodal/point_cloud.h"
#include "selflnn/multimodal/depth_estimation.h"
#include "selflnn/multimodal/liquid_vision.h"
#include "selflnn/core/common.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/platform.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>

#ifdef _WIN32
#include <io.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ==================== 内部常量定义 ==================== */

#define SLAM_MAX_KEYFRAMES 1000
#define SLAM_MAX_LANDMARKS 10000
#define SLAM_MAX_FEATURES_PER_FRAME 2000
#define SLAM_MAX_OBSERVATIONS_PER_LANDMARK 20
#define SLAM_MAX_LOOP_CANDIDATES 50
#define SLAM_MAX_OPTIMIZATION_ITERATIONS 100
#define SLAM_EPSILON 1e-6f
#define SLAM_DEFAULT_FOCAL_LENGTH 525.0f
#define SLAM_DEFAULT_PRINCIPAL_POINT 320.0f

#define SLAM_VOCABULARY_SIZE 1000
#define SLAM_VOCABULARY_DEPTH 6
#define SLAM_VOCABULARY_BRANCHING 10
#define SLAM_VOCABULARY_MAX_TRAIN 500
#define SLAM_TEMPORAL_WINDOW_SIZE 5
#define SLAM_TEMPORAL_CONSISTENCY_THRESHOLD 3
#define SLAM_COVISIBILITY_MIN_WEIGHT 3
#define SLAM_MAX_LOOP_CANDIDATES_SCORED 100
#define SLAM_FUSION_MERGE_DISTANCE 0.1f
#define SLAM_DRIFT_PROPAGATION_RADIUS 50.0f
#define SLAM_BOW_SIMILARITY_THRESHOLD 0.05f

#define SLAM_FEATURE_DESC_LENGTH 128  /* 标准SIFT: 4×4网格 × 8方向 = 128维 */
#define SLAM_ORB_PATTERN_SIZE 32
#define SLAM_HARRIS_K 0.04f
#define SLAM_HARRIS_THRESHOLD 0.01f
#define SLAM_LOWES_RATIO 0.7f
#define SLAM_MAX_MATCHES 1000
#define SLAM_MIN_MATCHES_FOR_ESTIMATION 8
#define SLAM_MIN_MATCHES_FOR_EPNP 4
#define SLAM_BA_NUM_ITERATIONS 20
#define SLAM_PGO_NUM_ITERATIONS 50
#define SLAM_GN_NUM_ITERATIONS 10
#define SLAM_LM_INIT_LAMBDA 1e-3f
#define SLAM_LM_MIN_LAMBDA 1e-10f
#define SLAM_LM_MAX_LAMBDA 1e6f

/* ==================== 内部数据结构 ==================== */

typedef enum {
    FEATURE_EXTRACTOR_HARRIS = 0,
    FEATURE_EXTRACTOR_FAST = 1,
    FEATURE_EXTRACTOR_ORB = 2,
    FEATURE_EXTRACTOR_SIFT = 3,
    FEATURE_EXTRACTOR_SURF = 4,
    FEATURE_EXTRACTOR_AKAZE = 5,
} FeatureExtractorType;

typedef struct {
    int x;
    int y;
    float response;
    float orientation;
    float size;
    int octave;
    float descriptor[SLAM_FEATURE_DESC_LENGTH];
    int descriptor_length;
} FeaturePoint;

typedef struct {
    int query_idx;
    int train_idx;
    float distance;
    float ratio;
} FeatureMatch;

typedef struct {
    FeatureExtractorType type;
    int max_features;
    float threshold;
    int octave_layers;
    float scale_factor;
    int descriptor_length;
    int initialized;
    float* harris_response;
    float* fast_corner_scores;
    float* orb_brief_pattern;
    float* sift_gaussian_pyramid;
    float* surf_haar_wavelets;
    float* descriptor_buffer;
    int buffer_size;
    int total_features_extracted;
    float total_extraction_time;
} FeatureExtractor;

typedef struct {
    float last_timestamp;
    float last_gyro[3];
    float last_acc[3];
    float gyro_bias[3];
    float acc_bias[3];
    float velocity[3];
    float position[3];
    float orientation[4];
    float preintegrated_rotation[4];
    float preintegrated_velocity[3];
    float preintegrated_position[3];
    float covariance[9];
    int initialized;
    int has_data;
} ImuData;

typedef struct {
    int id;
    int frame_id;
    int64_t timestamp;
    SlamPose pose;
    SlamPose pose_gt;
    FeaturePoint* features;
    FeaturePoint* features_2d;
    int num_features;
    float* image_data;
    int image_width;
    int image_height;
    float* depth_data;
    PointCloud* point_cloud;
    float scale;
    int covisible_count;
    int* landmark_ids;
    int num_landmarks;
    int is_keyframe;
    int keyframe_id;
} VoFrame;

typedef struct {
    Landmark* landmarks;
    int num_landmarks;
    int max_landmarks;
    KeyFrame* keyframes;
    int num_keyframes;
    int max_keyframes;
    float* map_points;
    size_t num_map_points;
    float bounds_min[3];
    float bounds_max[3];
    int initialized;
} LocalMap;

typedef struct {
    int num_poses;
    int num_landmarks;
    int num_observations;
    float* poses;
    float* landmarks;
    int* observation_frames;
    int* observation_landmarks;
    float* observation_points;
    float* observation_weights;
    float* camera_params;
} OptimizationProblem;

typedef struct VocabTreeNode {
    float descriptor[SLAM_FEATURE_DESC_LENGTH];
    int descriptor_length;
    float weight;
    int is_leaf;
    int leaf_index;
    int num_children;
    struct VocabTreeNode** children;
    int num_features_assigned;
    int* feature_indices;
    int image_count;
} VocabTreeNode;

typedef struct {
    VocabTreeNode* root;
    int vocabulary_size;
    int vocabulary_depth;
    int branching_factor;
    int descriptor_length;
    int num_leaf_nodes;
    VisualWord* leaf_words;
    int total_trained_features;
    int num_trained_frames;
    int enable_tf_idf;
    int incremental_update_enabled;
    int is_built;
} InternalVocabulary;

typedef struct {
    int* adjacency_matrix;
    int* connected_frames;
    int* connected_frame_counts;
    int num_frames;
    int max_frames;
    int* essential_graph_edges_from;
    int* essential_graph_edges_to;
    float* essential_graph_weights;
    int num_essential_edges;
    int essential_graph_built;
} InternalCovisibility;

typedef struct {
    int* temporal_queue;
    int temporal_queue_size;
    int temporal_queue_count;
    int temporal_queue_head;
    int temporal_consistency_threshold;
    LoopClosureCandidate* scored_candidates;
    int max_scored_candidates;
    int num_scored_candidates;
    int* fusion_table;
    int fusion_table_size;
    int fusion_pending;
    int total_fusions_performed;
    float* drift_deltas;
    int drift_delta_capacity;
    int drift_active;
    int drift_frame_start;
    int last_detection_frame;
    int consecutive_detections;
    float detection_confidence;
    int bow_computed_for_frame;
    float* bow_vector;
    int bow_vector_size;
} InternalLoopClosure;

struct SlamSystem {
    SlamConfig config;
    SlamState state;
    int is_initialized;
    int is_lost;
    VoFrame current_frame;
    VoFrame* vo_frames;
    int vo_frame_capacity;
    int vo_frame_count;
    int current_frame_id;
    LocalMap local_map;
    OptimizationProblem optimization_problem;
    int* loop_candidates;
    int num_loop_candidates;
    int last_loop_frame_id;
    void* feature_extractor;
    PointCloudProcessor* point_cloud_processor;
    DepthEstimator* depth_estimator;
    ImuData imu_data;
    float* image_buffer;
    float* depth_buffer;
    float* point_cloud_buffer;
    size_t buffer_capacity;
    LoopClosureConfig loop_closure_config;
    InternalVocabulary vocabulary;
    InternalCovisibility covisibility;
    InternalLoopClosure loop_closure_internal;
    float total_tracking_time;
    float total_mapping_time;
    float total_optimization_time;
    int frames_processed;
    SlamPose* trajectory;
    int trajectory_capacity;
    int trajectory_count;
};

/* ==================== 内存安全宏 ==================== */

#define slam_malloc(size)          safe_malloc(size)
#define slam_calloc(num, size)     safe_calloc(num, size)
#define slam_realloc(ptr, size)    safe_realloc(ptr, size)
#define slam_free(ptr)             safe_free((void**)&(ptr))

/* ==================== 数学辅助函数 ==================== */

float slam_norm_squared(const float* v, int n);
void slam_cross_product(const float* a, const float* b, float* result);
void slam_rodrigues_rotation(const float* axis, float angle, float* R);
void slam_quaternion_to_rotation_matrix(const float* q, float* R);
void slam_rotation_matrix_to_quaternion(const float* R, float* q);
void slam_project_point(const float* point3d, const float* R, const float* t,
                        const float* camera_params, float* u, float* v);
void slam_backproject_point(float u, float v, float depth, const float* camera_params,
                            float* point3d);
float slam_reprojection_error(const float* point3d, const float* R, const float* t,
                              const float* camera_params, float u_obs, float v_obs);
void slam_compute_relative_pose(const SlamPose* pose1, const SlamPose* pose2, float* delta);
void slam_apply_delta_to_pose(SlamPose* pose, const float* delta);
int slam_svd_3x3(const float* A, float* U, float* S, float* V);
int slam_solve_linear_system(float* A, float* x, int n);

/* ==================== 前端函数（特征提取/匹配/运动估计/三角化） ==================== */

int slam_initialize_vo(SlamSystem* system, const FeaturePoint* features,
                       int num_features, int64_t timestamp);
int slam_initialize_map(SlamSystem* system, const FeaturePoint* features1,
                        const FeaturePoint* features2,
                        const FeatureMatch* matches, int num_matches,
                        const float* R, const float* t,
                        const float* camera_params,
                        int64_t timestamp1, int64_t timestamp2);
int slam_extract_features(SlamSystem* system, const float* image_data,
                          int width, int height, FeaturePoint** features_out,
                          int* num_features_out);
int slam_match_features(const FeaturePoint* features1, int num_features1,
                        const FeaturePoint* features2, int num_features2,
                        FeatureMatch** matches_out, int* num_matches_out);
int slam_estimate_motion_2d2d(const FeaturePoint* features1,
                              const FeaturePoint* features2,
                              const FeatureMatch* matches, int num_matches,
                              const float* camera_params,
                              float* R, float* t);
int slam_estimate_motion_3d2d(const FeaturePoint* features2d,
                              const float* points3d, int num_points,
                              const float* camera_params,
                              const float* initial_pose,
                              float* optimized_pose);
int slam_triangulate_points(const FeaturePoint* features1,
                            const FeaturePoint* features2,
                            const FeatureMatch* matches, int num_matches,
                            const float* R, const float* t,
                            const float* camera_params,
                            float* points3d);
int slam_triangulate_dlt(const float* P1, const float* P2,
                         const float* pt1, const float* pt2,
                         float* point3d);
int slam_add_keyframe(SlamSystem* system, const FeaturePoint* features,
                      int num_features, const float* pose, int64_t timestamp);
int slam_add_landmark(SlamSystem* system, const float* point3d,
                      const float* descriptor, int keyframe_id,
                      int feature_idx);
int slam_update_landmark_observation(SlamSystem* system, int landmark_id,
                                     int keyframe_id, int feature_idx);
int slam_compute_orb_descriptor(const float* image, int width, int height,
                                int x, int y, float* descriptor, int descriptor_length);

/* ==================== 后端函数（BA/优化） ==================== */

int slam_optimize_local_bundle(SlamSystem* system, int window_size);
int slam_perform_global_bundle_adjustment(SlamSystem* system, int max_iterations);
int slam_build_optimization_problem(SlamSystem* system, OptimizationProblem* problem);
int slam_solve_optimization_problem(OptimizationProblem* problem, int max_iterations);
int slam_update_from_optimization(SlamSystem* system, const OptimizationProblem* problem);
void slam_free_optimization_problem(OptimizationProblem* problem);

int slam_compute_analytical_jacobian(const float* pose_params,
                                     const float* landmark_params,
                                     const float* camera_params,
                                     float* jac_pose, float* jac_landmark);

/* V-016: Gauss-Newton解算器 —— BA优化的简化快速解法 */
int slam_solve_gauss_newton(OptimizationProblem* problem, int max_iterations);

/* V-016: 观测关联验证 —— 标记外点观测 */
float slam_validate_observation_association(const OptimizationProblem* problem,
                                             int* outlier_mask,
                                             float* outlier_ratio);

/* ==================== 闭环检测函数 ==================== */

int slam_detect_loop_closure(SlamSystem* system, int frame_id, int* matched_frame_id);
int slam_correct_loop_closure(SlamSystem* system, int frame_id, int matched_frame_id);
int slam_compute_bow_vector(SlamSystem* system, int frame_id);
int slam_select_candidates_by_bow(SlamSystem* system, int frame_id,
                                  int* candidates, int max_candidates, float* scores);
int slam_select_candidates_hybrid(SlamSystem* system, int frame_id,
                                  int* candidates, int max_candidates, float* scores);
int slam_verify_loop_geometric_8point(SlamSystem* system, int frame_id, int candidate_id,
                                      int* num_inliers, float* inlier_ratio,
                                      float* fundamental_matrix);
int slam_temporal_consistency_check(SlamSystem* system, int candidate_frame_id);
int slam_fuse_loop_closure_map(SlamSystem* system, int frame_id, int matched_frame_id);
int slam_propagate_drift_correction(SlamSystem* system, int matched_frame_id,
                                    int current_frame_id, const float* corrected_poses,
                                    int num_corrected);
int slam_compute_fundamental_matrix_8point(const float* points1, const float* points2,
                                           int num_points, float* F);
float slam_compute_sampson_distance(const float* F, float x1, float y1, float x2, float y2);
int slam_find_homography_inliers(const FeaturePoint* f1, const FeaturePoint* f2,
                                 const FeatureMatch* matches, int num_matches,
                                 float* H, float threshold, int* inlier_mask, int max_inliers);

/* ==================== 视觉词汇表函数 ==================== */

int slam_vocabulary_init(InternalVocabulary* vocab, const VisualVocabularyConfig* config);
void slam_vocabulary_free(InternalVocabulary* vocab);
int slam_vocabulary_build(InternalVocabulary* vocab, const float* all_descriptors,
                          int num_descriptors, int descriptor_length);
int slam_vocabulary_compute_bow(InternalVocabulary* vocab, const float* descriptors,
                                int num_descriptors, int descriptor_length,
                                float* bow_vector, int* bow_vector_size);
float slam_vocabulary_compute_similarity(const float* bow1, int size1,
                                         const float* bow2, int size2);
int slam_vocabulary_add_frame(InternalVocabulary* vocab, const float* descriptors,
                              int num_descriptors, int descriptor_length);
int slam_vocabulary_update_tfidf(InternalVocabulary* vocab);
VocabTreeNode* slam_vocab_node_create(int descriptor_length);
void slam_vocab_node_free(VocabTreeNode* node);
int slam_vocab_node_train(VocabTreeNode* node, const float* descriptors,
                          int num_descriptors, int descriptor_length,
                          int depth, int max_depth, int branch_factor);
int slam_vocab_node_assign(VocabTreeNode* node, const float* descriptor,
                           int descriptor_length);

/* ==================== 共视图函数 ==================== */

int slam_covisibility_init(InternalCovisibility* cov, int max_frames);
void slam_covisibility_free(InternalCovisibility* cov);
int slam_covisibility_update(InternalCovisibility* cov, int frame_id,
                             int* landmark_ids, int num_landmarks,
                             const KeyFrame* keyframes, int num_keyframes);
int slam_covisibility_get_connected(InternalCovisibility* cov, int frame_id,
                                    int* connected_ids, int max_count);
int slam_covisibility_get_weight(InternalCovisibility* cov, int frame_id1, int frame_id2);
int slam_covisibility_build_essential_graph(InternalCovisibility* cov,
                                            const KeyFrame* keyframes,
                                            int num_keyframes);

/* ==================== I/O函数 ==================== */

int slam_save_trajectory_tum(const SlamSystem* system, const char* filepath);
int slam_save_map_binary(SlamSystem* system, const char* filepath);
int slam_load_map_binary(SlamSystem* system, const char* filepath);
int slam_read_ppm_token(FILE* f, char* token, int max_len);
int slam_kmeans_plus_plus(const float* descriptors, int num_descriptors,
                          int descriptor_length, int k, float* centers);
void slam_free_frame_reader(struct SlamFrameReader* reader);
void slam_free_raw_video_reader(struct SlamRawVideoReader* reader);

/* ==================== 内部辅助函数 ==================== */

void slam_system_reset_internal(SlamSystem* system);
int slam_has_sufficient_motion(const FeatureMatch* matches, int num_matches,
                               const FeaturePoint* features1,
                               const FeaturePoint* features2);

/* ==================== P1-06: 位姿图优化（新增） ==================== */

/**
 * @brief 位姿图节点（每个关键帧）
 */
typedef struct {
    float tx, ty, tz;
    float qx, qy, qz, qw;
    int fixed;
} PoseGraphNode;

/**
 * @brief 位姿图约束（边）
 */
typedef struct {
    int from_id;
    int to_id;
    float relative_tx, relative_ty, relative_tz;
    float relative_qx, relative_qy, relative_qz, relative_qw;
    float weight;
    int is_loop_closure;
} PoseGraphConstraint;

/**
 * @brief Levenberg-Marquardt位姿图优化
 * @param nodes 位姿图节点数组
 * @param num_nodes 节点数量
 * @param constraints 约束数组
 * @param num_constraints 约束数量
 * @param max_iterations 最大迭代次数
 * @param lambda_init 初始阻尼因子
 * @return 总残差平方和，负值表示失败
 */
float slam_pose_graph_optimize(PoseGraphNode* nodes, int num_nodes,
                                const PoseGraphConstraint* constraints, int num_constraints,
                                int max_iterations, float lambda_init);

/**
 * @brief 从SlamSystem关键帧构建位姿图并优化，结果写回关键帧
 * @param system SLAM系统句柄
 * @return 收敛后的总残差，负值表示失败
 */
float slam_optimize_pose_graph_from_keyframes(SlamSystem* system);

/* ============================================================================
 * DUP-003: 以下声明合并自 slam_enhance.h — SLAM高级增强功能
 * ============================================================================ */

/**
 * @brief 束调整优化（Bundle Adjustment, Schur补+LM）
 *
 * 最小化重投影误差，联合优化相机参数和3D点位置。
 *
 * @param camera_params  输入相机参数 [num_cameras × 15]
 *                       每相机：fx, fy, cx, cy, R(9), t(3)
 * @param num_cameras    相机数量
 * @param points_3d      输入3D点 [num_points × 3]
 * @param num_points     3D点数量
 * @param obs_camera_ids 观测相机ID [num_observations]
 * @param obs_point_ids  观测点ID [num_observations]
 * @param obs_uv         观测像素坐标 [num_observations × 2]
 * @param num_observations 观测数量
 * @param optimized_cameras 输出优化相机参数 [num_cameras × 15]
 * @param optimized_points  输出优化3D点 [num_points × 3]
 * @return int 成功0，失败-1
 */
SELFLNN_API int slam_enhance_bundle_adjust(
    const float* camera_params, int num_cameras,
    const float* points_3d, int num_points,
    const int* obs_camera_ids, const int* obs_point_ids,
    const float* obs_uv, int num_observations,
    float* optimized_cameras, float* optimized_points);

/**
 * @brief IMU预积分
 *
 * 对一段时间内的IMU测量进行预积分，得到帧间的相对运动估计。
 *
 * @param acc_data          加速度数据 [num_measurements × 3] (m/s²)
 * @param gyro_data         陀螺仪数据 [num_measurements × 3] (rad/s)
 * @param dt_data           时间间隔 [num_measurements] (秒)
 * @param num_measurements  测量次数
 * @param preint_delta_p    输出位置变化 [3] (m)
 * @param preint_delta_v    输出速度变化 [3] (m/s)
 * @param preint_delta_q    输出旋转四元数 [4] (w,x,y,z)
 * @return int 成功0，失败-1
 */
SELFLNN_API int slam_enhance_imu_preintegrate(
    const float* acc_data, const float* gyro_data, const float* dt_data,
    int num_measurements,
    float* preint_delta_p, float* preint_delta_v, float* preint_delta_q);

/**
 * @brief 全局位姿图优化（PGO，非线性最小二乘）
 *
 * 优化回环约束下的全局位姿图。此API使用原始浮点数组作为输入，
 * 与 slam_pose_graph_optimize（使用 PoseGraphNode 结构体）互补。
 *
 * @param num_nodes         节点数量
 * @param node_positions    节点位置 [num_nodes × 3]，输入且原地优化
 * @param num_edges         边数量
 * @param edge_from         边起点节点ID [num_edges]
 * @param edge_to           边终点节点ID [num_edges]
 * @param edge_relative_p   边相对位置约束 [num_edges × 3]
 * @param edge_weight       边权重 [num_edges]（NULL则全部为1.0）
 * @param fixed_nodes       固定节点ID数组 [num_fixed]（NULL则无固定节点）
 * @param num_fixed         固定节点数量
 * @return int 成功0，失败-1
 */
SELFLNN_API int slam_enhance_pose_graph_optimize(
    int num_nodes, float* node_positions,
    int num_edges,
    const int* edge_from, const int* edge_to,
    const float* edge_relative_p, const float* edge_weight,
    const int* fixed_nodes, int num_fixed);

/* ================================================================
 *: 离线数据集回放与轨迹评估API
 * 实现于 slam_io.c, 使SLAM系统可在无硬件时通过真实记录的数据集验证
 * ================================================================ */

/** @brief 加载TUM RGB-D数据集
 * @param directory 数据集目录路径
 * @param num_frames 输出: 总帧数
 * @param width 输出: 图像宽度
 * @param height 输出: 图像高度
 * @return 0=成功, -1=失败 */
int slam_load_dataset_tum(const char* directory, int* num_frames, int* width, int* height);

/** @brief 计算绝对轨迹误差(ATE)
 * @param estimated 估计位姿数组
 * @param est_count 估计位姿数
 * @param ground_truth 真值位姿数组
 * @param gt_count 真值位姿数
 * @param rmse_out 输出: RMSE (米)
 * @param mean_out 输出: 平均误差 (米)
 * @param max_out 输出: 最大误差 (米)
 * @return 0=成功, -1=失败 */
int slam_evaluate_ate(const SlamPose* estimated, int est_count,
                       const SlamPose* ground_truth, int gt_count,
                       float* rmse_out, float* mean_out, float* max_out);

/** @brief 计算相对位姿误差(RPE)
 * @param estimated 估计位姿数组
 * @param est_count 估计位姿数
 * @param ground_truth 真值位姿数组
 * @param gt_count 真值位姿数
 * @param delta_frames 帧间隔
 * @param rmse_trans_out 输出: 平移RMSE (米)
 * @param rmse_rot_out 输出: 旋转RMSE (弧度)
 * @return 0=成功, -1=失败 */
int slam_evaluate_rpe(const SlamPose* estimated, int est_count,
                       const SlamPose* ground_truth, int gt_count,
                       int delta_frames,
                       float* rmse_trans_out, float* rmse_rot_out);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SLAM_INTERNAL_H */
